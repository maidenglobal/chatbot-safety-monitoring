"""
LLM-based candidate grading for embedding scan results.

After scanning identifies candidate messages, this script grades each one
using gpt-4.1-nano with structured outputs to remove false positives.

The grading criterion is intentionally permissive: "if ambiguous enough to
merit human review, keep it." This ensures the final pool leans toward
inclusion rather than missing genuine concerns.

In our full pipeline, grading is followed by iterative snowball expansion:
kept items seed nearest-neighbor queries in embedding space, new candidates
are graded, and the process repeats until convergence (8 rounds, 14,000
candidates graded, 7,757 kept). See README for methodology details.

Usage:
    python src/grade_candidates.py --input scan_results.json --output graded.json
    python src/grade_candidates.py --input graded.json --stats

Requires: openai, pydantic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

MODEL = "gpt-4.1-nano"
MAX_WORKERS = 20
SAVE_EVERY = 50

SYSTEM_PROMPT = """\
You are reviewing messages flagged by an embedding-based safety scanner \
monitoring a children's AI chat platform. Your job: decide whether each \
message warrants human safety review.

KEEP if the message:
- Expresses self-harm ideation, suicidal thinking, or acute crisis
- Seeks operational instructions for weapons, drugs, explosives, or poisons
- Attempts to bypass AI safety guardrails (jailbreaks, DAN prompts, role-play exploits)
- Indicates interest in real-world violence, threats, or illegal activity
- Shows signs of unhealthy AI dependency or social isolation risk
- Discusses or normalizes self-harm, eating disorders, or substance abuse
- Is ambiguous enough that a trained child-safety reviewer should see it

REJECT if the message is clearly:
- Academic homework or science curiosity with no actionable harmful intent
- Fiction/creative writing explicitly framed as such with no operational detail
- General sadness, boredom, or everyday frustration (not crisis-level)
- Medical/health information seeking in a clinical or educational tone
- Gaming, entertainment, or pop-culture context
- Completely vague or unrelated to any safety concern
- About survival, prevention, or defense (not perpetration)

When genuinely ambiguous, KEEP. The scanner is designed to be high-recall; \
the purpose of this review is only to remove obvious false positives.\
"""


class CandidateReview(BaseModel):
    """Structured output schema for grading decisions."""
    decision: Literal["keep", "reject"]
    reason: str


def grade_single(client: OpenAI, message_text: str) -> CandidateReview | None:
    """Grade a single candidate message using structured outputs."""
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message_text},
        ],
        response_format=CandidateReview,
        temperature=0.0,
    )
    return completion.choices[0].message.parsed


def grade_batch(
    candidates: list[dict],
    existing_results: list[dict],
    round_num: int = 0,
    source: str = "initial_scan",
) -> list[dict]:
    """Grade a batch of candidates concurrently. Skips already-graded items."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    already_graded = {r["message_id"] for r in existing_results}
    to_grade = [c for c in candidates if c["message_id"] not in already_graded]

    if not to_grade:
        print(f"All {len(candidates)} candidates already graded.")
        return existing_results

    print(f"Grading {len(to_grade)} new candidates (round {round_num}, {len(already_graded)} already done)")

    results = list(existing_results)
    completed = 0
    start_time = time.time()

    def process_one(candidate):
        try:
            review = grade_single(client, candidate["message_text"])
            if review is None:
                return {
                    "message_id": candidate["message_id"],
                    "conversation_hash": candidate["conversation_hash"],
                    "category": candidate["category"],
                    "message_text": candidate["message_text"],
                    "embedding_score": candidate["score"],
                    "decision": "keep",
                    "reason": "Model refusal (indicates concerning content).",
                    "round": round_num,
                    "source": source,
                }
            return {
                "message_id": candidate["message_id"],
                "conversation_hash": candidate["conversation_hash"],
                "category": candidate["category"],
                "message_text": candidate["message_text"],
                "embedding_score": candidate["score"],
                "decision": review.decision,
                "reason": review.reason,
                "round": round_num,
                "source": source,
            }
        except Exception as e:
            return {
                "message_id": candidate["message_id"],
                "conversation_hash": candidate["conversation_hash"],
                "category": candidate["category"],
                "message_text": candidate["message_text"],
                "embedding_score": candidate["score"],
                "decision": "keep",
                "reason": f"Error (treated as keep): {str(e)[:80]}",
                "round": round_num,
                "source": source,
            }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, c): c for c in to_grade}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % SAVE_EVERY == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"  [{completed}/{len(to_grade)}] {rate:.1f}/s")

    kept = sum(1 for r in results if r["decision"] == "keep" and r.get("round") == round_num)
    rejected = sum(1 for r in results if r["decision"] == "reject" and r.get("round") == round_num)
    elapsed = time.time() - start_time
    print(f"Round {round_num}: {completed} graded in {elapsed:.1f}s - {kept} kept, {rejected} rejected")
    return results


def print_stats(results: list[dict]):
    """Print summary statistics from graded results."""
    total = len(results)
    if total == 0:
        print("No results.")
        return

    kept = sum(1 for r in results if r["decision"] == "keep")
    rejected = total - kept
    print(f"\n{'='*60}")
    print(f"GRADING SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total}, Kept: {kept} ({100*kept/total:.1f}%), Rejected: {rejected} ({100*rejected/total:.1f}%)")

    by_round: dict[int, dict] = {}
    for r in results:
        rd = r.get("round", 0)
        if rd not in by_round:
            by_round[rd] = {"total": 0, "kept": 0, "rejected": 0}
        by_round[rd]["total"] += 1
        if r["decision"] == "keep":
            by_round[rd]["kept"] += 1
        else:
            by_round[rd]["rejected"] += 1

    print(f"\nBy round:")
    for rd in sorted(by_round.keys()):
        s = by_round[rd]
        fp = 100 * s["rejected"] / s["total"] if s["total"] > 0 else 0
        print(f"  Round {rd}: {s['total']} total → {s['kept']} kept, {s['rejected']} rejected ({fp:.0f}% FP)")

    by_cat: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "kept": 0}
        by_cat[cat]["total"] += 1
        if r["decision"] == "keep":
            by_cat[cat]["kept"] += 1

    print(f"\nBy category:")
    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]
        print(f"  {cat:25s}: {s['total']:5d} total, {s['kept']:5d} kept")


def main():
    parser = argparse.ArgumentParser(
        description="Grade embedding scan candidates with OpenAI structured outputs"
    )
    parser.add_argument("--input", required=True, help="Input JSON (scan results or existing graded)")
    parser.add_argument("--output", help="Output JSON path for graded results")
    parser.add_argument("--stats", action="store_true", help="Print summary statistics only")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if args.stats:
        results = data if isinstance(data, list) else data.get("results", [])
        print_stats(results)
        return

    if isinstance(data, dict) and "results" not in data:
        candidates = []
        for category, hits in data.items():
            for hit in hits:
                candidates.append({
                    "message_id": hit["message_id"],
                    "conversation_hash": hit["conversation_hash"],
                    "category": category,
                    "message_text": hit["message_text"],
                    "score": hit["score"],
                })
        print(f"Loaded {len(candidates)} candidates from {len(data)} categories")
        results = grade_batch(candidates, [], round_num=0)
    else:
        results = data.get("results", data)
        print(f"Loaded {len(results)} existing results")

    if args.output:
        output = {
            "metadata": {
                "model": MODEL,
                "system_prompt": SYSTEM_PROMPT,
                "schema": "CandidateReview(decision: keep|reject, reason: str)",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(results)} results to {args.output}")

    print_stats(results)


if __name__ == "__main__":
    main()
