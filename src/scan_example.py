"""
Example: semantic similarity scanning with generated prototypes.

This demonstrates the core approach:
1. Generate search prototypes from the taxonomy
2. Embed prototypes with an embedding model
3. For each user message, compute max cosine similarity to any prototype
4. Flag messages above the category threshold

This is a minimal reference implementation. A production system would use
a vector database (e.g., ChromaDB, Pinecone) rather than brute-force search.

Usage:
    python src/scan_example.py --messages sample_messages.jsonl

Requires: openai, numpy, pyyaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_prototypes import generate_prototypes, load_taxonomy

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_texts(client: OpenAI, texts: list[str], model: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed a batch of texts. Returns (N, D) array of normalized embeddings."""
    BATCH_SIZE = 2048
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([e.embedding for e in response.data])
    return np.array(all_embeddings, dtype=np.float32)


def scan_messages(
    messages: list[str],
    prototype_embeddings: dict[str, np.ndarray],
    thresholds: dict[str, float],
    client: OpenAI,
) -> list[dict]:
    """
    Score each message against all category prototypes.
    Returns flagged messages with their category and score.
    """
    if not messages:
        return []

    msg_embeddings = embed_texts(client, messages)

    flagged = []
    for i, (msg, msg_emb) in enumerate(zip(messages, msg_embeddings)):
        msg_emb_norm = msg_emb / np.linalg.norm(msg_emb)

        for cat_key, proto_embs in prototype_embeddings.items():
            similarities = proto_embs @ msg_emb_norm
            max_sim = float(np.max(similarities))

            if max_sim >= thresholds[cat_key]:
                flagged.append({
                    "message_index": i,
                    "message_text": msg[:200],
                    "category": cat_key,
                    "score": round(max_sim, 4),
                    "threshold": thresholds[cat_key],
                })

    return flagged


def main():
    parser = argparse.ArgumentParser(description="Scan messages against taxonomy prototypes")
    parser.add_argument("--messages", required=True, help="JSONL file with 'text' field per line")
    parser.add_argument("--taxonomy", default=None, help="Path to taxonomy YAML")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable required", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    taxonomy = load_taxonomy(args.taxonomy)
    prototypes = generate_prototypes(taxonomy)

    print("Embedding prototypes...")
    prototype_embeddings: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for cat_key, data in prototypes.items():
        texts = [p["text"] for p in data["prototypes"]]
        if not texts:
            continue
        embs = embed_texts(client, texts)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        prototype_embeddings[cat_key] = embs / norms
        thresholds[cat_key] = data["threshold"]
        print(f"  {data['name']}: {len(texts)} prototypes embedded")

    print(f"\nLoading messages from {args.messages}...")
    messages = []
    with open(args.messages) as f:
        for line in f:
            obj = json.loads(line)
            messages.append(obj["text"])
    print(f"  {len(messages)} messages loaded")

    print("\nScanning...")
    flagged = scan_messages(messages, prototype_embeddings, thresholds, client)

    print(f"\n{'='*60}")
    print(f"Results: {len(flagged)} flags across {len(messages)} messages")
    print(f"{'='*60}")
    for hit in sorted(flagged, key=lambda x: x["score"], reverse=True):
        print(f"  [{hit['category']:25s}] score={hit['score']:.4f}  \"{hit['message_text'][:80]}...\"")


if __name__ == "__main__":
    main()
