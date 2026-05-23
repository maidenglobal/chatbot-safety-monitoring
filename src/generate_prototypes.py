"""
Generate search prototypes from the behavior × domain taxonomy.

This script generates the full set of embedding search prototypes used for
semantic similarity scanning. Prototypes are created combinatorially:
each behavior template is filled with each applicable domain term.

Usage:
    python src/generate_prototypes.py [--output prototypes.json]

Requires: pyyaml
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import yaml

TAXONOMY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "taxonomy.yaml")


def load_taxonomy(path: str | None = None) -> dict[str, Any]:
    with open(path or TAXONOMY_PATH) as f:
        return yaml.safe_load(f)


def generate_prototypes(taxonomy: dict[str, Any]) -> dict[str, dict]:
    """
    Generate all prototypes from the taxonomy.

    Returns a dictionary keyed by category, each containing:
        - name: human-readable category name
        - severity: critical | high
        - threshold: cosine similarity threshold used for flagging
        - prototypes: list of {"text": str, "behavior": str, "domain": str}
    """
    behaviors = taxonomy["behaviors"]
    domains = taxonomy["domains"]
    cat_settings = taxonomy["category_settings"]

    generated: dict[str, list[dict]] = {k: [] for k in cat_settings}

    for domain_key, domain in domains.items():
        cat_key = domain["category"]
        excludes = set(domain.get("exclude_behaviors", []))

        for behavior_key, behavior in behaviors.items():
            if behavior_key in excludes:
                continue

            for template in behavior["templates"]:
                for term in domain["terms"]:
                    proto_text = template.replace("{domain_term}", term)
                    generated[cat_key].append({
                        "text": proto_text,
                        "behavior": behavior_key,
                        "domain": domain_key,
                    })

    result: dict[str, dict] = {}

    for cat_key, settings in cat_settings.items():
        result[cat_key] = {
            "name": settings["name"],
            "severity": settings["severity"],
            "threshold": settings["threshold"],
            "prototypes": generated.get(cat_key, []),
        }

    for cat_key, cat_data in taxonomy.get("standalone_categories", {}).items():
        result[cat_key] = {
            "name": cat_data["name"],
            "severity": cat_data["severity"],
            "threshold": cat_data["threshold"],
            "prototypes": [
                {"text": p, "behavior": "standalone", "domain": "standalone"}
                for p in cat_data["prototypes"]
            ],
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate search prototypes from taxonomy")
    parser.add_argument("--taxonomy", default=TAXONOMY_PATH, help="Path to taxonomy YAML")
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    prototypes = generate_prototypes(taxonomy)

    summary = {
        cat_key: {
            "name": data["name"],
            "severity": data["severity"],
            "threshold": data["threshold"],
            "count": len(data["prototypes"]),
        }
        for cat_key, data in prototypes.items()
    }

    print("Generated prototypes by category:")
    for cat_key, info in summary.items():
        print(f"  {info['name']:45s} {info['count']:>4d} prototypes  (threshold={info['threshold']})")
    print(f"\n  TOTAL: {sum(s['count'] for s in summary.values())} prototypes")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(prototypes, f, indent=2)
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
