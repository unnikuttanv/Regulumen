"""Validate that every gold expected_citation is grounded in the real corpus.

Loads ``eval/gold/gold_qa.jsonl`` via the harness loader and, for each
non-abstain item, asserts that EACH ``expected_citation`` matches at least one
real chunk ``citation_label`` from ``eval/gold/_chunk_inventory.jsonl`` using
the harness's own matcher ``citation_hit``. This is the grounding guarantee
(retrievability is checked later, once embeddings exist).

Usage::

    python scripts/validate_gold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from regulumen.core.eval.metrics import citation_hit
from regulumen.core.eval.models import load_gold

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "eval" / "gold" / "gold_qa.jsonl"
INVENTORY_PATH = ROOT / "eval" / "gold" / "_chunk_inventory.jsonl"


def _inventory_labels() -> list[str]:
    return [
        json.loads(line)["citation_label"]
        for line in INVENTORY_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    labels = _inventory_labels()
    items = load_gold(GOLD_PATH)

    total = 0
    grounded = 0
    failures: list[str] = []
    n_abstain = 0

    for item in items:
        if item.must_abstain:
            n_abstain += 1
            continue
        for expected in item.expected_citations:
            total += 1
            if citation_hit(expected, labels):
                grounded += 1
            else:
                failures.append(f"{item.id}: {expected!r} NOT found in inventory")

    print(f"Gold items: {len(items)} ({n_abstain} abstain, "
          f"{len(items) - n_abstain} retrieval)")
    print(f"Expected citations grounded: {grounded}/{total}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("All expected citations are grounded in the real chunk inventory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
