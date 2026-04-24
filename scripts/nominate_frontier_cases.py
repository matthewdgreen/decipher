#!/usr/bin/env python3
"""Nominate frontier-suite candidates from parity outputs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from frontier.suite import load_parity_rows, nominate_frontier_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nominate frontier-suite candidates from parity summaries and artifacts.",
    )
    parser.add_argument("--summary", action="append", default=[], help="Parity summary CSV/JSONL.")
    parser.add_argument("--artifact-dir", action="append", default=[], help="Parity artifact directory.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()

    rows = load_parity_rows(summary_paths=args.summary, artifact_dirs=args.artifact_dir)
    nominations = nominate_frontier_candidates(rows)
    if args.format == "json":
        print(json.dumps(nominations, indent=2, ensure_ascii=False))
        return

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for nomination in nominations:
        grouped[str(nomination["frontier_class"])].append(nomination)

    for frontier_class in ("known_good", "shared_hard", "bad_result", "slow_result"):
        items = grouped.get(frontier_class, [])
        print(f"## {frontier_class}")
        if not items:
            print()
            print("_No nominations._")
            print()
            continue
        for item in items:
            tags = ", ".join(item["frontier_tags"]) or "none"
            print(f"- `{item['test_id']}` [{item['family']}]")
            print(f"  - tags: {tags}")
            print(f"  - reason: {item['reason']}")
            print(
                f"  - decipher: char={float(item['decipher_char_accuracy']):.1%} "
                f"time={float(item['decipher_elapsed_seconds']):.1f}s"
            )
            if item.get("zenith_char_accuracy") is not None:
                print(f"  - zenith: char={float(item['zenith_char_accuracy']):.1%}")
            if item.get("zkdecrypto_char_accuracy") is not None:
                print(f"  - zkdecrypto-lite: char={float(item['zkdecrypto_char_accuracy']):.1%}")
        print()


if __name__ == "__main__":
    main()
