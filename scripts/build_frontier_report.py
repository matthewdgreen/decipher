#!/usr/bin/env python3
"""Render a compact frontier-suite report from summary rows."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a report from frontier-suite summary rows.",
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Frontier suite summary CSV or JSONL.",
    )
    args = parser.parse_args()

    rows = _load_rows(Path(args.summary))
    print("# Frontier Report")
    print()
    print("## Pass/Fail By Class")
    print()
    print("| Class | Solver | Rows | Passing | Failing |")
    print("|---|---|---:|---:|---:|")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["frontier_class"]), str(row["solver"]))].append(row)
    for (frontier_class, solver), items in sorted(grouped.items()):
        passed = sum(1 for item in items if item.get("meets_expectations"))
        print(f"| {frontier_class} | {solver} | {len(items)} | {passed} | {len(items) - passed} |")

    print()
    print("## Largest Accuracy Regressions")
    print()
    for row in _top_failures(rows, key="char"):
        print(_format_failure(row))

    print()
    print("## Worst Runtime Regressions")
    print()
    for row in _top_failures(rows, key="time"):
        print(_format_failure(row))

    print()
    print("## Unsupported Wrapper Cases")
    print()
    unsupported = [
        row for row in rows
        if "wrapper_unsupported" in _tags(row)
        or "alphabet overflow" in str(row.get("error") or "").lower()
    ]
    if not unsupported:
        print("_None._")
    for row in unsupported:
        print(_format_failure(row))

    print()
    print("## Frontier Moved")
    print()
    moved = [
        row for row in rows
        if row.get("solver") == "decipher-automated"
        and row.get("frontier_class") in {"bad_result", "slow_result", "shared_hard"}
        and row.get("meets_expectations")
    ]
    if not moved:
        print("_No frontier cases have crossed their target thresholds yet._")
    for row in moved:
        print(
            f"- `{row['test_id']}` moved past `{row['frontier_class']}` thresholds "
            f"(char={float(row['char_accuracy']):.1%}, time={float(row['elapsed_seconds']):.1f}s)"
        )


def _load_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix == ".csv":
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    out: list[dict[str, object]] = []
    for row in rows:
        parsed = dict(row)
        parsed["char_accuracy"] = float(parsed.get("char_accuracy") or 0.0)
        parsed["elapsed_seconds"] = float(parsed.get("elapsed_seconds") or 0.0)
        parsed["meets_expectations"] = str(parsed.get("meets_expectations")).lower() in {"true", "1"}
        out.append(parsed)
    return out


def _top_failures(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    failures = [row for row in rows if not row.get("meets_expectations")]
    if key == "char":
        failures.sort(key=lambda row: (float(row.get("char_accuracy") or 0.0), float(row.get("elapsed_seconds") or 0.0)))
    else:
        failures.sort(key=lambda row: float(row.get("elapsed_seconds") or 0.0), reverse=True)
    return failures[:10]


def _format_failure(row: dict[str, object]) -> str:
    failures = row.get("expectation_failures") or ""
    if isinstance(failures, list):
        detail = "; ".join(str(item) for item in failures)
    else:
        detail = str(failures)
    return (
        f"- `{row['test_id']}` `{row['solver']}` "
        f"(class={row['frontier_class']}, char={float(row['char_accuracy']):.1%}, "
        f"time={float(row['elapsed_seconds']):.1f}s) — {detail or row.get('error', '')}"
    )


def _tags(row: dict[str, object]) -> set[str]:
    value = row.get("frontier_tags") or []
    if isinstance(value, list):
        return {str(item) for item in value}
    if isinstance(value, str):
        return {item.strip() for item in value.strip("[]").replace("'", "").split(",") if item.strip()}
    return set()


if __name__ == "__main__":
    main()
