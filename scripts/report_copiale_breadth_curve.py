#!/usr/bin/env python3
"""Estimate marginal value of Copiale null-mask breadth from one broad run.

This is an offline calibration report. It uses benchmark ground truth only to
label already-produced finalists. It does not rerun solvers. Newer artifacts
record explicit null-mask candidate IDs and evaluated indices; older compact
artifacts are matched with a best-effort row signature.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.scorer import score_decryption  # noqa: E402


DEFAULT_PREFIXES = (25, 50, 100, 150, 200, 300, 500, 800, 1200)


@dataclass(frozen=True)
class Finalist:
    test_id: str
    artifact: str
    run_label: str
    candidate_id: str
    mask: tuple[str, ...]
    source: str
    evaluated_index: int | None
    char_accuracy: float
    language_quality_rank_score: float | None
    validation_score_v2: float | None
    ensemble_score_v1: float | None
    selection_score: float | None
    decryption: str

    @property
    def mask_label(self) -> str:
        return ",".join(self.mask) if self.mask else "(none)"

    @property
    def preview(self) -> str:
        return self.decryption[:140]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report best generated Copiale finalist as breadth increases."
    )
    parser.add_argument(
        "--experiment-dir",
        default="artifacts/copiale_breadth_experiment/p084_slightly_wider",
        help="Breadth experiment directory containing ranker artifacts.",
    )
    parser.add_argument("--test-id", action="append", default=[])
    parser.add_argument(
        "--prefix",
        type=int,
        action="append",
        default=[],
        help="Candidate-count prefix to report. May be repeated.",
    )
    parser.add_argument("--output", default="", help="Markdown report path.")
    parser.add_argument("--json-output", default="", help="JSON report path.")
    parser.add_argument(
        "--max-finalists-per-artifact",
        type=int,
        default=0,
        help="Optional cap on selected+top_finalists scored from each artifact.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print artifact-level progress while scoring finalists.",
    )
    args = parser.parse_args()

    experiment_dir = resolve_path(Path(args.experiment_dir))
    finalists = load_finalists(
        experiment_dir,
        allowed_tests=set(args.test_id),
        max_finalists_per_artifact=args.max_finalists_per_artifact,
        progress=args.progress,
    )
    if not finalists:
        raise SystemExit(f"No finalists found under {experiment_dir}")
    prefixes = tuple(sorted(set(args.prefix or DEFAULT_PREFIXES)))
    payload = analyze(finalists, prefixes=prefixes)
    markdown = render_markdown(payload)
    output = resolve_path(Path(args.output)) if args.output else experiment_dir / "breadth_curve.md"
    json_output = (
        resolve_path(Path(args.json_output))
        if args.json_output
        else experiment_dir / "breadth_curve.json"
    )
    output.write_text(markdown, encoding="utf-8")
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")


def load_finalists(
    root: Path,
    *,
    allowed_tests: set[str],
    max_finalists_per_artifact: int = 0,
    progress: bool = False,
) -> list[Finalist]:
    finalists: list[Finalist] = []
    seen: set[str] = set()
    artifacts = sorted(root.glob("*/automated_only/*/*.json"))
    for artifact_index, artifact in enumerate(artifacts, start=1):
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        test_id = str(payload.get("test_id") or artifact.parent.name)
        if allowed_tests and test_id not in allowed_tests:
            continue
        ground_truth = str(payload.get("ground_truth") or "")
        null_step = next(
            (
                step for step in payload.get("steps") or []
                if isinstance(step, dict) and step.get("name") == "search_null_masks"
            ),
            None,
        )
        if not isinstance(null_step, dict):
            continue
        evaluated_rows = list(null_step.get("evaluated_rows") or [])
        evaluated_index = build_evaluated_index(evaluated_rows)
        evaluated_by_id = build_evaluated_id_index(evaluated_rows)
        run_label = str(null_step.get("ranker") or artifact.parent.parent.parent.name)
        rows = []
        selected = null_step.get("selected")
        if isinstance(selected, dict):
            rows.append(selected)
        rows.extend(row for row in (null_step.get("top_finalists") or []) if isinstance(row, dict))
        if max_finalists_per_artifact > 0:
            rows = rows[:max_finalists_per_artifact]
        if progress:
            print(
                f"[{artifact_index}/{len(artifacts)}] {test_id} "
                f"{run_label}: scoring {len(rows)} finalists",
                flush=True,
            )
        for row in rows:
            text = str(row.get("decryption") or row.get("plaintext") or "")
            if not text:
                continue
            digest = hashlib.sha256(
                (str(artifact) + "\0" + text).encode("utf-8")
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            scored = score_decryption(
                test_id,
                text,
                ground_truth,
                agent_score=0.0,
                status="completed",
            )
            candidate_id = str(row.get("candidate_id") or "")
            finalists.append(
                Finalist(
                    test_id=test_id,
                    artifact=str(artifact),
                    run_label=run_label,
                    candidate_id=candidate_id,
                    mask=tuple(str(symbol) for symbol in (row.get("mask") or [])),
                    source=str(row.get("source") or ""),
                    evaluated_index=lookup_evaluated_index(row, evaluated_by_id, evaluated_index),
                    char_accuracy=float(scored.char_accuracy),
                    language_quality_rank_score=float_or_none(row.get("language_quality_rank_score")),
                    validation_score_v2=float_or_none(row.get("validation_score_v2")),
                    ensemble_score_v1=float_or_none(row.get("ensemble_score_v1")),
                    selection_score=float_or_none(row.get("selection_score")),
                    decryption=text,
                )
            )
    return finalists


def build_evaluated_index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], int]:
    index: dict[tuple[Any, ...], int] = {}
    for position, row in enumerate(rows, start=1):
        key = row_signature(row)
        index.setdefault(key, position)
    return index


def build_evaluated_id_index(rows: list[dict[str, Any]]) -> dict[str, int]:
    index: dict[str, int] = {}
    for position, row in enumerate(rows, start=1):
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        parsed = int_or_none(row.get("evaluated_index"))
        index.setdefault(candidate_id, parsed if parsed is not None else position)
    return index


def lookup_evaluated_index(
    row: dict[str, Any],
    id_index: dict[str, int],
    legacy_index: dict[tuple[Any, ...], int],
) -> int | None:
    candidate_id = str(row.get("candidate_id") or "")
    if candidate_id and candidate_id in id_index:
        return id_index[candidate_id]
    parsed = int_or_none(row.get("evaluated_index"))
    if parsed is not None:
        return parsed
    return legacy_index.get(row_signature(row))


def row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(str(symbol) for symbol in (row.get("mask") or [])),
        str(row.get("source") or ""),
        rounded(float_or_none(row.get("validation_score_v2"))),
        rounded(float_or_none(row.get("ensemble_score_v1"))),
        rounded(float_or_none(row.get("language_quality_rank_score"))),
        str(row.get("preview") or "")[:80],
    )


def analyze(finalists: list[Finalist], *, prefixes: tuple[int, ...]) -> dict[str, Any]:
    tests: dict[str, list[Finalist]] = {}
    for row in finalists:
        tests.setdefault(row.test_id, []).append(row)
    reports = []
    for test_id, rows in sorted(tests.items()):
        known_rows = [row for row in rows if row.evaluated_index is not None]
        best_overall = max(rows, key=lambda row: row.char_accuracy)
        curve = []
        for prefix in prefixes:
            eligible = [
                row for row in known_rows
                if row.evaluated_index is not None and row.evaluated_index <= prefix
            ]
            if not eligible:
                continue
            best = max(eligible, key=lambda row: row.char_accuracy)
            curve.append({
                "prefix": prefix,
                "eligible_finalists": len(eligible),
                "best_char_accuracy": round(best.char_accuracy, 6),
                "best_mask": list(best.mask),
                "best_source": best.source,
                "best_run": best.run_label,
                "best_evaluated_index": best.evaluated_index,
                "best_preview": best.preview,
            })
        reports.append({
            "test_id": test_id,
            "finalist_count": len(rows),
            "matched_finalist_count": len(known_rows),
            "best_overall": compact(best_overall),
            "curve": curve,
        })
    return {
        "summary": {
            "test_count": len(reports),
            "finalist_count": len(finalists),
        },
        "tests": reports,
    }


def compact(row: Finalist) -> dict[str, Any]:
    return {
        "candidate_id": row.candidate_id,
        "mask": list(row.mask),
        "source": row.source,
        "run": row.run_label,
        "evaluated_index": row.evaluated_index,
        "char_accuracy": round(row.char_accuracy, 6),
        "preview": row.preview,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Copiale Breadth Curve Report",
        "",
        "Ground truth is used only to label already-produced finalists.",
        "",
    ]
    for test in payload["tests"]:
        best = test["best_overall"]
        lines.extend([
            f"## {test['test_id']}",
            "",
            f"Best scored finalist: {format_percent(best['char_accuracy'])} "
            f"mask `{','.join(best['mask']) or '(none)'}` "
            f"at evaluated index `{best['evaluated_index']}`.",
            "",
            "| Prefix | Matched Finalists | Best Char | Best Mask | Source | Run | Eval Index | Preview |",
            "|---:|---:|---:|---|---|---|---:|---|",
        ])
        for row in test["curve"]:
            lines.append(
                "| {prefix} | {count} | {char} | {mask} | {source} | {run} | {idx} | {preview} |".format(
                    prefix=row["prefix"],
                    count=row["eligible_finalists"],
                    char=format_percent(row["best_char_accuracy"]),
                    mask=",".join(row["best_mask"]) or "(none)",
                    source=row["best_source"],
                    run=row["best_run"],
                    idx=row["best_evaluated_index"],
                    preview=str(row["best_preview"]).replace("|", "/"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
