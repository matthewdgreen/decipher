#!/usr/bin/env python3
"""Summarize repeated Copiale null-mask basins across breadth artifacts.

This is an offline analysis script. It may use benchmark ground truth to label
already-produced candidates, but it never feeds those labels back into solving.
Use it to answer whether a promising null/codeword mask is stable across
ranker views, neighborhoods, consensus-polish runs, or repeated experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.scorer import score_decryption  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    test_id: str
    artifact: str
    run_label: str
    rank_in_artifact: int
    mask: tuple[str, ...]
    source: str
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
        return self.decryption[:150]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report stability of repeated Copiale null-mask candidates."
    )
    parser.add_argument(
        "--experiment-dir",
        default="artifacts/copiale_breadth_experiment/four_page_wide_text_only_default",
        help="Breadth experiment directory containing ranker artifacts.",
    )
    parser.add_argument("--test-id", action="append", default=[])
    parser.add_argument(
        "--mask",
        action="append",
        default=[],
        help="Optional comma-separated mask to focus on, e.g. S038,S104.",
    )
    parser.add_argument("--output", default="", help="Markdown report path.")
    parser.add_argument("--json-output", default="", help="JSON report path.")
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    experiment_dir = resolve_path(Path(args.experiment_dir))
    focus_masks = {parse_mask(value) for value in args.mask}
    allowed_tests = set(args.test_id)
    candidates = load_candidates(
        experiment_dir,
        allowed_tests=allowed_tests,
        focus_masks=focus_masks,
    )
    if args.test_id:
        candidates = [row for row in candidates if row.test_id in allowed_tests]
    if focus_masks:
        candidates = [row for row in candidates if row.mask in focus_masks]
    if not candidates:
        raise SystemExit(f"No candidates found under {experiment_dir}")

    payload = analyze(candidates, top_n=args.top_n)
    markdown = render_markdown(payload)
    output = resolve_path(Path(args.output)) if args.output else experiment_dir / "mask_stability.md"
    json_output = (
        resolve_path(Path(args.json_output))
        if args.json_output
        else experiment_dir / "mask_stability.json"
    )
    output.write_text(markdown, encoding="utf-8")
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")


def load_candidates(
    root: Path,
    *,
    allowed_tests: set[str] | None = None,
    focus_masks: set[tuple[str, ...]] | None = None,
) -> list[Candidate]:
    rows: list[Candidate] = []
    allowed_tests = allowed_tests or set()
    focus_masks = focus_masks or set()
    for artifact in sorted(root.glob("*/automated_only/*/*.json")):
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
        ranker = str(null_step.get("ranker") or artifact.parent.parent.parent.name)
        candidate_rows = list(null_step.get("top_finalists") or [])
        selected = null_step.get("selected")
        if isinstance(selected, dict):
            candidate_rows.insert(0, selected)
        for rank, row in enumerate(candidate_rows, start=1):
            if not isinstance(row, dict):
                continue
            mask = tuple(str(symbol) for symbol in (row.get("mask") or []))
            if focus_masks and mask not in focus_masks:
                continue
            text = str(row.get("decryption") or row.get("plaintext") or "")
            if not text:
                continue
            scored = score_decryption(
                test_id,
                text,
                ground_truth,
                agent_score=0.0,
                status="completed",
            )
            rows.append(
                Candidate(
                    test_id=test_id,
                    artifact=str(artifact),
                    run_label=ranker,
                    rank_in_artifact=rank,
                    mask=mask,
                    source=str(row.get("source") or ""),
                    char_accuracy=float(scored.char_accuracy),
                    language_quality_rank_score=float_or_none(row.get("language_quality_rank_score")),
                    validation_score_v2=float_or_none(row.get("validation_score_v2")),
                    ensemble_score_v1=float_or_none(row.get("ensemble_score_v1")),
                    selection_score=float_or_none(row.get("selection_score")),
                    decryption=text,
                )
            )
    return rows


def analyze(candidates: list[Candidate], *, top_n: int) -> dict[str, Any]:
    tests: dict[str, list[Candidate]] = defaultdict(list)
    for row in candidates:
        tests[row.test_id].append(row)
    reports = []
    for test_id, rows in sorted(tests.items()):
        groups: dict[tuple[str, ...], list[Candidate]] = defaultdict(list)
        for row in rows:
            groups[row.mask].append(row)
        group_reports = [summarize_group(mask, group) for mask, group in groups.items()]
        group_reports.sort(
            key=lambda row: (
                row["best_char_accuracy"],
                row["candidate_count"],
                row["mean_char_accuracy"],
            ),
            reverse=True,
        )
        reports.append({
            "test_id": test_id,
            "candidate_count": len(rows),
            "mask_count": len(group_reports),
            "top_masks": group_reports[:top_n],
        })
    return {
        "summary": {
            "test_count": len(reports),
            "candidate_count": len(candidates),
            "mask_count": sum(report["mask_count"] for report in reports),
        },
        "tests": reports,
    }


def summarize_group(mask: tuple[str, ...], rows: list[Candidate]) -> dict[str, Any]:
    chars = [row.char_accuracy for row in rows]
    best = max(rows, key=lambda row: row.char_accuracy)
    by_text: Counter[str] = Counter(
        hashlib.sha256(row.decryption.encode("utf-8")).hexdigest() for row in rows
    )
    unique_text_count = len(by_text)
    return {
        "mask": list(mask),
        "mask_label": ",".join(mask) if mask else "(none)",
        "candidate_count": len(rows),
        "unique_text_count": unique_text_count,
        "source_counts": dict(sorted(Counter(row.source or "(none)" for row in rows).items())),
        "run_counts": dict(sorted(Counter(row.run_label for row in rows).items())),
        "best_char_accuracy": round(best.char_accuracy, 6),
        "mean_char_accuracy": round(statistics.fmean(chars), 6),
        "median_char_accuracy": round(statistics.median(chars), 6),
        "min_char_accuracy": round(min(chars), 6),
        "max_char_accuracy": round(max(chars), 6),
        "stddev_char_accuracy": round(statistics.pstdev(chars), 6) if len(chars) > 1 else 0.0,
        "best_rank_in_artifact": best.rank_in_artifact,
        "best_source": best.source,
        "best_run": best.run_label,
        "best_lq_rank_score": rounded(best.language_quality_rank_score),
        "best_validation_score_v2": rounded(best.validation_score_v2),
        "best_ensemble_score_v1": rounded(best.ensemble_score_v1),
        "best_selection_score": rounded(best.selection_score),
        "best_preview": best.preview,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Copiale Mask Stability Report",
        "",
        "Ground truth is used only to label already-produced candidates.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"| {key} | {value} |")
    for test in payload["tests"]:
        lines.extend([
            "",
            f"## {test['test_id']}",
            "",
            "| Rank | Mask | Rows | Unique Texts | Best Char | Mean Char | Stddev | Best Source | Sources | Preview |",
            "|---:|---|---:|---:|---:|---:|---:|---|---|---|",
        ])
        for rank, row in enumerate(test["top_masks"], start=1):
            sources = ", ".join(f"{key}:{value}" for key, value in row["source_counts"].items())
            lines.append(
                "| {rank} | {mask} | {rows} | {unique} | {best} | {mean} | {stddev} | {source} | {sources} | {preview} |".format(
                    rank=rank,
                    mask=row["mask_label"],
                    rows=row["candidate_count"],
                    unique=row["unique_text_count"],
                    best=format_percent(row["best_char_accuracy"]),
                    mean=format_percent(row["mean_char_accuracy"]),
                    stddev=format_percent(row["stddev_char_accuracy"]),
                    source=row["best_source"],
                    sources=sources.replace("|", "/"),
                    preview=str(row["best_preview"]).replace("|", "/"),
                )
            )
    return "\n".join(lines) + "\n"


def parse_mask(value: str) -> tuple[str, ...]:
    if not value.strip() or value.strip() == "(none)":
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
