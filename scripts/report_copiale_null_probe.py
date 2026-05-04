#!/usr/bin/env python3
"""Summarize Copiale null-mask probe outputs.

This is intentionally cheap: it reads JSONL produced by
``probe_copiale_null_masks.py`` and compares the solver-selection, validation,
and post-hoc ground-truth winners. If the probe was run with
``--include-all-rows``, this report can be used to tune finalist validation
without rerunning the expensive null-mask solves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.probe_copiale_null_masks import null_mask_validation_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Copiale null-mask probe rankings.")
    parser.add_argument("probe_jsonl", help="JSONL output from probe_copiale_null_masks.py")
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="Number of per-test finalists to print after the summary table.",
    )
    args = parser.parse_args()

    payloads = load_probe_jsonl(Path(args.probe_jsonl))
    reports = [summarize_probe_payload(payload) for payload in payloads]
    print(render_markdown(reports, top=args.top))


def load_probe_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_absolute():
        path = REPO_ROOT / path
    payloads = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payloads.append(json.loads(line))
    return payloads


def summarize_probe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _candidate_rows(payload)
    original_length = max((int(row.get("filtered_length") or 0) for row in rows), default=0)
    for row in rows:
        if "validation_score" not in row:
            validation = null_mask_validation_score(row, original_length=original_length)
            row["validation_score"] = validation["score"]
            row["validation_components"] = validation["components"]

    rows_by_selection = sorted(rows, key=lambda item: (-float(item.get("selection_score") or 0.0), -float(item.get("char_accuracy") or 0.0)))
    rows_by_validation = sorted(rows, key=lambda item: (-float(item.get("validation_score") or 0.0), -float(item.get("char_accuracy") or 0.0)))
    rows_by_char = sorted(rows, key=lambda item: (-float(item.get("char_accuracy") or 0.0), -float(item.get("selection_score") or 0.0)))
    best_selection = rows_by_selection[0] if rows_by_selection else None
    best_validation = rows_by_validation[0] if rows_by_validation else None
    best_char = rows_by_char[0] if rows_by_char else None

    return {
        "test_id": payload.get("test_id") or "",
        "mask_count": payload.get("mask_count") or len(rows),
        "stored_rows": len(rows),
        "has_all_rows": bool(payload.get("all_rows")),
        "best_by_selection": best_selection,
        "best_by_validation": best_validation,
        "best_by_char_accuracy": best_char,
        "char_best_selection_rank": _rank_of(rows_by_selection, best_char),
        "char_best_validation_rank": _rank_of(rows_by_validation, best_char),
        "validation_best_char_gap": _char_gap(best_validation, best_char),
        "selection_best_char_gap": _char_gap(best_selection, best_char),
        "top_by_validation": rows_by_validation,
        "top_by_selection": rows_by_selection,
        "top_by_char_accuracy": rows_by_char,
    }


def render_markdown(reports: list[dict[str, Any]], *, top: int = 8) -> str:
    aggregate = _aggregate_report_metrics(reports)
    lines = [
        "# Copiale Null-Mask Probe Report",
        "",
        "Selection and validation are ground-truth-free. Char columns are post-hoc calibration only.",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| tests | {aggregate['tests']} |",
        f"| validation exact-best hits | {aggregate['validation_exact_hits']} |",
        f"| selection exact-best hits | {aggregate['selection_exact_hits']} |",
        f"| mean validation char gap | {_format_percent(aggregate['mean_validation_gap'])} |",
        f"| mean selection char gap | {_format_percent(aggregate['mean_selection_gap'])} |",
        f"| mean char-best validation rank | {aggregate['mean_char_best_validation_rank']:.2f} |",
        "",
        "## Tests",
        "",
        "| Test | Masks | Rows | Best selection | Best validation | Best char | Char-best rank by validation | Validation char gap |",
        "|---|---:|---:|---|---|---|---:|---:|",
    ]
    for report in reports:
        lines.append(
            "| {test} | {masks} | {rows}{row_note} | {sel} | {val} | {char} | {rank} | {gap} |".format(
                test=report["test_id"],
                masks=report["mask_count"],
                rows=report["stored_rows"],
                row_note="" if report["has_all_rows"] else "*",
                sel=_format_row(report["best_by_selection"], score_name="selection_score"),
                val=_format_row(report["best_by_validation"], score_name="validation_score"),
                char=_format_row(report["best_by_char_accuracy"], score_name="char_accuracy"),
                rank=report["char_best_validation_rank"] or "",
                gap=_format_percent(report["validation_best_char_gap"]),
            )
        )
    if any(not report["has_all_rows"] for report in reports):
        lines.extend([
            "",
            "`Rows*` means the probe JSONL did not include all evaluated rows; rankings use only saved top lists.",
        ])
    for report in reports:
        lines.extend([
            "",
            f"## {report['test_id']}",
            "",
            "| Rank | Mask | Validation | Selection | Char | Dict | Top letter | Preview |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ])
        for rank, row in enumerate(report["top_by_validation"][:top], start=1):
            diagnostics = row.get("diagnostics") or {}
            quality = row.get("quality") or {}
            lines.append(
                "| {rank} | {mask} | {val:.3f} | {sel:.3f} | {char} | {dict_rate:.3f} | {top_letter:.3f} | {preview} |".format(
                    rank=rank,
                    mask=_mask_label(row),
                    val=float(row.get("validation_score") or 0.0),
                    sel=float(row.get("selection_score") or 0.0),
                    char=_format_percent(row.get("char_accuracy")),
                    dict_rate=float(diagnostics.get("dict_rate") or 0.0),
                    top_letter=float(quality.get("top_letter_fraction") or 0.0),
                    preview=str(row.get("preview") or "")[:80],
                )
            )
        miss_lines = _validation_miss_lines(report)
        if miss_lines:
            lines.extend([
                "",
                "Validation miss analysis:",
                "",
                "| Component | Validation winner | Char winner | Delta |",
                "|---|---:|---:|---:|",
                *miss_lines,
            ])
    return "\n".join(lines)


def _aggregate_report_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    validation_gaps = [
        float(report["validation_best_char_gap"])
        for report in reports
        if report.get("validation_best_char_gap") is not None
    ]
    selection_gaps = [
        float(report["selection_best_char_gap"])
        for report in reports
        if report.get("selection_best_char_gap") is not None
    ]
    validation_ranks = [
        int(report["char_best_validation_rank"])
        for report in reports
        if report.get("char_best_validation_rank") is not None
    ]
    return {
        "tests": len(reports),
        "validation_exact_hits": sum(1 for gap in validation_gaps if abs(gap) < 1e-9),
        "selection_exact_hits": sum(1 for gap in selection_gaps if abs(gap) < 1e-9),
        "mean_validation_gap": sum(validation_gaps) / len(validation_gaps) if validation_gaps else None,
        "mean_selection_gap": sum(selection_gaps) / len(selection_gaps) if selection_gaps else None,
        "mean_char_best_validation_rank": (
            sum(validation_ranks) / len(validation_ranks) if validation_ranks else 0.0
        ),
    }


def _candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source_rows = payload.get("all_rows")
    if source_rows:
        return [dict(row) for row in source_rows]
    rows_by_mask: dict[tuple[str, ...], dict[str, Any]] = {}
    for list_name in ("top_by_validation", "top_by_selection", "top_by_char_accuracy"):
        for row in payload.get(list_name) or []:
            rows_by_mask.setdefault(tuple(row.get("mask") or []), dict(row))
    for key in ("best_by_validation", "best_by_selection", "best_by_char_accuracy"):
        row = payload.get(key)
        if row:
            rows_by_mask.setdefault(tuple(row.get("mask") or []), dict(row))
    return list(rows_by_mask.values())


def _validation_miss_lines(report: dict[str, Any]) -> list[str]:
    validation_winner = report.get("best_by_validation")
    char_winner = report.get("best_by_char_accuracy")
    if not validation_winner or not char_winner:
        return []
    if tuple(validation_winner.get("mask") or []) == tuple(char_winner.get("mask") or []):
        return []
    left_components = validation_winner.get("validation_components") or {}
    right_components = char_winner.get("validation_components") or {}
    names = sorted(set(left_components) | set(right_components))
    if not names:
        return []
    lines = [
        f"| mask | {_mask_label(validation_winner)} | {_mask_label(char_winner)} |  |",
        (
            f"| char_accuracy | {_format_percent(validation_winner.get('char_accuracy'))} | "
            f"{_format_percent(char_winner.get('char_accuracy'))} | "
            f"{_format_percent(_char_gap(validation_winner, char_winner))} |"
        ),
    ]
    for name in names:
        left = float(left_components.get(name) or 0.0)
        right = float(right_components.get(name) or 0.0)
        lines.append(f"| {name} | {left:+.3f} | {right:+.3f} | {right - left:+.3f} |")
    return lines


def _rank_of(rows: list[dict[str, Any]], target: dict[str, Any] | None) -> int | None:
    if target is None:
        return None
    target_mask = tuple(target.get("mask") or [])
    for idx, row in enumerate(rows, start=1):
        if tuple(row.get("mask") or []) == target_mask:
            return idx
    return None


def _char_gap(row: dict[str, Any] | None, best_char: dict[str, Any] | None) -> float | None:
    if row is None or best_char is None:
        return None
    return float(best_char.get("char_accuracy") or 0.0) - float(row.get("char_accuracy") or 0.0)


def _format_row(row: dict[str, Any] | None, *, score_name: str) -> str:
    if row is None:
        return ""
    score = row.get(score_name)
    if score_name == "char_accuracy":
        score_text = _format_percent(score)
    else:
        score_text = f"{float(score or 0.0):.3f}"
    return f"{_mask_label(row)} ({score_text}, char {_format_percent(row.get('char_accuracy'))})"


def _mask_label(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    mask = row.get("mask") or []
    return ",".join(mask) if mask else "(none)"


def _format_percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.1%}"


if __name__ == "__main__":
    main()
