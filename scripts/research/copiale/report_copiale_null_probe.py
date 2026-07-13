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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from probe_copiale_null_masks import (
    attach_null_mask_ensemble_scores,
    format_validation_components,
    null_mask_rank_key,
    null_mask_validation_score,
    null_mask_validation_score_v2,
)
from automated.runner import _automated_candidate_diagnostics
from analysis.dictionary import get_dictionary_path, load_word_set


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
    language = str(payload.get("language") or "de")
    for row in rows:
        _backfill_language_diagnostics(row, language=language)
        if "validation_score" not in row:
            validation = null_mask_validation_score(
                row,
                original_length=original_length,
                language=language,
            )
            row["validation_score"] = validation["score"]
            row["validation_components"] = validation["components"]
        validation_v2 = null_mask_validation_score_v2(
            row,
            original_length=original_length,
            language=language,
        )
        row["validation_score_v2"] = validation_v2["score"]
        row["validation_components_v2"] = validation_v2["components"]
    attach_null_mask_ensemble_scores(
        rows,
        original_length=original_length,
        language=language,
    )

    rows_by_selection = sorted(rows, key=lambda item: (-float(item.get("selection_score") or 0.0), -float(item.get("char_accuracy") or 0.0)))
    rows_by_validation = sorted(rows, key=lambda item: (-float(item.get("validation_score_v2") or 0.0), -float(item.get("char_accuracy") or 0.0)))
    rows_by_ensemble = sorted(rows, key=null_mask_rank_key, reverse=True)
    rows_by_char = sorted(rows, key=lambda item: (-float(item.get("char_accuracy") or 0.0), -float(item.get("selection_score") or 0.0)))
    best_selection = rows_by_selection[0] if rows_by_selection else None
    best_validation = rows_by_validation[0] if rows_by_validation else None
    best_ensemble = rows_by_ensemble[0] if rows_by_ensemble else None
    best_char = rows_by_char[0] if rows_by_char else None

    return {
        "test_id": payload.get("test_id") or "",
        "mask_count": payload.get("mask_count") or len(rows),
        "stored_rows": len(rows),
        "has_all_rows": bool(payload.get("all_rows")),
        "best_by_selection": best_selection,
        "best_by_validation": best_validation,
        "best_by_ensemble": best_ensemble,
        "best_by_char_accuracy": best_char,
        "char_best_selection_rank": _rank_of(rows_by_selection, best_char),
        "char_best_validation_rank": _rank_of(rows_by_validation, best_char),
        "char_best_ensemble_rank": _rank_of(rows_by_ensemble, best_char),
        "validation_best_char_gap": _char_gap(best_validation, best_char),
        "ensemble_best_char_gap": _char_gap(best_ensemble, best_char),
        "selection_best_char_gap": _char_gap(best_selection, best_char),
        "capture_by_validation_top_n": _capture_by_top_n(rows_by_validation, best_char, ns=(1, 3, 5, 8, 10)),
        "capture_by_ensemble_top_n": _capture_by_top_n(rows_by_ensemble, best_char, ns=(1, 3, 5, 8, 10)),
        "capture_by_selection_top_n": _capture_by_top_n(rows_by_selection, best_char, ns=(1, 3, 5, 8, 10)),
        "top_by_validation": rows_by_validation,
        "top_by_ensemble": rows_by_ensemble,
        "top_by_selection": rows_by_selection,
        "top_by_char_accuracy": rows_by_char,
    }


def _backfill_language_diagnostics(row: dict[str, Any], *, language: str) -> None:
    """Fill newly added scorer diagnostics from saved full candidate text.

    Older probe JSONL rows may have ``validation_text`` but lack newer
    diagnostics such as content-word quality. Recompute only missing fields so
    reports can compare new scorer versions without rerunning the expensive
    null-mask search.
    """
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        row["diagnostics"] = diagnostics
    if diagnostics.get("dictionary_content_word_count") is not None:
        return
    text = str(row.get("validation_text") or row.get("decryption") or "")
    if not text:
        return
    path = get_dictionary_path(language)
    if not path:
        return
    word_set = load_word_set(path)
    if not word_set:
        return
    recomputed = _automated_candidate_diagnostics(
        text,
        language=language,
        word_list=sorted(word_set),
        binary_model_path=None,
    )
    for key, value in recomputed.items():
        diagnostics.setdefault(key, value)


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
        f"| validation top-3 captures | {aggregate['validation_top_n_captures'][3]}/{aggregate['tests']} |",
        f"| validation top-5 captures | {aggregate['validation_top_n_captures'][5]}/{aggregate['tests']} |",
        f"| validation top-8 captures | {aggregate['validation_top_n_captures'][8]}/{aggregate['tests']} |",
        "",
        "## Tests",
        "",
        "| Test | Masks | Rows | Best selection | Best validation v2 | Best ensemble | Best char | Char-best rank by validation/ensemble | Validation/ensemble gap | Top-N capture |",
        "|---|---:|---:|---|---|---|---|---:|---:|---|",
    ]
    for report in reports:
        lines.append(
            "| {test} | {masks} | {rows}{row_note} | {sel} | {val} | {ens} | {char} | {rank}/{ens_rank} | {gap}/{ens_gap} | {capture} |".format(
                test=report["test_id"],
                masks=report["mask_count"],
                rows=report["stored_rows"],
                row_note="" if report["has_all_rows"] else "*",
                sel=_format_row(report["best_by_selection"], score_name="selection_score"),
                val=_format_row(report["best_by_validation"], score_name="validation_score_v2"),
                ens=_format_row(report["best_by_ensemble"], score_name="ensemble_score_v1"),
                char=_format_row(report["best_by_char_accuracy"], score_name="char_accuracy"),
                rank=report["char_best_validation_rank"] or "",
                ens_rank=report["char_best_ensemble_rank"] or "",
                gap=_format_percent(report["validation_best_char_gap"]),
                ens_gap=_format_percent(report["ensemble_best_char_gap"]),
                capture=_format_capture(report["capture_by_validation_top_n"]),
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
            "| Rank | Mask | Validation v2 | Ensemble | Selection | Char | Dict | Content | Lattice | Pseudo | Binary | Shape | Island | Preview |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for rank, row in enumerate(report["top_by_validation"][:top], start=1):
            diagnostics = row.get("diagnostics") or {}
            components = row.get("validation_components_v2") or {}
            lines.append(
                "| {rank} | {mask} | {val:.3f} | {ensemble:.3f} | {sel:.3f} | {char} | {dict_rate:.3f} | {content:.3f} | {lattice:.3f} | {pseudo:.3f} | {binary:.3f} | {shape:.3f} | {island:.3f} | {preview} |".format(
                    rank=rank,
                    mask=_mask_label(row),
                    val=float(row.get("validation_score_v2") or 0.0),
                    ensemble=float(row.get("ensemble_score_v1") or 0.0),
                    sel=float(row.get("selection_score") or 0.0),
                    char=_format_percent(row.get("char_accuracy")),
                    dict_rate=float(diagnostics.get("dict_rate") or 0.0),
                    content=float(components.get("content_word_quality") or 0.0),
                    lattice=float(components.get("word_lattice_quality") or 0.0),
                    pseudo=float(diagnostics.get("pseudo_word_fraction") or 0.0),
                    binary=float(components.get("binary_ngram_fit") or 0.0),
                    shape=float(
                        components.get("language_shape")
                        if components.get("language_shape") is not None
                        else components.get("german_shape") or 0.0
                    ),
                    island=(
                        abs(float(components.get("repetition_penalty") or 0.0))
                        + abs(float(components.get("template_island_penalty") or 0.0))
                        + abs(float(components.get("function_overuse_penalty") or 0.0))
                    ),
                    preview=str(row.get("preview") or "")[:80],
                )
            )
        if report["top_by_validation"]:
            lines.extend([
                "",
                "Top scalar-validation components:",
            ])
            for rank, row in enumerate(report["top_by_validation"][: min(3, top)], start=1):
                lines.append(
                    f"- {rank}. `{_mask_label(row)}`: "
                    f"{format_validation_components(row.get('validation_components_v2') or {})}"
                )
        if report["top_by_ensemble"]:
            lines.extend([
                "",
                "Top ensemble-only calibration ranks:",
            ])
            for rank, row in enumerate(report["top_by_ensemble"][: min(5, top)], start=1):
                lines.append(
                    f"- {rank}. `{_mask_label(row)}`: "
                    f"ensemble={float(row.get('ensemble_score_v1') or 0.0):+.3f}, "
                    f"validation_v2={float(row.get('validation_score_v2') or 0.0):+.3f}, "
                    f"char={_format_percent(row.get('char_accuracy'))}"
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
    capture_ns = (1, 3, 5, 8, 10)
    return {
        "tests": len(reports),
        "validation_exact_hits": sum(1 for gap in validation_gaps if abs(gap) < 1e-9),
        "selection_exact_hits": sum(1 for gap in selection_gaps if abs(gap) < 1e-9),
        "mean_validation_gap": sum(validation_gaps) / len(validation_gaps) if validation_gaps else None,
        "mean_selection_gap": sum(selection_gaps) / len(selection_gaps) if selection_gaps else None,
        "mean_char_best_validation_rank": (
            sum(validation_ranks) / len(validation_ranks) if validation_ranks else 0.0
        ),
        "validation_top_n_captures": {
            n: sum(1 for report in reports if (report.get("capture_by_validation_top_n") or {}).get(n))
            for n in capture_ns
        },
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
    left_components = validation_winner.get("validation_components_v2") or {}
    right_components = char_winner.get("validation_components_v2") or {}
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


def _capture_by_top_n(
    rows: list[dict[str, Any]],
    target: dict[str, Any] | None,
    *,
    ns: tuple[int, ...],
) -> dict[int, bool]:
    rank = _rank_of(rows, target)
    return {n: bool(rank is not None and rank <= n) for n in ns}


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


def _format_capture(capture: dict[int, bool]) -> str:
    return ", ".join(f"@{n}={'Y' if capture.get(n) else 'n'}" for n in (1, 3, 5, 8, 10))


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
