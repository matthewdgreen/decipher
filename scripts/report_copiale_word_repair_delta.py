#!/usr/bin/env python3
"""Compact before/after tables for Copiale word-repair probes.

Ground truth columns are post-hoc calibration only. Runtime repair generation
and ranking happen upstream in the probe script without benchmark plaintext.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report compact repair deltas from word-hypothesis probe JSON files."
    )
    parser.add_argument("probe_json", nargs="+", help="One or more *.word_hypothesis_repair.json files.")
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument("--output", default="", help="Optional markdown output path.")
    args = parser.parse_args()

    sections = [render_probe(Path(path), top=args.top) for path in args.probe_json]
    markdown = "\n\n".join(section.rstrip() for section in sections).rstrip() + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {output}")
    else:
        print(markdown)


def render_probe(path: Path, *, top: int) -> str:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = collect_variants(payload)
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    baseline_ref = reference_from_variant(baseline)
    variant_refs = {edits_key(row): reference_from_variant(row) for row in rows}
    rows = [row for row in rows if not is_baseline(row)]
    rows.sort(
        key=lambda row: (
            delta(row, before_reference(row, baseline_ref, variant_refs), "post_hoc_char_avg"),
            metric(row, "post_hoc_char_avg"),
        ),
        reverse=True,
    )
    lines = [
        f"# Word-Repair Delta Report: {payload.get('label') or path.stem}",
        "",
        "Ground truth is shown only as post-hoc calibration.",
        "",
        "| Rank | Corrections | Edits | Before | Flags | GT Before -> After | GT NoTarget Before -> After | Adj Before -> After | AdjNoTarget Before -> After | Global | Leverage | Marg | Robust Before -> After |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows[: max(0, top)], start=1):
        before = before_reference(row, baseline_ref, variant_refs)
        diagnostics = row_diagnostics(row, before)
        lines.append(
            f"| {index} | {corrections_cell(row)} | {edits_cell(row)} | {before_cell(before)} | "
            f"{flags_cell(diagnostics)} | "
            f"{pct_pair(metric(before, 'post_hoc_char_avg'), metric(row, 'post_hoc_char_avg'))} | "
            f"{pct_pair(metric(before, 'post_hoc_char_no_target_avg'), metric(row, 'post_hoc_char_no_target_avg'))} | "
            f"{float_pair(metric(before, 'adjudication_score'), metric(row, 'adjudication_score'))} | "
            f"{float_pair(metric(before, 'adjudication_no_target_score'), metric(row, 'adjudication_no_target_score'))} | "
            f"{fmt_float(metric(row, 'global_leverage_score'))} | "
            f"{fmt_float(metric(row, 'target_leverage_score'))} | "
            f"{fmt_float(metric(row, 'marginal_selector_score'))} | "
            f"{float_pair(metric(before, 'page_robust_score'), metric(row, 'page_robust_score'))} |"
        )
    disagreements = disagreement_rows(rows, baseline_ref, variant_refs)
    lines.extend([
        "",
        "## Score/GT Disagreements",
        "",
        "These are the rows where a runtime scorer and post-hoc ground truth move in different directions, or where total GT improves while the target-excluded GT falls.",
        "",
        "| Rank | Flags | Corrections | Edits | Before | GT Δ | GT NoTarget Δ | Adj Δ | AdjNoTarget Δ | Robust Δ |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for index, item in enumerate(disagreements[: max(0, top)], start=1):
        row = item["row"]
        before = item["before"]
        diagnostics = item["diagnostics"]
        lines.append(
            f"| {index} | {flags_cell(diagnostics)} | {corrections_cell(row)} | "
            f"{edits_cell(row)} | {before_cell(before)} | "
            f"{fmt_signed_pct(diagnostics.get('gt_delta'))} | "
            f"{fmt_signed_pct(diagnostics.get('gt_no_target_delta'))} | "
            f"{fmt_signed_float(diagnostics.get('adj_delta'))} | "
            f"{fmt_signed_float(diagnostics.get('adj_no_target_delta'))} | "
            f"{fmt_signed_float(diagnostics.get('robust_delta'))} |"
        )
    return "\n".join(lines)


def disagreement_rows(
    rows: list[dict[str, Any]],
    baseline_ref: dict[str, Any],
    variant_refs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        before = before_reference(row, baseline_ref, variant_refs)
        diagnostics = row_diagnostics(row, before)
        if diagnostics["flags"]:
            items.append({
                "row": row,
                "before": before,
                "diagnostics": diagnostics,
                "severity": disagreement_severity(diagnostics),
            })
    return sorted(items, key=lambda item: item["severity"], reverse=True)


def row_diagnostics(row: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    gt_delta = optional_delta(row, before, "post_hoc_char_avg")
    gt_no_target_delta = optional_delta(row, before, "post_hoc_char_no_target_avg")
    adj_delta = optional_delta(row, before, "adjudication_score")
    adj_no_target_delta = optional_delta(row, before, "adjudication_no_target_score")
    robust_delta = optional_delta(row, before, "page_robust_score")
    flags = []
    if sign_mismatch(gt_delta, adj_delta, value_epsilon=0.0005, score_epsilon=0.025):
        flags.append("GT vs Adj")
    if sign_mismatch(gt_no_target_delta, adj_no_target_delta, value_epsilon=0.0005, score_epsilon=0.025):
        flags.append("GT-NoTarget vs AdjNoTarget")
    if sign_mismatch(gt_delta, robust_delta, value_epsilon=0.0005, score_epsilon=0.001):
        flags.append("GT vs Robust")
    if is_positive(gt_delta, 0.0005) and is_negative(gt_no_target_delta, 0.0005):
        flags.append("target-local GT gain")
    if is_positive(gt_delta, 0.0005) and is_negative(adj_no_target_delta, 0.025):
        flags.append("GT up / AdjNoTarget down")
    if is_positive(adj_delta, 0.025) and is_negative(gt_no_target_delta, 0.0005):
        flags.append("Adj up / GT-NoTarget down")
    return {
        "gt_delta": gt_delta,
        "gt_no_target_delta": gt_no_target_delta,
        "adj_delta": adj_delta,
        "adj_no_target_delta": adj_no_target_delta,
        "robust_delta": robust_delta,
        "flags": flags,
    }


def disagreement_severity(diagnostics: dict[str, Any]) -> float:
    score = 0.0
    gt_delta = diagnostics.get("gt_delta")
    gt_no_target_delta = diagnostics.get("gt_no_target_delta")
    adj_delta = diagnostics.get("adj_delta")
    adj_no_target_delta = diagnostics.get("adj_no_target_delta")
    robust_delta = diagnostics.get("robust_delta")
    for value, scale in [
        (gt_delta, 100.0),
        (gt_no_target_delta, 100.0),
        (adj_delta, 0.25),
        (adj_no_target_delta, 0.5),
        (robust_delta, 25.0),
    ]:
        if isinstance(value, (int, float)):
            score += abs(float(value)) * scale
    score += 2.0 * len(diagnostics.get("flags") or [])
    return score


def flags_cell(diagnostics: dict[str, Any]) -> str:
    return "<br>".join(f"`{escape(flag)}`" for flag in diagnostics.get("flags") or [])


def collect_variants(payload: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    if isinstance(payload.get("all_variants"), list):
        variants.extend(normalize_variant(row) for row in payload["all_variants"])
    for section in [
        "top_variants",
        "top_variants_by_word_hypothesis",
        "top_variants_by_adjudication",
        "top_combination_variants_by_adjudication",
        "top_combination_variants_by_marginal",
        "top_variants_by_post_hoc",
    ]:
        for row in payload.get(section) or []:
            variants.append(normalize_variant(row))
    return dedupe_variants(variants)


def normalize_variant(row: Any) -> dict[str, Any]:
    row = row if isinstance(row, dict) else {}
    adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    marginal = row.get("marginal_contribution") if isinstance(row.get("marginal_contribution"), dict) else {}
    normalized = dict(row)
    for key in [
        "adjudication_score",
        "adjudication_no_target_score",
        "target_leverage_score",
    ]:
        if key not in normalized and key in adjudication:
            normalized[key] = adjudication.get(key)
    if "marginal_selector_score" not in normalized:
        normalized["marginal_selector_score"] = marginal.get("marginal_selector_score")
    normalized["_marginal"] = marginal
    return normalized


def dedupe_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = edits_key(row)
        old = best.get(key)
        if old is None or score_presence(row) > score_presence(old):
            best[key] = row
    return list(best.values())


def score_presence(row: dict[str, Any]) -> tuple[int, float]:
    fields = [
        "post_hoc_char_avg",
        "adjudication_score",
        "adjudication_no_target_score",
        "page_robust_score",
        "marginal_selector_score",
    ]
    marginal = row.get("_marginal") if isinstance(row.get("_marginal"), dict) else {}
    has_subset = 1 if isinstance(marginal.get("best_subset"), dict) and marginal.get("best_subset") else 0
    return (
        sum(1 for key in fields if isinstance(row.get(key), (int, float))),
        has_subset,
        metric(row, "post_hoc_char_avg"),
    )


def before_reference(
    row: dict[str, Any],
    baseline: dict[str, Any],
    variant_refs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    marginal = row.get("_marginal") if isinstance(row.get("_marginal"), dict) else {}
    subset = marginal.get("best_subset") if isinstance(marginal.get("best_subset"), dict) else {}
    if subset:
        subset_ref = reference_from_variant(subset)
        if variant_refs is not None:
            full_ref = variant_refs.get(edits_key(subset_ref))
            if full_ref:
                return merged_reference(subset_ref, full_ref)
        return subset_ref
    before = dict(baseline)
    if isinstance(row.get("post_hoc_char_no_target_baseline_avg"), (int, float)):
        before["post_hoc_char_no_target_avg"] = row.get("post_hoc_char_no_target_baseline_avg")
    if isinstance(row.get("post_hoc_char_target_baseline_avg"), (int, float)):
        before["post_hoc_char_target_avg"] = row.get("post_hoc_char_target_baseline_avg")
    return before


def merged_reference(preferred: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fallback)
    for key, value in preferred.items():
        if value not in (None, [], ""):
            merged[key] = value
    return merged


def reference_from_variant(row: dict[str, Any]) -> dict[str, Any]:
    row = row if isinstance(row, dict) else {}
    adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    return {
        "edits": row.get("edits") or [],
        "hypotheses": row.get("hypotheses") or [],
        "adjudication_score": row.get("adjudication_score", adjudication.get("adjudication_score")),
        "adjudication_no_target_score": row.get(
            "adjudication_no_target_score",
            adjudication.get("adjudication_no_target_score"),
        ),
        "target_leverage_score": row.get("target_leverage_score", adjudication.get("target_leverage_score")),
        "global_leverage_score": row.get("global_leverage_score", adjudication.get("global_leverage_score")),
        "page_robust_score": row.get("page_robust_score"),
        "page_validation_avg": row.get("page_validation_avg"),
        "page_language_quality_avg": row.get("page_language_quality_avg"),
        "post_hoc_char_avg": row.get("post_hoc_char_avg"),
        "post_hoc_char_no_target_baseline_avg": row.get("post_hoc_char_no_target_baseline_avg"),
        "post_hoc_char_target_baseline_avg": row.get("post_hoc_char_target_baseline_avg"),
        "post_hoc_char_no_target_avg": row.get("post_hoc_char_no_target_avg"),
        "post_hoc_char_target_avg": row.get("post_hoc_char_target_avg"),
    }


def corrections_cell(row: dict[str, Any]) -> str:
    items = []
    for hypothesis in row.get("word_hypotheses") or []:
        if not isinstance(hypothesis, dict):
            continue
        items.append(f"`{hypothesis.get('observed')}` -> `{hypothesis.get('target')}`")
    return "<br>".join(items[:4])


def edits_cell(row: dict[str, Any]) -> str:
    return "<br>".join(f"`{escape(str(item))}`" for item in (row.get("edits") or [])[:6])


def before_cell(row: dict[str, Any]) -> str:
    edits = [str(item) for item in row.get("edits") or []]
    if not edits:
        return "`baseline`"
    return "<br>".join(f"`{escape(item)}`" for item in edits[:4])


def is_baseline(row: dict[str, Any]) -> bool:
    edits = [str(item) for item in row.get("edits") or []]
    return not edits or edits == ["baseline"]


def edits_key(row: dict[str, Any]) -> str:
    return "|".join(str(item) for item in (row.get("edits") or []))


def metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key) if isinstance(row, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def delta(row: dict[str, Any], before: dict[str, Any], key: str) -> float:
    left = metric(row, key)
    right = metric(before, key)
    if left is None or right is None:
        return float("-inf")
    return left - right


def optional_delta(row: dict[str, Any], before: dict[str, Any], key: str) -> float | None:
    left = metric(row, key)
    right = metric(before, key)
    if left is None or right is None:
        return None
    return left - right


def sign_mismatch(
    left: float | None,
    right: float | None,
    *,
    value_epsilon: float,
    score_epsilon: float,
) -> bool:
    if left is None or right is None:
        return False
    if abs(left) <= value_epsilon or abs(right) <= score_epsilon:
        return False
    return (left > 0.0) != (right > 0.0)


def is_positive(value: float | None, epsilon: float) -> bool:
    return value is not None and value > epsilon


def is_negative(value: float | None, epsilon: float) -> bool:
    return value is not None and value < -epsilon


def pct_pair(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return ""
    return f"{before * 100:.2f}% -> {after * 100:.2f}%"


def float_pair(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return ""
    return f"{before:.3f} -> {after:.3f}"


def fmt_float(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else ""


def fmt_signed_float(value: float | None) -> str:
    return f"{value:+.3f}" if value is not None else ""


def fmt_signed_pct(value: float | None) -> str:
    return f"{value * 100:+.2f}%" if value is not None else ""


def escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
