#!/usr/bin/env python3
"""Explain multi-page Copiale selector misses.

This is an offline calibration report. It may read post-hoc character
accuracy fields already written by an experiment, but it never runs a solver
or feeds those labels back into candidate generation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report why a Copiale multi-page selector preferred one portfolio finalist over another."
    )
    parser.add_argument("experiment_json", help="JSON from run_copiale_multipage_experiment.py")
    parser.add_argument(
        "--section",
        choices=["portfolio_local_repair", "portfolio_refinement", "elite_page_rerank"],
        default="portfolio_local_repair",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    path = resolve_path(Path(args.experiment_json))
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = analyze(payload, section=args.section)
    markdown = render_markdown(report)
    output = (
        resolve_path(Path(args.output))
        if args.output
        else path.with_suffix(f".{args.section}.selector.md")
    )
    json_output = (
        resolve_path(Path(args.json_output))
        if args.json_output
        else output.with_suffix(".json")
    )
    output.write_text(markdown, encoding="utf-8")
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")


def analyze(payload: dict[str, Any], *, section: str) -> dict[str, Any]:
    block = payload.get(section) or {}
    rows = [
        enrich_selector_row(row)
        for row in (block.get("rows") or [])
        if isinstance(row, dict)
    ]
    if not rows:
        raise SystemExit(f"No rows found in section {section!r}.")
    policy = block.get("best_by_policy") or rows[0]
    posthoc = block.get("best_by_post_hoc_char") or max(
        rows,
        key=lambda row: float(row.get("post_hoc_char_avg") or 0.0),
    )
    policy_label = str(policy.get("label") or "")
    posthoc_label = str(posthoc.get("label") or "")
    by_label = {str(row.get("label") or ""): row for row in rows}
    policy_row = by_label.get(policy_label, policy)
    posthoc_row = by_label.get(posthoc_label, posthoc)
    robust_row = sort_rows(rows, "page_robust_score")[0]
    all_features = [row_feature_bundle(row) for row in rows]
    policy_features = row_feature_bundle(policy_row)
    posthoc_features = row_feature_bundle(posthoc_row)
    return {
        "experiment": payload.get("experiment"),
        "test_ids": payload.get("test_ids") or [],
        "section": section,
        "candidate_count": len(rows),
        "rank_policy": block.get("rank_policy"),
        "policy_winner": compact_row(policy_row),
        "robust_policy_winner": compact_row(robust_row),
        "post_hoc_best": compact_row(posthoc_row),
        "post_hoc_gap": round(
            float(posthoc_row.get("post_hoc_char_avg") or 0.0)
            - float(policy_row.get("post_hoc_char_avg") or 0.0),
            6,
        ),
        "post_hoc_best_policy_rank": rank_of(rows, posthoc_label, key="page_balanced_score"),
        "post_hoc_best_robust_rank": rank_of(rows, posthoc_label, key="page_robust_score"),
        "policy_winner_post_hoc_rank": rank_of(rows, policy_label, key="post_hoc_char_avg"),
        "robust_winner_post_hoc_rank": rank_of(rows, str(robust_row.get("label") or ""), key="post_hoc_char_avg"),
        "page_deltas_policy_minus_posthoc": page_delta_rows(policy_row, posthoc_row),
        "scalar_deltas_policy_minus_posthoc": scalar_delta_rows(policy_features, posthoc_features, all_features),
        "validation_component_deltas_policy_minus_posthoc": nested_delta_rows(
            policy_row,
            posthoc_row,
            rows,
            nested_key="validation_components_v2",
        ),
        "language_feature_deltas_policy_minus_posthoc": nested_delta_rows(
            policy_row,
            posthoc_row,
            rows,
            nested_key="language_quality_features",
        ),
        "top_by_policy": [compact_row(row) for row in sort_rows(rows, "page_balanced_score")[:12]],
        "top_by_robust": [compact_row(row) for row in sort_rows(rows, "page_robust_score")[:12]],
        "top_by_post_hoc": [compact_row(row) for row in sort_rows(rows, "post_hoc_char_avg")[:12]],
    }


def enrich_selector_row(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    if enriched.get("page_robust_score") is not None and enriched.get("fragment_illusion_penalty") is not None:
        return enriched
    runtime_scores = [item for item in (enriched.get("page_runtime_scores") or []) if isinstance(item, dict)]
    binary = nested_mean(runtime_scores, "validation_components_v2", "binary_ngram_fit")
    shape = nested_mean(runtime_scores, "validation_components_v2", "language_shape")
    coherence = nested_mean(runtime_scores, "validation_components_v2", "language_coherence")
    content = nested_mean(runtime_scores, "validation_components_v2", "content_word_quality")
    dispersion = nested_mean(runtime_scores, "language_quality_features", "language_evidence_dispersion")
    stability = nested_mean(runtime_scores, "language_quality_features", "language_window_stability")
    repetition = nested_mean(runtime_scores, "language_quality_features", "repetition_control")
    fragment_side = mean([content, coherence, shape])
    support_side = mean([binary, dispersion, stability, repetition])
    illusion = max(0.0, min(1.0, fragment_side - support_side))
    enriched["fragment_illusion_penalty"] = round(illusion, 6)
    std = float(enriched.get("page_validation_std") or 0.0)
    min_validation = float(enriched.get("page_validation_min") or 0.0)
    enriched["page_robust_score"] = round(
        min_validation
        + 0.35 * binary
        + 0.25 * dispersion
        + 0.20 * stability
        + 0.15 * repetition
        - 0.15 * std
        - 0.75 * illusion,
        6,
    )
    return enriched


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row.get("label"),
        "source_portfolio_reason": row.get("source_portfolio_reason"),
        "mask": row.get("mask") or [],
        "page_balanced_score": rounded(row.get("page_balanced_score")),
        "page_robust_score": rounded(row.get("page_robust_score")),
        "fragment_illusion_penalty": rounded(row.get("fragment_illusion_penalty")),
        "page_validation_avg": rounded(row.get("page_validation_avg")),
        "page_validation_min": rounded(row.get("page_validation_min")),
        "page_validation_std": rounded(row.get("page_validation_std")),
        "page_language_quality_avg": rounded(row.get("page_language_quality_avg")),
        "changed_page_count": row.get("changed_page_count"),
        "post_hoc_char_avg": rounded(row.get("post_hoc_char_avg")),
        "post_hoc_page_chars": row.get("post_hoc_page_chars") or [],
    }


def row_feature_bundle(row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for key in (
        "page_balanced_score",
        "page_robust_score",
        "fragment_illusion_penalty",
        "page_validation_avg",
        "page_validation_min",
        "page_validation_std",
        "page_language_quality_avg",
        "page_dict_avg",
        "page_content_char_avg",
        "page_pseudo_word_avg",
        "page_binary_component_avg",
        "page_shape_component_avg",
        "changed_page_count",
    ):
        parsed = float_or_none(row.get(key))
        if parsed is not None:
            features[key] = parsed
    runtime_scores = [item for item in (row.get("page_runtime_scores") or []) if isinstance(item, dict)]
    for nested_key, prefix in (
        ("validation_components_v2", "validation"),
        ("language_quality_features", "lq"),
        ("diagnostics", "diag"),
    ):
        merged = aggregate_nested(runtime_scores, nested_key=nested_key)
        for name, value in merged.items():
            features[f"{prefix}.{name}"] = value
    return features


def aggregate_nested(rows: list[dict[str, Any]], *, nested_key: str) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        nested = row.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key, value in nested.items():
            parsed = float_or_none(value)
            if parsed is not None:
                values.setdefault(str(key), []).append(parsed)
    return {key: mean(items) for key, items in values.items() if items}


def nested_mean(rows: list[dict[str, Any]], nested_key: str, feature: str) -> float:
    values = []
    for row in rows:
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            parsed = float_or_none(nested.get(feature))
            if parsed is not None:
                values.append(parsed)
    return mean(values) if values else 0.0


def scalar_delta_rows(
    policy_features: dict[str, float],
    posthoc_features: dict[str, float],
    all_features: list[dict[str, float]],
    *,
    limit: int = 18,
) -> list[dict[str, Any]]:
    names = sorted(set(policy_features) | set(posthoc_features))
    scales = feature_scales(all_features, names)
    rows = []
    for name in names:
        policy = float(policy_features.get(name) or 0.0)
        posthoc = float(posthoc_features.get(name) or 0.0)
        delta = policy - posthoc
        scale = scales.get(name) or 1.0
        rows.append({
            "feature": name,
            "policy": round(policy, 6),
            "post_hoc_best": round(posthoc, 6),
            "delta_policy_minus_post_hoc": round(delta, 6),
            "z_delta": round(delta / scale, 6),
        })
    rows.sort(key=lambda row: abs(float(row["z_delta"])), reverse=True)
    return rows[:limit]


def nested_delta_rows(
    policy_row: dict[str, Any],
    posthoc_row: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    nested_key: str,
    limit: int = 14,
) -> list[dict[str, Any]]:
    policy = aggregate_nested(policy_row.get("page_runtime_scores") or [], nested_key=nested_key)
    posthoc = aggregate_nested(posthoc_row.get("page_runtime_scores") or [], nested_key=nested_key)
    all_features = [
        aggregate_nested(row.get("page_runtime_scores") or [], nested_key=nested_key)
        for row in rows
    ]
    return scalar_delta_rows(policy, posthoc, all_features, limit=limit)


def page_delta_rows(policy_row: dict[str, Any], posthoc_row: dict[str, Any]) -> list[dict[str, Any]]:
    policy_pages = {
        str(item.get("test_id")): item
        for item in (policy_row.get("post_hoc_page_chars") or [])
        if isinstance(item, dict)
    }
    posthoc_pages = {
        str(item.get("test_id")): item
        for item in (posthoc_row.get("post_hoc_page_chars") or [])
        if isinstance(item, dict)
    }
    rows = []
    for test_id in sorted(set(policy_pages) | set(posthoc_pages)):
        policy = policy_pages.get(test_id, {})
        posthoc = posthoc_pages.get(test_id, {})
        pchar = float(policy.get("char_accuracy") or 0.0)
        bchar = float(posthoc.get("char_accuracy") or 0.0)
        rows.append({
            "test_id": test_id,
            "policy_char": round(pchar, 6),
            "post_hoc_best_char": round(bchar, 6),
            "delta_policy_minus_post_hoc": round(pchar - bchar, 6),
            "policy_edits": policy.get("selected_edits") or [],
            "post_hoc_edits": posthoc.get("selected_edits") or [],
        })
    rows.sort(key=lambda row: float(row["delta_policy_minus_post_hoc"]))
    return rows


def rank_of(rows: list[dict[str, Any]], label: str, *, key: str) -> int | None:
    for idx, row in enumerate(sort_rows(rows, key), start=1):
        if str(row.get("label") or "") == label:
            return idx
    return None


def sort_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(key) or float("-inf")), reverse=True)


def feature_scales(rows: list[dict[str, float]], names: list[str]) -> dict[str, float]:
    scales = {}
    for name in names:
        values = [float(row.get(name) or 0.0) for row in rows]
        if not values:
            scales[name] = 1.0
            continue
        avg = mean(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        scales[name] = math.sqrt(variance) or 1.0
    return scales


def render_markdown(report: dict[str, Any]) -> str:
    policy = report["policy_winner"]
    robust = report["robust_policy_winner"]
    posthoc = report["post_hoc_best"]
    lines = [
        "# Copiale Multi-Page Selector Diagnostics",
        "",
        "Ground truth is used only after generated candidates exist, to label selector misses.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| section | `{report['section']}` |",
        f"| candidates | {report['candidate_count']} |",
        f"| policy winner | `{policy.get('label')}` |",
        f"| policy winner char | {format_percent(policy.get('post_hoc_char_avg'))} |",
        f"| robust winner | `{robust.get('label')}` |",
        f"| robust winner char | {format_percent(robust.get('post_hoc_char_avg'))} |",
        f"| post-hoc best | `{posthoc.get('label')}` |",
        f"| post-hoc best char | {format_percent(posthoc.get('post_hoc_char_avg'))} |",
        f"| post-hoc gap | {format_percent(report.get('post_hoc_gap'))} |",
        f"| post-hoc best policy rank | {report.get('post_hoc_best_policy_rank')} |",
        f"| post-hoc best robust rank | {report.get('post_hoc_best_robust_rank')} |",
        f"| policy winner post-hoc rank | {report.get('policy_winner_post_hoc_rank')} |",
        f"| robust winner post-hoc rank | {report.get('robust_winner_post_hoc_rank')} |",
        "",
        "## Key Candidates",
        "",
        "| Candidate | Label | Mask | Robust | Balanced | Illusion | Page Avg | LQ Avg | Changed Pages | Post-Hoc Char | Page Chars |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        candidate_line("Policy winner", policy),
        candidate_line("Robust winner", robust),
        candidate_line("Post-hoc best", posthoc),
        "",
        "## Page Accuracy Deltas",
        "",
        "| Page | Policy Char | Post-Hoc Best Char | Delta | Policy Edits | Best Edits |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report["page_deltas_policy_minus_posthoc"]:
        lines.append(
            f"| {row['test_id']} | {format_percent(row['policy_char'])} | "
            f"{format_percent(row['post_hoc_best_char'])} | "
            f"{format_percent(row['delta_policy_minus_post_hoc'])} | "
            f"{escape_cell(';'.join(row.get('policy_edits') or []))} | "
            f"{escape_cell(';'.join(row.get('post_hoc_edits') or []))} |"
        )
    lines.extend([
        "",
        "## Largest Feature Deltas",
        "",
        "Positive delta means the policy winner scored higher than the post-hoc best.",
        "",
        feature_table(report["scalar_deltas_policy_minus_posthoc"]),
        "",
        "## Validation Component Deltas",
        "",
        feature_table(report["validation_component_deltas_policy_minus_posthoc"]),
        "",
        "## Language-Quality Feature Deltas",
        "",
        feature_table(report["language_feature_deltas_policy_minus_posthoc"]),
        "",
        "## Top By Policy",
        "",
        rank_table(report["top_by_policy"]),
        "",
        "## Top By Robust Policy",
        "",
        rank_table(report["top_by_robust"]),
        "",
        "## Top By Post-Hoc Character Accuracy",
        "",
        rank_table(report["top_by_post_hoc"]),
    ])
    return "\n".join(lines).rstrip() + "\n"


def candidate_line(label: str, row: dict[str, Any]) -> str:
    page_chars = ", ".join(
        f"{str(item.get('test_id')).rsplit('_', 1)[-1]}:{format_percent(item.get('char_accuracy'))}"
        for item in row.get("post_hoc_page_chars") or []
        if isinstance(item, dict)
    )
    return (
        f"| {label} | `{row.get('label')}` | {mask_label(row.get('mask') or [])} | "
        f"{format_number(row.get('page_robust_score'))} | "
        f"{format_number(row.get('page_balanced_score'))} | "
        f"{format_number(row.get('fragment_illusion_penalty'))} | "
        f"{format_number(row.get('page_validation_avg'))} | "
        f"{format_number(row.get('page_language_quality_avg'))} | "
        f"{row.get('changed_page_count') if row.get('changed_page_count') is not None else ''} | "
        f"{format_percent(row.get('post_hoc_char_avg'))} | {escape_cell(page_chars)} |"
    )


def rank_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Rank | Label | Mask | Robust | Balanced | Illusion | Page Avg | LQ Avg | Changed Pages | Post-Hoc Char |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | `{row.get('label')}` | {mask_label(row.get('mask') or [])} | "
            f"{format_number(row.get('page_robust_score'))} | "
            f"{format_number(row.get('page_balanced_score'))} | "
            f"{format_number(row.get('fragment_illusion_penalty'))} | "
            f"{format_number(row.get('page_validation_avg'))} | "
            f"{format_number(row.get('page_language_quality_avg'))} | "
            f"{row.get('changed_page_count') if row.get('changed_page_count') is not None else ''} | "
            f"{format_percent(row.get('post_hoc_char_avg'))} |"
        )
    return "\n".join(lines)


def feature_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Feature | Policy | Post-Hoc Best | Delta | Z-Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['feature']}` | {format_number(row['policy'])} | "
            f"{format_number(row['post_hoc_best'])} | "
            f"{format_number(row['delta_policy_minus_post_hoc'], signed=True)} | "
            f"{format_number(row['z_delta'], signed=True)} |"
        )
    return "\n".join(lines)


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def rounded(value: Any) -> float | None:
    parsed = float_or_none(value)
    return round(parsed, 6) if parsed is not None else None


def format_percent(value: Any) -> str:
    parsed = float_or_none(value)
    if parsed is None:
        return "n/a"
    return f"{parsed * 100:.1f}%"


def format_number(value: Any, *, signed: bool = False) -> str:
    parsed = float_or_none(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:+.3f}" if signed else f"{parsed:.3f}"


def mask_label(mask: list[Any]) -> str:
    return ",".join(str(item) for item in mask) or "(none)"


def escape_cell(value: str) -> str:
    return value.replace("|", "/").replace("\n", " ")


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    main()
