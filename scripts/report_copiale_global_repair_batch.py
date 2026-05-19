#!/usr/bin/env python3
"""Summarize a batch of Copiale multi-page global-repair reports.

This script is diagnostic. It may report post-hoc character accuracy when the
child reports contain it, but that score is never a runtime selection signal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize run_copiale_global_repair_batch.py output."
    )
    parser.add_argument("manifest_json", help="Batch manifest JSON.")
    parser.add_argument("--top-n", type=int, default=16, help="Rows to show in top-candidate tables.")
    parser.add_argument(
        "--per-label-top-n",
        type=int,
        default=12,
        help="Rows to show in each per-label rank matrix.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Markdown output path. Defaults to batch_summary.md beside the manifest.",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="JSON output path. Defaults to batch_summary.json beside the manifest.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_json).resolve()
    manifest = load_json(manifest_path)
    summary = summarize_manifest(
        manifest,
        manifest_path=manifest_path,
        top_n=max(1, args.top_n),
        per_label_top_n=max(1, args.per_label_top_n),
    )

    output_path = Path(args.output).resolve() if args.output else manifest_path.with_name("batch_summary.md")
    json_output_path = (
        Path(args.json_output).resolve() if args.json_output else manifest_path.with_name("batch_summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_markdown(
            summary,
            top_n=max(1, args.top_n),
            per_label_top_n=max(1, args.per_label_top_n),
        ),
        encoding="utf-8",
    )
    json_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {json_output_path}")


def summarize_manifest(
    manifest: dict[str, Any], *, manifest_path: Path, top_n: int, per_label_top_n: int
) -> dict[str, Any]:
    rows = manifest.get("rows") if isinstance(manifest.get("rows"), list) else []
    label_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    mask_counts: Counter[str] = Counter()
    edit_counts: Counter[str] = Counter()
    pick_key_counts: Counter[str] = Counter()

    for manifest_row in rows:
        if not isinstance(manifest_row, dict):
            continue
        label = str(manifest_row.get("label") or "")
        json_path_value = manifest_row.get("json_output") or ""
        child_path = Path(json_path_value)
        if not child_path.is_absolute():
            child_path = (manifest_path.parent / child_path).resolve()
        child = load_json(child_path) if child_path.exists() else {}
        candidates = collect_candidates(child, label=label, child_path=child_path)
        candidate_rows.extend(candidates)
        label_unique_candidates = dedupe_candidates(candidates)
        label_ranked_candidates = attach_metric_ranks(label_unique_candidates)

        ranker_pick = best_source(candidates, source="ranker_diverse_pick")
        robust_pick = best_source(candidates, source="robust_pick")
        best_review = best_by_char(
            [row for row in candidates if row["source"] in {"ranker_diverse", "ranker_top"}]
        )
        best_robust_list = best_by_char(
            [row for row in candidates if row["source"] in {"robust", "robust_pick"}]
        )
        best_any = best_by_char(candidates)

        for pick_name, pick in [("ranker", ranker_pick), ("robust", robust_pick)]:
            if not pick:
                continue
            mask_counts[mask_key(pick)] += 1
            for edit in pick.get("edits") or []:
                edit_counts[str(edit)] += 1
            pick_key_counts[f"{pick_name}:{variant_key(pick)}"] += 1

        label_rows.append({
            "label": label,
            "status": manifest_row.get("status"),
            "json_output": str(child_path),
            "ranker_pick": ranker_pick,
            "robust_pick": robust_pick,
            "best_review": best_review,
            "best_robust_list": best_robust_list,
            "best_any": best_any,
            "ranker_minus_robust_char": subtract_metric(ranker_pick, robust_pick, "post_hoc_char_avg"),
            "ranker_minus_robust_lq": subtract_metric(
                ranker_pick, robust_pick, "language_quality_rank_score"
            ),
            "ranker_minus_robust_robust_score": subtract_metric(
                ranker_pick, robust_pick, "page_robust_score"
            ),
            "ranker_robust_same_variant": (
                bool(ranker_pick and robust_pick and variant_key(ranker_pick) == variant_key(robust_pick))
            ),
            "ranker_robust_same_mask": (
                bool(ranker_pick and robust_pick and mask_key(ranker_pick) == mask_key(robust_pick))
            ),
            "candidate_rank_matrix": sorted(
                label_ranked_candidates,
                key=lambda row: (
                    row.get("gt_char_rank") or 10**9,
                    row.get("lq_rank") or 10**9,
                    row.get("robust_rank") or 10**9,
                ),
            )[:per_label_top_n],
        })

    completed = [row for row in label_rows if row.get("ranker_pick") and row.get("robust_pick")]
    char_deltas = [
        row["ranker_minus_robust_char"]
        for row in completed
        if isinstance(row.get("ranker_minus_robust_char"), (int, float))
    ]
    ranker_better = sum(1 for value in char_deltas if value > 1e-9)
    robust_better = sum(1 for value in char_deltas if value < -1e-9)
    ties = sum(1 for value in char_deltas if abs(value) <= 1e-9)

    unique_candidates = attach_metric_ranks(dedupe_candidates(candidate_rows), group_key=None)
    summary = {
        "manifest_json": str(manifest_path),
        "experiment_json": manifest.get("experiment_json"),
        "section": manifest.get("section"),
        "language_quality_ranker": manifest.get("language_quality_ranker"),
        "ground_truth_note": (
            "post_hoc_char_avg is reported only for diagnostics after candidate generation."
        ),
        "aggregate": {
            "labels": len(label_rows),
            "completed_pick_pairs": len(completed),
            "ranker_better_than_robust": ranker_better,
            "robust_better_than_ranker": robust_better,
            "ranker_robust_ties": ties,
            "mean_ranker_minus_robust_char": mean(char_deltas) if char_deltas else None,
            "best_ranker_pick": best_by_char(
                [row["ranker_pick"] for row in label_rows if row.get("ranker_pick")]
            ),
            "best_robust_pick": best_by_char(
                [row["robust_pick"] for row in label_rows if row.get("robust_pick")]
            ),
            "best_review_candidate": best_by_char(
                [row for row in unique_candidates if row["source"] in {"ranker_diverse", "ranker_top"}]
            ),
            "best_any_candidate": best_by_char(unique_candidates),
            "recurring_masks": counter_rows(mask_counts),
            "recurring_edits": counter_rows(edit_counts),
            "recurring_exact_picks": counter_rows(pick_key_counts),
        },
        "labels_detail": label_rows,
        "top_candidates_by_char": sorted(
            unique_candidates, key=lambda row: metric(row, "post_hoc_char_avg"), reverse=True
        )[:top_n],
        "top_candidates_by_lq": sorted(
            unique_candidates, key=lambda row: metric(row, "language_quality_rank_score"), reverse=True
        )[:top_n],
        "top_candidates_by_robust_score": sorted(
            unique_candidates, key=lambda row: metric(row, "page_robust_score"), reverse=True
        )[:top_n],
    }
    return summary


def collect_candidates(child: dict[str, Any], *, label: str, child_path: Path) -> list[dict[str, Any]]:
    if not child:
        return []
    rows: list[dict[str, Any]] = []
    ranker = child.get("language_quality_ranker") if isinstance(child.get("language_quality_ranker"), dict) else {}
    top_variants = child.get("top_variants") if isinstance(child.get("top_variants"), list) else []
    top_accepted = (
        child.get("top_accepted_variants") if isinstance(child.get("top_accepted_variants"), list) else []
    )
    for index, row in enumerate(top_variants):
        source = "robust_pick" if index == 0 else "robust"
        rows.append(compact_candidate(row, label=label, child_path=child_path, source=source, rank=index + 1))
    for index, row in enumerate(top_accepted):
        rows.append(
            compact_candidate(row, label=label, child_path=child_path, source="accepted", rank=index + 1)
        )
    for section_name, source in [
        ("diverse_review_shortlist", "ranker_diverse"),
        ("review_shortlist", "ranker_review"),
        ("top_by_ranker", "ranker_top"),
    ]:
        section_rows = ranker.get(section_name) if isinstance(ranker.get(section_name), list) else []
        for index, row in enumerate(section_rows):
            row_source = "ranker_diverse_pick" if source == "ranker_diverse" and index == 0 else source
            rows.append(compact_candidate(row, label=label, child_path=child_path, source=row_source, rank=index + 1))
    return rows


def compact_candidate(
    row: Any, *, label: str, child_path: Path, source: str, rank: int
) -> dict[str, Any]:
    row = row if isinstance(row, dict) else {}
    repair = row.get("repair_acceptance") if isinstance(row.get("repair_acceptance"), dict) else {}
    evidence = row.get("repair_evidence") if isinstance(row.get("repair_evidence"), dict) else {}
    return {
        "label": label,
        "source": source,
        "rank": rank,
        "child_json": str(child_path),
        "edits": [str(item) for item in (row.get("edits") or [])],
        "mask": [str(item) for item in (row.get("mask") or [])],
        "language_quality_rank_score": number(row.get("language_quality_rank_score")),
        "language_quality_rank_normalized": number(row.get("language_quality_rank_normalized")),
        "page_robust_score": number(row.get("page_robust_score")),
        "page_validation_avg": number(row.get("page_validation_avg")),
        "page_validation_min": number(row.get("page_validation_min")),
        "page_balanced_score": number(row.get("page_balanced_score")),
        "page_language_quality_avg": number(row.get("page_language_quality_avg")),
        "fragment_illusion_penalty": number(row.get("fragment_illusion_penalty")),
        "post_hoc_char_avg": number(row.get("post_hoc_char_avg")),
        "repair_decision": repair.get("decision"),
        "repair_accepted": repair.get("accepted"),
        "runtime_pages_improved": evidence.get("runtime_pages_improved"),
        "runtime_pages_regressed": evidence.get("runtime_pages_regressed"),
        "post_hoc_pages_improved": evidence.get("post_hoc_pages_improved"),
        "post_hoc_pages_regressed": evidence.get("post_hoc_pages_regressed"),
        "runtime_suspicious_pages": evidence.get("runtime_suspicious_pages"),
        "calibration_suspicious_pages": evidence.get("calibration_suspicious_pages"),
        "preview": str(row.get("preview") or "")[:220],
    }


def best_source(candidates: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    for row in candidates:
        if row.get("source") == source:
            return row
    return {}


def best_by_char(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row for row in candidates if isinstance(row.get("post_hoc_char_avg"), (int, float))]
    if not present:
        return {}
    return max(present, key=lambda row: row["post_hoc_char_avg"])


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_priority = {
        "ranker_diverse_pick": 0,
        "robust_pick": 1,
        "ranker_diverse": 2,
        "ranker_top": 3,
        "ranker_review": 4,
        "robust": 5,
        "accepted": 6,
    }
    for row in candidates:
        key = (str(row.get("label")), mask_key(row), edit_key(row))
        old = by_key.get(key)
        if old is None:
            row = dict(row)
            row["sources"] = [source_rank(row)]
            by_key[key] = row
            continue
        old_sources = old.setdefault("sources", [source_rank(old)])
        row_source = source_rank(row)
        if row_source not in old_sources:
            old_sources.append(row_source)
        if source_priority.get(str(row.get("source")), 99) < source_priority.get(str(old.get("source")), 99):
            merged = dict(row)
            merged["sources"] = old_sources
            by_key[key] = merged
    return list(by_key.values())


def subtract_metric(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    left_value = left.get(key) if isinstance(left, dict) else None
    right_value = right.get(key) if isinstance(right, dict) else None
    if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
        return None
    return left_value - right_value


def render_markdown(summary: dict[str, Any], *, top_n: int, per_label_top_n: int) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Copiale Global Repair Batch Summary",
        "",
        "Ground truth is used only for post-hoc diagnostics in this report. Runtime selection signals are LQ, validation, robust, and ensemble-style scores.",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| labels | {aggregate['labels']} |",
        f"| completed ranker/robust pick pairs | {aggregate['completed_pick_pairs']} |",
        f"| ranker pick beats robust pick | {aggregate['ranker_better_than_robust']} |",
        f"| robust pick beats ranker pick | {aggregate['robust_better_than_ranker']} |",
        f"| ties | {aggregate['ranker_robust_ties']} |",
        f"| mean ranker-minus-robust char | {fmt_signed_pct(aggregate['mean_ranker_minus_robust_char'])} |",
        f"| best ranker pick | {candidate_inline(aggregate.get('best_ranker_pick'))} |",
        f"| best robust pick | {candidate_inline(aggregate.get('best_robust_pick'))} |",
        f"| best review candidate | {candidate_inline(aggregate.get('best_review_candidate'))} |",
        f"| best any candidate | {candidate_inline(aggregate.get('best_any_candidate'))} |",
        "",
        "## Label Comparison",
        "",
        "| Label | Ranker Pick | Robust Pick | Delta Char | Same Mask | Best Review | Best Robust List |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in summary["labels_detail"]:
        lines.append(
            f"| `{row['label']}` | {candidate_inline(row.get('ranker_pick'))} | "
            f"{candidate_inline(row.get('robust_pick'))} | "
            f"{fmt_signed_pct(row.get('ranker_minus_robust_char'))} | "
            f"{yes_no(row.get('ranker_robust_same_mask'))} | "
            f"{candidate_inline(row.get('best_review'))} | "
            f"{candidate_inline(row.get('best_robust_list'))} |"
        )
    lines.extend([
        "",
        f"## Per-Label Rank Matrices",
        "",
        "Each table below is sorted by post-hoc ground-truth character score. The rank columns show where the same candidate landed under the runtime scorers. `GT rank` is diagnostic only.",
    ])
    for row in summary["labels_detail"]:
        lines.extend([
            "",
            f"### `{row['label']}`",
            "",
            candidate_rank_matrix(row.get("candidate_rank_matrix") or []),
        ])
    lines.extend([
        "",
        "## Recurring Masks And Edits",
        "",
        "### Masks",
        "",
        "| Mask | Count |",
        "|---|---:|",
    ])
    lines.extend(counter_table(aggregate.get("recurring_masks") or []))
    lines.extend([
        "",
        "### Edits",
        "",
        "| Edit | Count |",
        "|---|---:|",
    ])
    lines.extend(counter_table(aggregate.get("recurring_edits") or []))
    lines.extend([
        "",
        f"## Top {top_n} By Post-Hoc Character Accuracy",
        "",
        candidate_table(summary.get("top_candidates_by_char") or []),
        "",
        f"## Top {top_n} By Language-Quality Ranker",
        "",
        candidate_table(summary.get("top_candidates_by_lq") or []),
        "",
        f"## Top {top_n} By Robust Score",
        "",
        candidate_table(summary.get("top_candidates_by_robust_score") or []),
    ])
    return "\n".join(lines).rstrip() + "\n"


def candidate_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Label | Source | Rank | Edits | Mask | Char | LQ | Robust | Decision | Preview |",
        "|---|---|---:|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('label')}` | `{row.get('source')}` | {row.get('rank') or ''} | "
            f"{html_join(row.get('edits') or [])} | `{','.join(row.get('mask') or [])}` | "
            f"{fmt_pct(row.get('post_hoc_char_avg'))} | "
            f"{fmt_float(row.get('language_quality_rank_score'))} | "
            f"{fmt_float(row.get('page_robust_score'))} | "
            f"{row.get('repair_decision') or ''} | {escape_md(row.get('preview') or '')} |"
        )
    return "\n".join(lines)


def candidate_rank_matrix(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Candidate | GT Char | GT Rank | LQ Rank | LQ | Robust Rank | Robust | Val Rank | Val | Balanced Rank | Balanced | Sources |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {candidate_name(row)} | "
            f"{fmt_pct(row.get('post_hoc_char_avg'))} | "
            f"{fmt_rank(row.get('gt_char_rank'))} | "
            f"{fmt_rank(row.get('lq_rank'))} | "
            f"{fmt_float(row.get('language_quality_rank_score'))} | "
            f"{fmt_rank(row.get('robust_rank'))} | "
            f"{fmt_float(row.get('page_robust_score'))} | "
            f"{fmt_rank(row.get('validation_rank'))} | "
            f"{fmt_float(row.get('page_validation_avg'))} | "
            f"{fmt_rank(row.get('balanced_rank'))} | "
            f"{fmt_float(row.get('page_balanced_score'))} | "
            f"{source_summary(row)} |"
        )
    return "\n".join(lines)


def attach_metric_ranks(
    candidates: list[dict[str, Any]], *, group_key: str | None = "label"
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in candidates]
    groups: dict[str, list[dict[str, Any]]] = {}
    if group_key is None:
        groups["__all__"] = rows
    else:
        for row in rows:
            groups.setdefault(str(row.get(group_key) or ""), []).append(row)
    for group_rows in groups.values():
        assign_rank(group_rows, metric_key="post_hoc_char_avg", rank_key="gt_char_rank")
        assign_rank(group_rows, metric_key="language_quality_rank_score", rank_key="lq_rank")
        assign_rank(group_rows, metric_key="page_robust_score", rank_key="robust_rank")
        assign_rank(group_rows, metric_key="page_validation_avg", rank_key="validation_rank")
        assign_rank(group_rows, metric_key="page_balanced_score", rank_key="balanced_rank")
    return rows


def assign_rank(rows: list[dict[str, Any]], *, metric_key: str, rank_key: str) -> None:
    ranked = [
        row
        for row in rows
        if isinstance(row.get(metric_key), (int, float))
    ]
    ranked.sort(key=lambda row: float(row[metric_key]), reverse=True)
    previous_value: float | None = None
    previous_rank = 0
    for index, row in enumerate(ranked, start=1):
        value = float(row[metric_key])
        if previous_value is None or value != previous_value:
            previous_rank = index
            previous_value = value
        row[rank_key] = previous_rank


def candidate_name(row: dict[str, Any]) -> str:
    edits = html_join(row.get("edits") or [])
    mask = ",".join(row.get("mask") or [])
    return f"{edits}<br>`{mask}`"


def source_summary(row: dict[str, Any]) -> str:
    sources = row.get("sources") if isinstance(row.get("sources"), list) else []
    if sources:
        return "<br>".join(f"`{source}`" for source in sources[:4])
    source = row.get("source")
    rank = row.get("rank")
    if source and rank:
        return f"`{source} #{rank}`"
    return f"`{source or ''}`"


def candidate_inline(row: Any) -> str:
    if not isinstance(row, dict) or not row:
        return ""
    edit = html_join(row.get("edits") or [])
    mask = ",".join(row.get("mask") or [])
    return (
        f"{edit}<br>`{mask}`<br>"
        f"char {fmt_pct(row.get('post_hoc_char_avg'))}, "
        f"LQ {fmt_float(row.get('language_quality_rank_score'))}, "
        f"R {fmt_float(row.get('page_robust_score'))}"
    )


def counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common()]


def counter_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["|  | 0 |"]
    return [f"| `{row['value']}` | {row['count']} |" for row in rows[:16]]


def metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def mask_key(row: dict[str, Any]) -> str:
    return ",".join(str(item) for item in (row.get("mask") or []))


def edit_key(row: dict[str, Any]) -> str:
    return ";".join(str(item) for item in (row.get("edits") or []))


def variant_key(row: dict[str, Any]) -> str:
    return f"{mask_key(row)}|{edit_key(row)}"


def number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def fmt_signed_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:+.2f}%"
    except (TypeError, ValueError):
        return ""


def fmt_rank(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def yes_no(value: Any) -> str:
    return "Y" if value else "n"


def source_rank(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "")
    rank = row.get("rank")
    if rank:
        return f"{source} #{rank}"
    return source


def html_join(values: list[Any]) -> str:
    return "<br>".join(escape_md(str(value)) for value in values)


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
