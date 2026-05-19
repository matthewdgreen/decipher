#!/usr/bin/env python3
"""Explain a held-out language-ranker failure family.

This is an offline calibration/reporting helper. It reads a ranker
``summary.json`` and the source global-repair candidate JSON files, then
compares the model's top prediction against the post-hoc best candidate for a
selected held-out group. Ground truth appears only through already-produced
post-hoc labels and calibration deltas in the input artifacts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


SOURCE_RE = re.compile(r"^global_repair:(?P<file>.+?\.json):top_variant:(?P<rank>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report why a language candidate ranker missed a held-out family."
    )
    parser.add_argument("summary_json", help="Ranker summary.json from report_language_candidate_ranker.py")
    parser.add_argument(
        "--group",
        required=True,
        help="Exact group name or substring identifying the held-out group to inspect.",
    )
    parser.add_argument(
        "--candidate-dir",
        action="append",
        default=[],
        help="Directory containing global-repair candidate JSON files. May be repeated.",
    )
    parser.add_argument("--output", default="", help="Markdown output path.")
    args = parser.parse_args()

    summary_path = resolve_path(Path(args.summary_json))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    group = find_group(payload, args.group)
    candidate_dirs = [
        resolve_path(Path(path))
        for path in (args.candidate_dir or default_candidate_dirs(summary_path))
    ]
    index = build_candidate_index(candidate_dirs)
    report = render_report(payload, group, index, summary_path=summary_path)
    output = (
        resolve_path(Path(args.output))
        if args.output
        else summary_path.with_suffix(f".{safe_name(args.group)}.failure.md")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {output}")


def find_group(payload: dict[str, Any], needle: str) -> dict[str, Any]:
    groups = [group for group in payload.get("groups") or [] if isinstance(group, dict)]
    exact = [group for group in groups if str(group.get("group")) == needle]
    if exact:
        return exact[0]
    matches = [group for group in groups if needle in str(group.get("group"))]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available = "\n".join(str(group.get("group")) for group in groups)
        raise SystemExit(f"No group matched {needle!r}. Available groups:\n{available}")
    raise SystemExit(
        f"Group selector {needle!r} matched multiple groups:\n"
        + "\n".join(str(group.get("group")) for group in matches)
    )


def default_candidate_dirs(summary_path: Path) -> list[str]:
    root = REPO_ROOT / "artifacts" / "language_quality"
    return [
        str(root / "global_repair_ranker_inputs"),
        str(root / "global_repair_ranker_inputs_broad"),
        str(summary_path.parent),
    ]


def build_candidate_index(candidate_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for directory in candidate_dirs:
        if directory.is_file():
            paths = [directory]
        elif directory.exists():
            paths = sorted(directory.glob("*global_repair*.json"))
        else:
            continue
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows = payload.get("top_variants") if isinstance(payload.get("top_variants"), list) else []
            for rank, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                source = f"global_repair:{path.name}:top_variant:{rank}"
                index[source] = {
                    "path": str(path),
                    "rank": rank,
                    "row": row,
                    "payload": payload,
                }
    return index


def render_report(
    payload: dict[str, Any],
    group: dict[str, Any],
    index: dict[str, dict[str, Any]],
    *,
    summary_path: Path,
) -> str:
    top_predicted = first(group.get("top_predicted"))
    best_label = group.get("best_label") if isinstance(group.get("best_label"), dict) else first(group.get("top_labeled"))
    top_row = lookup_row(top_predicted, index)
    best_row = lookup_row(best_label, index)
    policy_rows = policy_top_rows(group, index)

    lines = [
        "# Language Ranker Failure-Family Report",
        "",
        "Ground truth is used only through post-hoc labels already present in calibration artifacts.",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| ranker summary | `{summary_path}` |",
        f"| holdout group | `{group.get('group')}` |",
        f"| holdout candidates | {group.get('holdout_count')} |",
        f"| training candidates | {group.get('train_count')} |",
        f"| best-label rank under model | {group.get('best_label_rank')} |",
        f"| top-predicted label gap | {format_percent(group.get('top_predicted_label_gap'))} |",
        f"| top predicted source | `{top_predicted.get('source') if top_predicted else ''}` |",
        f"| best labeled source | `{best_label.get('source') if best_label else ''}` |",
        "",
        "## Candidate Comparison",
        "",
        "| Candidate | Label | Raw Model | Mask | Edits | Robust | Balanced | Validation Avg | Dict Avg | LQ Avg | Preview |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    lines.append(candidate_line("top predicted", top_predicted, top_row))
    lines.append(candidate_line("best labeled", best_label, best_row))
    for name, compact, loaded in policy_rows:
        lines.append(candidate_line(f"policy: {name}", compact, loaded))

    if group.get("feature_deltas"):
        lines.extend([
            "",
            "## Feature Deltas",
            "",
            "Positive deltas mean the model's top prediction had more of that signal than the post-hoc best candidate.",
            "",
            "| Feature | Top Predicted | Best Labeled | Delta | Z-Delta |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in group.get("feature_deltas") or []:
            lines.append(
                "| {feature} | {top} | {best} | {delta} | {zdelta} |".format(
                    feature=row.get("feature"),
                    top=format_number(row.get("top_predicted")),
                    best=format_number(row.get("best_labeled")),
                    delta=format_signed(row.get("delta")),
                    zdelta=format_signed(row.get("z_delta")),
                )
            )

    lines.extend(page_evidence_section("Top Predicted", top_row))
    lines.extend(page_evidence_section("Best Labeled", best_row))
    lines.extend(policy_section(group))
    lines.extend(conclusion_section(group, top_row, best_row))
    return "\n".join(lines).rstrip() + "\n"


def first(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, dict):
        return value
    return {}


def lookup_row(compact: dict[str, Any], index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = str(compact.get("source") or "")
    return index.get(source, {})


def policy_top_rows(
    group: dict[str, Any],
    index: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    rows = []
    policies = group.get("policy_ranks") if isinstance(group.get("policy_ranks"), dict) else {}
    for name, policy in sorted(policies.items()):
        if not isinstance(policy, dict):
            continue
        compact = {
            "source": policy.get("top_source"),
            "label": policy.get("top_label"),
            "mask": policy.get("top_mask") or [],
            "raw_score": policy.get("top_score"),
            "preview": "",
        }
        rows.append((name, compact, lookup_row(compact, index)))
    return rows


def candidate_line(label: str, compact: dict[str, Any], loaded: dict[str, Any]) -> str:
    row = loaded.get("row") if isinstance(loaded.get("row"), dict) else {}
    edits = row.get("edits") or []
    return (
        f"| {label} | {format_percent(compact.get('label'))} | "
        f"{format_number(compact.get('raw_score'))} | "
        f"{escape_cell(','.join(compact.get('mask') or row.get('mask') or []))} | "
        f"{escape_cell('; '.join(edits))} | "
        f"{format_number(row.get('page_robust_score'))} | "
        f"{format_number(row.get('page_balanced_score'))} | "
        f"{format_number(row.get('page_validation_avg'))} | "
        f"{format_number(row.get('page_dict_avg'))} | "
        f"{format_number(row.get('page_language_quality_avg'))} | "
        f"{escape_cell(str(compact.get('preview') or row.get('preview') or '')[:160])} |"
    )


def page_evidence_section(title: str, loaded: dict[str, Any]) -> list[str]:
    row = loaded.get("row") if isinstance(loaded.get("row"), dict) else {}
    evidence = row.get("repair_evidence") if isinstance(row.get("repair_evidence"), dict) else {}
    pages = evidence.get("pages") if isinstance(evidence.get("pages"), list) else []
    lines = [
        "",
        f"## Page Evidence: {title}",
        "",
        "| Page | dVal | dLQ | dDict | dPseudo | dBinary | dChar* | Runtime Flags | Calibration Flags | Changed Excerpt |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    if not pages:
        lines.append("|  |  |  |  |  |  |  | no page evidence found |  |  |")
        return lines
    for page in pages:
        excerpt = page.get("changed_excerpt") if isinstance(page.get("changed_excerpt"), dict) else {}
        lines.append(
            "| {page} | {dval} | {dlq} | {ddict} | {dpseudo} | {dbin} | {dchar} | {runtime} | {calibration} | {excerpt} |".format(
                page=page.get("test_id") or "",
                dval=format_signed(page.get("validation_delta")),
                dlq=format_signed(page.get("language_quality_delta")),
                ddict=format_signed(page.get("dict_rate_delta")),
                dpseudo=format_signed(page.get("pseudo_word_fraction_delta")),
                dbin=format_signed(page.get("binary_ngram_fit_delta")),
                dchar=format_signed(page.get("post_hoc_char_delta")),
                runtime=escape_cell(", ".join(page.get("runtime_flags") or [])),
                calibration=escape_cell(", ".join(page.get("calibration_flags") or [])),
                excerpt=escape_cell(format_excerpt(excerpt)),
            )
        )
    lines.append("")
    lines.append("*dChar is post-hoc calibration only and was not available to the ranker.")
    return lines


def policy_section(group: dict[str, Any]) -> list[str]:
    policies = group.get("policy_ranks") if isinstance(group.get("policy_ranks"), dict) else {}
    if not policies:
        return []
    lines = [
        "",
        "## Simple Policy Behavior",
        "",
        "| Policy | Best Candidate Rank | Top Label | Top Mask | Feature |",
        "|---|---:|---:|---|---|",
    ]
    for name, policy in sorted(policies.items()):
        if not isinstance(policy, dict):
            continue
        lines.append(
            "| {name} | {rank} | {label} | {mask} | {feature} |".format(
                name=name,
                rank=policy.get("best_label_rank"),
                label=format_percent(policy.get("top_label")),
                mask=escape_cell(",".join(policy.get("top_mask") or [])),
                feature=policy.get("feature") or "",
            )
        )
    return lines


def conclusion_section(group: dict[str, Any], top_row: dict[str, Any], best_row: dict[str, Any]) -> list[str]:
    top = top_row.get("row") if isinstance(top_row.get("row"), dict) else {}
    best = best_row.get("row") if isinstance(best_row.get("row"), dict) else {}
    lines = [
        "",
        "## Diagnostic Read",
        "",
    ]
    if not top or not best:
        lines.append("- Could not load both underlying candidate rows, so diagnosis is limited to compact ranker summary fields.")
        return lines
    top_mask = set(top.get("mask") or [])
    best_mask = set(best.get("mask") or [])
    if top_mask != best_mask:
        lines.append(
            f"- The model preferred mask `{','.join(sorted(top_mask))}` while the post-hoc best used `{','.join(sorted(best_mask))}`."
        )
    gap = float(group.get("top_predicted_label_gap") or 0.0)
    if gap >= 0.01:
        lines.append(f"- This is a real miss, not a harmless near-tie: the post-hoc label gap is {format_percent(gap)}.")
    elif gap:
        lines.append(f"- This is mostly a near-tie: the post-hoc label gap is {format_percent(gap)}.")
    feature_deltas = group.get("feature_deltas") if isinstance(group.get("feature_deltas"), list) else []
    if feature_deltas:
        strongest = feature_deltas[0]
        lines.append(
            f"- The strongest misleading signal was `{strongest.get('feature')}`, where the top prediction beat the best candidate by {format_signed(strongest.get('delta'))}."
        )
    top_evidence = top.get("repair_evidence") if isinstance(top.get("repair_evidence"), dict) else {}
    best_evidence = best.get("repair_evidence") if isinstance(best.get("repair_evidence"), dict) else {}
    lines.append(
        "- Runtime suspicious pages: top predicted `{}` vs best labeled `{}`.".format(
            top_evidence.get("runtime_suspicious_pages", ""),
            best_evidence.get("runtime_suspicious_pages", ""),
        )
    )
    lines.append(
        "- Post-hoc regressed pages: top predicted `{}` vs best labeled `{}`.".format(
            top_evidence.get("post_hoc_pages_regressed", ""),
            best_evidence.get("post_hoc_pages_regressed", ""),
        )
    )
    lines.append(
        "- Practical next feature idea: penalize candidates whose high text-quality signals come from a narrower mask family when a competing mask family repeatedly wins post-hoc across the same source experiment."
    )
    return lines


def format_excerpt(value: dict[str, Any]) -> str:
    if not value or not value.get("changed"):
        return ""
    return f"{value.get('before') or ''} -> {value.get('after') or ''}"


def format_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def format_signed(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def escape_cell(text: str) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    return cleaned.strip("_")[:80] or "group"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
