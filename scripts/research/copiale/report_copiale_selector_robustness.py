#!/usr/bin/env python3
"""Compare Copiale multi-page selector policies across saved experiments.

This is an offline calibration report. It reads post-hoc character accuracy
fields already present in experiment JSON files, but it never runs solvers or
feeds labels back into candidate generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from report_copiale_multipage_selector import (  # noqa: E402
    compact_row,
    enrich_selector_row,
    format_percent,
    mask_label,
    sort_rows,
)


SECTIONS = ("elite_page_rerank", "portfolio_refinement", "portfolio_local_repair")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report post-hoc robustness of Copiale multi-page selector policies."
    )
    parser.add_argument(
        "--experiment-dir",
        default="artifacts/copiale_multipage_experiment",
        help="Directory containing copiale_multipage_*.json experiment outputs.",
    )
    parser.add_argument(
        "--section",
        choices=[*SECTIONS, "all"],
        default="all",
        help="Experiment section to evaluate.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    experiment_dir = resolve_path(Path(args.experiment_dir))
    sections = list(SECTIONS) if args.section == "all" else [args.section]
    report = analyze_directory(experiment_dir, sections=sections)
    markdown = render_markdown(report)
    output = resolve_path(Path(args.output)) if args.output else experiment_dir / "selector_robustness.md"
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


def analyze_directory(experiment_dir: Path, *, sections: list[str]) -> dict[str, Any]:
    files = discover_experiment_jsons(experiment_dir)
    cases: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            skipped.append({"path": str(path), "reason": f"json_decode_error: {exc}"})
            continue
        for section in sections:
            case = analyze_payload_section(payload, path=path, section=section)
            if case is None:
                continue
            cases.append(case)
    return {
        "experiment_dir": str(experiment_dir),
        "sections": sections,
        "file_count": len(files),
        "case_count": len(cases),
        "section_summaries": [summarize_section(section, cases) for section in sections],
        "cases": cases,
        "skipped": skipped,
    }


def analyze_payload_section(
    payload: dict[str, Any],
    *,
    path: Path,
    section: str,
) -> dict[str, Any] | None:
    block = payload.get(section)
    if not isinstance(block, dict):
        return None
    rows = [
        enrich_selector_row(row)
        for row in (block.get("rows") or [])
        if isinstance(row, dict)
    ]
    if not rows:
        return None

    balanced = sort_rows(rows, "page_balanced_score")[0]
    robust = sort_rows(rows, "page_robust_score")[0]
    posthoc = max(rows, key=lambda row: float(row.get("post_hoc_char_avg") or 0.0))
    balanced_char = char_score(balanced)
    robust_char = char_score(robust)
    best_char = char_score(posthoc)
    posthoc_label = str(posthoc.get("label") or "")
    robust_label = str(robust.get("label") or "")
    balanced_label = str(balanced.get("label") or "")
    return {
        "path": str(path),
        "artifact": path.name,
        "experiment": payload.get("experiment"),
        "test_ids": payload.get("test_ids") or [],
        "section": section,
        "candidate_count": len(rows),
        "rank_policy": block.get("rank_policy"),
        "balanced_winner": compact_row(balanced),
        "robust_winner": compact_row(robust),
        "post_hoc_best": compact_row(posthoc),
        "balanced_char": round(balanced_char, 6),
        "robust_char": round(robust_char, 6),
        "best_char": round(best_char, 6),
        "balanced_gap": round(best_char - balanced_char, 6),
        "robust_gap": round(best_char - robust_char, 6),
        "robust_delta_vs_balanced": round(robust_char - balanced_char, 6),
        "robust_exact_hit": robust_label == posthoc_label,
        "balanced_exact_hit": balanced_label == posthoc_label,
        "post_hoc_best_balanced_rank": rank_of(rows, posthoc_label, key="page_balanced_score"),
        "post_hoc_best_robust_rank": rank_of(rows, posthoc_label, key="page_robust_score"),
        "robust_winner_post_hoc_rank": rank_of(rows, robust_label, key="post_hoc_char_avg"),
        "balanced_winner_post_hoc_rank": rank_of(rows, balanced_label, key="post_hoc_char_avg"),
        "regression_over_0_5pct": robust_char + 0.005 < balanced_char,
        "improvement_over_0_5pct": robust_char > balanced_char + 0.005,
    }


def summarize_section(section: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    scoped = [case for case in cases if case["section"] == section]
    if not scoped:
        return {
            "section": section,
            "case_count": 0,
        }
    return {
        "section": section,
        "case_count": len(scoped),
        "balanced_exact_hits": sum(1 for case in scoped if case["balanced_exact_hit"]),
        "robust_exact_hits": sum(1 for case in scoped if case["robust_exact_hit"]),
        "mean_balanced_gap": round(mean(case["balanced_gap"] for case in scoped), 6),
        "mean_robust_gap": round(mean(case["robust_gap"] for case in scoped), 6),
        "mean_robust_delta_vs_balanced": round(
            mean(case["robust_delta_vs_balanced"] for case in scoped),
            6,
        ),
        "robust_improvements_over_0_5pct": sum(
            1 for case in scoped if case["improvement_over_0_5pct"]
        ),
        "robust_regressions_over_0_5pct": sum(
            1 for case in scoped if case["regression_over_0_5pct"]
        ),
        "mean_post_hoc_best_balanced_rank": round(
            mean(float(case["post_hoc_best_balanced_rank"] or 0) for case in scoped),
            2,
        ),
        "mean_post_hoc_best_robust_rank": round(
            mean(float(case["post_hoc_best_robust_rank"] or 0) for case in scoped),
            2,
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Copiale Multi-Page Selector Robustness",
        "",
        "Ground truth is used only after candidates exist, to compare selector policies.",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| files scanned | {report['file_count']} |",
        f"| evaluated sections | {report['case_count']} |",
        "",
        "## By Section",
        "",
        "| Section | Cases | Balanced Hits | Robust Hits | Balanced Gap | Robust Gap | Robust Delta | Robust +0.5pp | Robust -0.5pp | Best Balanced Rank | Best Robust Rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["section_summaries"]:
        if not row.get("case_count"):
            lines.append(f"| `{row['section']}` | 0 |  |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            f"| `{row['section']}` | {row['case_count']} | "
            f"{row['balanced_exact_hits']} | {row['robust_exact_hits']} | "
            f"{format_percent(row['mean_balanced_gap'])} | "
            f"{format_percent(row['mean_robust_gap'])} | "
            f"{format_percent(row['mean_robust_delta_vs_balanced'])} | "
            f"{row['robust_improvements_over_0_5pct']} | "
            f"{row['robust_regressions_over_0_5pct']} | "
            f"{row['mean_post_hoc_best_balanced_rank']:.2f} | "
            f"{row['mean_post_hoc_best_robust_rank']:.2f} |"
        )

    lines.extend([
        "",
        "## Cases",
        "",
        "| Section | Artifact | Candidates | Balanced | Robust | Best | Delta | Robust Rank Of Best | Robust Post-Hoc Rank |",
        "|---|---|---:|---|---|---|---:|---:|---:|",
    ])
    for case in sorted(report["cases"], key=lambda item: (item["section"], item["artifact"])):
        lines.append(
            f"| `{case['section']}` | `{case['artifact']}` | {case['candidate_count']} | "
            f"{winner_cell(case['balanced_winner'])} | "
            f"{winner_cell(case['robust_winner'])} | "
            f"{winner_cell(case['post_hoc_best'])} | "
            f"{format_percent(case['robust_delta_vs_balanced'])} | "
            f"{case['post_hoc_best_robust_rank']} | "
            f"{case['robust_winner_post_hoc_rank']} |"
        )
    if report.get("skipped"):
        lines.extend(["", "## Skipped", "", "| Path | Reason |", "|---|---|"])
        for item in report["skipped"]:
            lines.append(f"| `{item['path']}` | {item['reason']} |")
    return "\n".join(lines).rstrip() + "\n"


def discover_experiment_jsons(experiment_dir: Path) -> list[Path]:
    if not experiment_dir.exists():
        raise SystemExit(f"Experiment directory not found: {experiment_dir}")
    paths = []
    for path in sorted(experiment_dir.glob("copiale_multipage_*.json")):
        name = path.name
        if name.endswith(".artifact.json") or ".selector" in name:
            continue
        paths.append(path)
    return paths


def rank_of(rows: list[dict[str, Any]], label: str, *, key: str) -> int | None:
    for idx, row in enumerate(sort_rows(rows, key), start=1):
        if str(row.get("label") or "") == label:
            return idx
    return None


def char_score(row: dict[str, Any]) -> float:
    return float(row.get("post_hoc_char_avg") or 0.0)


def winner_cell(row: dict[str, Any]) -> str:
    label = row.get("label")
    mask = mask_label(row.get("mask") or [])
    char = format_percent(row.get("post_hoc_char_avg"))
    return f"`{label}` {char} `{mask}`"


def mean(values: Any) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    main()
