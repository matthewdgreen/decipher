#!/usr/bin/env python3
"""Run controlled breadth curves for Copiale word-hypothesis repair.

This is an experiment harness. Child probes generate and rank candidates
without ground truth; this script reads their completed artifacts and reports
post-hoc calibration ranks so we can see whether more breadth produces better
repair basins and whether runtime rankers can find them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIGS: dict[str, dict[str, int]] = {
    "small": {
        "max_hypotheses": 24,
        "max_hypotheses_per_window": 6,
        "combination_candidate_limit": 16,
        "max_combinations": 80,
        "max_hypothesis_set_size": 2,
        "max_combined_edits": 6,
    },
    "medium": {
        "max_hypotheses": 40,
        "max_hypotheses_per_window": 8,
        "combination_candidate_limit": 24,
        "max_combinations": 250,
        "max_hypothesis_set_size": 2,
        "max_combined_edits": 6,
    },
    "wide": {
        "max_hypotheses": 64,
        "max_hypotheses_per_window": 10,
        "combination_candidate_limit": 32,
        "max_combinations": 500,
        "max_hypothesis_set_size": 2,
        "max_combined_edits": 6,
    },
    "triple_smoke": {
        "max_hypotheses": 48,
        "max_hypotheses_per_window": 8,
        "combination_candidate_limit": 20,
        "max_combinations": 250,
        "max_hypothesis_set_size": 3,
        "max_combined_edits": 7,
    },
}


RANKERS: list[tuple[str, str]] = [
    ("adjudication", "adjudication_score"),
    ("no_target", "adjudication_no_target_score"),
    ("leverage", "target_leverage_score"),
    ("marginal", "marginal_selector_score"),
    ("robust", "page_robust_score"),
    ("validation", "page_validation_avg"),
    ("language_quality", "page_language_quality_avg"),
    ("word_hypothesis", "word_hypothesis_score"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Copiale word-repair breadth curves.")
    parser.add_argument("experiment_json", help="JSON from run_copiale_multipage_experiment.py")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--section", default="portfolio_local_repair")
    parser.add_argument("--labels", default="top9,top6")
    parser.add_argument(
        "--configs",
        default="small,medium",
        help=f"Comma-separated config names: {', '.join(CONFIGS)}",
    )
    parser.add_argument("--dictionary", default="resources/dictionaries/german_common.txt")
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/language_quality/word_repair_breadth_curve",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment_path = resolve_path(Path(args.experiment_json))
    labels = parse_csv(args.labels)
    configs = parse_configs(args.configs)
    output_dir = resolve_path(Path(args.artifact_dir)) / experiment_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label in labels:
        for config_name in configs:
            config = CONFIGS[config_name]
            stem = safe_stem(f"{label}_{config_name}")
            output = output_dir / f"{stem}.word_hypothesis_repair.md"
            json_output = output.with_suffix(".json")
            cmd = build_command(
                experiment_path=experiment_path,
                benchmark_root=args.benchmark_root,
                section=args.section,
                label=label,
                dictionary=args.dictionary,
                config=config,
                top_n=args.top_n,
                output=output,
                json_output=json_output,
            )
            print(f"[{len(rows) + 1}/{len(labels) * len(configs)}] {label} / {config_name}", flush=True)
            started = time.monotonic()
            if args.dry_run:
                print(" ".join(shell_quote(part) for part in cmd))
                result_code = 0
            else:
                result = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=None if args.verbose else subprocess.PIPE,
                    stderr=None if args.verbose else subprocess.PIPE,
                )
                result_code = result.returncode
                if result.returncode != 0:
                    if not args.verbose:
                        print((result.stdout or "")[-4000:], file=sys.stderr)
                        print((result.stderr or "")[-4000:], file=sys.stderr)
                    raise SystemExit(result.returncode)
            elapsed = time.monotonic() - started
            summary = summarize_probe(json_output) if not args.dry_run else {}
            row = {
                "label": label,
                "config": config_name,
                "settings": config,
                "elapsed_seconds": round(elapsed, 3),
                "returncode": result_code,
                "output": str(output),
                "json_output": str(json_output),
                "summary": summary,
            }
            rows.append(row)

    payload = {
        "experiment": "copiale_word_repair_breadth_curve",
        "source_experiment": str(experiment_path),
        "section": args.section,
        "labels": labels,
        "configs": configs,
        "rows": rows,
    }
    json_path = output_dir / "breadth_curve_summary.json"
    md_path = output_dir / "breadth_curve_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def build_command(
    *,
    experiment_path: Path,
    benchmark_root: str,
    section: str,
    label: str,
    dictionary: str,
    config: dict[str, int],
    top_n: int,
    output: Path,
    json_output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "probe_copiale_word_hypothesis_repair.py"),
        str(experiment_path),
        "--benchmark-root",
        benchmark_root,
        "--section",
        section,
        "--label",
        label,
        "--dictionary",
        dictionary,
        "--max-hypotheses",
        str(config["max_hypotheses"]),
        "--max-hypotheses-per-window",
        str(config["max_hypotheses_per_window"]),
        "--max-hypothesis-set-size",
        str(config["max_hypothesis_set_size"]),
        "--combination-candidate-limit",
        str(config["combination_candidate_limit"]),
        "--max-combinations",
        str(config["max_combinations"]),
        "--max-combined-edits",
        str(config["max_combined_edits"]),
        "--store-all-variants",
        "--top-n",
        str(top_n),
        "--output",
        str(output),
        "--json-output",
        str(json_output),
    ]


def summarize_probe(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = collect_variants(payload)
    baseline = normalize_variant(payload.get("baseline") or {})
    best_gt = max(variants, key=lambda row: metric(row, "post_hoc_char_avg"), default={})
    ranker_picks = {}
    for name, key in RANKERS:
        ranked = sorted(variants, key=lambda row: metric(row, key), reverse=True)
        pick = ranked[0] if ranked else {}
        ranker_picks[name] = {
            "metric": key,
            "pick": compact_pick(pick),
            "best_gt_rank": variant_rank(best_gt, ranked),
            "best_gt_top3": in_top_n(best_gt, ranked, 3),
            "best_gt_top10": in_top_n(best_gt, ranked, 10),
        }
    return {
        "hypothesis_count": payload.get("hypothesis_count"),
        "variant_count": payload.get("variant_count"),
        "baseline": compact_pick(baseline),
        "best_gt": compact_pick(best_gt),
        "best_gt_delta_vs_baseline": delta(best_gt, baseline, "post_hoc_char_avg"),
        "ranker_picks": ranker_picks,
    }


def collect_variants(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("all_variants") or []:
        rows.append(normalize_variant(row))
    if not rows:
        for section in [
            "top_variants",
            "top_variants_by_word_hypothesis",
            "top_variants_by_adjudication",
            "top_combination_variants_by_adjudication",
            "top_combination_variants_by_marginal",
            "top_variants_by_post_hoc",
        ]:
            for row in payload.get(section) or []:
                rows.append(normalize_variant(row))
    return dedupe(rows)


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
        if key not in normalized:
            normalized[key] = adjudication.get(key)
    if "marginal_selector_score" not in normalized:
        normalized["marginal_selector_score"] = marginal.get("marginal_selector_score")
    return normalized


def compact_pick(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "edits": row.get("edits") or [],
        "corrections": [
            f"{item.get('observed')}->{item.get('target')}"
            for item in (row.get("word_hypotheses") or [])
            if isinstance(item, dict)
        ],
        "post_hoc_char_avg": row.get("post_hoc_char_avg"),
        "post_hoc_char_no_target_baseline_avg": row.get("post_hoc_char_no_target_baseline_avg"),
        "post_hoc_char_target_baseline_avg": row.get("post_hoc_char_target_baseline_avg"),
        "post_hoc_char_no_target_avg": row.get("post_hoc_char_no_target_avg"),
        "post_hoc_char_target_avg": row.get("post_hoc_char_target_avg"),
        "adjudication_score": row.get("adjudication_score"),
        "adjudication_no_target_score": row.get("adjudication_no_target_score"),
        "target_leverage_score": row.get("target_leverage_score"),
        "marginal_selector_score": row.get("marginal_selector_score"),
        "page_robust_score": row.get("page_robust_score"),
        "page_validation_avg": row.get("page_validation_avg"),
        "page_language_quality_avg": row.get("page_language_quality_avg"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Copiale Word-Repair Breadth Curve",
        "",
        "Child probes are ground-truth-free. Character accuracy and best-GT ranks are post-hoc calibration.",
        "",
        "| Label | Config | Hyp | Var | Seconds | Baseline Char | Best Char | Best GT NoTarget | Delta | Best Edits | Adj Pick | Adj Rank | LQ Pick | LQ Rank | NoTarget Pick | NoTarget Rank | Marg Pick | Marg Rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---:|---|---:|---|---:|",
    ]
    for row in payload["rows"]:
        summary = row.get("summary") or {}
        baseline = summary.get("baseline") or {}
        best = summary.get("best_gt") or {}
        rankers = summary.get("ranker_picks") or {}
        lines.append(
            f"| `{row['label']}` | `{row['config']}` | "
            f"{summary.get('hypothesis_count') or ''} | {summary.get('variant_count') or ''} | "
            f"{row.get('elapsed_seconds') or 0:.1f} | "
            f"{fmt_pct(baseline.get('post_hoc_char_avg'))} | "
            f"{fmt_pct(best.get('post_hoc_char_avg'))} | "
            f"{fmt_pct(best.get('post_hoc_char_no_target_avg'))} | "
            f"{fmt_pct(summary.get('best_gt_delta_vs_baseline'))} | "
            f"{pick_cell(best)} | "
            f"{pick_cell((rankers.get('adjudication') or {}).get('pick') or {})} | "
            f"{(rankers.get('adjudication') or {}).get('best_gt_rank') or ''} | "
            f"{pick_cell((rankers.get('language_quality') or {}).get('pick') or {})} | "
            f"{(rankers.get('language_quality') or {}).get('best_gt_rank') or ''} | "
            f"{pick_cell((rankers.get('no_target') or {}).get('pick') or {})} | "
            f"{(rankers.get('no_target') or {}).get('best_gt_rank') or ''} | "
            f"{pick_cell((rankers.get('marginal') or {}).get('pick') or {})} | "
            f"{(rankers.get('marginal') or {}).get('best_gt_rank') or ''} |"
        )
    lines.extend([
        "",
        "## Ranker Detail",
        "",
    ])
    for row in payload["rows"]:
        summary = row.get("summary") or {}
        lines.extend([
            f"### {row['label']} / {row['config']}",
            "",
            "| Ranker | Pick Char | Best-GT Rank | Top-3 | Top-10 | Pick Edits |",
            "|---|---:|---:|---|---|---|",
        ])
        for name, _key in RANKERS:
            ranker = (summary.get("ranker_picks") or {}).get(name) or {}
            pick = ranker.get("pick") or {}
            lines.append(
                f"| `{name}` | {fmt_pct(pick.get('post_hoc_char_avg'))} | "
                f"{ranker.get('best_gt_rank') or ''} | "
                f"{yes_no(ranker.get('best_gt_top3'))} | {yes_no(ranker.get('best_gt_top10'))} | "
                f"{pick_cell(pick)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = {}
    for row in rows:
        key = "|".join(str(item) for item in (row.get("edits") or []))
        old = best.get(key)
        if old is None or metric(row, "post_hoc_char_avg") > metric(old, "post_hoc_char_avg"):
            best[key] = row
    return list(best.values())


def metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key) if isinstance(row, dict) else None
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def delta(row: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    left = row.get(key) if isinstance(row, dict) else None
    right = baseline.get(key) if isinstance(baseline, dict) else None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) - float(right)
    return None


def variant_rank(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> int | None:
    if not candidate:
        return None
    key = "|".join(str(item) for item in (candidate.get("edits") or []))
    for index, row in enumerate(rows, start=1):
        if "|".join(str(item) for item in (row.get("edits") or [])) == key:
            return index
    return None


def in_top_n(candidate: dict[str, Any], rows: list[dict[str, Any]], n: int) -> bool:
    rank = variant_rank(candidate, rows)
    return rank is not None and rank <= n


def pick_cell(row: dict[str, Any]) -> str:
    if not row:
        return ""
    corrections = ", ".join(row.get("corrections") or [])
    edits = ", ".join(str(item) for item in (row.get("edits") or []))
    return f"{fmt_pct(row.get('post_hoc_char_avg'))} `{escape(edits[:80])}`<br>{escape(corrections[:80])}"


def fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return ""


def yes_no(value: Any) -> str:
    return "Y" if value else "n"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_configs(value: str) -> list[str]:
    configs = parse_csv(value)
    unknown = [item for item in configs if item not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config(s): {', '.join(unknown)}")
    return configs


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "/._-+=" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
