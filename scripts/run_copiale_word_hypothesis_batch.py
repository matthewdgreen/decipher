#!/usr/bin/env python3
"""Batch Copiale word-hypothesis repair probes over a finalist portfolio.

This is an orchestration and reporting helper. Child probes remain
ground-truth-free for generation/ranking; post-hoc character accuracy is only
read from child reports for calibration in the batch summary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Copiale word-hypothesis repair probes across portfolio labels."
    )
    parser.add_argument("experiment_json", help="JSON from run_copiale_multipage_experiment.py")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument(
        "--section",
        choices=["portfolio_local_repair", "portfolio_refinement", "elite_page_rerank"],
        default="portfolio_local_repair",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels to probe. Defaults to first --label-count rows.",
    )
    parser.add_argument("--label-count", type=int, default=12)
    parser.add_argument("--dictionary", default="resources/dictionaries/german_common.txt")
    parser.add_argument("--consensus-top-n", type=int, default=12)
    parser.add_argument("--consensus-min-agreement", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--window-step", type=int, default=40)
    parser.add_argument("--windows-per-page", type=int, default=5)
    parser.add_argument("--min-word-len", type=int, default=5)
    parser.add_argument("--max-word-len", type=int, default=14)
    parser.add_argument("--max-edits", type=int, default=3)
    parser.add_argument("--max-hypotheses", type=int, default=40)
    parser.add_argument("--max-hypotheses-per-window", type=int, default=6)
    parser.add_argument("--include-hypothesis-pairs", action="store_true")
    parser.add_argument("--pair-candidate-limit", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=120)
    parser.add_argument("--max-hypothesis-set-size", type=int, default=1)
    parser.add_argument("--combination-candidate-limit", type=int, default=32)
    parser.add_argument("--max-combinations", type=int, default=800)
    parser.add_argument("--max-combined-edits", type=int, default=6)
    parser.add_argument("--allow-stable-edits", action="store_true")
    parser.add_argument("--store-all-variants", action="store_true")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/language_quality/word_hypothesis_repair_batch",
        help="Directory for child reports and batch summary.",
    )
    parser.add_argument("--verbose", action="store_true", help="Stream child probe output.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment_path = resolve_path(Path(args.experiment_json))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    labels = selected_labels(experiment, section=args.section, explicit=args.labels, count=args.label_count)
    if not labels:
        raise SystemExit(f"No labels found in section {args.section!r}.")

    output_dir = resolve_path(Path(args.artifact_dir)) / experiment_path.stem / args.section
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, label in enumerate(labels, start=1):
        stem = safe_stem(f"{index:02d}_{label}")
        output = output_dir / f"{stem}.word_hypothesis_repair.md"
        json_output = output.with_suffix(".json")
        cmd = build_command(
            experiment_path=experiment_path,
            benchmark_root=args.benchmark_root,
            section=args.section,
            label=label,
            dictionary=args.dictionary,
            consensus_top_n=args.consensus_top_n,
            consensus_min_agreement=args.consensus_min_agreement,
            window_size=args.window_size,
            window_step=args.window_step,
            windows_per_page=args.windows_per_page,
            min_word_len=args.min_word_len,
            max_word_len=args.max_word_len,
            max_edits=args.max_edits,
            max_hypotheses=args.max_hypotheses,
            max_hypotheses_per_window=args.max_hypotheses_per_window,
            include_hypothesis_pairs=args.include_hypothesis_pairs,
            pair_candidate_limit=args.pair_candidate_limit,
            max_pairs=args.max_pairs,
            max_hypothesis_set_size=args.max_hypothesis_set_size,
            combination_candidate_limit=args.combination_candidate_limit,
            max_combinations=args.max_combinations,
            max_combined_edits=args.max_combined_edits,
            allow_stable_edits=args.allow_stable_edits,
            store_all_variants=args.store_all_variants,
            top_n=args.top_n,
            output=output,
            json_output=json_output,
        )
        if args.dry_run:
            print(" ".join(shell_quote(part) for part in cmd))
            status = "planned"
            returncode = None
            child_summary = {}
        else:
            print(f"[{index}/{len(labels)}] {label}", flush=True)
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                text=True,
                stdout=None if args.verbose else subprocess.PIPE,
                stderr=None if args.verbose else subprocess.PIPE,
            )
            status = "completed" if result.returncode == 0 else "error"
            returncode = result.returncode
            child_summary = summarize_child(json_output) if result.returncode == 0 else {}
            if result.returncode != 0:
                if not args.verbose:
                    print((result.stdout or "")[-4000:], file=sys.stderr)
                    print((result.stderr or "")[-4000:], file=sys.stderr)
                rows.append(manifest_row(label, status, returncode, output, json_output, cmd, child_summary))
                break
        rows.append(manifest_row(label, status, returncode, output, json_output, cmd, child_summary))

    manifest = {
        "experiment": "copiale_word_hypothesis_repair_batch",
        "experiment_json": str(experiment_path),
        "section": args.section,
        "label_count": len(labels),
        "dry_run": args.dry_run,
        "settings": {
            "max_hypotheses": args.max_hypotheses,
            "max_hypotheses_per_window": args.max_hypotheses_per_window,
            "include_hypothesis_pairs": args.include_hypothesis_pairs,
            "max_hypothesis_set_size": args.max_hypothesis_set_size,
            "combination_candidate_limit": args.combination_candidate_limit,
            "max_combinations": args.max_combinations,
            "max_edits": args.max_edits,
            "store_all_variants": args.store_all_variants,
        },
        "rows": rows,
    }
    summary = summarize_manifest(manifest)
    manifest_path = output_dir / "batch_manifest.json"
    summary_json = output_dir / "batch_summary.json"
    summary_md = output_dir / "batch_summary.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(render_summary(summary), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {summary_json}")


def build_command(
    *,
    experiment_path: Path,
    benchmark_root: str,
    section: str,
    label: str,
    dictionary: str,
    consensus_top_n: int,
    consensus_min_agreement: float,
    window_size: int,
    window_step: int,
    windows_per_page: int,
    min_word_len: int,
    max_word_len: int,
    max_edits: int,
    max_hypotheses: int,
    max_hypotheses_per_window: int,
    include_hypothesis_pairs: bool,
    pair_candidate_limit: int,
    max_pairs: int,
    max_hypothesis_set_size: int,
    combination_candidate_limit: int,
    max_combinations: int,
    max_combined_edits: int,
    allow_stable_edits: bool,
    store_all_variants: bool,
    top_n: int,
    output: Path,
    json_output: Path,
) -> list[str]:
    cmd = [
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
        "--consensus-top-n",
        str(consensus_top_n),
        "--consensus-min-agreement",
        str(consensus_min_agreement),
        "--window-size",
        str(window_size),
        "--window-step",
        str(window_step),
        "--windows-per-page",
        str(windows_per_page),
        "--min-word-len",
        str(min_word_len),
        "--max-word-len",
        str(max_word_len),
        "--max-edits",
        str(max_edits),
        "--max-hypotheses",
        str(max_hypotheses),
        "--max-hypotheses-per-window",
        str(max_hypotheses_per_window),
        "--pair-candidate-limit",
        str(pair_candidate_limit),
        "--max-pairs",
        str(max_pairs),
        "--max-hypothesis-set-size",
        str(max_hypothesis_set_size),
        "--combination-candidate-limit",
        str(combination_candidate_limit),
        "--max-combinations",
        str(max_combinations),
        "--max-combined-edits",
        str(max_combined_edits),
        "--top-n",
        str(top_n),
        "--output",
        str(output),
        "--json-output",
        str(json_output),
    ]
    if include_hypothesis_pairs:
        cmd.append("--include-hypothesis-pairs")
    if allow_stable_edits:
        cmd.append("--allow-stable-edits")
    if store_all_variants:
        cmd.append("--store-all-variants")
    return cmd


def summarize_child(path: Path) -> dict[str, Any]:
    child = json.loads(path.read_text(encoding="utf-8"))
    candidates = collect_candidates(child)
    candidates_by_post_hoc = sorted(
        candidates,
        key=lambda row: metric(row, "post_hoc_char_avg"),
        reverse=True,
    )
    candidates_by_adjudication = sorted(
        candidates,
        key=lambda row: metric(row, "adjudication_score"),
        reverse=True,
    )
    best_runtime = first_candidate(child.get("top_variants"), "runtime")
    best_word = first_candidate(child.get("top_variants_by_word_hypothesis"), "word_hypothesis")
    best_adjudication = first_candidate(child.get("top_variants_by_adjudication"), "adjudication")
    best_combination = first_candidate(
        child.get("top_combination_variants_by_adjudication"), "combination_adjudication"
    )
    best_marginal_combination = first_candidate(
        child.get("top_combination_variants_by_marginal"), "combination_marginal"
    )
    best_post_hoc = first_candidate(child.get("top_variants_by_post_hoc"), "post_hoc")
    baseline = compact_candidate(child.get("baseline") or {}, "baseline")
    return {
        "label": child.get("label"),
        "hypothesis_count": child.get("hypothesis_count"),
        "variant_count": child.get("variant_count"),
        "accepted_variant_count": child.get("accepted_variant_count"),
        "baseline": baseline,
        "best_runtime": best_runtime,
        "best_word_hypothesis": best_word,
        "best_adjudication": best_adjudication,
        "best_combination": best_combination,
        "best_marginal_combination": best_marginal_combination,
        "best_post_hoc": best_post_hoc,
        "adjudication_pick_post_hoc_rank": candidate_rank(best_adjudication, candidates_by_post_hoc),
        "combination_pick_post_hoc_rank": candidate_rank(best_combination, candidates_by_post_hoc),
        "marginal_combination_pick_post_hoc_rank": candidate_rank(
            best_marginal_combination, candidates_by_post_hoc
        ),
        "post_hoc_pick_adjudication_rank": candidate_rank(best_post_hoc, candidates_by_adjudication),
        "top_adjudication": [
            compact_candidate(row, "adjudication")
            for row in (child.get("top_variants_by_adjudication") or [])[:8]
        ],
        "top_word_hypotheses": child.get("top_word_hypotheses") or [],
    }


def collect_candidates(child: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for section, source in [
        ("top_variants", "runtime"),
        ("top_variants_by_word_hypothesis", "word_hypothesis"),
        ("top_variants_by_adjudication", "adjudication"),
        ("top_combination_variants_by_adjudication", "combination_adjudication"),
        ("top_combination_variants_by_marginal", "combination_marginal"),
        ("top_variants_by_post_hoc", "post_hoc"),
    ]:
        for row in child.get(section) or []:
            rows.append(compact_candidate(row, source))
    return dedupe_candidates(rows)


def first_candidate(rows: Any, source: str) -> dict[str, Any]:
    if isinstance(rows, list) and rows:
        return compact_candidate(rows[0], source)
    return {}


def compact_candidate(row: Any, source: str) -> dict[str, Any]:
    row = row if isinstance(row, dict) else {}
    adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    acceptance = row.get("repair_acceptance") if isinstance(row.get("repair_acceptance"), dict) else {}
    return {
        "source": source,
        "edits": [str(item) for item in (row.get("edits") or [])],
        "hypotheses": [
            f"{item.get('observed')}->{item.get('target')}"
            for item in (row.get("word_hypotheses") or [])
            if isinstance(item, dict)
        ],
        "word_hypothesis_score": number(row.get("word_hypothesis_score")),
        "adjudication_score": number(adjudication.get("adjudication_score")),
        "adjudication_no_target_score": number(adjudication.get("adjudication_no_target_score")),
        "target_leverage_score": number(adjudication.get("target_leverage_score")),
        "target_word_gain_sum": number(adjudication.get("target_word_gain_sum")),
        "collateral_word_gain_sum": number(adjudication.get("collateral_word_gain_sum")),
        "collateral_word_damage_sum": number(adjudication.get("collateral_word_damage_sum")),
        "word_damaged_occurrences": adjudication.get("word_damaged_occurrences"),
        "marginal_selector_score": number(
            (row.get("marginal_contribution") or {}).get("marginal_selector_score")
            if isinstance(row.get("marginal_contribution"), dict)
            else None
        ),
        "runtime_decision": acceptance.get("decision"),
        "page_robust_score": number(row.get("page_robust_score")),
        "page_validation_avg": number(row.get("page_validation_avg")),
        "page_language_quality_avg": number(row.get("page_language_quality_avg")),
        "post_hoc_char_avg": number(row.get("post_hoc_char_avg")),
        "preview": str(row.get("preview") or "")[:180],
    }


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("rows") if isinstance(manifest.get("rows"), list) else []
    recurring_hypotheses: Counter[str] = Counter()
    recurring_edits: Counter[str] = Counter()
    completed = []
    for row in rows:
        summary = row.get("child_summary") if isinstance(row.get("child_summary"), dict) else {}
        if row.get("status") == "completed":
            completed.append(row)
        for candidate in summary.get("top_adjudication") or []:
            for hypothesis in candidate.get("hypotheses") or []:
                recurring_hypotheses[str(hypothesis)] += 1
            for edit in candidate.get("edits") or []:
                if edit != "baseline":
                    recurring_edits[str(edit)] += 1

    detail_rows = []
    for row in rows:
        summary = row.get("child_summary") if isinstance(row.get("child_summary"), dict) else {}
        baseline = summary.get("baseline") or {}
        best_adj = summary.get("best_adjudication") or {}
        best_post = summary.get("best_post_hoc") or {}
        detail_rows.append({
            "label": row.get("label"),
            "status": row.get("status"),
            "report": row.get("output"),
            "json": row.get("json_output"),
            "hypothesis_count": summary.get("hypothesis_count"),
            "variant_count": summary.get("variant_count"),
            "baseline_char": baseline.get("post_hoc_char_avg"),
            "runtime_pick": summary.get("best_runtime") or {},
            "word_pick": summary.get("best_word_hypothesis") or {},
            "adjudication_pick": best_adj,
            "combination_pick": summary.get("best_combination") or {},
            "marginal_combination_pick": summary.get("best_marginal_combination") or {},
            "post_hoc_pick": best_post,
            "adjudication_pick_post_hoc_rank": summary.get("adjudication_pick_post_hoc_rank"),
            "combination_pick_post_hoc_rank": summary.get("combination_pick_post_hoc_rank"),
            "post_hoc_pick_adjudication_rank": summary.get("post_hoc_pick_adjudication_rank"),
            "adjudication_gap_to_post_hoc": subtract_metric(
                best_post, best_adj, "post_hoc_char_avg"
            ),
        })
    return {
        "manifest": manifest,
        "aggregate": {
            "labels": len(rows),
            "completed": len(completed),
            "best_adjudication_char": best_by_metric(
                [row["adjudication_pick"] for row in detail_rows], "post_hoc_char_avg"
            ),
            "best_post_hoc_char": best_by_metric(
                [row["post_hoc_pick"] for row in detail_rows], "post_hoc_char_avg"
            ),
            "recurring_hypotheses": counter_rows(recurring_hypotheses),
            "recurring_edits": counter_rows(recurring_edits),
        },
        "labels_detail": detail_rows,
    }


def render_summary(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Copiale Word-Hypothesis Repair Batch",
        "",
        "Generation and ranking are ground-truth-free. Character accuracy columns are post-hoc calibration only.",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| labels | {aggregate['labels']} |",
        f"| completed | {aggregate['completed']} |",
        f"| best adjudication-pick char | {fmt_candidate(aggregate.get('best_adjudication_char'))} |",
        f"| best post-hoc char | {fmt_candidate(aggregate.get('best_post_hoc_char'))} |",
        "",
        "## Labels",
        "",
        "| Label | Hyp | Var | Baseline Char | Runtime Pick | Word Pick | Adjudication Pick | Best Combo | Marginal Combo | Post-Hoc Best | Adj Rank By Char | Post-Hoc Rank By Adj | Gap | Report |",
        "|---|---:|---:|---:|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary["labels_detail"]:
        lines.append(
            f"| `{row['label']}` | {row.get('hypothesis_count') or ''} | {row.get('variant_count') or ''} | "
            f"{fmt_pct(row.get('baseline_char'))} | "
            f"{candidate_cell(row.get('runtime_pick'))} | "
            f"{candidate_cell(row.get('word_pick'))} | "
            f"{candidate_cell(row.get('adjudication_pick'))} | "
            f"{candidate_cell(row.get('combination_pick'))} | "
            f"{candidate_cell(row.get('marginal_combination_pick'))} | "
            f"{candidate_cell(row.get('post_hoc_pick'))} | "
            f"{row.get('adjudication_pick_post_hoc_rank') or ''} | "
            f"{row.get('post_hoc_pick_adjudication_rank') or ''} | "
            f"{fmt_pct(row.get('adjudication_gap_to_post_hoc'))} | "
            f"`{row.get('report') or ''}` |"
        )
    lines.extend([
        "",
        "## Recurring Repair Hypotheses",
        "",
        "| Hypothesis | Count |",
        "|---|---:|",
    ])
    for item in aggregate["recurring_hypotheses"][:20]:
        lines.append(f"| `{escape_cell(item['item'])}` | {item['count']} |")
    lines.extend([
        "",
        "## Recurring Edits",
        "",
        "| Edit | Count |",
        "|---|---:|",
    ])
    for item in aggregate["recurring_edits"][:20]:
        lines.append(f"| `{escape_cell(item['item'])}` | {item['count']} |")
    return "\n".join(lines).rstrip() + "\n"


def selected_labels(payload: dict[str, Any], *, section: str, explicit: str, count: int) -> list[str]:
    if explicit.strip():
        return [item.strip() for item in explicit.split(",") if item.strip()]
    block = payload.get(section) if isinstance(payload.get(section), dict) else {}
    labels = []
    for row in block.get("rows") or []:
        if isinstance(row, dict) and row.get("label"):
            label = str(row["label"])
            if label not in labels:
                labels.append(label)
        if len(labels) >= max(1, count):
            break
    return labels


def manifest_row(
    label: str,
    status: str,
    returncode: int | None,
    output: Path,
    json_output: Path,
    command: list[str],
    child_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": label,
        "status": status,
        "returncode": returncode,
        "output": str(output),
        "json_output": str(json_output),
        "command": command,
        "child_summary": child_summary,
    }


def dedupe_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = variant_key(row)
        old = best.get(key)
        if old is None or metric(row, "post_hoc_char_avg") > metric(old, "post_hoc_char_avg"):
            best[key] = row
    return list(best.values())


def candidate_rank(candidate: dict[str, Any], ranked_rows: list[dict[str, Any]]) -> int | None:
    if not candidate:
        return None
    key = variant_key(candidate)
    for index, row in enumerate(ranked_rows, start=1):
        if variant_key(row) == key:
            return index
    return None


def variant_key(row: dict[str, Any]) -> str:
    edits = row.get("edits") or []
    return "|".join(str(item) for item in edits)


def best_by_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rows = [row for row in rows if row]
    return max(rows, key=lambda row: metric(row, key), default={})


def metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key) if isinstance(row, dict) else None
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def subtract_metric(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    if not left or not right:
        return None
    left_value = left.get(key)
    right_value = right.get(key)
    if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
        return float(left_value) - float(right_value)
    return None


def counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"item": item, "count": count}
        for item, count in counter.most_common()
    ]


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def fmt_candidate(row: Any) -> str:
    if not isinstance(row, dict) or not row:
        return ""
    return f"{fmt_pct(row.get('post_hoc_char_avg'))} `{escape_cell(', '.join(row.get('edits') or []))}`"


def candidate_cell(row: Any) -> str:
    if not isinstance(row, dict) or not row:
        return ""
    hypotheses = ", ".join(row.get("hypotheses") or [])
    edits = ", ".join(row.get("edits") or [])
    return (
        f"{fmt_pct(row.get('post_hoc_char_avg'))} "
        f"`{escape_cell(edits)}`"
        f"<br>{escape_cell(hypotheses[:80])}"
        f"<br>adj={fmt_float(row.get('adjudication_score'))}"
        f" noTarget={fmt_float(row.get('adjudication_no_target_score'))}"
        f" lev={fmt_float(row.get('target_leverage_score'))}"
        f" marg={fmt_float(row.get('marginal_selector_score'))}"
    )


def fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return ""


def fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return ""


def escape_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "/._-+=" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
