#!/usr/bin/env python3
"""Run Copiale global-repair probes for several portfolio labels.

This is an orchestration helper. It does not score or choose candidates on its
own; it repeatedly invokes ``probe_copiale_multipage_global_repair.py`` with
consistent settings and writes a small manifest of the child reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch Copiale multi-page global repair probes over section labels."
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
        help="Comma-separated labels to probe. Defaults to the first --label-count section rows.",
    )
    parser.add_argument("--label-count", type=int, default=5)
    parser.add_argument("--language-quality-ranker", default="")
    parser.add_argument("--ranker-review-top-k", type=int, default=8)
    parser.add_argument("--ranker-family-top-k", type=int, default=3)
    parser.add_argument("--consensus-top-n", type=int, default=12)
    parser.add_argument("--consensus-min-agreement", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--window-step", type=int, default=40)
    parser.add_argument("--windows-per-page", type=int, default=5)
    parser.add_argument("--max-symbols", type=int, default=10)
    parser.add_argument("--max-alternatives", type=int, default=4)
    parser.add_argument("--include-pairs", action="store_true")
    parser.add_argument("--pair-candidate-limit", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=120)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/copiale_global_repair_batch",
        help="Directory for child markdown/json reports and batch manifest.",
    )
    parser.add_argument("--verbose", action="store_true", help="Stream child probe output.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
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
        output = output_dir / f"{stem}.global_repair.md"
        json_output = output.with_suffix(".json")
        cmd = build_command(
            experiment_path=experiment_path,
            benchmark_root=args.benchmark_root,
            section=args.section,
            label=label,
            language_quality_ranker=args.language_quality_ranker,
            ranker_review_top_k=args.ranker_review_top_k,
            ranker_family_top_k=args.ranker_family_top_k,
            consensus_top_n=args.consensus_top_n,
            consensus_min_agreement=args.consensus_min_agreement,
            window_size=args.window_size,
            window_step=args.window_step,
            windows_per_page=args.windows_per_page,
            max_symbols=args.max_symbols,
            max_alternatives=args.max_alternatives,
            include_pairs=args.include_pairs,
            pair_candidate_limit=args.pair_candidate_limit,
            max_pairs=args.max_pairs,
            top_n=args.top_n,
            output=output,
            json_output=json_output,
        )
        if args.dry_run:
            print(" ".join(shell_quote(part) for part in cmd))
            status = "planned"
            returncode = None
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
            child_summary = child_report_summary(json_output) if result.returncode == 0 else {}
            if result.returncode != 0:
                if not args.verbose:
                    print((result.stdout or "")[-4000:], file=sys.stderr)
                    print((result.stderr or "")[-4000:], file=sys.stderr)
                rows.append({
                    "label": label,
                    "status": status,
                    "returncode": returncode,
                    "output": str(output),
                    "json_output": str(json_output),
                    "command": cmd,
                    "child_summary": {},
                })
                break
        rows.append({
            "label": label,
            "status": status,
            "returncode": returncode,
            "output": str(output),
            "json_output": str(json_output),
            "command": cmd,
            "child_summary": child_summary if not args.dry_run else {},
        })

    manifest = {
        "experiment_json": str(experiment_path),
        "section": args.section,
        "label_count": len(labels),
        "dry_run": args.dry_run,
        "language_quality_ranker": args.language_quality_ranker,
        "rows": rows,
    }
    manifest_path = output_dir / "batch_manifest.json"
    manifest_md = output_dir / "batch_manifest.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_md.write_text(render_manifest(manifest), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {manifest_md}")


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
        if len(labels) >= max(1, int(count)):
            break
    return labels


def build_command(
    *,
    experiment_path: Path,
    benchmark_root: str,
    section: str,
    label: str,
    language_quality_ranker: str,
    ranker_review_top_k: int,
    ranker_family_top_k: int,
    consensus_top_n: int,
    consensus_min_agreement: float,
    window_size: int,
    window_step: int,
    windows_per_page: int,
    max_symbols: int,
    max_alternatives: int,
    include_pairs: bool,
    pair_candidate_limit: int,
    max_pairs: int,
    top_n: int,
    output: Path,
    json_output: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "probe_copiale_multipage_global_repair.py"),
        str(experiment_path),
        "--benchmark-root",
        benchmark_root,
        "--section",
        section,
        "--label",
        label,
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
        "--max-symbols",
        str(max_symbols),
        "--max-alternatives",
        str(max_alternatives),
        "--pair-candidate-limit",
        str(pair_candidate_limit),
        "--max-pairs",
        str(max_pairs),
        "--top-n",
        str(top_n),
        "--output",
        str(output),
        "--json-output",
        str(json_output),
    ]
    if include_pairs:
        cmd.append("--include-pairs")
    if language_quality_ranker:
        cmd.extend([
            "--language-quality-ranker",
            language_quality_ranker,
            "--ranker-review-top-k",
            str(ranker_review_top_k),
            "--ranker-family-top-k",
            str(ranker_family_top_k),
        ])
    return cmd


def render_manifest(payload: dict[str, Any]) -> str:
    lines = [
        "# Copiale Global Repair Batch",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| experiment | `{payload['experiment_json']}` |",
        f"| section | `{payload['section']}` |",
        f"| labels | {payload['label_count']} |",
        f"| dry run | {1 if payload.get('dry_run') else 0} |",
        f"| language-quality ranker | `{payload.get('language_quality_ranker') or ''}` |",
        "",
        "| Label | Status | Ranker Pick | Robust Pick | Report | JSON |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        summary = row.get("child_summary") if isinstance(row.get("child_summary"), dict) else {}
        lines.append(
            f"| `{row['label']}` | {row['status']} | "
            f"{summary_cell(summary.get('ranker_pick'))} | "
            f"{summary_cell(summary.get('robust_pick'))} | "
            f"`{row['output']}` | `{row['json_output']}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def child_report_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    top_variants = payload.get("top_variants") if isinstance(payload.get("top_variants"), list) else []
    ranker = payload.get("language_quality_ranker") if isinstance(payload.get("language_quality_ranker"), dict) else {}
    shortlist = ranker.get("diverse_review_shortlist") if isinstance(ranker.get("diverse_review_shortlist"), list) else []
    robust_pick = top_variants[0] if top_variants else {}
    ranker_pick = shortlist[0] if shortlist else {}
    return {
        "ranker_pick": compact_child_row(ranker_pick),
        "robust_pick": compact_child_row(robust_pick),
    }


def compact_child_row(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "edits": row.get("edits") or [],
        "mask": row.get("mask") or [],
        "language_quality_rank_score": row.get("language_quality_rank_score"),
        "page_robust_score": row.get("page_robust_score"),
        "post_hoc_char_avg": row.get("post_hoc_char_avg"),
    }


def summary_cell(row: Any) -> str:
    if not isinstance(row, dict) or not row:
        return ""
    edits = "; ".join(str(item) for item in (row.get("edits") or []))
    mask = ",".join(str(item) for item in (row.get("mask") or []))
    lq = format_float(row.get("language_quality_rank_score"))
    robust = format_float(row.get("page_robust_score"))
    char = format_percent(row.get("post_hoc_char_avg"))
    return f"{edits}<br>`{mask}`<br>LQ {lq}, R {robust}, char {char}"


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "/._:=+-" for ch in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
