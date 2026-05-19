#!/usr/bin/env python3
"""Run a leave-one-page-out Copiale candidate-ranker sweep.

This script is an evaluation harness, not a solver primitive. It trains one
candidate-only language-quality model per held-out Copiale page from prior
solver finalist artifacts, then compares the default null-mask validator
against the trained ranker on fresh automated runs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_IDS = (
    "copiale_single_B_copiale_p017",
    "copiale_single_B_copiale_p035",
    "copiale_single_B_copiale_p052",
    "copiale_single_B_copiale_p068",
    "copiale_single_B_copiale_p084",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate held-out Copiale candidate-only rankers."
    )
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument("--training-artifact", action="append", default=[
        "artifacts/copiale_evidence_packet/automated_null_masks_candidate_blend",
    ])
    parser.add_argument("--test-id", action="append", default=[])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--l2", type=float, default=0.5)
    parser.add_argument("--training-objective", choices=("regression", "pairwise"), default="pairwise")
    parser.add_argument("--feature-set", choices=("all", "no_solver", "text_only"), default="all")
    parser.add_argument("--min-label-delta", type=float, default=0.02)
    parser.add_argument("--max-pairs-per-group", type=int, default=2000)
    parser.add_argument("--nonnegative-weights", action="store_true")
    parser.add_argument("--nonnegative-iterations", type=int, default=4000)
    parser.add_argument("--nonnegative-learning-rate", type=float, default=0.01)
    parser.add_argument("--skip-default", action="store_true")
    parser.add_argument("--skip-language-quality", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    test_ids = tuple(args.test_id or DEFAULT_TEST_IDS)
    output_dir = (
        resolve_path(Path(args.output_dir))
        if args.output_dir
        else REPO_ROOT / "artifacts" / "copiale_candidate_ranker_sweep" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    models_dir = output_dir / "models"
    default_dir = output_dir / "default_validation"
    lq_dir = output_dir / "language_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for test_id in test_ids:
        print(f"\n=== {test_id} ===", flush=True)
        model_path = models_dir / f"{test_id}_candidate_only.json"
        report_path = models_dir / f"{test_id}_candidate_only.md"
        train_cmd = [
            python_bin(),
            "scripts/train_language_quality_scorer.py",
            "--language", "de",
            "--candidate-only",
            "--holdout-group", test_id,
            "--leave-one-group-out",
            "--output", str(model_path),
            "--report", str(report_path),
            "--min-chars", str(args.min_chars),
            "--l2", str(args.l2),
            "--objective", args.training_objective,
            "--feature-set", args.feature_set,
            "--min-label-delta", str(args.min_label_delta),
            "--max-pairs-per-group", str(args.max_pairs_per_group),
        ]
        if args.nonnegative_weights:
            train_cmd.append("--nonnegative-weights")
            train_cmd.extend([
                "--nonnegative-iterations", str(args.nonnegative_iterations),
                "--nonnegative-learning-rate", str(args.nonnegative_learning_rate),
            ])
        for artifact in args.training_artifact:
            train_cmd.extend(["--artifact", artifact])
        run_command(train_cmd, dry_run=args.dry_run)

        default_result = None
        if not args.skip_default:
            default_result = run_decipher_case(
                test_id,
                artifact_dir=default_dir,
                benchmark_root=args.benchmark_root,
                split=args.split,
                dry_run=args.dry_run,
            )
        lq_result = None
        if not args.skip_language_quality:
            lq_result = run_decipher_case(
                test_id,
                artifact_dir=lq_dir,
                benchmark_root=args.benchmark_root,
                split=args.split,
                model_path=model_path,
                dry_run=args.dry_run,
            )
        rows.append({
            "test_id": test_id,
            "model_path": str(model_path),
            "model_report": str(report_path),
            "default": default_result,
            "language_quality": lq_result,
            "delta_char_accuracy": (
                round(
                    float((lq_result or {}).get("char_accuracy") or 0.0)
                    - float((default_result or {}).get("char_accuracy") or 0.0),
                    6,
                )
                if default_result and lq_result
                else None
            ),
        })
        write_outputs(output_dir, rows)

    write_outputs(output_dir, rows)
    print(f"\nWrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")


def run_decipher_case(
    test_id: str,
    *,
    artifact_dir: Path,
    benchmark_root: str,
    split: str,
    model_path: Path | None = None,
    dry_run: bool,
) -> dict[str, Any] | None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    if model_path is not None:
        env["DECIPHER_NULL_MASK_RANKER"] = "language_quality"
        env["DECIPHER_NULL_MASK_LANGUAGE_QUALITY_MODEL"] = str(model_path)
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "decipher"),
        "benchmark",
        benchmark_root,
        "--split", split,
        "--test-id", test_id,
        "--automated-only",
        "--homophonic-refinement", "null_masks",
        "--artifact-dir", str(artifact_dir),
    ]
    run_command(cmd, env=env, dry_run=dry_run)
    if dry_run:
        return None
    artifact = latest_artifact(artifact_dir, test_id)
    return summarize_artifact(artifact)


def summarize_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    null_step = next(
        (step for step in payload.get("steps") or [] if step.get("name") == "search_null_masks"),
        {},
    )
    selected = null_step.get("selected") if isinstance(null_step.get("selected"), dict) else {}
    return {
        "artifact": str(path),
        "status": payload.get("status"),
        "solver": payload.get("solver"),
        "char_accuracy": payload.get("char_accuracy"),
        "word_accuracy": payload.get("word_accuracy"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "ranker": null_step.get("ranker"),
        "selected_mask": null_step.get("selected_mask"),
        "selected_source": selected.get("source"),
        "selected_validation_score_v2": selected.get("validation_score_v2"),
        "selected_ensemble_score_v1": selected.get("ensemble_score_v1"),
        "selected_language_quality_rank_score": selected.get("language_quality_rank_score"),
        "selected_language_quality_raw_score": selected.get("language_quality_raw_score"),
        "selected_preview": str(selected.get("preview") or "")[:180],
        "null_mask_elapsed_seconds": null_step.get("elapsed_seconds"),
        "slow_null_mask_rows": slow_null_mask_rows(null_step),
    }


def slow_null_mask_rows(null_step: dict[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    rows = [
        row for row in (null_step.get("evaluated_rows") or [])
        if isinstance(row, dict) and row.get("elapsed_seconds") is not None
    ]
    rows.sort(key=lambda row: float(row.get("elapsed_seconds") or 0.0), reverse=True)
    return [
        {
            "mask": row.get("mask") or [],
            "source": row.get("source"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "status": row.get("status"),
        }
        for row in rows[:limit]
        if float(row.get("elapsed_seconds") or 0.0) >= 5.0
    ]


def latest_artifact(root: Path, test_id: str) -> Path:
    candidates = sorted(
        root.rglob(f"{test_id}/*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            root.rglob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"no artifact JSON produced under {root}")
    return candidates[0]


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Copiale Candidate Ranker Sweep",
        "",
        "| Test | Default Char | LQ Char | Delta | Default Time | LQ Time | Default Mask | LQ Mask | LQ Ranker |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        default = row.get("default") or {}
        lq = row.get("language_quality") or {}
        lines.append(
            "| {test_id} | {default_char} | {lq_char} | {delta} | {default_time} | {lq_time} | {default_mask} | {lq_mask} | {ranker} |".format(
                test_id=row["test_id"],
                default_char=format_percent(default.get("char_accuracy")),
                lq_char=format_percent(lq.get("char_accuracy")),
                delta=format_delta(row.get("delta_char_accuracy")),
                default_time=format_seconds(default.get("elapsed_seconds")),
                lq_time=format_seconds(lq.get("elapsed_seconds")),
                default_mask=",".join(default.get("selected_mask") or []),
                lq_mask=",".join(lq.get("selected_mask") or []),
                ranker=lq.get("ranker") or "",
            )
        )
    slow_sections = []
    for row in rows:
        for label, result_key in (("default", "default"), ("language_quality", "language_quality")):
            result = row.get(result_key) or {}
            slow_rows = result.get("slow_null_mask_rows") or []
            if slow_rows:
                slow_sections.append((row["test_id"], label, slow_rows))
    if slow_sections:
        lines.append("")
        lines.append("## Slow Null-Mask Rows")
        lines.append("")
        lines.append("Rows listed here took at least 5 seconds inside an individual mask solve.")
        lines.append("")
        lines.append("| Test | Run | Mask | Source | Time | Status |")
        lines.append("|---|---|---|---|---:|---|")
        for test_id, label, slow_rows in slow_sections:
            for item in slow_rows:
                lines.append(
                    "| {test_id} | {label} | {mask} | {source} | {time} | {status} |".format(
                        test_id=test_id,
                        label=label,
                        mask=",".join(item.get("mask") or []),
                        source=item.get("source") or "",
                        time=format_seconds(item.get("elapsed_seconds")),
                        status=item.get("status") or "",
                    )
                )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['test_id']}`")
        lines.append(f"  - model: `{row['model_path']}`")
        if row.get("default"):
            lines.append(f"  - default: `{row['default']['artifact']}`")
        if row.get("language_quality"):
            lines.append(f"  - language_quality: `{row['language_quality']['artifact']}`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def format_delta(value: Any) -> str:
    if value is None:
        return ""
    sign = "+" if float(value) >= 0 else ""
    return f"{sign}{float(value) * 100:.1f}%"


def format_seconds(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.1f}s"


def run_command(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def python_bin() -> str:
    return str(REPO_ROOT / ".venv" / "bin" / "python")


if __name__ == "__main__":
    main()
