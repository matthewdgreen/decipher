#!/usr/bin/env python3
"""Run a wider Copiale null-mask candidate-breadth experiment.

This is an evaluation harness, not a solver primitive.  It asks two separate
post-hoc questions:

1. Did a broader/diverse null-mask run generate a better candidate basin?
2. If it did, did the language-quality ranker put that candidate near the top?

Ground truth is used only after each solver run has completed, to label and
measure the finalist pool.  The solver subprocesses receive only ciphertext,
benchmark metadata normally available to the automated runner, and the
ground-truth-free language-quality model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.scorer import score_decryption


DEFAULT_TEST_IDS = ("copiale_single_B_copiale_p068",)
DEFAULT_RANKERS = ("validation", "ensemble", "language_quality")


@dataclass(frozen=True)
class CandidateRow:
    test_id: str
    run_label: str
    artifact: str
    candidate_id: str
    evaluated_index: int | None
    rank_in_artifact: int
    mask: tuple[str, ...]
    source: str
    selected: bool
    char_accuracy: float
    language_quality_raw_score: float | None
    language_quality_score: float | None
    language_quality_rank_score: float | None
    validation_score_v2: float | None
    ensemble_score_v1: float | None
    selection_score: float | None
    decryption: str
    preview: str

    @property
    def mask_label(self) -> str:
        return ",".join(self.mask) if self.mask else "(none)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and report a wide/diverse Copiale null-mask finalist experiment."
    )
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument("--test-id", action="append", default=[])
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--training-artifact",
        action="append",
        default=["artifacts/copiale_evidence_packet/automated_null_masks_candidate_blend"],
        help="Prior finalist artifact directory for training held-out LQ models.",
    )
    parser.add_argument(
        "--model-path",
        help=(
            "Use one existing language-quality model for all tests.  If omitted, "
            "one held-out model is trained per test_id."
        ),
    )
    parser.add_argument("--ranker", action="append", choices=DEFAULT_RANKERS, default=[])
    parser.add_argument(
        "--feature-set",
        choices=("all", "no_solver", "text_only"),
        default="text_only",
        help=(
            "Feature subset for the held-out LQ model. The default is text_only "
            "so mask/deletion metadata cannot dominate candidate readability."
        ),
    )
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument("--max-mask-size", type=int, default=3)
    parser.add_argument("--max-masks", type=int, default=1500)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--budget", choices=("screen", "full"), default="screen")
    parser.add_argument("--beam-width", type=int, default=36)
    parser.add_argument("--beam-max-masks", type=int, default=500)
    parser.add_argument("--neighborhood-top-n", type=int, default=24)
    parser.add_argument("--neighborhood-max-masks", type=int, default=500)
    parser.add_argument("--promote-top-n", type=int, default=0)
    parser.add_argument("--promote-reruns", type=int, default=0)
    parser.add_argument("--confirm-top-n", type=int, default=0)
    parser.add_argument("--confirm-reruns", type=int, default=0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--adaptive-max-masks", type=int, default=500)
    parser.add_argument("--adaptive-bridge-max-masks", type=int, default=240)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    test_ids = tuple(args.test_id or DEFAULT_TEST_IDS)
    rankers = tuple(args.ranker or DEFAULT_RANKERS)
    output_dir = (
        resolve_path(Path(args.output_dir))
        if args.output_dir
        else REPO_ROOT / "artifacts" / "copiale_breadth_experiment" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for test_id in test_ids:
        print(f"\n=== {test_id} ===", flush=True)
        model_path = (
            resolve_path(Path(args.model_path))
            if args.model_path
            else train_model_for_test(test_id, args, models_dir)
        )
        run_artifacts = []
        for ranker in rankers:
            run_artifacts.append(
                run_ranker_case(
                    test_id,
                    ranker=ranker,
                    model_path=model_path,
                    output_dir=output_dir / ranker,
                    args=args,
                )
            )
        if args.dry_run:
            continue
        candidates = aggregate_candidates(test_id, run_artifacts)
        report = summarize_candidate_pool(test_id, candidates)
        report["model_path"] = str(model_path)
        report["artifacts"] = [str(path) for path in run_artifacts]
        summary_rows.append(report)
        write_test_report(output_dir, test_id, candidates, report)
        write_outputs(output_dir, summary_rows)

    if not args.dry_run:
        write_outputs(output_dir, summary_rows)
        print(f"\nWrote {output_dir / 'summary.json'}")
        print(f"Wrote {output_dir / 'summary.md'}")


def train_model_for_test(test_id: str, args: argparse.Namespace, models_dir: Path) -> Path:
    model_path = models_dir / f"{test_id}_candidate_only.json"
    report_path = models_dir / f"{test_id}_candidate_only.md"
    cmd = [
        python_bin(),
        "scripts/train_language_quality_scorer.py",
        "--language", "de",
        "--candidate-only",
        "--holdout-group", test_id,
        "--leave-one-group-out",
        "--output", str(model_path),
        "--report", str(report_path),
        "--objective", "pairwise",
        "--feature-set", args.feature_set,
        "--nonnegative-weights",
    ]
    for artifact in args.training_artifact:
        cmd.extend(["--artifact", artifact])
    run_command(cmd, dry_run=args.dry_run)
    return model_path


def run_ranker_case(
    test_id: str,
    *,
    ranker: str,
    model_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["DECIPHER_NULL_MASK_RANKER"] = ranker
    env["DECIPHER_NULL_MASK_LANGUAGE_QUALITY_MODEL"] = str(model_path)
    env["DECIPHER_NULL_MASK_CANDIDATE_LIMIT"] = str(args.candidate_limit)
    env["DECIPHER_NULL_MASK_MAX_SIZE"] = str(args.max_mask_size)
    env["DECIPHER_NULL_MASK_MAX_MASKS"] = str(args.max_masks)
    env["DECIPHER_NULL_MASK_TOP_N"] = str(args.top_n)
    env["DECIPHER_NULL_MASK_BUDGET"] = args.budget
    env["DECIPHER_NULL_MASK_BEAM"] = "1"
    env["DECIPHER_NULL_MASK_BEAM_WIDTH"] = str(args.beam_width)
    env["DECIPHER_NULL_MASK_BEAM_MAX_SIZE"] = str(max(args.max_mask_size, 3))
    env["DECIPHER_NULL_MASK_BEAM_MAX_MASKS"] = str(args.beam_max_masks)
    env["DECIPHER_NULL_MASK_NEIGHBORHOOD"] = "1"
    env["DECIPHER_NULL_MASK_NEIGHBORHOOD_TOP_N"] = str(args.neighborhood_top_n)
    env["DECIPHER_NULL_MASK_NEIGHBORHOOD_MAX_SIZE"] = str(max(args.max_mask_size, 3))
    env["DECIPHER_NULL_MASK_NEIGHBORHOOD_MAX_MASKS"] = str(args.neighborhood_max_masks)
    env["DECIPHER_NULL_MASK_PROMOTE_TOP_N"] = str(args.promote_top_n)
    env["DECIPHER_NULL_MASK_PROMOTE_RERUNS"] = str(args.promote_reruns)
    env["DECIPHER_NULL_MASK_CONFIRM_TOP_N"] = str(args.confirm_top_n)
    env["DECIPHER_NULL_MASK_CONFIRM_RERUNS"] = str(args.confirm_reruns)
    env["DECIPHER_NULL_MASK_ADAPTIVE"] = "1" if args.adaptive else "0"
    env["DECIPHER_NULL_MASK_ADAPTIVE_MAX_MASKS"] = str(args.adaptive_max_masks)
    env["DECIPHER_NULL_MASK_ADAPTIVE_BRIDGE_MAX_MASKS"] = str(args.adaptive_bridge_max_masks)

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "decipher"),
        "benchmark",
        args.benchmark_root,
        "--split", args.split,
        "--test-id", test_id,
        "--automated-only",
        "--homophonic-refinement", "null_masks",
        "--artifact-dir", str(output_dir),
    ]
    run_command(cmd, env=env, dry_run=args.dry_run)
    if args.dry_run:
        return output_dir / "DRY_RUN.json"
    return latest_artifact(output_dir, test_id)


def aggregate_candidates(test_id: str, artifact_paths: list[Path]) -> list[CandidateRow]:
    seen: set[str] = set()
    rows: list[CandidateRow] = []
    for artifact_path in artifact_paths:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        ground_truth = str(payload.get("ground_truth") or "")
        null_step = next(
            (step for step in payload.get("steps") or [] if step.get("name") == "search_null_masks"),
            None,
        )
        if not isinstance(null_step, dict):
            continue
        run_label = str(null_step.get("ranker") or artifact_path.parent.name)
        finalist_rows: list[dict[str, Any]] = []
        selected = null_step.get("selected")
        if isinstance(selected, dict):
            finalist_rows.append(selected)
        finalist_rows.extend(
            candidate
            for candidate in (null_step.get("top_finalists") or [])
            if isinstance(candidate, dict)
        )
        for rank, candidate in enumerate(finalist_rows, start=1):
            if not isinstance(candidate, dict):
                continue
            is_selected = rank == 1 and isinstance(null_step.get("selected"), dict)
            decryption = str(candidate.get("decryption") or candidate.get("plaintext") or "")
            if not decryption:
                continue
            digest = hashlib.sha256(decryption.encode("utf-8")).hexdigest()
            if digest in seen and not is_selected:
                continue
            seen.add(digest)
            score = score_decryption(
                test_id,
                decryption,
                ground_truth,
                agent_score=0.0,
                status="completed",
            )
            rows.append(
                CandidateRow(
                    test_id=test_id,
                    run_label=run_label,
                    artifact=str(artifact_path),
                    candidate_id=str(candidate.get("candidate_id") or ""),
                    evaluated_index=int_or_none(candidate.get("evaluated_index")),
                    rank_in_artifact=rank,
                    mask=tuple(str(symbol) for symbol in (candidate.get("mask") or [])),
                    source=str(candidate.get("source") or ""),
                    selected=is_selected,
                    char_accuracy=score.char_accuracy,
                    language_quality_raw_score=float_or_none(candidate.get("language_quality_raw_score")),
                    language_quality_score=float_or_none(candidate.get("language_quality_score")),
                    language_quality_rank_score=float_or_none(candidate.get("language_quality_rank_score")),
                    validation_score_v2=float_or_none(candidate.get("validation_score_v2")),
                    ensemble_score_v1=float_or_none(candidate.get("ensemble_score_v1")),
                    selection_score=float_or_none(candidate.get("selection_score")),
                    decryption=decryption,
                    preview=decryption[:140],
                )
            )
    return rows


def summarize_candidate_pool(test_id: str, candidates: list[CandidateRow]) -> dict[str, Any]:
    if not candidates:
        return {"test_id": test_id, "candidate_count": 0}
    by_char = sorted(candidates, key=lambda row: row.char_accuracy, reverse=True)
    by_lq = sorted(candidates, key=candidate_language_quality_rank_key, reverse=True)
    best = by_char[0]
    lq_rank_by_digest = {id(row): rank for rank, row in enumerate(by_lq, start=1)}
    best_lq_rank = lq_rank_by_digest[id(best)]
    lq_pick = by_lq[0]
    lq_selected = next(
        (row for row in candidates if row.run_label == "language_quality" and row.selected),
        None,
    )
    char_values = sorted(row.char_accuracy for row in candidates)
    median = char_values[len(char_values) // 2]
    return {
        "test_id": test_id,
        "candidate_count": len(candidates),
        "best_char_accuracy": round(best.char_accuracy, 6),
        "best_char_mask": list(best.mask),
        "best_char_source": best.source,
        "best_char_run_label": best.run_label,
        "best_char_lq_rank": best_lq_rank,
        "best_char_lq_raw_score": best.language_quality_raw_score,
        "lq_pick_char_accuracy": round(lq_pick.char_accuracy, 6),
        "lq_pick_mask": list(lq_pick.mask),
        "lq_pick_source": lq_pick.source,
        "lq_pick_run_label": lq_pick.run_label,
        "lq_pick_raw_score": lq_pick.language_quality_raw_score,
        "lq_pick_rank_score": lq_pick.language_quality_rank_score,
        "lq_char_gap": round(best.char_accuracy - lq_pick.char_accuracy, 6),
        "lq_selected_char_accuracy": round(lq_selected.char_accuracy, 6) if lq_selected else None,
        "lq_selected_mask": list(lq_selected.mask) if lq_selected else [],
        "lq_selected_source": lq_selected.source if lq_selected else "",
        "lq_selected_rank_score": lq_selected.language_quality_rank_score if lq_selected else None,
        "lq_selected_gap": round(best.char_accuracy - lq_selected.char_accuracy, 6) if lq_selected else None,
        "median_char_accuracy": round(median, 6),
        "char_spread_best_minus_median": round(best.char_accuracy - median, 6),
        "lq_top_1_capture": best_lq_rank <= 1,
        "lq_top_3_capture": best_lq_rank <= 3,
        "lq_top_5_capture": best_lq_rank <= 5,
        "lq_top_10_capture": best_lq_rank <= 10,
        "lq_top_25_capture": best_lq_rank <= 25,
    }


def write_test_report(
    output_dir: Path,
    test_id: str,
    candidates: list[CandidateRow],
    report: dict[str, Any],
) -> None:
    by_lq = sorted(candidates, key=candidate_language_quality_rank_key, reverse=True)
    by_char = sorted(candidates, key=lambda row: row.char_accuracy, reverse=True)
    path = output_dir / f"{test_id}_breadth_report.md"
    lines = [
        f"# Candidate Breadth Report: {test_id}",
        "",
        "Ground truth is used only in this report, after candidate generation.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "candidate_count",
        "best_char_accuracy",
        "lq_pick_char_accuracy",
        "lq_char_gap",
        "best_char_lq_rank",
        "median_char_accuracy",
        "char_spread_best_minus_median",
    ):
        value = report.get(key)
        if key.endswith("accuracy") or key.endswith("gap") or key.startswith("char_spread"):
            value = format_percent(value)
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Top By Language Quality",
        "",
        candidate_table(by_lq[:25], best=by_char[0] if by_char else None),
        "",
        "## Top By Post-Hoc Character Accuracy",
        "",
        candidate_table(by_char[:25], best=by_char[0] if by_char else None),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def candidate_language_quality_rank_key(row: CandidateRow) -> tuple[float, ...]:
    """Runtime-equivalent language-quality ordering for post-hoc reports."""
    lq_score = row.language_quality_rank_score if row.language_quality_rank_score is not None else float("-inf")
    lq_bucket = round(lq_score, 2) if lq_score != float("-inf") else lq_score
    validation = row.validation_score_v2 if row.validation_score_v2 is not None else float("-inf")
    ensemble = row.ensemble_score_v1 if row.ensemble_score_v1 is not None else float("-inf")
    if validation == float("-inf") or ensemble == float("-inf"):
        tie_break = float("-inf")
    else:
        tie_break = validation + ensemble * 0.05
    return (
        lq_bucket,
        tie_break,
        ensemble,
        validation,
        row.selection_score if row.selection_score is not None else float("-inf"),
    )


def candidate_table(rows: list[CandidateRow], *, best: CandidateRow | None) -> str:
    lines = [
        "| Rank | Candidate | Eval Index | Mask | Run | Source | GT Char | LQ Raw | LQ Rank | Val2 | Ens | Note | Preview |",
        "|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(rows, start=1):
        notes = []
        if best is not None and row.decryption == best.decryption:
            notes.append("GT-best")
        if row.selected:
            notes.append("selected")
        note = ",".join(notes)
        lines.append(
            "| {rank} | {candidate_id} | {evaluated_index} | {mask} | {run} | {source} | {char} | {lq_raw} | {lq_rank} | {val2} | {ens} | {note} | {preview} |".format(
                rank=rank,
                candidate_id=row.candidate_id or "",
                evaluated_index=row.evaluated_index if row.evaluated_index is not None else "",
                mask=row.mask_label,
                run=row.run_label,
                source=row.source,
                char=format_percent(row.char_accuracy),
                lq_raw=format_number(row.language_quality_raw_score),
                lq_rank=format_number(row.language_quality_rank_score),
                val2=format_number(row.validation_score_v2),
                ens=format_number(row.ensemble_score_v1),
                note=note,
                preview=row.preview.replace("|", "/"),
            )
        )
    return "\n".join(lines)


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Copiale Candidate Breadth Experiment",
        "",
        "| Test | Candidates | Best Char | LQ Selected | LQ Selected Gap | Top LQ-Rank Char | Top LQ Gap | Best LQ Rank | Top-3 | Top-10 | Best Mask | LQ Selected Mask | Top LQ Mask |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {test_id} | {count} | {best} | {selected} | {selected_gap} | {pick} | {gap} | {rank} | {top3} | {top10} | {best_mask} | {selected_mask} | {pick_mask} |".format(
                test_id=row.get("test_id"),
                count=row.get("candidate_count", 0),
                best=format_percent(row.get("best_char_accuracy")),
                selected=format_percent(row.get("lq_selected_char_accuracy")),
                selected_gap=format_percent(row.get("lq_selected_gap")),
                pick=format_percent(row.get("lq_pick_char_accuracy")),
                gap=format_percent(row.get("lq_char_gap")),
                rank=row.get("best_char_lq_rank", ""),
                top3="Y" if row.get("lq_top_3_capture") else "n",
                top10="Y" if row.get("lq_top_10_capture") else "n",
                best_mask=",".join(row.get("best_char_mask") or []) or "(none)",
                selected_mask=",".join(row.get("lq_selected_mask") or []) or "(none)",
                pick_mask=",".join(row.get("lq_pick_mask") or []) or "(none)",
            )
        )
    lines.append("")
    lines.append("## Per-Test Reports")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row.get('test_id')}_breadth_report.md`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def latest_artifact(root: Path, test_id: str) -> Path:
    candidates = sorted(
        root.rglob(f"{test_id}/*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(root.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no artifact JSON produced under {root}")
    return candidates[0]


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


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def format_number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
