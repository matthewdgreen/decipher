#!/usr/bin/env python3
"""A/B comparison of the Gutenberg ``de`` model vs. the DTA period-German model.

Two parts, both writing artifacts under ``artifacts/german_model_ab/``:

1. **Calibration (separation quality).** Reuses
   ``scripts/audit_german_scoring.build_report`` to score the Copiale
   evidence-packet pages' known plaintext plus degraded controls (reversed,
   shuffled, rotated) under *both* binary models and report how cleanly each
   model separates true plaintext from the controls. This reads benchmark
   plaintext post-hoc — it is a calibration artifact, never solver routing.

2. **Solver-backed run.** Runs the automated (no-LLM) solver on the five
   evidence-packet pages with ``DECIPHER_NGRAM_MODEL_DE`` pointed at the DTA
   model, exercising the model-selection hook end to end and recording the
   resulting numbers (status, char/word accuracy, runtime).

The **baseline (Gutenberg de) leg** is not re-run by default: its numbers were
recorded by a prior automated run in the main checkout and are embedded here as
provenance (see ``BASELINE_LEG_PROVENANCE``) so the shipped packet is
self-contained. Pass ``--baseline-leg`` to additionally run the baseline model
in the *same process* — for future runs that want both legs measured together
under identical code. Pass ``--regenerate`` to rebuild ``ab_summary.json`` /
``ab_report.md`` from the numbers already recorded in ``ab_summary.json``
(enriches the packet with provenance without re-running any solver leg).

This is a reporting script. It changes no defaults; the orchestrator decides
whether to adopt the DTA model.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audit_german_scoring as audit  # noqa: E402
from frontier.suite import load_frontier_suite  # noqa: E402

DEFAULT_SUITE = REPO_ROOT / "frontier" / "copiale_evidence_packet.jsonl"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT.parent / "cipher_benchmark" / "benchmark"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "german_model_ab"
DEFAULT_BASELINE = REPO_ROOT / "models" / "ngram5_de.bin"
DEFAULT_DTA = REPO_ROOT / "models" / "ngram5_de_dta.bin"

# The "screen" budget profile fans out over these SA seeds (see automated runner
# budget_params); recorded in provenance so a reader can reproduce a leg.
SCREEN_SEEDS: list[int] = [0, 1, 2]

# --- Baseline (Gutenberg de) leg provenance --------------------------------
#
# The baseline leg is NOT re-run by this script by default. These numbers were
# produced by a prior automated run in the MAIN decipher checkout (not this
# worktree) under the same solver/config as the DTA leg below — only the ``de``
# n-gram model differs. They are embedded so the A/B packet stands on its own.
# Per-page char accuracy is the exact value recorded in that run's summary
# (rounds to 74.37 / 76.38 / 69.10 / 59.86 / 69.91).
BASELINE_LEG_PROVENANCE: dict[str, Any] = {
    "description": (
        "Baseline Gutenberg-de model leg. Recorded from a prior automated run; "
        "same solver/config as the DTA leg, only the de n-gram model differs."
    ),
    "artifact_path": "artifacts/baseline_20260713/copiale_null_masks/",
    "artifact_location": "main decipher checkout (not this worktree)",
    "model": "models/ngram5_de.bin",
    "model_sha256": "3fefceea86d468a87773a12976b35145764bac96eec3a93875f50a9c22333577",
    "config": {
        "homophonic_budget": "screen",
        "homophonic_refinement": "null_masks",
        "homophonic_solver": "zenith_native",
        "seeds": list(SCREEN_SEEDS),
    },
    "per_page_char_accuracy": {
        "copiale_single_B_copiale_p017": 0.743741,
        "copiale_single_B_copiale_p035": 0.763804,
        "copiale_single_B_copiale_p052": 0.690987,
        "copiale_single_B_copiale_p068": 0.598592,
        "copiale_single_B_copiale_p084": 0.699068,
    },
    "source": "recorded numbers; not re-run by this script (see --baseline-leg)",
}


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return out.stdout.strip() or None


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return bool(out.stdout.strip())


def run_calibration(
    *,
    suite_file: Path,
    benchmark_root: Path,
    baseline_model: Path,
    dta_model: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    report = audit.build_report(
        suite_file=suite_file,
        benchmark_root=benchmark_root,
        model_paths=[baseline_model, dta_model],
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (artifact_dir / "calibration_report.md").write_text(
        audit.render_markdown(report) + "\n", encoding="utf-8"
    )
    return report


def _model_signal_key(model_path: Path) -> str:
    return f"{model_path.stem}_mean_log_prob"


def summarize_calibration(report: dict[str, Any], baseline_model: Path, dta_model: Path) -> list[dict[str, Any]]:
    """Extract the plaintext-vs-controls separation for each binary model."""
    aggregate = report.get("aggregate", {})
    rows = []
    for label, model_path in (("gutenberg_de", baseline_model), ("dta_de", dta_model)):
        key = _model_signal_key(model_path)
        metrics = aggregate.get(key, {})
        rows.append(
            {
                "model": label,
                "model_path": str(model_path),
                "signal": key,
                "plaintext_beats_all_controls": metrics.get("plaintext_beats_all_controls"),
                "sample_count": metrics.get("sample_count"),
                "mean_margin": metrics.get("mean_margin"),
                "min_margin": metrics.get("min_margin"),
            }
        )
    return rows


def run_solver_backed(
    *,
    suite_file: Path,
    benchmark_root: Path,
    model: Path,
    artifact_dir: Path,
    homophonic_budget: str,
    homophonic_refinement: str,
    leg: str = "dta",
) -> list[dict[str, Any]]:
    # Select the given de model for the German language via the runner's env hook.
    os.environ["DECIPHER_NGRAM_MODEL_DE"] = str(model)

    from automated.runner import AutomatedBenchmarkRunner  # noqa: E402
    from benchmark.loader import BenchmarkLoader  # noqa: E402

    loader = BenchmarkLoader(benchmark_root)
    cases = load_frontier_suite(suite_file)
    runner = AutomatedBenchmarkRunner(
        artifact_dir=artifact_dir,
        homophonic_budget=homophonic_budget,
        homophonic_solver="zenith_native",
        homophonic_refinement=homophonic_refinement,
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        test_data = loader.load_test_data(case.test)
        print(f"[ab:{leg}] solving {case.test.test_id} (budget={homophonic_budget})…", flush=True)
        result = runner.run_test(test_data, language="de")
        rows.append(
            {
                "test_id": result.test_id,
                "status": result.status,
                "solver": result.solver,
                "char_accuracy": round(result.char_accuracy, 6),
                "word_accuracy": round(result.word_accuracy, 6),
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "error": result.error_message,
                "artifact_path": result.artifact_path,
            }
        )
        print(
            f"[ab:{leg}] {result.test_id}: status={result.status} "
            f"char={result.char_accuracy:.4f} word={result.word_accuracy:.4f} "
            f"solver={result.solver} {result.elapsed_seconds:.1f}s",
            flush=True,
        )
    return rows


def build_provenance(
    *,
    dta_model: Path,
    dta_model_meta: dict[str, Any] | None,
    homophonic_budget: str,
    homophonic_refinement: str,
    baseline_solver_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the self-contained provenance block for the A/B packet."""
    code_sha = _git_sha(REPO_ROOT)
    code_dirty = _git_dirty(REPO_ROOT)
    seeds = list(SCREEN_SEEDS) if homophonic_budget == "screen" else None

    dta_leg: dict[str, Any] = {
        "description": "DTA period-German model leg, measured by this script.",
        "model": str(dta_model),
        "model_sha256": (dta_model_meta or {}).get("sha256"),
        "config": {
            "homophonic_budget": homophonic_budget,
            "homophonic_refinement": homophonic_refinement,
            "homophonic_solver": "zenith_native",
            "seeds": seeds,
        },
        "code_git_sha": code_sha,
    }

    baseline_leg: dict[str, Any] = dict(BASELINE_LEG_PROVENANCE)
    if baseline_solver_rows:
        # --baseline-leg re-ran the baseline model in this process: replace the
        # recorded numbers with the freshly measured ones under this code SHA.
        baseline_leg = {
            **baseline_leg,
            "per_page_char_accuracy": {
                r["test_id"]: r["char_accuracy"] for r in baseline_solver_rows
            },
            "source": "measured in-process by this script (--baseline-leg)",
            "code_git_sha": code_sha,
        }

    return {
        "code_git_sha": code_sha,
        "code_git_dirty": code_dirty,
        "baseline_leg": baseline_leg,
        "dta_leg": dta_leg,
    }


def page_comparison(
    *, solver_rows: list[dict[str, Any]], baseline_leg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-page baseline-vs-DTA char accuracy with deltas (DTA minus baseline)."""
    base_by_id = baseline_leg.get("per_page_char_accuracy", {}) or {}
    rows: list[dict[str, Any]] = []
    for r in solver_rows:
        tid = r["test_id"]
        base = base_by_id.get(tid)
        dta = r.get("char_accuracy")
        delta = round(dta - base, 6) if (base is not None and dta is not None) else None
        rows.append(
            {
                "test_id": tid,
                "baseline_char_accuracy": base,
                "dta_char_accuracy": dta,
                "delta_char_accuracy": delta,
            }
        )
    return rows


def render_report(
    *,
    calibration_summary: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    dta_model: Path,
    dta_model_meta: dict[str, Any] | None,
    provenance: dict[str, Any],
    comparison: list[dict[str, Any]],
) -> str:
    lines = [
        "# German Model A/B — Gutenberg de vs. DTA period-German",
        "",
        "Calibration uses known plaintext + degraded controls (post-hoc). The",
        "solver-backed run selects the DTA model via `DECIPHER_NGRAM_MODEL_DE`.",
        "No defaults are changed by this report.",
        "",
        "## Provenance",
        "",
    ]
    code_sha = provenance.get("code_git_sha") or "unknown"
    dirty = provenance.get("code_git_dirty")
    dirty_note = " (working tree dirty)" if dirty else ("" if dirty is not None else " (git state unknown)")
    lines.append(f"- Code git SHA (both legs): `{code_sha}`{dirty_note}")
    lines.append("")

    base = provenance.get("baseline_leg", {})
    base_cfg = base.get("config", {})
    lines.extend(
        [
            "### Baseline leg (Gutenberg de)",
            "",
            f"- {base.get('description', '')}",
            f"- Recorded artifact: `{base.get('artifact_path')}` "
            f"({base.get('artifact_location', '')})",
            f"- Model: `{base.get('model')}` sha256 `{base.get('model_sha256')}`",
            f"- Config: budget={base_cfg.get('homophonic_budget')}, "
            f"refinement={base_cfg.get('homophonic_refinement')}, "
            f"solver={base_cfg.get('homophonic_solver')}, "
            f"seeds={base_cfg.get('seeds')}",
            f"- Source: {base.get('source')}",
            "",
        ]
    )

    dta_leg = provenance.get("dta_leg", {})
    dta_cfg = dta_leg.get("config", {})
    lines.extend(
        [
            "### DTA leg (period German)",
            "",
            f"- Model: `{dta_leg.get('model')}` sha256 `{dta_leg.get('model_sha256')}`",
            f"- Config: budget={dta_cfg.get('homophonic_budget')}, "
            f"refinement={dta_cfg.get('homophonic_refinement')}, "
            f"solver={dta_cfg.get('homophonic_solver')}, "
            f"seeds={dta_cfg.get('seeds')}",
            "",
            "## DTA model",
            "",
        ]
    )
    if dta_model_meta:
        stats = dta_model_meta.get("corpus_stats", {})
        lines.append(f"- Path: `{dta_model}`")
        lines.append(f"- Normalized chars: {stats.get('normalized_characters'):,}")
        lines.append(f"- Distinct 5-grams: {stats.get('distinct_seen_5grams'):,}")
        lines.append(f"- sha256: `{dta_model_meta.get('sha256', 'unknown')}`")
    else:
        lines.append(f"- Path: `{dta_model}` (metadata sidecar not found)")
    lines.extend(
        [
            "",
            "## Calibration — plaintext vs. controls separation",
            "",
            "| Model | Plaintext wins | Mean margin | Worst margin |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in calibration_summary:
        wins = row["plaintext_beats_all_controls"]
        total = row["sample_count"]
        lines.append(
            f"| {row['model']} | {wins}/{total} | "
            f"{_fmt(row['mean_margin'])} | {_fmt(row['min_margin'])} |"
        )
    lines.extend(
        [
            "",
            "## Solver-backed run (DTA model selected)",
            "",
            "| Test | Status | Char acc | Word acc | Solver | Seconds |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for row in solver_rows:
        lines.append(
            f"| {row['test_id']} | {row['status']} | "
            f"{row['char_accuracy']:.4f} | {row['word_accuracy']:.4f} | "
            f"{row['solver']} | {row['elapsed_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Per-page comparison — baseline (recorded) vs. DTA",
            "",
            "Char accuracy per page; delta is DTA minus baseline.",
            "",
            "| Test | Baseline char | DTA char | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparison:
        lines.append(
            f"| {row['test_id']} | {_fmt(row['baseline_char_accuracy'])} | "
            f"{_fmt(row['dta_char_accuracy'])} | {_fmt_delta(row['delta_char_accuracy'])} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _fmt_delta(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):+.4f}"


def _load_model_meta(model_path: Path) -> dict[str, Any] | None:
    sidecar = model_path.with_name(model_path.name + ".metadata.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _write_packet(
    *,
    artifact_dir: Path,
    combined: dict[str, Any],
    report_md: str,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "ab_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (artifact_dir / "ab_report.md").write_text(report_md, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite-file", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--dta-model", type=Path, default=DEFAULT_DTA)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--homophonic-budget",
        default="screen",
        choices=["screen", "full"],
        help="Solver budget for the 5-page run. Default: screen (bounded runtime).",
    )
    parser.add_argument(
        "--homophonic-refinement",
        default="null_masks",
        choices=["none", "two_stage", "targeted_repair", "family_repair", "null_masks"],
        help=(
            "Homophonic refinement stage for the solver-backed run. Default: "
            "null_masks (matches the old-model baseline config)."
        ),
    )
    parser.add_argument(
        "--skip-solver",
        action="store_true",
        help="Only run the calibration comparison (no solver-backed run).",
    )
    parser.add_argument(
        "--baseline-leg",
        action="store_true",
        help=(
            "Also run the baseline (Gutenberg de) model's solver leg in this "
            "process, so both legs are measured together under identical code. "
            "Default off (the baseline numbers are embedded as provenance)."
        ),
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "Rebuild ab_summary.json / ab_report.md from the numbers already "
            "recorded in ab_summary.json (adds provenance without re-running any "
            "solver or calibration leg)."
        ),
    )
    args = parser.parse_args(argv)

    artifact_dir = args.artifact_dir.expanduser()
    dta_model_meta = _load_model_meta(args.dta_model)

    if args.regenerate:
        existing_path = artifact_dir / "ab_summary.json"
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        calibration_summary = existing.get("calibration_summary", [])
        solver_rows = existing.get("solver_backed", [])
        baseline_solver_rows = existing.get("baseline_solver_backed", [])
        print(f"[ab] regenerating packet from recorded numbers in {existing_path}", flush=True)
    else:
        print("[ab] running calibration comparison…", flush=True)
        report = run_calibration(
            suite_file=args.suite_file,
            benchmark_root=args.benchmark_root,
            baseline_model=args.baseline_model,
            dta_model=args.dta_model,
            artifact_dir=artifact_dir,
        )
        calibration_summary = summarize_calibration(report, args.baseline_model, args.dta_model)

        baseline_solver_rows: list[dict[str, Any]] = []
        if args.baseline_leg:
            baseline_solver_rows = run_solver_backed(
                suite_file=args.suite_file,
                benchmark_root=args.benchmark_root,
                model=args.baseline_model,
                artifact_dir=artifact_dir / "baseline_leg",
                homophonic_budget=args.homophonic_budget,
                homophonic_refinement=args.homophonic_refinement,
                leg="baseline",
            )

        solver_rows = []
        if not args.skip_solver:
            solver_rows = run_solver_backed(
                suite_file=args.suite_file,
                benchmark_root=args.benchmark_root,
                model=args.dta_model,
                artifact_dir=artifact_dir,
                homophonic_budget=args.homophonic_budget,
                homophonic_refinement=args.homophonic_refinement,
                leg="dta",
            )

    provenance = build_provenance(
        dta_model=args.dta_model,
        dta_model_meta=dta_model_meta,
        homophonic_budget=args.homophonic_budget,
        homophonic_refinement=args.homophonic_refinement,
        baseline_solver_rows=baseline_solver_rows,
    )
    comparison = page_comparison(
        solver_rows=solver_rows, baseline_leg=provenance["baseline_leg"]
    )

    combined = {
        "calibration_summary": calibration_summary,
        "solver_backed": solver_rows,
        "dta_model": str(args.dta_model),
        "baseline_model": str(args.baseline_model),
        "provenance": provenance,
        "page_comparison": comparison,
    }
    if baseline_solver_rows:
        combined["baseline_solver_backed"] = baseline_solver_rows

    report_md = render_report(
        calibration_summary=calibration_summary,
        solver_rows=solver_rows,
        dta_model=args.dta_model,
        dta_model_meta=dta_model_meta,
        provenance=provenance,
        comparison=comparison,
    )
    _write_packet(artifact_dir=artifact_dir, combined=combined, report_md=report_md)
    print("\n" + report_md, flush=True)
    print(f"[ab] artifacts written under {artifact_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
