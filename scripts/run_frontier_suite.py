#!/usr/bin/env python3
"""Run the curated frontier automated solver suite."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_EXTERNAL_CONFIG = REPO_ROOT / "external_baselines" / "zenith_only.json"

from automated.runner import AutomatedBenchmarkRunner
from benchmark.loader import BenchmarkLoader
from benchmark.scorer import has_word_boundaries
from external_baselines import ExternalBaselineConfig, run_external_baseline
from frontier.suite import (
    FrontierCase,
    canonical_solver_name,
    evaluate_frontier_rows,
    load_frontier_suite,
    resolve_frontier_case,
)
from testgen.cache import PlaintextCache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the curated frontier automated solver suite.",
    )
    parser.add_argument(
        "--suite-file",
        default="frontier/automated_solver_frontier.jsonl",
        help="Frontier suite JSONL file.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=["decipher", "external"],
        default=["decipher"],
    )
    parser.add_argument("--frontier-class", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--test-id", action="append", default=[])
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--cache-dir", default="testgen_cache")
    parser.add_argument("--allow-generate", action="store_true")
    parser.add_argument(
        "--external-config",
        default=str(DEFAULT_EXTERNAL_CONFIG),
        help=(
            "External baseline config JSON. Defaults to the Zenith-only config; "
            "pass external_baselines/local_tools.json to include slower wrappers such as zkdecrypto-lite."
        ),
    )
    parser.add_argument("--oracle-select", action="store_true")
    parser.add_argument("--artifact-dir", default="artifacts/frontier_suite")
    parser.add_argument("--summary-jsonl")
    parser.add_argument("--summary-csv")
    parser.add_argument(
        "--homophonic-budget",
        choices=["full", "screen"],
        default="full",
        help="Search budget for Decipher automated homophonic runs.",
    )
    parser.add_argument(
        "--homophonic-refinement",
        choices=["none", "two_stage", "targeted_repair", "family_repair"],
        default="none",
        help="Optional second-stage local refinement for Decipher automated homophonic runs.",
    )
    parser.add_argument(
        "--legacy-homophonic",
        action="store_true",
        help="Use the older pre-zenith_native homophonic solver path for Decipher runs.",
    )
    parser.add_argument(
        "--transform-search",
        choices=["off", "auto", "screen", "wide", "rank", "full", "promote"],
        default="off",
        help=(
            "Run cheap transform-search diagnostics for Decipher automated runs. "
            "`auto` screens only when router signals are promising; `screen` "
            "records a structural candidate menu; `wide` runs a larger "
            "structural-only search; `rank`/`full` run bounded solver probes "
            "on top candidates; `promote` probes candidates from a prior "
            "wide/screen artifact."
        ),
    )
    parser.add_argument(
        "--transform-search-profile",
        choices=["fast", "broad", "wide"],
        default="broad",
        help=(
            "Candidate breadth profile for Decipher transform-search rank/full. "
            "`fast` is recommended for regression runs and trims mutations "
            "and confirmations; "
            "`broad` preserves the research-oriented candidate set; "
            "`wide` expands the structural-only candidate sweep."
        ),
    )
    parser.add_argument(
        "--transform-search-max-generated-candidates",
        type=int,
        help=(
            "Optional safety cap for transform-search structural candidate "
            "generation. Use this with --transform-search wide for larger "
            "candidate sweeps before solver promotion."
        ),
    )
    parser.add_argument(
        "--transform-promote-artifact",
        help="Source automated artifact containing transform_search.screen candidates to promote.",
    )
    parser.add_argument(
        "--transform-promote-candidate-id",
        action="append",
        default=[],
        help="Specific transform candidate id to promote from the source artifact. May be repeated.",
    )
    parser.add_argument(
        "--transform-promote-top-n",
        type=int,
        help="Promote the top N structural candidates from the source artifact.",
    )
    args = parser.parse_args()

    if "external" in args.solvers and not Path(args.external_config).exists():
        parser.error(f"--external-config not found: {args.external_config}")

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_jsonl = Path(args.summary_jsonl) if args.summary_jsonl else artifact_dir / "summary.jsonl"
    summary_csv = Path(args.summary_csv) if args.summary_csv else artifact_dir / "summary.csv"

    cases = load_frontier_suite(
        args.suite_file,
        frontier_classes=set(args.frontier_class) or None,
        tags=set(args.tag) or None,
        families=set(args.family) or None,
        test_ids=set(args.test_id) or None,
    )
    print(f"Frontier cases: {len(cases)}")
    print(f"Artifacts: {artifact_dir}")
    print(f"Summary JSONL: {summary_jsonl}")
    print(f"Summary CSV: {summary_csv}")
    print("Mode: automated frontier suite")
    print()

    benchmark_loader = BenchmarkLoader(args.benchmark_root)
    cache = PlaintextCache(args.cache_dir)
    decipher_runner = AutomatedBenchmarkRunner(
        artifact_dir=artifact_dir / "decipher",
        homophonic_budget=args.homophonic_budget,
        homophonic_refinement=args.homophonic_refinement,
        homophonic_solver="legacy" if args.legacy_homophonic else "zenith_native",
        transform_search=args.transform_search,
        transform_search_profile=args.transform_search_profile,
        transform_search_max_generated_candidates=args.transform_search_max_generated_candidates,
        transform_promote_artifact=args.transform_promote_artifact,
        transform_promote_candidate_ids=args.transform_promote_candidate_id,
        transform_promote_top_n=args.transform_promote_top_n,
    )
    external_configs = _load_external_configs(args.external_config, args.oracle_select) if "external" in args.solvers else []

    rows: list[dict[str, Any]] = []
    try:
        for idx, case in enumerate(cases, start=1):
            print(f"[{idx}/{len(cases)}] {case.test.test_id} — {case.frontier_class}")
            test_data = resolve_frontier_case(
                case,
                benchmark_loader=benchmark_loader,
                cache=cache,
                allow_generate=args.allow_generate,
                generation_api_key=_generation_api_key() if args.allow_generate else "",
            )

            if "decipher" in args.solvers and _should_run_solver(case, "decipher-automated"):
                print("  [decipher-automated] running...")
                t0 = time.time()
                try:
                    result = decipher_runner.run_test(
                        test_data,
                        language=case.synthetic_spec.language if case.synthetic_spec else test_data.plaintext_language or None,
                    )
                    row = _row_for_result(
                        case=case,
                        test_data=test_data,
                        solver="decipher-automated",
                        status=result.status,
                        char_accuracy=result.char_accuracy,
                        word_accuracy=result.word_accuracy,
                        elapsed=time.time() - t0,
                        artifact_path=getattr(result, "artifact_path", ""),
                        error=getattr(result, "error_message", ""),
                        candidates=1,
                    )
                except Exception as exc:  # noqa: BLE001
                    row = _row_for_exception(case, test_data, "decipher-automated", time.time() - t0, exc)
                rows.append(row)
                _write_frontier_summaries(summary_jsonl, summary_csv, rows)
                _print_result(evaluate_frontier_rows([row])[0])

            for config in external_configs:
                if not _should_run_solver(case, config.name):
                    continue
                print(f"  [{config.name}] running...")
                t0 = time.time()
                try:
                    result = run_external_baseline(
                        test_data,
                        config,
                        artifact_dir=artifact_dir / "external",
                    )
                    row = _row_for_result(
                        case=case,
                        test_data=test_data,
                        solver=config.name,
                        status=result.status,
                        char_accuracy=result.char_accuracy,
                        word_accuracy=result.word_accuracy,
                        elapsed=result.elapsed,
                        artifact_path=result.artifact_path,
                        error=result.error,
                        candidates=result.candidates_considered,
                    )
                except Exception as exc:  # noqa: BLE001
                    row = _row_for_exception(case, test_data, config.name, time.time() - t0, exc)
                rows.append(row)
                _write_frontier_summaries(summary_jsonl, summary_csv, rows)
                _print_result(evaluate_frontier_rows([row])[0])
            print()
    finally:
        _write_frontier_summaries(summary_jsonl, summary_csv, rows)

    _print_summary(evaluate_frontier_rows(rows))


def _should_run_solver(case: FrontierCase, solver: str) -> bool:
    solver_key = canonical_solver_name(solver)
    return not case.expected_solvers or solver_key in case.expected_solvers


def _load_external_configs(path: str | None, oracle_select: bool) -> list[ExternalBaselineConfig]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    configs = [ExternalBaselineConfig.from_dict(item) for item in data.get("solvers", [])]
    if oracle_select:
        return [
            ExternalBaselineConfig(
                name=c.name,
                command=c.command,
                timeout_seconds=c.timeout_seconds,
                cwd=c.cwd,
                env=c.env,
                solution_regex=c.solution_regex,
                selection="best_by_score",
            )
            for c in configs
        ]
    return configs


def _generation_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import keyring

        key = keyring.get_password("decipher", "anthropic_api_key")
        if key:
            return key
    except Exception:
        pass
    raise SystemExit(
        "Synthetic plaintext generation requires ANTHROPIC_API_KEY or the "
        "macOS Keychain entry service=decipher account=anthropic_api_key."
    )


def _row_for_result(
    case: FrontierCase,
    test_data,
    solver: str,
    status: str,
    char_accuracy: float,
    word_accuracy: float,
    elapsed: float,
    artifact_path: str,
    error: str,
    candidates: int,
) -> dict[str, Any]:
    return {
        "test_id": test_data.test.test_id,
        "source": case.source_mode,
        "family": test_data.test.cipher_system or case.test.cipher_system,
        "cipher_system": test_data.test.cipher_system,
        "language": case.synthetic_spec.language if case.synthetic_spec else (test_data.plaintext_language or ""),
        "approx_length": case.synthetic_spec.approx_length if case.synthetic_spec else "",
        "word_boundaries": has_word_boundaries(test_data.plaintext),
        "homophonic": bool(case.synthetic_spec.homophonic) if case.synthetic_spec else ("homophonic" in test_data.test.cipher_system.lower()),
        "seed": case.synthetic_spec.seed if case.synthetic_spec else "",
        "solver": solver,
        "solver_key": canonical_solver_name(solver),
        "status": status,
        "char_accuracy": round(float(char_accuracy), 6),
        "word_accuracy": round(float(word_accuracy), 6),
        "elapsed_seconds": round(float(elapsed), 3),
        "candidates": candidates,
        "artifact_path": artifact_path,
        "error": error,
        "frontier_class": case.frontier_class,
        "frontier_tags": list(case.frontier_tags),
        "expected_solvers": list(case.expected_solvers),
        "expected_status_by_solver": dict(case.expected_status_by_solver),
        "min_char_accuracy_by_solver": dict(case.min_char_accuracy_by_solver),
        "max_elapsed_seconds_by_solver": dict(case.max_elapsed_seconds_by_solver),
        "max_gap_vs_solver": dict(case.max_gap_vs_solver),
        "notes": case.notes,
    }


def _row_for_exception(
    case: FrontierCase,
    test_data,
    solver: str,
    elapsed: float,
    error: Exception,
) -> dict[str, Any]:
    return _row_for_result(
        case=case,
        test_data=test_data,
        solver=solver,
        status="failed",
        char_accuracy=0.0,
        word_accuracy=0.0,
        elapsed=elapsed,
        artifact_path="",
        error=f"{type(error).__name__}: {error}",
        candidates=0,
    )


def _write_frontier_summaries(summary_jsonl: Path, summary_csv: Path, rows: list[dict[str, Any]]) -> None:
    evaluated = evaluate_frontier_rows(rows)
    summary_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_jsonl, "w", encoding="utf-8") as handle:
        for row in evaluated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if not evaluated:
        summary_csv.write_text("", encoding="utf-8")
        return
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluated[0].keys()))
        writer.writeheader()
        writer.writerows(evaluated)


def _print_result(row: dict[str, Any]) -> None:
    word = f"{row['word_accuracy']:.1%}" if row["word_boundaries"] else "N/A"
    verdict = "pass" if row.get("meets_expectations") else "watch"
    print(
        f"    -> {row['status']}  char={row['char_accuracy']:.1%}  "
        f"word={word}  {row['elapsed_seconds']:.1f}s  [{verdict}]"
    )
    if row["error"]:
        print(f"       error: {row['error']}")
    failures = row.get("expectation_failures") or []
    if failures:
        print(f"       expectation_failures: {'; '.join(failures)}")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No rows produced.")
        return
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["frontier_class"]), str(row["solver"])), []).append(row)
    print("=" * 96)
    print("FRONTIER SUITE SUMMARY")
    print("=" * 96)
    print(
        f"{'Class':<14} {'Solver':<22} {'N':>4} {'Pass':>6} {'Avg char':>9} "
        f"{'Avg time':>9}"
    )
    print("-" * 96)
    for (frontier_class, solver), group in sorted(groups.items()):
        avg_char = sum(float(row["char_accuracy"]) for row in group) / len(group)
        avg_time = sum(float(row["elapsed_seconds"]) for row in group) / len(group)
        passed = sum(1 for row in group if row.get("meets_expectations"))
        print(
            f"{frontier_class:<14} {solver:<22} {len(group):>4} {passed:>6} "
            f"{avg_char:>8.1%} {avg_time:>8.1f}s"
        )


if __name__ == "__main__":
    main()
