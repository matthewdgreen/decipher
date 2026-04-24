#!/usr/bin/env python3
"""Run chunkable no-LLM automated parity matrices.

This harness keeps solving no-LLM. Synthetic tests are built only from
plaintext already present in ``testgen_cache`` by default; cache misses are
skipped unless ``--fail-on-cache-miss`` is requested. If ``--allow-generate``
is set, cache misses may use an LLM to generate plaintext, but deciphering
still runs through automated non-agentic solvers only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from automated.runner import AutomatedBenchmarkRunner
from benchmark.loader import BenchmarkLoader, TestData
from benchmark.scorer import has_word_boundaries
from external_baselines import ExternalBaselineConfig, run_external_baseline
from testgen.builder import build_test_case
from testgen.cache import PlaintextCache
from testgen.spec import DifficultyPreset, TestSpec


DEFAULT_PRESET_SEEDS = {
    DifficultyPreset.TINY: 1,
    DifficultyPreset.MEDIUM: 3,
    DifficultyPreset.HARD: 4,
    DifficultyPreset.HARDEST: 5,
}

FAMILY_PARAMS = {
    "simple-wb": {"word_boundaries": True, "homophonic": False},
    "simple-nb": {"word_boundaries": False, "homophonic": False},
    "homophonic-nb": {"word_boundaries": False, "homophonic": True},
}


@dataclass(frozen=True)
class MatrixCase:
    source: str
    family: str
    test_data: TestData
    spec: TestSpec | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run no-LLM Decipher/external automated parity tests in chunks.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=["decipher", "external"],
        default=["decipher"],
        help="Which solver lanes to run. Use both for direct parity comparison.",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=[p.value for p in DifficultyPreset],
        help="Synthetic preset cases to include.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(FAMILY_PARAMS),
        help="Custom synthetic families to include.",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        help="Approximate word lengths for custom synthetic families.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Seed list/ranges, e.g. '1-20,42'. Defaults to preset seeds for presets.",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--cache-dir", default="testgen_cache")
    parser.add_argument(
        "--fail-on-cache-miss",
        action="store_true",
        help="Fail instead of skipping synthetic specs whose plaintext is not cached.",
    )
    parser.add_argument(
        "--allow-generate",
        action="store_true",
        help=(
            "Allow LLM plaintext generation on synthetic cache miss. "
            "Solving still runs in automated no-LLM mode."
        ),
    )
    parser.add_argument(
        "--benchmark-root",
        default="../cipher_benchmark/benchmark",
        help="Benchmark checkout for --benchmark-split cases.",
    )
    parser.add_argument(
        "--benchmark-split",
        action="append",
        default=[],
        help="Benchmark split JSONL to include. May be passed more than once.",
    )
    parser.add_argument("--track", default="transcription2plaintext")
    parser.add_argument("--external-config", help="External baseline config JSON.")
    parser.add_argument(
        "--oracle-select",
        action="store_true",
        help="For external solvers only, select best emitted candidate by score.",
    )
    parser.add_argument("--artifact-dir", default="artifacts/automated_parity_matrix")
    parser.add_argument("--summary-jsonl", help="Summary JSONL output path.")
    parser.add_argument("--summary-csv", help="Summary CSV output path.")
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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if "external" in args.solvers and not args.external_config:
        parser.error("--external-config is required when --solvers includes external")
    if args.shard_count < 1:
        parser.error("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, shard-count)")

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_jsonl = Path(args.summary_jsonl) if args.summary_jsonl else artifact_dir / "summary.jsonl"
    summary_csv = Path(args.summary_csv) if args.summary_csv else artifact_dir / "summary.csv"

    cases = build_cases(args)
    selected = select_chunk(cases, args.offset, args.limit, args.shard_index, args.shard_count)

    print(f"Matrix cases: {len(cases)} total, {len(selected)} selected")
    print(f"Solvers: {', '.join(args.solvers)}")
    print(f"Artifacts: {artifact_dir}")
    print(f"Summary JSONL: {summary_jsonl}")
    print(f"Summary CSV: {summary_csv}")
    print("Mode: automated no-LLM")
    print()

    if args.dry_run:
        for idx, case in enumerate(selected, start=1):
            print(f"[{idx}/{len(selected)}] {case.test_data.test.test_id}  {case.family}")
        return

    rows: list[dict[str, Any]] = []
    external_configs = load_external_configs(args.external_config, args.oracle_select) if "external" in args.solvers else []
    decipher_runner = AutomatedBenchmarkRunner(
        artifact_dir=artifact_dir / "decipher",
        homophonic_budget=args.homophonic_budget,
        homophonic_refinement=args.homophonic_refinement,
    )

    try:
        for idx, case in enumerate(selected, start=1):
            td = case.test_data
            print(f"[{idx}/{len(selected)}] {td.test.test_id} — {case.family}")
            if "decipher" in args.solvers:
                print("  [decipher-automated] running...")
                t0 = time.time()
                try:
                    result = decipher_runner.run_test(td, language=case.spec.language if case.spec else None)
                    elapsed = time.time() - t0
                    row = row_for_result(
                        case=case,
                        solver="decipher-automated",
                        status=result.status,
                        char_accuracy=result.char_accuracy,
                        word_accuracy=result.word_accuracy,
                        elapsed=elapsed,
                        artifact_path=getattr(result, "artifact_path", ""),
                        error=getattr(result, "error_message", ""),
                        candidates=1,
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.time() - t0
                    row = row_for_exception(
                        case=case,
                        solver="decipher-automated",
                        elapsed=elapsed,
                        error=exc,
                    )
                rows.append(row)
                print_result(row)
                write_jsonl(summary_jsonl, rows)
                write_csv(summary_csv, rows)

            for config in external_configs:
                print(f"  [{config.name}] running...")
                t0 = time.time()
                try:
                    result = run_external_baseline(
                        td,
                        config,
                        artifact_dir=artifact_dir / "external",
                    )
                    row = row_for_result(
                        case=case,
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
                    elapsed = time.time() - t0
                    row = row_for_exception(
                        case=case,
                        solver=config.name,
                        elapsed=elapsed,
                        error=exc,
                    )
                rows.append(row)
                print_result(row)
                write_jsonl(summary_jsonl, rows)
                write_csv(summary_csv, rows)
            print()
    finally:
        write_jsonl(summary_jsonl, rows)
        write_csv(summary_csv, rows)

    print_summary(rows)


def build_cases(args: argparse.Namespace) -> list[MatrixCase]:
    cache = PlaintextCache(args.cache_dir)
    cases: list[MatrixCase] = []
    specs = synthetic_specs(args)
    for spec, family in specs:
        if cache.get(spec) is None:
            msg = f"cache miss for {spec}"
            if args.fail_on_cache_miss:
                raise SystemExit(msg)
            if not args.allow_generate:
                print(f"Skipping synthetic spec: {msg}", file=sys.stderr)
                continue
            print(
                f"Generating plaintext for synthetic spec: {spec} "
                "(LLM generation only; solver remains no-LLM)",
                file=sys.stderr,
            )
            test_data = build_test_case(spec, cache, api_key=_generation_api_key())
        else:
            test_data = build_test_case(spec, cache, api_key="")
        cases.append(MatrixCase(source="synthetic", family=family, test_data=test_data, spec=spec))

    if args.benchmark_split:
        loader = BenchmarkLoader(args.benchmark_root)
        for split in args.benchmark_split:
            tests = loader.load_tests(split, track=args.track)
            for test in tests:
                test_data = loader.load_test_data(test)
                cases.append(MatrixCase(
                    source=f"benchmark:{Path(split).name}",
                    family=test.cipher_system or "benchmark",
                    test_data=test_data,
                    spec=None,
                ))
    return cases


def synthetic_specs(args: argparse.Namespace) -> list[tuple[TestSpec, str]]:
    out: list[tuple[TestSpec, str]] = []
    seeds = parse_int_ranges(args.seeds) if args.seeds else []

    for preset_name in args.presets or []:
        preset = DifficultyPreset(preset_name)
        preset_seed_values = seeds or [DEFAULT_PRESET_SEEDS[preset]]
        for seed in preset_seed_values:
            spec = TestSpec.from_preset(preset, language=args.language, seed=seed)
            out.append((spec, f"preset:{preset.value}"))

    if args.families:
        if not args.lengths:
            raise SystemExit("--lengths is required when --families is used")
        if not seeds:
            raise SystemExit("--seeds is required when --families is used")
        for family in args.families:
            params = FAMILY_PARAMS[family]
            for length in args.lengths:
                for seed in seeds:
                    spec = TestSpec(
                        language=args.language,
                        approx_length=length,
                        word_boundaries=params["word_boundaries"],
                        homophonic=params["homophonic"],
                        seed=seed,
                    )
                    out.append((spec, family))

    if not out and not args.benchmark_split:
        for preset in DifficultyPreset:
            spec = TestSpec.from_preset(
                preset,
                language=args.language,
                seed=DEFAULT_PRESET_SEEDS[preset],
            )
            out.append((spec, f"preset:{preset.value}"))
    return out


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
        "macOS Keychain entry service=decipher account=anthropic_api_key. "
        "Omit --allow-generate to skip cache misses."
    )


def parse_int_ranges(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(int(part))
    return out


def select_chunk(
    cases: list[MatrixCase],
    offset: int,
    limit: int | None,
    shard_index: int,
    shard_count: int,
) -> list[MatrixCase]:
    selected = [case for idx, case in enumerate(cases) if idx % shard_count == shard_index]
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def load_external_configs(path: str | None, oracle_select: bool) -> list[ExternalBaselineConfig]:
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


def row_for_result(
    case: MatrixCase,
    solver: str,
    status: str,
    char_accuracy: float,
    word_accuracy: float,
    elapsed: float,
    artifact_path: str,
    error: str,
    candidates: int,
) -> dict[str, Any]:
    td = case.test_data
    spec = case.spec
    return {
        "test_id": td.test.test_id,
        "source": case.source,
        "family": case.family,
        "cipher_system": td.test.cipher_system,
        "language": spec.language if spec else "",
        "approx_length": spec.approx_length if spec else "",
        "word_boundaries": has_word_boundaries(td.plaintext),
        "homophonic": spec.homophonic if spec else ("homophonic" in td.test.cipher_system.lower()),
        "seed": spec.seed if spec else "",
        "solver": solver,
        "status": status,
        "char_accuracy": round(char_accuracy, 6),
        "word_accuracy": round(word_accuracy, 6),
        "elapsed_seconds": round(elapsed, 3),
        "candidates": candidates,
        "artifact_path": artifact_path,
        "error": error,
    }


def row_for_exception(
    case: MatrixCase,
    solver: str,
    elapsed: float,
    error: Exception,
) -> dict[str, Any]:
    return row_for_result(
        case=case,
        solver=solver,
        status="failed",
        char_accuracy=0.0,
        word_accuracy=0.0,
        elapsed=elapsed,
        artifact_path="",
        error=f"{type(error).__name__}: {error}",
        candidates=0,
    )


def print_result(row: dict[str, Any]) -> None:
    word = f"{row['word_accuracy']:.1%}" if row["word_boundaries"] else "N/A"
    print(
        f"    -> {row['status']}  char={row['char_accuracy']:.1%}  "
        f"word={word}  {row['elapsed_seconds']:.1f}s  artifact={row['artifact_path']}"
    )
    if row["error"]:
        print(f"       error: {row['error']}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No rows produced.")
        return
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["family"], row["solver"]), []).append(row)

    print("=" * 88)
    print("AUTOMATED PARITY MATRIX SUMMARY")
    print("=" * 88)
    print(f"{'Family':<20} {'Solver':<22} {'N':>4} {'Avg char':>9} {'Min char':>9} {'Avg time':>9}")
    print("-" * 88)
    for (family, solver), group in sorted(groups.items()):
        avg_char = sum(r["char_accuracy"] for r in group) / len(group)
        min_char = min(r["char_accuracy"] for r in group)
        avg_time = sum(r["elapsed_seconds"] for r in group) / len(group)
        print(f"{family:<20} {solver:<22} {len(group):>4} {avg_char:>8.1%} {min_char:>8.1%} {avg_time:>8.1f}s")


if __name__ == "__main__":
    main()
