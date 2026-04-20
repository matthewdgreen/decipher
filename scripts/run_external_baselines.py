#!/usr/bin/env python3
"""Run configured external solver baselines on synthetic Decipher tests."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.scorer import format_alignment, format_char_diff, has_word_boundaries
from external_baselines import ExternalBaselineConfig, run_external_baseline
from testgen.builder import build_test_case
from testgen.cache import PlaintextCache
from testgen.spec import DifficultyPreset, TestSpec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run external automatic cipher solvers against synthetic tests.",
    )
    parser.add_argument("--config", required=True, help="JSON config with a 'solvers' list")
    parser.add_argument(
        "--preset", "-p",
        nargs="+",
        choices=[p.value for p in DifficultyPreset],
        default=["hardest"],
        help="Synthetic preset(s) to run; default: hardest",
    )
    parser.add_argument("--cache-dir", default="testgen_cache")
    parser.add_argument("--artifact-dir", default="artifacts/external_baselines")
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument(
        "--allow-generate",
        action="store_true",
        help="Allow LLM plaintext generation on cache miss. Otherwise cache misses fail.",
    )
    parser.add_argument(
        "--oracle-select",
        action="store_true",
        help="Select the best emitted candidate by ground-truth score. Useful for diagnostics, not fair headline benchmarking.",
    )
    args = parser.parse_args()

    configs = _load_configs(Path(args.config))
    if args.oracle_select:
        configs = [
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

    cache = PlaintextCache(args.cache_dir)
    if args.flush_cache:
        n = cache.flush()
        print(f"Flushed {n} cache entries.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    specs = [
        TestSpec.from_preset(DifficultyPreset(p), language="en", seed=_default_seed_for_preset(DifficultyPreset(p)))
        for p in args.preset
    ]

    results = []
    for spec in specs:
        if cache.get(spec) is None and not args.allow_generate:
            raise SystemExit(
                f"Plaintext cache miss for {spec}. Re-run with --allow-generate or run the Decipher suite once first."
            )
        test_data = build_test_case(spec, cache, api_key)
        print(f"\n{test_data.test.test_id} — {test_data.test.description}")
        print(f"Cipher symbols: {len(set(test_data.canonical_transcription.replace('|', ' ').split()))}")

        for config in configs:
            print(f"  [{config.name}] running...")
            result = run_external_baseline(test_data, config, artifact_dir=args.artifact_dir)
            results.append(result)
            word = f"{result.word_accuracy:.1%}" if has_word_boundaries(result.ground_truth) else "N/A"
            oracle = " oracle" if result.oracle_selection else ""
            print(
                f"    → {result.status}  char={result.char_accuracy:.1%}  word={word}  "
                f"{result.elapsed:.1f}s  candidates={result.candidates_considered}{oracle}"
            )
            if result.error:
                print(f"      error: {result.error}")
            print(f"      artifact: {result.artifact_path}")

    _print_summary(results)


def _load_configs(path: Path) -> list[ExternalBaselineConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    solvers = data.get("solvers", [])
    if not isinstance(solvers, list) or not solvers:
        raise ValueError(f"{path} must contain a non-empty 'solvers' list")
    return [ExternalBaselineConfig.from_dict(item) for item in solvers]


def _default_seed_for_preset(preset: DifficultyPreset) -> int:
    defaults = {
        DifficultyPreset.TINY: 1,
        DifficultyPreset.MEDIUM: 3,
        DifficultyPreset.HARD: 4,
        DifficultyPreset.HARDEST: 5,
    }
    return defaults[preset]


def _print_summary(results) -> None:
    if not results:
        return
    sep = "=" * 80
    print()
    print(sep)
    print("EXTERNAL BASELINE RESULTS")
    print(sep)
    print()
    print(f"  {'Test ID':<24} {'Solver':<18} {'Status':<10} {'Char%':>6} {'Word%':>6} {'Time':>7} {'Cand':>5}")
    print("  " + "-" * 82)
    for result in results:
        word = f"{result.word_accuracy:.1%}" if has_word_boundaries(result.ground_truth) else "N/A"
        print(
            f"  {result.test_id:<24} {result.solver:<18} {result.status:<10} "
            f"{result.char_accuracy:>5.1%} {word:>6} {result.elapsed:>6.1f}s {result.candidates_considered:>5}"
        )

    print()
    for result in results:
        print(sep)
        print(f"  {result.solver}  [{result.test_id}]  char={result.char_accuracy:.1%}")
        print(sep)
        if result.error:
            print(f"  Error: {result.error}")
        elif not result.decryption:
            print("  (no plaintext candidate extracted)")
        elif has_word_boundaries(result.ground_truth):
            print(format_alignment(result.decryption, result.ground_truth, max_words=40))
        else:
            print(format_char_diff(result.decryption, result.ground_truth, context=12))
        print()


if __name__ == "__main__":
    main()
