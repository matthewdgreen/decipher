#!/usr/bin/env python3
"""Build a population JSONL file from cached plaintexts.

Generates synthetic transposition+substitution and transposition+homophonic
test cases.  Never makes LLM API calls: cases whose plaintext is not already
in the testgen cache are silently skipped.

Usage
-----
    PYTHONPATH=src:eval python eval/scripts/build_population.py \\
        --output eval/artifacts/populations/default.jsonl \\
        --cache-dir testgen_cache \\
        --verbose

To pre-populate the plaintext cache for the default sweep, run one testgen
benchmark against any config that uses language=en, approx_length=200 — the
cache is shared across all testgen runs.
"""
import argparse
import sys
from pathlib import Path

# Allow running as: PYTHONPATH=src:eval python eval/scripts/build_population.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from triage.population import default_sweep, generate_population, PIPELINE_TEMPLATES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build transform-triage population JSONL from cached plaintexts."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("eval/artifacts/populations/default.jsonl"),
        help="Output path for the population JSONL file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("testgen_cache"),
        help="Directory of the testgen plaintext cache.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        metavar="LANG",
        help="Languages to include in the sweep (default: en).",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[200],
        metavar="N",
        help="Approximate plaintext word counts to include (default: 200).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        metavar="N",
        help="Number of seeds per cell (default: 10).",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["t_substitution", "t_homophonic"],
        choices=["t_substitution", "t_homophonic"],
        help="Transform families to include.",
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        default=None,
        choices=PIPELINE_TEMPLATES,
        help="Pipeline templates to include (default: all).",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        type=int,
        default=[7, 9, 11, 13],
        metavar="N",
        help="Grid column widths to sweep (default: 7 9 11 13).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress.",
    )
    args = parser.parse_args()

    # Build a custom sweep from CLI args.
    from triage.population import SweepEntry
    seeds = list(range(1, args.seeds + 1))
    templates = args.templates or [
        "route_columns_down",
        "route_boustrophedon",
        "ndown_1_1",
        "row_reversals",
        "whole_reverse",
    ]

    sweep = []
    for language in args.languages:
        for approx_length in args.lengths:
            for transform_family in args.families:
                homophonic = transform_family == "t_homophonic"
                for columns in args.columns:
                    for template in templates:
                        sweep.append(SweepEntry(
                            language=language,
                            transform_family=transform_family,
                            approx_length=approx_length,
                            columns=columns,
                            pipeline_template=template,
                            seeds=seeds,
                        ))

    if args.verbose:
        print(
            f"Sweep: {len(sweep)} cells × ~{args.seeds} seeds "
            f"= up to {len(sweep) * args.seeds} cases"
        )
        print(f"Output: {args.output}")
        print(f"Cache:  {args.cache_dir}")

    n = generate_population(
        sweep=sweep,
        output_path=args.output,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
    )
    print(f"Done: {n} cases written to {args.output}")


if __name__ == "__main__":
    main()
