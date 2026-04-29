#!/usr/bin/env python3
"""Evaluate all triage strategies against labeled candidates.

Loads all captured-and-labeled case files for a population, runs each
registered strategy over the frozen candidate lists, and writes a CSV +
markdown summary report.

This step is fast (seconds for non-LLM strategies) and can be re-run freely
as new strategies are added or thresholds are adjusted.

Usage
-----
    PYTHONPATH=src:eval python eval/scripts/evaluate_strategies.py \\
        --population eval/artifacts/populations/default.jsonl \\
        --artifact-dir eval/artifacts \\
        --out-dir eval/artifacts/reports/run_001 \\
        --k 1 3 10 25 \\
        --strategies baseline_ngram score_only dict_rate_weighted cluster_dedupe mmr_diversity \\
        --verbose

Omit --strategies to run all registered strategies.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate triage strategies over labeled candidate files."
    )
    parser.add_argument(
        "--population", "-p",
        type=Path,
        required=True,
        help="Population JSONL file.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("eval/artifacts"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for report output (default: <artifact-dir>/reports/latest).",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 3, 10, 25],
        metavar="K",
        help="K values for recall@K (default: 1 3 10 25).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Strategies to evaluate (default: all registered).",
    )
    parser.add_argument(
        "--baseline",
        default="baseline_ngram",
        help="Strategy name to use as baseline for regression comparison.",
    )
    parser.add_argument(
        "--min-labeled",
        type=int,
        default=1,
        help="Skip cases with fewer than this many labeled candidates (default: 1).",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Human-readable run identifier for the report header.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-text notes to embed in the markdown report.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Import after path setup.
    # Eagerly import non_llm to trigger @register decorators.
    import triage.strategies.non_llm  # noqa: F401
    from triage.strategies.base import all_strategies, get_strategy
    from triage.capture import candidate_path
    from triage.metrics import evaluate_all
    from triage.population import load_population
    from triage.report import write_report
    from triage.types import CapturedCase

    # Resolve output directory.
    out_dir = args.out_dir or (args.artifact_dir / "reports" / "latest")

    # Load population.
    entries = load_population(args.population)
    if args.verbose:
        print(f"Population: {len(entries)} entries from {args.population}")

    # Load candidate files.
    cases = []
    missing = 0
    for entry in entries:
        path = candidate_path(entry.case_id, args.artifact_dir)
        if not path.exists():
            missing += 1
            continue
        case = CapturedCase.load(path)
        labeled = sum(1 for c in case.candidates if c.rescuable is not None)
        if labeled < args.min_labeled:
            continue
        cases.append(case)

    if missing and args.verbose:
        print(f"  {missing} entries have no candidate file (run capture first)")
    if not cases:
        print("No labeled cases found. Run capture_candidates.py and label_candidates.py first.")
        sys.exit(1)
    print(f"Loaded {len(cases)} labeled cases")

    # Resolve strategies.
    if args.strategies:
        strategies = [get_strategy(name) for name in args.strategies]
    else:
        strategies = all_strategies()

    # Ensure baseline is always included for regression comparison.
    strategy_names = {s.name for s in strategies}
    if args.baseline not in strategy_names:
        try:
            strategies.append(get_strategy(args.baseline))
        except KeyError:
            pass

    if args.verbose:
        print(f"Strategies: {[s.name for s in strategies]}")

    # Evaluate.
    t0 = time.time()
    rows = evaluate_all(
        strategies=strategies,
        cases=cases,
        k_values=args.k,
        baseline_name=args.baseline,
    )
    elapsed = time.time() - t0

    if args.verbose:
        print(f"Evaluation done in {elapsed:.2f}s — {len(rows)} metric rows")

    # Write report.
    md_path = write_report(rows, out_dir=out_dir, run_id=args.run_id, notes=args.notes)
    csv_path = out_dir / "metrics.csv"
    print(f"Report: {md_path}")
    print(f"CSV:    {csv_path}")

    # Print overall recall@10 summary to stdout.
    summary = [
        r for r in rows
        if r.language == "all" and r.transform_family == "all"
        and r.columns == "all" and r.k == 10
    ]
    if summary:
        print("\nOverall recall@10 (all cases):")
        print(f"  {'strategy':<30}  {'recall@10':>9}  {'MRR':>6}  {'regression':>10}  n_labeled")
        for r in sorted(summary, key=lambda r: r.recall_at_k, reverse=True):
            print(
                f"  {r.strategy_name:<30}  {r.recall_at_k:>9.3f}  "
                f"{r.mrr_rescuable:>6.3f}  {r.regression_rate:>10.3f}  {r.n_labeled_cases}"
            )


if __name__ == "__main__":
    main()
