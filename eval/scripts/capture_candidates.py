#!/usr/bin/env python3
"""Capture transform-search candidate lists for a population.

For each entry in a population JSONL file, runs screen_transform_candidates
(no LLM) and writes the full top-N result to a per-case JSON file.  Already-
captured cases are skipped unless --force is given.

This is the expensive-but-cacheable step.  Run it once per population; all
subsequent strategy evaluation and labeling uses the cached outputs.

Usage
-----
    PYTHONPATH=src:eval python eval/scripts/capture_candidates.py \\
        --population eval/artifacts/populations/default.jsonl \\
        --artifact-dir eval/artifacts \\
        --top-n 200 --profile small --workers 4 --verbose

The per-case files land at:
    <artifact-dir>/candidates/<case_id>.json
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _capture_one(args_tuple: tuple) -> tuple[str, bool, str]:
    """Worker function — returns (case_id, success, message)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    entry_dict, artifact_dir_str, top_n, profile, force = args_tuple
    from triage.types import PopulationEntry
    from triage.capture import candidate_path, capture_case

    entry = PopulationEntry.from_dict(entry_dict)
    out_path = candidate_path(entry.case_id, Path(artifact_dir_str))

    if not force and out_path.exists():
        return entry.case_id, True, "cached"

    try:
        captured = capture_case(entry, top_n=top_n, profile=profile)
        captured.save(out_path)
        return entry.case_id, True, f"{len(captured.candidates)} candidates"
    except Exception as exc:  # noqa: BLE001
        return entry.case_id, False, f"ERROR: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture transform-search candidate lists for a population."
    )
    parser.add_argument(
        "--population", "-p",
        type=Path,
        required=True,
        help="Population JSONL file (from build_population.py).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("eval/artifacts"),
        help="Root artifact directory (default: eval/artifacts).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Number of candidates to capture per case (default: 200).",
    )
    parser.add_argument(
        "--profile",
        default="small",
        choices=["small", "medium", "wide"],
        help="Transform search profile (default: small).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes (default: 4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-capture even if a cached file exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N cases (useful for smoke tests).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    from triage.population import load_population
    entries = load_population(args.population)
    if args.limit:
        entries = entries[: args.limit]

    total = len(entries)
    print(f"Capturing {total} cases (workers={args.workers}, top_n={args.top_n})")

    task_args = [
        (e.to_dict(), str(args.artifact_dir), args.top_n, args.profile, args.force)
        for e in entries
    ]

    done = 0
    errors = 0
    cached = 0
    t0 = time.time()

    if args.workers <= 1:
        for task in task_args:
            case_id, ok, msg = _capture_one(task)
            done += 1
            if not ok:
                errors += 1
            if msg == "cached":
                cached += 1
            if args.verbose or not ok:
                print(f"  [{done}/{total}] {case_id}: {msg}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_capture_one, t): t[0]["case_id"] for t in task_args}
            for fut in as_completed(futures):
                case_id, ok, msg = fut.result()
                done += 1
                if not ok:
                    errors += 1
                if msg == "cached":
                    cached += 1
                if args.verbose or not ok:
                    print(f"  [{done}/{total}] {case_id}: {msg}")

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s: {done - errors} OK, {errors} errors, "
        f"{cached} cached, {done - cached} newly captured"
    )
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
