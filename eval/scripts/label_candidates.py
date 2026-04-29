#!/usr/bin/env python3
"""Label captured candidates with ground-truth tiers.

For each captured case, computes:
  - transform_correct  (cheap: hash comparison, no solver)
  - readable_now       (solver: char accuracy vs plaintext)
  - rescuable          (solver result >= threshold)
  - decoded_text       (solver's best decryption)

Labels are written back into the same candidate JSON files in-place.
Already-labeled candidates within budget are not re-labeled unless --force.

Usage
-----
    PYTHONPATH=src:eval python eval/scripts/label_candidates.py \\
        --population eval/artifacts/populations/default.jsonl \\
        --artifact-dir eval/artifacts \\
        --top-rank-budget 25 --tail-sample 10 \\
        --homophonic-budget screen \\
        --workers 2 --verbose

Note: --workers > 1 uses process-level parallelism.  Keep it modest since
each worker may run a homophonic anneal (multi-threaded internally).
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _label_one(args_tuple: tuple) -> tuple[str, bool, str]:
    """Worker: label one case and save it.  Returns (case_id, ok, message)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    (
        case_id, artifact_dir_str, top_rank_budget, tail_sample,
        homophonic_budget, rescuable_threshold, force
    ) = args_tuple

    from triage.capture import candidate_path
    from triage.labeler import label_case
    from triage.types import CapturedCase

    path = candidate_path(case_id, Path(artifact_dir_str))
    if not path.exists():
        return case_id, False, "ERROR: candidate file not found"

    case = CapturedCase.load(path)

    # Skip if already fully labeled within budget and not forcing.
    if not force:
        labeled = sum(1 for c in case.candidates if c.rescuable is not None)
        if labeled >= min(top_rank_budget, len(case.candidates)):
            return case_id, True, f"already labeled ({labeled} candidates)"

    try:
        labeled_case = label_case(
            case,
            top_rank_budget=top_rank_budget,
            tail_sample=tail_sample,
            homophonic_budget=homophonic_budget,
            rescuable_threshold=rescuable_threshold,
        )
        labeled_case.save(path)
        rescuable = sum(1 for c in labeled_case.candidates if c.rescuable is True)
        newly_labeled = sum(1 for c in labeled_case.candidates if c.rescuable is not None)
        return case_id, True, f"{newly_labeled} labeled, {rescuable} rescuable"
    except Exception as exc:  # noqa: BLE001
        return case_id, False, f"ERROR: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label captured transform-search candidates with ground-truth tiers."
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
        "--top-rank-budget",
        type=int,
        default=25,
        help="Always label this many top-ranked candidates per case (default: 25).",
    )
    parser.add_argument(
        "--tail-sample",
        type=int,
        default=10,
        help="Random sample from remaining candidates to label (default: 10).",
    )
    parser.add_argument(
        "--homophonic-budget",
        default="screen",
        choices=["screen", "full"],
        help="Solver budget for homophonic cases (default: screen).",
    )
    parser.add_argument(
        "--rescuable-threshold",
        type=float,
        default=0.70,
        help="Minimum char accuracy to call a candidate rescuable (default: 0.70).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default: 1 — each solver may be threaded).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-label even if the case already has labels.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    from triage.population import load_population
    entries = load_population(args.population)
    if args.limit:
        entries = entries[: args.limit]

    total = len(entries)
    print(f"Labeling {total} cases (workers={args.workers})")

    task_args = [
        (
            e.case_id,
            str(args.artifact_dir),
            args.top_rank_budget,
            args.tail_sample,
            args.homophonic_budget,
            args.rescuable_threshold,
            args.force,
        )
        for e in entries
    ]

    done = errors = skipped = 0
    t0 = time.time()

    if args.workers <= 1:
        for task in task_args:
            case_id, ok, msg = _label_one(task)
            done += 1
            if not ok:
                errors += 1
            if "already labeled" in msg:
                skipped += 1
            if args.verbose or not ok:
                print(f"  [{done}/{total}] {case_id}: {msg}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_label_one, t): t[0] for t in task_args}
            for fut in as_completed(futures):
                case_id, ok, msg = fut.result()
                done += 1
                if not ok:
                    errors += 1
                if "already labeled" in msg:
                    skipped += 1
                if args.verbose or not ok:
                    print(f"  [{done}/{total}] {case_id}: {msg}")

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s: {done - errors - skipped} labeled, "
        f"{skipped} skipped (already done), {errors} errors"
    )
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
