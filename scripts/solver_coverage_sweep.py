#!/usr/bin/env python
"""No-LLM solver coverage sweep over a generated multi-family benchmark.

Runs the existing AutomatedBenchmarkRunner (no LLM) against every case in a
loader-compatible benchmark tree and reports a per-family coverage matrix
{family -> char accuracy, status}. Unsupported families fail gracefully
(recorded, never crash the sweep). Purpose: find which of the generated
cipher families our current automated solver already cracks vs which need
new algorithmic solvers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from automated.runner import AutomatedBenchmarkRunner  # noqa: E402
from benchmark.loader import BenchmarkLoader  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-root", required=True)
    ap.add_argument("--split", default="all_generated.jsonl")
    ap.add_argument("--per-family", type=int, default=1,
                    help="Cap cases run per family (default 1).")
    ap.add_argument("--out", default=None, help="Write the JSON matrix here.")
    args = ap.parse_args()

    loader = BenchmarkLoader(args.benchmark_root)
    tests = loader.load_tests(args.split)
    runner = AutomatedBenchmarkRunner(artifact_dir="artifacts/coverage_sweep")

    seen: dict[str, int] = defaultdict(int)
    rows: list[dict] = []
    t0 = time.time()
    for t in tests:
        fam = t.cipher_system
        if seen[fam] >= args.per_family:
            continue
        seen[fam] += 1
        td = loader.load_test_data(t)
        rt0 = time.time()
        try:
            res = runner.run_test(td, language=td.plaintext_language or None)
            row = {"family": fam, "test_id": t.test_id, "status": res.status,
                   "char_accuracy": round(res.char_accuracy, 4),
                   "elapsed_s": round(time.time() - rt0, 1)}
        except Exception as exc:  # noqa: BLE001 — the point is to find gaps
            row = {"family": fam, "test_id": t.test_id, "status": "crash",
                   "char_accuracy": 0.0, "error": f"{type(exc).__name__}: {exc}"[:160],
                   "elapsed_s": round(time.time() - rt0, 1)}
        rows.append(row)
        print(f"  {fam:26} {row['status']:10} char={row['char_accuracy']:.3f} "
              f"({row['elapsed_s']}s)" + (f"  !{row.get('error','')}" if row.get("error") else ""),
              flush=True)

    rows.sort(key=lambda r: (-r["char_accuracy"], r["family"]))
    solved = [r for r in rows if r["char_accuracy"] >= 0.90]
    partial = [r for r in rows if 0.40 <= r["char_accuracy"] < 0.90]
    gap = [r for r in rows if r["char_accuracy"] < 0.40]
    print("\n=== COVERAGE MATRIX ({} families, {:.0f}s) ===".format(len(rows), time.time() - t0))
    print(f"SOLVED (>=90%): {len(solved)}  |  PARTIAL (40-90%): {len(partial)}  |  GAP (<40%): {len(gap)}")
    print("\n-- solved --")
    for r in solved:  print(f"  {r['family']:26} {r['char_accuracy']:.3f}")
    print("-- partial --")
    for r in partial: print(f"  {r['family']:26} {r['char_accuracy']:.3f}")
    print("-- gap (need new solver) --")
    for r in gap:     print(f"  {r['family']:26} {r['char_accuracy']:.3f}  [{r['status']}]")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows,
            "summary": {"solved": len(solved), "partial": len(partial), "gap": len(gap)}},
            indent=2), encoding="utf-8")
        print(f"\nmatrix -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
