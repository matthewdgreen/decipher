"""M4 acceptance-3: real experiment-queue overlap demonstration (LOCAL compute).

Queues one transform screen (config={"transform_search": "screen"}) and one
homophonic anneal (config={}, the default path) CONCURRENTLY at S=2, then runs
the same pair SEQUENTIALLY in sync mode. The GATE (F8) is INTERVAL OVERLAP of
the two records' started_at/completed_at intervals; the wall-clock ratio is
reported informationally. No LLM spend — the automated solver runs locally.

Run:  PYTHONPATH=src .venv/bin/python scripts/demo_experiment_overlap.py
"""
from __future__ import annotations

import os
import random
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark.loader import parse_canonical_transcription
from investigation.experiments import (
    ExperimentQueue,
    dispatch_experiment_submit,
)
from investigation.state import InvestigationState
from workspace import Workspace


_PLAINTEXT = (
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG WHILE THE CLEVER CAT WATCHES "
    "FROM A HIGH STONE WALL NEAR THE OLD MILL WHERE THE RIVER BENDS SHARPLY "
    "TOWARD THE QUIET VILLAGE THAT SLEEPS BENEATH A PALE AUTUMN MOON AND EVERY "
    "LANTERN GLOWS SOFTLY AGAINST THE GATHERING MIST OF THE LONG COLD NIGHT"
)


def _synthetic_homophonic_cipher():
    """Build a homophonic synthetic cipher (alphabet > 26 -> homophonic)."""
    rng = random.Random(7)
    words = _PLAINTEXT.split()
    letters = sorted(set("".join(words)))
    homo: dict[str, list[str]] = {}
    counter = 0
    for letter in letters:
        n = rng.choice([1, 2, 3])
        homo[letter] = [f"S{counter + i:03d}" for i in range(n)]
        counter += n

    def enc(word: str) -> str:
        return " ".join(rng.choice(homo[c]) for c in word)

    canonical = " | ".join(enc(w) for w in words)
    return parse_canonical_transcription(canonical)


def _fresh_state():
    ct = _synthetic_homophonic_cipher()
    return ct, InvestigationState(workspace=Workspace(ct), language="en")


def _submit(queue, state, config, turn=1):
    return dispatch_experiment_submit(
        queue, state, state.workspace, SimpleNamespace(_model_variant=None),
        {"type": "automated_solver", "branch": "main", "config": config}, turn,
    )


def _warm_caches(state):
    """Warm parent-process caches (ngram/pattern/zenith binary model, in-process
    transform kernel) BEFORE the timed phases so the informational ratio is not
    biased by the concurrent phase paying the cold-cache cost first (F-3)."""
    t0 = time.time()
    qw = ExperimentQueue(synchronous=True)
    _submit(qw, state, {"transform_search": "screen"})
    _submit(qw, state, {})
    print(f"(warm-up sync pass: {time.time() - t0:.3f}s — caches primed, not timed)")


def main() -> int:
    ct, state = _fresh_state()
    print(f"Synthetic homophonic cipher: alphabet={ct.alphabet.size} symbols, "
          f"{len(ct.tokens)} tokens, {len(ct.words)} words")

    _warm_caches(state)
    ct, state = _fresh_state()  # fresh queue-less state for the timed phases

    # --- concurrent (async, S=2) ---
    qa = ExperimentQueue(synchronous=False, slots=2)
    print(f"Arbiter: W={qa.W} S={qa.S} I={qa.I}")
    t0 = time.time()
    r_screen = _submit(qa, state, {"transform_search": "screen"})
    r_anneal = _submit(qa, state, {})
    assert qa.wait_settled(r_screen["experiment_id"], timeout=600)
    assert qa.wait_settled(r_anneal["experiment_id"], timeout=600)
    qa.poll(state, 2)
    concurrent_wall = time.time() - t0

    recs = {r["experiment_id"]: r for r in state.experiment_queue}
    screen = recs[r_screen["experiment_id"]]
    anneal = recs[r_anneal["experiment_id"]]
    for label, rec in (("transform_screen", screen), ("homophonic_anneal", anneal)):
        print(f"  [{label}] status={rec['status']} "
              f"started={rec['started_at']:.3f} completed={rec['completed_at']:.3f} "
              f"elapsed={rec['elapsed_seconds']}s inner_workers={rec['inner_workers']}")

    overlap = min(screen["completed_at"], anneal["completed_at"]) - max(
        screen["started_at"], anneal["started_at"])
    print(f"  INTERVAL OVERLAP = {overlap:.3f}s  (GATE: must be > 0)")

    # --- sequential (sync) ---
    _ct2, state2 = _fresh_state()
    qs = ExperimentQueue(synchronous=True)
    t0 = time.time()
    _submit(qs, state2, {"transform_search": "screen"})
    _submit(qs, state2, {})
    sequential_wall = time.time() - t0
    seq_elapsed = [r["elapsed_seconds"] for r in state2.experiment_queue]

    ratio = concurrent_wall / sequential_wall if sequential_wall else float("nan")
    print()
    print("=== summary ===")
    print(f"W/S/I                 : {qa.W}/{qa.S}/{qa.I}")
    print(f"per-exp elapsed (conc): screen={screen['elapsed_seconds']}s "
          f"anneal={anneal['elapsed_seconds']}s")
    print(f"per-exp elapsed (seq) : {seq_elapsed}")
    print(f"concurrent wall-clock : {concurrent_wall:.3f}s")
    print(f"sequential wall-clock : {sequential_wall:.3f}s")
    print(f"ratio (conc/seq)      : {ratio:.3f}  (informational, not gated)")
    print(f"interval overlap      : {overlap:.3f}s")

    if overlap <= 0:
        print("GATE FAILED: intervals did not overlap.")
        return 1
    print("GATE PASSED: the two experiments ran concurrently (intervals overlap).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
