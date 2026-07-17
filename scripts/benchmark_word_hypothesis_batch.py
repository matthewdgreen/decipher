#!/usr/bin/env python
"""Measure M5.3 batch word-repair speed without an LLM or ground truth.

The three arms use identical word hypotheses:

* ``cold_singletons`` recreates the pre-M5.3 cost shape by constructing a
  fresh executor (and therefore a fresh repair menu) for every probe;
* ``cached_singletons`` uses the current executor cache across repeated legacy
  singleton calls; and
* ``batch`` sends all hypotheses through ``hypothesis_test_words`` once.

The acceptance comparison is cold-singletons versus batch. Cached-singletons
is reported separately so cache gains are not incorrectly attributed to the
batch primitive.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor  # noqa: E402
from analysis import dictionary as dictionary_module  # noqa: E402
from investigation.actions import execute_composite  # noqa: E402
from models.alphabet import Alphabet  # noqa: E402
from models.cipher_text import CipherText  # noqa: E402
from workspace import Workspace  # noqa: E402


PLAINTEXT = (
    "THE QUICK BROWN FOXES JUMPED OVER THE LAZY SLEEPING HOUNDS WHILE "
    "SEVERAL PEOPLE WATCHED FROM THEIR HOUSES NEAR THE RIVER"
)


def _executor() -> WorkspaceToolExecutor:
    def encode(ch: str) -> str:
        return chr(ord("a") + ord(ch) - ord("A"))

    raw = " ".join("".join(encode(ch) for ch in word)
                   for word in PLAINTEXT.split())
    alphabet = Alphabet.from_text(raw, ignore_chars={" "})
    cipher = CipherText(raw=raw, alphabet=alphabet, separator=" ")
    workspace = Workspace(cipher)
    plain_alphabet = workspace.plaintext_alphabet
    for symbol in alphabet.symbols:
        workspace.set_mapping(
            "main", alphabet.id_for(symbol),
            plain_alphabet.id_for(chr(ord("A") + ord(symbol) - ord("a"))),
        )
    # One incorrect global mapping creates a realistic near-solved basin.
    workspace.set_mapping(
        "main", alphabet.id_for("r"), plain_alphabet.id_for("X")
    )
    executor = WorkspaceToolExecutor(
        workspace, "en",
        dictionary_module.load_word_set(
            dictionary_module.get_dictionary_path("en")
        ),
        [], {}, declaration_policy=NoGatesPolicy(),
    )
    executor.set_iteration(1)
    return executor


def _run_single(executor: WorkspaceToolExecutor, hypothesis: dict[str, Any]) -> None:
    execute_composite(
        "hypothesis_test_word",
        {"branch": "main", **hypothesis},
        executor=executor, state_readings={}, turn=1,
    )


def _run_batch(executor: WorkspaceToolExecutor, hypotheses: list[dict[str, Any]]) -> None:
    result = execute_composite(
        "hypothesis_test_words",
        {"branch": "main", "hypotheses": hypotheses},
        executor=executor, state_readings={}, turn=1,
    )
    if result.get("status") != "ok" or result.get("count") != len(hypotheses):
        raise RuntimeError(f"batch benchmark failed: {result}")


def _time(fn: Callable[[], None]) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def measure(count: int, repeats: int) -> dict[str, Any]:
    words = PLAINTEXT.split()[:count]
    hypotheses = [
        {"word": word, "word_index": index}
        for index, word in enumerate(words)
    ]
    samples: dict[str, list[float]] = {
        "cold_singletons": [], "cached_singletons": [], "batch": [],
    }
    for _ in range(repeats):
        samples["cold_singletons"].append(_time(
            lambda: [_run_single(_executor(), hypothesis)
                     for hypothesis in hypotheses]
        ))
        cached_executor = _executor()
        samples["cached_singletons"].append(_time(
            lambda: [_run_single(cached_executor, hypothesis)
                     for hypothesis in hypotheses]
        ))
        batch_executor = _executor()
        samples["batch"].append(_time(
            lambda: _run_batch(batch_executor, hypotheses)
        ))

    medians = {name: statistics.median(values)
               for name, values in samples.items()}
    cold = medians["cold_singletons"]
    batch = medians["batch"]
    return {
        "hypothesis_count": count,
        "repeats": repeats,
        "samples_seconds": samples,
        "median_seconds": medians,
        "batch_reduction_vs_cold_singletons": (
            1.0 - batch / cold if cold > 0 else 0.0
        ),
        "batch_speedup_vs_cold_singletons": (
            cold / batch if batch > 0 else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8, choices=range(1, 17))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-reduction", type=float, default=0.70)
    parser.add_argument("--json-output")
    args = parser.parse_args()
    result = measure(args.count, args.repeats)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    reduction = float(result["batch_reduction_vs_cold_singletons"])
    if reduction < args.require_reduction:
        print(
            f"FAIL: {reduction:.1%} reduction is below "
            f"{args.require_reduction:.1%}", file=sys.stderr,
        )
        return 1
    print(f"PASS: batch reduced cold-singleton time by {reduction:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
