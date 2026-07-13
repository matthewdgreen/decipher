"""Ground-truth firewall regression tests (Phase-0 item 0.7).

Benchmark ground truth may be read only for post-hoc grading; it must never
flow toward the model or the solver and influence a run. These tests are a
first-pass leak detector on everything that reaches the model (agent path) and
a by-construction independence check on the automated solver (automated path).
"""
from __future__ import annotations

import json
import os
import random
import sys
from types import SimpleNamespace
from typing import Iterable

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.loop_v2 import run_v2
from automated.runner import run_automated
from benchmark.context import ScopedBenchmarkContext
from models.alphabet import Alphabet
from models.cipher_text import CipherText


# A distinctive plaintext that no tool would ever emit by chance.
GROUND_TRUTH = "WOMBATFESTIVALQUARTZGIZMOJACKDAWNXYLOPHONEVEXBECK"


def _caesar(text: str, shift: int) -> str:
    return "".join(chr((ord(c) - 65 + shift) % 26 + 65) for c in text)


# The ciphertext the harness actually sees — the plaintext never appears in it.
CIPHERTEXT = _caesar(GROUND_TRUTH, 7)


def _normalize(text: str) -> str:
    """Uppercase and strip all whitespace (also yields the no-boundary form)."""
    return "".join(str(text).upper().split())


def assert_no_ground_truth_leak(haystacks: Iterable[str], plaintext: str) -> None:
    """Assert the ground truth does not appear in any haystack.

    Comparison is case-insensitive and whitespace-insensitive (which also
    covers the no-boundary variant). Both the full string and its first 30
    characters are checked, so a partial leak is caught too.
    """
    needle = _normalize(plaintext)
    needle_prefix = needle[:30]
    assert len(needle_prefix) == 30, "test plaintext must be >= 30 chars"
    for haystack in haystacks:
        if haystack is None:
            continue
        norm = _normalize(haystack)
        assert needle not in norm, f"full ground truth leaked: ...{norm[:80]}..."
        assert needle_prefix not in norm, (
            f"ground-truth prefix leaked: ...{norm[:80]}..."
        )


def test_helper_detects_a_planted_leak():
    """Sanity check: the helper is not a no-op."""
    with pytest.raises(AssertionError):
        assert_no_ground_truth_leak([f"prefix {GROUND_TRUTH} suffix"], GROUND_TRUTH)
    # A no-boundary / lowercased variant is also caught.
    with pytest.raises(AssertionError):
        assert_no_ground_truth_leak([GROUND_TRUTH.lower()], GROUND_TRUTH)


class _RecordingAPI:
    """Fake provider that records every system prompt + message list it sees
    and drives the loop with a harmless read-only tool call."""

    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen: list = []
        self.systems: list = []
        self.n = 0

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.systems.append(system)
        self.n += 1
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id=f"d{self.n}",
                    name="decode_show",
                    input={"branch": "main"},
                )
            ],
        )


def _collect_agent_haystacks(api: _RecordingAPI, artifact) -> list[str]:
    haystacks: list[str] = list(api.systems)
    for msgs in api.messages_seen:
        for m in msgs:
            content = m.get("content")
            if isinstance(content, str):
                haystacks.append(content)
            elif isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        haystacks.append(str(c))
                    elif c.get("type") == "text":
                        haystacks.append(c.get("text", ""))
                    elif c.get("type") == "tool_result":
                        haystacks.append(json.dumps(c.get("content")))
                    else:
                        haystacks.append(json.dumps(c))
    for tc in artifact.tool_calls:
        haystacks.append(tc.result)
    haystacks.append(json.dumps(artifact.benchmark_context))
    return haystacks


def test_agent_path_never_sees_ground_truth():
    alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
    ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)

    # A benign benchmark context (the kind build_benchmark_context produces —
    # metadata/related-cipher notes, never the plaintext answer).
    benchmark_context = ScopedBenchmarkContext(
        policy="max",
        prompt=(
            "Related manuscript context: an 18th-century lodge document, "
            "German Masonic register. No decrypted plaintext is provided."
        ),
    )

    api = _RecordingAPI()
    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=2,
        cipher_id="firewall_agent",
        prior_context=benchmark_context.prompt,
        benchmark_context=benchmark_context,
    )

    # The run must be non-trivial for the assertion to mean anything.
    assert api.messages_seen, "provider was never called"
    assert api.systems and api.systems[0], "no system prompt captured"
    assert artifact.tool_calls, "no tool calls executed"

    haystacks = _collect_agent_haystacks(api, artifact)
    haystacks.append(benchmark_context.prompt)
    assert_no_ground_truth_leak(haystacks, GROUND_TRUTH)

    # Ground truth is never populated on the agent artifact by the loop itself;
    # it is only attached post-hoc by the benchmark runner for scoring.
    assert artifact.ground_truth is None


def test_automated_path_ground_truth_only_grades_never_influences():
    """run_automated accepts a ground_truth argument, but it is used only for
    post-hoc scoring — the produced decryption/key must be identical whether or
    not ground truth is supplied, and the plaintext must not leak into any
    solver-facing output."""
    def _run(ground_truth):
        alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
        ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)
        # Built the way runner_v2/cli build the call.
        return run_automated(
            cipher_text=ct,
            language="en",
            cipher_id="firewall_auto",
            ground_truth=ground_truth,
            cipher_system="simple_substitution",
        )

    random.seed(0)
    without = _run(None)
    random.seed(0)
    with_gt = _run(GROUND_TRUTH)

    # Loud guard against a vacuous pass: if the solver could not actually run
    # (e.g. the English binary n-gram model is absent), status is "error" and
    # the decryption is empty — every equality assertion below would then hold
    # trivially without testing anything.
    assert with_gt.status == "completed", (
        f"automated run did not complete (status={with_gt.status!r}, "
        f"error={with_gt.error_message!r}); firewall comparison would be vacuous"
    )
    assert with_gt.final_decryption, (
        "empty decryption; firewall comparison would be vacuous"
    )

    # Ground truth does not influence the solve.
    assert without.final_decryption == with_gt.final_decryption
    assert without.artifact["key"] == with_gt.artifact["key"]

    # Steps are identical too, once run-varying timing is stripped.
    def _strip_timing(steps):
        return [
            {k: v for k, v in step.items() if k != "elapsed_seconds"}
            for step in steps
        ]

    assert _strip_timing(without.steps) == _strip_timing(with_gt.steps)

    # And the plaintext does not leak into solver-facing outputs. It MAY appear
    # only in the artifact's post-hoc grading fields.
    solver_facing = [
        with_gt.final_decryption,
        json.dumps(with_gt.steps),
        json.dumps(with_gt.artifact["key"]),
        with_gt.artifact.get("solver", ""),
    ]
    assert_no_ground_truth_leak(solver_facing, GROUND_TRUTH)

    # Document the post-hoc grading channel: ground truth IS retained there.
    assert with_gt.artifact.get("ground_truth") == GROUND_TRUTH
