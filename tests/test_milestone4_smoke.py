from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.loop_v2 import run_v2
from automated.runner import AutomatedBenchmarkRunner
from benchmark.loader import BenchmarkLoader
from frontier.suite import (
    evaluate_frontier_rows,
    load_frontier_suite,
    resolve_frontier_case,
)
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from testgen.cache import PlaintextCache


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPO_ROOT / "frontier" / "agentic_milestone4_smoke.jsonl"
EXPECTED_CASE_IDS = {
    "borg_single_B_borg_0109v",
    "borg_single_B_borg_0045v",
    "borg_single_B_borg_0140v",
    "borg_single_B_borg_0171v",
    "parity_borg_latin_borg_0077v",
    "copiale_single_B_copiale_p068",
}


def test_milestone4_smoke_suite_defines_clean_automated_baselines():
    cases = load_frontier_suite(SUITE_PATH)

    assert {case.test.test_id for case in cases} == EXPECTED_CASE_IDS
    for case in cases:
        assert case.synthetic_spec is None
        assert case.test.track == "transcription2plaintext"
        assert case.test.context_records == []
        assert case.expected_solvers == ["decipher-automated"]
        assert case.expected_status_by_solver["decipher-automated"] == "completed"
        assert "milestone4_agentic_smoke" in case.frontier_tags
        assert "automated_baseline" in case.frontier_tags
        assert case.min_char_accuracy_by_solver["decipher-automated"] > 0.0


@pytest.mark.skipif(
    os.environ.get("DECIPHER_RUN_MILESTONE4_SMOKE") != "1",
    reason=(
        "set DECIPHER_RUN_MILESTONE4_SMOKE=1 to run the historical automated "
        "Milestone 4 smoke suite"
    ),
)
def test_milestone4_automated_only_solver_baselines(tmp_path):
    benchmark_root = Path(
        os.environ.get(
            "DECIPHER_BENCHMARK_ROOT",
            str((REPO_ROOT / "../cipher_benchmark/benchmark").resolve()),
        )
    )
    manifest = benchmark_root / "manifest" / "records.jsonl"
    if not manifest.exists():
        pytest.skip(f"benchmark manifest not found: {manifest}")

    cases = load_frontier_suite(SUITE_PATH)
    loader = BenchmarkLoader(benchmark_root)
    cache = PlaintextCache(tmp_path / "cache")
    runner = AutomatedBenchmarkRunner(artifact_dir=tmp_path / "artifacts")

    rows = []
    for case in cases:
        test_data = resolve_frontier_case(case, loader, cache)
        result = runner.run_test(test_data)
        artifact = result.artifact

        assert result.total_tokens == 0
        assert result.estimated_cost_usd == 0.0
        assert artifact["run_mode"] == "automated_only"
        assert artifact["automated_only"] is True
        assert artifact["total_input_tokens"] == 0
        assert artifact["total_output_tokens"] == 0
        assert artifact["estimated_cost_usd"] == 0.0

        rows.append({
            "test_id": case.test.test_id,
            "solver": "decipher-automated",
            "status": result.status,
            "char_accuracy": result.char_accuracy,
            "word_accuracy": result.word_accuracy,
            "elapsed_seconds": result.elapsed_seconds,
            "expected_status_by_solver": case.expected_status_by_solver,
            "min_char_accuracy_by_solver": case.min_char_accuracy_by_solver,
            "max_elapsed_seconds_by_solver": case.max_elapsed_seconds_by_solver,
            "max_gap_vs_solver": case.max_gap_vs_solver,
        })

    evaluated = evaluate_frontier_rows(rows)
    failures = {
        row["test_id"]: row["expectation_failures"]
        for row in evaluated
        if not row["meets_expectations"]
    }
    assert failures == {}


def _tool_response(*blocks: SimpleNamespace, input_tokens: int = 10, output_tokens: int = 2):
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        content=list(blocks),
    )


def _tool_use(name: str, args: dict, tool_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use",
        id=tool_id or name,
        name=name,
        input=args,
    )


def _preflight_key(mapping: dict[str, str]) -> dict[str, int]:
    pt = Alphabet.from_text("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return {
        str(ord(cipher_symbol) - ord("A")): pt.id_for(plain_letter)
        for cipher_symbol, plain_letter in mapping.items()
    }


class _InspectPreflightThenDeclareAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []
        self.tools_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.tools_seen.append(tools or [])
        if len(self.messages_seen) == 1:
            return _tool_response(_tool_use("workspace_branch_cards", {}))
        return _tool_response(
            _tool_use(
                "meta_declare_solution",
                {
                    "branch": "automated_preflight",
                    "rationale": "The protected no-LLM preflight already reads cleanly.",
                    "self_confidence": 0.9,
                    "reading_summary": "A tiny solved English smoke case.",
                    "further_iterations_helpful": False,
                    "further_iterations_note": "No further repair is needed.",
                },
            )
        )


def test_milestone4_fake_agent_can_declare_protected_preflight():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _InspectPreflightThenDeclareAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=3,
        cipher_id="milestone4_fake_preflight",
        automated_preflight={
            "enabled": True,
            "run_mode": "automated_only",
            "status": "completed",
            "solver": "fake_native",
            "summary": "Automated native solver preflight (no LLM access): THE",
            "key": _preflight_key({"A": "T", "B": "H", "C": "E"}),
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        },
    )

    first_content = api.messages_seen[0][0]["content"]
    first_turn_texts = [first_content] if isinstance(first_content, str) else [
        c["text"]
        for c in first_content
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    assert any("Protected baseline rule" in text for text in first_turn_texts)
    assert artifact.status == "solved"
    assert artifact.solution is not None
    assert artifact.solution.branch == "automated_preflight"
    assert [call.tool_name for call in artifact.tool_calls] == [
        "workspace_branch_cards",
        "meta_declare_solution",
    ]
    branch = next(b for b in artifact.branches if b.name == "automated_preflight")
    assert branch.decryption == "THE"
    assert "no_llm" in branch.tags


class _RepairPreflightThenDeclareAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []
        self.tools_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.tools_seen.append(tools or [])
        call = len(self.messages_seen)
        if call == 1:
            return _tool_response(
                _tool_use(
                    "act_set_mapping",
                    {
                        "branch": "automated_preflight",
                        "cipher_symbol": "D",
                        "plain_letter": "Y",
                    },
                )
            )
        if call == 2:
            return _tool_response(_tool_use("workspace_branch_cards", {}))
        return _tool_response(
            _tool_use(
                "meta_declare_solution",
                {
                    "branch": "automated_preflight",
                    "rationale": "One reading-driven symbol repair changed THN to THY.",
                    "self_confidence": 0.75,
                    "reading_summary": "A tiny archaic-English smoke case.",
                    "further_iterations_helpful": False,
                    "further_iterations_note": "The intended reading is already reached.",
                },
            )
        )


def test_milestone4_fake_agent_can_repair_preflight_then_declare():
    alpha = Alphabet(list("ABCD"))
    ct = CipherText(raw="ABC | ABD", alphabet=alpha, separator=" | ")
    api = _RepairPreflightThenDeclareAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=4,
        cipher_id="milestone4_fake_repair",
        automated_preflight={
            "enabled": True,
            "run_mode": "automated_only",
            "status": "completed",
            "solver": "fake_native",
            "summary": "Automated native solver preflight (no LLM access): THE | THN",
            "key": _preflight_key({"A": "T", "B": "H", "C": "E", "D": "N"}),
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        },
    )

    assert artifact.status == "solved"
    assert artifact.solution is not None
    assert artifact.solution.branch == "automated_preflight"
    assert [call.tool_name for call in artifact.tool_calls] == [
        "act_set_mapping",
        "workspace_branch_cards",
        "meta_declare_solution",
    ]
    branch = next(b for b in artifact.branches if b.name == "automated_preflight")
    assert branch.decryption == "THE | THY"
    mapping_call = next(c for c in artifact.tool_calls if c.tool_name == "act_set_mapping")
    mapping_result = (
        json.loads(mapping_call.result)
        if isinstance(mapping_call.result, str)
        else mapping_call.result
    )
    assert mapping_result["changed_words"][0]["before"] == "THN"
    assert mapping_result["changed_words"][0]["after"] == "THY"
