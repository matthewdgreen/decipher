"""Regression tests for agent-loop reliability guardrails."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.loop_v2 import FINAL_ITERATION_PREFLIGHT, run_v2
import agent.tools_v2 as tools_v2
from agent.tools_v2 import WorkspaceToolExecutor
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from services.claude_api import ClaudeAPIError
from workspace import Workspace


def _executor_for(raw: str, separator: str | None = None) -> WorkspaceToolExecutor:
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=separator)
    return WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "OLD", "BAKERY", "ON", "MAPLE", "STREET"},
        word_list=["THE", "OLD", "BAKERY", "ON", "MAPLE", "STREET"],
        pattern_dict={},
    )


def _homophonic_executor() -> WorkspaceToolExecutor:
    symbols = [f"S{i:02d}" for i in range(30)]
    raw = " ".join(symbols)
    alpha = Alphabet(symbols)
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "OLD", "BOOK", "STORE", "MAPLE", "STREET"},
        word_list=["THE", "OLD", "BOOK", "STORE", "MAPLE", "STREET"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    # Deliberately overload H and leave U absent.
    for i, sym in enumerate(symbols):
        letter = "H" if i < 5 else "E"
        ex.workspace.set_mapping("main", alpha.id_for(sym), pt.id_for(letter))
    return ex


def test_score_delta_reports_mixed_when_signals_disagree():
    ex = _executor_for("ABC")
    before = {"dict_rate": 0.90, "quad": -5.0}
    after = {"dict_rate": 0.80, "quad": -4.8}

    delta = ex._score_delta(before, after)

    assert delta["verdict"] == "mixed"
    assert delta["improved"] is False
    assert delta["dict_rate_delta"] == -0.1
    assert delta["quad_delta"] == 0.2


def test_decode_ambiguous_letter_groups_contexts_by_cipher_symbol():
    ex = _executor_for("ABCAAC", separator=None)
    ws = ex.workspace
    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("I"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("I"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("H"))

    out = ex._tool_decode_ambiguous_letter({
        "branch": "main",
        "decoded_letter": "I",
        "context": 1,
    })

    assert out["cipher_symbols"] == ["A", "C"]
    assert out["symbol_count"] == 2
    assert {g["cipher_symbol"] for g in out["groups"]} == {"A", "C"}
    assert "act_set_mapping" in out["groups"][0]["suggested_next_step"]


class _FakeAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=[SimpleNamespace(type="text", text="I forgot to declare.")],
        )


def test_final_turn_prefight_and_auto_declare_fallback():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _FakeAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=1,
        cipher_id="unit",
    )

    sent_texts = [
        c["text"]
        for m in api.messages_seen[-1]
        for c in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    assert any(FINAL_ITERATION_PREFLIGHT in t for t in sent_texts)
    assert artifact.status == "solved"
    assert artifact.solution is not None
    assert artifact.solution.branch == "main"
    assert artifact.solution.self_confidence == 0.0


class _ToolThenErrorAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.calls = 0

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="tu_1",
                        name="act_set_mapping",
                        input={
                            "branch": "main",
                            "cipher_symbol": "A",
                            "plain_letter": "T",
                        },
                    )
                ],
            )
        raise ClaudeAPIError("overloaded")


def test_api_error_after_progress_auto_declares_best_branch():
    alpha = Alphabet(list("A"))
    ct = CipherText(raw="A", alphabet=alpha, separator=None)

    artifact = run_v2(
        cipher_text=ct,
        claude_api=_ToolThenErrorAPI(),  # type: ignore[arg-type]
        language="en",
        max_iterations=2,
        cipher_id="unit",
    )

    assert artifact.status == "solved"
    assert artifact.error_message is not None
    assert "overloaded" in artifact.error_message
    assert artifact.solution is not None
    assert artifact.solution.branch == "main"
    assert artifact.branches[0].decryption == "T"


def test_search_anneal_restarts_complete_inherited_key_by_default(monkeypatch):
    ex = _executor_for("ABC", separator=None)
    ws = ex.workspace
    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("B"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("C"))

    def fake_anneal(session, score_fn, max_steps, t_start, t_end):
        return score_fn()

    monkeypatch.setattr(tools_v2, "simulated_anneal", fake_anneal)

    anchored = ex._tool_search_anneal({
        "branch": "main",
        "steps": 1,
        "restarts": 1,
        "preserve_existing": True,
    })
    fresh = ex._tool_search_anneal({
        "branch": "main",
        "steps": 1,
        "restarts": 1,
    })

    assert anchored["preserve_existing"] is True
    assert anchored["auto_seeded_symbols"] == 0
    assert fresh["preserve_existing"] is False
    assert fresh["preserved_symbols"] == 0
    assert fresh["auto_seeded_symbols"] == 3


def test_homophone_distribution_flags_absent_and_overloaded_letters():
    ex = _homophonic_executor()

    out = ex._tool_observe_homophone_distribution({"branch": "main"})

    assert out["is_likely_homophonic"] is True
    assert sum(r["expected_symbols"] for r in out["expected_symbol_counts"]) == 30
    actual_h = next(r for r in out["actual_vs_expected"] if r["letter"] == "H")
    assert actual_h["actual_symbols"] == 5
    assert any("absent" in w for w in out["warnings"])


def test_absent_letter_candidates_rank_symbol_remaps_with_score_delta():
    ex = _homophonic_executor()

    out = ex._tool_decode_absent_letter_candidates({
        "branch": "main",
        "missing_letter": "U",
        "source_letters": ["H"],
        "max_candidates": 3,
        "context": 1,
    })

    assert out["missing_letter"] == "U"
    assert out["source_letters_considered"] == ["H"]
    assert out["candidates"]
    cand = out["candidates"][0]
    assert cand["current_letter"] == "H"
    assert cand["candidate_letter"] == "U"
    assert "score_delta_if_remapped" in cand
    assert "act_set_mapping" in cand["suggested_call"]
