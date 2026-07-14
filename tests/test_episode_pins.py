"""Pre-extraction pin tests for the M2 A4/A10 discipline sweep.

These lock the current v2 behavior of the surfaces M2 refactors behind hooks:
- the finalize-phase guard dicts (extracted into DeclarationPolicy.finalize_guard),
- the post-search declare notes (extracted into DeclarationPolicy.search_note),
- the full hypothesis-branch metadata contract (routed through HypothesisBoard),
- the OpenAI Responses default kwargs (server-state plumbing must not change them).

They are written to pass on BOTH the pre- and post-extraction code (byte-identical
under v2 defaults).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.tools_v2 import WorkspaceToolExecutor
from benchmark.loader import parse_canonical_transcription
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


def _sub_executor() -> WorkspaceToolExecutor:
    raw = "ABC DEF ABC"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    return WorkspaceToolExecutor(Workspace(ct), "en", set(), [], {})


# ---------------------------------------------------------------------------
# A4: _search_declare_note text (search_note hook)
# ---------------------------------------------------------------------------
EXPECTED_ANNEAL_NOTE = (
    "Read decoded_preview carefully. If coherent en phrases appear, repair "
    "obvious residual errors and then consider declaring. Do not declare early "
    "merely because a few recognizable words appear; if useful tools remain, "
    "continued exploration beats a low-confidence partial. If a few residual "
    "errors remain, call decode_diagnose_and_fix(branch) to fix them all in one "
    "call, then do one anchored polish run with "
    "search_anneal(preserve_existing=true, score_fn='combined') before declaring."
)
EXPECTED_HILL_NOTE = (
    "hill_climb has converged. If decoded text shows any coherent en phrases, "
    "consider declaring; a few recognisable words alone are not enough if useful "
    "search or repair tools remain. If still stuck, fork and restart with "
    "search_anneal — it escapes local optima that hill_climb can't."
)


def test_pin_search_declare_note_v2_text():
    ex = _sub_executor()
    assert ex._search_declare_note("anneal") == EXPECTED_ANNEAL_NOTE
    assert ex._search_declare_note("hill_climb") == EXPECTED_HILL_NOTE


# ---------------------------------------------------------------------------
# A4: finalize-phase guard dicts (finalize_guard hook)
# ---------------------------------------------------------------------------
def test_pin_finalize_guard_hill_climb_block():
    ex = _sub_executor()
    ex._reading_attestations["main"] = {"comprehensibility_score": 7}
    result = ex._tool_search_hill_climb({"branch": "main"})
    assert result == {
        "status": "blocked",
        "reason": "finalize_phase",
        "branch": "main",
        "note": (
            "This branch is in finalize phase (comprehensibility score ≥ 7 "
            "recorded). Running hill_climb on a readable branch risks "
            "overwriting the correct key. Prefer meta_declare_solution. "
            "If further search is genuinely needed, re-call with "
            "justification=<reason>."
        ),
        "suggested_next_tools": ["meta_declare_solution"],
    }


def test_pin_finalize_guard_anneal_block():
    ex = _sub_executor()
    ex._reading_attestations["main"] = {"comprehensibility_score": 7}
    result = ex._tool_search_anneal({"branch": "main"})
    assert result == {
        "status": "blocked",
        "reason": "finalize_phase",
        "branch": "main",
        "note": (
            "This branch is in finalize phase (comprehensibility score ≥ 7 "
            "recorded). Re-annealing a readable branch risks overwriting "
            "the correct key with a random-restart artifact. "
            "Prefer meta_declare_solution. "
            "If further search is genuinely needed, re-call with "
            "justification=<reason>."
        ),
        "suggested_next_tools": ["meta_declare_solution"],
    }


# ---------------------------------------------------------------------------
# A10/F3: full hypothesis-branch metadata contract
# ---------------------------------------------------------------------------
EXPECTED_HYP_METADATA = {
    "cipher_mode": "quagmire3",
    "context_assumption_note": "Agent-declared from exposed benchmark context.",
    "context_mode_prior": {
        "confidence": "agent_declared_context_supported",
        "evidence": [{
            "branch": "hyp",
            "cipher_mode": "quagmire3",
            "evidence_source": "benchmark_context",
            "mode_evidence": "context says keyed tableau vigenere here",
        }],
        "note": (
            "The agent read exposed benchmark context and recorded this as a "
            "controlling cipher-family assumption."
        ),
        "prior": "keyed_tableau_polyalphabetic",
        "required_tools_before_rejection": ["search_quagmire3_keyword_alphabet"],
    },
    "context_supported_mode": True,
    "evidence_source": "benchmark_context",
    "hypothesis_notes": "context says keyed tableau vigenere here",
    "mode_confidence": "high",
    "mode_counter_evidence": "IC a bit high",
    "mode_evidence": "kasiski period 7",
    "mode_status": "active",
    "next_recommended_action": "run quagmire search",
    "required_tools_before_rejection": ["search_quagmire3_keyword_alphabet"],
}


def test_pin_hypothesis_metadata_contract():
    ct = parse_canonical_transcription("S001 S002 S003 | S004 S002 S005")
    ws = Workspace(ct)
    ex = WorkspaceToolExecutor(ws, "en", set(), [], {})
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp", "from_branch": "main",
        "cipher_mode": "quagmire3",
        "rationale": "context says keyed tableau vigenere here",
        "mode_confidence": "high", "evidence_source": "benchmark_context",
    })
    ex.execute("workspace_update_hypothesis", {
        "branch": "hyp", "cipher_mode": "quagmire3", "mode_status": "active",
        "mode_confidence": "high", "evidence": "kasiski period 7",
        "counter_evidence": "IC a bit high",
        "next_recommended_action": "run quagmire search",
        "evidence_source": "benchmark_context",
    })
    branch = ws.get_branch("hyp")
    assert dict(branch.metadata) == EXPECTED_HYP_METADATA
    assert branch.tags == ["hypothesis", "mode:quagmire3"]


# ---------------------------------------------------------------------------
# C7/F2: OpenAI Responses default kwargs must stay byte-identical
# ---------------------------------------------------------------------------
class _RecordingResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_1",
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="hi")],
            )],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  input_tokens_details=SimpleNamespace(cached_tokens=0)),
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _RecordingResponses()


def _make_openai_provider(model: str):
    from agent.model_provider import OpenAIModelProvider
    provider = OpenAIModelProvider.__new__(OpenAIModelProvider)
    provider.model = model
    provider.client = _FakeOpenAIClient()
    return provider


def test_pin_send_responses_default_kwargs_byte_identical(monkeypatch):
    monkeypatch.setenv("DECIPHER_OPENAI_REASONING_PASSBACK", "1")
    provider = _make_openai_provider("gpt-5.6-luna")
    tools = [{"name": "decode_show", "description": "d",
              "input_schema": {"type": "object", "properties": {}}}]
    provider.send(
        messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
        tools=tools, system="sys", max_tokens=1234,
    )
    kwargs = provider.client.responses.calls[0]
    # Defaults (store=None, previous_response_id=None) reproduce M1 kwargs exactly.
    assert kwargs["model"] == "gpt-5.6-luna"
    assert kwargs["instructions"] == "sys"
    assert kwargs["max_output_tokens"] == 1234
    assert kwargs["store"] is False
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in kwargs
    # A tools list passes through in converted form.
    assert kwargs["tools"] == [{
        "type": "function", "name": "decode_show", "description": "d",
        "parameters": {"type": "object", "properties": {}},
    }]


def test_pin_send_responses_tools_none_omits_tools_kwarg(monkeypatch):
    """A tool-less Responses send (an episode's final result nudge) must omit
    the tools kwarg entirely — `"tools": null` may 400."""
    monkeypatch.setenv("DECIPHER_OPENAI_REASONING_PASSBACK", "1")
    provider = _make_openai_provider("gpt-5.6-luna")
    provider.send(
        messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
        tools=None, system="sys", max_tokens=64,
    )
    kwargs = provider.client.responses.calls[0]
    assert "tools" not in kwargs


def test_pin_send_responses_passback_off_omits_store(monkeypatch):
    monkeypatch.setenv("DECIPHER_OPENAI_REASONING_PASSBACK", "0")
    provider = _make_openai_provider("gpt-5.6-luna")
    provider.send(
        messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
        tools=None, system="sys", max_tokens=64,
    )
    kwargs = provider.client.responses.calls[0]
    assert "store" not in kwargs
    assert "include" not in kwargs
    assert "previous_response_id" not in kwargs
    assert "tools" not in kwargs


def test_pin_send_responses_chained_kwargs_keep_include(monkeypatch):
    """F2: a chained (store=True) send keeps include=[reasoning.encrypted_content]
    and carries previous_response_id."""
    monkeypatch.setenv("DECIPHER_OPENAI_REASONING_PASSBACK", "1")
    provider = _make_openai_provider("gpt-5.6-luna")
    provider.send(
        messages=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
        tools=None, system="sys", max_tokens=64,
        store=True, previous_response_id="resp_prev",
    )
    kwargs = provider.client.responses.calls[0]
    assert kwargs["store"] is True
    assert kwargs["previous_response_id"] == "resp_prev"
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    assert "tools" not in kwargs
