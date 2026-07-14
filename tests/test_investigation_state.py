"""InvestigationState serialization + resume-identity tests (M1 Part 1/5)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor
from benchmark.loader import parse_canonical_transcription
from investigation.context import build_lead_context
from investigation.state import BudgetEntry, EvidenceEntry, InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


def _rich_state() -> InvestigationState:
    ct = parse_canonical_transcription("S001 S002 S003 | S004 S002 S005")
    ws = Workspace(ct)
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", ct.alphabet.id_for("S001"), pt.id_for("H"))
    ws.set_mapping("main", ct.alphabet.id_for("S002"), pt.id_for("E"))
    ws.fork("hyp", from_branch="main")
    branch = ws.get_branch("hyp")
    branch.metadata["cipher_mode"] = "homophonic"
    branch.metadata["mode_status"] = "active"
    branch.tags.append("hypothesis")
    ws.set_word_spans("hyp", ws.effective_word_spans("main"))

    state = InvestigationState(workspace=ws, language="de")
    state.add_evidence("diagnostic_preflight", 0, "diag", suspicion={"homophonic": 0.7})
    state.add_evidence("turn_summary", 1, "decode_show:ok")
    state.add_budget(BudgetEntry("lead", "openai", "gpt-5.5", 1000, 100, 200))
    state.repair_agenda.append(
        {"id": 1, "branch": "main", "from": "X", "to": "Y", "status": "open"}
    )
    state.recent_exchanges = [
        {"role": "assistant", "content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "t1", "name": "decode_show",
             "input": {"branch": "main"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "{\"ok\": true}"},
        ]},
    ]
    state.external_context = "External note: 18th-century lodge register."
    state.turn = 4
    return state


def _executor_for(state: InvestigationState) -> WorkspaceToolExecutor:
    return WorkspaceToolExecutor(
        workspace=state.workspace,
        language=state.language,
        word_set=set(),
        word_list=[],
        pattern_dict={},
        declaration_policy=NoGatesPolicy(),
    )


def test_state_roundtrip_is_json_safe_and_faithful():
    state = _rich_state()
    data = state.to_artifact_dict()
    # Must be JSON-serializable (it lands in the artifact).
    roundtripped = json.loads(json.dumps(data))
    restored = InvestigationState.from_artifact_dict(roundtripped)

    assert restored.language == "de"
    assert restored.turn == 4
    assert restored.cipher.tokens == state.cipher.tokens
    assert [len(w) for w in restored.cipher.words] == [len(w) for w in state.cipher.words]
    assert restored.workspace.branch_names() == state.workspace.branch_names()
    # A3: the three branch fields v2 resume drops survive.
    src = state.workspace.get_branch("hyp")
    dst = restored.workspace.get_branch("hyp")
    assert dst.key == src.key
    assert dst.metadata == src.metadata
    assert dst.word_spans == src.word_spans
    assert restored.external_context == state.external_context
    assert restored.repair_agenda == state.repair_agenda
    assert restored.recent_exchanges == state.recent_exchanges
    assert [e.kind for e in restored.evidence_log] == ["diagnostic_preflight", "turn_summary"]
    assert len(restored.budget_ledger) == 1


def test_resume_identity_next_turn_context_is_identical():
    state = _rich_state()
    ex1 = _executor_for(state)
    ctx1 = build_lead_context(state, ex1, turn=5, token_budget=8000)

    reloaded = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(state.to_artifact_dict()))
    )
    ex2 = _executor_for(reloaded)
    ctx2 = build_lead_context(reloaded, ex2, turn=5, token_budget=8000)

    assert ctx1 == ctx2


def test_budget_entry_cost_and_aggregation():
    state = InvestigationState(workspace=Workspace(
        CipherText(raw="ABC", alphabet=Alphabet.from_text("ABC", ignore_chars=set()),
                   separator=None)
    ))
    # gpt-5.5 pricing: (5.00, 30.00, 0.50) per M tokens.
    state.add_budget(BudgetEntry("lead", "openai", "gpt-5.5", 1_000_000, 0, 0))
    state.add_budget(BudgetEntry("lead", "openai", "gpt-5.5", 0, 1_000_000, 0))
    # billed_input = input - cache_read = 1M → $5; output 1M → $30.
    assert abs(state.total_cost() - 35.0) < 1e-6
    by_cat = state.budget_by_category()
    assert by_cat["lead"]["calls"] == 2
    assert by_cat["lead"]["input_tokens"] == 1_000_000
    assert abs(by_cat["lead"]["cost_usd"] - 35.0) < 1e-6


def test_recent_exchanges_trim_keeps_whole_exchanges():
    state = InvestigationState(workspace=Workspace(
        CipherText(raw="ABC", alphabet=Alphabet.from_text("ABC", ignore_chars=set()),
                   separator=None)
    ))
    state.max_recent_exchanges = 2
    for i in range(4):
        assistant = {"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "decode_show", "input": {}}]}
        tool_result = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "{}"}]}
        state.record_exchange(assistant, tool_result)
    # Only the last 2 whole exchanges (4 messages) remain, and each assistant is
    # immediately followed by its tool_result (never split).
    assert len(state.recent_exchanges) == 4
    assert state.recent_exchanges[0]["content"][0]["id"] == "t2"
    assert state.recent_exchanges[1]["content"][0]["tool_use_id"] == "t2"
    assert state.recent_exchanges[2]["content"][0]["id"] == "t3"
    assert state.recent_exchanges[3]["content"][0]["tool_use_id"] == "t3"


def test_evidence_entry_roundtrip():
    entry = EvidenceEntry("turn_summary", 3, "act_bulk_set:ok", data={"n": 2})
    restored = EvidenceEntry.from_dict(entry.to_dict())
    assert restored == entry
