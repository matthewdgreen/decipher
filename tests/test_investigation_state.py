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
    state.workflow_hint_keys.append("mid_budget_verify_hint:abc")
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
    assert restored.workflow_hint_keys == ["mid_budget_verify_hint:abc"]
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


def test_r7_token_order_and_transform_pipeline_survive_roundtrip():
    """R7: the two branch fields v2 resume drops must round-trip through state."""
    ct = parse_canonical_transcription("S001 S002 S003 | S004 S005 S006")
    ws = Workspace(ct)
    ws.fork("perm", from_branch="main")
    branch = ws.get_branch("perm")
    branch.token_order = [5, 4, 3, 2, 1, 0]
    branch.transform_pipeline = {"steps": [{"op": "columnar", "cols": 3}]}
    state = InvestigationState(workspace=ws, language="en")

    restored = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(state.to_artifact_dict()))
    )
    dst = restored.workspace.get_branch("perm")
    assert dst.token_order == [5, 4, 3, 2, 1, 0]
    assert dst.transform_pipeline == {"steps": [{"op": "columnar", "cols": 3}]}


def test_finalist_sessions_and_board_survive_roundtrip():
    """M2: the state-owned finalist store and hypothesis board round-trip."""
    ct = parse_canonical_transcription("S001 S002 | S003 S002")
    ws = Workspace(ct)
    state = InvestigationState(workspace=ws, language="en")
    state.finalist_sessions.new_session("word_repair", {"a": 1}, packets=[])
    ex = _executor_for(state)
    # Route a hypothesis through the board (single writer).
    ws.fork("h", from_branch="main")
    state.hypothesis_board.create(
        ws, "h", cipher_mode="substitution", mode_confidence="high",
        mode_status="active", mode_evidence="mono", evidence_source="agent_inference",
    )
    restored = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(state.to_artifact_dict()))
    )
    assert restored.finalist_sessions.get("word_repair", "word_repair_1") == {
        "a": 1, "packets": []}
    cards = {c["branch"]: c for c in restored.hypothesis_cards()}
    assert cards["h"]["cipher_mode"] == "substitution"


def test_readings_round_trip_and_m2_artifact_loads():
    """M3: state.readings round-trips; an M2 artifact (no readings key) loads."""
    from investigation.reading import Reading, ReadingFragment
    ct = parse_canonical_transcription("S001 S002 | S003 S002")
    state = InvestigationState(workspace=Workspace(ct), language="en")
    reading = Reading(branch="main", source="episode:e1", created_turn=2,
                      fragments=[ReadingFragment(text="HELLO", start=0, end=5)],
                      holes=["tail"], overall_confidence=0.8)
    state.readings[reading.reading_id] = reading.to_dict()
    state.repair_transactions.append({
        "transaction_id": "tx1",
        "status": "installed",
        "source_content_hash": "before",
        "result_content_hash": "after",
    })

    restored = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(state.to_artifact_dict()))
    )
    assert reading.reading_id in restored.readings
    rd = Reading.from_dict(restored.readings[reading.reading_id])
    assert rd.branch == "main"
    assert rd.fragments[0].start == 0
    assert rd.overall_confidence == 0.8
    assert restored.repair_transactions == state.repair_transactions

    # An M2 artifact predating the readings key loads with an empty map.
    m2_dict = state.to_artifact_dict()
    del m2_dict["readings"]
    del m2_dict["repair_transactions"]
    reloaded = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(m2_dict))
    )
    assert reloaded.readings == {}
    assert reloaded.repair_transactions == []


def test_resume_identity_holds_with_readings():
    """Resume identity (M1) extends over the readings section (M3)."""
    from investigation.reading import Reading, ReadingFragment
    state = _rich_state()
    reading = Reading(branch="main", source="lead", created_turn=3,
                      fragments=[ReadingFragment(text="HELLO WORLD")],
                      overall_confidence=0.6)
    state.readings[reading.reading_id] = reading.to_dict()
    ex1 = _executor_for(state)
    ctx1 = build_lead_context(state, ex1, turn=5, token_budget=8000)
    reloaded = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(state.to_artifact_dict()))
    )
    ex2 = _executor_for(reloaded)
    ctx2 = build_lead_context(reloaded, ex2, turn=5, token_budget=8000)
    assert ctx1 == ctx2
    # The readings section rendered into the context.
    assert "Recent readings" in json.dumps(ctx1, default=str)


# ---------------------------------------------------------------------------
# M4: experiment queue — load transition + budget bucket
# ---------------------------------------------------------------------------
def _state_with_experiment_records():
    ct = parse_canonical_transcription("S001 S002 S003 | S004 S002 S005")
    state = InvestigationState(workspace=Workspace(ct), language="en")
    state.experiment_queue = [
        {"experiment_id": "run1", "type": "automated_solver", "status": "running",
         "started_at": 100.0, "elapsed_seconds": None, "branch": "main",
         "dedup_key": "k1", "collected": False},
        {"experiment_id": "pend1", "type": "automated_solver", "status": "pending",
         "started_at": None, "branch": "main", "dedup_key": "k2", "collected": False},
        {"experiment_id": "done1", "type": "automated_solver", "status": "completed",
         "started_at": 50.0, "elapsed_seconds": 12.5, "branch": "main",
         "dedup_key": "k3", "collected": False, "result": {"status": "completed"}},
    ]
    return state


def test_experiment_records_load_transition_orphans_running_and_pending():
    state = _state_with_experiment_records()
    reloaded = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(state.to_artifact_dict())))
    by_id = {r["experiment_id"]: r for r in reloaded.experiment_queue}
    # running/pending -> orphaned(loaded); completed loads verbatim.
    assert by_id["run1"]["status"] == "orphaned"
    assert by_id["run1"]["orphan_reason"] == "loaded"
    assert by_id["pend1"]["status"] == "orphaned"
    assert by_id["pend1"]["orphan_reason"] == "loaded"
    assert by_id["done1"]["status"] == "completed"
    assert by_id["done1"]["result"] == {"status": "completed"}


def test_experiment_budget_bucket_additive():
    state = _state_with_experiment_records()
    buckets = state.budget_by_category()
    bucket = buckets["experiment:automated_solver"]
    # calls counts records that reached running (started_at set) = run1 + done1.
    assert bucket["calls"] == 2
    # elapsed sums completed/failed records only = done1's 12.5.
    assert bucket["elapsed_seconds"] == 12.5
    assert bucket["input_tokens"] == 0
    assert bucket["cost_usd"] == 0.0
