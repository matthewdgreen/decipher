"""M5.3 Slice 7 Part G — Sequence-B scripted end-to-end replay (master 540-552).

ONE 8-turn scripted run over the shared v3 fakes proving all eight master
bullets in a single artifact, closing on the inspect_artifact analyzer output.
No paid model, no network, no real solver compute: the verify/reading/repair
workers are scripted fakes, the experiment queue is synchronous, and the
automated_solver runner is a trivial no-solution stub — so the REAL Slice-5
typed experiment validation runs while the compute is free.

Known constraint (§7.5, NOT a Slice-7 change): a single (content, evidence) pair
cannot reach two evidence failures with only ONE reading (Slice 2 caps fresh
readings at one per pair, and an evidence-failed pair cannot be rerun). Live
exhaustion therefore needs a second interpretation; Sequence B seeds the
auxiliary reading (COTON) exactly as the landed Slice-2 tests do and produces
the first (CATON) live via a reading episode.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel
from agent.model_provider import ModelResponse, ModelUsage, ToolUseBlock
from investigation import experiments
from investigation import sessions as sessions_mod
from investigation.experiments import ExperimentQueue
from investigation.loop_v3 import run_v3
from investigation.sessions import SessionCapabilities
from investigation.state import (
    BudgetEntry,
    attestation_is_positive,
    saturation_key,
)

from tests.support.scripted_v3 import (
    NEGATIVE_LOCAL_REPAIR_VERDICT,
    OverBudgetReadingWorkerFake,
    keyed_catton_state,
    make_verify_builder,
    register_programmable_repair,
    seed_reading,
)


# The analyzer module is loaded the way tests/test_inspect_artifact.py does.
_SCRIPT_PATH = (
    os.path.join(os.path.dirname(__file__), "..", "scripts", "inspect_artifact.py")
)
_spec = importlib.util.spec_from_file_location("inspect_artifact", _SCRIPT_PATH)
inspect_artifact = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = inspect_artifact
_spec.loader.exec_module(inspect_artifact)


_NO_FORK_APPLIED = lambda forks: {  # noqa: E731
    "applied": True, "best_branch": None, "edits": [],
    "verdicts": [], "collateral": {}, "notes": "",
}


def _fake_runner(cipher, snapshot, config):
    return {"status": "no_solution", "solver": "fake", "error_message": "",
            "elapsed_seconds": 0.01, "key": {}, "final_decryption": "",
            "steps": []}


def _tool_result_payloads(blocks):
    out = []
    for message in blocks:
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                try:
                    out.append(json.loads(block.get("content") or ""))
                except (TypeError, json.JSONDecodeError):
                    pass
    return out


def _find_payload(blocks, predicate):
    for payload in _tool_result_payloads(blocks):
        if predicate(payload):
            return payload
    return None


class SequenceBLead:
    """The sequence-specific lead: one tool call per turn (§7.3)."""

    def __init__(self, rid_x):
        self.model = "gpt-5.5"
        self.provider_name = "openai"
        self.capabilities = SessionCapabilities()
        self._budget = []
        self._step = 0
        self.blocks_seen = []
        self.tools_seen = []
        self.rid_x = rid_x
        self.rid_a = None

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("lead", "openai", self.model, 100, 10, 0))
        self.blocks_seen.append(blocks)
        self.tools_seen.append(tools)
        self._step += 1
        step = self._step
        if step == 1:
            content = [ToolUseBlock(id="t1", name="episode_run", input={
                "kind": "verify", "goal": "verify main", "branches": ["main"]})]
        elif step == 2:
            content = [ToolUseBlock(id="t2", name="episode_run", input={
                "kind": "reading", "goal": "read main", "branches": ["main"],
                "max_tool_calls": 2})]
        elif step == 3:
            reading = _find_payload(
                blocks, lambda p: p.get("kind") == "reading" and p.get("reading_id")
            )
            self.rid_a = reading["reading_id"]
            content = [ToolUseBlock(id="t3", name="repair_transaction", input={
                "branch": "main", "reading_id": self.rid_a, "as_name": "tx1"})]
        elif step == 4:
            content = [ToolUseBlock(id="t4", name="repair_transaction", input={
                "branch": "main", "reading_id": self.rid_x, "as_name": "tx2"})]
        elif step == 5:
            content = [ToolUseBlock(id="t5", name="repair_transaction", input={
                "branch": "main", "reading_id": self.rid_x, "as_name": "tx3"})]
        elif step == 6:
            content = [ToolUseBlock(id="t6", name="experiment_submit", input={
                "type": "automated_solver", "branch": "main",
                "config": {"target_language": "en", "allow_homophones": True,
                           "max_runtime_seconds": 60}})]
        elif step == 7:
            failed = _find_payload(
                blocks, lambda p: p.get("corrected_example") is not None
            )
            content = [ToolUseBlock(id="t7", name="experiment_submit", input={
                "type": "automated_solver", "branch": "main",
                "config": failed["corrected_example"]})]
        else:
            content = [ToolUseBlock(id="t8", name="meta_declare_unsolved", input={
                "best_branch": "main", "rationale": "no viable local repair",
                "reading_summary": "middle word remains damaged"})]
        return ModelResponse(content=content, usage=ModelUsage(100, 10, 0))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": self.model, "exchanges": []}


def _run_sequence_b(monkeypatch):
    ct, state = keyed_catton_state()
    content_hash = _candidate_content_hash(
        _decoded_text_for_panel(state.workspace, "main")
    )
    # §7.5: seed only the auxiliary reading (the second distinct interpretation);
    # the first reading is produced live by the turn-2 reading episode.
    rid_x = seed_reading(state, "COTON")

    sessions_mod.register_session_builder(
        "episode:verify", make_verify_builder(NEGATIVE_LOCAL_REPAIR_VERDICT))
    sessions_mod.register_session_builder(
        "episode:reading", OverBudgetReadingWorkerFake)
    register_programmable_repair([
        {"apply": [{}], "result": _NO_FORK_APPLIED},   # evidence failure 1
        {"apply": [], "result": _NO_FORK_APPLIED},     # evidence failure 2 -> exhausted
    ])
    # The REAL Slice-5 typed validation runs; only the solver compute is trivial.
    monkeypatch.setitem(
        experiments.EXPERIMENT_TYPES["automated_solver"], "runner", _fake_runner)

    lead = SequenceBLead(rid_x)
    try:
        art = run_v3(
            ct, session=lead, language="en", max_iterations=10,
            cipher_id="v3_sequence_b", resume_state=state,
            experiment_queue=ExperimentQueue(synchronous=True),
        )
    finally:
        for role in ("episode:verify", "episode:reading", "episode:repair"):
            sessions_mod._SESSION_BUILDERS.pop(role, None)
    return art, lead, content_hash, rid_x


# --- assertion clusters (mapped 1:1 to master 540-552) ----------------------

def _assert_verification_early(art):
    assert art.episodes[0]["kind"] == "verify"
    assert art.episodes[0]["status"] == "ok"
    assert art.attestations[0]["created_turn"] == 1
    assert attestation_is_positive(art.attestations[0]) is False


def _assert_reading_feeds_bounded_repair(art, lead, rid_x):
    inv = art.investigation_state
    assert inv["repair_transactions"][0]["reading_id"] == lead.rid_a
    repair_entries = [e for e in art.episodes if e["kind"] == "repair"]
    assert repair_entries
    for entry in repair_entries:
        assert entry["registered_max_tool_calls"] == 6
        assert entry["tool_call_count"] <= 6
    reading_ids = {r["reading_id"] for r in art.readings}
    assert reading_ids == {lead.rid_a, rid_x}


def _assert_reading_reserve_submit(art):
    reading = next(e for e in art.episodes if e["kind"] == "reading")
    assert reading["requested_max_tool_calls"] == 2
    assert reading["budget"]["max_tool_calls"] == 2
    assert reading["registered_max_tool_calls"] == 16
    assert reading["tool_call_count"] == 2
    assert reading["suppressed_over_budget_calls"] == 2
    assert reading["status"] == "ok"
    assert reading["result"]["reading_text"]


def _assert_repair_saturates(art, content_hash, att_key):
    inv = art.investigation_state
    transactions = inv["repair_transactions"]
    assert len(transactions) == 2
    for tx in transactions:
        assert tx["status"] == "failed"
        assert tx["reason"] == "no_changed_finalists"
        assert tx["failure_class"] == "evidence"
        assert tx["counted_evidence_failure"] is True
    entry = inv["repair_saturation"][saturation_key(content_hash, att_key)]
    assert entry["evidence_failures"] == 2
    assert entry["exhausted"] is True
    tx3 = [json.loads(tc.result) for tc in art.tool_calls
           if tc.tool_name == "repair_transaction" and tc.episode_id is None][2]
    assert tx3 == {"status": "blocked", "reason": "repair_transaction_not_ready",
                   "workflow_state": "repair_exhausted"}


def _assert_alternate_search(art, lead, content_hash, att_key):
    # (a) offered on the lead context of a turn >= 6.
    offered = any(
        isinstance(block, dict) and block.get("type") == "text"
        and "Workflow state: repair_exhausted" in block.get("text", "")
        and "experiment_submit" in block.get("text", "")
        for turn_blocks in lead.blocks_seen[5:]
        for message in turn_blocks
        for block in (message.get("content") or [])
    )
    assert offered
    # (b)/(c): the invalid attempt is rejected with a corrected example; the
    # valid one runs to completion; exactly one record entered the queue.
    payloads = _tool_result_payloads(art.messages)
    invalid = next(p for p in payloads if p.get("error") == "invalid experiment config")
    assert any("target_language" in str(err) for err in invalid["config_errors"])
    assert "corrected_example" in invalid
    completed = next(
        p for p in payloads
        if p.get("experiment_id") and p.get("status") == "completed"
    )
    assert len(art.experiments) == 1
    # (d): the saturation entry points at the accepted experiment.
    entry = art.investigation_state["repair_saturation"][
        saturation_key(content_hash, att_key)]
    assert entry["pending_experiment_id"] == completed["experiment_id"]


def _assert_no_worker_over_budget(art):
    for entry in art.episodes:
        assert entry["tool_call_count"] <= entry["budget"]["max_tool_calls"]


def _assert_honest_termination(art):
    assert art.status == "unsolved"
    assert art.solution is None
    assert art.auto_declared is False
    assert art.attested_fallback is False
    assert art.branch_roles == {
        "best_scored_branch": "main",
        "workflow_branch": "main",
        "latest_installed_branch": None,
        "declared_or_selected_branch": "main",
    }
    # Exchange pairing: every tool_use id has exactly one tool_result.
    uses, results = set(), []
    for message in art.messages:
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                uses.add(block["id"])
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.append(block["tool_use_id"])
    assert uses == set(results)
    assert len(results) == len(set(results))


def _assert_analyzer_reports_full_path(art, content_hash):
    data = json.loads(json.dumps(art.to_dict()))
    facts = inspect_artifact.derive_run_facts(data)
    assert facts["status"] == "unsolved"
    assert facts["loop_version"] == "v3"
    assert facts["provider"] == "openai"
    assert facts["model"] == "gpt-5.5"
    assert facts["declared"] is False
    assert facts["final_branch"] == "main"

    budgets = inspect_artifact.format_episode_budgets(data)
    reading_row = next(
        line for line in budgets.splitlines() if line.strip().startswith("reading")
    )
    for token in ("2", "16"):
        assert token in reading_row.split()
    # requested/registered/effective/executed/skipped = 2 16 2 2 2
    assert reading_row.split()[1:6] == ["2", "16", "2", "2", "2"]

    assert "reading" in inspect_artifact.format_suppressed_calls(data)

    cycles = inspect_artifact.format_repair_cycles(data)
    assert cycles.count(" tx  ") == 1
    assert content_hash[:12] in cycles
    assert "2 pairs" in cycles

    saturation = inspect_artifact.format_saturation(data)
    assert "exhausted=True" in saturation
    assert "evidence_failures=2" in saturation
    assert "exhausted at turn 4" in saturation
    pending = art.investigation_state["repair_saturation"][
        list(art.investigation_state["repair_saturation"])[0]]["pending_experiment_id"]
    assert pending in saturation

    transactions = inspect_artifact.format_repair_transactions(data)
    failed_evidence = [
        line for line in transactions.splitlines()
        if "failed" in line and "class=evidence" in line
    ]
    assert len(failed_evidence) == 2

    exp_failures = inspect_artifact.format_experiment_validation_failures(
        inspect_artifact.build_timeline(data))
    assert "target_language" in exp_failures
    assert "corrected_example: yes" in exp_failures

    roles = inspect_artifact.format_branch_roles(data)
    for role in ("best_scored_branch", "workflow_branch",
                 "latest_installed_branch", "declared_or_selected_branch"):
        assert role in roles

    hyp = inspect_artifact.format_repair_hypothesis_time(data)
    assert any(
        "hypothesis_apply_reading" in line and "1 calls" in line
        for line in hyp.splitlines()
    )


def test_sequence_b_m5_2_shaped_replay_end_to_end(monkeypatch):
    art, lead, content_hash, rid_x = _run_sequence_b(monkeypatch)
    att_key = "ep:" + art.attestations[0]["episode_id"]

    _assert_verification_early(art)                                   # 1
    _assert_reading_feeds_bounded_repair(art, lead, rid_x)           # 2
    _assert_reading_reserve_submit(art)                             # 3
    _assert_repair_saturates(art, content_hash, att_key)            # 4
    _assert_alternate_search(art, lead, content_hash, att_key)      # 5
    _assert_no_worker_over_budget(art)                             # 6
    _assert_honest_termination(art)                                # 7
    _assert_analyzer_reports_full_path(art, content_hash)          # 8
