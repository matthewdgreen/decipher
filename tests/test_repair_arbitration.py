"""Verifier-arbitrated repair acceptance (v3/host level).

Spec: docs/specs/verifier_arbitrated_repair_spec.md (T1-T9, T12).
Evidence: docs/evidence/c56de7e6c600_repair_guard_false_reject.json.

These tests drive the opt-in ``verifier_arbitration`` flag on
``repair_transaction``. They register the arbitration verdict explicitly per
test (NOT the positive ``verify_fake`` fixture) so each case controls exactly
what the independent reader returns; the builder is popped in ``finally``.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agent.model_provider import TextBlock, ToolUseBlock
from investigation import sessions as sessions_mod
from investigation.host import (
    ARBITRATION_FLOOR_RECOVERABILITY,
    ARBITRATION_FLOOR_TLC,
    ARBITRATION_MARGIN,
    ARBITRATION_POLICY_ID,
    _arbitration_verdict,
    _branch_hash,
)
from investigation.loop_v3 import run_v3
from tests.support.scripted_v3 import (
    ScriptedSession,
    keyed_catton_state,
    make_verify_builder,
    register_programmable_repair,
    seed_negative_attestation,
    seed_reading,
)


# --------------------------------------------------------------------------- #
# Fixtures / driver
# --------------------------------------------------------------------------- #
# The scalar-decrease trigger program: applying reading "COTON" onto
# keyed_catton_state (main decodes CATON) deterministically yields quad_delta<0
# so check 8 (scalar_non_decrease) rejects — see
# test_loop_v3.py::test_s4_scalar_decrease_default_denied.
_COTON_PROGRAM = {
    "apply": [{"reading_text": "COTON", "as_name": "coton_fork"}],
    "result": lambda forks: {
        "applied": True, "best_branch": forks[0] if forks else None,
        "edits": [], "verdicts": [], "collateral": {}, "notes": "",
    },
}

# A fabricated-winner program (test_s4_fabricated_winner_rejected shape): the
# worker names a branch it never created, so check 1 (winner_named) rejects
# BEFORE any scoring check — arbitration must never engage.
_FABRICATED_WINNER_PROGRAM = {
    "apply": [{"reading_text": "LATER", "as_name": "real_fork"}],
    "result": lambda forks: {
        "applied": True, "best_branch": "branch_i_invented",
        "edits": [], "verdicts": [], "collateral": {}, "notes": "",
    },
}


def _verdict(tlc, rec, *, accepts_as_solution=False, damage_scope="local",
             repairability="local_repair", anomalies=("x",), coherence=6):
    """A verify-episode result dict (the keys _attestation_record_from_result reads)."""
    return {
        "coherence": coherence,
        "reader_accepts": False,
        "reader_accepts_as_solution": accepts_as_solution,
        "target_language_confidence": tlc,
        "semantic_recoverability": rec,
        "damage_scope": damage_scope,
        "repairability": repairability,
        "uncertainty_note": "",
        "gloss": "partly readable",
        "anomalies": list(anomalies),
        "confidence": "medium",
    }


def _run_arb(program, tool_input, *, reading_text, cipher_id,
             verify_verdict=None, incumbent=False):
    """Run one scripted repair_transaction turn under run_v3 (model_provider=None).

    Returns (artifact, state). ``state`` is the resume_state mutated in place by
    the host, so verify_attestations/workspace/repair_agenda/episode_ledger are
    all inspectable after the run.
    """
    ct, state = keyed_catton_state()
    if incumbent:
        seed_negative_attestation(state)  # incumbent tlc 0.8 / rec 0.7 on main
    seed_reading(state, reading_text)
    register_programmable_repair([program])
    verify_registered = False
    if verify_verdict is not None:
        sessions_mod.register_session_builder(
            "episode:verify", make_verify_builder(verify_verdict))
        verify_registered = True
    try:
        lead = ScriptedSession([
            [ToolUseBlock(id="t1", name="repair_transaction", input=tool_input)],
            [TextBlock(text="stop")],
        ])
        art = run_v3(ct, session=lead, language="en", max_iterations=5,
                     cipher_id=cipher_id, resume_state=state)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)
        if verify_registered:
            sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)
    return art, state


_ACCEPTANCE_KEYS_DEFAULT = {
    "policy", "checks", "supported_forks", "edit_evidence_count",
    "adjudicated", "scores_before", "scores_after", "score_deltas",
}


# --------------------------------------------------------------------------- #
# T1 — mechanical-reject + arbitration-accepts -> installs (margin rule)
# --------------------------------------------------------------------------- #
def test_t1_margin_accept_installs():
    art, state = _run_arb(
        _COTON_PROGRAM,
        {"branch": "main", "as_name": "transaction_repaired",
         "verifier_arbitration": True},
        reading_text="COTON", cipher_id="arb_t1",
        verify_verdict=_verdict(0.9, 0.8, anomalies=["x"]), incumbent=True,
    )
    tx = state.repair_transactions[0]
    assert tx["status"] == "installed"
    assert state.workspace.has_branch("transaction_repaired")
    acc = tx["acceptance"]
    assert acc["policy"] == "default_deny_v1+verifier_arbitration_v1"
    arb = acc["arbitration"]
    assert arb["status"] == "accepted"
    assert arb["rule"] == "margin_improvement"
    assert arb["trigger_check"] == "scalar_non_decrease"
    assert arb["attestation_recorded"] is True
    scalar_entry = next(c for c in acc["checks"] if c["check"] == "scalar_non_decrease")
    assert scalar_entry["passed"] is False
    assert scalar_entry["overruled_by_arbitration"] is True
    last_att = state.verify_attestations[-1]
    assert last_att["branch"] == "transaction_repaired"
    assert last_att["content_hash"] == _branch_hash(state.workspace, "transaction_repaired")
    # Non-positive + local_repair verdict seeds an open agenda item for the
    # INSTALLED branch (its content hash != source, so REP-6 does not close it).
    assert any(
        item.get("branch") == "transaction_repaired"
        and item.get("status") == "open"
        and item.get("source") == "verify_attestation"
        for item in state.repair_agenda
    )
    assert [e["kind"] for e in state.episode_ledger] == ["repair", "verify"]


# --------------------------------------------------------------------------- #
# T2 — mechanical-reject + arbitration-rejects -> fails
# --------------------------------------------------------------------------- #
def test_t2_margin_reject_fails():
    art, state = _run_arb(
        _COTON_PROGRAM,
        {"branch": "main", "as_name": "transaction_repaired",
         "verifier_arbitration": True},
        reading_text="COTON", cipher_id="arb_t2",
        verify_verdict=_verdict(0.75, 0.65), incumbent=True,  # below incumbent
    )
    tx = state.repair_transactions[0]
    assert tx["status"] == "failed"
    assert tx["reason"] == "materially_non_improving"
    assert tx["failure_class"] == "evidence"
    assert tx["counted_evidence_failure"] is True
    entries = list(state.repair_saturation.values())
    assert len(entries) == 1
    assert entries[0]["evidence_failures"] == 1
    arb = tx["acceptance"]["arbitration"]
    assert arb["status"] == "rejected"
    assert arb["rule"] is None
    assert arb["repaired_attestation"] is not None
    assert arb["incumbent_attestation"] is not None
    assert not state.workspace.has_branch("transaction_repaired")
    # The arbitration attestation is NOT written to state on a reject.
    assert len(state.verify_attestations) == 1  # only the seeded incumbent
    assert not any(
        item.get("source") == "verify_attestation" for item in state.repair_agenda
    )


# --------------------------------------------------------------------------- #
# T3 — flag default false -> today's behavior byte-identical
# --------------------------------------------------------------------------- #
def test_t3_default_false_byte_identical():
    art, state = _run_arb(
        _COTON_PROGRAM,
        {"branch": "main", "as_name": "transaction_repaired"},  # no flag
        reading_text="COTON", cipher_id="arb_t3",
        verify_verdict=None, incumbent=True,  # no verify builder registered
    )
    tx = state.repair_transactions[0]
    assert tx["status"] == "failed"
    assert tx["reason"] == "materially_non_improving"
    assert tx["failure_class"] == "evidence"
    assert tx["counted_evidence_failure"] is True
    failing = next(c for c in tx["acceptance"]["checks"] if not c["passed"])
    assert failing["check"] == "scalar_non_decrease"
    acc = tx["acceptance"]
    # Byte-identical acceptance record: no `arbitration` key, policy unchanged.
    assert set(acc.keys()) == _ACCEPTANCE_KEYS_DEFAULT
    assert acc["policy"] == "default_deny_v1"
    # No verify episode ran.
    assert [e["kind"] for e in state.episode_ledger] == ["repair"]


# --------------------------------------------------------------------------- #
# T4 — keyless -> typed unavailable, mechanical reject preserved
# --------------------------------------------------------------------------- #
def test_t4_keyless_typed_unavailable():
    art, state = _run_arb(
        _COTON_PROGRAM,
        {"branch": "main", "as_name": "transaction_repaired",
         "verifier_arbitration": True},
        reading_text="COTON", cipher_id="arb_t4",
        verify_verdict=None, incumbent=True,  # NO verify builder + provider None
    )
    tx = state.repair_transactions[0]
    assert tx["status"] == "failed"
    assert tx["reason"] == "materially_non_improving"
    assert tx["acceptance"]["arbitration"] == {
        "requested": True, "engaged": True,
        "policy_id": ARBITRATION_POLICY_ID,
        "trigger_check": "scalar_non_decrease",
        "constants": {
            "margin": ARBITRATION_MARGIN,
            "floor_target_language_confidence": ARBITRATION_FLOOR_TLC,
            "floor_semantic_recoverability": ARBITRATION_FLOOR_RECOVERABILITY,
        },
        "status": "unavailable", "reason": "no_verification_provider",
    }
    assert not any(e["kind"] == "verify" for e in state.episode_ledger)


# --------------------------------------------------------------------------- #
# T5 — genuinely-worse repair still fails arbitration (regression)
# --------------------------------------------------------------------------- #
def test_t5_genuinely_worse_floor_reject():
    art, state = _run_arb(
        _COTON_PROGRAM,
        {"branch": "main", "as_name": "transaction_repaired",
         "verifier_arbitration": True},
        reading_text="COTON", cipher_id="arb_t5",
        verify_verdict=_verdict(0.30, 0.20), incumbent=False,  # no incumbent
    )
    tx = state.repair_transactions[0]
    assert tx["status"] == "failed"
    assert tx["reason"] == "materially_non_improving"
    arb = tx["acceptance"]["arbitration"]
    assert arb["status"] == "rejected"
    assert arb["rule"] is None
    assert arb["incumbent_attestation"] is None
    assert not state.workspace.has_branch("transaction_repaired")
    assert len(state.verify_attestations) == 0


# --------------------------------------------------------------------------- #
# T6 — floor rule replica of the motivating case (0.97/0.88 installs)
# --------------------------------------------------------------------------- #
def test_t6_floor_rule_motivating_case():
    art, state = _run_arb(
        _COTON_PROGRAM,
        {"branch": "main", "as_name": "transaction_repaired",
         "verifier_arbitration": True},
        reading_text="COTON", cipher_id="arb_t6",
        verify_verdict=_verdict(0.97, 0.88, anomalies=[]), incumbent=False,
    )
    tx = state.repair_transactions[0]
    assert tx["status"] == "installed"
    arb = tx["acceptance"]["arbitration"]
    assert arb["status"] == "accepted"
    assert arb["rule"] == "absolute_floor"


# --------------------------------------------------------------------------- #
# T7 — _arbitration_verdict unit tests (pure function)
# --------------------------------------------------------------------------- #
def test_t7a_reader_accepts_short_circuits():
    ok, rule, _detail = _arbitration_verdict(
        {"reader_accepts_as_solution": True,
         "target_language_confidence": 0.0, "semantic_recoverability": 0.0},
        None,
    )
    assert ok is True and rule == "reader_accepts_as_solution"


def test_t7b_margin_boundary():
    # Zero incumbent so the repaired scalars ARE the deltas exactly (the spec
    # expresses this boundary in delta terms; subtracting non-zero baselines
    # loses float precision at the 0.05 boundary).
    inc = {"target_language_confidence": 0.0, "semantic_recoverability": 0.0}
    ok, rule, _ = _arbitration_verdict(
        {"target_language_confidence": 0.03, "semantic_recoverability": 0.02}, inc)
    assert ok is True and rule == "margin_improvement"  # deltas +0.03/+0.02, sum == 0.05
    ok, rule, _ = _arbitration_verdict(
        {"target_language_confidence": 0.03, "semantic_recoverability": 0.019}, inc)
    assert ok is False and rule is None  # deltas +0.03/+0.019, sum == 0.049


def test_t7c_non_monotone_rejected():
    inc = {"target_language_confidence": 0.8, "semantic_recoverability": 0.7}
    ok, rule, _ = _arbitration_verdict(
        {"target_language_confidence": 1.0, "semantic_recoverability": 0.69}, inc)
    assert ok is False and rule is None  # +0.20, -0.01


def test_t7d_legacy_incumbent_routes_to_floor():
    legacy = {"semantic_recoverability": 0.0}  # no target_language_confidence key
    ok, rule, detail = _arbitration_verdict(
        {"target_language_confidence": 0.95, "semantic_recoverability": 0.65}, legacy)
    assert ok is True and rule == "absolute_floor"
    assert detail["incumbent"] is None


def test_t7e_floor_boundaries():
    assert _arbitration_verdict(
        {"target_language_confidence": 0.90, "semantic_recoverability": 0.60}, None)[0] is True
    assert _arbitration_verdict(
        {"target_language_confidence": 0.899, "semantic_recoverability": 0.60}, None)[0] is False
    assert _arbitration_verdict(
        {"target_language_confidence": 0.90, "semantic_recoverability": 0.599}, None)[0] is False


def test_t7f_missing_scalars_clamp_and_reject():
    ok, rule, _ = _arbitration_verdict({}, None)
    assert ok is False and rule is None


# --------------------------------------------------------------------------- #
# T8 — arbitration never engages on evidence checks (1-5)
# --------------------------------------------------------------------------- #
def test_t8_never_engages_on_evidence_check():
    art, state = _run_arb(
        _FABRICATED_WINNER_PROGRAM,
        {"branch": "main", "as_name": "transaction_repaired",
         "verifier_arbitration": True},
        reading_text="LATER", cipher_id="arb_t8",
        # An ACCEPTING verify builder is registered but must never run.
        verify_verdict=_verdict(0.99, 0.99, accepts_as_solution=True), incumbent=False,
    )
    tx = state.repair_transactions[0]
    assert tx["status"] == "failed"
    assert tx["reason"] == "unsupported_winner"
    assert not any(e["kind"] == "verify" for e in state.episode_ledger)
    assert tx["acceptance"]["arbitration"] == {"requested": True, "engaged": False}


# --------------------------------------------------------------------------- #
# T9 — check-6 (collateral) trigger records full deltas and installs
# --------------------------------------------------------------------------- #
def _build_t9_host():
    """Build a v3-style InvestigationHost over keyed_catton_state and synthesize
    what mcp_server/repair.py synthesizes: one repair_compile ledger entry with a
    single changed winner snapshot plus a hypothesis_test_words ToolCall whose
    adjudication_summary reports damaged 3 / improved 0."""
    from artifact.schema import ToolCall
    from investigation import episodes as episodes_mod
    from investigation import state as state_module
    from mcp_server.runtime import InvestigationRuntime

    _ct, state = keyed_catton_state()
    doc = {"schema_version": 1, "meta": {"investigation_id": "t9", "revision": 1},
           "state": state.to_artifact_dict(), "records": {}}
    runtime = InvestigationRuntime(document=doc, emit=lambda *a, **k: None,
                                   verify_provider=None, max_cost_usd=5.0,
                                   synchronous_experiments=True)
    host = runtime.host
    source_hash = _branch_hash(host.workspace, "main")

    # A changed winner fork built in an isolated scratch workspace (remap 'a' so
    # the decode differs from CATON), serialized into a ledger snapshot.
    scratch = episodes_mod._build_episode_workspace(runtime.state, ["main"])
    scratch.fork("winner_fork", from_branch="main")
    scratch.set_mapping(
        "winner_fork", scratch.cipher_text.alphabet.id_for("a"),
        scratch.plaintext_alphabet.id_for("X"))
    winner_snap = state_module._serialize_branch(scratch, "winner_fork")

    compile_id = "rc_t9compile"
    runtime.state.episode_ledger.append({
        "episode_id": compile_id, "kind": "repair_compile",
        "goal": "client-compiled word hypotheses on main",
        "status": "ok", "failure_reason": None, "result": None,
        "summary": "1 hypotheses probed; 1 changed finalist fork(s)",
        "branch_snapshots": [winner_snap], "tool_call_count": 1,
        "budget_entries": [], "agenda_additions": [],
        "launching_turn": 1, "input_branches": ["main"],
    })
    result_obj = {
        "status": "ok",
        "items": [{
            "status": "ok", "installed_fork": "winner_fork", "edits": [],
            "adjudication_summary": {
                "damaged_occurrences": 3, "improved_occurrences": 0,
            },
        }],
        "finalists": [{"installed_fork": "winner_fork", "edits": []}],
    }
    host.episode_tool_calls.append(ToolCall(
        iteration=1, tool_name="hypothesis_test_words",
        tool_use_id=f"mcp_{compile_id}", arguments={},
        result=json.dumps(result_obj), episode_id=compile_id,
    ))

    base_record = {
        "transaction_id": "tx_t9",
        "source_branch": "main",
        "source_content_hash": source_hash,
        "reading_id": "read_t9",
        "interpretation_id": "read_t9",
        "interpretation_digest": "digest_t9",
        "attestation_key": "none",
        "saturation_key": "sat_t9",
        "pair_digest": "pair_t9",
        "retry_of": None,
        "episode_id": compile_id,
        "addressed_anomalies": [],
        "created_turn": 1,
    }
    episode_payload = {
        "episode_id": compile_id, "status": "ok",
        "result": {"applied": True, "best_branch": "winner_fork", "edits": [],
                   "verdicts": [], "collateral": {}},
    }
    tu = {"id": "t9", "name": "repair_transaction",
          "input": {"branch": "main", "as_name": "t9_repaired"}}
    return host, source_hash, base_record, episode_payload, tu


def test_t9_check6_trigger_records_full_deltas_and_installs():
    host, source_hash, base_record, episode_payload, tu = _build_t9_host()
    sessions_mod.register_session_builder(
        "episode:verify", make_verify_builder(
            _verdict(0.97, 0.88, accepts_as_solution=False, anomalies=[])))
    try:
        rendered = host.validate_and_install_repair(
            tu=tu, turn=1, branch="main", source_hash=source_hash,
            att_key="none", pair="pair_t9", base_record=base_record,
            episode_payload=episode_payload, as_name="t9_repaired",
            verifier_arbitration=True,
        )
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)
    record = json.loads(rendered)
    assert record["status"] == "installed"
    acc = record["acceptance"]
    arb = acc["arbitration"]
    assert arb["status"] == "accepted"
    assert arb["trigger_check"] == "collateral_within_limits"
    checks = {c["check"]: c for c in acc["checks"]}
    assert checks["collateral_within_limits"]["passed"] is False
    assert checks["collateral_within_limits"]["overruled_by_arbitration"] is True
    assert checks["no_op_probe"]["passed"] is True
    assert "scalar_non_decrease" in checks
    # The null-deltas observability gap is CLOSED on the arbitrated path.
    assert acc["scores_after"] is not None
    assert acc["score_deltas"] is not None


def test_t9_check6_flag_false_stops_at_check6():
    # Same fixture, flag false: today's shape — record stops at check 6 with
    # scores_after None (the probe never runs).
    host, source_hash, base_record, episode_payload, tu = _build_t9_host()
    rendered = host.validate_and_install_repair(
        tu=tu, turn=1, branch="main", source_hash=source_hash,
        att_key="none", pair="pair_t9", base_record=base_record,
        episode_payload=episode_payload, as_name="t9_repaired",
        verifier_arbitration=False,
    )
    record = json.loads(rendered)
    assert record["status"] == "failed"
    assert record["reason"] == "materially_non_improving"
    acc = record["acceptance"]
    assert acc["scores_after"] is None
    assert acc["score_deltas"] is None
    assert set(acc.keys()) == _ACCEPTANCE_KEYS_DEFAULT  # no arbitration key
    checks = {c["check"]: c for c in acc["checks"]}
    assert checks["collateral_within_limits"]["passed"] is False
    assert "no_op_probe" not in checks  # stopped at check 6


# --------------------------------------------------------------------------- #
# T11 — batch repair of SCATTERED errors arbitrates the whole fork (case 2).
# ONE damaged symbol (w -> I) occurs in THREE scattered words; a single key edit
# (w:I->W) fixes all three occurrences in ONE changed finalist fork. The
# acceptance pipeline operates on whole-fork CONTENT (nothing is span-local), so
# "distributed damage" installs through the existing surface once the whole-fork
# reader arbitrates. The collateral counter is n-gram-local; a deterministic
# host-level synthesis of "damaged 3 / improved 0" is the robust self-validating
# trigger (per spec §8.1: "MCP or host level").
# --------------------------------------------------------------------------- #
def _build_scattered_host():
    from artifact.schema import ToolCall
    from investigation import episodes as episodes_mod
    from investigation import state as state_module
    from investigation.state import InvestigationState
    from mcp_server.runtime import InvestigationRuntime
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText
    from workspace import Workspace

    raw = "dawn pawn lawn"  # 3 words, `w` scattered across all three
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ws = Workspace(ct)
    pt = ws.plaintext_alphabet
    for sym in alpha.symbols:  # correct key EXCEPT w -> I (decode "DAIN PAIN LAIN")
        target = "I" if sym == "w" else sym.upper()
        ws.set_mapping("main", alpha.id_for(sym), pt.id_for(target))
    state = InvestigationState(workspace=ws, language="en")
    state.turn = 0
    doc = {"schema_version": 1, "meta": {"investigation_id": "t11", "revision": 1},
           "state": state.to_artifact_dict(), "records": {}}
    runtime = InvestigationRuntime(document=doc, emit=lambda *a, **k: None,
                                   verify_provider=None, max_cost_usd=5.0,
                                   synchronous_experiments=True)
    host = runtime.host
    source_hash = _branch_hash(host.workspace, "main")

    # One key edit (w:I->W) in an isolated scratch fork fixes all three words.
    scratch = episodes_mod._build_episode_workspace(runtime.state, ["main"])
    scratch.fork("winner_fork", from_branch="main")
    scratch.set_mapping(
        "winner_fork", scratch.cipher_text.alphabet.id_for("w"),
        scratch.plaintext_alphabet.id_for("W"))
    winner_snap = state_module._serialize_branch(scratch, "winner_fork")

    compile_id = "rc_t11compile"
    runtime.state.episode_ledger.append({
        "episode_id": compile_id, "kind": "repair_compile",
        "goal": "client-compiled word hypotheses on main",
        "status": "ok", "failure_reason": None, "result": None,
        "summary": "1 hypotheses probed; 1 changed finalist fork(s)",
        "branch_snapshots": [winner_snap], "tool_call_count": 1,
        "budget_entries": [], "agenda_additions": [],
        "launching_turn": 1, "input_branches": ["main"],
    })
    result_obj = {
        "status": "ok",
        "items": [{
            "status": "ok", "installed_fork": "winner_fork", "edits": ["w:I->W"],
            "adjudication_summary": {
                "damaged_occurrences": 3, "improved_occurrences": 0,
            },
        }],
        "finalists": [{"installed_fork": "winner_fork", "edits": ["w:I->W"]}],
    }
    host.episode_tool_calls.append(ToolCall(
        iteration=1, tool_name="hypothesis_test_words",
        tool_use_id=f"mcp_{compile_id}", arguments={},
        result=json.dumps(result_obj), episode_id=compile_id,
    ))
    base_record = {
        "transaction_id": "tx_t11", "source_branch": "main",
        "source_content_hash": source_hash, "reading_id": "read_t11",
        "interpretation_id": "read_t11", "interpretation_digest": "digest_t11",
        "attestation_key": "none", "saturation_key": "sat_t11",
        "pair_digest": "pair_t11", "retry_of": None, "episode_id": compile_id,
        "addressed_anomalies": [], "created_turn": 1,
    }
    episode_payload = {
        "episode_id": compile_id, "status": "ok",
        "result": {"applied": True, "best_branch": "winner_fork",
                   "edits": ["w:I->W"], "verdicts": [], "collateral": {}},
    }
    tu = {"id": "t11", "name": "repair_transaction",
          "input": {"branch": "main", "as_name": "t11_repaired"}}
    return host, source_hash, base_record, episode_payload, tu


def test_t11_scattered_batch_installs_whole_fork():
    from agent.loop_shared import _decoded_text_for_panel

    host, source_hash, base_record, episode_payload, tu = _build_scattered_host()
    sessions_mod.register_session_builder(
        "episode:verify", make_verify_builder(_verdict(0.95, 0.85, anomalies=[])))
    try:
        rendered = host.validate_and_install_repair(
            tu=tu, turn=1, branch="main", source_hash=source_hash,
            att_key="none", pair="pair_t11", base_record=base_record,
            episode_payload=episode_payload, as_name="t11_repaired",
            verifier_arbitration=True,
        )
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)
    record = json.loads(rendered)
    acc = record["acceptance"]
    collateral = next(c for c in acc["checks"] if c["check"] == "collateral_within_limits")
    # Self-validating: distributed damage reported (damaged 3 > improved 0).
    assert collateral["damaged_occurrences"] > collateral["improved_occurrences"]
    assert record["status"] == "installed"
    assert acc["arbitration"]["status"] == "accepted"
    assert acc["arbitration"]["trigger_check"] == "collateral_within_limits"
    # One fork; the installed decode shows ALL three scattered corrections.
    decoded = _decoded_text_for_panel(host.workspace, record["installed_branch"]).upper()
    assert "DAWN" in decoded and "PAWN" in decoded and "LAWN" in decoded
