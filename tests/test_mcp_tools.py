"""MCP tool surface — the scripted in-process investigation (Part 11.3)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from investigation import sessions as sessions_mod
from investigation.host import _branch_hash
from tests.support.mcp import (
    BASIN_CORRECT_WORD,
    BASIN_DAMAGED_WORD_INDEX,
    apply_basin,
    call,
    make_server,
    start,
)
from tests.support.scripted_v3 import (
    NEGATIVE_LOCAL_REPAIR_VERDICT,
    VerifyWorkerFake,
    make_verify_builder,
    seed_reading,
)


# --------------------------------------------------------------------- intake
def test_letters_format_matches_crack_path(tmp_path):
    server = make_server(tmp_path)
    body = start(server, ciphertext="ZYXW ZYXW AB")
    assert body["_is_error"] is False
    # Same as the cmd_crack non-canonical path: distinct letters {Z,Y,X,W,A,B}=6.
    assert body["alphabet_size"] == 6
    assert body["tokens"] == 10
    assert body["words"] == 3
    assert "## Cipher" in body["brief"]


def test_canonical_and_auto_detection(tmp_path):
    server = make_server(tmp_path)
    canon = start(server, ciphertext="S001 S002 S003 | S002 S001", format="canonical")
    assert canon["_is_error"] is False and canon["tokens"] == 5
    # auto picks canonical for S-token text, letters otherwise.
    auto_canon = start(server, ciphertext="S001 S002 | S003")
    assert auto_canon["tokens"] == 3
    auto_letters = start(server, ciphertext="hello world")
    assert auto_letters["words"] == 2


def test_empty_and_oversized(tmp_path):
    server = make_server(tmp_path)
    assert start(server, ciphertext="   ")["reason"] == "empty_ciphertext"
    assert start(server, ciphertext="a" * 200_001)["reason"] == "ciphertext_too_large"


# ------------------------------------------------------------- status + reads
def test_status_brief_headers_and_advisory(tmp_path):
    server = make_server(tmp_path)
    iid = start(server)["investigation_id"]
    body = call(server, "investigation_status", investigation_id=iid)
    brief = body["brief"]
    for header in ("## Cipher", "## Diagnostic fingerprint", "## Investigation state",
                   "## Host guidance"):
        assert header in brief
    assert "[WF-1 advisory]" in brief
    assert "[DECL-1 ENFORCED]" in brief
    assert body["verification_available"] is False
    assert "UNAVAILABLE" in brief


def test_overview_and_diagnosis(tmp_path):
    server = make_server(tmp_path)
    iid = start(server)["investigation_id"]
    ov = call(server, "observe_overview", investigation_id=iid)
    assert "Diagnostic fingerprint" in ov["overview"]
    diag = call(server, "observe_diagnosis", investigation_id=iid)
    assert isinstance(diag.get("report"), dict)


def test_decode_show_dup1(tmp_path):
    server = make_server(tmp_path)
    iid = start(server)["investigation_id"]
    first = call(server, "decode_show", investigation_id=iid, branch="main")
    second = call(server, "decode_show", investigation_id=iid, branch="main")
    assert first.get("status") != "duplicate_suppressed"
    assert second["status"] == "duplicate_suppressed"


def test_branch_create_bumps_and_next_steps(tmp_path):
    server = make_server(tmp_path)
    b = start(server)
    iid, rev = b["investigation_id"], b["revision"]
    created = call(server, "hypothesis_branch_create", investigation_id=iid,
                   expected_revision=rev, new_name="polyhyp",
                   cipher_mode="periodic_polyalphabetic", rationale="IC suggests it")
    assert created["revision"] == rev + 1
    brief = call(server, "investigation_status", investigation_id=iid)["brief"]
    assert "polyhyp" in brief
    steps = call(server, "hypothesis_next_steps", investigation_id=iid)
    assert steps["advisory"] is True
    assert steps["policy_ids"] == ["WF-1"]


def test_candidate_list_and_show(tmp_path):
    server = make_server(tmp_path)
    b = start(server)
    iid = b["investigation_id"]
    cl = call(server, "candidate_list", investigation_id=iid)
    entry = cl["candidates"][0]
    assert set(entry) >= {"scores", "roles", "verification", "content_hash"}
    assert entry["verification"] == "unavailable"  # keyless
    cs = call(server, "candidate_show", investigation_id=iid, branch="main")
    assert "decoded_text" in cs and "attestation_history" in cs


# ------------------------------------------------------------------- records
def test_reading_record_and_sat4(tmp_path):
    server = make_server(tmp_path)
    b = start(server)
    iid, rev = b["investigation_id"], b["revision"]
    rt = server._runtimes[iid]
    apply_basin(rt)
    rec = call(server, "reading_record", investigation_id=iid, expected_revision=rev,
               branch="main", reading_text="THE QUICK BROWN FOXES JUMPED",
               overall_confidence=0.8)
    assert rec["status"] == "ok"
    stored = rt.state.readings[rec["reading_id"]]
    assert stored["source"] == "client:unknown"
    assert stored["candidate_content_hash"] == rec["content_hash"]
    # SAT-4: a second reading on unchanged content+evidence is suppressed.
    dup = call(server, "reading_record", investigation_id=iid, expected_revision=rec["revision"],
               branch="main", reading_text="THE QUICK BROWN FOXES JUMPED",
               overall_confidence=0.8)
    assert dup["reason"] == "duplicate_reading_suppressed"
    assert dup["existing_reading_id"] == rec["reading_id"]


def test_comparison_record_cmp2_split(tmp_path):
    server = make_server(tmp_path)
    b = start(server)
    iid, rev = b["investigation_id"], b["revision"]
    # a second branch to compare against
    call(server, "hypothesis_branch_create", investigation_id=iid, expected_revision=rev,
         new_name="alt", cipher_mode="mono", rationale="x")
    cmp_body = call(server, "comparison_record", investigation_id=iid, expected_revision=2,
                    branches=["main", "alt"], ranking=["main", "alt"],
                    best_partial=None, accepts_as_solution=False, rationale="both weak")
    assert cmp_body["best_partial"] is None
    assert cmp_body["accepts_as_solution"] is False
    assert set(cmp_body["branch_hashes"]) == {"main", "alt"}
    # invalid branch -> error
    bad = call(server, "comparison_record", investigation_id=iid, expected_revision=cmp_body["revision"],
               branches=["main", "ghost"], ranking=["main"], best_partial=None,
               accepts_as_solution=False, rationale="x")
    assert bad["status"] == "error"
    # the comparison shows up in the brief
    brief = call(server, "investigation_status", investigation_id=iid)["brief"]
    assert "Client comparisons" in brief


# --------------------------------------------------------------- experiments
def test_experiment_submit_invalid_and_valid(tmp_path):
    server = make_server(tmp_path)
    b = start(server)
    iid, rev = b["investigation_id"], b["revision"]
    bad = call(server, "experiment_submit", investigation_id=iid, expected_revision=rev,
               type="automated_solver", branch="main", config={"max_runtime_seconds": 5})
    assert "corrected_example" in bad
    rev = bad["revision"]  # a mutate call commits even when it returns a domain error
    ok = call(server, "experiment_submit", investigation_id=iid, expected_revision=rev,
              type="automated_solver", branch="main", config={})
    assert ok.get("experiment_id")
    rev = ok["revision"]
    coll = call(server, "experiment_collect", investigation_id=iid, expected_revision=rev,
                experiment_id=ok["experiment_id"])
    assert coll.get("experiment_id") == ok["experiment_id"]
    rev = coll["revision"]
    dup = call(server, "experiment_submit", investigation_id=iid, expected_revision=rev,
               type="automated_solver", branch="main", config={})
    assert dup.get("deduplicated") is True


def test_experiment_submit_surface_advertises_quagmire3_shotgun():
    """The MCP definition table exposes experiment_submit naming quagmire3_shotgun,
    and the mutate envelope preserves the anyOf config while adding the envelope
    keys."""
    from mcp_server.tools import MCP_TOOL_DEFINITIONS

    submit = next(d for d in MCP_TOOL_DEFINITIONS if d["name"] == "experiment_submit")
    assert "quagmire3_shotgun" in submit["description"]
    props = submit["input_schema"]["properties"]
    # Envelope keys added by _mutate_envelope.
    assert "investigation_id" in props and "expected_revision" in props
    # The anyOf config union survived the envelope wrap.
    titles = {b.get("title") for b in props["config"]["anyOf"]}
    assert titles == {
        "automated_solver config",
        "quagmire3_shotgun config",
        "composite_substitution_transposition config",
    }


# ------------------------------------------------------------- manifest fold
# I-0 pin: the MCP tools/list projection must stay byte-identical after the
# service-layer extraction, and the operation manifest must be the single
# source of truth (TOOL_CLASSES folded in, no parallel drift).
_EXPECTED_TOOL_ORDER = [
    "investigation_start", "investigation_list", "investigation_status",
    "observe_overview", "observe_diagnosis", "decode_show",
    "hypothesis_branch_create", "hypothesis_branch_update",
    "hypothesis_branch_reject", "hypothesis_next_steps",
    "candidate_list", "candidate_show", "experiment_submit",
    "experiment_collect", "reading_record", "comparison_record",
    "repair_hypotheses_test", "repair_transaction",
    "request_independent_verification", "branch_adjudicate",
    "act_set_model_variant", "meta_declare_solution", "meta_declare_unsolved",
]
_EXPECTED_CLI_VERBS = {
    "investigation_start": "start", "investigation_list": "list",
    "investigation_status": "status", "observe_overview": "overview",
    "observe_diagnosis": "diagnose", "decode_show": "decode",
    "hypothesis_branch_create": "branch-create",
    "hypothesis_branch_update": "branch-update",
    "hypothesis_branch_reject": "branch-reject",
    "hypothesis_next_steps": "next-steps", "candidate_list": "candidates",
    "candidate_show": "candidate", "experiment_submit": "experiment-submit",
    "experiment_collect": "experiment-collect", "reading_record": "reading-record",
    "comparison_record": "comparison-record", "repair_hypotheses_test": "repair-test",
    "repair_transaction": "repair-transaction",
    "request_independent_verification": "verify",
    "branch_adjudicate": "adjudicate", "act_set_model_variant": "set-model-variant",
    "meta_declare_solution": "declare-solution",
    "meta_declare_unsolved": "declare-unsolved",
}


def test_manifest_projection_and_fold_are_pinned():
    from investigation_service import manifest
    from mcp_server import tools

    # tools/list projection: exact order, exact key set, exact count.
    rendered = tools.render_tool_list()
    assert [t["name"] for t in rendered] == _EXPECTED_TOOL_ORDER
    for t in rendered:
        assert set(t) == {"name", "description", "inputSchema"}
    # Idempotent + stable across the tools shim and the manifest projection.
    assert rendered == manifest.render_tool_list()
    assert rendered == tools.render_tool_list()

    # TOOL_CLASSES is DERIVED from the manifest (fold, not a parallel map).
    assert tools.TOOL_CLASSES == {op.name: op.operation_class for op in manifest.OPERATIONS}
    assert set(tools.TOOL_NAMES) == set(_EXPECTED_TOOL_ORDER)

    # Friendly CLI verbs are explicit public metadata on each OperationSpec.
    assert {op.name: op.cli_verb for op in manifest.OPERATIONS} == _EXPECTED_CLI_VERBS

    # External-effect metadata: only verify is unconditional, only repair is
    # conditional, everything else is never.
    effects = {op.name: op.external_effect for op in manifest.OPERATIONS}
    assert effects["request_independent_verification"] == manifest.ExternalEffect.UNCONDITIONAL
    assert effects["repair_transaction"] == manifest.ExternalEffect.CONDITIONAL
    assert all(
        e == manifest.ExternalEffect.NEVER
        for n, e in effects.items()
        if n not in {"request_independent_verification", "repair_transaction"}
    )


# ---------------------------------------------------------- verify + gates
def test_keyless_verify_and_declare_blocked(tmp_path):
    server = make_server(tmp_path)
    b = start(server)
    iid, rev = b["investigation_id"], b["revision"]
    ver = call(server, "request_independent_verification", investigation_id=iid,
               expected_revision=rev, branch="main")
    assert ver["status"] == "unavailable"
    assert ver["reason"] == "no_verification_provider"
    assert ver["revision"] == rev  # revision UNCHANGED (no commit)
    decl = call(server, "meta_declare_solution", investigation_id=iid, expected_revision=rev,
                branch="main", rationale="x", self_confidence=0.9, reading_summary="x",
                further_iterations_helpful=False, further_iterations_note="x")
    assert decl["reason"] == "attestation_required"


def test_scripted_positive_verify_unlocks_declaration(tmp_path):
    sessions_mod.register_session_builder("episode:verify", VerifyWorkerFake)
    try:
        server = make_server(tmp_path, verify="dummy")
        b = start(server)
        iid, rev = b["investigation_id"], b["revision"]
        ver = call(server, "request_independent_verification", investigation_id=iid,
                   expected_revision=rev, branch="main")
        assert ver.get("attestation") is not None
        rt = server._runtimes[iid]
        att = rt.state.verify_attestations[-1]
        assert att["content_hash"] == _branch_hash(rt.workspace, "main")
        rev = ver["revision"]
        decl = call(server, "meta_declare_solution", investigation_id=iid, expected_revision=rev,
                    branch="main", rationale="reads well", self_confidence=0.95,
                    reading_summary="ok", further_iterations_helpful=False,
                    further_iterations_note="done")
        assert decl["terminal_status"] == "solved"
        assert rt.meta["status"] == "solved"
        # subsequent mutation blocked
        after = call(server, "hypothesis_branch_create", investigation_id=iid,
                     expected_revision=decl["revision"], new_name="x", cipher_mode="m",
                     rationale="y")
        assert after["reason"] == "investigation_terminal"
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)


def test_scripted_negative_verify_blocks_and_seeds_agenda(tmp_path):
    sessions_mod.register_session_builder(
        "episode:verify", make_verify_builder(NEGATIVE_LOCAL_REPAIR_VERDICT))
    try:
        server = make_server(tmp_path, verify="dummy")
        b = start(server)
        iid, rev = b["investigation_id"], b["revision"]
        ver = call(server, "request_independent_verification", investigation_id=iid,
                   expected_revision=rev, branch="main")
        rev = ver["revision"]
        rt = server._runtimes[iid]
        # repairability == local_repair with anomalies -> agenda seeded.
        assert any(item.get("source") == "verify_attestation" for item in rt.state.repair_agenda)
        decl = call(server, "meta_declare_solution", investigation_id=iid, expected_revision=rev,
                    branch="main", rationale="x", self_confidence=0.5, reading_summary="x",
                    further_iterations_helpful=True, further_iterations_note="x")
        assert decl["reason"] == "attestation_not_positive"
        assert decl["attestation"]["reader_accepts_as_solution"] is False
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)


def test_declare_unsolved_always_succeeds(tmp_path):
    server = make_server(tmp_path)
    b = start(server)
    iid, rev = b["investigation_id"], b["revision"]
    decl = call(server, "meta_declare_unsolved", investigation_id=iid, expected_revision=rev,
                rationale="resists", branches_considered=["main"], reading_summary="nada",
                further_iterations_helpful=False, further_iterations_note="n/a")
    assert decl["terminal_status"] == "unsolved"
    assert server._runtimes[iid].meta["status"] == "unsolved"


# ------------------------------------------------------------------- repair
def _basin_ready(tmp_path):
    server = make_server(tmp_path)
    b = start(server)
    iid, rev = b["investigation_id"], b["revision"]
    rt = server._runtimes[iid]
    apply_basin(rt)
    return server, iid, rev, rt


def test_repair_compile_and_dedup(tmp_path):
    server, iid, rev, rt = _basin_ready(tmp_path)
    rec = call(server, "reading_record", investigation_id=iid, expected_revision=rev,
               branch="main", reading_text="THE QUICK BROWN FOXES JUMPED",
               overall_confidence=0.8)
    rev = rec["revision"]
    comp = call(server, "repair_hypotheses_test", investigation_id=iid, expected_revision=rev,
                branch="main", hypotheses=[{"word": BASIN_CORRECT_WORD,
                                            "word_index": BASIN_DAMAGED_WORD_INDEX}])
    assert comp["status"] == "ok" and comp["changed_finalists"]
    assert any(e.get("kind") == "repair_compile" for e in rt.state.episode_ledger)
    rev = comp["revision"]
    dup = call(server, "repair_hypotheses_test", investigation_id=iid, expected_revision=rev,
               branch="main", hypotheses=[{"word": BASIN_CORRECT_WORD,
                                           "word_index": BASIN_DAMAGED_WORD_INDEX}])
    assert dup["status"] == "duplicate_suppressed"
    assert dup["compile_id"] == comp["compile_id"]


def test_repair_transaction_requires_reading_then_installs(tmp_path):
    server, iid, rev, rt = _basin_ready(tmp_path)
    # compile first (no reading yet)
    comp = call(server, "repair_hypotheses_test", investigation_id=iid, expected_revision=rev,
                branch="main", hypotheses=[{"word": BASIN_CORRECT_WORD,
                                            "word_index": BASIN_DAMAGED_WORD_INDEX}])
    rev = comp["revision"]
    winner = comp["changed_finalists"][0]["branch"]
    # REP-1: no reading -> fresh_reading_required
    blocked = call(server, "repair_transaction", investigation_id=iid, expected_revision=rev,
                   branch="main", compile_id=comp["compile_id"], winner=winner)
    assert blocked["reason"] == "fresh_reading_required"
    rev = blocked["revision"]  # blocked mutate still commits (turn bump)
    # record a reading, then install
    rec = call(server, "reading_record", investigation_id=iid, expected_revision=rev,
               branch="main", reading_text="THE QUICK BROWN FOXES JUMPED",
               overall_confidence=0.8)
    rev = rec["revision"]
    tx = call(server, "repair_transaction", investigation_id=iid, expected_revision=rev,
              branch="main", compile_id=comp["compile_id"], winner=winner)
    assert tx["status"] == "installed"
    assert tx["reverification_required"] is True
    assert all(c.get("passed") for c in tx["acceptance"]["checks"])
    assert rt.workspace.has_branch(tx["installed_branch"])


def test_repair_stale_compile(tmp_path):
    server, iid, rev, rt = _basin_ready(tmp_path)
    rec = call(server, "reading_record", investigation_id=iid, expected_revision=rev,
               branch="main", reading_text="THE QUICK BROWN FOXES JUMPED",
               overall_confidence=0.8)
    rev = rec["revision"]
    comp = call(server, "repair_hypotheses_test", investigation_id=iid, expected_revision=rev,
                branch="main", hypotheses=[{"word": BASIN_CORRECT_WORD,
                                            "word_index": BASIN_DAMAGED_WORD_INDEX}])
    rev = comp["revision"]
    winner = comp["changed_finalists"][0]["branch"]
    # White-box: change main's content, re-read on the new content so the
    # precondition passes but the compile is now stale.
    ws = rt.workspace
    alpha = ws.cipher_text.alphabet
    ws.set_mapping("main", alpha.id_for("t"), ws.plaintext_alphabet.id_for("Z"))
    rec2 = call(server, "reading_record", investigation_id=iid, expected_revision=rev,
                branch="main", reading_text="ZHE QUICK BXOWN FOXES JUMPED",
                overall_confidence=0.7)
    rev = rec2["revision"]
    tx = call(server, "repair_transaction", investigation_id=iid, expected_revision=rev,
              branch="main", compile_id=comp["compile_id"], winner=winner)
    assert tx["reason"] == "stale_compile"


def test_repair_unsupported_winner_and_duplicate_and_saturation(tmp_path):
    server, iid, rev, rt = _basin_ready(tmp_path)
    rec = call(server, "reading_record", investigation_id=iid, expected_revision=rev,
               branch="main", reading_text="THE QUICK BROWN FOXES JUMPED",
               overall_confidence=0.8)
    rev = rec["revision"]
    comp = call(server, "repair_hypotheses_test", investigation_id=iid, expected_revision=rev,
                branch="main", hypotheses=[{"word": BASIN_CORRECT_WORD,
                                            "word_index": BASIN_DAMAGED_WORD_INDEX}])
    rev = comp["revision"]
    winner = comp["changed_finalists"][0]["branch"]
    # (a) winner not a compile fork -> failed unsupported_winner
    bad = call(server, "repair_transaction", investigation_id=iid, expected_revision=rev,
               branch="main", compile_id=comp["compile_id"], winner="not_a_fork")
    assert bad["status"] == "failed"
    assert bad["reason"] in {"unsupported_winner", "winner_fork_from_failed_call"}
    rev = bad["revision"]
    # (b) install the real winner, then a duplicate source+interpretation is suppressed
    tx = call(server, "repair_transaction", investigation_id=iid, expected_revision=rev,
              branch="main", compile_id=comp["compile_id"], winner=winner)
    assert tx["status"] == "installed"
    rev = tx["revision"]
    dup = call(server, "repair_transaction", investigation_id=iid, expected_revision=rev,
               branch="main", compile_id=comp["compile_id"], winner=winner,
               reading_id=rec["reading_id"])
    assert dup["status"] == "duplicate_suppressed"


def test_saturation_latch_blocks_and_brief_shows_enforced(tmp_path):
    from investigation.state import saturation_key
    server, iid, rev, rt = _basin_ready(tmp_path)
    rec = call(server, "reading_record", investigation_id=iid, expected_revision=rev,
               branch="main", reading_text="THE QUICK BROWN FOXES JUMPED",
               overall_confidence=0.8)
    rev = rec["revision"]
    comp = call(server, "repair_hypotheses_test", investigation_id=iid, expected_revision=rev,
                branch="main", hypotheses=[{"word": BASIN_CORRECT_WORD,
                                            "word_index": BASIN_DAMAGED_WORD_INDEX}])
    rev = comp["revision"]
    winner = comp["changed_finalists"][0]["branch"]
    # White-box: latch the (content, no-attestation) saturation entry exhausted
    # — two evidence failures is the durable state SAT-3 enforces.
    content_hash = rec["content_hash"]
    key = saturation_key(content_hash, "none")
    entry = rt.state.repair_saturation.setdefault(key, {})
    entry.update({"exhausted": True, "evidence_failures": 2,
                  "candidate_content_hash": content_hash, "attestation_key": "none"})
    tx = call(server, "repair_transaction", investigation_id=iid, expected_revision=rev,
              branch="main", compile_id=comp["compile_id"], winner=winner)
    assert tx["reason"] == "repair_saturated"
    brief = call(server, "investigation_status", investigation_id=iid)["brief"]
    assert "[SAT-3 ENFORCED]" in brief


def test_check_repair_preconditions_no_reading_returns_blocked(tmp_path):
    # Host-refactor pin: called directly on a v3-style host built from a fixture.
    from tests.support.scripted_v3 import keyed_catton_state
    from mcp_server.runtime import InvestigationRuntime

    _ct, state = keyed_catton_state()
    doc = {"schema_version": 1, "meta": {"investigation_id": "x", "revision": 1},
           "state": state.to_artifact_dict(), "records": {}}
    runtime = InvestigationRuntime(document=doc, emit=lambda *a, **k: None,
                                   verify_provider=None, max_cost_usd=5.0,
                                   synchronous_experiments=True)
    pre = runtime.host.check_repair_preconditions(branch="main", reading_id_arg="", turn=1)
    assert "blocked" in pre
    assert pre["blocked"]["reason"] == "fresh_reading_required"


# ------------------------------------------ P4: decoded-branch repair guard
# declaration_hardening_spec.md §5: repair of a slim-record decoded branch (panel
# text served from metadata['decoded_text'], with or without an inherited
# snapshot key) returns a structured decoded_branch_no_base_key; a mask+key
# render (null-mask branch) keeps repair available.
def _install_decoded_branch(ws, name, *, text, cipher_mode):
    ws.fork(name, "main")
    ws.set_full_key(name, {})  # slim record: no per-symbol base key
    b = ws.get_branch(name)
    b.metadata["decoded_text"] = text
    b.metadata["cipher_mode"] = cipher_mode
    return b


def test_repair_hypotheses_test_blocks_decoded_branch(tmp_path):
    server, iid, rev, rt = _basin_ready(tmp_path)
    _install_decoded_branch(
        rt.workspace, "decoded", text="THE QUICK BROWN FOXES JUMPED",
        cipher_mode="composite_substitution_transposition")
    out = call(server, "repair_hypotheses_test", investigation_id=iid,
               expected_revision=rev, branch="decoded",
               hypotheses=[{"word": BASIN_CORRECT_WORD,
                            "word_index": BASIN_DAMAGED_WORD_INDEX}])
    assert out["status"] == "error"
    assert out["reason"] == "decoded_branch_no_base_key"
    assert "composite_substitution_transposition" in out["detail"]
    # A keyed branch (main, keyed by apply_basin) is untouched by the guard.
    out2 = call(server, "repair_hypotheses_test", investigation_id=iid,
                expected_revision=out["revision"], branch="main",
                hypotheses=[{"word": BASIN_CORRECT_WORD,
                             "word_index": BASIN_DAMAGED_WORD_INDEX}])
    assert out2.get("reason") != "decoded_branch_no_base_key"


def test_check_repair_preconditions_decoded_branch_blocked(tmp_path):
    from tests.support.scripted_v3 import keyed_catton_state
    from mcp_server.runtime import InvestigationRuntime

    _ct, state = keyed_catton_state()
    doc = {"schema_version": 1, "meta": {"investigation_id": "x", "revision": 1},
           "state": state.to_artifact_dict(), "records": {}}
    runtime = InvestigationRuntime(document=doc, emit=lambda *a, **k: None,
                                   verify_provider=None, max_cost_usd=5.0,
                                   synchronous_experiments=True)
    _install_decoded_branch(runtime.workspace, "decoded", text="SOMEDECODEDTEXT",
                            cipher_mode="quagmire3")
    pre = runtime.host.check_repair_preconditions(
        branch="decoded", reading_id_arg="", turn=1)
    assert "blocked" in pre
    assert pre["blocked"]["status"] == "error"
    assert pre["blocked"]["reason"] == "decoded_branch_no_base_key"
    # A keyed branch still reaches the reading gate (guard did not fire).
    keyed = runtime.host.check_repair_preconditions(
        branch="main", reading_id_arg="", turn=1)
    assert keyed["blocked"]["reason"] == "fresh_reading_required"


def test_decoded_branch_guard_inherited_key_and_mask_precedence(tmp_path):
    """Review finding #3: an inherited snapshot key does NOT exempt a decoded
    install — the panel binds metadata['decoded_text'], so repair against the
    stale key would edit a render verification never sees. Conversely a
    mask+key render (null-mask branch) wins the precedence and stays
    repairable, and a partial key WITHOUT decoded_text proceeds as before."""
    from tests.support.scripted_v3 import keyed_catton_state
    from mcp_server.runtime import InvestigationRuntime

    _ct, state = keyed_catton_state()
    doc = {"schema_version": 1, "meta": {"investigation_id": "x", "revision": 1},
           "state": state.to_artifact_dict(), "records": {}}
    runtime = InvestigationRuntime(document=doc, emit=lambda *a, **k: None,
                                   verify_provider=None, max_cost_usd=5.0,
                                   synchronous_experiments=True)
    ws = runtime.workspace
    # (a) decoded install that inherited the keyed source branch's key: FIRES.
    ws.fork("decoded_keyed", "main")
    b = ws.get_branch("decoded_keyed")
    assert b.key, "fixture requires an inherited non-empty key"
    b.metadata["decoded_text"] = "SOMEDECODEDTEXT"
    b.metadata["cipher_mode"] = "composite_substitution_transposition"
    pre = runtime.host.check_repair_preconditions(
        branch="decoded_keyed", reading_id_arg="", turn=1)
    assert pre["blocked"]["reason"] == "decoded_branch_no_base_key"
    # (b) null-mask branch: mask+key render wins over decoded_text; repairable.
    ws.fork("masked", "main")
    m = ws.get_branch("masked")
    m.metadata["decoded_text"] = "SHOULDNOTBIND"
    m.metadata["null_mask_finalist"] = {"mask": ["Z"]}
    pre2 = runtime.host.check_repair_preconditions(
        branch="masked", reading_id_arg="", turn=1)
    assert pre2["blocked"]["reason"] == "fresh_reading_required"
    # (c) partial key, no decoded_text: guard silent (spec: partial keys untouched).
    ws.fork("partial", "main")
    first_pair = next(iter(ws.get_branch("main").key.items()))
    ws.set_full_key("partial", dict([first_pair]))
    pre3 = runtime.host.check_repair_preconditions(
        branch="partial", reading_id_arg="", turn=1)
    assert pre3["blocked"]["reason"] == "fresh_reading_required"


# --------------------------------------------- verifier-arbitrated repair (MCP)
# Spec: docs/specs/verifier_arbitrated_repair_spec.md §5.7-5.8, T10-T11.
# Evidence: docs/evidence/c56de7e6c600_repair_guard_false_reject.json (case 1).
#
# Fixture: a lowercase-symbol substitution keyed correct EXCEPT the `w` symbol,
# which maps to `I`. Word 2 of "THE MISSING TRAWLER RESTED IN THE COVE" then
# decodes TRAILER (in the common list); the true word TRAWLER is NOT — exactly
# the c56de7e6c600 straddle where the mechanical occurrence counter false-rejects
# an objectively-correct correction. `w` occurs only in that word.
_TRAWLER_PLAINTEXT = "THE MISSING TRAWLER RESTED IN THE COVE"


def _mcp_verdict(tlc, rec, *, accepts_as_solution=False):
    return {
        "coherence": 7, "reader_accepts": False,
        "reader_accepts_as_solution": accepts_as_solution,
        "target_language_confidence": tlc, "semantic_recoverability": rec,
        "damage_scope": "local", "repairability": "local_repair",
        "uncertainty_note": "", "gloss": "reads as clear English",
        "anomalies": [], "confidence": "high",
    }


def _apply_wrong_symbol(runtime, plaintext_upper, wrong_symbol, wrong_target):
    """White-box: key each cipher symbol to its correct uppercase EXCEPT
    ``wrong_symbol`` -> ``wrong_target``."""
    ws = runtime.workspace
    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet
    for sym in alpha.symbols:
        target = wrong_target if sym == wrong_symbol else sym.upper()
        ws.set_mapping("main", alpha.id_for(sym), pt.id_for(target))


def _trawler_ready(server, plaintext=_TRAWLER_PLAINTEXT, wrong_symbol="w",
                   wrong_target="I", word="TRAWLER", word_index=2):
    """Start an investigation on the damaged basin, record the intended reading,
    compile the correcting hypothesis, and return (iid, rev, rt, comp)."""
    b = start(server, ciphertext=plaintext.lower())
    iid, rev = b["investigation_id"], b["revision"]
    rt = server._runtimes[iid]
    _apply_wrong_symbol(rt, plaintext, wrong_symbol, wrong_target)
    rec = call(server, "reading_record", investigation_id=iid, expected_revision=rev,
               branch="main", reading_text=plaintext, overall_confidence=0.8)
    rev = rec["revision"]
    comp = call(server, "repair_hypotheses_test", investigation_id=iid,
                expected_revision=rev, branch="main",
                hypotheses=[{"word": word, "word_index": word_index}])
    assert comp["status"] == "ok" and comp["changed_finalists"], comp
    return iid, comp["revision"], rt, comp


def _scoring_reject_fired(tx):
    """A collateral/scalar scoring check actually rejected (self-validation)."""
    return any(
        c["check"] in {"collateral_within_limits", "scalar_non_decrease"}
        and not c["passed"]
        for c in tx["acceptance"]["checks"]
    )


def test_t10_mcp_arbitration_installs_case1_replica(tmp_path):
    sessions_mod.register_session_builder(
        "episode:verify", make_verify_builder(_mcp_verdict(0.97, 0.88)))
    try:
        server = make_server(tmp_path, verify="dummy")
        iid, rev, rt, comp = _trawler_ready(server)
        winner = comp["changed_finalists"][0]["branch"]
        tx = call(server, "repair_transaction", investigation_id=iid,
                  expected_revision=rev, branch="main", compile_id=comp["compile_id"],
                  winner=winner, verifier_arbitration=True)
        # Self-validating: the mechanical trigger actually fired.
        assert _scoring_reject_fired(tx), tx["acceptance"]["checks"]
        assert tx["status"] == "installed"
        arb = tx["acceptance"]["arbitration"]
        assert arb["status"] == "accepted"
        assert tx["acceptance"]["policy"] == "default_deny_v1+verifier_arbitration_v1"
        # The installed branch's attestation history contains the arbitration att.
        show = call(server, "candidate_show", investigation_id=iid,
                    branch=tx["installed_branch"])
        assert any(
            a.get("target_language_confidence") == 0.97
            and a.get("episode_id") == arb["episode_id"]
            for a in show["attestation_history"]
        ), show["attestation_history"]
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)


def test_t10a_mcp_arbitration_keyless_unavailable(tmp_path):
    # verify="none" + no registered episode:verify builder -> keyless.
    server = make_server(tmp_path, verify="none")
    iid, rev, rt, comp = _trawler_ready(server)
    winner = comp["changed_finalists"][0]["branch"]
    tx = call(server, "repair_transaction", investigation_id=iid,
              expected_revision=rev, branch="main", compile_id=comp["compile_id"],
              winner=winner, verifier_arbitration=True)
    assert _scoring_reject_fired(tx), tx["acceptance"]["checks"]
    assert tx["status"] == "failed"
    assert tx["reason"] == "materially_non_improving"
    arb = tx["acceptance"]["arbitration"]
    assert arb["status"] == "unavailable"
    assert arb["reason"] == "no_verification_provider"


def test_t10b_mcp_flag_absent_todays_shape(tmp_path):
    server = make_server(tmp_path, verify="dummy")
    iid, rev, rt, comp = _trawler_ready(server)
    winner = comp["changed_finalists"][0]["branch"]
    tx = call(server, "repair_transaction", investigation_id=iid,
              expected_revision=rev, branch="main", compile_id=comp["compile_id"],
              winner=winner)  # flag absent
    assert tx["status"] == "failed"
    assert tx["reason"] == "materially_non_improving"
    assert "arbitration" not in tx["acceptance"]  # today's shape
    assert tx["acceptance"]["policy"] == "default_deny_v1"


def test_t10c_mcp_arbitration_rejecting_verdict_fails(tmp_path):
    sessions_mod.register_session_builder(
        "episode:verify", make_verify_builder(_mcp_verdict(0.4, 0.3)))
    try:
        server = make_server(tmp_path, verify="dummy")
        iid, rev, rt, comp = _trawler_ready(server)
        winner = comp["changed_finalists"][0]["branch"]
        tx = call(server, "repair_transaction", investigation_id=iid,
                  expected_revision=rev, branch="main", compile_id=comp["compile_id"],
                  winner=winner, verifier_arbitration=True)
        assert _scoring_reject_fired(tx), tx["acceptance"]["checks"]
        assert tx["status"] == "failed"
        assert tx["acceptance"]["arbitration"]["status"] == "rejected"
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)

# T11 lives at host level in tests/test_repair_arbitration.py (the acceptance
# pipeline operates on whole-fork content; the collateral counter is n-gram-
# local, so a deterministic host-level synthesis of "damaged 3 / improved 0" is
# the robust way to prove a scattered-fix fork installs through arbitration).
