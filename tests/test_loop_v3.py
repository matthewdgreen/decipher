"""run_v3 lead-loop tests (M1 Part 4/5) with scripted fake sessions."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agent.model_provider import ModelProviderError, ModelResponse, ModelUsage, TextBlock, ToolUseBlock
from investigation import sessions as sessions_mod
from investigation.loop_v3 import (
    _fresh_compare_winner,
    _unbound_edit_claims,
    run_v3,
)
from investigation.sessions import SessionCapabilities
from investigation.state import BudgetEntry, InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace

# M5.3 Slice 7 Part F: the scripted fakes + seed helpers are hoisted into the
# shared support module. Thin aliases below keep this file's existing
# underscore-prefixed call sites unchanged.
from tests.support.scripted_v3 import (
    ErrorSession,
    ScriptedSession,
    VerifyWorkerFake,
    keyed_catton_state,
    register_programmable_repair,
    seed_negative_attestation,
    seed_reading,
)

_keyed_catton_state = keyed_catton_state
_seed_reading = seed_reading
_seed_negative_attestation = seed_negative_attestation
_register_programmable_repair = register_programmable_repair


def _caesar(text: str, shift: int) -> str:
    return "".join(
        chr((ord(c) - 65 + shift) % 26 + 65) if c.isalpha() else c for c in text
    )


def _caesar_cipher(plaintext: str, shift: int = 3):
    raw = _caesar(plaintext, shift)
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    return CipherText(raw=raw, alphabet=alpha, separator=" "), alpha


@pytest.fixture
def verify_fake():
    sessions_mod.register_session_builder("episode:verify", VerifyWorkerFake)
    yield
    sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)


def _apply_caesar_key(workspace, alpha):
    for index in range(alpha.size):
        cipher_symbol = alpha.symbol_for(index)
        workspace.set_mapping(
            "main",
            workspace.cipher_text.alphabet.id_for(cipher_symbol),
            workspace.plaintext_alphabet.id_for(_caesar(cipher_symbol, -3)),
        )


def _seeded_caesar_state(ct, alpha, *, with_alt=False):
    state = InvestigationState(workspace=Workspace(ct), language="en")
    state.add_evidence("diagnostic_preflight", turn=0, summary="seeded test state")
    _apply_caesar_key(state.workspace, alpha)
    if with_alt:
        state.workspace.fork("alt", from_branch="main")
    return state


def _solve_scripts(alpha):
    return [
        [ToolUseBlock(id="t1", name="decode_show", input={"branch": "main"})],
        # M5: a verify episode on `main` before declaring (the AttestationPolicy
        # requires a fresh attestation).
        [ToolUseBlock(id="tv", name="episode_run",
                      input={"kind": "verify", "goal": "verify main",
                             "branches": ["main"]})],
        [TextBlock(text="Reads as English."),
         ToolUseBlock(id="t3", name="meta_declare_solution",
                      input={"branch": "main",
                             "rationale": "Caesar shift 3 recovered; reads as English.",
                             "self_confidence": 0.95})],
    ]


def test_run_v3_scripted_solve_verify_then_declare(verify_fake):
    ct, alpha = _caesar_cipher("THE DOG")
    scripts = _solve_scripts(alpha)
    # Fork an extra branch on turn 1 so a v2 branch_cards gate WOULD block the
    # declaration; M5's AttestationPolicy keeps only the one attestation check —
    # with a fresh attestation from the verify episode the declaration is
    # accepted immediately (no v2 cascade bounce).
    session = ScriptedSession(scripts)
    art = run_v3(ct, session=session, language="en", max_iterations=10,
                 cipher_id="v3_caesar",
                 resume_state=_seeded_caesar_state(ct, alpha, with_alt=True))

    assert art.status == "solved"
    assert art.loop_version == "v3"
    assert art.solution is not None and art.solution.branch == "main"
    assert abs(art.solution.self_confidence - 0.95) < 1e-9
    # The declaration was accepted on the FIRST call (no gate bounce).
    declare_results = [
        json.loads(tc.result) for tc in art.tool_calls
        if tc.tool_name == "meta_declare_solution"
    ]
    assert len(declare_results) == 1
    assert declare_results[0]["accepted"] is True
    # The verify attestation gated + is carried into the declaration + artifact.
    assert art.attestations and art.attestations[0]["reader_accepts"] is True
    assert art.solution.attestation is not None
    assert art.solution.attestation["coherence"] == 9
    # Decode is correct.
    main_decode = next(b.decryption for b in art.branches if b.name == "main")
    assert main_decode == "THE DOG"


def test_run_v3_tool_iteration_is_lead_turn(verify_fake):
    ct, alpha = _caesar_cipher("THE DOG")
    session = ScriptedSession(_solve_scripts(alpha))
    art = run_v3(ct, session=session, language="en", max_iterations=10,
                 cipher_id="v3_iter", resume_state=_seeded_caesar_state(ct, alpha))
    by_name = {tc.tool_name: tc.iteration for tc in art.tool_calls}
    assert by_name["decode_show"] == 1
    # The verify episode_run runs at lead turn 2 (episode_run is a lead-dispatch
    # tool, not an executor ToolCall, so it does not appear in by_name); the
    # declaration lands at turn 3.
    assert by_name["meta_declare_solution"] == 3


def test_run_v3_records_state_budget_and_transcript(verify_fake):
    ct, alpha = _caesar_cipher("THE DOG")
    session = ScriptedSession(_solve_scripts(alpha))
    art = run_v3(ct, session=session, language="en", max_iterations=10,
                 cipher_id="v3_state", resume_state=_seeded_caesar_state(ct, alpha))
    # Budget accounting flows from the session's per-send entries. Four lead
    # sends (decode_show, verify episode_run, declare) plus one
    # episode:verify send (50/10/0).
    assert art.budget_by_category["lead"]["calls"] == 3
    assert art.budget_by_category["episode:verify"]["calls"] == 1
    assert art.total_input_tokens == 3 * 1000 + 50
    assert art.total_output_tokens == 3 * 50 + 10
    assert art.total_cache_read_tokens == 3 * 100 + 0
    # Artifact carries the v3 additions.
    assert art.session_transcript["provider"] == "openai"
    assert art.investigation_state is not None
    kinds = [e["kind"] for e in art.investigation_state["evidence_log"]]
    assert "diagnostic_preflight" in kinds
    assert "turn_summary" in kinds
    # Logical transcript stores only new per-turn content (no rebuilt contexts).
    assert all(m["role"] in {"assistant", "user"} for m in art.messages)


def test_run_v3_provider_error_preserves_best_effort_without_declaration():
    ct, _alpha = _caesar_cipher("THE DOG")
    art = run_v3(ct, session=ErrorSession([]), language="en", max_iterations=5,
                 cipher_id="v3_error")
    assert art.status == "error"
    assert art.auto_declared is False
    assert art.solution is None
    assert art.fallback_selection is not None
    assert "API error" in art.error_message


def test_run_v3_exhaustion_is_honestly_unsolved():
    ct, _alpha = _caesar_cipher("THE DOG")
    # Text-only responses → no tool calls → exhausted → fallback.
    session = ScriptedSession([[TextBlock(text="I have no idea.")]])
    art = run_v3(ct, session=session, language="en", max_iterations=3,
                 cipher_id="v3_exhaust")
    assert art.status == "unsolved"
    assert art.auto_declared is False
    assert art.solution is None
    assert art.fallback_selection is not None


def test_v3_rejects_hidden_operator_tool_at_dispatch():
    ct, alpha = _caesar_cipher("THE DOG")
    session = ScriptedSession([
        [ToolUseBlock(id="hidden", name="act_set_mapping", input={
            "branch": "main", "cipher_symbol": alpha.symbol_for(0),
            "plain_letter": "T",
        })],
        [TextBlock(text="done")],
    ])
    art = run_v3(
        ct,
        session=session,
        language="en",
        max_iterations=2,
        cipher_id="v3_hidden_tool",
    )
    call = next(tc for tc in art.tool_calls if tc.tool_name == "act_set_mapping")
    assert json.loads(call.result)["reason"] == "lead_tool_not_available"
    main = next(
        branch for branch in art.investigation_state["workspace"]["branches"]
        if branch["name"] == "main"
    )
    assert not main["key"]
    assert any(event.event == "lead_tool_rejected" for event in art.loop_events)


def test_v3_suppresses_duplicate_decode_on_unchanged_content():
    ct, _alpha = _caesar_cipher("THE DOG")
    session = ScriptedSession([
        [ToolUseBlock(id="read1", name="decode_show", input={"branch": "main"})],
        [ToolUseBlock(id="read2", name="decode_show", input={"branch": "main"})],
        [TextBlock(text="done")],
    ])
    art = run_v3(
        ct,
        session=session,
        language="en",
        max_iterations=3,
        cipher_id="v3_duplicate_read",
    )
    calls = [tc for tc in art.tool_calls if tc.tool_name == "decode_show"]
    assert len(calls) == 2
    assert json.loads(calls[1].result)["status"] == "duplicate_suppressed"
    assert any(event.event == "duplicate_read_suppressed" for event in art.loop_events)
    assert any(event.event == "repeated_call" for event in art.loop_events)
    assert max(art.investigation_state["call_signature_counts"].values()) == 2
    assert art.investigation_state["no_new_information_streak"] >= 1


def test_repair_required_state_narrows_episode_schema_and_dispatch():
    from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel

    ct, alpha = _caesar_cipher("THE DOG")
    state = _seeded_caesar_state(ct, alpha)
    decoded = _decoded_text_for_panel(state.workspace, "main")
    state.verify_attestations.append({
        "branch": "main",
        "content_hash": _candidate_content_hash(decoded),
        "coherence": 4,
        "reader_accepts": False,
        "reader_accepts_as_solution": False,
        "target_language_confidence": 0.8,
        "semantic_recoverability": 0.7,
        "damage_scope": "local",
        "repairability": "local_repair",
        "gloss": "partly readable",
        "anomalies": ["broken clause"],
        "created_turn": 0,
    })
    session = ScriptedSession([
        [ToolUseBlock(id="search", name="episode_run", input={
            "kind": "search",
            "goal": "wander away",
            "branches": ["main"],
            "search_tool": "search_anneal",
        })],
        [TextBlock(text="done")],
    ])
    art = run_v3(
        ct,
        session=session,
        language="en",
        max_iterations=2,
        cipher_id="v3_repair_transition",
        resume_state=state,
    )
    episode_def = next(
        definition for definition in session.tools_seen[0]
        if definition["name"] == "episode_run"
    )
    kinds = episode_def["input_schema"]["properties"]["kind"]["enum"]
    assert "repair" in kinds and "reading" in kinds
    assert "search" not in kinds and "survey" not in kinds
    result_blocks = [
        block
        for message in art.messages
        for block in (message.get("content") or [])
        if isinstance(block, dict) and block.get("tool_use_id") == "search"
    ]
    assert result_blocks
    assert json.loads(result_blocks[0]["content"])["reason"] == (
        "episode_kind_not_available"
    )


def test_run_v3_positive_attestation_drives_attested_fallback(verify_fake):
    ct, alpha = _caesar_cipher("THE DOG")
    scripts = [
        [ToolUseBlock(id="verify", name="episode_run",
                      input={"kind": "verify", "goal": "verify main",
                             "branches": ["main"]})],
    ]
    art = run_v3(ct, session=ScriptedSession(scripts), language="en",
                 max_iterations=1, cipher_id="v3_attested_fallback",
                 resume_state=_seeded_caesar_state(ct, alpha))
    assert art.status == "fallback_declared"
    assert art.auto_declared is True
    assert art.attested_fallback is True
    assert art.fallback_selection["tier"] == "fresh_positive_attestation"
    assert art.solution.attestation["reader_accepts"] is True
    assert art.fallback_selection["attestation"]["reader_accepts_as_solution"] is True
    # Slice 6: fallback confidence = mean(target_language_confidence,
    # semantic_recoverability) = (0.9 + 0.8) / 2.
    assert abs(art.solution.self_confidence - 0.85) < 1e-9


def test_fresh_compare_winner_rejects_stale_hash_binding():
    ct, state = _keyed_catton_state()
    ws = state.workspace
    ws.fork("alternate", from_branch="main")
    from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel

    hashes = {
        name: _candidate_content_hash(_decoded_text_for_panel(ws, name))
        for name in ("main", "alternate")
    }
    state.episode_ledger.append({
        "episode_id": "cmp1",
        "kind": "compare",
        "status": "ok",
        "comparison_binding": {
            "branch_hashes": hashes,
            "winner": "alternate",
            "winner_hash": hashes["alternate"],
        },
    })
    assert _fresh_compare_winner(state)[0] == "alternate"

    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet
    ws.set_mapping("alternate", alpha.id_for("a"), pt.id_for("Z"))
    assert _fresh_compare_winner(state) is None


def test_run_v3_declare_without_verify_is_blocked_then_unsolved():
    """M5: meta_declare_solution with no verify attestation is blocked; the run
    exhausts honestly unsolved while retaining a best-effort branch."""
    ct, alpha = _caesar_cipher("THE DOG")
    scripts = [
        [ToolUseBlock(id="d1", name="meta_declare_solution",
                      input={"branch": "main", "rationale": "reads",
                             "self_confidence": 0.9})],
    ]
    art = run_v3(ct, session=ScriptedSession(scripts), language="en",
                 max_iterations=4, cipher_id="v3_no_verify",
                 resume_state=_seeded_caesar_state(ct, alpha))
    declare_results = [json.loads(tc.result) for tc in art.tool_calls
                       if tc.tool_name == "meta_declare_solution"]
    assert declare_results
    assert all(r.get("reason") == "attestation_required" for r in declare_results)
    assert art.status == "unsolved"
    assert art.auto_declared is False
    assert not art.attestations
    assert art.solution is None
    assert art.fallback_selection is not None


class _WeakVerifyFake(VerifyWorkerFake):
    """A verify worker that reports words-but-not-sentences (weak, not accepted)."""

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(
            BudgetEntry("episode:verify", "openai", "fake-luna", 50, 10, 0))
        result = {"coherence": 3, "reader_accepts": False,
                  "reader_accepts_as_solution": False,
                  "target_language_confidence": 0.8,
                  "semantic_recoverability": 0.7,
                  "damage_scope": "local", "repairability": "local_repair",
                  "uncertainty_note": "middle clause",
                  "gloss": "reads as words but not coherent sentences",
                  "anomalies": ["non-word run", "broken clause"], "confidence": "low"}
        return ModelResponse(
            content=[ToolUseBlock(id="v1", name="episode_submit_result",
                                  input={"result": result, "summary": "weak"})],
            usage=ModelUsage(50, 10, 0))


def test_run_v3_weak_attestation_blocks_declare_and_seeds_agenda():
    """M5.3 Slice 6 (C6 reversed): a WEAK attestation BLOCKS declaration; the
    run exhausts honestly unsolved, and the local_repair verdict seeds the
    repair agenda with the reported anomalies."""
    sessions_mod.register_session_builder("episode:verify", _WeakVerifyFake)
    try:
        ct, alpha = _caesar_cipher("THE DOG")
        art = run_v3(ct, session=ScriptedSession(_solve_scripts(alpha)),
                     language="en", max_iterations=10, cipher_id="v3_weak",
                     resume_state=_seeded_caesar_state(ct, alpha))
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)
    declare_results = [json.loads(tc.result) for tc in art.tool_calls
                       if tc.tool_name == "meta_declare_solution"]
    assert declare_results
    assert all(
        r.get("reason") == "attestation_not_positive" for r in declare_results
    )
    assert art.status == "unsolved"
    assert art.solution is None
    assert art.attestations[0]["reader_accepts_as_solution"] is False
    verify_items = [
        item for item in art.repair_agenda
        if item.get("source") == "verify_attestation"
    ]
    assert {item["anomaly"] for item in verify_items} == {
        "non-word run", "broken clause",
    }
    assert all(
        item["repairability"] == "local_repair" for item in verify_items
    )


class _OutOfScaleVerifyFake(VerifyWorkerFake):
    """Review F-1: a reader answering coherence=12 — a scale violation (likely a
    0-100-scale reading, i.e. LOW) — while REJECTING the text."""

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(
            BudgetEntry("episode:verify", "openai", "fake-luna", 50, 10, 0))
        result = {"coherence": 12, "reader_accepts": False,
                  "reader_accepts_as_solution": False,
                  "target_language_confidence": 0.2,
                  "semantic_recoverability": 0.1,
                  "damage_scope": "basin_wide", "repairability": "none",
                  "uncertainty_note": "",
                  "gloss": "scattered words only",
                  "anomalies": ["non-words throughout"], "confidence": "high"}
        return ModelResponse(
            content=[ToolUseBlock(id="v1", name="episode_submit_result",
                                  input={"result": result, "summary": "rejected"})],
            usage=ModelUsage(50, 10, 0))


def test_run_v3_out_of_scale_coherence_records_floor_not_maximum():
    """Review F-1: coherence > 10 violates the stated 0-10 scale and must NOT be
    recorded as 10/10 (a 12 on a 0-100 scale is a LOW reading; recording 10
    would mint a top-coherence attestation on a decode the reader rejected).
    It is recorded as the conservative floor 0."""
    sessions_mod.register_session_builder("episode:verify", _OutOfScaleVerifyFake)
    try:
        ct, alpha = _caesar_cipher("THE DOG")
        art = run_v3(ct, session=ScriptedSession(_solve_scripts(alpha)),
                     language="en", max_iterations=10, cipher_id="v3_scale",
                     resume_state=_seeded_caesar_state(ct, alpha))
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)
    assert art.attestations, "verify episode should have produced an attestation"
    att = art.attestations[0]
    assert att["coherence"] != 10, "scale violation must not be recorded as maximum"
    assert att["coherence"] == 0
    assert att["reader_accepts"] is False
    # Slice 6 (C6 reversed): the hash-matched but non-positive weak attestation
    # no longer lets the declaration through — the run exhausts unsolved.
    assert art.status == "unsolved"
    assert art.solution is None


def test_run_v3_post_declare_tools_in_same_batch_do_not_run():
    """Review F-2: [meta_declare_solution, act_set_mapping] in ONE assistant
    turn — the mutation after the accepted declaration must NOT execute (it
    would void attested == declared == scored), the skipped tool_use still gets
    a paired `run_terminated` tool_result, and the declaration keeps its
    attestation."""
    ct, alpha = _caesar_cipher("THE DOG")
    scripts = [
        [ToolUseBlock(id="tv", name="episode_run",
                      input={"kind": "verify", "goal": "verify main",
                             "branches": ["main"]})],
        # One batch: declare THEN mutate the just-declared branch.
        [ToolUseBlock(id="td", name="meta_declare_solution",
                      input={"branch": "main", "rationale": "reads",
                             "self_confidence": 0.9}),
         ToolUseBlock(id="tm", name="act_set_mapping",
                      input={"branch": "main",
                             "cipher_symbol": alpha.symbol_for(0),
                             "plain_letter": "Z"})],
    ]
    sessions_mod.register_session_builder("episode:verify", VerifyWorkerFake)
    try:
        art = run_v3(ct, session=ScriptedSession(scripts), language="en",
                     max_iterations=10, cipher_id="v3_post_declare",
                     resume_state=_seeded_caesar_state(ct, alpha))
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)

    assert art.status == "solved"
    # The mutating tool never executed: no ToolCall logged for it, and the
    # declared branch's decode is unchanged.
    assert all(tc.tool_name != "act_set_mapping" for tc in art.tool_calls)
    main_decode = next(b.decryption for b in art.branches if b.name == "main")
    assert main_decode == "THE DOG"
    # attested == declared == scored held: the attestation attached.
    assert art.solution is not None and art.solution.attestation is not None
    # The skipped tool_use got a paired synthesized run_terminated result.
    results = {}
    for m in art.messages:
        for b in (m.get("content") or []):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                results[b["tool_use_id"]] = b["content"]
    assert "td" in results and "tm" in results  # one result per tool_use
    assert json.loads(results["tm"])["status"] == "run_terminated"


def test_run_v3_resume_from_state_continues(verify_fake):
    ct, alpha = _caesar_cipher("THE DOG")
    session = ScriptedSession(_solve_scripts(alpha))
    art = run_v3(ct, session=session, language="en", max_iterations=10,
                 cipher_id="v3_resume",
                 resume_state=_seeded_caesar_state(ct, alpha))
    # Reload state from the artifact and confirm it is a valid resume seed.
    from investigation.state import InvestigationState
    reloaded = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(art.investigation_state))
    )
    assert reloaded.workspace.get_branch("main").key  # key survived
    assert reloaded.language == "en"


def test_run_v3_resume_continues_from_state_turn_monotonic(verify_fake):
    """R1: resume continues from state.turn + 1 with monotonic turn numbers."""
    from investigation.state import InvestigationState
    ct, alpha = _caesar_cipher("THE DOG")
    # Leg 1: two non-declaring turns, exhausts at max_iterations=2.
    noop = [[ToolUseBlock(id="n1", name="decode_show", input={"branch": "main"})],
            [ToolUseBlock(id="n2", name="decode_show", input={"branch": "main"})]]
    first = run_v3(ct, session=ScriptedSession(noop), language="en",
                   max_iterations=2, cipher_id="v3_resume_leg1")
    assert first.investigation_state["turn"] == 2

    reloaded = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(first.investigation_state)))
    _apply_caesar_key(reloaded.workspace, alpha)
    # Leg 2: resume and solve. Turns continue strictly past the pre-resume turns.
    seen = []
    second = run_v3(
        ct, session=ScriptedSession(_solve_scripts(alpha)), language="en",
        max_iterations=8, cipher_id="v3_resume_leg2", resume_state=reloaded,
        on_event=lambda e, p: seen.append(p["iteration"])
        if e == "iteration_start" else None,
    )
    assert seen == sorted(seen)      # monotonic
    assert seen and seen[0] == 3     # resumed at state.turn + 1
    assert second.status == "solved"
    # The turn-0 diagnostic preflight is not re-added on resume.
    kinds = [e["kind"] for e in second.investigation_state["evidence_log"]]
    assert kinds.count("diagnostic_preflight") == 1


def test_run_v3_max_iterations_zero_no_nameerror():
    """R8(c): max_iterations=0 must not raise a NameError at the run_complete
    emit (the turn variable is pre-initialized)."""
    ct, _alpha = _caesar_cipher("THE DOG")
    art = run_v3(ct, session=ScriptedSession([[TextBlock(text="hi")]]),
                 language="en", max_iterations=0, cipher_id="v3_zero")
    # It finishes cleanly (exhausted → fallback), no exception.
    assert art.status == "unsolved"


def test_run_v3_resume_uses_state_language_not_param():
    """R8(a): resuming a `de` state must load German resources, not English
    (the language param is ignored in favor of state.language on resume)."""
    from investigation.state import InvestigationState
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText
    from workspace import Workspace

    raw = "ABCDEF"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    resume = InvestigationState(workspace=Workspace(ct), language="de")
    resume.turn = 1

    seen_langs = []

    class LangSession(ScriptedSession):
        pass

    art = run_v3(ct, session=ScriptedSession([[TextBlock(text="done")]]),
                 language="en",  # caller passes the WRONG language
                 max_iterations=3, cipher_id="v3_lang", resume_state=resume)
    # The run adopts the state's language, not the param.
    assert art.language == "de"
    assert art.investigation_state["language"] == "de"


# ---------------------------------------------------------------------------
# M3: reading -> repair -> install -> branch_adjudicate -> declare (end-to-end)
# ---------------------------------------------------------------------------
import re

from investigation import sessions as sessions_mod
from workspace import Workspace


BOUNDARY_ACTUATORS = {
    "act_split_cipher_word", "act_merge_cipher_words", "act_merge_decoded_words",
    "act_apply_boundary_candidate", "act_resegment_by_reading",
    "act_resegment_from_reading_repair", "act_resegment_window_by_reading",
}


def _episode_id_from_blocks(blocks, kind):
    for m in blocks:
        for b in (m.get("content") or []):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                try:
                    data = json.loads(b.get("content") or "")
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(data, dict) and data.get("kind") == kind:
                    return data
    return None


class ReadingWorkerFake:
    def __init__(self, provider, system, role):
        self.model = "fake-reader"; self.provider_name = "openai"
        self.capabilities = SessionCapabilities(); self._budget = []; self._n = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("episode:reading", "openai", "fake-reader", 10, 5, 0))
        step = self._n; self._n += 1
        if step == 0:
            content = [ToolUseBlock(id="rd0", name="decode_show", input={"branch": "main"})]
        else:
            content = [ToolUseBlock(id="rd1", name="episode_submit_result", input={
                "result": {"reading_text": "COTON",
                           "fragments": [{"text": "COTON", "repair_text": "COTON",
                                          "confidence": 0.9}],
                           "holes": [], "overall_confidence": 0.8},
                "summary": "read as COTON",
            })]
        return ModelResponse(content=content, usage=ModelUsage(10, 5, 0))

    def usage_entries(self): return list(self._budget)
    def export_transcript(self): return {"provider": "openai", "exchanges": []}


class RepairWorkerFake:
    """Reads the injected reading id from its context and applies it."""

    def __init__(self, provider, system, role):
        self.model = "fake-repair"; self.provider_name = "openai"
        self.capabilities = SessionCapabilities(); self._budget = []; self._n = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("episode:repair", "openai", "fake-repair", 10, 5, 0))
        step = self._n; self._n += 1
        if step == 0:
            text = json.dumps(blocks, default=str)
            m = re.search(r"id `([0-9a-f]{12})`", text)
            rid = m.group(1) if m else None
            content = [ToolUseBlock(id="rp0", name="hypothesis_apply_reading",
                                    input={"branch": "main", "reading_id": rid})]
        else:
            content = [ToolUseBlock(id="rp1", name="episode_submit_result", input={
                "result": {"applied": True, "best_branch": None, "edits": [],
                           "verdicts": [{"action": "apply_reading", "target": "main",
                                         "verdict": "kept"}],
                           "collateral": {}, "notes": "applied reading fork"},
                "summary": "applied the reading",
            })]
        return ModelResponse(content=content, usage=ModelUsage(10, 5, 0))

    def usage_entries(self): return list(self._budget)
    def export_transcript(self): return {"provider": "openai", "exchanges": []}


class _RepairLead:
    def __init__(self):
        self.model = "fake-lead"; self.provider_name = "openai"
        self.capabilities = SessionCapabilities(); self._budget = []; self._step = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("lead", "openai", "fake-lead", 100, 10, 0))
        step = self._step; self._step += 1
        if step == 0:
            content = [ToolUseBlock(id="e1", name="episode_run", input={
                "kind": "reading", "goal": "read main", "branches": ["main"]})]
        elif step == 1:
            reading = _episode_id_from_blocks(blocks, "reading")
            content = [ToolUseBlock(id="e2", name="episode_run", input={
                "kind": "repair", "goal": "apply reading", "branches": ["main"],
                "reading_id": reading["reading_id"]})]
        elif step == 2:
            repair = _episode_id_from_blocks(blocks, "repair")
            fork = next(s for s in repair["snapshots"] if s.startswith("reading_"))
            content = [ToolUseBlock(id="e3", name="episode_install_branch", input={
                "episode_id": repair["episode_id"], "branch": fork,
                "as_name": "repaired"})]
        elif step == 3:
            content = [ToolUseBlock(id="e4", name="branch_adjudicate", input={
                "branches": ["main", "repaired"]})]
        elif step == 4:
            # M5: verify the branch before declaring it.
            content = [ToolUseBlock(id="ev", name="episode_run", input={
                "kind": "verify", "goal": "verify repaired",
                "branches": ["repaired"]})]
        else:
            content = [ToolUseBlock(id="e5", name="meta_declare_solution", input={
                "branch": "repaired", "rationale": "repaired reads well",
                "self_confidence": 0.8})]
        return ModelResponse(content=content, usage=ModelUsage(100, 10, 0))

    def usage_entries(self): return list(self._budget)
    def export_transcript(self): return {"provider": "openai", "exchanges": []}


def test_run_v3_reading_repair_install_adjudicate_declare_end_to_end(verify_fake):
    sessions_mod.register_session_builder("episode:reading", ReadingWorkerFake)
    sessions_mod.register_session_builder("episode:repair", RepairWorkerFake)
    try:
        ct, state = _keyed_catton_state()
        art = run_v3(ct, session=_RepairLead(), language="en", max_iterations=8,
                     cipher_id="v3_repair", resume_state=state)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:reading", None)
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)

    # Reading was compiled + stored by the lead (A1: workers never write it).
    assert art.readings and art.readings[0]["fragments"][0]["text"] == "COTON"
    reading_id = art.readings[0]["reading_id"]

    # Three episodes: reading (ok), repair (ok), then verify (ok) before declare.
    assert [e["kind"] for e in art.episodes] == ["reading", "repair", "verify"]
    assert all(e["status"] == "ok" for e in art.episodes)
    # The declaration carries the verify attestation for `repaired`.
    assert art.solution is not None and art.solution.branch == "repaired"
    assert art.solution.attestation is not None
    assert art.attestations and art.attestations[0]["branch"] == "repaired"

    # The repaired fork was installed into the lead workspace as `repaired`.
    assert any(b.name == "repaired" for b in art.branches)

    # The composite ran inside the repair episode (episode_id-stamped) AND on the
    # lead (branch_adjudicate, iteration = lead turn 4).
    apply_calls = [tc for tc in art.tool_calls if tc.tool_name == "hypothesis_apply_reading"]
    assert apply_calls and apply_calls[0].episode_id is not None
    adjudicate = next(tc for tc in art.tool_calls if tc.tool_name == "branch_adjudicate")
    assert adjudicate.episode_id is None and adjudicate.iteration == 4

    # A fully aligned repair has no residual agenda item.
    residual = [i for i in art.repair_agenda if i.get("kind") == "reading_residual"]
    assert residual == []

    # No v2 boundary actuator was ever called across the whole v3 run.
    called = {tc.tool_name for tc in art.tool_calls}
    assert BOUNDARY_ACTUATORS.isdisjoint(called)
    assert art.status == "solved"


class TransactionReadingWorkerFake:
    def __init__(self, provider, system, role):
        self.model = "fake-reader"; self.provider_name = "openai"
        self.capabilities = SessionCapabilities(); self._budget = []

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("episode:reading", "openai", self.model, 10, 5, 0))
        result = {
            "reading_text": "LATER",
            "fragments": [{"text": "LATER", "repair_text": "LATER", "confidence": 0.95}],
            "holes": [],
            "overall_confidence": 0.95,
        }
        return ModelResponse(
            content=[ToolUseBlock(id="trd", name="episode_submit_result",
                                  input={"result": result, "summary": "one supported repair"})],
            usage=ModelUsage(10, 5, 0),
        )

    def usage_entries(self): return list(self._budget)
    def export_transcript(self): return {"provider": "openai", "exchanges": []}


class TransactionRepairWorkerFake:
    def __init__(self, provider, system, role):
        self.model = "fake-repair"; self.provider_name = "openai"
        self.capabilities = SessionCapabilities(); self._budget = []; self._step = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("episode:repair", "openai", self.model, 10, 5, 0))
        self._step += 1
        if self._step == 1:
            text = json.dumps(blocks, default=str)
            reading_id = re.search(r"id `([0-9a-f]{12})`", text).group(1)
            content = [ToolUseBlock(
                id="trp1", name="hypothesis_apply_reading",
                input={"branch": "main", "reading_id": reading_id},
            )]
        else:
            fork = None
            edits = []
            for message in blocks:
                for block in message.get("content") or []:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    try:
                        payload = json.loads(block.get("content") or "")
                    except (TypeError, json.JSONDecodeError):
                        continue
                    fork = payload.get("fork") or fork
                    edits = payload.get("edits") or edits
            result = {
                "applied": True,
                "best_branch": fork,
                "edits": edits,
                "verdicts": [{
                    "action": "apply_reading", "target": str(fork),
                    "verdict": "kept", "rationale": "bounded supported edit",
                }],
                "collateral": {"checked": True},
                "notes": "selected the only changed supported fork",
            }
            content = [ToolUseBlock(
                id="trp2", name="episode_submit_result",
                input={"result": result, "summary": "installed candidate ready"},
            )]
        return ModelResponse(content=content, usage=ModelUsage(10, 5, 0))

    def usage_entries(self): return list(self._budget)
    def export_transcript(self): return {"provider": "openai", "exchanges": []}


class TransactionLead:
    def __init__(self):
        self.model = "fake-lead"; self.provider_name = "openai"
        self.capabilities = SessionCapabilities(); self._budget = []; self._step = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("lead", "openai", self.model, 100, 10, 0))
        self._step += 1
        if self._step == 1:
            content = [ToolUseBlock(id="tx1", name="episode_run", input={
                "kind": "reading", "goal": "read main", "branches": ["main"],
            })]
        elif self._step == 2:
            content = [ToolUseBlock(id="tx2", name="repair_transaction", input={
                "branch": "main", "as_name": "transaction_repaired",
            })]
        elif self._step == 3:
            content = [ToolUseBlock(id="tx3", name="episode_run", input={
                "kind": "verify", "goal": "verify repaired",
                "branches": ["transaction_repaired"],
            })]
        else:
            content = [ToolUseBlock(id="tx4", name="meta_declare_solution", input={
                "branch": "transaction_repaired", "rationale": "freshly verified repair",
                "self_confidence": 0.8,
            })]
        return ModelResponse(content=content, usage=ModelUsage(100, 10, 0))

    def usage_entries(self): return list(self._budget)
    def export_transcript(self): return {"provider": "openai", "exchanges": []}


def test_repair_transaction_runs_validates_installs_and_requires_reverify(verify_fake):
    sessions_mod.register_session_builder("episode:reading", TransactionReadingWorkerFake)
    sessions_mod.register_session_builder("episode:repair", TransactionRepairWorkerFake)
    try:
        ct, state = _keyed_catton_state()
        from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel
        source_hash = _candidate_content_hash(
            _decoded_text_for_panel(state.workspace, "main")
        )
        state.verify_attestations.append({
            "branch": "main", "content_hash": source_hash,
            "renderer_id": "decoded_text_v1", "episode_id": "prior_verify",
            "coherence": 4, "reader_accepts": False,
            "reader_accepts_as_solution": False,
            "target_language_confidence": 0.8, "semantic_recoverability": 0.7,
            "damage_scope": "local", "repairability": "local_repair",
            "gloss": "partly readable",
            "anomalies": ["damaged middle word"], "created_turn": 0,
        })
        state.repair_agenda.append({
            "id": 1, "kind": "verify_anomaly",
            "source": "verify_attestation", "branch": "main",
            "content_hash": source_hash, "anomaly": "damaged middle word",
            "status": "open", "created_turn": 0,
        })
        art = run_v3(
            ct, session=TransactionLead(), language="en", max_iterations=6,
            cipher_id="v3_repair_transaction", resume_state=state,
        )
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:reading", None)
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)

    assert art.status == "solved"
    repaired = next(branch for branch in art.branches if branch.name == "transaction_repaired")
    assert repaired.decryption == "LATER"
    transactions = art.investigation_state["repair_transactions"]
    assert len(transactions) == 1
    assert transactions[0]["status"] == "installed"
    assert transactions[0]["reverification_required"] is True
    assert transactions[0]["source_content_hash"] != transactions[0]["result_content_hash"]
    assert transactions[0]["addressed_anomalies"] == ["damaged middle word"]
    # B2 identity fields + Slice-4 acceptance record (host-validated install).
    assert transactions[0]["interpretation_id"] == transactions[0]["reading_id"]
    assert len(transactions[0]["interpretation_digest"]) == 64
    assert transactions[0]["retry_of"] is None
    acceptance = transactions[0]["acceptance"]
    assert acceptance["policy"] == "default_deny_v1"
    assert [c["check"] for c in acceptance["checks"]] == [
        "winner_named", "worker_applied", "winner_fork_evidence",
        "edit_claims_bound", "winner_adjudicated", "no_op_probe",
        "scalar_non_decrease",
    ]
    assert all(c["passed"] for c in acceptance["checks"])
    assert acceptance["score_deltas"]["dict_rate_delta"] >= 0
    assert acceptance["score_deltas"]["quad_delta"] >= 0
    assert transactions[0]["installed_branch"] in acceptance["supported_forks"] or \
        transactions[0]["worker_winner"] in acceptance["supported_forks"]
    agenda = art.investigation_state["repair_agenda"]
    assert agenda[0]["status"] == "addressed"
    assert agenda[0]["addressed_by_transaction"] == transactions[0]["transaction_id"]
    assert art.attestations[-1]["branch"] == "transaction_repaired"
    assert [episode["kind"] for episode in art.episodes] == ["reading", "repair", "verify"]
    assert any(call.tool_name == "repair_transaction" for call in art.tool_calls)


# ---------------------------------------------------------------------------
# M5.3 Slice 7 Part A: the four distinguished branch roles
# ---------------------------------------------------------------------------
def _run_transaction_repair_flow():
    """The reading → repair_transaction install → verify → declare flow used by
    test_repair_transaction_runs_validates_installs_and_requires_reverify,
    factored so the branch-role tests reuse the exact fixtures."""
    from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel
    sessions_mod.register_session_builder("episode:reading", TransactionReadingWorkerFake)
    sessions_mod.register_session_builder("episode:repair", TransactionRepairWorkerFake)
    try:
        ct, state = _keyed_catton_state()
        source_hash = _candidate_content_hash(
            _decoded_text_for_panel(state.workspace, "main")
        )
        state.verify_attestations.append({
            "branch": "main", "content_hash": source_hash,
            "renderer_id": "decoded_text_v1", "episode_id": "prior_verify",
            "coherence": 4, "reader_accepts": False,
            "reader_accepts_as_solution": False,
            "target_language_confidence": 0.8, "semantic_recoverability": 0.7,
            "damage_scope": "local", "repairability": "local_repair",
            "gloss": "partly readable",
            "anomalies": ["damaged middle word"], "created_turn": 0,
        })
        state.repair_agenda.append({
            "id": 1, "kind": "verify_anomaly",
            "source": "verify_attestation", "branch": "main",
            "content_hash": source_hash, "anomaly": "damaged middle word",
            "status": "open", "created_turn": 0,
        })
        art = run_v3(
            ct, session=TransactionLead(), language="en", max_iterations=6,
            cipher_id="v3_branch_roles", resume_state=state,
        )
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:reading", None)
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)
    return art


def test_branch_roles_in_snapshot_and_artifact(verify_fake):
    art = _run_transaction_repair_flow()
    assert art.status == "solved"
    assert set(art.branch_roles) == {
        "best_scored_branch", "workflow_branch",
        "latest_installed_branch", "declared_or_selected_branch",
    }
    assert art.branch_roles["latest_installed_branch"] == "transaction_repaired"
    assert art.branch_roles["declared_or_selected_branch"] == "transaction_repaired"
    assert isinstance(art.branch_roles["workflow_branch"], str)
    # At least one mid-run workspace_snapshot event carries branch_roles with a
    # None declared_or_selected_branch (termination-scoped role).
    snapshots = [
        event.payload for event in art.loop_events
        if event.event == "workspace_snapshot"
    ]
    assert snapshots, "expected workspace_snapshot events"
    assert any(
        isinstance(payload.get("branch_roles"), dict)
        and payload["branch_roles"]["declared_or_selected_branch"] is None
        for payload in snapshots
    )


def test_branch_roles_recomputable_after_resume(verify_fake):
    from investigation.loop_v3 import _compute_branch_roles
    art = _run_transaction_repair_flow()
    restored = InvestigationState.from_artifact_dict(art.investigation_state)
    ex = _slice_executor(restored)
    roles = _compute_branch_roles(restored, ex)
    for key in ("best_scored_branch", "workflow_branch", "latest_installed_branch"):
        assert roles[key] == art.branch_roles[key]
    # declared_or_selected is termination-scoped: None on a resume recompute.
    assert roles["declared_or_selected_branch"] is None
    # Derived, not stored in InvestigationState.
    assert "branch_roles" not in art.investigation_state


def test_branch_roles_honest_unsolved_fallback():
    ct, _alpha = _caesar_cipher("THE DOG")
    session = ScriptedSession([[TextBlock(text="I have no idea.")]])
    art = run_v3(ct, session=session, language="en", max_iterations=3,
                 cipher_id="v3_roles_unsolved")
    assert art.status == "unsolved"
    assert art.fallback_selection is not None
    selected = next(
        event.payload.get("branch")
        for event in art.loop_events
        if event.event == "best_effort_selected"
    )
    # A fallback-selected branch counts as "selected" (master 509).
    assert art.branch_roles["declared_or_selected_branch"] == selected


def test_run_v3_interrupt_pairs_all_tool_results(monkeypatch):
    """R5: a mid-batch interrupt pairs every tool_use with a stopped result."""
    from agent import tools_v2
    ct, _alpha = _caesar_cipher("THE DOG")
    # One turn emitting TWO tool calls; the executor is interrupted on the 1st.
    scripts = [[
        ToolUseBlock(id="x1", name="decode_show", input={"branch": "main"}),
        ToolUseBlock(id="x2", name="decode_show", input={"branch": "main"}),
    ]]

    def boom(self, name, args, tool_use_id=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(tools_v2.WorkspaceToolExecutor, "execute", boom)
    art = run_v3(ct, session=ScriptedSession(scripts), language="en",
                 max_iterations=1, cipher_id="v3_interrupt")
    assert art.status == "stopped"

    def pair_sets(messages):
        uses, results = set(), set()
        for m in messages:
            for b in (m.get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    uses.add(b["id"])
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results.add(b["tool_use_id"])
        return uses, results

    # Both the logical transcript and the serialized state exchange are paired.
    assert pair_sets(art.messages) == ({"x1", "x2"}, {"x1", "x2"})
    assert pair_sets(art.investigation_state["recent_exchanges"]) == (
        {"x1", "x2"}, {"x1", "x2"})


# ---------------------------------------------------------------------------
# M5.3 Slice 1: the per-run paid ceiling (max_cost_usd). Uses gpt-5.5 fake
# sessions so estimate_provider_cost yields a nonzero, deterministic cost:
#   lead send   (1000, 50, 100) -> $0.00605
#   worker send (100,  20,   5) -> $0.0010775
# No paid model is called — the sessions are scripted fakes (Verification A).
# ---------------------------------------------------------------------------
def test_m53_max_cost_usd_prevents_next_lead_send():
    """A cost ceiling reached after the first lead send prevents the next paid
    lead send and still produces a complete, honestly-terminated artifact."""
    ct, _alpha = _caesar_cipher("THE DOG")
    lead = ScriptedSession([
        [ToolUseBlock(id="r1", name="decode_show", input={"branch": "main"})],
        [ToolUseBlock(id="r2", name="decode_show", input={"branch": "main"})],
    ], model="gpt-5.5")
    # After turn 1's $0.00605 lead send, $0.003 is exceeded -> turn 2 is blocked.
    art = run_v3(ct, session=lead, language="en", max_iterations=5,
                 cipher_id="v3_cost_lead", max_cost_usd=0.003)
    # Exactly one paid lead send happened.
    assert len(lead.blocks_seen) == 1
    assert any(event.event == "cost_ceiling_reached" for event in art.loop_events)
    # Honest termination with a complete artifact.
    assert art.status == "unsolved"
    assert art.solution is None
    assert art.investigation_state is not None
    assert art.budget_by_category  # finalize ran
    assert art.session_transcript is not None
    assert art.max_cost_usd == pytest.approx(0.003)


def test_invalid_search_episode_returns_exact_choices_and_both_routes():
    """Provider-schema bypasses still get an actionable host correction."""
    ct, _alpha = _caesar_cipher("THE DOG")
    lead = ScriptedSession([
        [ToolUseBlock(id="bad-search", name="episode_run", input={
            "kind": "search", "goal": "run automated solver",
            "branches": ["main"], "search_tool": "automated_solver",
        })],
        [TextBlock(text="stop")],
    ])
    art = run_v3(
        ct, session=lead, language="en", max_iterations=2,
        cipher_id="v3_invalid_search_name",
    )
    payload = next(
        item for item in _tool_result_payloads(art)
        if item.get("valid_search_tools")
    )
    assert "search_anneal" in payload["valid_search_tools"]
    assert payload["episode_corrected_example"]["search_tool"] == "search_anneal"
    assert payload["automated_solver_example"]["tool"] == "experiment_submit"


def test_repair_edit_claim_binding_accepts_only_host_evidenced_labels():
    evidence = {"W:D->F", "S009:R->S", "c=R"}
    claims = [
        "W:D->F",
        "Applied monoalphabetic key edit S009:R->S after review.",
        "Installed c=R on the supported fork.",
    ]
    unbound, normalized = _unbound_edit_claims(claims, evidence)
    assert unbound == []
    assert normalized[claims[0]] == ["W:D->F"]
    assert normalized[claims[1]] == ["S009:R->S"]
    assert normalized[claims[2]] == ["c=R"]


def test_repair_edit_claim_binding_rejects_invented_or_label_free_prose():
    evidence = {"W:D->F"}
    claims = [
        "Applied W:D->F and also X:A->B.",
        "Changed DEDECTU to DEFECTU.",
    ]
    unbound, normalized = _unbound_edit_claims(claims, evidence)
    assert unbound == sorted(claims)
    assert normalized == {}


def test_m53_cost_ceiling_between_two_worker_sends_terminates_episode():
    """A4: a ceiling that trips BETWEEN two worker sends ends the active episode
    budget-class (no second worker send) and makes no further paid call."""
    survey_good = {"findings": ["f"], "suspected_modes": [], "recommended_next": []}

    class CostlyWorker:
        """Two sends: a tool call, then a submit that must never fire."""

        def __init__(self, provider=None, system="", role="episode:survey"):
            self.model = "gpt-5.5"
            self.provider_name = "openai"
            self.capabilities = SessionCapabilities()
            self._budget: list[BudgetEntry] = []
            self._n = 0

        def send(self, blocks, tools=None, max_tokens=8192):
            self._budget.append(
                BudgetEntry("episode:survey", "openai", "gpt-5.5", 100, 20, 5))
            step = self._n
            self._n += 1
            if step == 0:
                content = [ToolUseBlock(id="w0", name="decode_show",
                                        input={"branch": "main"})]
            else:  # would submit — but the ceiling must block this send
                content = [ToolUseBlock(id="w1", name="episode_submit_result",
                                        input={"result": survey_good,
                                               "summary": "s"})]
            return ModelResponse(content=content, usage=ModelUsage(100, 20, 5))

        def usage_entries(self):
            return list(self._budget)

        def export_transcript(self):
            return {"provider": "openai", "model": self.model, "exchanges": []}

    workers: list[CostlyWorker] = []

    def _builder(provider, system, role):
        worker = CostlyWorker(provider, system, role)
        workers.append(worker)
        return worker

    sessions_mod.register_session_builder("episode:survey", _builder)
    try:
        ct, _alpha = _caesar_cipher("THE DOG")
        lead = ScriptedSession([
            [ToolUseBlock(id="e1", name="episode_run", input={
                "kind": "survey", "goal": "diagnose", "branches": ["main"]})],
            [TextBlock(text="would continue")],
        ], model="gpt-5.5")
        # base after lead send = $0.00605 (< 0.0068 -> worker send 1 allowed);
        # after worker send 1 = $0.0071275 (>= 0.0068 -> worker send 2 blocked).
        art = run_v3(ct, session=lead, language="en", max_iterations=5,
                     cipher_id="v3_cost_episode", max_cost_usd=0.0068)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:survey", None)

    # Exactly one worker send happened (the second was ceiling-blocked).
    assert len(workers) == 1
    assert len(workers[0]._budget) == 1
    # The episode is recorded budget-terminated in the ledger.
    assert art.episodes[-1]["kind"] == "survey"
    assert art.episodes[-1]["status"] == "episode_failed"
    assert art.episodes[-1]["failure_reason"] == "cost_ceiling_reached"
    assert art.episodes[-1]["tool_call_count"] == 1
    # No further paid lead send after the ceiling; complete honest artifact.
    assert len(lead.blocks_seen) == 1
    assert any(event.event == "cost_ceiling_reached" for event in art.loop_events)
    assert art.status == "unsolved"
    assert art.solution is None
    assert art.investigation_state is not None


# ---------------------------------------------------------------------------
# M5.3 Slices 2 + 4 — repair saturation, identity, host-validated acceptance
# ---------------------------------------------------------------------------
from agent.loop_shared import _candidate_content_hash as _cc_hash
from agent.loop_shared import _decoded_text_for_panel as _dt_panel
from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor
from investigation.context import allowed_episode_kinds, workflow_state
from investigation.reading import Reading, build_candidate_reading_packet
from investigation.state import (
    attestation_key, new_saturation_entry, pair_digest, saturation_key,
)


def _slice_executor(state):
    return WorkspaceToolExecutor(
        workspace=state.workspace, language="en",
        word_set={"LATER", "WATER"}, word_list=["LATER", "WATER"],
        pattern_dict={}, declaration_policy=NoGatesPolicy())


def _lead_results(art, tool_name):
    return [
        json.loads(tc.result) for tc in art.tool_calls
        if tc.tool_name == tool_name and tc.episode_id is None
    ]


def _tool_result_payloads(art):
    """Parsed tool_result payloads from the recorded exchange messages.

    Successful episode_run lead calls return directly (not through the call
    logger), so their results only appear here, not in art.tool_calls."""
    out = []
    for message in art.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                try:
                    out.append(json.loads(block.get("content") or ""))
                except (TypeError, json.JSONDecodeError):
                    pass
    return out


def _entry_for(art, content_hash, att_key):
    return art.investigation_state["repair_saturation"][
        saturation_key(content_hash, att_key)
    ]


_NO_FORK_APPLIED = lambda forks: {  # noqa: E731
    "applied": True, "best_branch": None, "edits": [],
    "verdicts": [], "collateral": {}, "notes": "",
}


def _run_single_repair(program, *, reading_text="LATER", cipher_id="s4"):
    ct, state = _keyed_catton_state()
    _seed_reading(state, reading_text)
    _register_programmable_repair([program])
    try:
        lead = ScriptedSession([
            [ToolUseBlock(id="t1", name="repair_transaction",
                          input={"branch": "main", "as_name": "transaction_repaired"})],
            [TextBlock(text="stop")],
        ])
        art = run_v3(ct, session=lead, language="en", max_iterations=5,
                     cipher_id=cipher_id, resume_state=state)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)
    return art


# --- Slice 2 -----------------------------------------------------------------

def test_s2_repeated_reading_suppressed_on_unchanged_content(verify_fake):
    sessions_mod.register_session_builder("episode:reading", TransactionReadingWorkerFake)
    try:
        ct, state = _keyed_catton_state()
        h = _seed_negative_attestation(state)
        lead = ScriptedSession([
            [ToolUseBlock(id="r1", name="episode_run",
                          input={"kind": "reading", "goal": "read", "branches": ["main"]})],
            [ToolUseBlock(id="r2", name="episode_run",
                          input={"kind": "reading", "goal": "read again", "branches": ["main"]})],
            [TextBlock(text="stop")],
        ])
        art = run_v3(ct, session=lead, language="en", max_iterations=6,
                     cipher_id="s2_reading_suppress", resume_state=state)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:reading", None)

    payloads = _tool_result_payloads(art)
    first = next(p for p in payloads if p.get("kind") == "reading" and p.get("reading_id"))
    suppressed = next(p for p in payloads if p.get("reason") == "duplicate_reading_suppressed")
    assert suppressed["status"] == "blocked"
    assert suppressed["existing_reading_id"] == first["reading_id"]
    assert [e["kind"] for e in art.episodes] == ["reading"]
    assert _entry_for(art, h, "ep:prior_verify")["readings"] == 1


def test_s2_evidence_failed_pair_blocked_under_new_as_name(verify_fake):
    ct, state = _keyed_catton_state()
    h = _seed_negative_attestation(state)
    _seed_reading(state, "LATER")
    _register_programmable_repair([{"apply": [], "result": _NO_FORK_APPLIED}])
    try:
        lead = ScriptedSession([
            [ToolUseBlock(id="t1", name="repair_transaction", input={"branch": "main"})],
            [ToolUseBlock(id="t2", name="repair_transaction",
                          input={"branch": "main", "as_name": "second_try"})],
            [TextBlock(text="stop")],
        ])
        art = run_v3(ct, session=lead, language="en", max_iterations=6,
                     cipher_id="s2_pair_blocked", resume_state=state)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)

    tx = art.investigation_state["repair_transactions"]
    assert len(tx) == 1
    assert tx[0]["status"] == "failed"
    assert tx[0]["reason"] == "no_changed_finalists"
    assert tx[0]["failure_class"] == "evidence"
    assert tx[0]["counted_evidence_failure"] is True
    second = _lead_results(art, "repair_transaction")[1]
    assert second["status"] == "blocked"
    assert second["reason"] == "pair_evidence_failed"
    entry = _entry_for(art, h, "ep:prior_verify")
    assert entry["evidence_failures"] == 1
    assert entry["exhausted"] is False


def test_s2_one_process_retry_can_succeed_and_install(verify_fake):
    ct, state = _keyed_catton_state()
    h = _seed_negative_attestation(state)
    _seed_reading(state, "LATER")
    _register_programmable_repair([
        {"apply": [{"reading_text": "LATER", "as_name": "fork_a"}],
         "result": lambda forks: {"applied": False, "best_branch": forks[0] if forks else None,
                                  "edits": [], "verdicts": [], "collateral": {}, "notes": ""}},
        {"apply": [{"reading_text": "LATER", "as_name": "fork_b"}],
         "result": lambda forks: {"applied": True, "best_branch": forks[0] if forks else None,
                                  "edits": [], "verdicts": [], "collateral": {}, "notes": ""}},
    ])
    try:
        lead = ScriptedSession([
            [ToolUseBlock(id="t1", name="repair_transaction",
                          input={"branch": "main", "as_name": "tx1"})],
            [ToolUseBlock(id="t2", name="repair_transaction",
                          input={"branch": "main", "as_name": "tx2"})],
            [TextBlock(text="stop")],
        ])
        art = run_v3(ct, session=lead, language="en", max_iterations=6,
                     cipher_id="s2_process_retry", resume_state=state)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)

    tx = art.investigation_state["repair_transactions"]
    assert len(tx) == 2
    assert tx[0]["reason"] == "worker_did_not_apply"
    assert tx[0]["failure_class"] == "process"
    assert tx[0]["counted_evidence_failure"] is False
    assert tx[0]["retry_of"] is None
    assert tx[1]["status"] == "installed"
    assert tx[1]["retry_of"] == tx[0]["transaction_id"]
    entry = _entry_for(art, h, "ep:prior_verify")
    assert entry["evidence_failures"] == 0
    assert entry["process_failures"][tx[0]["pair_digest"]] == 1
    assert entry["exhausted"] is False


def test_s2_two_evidence_failures_enter_repair_exhausted(verify_fake):
    ct, state = _keyed_catton_state()
    h = _seed_negative_attestation(state)
    rid1 = _seed_reading(state, "LATER")
    rid2 = _seed_reading(state, "WATER")
    _register_programmable_repair([
        {"apply": [], "result": _NO_FORK_APPLIED},
        {"apply": [], "result": _NO_FORK_APPLIED},
    ])
    try:
        lead = ScriptedSession([
            [ToolUseBlock(id="t1", name="repair_transaction",
                          input={"branch": "main", "reading_id": rid1})],
            [ToolUseBlock(id="t2", name="repair_transaction",
                          input={"branch": "main", "reading_id": rid2})],
            [ToolUseBlock(id="t3", name="repair_transaction",
                          input={"branch": "main", "reading_id": rid1})],
            [ToolUseBlock(id="t4", name="episode_run",
                          input={"kind": "reading", "goal": "read", "branches": ["main"]})],
            [TextBlock(text="stop")],
        ])
        art = run_v3(ct, session=lead, language="en", max_iterations=8,
                     cipher_id="s2_exhausted", resume_state=state)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)

    entry = _entry_for(art, h, "ep:prior_verify")
    assert entry["evidence_failures"] == 2
    assert entry["exhausted"] is True

    restored = InvestigationState.from_artifact_dict(art.investigation_state)
    ex = _slice_executor(restored)
    menu = workflow_state(restored, ex)
    assert menu["state"] == "repair_exhausted"
    joined = " ".join(menu["actions"])
    assert "experiment_submit" in joined
    assert "genuinely distinct" in joined
    assert "meta_declare_unsolved" in joined
    assert allowed_episode_kinds(restored, ex) == ["search", "compare", "verify"]

    tx3 = _lead_results(art, "repair_transaction")[2]
    assert tx3["reason"] == "repair_transaction_not_ready"
    assert tx3["workflow_state"] == "repair_exhausted"
    reading_block = _lead_results(art, "episode_run")[-1]
    assert reading_block["reason"] == "episode_kind_not_available"
    assert reading_block["allowed_kinds"] == ["compare", "search", "verify"]


def test_s2_experiment_submit_records_pending_pointer_when_exhausted(verify_fake, monkeypatch):
    ct, state = _keyed_catton_state()
    h = _seed_negative_attestation(state)
    att_key = "ep:prior_verify"
    entry = new_saturation_entry(h, att_key, 0)
    entry["evidence_failures"] = 2
    entry["exhausted"] = True
    entry["evidence_failed_pairs"] = ["p1", "p2"]
    state.repair_saturation[saturation_key(h, att_key)] = entry

    import investigation.loop_v3 as loop_v3_mod
    monkeypatch.setattr(
        loop_v3_mod, "dispatch_experiment_submit",
        lambda *a, **k: {"experiment_id": "exp999", "status": "pending"},
    )
    lead = ScriptedSession([
        [ToolUseBlock(id="x1", name="experiment_submit", input={"type": "automated_solver"})],
        [TextBlock(text="stop")],
    ])
    art = run_v3(ct, session=lead, language="en", max_iterations=5,
                 cipher_id="s2_pending", resume_state=state)

    entry_after = _entry_for(art, h, att_key)
    assert entry_after["pending_experiment_id"] == "exp999"

    restored = InvestigationState.from_artifact_dict(art.investigation_state)
    restored.experiment_queue.append({
        "experiment_id": "exp999", "type": "automated_solver",
        "status": "running", "collected": False,
    })
    ex = _slice_executor(restored)
    menu = workflow_state(restored, ex)
    assert menu["state"] == "repair_exhausted"
    assert sum("exp999" in a for a in menu["actions"]) == 1


# --- Slice 4 -----------------------------------------------------------------

def test_s4_fabricated_winner_rejected(verify_fake):
    art = _run_single_repair({
        "apply": [{"reading_text": "LATER", "as_name": "real_fork"}],
        "result": lambda forks: {"applied": True, "best_branch": "branch_i_invented",
                                 "edits": [], "verdicts": [], "collateral": {}, "notes": ""},
    }, cipher_id="s4_fabricated")
    tx = art.investigation_state["repair_transactions"][0]
    assert tx["status"] == "failed"
    assert tx["reason"] == "unsupported_winner"
    assert tx["failure_class"] == "process"
    assert tx["counted_evidence_failure"] is False
    assert "transaction_repaired" not in {b.name for b in art.branches}
    first = tx["acceptance"]["checks"][0]
    assert first["check"] == "winner_named"
    assert first["passed"] is False


def test_s4_fork_from_failed_call_rejected(verify_fake, monkeypatch):
    import investigation.actions as actions_mod

    def _boom(executor, args, state_readings, turn):
        ws = executor.workspace
        name = str(args.get("as_name") or "failfork")
        if not ws.has_branch(name):
            ws.fork(name, from_branch=str(args.get("branch") or "main"))
            ca = ws.cipher_text.alphabet
            pa = ws.plaintext_alphabet
            ws.set_mapping(name, ca.id_for("a"), pa.id_for("L"))
        raise RuntimeError("boom")

    monkeypatch.setattr(actions_mod, "_hypothesis_apply_reading", _boom)
    ct, state = _keyed_catton_state()
    _seed_reading(state, "LATER")
    _register_programmable_repair([{
        "apply": [{"reading_text": "LATER", "as_name": "failfork"}],
        "result": lambda forks: {"applied": True, "best_branch": "failfork",
                                 "edits": [], "verdicts": [], "collateral": {}, "notes": ""},
    }])
    try:
        lead = ScriptedSession([
            [ToolUseBlock(id="t1", name="repair_transaction",
                          input={"branch": "main", "as_name": "transaction_repaired"})],
            [TextBlock(text="stop")],
        ])
        art = run_v3(ct, session=lead, language="en", max_iterations=5,
                     cipher_id="s4_failed_call", resume_state=state)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)

    tx = art.investigation_state["repair_transactions"][0]
    assert tx["reason"] == "winner_fork_from_failed_call"
    assert tx["failure_class"] == "process"
    assert "transaction_repaired" not in {b.name for b in art.branches}
    failing = next(c for c in tx["acceptance"]["checks"] if not c["passed"])
    assert failing["check"] == "winner_fork_evidence"


def test_s4_unadjudicated_multi_finalist_and_evidence_reason_split(verify_fake):
    art_a = _run_single_repair({
        "apply": [{"reading_text": "LATER", "as_name": "fa"},
                  {"reading_text": "WATER", "as_name": "fb"}],
        "result": lambda forks: {"applied": True, "best_branch": forks[0],
                                 "edits": [], "verdicts": [], "collateral": {}, "notes": ""},
    }, cipher_id="s4_multi_a")
    txa = art_a.investigation_state["repair_transactions"][0]
    assert txa["reason"] == "no_winner_named_with_multiple_changed_finalists"
    assert txa["failure_class"] == "process"
    assert next(c for c in txa["acceptance"]["checks"] if not c["passed"])["check"] == "winner_adjudicated"

    art_b = _run_single_repair({
        "apply": [{"reading_text": "LATER", "as_name": "fa"},
                  {"reading_text": "WATER", "as_name": "fb"}],
        "result": lambda forks: {"applied": True, "best_branch": None,
                                 "edits": [], "verdicts": [], "collateral": {}, "notes": ""},
    }, cipher_id="s4_multi_b")
    txb = art_b.investigation_state["repair_transactions"][0]
    assert txb["reason"] == "no_winner_named_with_multiple_changed_finalists"
    assert txb["failure_class"] == "process"
    assert next(c for c in txb["acceptance"]["checks"] if not c["passed"])["check"] == "winner_named"

    art_c = _run_single_repair({"apply": [], "result": _NO_FORK_APPLIED}, cipher_id="s4_multi_c")
    txc = art_c.investigation_state["repair_transactions"][0]
    assert txc["reason"] == "no_changed_finalists"
    assert txc["failure_class"] == "evidence"

    art_d = _run_single_repair({
        "apply": [{"reading_text": "LATER", "as_name": "fa"}],
        "result": lambda forks: {"applied": True, "best_branch": None, "edits": [],
                                 "verdicts": [{"action": "apply_reading", "target": forks[0],
                                               "verdict": "rejected"}],
                                 "collateral": {}, "notes": ""},
    }, cipher_id="s4_multi_d")
    txd = art_d.investigation_state["repair_transactions"][0]
    assert txd["reason"] == "all_finalists_rejected"
    assert txd["failure_class"] == "evidence"

    for art in (art_a, art_b, art_c, art_d):
        assert "ambiguous_or_unchanged_finalists" not in json.dumps(art.investigation_state, default=str)
        for tc in art.tool_calls:
            assert "ambiguous_or_unchanged_finalists" not in (tc.result or "")


def test_s4_scalar_decrease_default_denied(verify_fake):
    art = _run_single_repair({
        "apply": [{"reading_text": "COTON", "as_name": "coton_fork"}],
        "result": lambda forks: {"applied": True, "best_branch": forks[0],
                                 "edits": [], "verdicts": [], "collateral": {}, "notes": ""},
    }, reading_text="COTON", cipher_id="s4_scalar")
    tx = art.investigation_state["repair_transactions"][0]
    assert tx["status"] == "failed"
    assert tx["reason"] == "materially_non_improving"
    assert tx["failure_class"] == "evidence"
    assert tx["counted_evidence_failure"] is True
    failing = next(c for c in tx["acceptance"]["checks"] if not c["passed"])
    assert failing["check"] == "scalar_non_decrease"
    assert tx["acceptance"]["score_deltas"]["quad_delta"] < 0
    assert tx["acceptance"]["scores_before"] is not None
    assert tx["acceptance"]["scores_after"] is not None
    assert "transaction_repaired" not in {b.name for b in art.branches}
    assert [e["kind"] for e in art.episodes] == ["repair"]
    payload = _lead_results(art, "repair_transaction")[0]
    assert payload["saturation"]["remaining_before_exhausted"] == 1


def test_s4_no_op_named_winner_is_evidence_no_op(verify_fake):
    art = _run_single_repair({
        "apply": [],
        "result": lambda forks: {"applied": True, "best_branch": "main", "edits": [],
                                 "verdicts": [], "collateral": {}, "notes": ""},
    }, cipher_id="s4_noop")
    tx = art.investigation_state["repair_transactions"][0]
    assert tx["reason"] == "no_op"
    assert tx["failure_class"] == "evidence"


def test_s4_duplicate_by_interpretation_digest(verify_fake):
    ct, state = _keyed_catton_state()
    rid1 = _seed_reading(state, "LATER", reading_id="a" * 12)
    rid2 = _seed_reading(state, "LATER", reading_id="b" * 12)
    _register_programmable_repair([{
        "apply": [{"reading_text": "LATER", "as_name": "first_fork"}],
        "result": lambda forks: {"applied": True, "best_branch": forks[0], "edits": [],
                                 "verdicts": [], "collateral": {}, "notes": ""},
    }])
    try:
        lead = ScriptedSession([
            [ToolUseBlock(id="t1", name="repair_transaction",
                          input={"branch": "main", "reading_id": rid1,
                                 "as_name": "transaction_repaired"})],
            [ToolUseBlock(id="t2", name="repair_transaction",
                          input={"branch": "main", "reading_id": rid2})],
            [TextBlock(text="stop")],
        ])
        art = run_v3(ct, session=lead, language="en", max_iterations=6,
                     cipher_id="s4_dup_digest", resume_state=state)
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)

    tx = art.investigation_state["repair_transactions"]
    assert len(tx) == 1
    assert tx[0]["status"] == "installed"
    second = _lead_results(art, "repair_transaction")[1]
    assert second["status"] == "duplicate_suppressed"
    assert second["reason"] == "source_and_reading_already_handled"


# ---------------------------------------------------------------------------
# Slice 6: fallback re-key on reader_accepts_as_solution + dispatcher coercion
# ---------------------------------------------------------------------------
def _positive_att(branch, chash, *, recov, turn, episode_id):
    return {
        "branch": branch, "content_hash": chash,
        "renderer_id": "decoded_text_v1", "episode_id": episode_id,
        "coherence": 9, "reader_accepts": True,
        "reader_accepts_as_solution": True,
        "target_language_confidence": 0.9, "semantic_recoverability": recov,
        "damage_scope": "local", "repairability": "local_repair",
        "created_turn": turn, "anomalies": [],
    }


def _negative_att(branch, chash, *, turn, episode_id):
    return {
        "branch": branch, "content_hash": chash,
        "renderer_id": "decoded_text_v1", "episode_id": episode_id,
        "coherence": 1, "reader_accepts": False,
        "reader_accepts_as_solution": False,
        "target_language_confidence": 0.2, "semantic_recoverability": 0.1,
        "damage_scope": "basin_wide", "repairability": "none",
        "created_turn": turn, "anomalies": [],
    }


def test_v3_fallback_rekeys_on_reader_accepts_as_solution():
    from investigation.loop_v3 import _select_v3_fallback

    ct, state = _keyed_catton_state()
    ws = state.workspace
    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet
    ws.fork("alt", from_branch="main")
    ws.set_mapping("alt", alpha.id_for("a"), pt.id_for("W"))  # WATON vs CATON
    hash_main = _cc_hash(_dt_panel(ws, "main"))
    hash_alt = _cc_hash(_dt_panel(ws, "alt"))
    assert hash_main != hash_alt
    executor = _slice_executor(state)

    # Both positive; alt has the higher semantic_recoverability -> alt wins.
    state.verify_attestations.append(
        _positive_att("main", hash_main, recov=0.6, turn=1, episode_id="a1"))
    state.verify_attestations.append(
        _positive_att("alt", hash_alt, recov=0.9, turn=1, episode_id="b1"))
    branch, selection = _select_v3_fallback(state, executor)
    assert branch == "alt"
    assert selection["tier"] == "fresh_positive_attestation"

    # A NEWER negative on alt's hash supersedes -> alt drops out, main wins.
    state.verify_attestations.append(
        _negative_att("alt", hash_alt, turn=2, episode_id="b2"))
    branch2, selection2 = _select_v3_fallback(state, executor)
    assert branch2 == "main"
    assert selection2["tier"] == "fresh_positive_attestation"

    # main's newest also negative -> no positive tier remains.
    state.verify_attestations.append(
        _negative_att("main", hash_main, turn=2, episode_id="a2"))
    _branch3, selection3 = _select_v3_fallback(state, executor)
    assert selection3["tier"] in {"fresh_compare_winner", "scalar_fallback"}


class _ClampVerifyFake(VerifyWorkerFake):
    """Submits out-of-range unit fields + omits the optional routing fields."""

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(
            BudgetEntry("episode:verify", "openai", "fake-luna", 50, 10, 0))
        result = {"coherence": 5, "reader_accepts": False,
                  "reader_accepts_as_solution": False,
                  "target_language_confidence": 1.7,
                  "semantic_recoverability": -0.2,
                  "gloss": "some words", "anomalies": ["odd run"],
                  "confidence": "low"}
        return ModelResponse(
            content=[ToolUseBlock(id="v1", name="episode_submit_result",
                                  input={"result": result, "summary": "clamp"})],
            usage=ModelUsage(50, 10, 0))


def test_run_v3_verify_dispatcher_clamps_and_defaults():
    sessions_mod.register_session_builder("episode:verify", _ClampVerifyFake)
    try:
        ct, alpha = _caesar_cipher("THE DOG")
        art = run_v3(ct, session=ScriptedSession(_solve_scripts(alpha)),
                     language="en", max_iterations=10, cipher_id="v3_clamp",
                     resume_state=_seeded_caesar_state(ct, alpha))
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)
    att = art.attestations[0]
    assert att["target_language_confidence"] == 1.0
    assert att["semantic_recoverability"] == 0.0
    assert att["damage_scope"] == "basin_wide"
    assert att["repairability"] == "none"
    assert att["uncertainty_note"] == ""
    # repairability "none" -> agenda NOT seeded even though an anomaly is present.
    verify_items = [
        item for item in art.repair_agenda
        if item.get("source") == "verify_attestation"
    ]
    assert verify_items == []
