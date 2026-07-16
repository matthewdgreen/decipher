"""run_v3 lead-loop tests (M1 Part 4/5) with scripted fake sessions."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agent.model_provider import ModelProviderError, ModelResponse, ModelUsage, TextBlock, ToolUseBlock
from investigation import sessions as sessions_mod
from investigation.loop_v3 import _fresh_compare_winner, run_v3
from investigation.sessions import SessionCapabilities
from investigation.state import BudgetEntry, InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


def _caesar(text: str, shift: int) -> str:
    return "".join(
        chr((ord(c) - 65 + shift) % 26 + 65) if c.isalpha() else c for c in text
    )


def _caesar_cipher(plaintext: str, shift: int = 3):
    raw = _caesar(plaintext, shift)
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    return CipherText(raw=raw, alphabet=alpha, separator=" "), alpha


class ScriptedSession:
    """A ModelSession fake driven by a list of per-turn content-block lists."""

    def __init__(self, scripts, *, model="fake-openai", provider_name="openai"):
        self.model = model
        self.provider_name = provider_name
        self.capabilities = SessionCapabilities()
        self._scripts = list(scripts)
        self._budget: list[BudgetEntry] = []
        self.blocks_seen: list = []
        self.tools_seen: list = []
        self._n = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self.blocks_seen.append(blocks)
        self.tools_seen.append(tools)
        self._budget.append(
            BudgetEntry("lead", self.provider_name, self.model, 1000, 50, 100)
        )
        content = self._scripts[min(self._n, len(self._scripts) - 1)]
        self._n += 1
        return ModelResponse(content=content, usage=ModelUsage(1000, 50, 100))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": self.provider_name, "model": self.model,
                "exchanges": [{"n": self._n}]}


class ErrorSession(ScriptedSession):
    def send(self, blocks, tools=None, max_tokens=8192):
        raise ModelProviderError("simulated API overload")


class VerifyWorkerFake:
    """M5: a verify episode worker that accepts the candidate as coherent English.

    Registered per test via the ``verify_fake`` fixture; the run's
    AttestationPolicy needs an attestation before meta_declare_solution is
    allowed.
    """

    def __init__(self, provider=None, system="", role="episode:verify"):
        self.model = "fake-luna"
        self.provider_name = "openai"
        self.capabilities = SessionCapabilities()
        self._budget: list[BudgetEntry] = []

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(
            BudgetEntry("episode:verify", "openai", "fake-luna", 50, 10, 0)
        )
        result = {"coherence": 9, "reader_accepts": True,
                  "gloss": "reads as clear English", "anomalies": [],
                  "confidence": "high"}
        return ModelResponse(
            content=[ToolUseBlock(id="v1", name="episode_submit_result",
                                  input={"result": result, "summary": "reads well"})],
            usage=ModelUsage(50, 10, 0))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": self.model, "exchanges": []}


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


def test_run_v3_fallback_declared_on_provider_error():
    ct, _alpha = _caesar_cipher("THE DOG")
    art = run_v3(ct, session=ErrorSession([]), language="en", max_iterations=5,
                 cipher_id="v3_error")
    assert art.status == "fallback_declared"
    assert art.auto_declared is True
    assert art.solution is not None
    assert art.solution.self_confidence == 0.0
    assert "API error" in art.error_message


def test_run_v3_exhaustion_falls_back():
    ct, _alpha = _caesar_cipher("THE DOG")
    # Text-only responses → no tool calls → exhausted → fallback.
    session = ScriptedSession([[TextBlock(text="I have no idea.")]])
    art = run_v3(ct, session=session, language="en", max_iterations=3,
                 cipher_id="v3_exhaust")
    assert art.status == "fallback_declared"
    assert art.auto_declared is True
    assert art.solution is not None


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
    assert art.solution.self_confidence == 0.9


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


def test_run_v3_declare_without_verify_is_blocked_then_falls_back():
    """M5: meta_declare_solution with no verify attestation is blocked; the run
    exhausts to the fallback declaration, which itself needs no attestation."""
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
    # Fallback path (needs no attestation).
    assert art.status == "fallback_declared"
    assert art.auto_declared is True
    assert not art.attestations
    # A fallback declaration carries no attestation.
    assert art.solution is not None and art.solution.attestation is None


class _WeakVerifyFake(VerifyWorkerFake):
    """A verify worker that reports words-but-not-sentences (weak, not accepted)."""

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(
            BudgetEntry("episode:verify", "openai", "fake-luna", 50, 10, 0))
        result = {"coherence": 3, "reader_accepts": False,
                  "gloss": "reads as words but not coherent sentences",
                  "anomalies": ["non-word run", "broken clause"], "confidence": "low"}
        return ModelResponse(
            content=[ToolUseBlock(id="v1", name="episode_submit_result",
                                  input={"result": result, "summary": "weak"})],
            usage=ModelUsage(50, 10, 0))


def test_run_v3_weak_attestation_allows_declare_and_carries_weakness():
    """M5/C6: a WEAK attestation does not block a deliberate declaration; the
    declaration carries the weakness so a weak-but-declared solve is visibly
    weak in the artifact."""
    sessions_mod.register_session_builder("episode:verify", _WeakVerifyFake)
    try:
        ct, alpha = _caesar_cipher("THE DOG")
        art = run_v3(ct, session=ScriptedSession(_solve_scripts(alpha)),
                     language="en", max_iterations=10, cipher_id="v3_weak",
                     resume_state=_seeded_caesar_state(ct, alpha))
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)
    assert art.status == "solved"
    assert art.solution is not None and art.solution.attestation is not None
    assert art.solution.attestation["reader_accepts"] is False
    assert art.solution.attestation["coherence"] == 3
    assert art.solution.attestation["anomalies"] == ["non-word run", "broken clause"]


class _OutOfScaleVerifyFake(VerifyWorkerFake):
    """Review F-1: a reader answering coherence=12 — a scale violation (likely a
    0-100-scale reading, i.e. LOW) — while REJECTING the text."""

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(
            BudgetEntry("episode:verify", "openai", "fake-luna", 50, 10, 0))
        result = {"coherence": 12, "reader_accepts": False,
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
    # The hash-gated declaration still proceeds (weak-doesn't-block, C6) and
    # carries the conservative record.
    assert art.status == "solved"
    assert art.solution.attestation["coherence"] == 0


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
    assert art.status in {"exhausted", "fallback_declared"}


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


def _keyed_catton_state():
    raw = "abcde"  # single word, decodes to CATON
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ws = Workspace(ct)
    pt = ws.plaintext_alphabet
    from investigation.state import InvestigationState
    for sym, letter in {"a": "C", "b": "A", "c": "T", "d": "O", "e": "N"}.items():
        ws.set_mapping("main", alpha.id_for(sym), pt.id_for(letter))
    state = InvestigationState(workspace=ws, language="en")
    state.turn = 0
    return ct, state


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
                "result": {"reading_text": "CATTON",
                           "fragments": [{"text": "CATTON", "confidence": 0.9}],
                           "holes": ["extra T"], "overall_confidence": 0.6},
                "summary": "read as CATTON",
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
    assert art.readings and art.readings[0]["fragments"][0]["text"] == "CATTON"
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

    # The repair episode's residual agenda merged into state.repair_agenda on
    # install, remapped to the installed branch name.
    residual = [i for i in art.repair_agenda if i.get("kind") == "reading_residual"]
    assert residual and residual[0]["branch"] == "repaired"

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
            "reading_text": "COTON",
            "fragments": [{"text": "COTON", "repair_text": "COTON", "confidence": 0.95}],
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
        art = run_v3(
            ct, session=TransactionLead(), language="en", max_iterations=6,
            cipher_id="v3_repair_transaction", resume_state=state,
        )
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:reading", None)
        sessions_mod._SESSION_BUILDERS.pop("episode:repair", None)

    assert art.status == "solved"
    repaired = next(branch for branch in art.branches if branch.name == "transaction_repaired")
    assert repaired.decryption == "COTON"
    transactions = art.investigation_state["repair_transactions"]
    assert len(transactions) == 1
    assert transactions[0]["status"] == "installed"
    assert transactions[0]["reverification_required"] is True
    assert transactions[0]["source_content_hash"] != transactions[0]["result_content_hash"]
    assert art.attestations[-1]["branch"] == "transaction_repaired"
    assert [episode["kind"] for episode in art.episodes] == ["reading", "repair", "verify"]
    assert any(call.tool_name == "repair_transaction" for call in art.tool_calls)


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
