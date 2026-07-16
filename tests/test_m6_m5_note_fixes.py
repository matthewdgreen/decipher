"""M6 Part A.2 — the three M5-review-note fixes (F5, F6, F7).

F5: episode_install_branch rename-on-collision keeps live attestation branch
    labels in sync (no mislabel).
F6: verify episodes require EXACTLY ONE existing branch (structured error on
    arity != 1 AND on a missing branch — the old ["typo","main"] silently
    verified main).
F7: the LOOP emits a `late_turn_attestation_hint` LoopEvent (context.py stays
    render-only); it lands in the artifact.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel
from agent.model_provider import ModelResponse, ModelUsage, TextBlock, ToolUseBlock
from investigation.loop_v3 import _resync_attestation_branch_on_rename, run_v3
from investigation.sessions import SessionCapabilities
from investigation.state import BudgetEntry, InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------
class ScriptedSession:
    """A ModelSession fake driven by a list of per-turn content-block lists."""

    def __init__(self, scripts, *, model="fake-openai", provider_name="openai"):
        self.model = model
        self.provider_name = provider_name
        self.capabilities = SessionCapabilities()
        self._scripts = list(scripts)
        self._budget: list[BudgetEntry] = []
        self._n = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(
            BudgetEntry("lead", self.provider_name, self.model, 1000, 50, 100)
        )
        content = self._scripts[min(self._n, len(self._scripts) - 1)]
        self._n += 1
        return ModelResponse(content=content, usage=ModelUsage(1000, 50, 100))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": self.provider_name, "model": self.model, "exchanges": []}


_CATON_MAP = {"a": "C", "b": "A", "c": "T", "d": "O", "e": "N"}


def _keyed_state(raw="abcde", mapping=None):
    """A resume-style state whose `main` branch decodes CATON."""
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ws = Workspace(ct)
    pt = ws.plaintext_alphabet
    for sym, letter in (mapping or _CATON_MAP).items():
        ws.set_mapping("main", alpha.id_for(sym), pt.id_for(letter))
    state = InvestigationState(workspace=ws, language="en")
    state.turn = 0
    return ct, state, alpha, pt


def _all_tool_results(art):
    out = []
    for m in art.messages:
        if m.get("role") != "user":
            continue
        for b in m.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                try:
                    out.append(json.loads(b.get("content") or ""))
                except (json.JSONDecodeError, TypeError):
                    pass
    return out


# ---------------------------------------------------------------------------
# F5 — rename-on-collision resyncs attestation branch, no mislabel
# ---------------------------------------------------------------------------
def test_f5_rename_resyncs_attestation_branch_no_mislabel():
    """End-to-end: an episode_install_branch collision renames the installed
    branch, and declaring the renamed branch carries a correctly-labeled
    attestation (the label followed the content to the new name)."""
    ct, state, alpha, pt = _keyed_state()  # main initially decodes CATON
    ws = state.workspace
    # Collision target `cand` already exists and decodes DIFFERENTLY (no key).
    ws.restore_branch("cand", key={})
    # Ledger entry whose fork, once installed, decodes CATON (main's key).
    main_key = dict(ws.get_branch("main").key)
    snap = {
        "name": "fork", "parent": "main", "created_iteration": 0,
        "key": {str(k): int(v) for k, v in main_key.items()},
        "tags": [], "metadata": {}, "word_spans": None,
        "token_order": None, "transform_pipeline": None,
    }
    state.episode_ledger.append({
        "episode_id": "ep1", "kind": "repair",
        "branch_snapshots": [snap], "agenda_additions": [],
    })
    # An attestation recorded under the INTENDED name `cand` for CATON content.
    caton_hash = _candidate_content_hash(_decoded_text_for_panel(ws, "main"))
    state.verify_attestations.append({
        "branch": "cand", "content_hash": caton_hash,
        "renderer_id": "decoded_text_v1", "episode_id": "epV",
        "coherence": 9, "reader_accepts": True, "gloss": "",
        "anomalies": [], "created_turn": 0,
    })
    # Move the live source away from CATON after capturing the snapshot. The
    # install is now content-distinct (so dedup does not intentionally alias it
    # back to main), while still exercising the collision rename/resync path.
    ws.set_mapping("main", alpha.id_for("a"), pt.id_for("Z"))

    scripts = [
        [ToolUseBlock(id="i1", name="episode_install_branch",
                      input={"episode_id": "ep1", "branch": "fork",
                             "as_name": "cand"})],
        [ToolUseBlock(id="d1", name="meta_declare_solution",
                      input={"branch": "cand_2",
                             "rationale": "reads as CATON",
                             "self_confidence": 0.9})],
    ]
    art = run_v3(ct, session=ScriptedSession(scripts), language="en",
                 max_iterations=5, cipher_id="v3_f5", resume_state=state)

    # The collision renamed the installed branch to cand_2.
    assert any(b.name == "cand_2" for b in art.branches)
    # The attestation was re-pointed cand -> cand_2 (it now names the live branch
    # that actually holds the attested CATON content — no mislabel).
    assert art.attestations and art.attestations[0]["branch"] == "cand_2"
    # Declaration ALLOWED (hash match) and carries the correctly-labeled att.
    assert art.status == "solved"
    assert art.solution is not None and art.solution.branch == "cand_2"
    assert art.solution.attestation is not None
    assert art.solution.attestation["branch"] == "cand_2"


def test_f5_helper_repoints_only_moved_content():
    _ct, state, _a, _p = _keyed_state()
    ws = state.workspace
    ws.restore_branch("cand", key={})                          # decodes ?????
    ws.restore_branch("cand_2", key=dict(ws.get_branch("main").key))  # CATON
    caton_hash = _candidate_content_hash(_decoded_text_for_panel(ws, "cand_2"))
    atts = [{"branch": "cand", "content_hash": caton_hash}]
    _resync_attestation_branch_on_rename(ws, atts, "cand", "cand_2")
    assert atts[0]["branch"] == "cand_2"  # content moved -> label follows


def test_f5_helper_no_mislabel_when_target_still_owns_content():
    _ct, state, _a, _p = _keyed_state()
    ws = state.workspace
    ws.restore_branch("cand", key=dict(ws.get_branch("main").key))  # CATON
    ws.restore_branch("cand_2", key={})                             # ?????
    caton_hash = _candidate_content_hash(_decoded_text_for_panel(ws, "cand"))
    atts = [{"branch": "cand", "content_hash": caton_hash}]
    _resync_attestation_branch_on_rename(ws, atts, "cand", "cand_2")
    # `cand` legitimately still owns the attested content -> label NOT stolen.
    assert atts[0]["branch"] == "cand"


# ---------------------------------------------------------------------------
# F6 — verify requires exactly one existing branch
# ---------------------------------------------------------------------------
def test_f6_verify_requires_exactly_one_existing_branch():
    """Arity != 1 AND missing-branch both yield structured errors; a stray name
    ["typo","main"] no longer silently verifies (mislabels) `main`."""
    ct, state, _a, _p = _keyed_state()
    scripts = [
        # Two names -> arity error (main NOT silently attested).
        [ToolUseBlock(id="v_multi", name="episode_run",
                      input={"kind": "verify", "goal": "x",
                             "branches": ["typo", "main"]})],
        # Empty list -> arity error.
        [ToolUseBlock(id="v_empty", name="episode_run",
                      input={"kind": "verify", "goal": "x", "branches": []})],
        # One name that does not exist -> existence error naming it.
        [ToolUseBlock(id="v_missing", name="episode_run",
                      input={"kind": "verify", "goal": "x",
                             "branches": ["nope"]})],
        [TextBlock(text="done")],  # no tool -> exhaust
    ]
    art = run_v3(ct, session=ScriptedSession(scripts), language="en",
                 max_iterations=6, cipher_id="v3_f6", resume_state=state)
    results = _all_tool_results(art)
    blob = json.dumps(results)
    assert "exactly one branch" in blob   # both arity cases
    assert "does not exist" in blob       # missing-branch case
    assert "'nope'" in blob               # error names the missing branch
    # No verify episode ran, so NO attestation was minted (the mislabel risk).
    assert art.attestations == []
    assert not any(e["kind"] == "verify" for e in art.episodes)


# ---------------------------------------------------------------------------
# F7 — the loop emits the late-turn attestation hint as a LoopEvent
# ---------------------------------------------------------------------------
def test_f7_late_turn_attestation_hint_emitted_as_loop_event():
    """With ≤2 turns left and the best branch unattested, the LOOP emits a
    `late_turn_attestation_hint` LoopEvent that lands in the artifact — and only
    in the last-2-turns window (predicate correctness)."""
    ct, state, _a, _p = _keyed_state()
    # Never declare/verify: read-only decode_show each turn. max_iterations=5 ->
    # the predicate is true only on turns 3, 4, 5.
    scripts = [[ToolUseBlock(id=f"s{i}", name="decode_show",
                             input={"branch": "main"})] for i in range(6)]
    art = run_v3(ct, session=ScriptedSession(scripts), language="en",
                 max_iterations=5, cipher_id="v3_f7", resume_state=state)

    hints = [e for e in art.loop_events if e.event == "late_turn_attestation_hint"]
    assert hints, "expected the late-turn attestation hint LoopEvent in the artifact"
    assert all(e.outer_iteration >= 3 for e in hints)   # last-2-turns window only
    # M5.1 deduplicates the reminder per unchanged content hash.
    assert len(hints) == 1
    assert all(e.payload["branch"] == "main" for e in hints)
    assert all(e.payload["turns_remaining"] <= 2 for e in hints)


def test_f7_no_hint_when_best_branch_freshly_attested(verify_fake):
    """The negative: a fresh attestation on the best branch suppresses the hint
    even inside the late-turn window (the predicate keys off attestation, not the
    turn count alone)."""
    ct, state, alpha, _p = _keyed_state()
    mapping = {alpha.id_for(s): _p.id_for(l) for s, l in _CATON_MAP.items()}
    # turn 1 verify main; turns 2-5 read-only (never declare).
    scripts = [
        [ToolUseBlock(id="ev", name="episode_run",
                      input={"kind": "verify", "goal": "verify main",
                             "branches": ["main"]})],
        [ToolUseBlock(id="s1", name="decode_show", input={"branch": "main"})],
        [ToolUseBlock(id="s2", name="decode_show", input={"branch": "main"})],
        [ToolUseBlock(id="s3", name="decode_show", input={"branch": "main"})],
        [ToolUseBlock(id="s4", name="decode_show", input={"branch": "main"})],
    ]
    art = run_v3(ct, session=ScriptedSession(scripts), language="en",
                 max_iterations=5, cipher_id="v3_f7neg", resume_state=state)
    # main was attested on turn 1 and never mutated -> no late-turn hint fires.
    assert not any(
        e.event == "late_turn_attestation_hint" for e in art.loop_events
    )
    assert art.attestations and art.attestations[0]["branch"] == "main"


class _VerifyWorkerFake:
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
    from investigation import sessions as sessions_mod
    sessions_mod.register_session_builder("episode:verify", _VerifyWorkerFake)
    yield
    sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)
