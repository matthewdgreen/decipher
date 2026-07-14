"""Episode runtime tests (M2 Part 8): isolation, A9 failure paths, A4 filter,
finalist sharing, budget categories, and the scripted end-to-end workflow."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.model_provider import ModelResponse, ModelUsage, TextBlock, ToolUseBlock
from investigation import sessions as sessions_mod
from investigation.board import CARD_MIRROR_KEYS
from investigation.episodes import (
    EPISODE_KINDS,
    EpisodeSpec,
    _build_episode_workspace,
    episode_toolset_for,
    run_episode,
    validate_against_schema,
)
from investigation.sessions import SessionCapabilities
from investigation.state import BudgetEntry, InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class EpisodeFake:
    """A scripted episode session: a list of per-send content-block lists."""

    def __init__(self, scripts, *, role="episode:survey", model="fake-luna",
                 provider="openai", category=None):
        self.model = model
        self.provider_name = provider
        self._role = role
        self._category = category or role
        self.capabilities = SessionCapabilities()
        self._scripts = list(scripts)
        self._n = 0
        self._budget: list[BudgetEntry] = []
        self.blocks_seen: list = []

    def send(self, blocks, tools=None, max_tokens=8192):
        self.blocks_seen.append(blocks)
        self._budget.append(
            BudgetEntry(self._category, self.provider_name, self.model, 100, 20, 5)
        )
        content = self._scripts[min(self._n, len(self._scripts) - 1)]
        self._n += 1
        return ModelResponse(content=content, usage=ModelUsage(100, 20, 5))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": self.provider_name, "model": self.model, "exchanges": []}


def _submit(result, summary="did the thing", tid="s1"):
    return ToolUseBlock(id=tid, name="episode_submit_result",
                        input={"result": result, "summary": summary})


def _simple_state(raw="ABCDEFGHIJKL"):
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    return InvestigationState(workspace=Workspace(ct), language="en")


# ---------------------------------------------------------------------------
# Toolset validation
# ---------------------------------------------------------------------------
def test_episode_spec_validates_toolset():
    # A valid survey spec constructs.
    spec = EpisodeSpec("survey", "diagnose", inputs={"branches": ["main"]})
    assert "decode_show" in spec.toolset
    assert spec.budget.max_tool_calls == 10

    # Excluded / meta / hypothesis tools are rejected.
    for bad in [
        "search_transform_homophonic",           # EPISODE_EXCLUDED_TOOLS
        "meta_declare_solution",                  # meta_*
        "workspace_create_hypothesis_branch",     # hypothesis handler
        "inspect_benchmark_context",              # inspect_*
        "list_related_records",                   # list_*
    ]:
        with pytest.raises(ValueError):
            EpisodeSpec("survey", "g", inputs={"branches": ["main"]},
                        toolset=["decode_show", bad])


def test_search_toolset_includes_companions():
    ts = episode_toolset_for("search", {"search_tool": "search_pure_transposition"})
    assert "search_pure_transposition" in ts
    assert "search_review_pure_transposition_finalists" in ts
    assert "act_install_pure_transposition_finalists" in ts
    assert {"decode_show", "score_panel"} <= ts


# ---------------------------------------------------------------------------
# A1 + F5: isolation with TRUE deep copies (nested metadata + pipeline)
# ---------------------------------------------------------------------------
def test_episode_workspace_is_deep_copied():
    state = _simple_state()
    ws = state.workspace
    ws.fork("hyp", from_branch="main")
    hyp = ws.get_branch("hyp")
    hyp.key[0] = 1
    hyp.metadata["nested"] = {"count": 1, "tags": ["a"]}
    hyp.transform_pipeline = {"steps": [{"op": "reverse"}]}

    ep_ws = _build_episode_workspace(state, ["hyp"])
    ep = ep_ws.get_branch("hyp")
    # Mutate NESTED metadata, the pipeline, and the key inside the episode copy.
    ep.metadata["nested"]["count"] = 999
    ep.metadata["nested"]["tags"].append("hacked")
    ep.metadata["new_key"] = "x"
    ep.transform_pipeline["steps"].append({"op": "hack"})
    ep.key[0] = 42

    # The lead branch is untouched.
    assert hyp.metadata["nested"] == {"count": 1, "tags": ["a"]}
    assert "new_key" not in hyp.metadata
    assert hyp.transform_pipeline == {"steps": [{"op": "reverse"}]}
    assert hyp.key[0] == 1


# ---------------------------------------------------------------------------
# A9 failure semantics
# ---------------------------------------------------------------------------
def test_a9_schema_double_failure():
    state = _simple_state()
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})
    bad = {"findings": "not-an-array"}  # wrong type
    scripts = [
        [_submit(bad, tid="a")],
        [_submit(bad, tid="b")],  # second mismatch → episode_failed
    ]
    fake = EpisodeFake(scripts, role="episode:survey")
    res = run_episode(spec, state, session=fake)
    assert res.status == "episode_failed"
    assert res.failure_reason == "schema_mismatch"
    assert res.raw_text is not None
    # The ledger recorded the failure.
    assert state.episode_ledger[-1]["failure_reason"] == "schema_mismatch"


def test_a9_schema_retry_then_success():
    state = _simple_state()
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})
    good = {"findings": ["f"], "suspected_modes": [], "recommended_next": []}
    scripts = [
        [_submit({"findings": 3}, tid="a")],  # first mismatch → retry
        [_submit(good, tid="b")],             # second attempt valid → ok
    ]
    res = run_episode(spec, state, session=EpisodeFake(scripts))
    assert res.status == "ok"
    assert res.result == good


def test_a9_budget_exhausted_call_count_then_final_send():
    state = _simple_state()
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"],
                                              "max_tool_calls": 1})
    good = {"findings": ["f"], "suspected_modes": [], "recommended_next": []}
    scripts = [
        [ToolUseBlock(id="t1", name="decode_show", input={"branch": "main"})],
        # budget hit → final tool-less send returns the JSON as text
        [TextBlock(text=json.dumps(good))],
    ]
    res = run_episode(spec, state, session=EpisodeFake(scripts))
    assert res.status == "ok"
    assert res.result == good
    assert res.tool_call_count == 1


def test_a9_budget_exhausted_wall_clock(monkeypatch):
    import investigation.episodes as ep_mod
    state = _simple_state()
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})
    # Clock jumps past the wall-clock budget before the first send.
    ticks = iter([0.0, 10_000.0, 10_001.0, 10_002.0, 10_003.0])
    monkeypatch.setattr(ep_mod.time, "time", lambda: next(ticks, 10_100.0))
    # Final tool-less send returns invalid text → budget_exhausted failure.
    scripts = [[TextBlock(text="I could not finish.")]]
    res = run_episode(spec, state, session=EpisodeFake(scripts))
    assert res.status == "episode_failed"
    assert res.failure_reason == "budget_exhausted"


def test_a9_handler_exception_rides_in_tool_result():
    state = _simple_state()
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})
    good = {"findings": ["f"], "suspected_modes": [], "recommended_next": []}
    # decode_show on a non-existent branch raises inside the handler; the
    # executor turns it into an error tool result and the episode continues.
    scripts = [
        [ToolUseBlock(id="t1", name="decode_show", input={"branch": "nope"})],
        [_submit(good, tid="b")],
    ]
    res = run_episode(spec, state, session=EpisodeFake(scripts))
    assert res.status == "ok"


def test_a9_interrupt_records_ledger_and_reraises():
    state = _simple_state()
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})

    class Boom(EpisodeFake):
        def send(self, blocks, tools=None, max_tokens=8192):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_episode(spec, state, session=Boom([]))
    assert state.episode_ledger[-1]["failure_reason"] == "interrupted"
    # Budget entries from the (empty) session are merged; no crash.


# ---------------------------------------------------------------------------
# A4: suggestion filter (nested dicts) inside an episode toolset
# ---------------------------------------------------------------------------
def test_a4_suggestion_filter_nested():
    from agent.tools_v2 import _filter_next_tool_hints
    toolset = {"decode_show", "score_panel"}
    obj = {
        "status": "ok",
        "suggested_next_tools": ["decode_show", "search_anneal", "meta_declare_solution"],
        "recommended_next_tool": "search_anneal",
        "fix_tool": "decode_show",
        "nested": {
            "suggested_next_tools": ["score_panel", "workspace_fork"],
            "recommended_next_tool": "score_panel",
        },
    }
    out = _filter_next_tool_hints(obj, toolset)
    assert out["suggested_next_tools"] == ["decode_show"]
    assert out["recommended_next_tool"] is None       # off-toolset → nulled
    assert out["fix_tool"] == "decode_show"           # in toolset → kept
    assert out["nested"]["suggested_next_tools"] == ["score_panel"]
    assert out["nested"]["recommended_next_tool"] == "score_panel"
    # Input untouched.
    assert obj["suggested_next_tools"][1] == "search_anneal"


def test_a4_episode_rejects_off_toolset_tool():
    state = _simple_state()
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})
    good = {"findings": ["f"], "suspected_modes": [], "recommended_next": []}
    # workspace_fork is not in the survey toolset → neutral rejection.
    scripts = [
        [ToolUseBlock(id="t1", name="workspace_fork",
                      input={"new_name": "x", "from_branch": "main"})],
        [_submit(good, tid="b")],
    ]
    res = run_episode(spec, state, session=EpisodeFake(scripts))
    reject = next(tc for tc in res.tool_calls if tc.tool_name == "workspace_fork")
    payload = json.loads(reject.result)
    assert "not in this episode's toolset" in payload["error"]
    assert "allowed_tools" in payload
    assert res.status == "ok"


# ---------------------------------------------------------------------------
# A7: episode:<kind> budget categories under two different fake models
# ---------------------------------------------------------------------------
def test_budget_categories_per_kind_and_model():
    state = _simple_state()
    survey_good = {"findings": [], "suspected_modes": [], "recommended_next": []}
    compare_good = {"ranking": ["main"], "verdicts": [], "winner": "main"}
    run_episode(
        EpisodeSpec("survey", "g", inputs={"branches": ["main"]}),
        state, session=EpisodeFake([[_submit(survey_good)]],
                                   role="episode:survey", model="model-A"),
    )
    run_episode(
        EpisodeSpec("compare", "g", inputs={"branches": ["main"]}),
        state, session=EpisodeFake([[_submit(compare_good)]],
                                   role="episode:compare", model="model-B"),
    )
    cats = state.budget_by_category()
    assert "episode:survey" in cats
    assert "episode:compare" in cats
    models = {e.model for e in state.budget_ledger}
    assert models == {"model-A", "model-B"}


# ---------------------------------------------------------------------------
# Finalist-session sharing across the episode boundary + store round-trip (F7)
# ---------------------------------------------------------------------------
def test_finalist_store_shared_and_roundtrips():
    from agent.finalist_sessions import FinalistSessionStore
    state = _simple_state()
    # A "search" episode's finalist session is written into the state store.
    sid = state.finalist_sessions.new_session("word_repair", {"x": 1}, packets=[])
    assert sid == "word_repair_1"
    # Round-trip preserves counters (ids don't restart) and payloads.
    data = json.loads(json.dumps(state.finalist_sessions.to_dict()))
    restored = FinalistSessionStore.from_dict(data)
    assert restored.get("word_repair", "word_repair_1") == {"x": 1, "packets": []}
    # Next id continues from the counter, not from 1.
    assert restored.new_session("word_repair", {"y": 2}, packets=[]) == "word_repair_2"
    # All four kinds round-trip.
    for kind in ("transform_search", "pure_transposition", "null_mask"):
        state.finalist_sessions.new_session(kind, {"k": kind}, packets=[])
    data2 = json.loads(json.dumps(state.finalist_sessions.to_dict()))
    r2 = FinalistSessionStore.from_dict(data2)
    for kind in ("transform_search", "pure_transposition", "null_mask", "word_repair"):
        assert any(sid.startswith(kind) for sid, _ in r2.sessions(kind))


# ---------------------------------------------------------------------------
# Local schema validator (nullable types, nested arrays)
# ---------------------------------------------------------------------------
def test_schema_validator_nullable_and_nested():
    schema = EPISODE_KINDS["search"]["result_schema"]
    ok = {"improved": True, "best_branch": None, "score_summary": {},
          "notes": "n", "finalist_session_id": None}
    assert validate_against_schema(ok, schema) == []
    bad = {"improved": "yes", "best_branch": 3, "score_summary": [], "notes": "n"}
    errs = validate_against_schema(bad, schema)
    assert any("improved" in e for e in errs)
    assert any("best_branch" in e for e in errs)


# ---------------------------------------------------------------------------
# Scripted survey → search → reading → compare end-to-end (acceptance #2)
# ---------------------------------------------------------------------------
def _find_episode_id(blocks, kind):
    for m in blocks:
        for b in (m.get("content") or []):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                try:
                    data = json.loads(b.get("content") or "")
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(data, dict) and data.get("kind") == kind and data.get("episode_id"):
                    return data["episode_id"]
    return None


class WorkflowLeadSession:
    """A lead fake that drives survey→search→reading→compare→install→declare,
    reading the search episode_id back out of the incoming tool results."""

    def __init__(self):
        self.model = "fake-lead"
        self.provider_name = "openai"
        self.capabilities = SessionCapabilities()
        self._budget: list[BudgetEntry] = []
        self._step = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("lead", "openai", "fake-lead", 500, 40, 0))
        step = self._step
        self._step += 1
        if step == 0:
            content = [ToolUseBlock(id="e1", name="episode_run", input={
                "kind": "survey", "goal": "diagnose", "branches": ["main"]})]
        elif step == 1:
            content = [ToolUseBlock(id="e2", name="episode_run", input={
                "kind": "search", "goal": "improve", "branches": ["main"],
                "search_tool": "search_anneal"})]
        elif step == 2:
            content = [ToolUseBlock(id="e3", name="episode_run", input={
                "kind": "reading", "goal": "read", "branches": ["main"]})]
        elif step == 3:
            search_id = _find_episode_id(blocks, "search")
            content = [ToolUseBlock(id="e4", name="episode_install_branch", input={
                "episode_id": search_id, "branch": "main", "as_name": "search_result"})]
        elif step == 4:
            content = [ToolUseBlock(id="e5", name="episode_run", input={
                "kind": "compare", "goal": "rank",
                "branches": ["main", "search_result"]})]
        else:
            content = [
                TextBlock(text="search_result wins."),
                ToolUseBlock(id="e6", name="meta_declare_solution", input={
                    "branch": "search_result",
                    "rationale": "compare picked it",
                    "self_confidence": 0.9}),
            ]
        return ModelResponse(content=content, usage=ModelUsage(500, 40, 0))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": "fake-lead", "exchanges": []}


@pytest.fixture
def _episode_fakes():
    survey_good = {"findings": ["alphabet small"],
                   "suspected_modes": [{"mode": "substitution", "confidence": "high"}],
                   "recommended_next": ["search_anneal"]}
    search_good = {"improved": True, "best_branch": "main", "score_summary": {"quad": -3.0},
                   "finalist_session_id": None, "notes": "annealed"}
    reading_good = {"reading_text": "the quick brown", "fragments": [], "holes": [],
                    "overall_confidence": 0.6}
    compare_good = {"ranking": ["search_result", "main"],
                    "verdicts": [{"branch": "search_result", "verdict": "best"}],
                    "winner": "search_result"}
    builders = {
        "episode:survey": lambda p, s, r: EpisodeFake(
            [[ToolUseBlock(id="s0", name="observe_frequency", input={"branch": "main"})],
             [_submit(survey_good)]], role=r),
        "episode:search": lambda p, s, r: EpisodeFake(
            [[ToolUseBlock(id="s0", name="decode_show", input={"branch": "main"})],
             [_submit(search_good)]], role=r),
        "episode:reading": lambda p, s, r: EpisodeFake(
            [[ToolUseBlock(id="s0", name="decode_show", input={"branch": "main"})],
             [_submit(reading_good)]], role=r),
        "episode:compare": lambda p, s, r: EpisodeFake(
            [[ToolUseBlock(id="s0", name="workspace_compare",
                           input={"branch_a": "main", "branch_b": "search_result"})],
             [_submit(compare_good)]], role=r),
    }
    for role, builder in builders.items():
        sessions_mod.register_session_builder(role, builder)
    yield compare_good
    for role in builders:
        sessions_mod._SESSION_BUILDERS.pop(role, None)


def test_scripted_workflow_end_to_end(_episode_fakes):
    from investigation.loop_v3 import run_v3
    raw = "ABC DEF ABC"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    art = run_v3(ct, session=WorkflowLeadSession(), language="en",
                 max_iterations=8, cipher_id="wf")

    # Four "ok" episodes in the ledger.
    ledger = art.episodes
    assert [e["kind"] for e in ledger] == ["survey", "search", "reading", "compare"]
    assert all(e["status"] == "ok" for e in ledger)

    # Four episode:<kind> budget rows.
    for kind in ("survey", "search", "reading", "compare"):
        assert f"episode:{kind}" in art.budget_by_category

    # Compare named the installed branch.
    compare = ledger[-1]
    assert compare["result"]["winner"] == "search_result"
    assert any(b.name == "search_result" for b in art.branches)

    # The reading result is STORED (in the ledger) but NOT applied (A8): no
    # branch carries the reading text as its decode.
    reading = next(e for e in ledger if e["kind"] == "reading")
    assert reading["result"]["reading_text"] == "the quick brown"
    for b in art.branches:
        assert "the quick brown" not in (b.decryption or "")

    # Episode tool calls are merged into the artifact with episode_id set.
    ep_calls = [tc for tc in art.tool_calls if tc.episode_id]
    assert ep_calls and all(tc.episode_id for tc in ep_calls)
    assert art.status == "solved"


# ---------------------------------------------------------------------------
# F4: fork/install reconciliation (spec-author amendment: card fields are
# routed through board.update on install so their content survives)
# ---------------------------------------------------------------------------
def test_f4_install_routes_card_fields_through_board():
    from investigation.loop_v3 import run_v3
    # Seed a state whose "search" episode produced a hypothesis-tagged branch.
    raw = "ABCDEF"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)

    snap = {
        "name": "hypbranch",
        "parent": "main",
        "created_iteration": 1,
        "key": {"0": 1},
        "tags": ["hypothesis", "mode:substitution"],
        "metadata": {"cipher_mode": "substitution", "mode_status": "active",
                     "mode_evidence": "IC in mono range",
                     "hypothesis_notes": "looks mono",
                     "nested": {"depth": {"n": 1}}},
        "word_spans": None,
        "token_order": None,
        "transform_pipeline": None,
    }

    class InstallLead:
        def __init__(self):
            self.model = "fake"; self.provider_name = "openai"
            self.capabilities = SessionCapabilities()
            self._budget = []; self._step = 0

        def send(self, blocks, tools=None, max_tokens=8192):
            self._budget.append(BudgetEntry("lead", "openai", "fake", 1, 1, 0))
            step = self._step; self._step += 1
            if step == 0:
                c = [ToolUseBlock(id="i1", name="episode_install_branch",
                                  input={"episode_id": "ep123456", "branch": "hypbranch",
                                         "as_name": "adopted"})]
            else:
                c = [ToolUseBlock(id="i2", name="decode_show", input={"branch": "main"})]
            return ModelResponse(content=c, usage=ModelUsage(1, 1, 0))

        def usage_entries(self): return list(self._budget)
        def export_transcript(self): return {"provider": "openai", "exchanges": []}

    from investigation.state import InvestigationState as IS
    resume = IS(workspace=Workspace(ct), language="en")
    ledger_entry = {
        "episode_id": "ep123456", "kind": "search", "status": "ok",
        "branch_snapshots": [snap],
        "agenda_additions": [
            {"id": 1, "branch": "hypbranch", "from": "X", "to": "Y",
             "status": "open"},
        ],
    }
    resume.episode_ledger.append(ledger_entry)
    resume.turn = 0

    art = run_v3(ct, session=InstallLead(), language="en", max_iterations=3,
                 cipher_id="f4", resume_state=resume)

    # F4 (amended): the snapshot's card fields were routed through board.update,
    # which re-mirrors them into metadata — the original mode/status/evidence
    # SURVIVE the install (not a degenerate re-adopt).
    installed = next(b for b in art.branches if b.name == "adopted")
    assert installed.metadata.get("cipher_mode") == "substitution"
    assert installed.metadata.get("mode_status") == "active"
    assert installed.metadata.get("mode_evidence") == "IC in mono range"
    assert installed.metadata.get("hypothesis_notes") == "looks mono"
    assert "hypothesis" in installed.tags
    # The board card agrees (single writer — board wrote the metadata mirror).
    board = art.investigation_state["hypothesis_board"]
    card = next(c for c in board["cards"] if c["branch"] == "adopted")
    assert card["cipher_mode"] == "substitution"
    assert card["mode_status"] == "active"
    assert card["mode_evidence"] == "IC in mono range"
    # Fix 8: the install deep-copied the snapshot — the ledger entry's nested
    # metadata is untouched by the install/mirror writes.
    assert ledger_entry["branch_snapshots"][0]["metadata"]["nested"] == {
        "depth": {"n": 1}}
    # Fix 2 (lead side): the episode's agenda_additions merged into
    # state.repair_agenda with the branch remapped to the installed name.
    merged = [item for item in art.repair_agenda if item.get("from") == "X"]
    assert merged and merged[0]["branch"] == "adopted"


def test_setup_failure_is_structured_never_a_lead_crash():
    """Fix 1 (blocking): a raising session builder — or any episode-setup
    exception — becomes episode_failed(runner_error), never a lead crash."""
    from investigation.loop_v3 import run_v3

    def boom_builder(provider, system, role):
        raise RuntimeError("no session for you")

    sessions_mod.register_session_builder("episode:survey", boom_builder)
    try:
        # Direct run_episode: structured failure + ledger entry.
        state = _simple_state()
        spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})
        res = run_episode(spec, state, provider=None)
        assert res.status == "episode_failed"
        assert res.failure_reason == "runner_error"
        assert "no session for you" in (res.raw_text or "")
        assert state.episode_ledger[-1]["failure_reason"] == "runner_error"

        # Lead-side: the episode_run tool result is structured JSON and the
        # lead run continues to completion.
        class Lead:
            def __init__(self):
                self.model = "fake"; self.provider_name = "openai"
                self.capabilities = SessionCapabilities()
                self._budget = []; self._step = 0

            def send(self, blocks, tools=None, max_tokens=8192):
                self._budget.append(BudgetEntry("lead", "openai", "fake", 1, 1, 0))
                step = self._step; self._step += 1
                if step == 0:
                    c = [ToolUseBlock(id="e1", name="episode_run", input={
                        "kind": "survey", "goal": "diag", "branches": ["main"]})]
                else:
                    c = [TextBlock(text="giving up")]
                return ModelResponse(content=c, usage=ModelUsage(1, 1, 0))

            def usage_entries(self): return list(self._budget)
            def export_transcript(self): return {"provider": "openai", "exchanges": []}

        raw = "ABCDEF"
        alpha = Alphabet.from_text(raw, ignore_chars=set())
        ct = CipherText(raw=raw, alphabet=alpha, separator=None)
        art = run_v3(ct, session=Lead(), language="en", max_iterations=2,
                     cipher_id="setupfail")
        # No crash; the episode result rode back as a structured tool result.
        ep_tc = next(m for m in art.messages if m["role"] == "user"
                     and any(b.get("type") == "tool_result"
                             for b in m["content"]))
        payload = json.loads(ep_tc["content"][0]["content"])
        assert payload["status"] == "episode_failed"
        assert payload["failure_reason"] == "runner_error"
        assert art.episodes[-1]["failure_reason"] == "runner_error"
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:survey", None)


def test_agenda_additions_exported_from_episode(monkeypatch):
    """Fix 2: the episode-local repair agenda rides out in the packet."""
    from agent.tools_v2 import WorkspaceToolExecutor

    def fake_decode_show(self, args):
        # A handler that (like the real repair planners) appends an agenda item.
        self.repair_agenda.append({
            "id": self._next_repair_agenda_id, "branch": args.get("branch"),
            "from": "Q", "to": "K", "status": "open",
        })
        self._next_repair_agenda_id += 1
        return {"status": "ok"}

    monkeypatch.setattr(WorkspaceToolExecutor, "_tool_decode_show",
                        fake_decode_show)
    state = _simple_state()
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})
    good = {"findings": ["f"], "suspected_modes": [], "recommended_next": []}
    scripts = [
        [ToolUseBlock(id="t1", name="decode_show", input={"branch": "main"})],
        [_submit(good, tid="b")],
    ]
    res = run_episode(spec, state, session=EpisodeFake(scripts))
    assert res.status == "ok"
    assert res.agenda_additions and res.agenda_additions[0]["from"] == "Q"
    # The ledger dict carries it for the lead's install-time merge.
    assert state.episode_ledger[-1]["agenda_additions"][0]["from"] == "Q"
    # The lead's own agenda is untouched until an install merges it.
    assert state.repair_agenda == []


def test_episode_fakes_assert_contract_and_no_lead_bleed():
    """Fix 7: the worker's system prompt carries the contract + toolset, and
    the first send's blocks contain no lead-context bleed-through."""
    captured: dict = {}

    class AssertingFake(EpisodeFake):
        def __init__(self, system, role):
            super().__init__([[
                _submit({"findings": [], "suspected_modes": [],
                         "recommended_next": []})]], role=role)
            captured["system"] = system

    sessions_mod.register_session_builder(
        "episode:survey", lambda p, s, r: AssertingFake(s, r))
    try:
        state = _simple_state()
        spec = EpisodeSpec("survey", "diagnose it", inputs={"branches": ["main"]})
        res = run_episode(spec, state, provider=None)
        assert res.status == "ok"
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:survey", None)

    system = captured["system"]
    # Contract present.
    assert "SURVEY worker" in system
    assert "episode_submit_result" in system
    # Toolset listed.
    for name in spec.toolset:
        assert name in system


def test_episode_context_has_no_lead_transcript_bleed():
    state = _simple_state()
    # Give the lead some transcript/evidence that must NOT reach a worker.
    state.external_context = "SECRET-LEAD-CONTEXT"
    state.add_evidence("turn_summary", 1, "SECRET-EVIDENCE")
    state.recent_exchanges = [
        {"role": "assistant", "content": [{"type": "text",
                                           "text": "SECRET-LEAD-TRANSCRIPT"}]},
    ]
    spec = EpisodeSpec("survey", "diagnose", inputs={"branches": ["main"]})
    good = {"findings": [], "suspected_modes": [], "recommended_next": []}
    fake = EpisodeFake([[_submit(good)]])
    res = run_episode(spec, state, session=fake)
    assert res.status == "ok"
    all_blocks = json.dumps(fake.blocks_seen, default=str)
    assert "SECRET-LEAD-CONTEXT" not in all_blocks
    assert "SECRET-EVIDENCE" not in all_blocks
    assert "SECRET-LEAD-TRANSCRIPT" not in all_blocks
    assert "## Investigation state" not in all_blocks


def test_stale_cards_filtered_after_branch_delete():
    """Fix 5: cards for deleted branches drop out of the rendered board but
    stay in the serialized board."""
    from investigation.context import _render_hypothesis_board
    state = _simple_state()
    ws = state.workspace
    ws.fork("h1", from_branch="main")
    state.hypothesis_board.create(
        ws, "h1", cipher_mode="substitution", mode_confidence="high",
        mode_status="active", mode_evidence="e", evidence_source="agent_inference",
    )
    assert any(c["branch"] == "h1" for c in state.hypothesis_cards())
    ws.delete("h1")
    # Filtered from the render-facing view...
    assert not any(c["branch"] == "h1" for c in state.hypothesis_cards())
    assert "h1" not in _render_hypothesis_board(state)
    # ...but preserved in the serialized board (full trail).
    assert any(c["branch"] == "h1"
               for c in state.hypothesis_board.to_dict()["cards"])


def test_install_deep_copies_snapshot_no_ledger_aliasing():
    """Fix 8: mutating the installed live branch must not mutate the episode
    ledger's stored snapshot (nested metadata / pipeline aliasing)."""
    from investigation.loop_v3 import run_v3
    raw = "ABCDEF"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)

    snap = {
        "name": "b1", "parent": "main", "created_iteration": 1,
        "key": {}, "tags": [],
        "metadata": {"notes": {"inner": ["a"]}},
        "word_spans": None, "token_order": None,
        "transform_pipeline": {"steps": [{"op": "reverse"}]},
    }

    class Lead:
        def __init__(self):
            self.model = "fake"; self.provider_name = "openai"
            self.capabilities = SessionCapabilities()
            self._budget = []; self._step = 0

        def send(self, blocks, tools=None, max_tokens=8192):
            self._budget.append(BudgetEntry("lead", "openai", "fake", 1, 1, 0))
            step = self._step; self._step += 1
            if step == 0:
                c = [ToolUseBlock(id="i1", name="episode_install_branch",
                                  input={"episode_id": "epA", "branch": "b1",
                                         "as_name": "inst"})]
            else:
                c = [TextBlock(text="done")]
            return ModelResponse(content=c, usage=ModelUsage(1, 1, 0))

        def usage_entries(self): return list(self._budget)
        def export_transcript(self): return {"provider": "openai", "exchanges": []}

    resume = InvestigationState(workspace=Workspace(ct), language="en")
    entry = {"episode_id": "epA", "kind": "search", "status": "ok",
             "branch_snapshots": [snap]}
    resume.episode_ledger.append(entry)
    resume.turn = 0

    run_v3(ct, session=Lead(), language="en", max_iterations=2,
           cipher_id="alias", resume_state=resume)
    ws_branch = resume.workspace.get_branch("inst")
    # Mutate the live installed branch deeply.
    ws_branch.metadata["notes"]["inner"].append("hacked")
    ws_branch.transform_pipeline["steps"].append({"op": "hack"})
    # The ledger snapshot is untouched.
    assert entry["branch_snapshots"][0]["metadata"]["notes"] == {"inner": ["a"]}
    assert entry["branch_snapshots"][0]["transform_pipeline"] == {
        "steps": [{"op": "reverse"}]}
