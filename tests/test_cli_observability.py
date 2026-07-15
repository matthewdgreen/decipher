"""Tests for the CLI observability revamp (docs/specs/cli_observability_spec.md).

Covers: the narrate renderer's transcript structure + cost ticker + declaration
block; make_agent_renderer registration/verbose; the automated on_step callback;
episode on_event emission; v3 end-of-turn workspace_snapshot + budget_update
emission and episode-event forwarding into the lead stream; and the cmd_crack
"no plain prints while a renderer is active" ordering regression (decision 6).

Structural markers are asserted (not exact spacing).
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.display import make_agent_renderer
from agent.model_provider import ModelResponse, ModelUsage, TextBlock, ToolUseBlock
from agent.narrate import NarrateAgentRenderer, _compact_args, _format_elapsed
from investigation import sessions as sessions_mod
from investigation.episodes import EpisodeSpec, run_episode
from investigation.loop_v3 import run_v3
from investigation.sessions import SessionCapabilities
from investigation.state import BudgetEntry, InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


# ---------------------------------------------------------------------------
# Narrate renderer
# ---------------------------------------------------------------------------
def _scripted_narrate(verbose: bool = False) -> tuple[NarrateAgentRenderer, io.StringIO]:
    stream = io.StringIO()
    return NarrateAgentRenderer(stream, verbose=verbose), stream


def test_narrate_renders_full_transcript_shape():
    r, stream = _scripted_narrate()
    r.start_test(
        "borg_0109v", "A borg page", model="gpt-5.5", max_iterations=9,
        language="la", source="borg", agent_loop="v3",
    )
    r.event("preflight_result", {
        "solver": "zenith_native", "status": "solved", "elapsed_seconds": 12.4,
    })
    r.event("iteration_start", {"iteration": 1})
    r.event("tool_start", {"tool": "observe_frequency", "arguments": {"branch": "main"}})
    r.event("tool_call", {"tool": "observe_frequency",
                          "result_summary": {"branch": "main", "status": "ok"}})
    # The snapshot advances the cost ticker for subsequent lines.
    r.event("workspace_snapshot", {
        "branch": "main", "decryption": "AB", "scores": {},
        "total_tokens": 41000, "estimated_cost_usd": 0.14,
    })
    r.event("tool_start", {"tool": "act_set_mapping", "arguments": {"from": "S12", "to": "E"}})
    r.event("tool_call", {"tool": "act_set_mapping",
                          "result_summary": {"from": "S12", "to": "E", "status": "ok"}})
    r.event("declared_solution", {
        "branch": "main", "confidence": "high",
        "attestation": {"coherence": 8, "reader_accepts": True, "anomalies": ["x"]},
    })
    r.finish(SimpleNamespace(
        status="solved", char_accuracy=0.957, word_accuracy=0.731,
        iterations_used=9, estimated_cost_usd=2.31, elapsed_seconds=252.0,
        artifact_path="artifacts/borg_0109v/abc.json", error_message="",
        final_summary="",
    ))
    out = stream.getvalue()
    # Header
    assert "▶ borg_0109v" in out
    assert "loop=v3" in out
    assert "(borg, la)" in out
    # Preflight
    assert "preflight" in out and "zenith_native" in out
    # One indexed line per lead tool call
    assert "1 │ observe_frequency" in out
    assert "2 │ act_set_mapping" in out
    # Cost ticker reflects the last-known snapshot on the 2nd tool line
    assert "$0.14" in out
    # Declaration + attestation block
    assert "✓ DECLARED solution on branch main" in out
    assert "attestation:" in out and "coherence 8/10" in out
    # Result block + artifact path
    assert "── result ──" in out
    assert "status=solved" in out and "char=95.7%" in out
    assert "artifact: artifacts/borg_0109v/abc.json" in out


def test_narrate_nested_episode_block_orders_parent_above_children():
    # Event order mirrors the real v3 dispatch: tool_start(episode_run) fires
    # FIRST, then the forwarded episode_* children, then episode_complete, then
    # the lead tool_call. The renderer must print the numbered parent line at
    # tool_start so the ↳ children nest UNDER it (F-1 regression).
    r, stream = _scripted_narrate()
    r.event("iteration_start", {"iteration": 2})
    r.event("tool_start", {"tool": "episode_run",
                           "arguments": {"kind": "search", "branch": "main"}})
    r.event("episode_turn_start", {"episode_id": "abcdef", "kind": "search", "turn": 1})
    r.event("episode_tool_call", {"episode_id": "abcdef", "kind": "search",
                                  "turn": 1, "tool": "search_hill_climb",
                                  "arguments": {"branch": "main"}})
    r.event("episode_submit", {"episode_id": "abcdef", "kind": "search",
                               "turn": 2, "accepted": True})
    r.event("episode_complete", {"episode_id": "abcdef", "kind": "search",
                                 "status": "ok", "calls": 3, "spend_usd": 0.21})
    r.event("tool_call", {"tool": "episode_run", "result_summary": {}})
    out = stream.getvalue()
    lines = out.splitlines()

    # The numbered parent line exists and precedes EVERY ↳ child line (ORDER,
    # not just presence — the previous test missed the inversion).
    parent_idx = next(i for i, ln in enumerate(lines)
                      if "│ episode_run(" in ln and ln.lstrip().startswith("1 "))
    child_indices = [i for i, ln in enumerate(lines) if "↳" in ln]
    assert child_indices, "expected nested ↳ child lines"
    assert parent_idx < min(child_indices), (
        "parent episode_run line must come ABOVE its ↳ children"
    )
    # The parent line is unnumbered-result (children carry the outcome).
    assert "search_hill_climb" in out
    assert "episode_submit → ok" in out
    assert "calls=3" in out and "$0.21" in out


def test_narrate_finish_suppresses_accuracy_without_ground_truth():
    # F-4: a crack-style result (has_ground_truth=False) must NOT show the
    # misleading char=0.0% word=0.0% line; benchmark results still do.
    r, stream = _scripted_narrate()
    r.finish(SimpleNamespace(
        status="solved", char_accuracy=0.0, word_accuracy=0.0,
        has_ground_truth=False, iterations_used=3, estimated_cost_usd=0.5,
        elapsed_seconds=12.0, artifact_path="a.json", error_message="",
        final_summary="",
    ))
    out = stream.getvalue()
    assert "status=solved" in out
    assert "char=" not in out and "word=" not in out
    assert "$0.50" in out and "artifact: a.json" in out


def test_narrate_verbose_shows_full_args_nonverbose_truncates():
    args = {"branch": "main", "a": 1, "b": 2, "c": 3, "d": 4}
    assert "d=4" in _compact_args(args, verbose=True)
    non = _compact_args(args, verbose=False)
    assert "d=4" not in non and non.endswith("…")


def test_format_elapsed_minutes_and_seconds():
    assert _format_elapsed(9.4) == "9.4s"
    assert _format_elapsed(252) == "4m12s"


def test_make_agent_renderer_registers_narrate_and_verbose():
    r = make_agent_renderer("narrate", stream=io.StringIO(), verbose=True)
    assert isinstance(r, NarrateAgentRenderer)
    assert r.verbose is True
    assert make_agent_renderer("off") is None


# ---------------------------------------------------------------------------
# Automated on_step callback (F8)
# ---------------------------------------------------------------------------
def test_step_list_fires_on_step_with_elapsed():
    from automated.runner import _StepList

    calls: list[tuple] = []
    clock = iter([0.0, 1.0, 3.5])
    sl = _StepList(on_step=lambda n, s, e: calls.append((n, s, e)),
                   clock=lambda: next(clock))
    sl.append({"name": "route_automated_solver", "status": "ok"})
    sl.append({"name": "screen_transform_candidates"})
    assert calls[0][0] == "route_automated_solver" and calls[0][1] == "ok"
    assert abs(calls[0][2] - 1.0) < 1e-9
    assert calls[1][0] == "screen_transform_candidates" and calls[1][1] is None
    assert abs(calls[1][2] - 2.5) < 1e-9
    # Still a real list.
    assert [s["name"] for s in sl] == [
        "route_automated_solver", "screen_transform_candidates",
    ]


def test_step_list_no_callback_is_plain_list():
    from automated.runner import _StepList

    sl = _StepList()
    sl.append({"name": "x"})
    assert list(sl) == [{"name": "x"}]


# ---------------------------------------------------------------------------
# Episode on_event emission (F6)
# ---------------------------------------------------------------------------
class _EpisodeFake:
    def __init__(self, scripts, *, model="fake-luna", provider="openai",
                 category="episode:survey"):
        self.model = model
        self.provider_name = provider
        self._category = category
        self.capabilities = SessionCapabilities()
        self._scripts = list(scripts)
        self._n = 0
        self._budget: list[BudgetEntry] = []

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry(self._category, self.provider_name,
                                        self.model, 100, 20, 5))
        content = self._scripts[min(self._n, len(self._scripts) - 1)]
        self._n += 1
        return ModelResponse(content=content, usage=ModelUsage(100, 20, 5))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": self.provider_name, "model": self.model, "exchanges": []}


def _simple_state(raw="ABCDEFGHIJKL") -> InvestigationState:
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    return InvestigationState(workspace=Workspace(ct), language="en")


def test_run_episode_forwards_on_event():
    state = _simple_state()
    events: list[tuple] = []
    survey_good = {"findings": [], "suspected_modes": [], "recommended_next": []}
    scripts = [
        [ToolUseBlock(id="a", name="decode_show", input={"branch": "main"})],
        [ToolUseBlock(id="s", name="episode_submit_result",
                      input={"result": survey_good, "summary": "ok"})],
    ]
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})
    run_episode(spec, state, session=_EpisodeFake(scripts),
                on_event=lambda ev, pl: events.append((ev, pl)))

    names = [ev for ev, _ in events]
    assert "episode_turn_start" in names
    assert "episode_tool_call" in names
    assert "episode_submit" in names
    tc = next(pl for ev, pl in events if ev == "episode_tool_call")
    assert tc["kind"] == "survey" and "episode_id" in tc
    assert tc["turn"] == 1 and tc["tool"] == "decode_show"
    sub = next(pl for ev, pl in events if ev == "episode_submit")
    assert sub["accepted"] is True


def test_compact_episode_args_bounds_heavy_payloads():
    # F-3: forwarded episode_tool_call args are compacted so the stored event
    # does not duplicate arg-heavy payloads (mappings dicts, run_python code).
    from investigation.episodes import _compact_episode_args

    big = {"branch": "main", "mappings": {i: i for i in range(30)},
           "code": "x" * 200, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    out = _compact_episode_args(big)
    assert out["mappings"] == "<dict[30]>"
    assert out["code"].endswith("…") and len(out["code"]) <= 41
    assert len([k for k in out if k != "…"]) <= 6  # handful of keys
    assert "…" in out  # overflow marker


def test_run_episode_forwarded_tool_args_are_compacted():
    # An episode tool call with a heavy args dict must forward a COMPACT summary.
    state = _simple_state()
    events: list[tuple] = []
    survey_good = {"findings": [], "suspected_modes": [], "recommended_next": []}
    heavy = {"branch": "main", "mappings": {i: i for i in range(40)}}
    scripts = [
        [ToolUseBlock(id="a", name="act_bulk_set", input=heavy)],
        [ToolUseBlock(id="s", name="episode_submit_result",
                      input={"result": survey_good, "summary": "ok"})],
    ]
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})
    run_episode(spec, state, session=_EpisodeFake(scripts),
                on_event=lambda ev, pl: events.append((ev, pl)))
    tc = next(pl for ev, pl in events if ev == "episode_tool_call")
    assert tc["arguments"]["mappings"] == "<dict[40]>"


def test_run_episode_without_on_event_is_silent():
    state = _simple_state()
    survey_good = {"findings": [], "suspected_modes": [], "recommended_next": []}
    scripts = [[ToolUseBlock(id="s", name="episode_submit_result",
                             input={"result": survey_good, "summary": "ok"})]]
    spec = EpisodeSpec("survey", "g", inputs={"branches": ["main"]})
    # No on_event -> no crash, normal result.
    res = run_episode(spec, state, session=_EpisodeFake(scripts))
    assert res.status == "ok"


# ---------------------------------------------------------------------------
# v3 end-of-turn workspace_snapshot + budget_update + episode forwarding (F5/F6)
# ---------------------------------------------------------------------------
def _caesar(text: str, shift: int) -> str:
    return "".join(
        chr((ord(c) - 65 + shift) % 26 + 65) if c.isalpha() else c for c in text
    )


def _caesar_cipher(plaintext: str, shift: int = 3):
    raw = _caesar(plaintext, shift)
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    return CipherText(raw=raw, alphabet=alpha, separator=" "), alpha


class _ScriptedSession:
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


def test_run_v3_emits_end_of_turn_snapshot_and_budget():
    ct, _alpha = _caesar_cipher("THE DOG")
    events: list[tuple] = []
    scripts = [[ToolUseBlock(id="t1", name="decode_show", input={"branch": "main"})]]
    session = _ScriptedSession(scripts)
    run_v3(ct, session=session, language="en", max_iterations=1,
           cipher_id="v3_snap", on_event=lambda ev, pl: events.append((ev, pl)))

    names = [ev for ev, _ in events]
    assert "workspace_snapshot" in names
    assert "budget_update" in names
    snap = next(pl for ev, pl in events if ev == "workspace_snapshot")
    assert snap["branch"] == "main"
    assert "decryption" in snap
    assert snap["total_tokens"] == 1050  # one lead send: 1000 in + 50 out
    bud = next(pl for ev, pl in events if ev == "budget_update")
    assert "budget_by_category" in bud
    assert "lead" in bud["budget_by_category"]
    assert bud["total_cost_usd"] >= 0.0
    assert bud["total_tokens"] == 1050


class _SurveyWorkerFake:
    def __init__(self, provider=None, system="", role="episode:survey"):
        self.model = "fake-luna"
        self.provider_name = "openai"
        self.capabilities = SessionCapabilities()
        self._budget: list[BudgetEntry] = []
        self._n = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("episode:survey", "openai",
                                        "fake-luna", 50, 10, 0))
        self._n += 1
        if self._n == 1:
            return ModelResponse(
                content=[ToolUseBlock(id="d1", name="decode_show",
                                      input={"branch": "main"})],
                usage=ModelUsage(50, 10, 0))
        survey_good = {"findings": [], "suspected_modes": [], "recommended_next": []}
        return ModelResponse(
            content=[ToolUseBlock(id="s1", name="episode_submit_result",
                                  input={"result": survey_good, "summary": "ok"})],
            usage=ModelUsage(50, 10, 0))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": self.model, "exchanges": []}


@pytest.fixture
def survey_fake():
    sessions_mod.register_session_builder("episode:survey", _SurveyWorkerFake)
    yield
    sessions_mod._SESSION_BUILDERS.pop("episode:survey", None)


def test_run_v3_forwards_episode_events_into_lead_stream(survey_fake):
    ct, _alpha = _caesar_cipher("THE DOG")
    events: list[tuple] = []
    scripts = [[ToolUseBlock(id="e1", name="episode_run",
                             input={"kind": "survey", "goal": "survey",
                                    "branches": ["main"]})]]
    session = _ScriptedSession(scripts)
    run_v3(ct, session=session, language="en", max_iterations=1,
           cipher_id="v3_ep", on_event=lambda ev, pl: events.append((ev, pl)))

    names = [ev for ev, _ in events]
    assert "episode_tool_call" in names
    assert "episode_submit" in names
    assert "episode_complete" in names
    # episode_complete now carries per-episode spend for the ticker (F6c).
    ec = next(pl for ev, pl in events if ev == "episode_complete")
    assert "spend_usd" in ec and ec["kind"] == "survey"
    # Forwarded episode events retain the nesting identity.
    tc = next(pl for ev, pl in events if ev == "episode_tool_call")
    assert tc["kind"] == "survey" and "episode_id" in tc


# ---------------------------------------------------------------------------
# cmd_crack: no plain prints while a renderer is active (decision 6)
# ---------------------------------------------------------------------------
def test_cmd_crack_plain_prints_after_renderer_finish(tmp_path, monkeypatch, capsys):
    import cli
    import agent.display as disp
    import agent.loop_v2 as lv2
    import agent.final_summary as fsmod

    class _FakeBranch:
        name = "main"
        decryption = "HELLO WORLD"

    class _FakeArtifact:
        def __init__(self):
            self.status = "solved"
            self.solution = None
            self.tool_calls = []
            self.branches = [_FakeBranch()]
            self.run_id = "run123"
            self.finished_at = 10.0
            self.started_at = 0.0
            self.total_input_tokens = 100
            self.total_output_tokens = 50
            self.estimated_cost_usd = 1.23
            self.error_message = ""
            self.final_summary = ""

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("{}", encoding="utf-8")

    class _RecRenderer:
        def __init__(self):
            self.finished = False

        def start_test(self, *a, **k):
            pass

        def event(self, *a, **k):
            pass

        def finish(self, result):
            self.finished = True
            print("<<RENDERER_FINISH>>")

    rec = _RecRenderer()

    monkeypatch.setattr(cli, "_preflight_model_check", lambda args: None)
    monkeypatch.setattr(cli, "_resolve_provider_and_model", lambda args: ("openai", "fake"))
    monkeypatch.setattr(cli, "_make_agent_provider", lambda args: SimpleNamespace(model="fake"))
    monkeypatch.setattr(cli, "_read_external_context", lambda args: None)
    monkeypatch.setattr(cli, "_resolve_system_prompt_style", lambda args: "full")
    monkeypatch.setattr(cli, "_maybe_write_artifact_analysis", lambda path, args: None)
    monkeypatch.setattr(cli, "_resolve_agent_display", lambda args: "narrate")
    monkeypatch.setattr(disp, "make_agent_renderer", lambda mode, **kw: rec)
    monkeypatch.setattr(lv2, "run_v2", lambda **kwargs: _FakeArtifact())
    monkeypatch.setattr(fsmod, "build_final_summary", lambda *a, **k: "a summary")

    inp = tmp_path / "cipher.txt"
    inp.write_text("A B C\n", encoding="utf-8")
    args = SimpleNamespace(
        file=str(inp), canonical=False, cipher_id="cli",
        artifact_dir=str(tmp_path / "art"), language="en", agentic=True,
        agent_loop="v2", max_iterations=3, verbose=False, display="narrate",
        no_automated_preflight=True, analyze=False, model="fake", provider=None,
    )
    cli.cmd_crack(args)

    out = capsys.readouterr().out
    assert rec.finished is True
    assert "<<RENDERER_FINISH>>" in out
    assert "Status: solved" in out
    # decision 6: the plain summary prints occur AFTER renderer.finish.
    assert out.index("<<RENDERER_FINISH>>") < out.index("Status: solved")
    assert out.index("<<RENDERER_FINISH>>") < out.index("Final decryption")


class _FakeCrackBranch:
    name = "main"
    decryption = "HELLO WORLD"


class _FakeCrackArtifact:
    def __init__(self):
        self.status = "solved"
        self.solution = None
        self.tool_calls = []
        self.branches = [_FakeCrackBranch()]
        self.run_id = "run123"
        self.finished_at = 10.0
        self.started_at = 0.0
        self.total_input_tokens = 100
        self.total_output_tokens = 50
        self.estimated_cost_usd = 1.23
        self.error_message = ""
        self.final_summary = ""

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("{}", encoding="utf-8")


def test_cmd_crack_jsonl_stdout_is_pure_json(tmp_path, monkeypatch, capsys):
    """F-2: in --display jsonl agentic crack, stdout is machine-readable ONLY.

    Uses the REAL JsonlAgentRenderer (not a fake) so stdout carries its JSON
    events; every human print (Alphabet header, Status/Final summary) must be
    routed to stderr.
    """
    import json as _json

    import cli
    import agent.loop_v2 as lv2
    import agent.final_summary as fsmod

    monkeypatch.setattr(cli, "_preflight_model_check", lambda args: None)
    monkeypatch.setattr(cli, "_resolve_provider_and_model", lambda args: ("openai", "fake"))
    monkeypatch.setattr(cli, "_make_agent_provider", lambda args: SimpleNamespace(model="fake"))
    monkeypatch.setattr(cli, "_read_external_context", lambda args: None)
    monkeypatch.setattr(cli, "_resolve_system_prompt_style", lambda args: "full")
    monkeypatch.setattr(cli, "_maybe_write_artifact_analysis", lambda path, args: None)
    monkeypatch.setattr(lv2, "run_v2", lambda **kwargs: _FakeCrackArtifact())
    monkeypatch.setattr(fsmod, "build_final_summary", lambda *a, **k: "a summary")
    # NOTE: make_agent_renderer + _resolve_agent_display are NOT patched — the
    # real JsonlAgentRenderer must run so its stdout output is exercised.

    inp = tmp_path / "cipher.txt"
    inp.write_text("A B C\n", encoding="utf-8")
    args = SimpleNamespace(
        file=str(inp), canonical=False, cipher_id="cli",
        artifact_dir=str(tmp_path / "art"), language="en", agentic=True,
        agent_loop="v2", max_iterations=3, verbose=False, display="jsonl",
        no_automated_preflight=True, analyze=False, model="fake", provider=None,
    )
    cli.cmd_crack(args)

    captured = capsys.readouterr()
    stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert stdout_lines, "expected jsonl events on stdout"
    for ln in stdout_lines:
        _json.loads(ln)  # every stdout line must be valid JSON
    events = [_json.loads(ln)["event"] for ln in stdout_lines]
    assert "test_start" in events and "test_finish" in events
    # The human-readable header + summary are routed to stderr.
    assert "Alphabet:" in captured.err
    assert "Status: solved" in captured.err
    assert "Alphabet:" not in captured.out


# ---------------------------------------------------------------------------
# cmd_benchmark: artifact path + result block + report table always printed
# ---------------------------------------------------------------------------
def _fake_bench_result(test_id, status):
    return SimpleNamespace(
        test_id=test_id, status=status, char_accuracy=0.95, word_accuracy=0.73,
        self_confidence=0.9, iterations_used=5, elapsed_seconds=1.2,
        artifact_path=f"/tmp/art/{test_id}/run.json", error_message="",
        final_decryption="HELLO", estimated_cost_usd=2.31,
    )


def _bench_args(tmp_path, *, agentic, display):
    return SimpleNamespace(
        benchmark_path="/does/not/matter", multipage_group=None, analyze=False,
        agentic=agentic, split=None, source="borg", track=None, test_id=None,
        limit=None, language="la", verbose=False,
        artifact_dir=str(tmp_path / "art"), max_iterations=5,
        homophonic_budget="full", homophonic_refinement="none",
        legacy_homophonic=False, model_variant=None, transform_search="off",
        transform_search_profile="broad",
        transform_search_max_generated_candidates=None,
        transform_promote_artifact=None, transform_promote_candidate_id=[],
        transform_promote_top_n=None, no_automated_preflight=True,
        benchmark_context="max", agent_loop="v2", display=display,
    )


def _patch_bench(monkeypatch):
    import cli
    import benchmark.loader as loader_mod
    import automated.runner as auto_mod
    import benchmark.runner_v2 as v2_mod

    class _FakeTest:
        def __init__(self, tid):
            self.test_id = tid
            self.description = "a test"

    class _FakeLoader:
        def __init__(self, path):
            pass

        def load_tests(self, split, track=None, source=None):
            return [_FakeTest("borg_x")]

        def load_test_data(self, test):
            return SimpleNamespace(test=test)

    class _FakeAuto:
        def __init__(self, **k):
            pass

        def run_test(self, td, on_step=None):
            if on_step is not None:
                on_step("route_automated_solver", "ok", 1.5)
            return _fake_bench_result(td.test.test_id, "completed")

    class _FakeV2:
        def __init__(self, **k):
            pass

        def run_test(self, td):
            return _fake_bench_result(td.test.test_id, "solved")

    monkeypatch.setattr(loader_mod, "BenchmarkLoader", _FakeLoader)
    monkeypatch.setattr(auto_mod, "AutomatedBenchmarkRunner", _FakeAuto)
    monkeypatch.setattr(v2_mod, "BenchmarkRunnerV2", _FakeV2)
    monkeypatch.setattr(cli, "_preflight_model_check", lambda args: None)
    monkeypatch.setattr(cli, "_resolve_provider_and_model", lambda args: ("openai", "fake"))
    monkeypatch.setattr(cli, "_make_agent_provider", lambda args: SimpleNamespace(model="fake"))
    monkeypatch.setattr(cli, "_read_external_context", lambda args: None)
    monkeypatch.setattr(cli, "_resolve_system_prompt_style", lambda args: "full")
    monkeypatch.setattr(cli, "_maybe_write_artifact_analysis", lambda path, args: None)


def test_cmd_benchmark_automated_prints_result_block_artifacts_and_report(tmp_path, monkeypatch, capsys):
    import cli

    _patch_bench(monkeypatch)
    cli.cmd_benchmark(_bench_args(tmp_path, agentic=False, display="narrate"))
    out = capsys.readouterr().out
    # Per-test result block + artifact path
    assert "Status: completed" in out
    assert "Artifact: /tmp/art/borg_x/run.json" in out
    # F8 step narration on the automated path
    assert "· step: route_automated_solver" in out
    # Report table (format_report) + consolidated Artifacts list (decision 4/5)
    assert "BENCHMARK RESULTS" in out
    assert "Agent%" not in out
    assert "Artifacts:" in out
    assert "Run dir root:" in out


@pytest.mark.parametrize("display", ["narrate", "pretty"])
def test_cmd_benchmark_agentic_prints_artifact_path_to_stdout(tmp_path, monkeypatch, capsys, display):
    import cli

    _patch_bench(monkeypatch)
    cli.cmd_benchmark(_bench_args(tmp_path, agentic=True, display=display))
    out = capsys.readouterr().out
    assert "Artifact: /tmp/art/borg_x/run.json" in out
    assert "BENCHMARK RESULTS" in out
    assert "Artifacts:" in out


def test_cmd_benchmark_agentic_jsonl_routes_human_prints_to_stderr(tmp_path, monkeypatch, capsys):
    import cli

    _patch_bench(monkeypatch)
    cli.cmd_benchmark(_bench_args(tmp_path, agentic=True, display="jsonl"))
    captured = capsys.readouterr()
    # F1: stdout stays machine-clean; the human result block/report go to stderr.
    assert "Artifact:" not in captured.out
    assert "BENCHMARK RESULTS" not in captured.out
    assert "Artifact: /tmp/art/borg_x/run.json" in captured.err
    assert "BENCHMARK RESULTS" in captured.err


def test_narrate_renders_decode_on_change_only():
    # Live progress: a workspace_snapshot whose decode CHANGED prints one
    # cyan decode line; an identical snapshot prints nothing new.
    r, stream = _scripted_narrate()
    r.event("workspace_snapshot", {
        "branch": "main", "total_tokens": 100, "estimated_cost_usd": 0.01,
        "decryption_preview": "HELLO WORLD THIS IS PROGRESS",
    })
    first = stream.getvalue()
    assert "decode [main]" in first
    assert "HELLO WORLD THIS IS PROGRESS" in first
    # unchanged decode -> no second decode line
    r.event("workspace_snapshot", {
        "branch": "main", "total_tokens": 200, "estimated_cost_usd": 0.02,
        "decryption_preview": "HELLO WORLD THIS IS PROGRESS",
    })
    assert stream.getvalue().count("decode [main]") == 1
    # changed decode -> a new line
    r.event("workspace_snapshot", {
        "branch": "main", "total_tokens": 300, "estimated_cost_usd": 0.03,
        "decryption_preview": "HELLO WORLD THIS IS BETTER",
    })
    out = stream.getvalue()
    assert out.count("decode [main]") == 2
    assert "THIS IS BETTER" in out


def test_narrate_finish_prints_final_plaintext():
    r, stream = _scripted_narrate()
    r.finish(SimpleNamespace(
        status="solved", char_accuracy=0.9, word_accuracy=0.7,
        iterations_used=5, estimated_cost_usd=1.0, elapsed_seconds=30.0,
        artifact_path="b.json", error_message="", final_summary="",
        final_decryption="THE FINAL RECOVERED PLAINTEXT OF THE RUN",
    ))
    out = stream.getvalue()
    assert "plaintext:" in out
    assert "THE FINAL RECOVERED PLAINTEXT OF THE RUN" in out
    # long text truncates at non-verbose
    r2, s2 = _scripted_narrate()
    r2.finish(SimpleNamespace(
        status="solved", char_accuracy=0.9, word_accuracy=0.7,
        iterations_used=5, estimated_cost_usd=1.0, elapsed_seconds=30.0,
        artifact_path="b.json", error_message="", final_summary="",
        final_decryption="X" * 400,
    ))
    assert "…" in s2.getvalue()
    # absent final_decryption (crack path) -> no plaintext block
    r3, s3 = _scripted_narrate()
    r3.finish(SimpleNamespace(
        status="solved", char_accuracy=0.0, word_accuracy=0.0,
        has_ground_truth=False, iterations_used=3, estimated_cost_usd=0.5,
        elapsed_seconds=12.0, artifact_path="a.json", error_message="",
        final_summary="",
    ))
    assert "plaintext:" not in s3.getvalue()
