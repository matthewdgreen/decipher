"""Ground-truth firewall regression tests (Phase-0 item 0.7).

Benchmark ground truth may be read only for post-hoc grading; it must never
flow toward the model or the solver and influence a run. These tests are a
first-pass leak detector on everything that reaches the model (agent path) and
a by-construction independence check on the automated solver (automated path).
"""
from __future__ import annotations

import json
import os
import random
import sys
from types import SimpleNamespace
from typing import Iterable

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import automated.runner as runner_module
from agent.loop_v2 import run_v2
from automated.runner import run_automated
from benchmark.context import ScopedBenchmarkContext
from models.alphabet import Alphabet
from models.cipher_text import CipherText


# A distinctive plaintext that no tool would ever emit by chance.
GROUND_TRUTH = "WOMBATFESTIVALQUARTZGIZMOJACKDAWNXYLOPHONEVEXBECK"


def _caesar(text: str, shift: int) -> str:
    return "".join(chr((ord(c) - 65 + shift) % 26 + 65) for c in text)


# The ciphertext the harness actually sees — the plaintext never appears in it.
CIPHERTEXT = _caesar(GROUND_TRUTH, 7)


def _normalize(text: str) -> str:
    """Uppercase and strip all whitespace (also yields the no-boundary form)."""
    return "".join(str(text).upper().split())


def assert_no_ground_truth_leak(haystacks: Iterable[str], plaintext: str) -> None:
    """Assert the ground truth does not appear in any haystack.

    Comparison is case-insensitive and whitespace-insensitive (which also
    covers the no-boundary variant). Both the full string and its first 30
    characters are checked, so a partial leak is caught too.
    """
    needle = _normalize(plaintext)
    needle_prefix = needle[:30]
    assert len(needle_prefix) == 30, "test plaintext must be >= 30 chars"
    for haystack in haystacks:
        if haystack is None:
            continue
        norm = _normalize(haystack)
        assert needle not in norm, f"full ground truth leaked: ...{norm[:80]}..."
        assert needle_prefix not in norm, (
            f"ground-truth prefix leaked: ...{norm[:80]}..."
        )


def test_helper_detects_a_planted_leak():
    """Sanity check: the helper is not a no-op."""
    with pytest.raises(AssertionError):
        assert_no_ground_truth_leak([f"prefix {GROUND_TRUTH} suffix"], GROUND_TRUTH)
    # A no-boundary / lowercased variant is also caught.
    with pytest.raises(AssertionError):
        assert_no_ground_truth_leak([GROUND_TRUTH.lower()], GROUND_TRUTH)


class _RecordingAPI:
    """Fake provider that records every system prompt + message list it sees
    and drives the loop with a harmless read-only tool call."""

    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen: list = []
        self.systems: list = []
        self.n = 0

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.systems.append(system)
        self.n += 1
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id=f"d{self.n}",
                    name="decode_show",
                    input={"branch": "main"},
                )
            ],
        )


def _collect_agent_haystacks(api: _RecordingAPI, artifact) -> list[str]:
    haystacks: list[str] = list(api.systems)
    for msgs in api.messages_seen:
        for m in msgs:
            content = m.get("content")
            if isinstance(content, str):
                haystacks.append(content)
            elif isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        haystacks.append(str(c))
                    elif c.get("type") == "text":
                        haystacks.append(c.get("text", ""))
                    elif c.get("type") == "tool_result":
                        haystacks.append(json.dumps(c.get("content")))
                    else:
                        haystacks.append(json.dumps(c))
    for tc in artifact.tool_calls:
        haystacks.append(tc.result)
    haystacks.append(json.dumps(artifact.benchmark_context))
    return haystacks


def test_agent_path_never_sees_ground_truth():
    alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
    ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)

    # A benign benchmark context (the kind build_benchmark_context produces —
    # metadata/related-cipher notes, never the plaintext answer).
    benchmark_context = ScopedBenchmarkContext(
        policy="max",
        prompt=(
            "Related manuscript context: an 18th-century lodge document, "
            "German Masonic register. No decrypted plaintext is provided."
        ),
    )

    api = _RecordingAPI()
    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=2,
        cipher_id="firewall_agent",
        prior_context=benchmark_context.prompt,
        benchmark_context=benchmark_context,
    )

    # The run must be non-trivial for the assertion to mean anything.
    assert api.messages_seen, "provider was never called"
    assert api.systems and api.systems[0], "no system prompt captured"
    assert artifact.tool_calls, "no tool calls executed"

    haystacks = _collect_agent_haystacks(api, artifact)
    haystacks.append(benchmark_context.prompt)
    assert_no_ground_truth_leak(haystacks, GROUND_TRUTH)

    # Ground truth is never populated on the agent artifact by the loop itself;
    # it is only attached post-hoc by the benchmark runner for scoring.
    assert artifact.ground_truth is None


def test_automated_path_ground_truth_only_grades_never_influences():
    """run_automated accepts a ground_truth argument, but it is used only for
    post-hoc scoring — the produced decryption/key must be identical whether or
    not ground truth is supplied, and the plaintext must not leak into any
    solver-facing output."""
    def _run(ground_truth):
        alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
        ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)
        # Built the way runner_v2/cli build the call.
        return run_automated(
            cipher_text=ct,
            language="en",
            cipher_id="firewall_auto",
            ground_truth=ground_truth,
            cipher_system="simple_substitution",
        )

    random.seed(0)
    without = _run(None)
    random.seed(0)
    with_gt = _run(GROUND_TRUTH)

    # Loud guard against a vacuous pass: if the solver could not actually run
    # (e.g. the English binary n-gram model is absent), status is "error" and
    # the decryption is empty — every equality assertion below would then hold
    # trivially without testing anything.
    assert with_gt.status == "completed", (
        f"automated run did not complete (status={with_gt.status!r}, "
        f"error={with_gt.error_message!r}); firewall comparison would be vacuous"
    )
    assert with_gt.final_decryption, (
        "empty decryption; firewall comparison would be vacuous"
    )

    # Ground truth does not influence the solve.
    assert without.final_decryption == with_gt.final_decryption
    assert without.artifact["key"] == with_gt.artifact["key"]

    # Steps are identical too, once run-varying timing is stripped.
    def _strip_timing(steps):
        return [
            {k: v for k, v in step.items() if k != "elapsed_seconds"}
            for step in steps
        ]

    assert _strip_timing(without.steps) == _strip_timing(with_gt.steps)

    # And the plaintext does not leak into solver-facing outputs. It MAY appear
    # only in the artifact's post-hoc grading fields.
    solver_facing = [
        with_gt.final_decryption,
        json.dumps(with_gt.steps),
        json.dumps(with_gt.artifact["key"]),
        with_gt.artifact.get("solver", ""),
    ]
    assert_no_ground_truth_leak(solver_facing, GROUND_TRUTH)

    # Document the post-hoc grading channel: ground truth IS retained there.
    assert with_gt.artifact.get("ground_truth") == GROUND_TRUTH


def test_multipage_group_route_is_ground_truth_free(monkeypatch, tmp_path):
    """Phase-2.4 extension: the multipage shared-key route must stay
    ground-truth-free. The homophonic solve is stubbed to a canned identity
    solver so the real projection/scoring path runs, and the run uses the
    ``word_repair`` refinement WITHOUT stubbing ``propose_word_repairs`` — the
    real word-repair library processes the group under the leak assertion, and
    F1 guarantees the pages it receives carry no plaintext (verified by a spy).
    Each page's plaintext is a distinctive ground truth that must not influence
    the solve or leak into any solver-facing output (combined solve steps,
    shared key, per-page projected decryptions, or the word-repair menu). It
    MAY appear only in the per-page artifact's post-hoc grading fields."""
    from automated.multipage_route import run_automated_multipage

    # A 30-symbol S-token page whose *plaintext* is the distinctive ground truth.
    n_symbols = 30
    symbols = " ".join(f"S{i:03d}" for i in range(n_symbols))

    def canned_homophonic(cipher_text, language, **_kwargs):
        # Identity-letter key: decode each symbol to its own (id mod 26) letter,
        # reproducing a ciphertext stand-in, never the ground-truth plaintext.
        key = {
            tid: (tid % 26) for tid in range(cipher_text.alphabet.size)
        }
        decryption = "".join(chr(ord("A") + key[t]) for t in cipher_text.tokens)
        step = {"name": "search_homophonic_anneal", "solver": "native_homophonic_anneal"}
        return "native_homophonic_anneal", key, decryption, step

    monkeypatch.setattr(runner_module, "_run_homophonic", canned_homophonic)

    # Firewall-by-construction spy (F1): the pages that reach the word-repair
    # refinement must have plaintext stripped. NOT a stub — the real
    # refinement (and the real propose_word_repairs pipeline) still runs.
    real_refinement = runner_module._word_repair_refinement_on_pages
    refinement_calls: list[int] = []

    def spying_refinement(**kwargs):
        assert all(page.plaintext == "" for page in kwargs["pages"]), (
            "plaintext reached the word-repair refinement"
        )
        refinement_calls.append(len(kwargs["pages"]))
        return real_refinement(**kwargs)

    monkeypatch.setattr(
        runner_module, "_word_repair_refinement_on_pages", spying_refinement
    )

    class _FakeLoader:
        def __init__(self, data):
            self._data = data

        def load_tests(self, split):  # noqa: ARG002
            return [SimpleNamespace(test_id=tid) for tid in self._data]

        def load_test_data(self, test):
            canonical, plaintext = self._data[test.test_id]
            return SimpleNamespace(canonical_transcription=canonical, plaintext=plaintext)

    data = {
        "pgA": (symbols, GROUND_TRUTH),
        "pgB": (symbols, GROUND_TRUTH),
    }
    group = {
        "name": "firewall_group",
        "benchmark_split": "x.jsonl",
        "language": "en",
        "description": "firewall",
        "test_ids": ["pgA", "pgB"],
    }
    result = run_automated_multipage(
        loader=_FakeLoader(data),
        group=group,
        homophonic_budget="screen",
        homophonic_refinement="word_repair",
        artifact_dir=str(tmp_path),
    )

    assert result.status == "completed", "route did not complete; comparison vacuous"
    # The real word-repair pipeline actually ran on the full 2-page group.
    assert refinement_calls == [2], "word-repair refinement was not exercised"
    assert result.word_repair is not None
    assert result.word_repair.get("status") == "completed", (
        f"word repair did not complete: {result.word_repair.get('reason')!r}"
    )

    group_artifact = json.loads(
        open(result.group_artifact_path, encoding="utf-8").read()
    )
    # Solver-facing outputs: combined solve steps, shared keys (final +
    # pre-refinement), per-page projected decryptions, the word-repair step
    # (real candidate menu included), and per-page artifact solver fields.
    solver_facing = [
        json.dumps(group_artifact["combined_solve_steps"]),
        json.dumps(group_artifact.get("key")),
        json.dumps(group_artifact["combined"].get("key")),
        json.dumps(group_artifact.get("word_repair")),
        *[pr.decryption for pr in result.page_results],
    ]
    for page in result.page_results:
        page_artifact = json.loads(open(page.artifact_path, encoding="utf-8").read())
        solver_facing.append(page_artifact["decryption"])
        solver_facing.append(json.dumps(page_artifact["key"]))
        solver_facing.append(json.dumps(page_artifact["steps"]))
        solver_facing.append(json.dumps(page_artifact["multipage_group"]))
        # Post-hoc grading channel retains ground truth (by design).
        assert page_artifact.get("ground_truth") == GROUND_TRUTH
    assert_no_ground_truth_leak(solver_facing, GROUND_TRUTH)


def test_automated_word_repair_refinement_is_ground_truth_free(monkeypatch):
    """Phase-2b extension: a run with the word_repair refinement enabled must
    stay ground-truth-free. The homophonic solve is stubbed to a fast, non-GT
    canned result so the real propose_word_repairs pipeline runs on the solved
    basin; the plaintext must not influence the repair nor leak into the
    word-repair artifact step / candidate menu."""
    def canned_homophonic(cipher_text, language, **_kwargs):
        # Identity-letter key: decode each cipher symbol to its own letter, so
        # the "solve" reproduces the (unreadable) ciphertext, never the GT.
        key = {
            token_id: ord(cipher_text.alphabet.symbol_for(token_id)) - ord("A")
            for token_id in range(cipher_text.alphabet.size)
        }
        decryption = "".join(chr(ord("A") + key[t]) for t in cipher_text.tokens)
        step = {"name": "search_homophonic_anneal", "solver": "native_homophonic_anneal"}
        return "native_homophonic_anneal", key, decryption, step

    monkeypatch.setattr(runner_module, "_run_homophonic", canned_homophonic)

    def _run(ground_truth):
        alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
        ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)
        return run_automated(
            cipher_text=ct,
            language="en",
            cipher_id="firewall_word_repair",
            ground_truth=ground_truth,
            cipher_system="homophonic_substitution",
            homophonic_refinement="word_repair",
        )

    without = _run(None)
    with_gt = _run(GROUND_TRUTH)

    assert with_gt.status == "completed", (
        f"run did not complete (status={with_gt.status!r}, "
        f"error={with_gt.error_message!r}); firewall comparison would be vacuous"
    )
    assert with_gt.final_decryption, "empty decryption; comparison would be vacuous"
    # The word-repair refinement actually ran and its step is present.
    assert any(step.get("name") == "search_word_repair" for step in with_gt.steps), (
        "word_repair step missing; the refinement path was not exercised"
    )

    # Ground truth does not influence the solve or the repair.
    assert without.final_decryption == with_gt.final_decryption
    assert without.artifact["key"] == with_gt.artifact["key"]

    # Absolute expectation (not just run-vs-run equality): the final key must
    # equal the identity-letter key the canned solver produced, computed here
    # independently of the runner. An in-place key mutation by the real
    # word-repair library would break this even if it hit both runs
    # identically.
    expected_alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
    expected_key = {
        str(token_id): ord(expected_alpha.symbol_for(token_id)) - ord("A")
        for token_id in range(expected_alpha.size)
    }
    assert with_gt.artifact["key"] == expected_key

    def _strip_timing(steps):
        return [
            {k: v for k, v in step.items() if k != "elapsed_seconds"}
            for step in steps
        ]

    assert _strip_timing(without.steps) == _strip_timing(with_gt.steps)

    # No leak into solver-facing outputs, including the word-repair step and its
    # candidate menu. The plaintext MAY appear only in post-hoc grading fields.
    solver_facing = [
        with_gt.final_decryption,
        json.dumps(with_gt.steps),
        json.dumps(with_gt.artifact["key"]),
        with_gt.artifact.get("solver", ""),
    ]
    assert_no_ground_truth_leak(solver_facing, GROUND_TRUTH)
    assert with_gt.artifact.get("ground_truth") == GROUND_TRUTH


def test_agent_word_repair_menu_tools_are_ground_truth_free(monkeypatch):
    """Phase-2.5 extension: the agent word-repair menu triplet operates on the
    workspace only (which never holds ground truth). Every model-visible result
    of search/review/rate/install must carry neither the benchmark plaintext nor
    the server-side candidate packets."""
    import agent.tools_v2 as tools_v2
    from agent.tools_v2 import WorkspaceToolExecutor
    from analysis.candidate_packet import CandidatePacket
    from automated.runner import WordRepairMenu
    from workspace import Workspace

    alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
    ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE"},
        word_list=["THE"],
        pattern_dict={},
    )
    ex.set_iteration(1)
    # Give main the (wrong-on-purpose) identity-ish key so the tool has a basin.
    ex.workspace.set_full_key("main", {tid: tid % 26 for tid in set(ct.tokens)})
    ex.workspace.get_branch("main").metadata["cipher_mode"] = "homophonic_substitution"

    packet = CandidatePacket(
        candidate_id="edits:A1:Q->T",
        kind="word_repair",
        rank=1,
        text=None,
        preview="THE QUICK BROWN",  # a candidate decode, never the plaintext
        solver_scores={"page_validation_avg": 2.0, "adjudication_score": 1.0},
        validation={"accepted": True, "decision": "runtime_accept", "reasons": ["ok"],
                    "deltas": {"page_validation_avg": 1.0}},
        provenance={"edits": ["A1:Q->T"], "mask": [],
                    "word_hypotheses": [{"observed": "QUICX", "target": "QUICK"}],
                    "collateral_evidence": {"collateral_occurrences": 1}},
    )

    def fake_build(**_kwargs):
        return WordRepairMenu(packets=[packet], baseline_validation=1.0, page=None, alphabet=alpha)

    monkeypatch.setattr(tools_v2.automated_runner, "build_word_repair_menu", fake_build)
    monkeypatch.setattr(
        tools_v2.automated_runner,
        "zenith_native_model_path",
        lambda _lang, variant=None: None,
    )

    results: list[str] = []
    menu = ex._tool_search_word_repair_menu({"branch": "main", "top_n": 8})
    results.append(json.dumps(menu, default=str))
    session_id = menu["search_session_id"]
    results.append(json.dumps(
        ex._tool_search_review_word_repair_finalists({"search_session_id": session_id}),
        default=str,
    ))
    results.append(json.dumps(
        ex._tool_act_rate_transform_finalist({
            "search_session_id": session_id,
            "rank": 1,
            "readability_score": 3,
            "label": "partial_clause",
            "rationale": "readable-ish",
        }),
        default=str,
    ))
    results.append(json.dumps(
        ex._tool_act_install_word_repair_finalists({"search_session_id": session_id, "ranks": [1]}),
        default=str,
    ))

    assert_no_ground_truth_leak(results, GROUND_TRUTH)
    for result in results:
        assert "packet" not in result, "candidate packet leaked into a model-visible result"


def test_v3_lead_context_never_sees_ground_truth():
    """The v3 lead loop rebuilds context from state each turn. Everything it
    sends to the model (rendered context blocks) and every tool result must be
    ground-truth free, including under a benign benchmark context."""
    from agent.model_provider import ModelResponse, ModelUsage, ToolUseBlock
    from investigation.loop_v3 import run_v3
    from investigation.sessions import SessionCapabilities
    from investigation.state import BudgetEntry

    alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
    ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)

    benchmark_context = ScopedBenchmarkContext(
        policy="max",
        prompt=(
            "Related manuscript context: an 18th-century lodge document, "
            "German Masonic register. No decrypted plaintext is provided."
        ),
    )

    class _RecordingSession:
        model = "fake-openai"
        provider_name = "openai"

        def __init__(self) -> None:
            self.capabilities = SessionCapabilities()
            self.blocks_seen: list = []
            self._budget: list = []
            self._n = 0

        def send(self, blocks, tools=None, max_tokens=8192):
            self.blocks_seen.append(blocks)
            self._budget.append(BudgetEntry("lead", "openai", "fake-openai", 10, 2, 0))
            self._n += 1
            return ModelResponse(
                content=[ToolUseBlock(id=f"d{self._n}", name="decode_show",
                                      input={"branch": "main"})],
                usage=ModelUsage(10, 2, 0),
            )

        def usage_entries(self):
            return list(self._budget)

        def export_transcript(self):
            return {"provider": "openai", "model": "fake-openai", "exchanges": []}

    session = _RecordingSession()
    artifact = run_v3(
        ct,
        session=session,
        language="en",
        max_iterations=2,
        cipher_id="firewall_v3",
        prior_context=benchmark_context.prompt,
        benchmark_context=benchmark_context,
    )

    assert session.blocks_seen, "session was never called"
    assert artifact.tool_calls, "no tool calls executed"

    haystacks: list[str] = [benchmark_context.prompt]
    for blocks in session.blocks_seen:
        for message in blocks:
            content = message.get("content")
            if isinstance(content, str):
                haystacks.append(content)
            elif isinstance(content, list):
                for c in content:
                    haystacks.append(json.dumps(c))
    for tc in artifact.tool_calls:
        haystacks.append(tc.result)
    haystacks.append(json.dumps(artifact.investigation_state))

    assert_no_ground_truth_leak(haystacks, GROUND_TRUTH)
    # Ground truth is never populated on the v3 artifact by the loop itself.
    assert artifact.ground_truth is None


def test_v3_repair_episode_and_composites_never_see_ground_truth():
    """M3: the repair-episode contract + rendered context (incl. the injected
    reading section) and all three composites' result packets are ground-truth
    free for a benchmark-backed cipher."""
    from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor
    from investigation.actions import execute_composite
    from investigation.episodes import (
        EPISODE_KINDS,
        EpisodeSpec,
        _build_episode_workspace,
        _episode_system_prompt,
        build_episode_context,
    )
    from investigation.reading import Reading, ReadingFragment
    from investigation.state import InvestigationState
    from workspace import Workspace

    alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
    ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)
    ws = Workspace(ct)
    pt = ws.plaintext_alphabet
    # A deliberately WRONG caesar key on main (shift 5, not the inverse -7), so
    # the decode is gibberish — never the ground-truth plaintext.
    for token_id in range(alpha.size):
        sym = alpha.symbol_for(token_id)
        ws.set_mapping("main", token_id, pt.id_for(_caesar(sym, 5)))
    ws.fork("alt", "main")
    state = InvestigationState(workspace=ws, language="en")

    # A benign reading (proposed plaintext the worker drafted — never GT).
    reading = Reading(branch="main", source="episode:e1", created_turn=1,
                      fragments=[ReadingFragment(text="A PROPOSED READING")],
                      holes=["unresolved tail"], overall_confidence=0.5)
    reading_dict = reading.to_dict()

    spec = EpisodeSpec("repair", "compile the reading", inputs={
        "branches": ["main"], "reading": reading_dict,
        "context_note": "manuscript register, no plaintext"})
    ep_ws = _build_episode_workspace(state, ["main"])
    executor = WorkspaceToolExecutor(ep_ws, "en", set(), [], {},
                                     declaration_policy=NoGatesPolicy(),
                                     episode_toolset=set(EPISODE_KINDS["repair"]["toolset"]))
    executor.episode_id = "ep_repair"
    executor.set_iteration(1)

    context_text = build_episode_context(spec, ep_ws, executor)
    system_prompt = _episode_system_prompt(spec)

    readings = {reading.reading_id: reading_dict}
    haystacks = [context_text, system_prompt]
    haystacks.append(json.dumps(execute_composite(
        "hypothesis_apply_reading",
        {"branch": "main", "reading_id": reading.reading_id, "dry_run": True},
        executor=executor, state_readings=readings, turn=1), default=str))
    haystacks.append(json.dumps(execute_composite(
        "hypothesis_test_word",
        {"branch": "main", "char_start": 0, "word": CIPHERTEXT[:5]},
        executor=executor, state_readings=readings, turn=1), default=str))
    haystacks.append(json.dumps(execute_composite(
        "branch_adjudicate",
        {"branches": ["main", "alt"], "include_window": True},
        executor=executor, state_readings=readings, turn=1), default=str))
    for tc in executor.call_log:
        haystacks.append(tc.result)

    assert_no_ground_truth_leak(haystacks, GROUND_TRUTH)


def test_v3_episode_surface_never_sees_ground_truth():
    """Episodes get NO benchmark context; the rendered episode context and the
    contract system prompt for a benchmark-backed cipher are ground-truth free."""
    from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor
    from investigation.episodes import (
        EpisodeSpec,
        _build_episode_workspace,
        _episode_system_prompt,
        build_episode_context,
    )
    from investigation.state import InvestigationState
    from workspace import Workspace

    alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
    ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)
    state = InvestigationState(workspace=Workspace(ct), language="en")

    spec = EpisodeSpec("survey", "diagnose the cipher",
                       inputs={"branches": ["main"],
                               "context_note": "manuscript register, no plaintext"})
    ep_ws = _build_episode_workspace(state, ["main"])
    executor = WorkspaceToolExecutor(ep_ws, "en", set(), [], {},
                                     declaration_policy=NoGatesPolicy())
    context_text = build_episode_context(spec, ep_ws, executor)
    system_prompt = _episode_system_prompt(spec)

    assert_no_ground_truth_leak([context_text, system_prompt], GROUND_TRUTH)


# ---------------------------------------------------------------------------
# M4 Part 6: experiment specs and results are leak-checked surfaces (C6).
# ---------------------------------------------------------------------------
def test_experiment_surfaces_never_leak_ground_truth():
    from analysis import dictionary, pattern
    from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor
    from investigation.context import _render_experiment_queue
    from investigation.experiments import (
        EXPERIMENT_TYPES,
        ExperimentQueue,
        dispatch_experiment_collect,
        dispatch_experiment_submit,
        register_experiment_type,
    )
    from investigation.state import InvestigationState
    from workspace import Workspace

    alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
    ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)
    ws = Workspace(cipher_text=ct)
    state = InvestigationState(workspace=ws, language="en")
    # A benign benchmark context (never the plaintext answer).
    state.external_context = (
        "Related manuscript context: an 18th-century lodge document. "
        "No decrypted plaintext is provided."
    )

    dict_path = dictionary.get_dictionary_path("en")
    word_set = dictionary.load_word_set(dict_path) if dict_path else set()
    word_list = pattern.load_word_list(dict_path) if dict_path else []
    executor = WorkspaceToolExecutor(
        workspace=ws, language="en", word_set=word_set, word_list=word_list,
        pattern_dict=pattern.build_pattern_dictionary(word_list),
        declaration_policy=NoGatesPolicy(),
        finalist_sessions=state.finalist_sessions,
    )

    null_step = {
        "name": "search_null_masks", "status": "completed",
        "selected": {"decryption": "BENIGN READING", "mask": [1],
                     "validation_score_v2": 0.5, "selection_score": 0.4},
        "top_finalists": [
            {"mask": [1], "status": "completed", "decryption": "BENIGN READING",
             "validation_score_v2": 0.5, "selection_score": 0.4, "candidate_id": "m1"},
        ],
        "evaluated_rows": [], "candidate_symbols": [],
        "packet": {"secret": "artifact-only"},
    }

    def runner(cipher, snapshot, config):
        return {"status": "completed", "solver": "zenith_native",
                "final_decryption": "BENIGN READING", "error_message": "",
                "key": {"0": 0}, "steps": [{"name": "route_automated_solver"}, null_step]}

    register_experiment_type("fw", {"config_schema": {"type": "object", "properties": {}},
                                    "config_defaults": {}, "runner": runner, "description": "fw"})
    try:
        q = ExperimentQueue(synchronous=True)
        sub = dispatch_experiment_submit(q, state, ws, executor,
                                         {"type": "fw", "branch": "main"}, 1)
        # (a) rendered queue section, (b) submit result.
        queue_section = _render_experiment_queue(state)
        # (c) collect packet with an installed null-mask case.
        packet = dispatch_experiment_collect(
            q, state, ws, executor,
            {"experiment_id": sub["experiment_id"], "install": True}, 2)
    finally:
        EXPERIMENT_TYPES.pop("fw", None)

    haystacks = [
        queue_section,
        json.dumps(sub, default=str),
        json.dumps(packet, default=str),
        # (d) raw record dicts.
        json.dumps(state.experiment_queue, default=str),
        json.dumps(state.finalist_sessions.to_dict(), default=str),
    ]
    assert_no_ground_truth_leak(haystacks, GROUND_TRUTH)


def test_diagnose_report_does_not_leak_ground_truth():
    """INV-0: format_diagnosis + to_dict for a solved cipher never surface plaintext."""
    from investigation.diagnosis import diagnose, format_diagnosis

    rendering = CIPHERTEXT  # the ciphertext letters (never the plaintext)
    tokens = [ord(c) - 65 for c in rendering]
    report = diagnose(tokens, alphabet_size=26, alphabet_class="letters",
                      language="en", letter_rendering=rendering)
    haystacks = [format_diagnosis(report), json.dumps(report.to_dict(), default=str)]
    assert_no_ground_truth_leak(haystacks, GROUND_TRUTH)


def test_observe_diagnosis_dispatch_no_leak():
    """INV-0: the observe_diagnosis tool dispatches and never surfaces plaintext."""
    from agent.tools_v2 import WorkspaceToolExecutor
    from workspace import Workspace

    alpha = Alphabet.from_text(CIPHERTEXT, ignore_chars=set())
    ct = CipherText(raw=CIPHERTEXT, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct), language="en",
        word_set={"THE"}, word_list=["THE"], pattern_dict={},
    )
    ex.set_iteration(1)
    handler_result = ex._tool_observe_diagnosis({"branch": "main"})
    assert "report" in handler_result and "formatted" in handler_result
    assert handler_result["report"]["verdict"] in {"confident", "uncertain"}
    dispatched = ex.execute("observe_diagnosis", {"branch": "main"})  # serialized envelope
    haystacks = [json.dumps(handler_result, default=str), str(dispatched)]
    assert_no_ground_truth_leak(haystacks, GROUND_TRUTH)
