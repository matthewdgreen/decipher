"""Regression tests for agent-loop reliability guardrails."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.loop_v2 import (
    FINAL_ITERATION_PREFLIGHT,
    FULL_READING_WORKFLOW_TOOL_NAMES,
    PENULTIMATE_READING_WORKFLOW_PREFLIGHT,
    PENULTIMATE_ALLOWED_TOOL_NAMES,
    READING_WORKFLOW_GATE_PREFLIGHT,
    build_workspace_panel,
    run_v2,
)
from agent.prompts_v2 import get_system_prompt, initial_context
import agent.tools_v2 as tools_v2
from agent.tools_v2 import WorkspaceToolExecutor
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from services.claude_api import ClaudeAPIError
from workspace import Workspace


def _executor_for(raw: str, separator: str | None = None) -> WorkspaceToolExecutor:
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=separator)
    return WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "OLD", "BAKERY", "ON", "MAPLE", "STREET"},
        word_list=["THE", "OLD", "BAKERY", "ON", "MAPLE", "STREET"],
        pattern_dict={},
    )


def _homophonic_executor() -> WorkspaceToolExecutor:
    symbols = [f"S{i:02d}" for i in range(30)]
    raw = " ".join(symbols)
    alpha = Alphabet(symbols)
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "OLD", "BOOK", "STORE", "MAPLE", "STREET"},
        word_list=["THE", "OLD", "BOOK", "STORE", "MAPLE", "STREET"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    # Deliberately overload H and leave U absent.
    for i, sym in enumerate(symbols):
        letter = "H" if i < 5 else "E"
        ex.workspace.set_mapping("main", alpha.id_for(sym), pt.id_for(letter))
    return ex


def test_score_delta_reports_mixed_when_signals_disagree():
    ex = _executor_for("ABC")
    before = {"dict_rate": 0.90, "quad": -5.0}
    after = {"dict_rate": 0.80, "quad": -4.8}

    delta = ex._score_delta(before, after)

    assert delta["verdict"] == "mixed"
    assert delta["improved"] is False
    assert delta["dict_rate_delta"] == -0.1
    assert delta["quad_delta"] == 0.2


def test_reading_score_delta_strips_verdict_for_act_set_mapping():
    """Reading-driven repair primitives must not surface a `verdict` /
    `improved` quality judgement on their score delta — the score is
    advisory, the agent's reading is authoritative.
    """
    ex = _executor_for("ABC")
    before = {"dict_rate": 0.90, "quad": -5.0}
    after = {"dict_rate": 0.80, "quad": -4.8}

    delta = ex._reading_score_delta(before, after)

    assert "verdict" not in delta
    assert "improved" not in delta
    # Raw deltas remain so the agent has the data, just not a label.
    assert delta["dict_rate_delta"] == -0.1
    assert delta["quad_delta"] == 0.2


def test_act_set_mapping_returns_changed_words_and_no_verdict():
    """Reading-driven discipline regression: act_set_mapping must report
    `changed_words` (was → now) so the agent decides by reading rather
    than by score, and must not surface a `verdict` field in score_delta.
    """
    raw = "AB | CD | AB"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"AT", "OF"},
        word_list=["AT", "OF"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    # Initial key: A→A, B→T, C→O, D→F → words decode as "AT", "OF", "AT".
    # All three are in word_set; dict_rate starts at 1.0.
    ws = ex.workspace
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("T"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("O"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("F"))

    # Now change A → X. Two of three words now read "XT" instead of "AT".
    out = ex._tool_act_set_mapping({
        "branch": "main",
        "cipher_symbol": "A",
        "plain_letter": "X",
    })

    assert out["status"] == "ok"
    # changed_words must list the affected words (was → now).
    changed = out["changed_words"]
    assert isinstance(changed, list)
    assert len(changed) >= 2  # both AT instances changed
    sample = changed[0]
    assert sample["before"] == "AT"
    assert sample["after"] == "XT"
    # score_delta exposes raw deltas but no authoritative verdict.
    assert "verdict" not in out["score_delta"]
    assert "improved" not in out["score_delta"]
    assert "dict_rate_delta" in out["score_delta"]
    # Note must instruct the agent that score deltas are advisory.
    assert "advisory" in out["note"].lower()
    assert "changed_words" in out["note"]


def test_act_set_mapping_keeps_reading_positive_negative_score_advisory(monkeypatch):
    """Discipline regression: a reading-positive mapping may lower the
    scorer. The tool must not label that as worse; it must surface the
    changed words so the agent can keep the readable repair.
    """
    raw = "XAT | XAR | OF"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"BAT", "BAR", "OF"},
        word_list=["BAT", "BAR", "OF"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    for cipher_sym, plain in {
        "X": "C",
        "A": "A",
        "T": "T",
        "R": "R",
        "O": "O",
        "F": "F",
    }.items():
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(plain))

    def fake_scores(branch: str) -> dict:
        decoded = ex.workspace.apply_key(branch)
        if decoded.startswith("CAT | CAR"):
            return {"dict_rate": 0.90, "quad": -5.0}
        if decoded.startswith("BAT | BAR"):
            return {"dict_rate": 0.70, "quad": -5.2}
        return {"dict_rate": 0.0, "quad": -99.0}

    monkeypatch.setattr(ex, "_compute_quick_scores", fake_scores)

    out = ex._tool_act_set_mapping({
        "branch": "main",
        "cipher_symbol": "X",
        "plain_letter": "B",
    })

    assert out["status"] == "ok"
    assert ex.workspace.apply_key("main").startswith("BAT | BAR")
    assert {"index": 0, "before": "CAT", "after": "BAT"} in out["changed_words"]
    assert {"index": 1, "before": "CAR", "after": "BAR"} in out["changed_words"]
    assert out["score_delta"]["dict_rate_delta"] < 0
    assert "verdict" not in out["score_delta"]
    assert "improved" not in out["score_delta"]
    assert "keep the change" in out["note"].lower()


def test_act_bulk_set_and_anchor_word_use_reading_score_delta():
    raw = "AB | CD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"AT", "OF"},
        word_list=["AT", "OF"],
        pattern_dict={},
    )

    bulk = ex._tool_act_bulk_set({
        "branch": "main",
        "mappings": {"A": "A", "B": "T", "C": "O", "D": "F"},
    })
    assert bulk["status"] == "ok"
    assert bulk["changed_words"]
    assert "verdict" not in bulk["score_delta"]
    assert "improved" not in bulk["score_delta"]

    ex.workspace.fork("anchor")
    anchor = ex._tool_act_anchor_word({
        "branch": "anchor",
        "cipher_word_index": 0,
        "plaintext": "IT",
    })
    assert anchor["status"] == "ok"
    assert anchor["changed_words"]
    assert "verdict" not in anchor["score_delta"]
    assert "improved" not in anchor["score_delta"]


def test_act_swap_decoded_revert_lists_unidirectional_alternatives():
    """When act_swap_decoded auto-reverts, the result must include
    `unidirectional_alternatives` listing specific act_set_mapping calls
    that would make the same intent without the bidirectional side-effect.
    """
    raw = "AB | CD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"AT", "OF"},
        word_list=["AT", "OF"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    # Set up A→A, B→T, C→O, D→F so the branch decodes as ["AT", "OF"]
    # — both dictionary words. A swap of A↔O will break both.
    ws = ex.workspace
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("T"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("O"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("F"))

    out = ex._tool_act_swap_decoded({
        "branch": "main",
        "letter_a": "A",
        "letter_b": "O",
        "auto_revert_if_worse": True,
    })

    assert out["status"] == "reverted"
    alternatives = out["unidirectional_alternatives"]
    assert isinstance(alternatives, list)
    assert any("act_set_mapping" in alt for alt in alternatives)
    # The note must explicitly steer toward act_set_mapping for
    # reading-driven repairs.
    assert "act_set_mapping" in out["note"]
    assert "bidirectional" in out["note"].lower()


def test_recommended_boundary_tool_yields_none_when_letter_candidates_present():
    """Boundary edits must not be promoted as the next move when the
    diagnostic also surfaced letter-level corrections — letter-level fixes
    are far higher leverage on boundary-preserved ciphers.
    """
    ex = _executor_for("AB")
    boundary = [{"type": "split", "cipher_word_index": 0}]
    letter = [{"wrong": "X", "correct": "Y", "evidence_count": 3}]

    # Boundary alone → recommend boundary actuator.
    assert ex._recommended_boundary_tool(boundary) == (
        "act_apply_boundary_candidate(branch='...', candidate_index=0)"
    )

    # Boundary + letter-level → no recommendation (letter fixes dominate).
    assert ex._recommended_boundary_tool(
        boundary, letter_candidates=letter
    ) is None

    # Empty letter list → boundary recommendation still surfaces.
    assert ex._recommended_boundary_tool(
        boundary, letter_candidates=[]
    ) == "act_apply_boundary_candidate(branch='...', candidate_index=0)"


def test_system_prompt_carries_reading_first_discipline():
    """The Reading-driven repair section of the prompt must declare that
    the agent's reading is authoritative when it can read coherent target-
    language words, must instruct the cipher-symbol mental model, must
    warn against bidirectional act_swap_decoded, and must require an
    applied reading anchor before anchored polish.
    """
    prompt = get_system_prompt("la")

    # Hierarchy
    assert "reading is authoritative" in prompt
    # Cipher-symbol framing
    assert "cipher-symbol" in prompt.lower()
    # Worked example uses placeholder symbols, not Latin words
    assert "TREUITER" not in prompt
    assert "QUEDAM" not in prompt
    # act_swap_decoded warning
    assert "act_swap_decoded" in prompt
    assert "bidirectional" in prompt.lower()
    # Anchored-polish sequencing rule
    assert "before any reading-driven anchor" in prompt
    # Tool-output discipline
    assert "changed_words" in prompt
    # Boundary-normalization discipline for solved streams with bad word breaks
    assert "boundary-normalization pass" in prompt
    assert "WITH | OUT" in prompt
    assert "act_resegment_by_reading" in prompt
    assert "act_resegment_from_reading_repair" in prompt
    assert "decode_validate_reading_repair" in prompt
    assert "act_merge_decoded_words" in prompt
    # Generalised dict_rate guidance — no fixed Latin threshold
    assert "0.15" not in prompt
    assert "Declare on reading" in prompt


def test_decode_ambiguous_letter_groups_contexts_by_cipher_symbol():
    ex = _executor_for("ABCAAC", separator=None)
    ws = ex.workspace
    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("I"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("I"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("H"))

    out = ex._tool_decode_ambiguous_letter({
        "branch": "main",
        "decoded_letter": "I",
        "context": 1,
    })

    assert out["cipher_symbols"] == ["A", "C"]
    assert out["symbol_count"] == 2
    assert {g["cipher_symbol"] for g in out["groups"]} == {"A", "C"}
    assert "act_set_mapping" in out["groups"][0]["suggested_next_step"]


def test_decode_repair_no_boundary_returns_text_only_repair_preview():
    alpha = Alphabet.from_text("THERQ CAT", ignore_chars=set())
    ct = CipherText(raw="THERQ CAT", alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THERE", "CAT"},
        word_list=["THERE", "CAT"],
        pattern_dict={},
    )
    alpha = ex.workspace.cipher_text.alphabet
    pt = ex.workspace.plaintext_alphabet
    for letter in ["T", "H", "E", "R", "Q", "C", "A"]:
        ex.workspace.set_mapping("main", alpha.id_for(letter), pt.id_for(letter))

    out = ex._tool_decode_repair_no_boundary({
        "branch": "main",
    })

    assert out["applied"] is True
    assert out["repaired_text"] == "THERE CAT"
    assert out["after"]["dict_rate"] == 1.0


def test_initial_context_discourages_remeasuring_without_leaking_cipher_label():
    msg = initial_context(
        cipher_display="01 02 03",
        alphabet_symbols=[f"{i:02d}" for i in range(57)],
        total_tokens=1096,
        total_words=1,
        ic_value=0.0215,
        language="en",
    )

    assert "Do not spend your first turns re-running frequency or IC" in msg
    assert "use that solver immediately" in msg
    assert "homophonic" not in msg.lower()


def test_system_prompt_routes_from_measured_facts_to_homophonic_solver():
    system = get_system_prompt("en")

    assert "opening measured facts" in system
    assert "many-symbol alphabet and no word boundaries" in system
    assert "search_homophonic_anneal" in system
    assert "search_automated_solver" in system
    assert "automated_preflight" in system
    assert "Do not spend early turns re-measuring facts" in system


class _FakeAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []
        self.tools_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.tools_seen.append(tools or [])
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=[SimpleNamespace(type="text", text="I forgot to declare.")],
        )


def test_final_turn_prefight_and_auto_declare_fallback():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _FakeAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=1,
        cipher_id="unit",
    )

    sent_texts = [
        c["text"]
        for m in api.messages_seen[-1]
        for c in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    assert any(FINAL_ITERATION_PREFLIGHT in t for t in sent_texts)
    assert artifact.status == "solved"
    assert artifact.solution is not None
    assert artifact.solution.branch == "main"
    assert artifact.solution.self_confidence == 0.0


def test_two_turn_runs_do_not_gate_first_action_turn():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _FakeAPI()

    run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=2,
        cipher_id="unit",
    )

    first_turn_texts = [
        c["text"]
        for m in api.messages_seen[0]
        for c in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    first_turn_tool_names = {tool["name"] for tool in api.tools_seen[0]}
    assert not any(PENULTIMATE_READING_WORKFLOW_PREFLIGHT in t for t in first_turn_texts)
    assert "search_anneal" in first_turn_tool_names
    assert "act_bulk_set" in first_turn_tool_names


def test_prefinal_window_gates_local_edit_tools():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _FakeAPI()

    run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=3,
        cipher_id="unit",
    )

    first_turn_texts = [
        c["text"]
        for m in api.messages_seen[0]
        for c in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    first_turn_tool_names = {tool["name"] for tool in api.tools_seen[0]}
    assert any(READING_WORKFLOW_GATE_PREFLIGHT in t for t in first_turn_texts)
    assert first_turn_tool_names == PENULTIMATE_ALLOWED_TOOL_NAMES
    assert "search_anneal" not in first_turn_tool_names
    assert "act_bulk_set" not in first_turn_tool_names


def test_final_turn_only_exposes_declare_tool():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _FakeAPI()

    run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=1,
        cipher_id="unit",
    )

    final_turn_tool_names = {tool["name"] for tool in api.tools_seen[-1]}
    assert final_turn_tool_names == {"meta_declare_solution"}


def test_automated_preflight_context_and_branch_available_on_first_turn():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _FakeAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=1,
        cipher_id="unit",
        automated_preflight={
            "enabled": True,
            "run_mode": "automated_only",
            "status": "solved",
            "solver": "fake_native",
            "summary": "Automated native solver preflight (no LLM access): THE",
            "key": {"0": 19, "1": 7, "2": 4},
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        },
    )

    first_content = api.messages_seen[0][0]["content"]
    first_turn_texts = [first_content] if isinstance(first_content, str) else [
        c["text"]
        for c in first_content
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    assert any("Automated native solver preflight" in t for t in first_turn_texts)
    assert artifact.automated_preflight is not None
    branch = next(b for b in artifact.branches if b.name == "automated_preflight")
    assert branch.decryption == "THE"
    assert "no_llm" in branch.tags


class _ToolThenErrorAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.calls = 0

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="tu_1",
                        name="act_set_mapping",
                        input={
                            "branch": "main",
                            "cipher_symbol": "A",
                            "plain_letter": "T",
                        },
                    )
                ],
            )
        raise ClaudeAPIError("overloaded")


def test_api_error_after_progress_auto_declares_best_branch():
    alpha = Alphabet(list("A"))
    ct = CipherText(raw="A", alphabet=alpha, separator=None)

    artifact = run_v2(
        cipher_text=ct,
        claude_api=_ToolThenErrorAPI(),  # type: ignore[arg-type]
        language="en",
        max_iterations=2,
        cipher_id="unit",
    )

    assert artifact.status == "solved"
    assert artifact.error_message is not None
    assert "overloaded" in artifact.error_message
    assert artifact.solution is not None
    assert artifact.solution.branch == "main"
    assert artifact.branches[0].decryption == "T"


def test_search_anneal_restarts_complete_inherited_key_by_default(monkeypatch):
    ex = _executor_for("ABC", separator=None)
    ws = ex.workspace
    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("B"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("C"))

    def fake_anneal(session, score_fn, max_steps, t_start, t_end):
        return score_fn()

    monkeypatch.setattr(tools_v2, "simulated_anneal", fake_anneal)

    anchored = ex._tool_search_anneal({
        "branch": "main",
        "steps": 1,
        "restarts": 1,
        "preserve_existing": True,
    })
    fresh = ex._tool_search_anneal({
        "branch": "main",
        "steps": 1,
        "restarts": 1,
    })

    assert anchored["preserve_existing"] is True
    assert anchored["auto_seeded_symbols"] == 0
    assert fresh["preserve_existing"] is False
    assert fresh["preserved_symbols"] == 0
    assert fresh["auto_seeded_symbols"] == 3


def test_search_anneal_runs_key_repair_and_anchor_refine(monkeypatch):
    ex = _executor_for("ABC", separator=None)
    ws = ex.workspace
    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet

    repaired_key = {
        alpha.id_for("A"): pt.id_for("T"),
        alpha.id_for("B"): pt.id_for("H"),
        alpha.id_for("C"): pt.id_for("E"),
    }
    refined_key = {
        alpha.id_for("A"): pt.id_for("E"),
        alpha.id_for("B"): pt.id_for("T"),
        alpha.id_for("C"): pt.id_for("A"),
    }

    def fake_anneal(session, score_fn, max_steps, t_start, t_end):
        session.set_full_key({
            alpha.id_for("A"): pt.id_for("A"),
            alpha.id_for("B"): pt.id_for("B"),
            alpha.id_for("C"): pt.id_for("C"),
        })
        return 0.5

    monkeypatch.setattr(tools_v2, "simulated_anneal", fake_anneal)
    monkeypatch.setattr(
        tools_v2.automated_runner,
        "_run_key_consistent_repair",
        lambda **kwargs: {
            "applied": True,
            "reason": "test",
            "key": dict(repaired_key),
        },
    )
    monkeypatch.setattr(
        tools_v2.automated_runner,
        "_maybe_anchor_refine_substitution",
        lambda **kwargs: {
            "applied": True,
            "reason": "test",
            "key": dict(refined_key),
            "score": 1.75,
        },
    )

    out = ex._tool_search_anneal({
        "branch": "main",
        "steps": 1,
        "restarts": 1,
        "score_fn": "combined",
    })

    assert out["key_repair"]["applied"] is True
    assert out["anchor_refine"]["applied"] is True
    assert out["after"] == 1.75
    assert ws.get_branch("main").key == refined_key


def test_split_and_merge_cipher_word_tools_update_branch_local_boundaries():
    ex = _executor_for("ABC AB", separator=" ")
    ws = ex.workspace
    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("B"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("C"))
    ws.fork("exp")
    ws.set_full_key("exp", dict(ws.get_branch("main").key))

    split_out = ex._tool_act_split_cipher_word({
        "branch": "exp",
        "cipher_word_index": 0,
        "split_at_token_offset": 1,
    })

    assert split_out["status"] == "ok"
    assert split_out["left_cipher_word"] == "A"
    assert split_out["right_cipher_word"] == "BC"
    assert ws.apply_key("exp") == "A BC AB"
    assert ws.apply_key("main") == "ABC AB"

    merge_out = ex._tool_act_merge_cipher_words({
        "branch": "exp",
        "left_word_index": 0,
    })

    assert merge_out["status"] == "ok"
    assert merge_out["merged_cipher_word"] == "ABC"
    assert ws.apply_key("exp") == "ABC AB"


def test_workspace_panel_reflects_branch_local_word_boundaries():
    ex = _executor_for("ABC AB", separator=" ")
    ws = ex.workspace
    alpha = ws.cipher_text.alphabet
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("B"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("C"))
    ws.fork("exp")
    ws.set_full_key("exp", dict(ws.get_branch("main").key))
    ws.split_cipher_word("exp", 0, 1)

    panel = build_workspace_panel(
        ws,
        iteration=1,
        language="en",
        word_set={"A", "AB", "ABC", "BC"},
    )

    assert "### Branch `exp`" in panel
    assert "custom_boundaries=3" in panel
    assert "A | BC | AB" in panel


def test_workspace_panel_includes_penultimate_reading_workflow_warning():
    ex = _executor_for("AP PLY", separator=" ")

    panel = build_workspace_panel(
        ex.workspace,
        iteration=14,
        max_iterations=15,
        language="en",
        word_set={"APPLY"},
    )

    assert PENULTIMATE_READING_WORKFLOW_PREFLIGHT in panel
    assert "act_resegment_from_reading_repair" in panel


def test_decode_diagnose_can_suggest_merging_adjacent_cipher_words():
    alpha = Alphabet.from_text("AB CD", ignore_chars=set())
    ct = CipherText(raw="AB CD", alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="la",
        word_set={"CURA"},
        word_list=["CURA"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("C"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("U"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("R"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("A"))

    out = ex._tool_decode_diagnose({"branch": "main"})

    assert out["candidate_corrections"] == []
    assert out["boundary_candidates"]
    cand = out["boundary_candidates"][0]
    assert cand["type"] == "merge"
    assert cand["decoded_before"] == "CU | RA"
    assert cand["decoded_after"] == "CURA"
    assert "act_merge_cipher_words" in cand["suggested_call"]
    assert out["recommended_next_tool"] == "act_apply_boundary_candidate(branch='...', candidate_index=0)"


def test_decode_diagnose_can_suggest_compound_merge_when_parts_are_words():
    alpha = Alphabet.from_text("ABCD EFG", ignore_chars=set())
    ct = CipherText(raw="ABCD EFG", alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"WITH", "OUT", "WITHOUT"},
        word_list=["WITH", "OUT", "WITHOUT"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym, plain in {
        "A": "W",
        "B": "I",
        "C": "T",
        "D": "H",
        "E": "O",
        "F": "U",
        "G": "T",
    }.items():
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(plain))

    out = ex._tool_decode_diagnose({"branch": "main"})

    merge = next(c for c in out["boundary_candidates"] if c["type"] == "merge")
    assert merge["decoded_before"] == "WITH | OUT"
    assert merge["decoded_after"] == "WITHOUT"
    assert "both split parts" in merge["evidence"]
    assert out["recommended_next_tool"] == "act_apply_boundary_candidate(branch='...', candidate_index=0)"


def test_act_merge_decoded_words_finds_current_pair_after_prior_merge():
    alpha = Alphabet.from_text("AB CD EF GH", ignore_chars=set())
    ct = CipherText(raw="AB CD EF GH", alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"ABCD", "EFGH"},
        word_list=["ABCD", "EFGH"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    first = ex._tool_act_merge_decoded_words({
        "branch": "main",
        "left_decoded": "AB",
        "right_decoded": "CD",
    })
    assert first["status"] == "ok"
    assert first["matched_left_word_index"] == 0
    assert ws.apply_key("main") == "ABCD EF GH"

    second = ex._tool_act_merge_decoded_words({
        "branch": "main",
        "left_decoded": "EF",
        "right_decoded": "GH",
    })
    assert second["status"] == "ok"
    # EF | GH is now at index 1, not its original index 2.
    assert second["matched_left_word_index"] == 1
    assert ws.apply_key("main") == "ABCD EFGH"


def test_act_resegment_by_reading_applies_character_preserving_boundaries():
    raw = "THERE FORE THE OLD PHYSICS ER DID AP PLY A SALVE UN TO"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={
            "THEREFORE", "THE", "OLD", "PHYSICSER", "DID", "APPLY",
            "A", "SALVE", "UNTO",
        },
        word_list=[
            "THE", "A", "DID", "OLD", "APPLY", "UNTO", "SALVE",
            "THEREFORE", "PHYSICSER",
        ],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    proposed = "THEREFORE THE OLD PHYSICSER DID APPLY A SALVE UNTO"
    out = ex._tool_act_resegment_by_reading({
        "branch": "main",
        "proposed_text": proposed,
    })

    assert out["status"] == "ok"
    assert out["old_word_count"] == 13
    assert out["new_word_count"] == 9
    assert out["dictionary_after"]["dictionary_rate"] == 1.0
    assert ws.apply_key("main") == proposed


def test_act_resegment_by_reading_rejects_letter_changing_proposal():
    raw = "PHYSICS ER"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"PHYSICKER"},
        word_list=["PHYSICKER"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    out = ex._tool_act_resegment_by_reading({
        "branch": "main",
        "proposed_text": "PHYSICKER",
    })

    assert "error" in out
    assert out["character_preserving"] is False
    assert out["mismatches"][0]["current_char"] == "S"
    assert out["mismatches"][0]["proposed_char"] == "K"
    assert ws.apply_key("main") == "PHYSICS ER"


def test_decode_validate_reading_repair_classifies_boundary_vs_letter_changes():
    raw = "PHYSICS ER DID AP PLY"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"PHYSICKER", "DID", "APPLY"},
        word_list=["DID", "APPLY", "PHYSICKER"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    boundary_only = ex._tool_decode_validate_reading_repair({
        "branch": "main",
        "proposed_text": "PHYSICSER DID APPLY",
    })
    letter_repair = ex._tool_decode_validate_reading_repair({
        "branch": "main",
        "proposed_text": "PHYSICKER DID APPLY",
    })

    assert boundary_only["character_preserving"] is True
    assert "act_resegment_by_reading" in boundary_only["recommendation"]
    assert letter_repair["character_preserving"] is False
    assert letter_repair["mismatches"][0]["current_char"] == "S"
    assert letter_repair["mismatches"][0]["proposed_char"] == "K"
    assert "act_set_mapping" in letter_repair["recommendation"]
    assert letter_repair["boundary_projection"]["applicable"] is True
    assert "act_resegment_from_reading_repair" in letter_repair["recommendation"]


def test_act_resegment_from_reading_repair_applies_boundaries_despite_letter_diffs():
    raw = "PHYSICS ER DID AP PLY"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"PHYSICKER", "PHYSICSER", "DID", "APPLY"},
        word_list=["DID", "APPLY", "PHYSICKER", "PHYSICSER"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    out = ex._tool_act_resegment_from_reading_repair({
        "branch": "main",
        "proposed_text": "PHYSICKER DID APPLY",
    })

    assert out["status"] == "ok"
    assert out["mode"] == "boundary_projection_from_repair_reading"
    assert out["character_preserving"] is False
    assert out["mismatches"][0]["current_char"] == "S"
    assert out["mismatches"][0]["proposed_char"] == "K"
    assert out["projected_text"] == "PHYSICSER DID APPLY"
    assert ws.apply_key("main") == "PHYSICSER DID APPLY"


def test_act_resegment_from_reading_repair_rejects_length_mismatch():
    raw = "AP PLY"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"APPLYING"},
        word_list=["APPLYING"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    out = ex._tool_act_resegment_from_reading_repair({
        "branch": "main",
        "proposed_text": "APPLYING",
    })

    assert "error" in out
    assert out["boundary_projection"]["applicable"] is False
    assert ws.apply_key("main") == "AP PLY"


def test_meta_declare_blocks_boundary_damaged_branch_before_reading_workflow():
    ex = _executor_for("AP PLY", separator=" ")
    ex.set_max_iterations(10)
    ex.set_iteration(4)

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Readable text remains, but there are boundary issues.",
        "self_confidence": 0.7,
    })

    assert out["status"] == "blocked"
    assert out["accepted"] is False
    assert ex.terminated is False
    assert "decode_validate_reading_repair" in out["note"]

    ex.call_log.append(SimpleNamespace(
        tool_name="decode_validate_reading_repair",
        arguments={"branch": "main"},
    ))
    accepted = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Readable text remains, but there are boundary issues.",
        "self_confidence": 0.7,
    })

    assert accepted["status"] == "ok"
    assert accepted["accepted"] is True
    assert ex.terminated is True


def test_meta_declare_allows_final_turn_even_without_reading_workflow():
    ex = _executor_for("AP PLY", separator=" ")
    ex.set_max_iterations(10)
    ex.set_iteration(10)

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Partial readable text remains, with boundary issues.",
        "self_confidence": 0.7,
    })

    assert out["status"] == "ok"
    assert out["accepted"] is True
    assert ex.terminated is True


def test_execute_rejects_tools_not_allowed_on_gated_turn():
    ex = _executor_for("AP PLY", separator=" ")
    ex.set_allowed_tool_names({"decode_validate_reading_repair"})

    raw = ex.execute(
        "act_bulk_set",
        {"branch": "main", "mappings": {"A": "B"}},
        tool_use_id="unit",
    )

    assert "tool_gated" in raw
    assert "decode_validate_reading_repair" in raw
    assert "no longer allowed" in raw
    assert "allowed_tools" in raw
    assert "Do not call it again" in raw
    assert ex.call_log[-1].tool_name == "act_bulk_set"


def test_decode_diagnose_can_suggest_splitting_cipher_word():
    alpha = Alphabet.from_text("ABCDEF", ignore_chars=set())
    ct = CipherText(raw="ABCDEF", alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="la",
        word_set={"CURA", "ET"},
        word_list=["CURA", "ET"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("C"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("U"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("R"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("E"), pt.id_for("E"))
    ws.set_mapping("main", alpha.id_for("F"), pt.id_for("T"))

    out = ex._tool_decode_diagnose({"branch": "main"})

    assert out["boundary_candidates"]
    split = next(c for c in out["boundary_candidates"] if c["type"] == "split")
    assert split["decoded_before"] == "CURAET"
    assert split["decoded_after"] == "CURA | ET"
    assert split["split_at_token_offset"] == 4
    assert "act_split_cipher_word" in split["suggested_call"]
    assert out["recommended_next_tool"] == "act_apply_boundary_candidate(branch='...', candidate_index=0)"


def test_decode_diagnose_and_fix_surfaces_boundary_candidates_when_letter_fixes_are_weak():
    alpha = Alphabet.from_text("AB CD", ignore_chars=set())
    ct = CipherText(raw="AB CD", alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="la",
        word_set={"CURA"},
        word_list=["CURA"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("C"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("U"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("R"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("A"))

    out = ex._tool_decode_diagnose_and_fix({
        "branch": "main",
        "top_k": 5,
        "min_evidence": 2,
    })

    assert out["fixes_applied"] == []
    assert out["boundary_candidates"]
    assert out["boundary_candidates"][0]["type"] == "merge"
    assert "act_merge_cipher_words" in out["boundary_candidates"][0]["suggested_call"]
    assert out["recommended_next_tool"] == "act_apply_boundary_candidate(branch='...', candidate_index=0)"
    assert "Boundary edits look more promising than letter swaps here" in out["note"]


def test_act_apply_boundary_candidate_applies_top_merge_suggestion():
    alpha = Alphabet.from_text("AB CD", ignore_chars=set())
    ct = CipherText(raw="AB CD", alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="la",
        word_set={"CURA"},
        word_list=["CURA"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("C"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("U"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("R"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("A"))

    out = ex._tool_act_apply_boundary_candidate({
        "branch": "main",
        "candidate_index": 0,
    })

    assert out["status"] == "ok"
    assert out["applied_candidate"]["type"] == "merge"
    assert out["merged_cipher_word"] == "ABCD"
    assert ws.apply_key("main") == "CURA"


def test_homophone_distribution_flags_absent_and_overloaded_letters():
    ex = _homophonic_executor()

    out = ex._tool_observe_homophone_distribution({"branch": "main"})

    assert out["is_likely_homophonic"] is True
    assert sum(r["expected_symbols"] for r in out["expected_symbol_counts"]) == 30
    actual_h = next(r for r in out["actual_vs_expected"] if r["letter"] == "H")
    assert actual_h["actual_symbols"] == 5
    assert any("absent" in w for w in out["warnings"])


def test_absent_letter_candidates_rank_symbol_remaps_with_score_delta():
    ex = _homophonic_executor()

    out = ex._tool_decode_absent_letter_candidates({
        "branch": "main",
        "missing_letter": "U",
        "source_letters": ["H"],
        "max_candidates": 3,
        "context": 1,
    })

    assert out["missing_letter"] == "U"
    assert out["source_letters_considered"] == ["H"]
    assert out["candidates"]
    cand = out["candidates"][0]
    assert cand["current_letter"] == "H"
    assert cand["candidate_letter"] == "U"
    assert "score_delta_if_remapped" in cand
    assert "act_set_mapping" in cand["suggested_call"]


def test_decode_diagnose_uses_targeted_mapping_for_homophonic_single_symbol(monkeypatch):
    ex = _homophonic_executor()
    alpha = ex.workspace.cipher_text.alphabet
    pt = ex.workspace.plaintext_alphabet
    for sym, letter in {"S00": "U", "S01": "N", "S02": "B"}.items():
        ex.workspace.set_mapping("main", alpha.id_for(sym), pt.id_for(letter))

    import analysis.segment as segment

    monkeypatch.setattr(
        segment,
        "segment_text",
        lambda normalized, word_set, freq_rank: SimpleNamespace(
            pseudo_words=["UNB"],
            words=["UNB"],
            segmented="UNB",
            dict_rate=0.0,
        ),
    )
    monkeypatch.setattr(
        segment,
        "find_one_edit_corrections",
        lambda word, word_set: [("UND", "B", "D")],
    )

    out = ex._tool_decode_diagnose({"branch": "main", "top_k": 1})

    cand = out["candidate_corrections"][0]
    assert cand["ambiguous"] is False
    assert cand["culprit_symbol"] == "S02"
    assert "act_set_mapping" in cand["suggested_call"]
    assert "act_swap_decoded" not in cand["suggested_call"]


def test_act_swap_decoded_auto_reverts_worsening_swap(monkeypatch):
    ex = _executor_for("AB", separator=None)
    alpha = ex.workspace.cipher_text.alphabet
    pt = ex.workspace.plaintext_alphabet
    ex.workspace.set_mapping("main", alpha.id_for("A"), pt.id_for("B"))
    ex.workspace.set_mapping("main", alpha.id_for("B"), pt.id_for("D"))

    def fake_scores(branch: str) -> dict:
        decoded = ex.workspace.apply_key(branch)
        if decoded == "BD":
            return {"dict_rate": 0.9, "quad": -1.0}
        return {"dict_rate": 0.5, "quad": -2.0}

    monkeypatch.setattr(ex, "_compute_quick_scores", fake_scores)

    out = ex._tool_act_swap_decoded({
        "branch": "main",
        "letter_a": "B",
        "letter_b": "D",
    })

    assert out["status"] == "reverted"
    assert out["score_delta"]["verdict"] == "worse"
    assert ex.workspace.apply_key("main") == "BD"
