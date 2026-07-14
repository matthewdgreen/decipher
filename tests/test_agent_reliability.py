"""Regression tests for agent-loop reliability guardrails."""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.loop_v2 import (
    BOUNDARY_PROJECTION_COUNT_RETRY_PREFLIGHT,
    BOUNDARY_PROJECTION_RETRY_PREFLIGHT,
    FINAL_DECLARATION_RETRY_PREFLIGHT,
    FINAL_ITERATION_PREFLIGHT,
    FULL_READING_WORKFLOW_TOOL_NAMES,
    INSPECTION_SANDBOX_CONTINUE_PREFLIGHT,
    PANEL_HEADER_MARKER,
    PENULTIMATE_READING_WORKFLOW_PREFLIGHT,
    PENULTIMATE_ALLOWED_TOOL_NAMES,
    REPAIR_SANDBOX_CONTINUE_PREFLIGHT,
    READING_WORKFLOW_GATE_PREFLIGHT,
    TOOL_RESULT_STUB,
    _collect_assistant_blocks,
    _compress_history,
    _is_boundary_projection_count_failure,
    _is_tool_gated_result,
    _branch_snapshot_for,
    _workspace_snapshot_payload,
    build_workspace_panel,
    run_v2,
)
from agent.model_provider import (
    ClaudeModelProvider,
    ModelResponse,
    OpenAIModelProvider,
    ProviderExtraBlock,
    TextBlock,
    ToolUseBlock,
    _messages_to_gemini_contents,
    _messages_to_openai_chat,
    _messages_to_openai_responses,
    _openai_chat_response_to_model_response,
    _openai_responses_response_to_model_response,
    _reasoning_passback_enabled,
    _requires_responses_api,
    _sanitize_reasoning_item_for_input,
    _schema_for_gemini,
    _strip_provider_extra_blocks,
    _tools_to_openai_chat,
    _tools_to_openai_responses,
    default_model_for_provider,
    estimate_provider_cost,
    infer_provider_from_model,
    normalize_model_response,
)
from agent.orchestration import AgentMode, AgentRunState, MODE_ALLOWED_TOOLS
from agent.prompts_v2 import get_system_prompt, initial_context
import agent.tools_v2 as tools_v2
from agent.tools_v2 import WorkspaceToolExecutor
from artifact.schema import ToolCall
from benchmark.context import ScopedBenchmarkContext
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


def _context_executor_for(
    tmp_path,
    *,
    policy: str = "max",
) -> WorkspaceToolExecutor:
    root = tmp_path / "benchmark"
    (root / "sources" / "test" / "transcriptions").mkdir(parents=True)
    (root / "sources" / "test" / "plaintext").mkdir(parents=True)
    (root / "sources" / "test" / "documents").mkdir(parents=True)
    (root / "sources" / "test" / "transcriptions" / "related.canonical.txt").write_text(
        "S001 S002 S003\n"
    )
    (root / "sources" / "test" / "plaintext" / "related.txt").write_text(
        "KNOWN SOLVED TEXT\n"
    )
    (root / "sources" / "test" / "documents" / "context.txt").write_text(
        "This is a longer associated context document.\n"
    )
    (root / "sources" / "test" / "documents" / "solution.txt").write_text(
        "This document contains a solution and should be gated.\n"
    )

    ctx = ScopedBenchmarkContext(
        policy=policy,
        prompt="## Benchmark Context\nTest context",
        injected_layers=[
            {
                "record_id": "target",
                "layer": "minimal",
                "label": "Basic context",
                "text": "A handwritten cipher note.",
            }
        ],
        target_record_ids=["target"],
        context_record_ids=[],
        benchmark_root=str(root),
        records={
            "target": {
                "id": "target",
                "area": "benchmark",
                "source": "test",
                "transcription_canonical_file": "sources/test/transcriptions/target.canonical.txt",
                "plaintext_file": "sources/test/plaintext/target.txt",
                "root": str(root),
            },
            "related": {
                "id": "related",
                "area": "benchmark",
                "source": "test",
                "transcription_canonical_file": "sources/test/transcriptions/related.canonical.txt",
                "plaintext_file": "sources/test/plaintext/related.txt",
                "root": str(root),
                "relationship": {
                    "relationship": "same_source_known_solution",
                    "solution_available": True,
                },
            },
        },
        related_records={
            "related": {
                "id": "related",
                "area": "benchmark",
                "source": "test",
                "transcription_canonical_file": "sources/test/transcriptions/related.canonical.txt",
                "plaintext_file": "sources/test/plaintext/related.txt",
                "root": str(root),
                "relationship": {
                    "relationship": "same_source_known_solution",
                    "solution_available": True,
                },
            }
        },
        associated_documents={
            "context_doc": {
                "record_id": "target",
                "document": {
                    "id": "context_doc",
                    "title": "Context document",
                    "document_type": "letter",
                    "text_file": "sources/test/documents/context.txt",
                    "safe_context_layers": ["historical", "related_metadata", "max"],
                },
            },
            "solution_doc": {
                "record_id": "target",
                "document": {
                    "id": "solution_doc",
                    "title": "Solution document",
                    "document_type": "solution_note",
                    "text_file": "sources/test/documents/solution.txt",
                    "contains_solution": True,
                    "safe_context_layers": ["related_solutions", "max"],
                },
            },
        },
    )
    ex = _executor_for("ABC")
    ex.benchmark_context = ctx
    return ex


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


def test_benchmark_context_tool_lists_scoped_material(tmp_path):
    ex = _context_executor_for(tmp_path, policy="max")

    out = json.loads(ex.execute("inspect_benchmark_context", {}))

    assert out["available"] is True
    assert out["policy"] == "max"
    assert out["target_record_ids"] == ["target"]
    assert out["related_solution_allowed"] is True
    assert out["related_records_available"][0]["record_id"] == "related"
    assert out["associated_documents_available"][0]["document_id"] in {
        "context_doc",
        "solution_doc",
    }
    assert ex.benchmark_context.access_log[-1]["tool"] == "inspect_benchmark_context"


def test_related_solution_tool_blocks_when_policy_disallows(tmp_path):
    ex = _context_executor_for(tmp_path, policy="historical")

    out = json.loads(ex.execute("inspect_related_solution", {"record_id": "related"}))

    assert "disabled by benchmark context policy" in out["error"]
    assert ex.benchmark_context.access_log[-1]["allowed"] is False
    assert ex.benchmark_context.access_log[-1]["error"] == "solution_context_disabled"


def test_related_solution_tool_allows_related_but_never_target(tmp_path):
    ex = _context_executor_for(tmp_path, policy="max")

    related = json.loads(ex.execute("inspect_related_solution", {"record_id": "related"}))
    target = json.loads(ex.execute("inspect_related_solution", {"record_id": "target"}))

    assert related["content_type"] == "plaintext_solution"
    assert related["text"] == "KNOWN SOLVED TEXT"
    assert "target record's solution is never exposed" in target["error"]


def test_related_transcription_rejects_unlisted_record(tmp_path):
    ex = _context_executor_for(tmp_path, policy="max")

    out = json.loads(ex.execute("inspect_related_transcription", {"record_id": "not_allowed"}))

    assert "not in this run's benchmark context allowlist" in out["error"]
    assert out["allowed_record_ids"] == ["related", "target"]
    assert ex.benchmark_context.access_log[-1]["allowed"] is False


def test_associated_document_tool_respects_solution_gate(tmp_path):
    ex = _context_executor_for(tmp_path, policy="historical")

    context_doc = json.loads(
        ex.execute("inspect_associated_document", {"document_id": "context_doc"})
    )
    solution_doc = json.loads(
        ex.execute("inspect_associated_document", {"document_id": "solution_doc"})
    )

    assert context_doc["text"] == "This is a longer associated context document."
    assert "solution-bearing" in solution_doc["error"]


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


def test_act_set_mapping_dry_run_restores_branch_and_reports_undo():
    raw = "AB | AC"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"AT", "AX"},
        word_list=["AT", "AX"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("T"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("X"))

    out = ex._tool_act_set_mapping({
        "branch": "main",
        "cipher_symbol": "A",
        "plain_letter": "B",
        "dry_run": True,
    })

    assert out["status"] == "preview"
    assert out["dry_run"] is True
    assert out["previous_mapping"] == "A -> A"
    assert "act_set_mapping" in out["undo_call"]
    assert ws.apply_key("main") == "AT | AX"
    assert {"index": 0, "before": "AT", "after": "BT"} in out["changed_words"]


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


def test_decode_plan_word_repair_identifies_symbol_mapping_and_preview():
    raw = "ABC | ABD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"BAT", "BAR"},
        word_list=["BAT", "BAR"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("C"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("T"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("R"))

    out = ex._tool_decode_plan_word_repair({
        "branch": "main",
        "decoded_word": "CAT",
        "target_word": "BAT",
    })

    assert out["applicable"] is True
    assert out["cipher_word_index"] == 0
    assert out["current_decoded"] == "CAT"
    assert out["target_word"] == "BAT"
    assert out["proposed_mappings"] == {"A": "B"}
    assert {"index": 0, "before": "CAT", "after": "BAT"} in out["changed_words_preview"]
    assert {"index": 1, "before": "CAR", "after": "BAR"} in out["changed_words_preview"]
    assert out["agenda_item"]["status"] == "open"
    assert out["agenda_item"]["from"] == "CAT"
    assert out["agenda_item"]["to"] == "BAT"
    assert ex.repair_agenda[0]["id"] == out["agenda_item"]["id"]
    assert "_token_mappings" not in out


def test_act_apply_word_repair_applies_mapping_with_changed_words():
    raw = "ABC | ABD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"BAT", "BAR"},
        word_list=["BAT", "BAR"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("C"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("T"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("R"))

    out = ex._tool_act_apply_word_repair({
        "branch": "main",
        "decoded_word": "CAT",
        "target_word": "BAT",
    })

    assert out["status"] == "ok"
    assert out["mappings"] == {"A": "B"}
    assert out["mappings_set"] == 1
    assert {"index": 0, "before": "CAT", "after": "BAT"} in out["changed_words"]
    assert {"index": 1, "before": "CAR", "after": "BAR"} in out["changed_words"]
    assert ws.apply_key("main") == "BAT | BAR"
    assert "verdict" not in out["score_delta"]
    assert out["agenda_item"]["status"] == "applied"
    assert out["agenda_item"]["last_result"]["mappings_set"] == 1


def test_act_apply_word_repair_dry_run_restores_branch_and_agenda():
    raw = "ABC | ABD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"BAT", "BAR"},
        word_list=["BAT", "BAR"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("C"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("T"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("R"))

    out = ex._tool_act_apply_word_repair({
        "branch": "main",
        "decoded_word": "CAT",
        "target_word": "BAT",
        "dry_run": True,
    })

    assert out["status"] == "preview"
    assert out["dry_run"] is True
    assert out["mappings"] == {"A": "B"}
    assert out["undo_mappings"] == {"A": "C"}
    assert ws.apply_key("main") == "CAT | CAR"
    assert ex.repair_agenda == []
    assert {"index": 0, "before": "CAT", "after": "BAT"} in out["changed_words"]


def test_word_repair_blocks_repeated_symbol_conflict_before_mutating():
    raw = "MKGMAQ"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"PLURES", "RLURES"},
        word_list=["PLURES", "RLURES"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    for cipher_sym, plain in {
        "M": "R",
        "K": "L",
        "G": "U",
        "A": "E",
        "Q": "S",
    }.items():
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(plain))

    plan = ex._tool_decode_plan_word_repair({
        "branch": "main",
        "decoded_word": "RLURES",
        "target_word": "PLURES",
    })

    assert plan["applicable"] is False
    assert plan["conflicts"]
    assert "same cipher symbol appears multiple times" in plan["conflicts"][0]["reason"]
    assert plan["agenda_item"]["status"] == "blocked"
    assert "suggested_call" not in plan

    applied = ex._tool_act_apply_word_repair({
        "branch": "main",
        "decoded_word": "RLURES",
        "target_word": "PLURES",
    })

    assert applied["error"] == "Word repair is not directly applicable."
    assert ws.apply_key("main") == "RLURES"


def test_decode_plan_word_repair_menu_compares_options_without_agenda_mutation():
    raw = "MKGMAQ"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"PLURES", "RLURES"},
        word_list=["PLURES", "RLURES"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    for cipher_sym, plain in {
        "M": "R",
        "K": "L",
        "G": "U",
        "A": "E",
        "Q": "S",
    }.items():
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(plain))

    menu = ex._tool_decode_plan_word_repair_menu({
        "branch": "main",
        "decoded_word": "RLURES",
        "target_words": ["PLURES", "RLURES"],
    })

    assert menu["status"] == "ok"
    assert menu["current_decoded"] == "RLURES"
    assert len(menu["options"]) == 2
    plures = next(opt for opt in menu["options"] if opt["target_word"] == "PLURES")
    rlures = next(opt for opt in menu["options"] if opt["target_word"] == "RLURES")
    assert plures["applicable"] is False
    assert plures["conflicts"]
    assert plures["recommendation"].startswith("do_not_apply_directly")
    assert "suggested_call" not in plures
    assert plures["effect_summary"]["changed_words_preview"][0]["after"] == "PLUPES"
    assert rlures["recommendation"].startswith("already_matches")
    assert ex.repair_agenda == []


def test_branch_cards_surface_basin_status(monkeypatch):
    ex = _executor_for("ABC")

    monkeypatch.setattr(
        ex,
        "_branch_basin_status",
        lambda branch: {
            "status": "word_islands_only",
            "repair_policy": "search_before_local_repair",
            "reason": "test basin warning",
            "suggested_next_tools": ["search_transform_homophonic"],
        },
    )

    cards = ex._tool_workspace_branch_cards({"branch": "main"})

    assert cards["cards"][0]["basin"]["status"] == "word_islands_only"
    assert cards["cards"][0]["basin"]["repair_policy"] == "search_before_local_repair"
    assert "basin status" in cards["note"]


def test_decode_plan_word_repair_menu_warns_on_bad_basin(monkeypatch):
    raw = "ABC | ABD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"BAT", "BAR"},
        word_list=["BAT", "BAR"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    for cipher_sym, plain in {
        "A": "C",
        "B": "A",
        "C": "T",
        "D": "R",
    }.items():
        ex.workspace.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(plain))

    monkeypatch.setattr(
        ex,
        "_branch_basin_status",
        lambda branch: {
            "status": "word_islands_only",
            "repair_policy": "search_before_local_repair",
            "reason": "isolated dictionary words without a coherent clause",
            "suggested_next_tools": ["search_transform_homophonic"],
        },
    )

    menu = ex._tool_decode_plan_word_repair_menu({
        "branch": "main",
        "decoded_word": "CAT",
        "target_words": ["BAT"],
    })

    assert menu["status"] == "ok"
    assert menu["basin_warning"]["status"] == "word_islands_only"
    option = menu["options"][0]
    assert option["applicable"] is True
    assert option["recommendation"].startswith("search_before_apply")
    assert "allow_bad_basin_repair=true" in option["recommendation"]
    assert ex.repair_agenda == []


def test_act_apply_word_repair_blocks_bad_basin_without_override(monkeypatch):
    raw = "ABC | ABD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"BAT", "BAR"},
        word_list=["BAT", "BAR"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    for cipher_sym, plain in {
        "A": "C",
        "B": "A",
        "C": "T",
        "D": "R",
    }.items():
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(plain))

    monkeypatch.setattr(
        ex,
        "_branch_basin_status",
        lambda branch: {
            "status": "word_islands_only",
            "repair_policy": "search_before_local_repair",
            "reason": "isolated dictionary words without a coherent clause",
            "suggested_next_tools": ["search_transform_homophonic"],
        },
    )

    out = ex._tool_act_apply_word_repair({
        "branch": "main",
        "decoded_word": "CAT",
        "target_word": "BAT",
    })

    assert out["status"] == "blocked"
    assert out["reason"] == "bad_basin_word_repair_blocked"
    assert ws.apply_key("main") == "CAT | CAR"
    assert out["agenda_item"]["status"] == "blocked"
    assert "search_transform_homophonic" in out["suggested_next_tools"]


def test_act_apply_word_repair_allows_bad_basin_with_explicit_override(monkeypatch):
    raw = "ABC | ABD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"BAT", "BAR"},
        word_list=["BAT", "BAR"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    for cipher_sym, plain in {
        "A": "C",
        "B": "A",
        "C": "T",
        "D": "R",
    }.items():
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(plain))

    monkeypatch.setattr(
        ex,
        "_branch_basin_status",
        lambda branch: {
            "status": "word_islands_only",
            "repair_policy": "search_before_local_repair",
            "reason": "isolated dictionary words without a coherent clause",
            "suggested_next_tools": ["search_transform_homophonic"],
        },
    )

    out = ex._tool_act_apply_word_repair({
        "branch": "main",
        "decoded_word": "CAT",
        "target_word": "BAT",
        "allow_bad_basin_repair": True,
    })

    assert out["status"] == "ok"
    assert ws.apply_key("main") == "BAT | BAR"
    assert out["basin_before"]["status"] == "word_islands_only"


def test_repair_agenda_list_and_update_track_word_repair_state():
    raw = "ABC | ABD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"BAT", "BAR"},
        word_list=["BAT", "BAR"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("C"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("T"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("R"))

    plan = ex._tool_decode_plan_word_repair({
        "branch": "main",
        "cipher_word_index": 0,
        "target_word": "BAT",
    })
    listed = ex._tool_repair_agenda_list({"branch": "main"})

    assert listed["count"] == 1
    assert listed["unresolved_count"] == 1
    assert listed["items"][0]["id"] == plan["agenda_item"]["id"]

    updated = ex._tool_repair_agenda_update({
        "item_id": plan["agenda_item"]["id"],
        "status": "held",
        "notes": "Collateral damage needs review.",
    })

    assert updated["status"] == "ok"
    assert updated["agenda_item"]["status"] == "held"
    assert "Collateral" in updated["agenda_item"]["notes"]


def test_word_repair_flags_latin_orthography_shift_and_branch_card():
    raw = "AB | AC | AD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="la",
        word_set={"UT", "UM", "US", "VT", "VM", "VS"},
        word_list=["UT", "UM", "US", "VT", "VM", "VS"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("U"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("T"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("M"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("S"))

    out = ex._tool_act_apply_word_repair({
        "branch": "main",
        "cipher_word_index": 0,
        "target_word": "VT",
    })

    assert out["status"] == "ok"
    assert out["orthography_risks"]
    assert out["orthography_risks"][0]["type"] == "latin_orthography_shift"
    assert out["orthography_risks"][0]["from"] == "U"
    assert out["orthography_risks"][0]["to"] == "V"

    cards = ex._tool_workspace_branch_cards({"branch": "main"})
    card = cards["cards"][0]
    assert card["branch"] == "main"
    assert card["repair_counts"]["applied"] == 1
    assert card["orthography_risks"]
    assert "automated_preflight" in cards["note"]


def test_workspace_branch_cards_mark_automated_preflight_as_protected_baseline():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE"},
        word_list=["THE"],
        pattern_dict={},
    )
    ex.workspace.fork("automated_preflight", from_branch="main")

    cards = ex._tool_workspace_branch_cards({"branch": "automated_preflight"})

    assert cards["cards"][0]["protected_baseline"] is True


def test_workspace_fork_best_copies_automated_preflight_instead_of_empty_main():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE"},
        word_list=["THE"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ex.workspace.fork("automated_preflight", from_branch="main")
    ex.workspace.set_mapping("automated_preflight", alpha.id_for("A"), pt.id_for("T"))
    ex.workspace.set_mapping("automated_preflight", alpha.id_for("B"), pt.id_for("H"))
    ex.workspace.set_mapping("automated_preflight", alpha.id_for("C"), pt.id_for("E"))
    ex.workspace.tag("automated_preflight", "automated_preflight")
    ex.workspace.tag("automated_preflight", "no_llm")

    out = ex._tool_workspace_fork_best({"new_name": "repair_preflight"})

    assert out["status"] == "ok"
    assert out["source_branch"] == "automated_preflight"
    assert out["parent"] == "automated_preflight"
    assert out["inherited_mapped_count"] == 3
    assert len(ex.workspace.get_branch("repair_preflight").key) == 3


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
    assert "workspace_fork_best" in prompt
    assert "defaults to `main`" in prompt
    # Generalised dict_rate guidance — no fixed Latin threshold
    assert "0.15" not in prompt
    assert "Declare on reading" in prompt
    assert "partial is **not** better than continued work" in prompt
    assert "partial solution is always better" not in prompt


def test_compact_system_prompt_style():
    """Compact style replaces the verbose toolkit with a sequencing cheatsheet
    and is shorter, but retains all critical sequencing rules."""
    full = get_system_prompt("en", "full")
    compact = get_system_prompt("en", "compact")

    # Compact must be meaningfully shorter
    assert len(compact) < len(full) - 5000, (
        f"compact ({len(compact)}) should be ≥5000 chars shorter than full ({len(full)})"
    )
    # Critical sequencing rules still present in compact
    assert "search_anneal" in compact
    assert "preserve_existing" in compact
    assert "search_transform" in compact
    assert "search_automated_solver" in compact
    # Compact does NOT contain the long Quagmire-style narrative
    assert "Quagmire-style" not in compact
    # Full still has the verbose Quagmire guidance
    assert "Quagmire-style" in full
    # Both styles retain the core reading-first discipline
    assert "Declare on reading" in compact
    assert "reading is authoritative" in compact


def test_compact_is_default_for_ollama():
    """_resolve_system_prompt_style returns 'compact' for --provider ollama."""
    import argparse
    import sys
    sys.path.insert(0, "src")
    from cli import _resolve_system_prompt_style

    def make_args(provider=None, model=None, style="auto"):
        ns = argparse.Namespace()
        ns.provider = provider
        ns.model = model
        ns.system_prompt_style = style
        return ns

    # auto + ollama → compact
    assert _resolve_system_prompt_style(make_args(provider="ollama")) == "compact"
    # auto + anthropic → full
    assert _resolve_system_prompt_style(make_args(provider="anthropic")) == "full"
    # explicit full overrides auto
    assert _resolve_system_prompt_style(make_args(provider="ollama", style="full")) == "full"
    # explicit compact always wins
    assert _resolve_system_prompt_style(make_args(provider="anthropic", style="compact")) == "compact"


def test_workspace_panel_does_not_encourage_early_partial_declaration():
    ex = _executor_for("ABC DEF", separator=" ")
    panel = build_workspace_panel(
        ex.workspace,
        iteration=10,
        max_iterations=30,
        language="en",
        word_set=ex.word_set,
    )

    assert "A partial declaration is not better than continued work" in panel
    assert "5% beats 0%" not in panel
    assert "partial solution is always better" not in panel


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


def test_initial_context_can_include_cipher_diagnostic_preflight():
    msg = initial_context(
        cipher_display="LXFOPVEFRNHR",
        alphabet_symbols=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        total_tokens=12,
        total_words=1,
        ic_value=0.038,
        language="en",
        cipher_id_context="Top suspicions: periodic_polyalphabetic=0.72",
    )

    assert "Cipher-diagnostic preflight" in msg
    assert "periodic_polyalphabetic=0.72" in msg
    assert "Treat this as evidence, not a verdict" in msg


def test_system_prompt_routes_from_measured_facts_to_homophonic_solver():
    system = get_system_prompt("en")

    assert "opening measured facts" in system
    assert "many-symbol alphabet and no word boundaries" in system
    assert "search_homophonic_anneal" in system
    assert "search_automated_solver" in system
    assert "automated_preflight" in system
    assert "Do not spend early turns re-measuring facts" in system


def test_system_prompt_includes_unknown_cipher_mode_workflow():
    system = get_system_prompt("en")

    assert "Unknown-cipher discipline" in system
    assert "observe_cipher_id" in system
    assert "workspace_create_hypothesis_branch" in system
    assert "workspace_update_hypothesis" in system
    assert "workspace_reject_hypothesis" in system
    assert "Local spelling and boundary repairs are for near-solves" in system


def test_system_prompt_distinguishes_pure_and_mixed_transposition_workflows():
    system = get_system_prompt("en")

    assert "Make the pure-vs-mixed transposition decision explicit" in system
    assert "`search_pure_transposition`" in system
    assert "`search_review_pure_transposition_finalists`" in system
    assert "`act_install_pure_transposition_finalists`" in system
    assert "`search_transform_homophonic`" in system
    assert "word islands" in system
    assert "switch the cipher-mode hypothesis" in system


def test_system_prompt_mentions_scoped_benchmark_context_tools():
    system = get_system_prompt("en")

    assert "`benchmark_*`" in system
    assert "inspect_benchmark_context" in system
    assert "list_related_records" in system
    assert "inspect_related_transcription" in system
    assert "inspect_related_solution" in system
    assert "target record's solution is never" in system


def test_workspace_panel_shows_all_words_for_compact_pages():
    symbols = [f"S{i:02d}" for i in range(78)]
    raw = " | ".join(symbols)
    ct = CipherText(raw=raw, alphabet=Alphabet(symbols), separator=" | ")
    ws = Workspace(ct)

    panel = build_workspace_panel(
        ws,
        iteration=1,
        language="en",
        word_set=set(),
        max_iterations=10,
    )

    assert "words 0..77 (all)" in panel
    assert "S77" in panel


class _InspectThenDeclareAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        call = len(self.messages_seen)
        if call == 1:
            content = [
                SimpleNamespace(
                    type="tool_use",
                    id="show1",
                    name="decode_show",
                    input={"branch": "main"},
                )
            ]
        elif call == 2:
            content = [
                SimpleNamespace(
                    type="tool_use",
                    id="cards",
                    name="workspace_branch_cards",
                    input={},
                )
            ]
        else:
            content = [
                SimpleNamespace(
                    type="tool_use",
                    id="declare",
                    name="meta_declare_solution",
                    input={
                        "branch": "main",
                        "rationale": "Inspection is complete.",
                        "self_confidence": 0.5,
                        "reading_summary": "Toy continuation.",
                        "further_iterations_helpful": False,
                        "further_iterations_note": "No further work needed.",
                    },
                )
            ]
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=content,
        )


def test_read_only_inspection_can_continue_inside_one_outer_iteration():
    alpha = Alphabet.from_text("ABC")
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _InspectThenDeclareAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=5,
        cipher_id="unit",
    )

    assert artifact.status == "solved"
    assert artifact.solution is not None
    assert artifact.solution.declared_at_iteration == 1
    assert [tc.iteration for tc in artifact.tool_calls] == [1, 1, 1]
    assert sum(
        1 for event in artifact.loop_events
        if event.event == "inspection_sandbox_continue"
    ) == 2
    assert INSPECTION_SANDBOX_CONTINUE_PREFLIGHT in str(api.messages_seen[1])


class _InterruptingAPI:
    model = "claude-sonnet-4-6"

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        raise KeyboardInterrupt


def test_run_v2_returns_partial_artifact_on_keyboard_interrupt():
    alpha = Alphabet.from_text("ABCABC")
    ct = CipherText(raw="ABCABC", alphabet=alpha, separator=None)

    artifact = run_v2(
        cipher_text=ct,
        claude_api=_InterruptingAPI(),  # type: ignore[arg-type]
        language="en",
        max_iterations=5,
        cipher_id="interrupt_smoke",
    )

    assert artifact.status == "stopped"
    assert "Interrupted by user" in artifact.error_message
    assert artifact.finished_at > 0
    assert artifact.branches
    assert artifact.branches[0].name == "main"
    assert artifact.messages
    assert any(event.event == "interrupted" for event in artifact.loop_events)


class _ServedModelAPI:
    """Fake ClaudeAPI whose response advertises a (possibly different) served model."""

    def __init__(self, requested: str, served: str) -> None:
        self.model = requested
        self._served = served

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        return SimpleNamespace(
            model=self._served,
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=[SimpleNamespace(type="text", text="thinking, no declaration yet.")],
        )


def test_run_v2_records_served_model_without_gate_when_matching():
    # Served model matches the requested model -> recorded, no safety gate (F7).
    alpha = Alphabet.from_text("ABCABC")
    ct = CipherText(raw="ABCABC", alphabet=alpha, separator=None)
    api = _ServedModelAPI("claude-sonnet-4-6", "claude-sonnet-4-6-20260101")
    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=1,
        cipher_id="served_ok",
    )
    assert artifact.served_models == ["claude-sonnet-4-6-20260101"]
    assert artifact.safety_gate_fired is False


def test_run_v2_flags_safety_gate_when_served_model_differs():
    # Requested Fable but served Opus -> the safety gate fired; flag it (F7).
    alpha = Alphabet.from_text("ABCABC")
    ct = CipherText(raw="ABCABC", alphabet=alpha, separator=None)
    api = _ServedModelAPI("claude-fable-5", "claude-opus-4-8")
    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=1,
        cipher_id="served_gate",
    )
    assert artifact.served_models == ["claude-opus-4-8"]
    assert artifact.safety_gate_fired is True


def test_workspace_panel_shows_metadata_decoded_branches():
    alpha = Alphabet.from_text("ABCABC")
    workspace = Workspace(CipherText(raw="ABCABC", alphabet=alpha, separator=None))
    workspace.fork("quag3_candidate", from_branch="main")
    branch = workspace.get_branch("quag3_candidate")
    branch.metadata.update({
        "cipher_mode": "quagmire3",
        "key_type": "QuagmireKey",
        "decoded_text": "THEOLDMANANDTHESEA",
    })

    panel = build_workspace_panel(
        workspace,
        iteration=2,
        language="en",
        word_set={"THE", "OLD", "MAN", "AND", "SEA"},
    )

    assert "Branch `quag3_candidate`" in panel
    assert "key_type=QuagmireKey" in panel
    assert "THEOLDMANANDTHESEA" in panel


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


class _HypothesisThenDeclareAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="hyp1",
                    name="workspace_create_hypothesis_branch",
                    input={
                        "new_name": "hyp_vig",
                        "cipher_mode": "periodic_polyalphabetic",
                        "rationale": "Depressed IC and periodic evidence suggest Vigenere-family testing.",
                        "mode_confidence": "medium",
                    },
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="hyp2",
                    name="workspace_update_hypothesis",
                    input={
                        "branch": "hyp_vig",
                        "mode_status": "active",
                        "mode_confidence": "high",
                        "evidence": "Standard periodic diagnostics should be tried before word repair.",
                        "counter_evidence": "No plaintext basin yet.",
                        "next_recommended_action": "Run search_periodic_polyalphabetic.",
                    },
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="next",
                    name="workspace_hypothesis_next_steps",
                    input={"branch": "hyp_vig"},
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="cards",
                    name="workspace_branch_cards",
                    input={},
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="declare",
                    name="meta_declare_solution",
                    input={
                        "branch": "hyp_vig",
                        "rationale": "This is a workflow smoke test declaration.",
                        "self_confidence": 0.75,
                        "reading_summary": "Hypothesis bookkeeping smoke test.",
                        "further_iterations_helpful": False,
                        "further_iterations_note": "No further work needed.",
                    },
                ),
            ],
        )


def test_agent_artifact_records_cipher_id_report_and_hypothesis_trail():
    alpha = Alphabet.from_text("LXFOPVEFRNHR")
    ct = CipherText(raw="LXFOPVEFRNHR", alphabet=alpha, separator=None)
    api = _HypothesisThenDeclareAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=10,
        cipher_id="hypothesis_smoke",
    )

    assert artifact.status == "solved"
    assert artifact.cipher_id_report is not None
    assert "suspicion_scores" in artifact.cipher_id_report
    first_message = str(api.messages_seen[0])
    assert "Cipher-diagnostic preflight" in first_message
    assert artifact.cipher_hypotheses
    hyp = next(row for row in artifact.cipher_hypotheses if row["branch"] == "hyp_vig")
    assert hyp["cipher_mode"] == "periodic_polyalphabetic"
    assert hyp["mode_confidence"] == "high"
    assert hyp["mode_counter_evidence"] == "No plaintext basin yet."
    branch = next(row for row in artifact.branches if row.name == "hyp_vig")
    assert branch.metadata["next_recommended_action"] == "Run search_periodic_polyalphabetic."


def test_substitution_repair_blocks_on_periodic_hypothesis_unless_overridden():
    ex = _executor_for("ABC", separator=None)
    ex._tool_workspace_create_hypothesis_branch({
        "new_name": "hyp_poly",
        "cipher_mode": "periodic_polyalphabetic",
        "rationale": "Periodic diagnostics are stronger than monoalphabetic evidence.",
    })

    blocked = ex._tool_act_set_mapping({
        "branch": "hyp_poly",
        "cipher_symbol": "A",
        "plain_letter": "T",
    })

    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "mode_mismatch_substitution_repair_blocked"
    assert blocked["cipher_mode"] == "periodic_polyalphabetic"
    assert "act_set_periodic_shift" in blocked["suggested_next_tools"]

    accepted = ex._tool_act_set_mapping({
        "branch": "hyp_poly",
        "cipher_symbol": "A",
        "plain_letter": "T",
        "allow_mode_mismatch_repair": True,
    })

    assert accepted["status"] == "ok"
    assert accepted["mapping"] == "A -> T"


def test_hypothesis_next_steps_marks_tried_tools_and_next_pending_action():
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_poly",
        "cipher_mode": "periodic_polyalphabetic",
        "rationale": "Depressed IC suggests periodic testing.",
    })

    first = json.loads(ex.execute("workspace_hypothesis_next_steps", {
        "branch": "hyp_poly",
    }))
    report = first["reports"][0]
    assert report["next_step"]["tool"] == "observe_periodic_ic"
    assert "observe_periodic_ic" in report["pending_tools"]
    menu = report["tool_menu"]
    assert "search_periodic_polyalphabetic" in menu["foreground_tools"]
    assert "act_set_periodic_shift" in menu["foreground_tools"]
    assert "act_set_mapping" in menu["discouraged_tools"]
    assert "workspace_reject_hypothesis" in menu["escape_tools"]

    ex.execute("observe_periodic_ic", {"branch": "hyp_poly", "max_period": 6})
    second = json.loads(ex.execute("workspace_hypothesis_next_steps", {
        "branch": "hyp_poly",
    }))
    report = second["reports"][0]
    assert "observe_periodic_ic" in report["already_tried_tools"]
    assert report["next_step"]["tool"] == "observe_kasiski"


def test_hypothesis_next_steps_returns_transform_mode_tool_menu():
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_transform",
        "cipher_mode": "transposition_homophonic",
        "rationale": "Word islands suggest order plus homophonic search.",
    })

    out = json.loads(ex.execute("workspace_hypothesis_next_steps", {
        "branch": "hyp_transform",
    }))

    menu = out["reports"][0]["tool_menu"]
    assert "observe_transform_suspicion" in menu["foreground_tools"]
    assert "search_transform_homophonic" in menu["foreground_tools"]
    assert "act_rate_transform_finalist" in menu["foreground_tools"]
    assert "act_set_mapping" in menu["discouraged_tools"]


def test_hypothesis_next_steps_recommends_null_masks_from_structure_not_language():
    ex = _homophonic_executor()
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_hom",
        "cipher_mode": "homophonic_substitution",
        "rationale": "Large symbol inventory and coarse boundaries suggest homophonic search.",
    })

    out = json.loads(ex.execute("workspace_hypothesis_next_steps", {
        "branch": "hyp_hom",
    }))

    report = out["reports"][0]
    guidance = report["null_mask_guidance"]
    assert guidance["applies"] is True
    assert guidance["already_tried"] is False
    assert guidance["overcomplete_alphabet"] is True
    assert guidance["coarse_or_missing_boundaries"] is True
    assert "target language by itself is not evidence" in guidance["not_language_assumption"]
    assert guidance["suggested_args"]["homophonic_refinement"] == "null_masks"
    assert guidance["review_args"]["count"] == 12
    assert "top 8 scalar-validation candidates" in guidance["reading_instruction"]
    assert report["next_step"]["tool"] == "search_automated_solver"
    assert report["next_step"]["status"] == "pending_structural_refinement"
    assert report["next_step"]["suggested_args"]["homophonic_budget"] == "screen"


def test_transform_and_unsolved_block_until_null_masks_are_tried():
    ex = _homophonic_executor()
    ex.max_iterations = 25
    ex._current_iteration = 5
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_hom",
        "cipher_mode": "homophonic_substitution",
        "rationale": "Large symbol inventory and coarse boundaries suggest homophonic search.",
    })

    transform = ex._tool_search_transform_homophonic({
        "branch": "hyp_hom",
        "profile": "medium",
    })

    assert transform["status"] == "blocked"
    assert transform["reason"] == "null_mask_structural_refinement_pending"
    assert transform["blocked_tool"] == "search_transform_homophonic"
    assert transform["suggested_args"]["homophonic_refinement"] == "null_masks"

    unsolved = ex._tool_meta_declare_unsolved({
        "rationale": "No coherent reading yet.",
        "best_branch": "hyp_hom",
        "branches_considered": ["hyp_hom"],
        "reading_summary": "Only word islands.",
        "further_iterations_helpful": False,
    })

    assert unsolved["status"] == "blocked"
    assert unsolved["reason"] == "null_mask_workflow_pending_before_unsolved"
    assert unsolved["pending_null_mask_work"][0]["reason"] == "null_mask_structural_refinement_pending"
    assert "search_automated_solver" in unsolved["suggested_next_tools"]


def test_hypothesis_next_steps_does_not_recommend_null_masks_for_german_alone():
    raw = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="de",
        word_set={"DER", "DIE", "DAS", "UND"},
        word_list=["DER", "DIE", "DAS", "UND"],
        pattern_dict={},
    )
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_hom",
        "cipher_mode": "homophonic_substitution",
        "rationale": "Testing that language alone does not trigger null masks.",
    })

    out = json.loads(ex.execute("workspace_hypothesis_next_steps", {
        "branch": "hyp_hom",
    }))

    guidance = out["reports"][0]["null_mask_guidance"]
    assert guidance["applies"] is False
    assert guidance["overcomplete_alphabet"] is False
    assert guidance["coarse_or_missing_boundaries"] is True
    assert "target language by itself is not evidence" in guidance["not_language_assumption"]


def test_hypothesis_next_steps_returns_quagmire_tool_menu():
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_quag",
        "cipher_mode": "quagmire3",
        "rationale": "Periodic evidence remains but ordinary Vigenere failed.",
    })

    out = json.loads(ex.execute("workspace_hypothesis_next_steps", {
        "branch": "hyp_quag",
    }))

    report = out["reports"][0]
    tools = [step["tool"] for step in report["playbook"]]
    menu = report["tool_menu"]
    assert "search_quagmire3_keyword_alphabet" in tools
    assert "search_quagmire3_keyword_alphabet" in menu["foreground_tools"]
    assert "search_periodic_polyalphabetic" in menu["foreground_tools"]
    assert "act_set_mapping" in menu["discouraged_tools"]
    guidance = report["quagmire_budget_guidance"]
    assert guidance["tool"] == "search_quagmire3_keyword_alphabet"
    assert guidance["profiles"][1]["suggested_args"]["estimate_only"] is True
    assert guidance["profiles"][1]["suggested_args"]["hillclimbs"] == 10000
    assert "context-seeded" in guidance["seeded_keyword_warning"]


def test_context_supported_keyed_vigenere_blocks_premature_poly_rejection():
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.benchmark_context = ScopedBenchmarkContext(
        policy="historical",
        injected_layers=[
            {
                "record_id": "kryptos_k2",
                "layer": "standard",
                "label": "Standard cipher metadata",
                "contains_cipher_type_hint": True,
                "text": (
                    "K2 is an English alphabetic Kryptos section classified "
                    "here as a keyed Vigenere-style polyalphabetic cipher."
                ),
            }
        ],
        target_record_ids=["kryptos_k2"],
    )
    created = json.loads(ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_poly",
        "cipher_mode": "keyed_tableau_polyalphabetic",
        "rationale": "Context and IC suggest keyed Vigenere.",
        "evidence_source": "benchmark_context",
    }))
    assert created["context_mode_prior"]["prior"] == "keyed_tableau_polyalphabetic"
    assert created["evidence_source"] == "benchmark_context"

    blocked = json.loads(ex.execute("workspace_reject_hypothesis", {
        "branch": "hyp_poly",
        "reason": "Ordinary Vigenere failed.",
    }))

    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "pending_required_tools_before_rejection"
    assert blocked["pending_required_tools"] == ["search_quagmire3_keyword_alphabet"]
    assert "Plain Vigenere failure" in blocked["note"]


def test_workspace_snapshot_uses_metadata_decoded_text_for_periodic_branches():
    raw = "ABCABCABCABC"
    alpha = Alphabet.from_text(raw)
    workspace = Workspace(CipherText(raw=raw, alphabet=alpha, separator=None))
    branch = workspace.fork("quag3_candidate", from_branch="main")
    branch.metadata.update({
        "cipher_mode": "quagmire3",
        "key_type": "QuagmireKey",
        "decoded_text": "THEOLDTHEOLD",
        "decoded_text_source": "unit_test",
    })

    payload = _workspace_snapshot_payload(
        workspace,
        "en",
        {"THE", "OLD"},
        {"THE": 1, "OLD": 2},
        iteration=3,
        max_iterations=10,
    )

    assert payload["branch"] == "quag3_candidate"
    assert payload["decryption"] == "THEOLDTHEOLD"
    assert payload["decryption_preview"].startswith("THEOLD")

    workspace.set_word_spans("quag3_candidate", [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
    ])
    segmented_payload = _workspace_snapshot_payload(
        workspace,
        "en",
        {"THE", "OLD"},
        {"THE": 1, "OLD": 2},
        iteration=4,
        max_iterations=10,
    )

    assert segmented_payload["decryption"] == "THE OLD THE OLD"
    assert segmented_payload["decryption_preview"].startswith("THE OLD")


def test_periodic_family_coverage_debt_prioritizes_quagmire_after_plain_search():
    # K2-like statistics: low IC, flat alphabet, and a period-8 bump. No
    # benchmark context is used here, so this is the zero-context escalation
    # guard rather than a context-prior guard.
    raw = (
        "VFPJUDEEHZWETZYVGWHKKQETGFQJNCEGGWHKK?DQMCPFQZDQMMIAGPFXHQRLGTIMVMZJANQLVKQEDAGDVFRPJUNGEUNA"
        "QZGZLECGYUXUEENJTBJLBQCRTBJDFHRRYIZETKZEMVDUFKSJHKFWHKUWQLSZFTIHHDDDUVH?DWKBFUFPWNTDFIYCUQZERE"
        "EVLDKFEZMOQQJLTTUGSYQPFEUNLAVIDXFLGGTEZ?FKZBSFDQVGOGIPUFXHHDRKFFHQNTGPUAECNUVPDJMQCLQUMUNEDFQE"
        "LZZVRRGKFFVOEEXBDMVPNFQXEZLGREDNQFMPNZGLFLPMRJQYALMGNUVPDXVKPDQUMEBEDMHDAFMJGZNUPLGEWJLLAETG"
    )
    alpha = Alphabet.from_text(raw)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(CipherText(raw=raw, alphabet=alpha, separator=None)),
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_poly",
        "cipher_mode": "periodic_polyalphabetic",
        "rationale": "Low IC and period evidence suggest a periodic family.",
    })
    ex.call_log.append(ToolCall(
        iteration=1,
        tool_name="search_periodic_polyalphabetic",
        tool_use_id="plain-vig",
        arguments={"branch": "hyp_poly"},
        result="{}",
    ))
    ex.call_log.append(ToolCall(
        iteration=2,
        tool_name="search_quagmire3_keyword_alphabet",
        tool_use_id="estimate-quag",
        arguments={"branch": "hyp_poly", "estimate_only": True},
        result='{"status":"estimated"}',
    ))

    out = json.loads(ex.execute("workspace_hypothesis_next_steps", {
        "branch": "hyp_poly",
    }))
    report = out["reports"][0]

    assert report["next_step"]["tool"] == "search_quagmire3_keyword_alphabet"
    assert report["pending_required_tools"] == ["search_quagmire3_keyword_alphabet"]
    assert report["family_coverage_debt"]
    assert report["statistical_family_prior"]["prior"] == "polyalphabetic_family"


def test_diagnostic_quagmire_search_does_not_clear_family_coverage():
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_quag",
        "cipher_mode": "keyed_tableau_polyalphabetic",
        "rationale": "Context says this is a keyed-tableau family.",
        "evidence_source": "benchmark_context",
    })
    ex.call_log.append(ToolCall(
        iteration=2,
        tool_name="search_quagmire3_keyword_alphabet",
        tool_use_id="tiny-quag",
        arguments={"branch": "hyp_quag"},
        result='{"status":"completed","nominal_proposals":40000,"budget_class":"diagnostic"}',
    ))

    out = json.loads(ex.execute("workspace_hypothesis_next_steps", {
        "branch": "hyp_quag",
    }))
    report = out["reports"][0]

    assert report["pending_required_tools"] == ["search_quagmire3_keyword_alphabet"]
    assert report["quagmire_search_status"]["budget_class"] == "diagnostic"
    assert report["quagmire_search_status"]["sufficient_to_reject"] is False


def test_moderate_quagmire_search_clears_family_coverage_for_installed_branch():
    raw = "ABC" * 40
    ex = _executor_for(raw, separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(12)
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_quag",
        "cipher_mode": "keyed_tableau_polyalphabetic",
        "rationale": "Context says this is a keyed-tableau family.",
        "evidence_source": "benchmark_context",
    })
    branch = ex.workspace.fork("quag3_KRYPTOS_ABSCISSA_1", from_branch="hyp_quag")
    branch.metadata.update({
        "cipher_mode": "quagmire3",
        "mode_status": "active",
        "key_type": "QuagmireKey",
        "decoded_text": "THEOLD" * 20,
        "search_metadata": {
            "engine": "rust_shotgun",
            "nominal_proposals": 50_000_000,
            "budget_class": "broad",
        },
    })
    ex.workspace.tag("quag3_KRYPTOS_ABSCISSA_1", "hypothesis")
    ex.execute("workspace_hypothesis_next_steps", {
        "branch": "quag3_KRYPTOS_ABSCISSA_1",
    })
    ex.execute("workspace_branch_cards", {})

    out = ex._tool_meta_declare_solution({
        "branch": "quag3_KRYPTOS_ABSCISSA_1",
        "rationale": "The Quagmire branch is coherent and the family search was broad.",
        "self_confidence": 0.98,
        "reading_summary": "The plaintext reads coherently.",
        "further_iterations_helpful": False,
        "further_iterations_note": "No further work needed.",
    })

    assert out["status"] == "blocked"
    assert out["reason"] == "word_boundary_pass_required"

    segmented = ex._tool_act_resegment_by_reading({
        "branch": "quag3_KRYPTOS_ABSCISSA_1",
        "proposed_text": " ".join(["THE", "OLD"] * 20),
    })

    assert segmented["status"] == "ok"
    assert segmented["decoded_preview"].startswith("THE | OLD")

    out = ex._tool_meta_declare_solution({
        "branch": "quag3_KRYPTOS_ABSCISSA_1",
        "rationale": "The Quagmire branch is coherent and the family search was broad.",
        "self_confidence": 0.98,
        "reading_summary": "The plaintext reads coherently.",
        "further_iterations_helpful": False,
        "further_iterations_note": "No further work needed.",
    })

    assert out["status"] == "ok"
    assert out["accepted"] is True


def test_word_boundary_block_fires_for_pure_transposition_branch():
    """A pure-transposition branch (empty key, token_order set, decoded_text in
    metadata) should be blocked at declaration until act_resegment_by_reading
    is called — same as the polyalphabetic path."""
    # 100-char no-boundary cipher
    raw = "ABCDE" * 20
    ex = _executor_for(raw, separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(5)

    # Fork a branch and install it as a pure-transposition result:
    # token_order = identity (simplest valid permutation), decoded_text = readable text.
    ex.workspace.fork("transpos_branch", from_branch="main")
    pipeline = {"steps": [{"name": "Reverse", "data": {"rangeStart": 0, "rangeEnd": 99}}]}
    ex.workspace.apply_transform_pipeline("transpos_branch", pipeline)
    ex.workspace.get_branch("transpos_branch").metadata["decoded_text"] = "EDCBA" * 20
    ex.execute("workspace_branch_cards", {})

    out = ex._tool_meta_declare_solution({
        "branch": "transpos_branch",
        "rationale": "Pure transposition solved; letters are correct.",
        "self_confidence": 0.97,
        "reading_summary": "The plaintext reads coherently.",
        "further_iterations_helpful": False,
        "further_iterations_note": "",
    })

    assert out["status"] == "blocked"
    assert out["reason"] == "word_boundary_pass_required"

    # After adding boundaries, declaration should be accepted.
    plain = "EDCBA" * 20
    spaced = " ".join(plain[i:i+5] for i in range(0, 100, 5))
    seg = ex._tool_act_resegment_by_reading({
        "branch": "transpos_branch",
        "proposed_text": spaced,
    })
    assert seg["status"] == "ok"

    out2 = ex._tool_meta_declare_solution({
        "branch": "transpos_branch",
        "rationale": "Pure transposition solved; letters are correct.",
        "self_confidence": 0.97,
        "reading_summary": "The plaintext reads coherently.",
        "further_iterations_helpful": False,
        "further_iterations_note": "",
    })
    assert out2["status"] == "ok"
    assert out2["accepted"] is True


def test_word_boundary_block_fires_for_no_boundary_substitution_branch():
    """A fully-mapped substitution branch on a no-boundary cipher is also
    blocked at declaration until a word-boundary overlay is applied."""
    # 100-char no-boundary cipher with 5 distinct symbols
    raw = "ABCDE" * 20
    ex = _executor_for(raw, separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(5)

    # Map all 5 symbols to real letters so decoded text is 100 non-'?' chars.
    pt_alpha = ex.workspace.plaintext_alphabet
    mapping = {
        ex.workspace.cipher_text.alphabet.id_for("A"): pt_alpha.id_for("H"),
        ex.workspace.cipher_text.alphabet.id_for("B"): pt_alpha.id_for("E"),
        ex.workspace.cipher_text.alphabet.id_for("C"): pt_alpha.id_for("L"),
        ex.workspace.cipher_text.alphabet.id_for("D"): pt_alpha.id_for("L"),
        ex.workspace.cipher_text.alphabet.id_for("E"): pt_alpha.id_for("O"),
    }
    ex.workspace.set_full_key("main", mapping)
    ex.execute("workspace_branch_cards", {})

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "All symbols mapped; reads as HELLO repeated.",
        "self_confidence": 0.95,
        "reading_summary": "The plaintext reads coherently.",
        "further_iterations_helpful": False,
        "further_iterations_note": "",
    })

    assert out["status"] == "blocked"
    assert out["reason"] == "word_boundary_pass_required"


def test_word_boundary_block_skipped_when_spans_already_set():
    """If word_spans have already been installed, the block should not fire."""
    raw = "ABCDEF" * 20   # 120 tokens, matches "THEOLD" * 20
    ex = _executor_for(raw, separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(5)

    ex.workspace.fork("poly_branch", from_branch="main")
    br = ex.workspace.get_branch("poly_branch")
    plain = "THEOLD" * 20   # 120 chars
    br.metadata["decoded_text"] = plain
    # Pre-install word spans (THE | OLD | THE | ...) so the block should not fire.
    spans = []
    for i in range(0, 120, 6):
        spans.append((i, i + 3))      # THE
        spans.append((i + 3, i + 6))  # OLD
    ex.workspace.set_word_spans("poly_branch", spans)
    ex.execute("workspace_branch_cards", {})

    out = ex._tool_meta_declare_solution({
        "branch": "poly_branch",
        "rationale": "Fully segmented.",
        "self_confidence": 0.95,
        "reading_summary": "The plaintext reads coherently.",
        "further_iterations_helpful": False,
        "further_iterations_note": "",
    })

    # Should NOT be blocked for word boundaries (may be blocked for other reasons,
    # but not word_boundary_pass_required).
    assert out.get("reason") != "word_boundary_pass_required"


def test_meta_declare_blocks_when_context_keyed_tableau_work_pending():
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(6)
    ex.benchmark_context = ScopedBenchmarkContext(
        policy="historical",
        injected_layers=[
            {
                "record_id": "kryptos_k2",
                "layer": "standard",
                "label": "Standard cipher metadata",
                "contains_cipher_type_hint": True,
                "text": (
                    "K2 is an English alphabetic Kryptos section classified "
                    "here as a keyed Vigenere-style polyalphabetic cipher."
                ),
            }
        ],
        target_record_ids=["kryptos_k2"],
    )
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_poly",
        "cipher_mode": "keyed_tableau_polyalphabetic",
        "rationale": "Context and IC suggest keyed Vigenere.",
        "evidence_source": "benchmark_context",
    })
    ex.execute("workspace_hypothesis_next_steps", {"branch": "hyp_poly"})
    ex.call_log.append(ToolCall(
        iteration=5,
        tool_name="search_periodic_polyalphabetic",
        tool_use_id="plain-vig",
        arguments={"branch": "hyp_poly"},
        result="{}",
    ))
    ex.execute("workspace_branch_cards", {})

    blocked = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Stopping with no coherent result.",
        "self_confidence": 0.12,
        "reading_summary": "No coherent plaintext recovered.",
        "further_iterations_helpful": False,
        "further_iterations_note": "No useful work remains.",
    })

    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "family_coverage_pending"
    assert "search_quagmire3_keyword_alphabet" in blocked["suggested_next_tools"]


def test_meta_declare_unsolved_terminates_without_solution():
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.set_max_iterations(10)
    ex.set_iteration(10)

    out = ex._tool_meta_declare_unsolved({
        "rationale": "No coherent plaintext branch emerged after available work.",
        "branches_considered": ["main"],
        "best_branch": "main",
        "reading_summary": "No coherent plaintext recovered.",
        "further_iterations_helpful": True,
        "further_iterations_note": "A larger search might be useful later.",
    })

    assert out["status"] == "ok"
    assert out["outcome"] == "unsolved"
    assert ex.terminated is True
    assert ex.solution is None
    assert ex.unsolved_declaration["best_branch"] == "main"


def test_context_keyed_tableau_blocks_off_family_transform_search():
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.benchmark_context = ScopedBenchmarkContext(
        policy="historical",
        injected_layers=[
            {
                "record_id": "kryptos_k2",
                "layer": "standard",
                "label": "Standard cipher metadata",
                "contains_cipher_type_hint": True,
                "text": (
                    "K2 is an English alphabetic Kryptos section classified "
                    "here as a keyed Vigenere-style polyalphabetic cipher."
                ),
            }
        ],
        target_record_ids=["kryptos_k2"],
    )
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_quag",
        "cipher_mode": "keyed_tableau_polyalphabetic",
        "rationale": "I read the benchmark context as keyed Vigenere/Kryptos-style.",
        "evidence_source": "benchmark_context",
    })

    blocked = json.loads(ex.execute("search_transform_homophonic", {
        "branch": "main",
        "profile": "small",
    }))

    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "context_cipher_family_mismatch"
    assert blocked["context_cipher_family"] == "keyed_tableau_polyalphabetic"
    assert "search_quagmire3_keyword_alphabet" in blocked["suggested_next_tools"]
    assert blocked["override_fields"]["override_context_cipher_family"] is True


def test_exposed_context_alone_does_not_hard_gate_tools():
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.benchmark_context = ScopedBenchmarkContext(
        policy="historical",
        injected_layers=[
            {
                "record_id": "kryptos_k2",
                "layer": "standard",
                "label": "Standard cipher metadata",
                "contains_cipher_type_hint": True,
                "text": (
                    "K2 is an English alphabetic Kryptos section classified "
                    "here as a keyed Vigenere-style polyalphabetic cipher."
                ),
            }
        ],
        target_record_ids=["kryptos_k2"],
    )

    assert ex._context_cipher_family_tool_block("search_transform_homophonic", {
        "branch": "main",
        "profile": "small",
    }) is None


def test_context_family_override_requires_explicit_rationale():
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.benchmark_context = ScopedBenchmarkContext(
        policy="historical",
        injected_layers=[
            {
                "record_id": "kryptos_k2",
                "layer": "standard",
                "label": "Standard cipher metadata",
                "contains_cipher_type_hint": True,
                "text": (
                    "K2 is an English alphabetic Kryptos section classified "
                    "here as a keyed Vigenere-style polyalphabetic cipher."
                ),
            }
        ],
        target_record_ids=["kryptos_k2"],
    )
    ex.execute("workspace_create_hypothesis_branch", {
        "new_name": "hyp_quag",
        "cipher_mode": "keyed_tableau_polyalphabetic",
        "rationale": "I read the benchmark context as keyed Vigenere/Kryptos-style.",
        "evidence_source": "benchmark_context",
    })

    blocked = ex._context_cipher_family_tool_block("search_transform_homophonic", {
        "branch": "main",
        "override_context_cipher_family": True,
        "context_override_rationale": "try transform",
    })
    assert blocked is not None
    assert blocked["reason"] == "context_override_rationale_required"

    allowed = ex._context_cipher_family_tool_block("search_transform_homophonic", {
        "branch": "main",
        "override_context_cipher_family": True,
        "context_override_rationale": (
            "The context-supported keyed-tableau route has produced only "
            "negative results, and independent evidence now suggests order "
            "transposition should be tested as a deliberate override."
        ),
    })
    assert allowed is None
    assert ex.context_family_overrides
    assert ex.context_family_overrides[0]["tool"] == "search_transform_homophonic"


def test_meta_declare_requires_hypothesis_next_steps_for_hypothesis_branch():
    ex = _executor_for("ABCABC", separator=None)
    ex._tool_workspace_create_hypothesis_branch({
        "new_name": "hyp_poly",
        "cipher_mode": "periodic_polyalphabetic",
        "rationale": "Periodic diagnostics should be tested.",
    })
    ex.execute("workspace_branch_cards", {})

    blocked = ex._tool_meta_declare_solution({
        "branch": "hyp_poly",
        "rationale": "Declaring a hypothesis branch without playbook review.",
        "self_confidence": 0.7,
        "reading_summary": "No real reading.",
        "further_iterations_helpful": False,
        "further_iterations_note": "No further work needed.",
    })

    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "hypothesis_next_steps_required"
    assert "workspace_hypothesis_next_steps" in blocked["suggested_next_tools"]

    ex.execute("workspace_hypothesis_next_steps", {"branch": "hyp_poly"})
    accepted = ex._tool_meta_declare_solution({
        "branch": "hyp_poly",
        "rationale": "Playbook reviewed for smoke-test declaration.",
        "self_confidence": 0.7,
        "reading_summary": "No real reading.",
        "further_iterations_helpful": False,
        "further_iterations_note": "No further work needed.",
    })

    assert accepted["status"] == "ok"
    assert accepted["accepted"] is True


class _RejectSwitchDeclareAPI:
    model = "claude-sonnet-4-6"

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="hyp_sub",
                    name="workspace_create_hypothesis_branch",
                    input={
                        "new_name": "hyp_sub",
                        "cipher_mode": "monoalphabetic_substitution",
                        "rationale": "Initial fallback hypothesis.",
                        "mode_confidence": "low",
                    },
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="reject_sub",
                    name="workspace_reject_hypothesis",
                    input={
                        "branch": "hyp_sub",
                        "reason": "Depressed IC and failed monoalphabetic evidence favor a periodic mode.",
                    },
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="hyp_poly",
                    name="workspace_create_hypothesis_branch",
                    input={
                        "new_name": "hyp_poly",
                        "cipher_mode": "periodic_polyalphabetic",
                        "rationale": "Switching to periodic polyalphabetic after rejecting monoalphabetic.",
                        "mode_confidence": "medium",
                    },
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="cards",
                    name="workspace_branch_cards",
                    input={},
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="next",
                    name="workspace_hypothesis_next_steps",
                    input={"branch": "hyp_poly"},
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="declare",
                    name="meta_declare_solution",
                    input={
                        "branch": "hyp_poly",
                        "rationale": "Smoke test: the active hypothesis switched modes.",
                        "self_confidence": 0.7,
                        "reading_summary": "Hypothesis switching smoke test.",
                        "further_iterations_helpful": False,
                        "further_iterations_note": "No further work needed.",
                    },
                ),
            ],
        )


class _NextStepsThenSearchAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.calls = 0

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.calls += 1
        if self.calls == 1:
            content = [
                SimpleNamespace(
                    type="tool_use",
                    id="hyp",
                    name="workspace_create_hypothesis_branch",
                    input={
                        "new_name": "hyp_poly",
                        "cipher_mode": "periodic_polyalphabetic",
                        "rationale": "Periodic evidence should be tested before substitution repair.",
                    },
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="next",
                    name="workspace_hypothesis_next_steps",
                    input={"branch": "hyp_poly"},
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="search",
                    name="search_periodic_polyalphabetic",
                    input={
                        "branch": "hyp_poly",
                        "periods": [3],
                        "variants": ["vigenere"],
                        "top_n": 1,
                        "install_top_n": 0,
                    },
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="update",
                    name="workspace_update_hypothesis",
                    input={
                        "branch": "hyp_poly",
                        "mode_status": "active",
                        "evidence": "Followed the periodic playbook through search_periodic_polyalphabetic.",
                        "next_recommended_action": "Read the search candidates and decide whether to install.",
                    },
                ),
            ]
        else:
            content = [
                SimpleNamespace(
                    type="tool_use",
                    id="cards",
                    name="workspace_branch_cards",
                    input={},
                ),
                SimpleNamespace(
                    type="tool_use",
                    id="declare",
                    name="meta_declare_solution",
                    input={
                        "branch": "hyp_poly",
                        "rationale": "Smoke test: followed hypothesis next-step playbook.",
                        "self_confidence": 0.7,
                        "reading_summary": "Hypothesis next-step smoke test.",
                        "further_iterations_helpful": False,
                        "further_iterations_note": "No further work needed.",
                    },
                ),
            ]
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=content,
        )


def test_agent_can_request_hypothesis_next_steps_then_follow_periodic_search():
    alpha = Alphabet.from_text("LXFOPVEFRNHR")
    ct = CipherText(raw="LXFOPVEFRNHR", alphabet=alpha, separator=None)

    artifact = run_v2(
        cipher_text=ct,
        claude_api=_NextStepsThenSearchAPI(),  # type: ignore[arg-type]
        language="en",
        max_iterations=10,
        cipher_id="hypothesis_next_steps_smoke",
    )

    tool_names = [call.tool_name for call in artifact.tool_calls]
    assert "workspace_hypothesis_next_steps" in tool_names
    assert "search_periodic_polyalphabetic" in tool_names
    hyp = next(row for row in artifact.cipher_hypotheses if row["branch"] == "hyp_poly")
    assert hyp["next_recommended_action"] == "Read the search candidates and decide whether to install."


def test_agent_artifact_records_rejected_and_switched_hypotheses():
    alpha = Alphabet.from_text("LXFOPVEFRNHR")
    ct = CipherText(raw="LXFOPVEFRNHR", alphabet=alpha, separator=None)

    artifact = run_v2(
        cipher_text=ct,
        claude_api=_RejectSwitchDeclareAPI(),  # type: ignore[arg-type]
        language="en",
        max_iterations=10,
        cipher_id="hypothesis_switch_smoke",
    )

    hypotheses = {row["branch"]: row for row in artifact.cipher_hypotheses}
    assert hypotheses["hyp_sub"]["mode_status"] == "rejected"
    assert hypotheses["hyp_sub"]["rejection_reason"].startswith("Depressed IC")
    assert hypotheses["hyp_poly"]["mode_status"] == "active"
    assert artifact.solution is not None
    assert artifact.solution.branch == "hyp_poly"


class _GatedThenDeclareAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []
        self.tools_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.tools_seen.append(tools or [])
        if len(self.messages_seen) == 1:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=3),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="bad_tool",
                        name="search_anneal",
                        input={"branch": "main"},
                    )
                ],
            )
        if len(self.messages_seen) == 3:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=12, output_tokens=4),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="cards",
                        name="workspace_branch_cards",
                        input={},
                    )
                ],
            )
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="declare",
                    name="meta_declare_solution",
                    input={
                        "branch": "main",
                        "rationale": "Best available branch after retry.",
                        "self_confidence": 0.2,
                    },
                )
            ],
        )


class _BoundaryCountRetryAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []
        self.tools_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.tools_seen.append(tools or [])
        if len(self.messages_seen) == 1:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=3),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="short_reading",
                        name="decode_validate_reading_repair",
                        input={
                            "branch": "automated_preflight",
                            "proposed_text": "A",
                        },
                    )
                ],
            )
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="fixed_reading",
                    name="act_resegment_by_reading",
                    input={
                        "branch": "automated_preflight",
                        "proposed_text": "AB C",
                    },
                )
            ],
        )


class _FinalGatedToolThenDeclareAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        if len(self.messages_seen) == 1:
            content = [
                SimpleNamespace(
                    type="tool_use",
                    id="late_boundary",
                    name="act_resegment_by_reading",
                    input={"branch": "main", "proposed_text": "ABC"},
                )
            ]
        else:
            content = [
                SimpleNamespace(
                    type="tool_use",
                    id="declare",
                    name="meta_declare_solution",
                    input={
                        "branch": "main",
                        "rationale": "Declared after final gated-tool retry.",
                        "self_confidence": 0.4,
                        "reading_summary": "Toy final declaration.",
                        "further_iterations_helpful": False,
                        "further_iterations_note": "No further work needed.",
                    },
                )
            ]
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=content,
        )


class _BoundaryCountRetryTwiceThenDeclareAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []
        self.tools_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.tools_seen.append(tools or [])
        if len(self.messages_seen) <= 2:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=3),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id=f"short_reading_{len(self.messages_seen)}",
                        name="decode_validate_reading_repair",
                        input={
                            "branch": "automated_preflight",
                            "proposed_text": "A",
                        },
                    )
                ],
            )
        if len(self.messages_seen) == 3:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=12, output_tokens=4),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="cards",
                        name="workspace_branch_cards",
                        input={},
                    )
                ],
            )
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="declare",
                    name="meta_declare_solution",
                    input={
                        "branch": "automated_preflight",
                        "rationale": "Declared after repeated projection retries.",
                        "self_confidence": 0.3,
                    },
                )
            ],
        )


class _SandboxContinueThenDeclareAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []
        self.tools_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.tools_seen.append(tools or [])
        if len(self.messages_seen) == 1:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=3),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="show",
                        name="decode_show",
                        input={"branch": "automated_preflight"},
                    )
                ],
            )
        if len(self.messages_seen) == 2:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=12, output_tokens=4),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="cards",
                        name="workspace_branch_cards",
                        input={},
                    )
                ],
            )
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="declare",
                    name="meta_declare_solution",
                    input={
                        "branch": "automated_preflight",
                        "rationale": "Declared after sandbox inspection.",
                        "self_confidence": 0.3,
                    },
                )
            ],
        )


class _FinalBookkeepingThenDeclareAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []
        self.tools_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.tools_seen.append(tools or [])
        if len(self.messages_seen) == 1:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=3),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="cards",
                        name="workspace_branch_cards",
                        input={},
                    )
                ],
            )
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="declare",
                    name="meta_declare_solution",
                    input={
                        "branch": "main",
                        "rationale": "Best branch after final cards.",
                        "self_confidence": 0.4,
                    },
                )
            ],
        )


def test_normalize_model_response_decouples_loop_from_provider_shape():
    raw = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            cache_read_input_tokens=3,
        ),
        content=[
            SimpleNamespace(type="text", text="Plan."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_1",
                name="decode_show",
                input={"branch": "main"},
            ),
        ],
    )

    response = normalize_model_response(raw)

    assert isinstance(response, ModelResponse)
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.cache_read_input_tokens == 3
    assert response.raw is raw
    assert response.content == [
        TextBlock(text="Plan."),
        ToolUseBlock(id="toolu_1", name="decode_show", input={"branch": "main"}),
    ]


def test_claude_model_provider_wraps_existing_api_without_loop_specifics():
    api = _FakeAPI()
    provider = ClaudeModelProvider(api)

    response = provider.send(messages=[], tools=[], system="", max_tokens=128)

    assert provider.provider_name == "anthropic"
    assert provider.model == api.model
    assert isinstance(response, ModelResponse)
    assert response.content == [TextBlock(text="I forgot to declare.")]


def test_provider_inference_and_defaults_cover_supported_families():
    assert infer_provider_from_model("claude-sonnet-4-6") == "anthropic"
    assert infer_provider_from_model("gpt-5.4-mini") == "openai"
    assert infer_provider_from_model("gemini-3-flash") == "gemini"
    assert infer_provider_from_model("claude-sonnet-4-6", "openai") == "openai"
    assert default_model_for_provider("openai").startswith("gpt-")
    assert default_model_for_provider("gemini").startswith("gemini-")


def test_openai_adapter_converts_anthropic_style_tool_history():
    messages = [
        {"role": "user", "content": "Initial text"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will inspect."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "decode_show",
                    "input": {"branch": "main"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": '{"ok": true}',
                },
                {"type": "text", "text": "Continue."},
            ],
        },
    ]

    converted = _messages_to_openai_chat(messages, system="System prompt")

    assert converted[0] == {"role": "system", "content": "System prompt"}
    assert converted[2]["role"] == "assistant"
    assert converted[2]["tool_calls"][0]["id"] == "toolu_1"
    assert converted[2]["tool_calls"][0]["function"]["name"] == "decode_show"
    assert converted[3] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": '{"ok": true}',
    }
    assert converted[4] == {"role": "user", "content": "Continue."}


def test_openai_adapter_normalizes_text_and_tool_calls():
    raw = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Plan",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(
                                name="score_panel",
                                arguments='{"branch": "main"}',
                            ),
                        )
                    ],
                )
            )
        ],
    )

    response = _openai_chat_response_to_model_response(raw)

    assert response.content == [
        TextBlock(text="Plan"),
        ToolUseBlock(id="call_1", name="score_panel", input={"branch": "main"}),
    ]
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4
    assert response.usage.cache_read_input_tokens == 2


def test_tool_schema_converters_cover_openai_and_gemini_shapes():
    tools = [
        {
            "name": "decode_show",
            "description": "Show decode.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch name",
                        "default": "main",
                    }
                },
                "required": ["branch"],
                "additionalProperties": False,
            },
        }
    ]

    openai_tools = _tools_to_openai_chat(tools)
    gemini_schema = _schema_for_gemini(tools[0]["input_schema"])

    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "decode_show"
    assert gemini_schema["type"] == "object"
    assert "default" not in gemini_schema["properties"]["branch"]
    assert "additionalProperties" not in gemini_schema


class _FakeOpenAIEndpoint:
    """Captures kwargs passed to a create(...) call and returns a canned response."""

    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeOpenAIClient:
    def __init__(self, *, chat_response=None, responses_response=None):
        self.responses = _FakeOpenAIEndpoint(responses_response)
        self.chat = SimpleNamespace(
            completions=_FakeOpenAIEndpoint(chat_response)
        )


def _fake_responses_reply():
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="score_panel",
                arguments='{"branch": "main"}',
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=3,
            input_tokens_details=SimpleNamespace(cached_tokens=1),
        ),
    )


def _fake_chat_reply():
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=2,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=[])
            )
        ],
    )


def test_openai_provider_routes_gpt_5_6_to_responses_api():
    provider = OpenAIModelProvider(api_key="sk-test", model="gpt-5.6-sol")
    fake = _FakeOpenAIClient(responses_response=_fake_responses_reply())
    provider.client = fake

    response = provider.send(
        messages=[{"role": "user", "content": "Hi"}],
        tools=[
            {
                "name": "score_panel",
                "description": "Score.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        system="System prompt",
        max_tokens=1024,
    )

    assert fake.responses.calls, "expected responses.create to be called"
    assert not fake.chat.completions.calls
    call = fake.responses.calls[0]
    assert call["model"] == "gpt-5.6-sol"
    assert call["instructions"] == "System prompt"
    assert call["max_output_tokens"] == 1024
    # Flat responses-format tool (not the nested chat shape).
    assert call["tools"] == [
        {
            "type": "function",
            "name": "score_panel",
            "description": "Score.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    # Converted input items (system is carried by instructions=, not here).
    assert call["input"] == [{"role": "user", "content": "Hi"}]
    # No reasoning parameter is set: the model default effort applies.
    assert "reasoning" not in call
    assert isinstance(response, ModelResponse)
    assert response.content == [
        ToolUseBlock(id="call_1", name="score_panel", input={"branch": "main"})
    ]
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 3
    assert response.usage.cache_read_input_tokens == 1


def test_openai_provider_keeps_gpt_5_5_on_chat_completions():
    provider = OpenAIModelProvider(api_key="sk-test", model="gpt-5.5")
    fake = _FakeOpenAIClient(chat_response=_fake_chat_reply())
    provider.client = fake

    response = provider.send(
        messages=[{"role": "user", "content": "Hi"}],
        tools=[
            {
                "name": "score_panel",
                "description": "Score.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        system="System prompt",
        max_tokens=1024,
    )

    assert fake.chat.completions.calls, "expected chat.completions.create to be called"
    assert not fake.responses.calls
    call = fake.chat.completions.calls[0]
    assert call["model"] == "gpt-5.5"
    # Chat path kwargs stay exactly as before: nested tool, system message,
    # and max_completion_tokens.
    assert call["max_completion_tokens"] == 1024
    assert "max_output_tokens" not in call
    assert call["tools"][0]["type"] == "function"
    assert call["tools"][0]["function"]["name"] == "score_panel"
    assert call["messages"][0] == {"role": "system", "content": "System prompt"}
    assert response.content == [TextBlock(text="ok")]


def test_openai_responses_converts_anthropic_style_tool_history():
    messages = [
        {"role": "user", "content": "Initial text"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will inspect."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "decode_show",
                    "input": {"branch": "main"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": '{"ok": true}',
                },
                {"type": "text", "text": "Continue."},
            ],
        },
    ]

    items = _messages_to_openai_responses(messages)

    assert items == [
        {"role": "user", "content": "Initial text"},
        {"role": "assistant", "content": "I will inspect."},
        {
            "type": "function_call",
            "call_id": "toolu_1",
            "name": "decode_show",
            "arguments": '{"branch": "main"}',
        },
        {
            "type": "function_call_output",
            "call_id": "toolu_1",
            "output": '{"ok": true}',
        },
        {"role": "user", "content": "Continue."},
    ]


def test_openai_responses_normalizes_text_and_tool_calls():
    raw = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="score_panel",
                arguments='{"branch": "main"}',
            ),
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Plan")],
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=4,
            input_tokens_details=SimpleNamespace(cached_tokens=2),
        ),
    )

    response = _openai_responses_response_to_model_response(raw)

    assert response.content == [
        ToolUseBlock(id="call_1", name="score_panel", input={"branch": "main"}),
        TextBlock(text="Plan"),
    ]
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4
    assert response.usage.cache_read_input_tokens == 2


def test_tools_to_openai_responses_uses_flat_function_shape():
    tools = [
        {
            "name": "decode_show",
            "description": "Show decode.",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]

    assert _tools_to_openai_responses(None) is None
    assert _tools_to_openai_responses([]) is None
    assert _tools_to_openai_responses(tools) == [
        {
            "type": "function",
            "name": "decode_show",
            "description": "Show decode.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_requires_responses_api_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv("DECIPHER_OPENAI_API", raising=False)
    # Default routing: only gpt-5.6* uses the Responses API.
    assert _requires_responses_api("gpt-5.6-sol") is True
    assert _requires_responses_api("gpt-5.6-terra") is True
    assert _requires_responses_api("gpt-5.5") is False
    assert _requires_responses_api("gpt-5.4-mini") is False

    # Override forces responses even for a chat-default model.
    monkeypatch.setenv("DECIPHER_OPENAI_API", "responses")
    assert _requires_responses_api("gpt-5.5") is True

    # Override forces chat even for a responses-default model.
    monkeypatch.setenv("DECIPHER_OPENAI_API", "chat")
    assert _requires_responses_api("gpt-5.6-sol") is False


def test_openai_api_env_override_routes_send(monkeypatch):
    monkeypatch.setenv("DECIPHER_OPENAI_API", "responses")
    provider = OpenAIModelProvider(api_key="sk-test", model="gpt-5.5")
    fake = _FakeOpenAIClient(responses_response=_fake_responses_reply())
    provider.client = fake

    provider.send(
        messages=[{"role": "user", "content": "Hi"}],
        tools=None,
        system="System prompt",
        max_tokens=256,
    )

    assert fake.responses.calls, "override should force the responses path"
    assert not fake.chat.completions.calls


# ---------------------------------------------------------------------------
# OpenAI reasoning-item passback (Responses API)
# ---------------------------------------------------------------------------

def _reasoning_item():
    """A SimpleNamespace reasoning output item (no model_dump; exercises vars()).

    Includes the response-only ``status`` field and a None-valued ``content``,
    as real Responses API dumps do — the API 400s (unknown_parameter) if these
    are re-sent as input, so tests assert they are captured verbatim but
    sanitized away at re-emit time.
    """
    return SimpleNamespace(
        type="reasoning",
        id="rs_1",
        summary=[],
        encrypted_content="ENC==",
        content=None,
        status="completed",
    )


def _fake_responses_reply_with_reasoning():
    """Responses reply where a reasoning item precedes a function_call item."""
    return SimpleNamespace(
        output=[
            _reasoning_item(),
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="score_panel",
                arguments='{"branch": "main"}',
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=3,
            input_tokens_details=SimpleNamespace(cached_tokens=1),
        ),
    )


def test_send_responses_sets_store_false_and_include_by_default(monkeypatch):
    monkeypatch.delenv("DECIPHER_OPENAI_REASONING_PASSBACK", raising=False)
    provider = OpenAIModelProvider(api_key="sk-test", model="gpt-5.6-sol")
    fake = _FakeOpenAIClient(responses_response=_fake_responses_reply())
    provider.client = fake

    provider.send(messages=[{"role": "user", "content": "Hi"}], tools=None, system="S")

    call = fake.responses.calls[0]
    assert call["store"] is False
    assert call["include"] == ["reasoning.encrypted_content"]


def test_send_responses_omits_store_include_when_passback_disabled(monkeypatch):
    monkeypatch.setenv("DECIPHER_OPENAI_REASONING_PASSBACK", "0")
    provider = OpenAIModelProvider(api_key="sk-test", model="gpt-5.6-sol")
    fake = _FakeOpenAIClient(responses_response=_fake_responses_reply())
    provider.client = fake

    provider.send(messages=[{"role": "user", "content": "Hi"}], tools=None, system="S")

    call = fake.responses.calls[0]
    assert "store" not in call
    assert "include" not in call


def test_openai_responses_captures_reasoning_as_provider_extra(monkeypatch):
    monkeypatch.delenv("DECIPHER_OPENAI_REASONING_PASSBACK", raising=False)
    response = _openai_responses_response_to_model_response(
        _fake_responses_reply_with_reasoning()
    )

    # Capture is verbatim: response-only fields (status, None content) are kept
    # in history and only stripped at re-emit time.
    assert response.content == [
        ToolUseBlock(id="call_1", name="score_panel", input={"branch": "main"}),
        ProviderExtraBlock(
            provider="openai",
            kind="reasoning",
            items=[
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": "ENC==",
                    "content": None,
                    "status": "completed",
                }
            ],
        ),
    ]


def test_openai_responses_skips_reasoning_capture_when_disabled(monkeypatch):
    monkeypatch.setenv("DECIPHER_OPENAI_REASONING_PASSBACK", "off")
    response = _openai_responses_response_to_model_response(
        _fake_responses_reply_with_reasoning()
    )

    assert response.content == [
        ToolUseBlock(id="call_1", name="score_panel", input={"branch": "main"}),
    ]


def test_reasoning_round_trips_positioned_before_function_call(monkeypatch):
    monkeypatch.delenv("DECIPHER_OPENAI_REASONING_PASSBACK", raising=False)
    response = _openai_responses_response_to_model_response(
        _fake_responses_reply_with_reasoning()
    )
    # Mirror the loop: normalized blocks -> assistant message history.
    assistant_blocks, _tool_uses, _text = _collect_assistant_blocks(response)
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": assistant_blocks},
    ]

    items = _messages_to_openai_responses(messages)

    # Re-emitted item is sanitized: the response-only `status` field and the
    # None-valued `content` are dropped (the API 400s on them as input), while
    # the whitelisted fields survive verbatim.
    assert items == [
        {"role": "user", "content": "Hi"},
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "ENC==",
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "score_panel",
            "arguments": '{"branch": "main"}',
        },
    ]
    emitted_reasoning = next(it for it in items if it.get("type") == "reasoning")
    assert "status" not in emitted_reasoning
    assert "content" not in emitted_reasoning
    reasoning_idx = next(i for i, it in enumerate(items) if it.get("type") == "reasoning")
    call_idx = next(i for i, it in enumerate(items) if it.get("type") == "function_call")
    assert reasoning_idx < call_idx


def test_sanitize_reasoning_item_for_input_whitelists_fields():
    captured = {
        "type": "reasoning",
        "id": "rs_9",
        "summary": [{"type": "summary_text", "text": "s"}],
        "encrypted_content": "ENC==",
        "content": [{"type": "reasoning_text", "text": "visible"}],
        "status": "completed",           # response-only: must be dropped
        "some_future_field": "x",        # unknown: must be dropped
    }

    sanitized = _sanitize_reasoning_item_for_input(captured)

    assert sanitized == {
        "type": "reasoning",
        "id": "rs_9",
        "summary": [{"type": "summary_text", "text": "s"}],
        "encrypted_content": "ENC==",
        "content": [{"type": "reasoning_text", "text": "visible"}],
    }
    # None-valued whitelisted fields are dropped too.
    assert _sanitize_reasoning_item_for_input(
        {"type": "reasoning", "id": "rs_9", "content": None, "encrypted_content": None}
    ) == {"type": "reasoning", "id": "rs_9"}


def _assistant_turn_with_reasoning(turn: int) -> dict:
    """An assistant history turn: one tool_use plus its provider_extra block."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": f"call_{turn}",
                "name": "decode_show",
                "input": {"turn": turn},
            },
            {
                "type": "provider_extra",
                "provider": "openai",
                "kind": "reasoning",
                "items": [
                    {
                        "type": "reasoning",
                        "id": f"rs_{turn}",
                        "summary": [],
                        "encrypted_content": f"ENC{turn}==",
                        "status": "completed",
                    }
                ],
            },
        ],
    }


def test_reasoning_round_trip_two_turns_keeps_per_turn_positioning(monkeypatch):
    monkeypatch.delenv("DECIPHER_OPENAI_REASONING_PASSBACK", raising=False)
    messages = [
        {"role": "user", "content": "Hi"},
        _assistant_turn_with_reasoning(1),
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "r1"},
            ],
        },
        _assistant_turn_with_reasoning(2),
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_2", "content": "r2"},
            ],
        },
    ]

    items = _messages_to_openai_responses(messages)

    def index_of(predicate):
        return next(i for i, it in enumerate(items) if predicate(it))

    rs1 = index_of(lambda it: it.get("type") == "reasoning" and it.get("id") == "rs_1")
    fc1 = index_of(
        lambda it: it.get("type") == "function_call" and it.get("call_id") == "call_1"
    )
    out1 = index_of(
        lambda it: it.get("type") == "function_call_output"
        and it.get("call_id") == "call_1"
    )
    rs2 = index_of(lambda it: it.get("type") == "reasoning" and it.get("id") == "rs_2")
    fc2 = index_of(
        lambda it: it.get("type") == "function_call" and it.get("call_id") == "call_2"
    )
    out2 = index_of(
        lambda it: it.get("type") == "function_call_output"
        and it.get("call_id") == "call_2"
    )
    # Each turn's reasoning stays with its own turn: turn 2's reasoning comes
    # after turn 1's function_call_output and before turn 2's function_call.
    assert rs1 < fc1 < out1 < rs2 < fc2 < out2


def test_reasoning_only_turn_is_not_reemitted_dangling(monkeypatch):
    monkeypatch.delenv("DECIPHER_OPENAI_REASONING_PASSBACK", raising=False)
    # A reasoning-only assistant turn (no text, no tool_use) — e.g. produced by
    # max-token exhaustion — must not re-emit a dangling reasoning item.
    messages = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "provider_extra",
                    "provider": "openai",
                    "kind": "reasoning",
                    "items": [
                        {
                            "type": "reasoning",
                            "id": "rs_1",
                            "summary": [],
                            "encrypted_content": "ENC==",
                        }
                    ],
                },
            ],
        },
    ]

    items = _messages_to_openai_responses(messages)

    assert items == [{"role": "user", "content": "Hi"}]


def test_reasoning_item_without_encrypted_content_is_skipped(monkeypatch):
    monkeypatch.delenv("DECIPHER_OPENAI_REASONING_PASSBACK", raising=False)
    # An item lacking encrypted_content (e.g. captured while the passback env
    # flag was off) is unresolvable under store=False; only the resolvable
    # sibling is re-emitted, and the function_call still goes out.
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "decode_show",
                    "input": {},
                },
                {
                    "type": "provider_extra",
                    "provider": "openai",
                    "kind": "reasoning",
                    "items": [
                        {"type": "reasoning", "id": "rs_bare", "summary": []},
                        {
                            "type": "reasoning",
                            "id": "rs_ok",
                            "summary": [],
                            "encrypted_content": "ENC==",
                        },
                    ],
                },
            ],
        },
    ]

    items = _messages_to_openai_responses(messages)

    reasoning_ids = [it.get("id") for it in items if it.get("type") == "reasoning"]
    assert reasoning_ids == ["rs_ok"]
    assert any(it.get("type") == "function_call" for it in items)


def test_reasoning_absent_from_request_when_passback_disabled(monkeypatch):
    monkeypatch.delenv("DECIPHER_OPENAI_REASONING_PASSBACK", raising=False)
    response = _openai_responses_response_to_model_response(
        _fake_responses_reply_with_reasoning()
    )
    assistant_blocks, _tool_uses, _text = _collect_assistant_blocks(response)
    messages = [{"role": "assistant", "content": assistant_blocks}]

    # provider_extra is still in history, but re-emit is gated off.
    monkeypatch.setenv("DECIPHER_OPENAI_REASONING_PASSBACK", "0")
    items = _messages_to_openai_responses(messages)

    assert all(it.get("type") != "reasoning" for it in items)
    assert any(it.get("type") == "function_call" for it in items)


def test_collect_assistant_blocks_carries_provider_extra_without_tool_use():
    response = ModelResponse(
        content=[
            TextBlock(text="Thinking about it."),
            ToolUseBlock(id="t1", name="decode_show", input={"branch": "main"}),
            ProviderExtraBlock(
                provider="openai",
                kind="reasoning",
                items=[{"type": "reasoning", "encrypted_content": "ENC=="}],
            ),
        ]
    )

    assistant_blocks, tool_uses, text_parts = _collect_assistant_blocks(response)

    # The opaque block lands in history verbatim as a plain dict...
    assert {
        "type": "provider_extra",
        "provider": "openai",
        "kind": "reasoning",
        "items": [{"type": "reasoning", "encrypted_content": "ENC=="}],
    } in assistant_blocks
    assert all(isinstance(b, dict) for b in assistant_blocks)
    # ...but does not leak into extracted text or tool calls.
    assert text_parts == ["Thinking about it."]
    assert tool_uses == [{"id": "t1", "name": "decode_show", "input": {"branch": "main"}}]


def test_compress_history_preserves_provider_extra_and_stubs_old_tool_results():
    messages: list[dict] = []
    # Six assistant turns each followed by a user tool_result turn.  The
    # assistant turns carry a provider_extra block that must survive intact,
    # while tool_results older than TOOL_RESULT_HISTORY_DEPTH get stubbed.
    for i in range(6):
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"t{i}", "name": "decode_show", "input": {}},
                {
                    "type": "provider_extra",
                    "provider": "openai",
                    "kind": "reasoning",
                    "items": [{"type": "reasoning", "id": f"rs_{i}"}],
                },
            ],
        })
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"t{i}", "content": f"result {i}"},
            ],
        })

    compressed = _compress_history(messages)

    # All provider_extra blocks in assistant turns are untouched.
    provider_extras = [
        c
        for m in compressed
        if m["role"] == "assistant"
        for c in m["content"]
        if isinstance(c, dict) and c.get("type") == "provider_extra"
    ]
    assert len(provider_extras) == 6
    assert provider_extras[0]["items"] == [{"type": "reasoning", "id": "rs_0"}]
    # The oldest tool_result (turn 0) is stubbed; the most recent is not.
    first_result = compressed[1]["content"][0]
    last_result = compressed[-1]["content"][0]
    assert first_result["content"] == TOOL_RESULT_STUB
    assert last_result["content"] == "result 5"


def test_openai_chat_converter_drops_provider_extra():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Plan."},
                {
                    "type": "provider_extra",
                    "provider": "openai",
                    "kind": "reasoning",
                    "items": [{"type": "reasoning", "encrypted_content": "ENC=="}],
                },
                {"type": "tool_use", "id": "t1", "name": "decode_show", "input": {}},
            ],
        },
    ]

    chat = _messages_to_openai_chat(messages, system="")

    assert len(chat) == 1
    msg = chat[0]
    assert msg["content"] == "Plan."
    assert msg["tool_calls"][0]["function"]["name"] == "decode_show"
    # No provider_extra leaks into the chat payload.
    assert "provider_extra" not in json.dumps(chat)


def test_gemini_converter_drops_provider_extra():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Plan."},
                {
                    "type": "provider_extra",
                    "provider": "openai",
                    "kind": "reasoning",
                    "items": [{"type": "reasoning", "encrypted_content": "ENC=="}],
                },
            ],
        },
    ]

    contents = _messages_to_gemini_contents(messages)

    # One Content with a single text part; the opaque block is silently skipped.
    assert len(contents) == 1
    rendered = " ".join(getattr(p, "text", "") or "" for p in contents[0].parts)
    assert "Plan." in rendered
    assert "provider_extra" not in rendered
    assert "ENC==" not in rendered


def test_strip_provider_extra_blocks_removes_only_those_blocks():
    messages = [
        {"role": "user", "content": "plain string stays"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "keep"},
                {"type": "provider_extra", "provider": "openai", "kind": "reasoning", "items": []},
                {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
            ],
        },
    ]

    stripped = _strip_provider_extra_blocks(messages)

    assert stripped[0] == {"role": "user", "content": "plain string stays"}
    assert stripped[1]["content"] == [
        {"type": "text", "text": "keep"},
        {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
    ]


def test_anthropic_path_drops_provider_extra_before_send():
    captured: dict = {}

    def _fake_send_message(*, messages, tools, system, max_tokens):
        captured["messages"] = messages
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(
                input_tokens=1, output_tokens=1, cache_read_input_tokens=0
            ),
        )

    fake_api = SimpleNamespace(model="claude-sonnet-4-6", send_message=_fake_send_message)
    provider = ClaudeModelProvider(fake_api)

    provider.send(
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "provider_extra",
                        "provider": "openai",
                        "kind": "reasoning",
                        "items": [{"type": "reasoning", "encrypted_content": "ENC=="}],
                    },
                ],
            }
        ],
        system="S",
    )

    sent = captured["messages"]
    assert not any(
        isinstance(c, dict) and c.get("type") == "provider_extra"
        for m in sent
        for c in (m["content"] if isinstance(m["content"], list) else [])
    )
    assert "ENC==" not in json.dumps(sent)


def test_reasoning_passback_env_switch(monkeypatch):
    monkeypatch.delenv("DECIPHER_OPENAI_REASONING_PASSBACK", raising=False)
    assert _reasoning_passback_enabled() is True
    for off_value in ("0", "false", "no", "off", "OFF", "False"):
        monkeypatch.setenv("DECIPHER_OPENAI_REASONING_PASSBACK", off_value)
        assert _reasoning_passback_enabled() is False
    for on_value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("DECIPHER_OPENAI_REASONING_PASSBACK", on_value)
        assert _reasoning_passback_enabled() is True


def test_inspect_artifact_build_timeline_handles_provider_extra():
    import importlib.util

    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "inspect_artifact_under_test", repo_root / "scripts" / "inspect_artifact.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve annotations via sys.modules.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    artifact = {
        "messages": [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "planning"},
                    {
                        "type": "provider_extra",
                        "provider": "openai",
                        "kind": "reasoning",
                        "items": [{"type": "reasoning", "encrypted_content": "ENC=="}],
                    },
                    {"type": "tool_use", "id": "t1", "name": "decode_show", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "{}"},
                ],
            },
        ]
    }

    timeline = module.build_timeline(artifact)

    assert len(timeline) == 1
    assert timeline[0]["tools"][0]["name"] == "decode_show"
    assert timeline[0]["reasoning"] == "planning"


def test_provider_cost_estimation_is_provider_specific():
    anthropic = estimate_provider_cost("anthropic", "claude-sonnet-4-6", 1000, 100)
    openai = estimate_provider_cost("openai", "gpt-5.4-mini", 1000, 100)
    gemini = estimate_provider_cost("gemini", "gemini-3-flash", 1000, 100)

    assert anthropic > 0
    assert openai > 0
    assert gemini > 0
    assert len({anthropic, openai, gemini}) == 3


def test_agent_run_state_exposes_mode_specific_tool_gates():
    state = AgentRunState()

    assert state.active_mode is AgentMode.EXPLORE
    assert state.allowed_tools() == MODE_ALLOWED_TOOLS[AgentMode.EXPLORE]

    state.outer_iteration = 4
    state.set_mode(AgentMode.BOUNDARY_PROJECTION, branch="repair")
    state.mark_workflow_started("full_reading_repair")
    state.mark_workflow_finished("full_reading_repair", note="projected boundaries")

    assert state.active_branch == "repair"
    assert state.allowed_tools() == MODE_ALLOWED_TOOLS[AgentMode.BOUNDARY_PROJECTION]
    workflow = state.workflows["repair:full_reading_repair"]
    assert workflow.status == "completed"
    assert workflow.started_outer_iteration == 4
    assert workflow.finished_outer_iteration == 4
    assert workflow.notes == ["projected boundaries"]


def test_tool_gated_detector_ignores_non_object_json_results():
    assert _is_tool_gated_result('{"reason": "tool_gated"}') is True
    assert _is_tool_gated_result('[{"symbol": "A", "count": 12}]') is False
    assert _is_tool_gated_result("not json") is False


def test_boundary_projection_count_failure_detector():
    result = (
        '{"error": "Cannot project", "same_character_count": false, '
        '"current_char_count": 345, "proposed_char_count": 178}'
    )

    assert _is_boundary_projection_count_failure(
        "decode_validate_reading_repair",
        result,
    ) is True
    assert _is_boundary_projection_count_failure("score_panel", result) is False
    assert _is_boundary_projection_count_failure(
        "decode_validate_reading_repair",
        '{"status": "ok", "same_character_count": true}',
    ) is False


def test_run_v2_records_loop_events_for_future_inner_loop_observability():
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

    event_names = [event.event for event in artifact.loop_events]
    assert "iteration_start" in event_names
    assert "no_tool_calls" in event_names
    assert "auto_declared_solution" in event_names
    assert "run_complete" in event_names


def test_boundary_projection_gate_retries_inside_same_outer_iteration():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _GatedThenDeclareAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=3,
        cipher_id="unit",
    )

    assert artifact.status == "solved"
    assert artifact.auto_declared is False
    assert len(api.messages_seen) >= 2
    assert artifact.solution is not None
    assert artifact.solution.declared_at_iteration == 1
    assert [call.tool_name for call in artifact.tool_calls[:2]] == [
        "search_anneal",
        "meta_declare_solution",
    ]
    retry_events = [e for e in artifact.loop_events if e.event == "gated_tool_retry"]
    assert len(retry_events) == 1
    assert retry_events[0].outer_iteration == 1
    assert retry_events[0].inner_step == 1
    assert retry_events[0].mode == "boundary_projection"
    assert retry_events[0].payload["attempted_tools"] == ["search_anneal"]
    retry_texts = [
        c["text"]
        for m in api.messages_seen[1]
        for c in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    assert any(BOUNDARY_PROJECTION_RETRY_PREFLIGHT in t for t in retry_texts)


def test_boundary_projection_count_mismatch_retries_inside_same_outer_iteration():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _BoundaryCountRetryAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=3,
        cipher_id="unit",
        automated_preflight={
            "enabled": True,
            "run_mode": "automated_only",
            "status": "solved",
            "solver": "fake_native",
            "summary": "Automated native solver preflight (no LLM access): ABC",
            "key": {"0": 0, "1": 1, "2": 2},
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        },
    )

    assert len(api.messages_seen) >= 2
    assert [call.tool_name for call in artifact.tool_calls[:2]] == [
        "decode_validate_reading_repair",
        "act_resegment_by_reading",
    ]
    retry_events = [
        e for e in artifact.loop_events
        if e.event == "boundary_projection_count_retry"
    ]
    assert len(retry_events) == 1
    assert retry_events[0].outer_iteration == 1
    assert retry_events[0].inner_step == 1
    assert retry_events[0].mode == "boundary_projection"
    assert retry_events[0].payload["attempted_tools"] == [
        "decode_validate_reading_repair"
    ]
    retry_texts = [
        c["text"]
        for m in api.messages_seen[1]
        for c in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    assert any(BOUNDARY_PROJECTION_COUNT_RETRY_PREFLIGHT in t for t in retry_texts)


def test_boundary_projection_count_mismatch_can_retry_more_than_once():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _BoundaryCountRetryTwiceThenDeclareAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=3,
        cipher_id="unit",
        automated_preflight={
            "enabled": True,
            "run_mode": "automated_only",
            "status": "solved",
            "solver": "fake_native",
            "summary": "Automated native solver preflight (no LLM access): ABC",
            "key": {"0": 0, "1": 1, "2": 2},
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        },
    )

    assert artifact.status == "solved"
    assert artifact.solution is not None
    assert artifact.solution.declared_at_iteration == 1
    retry_events = [
        e for e in artifact.loop_events
        if e.event == "boundary_projection_count_retry"
    ]
    assert len(retry_events) == 2
    assert [event.inner_step for event in retry_events] == [1, 2]
    assert [call.tool_name for call in artifact.tool_calls[:4]] == [
        "decode_validate_reading_repair",
        "decode_validate_reading_repair",
        "workspace_branch_cards",
        "meta_declare_solution",
    ]


def test_repair_sandbox_continues_after_low_cost_tool_without_outer_iteration():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _SandboxContinueThenDeclareAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=3,
        cipher_id="unit",
        automated_preflight={
            "enabled": True,
            "run_mode": "automated_only",
            "status": "solved",
            "solver": "fake_native",
            "summary": "Automated native solver preflight (no LLM access): ABC",
            "key": {"0": 0, "1": 1, "2": 2},
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        },
    )

    assert artifact.status == "solved"
    assert artifact.solution is not None
    assert artifact.solution.declared_at_iteration == 1
    assert [call.tool_name for call in artifact.tool_calls[:3]] == [
        "decode_show",
        "workspace_branch_cards",
        "meta_declare_solution",
    ]
    sandbox_events = [
        e for e in artifact.loop_events
        if e.event == "repair_sandbox_continue"
    ]
    assert len(sandbox_events) == 2
    assert sandbox_events[0].outer_iteration == 1
    retry_texts = [
        c["text"]
        for m in api.messages_seen[1]
        for c in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    assert any(REPAIR_SANDBOX_CONTINUE_PREFLIGHT in t for t in retry_texts)


def test_final_turn_bookkeeping_without_declare_retries_inside_same_iteration():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _FinalBookkeepingThenDeclareAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=1,
        cipher_id="unit",
    )

    assert artifact.status == "solved"
    assert artifact.solution is not None
    assert artifact.solution.branch == "main"
    assert [call.tool_name for call in artifact.tool_calls[:2]] == [
        "workspace_branch_cards",
        "meta_declare_solution",
    ]
    retry_events = [
        e for e in artifact.loop_events
        if e.event == "final_declare_retry"
    ]
    assert len(retry_events) == 1
    retry_texts = [
        c["text"]
        for m in api.messages_seen[1]
        for c in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    assert any(FINAL_DECLARATION_RETRY_PREFLIGHT in t for t in retry_texts)


def test_final_turn_gated_tool_retries_with_declaration_nudge():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
    api = _FinalGatedToolThenDeclareAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=1,
        cipher_id="unit",
    )

    assert artifact.status == "solved"
    assert artifact.solution is not None
    assert artifact.solution.reading_summary == "Toy final declaration."
    assert [call.tool_name for call in artifact.tool_calls[:2]] == [
        "act_resegment_by_reading",
        "meta_declare_solution",
    ]
    retry_events = [
        e for e in artifact.loop_events
        if e.event == "final_declare_retry"
    ]
    assert len(retry_events) == 1
    assert retry_events[0].payload["reason"] == "gated_tool_on_final_iteration"
    retry_texts = [
        c["text"]
        for m in api.messages_seen[1]
        for c in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    assert any(FINAL_DECLARATION_RETRY_PREFLIGHT in t for t in retry_texts)


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
    assert artifact.status == "fallback_declared"
    assert artifact.auto_declared is True
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


def test_final_turn_exposes_declare_and_bookkeeping_tools():
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
    assert final_turn_tool_names == {
        "workspace_branch_cards",
        "workspace_hypothesis_next_steps",
        "repair_agenda_list",
        "repair_agenda_update",
        "meta_declare_solution",
        "meta_declare_unsolved",
    }


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
    assert any("Protected baseline rule" in t for t in first_turn_texts)
    assert artifact.automated_preflight is not None
    branch = next(b for b in artifact.branches if b.name == "automated_preflight")
    assert branch.decryption == "THE"
    assert "no_llm" in branch.tags


def test_pure_transposition_preflight_installs_branch_with_decryption():
    """Pure transposition returns empty key but a non-empty decryption string.
    The automated_preflight branch should still be created and carry the
    decryption text so BranchSnapshot.decryption is non-empty.
    """
    # 3-token cipher: CBA — reverse is ABC
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="CBA", alphabet=alpha, separator=None)
    api = _FakeAPI()

    # Pipeline that reverses the 3 tokens: [2,1,0] → indices [2,1,0]
    reverse_pipeline = {
        "steps": [{"name": "Reverse", "data": {"rangeStart": 0, "rangeEnd": 2}}]
    }

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=1,
        cipher_id="unit_transpos",
        automated_preflight={
            "enabled": True,
            "run_mode": "automated_only",
            "status": "completed",
            "solver": "pure_transposition_screen_rust",
            "summary": "Automated native solver preflight: ABC",
            "key": {},  # pure transposition — no substitution key
            "decryption": "ABC",
            "steps": [
                {
                    "name": "screen_pure_transposition",
                    "selected": {"pipeline": reverse_pipeline},
                }
            ],
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        },
    )

    branch = next(
        (b for b in artifact.branches if b.name == "automated_preflight"), None
    )
    assert branch is not None, "automated_preflight branch should be installed"
    assert "no_llm" in branch.tags
    assert branch.decryption == "ABC", (
        f"decryption should be 'ABC', got {branch.decryption!r}"
    )
    # token_order should reflect the reverse pipeline
    assert branch.token_order == [2, 1, 0], (
        f"token_order should be [2,1,0], got {branch.token_order}"
    )


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

    assert artifact.status == "fallback_declared"
    assert artifact.auto_declared is True
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

    # The panel reminder only fires when the full reading workflow has already
    # been used (otherwise the gate user message is the single carrier).
    panel = build_workspace_panel(
        ex.workspace,
        iteration=14,
        max_iterations=15,
        language="en",
        word_set={"APPLY"},
        full_reading_workflow_used=True,
    )

    assert PENULTIMATE_READING_WORKFLOW_PREFLIGHT in panel
    assert "act_resegment_from_reading_repair" in panel


def test_workspace_panel_omits_penultimate_warning_when_workflow_unused():
    ex = _executor_for("AP PLY", separator=" ")

    panel = build_workspace_panel(
        ex.workspace,
        iteration=14,
        max_iterations=15,
        language="en",
        word_set={"APPLY"},
        full_reading_workflow_used=False,
    )

    assert PENULTIMATE_READING_WORKFLOW_PREFLIGHT not in panel


class _ScorePanelEveryTurnAPI:
    """Keeps the loop alive with a harmless read-only tool; never uses the
    full reading workflow, so the gate user message is the single carrier."""

    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen: list = []
        self.n = 0

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.n += 1
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id=f"sp{self.n}",
                    name="score_panel",
                    input={"branch": "main"},
                )
            ],
        )


class _ReadingWorkflowThenScorePanelAPI:
    """Uses a reading-workflow actuator on turn 1 (so the workflow counts as
    used), then keeps the loop alive with a harmless read-only tool."""

    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen: list = []
        self.n = 0

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        self.n += 1
        if self.n == 1:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="reseg1",
                        name="act_resegment_by_reading",
                        input={"branch": "main", "proposed_text": "ABCABCABC"},
                    )
                ],
            )
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id=f"sp{self.n}",
                    name="score_panel",
                    input={"branch": "main"},
                )
            ],
        )


def _count_penultimate_preflight(messages: list) -> tuple[int, int]:
    """Return (standalone_gate_count, in_panel_count) of the penultimate
    reading-workflow preflight across a single send's visible messages."""
    standalone = 0
    in_panel = 0
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                text = c.get("text", "")
                if PENULTIMATE_READING_WORKFLOW_PREFLIGHT in text:
                    if PANEL_HEADER_MARKER in text:
                        in_panel += 1
                    else:
                        standalone += 1
    return standalone, in_panel


def test_penultimate_preflight_not_duplicated_when_workflow_unused():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC ABC ABC", alphabet=alpha, separator=" ")
    api = _ScorePanelEveryTurnAPI()

    run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=4,
        cipher_id="unit",
    )

    counts = [_count_penultimate_preflight(msgs) for msgs in api.messages_seen]
    # The model must see exactly one copy per turn: never more than one across
    # a single send's visible messages.
    assert all((standalone + in_panel) <= 1 for standalone, in_panel in counts)
    # The gate user message is the single carrier; the panel copy is suppressed.
    assert any(standalone == 1 for standalone, _ in counts)
    assert all(in_panel == 0 for _, in_panel in counts)


def test_penultimate_preflight_uses_panel_copy_when_workflow_used():
    alpha = Alphabet(list("ABC"))
    ct = CipherText(raw="ABC ABC ABC", alphabet=alpha, separator=" ")
    api = _ReadingWorkflowThenScorePanelAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=4,
        cipher_id="unit",
    )

    assert "act_resegment_by_reading" in [tc.tool_name for tc in artifact.tool_calls]

    counts = [_count_penultimate_preflight(msgs) for msgs in api.messages_seen]
    # Still at most one copy per turn.
    assert all((standalone + in_panel) <= 1 for standalone, in_panel in counts)
    # The panel copy is the carrier; the gate user message never fires.
    assert any(in_panel == 1 for _, in_panel in counts)
    assert all(standalone == 0 for standalone, _ in counts)


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


def test_act_resegment_by_reading_shows_overlay_on_no_boundary_cipher():
    raw = "THEREFORE"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THERE", "FORE", "THEREFORE"},
        word_list=["THERE", "FORE", "THEREFORE"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    out = ex._tool_act_resegment_by_reading({
        "branch": "main",
        "proposed_text": "THERE FORE",
    })

    assert out["status"] == "ok"
    assert out["old_word_count"] == 1
    assert out["new_word_count"] == 2
    assert ws.apply_key("main") == "THERE FORE"
    snap = ws.snapshot_branch("main")
    assert snap["custom_word_boundaries"] is True
    assert snap["decryption"] == "THERE FORE"


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


def test_act_resegment_by_reading_partial_apply_saves_correct_prefix():
    """When proposed text inserts a spurious word, apply the correct prefix and
    report exactly where the divergence occurred."""
    # Simulate the 'OCCUPIED A NARROW' scenario: decoded stream is
    # "THEOLDBOOKSTOREOCCUPIEDNARROW" (no spaces), agent proposes with extra A.
    decoded = "THEOLDBOOKSTOREOCCUPIEDNARROW"
    raw = decoded  # single-char alphabet, separator=None
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "OLD", "BOOKSTORE", "OCCUPIED", "NARROW"},
        word_list=["THE", "OLD", "BOOKSTORE", "OCCUPIED", "NARROW"],
        pattern_dict={},
    )
    # Set identity mapping so decoded text equals cipher text
    alpha_obj = ex.workspace.cipher_text.alphabet
    pt = ex.workspace.plaintext_alphabet
    for sym in alpha_obj.symbols:
        ex.workspace.set_mapping("main", alpha_obj.id_for(sym), pt.id_for(sym))

    # Propose text with a spurious "A" inserted: one extra char
    out = ex._tool_act_resegment_by_reading({
        "branch": "main",
        "proposed_text": "THE OLD BOOKSTORE OCCUPIED A NARROW",
    })

    # Should return partial, not error
    assert out["status"] == "partial"
    assert out["applied"] is True
    # The first 4 words are correct (THE, OLD, BOOKSTORE, OCCUPIED)
    assert out["applied_word_count"] == 4
    # The mismatch is at word index 4, proposed word "A"
    assert out["mismatch"]["at_proposed_word_index"] == 4
    assert out["mismatch"]["proposed_word"] == "A"
    assert out["mismatch"]["decoded_char"] == "N"   # 'N' for NARROW
    # Correct prefix boundaries were applied; remaining stream starts with NARROW
    assert out["remaining_stream_preview"].startswith("NARROW")
    # Branch now has 5 spans: THE, OLD, BOOKSTORE, OCCUPIED, NARROW (one big)
    spans = ex.workspace.effective_word_spans("main")
    assert len(spans) == 5
    assert spans[0] == (0, 3)   # THE
    assert spans[1] == (3, 6)   # OLD
    assert spans[2] == (6, 15)  # BOOKSTORE
    assert spans[3] == (15, 23) # OCCUPIED
    assert spans[4] == (23, 29) # NARROW (remaining, unsegmented as one block)

    # Now the corrected call (without 'A') should succeed
    out2 = ex._tool_act_resegment_by_reading({
        "branch": "main",
        "proposed_text": "THE OLD BOOKSTORE OCCUPIED NARROW",
    })
    assert out2["status"] == "ok"
    assert out2["applied"] is True
    spans2 = ex.workspace.effective_word_spans("main")
    assert len(spans2) == 5
    assert spans2[4] == (23, 29)  # NARROW as its own word


def test_act_resegment_by_reading_no_partial_when_mismatch_at_first_char():
    """When the mismatch is at character 0 there is nothing to save."""
    raw = "ABCDE"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )
    alpha_obj = ex.workspace.cipher_text.alphabet
    pt = ex.workspace.plaintext_alphabet
    for sym in alpha_obj.symbols:
        ex.workspace.set_mapping("main", alpha_obj.id_for(sym), pt.id_for(sym))

    # Propose a completely different (longer) prefix — diverges at char 0
    out = ex._tool_act_resegment_by_reading({
        "branch": "main",
        "proposed_text": "X ABCDE",
    })

    # Should return error, not partial (nothing to save)
    assert "error" in out
    assert out["character_preserving"] is False


def test_act_resegment_window_by_reading_merges_local_words():
    raw = "LIBE BITUR SI"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="la",
        word_set={"LIBEBITUR", "SI"},
        word_list=["SI", "LIBEBITUR"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    out = ex._tool_act_resegment_window_by_reading({
        "branch": "main",
        "start_word_index": 0,
        "word_count": 2,
        "proposed_text": "LIBEBITUR",
    })

    assert out["status"] == "ok"
    assert out["character_preserving"] is True
    assert out["old_window_word_count"] == 2
    assert out["new_window_word_count"] == 1
    assert ws.apply_key("main") == "LIBEBITUR SI"


def test_act_resegment_window_by_reading_splits_local_word():
    raw = "A POTESTQUIBUS EUA"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="la",
        word_set={"A", "POTEST", "QUIBUS", "EUA"},
        word_list=["A", "POTEST", "QUIBUS", "EUA"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    out = ex._tool_act_resegment_window_by_reading({
        "branch": "main",
        "start_word_index": 1,
        "word_count": 1,
        "proposed_text": "POTEST QUIBUS",
    })

    assert out["status"] == "ok"
    assert out["old_window_word_count"] == 1
    assert out["new_window_word_count"] == 2
    assert ws.apply_key("main") == "A POTEST QUIBUS EUA"


def test_act_resegment_window_by_reading_projects_boundary_despite_letter_diff():
    raw = "PHYSICS ER DID"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"PHYSICKER", "DID"},
        word_list=["DID", "PHYSICKER"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    out = ex._tool_act_resegment_window_by_reading({
        "branch": "main",
        "start_word_index": 0,
        "word_count": 2,
        "proposed_text": "PHYSICKER",
    })

    assert out["status"] == "ok"
    assert out["character_preserving"] is False
    assert out["same_character_count"] is True
    assert out["projected_words"] == ["PHYSICSER"]
    assert out["mismatches"][0]["current_char"] == "S"
    assert out["mismatches"][0]["proposed_char"] == "K"
    assert ws.apply_key("main") == "PHYSICSER DID"


def test_act_resegment_window_by_reading_rejects_local_count_mismatch():
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

    out = ex._tool_act_resegment_window_by_reading({
        "branch": "main",
        "start_word_index": 0,
        "word_count": 2,
        "proposed_text": "APPLYING",
    })

    assert "error" in out
    assert out["same_character_count"] is False
    assert ws.apply_key("main") == "AP PLY"


def test_act_resegment_window_by_reading_suggests_nearby_matching_window():
    raw = "EAER LIBE BITUR SI"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="la",
        word_set={"LIBEBITUR"},
        word_list=["LIBEBITUR"],
        pattern_dict={},
    )
    ws = ex.workspace
    pt = ws.plaintext_alphabet
    for cipher_sym in alpha.symbols:
        ws.set_mapping("main", alpha.id_for(cipher_sym), pt.id_for(cipher_sym))

    out = ex._tool_act_resegment_window_by_reading({
        "branch": "main",
        "start_word_index": 2,
        "word_count": 2,
        "proposed_text": "LIBEBITUR",
    })

    assert "error" in out
    assert out["same_character_count"] is False
    suggestions = out["nearby_compatible_windows"]
    assert suggestions
    assert suggestions[0]["start_word_index"] == 1
    assert suggestions[0]["word_count"] == 2
    assert suggestions[0]["current_window_words"] == ["LIBE", "BITUR"]
    assert suggestions[0]["character_preserving"] is True
    assert "start_word_index=1" in suggestions[0]["suggested_call"]
    assert ws.apply_key("main") == "EAER LIBE BITUR SI"


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
        tool_name="act_resegment_from_reading_repair",
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


def test_pin_full_reading_workflow_required_reason():
    """A2 pin (pre-extraction): the reading-workflow declaration gate keeps its
    exact reason string and suggested-tool order. This behavior must be
    byte-identical before and after the DeclarationPolicy extraction."""
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
    assert out["reason"] == "full_reading_workflow_required"
    assert out["suggested_next_tools"] == [
        "decode_validate_reading_repair",
        "act_resegment_by_reading",
        "act_resegment_from_reading_repair",
    ]
    assert ex.terminated is False


def test_pin_multi_prerequisite_declare_batch_shape():
    """A2 pin (pre-extraction): when two quick prerequisites are unmet the
    batch response collapses to reason `prerequisites_required`, reports all
    unmet preconditions, appends meta_declare_solution to suggested tools, and
    records the pending-declare state on the executor. Byte-identical across the
    DeclarationPolicy extraction."""
    ex = _executor_for("ABCABC", separator=None)
    ex._tool_workspace_create_hypothesis_branch({
        "new_name": "hyp_poly",
        "cipher_mode": "periodic_polyalphabetic",
        "rationale": "Periodic diagnostics should be tested.",
    })
    # Deliberately do NOT call workspace_branch_cards, so both
    # branch_cards_required and hypothesis_next_steps_required are unmet.
    blocked = ex._tool_meta_declare_solution({
        "branch": "hyp_poly",
        "rationale": "Declaring without prerequisites.",
        "self_confidence": 0.7,
        "further_iterations_helpful": False,
    })

    assert blocked["status"] == "blocked"
    assert blocked["accepted"] is False
    assert blocked["reason"] == "prerequisites_required"
    assert blocked["preconditions_unmet"] == 2
    reasons = {p["reason"] for p in blocked["preconditions"]}
    assert reasons == {"branch_cards_required", "hypothesis_next_steps_required"}
    assert blocked["suggested_next_tools"][-1] == "meta_declare_solution"
    # Gate-state side effect is recorded on the executor (discharged elsewhere).
    assert ex._pending_declare_branch == "hyp_poly"
    assert ex._pending_declare_prerequisites == reasons
    assert ex.terminated is False


def test_meta_declare_blocks_unresolved_repair_agenda_until_item_resolved():
    raw = "ABC | ABD"
    alpha = Alphabet.from_text(raw, ignore_chars={" ", "|"})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" | ")
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"BAT", "BAR"},
        word_list=["BAT", "BAR"],
        pattern_dict={},
    )
    pt = ex.workspace.plaintext_alphabet
    ws = ex.workspace
    ws.set_mapping("main", alpha.id_for("A"), pt.id_for("C"))
    ws.set_mapping("main", alpha.id_for("B"), pt.id_for("A"))
    ws.set_mapping("main", alpha.id_for("C"), pt.id_for("T"))
    ws.set_mapping("main", alpha.id_for("D"), pt.id_for("R"))

    plan = ex._tool_decode_plan_word_repair({
        "branch": "main",
        "decoded_word": "CAT",
        "target_word": "BAT",
    })

    blocked = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "CAT should probably stay because the collateral is bad.",
        "self_confidence": 0.5,
    })

    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "repair_agenda_unresolved"
    assert ex.terminated is False

    ex._tool_repair_agenda_update({
        "item_id": plan["agenda_item"]["id"],
        "status": "held",
        "notes": "Collateral damage is too broad.",
    })
    accepted = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Best available branch; repair was held.",
        "self_confidence": 0.5,
    })

    assert accepted["status"] == "ok"
    assert accepted["accepted"] is True
    assert ex.terminated is True


def test_meta_declare_requires_branch_cards_when_multiple_branches_exist():
    ex = _executor_for("ABC", separator=None)
    ex.workspace.fork("candidate", from_branch="main")

    blocked = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Best branch.",
        "self_confidence": 0.5,
    })

    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "branch_cards_required"
    assert ex.terminated is False

    cards = ex._tool_workspace_branch_cards({})
    assert cards["status"] == "ok"
    ex.call_log.append(SimpleNamespace(
        tool_name="workspace_branch_cards",
        arguments={},
    ))

    accepted = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Best branch after branch-card comparison.",
        "self_confidence": 0.5,
    })

    assert accepted["status"] == "ok"
    assert accepted["accepted"] is True
    assert ex.terminated is True


def test_meta_declare_requires_fresh_branch_cards_for_new_branch():
    ex = _executor_for("ABC", separator=None)
    ex.workspace.fork("old_candidate", from_branch="main")
    ex.call_log.append(SimpleNamespace(
        tool_name="workspace_branch_cards",
        arguments={},
        iteration=1,
    ))
    ex.workspace.set_iteration(5)
    ex.workspace.fork("new_candidate", from_branch="main")

    blocked = ex._tool_meta_declare_solution({
        "branch": "new_candidate",
        "rationale": "Best newly-created branch.",
        "self_confidence": 0.8,
    })

    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "branch_cards_required"


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


def test_meta_declare_records_final_reading_summary_and_iteration_assessment():
    ex = _executor_for("ABC", separator=None)
    ex.set_iteration(3)

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Best available branch after reading.",
        "self_confidence": 0.82,
        "reading_summary": "This appears to be a short test passage.",
        "further_iterations_helpful": True,
        "further_iterations_note": "More iterations could test one unresolved spelling.",
    })

    assert out["status"] == "ok"
    assert out["accepted"] is True
    assert ex.solution is not None
    assert ex.solution.reading_summary == "This appears to be a short test passage."
    assert ex.solution.further_iterations_helpful is True
    assert ex.solution.further_iterations_note == "More iterations could test one unresolved spelling."


def test_meta_declare_blocks_low_confidence_helpful_declaration_before_final_turn():
    ex = _executor_for("ABC", separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(13)

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Best current low-confidence hypothesis.",
        "self_confidence": 0.15,
        "reading_summary": "Only scattered word islands are visible.",
        "further_iterations_helpful": True,
        "further_iterations_note": "More iterations should try another hypothesis.",
    })

    assert out["status"] == "blocked"
    assert out["accepted"] is False
    assert out["reason"] == "low_confidence_more_work_required"
    assert "search_transform_homophonic" in out["suggested_next_tools"]
    assert ex.terminated is False


def test_meta_declare_blocks_helpful_declaration_before_final_turn_even_when_confident():
    ex = _executor_for("ABC", separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(13)

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Readable but still rough.",
        "self_confidence": 0.74,
        "reading_summary": "The broad topic is visible, but exact text is not clean.",
        "further_iterations_helpful": True,
        "further_iterations_note": "More iterations should repair obvious words.",
    })

    assert out["status"] == "blocked"
    assert out["accepted"] is False
    assert out["reason"] == "further_iterations_requested"
    assert ex.terminated is False


def test_meta_declare_forced_partial_does_not_override_more_work_needed():
    ex = _executor_for("ABC", separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(26)

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "This is only a partial result.",
        "self_confidence": 0.30,
        "reading_summary": "Only scattered word islands are visible.",
        "further_iterations_helpful": True,
        "further_iterations_note": "A broader transform search would likely help.",
        "forced_partial": True,
    })

    assert out["status"] == "blocked"
    assert out["accepted"] is False
    assert out["reason"] == "further_iterations_requested"
    assert ex.terminated is False


def test_meta_declare_blocks_early_low_confidence_forced_partial():
    ex = _executor_for("ABC", separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(6)

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Best current partial, but it is not readable.",
        "self_confidence": 0.18,
        "reading_summary": "No trustworthy plaintext recovered.",
        "further_iterations_helpful": False,
        "further_iterations_note": "No immediate small repair is obvious.",
        "forced_partial": True,
    })

    assert out["status"] == "blocked"
    assert out["accepted"] is False
    assert out["reason"] == "partial_too_early"
    assert "search_transform_homophonic" in out["suggested_next_tools"]
    assert ex.terminated is False


def test_meta_declare_allows_low_confidence_forced_partial_in_final_stretch():
    ex = _executor_for("ABC", separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(25)

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Late-run partial after larger swings failed.",
        "self_confidence": 0.18,
        "reading_summary": "No trustworthy plaintext recovered.",
        "further_iterations_helpful": False,
        "further_iterations_note": "Remaining work is unlikely to help within this run.",
        "forced_partial": True,
    })

    assert out["status"] == "ok"
    assert out["accepted"] is True
    assert ex.terminated is True


def test_meta_declare_blocks_untried_transform_work_when_note_names_it():
    ex = _executor_for("ABC", separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(13)

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Best current hypothesis, but columnar transposition may be involved.",
        "self_confidence": 0.7,
        "reading_summary": "Only scattered word islands are visible.",
        "further_iterations_helpful": True,
        "further_iterations_note": "Further iterations should try columnar transposition.",
    })

    assert out["status"] == "blocked"
    assert out["reason"] == "transform_work_untried"
    assert "observe_transform_pipeline" in out["suggested_next_tools"]
    assert "search_transform_homophonic" in out["suggested_next_tools"]
    assert ex.terminated is False


def test_meta_declare_allows_low_confidence_helpful_on_final_turn():
    ex = _executor_for("ABC", separator=None)
    ex.set_max_iterations(30)
    ex.set_iteration(30)

    out = ex._tool_meta_declare_solution({
        "branch": "main",
        "rationale": "Final turn; submit best current hypothesis.",
        "self_confidence": 0.15,
        "reading_summary": "Only scattered word islands are visible.",
        "further_iterations_helpful": True,
        "further_iterations_note": "More work might help, but the run is out of turns.",
    })

    assert out["status"] == "ok"
    assert out["accepted"] is True
    assert ex.terminated is True


def test_search_transform_homophonic_blocks_already_transformed_branch():
    ex = _executor_for("ABCDE", separator=None)
    ex.workspace.apply_transform_pipeline(
        "main",
        {"steps": [{"name": "Reverse", "data": {"rangeStart": 0, "rangeEnd": 4}}]},
    )

    out = ex._tool_search_transform_homophonic({
        "branch": "main",
        "profile": "small",
        "homophonic_budget": "screen",
    })

    assert out["status"] == "blocked"
    assert out["reason"] == "transform_branch_not_supported"
    assert "search_homophonic_anneal" in out["suggested_next_tools"]


def test_branch_snapshot_records_transform_overlay_metadata():
    ex = _executor_for("ABCDE", separator=None)
    pipeline = {"steps": [{"name": "Reverse", "data": {"rangeStart": 0, "rangeEnd": 4}}]}
    ex.workspace.apply_transform_pipeline("main", pipeline)

    snapshot = _branch_snapshot_for(ex.workspace, "main")

    assert snapshot.token_order == [4, 3, 2, 1, 0]
    assert snapshot.transform_pipeline == pipeline


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
    assert "not available on this turn" in raw
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


# ---------------------------------------------------------------------------
# meta_attest_reading_comprehensibility — Problem 1 fix
# ---------------------------------------------------------------------------

def test_attest_reading_comprehensibility_records_strong_attestation():
    """score>=8 with valid excerpt is accepted and unlocks declaration."""
    # Use a branch with decoded_text long enough to have a 20-char excerpt
    raw = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "QUICK", "BROWN"},
        word_list=["THE", "QUICK", "BROWN"],
        pattern_dict={},
    )
    branch = ex.workspace.get_branch("main")
    # Set a decoded_text with enough letters to extract an excerpt
    branch.metadata["decoded_text"] = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"

    out = ex._tool_meta_attest_reading_comprehensibility({
        "branch": "main",
        "comprehensibility_score": 9,
        "verbatim_excerpt": "QUICK BROWN FOX JUMPS OVER",   # 22 letters
        "reading_notes": "Clearly natural English sentence.",
    })

    assert out["status"] == "recorded"
    assert out["strong_attestation"] is True
    assert ex._reading_attestations["main"]["comprehensibility_score"] == 9


def test_attest_reading_comprehensibility_rejects_short_excerpt():
    """Excerpts with fewer than 20 alphabetic chars are rejected."""
    raw = "ABCDE"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )
    ex.workspace.get_branch("main").metadata["decoded_text"] = "ABCDE ABCDE ABCDE"

    out = ex._tool_meta_attest_reading_comprehensibility({
        "branch": "main",
        "comprehensibility_score": 9,
        "verbatim_excerpt": "ABC",
    })

    assert out["status"] == "rejected"
    assert out["reason"] == "excerpt_too_short"


def test_attest_reading_comprehensibility_rejects_fabricated_excerpt():
    """An excerpt that doesn't appear in the decoded text is rejected."""
    raw = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )
    ex.workspace.get_branch("main").metadata["decoded_text"] = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    out = ex._tool_meta_attest_reading_comprehensibility({
        "branch": "main",
        "comprehensibility_score": 10,
        # fabricated: these letters don't appear in order in the decoded text
        "verbatim_excerpt": "ZZZZZZZZZZZZZZZZZZZZZZZZZZ",
    })

    assert out["status"] == "rejected"
    assert out["reason"] == "excerpt_not_found"


def test_attest_reading_comprehensibility_weak_score_does_not_bypass_gate():
    """score < 8 is recorded but does NOT bypass the family-coverage gate."""
    raw = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )
    branch = ex.workspace.get_branch("main")
    branch.metadata["decoded_text"] = "THE QUICK BROWN FOX JUMPS OVER LAZY DOG"

    out = ex._tool_meta_attest_reading_comprehensibility({
        "branch": "main",
        "comprehensibility_score": 6,
        "verbatim_excerpt": "THE QUICK BROWN FOX JUMPS",
    })
    assert out["status"] == "recorded"
    assert out["strong_attestation"] is False
    assert not ex._has_strong_reading_attestation("main")


def test_family_coverage_gate_bypassed_by_strong_attestation():
    """A strong attestation (score>=8) bypasses the family_coverage_pending gate."""
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.benchmark_context = ScopedBenchmarkContext(
        policy="historical",
        injected_layers=[{
            "record_id": "kryptos_k2",
            "layer": "standard",
            "label": "Standard cipher metadata",
            "contains_cipher_type_hint": True,
            "text": "K2 is a keyed Vigenere-style polyalphabetic cipher.",
        }],
        target_record_ids=["kryptos_k2"],
    )
    # Create the hypothesis branch so the coverage gate would normally fire
    hyp = ex.workspace.fork("hyp_keyed", from_branch="main")
    hyp.metadata.update({
        "cipher_mode": "keyed_tableau_polyalphabetic",
        "context_supported_mode": True,
        "mode_status": "active",
        "evidence_source": "benchmark_context",
    })
    ex.workspace.tag("hyp_keyed", "hypothesis")

    # No quagmire search has been run → would normally trigger family_coverage_pending
    # But we record a strong reading attestation directly on "main"
    main_branch = ex.workspace.get_branch("main")
    main_branch.metadata["decoded_text"] = (
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG TODAY"
    )
    ex._reading_attestations["main"] = {
        "branch": "main",
        "comprehensibility_score": 9,
        "verbatim_excerpt": "THE QUICK BROWN FOX JUMPS",
        "iteration": 5,
    }
    # _family_coverage_declaration_block should return None (no block)
    result = ex._family_coverage_declaration_block("main", forced_partial=False)
    assert result is None


def test_family_coverage_gate_mentions_attest_tool_in_suggested_tools():
    """When family_coverage_pending fires, the error lists meta_attest tool."""
    ex = _executor_for("ABCABCABCABC", separator=None)
    ex.benchmark_context = ScopedBenchmarkContext(
        policy="historical",
        injected_layers=[{
            "record_id": "kryptos_k2",
            "layer": "standard",
            "label": "Standard cipher metadata",
            "contains_cipher_type_hint": True,
            "text": "K2 is a keyed Vigenere-style polyalphabetic cipher.",
        }],
        target_record_ids=["kryptos_k2"],
    )
    # Create a hypothesis branch with context_supported_mode so the
    # _context_keyed_tableau_prior fires (agent must declare this explicitly)
    hyp = ex.workspace.fork("hyp_keyed", from_branch="main")
    hyp.metadata.update({
        "cipher_mode": "keyed_tableau_polyalphabetic",
        "context_supported_mode": True,
        "mode_status": "active",
        "evidence_source": "benchmark_context",
        "hypothesis_notes": "Context says keyed Vigenere.",
    })
    ex.workspace.tag("hyp_keyed", "hypothesis")
    ex.max_iterations = 20
    ex._current_iteration = 5  # not at limit
    result = ex._family_coverage_declaration_block("main", forced_partial=False)
    assert result is not None
    assert result["reason"] == "family_coverage_pending"
    assert "meta_attest_reading_comprehensibility" in result["suggested_next_tools"]
    assert "meta_attest_reading_comprehensibility" in result["note"]


# ---------------------------------------------------------------------------
# coverage_upgrade_of metadata inheritance — Problem 2 fix
# ---------------------------------------------------------------------------

def test_has_seen_branch_cards_inherits_from_coverage_upgrade_parent():
    """branch_cards call for parent branch counts for upgrade child."""
    from artifact.schema import ToolCall

    ex = _executor_for("ABCABCABCABC", separator=None)

    # Create parent quagmire3 branch at iteration 2
    parent = ex.workspace.fork("quag3_early_1", from_branch="main")
    parent.metadata["cipher_mode"] = "quagmire3"
    parent.created_iteration = 2

    # Simulate agent calling workspace_branch_cards for the parent at iteration 4
    ex.call_log.append(ToolCall(
        tool_name="workspace_branch_cards",
        tool_use_id="bc-parent",
        arguments={"branch": "quag3_early_1"},
        result="{}",
        iteration=4,
    ))

    # Create upgrade branch at iteration 6 (after the branch_cards call)
    upgrade = ex.workspace.fork("quag3_upgrade_1", from_branch="main")
    upgrade.metadata["cipher_mode"] = "quagmire3"
    upgrade.metadata["coverage_upgrade_of"] = "quag3_early_1"
    upgrade.created_iteration = 6

    # Without fix: would fail because branch_cards was at iter 4 < min_iter 6
    # With fix: should pass because we use parent's created_iteration = 2
    assert ex._has_seen_branch_cards("quag3_upgrade_1") is True


def test_has_seen_hypothesis_next_steps_inherits_from_coverage_upgrade_parent():
    """hypothesis_next_steps call for parent counts for upgrade child."""
    from artifact.schema import ToolCall

    ex = _executor_for("ABCABCABCABC", separator=None)

    parent = ex.workspace.fork("quag3_early_1", from_branch="main")
    parent.metadata["cipher_mode"] = "quagmire3"
    parent.created_iteration = 2

    ex.call_log.append(ToolCall(
        tool_name="workspace_hypothesis_next_steps",
        tool_use_id="hns-parent",
        arguments={"branch": "quag3_early_1"},
        result="{}",
        iteration=5,
    ))

    upgrade = ex.workspace.fork("quag3_upgrade_1", from_branch="main")
    upgrade.metadata["cipher_mode"] = "quagmire3"
    upgrade.metadata["coverage_upgrade_of"] = "quag3_early_1"
    upgrade.created_iteration = 7

    assert ex._has_seen_hypothesis_next_steps("quag3_upgrade_1") is True


def test_has_seen_branch_cards_no_inherit_without_coverage_upgrade_of():
    """Without coverage_upgrade_of, original strict min_iteration check applies."""
    from artifact.schema import ToolCall

    ex = _executor_for("ABCABCABCABC", separator=None)

    branch = ex.workspace.fork("quag3_v2_1", from_branch="main")
    branch.metadata["cipher_mode"] = "quagmire3"
    branch.created_iteration = 6  # branch created at iteration 6

    # branch_cards was called at iteration 4 (before branch creation)
    ex.call_log.append(ToolCall(
        tool_name="workspace_branch_cards",
        tool_use_id="bc-before",
        arguments={"branch": "quag3_v2_1"},
        result="{}",
        iteration=4,
    ))

    # Without coverage_upgrade_of: should still be blocked (call before creation)
    assert ex._has_seen_branch_cards("quag3_v2_1") is False


def test_coverage_upgrade_of_is_set_on_new_branch_when_prior_quag3_exists(monkeypatch):
    """When a quagmire3 branch exists, new branches get coverage_upgrade_of."""
    import analysis.polyalphabetic as poly

    ex = _executor_for("ABCABCABC", separator=None)

    # Simulate an existing quagmire3 branch
    prior = ex.workspace.fork("quag3_early_KEYWORD_CW_1", from_branch="main")
    prior.metadata["cipher_mode"] = "quagmire3"
    prior.created_iteration = 3

    # Patch the search to return a fake completed result
    monkeypatch.setattr(
        poly,
        "search_quagmire3_keyword_alphabet",
        lambda *a, **kw: {
            "status": "completed",
            "solver": "python_screen",
            "top_candidates": [{
                "score": 0.8,
                "selection_score": 0.8,
                "preview": "THE CAT",
                "plaintext": "THE CAT SAT ON THE MAT",
                "period": 4,
                "metadata": {
                    "alphabet_keyword": "NEWKW",
                    "cycleword": "NEWCW",
                    "quagmire_type": "quag3",
                },
                "key": "NEWCW",
                "shifts": [0, 1, 2, 3],
            }],
        },
    )

    result = json.loads(ex.execute("search_quagmire3_keyword_alphabet", {
        "branch": "main",
        "engine": "python_screen",
        "keyword_lengths": [5],
        "cycleword_lengths": [4],
        "install_top_n": 1,
    }))

    installed = result.get("installed_branches", [])
    assert len(installed) == 1
    new_branch_name = installed[0]["branch"]
    new_branch = ex.workspace.get_branch(new_branch_name)
    assert new_branch.metadata.get("coverage_upgrade_of") == "quag3_early_KEYWORD_CW_1"


def test_coverage_upgrade_of_not_set_when_no_prior_quag3_branch(monkeypatch):
    """If no prior quagmire3 branch exists, coverage_upgrade_of is not set."""
    import analysis.polyalphabetic as poly

    ex = _executor_for("ABCABCABC", separator=None)

    monkeypatch.setattr(
        poly,
        "search_quagmire3_keyword_alphabet",
        lambda *a, **kw: {
            "status": "completed",
            "solver": "python_screen",
            "top_candidates": [{
                "score": 0.8,
                "selection_score": 0.8,
                "preview": "THE CAT",
                "plaintext": "THE CAT SAT ON THE MAT",
                "period": 4,
                "metadata": {
                    "alphabet_keyword": "FIRSKW",
                    "cycleword": "FIRSCW",
                    "quagmire_type": "quag3",
                },
                "key": "FIRSCW",
                "shifts": [0, 1, 2, 3],
            }],
        },
    )

    result = json.loads(ex.execute("search_quagmire3_keyword_alphabet", {
        "branch": "main",
        "engine": "python_screen",
        "keyword_lengths": [5],
        "cycleword_lengths": [4],
        "install_top_n": 1,
    }))

    installed = result.get("installed_branches", [])
    assert len(installed) == 1
    new_branch_name = installed[0]["branch"]
    new_branch = ex.workspace.get_branch(new_branch_name)
    assert "coverage_upgrade_of" not in new_branch.metadata
