"""Regression tests for agent-loop reliability guardrails."""
from __future__ import annotations

import os
import sys
import json
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.loop_v2 import (
    BOUNDARY_PROJECTION_COUNT_RETRY_PREFLIGHT,
    BOUNDARY_PROJECTION_RETRY_PREFLIGHT,
    FINAL_DECLARATION_RETRY_PREFLIGHT,
    FINAL_ITERATION_PREFLIGHT,
    FULL_READING_WORKFLOW_TOOL_NAMES,
    INSPECTION_SANDBOX_CONTINUE_PREFLIGHT,
    PENULTIMATE_READING_WORKFLOW_PREFLIGHT,
    PENULTIMATE_ALLOWED_TOOL_NAMES,
    REPAIR_SANDBOX_CONTINUE_PREFLIGHT,
    READING_WORKFLOW_GATE_PREFLIGHT,
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
    TextBlock,
    ToolUseBlock,
    _messages_to_openai_chat,
    _openai_chat_response_to_model_response,
    _schema_for_gemini,
    _tools_to_openai_chat,
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
