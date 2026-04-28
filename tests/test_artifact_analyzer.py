from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from artifact.analyzer import analyze_artifact, summarize_findings


def test_analyzer_flags_delayed_homophonic_search_and_bad_swap():
    artifact = {
        "cipher_alphabet_size": 57,
        "cipher_word_count": 1,
        "solution": {"branch": "homo1", "declared_at_iteration": 7},
        "branches": [{"name": "homo1", "char_accuracy": 0.93}],
        "tool_calls": [
            {"iteration": 1, "tool_name": "observe_frequency", "result": "{}"},
            {"iteration": 4, "tool_name": "search_homophonic_anneal", "result": "{}"},
            {
                "iteration": 6,
                "tool_name": "act_swap_decoded",
                "arguments": {"branch": "homo1", "letter_a": "B", "letter_b": "D"},
                "result": {
                    "status": "ok",
                    "score_delta": {"verdict": "worse"},
                },
            },
            {"iteration": 7, "tool_name": "meta_declare_solution", "result": "{}"},
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert summary["labels"]["agent_wrong_tool"] == 2
    assert summary["labels"]["premature_declaration"] == 2


def test_analyzer_does_not_flag_reverted_worsening_swap_as_declaration_error():
    artifact = {
        "cipher_alphabet_size": 57,
        "cipher_word_count": 1,
        "tool_calls": [
            {"iteration": 1, "tool_name": "search_homophonic_anneal", "result": "{}"},
            {
                "iteration": 2,
                "tool_name": "act_swap_decoded",
                "arguments": {"branch": "main", "letter_a": "B", "letter_b": "D"},
                "result": {
                    "status": "reverted",
                    "score_delta": {"verdict": "worse"},
                },
            },
            {"iteration": 3, "tool_name": "meta_declare_solution", "result": "{}"},
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert summary["labels"]["agent_wrong_tool"] == 1
    assert summary["labels"]["scoring_false_positive"] == 1
    assert "premature_declaration" not in summary["labels"]


def test_analyzer_flags_declared_branch_that_is_not_best_final_branch():
    artifact = {
        "cipher_alphabet_size": 26,
        "cipher_word_count": 1,
        "solution": {"branch": "main", "declared_at_iteration": 5},
        "branches": [
            {"name": "main", "char_accuracy": 0.75},
            {"name": "candidate", "char_accuracy": 0.92},
        ],
        "tool_calls": [],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert summary["labels"] == {"premature_declaration": 1}


def test_analyzer_flags_score_overrode_reading():
    """Reading-driven fix touched 2+ words and dropped dict_rate; the run
    then declared a different branch. This is the central failure pattern
    the label exists to catch."""
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "solution": {"branch": "polish", "declared_at_iteration": 15},
        "branches": [
            {"name": "polish", "char_accuracy": 0.14},
            {"name": "test_swap", "char_accuracy": 0.13},
        ],
        "tool_calls": [
            {
                "iteration": 13,
                "tool_name": "act_set_mapping",
                "arguments": {"branch": "test_swap", "cipher_symbol": "P", "plain_letter": "B"},
                "result": {
                    "status": "ok",
                    "branch": "test_swap",
                    "mapping": "P -> B",
                    "score_delta": {"dict_rate_delta": -0.012, "quad_delta": -0.08},
                    "changed_words": [
                        {"index": 11, "before": "TREUITER", "after": "BREUITER"},
                        {"index": 27, "before": "MORIETANTUR", "after": "MORIEBANTUR"},
                        {"index": 21, "before": "LITE", "after": "LIBE"},
                        {"index": 22, "before": "TITUR", "after": "BITUR"},
                    ],
                },
            },
            {"iteration": 15, "tool_name": "meta_declare_solution", "result": "{}"},
        ],
        "messages": [],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert summary["labels"].get("score_overrode_reading") == 1


def test_analyzer_does_not_flag_score_overrode_when_change_is_kept():
    """Same negative-delta + multi-word change, but the declared branch IS
    the branch the change was applied to — no override happened."""
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "solution": {"branch": "repair", "declared_at_iteration": 10},
        "branches": [{"name": "repair", "char_accuracy": 0.20}],
        "tool_calls": [
            {
                "iteration": 5,
                "tool_name": "act_set_mapping",
                "arguments": {"branch": "repair", "cipher_symbol": "P", "plain_letter": "B"},
                "result": {
                    "status": "ok",
                    "branch": "repair",
                    "score_delta": {"dict_rate_delta": -0.01},
                    "changed_words": [
                        {"index": 1, "before": "FOO", "after": "BAR"},
                        {"index": 2, "before": "BAZ", "after": "QUX"},
                    ],
                },
            },
            {"iteration": 10, "tool_name": "meta_declare_solution", "result": "{}"},
        ],
        "messages": [],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert "score_overrode_reading" not in summary["labels"]


def test_analyzer_flags_unresolved_repair_agenda():
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "solution": {"branch": "repair", "declared_at_iteration": 12},
        "repair_agenda": [
            {
                "id": 1,
                "branch": "repair",
                "from": "TREUITER",
                "to": "BREUITER",
                "status": "open",
            },
            {
                "id": 2,
                "branch": "repair",
                "from": "SIMALITER",
                "to": "SIMILITER",
                "status": "applied",
            },
        ],
        "tool_calls": [],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert summary["labels"]["repair_agenda_unresolved"] == 1


def test_analyzer_treats_held_repair_agenda_as_resolved():
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "solution": {"branch": "repair", "declared_at_iteration": 12},
        "repair_agenda": [
            {
                "id": 1,
                "branch": "repair",
                "from": "RLVRES",
                "to": "PLVRES",
                "status": "held",
            },
        ],
        "tool_calls": [],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert "repair_agenda_unresolved" not in summary["labels"]


def test_analyzer_flags_unattempted_reading_fix():
    """Assistant text proposes a cipher-symbol → letter fix; no mutation
    primitive follows within two iterations."""
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "tool_calls": [
            {"iteration": 1, "tool_name": "decode_show", "result": "{}"},
            {"iteration": 2, "tool_name": "decode_letter_stats", "result": "{}"},
            {"iteration": 3, "tool_name": "search_anneal", "result": "{}"},
        ],
        "messages": [
            {"role": "user", "content": "system + initial context"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Looking at the decode."},
                    {"type": "tool_use", "name": "decode_show", "input": {}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "TREUITER → BREUITER: cipher P should be B. "
                            "I will run an anneal to polish."
                        ),
                    },
                    {"type": "tool_use", "name": "decode_letter_stats", "input": {}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running anneal."},
                    {"type": "tool_use", "name": "search_anneal", "input": {}},
                ],
            },
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert summary["labels"].get("unattempted_reading_fix") == 1


def test_analyzer_does_not_flag_unattempted_when_mutation_follows():
    """Same hypothesis, but act_set_mapping is called next iteration."""
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "tool_calls": [
            {"iteration": 1, "tool_name": "decode_show", "result": "{}"},
            {"iteration": 2, "tool_name": "act_set_mapping", "result": "{}"},
        ],
        "messages": [
            {"role": "user", "content": "init"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "TREUITER should be BREUITER — cipher P → B.",
                    },
                    {"type": "tool_use", "name": "decode_show", "input": {}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Applying the fix."},
                    {"type": "tool_use", "name": "act_set_mapping", "input": {}},
                ],
            },
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert "unattempted_reading_fix" not in summary["labels"]


def test_analyzer_does_not_flag_speculative_reading_notes_as_unattempted():
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "tool_calls": [
            {"iteration": 1, "tool_name": "search_anneal", "result": "{}"},
            {"iteration": 2, "tool_name": "workspace_fork", "result": "{}"},
        ],
        "messages": [
            {"role": "user", "content": "init"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "I can see SIMALITER → SIMILITER ✓ and "
                            "MORIETANTER → likely MORIRENTUR or similar. "
                            "Let me fork the better branch and inspect more."
                        ),
                    },
                    {"type": "tool_use", "name": "workspace_fork", "input": {}},
                ],
            },
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert "unattempted_reading_fix" not in summary["labels"]


def test_analyzer_counts_word_repair_plan_as_attempted_reading_fix():
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "tool_calls": [
            {"iteration": 1, "tool_name": "decode_plan_word_repair", "result": "{}"},
        ],
        "messages": [
            {"role": "user", "content": "init"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Key repairs I can read: SIMALITER → SIMILITER. "
                            "Let me plan the clearest word repair."
                        ),
                    },
                    {
                        "type": "tool_use",
                        "name": "decode_plan_word_repair",
                        "input": {},
                    },
                ],
            },
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert "unattempted_reading_fix" not in summary["labels"]


def test_analyzer_counts_word_repair_menu_as_attempted_reading_fix():
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "tool_calls": [
            {"iteration": 1, "tool_name": "decode_plan_word_repair_menu", "result": "{}"},
        ],
        "messages": [
            {"role": "user", "content": "init"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "RLURES -> PLURES might be right, but I need to "
                            "compare repair options before applying it."
                        ),
                    },
                    {
                        "type": "tool_use",
                        "name": "decode_plan_word_repair_menu",
                        "input": {},
                    },
                ],
            },
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert "unattempted_reading_fix" not in summary["labels"]


def test_analyzer_does_not_flag_rejected_repair_notes_as_unattempted():
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "tool_calls": [
            {"iteration": 15, "tool_name": "meta_declare_solution", "result": "{}"},
        ],
        "messages": [
            {"role": "user", "content": "init"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "I can see repair #2 (RLURES→PLURES) actually "
                            "broke things badly (M→P conflicted with R). "
                            "The current workspace shows the branch reads as "
                            "excellent Latin. Let me declare now."
                        ),
                    },
                    {
                        "type": "tool_use",
                        "name": "meta_declare_solution",
                        "input": {},
                    },
                ],
            },
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert "unattempted_reading_fix" not in summary["labels"]


def test_analyzer_labels_gated_and_projection_retry_events():
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "tool_calls": [],
        "loop_events": [
            {
                "event": "gated_tool_retry",
                "outer_iteration": 13,
                "inner_step": 1,
                "mode": "boundary_projection",
                "payload": {
                    "attempted_tools": ["search_anneal"],
                    "allowed_tools": ["decode_show"],
                },
            },
            {
                "event": "boundary_projection_count_retry",
                "outer_iteration": 14,
                "inner_step": 1,
                "mode": "boundary_projection",
                "payload": {
                    "attempted_tools": ["act_resegment_from_reading_repair"],
                },
            },
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert summary["labels"]["gated_tool_retry"] == 1
    assert summary["labels"]["same_length_projection_failed"] == 1


def test_analyzer_labels_projection_tool_count_failure():
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "tool_calls": [
            {
                "iteration": 13,
                "tool_name": "decode_validate_reading_repair",
                "result": {
                    "branch": "main",
                    "same_character_count": False,
                    "current_char_count": 345,
                    "proposed_char_count": 178,
                },
            },
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert summary["labels"]["same_length_projection_failed"] == 1


def test_analyzer_treats_word_repair_as_reading_primitive():
    artifact = {
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "messages": [
            {"role": "user", "content": "init"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "TREUITER should be BREUITER."},
                    {"type": "tool_use", "name": "decode_show", "input": {}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Applying the word repair."},
                    {"type": "tool_use", "name": "act_apply_word_repair", "input": {}},
                ],
            },
        ],
    }

    summary = summarize_findings(analyze_artifact(artifact))

    assert "unattempted_reading_fix" not in summary["labels"]
