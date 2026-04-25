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
