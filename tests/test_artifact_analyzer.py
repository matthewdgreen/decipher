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
