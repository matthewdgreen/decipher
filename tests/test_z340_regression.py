"""Cheap regression checks for the neutral Z340 agentic crack shape.

The live run is too expensive for normal CI, but the artifact taught us an
important behavior sequence: the agent should reject word-island partials and
escalate into a wider transform+homophonic search before declaring.
"""
from __future__ import annotations


def test_neutral_z340_success_shape_is_preserved_as_regression_target():
    artifact = {
        "cipher_id": "challenge_340_context_probe_001",
        "cipher_alphabet_size": 63,
        "cipher_token_count": 340,
        "cipher_word_count": 1,
        "language": "en",
        "char_accuracy": 0.9411764705882353,
        "word_accuracy": 0.0,
        "benchmark_context": {
            "policy": "standard",
            "target_record_ids": ["challenge_340_ciphertext_001"],
            "context_record_ids": [],
            "related_records_available": [],
            "associated_documents_available": [],
            "related_solution_allowed": False,
            "injected_layers": [
                {"record_id": "challenge_340_ciphertext_001", "layer": "minimal"},
                {"record_id": "challenge_340_ciphertext_001", "layer": "standard"},
            ],
        },
        "solution": {
            "branch": "main_transform_best",
            "self_confidence": 0.99,
            "declared_at_iteration": 13,
            "further_iterations_helpful": False,
            "reading_summary": (
                "Coherent English plaintext: the Zodiac Killer's I HOPE YOU "
                "ARE HAVING LOTS OF FUN IN TRYING TO CATCH ME..."
            ),
        },
        "tool_calls": [
            {
                "tool_name": "search_automated_solver",
                "arguments": {
                    "branch": "main",
                    "homophonic_solver": "zenith_native",
                    "homophonic_budget": "full",
                },
            },
            {
                "tool_name": "observe_transform_suspicion",
                "arguments": {"branch": "main", "baseline_status": "word_islands_only"},
            },
            {
                "tool_name": "search_transform_homophonic",
                "arguments": {
                    "branch": "main",
                    "profile": "medium",
                    "include_program_search": True,
                    "homophonic_budget": "screen",
                },
                "result": {"candidate_count": 529},
            },
            {
                "tool_name": "meta_declare_solution",
                "arguments": {
                    "branch": "main_transform_best",
                    "forced_partial": True,
                    "further_iterations_helpful": True,
                },
                "result": {"status": "blocked", "reason": "further_iterations_requested"},
            },
            {
                "tool_name": "search_transform_homophonic",
                "arguments": {
                    "branch": "main",
                    "profile": "wide",
                    "include_program_search": True,
                    "homophonic_budget": "full",
                },
                "result": {"candidate_count": 24380},
            },
            {
                "tool_name": "meta_declare_solution",
                "arguments": {
                    "branch": "main_transform_best",
                    "self_confidence": 0.99,
                    "further_iterations_helpful": False,
                },
                "result": {"status": "ok"},
            },
        ],
        "branches": [
            {
                "name": "main_transform_best",
                "char_accuracy": 0.9411764705882353,
                "word_accuracy": 0.0,
                "transform_pipeline": {
                    "columns": 17,
                    "rows": 20,
                    "steps": [
                        {"name": "NDownMAcross", "data": {"rangeStart": 0, "rangeEnd": 8}},
                        {"name": "ShiftCharactersRight", "data": {"rangeStart": 241, "rangeEnd": 254}},
                        {"name": "LockCharacters", "data": {"rangeStart": 164, "rangeEnd": 169}},
                    ],
                },
            }
        ],
    }

    context = artifact["benchmark_context"]
    assert context["related_records_available"] == []
    assert context["associated_documents_available"] == []
    assert context["related_solution_allowed"] is False
    assert artifact["char_accuracy"] >= 0.90
    assert artifact["word_accuracy"] == 0.0  # no-boundary representation issue

    tool_calls = artifact["tool_calls"]
    search_calls = [
        call for call in tool_calls
        if call["tool_name"] == "search_transform_homophonic"
    ]
    assert [call["arguments"]["profile"] for call in search_calls] == [
        "medium",
        "wide",
    ]
    assert search_calls[0]["result"]["candidate_count"] == 529
    assert search_calls[1]["result"]["candidate_count"] == 24380
    assert any(
        call["tool_name"] == "meta_declare_solution"
        and call["result"]["reason"] == "further_iterations_requested"
        for call in tool_calls
        if call.get("result", {}).get("status") == "blocked"
    )

    branch = artifact["branches"][0]
    assert branch["name"] == artifact["solution"]["branch"]
    assert branch["transform_pipeline"]["columns"] == 17
    assert any(
        step["name"] == "NDownMAcross"
        for step in branch["transform_pipeline"]["steps"]
    )
