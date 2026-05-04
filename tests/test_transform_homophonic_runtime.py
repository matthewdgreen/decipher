from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from automated.transform_homophonic_runtime import (
    build_transform_homophonic_batch_context,
    plaintext_quality_score,
    select_transform_confirmation_finalists,
    transform_homophonic_scoring_policy,
    transform_homophonic_probe_policy,
    transform_mutation_penalty,
    transform_selection_score,
)


def test_runtime_context_builds_batch_context_from_runner_inputs():
    calls = []

    def budget_params_fn(budget, short_homophonic, *, search_profile):
        calls.append((budget, short_homophonic, search_profile))
        return {"epochs": 2, "sampler_iterations": 30, "seeds": [3, "4"]}

    context = build_transform_homophonic_batch_context(
        language="en",
        token_count=340,
        budget="screen",
        model_path=Path("/tmp/model.bin"),
        plaintext_symbols=["a", "B"],
        search_profile="dev",
        threads=7,
        budget_params_fn=budget_params_fn,
        purpose="test probe",
    )

    assert calls == [("screen", True, "dev")]
    assert context.model_path == "/tmp/model.bin"
    assert context.plaintext_ids == [0, 1]
    assert context.id_to_letter == {0: "A", 1: "B"}
    assert context.epochs == 2
    assert context.sampler_iterations == 30
    assert context.seeds == [3, 4]
    assert context.threads == 7


def test_runtime_context_reports_missing_model_with_purpose():
    try:
        build_transform_homophonic_batch_context(
            language="la",
            token_count=120,
            budget="full",
            model_path=None,
            plaintext_symbols=["A"],
            search_profile="full",
            threads=0,
            budget_params_fn=lambda *args, **kwargs: {},
            purpose="rank test",
        )
    except FileNotFoundError as exc:
        assert "rank test requires an ngram5 model for language='la'" in str(exc)
    else:
        raise AssertionError("expected missing model error")


def test_transform_homophonic_runtime_scoring_policy_matches_components():
    assert transform_mutation_penalty({"provenance": "local_mutation"}) == 0.08
    assert transform_mutation_penalty({
        "provenance": "program_search",
        "params": {"program_depth": 9},
    }) == 0.12
    assert transform_mutation_penalty({
        "provenance": "program_search",
        "params": {"template": "route_repair_constructed", "program_depth": 9},
    }) == 0.0

    quality = plaintext_quality_score("THEANDTHISWASNOTGIBBERISHTEXT", "en")
    assert quality > 0
    selection = transform_selection_score(
        anneal_score=-5.0,
        quality_score=quality,
        structural_score=2.0,
        mutation_penalty=0.08,
    )
    policy = transform_homophonic_scoring_policy("en")
    assert policy.quality_score_fn("THEANDTHISWASNOTGIBBERISHTEXT") == quality
    assert policy.mutation_penalty_fn({"provenance": "local_mutation"}) == 0.08
    assert policy.selection_score_fn(
        anneal_score=-5.0,
        quality_score=quality,
        structural_score=2.0,
        mutation_penalty=0.08,
    ) == selection


def test_transform_homophonic_probe_policy_and_finalist_selection():
    policy = transform_homophonic_probe_policy(
        budget="full",
        adaptive_confirmations=2,
    )
    assert policy["rank_top_n"] == 3
    assert policy["confirmation_policy"].budget == "full"
    assert policy["confirmation_policy"].adaptive_confirmations == 2

    ranked = [
        {"candidate_id": "a", "status": "completed", "pipeline": {"steps": []}},
        {"candidate_id": "b", "status": "error", "pipeline": {"steps": []}},
        {"candidate_id": "c", "status": "completed", "pipeline": {"steps": []}},
        {"candidate_id": "000_identity", "status": "completed", "pipeline": {"steps": []}},
    ]
    finalists = select_transform_confirmation_finalists(ranked, confirm_count=1)
    assert [item["candidate_id"] for item in finalists] == ["a", "000_identity"]

    ranked_with_identity_first = [ranked[3], *ranked[:3]]
    finalists = select_transform_confirmation_finalists(
        ranked_with_identity_first,
        confirm_count=2,
    )
    assert [item["candidate_id"] for item in finalists] == ["000_identity", "a"]
