from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.transform_homophonic_batch import (
    build_zenith_transform_batch_request,
    build_zenith_transform_batch_context,
    build_confirmation_batch_payload,
    confirmation_seed_offset,
    dedupe_transform_batch_candidates,
    failed_batch_candidate_record,
    failed_confirmation_record,
    missing_pipeline_confirmation_record,
    normalize_rust_rank_batch_results,
    rust_rank_candidate_record,
    run_zenith_transform_batch,
    run_zenith_transform_confirmation_batches,
    run_zenith_transform_rank_batch,
    successful_confirmation_record,
    transform_pipeline_key,
    ZenithTransformConfirmationPolicy,
    ZenithTransformRankScoringPolicy,
)


def test_dedupe_transform_batch_candidates_uses_stable_pipeline_key():
    pipeline = {"steps": [{"name": "Reverse", "data": {"rangeEnd": 9, "rangeStart": 0}}]}
    candidates = [
        {"candidate_id": "a", "pipeline": pipeline, "family": "reverse"},
        {"candidate_id": "b", "pipeline": {"steps": [{"data": {"rangeStart": 0, "rangeEnd": 9}, "name": "Reverse"}]}},
        {"candidate_id": "c", "pipeline": {"steps": []}},
    ]

    payload, metadata = dedupe_transform_batch_candidates(candidates)

    assert transform_pipeline_key(candidates[0]["pipeline"]) == transform_pipeline_key(candidates[1]["pipeline"])
    assert [item["candidate_id"] for item in payload] == ["a", "c"]
    assert set(metadata) == {"a", "c"}
    assert metadata["a"]["family"] == "reverse"


def test_zenith_transform_batch_request_and_runner_shape_call_kwargs():
    context = build_zenith_transform_batch_context(
        plaintext_symbols=["A", "b"],
        model_path="/tmp/model.bin",
        budget_params={"epochs": "2", "sampler_iterations": "30", "seeds": ["1", 2]},
        threads=8,
    )
    request = build_zenith_transform_batch_request(
        tokens=[3, 2, 1],
        candidates=[{"candidate_id": "a", "pipeline": {"steps": []}}],
        context=context,
        top_n=5,
    )
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "candidate_count": len(kwargs["candidates"])}

    result = run_zenith_transform_batch(request, batch_runner=fake_runner)

    assert result == {"status": "ok", "candidate_count": 1}
    assert captured == {
        "tokens": [3, 2, 1],
        "candidates": [{"candidate_id": "a", "pipeline": {"steps": []}}],
        "plaintext_ids": [0, 1],
        "id_to_letter": {0: "A", 1: "B"},
        "model_path": "/tmp/model.bin",
        "epochs": 2,
        "sampler_iterations": 30,
        "seeds": [1, 2],
        "top_n": 5,
        "threads": 8,
    }


def test_build_confirmation_batch_payload_adds_seeded_batch_ids_and_metadata():
    items = [
        {"candidate_id": "cand1", "family": "route", "pipeline": {"steps": []}, "grid": {"columns": 17}},
        {"candidate_id": "cand2", "family": "route", "pipeline": {"steps": [{"name": "Reverse", "data": {}}]}},
    ]

    payload, metadata, missing = build_confirmation_batch_payload(
        items,
        reason="initial_finalist",
        start_index=3,
    )

    assert missing == []
    assert [item["seed_offset"] for item in payload] == [
        confirmation_seed_offset(3),
        confirmation_seed_offset(4),
    ]
    assert payload[0]["candidate_id"] == "cand1__confirm_13000"
    assert payload[0]["grid"] == {"columns": 17}
    assert metadata["cand1__confirm_13000"].item is items[0]
    assert metadata["cand1__confirm_13000"].original_id == "cand1"
    assert metadata["cand2__confirm_14000"].reason == "initial_finalist"


def test_build_confirmation_batch_payload_reports_missing_pipeline():
    items = [
        {"candidate_id": "missing", "family": "route"},
        {"candidate_id": "ok", "pipeline": {"steps": []}},
    ]

    payload, metadata, missing = build_confirmation_batch_payload(
        items,
        reason="adaptive_near_margin",
        start_index=0,
    )

    assert [item["candidate_id"] for item in payload] == ["ok__confirm_11000"]
    assert set(metadata) == {"ok__confirm_11000"}
    assert len(missing) == 1
    assert missing[0]["item"] is items[0]
    assert missing[0]["original_id"] == "missing"
    assert missing[0]["seed_offset"] == 10000
    assert missing[0]["error"] == "missing transform pipeline"


def test_rust_rank_candidate_record_normalizes_artifact_shape():
    row = {
        "candidate_id": "cand",
        "family": "route",
        "decryption": "HELLOWORLD",
        "elapsed_seconds": 1.23456,
        "key": {"1": 2},
        "best_seed": 7,
        "attempts": [{"seed": 7}],
        "token_order_hash": "abc",
    }
    candidate = {
        "candidate_id": "cand",
        "family": "route_columns_down",
        "provenance": "bounded_search",
        "params": {"columns": 17},
        "pipeline": {"steps": []},
        "score": 3.5,
        "delta_vs_identity": 0.25,
        "metrics": {"matrix_rank_score": 0.8, "best_period": 17},
    }

    record = rust_rank_candidate_record(
        row=row,
        candidate=candidate,
        candidate_id="cand",
        anneal_score=-5.1,
        quality_score=0.42,
        mutation_penalty=0.01,
        selection_score=-4.9,
        batch={"elapsed_seconds": 9.0, "threads": 8, "seed_count": 3},
        total_elapsed_seconds=10.2,
    )

    assert record["candidate_id"] == "cand"
    assert record["engine"] == "rust_batch"
    assert record["decryption_preview"] == "HELLOWORLD"
    assert record["plaintext_quality_score"] == 0.42
    assert record["selection_score"] == -4.9
    assert record["elapsed_seconds"] == 1.235
    assert record["key"] == {"1": 2}
    assert record["rust_batch"]["threads"] == 8
    assert record["rust_batch"]["total_elapsed_seconds"] == 10.2


def test_failed_batch_candidate_record_preserves_candidate_metadata():
    row = {"candidate_id": "cand", "family": "row", "reason": "bad pipeline"}
    candidate = {"candidate_id": "cand", "family": "route", "pipeline": {"steps": []}}

    record = failed_batch_candidate_record(row=row, candidate=candidate, candidate_id="cand")

    assert record == {
        "candidate_id": "cand",
        "family": "route",
        "pipeline": {"steps": []},
        "reason": "bad pipeline",
    }


def test_normalize_rust_rank_batch_results_uses_runner_scoring_callbacks():
    batch = {
        "elapsed_seconds": 8.0,
        "threads": 4,
        "seed_count": 2,
        "results": [
            {
                "candidate_id": "good",
                "status": "completed",
                "decryption": "HELLOWORLD",
                "normalized_score": "-5.25",
                "elapsed_seconds": 1.0,
            },
            {
                "candidate_id": "bad",
                "status": "error",
                "reason": "invalid transform",
            },
        ],
    }
    metadata_by_id = {
        "good": {
            "candidate_id": "good",
            "family": "route",
            "pipeline": {"steps": []},
            "score": 0.75,
            "delta_vs_identity": 0.1,
        },
        "bad": {
            "candidate_id": "bad",
            "family": "route",
            "pipeline": {"steps": [{"name": "Bad"}]},
        },
    }
    calls = {"quality": [], "penalty": [], "selection": []}

    def quality_score(text: str) -> float:
        calls["quality"].append(text)
        return 0.5

    def mutation_penalty(candidate: dict) -> float:
        calls["penalty"].append(candidate["candidate_id"])
        return 0.02

    def selection_score(**kwargs) -> float:
        calls["selection"].append(kwargs)
        return -4.7

    ranked, skipped = normalize_rust_rank_batch_results(
        batch=batch,
        metadata_by_id=metadata_by_id,
        total_elapsed_seconds=9.25,
        quality_score_fn=quality_score,
        mutation_penalty_fn=mutation_penalty,
        selection_score_fn=selection_score,
    )

    assert len(ranked) == 1
    assert ranked[0]["candidate_id"] == "good"
    assert ranked[0]["anneal_score"] == -5.25
    assert ranked[0]["plaintext_quality_score"] == 0.5
    assert ranked[0]["local_mutation_penalty"] == 0.02
    assert ranked[0]["selection_score"] == -4.7
    assert ranked[0]["rust_batch"]["total_elapsed_seconds"] == 9.25
    assert skipped == [{
        "candidate_id": "bad",
        "family": "route",
        "pipeline": {"steps": [{"name": "Bad"}]},
        "reason": "invalid transform",
    }]
    assert calls["quality"] == ["HELLOWORLD"]
    assert calls["penalty"] == ["good"]
    assert calls["selection"] == [{
        "anneal_score": -5.25,
        "quality_score": 0.5,
        "structural_score": 0.75,
        "mutation_penalty": 0.02,
    }]


def test_run_zenith_transform_rank_batch_dedupes_runs_and_normalizes_results():
    context = build_zenith_transform_batch_context(
        plaintext_symbols=["A", "B", "C"],
        model_path="/tmp/model.bin",
        budget_params={"epochs": 1, "sampler_iterations": 2, "seeds": [3, 4]},
        threads=6,
    )
    pipeline = {"steps": [{"name": "Reverse", "data": {}}]}
    raw_candidates = [
        {"candidate_id": "a", "family": "reverse", "pipeline": pipeline, "score": 1.0},
        {"candidate_id": "b", "family": "reverse", "pipeline": pipeline, "score": 2.0},
        {"candidate_id": "c", "family": "identity", "pipeline": {"steps": []}, "score": 0.5},
    ]
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {
            "elapsed_seconds": 3.0,
            "threads": kwargs["threads"],
            "seed_count": len(kwargs["seeds"]),
            "results": [
                {
                    "candidate_id": "a",
                    "status": "completed",
                    "decryption": "ABCA",
                    "normalized_score": -2.0,
                },
                {
                    "candidate_id": "c",
                    "status": "error",
                    "reason": "bad candidate",
                },
            ],
        }

    ranked, skipped = run_zenith_transform_rank_batch(
        tokens=[2, 1, 0],
        raw_candidates=raw_candidates,
        context=context,
        top_n=1,
        scoring_policy=ZenithTransformRankScoringPolicy(
            quality_score_fn=lambda text: len(text) / 10.0,
            mutation_penalty_fn=lambda candidate: candidate["score"] / 100.0,
            selection_score_fn=lambda **kwargs: kwargs["anneal_score"] + kwargs["quality_score"],
        ),
        batch_runner=fake_runner,
        elapsed_seconds=lambda: 4.5,
    )

    assert [candidate["candidate_id"] for candidate in captured["candidates"]] == ["a", "c"]
    assert captured["tokens"] == [2, 1, 0]
    assert captured["top_n"] == 1
    assert captured["threads"] == 6
    assert ranked[0]["candidate_id"] == "a"
    assert ranked[0]["selection_score"] == -1.6
    assert ranked[0]["rust_batch"]["total_elapsed_seconds"] == 4.5
    assert skipped == [{
        "candidate_id": "c",
        "family": "identity",
        "pipeline": {"steps": []},
        "reason": "bad candidate",
    }]


def test_confirmation_record_helpers_shape_success_and_failures():
    item = {"candidate_id": "cand", "family": "route"}
    confirmation, skipped, confirmed_score = missing_pipeline_confirmation_record(
        item=item,
        seed_offset=10000,
        reason="initial_finalist",
        error="missing transform pipeline",
        fallback_score=-7.0,
    )
    assert confirmation["status"] == "error"
    assert skipped["reason"] == "missing transform pipeline"
    assert confirmed_score == -7.12

    confirmation, skipped, confirmed_score = failed_confirmation_record(
        row={"reason": "rust failed"},
        item=item,
        seed_offset=11000,
        reason="adaptive_near_margin",
        fallback_score=-6.5,
    )
    assert confirmation["engine"] == "rust_batch"
    assert skipped["confirmation_reason"] == "adaptive_near_margin"
    assert confirmed_score == -6.62

    confirmation, confirmed_score, summary = successful_confirmation_record(
        row={
            "decryption": "HELLOWORLD",
            "elapsed_seconds": 2.25,
            "key": {"3": 4},
            "best_seed": 9,
        },
        item=item,
        seed_offset=12000,
        reason="initial_finalist",
        budget="screen",
        anneal_score=-5.2,
        quality_score=0.7,
        selection_score=-5.0,
        primary_score=-4.95,
        distance=0.1,
    )
    assert confirmation["status"] == "completed"
    assert confirmation["selection_delta_vs_primary"] == -0.05
    assert confirmation["stability_score"] == 0.9
    assert confirmation["confirmation_reasons"] == []
    assert confirmed_score == -5.008
    assert summary["confirmed_selection_score"] == confirmed_score
    assert summary["reasons"] == []


def test_run_zenith_transform_confirmation_batches_mutates_candidates_and_adapts():
    context = build_zenith_transform_batch_context(
        plaintext_symbols=["A", "B"],
        model_path="/tmp/model.bin",
        budget_params={"epochs": 1, "sampler_iterations": 2, "seeds": [1]},
        threads=2,
    )
    ranked = [
        {
            "candidate_id": "top",
            "family": "route",
            "pipeline": {"steps": []},
            "selection_score": -5.0,
            "validated_selection_score": -4.9,
            "structural_score": 0.6,
            "decryption": "AAAA",
        },
        {
            "candidate_id": "missing",
            "family": "route",
            "selection_score": -4.95,
            "validated_selection_score": -4.93,
        },
        {
            "candidate_id": "near",
            "family": "route",
            "pipeline": {"steps": [{"name": "Reverse"}]},
            "selection_score": -4.91,
            "validated_selection_score": -4.91,
            "structural_score": 0.4,
            "decryption": "BBBB",
        },
        {
            "candidate_id": "far",
            "family": "route",
            "pipeline": {"steps": [{"name": "Rows"}]},
            "selection_score": -6.0,
            "validated_selection_score": -6.0,
        },
    ]
    finalists = [ranked[0], ranked[1]]
    batch_calls = []

    def fake_runner(**kwargs):
        batch_calls.append(kwargs)
        results = []
        for candidate in kwargs["candidates"]:
            candidate_id = candidate["candidate_id"]
            if candidate_id.startswith("top__confirm"):
                results.append({
                    "candidate_id": candidate_id,
                    "status": "completed",
                    "decryption": "AAAB",
                    "normalized_score": -5.0,
                    "best_seed": 1,
                })
            elif candidate_id.startswith("near__confirm"):
                results.append({
                    "candidate_id": candidate_id,
                    "status": "completed",
                    "decryption": "BBBB",
                    "normalized_score": -4.8,
                    "best_seed": 2,
                })
            else:
                results.append({
                    "candidate_id": candidate_id,
                    "status": "error",
                    "reason": "unexpected",
                })
        return {"results": results, "elapsed_seconds": 1.0, "threads": 2, "seed_count": 1}

    summary = run_zenith_transform_confirmation_batches(
        tokens=[0, 1, 0, 1],
        ranked=ranked,
        finalists=finalists,
        context=context,
        scoring_policy=ZenithTransformRankScoringPolicy(
            quality_score_fn=lambda text: 0.5 if text == "AAAB" else 0.7,
            mutation_penalty_fn=lambda _candidate: 0.0,
            selection_score_fn=lambda **kwargs: kwargs["anneal_score"] + kwargs["quality_score"],
        ),
        confirmation_policy=ZenithTransformConfirmationPolicy(
            budget="screen",
            adaptive_confirmations=1,
            adaptive_margin=0.2,
            unconfirmed_penalty=0.12,
        ),
        plaintext_distance_fn=lambda left, right: 0.25 if left != right else 0.0,
        batch_runner=fake_runner,
    )

    assert len(batch_calls) == 2
    assert [item["candidate_id"] for item in batch_calls[0]["candidates"]] == ["top__confirm_10000"]
    assert [item["candidate_id"] for item in batch_calls[1]["candidates"]] == ["near__confirm_12000"]
    assert ranked[0]["confirmation"]["status"] == "completed"
    assert ranked[0]["confirmation"]["confirmation_reason"] == "initial_finalist"
    assert ranked[1]["confirmation"]["status"] == "error"
    assert ranked[1]["confirmation"]["error"] == "missing transform pipeline"
    assert ranked[2]["confirmation"]["confirmation_reason"] == "adaptive_near_margin"
    assert ranked[3]["confirmation"]["status"] == "not_run"
    assert summary["confirmed_candidate_count"] == 2
    assert summary["adaptive_confirmed_candidate_count"] == 1
    assert summary["unconfirmed_candidate_count"] == 1
    assert summary["skipped_candidates"][0]["candidate_id"] == "missing"
