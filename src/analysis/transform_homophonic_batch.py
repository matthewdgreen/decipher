"""Batch-payload helpers for transform+homophonic finalist evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass
import math
import time
from typing import Any, Callable


@dataclass(frozen=True)
class ZenithTransformBatchContext:
    """Shared Rust transform+Zenith settings for a rank/confirmation run."""

    plaintext_ids: list[int]
    id_to_letter: dict[int, str]
    model_path: str
    epochs: int
    sampler_iterations: int
    seeds: list[int]
    threads: int


@dataclass(frozen=True)
class ZenithTransformBatchRequest:
    """Concrete request shape for the Rust transform+Zenith batch kernel."""

    tokens: list[int]
    candidates: list[dict[str, Any]]
    plaintext_ids: list[int]
    id_to_letter: dict[int, str]
    model_path: str
    epochs: int
    sampler_iterations: int
    seeds: list[int]
    top_n: int
    threads: int

    def to_call_kwargs(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "candidates": self.candidates,
            "plaintext_ids": self.plaintext_ids,
            "id_to_letter": self.id_to_letter,
            "model_path": self.model_path,
            "epochs": self.epochs,
            "sampler_iterations": self.sampler_iterations,
            "seeds": self.seeds,
            "top_n": self.top_n,
            "threads": self.threads,
        }


@dataclass(frozen=True)
class ConfirmationBatchMetadata:
    """Runner-side metadata for one Rust confirmation payload candidate."""

    item: dict[str, Any]
    original_id: str
    seed_offset: int
    reason: str


@dataclass(frozen=True)
class ZenithTransformRankScoringPolicy:
    """Caller-owned scoring callbacks for Rust transform rank probes."""

    quality_score_fn: Callable[[str], float]
    mutation_penalty_fn: Callable[[dict[str, Any]], float]
    selection_score_fn: Callable[..., float]


@dataclass(frozen=True)
class ZenithTransformConfirmationPolicy:
    """Confirmation policy knobs for Rust transform finalist reruns."""

    budget: str
    adaptive_confirmations: int = 2
    adaptive_margin: float = 0.04
    unconfirmed_penalty: float = 0.12


def build_zenith_transform_batch_request(
    *,
    tokens: list[int],
    candidates: list[dict[str, Any]],
    context: ZenithTransformBatchContext,
    top_n: int,
) -> ZenithTransformBatchRequest:
    """Build a Rust transform+Zenith request from shared batch context."""

    return ZenithTransformBatchRequest(
        tokens=list(tokens),
        candidates=candidates,
        plaintext_ids=list(context.plaintext_ids),
        id_to_letter=dict(context.id_to_letter),
        model_path=context.model_path,
        epochs=context.epochs,
        sampler_iterations=context.sampler_iterations,
        seeds=list(context.seeds),
        top_n=int(top_n),
        threads=context.threads,
    )


def build_zenith_transform_batch_context(
    *,
    plaintext_symbols: list[str],
    model_path: str,
    budget_params: dict[str, Any],
    threads: int,
) -> ZenithTransformBatchContext:
    """Build shared settings for Rust transform+Zenith batch calls."""

    plaintext_ids = list(range(len(plaintext_symbols)))
    id_to_letter = {
        idx: str(symbol).upper()
        for idx, symbol in enumerate(plaintext_symbols)
    }
    return ZenithTransformBatchContext(
        plaintext_ids=plaintext_ids,
        id_to_letter=id_to_letter,
        model_path=str(model_path),
        epochs=int(budget_params["epochs"]),
        sampler_iterations=int(budget_params["sampler_iterations"]),
        seeds=[int(seed) for seed in budget_params["seeds"]],
        threads=int(threads),
    )


def run_zenith_transform_batch(
    request: ZenithTransformBatchRequest,
    *,
    batch_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute the Rust transform+Zenith batch request.

    Tests may inject ``batch_runner``; production uses
    ``analysis.zenith_fast.zenith_transform_candidates_batch_fast``.
    """

    if batch_runner is None:
        from analysis.zenith_fast import zenith_transform_candidates_batch_fast

        batch_runner = zenith_transform_candidates_batch_fast
    return batch_runner(**request.to_call_kwargs())


def transform_pipeline_key(pipeline_raw: Any) -> str:
    """Return a stable key for deduplicating transform pipelines."""

    return json.dumps(pipeline_raw, sort_keys=True)


def dedupe_transform_batch_candidates(
    raw_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Deduplicate candidates by transform pipeline for Rust batch probes.

    The Rust batch kernel should not spend solver budget on identical token
    orders. Metadata remains keyed by candidate id so batch results can be
    normalized back into artifact candidates.
    """

    seen_pipeline: set[str] = set()
    payload_candidates: list[dict[str, Any]] = []
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for candidate in raw_candidates:
        pipeline_key = transform_pipeline_key(candidate.get("pipeline"))
        if pipeline_key in seen_pipeline:
            continue
        seen_pipeline.add(pipeline_key)
        candidate_id = str(candidate.get("candidate_id"))
        payload_candidates.append(candidate)
        metadata_by_id[candidate_id] = candidate
    return payload_candidates, metadata_by_id


def confirmation_seed_offset(index: int) -> int:
    """Seed offset policy for independent transform confirmation probes."""

    return 10_000 + index * 1_000


def build_confirmation_batch_payload(
    items: list[dict[str, Any]],
    *,
    reason: str,
    start_index: int,
) -> tuple[list[dict[str, Any]], dict[str, ConfirmationBatchMetadata], list[dict[str, Any]]]:
    """Build Rust confirmation payload candidates and report missing pipelines."""

    payload_candidates: list[dict[str, Any]] = []
    metadata: dict[str, ConfirmationBatchMetadata] = {}
    missing_pipeline: list[dict[str, Any]] = []
    for offset, item in enumerate(items):
        seed_offset = confirmation_seed_offset(start_index + offset)
        original_id = str(item.get("candidate_id"))
        pipeline = item.get("pipeline")
        if not pipeline:
            missing_pipeline.append({
                "item": item,
                "original_id": original_id,
                "seed_offset": seed_offset,
                "reason": reason,
                "error": "missing transform pipeline",
            })
            continue
        batch_id = f"{original_id}__confirm_{seed_offset}"
        payload_candidates.append({
            "candidate_id": batch_id,
            "family": item.get("family"),
            "pipeline": pipeline,
            "grid": item.get("grid"),
            "seed_offset": seed_offset,
        })
        metadata[batch_id] = ConfirmationBatchMetadata(
            item=item,
            original_id=original_id,
            seed_offset=seed_offset,
            reason=reason,
        )
    return payload_candidates, metadata, missing_pipeline


def failed_batch_candidate_record(
    *,
    row: dict[str, Any],
    candidate: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    """Normalize a failed Rust transform batch row for artifact skipped lists."""

    return {
        "candidate_id": candidate.get("candidate_id", candidate_id),
        "family": candidate.get("family") or row.get("family"),
        "pipeline": candidate.get("pipeline"),
        "reason": row.get("reason") or "rust_batch_candidate_failed",
    }


def normalize_rust_rank_batch_results(
    *,
    batch: dict[str, Any],
    metadata_by_id: dict[str, dict[str, Any]],
    total_elapsed_seconds: float,
    quality_score_fn: Callable[[str], float],
    mutation_penalty_fn: Callable[[dict[str, Any]], float],
    selection_score_fn: Callable[..., float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize Rust rank/probe batch results using caller-owned scoring."""

    ranked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in batch.get("results", []):
        candidate_id = str(row.get("candidate_id"))
        candidate = metadata_by_id.get(candidate_id, {})
        if row.get("status") != "completed":
            skipped.append(failed_batch_candidate_record(
                row=row,
                candidate=candidate,
                candidate_id=candidate_id,
            ))
            continue
        decryption = str(row.get("decryption") or "")
        anneal_score = _float_or_none(row.get("normalized_score"))
        quality_score = quality_score_fn(decryption)
        mutation_penalty = mutation_penalty_fn(candidate)
        selection_score = selection_score_fn(
            anneal_score=anneal_score,
            quality_score=quality_score,
            structural_score=candidate.get("score"),
            mutation_penalty=mutation_penalty,
        )
        ranked.append(rust_rank_candidate_record(
            row=row,
            candidate=candidate,
            candidate_id=candidate_id,
            anneal_score=anneal_score,
            quality_score=quality_score,
            mutation_penalty=mutation_penalty,
            selection_score=selection_score,
            batch=batch,
            total_elapsed_seconds=total_elapsed_seconds,
        ))
    return ranked, skipped


def run_zenith_transform_rank_batch(
    *,
    tokens: list[int],
    raw_candidates: list[dict[str, Any]],
    context: ZenithTransformBatchContext,
    top_n: int,
    scoring_policy: ZenithTransformRankScoringPolicy,
    batch_runner: Callable[..., dict[str, Any]] | None = None,
    elapsed_seconds: Callable[[], float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run and normalize one Rust transform+homophonic rank/probe batch."""

    started = time.time()
    payload_candidates, metadata_by_id = dedupe_transform_batch_candidates(raw_candidates)
    request = build_zenith_transform_batch_request(
        tokens=tokens,
        candidates=payload_candidates,
        context=context,
        top_n=top_n,
    )
    batch = run_zenith_transform_batch(request, batch_runner=batch_runner)
    total_elapsed_seconds = (
        elapsed_seconds() if elapsed_seconds is not None else time.time() - started
    )
    return normalize_rust_rank_batch_results(
        batch=batch,
        metadata_by_id=metadata_by_id,
        total_elapsed_seconds=total_elapsed_seconds,
        quality_score_fn=scoring_policy.quality_score_fn,
        mutation_penalty_fn=scoring_policy.mutation_penalty_fn,
        selection_score_fn=scoring_policy.selection_score_fn,
    )


def run_zenith_transform_confirmation_batches(
    *,
    tokens: list[int],
    ranked: list[dict[str, Any]],
    finalists: list[dict[str, Any]],
    context: ZenithTransformBatchContext,
    scoring_policy: ZenithTransformRankScoringPolicy,
    confirmation_policy: ZenithTransformConfirmationPolicy,
    plaintext_distance_fn: Callable[[str, str], float],
    batch_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Confirm transform finalists with independent Rust batch probes.

    This mutates ``ranked``/``finalists`` records in-place, matching the
    artifact contract expected by downstream finalist sorting.
    """

    confirmed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    confirmed_ids: set[str] = set()

    def run_confirmation_batch(
        items: list[dict[str, Any]],
        *,
        reason: str,
        start_index: int,
    ) -> None:
        payload_candidates, metadata, missing_pipeline = build_confirmation_batch_payload(
            items,
            reason=reason,
            start_index=start_index,
        )
        for missing in missing_pipeline:
            item = missing["item"]
            original_id = str(missing["original_id"])
            seed_offset = int(missing["seed_offset"])
            confirmation, skipped_record, confirmed_score = missing_pipeline_confirmation_record(
                item=item,
                seed_offset=seed_offset,
                reason=reason,
                error=str(missing["error"]),
                fallback_score=_primary_candidate_score(item),
            )
            item["confirmation"] = confirmation
            item["confirmed_selection_score"] = confirmed_score
            confirmed_ids.add(original_id)
            skipped.append(skipped_record)

        if not payload_candidates:
            return
        request = build_zenith_transform_batch_request(
            tokens=tokens,
            candidates=payload_candidates,
            context=context,
            top_n=1,
        )
        batch = run_zenith_transform_batch(request, batch_runner=batch_runner)
        for row in batch.get("results", []):
            batch_id = str(row.get("candidate_id"))
            meta = metadata[batch_id]
            item = meta.item
            original_id = meta.original_id
            seed_offset = meta.seed_offset
            confirmation_reason = meta.reason
            if row.get("status") != "completed":
                confirmation, skipped_record, confirmed_score = failed_confirmation_record(
                    row=row,
                    item=item,
                    seed_offset=seed_offset,
                    reason=confirmation_reason,
                    fallback_score=_primary_candidate_score(item),
                )
                item["confirmation"] = confirmation
                item["confirmed_selection_score"] = confirmed_score
                confirmed_ids.add(original_id)
                skipped.append(skipped_record)
                continue

            decryption = str(row.get("decryption") or "")
            anneal_score = _float_or_none(row.get("normalized_score"))
            quality_score = scoring_policy.quality_score_fn(decryption)
            mutation_penalty = scoring_policy.mutation_penalty_fn(item)
            confirmation_selection = scoring_policy.selection_score_fn(
                anneal_score=anneal_score,
                quality_score=quality_score,
                structural_score=item.get("structural_score"),
                mutation_penalty=mutation_penalty,
            )
            primary_score = _primary_candidate_score(item)
            primary_text = str(item.get("decryption") or "")
            distance = plaintext_distance_fn(primary_text, decryption)
            confirmation, confirmed_score, confirmed_record = successful_confirmation_record(
                row=row,
                item=item,
                seed_offset=seed_offset,
                reason=confirmation_reason,
                budget=confirmation_policy.budget,
                anneal_score=anneal_score,
                quality_score=quality_score,
                selection_score=confirmation_selection,
                primary_score=primary_score,
                distance=distance,
            )
            item["confirmation"] = confirmation
            item["confirmed_selection_score"] = confirmed_score
            confirmed_ids.add(original_id)
            confirmed.append(confirmed_record)

    run_confirmation_batch(finalists, reason="initial_finalist", start_index=0)
    best_confirmed = max(
        (
            _float_or_none(item.get("confirmed_selection_score")) or float("-inf")
            for item in ranked
            if str(item.get("candidate_id")) in confirmed_ids
        ),
        default=float("-inf"),
    )

    adaptive_items: list[dict[str, Any]] = []
    max_adaptive_confirmations = max(0, confirmation_policy.adaptive_confirmations)
    if max_adaptive_confirmations > 0 and math.isfinite(best_confirmed):
        for item in ranked:
            if len(adaptive_items) >= max_adaptive_confirmations:
                break
            candidate_id = str(item.get("candidate_id"))
            if candidate_id in confirmed_ids:
                continue
            base_score = _primary_candidate_score(item)
            if base_score < best_confirmed - confirmation_policy.adaptive_margin:
                continue
            adaptive_items.append(item)
    run_confirmation_batch(
        adaptive_items,
        reason="adaptive_near_margin",
        start_index=len(confirmed_ids),
    )

    unconfirmed_count = 0
    for item in ranked:
        candidate_id = str(item.get("candidate_id"))
        if candidate_id in confirmed_ids:
            continue
        base_score = _primary_candidate_score(item)
        item["confirmation"] = {
            "status": "not_run",
            "reason": "outside_confirmation_budget",
            "unconfirmed_penalty": confirmation_policy.unconfirmed_penalty,
        }
        item["confirmed_selection_score"] = round(
            base_score - confirmation_policy.unconfirmed_penalty,
            6,
        )
        unconfirmed_count += 1

    return {
        "stage": "independent_seed_confirmation",
        "engine": "rust_batch",
        "confirmed_candidate_count": len(confirmed),
        "adaptive_confirmed_candidate_count": len(adaptive_items),
        "adaptive_margin": confirmation_policy.adaptive_margin,
        "unconfirmed_candidate_count": unconfirmed_count,
        "unconfirmed_penalty": confirmation_policy.unconfirmed_penalty,
        "skipped_candidates": skipped,
        "confirmed_candidates": confirmed,
        "policy": (
            "Stage C reruns the top transform finalists with independent seed "
            "offsets using the Rust transform+Zenith batch kernel, always "
            "includes the identity control when available, and rewards "
            "candidates whose scores and plaintexts are stable across probes."
        ),
    }


def rust_rank_candidate_record(
    *,
    row: dict[str, Any],
    candidate: dict[str, Any],
    candidate_id: str,
    anneal_score: float | None,
    quality_score: float,
    mutation_penalty: float,
    selection_score: float,
    batch: dict[str, Any],
    total_elapsed_seconds: float,
) -> dict[str, Any]:
    """Normalize one successful Rust rank/probe row into artifact shape."""

    decryption = str(row.get("decryption") or "")
    return {
        "candidate_id": candidate.get("candidate_id", candidate_id),
        "family": candidate.get("family") or row.get("family"),
        "provenance": candidate.get("provenance"),
        "params": candidate.get("params"),
        "pipeline": candidate.get("pipeline"),
        "status": "completed",
        "solver": "zenith_native",
        "engine": "rust_batch",
        "anneal_score": anneal_score,
        "plaintext_quality_score": round(quality_score, 6),
        "local_mutation_penalty": mutation_penalty,
        "selection_score": round(selection_score, 6),
        "elapsed_seconds": _rounded_float(row.get("elapsed_seconds"), 3),
        "decryption_preview": decryption[:500],
        "decryption": decryption,
        "key": {str(k): int(v) for k, v in dict(row.get("key") or {}).items()},
        "structural_score": candidate.get("score"),
        "structural_delta_vs_identity": candidate.get("delta_vs_identity"),
        "matrix_rank_score": (candidate.get("metrics") or {}).get("matrix_rank_score"),
        "best_period": (candidate.get("metrics") or {}).get("best_period"),
        "inverse_best_period": (candidate.get("metrics") or {}).get("inverse_best_period"),
        "rust_batch": {
            "best_seed": row.get("best_seed"),
            "attempts": row.get("attempts") or [],
            "token_order_hash": row.get("token_order_hash"),
            "candidate_elapsed_seconds": row.get("elapsed_seconds"),
            "batch_elapsed_seconds": batch.get("elapsed_seconds"),
            "threads": batch.get("threads"),
            "seed_count": batch.get("seed_count"),
            "total_elapsed_seconds": round(total_elapsed_seconds, 3),
        },
    }


def missing_pipeline_confirmation_record(
    *,
    item: dict[str, Any],
    seed_offset: int,
    reason: str,
    error: str,
    fallback_score: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Normalize confirmation failure for a candidate with no pipeline."""

    confirmation = {
        "status": "error",
        "seed_offset": seed_offset,
        "error": error,
    }
    skipped = {
        "candidate_id": item.get("candidate_id"),
        "family": item.get("family"),
        "seed_offset": seed_offset,
        "confirmation_reason": reason,
        "reason": error,
    }
    return confirmation, skipped, fallback_score - 0.12


def failed_confirmation_record(
    *,
    row: dict[str, Any],
    item: dict[str, Any],
    seed_offset: int,
    reason: str,
    fallback_score: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Normalize failed Rust confirmation row into item/skipped artifact shape."""

    error = row.get("reason") or "rust_batch_candidate_failed"
    confirmation = {
        "status": "error",
        "seed_offset": seed_offset,
        "error": error,
        "engine": "rust_batch",
    }
    skipped = {
        "candidate_id": item.get("candidate_id"),
        "family": item.get("family"),
        "seed_offset": seed_offset,
        "confirmation_reason": reason,
        "reason": error,
    }
    return confirmation, skipped, fallback_score - 0.12


def successful_confirmation_record(
    *,
    row: dict[str, Any],
    item: dict[str, Any],
    seed_offset: int,
    reason: str,
    budget: str,
    anneal_score: float | None,
    quality_score: float,
    selection_score: float,
    primary_score: float,
    distance: float,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Normalize a successful Rust confirmation probe.

    Returns ``(confirmation_block, confirmed_selection_score, summary_row)``.
    """

    stability_score = max(0.0, 1.0 - distance)
    confirmation_delta = (
        selection_score - primary_score
        if math.isfinite(primary_score) else None
    )
    penalty = 0.08 * (1.0 - stability_score)
    reasons: list[str] = []
    if confirmation_delta is not None and confirmation_delta < -0.08:
        penalty += 0.08
        reasons.append("confirmation_selection_dropped")
    if stability_score < 0.55:
        penalty += 0.05
        reasons.append("confirmation_plaintext_unstable")
    confirmed_score = (
        min(primary_score, selection_score)
        if math.isfinite(primary_score) else selection_score
    )
    confirmed_score -= penalty
    decryption = str(row.get("decryption") or "")
    confirmation = {
        "status": "completed",
        "solver": "zenith_native",
        "engine": "rust_batch",
        "seed_offset": seed_offset,
        "best_seed": row.get("best_seed"),
        "confirmation_reason": reason,
        "budget": budget,
        "anneal_score": anneal_score,
        "plaintext_quality_score": round(quality_score, 6),
        "selection_score": round(selection_score, 6),
        "selection_delta_vs_primary": (
            round(confirmation_delta, 6)
            if confirmation_delta is not None else None
        ),
        "plaintext_distance_ratio": round(distance, 6),
        "stability_score": round(stability_score, 6),
        "confirmation_penalty": round(penalty, 6),
        "confirmation_reasons": reasons,
        "elapsed_seconds": _rounded_float(row.get("elapsed_seconds"), 3),
        "decryption_preview": decryption[:500],
        "key": {str(k): int(v) for k, v in dict(row.get("key") or {}).items()},
    }
    summary = {
        "candidate_id": item.get("candidate_id"),
        "family": item.get("family"),
        "seed_offset": seed_offset,
        "confirmation_reason": reason,
        "selection_score": confirmation["selection_score"],
        "selection_delta_vs_primary": confirmation["selection_delta_vs_primary"],
        "stability_score": confirmation["stability_score"],
        "confirmed_selection_score": round(confirmed_score, 6),
        "reasons": reasons,
    }
    return confirmation, round(confirmed_score, 6), summary


def _rounded_float(value: Any, places: int) -> float:
    try:
        return round(float(value or 0.0), places)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _primary_candidate_score(item: dict[str, Any]) -> float:
    return (
        _float_or_none(item.get("validated_selection_score"))
        or _float_or_none(item.get("selection_score"))
        or float("-inf")
    )
