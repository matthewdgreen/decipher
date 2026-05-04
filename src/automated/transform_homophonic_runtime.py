"""Runtime policy helpers for transform+homophonic automated probes."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from analysis.transform_homophonic_batch import (
    build_zenith_transform_batch_context,
    ZenithTransformBatchContext,
    ZenithTransformConfirmationPolicy,
    ZenithTransformRankScoringPolicy,
)


def build_transform_homophonic_batch_context(
    *,
    language: str,
    token_count: int,
    budget: str,
    model_path: Path | str | None,
    plaintext_symbols: list[str],
    search_profile: str,
    threads: int,
    budget_params_fn: Callable[..., dict[str, Any]],
    purpose: str,
) -> ZenithTransformBatchContext:
    """Build model/budget/thread context for Rust transform+Zenith probes."""

    if model_path is None:
        raise FileNotFoundError(
            f"{purpose} requires an ngram5 model for language={language!r}"
        )
    budget_params = budget_params_fn(
        budget,
        token_count < 600,
        search_profile=search_profile,
    )
    return build_zenith_transform_batch_context(
        plaintext_symbols=plaintext_symbols,
        model_path=str(model_path),
        budget_params=budget_params,
        threads=threads,
    )


def transform_homophonic_scoring_policy(language: str) -> ZenithTransformRankScoringPolicy:
    """Package scoring callbacks shared by rank and confirmation probes."""

    return ZenithTransformRankScoringPolicy(
        quality_score_fn=lambda text: plaintext_quality_score(text, language),
        mutation_penalty_fn=transform_mutation_penalty,
        selection_score_fn=transform_selection_score,
    )


def transform_homophonic_probe_policy(
    *,
    budget: str,
    adaptive_confirmations: int,
    rank_top_n: int = 3,
) -> dict[str, Any]:
    """Package rank/confirmation probe knobs for Rust transform+Zenith probes."""

    return {
        "rank_top_n": int(rank_top_n),
        "confirmation_policy": ZenithTransformConfirmationPolicy(
            budget=budget,
            adaptive_confirmations=adaptive_confirmations,
        ),
    }


def select_transform_confirmation_finalists(
    ranked: list[dict[str, Any]],
    *,
    confirm_count: int,
) -> list[dict[str, Any]]:
    """Select top completed transform finalists plus identity control if needed."""

    finalists = [
        item for item in ranked
        if item.get("status") == "completed" and item.get("pipeline")
    ][:confirm_count]
    identity = next(
        (
            item for item in ranked
            if item.get("candidate_id") == "000_identity"
            and item.get("status") == "completed"
            and item.get("pipeline")
        ),
        None,
    )
    if (
        identity is not None
        and all(item.get("candidate_id") != "000_identity" for item in finalists)
    ):
        finalists.append(identity)
    return finalists


def transform_mutation_penalty(candidate: dict[str, Any]) -> float:
    if candidate.get("provenance") == "local_mutation":
        return 0.08
    if candidate.get("provenance") == "program_search":
        params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
        if params.get("template") in {"banded_ndown_constructed", "route_repair_constructed"}:
            return 0.0
        return min(0.12, 0.02 * int(params.get("program_depth") or 1))
    return 0.0


def transform_selection_score(
    *,
    anneal_score: Any,
    quality_score: float,
    structural_score: Any,
    mutation_penalty: float,
) -> float:
    return (
        float(anneal_score or float("-inf"))
        + quality_score * 0.05
        + float(structural_score or 0.0) * 0.03
        - mutation_penalty
    )


def plaintext_quality_score(text: str, language: str) -> float:
    """Tiny no-boundary readability signal for transform finalist ranking."""

    cleaned = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if len(cleaned) < 20:
        return 0.0
    if language.lower().startswith("en"):
        fragments = (
            "THE", "AND", "ING", "ION", "ENT", "THAT", "WITH", "HER", "HIS",
            "FOR", "YOU", "NOT", "WAS", "HAVE", "THIS", "ARE",
        )
    else:
        fragments = ("THE", "AND", "ING", "ION", "ENT")
    fragment_hits = sum(cleaned.count(fragment) for fragment in fragments)
    fragment_rate = min(1.0, fragment_hits / max(1, len(cleaned) / 18))
    vowel_rate = sum(1 for ch in cleaned if ch in "AEIOU") / len(cleaned)
    vowel_score = max(0.0, 1.0 - abs(vowel_rate - 0.38) / 0.25)
    repeat_penalty = max(
        (len(match.group(0)) - 2) / len(cleaned)
        for match in re.finditer(r"([A-Z])\1{2,}", cleaned)
    ) if re.search(r"([A-Z])\1{2,}", cleaned) else 0.0
    return max(0.0, fragment_rate * 0.7 + vowel_score * 0.3 - repeat_penalty)
