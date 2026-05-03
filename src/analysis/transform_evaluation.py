"""Shared finalist-menu evaluation for transform search.

Transform searches have different expensive inner loops: pure transposition can
score transformed A-Z text directly, while Z340-style candidates need a
homophonic solver probe. Once those loops produce a menu of plaintext
finalists, the cheap readability validation and score bookkeeping should be
shared so reports, agents, and automated selectors see the same evidence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from analysis.finalist_validation import validate_plaintext_finalist, validation_adjustment


@dataclass(frozen=True)
class FinalistMenuValidationPolicy:
    """How to attach plaintext validation evidence to a candidate menu."""

    language: str = "en"
    plaintext_fields: tuple[str, ...] = ("plaintext", "decryption", "preview")
    base_score_field: str = "selection_score"
    output_score_field: str = "validated_selection_score"
    adjustment_weight: float = 1.0
    score_precision: int = 5


def validate_finalist_menu(
    candidates: list[dict[str, Any]],
    *,
    policy: FinalistMenuValidationPolicy | None = None,
) -> dict[str, Any]:
    """Mutate candidates with validation blocks and return a compact summary.

    The score adjustment is intentionally explicit. Some callers, such as pure
    transposition, let plaintext validation strongly rerank an n-gram menu.
    Other callers, such as transform+homophonic validation, may use a very
    small weight so the evidence is visible without dominating anneal
    stability and identity-control gates.
    """

    cfg = policy or FinalistMenuValidationPolicy()
    label_counts: Counter[str] = Counter()
    recommendation_counts: Counter[str] = Counter()
    validated = 0
    skipped = 0
    for candidate in candidates:
        text = _candidate_plaintext(candidate, cfg.plaintext_fields)
        if not text:
            skipped += 1
            continue
        validation = validate_plaintext_finalist(text, language=cfg.language)
        raw_adjustment = validation_adjustment(validation)
        weighted_adjustment = round(raw_adjustment * cfg.adjustment_weight, cfg.score_precision)
        candidate["validation"] = validation
        candidate["validation_adjustment"] = raw_adjustment
        candidate["validation_adjustment_weight"] = cfg.adjustment_weight
        candidate["weighted_validation_adjustment"] = weighted_adjustment
        base_score = _float_or_none(candidate.get(cfg.base_score_field))
        if base_score is not None:
            candidate[cfg.output_score_field] = round(
                base_score + weighted_adjustment,
                cfg.score_precision,
            )
        label = str(validation.get("validation_label") or "unlabelled")
        recommendation = str(validation.get("recommendation") or "unknown")
        label_counts[label] += 1
        recommendation_counts[recommendation] += 1
        validated += 1
    return {
        "stage": "plaintext_finalist_validation",
        "language": cfg.language,
        "candidate_count": len(candidates),
        "validated_candidate_count": validated,
        "skipped_candidate_count": skipped,
        "base_score_field": cfg.base_score_field,
        "output_score_field": cfg.output_score_field,
        "adjustment_weight": cfg.adjustment_weight,
        "label_counts": dict(label_counts),
        "recommendation_counts": dict(recommendation_counts),
        "policy": (
            "Attach cheap plaintext validation evidence to transform finalists "
            "and optionally fold a weighted validation adjustment into the "
            "candidate score."
        ),
    }


def sort_finalist_menu(
    candidates: list[dict[str, Any]],
    *,
    primary_score_field: str = "validated_selection_score",
    secondary_score_fields: Iterable[str] = ("selection_score", "score"),
) -> None:
    """Sort candidate menus in-place by status and ranked score fields."""

    fields = tuple(secondary_score_fields)
    candidates.sort(
        key=lambda item: (
            item.get("status") in {None, "completed"},
            _score_tuple(item, primary_score_field, fields),
        ),
        reverse=True,
    )


def _candidate_plaintext(candidate: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = candidate.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _score_tuple(
    candidate: dict[str, Any],
    primary: str,
    secondary: tuple[str, ...],
) -> tuple[float, ...]:
    values = [_float_or_none(candidate.get(primary))]
    values.extend(_float_or_none(candidate.get(field)) for field in secondary)
    return tuple(value if value is not None else float("-inf") for value in values)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
