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
from typing import Any, Callable, Iterable

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


@dataclass(frozen=True)
class FinalistMenuEvaluationPlan:
    """Shared orchestration settings for a transform finalist menu."""

    stage: str = "transform_finalist_menu_evaluation"
    validation_policy: FinalistMenuValidationPolicy | None = None
    pre_confirmation_score_field: str = "validated_selection_score"
    pre_confirmation_secondary_fields: tuple[str, ...] = ("selection_score", "score")
    final_score_fields: tuple[str, ...] = (
        "confirmed_selection_score",
        "validated_selection_score",
        "selection_score",
        "score",
    )
    selection_policy: str = (
        "Validate finalists, optionally confirm expensive candidates, label "
        "or gate them, then select from a normalized finalist menu."
    )
    note: str = (
        "Probe engines may differ, but downstream finalist evidence is shaped "
        "through the same evaluation skeleton."
    )


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


def evaluate_finalist_menu(
    candidates: list[dict[str, Any]],
    *,
    plan: FinalistMenuEvaluationPlan | None = None,
    validate: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    confirm: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    label: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    choose: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    diagnose: Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]] | None = None,
    final_sort_key: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run the shared finalist-menu evaluation skeleton.

    Callers still own expensive probing: a pure-transposition screen may arrive
    with direct scores already attached, while a transform+homophonic ranker
    may pass a confirmation callback that reruns a small finalist set. This
    helper normalizes the surrounding bookkeeping and artifact shape.
    """

    cfg = plan or FinalistMenuEvaluationPlan()
    if validate is not None:
        validation_report = validate(candidates)
    else:
        validation_report = validate_finalist_menu(
            candidates,
            policy=cfg.validation_policy,
        )
    sort_finalist_menu(
        candidates,
        primary_score_field=cfg.pre_confirmation_score_field,
        secondary_score_fields=cfg.pre_confirmation_secondary_fields,
    )
    confirmation_report = (
        confirm(candidates)
        if confirm is not None
        else _no_confirmation_report()
    )
    finalist_report = (
        label(candidates)
        if label is not None
        else _default_finalist_labels(candidates)
    )
    if final_sort_key is not None:
        candidates.sort(key=final_sort_key, reverse=True)
    else:
        _sort_by_score_fields(candidates, cfg.final_score_fields)
    selection_report = (
        choose(candidates)
        if choose is not None
        else _choose_first_completed(candidates)
    )
    diagnostic_report = (
        diagnose(candidates, selection_report)
        if diagnose is not None
        else _default_diagnostics(candidates, selection_report)
    )
    return {
        "stage": cfg.stage,
        "evaluated_candidates": len(candidates),
        "selection_policy": cfg.selection_policy,
        "validation": validation_report,
        "confirmation": confirmation_report,
        "finalists": finalist_report,
        "selection": selection_report,
        "diagnostics": diagnostic_report,
        "top_ranked_candidates": candidates,
        "note": cfg.note,
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


def _sort_by_score_fields(
    candidates: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> None:
    candidates.sort(
        key=lambda item: (
            item.get("status") in {None, "completed"},
            _score_tuple(item, fields[0], fields[1:]) if fields else (float("-inf"),),
        ),
        reverse=True,
    )


def _no_confirmation_report() -> dict[str, Any]:
    return {
        "stage": "confirmation_not_required",
        "status": "not_run",
        "policy": (
            "No independent confirmation phase was requested for this finalist "
            "menu; selection uses the scores and validation evidence already "
            "attached to each candidate."
        ),
    }


def _default_finalist_labels(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    selectable = 0
    for candidate in candidates:
        if candidate.get("status") not in {None, "completed"}:
            label = "failed_candidate"
            is_selectable = False
        else:
            label = "direct_score_candidate"
            is_selectable = True
        candidate["finalist_label"] = label
        candidate["selectable_transform_candidate"] = is_selectable
        label_counts[label] += 1
        selectable += int(is_selectable)
    return {
        "stage": "default_direct_score_labels",
        "label_counts": dict(label_counts),
        "selectable_candidate_count": selectable,
        "policy": (
            "Direct-score menus are selectable when the candidate completed; "
            "callers may provide stricter family-specific gates."
        ),
    }


def _choose_first_completed(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.get("status") in {None, "completed"}:
            return {
                "selected": True,
                "selected_candidate_id": candidate.get("candidate_id"),
                "family": candidate.get("family"),
                "finalist_label": candidate.get("finalist_label"),
                "selection_score": _first_score(
                    candidate,
                    (
                        "confirmed_selection_score",
                        "validated_selection_score",
                        "selection_score",
                        "score",
                    ),
                ),
                "reason": "best_completed_candidate_in_normalized_menu",
            }
    return {
        "selected": False,
        "selected_candidate_id": None,
        "reason": "no_completed_candidate",
    }


def _default_diagnostics(
    candidates: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    label_counts = Counter(str(item.get("finalist_label") or "unlabelled") for item in candidates)
    return {
        "stage": "default_finalist_menu_diagnostics",
        "selected_candidate_id": selection.get("selected_candidate_id"),
        "label_counts": dict(label_counts),
        "completed_candidate_count": sum(
            1 for item in candidates if item.get("status") in {None, "completed"}
        ),
    }


def _first_score(candidate: dict[str, Any], fields: tuple[str, ...]) -> float | None:
    for field in fields:
        value = _float_or_none(candidate.get(field))
        if value is not None:
            return value
    return None


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
