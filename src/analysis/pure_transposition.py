"""Pure-transposition screening.

This module assumes a caller already has evidence for a transposition-only
cipher: ordinary A-Z alphabet, language-like symbol frequencies, but scrambled
order. It deliberately scores transformed text directly, unlike the Z340 path
that must run a downstream homophonic solver for each transform finalist.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any

from analysis.transform_evaluation import (
    FinalistMenuEvaluationPlan,
    FinalistMenuValidationPolicy,
    evaluate_finalist_menu,
)
from analysis.transform_fast import score_pure_transposition_candidates_fast_batch
from analysis.transform_search import TransformCandidate, iter_transform_candidates, plausible_grid_dimensions
from analysis.transformers import TransformPipeline, TransformStep
from models.cipher_text import CipherText

_CACHE_VERSION = 6
_SCREEN_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


@dataclass(frozen=True)
class PureTranspositionSearchConfig:
    """Candidate-generation settings for direct-scored transposition screens."""

    token_count: int
    profile: str = "wide"
    max_candidates: int | None = None
    include_matrix_rotate: bool = True
    include_transmatrix: bool = True
    include_rail_fence: bool = True
    include_route_composites: bool = True
    include_route_offsets: bool = True
    include_mask_routes: bool = True
    transmatrix_min_width: int = 2
    transmatrix_max_width: int | None = None
    provenance: str = "pure_transposition_screen"

    @property
    def profile_key(self) -> str:
        return (self.profile or "wide").strip().lower()

    def effective_transmatrix_max_width(self) -> int:
        return _effective_transmatrix_max_width(
            self.token_count,
            self.transmatrix_min_width,
            self.transmatrix_max_width,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "profile": self.profile_key,
            "max_candidates": self.max_candidates,
            "include_matrix_rotate": self.include_matrix_rotate,
            "include_transmatrix": self.include_transmatrix,
            "include_rail_fence": self.include_rail_fence,
            "include_route_composites": self.include_route_composites,
            "include_route_offsets": self.include_route_offsets,
            "include_mask_routes": self.include_mask_routes,
            "transmatrix_min_width": self.transmatrix_min_width,
            "transmatrix_max_width": self.effective_transmatrix_max_width(),
            "provenance": self.provenance,
        }


def screen_pure_transposition(
    cipher_text: CipherText,
    *,
    language: str = "en",
    profile: str = "wide",
    top_n: int = 25,
    max_candidates: int | None = None,
    include_matrix_rotate: bool = True,
    include_transmatrix: bool = True,
    include_rail_fence: bool = True,
    include_route_composites: bool = True,
    include_route_offsets: bool = True,
    include_mask_routes: bool = True,
    transmatrix_min_width: int = 2,
    transmatrix_max_width: int | None = None,
    threads: int = 0,
) -> dict[str, Any]:
    """Run a broad Rust-scored pure-transposition screen."""

    started = time.monotonic()
    values = az_values_from_cipher_symbols(cipher_text)
    cache_enabled = _pure_transposition_cache_enabled()
    cache_key = _screen_cache_key(
        values=values,
        language=language,
        profile=profile,
        top_n=top_n,
        max_candidates=max_candidates,
        include_matrix_rotate=include_matrix_rotate,
        include_transmatrix=include_transmatrix,
        include_rail_fence=include_rail_fence,
        include_route_composites=include_route_composites,
        include_route_offsets=include_route_offsets,
        include_mask_routes=include_mask_routes,
        transmatrix_min_width=transmatrix_min_width,
        transmatrix_max_width=transmatrix_max_width,
    )
    if cache_enabled and cache_key in _SCREEN_CACHE:
        cached = copy.deepcopy(_SCREEN_CACHE[cache_key])
        _SCREEN_CACHE.move_to_end(cache_key)
        original_elapsed = cached.get("elapsed_seconds")
        cached["elapsed_seconds"] = round(time.monotonic() - started, 6)
        cached["cache"] = {
            "enabled": True,
            "hit": True,
            "cache_key": cache_key[:16],
            "entry_count": len(_SCREEN_CACHE),
            "original_elapsed_seconds": original_elapsed,
        }
        return cached

    config = PureTranspositionSearchConfig(
        token_count=len(values),
        profile=profile,
        max_candidates=max_candidates,
        include_matrix_rotate=include_matrix_rotate,
        include_transmatrix=include_transmatrix,
        include_rail_fence=include_rail_fence,
        include_route_composites=include_route_composites,
        include_route_offsets=include_route_offsets,
        include_mask_routes=include_mask_routes,
        transmatrix_min_width=transmatrix_min_width,
        transmatrix_max_width=transmatrix_max_width,
    )
    candidates = generate_pure_transposition_candidates(config=config)
    candidate_dicts = [candidate.to_dict() for candidate in candidates]
    validation_pool_n = _validation_pool_size(top_n)
    scored = score_pure_transposition_candidates_fast_batch(
        values,
        candidate_dicts,
        language=language,
        top_n=validation_pool_n,
        threads=threads,
    )
    top: list[dict[str, Any]] = []
    for rust_rank, row in enumerate(scored.get("top_candidates") or [], start=1):
        index = int(row["candidate_index"])
        candidate = candidates[index].to_dict()
        plaintext = str(row.get("plaintext", ""))
        try:
            base_selection_score = float(row.get("selection_score"))
        except (TypeError, ValueError):
            base_selection_score = float("-inf")
        top.append({
            **candidate,
            "rank": row.get("rank"),
            "rust_rank": int(row.get("rank") or rust_rank),
            "score": row.get("score"),
            "selection_score": base_selection_score,
            "plaintext": plaintext,
            "preview": row.get("preview", ""),
        })
    evaluation_report = evaluate_finalist_menu(
        top,
        plan=FinalistMenuEvaluationPlan(
            stage="pure_transposition_finalist_menu_evaluation",
            validation_policy=FinalistMenuValidationPolicy(
                language=language,
                plaintext_fields=("plaintext", "preview"),
                base_score_field="selection_score",
                output_score_field="validated_selection_score",
                adjustment_weight=1.0,
                score_precision=5,
            ),
            pre_confirmation_score_field="validated_selection_score",
            pre_confirmation_secondary_fields=("selection_score",),
            final_score_fields=("validated_selection_score", "selection_score", "score"),
            selection_policy=(
                "Pure-transposition candidates are scored directly by the Rust "
                "language-model batch kernel, then reranked with the shared "
                "plaintext finalist validator. No homophonic confirmation "
                "phase is required."
            ),
            note=(
                "Direct-score pure-transposition finalist menu evaluated "
                "through the shared transform finalist skeleton."
            ),
        ),
    )
    top = list(evaluation_report.get("top_ranked_candidates") or top)
    for rank, row in enumerate(top, start=1):
        row["rank"] = rank
    top = top[:top_n]
    family_counts = Counter(_family_bucket(candidate.family) for candidate in candidates)
    top_family_counts = Counter(_family_bucket(str(row.get("family", ""))) for row in top)
    best = top[0] if top else None
    result = {
        "status": scored.get("status", "completed"),
        "solver": "pure_transposition_screen_rust",
        "engine": scored.get("solver"),
        "language": language,
        "profile": profile,
        "threads": scored.get("threads"),
        "candidate_count": scored.get("candidate_count", len(candidates)),
        "valid_candidate_count": scored.get("valid_candidate_count"),
        "validation_pool_size": validation_pool_n,
        "validation": evaluation_report.get("validation"),
        "evaluation": {
            key: value
            for key, value in evaluation_report.items()
            if key != "top_ranked_candidates"
        },
        "elapsed_seconds": scored.get("elapsed_seconds"),
        "scoring_elapsed_seconds": scored.get("elapsed_seconds"),
        "candidate_plan": config.to_metadata(),
        "include_matrix_rotate": include_matrix_rotate,
        "include_transmatrix": include_transmatrix,
        "include_rail_fence": include_rail_fence,
        "include_route_composites": include_route_composites,
        "include_route_offsets": include_route_offsets,
        "include_mask_routes": include_mask_routes,
        "transmatrix_min_width": transmatrix_min_width,
        "transmatrix_max_width": _effective_transmatrix_max_width(
            len(values),
            transmatrix_min_width,
            transmatrix_max_width,
        ),
        "family_counts": dict(family_counts),
        "top_family_counts": dict(top_family_counts),
        "best_candidate": best,
        "top_candidates": top,
        "cache": {
            "enabled": cache_enabled,
            "hit": False,
            "cache_key": cache_key[:16],
            "entry_count": len(_SCREEN_CACHE),
        },
        "note": (
            "Broad pure-transposition screen: transform candidate -> Rust "
            "pipeline application -> direct language-model score -> Python "
            "finalist validation/rerank. This is not the "
            "transposition+homophonic Z340 harness."
        ),
    }
    if cache_enabled:
        _SCREEN_CACHE[cache_key] = copy.deepcopy(result)
        _SCREEN_CACHE.move_to_end(cache_key)
        _trim_screen_cache()
        result["cache"]["entry_count"] = len(_SCREEN_CACHE)
    return result


def generate_pure_transposition_candidates(
    *,
    token_count: int | None = None,
    profile: str = "wide",
    max_candidates: int | None = None,
    include_matrix_rotate: bool = True,
    include_transmatrix: bool = True,
    include_rail_fence: bool = True,
    include_route_composites: bool = True,
    include_route_offsets: bool = True,
    include_mask_routes: bool = True,
    transmatrix_min_width: int = 2,
    transmatrix_max_width: int | None = None,
    config: PureTranspositionSearchConfig | None = None,
) -> list[TransformCandidate]:
    """Generate provenance-bearing candidates for pure-transposition scoring."""

    if config is None:
        if token_count is None:
            raise ValueError("token_count is required when config is not supplied")
        config = PureTranspositionSearchConfig(
            token_count=token_count,
            profile=profile,
            max_candidates=max_candidates,
            include_matrix_rotate=include_matrix_rotate,
            include_transmatrix=include_transmatrix,
            include_rail_fence=include_rail_fence,
            include_route_composites=include_route_composites,
            include_route_offsets=include_route_offsets,
            include_mask_routes=include_mask_routes,
            transmatrix_min_width=transmatrix_min_width,
            transmatrix_max_width=transmatrix_max_width,
        )
    return list(iter_pure_transposition_candidates(config))


def iter_pure_transposition_candidates(
    config: PureTranspositionSearchConfig,
):
    """Yield K3-compatible pure-transposition candidates with shared provenance."""

    out: list[TransformCandidate] = []
    seen: set[str] = set()

    def add(candidate: TransformCandidate) -> TransformCandidate | None:
        if config.max_candidates is not None and len(out) >= config.max_candidates:
            return None
        raw = repr(candidate.pipeline.to_raw())
        if raw in seen:
            return None
        seen.add(raw)
        out.append(candidate)
        return candidate

    for candidate in iter_transform_candidates(
        token_count=config.token_count,
        profile=config.profile_key,
        provenance=config.provenance,
        max_candidates=config.max_candidates,
    ):
        candidate = add(candidate)
        if candidate is not None:
            yield candidate

    if config.include_matrix_rotate and (
        config.max_candidates is None or len(out) < config.max_candidates
    ):
        ordinal = 0
        for width in _matrix_rotate_widths(config.token_count, config.profile_key):
            if config.max_candidates is not None and len(out) >= config.max_candidates:
                break
            for direction in ("cw", "ccw"):
                if config.max_candidates is not None and len(out) >= config.max_candidates:
                    break
                ordinal += 1
                candidate = add(_matrix_rotate_candidate(ordinal, width, direction, config.provenance))
                if candidate is not None:
                    yield candidate

    if config.include_rail_fence and (
        config.max_candidates is None or len(out) < config.max_candidates
    ):
        ordinal = 0
        for params in _rail_fence_params(config.token_count, config.profile_key):
            if config.max_candidates is not None and len(out) >= config.max_candidates:
                break
            ordinal += 1
            candidate = add(_rail_fence_candidate(ordinal, params, config.provenance))
            if candidate is not None:
                yield candidate

    if config.include_route_composites and (
        config.max_candidates is None or len(out) < config.max_candidates
    ):
        ordinal = 0
        for params in _route_composite_params(config.token_count, config.profile_key):
            if config.max_candidates is not None and len(out) >= config.max_candidates:
                break
            ordinal += 1
            candidate = add(_route_composite_candidate(ordinal, params, config.provenance))
            if candidate is not None:
                yield candidate

    if config.include_route_offsets and (
        config.max_candidates is None or len(out) < config.max_candidates
    ):
        ordinal = 0
        for params in _route_offset_params(config.token_count, config.profile_key):
            if config.max_candidates is not None and len(out) >= config.max_candidates:
                break
            ordinal += 1
            candidate = add(_route_offset_candidate(ordinal, params, config.provenance))
            if candidate is not None:
                yield candidate

    if config.include_mask_routes and (
        config.max_candidates is None or len(out) < config.max_candidates
    ):
        ordinal = 0
        for params in _mask_route_params(config.token_count, config.profile_key):
            if config.max_candidates is not None and len(out) >= config.max_candidates:
                break
            ordinal += 1
            candidate = add(_mask_route_candidate(ordinal, params, config.provenance))
            if candidate is not None:
                yield candidate

    if config.include_transmatrix and (
        config.max_candidates is None or len(out) < config.max_candidates
    ):
        hi = config.effective_transmatrix_max_width()
        lo = max(2, min(config.transmatrix_min_width, hi))
        ordinal = 0
        for w1 in range(lo, hi + 1):
            if config.max_candidates is not None and len(out) >= config.max_candidates:
                break
            for w2 in range(lo, hi + 1):
                if config.max_candidates is not None and len(out) >= config.max_candidates:
                    break
                for direction in ("cw", "ccw"):
                    if config.max_candidates is not None and len(out) >= config.max_candidates:
                        break
                    ordinal += 1
                    candidate = add(_transmatrix_candidate(ordinal, w1, w2, direction, config.provenance))
                    if candidate is not None:
                        yield candidate


def az_values_from_cipher_symbols(cipher_text: CipherText) -> list[int]:
    values: list[int] = []
    skipped: list[str] = []
    for token in cipher_text.tokens:
        symbol = cipher_text.alphabet.symbol_for(token).upper()
        if len(symbol) == 1 and "A" <= symbol <= "Z":
            values.append(ord(symbol) - ord("A"))
        else:
            skipped.append(symbol)
    if skipped:
        preview = ", ".join(skipped[:8])
        raise ValueError(
            "pure transposition screen currently requires A-Z ciphertext symbols; "
            f"unsupported symbols: {preview}"
        )
    return values


def pure_transposition_profile_from_env() -> str:
    return os.environ.get("DECIPHER_PURE_TRANSPOSITION_PROFILE", "wide").strip().lower() or "wide"


def pure_transposition_threads_from_env() -> int:
    raw = os.environ.get("DECIPHER_PURE_TRANSPOSITION_THREADS", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _validation_pool_size(top_n: int) -> int:
    """Oversample the Rust n-gram heap so Python validation can rerank menus."""

    return max(int(top_n), min(250, max(40, int(top_n) * 6)))


def pure_transposition_cache_info() -> dict[str, Any]:
    return {
        "enabled": _pure_transposition_cache_enabled(),
        "entry_count": len(_SCREEN_CACHE),
        "max_entries": _pure_transposition_cache_size(),
        "version": _CACHE_VERSION,
    }


def clear_pure_transposition_cache() -> None:
    _SCREEN_CACHE.clear()


def _matrix_rotate_candidate(
    ordinal: int,
    width: int,
    direction: str,
    provenance: str,
) -> TransformCandidate:
    pipeline = TransformPipeline(
        steps=(TransformStep("MatrixRotate", {"width": width, "direction": direction}),)
    )
    return TransformCandidate(
        candidate_id=f"mr_{ordinal:05d}_{width}_{direction}",
        family=f"matrix_rotate_{direction}",
        params={"width": width, "direction": direction},
        pipeline=pipeline,
        provenance=provenance,
    )


def _transmatrix_candidate(
    ordinal: int,
    w1: int,
    w2: int,
    direction: str,
    provenance: str,
) -> TransformCandidate:
    pipeline = TransformPipeline(
        steps=(TransformStep("TransMatrix", {"w1": w1, "w2": w2, "direction": direction}),)
    )
    return TransformCandidate(
        candidate_id=f"tm_{ordinal:06d}_{w1}_{w2}_{direction}",
        family="transmatrix",
        params={"w1": w1, "w2": w2, "direction": direction},
        pipeline=pipeline,
        provenance=provenance,
    )


def _rail_fence_candidate(
    ordinal: int,
    params: dict[str, Any],
    provenance: str,
) -> TransformCandidate:
    pipeline = TransformPipeline(
        steps=(TransformStep("RailFenceRoute", dict(params)),)
    )
    rails = params["rails"]
    offset = params.get("offset", 0)
    direction = params.get("direction", "down")
    rail_order = params.get("railOrder", "top_down")
    return TransformCandidate(
        candidate_id=f"rf_{ordinal:05d}_{rails}_{offset}_{direction}_{rail_order}",
        family="rail_fence",
        params=dict(params),
        pipeline=pipeline,
        provenance=provenance,
    )


def _rail_fence_params(token_count: int, profile: str) -> list[dict[str, Any]]:
    if token_count < 6:
        return []
    profile_key = (profile or "wide").strip().lower()
    max_rails = min(
        max(2, token_count // 3),
        20 if profile_key == "wide" else 12 if profile_key == "medium" else 6,
    )
    rail_orders = (
        ("top_down", "bottom_up", "even_odd", "odd_even")
        if profile_key == "wide"
        else ("top_down", "bottom_up")
        if profile_key == "medium"
        else ("top_down",)
    )
    directions = ("down", "up") if profile_key != "small" else ("down",)
    out: list[dict[str, Any]] = []
    for rails in range(2, max_rails + 1):
        period = max(1, 2 * (rails - 1))
        offsets = range(period) if profile_key == "wide" else range(min(period, 3))
        for direction in directions:
            for rail_order in rail_orders:
                for offset in offsets:
                    out.append({
                        "rails": rails,
                        "offset": offset,
                        "direction": direction,
                        "railOrder": rail_order,
                    })
    return out


def _route_composite_candidate(
    ordinal: int,
    params: dict[str, Any],
    provenance: str,
) -> TransformCandidate:
    route = str(params["route"])
    width = int(params["width"])
    repair = str(params["repair"])
    steps = [TransformStep("RouteRead", {"route": route})]
    if repair == "reverse":
        steps.append(TransformStep("Reverse", {}))
    elif repair in {"matrix_rotate_cw", "matrix_rotate_ccw"}:
        direction = "cw" if repair.endswith("_cw") else "ccw"
        steps.append(TransformStep("MatrixRotate", {"width": width, "direction": direction}))
    else:
        raise ValueError(f"unsupported route-composite repair: {repair}")
    pipeline = TransformPipeline(columns=width, steps=tuple(steps))
    return TransformCandidate(
        candidate_id=f"rc_{ordinal:05d}_{width}_{route}_{repair}",
        family=f"route_composite_{repair}",
        params=dict(params),
        pipeline=pipeline,
        provenance=provenance,
    )


def _route_composite_params(token_count: int, profile: str) -> list[dict[str, Any]]:
    if token_count <= 8:
        return []
    profile_key = (profile or "wide").strip().lower()
    max_columns = min(token_count - 1, 90 if profile_key == "wide" else 45 if profile_key == "medium" else 24)
    max_results = 28 if profile_key == "wide" else 16 if profile_key == "medium" else 8
    widths: list[int] = []

    def add_width(width: int) -> None:
        if 1 < width <= max_columns and width not in widths:
            widths.append(width)

    for grid in plausible_grid_dimensions(
        token_count,
        max_columns=max_columns,
        max_results=max_results,
    ):
        add_width(int(grid["columns"]))
        add_width(int(grid["rows"]))
    for width in (10, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24):
        add_width(width)
    if profile_key == "wide":
        for width in range(2, min(max_columns, 36) + 1):
            add_width(width)

    routes = (
        (
            "columns_down",
            "columns_up",
            "rows_boustrophedon",
            "columns_boustrophedon",
            "diagonal_down_right",
            "diagonal_up_left",
            "spiral_clockwise",
            "spiral_counterclockwise",
        )
        if profile_key == "wide"
        else (
            "columns_down",
            "columns_up",
            "rows_boustrophedon",
            "diagonal_down_right",
            "spiral_clockwise",
        )
        if profile_key == "medium"
        else ("columns_down", "rows_boustrophedon", "diagonal_down_right")
    )
    repairs = (
        ("matrix_rotate_cw", "matrix_rotate_ccw", "reverse")
        if profile_key != "small"
        else ("matrix_rotate_cw", "reverse")
    )
    out: list[dict[str, Any]] = []
    for width in widths:
        for route in routes:
            for repair in repairs:
                out.append({"width": width, "route": route, "repair": repair})
    return out


def _route_offset_candidate(
    ordinal: int,
    params: dict[str, Any],
    provenance: str,
) -> TransformCandidate:
    width = int(params["width"])
    route = str(params["route"])
    order_offset = int(params["orderOffset"])
    pipeline = TransformPipeline(
        columns=width,
        steps=(TransformStep("RouteRead", {"route": route, "orderOffset": order_offset}),),
    )
    return TransformCandidate(
        candidate_id=f"ro_{ordinal:05d}_{width}_{route}_{order_offset}",
        family=f"route_offset_{route}",
        params=dict(params),
        pipeline=pipeline,
        provenance=provenance,
    )


def _route_offset_params(token_count: int, profile: str) -> list[dict[str, Any]]:
    if token_count <= 8:
        return []
    profile_key = (profile or "wide").strip().lower()
    max_columns = min(token_count - 1, 80 if profile_key == "wide" else 40 if profile_key == "medium" else 24)
    max_results = 24 if profile_key == "wide" else 14 if profile_key == "medium" else 8
    widths: list[int] = []

    def add_width(width: int) -> None:
        if 1 < width <= max_columns and width not in widths:
            widths.append(width)

    for grid in plausible_grid_dimensions(
        token_count,
        max_columns=max_columns,
        max_results=max_results,
    ):
        add_width(int(grid["columns"]))
        add_width(int(grid["rows"]))
    for width in (10, 12, 13, 14, 15, 16, 17, 18, 20, 21):
        add_width(width)

    routes = (
        ("spiral_clockwise", "spiral_counterclockwise", "diagonal_down_right", "diagonal_up_left")
        if profile_key != "small"
        else ("spiral_clockwise", "diagonal_down_right")
    )
    offset_fractions = (4, 3, 2) if profile_key == "wide" else (4, 2)
    out: list[dict[str, Any]] = []
    for width in widths:
        rows = token_count // width
        usable = rows * width
        if rows <= 0 or usable <= 0:
            continue
        offsets = {1, max(1, width // 2), max(1, width - 1)}
        row_multipliers = (1, 2, 3, 5, 8) if profile_key == "wide" else (1, 2, 3)
        for multiplier in row_multipliers:
            offsets.add(width * multiplier)
        for denom in offset_fractions:
            offsets.add(max(1, usable // denom))
        if profile_key == "wide":
            offsets.update({max(1, usable // 5), max(1, usable // 6)})
        for route in routes:
            for order_offset in sorted(offset for offset in offsets if 0 < offset < usable):
                out.append({"width": width, "route": route, "orderOffset": order_offset})
    return out


def _mask_route_candidate(
    ordinal: int,
    params: dict[str, Any],
    provenance: str,
) -> TransformCandidate:
    width = int(params["width"])
    pattern = str(params["pattern"])
    first_route = str(params.get("firstRoute") or "rows")
    second_route = str(params.get("secondRoute") or "rows")
    mask_order = str(params.get("maskOrder") or "mask_first")
    pipeline = TransformPipeline(
        columns=width,
        steps=(TransformStep("MaskRoute", {
            "pattern": pattern,
            "firstRoute": first_route,
            "secondRoute": second_route,
            "maskOrder": mask_order,
        }),),
    )
    return TransformCandidate(
        candidate_id=f"mk_{ordinal:05d}_{width}_{pattern}_{first_route}_{second_route}_{mask_order}",
        family=f"mask_route_{pattern}",
        params=dict(params),
        pipeline=pipeline,
        provenance=provenance,
    )


def _mask_route_params(token_count: int, profile: str) -> list[dict[str, Any]]:
    if token_count <= 12:
        return []
    profile_key = (profile or "wide").strip().lower()
    max_columns = min(token_count - 1, 70 if profile_key == "wide" else 36 if profile_key == "medium" else 20)
    max_results = 18 if profile_key == "wide" else 10 if profile_key == "medium" else 6
    widths: list[int] = []

    def add_width(width: int) -> None:
        if 1 < width <= max_columns and width not in widths:
            widths.append(width)

    for grid in plausible_grid_dimensions(
        token_count,
        max_columns=max_columns,
        max_results=max_results,
    ):
        add_width(int(grid["columns"]))
        add_width(int(grid["rows"]))
    for width in (10, 12, 13, 14, 15, 16, 17, 18, 20, 21):
        add_width(width)

    patterns = (
        ("border", "cross", "quadrants_tl_br", "quadrants_tr_bl")
        if profile_key == "wide"
        else ("border", "cross", "quadrants_tl_br")
        if profile_key == "medium"
        else ("border",)
    )
    route_pairs = (
        (("rows", "rows"), ("rows_boustrophedon", "rows"), ("columns_down", "rows"))
        if profile_key != "small"
        else (("rows", "rows"),)
    )
    mask_orders = ("mask_first", "complement_first") if profile_key == "wide" else ("mask_first",)
    out: list[dict[str, Any]] = []
    for width in widths:
        rows = token_count // width
        usable = rows * width
        if rows <= 1 or usable <= 0:
            continue
        for pattern in patterns:
            for first_route, second_route in route_pairs:
                for mask_order in mask_orders:
                    out.append({
                        "width": width,
                        "pattern": pattern,
                        "firstRoute": first_route,
                        "secondRoute": second_route,
                        "maskOrder": mask_order,
                    })
    return out


def _effective_transmatrix_max_width(
    token_count: int,
    min_width: int,
    requested: int | None,
) -> int:
    default = min(120, max(2, token_count - 1))
    if requested is None:
        return max(min_width, default)
    return max(min_width, min(int(requested), max(2, token_count - 1)))


def _matrix_rotate_widths(token_count: int, profile: str) -> list[int]:
    if token_count <= 2:
        return []
    profile_key = (profile or "wide").strip().lower()
    max_width = min(token_count - 1, 120 if profile_key == "wide" else 60 if profile_key == "medium" else 30)
    max_results = 36 if profile_key == "wide" else 20 if profile_key == "medium" else 10
    widths: list[int] = []

    def add(width: int) -> None:
        if 1 < width <= max_width and width not in widths:
            widths.append(width)

    for grid in plausible_grid_dimensions(
        token_count,
        max_columns=max_width,
        max_results=max_results,
    ):
        add(int(grid["columns"]))
        add(int(grid["rows"]))
    for width in (17, 20, 16, 18, 19, 21, 24, 15, 25, 26, 28, 30, 12, 13, 14):
        add(width)
    if profile_key == "wide":
        for width in range(2, max_width + 1):
            add(width)
    return widths[:max_results if profile_key != "wide" else len(widths)]


def _screen_cache_key(
    *,
    values: list[int],
    language: str,
    profile: str,
    top_n: int,
    max_candidates: int | None,
    include_transmatrix: bool,
    include_matrix_rotate: bool,
    include_rail_fence: bool,
    include_route_composites: bool,
    include_route_offsets: bool,
    include_mask_routes: bool,
    transmatrix_min_width: int,
    transmatrix_max_width: int | None,
) -> str:
    payload = {
        "version": _CACHE_VERSION,
        "values_sha256": hashlib.sha256(bytes(values)).hexdigest(),
        "language": language,
        "profile": profile,
        "top_n": int(top_n),
        "max_candidates": max_candidates,
        "include_matrix_rotate": include_matrix_rotate,
        "include_transmatrix": include_transmatrix,
        "include_rail_fence": include_rail_fence,
        "include_route_composites": include_route_composites,
        "include_route_offsets": include_route_offsets,
        "include_mask_routes": include_mask_routes,
        "transmatrix_min_width": transmatrix_min_width,
        "transmatrix_max_width": transmatrix_max_width,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _pure_transposition_cache_enabled() -> bool:
    raw = os.environ.get("DECIPHER_PURE_TRANSPOSITION_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"} and _pure_transposition_cache_size() > 0


def _pure_transposition_cache_size() -> int:
    raw = os.environ.get("DECIPHER_PURE_TRANSPOSITION_CACHE_SIZE", "8").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 8


def _trim_screen_cache() -> None:
    max_entries = _pure_transposition_cache_size()
    while len(_SCREEN_CACHE) > max_entries:
        _SCREEN_CACHE.popitem(last=False)


def _family_bucket(family: str) -> str:
    if family.startswith("matrix_rotate"):
        return "matrix_rotate"
    if family.startswith("route_composite"):
        return "route_composite"
    if family.startswith("route_offset"):
        return "route_offset"
    if family.startswith("mask_route"):
        return "mask_route"
    if family.startswith("route_"):
        return "route"
    if family.startswith("columnar_transposition"):
        return "columnar_transposition"
    if family.startswith("unwrap_transposition"):
        return "unwrap_transposition"
    if family.startswith("split_"):
        return "split_grid"
    if family.startswith("grid_permute"):
        return "grid_permute"
    if family.startswith("rail_fence"):
        return "rail_fence"
    if family.startswith("wide_route"):
        return "wide_route_repair"
    if family.startswith("transmatrix"):
        return "transmatrix"
    return family
