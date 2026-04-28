"""Lightweight transform-search diagnostics and candidate generation.

This module deliberately does not run the expensive solver loop. It gives the
automated router and the agent a cheap way to ask whether transform search is
worth trying, and to produce a bounded candidate menu with provenance.
"""
from __future__ import annotations

import hashlib
import heapq
from array import array
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from analysis.transformers import (
    TransformPipeline,
    TransformStep,
    apply_transform_pipeline,
)


@dataclass(frozen=True)
class TransformCandidate:
    candidate_id: str
    family: str
    params: dict[str, Any]
    pipeline: TransformPipeline
    inverse_mode: bool | None = None
    grid: dict[str, int] | None = None
    provenance: str = "bounded_search"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "params": dict(self.params),
            "pipeline": self.pipeline.to_raw(),
            "inverse_mode": self.inverse_mode,
            "grid": dict(self.grid) if self.grid else None,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CandidateScore:
    candidate: TransformCandidate
    score: float
    delta_vs_identity: float
    metrics: dict[str, float]
    token_order_hash: str
    transformed_preview: list[int] = field(default_factory=list)
    position_order_preview: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "score": round(self.score, 6),
            "delta_vs_identity": round(self.delta_vs_identity, 6),
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "token_order_hash": self.token_order_hash,
            "transformed_preview": list(self.transformed_preview),
            "position_order_preview": list(self.position_order_preview),
        }


def plausible_grid_dimensions(
    token_count: int,
    *,
    columns: int | None = None,
    max_columns: int = 40,
    max_results: int = 12,
) -> list[dict[str, int]]:
    """Return plausible row/column choices for grid-based transforms."""

    if token_count <= 0:
        return []
    dims: list[dict[str, int]] = []
    if columns and columns > 1:
        rows = token_count // columns
        if rows > 1:
            dims.append({
                "columns": columns,
                "rows": rows,
                "remainder": token_count % columns,
                "source": "explicit",
            })
    common_widths = [17, 20, 16, 18, 19, 21, 24, 15, 25, 26, 28, 30]
    common_rank = {width: i for i, width in enumerate(common_widths)}
    seen = {(item["columns"], item["rows"]) for item in dims}
    for width in common_widths + list(range(2, max_columns + 1)):
        if width <= 1 or width > token_count:
            continue
        rows = token_count // width
        if rows <= 1:
            continue
        remainder = token_count % width
        if remainder and remainder > width // 4:
            continue
        key = (width, rows)
        if key in seen:
            continue
        dims.append({
            "columns": width,
            "rows": rows,
            "remainder": remainder,
            "source": "factor" if remainder == 0 else "near_factor",
        })
        seen.add(key)
    dims.sort(key=lambda d: (
        0 if d["source"] == "explicit" else 1,
        0 if d["columns"] in common_rank else 1,
        common_rank.get(d["columns"], 999),
        d["remainder"],
        abs(d["columns"] - 17),
        d["columns"],
    ))
    return dims[:max(1, max_results)]


def inspect_transform_suspicion(
    *,
    token_count: int,
    cipher_alphabet_size: int,
    plaintext_alphabet_size: int,
    word_group_count: int,
    cipher_system: str = "",
    columns: int | None = None,
    baseline_status: str | None = None,
    baseline_score: float | None = None,
) -> dict[str, Any]:
    """Cheap router signal for whether transform search is worth trying."""

    system = cipher_system.lower()
    grid_dims = plausible_grid_dimensions(token_count, columns=columns)
    reasons: list[str] = []
    cautions: list[str] = []
    score = 0

    if any(term in system for term in ("transposition", "z340", "zodiac340")):
        score += 4
        reasons.append("cipher_system mentions transposition")
    if any(term in system for term in ("homophonic", "zodiac", "copiale")):
        score += 2
        reasons.append("cipher_system mentions homophonic-style solving")
    if cipher_alphabet_size > plaintext_alphabet_size:
        score += 2
        reasons.append("cipher alphabet is larger than plaintext alphabet")
    if word_group_count <= 1 and token_count >= 80:
        score += 1
        reasons.append("cipher has little/no word-boundary structure")
    if grid_dims:
        exact = [d for d in grid_dims if d["remainder"] == 0]
        score += 2 if exact else 1
        reasons.append("cipher length has plausible grid dimensions")
    if baseline_status and baseline_status not in {"completed", "solved"}:
        score += 1
        reasons.append(f"baseline status is {baseline_status}")
    if baseline_score is not None and baseline_score < 0.65:
        score += 1
        reasons.append("baseline score is weak")
    if token_count < 60:
        score -= 1
        cautions.append("short ciphertext makes transform evidence noisy")
    if not grid_dims:
        cautions.append("no strong grid dimensions found")

    if score >= 6:
        recommendation = "run_screen"
    elif score >= 3:
        recommendation = "consider_screen"
    else:
        recommendation = "stay_baseline"

    return {
        "recommendation": recommendation,
        "activation_score": score,
        "reasons": reasons,
        "cautions": cautions,
        "plausible_grids": grid_dims,
        "policy": (
            "Run transform search when the ordinary solver is weak and cheap "
            "grid/order evidence exists; do not escalate every hard cipher."
        ),
    }


def generate_transform_candidates(
    *,
    token_count: int,
    columns: int | None = None,
    profile: str = "small",
    provenance: str = "bounded_search",
    max_candidates: int | None = None,
) -> list[TransformCandidate]:
    """Generate a bounded candidate menu with explicit provenance."""

    return list(iter_transform_candidates(
        token_count=token_count,
        columns=columns,
        profile=profile,
        provenance=provenance,
        max_candidates=max_candidates,
    ))


def iter_transform_candidates(
    *,
    token_count: int,
    columns: int | None = None,
    profile: str = "small",
    provenance: str = "bounded_search",
    max_candidates: int | None = None,
):
    """Yield a bounded candidate menu with explicit provenance."""

    profile_key = (profile or "small").strip().lower()
    if profile_key not in {"small", "medium", "wide"}:
        raise ValueError("profile must be one of: small, medium, wide")
    emitted_count = 0
    raw_count = 0
    seen_pipelines: set[str] = set()

    def make(
        family: str,
        steps: list[TransformStep],
        *,
        params: dict[str, Any] | None = None,
        grid: dict[str, int] | None = None,
        inverse_mode: bool | None = None,
    ) -> TransformCandidate | None:
        nonlocal emitted_count, raw_count
        if max_candidates is not None and emitted_count >= max_candidates:
            return None
        pipeline = TransformPipeline(
            steps=tuple(steps),
            columns=(grid or {}).get("columns") or columns,
            rows=(grid or {}).get("rows"),
        )
        cid = f"{raw_count:03d}_{family}"
        raw_count += 1
        raw = repr(pipeline.to_raw())
        if raw in seen_pipelines:
            return None
        seen_pipelines.add(raw)
        emitted_count += 1
        return TransformCandidate(
            candidate_id=cid,
            family=family,
            params=params or {},
            pipeline=pipeline,
            inverse_mode=inverse_mode,
            grid=grid,
            provenance=provenance,
        )

    candidate = make("identity", [], params={})
    if candidate is not None:
        yield candidate
    if token_count <= 1:
        return

    for candidate in (
        make("whole_reverse", [TransformStep("Reverse", {})]),
        make(
        "whole_shift_left",
        [TransformStep("ShiftCharactersLeft", {"rangeStart": 0, "rangeEnd": token_count - 1})],
        ),
        make(
        "whole_shift_right",
        [TransformStep("ShiftCharactersRight", {"rangeStart": 0, "rangeEnd": token_count - 1})],
        ),
    ):
        if candidate is not None:
            yield candidate

    grids = plausible_grid_dimensions(
        token_count,
        columns=columns,
        max_columns=120 if profile_key == "wide" else 40,
        max_results=24 if profile_key == "wide" else 12,
    )
    if profile_key == "small":
        grids = grids[:2]
    elif profile_key == "medium":
        grids = grids[:5]
    else:
        grids = grids[:24]

    for grid in grids:
        if max_candidates is not None and emitted_count >= max_candidates:
            break
        width = grid["columns"]
        rows = grid["rows"]
        usable = width * rows
        if usable <= 0:
            continue
        candidate = make(
            "row_reversals",
            [
                TransformStep(
                    "Reverse",
                    {"rangeStart": row * width, "rangeEnd": min((row + 1) * width, token_count) - 1},
                )
                for row in range(rows)
            ],
            params={"columns": width, "rows": rows},
            grid={"columns": width, "rows": rows},
        )
        if candidate is not None:
            yield candidate
        for route in ("columns_down", "columns_up", "rows_boustrophedon", "columns_boustrophedon"):
            candidate = make(
                f"route_{route}",
                [TransformStep("RouteRead", {"route": route})],
                params={"route": route, "columns": width, "rows": rows},
                grid={"columns": width, "rows": rows},
            )
            if candidate is not None:
                yield candidate
        candidate = make(
            "ndownmacross_1_1",
            [TransformStep("NDownMAcross", {"rangeStart": 0, "rangeEnd": rows - 1, "down": 1, "across": 1})],
            params={"down": 1, "across": 1},
            grid={"columns": width, "rows": rows},
        )
        if candidate is not None:
            yield candidate
        if profile_key != "small":
            for route in (
                "spiral_clockwise",
                "spiral_counterclockwise",
                "diagonal_down_right",
                "diagonal_down_left",
                "diagonal_up_right",
                "diagonal_up_left",
                "diagonal_zigzag_down_right",
                "diagonal_zigzag_down_left",
                "checkerboard_even_odd",
                "checkerboard_odd_even",
                "row_column_interleave",
                "column_row_interleave",
            ):
                candidate = make(
                    f"route_{route}",
                    [TransformStep("RouteRead", {"route": route})],
                    params={"route": route, "columns": width, "rows": rows},
                    grid={"columns": width, "rows": rows},
                )
                if candidate is not None:
                    yield candidate
            for route in ("rows_progressive_shift", "columns_progressive_shift"):
                shifts = (
                    _wide_progressive_shifts(width if route.startswith("rows") else rows)
                    if profile_key == "wide"
                    else _progressive_shifts(width if route.startswith("rows") else rows)
                )
                for shift in shifts:
                    candidate = make(
                        f"route_{route}_{shift}",
                        [TransformStep("RouteRead", {"route": route, "shift": shift})],
                        params={"route": route, "shift": shift, "columns": width, "rows": rows},
                        grid={"columns": width, "rows": rows},
                    )
                    if candidate is not None:
                        yield candidate
            offset_steps = (
                _wide_offset_chain_steps(width * rows)
                if profile_key == "wide"
                else _offset_chain_steps(width * rows)
            )
            for step in offset_steps:
                candidate = make(
                    f"route_offset_chain_{step}",
                    [TransformStep("RouteRead", {"route": "offset_chain", "step": step})],
                    params={"route": "offset_chain", "step": step, "columns": width, "rows": rows},
                    grid={"columns": width, "rows": rows},
                )
                if candidate is not None:
                    yield candidate
            split_candidates = (
                _wide_split_grid_candidates(width, rows)
                if profile_key == "wide"
                else _split_grid_candidates(width, rows)
            )
            for split_candidate in split_candidates:
                candidate = make(
                    split_candidate["family"],
                    [TransformStep("SplitGridRoute", split_candidate["data"])],
                    params=split_candidate["data"],
                    grid={"columns": width, "rows": rows},
                )
                if candidate is not None:
                    yield candidate
            grid_candidates = (
                _wide_grid_permute_candidates(width, rows)
                if profile_key == "wide"
                else _grid_permute_candidates(width, rows)
            )
            for grid_candidate in grid_candidates:
                candidate = make(
                    grid_candidate["family"],
                    [TransformStep("GridPermute", grid_candidate["data"])],
                    params=grid_candidate["data"],
                    grid={"columns": width, "rows": rows},
                )
                if candidate is not None:
                    yield candidate
            candidate = make(
                "ndownmacross_1_2",
                [TransformStep("NDownMAcross", {"rangeStart": 0, "rangeEnd": rows - 1, "down": 1, "across": 2})],
                params={"down": 1, "across": 2},
                grid={"columns": width, "rows": rows},
            )
            if candidate is not None:
                yield candidate
            key = "".join(chr(ord("a") + i) for i in range(min(width, 26)))
            if 1 < len(key) < token_count:
                key_probes = (
                    _wide_columnar_probe_keys(len(key))
                    if profile_key == "wide"
                    else _columnar_probe_keys(len(key))
                )
                for key_family, key in key_probes:
                    candidate = make(
                        f"columnar_transposition_{key_family}",
                        [TransformStep("Transposition", {"key": key})],
                        params={"key": key, "key_family": key_family},
                        grid={"columns": len(key), "rows": token_count // len(key)},
                        inverse_mode=False,
                    )
                    if candidate is not None:
                        yield candidate
                    candidate = make(
                        f"unwrap_transposition_{key_family}",
                        [TransformStep("UnwrapTransposition", {"key": key})],
                        params={"key": key, "key_family": key_family},
                        grid={"columns": len(key), "rows": token_count // len(key)},
                        inverse_mode=True,
                    )
                    if candidate is not None:
                        yield candidate
            for composite in _composite_route_candidates(width, rows, token_count):
                candidate = make(
                    composite["family"],
                    composite["steps"],
                    params=composite["params"],
                    grid={"columns": width, "rows": rows},
                )
                if candidate is not None:
                    yield candidate
            for composite in _banded_ndown_lock_shift_candidates(width, rows, token_count):
                candidate = make(
                    composite["family"],
                    composite["steps"],
                    params=composite["params"],
                    grid={"columns": width, "rows": rows},
                )
                if candidate is not None:
                    yield candidate
            if profile_key == "wide":
                for repair in _wide_local_range_candidates(width, rows, token_count):
                    candidate = make(
                        repair["family"],
                        repair["steps"],
                        params=repair["params"],
                        grid={"columns": width, "rows": rows},
                    )
                    if candidate is not None:
                        yield candidate
                route_repair_limit = _wide_route_repair_limit(
                    max_candidates,
                    grid_count=max(1, len(grids)),
                )
                for composite in _wide_route_repair_candidates(
                    width,
                    rows,
                    token_count,
                    limit=route_repair_limit,
                ):
                    candidate = make(
                        composite["family"],
                        composite["steps"],
                        params=composite["params"],
                        grid={"columns": width, "rows": rows},
                    )
                    if candidate is not None:
                        yield candidate
                double_repair_limit = _wide_route_double_repair_limit(
                    max_candidates,
                    grid_count=max(1, len(grids)),
                )
                if double_repair_limit:
                    for composite in _wide_route_double_repair_candidates(
                        width,
                        rows,
                        token_count,
                        limit=double_repair_limit,
                    ):
                        candidate = make(
                            composite["family"],
                            composite["steps"],
                            params=composite["params"],
                            grid={"columns": width, "rows": rows},
                        )
                        if candidate is not None:
                            yield candidate


def _progressive_shifts(size: int) -> list[int]:
    if size <= 2:
        return []
    probes = [1, 2, max(1, size // 3)]
    out: list[int] = []
    for shift in probes:
        shift = shift % size
        if shift and shift not in out:
            out.append(shift)
    return out[:3]


def _wide_progressive_shifts(size: int, *, limit: int = 64) -> list[int]:
    if size <= 2:
        return []
    out: list[int] = []
    for shift in range(1, size):
        if shift not in out:
            out.append(shift)
        if len(out) >= limit:
            break
    return out


def _composite_route_candidates(width: int, rows: int, token_count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    usable = min(width * rows, token_count)
    if usable <= 4:
        return candidates
    route_seeds = [
        "columns_down",
        "rows_boustrophedon",
        "diagonal_zigzag_down_right",
        "checkerboard_even_odd",
    ]
    repair_ranges = [
        (0, min(width - 1, usable - 1), "head_row_reverse"),
        (max(0, usable - width), usable - 1, "tail_row_reverse"),
        (0, min(3, usable - 1), "head_shift"),
    ]
    for route in route_seeds:
        for start, end, label in repair_ranges[:2]:
            if start >= end:
                continue
            candidates.append({
                "family": f"composite_{route}_{label}",
                "steps": [
                    TransformStep("RouteRead", {"route": route}),
                    TransformStep("Reverse", {"rangeStart": start, "rangeEnd": end}),
                ],
                "params": {
                    "route": route,
                    "repair": "Reverse",
                    "rangeStart": start,
                    "rangeEnd": end,
                    "label": label,
                },
            })
        start, end, label = repair_ranges[2]
        if start < end:
            candidates.append({
                "family": f"composite_{route}_{label}",
                "steps": [
                    TransformStep("RouteRead", {"route": route}),
                    TransformStep("ShiftCharactersRight", {"rangeStart": start, "rangeEnd": end}),
                ],
                "params": {
                    "route": route,
                    "repair": "ShiftCharactersRight",
                    "rangeStart": start,
                    "rangeEnd": end,
                    "label": label,
                },
            })
    return candidates[:12]


def _banded_ndown_lock_shift_candidates(width: int, rows: int, token_count: int) -> list[dict[str, Any]]:
    """Bounded banded NDown templates with local repair probes."""

    usable = min(width * rows, token_count)
    if width < 10 or rows < 10 or usable < 160:
        return []

    candidates: list[dict[str, Any]] = []
    split = max(1, rows // 2 - 1)
    second_end = max(split, rows - 3)
    across_values = [2]
    if width <= 14:
        across_values.append(1)

    for across in across_values:
        steps = [
            TransformStep("NDownMAcross", {"rangeStart": 0, "rangeEnd": split - 1, "down": 1, "across": across}),
            TransformStep(
                "ShiftCharactersRight",
                {
                    "rangeStart": min((rows * 7 // 10) * width + max(0, width // 5), usable - 1),
                    "rangeEnd": min((rows * 7 // 10) * width + width - 1, usable - 1),
                },
            ),
            TransformStep(
                "LockCharacters",
                {
                    "rangeStart": min(split * width + max(0, (width * 2) // 3), usable - 1),
                    "rangeEnd": min(split * width + width - 1, usable - 1),
                },
            ),
            TransformStep("NDownMAcross", {"rangeStart": split, "rangeEnd": second_end, "down": 1, "across": across}),
        ]
        candidates.append({
            "family": f"banded_ndown_lock_shift_across_{across}",
            "steps": steps,
            "params": {
                "template": "banded_ndown_lock_shift",
                "down": 1,
                "across": across,
                "split_row": split,
                "second_band_end_row": second_end,
            },
        })

    return candidates[:6]


def _columnar_probe_keys(width: int) -> list[tuple[str, str]]:
    if width < 2:
        return []
    orders: list[tuple[str, list[int]]] = [
        ("identity_key", list(range(width))),
        ("reverse_key", list(range(width - 1, -1, -1))),
        ("even_odd_key", list(range(0, width, 2)) + list(range(1, width, 2))),
        ("odd_even_key", list(range(1, width, 2)) + list(range(0, width, 2))),
    ]
    if width >= 4:
        orders.append(("outside_in_key", _outside_in_order(width)))
        orders.append(("inside_out_key", list(reversed(_outside_in_order(width)))))
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, order in orders:
        key = _key_for_column_order(order)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, key))
    return out


def _wide_columnar_probe_keys(width: int, *, limit: int = 1024) -> list[tuple[str, str]]:
    """Generate a wider but still bounded family of simple column orders."""

    if width < 2:
        return []
    candidates: list[tuple[str, list[int]]] = [
        (name, _column_order_from_key(key))
        for name, key in _columnar_probe_keys(width)
    ]
    base = list(range(width))
    for shift in range(1, width):
        candidates.append((f"rotate_left_{shift}_key", base[shift:] + base[:shift]))
        candidates.append((f"rotate_right_{shift}_key", base[-shift:] + base[:-shift]))
    for step in range(2, width):
        if _gcd(step, width) != 1:
            continue
        for offset in range(width):
            order = [((offset + step * i) % width) for i in range(width)]
            candidates.append((f"affine_s{step}_o{offset}_key", order))
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, order in candidates:
        key = _key_for_column_order(order)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, key))
        if len(out) >= limit:
            break
    return out


def _column_order_from_key(key: str) -> list[int]:
    ranked = sorted((ch, idx) for idx, ch in enumerate(key))
    return [idx for _ch, idx in ranked]


def _outside_in_order(width: int) -> list[int]:
    order: list[int] = []
    left = 0
    right = width - 1
    while left <= right:
        order.append(left)
        if left != right:
            order.append(right)
        left += 1
        right -= 1
    return order


def _key_for_column_order(order: list[int]) -> str:
    if len(order) > 26:
        order = order[:26]
    ranks = [0] * len(order)
    for rank, column in enumerate(order):
        ranks[column] = rank
    return "".join(chr(ord("a") + rank) for rank in ranks)


def _offset_chain_steps(size: int) -> list[int]:
    probes = [2, 3, 5, 7, 11, 13, 17, 19]
    out: list[int] = []
    for step in probes:
        if step < size and _gcd(step, size) == 1:
            out.append(step)
    return out[:4]


def _wide_offset_chain_steps(size: int, *, limit: int = 512) -> list[int]:
    out: list[int] = []
    for step in range(2, size):
        if _gcd(step, size) == 1:
            out.append(step)
        if len(out) >= limit:
            break
    return out


def _split_grid_candidates(width: int, rows: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    horizontal_splits = _interesting_splits(rows)
    vertical_splits = _interesting_splits(width)
    for split in horizontal_splits[:2]:
        candidates.append({
            "family": f"split_horizontal_{split}_top_reverse",
            "data": {
                "orientation": "horizontal",
                "split": split,
                "firstRoute": "rows_reverse",
                "secondRoute": "rows",
                "regionOrder": "normal",
            },
        })
        candidates.append({
            "family": f"split_horizontal_{split}_swap",
            "data": {
                "orientation": "horizontal",
                "split": split,
                "firstRoute": "rows",
                "secondRoute": "rows",
                "regionOrder": "swap",
            },
        })
    for split in vertical_splits[:2]:
        candidates.append({
            "family": f"split_vertical_{split}_left_reverse",
            "data": {
                "orientation": "vertical",
                "split": split,
                "firstRoute": "rows_reverse",
                "secondRoute": "rows",
                "regionOrder": "normal",
            },
        })
        candidates.append({
            "family": f"split_vertical_{split}_swap",
            "data": {
                "orientation": "vertical",
                "split": split,
                "firstRoute": "rows",
                "secondRoute": "rows",
                "regionOrder": "swap",
            },
        })
    return candidates[:8]


def _wide_split_grid_candidates(width: int, rows: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    horizontal_splits = _wide_interesting_splits(rows)
    vertical_splits = _wide_interesting_splits(width)
    region_orders = ("normal", "swap")
    route_pairs = (
        ("rows", "rows"),
        ("rows_reverse", "rows"),
        ("rows", "rows_reverse"),
        ("columns_down", "rows"),
        ("rows_boustrophedon", "rows"),
    )
    for split in horizontal_splits:
        for first_route, second_route in route_pairs:
            for region_order in region_orders:
                candidates.append({
                    "family": f"split_horizontal_{split}_{first_route}_{second_route}_{region_order}",
                    "data": {
                        "orientation": "horizontal",
                        "split": split,
                        "firstRoute": first_route,
                        "secondRoute": second_route,
                        "regionOrder": region_order,
                    },
                })
    for split in vertical_splits:
        for first_route, second_route in route_pairs:
            for region_order in region_orders:
                candidates.append({
                    "family": f"split_vertical_{split}_{first_route}_{second_route}_{region_order}",
                    "data": {
                        "orientation": "vertical",
                        "split": split,
                        "firstRoute": first_route,
                        "secondRoute": second_route,
                        "regionOrder": region_order,
                    },
                })
    return candidates[:1200]


def _grid_permute_candidates(width: int, rows: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    row_orders = ["reverse", "even_odd", "odd_even"]
    column_orders = ["reverse", "even_odd", "odd_even", "outside_in", "inside_out"]
    for order in row_orders:
        candidates.append({
            "family": f"grid_permute_rows_{order}",
            "data": {
                "rowOrder": order,
                "columnOrder": "identity",
            },
        })
    for order in column_orders:
        candidates.append({
            "family": f"grid_permute_columns_{order}",
            "data": {
                "rowOrder": "identity",
                "columnOrder": order,
            },
        })
    if width >= 4 and rows >= 4:
        candidates.append({
            "family": "grid_permute_rows_even_odd_columns_reverse",
            "data": {
                "rowOrder": "even_odd",
                "columnOrder": "reverse",
            },
        })
        candidates.append({
            "family": "grid_permute_rows_reverse_columns_even_odd",
            "data": {
                "rowOrder": "reverse",
                "columnOrder": "even_odd",
            },
        })
    return candidates[:10]


def _wide_grid_permute_candidates(width: int, rows: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    row_orders = [
        "identity",
        "reverse",
        "even_odd",
        "odd_even",
        "outside_in",
        "inside_out",
    ]
    column_orders = [
        "identity",
        "reverse",
        "even_odd",
        "odd_even",
        "outside_in",
        "inside_out",
    ]
    for row_order in row_orders:
        for column_order in column_orders:
            if row_order == "identity" and column_order == "identity":
                continue
            candidates.append({
                "family": f"grid_permute_rows_{row_order}_columns_{column_order}",
                "data": {
                    "rowOrder": row_order,
                    "columnOrder": column_order,
                },
            })
    return candidates


def _interesting_splits(size: int) -> list[int]:
    if size <= 2:
        return []
    raw = [size // 2, size // 3, (size * 2) // 3]
    out: list[int] = []
    for split in raw:
        if 0 < split < size and split not in out:
            out.append(split)
    return out


def _wide_interesting_splits(size: int, *, limit: int = 24) -> list[int]:
    if size <= 2:
        return []
    raw = [
        size // 2,
        size // 3,
        (size * 2) // 3,
        size // 4,
        (size * 3) // 4,
    ]
    raw.extend(range(2, size - 1))
    out: list[int] = []
    for split in raw:
        if 0 < split < size and split not in out:
            out.append(split)
        if len(out) >= limit:
            break
    return out


def _wide_local_range_candidates(width: int, rows: int, token_count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for start, end, label in _wide_local_ranges(width, rows, token_count):
        if start >= end:
            continue
        candidates.append({
            "family": f"range_reverse_{label}",
            "steps": [TransformStep("Reverse", {"rangeStart": start, "rangeEnd": end})],
            "params": {
                "operation": "Reverse",
                "rangeStart": start,
                "rangeEnd": end,
                "label": label,
            },
        })
        candidates.append({
            "family": f"range_shift_right_{label}",
            "steps": [TransformStep("ShiftCharactersRight", {"rangeStart": start, "rangeEnd": end})],
            "params": {
                "operation": "ShiftCharactersRight",
                "rangeStart": start,
                "rangeEnd": end,
                "label": label,
            },
        })
        candidates.append({
            "family": f"range_shift_left_{label}",
            "steps": [TransformStep("ShiftCharactersLeft", {"rangeStart": start, "rangeEnd": end})],
            "params": {
                "operation": "ShiftCharactersLeft",
                "rangeStart": start,
                "rangeEnd": end,
                "label": label,
            },
        })
    return candidates[:1200]


def _wide_route_repair_candidates(
    width: int,
    rows: int,
    token_count: int,
    *,
    limit: int = 5000,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    route_specs: list[tuple[str, dict[str, Any]]] = [
        ("columns_down", {"route": "columns_down"}),
        ("columns_up", {"route": "columns_up"}),
        ("rows_boustrophedon", {"route": "rows_boustrophedon"}),
        ("columns_boustrophedon", {"route": "columns_boustrophedon"}),
        ("diagonal_down_right", {"route": "diagonal_down_right"}),
        ("diagonal_up_left", {"route": "diagonal_up_left"}),
        ("diagonal_zigzag_down_right", {"route": "diagonal_zigzag_down_right"}),
        ("checkerboard_even_odd", {"route": "checkerboard_even_odd"}),
        ("row_column_interleave", {"route": "row_column_interleave"}),
        ("column_row_interleave", {"route": "column_row_interleave"}),
    ]
    for shift in _wide_progressive_shifts(width)[:8]:
        route_specs.append((f"rows_progressive_shift_{shift}", {"route": "rows_progressive_shift", "shift": shift}))
    for shift in _wide_progressive_shifts(rows)[:8]:
        route_specs.append((f"columns_progressive_shift_{shift}", {"route": "columns_progressive_shift", "shift": shift}))
    for step in _wide_offset_chain_steps(width * rows)[:16]:
        route_specs.append((f"offset_chain_{step}", {"route": "offset_chain", "step": step}))

    ranges = _wide_local_ranges(width, rows, token_count)[:96]
    for route_label, route_data in route_specs:
        for start, end, range_label in ranges:
            for operation in ("Reverse", "ShiftCharactersRight", "ShiftCharactersLeft"):
                repair_data = {"rangeStart": start, "rangeEnd": end}
                candidates.append({
                    "family": f"wide_route_repair_{route_label}_{operation}_{range_label}",
                    "steps": [
                        TransformStep("RouteRead", dict(route_data)),
                        TransformStep(operation, repair_data),
                    ],
                    "params": {
                        "route": dict(route_data),
                        "repair": operation,
                        "rangeStart": start,
                        "rangeEnd": end,
                        "label": range_label,
                    },
                })
                if len(candidates) >= limit:
                    return candidates
    return candidates


def _wide_route_repair_limit(
    max_generated_candidates: int | None,
    *,
    grid_count: int,
) -> int:
    """Allocate route+repair breadth so wide search can use its explicit cap."""

    if max_generated_candidates is None:
        return 5000
    if max_generated_candidates <= 0:
        return 0
    per_grid = max_generated_candidates // max(1, grid_count)
    return max(5000, min(20000, per_grid + 2500))


def _wide_route_double_repair_candidates(
    width: int,
    rows: int,
    token_count: int,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if limit <= 0:
        return candidates
    route_specs: list[tuple[str, dict[str, Any]]] = [
        ("columns_down", {"route": "columns_down"}),
        ("columns_up", {"route": "columns_up"}),
        ("rows_boustrophedon", {"route": "rows_boustrophedon"}),
        ("columns_boustrophedon", {"route": "columns_boustrophedon"}),
        ("diagonal_down_right", {"route": "diagonal_down_right"}),
        ("diagonal_up_left", {"route": "diagonal_up_left"}),
        ("row_column_interleave", {"route": "row_column_interleave"}),
        ("column_row_interleave", {"route": "column_row_interleave"}),
    ]
    for shift in _wide_progressive_shifts(width)[:4]:
        route_specs.append((f"rows_progressive_shift_{shift}", {"route": "rows_progressive_shift", "shift": shift}))
    for shift in _wide_progressive_shifts(rows)[:4]:
        route_specs.append((f"columns_progressive_shift_{shift}", {"route": "columns_progressive_shift", "shift": shift}))
    for step in _wide_offset_chain_steps(width * rows)[:8]:
        route_specs.append((f"offset_chain_{step}", {"route": "offset_chain", "step": step}))

    ranges = _wide_local_ranges(width, rows, token_count)[:96]
    operation_pairs = (
        ("Reverse", "Reverse"),
        ("Reverse", "ShiftCharactersRight"),
        ("Reverse", "ShiftCharactersLeft"),
        ("ShiftCharactersRight", "Reverse"),
        ("ShiftCharactersLeft", "Reverse"),
        ("ShiftCharactersRight", "ShiftCharactersLeft"),
        ("ShiftCharactersLeft", "ShiftCharactersRight"),
    )
    for route_label, route_data in route_specs:
        for first_index, (start_a, end_a, label_a) in enumerate(ranges):
            for start_b, end_b, label_b in ranges[first_index + 1:]:
                if not (end_a < start_b or end_b < start_a):
                    continue
                for operation_a, operation_b in operation_pairs:
                    repair_a = {"rangeStart": start_a, "rangeEnd": end_a}
                    repair_b = {"rangeStart": start_b, "rangeEnd": end_b}
                    candidates.append({
                        "family": (
                            "wide_route_double_repair_"
                            f"{route_label}_{operation_a}_{label_a}_{operation_b}_{label_b}"
                        ),
                        "steps": [
                            TransformStep("RouteRead", dict(route_data)),
                            TransformStep(operation_a, repair_a),
                            TransformStep(operation_b, repair_b),
                        ],
                        "params": {
                            "route": dict(route_data),
                            "repairs": [
                                {
                                    "operation": operation_a,
                                    "rangeStart": start_a,
                                    "rangeEnd": end_a,
                                    "label": label_a,
                                },
                                {
                                    "operation": operation_b,
                                    "rangeStart": start_b,
                                    "rangeEnd": end_b,
                                    "label": label_b,
                                },
                            ],
                        },
                    })
                    if len(candidates) >= limit:
                        return candidates
    return candidates


def _wide_route_double_repair_limit(
    max_generated_candidates: int | None,
    *,
    grid_count: int,
) -> int:
    """Allocate second-repair breadth only for larger wide sweeps."""

    if max_generated_candidates is None or max_generated_candidates <= 150000:
        return 0
    per_grid_extra = (max_generated_candidates - 150000) // max(1, grid_count)
    return max(1000, min(20000, per_grid_extra + 1000))


def _wide_local_ranges(width: int, rows: int, token_count: int) -> list[tuple[int, int, str]]:
    usable = min(width * rows, token_count)
    if usable <= 3:
        return []
    ranges: list[tuple[int, int, str]] = []
    for row in range(rows):
        start = row * width
        end = min(start + width - 1, usable - 1)
        if start < end:
            ranges.append((start, end, f"row{row}_full"))
            ranges.append((start, min(start + max(1, width // 3), end), f"row{row}_head"))
            ranges.append((max(start, end - max(1, width // 3)), end, f"row{row}_tail"))
    ranges.extend(_local_repair_ranges(token_count, width, rows))
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int, str]] = []
    for start, end, label in ranges:
        if start >= end:
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append((start, end, label))
    return out


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def screen_transform_candidates(
    tokens: list[int],
    *,
    columns: int | None = None,
    profile: str = "small",
    top_n: int = 10,
    max_generated_candidates: int | None = None,
    streaming: bool = False,
    include_mutations: bool = False,
    mutation_seed_count: int = 6,
    include_program_search: bool = False,
    program_max_depth: int = 5,
    program_beam_width: int = 20,
) -> dict[str, Any]:
    """Rank transform candidates using cheap token-order signals only."""

    if streaming and not include_mutations:
        return _screen_transform_candidates_streaming(
            tokens,
            columns=columns,
            profile=profile,
            top_n=top_n,
            max_generated_candidates=max_generated_candidates,
            include_program_search=include_program_search,
            program_max_depth=program_max_depth,
            program_beam_width=program_beam_width,
        )

    candidate_iter = iter_transform_candidates(
        token_count=len(tokens),
        columns=columns,
        profile=profile,
        max_candidates=max_generated_candidates,
    )
    candidates = list(candidate_iter)
    identity_metrics = _token_order_metrics(tokens, list(range(len(tokens))), None)
    identity_score = _combined_metric_score(identity_metrics)
    scored: list[CandidateScore] = []
    rejected: list[dict[str, Any]] = []
    seen_orders: set[str] = set()

    def score_candidates(batch: list[TransformCandidate], *, stage: str) -> None:
        for candidate in batch:
            validation = validate_transform_candidate(len(tokens), candidate)
            if not validation["valid"]:
                rejected.append({
                    **candidate.to_dict(),
                    "stage": stage,
                    "reason": validation["reason"],
                })
                continue
            order = validation["position_order"]
            order_hash = str(validation["token_order_hash"])
            preserve_duplicate = (
                stage == "program_search"
                and isinstance(candidate.params, dict)
                and bool(candidate.params.get("template"))
            )
            if order_hash in seen_orders and not preserve_duplicate:
                rejected.append({
                    **candidate.to_dict(),
                    "stage": stage,
                    "reason": "duplicate_token_order",
                    "token_order_hash": order_hash,
                })
                continue
            if not preserve_duplicate:
                seen_orders.add(order_hash)
            metrics = _token_order_metrics_for_order(tokens, order, candidate.grid)
            score = _combined_metric_score(metrics) + _program_shape_bonus(candidate)
            scored.append(
                CandidateScore(
                    candidate=candidate,
                    score=score,
                    delta_vs_identity=score - identity_score,
                    metrics=metrics,
                    token_order_hash=order_hash,
                    transformed_preview=_tokens_for_order(tokens, order, limit=60),
                    position_order_preview=order[:60],
                )
            )

    score_candidates(candidates, stage="base")
    base_scored_candidate_count = len(scored)
    program_candidate_count = 0
    program_scored_candidate_count = 0
    program_report: dict[str, Any] | None = None
    if include_program_search:
        before_program = len(scored)
        program_candidates, program_report = _program_search_candidates(
            tokens,
            columns=columns,
            profile=profile,
            max_depth=program_max_depth,
            beam_width=program_beam_width,
        )
        program_candidate_count = len(program_candidates)
        score_candidates(program_candidates, stage="program_search")
        program_scored_candidate_count = len(scored) - before_program
    mutation_candidate_count = 0
    mutation_scored_candidate_count = 0
    if include_mutations and scored:
        scored.sort(key=lambda item: (item.score, item.delta_vs_identity), reverse=True)
        seeds = [item.candidate for item in scored[:max(0, mutation_seed_count)]]
        before_mutation = len(scored)
        mutations = _mutate_candidate_neighborhood(
            seeds,
            token_count=len(tokens),
            profile=profile,
        )
        mutation_candidate_count = len(mutations)
        score_candidates(mutations, stage="local_mutation")
        mutation_scored_candidate_count = len(scored) - before_mutation
    scored.sort(key=lambda item: (item.score, item.delta_vs_identity), reverse=True)
    identity_candidate = next(
        (item.to_dict() for item in scored if item.candidate.family == "identity"),
        None,
    )
    anchor_candidates = _anchor_transform_candidates(scored, top_n=24)
    family_counts = Counter(_family_count_key(candidate.family) for candidate in candidates)
    scored_family_counts = Counter(_family_count_key(item.candidate.family) for item in scored)
    top_family_counts = Counter(_family_count_key(item.candidate.family) for item in scored[:top_n])
    rejected_reason_counts = Counter(item["reason"] for item in rejected)
    return {
        "profile": profile,
        "streaming": False,
        "max_generated_candidates": max_generated_candidates,
        "generation_limit_reached": (
            max_generated_candidates is not None
            and len(candidates) >= max(1, int(max_generated_candidates * 0.95))
        ),
        "candidate_count": len(candidates),
        "family_counts": dict(family_counts),
        "scored_family_counts": dict(scored_family_counts),
        "top_family_counts": dict(top_family_counts),
        "mutation_candidate_count": mutation_candidate_count,
        "mutation_scored_candidate_count": mutation_scored_candidate_count,
        "program_candidate_count": program_candidate_count,
        "program_scored_candidate_count": program_scored_candidate_count,
        "program_search": program_report,
        "base_scored_candidate_count": base_scored_candidate_count,
        "deduped_candidate_count": len(scored),
        "rejected_reason_counts": dict(rejected_reason_counts),
        "identity_score": round(identity_score, 6),
        "identity_candidate": identity_candidate,
        "anchor_candidates": anchor_candidates,
        "top_candidates": [item.to_dict() for item in scored[:top_n]],
        "rejected_candidates": rejected[:40],
        "note": (
            "This is a cheap structural screen only. Treat it as a menu for "
            "agent/automated follow-up, not as evidence of a solved cipher."
        ),
    }


def _screen_transform_candidates_streaming(
    tokens: list[int],
    *,
    columns: int | None,
    profile: str,
    top_n: int,
    max_generated_candidates: int | None,
    include_program_search: bool,
    program_max_depth: int,
    program_beam_width: int,
) -> dict[str, Any]:
    """Score candidates while retaining only compact counters and top-N heaps."""

    candidate_iter = iter_transform_candidates(
        token_count=len(tokens),
        columns=columns,
        profile=profile,
        max_candidates=max_generated_candidates,
    )
    identity_metrics = _token_order_metrics(tokens, list(range(len(tokens))), None)
    identity_score = _combined_metric_score(identity_metrics)
    rejected: list[dict[str, Any]] = []
    rejected_reason_counts: Counter[str] = Counter()
    seen_orders: set[str] = set()
    scored_family_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    top_heap: list[tuple[float, float, int, CandidateScore]] = []
    anchor_heap: list[tuple[float, float, int, CandidateScore]] = []
    identity_candidate: dict[str, Any] | None = None
    sequence = 0
    candidate_count = 0
    scored_count = 0
    use_fast_structural_metrics = (
        profile == "wide"
        and max_generated_candidates is not None
        and max_generated_candidates >= 100000
    )
    identity_positions = np.arange(len(tokens), dtype=np.int32) if use_fast_structural_metrics else None

    def push_heap(
        heap: list[tuple[float, float, int, CandidateScore]],
        item: CandidateScore,
        *,
        limit: int,
    ) -> None:
        entry = (item.score, item.delta_vs_identity, -sequence, item)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[:3] > heap[0][:3]:
            heapq.heapreplace(heap, entry)

    def score_one(candidate: TransformCandidate, *, stage: str) -> None:
        nonlocal identity_candidate, sequence, scored_count
        validation = validate_transform_candidate(len(tokens), candidate)
        if not validation["valid"]:
            rejected_reason_counts[str(validation["reason"])] += 1
            if len(rejected) < 40:
                rejected.append({
                    **candidate.to_dict(),
                    "stage": stage,
                    "reason": validation["reason"],
                })
            return
        order = validation["position_order"]
        order_hash = str(validation["token_order_hash"])
        preserve_duplicate = (
            stage == "program_search"
            and isinstance(candidate.params, dict)
            and bool(candidate.params.get("template"))
        )
        if order_hash in seen_orders and not preserve_duplicate:
            rejected_reason_counts["duplicate_token_order"] += 1
            if len(rejected) < 40:
                rejected.append({
                    **candidate.to_dict(),
                    "stage": stage,
                    "reason": "duplicate_token_order",
                    "token_order_hash": order_hash,
                })
            return
        if not preserve_duplicate:
            seen_orders.add(order_hash)
        if use_fast_structural_metrics:
            metrics = _fast_structural_metrics_for_order(order, candidate.grid, identity_positions)
        else:
            metrics = _token_order_metrics_for_order(tokens, order, candidate.grid)
        score = _combined_metric_score(metrics) + _program_shape_bonus(candidate)
        item = CandidateScore(
            candidate=candidate,
            score=score,
            delta_vs_identity=score - identity_score,
            metrics=metrics,
            token_order_hash=order_hash,
            transformed_preview=_tokens_for_order(tokens, order, limit=60),
            position_order_preview=order[:60],
        )
        scored_count += 1
        scored_family_counts[_family_count_key(candidate.family)] += 1
        if candidate.family == "identity":
            identity_candidate = item.to_dict()
        sequence += 1
        push_heap(top_heap, item, limit=max(1, top_n))
        if _anchor_candidate_key(item) is not None:
            push_heap(anchor_heap, item, limit=max(24, top_n))

    for candidate in candidate_iter:
        candidate_count += 1
        family_counts[_family_count_key(candidate.family)] += 1
        score_one(candidate, stage="base")

    base_scored_candidate_count = scored_count
    program_candidate_count = 0
    program_scored_candidate_count = 0
    program_report: dict[str, Any] | None = None
    if include_program_search:
        before_program = scored_count
        program_candidates, program_report = _program_search_candidates(
            tokens,
            columns=columns,
            profile=profile,
            max_depth=program_max_depth,
            beam_width=program_beam_width,
        )
        program_candidate_count = len(program_candidates)
        for candidate in program_candidates:
            score_one(candidate, stage="program_search")
        program_scored_candidate_count = scored_count - before_program

    top_items = [
        entry[3]
        for entry in sorted(top_heap, key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
    ]
    anchor_items: list[CandidateScore] = []
    anchor_seen_keys: set[tuple[str, int | None, int | None]] = set()
    for entry in sorted(anchor_heap, key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True):
        key = _anchor_candidate_key(entry[3])
        if key is None or key in anchor_seen_keys:
            continue
        anchor_seen_keys.add(key)
        anchor_items.append(entry[3])
        if len(anchor_items) >= 24:
            break
    top_family_counts = Counter(_family_count_key(item.candidate.family) for item in top_items)
    return {
        "profile": profile,
        "streaming": True,
        "fast_structural_metrics": use_fast_structural_metrics,
        "max_generated_candidates": max_generated_candidates,
        "generation_limit_reached": (
            max_generated_candidates is not None
            and candidate_count >= max(1, int(max_generated_candidates * 0.95))
        ),
        "candidate_count": candidate_count,
        "family_counts": dict(family_counts),
        "scored_family_counts": dict(scored_family_counts),
        "top_family_counts": dict(top_family_counts),
        "mutation_candidate_count": 0,
        "mutation_scored_candidate_count": 0,
        "program_candidate_count": program_candidate_count,
        "program_scored_candidate_count": program_scored_candidate_count,
        "program_search": program_report,
        "base_scored_candidate_count": base_scored_candidate_count,
        "deduped_candidate_count": scored_count,
        "rejected_reason_counts": dict(rejected_reason_counts),
        "identity_score": round(identity_score, 6),
        "identity_candidate": identity_candidate,
        "anchor_candidates": [item.to_dict() for item in anchor_items],
        "top_candidates": [item.to_dict() for item in top_items],
        "rejected_candidates": rejected,
        "note": (
            "This is a streaming structural screen only. It retains compact "
            "counters and top candidates instead of materializing every scored "
            "candidate; treat it as a promotion menu, not a solved cipher."
        ),
    }


def _anchor_candidate_key(item: CandidateScore) -> tuple[str, int | None, int | None] | None:
    anchor_prefixes = (
        "route_rows_boustrophedon",
        "route_columns_down",
        "route_columns_up",
        "ndownmacross",
        "banded_ndown_lock_shift",
        "program_",
        "row_reversals",
    )
    family = item.candidate.family
    if not family.startswith(anchor_prefixes):
        return None
    grid = item.candidate.grid or {}
    return (family, grid.get("columns"), grid.get("rows"))


def _family_count_key(family: str) -> str:
    """Compact high-cardinality generated family names for summary counters."""

    if family.startswith("wide_route_double_repair_"):
        return "wide_route_double_repair"
    if family.startswith("wide_route_repair_"):
        return "wide_route_repair"
    if family.startswith("route_offset_chain_"):
        return "route_offset_chain"
    if family.startswith("route_rows_progressive_shift_"):
        return "route_rows_progressive_shift"
    if family.startswith("route_columns_progressive_shift_"):
        return "route_columns_progressive_shift"
    if family.startswith("split_horizontal_"):
        return "split_horizontal"
    if family.startswith("split_vertical_"):
        return "split_vertical"
    if family.startswith("columnar_transposition_affine_"):
        return "columnar_transposition_affine"
    if family.startswith("unwrap_transposition_affine_"):
        return "unwrap_transposition_affine"
    if family.startswith("columnar_transposition_rotate_left_"):
        return "columnar_transposition_rotate_left"
    if family.startswith("columnar_transposition_rotate_right_"):
        return "columnar_transposition_rotate_right"
    if family.startswith("unwrap_transposition_rotate_left_"):
        return "unwrap_transposition_rotate_left"
    if family.startswith("unwrap_transposition_rotate_right_"):
        return "unwrap_transposition_rotate_right"
    if family.startswith("range_reverse_"):
        return "range_reverse"
    if family.startswith("range_shift_right_"):
        return "range_shift_right"
    if family.startswith("range_shift_left_"):
        return "range_shift_left"
    return family


def _anchor_transform_candidates(
    scored: list[CandidateScore],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Keep simple, important route families visible even in broad screens."""

    anchor_prefixes = (
        "route_rows_boustrophedon",
        "route_columns_down",
        "route_columns_up",
        "ndownmacross",
        "banded_ndown_lock_shift",
        "program_",
        "row_reversals",
    )
    anchors: list[CandidateScore] = []
    seen_keys: set[tuple[str, int | None, int | None]] = set()
    for item in sorted(scored, key=lambda entry: (entry.score, entry.delta_vs_identity), reverse=True):
        key = _anchor_candidate_key(item)
        if key is None:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        anchors.append(item)
        if len(anchors) >= top_n:
            break
    return [item.to_dict() for item in anchors]


def validate_transform_candidate(token_count: int, candidate: TransformCandidate) -> dict[str, Any]:
    """Check whether a candidate produces a true position permutation."""

    try:
        order = apply_transform_pipeline(list(range(token_count)), candidate.pipeline).tokens
    except Exception as exc:  # noqa: BLE001
        return {
            "valid": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "token_order_hash": None,
        }
    if len(order) != token_count:
        return {
            "valid": False,
            "reason": f"length_changed:{len(order)}",
            "token_order_hash": _token_hash(order),
        }
    permutation_check = _permutation_check(order, token_count)
    if not permutation_check["valid"]:
        return {
            "valid": False,
            "reason": "not_a_permutation",
            "missing_positions": permutation_check["missing"],
            "duplicate_positions": permutation_check["duplicates"],
            "token_order_hash": _token_hash(order),
        }
    return {
        "valid": True,
        "reason": "ok",
        "token_order_hash": _token_hash(order),
        "position_order": order,
    }


def _permutation_check(order: list[int], token_count: int) -> dict[str, Any]:
    seen = bytearray(token_count)
    duplicates: list[int] = []
    for value in order:
        if value < 0 or value >= token_count:
            duplicates.append(value)
            continue
        if seen[value]:
            duplicates.append(value)
            continue
        seen[value] = 1
    if not duplicates and all(seen):
        return {"valid": True, "missing": [], "duplicates": []}
    missing = [idx for idx, flag in enumerate(seen) if not flag][:10]
    return {
        "valid": False,
        "missing": missing,
        "duplicates": duplicates[:10],
    }


def _mutate_candidate_neighborhood(
    seeds: list[TransformCandidate],
    *,
    token_count: int,
    profile: str,
) -> list[TransformCandidate]:
    """Generate generic local repairs around promising transform candidates.

    These are intentionally not a known Z340 recipe. They encode the broader
    AZdecrypt-style idea that a promising grid/order candidate may need small
    local range repairs or row-band rewrites before the language solver sees a
    clean stream.
    """

    out: list[TransformCandidate] = []
    max_per_seed = 8 if profile == "small" else 16
    for seed in seeds:
        if seed.provenance == "program_search":
            continue
        mutations: list[TransformCandidate] = []
        columns = (seed.grid or {}).get("columns") or seed.pipeline.columns
        rows = (seed.grid or {}).get("rows") or seed.pipeline.rows
        local_ranges = _local_repair_ranges(token_count, columns, rows)
        for idx, (start, end, label) in enumerate(local_ranges):
            if len(mutations) >= max_per_seed:
                break
            mutations.append(_append_mutation(
                seed,
                family_suffix=f"reverse_{label}",
                step=TransformStep("Reverse", {"rangeStart": start, "rangeEnd": end}),
                params={"mutation": "reverse", "rangeStart": start, "rangeEnd": end, "label": label},
                ordinal=idx,
            ))
            if len(mutations) >= max_per_seed:
                break
            direction = "ShiftCharactersRight" if idx % 2 == 0 else "ShiftCharactersLeft"
            mutations.append(_append_mutation(
                seed,
                family_suffix=f"shift_{label}",
                step=TransformStep(direction, {"rangeStart": start, "rangeEnd": end}),
                params={
                    "mutation": direction,
                    "rangeStart": start,
                    "rangeEnd": end,
                    "label": label,
                },
                ordinal=idx,
            ))
        if columns and rows and rows > 3:
            for idx, (start_row, end_row, label) in enumerate(_row_band_ranges(rows)):
                if len(mutations) >= max_per_seed:
                    break
                for across in (1, 2):
                    if len(mutations) >= max_per_seed:
                        break
                    mutations.append(_append_mutation(
                        seed,
                        family_suffix=f"ndown_{label}_{across}",
                        step=TransformStep(
                            "NDownMAcross",
                            {
                                "rangeStart": start_row,
                                "rangeEnd": end_row,
                                "down": 1,
                                "across": across,
                            },
                        ),
                        params={
                            "mutation": "NDownMAcross",
                            "rangeStart": start_row,
                            "rangeEnd": end_row,
                            "down": 1,
                            "across": across,
                            "label": label,
                        },
                        ordinal=idx,
                    ))
        out.extend(mutations[:max_per_seed])
    return _dedupe_candidates(out)


def _program_search_candidates(
    tokens: list[int],
    *,
    columns: int | None,
    profile: str,
    max_depth: int,
    beam_width: int,
) -> tuple[list[TransformCandidate], dict[str, Any]]:
    """Construct transform pipelines from a small grammar with beam pruning."""

    token_count = len(tokens)
    grids = plausible_grid_dimensions(token_count, columns=columns)
    if profile == "small":
        grids = grids[:1]
    elif profile == "medium":
        grids = grids[:3]
    else:
        grids = grids[:5]
    all_candidates: list[TransformCandidate] = []
    layer_counts: Counter[int] = Counter()
    selected_layer_counts: Counter[int] = Counter()
    grid_reports: list[dict[str, Any]] = []
    survivor_summaries: list[dict[str, Any]] = []
    candidate_index = 0

    for grid in grids:
        width = grid["columns"]
        rows = grid["rows"]
        operations = _program_operations(width, rows, token_count)
        if not operations:
            continue
        beam = [
            TransformCandidate(
                candidate_id=f"program_seed_{width}x{rows}",
                family="program_seed",
                params={"program_depth": 0, "operation_labels": [], "grammar": "grid_program"},
                pipeline=TransformPipeline(columns=width, rows=rows),
                grid={"columns": width, "rows": rows},
                provenance="program_search",
            )
        ]
        seen_states: set[str] = set()
        grid_report = {
            "columns": width,
            "rows": rows,
            "operation_count": len(operations),
            "layers": [],
        }
        for depth in range(1, max(1, max_depth) + 1):
            layer: list[tuple[float, TransformCandidate]] = []
            for seed in beam:
                labels = list(seed.params.get("operation_labels") or [])
                last_order = int(seed.params.get("last_operation_order", 0))
                used = set(labels)
                for operation in operations:
                    label = str(operation["label"])
                    order = int(operation["order"])
                    if label in used or order < last_order:
                        continue
                    if not _program_operation_allowed(label, labels):
                        continue
                    candidate = _append_program_operation(
                        seed,
                        operation=operation,
                        candidate_index=candidate_index,
                        depth=depth,
                    )
                    candidate_index += 1
                    validation = validate_transform_candidate(token_count, candidate)
                    if not validation["valid"]:
                        continue
                    state_key = repr(candidate.pipeline.to_raw())
                    if state_key in seen_states:
                        continue
                    seen_states.add(state_key)
                    metrics = _token_order_metrics_for_order(
                        tokens,
                        validation["position_order"],
                        candidate.grid,
                    )
                    score = _combined_metric_score(metrics) + _program_shape_bonus(candidate)
                    layer.append((score, candidate))
            layer.sort(key=lambda item: item[0], reverse=True)
            selected = _select_program_beam(layer, max(1, beam_width))
            beam = [candidate for _score, candidate in selected]
            layer_counts[depth] += len(layer)
            selected_layer_counts[depth] += len(selected)
            grid_report["layers"].append({
                "depth": depth,
                "candidate_count": len(layer),
                "selected_count": len(selected),
                "top_score": round(selected[0][0], 6) if selected else None,
                "top_family": selected[0][1].family if selected else None,
                "top_operation_labels": selected[0][1].params.get("operation_labels") if selected else None,
                "top_by_class": _program_layer_top_by_class(layer),
            })
            survivor_summaries.extend(
                _program_candidate_summary(candidate, score)
                for score, candidate in selected[: min(5, len(selected))]
            )
            all_candidates.extend(candidate for _score, candidate in selected)
            if not beam:
                break
        expansions = _program_template_expansions(
            operations,
            width=width,
            rows=rows,
            start_index=candidate_index,
        )
        candidate_index += len(expansions)
        all_candidates.extend(expansions)
        grid_report["template_expansion_count"] = len(expansions)
        grid_reports.append(grid_report)

    report = {
        "enabled": True,
        "grammar": "grid_program_v1",
        "max_depth": max_depth,
        "beam_width": beam_width,
        "grid_count": len(grid_reports),
        "layer_candidate_counts": {str(k): v for k, v in sorted(layer_counts.items())},
        "selected_layer_counts": {str(k): v for k, v in sorted(selected_layer_counts.items())},
        "grids": grid_reports,
        "top_programs": sorted(
            survivor_summaries,
            key=lambda item: item["score"],
            reverse=True,
        )[:20],
        "policy": (
            "Program search composes small legal transform operations and "
            "structurally prunes each depth before any homophonic solver probe."
        ),
    }
    return _dedupe_candidates(all_candidates), report


def _program_template_expansions(
    operations: list[dict[str, Any]],
    *,
    width: int,
    rows: int,
    start_index: int,
) -> list[TransformCandidate]:
    """Materialize a few grammar-derived composite programs as beam safeguards."""

    by_label = {str(operation["label"]): operation for operation in operations}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for operation in operations:
        grouped.setdefault(str(operation["params"].get("group") or "other"), []).append(operation)
    out: list[TransformCandidate] = []
    index = start_index

    def emit(sequence: list[dict[str, Any]]) -> None:
        nonlocal index
        seed = TransformCandidate(
            candidate_id=f"program_template_seed_{index}",
            family="program_seed",
            params={"program_depth": 0, "operation_labels": [], "grammar": "grid_program_v1"},
            pipeline=TransformPipeline(columns=width, rows=rows),
            grid={"columns": width, "rows": rows},
            provenance="program_search",
        )
        candidate = seed
        for depth, operation in enumerate(sequence, start=1):
            candidate = _append_program_operation(
                candidate,
                operation=operation,
                candidate_index=index,
                depth=depth,
            )
            index += 1
        out.append(candidate)

    canonical_labels = [
        "ndown_top_a2",
        "late_shift_right",
        "mid_late_lock",
        "ndown_lower_a2",
        "tail_repair_pack",
    ]
    if all(label in by_label for label in canonical_labels):
        emit([by_label[label] for label in canonical_labels])

    for top in grouped.get("ndown_top", [])[:6]:
        top_params = top["params"]
        split = top_params.get("split")
        across = top_params.get("across")
        lowers = [
            item for item in grouped.get("ndown_lower", [])
            if item["params"].get("split") == split
            and item["params"].get("across") == across
        ]
        if not lowers:
            continue
        for shift in grouped.get("shift", [])[:2]:
            locks = [
                item for item in grouped.get("lock", [])
                if item["params"].get("split") == split
            ][:1]
            for lock in locks:
                for tail in grouped.get("tail_repair", [])[:1]:
                    sequence = [top, shift, lock, lowers[0], tail]
                    labels = [str(item["label"]) for item in sequence]
                    if labels == canonical_labels:
                        continue
                    emit(sequence)
                    if len(out) >= 8:
                        return out
    return out


def _select_program_beam(
    layer: list[tuple[float, TransformCandidate]],
    beam_width: int,
) -> list[tuple[float, TransformCandidate]]:
    """Keep route and banded/grid programs alive in the same beam."""

    buckets: dict[str, list[tuple[float, TransformCandidate]]] = {}
    for item in layer:
        buckets.setdefault(_program_beam_class(item[1]), []).append(item)
    for items in buckets.values():
        items.sort(key=lambda entry: entry[0], reverse=True)
    selected: list[tuple[float, TransformCandidate]] = []
    seen: set[str] = set()

    def add(item: tuple[float, TransformCandidate]) -> None:
        key = item[1].candidate_id
        if key not in seen and len(selected) < beam_width:
            selected.append(item)
            seen.add(key)

    per_class = max(4, beam_width // 2)
    for class_name in ("banded_program", "route_program", "other_program"):
        for item in buckets.get(class_name, [])[:per_class]:
            add(item)
    for item in layer:
        add(item)
        if len(selected) >= beam_width:
            break
    selected.sort(key=lambda entry: entry[0], reverse=True)
    return selected


def _program_layer_top_by_class(
    layer: list[tuple[float, TransformCandidate]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for score, candidate in layer:
        class_name = _program_beam_class(candidate)
        if class_name in out:
            continue
        out[class_name] = _program_candidate_summary(candidate, score)
    return out


def _program_beam_class(candidate: TransformCandidate) -> str:
    labels = list(candidate.params.get("operation_labels") or [])
    groups = [_program_label_group(label) for label in labels]
    if "ndown_top" in groups:
        return "banded_program"
    if "route" in groups:
        return "route_program"
    return "other_program"


def _program_operations(width: int, rows: int, token_count: int) -> list[dict[str, Any]]:
    usable = min(width * rows, token_count)
    if width < 4 or rows < 4 or usable < 40:
        return []

    default_split = max(1, rows // 2 - 1)
    split_rows = _program_split_rows(rows)

    operations: list[dict[str, Any]] = []

    def add(label: str, order: int, steps: list[TransformStep], params: dict[str, Any], *, group: str) -> None:
        operations.append({
            "label": label,
            "order": order,
            "steps": steps,
            "params": {**params, "group": group},
        })

    for route in ("columns_down", "columns_up", "rows_boustrophedon"):
        add(
            f"route_{route}",
            8,
            [TransformStep("RouteRead", {"route": route})],
            {"operation": "RouteRead", "route": route},
            group="route",
        )
    for split_candidate in _split_grid_candidates(width, rows)[:4]:
        add(
            f"split_{split_candidate['family']}",
            8,
            [TransformStep("SplitGridRoute", split_candidate["data"])],
            {"operation": "SplitGridRoute", **split_candidate["data"]},
            group="route",
        )
    for split in split_rows:
        split_label = "" if split == default_split else f"_s{split}"
        second_end = max(split, rows - 3)
        for across in (1, 2):
            add(
                f"ndown_top{split_label}_a{across}",
                10,
                [TransformStep("NDownMAcross", {"rangeStart": 0, "rangeEnd": split - 1, "down": 1, "across": across})],
                {
                    "operation": "NDownMAcross",
                    "band": "top",
                    "split": split,
                    "rangeStart": 0,
                    "rangeEnd": split - 1,
                    "down": 1,
                    "across": across,
                },
                group="ndown_top",
            )
        for lock_start, lock_end, lock_label in _program_lock_ranges(width, rows, split, usable):
            add(
                f"{lock_label}{split_label}",
                30,
                [TransformStep("LockCharacters", {"rangeStart": lock_start, "rangeEnd": lock_end})],
                {"operation": "LockCharacters", "split": split, "rangeStart": lock_start, "rangeEnd": lock_end},
                group="lock",
            )
        for across in (1, 2):
            add(
                f"ndown_lower{split_label}_a{across}",
                40,
                [TransformStep("NDownMAcross", {"rangeStart": split, "rangeEnd": second_end, "down": 1, "across": across})],
                {
                    "operation": "NDownMAcross",
                    "band": "lower",
                    "split": split,
                    "rangeStart": split,
                    "rangeEnd": second_end,
                    "down": 1,
                    "across": across,
                },
                group="ndown_lower",
            )
    for shift_start, shift_end, shift_label in _program_shift_ranges(width, rows, usable):
        if shift_start < shift_end:
            add(
                f"{shift_label}_right",
                20,
                [TransformStep("ShiftCharactersRight", {"rangeStart": shift_start, "rangeEnd": shift_end})],
                {"operation": "ShiftCharactersRight", "rangeStart": shift_start, "rangeEnd": shift_end},
                group="shift",
            )
            add(
                f"{shift_label}_left",
                20,
                [TransformStep("ShiftCharactersLeft", {"rangeStart": shift_start, "rangeEnd": shift_end})],
                {"operation": "ShiftCharactersLeft", "rangeStart": shift_start, "rangeEnd": shift_end},
                group="shift",
            )
    for tail_repairs, tail_label in _tail_repair_packs(width, rows, token_count):
        if tail_repairs:
            add(
                tail_label,
                50,
                [TransformStep("Reverse", {"rangeStart": start, "rangeEnd": end}) for start, end in tail_repairs],
                {"operation": "tail_repair_pack", "ranges": tail_repairs, "label": tail_label},
                group="tail_repair",
            )
    for start, end, label in _local_repair_ranges(token_count, width, rows)[:6]:
        add(
            f"reverse_{label}",
            60,
            [TransformStep("Reverse", {"rangeStart": start, "rangeEnd": end})],
            {"operation": "Reverse", "rangeStart": start, "rangeEnd": end, "label": label},
            group="local_repair",
        )
    return operations


def _program_operation_allowed(label: str, existing_labels: list[str]) -> bool:
    label_group = _program_label_group(label)
    existing_groups = [_program_label_group(item) for item in existing_labels]
    if label_group in {
        "route",
        "ndown_top",
        "shift",
        "lock",
        "ndown_lower",
        "tail_repair",
    } and label_group in existing_groups:
        return False
    if label_group == "route":
        return not existing_labels
    if label_group == "ndown_top":
        return "route" not in existing_groups
    if label_group == "shift":
        return (
            "ndown_top" in existing_groups
            or "route" in existing_groups
            or not existing_labels
        )
    if label_group == "lock":
        return "route" not in existing_groups
    if label_group == "ndown_lower":
        return (
            "ndown_top" in existing_groups
            and "route" not in existing_groups
            and ("shift" in existing_groups or "lock" in existing_groups)
            and _program_split_for_label(label) in {
                _program_split_for_label(item)
                for item in existing_labels
                if _program_label_group(item) == "ndown_top"
            }
        )
    if label_group == "tail_repair":
        return "ndown_lower" in existing_groups
    if label.startswith("reverse_"):
        return bool({"tail_repair", "route"} & set(existing_groups))
    return True


def _program_split_rows(rows: int) -> list[int]:
    default_split = max(1, rows // 2 - 1)
    raw = [default_split, rows // 3, rows // 2, (rows * 2) // 3]
    out: list[int] = []
    for split in raw:
        if 1 <= split < rows - 1 and split not in out:
            out.append(split)
    return out[:4]


def _program_shift_ranges(width: int, rows: int, usable: int) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for row, label in (
        (min(rows - 1, (rows * 7) // 10), "late_shift"),
        (max(0, rows // 2), "middle_shift"),
    ):
        start = min(row * width + max(0, width // 5), usable - 1)
        end = min(row * width + width - 1, usable - 1)
        if start < end:
            ranges.append((start, end, label))
    return ranges


def _program_lock_ranges(width: int, rows: int, split: int, usable: int) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for row, label in ((split, "mid_late_lock"), (max(0, split - 1), "upper_lock")):
        start = min(row * width + max(0, (width * 2) // 3), usable - 1)
        end = min(row * width + width - 1, usable - 1)
        if start < end:
            ranges.append((start, end, label))
    return ranges[:2]


def _tail_repair_packs(width: int, rows: int, token_count: int) -> list[tuple[list[tuple[int, int]], str]]:
    if width < 10 or rows < 4:
        return []
    tail_start = (rows - 2) * width
    if tail_start >= token_count:
        return []
    offsets = [
        (0, min(3, width - 1)),
        (max(0, width // 2), max(0, width // 2 + 1)),
        (max(0, width // 2 + 2), max(0, width // 2 + 3)),
        (max(0, width - 1), width + 1),
        (width + 2, width + 3),
        (width + 4, min(width + 11, width * 2 - 1)),
    ]
    ranges: list[tuple[int, int]] = []
    for start_offset, end_offset in offsets:
        start = tail_start + start_offset
        end = min(tail_start + end_offset, token_count - 1)
        if start < end:
            ranges.append((start, end))
    packs: list[tuple[list[tuple[int, int]], str]] = []
    if ranges:
        packs.append((ranges, "tail_repair_pack"))
    if len(ranges) >= 3:
        packs.append((ranges[:3], "tail_repair_short"))
    return packs


def _append_program_operation(
    seed: TransformCandidate,
    *,
    operation: dict[str, Any],
    candidate_index: int,
    depth: int,
) -> TransformCandidate:
    steps = tuple(seed.pipeline.steps) + tuple(operation["steps"])
    labels = list(seed.params.get("operation_labels") or []) + [operation["label"]]
    operations = list(seed.params.get("operations") or []) + [operation["params"]]
    template = _program_template_label(
        labels,
        width=seed.pipeline.columns,
        rows=seed.pipeline.rows,
    )
    params = {
        "program_depth": depth,
        "operation_labels": labels,
        "operations": operations,
        "last_operation_order": operation["order"],
        "grammar": "grid_program_v1",
    }
    if template:
        params["template"] = template
        params["constructed_template_match"] = True
        params["template_source"] = "program_grammar"
    family_suffix = template or operation["label"]
    return TransformCandidate(
        candidate_id=f"program_{candidate_index:04d}_d{depth}_{operation['label']}",
        family=f"program_{family_suffix}",
        params=params,
        pipeline=TransformPipeline(
            steps=steps,
            columns=seed.pipeline.columns,
            rows=seed.pipeline.rows,
        ),
        inverse_mode=seed.inverse_mode,
        grid=dict(seed.grid) if seed.grid else None,
        provenance="program_search",
    )


def _program_candidate_summary(candidate: TransformCandidate, score: float) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "score": round(score, 6),
        "grid": dict(candidate.grid or {}),
        "program_depth": candidate.params.get("program_depth"),
        "template": candidate.params.get("template"),
        "operation_labels": list(candidate.params.get("operation_labels") or []),
        "pipeline": candidate.pipeline.to_raw(),
    }


def _program_label_group(label: str) -> str:
    if label.startswith("route_") or label.startswith("split_"):
        return "route"
    if label.startswith("ndown_top"):
        return "ndown_top"
    if label.startswith("ndown_lower"):
        return "ndown_lower"
    if label.endswith("_shift_right") or label.endswith("_shift_left"):
        return "shift"
    if label.startswith("mid_late_lock") or label.startswith("upper_lock") or label.endswith("_lock"):
        return "lock"
    if label.startswith("tail_repair"):
        return "tail_repair"
    if label.startswith("reverse_"):
        return "local_repair"
    return "other"


def _program_split_for_label(label: str) -> int | None:
    marker = "_s"
    if marker not in label:
        return None
    tail = label.split(marker, 1)[1]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None


def _program_template_label(labels: list[str], *, width: int | None, rows: int | None) -> str | None:
    groups = [_program_label_group(label) for label in labels]
    if (
        groups == ["ndown_top", "shift", "lock", "ndown_lower", "tail_repair"]
        and _program_split_for_label(labels[0]) == _program_split_for_label(labels[3])
    ):
        return "banded_ndown_constructed"
    if groups == ["route", "local_repair"]:
        return "route_repair_constructed"
    return None


def _program_shape_bonus(candidate: TransformCandidate) -> float:
    labels = candidate.params.get("operation_labels") or []
    if candidate.params.get("template") == "banded_ndown_constructed":
        return 0.30
    if candidate.params.get("template") == "route_repair_constructed":
        return 0.18
    expected_prefix = ["ndown_top", "shift", "lock", "ndown_lower", "tail_repair"]
    groups = [_program_label_group(label) for label in labels]
    prefix_len = 0
    for observed, expected in zip(groups, expected_prefix):
        if observed != expected:
            break
        prefix_len += 1
    route_prefix = 0
    for observed, expected in zip(groups, ["route", "local_repair"]):
        if observed != expected:
            break
        route_prefix += 1
    return max(prefix_len * 0.06, route_prefix * 0.05)


def _append_mutation(
    seed: TransformCandidate,
    *,
    family_suffix: str,
    step: TransformStep,
    params: dict[str, Any],
    ordinal: int,
) -> TransformCandidate:
    pipeline = TransformPipeline(
        steps=tuple(seed.pipeline.steps) + (step,),
        columns=seed.pipeline.columns,
        rows=seed.pipeline.rows,
    )
    return TransformCandidate(
        candidate_id=f"{seed.candidate_id}m{ordinal:02d}_{family_suffix}",
        family=f"{seed.family}+{family_suffix}",
        params={"base_candidate_id": seed.candidate_id, **params},
        pipeline=pipeline,
        inverse_mode=seed.inverse_mode,
        grid=dict(seed.grid) if seed.grid else None,
        provenance="local_mutation",
    )


def _local_repair_ranges(
    token_count: int,
    columns: int | None,
    rows: int | None,
) -> list[tuple[int, int, str]]:
    if token_count <= 1:
        return []
    ranges: list[tuple[int, int, str]] = []

    def add(start: int, end: int, label: str) -> None:
        start = max(0, min(start, token_count - 1))
        end = max(0, min(end, token_count - 1))
        if start < end:
            ranges.append((start, end, label))

    for size in (2, 3, 4):
        add(0, size - 1, f"head{size}")
        add(token_count - size, token_count - 1, f"tail{size}")
    if columns and columns > 1:
        usable_rows = rows or token_count // columns
        interesting_rows = [0, 1, max(0, usable_rows // 2), max(0, usable_rows - 2), max(0, usable_rows - 1)]
        seen_rows: set[int] = set()
        for row in interesting_rows:
            if row in seen_rows or row < 0:
                continue
            seen_rows.add(row)
            start = row * columns
            if start >= token_count:
                continue
            add(start, min(start + columns - 1, token_count - 1), f"row{row}")
            add(max(start - 2, 0), min(start + 2, token_count - 1), f"boundary{row}")
    deduped: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for start, end, label in ranges:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((start, end, label))
    return deduped[:12]


def _row_band_ranges(rows: int) -> list[tuple[int, int, str]]:
    if rows <= 1:
        return []
    spans = {
        max(2, rows // 3),
        max(2, rows // 2),
        max(2, round(rows * 0.45)),
    }
    bands: list[tuple[int, int, str]] = []
    for span in sorted(spans):
        if span >= rows:
            continue
        starts = [0, max(0, (rows - span) // 2), rows - span]
        for start in starts:
            end = min(rows - 1, start + span - 1)
            bands.append((start, end, f"rows{start}_{end}"))
    deduped: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for start, end, label in bands:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((start, end, label))
    return deduped


def _dedupe_candidates(candidates: list[TransformCandidate]) -> list[TransformCandidate]:
    seen: set[str] = set()
    out: list[TransformCandidate] = []
    for candidate in candidates:
        raw = repr(candidate.pipeline.to_raw())
        if raw in seen:
            continue
        seen.add(raw)
        out.append(candidate)
    return out


def _token_hash(tokens: list[int]) -> str:
    h = hashlib.sha1()
    h.update(array("I", tokens).tobytes())
    return h.hexdigest()[:16]


def _tokens_for_order(tokens: list[int], order: list[int], *, limit: int | None = None) -> list[int]:
    selected = order if limit is None else order[:limit]
    return [tokens[position] for position in selected]


def _token_order_metrics_for_order(
    source_tokens: list[int],
    position_order: list[int],
    grid: dict[str, int] | None,
) -> dict[str, float]:
    if not source_tokens:
        return _token_order_metrics([], position_order, grid)
    total = len(position_order)
    counts: Counter[int] = Counter()
    bigram_counts: Counter[tuple[int, int]] = Counter()
    trigram_counts: Counter[tuple[int, int, int]] = Counter()
    alternation_hits = 0
    prev2: int | None = None
    prev1: int | None = None
    for idx, position in enumerate(position_order):
        value = source_tokens[position]
        counts[value] += 1
        if prev1 is not None:
            bigram_counts[(prev1, value)] += 1
        if prev2 is not None and prev1 is not None:
            trigram_counts[(prev2, prev1, value)] += 1
            if prev2 == value:
                alternation_hits += 1
        prev2 = prev1
        prev1 = value
    bigram_total = max(0, total - 1)
    trigram_total = max(0, total - 2)
    position_metrics = _position_order_metrics(position_order, grid)
    return {
        "repeated_bigram_rate": _repeat_rate_from_counts(bigram_counts, bigram_total),
        "repeated_trigram_rate": _repeat_rate_from_counts(trigram_counts, trigram_total),
        "alternation_rate": alternation_hits / trigram_total if trigram_total else 0.0,
        "symbol_ioc": sum(n * (n - 1) for n in counts.values()) / max(1, total * (total - 1)),
        **position_metrics,
    }


def _token_order_metrics(
    tokens: list[int],
    position_order: list[int],
    grid: dict[str, int] | None,
) -> dict[str, float]:
    if not tokens:
        return {
            "repeated_bigram_rate": 0.0,
            "repeated_trigram_rate": 0.0,
            "alternation_rate": 0.0,
            "symbol_ioc": 0.0,
            "position_nontriviality": 0.0,
            "position_adjacent_rate": 0.0,
            "position_step_repeat_rate": 0.0,
            "periodic_redundancy": 0.0,
            "inverse_periodic_redundancy": 0.0,
            "best_period": 0.0,
            "inverse_best_period": 0.0,
            "grid_row_step_rate": 0.0,
            "grid_column_step_rate": 0.0,
            "periodic_structure_score": 0.0,
            "matrix_rank_score": 0.0,
        }
    counts = Counter(tokens)
    total = len(tokens)
    position_metrics = _position_order_metrics(position_order, grid)
    return {
        "repeated_bigram_rate": _repeat_ngram_rate(tokens, 2),
        "repeated_trigram_rate": _repeat_ngram_rate(tokens, 3),
        "alternation_rate": _alternation_rate(tokens),
        "symbol_ioc": sum(n * (n - 1) for n in counts.values()) / max(1, total * (total - 1)),
        **position_metrics,
    }


def _fast_structural_metrics_for_order(
    position_order: list[int],
    grid: dict[str, int] | None,
    identity_positions: np.ndarray | None,
) -> dict[str, float]:
    """Fast position-only metrics for very large structural screens."""

    if len(position_order) < 2:
        return {
            "repeated_bigram_rate": 0.0,
            "repeated_trigram_rate": 0.0,
            "alternation_rate": 0.0,
            "symbol_ioc": 0.0,
            "position_nontriviality": 0.0,
            "position_adjacent_rate": 0.0,
            "position_step_repeat_rate": 0.0,
            "periodic_redundancy": 0.0,
            "inverse_periodic_redundancy": 0.0,
            "best_period": 0.0,
            "inverse_best_period": 0.0,
            "grid_row_step_rate": 0.0,
            "grid_column_step_rate": 0.0,
            "periodic_structure_score": 0.0,
            "matrix_rank_score": 0.0,
        }
    order = np.asarray(position_order, dtype=np.int32)
    n = int(order.size)
    identity = identity_positions
    if identity is None or int(identity.size) != n:
        identity = np.arange(n, dtype=np.int32)
    nontriviality = 1.0 - (float(np.count_nonzero(order == identity)) / n)
    deltas = np.diff(order)
    denom = int(deltas.size)
    adjacent_rate = float(np.count_nonzero(np.abs(deltas) == 1)) / denom
    _, counts = np.unique(deltas, return_counts=True)
    step_repeat_rate = float(counts[counts > 1].sum()) / denom

    columns = (grid or {}).get("columns")
    rows = (grid or {}).get("rows")
    column_step_rate = 0.0
    if columns and columns > 1:
        column_step_rate = float(np.count_nonzero(np.abs(deltas) == columns)) / denom
    periods = _periodic_probe_periods(denom, columns=columns, rows=rows)
    periodic_profile = _periodic_profile_np(deltas, periods=periods)
    periodic = periodic_profile["periodic_redundancy"]
    periodic_structure = nontriviality * periodic
    matrix_rank_score = nontriviality * (
        periodic * 0.45
        + step_repeat_rate * 0.25
        + max(adjacent_rate, column_step_rate) * 0.2
        + (1.0 - min(1.0, adjacent_rate)) * 0.1
    )
    return {
        "repeated_bigram_rate": 0.0,
        "repeated_trigram_rate": 0.0,
        "alternation_rate": 0.0,
        "symbol_ioc": 0.0,
        "position_nontriviality": nontriviality,
        "position_adjacent_rate": adjacent_rate,
        "position_step_repeat_rate": step_repeat_rate,
        "periodic_redundancy": periodic,
        "inverse_periodic_redundancy": 0.0,
        "best_period": periodic_profile["best_period"],
        "inverse_best_period": 0.0,
        "grid_row_step_rate": adjacent_rate if columns else 0.0,
        "grid_column_step_rate": column_step_rate,
        "periodic_structure_score": periodic_structure,
        "matrix_rank_score": matrix_rank_score,
    }


def _repeat_rate_from_counts(counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    repeated = sum(n for n in counts.values() if n > 1)
    return repeated / total


def _repeat_ngram_rate(tokens: list[int], n: int) -> float:
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    counts: Counter[tuple[int, ...]] = Counter(
        tuple(tokens[i:i + n]) for i in range(total)
    )
    return _repeat_rate_from_counts(counts, total)


def _repeat_rate(items: list[tuple[int, ...]]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    repeated = sum(n for n in counts.values() if n > 1)
    return repeated / len(items)


def _repeat_scalar_rate(items: list[int]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    repeated = sum(n for n in counts.values() if n > 1)
    return repeated / len(items)


def _alternation_rate(tokens: list[int]) -> float:
    if len(tokens) < 3:
        return 0.0
    hits = sum(1 for i in range(len(tokens) - 2) if tokens[i] == tokens[i + 2])
    return hits / (len(tokens) - 2)


def _position_order_metrics(order: list[int], grid: dict[str, int] | None) -> dict[str, float]:
    if len(order) < 2:
        return {
            "position_nontriviality": 0.0,
            "position_adjacent_rate": 0.0,
            "position_step_repeat_rate": 0.0,
            "periodic_redundancy": 0.0,
            "inverse_periodic_redundancy": 0.0,
            "best_period": 0.0,
            "inverse_best_period": 0.0,
            "grid_row_step_rate": 0.0,
            "grid_column_step_rate": 0.0,
            "periodic_structure_score": 0.0,
            "matrix_rank_score": 0.0,
        }
    n = len(order)
    fixed = sum(1 for i, value in enumerate(order) if i == value)
    nontriviality = 1.0 - fixed / n
    deltas = [order[i + 1] - order[i] for i in range(n - 1)]
    adjacent_rate = sum(1 for delta in deltas if abs(delta) == 1) / len(deltas)
    step_repeat_rate = _repeat_scalar_rate(deltas)
    columns = (grid or {}).get("columns")
    rows = (grid or {}).get("rows")
    periods = _periodic_probe_periods(len(deltas), columns=columns, rows=rows)
    periodic_profile = _periodic_profile(deltas, periods=periods)
    periodic = periodic_profile["periodic_redundancy"]
    inverse = _inverse_permutation(order)
    inverse_deltas = [inverse[i + 1] - inverse[i] for i in range(n - 1)]
    inverse_profile = _periodic_profile(inverse_deltas, periods=periods)
    inverse_periodic = inverse_profile["periodic_redundancy"]
    row_step_rate = 0.0
    column_step_rate = 0.0
    if columns and columns > 1:
        row_step_rate = adjacent_rate
        column_step_rate = sum(1 for delta in deltas if abs(delta) == columns) / len(deltas)
    periodic_structure = nontriviality * max(periodic, inverse_periodic)
    matrix_rank_score = nontriviality * (
        max(periodic, inverse_periodic) * 0.45
        + step_repeat_rate * 0.25
        + max(row_step_rate, column_step_rate) * 0.2
        + (1.0 - min(1.0, adjacent_rate)) * 0.1
    )
    return {
        "position_nontriviality": nontriviality,
        "position_adjacent_rate": adjacent_rate,
        "position_step_repeat_rate": step_repeat_rate,
        "periodic_redundancy": periodic,
        "inverse_periodic_redundancy": inverse_periodic,
        "best_period": periodic_profile["best_period"],
        "inverse_best_period": inverse_profile["best_period"],
        "grid_row_step_rate": row_step_rate,
        "grid_column_step_rate": column_step_rate,
        "periodic_structure_score": periodic_structure,
        "matrix_rank_score": matrix_rank_score,
    }


def _inverse_permutation(order: list[int]) -> list[int]:
    inverse = [0] * len(order)
    for output_index, input_index in enumerate(order):
        inverse[input_index] = output_index
    return inverse


def _periodic_probe_periods(
    length: int,
    *,
    columns: int | None,
    rows: int | None,
) -> list[int]:
    max_period = min(40, length // 2)
    if max_period <= 0:
        return []
    raw = [1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 32, 40]
    if columns and columns > 1:
        raw.extend([columns - 1, columns, columns + 1, columns * 2])
    if rows and rows > 1:
        raw.extend([rows - 1, rows, rows + 1, rows * 2])
    periods: list[int] = []
    for period in raw:
        if 1 <= period <= max_period and period not in periods:
            periods.append(period)
    return periods


def _periodic_profile(values: list[int], *, periods: list[int] | None = None) -> dict[str, float]:
    if len(values) < 4:
        return {"periodic_redundancy": 0.0, "best_period": 0.0}
    if periods is None:
        max_period = min(40, len(values) // 2)
        periods = list(range(1, max_period + 1))
    best = 0.0
    best_period = 0
    for period in periods:
        hits = sum(1 for i in range(len(values) - period) if values[i] == values[i + period])
        denom = len(values) - period
        if denom and hits / denom > best:
            best = hits / denom
            best_period = period
    return {"periodic_redundancy": best, "best_period": float(best_period)}


def _periodic_profile_np(values: np.ndarray, *, periods: list[int]) -> dict[str, float]:
    if int(values.size) < 4:
        return {"periodic_redundancy": 0.0, "best_period": 0.0}
    best = 0.0
    best_period = 0
    size = int(values.size)
    for period in periods:
        if period <= 0 or period >= size:
            continue
        denom = size - period
        hits = int(np.count_nonzero(values[:-period] == values[period:]))
        value = hits / denom if denom else 0.0
        if value > best:
            best = value
            best_period = period
    return {"periodic_redundancy": best, "best_period": float(best_period)}


def _combined_metric_score(metrics: dict[str, float]) -> float:
    return (
        metrics["repeated_bigram_rate"] * 2.0
        + metrics["repeated_trigram_rate"] * 3.0
        + metrics["alternation_rate"] * 0.5
        + metrics["symbol_ioc"] * 0.2
        + metrics["periodic_structure_score"] * 0.2
        + metrics["matrix_rank_score"] * 0.25
        + metrics["position_step_repeat_rate"] * metrics["position_nontriviality"] * 0.05
        + metrics["grid_column_step_rate"] * metrics["position_nontriviality"] * 0.05
    )
