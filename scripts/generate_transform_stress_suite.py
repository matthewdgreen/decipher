#!/usr/bin/env python3
"""Generate the overnight synthetic transform stress suite.

This suite is intentionally broader and noisier than the curated frontier
suite.  It is meant for long no-LLM sweeps over transform families, not for
fast commit-time regression checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "frontier" / "transform_stress_overnight.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic overnight transform stress suite.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Destination JSONL suite file.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=180,
        help=(
            "Number of cases to emit. The default 180-case packet is the "
            "calibrated overnight suite; values up to 200 preserve the same "
            "family ordering."
        ),
    )
    args = parser.parse_args()

    cases = build_cases(max_cases=args.count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, separators=(",", ":"), sort_keys=False))
            handle.write("\n")

    print(f"Wrote {len(cases)} cases to {output}")
    print("Family counts:")
    counts: dict[str, int] = {}
    for case in cases:
        key = case["frontier_tags"][1]
        counts[key] = counts.get(key, 0) + 1
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


def build_cases(max_cases: int = 180) -> list[dict[str, Any]]:
    """Build deterministic synthetic transform stress cases."""

    if max_cases < 1:
        return []
    cases: list[dict[str, Any]] = []
    seed = 1000

    for recipe_index in range(112):
        family, pipeline, approx_length, floor, extra_tags = _pure_recipe(recipe_index)
        cases.append(_case(
            test_id=f"synth_en_{approx_length}ptnb_{family}_s{seed}",
            cipher_system="pure_transposition",
            description=(
                "Overnight synthetic pure-transposition stress case with hidden "
                f"{family.replace('_', ' ')} pipeline."
            ),
            frontier_class="known_good" if floor >= 0.90 else "shared_hard",
            tags=["transform_stress_overnight", family, "pure_transposition", *extra_tags],
            min_char=floor,
            max_elapsed=240.0,
            spec={
                "language": "en",
                "approx_length": approx_length,
                "word_boundaries": False,
                "transposition_only": True,
                "seed": seed,
                "topic": "general",
                "frequency_style": "normal",
                "transform_pipeline": pipeline,
            },
            hide_pipeline=True,
            notes=(
                "Pure-transposition stress row: the ciphertext order is hidden "
                "from the solver and should be recovered by the Rust-scored "
                "direct transposition screen."
            ),
        ))
        seed += 1
        if len(cases) >= max_cases:
            return cases

    for recipe_index in range(36):
        family, pipeline, approx_length, floor, extra_tags = _known_homophonic_recipe(recipe_index)
        cases.append(_case(
            test_id=f"synth_en_{approx_length}thonb_known_{family}_s{seed}",
            cipher_system="transposition_homophonic",
            description=(
                "Overnight synthetic transposition+homophonic replay stress case "
                f"with known {family.replace('_', ' ')} pipeline."
            ),
            frontier_class="known_good" if floor >= 0.90 else "shared_hard",
            tags=["transform_stress_overnight", family, "transposition_homophonic", "known_pipeline", *extra_tags],
            min_char=floor,
            max_elapsed=300.0,
            spec={
                "language": "en",
                "approx_length": approx_length,
                "word_boundaries": False,
                "homophonic": True,
                "seed": seed,
                "topic": "general",
                "frequency_style": "normal",
                "transform_pipeline": pipeline,
            },
            hide_pipeline=False,
            notes=(
                "Known-pipeline replay row: the transform is supplied to the "
                "solver, so this stresses transformer correctness plus the "
                "downstream homophonic solve."
            ),
        ))
        seed += 1
        if len(cases) >= max_cases:
            return cases

    for recipe_index in range(52):
        family, pipeline, approx_length, floor, extra_tags = _hidden_homophonic_recipe(recipe_index)
        cases.append(_case(
            test_id=f"synth_en_{approx_length}thonb_hidden_{family}_s{seed}",
            cipher_system="transposition_homophonic",
            description=(
                "Overnight synthetic transposition+homophonic candidate-search "
                f"stress case with hidden {family.replace('_', ' ')} pipeline."
            ),
            frontier_class="shared_hard",
            tags=["transform_stress_overnight", family, "transposition_homophonic", "hidden_pipeline", "candidate_search", *extra_tags],
            min_char=floor,
            max_elapsed=360.0,
            spec={
                "language": "en",
                "approx_length": approx_length,
                "word_boundaries": False,
                "homophonic": True,
                "seed": seed,
                "topic": "general",
                "frequency_style": "normal",
                "transform_pipeline": pipeline,
            },
            hide_pipeline=True,
            runner_options={
                "transform_search": "rank",
                "transform_search_profile": "fast",
            },
            notes=(
                "Hidden transform+homophonic row: the pipeline is withheld and "
                "the row opts into bounded transform ranking before a screen/full "
                "homophonic confirmation, depending on the suite command."
            ),
        ))
        seed += 1
        if len(cases) >= max_cases:
            return cases

    return cases[:max_cases]


def _case(
    *,
    test_id: str,
    cipher_system: str,
    description: str,
    frontier_class: str,
    tags: list[str],
    min_char: float,
    max_elapsed: float,
    spec: dict[str, Any],
    hide_pipeline: bool,
    notes: str,
    runner_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case: dict[str, Any] = {
        "test_id": test_id,
        "track": "transcription2plaintext",
        "cipher_system": cipher_system,
        "target_records": [],
        "context_records": [],
        "description": description,
        "frontier_class": frontier_class,
        "frontier_tags": tags,
        "expected_solvers": ["decipher-automated"],
        "expected_status_by_solver": {"decipher-automated": "completed"},
        "min_char_accuracy_by_solver": {"decipher-automated": min_char},
        "max_elapsed_seconds_by_solver": {"decipher-automated": max_elapsed},
        "synthetic_spec": spec,
        "notes": notes,
    }
    if hide_pipeline:
        case["hide_transform_pipeline_from_solver"] = True
    if runner_options:
        case["decipher_runner_options"] = runner_options
    return case


def _pure_recipe(index: int) -> tuple[str, dict[str, Any], int, float, list[str]]:
    families = (
        _matrix_rotate_recipe,
        _route_recipe,
        _route_offset_recipe,
        _route_composite_recipe,
        _rail_fence_recipe,
        _transmatrix_recipe,
        _mask_route_recipe,
        _turning_mask_recipe,
        _block_route_recipe,
        _split_grid_recipe,
    )
    return families[index % len(families)](index // len(families), pure=True)


def _known_homophonic_recipe(index: int) -> tuple[str, dict[str, Any], int, float, list[str]]:
    families = (
        _reverse_shift_recipe,
        _route_recipe,
        _route_composite_recipe,
        _matrix_rotate_recipe,
        _transmatrix_recipe,
        _rail_fence_recipe,
    )
    return families[index % len(families)](index // len(families), pure=False)


def _hidden_homophonic_recipe(index: int) -> tuple[str, dict[str, Any], int, float, list[str]]:
    families = (
        _route_recipe,
        _route_offset_recipe,
        _route_composite_recipe,
        _matrix_rotate_recipe,
        _rail_fence_recipe,
        _split_grid_recipe,
        _mask_route_recipe,
        _transmatrix_recipe,
    )
    return families[index % len(families)](index // len(families), pure=False, hidden=True)


def _matrix_rotate_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    widths = [11, 13, 15, 17, 19, 21, 24]
    directions = ["cw", "ccw"]
    width = widths[i % len(widths)]
    direction = directions[(i // len(widths)) % len(directions)]
    return (
        "matrix_rotate",
        {"steps": [{"name": "MatrixRotate", "data": {"width": width, "direction": direction}}]},
        _length_for(i, pure=pure, hidden=hidden),
        0.90 if pure else 0.85,
        [direction],
    )


def _route_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    routes = [
        "columns_down",
        "columns_up",
        "rows_boustrophedon",
        "columns_boustrophedon",
        "diagonal_down_right",
        "diagonal_up_left",
        "diagonal_zigzag_down_right",
        "spiral_clockwise",
        "spiral_counterclockwise",
        "checkerboard_even_odd",
    ]
    columns = [11, 13, 15, 17, 19, 21][i % 6]
    route = routes[(i // 6) % len(routes)]
    return (
        f"route_{route}",
        {"columns": columns, "steps": [{"name": "RouteRead", "data": {"route": route}}]},
        _length_for(i, pure=pure, hidden=hidden),
        0.95 if route in {"columns_down", "columns_up", "rows_boustrophedon"} else 0.45,
        ["route"],
    )


def _route_offset_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    routes = ["spiral_clockwise", "spiral_counterclockwise", "diagonal_down_right", "diagonal_up_left"]
    columns = [12, 15, 17, 20][i % 4]
    route = routes[(i // 4) % len(routes)]
    offset = [1, columns // 2, columns, columns * 2][i % 4]
    return (
        f"route_offset_{route}",
        {"columns": columns, "steps": [{"name": "RouteRead", "data": {"route": route, "orderOffset": offset}}]},
        _length_for(i + 2, pure=pure, hidden=hidden),
        0.75 if pure else 0.35,
        ["route_offset"],
    )


def _route_composite_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    routes = ["columns_down", "columns_up", "rows_boustrophedon", "diagonal_down_right", "spiral_clockwise"]
    repairs = ["matrix_rotate_cw", "matrix_rotate_ccw", "matrix_rotate_cw_reverse", "reverse"]
    columns = [12, 15, 17, 20][i % 4]
    route = routes[(i // 4) % len(routes)]
    repair = repairs[(i // 2) % len(repairs)]
    steps = [{"name": "RouteRead", "data": {"route": route}}]
    if repair == "reverse":
        steps.append({"name": "Reverse", "data": {}})
    else:
        direction = "cw" if "cw" in repair else "ccw"
        steps.append({"name": "MatrixRotate", "data": {"width": columns, "direction": direction}})
        if repair.endswith("_reverse"):
            steps.append({"name": "Reverse", "data": {}})
    return (
        "route_composite",
        {"columns": columns, "steps": steps},
        _length_for(i + 3, pure=pure, hidden=hidden),
        0.75 if pure else 0.30,
        ["route_composite"],
    )


def _rail_fence_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    rails = [3, 4, 5, 6, 7, 8][i % 6]
    rail_orders = ["top_down", "bottom_up", "even_odd", "odd_even"]
    params = {
        "rails": rails,
        "offset": i % max(1, 2 * (rails - 1)),
        "direction": "up" if i % 2 else "down",
        "railOrder": rail_orders[(i // 2) % len(rail_orders)],
    }
    return (
        "rail_fence",
        {"steps": [{"name": "RailFenceRoute", "data": params}]},
        _length_for(i + 4, pure=pure, hidden=hidden),
        0.65 if pure else 0.25,
        ["rail_fence"],
    )


def _transmatrix_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    pairs = [(11, 23), (12, 28), (14, 24), (16, 21), (17, 20), (19, 18), (21, 16)]
    w1, w2 = pairs[i % len(pairs)]
    direction = "ccw" if i % 2 else "cw"
    return (
        "transmatrix",
        {"steps": [{"name": "TransMatrix", "data": {"w1": w1, "w2": w2, "direction": direction}}]},
        180 if pure and i % 3 == 0 else _length_for(i + 5, pure=pure, hidden=hidden),
        0.55 if pure else 0.20,
        ["transmatrix"],
    )


def _mask_route_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    patterns = ["border", "cross", "checkerboard_even", "checkerboard_odd", "quadrants_tl_br", "quadrants_tr_bl"]
    route_pairs = [("rows", "rows"), ("rows_boustrophedon", "rows"), ("columns_down", "rows")]
    columns = [12, 15, 18, 21][i % 4]
    first_route, second_route = route_pairs[(i // 4) % len(route_pairs)]
    pattern = patterns[(i // 2) % len(patterns)]
    order = "complement_first" if i % 2 else "mask_first"
    return (
        f"mask_route_{pattern}",
        {
            "columns": columns,
            "steps": [{
                "name": "MaskRoute",
                "data": {
                    "pattern": pattern,
                    "firstRoute": first_route,
                    "secondRoute": second_route,
                    "maskOrder": order,
                },
            }],
        },
        _length_for(i + 6, pure=pure, hidden=hidden),
        0.45 if pure else 0.15,
        ["mask_route", pattern],
    )


def _turning_mask_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    block_size = [4, 6, 8][i % 3]
    pattern = "top_right_quadrant" if i % 2 else "top_left_quadrant"
    route = ["rows", "rows_boustrophedon", "columns_down"][(i // 3) % 3]
    direction = "ccw" if i % 2 else "cw"
    return (
        "turning_mask",
        {
            "steps": [{
                "name": "TurningMaskRoute",
                "data": {
                    "blockSize": block_size,
                    "pattern": pattern,
                    "route": route,
                    "direction": direction,
                    "turnOffset": i % 4,
                },
            }],
        },
        max(120, _length_for(i + 7, pure=pure, hidden=hidden)),
        0.40 if pure else 0.10,
        ["turning_mask"],
    )


def _block_route_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    block_size = [3, 4, 5, 6, 8][i % 5]
    columns = [6, 8, 10, 12, 15][(i // 2) % 5]
    route = ["columns_down", "columns_up", "rows_boustrophedon", "spiral_clockwise"][(i // 3) % 4]
    block_order = "reverse" if i % 2 else "normal"
    return (
        "block_route",
        {
            "steps": [{
                "name": "BlockRoute",
                "data": {
                    "blockSize": block_size,
                    "columns": columns,
                    "route": route,
                    "blockOrder": block_order,
                    "orderOffset": (i % 4) * columns,
                },
            }],
        },
        _length_for(i + 8, pure=pure, hidden=hidden),
        0.50 if pure else 0.15,
        ["block_route"],
    )


def _split_grid_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    columns = [12, 15, 18, 21][i % 4]
    orientation = "vertical" if i % 2 else "horizontal"
    split = 4 if orientation == "horizontal" else max(2, columns // 3)
    return (
        "split_grid_route",
        {
            "columns": columns,
            "steps": [{
                "name": "SplitGridRoute",
                "data": {
                    "orientation": orientation,
                    "split": split,
                    "firstRoute": "columns_down",
                    "secondRoute": "rows_boustrophedon",
                    "regionOrder": "swap" if i % 3 else "normal",
                },
            }],
        },
        _length_for(i + 9, pure=pure, hidden=hidden),
        0.35 if pure else 0.10,
        ["split_grid"],
    )


def _reverse_shift_recipe(i: int, *, pure: bool, hidden: bool = False) -> tuple[str, dict[str, Any], int, float, list[str]]:
    starts = [0, 20, 45, 70]
    start = starts[i % len(starts)]
    steps = [
        {"name": "Reverse", "data": {"rangeStart": start, "rangeEnd": start + 18}},
        {
            "name": "ShiftCharactersRight" if i % 2 else "ShiftCharactersLeft",
            "data": {"rangeStart": start + 30, "rangeEnd": start + 48},
        },
    ]
    if i % 3 == 0:
        steps.append({"name": "Reverse", "data": {}})
    return (
        "reverse_shift",
        {"steps": steps},
        _length_for(i, pure=pure, hidden=hidden),
        0.90,
        ["range_ops"],
    )


def _length_for(i: int, *, pure: bool, hidden: bool) -> int:
    if pure:
        lengths = [80, 100, 120, 140, 160, 180, 220]
    elif hidden:
        lengths = [80, 100, 120, 140]
    else:
        lengths = [80, 100, 120, 150]
    return lengths[i % len(lengths)]


if __name__ == "__main__":
    main()
