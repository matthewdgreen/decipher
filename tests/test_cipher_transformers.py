from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.transformers import (  # noqa: E402
    TransformPipeline,
    apply_transform_pipeline,
    make_inverse_input_for_pipeline,
)
from analysis.transform_search import (  # noqa: E402
    generate_transform_candidates,
    inspect_transform_suspicion,
    iter_transform_candidates,
    plausible_grid_dimensions,
    screen_transform_candidates,
    validate_transform_candidate,
)
from automated import runner as automated_runner  # noqa: E402
from automated.runner import run_automated  # noqa: E402
from agent.tools_v2 import WorkspaceToolExecutor  # noqa: E402
from benchmark.loader import parse_canonical_transcription  # noqa: E402
from models.alphabet import Alphabet  # noqa: E402
from models.cipher_text import CipherText  # noqa: E402
from testgen.builder import build_test_case  # noqa: E402
from testgen.cache import PlaintextCache  # noqa: E402
from testgen.spec import TestSpec as SyntheticSpec  # noqa: E402
from workspace import Workspace  # noqa: E402


def test_reverse_and_shift_transformers_use_zenith_inclusive_ranges():
    tokens = [0, 1, 2, 3, 4]

    reversed_result = apply_transform_pipeline(
        tokens,
        {"steps": [{"name": "Reverse", "data": {"rangeStart": 1, "rangeEnd": 3}}]},
    )
    shifted_right = apply_transform_pipeline(
        tokens,
        {"steps": [{"name": "ShiftCharactersRight", "data": {"rangeStart": 1, "rangeEnd": 3}}]},
    )
    shifted_left = apply_transform_pipeline(
        tokens,
        {"steps": [{"name": "ShiftCharactersLeft", "data": {"rangeStart": 1, "rangeEnd": 3}}]},
    )

    assert reversed_result.tokens == [0, 3, 2, 1, 4]
    assert shifted_right.tokens == [0, 3, 1, 2, 4]
    assert shifted_left.tokens == [0, 2, 3, 1, 4]


def test_transposition_and_unwrap_transposition_are_inverses():
    tokens = [0, 1, 2, 3, 4, 5]
    transposed = apply_transform_pipeline(
        tokens,
        {"steps": [{"name": "Transposition", "data": {"key": "cab"}}]},
    )
    unwrapped = apply_transform_pipeline(
        transposed.tokens,
        {"steps": [{"name": "UnwrapTransposition", "data": {"key": "cab"}}]},
    )

    assert transposed.tokens == [1, 4, 2, 5, 0, 3]
    assert unwrapped.tokens == tokens


def test_route_read_transformers_cover_grid_orders():
    tokens = list(range(12))

    columns_down = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{"name": "RouteRead", "data": {"route": "columns_down"}}],
        },
    )
    rows_boustro = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{"name": "RouteRead", "data": {"route": "rows_boustrophedon"}}],
        },
    )
    spiral = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{"name": "RouteRead", "data": {"route": "spiral_clockwise"}}],
        },
    )

    assert columns_down.tokens == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    assert rows_boustro.tokens == [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11]
    assert spiral.tokens == [0, 1, 2, 3, 7, 11, 10, 9, 8, 4, 5, 6]
    assert sorted(spiral.tokens) == tokens


def test_n_down_m_across_with_locked_tail_matches_permutation_shape():
    pipeline = {
        "columns": 17,
        "rows": 20,
        "steps": [
            {"name": "LockCharacters", "data": {"rangeStart": 164, "rangeEnd": 169}},
            {"name": "NDownMAcross", "data": {"rangeStart": 0, "rangeEnd": 19, "down": 1, "across": 2}},
        ],
    }

    result = apply_transform_pipeline(list(range(340)), pipeline)

    assert sorted(result.tokens) == list(range(340))
    assert result.locked.count(True) == 6
    assert result.tokens[-6:] == [166, 167, 168, 169, 164, 165]


def test_inverse_input_for_pipeline_scrambles_to_replay_target():
    target = [0, 1, 2, 3, 4]
    pipeline = {
        "steps": [
            {"name": "Reverse", "data": {}},
            {"name": "ShiftCharactersLeft", "data": {"rangeStart": 1, "rangeEnd": 3}},
        ]
    }

    scrambled = make_inverse_input_for_pipeline(target, pipeline)
    replayed = apply_transform_pipeline(scrambled, pipeline)

    assert replayed.tokens == target


def test_automated_runner_replays_known_transform_before_homophonic(monkeypatch):
    cipher_text = parse_canonical_transcription("S001 S002 S003")
    seen = {}

    def fake_homophonic(cipher_text, language, budget, refinement, solver_profile, ground_truth, seed_offset=0):
        seen["tokens"] = list(cipher_text.tokens)
        return (
            "fake_homophonic",
            {0: 0, 1: 1, 2: 2},
            "ABC",
            {"name": "search_homophonic_anneal", "solver": "fake"},
        )

    monkeypatch.setattr(automated_runner, "_run_homophonic", fake_homophonic)
    result = run_automated(
        cipher_text=cipher_text,
        language="en",
        cipher_id="known_transform",
        cipher_system="transposition_homophonic",
        transform_pipeline={"steps": [{"name": "Reverse", "data": {}}]},
    )

    assert result.status == "completed"
    assert seen["tokens"] == [2, 1, 0]
    assert result.artifact["transform_pipeline"]["steps"][0]["name"] == "Reverse"
    assert result.artifact["steps"][0]["name"] == "apply_cipher_transform"


def test_mixed_transposition_without_pipeline_is_explicit_capability_gap():
    cipher_text = parse_canonical_transcription("S001 S002 S003")

    result = run_automated(
        cipher_text=cipher_text,
        language="en",
        cipher_id="missing_transform",
        cipher_system="transposition_homophonic",
    )

    assert result.status == "error"
    assert "requires an explicit ciphertext transform pipeline" in result.error_message


def test_synthetic_builder_records_transform_pipeline(tmp_path):
    spec = SyntheticSpec(
        language="en",
        approx_length=12,
        word_boundaries=False,
        homophonic=True,
        seed=7,
        transform_pipeline={
            "steps": [{"name": "Reverse", "data": {}}],
        },
    )
    cache = PlaintextCache(tmp_path / "cache")
    cache.put(spec, "THE QUICK BROWN FOX")

    test_data = build_test_case(spec, cache, api_key="")
    cipher_text = parse_canonical_transcription(test_data.canonical_transcription)
    replayed = apply_transform_pipeline(
        cipher_text.tokens,
        TransformPipeline.from_raw(test_data.transform_pipeline),
    )

    assert test_data.test.cipher_system == "transposition_homophonic"
    assert test_data.transform_pipeline is not None
    assert replayed.tokens != cipher_text.tokens
    assert len(replayed.tokens) == len(test_data.plaintext)


def test_workspace_branch_transform_overlay_changes_decode_order():
    alphabet = Alphabet(["A", "B", "C"])
    workspace = Workspace(
        CipherText(raw="ABC", alphabet=alphabet, source="test", separator=None),
        plaintext_alphabet=Alphabet(["X", "Y", "Z"]),
    )
    workspace.set_full_key("main", {0: 0, 1: 1, 2: 2})

    workspace.apply_transform_pipeline("main", {"steps": [{"name": "Reverse", "data": {}}]})

    assert workspace.effective_tokens("main") == [2, 1, 0]
    assert workspace.apply_key("main") == "ZYX"


def test_agent_transform_tool_applies_branch_overlay():
    alphabet = Alphabet(["A", "B", "C"])
    workspace = Workspace(
        CipherText(raw="ABC", alphabet=alphabet, source="test", separator=None),
        plaintext_alphabet=Alphabet(["X", "Y", "Z"]),
    )
    workspace.set_full_key("main", {0: 0, 1: 1, 2: 2})
    executor = WorkspaceToolExecutor(
        workspace=workspace,
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )

    out = executor._tool_act_apply_transform_pipeline({
        "branch": "main",
        "pipeline": {"steps": [{"name": "Reverse", "data": {}}]},
    })

    assert out["transformed"] is True
    assert workspace.apply_key("main") == "ZYX"
    assert out["pipeline"]["steps"][0]["name"] == "Reverse"


def test_transform_search_suspicion_prefers_grid_homophonic_cases():
    report = inspect_transform_suspicion(
        token_count=340,
        cipher_alphabet_size=63,
        plaintext_alphabet_size=26,
        word_group_count=1,
        cipher_system="unknown",
    )

    assert report["recommendation"] in {"run_screen", "consider_screen"}
    assert any(grid["columns"] == 17 and grid["rows"] == 20 for grid in report["plausible_grids"])
    assert "cipher alphabet is larger than plaintext alphabet" in report["reasons"]


def test_transform_candidate_screen_records_provenance_and_dedupes():
    tokens = [0, 1, 0, 2, 0, 1, 0, 2] * 3

    grids = plausible_grid_dimensions(len(tokens), columns=6)
    candidates = generate_transform_candidates(token_count=len(tokens), columns=6, profile="medium")
    screen = screen_transform_candidates(tokens, columns=6, profile="medium", top_n=5)

    assert grids[0]["columns"] == 6
    assert candidates[0].family == "identity"
    assert any(candidate.family == "row_reversals" for candidate in candidates)
    assert any(candidate.family == "route_columns_down" for candidate in candidates)
    assert any(candidate.family == "route_spiral_clockwise" for candidate in candidates)
    assert screen["candidate_count"] >= screen["deduped_candidate_count"] >= 1
    assert screen["family_counts"]
    assert screen["scored_family_counts"]
    assert screen["top_family_counts"]
    assert "rejected_reason_counts" in screen
    assert screen["identity_candidate"]["family"] == "identity"
    assert screen["anchor_candidates"]
    assert screen["top_candidates"][0]["pipeline"]["steps"] is not None
    assert "token_order_hash" in screen["top_candidates"][0]
    assert "periodic_redundancy" in screen["top_candidates"][0]["metrics"]
    assert "matrix_rank_score" in screen["top_candidates"][0]["metrics"]
    assert "best_period" in screen["top_candidates"][0]["metrics"]
    assert "position_order_preview" in screen["top_candidates"][0]


def test_wide_streaming_screen_keeps_simple_route_anchors_visible():
    tokens = [i % 23 for i in range(460)]

    screen = screen_transform_candidates(
        tokens,
        columns=17,
        profile="wide",
        top_n=5,
        max_generated_candidates=25000,
        streaming=True,
        include_program_search=True,
        program_max_depth=5,
        program_beam_width=48,
    )

    anchor_families = [candidate["family"] for candidate in screen["anchor_candidates"]]
    assert "route_rows_boustrophedon" in anchor_families
    assert "route_columns_up" in anchor_families
    assert "route_columns_down" in anchor_families


def test_plausible_grid_dimensions_keep_ragged_common_widths():
    dims = plausible_grid_dimensions(460, max_columns=120, max_results=24)
    widths = [item["columns"] for item in dims]

    assert 16 in widths
    assert 17 in widths
    assert 20 in widths


def test_route_read_supports_diagonal_and_offset_chain_routes():
    tokens = list(range(12))

    diagonal = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{"name": "RouteRead", "data": {"route": "diagonal_down_right"}}],
        },
    ).tokens
    offset = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{"name": "RouteRead", "data": {"route": "offset_chain", "step": 5}}],
        },
    ).tokens

    assert diagonal == [0, 1, 4, 2, 5, 8, 3, 6, 9, 7, 10, 11]
    assert sorted(offset) == tokens
    assert offset != tokens


def test_route_read_supports_grille_and_progressive_breadth_routes():
    tokens = list(range(12))

    checkerboard = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{"name": "RouteRead", "data": {"route": "checkerboard_even_odd"}}],
        },
    ).tokens
    progressive = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{"name": "RouteRead", "data": {"route": "rows_progressive_shift", "shift": 1}}],
        },
    ).tokens
    interleave = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{"name": "RouteRead", "data": {"route": "row_column_interleave"}}],
        },
    ).tokens

    assert checkerboard == [0, 2, 5, 7, 8, 10, 1, 3, 4, 6, 9, 11]
    assert progressive == [0, 1, 2, 3, 5, 6, 7, 4, 10, 11, 8, 9]
    assert sorted(interleave) == tokens
    assert interleave != tokens


def test_split_grid_route_supports_region_routes_and_swaps():
    tokens = list(range(12))

    horizontal = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{
                "name": "SplitGridRoute",
                "data": {
                    "orientation": "horizontal",
                    "split": 1,
                    "firstRoute": "rows_reverse",
                    "secondRoute": "rows",
                },
            }],
        },
    ).tokens
    vertical_swap = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{
                "name": "SplitGridRoute",
                "data": {
                    "orientation": "vertical",
                    "split": 2,
                    "firstRoute": "rows",
                    "secondRoute": "rows",
                    "regionOrder": "swap",
                },
            }],
        },
    ).tokens

    assert horizontal == [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
    assert vertical_swap == [2, 3, 6, 7, 10, 11, 0, 1, 4, 5, 8, 9]
    assert sorted(vertical_swap) == tokens


def test_grid_permute_supports_row_and_column_orders():
    tokens = list(range(12))

    row_reverse = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{"name": "GridPermute", "data": {"rowOrder": "reverse"}}],
        },
    ).tokens
    column_even_odd = apply_transform_pipeline(
        tokens,
        {
            "columns": 4,
            "rows": 3,
            "steps": [{"name": "GridPermute", "data": {"columnOrder": "even_odd"}}],
        },
    ).tokens

    assert row_reverse == [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    assert column_even_odd == [0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11]
    assert sorted(column_even_odd) == tokens


def test_transform_candidate_generation_includes_breadth_families():
    candidates = generate_transform_candidates(token_count=340, columns=17, profile="medium")
    families = {candidate.family for candidate in candidates}

    assert "route_diagonal_down_right" in families
    assert "route_checkerboard_even_odd" in families
    assert "route_row_column_interleave" in families
    assert any(family.startswith("route_rows_progressive_shift_") for family in families)
    assert any(family.startswith("route_offset_chain_") for family in families)
    assert any(family.startswith("split_horizontal_") for family in families)
    assert any(family.startswith("split_vertical_") for family in families)
    assert any(family.startswith("composite_") for family in families)
    assert "banded_ndown_lock_shift_across_2" in families
    assert "grid_permute_rows_reverse" in families
    assert "grid_permute_columns_outside_in" in families
    assert "columnar_transposition_reverse_key" in families
    assert "unwrap_transposition_even_odd_key" in families


def test_wide_transform_candidate_generation_expands_structural_families():
    medium = generate_transform_candidates(token_count=340, columns=17, profile="medium")
    wide = generate_transform_candidates(
        token_count=340,
        columns=17,
        profile="wide",
        max_candidates=5000,
    )
    families = {candidate.family for candidate in wide}

    assert len(wide) > len(medium)
    assert any(family.startswith("columnar_transposition_affine_") for family in families)
    assert any(family.startswith("route_offset_chain_") for family in families)
    assert any(family.startswith("range_reverse_") for family in families)
    assert any(family.startswith("split_horizontal_") for family in families)


def test_wide_transform_generation_uses_large_explicit_cap():
    count = 0
    saw_double_repair = False
    for candidate in iter_transform_candidates(
        token_count=340,
        columns=17,
        profile="wide",
        max_candidates=200000,
    ):
        count += 1
        if candidate.family.startswith("wide_route_double_repair_"):
            saw_double_repair = True

    assert count == 200000
    assert saw_double_repair is True


def test_wide_transform_screen_is_structural_only_and_capped():
    screen = screen_transform_candidates(
        list(range(340)),
        columns=17,
        profile="wide",
        top_n=25,
        max_generated_candidates=750,
        streaming=True,
        include_program_search=False,
    )

    assert screen["profile"] == "wide"
    assert screen["streaming"] is True
    assert screen["rust_structural_metrics"] is True
    assert screen["candidate_count"] <= 750
    assert screen["max_generated_candidates"] == 750
    assert screen["top_candidates"]
    assert "anneal_score" not in screen["top_candidates"][0]


def test_streaming_transform_screen_matches_materialized_top_candidates():
    tokens = [0, 1, 0, 2, 3, 1, 0, 2] * 8
    materialized = screen_transform_candidates(
        tokens,
        columns=8,
        profile="medium",
        top_n=12,
        streaming=False,
    )
    streamed = screen_transform_candidates(
        tokens,
        columns=8,
        profile="medium",
        top_n=12,
        streaming=True,
    )

    assert streamed["streaming"] is True
    assert streamed["candidate_count"] == materialized["candidate_count"]
    assert streamed["deduped_candidate_count"] == materialized["deduped_candidate_count"]
    assert [item["candidate_id"] for item in streamed["top_candidates"]] == [
        item["candidate_id"] for item in materialized["top_candidates"]
    ]


def test_program_search_constructs_banded_ndown_shape_from_grammar():
    screen = screen_transform_candidates(
        list(range(340)),
        columns=17,
        profile="medium",
        top_n=40,
        include_program_search=True,
        program_max_depth=5,
        program_beam_width=24,
    )
    constructed = next(
        candidate for candidate in screen["top_candidates"]
        if candidate["params"].get("operation_labels") == [
            "ndown_top_a2",
            "late_shift_right",
            "mid_late_lock",
            "ndown_lower_a2",
            "tail_repair_pack",
        ]
    )
    raw = constructed["pipeline"]
    step_names = [step["name"] for step in raw["steps"]]

    assert screen["program_candidate_count"] > 0
    assert screen["program_scored_candidate_count"] > 0
    assert screen["program_search"]["grammar"] == "grid_program_v1"
    assert constructed["params"]["operation_labels"] == [
        "ndown_top_a2",
        "late_shift_right",
        "mid_late_lock",
        "ndown_lower_a2",
        "tail_repair_pack",
    ]
    assert constructed["family"] == "program_banded_ndown_constructed"
    assert constructed["params"]["template"] == "banded_ndown_constructed"
    assert raw["columns"] == 17
    assert raw["rows"] == 20
    assert step_names[:4] == [
        "NDownMAcross",
        "ShiftCharactersRight",
        "LockCharacters",
        "NDownMAcross",
    ]
    assert raw["steps"][0]["data"] == {"rangeStart": 0, "rangeEnd": 8, "down": 1, "across": 2}
    assert raw["steps"][1]["data"] == {"rangeStart": 241, "rangeEnd": 254}
    assert raw["steps"][2]["data"] == {"rangeStart": 164, "rangeEnd": 169}
    assert raw["steps"][3]["data"] == {"rangeStart": 9, "rangeEnd": 17, "down": 1, "across": 2}
    assert step_names[4:] == ["Reverse"] * 6


def test_program_search_constructs_non_z340_banded_shape_from_grammar():
    screen = screen_transform_candidates(
        list(range(180)),
        columns=15,
        profile="medium",
        top_n=500,
        include_program_search=True,
        program_max_depth=5,
        program_beam_width=24,
    )
    constructed = next(
        candidate for candidate in screen["top_candidates"]
        if candidate["params"].get("template") == "banded_ndown_constructed"
    )

    assert constructed["family"] == "program_banded_ndown_constructed"
    assert constructed["params"]["operation_labels"][0].startswith("ndown_top")
    assert constructed["params"]["operation_labels"][-1].startswith("tail_repair")
    assert screen["program_search"]["top_programs"]
    assert "top_by_class" in screen["program_search"]["grids"][0]["layers"][0]


def test_transform_candidate_screen_can_add_local_mutations():
    tokens = [0, 1, 0, 2, 3, 1, 0, 2] * 6

    screen = screen_transform_candidates(
        tokens,
        columns=8,
        profile="medium",
        top_n=20,
        include_mutations=True,
        mutation_seed_count=4,
    )

    assert screen["mutation_candidate_count"] > 0
    assert screen["mutation_scored_candidate_count"] > 0
    assert screen["deduped_candidate_count"] >= screen["base_scored_candidate_count"]


def test_agent_transform_suspicion_tool_exposes_candidate_menu():
    alphabet = Alphabet(["A", "B", "C", "D", "E", "F"])
    workspace = Workspace(
        CipherText(raw="ABCDEF" * 20, alphabet=alphabet, source="test", separator=None),
        plaintext_alphabet=Alphabet(["X", "Y", "Z"]),
    )
    executor = WorkspaceToolExecutor(
        workspace=workspace,
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )

    out = executor._tool_observe_transform_suspicion({"branch": "main", "columns": 6})

    assert out["branch"] == "main"
    assert out["recommended_next_tool"] == "search_transform_homophonic"
    assert out["candidate_screen"]["top_candidates"]


def test_agent_wide_transform_candidate_tool_does_not_run_solver():
    alphabet = Alphabet(["A", "B", "C", "D", "E", "F"])
    workspace = Workspace(
        CipherText(raw="ABCDEF" * 30, alphabet=alphabet, source="test", separator=None),
        plaintext_alphabet=Alphabet(["X", "Y", "Z"]),
    )
    executor = WorkspaceToolExecutor(
        workspace=workspace,
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )

    out = executor._tool_search_transform_candidates({
        "branch": "main",
        "columns": 12,
        "breadth": "wide",
        "top_n": 10,
        "max_generated_candidates": 1000,
    })

    assert out["breadth"] == "wide"
    assert out["deduped_candidate_count"] > 0
    assert out["recommended_next_tool"] == "search_transform_homophonic"
    assert "anneal_score" not in out["top_candidates"][0]


def test_transform_screen_rejects_non_permutation_ndownmacross_candidate():
    tokens = list(range(88))
    candidates = generate_transform_candidates(token_count=88, columns=22, profile="small")
    bad = next(
        candidate for candidate in candidates
        if candidate.family == "ndownmacross_1_1" and candidate.grid == {"columns": 22, "rows": 4}
    )

    validation = validate_transform_candidate(len(tokens), bad)
    screen = screen_transform_candidates(tokens, columns=22, profile="small", top_n=20)

    assert validation["valid"] is False
    assert validation["reason"] == "not_a_permutation"
    assert any(
        item["candidate_id"] == bad.candidate_id and item["reason"] == "not_a_permutation"
        for item in screen["rejected_candidates"]
    )
    assert all(item["candidate_id"] != bad.candidate_id for item in screen["top_candidates"])


def test_agent_transform_search_skips_invalid_candidates(monkeypatch):
    alphabet = Alphabet(["A", "B", "C", "D", "E"])
    workspace = Workspace(
        CipherText(raw=("ABCDE" * 18)[:88], alphabet=alphabet, source="test", separator=None),
        plaintext_alphabet=Alphabet(["X", "Y", "Z", "W", "V"]),
    )
    executor = WorkspaceToolExecutor(
        workspace=workspace,
        language="en",
        word_set=set(),
        word_list=[],
        pattern_dict={},
    )

    def fake_run_automated(**kwargs):
        class Result:
            status = "completed"
            solver = "fake"
            elapsed_seconds = 0.01
            final_decryption = "XYZ"
            artifact = {
                "steps": [{"name": "search_homophonic_anneal", "anneal_score": -1.0}],
                "key": {"0": 0},
            }
        return Result()

    monkeypatch.setattr("agent.tools_v2.run_automated", fake_run_automated)

    out = executor._tool_search_transform_homophonic({
        "branch": "main",
        "columns": 22,
        "profile": "small",
        "top_n": 3,
        "write_best_branch": False,
    })

    assert out["top_candidates"]
    assert any(
        item["reason"] == "not_a_permutation"
        for item in out["structural_screen"]["rejected_candidates"]
    )


def test_agent_transform_search_returns_finalist_review_and_branches(monkeypatch):
    alphabet = Alphabet(["A", "B", "C", "D", "E"])
    workspace = Workspace(
        CipherText(raw=("ABCDE" * 18)[:88], alphabet=alphabet, source="test", separator=None),
        plaintext_alphabet=Alphabet(["X", "Y", "Z", "W", "V"]),
    )
    executor = WorkspaceToolExecutor(
        workspace=workspace,
        language="en",
        word_set={"XYZ"},
        word_list=["XYZ"],
        pattern_dict={},
    )

    def fake_run_automated(**kwargs):
        class Result:
            status = "completed"
            solver = "fake"
            elapsed_seconds = 0.01
            final_decryption = "XYZXYZXYZ"
            artifact = {
                "steps": [{"name": "search_homophonic_anneal", "anneal_score": -1.0}],
                "key": {"0": 0, "1": 1, "2": 2},
            }
        return Result()

    monkeypatch.setattr("agent.tools_v2.run_automated", fake_run_automated)

    out = executor._tool_search_transform_homophonic({
        "branch": "main",
        "columns": 22,
        "profile": "small",
        "top_n": 3,
        "write_best_branch": True,
        "write_candidate_branches": True,
        "candidate_branch_count": 2,
        "review_chars": 120,
        "good_score_gap": 0.25,
    })

    assert out["written_branch"] == "main_transform_best"
    assert out["search_session_id"] == "transform_search_1"
    assert out["written_candidate_branches"] == [
        "main_transform_rank1",
        "main_transform_rank2",
    ]
    assert executor.workspace.has_branch("main_transform_best")
    assert executor.workspace.has_branch("main_transform_rank1")
    assert executor.workspace.has_branch("main_transform_rank2")
    assert out["finalist_review_count"] == 3
    first = out["finalist_review"][0]
    assert first["rank"] == 1
    assert first["branch"] == "main_transform_rank1"
    assert first["quick_scores"] is not None
    assert first["basin"] is not None
    assert first["decoded_preview"].startswith("XYZ")
    assert first["readability_judgment"]["primary_signal"] == "agent_semantic_reading"
    assert "paraphrasable clause" in first["readability_judgment"]["agent_task"]
    assert first["ranking_score"]["primary"]["name"] == "agent_contextual_readability"
    assert first["ranking_score"]["primary"]["must_be_supplied_by_agent"] is True
    assert first["agent_readability_judgment"] is None
    assert "supporting" in first["ranking_score"]
    assert out["primary_ranking_signal"] == "agent_contextual_readability"
    assert out["numeric_scores_role"] == "supporting_evidence"
    card = executor._tool_workspace_branch_cards({"branch": "main_transform_rank1"})["cards"][0]
    assert card["transform_finalist"]["rank"] == 1
    assert card["transform_finalist"]["search_session_id"] == "transform_search_1"
    assert card["transform_finalist"]["primary_ranking_signal"] == "agent_contextual_readability"
    assert card["transform_finalist"]["numeric_scores_role"] == "supporting_evidence"
    assert "transform_rank_1" in card["tags"]

    blocked = executor._tool_meta_declare_solution({
        "branch": "main_transform_rank1",
        "rationale": "This transformed finalist looks promising.",
        "self_confidence": 0.7,
        "reading_summary": "Possible transformed solution.",
        "further_iterations_helpful": False,
        "further_iterations_note": "No obvious next work.",
    })
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "transform_finalist_readability_required"
    assert "act_rate_transform_finalist" in blocked["suggested_next_tools"]

    rating = executor._tool_act_rate_transform_finalist({
        "search_session_id": out["search_session_id"],
        "rank": 1,
        "readability_score": 1,
        "label": "word_islands_only",
        "rationale": "The preview has fragments but no paraphrasable clause.",
    })
    assert rating["status"] == "ok"
    assert rating["rating"]["readability_score"] == 1.0
    assert "main_transform_rank1" in rating["updated_branches"]
    assert rating["finalist"]["ranking_score"]["primary"]["value"] == 1.0
    card_after_rating = executor._tool_workspace_branch_cards({
        "branch": "main_transform_rank1",
    })["cards"][0]
    assert card_after_rating["transform_finalist"]["agent_readability_score"] == 1.0
    assert card_after_rating["transform_finalist"]["agent_readability_label"] == "word_islands_only"

    assert out["good_score_finalist_count"] >= out["finalist_review_count"]
    assert "more_good_score_finalists" in out
    assert "search_review_transform_finalists" in out["review_instruction"]
    assert "act_rate_transform_finalist" in out["review_instruction"]

    page = executor._tool_search_review_transform_finalists({
        "search_session_id": out["search_session_id"],
        "start_rank": 2,
        "count": 2,
    })
    assert page["finalist_review"][0]["rank"] == 2
    assert page["finalist_review"][0]["branch"] == "main_transform_rank2"
    assert page["finalist_review"][0]["readability_judgment"]["primary_signal"] == "agent_semantic_reading"
    assert page["finalist_review"][0]["ranking_score"]["ranking_rule"].startswith("Rank finalists")
    assert page["rated_finalist_count"] == 1
    assert page["review_instruction"]
    assert "contextual readability score" in page["review_instruction"]

    installed = executor._tool_act_install_transform_finalists({
        "search_session_id": out["search_session_id"],
        "ranks": [3],
    })
    assert installed["status"] == "ok"
    assert installed["installed"][0]["rank"] == 3
    assert executor.workspace.has_branch("main_transform_rank3")
