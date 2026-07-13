from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from analysis.candidate_packet import (
    CandidatePacket,
    packet_from_null_mask_row,
    packet_from_pure_transposition_row,
    packet_from_transform_row,
)
from agent.finalist_sessions import FinalistSessionStore


# ---------------------------------------------------------------------------
# Realistic fixture rows (lifted from the three regression tests' mock data).
# ---------------------------------------------------------------------------

def _null_mask_row() -> dict:
    return {
        "status": "completed",
        "mask": ["01"],
        "mask_size": 1,
        "filtered_length": 5,
        "key": {"0": 22, "1": 4, "2": 13},
        "decryption": "WENIGUND",
        "preview": "WENIGUND",
        "selection_score": -4.2,
        "ensemble_score_v1": 2.8,
        "validation_score_v2": -3.2,
        "confirmed_validation_score_v2": -3.1,
        "validation_components_v2": {"content": 0.5},
        "diagnostics": {
            "dict_rate": 0.62,
            "segmentation_cost": 12,
            "dictionary_content_word_count": 2,
        },
        "quality": {"top_letter_fraction": 0.2, "unique_letters": 12},
    }


def _transform_row() -> dict:
    return {
        "candidate_index": 0,
        "candidate": {
            "candidate_id": "011_route_columns_down",
            "family": "route_columns_down",
            "params": {"columns": 20, "route": "columns_down"},
        },
        "pipeline": [{"name": "RouteRead", "params": {"route": "columns_down"}}],
        "status": "completed",
        "solver": "fake_homophonic",
        "anneal_score": -1.0,
        "elapsed_seconds": 0.01,
        "decoded_preview": "XYZXYZXYZ",
        "validation": {"validation_score": 1.0, "validation_label": "coherent_candidate"},
        "key": {"0": 0, "1": 1, "2": 2},
    }


def _pure_transposition_row() -> dict:
    return {
        "candidate_id": "011_route_columns_down",
        "family": "route_columns_down",
        "params": {"columns": 20, "route": "columns_down", "rows": 6},
        "pipeline": {"steps": [{"name": "RouteRead"}], "columns": 20, "rows": 6},
        "inverse_mode": None,
        "grid": {"columns": 20, "rows": 6},
        "provenance": "bounded_search",
        "rank": 1,
        "rust_rank": 1,
        "score": -5.67255,
        "selection_score": -5.67255,
        "plaintext": "THEQUICKBROWNFOX",
        "preview": "THEQUICKBROWNFOX",
        "validation": {"validation_score": 2.72, "validation_label": "coherent_candidate"},
        "validated_selection_score": -4.2974,
    }


# ---------------------------------------------------------------------------
# Adapter field-mapping + round-trip
# ---------------------------------------------------------------------------

def test_null_mask_adapter_fields_and_round_trip():
    row = _null_mask_row()
    packet = packet_from_null_mask_row(row, source_branch="main")

    assert packet.kind == "null_mask"
    # No candidate_id on the row -> derived from the mask.
    assert packet.candidate_id == "mask:01"
    assert packet.text == "WENIGUND"
    assert packet.preview == "WENIGUND"
    assert packet.validation is None  # null-mask rows have no finalist_validation block
    assert packet.solver_scores == {
        "selection_score": -4.2,
        "validation_score_v2": -3.2,
        "confirmed_validation_score_v2": -3.1,
        "ensemble_score_v1": 2.8,
    }
    assert packet.provenance == {
        "mask": ["01"],
        "mask_size": 1,
        "filtered_length": 5,
        "key": {"0": 22, "1": 4, "2": 13},
    }
    assert packet.source == {"solver": None, "source_branch": "main"}
    # validation_components_v2 / diagnostics / quality preserved under extras.
    assert packet.extras["validation_components_v2"] == {"content": 0.5}
    assert packet.extras["diagnostics"]["dict_rate"] == 0.62
    assert packet.extras["quality"]["unique_letters"] == 12

    assert CandidatePacket.from_dict(packet.to_dict()) == packet


def test_null_mask_adapter_uses_explicit_candidate_id_and_language_quality():
    row = _null_mask_row()
    row["candidate_id"] = "src:000123"
    row["language_quality_raw_score"] = 0.7
    row["language_quality_score"] = 0.8
    packet = packet_from_null_mask_row(row)
    assert packet.candidate_id == "src:000123"
    assert packet.extras["language_quality"] == {
        "language_quality_raw_score": 0.7,
        "language_quality_score": 0.8,
    }
    assert CandidatePacket.from_dict(packet.to_dict()) == packet


def test_transform_adapter_fields_and_round_trip():
    row = _transform_row()
    packet = packet_from_transform_row(row)

    assert packet.kind == "transform"
    assert packet.candidate_id == "011_route_columns_down"
    assert packet.text is None
    assert packet.preview == "XYZXYZXYZ"
    assert packet.solver_scores == {"anneal_score": -1.0}
    assert packet.validation == {
        "validation_score": 1.0,
        "validation_label": "coherent_candidate",
    }
    assert packet.provenance == {
        "pipeline": [{"name": "RouteRead", "params": {"route": "columns_down"}}],
        "key": {"0": 0, "1": 1, "2": 2},
        "family": "route_columns_down",
        "params": {"columns": 20, "route": "columns_down"},
    }
    assert CandidatePacket.from_dict(packet.to_dict()) == packet


def test_transform_adapter_candidate_id_fallback_from_index():
    row = _transform_row()
    row["candidate"] = {"family": "route", "params": {}}  # no candidate_id
    packet = packet_from_transform_row(row)
    assert packet.candidate_id == "cand_0"
    assert CandidatePacket.from_dict(packet.to_dict()) == packet


def test_transform_adapter_handles_runner_flat_shape():
    # The automated-runner transform row has no nested "candidate" / "candidate_index";
    # the adapter must not raise and should still map the flat fields.
    runner_row = {
        "candidate_id": "007_route",
        "family": "route_columns_down",
        "params": {"columns": 20},
        "provenance": "bounded_search",
        "pipeline": [{"name": "RouteRead"}],
        "status": "completed",
        "solver": "zenith_native",
        "anneal_score": -2.0,
        "decryption_preview": "THEQUICK",
        "decryption": "THEQUICKBROWNFOX",
        "key": {"0": 0},
        "validation": {"validation_score": 2.0},
    }
    packet = packet_from_transform_row(runner_row)
    assert packet.candidate_id == "007_route"
    assert packet.preview == "THEQUICK"
    assert packet.provenance["family"] == "route_columns_down"
    assert packet.provenance["params"] == {"columns": 20}
    # The full decryption is not dropped -- it lands in extras.
    assert packet.extras["decryption"] == "THEQUICKBROWNFOX"
    assert CandidatePacket.from_dict(packet.to_dict()) == packet


def test_pure_transposition_adapter_fields_and_round_trip():
    row = _pure_transposition_row()
    packet = packet_from_pure_transposition_row(row)

    assert packet.kind == "pure_transposition"
    assert packet.candidate_id == "011_route_columns_down"
    assert packet.rank == 1
    assert packet.text == "THEQUICKBROWNFOX"
    assert packet.preview == "THEQUICKBROWNFOX"
    assert packet.solver_scores == {
        "score": -5.67255,
        "selection_score": -5.67255,
        "validated_selection_score": -4.2974,
        "rust_rank": 1,
    }
    assert packet.validation == {
        "validation_score": 2.72,
        "validation_label": "coherent_candidate",
    }
    assert packet.provenance == {
        "candidate_id": "011_route_columns_down",
        "family": "route_columns_down",
        "params": {"columns": 20, "route": "columns_down", "rows": 6},
        "pipeline": {"steps": [{"name": "RouteRead"}], "columns": 20, "rows": 6},
        "inverse_mode": None,
        "grid": {"columns": 20, "rows": 6},
        "provenance": "bounded_search",
    }
    assert CandidatePacket.from_dict(packet.to_dict()) == packet


# ---------------------------------------------------------------------------
# Unknown-key tolerance and non-mutation (all three adapters)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "adapter, make_row",
    [
        (packet_from_null_mask_row, _null_mask_row),
        (packet_from_transform_row, _transform_row),
        (packet_from_pure_transposition_row, _pure_transposition_row),
    ],
)
def test_adapter_unknown_key_lands_in_extras(adapter, make_row):
    row = make_row()
    row["surprise_new_field"] = {"nested": [1, 2, 3]}
    packet = adapter(row)
    assert packet.extras["surprise_new_field"] == {"nested": [1, 2, 3]}
    assert CandidatePacket.from_dict(packet.to_dict()) == packet


@pytest.mark.parametrize(
    "adapter, make_row",
    [
        (packet_from_null_mask_row, _null_mask_row),
        (packet_from_transform_row, _transform_row),
        (packet_from_pure_transposition_row, _pure_transposition_row),
    ],
)
def test_adapter_does_not_mutate_input_row(adapter, make_row):
    row = make_row()
    before = copy.deepcopy(row)
    adapter(row)
    assert row == before


def test_adapter_rank_parameter_overrides_row_rank():
    # Explicit menu-position rank wins; row rank is only a fallback.
    null_packet = packet_from_null_mask_row(_null_mask_row(), rank=3)
    assert null_packet.rank == 3
    transform_packet = packet_from_transform_row(_transform_row(), rank=2)
    assert transform_packet.rank == 2
    pure_row = _pure_transposition_row()  # carries rank=1 natively
    assert packet_from_pure_transposition_row(pure_row, rank=4).rank == 4
    assert packet_from_pure_transposition_row(pure_row).rank == 1  # fallback


def test_from_dict_promotes_unknown_top_level_keys_into_extras():
    packet = packet_from_transform_row(_transform_row())
    data = packet.to_dict()
    data["mystery"] = 42
    restored = CandidatePacket.from_dict(data)
    assert restored.extras["mystery"] == 42
    # Round-trip is now stable (mystery stays in extras, not re-promoted).
    assert CandidatePacket.from_dict(restored.to_dict()) == restored


# ---------------------------------------------------------------------------
# FinalistSessionStore
# ---------------------------------------------------------------------------

def _packets(kind: str, n: int) -> list[CandidatePacket]:
    return [CandidatePacket(candidate_id=f"{kind}_{i}", kind=kind) for i in range(n)]


def test_store_id_sequence_is_per_kind():
    store = FinalistSessionStore()
    assert store.new_session("transform_search", {}, packets=[]) == "transform_search_1"
    assert store.new_session("transform_search", {}, packets=[]) == "transform_search_2"
    assert store.new_session("pure_transposition", {}, packets=[]) == "pure_transposition_1"
    assert store.new_session("null_mask", {}, packets=[]) == "null_mask_1"
    assert store.new_session("transform_search", {}, packets=[]) == "transform_search_3"
    assert store.new_session("null_mask", {}, packets=[]) == "null_mask_2"


def test_store_attaches_packet_dicts_to_payload():
    store = FinalistSessionStore()
    packets = _packets("null_mask", 2)
    payload: dict = {"ranked": ["a", "b"]}
    session_id = store.new_session("null_mask", payload, packets=packets)
    stored = store.get("null_mask", session_id)
    assert stored is payload  # stored by reference
    assert stored["packets"] == [p.to_dict() for p in packets]
    assert stored["ranked"] == ["a", "b"]  # existing keys preserved verbatim


def test_store_get_unknown_returns_none():
    store = FinalistSessionStore()
    assert store.get("null_mask", "null_mask_99") is None
    assert store.get("transform_search", "transform_search_1") is None


def test_store_find_id_uses_identity_not_equality():
    store = FinalistSessionStore()
    payload = {"source_branch": "main"}
    session_id = store.new_session("transform_search", payload, packets=[])
    assert store.find_id("transform_search", payload) == session_id
    # An equal-but-distinct dict is not found (identity, not equality).
    assert store.find_id("transform_search", {"source_branch": "main"}) is None
    # Wrong kind is not found.
    assert store.find_id("null_mask", payload) is None


def test_store_iteration_order():
    store = FinalistSessionStore()
    p1 = {"n": 1}
    p2 = {"n": 2}
    p3 = {"n": 3}
    id1 = store.new_session("null_mask", p1, packets=[])
    id2 = store.new_session("null_mask", p2, packets=[])
    id3 = store.new_session("null_mask", p3, packets=[])
    assert list(store.sessions("null_mask")) == [(id1, p1), (id2, p2), (id3, p3)]
    assert list(store.last_sessions("null_mask")) == [(id3, p3), (id2, p2), (id1, p1)]
    assert list(store.sessions("transform_search")) == []


# ---------------------------------------------------------------------------
# Dependency-freedom (v3 requirement)
# ---------------------------------------------------------------------------

def test_finalist_sessions_module_imports_no_tools_or_loop():
    src = Path(__file__).resolve().parents[1] / "src" / "agent" / "finalist_sessions.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("tools_v2" in name or "loop_v2" in name for name in imported), (
        f"finalist_sessions must stay dependency-free; imports={sorted(imported)}"
    )
