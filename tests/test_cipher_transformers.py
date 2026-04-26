from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.transformers import (  # noqa: E402
    TransformPipeline,
    apply_transform_pipeline,
    make_inverse_input_for_pipeline,
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

    def fake_homophonic(cipher_text, language, budget, refinement, solver_profile, ground_truth):
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
