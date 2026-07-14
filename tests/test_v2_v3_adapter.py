"""Tests for the v2-artifact -> v3-state adapter (M6 Part A / F2, F9a)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from investigation.adapter import (
    V2ArtifactAdapterError,
    state_from_v2_artifact,
)
from investigation.state import InvestigationState
from models.alphabet import Alphabet


FIXTURE = Path(__file__).parent / "fixtures" / "v2_artifact_synth_en_40wb_s1.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Round-trip on the real stored v2 artifact fixture (F9a)
# ---------------------------------------------------------------------------
def test_round_trip_fixpoint_on_real_v2_fixture():
    """F9a: adapter -> to_artifact_dict -> from_artifact_dict is a FIXPOINT.

    v2 -> v3 is one-way (the adapter); once in v3 form the state round-trips
    identically (the resume identity that keeps resume-artifact working).
    """
    artifact = _load_fixture()
    state = state_from_v2_artifact(artifact)

    d1 = state.to_artifact_dict()
    reloaded = InvestigationState.from_artifact_dict(d1)
    d2 = reloaded.to_artifact_dict()
    assert d1 == d2


def test_adapter_restores_cipher_branches_language_and_evidence():
    artifact = _load_fixture()
    state = state_from_v2_artifact(artifact)

    # Language carried over.
    assert state.language == artifact.get("language", "en")
    # Plaintext alphabet is standard English (F2b).
    assert state.workspace.plaintext_alphabet.symbols == Alphabet.standard_english().symbols
    # The cipher reconstructed (non-empty tokens).
    assert len(state.cipher.tokens) > 0
    # Branch snapshot(s) restored, including the declared `main`.
    assert "main" in state.workspace.branch_names()
    assert len(state.workspace.get_branch("main").key) > 0
    # Cipher identity landed in external_context (F2d), not a cipher_id field.
    assert "cipher_id=" in state.external_context
    assert artifact["cipher_id"] in state.external_context
    assert not hasattr(state, "cipher_id")
    # Key v2 findings became evidence.
    kinds = [e.kind for e in state.evidence_log]
    assert "v2_declaration" in kinds
    assert "v2_final_scores" in kinds


# ---------------------------------------------------------------------------
# Synthetic-artifact unit test (F2 + F2c error path)
# ---------------------------------------------------------------------------
def _synthetic_v2_artifact() -> dict:
    """A minimal v2 artifact: a plain-letter notation block + one branch + a
    declaration + final scores. No loop_version (v2/April era)."""
    notation = "GURQBT"  # continuous ciphertext (ROT13 of "THEDOG"); adapter needs no key
    prompt = (
        "You are analyzing a manuscript.\n\n"
        "## The manuscript notation system\n"
        f"```\n{notation}\n```\n\n"
        "Recover the plaintext."
    )
    return {
        "run_id": "deadbeef0001",
        "cipher_id": "synth_unit",
        "model": "fake-model",
        "language": "en",
        "status": "solved",
        "char_accuracy": 1.0,
        "word_accuracy": 1.0,
        "messages": [{"role": "user", "content": prompt}],
        "branches": [
            {
                "name": "main",
                "parent": None,
                "created_iteration": 0,
                "key": {"0": 19, "1": 7, "2": 4},  # G->T, U->H, R->E (ids)
                "tags": [],
                "mapped_count": 3,
                "char_accuracy": 1.0,
                "word_accuracy": 1.0,
                "decryption": "THE DOG",
            }
        ],
        "solution": {
            "branch": "main",
            "self_confidence": 0.9,
            "rationale": "ROT13 recovered; reads as English.",
            "declared_at_iteration": 3,
        },
    }


def test_synthetic_artifact_adapts_and_round_trips():
    artifact = _synthetic_v2_artifact()
    state = state_from_v2_artifact(artifact)

    assert state.language == "en"
    assert state.workspace.branch_names() == ["main"]
    assert state.workspace.get_branch("main").key == {0: 19, 1: 7, 2: 4}
    # cipher block reconstructed: one continuous word, 6 distinct symbols.
    assert len(state.cipher.words) == 1
    assert state.cipher.alphabet.size == 6
    # Declaration evidence captured the branch + confidence + rationale.
    decl = next(e for e in state.evidence_log if e.kind == "v2_declaration")
    assert decl.data["branch"] == "main"
    assert decl.data["self_confidence"] == 0.9
    assert "ROT13" in decl.data["rationale"]
    # Final-scores evidence captured per-branch accuracies.
    scores = next(e for e in state.evidence_log if e.kind == "v2_final_scores")
    assert scores.data["status"] == "solved"
    assert scores.data["branches"][0]["name"] == "main"
    assert scores.data["branches"][0]["char_accuracy"] == 1.0

    # Fixpoint on the synthetic artifact too.
    d1 = state.to_artifact_dict()
    d2 = InvestigationState.from_artifact_dict(d1).to_artifact_dict()
    assert d1 == d2


def test_canonical_transcription_notation_is_parsed():
    """A canonical S-token notation block (with `|`) parses via
    parse_canonical_transcription (multi-word structure preserved)."""
    artifact = _synthetic_v2_artifact()
    artifact["messages"][0]["content"] = (
        "## The manuscript notation system\n"
        "```\nS01 S02 S03 | S02 S04\n```\n"
    )
    artifact["branches"] = []
    state = state_from_v2_artifact(artifact)
    assert len(state.cipher.words) == 2
    assert state.cipher.alphabet.size == 4  # S01..S04


def test_unextractable_notation_block_raises_structured_error():
    """F2c: no fenced notation block -> a structured V2ArtifactAdapterError."""
    artifact = {
        "run_id": "x",
        "cipher_id": "y",
        "language": "en",
        "messages": [{"role": "user", "content": "no fenced code block at all"}],
    }
    with pytest.raises(V2ArtifactAdapterError):
        state_from_v2_artifact(artifact)


def test_empty_messages_raises_structured_error():
    with pytest.raises(V2ArtifactAdapterError):
        state_from_v2_artifact({"cipher_id": "z", "messages": []})


def test_v3_artifact_is_rejected():
    """A v3 artifact should load via from_artifact_dict, not this adapter."""
    with pytest.raises(V2ArtifactAdapterError):
        state_from_v2_artifact({"loop_version": "v3", "messages": []})
