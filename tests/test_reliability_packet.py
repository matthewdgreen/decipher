from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from investigation.state import InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_reliability_packet.py"
spec = importlib.util.spec_from_file_location("build_reliability_packet", SCRIPT)
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


@pytest.mark.parametrize("family", [*packet.FAMILIES, "bifid"])
def test_generators_preserve_token_identity_and_length(family):
    text = "THE JOURNAL RECORDS WHAT HAPPENED WHEN THE RIVER FROZE AND THE BOATS COULD NOT PASS"
    first = packet.encrypt_case(family, text, random.Random(304), variant=1)
    assert first == packet.encrypt_case(family, text, random.Random(304), variant=1)
    ciphertext, fmt, private = first
    assert private["roundtrip_verified"]
    expected = packet.letters(text)
    if family == "bifid":
        expected = expected.replace("J", "I")
    assert private["plaintext"] == expected
    case = packet.runtime_case("r0123456789ab", ciphertext, fmt, "en")
    assert set(case) == packet.RUNTIME_FIELDS
    assert "plaintext" not in case and "key" not in case and "family" not in case
    assert private["plaintext"] not in json.dumps(case)


def test_audit_separates_saved_verdict_from_corrected_rendering():
    workspace = Workspace(CipherText(raw="AB | CD", alphabet=Alphabet(list("ABCD")), separator=" | "),
                          plaintext_alphabet=Alphabet.standard_english())
    branch = workspace.fork("partial", "main")
    branch.key = dict(enumerate(range(4)))
    branch.metadata.update(candidate_renderer="key_with_null_mask_v1",
                           null_mask_selected={"mask": ["B"]}, decoded_text="ACD")
    state = InvestigationState(workspace=workspace, language="en")
    artifact = {
        "run_id": "audit_fixture", "ground_truth": "A CD", "status": "unsolved",
        "investigation_state": state.to_artifact_dict(),
        "solution": {"branch": "partial"},
        "branches": [{"name": "partial", "decryption": "ACD"}],
        "attestations": [{"content_hash": packet._candidate_content_hash("ACD"),
                          "reader_accepts_as_solution": False}],
        "loop_events": [{"event": "workspace_snapshot", "payload": {"branch": "partial", "decryption": "ACD"}}],
    }
    before = json.dumps(artifact, sort_keys=True)
    result = packet.audit_artifact(artifact)
    row = result["branches"][0]
    assert result["selected_branch"] == "partial"
    assert row["historical_boundary_loss"]
    assert row["matching_verdicts"] == []
    assert row["historical_unsegmented_verdict_count"] == 1
    assert row["roundtrip_content_equal"] and row["roundtrip_structure_equal"]
    assert row["saved_render_hash"] != row["content_hash"]
    assert result["historical_code_revision"] is None
    assert "preview" not in row
    assert "ground_truth" not in result
    assert json.dumps(artifact, sort_keys=True) == before


def test_missing_state_and_duplicate_sources_are_not_success(tmp_path):
    assert packet.audit_artifact({"run_id": "old"})["replay_status"] == "unavailable"
    missing = packet.audit_saved(tmp_path)
    assert all(row["reason"] == "missing_artifact" for row in missing)
    for name in ("one", "two"):
        target = tmp_path / name / (packet.AUDIT_RUNS[0] + ".json")
        target.parent.mkdir()
        target.write_text("{}")
    rows = packet.audit_saved(tmp_path)
    assert rows[0]["reason"] == "ambiguous_artifact"


def test_frozen_files_are_idempotent_and_never_overwritten(tmp_path):
    target = tmp_path / "packet.json"
    packet.write_new_or_identical(target, "original")
    packet.write_new_or_identical(target, "original")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        packet.write_new_or_identical(target, "changed")
    assert target.read_text() == "original"


def test_packet_source_split_and_runtime_firewall(tmp_path, monkeypatch):
    from types import SimpleNamespace

    records = [SimpleNamespace(
        id=f"source_{i}", source_file=f"book_{i}.txt", length_words=150,
        text="THE JOURNAL RECORDS THE RIVER AND THE BOATS " * 20,
        content_hash=f"hash_{i}", provenance="test source",
    ) for i in range(13)]
    # Additional offsets from the same book cannot count as separate sources.
    monkeypatch.setattr(packet, "load_library", lambda *a, **kw: records + records)
    runtime, grading = packet.build_packet(tmp_path / "missing_benchmark")
    assert len(runtime) == 14  # twelve generated cases and two controls
    assert len({row["case_id"] for row in runtime}) == 14
    assert all(set(row) == packet.RUNTIME_FIELDS for row in runtime)
    assert all(row["ciphertext_sha256"] == packet.digest(row["ciphertext"]) for row in runtime)
    generated = [row for row in grading["cases"] if row["family"] in packet.FAMILIES]
    assert len({row["source_file"] for row in generated}) == 12
    for family in packet.FAMILIES:
        pair = [row for row in generated if row["family"] == family]
        assert {row["role"] for row in pair} == {"development", "held_out"}
        assert len({row["generation_seed"] for row in pair}) == 2
    assert {row["case_id"] for row in grading["unavailable"] if "case_id" in row} == {
        packet.opaque_id(i) for i in range(12, 16)
    }
    # Generation and ordering are reproducible without any solver invocation.
    assert (runtime, grading) == packet.build_packet(tmp_path / "missing_benchmark")
