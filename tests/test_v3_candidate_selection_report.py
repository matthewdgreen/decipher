from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from agent.loop_shared import _decoded_text_for_panel
from investigation.state import InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_v3_candidate_selection.py"
SPEC = importlib.util.spec_from_file_location("report_v3_candidate_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)


def _null_mask_workspace(*, separator: str | None) -> Workspace:
    cipher_alpha = Alphabet(list("ABCD"))
    raw = "AB | CD" if separator else "ABCD"
    workspace = Workspace(
        CipherText(raw=raw, alphabet=cipher_alpha, separator=separator),
        plaintext_alphabet=Alphabet.standard_english(),
    )
    branch = workspace.fork("nullmask", "main")
    branch.key = {index: index for index in range(4)}
    branch.metadata.update({
        "candidate_renderer": "key_with_null_mask_v1",
        "decoded_text": "ACD",
        "null_mask_selected": {"mask": ["B"]},
    })
    return workspace


def test_null_mask_renderer_preserves_source_boundaries():
    workspace = _null_mask_workspace(separator=" | ")
    assert workspace.get_branch("nullmask").word_spans is None
    assert _decoded_text_for_panel(workspace, "nullmask") == "A CD"


def test_null_mask_renderer_keeps_no_boundary_cipher_flat():
    workspace = _null_mask_workspace(separator=None)
    assert _decoded_text_for_panel(workspace, "nullmask") == "ACD"


def test_report_surfaces_historical_boundary_loss_and_post_hoc_effect():
    workspace = _null_mask_workspace(separator=" | ")
    state = InvestigationState(workspace=workspace, language="en")
    artifact = {
        "run_id": "fixture",
        "cipher_id": "fixture_case",
        "status": "unsolved",
        "ground_truth": "A CD",
        "investigation_state": state.to_artifact_dict(),
        "branch_roles": {
            "best_scored_branch": "nullmask",
            "workflow_branch": "nullmask",
            "latest_installed_branch": None,
            "declared_or_selected_branch": "nullmask",
        },
        "attestations": [],
        "loop_events": [{
            "event": "workspace_snapshot",
            "payload": {
                "iteration": 3,
                "branch": "nullmask",
                "scores": {"dict_rate": 1.0, "quad": -1.0},
                "decryption": "ACD",
            },
        }],
    }

    result = report.analyze_artifact_data(artifact)
    row = next(item for item in result["branches"] if item["branch"] == "nullmask")

    assert row["historical_boundary_loss"] is True
    assert row["boundary_mode"] == "source"
    assert row["post_hoc"]["char_accuracy"] == 1.0
    assert row["post_hoc"]["word_accuracy"] == 1.0
    assert row["unsegmented_variant"]["post_hoc"]["word_accuracy"] == 0.0
    assert any("flattened canonical source boundaries" in finding for finding in result["findings"])


def test_report_treats_equal_post_hoc_candidates_as_tied_best():
    workspace = _null_mask_workspace(separator=" | ")
    alias = workspace.fork("nullmask_alias", "nullmask")
    state = InvestigationState(workspace=workspace, language="en")
    artifact = {
        "run_id": "fixture_tie",
        "cipher_id": "fixture_case",
        "status": "unsolved",
        "ground_truth": "A CD",
        "investigation_state": state.to_artifact_dict(),
        "branch_roles": {
            "workflow_branch": alias.name,
            "declared_or_selected_branch": alias.name,
        },
        "attestations": [],
        "loop_events": [],
    }

    result = report.analyze_artifact_data(artifact)

    assert result["post_hoc_best_branches"] == ["nullmask", "nullmask_alias"]
    assert not any("Selected" in finding for finding in result["findings"])
