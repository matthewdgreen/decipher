from __future__ import annotations

from cli import HOMOPHONIC_REFINEMENT_CHOICES


def test_cli_accepts_null_mask_homophonic_refinement_aliases():
    assert "null_masks" in HOMOPHONIC_REFINEMENT_CHOICES
    assert "homophonic_nulls" in HOMOPHONIC_REFINEMENT_CHOICES
    assert "copiale_nulls" in HOMOPHONIC_REFINEMENT_CHOICES
