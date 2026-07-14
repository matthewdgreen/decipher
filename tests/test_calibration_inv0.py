"""Regression guard for the INV-0 reconciled source of truth.

``scripts/research/calibrate_inv0_scoring.py`` is the executable derivation of every
weight/threshold/fixture number in the spec. It MUST stay green: any change to the
shipped catalog that breaks it (or vice versa) is a divergence. This test runs its
FIXED-mode checks in-process.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "research"))

if not os.path.isdir(os.path.join(_REPO, "corpus_data", "en")):
    pytest.skip("corpus_data/en not available", allow_module_level=True)

import calibrate_inv0_scoring as C  # noqa: E402


def test_fixed_catalog_fixtures_pass(capsys):
    all_pass, _rows = C.fixture_table("FIXED")
    capsys.readouterr()
    assert all_pass


def test_fixed_catalog_beale_acceptance(capsys):
    ok = C.beale_acceptance("FIXED")
    capsys.readouterr()
    assert ok


def test_part5_island_calibration(capsys):
    ok = C.island_calibration()
    capsys.readouterr()
    assert ok


def test_original_catalog_reproduces_failures(capsys):
    # The ORIGINAL (pre-fix) catalog must still reproduce the reviewer failures,
    # proving the fixes are load-bearing.
    all_pass, _rows = C.fixture_table("ORIGINAL")
    beale_ok = C.beale_acceptance("ORIGINAL")
    capsys.readouterr()
    assert not all_pass
    assert not beale_ok
