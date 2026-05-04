from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPORT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_copiale_evidence.py"
report_spec = importlib.util.spec_from_file_location("report_copiale_evidence", REPORT_PATH)
assert report_spec is not None and report_spec.loader is not None
copiale_report = importlib.util.module_from_spec(report_spec)
sys.modules[report_spec.name] = copiale_report
report_spec.loader.exec_module(copiale_report)

PROBE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_copiale_null_masks.py"
probe_spec = importlib.util.spec_from_file_location("probe_copiale_null_masks", PROBE_PATH)
assert probe_spec is not None and probe_spec.loader is not None
copiale_probe = importlib.util.module_from_spec(probe_spec)
sys.modules[probe_spec.name] = copiale_probe
probe_spec.loader.exec_module(copiale_probe)


def test_diagnose_canonical_transcription_reports_cipher_side_features():
    diag = copiale_report.diagnose_canonical_transcription(
        "S001 S002 | S001 S003 | S001 S002 | S004"
    )

    assert diag["token_count"] == 7
    assert diag["word_count"] == 4
    assert diag["unique_symbols"] == 4
    assert diag["singleton_symbol_count"] == 2
    assert diag["top_symbols"][0]["symbol"] == "S001"
    assert diag["top_symbols"][0]["count"] == 3
    assert diag["repeated_cipher_words"] == [{"word": "S001 S002", "count": 2}]
    assert "short_page" in diag["diagnostic_flags"]


def test_diagnose_canonical_transcription_uses_solver_artifact_for_basin_diagnostics():
    diag = copiale_report.diagnose_canonical_transcription(
        "S001 S002 S001 S003 | S004 S001",
        artifact={
            "key": {"0": 4, "1": 13, "2": 18, "3": 4},
            "steps": [{"quality": {"collapsed": True}}],
        },
    )

    assert diag["homophone_families"][0]["letter"] == "E"
    assert diag["homophone_families"][0]["symbol_count"] == 2
    assert diag["null_codeword_candidates"]
    assert any(
        "maps-to-collapsed-E" in item["reasons"]
        for item in diag["null_codeword_candidates"]
    )
    assert "possible_null_or_function_symbols" in diag["diagnostic_flags"]


def test_calibrate_against_ground_truth_counts_insertion_symbols():
    calibration = copiale_report.calibrate_against_ground_truth(
        canonical_text="S001 S002 S003",
        artifact={"decryption": "AXBC"},
        ground_truth="ABC",
    )

    assert calibration["mode"] == "post_hoc_ground_truth_only"
    assert calibration["length_gap_decoded_minus_truth"] == 1
    assert calibration["alignment_insertions"] == 1
    assert calibration["top_insertion_symbols"][0]["symbol"] == "S002"
    assert calibration["top_insertion_symbols"][0]["insertions"] == 1


def test_probe_generates_bounded_null_masks():
    masks = copiale_probe._generate_masks(["A", "B", "C", "D", "E"], max_mask_size=2)

    assert masks[:6] == [(), ("A",), ("B",), ("C",), ("D",), ("E",)]
    assert ("A", "B") in masks
    assert ("A", "E") in masks
    assert ("A", "B", "C") not in masks
