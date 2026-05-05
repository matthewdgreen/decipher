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

NULL_REPORT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_copiale_null_probe.py"
null_report_spec = importlib.util.spec_from_file_location("report_copiale_null_probe", NULL_REPORT_PATH)
assert null_report_spec is not None and null_report_spec.loader is not None
copiale_null_report = importlib.util.module_from_spec(null_report_spec)
sys.modules[null_report_spec.name] = copiale_null_report
null_report_spec.loader.exec_module(copiale_null_report)


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
    masks = copiale_probe.generate_null_masks(["A", "B", "C", "D", "E"], max_mask_size=2)

    assert masks[:6] == [(), ("A",), ("B",), ("C",), ("D",), ("E",)]
    assert ("A", "B") in masks
    assert ("A", "E") in masks
    assert ("A", "B", "C") not in masks


def test_null_mask_validation_rewards_german_readability_signals():
    weak = {
        "mask": ["S001"],
        "filtered_length": 100,
        "selection_score": -8.0,
        "diagnostics": {
            "dict_rate": 0.45,
            "letter_count": 100,
            "segmentation_cost": 600,
        },
        "quality": {"top_letter_fraction": 0.25, "unique_letters": 14},
        "preview": "TERNARIRENUNDASSDIEINGEORDNENSEIENENNINSDIEDARBEINE",
    }
    readable = {
        "mask": ["S002"],
        "filtered_length": 100,
        "selection_score": -8.15,
        "diagnostics": {
            "dict_rate": 0.61,
            "letter_count": 100,
            "segmentation_cost": 500,
        },
        "quality": {"top_letter_fraction": 0.25, "unique_letters": 20},
        "preview": "DEKLARRETENDASSDIENESGEOFTNETSSENINEMLEEDENARBEIT",
    }

    weak_score = copiale_probe.null_mask_validation_score(weak, original_length=105)
    readable_score = copiale_probe.null_mask_validation_score(readable, original_length=105)

    assert readable_score["score"] > weak_score["score"]
    assert readable_score["components"]["dictionary"] > weak_score["components"]["dictionary"]
    assert readable_score["components"]["letter_diversity"] > weak_score["components"]["letter_diversity"]


def test_null_probe_report_summarizes_all_rows_and_validation_rank():
    payload = {
        "test_id": "copiale_probe_fixture",
        "mask_count": 3,
        "all_rows": [
            {
                "mask": ["S001"],
                "filtered_length": 100,
                "selection_score": -7.9,
                "validation_score": -6.8,
                "char_accuracy": 0.55,
                "diagnostics": {"dict_rate": 0.44},
                "quality": {"top_letter_fraction": 0.29},
                "preview": "TERNENUNDASSDIE",
            },
            {
                "mask": ["S002"],
                "filtered_length": 100,
                "selection_score": -8.2,
                "validation_score": -5.7,
                "char_accuracy": 0.70,
                "diagnostics": {"dict_rate": 0.61},
                "quality": {"top_letter_fraction": 0.24},
                "preview": "WENIGSICHUNDHER",
            },
        ],
    }

    report = copiale_null_report.summarize_probe_payload(payload)
    rendered = copiale_null_report.render_markdown([report], top=2)

    assert report["has_all_rows"] is True
    assert report["best_by_validation"]["mask"] == ["S002"]
    assert report["char_best_validation_rank"] == 1
    assert report["capture_by_validation_top_n"][1] is True
    assert report["capture_by_validation_top_n"][3] is True
    assert "validation exact-best hits" in rendered
    assert "validation top-3 captures" in rendered
    assert "copiale_probe_fixture" in rendered
    assert "S002" in rendered


def test_null_probe_report_explains_validation_misses():
    payload = {
        "test_id": "copiale_probe_miss_fixture",
        "mask_count": 2,
        "all_rows": [
            {
                "mask": ["S001"],
                "filtered_length": 100,
                "selection_score": -7.8,
                "validation_score": -4.0,
                "validation_components": {"selection": -7.8, "dictionary": 2.4},
                "char_accuracy": 0.60,
                "diagnostics": {"dict_rate": 0.60},
                "quality": {"top_letter_fraction": 0.24},
                "preview": "WENIGSICH",
            },
            {
                "mask": ["S002"],
                "filtered_length": 100,
                "selection_score": -8.1,
                "validation_score": -4.2,
                "validation_components": {"selection": -8.1, "dictionary": 2.1},
                "char_accuracy": 0.72,
                "diagnostics": {"dict_rate": 0.53},
                "quality": {"top_letter_fraction": 0.25},
                "preview": "DASUND",
            },
        ],
    }

    report = copiale_null_report.summarize_probe_payload(payload)
    rendered = copiale_null_report.render_markdown([report], top=2)

    assert report["char_best_validation_rank"] == 2
    assert report["capture_by_validation_top_n"][1] is False
    assert report["capture_by_validation_top_n"][3] is True
    assert "Validation miss analysis" in rendered
    assert "char_accuracy" in rendered
    assert "dictionary" in rendered
