from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from analysis.language_scoring import LANGUAGE_QUALITY_FEATURES, LinearLanguageQualityModel
from models.alphabet import Alphabet


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

BREADTH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_copiale_breadth_experiment.py"
breadth_spec = importlib.util.spec_from_file_location("run_copiale_breadth_experiment", BREADTH_PATH)
assert breadth_spec is not None and breadth_spec.loader is not None
copiale_breadth = importlib.util.module_from_spec(breadth_spec)
sys.modules[breadth_spec.name] = copiale_breadth
breadth_spec.loader.exec_module(copiale_breadth)

BREADTH_DIAG_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_copiale_breadth_diagnostics.py"
breadth_diag_spec = importlib.util.spec_from_file_location("report_copiale_breadth_diagnostics", BREADTH_DIAG_PATH)
assert breadth_diag_spec is not None and breadth_diag_spec.loader is not None
copiale_breadth_diag = importlib.util.module_from_spec(breadth_diag_spec)
sys.modules[breadth_diag_spec.name] = copiale_breadth_diag
breadth_diag_spec.loader.exec_module(copiale_breadth_diag)

BREADTH_CURVE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_copiale_breadth_curve.py"
breadth_curve_spec = importlib.util.spec_from_file_location("report_copiale_breadth_curve", BREADTH_CURVE_PATH)
assert breadth_curve_spec is not None and breadth_curve_spec.loader is not None
copiale_breadth_curve = importlib.util.module_from_spec(breadth_curve_spec)
sys.modules[breadth_curve_spec.name] = copiale_breadth_curve
breadth_curve_spec.loader.exec_module(copiale_breadth_curve)

MASK_STABILITY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_copiale_mask_stability.py"
mask_stability_spec = importlib.util.spec_from_file_location("report_copiale_mask_stability", MASK_STABILITY_PATH)
assert mask_stability_spec is not None and mask_stability_spec.loader is not None
copiale_mask_stability = importlib.util.module_from_spec(mask_stability_spec)
sys.modules[mask_stability_spec.name] = copiale_mask_stability
mask_stability_spec.loader.exec_module(copiale_mask_stability)

REPAIR_AGENDA_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_copiale_repair_agenda.py"
repair_agenda_spec = importlib.util.spec_from_file_location("report_copiale_repair_agenda", REPAIR_AGENDA_PATH)
assert repair_agenda_spec is not None and repair_agenda_spec.loader is not None
copiale_repair_agenda = importlib.util.module_from_spec(repair_agenda_spec)
sys.modules[repair_agenda_spec.name] = copiale_repair_agenda
repair_agenda_spec.loader.exec_module(copiale_repair_agenda)

REPAIR_VARIANTS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_copiale_repair_variants.py"
repair_variants_spec = importlib.util.spec_from_file_location("probe_copiale_repair_variants", REPAIR_VARIANTS_PATH)
assert repair_variants_spec is not None and repair_variants_spec.loader is not None
copiale_repair_variants = importlib.util.module_from_spec(repair_variants_spec)
sys.modules[repair_variants_spec.name] = copiale_repair_variants
repair_variants_spec.loader.exec_module(copiale_repair_variants)

TARGETED_REPAIR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_copiale_targeted_repair.py"
targeted_repair_spec = importlib.util.spec_from_file_location("run_copiale_targeted_repair", TARGETED_REPAIR_PATH)
assert targeted_repair_spec is not None and targeted_repair_spec.loader is not None
copiale_targeted_repair = importlib.util.module_from_spec(targeted_repair_spec)
sys.modules[targeted_repair_spec.name] = copiale_targeted_repair
targeted_repair_spec.loader.exec_module(copiale_targeted_repair)

WINDOW_REPAIR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_copiale_window_repair.py"
window_repair_spec = importlib.util.spec_from_file_location("probe_copiale_window_repair", WINDOW_REPAIR_PATH)
assert window_repair_spec is not None and window_repair_spec.loader is not None
copiale_window_repair = importlib.util.module_from_spec(window_repair_spec)
sys.modules[window_repair_spec.name] = copiale_window_repair
window_repair_spec.loader.exec_module(copiale_window_repair)

MULTIPAGE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_copiale_multipage_experiment.py"
multipage_spec = importlib.util.spec_from_file_location("run_copiale_multipage_experiment", MULTIPAGE_PATH)
assert multipage_spec is not None and multipage_spec.loader is not None
copiale_multipage = importlib.util.module_from_spec(multipage_spec)
sys.modules[multipage_spec.name] = copiale_multipage
multipage_spec.loader.exec_module(copiale_multipage)

MULTIPAGE_SELECTOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_copiale_multipage_selector.py"
multipage_selector_spec = importlib.util.spec_from_file_location(
    "report_copiale_multipage_selector",
    MULTIPAGE_SELECTOR_PATH,
)
assert multipage_selector_spec is not None and multipage_selector_spec.loader is not None
copiale_multipage_selector = importlib.util.module_from_spec(multipage_selector_spec)
sys.modules[multipage_selector_spec.name] = copiale_multipage_selector
multipage_selector_spec.loader.exec_module(copiale_multipage_selector)

SELECTOR_ROBUSTNESS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_copiale_selector_robustness.py"
selector_robustness_spec = importlib.util.spec_from_file_location(
    "report_copiale_selector_robustness",
    SELECTOR_ROBUSTNESS_PATH,
)
assert selector_robustness_spec is not None and selector_robustness_spec.loader is not None
copiale_selector_robustness = importlib.util.module_from_spec(selector_robustness_spec)
sys.modules[selector_robustness_spec.name] = copiale_selector_robustness
selector_robustness_spec.loader.exec_module(copiale_selector_robustness)

GLOBAL_REPAIR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_copiale_multipage_global_repair.py"
global_repair_spec = importlib.util.spec_from_file_location(
    "probe_copiale_multipage_global_repair",
    GLOBAL_REPAIR_PATH,
)
assert global_repair_spec is not None and global_repair_spec.loader is not None
copiale_global_repair = importlib.util.module_from_spec(global_repair_spec)
sys.modules[global_repair_spec.name] = copiale_global_repair
global_repair_spec.loader.exec_module(copiale_global_repair)


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


def test_breadth_experiment_aggregates_ranker_finalists_post_hoc(tmp_path):
    artifact = {
        "test_id": "case_001",
        "ground_truth": "ABCDEF",
        "steps": [
            {
                "name": "search_null_masks",
                "ranker": "language_quality",
                "selected": {
                    "mask": ["S004"],
                    "source": "selected",
                    "decryption": "ABCDFF",
                    "language_quality_raw_score": 0.95,
                    "language_quality_rank_score": 1.6,
                    "validation_score_v2": 0.4,
                },
                "top_finalists": [
                    {
                        "mask": ["S001"],
                        "source": "initial",
                        "decryption": "ABXDEF",
                        "language_quality_raw_score": 0.91,
                        "language_quality_rank_score": 1.5,
                        "validation_score_v2": 0.3,
                    },
                    {
                        "mask": ["S002"],
                        "source": "beam",
                        "decryption": "ABCDEF",
                        "language_quality_raw_score": 0.72,
                        "language_quality_rank_score": 1.0,
                        "validation_score_v2": 0.2,
                    },
                    {
                        "mask": ["S003"],
                        "source": "duplicate",
                        "decryption": "ABCDEF",
                        "language_quality_raw_score": 0.60,
                        "language_quality_rank_score": 0.8,
                    },
                ],
            }
        ],
    }
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    candidates = copiale_breadth.aggregate_candidates("case_001", [artifact_path])
    report = copiale_breadth.summarize_candidate_pool("case_001", candidates)

    assert len(candidates) == 3
    assert report["best_char_mask"] == ["S002"]
    assert report["best_char_accuracy"] == 1.0
    assert report["lq_pick_mask"] == ["S004"]
    assert report["best_char_lq_rank"] == 3
    assert report["lq_top_3_capture"] is True


def test_breadth_diagnostics_compares_lq_pick_to_best_candidate(tmp_path):
    artifact = {
        "test_id": "case_001",
        "ground_truth": "ABCDEF",
        "steps": [
            {
                "name": "search_null_masks",
                "ranker": "language_quality",
                "selected": {
                    "mask": ["S001"],
                    "source": "initial",
                    "decryption": "ABXDEF",
                    "language_quality_raw_score": 0.91,
                    "language_quality_rank_score": 1.5,
                    "validation_score_v2": 0.3,
                    "ensemble_score_v1": 1.0,
                    "language_quality_features": {"dict_rate": 0.8, "deletion_control": 1.0},
                    "validation_components_v2": {"dictionary": 1.0},
                },
                "top_finalists": [
                    {
                        "mask": ["S001"],
                        "source": "initial",
                        "decryption": "ABXDEF",
                        "language_quality_raw_score": 0.91,
                        "language_quality_rank_score": 1.5,
                        "validation_score_v2": 0.3,
                        "ensemble_score_v1": 1.0,
                        "language_quality_features": {"dict_rate": 0.8, "deletion_control": 1.0},
                        "validation_components_v2": {"dictionary": 1.0},
                    },
                    {
                        "mask": ["S002"],
                        "source": "beam",
                        "decryption": "ABCDEF",
                        "language_quality_raw_score": 0.72,
                        "language_quality_rank_score": 1.0,
                        "validation_score_v2": 0.2,
                        "ensemble_score_v1": 1.2,
                        "language_quality_features": {"dict_rate": 0.7, "deletion_control": 0.5},
                        "validation_components_v2": {"dictionary": 0.8},
                    },
                ],
            }
        ],
    }
    artifact_path = tmp_path / "ranker" / "automated_only" / "case_001" / "run.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    artifacts = copiale_breadth_diag.discover_artifacts(tmp_path)
    candidates = copiale_breadth_diag.load_candidates("case_001", artifacts["case_001"])
    report = copiale_breadth_diag.analyze_test("case_001", candidates, model=None, top_n=4)

    assert report["best_char"]["mask"] == ["S002"]
    assert report["lq_pick"]["mask"] == ["S001"]
    assert report["best_lq_rank"] == 2
    assert report["lq_gap"] > 0.0
    assert report["feature_deltas_lq_minus_best"][0]["feature"] == "deletion_control"


def test_breadth_curve_prefers_explicit_candidate_index(tmp_path):
    artifact = {
        "test_id": "case_001",
        "ground_truth": "ABCDEF",
        "steps": [
            {
                "name": "search_null_masks",
                "ranker": "validation",
                "selected": {
                    "candidate_id": "initial:000005",
                    "evaluated_index": 5,
                    "mask": ["S001"],
                    "source": "initial",
                    "decryption": "ABXDEF",
                },
                "top_finalists": [
                    {
                        "candidate_id": "neighborhood:000042",
                        "evaluated_index": 42,
                        "mask": ["S002"],
                        "source": "neighborhood",
                        "decryption": "ABCDEF",
                    }
                ],
                "evaluated_rows": [
                    {
                        "candidate_id": "initial:000005",
                        "evaluated_index": 5,
                        "mask": ["S001"],
                    },
                    {
                        "candidate_id": "neighborhood:000042",
                        "evaluated_index": 42,
                        "mask": ["S002"],
                    },
                ],
            }
        ],
    }
    artifact_path = tmp_path / "validation" / "automated_only" / "case_001" / "run.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    finalists = copiale_breadth_curve.load_finalists(tmp_path, allowed_tests={"case_001"})
    payload = copiale_breadth_curve.analyze(finalists, prefixes=(5, 42))

    curve = payload["tests"][0]["curve"]
    assert payload["tests"][0]["best_overall"]["candidate_id"] == "neighborhood:000042"
    assert curve[0]["best_evaluated_index"] == 5
    assert curve[1]["best_evaluated_index"] == 42
    assert curve[1]["best_char_accuracy"] == 1.0


def test_mask_stability_groups_repeated_masks(tmp_path):
    artifact = {
        "test_id": "case_001",
        "ground_truth": "ABCDEF",
        "steps": [
            {
                "name": "search_null_masks",
                "ranker": "language_quality",
                "selected": {
                    "mask": ["S001"],
                    "source": "selected",
                    "decryption": "ABCDEF",
                    "language_quality_rank_score": 0.8,
                    "validation_score_v2": 0.7,
                    "ensemble_score_v1": 2.0,
                },
                "top_finalists": [
                    {
                        "mask": ["S001"],
                        "source": "neighborhood",
                        "decryption": "ABXDEF",
                        "language_quality_rank_score": 0.7,
                        "validation_score_v2": 0.6,
                        "ensemble_score_v1": 1.8,
                    },
                    {
                        "mask": ["S002"],
                        "source": "initial",
                        "decryption": "AXXXXX",
                        "language_quality_rank_score": 0.9,
                        "validation_score_v2": 0.9,
                        "ensemble_score_v1": 2.1,
                    },
                ],
            }
        ],
    }
    artifact_path = tmp_path / "language_quality" / "automated_only" / "case_001" / "run.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    candidates = copiale_mask_stability.load_candidates(tmp_path)
    report = copiale_mask_stability.analyze(candidates, top_n=4)

    assert report["summary"]["candidate_count"] == 3
    top = report["tests"][0]["top_masks"][0]
    assert top["mask"] == ["S001"]
    assert top["candidate_count"] == 2
    assert top["unique_text_count"] == 2
    assert top["best_char_accuracy"] == 1.0


def test_repair_agenda_marks_disputed_symbols_without_ground_truth(tmp_path):
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText

    cipher = CipherText(
        raw="S001 S002 S003 S001 S002 S003 S004",
        alphabet=Alphabet(["S001", "S002", "S003", "S004"]),
        separator=None,
    )
    artifact = tmp_path / "run.json"
    candidates = [
        copiale_repair_agenda.Candidate(
            rank=1,
            artifact=artifact,
            mask=(),
            source="selected",
            sort_score=3.0,
            validation_score_v2=0.3,
            language_quality_rank_score=0.6,
            ensemble_score_v1=2.0,
            selection_score=-7.0,
            decryption="WENWENI",
            key={0: 22, 1: 4, 2: 13, 3: 8},
            filtered_length=7,
            row={},
        ),
        copiale_repair_agenda.Candidate(
            rank=2,
            artifact=artifact,
            mask=(),
            source="candidate",
            sort_score=2.0,
            validation_score_v2=0.2,
            language_quality_rank_score=0.5,
            ensemble_score_v1=1.0,
            selection_score=-8.0,
            decryption="WETWETI",
            key={0: 22, 1: 4, 2: 19, 3: 8},
            filtered_length=7,
            row={},
        ),
    ]

    consensus = copiale_repair_agenda.consensus_assignments(
        candidates,
        cipher,
        min_agreement=0.75,
    )
    windows = copiale_repair_agenda.damaged_windows(
        candidates[0],
        cipher,
        consensus=consensus,
        language="de",
        window_size=6,
        step=3,
        limit=3,
    )

    assert consensus["S003"]["stable"] is False
    assert consensus["S003"]["counts"] == {"N": 1, "T": 1}
    assert any(
        symbol["symbol"] == "S003"
        for window in windows
        for symbol in window["disputed_symbols"]
    )


def test_repair_variant_edit_groups_prioritize_window_pressure():
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText

    cipher = CipherText(
        raw="S001 S002 S003 S002",
        alphabet=Alphabet(["S001", "S002", "S003"]),
        separator=None,
    )
    agenda = {
        "most_disputed_symbols": [
            {"symbol": "S001", "agreement": 0.5, "counts": {"A": 2, "B": 1}},
            {"symbol": "S002", "agreement": 0.5, "counts": {"C": 2, "D": 1}},
        ],
        "repair_windows": [
            {
                "disputed_symbols": [
                    {"symbol": "S002", "count": 4},
                    {"symbol": "S001", "count": 1},
                ]
            }
        ],
    }
    groups = copiale_repair_variants.build_edit_groups(
        agenda,
        cipher,
        baseline_key={0: 0, 1: 2, 2: 4},
        baseline_mask=(),
        max_symbols=2,
        max_alternatives=2,
    )

    assert groups[0]["symbol"] == "S002"
    assert groups[0]["current"] == "C"
    assert groups[0]["alternatives"] == ["D"]


def test_targeted_repair_freezes_only_stable_unmasked_symbols():
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText

    cipher = CipherText(
        raw="S001 S002 S003 S004",
        alphabet=Alphabet(["S001", "S002", "S003", "S004"]),
        separator=None,
    )

    fixed = copiale_targeted_repair.fixed_symbol_ids(
        cipher,
        baseline_key={0: 0, 1: 1, 2: 2, 3: 3},
        baseline_mask=("S004",),
        mutable_symbols={"S002"},
    )

    assert fixed == {0, 2}
    assert copiale_targeted_repair.parse_edit("S041:E->M") == ("S041", "E", "M")


def test_targeted_repair_can_pin_variant_edit(monkeypatch):
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText

    cipher = CipherText(
        raw="S001 S002",
        alphabet=Alphabet(["S001", "S002"]),
        separator=None,
    )
    captured = {}

    def fake_run_homophonic(cipher_text, language, budget, solver_profile, initial_key, fixed_cipher_ids):
        captured["fixed_cipher_ids"] = set(fixed_cipher_ids)
        return (
            "fake",
            dict(initial_key),
            "AB",
            {"anneal_score": 1.0, "selection_score": 1.0},
        )

    monkeypatch.setattr(copiale_targeted_repair, "_run_homophonic", fake_run_homophonic)
    monkeypatch.setattr(copiale_targeted_repair, "_plaintext_quality", lambda text, key: {"top_letter_fraction": 0.5})
    monkeypatch.setattr(
        copiale_targeted_repair,
        "_automated_candidate_diagnostics",
        lambda text, language, word_list: {"dict_rate": 0.0, "letter_count": len(text)},
    )

    copiale_targeted_repair.run_seed_variant(
        index=1,
        row={"edits": ["S002:B->C"]},
        cipher=cipher,
        baseline_key={0: 0, 1: 1},
        baseline_mask=(),
        stable_fixed_ids={0},
        budget="screen",
        pin_edits=True,
    )

    assert captured["fixed_cipher_ids"] == {0, 1}


def test_targeted_repair_result_deltas_use_baseline():
    rows = [
        {
            "edits": ["variant"],
            "validation_score_v2": 1.25,
            "validation_score_v2_no_selection": 2.5,
            "anneal_score": -8.0,
        },
        {
            "edits": ["baseline"],
            "validation_score_v2": 1.0,
            "validation_score_v2_no_selection": 2.0,
            "anneal_score": -7.5,
        },
    ]

    copiale_targeted_repair.attach_result_deltas(rows)

    assert rows[0]["delta_vs_baseline"]["validation_score_v2"] == 0.25
    assert rows[0]["delta_vs_baseline"]["validation_score_v2_no_selection"] == 0.5
    assert rows[0]["delta_vs_baseline"]["anneal_score"] == -0.5
    assert rows[1]["delta_vs_baseline"]["validation_score_v2"] == 0.0


def test_targeted_repair_edit_substring_filter():
    assert copiale_targeted_repair.edit_substring_matches(
        {"edits": ["S081:E-><null>"]},
        "<null>",
    )
    assert not copiale_targeted_repair.edit_substring_matches(
        {"edits": ["S041:E->M"]},
        "<null>",
    )


def test_window_repair_applies_edits_only_inside_window():
    text = "ABCB"
    sources = ["S001", "S002", "S003", "S002"]
    edit_set = (({"symbol": "S002"}, "D"),)

    repaired, changed, deleted = copiale_window_repair.apply_localized_edits(
        text,
        sources,
        edit_set,
        windows=[(1, 3)],
    )

    assert repaired == "ADCB"
    assert changed == 1
    assert deleted == 0


def test_window_repair_deletes_only_inside_window():
    text = "ABCB"
    sources = ["S001", "S002", "S003", "S002"]
    edit_set = (({"symbol": "S002"}, "<null>"),)

    repaired, changed, deleted = copiale_window_repair.apply_localized_edits(
        text,
        sources,
        edit_set,
        windows=[(1, 3)],
    )

    assert repaired == "ACB"
    assert changed == 0
    assert deleted == 1


def test_multipage_projection_uses_shared_symbol_key_and_page_masks():
    pages = [
        copiale_multipage.PageBundle(
            test_id="p1",
            canonical_transcription="S001 S002",
            plaintext="AB",
            symbols=["S001", "S002"],
            token_ids=[0, 1],
        ),
        copiale_multipage.PageBundle(
            test_id="p2",
            canonical_transcription="S002 S003 S001",
            plaintext="BCA",
            symbols=["S002", "S003", "S001"],
            token_ids=[1, 2, 0],
        ),
    ]

    rows = copiale_multipage.project_pages(
        pages=pages,
        key={0: 0, 1: 1, 2: 2},
        mask=("S002",),
    )

    assert rows[0]["decryption"] == "A"
    assert rows[1]["decryption"] == "CA"
    assert rows[1]["filtered_length"] == 2


def test_multipage_consensus_identifies_stable_shared_letters():
    alphabet = copiale_multipage.Alphabet(["S001", "S002"])
    artifact = {
        "steps": [
            {
                "name": "search_null_masks",
                "selected": {"key": {"0": 0, "1": 1}, "mask": []},
                "top_finalists": [
                    {"key": {"0": 0, "1": 2}, "mask": []},
                    {"key": {"0": 0, "1": 1}, "mask": ["S002"]},
                ],
            }
        ]
    }

    consensus = copiale_multipage.consensus_from_finalists(
        artifact=artifact,
        alphabet=alphabet,
        top_n=3,
        min_agreement=0.75,
    )

    assert consensus["S001"]["stable"] is True
    assert consensus["S001"]["winner"] == "A"
    assert consensus["S002"]["stable"] is False
    assert consensus["S002"]["counts"] == {"B": 1, "C": 1, "<null>": 1}


def test_multipage_finalist_rows_label_and_deduplicate_selected():
    artifact = {
        "steps": [
            {
                "name": "search_null_masks",
                "selected": {"key": {"0": 0}, "mask": ["S001"]},
                "top_finalists": [
                    {"key": {"0": 0}, "mask": ["S001"]},
                    {"key": {"0": 1}, "mask": []},
                ],
            }
        ]
    }

    rows = copiale_multipage.finalist_rows(artifact, top_n=8)

    assert [row["_label"] for row in rows] == ["selected", "top2"]
    assert rows[1]["key"] == {"0": 1}


def test_multipage_elite_portfolio_preserves_policy_and_balanced_candidates():
    rows = [
        {
            "label": "selected",
            "finalist_order": 0,
            "page_balanced_score": 10.0,
            "page_validation_avg": 10.0,
            "post_hoc_char_avg": 0.70,
            "post_hoc_page_chars": [],
            "mask": [],
            "policy_scores": {"a": 1.0, "b": 1.0},
        },
        {
            "label": "policy_a",
            "finalist_order": 1,
            "page_balanced_score": 9.0,
            "page_validation_avg": 9.0,
            "post_hoc_char_avg": 0.72,
            "post_hoc_page_chars": [],
            "mask": ["S001"],
            "policy_scores": {"a": 5.0, "b": 0.5},
        },
        {
            "label": "policy_b",
            "finalist_order": 2,
            "page_balanced_score": 8.0,
            "page_validation_avg": 8.0,
            "post_hoc_char_avg": 0.74,
            "post_hoc_page_chars": [],
            "mask": ["S002"],
            "policy_scores": {"a": 0.5, "b": 5.0},
        },
    ]

    portfolio = copiale_multipage.build_elite_portfolio(
        rows,
        ranked=rows,
        size=3,
    )

    assert [row["label"] for row in portfolio["rows"]] == [
        "selected",
        "policy_a",
        "policy_b",
    ]
    assert portfolio["captures_post_hoc_best"] is True
    assert portfolio["best_post_hoc_label_in_portfolio"] == "policy_b"


def test_multipage_compact_refined_pages_keeps_report_fields():
    rows = [
        {
            "test_id": "case_001",
            "status": "completed",
            "filtered_length": 10,
            "fixed_symbol_count": 5,
            "elapsed_seconds": 0.2,
            "char_accuracy": 0.8,
            "word_accuracy": 0.0,
            "preview": "ABC",
            "key": {"1": 2},
            "step": {"large": "omitted"},
        }
    ]

    compact = copiale_multipage.compact_refined_pages(rows)

    assert compact == [
        {
            "test_id": "case_001",
            "status": "completed",
            "filtered_length": 10,
            "fixed_symbol_count": 5,
            "elapsed_seconds": 0.2,
            "char_accuracy": 0.8,
            "word_accuracy": 0.0,
            "preview": "ABC",
        }
    ]


def test_multipage_local_edit_sets_use_disputed_consensus_only():
    alphabet = Alphabet(["S001", "S002"])
    consensus = {
        "S001": {
            "symbol": "S001",
            "token_id": 0,
            "winner": "A",
            "stable": False,
            "counts": {"A": 2, "B": 1, "<null>": 1},
        },
        "S002": {
            "symbol": "S002",
            "token_id": 1,
            "winner": "C",
            "stable": True,
            "counts": {"C": 4},
        },
    }
    windows = [
        {
            "start": 0,
            "end": 5,
            "disputed_symbols": [
                {"symbol": "S001", "count": 2},
                {"symbol": "S002", "count": 2},
            ],
        }
    ]

    edits = copiale_multipage.local_edit_sets(
        windows=windows,
        key={0: 0, 1: 2},
        mask=(),
        alphabet=alphabet,
        consensus=consensus,
        max_symbols=4,
        max_alternatives=4,
        include_pairs=False,
    )

    assert edits == [(("S001", "B"),), (("S001", "<null>"),)]


def test_multipage_apply_window_edits_keeps_global_text_outside_window():
    repaired, changed, deleted = copiale_multipage.apply_window_edits(
        "ABABA",
        ["S001", "S002", "S001", "S002", "S001"],
        (("S001", "C"),),
        {"start": 1, "end": 4},
    )

    assert repaired == "ABCBA"
    assert changed == 1
    assert deleted == 0


def test_multipage_local_repair_page_keeps_posthoc_fields(monkeypatch):
    alphabet = Alphabet(["S001", "S002"])
    page = copiale_multipage.PageBundle(
        test_id="case_001",
        canonical_transcription="S001 S002 S001 S002",
        plaintext="ABAB",
        symbols=["S001", "S002", "S001", "S002"],
        token_ids=[0, 1, 0, 1],
    )
    consensus = {
        "S001": {
            "symbol": "S001",
            "token_id": 0,
            "winner": "C",
            "stable": False,
            "counts": {"C": 2, "A": 1},
        },
        "S002": {
            "symbol": "S002",
            "token_id": 1,
            "winner": "B",
            "stable": True,
            "counts": {"B": 3},
        },
    }
    monkeypatch.setattr(
        copiale_multipage,
        "damaged_windows_for_text",
        lambda **_kwargs: [
            {
                "start": 0,
                "end": 4,
                "damage_score": 0.5,
                "disputed_symbol_count": 1,
                "disputed_symbols": [{"symbol": "S001", "count": 2}],
            }
        ],
    )
    monkeypatch.setattr(
        copiale_multipage,
        "score_page_runtime",
        lambda row, key, mask: {
            "validation_score_v2": 10.0 if row["decryption"] == "CBCB" else 1.0,
            "language_quality_mean": 0.5,
            "dict_rate": 0.0,
        },
    )

    repaired = copiale_multipage.repair_page_locally(
        page=page,
        key={0: 0, 1: 1},
        mask=(),
        alphabet=alphabet,
        consensus=consensus,
        window_size=4,
        window_step=4,
        window_limit=1,
        max_symbols=2,
        max_alternatives=2,
        include_pairs=False,
        min_validation_delta=0.03,
    )

    assert repaired["decryption"] == "CBCB"
    assert repaired["selected_edits"] == ["S001:A->C"]
    assert repaired["candidate_count"] == 2


def test_multipage_local_repair_requires_validation_margin(monkeypatch):
    alphabet = Alphabet(["S001", "S002"])
    page = copiale_multipage.PageBundle(
        test_id="case_001",
        canonical_transcription="S001 S002 S001 S002",
        plaintext="ABAB",
        symbols=["S001", "S002", "S001", "S002"],
        token_ids=[0, 1, 0, 1],
    )
    consensus = {
        "S001": {
            "symbol": "S001",
            "token_id": 0,
            "winner": "C",
            "stable": False,
            "counts": {"C": 2, "A": 1},
        },
        "S002": {
            "symbol": "S002",
            "token_id": 1,
            "winner": "B",
            "stable": True,
            "counts": {"B": 3},
        },
    }
    monkeypatch.setattr(
        copiale_multipage,
        "damaged_windows_for_text",
        lambda **_kwargs: [
            {
                "start": 0,
                "end": 4,
                "damage_score": 0.5,
                "disputed_symbol_count": 1,
                "disputed_symbols": [{"symbol": "S001", "count": 2}],
            }
        ],
    )
    monkeypatch.setattr(
        copiale_multipage,
        "score_page_runtime",
        lambda row, key, mask: {
            "validation_score_v2": 1.01 if row["decryption"] == "CBCB" else 1.0,
            "language_quality_mean": 0.5,
            "dict_rate": 0.0,
        },
    )

    repaired = copiale_multipage.repair_page_locally(
        page=page,
        key={0: 0, 1: 1},
        mask=(),
        alphabet=alphabet,
        consensus=consensus,
        window_size=4,
        window_step=4,
        window_limit=1,
        max_symbols=2,
        max_alternatives=2,
        include_pairs=False,
        min_validation_delta=0.03,
    )

    assert repaired["decryption"] == "ABAB"
    assert repaired["selected_edits"] == ["baseline"]
    assert repaired["accepted_best_variant"] is False


def test_multipage_selector_diagnostics_explains_policy_miss():
    payload = {
        "experiment": "copiale_multipage_shared_key",
        "test_ids": ["p1", "p2"],
        "portfolio_local_repair": {
            "rank_policy": "score",
            "best_by_policy": {"label": "policy"},
            "best_by_post_hoc_char": {"label": "best"},
            "rows": [
                {
                    "label": "policy",
                    "mask": ["S001"],
                    "page_balanced_score": 5.0,
                    "page_validation_avg": 4.0,
                    "page_language_quality_avg": 0.8,
                    "changed_page_count": 1,
                    "post_hoc_char_avg": 0.70,
                    "post_hoc_page_chars": [
                        {"test_id": "p1", "char_accuracy": 0.75, "selected_edits": ["baseline"]},
                        {"test_id": "p2", "char_accuracy": 0.65, "selected_edits": ["S001:A->B"]},
                    ],
                    "page_runtime_scores": [
                        {
                            "validation_components_v2": {"dictionary": 1.0},
                            "language_quality_features": {"dict_rate": 0.8},
                            "diagnostics": {"pseudo_word_fraction": 0.2},
                        }
                    ],
                },
                {
                    "label": "best",
                    "mask": ["S002"],
                    "page_balanced_score": 4.5,
                    "page_validation_avg": 3.8,
                    "page_language_quality_avg": 0.7,
                    "changed_page_count": 0,
                    "post_hoc_char_avg": 0.80,
                    "post_hoc_page_chars": [
                        {"test_id": "p1", "char_accuracy": 0.82, "selected_edits": ["baseline"]},
                        {"test_id": "p2", "char_accuracy": 0.78, "selected_edits": ["baseline"]},
                    ],
                    "page_runtime_scores": [
                        {
                            "validation_components_v2": {"dictionary": 0.7},
                            "language_quality_features": {"dict_rate": 0.7},
                            "diagnostics": {"pseudo_word_fraction": 0.1},
                        }
                    ],
                },
            ],
        },
    }

    report = copiale_multipage_selector.analyze(payload, section="portfolio_local_repair")
    rendered = copiale_multipage_selector.render_markdown(report)

    assert report["post_hoc_gap"] == 0.1
    assert report["post_hoc_best_policy_rank"] == 2
    assert report["policy_winner_post_hoc_rank"] == 2
    assert report["page_deltas_policy_minus_posthoc"][0]["test_id"] == "p2"
    assert any(
        row["feature"] == "dictionary"
        for row in report["validation_component_deltas_policy_minus_posthoc"]
    )
    assert "Policy winner" in rendered
    assert "Post-hoc best" in rendered


def test_selector_robustness_compares_balanced_and_robust_policies(tmp_path):
    payload = {
        "experiment": "copiale_multipage_shared_key",
        "test_ids": ["p1", "p2"],
        "portfolio_local_repair": {
            "rank_policy": "robust",
            "rows": [
                {
                    "label": "balanced",
                    "mask": ["S001"],
                    "page_balanced_score": 10.0,
                    "page_robust_score": 3.0,
                    "fragment_illusion_penalty": 0.1,
                    "page_validation_avg": 4.0,
                    "post_hoc_char_avg": 0.70,
                    "post_hoc_page_chars": [],
                    "page_runtime_scores": [],
                },
                {
                    "label": "robust",
                    "mask": ["S002"],
                    "page_balanced_score": 9.0,
                    "page_robust_score": 8.0,
                    "fragment_illusion_penalty": 0.0,
                    "page_validation_avg": 3.0,
                    "post_hoc_char_avg": 0.82,
                    "post_hoc_page_chars": [],
                    "page_runtime_scores": [],
                },
            ],
        },
    }
    path = tmp_path / "copiale_multipage_case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = copiale_selector_robustness.analyze_directory(
        tmp_path,
        sections=["portfolio_local_repair"],
    )
    rendered = copiale_selector_robustness.render_markdown(report)

    case = report["cases"][0]
    summary = report["section_summaries"][0]
    assert case["balanced_winner"]["label"] == "balanced"
    assert case["robust_winner"]["label"] == "robust"
    assert case["robust_exact_hit"] is True
    assert case["robust_delta_vs_balanced"] == 0.12
    assert summary["robust_exact_hits"] == 1
    assert summary["robust_improvements_over_0_5pct"] == 1
    assert "Copiale Multi-Page Selector Robustness" in rendered


def test_selector_robustness_skips_missing_sections(tmp_path):
    payload = {
        "experiment": "copiale_multipage_shared_key",
        "elite_page_rerank": {
            "rows": [
                {
                    "label": "candidate",
                    "mask": [],
                    "page_balanced_score": 1.0,
                    "page_robust_score": 1.0,
                    "fragment_illusion_penalty": 0.0,
                    "page_validation_avg": 1.0,
                    "post_hoc_char_avg": 0.5,
                    "post_hoc_page_chars": [],
                    "page_runtime_scores": [],
                }
            ],
        },
    }
    path = tmp_path / "copiale_multipage_case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = copiale_selector_robustness.analyze_directory(
        tmp_path,
        sections=["elite_page_rerank", "portfolio_local_repair"],
    )

    assert report["case_count"] == 1
    assert report["section_summaries"][0]["case_count"] == 1
    assert report["section_summaries"][1]["case_count"] == 0


def test_multipage_global_repair_variants_apply_shared_key_edits():
    pages = [
        copiale_multipage.PageBundle(
            test_id="p1",
            canonical_transcription="S001 S002",
            plaintext="AB",
            symbols=["S001", "S002"],
            token_ids=[0, 1],
        ),
        copiale_multipage.PageBundle(
            test_id="p2",
            canonical_transcription="S002 S001",
            plaintext="BA",
            symbols=["S002", "S001"],
            token_ids=[1, 0],
        ),
    ]
    variants = copiale_global_repair.evaluate_global_variants(
        pages=pages,
        baseline_key={0: 0, 1: 1},
        baseline_mask=(),
        edit_groups=[
            {
                "symbol": "S002",
                "token_id": 1,
                "current": "B",
                "pressure": 2,
                "alternatives": ["C", "<null>"],
            }
        ],
        include_pairs=False,
    )

    by_edit = {";".join(row["edits"]): row for row in variants}
    assert by_edit["baseline"]["post_hoc_char_avg"] == 1.0
    assert by_edit["S002:B->C"]["post_hoc_page_chars"][0]["char_accuracy"] == 0.5
    assert by_edit["S002:B-><null>"]["mask"] == ["S002"]


def test_multipage_global_repair_prunes_pair_variants():
    pages = [
        copiale_multipage.PageBundle(
            test_id="p1",
            canonical_transcription="S001 S002",
            plaintext="AB",
            symbols=["S001", "S002"],
            token_ids=[0, 1],
        )
    ]

    variants = copiale_global_repair.evaluate_global_variants(
        pages=pages,
        baseline_key={0: 0, 1: 1, 2: 2},
        baseline_mask=(),
        edit_groups=[
            {
                "symbol": "S002",
                "token_id": 1,
                "current": "B",
                "pressure": 10,
                "alternatives": ["C", "D"],
                "counts": {"C": 5, "D": 4},
            },
            {
                "symbol": "S001",
                "token_id": 0,
                "current": "A",
                "pressure": 8,
                "alternatives": ["E"],
                "counts": {"E": 3},
            },
        ],
        include_pairs=True,
        pair_candidate_limit=3,
        max_pairs=10,
    )

    edits = {";".join(row["edits"]) for row in variants}
    assert len(variants) == 6
    assert "S002:B->C;S002:B->D" not in edits
    assert "S002:B->C;S001:A->E" in edits
    assert "S002:B->D;S001:A->E" in edits


def test_multipage_global_repair_acceptance_requires_multiple_runtime_gains():
    baseline = {
        "edits": ["baseline"],
        "page_robust_score": 4.0,
        "page_balanced_score": 4.0,
        "page_validation_avg": 3.0,
        "page_validation_min": 2.0,
        "fragment_illusion_penalty": 0.10,
        "page_language_quality_avg": 0.50,
    }
    good = {
        "edits": ["S001:A->B"],
        "page_robust_score": 4.04,
        "page_balanced_score": 4.01,
        "page_validation_avg": 3.01,
        "page_validation_min": 1.99,
        "fragment_illusion_penalty": 0.11,
        "page_language_quality_avg": 0.51,
    }
    tempting = {
        "edits": ["S002:B->C"],
        "page_robust_score": 4.04,
        "page_balanced_score": 3.99,
        "page_validation_avg": 3.02,
        "page_validation_min": 1.99,
        "fragment_illusion_penalty": 0.11,
        "page_language_quality_avg": 0.49,
    }
    pair = {
        "edits": ["S001:A->B", "S002:B->C"],
        "page_robust_score": 4.10,
        "page_balanced_score": 4.10,
        "page_validation_avg": 3.10,
        "page_validation_min": 2.10,
        "fragment_illusion_penalty": 0.08,
        "page_language_quality_avg": 0.60,
    }

    copiale_global_repair.annotate_acceptance(
        [baseline, good, tempting, pair],
        baseline=baseline,
        robust_margin=0.03,
        min_page_drop=0.02,
        max_illusion_increase=0.02,
    )

    assert baseline["repair_acceptance"]["decision"] == "baseline"
    assert good["repair_acceptance"]["accepted"] is True
    assert tempting["repair_acceptance"]["accepted"] is False
    assert any("balanced score regresses" in reason for reason in tempting["repair_acceptance"]["reasons"])
    assert pair["repair_acceptance"]["accepted"] is False
    assert any("multi-edit variant is review-only" in reason for reason in pair["repair_acceptance"]["reasons"])

    copiale_global_repair.annotate_acceptance(
        [pair],
        baseline=baseline,
        robust_margin=0.03,
        min_page_drop=0.02,
        max_illusion_increase=0.02,
        allow_pair_acceptance=True,
    )
    assert pair["repair_acceptance"]["accepted"] is True


def test_multipage_global_repair_evidence_marks_runtime_posthoc_split():
    baseline = {
        "edits": ["baseline"],
        "page_runtime_scores": [
            {
                "test_id": "p1",
                "validation_score_v2": 1.0,
                "language_quality_mean": 0.5,
                "dict_rate": 0.6,
                "diagnostics": {"pseudo_word_fraction": 0.1},
                "validation_components_v2": {
                    "binary_ngram_fit": 0.2,
                    "language_coherence": 0.3,
                    "language_shape": 0.4,
                },
            }
        ],
        "post_hoc_page_chars": [{"test_id": "p1", "char_accuracy": 0.8}],
        "page_previews": [{"test_id": "p1", "preview": "ABCDE"}],
    }
    row = {
        "edits": ["S001:A->B"],
        "page_runtime_scores": [
            {
                "test_id": "p1",
                "validation_score_v2": 1.1,
                "language_quality_mean": 0.49,
                "dict_rate": 0.58,
                "diagnostics": {"pseudo_word_fraction": 0.12},
                "validation_components_v2": {
                    "binary_ngram_fit": 0.25,
                    "language_coherence": 0.31,
                    "language_shape": 0.41,
                },
            }
        ],
        "post_hoc_page_chars": [{"test_id": "p1", "char_accuracy": 0.79}],
        "page_previews": [{"test_id": "p1", "preview": "ABXDE"}],
    }

    copiale_global_repair.annotate_repair_evidence([row], baseline=baseline)
    evidence = row["repair_evidence"]

    assert evidence["runtime_pages_improved"] == 1
    assert evidence["preview_pages_changed"] == 1
    assert evidence["runtime_suspicious_pages"] == 1
    assert evidence["calibration_suspicious_pages"] == 1
    page = evidence["pages"][0]
    assert page["changed_excerpt"]["changed"] is True
    assert "validation_up_without_lq_gain" in page["runtime_flags"]
    assert "runtime_up_posthoc_char_down" in page["calibration_flags"]


def test_null_candidate_selection_reserves_rare_anchors():
    diagnostics = {
        "token_count": 200,
        "null_codeword_candidates": [
            {"symbol": f"R{i}", "count": 1, "score": 0.60, "reasons": ["rare", "localized"]}
            for i in range(10)
        ],
        "homophone_families": [
            {"letter": "E", "token_count": 80, "symbols": [f"E{i}" for i in range(10)]},
            {"letter": "N", "token_count": 60, "symbols": [f"N{i}" for i in range(10)]},
        ],
    }

    selected = copiale_probe.select_null_candidate_symbols(diagnostics, limit=18)

    assert "R8" in selected
    assert len([symbol for symbol in selected if symbol.startswith("R")]) >= 9


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


def test_null_mask_validation_v2_uses_binary_ngram_as_tiebreaker():
    weak = {
        "mask": ["S001"],
        "filtered_length": 100,
        "selection_score": -8.0,
        "diagnostics": {
            "dict_rate": 0.55,
            "letter_count": 100,
            "segmentation_cost": 500,
            "binary_ngram_mean_log_prob": -15.8,
        },
        "quality": {"top_letter_fraction": 0.25, "unique_letters": 18},
        "preview": "DIEUNDDASEINERDENUNDDASEINERDENUNDDASEINER",
    }
    stronger = {
        **weak,
        "mask": ["S002"],
        "diagnostics": {
            **weak["diagnostics"],
            "binary_ngram_mean_log_prob": -11.2,
        },
    }

    weak_score = copiale_probe.null_mask_validation_score_v2(weak, original_length=105)
    stronger_score = copiale_probe.null_mask_validation_score_v2(stronger, original_length=105)

    assert stronger_score["score"] > weak_score["score"]
    assert (
        stronger_score["components"]["binary_ngram_fit"]
        > weak_score["components"]["binary_ngram_fit"]
    )


def test_null_mask_ensemble_can_demote_fragment_rich_damaged_basin():
    fragment_rich = {
        "status": "completed",
        "mask": ["fraggy"],
        "filtered_length": 598,
        "selection_score": -7.83,
        "diagnostics": {
            "dict_rate": 0.551,
            "letter_count": 598,
            "segmentation_cost": 2809.8,
            "pseudo_word_fraction": 0.449,
            "long_pseudo_word_fraction": 0.236,
            "short_word_fraction": 0.067,
            "binary_ngram_mean_log_prob": -11.84,
        },
        "quality": {"top_letter_fraction": 0.236, "unique_letters": 17},
        "preview": "TETODEDERSTERDENCHTERSIESSERNENDISIEDENIZURENNEN",
    }
    better_lattice = {
        "status": "completed",
        "mask": ["better"],
        "filtered_length": 599,
        "selection_score": -8.0,
        "diagnostics": {
            "dict_rate": 0.577,
            "letter_count": 599,
            "segmentation_cost": 2838.1,
            "pseudo_word_fraction": 0.423,
            "long_pseudo_word_fraction": 0.18,
            "short_word_fraction": 0.035,
            "binary_ngram_mean_log_prob": -11.81,
        },
        "quality": {"top_letter_fraction": 0.25, "unique_letters": 18},
        "preview": "TEESDATENSTERDESEMTERNSELIERTENDSLIEDERSAURETREN",
    }

    rows = [fragment_rich, better_lattice]
    copiale_probe.attach_null_mask_ensemble_scores(rows, original_length=600, language="de")
    ranked = sorted(rows, key=copiale_probe.null_mask_rank_key, reverse=True)

    assert ranked[0]["mask"] == ["better"]
    assert (
        fragment_rich["ensemble_features_v1"]["language_content"]
        > better_lattice["ensemble_features_v1"]["language_content"]
    )
    assert better_lattice["ensemble_score_v1"] > fragment_rich["ensemble_score_v1"]


def test_null_mask_language_quality_ranker_uses_raw_model_score():
    low_quality = {
        "status": "completed",
        "mask": ["low"],
        "filtered_length": 120,
        "selection_score": -7.0,
        "diagnostics": {"letter_count": 120, "dict_rate": 0.1},
        "quality": {"top_letter_fraction": 0.18, "unique_letters": 18},
        "preview": "ABCDEF" * 20,
    }
    high_quality = {
        "status": "completed",
        "mask": ["high"],
        "filtered_length": 120,
        "selection_score": -8.0,
        "diagnostics": {
            "letter_count": 120,
            "dict_rate": 0.7,
            "dictionary_content_word_count": 12,
            "dictionary_long_content_word_count": 4,
            "dictionary_content_word_fraction": 0.5,
            "dictionary_content_char_fraction": 0.7,
        },
        "quality": {"top_letter_fraction": 0.18, "unique_letters": 18},
        "preview": "WENIGSICHUNDARBEITDASSSEINERORDENBRUDERGEORDNET" * 3,
    }
    model = LinearLanguageQualityModel(
        language="de",
        feature_names=LANGUAGE_QUALITY_FEATURES,
        intercept=0.0,
        weights=tuple(1.0 if name == "dict_rate" else 0.0 for name in LANGUAGE_QUALITY_FEATURES),
        means=tuple(0.0 for _name in LANGUAGE_QUALITY_FEATURES),
        scales=tuple(1.0 for _name in LANGUAGE_QUALITY_FEATURES),
        training_summary={},
    )

    rows = [low_quality, high_quality]
    copiale_probe.attach_null_mask_ensemble_scores(
        rows,
        original_length=120,
        language="de",
        language_quality_model=model,
    )
    ranked = sorted(rows, key=copiale_probe.null_mask_language_quality_rank_key, reverse=True)

    assert ranked[0]["mask"] == ["high"]
    assert ranked[0]["language_quality_raw_score"] > ranked[1]["language_quality_raw_score"]


def test_null_mask_language_quality_ranker_tiebreaks_tiny_lq_gaps_with_ensemble():
    slightly_higher_lq = {
        "mask": ["smooth"],
        "language_quality_rank_score": 0.9584,
        "language_quality_raw_score": 0.70,
        "validation_score_v2": 0.70,
        "ensemble_score_v1": 3.0,
        "selection_score": -7.0,
    }
    stronger_ensemble = {
        "mask": ["rougher"],
        "language_quality_rank_score": 0.9575,
        "language_quality_raw_score": 0.61,
        "validation_score_v2": 0.76,
        "ensemble_score_v1": 3.3,
        "selection_score": -7.1,
    }

    ranked = sorted(
        [slightly_higher_lq, stronger_ensemble],
        key=copiale_probe.null_mask_language_quality_rank_key,
        reverse=True,
    )

    assert ranked[0]["mask"] == ["rougher"]


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
                "validation_text": "WENIGSICHUNDARBEITORDENBRUDERGEHEIMNISBEWEGEN",
            },
        ],
    }

    report = copiale_null_report.summarize_probe_payload(payload)
    rendered = copiale_null_report.render_markdown([report], top=2)

    assert report["has_all_rows"] is True
    assert report["best_by_validation"]["mask"] == ["S002"]
    assert (
        report["best_by_validation"]["diagnostics"]["dictionary_content_word_count"]
        >= 4
    )
    assert (
        report["best_by_validation"]["validation_components_v2"]["content_word_quality"]
        > 0
    )
    assert report["char_best_validation_rank"] == 1
    assert report["capture_by_validation_top_n"][1] is True
    assert report["capture_by_validation_top_n"][3] is True
    assert "validation exact-best hits" in rendered
    assert "validation top-3 captures" in rendered
    assert "Pseudo" in rendered
    assert "Lattice" in rendered
    assert "Content" in rendered
    assert "Binary" in rendered
    assert "Top scalar-validation components" in rendered
    assert "Top ensemble-only calibration ranks" in rendered
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
