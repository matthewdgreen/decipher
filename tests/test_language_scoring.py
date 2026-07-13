import importlib.util
import json
import sys
from pathlib import Path

import pytest

from analysis.language_scoring import (
    LANGUAGE_QUALITY_FEATURES,
    LinearLanguageQualityModel,
    _az,
    content_lattice_consistency_score,
    content_rhythm_control_score,
    content_word_metrics,
    content_word_quality_score,
    get_language_scoring_profile,
    language_coherence_score,
    language_evidence_dispersion_score,
    language_quality_feature_dict,
    language_quality_solver_evidence_features,
    language_window_stability_score,
    language_shape_score,
    short_fragment_concentration_score,
    train_gradient_boosted_language_quality_model,
    train_linear_language_quality_model,
    train_pairwise_language_quality_model,
    word_lattice_quality_score,
    word_island_template_penalty,
)


TRAIN_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_language_quality_scorer.py"
train_spec = importlib.util.spec_from_file_location("train_language_quality_scorer", TRAIN_SCRIPT_PATH)
assert train_spec is not None and train_spec.loader is not None
train_lq = importlib.util.module_from_spec(train_spec)
sys.modules[train_spec.name] = train_lq
train_spec.loader.exec_module(train_lq)

REPORT_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_language_candidate_ranker.py"
report_spec = importlib.util.spec_from_file_location("report_language_candidate_ranker", REPORT_SCRIPT_PATH)
assert report_spec is not None and report_spec.loader is not None
report_ranker = importlib.util.module_from_spec(report_spec)
sys.modules[report_spec.name] = report_ranker
report_spec.loader.exec_module(report_ranker)

FAILURE_REPORT_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_language_ranker_failure_family.py"
failure_report_spec = importlib.util.spec_from_file_location(
    "report_language_ranker_failure_family",
    FAILURE_REPORT_SCRIPT_PATH,
)
assert failure_report_spec is not None and failure_report_spec.loader is not None
failure_report = importlib.util.module_from_spec(failure_report_spec)
sys.modules[failure_report_spec.name] = failure_report
failure_report_spec.loader.exec_module(failure_report)


def test_language_profiles_are_selected_by_code():
    assert get_language_scoring_profile("de").name == "German"
    assert get_language_scoring_profile("en").name == "English"
    assert get_language_scoring_profile("la").name == "Latin"
    assert get_language_scoring_profile("unknown").name == "English"


def test_language_quality_az_folds_latin_diacritics():
    assert _az("für größere Brüder; déjà vu") == "FURGROSSEREBRUDERDEJAVU"


def test_language_shape_uses_profile_specific_anchors():
    german_text = (
        "WENIGSICHUNDARBEITDASSSEINERORDENBRUDERGEORDNET"
        "WENIGSICHUNDARBEITDASSSEINERORDENBRUDERGEORDNET"
    )
    english_text = (
        "THEANDTHATWITHHAVEINGTIONMENTTHEANDTHATWITHHAVEINGTIONMENT"
        "THEANDTHATWITHHAVEINGTIONMENT"
    )

    assert language_coherence_score(german_text, "de") > 0.5
    assert language_shape_score(german_text, "de") > language_shape_score(german_text, "en")
    assert language_shape_score(english_text, "en") > language_shape_score(english_text, "de")


def test_language_dispersion_and_short_fragment_concentration():
    spread = (
        "WENIGSICHUNDARBEITDASSSEINERORDENBRUDER"
        "GEORDNETUNDANFANGNICHTSEINERARBEITORDEN"
        "DASSWENIGSICHBRUDERGEORDNETANFANG"
    )
    clumped = "WENIGSICHUNDARBEITDASSSEINERORDENBRUDER" + "X" * 120
    fragment_soup = "DERDENDERDIEEINERDERDENDESDIEDERDENEINER" * 6

    assert language_evidence_dispersion_score(spread, "de") > language_evidence_dispersion_score(clumped, "de")
    assert short_fragment_concentration_score(fragment_soup, "de") > short_fragment_concentration_score(spread, "de")


def test_language_window_stability_rewards_distributed_evidence():
    stable = (
        "WENIGSICHUNDARBEITDASSSEINERORDENBRUDER"
        "GEORDNETUNDANFANGNICHTSEINERARBEITORDEN"
        "DASSWENIGSICHBRUDERGEORDNETANFANG"
        "UNDSEINERARBEITDASSORDENBRUDER"
    )
    island = "WENIGSICHUNDARBEITDASSSEINERORDENBRUDER" + "X" * 170

    assert language_window_stability_score(stable, "de") > language_window_stability_score(island, "de")


def test_word_lattice_quality_uses_generic_segmentation_diagnostics():
    weak = {
        "dict_rate": 0.45,
        "letter_count": 100,
        "segmentation_cost": 600,
        "pseudo_word_fraction": 0.55,
        "long_pseudo_word_fraction": 0.30,
        "short_word_fraction": 0.20,
    }
    stronger = {
        "dict_rate": 0.62,
        "letter_count": 100,
        "segmentation_cost": 280,
        "pseudo_word_fraction": 0.20,
        "long_pseudo_word_fraction": 0.05,
        "short_word_fraction": 0.08,
    }

    assert word_lattice_quality_score(stronger) > word_lattice_quality_score(weak)


def test_content_word_quality_rewards_non_stopword_dictionary_hits():
    word_set = {
        "DIE",
        "DER",
        "UND",
        "ARBEIT",
        "ORDEN",
        "BRUDER",
        "GEHEIMNIS",
        "BEWEGEN",
    }
    function_heavy = content_word_metrics(
        ["DIE", "DER", "UND", "DIE", "DER", "UND", "ARBEIT"],
        word_set,
        "de",
    )
    content_rich = content_word_metrics(
        ["DIE", "ARBEIT", "ORDEN", "BRUDER", "GEHEIMNIS", "BEWEGEN"],
        word_set,
        "de",
    )

    assert content_rich["dictionary_content_word_count"] > function_heavy["dictionary_content_word_count"]
    assert content_rich["dictionary_long_content_word_count"] > function_heavy["dictionary_long_content_word_count"]
    assert content_word_quality_score(content_rich) > content_word_quality_score(function_heavy)


def test_content_lattice_consistency_penalizes_contentless_clean_lattices():
    supported = content_lattice_consistency_score(lattice_quality=0.68, content_quality=0.62)
    unsupported = content_lattice_consistency_score(lattice_quality=0.68, content_quality=0.05)
    damaged_content = content_lattice_consistency_score(lattice_quality=0.42, content_quality=0.62)

    assert supported > unsupported
    assert damaged_content > unsupported


def test_content_rhythm_control_rewards_balanced_content_evidence():
    balanced = content_rhythm_control_score(
        content_quality=0.75,
        function_content_balance=0.90,
        function_overuse=0.15,
        short_fragment_concentration=0.10,
        short_word_control=0.95,
        binary_ngram_fit=0.70,
    )
    fragment_soup = content_rhythm_control_score(
        content_quality=0.45,
        function_content_balance=0.35,
        function_overuse=0.75,
        short_fragment_concentration=0.65,
        short_word_control=0.40,
        binary_ngram_fit=0.45,
    )

    assert balanced > fragment_soup


def test_word_island_template_penalty_targets_repetitive_content_soup():
    cleaner = {
        "dict_rate": 0.58,
        "pseudo_word_fraction": 0.30,
        "long_pseudo_word_fraction": 0.08,
        "dictionary_content_word_fraction": 0.30,
        "dictionary_content_char_fraction": 0.58,
        "dictionary_content_word_count": 8,
        "dictionary_long_content_word_count": 2,
    }
    island_soup = {
        "dict_rate": 0.61,
        "pseudo_word_fraction": 0.43,
        "long_pseudo_word_fraction": 0.24,
        "dictionary_content_word_fraction": 0.39,
        "dictionary_content_char_fraction": 0.72,
        "dictionary_content_word_count": 35,
        "dictionary_long_content_word_count": 9,
    }

    assert word_island_template_penalty(island_soup, repetition=1.0) > 0.5
    assert word_island_template_penalty(cleaner, repetition=0.4) < 0.25


def test_language_quality_feature_dict_is_bounded():
    features = language_quality_feature_dict(
        "WENIGSICHUNDARBEITDASSSEINERORDENBRUDERGEORDNET",
        diagnostics={
            "dict_rate": 0.7,
            "letter_count": 80,
            "segmentation_cost": 160,
            "pseudo_word_fraction": 0.1,
            "long_pseudo_word_fraction": 0.02,
            "short_word_fraction": 0.05,
            "dictionary_content_word_fraction": 0.4,
            "dictionary_content_char_fraction": 0.75,
            "dictionary_content_word_count": 8,
            "dictionary_long_content_word_count": 3,
            "unique_letters": 16,
            "top_letter_fraction": 0.18,
        },
        language="de",
        original_length=100,
        filtered_length=96,
        mask_size=2,
    )

    assert set(LANGUAGE_QUALITY_FEATURES) <= set(features)
    assert all(0.0 <= value <= 1.0 for value in features.values())
    assert features["content_word_quality"] > 0.5


def test_language_quality_solver_evidence_features_are_bounded_and_neutral():
    neutral = language_quality_solver_evidence_features({})
    stronger = language_quality_solver_evidence_features({
        "validation_score_v2": 0.4,
        "ensemble_score_v1": 2.8,
        "selection_score": -7.2,
    })

    assert neutral["solver_evidence_present"] == 0.0
    assert stronger["solver_evidence_present"] == 1.0
    assert stronger["validation_score_control"] > neutral["validation_score_control"]
    assert stronger["ensemble_score_control"] > neutral["ensemble_score_control"]
    assert stronger["selection_score_control"] > neutral["selection_score_control"]
    assert all(0.0 <= value <= 1.0 for value in stronger.values())


def test_train_linear_language_quality_model_ranks_cleaner_example(tmp_path):
    good_features = {name: 0.85 for name in LANGUAGE_QUALITY_FEATURES}
    bad_features = {name: 0.15 for name in LANGUAGE_QUALITY_FEATURES}
    examples = [
        {"features": good_features, "label": 0.95},
        {"features": bad_features, "label": 0.05},
        {"features": {name: 0.65 for name in LANGUAGE_QUALITY_FEATURES}, "label": 0.70},
        {"features": {name: 0.35 for name in LANGUAGE_QUALITY_FEATURES}, "label": 0.30},
    ]

    model = train_linear_language_quality_model(examples, language="de", l2=0.01)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = LinearLanguageQualityModel.load(path)

    assert loaded.training_summary["example_count"] == 4
    assert loaded.score_features(good_features) > loaded.score_features(bad_features)


def test_linear_language_quality_model_exposes_unclipped_score():
    features = {name: 1.0 for name in LANGUAGE_QUALITY_FEATURES}
    model = LinearLanguageQualityModel(
        language="de",
        feature_names=LANGUAGE_QUALITY_FEATURES,
        intercept=1.2,
        weights=tuple(0.1 for _ in LANGUAGE_QUALITY_FEATURES),
        means=tuple(0.0 for _ in LANGUAGE_QUALITY_FEATURES),
        scales=tuple(1.0 for _ in LANGUAGE_QUALITY_FEATURES),
        training_summary={},
    )

    assert model.raw_score_features(features) > 1.0
    assert model.score_features(features) == 1.0


def test_pairwise_language_quality_model_learns_within_group_rank():
    low = {name: 0.1 for name in LANGUAGE_QUALITY_FEATURES}
    mid = {name: 0.5 for name in LANGUAGE_QUALITY_FEATURES}
    high = {name: 0.9 for name in LANGUAGE_QUALITY_FEATURES}
    examples = [
        {"features": high, "label": 0.90, "group": "page_a"},
        {"features": mid, "label": 0.55, "group": "page_a"},
        {"features": low, "label": 0.15, "group": "page_a"},
        {"features": high, "label": 0.88, "group": "page_b"},
        {"features": mid, "label": 0.58, "group": "page_b"},
        {"features": low, "label": 0.20, "group": "page_b"},
    ]

    model = train_pairwise_language_quality_model(
        examples,
        language="de",
        l2=0.01,
        min_label_delta=0.05,
    )

    assert model.raw_score_features(high) > model.raw_score_features(mid)
    assert model.raw_score_features(mid) > model.raw_score_features(low)
    assert model.training_summary["objective"] == "pairwise_within_group_label_delta"
    assert model.training_summary["pair_count"] > 0


def test_pairwise_language_quality_model_can_constrain_weights_nonnegative():
    low = {name: 0.1 for name in LANGUAGE_QUALITY_FEATURES}
    mid = {name: 0.5 for name in LANGUAGE_QUALITY_FEATURES}
    high = {name: 0.9 for name in LANGUAGE_QUALITY_FEATURES}
    examples = [
        {"features": high, "label": 0.90, "group": "page_a"},
        {"features": mid, "label": 0.55, "group": "page_a"},
        {"features": low, "label": 0.15, "group": "page_a"},
        {"features": high, "label": 0.88, "group": "page_b"},
        {"features": mid, "label": 0.58, "group": "page_b"},
        {"features": low, "label": 0.20, "group": "page_b"},
    ]

    model = train_pairwise_language_quality_model(
        examples,
        language="de",
        l2=0.01,
        min_label_delta=0.05,
        nonnegative_weights=True,
        nonnegative_iterations=50,
    )

    assert all(weight >= 0.0 for weight in model.weights)
    assert model.raw_score_features(high) > model.raw_score_features(low)
    assert model.training_summary["nonnegative_weights"] is True


def test_pairwise_nonnegative_training_clips_unstable_weights():
    low = {name: 0.0 for name in LANGUAGE_QUALITY_FEATURES}
    high = {name: 1.0 for name in LANGUAGE_QUALITY_FEATURES}
    examples = [
        {"features": high, "label": 0.90, "group": "page_a"},
        {"features": low, "label": 0.10, "group": "page_a"},
        {"features": high, "label": 0.88, "group": "page_b"},
        {"features": low, "label": 0.12, "group": "page_b"},
    ]

    model = train_pairwise_language_quality_model(
        examples,
        language="de",
        l2=0.0,
        min_label_delta=0.01,
        nonnegative_weights=True,
        nonnegative_iterations=20,
        nonnegative_learning_rate=1_000.0,
    )

    assert all(0.0 <= weight <= 10.0 for weight in model.weights)
    assert abs(model.training_summary["training_pearson"]) <= 1.0


def test_gradient_boosted_language_quality_model_learns_simple_interaction():
    low = {name: 0.0 for name in LANGUAGE_QUALITY_FEATURES}
    high = {name: 0.0 for name in LANGUAGE_QUALITY_FEATURES}
    mixed = {name: 0.0 for name in LANGUAGE_QUALITY_FEATURES}
    high["dict_rate"] = 0.9
    high["repair_signal_consensus_control"] = 0.9
    mixed["dict_rate"] = 0.9
    mixed["repair_signal_consensus_control"] = 0.1
    examples = [
        {"features": low, "label": 0.15, "group": "a"},
        {"features": mixed, "label": 0.35, "group": "a"},
        {"features": high, "label": 0.90, "group": "a"},
        {"features": low, "label": 0.20, "group": "b"},
        {"features": mixed, "label": 0.40, "group": "b"},
        {"features": high, "label": 0.85, "group": "b"},
    ]

    model = train_gradient_boosted_language_quality_model(
        examples,
        language="de",
        feature_names=("dict_rate", "repair_signal_consensus_control"),
        n_estimators=25,
        max_depth=2,
        learning_rate=0.1,
        min_samples_leaf=1,
    )
    loaded = LinearLanguageQualityModel.from_dict(model.to_dict())

    assert loaded.raw_score_features(high) > loaded.raw_score_features(mixed) > loaded.raw_score_features(low)
    assert loaded.training_summary["objective"] == "gradient_boosted_regression_tree"


def test_global_repair_examples_map_runtime_features(tmp_path):
    payload = {
        "label": "top9",
        "top_variants": [
            {
                "edits": ["baseline"],
                "mask": [],
                "page_validation_avg": 3.0,
                "page_robust_score": 4.0,
                "page_balanced_score": 3.5,
                "page_dict_avg": 0.62,
                "page_content_char_avg": 0.70,
                "page_pseudo_word_avg": 0.10,
                "page_binary_component_avg": 0.55,
                "page_shape_component_avg": 0.60,
                "page_evidence_dispersion_avg": 0.65,
                "page_window_stability_avg": 0.66,
                "page_repetition_control_avg": 0.90,
                "page_content_word_quality_avg": 0.72,
                "page_language_coherence_avg": 0.61,
                "post_hoc_char_avg": 0.80,
                "preview": "WENIGSICHUNDARBEITDASSSEINERORDENBRUDER" * 2,
                "repair_evidence": {
                    "page_count": 5,
                    "runtime_suspicious_pages": 1,
                    "calibration_suspicious_pages": 2,
                    "preview_pages_changed": 5,
                },
            }
        ],
    }
    path = tmp_path / "case.global_repair.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    examples = train_lq.load_global_repair_examples(path, language="de")

    assert len(examples) == 1
    example = examples[0]
    assert example.label == 0.80
    assert example.source.startswith("global_repair:")
    assert example.features["solver_evidence_present"] == 1.0
    assert example.features["dict_rate"] == 0.62
    assert example.features["pseudo_word_control"] == 0.90
    assert example.features["template_island_control"] == 0.80
    assert example.features["function_overuse_control"] == 0.80
    assert example.features["mask_family_support_control"] == 1.0
    assert example.features["mask_family_dictionary_control"] == 0.62


def test_global_repair_examples_can_train_against_adjudication_no_target(tmp_path):
    payload = {
        "label": "top6",
        "top_variants": [
            {
                "edits": ["S001:A->E"],
                "post_hoc_char_avg": 0.90,
                "preview": "WENIGSICHUNDARBEITDASSSEINERORDENBRUDER" * 2,
                "repair_adjudication": {"adjudication_no_target_score": -4.0},
            },
            {
                "edits": ["S002:T->N"],
                "post_hoc_char_avg": 0.10,
                "preview": "WENIGSICHUNDARBEITDASSSEINERORDENBRUDER" * 2,
                "repair_adjudication": {"adjudication_no_target_score": 4.0},
            },
        ],
    }
    path = tmp_path / "case.word_hypothesis_repair.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    default_examples = train_lq.load_global_repair_examples(path, language="de")
    no_target_examples = train_lq.load_global_repair_examples(
        path,
        language="de",
        label_target="adjudication_no_target",
    )

    assert [example.label for example in default_examples] == [0.90, 0.10]
    assert [example.label for example in no_target_examples] == [0.0, 1.0]
    assert no_target_examples[0].metadata["raw_label"] == -4.0


def test_global_repair_examples_add_source_mask_family_features(tmp_path):
    base = {
        "source_experiment": "/tmp/source_a.json",
        "test_ids": ["p017", "p052"],
        "top_variants": [
            {
                "mask": ["S001"],
                "post_hoc_char_avg": 0.70,
                "page_validation_avg": 1.0,
                "page_balanced_score": 2.0,
                "page_dict_avg": 0.40,
                "page_binary_component_avg": 0.30,
                "page_robust_score": 3.0,
                "preview": "AAA",
            },
            {
                "mask": ["S002"],
                "post_hoc_char_avg": 0.80,
                "page_validation_avg": 3.0,
                "page_balanced_score": 4.0,
                "page_dict_avg": 0.60,
                "page_binary_component_avg": 0.50,
                "page_robust_score": 5.0,
                "preview": "BBB",
            },
        ],
    }
    sibling = {
        "source_experiment": "/tmp/source_a.json",
        "test_ids": ["p017", "p052"],
        "top_variants": [
            {
                "mask": ["S002"],
                "post_hoc_char_avg": 0.81,
                "page_validation_avg": 5.0,
                "page_balanced_score": 6.0,
                "page_dict_avg": 0.80,
                "page_binary_component_avg": 0.70,
                "page_robust_score": 7.0,
                "preview": "CCC",
            }
        ],
    }
    (tmp_path / "a.global_repair.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "b.global_repair.json").write_text(json.dumps(sibling), encoding="utf-8")

    examples = train_lq.load_global_repair_examples(tmp_path, language="de")
    by_mask = {
        tuple(example.metadata["mask"]): example
        for example in examples
    }

    assert by_mask[("S001",)].features["mask_family_support_control"] == 1 / 3
    assert by_mask[("S002",)].features["mask_family_support_control"] == 2 / 3
    assert by_mask[("S002",)].features["mask_family_dictionary_control"] == 0.7
    assert by_mask[("S002",)].features["mask_family_binary_control"] == 0.6


def test_global_repair_examples_add_edit_level_runtime_features(tmp_path):
    payload = {
        "label": "edit_probe",
        "top_variants": [
            {
                "edits": ["S001:A->E", "S002:T->N"],
                "mask": ["S001"],
                "post_hoc_char_avg": 0.72,
                "preview": "AAA",
                "repair_acceptance": {
                    "accepted": True,
                    "positive_signal_count": 3,
                },
                "repair_evidence": {
                    "page_count": 3,
                    "pages": [
                        {
                            "validation_delta": 0.04,
                            "language_quality_delta": 0.02,
                            "dict_rate_delta": 0.01,
                            "binary_ngram_fit_delta": 0.01,
                            "pseudo_word_fraction_delta": -0.01,
                        },
                        {
                            "validation_delta": 0.03,
                            "language_quality_delta": 0.01,
                            "dict_rate_delta": 0.00,
                            "binary_ngram_fit_delta": 0.02,
                            "pseudo_word_fraction_delta": -0.02,
                        },
                        {
                            "validation_delta": -0.01,
                            "language_quality_delta": 0.00,
                            "dict_rate_delta": 0.01,
                            "binary_ngram_fit_delta": -0.01,
                            "pseudo_word_fraction_delta": 0.00,
                        },
                    ],
                },
            }
        ],
    }
    path = tmp_path / "case.global_repair.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    example = train_lq.load_global_repair_examples(path, language="de")[0]

    assert example.features["repair_validation_delta_control"] > 0.5
    assert example.features["repair_runtime_page_agreement_control"] == pytest.approx(2 / 3)
    assert example.features["repair_signal_consensus_control"] > 0.75
    assert example.features["repair_delta_stability_control"] > 0.70
    assert example.features["repair_language_delta_control"] > 0.5
    assert example.features["repair_binary_delta_control"] > 0.5
    assert example.features["repair_dict_delta_control"] > 0.5
    assert example.features["repair_pseudo_delta_control"] > 0.5
    assert example.features["repair_correlated_gain_control"] > 0.7
    assert example.features["repair_window_change_rate_control"] == 0.5
    assert example.features["repair_page_signal_floor_control"] > 0.5
    assert example.features["repair_page_signal_range_control"] > 0.5
    assert example.features["repair_validation_range_control"] > 0.5
    assert example.features["repair_edit_count_control"] == 1.0
    assert example.features["repair_acceptance_control"] > 0.7


def test_global_repair_window_features_penalize_repetitive_changed_text(tmp_path):
    payload = {
        "label": "window_probe",
        "top_variants": [
            {
                "edits": ["S001:A->E"],
                "mask": ["S001"],
                "post_hoc_char_avg": 0.72,
                "preview": "AAA",
                "repair_evidence": {
                    "page_count": 2,
                    "pages": [
                        {
                            "validation_delta": 0.05,
                            "language_quality_delta": 0.02,
                            "dict_rate_delta": 0.01,
                            "binary_ngram_fit_delta": 0.02,
                            "pseudo_word_fraction_delta": -0.01,
                            "changed_excerpt": {
                                "changed": True,
                                "before": "ABABABABABABABABABAB",
                                "after": "WENIGSICHORDENBRUDER",
                            },
                        },
                        {
                            "validation_delta": 0.04,
                            "language_quality_delta": 0.01,
                            "dict_rate_delta": 0.01,
                            "binary_ngram_fit_delta": 0.01,
                            "pseudo_word_fraction_delta": -0.01,
                            "changed_excerpt": {
                                "changed": True,
                                "before": "DENDENDENDENDENDENDEN",
                                "after": "ARBEITGEORDNETSEINER",
                            },
                        },
                    ],
                },
            },
            {
                "edits": ["S002:T->N"],
                "mask": ["S002"],
                "post_hoc_char_avg": 0.70,
                "preview": "BBB",
                "repair_evidence": {
                    "page_count": 2,
                    "pages": [
                        {
                            "validation_delta": 0.05,
                            "language_quality_delta": 0.02,
                            "dict_rate_delta": 0.01,
                            "binary_ngram_fit_delta": 0.02,
                            "pseudo_word_fraction_delta": -0.01,
                            "changed_excerpt": {
                                "changed": True,
                                "before": "ARBEITGEORDNETSEINER",
                                "after": "EEEEEEEEEEEEEEEEEEEE",
                            },
                        },
                        {
                            "validation_delta": 0.04,
                            "language_quality_delta": 0.01,
                            "dict_rate_delta": 0.01,
                            "binary_ngram_fit_delta": 0.01,
                            "pseudo_word_fraction_delta": -0.01,
                            "changed_excerpt": {
                                "changed": True,
                                "before": "WENIGSICHORDENBRUDER",
                                "after": "DENDENDENDENDENDENDEN",
                            },
                        },
                    ],
                },
            },
        ],
    }
    path = tmp_path / "case.global_repair.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    good, repetitive = train_lq.load_global_repair_examples(path, language="de")

    assert good.features["repair_window_quality_control"] > repetitive.features["repair_window_quality_control"]
    assert good.features["repair_window_quality_delta_control"] > repetitive.features["repair_window_quality_delta_control"]
    assert good.features["repair_window_diversity_control"] > repetitive.features["repair_window_diversity_control"]
    assert good.features["repair_window_repetition_control"] > repetitive.features["repair_window_repetition_control"]
    assert good.features["repair_window_quality_floor_control"] > repetitive.features["repair_window_quality_floor_control"]
    assert good.features["repair_window_gain_agreement_control"] == 1.0


def test_cross_page_features_penalize_outlier_pages(tmp_path):
    stable = {
        "edits": ["S001:A->E"],
        "mask": ["S001"],
        "post_hoc_char_avg": 0.72,
        "preview": "AAA",
        "repair_evidence": {
            "page_count": 3,
            "pages": [
                {
                    "validation_delta": 0.03,
                    "language_quality_delta": 0.01,
                    "dict_rate_delta": 0.01,
                    "binary_ngram_fit_delta": 0.01,
                    "pseudo_word_fraction_delta": -0.01,
                    "changed_excerpt": {"changed": True, "before": "ABCDABCD", "after": "WENIGSICH"},
                },
                {
                    "validation_delta": 0.02,
                    "language_quality_delta": 0.01,
                    "dict_rate_delta": 0.01,
                    "binary_ngram_fit_delta": 0.01,
                    "pseudo_word_fraction_delta": -0.01,
                    "changed_excerpt": {"changed": True, "before": "EFGHEFGH", "after": "ORDENBRUDER"},
                },
                {
                    "validation_delta": 0.025,
                    "language_quality_delta": 0.00,
                    "dict_rate_delta": 0.01,
                    "binary_ngram_fit_delta": 0.01,
                    "pseudo_word_fraction_delta": -0.01,
                    "changed_excerpt": {"changed": True, "before": "IJKLIJKL", "after": "ARBEITSEINER"},
                },
            ],
        },
    }
    outlier = {
        "edits": ["S002:T->N"],
        "mask": ["S002"],
        "post_hoc_char_avg": 0.70,
        "preview": "BBB",
        "repair_evidence": {
            "page_count": 3,
            "pages": [
                {
                    "validation_delta": 0.05,
                    "language_quality_delta": 0.02,
                    "dict_rate_delta": 0.01,
                    "binary_ngram_fit_delta": 0.01,
                    "pseudo_word_fraction_delta": -0.01,
                    "changed_excerpt": {"changed": True, "before": "ABCDABCD", "after": "WENIGSICH"},
                },
                {
                    "validation_delta": 0.04,
                    "language_quality_delta": 0.02,
                    "dict_rate_delta": 0.01,
                    "binary_ngram_fit_delta": 0.01,
                    "pseudo_word_fraction_delta": -0.01,
                    "changed_excerpt": {"changed": True, "before": "EFGHEFGH", "after": "WENIGSICH"},
                },
                {
                    "validation_delta": -0.08,
                    "language_quality_delta": -0.03,
                    "dict_rate_delta": -0.02,
                    "binary_ngram_fit_delta": -0.02,
                    "pseudo_word_fraction_delta": 0.02,
                    "changed_excerpt": {"changed": True, "before": "IJKLIJKL", "after": "EEEEEEEE"},
                },
            ],
        },
    }
    path = tmp_path / "case.global_repair.json"
    path.write_text(json.dumps({"top_variants": [stable, outlier]}), encoding="utf-8")

    stable_example, outlier_example = train_lq.load_global_repair_examples(path, language="de")

    assert stable_example.features["repair_page_signal_floor_control"] > outlier_example.features["repair_page_signal_floor_control"]
    assert stable_example.features["repair_page_signal_range_control"] > outlier_example.features["repair_page_signal_range_control"]
    assert stable_example.features["repair_validation_range_control"] > outlier_example.features["repair_validation_range_control"]
    assert stable_example.features["repair_window_quality_range_control"] > outlier_example.features["repair_window_quality_range_control"]
    assert stable_example.features["repair_cross_page_edit_consistency_control"] > outlier_example.features["repair_cross_page_edit_consistency_control"]


def test_global_repair_features_ignore_posthoc_calibration_flags():
    base = {
        "page_dict_avg": 0.6,
        "page_content_word_quality_avg": 0.6,
        "page_content_char_avg": 0.6,
        "repair_evidence": {
            "page_count": 2,
            "runtime_suspicious_pages": 0,
            "calibration_suspicious_pages": 0,
            "preview_pages_changed": 1,
        },
    }
    flagged = {
        **base,
        "repair_evidence": {
            **base["repair_evidence"],
            "calibration_suspicious_pages": 2,
            "calibration_flags": ["post-hoc only"],
        },
    }

    base_features = train_lq.global_repair_feature_dict(base)
    flagged_features = train_lq.global_repair_feature_dict(flagged)

    assert flagged_features["function_overuse_control"] == base_features["function_overuse_control"]


def test_correlated_gain_control_penalizes_orphan_validation_gain():
    supported = train_lq.correlated_gain_control(
        validation_delta=0.05,
        signal_consensus=0.9,
        language_delta=0.02,
        binary_delta=0.02,
        dict_delta=0.01,
        pseudo_delta=-0.01,
    )
    orphan = train_lq.correlated_gain_control(
        validation_delta=0.05,
        signal_consensus=0.25,
        language_delta=-0.01,
        binary_delta=-0.01,
        dict_delta=-0.01,
        pseudo_delta=0.01,
    )

    assert supported > orphan
    assert orphan < 0.5


def test_no_solver_feature_set_excludes_runtime_repair_features():
    features = set(train_lq.feature_names_for_mode("no_solver"))

    assert "dict_rate" in features
    assert "repair_validation_delta_control" not in features
    assert "repair_window_quality_control" not in features
    assert "repair_page_signal_floor_control" not in features
    assert "mask_family_support_control" not in features


def test_global_repair_examples_are_not_deduped_by_shared_preview(tmp_path):
    examples = [
        train_lq.TrainingExample(
            text="SAMEPREVIEW",
            label=0.80,
            source="global_repair:case.json:top_variant:1",
            group="case",
            features={"dict_rate": 0.8},
            metadata={"kind": "global_repair_candidate"},
        ),
        train_lq.TrainingExample(
            text="SAMEPREVIEW",
            label=0.79,
            source="global_repair:case.json:top_variant:2",
            group="case",
            features={"dict_rate": 0.7},
            metadata={"kind": "global_repair_candidate"},
        ),
    ]

    assert len(train_lq.dedupe_examples(examples)) == 2


def test_candidate_ranker_reports_simple_policy_ranks():
    predictions = [
        {
            "source": "global_repair:a:top_variant:1",
            "group": "a",
            "label": 0.70,
            "raw_score": 0.10,
            "score": 0.10,
            "mask": ["S001"],
            "preview": "AAA",
            "features": {
                "validation_score_control": 0.10,
                "ensemble_score_control": 0.20,
                "selection_score_control": 0.30,
                "dict_rate": 0.40,
                "language_coherence": 0.50,
            },
            "metadata": {},
        },
        {
            "source": "global_repair:a:top_variant:2",
            "group": "a",
            "label": 0.90,
            "raw_score": 0.20,
            "score": 0.20,
            "mask": ["S002"],
            "preview": "BBB",
            "features": {
                "validation_score_control": 0.80,
                "ensemble_score_control": 0.70,
                "selection_score_control": 0.60,
                "dict_rate": 0.55,
                "language_coherence": 0.40,
            },
            "metadata": {},
        },
    ]

    model = type("M", (), {"feature_names": (), "training_summary": {}})()
    report = report_ranker.group_report("a", [], [], predictions, model=model)

    assert report["best_label_rank"] == 1
    assert report["top_predicted_label_gap"] == 0.0
    assert report["policy_ranks"]["validation"]["best_label_rank"] == 1
    assert report["policy_ranks"]["language_quality"]["best_label_rank"] == 2


def test_candidate_ranker_summary_includes_policy_baselines():
    groups = [
        {
            "status": "completed",
            "best_label_rank": 1,
            "policy_ranks": {
                "validation": {"best_label_rank": 3},
                "ensemble": {"best_label_rank": 2},
            },
        },
        {
            "status": "completed",
            "best_label_rank": 2,
            "policy_ranks": {
                "validation": {"best_label_rank": 1},
                "ensemble": {"best_label_rank": 4},
            },
        },
    ]

    summary = report_ranker.summarize_group_reports(groups)

    assert summary["mean_best_label_rank"] == 1.5
    assert summary["mean_top_predicted_label_gap"] == 0.0
    assert summary["top_predicted_within_005"] == 2
    assert summary["policy_summary"]["validation"]["mean_best_label_rank"] == 2.0
    assert summary["policy_summary"]["ensemble"]["top3_captures"] == 1


def test_candidate_ranker_can_cluster_holdout_by_source_metadata():
    example = train_lq.TrainingExample(
        text="TEXT",
        label=0.8,
        source="global_repair:case.json:top_variant:1",
        group="global_repair:case:top1",
        features={},
        metadata={
            "source_experiment": "/tmp/source_a.json",
            "source_artifact": "/tmp/source_a.artifact.json",
            "section": "elite_page_rerank",
            "label": "top6",
            "test_ids": ["p017", "p052"],
        },
    )

    assert report_ranker.evaluation_group(example, "group") == "global_repair:case:top1"
    assert report_ranker.evaluation_group(example, "source_experiment") == "/tmp/source_a.json"
    assert report_ranker.evaluation_group(example, "test_set") == "p017+p052"
    regrouped = report_ranker.regroup_examples([example], "source_experiment")
    assert regrouped[0].group == "/tmp/source_a.json"
    assert regrouped[0].source == example.source


def test_failure_family_report_indexes_global_repair_sources(tmp_path):
    payload = {
        "top_variants": [
            {
                "edits": ["baseline"],
                "mask": ["S001"],
                "post_hoc_char_avg": 0.8,
                "preview": "AAA",
            },
            {
                "edits": ["S002:A->B"],
                "mask": ["S001", "S002"],
                "post_hoc_char_avg": 0.82,
                "preview": "BBB",
            },
        ]
    }
    path = tmp_path / "case.global_repair.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    index = failure_report.build_candidate_index([tmp_path])

    source = "global_repair:case.global_repair.json:top_variant:2"
    assert index[source]["rank"] == 2
    assert index[source]["row"]["edits"] == ["S002:A->B"]


def test_failure_family_report_finds_group_by_substring():
    payload = {
        "groups": [
            {"group": "/tmp/source_a.json", "status": "completed"},
            {"group": "/tmp/source_b.json", "status": "completed"},
        ]
    }

    assert failure_report.find_group(payload, "source_b")["group"] == "/tmp/source_b.json"


def test_candidate_ranker_builds_mask_family_examples():
    rows = [
        train_lq.TrainingExample(
            text="AAA",
            label=0.70,
            source="row1",
            group="menu_a",
            features={"dict_rate": 0.4, "mask_family_dictionary_control": 0.6},
            metadata={"mask": ["S001"], "source_experiment": "source_a"},
        ),
        train_lq.TrainingExample(
            text="BBB",
            label=0.80,
            source="row2",
            group="menu_b",
            features={"dict_rate": 0.8, "mask_family_dictionary_control": 0.7},
            metadata={"mask": ["S001"], "source_experiment": "source_a"},
        ),
        train_lq.TrainingExample(
            text="CCC",
            label=0.75,
            source="row3",
            group="menu_b",
            features={"dict_rate": 0.5, "mask_family_dictionary_control": 0.9},
            metadata={"mask": ["S002"], "source_experiment": "source_a"},
        ),
    ]

    families = report_ranker.build_mask_family_examples(rows, "source_experiment")
    edit_rows = report_ranker.build_mask_family_edit_examples(rows, "source_experiment")
    by_mask = {tuple(item.metadata["mask"]): item for item in families}

    assert len(families) == 2
    assert by_mask[("S001",)].label == 0.80
    assert by_mask[("S001",)].features["dict_rate"] == pytest.approx(0.6)
    assert by_mask[("S001",)].features["mask_family_dictionary_control"] == 0.7
    assert by_mask[("S001",)].metadata["member_count"] == 2
    assert edit_rows[0].group == "source_a::S001"
    assert edit_rows[2].group == "source_a::S002"


def test_two_stage_mask_family_report_prioritizes_family_then_candidate():
    predictions = [
        {
            "source": "row1",
            "label": 0.70,
            "raw_score": 0.95,
            "score": 0.95,
            "mask": ["S001"],
            "features": {"family": 0.1},
        },
        {
            "source": "row2",
            "label": 0.90,
            "raw_score": 0.50,
            "score": 0.50,
            "mask": ["S002"],
            "features": {"family": 0.9},
        },
        {
            "source": "row3",
            "label": 0.80,
            "raw_score": 0.40,
            "score": 0.40,
            "mask": ["S002"],
            "features": {"family": 0.9},
        },
    ]

    class FamilyModel:
        feature_names = ("family",)

        def raw_score_features(self, features):
            return features.get("family", 0.0)

        def score_features(self, features):
            return self.raw_score_features(features)

    report = report_ranker.two_stage_mask_family_report(
        predictions,
        family_model=FamilyModel(),
        candidate_model=FamilyModel(),
        family_top_k=1,
        review_shortlist_k=2,
    )

    assert report["best_label_rank"] == 1
    assert report["best_family_rank"] == 1
    assert report["top_predicted"]["mask"] == ["S002"]
    assert report["top_family"]["mask"] == ["S002"]
    assert report["review_shortlist_contains_best"] is True
    assert report["review_shortlist_best_label_gap"] == 0.0


def test_two_stage_mask_family_can_shortlist_multiple_families():
    predictions = [
        {
            "source": "row1",
            "label": 0.70,
            "raw_score": 0.0,
            "score": 0.0,
            "mask": ["S001"],
            "features": {"family": 1.0, "edit": 0.2},
        },
        {
            "source": "row2",
            "label": 0.95,
            "raw_score": 0.0,
            "score": 0.0,
            "mask": ["S002"],
            "features": {"family": 0.9, "edit": 0.9},
        },
        {
            "source": "row3",
            "label": 0.80,
            "raw_score": 0.0,
            "score": 0.0,
            "mask": ["S003"],
            "features": {"family": 0.1, "edit": 1.0},
        },
    ]

    class FamilyModel:
        feature_names = ("family",)

        def raw_score_features(self, features):
            return features.get("family", 0.0)

        def score_features(self, features):
            return self.raw_score_features(features)

    class EditModel:
        feature_names = ("edit",)

        def raw_score_features(self, features):
            return features.get("edit", 0.0)

        def score_features(self, features):
            return self.raw_score_features(features)

    top1 = report_ranker.two_stage_mask_family_report(
        predictions,
        family_model=FamilyModel(),
        candidate_model=EditModel(),
        family_top_k=1,
    )
    top2 = report_ranker.two_stage_mask_family_report(
        predictions,
        family_model=FamilyModel(),
        candidate_model=EditModel(),
        family_top_k=2,
    )

    assert top1["top_predicted"]["mask"] == ["S001"]
    assert top2["top_predicted"]["mask"] == ["S002"]
    assert top2["best_label_rank"] == 1


def test_two_stage_review_shortlist_reports_best_available_candidate():
    predictions = [
        {
            "source": "row1",
            "label": 0.70,
            "raw_score": 0.0,
            "score": 0.0,
            "mask": ["S001"],
            "features": {"family": 1.0, "edit": 0.9},
        },
        {
            "source": "row2",
            "label": 0.90,
            "raw_score": 0.0,
            "score": 0.0,
            "mask": ["S001"],
            "features": {"family": 1.0, "edit": 0.8},
        },
        {
            "source": "row3",
            "label": 0.95,
            "raw_score": 0.0,
            "score": 0.0,
            "mask": ["S001"],
            "features": {"family": 1.0, "edit": 0.1},
        },
    ]

    class FamilyModel:
        feature_names = ("family",)

        def raw_score_features(self, features):
            return features.get("family", 0.0)

        def score_features(self, features):
            return self.raw_score_features(features)

    class EditModel:
        feature_names = ("edit",)

        def raw_score_features(self, features):
            return features.get("edit", 0.0)

        def score_features(self, features):
            return self.raw_score_features(features)

    report = report_ranker.two_stage_mask_family_report(
        predictions,
        family_model=FamilyModel(),
        candidate_model=EditModel(),
        family_top_k=1,
        review_shortlist_k=2,
    )

    assert report["best_label_rank"] == 3
    assert report["review_shortlist_contains_best"] is False
    assert report["review_shortlist_best"]["source"] == "row2"
    assert report["review_shortlist_best_label_gap"] == pytest.approx(0.05)
    assert report["diverse_review_shortlist_best"]["source"] == "row2"
    assert report["diverse_review_shortlist_best_label_gap"] == pytest.approx(0.05)


def test_two_stage_diverse_review_shortlist_includes_top_families():
    predictions = [
        {
            "source": "row1",
            "label": 0.70,
            "raw_score": 0.0,
            "score": 0.0,
            "mask": ["S001"],
            "features": {"family": 1.0, "edit": 1.0},
        },
        {
            "source": "row2",
            "label": 0.71,
            "raw_score": 0.0,
            "score": 0.0,
            "mask": ["S001"],
            "features": {"family": 1.0, "edit": 0.9},
        },
        {
            "source": "row3",
            "label": 0.95,
            "raw_score": 0.0,
            "score": 0.0,
            "mask": ["S002"],
            "features": {"family": 0.8, "edit": 0.1},
        },
    ]

    class FamilyModel:
        feature_names = ("family",)

        def raw_score_features(self, features):
            return features.get("family", 0.0)

        def score_features(self, features):
            return self.raw_score_features(features)

    class EditModel:
        feature_names = ("edit",)

        def raw_score_features(self, features):
            return features.get("edit", 0.0)

        def score_features(self, features):
            return self.raw_score_features(features)

    report = report_ranker.two_stage_mask_family_report(
        predictions,
        family_model=FamilyModel(),
        candidate_model=EditModel(),
        family_top_k=2,
        review_shortlist_k=2,
    )

    assert report["review_shortlist_contains_best"] is False
    assert report["diverse_review_shortlist_contains_best"] is True
    assert [row["source"] for row in report["diverse_review_shortlist"]] == ["row1", "row3"]


def test_two_stage_summary_includes_review_shortlist_metrics():
    groups = [
        {
            "status": "completed",
            "best_label_rank": 3,
            "two_stage_mask_family": {
                "status": "completed",
                "best_label_rank": 3,
                "best_family_rank": 1,
                "top_predicted_label_gap": 0.02,
                "review_shortlist_k": 5,
                "review_shortlist_contains_best": True,
                "review_shortlist_best_label_gap": 0.0,
                "diverse_review_shortlist_contains_best": True,
                "diverse_review_shortlist_best_label_gap": 0.0,
            },
        },
        {
            "status": "completed",
            "best_label_rank": 8,
            "two_stage_mask_family": {
                "status": "completed",
                "best_label_rank": 8,
                "best_family_rank": 2,
                "top_predicted_label_gap": 0.03,
                "review_shortlist_k": 5,
                "review_shortlist_contains_best": False,
                "review_shortlist_best_label_gap": 0.004,
                "diverse_review_shortlist_contains_best": True,
                "diverse_review_shortlist_best_label_gap": 0.0,
            },
        },
    ]

    summary = report_ranker.summarize_two_stage_reports(groups)

    assert summary["review_shortlist_k"] == 5
    assert summary["review_shortlist_contains_best"] == 1
    assert summary["mean_review_shortlist_best_label_gap"] == pytest.approx(0.002)
    assert summary["review_shortlist_within_005"] == 2
    assert summary["diverse_review_shortlist_contains_best"] == 2
    assert summary["mean_diverse_review_shortlist_best_label_gap"] == 0.0
