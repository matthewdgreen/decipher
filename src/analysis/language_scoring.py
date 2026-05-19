"""Language-aware plaintext quality signals for damaged no-boundary text.

The functions here are intentionally lightweight and ground-truth-free. They
score whether a candidate looks like a plausible basin for a target language,
not whether it is a finished decipherment. German/Copiale is the first concrete
profile, but the API is language-profile driven so other languages can add
anchors and function fragments without changing the solver loop.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LanguageScoringProfile:
    code: str
    name: str
    long_fragments: dict[str, float]
    function_fragments: dict[str, float]
    anchors: tuple[str, ...]
    function_overuse_fragments: tuple[str, ...]
    stop_words: tuple[str, ...] = ()
    coherence_expected_span: float = 95.0
    function_expected_span: float = 18.0
    concentration_ceiling: float = 0.32


GERMAN_PROFILE = LanguageScoringProfile(
    code="de",
    name="German",
    long_fragments={
        "WENIG": 1.2,
        "SICH": 1.0,
        "DASS": 1.0,
        "EINE": 0.9,
        "EINER": 1.1,
        "SEINE": 1.0,
        "SEINER": 1.1,
        "ARBEIT": 1.35,
        "BEWEG": 1.2,
        "GEORD": 1.2,
        "ANFANG": 1.2,
        "HEIM": 1.1,
        "BRUDER": 1.2,
        "ORDEN": 1.2,
        "NICHT": 1.0,
    },
    function_fragments={
        "UND": 0.35,
        "DER": 0.30,
        "DIE": 0.30,
        "DAS": 0.30,
        "DEN": 0.25,
        "DES": 0.25,
        "EIN": 0.22,
    },
    anchors=(
        "WENIG",
        "DASS",
        "EINE",
        "EINER",
        "SEINE",
        "SEINER",
        "NICHT",
        "ARBEIT",
        "BEWEG",
        "GEORD",
        "ANFANG",
        "HEIM",
        "ORDEN",
        "BRUDER",
        "GRAD",
        "LOGE",
        "MEISTER",
    ),
    function_overuse_fragments=("UND", "DER", "DIE", "DAS", "DEN", "DES", "EIN", "EINE", "ER"),
    stop_words=(
        "ABER",
        "ALS",
        "AM",
        "AN",
        "AUF",
        "AUS",
        "BEI",
        "DA",
        "DAS",
        "DEM",
        "DEN",
        "DER",
        "DES",
        "DIE",
        "DIES",
        "EIN",
        "EINE",
        "EINEM",
        "EINEN",
        "EINER",
        "EINES",
        "ER",
        "ES",
        "IST",
        "MIT",
        "NICHT",
        "SICH",
        "SIE",
        "UND",
        "VON",
        "ZU",
    ),
)


ENGLISH_PROFILE = LanguageScoringProfile(
    code="en",
    name="English",
    long_fragments={
        "THE": 0.8,
        "AND": 0.7,
        "THAT": 0.9,
        "WITH": 0.9,
        "HAVE": 0.8,
        "TION": 0.9,
        "ING": 0.8,
        "MENT": 0.8,
    },
    function_fragments={
        "THE": 0.35,
        "AND": 0.30,
        "ING": 0.22,
        "ION": 0.20,
        "ENT": 0.18,
        "HER": 0.16,
    },
    anchors=("THE", "AND", "THAT", "WITH", "TION", "ING", "MENT", "HAVE"),
    function_overuse_fragments=("THE", "AND", "ING", "ION", "ENT", "ER", "HE"),
    stop_words=("THE", "AND", "THAT", "WITH", "THIS", "HAVE", "HAS", "ARE", "WAS", "WERE", "FOR", "NOT"),
)


LATIN_PROFILE = LanguageScoringProfile(
    code="la",
    name="Latin",
    long_fragments={
        "QUE": 0.9,
        "EST": 0.8,
        "ENT": 0.7,
        "ERE": 0.7,
        "TION": 0.8,
        "BUS": 0.8,
        "MUS": 0.8,
        "UM": 0.5,
        "QUOD": 1.0,
    },
    function_fragments={
        "ET": 0.28,
        "IN": 0.22,
        "QU": 0.24,
        "US": 0.18,
        "UM": 0.18,
        "ER": 0.16,
    },
    anchors=("QUE", "EST", "ENT", "ERE", "BUS", "MUS", "QUOD", "TION"),
    function_overuse_fragments=("ET", "IN", "QU", "US", "UM", "ER", "IS"),
    stop_words=("ET", "IN", "EST", "QUOD", "NON", "SI", "AUTEM", "CUM", "UT", "AD"),
)


PROFILES = {
    profile.code: profile
    for profile in (GERMAN_PROFILE, ENGLISH_PROFILE, LATIN_PROFILE)
}


LANGUAGE_QUALITY_FEATURE_SCALE_FLOOR = 0.01


LANGUAGE_QUALITY_FEATURES: tuple[str, ...] = (
    "dict_rate",
    "word_lattice_quality",
    "content_word_quality",
    "content_lattice_consistency",
    "language_coherence",
    "language_shape",
    "language_evidence_dispersion",
    "function_content_balance",
    "content_rhythm_control",
    "language_window_stability",
    "binary_ngram_fit",
    "letter_diversity",
    "top_letter_control",
    "short_fragment_control",
    "pseudo_word_control",
    "long_pseudo_control",
    "short_word_control",
    "repetition_control",
    "template_island_control",
    "function_overuse_control",
    "deletion_control",
    "mask_size_control",
    "validation_score_control",
    "ensemble_score_control",
    "selection_score_control",
    "solver_evidence_present",
    "mask_family_support_control",
    "mask_family_validation_control",
    "mask_family_balanced_control",
    "mask_family_dictionary_control",
    "mask_family_binary_control",
    "mask_family_robust_control",
    "repair_validation_delta_control",
    "repair_min_validation_delta_control",
    "repair_runtime_page_agreement_control",
    "repair_signal_consensus_control",
    "repair_delta_stability_control",
    "repair_language_delta_control",
    "repair_binary_delta_control",
    "repair_dict_delta_control",
    "repair_pseudo_delta_control",
    "repair_correlated_gain_control",
    "repair_window_quality_control",
    "repair_window_quality_delta_control",
    "repair_window_diversity_control",
    "repair_window_repetition_control",
    "repair_window_change_rate_control",
    "repair_page_signal_floor_control",
    "repair_page_signal_range_control",
    "repair_validation_range_control",
    "repair_window_quality_floor_control",
    "repair_window_quality_range_control",
    "repair_window_gain_agreement_control",
    "repair_cross_page_edit_consistency_control",
    "repair_edit_count_control",
    "repair_acceptance_control",
)


@dataclass(frozen=True)
class LinearLanguageQualityModel:
    """Small transparent model for fast damaged-plaintext finalist scoring."""

    language: str
    feature_names: tuple[str, ...]
    intercept: float
    weights: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    training_summary: dict[str, Any]
    version: int = 1

    def raw_score_features(self, features: dict[str, Any]) -> float:
        value = self.intercept
        for idx, name in enumerate(self.feature_names):
            raw = _as_float(features.get(name))
            scale = self.scales[idx] or 1.0
            value += self.weights[idx] * ((raw - self.means[idx]) / scale)
        return value

    def score_features(self, features: dict[str, Any]) -> float:
        return max(0.0, min(1.0, self.raw_score_features(features)))

    def score_text(
        self,
        text: str,
        *,
        diagnostics: dict[str, Any] | None = None,
        original_length: int | None = None,
        filtered_length: int | None = None,
        mask_size: int = 0,
    ) -> float:
        return self.score_features(
            language_quality_feature_dict(
                text,
                diagnostics=diagnostics or {},
                language=self.language,
                original_length=original_length,
                filtered_length=filtered_length,
                mask_size=mask_size,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "type": "linear_language_quality",
            "language": self.language,
            "feature_names": list(self.feature_names),
            "intercept": self.intercept,
            "weights": list(self.weights),
            "means": list(self.means),
            "scales": list(self.scales),
            "training_summary": self.training_summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearLanguageQualityModel":
        if payload.get("type") == "gradient_boosted_language_quality":
            return GradientBoostedLanguageQualityModel.from_dict(payload)  # type: ignore[return-value]
        if payload.get("type") != "linear_language_quality":
            raise ValueError("not a linear_language_quality model payload")
        return cls(
            language=str(payload.get("language") or "en"),
            feature_names=tuple(str(name) for name in payload.get("feature_names") or ()),
            intercept=float(payload.get("intercept") or 0.0),
            weights=tuple(float(value) for value in payload.get("weights") or ()),
            means=tuple(float(value) for value in payload.get("means") or ()),
            scales=tuple(float(value) for value in payload.get("scales") or ()),
            training_summary=dict(payload.get("training_summary") or {}),
            version=int(payload.get("version") or 1),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearLanguageQualityModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class GradientBoostedLanguageQualityModel:
    """Tiny gradient-boosted regression-tree scorer.

    This intentionally implements only the small subset we need for finalist
    ranking: squared-error boosting, shallow numeric trees, JSON persistence,
    and the same scoring API as ``LinearLanguageQualityModel``. Keeping it
    local avoids making scikit-learn a required runtime dependency.
    """

    language: str
    feature_names: tuple[str, ...]
    initial_prediction: float
    learning_rate: float
    trees: tuple[dict[str, Any], ...]
    training_summary: dict[str, Any]
    version: int = 1

    def raw_score_features(self, features: dict[str, Any]) -> float:
        value = self.initial_prediction
        row = [float(features.get(name, 0.0) or 0.0) for name in self.feature_names]
        for tree in self.trees:
            value += self.learning_rate * _predict_tree(tree, row)
        return value

    def score_features(self, features: dict[str, Any]) -> float:
        return max(0.0, min(1.0, self.raw_score_features(features)))

    def score_text(
        self,
        text: str,
        *,
        diagnostics: dict[str, Any] | None = None,
        original_length: int | None = None,
        filtered_length: int | None = None,
        mask_size: int = 0,
    ) -> float:
        return self.score_features(
            language_quality_feature_dict(
                text,
                diagnostics=diagnostics or {},
                language=self.language,
                original_length=original_length,
                filtered_length=filtered_length,
                mask_size=mask_size,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "type": "gradient_boosted_language_quality",
            "language": self.language,
            "feature_names": list(self.feature_names),
            "initial_prediction": self.initial_prediction,
            "learning_rate": self.learning_rate,
            "trees": list(self.trees),
            "training_summary": self.training_summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GradientBoostedLanguageQualityModel":
        if payload.get("type") != "gradient_boosted_language_quality":
            raise ValueError("not a gradient_boosted_language_quality model payload")
        return cls(
            language=str(payload.get("language") or "en"),
            feature_names=tuple(str(name) for name in payload.get("feature_names") or ()),
            initial_prediction=float(payload.get("initial_prediction") or 0.0),
            learning_rate=float(payload.get("learning_rate") or 0.1),
            trees=tuple(dict(tree) for tree in payload.get("trees") or ()),
            training_summary=dict(payload.get("training_summary") or {}),
            version=int(payload.get("version") or 1),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GradientBoostedLanguageQualityModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def get_language_scoring_profile(language: str | None) -> LanguageScoringProfile:
    """Return a profile for ``language``, falling back to English."""
    code = (language or "en").strip().lower()
    return PROFILES.get(code, ENGLISH_PROFILE)


def language_quality_feature_dict(
    text: str,
    *,
    diagnostics: dict[str, Any] | None = None,
    language: str | None = None,
    original_length: int | None = None,
    filtered_length: int | None = None,
    mask_size: int = 0,
) -> dict[str, float]:
    """Return bounded features for trained fast readability scoring.

    These features are all available without benchmark ground truth. Training
    scripts may use labels offline, but solver-time scoring should only consume
    the text/diagnostic features plus a frozen model.
    """
    diagnostics = diagnostics or {}
    cleaned = _az(text)
    counts: dict[str, int] = {}
    for ch in cleaned:
        counts[ch] = counts.get(ch, 0) + 1
    letter_count = max(1.0, _as_float(diagnostics.get("letter_count")) or float(len(cleaned) or 1))
    unique_letters = _as_float(diagnostics.get("unique_letters")) or float(len(counts))
    top_letter_fraction = _as_float(diagnostics.get("top_letter_fraction"))
    if top_letter_fraction <= 0.0 and counts:
        top_letter_fraction = max(counts.values()) / max(1, len(cleaned))
    pseudo_word_fraction = _as_float(diagnostics.get("pseudo_word_fraction"))
    long_pseudo_word_fraction = _as_float(diagnostics.get("long_pseudo_word_fraction"))
    short_word_fraction = _as_float(diagnostics.get("short_word_fraction"))
    repetition = repetitive_word_island_penalty(cleaned)
    template_islands = word_island_template_penalty(diagnostics, repetition=repetition)
    overuse = function_overuse_penalty(cleaned, language)
    short_fragment_concentration = short_fragment_concentration_score(cleaned, language)
    function_content_balance = function_content_balance_score(cleaned, language)
    binary_fit = binary_ngram_fit_score(diagnostics.get("binary_ngram_mean_log_prob"))
    if original_length and filtered_length is not None:
        deletion_fraction = max(0.0, (int(original_length) - int(filtered_length)) / max(1, int(original_length)))
    else:
        deletion_fraction = 0.0
    lattice_quality = word_lattice_quality_score(diagnostics)
    content_quality = content_word_quality_score(diagnostics)
    short_word_control = max(0.0, min(1.0, 1.0 - max(0.0, short_word_fraction - 0.10) * 3.0))
    return {
        "dict_rate": max(0.0, min(1.0, _as_float(diagnostics.get("dict_rate")))),
        "word_lattice_quality": lattice_quality,
        "content_word_quality": content_quality,
        "content_lattice_consistency": content_lattice_consistency_score(lattice_quality, content_quality),
        "language_coherence": language_coherence_score(cleaned, language),
        "language_shape": language_shape_score(cleaned, language),
        "language_evidence_dispersion": language_evidence_dispersion_score(cleaned, language),
        "function_content_balance": function_content_balance,
        "content_rhythm_control": content_rhythm_control_score(
            content_quality=content_quality,
            function_content_balance=function_content_balance,
            function_overuse=overuse,
            short_fragment_concentration=short_fragment_concentration,
            short_word_control=short_word_control,
            binary_ngram_fit=binary_fit,
        ),
        "language_window_stability": language_window_stability_score(cleaned, language),
        "binary_ngram_fit": binary_fit,
        "letter_diversity": max(0.0, min(1.0, unique_letters / 20.0)),
        "top_letter_control": max(0.0, min(1.0, 1.0 - max(0.0, top_letter_fraction - 0.18) / 0.18)),
        "short_fragment_control": max(0.0, min(1.0, 1.0 - short_fragment_concentration)),
        "pseudo_word_control": max(0.0, min(1.0, 1.0 - pseudo_word_fraction)),
        "long_pseudo_control": max(0.0, min(1.0, 1.0 - long_pseudo_word_fraction * 2.2)),
        "short_word_control": short_word_control,
        "repetition_control": max(0.0, min(1.0, 1.0 - repetition)),
        "template_island_control": max(0.0, min(1.0, 1.0 - template_islands)),
        "function_overuse_control": max(0.0, min(1.0, 1.0 - overuse)),
        "deletion_control": max(0.0, min(1.0, 1.0 - max(0.0, deletion_fraction - 0.04) / 0.18)),
        "mask_size_control": max(0.0, min(1.0, 1.0 - max(0, int(mask_size) - 2) / 4.0)),
        "validation_score_control": 0.5,
        "ensemble_score_control": 0.5,
        "selection_score_control": 0.5,
        "solver_evidence_present": 0.0,
        "mask_family_support_control": 0.5,
        "mask_family_validation_control": 0.5,
        "mask_family_balanced_control": 0.5,
        "mask_family_dictionary_control": 0.5,
        "mask_family_binary_control": 0.5,
        "mask_family_robust_control": 0.5,
        "repair_validation_delta_control": 0.5,
        "repair_min_validation_delta_control": 0.5,
        "repair_runtime_page_agreement_control": 0.5,
        "repair_signal_consensus_control": 0.5,
        "repair_delta_stability_control": 0.5,
        "repair_language_delta_control": 0.5,
        "repair_binary_delta_control": 0.5,
        "repair_dict_delta_control": 0.5,
        "repair_pseudo_delta_control": 0.5,
        "repair_correlated_gain_control": 0.5,
        "repair_window_quality_control": 0.5,
        "repair_window_quality_delta_control": 0.5,
        "repair_window_diversity_control": 0.5,
        "repair_window_repetition_control": 0.5,
        "repair_window_change_rate_control": 0.5,
        "repair_page_signal_floor_control": 0.5,
        "repair_page_signal_range_control": 0.5,
        "repair_validation_range_control": 0.5,
        "repair_window_quality_floor_control": 0.5,
        "repair_window_quality_range_control": 0.5,
        "repair_window_gain_agreement_control": 0.5,
        "repair_cross_page_edit_consistency_control": 0.5,
        "repair_edit_count_control": 0.5,
        "repair_acceptance_control": 0.5,
    }


def language_quality_features_from_row(
    row: dict[str, Any],
    *,
    language: str | None = None,
    original_length: int | None = None,
) -> dict[str, float]:
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
    text = str(row.get("decryption") or row.get("validation_text") or row.get("preview") or "")
    features = language_quality_feature_dict(
        text,
        diagnostics=diagnostics,
        language=language,
        original_length=original_length,
        filtered_length=int(row.get("filtered_length") or 0) or None,
        mask_size=len(row.get("mask") or []),
    )
    features.update(language_quality_solver_evidence_features(row))
    return features


def language_quality_solver_evidence_features(row: dict[str, Any]) -> dict[str, float]:
    """Return bounded, ground-truth-free solver/ranker evidence features."""
    validation = _maybe_float(row.get("validation_score_v2"))
    ensemble = _maybe_float(row.get("ensemble_score_v1"))
    selection = _maybe_float(row.get("selection_score"))
    present = any(value is not None for value in (validation, ensemble, selection))
    return {
        # Observed null-mask validation scores commonly span about -2.0..0.6.
        # Keep the transform deliberately loose; the linear model still learns
        # centering/scaling from calibration data.
        "validation_score_control": _bounded_linear(validation, low=-1.5, high=0.6, neutral=0.5),
        # Ensemble scores are usually 0..3-ish after pairwise voting.
        "ensemble_score_control": _bounded_linear(ensemble, low=0.0, high=3.2, neutral=0.5),
        # Selection scores are negative anneal-derived values; less negative is
        # usually better in the null-mask menu.
        "selection_score_control": _bounded_linear(selection, low=-10.0, high=-7.0, neutral=0.5),
        "solver_evidence_present": 1.0 if present else 0.0,
    }


def train_linear_language_quality_model(
    examples: list[dict[str, Any]],
    *,
    language: str,
    feature_names: tuple[str, ...] = LANGUAGE_QUALITY_FEATURES,
    l2: float = 0.1,
) -> LinearLanguageQualityModel:
    """Fit a tiny ridge-regression scorer from labeled examples.

    Labels should be in ``[0, 1]``. The model is intentionally simple and
    inspectable; it is a fast ranker for finalist menus, not an oracle.
    """
    if len(examples) < 2:
        raise ValueError("need at least two labeled examples to train a scorer")
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required to train language quality models") from exc

    matrix = []
    labels = []
    for example in examples:
        features = example.get("features")
        if not isinstance(features, dict):
            features = language_quality_feature_dict(
                str(example.get("text") or ""),
                diagnostics=example.get("diagnostics") if isinstance(example.get("diagnostics"), dict) else {},
                language=language,
                original_length=example.get("original_length"),
                filtered_length=example.get("filtered_length"),
                mask_size=int(example.get("mask_size") or 0),
            )
        matrix.append([float(features.get(name, 0.0) or 0.0) for name in feature_names])
        labels.append(max(0.0, min(1.0, float(example.get("label") or 0.0))))
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-6] = 1.0
    scales[(scales > 0.0) & (scales < LANGUAGE_QUALITY_FEATURE_SCALE_FLOOR)] = LANGUAGE_QUALITY_FEATURE_SCALE_FLOOR
    z = (x - means) / scales
    design = np.column_stack([np.ones(z.shape[0]), z])
    penalty = np.eye(design.shape[1]) * max(0.0, float(l2))
    penalty[0, 0] = 0.0
    coeffs = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    preds = np.clip(design @ coeffs, 0.0, 1.0)
    mae = float(np.mean(np.abs(preds - y)))
    corr = _pearson([float(v) for v in preds], [float(v) for v in y])
    summary = {
        "example_count": len(examples),
        "label_min": round(float(y.min()), 6),
        "label_max": round(float(y.max()), 6),
        "label_mean": round(float(y.mean()), 6),
        "training_mae": round(mae, 6),
        "training_pearson": round(corr, 6),
        "l2": float(l2),
    }
    return LinearLanguageQualityModel(
        language=language,
        feature_names=tuple(feature_names),
        intercept=float(coeffs[0]),
        weights=tuple(float(value) for value in coeffs[1:]),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        training_summary=summary,
    )


def train_pairwise_language_quality_model(
    examples: list[dict[str, Any]],
    *,
    language: str,
    feature_names: tuple[str, ...] = LANGUAGE_QUALITY_FEATURES,
    l2: float = 0.1,
    min_label_delta: float = 0.02,
    max_pairs_per_group: int = 2000,
    nonnegative_weights: bool = False,
    nonnegative_iterations: int = 4000,
    nonnegative_learning_rate: float = 0.01,
) -> LinearLanguageQualityModel:
    """Fit a transparent ranker from within-group candidate preferences.

    This is intended for finalist menus: labels may come from offline solved
    calibration artifacts, but runtime scoring consumes only feature values.
    The learned model is stored as the same ``LinearLanguageQualityModel`` so
    existing solver/ranker plumbing can load it unchanged.
    """
    if len(examples) < 2:
        raise ValueError("need at least two labeled examples to train a ranker")
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required to train language quality models") from exc

    rows: list[tuple[str, list[float], float]] = []
    for index, example in enumerate(examples):
        features = example.get("features")
        if not isinstance(features, dict):
            features = language_quality_feature_dict(
                str(example.get("text") or ""),
                diagnostics=example.get("diagnostics") if isinstance(example.get("diagnostics"), dict) else {},
                language=language,
                original_length=example.get("original_length"),
                filtered_length=example.get("filtered_length"),
                mask_size=int(example.get("mask_size") or 0),
            )
        label = max(0.0, min(1.0, float(example.get("label") or 0.0)))
        group = str(example.get("group") or "__all__")
        rows.append((group, [float(features.get(name, 0.0) or 0.0) for name in feature_names], label))
    x = np.asarray([row[1] for row in rows], dtype=float)
    y_abs = np.asarray([row[2] for row in rows], dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-6] = 1.0
    scales[(scales > 0.0) & (scales < LANGUAGE_QUALITY_FEATURE_SCALE_FLOOR)] = LANGUAGE_QUALITY_FEATURE_SCALE_FLOOR
    z = (x - means) / scales

    by_group: dict[str, list[int]] = {}
    for idx, (group, _features, _label) in enumerate(rows):
        by_group.setdefault(group, []).append(idx)
    pair_vectors = []
    pair_targets = []
    for indices in by_group.values():
        pair_count = 0
        sorted_indices = sorted(indices, key=lambda idx: y_abs[idx], reverse=True)
        for left_pos, left in enumerate(sorted_indices):
            for right in sorted_indices[left_pos + 1:]:
                delta = float(y_abs[left] - y_abs[right])
                if delta < min_label_delta:
                    continue
                pair_vectors.append(z[left] - z[right])
                pair_targets.append(delta)
                pair_count += 1
                if pair_count >= max_pairs_per_group:
                    break
            if pair_count >= max_pairs_per_group:
                break
    if not pair_vectors:
        raise ValueError("no within-group label-separated pairs available")
    d = np.asarray(pair_vectors, dtype=float)
    targets = np.asarray(pair_targets, dtype=float)
    penalty = np.eye(d.shape[1]) * max(0.0, float(l2))
    try:
        weights = np.linalg.solve(d.T @ d + penalty, d.T @ targets)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(d.T @ d + penalty, d.T @ targets, rcond=None)[0]
    if nonnegative_weights:
        weights = _fit_nonnegative_pairwise_weights(
            d,
            targets,
            initial=weights,
            l2=float(l2),
            iterations=nonnegative_iterations,
            learning_rate=nonnegative_learning_rate,
        )
    intercept = float(np.mean(y_abs - z @ weights))
    preds_raw = intercept + z @ weights
    preds = np.clip(preds_raw, 0.0, 1.0)
    mae = float(np.mean(np.abs(preds - y_abs)))
    corr = _pearson([float(v) for v in preds_raw], [float(v) for v in y_abs])
    ranking = _ranking_summary_from_scores(
        [
            {
                "group": rows[idx][0],
                "label": float(y_abs[idx]),
                "score": float(preds_raw[idx]),
            }
            for idx in range(len(rows))
        ]
    )
    summary = {
        "example_count": len(examples),
        "pair_count": len(pair_vectors),
        "group_count": len(by_group),
        "label_min": round(float(y_abs.min()), 6),
        "label_max": round(float(y_abs.max()), 6),
        "label_mean": round(float(y_abs.mean()), 6),
        "training_mae": round(mae, 6),
        "training_pearson": round(corr, 6),
        "mean_best_label_prediction_rank": _round_or_none(ranking.get("mean_best_label_prediction_rank")),
        "top3_captures": ranking.get("top3_captures"),
        "top5_captures": ranking.get("top5_captures"),
        "l2": float(l2),
        "min_label_delta": float(min_label_delta),
        "max_pairs_per_group": int(max_pairs_per_group),
        "nonnegative_weights": bool(nonnegative_weights),
        "nonnegative_iterations": int(nonnegative_iterations) if nonnegative_weights else 0,
        "nonnegative_learning_rate": float(nonnegative_learning_rate) if nonnegative_weights else 0.0,
        "objective": "pairwise_within_group_label_delta",
    }
    return LinearLanguageQualityModel(
        language=language,
        feature_names=tuple(feature_names),
        intercept=intercept,
        weights=tuple(float(value) for value in weights),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        training_summary=summary,
    )


def train_gradient_boosted_language_quality_model(
    examples: list[dict[str, Any]],
    *,
    language: str,
    feature_names: tuple[str, ...] = LANGUAGE_QUALITY_FEATURES,
    n_estimators: int = 75,
    max_depth: int = 3,
    learning_rate: float = 0.06,
    min_samples_leaf: int = 2,
) -> GradientBoostedLanguageQualityModel:
    """Fit a compact boosted-tree scorer from labeled examples.

    This is deliberately small and deterministic. It optimizes squared-error
    residuals and is mainly for finalist-menu ranking experiments where shallow
    feature interactions matter more than a globally linear signal.
    """
    if len(examples) < 4:
        raise ValueError("need at least four labeled examples to train a boosted scorer")
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required to train language quality models") from exc

    matrix = []
    labels = []
    groups = []
    for example in examples:
        features = example.get("features")
        if not isinstance(features, dict):
            features = language_quality_feature_dict(
                str(example.get("text") or ""),
                diagnostics=example.get("diagnostics") if isinstance(example.get("diagnostics"), dict) else {},
                language=language,
                original_length=example.get("original_length"),
                filtered_length=example.get("filtered_length"),
                mask_size=int(example.get("mask_size") or 0),
            )
        matrix.append([float(features.get(name, 0.0) or 0.0) for name in feature_names])
        labels.append(max(0.0, min(1.0, float(example.get("label") or 0.0))))
        groups.append(str(example.get("group") or "__all__"))

    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=float)
    prediction = np.full(y.shape, float(y.mean()), dtype=float)
    trees: list[dict[str, Any]] = []
    for _ in range(max(1, int(n_estimators))):
        residual = y - prediction
        tree = _fit_regression_tree(
            x,
            residual,
            max_depth=max(1, int(max_depth)),
            min_samples_leaf=max(1, int(min_samples_leaf)),
        )
        update = np.asarray([_predict_tree(tree, row.tolist()) for row in x], dtype=float)
        prediction += float(learning_rate) * update
        trees.append(tree)

    preds = np.clip(prediction, 0.0, 1.0)
    mae = float(np.mean(np.abs(preds - y)))
    corr = _pearson([float(v) for v in prediction], [float(v) for v in y])
    ranking = _ranking_summary_from_scores(
        [
            {"group": groups[idx], "label": float(y[idx]), "score": float(prediction[idx])}
            for idx in range(len(groups))
        ]
    )
    summary = {
        "example_count": len(examples),
        "tree_count": len(trees),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "min_samples_leaf": int(min_samples_leaf),
        "label_min": round(float(y.min()), 6),
        "label_max": round(float(y.max()), 6),
        "label_mean": round(float(y.mean()), 6),
        "training_mae": round(mae, 6),
        "training_pearson": round(corr, 6),
        "mean_best_label_prediction_rank": _round_or_none(ranking.get("mean_best_label_prediction_rank")),
        "top3_captures": ranking.get("top3_captures"),
        "top5_captures": ranking.get("top5_captures"),
        "objective": "gradient_boosted_regression_tree",
    }
    return GradientBoostedLanguageQualityModel(
        language=language,
        feature_names=tuple(feature_names),
        initial_prediction=float(y.mean()),
        learning_rate=float(learning_rate),
        trees=tuple(trees),
        training_summary=summary,
    )


def _fit_regression_tree(
    x: Any,
    y: Any,
    *,
    max_depth: int,
    min_samples_leaf: int,
) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required to train language quality models") from exc

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    value = float(y_arr.mean()) if len(y_arr) else 0.0
    if max_depth <= 0 or len(y_arr) < max(2, min_samples_leaf * 2):
        return {"leaf": value}

    parent_error = float(np.sum((y_arr - value) ** 2))
    best: tuple[float, int, float, Any, Any] | None = None
    for feature_idx in range(x_arr.shape[1]):
        column = x_arr[:, feature_idx]
        for threshold in _candidate_thresholds(column):
            left_mask = column <= threshold
            right_mask = ~left_mask
            left_count = int(np.sum(left_mask))
            right_count = int(np.sum(right_mask))
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue
            left_y = y_arr[left_mask]
            right_y = y_arr[right_mask]
            left_mean = float(left_y.mean())
            right_mean = float(right_y.mean())
            error = float(np.sum((left_y - left_mean) ** 2) + np.sum((right_y - right_mean) ** 2))
            gain = parent_error - error
            if best is None or gain > best[0]:
                best = (gain, feature_idx, float(threshold), left_mask, right_mask)
    if best is None or best[0] <= 1e-12:
        return {"leaf": value}
    _gain, feature_idx, threshold, left_mask, right_mask = best
    return {
        "feature": int(feature_idx),
        "threshold": float(threshold),
        "left": _fit_regression_tree(
            x_arr[left_mask],
            y_arr[left_mask],
            max_depth=max_depth - 1,
            min_samples_leaf=min_samples_leaf,
        ),
        "right": _fit_regression_tree(
            x_arr[right_mask],
            y_arr[right_mask],
            max_depth=max_depth - 1,
            min_samples_leaf=min_samples_leaf,
        ),
    }


def _candidate_thresholds(values: Any) -> list[float]:
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required to train language quality models") from exc

    unique = sorted({float(value) for value in values})
    if len(unique) <= 1:
        return []
    mids = [(left + right) / 2.0 for left, right in zip(unique, unique[1:])]
    if len(mids) <= 24:
        return mids
    quantiles = np.linspace(0, len(mids) - 1, 24)
    return [mids[int(round(q))] for q in quantiles]


def _predict_tree(tree: dict[str, Any], row: list[float]) -> float:
    node = tree
    while "leaf" not in node:
        feature = int(node.get("feature") or 0)
        threshold = float(node.get("threshold") or 0.0)
        node = node["left"] if row[feature] <= threshold else node["right"]
    return float(node.get("leaf") or 0.0)


def _fit_nonnegative_pairwise_weights(
    pair_vectors: Any,
    targets: Any,
    *,
    initial: Any,
    l2: float,
    iterations: int,
    learning_rate: float,
) -> Any:
    """Fit pairwise ridge weights under a simple non-negative constraint.

    The language-quality features are all oriented so that larger means
    "better". In small calibration sets, unconstrained ridge regression can
    assign negative coefficients to genuinely good signals because of
    collinearity. Projected gradient keeps the model transparent and monotone.
    """
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("numpy is required to train language quality models") from exc

    d = np.asarray(pair_vectors, dtype=float)
    t = np.asarray(targets, dtype=float)
    if d.size == 0:
        return np.asarray(initial, dtype=float)
    weights = np.clip(np.asarray(initial, dtype=float), 0.0, 10.0)
    steps = max(1, int(iterations))
    rate = max(1e-6, float(learning_rate))
    n_pairs = max(1, d.shape[0])
    penalty = max(0.0, float(l2))
    for _ in range(steps):
        residual = d @ weights - t
        gradient = (2.0 / n_pairs) * (d.T @ residual) + (2.0 * penalty / n_pairs) * weights
        weights = np.clip(weights - rate * gradient, 0.0, 10.0)
    return weights


def language_fragment_score(text: str, language: str | None = None) -> float:
    """Cheap fragment-density score using a language profile."""
    profile = get_language_scoring_profile(language)
    cleaned = _az(text)
    if len(cleaned) < 20:
        return 0.0
    score = 0.0
    weighted = {**profile.function_fragments, **profile.long_fragments}
    for fragment, weight in weighted.items():
        score += cleaned.count(fragment) * weight
    expected_slots = max(1.0, len(cleaned) / 22.0)
    return min(1.0, score / expected_slots)


def language_coherence_score(text: str, language: str | None = None) -> float:
    """Less saturating language signal than raw fragment counts."""
    profile = get_language_scoring_profile(language)
    cleaned = _az(text)
    if len(cleaned) < 20:
        return 0.0
    score = 0.0
    distinct = 0
    for fragment, weight in profile.long_fragments.items():
        count = cleaned.count(fragment)
        if count:
            distinct += 1
            score += min(2, count) * weight
    function_score = 0.0
    for fragment, weight in profile.function_fragments.items():
        count = cleaned.count(fragment)
        if count:
            distinct += 1
            function_score += min(4, count) * weight
    score += min(function_score, 2.4)
    distinct_bonus = min(1.0, distinct / 8.0)
    expected_slots = max(1.0, len(cleaned) / profile.coherence_expected_span)
    return min(1.0, (score / expected_slots) / 5.0 + distinct_bonus * 0.25)


def language_shape_score(text: str, language: str | None = None) -> float:
    """Reward diverse language evidence spread through a no-boundary stream."""
    profile = get_language_scoring_profile(language)
    cleaned = _az(text)
    if len(cleaned) < 40:
        return 0.0
    function_fragments = tuple(profile.function_fragments)
    anchor_hits = {fragment: cleaned.count(fragment) for fragment in profile.anchors}
    function_hits = {fragment: cleaned.count(fragment) for fragment in function_fragments}
    distinct_anchors = sum(1 for count in anchor_hits.values() if count)
    distinct_functions = sum(1 for count in function_hits.values() if count)

    windows = _text_windows(cleaned, 4)
    active_windows = 0
    for window in windows:
        if (
            any(fragment in window for fragment in profile.anchors)
            or sum(window.count(fragment) for fragment in function_fragments) >= 2
        ):
            active_windows += 1
    spread = active_windows / max(1, len(windows))
    anchor_density = min(
        1.0,
        sum(min(count, 2) for count in anchor_hits.values()) / max(2.0, len(cleaned) / 80.0),
    )
    function_balance = min(1.0, distinct_functions / 5.0)
    return min(
        1.0,
        distinct_anchors / 7.0 * 0.35
        + anchor_density * 0.25
        + spread * 0.25
        + function_balance * 0.15,
    )


def language_evidence_dispersion_score(text: str, language: str | None = None) -> float:
    """Reward language evidence spread across the whole candidate text."""
    profile = get_language_scoring_profile(language)
    cleaned = _az(text)
    if len(cleaned) < 60:
        return 0.0
    evidence_fragments = tuple(dict.fromkeys((*profile.anchors, *profile.long_fragments)))
    if not evidence_fragments:
        return 0.0
    windows = _text_windows(cleaned, 6)
    window_hits = [
        sum(window.count(fragment) for fragment in evidence_fragments)
        for window in windows
    ]
    total_hits = sum(window_hits)
    if total_hits <= 0:
        return 0.0
    active_fraction = sum(1 for hits in window_hits if hits > 0) / max(1, len(window_hits))
    distribution = [hits / total_hits for hits in window_hits if hits > 0]
    entropy = -sum(p * math.log(p) for p in distribution) / max(1e-9, math.log(len(windows)))
    return max(0.0, min(1.0, active_fraction * 0.55 + entropy * 0.45))


def function_content_balance_score(text: str, language: str | None = None) -> float:
    """Reward a plausible mix of content anchors and function fragments."""
    profile = get_language_scoring_profile(language)
    cleaned = _az(text)
    if len(cleaned) < 60:
        return 0.0
    content_fragments = tuple(profile.long_fragments)
    function_fragments = tuple(profile.function_fragments)
    content_hits = sum(cleaned.count(fragment) for fragment in content_fragments)
    function_hits = sum(cleaned.count(fragment) for fragment in function_fragments)
    total = content_hits + function_hits
    if total <= 0:
        return 0.0
    content_share = content_hits / total
    # Good damaged text can be function-heavy, but pure function soup should
    # lose to candidates with some content-bearing evidence.
    balance = 1.0 - min(1.0, abs(content_share - 0.38) / 0.38)
    content_presence = min(1.0, content_hits / max(2.0, len(cleaned) / 140.0))
    return max(0.0, min(1.0, balance * 0.55 + content_presence * 0.45))


def content_rhythm_control_score(
    *,
    content_quality: Any,
    function_content_balance: Any,
    function_overuse: Any,
    short_fragment_concentration: Any,
    short_word_control: Any,
    binary_ngram_fit: Any,
) -> float:
    """Reward content-bearing text with plausible function-word rhythm.

    This deliberately composes general signals rather than adding
    Copiale-specific fragments. It helps separate real damaged prose from
    candidates that are locally word-like but held together by repeated short
    fragments.
    """
    content = max(0.0, min(1.0, _as_float(content_quality)))
    balance = max(0.0, min(1.0, _as_float(function_content_balance)))
    overuse_control = max(0.0, min(1.0, 1.0 - _as_float(function_overuse)))
    short_fragment_control = max(0.0, min(1.0, 1.0 - _as_float(short_fragment_concentration)))
    short_words = max(0.0, min(1.0, _as_float(short_word_control)))
    binary_fit = max(0.0, min(1.0, _as_float(binary_ngram_fit)))
    return max(
        0.0,
        min(
            1.0,
            content * 0.26
            + balance * 0.24
            + overuse_control * 0.18
            + short_fragment_control * 0.14
            + short_words * 0.10
            + binary_fit * 0.08,
        ),
    )


def language_window_stability_score(text: str, language: str | None = None) -> float:
    """Reward language evidence that is not confined to a few hot windows."""
    profile = get_language_scoring_profile(language)
    cleaned = _az(text)
    if len(cleaned) < 80:
        return 0.0
    content_fragments = tuple(profile.long_fragments)
    function_fragments = tuple(profile.function_fragments)
    windows = _text_windows(cleaned, 8)
    if not windows:
        return 0.0
    window_scores: list[float] = []
    balanced_windows = 0
    for window in windows:
        content_hits = sum(window.count(fragment) for fragment in content_fragments)
        function_hits = sum(window.count(fragment) for fragment in function_fragments)
        if content_hits or function_hits:
            total = content_hits + function_hits
            balance = 1.0 - min(1.0, abs((content_hits / max(1, total)) - 0.38) / 0.38)
            window_scores.append(min(1.0, total / 4.0) * (0.55 + balance * 0.45))
            if content_hits and function_hits:
                balanced_windows += 1
        else:
            window_scores.append(0.0)
    active = sum(1 for score in window_scores if score > 0.0) / len(window_scores)
    balanced = balanced_windows / len(windows)
    mean_score = sum(window_scores) / len(window_scores)
    variance = sum((score - mean_score) ** 2 for score in window_scores) / len(window_scores)
    stability = max(0.0, 1.0 - math.sqrt(variance) / 0.55)
    return max(0.0, min(1.0, active * 0.30 + balanced * 0.30 + mean_score * 0.25 + stability * 0.15))


def short_fragment_concentration_score(text: str, language: str | None = None) -> float:
    """Score suspicious concentration in short language-like fragments."""
    profile = get_language_scoring_profile(language)
    cleaned = _az(text)
    if len(cleaned) < 60:
        return 0.0
    fragments = tuple(
        fragment for fragment in profile.function_overuse_fragments
        if 2 <= len(fragment) <= 4
    )
    if not fragments:
        return 0.0
    counts = {fragment: cleaned.count(fragment) for fragment in fragments}
    total = sum(counts.values())
    if total <= 4:
        return 0.0
    max_share = max(counts.values()) / max(1, total)
    density = total / max(1.0, len(cleaned) / 18.0)
    low_diversity = max(0.0, (5 - sum(1 for count in counts.values() if count)) / 5.0)
    return max(
        0.0,
        min(
            1.0,
            max(0.0, max_share - 0.30) * 1.2
            + max(0.0, density - 1.0) * 0.35
            + low_diversity * 0.25,
        ),
    )


def function_overuse_penalty(text: str, language: str | None = None) -> float:
    """Penalize outputs built from too few repeated language function chunks."""
    profile = get_language_scoring_profile(language)
    cleaned = _az(text)
    if len(cleaned) < 40:
        return 0.0
    counts = {
        fragment: cleaned.count(fragment)
        for fragment in profile.function_overuse_fragments
    }
    total = sum(counts.values())
    if total <= 4:
        return 0.0
    distinct = sum(1 for count in counts.values() if count)
    max_count = max(counts.values())
    expected = max(4.0, len(cleaned) / profile.function_expected_span)
    concentration = max_count / max(1, total)
    excess = max(0.0, (total - expected) / expected)
    low_diversity = max(0.0, (5 - distinct) / 5.0)
    return min(
        1.0,
        excess * 0.55
        + max(0.0, concentration - profile.concentration_ceiling) * 1.3
        + low_diversity * 0.35,
    )


def binary_ngram_fit_score(mean_log_prob: Any) -> float:
    """Map a binary mean log probability into a small 0..1 fit signal."""
    try:
        score = float(mean_log_prob)
    except (TypeError, ValueError):
        return 0.0
    # Known-good Zenith-format plaintext under current Decipher-built models
    # often sits near -11 mean natural log probability. Obvious degraded
    # controls are several points lower. Keep this deliberately mild: it is a
    # tie-breaker against word-islands, not a declaration criterion.
    return max(0.0, min(1.0, (score + 16.0) / 6.0))


def word_lattice_quality_score(diagnostics: dict[str, Any]) -> float:
    """Score dictionary segmentation quality from language-independent fields."""
    dict_rate = _as_float(diagnostics.get("dict_rate"))
    letter_count = max(1.0, _as_float(diagnostics.get("letter_count")))
    segmentation_cost = _as_float(diagnostics.get("segmentation_cost"))
    pseudo_word_fraction = _as_float(diagnostics.get("pseudo_word_fraction"))
    long_pseudo_word_fraction = _as_float(diagnostics.get("long_pseudo_word_fraction"))
    short_word_fraction = _as_float(diagnostics.get("short_word_fraction"))
    cost_per_char = segmentation_cost / letter_count
    cost_quality = max(0.0, min(1.0, 1.0 - cost_per_char / 6.0))
    pseudo_quality = max(0.0, 1.0 - pseudo_word_fraction)
    long_pseudo_quality = max(0.0, 1.0 - long_pseudo_word_fraction * 2.2)
    short_word_quality = max(0.0, 1.0 - max(0.0, short_word_fraction - 0.10) * 3.0)
    return max(
        0.0,
        min(
            1.0,
            dict_rate * 0.42
            + cost_quality * 0.24
            + pseudo_quality * 0.20
            + long_pseudo_quality * 0.10
            + short_word_quality * 0.04,
        ),
    )


def content_word_metrics(
    segmented_words: list[str],
    word_set: set[str],
    language: str | None = None,
) -> dict[str, Any]:
    """Return profile-aware content-word evidence from segmented words."""
    profile = get_language_scoring_profile(language)
    stop_words = set(profile.stop_words)
    hits = [word.upper() for word in segmented_words if word.upper() in word_set]
    content_hits = [
        word for word in hits
        if len(word) >= 4 and word not in stop_words
    ]
    long_content_hits = [word for word in content_hits if len(word) >= 6]
    hit_count = len(hits)
    word_count = len(segmented_words)
    hit_char_count = sum(len(word) for word in hits)
    content_char_count = sum(len(word) for word in content_hits)
    return {
        "dictionary_hit_count": hit_count,
        "dictionary_content_word_count": len(content_hits),
        "dictionary_long_content_word_count": len(long_content_hits),
        "dictionary_content_word_fraction": round(len(content_hits) / word_count, 4) if word_count else 0.0,
        "dictionary_content_char_fraction": round(content_char_count / max(1, hit_char_count), 4),
        "dictionary_content_sample": content_hits[:12],
        "dictionary_long_content_sample": long_content_hits[:8],
    }


def content_word_quality_score(diagnostics: dict[str, Any]) -> float:
    """Reward real dictionary content words over short function fragments."""
    content_fraction = _as_float(diagnostics.get("dictionary_content_word_fraction"))
    content_char_fraction = _as_float(diagnostics.get("dictionary_content_char_fraction"))
    long_count = _as_float(diagnostics.get("dictionary_long_content_word_count"))
    content_count = _as_float(diagnostics.get("dictionary_content_word_count"))
    count_quality = min(1.0, content_count / 10.0)
    long_quality = min(1.0, long_count / 4.0)
    return max(
        0.0,
        min(
            1.0,
            content_fraction * 0.35
            + content_char_fraction * 0.30
            + count_quality * 0.20
            + long_quality * 0.15,
        ),
    )


def content_lattice_consistency_score(lattice_quality: Any, content_quality: Any) -> float:
    """Reward segmentations whose clean lattice is supported by content words.

    A bad basin can segment neatly into short function fragments while lacking
    credible content-bearing dictionary words. This feature stays language
    agnostic: it only compares two already-profiled quality signals.
    """
    lattice = max(0.0, min(1.0, _as_float(lattice_quality)))
    content = max(0.0, min(1.0, _as_float(content_quality)))
    unsupported_lattice = max(0.0, lattice - content - 0.12)
    return max(
        0.0,
        min(
            1.0,
            content * 0.72
            + min(lattice, content + 0.12) * 0.28
            - unsupported_lattice * 0.65,
        ),
    )


def word_island_template_penalty(
    diagnostics: dict[str, Any],
    *,
    repetition: float,
) -> float:
    """Penalize candidates that look like templated dictionary-word islands.

    This is deliberately separate from raw dictionary/content rewards. A bad
    homophonic basin can contain many real content words while still being
    held together by repeated fragments and pseudo-word glue. The penalty is
    strongest when all three pressures coincide.
    """
    content_quality = content_word_quality_score(diagnostics)
    pseudo_word_fraction = _as_float(diagnostics.get("pseudo_word_fraction"))
    long_pseudo_word_fraction = _as_float(diagnostics.get("long_pseudo_word_fraction"))
    dict_rate = _as_float(diagnostics.get("dict_rate"))
    repeated_pressure = max(0.0, min(1.0, (float(repetition) - 0.64) / 0.36))
    content_pressure = max(0.0, min(1.0, (content_quality - 0.70) / 0.30))
    pseudo_pressure = max(0.0, min(1.0, (pseudo_word_fraction - 0.36) / 0.28))
    long_pseudo_pressure = max(0.0, min(1.0, (long_pseudo_word_fraction - 0.18) / 0.28))
    dictionary_pressure = max(0.0, min(1.0, (dict_rate - 0.54) / 0.12))
    return max(
        0.0,
        min(
            1.0,
            repeated_pressure * 0.42
            + content_pressure * 0.18
            + pseudo_pressure * 0.16
            + long_pseudo_pressure * 0.12
            + dictionary_pressure * 0.12,
        ),
    )


def repetitive_word_island_penalty(text: str) -> float:
    """Penalty for repetitive fragment islands that lack broader coherence."""
    cleaned = _az(text)
    if len(cleaned) < 24:
        return 0.0
    penalty = 0.0
    for n in (3, 4, 5):
        counts: dict[str, int] = {}
        for idx in range(0, max(0, len(cleaned) - n + 1)):
            gram = cleaned[idx: idx + n]
            counts[gram] = counts.get(gram, 0) + 1
        repeated = sum(max(0, count - 2) for count in counts.values())
        penalty += repeated / max(1.0, len(cleaned) / (n * 2.0))
    return min(1.0, penalty / 3.0)


def segmentation_shape_penalty(
    *,
    pseudo_word_fraction: float,
    long_pseudo_word_fraction: float,
    short_word_fraction: float,
) -> float:
    """Penalty for segmentations held together by pseudo-word glue."""
    pseudo_pressure = max(0.0, pseudo_word_fraction - 0.42) / 0.38
    long_pseudo_pressure = max(0.0, long_pseudo_word_fraction - 0.20) / 0.35
    short_word_pressure = max(0.0, short_word_fraction - 0.16) / 0.25
    return min(
        1.0,
        pseudo_pressure * 0.45
        + long_pseudo_pressure * 0.40
        + short_word_pressure * 0.15,
    )


def _az(text: str) -> str:
    return "".join(ch for ch in text.upper() if "A" <= ch <= "Z")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded_linear(value: float | None, *, low: float, high: float, neutral: float) -> float:
    if value is None or high <= low:
        return neutral
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    if not all(math.isfinite(value) for value in xs + ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    try:
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
        den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    except OverflowError:
        return 0.0
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return num / (den_x * den_y)


def _ranking_summary_from_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("group") or "__all__"), []).append(row)
    ranks: list[int] = []
    top3 = 0
    top5 = 0
    for group_rows in groups.values():
        if not group_rows:
            continue
        by_label = sorted(group_rows, key=lambda row: float(row.get("label") or 0.0), reverse=True)
        by_score = sorted(group_rows, key=lambda row: float(row.get("score") or 0.0), reverse=True)
        best = by_label[0]
        rank = by_score.index(best) + 1 if best in by_score else None
        if rank is None:
            continue
        ranks.append(rank)
        top3 += int(rank <= 3)
        top5 += int(rank <= 5)
    return {
        "group_count": len(ranks),
        "mean_best_label_prediction_rank": sum(ranks) / len(ranks) if ranks else None,
        "top3_captures": top3,
        "top5_captures": top5,
    }


def _round_or_none(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _text_windows(text: str, count: int) -> list[str]:
    if count <= 1 or len(text) <= count:
        return [text]
    size = max(1, len(text) // count)
    windows = [text[idx: idx + size] for idx in range(0, len(text), size)]
    return windows[:count]
