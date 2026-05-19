#!/usr/bin/env python3
"""Train a transparent fast language-quality scorer.

This is an offline calibration tool. It may use labels derived from solved
artifacts or human/corpus positives, but the saved model consumes only
ground-truth-free finalist features at solver time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from analysis.language_scoring import (  # noqa: E402
    LANGUAGE_QUALITY_FEATURES,
    get_language_scoring_profile,
    language_quality_feature_dict,
    language_quality_solver_evidence_features,
    train_gradient_boosted_language_quality_model,
    train_linear_language_quality_model,
    train_pairwise_language_quality_model,
)
from automated.runner import (  # noqa: E402
    _automated_candidate_diagnostics,
    _word_list,
    _zenith_native_model_path,
)
from benchmark.scorer import score_decryption  # noqa: E402


@dataclass
class TrainingExample:
    text: str
    label: float
    source: str
    group: str
    features: dict[str, float]
    metadata: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a fast language-quality scorer from corpus/artifact examples."
    )
    parser.add_argument("--language", default="de", help="Language code for scoring profile.")
    parser.add_argument("--artifact", action="append", default=[], help="Artifact file or directory to mine.")
    parser.add_argument("--probe-jsonl", action="append", default=[], help="Probe JSONL file to mine.")
    parser.add_argument(
        "--global-repair-json",
        action="append",
        default=[],
        help="Global repair probe JSON file or directory to mine as candidate menus.",
    )
    parser.add_argument("--positive-text", action="append", default=[], help="Plaintext corpus file with positive examples.")
    parser.add_argument("--output", required=True, help="Path for trained model JSON.")
    parser.add_argument("--report", help="Optional markdown report path.")
    parser.add_argument(
        "--holdout-group",
        action="append",
        default=[],
        help="Group/test_id to exclude from training and evaluate separately.",
    )
    parser.add_argument(
        "--leave-one-group-out",
        action="store_true",
        help="Add a compact leave-one-group-out evaluation to the report.",
    )
    parser.add_argument("--chunk-size", type=int, default=700, help="Positive corpus chunk size.")
    parser.add_argument("--chunk-stride", type=int, default=500, help="Positive corpus chunk stride.")
    parser.add_argument("--min-chars", type=int, default=120, help="Minimum A-Z chars per example.")
    parser.add_argument(
        "--synthetic-negatives-per-positive",
        type=int,
        default=0,
        help="Generate this many word-island negative controls per positive example.",
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help=(
            "Train only on solver/probe finalist candidates, excluding clean "
            "ground-truth and positive-corpus examples. Useful for tuning "
            "within-finalist ranking rather than clean-vs-damaged detection."
        ),
    )
    parser.add_argument("--l2", type=float, default=0.25, help="Ridge regularization strength.")
    parser.add_argument(
        "--objective",
        choices=("regression", "pairwise"),
        default="regression",
        help=(
            "Training objective. pairwise learns within-menu candidate ordering "
            "from same-group label deltas."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=("linear", "gbt"),
        default="linear",
        help="Model family: transparent linear/pairwise ranker or small gradient-boosted trees.",
    )
    parser.add_argument(
        "--label-target",
        choices=("post_hoc_char", "adjudication_no_target"),
        default="post_hoc_char",
        help=(
            "Offline label to train against for repair JSONs. "
            "adjudication_no_target uses the runtime collateral-health score, "
            "not benchmark plaintext."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=("all", "no_solver", "text_only"),
        default="all",
        help=(
            "Feature subset for calibration. no_solver drops scalar solver "
            "evidence; text_only also drops mask/deletion controls."
        ),
    )
    parser.add_argument(
        "--min-label-delta",
        type=float,
        default=0.02,
        help="Minimum within-group label gap for pairwise training examples.",
    )
    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=2000,
        help="Maximum pairwise comparisons to use per group.",
    )
    parser.add_argument(
        "--nonnegative-weights",
        action="store_true",
        help=(
            "For pairwise training, constrain feature weights to be non-negative. "
            "This matches feature sets where every signal is oriented so larger is better."
        ),
    )
    parser.add_argument(
        "--nonnegative-iterations",
        type=int,
        default=4000,
        help="Projected-gradient iterations for --nonnegative-weights.",
    )
    parser.add_argument(
        "--nonnegative-learning-rate",
        type=float,
        default=0.01,
        help="Projected-gradient learning rate for --nonnegative-weights.",
    )
    parser.add_argument("--gbt-trees", type=int, default=75, help="Number of boosted trees for --model-type gbt.")
    parser.add_argument("--gbt-depth", type=int, default=3, help="Maximum tree depth for --model-type gbt.")
    parser.add_argument("--gbt-learning-rate", type=float, default=0.06, help="Boosted-tree learning rate.")
    parser.add_argument("--gbt-min-samples-leaf", type=int, default=2, help="Minimum samples per tree leaf.")
    args = parser.parse_args()

    examples: list[TrainingExample] = []
    for path in args.artifact:
        examples.extend(load_artifact_examples(Path(path), language=args.language, min_chars=args.min_chars))
    for path in args.probe_jsonl:
        examples.extend(load_probe_examples(Path(path), language=args.language, min_chars=args.min_chars))
    for path in args.global_repair_json:
        examples.extend(load_global_repair_examples(
            Path(path),
            language=args.language,
            label_target=args.label_target,
        ))
    for path in args.positive_text:
        examples.extend(load_positive_text_examples(
            Path(path),
            language=args.language,
            chunk_size=args.chunk_size,
            chunk_stride=args.chunk_stride,
            min_chars=args.min_chars,
        ))

    examples = dedupe_examples(examples)
    if args.candidate_only:
        examples = [
            example for example in examples
            if (
                ":top_finalist:" in example.source
                or example.source.startswith("probe:")
                or example.source.startswith("global_repair:")
            )
            and not example.source.startswith("positive_text:")
            and ":ground_truth" not in example.source
            and (example.metadata.get("kind") != "synthetic_negative")
        ]
    if args.synthetic_negatives_per_positive > 0:
        examples = dedupe_examples(
            examples
            + synthetic_negative_examples(
                examples,
                language=args.language,
                per_positive=args.synthetic_negatives_per_positive,
                min_chars=args.min_chars,
            )
        )
    if len(examples) < 4:
        raise SystemExit(
            "Need at least four examples. Provide solved artifacts/probe JSONL "
            "and/or --positive-text corpus files."
        )
    holdout_groups = {str(group) for group in args.holdout_group}
    train_examples = [
        example for example in examples
        if example.group not in holdout_groups
    ]
    holdout_examples = [
        example for example in examples
        if example.group in holdout_groups
    ]
    if len(train_examples) < 4:
        raise SystemExit(
            "Need at least four non-holdout examples to train a scorer."
        )

    model = train_model(
        train_examples,
        language=args.language,
        objective=args.objective,
        model_type=args.model_type,
        feature_names=feature_names_for_mode(args.feature_set),
        l2=args.l2,
        min_label_delta=args.min_label_delta,
        max_pairs_per_group=args.max_pairs_per_group,
        nonnegative_weights=args.nonnegative_weights,
        nonnegative_iterations=args.nonnegative_iterations,
        nonnegative_learning_rate=args.nonnegative_learning_rate,
        gbt_trees=args.gbt_trees,
        gbt_depth=args.gbt_depth,
        gbt_learning_rate=args.gbt_learning_rate,
        gbt_min_samples_leaf=args.gbt_min_samples_leaf,
    )
    output = resolve_path(Path(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    logo_report = (
        leave_one_group_out_report(
            examples,
            language=args.language,
            objective=args.objective,
            model_type=args.model_type,
            feature_names=feature_names_for_mode(args.feature_set),
            l2=args.l2,
            min_label_delta=args.min_label_delta,
            max_pairs_per_group=args.max_pairs_per_group,
            nonnegative_weights=args.nonnegative_weights,
            nonnegative_iterations=args.nonnegative_iterations,
            nonnegative_learning_rate=args.nonnegative_learning_rate,
            gbt_trees=args.gbt_trees,
            gbt_depth=args.gbt_depth,
            gbt_learning_rate=args.gbt_learning_rate,
            gbt_min_samples_leaf=args.gbt_min_samples_leaf,
        )
        if args.leave_one_group_out
        else []
    )
    report = render_report(
        train_examples,
        model.to_dict(),
        output,
        holdout_examples=holdout_examples,
        all_examples=examples,
        leave_one_group_out_rows=logo_report,
    )
    if args.report:
        report_path = resolve_path(Path(args.report))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    print(report)


def load_artifact_examples(path: Path, *, language: str, min_chars: int) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for artifact_path in expand_paths(path, "*.json"):
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ground_truth = str(artifact.get("ground_truth") or "")
        if ground_truth:
            examples.append(make_example(
                text=ground_truth,
                label=1.0,
                source=f"artifact:{artifact_path.name}:ground_truth",
                group=str(artifact.get("test_id") or artifact_path.stem),
                language=language,
                min_chars=min_chars,
                metadata={"artifact": str(artifact_path), "kind": "ground_truth"},
            ))
        for step in artifact.get("steps") or []:
            if step.get("name") != "search_null_masks":
                continue
            for rank, row in enumerate(step.get("top_finalists") or [], start=1):
                text = str(row.get("decryption") or row.get("validation_text") or row.get("preview") or "")
                if not text or not ground_truth:
                    continue
                score = score_decryption(
                    str(artifact.get("test_id") or artifact_path.stem),
                    text,
                    ground_truth,
                    0.0,
                    "completed",
                )
                examples.append(make_example(
                    text=text,
                    label=score.char_accuracy,
                    source=f"artifact:{artifact_path.name}:top_finalist:{rank}",
                    group=str(artifact.get("test_id") or artifact_path.stem),
                    language=language,
                    min_chars=min_chars,
                    diagnostics=row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else None,
                    original_length=artifact.get("original_cipher_token_count") or artifact.get("cipher_token_count"),
                    filtered_length=row.get("filtered_length"),
                    mask_size=len(row.get("mask") or []),
                    metadata={
                        "artifact": str(artifact_path),
                        "kind": "candidate",
                        "rank": rank,
                        "mask": row.get("mask") or [],
                        "validation_score_v2": row.get("validation_score_v2"),
                        "ensemble_score_v1": row.get("ensemble_score_v1"),
                        "selection_score": row.get("selection_score"),
                    },
                    solver_evidence=row,
                ))
    return [example for example in examples if example.text]


def load_probe_examples(path: Path, *, language: str, min_chars: int) -> list[TrainingExample]:
    path = resolve_path(path)
    examples: list[TrainingExample] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        rows = list(payload.get("all_rows") or payload.get("top_rows") or payload.get("rows") or [])
        group = str(payload.get("test_id") or f"{path.stem}:{line_no}")
        for rank, row in enumerate(rows, start=1):
            text = str(row.get("validation_text") or row.get("decryption") or row.get("preview") or "")
            label = row.get("char_accuracy")
            if text and label is not None:
                examples.append(make_example(
                    text=text,
                    label=float(label),
                    source=f"probe:{path.name}:{line_no}:{rank}",
                    group=group,
                    language=language,
                    min_chars=min_chars,
                    diagnostics=row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else None,
                    filtered_length=row.get("filtered_length"),
                    mask_size=len(row.get("mask") or []),
                    metadata={
                        "probe_jsonl": str(path),
                        "kind": "candidate",
                        "line": line_no,
                        "rank": rank,
                        "mask": row.get("mask") or [],
                    },
                    solver_evidence=row,
                ))
    return [example for example in examples if example.text]


def load_global_repair_examples(
    path: Path,
    *,
    language: str,
    label_target: str = "post_hoc_char",
) -> list[TrainingExample]:
    """Load global repair probe variants as offline ranking examples.

    These examples often represent multi-page candidates and may not carry a
    single full plaintext string. Their labels are post-hoc calibration values
    from completed probe reports; features are runtime-only aggregate signals.
    """
    entries: list[dict[str, Any]] = []
    for repair_path in expand_repair_paths(path):
        try:
            payload = json.loads(repair_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("all_variants") if isinstance(payload.get("all_variants"), list) else []
        if not rows:
            rows = payload.get("top_variants") if isinstance(payload.get("top_variants"), list) else []
        group = str(payload.get("label") or repair_path.stem)
        group = f"global_repair:{repair_path.stem}:{group}"
        for rank, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            label, raw_label = repair_training_label(row, label_target=label_target)
            if label is None:
                continue
            text = str(row.get("preview") or "")
            features = global_repair_feature_dict(row)
            example = TrainingExample(
                text=text or "__GLOBAL_REPAIR_CANDIDATE__",
                label=max(0.0, min(1.0, float(label))),
                source=f"global_repair:{repair_path.name}:top_variant:{rank}",
                group=group,
                features=features,
                metadata={
                    "global_repair_json": str(repair_path),
                    "kind": "global_repair_candidate",
                    "rank": rank,
                    "label_target": label_target,
                    "raw_label": raw_label,
                    "edits": row.get("edits") or [],
                    "mask": row.get("mask") or [],
                    "acceptance": row.get("repair_acceptance") or {},
                    "source_experiment": payload.get("source_experiment"),
                    "source_artifact": payload.get("source_artifact"),
                    "section": payload.get("section"),
                    "label": payload.get("label"),
                    "test_ids": payload.get("test_ids") or [],
                },
            )
            entries.append({
                "example": example,
                "row": row,
                "payload": payload,
            })
    add_global_repair_family_features(entries)
    return [entry["example"] for entry in entries]


def repair_training_label(row: dict[str, Any], *, label_target: str) -> tuple[float | None, float | None]:
    if label_target == "adjudication_no_target":
        adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
        raw = row.get("adjudication_no_target_score", adjudication.get("adjudication_no_target_score"))
        if raw is None:
            return None, None
        raw_float = _maybe_float(raw, 0.0)
        return _bounded_linear(raw_float, low=-4.0, high=4.0), raw_float
    raw = row.get("post_hoc_char_avg")
    if raw is None:
        return None, None
    raw_float = _maybe_float(raw, 0.0)
    return _clamp(raw_float), raw_float


def global_repair_feature_dict(row: dict[str, Any]) -> dict[str, float]:
    """Map multi-page repair runtime evidence into the shared feature schema."""
    features = {name: 0.5 for name in LANGUAGE_QUALITY_FEATURES}
    dict_rate = _maybe_float(row.get("page_dict_avg"), 0.5)
    content_quality = _maybe_float(row.get("page_content_word_quality_avg"), 0.5)
    lattice_quality = _maybe_float(row.get("page_content_char_avg"), dict_rate)
    pseudo_avg = _maybe_float(row.get("page_pseudo_word_avg"), 0.5)
    features.update({
        "dict_rate": _clamp(dict_rate),
        "word_lattice_quality": _clamp(lattice_quality),
        "content_word_quality": _clamp(content_quality),
        "content_lattice_consistency": _clamp((lattice_quality + content_quality) / 2.0),
        "language_coherence": _clamp(_maybe_float(row.get("page_language_coherence_avg"), 0.5)),
        "language_shape": _clamp(_maybe_float(row.get("page_shape_component_avg"), 0.5)),
        "language_evidence_dispersion": _clamp(_maybe_float(row.get("page_evidence_dispersion_avg"), 0.5)),
        "function_content_balance": _clamp(_maybe_float(row.get("page_content_char_avg"), 0.5)),
        "content_rhythm_control": _clamp(0.5 * content_quality + 0.5 * (1.0 - pseudo_avg)),
        "language_window_stability": _clamp(_maybe_float(row.get("page_window_stability_avg"), 0.5)),
        "binary_ngram_fit": _clamp(_maybe_float(row.get("page_binary_component_avg"), 0.5)),
        "pseudo_word_control": _clamp(1.0 - pseudo_avg),
        "repetition_control": _clamp(_maybe_float(row.get("page_repetition_control_avg"), 0.5)),
        "deletion_control": 1.0,
        "mask_size_control": _clamp(1.0 - max(0, len(row.get("mask") or []) - 2) / 4.0),
        "validation_score_control": _bounded_linear(
            _maybe_float(row.get("page_validation_avg"), 0.0),
            low=-1.5,
            high=5.0,
        ),
        "ensemble_score_control": _bounded_linear(
            _maybe_float(row.get("page_robust_score"), 0.0),
            low=-1.5,
            high=5.0,
        ),
        "selection_score_control": _bounded_linear(
            _maybe_float(row.get("page_balanced_score"), 0.0),
            low=-1.5,
            high=5.0,
        ),
        "solver_evidence_present": 1.0,
        "mask_family_support_control": 0.5,
        "mask_family_validation_control": 0.5,
        "mask_family_balanced_control": 0.5,
        "mask_family_dictionary_control": 0.5,
        "mask_family_binary_control": 0.5,
        "mask_family_robust_control": 0.5,
    })
    features.update(global_repair_edit_feature_dict(row))
    evidence = row.get("repair_evidence") if isinstance(row.get("repair_evidence"), dict) else {}
    page_count = max(1.0, float(evidence.get("page_count") or 1.0))
    runtime_suspicious = float(evidence.get("runtime_suspicious_pages") or 0.0) / page_count
    changed = float(evidence.get("preview_pages_changed") or 0.0) / page_count
    # Reuse generic controls for repair-specific evidence without extending the
    # runtime model schema yet. Do not use calibration_suspicious_pages here:
    # those flags are post-hoc ground-truth diagnostics and must never affect
    # a saved model's runtime score.
    features["template_island_control"] = _clamp(1.0 - runtime_suspicious)
    features["function_overuse_control"] = _clamp(1.0 - runtime_suspicious)
    features["short_fragment_control"] = _clamp(changed)
    return features


def global_repair_edit_feature_dict(row: dict[str, Any]) -> dict[str, float]:
    """Return ground-truth-free edit-level repair evidence.

    These signals summarize whether a local repair changed pages in a coherent
    way according to runtime metrics only. They deliberately ignore post-hoc
    calibration fields such as character accuracy.
    """
    neutral = {
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
    evidence = row.get("repair_evidence") if isinstance(row.get("repair_evidence"), dict) else {}
    pages = [
        page for page in (evidence.get("pages") or [])
        if isinstance(page, dict)
    ]
    edits = row.get("edits") if isinstance(row.get("edits"), list) else []
    edit_count = len([edit for edit in edits if str(edit).strip().lower() != "baseline"])
    acceptance = row.get("repair_acceptance") if isinstance(row.get("repair_acceptance"), dict) else {}
    positive_signal_count = _maybe_float(acceptance.get("positive_signal_count"), 0.0)
    accepted_bonus = 0.15 if acceptance.get("accepted") is True else 0.0
    features = dict(neutral)
    features["repair_edit_count_control"] = _clamp(1.0 - max(0, edit_count - 2) / 4.0)
    features["repair_acceptance_control"] = _clamp((positive_signal_count / 5.0) + accepted_bonus)
    features.update(adjudication_component_features(row))
    if not pages:
        return features

    validation_deltas = page_deltas(pages, "validation_delta")
    language_deltas = page_deltas(pages, "language_quality_delta")
    binary_deltas = page_deltas(pages, "binary_ngram_fit_delta")
    dict_deltas = page_deltas(pages, "dict_rate_delta")
    pseudo_deltas = page_deltas(pages, "pseudo_word_fraction_delta")
    before_after = changed_window_pairs(pages)
    before_quality = [
        changed_window_quality(before)
        for before, _after in before_after
        if before
    ]
    after_quality = [
        changed_window_quality(after)
        for _before, after in before_after
        if after
    ]
    window_deltas = [
        changed_window_quality(after) - changed_window_quality(before)
        for before, after in before_after
        if before or after
    ]
    page_signal_values = [page_signal_consensus(page) for page in pages]
    after_text = "".join(after for _before, after in before_after)
    page_count = len(pages)
    signal_consensus = average_float(page_signal_values, default=0.5)
    features.update({
        "repair_validation_delta_control": _bounded_linear(
            average_float(validation_deltas),
            low=-0.08,
            high=0.08,
        ),
        "repair_min_validation_delta_control": _bounded_linear(
            min(validation_deltas) if validation_deltas else 0.0,
            low=-0.08,
            high=0.03,
        ),
        "repair_runtime_page_agreement_control": _clamp(
            sum(1 for value in validation_deltas if value >= 0.0) / max(1, page_count)
        ),
        "repair_signal_consensus_control": _clamp(
            signal_consensus
        ),
        "repair_delta_stability_control": _clamp(
            1.0 - population_stdev(validation_deltas) / 0.08
        ),
        "repair_language_delta_control": _bounded_linear(
            average_float(language_deltas),
            low=-0.04,
            high=0.04,
        ),
        "repair_binary_delta_control": _bounded_linear(
            average_float(binary_deltas),
            low=-0.04,
            high=0.04,
        ),
        "repair_dict_delta_control": _bounded_linear(
            average_float(dict_deltas),
            low=-0.03,
            high=0.03,
        ),
        "repair_pseudo_delta_control": _bounded_linear(
            -average_float(pseudo_deltas),
            low=-0.03,
            high=0.03,
        ),
        "repair_correlated_gain_control": correlated_gain_control(
            validation_delta=average_float(validation_deltas),
            signal_consensus=signal_consensus,
            language_delta=average_float(language_deltas),
            binary_delta=average_float(binary_deltas),
            dict_delta=average_float(dict_deltas),
            pseudo_delta=average_float(pseudo_deltas),
        ),
        "repair_window_quality_control": _clamp(average_float(after_quality, default=0.5)),
        "repair_window_quality_delta_control": _bounded_linear(
            average_float(window_deltas),
            low=-0.18,
            high=0.18,
        ),
        "repair_window_diversity_control": changed_window_diversity_control(after_text),
        "repair_window_repetition_control": changed_window_repetition_control(after_text),
        "repair_window_change_rate_control": (
            _clamp(len(before_after) / max(1, page_count))
            if before_after
            else 0.5
        ),
        "repair_page_signal_floor_control": (
            _clamp(min(page_signal_values))
            if page_signal_values
            else 0.5
        ),
        "repair_page_signal_range_control": (
            _clamp(1.0 - (max(page_signal_values) - min(page_signal_values)))
            if page_signal_values
            else 0.5
        ),
        "repair_validation_range_control": _clamp(
            1.0 - value_range(validation_deltas) / 0.16
        ),
        "repair_window_quality_floor_control": (
            _clamp(min(after_quality))
            if after_quality
            else 0.5
        ),
        "repair_window_quality_range_control": _clamp(
            1.0 - value_range(after_quality) / 0.50
        ),
        "repair_window_gain_agreement_control": (
            _clamp(sum(1 for value in window_deltas if value >= 0.0) / len(window_deltas))
            if window_deltas
            else 0.5
        ),
        "repair_cross_page_edit_consistency_control": cross_page_edit_consistency_control(before_after),
    })
    return features


def adjudication_component_features(row: dict[str, Any]) -> dict[str, float]:
    """Project word-repair adjudication components into generic repair controls.

    Do not include the final adjudication/adjudication_no_target scalar here:
    those may be offline labels. The component controls expose the same
    ground-truth-free evidence the solver has at runtime: collateral gain,
    collateral damage, target-only risk, global support, and edit breadth.
    """
    adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    if not adjudication:
        adjudication = row
    occurrence_count = _maybe_float(adjudication.get("occurrence_count"), 0.0)
    collateral_count = _maybe_float(adjudication.get("collateral_occurrences"), 0.0)
    target_only_penalty_value = _maybe_float(adjudication.get("target_only_penalty"), 0.0)
    collateral_gain = _maybe_float(adjudication.get("collateral_gain_avg"), 0.0)
    collateral_damage = _maybe_float(adjudication.get("collateral_damage_avg"), 0.0)
    collateral_word_gain = _maybe_float(adjudication.get("collateral_word_gain_weighted_avg"), 0.0)
    collateral_word_damage = _maybe_float(adjudication.get("collateral_word_damage_weighted_avg"), 0.0)
    word_improved = _maybe_float(adjudication.get("word_improved_weighted_occurrences"), 0.0)
    word_damaged = _maybe_float(adjudication.get("word_damaged_weighted_occurrences"), 0.0)
    global_leverage = _maybe_float(adjudication.get("global_leverage_score"), 0.0)
    edited_symbols = _maybe_float(adjudication.get("edited_symbol_count"), 0.0)
    collateral_net = collateral_gain - collateral_damage
    word_net = collateral_word_gain - collateral_word_damage
    improvement_balance = (word_improved + 1.0) / (word_improved + word_damaged + 2.0)
    return {
        "repair_validation_delta_control": _bounded_linear(collateral_net, low=-0.08, high=0.08),
        "repair_min_validation_delta_control": _bounded_linear(-collateral_damage, low=-0.08, high=0.0),
        "repair_runtime_page_agreement_control": _clamp(improvement_balance),
        "repair_signal_consensus_control": _bounded_linear(global_leverage, low=-3.0, high=6.0),
        "repair_delta_stability_control": _clamp(1.0 - min(1.0, target_only_penalty_value / 4.0)),
        "repair_language_delta_control": _bounded_linear(word_net, low=-1.0, high=1.0),
        "repair_binary_delta_control": _bounded_linear(collateral_net + 0.25 * word_net, low=-0.6, high=0.6),
        "repair_dict_delta_control": _bounded_linear(collateral_word_gain, low=0.0, high=1.0),
        "repair_pseudo_delta_control": _bounded_linear(-collateral_word_damage, low=-1.0, high=0.0),
        "repair_correlated_gain_control": _clamp(
            0.40 * _bounded_linear(global_leverage, low=-3.0, high=6.0)
            + 0.35 * improvement_balance
            + 0.25 * _bounded_linear(word_net, low=-1.0, high=1.0)
        ),
        "repair_window_quality_control": _bounded_linear(collateral_gain, low=0.0, high=0.08),
        "repair_window_quality_delta_control": _bounded_linear(collateral_net, low=-0.08, high=0.08),
        "repair_window_diversity_control": _clamp(min(1.0, collateral_count / 12.0)),
        "repair_window_repetition_control": _clamp(1.0 - min(1.0, word_damaged / 8.0)),
        "repair_window_change_rate_control": _clamp(occurrence_count / 24.0),
        "repair_page_signal_floor_control": _bounded_linear(word_net, low=-1.0, high=0.4),
        "repair_page_signal_range_control": _clamp(1.0 - min(1.0, collateral_word_damage / 2.0)),
        "repair_validation_range_control": _clamp(1.0 - min(1.0, collateral_damage / 0.12)),
        "repair_window_quality_floor_control": _clamp(1.0 - min(1.0, collateral_word_damage / 1.0)),
        "repair_window_quality_range_control": _clamp(1.0 - min(1.0, word_damaged / 6.0)),
        "repair_window_gain_agreement_control": _clamp(improvement_balance),
        "repair_cross_page_edit_consistency_control": _clamp(
            0.5 * min(1.0, collateral_count / 12.0)
            + 0.5 * _bounded_linear(global_leverage, low=-3.0, high=6.0)
        ),
        "repair_edit_count_control": _clamp(1.0 - max(0.0, edited_symbols - 2.0) / 4.0),
    }


def page_deltas(pages: list[dict[str, Any]], key: str) -> list[float]:
    return [
        _maybe_float(page.get(key), 0.0)
        for page in pages
        if page.get(key) is not None
    ]


def page_signal_consensus(page: dict[str, Any]) -> float:
    """Return fraction of runtime page signals moving in a good direction."""
    signals = [
        _maybe_float(page.get("validation_delta"), 0.0),
        _maybe_float(page.get("language_quality_delta"), 0.0),
        _maybe_float(page.get("dict_rate_delta"), 0.0),
        _maybe_float(page.get("binary_ngram_fit_delta"), 0.0),
        -_maybe_float(page.get("pseudo_word_fraction_delta"), 0.0),
    ]
    return sum(1 for value in signals if value >= 0.0) / len(signals)


def correlated_gain_control(
    *,
    validation_delta: float,
    signal_consensus: float,
    language_delta: float,
    binary_delta: float,
    dict_delta: float,
    pseudo_delta: float,
) -> float:
    """Reward validation gains corroborated by other runtime signals."""
    corroboration = average_float([
        _bounded_linear(language_delta, low=-0.03, high=0.03),
        _bounded_linear(binary_delta, low=-0.03, high=0.03),
        _bounded_linear(dict_delta, low=-0.02, high=0.02),
        _bounded_linear(-pseudo_delta, low=-0.02, high=0.02),
        signal_consensus,
    ], default=0.5)
    if validation_delta <= 0.0:
        return _clamp(0.45 + 0.35 * corroboration)
    validation_gain = _bounded_linear(validation_delta, low=0.0, high=0.08)
    orphan_penalty = validation_gain * max(0.0, 0.62 - corroboration)
    return _clamp(corroboration - orphan_penalty)


def changed_window_pairs(pages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for page in pages:
        excerpt = page.get("changed_excerpt")
        if not isinstance(excerpt, dict) or excerpt.get("changed") is False:
            continue
        before = "".join(ch for ch in str(excerpt.get("before") or "").upper() if "A" <= ch <= "Z")
        after = "".join(ch for ch in str(excerpt.get("after") or "").upper() if "A" <= ch <= "Z")
        if before or after:
            pairs.append((before, after))
    return pairs


def changed_window_quality(text: str) -> float:
    cleaned = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if len(cleaned) < 8:
        return 0.5
    return _clamp(
        0.40 * changed_window_diversity_control(cleaned)
        + 0.35 * changed_window_repetition_control(cleaned)
        + 0.25 * changed_window_top_letter_control(cleaned)
    )


def changed_window_diversity_control(text: str) -> float:
    cleaned = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if not cleaned:
        return 0.5
    # Changed windows are short, so do not require full alphabet diversity.
    return _clamp(len(set(cleaned)) / min(16.0, max(8.0, len(cleaned) / 3.0)))


def changed_window_repetition_control(text: str) -> float:
    cleaned = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if len(cleaned) < 8:
        return 0.5
    repeats = 0
    total = 0
    for size in (2, 3, 4):
        grams = [cleaned[idx:idx + size] for idx in range(0, len(cleaned) - size + 1)]
        total += len(grams)
        repeats += len(grams) - len(set(grams))
    if total <= 0:
        return 0.5
    return _clamp(1.0 - (repeats / total) * 2.4)


def changed_window_top_letter_control(text: str) -> float:
    cleaned = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if not cleaned:
        return 0.5
    counts: dict[str, int] = {}
    for ch in cleaned:
        counts[ch] = counts.get(ch, 0) + 1
    top = max(counts.values()) / len(cleaned)
    return _clamp(1.0 - max(0.0, top - 0.22) / 0.22)


def average_float(values: list[float], *, default: float = 0.0) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def population_stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = average_float(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def value_range(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def cross_page_edit_consistency_control(pairs: list[tuple[str, str]]) -> float:
    """Reward changed windows that are neither identical nor wholly unrelated."""
    afters = [after for _before, after in pairs if after]
    if len(afters) <= 1:
        return 0.5
    unique_rate = len(set(afters)) / len(afters)
    lengths = [len(after) for after in afters]
    length_control = 1.0 - (population_stdev([float(length) for length in lengths]) / max(1.0, average_float([float(length) for length in lengths])))
    # Repeated identical windows suggest a mechanical word-island artifact.
    # Length stability is tracked separately so full uniqueness is acceptable.
    uniqueness_control = unique_rate
    return _clamp(0.55 * uniqueness_control + 0.45 * length_control)


def add_global_repair_family_features(entries: list[dict[str, Any]]) -> None:
    """Add candidate-set mask-family evidence to global-repair examples.

    The features are ground-truth-free. They summarize how the row's mask
    family performs across sibling candidates from the same source experiment,
    using runtime scores only.
    """
    if not entries:
        return
    by_source: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        source = global_repair_source_key(entry)
        by_source.setdefault(source, []).append(entry)
    for source_entries in by_source.values():
        by_family: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for entry in source_entries:
            by_family.setdefault(global_repair_mask_key(entry.get("row") or {}), []).append(entry)
        total = max(1, len(source_entries))
        for family_entries in by_family.values():
            rows = [entry.get("row") or {} for entry in family_entries]
            family_features = {
                "mask_family_support_control": _clamp(len(family_entries) / total),
                "mask_family_validation_control": _bounded_linear(
                    average_numeric(rows, "page_validation_avg"),
                    low=-1.5,
                    high=5.0,
                ),
                "mask_family_balanced_control": _bounded_linear(
                    average_numeric(rows, "page_balanced_score"),
                    low=-1.5,
                    high=5.0,
                ),
                "mask_family_dictionary_control": _clamp(average_numeric(rows, "page_dict_avg", default=0.5)),
                "mask_family_binary_control": _clamp(average_numeric(rows, "page_binary_component_avg", default=0.5)),
                "mask_family_robust_control": _bounded_linear(
                    average_numeric(rows, "page_robust_score"),
                    low=-1.5,
                    high=5.0,
                ),
            }
            for entry in family_entries:
                example = entry["example"]
                example.features.update(family_features)


def global_repair_source_key(entry: dict[str, Any]) -> str:
    payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
    source = payload.get("source_experiment") or payload.get("source_artifact")
    if source:
        return str(source)
    test_ids = payload.get("test_ids") if isinstance(payload.get("test_ids"), list) else []
    if test_ids:
        return "+".join(str(item) for item in test_ids)
    example = entry.get("example")
    return getattr(example, "group", "__global_repair__")


def global_repair_mask_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in (row.get("mask") or [])))


def average_numeric(rows: list[dict[str, Any]], key: str, *, default: float = 0.0) -> float:
    values = [
        _maybe_float(row.get(key), default)
        for row in rows
        if row.get(key) is not None
    ]
    if not values:
        return default
    return sum(values) / len(values)


def load_positive_text_examples(
    path: Path,
    *,
    language: str,
    chunk_size: int,
    chunk_stride: int,
    min_chars: int,
) -> list[TrainingExample]:
    path = resolve_path(path)
    text = path.read_text(encoding="utf-8")
    cleaned = "".join(ch.upper() for ch in text if ch.isalpha() or ch.isspace())
    examples: list[TrainingExample] = []
    chunk_size = max(min_chars, chunk_size)
    chunk_stride = max(1, chunk_stride)
    for idx, start in enumerate(range(0, max(1, len(cleaned) - min_chars + 1), chunk_stride), start=1):
        chunk = cleaned[start:start + chunk_size]
        if _az_len(chunk) < min_chars:
            continue
        examples.append(make_example(
            text=chunk,
            label=1.0,
            source=f"positive_text:{path.name}:{idx}",
            group=path.stem,
            language=language,
            min_chars=min_chars,
            metadata={"positive_text": str(path), "chunk": idx},
        ))
    return [example for example in examples if example.text]


def make_example(
    *,
    text: str,
    label: float,
    source: str,
    group: str,
    language: str,
    min_chars: int,
    diagnostics: dict[str, Any] | None = None,
    original_length: int | None = None,
    filtered_length: int | None = None,
    mask_size: int = 0,
    metadata: dict[str, Any] | None = None,
    solver_evidence: dict[str, Any] | None = None,
) -> TrainingExample:
    if _az_len(text) < min_chars:
        return TrainingExample("", 0.0, source, group, {}, metadata or {})
    diagnostics = dict(diagnostics or compute_diagnostics(text, language=language))
    features = language_quality_feature_dict(
        text,
        diagnostics=diagnostics,
        language=language,
        original_length=original_length,
        filtered_length=filtered_length,
        mask_size=mask_size,
    )
    if solver_evidence is not None:
        features.update(language_quality_solver_evidence_features(solver_evidence))
    return TrainingExample(
        text=text,
        label=max(0.0, min(1.0, float(label))),
        source=source,
        group=group,
        features=features,
        metadata=metadata or {},
    )


def compute_diagnostics(text: str, *, language: str) -> dict[str, Any]:
    word_list = _word_list(language)
    bin_path = _zenith_native_model_path(language)
    return _automated_candidate_diagnostics(
        text,
        language=language,
        word_list=word_list,
        binary_model_path=bin_path,
    )


def dedupe_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    seen: set[tuple[str, str]] = set()
    deduped: list[TrainingExample] = []
    for example in examples:
        if example.metadata.get("kind") == "global_repair_candidate":
            key = (example.group, example.source)
        else:
            key = (example.group, "".join(ch for ch in example.text.upper() if "A" <= ch <= "Z")[:400])
        if not example.text or key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def synthetic_negative_examples(
    examples: list[TrainingExample],
    *,
    language: str,
    per_positive: int,
    min_chars: int,
) -> list[TrainingExample]:
    """Generate deterministic word-island controls from positive examples."""
    if per_positive <= 0:
        return []
    negatives: list[TrainingExample] = []
    positives = [example for example in examples if example.label >= 0.95]
    for example in positives:
        cleaned = "".join(ch for ch in example.text.upper() if "A" <= ch <= "Z")
        if len(cleaned) < min_chars:
            continue
        for variant in range(1, per_positive + 1):
            text = make_word_island_negative(cleaned, language=language, variant=variant)
            negatives.append(make_example(
                text=text,
                label=0.05,
                source=f"synthetic_negative:{example.source}:{variant}",
                group=example.group,
                language=language,
                min_chars=min_chars,
                metadata={
                    "kind": "synthetic_negative",
                    "parent_source": example.source,
                    "variant": variant,
                },
            ))
    return [example for example in negatives if example.text]


def make_word_island_negative(text: str, *, language: str, variant: int) -> str:
    """Create German-looking fragment soup without preserving sentence order."""
    profile = get_language_scoring_profile(language)
    seed = int(hashlib.sha256(f"{language}:{variant}:{text[:80]}".encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    fragments: list[str] = list(profile.function_overuse_fragments) * 3
    fragments.extend(profile.anchors)
    for size in (3, 4, 5, 6, 7):
        for idx in range(0, max(0, len(text) - size + 1), max(3, size * 2)):
            piece = text[idx:idx + size]
            if len(set(piece)) >= 2:
                fragments.append(piece)
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return text[::-1]
    out: list[str] = []
    target = min(max(700, len(text)), max(900, len(text) + 80))
    while sum(len(item) for item in out) < target:
        fragment = rng.choice(fragments)
        if rng.random() < 0.28:
            fragment = fragment + rng.choice(fragments[: max(1, min(12, len(fragments)))])
        out.append(fragment)
        if rng.random() < 0.20 and out:
            out.append(out[-1])
    return "".join(out)[:target]


def render_report(
    examples: list[TrainingExample],
    model_payload: dict[str, Any],
    output: Path,
    *,
    holdout_examples: list[TrainingExample] | None = None,
    all_examples: list[TrainingExample] | None = None,
    leave_one_group_out_rows: list[dict[str, Any]] | None = None,
) -> str:
    from analysis.language_scoring import LinearLanguageQualityModel

    model = LinearLanguageQualityModel.from_dict(model_payload)
    rows = prediction_rows(examples, model)
    holdout_rows = prediction_rows(holdout_examples or [], model)
    all_rows = prediction_rows(all_examples or examples, model)
    rows_by_error = sorted(rows, key=lambda row: row["error"], reverse=True)
    rows_by_pred = sorted(rows, key=_prediction_sort_key, reverse=True)
    rows_by_label = sorted(rows, key=lambda row: row["label"], reverse=True)
    holdout_metrics = ranking_metrics_by_group(holdout_rows)
    all_metrics = ranking_metrics_by_group(all_rows)
    lines = [
        "# Language Quality Scorer Training Report",
        "",
        f"Model: `{output}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| examples | {len(examples)} |",
        f"| all loaded examples | {len(all_examples or examples)} |",
        f"| holdout examples | {len(holdout_examples or [])} |",
        f"| language | {model.language} |",
        f"| objective | {model.training_summary.get('objective', 'regression')} |",
        f"| pair count | {model.training_summary.get('pair_count', 'n/a')} |",
        f"| training MAE | {model.training_summary.get('training_mae')} |",
        f"| training Pearson | {model.training_summary.get('training_pearson')} |",
        f"| label mean | {model.training_summary.get('label_mean')} |",
        f"| all-groups mean best-label prediction rank | {_format_metric(all_metrics.get('mean_best_label_prediction_rank'))} |",
        f"| holdout mean best-label prediction rank | {_format_metric(holdout_metrics.get('mean_best_label_prediction_rank'))} |",
        f"| holdout top-3 captures | {holdout_metrics.get('top3_captures', 0)}/{holdout_metrics.get('group_count', 0)} |",
        f"| holdout top-5 captures | {holdout_metrics.get('top5_captures', 0)}/{holdout_metrics.get('group_count', 0)} |",
        "",
        "## Weights",
        "",
        "| Feature | Weight |",
        "|---|---:|",
    ]
    for name, weight in model_feature_weight_rows(model):
        lines.append(f"| `{name}` | {weight:+.4f} |")
    lines.extend([
        "",
        "## Top Predicted",
        "",
        "| Rank | Raw | Prediction | Label | Source | Preview |",
        "|---:|---:|---:|---:|---|---|",
    ])
    for rank, row in enumerate(rows_by_pred[:12], start=1):
        lines.append(_row_line(rank, row))
    lines.extend([
        "",
        "## Top Labeled",
        "",
        "| Rank | Raw | Prediction | Label | Source | Preview |",
        "|---:|---:|---:|---:|---|---|",
    ])
    for rank, row in enumerate(rows_by_label[:12], start=1):
        lines.append(_row_line(rank, row))
    lines.extend([
        "",
        "## Largest Errors",
        "",
        "| Rank | Raw | Prediction | Label | Source | Preview |",
        "|---:|---:|---:|---:|---|---|",
    ])
    for rank, row in enumerate(rows_by_error[:12], start=1):
        lines.append(_row_line(rank, row))
    if holdout_rows:
        holdout_candidate_rows = [
            row for row in holdout_rows
            if ":ground_truth" not in str(row.get("source"))
            and "synthetic_negative:" not in str(row.get("source"))
        ]
        lines.extend([
            "",
            "## Holdout Candidate Predictions",
            "",
            "| Rank | Raw | Prediction | Label | Source | Preview |",
            "|---:|---:|---:|---:|---|---|",
        ])
        for rank, row in enumerate(sorted(holdout_candidate_rows, key=_prediction_sort_key, reverse=True)[:16], start=1):
            lines.append(_row_line(rank, row))
        lines.extend([
            "",
            "## Holdout Best-Labeled Candidates",
            "",
            "| Rank | Raw | Prediction | Label | Source | Preview |",
            "|---:|---:|---:|---:|---|---|",
        ])
        for rank, row in enumerate(sorted(holdout_candidate_rows, key=lambda row: row["label"], reverse=True)[:16], start=1):
            lines.append(_row_line(rank, row))
    if leave_one_group_out_rows:
        lines.extend([
            "",
            "## Leave-One-Group-Out",
            "",
            "| Group | Train N | Holdout N | Best-label prediction rank | Top-3 | Top-5 | Pearson | MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in leave_one_group_out_rows:
            lines.append(
                "| {group} | {train_n} | {holdout_n} | {rank} | {top3} | {top5} | {pearson:.3f} | {mae:.3f} |".format(
                    group=row["group"],
                    train_n=row["train_count"],
                    holdout_n=row["holdout_count"],
                    rank=row["best_label_prediction_rank"] or "",
                    top3=1 if row.get("best_label_top3") else 0,
                    top5=1 if row.get("best_label_top5") else 0,
                    pearson=float(row.get("pearson") or 0.0),
                    mae=float(row.get("mae") or 0.0),
                )
            )
    return "\n".join(lines) + "\n"


def model_feature_weight_rows(model: Any) -> list[tuple[str, float]]:
    weights = getattr(model, "weights", None)
    if weights is not None:
        return sorted(zip(model.feature_names, weights), key=lambda item: abs(item[1]), reverse=True)
    trees = getattr(model, "trees", ())
    counts = {name: 0.0 for name in getattr(model, "feature_names", ())}
    for tree in trees:
        collect_tree_feature_counts(tree, counts, getattr(model, "feature_names", ()))
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def collect_tree_feature_counts(
    tree: dict[str, Any],
    counts: dict[str, float],
    feature_names: tuple[str, ...],
) -> None:
    if "leaf" in tree:
        return
    feature_idx = int(tree.get("feature") or 0)
    if 0 <= feature_idx < len(feature_names):
        counts[feature_names[feature_idx]] = counts.get(feature_names[feature_idx], 0.0) + 1.0
    collect_tree_feature_counts(tree.get("left") or {}, counts, feature_names)
    collect_tree_feature_counts(tree.get("right") or {}, counts, feature_names)


def prediction_rows(
    examples: list[TrainingExample],
    model: Any,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        raw_prediction = (
            model.raw_score_features(example.features)
            if hasattr(model, "raw_score_features")
            else model.score_features(example.features)
        )
        prediction = model.score_features(example.features)
        rows.append({
            "source": example.source,
            "group": example.group,
            "label": example.label,
            "raw_prediction": raw_prediction,
            "prediction": prediction,
            "error": abs(prediction - example.label),
            "preview": "".join(ch for ch in example.text.upper() if "A" <= ch <= "Z")[:90],
        })
    return rows


def ranking_metrics_by_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["group"]), []).append(row)
    ranks = []
    top3 = 0
    top5 = 0
    for group_rows in groups.values():
        candidates = [
            row for row in group_rows
            if ":ground_truth" not in str(row.get("source"))
            and "synthetic_negative:" not in str(row.get("source"))
        ]
        if not candidates:
            candidates = group_rows
        by_label = sorted(candidates, key=lambda row: row["label"], reverse=True)
        by_pred = sorted(candidates, key=_prediction_sort_key, reverse=True)
        if not by_label:
            continue
        best = by_label[0]
        rank = by_pred.index(best) + 1 if best in by_pred else None
        if rank is not None:
            ranks.append(rank)
            top3 += 1 if rank <= 3 else 0
            top5 += 1 if rank <= 5 else 0
    return {
        "group_count": len(ranks),
        "mean_best_label_prediction_rank": sum(ranks) / len(ranks) if ranks else None,
        "top3_captures": top3,
        "top5_captures": top5,
    }


def leave_one_group_out_report(
    examples: list[TrainingExample],
    *,
    language: str,
    objective: str,
    model_type: str,
    feature_names: tuple[str, ...],
    l2: float,
    min_label_delta: float,
    max_pairs_per_group: int,
    nonnegative_weights: bool = False,
    nonnegative_iterations: int = 4000,
    nonnegative_learning_rate: float = 0.01,
    gbt_trees: int = 75,
    gbt_depth: int = 3,
    gbt_learning_rate: float = 0.06,
    gbt_min_samples_leaf: int = 2,
) -> list[dict[str, Any]]:
    from analysis.language_scoring import _pearson

    groups = sorted({example.group for example in examples})
    rows: list[dict[str, Any]] = []
    for group in groups:
        train = [example for example in examples if example.group != group]
        holdout = [example for example in examples if example.group == group]
        if len(train) < 4 or not holdout:
            continue
        try:
            model = train_model(
                train,
                language=language,
                objective=objective,
                model_type=model_type,
                feature_names=feature_names,
                l2=l2,
                min_label_delta=min_label_delta,
                max_pairs_per_group=max_pairs_per_group,
                nonnegative_weights=nonnegative_weights,
                nonnegative_iterations=nonnegative_iterations,
                nonnegative_learning_rate=nonnegative_learning_rate,
                gbt_trees=gbt_trees,
                gbt_depth=gbt_depth,
                gbt_learning_rate=gbt_learning_rate,
                gbt_min_samples_leaf=gbt_min_samples_leaf,
            )
        except ValueError:
            continue
        pred_rows = prediction_rows(holdout, model)
        candidates = [
            row for row in pred_rows
            if ":ground_truth" not in str(row.get("source"))
            and "synthetic_negative:" not in str(row.get("source"))
        ]
        if not candidates:
            candidates = pred_rows
        by_label = sorted(candidates, key=lambda row: row["label"], reverse=True)
        by_pred = sorted(candidates, key=_prediction_sort_key, reverse=True)
        best = by_label[0] if by_label else None
        rank = by_pred.index(best) + 1 if best in by_pred else None
        labels = [float(row["label"]) for row in candidates]
        preds = [float(row.get("raw_prediction", row["prediction"])) for row in candidates]
        mae = sum(abs(p - y) for p, y in zip(preds, labels)) / len(labels) if labels else 0.0
        rows.append({
            "group": group,
            "train_count": len(train),
            "holdout_count": len(holdout),
            "best_label_prediction_rank": rank,
            "best_label_top3": rank is not None and rank <= 3,
            "best_label_top5": rank is not None and rank <= 5,
            "pearson": _pearson(preds, labels),
            "mae": mae,
        })
    return rows


def train_model(
    examples: list[TrainingExample],
    *,
    language: str,
    objective: str,
    model_type: str = "linear",
    feature_names: tuple[str, ...] = LANGUAGE_QUALITY_FEATURES,
    l2: float,
    min_label_delta: float,
    max_pairs_per_group: int,
    nonnegative_weights: bool = False,
    nonnegative_iterations: int = 4000,
    nonnegative_learning_rate: float = 0.01,
    gbt_trees: int = 75,
    gbt_depth: int = 3,
    gbt_learning_rate: float = 0.06,
    gbt_min_samples_leaf: int = 2,
) -> Any:
    model_rows = [
        {
            "features": example.features,
            "label": example.label,
            "group": example.group,
        }
        for example in examples
    ]
    if model_type == "gbt":
        return train_gradient_boosted_language_quality_model(
            model_rows,
            language=language,
            feature_names=feature_names,
            n_estimators=gbt_trees,
            max_depth=gbt_depth,
            learning_rate=gbt_learning_rate,
            min_samples_leaf=gbt_min_samples_leaf,
        )
    if objective == "pairwise":
        return train_pairwise_language_quality_model(
            model_rows,
            language=language,
            feature_names=feature_names,
            l2=l2,
            min_label_delta=min_label_delta,
            max_pairs_per_group=max_pairs_per_group,
            nonnegative_weights=nonnegative_weights,
            nonnegative_iterations=nonnegative_iterations,
            nonnegative_learning_rate=nonnegative_learning_rate,
        )
    return train_linear_language_quality_model(
        model_rows,
        language=language,
        feature_names=feature_names,
        l2=l2,
    )


def feature_names_for_mode(mode: str) -> tuple[str, ...]:
    """Return a named feature subset for calibration experiments."""
    excluded: set[str] = set()
    if mode in {"no_solver", "text_only"}:
        excluded.update({
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
        })
    if mode == "text_only":
        excluded.update({
            "mask_size_control",
            "deletion_control",
            # Concentration/diversity is already represented in the scalar
            # validation layer. Let the trained text-only model focus on
            # phrase quality and word-island plausibility instead of learning
            # to prefer smoother-looking but semantically weaker basins.
            "letter_diversity",
            "top_letter_control",
        })
    return tuple(name for name in LANGUAGE_QUALITY_FEATURES if name not in excluded)


def _row_line(rank: int, row: dict[str, Any]) -> str:
    return (
        f"| {rank} | {row.get('raw_prediction', row['prediction']):.3f} | "
        f"{row['prediction']:.3f} | {row['label']:.3f} | "
        f"{row['source']} | {row['preview']} |"
    )


def _prediction_sort_key(row: dict[str, Any]) -> float:
    return float(row.get("raw_prediction", row.get("prediction", 0.0)))


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _maybe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _bounded_linear(value: float, *, low: float, high: float) -> float:
    if high <= low:
        return 0.5
    return _clamp((float(value) - low) / (high - low))


def expand_paths(path: Path, pattern: str) -> list[Path]:
    path = resolve_path(path)
    if path.is_dir():
        return sorted(path.rglob(pattern))
    return [path]


def expand_repair_paths(path: Path) -> list[Path]:
    path = resolve_path(path)
    if not path.is_dir():
        return [path]
    patterns = (
        "*global_repair*.json",
        "*word_hypothesis_repair*.json",
    )
    seen: dict[Path, None] = {}
    for pattern in patterns:
        for item in sorted(path.rglob(pattern)):
            seen[item] = None
    return list(seen)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _az_len(text: str) -> int:
    return sum(1 for ch in text.upper() if "A" <= ch <= "Z")


if __name__ == "__main__":
    main()
