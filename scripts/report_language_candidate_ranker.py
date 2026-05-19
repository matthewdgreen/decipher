#!/usr/bin/env python3
"""Evaluate a language-quality ranker on held-out solver finalist menus.

This is an offline calibration report. It may use solved benchmark plaintext
to label historical finalist candidates, but the trained model it emits only
consumes ground-truth-free features at runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_language_quality_scorer import (  # noqa: E402
    TrainingExample,
    dedupe_examples,
    load_artifact_examples,
    load_global_repair_examples,
    load_probe_examples,
    feature_names_for_mode,
    resolve_path,
    train_model,
)

POLICY_FEATURES = {
    "validation": "validation_score_control",
    "ensemble": "ensemble_score_control",
    "selection": "selection_score_control",
    "dictionary": "dict_rate",
    "language_quality": "language_coherence",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report leave-one-group-out finalist-menu ranking quality."
    )
    parser.add_argument("--language", default="de")
    parser.add_argument("--artifact", action="append", default=[], help="Artifact file or directory to mine.")
    parser.add_argument("--probe-jsonl", action="append", default=[], help="Probe JSONL file to mine.")
    parser.add_argument(
        "--global-repair-json",
        action="append",
        default=[],
        help="Global repair probe JSON file or directory to mine as candidate menus.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="candidate_ranker.json")
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--objective", choices=("regression", "pairwise"), default="pairwise")
    parser.add_argument("--feature-set", choices=("all", "no_solver", "text_only"), default="all")
    parser.add_argument(
        "--holdout-group-by",
        choices=("group", "source_experiment", "source_artifact", "test_set", "section", "label"),
        default="group",
        help=(
            "Unit for leave-one-out evaluation. group holds out one menu; "
            "source_experiment/source_artifact/test_set create harsher clustered holdouts."
        ),
    )
    parser.add_argument(
        "--training-group-by",
        choices=("group", "source_experiment", "source_artifact", "test_set", "section", "label"),
        default="group",
        help=(
            "Within-group unit used to create pairwise training preferences. "
            "Use source_experiment to let the model compare sibling repair menus."
        ),
    )
    parser.add_argument(
        "--two-stage-mask-family",
        action="store_true",
        help=(
            "Also train/evaluate a two-stage ranker: choose a mask family first, "
            "then rank edits inside that family."
        ),
    )
    parser.add_argument(
        "--two-stage-edit-group-by",
        choices=("training_group", "mask_family"),
        default="training_group",
        help=(
            "Pairwise grouping for the second-stage edit ranker. "
            "mask_family compares edits only within the same held-out training unit and mask."
        ),
    )
    parser.add_argument(
        "--two-stage-family-top-k",
        type=int,
        default=1,
        help="Allow the edit ranker to choose among the top K mask families.",
    )
    parser.add_argument(
        "--two-stage-review-shortlist-k",
        type=int,
        default=5,
        help=(
            "Report whether the best held-out candidate appears in the first K "
            "two-stage candidates. This is a conservative review/agent menu metric, "
            "not an automatic adoption policy."
        ),
    )
    parser.add_argument("--l2", type=float, default=0.5)
    parser.add_argument("--min-label-delta", type=float, default=0.02)
    parser.add_argument("--max-pairs-per-group", type=int, default=2000)
    parser.add_argument(
        "--nonnegative-weights",
        action="store_true",
        help="For pairwise training, constrain quality/control feature weights to be non-negative.",
    )
    parser.add_argument("--nonnegative-iterations", type=int, default=4000)
    parser.add_argument("--nonnegative-learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="Include clean ground-truth/positive examples; default is finalist candidates only.",
    )
    args = parser.parse_args()

    examples: list[TrainingExample] = []
    for path in args.artifact:
        examples.extend(load_artifact_examples(Path(path), language=args.language, min_chars=args.min_chars))
    for path in args.probe_jsonl:
        examples.extend(load_probe_examples(Path(path), language=args.language, min_chars=args.min_chars))
    for path in args.global_repair_json:
        examples.extend(load_global_repair_examples(Path(path), language=args.language))
    examples = dedupe_examples(examples)
    if not args.include_clean:
        examples = [
            example for example in examples
            if (
                ":top_finalist:" in example.source
                or example.source.startswith("probe:")
                or example.source.startswith("global_repair:")
            )
            and ":ground_truth" not in example.source
            and example.metadata.get("kind") != "synthetic_negative"
        ]
    if len(examples) < 4:
        raise SystemExit("Need at least four finalist examples to evaluate a ranker.")
    groups = sorted({evaluation_group(example, args.holdout_group_by) for example in examples})
    if len(groups) < 2:
        raise SystemExit("Need at least two groups/pages for leave-one-group-out evaluation.")

    output_dir = resolve_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    full_model = train_model(
        regroup_examples(examples, args.training_group_by),
        language=args.language,
        objective=args.objective,
        feature_names=feature_names_for_mode(args.feature_set),
        l2=args.l2,
        min_label_delta=args.min_label_delta,
        max_pairs_per_group=args.max_pairs_per_group,
        nonnegative_weights=args.nonnegative_weights,
        nonnegative_iterations=args.nonnegative_iterations,
        nonnegative_learning_rate=args.nonnegative_learning_rate,
    )
    model_path = output_dir / args.model_name
    full_model.save(model_path)

    group_reports = []
    for group in groups:
        train = [
            example for example in examples
            if evaluation_group(example, args.holdout_group_by) != group
        ]
        holdout = [
            example for example in examples
            if evaluation_group(example, args.holdout_group_by) == group
        ]
        if len(train) < 4 or len(holdout) < 2:
            continue
        try:
            model = train_model(
                regroup_examples(train, args.training_group_by),
                language=args.language,
                objective=args.objective,
                feature_names=feature_names_for_mode(args.feature_set),
                l2=args.l2,
                min_label_delta=args.min_label_delta,
                max_pairs_per_group=args.max_pairs_per_group,
                nonnegative_weights=args.nonnegative_weights,
                nonnegative_iterations=args.nonnegative_iterations,
                nonnegative_learning_rate=args.nonnegative_learning_rate,
            )
        except ValueError as exc:
            group_reports.append({
                "group": group,
                "status": "skipped",
                "reason": str(exc),
                "train_count": len(train),
                "holdout_count": len(holdout),
            })
            continue
        predictions = prediction_rows(holdout, model)
        two_stage_report = None
        if args.two_stage_mask_family:
            try:
                family_model = train_model(
                    build_mask_family_examples(train, args.training_group_by),
                    language=args.language,
                    objective=args.objective,
                    feature_names=feature_names_for_mode(args.feature_set),
                    l2=args.l2,
                    min_label_delta=args.min_label_delta,
                    max_pairs_per_group=args.max_pairs_per_group,
                    nonnegative_weights=args.nonnegative_weights,
                    nonnegative_iterations=args.nonnegative_iterations,
                    nonnegative_learning_rate=args.nonnegative_learning_rate,
                )
                edit_training_examples = (
                    build_mask_family_edit_examples(train, args.training_group_by)
                    if args.two_stage_edit_group_by == "mask_family"
                    else regroup_examples(train, args.training_group_by)
                )
                edit_model = train_model(
                    edit_training_examples,
                    language=args.language,
                    objective=args.objective,
                    feature_names=feature_names_for_mode(args.feature_set),
                    l2=args.l2,
                    min_label_delta=args.min_label_delta,
                    max_pairs_per_group=args.max_pairs_per_group,
                    nonnegative_weights=args.nonnegative_weights,
                    nonnegative_iterations=args.nonnegative_iterations,
                    nonnegative_learning_rate=args.nonnegative_learning_rate,
                )
                two_stage_report = two_stage_mask_family_report(
                    predictions,
                    family_model=family_model,
                    candidate_model=edit_model,
                    family_top_k=args.two_stage_family_top_k,
                    review_shortlist_k=args.two_stage_review_shortlist_k,
                )
            except ValueError as exc:
                two_stage_report = {"status": "skipped", "reason": str(exc)}
        group_reports.append(
            group_report(
                group,
                train,
                holdout,
                predictions,
                model,
                two_stage_report=two_stage_report,
            )
        )

    summary = summarize_group_reports(group_reports)
    payload = {
        "language": args.language,
        "objective": args.objective,
        "feature_set": args.feature_set,
        "holdout_group_by": args.holdout_group_by,
        "training_group_by": args.training_group_by,
        "two_stage_mask_family": args.two_stage_mask_family,
        "two_stage_edit_group_by": args.two_stage_edit_group_by,
        "two_stage_family_top_k": args.two_stage_family_top_k,
        "two_stage_review_shortlist_k": args.two_stage_review_shortlist_k,
        "model_path": str(model_path),
        "example_count": len(examples),
        "group_count": len(groups),
        "training_summary": full_model.training_summary,
        "summary": summary,
        "groups": group_reports,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_markdown(payload)
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {model_path}")


def prediction_rows(examples: list[TrainingExample], model: Any) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        raw_score = model.raw_score_features(example.features)
        rows.append({
            "source": example.source,
            "group": example.group,
            "label": example.label,
            "raw_score": raw_score,
            "score": model.score_features(example.features),
            "preview": "".join(ch for ch in example.text.upper() if "A" <= ch <= "Z")[:120],
            "mask": example.metadata.get("mask") or [],
            "features": dict(example.features),
            "metadata": example.metadata,
        })
    return rows


def evaluation_group(example: TrainingExample, mode: str) -> str:
    """Return the leave-one-out unit for an example."""
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    if mode == "group":
        return example.group
    if mode == "test_set":
        test_ids = metadata.get("test_ids") if isinstance(metadata.get("test_ids"), list) else []
        if test_ids:
            return "+".join(str(item) for item in test_ids)
    value = metadata.get(mode)
    if value:
        return str(value)
    return example.group


def regroup_examples(examples: list[TrainingExample], mode: str) -> list[TrainingExample]:
    """Return shallow copies with training groups set to the requested unit."""
    if mode == "group":
        return examples
    return [
        TrainingExample(
            text=example.text,
            label=example.label,
            source=example.source,
            group=evaluation_group(example, mode),
            features=example.features,
            metadata=example.metadata,
        )
        for example in examples
    ]


def build_mask_family_examples(examples: list[TrainingExample], training_group_by: str) -> list[TrainingExample]:
    """Collapse candidate rows into mask-family rows for first-stage training."""
    buckets: dict[tuple[str, tuple[str, ...]], list[TrainingExample]] = {}
    for example in examples:
        group = evaluation_group(example, training_group_by)
        mask = mask_key_from_example(example)
        buckets.setdefault((group, mask), []).append(example)
    family_examples: list[TrainingExample] = []
    for (group, mask), rows in buckets.items():
        best = max(rows, key=lambda row: row.label)
        features = aggregate_feature_dict(rows)
        source = f"mask_family:{group}:{','.join(mask) or '(none)'}"
        family_examples.append(TrainingExample(
            text=best.text,
            label=best.label,
            source=source,
            group=group,
            features=features,
            metadata={
                "kind": "mask_family_candidate",
                "mask": list(mask),
                "member_count": len(rows),
                "best_source": best.source,
            },
        ))
    return family_examples


def build_mask_family_edit_examples(examples: list[TrainingExample], training_group_by: str) -> list[TrainingExample]:
    """Return examples grouped for within-mask-family edit training."""
    return [
        TrainingExample(
            text=example.text,
            label=example.label,
            source=example.source,
            group=f"{evaluation_group(example, training_group_by)}::{','.join(mask_key_from_example(example)) or '(none)'}",
            features=example.features,
            metadata=example.metadata,
        )
        for example in examples
    ]


def aggregate_feature_dict(rows: list[TrainingExample]) -> dict[str, float]:
    """Average feature values across examples, retaining max support signals."""
    names = sorted({name for row in rows for name in row.features})
    features: dict[str, float] = {}
    for name in names:
        values = [float(row.features.get(name) or 0.0) for row in rows]
        if name.startswith("mask_family_"):
            features[name] = max(values) if values else 0.0
        else:
            features[name] = sum(values) / len(values) if values else 0.0
    return features


def mask_key_from_example(example: TrainingExample) -> tuple[str, ...]:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    return tuple(sorted(str(item) for item in (metadata.get("mask") or [])))


def group_report(
    group: str,
    train: list[TrainingExample],
    holdout: list[TrainingExample],
    predictions: list[dict[str, Any]],
    model: Any,
    two_stage_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_label = sorted(predictions, key=lambda row: float(row["label"]), reverse=True)
    by_score = sorted(predictions, key=lambda row: float(row["raw_score"]), reverse=True)
    best = by_label[0]
    top_predicted = by_score[0]
    rank = by_score.index(best) + 1
    top_predicted_label_gap = max(
        0.0,
        float(best.get("label") or 0.0) - float(top_predicted.get("label") or 0.0),
    )
    policy_ranks = policy_rank_report(best, predictions)
    report = {
        "group": group,
        "status": "completed",
        "train_count": len(train),
        "holdout_count": len(holdout),
        "pair_count": model.training_summary.get("pair_count"),
        "best_label_rank": rank,
        "top_predicted_label_gap": round(top_predicted_label_gap, 6),
        "top3": rank <= 3,
        "top5": rank <= 5,
        "policy_ranks": policy_ranks,
        "best_label": compact_prediction(best),
        "top_predicted": [compact_prediction(row) for row in by_score[:8]],
        "top_labeled": [compact_prediction(row) for row in by_label[:8]],
        "feature_deltas": feature_delta_rows(
            by_score[0],
            best,
            predictions,
            feature_names=tuple(model.feature_names),
        ),
    }
    if two_stage_report is not None:
        report["two_stage_mask_family"] = two_stage_report
    return report


def two_stage_mask_family_report(
    predictions: list[dict[str, Any]],
    *,
    family_model: Any,
    candidate_model: Any,
    family_top_k: int = 1,
    review_shortlist_k: int = 5,
) -> dict[str, Any]:
    """Rank holdout candidates by family score first, edit score second."""
    if not predictions:
        return {"status": "skipped", "reason": "no predictions"}
    by_label = sorted(predictions, key=lambda row: float(row["label"]), reverse=True)
    best = by_label[0]
    family_rows = mask_family_prediction_rows(predictions, family_model)
    family_rank = {
        row["mask_key"]: rank
        for rank, row in enumerate(sorted(family_rows, key=lambda row: float(row["raw_score"]), reverse=True), start=1)
    }
    top_k = max(1, int(family_top_k))
    ordered = sorted(
        predictions,
        key=lambda row: (
            0 if family_rank.get(mask_key_from_prediction(row), 10**9) <= top_k else 1,
            -candidate_model.raw_score_features(row.get("features") or {}),
            family_rank.get(mask_key_from_prediction(row), 10**9),
        ),
    )
    top = ordered[0]
    rank = ordered.index(best) + 1
    gap = max(0.0, float(best.get("label") or 0.0) - float(top.get("label") or 0.0))
    shortlist_k = max(1, int(review_shortlist_k))
    shortlist = ordered[:shortlist_k]
    shortlist_best = max(shortlist, key=lambda row: float(row.get("label") or 0.0))
    shortlist_best_gap = max(
        0.0,
        float(best.get("label") or 0.0) - float(shortlist_best.get("label") or 0.0),
    )
    shortlist_best_rank = ordered.index(shortlist_best) + 1
    best_in_shortlist = rank <= shortlist_k
    diverse_shortlist = diverse_review_shortlist(
        ordered,
        family_rank=family_rank,
        family_top_k=top_k,
        limit=shortlist_k,
    )
    diverse_shortlist_best = max(diverse_shortlist, key=lambda row: float(row.get("label") or 0.0))
    diverse_shortlist_best_gap = max(
        0.0,
        float(best.get("label") or 0.0) - float(diverse_shortlist_best.get("label") or 0.0),
    )
    diverse_best_in_shortlist = best in diverse_shortlist
    top_family = next(
        (row for row in family_rows if row["mask_key"] == mask_key_from_prediction(top)),
        {},
    )
    best_family = next(
        (row for row in family_rows if row["mask_key"] == mask_key_from_prediction(best)),
        {},
    )
    ranked_families = sorted(family_rows, key=lambda row: float(row["raw_score"]), reverse=True)
    best_family_rank = (
        ranked_families.index(best_family) + 1
        if best_family in ranked_families
        else None
    )
    return {
        "status": "completed",
        "best_label_rank": rank,
        "best_family_rank": best_family_rank,
        "top_predicted_label_gap": round(gap, 6),
        "review_shortlist_k": shortlist_k,
        "review_shortlist_contains_best": best_in_shortlist,
        "review_shortlist_best_label_gap": round(shortlist_best_gap, 6),
        "review_shortlist_best_rank": shortlist_best_rank,
        "review_shortlist_best": compact_prediction(shortlist_best),
        "review_shortlist": [compact_prediction(row) for row in shortlist],
        "diverse_review_shortlist_contains_best": diverse_best_in_shortlist,
        "diverse_review_shortlist_best_label_gap": round(diverse_shortlist_best_gap, 6),
        "diverse_review_shortlist_best": compact_prediction(diverse_shortlist_best),
        "diverse_review_shortlist": [compact_prediction(row) for row in diverse_shortlist],
        "top3": rank <= 3,
        "top5": rank <= 5,
        "top_predicted": compact_prediction(top),
        "best_label": compact_prediction(best),
        "top_predicted_edit_raw_score": round(
            float(candidate_model.raw_score_features(top.get("features") or {})),
            6,
        ),
        "top_family": compact_family_prediction(top_family),
        "best_family": compact_family_prediction(best_family),
        "top_families": [compact_family_prediction(row) for row in ranked_families[:5]],
        "family_count": len(family_rows),
        "family_top_k": top_k,
        "candidate_model_feature_count": len(getattr(candidate_model, "feature_names", ()) or ()),
        "family_model_feature_count": len(getattr(family_model, "feature_names", ()) or ()),
    }


def diverse_review_shortlist(
    ordered: list[dict[str, Any]],
    *,
    family_rank: dict[tuple[str, ...], int],
    family_top_k: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Build a review menu that cannot be monopolized by one mask family."""
    if not ordered:
        return []
    limit = max(1, int(limit))
    family_top_k = max(1, int(family_top_k))
    selected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    eligible_families = [
        family
        for family, rank in sorted(family_rank.items(), key=lambda item: item[1])
        if rank <= family_top_k
    ]
    for family in eligible_families:
        candidate = next(
            (row for row in ordered if mask_key_from_prediction(row) == family),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            seen_sources.add(str(candidate.get("source")))
        if len(selected) >= limit:
            return selected
    for row in ordered:
        source = str(row.get("source"))
        if source in seen_sources:
            continue
        selected.append(row)
        seen_sources.add(source)
        if len(selected) >= limit:
            break
    return selected


def mask_family_prediction_rows(predictions: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in predictions:
        buckets.setdefault(mask_key_from_prediction(row), []).append(row)
    family_rows = []
    for mask, rows in buckets.items():
        features = aggregate_prediction_feature_dict(rows)
        raw_score = model.raw_score_features(features)
        best_label = max(float(row.get("label") or 0.0) for row in rows)
        family_rows.append({
            "mask_key": mask,
            "member_count": len(rows),
            "raw_score": raw_score,
            "score": model.score_features(features),
            "best_label": best_label,
            "top_candidate_label": max(float(row.get("label") or 0.0) for row in rows),
            "features": features,
        })
    return family_rows


def aggregate_prediction_feature_dict(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = sorted({
        name
        for row in rows
        if isinstance(row.get("features"), dict)
        for name in row["features"]
    })
    features: dict[str, float] = {}
    for name in names:
        values = [float((row.get("features") or {}).get(name) or 0.0) for row in rows]
        if name.startswith("mask_family_"):
            features[name] = max(values) if values else 0.0
        else:
            features[name] = sum(values) / len(values) if values else 0.0
    return features


def mask_key_from_prediction(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in (row.get("mask") or [])))


def policy_rank_report(best: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank the best-labeled candidate under simple runtime-only policies."""
    rows: dict[str, Any] = {}
    for policy, feature in POLICY_FEATURES.items():
        eligible = [
            row for row in predictions
            if isinstance(row.get("features"), dict)
        ]
        if not eligible:
            continue
        ranked = sorted(
            eligible,
            key=lambda row: (
                float((row.get("features") or {}).get(feature) or 0.0),
                float((row.get("features") or {}).get("dict_rate") or 0.0),
            ),
            reverse=True,
        )
        try:
            rank = ranked.index(best) + 1
        except ValueError:
            continue
        top = ranked[0] if ranked else {}
        rows[policy] = {
            "feature": feature,
            "best_label_rank": rank,
            "top_label": round(float(top.get("label") or 0.0), 6),
            "top_mask": top.get("mask") or [],
            "top_source": top.get("source"),
            "top_score": round(float((top.get("features") or {}).get(feature) or 0.0), 6),
        }
    return rows


def compact_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": row.get("source"),
        "label": round(float(row.get("label") or 0.0), 6),
        "raw_score": round(float(row.get("raw_score") or 0.0), 6),
        "score": round(float(row.get("score") or 0.0), 6),
        "mask": row.get("mask") or [],
        "preview": row.get("preview"),
    }


def compact_family_prediction(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "mask": list(row.get("mask_key") or ()),
        "member_count": row.get("member_count"),
        "best_label": round(float(row.get("best_label") or 0.0), 6),
        "raw_score": round(float(row.get("raw_score") or 0.0), 6),
        "score": round(float(row.get("score") or 0.0), 6),
    }


def feature_delta_rows(
    predicted: dict[str, Any],
    labeled: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Explain how the top-predicted candidate differs from the best-labeled one."""
    pred_features = predicted.get("features") if isinstance(predicted.get("features"), dict) else {}
    label_features = labeled.get("features") if isinstance(labeled.get("features"), dict) else {}
    if not pred_features or not label_features:
        return []
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    names = tuple(feature_names or sorted(set(pred_features) | set(label_features)))
    for name in names:
        values = [
            float((row.get("features") or {}).get(name) or 0.0)
            for row in predictions
            if isinstance(row.get("features"), dict)
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means[name] = mean
        scales[name] = variance ** 0.5 or 1.0
    rows = []
    for name in names:
        pred_value = float(pred_features.get(name) or 0.0)
        label_value = float(label_features.get(name) or 0.0)
        delta = pred_value - label_value
        z_delta = delta / scales.get(name, 1.0)
        rows.append({
            "feature": name,
            "top_predicted": round(pred_value, 6),
            "best_labeled": round(label_value, 6),
            "delta": round(delta, 6),
            "z_delta": round(z_delta, 6),
        })
    rows.sort(key=lambda row: abs(float(row["z_delta"])), reverse=True)
    return rows[:limit]


def summarize_group_reports(groups: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [group for group in groups if group.get("status") == "completed"]
    ranks = [int(group["best_label_rank"]) for group in completed]
    gaps = [float(group.get("top_predicted_label_gap") or 0.0) for group in completed]
    policy_summary: dict[str, dict[str, Any]] = {}
    for policy in POLICY_FEATURES:
        policy_ranks = [
            int((group.get("policy_ranks") or {}).get(policy, {}).get("best_label_rank"))
            for group in completed
            if (group.get("policy_ranks") or {}).get(policy, {}).get("best_label_rank") is not None
        ]
        if not policy_ranks:
            continue
        policy_summary[policy] = {
            "completed_group_count": len(policy_ranks),
            "mean_best_label_rank": round(sum(policy_ranks) / len(policy_ranks), 4),
            "top1_captures": sum(1 for rank in policy_ranks if rank <= 1),
            "top3_captures": sum(1 for rank in policy_ranks if rank <= 3),
            "top5_captures": sum(1 for rank in policy_ranks if rank <= 5),
        }
    summary = {
        "completed_group_count": len(completed),
        "mean_best_label_rank": round(sum(ranks) / len(ranks), 4) if ranks else None,
        "mean_top_predicted_label_gap": round(sum(gaps) / len(gaps), 6) if gaps else None,
        "top_predicted_within_001": sum(1 for gap in gaps if gap <= 0.001),
        "top_predicted_within_005": sum(1 for gap in gaps if gap <= 0.005),
        "top_predicted_within_010": sum(1 for gap in gaps if gap <= 0.010),
        "top1_captures": sum(1 for rank in ranks if rank <= 1),
        "top3_captures": sum(1 for rank in ranks if rank <= 3),
        "top5_captures": sum(1 for rank in ranks if rank <= 5),
        "policy_summary": policy_summary,
    }
    two_stage = summarize_two_stage_reports(completed)
    if two_stage:
        summary["two_stage_mask_family"] = two_stage
    return summary


def summarize_two_stage_reports(groups: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        group.get("two_stage_mask_family")
        for group in groups
        if isinstance(group.get("two_stage_mask_family"), dict)
        and group["two_stage_mask_family"].get("status") == "completed"
    ]
    if not rows:
        return {}
    ranks = [int(row["best_label_rank"]) for row in rows]
    family_ranks = [
        int(row["best_family_rank"])
        for row in rows
        if row.get("best_family_rank") is not None
    ]
    gaps = [float(row.get("top_predicted_label_gap") or 0.0) for row in rows]
    shortlist_gaps = [float(row.get("review_shortlist_best_label_gap") or 0.0) for row in rows]
    shortlist_hits = [bool(row.get("review_shortlist_contains_best")) for row in rows]
    diverse_shortlist_gaps = [float(row.get("diverse_review_shortlist_best_label_gap") or 0.0) for row in rows]
    diverse_shortlist_hits = [bool(row.get("diverse_review_shortlist_contains_best")) for row in rows]
    return {
        "completed_group_count": len(rows),
        "mean_best_label_rank": round(sum(ranks) / len(ranks), 4),
        "mean_best_family_rank": round(sum(family_ranks) / len(family_ranks), 4) if family_ranks else None,
        "family_top1_captures": sum(1 for rank in family_ranks if rank <= 1),
        "family_top3_captures": sum(1 for rank in family_ranks if rank <= 3),
        "mean_top_predicted_label_gap": round(sum(gaps) / len(gaps), 6),
        "top_predicted_within_001": sum(1 for gap in gaps if gap <= 0.001),
        "top_predicted_within_005": sum(1 for gap in gaps if gap <= 0.005),
        "top_predicted_within_010": sum(1 for gap in gaps if gap <= 0.010),
        "top1_captures": sum(1 for rank in ranks if rank <= 1),
        "top3_captures": sum(1 for rank in ranks if rank <= 3),
        "top5_captures": sum(1 for rank in ranks if rank <= 5),
        "review_shortlist_k": rows[0].get("review_shortlist_k"),
        "review_shortlist_contains_best": sum(1 for hit in shortlist_hits if hit),
        "mean_review_shortlist_best_label_gap": round(sum(shortlist_gaps) / len(shortlist_gaps), 6),
        "review_shortlist_within_001": sum(1 for gap in shortlist_gaps if gap <= 0.001),
        "review_shortlist_within_005": sum(1 for gap in shortlist_gaps if gap <= 0.005),
        "review_shortlist_within_010": sum(1 for gap in shortlist_gaps if gap <= 0.010),
        "diverse_review_shortlist_contains_best": sum(1 for hit in diverse_shortlist_hits if hit),
        "mean_diverse_review_shortlist_best_label_gap": round(
            sum(diverse_shortlist_gaps) / len(diverse_shortlist_gaps),
            6,
        ),
        "diverse_review_shortlist_within_001": sum(1 for gap in diverse_shortlist_gaps if gap <= 0.001),
        "diverse_review_shortlist_within_005": sum(1 for gap in diverse_shortlist_gaps if gap <= 0.005),
        "diverse_review_shortlist_within_010": sum(1 for gap in diverse_shortlist_gaps if gap <= 0.010),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Language Candidate Ranker Report",
        "",
        f"Model: `{payload['model_path']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| language | {payload['language']} |",
        f"| objective | {payload['objective']} |",
        f"| feature set | {payload.get('feature_set', 'all')} |",
        f"| holdout group by | {payload.get('holdout_group_by', 'group')} |",
        f"| training group by | {payload.get('training_group_by', 'group')} |",
        f"| two-stage mask family | {1 if payload.get('two_stage_mask_family') else 0} |",
        f"| two-stage edit group by | {payload.get('two_stage_edit_group_by', 'training_group')} |",
        f"| two-stage family top-k | {payload.get('two_stage_family_top_k', 1)} |",
        f"| two-stage review shortlist-k | {payload.get('two_stage_review_shortlist_k', 5)} |",
        f"| examples | {payload['example_count']} |",
        f"| groups | {payload['group_count']} |",
        f"| completed groups | {summary.get('completed_group_count')} |",
        f"| mean best-label rank | {summary.get('mean_best_label_rank')} |",
        f"| mean top-predicted label gap | {format_percent(summary.get('mean_top_predicted_label_gap'))} |",
        f"| top predicted within 0.1% | {summary.get('top_predicted_within_001')} |",
        f"| top predicted within 0.5% | {summary.get('top_predicted_within_005')} |",
        f"| top predicted within 1.0% | {summary.get('top_predicted_within_010')} |",
        f"| top-1 captures | {summary.get('top1_captures')} |",
        f"| top-3 captures | {summary.get('top3_captures')} |",
        f"| top-5 captures | {summary.get('top5_captures')} |",
        f"| full-model pair count | {payload['training_summary'].get('pair_count', 'n/a')} |",
        "",
        "## Two-Stage Mask-Family Ranker",
        "",
    ]
    two_stage = summary.get("two_stage_mask_family") if isinstance(summary.get("two_stage_mask_family"), dict) else {}
    if two_stage:
        lines.extend([
            "| Metric | Value |",
            "|---|---:|",
            f"| completed groups | {two_stage.get('completed_group_count')} |",
            f"| mean best-label rank | {two_stage.get('mean_best_label_rank')} |",
            f"| mean best-family rank | {two_stage.get('mean_best_family_rank')} |",
            f"| family top-1 captures | {two_stage.get('family_top1_captures')} |",
            f"| family top-3 captures | {two_stage.get('family_top3_captures')} |",
            f"| mean top-predicted label gap | {format_percent(two_stage.get('mean_top_predicted_label_gap'))} |",
            f"| top predicted within 0.1% | {two_stage.get('top_predicted_within_001')} |",
            f"| top predicted within 0.5% | {two_stage.get('top_predicted_within_005')} |",
            f"| top predicted within 1.0% | {two_stage.get('top_predicted_within_010')} |",
            f"| top-1 captures | {two_stage.get('top1_captures')} |",
            f"| top-3 captures | {two_stage.get('top3_captures')} |",
            f"| top-5 captures | {two_stage.get('top5_captures')} |",
            f"| review shortlist K | {two_stage.get('review_shortlist_k')} |",
            f"| review shortlist contains exact best | {two_stage.get('review_shortlist_contains_best')} |",
            f"| review shortlist mean best-label gap | {format_percent(two_stage.get('mean_review_shortlist_best_label_gap'))} |",
            f"| review shortlist within 0.5% | {two_stage.get('review_shortlist_within_005')} |",
            f"| diverse review contains exact best | {two_stage.get('diverse_review_shortlist_contains_best')} |",
            f"| diverse review mean best-label gap | {format_percent(two_stage.get('mean_diverse_review_shortlist_best_label_gap'))} |",
            f"| diverse review within 0.5% | {two_stage.get('diverse_review_shortlist_within_005')} |",
        ])
    else:
        lines.append("Not enabled or no completed two-stage groups.")
    lines.extend([
        "",
        "## Policy Baselines",
        "",
        "These simple policies use the same ground-truth-free feature packet as the model.",
        "",
        "| Policy | Mean Best Rank | Top-1 | Top-3 | Top-5 | Groups |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for policy, row in sorted((summary.get("policy_summary") or {}).items()):
        lines.append(
            "| {policy} | {mean} | {top1} | {top3} | {top5} | {groups} |".format(
                policy=policy,
                mean=row.get("mean_best_label_rank"),
                top1=row.get("top1_captures"),
                top3=row.get("top3_captures"),
                top5=row.get("top5_captures"),
                groups=row.get("completed_group_count"),
            )
        )
    lines.extend([
        "",
        "## Leave-One-Group-Out",
        "",
        "| Group | Holdout N | Pair N | Model Best Rank | Two-Stage Rank | Review Hit | Review Gap | Diverse Hit | Diverse Gap | Best Family Rank | Gap | Two-Stage Gap | Best Baseline Rank | Top-3 | Top-5 | Best Label | Best Mask | Top Predicted Mask | Top Predicted Label |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ])
    for group in payload["groups"]:
        if group.get("status") != "completed":
            lines.append(
                f"| {group.get('group')} | {group.get('holdout_count', '')} |  |  |  |  |  |  | skipped: {group.get('reason')} |  |  |"
            )
            continue
        best = group["best_label"]
        top = group["top_predicted"][0] if group.get("top_predicted") else {}
        baseline_rank = best_policy_rank(group.get("policy_ranks") or {})
        two_stage_row = group.get("two_stage_mask_family") if isinstance(group.get("two_stage_mask_family"), dict) else {}
        lines.append(
            "| {group} | {holdout} | {pairs} | {rank} | {two_stage_rank} | {review_hit} | {review_gap} | {diverse_hit} | {diverse_gap} | {best_family_rank} | {gap} | {two_stage_gap} | {baseline_rank} | {top3} | {top5} | {best_label:.3f} | {best_mask} | {top_mask} | {top_label:.3f} |".format(
                group=group["group"],
                holdout=group["holdout_count"],
                pairs=group.get("pair_count", ""),
                rank=group["best_label_rank"],
                two_stage_rank=two_stage_row.get("best_label_rank", ""),
                review_hit=1 if two_stage_row.get("review_shortlist_contains_best") else 0 if two_stage_row else "",
                review_gap=format_percent(two_stage_row.get("review_shortlist_best_label_gap")) if two_stage_row else "",
                diverse_hit=1 if two_stage_row.get("diverse_review_shortlist_contains_best") else 0 if two_stage_row else "",
                diverse_gap=format_percent(two_stage_row.get("diverse_review_shortlist_best_label_gap")) if two_stage_row else "",
                best_family_rank=two_stage_row.get("best_family_rank", ""),
                gap=format_percent(group.get("top_predicted_label_gap")),
                two_stage_gap=format_percent(two_stage_row.get("top_predicted_label_gap")) if two_stage_row else "",
                baseline_rank=baseline_rank,
                top3=1 if group["top3"] else 0,
                top5=1 if group["top5"] else 0,
                best_label=float(best["label"]),
                best_mask=",".join(best.get("mask") or []),
                top_mask=",".join(top.get("mask") or []),
                top_label=float(top.get("label") or 0.0),
            )
        )
    lines.extend([
        "",
        "## Group Details",
        "",
    ])
    for group in payload["groups"]:
        if group.get("status") != "completed":
            continue
        lines.extend([
            f"### {group['group']}",
            "",
            "Top predicted:",
            "",
            "| Rank | Label | Raw | Mask | Preview |",
            "|---:|---:|---:|---|---|",
        ])
        for rank, row in enumerate(group["top_predicted"][:5], start=1):
            lines.append(_prediction_line(rank, row))
        lines.extend([
            "",
            "Top labeled:",
            "",
            "| Rank | Label | Raw | Mask | Preview |",
            "|---:|---:|---:|---|---|",
        ])
        for rank, row in enumerate(group["top_labeled"][:5], start=1):
            lines.append(_prediction_line(rank, row))
        lines.append("")
        if group.get("best_label_rank", 1) > 1 and group.get("feature_deltas"):
            lines.extend([
                "Top-predicted vs best-labeled feature deltas:",
                "",
                "| Feature | Top Predicted | Best Labeled | Delta | Z-Delta |",
                "|---|---:|---:|---:|---:|",
            ])
            for row in group["feature_deltas"][:8]:
                lines.append(
                    "| {feature} | {top:.3f} | {best:.3f} | {delta:+.3f} | {zdelta:+.3f} |".format(
                        feature=row["feature"],
                        top=float(row["top_predicted"]),
                        best=float(row["best_labeled"]),
                        delta=float(row["delta"]),
                        zdelta=float(row["z_delta"]),
                    )
                )
            lines.append("")
        if group.get("policy_ranks"):
            lines.extend([
                "Simple policy ranks for the best-labeled candidate:",
                "",
                "| Policy | Best Rank | Top Label | Top Mask | Feature |",
                "|---|---:|---:|---|---|",
            ])
            for policy, row in sorted(group["policy_ranks"].items()):
                lines.append(
                    "| {policy} | {rank} | {label:.3f} | {mask} | {feature} |".format(
                        policy=policy,
                        rank=row.get("best_label_rank"),
                        label=float(row.get("top_label") or 0.0),
                        mask=",".join(row.get("top_mask") or []),
                        feature=row.get("feature") or "",
                    )
                )
            lines.append("")
        two_stage_row = group.get("two_stage_mask_family") if isinstance(group.get("two_stage_mask_family"), dict) else {}
        if two_stage_row and two_stage_row.get("status") == "completed":
            top_family = two_stage_row.get("top_family") if isinstance(two_stage_row.get("top_family"), dict) else {}
            best_family = two_stage_row.get("best_family") if isinstance(two_stage_row.get("best_family"), dict) else {}
            top_prediction = two_stage_row.get("top_predicted") if isinstance(two_stage_row.get("top_predicted"), dict) else {}
            lines.extend([
                "Two-stage mask-family result:",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| best-label rank | {two_stage_row.get('best_label_rank')} |",
                f"| best-family rank | {two_stage_row.get('best_family_rank')} |",
                f"| family top-k | {two_stage_row.get('family_top_k')} |",
                f"| top-predicted label gap | {format_percent(two_stage_row.get('top_predicted_label_gap'))} |",
                f"| review shortlist K | {two_stage_row.get('review_shortlist_k')} |",
                f"| review shortlist contains exact best | {1 if two_stage_row.get('review_shortlist_contains_best') else 0} |",
                f"| review shortlist best-label gap | {format_percent(two_stage_row.get('review_shortlist_best_label_gap'))} |",
                f"| review shortlist best mask | {','.join((two_stage_row.get('review_shortlist_best') or {}).get('mask') or [])} |",
                f"| diverse review contains exact best | {1 if two_stage_row.get('diverse_review_shortlist_contains_best') else 0} |",
                f"| diverse review best-label gap | {format_percent(two_stage_row.get('diverse_review_shortlist_best_label_gap'))} |",
                f"| diverse review best mask | {','.join((two_stage_row.get('diverse_review_shortlist_best') or {}).get('mask') or [])} |",
                f"| top predicted mask | {','.join(top_prediction.get('mask') or [])} |",
                f"| top family mask | {','.join(top_family.get('mask') or [])} |",
                f"| top family raw | {top_family.get('raw_score', '')} |",
                f"| best family mask | {','.join(best_family.get('mask') or [])} |",
                f"| best family raw | {best_family.get('raw_score', '')} |",
                "",
            ])
    return "\n".join(lines) + "\n"


def best_policy_rank(policy_ranks: dict[str, Any]) -> str:
    ranks = [
        int(row["best_label_rank"])
        for row in policy_ranks.values()
        if isinstance(row, dict) and row.get("best_label_rank") is not None
    ]
    return str(min(ranks)) if ranks else ""


def _prediction_line(rank: int, row: dict[str, Any]) -> str:
    return (
        f"| {rank} | {float(row.get('label') or 0.0):.3f} | "
        f"{float(row.get('raw_score') or 0.0):.3f} | "
        f"{','.join(row.get('mask') or [])} | {row.get('preview') or ''} |"
    )


def format_percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.2f}%"


if __name__ == "__main__":
    main()
