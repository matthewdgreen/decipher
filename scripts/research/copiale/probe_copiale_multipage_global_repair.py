#!/usr/bin/env python3
"""Probe global shared-key repairs for multi-page Copiale finalists.

This is a ground-truth-free repair experiment at selection time. It starts
from a selected multi-page finalist, identifies disputed symbols in damaged
windows across pages, tries bounded shared-key/null edits globally, and ranks
variants with the same page-aware runtime validation used by the multi-page
experiment. Benchmark plaintext is reported only post hoc for calibration.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from benchmark.loader import BenchmarkLoader  # noqa: E402
from analysis.language_scoring import LinearLanguageQualityModel  # noqa: E402
from run_copiale_multipage_experiment import (  # noqa: E402
    PageBundle,
    attach_page_scores,
    build_combined_cipher,
    consensus_from_finalists,
    damaged_windows_for_text,
    finalist_rows,
    page_runtime_metrics,
    project_page_with_sources,
    project_pages,
    score_page_runtime,
)
from train_language_quality_scorer import (  # noqa: E402
    TrainingExample,
    add_global_repair_family_features,
    global_repair_feature_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Try bounded global shared-key repair variants from a multi-page Copiale artifact."
    )
    parser.add_argument("experiment_json", help="JSON from run_copiale_multipage_experiment.py")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument(
        "--section",
        choices=["portfolio_local_repair", "portfolio_refinement", "elite_page_rerank"],
        default="portfolio_local_repair",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Finalist label to repair, e.g. top9. Defaults to section best_by_policy.",
    )
    parser.add_argument("--consensus-top-n", type=int, default=12)
    parser.add_argument("--consensus-min-agreement", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--window-step", type=int, default=40)
    parser.add_argument("--windows-per-page", type=int, default=5)
    parser.add_argument("--max-symbols", type=int, default=10)
    parser.add_argument("--max-alternatives", type=int, default=4)
    parser.add_argument("--include-pairs", action="store_true")
    parser.add_argument(
        "--pair-candidate-limit",
        type=int,
        default=16,
        help="When --include-pairs is set, build pairs only from the strongest N edit atoms.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=120,
        help="Maximum pair variants to evaluate after pruning. Use 0 to disable pairs.",
    )
    parser.add_argument(
        "--acceptance-margin",
        type=float,
        default=0.03,
        help="Minimum robust-score gain for a repair to be marked runtime-accepted.",
    )
    parser.add_argument(
        "--min-page-drop",
        type=float,
        default=0.02,
        help="Maximum allowed drop in the worst-page validation score for runtime acceptance.",
    )
    parser.add_argument(
        "--max-illusion-increase",
        type=float,
        default=0.02,
        help="Maximum allowed increase in fragment-illusion penalty for runtime acceptance.",
    )
    parser.add_argument(
        "--allow-pair-acceptance",
        action="store_true",
        help="Allow two-edit variants to be marked runtime-accepted. By default pairs are review-only.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print variant-evaluation progress to stderr.",
    )
    parser.add_argument(
        "--language-quality-ranker",
        default="",
        help=(
            "Optional saved LinearLanguageQualityModel JSON. When set, score "
            "variants with ground-truth-free language-quality features and "
            "emit a diverse review shortlist."
        ),
    )
    parser.add_argument(
        "--ranker-review-top-k",
        type=int,
        default=8,
        help="Number of candidates to include in the language-ranker review shortlist.",
    )
    parser.add_argument(
        "--ranker-family-top-k",
        type=int,
        default=3,
        help="Number of top mask families to force into the diverse ranker shortlist.",
    )
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    experiment_path = resolve_path(Path(args.experiment_json))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    artifact_path = sibling_artifact_path(experiment_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    test_ids = [str(item) for item in (experiment.get("test_ids") or [])]
    if not test_ids:
        raise SystemExit("Experiment JSON has no test_ids.")

    loader = BenchmarkLoader(resolve_path(Path(args.benchmark_root)))
    tests = {test.test_id: test for test in loader.load_tests(args.split)}
    _combined, pages = build_combined_cipher(loader, tests, test_ids)
    alphabet = pages_to_alphabet(pages)
    finalists = finalist_rows(artifact, top_n=10_000)
    label = args.label or selected_label_from_section(experiment, args.section)
    selected = finalist_by_label(finalists, label)
    baseline_key = parse_key(selected.get("key"))
    baseline_mask = tuple(str(symbol) for symbol in (selected.get("mask") or []))
    consensus = consensus_from_finalists(
        artifact=artifact,
        alphabet=alphabet,
        top_n=args.consensus_top_n,
        min_agreement=args.consensus_min_agreement,
    )
    agenda = build_global_agenda(
        pages=pages,
        alphabet=alphabet,
        key=baseline_key,
        mask=baseline_mask,
        consensus=consensus,
        window_size=args.window_size,
        window_step=args.window_step,
        windows_per_page=args.windows_per_page,
        max_symbols=args.max_symbols,
        max_alternatives=args.max_alternatives,
    )
    variants = evaluate_global_variants(
        pages=pages,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        edit_groups=agenda["edit_groups"],
        include_pairs=args.include_pairs,
        pair_candidate_limit=args.pair_candidate_limit,
        max_pairs=args.max_pairs,
        progress=args.progress,
    )
    baseline_variant = next(
        (row for row in variants if row.get("edits") == ["baseline"]),
        variants[0] if variants else {},
    )
    annotate_acceptance(
        variants,
        baseline=baseline_variant,
        robust_margin=args.acceptance_margin,
        min_page_drop=args.min_page_drop,
        max_illusion_increase=args.max_illusion_increase,
        allow_pair_acceptance=args.allow_pair_acceptance,
    )
    annotate_repair_evidence(variants, baseline=baseline_variant)
    ranker_payload: dict[str, Any] | None = None
    if args.language_quality_ranker:
        ranker_model_path = resolve_path(Path(args.language_quality_ranker))
        ranker_model = LinearLanguageQualityModel.load(ranker_model_path)
        ranker_payload = annotate_language_quality_ranker(
            variants,
            model=ranker_model,
            model_path=ranker_model_path,
            source_experiment=str(experiment_path),
            source_artifact=str(artifact_path),
            section=args.section,
            label=label,
            test_ids=test_ids,
            review_top_k=args.ranker_review_top_k,
            family_top_k=args.ranker_family_top_k,
        )
    variants.sort(key=variant_rank_key, reverse=True)
    accepted = [row for row in variants if row.get("repair_acceptance", {}).get("accepted")]
    payload = {
        "experiment": "copiale_multipage_global_repair_probe",
        "source_experiment": str(experiment_path),
        "source_artifact": str(artifact_path),
        "section": args.section,
        "label": label,
        "test_ids": test_ids,
        "baseline": variant_summary(baseline_variant) if baseline_variant else {},
        "variant_count": len(variants),
        "accepted_variant_count": len(accepted),
        "acceptance_thresholds": {
            "robust_margin": args.acceptance_margin,
            "min_page_drop": args.min_page_drop,
            "max_illusion_increase": args.max_illusion_increase,
            "allow_pair_acceptance": args.allow_pair_acceptance,
        },
        "consensus_top_n": args.consensus_top_n,
        "consensus_min_agreement": args.consensus_min_agreement,
        "agenda": agenda,
        "top_accepted_variants": [variant_summary(row) for row in accepted[: args.top_n]],
        "top_variants": [variant_summary(row) for row in variants[: args.top_n]],
    }
    if ranker_payload is not None:
        payload["language_quality_ranker"] = ranker_payload
    markdown = render_markdown(payload)
    output = (
        resolve_path(Path(args.output))
        if args.output
        else experiment_path.with_suffix(f".{args.section}.global_repair.md")
    )
    json_output = (
        resolve_path(Path(args.json_output))
        if args.json_output
        else output.with_suffix(".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")


def build_global_agenda(
    *,
    pages: list[PageBundle],
    alphabet: Any,
    key: dict[int, int],
    mask: tuple[str, ...],
    consensus: dict[str, dict[str, Any]],
    window_size: int,
    window_step: int,
    windows_per_page: int,
    max_symbols: int,
    max_alternatives: int,
) -> dict[str, Any]:
    symbol_pressure: dict[str, int] = {}
    windows_by_page = []
    for page in pages:
        text, sources = project_page_with_sources(page=page, key=key, mask=mask, alphabet=alphabet)
        windows = damaged_windows_for_text(
            text=text,
            sources=sources,
            consensus=consensus,
            window_size=window_size,
            step=window_step,
            limit=windows_per_page,
        )
        for window in windows:
            for item in window.get("disputed_symbols") or []:
                symbol = str(item.get("symbol") or "")
                if symbol:
                    symbol_pressure[symbol] = symbol_pressure.get(symbol, 0) + int(item.get("count") or 0)
        windows_by_page.append({
            "test_id": page.test_id,
            "windows": compact_windows(windows),
        })
    disputed = {
        str(row.get("symbol")): row
        for row in consensus.values()
        if row.get("symbol") and not row.get("stable")
    }
    symbols = sorted(
        disputed,
        key=lambda symbol: (
            -symbol_pressure.get(symbol, 0),
            float(disputed[symbol].get("agreement") or 1.0),
            symbol,
        ),
    )
    edit_groups = []
    for symbol in symbols[: max(0, max_symbols)]:
        if not alphabet.has_symbol(symbol):
            continue
        token_id = alphabet.id_for(symbol)
        current = current_assignment(symbol, token_id, key, mask)
        counts = disputed[symbol].get("counts") if isinstance(disputed[symbol].get("counts"), dict) else {}
        alternatives = [
            str(value)
            for value, _count in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
            if str(value) != current and str(value) != "?"
        ][: max(0, max_alternatives)]
        if alternatives:
            edit_groups.append({
                "symbol": symbol,
                "token_id": token_id,
                "current": current,
                "pressure": symbol_pressure.get(symbol, 0),
                "agreement": disputed[symbol].get("agreement"),
                "alternatives": alternatives,
                "counts": counts,
            })
    return {
        "window_size": window_size,
        "window_step": window_step,
        "windows_per_page": windows_per_page,
        "windows_by_page": windows_by_page,
        "symbol_pressure": dict(sorted(symbol_pressure.items(), key=lambda item: (-item[1], item[0]))),
        "edit_groups": edit_groups,
    }


def evaluate_global_variants(
    *,
    pages: list[PageBundle],
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    edit_groups: list[dict[str, Any]],
    include_pairs: bool,
    pair_candidate_limit: int = 16,
    max_pairs: int = 120,
    progress: bool = False,
) -> list[dict[str, Any]]:
    edit_atoms: list[dict[str, Any]] = []
    for group_index, group in enumerate(edit_groups):
        for alternative in group["alternatives"]:
            edit_atoms.append({
                "group": group,
                "group_index": group_index,
                "alternative": alternative,
                "score": edit_atom_score(group, alternative),
            })
    edit_atoms.sort(key=lambda atom: (-float(atom["score"]), str(atom["group"]["symbol"]), str(atom["alternative"])))
    edit_sets: list[tuple[dict[str, Any], ...]] = [()]
    edit_sets.extend((item,) for item in edit_atoms)
    if include_pairs:
        pair_atoms = edit_atoms[: max(0, pair_candidate_limit)]
        pairs = [
            tuple(pair)
            for pair in itertools.combinations(pair_atoms, 2)
            if pair[0]["group"]["symbol"] != pair[1]["group"]["symbol"]
        ]
        pairs.sort(key=lambda pair: (-(float(pair[0]["score"]) + float(pair[1]["score"])), pair_signature(pair)))
        edit_sets.extend(pairs[: max(0, max_pairs)])
    variants = []
    seen: set[tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = set()
    started = time.monotonic()
    total = len(edit_sets)
    progress_every = max(1, min(50, total // 20 if total >= 20 else total))
    for index, edit_set in enumerate(edit_sets, start=1):
        key = dict(baseline_key)
        mask = set(baseline_mask)
        edits = []
        for atom in edit_set:
            group = atom["group"]
            alternative = str(atom["alternative"])
            symbol = str(group["symbol"])
            token_id = int(group["token_id"])
            before = current_assignment(symbol, token_id, key, tuple(sorted(mask)))
            apply_assignment(symbol, token_id, alternative, key, mask)
            edits.append(f"{symbol}:{before}->{alternative}")
        identity = (tuple(sorted(mask)), tuple(sorted(key.items())))
        if identity in seen:
            continue
        seen.add(identity)
        if progress and (index == 1 or index == total or index % progress_every == 0):
            elapsed = time.monotonic() - started
            print(
                f"Evaluating repair variants: {index}/{total} "
                f"({index / max(1, total):.0%}) elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        page_rows = project_pages(pages=pages, key=key, mask=tuple(sorted(mask)))
        attach_page_scores(page_rows)
        runtime_scores = [
            score_page_runtime(row, key=key, mask=tuple(sorted(mask)))
            for row in page_rows
        ]
        metrics = page_runtime_metrics(runtime_scores)
        avg_char = sum(float(row.get("char_accuracy") or 0.0) for row in page_rows) / max(1, len(page_rows))
        variants.append({
            "edits": edits or ["baseline"],
            "mask": list(sorted(mask)),
            "page_runtime_scores": runtime_scores,
            "post_hoc_char_avg": round(avg_char, 6),
            "post_hoc_page_chars": [
                {
                    "test_id": row["test_id"],
                    "char_accuracy": round(float(row.get("char_accuracy") or 0.0), 6),
                    "word_accuracy": round(float(row.get("word_accuracy") or 0.0), 6),
                }
                for row in page_rows
            ],
            "page_previews": [
                {
                    "test_id": row["test_id"],
                    "preview": str(row.get("preview") or ""),
                    "filtered_length": int(row.get("filtered_length") or 0),
                }
                for row in page_rows
            ],
            "preview": "; ".join(str(row.get("preview") or "")[:70] for row in page_rows[:3]),
            **metrics,
        })
    return variants


def edit_atom_score(group: dict[str, Any], alternative: str) -> float:
    counts = group.get("counts") if isinstance(group.get("counts"), dict) else {}
    alt_count = float(counts.get(alternative) or 0.0)
    pressure = float(group.get("pressure") or 0.0)
    agreement = float(group.get("agreement") or 0.0)
    null_bonus = 0.5 if alternative == "<null>" else 0.0
    return pressure + alt_count + null_bonus - agreement


def pair_signature(pair: tuple[dict[str, Any], dict[str, Any]]) -> str:
    parts = [
        f"{atom['group']['symbol']}->{atom['alternative']}"
        for atom in pair
    ]
    return ";".join(sorted(parts))


def annotate_acceptance(
    variants: list[dict[str, Any]],
    *,
    baseline: dict[str, Any],
    robust_margin: float,
    min_page_drop: float,
    max_illusion_increase: float,
    allow_pair_acceptance: bool = False,
) -> None:
    for row in variants:
        row["repair_acceptance"] = repair_acceptance(
            row,
            baseline=baseline,
            robust_margin=robust_margin,
            min_page_drop=min_page_drop,
            max_illusion_increase=max_illusion_increase,
            allow_pair_acceptance=allow_pair_acceptance,
        )


def annotate_repair_evidence(variants: list[dict[str, Any]], *, baseline: dict[str, Any]) -> None:
    for row in variants:
        row["repair_evidence"] = repair_evidence(row, baseline=baseline)


def annotate_language_quality_ranker(
    variants: list[dict[str, Any]],
    *,
    model: LinearLanguageQualityModel,
    model_path: Path,
    source_experiment: str,
    source_artifact: str,
    section: str,
    label: str,
    test_ids: list[str],
    review_top_k: int,
    family_top_k: int,
) -> dict[str, Any]:
    """Score repair variants with a saved ground-truth-free ranker."""
    entries: list[dict[str, Any]] = []
    payload = {
        "source_experiment": source_experiment,
        "source_artifact": source_artifact,
        "section": section,
        "label": label,
        "test_ids": test_ids,
    }
    for index, row in enumerate(variants, start=1):
        features = global_repair_feature_dict(row)
        example = TrainingExample(
            text=str(row.get("preview") or "__GLOBAL_REPAIR_CANDIDATE__"),
            label=0.0,
            source=f"runtime_global_repair:{index}",
            group=f"{section}:{label}",
            features=features,
            metadata={
                "kind": "runtime_global_repair_candidate",
                "rank": index,
                "mask": row.get("mask") or [],
                "edits": row.get("edits") or [],
            },
        )
        entries.append({"example": example, "row": row, "payload": payload})
    add_global_repair_family_features(entries)
    for entry in entries:
        row = entry["row"]
        example = entry["example"]
        row["language_quality_rank_score"] = round(float(model.raw_score_features(example.features)), 6)
        row["language_quality_rank_normalized"] = round(float(model.score_features(example.features)), 6)
    ranked = sorted(
        variants,
        key=lambda row: (
            float(row.get("language_quality_rank_score") or 0.0),
            float(row.get("page_robust_score") or 0.0),
        ),
        reverse=True,
    )
    family_rows = language_quality_family_rows(ranked)
    shortlist = diverse_language_quality_shortlist(
        ranked,
        family_rows=family_rows,
        review_top_k=review_top_k,
        family_top_k=family_top_k,
    )
    return {
        "model_path": str(model_path),
        "model_language": model.language,
        "model_feature_count": len(model.feature_names),
        "review_top_k": max(1, int(review_top_k)),
        "family_top_k": max(1, int(family_top_k)),
        "top_by_ranker": [variant_summary(row) for row in ranked[: max(1, int(review_top_k))]],
        "diverse_review_shortlist": [variant_summary(row) for row in shortlist],
        "top_families": [
            {
                "mask": list(mask),
                "member_count": len(rows),
                "best_rank_score": round(float(rows[0].get("language_quality_rank_score") or 0.0), 6),
                "best_robust_score": rows[0].get("page_robust_score"),
            }
            for mask, rows in family_rows[: max(1, int(family_top_k))]
        ],
    }


def language_quality_family_rows(rows: list[dict[str, Any]]) -> list[tuple[tuple[str, ...], list[dict[str, Any]]]]:
    by_family: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(mask_key(row), []).append(row)
    families = list(by_family.items())
    families.sort(
        key=lambda item: (
            float(item[1][0].get("language_quality_rank_score") or 0.0),
            float(item[1][0].get("page_robust_score") or 0.0),
        ),
        reverse=True,
    )
    return families


def diverse_language_quality_shortlist(
    ranked: list[dict[str, Any]],
    *,
    family_rows: list[tuple[tuple[str, ...], list[dict[str, Any]]]],
    review_top_k: int,
    family_top_k: int,
) -> list[dict[str, Any]]:
    limit = max(1, int(review_top_k))
    selected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for _mask, rows in family_rows[: max(1, int(family_top_k))]:
        if rows:
            selected.append(rows[0])
            seen_ids.add(id(rows[0]))
        if len(selected) >= limit:
            return selected
    for row in ranked:
        if id(row) in seen_ids:
            continue
        selected.append(row)
        seen_ids.add(id(row))
        if len(selected) >= limit:
            break
    return selected


def mask_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in (row.get("mask") or [])))


def repair_evidence(row: dict[str, Any], *, baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_runtime = keyed_by_test_id(baseline.get("page_runtime_scores") or [])
    baseline_chars = keyed_by_test_id(baseline.get("post_hoc_page_chars") or [])
    baseline_previews = keyed_by_test_id(baseline.get("page_previews") or [])
    page_evidence = []
    runtime_improved = 0
    runtime_regressed = 0
    preview_changed = 0
    runtime_suspicious_pages = 0
    calibration_suspicious_pages = 0
    posthoc_improved = 0
    posthoc_regressed = 0
    for runtime in row.get("page_runtime_scores") or []:
        test_id = str(runtime.get("test_id") or "")
        base_runtime = baseline_runtime.get(test_id, {})
        base_char = baseline_chars.get(test_id, {})
        char = (keyed_by_test_id(row.get("post_hoc_page_chars") or []).get(test_id, {}))
        base_preview = str((baseline_previews.get(test_id) or {}).get("preview") or "")
        preview = str((keyed_by_test_id(row.get("page_previews") or {}).get(test_id, {})).get("preview") or "")
        val_delta = numeric_delta(runtime, base_runtime, "validation_score_v2")
        lq_delta = numeric_delta(runtime, base_runtime, "language_quality_mean")
        dict_delta = numeric_delta(runtime, base_runtime, "dict_rate")
        pseudo_delta = nested_delta(runtime, base_runtime, "diagnostics", "pseudo_word_fraction")
        binary_delta = nested_delta(runtime, base_runtime, "validation_components_v2", "binary_ngram_fit")
        coherence_delta = nested_delta(runtime, base_runtime, "validation_components_v2", "language_coherence")
        shape_delta = nested_delta(runtime, base_runtime, "validation_components_v2", "language_shape")
        char_delta = numeric_delta(char, base_char, "char_accuracy")
        changed = preview != base_preview
        if val_delta > 0.005:
            runtime_improved += 1
        elif val_delta < -0.005:
            runtime_regressed += 1
        if char_delta > 0.001:
            posthoc_improved += 1
        elif char_delta < -0.001:
            posthoc_regressed += 1
        if changed:
            preview_changed += 1
        runtime_flags = []
        if val_delta > 0.005 and lq_delta <= 0.0:
            runtime_flags.append("validation_up_without_lq_gain")
        if val_delta > 0.005 and dict_delta < -0.01:
            runtime_flags.append("validation_up_dictionary_down")
        if val_delta > 0.005 and pseudo_delta > 0.01:
            runtime_flags.append("validation_up_more_pseudowords")
        if val_delta > 0.005 and not changed:
            runtime_flags.append("validation_up_preview_unchanged")
        if runtime_flags:
            runtime_suspicious_pages += 1
        calibration_flags = []
        if val_delta > 0.005 and char_delta < -0.001:
            calibration_flags.append("runtime_up_posthoc_char_down")
        if calibration_flags:
            calibration_suspicious_pages += 1
        page_evidence.append({
            "test_id": test_id,
            "validation_delta": val_delta,
            "language_quality_delta": lq_delta,
            "dict_rate_delta": dict_delta,
            "pseudo_word_fraction_delta": pseudo_delta,
            "binary_ngram_fit_delta": binary_delta,
            "language_coherence_delta": coherence_delta,
            "language_shape_delta": shape_delta,
            "post_hoc_char_delta": char_delta,
            "preview_changed": changed,
            "changed_excerpt": changed_excerpt(base_preview, preview),
            "runtime_flags": runtime_flags,
            "calibration_flags": calibration_flags,
        })
    page_evidence.sort(
        key=lambda item: (
            -len(item["runtime_flags"]) - len(item["calibration_flags"]),
            -abs(float(item["validation_delta"])),
            item["test_id"],
        )
    )
    runtime_decision_flags = []
    if runtime_suspicious_pages:
        runtime_decision_flags.append(f"{runtime_suspicious_pages} page(s) improve by validation but have weak supporting signals")
    if runtime_improved <= runtime_regressed and row.get("edits") != ["baseline"]:
        runtime_decision_flags.append("runtime improvements are not page-majority")
    if not preview_changed and row.get("edits") != ["baseline"]:
        runtime_decision_flags.append("no preview changed")
    calibration_flags = []
    if calibration_suspicious_pages:
        calibration_flags.append(f"{calibration_suspicious_pages} page(s) improve by runtime score but lose post-hoc char")
    if posthoc_regressed > posthoc_improved and row.get("edits") != ["baseline"]:
        calibration_flags.append("post-hoc char regresses on more pages than it improves")
    return {
        "page_count": len(page_evidence),
        "runtime_pages_improved": runtime_improved,
        "runtime_pages_regressed": runtime_regressed,
        "preview_pages_changed": preview_changed,
        "runtime_suspicious_pages": runtime_suspicious_pages,
        "post_hoc_pages_improved": posthoc_improved,
        "post_hoc_pages_regressed": posthoc_regressed,
        "calibration_suspicious_pages": calibration_suspicious_pages,
        "runtime_decision_flags": runtime_decision_flags,
        "calibration_flags": calibration_flags,
        "pages": page_evidence,
    }


def repair_acceptance(
    row: dict[str, Any],
    *,
    baseline: dict[str, Any],
    robust_margin: float,
    min_page_drop: float,
    max_illusion_increase: float,
    allow_pair_acceptance: bool = False,
) -> dict[str, Any]:
    deltas = {
        "page_robust_score": numeric_delta(row, baseline, "page_robust_score"),
        "page_balanced_score": numeric_delta(row, baseline, "page_balanced_score"),
        "page_validation_avg": numeric_delta(row, baseline, "page_validation_avg"),
        "page_validation_min": numeric_delta(row, baseline, "page_validation_min"),
        "fragment_illusion_penalty": numeric_delta(row, baseline, "fragment_illusion_penalty"),
        "page_language_quality_avg": numeric_delta(row, baseline, "page_language_quality_avg"),
    }
    edits = row.get("edits") or []
    if edits == ["baseline"]:
        return {
            "accepted": False,
            "decision": "baseline",
            "deltas": deltas,
            "reasons": ["baseline candidate"],
        }
    reasons = []
    accepted = True
    edit_count = len(edits)
    if edit_count > 1 and not allow_pair_acceptance:
        accepted = False
        reasons.append("multi-edit variant is review-only unless --allow-pair-acceptance is set")
    if deltas["page_robust_score"] < robust_margin:
        accepted = False
        reasons.append(f"robust gain below margin ({deltas['page_robust_score']:.3f} < {robust_margin:.3f})")
    else:
        reasons.append(f"robust gain clears margin ({deltas['page_robust_score']:.3f})")
    if deltas["page_balanced_score"] < 0.0:
        accepted = False
        reasons.append(f"balanced score regresses ({deltas['page_balanced_score']:.3f})")
    if deltas["page_validation_avg"] < 0.0:
        accepted = False
        reasons.append(f"average page validation regresses ({deltas['page_validation_avg']:.3f})")
    if deltas["page_validation_min"] < -min_page_drop:
        accepted = False
        reasons.append(f"worst page drops too much ({deltas['page_validation_min']:.3f})")
    if deltas["fragment_illusion_penalty"] > max_illusion_increase:
        accepted = False
        reasons.append(f"fragment illusion rises too much ({deltas['fragment_illusion_penalty']:.3f})")
    positive_support = sum(
        1
        for key in ("page_robust_score", "page_balanced_score", "page_validation_avg", "page_language_quality_avg")
        if deltas[key] > 0.0
    )
    if positive_support < 2:
        accepted = False
        reasons.append(f"only {positive_support} runtime signals improve")
    return {
        "accepted": accepted,
        "decision": "runtime_accept" if accepted else "hold_for_review",
        "deltas": deltas,
        "positive_signal_count": positive_support,
        "reasons": reasons,
    }


def numeric_delta(row: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(float(row.get(key) or 0.0) - float(baseline.get(key) or 0.0), 6)


def nested_delta(row: dict[str, Any], baseline: dict[str, Any], parent: str, key: str) -> float:
    return round(
        float((row.get(parent) or {}).get(key) or 0.0)
        - float((baseline.get(parent) or {}).get(key) or 0.0),
        6,
    )


def keyed_by_test_id(rows: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("test_id") or ""): row
        for row in rows
        if isinstance(row, dict)
    }


def changed_excerpt(before: str, after: str, *, radius: int = 28) -> dict[str, Any]:
    if before == after:
        return {"changed": False, "before": "", "after": "", "offset": None}
    limit = min(len(before), len(after))
    start = 0
    while start < limit and before[start] == after[start]:
        start += 1
    left = max(0, start - radius)
    right = min(max(len(before), len(after)), start + radius)
    return {
        "changed": True,
        "offset": start,
        "before": before[left:right],
        "after": after[left:right],
    }


def variant_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("page_robust_score") or 0.0),
        float(row.get("page_balanced_score") or 0.0),
        float(row.get("page_validation_avg") or 0.0),
        -len(row.get("edits") or []),
    )


def variant_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "edits": row.get("edits") or [],
        "mask": row.get("mask") or [],
        "page_robust_score": row.get("page_robust_score"),
        "page_balanced_score": row.get("page_balanced_score"),
        "page_validation_avg": row.get("page_validation_avg"),
        "page_validation_min": row.get("page_validation_min"),
        "fragment_illusion_penalty": row.get("fragment_illusion_penalty"),
        "page_language_quality_avg": row.get("page_language_quality_avg"),
        "page_dict_avg": row.get("page_dict_avg"),
        "page_content_char_avg": row.get("page_content_char_avg"),
        "page_pseudo_word_avg": row.get("page_pseudo_word_avg"),
        "page_binary_component_avg": row.get("page_binary_component_avg"),
        "page_shape_component_avg": row.get("page_shape_component_avg"),
        "page_evidence_dispersion_avg": row.get("page_evidence_dispersion_avg"),
        "page_window_stability_avg": row.get("page_window_stability_avg"),
        "page_repetition_control_avg": row.get("page_repetition_control_avg"),
        "page_content_word_quality_avg": row.get("page_content_word_quality_avg"),
        "page_language_coherence_avg": row.get("page_language_coherence_avg"),
        "language_quality_rank_score": row.get("language_quality_rank_score"),
        "language_quality_rank_normalized": row.get("language_quality_rank_normalized"),
        "post_hoc_char_avg": row.get("post_hoc_char_avg"),
        "post_hoc_page_chars": row.get("post_hoc_page_chars") or [],
        "repair_acceptance": row.get("repair_acceptance") or {},
        "repair_evidence": row.get("repair_evidence") or {},
        "preview": row.get("preview"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    baseline = payload.get("baseline") or {}
    ranker = payload.get("language_quality_ranker") if isinstance(payload.get("language_quality_ranker"), dict) else {}
    lines = [
        "# Copiale Multi-Page Global Repair Probe",
        "",
        "Ground truth is not used for ranking. Character accuracy below is post-hoc calibration only.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| source label | `{payload['label']}` |",
        f"| variants | {payload['variant_count']} |",
        f"| runtime-accepted variants | {payload.get('accepted_variant_count', 0)} |",
        f"| baseline robust | {format_number(baseline.get('page_robust_score'))} |",
        f"| baseline post-hoc char | {format_percent(baseline.get('post_hoc_char_avg'))} |",
    ]
    if ranker:
        lines.extend([
            f"| language-quality ranker | `{ranker.get('model_path')}` |",
            f"| ranker review shortlist | {ranker.get('review_top_k')} candidates, {ranker.get('family_top_k')} forced families |",
        ])
    lines.extend([
        "",
        "## Edit Groups",
        "",
        "| Symbol | Current | Pressure | Agreement | Alternatives | Counts |",
        "|---|---|---:|---:|---|---|",
    ])
    for group in payload["agenda"]["edit_groups"]:
        lines.append(
            f"| {group['symbol']} | {group['current']} | {group['pressure']} | "
            f"{format_number(group.get('agreement'))} | {', '.join(group['alternatives'])} | "
            f"{escape_cell(format_counts(group.get('counts') or {}))} |"
        )
    lines.extend([
        "",
        "## Top Variants",
        "",
        "| Rank | Edits | Accept | Evidence | LQ Rank | dRobust | Robust | Balanced | Page Avg | Page Min | Illusion | Post-Hoc Char | Preview |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for idx, row in enumerate(payload["top_variants"], start=1):
        acceptance = row.get("repair_acceptance") or {}
        deltas = acceptance.get("deltas") if isinstance(acceptance.get("deltas"), dict) else {}
        lines.append(
            f"| {idx} | {escape_cell('; '.join(row['edits']))} | "
            f"{format_acceptance(acceptance)} | "
            f"{escape_cell(format_evidence_summary(row.get('repair_evidence') or {}))} | "
            f"{format_number(row.get('language_quality_rank_score'))} | "
            f"{format_signed(deltas.get('page_robust_score'))} | "
            f"{format_number(row.get('page_robust_score'))} | "
            f"{format_number(row.get('page_balanced_score'))} | "
            f"{format_number(row.get('page_validation_avg'))} | "
            f"{format_number(row.get('page_validation_min'))} | "
            f"{format_number(row.get('fragment_illusion_penalty'))} | "
            f"{format_percent(row.get('post_hoc_char_avg'))} | "
            f"{escape_cell(str(row.get('preview') or ''))} |"
        )
    if ranker:
        lines.extend([
            "",
            "## Language-Ranker Review Shortlist",
            "",
            "This shortlist is ground-truth-free. It is meant for agent/human review, not automatic repair adoption.",
            "",
            "| Rank | Edits | Mask | LQ Rank | Robust | Balanced | Page Avg | Post-Hoc Char | Preview |",
            "|---:|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for idx, row in enumerate(ranker.get("diverse_review_shortlist") or [], start=1):
            lines.append(
                f"| {idx} | {escape_cell('; '.join(row.get('edits') or []))} | "
                f"{escape_cell(','.join(row.get('mask') or []))} | "
                f"{format_number(row.get('language_quality_rank_score'))} | "
                f"{format_number(row.get('page_robust_score'))} | "
                f"{format_number(row.get('page_balanced_score'))} | "
                f"{format_number(row.get('page_validation_avg'))} | "
                f"{format_percent(row.get('post_hoc_char_avg'))} | "
                f"{escape_cell(str(row.get('preview') or ''))} |"
            )
        lines.extend([
            "",
            "Top ranker families:",
            "",
            "| Rank | Mask | Members | Best LQ Rank | Best Robust |",
            "|---:|---|---:|---:|---:|",
        ])
        for idx, row in enumerate(ranker.get("top_families") or [], start=1):
            lines.append(
                f"| {idx} | {escape_cell(','.join(row.get('mask') or []))} | "
                f"{row.get('member_count')} | "
                f"{format_number(row.get('best_rank_score'))} | "
                f"{format_number(row.get('best_robust_score'))} |"
            )
    accepted = payload.get("top_accepted_variants") if isinstance(payload.get("top_accepted_variants"), list) else []
    if accepted:
        lines.extend([
            "",
            "## Runtime-Accepted Variants",
            "",
            "These pass the ground-truth-free acceptance checks. Character accuracy is still post-hoc only.",
            "",
            "| Rank | Edits | dRobust | dBalanced | dPage Avg | dPage Min | dIllusion | Post-Hoc Char |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for idx, row in enumerate(accepted, start=1):
            acceptance = row.get("repair_acceptance") or {}
            deltas = acceptance.get("deltas") if isinstance(acceptance.get("deltas"), dict) else {}
            lines.append(
                f"| {idx} | {escape_cell('; '.join(row['edits']))} | "
                f"{format_signed(deltas.get('page_robust_score'))} | "
                f"{format_signed(deltas.get('page_balanced_score'))} | "
                f"{format_signed(deltas.get('page_validation_avg'))} | "
                f"{format_signed(deltas.get('page_validation_min'))} | "
                f"{format_signed(deltas.get('fragment_illusion_penalty'))} | "
                f"{format_percent(row.get('post_hoc_char_avg'))} |"
            )
    lines.extend(["", "## Repair Evidence Details", ""])
    lines.append(
        "Runtime flags are ground-truth-free. Calibration flags use post-hoc "
        "benchmark plaintext only to diagnose this experiment after variants "
        "have already been generated."
    )
    lines.append("")
    for idx, row in enumerate(payload["top_variants"][: min(8, len(payload["top_variants"]))], start=1):
        evidence = row.get("repair_evidence") or {}
        runtime_flags = evidence.get("runtime_decision_flags") or []
        calibration_flags = evidence.get("calibration_flags") or []
        lines.append(f"### {idx}. {escape_cell('; '.join(row['edits']))}")
        lines.append("")
        if runtime_flags:
            lines.append(f"- Runtime review: {escape_cell('; '.join(runtime_flags))}")
        else:
            lines.append("- Runtime review: no aggregate runtime warning")
        if calibration_flags:
            lines.append(f"- Post-hoc calibration warning: {escape_cell('; '.join(calibration_flags))}")
        else:
            lines.append("- Post-hoc calibration warning: none")
        lines.append("")
        lines.append("| Page | dVal | dLQ | dDict | dPseudo | dBinary | dChar* | Flags | Changed Excerpt |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
        for page in (evidence.get("pages") or [])[:5]:
            flags = list(page.get("runtime_flags") or []) + [
                f"posthoc:{flag}" for flag in (page.get("calibration_flags") or [])
            ]
            excerpt = page.get("changed_excerpt") if isinstance(page.get("changed_excerpt"), dict) else {}
            lines.append(
                f"| {page.get('test_id', '')} | "
                f"{format_signed(page.get('validation_delta'))} | "
                f"{format_signed(page.get('language_quality_delta'))} | "
                f"{format_signed(page.get('dict_rate_delta'))} | "
                f"{format_signed(page.get('pseudo_word_fraction_delta'))} | "
                f"{format_signed(page.get('binary_ngram_fit_delta'))} | "
                f"{format_signed(page.get('post_hoc_char_delta'))} | "
                f"{escape_cell(', '.join(flags))} | "
                f"{escape_cell(format_excerpt(excerpt))} |"
            )
        lines.append("")
    lines.append("*dChar is post-hoc calibration only and is not used for ranking or acceptance.")
    lines.extend(["", "## Damaged Windows Used For Agenda", ""])
    for page in payload["agenda"]["windows_by_page"]:
        lines.append(f"### {page['test_id']}")
        lines.append("")
        lines.append("| Span | Damage | Disputed | Text |")
        lines.append("|---|---:|---:|---|")
        for window in page["windows"]:
            lines.append(
                f"| {window['start']}-{window['end']} | {format_number(window['damage_score'])} | "
                f"{window['disputed_symbol_count']} | {escape_cell(window['text'])} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def selected_label_from_section(payload: dict[str, Any], section: str) -> str:
    block = payload.get(section) if isinstance(payload.get(section), dict) else {}
    best = block.get("best_by_policy") if isinstance(block.get("best_by_policy"), dict) else {}
    if best.get("label"):
        return str(best["label"])
    rows = block.get("rows") if isinstance(block.get("rows"), list) else []
    if rows and isinstance(rows[0], dict) and rows[0].get("label"):
        return str(rows[0]["label"])
    return "selected"


def finalist_by_label(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("_label") or "") == label:
            return row
    available = ", ".join(str(row.get("_label")) for row in rows[:20])
    raise SystemExit(f"Finalist label not found: {label}. Available: {available}")


def pages_to_alphabet(pages: list[PageBundle]) -> Any:
    symbols = []
    seen = set()
    for page in pages:
        for symbol in page.symbols:
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    from models.alphabet import Alphabet

    return Alphabet(symbols)


def current_assignment(symbol: str, token_id: int, key: dict[int, int], mask: tuple[str, ...]) -> str:
    if symbol in set(mask):
        return "<null>"
    value = key.get(token_id)
    if value is None or value < 0 or value > 25:
        return "?"
    return chr(ord("A") + value)


def apply_assignment(
    symbol: str,
    token_id: int,
    assignment: str,
    key: dict[int, int],
    mask: set[str],
) -> None:
    if assignment == "<null>":
        mask.add(symbol)
        return
    mask.discard(symbol)
    if len(assignment) == 1 and "A" <= assignment <= "Z":
        key[token_id] = ord(assignment) - ord("A")


def parse_key(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    parsed = {}
    for key, item in value.items():
        try:
            parsed[int(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return parsed


def compact_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": row.get("start"),
            "end": row.get("end"),
            "damage_score": row.get("damage_score"),
            "disputed_symbol_count": row.get("disputed_symbol_count"),
            "disputed_symbols": [
                {
                    "symbol": item.get("symbol"),
                    "count": item.get("count"),
                    "winner": item.get("winner"),
                    "agreement": item.get("agreement"),
                    "assignments": item.get("assignments"),
                }
                for item in (row.get("disputed_symbols") or [])[:6]
                if isinstance(item, dict)
            ],
            "text": str(row.get("text") or "")[:180],
        }
        for row in windows
    ]


def sibling_artifact_path(path: Path) -> Path:
    artifact_path = path.with_suffix(".artifact.json")
    if artifact_path.exists():
        return artifact_path
    raise SystemExit(f"Sibling artifact JSON not found: {artifact_path}")


def format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in counts.items())


def format_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def format_signed(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return str(value)


def format_acceptance(value: dict[str, Any]) -> str:
    if not value:
        return ""
    if value.get("accepted"):
        return "accept"
    decision = str(value.get("decision") or "")
    if decision == "baseline":
        return "baseline"
    return "review"


def format_evidence_summary(value: dict[str, Any]) -> str:
    if not value:
        return ""
    parts = [
        f"runtime +{value.get('runtime_pages_improved', 0)}/-{value.get('runtime_pages_regressed', 0)}",
        f"changed {value.get('preview_pages_changed', 0)}",
    ]
    if value.get("runtime_suspicious_pages"):
        parts.append(f"runtime flags {value['runtime_suspicious_pages']}")
    if value.get("calibration_suspicious_pages"):
        parts.append(f"posthoc flags {value['calibration_suspicious_pages']}")
    return "; ".join(parts)


def format_excerpt(value: dict[str, Any]) -> str:
    if not value or not value.get("changed"):
        return ""
    before = str(value.get("before") or "")
    after = str(value.get("after") or "")
    return f"{before} -> {after}"


def escape_cell(text: str) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
