#!/usr/bin/env python3
"""Probe word-hypothesis repairs for multi-page Copiale finalists.

This is a ground-truth-free candidate generator at runtime. It looks for
damaged windows, proposes same-length dictionary word hypotheses for garbled
substrings, converts each word hypothesis into a multi-symbol global edit set,
and then scores the resulting shared-key variants across all pages.

Benchmark plaintext is reported only after variants have been generated, for
calibration and debugging.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from benchmark.scorer import score_decryption  # noqa: E402
from analysis.language_scoring import language_quality_feature_dict  # noqa: E402
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
from report_copiale_repair_agenda import window_damage_score  # noqa: E402
from probe_copiale_multipage_global_repair import (  # noqa: E402
    annotate_acceptance,
    annotate_repair_evidence,
    current_assignment,
    parse_key,
    apply_assignment,
    pages_to_alphabet,
    resolve_path,
    selected_label_from_section,
    sibling_artifact_path,
    variant_rank_key,
    variant_summary,
)


@dataclass(frozen=True)
class WordHypothesis:
    test_id: str
    window_start: int
    start: int
    end: int
    observed: str
    target: str
    edits: tuple[tuple[str, str], ...]
    distance: int
    dictionary_rank: int
    local_score: float


@dataclass(frozen=True)
class WordEvidenceIndex:
    dictionary: dict[int, list[tuple[str, int]]]
    patterns: dict[int, dict[str, list[tuple[str, int]]]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Try dictionary word-hypothesis multi-symbol repairs for a Copiale multi-page finalist."
    )
    parser.add_argument("experiment_json", help="JSON from run_copiale_multipage_experiment.py")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument(
        "--section",
        choices=["portfolio_local_repair", "portfolio_refinement", "elite_page_rerank"],
        default="portfolio_local_repair",
    )
    parser.add_argument("--label", default="", help="Finalist label to repair, e.g. top9.")
    parser.add_argument("--dictionary", default="resources/dictionaries/german_common.txt")
    parser.add_argument("--consensus-top-n", type=int, default=12)
    parser.add_argument("--consensus-min-agreement", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--window-step", type=int, default=40)
    parser.add_argument("--windows-per-page", type=int, default=5)
    parser.add_argument("--min-word-len", type=int, default=5)
    parser.add_argument("--max-word-len", type=int, default=14)
    parser.add_argument("--max-edits", type=int, default=4)
    parser.add_argument("--max-hypotheses", type=int, default=160)
    parser.add_argument("--max-hypotheses-per-window", type=int, default=16)
    parser.add_argument("--include-hypothesis-pairs", action="store_true")
    parser.add_argument("--pair-candidate-limit", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=120)
    parser.add_argument(
        "--max-hypothesis-set-size",
        type=int,
        default=1,
        help="Maximum compatible word-hypothesis set size to evaluate. Singletons are always included.",
    )
    parser.add_argument(
        "--combination-candidate-limit",
        type=int,
        default=32,
        help="Top hypotheses considered when building multi-hypothesis combinations.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=800,
        help="Maximum multi-hypothesis combinations to evaluate after compatibility filtering.",
    )
    parser.add_argument("--max-combined-edits", type=int, default=6)
    parser.add_argument("--allow-stable-edits", action="store_true")
    parser.add_argument(
        "--store-all-variants",
        action="store_true",
        help="Store compact summaries for every evaluated variant for rank-curve diagnostics.",
    )
    parser.add_argument("--acceptance-margin", type=float, default=0.03)
    parser.add_argument("--min-page-drop", type=float, default=0.02)
    parser.add_argument("--max-illusion-increase", type=float, default=0.02)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--progress", action="store_true")
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
    selected = next((row for row in finalists if row.get("_label") == label), None)
    if selected is None:
        raise SystemExit(f"No finalist label {label!r} found.")
    baseline_key = parse_key(selected.get("key"))
    baseline_mask = tuple(str(symbol) for symbol in (selected.get("mask") or []))
    consensus = consensus_from_finalists(
        artifact=artifact,
        alphabet=alphabet,
        top_n=args.consensus_top_n,
        min_agreement=args.consensus_min_agreement,
    )
    dictionary_path = resolve_path(Path(args.dictionary))
    dictionary = load_dictionary(dictionary_path, args.min_word_len, args.max_word_len)
    collateral_dictionary = load_dictionary(dictionary_path, 3, args.max_word_len)
    collateral_index = build_word_evidence_index(collateral_dictionary)
    page_windows = build_page_windows(
        pages=pages,
        alphabet=alphabet,
        key=baseline_key,
        mask=baseline_mask,
        consensus=consensus,
        window_size=args.window_size,
        window_step=args.window_step,
        windows_per_page=args.windows_per_page,
    )
    hypotheses = generate_word_hypotheses(
        page_windows=page_windows,
        dictionary=dictionary,
        consensus=consensus,
        alphabet=alphabet,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        min_word_len=args.min_word_len,
        max_word_len=args.max_word_len,
        max_edits=args.max_edits,
        max_per_window=args.max_hypotheses_per_window,
        allow_stable_edits=args.allow_stable_edits,
    )
    hypotheses = hypotheses[: max(0, args.max_hypotheses)]
    max_hypothesis_set_size = max(
        1,
        int(args.max_hypothesis_set_size),
        2 if args.include_hypothesis_pairs else 1,
    )
    combination_candidate_limit = (
        args.pair_candidate_limit
        if args.include_hypothesis_pairs and args.combination_candidate_limit == 32
        else args.combination_candidate_limit
    )
    max_combinations = (
        args.max_pairs
        if args.include_hypothesis_pairs and args.max_combinations == 800
        else args.max_combinations
    )
    variants = evaluate_hypotheses(
        pages=pages,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        hypotheses=hypotheses,
        max_hypothesis_set_size=max_hypothesis_set_size,
        combination_candidate_limit=combination_candidate_limit,
        max_combinations=max_combinations,
        max_combined_edits=args.max_combined_edits,
        collateral_index=collateral_index,
        edit_support=build_edit_support(hypotheses),
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
        allow_pair_acceptance=True,
    )
    annotate_repair_evidence(variants, baseline=baseline_variant)
    second_stage = attach_second_stage_portfolio(variants, baseline=baseline_variant)
    variants.sort(key=variant_rank_key, reverse=True)
    accepted = [row for row in variants if row.get("repair_acceptance", {}).get("accepted")]

    payload = {
        "experiment": "copiale_word_hypothesis_repair_probe",
        "source_experiment": str(experiment_path),
        "source_artifact": str(artifact_path),
        "section": args.section,
        "label": label,
        "test_ids": test_ids,
        "dictionary": str(resolve_path(Path(args.dictionary))),
        "hypothesis_count": len(hypotheses),
        "variant_count": len(variants),
        "accepted_variant_count": len(accepted),
        "settings": {
            "min_word_len": args.min_word_len,
            "max_word_len": args.max_word_len,
            "max_edits": args.max_edits,
            "max_hypotheses": args.max_hypotheses,
            "max_hypotheses_per_window": args.max_hypotheses_per_window,
            "max_hypothesis_set_size": max_hypothesis_set_size,
            "combination_candidate_limit": combination_candidate_limit,
            "max_combinations": max_combinations,
            "max_combined_edits": args.max_combined_edits,
            "allow_stable_edits": args.allow_stable_edits,
            "store_all_variants": args.store_all_variants,
        },
        "page_windows": [
            {
                "test_id": row["test_id"],
                "windows": [
                    {
                        "start": window["start"],
                        "end": window["end"],
                        "damage_score": window["damage_score"],
                        "text": window["text"],
                    }
                    for window in row["windows"]
                ],
            }
            for row in page_windows
        ],
        "top_word_hypotheses": [hypothesis_to_dict(row) for row in hypotheses[: args.top_n]],
        "baseline": word_variant_summary(baseline_variant) if baseline_variant else {},
        "second_stage_portfolio": second_stage,
        "top_accepted_variants": [word_variant_summary(row) for row in accepted[: args.top_n]],
        "top_variants": [word_variant_summary(row) for row in variants[: args.top_n]],
        "top_variants_by_second_stage": [
            word_variant_summary(row)
            for row in sorted(
                variants,
                key=lambda row: (
                    float((row.get("second_stage") or {}).get("portfolio_score") or 0.0),
                    float((row.get("second_stage") or {}).get("rank_fusion_score") or 0.0),
                    float(row.get("page_robust_score") or 0.0),
                ),
                reverse=True,
            )[: args.top_n]
        ],
        "top_variants_by_second_stage_review": [
            word_variant_summary(row)
            for row in sorted(
                [
                    row for row in variants
                    if (row.get("second_stage") or {}).get("diverse_review_member")
                ],
                key=lambda row: int((row.get("second_stage") or {}).get("diverse_review_rank") or 10_000),
            )
        ],
        "top_variants_by_word_hypothesis": [
            word_variant_summary(row)
            for row in sorted(
                variants,
                key=lambda row: (
                    float(row.get("word_hypothesis_score") or 0.0),
                    float(row.get("page_robust_score") or 0.0),
                ),
                reverse=True,
            )[: args.top_n]
        ],
        "top_variants_by_adjudication": [
            word_variant_summary(row)
            for row in sorted(
                variants,
                key=lambda row: (
                    float((row.get("repair_adjudication") or {}).get("adjudication_score") or 0.0),
                    float(row.get("page_robust_score") or 0.0),
                ),
                reverse=True,
            )[: args.top_n]
        ],
        "top_combination_variants_by_adjudication": [
            word_variant_summary(row)
            for row in sorted(
                [row for row in variants if len(row.get("word_hypotheses") or []) > 1],
                key=lambda row: (
                    float((row.get("repair_adjudication") or {}).get("adjudication_score") or 0.0),
                    float(row.get("page_robust_score") or 0.0),
                ),
                reverse=True,
            )[: args.top_n]
        ],
        "top_combination_variants_by_marginal": [
            word_variant_summary(row)
            for row in sorted(
                [row for row in variants if len(row.get("word_hypotheses") or []) > 1],
                key=lambda row: (
                    float((row.get("marginal_contribution") or {}).get("marginal_selector_score") or 0.0),
                    float((row.get("repair_adjudication") or {}).get("adjudication_score") or 0.0),
                ),
                reverse=True,
            )[: args.top_n]
        ],
        "top_variants_by_post_hoc": [
            word_variant_summary(row)
            for row in sorted(
                variants,
                key=lambda row: float(row.get("post_hoc_char_avg") or 0.0),
                reverse=True,
            )[: args.top_n]
        ],
    }
    if args.store_all_variants:
        payload["all_variants"] = [
            compact_variant_summary(row)
            for row in sorted(
                variants,
                key=lambda row: float(row.get("post_hoc_char_avg") or 0.0),
                reverse=True,
            )
        ]
    markdown = render_markdown(payload)
    output = (
        resolve_path(Path(args.output))
        if args.output
        else experiment_path.with_suffix(f".{args.section}.{label}.word_hypothesis_repair.md")
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


def build_page_windows(
    *,
    pages: list[PageBundle],
    alphabet: Any,
    key: dict[int, int],
    mask: tuple[str, ...],
    consensus: dict[str, dict[str, Any]],
    window_size: int,
    window_step: int,
    windows_per_page: int,
) -> list[dict[str, Any]]:
    rows = []
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
        rows.append({
            "test_id": page.test_id,
            "text": text,
            "sources": sources,
            "windows": windows,
        })
    return rows


def generate_word_hypotheses(
    *,
    page_windows: list[dict[str, Any]],
    dictionary: dict[int, list[tuple[str, int]]],
    consensus: dict[str, dict[str, Any]],
    alphabet: Any,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    min_word_len: int,
    max_word_len: int,
    max_edits: int,
    max_per_window: int,
    allow_stable_edits: bool,
) -> list[WordHypothesis]:
    hypotheses: list[WordHypothesis] = []
    seen: set[tuple[str, int, int, str, tuple[tuple[str, str], ...]]] = set()
    for page in page_windows:
        text = str(page["text"])
        sources = list(page["sources"])
        for window in page["windows"]:
            window_start = int(window.get("start") or 0)
            window_end = int(window.get("end") or window_start)
            local_rows: list[WordHypothesis] = []
            for start in range(window_start, max(window_start, window_end - min_word_len + 1)):
                for length in range(min_word_len, max_word_len + 1):
                    end = start + length
                    if end > window_end or end > len(text):
                        continue
                    observed = text[start:end]
                    if not observed.isalpha():
                        continue
                    for target, dictionary_rank in dictionary.get(length, []):
                        distance = hamming_distance(observed, target)
                        if distance <= 0 or distance > max_edits:
                            continue
                        edits = implied_edits(
                            observed=observed,
                            target=target,
                            sources=sources[start:end],
                            consensus=consensus,
                            alphabet=alphabet,
                            baseline_key=baseline_key,
                            baseline_mask=baseline_mask,
                            allow_stable_edits=allow_stable_edits,
                        )
                        if not edits:
                            continue
                        if len(edits) > max_edits:
                            continue
                        key = (str(page["test_id"]), start, end, target, edits)
                        if key in seen:
                            continue
                        seen.add(key)
                        rank_bonus = 1.0 / max(1.0, dictionary_rank ** 0.35)
                        score = length - distance * 1.7 + rank_bonus * 3.0 + float(window.get("damage_score") or 0.0)
                        local_rows.append(
                            WordHypothesis(
                                test_id=str(page["test_id"]),
                                window_start=window_start,
                                start=start,
                                end=end,
                                observed=observed,
                                target=target,
                                edits=edits,
                                distance=distance,
                                dictionary_rank=dictionary_rank,
                                local_score=score,
                            )
                        )
            local_rows.sort(
                key=lambda row: (
                    row.local_score,
                    len(row.target),
                    -row.distance,
                    -row.dictionary_rank,
                ),
                reverse=True,
            )
            hypotheses.extend(local_rows[: max(0, max_per_window)])
    hypotheses.sort(
        key=lambda row: (
            row.local_score,
            len(row.target),
            -row.distance,
            -row.dictionary_rank,
        ),
        reverse=True,
    )
    return dedupe_hypotheses_by_edits(hypotheses)


def implied_edits(
    *,
    observed: str,
    target: str,
    sources: list[str],
    consensus: dict[str, dict[str, Any]],
    alphabet: Any,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    allow_stable_edits: bool,
) -> tuple[tuple[str, str], ...]:
    symbol_targets: dict[str, str] = {}
    masked = set(baseline_mask)
    for observed_char, target_char, symbol in zip(observed, target, sources):
        if observed_char == target_char:
            continue
        if not symbol or not alphabet.has_symbol(symbol):
            return ()
        info = consensus.get(symbol) or {}
        if info.get("stable") and not allow_stable_edits:
            return ()
        if symbol in masked:
            return ()
        prior = symbol_targets.get(symbol)
        if prior is not None and prior != target_char:
            return ()
        token_id = alphabet.id_for(symbol)
        current = current_assignment(symbol, token_id, baseline_key, baseline_mask)
        if current == target_char:
            continue
        symbol_targets[symbol] = target_char
    edits = tuple(sorted(symbol_targets.items()))
    return edits


def evaluate_hypotheses(
    *,
    pages: list[PageBundle],
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    hypotheses: list[WordHypothesis],
    max_hypothesis_set_size: int,
    combination_candidate_limit: int,
    max_combinations: int,
    max_combined_edits: int,
    collateral_index: WordEvidenceIndex,
    edit_support: dict[tuple[str, str], dict[str, Any]],
    progress: bool,
) -> list[dict[str, Any]]:
    hypothesis_sets = build_hypothesis_sets(
        hypotheses=hypotheses,
        max_hypothesis_set_size=max_hypothesis_set_size,
        combination_candidate_limit=combination_candidate_limit,
        max_combinations=max_combinations,
        max_combined_edits=max_combined_edits,
    )

    variants: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = set()
    started = time.monotonic()
    total = len(hypothesis_sets)
    progress_every = max(1, min(50, total // 20 if total >= 20 else total))
    baseline_page_rows = project_pages(pages=pages, key=baseline_key, mask=tuple(sorted(baseline_mask)))
    attach_page_scores(baseline_page_rows)
    for index, hypothesis_set in enumerate(hypothesis_sets, start=1):
        edit_map = combined_edit_map(hypothesis_set)
        if edit_map is None:
            continue
        key = dict(baseline_key)
        mask = set(baseline_mask)
        edits = []
        for symbol, target in sorted(edit_map.items()):
            token_id = next_token_id(pages, symbol)
            before = current_assignment(symbol, token_id, key, tuple(sorted(mask)))
            apply_assignment(symbol, token_id, target, key, mask)
            edits.append(f"{symbol}:{before}->{target}")
        identity = (tuple(sorted(mask)), tuple(sorted(key.items())))
        if identity in seen:
            continue
        seen.add(identity)
        if progress and (index == 1 or index == total or index % progress_every == 0):
            elapsed = time.monotonic() - started
            print(
                f"Evaluating word-hypothesis variants: {index}/{total} "
                f"({index / max(1, total):.0%}) elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        page_rows = project_pages(pages=pages, key=key, mask=tuple(sorted(mask)))
        attach_page_scores(page_rows)
        runtime_scores = [score_page_runtime(row, key=key, mask=tuple(sorted(mask))) for row in page_rows]
        metrics = page_runtime_metrics(runtime_scores)
        avg_char = sum(float(row.get("char_accuracy") or 0.0) for row in page_rows) / max(1, len(page_rows))
        baseline_no_target_char = post_hoc_char_excluding_target_words(
            baseline_page_rows,
            hypothesis_set,
        )
        baseline_target_only_char = post_hoc_char_target_words_only(
            baseline_page_rows,
            hypothesis_set,
        )
        no_target_char = post_hoc_char_excluding_target_words(page_rows, hypothesis_set)
        target_only_char = post_hoc_char_target_words_only(page_rows, hypothesis_set)
        adjudication = adjudicate_repair(
            pages=pages,
            before_key=baseline_key,
            after_key=key,
            mask=tuple(sorted(mask)),
            hypotheses=hypothesis_set,
            edits=edit_map,
            word_index=collateral_index,
            edit_support=edit_support,
        )
        adjudication["word_hypothesis_score"] = round(sum(row.local_score for row in hypothesis_set), 6)
        adjudication["adjudication_score"] = round(
            adjudication_score(adjudication, sum(row.local_score for row in hypothesis_set)),
            6,
        )
        adjudication["adjudication_no_target_score"] = round(
            adjudication_no_target_score(adjudication),
            6,
        )
        adjudication["target_leverage_score"] = round(
            float(adjudication["adjudication_score"])
            - float(adjudication["adjudication_no_target_score"]),
            6,
        )
        variants.append({
            "edits": edits or ["baseline"],
            "mask": list(sorted(mask)),
            "word_hypotheses": [hypothesis_to_dict(row) for row in hypothesis_set],
            "word_hypothesis_score": round(sum(row.local_score for row in hypothesis_set), 6),
            "repair_adjudication": adjudication,
            "page_runtime_scores": runtime_scores,
            "post_hoc_char_avg": round(avg_char, 6),
            "post_hoc_char_no_target_baseline_avg": (
                round(baseline_no_target_char, 6) if baseline_no_target_char is not None else None
            ),
            "post_hoc_char_target_baseline_avg": (
                round(baseline_target_only_char, 6) if baseline_target_only_char is not None else None
            ),
            "post_hoc_char_no_target_avg": (
                round(no_target_char, 6) if no_target_char is not None else None
            ),
            "post_hoc_char_target_avg": (
                round(target_only_char, 6) if target_only_char is not None else None
            ),
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
    attach_marginal_contribution_diagnostics(variants)
    return variants


def attach_marginal_contribution_diagnostics(variants: list[dict[str, Any]]) -> None:
    by_hypothesis_signature: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in variants:
        signature = variant_hypothesis_signature(row)
        by_hypothesis_signature[signature] = row
    for row in variants:
        signature = variant_hypothesis_signature(row)
        if len(signature) <= 1:
            row["marginal_contribution"] = singleton_marginal_summary(row)
            continue
        subset_rows = []
        for size in range(1, len(signature)):
            for subset in itertools.combinations(signature, size):
                subset_row = by_hypothesis_signature.get(tuple(sorted(subset)))
                if subset_row:
                    subset_rows.append(subset_row)
        best_subset = max(
            subset_rows,
            key=lambda item: float((item.get("repair_adjudication") or {}).get("adjudication_score") or 0.0),
            default={},
        )
        best_singleton = max(
            [item for item in subset_rows if len(variant_hypothesis_signature(item)) == 1],
            key=lambda item: float((item.get("repair_adjudication") or {}).get("adjudication_score") or 0.0),
            default={},
        )
        member_margins = []
        for member in signature:
            without = tuple(item for item in signature if item != member)
            without_row = by_hypothesis_signature.get(tuple(sorted(without))) if without else {}
            member_margins.append(marginal_member_summary(row, member, without_row or {}))
        summary = {
            "hypothesis_count": len(signature),
            "best_subset": compact_marginal_reference(best_subset),
            "best_singleton": compact_marginal_reference(best_singleton),
            "delta_vs_best_subset": metric_deltas(row, best_subset),
            "delta_vs_best_singleton": metric_deltas(row, best_singleton),
            "member_margins": member_margins,
        }
        summary["negative_member_count"] = sum(
            1 for item in member_margins if float(item.get("delta_adjudication_score") or 0.0) <= 0.0
        )
        summary["marginal_selector_score"] = round(marginal_selector_score(row, summary), 6)
        row["marginal_contribution"] = summary


SECOND_STAGE_RANKERS: tuple[tuple[str, str, float], ...] = (
    ("adjudication", "adjudication_score", 0.23),
    ("global_leverage", "global_leverage_score", 0.20),
    ("marginal", "marginal_selector_score", 0.17),
    ("adjudication_no_target", "adjudication_no_target_score", 0.08),
    ("runtime_robust", "page_robust_score", 0.12),
    ("runtime_validation", "page_validation_avg", 0.08),
    ("language_quality", "page_language_quality_avg", 0.07),
    ("word_hypothesis", "word_hypothesis_score", 0.05),
)


def attach_second_stage_portfolio(
    variants: list[dict[str, Any]],
    *,
    baseline: dict[str, Any],
    per_ranker_limit: int = 40,
    per_edit_limit: int = 3,
) -> dict[str, Any]:
    """Attach a conservative second-stage portfolio score.

    Stage 1 intentionally throws a broad repair menu at the wall. Stage 2 does
    not try to become a single clever scalar; it asks whether a candidate
    remains interesting across several independent runtime-only views, while
    keeping diverse edit families visible for human/iterative follow-up.
    """
    if not variants:
        return {
            "rankers": [],
            "shortlist_count": 0,
            "selected": [],
        }

    rank_maps = build_second_stage_rank_maps(variants)
    shortlist = build_second_stage_shortlist(
        variants,
        rank_maps=rank_maps,
        per_ranker_limit=per_ranker_limit,
        per_edit_limit=per_edit_limit,
    )
    baseline_metrics = second_stage_metric_snapshot(baseline)
    for row in variants:
        score, details = second_stage_score(
            row,
            rank_maps=rank_maps,
            shortlist=shortlist,
            baseline_metrics=baseline_metrics,
        )
        row["second_stage"] = {
            "portfolio_score": round(score, 6),
            **details,
        }

    selected = sorted(
        shortlist,
        key=lambda row: (
            float((row.get("second_stage") or {}).get("portfolio_score") or 0.0),
            float((row.get("second_stage") or {}).get("rank_fusion_score") or 0.0),
            float(row.get("page_robust_score") or 0.0),
        ),
        reverse=True,
    )
    diverse_review = diverse_review_shortlist(selected, rank_maps=rank_maps, limit=8)
    for rank, row in enumerate(diverse_review, start=1):
        stage = row.get("second_stage") if isinstance(row.get("second_stage"), dict) else {}
        stage["diverse_review_member"] = True
        stage["diverse_review_rank"] = rank
        row["second_stage"] = stage
    return {
        "rankers": [
            {"name": name, "metric": metric, "weight": weight}
            for name, metric, weight in SECOND_STAGE_RANKERS
        ],
        "per_ranker_limit": per_ranker_limit,
        "per_edit_limit": per_edit_limit,
        "shortlist_count": len(shortlist),
        "review_shortlist_policy": "diverse_review_k8",
        "review_shortlist": [second_stage_reference(row) for row in diverse_review],
        "selected": [second_stage_reference(row) for row in selected[:24]],
    }


def build_second_stage_rank_maps(
    variants: list[dict[str, Any]]
) -> dict[str, dict[int, int]]:
    rank_maps: dict[str, dict[int, int]] = {}
    for _name, metric, _weight in SECOND_STAGE_RANKERS:
        ranked = sorted(
            enumerate(variants),
            key=lambda item: (
                second_stage_metric_value(item[1], metric),
                float(item[1].get("page_robust_score") or 0.0),
                -len(item[1].get("edits") or []),
            ),
            reverse=True,
        )
        rank_maps[metric] = {id(variant): rank for rank, (_index, variant) in enumerate(ranked, start=1)}
    return rank_maps


def build_second_stage_shortlist(
    variants: list[dict[str, Any]],
    *,
    rank_maps: dict[str, dict[int, int]],
    per_ranker_limit: int,
    per_edit_limit: int,
) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for _name, metric, _weight in SECOND_STAGE_RANKERS:
        ranked = sorted(
            variants,
            key=lambda row: (
                second_stage_metric_value(row, metric),
                float(row.get("page_robust_score") or 0.0),
            ),
            reverse=True,
        )
        for row in ranked[: max(0, per_ranker_limit)]:
            selected[id(row)] = row

    # Force edit-family diversity. Single scalar rankers have repeatedly
    # preferred clusters of very similar word-island repairs; this keeps
    # alternatives available for the second-stage selector and for iteration.
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in variants:
        for edit in row.get("edits") or []:
            symbol = str(edit).split(":", 1)[0]
            if symbol and symbol != "baseline":
                by_symbol.setdefault(symbol, []).append(row)
    for rows in by_symbol.values():
        rows.sort(
            key=lambda row: (
                rank_fusion_score(row, rank_maps),
                float((row.get("repair_adjudication") or {}).get("adjudication_no_target_score") or 0.0),
                float(row.get("page_robust_score") or 0.0),
            ),
            reverse=True,
        )
        for row in rows[: max(0, per_edit_limit)]:
            selected[id(row)] = row

    if variants:
        selected[id(variants[0])] = variants[0]
    return list(selected.values())


def second_stage_score(
    row: dict[str, Any],
    *,
    rank_maps: dict[str, dict[int, int]],
    shortlist: list[dict[str, Any]],
    baseline_metrics: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    metric_snapshot = second_stage_metric_snapshot(row)
    rank_fusion = rank_fusion_score(row, rank_maps)
    deltas = {
        metric: metric_snapshot.get(metric, 0.0) - baseline_metrics.get(metric, 0.0)
        for _name, metric, _weight in SECOND_STAGE_RANKERS
    }
    adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    marginal = row.get("marginal_contribution") if isinstance(row.get("marginal_contribution"), dict) else {}
    target_only_penalty_value = float(adjudication.get("target_only_penalty") or 0.0)
    negative_members = float(marginal.get("negative_member_count") or 0.0)
    collateral_damage = float(adjudication.get("collateral_damage_sum") or 0.0)
    collateral_word_damage = float(adjudication.get("collateral_word_damage_weighted_sum") or 0.0)
    damaged_words = float(adjudication.get("word_damaged_weighted_occurrences") or 0.0)
    edited_symbols = max(1.0, float(adjudication.get("edited_symbol_count") or 1.0))
    hypothesis_count = max(1.0, float(len(row.get("word_hypotheses") or [])))
    collateral_occurrences = float(adjudication.get("collateral_occurrences") or 0.0)
    occurrence_count = float(adjudication.get("occurrence_count") or 0.0)
    robust_delta = deltas.get("page_robust_score", 0.0)
    validation_delta = deltas.get("page_validation_avg", 0.0)
    no_target_delta = deltas.get("adjudication_no_target_score", 0.0)
    global_delta = deltas.get("global_leverage_score", 0.0)
    no_target_slack = 0.35
    support_bonus = min(1.25, collateral_occurrences / 18.0)
    breadth_bonus = min(0.8, occurrence_count / 35.0) + min(0.6, edited_symbols / 4.0)
    improvement_bonus = (
        1.8 * max(0.0, no_target_delta)
        + 1.15 * max(0.0, global_delta)
        + 5.0 * max(0.0, robust_delta)
        + 2.5 * max(0.0, validation_delta)
    )
    regression_penalty = (
        3.0 * max(0.0, -no_target_delta - no_target_slack)
        + 2.0 * max(0.0, -global_delta)
        + 8.0 * max(0.0, -robust_delta)
        + 4.0 * max(0.0, -validation_delta)
    )
    singleton_bonus = 0.65 if hypothesis_count <= 1.0 and edited_symbols <= 2.0 else 0.0
    bundle_penalty = max(0.0, hypothesis_count - 1.0) * 3.0 + max(0.0, edited_symbols - 2.0) * 0.85
    risk_penalty = (
        0.18 * collateral_damage
        + 0.45 * collateral_word_damage
        + 0.28 * damaged_words
        + 0.55 * negative_members
        + 0.55 * min(5.0, target_only_penalty_value)
        + bundle_penalty
    )
    shortlist_bonus = 0.45 if any(row is item for item in shortlist) else 0.0
    score = (
        12.0 * rank_fusion
        + improvement_bonus
        + support_bonus
        + breadth_bonus
        + singleton_bonus
        + shortlist_bonus
        - regression_penalty
        - risk_penalty
    )
    return score, {
        "rank_fusion_score": round(rank_fusion, 6),
        "metric_deltas_vs_baseline": {
            key: round(value, 6) for key, value in sorted(deltas.items())
        },
        "support_bonus": round(support_bonus, 6),
        "breadth_bonus": round(breadth_bonus, 6),
        "singleton_bonus": round(singleton_bonus, 6),
        "bundle_penalty": round(bundle_penalty, 6),
        "improvement_bonus": round(improvement_bonus, 6),
        "regression_penalty": round(regression_penalty, 6),
        "risk_penalty": round(risk_penalty, 6),
        "no_target_slack": no_target_slack,
        "shortlist_member": any(row is item for item in shortlist),
        "flags": second_stage_flags(
            target_only_penalty_value=target_only_penalty_value,
            collateral_damage=collateral_damage,
            collateral_word_damage=collateral_word_damage,
            negative_members=negative_members,
            robust_delta=robust_delta,
            validation_delta=validation_delta,
            no_target_delta=no_target_delta,
            global_delta=global_delta,
            collateral_occurrences=collateral_occurrences,
            hypothesis_count=hypothesis_count,
        ),
        "ranks": second_stage_ranks(row, rank_maps),
    }


def rank_fusion_score(row: dict[str, Any], rank_maps: dict[str, dict[int, int]]) -> float:
    score = 0.0
    for _name, metric, weight in SECOND_STAGE_RANKERS:
        rank = rank_maps.get(metric, {}).get(id(row), 10_000)
        score += weight / (8.0 + float(rank))
    return score


def second_stage_metric_snapshot(row: dict[str, Any]) -> dict[str, float]:
    return {
        metric: second_stage_metric_value(row, metric)
        for _name, metric, _weight in SECOND_STAGE_RANKERS
    }


def second_stage_metric_value(row: dict[str, Any], metric: str) -> float:
    adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    marginal = row.get("marginal_contribution") if isinstance(row.get("marginal_contribution"), dict) else {}
    if metric in adjudication:
        return float(adjudication.get(metric) or 0.0)
    if metric in marginal:
        return float(marginal.get(metric) or 0.0)
    return float(row.get(metric) or 0.0)


def second_stage_ranks(
    row: dict[str, Any],
    rank_maps: dict[str, dict[int, int]],
) -> dict[str, int]:
    return {
        metric: int(rank_maps.get(metric, {}).get(id(row), 10_000))
        for _name, metric, _weight in SECOND_STAGE_RANKERS
    }


def second_stage_flags(
    *,
    target_only_penalty_value: float,
    collateral_damage: float,
    collateral_word_damage: float,
    negative_members: float,
    robust_delta: float,
    validation_delta: float,
    no_target_delta: float,
    global_delta: float,
    collateral_occurrences: float,
    hypothesis_count: float,
) -> list[str]:
    flags = []
    if hypothesis_count > 1:
        flags.append("bundle")
    if target_only_penalty_value > 0.0:
        flags.append("target_only")
    if collateral_occurrences <= 1:
        flags.append("low_collateral_support")
    if negative_members > 0:
        flags.append("negative_combo_member")
    if collateral_damage > 0.35 or collateral_word_damage > 0.75:
        flags.append("collateral_damage")
    if robust_delta < -0.015:
        flags.append("runtime_robust_down")
    if validation_delta < -0.02:
        flags.append("validation_down")
    if no_target_delta < -0.7 and global_delta > 0.25:
        flags.append("global_no_target_disagree")
    return flags


def second_stage_reference(row: dict[str, Any]) -> dict[str, Any]:
    stage = row.get("second_stage") if isinstance(row.get("second_stage"), dict) else {}
    return {
        "edits": row.get("edits") or [],
        "hypotheses": variant_hypothesis_signature(row),
        "portfolio_score": stage.get("portfolio_score"),
        "rank_fusion_score": stage.get("rank_fusion_score"),
        "singleton_bonus": stage.get("singleton_bonus"),
        "bundle_penalty": stage.get("bundle_penalty"),
        "flags": stage.get("flags") or [],
        "ranks": stage.get("ranks") or {},
        "post_hoc_char_avg": row.get("post_hoc_char_avg"),
        "post_hoc_char_no_target_avg": row.get("post_hoc_char_no_target_avg"),
        "preview": str(row.get("preview") or "")[:220],
    }


def diverse_review_shortlist(
    rows: list[dict[str, Any]],
    *,
    rank_maps: dict[str, dict[int, int]],
    limit: int,
) -> list[dict[str, Any]]:
    """Return a compact human/agent review shortlist with edit-family diversity."""
    selected: list[dict[str, Any]] = []
    seen_edit_families: set[str] = set()
    singleton_rows = [
        row for row in rows
        if len(row.get("word_hypotheses") or []) <= 1
        and count_real_edits(row) <= 2
    ]
    bundle_rows = [row for row in rows if row not in singleton_rows]
    # Round-robin through independent rankers before filling by portfolio. This
    # is intentionally a review menu, not a scalar winner-take-all choice.
    for _name, metric, _weight in SECOND_STAGE_RANKERS:
        ranked = sorted(
            singleton_rows,
            key=lambda row: rank_maps.get(metric, {}).get(id(row), 10_000),
        )
        add_diverse_review_row(ranked, selected, seen_edit_families, limit=limit)
        if len(selected) >= limit:
            return selected
    for row in singleton_rows + bundle_rows:
        add_diverse_review_row([row], selected, seen_edit_families, limit=limit)
        if len(selected) >= limit:
            break
    return selected


def add_diverse_review_row(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    seen_edit_families: set[str],
    *,
    limit: int,
) -> None:
    for row in rows:
        if row in selected:
            continue
        family = edit_family_signature(row)
        if family in seen_edit_families and len(selected) < max(2, limit // 2):
            continue
        selected.append(row)
        seen_edit_families.add(family)
        return


def edit_family_signature(row: dict[str, Any]) -> str:
    edits = [str(edit).split(":", 1)[0] for edit in (row.get("edits") or [])]
    edits = [edit for edit in edits if edit and edit != "baseline"]
    if not edits:
        return "baseline"
    return "+".join(sorted(edits))


def count_real_edits(row: dict[str, Any]) -> int:
    return sum(1 for edit in (row.get("edits") or []) if str(edit).strip().lower() != "baseline")


def post_hoc_char_excluding_target_words(
    page_rows: list[dict[str, Any]],
    hypotheses: tuple[WordHypothesis, ...],
) -> float | None:
    ranges_by_page = hypothesis_ranges_by_page(hypotheses)
    if not ranges_by_page:
        return None
    scores = []
    for row in page_rows:
        ranges = ranges_by_page.get(str(row.get("test_id") or ""))
        if not ranges:
            continue
        decryption = str(row.get("decryption") or "")
        plaintext = str(row.get("plaintext") or "")
        if not decryption or not plaintext:
            continue
        redacted_decryption = remove_ranges(decryption, ranges)
        redacted_plaintext = remove_ranges(plaintext, ranges)
        if not redacted_decryption or not redacted_plaintext:
            continue
        score = score_decryption(
            str(row.get("test_id") or "target_excluded"),
            redacted_decryption,
            redacted_plaintext,
            agent_score=0.0,
            status="completed",
        )
        scores.append(float(score.char_accuracy))
    if not scores:
        return None
    return sum(scores) / len(scores)


def post_hoc_char_target_words_only(
    page_rows: list[dict[str, Any]],
    hypotheses: tuple[WordHypothesis, ...],
) -> float | None:
    ranges_by_page = hypothesis_ranges_by_page(hypotheses)
    if not ranges_by_page:
        return None
    scores = []
    for row in page_rows:
        ranges = ranges_by_page.get(str(row.get("test_id") or ""))
        if not ranges:
            continue
        decryption = str(row.get("decryption") or "")
        plaintext = str(row.get("plaintext") or "")
        target_decryption = keep_ranges(decryption, ranges)
        target_plaintext = keep_ranges(plaintext, ranges)
        if not target_decryption or not target_plaintext:
            continue
        score = score_decryption(
            str(row.get("test_id") or "target_only"),
            target_decryption,
            target_plaintext,
            agent_score=0.0,
            status="completed",
        )
        scores.append(float(score.char_accuracy))
    if not scores:
        return None
    return sum(scores) / len(scores)


def hypothesis_ranges_by_page(
    hypotheses: tuple[WordHypothesis, ...],
) -> dict[str, list[tuple[int, int]]]:
    ranges: dict[str, list[tuple[int, int]]] = {}
    for hypothesis in hypotheses:
        ranges.setdefault(hypothesis.test_id, []).append((hypothesis.start, hypothesis.end))
    return {test_id: merge_ranges(page_ranges) for test_id, page_ranges in ranges.items()}


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((max(0, start), max(0, end)) for start, end in ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def remove_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    if not ranges:
        return text
    chunks = []
    cursor = 0
    for start, end in ranges:
        start = min(max(0, start), len(text))
        end = min(max(start, end), len(text))
        chunks.append(text[cursor:start])
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def keep_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    chunks = []
    for start, end in ranges:
        start = min(max(0, start), len(text))
        end = min(max(start, end), len(text))
        chunks.append(text[start:end])
    return "".join(chunks)


def singleton_marginal_summary(row: dict[str, Any]) -> dict[str, Any]:
    adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    return {
        "hypothesis_count": len(variant_hypothesis_signature(row)),
        "marginal_selector_score": adjudication.get("adjudication_score"),
        "negative_member_count": 0,
        "best_subset": {},
        "best_singleton": compact_marginal_reference(row),
        "delta_vs_best_subset": {},
        "delta_vs_best_singleton": {},
        "member_margins": [],
    }


def marginal_member_summary(
    row: dict[str, Any], member: str, without_row: dict[str, Any]
) -> dict[str, Any]:
    deltas = metric_deltas(row, without_row)
    return {
        "hypothesis": member,
        "without": compact_marginal_reference(without_row),
        "delta_adjudication_score": deltas.get("adjudication_score"),
        "delta_page_robust_score": deltas.get("page_robust_score"),
        "delta_page_validation_avg": deltas.get("page_validation_avg"),
        "delta_page_language_quality_avg": deltas.get("page_language_quality_avg"),
        "delta_post_hoc_char_avg": deltas.get("post_hoc_char_avg"),
        "delta_post_hoc_char_no_target_avg": deltas.get("post_hoc_char_no_target_avg"),
    }


def marginal_selector_score(row: dict[str, Any], summary: dict[str, Any]) -> float:
    adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    base_score = float(adjudication.get("adjudication_score") or 0.0)
    subset_delta = summary.get("delta_vs_best_subset") if isinstance(summary.get("delta_vs_best_subset"), dict) else {}
    robust_delta = float(subset_delta.get("page_robust_score") or 0.0)
    validation_delta = float(subset_delta.get("page_validation_avg") or 0.0)
    lq_delta = float(subset_delta.get("page_language_quality_avg") or 0.0)
    adjudication_delta = float(subset_delta.get("adjudication_score") or 0.0)
    negative_members = float(summary.get("negative_member_count") or 0.0)
    collateral_word_damage = float(adjudication.get("collateral_word_damage_sum") or 0.0)
    word_damaged = float(adjudication.get("word_damaged_occurrences") or 0.0)
    global_delta_score = (
        85.0 * robust_delta
        + 45.0 * validation_delta
        + 35.0 * lq_delta
        - 55.0 * max(0.0, -robust_delta)
        - 30.0 * max(0.0, -validation_delta)
    )
    collateral_penalty = 0.12 * collateral_word_damage + 0.06 * word_damaged
    summary["global_delta_score"] = round(global_delta_score, 6)
    summary["collateral_penalty"] = round(collateral_penalty, 6)
    return (
        0.45 * base_score
        + 0.55 * max(0.0, adjudication_delta)
        - 2.4 * max(0.0, -adjudication_delta)
        + global_delta_score
        - collateral_penalty
        - 0.9 * negative_members
    )


def metric_deltas(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    if not baseline:
        return {}
    row_adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    base_adjudication = (
        baseline.get("repair_adjudication")
        if isinstance(baseline.get("repair_adjudication"), dict)
        else {}
    )
    metrics = {
        "adjudication_score": (
            float(row_adjudication.get("adjudication_score") or 0.0)
            - float(base_adjudication.get("adjudication_score") or 0.0)
        ),
        "adjudication_no_target_score": (
            float(row_adjudication.get("adjudication_no_target_score") or 0.0)
            - float(base_adjudication.get("adjudication_no_target_score") or 0.0)
        ),
        "target_leverage_score": (
            float(row_adjudication.get("target_leverage_score") or 0.0)
            - float(base_adjudication.get("target_leverage_score") or 0.0)
        ),
    }
    for key in [
        "page_robust_score",
        "page_validation_avg",
        "page_language_quality_avg",
        "post_hoc_char_avg",
        "post_hoc_char_no_target_avg",
    ]:
        left = row.get(key)
        right = baseline.get(key)
        metrics[key] = (
            float(left) - float(right)
            if isinstance(left, (int, float)) and isinstance(right, (int, float))
            else None
        )
    return metrics


def compact_marginal_reference(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    return {
        "edits": row.get("edits") or [],
        "hypotheses": variant_hypothesis_signature(row),
        "adjudication_score": adjudication.get("adjudication_score"),
        "adjudication_no_target_score": adjudication.get("adjudication_no_target_score"),
        "target_leverage_score": adjudication.get("target_leverage_score"),
        "page_robust_score": row.get("page_robust_score"),
        "page_validation_avg": row.get("page_validation_avg"),
        "page_language_quality_avg": row.get("page_language_quality_avg"),
        "post_hoc_char_avg": row.get("post_hoc_char_avg"),
        "post_hoc_char_no_target_avg": row.get("post_hoc_char_no_target_avg"),
    }


def variant_hypothesis_signature(row: dict[str, Any]) -> tuple[str, ...]:
    hypotheses = row.get("word_hypotheses") if isinstance(row.get("word_hypotheses"), list) else []
    signature = []
    for item in hypotheses:
        if not isinstance(item, dict):
            continue
        edits = ",".join(str(edit) for edit in (item.get("edits") or []))
        signature.append(
            f"{item.get('test_id')}:{item.get('start')}-{item.get('end')}:"
            f"{item.get('observed')}->{item.get('target')}:{edits}"
        )
    return tuple(sorted(signature))


def build_hypothesis_sets(
    *,
    hypotheses: list[WordHypothesis],
    max_hypothesis_set_size: int,
    combination_candidate_limit: int,
    max_combinations: int,
    max_combined_edits: int,
) -> list[tuple[WordHypothesis, ...]]:
    hypothesis_sets: list[tuple[WordHypothesis, ...]] = [()]
    hypothesis_sets.extend((row,) for row in hypotheses)
    max_size = max(1, int(max_hypothesis_set_size))
    if max_size <= 1:
        return hypothesis_sets

    pool = hypotheses[: max(0, int(combination_candidate_limit))]
    combinations: list[tuple[WordHypothesis, ...]] = []
    seen_edits: set[tuple[tuple[str, str], ...]] = set()
    for size in range(2, max_size + 1):
        for candidate_set in itertools.combinations(pool, size):
            if not compatible_hypothesis_set(candidate_set, max_combined_edits=max_combined_edits):
                continue
            edit_map = combined_edit_map(candidate_set)
            if not edit_map:
                continue
            edit_signature = tuple(sorted(edit_map.items()))
            if edit_signature in seen_edits:
                continue
            seen_edits.add(edit_signature)
            combinations.append(candidate_set)

    combinations.sort(key=combination_prescreen_key, reverse=True)
    if max_combinations >= 0:
        combinations = combinations[: max(0, int(max_combinations))]
    hypothesis_sets.extend(combinations)
    return hypothesis_sets


def combination_prescreen_key(hypotheses: tuple[WordHypothesis, ...]) -> tuple[float, int, int, str]:
    pages = {row.test_id for row in hypotheses}
    windows = {(row.test_id, row.window_start) for row in hypotheses}
    edits = combined_edit_map(hypotheses) or {}
    return (
        sum(row.local_score for row in hypotheses),
        len(pages) + len(windows),
        -len(edits),
        pair_signature(hypotheses),
    )


def compatible_hypothesis_set(
    hypotheses: tuple[WordHypothesis, ...], *, max_combined_edits: int
) -> bool:
    edit_map: dict[str, str] = {}
    for hypothesis in hypotheses:
        for symbol, target in hypothesis.edits:
            old = edit_map.get(symbol)
            if old is not None and old != target:
                return False
            edit_map[symbol] = target
    return 0 < len(edit_map) <= max_combined_edits


def combined_edit_map(hypotheses: tuple[WordHypothesis, ...]) -> dict[str, str] | None:
    edit_map: dict[str, str] = {}
    for hypothesis in hypotheses:
        for symbol, target in hypothesis.edits:
            old = edit_map.get(symbol)
            if old is not None and old != target:
                return None
            edit_map[symbol] = target
    return edit_map


def build_edit_support(hypotheses: list[WordHypothesis]) -> dict[tuple[str, str], dict[str, Any]]:
    support: dict[tuple[str, str], dict[str, Any]] = {}
    targets_by_symbol: dict[str, set[str]] = {}
    for hypothesis in hypotheses:
        for symbol, target in hypothesis.edits:
            key = (symbol, target)
            row = support.setdefault(
                key,
                {
                    "hypothesis_count": 0,
                    "local_score_sum": 0.0,
                    "pages": set(),
                    "windows": set(),
                    "examples": [],
                },
            )
            row["hypothesis_count"] += 1
            row["local_score_sum"] += float(hypothesis.local_score)
            row["pages"].add(hypothesis.test_id)
            row["windows"].add((hypothesis.test_id, hypothesis.window_start))
            if len(row["examples"]) < 6:
                row["examples"].append(f"{hypothesis.observed}->{hypothesis.target}")
            targets_by_symbol.setdefault(symbol, set()).add(target)
    normalized = {}
    for key, row in support.items():
        symbol, _target = key
        competing_targets = max(0, len(targets_by_symbol.get(symbol, set())) - 1)
        normalized[key] = {
            "hypothesis_count": int(row["hypothesis_count"]),
            "local_score_sum": round(float(row["local_score_sum"]), 6),
            "page_count": len(row["pages"]),
            "window_count": len(row["windows"]),
            "competing_target_count": competing_targets,
            "examples": row["examples"],
        }
    return normalized


def next_token_id(pages: list[PageBundle], symbol: str) -> int:
    for page in pages:
        for page_symbol, token_id in zip(page.symbols, page.token_ids):
            if page_symbol == symbol:
                return token_id
    raise KeyError(symbol)


def adjudicate_repair(
    *,
    pages: list[PageBundle],
    before_key: dict[int, int],
    after_key: dict[int, int],
    mask: tuple[str, ...],
    hypotheses: tuple[WordHypothesis, ...],
    edits: dict[str, str],
    word_index: WordEvidenceIndex,
    edit_support: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    if not edits:
        return {
            "edited_symbol_count": 0,
            "occurrence_count": 0,
            "target_occurrences": 0,
            "collateral_occurrences": 0,
            "improved_occurrences": 0,
            "damaged_occurrences": 0,
            "target_gain_sum": 0.0,
            "collateral_gain_sum": 0.0,
            "collateral_damage_sum": 0.0,
            "occurrence_impacts": [],
        }
    target_ranges: dict[str, list[tuple[int, int]]] = {}
    for hypothesis in hypotheses:
        target_ranges.setdefault(hypothesis.test_id, []).append((hypothesis.start, hypothesis.end))
    target_repairs = target_word_repairs(
        pages=pages,
        before_key=before_key,
        after_key=after_key,
        mask=mask,
        hypotheses=hypotheses,
    )
    impacts = []
    for page in pages:
        before_text, sources = project_text_sources(page, before_key, mask)
        after_text, after_sources = project_text_sources(page, after_key, mask)
        if sources != after_sources:
            continue
        for pos, symbol in enumerate(sources):
            if symbol not in edits:
                continue
            before_snippet = local_snippet(before_text, pos)
            after_snippet = local_snippet(after_text, pos)
            before_quality = local_window_quality(before_snippet)
            after_quality = local_window_quality(after_snippet)
            delta = after_quality - before_quality
            before_word = best_covering_word_evidence(before_text, pos, word_index)
            after_word = best_covering_word_evidence(after_text, pos, word_index)
            word_delta = float(after_word["score"]) - float(before_word["score"])
            in_target = any(start <= pos < end for start, end in target_ranges.get(page.test_id, []))
            impacts.append({
                "test_id": page.test_id,
                "symbol": symbol,
                "position": pos,
                "target": edits[symbol],
                "in_hypothesis_target": in_target,
                "before": before_snippet,
                "after": after_snippet,
                "before_quality": round(before_quality, 6),
                "after_quality": round(after_quality, 6),
                "quality_delta": round(delta, 6),
                "before_word_evidence": before_word,
                "after_word_evidence": after_word,
                "word_evidence_delta": round(word_delta, 6),
            })
    target_impacts = [row for row in impacts if row["in_hypothesis_target"]]
    collateral_impacts = [row for row in impacts if not row["in_hypothesis_target"]]
    target_gain_sum = sum(float(row["quality_delta"]) for row in target_impacts)
    collateral_gain_sum = sum(max(0.0, float(row["quality_delta"])) for row in collateral_impacts)
    collateral_damage_sum = sum(max(0.0, -float(row["quality_delta"])) for row in collateral_impacts)
    target_word_evidence_gain_sum = sum(
        max(0.0, float(row["word_evidence_delta"])) for row in target_impacts
    )
    collateral_word_gain_sum = sum(
        max(0.0, float(row["word_evidence_delta"])) for row in collateral_impacts
    )
    collateral_word_damage_sum = sum(
        max(0.0, -float(row["word_evidence_delta"])) for row in collateral_impacts
    )
    collateral_word_gain_weighted_sum = sum(
        weighted_word_evidence_gain(row) for row in collateral_impacts
    )
    collateral_word_damage_weighted_sum = sum(
        weighted_word_evidence_damage(row) for row in collateral_impacts
    )
    target_word_gain_sum = sum(float(row["target_word_gain"]) for row in target_repairs if row["success"])
    symbol_leverage = symbol_leverage_summary(
        edits=edits,
        impacts=impacts,
        edit_support=edit_support,
    )
    global_leverage_score = sum(float(row["global_leverage_score"]) for row in symbol_leverage)
    word_improved_weighted = sum(
        1
        for row in impacts
        if weighted_word_evidence_gain(row) > 0.35
    )
    word_damaged_weighted = sum(
        1
        for row in impacts
        if weighted_word_evidence_damage(row) > 0.35
    )
    impacts.sort(
        key=lambda row: (
            not row["in_hypothesis_target"],
            -abs(float(row["word_evidence_delta"])),
            -abs(float(row["quality_delta"])),
            row["test_id"],
            row["position"],
        )
    )
    return {
        "edited_symbol_count": len(edits),
        "occurrence_count": len(impacts),
        "target_occurrences": len(target_impacts),
        "collateral_occurrences": len(collateral_impacts),
        "improved_occurrences": sum(1 for row in impacts if float(row["quality_delta"]) > 0.015),
        "damaged_occurrences": sum(1 for row in impacts if float(row["quality_delta"]) < -0.015),
        "target_gain_sum": round(target_gain_sum, 6),
        "target_word_gain_sum": round(target_word_gain_sum, 6),
        "target_word_evidence_gain_sum": round(target_word_evidence_gain_sum, 6),
        "collateral_gain_sum": round(collateral_gain_sum, 6),
        "collateral_damage_sum": round(collateral_damage_sum, 6),
        "collateral_word_gain_sum": round(collateral_word_gain_sum, 6),
        "collateral_word_damage_sum": round(collateral_word_damage_sum, 6),
        "collateral_word_gain_weighted_sum": round(collateral_word_gain_weighted_sum, 6),
        "collateral_word_damage_weighted_sum": round(collateral_word_damage_weighted_sum, 6),
        "target_gain_avg": round(target_gain_sum / max(1, len(target_impacts)), 6),
        "collateral_gain_avg": round(collateral_gain_sum / max(1, len(collateral_impacts)), 6),
        "collateral_damage_avg": round(collateral_damage_sum / max(1, len(collateral_impacts)), 6),
        "target_word_evidence_gain_avg": round(
            target_word_evidence_gain_sum / max(1, len(target_impacts)), 6
        ),
        "collateral_word_gain_avg": round(
            collateral_word_gain_sum / max(1, len(collateral_impacts)), 6
        ),
        "collateral_word_damage_avg": round(
            collateral_word_damage_sum / max(1, len(collateral_impacts)), 6
        ),
        "collateral_word_gain_weighted_avg": round(
            collateral_word_gain_weighted_sum / max(1, len(collateral_impacts)), 6
        ),
        "collateral_word_damage_weighted_avg": round(
            collateral_word_damage_weighted_sum / max(1, len(collateral_impacts)), 6
        ),
        "word_improved_occurrences": sum(
            1 for row in impacts if float(row["word_evidence_delta"]) > 0.35
        ),
        "word_damaged_occurrences": sum(
            1 for row in impacts if float(row["word_evidence_delta"]) < -0.35
        ),
        "word_improved_weighted_occurrences": word_improved_weighted,
        "word_damaged_weighted_occurrences": word_damaged_weighted,
        "global_leverage_score": round(global_leverage_score, 6),
        "symbol_leverage": symbol_leverage,
        "target_only_penalty": round(target_only_penalty(
            edited_symbol_count=len(edits),
            target_word_gain_sum=target_word_gain_sum,
            collateral_occurrences=len(collateral_impacts),
        ), 6),
        "target_repairs": target_repairs,
        "occurrence_impacts": impacts[:24],
    }


def adjudication_score(adjudication: dict[str, Any], hypothesis_score: float) -> float:
    target_word_gain = float(adjudication.get("target_word_gain_sum") or 0.0)
    target_word_evidence = float(adjudication.get("target_word_evidence_gain_avg") or 0.0)
    target_gain = float(adjudication.get("target_gain_avg") or 0.0)
    collateral_gain = float(adjudication.get("collateral_gain_avg") or 0.0)
    collateral_damage = float(adjudication.get("collateral_damage_avg") or 0.0)
    collateral_word_gain = float(adjudication.get("collateral_word_gain_weighted_avg") or 0.0)
    collateral_word_damage = float(adjudication.get("collateral_word_damage_weighted_avg") or 0.0)
    damaged = float(adjudication.get("damaged_occurrences") or 0.0)
    word_damaged = float(adjudication.get("word_damaged_weighted_occurrences") or 0.0)
    target_count = float(adjudication.get("target_occurrences") or 0.0)
    occurrence_count = float(adjudication.get("occurrence_count") or 0.0)
    target_only = float(adjudication.get("target_only_penalty") or 0.0)
    global_leverage = float(adjudication.get("global_leverage_score") or 0.0)
    support = min(1.0, target_count / max(1.0, occurrence_count))
    return (
        0.25 * float(hypothesis_score)
        + 0.45 * target_word_gain
        + 1.4 * target_word_evidence
        + 5.0 * target_gain
        + 2.0 * collateral_gain
        + 0.9 * collateral_word_gain
        + 0.42 * global_leverage
        + 0.35 * support
        - 6.0 * collateral_damage
        - 1.5 * collateral_word_damage
        - 0.12 * damaged
        - 0.18 * word_damaged
        - target_only
    )


def adjudication_no_target_score(adjudication: dict[str, Any]) -> float:
    """Score repair collateral only, excluding the target word span.

    This deliberately ignores target-word gain. It is not meant as a final
    selector; it separates "one pretty repaired word" from edits that also
    improve or preserve the surrounding shared-key text.
    """
    collateral_gain = float(adjudication.get("collateral_gain_avg") or 0.0)
    collateral_damage = float(adjudication.get("collateral_damage_avg") or 0.0)
    collateral_word_gain = float(adjudication.get("collateral_word_gain_weighted_avg") or 0.0)
    collateral_word_damage = float(adjudication.get("collateral_word_damage_weighted_avg") or 0.0)
    word_damaged = float(adjudication.get("word_damaged_weighted_occurrences") or 0.0)
    word_improved = float(adjudication.get("word_improved_weighted_occurrences") or 0.0)
    collateral_count = float(adjudication.get("collateral_occurrences") or 0.0)
    support = min(1.0, collateral_count / 12.0)
    global_leverage = float(adjudication.get("global_leverage_score") or 0.0)
    return (
        1.25 * collateral_word_gain
        + 3.0 * collateral_gain
        + 0.35 * global_leverage
        + 0.12 * word_improved
        + 0.25 * support
        - 2.25 * collateral_word_damage
        - 5.5 * collateral_damage
        - 0.16 * word_damaged
    )


def symbol_leverage_summary(
    *,
    edits: dict[str, str],
    impacts: list[dict[str, Any]],
    edit_support: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for symbol, target in sorted(edits.items()):
        symbol_impacts = [row for row in impacts if row.get("symbol") == symbol]
        collateral = [row for row in symbol_impacts if not row.get("in_hypothesis_target")]
        target_rows = [row for row in symbol_impacts if row.get("in_hypothesis_target")]
        support = edit_support.get((symbol, target), {})
        collateral_pages = {str(row.get("test_id")) for row in collateral}
        gain = sum(weighted_word_evidence_gain(row) for row in collateral)
        damage = sum(weighted_word_evidence_damage(row) for row in collateral)
        window_gain = sum(max(0.0, float(row.get("quality_delta") or 0.0)) for row in collateral)
        window_damage = sum(max(0.0, -float(row.get("quality_delta") or 0.0)) for row in collateral)
        support_score = (
            0.45 * min(3.0, float(support.get("hypothesis_count") or 0.0))
            + 0.025 * min(40.0, float(support.get("local_score_sum") or 0.0))
            + 0.35 * max(0.0, float(support.get("page_count") or 0.0) - 1.0)
            + 0.15 * max(0.0, float(support.get("window_count") or 0.0) - 1.0)
            - 0.22 * float(support.get("competing_target_count") or 0.0)
        )
        breadth_score = (
            0.10 * min(30.0, float(len(collateral)))
            + 0.45 * max(0.0, float(len(collateral_pages)) - 1.0)
            + 0.10 * min(10.0, float(len(target_rows)))
        )
        net_context = 0.85 * gain + 4.5 * window_gain - 0.65 * damage - 3.5 * window_damage
        target_only_discount = 1.8 if not collateral and target_rows else 0.0
        score = support_score + breadth_score + net_context - target_only_discount
        rows.append({
            "symbol": symbol,
            "target": target,
            "global_leverage_score": round(score, 6),
            "support_score": round(support_score, 6),
            "breadth_score": round(breadth_score, 6),
            "net_context_score": round(net_context, 6),
            "collateral_occurrences": len(collateral),
            "collateral_page_count": len(collateral_pages),
            "target_occurrences": len(target_rows),
            "weighted_word_gain": round(gain, 6),
            "weighted_word_damage": round(damage, 6),
            "window_gain": round(window_gain, 6),
            "window_damage": round(window_damage, 6),
            "support": support,
        })
    return rows


def weighted_word_evidence_gain(impact: dict[str, Any]) -> float:
    delta = float(impact.get("word_evidence_delta") or 0.0)
    if delta <= 0.0:
        return 0.0
    after = impact.get("after_word_evidence") if isinstance(impact.get("after_word_evidence"), dict) else {}
    return delta * word_evidence_reliability(after)


def weighted_word_evidence_damage(impact: dict[str, Any]) -> float:
    delta = float(impact.get("word_evidence_delta") or 0.0)
    if delta >= 0.0:
        return 0.0
    before = impact.get("before_word_evidence") if isinstance(impact.get("before_word_evidence"), dict) else {}
    return -delta * word_evidence_reliability(before)


def word_evidence_reliability(evidence: dict[str, Any]) -> float:
    """Downweight fragile no-boundary word-island evidence.

    Short exact words and long fuzzy words are useful hints, but they are too
    easy to hallucinate in damaged no-boundary Copiale candidates. This keeps
    them from overpowering shared-key/global evidence.
    """
    word = str(evidence.get("word") or evidence.get("observed") or "")
    if not word:
        return 0.0
    length = len(word)
    distance = evidence.get("distance")
    distance = int(distance) if isinstance(distance, int) else 99
    if distance == 0:
        if length <= 3:
            return 0.22
        if length == 4:
            return 0.55
        return 1.0
    if distance == 1:
        if length >= 8:
            return 0.55
        if length >= 6:
            return 0.35
        return 0.18
    if distance == 2:
        if length >= 9:
            return 0.28
        if length >= 7:
            return 0.18
    return 0.08


def target_only_penalty(
    *,
    edited_symbol_count: int,
    target_word_gain_sum: float,
    collateral_occurrences: int,
) -> float:
    if collateral_occurrences > 0:
        return 0.0
    # A repair that only affects the hand-picked target word is a weak shared
    # key claim. Keep it visible, but do not let pretty local words dominate.
    return min(8.0, 0.45 * target_word_gain_sum + 1.0 * max(1, edited_symbol_count))


def target_word_repairs(
    *,
    pages: list[PageBundle],
    before_key: dict[int, int],
    after_key: dict[int, int],
    mask: tuple[str, ...],
    hypotheses: tuple[WordHypothesis, ...],
) -> list[dict[str, Any]]:
    before_after_by_page: dict[str, tuple[str, str]] = {}
    for page in pages:
        before_text, _sources = project_text_sources(page, before_key, mask)
        after_text, _after_sources = project_text_sources(page, after_key, mask)
        before_after_by_page[page.test_id] = (before_text, after_text)
    rows = []
    for hypothesis in hypotheses:
        before_text, after_text = before_after_by_page.get(hypothesis.test_id, ("", ""))
        before = before_text[hypothesis.start:hypothesis.end]
        after = after_text[hypothesis.start:hypothesis.end]
        success = after == hypothesis.target
        rows.append({
            "test_id": hypothesis.test_id,
            "span": [hypothesis.start, hypothesis.end],
            "observed": hypothesis.observed,
            "before": before,
            "after": after,
            "target": hypothesis.target,
            "success": success,
            "distance": hypothesis.distance,
            "dictionary_rank": hypothesis.dictionary_rank,
            "target_word_gain": round(hypothesis.local_score if success else 0.0, 6),
        })
    return rows


def project_text_sources(
    page: PageBundle,
    key: dict[int, int],
    mask: tuple[str, ...],
) -> tuple[str, list[str]]:
    masked = set(mask)
    chars: list[str] = []
    sources: list[str] = []
    for symbol, token_id in zip(page.symbols, page.token_ids):
        if symbol in masked:
            continue
        value = key.get(token_id)
        if value is None or value < 0 or value > 25:
            continue
        chars.append(chr(ord("A") + value))
        sources.append(symbol)
    return "".join(chars), sources


def local_snippet(text: str, pos: int, radius: int = 14) -> str:
    start = max(0, pos - radius)
    end = min(len(text), pos + radius + 1)
    return text[start:end]


def local_window_quality(text: str) -> float:
    if not text:
        return 0.0
    features = language_quality_feature_dict(text, language="de")
    return max(0.0, min(1.0, 1.0 - window_damage_score(features)))


def best_covering_word_evidence(
    text: str,
    pos: int,
    word_index: WordEvidenceIndex,
) -> dict[str, Any]:
    """Return the strongest dictionary-like word island covering ``pos``.

    Copiale candidates have no trustworthy spaces, so this is intentionally a
    local island signal rather than a segmentation claim. It asks whether a
    touched symbol participates in an exact or near dictionary word after an
    edit, and whether the edit destroys such an island elsewhere.
    """
    if not text or pos < 0 or pos >= len(text):
        return empty_word_evidence()
    best = empty_word_evidence()
    for length in word_index.dictionary:
        if length < 3:
            continue
        start_min = max(0, pos - length + 1)
        start_max = min(pos, len(text) - length)
        for start in range(start_min, start_max + 1):
            observed = text[start:start + length]
            if not observed.isalpha():
                continue
            max_distance = max_word_evidence_distance(length)
            for target, rank in near_dictionary_words(observed, word_index, max_distance):
                distance = hamming_distance(observed, target)
                if distance > max_distance:
                    continue
                score = word_evidence_score(length=length, distance=distance, rank=rank)
                if score > float(best["score"]):
                    best = {
                        "score": round(score, 6),
                        "span": [start, start + length],
                        "observed": observed,
                        "word": target,
                        "distance": distance,
                        "dictionary_rank": rank,
                    }
    return best


def build_word_evidence_index(dictionary: dict[int, list[tuple[str, int]]]) -> WordEvidenceIndex:
    patterns: dict[int, dict[str, list[tuple[str, int]]]] = {}
    for length, words in dictionary.items():
        length_patterns: dict[str, list[tuple[str, int]]] = {}
        max_distance = max_word_evidence_distance(length)
        for word, rank in words:
            for pattern in wildcard_patterns(word, max_distance):
                length_patterns.setdefault(pattern, []).append((word, rank))
        patterns[length] = length_patterns
    return WordEvidenceIndex(dictionary=dictionary, patterns=patterns)


def near_dictionary_words(
    observed: str,
    word_index: WordEvidenceIndex,
    max_distance: int,
) -> list[tuple[str, int]]:
    by_word: dict[str, int] = {}
    for pattern in wildcard_patterns(observed, max_distance):
        for word, rank in word_index.patterns.get(len(observed), {}).get(pattern, []):
            old = by_word.get(word)
            if old is None or rank < old:
                by_word[word] = rank
    return [(word, rank) for word, rank in by_word.items()]


def wildcard_patterns(word: str, max_wildcards: int) -> list[str]:
    positions = range(len(word))
    patterns = [word]
    for count in range(1, max(0, max_wildcards) + 1):
        for combo in itertools.combinations(positions, count):
            chars = list(word)
            for pos in combo:
                chars[pos] = "?"
            patterns.append("".join(chars))
    return patterns


def empty_word_evidence() -> dict[str, Any]:
    return {
        "score": 0.0,
        "span": [],
        "observed": "",
        "word": "",
        "distance": None,
        "dictionary_rank": None,
    }


def max_word_evidence_distance(length: int) -> int:
    if length <= 4:
        return 0
    if length <= 7:
        return 1
    return 2


def word_evidence_score(*, length: int, distance: int, rank: int) -> float:
    if distance > max_word_evidence_distance(length):
        return 0.0
    rank_bonus = 1.0 / max(1.0, rank ** 0.35)
    # Exact short function words should count, but near long content words
    # should be allowed to compete because these candidates are damaged basins.
    raw = length - 2.25 * distance
    return max(0.0, raw / max(1.0, length ** 0.5) + 0.65 * rank_bonus)


def dedupe_hypotheses_by_edits(hypotheses: list[WordHypothesis]) -> list[WordHypothesis]:
    best: dict[tuple[tuple[str, str], ...], WordHypothesis] = {}
    for row in hypotheses:
        old = best.get(row.edits)
        if old is None or row.local_score > old.local_score:
            best[row.edits] = row
    return sorted(
        best.values(),
        key=lambda row: (
            row.local_score,
            len(row.target),
            -row.distance,
            -row.dictionary_rank,
        ),
        reverse=True,
    )


def load_dictionary(path: Path, min_len: int, max_len: int) -> dict[int, list[tuple[str, int]]]:
    words_by_len: dict[int, list[tuple[str, int]]] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        word = normalize_word(line)
        if min_len <= len(word) <= max_len and word.isalpha():
            words_by_len.setdefault(len(word), []).append((word, index))
    return words_by_len


def normalize_word(value: str) -> str:
    value = value.strip().upper()
    return (
        value.replace("Ä", "AE")
        .replace("Ö", "OE")
        .replace("Ü", "UE")
        .replace("ß", "SS")
    )


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(1 for a, b in zip(left, right) if a != b)


def pair_signature(hypotheses: tuple[WordHypothesis, ...]) -> str:
    return ";".join(
        sorted(f"{row.test_id}:{row.start}-{row.end}:{row.target}" for row in hypotheses)
    )


def hypothesis_to_dict(row: WordHypothesis) -> dict[str, Any]:
    return {
        "test_id": row.test_id,
        "window_start": row.window_start,
        "start": row.start,
        "end": row.end,
        "observed": row.observed,
        "target": row.target,
        "edits": [f"{symbol}->{target}" for symbol, target in row.edits],
        "distance": row.distance,
        "dictionary_rank": row.dictionary_rank,
        "local_score": round(row.local_score, 6),
    }


def word_variant_summary(row: dict[str, Any]) -> dict[str, Any]:
    summary = variant_summary(row)
    summary["word_hypotheses"] = row.get("word_hypotheses") or []
    summary["word_hypothesis_score"] = row.get("word_hypothesis_score")
    summary["repair_adjudication"] = row.get("repair_adjudication") or {}
    summary["marginal_contribution"] = row.get("marginal_contribution") or {}
    summary["second_stage"] = row.get("second_stage") or {}
    return summary


def compact_variant_summary(row: dict[str, Any]) -> dict[str, Any]:
    adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    marginal = row.get("marginal_contribution") if isinstance(row.get("marginal_contribution"), dict) else {}
    second_stage = row.get("second_stage") if isinstance(row.get("second_stage"), dict) else {}
    return {
        "edits": row.get("edits") or [],
        "word_hypotheses": row.get("word_hypotheses") or [],
        "word_hypothesis_score": row.get("word_hypothesis_score"),
        "adjudication_score": adjudication.get("adjudication_score"),
        "adjudication_no_target_score": adjudication.get("adjudication_no_target_score"),
        "target_leverage_score": adjudication.get("target_leverage_score"),
        "global_leverage_score": adjudication.get("global_leverage_score"),
        "marginal_selector_score": marginal.get("marginal_selector_score"),
        "marginal_contribution": {
            "best_subset": marginal.get("best_subset") or {},
            "delta_vs_best_subset": marginal.get("delta_vs_best_subset") or {},
            "negative_member_count": marginal.get("negative_member_count"),
            "marginal_selector_score": marginal.get("marginal_selector_score"),
        },
        "second_stage": second_stage,
        "page_robust_score": row.get("page_robust_score"),
        "page_validation_avg": row.get("page_validation_avg"),
        "page_language_quality_avg": row.get("page_language_quality_avg"),
        "post_hoc_char_avg": row.get("post_hoc_char_avg"),
        "post_hoc_char_no_target_baseline_avg": row.get("post_hoc_char_no_target_baseline_avg"),
        "post_hoc_char_target_baseline_avg": row.get("post_hoc_char_target_baseline_avg"),
        "post_hoc_char_no_target_avg": row.get("post_hoc_char_no_target_avg"),
        "post_hoc_char_target_avg": row.get("post_hoc_char_target_avg"),
        "preview": str(row.get("preview") or "")[:220],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Copiale Word-Hypothesis Repair Probe",
        "",
        "Ground truth is not used to generate or rank hypotheses. Post-hoc character scores are calibration only.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| label | `{payload['label']}` |",
        f"| pages | {len(payload['test_ids'])} |",
        f"| word hypotheses | {payload['hypothesis_count']} |",
        f"| variants | {payload['variant_count']} |",
        f"| runtime-accepted variants | {payload['accepted_variant_count']} |",
        "",
        "## Top Word Hypotheses",
        "",
        "| Rank | Page | Span | Observed | Target | Edits | Distance | Dict Rank | Local Score |",
        "|---:|---|---:|---|---|---|---:|---:|---:|",
    ]
    for index, row in enumerate(payload["top_word_hypotheses"], start=1):
        lines.append(
            f"| {index} | `{row['test_id']}` | {row['start']}-{row['end']} | "
            f"`{row['observed']}` | `{row['target']}` | {escape_cell('<br>'.join(row['edits']))} | "
            f"{row['distance']} | {row['dictionary_rank']} | {row['local_score']:.3f} |"
        )
    lines.extend([
        "",
        "## Second-Stage Diverse Review Shortlist",
        "",
        "This is the primary second-stage output: a compact, runtime-only, edit-family-diverse review set. Final scalar rank is secondary.",
        "",
        variant_table(payload.get("top_variants_by_second_stage_review") or []),
        "",
        "## Top Variants By Runtime Score",
        "",
        variant_table(payload["top_variants"]),
        "",
        "## Top Variants By Second-Stage Portfolio",
        "",
        variant_table(payload["top_variants_by_second_stage"]),
        "",
        "## Top Variants By Word-Hypothesis Score",
        "",
        variant_table(payload["top_variants_by_word_hypothesis"]),
        "",
        "## Top Variants By Repair Adjudication",
        "",
        variant_table(payload["top_variants_by_adjudication"]),
        "",
        "## Top Combination Variants By Repair Adjudication",
        "",
        variant_table(payload["top_combination_variants_by_adjudication"]),
        "",
        "## Top Combination Variants By Marginal Selector",
        "",
        variant_table(payload["top_combination_variants_by_marginal"]),
        "",
        "## Top Variants By Post-Hoc Character Score",
        "",
        variant_table(payload["top_variants_by_post_hoc"]),
        "",
        "## Damaged Windows",
        "",
        "| Page | Span | Damage | Text |",
        "|---|---:|---:|---|",
    ])
    for page in payload["page_windows"]:
        for window in page["windows"]:
            lines.append(
                f"| `{page['test_id']}` | {window['start']}-{window['end']} | "
                f"{float(window['damage_score']):.3f} | `{escape_cell(window['text'][:100])}` |"
            )
    return "\n".join(lines).rstrip() + "\n"


def variant_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Rank | Edits | Hypothesis | Stage2 | Flags | Hyp Score | Adj | AdjNoTarget | Leverage | Marg | Impact | Runtime Decision | Robust | Val | LQ | Char | Preview |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(rows, start=1):
        hypotheses = row.get("word_hypotheses") if isinstance(row.get("word_hypotheses"), list) else []
        hypothesis_text = ""
        if hypotheses:
            hypothesis_text = "<br>".join(
                f"`{item.get('observed')}` -> `{item.get('target')}`"
                for item in hypotheses[:3]
            )
        acceptance = row.get("repair_acceptance") if isinstance(row.get("repair_acceptance"), dict) else {}
        adjudication = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
        marginal = row.get("marginal_contribution") if isinstance(row.get("marginal_contribution"), dict) else {}
        second_stage = row.get("second_stage") if isinstance(row.get("second_stage"), dict) else {}
        lines.append(
            f"| {index} | {escape_cell('<br>'.join(row.get('edits') or []))} | "
            f"{hypothesis_text} | {fmt_float(second_stage.get('portfolio_score'))} | "
            f"{escape_cell(', '.join(second_stage.get('flags') or []))} | "
            f"{fmt_float(row.get('word_hypothesis_score'))} | "
            f"{fmt_float(adjudication.get('adjudication_score'))} | "
            f"{fmt_float(adjudication.get('adjudication_no_target_score'))} | "
            f"{fmt_float(adjudication.get('target_leverage_score'))} | "
            f"{fmt_float(marginal.get('marginal_selector_score'))} | "
            f"{impact_summary(adjudication)} | "
            f"{acceptance.get('decision') or ''} | "
            f"{fmt_float(row.get('page_robust_score'))} | {fmt_float(row.get('page_validation_avg'))} | "
            f"{fmt_float(row.get('page_language_quality_avg'))} | {fmt_pct(row.get('post_hoc_char_avg'))} | "
            f"{escape_cell(str(row.get('preview') or '')[:160])} |"
        )
    return "\n".join(lines)


def impact_summary(adjudication: dict[str, Any]) -> str:
    if not adjudication:
        return ""
    return (
        f"word {fmt_float(adjudication.get('target_word_gain_sum'))}; "
        f"island +{fmt_float(adjudication.get('collateral_word_gain_sum'))}/"
        f"-{fmt_float(adjudication.get('collateral_word_damage_sum'))}; "
        f"window {fmt_float(adjudication.get('target_gain_sum'))}; "
        f"collateral +{fmt_float(adjudication.get('collateral_gain_sum'))}/"
        f"-{fmt_float(adjudication.get('collateral_damage_sum'))}; "
        f"damaged {int(adjudication.get('damaged_occurrences') or 0)}"
        f"/{int(adjudication.get('word_damaged_occurrences') or 0)}"
    )


def fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def escape_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
