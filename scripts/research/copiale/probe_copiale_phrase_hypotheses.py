#!/usr/bin/env python3
"""Probe phrase-first Copiale repair hypotheses.

This is an offline diagnostic harness. It generates same-length plaintext
phrase hypotheses for damaged no-boundary Copiale spans, derives the implied
symbol edits, applies those edits globally, and scores the collateral effect.

The runtime generators and rankers are ground-truth-free. Benchmark plaintext
is reported only in post-hoc calibration fields after variants are produced.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from agent.model_provider import canonical_provider, estimate_provider_cost  # noqa: E402
from probe_copiale_multipage_global_repair import (  # noqa: E402
    annotate_acceptance,
    annotate_repair_evidence,
    resolve_path,
)
from probe_copiale_word_hypothesis_repair import (  # noqa: E402
    WordHypothesis,
    attach_second_stage_portfolio,
    build_edit_support,
    build_page_windows,
    build_word_evidence_index,
    evaluate_hypotheses,
    hypothesis_to_dict,
    implied_edits,
    load_dictionary,
    word_variant_summary,
)
from rank_candidate_texts_with_llm import (  # noqa: E402
    call_llm,
    parse_json_response,
    visible_response_text,
)
from run_copiale_iterative_repair_tree import load_context  # noqa: E402


AZ_RE = re.compile(r"[^A-Z]")


@dataclass(frozen=True)
class PhraseCandidate:
    target: str
    words: tuple[str, ...]
    distance: int
    rank_score: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and verify phrase-first Copiale repair hypotheses."
    )
    parser.add_argument("experiment_json", help="JSON from run_copiale_multipage_experiment.py")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument(
        "--section",
        choices=["portfolio_local_repair", "portfolio_refinement", "elite_page_rerank"],
        default="portfolio_local_repair",
    )
    parser.add_argument("--label", default="", help="Finalist label to repair, e.g. top6.")
    parser.add_argument("--dictionary", default="resources/dictionaries/german_common.txt")
    parser.add_argument("--mode", choices=["local", "llm"], default="local")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument(
        "--llm-replay-json",
        default="",
        help="Reuse parsed LLM proposals from a prior phrase-hypothesis JSON instead of calling the model.",
    )
    parser.add_argument(
        "--llm-passes",
        type=int,
        default=1,
        help="Number of independent symbol-aware LLM phrase-generation passes to aggregate.",
    )
    parser.add_argument(
        "--llm-salvage-fragments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For rejected LLM phrase proposals, try mechanically valid contiguous subphrase fragments.",
    )
    parser.add_argument("--llm-salvage-min-len", type=int, default=12)
    parser.add_argument("--llm-salvage-max-len", type=int, default=36)
    parser.add_argument("--llm-salvage-top-n", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--consensus-top-n", type=int, default=12)
    parser.add_argument("--consensus-min-agreement", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--window-step", type=int, default=40)
    parser.add_argument("--windows-per-page", type=int, default=4)
    parser.add_argument("--min-word-len", type=int, default=2)
    parser.add_argument("--max-word-len", type=int, default=14)
    parser.add_argument("--min-phrase-len", type=int, default=10)
    parser.add_argument("--max-phrase-len", type=int, default=28)
    parser.add_argument("--max-phrase-words", type=int, default=4)
    parser.add_argument("--max-phrase-distance", type=int, default=6)
    parser.add_argument("--max-edits", type=int, default=6)
    parser.add_argument(
        "--phrase-edit-model",
        choices=["same_length", "delete_to_null"],
        default="same_length",
        help=(
            "same_length only accepts direct substitutions. delete_to_null also "
            "aligns shorter targets to observed spans and converts skipped observed "
            "symbols into <null> edits."
        ),
    )
    parser.add_argument(
        "--llm-max-span-shift",
        type=int,
        default=80,
        help="For LLM mode, locate the supplied observed text within this distance of the supplied start.",
    )
    parser.add_argument("--span-step", type=int, default=2)
    parser.add_argument("--beam-per-offset", type=int, default=10)
    parser.add_argument("--max-segment-distance", type=int, default=3)
    parser.add_argument("--max-word-candidates-per-segment", type=int, default=24)
    parser.add_argument("--max-hypotheses", type=int, default=120)
    parser.add_argument("--max-hypotheses-per-window", type=int, default=10)
    parser.add_argument("--max-hypothesis-set-size", type=int, default=1)
    parser.add_argument("--combination-candidate-limit", type=int, default=32)
    parser.add_argument("--max-combinations", type=int, default=400)
    parser.add_argument("--max-combined-edits", type=int, default=8)
    parser.add_argument("--allow-stable-edits", action="store_true")
    parser.add_argument("--acceptance-margin", type=float, default=0.03)
    parser.add_argument("--min-page-drop", type=float, default=0.02)
    parser.add_argument("--max-illusion-increase", type=float, default=0.02)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    started = time.monotonic()
    context = load_context(args)
    page_windows = build_page_windows(
        pages=context["pages"],
        alphabet=context["alphabet"],
        key=context["root_key"],
        mask=context["root_mask"],
        consensus=context["consensus"],
        window_size=args.window_size,
        window_step=args.window_step,
        windows_per_page=args.windows_per_page,
    )
    dictionary_path = resolve_path(Path(args.dictionary))
    dictionary = load_dictionary(dictionary_path, args.min_word_len, args.max_word_len)
    collateral_dictionary = load_dictionary(dictionary_path, 3, args.max_word_len)
    llm_packet: dict[str, Any] = {}
    if args.mode == "local":
        hypotheses = generate_local_phrase_hypotheses(
            page_windows=page_windows,
            dictionary=dictionary,
            context=context,
            args=args,
        )
    else:
        hypotheses, llm_packet = generate_llm_phrase_hypotheses(
            page_windows=page_windows,
            context=context,
            args=args,
        )
    hypotheses = hypotheses[: max(0, args.max_hypotheses)]
    variants = evaluate_hypotheses(
        pages=context["pages"],
        baseline_key=context["root_key"],
        baseline_mask=context["root_mask"],
        hypotheses=hypotheses,
        max_hypothesis_set_size=args.max_hypothesis_set_size,
        combination_candidate_limit=args.combination_candidate_limit,
        max_combinations=args.max_combinations,
        max_combined_edits=args.max_combined_edits,
        collateral_index=build_word_evidence_index(collateral_dictionary),
        edit_support=build_edit_support(hypotheses),
        progress=args.progress,
    )
    baseline_variant = next(
        (row for row in variants if row.get("edits") == ["baseline"]),
        variants[0] if variants else {},
    )
    if baseline_variant:
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
    else:
        second_stage = {}
    variants.sort(
        key=lambda row: (
            float((row.get("second_stage") or {}).get("portfolio_score") or 0.0),
            float((row.get("repair_adjudication") or {}).get("adjudication_score") or 0.0),
            float(row.get("page_robust_score") or 0.0),
        ),
        reverse=True,
    )
    payload = {
        "experiment": "copiale_phrase_hypothesis_probe",
        "mode": args.mode,
        "source_experiment": str(resolve_path(Path(args.experiment_json))),
        "label": context["label"],
        "test_ids": context["test_ids"],
        "dictionary": str(dictionary_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "hypothesis_count": len(hypotheses),
        "variant_count": len(variants),
        "settings": serializable_settings(args),
        "page_windows": compact_page_windows(page_windows),
        "llm_packet": llm_packet,
        "top_phrase_hypotheses": [hypothesis_to_dict(row) for row in hypotheses[: args.top_n]],
        "baseline": word_variant_summary(baseline_variant) if baseline_variant else {},
        "selection_summary": selection_summary(variants, baseline_variant),
        "second_stage_portfolio": second_stage,
        "top_variants": [word_variant_summary(row) for row in variants[: args.top_n]],
        "top_variants_by_post_hoc": [
            word_variant_summary(row)
            for row in sorted(
                variants,
                key=lambda row: float(row.get("post_hoc_char_avg") or 0.0),
                reverse=True,
            )[: args.top_n]
        ],
    }
    markdown = render_markdown(payload)
    stem = f"{Path(args.experiment_json).stem}.{context['label']}.{args.mode}_phrase_hypotheses"
    output = (
        resolve_path(Path(args.output))
        if args.output
        else resolve_path(Path("artifacts/language_quality/phrase_hypotheses") / f"{stem}.md")
    )
    json_output = (
        resolve_path(Path(args.json_output))
        if args.json_output
        else output.with_suffix(".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")


def selection_summary(
    variants: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    if not variants:
        return {}
    scorer_pick = variants[0]
    best_post_hoc = max(variants, key=lambda row: float(row.get("post_hoc_char_avg") or 0.0))
    baseline_score = float(baseline.get("post_hoc_char_avg") or 0.0) if baseline else None
    scorer_score = float(scorer_pick.get("post_hoc_char_avg") or 0.0)
    best_score = float(best_post_hoc.get("post_hoc_char_avg") or 0.0)
    scorer_ranks = {id(row): index for index, row in enumerate(variants, start=1)}
    post_hoc_sorted = sorted(variants, key=lambda row: float(row.get("post_hoc_char_avg") or 0.0), reverse=True)
    post_hoc_ranks = {id(row): index for index, row in enumerate(post_hoc_sorted, start=1)}
    return {
        "scorer_metric": "second_stage.portfolio_score, then repair_adjudication.adjudication_score, then page_robust_score",
        "variant_count": len(variants),
        "scorer_pick_rank": 1,
        "scorer_pick_post_hoc_rank": post_hoc_ranks.get(id(scorer_pick)),
        "post_hoc_best_scorer_rank": scorer_ranks.get(id(best_post_hoc)),
        "post_hoc_best_in_scorer_top_3": bool((scorer_ranks.get(id(best_post_hoc)) or 10**9) <= 3),
        "post_hoc_best_in_scorer_top_5": bool((scorer_ranks.get(id(best_post_hoc)) or 10**9) <= 5),
        "post_hoc_best_in_scorer_top_10": bool((scorer_ranks.get(id(best_post_hoc)) or 10**9) <= 10),
        "scorer_pick_post_hoc_char": scorer_score,
        "post_hoc_best_char": best_score,
        "scorer_pick_gap_from_best": max(0.0, best_score - scorer_score),
        "baseline_post_hoc_char": baseline_score,
        "scorer_pick_gain_over_baseline": None if baseline_score is None else scorer_score - baseline_score,
        "post_hoc_best_gain_over_baseline": None if baseline_score is None else best_score - baseline_score,
        "scorer_pick": word_variant_summary(scorer_pick),
        "post_hoc_best": word_variant_summary(best_post_hoc),
    }


def generate_local_phrase_hypotheses(
    *,
    page_windows: list[dict[str, Any]],
    dictionary: dict[int, list[tuple[str, int]]],
    context: dict[str, Any],
    args: argparse.Namespace,
) -> list[WordHypothesis]:
    rows: list[WordHypothesis] = []
    seen: set[tuple[str, int, int, str, tuple[tuple[str, str], ...]]] = set()
    for page in page_windows:
        text = str(page["text"])
        sources = list(page["sources"])
        for window in page["windows"]:
            window_start = int(window.get("start") or 0)
            window_end = int(window.get("end") or window_start)
            local_rows: list[WordHypothesis] = []
            for start in range(window_start, max(window_start, window_end - args.min_phrase_len + 1), max(1, args.span_step)):
                for length in range(args.min_phrase_len, args.max_phrase_len + 1):
                    end = start + length
                    if end > window_end or end > len(text):
                        continue
                    observed = text[start:end]
                    if not observed.isalpha():
                        continue
                    for phrase in local_phrase_candidates(
                        observed,
                        dictionary=dictionary,
                        max_words=args.max_phrase_words,
                        max_distance=args.max_phrase_distance,
                        max_segment_distance=args.max_segment_distance,
                        max_word_candidates_per_segment=args.max_word_candidates_per_segment,
                        beam_per_offset=args.beam_per_offset,
                    ):
                        edits = implied_edits(
                            observed=observed,
                            target=phrase.target,
                            sources=sources[start:end],
                            consensus=context["consensus"],
                            alphabet=context["alphabet"],
                            baseline_key=context["root_key"],
                            baseline_mask=context["root_mask"],
                            allow_stable_edits=args.allow_stable_edits,
                        ) if args.phrase_edit_model == "same_length" else implied_phrase_edits(
                            observed=observed,
                            target=phrase.target,
                            sources=sources[start:end],
                            consensus=context["consensus"],
                            alphabet=context["alphabet"],
                            baseline_key=context["root_key"],
                            baseline_mask=context["root_mask"],
                            allow_stable_edits=args.allow_stable_edits,
                            allow_delete_to_null=True,
                        )
                        if not edits or len(edits) > args.max_edits:
                            continue
                        key = (str(page["test_id"]), start, end, phrase.target, edits)
                        if key in seen:
                            continue
                        seen.add(key)
                        score = (
                            phrase.rank_score
                            + float(window.get("damage_score") or 0.0)
                            - 0.35 * len(edits)
                        )
                        local_rows.append(
                            WordHypothesis(
                                test_id=str(page["test_id"]),
                                window_start=window_start,
                                start=start,
                                end=end,
                                observed=observed,
                                target=phrase.target,
                                edits=edits,
                                distance=phrase.distance,
                                dictionary_rank=0,
                                local_score=score,
                            )
                        )
            local_rows.sort(
                key=lambda row: (
                    row.local_score,
                    len(row.target),
                    -row.distance,
                    -len(row.edits),
                ),
                reverse=True,
            )
            rows.extend(local_rows[: max(0, args.max_hypotheses_per_window)])
    rows.sort(
        key=lambda row: (
            row.local_score,
            len(row.target),
            -row.distance,
            -len(row.edits),
        ),
        reverse=True,
    )
    return dedupe_by_edits(rows)


def local_phrase_candidates(
    observed: str,
    *,
    dictionary: dict[int, list[tuple[str, int]]],
    max_words: int,
    max_distance: int,
    max_segment_distance: int,
    max_word_candidates_per_segment: int,
    beam_per_offset: int,
) -> list[PhraseCandidate]:
    length = len(observed)
    segment_cache: dict[str, list[tuple[str, int, int, float]]] = {}

    def candidates_for(segment: str) -> list[tuple[str, int, int, float]]:
        cached = segment_cache.get(segment)
        if cached is not None:
            return cached
        rows = []
        for word, rank in dictionary.get(len(segment), []):
            distance = hamming_distance(segment, word)
            if distance > max_segment_distance:
                continue
            rank_bonus = 1.0 / max(1.0, rank ** 0.30)
            score = len(word) - 1.35 * distance + 2.0 * rank_bonus
            rows.append((word, rank, distance, score))
        rows.sort(key=lambda row: (row[3], -row[2], -row[1]), reverse=True)
        cached = rows[: max(1, max_word_candidates_per_segment)]
        segment_cache[segment] = cached
        return cached

    states: dict[int, list[PhraseCandidate]] = {
        0: [PhraseCandidate(target="", words=(), distance=0, rank_score=0.0)]
    }
    for offset in range(length):
        current = states.get(offset) or []
        if not current:
            continue
        for state in current:
            if len(state.words) >= max_words:
                continue
            remaining = length - offset
            for word_len, words in dictionary.items():
                if word_len > remaining:
                    continue
                segment = observed[offset : offset + word_len]
                for word, _rank, distance, segment_score in candidates_for(segment):
                    total_distance = state.distance + distance
                    if total_distance > max_distance:
                        continue
                    next_state = PhraseCandidate(
                        target=state.target + word,
                        words=state.words + (word,),
                        distance=total_distance,
                        rank_score=state.rank_score + segment_score,
                    )
                    states.setdefault(offset + word_len, []).append(next_state)
        for key, rows in list(states.items()):
            if len(rows) > beam_per_offset:
                rows.sort(
                    key=lambda row: (
                        row.rank_score,
                        -row.distance,
                        len(row.words),
                    ),
                    reverse=True,
                )
                states[key] = rows[:beam_per_offset]
    finals = [
        row for row in states.get(length, [])
        if len(row.words) >= 2 and 0 < row.distance <= max_distance
    ]
    finals.sort(
        key=lambda row: (
            row.rank_score,
            len(row.target),
            -row.distance,
        ),
        reverse=True,
    )
    return finals[:beam_per_offset]


def generate_llm_phrase_hypotheses(
    *,
    page_windows: list[dict[str, Any]],
    context: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[WordHypothesis], dict[str, Any]]:
    system = (
        "You propose damaged German plaintext span repairs for a cipher-solving harness. "
        "Return only JSON. Do not use tools."
    )
    pass_count = max(1, int(getattr(args, "llm_passes", 1) or 1))
    packet: dict[str, Any] = {
        "provider": canonical_provider(args.provider),
        "model": args.model,
        "passes_requested": pass_count,
        "dry_run": bool(args.dry_run),
    }
    if getattr(args, "llm_replay_json", ""):
        replay_path = resolve_path(Path(args.llm_replay_json))
        replay_payload = json.loads(replay_path.read_text(encoding="utf-8"))
        replay_llm = replay_payload.get("llm_packet") if isinstance(replay_payload, dict) else {}
        parsed = replay_llm.get("parsed_response") if isinstance(replay_llm, dict) else None
        if not isinstance(parsed, dict):
            raise SystemExit(f"No parsed LLM response found in replay JSON: {replay_path}")
        hypotheses, rejection_summary = llm_response_to_hypotheses(
            parsed=parsed,
            page_windows=page_windows,
            context=context,
            args=args,
        )
        packet.update({
            "replay_json": str(replay_path),
            "parsed_response": parsed,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
            "rejection_summary": rejection_summary,
            "accepted_after_dedupe": len(hypotheses),
        })
        return hypotheses, packet
    if args.dry_run:
        return [], packet
    all_rows: list[WordHypothesis] = []
    all_raw: list[dict[str, Any]] = []
    pass_packets: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    total_cache = 0
    total_cost = 0.0
    aggregate_rejections = {"counts": {}, "examples": [], "accepted_before_dedupe": 0}
    max_windows = max(1, args.windows_per_page * len(page_windows))
    for pass_index in range(1, pass_count + 1):
        prompt = build_llm_prompt(
            page_windows,
            max_windows=max_windows,
            edit_model=args.phrase_edit_model,
            pass_index=pass_index,
            pass_count=pass_count,
        )
        print(
            f"Calling LLM phrase generator pass {pass_index}/{pass_count} "
            f"({canonical_provider(args.provider)}/{args.model})...",
            flush=True,
        )
        response, usage = call_llm(
            provider=args.provider,
            model=args.model,
            system=system,
            prompt=prompt,
            max_tokens=args.max_tokens,
        )
        response_text = visible_response_text(response)
        parsed = parse_json_response(response_text)
        rows, rejection_summary = llm_response_to_hypotheses(
            parsed=parsed,
            page_windows=page_windows,
            context=context,
            args=args,
        )
        raw = []
        if isinstance(parsed, dict) and isinstance(parsed.get("hypotheses"), list):
            raw = [row for row in parsed.get("hypotheses") if isinstance(row, dict)]
            all_raw.extend(raw)
        all_rows.extend(rows)
        total_input += usage.input_tokens
        total_output += usage.output_tokens
        total_cache += usage.cache_read_input_tokens
        cost = estimate_provider_cost(
            canonical_provider(args.provider),
            args.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
        )
        total_cost += cost
        merge_rejection_summary(aggregate_rejections, rejection_summary)
        pass_packets.append({
            "pass_index": pass_index,
            "prompt": prompt,
            "response_text": response_text,
            "parsed_response": parsed,
            "raw_hypotheses": len(raw),
            "accepted_before_dedupe": len(rows),
            "rejection_summary": rejection_summary,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_input_tokens,
                "estimated_cost_usd": cost,
            },
        })
    hypotheses = dedupe_by_edits(all_rows)
    hypotheses.sort(key=lambda row: (row.local_score, len(row.target), -row.distance), reverse=True)
    packet.update({
        "passes": pass_packets,
        "prompt": pass_packets[0]["prompt"] if pass_packets else "",
        "response_text": "\n\n".join(str(row.get("response_text") or "") for row in pass_packets),
        "parsed_response": {"hypotheses": all_raw},
        "usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_read_tokens": total_cache,
            "estimated_cost_usd": total_cost,
        },
        "rejection_summary": aggregate_rejections,
        "accepted_after_dedupe": len(hypotheses),
    })
    return hypotheses, packet


def merge_rejection_summary(target: dict[str, Any], source: dict[str, Any]) -> None:
    counts = target.setdefault("counts", {})
    for reason, count in (source.get("counts") or {}).items():
        counts[reason] = int(counts.get(reason, 0)) + int(count)
    target["accepted_before_dedupe"] = int(target.get("accepted_before_dedupe") or 0) + int(
        source.get("accepted_before_dedupe") or 0
    )
    examples = target.setdefault("examples", [])
    for row in source.get("examples") or []:
        if len(examples) >= 20:
            break
        examples.append(row)


def build_llm_prompt(
    page_windows: list[dict[str, Any]],
    *,
    max_windows: int,
    edit_model: str,
    pass_index: int = 1,
    pass_count: int = 1,
) -> str:
    windows = []
    for page in page_windows:
        page_sources = list(page.get("sources") or [])
        for window in page.get("windows") or []:
            start = int(window.get("start") or 0)
            end = int(window.get("end") or start)
            text = str(window.get("text") or "")
            sources = page_sources[start:end]
            windows.append({
                "test_id": page.get("test_id"),
                "start": start,
                "end": end,
                "text": text,
                "symbol_sequence": " ".join(sources),
                "repeated_symbols": repeated_symbol_prompt_rows(text, sources, absolute_start=start),
            })
    if windows and pass_count > 1:
        offset = ((pass_index - 1) * max_windows) // pass_count
        windows = windows[offset:] + windows[:offset]
    windows = windows[:max_windows]
    if edit_model == "delete_to_null":
        length_rule = (
            "Your target may be the same length as the observed span or shorter. "
            "It must NEVER be longer than the observed span, because this verifier "
            "can delete/null extra observed letters but cannot insert missing letters. "
            "If the natural German repair needs inserted letters, choose a different "
            "span or skip that hypothesis."
        )
    else:
        length_rule = (
            "Your target must be exactly the same length as the observed span. "
            "If the natural German repair needs inserted/deleted letters, choose a "
            "different span or skip that hypothesis."
        )
    return (
        "You will see damaged no-boundary uppercase German plaintext windows from a cipher solve.\n"
        f"This is phrase-generation pass {pass_index} of {pass_count}. "
        "Prefer repairs different from the most obvious first-span guesses; look for short, "
        "mechanically consistent phrase repairs across the supplied windows.\n"
        "Propose phrase-level repairs. Each proposal must identify an exact span from one window and "
        "an A-Z target string for that span. Remove spaces, punctuation, and editorial symbols from "
        "your target; use only A-Z. Normalize umlauts as one character, not two: use A/O/U, never "
        "AE/OE/UE. "
        f"{length_rule} "
        "Each window includes cipher symbols below the text. This is a homophonic-substitution "
        "key repair task: if the same cipher symbol appears more than once in your proposed span, "
        "your repair must assign that symbol the same final plaintext letter at every retained "
        "position. You may make extra observed positions null/deleted when shorter targets are "
        "allowed, but do not propose a phrase that would require one symbol to become two different "
        "letters. The symbol_sequence is space-separated and aligned to the text characters: "
        "symbol_sequence[0] is the symbol for text[start], symbol_sequence[1] for text[start+1], "
        "and so on. Use the repeated_symbols hints to avoid inconsistent spans. "
        "Prefer phrase hypotheses that would repair multiple adjacent "
        "damaged words, not isolated obvious words.\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "hypotheses": [\n'
        '    {"test_id": "...", "start": 123, "end": 145, "observed": "EXACTSPAN", '
        '"target": "SAMELENGTHAZ", "reason": "brief"}\n'
        "  ]\n"
        "}\n\n"
        "Windows:\n"
        f"{json.dumps(windows, ensure_ascii=False, indent=2)}"
    )

def repeated_symbol_prompt_rows(
    text: str,
    sources: list[str],
    *,
    absolute_start: int,
    max_rows: int = 16,
) -> list[dict[str, Any]]:
    positions: dict[str, list[int]] = {}
    chars: dict[str, list[str]] = {}
    for offset, (char, symbol) in enumerate(zip(text, sources)):
        positions.setdefault(symbol, []).append(absolute_start + offset)
        chars.setdefault(symbol, []).append(char)
    rows = [
        {
            "symbol": symbol,
            "positions": symbol_positions,
            "current_chars": "".join(chars.get(symbol, [])),
        }
        for symbol, symbol_positions in positions.items()
        if len(symbol_positions) > 1
    ]
    rows.sort(key=lambda row: (-len(row["positions"]), row["symbol"]))
    return rows[:max_rows]


def llm_response_to_hypotheses(
    *,
    parsed: Any,
    page_windows: list[dict[str, Any]],
    context: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[WordHypothesis], dict[str, Any]]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("hypotheses"), list):
        return [], {"invalid_response": 1}
    page_lookup = {str(page.get("test_id")): page for page in page_windows}
    rows: list[WordHypothesis] = []
    seen: set[tuple[str, int, int, str, tuple[tuple[str, str], ...]]] = set()
    rejection_counts: dict[str, int] = {}
    rejection_examples: list[dict[str, Any]] = []

    def reject(reason: str, item: dict[str, Any]) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        if len(rejection_examples) < 20:
            rejection_examples.append({
                "reason": reason,
                "test_id": item.get("test_id"),
                "start": item.get("start"),
                "end": item.get("end"),
                "observed": item.get("observed"),
                "target": item.get("target"),
            })

    for index, item in enumerate(parsed.get("hypotheses") or [], start=1):
        if not isinstance(item, dict):
            continue
        test_id = str(item.get("test_id") or "")
        page = page_lookup.get(test_id)
        if not page:
            reject("unknown_test_id", item)
            continue
        try:
            start = int(item.get("start"))
            end = int(item.get("end"))
        except (TypeError, ValueError):
            reject("bad_span", item)
            continue
        text = str(page.get("text") or "")
        sources = list(page.get("sources") or [])
        supplied_observed = normalize_az(str(item.get("observed") or ""))
        located = locate_observed_span(
            text=text,
            supplied_observed=supplied_observed,
            requested_start=start,
            requested_end=end,
            max_shift=args.llm_max_span_shift,
        )
        if located is None:
            reject("observed_not_found_near_span", item)
            continue
        start, end = located
        observed = text[start:end]
        target = normalize_az(str(item.get("target") or ""))
        if not target or target == observed:
            reject("target_length_mismatch_or_identity", {**item, "start": start, "end": end, "observed": observed})
            continue
        if args.phrase_edit_model == "same_length" and len(target) != len(observed):
            reject("target_length_mismatch_or_identity", {**item, "start": start, "end": end, "observed": observed})
            continue
        if args.phrase_edit_model == "same_length":
            edits = implied_edits(
                observed=observed,
                target=target,
                sources=sources[start:end],
                consensus=context["consensus"],
                alphabet=context["alphabet"],
                baseline_key=context["root_key"],
                baseline_mask=context["root_mask"],
                allow_stable_edits=args.allow_stable_edits,
            )
            rejection_reason = "no_edits"
        else:
            edits, rejection_reason = derive_phrase_edits(
                observed=observed,
                target=target,
                sources=sources[start:end],
                consensus=context["consensus"],
                alphabet=context["alphabet"],
                baseline_key=context["root_key"],
                baseline_mask=context["root_mask"],
                allow_stable_edits=args.allow_stable_edits,
                allow_delete_to_null=True,
                max_edits=args.max_edits,
            )
        if not edits:
            if bool(getattr(args, "llm_salvage_fragments", True)) and rejection_reason in {
                "conflicting_symbol_edits",
                "too_many_edits",
                "target_requires_insertions",
            }:
                salvaged = salvage_phrase_fragments(
                    test_id=test_id,
                    window_start=window_start_for(page, start),
                    start=start,
                    observed=observed,
                    target=target,
                    sources=sources[start:end],
                    context=context,
                    args=args,
                    dictionary_rank=index,
                )
                for row in salvaged:
                    key = (row.test_id, row.start, row.end, row.target, row.edits)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(row)
                if salvaged:
                    rejection_reason = f"{rejection_reason}_salvaged"
            reject(rejection_reason, {**item, "start": start, "end": end, "observed": observed})
            continue
        if len(edits) > args.max_edits:
            reject("too_many_edits", {**item, "start": start, "end": end, "observed": observed})
            continue
        key = (test_id, start, end, target, edits)
        if key in seen:
            reject("duplicate", {**item, "start": start, "end": end, "observed": observed})
            continue
        seen.add(key)
        rows.append(
            WordHypothesis(
                test_id=test_id,
                window_start=window_start_for(page, start),
                start=start,
                end=end,
                observed=observed,
                target=target,
                edits=edits,
                distance=edit_distance(observed, target),
                dictionary_rank=index,
                local_score=float(len(target)) - 1.2 * edit_distance(observed, target),
            )
        )
    rows.sort(key=lambda row: (row.local_score, len(row.target), -row.distance), reverse=True)
    return dedupe_by_edits(rows), {
        "counts": rejection_counts,
        "examples": rejection_examples,
        "accepted_before_dedupe": len(rows),
    }


def salvage_phrase_fragments(
    *,
    test_id: str,
    window_start: int,
    start: int,
    observed: str,
    target: str,
    sources: list[str],
    context: dict[str, Any],
    args: argparse.Namespace,
    dictionary_rank: int,
) -> list[WordHypothesis]:
    alignment = align_observed_to_target_without_insertions(observed, target)
    if alignment is None:
        return []
    positioned: list[dict[str, Any]] = []
    observed_index = 0
    for observed_char, target_char in alignment:
        if observed_char is None:
            return []
        positioned.append({
            "observed_index": observed_index,
            "observed_char": observed_char,
            "target_char": target_char,
        })
        observed_index += 1
    candidates: list[WordHypothesis] = []
    min_len = max(1, int(getattr(args, "llm_salvage_min_len", 12) or 12))
    max_len = max(min_len, int(getattr(args, "llm_salvage_max_len", 36) or 36))
    step = 2
    for left in range(0, len(positioned), step):
        for right in range(left + min_len, min(len(positioned), left + max_len) + 1, step):
            segment = positioned[left:right]
            observed_fragment = "".join(str(row["observed_char"]) for row in segment)
            target_fragment = "".join(str(row["target_char"]) for row in segment if row["target_char"] is not None)
            if len(target_fragment) < min_len or target_fragment == observed_fragment:
                continue
            source_fragment = sources[left:right]
            edits, reason = derive_phrase_edits(
                observed=observed_fragment,
                target=target_fragment,
                sources=source_fragment,
                consensus=context["consensus"],
                alphabet=context["alphabet"],
                baseline_key=context["root_key"],
                baseline_mask=context["root_mask"],
                allow_stable_edits=args.allow_stable_edits,
                allow_delete_to_null=True,
                max_edits=args.max_edits,
            )
            if not edits or reason:
                continue
            distance = edit_distance(observed_fragment, target_fragment)
            candidates.append(
                WordHypothesis(
                    test_id=test_id,
                    window_start=window_start,
                    start=start + left,
                    end=start + right,
                    observed=observed_fragment,
                    target=target_fragment,
                    edits=edits,
                    distance=distance,
                    dictionary_rank=dictionary_rank,
                    local_score=float(len(target_fragment)) - 1.2 * distance - 0.75,
                )
            )
    candidates.sort(key=lambda row: (row.local_score, len(row.target), -row.distance), reverse=True)
    return dedupe_by_edits(candidates)[: max(0, int(getattr(args, "llm_salvage_top_n", 4) or 4))]


def locate_observed_span(
    *,
    text: str,
    supplied_observed: str,
    requested_start: int,
    requested_end: int,
    max_shift: int,
) -> tuple[int, int] | None:
    if supplied_observed:
        exact_positions = [match.start() for match in re.finditer(re.escape(supplied_observed), text)]
        if exact_positions:
            best = min(exact_positions, key=lambda pos: abs(pos - requested_start))
            if abs(best - requested_start) <= max_shift:
                return best, best + len(supplied_observed)
    if 0 <= requested_start < requested_end <= len(text):
        return requested_start, requested_end
    return None


def window_start_for(page: dict[str, Any], position: int) -> int:
    candidates = [
        int(window.get("start") or 0)
        for window in (page.get("windows") or [])
        if int(window.get("start") or 0) <= position < int(window.get("end") or 0)
    ]
    return max(candidates) if candidates else 0


def implied_phrase_edits(
    *,
    observed: str,
    target: str,
    sources: list[str],
    consensus: dict[str, dict[str, Any]],
    alphabet: Any,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    allow_stable_edits: bool,
    allow_delete_to_null: bool,
) -> tuple[tuple[str, str], ...]:
    return derive_phrase_edits(
        observed=observed,
        target=target,
        sources=sources,
        consensus=consensus,
        alphabet=alphabet,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        allow_stable_edits=allow_stable_edits,
        allow_delete_to_null=allow_delete_to_null,
        max_edits=None,
    )[0]


def derive_phrase_edits(
    *,
    observed: str,
    target: str,
    sources: list[str],
    consensus: dict[str, dict[str, Any]],
    alphabet: Any,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    allow_stable_edits: bool,
    allow_delete_to_null: bool,
    max_edits: int | None,
) -> tuple[tuple[tuple[str, str], ...], str]:
    if len(observed) != len(sources):
        return (), "source_alignment_mismatch"
    if allow_delete_to_null:
        return project_phrase_edits(
            observed=observed,
            target=target,
            sources=sources,
            consensus=consensus,
            alphabet=alphabet,
            baseline_key=baseline_key,
            baseline_mask=baseline_mask,
            allow_stable_edits=allow_stable_edits,
            max_edits=max_edits,
        )
    return derive_phrase_edits_from_alignment(
        alignment=align_observed_to_target(observed, target),
        observed=observed,
        sources=sources,
        consensus=consensus,
        alphabet=alphabet,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        allow_stable_edits=allow_stable_edits,
    )


def derive_phrase_edits_from_alignment(
    *,
    alignment: list[tuple[str | None, str | None]],
    observed: str,
    sources: list[str],
    consensus: dict[str, dict[str, Any]],
    alphabet: Any,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    allow_stable_edits: bool,
) -> tuple[tuple[tuple[str, str], ...], str]:
    from probe_copiale_multipage_global_repair import current_assignment

    symbol_targets: dict[str, str] = {}
    masked = set(baseline_mask)
    observed_index = 0
    for observed_char, target_char in alignment:
        if observed_char is None:
            # Insertions require unmasking or logogram expansion. That is a
            # richer model than this first verifier slice supports.
            return (), "target_requires_insertions"
        if observed_index >= len(sources):
            return (), "source_alignment_mismatch"
        symbol = sources[observed_index]
        observed_index += 1
        if target_char is None:
            if not allow_delete_to_null:
                return (), "target_deletion_not_enabled"
            proposed = "<null>"
        elif observed_char == target_char:
            continue
        else:
            proposed = target_char
        if not symbol or not alphabet.has_symbol(symbol):
            return (), "unknown_source_symbol"
        info = consensus.get(symbol) or {}
        if info.get("stable") and not allow_stable_edits:
            return (), "would_edit_stable_symbol"
        if symbol in masked and proposed == "<null>":
            continue
        if symbol in masked and proposed != "<null>":
            return (), "would_unmask_to_letter"
        prior = symbol_targets.get(symbol)
        if prior is not None and prior != proposed:
            return (), "conflicting_symbol_edits"
        if proposed != "<null>":
            token_id = alphabet.id_for(symbol)
            current = current_assignment(symbol, token_id, baseline_key, baseline_mask)
            if current == proposed:
                continue
        symbol_targets[symbol] = proposed
    if observed_index != len(sources):
        return (), "source_alignment_mismatch"
    edits = tuple(sorted(symbol_targets.items()))
    if not edits:
        return (), "no_effective_edits"
    return edits, ""


def align_observed_to_target(observed: str, target: str) -> list[tuple[str | None, str | None]]:
    """Align observed plaintext to target phrase.

    Substitution and observed deletion are supported; target insertions are
    represented in the alignment but later rejected by the current edit model.
    """
    m = len(observed)
    n = len(target)
    costs = [[0] * (n + 1) for _ in range(m + 1)]
    back: list[list[str | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        costs[i][0] = costs[i - 1][0] + 1
        back[i][0] = "delete"
    for j in range(1, n + 1):
        costs[0][j] = costs[0][j - 1] + 3
        back[0][j] = "insert"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = 0 if observed[i - 1] == target[j - 1] else 2
            candidates = [
                (costs[i - 1][j - 1] + sub_cost, 3, "diag"),
                (costs[i - 1][j] + 1, 2, "delete"),
                (costs[i][j - 1] + 3, 1, "insert"),
            ]
            best_cost, _rank, move = min(candidates, key=lambda row: (row[0], -row[1]))
            costs[i][j] = best_cost
            back[i][j] = move
    rows: list[tuple[str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            rows.append((observed[i - 1], target[j - 1]))
            i -= 1
            j -= 1
        elif move == "delete":
            rows.append((observed[i - 1], None))
            i -= 1
        else:
            rows.append((None, target[j - 1]))
            j -= 1
    rows.reverse()
    return rows


def align_observed_to_target_without_insertions(
    observed: str,
        target: str,
    ) -> list[tuple[str | None, str | None]] | None:
    """Align target onto observed positions using substitutions and deletions only.

    This matches the phrase verifier's edit language: every target character
    must be assigned to an existing cipher symbol, while extra observed
    symbols may be proposed as nulls.
    """
    m = len(observed)
    n = len(target)
    if n > m:
        return None
    costs = [[10**9] * (n + 1) for _ in range(m + 1)]
    back: list[list[str | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    costs[0][0] = 0
    for i in range(1, m + 1):
        costs[i][0] = costs[i - 1][0] + 1
        back[i][0] = "delete"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = 0 if observed[i - 1] == target[j - 1] else 2
            candidates = [
                (costs[i - 1][j - 1] + sub_cost, 2, "diag"),
                (costs[i - 1][j] + 1, 1, "delete"),
            ]
            best_cost, _rank, move = min(candidates, key=lambda row: (row[0], -row[1]))
            costs[i][j] = best_cost
            back[i][j] = move
    rows: list[tuple[str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            rows.append((observed[i - 1], target[j - 1]))
            i -= 1
            j -= 1
        elif move == "delete":
            rows.append((observed[i - 1], None))
            i -= 1
        else:
            return None
    rows.reverse()
    return rows


def project_phrase_edits(
    *,
    observed: str,
    target: str,
    sources: list[str],
    consensus: dict[str, dict[str, Any]],
    alphabet: Any,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    allow_stable_edits: bool,
    max_edits: int | None,
) -> tuple[tuple[tuple[str, str], ...], str]:
    """Find a no-insertion phrase projection that is consistent with one key.

    Unlike generic edit-distance alignment, this searches over possible
    substitution/delete alignments while enforcing that repeated cipher symbols
    receive the same final assignment everywhere in the phrase.
    """
    if len(target) > len(observed):
        return (), "target_requires_insertions"
    from probe_copiale_multipage_global_repair import current_assignment

    masked = set(baseline_mask)
    beam_cap = 500
    # score, consumed target chars, edits
    states: list[tuple[float, int, tuple[tuple[str, str], ...]]] = [(0.0, 0, ())]

    def extend_edits(
        edits: tuple[tuple[str, str], ...],
        symbol: str,
        required: str,
    ) -> tuple[tuple[tuple[str, str], ...] | None, str]:
        existing = dict(edits)
        prior = existing.get(symbol)
        if prior is not None:
            if prior == required:
                return edits, ""
            return None, "conflicting_symbol_edits"
        if not symbol or not alphabet.has_symbol(symbol):
            return None, "unknown_source_symbol"
        info = consensus.get(symbol) or {}
        if info.get("stable") and not allow_stable_edits:
            return None, "would_edit_stable_symbol"
        if symbol in masked and required == "<null>":
            return edits, ""
        if symbol in masked and required != "<null>":
            return None, "would_unmask_to_letter"
        token_id = alphabet.id_for(symbol)
        current = current_assignment(symbol, token_id, baseline_key, baseline_mask)
        if current == required:
            return edits, ""
        new_edits = tuple(sorted((*edits, (symbol, required))))
        if max_edits is not None and len(new_edits) > max_edits:
            return None, "too_many_edits"
        return new_edits, ""

    last_reason = "conflicting_symbol_edits"
    for i, observed_char in enumerate(observed):
        symbol = sources[i]
        next_states: list[tuple[float, int, tuple[tuple[str, str], ...]]] = []
        remaining_observed_after = len(observed) - i - 1
        for score, j, edits in states:
            # Delete/null this observed symbol if enough observed symbols remain
            # to cover the rest of the target.
            if remaining_observed_after >= len(target) - j:
                next_edits, reason = extend_edits(edits, symbol, "<null>")
                if next_edits is not None:
                    next_states.append((score + 1.0, j, next_edits))
                elif reason:
                    last_reason = reason
            # Consume the next target character at this observed position.
            if j < len(target):
                required = target[j]
                next_edits, reason = extend_edits(edits, symbol, required)
                if next_edits is not None:
                    mismatch = 0.0 if observed_char == required else 2.0
                    edit_pressure = 0.05 * max(0, len(next_edits) - len(edits))
                    next_states.append((score + mismatch + edit_pressure, j + 1, next_edits))
                elif reason:
                    last_reason = reason
        if not next_states:
            return (), last_reason
        next_states.sort(key=lambda row: (row[0], len(row[2]), -row[1]))
        deduped: list[tuple[float, int, tuple[tuple[str, str], ...]]] = []
        seen: set[tuple[int, tuple[tuple[str, str], ...]]] = set()
        for row in next_states:
            signature = (row[1], row[2])
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(row)
            if len(deduped) >= beam_cap:
                break
        states = deduped
    finals = [row for row in states if row[1] == len(target)]
    if not finals:
        return (), "target_requires_insertions"
    finals.sort(key=lambda row: (row[0], len(row[2])))
    edits = finals[0][2]
    if not edits:
        return (), "no_effective_edits"
    return edits, ""


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left_char == right_char else 1),
                )
            )
        previous = current
    return previous[-1]


def dedupe_by_edits(rows: list[WordHypothesis]) -> list[WordHypothesis]:
    selected: list[WordHypothesis] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        signature = tuple(sorted(row.edits))
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(row)
    return selected


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(1 for a, b in zip(left, right) if a != b)


def normalize_az(text: str) -> str:
    return AZ_RE.sub("", text.upper().replace("Ä", "A").replace("Ö", "O").replace("Ü", "U"))


def compact_page_windows(page_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "test_id": row.get("test_id"),
            "windows": [
                {
                    "start": window.get("start"),
                    "end": window.get("end"),
                    "damage_score": window.get("damage_score"),
                    "text": window.get("text"),
                }
                for window in (row.get("windows") or [])
            ],
        }
        for row in page_windows
    ]


def serializable_settings(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
    return result


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Copiale Phrase Hypothesis Probe",
        "",
        "Generation and ranking are ground-truth-free. Post-hoc character accuracy is calibration only.",
        "",
        f"- Mode: `{payload['mode']}`",
        f"- Label: `{payload['label']}`",
        f"- Hypotheses: `{payload['hypothesis_count']}`",
        f"- Variants: `{payload['variant_count']}`",
        f"- Elapsed seconds: `{payload['elapsed_seconds']}`",
        "",
    ]
    llm = payload.get("llm_packet") if isinstance(payload.get("llm_packet"), dict) else {}
    usage = llm.get("usage") if isinstance(llm.get("usage"), dict) else {}
    if usage:
        lines.extend([
            "## LLM Usage",
            "",
            f"- Input tokens: `{usage.get('input_tokens', 0)}`",
            f"- Output tokens: `{usage.get('output_tokens', 0)}`",
            f"- Estimated cost: `${float(usage.get('estimated_cost_usd') or 0.0):.4f}`",
            "",
        ])
    selection = payload.get("selection_summary") if isinstance(payload.get("selection_summary"), dict) else {}
    if selection:
        scorer_pick = selection.get("scorer_pick") if isinstance(selection.get("scorer_pick"), dict) else {}
        post_hoc_best = selection.get("post_hoc_best") if isinstance(selection.get("post_hoc_best"), dict) else {}
        lines.extend([
            "## Selection Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Variants evaluated | {selection.get('variant_count', '')} |",
            f"| Scorer pick post-hoc char | {format_pct(selection.get('scorer_pick_post_hoc_char'))} |",
            f"| Best post-hoc char | {format_pct(selection.get('post_hoc_best_char'))} |",
            f"| Scorer pick gap from best | {format_pct(selection.get('scorer_pick_gap_from_best'))} |",
            f"| Scorer pick gain over baseline | {format_signed_pct(selection.get('scorer_pick_gain_over_baseline'))} |",
            f"| Best post-hoc gain over baseline | {format_signed_pct(selection.get('post_hoc_best_gain_over_baseline'))} |",
            f"| Best post-hoc rank by scorer | {selection.get('post_hoc_best_scorer_rank', '')} |",
            f"| Scorer pick post-hoc rank | {selection.get('scorer_pick_post_hoc_rank', '')} |",
            f"| Best captured in scorer top-3 | {format_bool(selection.get('post_hoc_best_in_scorer_top_3'))} |",
            f"| Best captured in scorer top-5 | {format_bool(selection.get('post_hoc_best_in_scorer_top_5'))} |",
            f"| Best captured in scorer top-10 | {format_bool(selection.get('post_hoc_best_in_scorer_top_10'))} |",
            "",
            "| Pick | Post-Hoc | Adj | AdjNoTarget | Robust | Edits | Preview |",
            "|---|---:|---:|---:|---:|---|---|",
            selection_table_row("Scorer pick", scorer_pick),
            selection_table_row("Post-hoc best", post_hoc_best),
            "",
        ])
    if llm:
        parsed = llm.get("parsed_response") if isinstance(llm.get("parsed_response"), dict) else {}
        raw_hypotheses = parsed.get("hypotheses") if isinstance(parsed.get("hypotheses"), list) else []
        rejection = llm.get("rejection_summary") if isinstance(llm.get("rejection_summary"), dict) else {}
        counts = rejection.get("counts") if isinstance(rejection.get("counts"), dict) else {}
        examples = rejection.get("examples") if isinstance(rejection.get("examples"), list) else []
        lines.extend([
            "## LLM Proposal Diagnostics",
            "",
            f"- Raw proposals: `{len(raw_hypotheses)}`",
            f"- Accepted before dedupe: `{rejection.get('accepted_before_dedupe', 0)}`",
            "",
        ])
        if counts:
            lines.extend([
                "| Rejection reason | Count |",
                "|---|---:|",
            ])
            for reason, count in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0]))):
                lines.append(f"| `{reason}` | {count} |")
            lines.append("")
        if examples:
            lines.extend([
                "| Reason | Test | Span | Observed | Target |",
                "|---|---|---|---|---|",
            ])
            for row in examples[:12]:
                lines.append(
                    f"| `{row.get('reason')}` | `{row.get('test_id')}` | "
                    f"{row.get('start')}-{row.get('end')} | "
                    f"`{escape_cell(str(row.get('observed') or ''))}` | "
                    f"`{escape_cell(str(row.get('target') or ''))}` |"
                )
            lines.append("")
    lines.extend([
        "## Top Phrase Hypotheses",
        "",
        "| Rank | Test | Span | Observed | Target | Distance | Edits | Score |",
        "|---:|---|---|---|---|---:|---|---:|",
    ])
    for index, row in enumerate(payload.get("top_phrase_hypotheses") or [], start=1):
        lines.append(
            f"| {index} | `{row.get('test_id')}` | {row.get('start')}-{row.get('end')} | "
            f"`{row.get('observed')}` | `{row.get('target')}` | {row.get('distance')} | "
            f"{escape_cell('<br>'.join(f'`{edit}`' for edit in (row.get('edits') or [])))} | "
            f"{float(row.get('local_score') or 0.0):.3f} |"
        )
    lines.extend([
        "",
        "## Top Variants",
        "",
        "| Rank | Post-Hoc | Adj | AdjNoTarget | Robust | Edits | Preview |",
        "|---:|---:|---:|---:|---:|---|---|",
    ])
    for index, row in enumerate(payload.get("top_variants") or [], start=1):
        adj = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
        lines.append(
            f"| {index} | {format_pct(row.get('post_hoc_char_avg'))} | "
            f"{format_float(adj.get('adjudication_score'))} | "
            f"{format_float(adj.get('adjudication_no_target_score'))} | "
            f"{format_float(row.get('page_robust_score'))} | "
            f"{escape_cell('<br>'.join(f'`{edit}`' for edit in (row.get('edits') or [])))} | "
            f"{escape_cell(str(row.get('preview') or '')[:180])} |"
        )
    lines.extend([
        "",
        "## Top By Post-Hoc Character Accuracy",
        "",
        "| Rank | Post-Hoc | Adj | AdjNoTarget | Robust | Edits | Preview |",
        "|---:|---:|---:|---:|---:|---|---|",
    ])
    for index, row in enumerate(payload.get("top_variants_by_post_hoc") or [], start=1):
        adj = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
        lines.append(
            f"| {index} | {format_pct(row.get('post_hoc_char_avg'))} | "
            f"{format_float(adj.get('adjudication_score'))} | "
            f"{format_float(adj.get('adjudication_no_target_score'))} | "
            f"{format_float(row.get('page_robust_score'))} | "
            f"{escape_cell('<br>'.join(f'`{edit}`' for edit in (row.get('edits') or [])))} | "
            f"{escape_cell(str(row.get('preview') or '')[:180])} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def format_pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else ""


def format_signed_pct(value: Any) -> str:
    return f"{float(value) * 100:+.1f}%" if isinstance(value, (int, float)) else ""


def format_float(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else ""


def format_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return ""


def selection_table_row(label: str, row: dict[str, Any]) -> str:
    adj = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    return (
        f"| {label} | {format_pct(row.get('post_hoc_char_avg'))} | "
        f"{format_float(adj.get('adjudication_score'))} | "
        f"{format_float(adj.get('adjudication_no_target_score'))} | "
        f"{format_float(row.get('page_robust_score'))} | "
        f"{escape_cell('<br>'.join(f'`{edit}`' for edit in (row.get('edits') or [])))} | "
        f"{escape_cell(str(row.get('preview') or '')[:180])} |"
    )


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
