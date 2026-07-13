#!/usr/bin/env python3
"""Find reader-visible missing-word slots in damaged no-boundary plaintext.

This is deliberately reading-first.  It starts from a completed candidate
decipherment, segments the damaged plaintext into word islands, and asks where
the reader would naturally suspect that an entire word is missing.  Only after
that does it attach those slots back to cipher symbols and recurrence counts.

Ground truth, when available through Copiale metadata, is used only in the
post-hoc calibration columns marked as such.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from analysis.dictionary import get_dictionary_path, load_word_set  # noqa: E402
from analysis.segment import segment_text  # noqa: E402
from benchmark.loader import BenchmarkLoader, parse_canonical_transcription, resolve_test_language  # noqa: E402
from agent.model_provider import canonical_provider, estimate_provider_cost  # noqa: E402
from analysis.nomenclator import render_token_views  # noqa: E402
from probe_logogram_hypotheses import DEFAULT_CANDIDATES  # noqa: E402
from rank_candidate_texts_with_llm import call_llm, parse_json_response, visible_response_text  # noqa: E402
from run_copiale_logogram_repair_experiment import (  # noqa: E402
    artifact_paths_from_args,
    basin_hints_from_summary,
    load_true_logogram_symbols,
    select_basins,
)


@dataclass(frozen=True)
class WordSpan:
    word: str
    start: int
    end: int
    known: bool


@dataclass(frozen=True)
class SideWord:
    word: str
    known: bool
    strength: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument("--summary-json", default="artifacts/copiale_breadth_experiment/four_page_wide/summary.json")
    parser.add_argument("--artifact", action="append", default=[], help="Additional artifact JSON to inspect.")
    parser.add_argument("--test-id", action="append", default=[], help="Restrict to one or more test ids.")
    parser.add_argument("--top-basins-per-test", type=int, default=1)
    parser.add_argument("--context-chars", type=int, default=28)
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument(
        "--max-hypothesis-symbols",
        type=int,
        default=80,
        help="How many symbol-level hole candidates to carry into the recurrence-level missing-word hypothesis stage.",
    )
    parser.add_argument("--top-hypotheses", type=int, default=16)
    parser.add_argument("--max-occurrences-per-symbol", type=int, default=12)
    parser.add_argument("--include-one-letter-symbols", action="store_true")
    parser.add_argument("--min-occurrences", type=int, default=1)
    parser.add_argument(
        "--rereader",
        choices=("local", "llm"),
        default="local",
        help="Use a local recurrence scorer or an LLM semantic rereader for missing-word hypotheses.",
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/copiale_reading_hole_probe")
    args = parser.parse_args()

    started = time.monotonic()
    benchmark_root = resolve_path(Path(args.benchmark_root))
    output_dir = resolve_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = BenchmarkLoader(benchmark_root)
    test_rows = {test.test_id: test for test in loader.load_tests(args.split)}
    basins = select_basins(
        artifact_paths=artifact_paths_from_args(args),
        loader=loader,
        test_rows=test_rows,
        test_filter=set(args.test_id),
        top_per_test=args.top_basins_per_test,
        hints=basin_hints_from_summary(args),
    )
    if not basins:
        raise SystemExit("No candidate basins found. Check --summary-json/--artifact paths.")

    true_logograms = load_true_logogram_symbols(benchmark_root)
    rows = []
    for index, basin in enumerate(basins, start=1):
        print(f"[{index}/{len(basins)}] {basin.test_id} {basin.label}", flush=True)
        rows.append(
            analyze_basin(
                basin=basin,
                loader=loader,
                test_rows=test_rows,
                true_logograms=true_logograms,
                args=args,
            )
        )

    payload = {
        "experiment": "reading_first_missing_word_holes",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "benchmark_root": str(benchmark_root),
        "split": args.split,
        "settings": {
            "top_basins_per_test": args.top_basins_per_test,
            "context_chars": args.context_chars,
            "max_symbols": args.max_symbols,
            "max_hypothesis_symbols": args.max_hypothesis_symbols,
            "top_hypotheses": args.top_hypotheses,
            "max_occurrences_per_symbol": args.max_occurrences_per_symbol,
            "include_one_letter_symbols": args.include_one_letter_symbols,
            "min_occurrences": args.min_occurrences,
            "rereader": args.rereader,
            "provider": canonical_provider(args.provider),
            "model": args.model,
            "dry_run": bool(args.dry_run),
        },
        "aggregate": aggregate(rows),
        "rows": rows,
    }
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def analyze_basin(*, basin: Any, loader: BenchmarkLoader, test_rows: dict[str, Any], true_logograms: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    test_data = loader.load_test_data(test_rows[basin.test_id])
    language = resolve_test_language(test_data, "de")
    word_set = language_word_set(language)
    cipher_text = parse_canonical_transcription(test_data.canonical_transcription)
    views, baseline = render_token_views(cipher_text, key=basin.key, mask=basin.mask)
    spans = segment_spans(baseline, word_set)
    broken_words = broken_word_rows(spans, baseline, limit=16)

    by_symbol: dict[str, list[Any]] = {}
    for view in views:
        if view.assignment in {"<null>", "?"} or (args.include_one_letter_symbols and len(view.rendered) == 1):
            by_symbol.setdefault(view.symbol, []).append(view)

    symbol_rows = []
    for symbol, symbol_views in by_symbol.items():
        if len(symbol_views) < args.min_occurrences:
            continue
        symbol_rows.append(
            score_symbol_as_reader_hole(
                symbol=symbol,
                symbol_views=symbol_views,
                baseline=baseline,
                word_set=word_set,
                context_chars=args.context_chars,
                occurrence_limit=args.max_occurrences_per_symbol,
            )
        )
    symbol_rows.sort(key=lambda row: (row["hole_score"], row["hole_occurrences"], row["occurrence_count"]), reverse=True)
    hypothesis_input = symbol_rows[: max(args.max_symbols, args.max_hypothesis_symbols, 0)]
    all_missing_word_hypotheses = build_missing_word_hypotheses(
        hypothesis_input,
        top_n=len(hypothesis_input),
    )
    if args.rereader == "llm":
        missing_word_hypotheses, rereader_packet = llm_missing_word_rereader(
            local_hypotheses=all_missing_word_hypotheses[: args.max_hypothesis_symbols],
            baseline=baseline,
            language=language,
            args=args,
        )
        missing_word_hypotheses = missing_word_hypotheses[: args.top_hypotheses]
    else:
        missing_word_hypotheses = all_missing_word_hypotheses[: args.top_hypotheses]
        rereader_packet = {}
    symbol_rows = symbol_rows[: max(0, args.max_symbols)]
    true_present = sorted(set(token_symbols(cipher_text)) & set(true_logograms))
    recognized = [str(row["symbol"]) for row in symbol_rows]
    hypothesized = [str(row["symbol"]) for row in missing_word_hypotheses]
    return {
        "test_id": basin.test_id,
        "artifact": str(basin.artifact),
        "basin": basin.label,
        "baseline_char": round(float(basin.char_accuracy), 6),
        "baseline_preview": baseline[:260],
        "segmented_preview": " ".join(span.word for span in spans[:42]),
        "broken_words": broken_words,
        "true_logogram_symbols_present": [
            {"symbol": symbol, "gloss": true_logograms[symbol]}
            for symbol in true_present
        ],
        "recognized_true_logograms": sorted(set(recognized) & set(true_present)),
        "hypothesized_true_logograms": sorted(set(hypothesized) & set(true_present)),
        "true_logogram_hypothesis_ranks": true_symbol_ranks(
            all_missing_word_hypotheses,
            true_logograms=true_logograms,
            true_present=true_present,
        ),
        "recognition_hit_rate": safe_rate(len(set(recognized) & set(true_present)), len(true_present)),
        "hypothesis_hit_rate": safe_rate(len(set(hypothesized) & set(true_present)), len(true_present)),
        "symbol_rows": symbol_rows,
        "missing_word_hypotheses": missing_word_hypotheses,
        "rereader_packet": rereader_packet,
    }


def language_word_set(language: str) -> set[str]:
    path = get_dictionary_path(language)
    words = load_word_set(path) if path else set()
    for word in DEFAULT_CANDIDATES.get(language, []):
        if len(word) > 1:
            words.add(word.upper())
    # A small Copiale/Masonic German overlay helps the segmenter treat obvious
    # islands as islands; it does not nominate logogram symbols or use GT.
    if language == "de":
        words.update({
            "AUFRICHTIGEN", "BEWEGEN", "BRUDER", "BRUEDER", "FREIMAURER",
            "GEHEIMNIS", "GRIFF", "HAND", "LOGE", "MEISTER", "ORDEN",
            "RECHTSCHAFFENEN", "SCHLIESSUNG", "ZEICHEN",
        })
    return words


def segment_spans(text: str, word_set: set[str]) -> list[WordSpan]:
    result = segment_text(text, word_set)
    spans: list[WordSpan] = []
    cursor = 0
    for word in result.words:
        start = cursor
        end = start + len(word)
        spans.append(WordSpan(word=word, start=start, end=end, known=word in word_set))
        cursor = end
    return spans


def broken_word_rows(spans: list[WordSpan], baseline: str, *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for span in spans:
        if span.known or len(span.word) < 5:
            continue
        left = baseline[max(0, span.start - 18):span.start]
        right = baseline[span.end:min(len(baseline), span.end + 18)]
        rows.append({
            "word": span.word,
            "start": span.start,
            "length": len(span.word),
            "context": left + "⟦" + span.word + "⟧" + right,
        })
    rows.sort(key=lambda row: (row["length"], -row["start"]), reverse=True)
    return rows[:limit]


def score_symbol_as_reader_hole(
    *,
    symbol: str,
    symbol_views: list[Any],
    baseline: str,
    word_set: set[str],
    context_chars: int,
    occurrence_limit: int,
) -> dict[str, Any]:
    occurrence_rows = [
        score_occurrence_as_hole(
            view=view,
            baseline=baseline,
            word_set=word_set,
            context_chars=context_chars,
        )
        for view in symbol_views[: max(1, occurrence_limit)]
    ]
    hole_rows = [row for row in occurrence_rows if row["hole_score"] >= 1.45]
    score = 0.0
    if occurrence_rows:
        score = sum(float(row["hole_score"]) for row in occurrence_rows) / len(occurrence_rows)
    assignment = str(symbol_views[0].assignment)
    return {
        "symbol": symbol,
        "assignment": assignment,
        "signal": "missing_or_unmapped_symbol" if assignment in {"<null>", "?"} else "one_letter_symbol",
        "occurrence_count": len(symbol_views),
        "reviewed_occurrences": len(occurrence_rows),
        "hole_occurrences": len(hole_rows),
        "hole_rate_reviewed": safe_rate(len(hole_rows), len(occurrence_rows)),
        "hole_score": round(score + min(0.45, len(symbol_views) / 40.0), 6),
        "examples": sorted(occurrence_rows, key=lambda row: row["hole_score"], reverse=True)[:5],
    }


def build_missing_word_hypotheses(symbol_rows: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    rows = []
    for row in symbol_rows:
        reviewed = int(row.get("reviewed_occurrences") or 0)
        holes = int(row.get("hole_occurrences") or 0)
        occurrences = int(row.get("occurrence_count") or 0)
        hole_rate = float(row.get("hole_rate_reviewed") or 0.0)
        examples = list(row.get("examples") or [])
        strong_examples = [example for example in examples if float(example.get("hole_score") or 0.0) >= 1.65]
        support = recurrence_support(holes=holes, reviewed=reviewed, occurrences=occurrences)
        shape = shape_consistency(str(row.get("assignment") or ""), examples)
        score = (
            float(row.get("hole_score") or 0.0) * 0.45
            + support * 0.35
            + shape * 0.20
            + min(0.25, len(strong_examples) * 0.04)
        )
        if row.get("signal") == "one_letter_symbol":
            score -= ordinary_letter_penalty(str(row.get("assignment") or ""), occurrences)
        status = classify_missing_word_score(score=score, holes=holes, hole_rate=hole_rate)
        rows.append({
            "symbol": row.get("symbol"),
            "assignment": row.get("assignment"),
            "signal": row.get("signal"),
            "occurrence_count": occurrences,
            "reviewed_occurrences": reviewed,
            "hole_occurrences": holes,
            "hole_rate_reviewed": row.get("hole_rate_reviewed"),
            "missing_word_score": round(score, 6),
            "status": status,
            "reason": missing_word_reason(row, score=score, support=support, shape=shape),
            "recurrence_reread": [
                missing_word_context(str(row.get("symbol") or ""), example)
                for example in examples[:5]
            ],
        })
    rows.sort(
        key=lambda row: (
            status_rank(str(row.get("status") or "")),
            float(row.get("missing_word_score") or 0.0),
            int(row.get("hole_occurrences") or 0),
            int(row.get("occurrence_count") or 0),
        ),
        reverse=True,
    )
    return rows[: max(0, top_n)]


def true_symbol_ranks(
    hypotheses: list[dict[str, Any]],
    *,
    true_logograms: dict[str, str],
    true_present: list[str],
) -> list[dict[str, Any]]:
    by_symbol = {str(row.get("symbol")): (rank, row) for rank, row in enumerate(hypotheses, start=1)}
    rows = []
    for symbol in true_present:
        found = by_symbol.get(symbol)
        if not found:
            rows.append({
                "symbol": symbol,
                "gloss": true_logograms.get(symbol, ""),
                "rank": None,
            })
            continue
        rank, row = found
        rows.append({
            "symbol": symbol,
            "gloss": true_logograms.get(symbol, ""),
            "rank": rank,
            "status": row.get("status"),
            "missing_word_score": row.get("missing_word_score"),
            "hole_occurrences": row.get("hole_occurrences"),
            "occurrence_count": row.get("occurrence_count"),
        })
    return rows


def llm_missing_word_rereader(
    *,
    local_hypotheses: list[dict[str, Any]],
    baseline: str,
    language: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = build_llm_rereader_prompt(
        local_hypotheses=local_hypotheses,
        baseline=baseline,
        language=language,
    )
    packet: dict[str, Any] = {
        "provider": canonical_provider(args.provider),
        "model": args.model,
        "dry_run": bool(args.dry_run),
        "prompt": prompt,
    }
    if args.dry_run:
        return [], packet
    print(
        f"Calling LLM missing-word rereader ({canonical_provider(args.provider)}/{args.model})...",
        flush=True,
    )
    response, usage = call_llm(
        provider=args.provider,
        model=args.model,
        system=(
            "You are a careful cryptanalytic rereader. You distinguish real "
            "missing whole-word/codeword slots from ordinary damaged letters. "
            "Return only JSON."
        ),
        prompt=prompt,
        max_tokens=args.max_tokens,
    )
    text = visible_response_text(response)
    parsed = parse_json_response(text)
    hypotheses = normalize_llm_rereader_response(parsed, local_hypotheses)
    packet.update({
        "response_text": text,
        "parsed_response": parsed,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_input_tokens,
            "estimated_cost_usd": estimate_provider_cost(
                canonical_provider(args.provider),
                args.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_input_tokens,
            ),
        },
        "accepted_hypotheses": len(hypotheses),
    })
    return hypotheses, packet


def build_llm_rereader_prompt(
    *,
    local_hypotheses: list[dict[str, Any]],
    baseline: str,
    language: str,
) -> str:
    candidates = []
    for row in local_hypotheses:
        candidates.append({
            "symbol": row.get("symbol"),
            "assignment": row.get("assignment"),
            "local_status": row.get("status"),
            "local_missing_word_score": row.get("missing_word_score"),
            "occurrence_count": row.get("occurrence_count"),
            "hole_occurrences": row.get("hole_occurrences"),
            "hole_rate_reviewed": row.get("hole_rate_reviewed"),
            "recurrence_reread": row.get("recurrence_reread"),
        })
    return (
        "You are reading damaged no-boundary plaintext from a historical "
        "homophonic/nomenclator cipher.\n\n"
        "Task: rank which candidate symbols are most likely to represent a "
        "missing whole word or codeword/logogram, rather than an ordinary "
        "wrong/damaged single letter. A true missing-word candidate should make "
        "sense as a missing unit across several recurrences of the same symbol. "
        "It is fine to mark a candidate as unknown_missing_word when the exact "
        "word is not recoverable. Reject candidates where the marker is just "
        "between two segmenter-created word islands but the sentence does not "
        "read like a missing word.\n\n"
        "Do not use outside knowledge of Copiale or any known plaintext. Judge "
        "only the visible damaged text and recurrence rereads below. Nearby "
        "letters may be wrong; do not require perfection. Be conservative: "
        "ordinary common letters at word boundaries are common false positives.\n\n"
        f"Language: {language}\n\n"
        "Damaged plaintext preview:\n"
        f"{baseline[:900]}\n\n"
        "Candidate recurrence packets. In each reread, <MISSING_WORD?> marks "
        "where the same cipher symbol would be interpreted as a whole missing word:\n"
        f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
        "Return exactly JSON:\n"
        "{\n"
        '  "hypotheses": [\n'
        '    {"symbol": "S090", "judgement": "strong|plausible|weak|reject", '
        '"confidence": 0.0, "kind": "missing_word|unknown_missing_word|ordinary_letter|null", '
        '"candidate_plaintext": "", "reason": "recurrence-aware reason"}\n'
        "  ],\n"
        '  "notes": "brief note"\n'
        "}\n"
    )


def normalize_llm_rereader_response(parsed: Any, local_hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    local_by_symbol = {str(row.get("symbol")): row for row in local_hypotheses}
    raw_rows = parsed.get("hypotheses") if isinstance(parsed, dict) else None
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip()
        if symbol not in local_by_symbol:
            continue
        local = local_by_symbol[symbol]
        confidence = parse_confidence(item.get("confidence"))
        judgement = str(item.get("judgement") or "weak").strip().lower()
        kind = str(item.get("kind") or "").strip().lower()
        status = llm_status(judgement=judgement, kind=kind, confidence=confidence)
        missing_confidence = confidence if status != "unlikely_missing_word_slot" else 1.0 - confidence
        row = dict(local)
        row.update({
            "rereader": "llm",
            "llm_judgement": judgement,
            "llm_kind": kind,
            "llm_confidence": confidence,
            "llm_reject_confidence": confidence if status == "unlikely_missing_word_slot" else 0.0,
            "llm_candidate_plaintext": clean_candidate_plaintext(item.get("candidate_plaintext")),
            "missing_word_score": round(max(0.0, missing_confidence), 6),
            "status": status,
            "reason": str(item.get("reason") or "").strip(),
            "local_missing_word_score": local.get("missing_word_score"),
            "local_status": local.get("status"),
        })
        rows.append(row)
    rows.sort(
        key=lambda row: (
            status_rank(str(row.get("status") or "")),
            float(row.get("llm_confidence") or 0.0),
            float(row.get("local_missing_word_score") or 0.0),
        ),
        reverse=True,
    )
    return rows


def parse_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def clean_candidate_plaintext(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if "A" <= ch <= "Z")


def llm_status(*, judgement: str, kind: str, confidence: float) -> str:
    if judgement == "strong" and kind in {"missing_word", "unknown_missing_word"} and confidence >= 0.70:
        return "strong_missing_word_slot"
    if judgement in {"strong", "plausible"} and kind in {"missing_word", "unknown_missing_word"} and confidence >= 0.45:
        return "plausible_missing_word_slot"
    if judgement != "reject" and kind in {"missing_word", "unknown_missing_word"} and confidence >= 0.25:
        return "weak_missing_word_slot"
    return "unlikely_missing_word_slot"


def recurrence_support(*, holes: int, reviewed: int, occurrences: int) -> float:
    if reviewed <= 0:
        return 0.0
    hole_rate = holes / reviewed
    recurrence_term = min(1.0, occurrences / 10.0)
    return min(1.0, (hole_rate * 0.70) + (recurrence_term * 0.30))


def shape_consistency(assignment: str, examples: list[dict[str, Any]]) -> float:
    if not examples:
        return 0.0
    good = 0.0
    for example in examples:
        left_known = bool(example.get("left_known"))
        right_known = bool(example.get("right_known"))
        delta = float(example.get("removal_quality_delta") or 0.0)
        if left_known and right_known:
            good += 0.45
        if delta >= -0.03:
            good += 0.25
        if assignment in {"<null>", "?"}:
            good += 0.20
        elif len(assignment) == 1 and assignment not in {"E", "N", "S", "T", "R"}:
            good += 0.10
    return min(1.0, good / max(1, len(examples)))


def ordinary_letter_penalty(assignment: str, occurrences: int) -> float:
    if len(assignment) != 1:
        return 0.0
    penalty = 0.08
    if assignment in {"E", "N", "S", "T", "R"}:
        penalty += 0.14
    if occurrences >= 16:
        penalty += 0.08
    if occurrences >= 28:
        penalty += 0.12
    return penalty


def classify_missing_word_score(*, score: float, holes: int, hole_rate: float) -> str:
    if score >= 1.45 and holes >= 3 and hole_rate >= 0.35:
        return "strong_missing_word_slot"
    if score >= 1.15 and holes >= 2:
        return "plausible_missing_word_slot"
    if score >= 0.85 and holes >= 1:
        return "weak_missing_word_slot"
    return "unlikely_missing_word_slot"


def status_rank(status: str) -> int:
    return {
        "strong_missing_word_slot": 3,
        "plausible_missing_word_slot": 2,
        "weak_missing_word_slot": 1,
        "unlikely_missing_word_slot": 0,
    }.get(status, 0)


def missing_word_reason(row: dict[str, Any], *, score: float, support: float, shape: float) -> str:
    return (
        f"{row.get('hole_occurrences')} of {row.get('reviewed_occurrences')} reviewed recurrences "
        f"look like reader-visible slots; recurrence support={support:.2f}, "
        f"shape consistency={shape:.2f}, missing-word score={score:.2f}."
    )


def missing_word_context(symbol: str, example: dict[str, Any]) -> dict[str, Any]:
    context = str(example.get("context") or "")
    assignment = str(example.get("assignment") or "")
    marker = f"⟦{symbol}:{assignment}⟧"
    reread = context.replace(marker, f"⟦{symbol}:<MISSING_WORD?>⟧")
    return {
        "output_index": example.get("output_index"),
        "left_word": example.get("left_word"),
        "right_word": example.get("right_word"),
        "hole_score": example.get("hole_score"),
        "reread": reread,
    }


def score_occurrence_as_hole(*, view: Any, baseline: str, word_set: set[str], context_chars: int) -> dict[str, Any]:
    start = int(view.output_start)
    end = int(view.output_end)
    left_text = baseline[max(0, start - context_chars):start]
    right_text = baseline[end:min(len(baseline), end + context_chars)]
    left = side_word(left_text, word_set, side="left")
    right = side_word(right_text, word_set, side="right")
    local_before = baseline[max(0, start - context_chars):min(len(baseline), end + context_chars)]
    local_after = left_text + right_text
    before_quality = local_quality(local_before, word_set)
    after_quality = local_quality(local_after, word_set)
    removal_delta = after_quality - before_quality
    assignment = str(view.assignment)
    symbol_prior = 0.55 if assignment in {"<null>", "?"} else 0.20
    island_score = left.strength + right.strength
    score = island_score + symbol_prior + max(-0.35, min(0.75, removal_delta))
    if left.known and right.known:
        score += 0.35
    context = left_text[-context_chars:] + f"⟦{view.symbol}:{assignment}⟧" + right_text[:context_chars]
    return {
        "output_index": start,
        "assignment": assignment,
        "left_word": left.word,
        "left_known": left.known,
        "right_word": right.word,
        "right_known": right.known,
        "local_quality_before": round(before_quality, 6),
        "local_quality_after_without_symbol": round(after_quality, 6),
        "removal_quality_delta": round(removal_delta, 6),
        "hole_score": round(score, 6),
        "context": context,
    }


def side_word(text: str, word_set: set[str], *, side: str) -> SideWord:
    result = segment_text(text, word_set)
    if not result.words:
        return SideWord("", False, 0.0)
    word = result.words[-1] if side == "left" else result.words[0]
    known = word in word_set
    strength = word_strength(word, known)
    return SideWord(word, known, strength)


def word_strength(word: str, known: bool) -> float:
    if not word:
        return 0.0
    if not known:
        return 0.15 if len(word) >= 6 else 0.0
    if len(word) == 1:
        return 0.15
    if len(word) == 2:
        return 0.45
    if len(word) <= 4:
        return 0.75
    return 1.0


def local_quality(text: str, word_set: set[str]) -> float:
    result = segment_text(text, word_set)
    if not result.words:
        return 0.0
    known_weight = sum(word_strength(word, word in word_set) for word in result.words)
    pseudo_penalty = sum(0.06 * min(10, len(word)) for word in result.pseudo_words)
    return (known_weight / max(1, len(result.words))) - pseudo_penalty


def token_symbols(cipher_text: Any) -> list[str]:
    return [token for token in cipher_text.raw.replace("|", " ").split() if token]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [float(row["recognition_hit_rate"]) for row in rows if row.get("recognition_hit_rate") is not None]
    hypothesis_rates = [float(row["hypothesis_hit_rate"]) for row in rows if row.get("hypothesis_hit_rate") is not None]
    return {
        "basins": len(rows),
        "mean_true_logogram_recognition_rate": round(sum(rates) / len(rates), 6) if rates else None,
        "mean_true_logogram_hypothesis_rate": round(sum(hypothesis_rates) / len(hypothesis_rates), 6) if hypothesis_rates else None,
        "basins_with_any_true_logogram_recognized": sum(1 for row in rows if row.get("recognized_true_logograms")),
        "basins_with_any_true_logogram_hypothesized": sum(1 for row in rows if row.get("hypothesized_true_logograms")),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Reading-First Missing-Word Hole Probe",
        "",
        "This report starts from damaged plaintext, identifies reader-visible missing-word slots, then attaches those slots back to cipher symbols. Ground-truth logogram columns are post-hoc diagnostics only.",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (payload.get("aggregate") or {}).items():
        rendered = format_percent(value) if key.endswith("rate") else str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.extend([
        "",
        "## Basins",
        "",
        "| Test | Basin | Char | True Logograms | Recognized True | Hypothesized True | Hit Rate | Hyp Rate |",
        "|---|---|---:|---|---|---|---:|---:|",
    ])
    for row in payload.get("rows") or []:
        lines.append(
            "| {test} | {basin} | {char} | {true} | {hits} | {hhits} | {rate} | {hrate} |".format(
                test=row.get("test_id"),
                basin=escape_cell(str(row.get("basin") or "")),
                char=format_percent(row.get("baseline_char")),
                true=escape_cell(symbol_gloss_list(row.get("true_logogram_symbols_present") or [])),
                hits=escape_cell(", ".join(row.get("recognized_true_logograms") or []) or "(none)"),
                hhits=escape_cell(", ".join(row.get("hypothesized_true_logograms") or []) or "(none)"),
                rate=format_percent(row.get("recognition_hit_rate")),
                hrate=format_percent(row.get("hypothesis_hit_rate")),
            )
        )
    for row in payload.get("rows") or []:
        lines.extend(render_row(row))
    return "\n".join(lines) + "\n"


def render_row(row: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"## {row.get('test_id')} — {row.get('basin')}",
        "",
        f"- Artifact: `{row.get('artifact')}`",
        f"- Baseline char: {format_percent(row.get('baseline_char'))}",
        f"- Baseline preview: `{escape_cell(str(row.get('baseline_preview') or ''))}`",
        f"- Segmented preview: `{escape_cell(str(row.get('segmented_preview') or ''))}`",
        f"- Post-hoc true-logogram ranks in recurrence stage: {escape_cell(true_rank_summary(row.get('true_logogram_hypothesis_ranks') or []))}",
        f"- Rereader: {escape_cell(rereader_summary(row.get('rereader_packet') or {}))}",
        "",
        "### Missing-Word Hypotheses",
        "",
        "| Rank | Symbol | Assign | Status | Missing | LLM Conf | Reject Conf | Plain | Occ | Hole Occ | Reason |",
        "|---:|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for rank, item in enumerate(row.get("missing_word_hypotheses") or [], start=1):
        lines.append(
            "| {rank} | {symbol} | {assign} | {status} | {score} | {llm} | {reject} | {plain} | {occ} | {holes} | {reason} |".format(
                rank=rank,
                symbol=item.get("symbol"),
                assign=escape_cell(str(item.get("assignment") or "")),
                status=item.get("status"),
                score=format_number(item.get("missing_word_score")),
                llm=format_number(item.get("llm_confidence")),
                reject=format_number(item.get("llm_reject_confidence")),
                plain=escape_cell(str(item.get("llm_candidate_plaintext") or "")),
                occ=item.get("occurrence_count"),
                holes=item.get("hole_occurrences"),
                reason=escape_cell(str(item.get("reason") or "")),
            )
        )
    lines.extend([
        "",
        "### Hypothesis Rereads",
        "",
        "| Symbol | Left | Right | Reread |",
        "|---|---|---|---|",
    ])
    for item in (row.get("missing_word_hypotheses") or [])[:8]:
        for reread in (item.get("recurrence_reread") or [])[:3]:
            lines.append(
                "| {symbol} | {left} | {right} | `{context}` |".format(
                    symbol=item.get("symbol"),
                    left=escape_cell(str(reread.get("left_word") or "")),
                    right=escape_cell(str(reread.get("right_word") or "")),
                    context=escape_cell(str(reread.get("reread") or "")[:140]),
                )
            )
    lines.extend([
        "",
        "### Missing-Word Symbol Shortlist",
        "",
        "| Rank | Symbol | Assign | Signal | Score | Occ | Hole Occ | Left/Right Examples |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ])
    for rank, item in enumerate(row.get("symbol_rows") or [], start=1):
        examples = []
        for example in item.get("examples") or []:
            examples.append(f"{example.get('left_word')}/{example.get('right_word')}")
        lines.append(
            "| {rank} | {symbol} | {assign} | {signal} | {score} | {occ} | {holes} | {examples} |".format(
                rank=rank,
                symbol=item.get("symbol"),
                assign=escape_cell(str(item.get("assignment") or "")),
                signal=item.get("signal"),
                score=format_number(item.get("hole_score")),
                occ=item.get("occurrence_count"),
                holes=item.get("hole_occurrences"),
                examples=escape_cell(", ".join(examples[:4])),
            )
        )
    lines.extend([
        "",
        "### Top Hole Examples",
        "",
        "| Symbol | Score | Left | Right | Δ Local | Context |",
        "|---|---:|---|---|---:|---|",
    ])
    for item in row.get("symbol_rows") or []:
        for example in (item.get("examples") or [])[:2]:
            lines.append(
                "| {symbol} | {score} | {left} | {right} | {delta} | `{context}` |".format(
                    symbol=item.get("symbol"),
                    score=format_number(example.get("hole_score")),
                    left=escape_cell(str(example.get("left_word") or "")),
                    right=escape_cell(str(example.get("right_word") or "")),
                    delta=format_number(example.get("removal_quality_delta")),
                    context=escape_cell(str(example.get("context") or "")[:140]),
                )
            )
    if row.get("broken_words"):
        lines.extend([
            "",
            "### Broken Word Islands",
            "",
            "| Word | Start | Context |",
            "|---|---:|---|",
        ])
        for item in row.get("broken_words") or []:
            lines.append(
                f"| `{escape_cell(str(item.get('word') or ''))}` | {item.get('start')} | `{escape_cell(str(item.get('context') or '')[:140])}` |"
            )
    return lines


def safe_rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 6)


def format_percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def format_number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def escape_cell(value: str) -> str:
    return value.replace("|", "/").replace("\n", " ")


def symbol_gloss_list(rows: list[dict[str, Any]]) -> str:
    return ", ".join(f"{row.get('symbol')}:{row.get('gloss')}" for row in rows) or "(none)"


def true_rank_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(none)"
    rendered = []
    for row in rows:
        rank = row.get("rank")
        rank_text = f"rank {rank}" if rank is not None else "not reviewed"
        rendered.append(f"{row.get('symbol')}:{row.get('gloss')} ({rank_text})")
    return ", ".join(rendered)


def rereader_summary(packet: dict[str, Any]) -> str:
    if not packet:
        return "local"
    if packet.get("dry_run"):
        return f"llm dry-run {packet.get('provider')}/{packet.get('model')}"
    usage = packet.get("usage") or {}
    cost = usage.get("estimated_cost_usd")
    cost_text = f", estimated cost ${float(cost):.4f}" if cost is not None else ""
    return (
        f"llm {packet.get('provider')}/{packet.get('model')}, "
        f"accepted={packet.get('accepted_hypotheses')}{cost_text}"
    )


if __name__ == "__main__":
    main()
