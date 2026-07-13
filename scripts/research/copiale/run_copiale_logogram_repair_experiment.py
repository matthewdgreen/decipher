#!/usr/bin/env python3
"""Run logogram/codeword probes over strong Copiale finalist basins.

This is a post-hoc diagnostic harness, not a solver.  It starts from already
generated Copiale finalist basins, runs the ground-truth-free logogram
hypothesis machinery over each basin, then reports two calibration questions:

1. Does any candidate expansion improve ordinary character accuracy?
2. Did the machinery at least identify true Copiale logogram symbols as
   plausible missing/codeword symbols?

Ground truth is used only to select "best" basins from prior artifacts and to
grade the completed hypotheses.
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

from benchmark.loader import BenchmarkLoader, parse_canonical_transcription, resolve_test_language  # noqa: E402
from benchmark.scorer import DiagnosticPlaintextUnit, score_decryption, score_decryption_diagnostic  # noqa: E402
from probe_logogram_hypotheses import (  # noqa: E402
    DEFAULT_CANDIDATES,
    candidate_expansion_reviews,
    explicit_candidate_words,
    generate_llm_hypotheses,
    generate_local_hypotheses,
    render_token_views,
    render_with_expansion,
    reread_occurrences,
    suspicious_symbol_groups,
    symbol_context_packet,
    evaluate_hypotheses,
    scalar_language_quality,
)
from analysis.language_scoring import language_quality_feature_dict  # noqa: E402


@dataclass(frozen=True)
class BasinCandidate:
    test_id: str
    artifact: Path
    run_label: str
    source: str
    rank_in_artifact: int
    selected: bool
    mask: tuple[str, ...]
    key: dict[int, int]
    decryption: str
    char_accuracy: float
    candidate_id: str

    @property
    def label(self) -> str:
        mask = ",".join(self.mask) if self.mask else "(none)"
        selected = "selected" if self.selected else f"rank{self.rank_in_artifact}"
        return f"{self.run_label}:{selected}:{self.source}:{mask}"


@dataclass(frozen=True)
class BasinHint:
    test_id: str
    run_label: str
    source: str
    mask: tuple[str, ...]
    char_accuracy: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument(
        "--summary-json",
        default="artifacts/copiale_breadth_experiment/four_page_wide/summary.json",
        help="Breadth experiment summary containing artifact paths.",
    )
    parser.add_argument("--artifact", action="append", default=[], help="Additional artifact JSON to inspect.")
    parser.add_argument("--test-id", action="append", default=[], help="Restrict to one or more test ids.")
    parser.add_argument("--top-basins-per-test", type=int, default=3)
    parser.add_argument("--mode", choices=["local", "llm"], default="local")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--candidate-words", default="")
    parser.add_argument("--max-symbols", type=int, default=18)
    parser.add_argument("--context-chars", type=int, default=44)
    parser.add_argument("--recurrence-limit", type=int, default=8)
    parser.add_argument("--min-occurrences", type=int, default=1)
    parser.add_argument("--include-one-letter-symbols", action="store_true")
    parser.add_argument(
        "--one-letter-hypothesis-symbols",
        type=int,
        default=0,
        help=(
            "Also generate local logogram expansion hypotheses for this many "
            "one-letter symbol contexts. Default 0 keeps the older missing/null-only behavior."
        ),
    )
    parser.add_argument(
        "--symbol-ranker",
        choices=("legacy", "suspicion"),
        default="suspicion",
        help="Rank one-letter collapsed-logogram suspects by transparent language/context features.",
    )
    parser.add_argument("--llm-candidate-review-top-words", type=int, default=10)
    parser.add_argument("--llm-candidate-review-recurrences", type=int, default=6)
    parser.add_argument("--top-n-hypotheses", type=int, default=12)
    parser.add_argument(
        "--recognition-top-n",
        type=int,
        default=12,
        help="How many recognized suspect symbols to carry into the expansion stage.",
    )
    parser.add_argument(
        "--expand-signals",
        default="one_letter_symbol",
        help=(
            "Comma-separated recognition signals to expand. Defaults to "
            "one_letter_symbol so null/missing-symbol review does not swamp collapsed logograms."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="artifacts/copiale_logogram_repair_experiment",
    )
    args = parser.parse_args()

    started = time.monotonic()
    benchmark_root = resolve_path(Path(args.benchmark_root))
    output_dir = resolve_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = BenchmarkLoader(benchmark_root)
    test_rows = {test.test_id: test for test in loader.load_tests(args.split)}
    true_logograms = load_true_logogram_symbols(benchmark_root)
    artifact_paths = artifact_paths_from_args(args)
    basin_hints = basin_hints_from_summary(args)
    basins = select_basins(
        artifact_paths=artifact_paths,
        loader=loader,
        test_rows=test_rows,
        test_filter=set(args.test_id),
        top_per_test=args.top_basins_per_test,
        hints=basin_hints,
    )
    if not basins:
        raise SystemExit("No finalist basins found. Check --summary-json/--artifact paths.")

    rows = []
    for index, basin in enumerate(basins, start=1):
        print(f"[{index}/{len(basins)}] {basin.test_id} {basin.label}", flush=True)
        row = evaluate_basin(
            basin=basin,
            loader=loader,
            test_rows=test_rows,
            true_logogram_symbols=true_logograms,
            args=args,
        )
        rows.append(row)

    payload = {
        "experiment": "copiale_logogram_repair_experiment",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "benchmark_root": str(benchmark_root),
        "split": args.split,
        "mode": args.mode,
        "settings": {
            "top_basins_per_test": args.top_basins_per_test,
            "max_symbols": args.max_symbols,
            "include_one_letter_symbols": args.include_one_letter_symbols,
            "one_letter_hypothesis_symbols": args.one_letter_hypothesis_symbols,
            "symbol_ranker": args.symbol_ranker,
            "min_occurrences": args.min_occurrences,
            "top_n_hypotheses": args.top_n_hypotheses,
            "recognition_top_n": args.recognition_top_n,
            "expand_signals": args.expand_signals,
        },
        "true_logogram_symbols": true_logograms,
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def artifact_paths_from_args(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.summary_json:
        summary_path = resolve_path(Path(args.summary_json))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary.get("rows") or []:
            for raw in row.get("artifacts") or []:
                paths.append(resolve_path(Path(raw)))
    for raw in args.artifact:
        paths.append(resolve_path(Path(raw)))
    deduped: list[Path] = []
    seen = set()
    for path in paths:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def basin_hints_from_summary(args: argparse.Namespace) -> dict[str, list[BasinHint]]:
    """Load prior breadth-summary basin hints.

    This avoids rescoring hundreds of finalist strings just to recover the
    already-reported post-hoc best basins.  If no summary exists, the selector
    falls back to scoring a small selected-candidate set.
    """
    if not args.summary_json:
        return {}
    summary_path = resolve_path(Path(args.summary_json))
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[BasinHint]] = {}
    for row in summary.get("rows") or []:
        test_id = str(row.get("test_id") or "")
        if not test_id:
            continue
        add_hint(
            grouped,
            test_id=test_id,
            run_label=str(row.get("best_char_run_label") or ""),
            source=str(row.get("best_char_source") or ""),
            mask=tuple(str(item) for item in (row.get("best_char_mask") or [])),
            char_accuracy=float(row.get("best_char_accuracy") or 0.0),
        )
        add_hint(
            grouped,
            test_id=test_id,
            run_label=str(row.get("lq_pick_run_label") or ""),
            source=str(row.get("lq_pick_source") or ""),
            mask=tuple(str(item) for item in (row.get("lq_pick_mask") or [])),
            char_accuracy=float(row.get("lq_pick_char_accuracy") or 0.0),
        )
    return grouped


def add_hint(
    grouped: dict[str, list[BasinHint]],
    *,
    test_id: str,
    run_label: str,
    source: str,
    mask: tuple[str, ...],
    char_accuracy: float,
) -> None:
    if not run_label and not source and not mask:
        return
    hint = BasinHint(
        test_id=test_id,
        run_label=run_label,
        source=source,
        mask=mask,
        char_accuracy=char_accuracy,
    )
    rows = grouped.setdefault(test_id, [])
    if hint not in rows:
        rows.append(hint)


def load_true_logogram_symbols(benchmark_root: Path) -> dict[str, str]:
    path = benchmark_root / "sources" / "copiale" / "metadata" / "copiale_symbol_map.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbol_map = payload.get("symbol_map") or payload.get("mapping") or {}
    glossary = payload.get("logogram_glossary") or {}
    rows: dict[str, str] = {}
    for mnemonic, gloss in glossary.items():
        symbol = symbol_map.get(mnemonic)
        if symbol:
            rows[str(symbol)] = str(gloss)
    return rows


def select_basins(
    *,
    artifact_paths: list[Path],
    loader: BenchmarkLoader,
    test_rows: dict[str, Any],
    test_filter: set[str],
    top_per_test: int,
    hints: dict[str, list[BasinHint]],
) -> list[BasinCandidate]:
    grouped: dict[str, list[BasinCandidate]] = {}
    for artifact_path in artifact_paths:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        test_id = str(artifact.get("test_id") or artifact.get("cipher_id") or "")
        if not test_id or (test_filter and test_id not in test_filter):
            continue
        if test_id not in test_rows:
            continue
        wanted = hints.get(test_id) or []
        test_data = None if wanted else loader.load_test_data(test_rows[test_id])
        for candidate in artifact_finalists(artifact):
            hint = matching_hint(candidate, wanted)
            if wanted and hint is None:
                continue
            decryption = str(candidate.get("decryption") or candidate.get("plaintext") or "")
            key = parse_candidate_key(candidate)
            if not decryption or not key:
                continue
            if hint is not None:
                char_accuracy = hint.char_accuracy
            else:
                assert test_data is not None
                score = score_decryption(test_id, decryption, test_data.plaintext, 0.0, "completed")
                char_accuracy = float(score.char_accuracy)
            row = BasinCandidate(
                test_id=test_id,
                artifact=artifact_path,
                run_label=str(candidate.get("_run_label") or artifact_path.parent.name),
                source=str(candidate.get("source") or ""),
                rank_in_artifact=int(candidate.get("_rank_in_artifact") or 0),
                selected=bool(candidate.get("_selected")),
                mask=tuple(str(item) for item in (candidate.get("mask") or [])),
                key=key,
                decryption=decryption,
                char_accuracy=char_accuracy,
                candidate_id=str(candidate.get("candidate_id") or ""),
            )
            grouped.setdefault(test_id, []).append(row)

    selected: list[BasinCandidate] = []
    for test_id, rows in grouped.items():
        rows.sort(key=lambda row: (row.char_accuracy, row.selected), reverse=True)
        selected.extend(rows[: max(1, top_per_test)])
    selected.sort(key=lambda row: (row.test_id, -row.char_accuracy, row.label))
    return selected


def matching_hint(candidate: dict[str, Any], hints: list[BasinHint]) -> BasinHint | None:
    if not hints:
        return None
    run_label = str(candidate.get("_run_label") or "")
    source = str(candidate.get("source") or "")
    mask = tuple(str(item) for item in (candidate.get("mask") or []))
    for hint in hints:
        if hint.run_label and run_label != hint.run_label:
            continue
        if hint.source and source != hint.source:
            continue
        if mask != hint.mask:
            continue
        return hint
    return None


def artifact_finalists(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    null_step = next(
        (step for step in artifact.get("steps") or [] if isinstance(step, dict) and step.get("name") == "search_null_masks"),
        None,
    )
    if not isinstance(null_step, dict):
        return []
    run_label = str(null_step.get("ranker") or "")
    rows: list[dict[str, Any]] = []
    selected = null_step.get("selected")
    if isinstance(selected, dict):
        item = dict(selected)
        item["_rank_in_artifact"] = 1
        item["_selected"] = True
        item["_run_label"] = run_label
        rows.append(item)
    for rank, candidate in enumerate(null_step.get("top_finalists") or [], start=1):
        if not isinstance(candidate, dict):
            continue
        item = dict(candidate)
        item["_rank_in_artifact"] = rank
        item["_selected"] = False
        item["_run_label"] = run_label
        rows.append(item)
    return rows


def parse_candidate_key(candidate: dict[str, Any]) -> dict[int, int]:
    parsed: dict[int, int] = {}
    raw_key = candidate.get("key") or {}
    if not isinstance(raw_key, dict):
        return parsed
    for raw_k, raw_v in raw_key.items():
        try:
            parsed[int(raw_k)] = int(raw_v)
        except (TypeError, ValueError):
            continue
    return parsed


def evaluate_basin(
    *,
    basin: BasinCandidate,
    loader: BenchmarkLoader,
    test_rows: dict[str, Any],
    true_logogram_symbols: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    test_data = loader.load_test_data(test_rows[basin.test_id])
    language = resolve_test_language(test_data, "de")
    cipher_text = parse_canonical_transcription(test_data.canonical_transcription)
    views, baseline = render_token_views(cipher_text, key=basin.key, mask=basin.mask)
    raw_groups = suspicious_symbol_groups(
        views,
        include_one_letter_symbols=args.include_one_letter_symbols,
        min_occurrences=args.min_occurrences,
        max_symbols=max(args.max_symbols, args.one_letter_hypothesis_symbols, 1) * 3
        if args.symbol_ranker == "suspicion"
        else args.max_symbols,
    )
    ranked_groups, symbol_rank_rows = rank_symbol_groups(
        raw_groups,
        views=views,
        baseline=baseline,
        language=language,
        max_symbols=args.max_symbols,
        enabled=args.symbol_ranker == "suspicion",
    )
    groups = ranked_groups
    contexts = [
        symbol_context_packet(
            symbol=symbol,
            views=symbol_views,
            baseline=baseline,
            context_chars=args.context_chars,
            recurrence_limit=args.recurrence_limit,
        )
        for symbol, symbol_views in groups
    ]
    candidate_words = explicit_candidate_words(args.candidate_words) or DEFAULT_CANDIDATES.get(language, DEFAULT_CANDIDATES["de"])
    expansion_contexts = contexts_for_expansion(
        contexts=contexts,
        symbol_rank_rows=symbol_rank_rows,
        top_n=args.recognition_top_n,
        signals=parse_signal_set(args.expand_signals),
    )
    if args.mode == "llm":
        candidate_reviews = candidate_expansion_reviews(
            contexts=expansion_contexts,
            candidate_words=candidate_words,
            cipher_text=cipher_text,
            key=basin.key,
            mask=basin.mask,
            context_chars=args.context_chars,
            top_words=args.llm_candidate_review_top_words,
            recurrence_limit=args.llm_candidate_review_recurrences,
        )
        hypotheses, llm_packet = generate_llm_hypotheses(
            contexts=contexts,
            candidate_reviews=candidate_reviews,
            language=language,
            candidate_words=candidate_words,
            args=args,
        )
    else:
        hypotheses = generate_local_hypotheses(
            contexts=[
                context for context in expansion_contexts
                if context.get("signal") == "missing_or_unmapped_symbol"
            ],
            candidate_words=candidate_words,
        )
        hypotheses.extend(
            generate_one_letter_hypotheses(
                contexts=expansion_contexts,
                candidate_words=candidate_words,
                max_symbols=args.recognition_top_n,
            )
        )
        llm_packet = {}
    evaluated = evaluate_hypotheses(
        hypotheses,
        cipher_text=cipher_text,
        key=basin.key,
        mask=basin.mask,
        baseline=baseline,
        ground_truth=test_data.plaintext,
        language=language,
        context_chars=args.context_chars,
        recurrence_limit=args.recurrence_limit,
    )
    evaluated_with_char = [
        row for row in evaluated
        if isinstance(row.get("post_hoc_char"), int | float)
    ]
    best_expansion = max(
        evaluated_with_char,
        key=lambda row: float(row.get("post_hoc_char") or 0.0),
        default=None,
    )
    candidate_symbols = [str(row.get("symbol")) for row in contexts]
    recognized_symbols = [str(row.get("symbol")) for row in symbol_rank_rows[: args.recognition_top_n]]
    expanded_symbols = [str(row.get("symbol")) for row in expansion_contexts]
    true_present = sorted(set(candidate_symbols_in_cipher(cipher_text)) & set(true_logogram_symbols))
    candidate_hits = sorted(set(candidate_symbols) & set(true_present))
    recognized_hits = sorted(set(recognized_symbols) & set(true_present))
    expanded_hits = sorted(set(expanded_symbols) & set(true_present))
    hypothesis_symbols = [str(row.get("symbol")) for row in evaluated[: args.top_n_hypotheses]]
    all_hypothesis_symbols = [str(row.get("symbol")) for row in evaluated]
    hypothesis_hits = sorted(set(hypothesis_symbols) & set(true_present))
    any_hypothesis_hits = sorted(set(all_hypothesis_symbols) & set(true_present))
    diagnostic = diagnostic_for_context_symbols(
        views=views,
        contexts=contexts,
        ground_truth=test_data.plaintext,
        test_id=basin.test_id,
    )
    return {
        "test_id": basin.test_id,
        "artifact": str(basin.artifact),
        "basin": basin.label,
        "candidate_id": basin.candidate_id,
        "baseline_char": round(basin.char_accuracy, 6),
        "baseline_preview": baseline[:180],
        "mask": list(basin.mask),
        "candidate_symbol_count": len(contexts),
        "candidate_symbols": candidate_symbols,
        "symbol_ranker": args.symbol_ranker,
        "symbol_ranker_top": symbol_rank_rows[: max(args.top_n_hypotheses, 12)],
        "recognition_top_n": args.recognition_top_n,
        "recognized_symbols": recognized_symbols,
        "expanded_symbols": expanded_symbols,
        "true_logogram_symbols_present": [
            {"symbol": symbol, "gloss": true_logogram_symbols[symbol]}
            for symbol in true_present
        ],
        "true_logogram_candidate_hits": candidate_hits,
        "true_logogram_recognition_hits_top_n": recognized_hits,
        "true_logogram_expansion_stage_hits": expanded_hits,
        "true_logogram_hypothesis_hits_top_n": hypothesis_hits,
        "true_logogram_hypothesis_hits_any": any_hypothesis_hits,
        "candidate_hit_rate": safe_rate(len(candidate_hits), len(true_present)),
        "recognition_hit_rate_top_n": safe_rate(len(recognized_hits), len(true_present)),
        "expansion_stage_hit_rate": safe_rate(len(expanded_hits), len(true_present)),
        "hypothesis_hit_rate_top_n": safe_rate(len(hypothesis_hits), len(true_present)),
        "hypothesis_hit_rate_any": safe_rate(len(any_hypothesis_hits), len(true_present)),
        "best_expansion": compact_hypothesis(best_expansion),
        "best_expansion_char_delta": (
            round(float(best_expansion.get("post_hoc_char")) - basin.char_accuracy, 6)
            if best_expansion and isinstance(best_expansion.get("post_hoc_char"), int | float)
            else None
        ),
        "top_hypotheses": [compact_hypothesis(row) for row in evaluated[: args.top_n_hypotheses]],
        "diagnostic_unknown_spans": [
            {
                "symbol": span.symbol,
                "kind": span.kind,
                "ground_truth_text": span.ground_truth_text,
                "context": span.context_before + "⟦" + span.ground_truth_text + "⟧" + span.context_after,
            }
            for span in diagnostic.unknown_spans
            if span.ground_truth_text
        ][: args.top_n_hypotheses],
        "llm_packet": compact_llm_packet(llm_packet),
    }


def candidate_symbols_in_cipher(cipher_text: Any) -> list[str]:
    return [token for token in cipher_text.raw.replace("|", " ").split() if token]


def parse_signal_set(raw: str) -> set[str]:
    return {
        item.strip()
        for item in raw.split(",")
        if item.strip()
    }


def contexts_for_expansion(
    *,
    contexts: list[dict[str, Any]],
    symbol_rank_rows: list[dict[str, Any]],
    top_n: int,
    signals: set[str],
) -> list[dict[str, Any]]:
    by_symbol = {str(context.get("symbol")): context for context in contexts}
    selected = []
    for row in symbol_rank_rows:
        symbol = str(row.get("symbol") or "")
        signal = str(row.get("signal") or "")
        if signal not in signals:
            continue
        context = by_symbol.get(symbol)
        if context is not None:
            selected.append(context)
        if len(selected) >= max(0, top_n):
            break
    return selected


def rank_symbol_groups(
    groups: list[tuple[str, list[Any]]],
    *,
    views: list[Any],
    baseline: str,
    language: str,
    max_symbols: int,
    enabled: bool,
) -> tuple[list[tuple[str, list[Any]]], list[dict[str, Any]]]:
    if not enabled:
        return groups[: max(0, max_symbols)], []
    base_lq = scalar_language_quality(language_quality_feature_dict(baseline, language=language))
    ranked: list[tuple[float, tuple[str, list[Any]], dict[str, Any]]] = []
    for symbol, symbol_views in groups:
        if not symbol_views:
            continue
        assignment = str(symbol_views[0].assignment)
        occurrence_count = len(symbol_views)
        signal = "missing_or_unmapped_symbol" if assignment in {"<null>", "?"} else "one_letter_symbol"
        omitted = render_without_symbol(views, symbol)
        omitted_lq = scalar_language_quality(language_quality_feature_dict(omitted, language=language))
        omission_delta = omitted_lq - base_lq
        dispersion = occurrence_dispersion(symbol_views, max(1, len(baseline)))
        local_break = local_boundary_break_score(symbol_views, baseline)
        if signal == "missing_or_unmapped_symbol":
            score = (
                4.0
                + min(2.0, occurrence_count / 8.0)
                + max(-1.0, min(1.5, omission_delta))
                + local_break * 0.25
            )
        else:
            # Collapsed logograms usually masquerade as ordinary high-confidence
            # letters.  Ordinary homophones can occur dozens of times, so use a
            # moderate-frequency band instead of raw recurrence.
            omission_term = max(-1.5, min(2.5, omission_delta * 2.0))
            occurrence_term = collapsed_logogram_occurrence_prior(occurrence_count)
            score = (
                occurrence_term * 1.35
                + omission_term
                + local_break * 0.55
                + dispersion * 0.85
            )
        features = {
            "symbol": symbol,
            "assignment": assignment,
            "signal": signal,
            "score": round(score, 6),
            "occurrence_count": occurrence_count,
            "omission_lq_delta": round(omission_delta, 6),
            "local_break_score": round(local_break, 6),
            "dispersion": round(dispersion, 6),
        }
        ranked.append((score, (symbol, symbol_views), features))
    ranked.sort(
        key=lambda item: (
            item[2]["signal"] == "missing_or_unmapped_symbol",
            item[0],
            item[2]["occurrence_count"],
        ),
        reverse=True,
    )
    return [item[1] for item in ranked[: max(0, max_symbols)]], [item[2] for item in ranked]


def render_without_symbol(views: list[Any], symbol: str) -> str:
    return "".join(view.rendered for view in views if view.symbol != symbol and view.rendered)


def occurrence_dispersion(symbol_views: list[Any], baseline_len: int) -> float:
    if len(symbol_views) <= 1 or baseline_len <= 0:
        return 0.0
    positions = [max(0, min(baseline_len, int(view.output_start))) / baseline_len for view in symbol_views]
    return max(0.0, min(1.0, max(positions) - min(positions)))


def collapsed_logogram_occurrence_prior(occurrence_count: int) -> float:
    """Prior for a one-letter symbol that may really be a whole-word codeword.

    Too few occurrences give little recurrence evidence; too many are more
    likely to be ordinary homophones for common letters.  The sweet spot is a
    moderate number spread across the page.
    """
    if occurrence_count <= 0:
        return 0.0
    if occurrence_count <= 8:
        return occurrence_count / 8.0
    if occurrence_count <= 14:
        return 1.0
    return max(0.0, 1.0 - ((occurrence_count - 14) / 18.0))


def local_boundary_break_score(symbol_views: list[Any], baseline: str) -> float:
    """Approximate how often a symbol sits in a visibly damaged local island."""
    if not symbol_views:
        return 0.0
    total = 0.0
    sample = symbol_views[:24]
    vowels = set("AEIOU")
    for view in sample:
        left = baseline[max(0, view.output_start - 5):view.output_start]
        right = baseline[view.output_end:min(len(baseline), view.output_end + 5)]
        island = left + (view.rendered or "") + right
        if len(island) >= 7:
            vowel_rate = sum(1 for ch in island if ch in vowels) / max(1, len(island))
            if vowel_rate < 0.20 or vowel_rate > 0.62:
                total += 0.45
        if has_long_run(island):
            total += 0.30
        if view.rendered and left and right:
            tri = left[-1] + view.rendered + right[0]
            if tri in {"SSS", "NNN", "EEE", "TTT", "RRR"}:
                total += 0.35
    return min(1.0, total / max(1, len(sample)))


def has_long_run(text: str) -> bool:
    if not text:
        return False
    run = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            run += 1
            if run >= 4:
                return True
        else:
            prev = ch
            run = 1
    return False


def generate_one_letter_hypotheses(
    *,
    contexts: list[dict[str, Any]],
    candidate_words: list[str],
    max_symbols: int,
) -> list[dict[str, Any]]:
    if max_symbols <= 0:
        return []
    rows: list[dict[str, Any]] = []
    one_letter_contexts = [
        context for context in contexts
        if context.get("signal") == "one_letter_symbol"
    ][:max_symbols]
    for context in one_letter_contexts:
        occurrences = int(context.get("occurrence_count") or 0)
        recurrence_confidence = min(0.22, occurrences / 30.0 * 0.22)
        for rank, word in enumerate(candidate_words, start=1):
            candidate_prior = min(0.12, 0.12 / max(1.0, rank ** 0.35))
            rows.append({
                "symbol": context["symbol"],
                "kind": "logogram",
                "plaintext": word,
                "confidence": round(min(0.72, 0.28 + recurrence_confidence + candidate_prior), 3),
                "candidate_rank": rank,
                "status": "candidate",
                "reason": (
                    "Local collapsed-logogram sweep over a one-letter symbol; "
                    f"{occurrences} recurrence(s) make this a possible codeword/logogram candidate."
                ),
            })
    return rows


def diagnostic_for_context_symbols(
    *,
    views: list[Any],
    contexts: list[dict[str, Any]],
    ground_truth: str,
    test_id: str,
) -> Any:
    symbols = {str(row.get("symbol")) for row in contexts}
    units: list[DiagnosticPlaintextUnit] = []
    last = 0
    # TokenView output_start/output_end are offsets in the rendered baseline.
    for view in views:
        if view.output_start > last:
            # This case is covered by previous rendered token text appends; keep
            # the guard for defensive clarity.
            last = view.output_start
        if view.symbol in symbols and view.assignment in {"<null>", "?"}:
            units.append(DiagnosticPlaintextUnit("possible_logogram", symbol=view.symbol))
        elif view.rendered:
            units.append(DiagnosticPlaintextUnit("text", view.rendered))
            last = view.output_end
        elif view.assignment == "<null>":
            units.append(DiagnosticPlaintextUnit("null", symbol=view.symbol))
    return score_decryption_diagnostic(test_id, units, ground_truth)


def compact_hypothesis(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "symbol": row.get("symbol"),
        "kind": row.get("kind"),
        "plaintext": row.get("plaintext"),
        "confidence": row.get("confidence"),
        "occurrence_count": row.get("occurrence_count"),
        "language_quality_delta": row.get("language_quality_delta"),
        "post_hoc_char": row.get("post_hoc_char"),
        "reason": row.get("reason"),
        "preview": str(row.get("preview") or "")[:160],
    }


def compact_llm_packet(packet: dict[str, Any]) -> dict[str, Any]:
    if not packet:
        return {}
    return {
        "provider": packet.get("provider"),
        "model": packet.get("model"),
        "usage": packet.get("usage"),
        "estimated_cost": packet.get("estimated_cost"),
    }


def safe_rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 6)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    deltas = [
        float(row["best_expansion_char_delta"])
        for row in rows
        if row.get("best_expansion_char_delta") is not None
    ]
    candidate_rates = [
        float(row["candidate_hit_rate"])
        for row in rows
        if row.get("candidate_hit_rate") is not None
    ]
    hypothesis_rates = [
        float(row["hypothesis_hit_rate_top_n"])
        for row in rows
        if row.get("hypothesis_hit_rate_top_n") is not None
    ]
    any_hypothesis_rates = [
        float(row["hypothesis_hit_rate_any"])
        for row in rows
        if row.get("hypothesis_hit_rate_any") is not None
    ]
    recognition_rates = [
        float(row["recognition_hit_rate_top_n"])
        for row in rows
        if row.get("recognition_hit_rate_top_n") is not None
    ]
    expansion_rates = [
        float(row["expansion_stage_hit_rate"])
        for row in rows
        if row.get("expansion_stage_hit_rate") is not None
    ]
    return {
        "basins": len(rows),
        "basins_with_positive_char_delta_count": sum(1 for delta in deltas if delta > 0),
        "positive_char_delta_rate": round(sum(1 for delta in deltas if delta > 0) / len(deltas), 6) if deltas else None,
        "best_char_delta": round(max(deltas), 6) if deltas else None,
        "mean_char_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
        "mean_candidate_logogram_hit_rate": round(sum(candidate_rates) / len(candidate_rates), 6) if candidate_rates else None,
        "mean_recognition_logogram_hit_rate": round(sum(recognition_rates) / len(recognition_rates), 6) if recognition_rates else None,
        "mean_expansion_stage_logogram_hit_rate": round(sum(expansion_rates) / len(expansion_rates), 6) if expansion_rates else None,
        "mean_top_hypothesis_logogram_hit_rate": round(sum(hypothesis_rates) / len(hypothesis_rates), 6) if hypothesis_rates else None,
        "mean_any_hypothesis_logogram_hit_rate": round(sum(any_hypothesis_rates) / len(any_hypothesis_rates), 6) if any_hypothesis_rates else None,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Copiale Logogram Repair Experiment",
        "",
        "This is post-hoc diagnostics. Ground truth is used to select strong prior basins and grade outcomes; hypothesis generation/ranking is ground-truth-free.",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (payload.get("aggregate") or {}).items():
        if key.endswith("rate") or key.endswith("delta"):
            rendered = format_percent(value)
        else:
            rendered = str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.extend([
        "",
        "## Basins",
        "",
        "| Test | Basin | Base Char | Best Expansion | Δ Char | True Logograms Present | Candidate Hits | Recognition Hits | Expansion Hits | Top Hypothesis Hits | Any Hypothesis Hits |",
        "|---|---|---:|---|---:|---|---|---|---|---|---|",
    ])
    for row in payload.get("rows") or []:
        best = row.get("best_expansion") or {}
        best_label = ""
        if best:
            best_label = f"{best.get('symbol')} → {best.get('plaintext')} ({format_percent(best.get('post_hoc_char'))})"
        lines.append(
            "| {test} | {basin} | {base} | {best} | {delta} | {present} | {hits} | {rhits} | {ehits} | {hhits} | {anyhits} |".format(
                test=row.get("test_id"),
                basin=escape_cell(str(row.get("basin") or "")),
                base=format_percent(row.get("baseline_char")),
                best=escape_cell(best_label),
                delta=format_percent(row.get("best_expansion_char_delta"), signed=True),
                present=escape_cell(symbol_list(row.get("true_logogram_symbols_present") or [])),
                hits=escape_cell(", ".join(row.get("true_logogram_candidate_hits") or []) or "(none)"),
                rhits=escape_cell(", ".join(row.get("true_logogram_recognition_hits_top_n") or []) or "(none)"),
                ehits=escape_cell(", ".join(row.get("true_logogram_expansion_stage_hits") or []) or "(none)"),
                hhits=escape_cell(", ".join(row.get("true_logogram_hypothesis_hits_top_n") or []) or "(none)"),
                anyhits=escape_cell(", ".join(row.get("true_logogram_hypothesis_hits_any") or []) or "(none)"),
            )
        )
    for row in payload.get("rows") or []:
        lines.extend(render_row_details(row))
    return "\n".join(lines) + "\n"


def render_row_details(row: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"## {row.get('test_id')} — {row.get('basin')}",
        "",
        f"- Artifact: `{row.get('artifact')}`",
        f"- Baseline char: {format_percent(row.get('baseline_char'))}",
        f"- Candidate symbols: {', '.join(row.get('candidate_symbols') or []) or '(none)'}",
        f"- Recognition shortlist: {', '.join(row.get('recognized_symbols') or []) or '(none)'}",
        f"- Expansion-stage symbols: {', '.join(row.get('expanded_symbols') or []) or '(none)'}",
        f"- Baseline preview: `{escape_cell(str(row.get('baseline_preview') or ''))}`",
        "",
        "### Symbol Shortlist",
        "",
        "| Rank | Symbol | Assignment | Signal | Score | Occ | Omit LQ Δ | Dispersion | Local Break |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(row.get("symbol_ranker_top") or [], start=1):
        lines.append(
            "| {rank} | {symbol} | {assignment} | {signal} | {score} | {occ} | {omit} | {disp} | {break_score} |".format(
                rank=rank,
                symbol=item.get("symbol") or "",
                assignment=escape_cell(str(item.get("assignment") or "")),
                signal=item.get("signal") or "",
                score=format_number(item.get("score")),
                occ=item.get("occurrence_count", ""),
                omit=format_number(item.get("omission_lq_delta")),
                disp=format_number(item.get("dispersion")),
                break_score=format_number(item.get("local_break_score")),
            )
        )
    lines.extend([
        "",
        "### Top Hypotheses",
        "",
        "| Rank | Symbol | Kind | Plaintext | Post-Hoc | Δ LQ | Reason |",
        "|---:|---|---|---|---:|---:|---|",
    ])
    base = float(row.get("baseline_char") or 0.0)
    for rank, hyp in enumerate(row.get("top_hypotheses") or [], start=1):
        lines.append(
            "| {rank} | {symbol} | {kind} | {plain} | {char} | {lq} | {reason} |".format(
                rank=rank,
                symbol=hyp.get("symbol") or "",
                kind=hyp.get("kind") or "",
                plain=hyp.get("plaintext") or "",
                char=format_percent(hyp.get("post_hoc_char")),
                lq=format_number(hyp.get("language_quality_delta")),
                reason=escape_cell(str(hyp.get("reason") or "")[:120]),
            )
        )
    if row.get("diagnostic_unknown_spans"):
        lines.extend([
            "",
            "### Diagnostic Unknown Spans",
            "",
            "| Symbol | Inferred GT Span | Context |",
            "|---|---|---|",
        ])
        for span in row.get("diagnostic_unknown_spans") or []:
            lines.append(
                f"| {span.get('symbol')} | `{escape_cell(str(span.get('ground_truth_text') or ''))}` | `{escape_cell(str(span.get('context') or '')[:140])}` |"
            )
    return lines


def symbol_list(rows: list[dict[str, Any]]) -> str:
    return ", ".join(f"{row.get('symbol')}:{row.get('gloss')}" for row in rows) or "(none)"


def format_percent(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return ""
    number = float(value)
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number * 100:.1f}%"


def format_number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def escape_cell(value: str) -> str:
    return value.replace("|", "/").replace("\n", " ")


if __name__ == "__main__":
    main()
