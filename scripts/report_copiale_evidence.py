#!/usr/bin/env python3
"""Report cipher-side diagnostics for the compact Copiale evidence packet."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.loader import BenchmarkLoader, BenchmarkRecord, parse_canonical_transcription
from benchmark.scorer import align_char_sequences, normalize_text
from frontier.suite import load_frontier_suite


DEFAULT_SUITE = REPO_ROOT / "frontier" / "copiale_evidence_packet.jsonl"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT.parent / "cipher_benchmark" / "benchmark"


def diagnose_canonical_transcription(
    canonical_text: str,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ground-truth-free diagnostics for a canonical S-token text."""
    cipher = parse_canonical_transcription(canonical_text)
    symbols = [cipher.alphabet.decode([token]) for token in cipher.tokens]
    symbol_counts = Counter(symbols)
    token_count = len(symbols)
    word_lengths = [len(word) for word in cipher.words]
    cipher_words = [
        " ".join(cipher.alphabet.decode([token]) for token in word)
        for word in cipher.words
        if word
    ]
    word_counts = Counter(cipher_words)

    high_frequency_symbols = [
        {
            "symbol": symbol,
            "count": count,
            "frequency": count / token_count if token_count else 0.0,
        }
        for symbol, count in symbol_counts.most_common()
        if token_count and count / token_count >= 0.06
    ]
    rare_symbols = [symbol for symbol, count in symbol_counts.items() if count == 1]
    repeated_cipher_words = [
        {"word": word, "count": count}
        for word, count in word_counts.most_common(8)
        if count > 1
    ]
    short_repeated_words = [
        item
        for item in repeated_cipher_words
        if len(item["word"].split()) <= 3
    ][:5]
    repeated_symbol_ngrams = _repeated_symbol_ngrams(symbols)
    solver_mapping = _solver_mapping(cipher, artifact)
    homophone_families = _homophone_families(symbol_counts, solver_mapping)
    null_candidates = _null_candidates(
        symbols=symbols,
        words=cipher.words,
        alphabet=cipher.alphabet,
        symbol_counts=symbol_counts,
        solver_mapping=solver_mapping,
        artifact=artifact,
    )

    return {
        "token_count": token_count,
        "word_count": len(cipher.words),
        "unique_symbols": len(symbol_counts),
        "overcomplete_ratio_vs_26": len(symbol_counts) / 26.0,
        "mean_word_length": sum(word_lengths) / len(word_lengths) if word_lengths else 0.0,
        "max_word_length": max(word_lengths) if word_lengths else 0,
        "singleton_symbol_count": len(rare_symbols),
        "singleton_symbol_ratio": len(rare_symbols) / len(symbol_counts) if symbol_counts else 0.0,
        "top_symbols": [
            {
                "symbol": symbol,
                "count": count,
                "frequency": count / token_count if token_count else 0.0,
            }
            for symbol, count in symbol_counts.most_common(10)
        ],
        "high_frequency_symbols": high_frequency_symbols,
        "rare_symbol_examples": sorted(rare_symbols)[:12],
        "repeated_cipher_words": repeated_cipher_words,
        "short_repeated_words": short_repeated_words,
        "repeated_symbol_ngrams": repeated_symbol_ngrams,
        "homophone_families": homophone_families,
        "null_codeword_candidates": null_candidates,
        "diagnostic_flags": _diagnostic_flags(
            unique_symbols=len(symbol_counts),
            token_count=token_count,
            max_symbol_frequency=max((count / token_count for count in symbol_counts.values()), default=0.0),
            mean_word_length=sum(word_lengths) / len(word_lengths) if word_lengths else 0.0,
            high_frequency_count=len(high_frequency_symbols),
            singleton_ratio=len(rare_symbols) / len(symbol_counts) if symbol_counts else 0.0,
            repeated_short_count=len(short_repeated_words),
            null_candidate_count=len(null_candidates),
        ),
    }


def build_report_rows(
    suite_file: Path,
    benchmark_root: Path,
    summary_jsonl: Path | None = None,
    ground_truth_calibration: bool = False,
) -> list[dict[str, Any]]:
    loader = BenchmarkLoader(benchmark_root)
    cases = load_frontier_suite(suite_file)
    summary_by_test = _read_latest_summary(summary_jsonl)
    rows = []
    for case in cases:
        canonical_text, records = _load_target_canonical_text(loader, case.test.target_records)
        score_row = summary_by_test.get(case.test.test_id, {})
        artifact = _read_artifact(score_row)
        diag = diagnose_canonical_transcription(canonical_text, artifact=artifact)
        calibration = None
        if ground_truth_calibration and artifact:
            plaintext = _load_target_plaintext(loader, case.test.target_records)
            calibration = calibrate_against_ground_truth(
                canonical_text=canonical_text,
                artifact=artifact,
                ground_truth=plaintext,
            )
        rows.append({
            "test_id": case.test.test_id,
            "records": [record.id for record in records],
            "source": records[0].source if records else "",
            "language": _resolve_language(records),
            "cipher_type": sorted({item for record in records for item in record.cipher_type}),
            "diagnostics": diag,
            "summary": score_row,
            "ground_truth_calibration": calibration,
        })
    return rows


def calibrate_against_ground_truth(
    canonical_text: str,
    artifact: dict[str, Any],
    ground_truth: str,
) -> dict[str, Any]:
    """Post-hoc insertion/null calibration using ground truth.

    This is intentionally not used by runtime tools. It exists to tune
    hypotheses after a solver candidate already exists.
    """
    cipher = parse_canonical_transcription(canonical_text)
    decryption = str(artifact.get("decryption") or "")
    decoded_chars = "".join(ch for ch in decryption.upper() if "A" <= ch <= "Z")
    ground_truth_chars = normalize_text(ground_truth).replace(" ", "")
    alignment = align_char_sequences(decoded_chars, ground_truth_chars)
    insertions_by_symbol: Counter[str] = Counter()
    substitutions_by_symbol: Counter[str] = Counter()
    matches_by_symbol: Counter[str] = Counter()
    deletes = 0
    for row in alignment:
        if row.decoded_index is None:
            deletes += 1
            continue
        if row.decoded_index >= len(cipher.tokens):
            continue
        symbol = cipher.alphabet.decode([cipher.tokens[row.decoded_index]])
        if row.op == "insert":
            insertions_by_symbol[symbol] += 1
        elif row.op == "substitute":
            substitutions_by_symbol[symbol] += 1
        elif row.op == "match":
            matches_by_symbol[symbol] += 1

    symbol_counts = Counter(cipher.alphabet.decode([token]) for token in cipher.tokens)
    rows = []
    for symbol, count in symbol_counts.items():
        inserts = insertions_by_symbol[symbol]
        subs = substitutions_by_symbol[symbol]
        matches = matches_by_symbol[symbol]
        if inserts == 0 and subs == 0:
            continue
        rows.append({
            "symbol": symbol,
            "count": count,
            "insertions": inserts,
            "substitutions": subs,
            "matches": matches,
            "insertion_rate": inserts / count if count else 0.0,
            "error_rate": (inserts + subs) / count if count else 0.0,
        })
    rows.sort(key=lambda item: (-item["insertions"], -item["insertion_rate"], item["symbol"]))
    return {
        "mode": "post_hoc_ground_truth_only",
        "decoded_length": len(decoded_chars),
        "ground_truth_length": len(ground_truth_chars),
        "length_gap_decoded_minus_truth": len(decoded_chars) - len(ground_truth_chars),
        "alignment_insertions": sum(insertions_by_symbol.values()),
        "alignment_deletions": deletes,
        "top_insertion_symbols": rows[:12],
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Copiale Evidence Report",
        "",
        "This report uses ciphertext/transcription data only. Any score columns are imported from completed run summaries after the solver has produced candidates.",
        "",
        "| Test | Tokens | Words | Symbols | Singletons | Null/Code Candidates | Homophone Families | Char | Time | Flags |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for row in rows:
        diag = row["diagnostics"]
        summary = row.get("summary") or {}
        lines.append(
            "| {test_id} | {tokens} | {words} | {symbols} | {singletons} | {nulls} | {families} | {char} | {elapsed} | {flags} |".format(
                test_id=row["test_id"],
                tokens=diag["token_count"],
                words=diag["word_count"],
                symbols=diag["unique_symbols"],
                singletons=diag["singleton_symbol_count"],
                nulls=_format_candidate_list(diag["null_codeword_candidates"], limit=3),
                families=_format_family_list(diag["homophone_families"], limit=3),
                char=_format_percent(summary.get("char_accuracy")),
                elapsed=_format_seconds(summary.get("elapsed_seconds")),
                flags=", ".join(diag["diagnostic_flags"]) or "none",
            )
        )

    lines.extend(["", "## Per-Page Detail", ""])
    for row in rows:
        diag = row["diagnostics"]
        calibration = row.get("ground_truth_calibration")
        lines.extend([
            f"### {row['test_id']}",
            "",
            f"- Records: {', '.join(row['records'])}",
            f"- Language: {row['language'] or 'unknown'}",
            f"- Cipher type metadata: {', '.join(row['cipher_type']) or 'none'}",
            f"- Overcomplete ratio vs 26 letters: {diag['overcomplete_ratio_vs_26']:.2f}",
            f"- Mean/max cipher-word length: {diag['mean_word_length']:.1f}/{diag['max_word_length']}",
            f"- Top symbols: {_format_symbol_list(diag['top_symbols'], limit=10)}",
            f"- Rare symbol examples: {', '.join(diag['rare_symbol_examples']) or 'none'}",
            f"- Repeated cipher words: {_format_word_list(diag['repeated_cipher_words'], limit=8) or 'none'}",
            f"- Repeated symbol ngrams: {_format_ngram_list(diag['repeated_symbol_ngrams'], limit=8) or 'none'}",
            f"- Largest solver homophone families: {_format_family_list(diag['homophone_families'], limit=8) or 'none'}",
            f"- Null/codeword candidates: {_format_candidate_list(diag['null_codeword_candidates'], limit=8) or 'none'}",
            f"- Diagnostic flags: {', '.join(diag['diagnostic_flags']) or 'none'}",
        ])
        if calibration:
            lines.extend([
                f"- Ground-truth calibration mode: {calibration['mode']}",
                f"- Decoded/truth length gap: {calibration['decoded_length']} - {calibration['ground_truth_length']} = {calibration['length_gap_decoded_minus_truth']}",
                f"- Alignment insertions/deletions: {calibration['alignment_insertions']}/{calibration['alignment_deletions']}",
                f"- Top insertion-heavy symbols: {_format_insertion_symbol_list(calibration['top_insertion_symbols'], limit=8) or 'none'}",
            ])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report ground-truth-free diagnostics for the Copiale evidence packet.",
    )
    parser.add_argument("--suite-file", default=str(DEFAULT_SUITE))
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument(
        "--summary-jsonl",
        help="Optional run_frontier_suite summary JSONL to attach post-hoc scores.",
    )
    parser.add_argument(
        "--ground-truth-calibration",
        action="store_true",
        help=(
            "Attach post-hoc insertion/null calibration using benchmark plaintext. "
            "This is for analysis only and must not be used by runtime solvers."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
    )
    args = parser.parse_args()

    rows = build_report_rows(
        suite_file=Path(args.suite_file),
        benchmark_root=Path(args.benchmark_root),
        summary_jsonl=Path(args.summary_jsonl) if args.summary_jsonl else None,
        ground_truth_calibration=args.ground_truth_calibration,
    )
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(render_markdown(rows), end="")


def _load_target_canonical_text(
    loader: BenchmarkLoader,
    record_ids: list[str],
) -> tuple[str, list[BenchmarkRecord]]:
    canonical_parts = []
    records = []
    for record_id in record_ids:
        record = loader.get_record(record_id)
        if record is None:
            raise ValueError(f"record not found in benchmark manifest: {record_id}")
        records.append(record)
        if not record.transcription_canonical_file:
            continue
        path = loader.root / record.transcription_canonical_file
        if not path.exists():
            raise FileNotFoundError(f"canonical transcription not found: {path}")
        canonical_parts.append(path.read_text(encoding="utf-8").strip())
    return "\n".join(canonical_parts), records


def _load_target_plaintext(loader: BenchmarkLoader, record_ids: list[str]) -> str:
    plaintext_parts = []
    for record_id in record_ids:
        record = loader.get_record(record_id)
        if record is None:
            raise ValueError(f"record not found in benchmark manifest: {record_id}")
        if not record.plaintext_file:
            continue
        path = loader.root / record.plaintext_file
        if not path.exists():
            raise FileNotFoundError(f"plaintext not found: {path}")
        plaintext_parts.append(path.read_text(encoding="utf-8").strip())
    return "\n".join(plaintext_parts)


def _read_latest_summary(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    by_test: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        test_id = str(row.get("test_id") or "")
        if test_id:
            by_test[test_id] = row
    return by_test


def _read_artifact(summary_row: dict[str, Any]) -> dict[str, Any] | None:
    raw_path = str(summary_row.get("artifact_path") or "")
    if not raw_path:
        return None
    candidates = [Path(raw_path)]
    if not Path(raw_path).is_absolute():
        candidates.append(REPO_ROOT / raw_path)
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _solver_mapping(
    cipher,
    artifact: dict[str, Any] | None,
) -> dict[str, str]:
    if not artifact:
        return {}
    key = artifact.get("key")
    if not isinstance(key, dict):
        return {}
    mapping = {}
    for token in range(len(cipher.alphabet.symbols)):
        value = key.get(str(token), key.get(token))
        if value is None:
            continue
        try:
            plain_idx = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= plain_idx < 26:
            mapping[cipher.alphabet.decode([token])] = chr(ord("A") + plain_idx)
    return mapping


def _homophone_families(
    symbol_counts: Counter[str],
    solver_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    if not solver_mapping:
        return []
    families: dict[str, list[str]] = {}
    for symbol, letter in solver_mapping.items():
        families.setdefault(letter, []).append(symbol)
    rows = []
    for letter, symbols in families.items():
        symbols_sorted = sorted(symbols, key=lambda item: (-symbol_counts[item], item))
        rows.append({
            "letter": letter,
            "symbol_count": len(symbols_sorted),
            "token_count": sum(symbol_counts[item] for item in symbols_sorted),
            "symbols": symbols_sorted[:8],
        })
    return sorted(rows, key=lambda item: (-item["token_count"], -item["symbol_count"], item["letter"]))


def _null_candidates(
    symbols: list[str],
    words: list[list[int]],
    alphabet,
    symbol_counts: Counter[str],
    solver_mapping: dict[str, str],
    artifact: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not symbols:
        return []
    word_index_by_pos = []
    for word_index, word in enumerate(words):
        word_index_by_pos.extend([word_index] * len(word))

    left_neighbors: dict[str, set[str]] = {symbol: set() for symbol in symbol_counts}
    right_neighbors: dict[str, set[str]] = {symbol: set() for symbol in symbol_counts}
    word_spread: dict[str, set[int]] = {symbol: set() for symbol in symbol_counts}
    for idx, symbol in enumerate(symbols):
        if idx > 0:
            left_neighbors[symbol].add(symbols[idx - 1])
        if idx + 1 < len(symbols):
            right_neighbors[symbol].add(symbols[idx + 1])
        if idx < len(word_index_by_pos):
            word_spread[symbol].add(word_index_by_pos[idx])

    mapped_letter_counts: Counter[str] = Counter()
    for symbol, count in symbol_counts.items():
        letter = solver_mapping.get(symbol)
        if letter:
            mapped_letter_counts[letter] += count
    top_letter, top_letter_count = ("", 0)
    if mapped_letter_counts:
        top_letter, top_letter_count = mapped_letter_counts.most_common(1)[0]
    top_letter_fraction = top_letter_count / len(symbols) if symbols else 0.0
    quality = _artifact_quality(artifact)
    collapsed = bool(quality.get("collapsed")) or top_letter_fraction >= 0.18

    rows = []
    for symbol, count in symbol_counts.items():
        frequency = count / len(symbols)
        rare_score = max(0.0, min(1.0, (8 - count) / 7.0))
        spread = len(word_spread[symbol])
        localization_score = 1.0 - min(1.0, spread / max(1, len(words)))
        neighbor_slots = max(1, min(count, 12) * 2)
        neighbor_diversity = (len(left_neighbors[symbol]) + len(right_neighbors[symbol])) / neighbor_slots
        context_specific_score = 1.0 - min(1.0, neighbor_diversity)
        mapped_letter = solver_mapping.get(symbol, "")
        collapse_score = 1.0 if collapsed and mapped_letter and mapped_letter == top_letter else 0.0
        score = (
            0.38 * rare_score
            + 0.25 * localization_score
            + 0.22 * context_specific_score
            + 0.15 * collapse_score
        )
        reasons = []
        if count <= 3:
            reasons.append("rare")
        if localization_score >= 0.65:
            reasons.append("localized")
        if context_specific_score >= 0.55:
            reasons.append("context-specific")
        if collapse_score:
            reasons.append(f"maps-to-collapsed-{top_letter}")
        if count > 12 and not collapse_score:
            score *= 0.55
            reasons.append("frequent-symbol-lower-priority")
        if score < 0.35 and count > 4:
            continue
        rows.append({
            "symbol": symbol,
            "count": count,
            "frequency": frequency,
            "mapped_letter": mapped_letter,
            "score": round(score, 3),
            "word_spread": spread,
            "left_neighbors": len(left_neighbors[symbol]),
            "right_neighbors": len(right_neighbors[symbol]),
            "reasons": reasons or ["weak-candidate"],
        })
    return sorted(rows, key=lambda item: (-item["score"], item["count"], item["symbol"]))[:12]


def _artifact_quality(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not artifact:
        return {}
    for step in reversed(artifact.get("steps") or []):
        quality = step.get("quality")
        if isinstance(quality, dict):
            return quality
    return {}


def _repeated_symbol_ngrams(symbols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for size in range(2, 6):
        counts = Counter(
            " ".join(symbols[index:index + size])
            for index in range(0, max(0, len(symbols) - size + 1))
        )
        for ngram, count in counts.most_common(8):
            if count > 1:
                rows.append({"ngram": ngram, "count": count, "size": size})
    return sorted(rows, key=lambda item: (-item["count"], item["size"], item["ngram"]))[:12]


def _resolve_language(records: list[BenchmarkRecord]) -> str:
    languages = sorted({record.plaintext_language for record in records if record.plaintext_language})
    return ",".join(languages)


def _diagnostic_flags(
    unique_symbols: int,
    token_count: int,
    max_symbol_frequency: float,
    mean_word_length: float,
    high_frequency_count: int,
    singleton_ratio: float,
    repeated_short_count: int,
    null_candidate_count: int,
) -> list[str]:
    flags = []
    if unique_symbols > 40:
        flags.append("overcomplete_symbol_inventory")
    if max_symbol_frequency < 0.05:
        flags.append("flat_homophonic_distribution")
    if mean_word_length > 12:
        flags.append("coarse_or_missing_word_boundaries")
    if token_count < 250:
        flags.append("short_page")
    if high_frequency_count >= 3:
        flags.append("possible_null_or_function_symbols")
    if singleton_ratio >= 0.18:
        flags.append("many_rare_symbols")
    if repeated_short_count >= 2:
        flags.append("repeated_short_codeword_candidates")
    if null_candidate_count >= 5:
        flags.append("null_or_codeword_candidates_present")
    return flags


def _format_symbol_list(items: list[dict[str, Any]], limit: int) -> str:
    parts = []
    for item in items[:limit]:
        parts.append(f"{item['symbol']}:{item['count']}({item['frequency'] * 100:.1f}%)")
    return ", ".join(parts)


def _format_word_list(items: list[dict[str, Any]], limit: int) -> str:
    return ", ".join(f"{item['word']}x{item['count']}" for item in items[:limit])


def _format_ngram_list(items: list[dict[str, Any]], limit: int) -> str:
    return ", ".join(f"{item['ngram']}x{item['count']}" for item in items[:limit])


def _format_family_list(items: list[dict[str, Any]], limit: int) -> str:
    parts = []
    for item in items[:limit]:
        symbols = ",".join(item["symbols"][:4])
        parts.append(f"{item['letter']}:{item['token_count']}/{item['symbol_count']}[{symbols}]")
    return ", ".join(parts)


def _format_candidate_list(items: list[dict[str, Any]], limit: int) -> str:
    parts = []
    for item in items[:limit]:
        mapped = f"->{item['mapped_letter']}" if item.get("mapped_letter") else ""
        reasons = "+".join(item.get("reasons") or [])
        parts.append(f"{item['symbol']}{mapped}:{item['score']:.2f}({reasons})")
    return ", ".join(parts)


def _format_insertion_symbol_list(items: list[dict[str, Any]], limit: int) -> str:
    parts = []
    for item in items[:limit]:
        parts.append(
            f"{item['symbol']}:ins{item['insertions']}/{item['count']}"
            f"({item['insertion_rate'] * 100:.0f}%,sub{item['substitutions']})"
        )
    return ", ".join(parts)


def _format_percent(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value) * 100:.1f}%"


def _format_seconds(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.1f}s"


if __name__ == "__main__":
    main()
