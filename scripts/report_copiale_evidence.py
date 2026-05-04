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
from frontier.suite import load_frontier_suite


DEFAULT_SUITE = REPO_ROOT / "frontier" / "copiale_evidence_packet.jsonl"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT.parent / "cipher_benchmark" / "benchmark"


def diagnose_canonical_transcription(canonical_text: str) -> dict[str, Any]:
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
        "diagnostic_flags": _diagnostic_flags(
            unique_symbols=len(symbol_counts),
            token_count=token_count,
            max_symbol_frequency=max((count / token_count for count in symbol_counts.values()), default=0.0),
            mean_word_length=sum(word_lengths) / len(word_lengths) if word_lengths else 0.0,
            high_frequency_count=len(high_frequency_symbols),
            singleton_ratio=len(rare_symbols) / len(symbol_counts) if symbol_counts else 0.0,
            repeated_short_count=len(short_repeated_words),
        ),
    }


def build_report_rows(
    suite_file: Path,
    benchmark_root: Path,
    summary_jsonl: Path | None = None,
) -> list[dict[str, Any]]:
    loader = BenchmarkLoader(benchmark_root)
    cases = load_frontier_suite(suite_file)
    summary_by_test = _read_latest_summary(summary_jsonl)
    rows = []
    for case in cases:
        canonical_text, records = _load_target_canonical_text(loader, case.test.target_records)
        diag = diagnose_canonical_transcription(canonical_text)
        score_row = summary_by_test.get(case.test.test_id, {})
        rows.append({
            "test_id": case.test.test_id,
            "records": [record.id for record in records],
            "source": records[0].source if records else "",
            "language": _resolve_language(records),
            "cipher_type": sorted({item for record in records for item in record.cipher_type}),
            "diagnostics": diag,
            "summary": score_row,
        })
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Copiale Evidence Report",
        "",
        "This report uses ciphertext/transcription data only. Any score columns are imported from completed run summaries after the solver has produced candidates.",
        "",
        "| Test | Tokens | Words | Symbols | Singletons | High-Freq Symbols | Repeated Cipher Words | Char | Time | Flags |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for row in rows:
        diag = row["diagnostics"]
        summary = row.get("summary") or {}
        lines.append(
            "| {test_id} | {tokens} | {words} | {symbols} | {singletons} | {high_freq} | {repeats} | {char} | {elapsed} | {flags} |".format(
                test_id=row["test_id"],
                tokens=diag["token_count"],
                words=diag["word_count"],
                symbols=diag["unique_symbols"],
                singletons=diag["singleton_symbol_count"],
                high_freq=_format_symbol_list(diag["high_frequency_symbols"], limit=4),
                repeats=_format_word_list(diag["short_repeated_words"], limit=3),
                char=_format_percent(summary.get("char_accuracy")),
                elapsed=_format_seconds(summary.get("elapsed_seconds")),
                flags=", ".join(diag["diagnostic_flags"]) or "none",
            )
        )

    lines.extend(["", "## Per-Page Detail", ""])
    for row in rows:
        diag = row["diagnostics"]
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
            f"- Diagnostic flags: {', '.join(diag['diagnostic_flags']) or 'none'}",
            "",
        ])
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
        "--format",
        choices=["markdown", "json"],
        default="markdown",
    )
    args = parser.parse_args()

    rows = build_report_rows(
        suite_file=Path(args.suite_file),
        benchmark_root=Path(args.benchmark_root),
        summary_jsonl=Path(args.summary_jsonl) if args.summary_jsonl else None,
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
    return flags


def _format_symbol_list(items: list[dict[str, Any]], limit: int) -> str:
    parts = []
    for item in items[:limit]:
        parts.append(f"{item['symbol']}:{item['count']}({item['frequency'] * 100:.1f}%)")
    return ", ".join(parts)


def _format_word_list(items: list[dict[str, Any]], limit: int) -> str:
    return ", ".join(f"{item['word']}x{item['count']}" for item in items[:limit])


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
