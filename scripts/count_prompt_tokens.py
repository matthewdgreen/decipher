#!/usr/bin/env python3
"""Report per-component prompt size for any benchmark test.

Reconstructs the exact initial prompt (system + tools + user message) for a
given test without making any API calls. Prints a breakdown table and a
before/after comparison showing the savings from mode-driven tool filtering.

Usage:
    PYTHONPATH=src python scripts/count_prompt_tokens.py \\
        --test-id parity_tool_zenith_zodiac408 \\
        --split ~/Dropbox/src2/cipher_benchmark/benchmark/splits/parity_zodiac.jsonl \\
        --benchmark-root ~/Dropbox/src2/cipher_benchmark/benchmark

Alternatively, pass a raw cipher file instead of a benchmark test:
    PYTHONPATH=src python scripts/count_prompt_tokens.py \\
        --cipher-file path/to/cipher.txt --language en
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _tok(chars: int) -> str:
    """Rough token estimate: 4 chars ≈ 1 token."""
    return f"~{chars // 4:,}"


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "  —"
    return f"{100 * part / total:4.0f}%"


def _tool_schema_chars(tools: list[dict]) -> int:
    return sum(len(json.dumps(t)) for t in tools)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count initial-prompt chars/tokens for a benchmark test (no API call)."
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--test-id", help="Benchmark test ID")
    src_group.add_argument("--cipher-file", help="Raw cipher text file")

    parser.add_argument("--split", help="Benchmark split JSONL file")
    parser.add_argument(
        "--benchmark-root",
        default="~/Dropbox/src2/cipher_benchmark/benchmark",
        help="Benchmark root directory",
    )
    parser.add_argument("--language", "-l", default="en")
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip running the automated preflight (faster, no preflight context)",
    )
    parser.add_argument(
        "--benchmark-context",
        default="max",
        choices=[
            "none", "minimal", "standard", "historical",
            "related_metadata", "related_solutions", "max",
        ],
    )
    args = parser.parse_args()

    # ── load cipher text ────────────────────────────────────────────────────
    if args.test_id:
        from benchmark.loader import BenchmarkLoader
        from benchmark.runner_v2 import _run_automated_preflight
        from benchmark.scorer import has_word_boundaries
        from preprocessing import convert_s_tokens_to_letters, estimate_normalization_benefit
        from benchmark.loader import parse_canonical_transcription, resolve_test_language
        from benchmark.context import build_benchmark_context

        benchmark_root = Path(args.benchmark_root).expanduser()
        loader = BenchmarkLoader(benchmark_root)

        # Find the split file
        split_file: Path | None = None
        if args.split:
            split_file = Path(args.split).expanduser()
        else:
            # try auto-detect from common splits
            for candidate in [
                benchmark_root / "splits" / "parity_zodiac.jsonl",
                benchmark_root / "splits" / "borg_tests.jsonl",
                benchmark_root / "splits" / "copiale_tests.jsonl",
            ]:
                if candidate.exists():
                    # peek to see if test_id is in it
                    for line in candidate.read_text().splitlines():
                        row = json.loads(line)
                        if row.get("test_id") == args.test_id:
                            split_file = candidate
                            break
                    if split_file:
                        break
            if not split_file:
                # Search all splits
                for p in sorted((benchmark_root / "splits").glob("*.jsonl")):
                    for line in p.read_text().splitlines():
                        row = json.loads(line)
                        if row.get("test_id") == args.test_id:
                            split_file = p
                            break
                    if split_file:
                        break

        if not split_file:
            sys.exit(f"Could not find split file for test '{args.test_id}'. "
                     "Pass --split explicitly.")

        tests = loader.load_tests(split_file)
        test = next((t for t in tests if t.test_id == args.test_id), None)
        if test is None:
            sys.exit(f"Test '{args.test_id}' not found in {split_file}.")
        test_data = loader.load_test_data(test)

        lang = resolve_test_language(test_data, args.language if args.language != "en" else None)

        raw = test_data.canonical_transcription
        if estimate_normalization_benefit(raw) == "high":
            converted, _ = convert_s_tokens_to_letters(raw)
            cipher_text = parse_canonical_transcription(converted)
        else:
            cipher_text = parse_canonical_transcription(raw)

        automated_preflight: dict | None = None
        if not args.no_preflight:
            print("Running automated preflight (no LLM)...", end="", flush=True)
            automated_preflight = _run_automated_preflight(
                cipher_text, lang, args.test_id,
                test_data.test.cipher_system or "",
            )
            print(f" {automated_preflight.get('status', '?')}")

        benchmark_context = build_benchmark_context(
            test_data, policy=args.benchmark_context
        )
        benchmark_ctx_prompt = benchmark_context.prompt if benchmark_context else None

    else:
        from benchmark.loader import parse_canonical_transcription
        lang = args.language
        raw = Path(args.cipher_file).expanduser().read_text()
        cipher_text = parse_canonical_transcription(raw)
        automated_preflight = None
        benchmark_ctx_prompt = None

    # ── build prompt components ──────────────────────────────────────────────
    import analysis.cipher_id as cipher_id_analysis
    from agent.prompts_v2 import get_system_prompt, initial_context
    from agent.tools_v2 import TOOL_DEFINITIONS
    from agent.loop_v2 import (
        _active_modes_from_suspicion,
        _apply_mode_filter,
        _mode_filter_note,
        _preflight_context,
    )
    from analysis import ic

    system_prompt = get_system_prompt(lang)

    ic_value = ic.index_of_coincidence(cipher_text.tokens, cipher_text.alphabet.size)
    fingerprint = cipher_id_analysis.compute_cipher_fingerprint(
        cipher_text.tokens,
        cipher_text.alphabet.size,
        language=lang,
        word_group_count=len(cipher_text.words),
    )
    cipher_id_ctx = cipher_id_analysis.format_fingerprint_for_context(fingerprint)

    _MIN_TOKENS_FOR_MODE_FILTER = 30
    active_modes = (
        _active_modes_from_suspicion(fingerprint.suspicion_scores)
        if fingerprint and len(cipher_text.tokens) >= _MIN_TOKENS_FOR_MODE_FILTER
        else set()
    )
    filtered_tools = _apply_mode_filter(TOOL_DEFINITIONS, active_modes)
    filter_note = _mode_filter_note(TOOL_DEFINITIONS, filtered_tools)

    context_parts = [
        p for p in [benchmark_ctx_prompt, _preflight_context(automated_preflight)] if p
    ]
    user_message_after = initial_context(
        cipher_display=cipher_text.raw,
        alphabet_symbols=cipher_text.alphabet.symbols,
        total_tokens=len(cipher_text.tokens),
        total_words=len(cipher_text.words),
        ic_value=ic_value,
        language=lang,
        prior_context="\n\n".join(context_parts) if context_parts else None,
        cipher_id_context=cipher_id_ctx,
        tool_filter_note=filter_note,
    )

    # "before" version: all tools, no filter note
    user_message_before = initial_context(
        cipher_display=cipher_text.raw,
        alphabet_symbols=cipher_text.alphabet.symbols,
        total_tokens=len(cipher_text.tokens),
        total_words=len(cipher_text.words),
        ic_value=ic_value,
        language=lang,
        prior_context="\n\n".join(context_parts) if context_parts else None,
        cipher_id_context=cipher_id_ctx,
        tool_filter_note=None,
    )

    # ── measure ──────────────────────────────────────────────────────────────
    sys_chars = len(system_prompt)
    tools_before_chars = _tool_schema_chars(TOOL_DEFINITIONS)
    tools_after_chars = _tool_schema_chars(filtered_tools)
    user_before_chars = len(user_message_before)
    user_after_chars = len(user_message_after)

    total_before = sys_chars + tools_before_chars + user_before_chars
    total_after = sys_chars + tools_after_chars + user_after_chars

    # Per-section breakdown of user message
    sections: dict[str, str] = {}
    cur_section = "(header)"
    cur_buf: list[str] = []
    for line in user_message_before.splitlines():
        if line.startswith("## "):
            sections[cur_section] = "\n".join(cur_buf)
            cur_section = line.strip()
            cur_buf = [line]
        else:
            cur_buf.append(line)
    sections[cur_section] = "\n".join(cur_buf)

    # ── print report ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  Prompt size report — {args.test_id or args.cipher_file}")
    print(f"  cipher: {len(cipher_text.tokens)} tokens, {cipher_text.alphabet.size} symbols, lang={lang}")
    print(f"  active modes (≥0.15): {sorted(active_modes) or '(none / too short)'}")
    print(f"  tool count: {len(TOOL_DEFINITIONS)} → {len(filtered_tools)}  "
          f"({len(TOOL_DEFINITIONS) - len(filtered_tools)} hidden)")
    print("=" * 72)
    print()
    print(f"{'Component':<42} {'Chars':>8}  {'≈Tokens':>8}  {'%':>5}")
    print("-" * 72)

    def row(label: str, chars: int, total: int) -> None:
        print(f"  {label:<40} {chars:>8,}  {_tok(chars):>8}  {_pct(chars, total)}")

    row("System prompt", sys_chars, total_before)
    row("Tool schemas (ALL tools)", tools_before_chars, total_before)
    row("User message (before filter note)", user_before_chars, total_before)
    print(f"  {'TOTAL (before)':<40} {total_before:>8,}  {_tok(total_before):>8}  100%")
    print()
    row("Tool schemas (after mode filter)", tools_after_chars, total_before)
    row("User message (after filter note)", user_after_chars, total_before)
    saved = total_before - total_after
    print(f"  {'TOTAL (after mode filter)':<40} {total_after:>8,}  {_tok(total_after):>8}  "
          f"{_pct(total_after, total_before)}")
    print(f"  {'  Savings':<40} {saved:>8,}  {_tok(saved):>8}  {_pct(saved, total_before)}")
    print()

    print("User-message section breakdown (before):")
    print("-" * 72)
    for name, text in sorted(sections.items(), key=lambda kv: -len(kv[1])):
        if not text.strip():
            continue
        row(name[:40], len(text), user_before_chars)
    print()

    print("Suspicion scores:")
    print("-" * 72)
    scores = fingerprint.suspicion_scores if fingerprint else {}
    for mode, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        marker = " ✓" if mode in active_modes else "  "
        print(f"  {marker} {mode:<38} {score:.3f}")
    print()

    if args.test_id and args.no_preflight:
        print("(preflight skipped — add automated preflight context with omitting --no-preflight)")

    print("=" * 72)


if __name__ == "__main__":
    main()
