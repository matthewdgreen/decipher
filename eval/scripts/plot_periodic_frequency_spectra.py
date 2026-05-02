#!/usr/bin/env python3
"""Visualize per-phase frequency spectra for periodic/keyed Vigenere text.

This is a diagnostic script, not a solver. It groups ciphertext symbols by
position modulo a candidate period, computes raw per-phase letter frequencies,
then optionally applies a supplied/estimated shift and plots the shifted
plaintext-letter spectra next to the raw spectra.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont

from analysis.polyalphabetic import (
    AZ_ALPHABET,
    cipher_values_from_text,
    keyed_alphabet_from_keyword,
    normalize_keyed_alphabet,
    parse_periodic_key,
    _initial_keyed_shifts,
    _keyed_cipher_indices,
)
from benchmark.loader import BenchmarkLoader, parse_canonical_transcription


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _load_test_data(benchmark_root: Path, split: str, test_id: str):
    loader = BenchmarkLoader(benchmark_root)
    tests = loader.load_tests(split)
    match = next((test for test in tests if test.test_id == test_id), None)
    if match is None:
        raise SystemExit(f"test_id not found in {split}: {test_id}")
    return loader.load_test_data(match)


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _counts_to_freqs(values: list[int]) -> list[float]:
    counts = Counter(values)
    total = max(1, len(values))
    return [counts[i] / total for i in range(26)]


def _top_letters(freqs: list[float], alphabet: str = LETTERS, n: int = 6) -> str:
    ranked = sorted(range(26), key=lambda idx: freqs[idx], reverse=True)[:n]
    return " ".join(f"{alphabet[idx]}:{freqs[idx] * 100:4.1f}" for idx in ranked)


def _shifted_plain_values(stream: list[int], shift: int, keyed_alphabet: str) -> list[int]:
    return [
        ord(keyed_alphabet[(value - shift) % 26]) - ord("A")
        for value in stream
    ]


def _heat_color(value: float, max_value: float) -> tuple[int, int, int]:
    if max_value <= 0:
        return (245, 245, 245)
    intensity = min(1.0, value / max_value)
    # White -> blue-black, with enough contrast to see weak spectra.
    r = int(250 - 210 * intensity)
    g = int(250 - 175 * intensity)
    b = int(250 - 70 * intensity)
    return (r, g, b)


def _draw_heatmap(
    *,
    raw_freqs: list[list[float]],
    shifted_freqs: list[list[float]],
    shifts: list[int],
    keyed_alphabet: str,
    output: Path,
    title: str,
) -> None:
    period = len(raw_freqs)
    cell = 22
    row_h = 24
    left = 96
    top = 86
    gap = 66
    label_h = 26
    width = left + 26 * cell * 2 + gap + 28
    height = top + label_h + period * row_h + 64
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(16)
    font = _font(11)
    small = _font(10)

    draw.text((18, 16), title, fill=(20, 20, 20), font=title_font)
    draw.text((left, 52), "Raw per-phase ciphertext values", fill=(20, 20, 20), font=font)
    shifted_x = left + 26 * cell + gap
    draw.text((shifted_x, 52), "After phase shift: plaintext-letter spectra", fill=(20, 20, 20), font=font)

    raw_max = max((max(row) for row in raw_freqs), default=0.0)
    shifted_max = max((max(row) for row in shifted_freqs), default=0.0)
    for block_x, alphabet, max_value in [
        (left, keyed_alphabet, raw_max),
        (shifted_x, LETTERS, shifted_max),
    ]:
        for col, letter in enumerate(alphabet):
            draw.text((block_x + col * cell + 7, top - 22), letter, fill=(40, 40, 40), font=small)

    for phase in range(period):
        y = top + phase * row_h
        draw.text((18, y + 4), f"phase {phase:02d}", fill=(30, 30, 30), font=font)
        draw.text((70, y + 4), keyed_alphabet[shifts[phase] % 26], fill=(30, 30, 30), font=font)
        for col, value in enumerate(raw_freqs[phase]):
            x = left + col * cell
            draw.rectangle(
                (x, y, x + cell - 2, y + row_h - 3),
                fill=_heat_color(value, raw_max),
                outline=(225, 225, 225),
            )
        for col, value in enumerate(shifted_freqs[phase]):
            x = shifted_x + col * cell
            draw.rectangle(
                (x, y, x + cell - 2, y + row_h - 3),
                fill=_heat_color(value, shifted_max),
                outline=(225, 225, 225),
            )

    draw.text(
        (18, height - 42),
        "Left row label includes the phase and its shift letter. Darker cells are more frequent.",
        fill=(55, 55, 55),
        font=small,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot periodic per-phase frequency spectra.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("../cipher_benchmark/benchmark"))
    parser.add_argument("--split", default="kryptos_tests.jsonl")
    parser.add_argument("--test-id", default="kryptos_k2_keyed_vigenere")
    parser.add_argument("--period", type=int, default=None)
    parser.add_argument("--key", default=None, help="Periodic key letters/shifts. If omitted, estimate shifts by chi-square.")
    parser.add_argument("--keyed-alphabet", default=None)
    parser.add_argument("--alphabet-keyword", default=None)
    parser.add_argument(
        "--use-solver-hints",
        action="store_true",
        help="Use benchmark solver_hints key/tableau if available. This is solution-bearing.",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("eval/artifacts/frequency_spectra/k2_period8.png"))
    args = parser.parse_args()

    data = _load_test_data(args.benchmark_root, args.split, args.test_id)
    hints: dict[str, Any] = data.solver_hints or {}
    key = args.key
    keyed_alphabet = args.keyed_alphabet
    alphabet_keyword = args.alphabet_keyword
    if args.use_solver_hints:
        key = key or hints.get("periodic_key")
        keyed_alphabet = keyed_alphabet or hints.get("keyed_alphabet")
        alphabet_keyword = alphabet_keyword or hints.get("alphabet_keyword")
    if keyed_alphabet:
        keyed_alphabet = normalize_keyed_alphabet(keyed_alphabet=keyed_alphabet)
    elif alphabet_keyword:
        keyed_alphabet = keyed_alphabet_from_keyword(alphabet_keyword)
    else:
        keyed_alphabet = AZ_ALPHABET

    cipher_text = parse_canonical_transcription(data.canonical_transcription)
    if keyed_alphabet == AZ_ALPHABET:
        values, skipped = cipher_values_from_text(cipher_text)
    else:
        values, _positions, skipped = _keyed_cipher_indices(cipher_text, keyed_alphabet)
    period = args.period or (len([ch for ch in key or "" if "A" <= ch.upper() <= "Z"]) if key else 0)
    if period <= 0:
        raise SystemExit("--period is required when --key is not supplied")
    if period > len(values):
        raise SystemExit(f"period {period} is longer than token count {len(values)}")

    if key:
        shifts = parse_periodic_key(key=key, variant="vigenere")
        if len(shifts) != period:
            raise SystemExit(f"key length {len(shifts)} does not match period {period}")
        if keyed_alphabet != AZ_ALPHABET:
            index = {ch: i for i, ch in enumerate(keyed_alphabet)}
            shifts = [index[ch] for ch in key.upper() if "A" <= ch <= "Z"]
    else:
        shifts = _initial_keyed_shifts(values, period, keyed_alphabet)

    raw_freqs: list[list[float]] = []
    shifted_freqs: list[list[float]] = []
    rows: list[dict[str, Any]] = []
    for phase in range(period):
        stream = values[phase::period]
        shifted = _shifted_plain_values(stream, shifts[phase], keyed_alphabet)
        raw = _counts_to_freqs(stream)
        plain = _counts_to_freqs(shifted)
        raw_freqs.append(raw)
        shifted_freqs.append(plain)
        rows.append({
            "phase": phase,
            "length": len(stream),
            "shift": shifts[phase],
            "shift_letter": keyed_alphabet[shifts[phase] % 26],
            "raw_top": _top_letters(raw, keyed_alphabet),
            "shifted_top": _top_letters(plain, LETTERS),
        })

    print(f"{args.test_id}: period={period} keyed_alphabet={keyed_alphabet}")
    print("phase len shift raw-top                         shifted/plain-top")
    for row in rows:
        print(
            f"{row['phase']:>5} {row['length']:>3} "
            f"{row['shift_letter']:>5} "
            f"{row['raw_top']:<31} {row['shifted_top']}"
        )

    title = f"{args.test_id} periodic spectra, period {period}"
    if args.use_solver_hints:
        title += " (solver hints)"
    _draw_heatmap(
        raw_freqs=raw_freqs,
        shifted_freqs=shifted_freqs,
        shifts=shifts,
        keyed_alphabet=keyed_alphabet,
        output=args.output,
        title=title,
    )
    print(f"Wrote {args.output}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "test_id": args.test_id,
            "period": period,
            "keyed_alphabet": keyed_alphabet,
            "key": key,
            "solution_bearing": bool(args.use_solver_hints),
            "skipped_symbol_count": len(skipped),
            "skipped_symbols": skipped[:20],
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
