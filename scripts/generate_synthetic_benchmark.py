#!/usr/bin/env python3
"""Generate synthetic substitution-cipher benchmarks in multiple languages.

Downloads public-domain texts from Project Gutenberg, slices them into
segments of varying length, applies random A-Z substitution keys, and writes
the result into the benchmark directory layout understood by BenchmarkLoader.

Two tiers are generated per language:
  Tier 1 — word boundaries preserved (| separators).
  Tier 2 — no word boundaries (one continuous letter stream).

Usage:
    # Generate all supported languages
    python scripts/generate_synthetic_benchmark.py \\
        --output ~/Dropbox/src2/cipher_benchmark/benchmark

    # Single language
    python scripts/generate_synthetic_benchmark.py \\
        --output ~/Dropbox/src2/cipher_benchmark/benchmark --lang fr

Run tier 1 for a language:
    .venv/bin/decipher benchmark <root> \\
        --split <lang>_ss_synth_tests.jsonl --v2 --language <lang>

Run tier 2 (no word boundaries):
    .venv/bin/decipher benchmark <root> \\
        --split <lang>_ss_synth_nb_tests.jsonl --v2 --language <lang>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import textwrap
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-language source texts (Project Gutenberg, public domain).
# Each entry is a list so multiple texts can pool words.
# ---------------------------------------------------------------------------
LANG_SOURCES: dict[str, list[dict]] = {
    "en": [
        {
            "title": "Pride and Prejudice",
            "author": "Jane Austen",
            "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
            "start_marker": "It is a truth universally acknowledged",
            "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        },
        {
            "title": "A Tale of Two Cities",
            "author": "Charles Dickens",
            "url": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
            "start_marker": "It was the best of times",
            "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        },
        {
            "title": "The Adventures of Sherlock Holmes",
            "author": "Arthur Conan Doyle",
            "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
            "start_marker": "To Sherlock Holmes she is always",
            "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        },
    ],
    "fr": [
        {
            "title": "Les Misérables",
            "author": "Victor Hugo",
            "url": "https://www.gutenberg.org/cache/epub/17489/pg17489.txt",
            "start_marker": "En 1815, M. Charles-François-Bienvenu Myriel",
            "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        },
        {
            "title": "Madame Bovary",
            "author": "Gustave Flaubert",
            "url": "https://www.gutenberg.org/cache/epub/14155/pg14155.txt",
            "start_marker": "Nous étions à l'Étude, quand le Proviseur entra",
            "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        },
    ],
    "de": [
        {
            "title": "Die Verwandlung",
            "author": "Franz Kafka",
            "url": "https://www.gutenberg.org/cache/epub/22367/pg22367.txt",
            "start_marker": "Als Gregor Samsa eines Morgens",
            "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        },
        {
            "title": "Buddenbrooks",
            "author": "Thomas Mann",
            "url": "https://www.gutenberg.org/cache/epub/34811/pg34811.txt",
            "start_marker": "Was ist das",
            "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        },
    ],
    "it": [
        {
            "title": "I Promessi Sposi",
            "author": "Alessandro Manzoni",
            "url": "https://www.gutenberg.org/cache/epub/45334/pg45334.txt",
            "start_marker": "L'historia si può veramente deffinire",
            "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        },
        {
            "title": "Le avventure di Pinocchio",
            "author": "Carlo Collodi",
            "url": "https://www.gutenberg.org/cache/epub/52484/pg52484.txt",
            "start_marker": "C'era una volta un pezzo di legno.",
            "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        },
    ],
}

LANG_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
}

SEGMENT_LENGTHS = (
    [40, 45, 50, 55, 60] * 2
    + [80, 90, 100, 110, 120] * 2
    + [160, 180, 200, 240, 300] * 2
)


def notes_for(lang: str, sources: list[dict]) -> str:
    lang_name = LANG_NAMES.get(lang, lang.upper())
    source_lines = "\n".join(
        f"- {s['title']} — {s['author']}" for s in sources
    )
    pfx = f"{lang}_ss_synth"
    return textwrap.dedent(f"""\
        # {lang_name} Synthetic Cipher Benchmark — Notes

        Generated by `scripts/generate_synthetic_benchmark.py`.

        ## Sources (public domain, Project Gutenberg)
        {source_lines}

        ## Tiers

        ### Tier 1 — Word boundaries preserved (`{pfx}`)
        Split file: `splits/{pfx}_tests.jsonl`
        - 30 tests, IDs `{pfx}_001` … `{pfx}_030`
        - Ciphertext: single cipher letters separated by spaces,
          words delimited by ` | `.

        ### Tier 2 — No word boundaries (`{pfx}_nb`)
        Split file: `splits/{pfx}_nb_tests.jsonl`
        - 30 tests, IDs `{pfx}_nb_001` … `{pfx}_nb_030`
        - Same plaintext segments as Tier 1, independent random keys.
        - Ciphertext: one continuous stream of cipher letters (no ` | `).
        - Plaintext: letters only, no spaces.

        ## Notes
        Accented characters (é, è, à, ü, ö, etc.) are normalised to their
        base ASCII letters before encipherment, so the cipher alphabet is
        always A–Z.

        Generated with seed=42.
    """)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def fetch_gutenberg(src: dict) -> str:
    url, start, end = src["url"], src["start_marker"], src["end_marker"]
    print(f"    Fetching {src['title']} ({url}) …")
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode("utf-8", errors="replace")
    s = raw.find(start)
    e = raw.find(end)
    if s == -1:
        print(f"    WARNING: start marker not found, using full text")
        s = 0
    return raw[s:e] if e != -1 else raw[s:]


# Transliteration table: common accented chars → ASCII
_ACCENT_MAP = str.maketrans(
    "ÀÁÂÃÄÅàáâãäåÈÉÊËèéêëÌÍÎÏìíîïÒÓÔÕÖòóôõöÙÚÛÜùúûüÝýÑñÇçŒœÆæß",
    "AAAAAAaaaaaаEEEEeeeeIIIIiiiiOOOOOoooooUUUUuuuuYyNnCcOoAes",
)

def normalize_text(text: str) -> list[str]:
    """Uppercase, transliterate accents, keep only A-Z words of length >= 2."""
    text = text.upper().translate(_ACCENT_MAP)
    words = []
    for w in re.split(r"\s+", text):
        clean = re.sub(r"[^A-Z]", "", w)
        if len(clean) >= 2:
            words.append(clean)
    return words


def random_key(rng: random.Random) -> dict[str, str]:
    shuffled = list(string.ascii_uppercase)
    rng.shuffle(shuffled)
    return dict(zip(string.ascii_uppercase, shuffled))


def apply_key(words: list[str], key: dict[str, str]) -> list[str]:
    return ["".join(key[c] for c in w) for w in words]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_canonical_boundaries(cipher_words: list[str], path: Path) -> None:
    path.write_text(" | ".join(" ".join(w) for w in cipher_words) + "\n")


def write_canonical_no_boundaries(cipher_words: list[str], path: Path) -> None:
    path.write_text(" ".join(c for w in cipher_words for c in w) + "\n")


def write_plaintext_boundaries(plain_words: list[str], path: Path) -> None:
    path.write_text(" ".join(plain_words) + "\n")


def write_plaintext_no_boundaries(plain_words: list[str], path: Path) -> None:
    path.write_text("".join(plain_words) + "\n")


def append_jsonl(record: dict, path: Path) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-language generation
# ---------------------------------------------------------------------------

def generate_language(
    lang: str,
    root: Path,
    rng: random.Random,
    count: int,
) -> None:
    lang_name = LANG_NAMES.get(lang, lang.upper())
    pfx = f"{lang}_ss_synth"
    print(f"\n{'='*60}")
    print(f"  {lang_name} ({lang})")
    print(f"{'='*60}")

    sources = LANG_SOURCES[lang]

    # Fetch text
    all_words: list[str] = []
    for src in sources:
        try:
            raw = fetch_gutenberg(src)
            words = normalize_text(raw)
            print(f"    → {len(words)} words")
            all_words.extend(words)
        except Exception as exc:
            print(f"    WARNING: {exc}")

    if not all_words:
        print(f"  ERROR: no text fetched for {lang}, skipping.")
        return

    print(f"  Word pool: {len(all_words)} words")

    # Directories
    for tier in ("", "_nb"):
        for sub in ("transcriptions", "plaintext", "keys"):
            (root / "sources" / f"{pfx}{tier}" / sub).mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest" / "records.jsonl"
    split_t1 = root / "splits" / f"{pfx}_tests.jsonl"
    split_t2 = root / "splits" / f"{pfx}_nb_tests.jsonl"
    for p in (split_t1, split_t2):
        if p.exists():
            p.unlink()

    # Slice segments
    lengths = SEGMENT_LENGTHS[:count]
    rng.shuffle(lengths)
    segments: list[list[str]] = []
    pos = 0
    for seg_len in lengths:
        if pos + seg_len > len(all_words):
            pos = 0
        segments.append(all_words[pos: pos + seg_len])
        pos += seg_len

    # Tier 1
    print(f"  Tier 1 (word boundaries) …")
    for i, plain_words in enumerate(segments):
        sid = f"{pfx}_{i + 1:03d}"
        key = random_key(rng)
        cipher_words = apply_key(plain_words, key)
        tokens = sum(len(w) for w in cipher_words)

        cr = f"sources/{pfx}/transcriptions/{sid}.canonical.txt"
        pr = f"sources/{pfx}/plaintext/{sid}.txt"
        write_canonical_boundaries(cipher_words, root / cr)
        write_plaintext_boundaries(plain_words, root / pr)
        (root / "sources" / pfx / "keys" / f"{sid}_key.json").write_text(
            json.dumps(key, indent=2)
        )
        append_jsonl({
            "id": sid, "source": pfx,
            "cipher_type": ["simple_substitution"],
            "symbol_set": ["alphabetic"], "symbol_count": 26,
            "plaintext_language": lang,
            "date_or_century": "synthetic",
            "provenance": "Generated from Project Gutenberg public-domain texts",
            "task_tracks": ["transcription2plaintext"],
            "status": "solved_verified",
            "transcription_canonical_file": cr, "plaintext_file": pr,
            "has_key": True, "has_inline_plaintext": False,
            "word_boundaries": True,
            "token_count": tokens, "word_count": len(cipher_words),
        }, manifest_path)
        append_jsonl({
            "test_id": f"{pfx}_single_B_{sid}",
            "track": "transcription2plaintext",
            "cipher_system": f"{pfx}_substitution",
            "target_records": [sid], "context_records": [],
            "description": (
                f"Tier 1 ({lang_name}, word boundaries). "
                f"{len(cipher_words)} words (~{tokens} tokens)"
            ),
        }, split_t1)
        print(f"    [{i+1:2d}/{count}] {sid}  words={len(cipher_words):3d}  tokens={tokens:4d}")

    # Tier 2
    print(f"  Tier 2 (no word boundaries) …")
    for i, plain_words in enumerate(segments):
        sid = f"{pfx}_nb_{i + 1:03d}"
        key = random_key(rng)
        cipher_words = apply_key(plain_words, key)
        tokens = sum(len(w) for w in cipher_words)

        cr = f"sources/{pfx}_nb/transcriptions/{sid}.canonical.txt"
        pr = f"sources/{pfx}_nb/plaintext/{sid}.txt"
        write_canonical_no_boundaries(cipher_words, root / cr)
        write_plaintext_no_boundaries(plain_words, root / pr)
        (root / "sources" / f"{pfx}_nb" / "keys" / f"{sid}_key.json").write_text(
            json.dumps(key, indent=2)
        )
        append_jsonl({
            "id": sid, "source": f"{pfx}_nb",
            "cipher_type": ["simple_substitution"],
            "symbol_set": ["alphabetic"], "symbol_count": 26,
            "plaintext_language": lang,
            "date_or_century": "synthetic",
            "provenance": "Generated from Project Gutenberg public-domain texts",
            "task_tracks": ["transcription2plaintext"],
            "status": "solved_verified",
            "transcription_canonical_file": cr, "plaintext_file": pr,
            "has_key": True, "has_inline_plaintext": False,
            "word_boundaries": False,
            "token_count": tokens, "word_count": len(cipher_words),
            "notes": "No word boundaries. Use char-level accuracy.",
        }, manifest_path)
        append_jsonl({
            "test_id": f"{pfx}_nb_single_B_{sid}",
            "track": "transcription2plaintext",
            "cipher_system": f"{pfx}_nb_substitution",
            "target_records": [sid], "context_records": [],
            "description": (
                f"Tier 2 ({lang_name}, NO word boundaries). "
                f"~{tokens} tokens as one continuous stream"
            ),
        }, split_t2)
        print(f"    [{i+1:2d}/{count}] {sid}  words={len(cipher_words):3d}  tokens={tokens:4d}")

    # Notes file
    docs_dir = root / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / f"{pfx}_notes.md").write_text(
        notes_for(lang, sources), encoding="utf-8"
    )
    print(f"  Notes → {docs_dir / f'{pfx}_notes.md'}")
    print(f"  Split T1 → {split_t1}")
    print(f"  Split T2 → {split_t2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic cipher benchmarks (all languages or one)."
    )
    parser.add_argument("--output", required=True, help="Benchmark root directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=30, help="Tests per tier per language.")
    parser.add_argument(
        "--lang",
        choices=list(LANG_SOURCES.keys()) + ["all"],
        default="all",
        help="Language to generate (default: all).",
    )
    args = parser.parse_args()

    root = Path(args.output).expanduser().resolve()
    rng = random.Random(args.seed)

    langs = list(LANG_SOURCES.keys()) if args.lang == "all" else [args.lang]

    for lang in langs:
        generate_language(lang, root, rng, args.count)

    print(f"\nDone. Generated {len(langs)} language(s) × {args.count * 2} benchmarks each.")
    print("\nRun examples:")
    for lang in langs:
        pfx = f"{lang}_ss_synth"
        print(f"  # {LANG_NAMES.get(lang, lang)} tier 1:")
        print(f"  .venv/bin/decipher benchmark {root} --split {pfx}_tests.jsonl --v2 --language {lang}")
        print(f"  # {LANG_NAMES.get(lang, lang)} tier 2 (no boundaries):")
        print(f"  .venv/bin/decipher benchmark {root} --split {pfx}_nb_tests.jsonl --v2 --language {lang}")


if __name__ == "__main__":
    main()
