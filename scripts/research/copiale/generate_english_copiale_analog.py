#!/usr/bin/env python3
"""Generate a small English Copiale-style benchmark fixture.

This is a human-intuition fixture, not a historical claim. It creates an
archaic-English nomenclator-like cipher with:

- homophonic A-Z letter symbols
- null symbols inserted into the stream
- whole-word logogram symbols for common lodge-style words

The fixture is deliberately benchmark-shaped so ordinary Decipher commands can
run against it, while the secret key sidecar remains outside solver inputs.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


DEFAULT_OUT = Path("fixtures/benchmarks/english_copiale_analog")

PLAINTEXT = """
WHEN THE LODGE WAS MADE READY AND THE LAMPS WERE SET IN THEIR PLACES THE MASTER
COMMANDED THAT EVERY BROTHER SHOULD STAND IN SILENCE. THEN THE DOOR WAS SHUT
AND THE WARDEN GAVE THE SIGN WITH HIS RIGHT HAND. THE BRETHREN ANSWERED IN LIKE
MANNER AND NO MAN SPOKE UNTIL THE MASTER HAD COUNTED THE STROKES UPON THE TABLE.
AFTER THIS HE OPENED THE SMALL BOOK AND READ THE OLD CHARGE CONCERNING CHARITY
FIDELITY AND THE GOVERNMENT OF THE TONGUE. HE SAID THAT A BROTHER WHO KEEPETH
HIS WORD IS STRONGER THAN A WALL AND THAT A MAN WHO BREAKETH COUNSEL BRINGETH
DARKNESS INTO THE HOUSE. THE CANDIDATE WAS THEN LED THREE TIMES ABOUT THE ROOM
AND AT EACH TURN HE WAS TAUGHT A TOKEN BY WHICH HE MIGHT KNOW A FRIEND IN
TRAVEL. THE FIRST TOKEN WAS OF THE HAND THE SECOND WAS OF THE EYE AND THE THIRD
WAS OF THE HEART. WHEN THESE THINGS WERE FINISHED THE MASTER GAVE HIM LIGHT AND
THE BRETHREN RECEIVED HIM WITH QUIET JOY. THEY WROTE HIS NAME IN THE REGISTER
AND SET BESIDE IT THE MARK OF THE ORDER. AFTERWARD THE COMPANY SAT TOGETHER AND
SPOKE OF GOOD WORKS OF THE RELIEF OF WIDOWS AND OF THE SCHOOLING OF POOR
CHILDREN. BEFORE THE NIGHT WAS ENDED THE MASTER WARNED THEM THAT TRUE KNOWLEDGE
IS NOT NOISE NOR DISPLAY BUT A STEADY FLAME KEPT UNDER GLASS. THEREFORE EACH
BROTHER DEPARTED PEACEABLY AND CARRIED THE LESSON HOME.
"""

LOGOGRAM_WORDS = (
    "THE",
    "AND",
    "OF",
    "THAT",
    "MASTER",
    "BROTHER",
    "BRETHREN",
    "LODGE",
    "SIGN",
    "HAND",
    "ORDER",
)

LETTER_HOMOPHONE_COUNTS = {
    "E": 8,
    "T": 6,
    "A": 5,
    "O": 5,
    "N": 5,
    "I": 4,
    "S": 4,
    "H": 4,
    "R": 4,
    "D": 3,
    "L": 3,
    "U": 3,
    "M": 3,
    "C": 2,
    "W": 2,
    "F": 2,
    "G": 2,
    "Y": 2,
    "P": 2,
    "B": 2,
    "V": 1,
    "K": 1,
    "J": 1,
    "X": 1,
    "Q": 1,
    "Z": 1,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--null-rate", type=float, default=0.055)
    parser.add_argument("--logogram-rate", type=float, default=0.82)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    words = normalize_words(PLAINTEXT)
    key = make_key(rng)
    encoded = encode_words(
        words,
        key=key,
        rng=rng,
        null_rate=args.null_rate,
        logogram_rate=args.logogram_rate,
    )
    write_fixture(args.out, words, encoded, key, args)
    print(f"Wrote {args.out}")
    print(f"Words: {len(words)}")
    print(f"Cipher tokens: {sum(len(word) for word in encoded)}")
    print(f"Cipher symbols: {len(key['symbols'])}")
    print(f"Logograms: {', '.join(LOGOGRAM_WORDS)}")


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[A-Z]+", text.upper())


def make_key(rng: random.Random) -> dict:
    symbols: dict[str, dict] = {}
    letter_to_symbols: dict[str, list[str]] = {}
    next_id = 1

    for letter, count in LETTER_HOMOPHONE_COUNTS.items():
        letter_to_symbols[letter] = []
        for _ in range(count):
            symbol = f"S{next_id:03d}"
            next_id += 1
            symbols[symbol] = {"type": "letter", "plaintext": letter}
            letter_to_symbols[letter].append(symbol)

    nulls = []
    for _ in range(12):
        symbol = f"S{next_id:03d}"
        next_id += 1
        symbols[symbol] = {"type": "null", "plaintext": ""}
        nulls.append(symbol)

    logograms = {}
    for word in LOGOGRAM_WORDS:
        symbol = f"S{next_id:03d}"
        next_id += 1
        symbols[symbol] = {"type": "logogram", "plaintext": word}
        logograms[word] = symbol

    return {
        "letter_to_symbols": letter_to_symbols,
        "nulls": nulls,
        "logograms": logograms,
        "symbols": symbols,
    }


def encode_words(
    words: list[str],
    *,
    key: dict,
    rng: random.Random,
    null_rate: float,
    logogram_rate: float,
) -> list[list[str]]:
    encoded: list[list[str]] = []
    for word in words:
        tokens: list[str] = []
        if word in key["logograms"] and rng.random() < logogram_rate:
            tokens.append(key["logograms"][word])
        else:
            for ch in word:
                if rng.random() < null_rate:
                    tokens.append(rng.choice(key["nulls"]))
                tokens.append(rng.choice(key["letter_to_symbols"][ch]))
                if rng.random() < null_rate * 0.45:
                    tokens.append(rng.choice(key["nulls"]))
        if rng.random() < null_rate:
            tokens.append(rng.choice(key["nulls"]))
        encoded.append(tokens)
    return encoded


def write_fixture(out: Path, words: list[str], encoded: list[list[str]], key: dict, args: argparse.Namespace) -> None:
    source_dir = out / "sources" / "english_copiale_analog"
    trans_dir = source_dir / "transcriptions"
    plain_dir = source_dir / "plaintext"
    meta_dir = source_dir / "metadata"
    split_dir = out / "splits"
    manifest_dir = out / "manifest"
    for path in (trans_dir, plain_dir, meta_dir, split_dir, manifest_dir):
        path.mkdir(parents=True, exist_ok=True)

    canonical_lines = wrap_tokens([token for word in encoded for token in word], width=52)
    (trans_dir / "english_copiale_analog_001.canonical.txt").write_text(
        "\n".join(" ".join(line) for line in canonical_lines) + "\n",
        encoding="utf-8",
    )
    (plain_dir / "english_copiale_analog_001.txt").write_text(
        " ".join(words) + "\n",
        encoding="utf-8",
    )
    (meta_dir / "english_copiale_analog_001.key.json").write_text(
        json.dumps(
            {
                "note": "Secret key for diagnostics only. Do not expose to solvers.",
                "seed": args.seed,
                "null_rate": args.null_rate,
                "logogram_rate": args.logogram_rate,
                **key,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    record = {
        "id": "english_copiale_analog_001",
        "source": "decipher_fixture",
        "status": "synthetic_solved",
        "cipher_type": ["homophonic_substitution", "nomenclator", "nulls", "logograms"],
        "symbol_set": ["synthetic_s_tokens", "homophonic", "nulls", "word_signs"],
        "symbol_count": len(key["symbols"]),
        "plaintext_language": "en",
        "date_or_century": "synthetic archaic English",
        "transcription_canonical_file": "sources/english_copiale_analog/transcriptions/english_copiale_analog_001.canonical.txt",
        "plaintext_file": "sources/english_copiale_analog/plaintext/english_copiale_analog_001.txt",
        "has_key": True,
        "known_cipher_parameters": {
            "homophonic": True,
            "has_nulls": True,
            "has_logograms": True,
            "plaintext_units": ["A-Z letters", "null", "whole-word signs"],
        },
        "curation_notes": (
            "Invented English Copiale-style analog for debugging human and agent "
            "intuition on homophones, nulls, and logograms. The plaintext is "
            "synthetic archaic lodge prose, not a historical source."
        ),
        "context_layers": {
            "minimal": {
                "label": "Minimal synthetic context",
                "text": "Synthetic manuscript-style cipher fixture. No solution or key is included in this context.",
                "contains_solution": False,
                "contains_plaintext_hint": False,
                "contains_cipher_type_hint": False,
            },
            "standard": {
                "label": "Synthetic cipher-family context",
                "text": (
                    "Plaintext language: English. The cipher is a Copiale-style "
                    "homophonic/nomenclator analog with nulls and whole-word signs."
                ),
                "contains_solution": False,
                "contains_plaintext_hint": True,
                "contains_cipher_type_hint": True,
            },
        },
    }
    (manifest_dir / "records.jsonl").write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")

    split = {
        "test_id": "english_copiale_analog_001",
        "track": "transcription2plaintext",
        "cipher_system": "english_copiale_analog_nomenclator",
        "target_records": ["english_copiale_analog_001"],
        "context_records": [],
        "description": "Synthetic English Copiale-style analog with homophones, nulls, and logograms.",
    }
    (split_dir / "english_copiale_analog.jsonl").write_text(
        json.dumps(split, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    (out / "README.md").write_text(readme_text(args), encoding="utf-8")


def wrap_tokens(tokens: list[str], *, width: int) -> list[list[str]]:
    return [tokens[idx : idx + width] for idx in range(0, len(tokens), width)]


def readme_text(args: argparse.Namespace) -> str:
    return f"""# English Copiale Analog Fixture

This local benchmark fixture is a synthetic, readable-English analog of a
Copiale-like nomenclator. It is for intuition-building and solver diagnostics,
not historical evaluation.

The plaintext is invented archaic lodge prose. The cipher uses:

- homophonic symbols for A-Z letters
- null symbols
- whole-word logograms for: {", ".join(LOGOGRAM_WORDS)}

Generation parameters:

- seed: `{args.seed}`
- null rate: `{args.null_rate}`
- logogram rate: `{args.logogram_rate}`

The secret key is written to
`sources/english_copiale_analog/metadata/english_copiale_analog_001.key.json`
for post-hoc diagnostics only.

Automated run:

```bash
PYTHONPATH=src .venv/bin/decipher benchmark fixtures/benchmarks/english_copiale_analog \\
  --split english_copiale_analog.jsonl \\
  --test-id english_copiale_analog_001 \\
  --automated-only \\
  --homophonic-budget screen \\
  --homophonic-refinement null_masks \\
  --artifact-dir artifacts/english_copiale_analog_automated
```

Agentic run:

```bash
PYTHONPATH=src .venv/bin/decipher benchmark fixtures/benchmarks/english_copiale_analog \\
  --split english_copiale_analog.jsonl \\
  --test-id english_copiale_analog_001 \\
  --agentic \\
  --provider openai \\
  --model gpt-5.4 \\
  --benchmark-context standard \\
  --max-iterations 30 \\
  --homophonic-budget screen \\
  --homophonic-refinement null_masks \\
  --artifact-dir artifacts/english_copiale_analog_agentic \\
  --analyze
```
"""


if __name__ == "__main__":
    main()
