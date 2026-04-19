from __future__ import annotations

import math
import os
from collections import Counter

# Supported languages and their dictionary filenames.
# "unknown" has no dictionary; tools that require one receive an empty set.
LANGUAGE_DICTS: dict[str, str | None] = {
    "en": "english_common.txt",
    "la": "latin_common.txt",
    "de": "german_common.txt",
    "fr": "fr_common.txt",
    "it": "it_common.txt",
    "unknown": None,
}

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "la": "Latin",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "unknown": "Unknown",
}


def get_dictionary_dir() -> str:
    """Return the path to the dictionaries directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "resources", "dictionaries",
    )


def get_dictionary_path(language: str = "en") -> str | None:
    """Return the path to the dictionary file for the given language code.

    Returns None for 'unknown' (no dictionary available).
    Raises ValueError for unrecognised language codes.
    """
    if language not in LANGUAGE_DICTS:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported: {', '.join(LANGUAGE_DICTS.keys())}"
        )
    filename = LANGUAGE_DICTS[language]
    if filename is None:
        return None
    return os.path.join(get_dictionary_dir(), filename)


def load_word_set(path: str) -> set[str]:
    """Load a dictionary file into a set of uppercase words."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip().upper() for line in f if line.strip()}


def score_plaintext(text: str, word_set: set[str]) -> float:
    """Score plaintext by fraction of words found in dictionary.

    Returns 0.0-1.0. Higher is better.
    Ignores separator tokens like '|' and single-char non-alpha tokens.
    """
    words = text.upper().split()
    # Filter out separator tokens and non-word tokens
    words = [w for w in words if w.isalpha() or "?" in w]
    if not words:
        return 0.0
    found = sum(1 for w in words if w in word_set)
    return found / len(words)


# Quadgram scoring for text without word boundaries

def load_quadgrams(path: str) -> dict[str, float]:
    """Load quadgram log-probability table.

    File format: each line is "QUAD count", e.g. "TION 13168375".
    Returns log10 probabilities.
    """
    if not os.path.exists(path):
        return {}
    counts: dict[str, int] = {}
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                quad, count = parts[0].upper(), int(parts[1])
                counts[quad] = count
                total += count
    if total == 0:
        return {}
    floor_log = math.log10(0.01 / total)
    return {
        quad: math.log10(count / total)
        for quad, count in counts.items()
    } | {"_floor": floor_log}


def score_quadgrams(text: str, quadgram_log_probs: dict[str, float]) -> float:
    """Score text using quadgram log-probabilities.

    Higher (less negative) is better. Returns sum of log10 probs.
    """
    text = text.upper()
    # Remove non-alpha characters
    text = "".join(c for c in text if c.isalpha())
    if len(text) < 4:
        return float("-inf")
    floor = quadgram_log_probs.get("_floor", -10.0)
    score = 0.0
    for i in range(len(text) - 3):
        quad = text[i : i + 4]
        score += quadgram_log_probs.get(quad, floor)
    return score
