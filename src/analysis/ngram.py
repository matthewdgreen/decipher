"""N-gram models built from wordlists with lazy per-language caching.

We synthesize quadgram and bigram tables by concatenating the wordlist with
space padding, which captures word-boundary structure. Not as rich as a
full-corpus table, but language-faithful enough to rank decryptions and
far better than flat wordlist hits for medieval Latin / archaic German.
"""
from __future__ import annotations

import math
from collections import Counter


def _clean_text(text: str) -> str:
    """Uppercase and collapse non-alpha runs to single spaces."""
    text = text.upper()
    out: list[str] = []
    prev_space = True
    for c in text:
        if c.isalpha():
            out.append(c)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return "".join(out).strip()


def build_ngram_counts(word_list: list[str], n: int) -> dict[str, int]:
    """Count n-grams across a wordlist with word-boundary padding."""
    counts: Counter[str] = Counter()
    pad = " " * (n - 1)
    for word in word_list:
        w = word.upper().strip()
        if not w:
            continue
        padded = f"{pad}{w}{pad}"
        for i in range(len(padded) - n + 1):
            gram = padded[i : i + n]
            counts[gram] += 1
    return dict(counts)


def to_log_probs(counts: dict[str, int]) -> dict[str, float]:
    """Convert raw counts to log10 probabilities with a floor for unseen."""
    total = sum(counts.values())
    if total == 0:
        return {}
    floor = math.log10(0.01 / total)
    lp: dict[str, float] = {g: math.log10(c / total) for g, c in counts.items()}
    lp["_floor"] = floor
    return lp


def ngram_loglik(text: str, log_probs: dict[str, float], n: int) -> float:
    """Total log-likelihood of text under the n-gram model."""
    clean = _clean_text(text)
    if len(clean) < n:
        return float("-inf")
    pad = " " * (n - 1)
    padded = f"{pad}{clean}{pad}"
    floor = log_probs.get("_floor", -10.0)
    total = 0.0
    for i in range(len(padded) - n + 1):
        gram = padded[i : i + n]
        total += log_probs.get(gram, floor)
    return total


def normalized_ngram_score(text: str, log_probs: dict[str, float], n: int = 4) -> float:
    """Mean log-prob per n-gram. Comparable across texts of different lengths."""
    clean = _clean_text(text)
    if len(clean) < n:
        return float("-inf")
    pad = " " * (n - 1)
    padded = f"{pad}{clean}{pad}"
    count = len(padded) - n + 1
    if count <= 0:
        return float("-inf")
    floor = log_probs.get("_floor", -10.0)
    total = 0.0
    for i in range(count):
        gram = padded[i : i + n]
        total += log_probs.get(gram, floor)
    return total / count


def ngram_position_logprobs(
    text: str, log_probs: dict[str, float], n: int = 4
) -> list[tuple[int, str, float]]:
    """Per-position n-gram log-probabilities. Returns (char_index, gram, log_prob).

    char_index is the offset in the cleaned text where the n-gram begins (0-based,
    after _clean_text has normalized whitespace but before boundary padding).
    n-grams that span the padding boundary receive a char_index of -1 (leading)
    or len(clean) (trailing).
    """
    clean = _clean_text(text)
    pad = " " * (n - 1)
    padded = f"{pad}{clean}{pad}"
    floor = log_probs.get("_floor", -10.0)
    out: list[tuple[int, str, float]] = []
    for i in range(len(padded) - n + 1):
        gram = padded[i : i + n]
        lp = log_probs.get(gram, floor)
        char_index = i - (n - 1)
        out.append((char_index, gram, lp))
    return out


class NGramCache:
    """Lazy per-language n-gram tables, built from the wordlist on first use."""

    def __init__(self) -> None:
        self._tables: dict[tuple[str, int], dict[str, float]] = {}

    def get(self, language: str, n: int) -> dict[str, float]:
        key = (language, n)
        if key not in self._tables:
            from analysis.dictionary import get_dictionary_path
            from analysis.pattern import load_word_list

            path = get_dictionary_path(language)
            words = load_word_list(path)
            counts = build_ngram_counts(words, n)
            self._tables[key] = to_log_probs(counts)
        return self._tables[key]

    def clear(self) -> None:
        self._tables.clear()


NGRAM_CACHE = NGramCache()
