from __future__ import annotations

from collections import Counter

# Reference unigram frequencies (percentage, A-Z) per language.
# Sources: Wikipedia letter-frequency articles; Latin from corpus analysis.
# Values are approximate — used for chi² distance, not exact probability.

ENGLISH_LETTER_FREQ: dict[str, float] = {
    "A": 8.167, "B": 1.492, "C": 2.782, "D": 4.253, "E": 12.702,
    "F": 2.228, "G": 2.015, "H": 6.094, "I": 6.966, "J": 0.153,
    "K": 0.772, "L": 4.025, "M": 2.406, "N": 6.749, "O": 7.507,
    "P": 1.929, "Q": 0.095, "R": 5.987, "S": 6.327, "T": 9.056,
    "U": 2.758, "V": 0.978, "W": 2.360, "X": 0.150, "Y": 1.974,
    "Z": 0.074,
}

LATIN_LETTER_FREQ: dict[str, float] = {
    "A": 10.0, "B": 1.6,  "C": 4.4,  "D": 2.6,  "E": 10.0,
    "F": 0.9,  "G": 1.5,  "H": 1.0,  "I": 10.5, "J": 0.05,
    "K": 0.2,  "L": 4.0,  "M": 4.7,  "N": 6.5,  "O": 4.7,
    "P": 3.3,  "Q": 2.1,  "R": 6.3,  "S": 7.5,  "T": 9.5,
    "U": 7.5,  "V": 0.3,  "W": 0.05, "X": 0.6,  "Y": 0.1,
    "Z": 0.1,
}

GERMAN_LETTER_FREQ: dict[str, float] = {
    "A": 6.5,  "B": 1.9,  "C": 2.7,  "D": 5.1,  "E": 17.4,
    "F": 1.7,  "G": 3.0,  "H": 4.2,  "I": 7.6,  "J": 0.3,
    "K": 1.7,  "L": 3.4,  "M": 2.5,  "N": 9.8,  "O": 2.5,
    "P": 0.9,  "Q": 0.1,  "R": 7.0,  "S": 7.3,  "T": 6.2,
    "U": 4.4,  "V": 0.7,  "W": 1.9,  "X": 0.1,  "Y": 0.05,
    "Z": 1.1,
}

FRENCH_LETTER_FREQ: dict[str, float] = {
    "A": 7.61, "B": 0.90, "C": 3.26, "D": 3.67, "E": 14.72,
    "F": 1.12, "G": 1.13, "H": 0.69, "I": 7.53, "J": 0.27,
    "K": 0.05, "L": 5.45, "M": 2.97, "N": 7.10, "O": 5.27,
    "P": 3.01, "Q": 0.89, "R": 6.55, "S": 7.95, "T": 7.24,
    "U": 6.31, "V": 1.63, "W": 0.04, "X": 0.55, "Y": 0.24,
    "Z": 0.15,
}

ITALIAN_LETTER_FREQ: dict[str, float] = {
    "A": 11.74, "B": 0.92, "C": 4.50, "D": 3.73, "E": 11.74,
    "F": 1.22,  "G": 1.64, "H": 1.54, "I": 11.28, "J": 0.0,
    "K": 0.0,   "L": 6.51, "M": 2.51, "N": 6.88,  "O": 9.83,
    "P": 3.05,  "Q": 0.51, "R": 6.37, "S": 4.98,  "T": 5.62,
    "U": 3.01,  "V": 1.57, "W": 0.0,  "X": 0.06,  "Y": 0.0,
    "Z": 0.49,
}

LANGUAGE_LETTER_FREQS: dict[str, dict[str, float]] = {
    "en": ENGLISH_LETTER_FREQ,
    "la": LATIN_LETTER_FREQ,
    "de": GERMAN_LETTER_FREQ,
    "fr": FRENCH_LETTER_FREQ,
    "it": ITALIAN_LETTER_FREQ,
}


def mono_frequency(tokens: list[int]) -> dict[int, int]:
    """Count occurrences of each token."""
    return dict(Counter(tokens))


def mono_frequency_pct(tokens: list[int]) -> dict[int, float]:
    """Percentage frequency of each token."""
    total = len(tokens)
    if total == 0:
        return {}
    counts = Counter(tokens)
    return {t: (c / total) * 100.0 for t, c in counts.items()}


def bigram_frequency(tokens: list[int]) -> dict[tuple[int, int], int]:
    """Count occurrences of each consecutive token pair."""
    counts: dict[tuple[int, int], int] = Counter()
    for i in range(len(tokens) - 1):
        counts[(tokens[i], tokens[i + 1])] += 1
    return dict(counts)


def trigram_frequency(tokens: list[int]) -> dict[tuple[int, int, int], int]:
    """Count occurrences of each consecutive token triple."""
    counts: dict[tuple[int, int, int], int] = Counter()
    for i in range(len(tokens) - 2):
        counts[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1
    return dict(counts)


def chi_squared(observed: dict[int, float], expected: dict[int, float]) -> float:
    """Chi-squared statistic between two frequency distributions (percentages)."""
    total = 0.0
    for key in expected:
        obs = observed.get(key, 0.0)
        exp = expected[key]
        if exp > 0:
            total += (obs - exp) ** 2 / exp
    return total


def sorted_frequency(tokens: list[int]) -> list[tuple[int, int]]:
    """Return (token_id, count) pairs sorted by count descending."""
    return Counter(tokens).most_common()


def unigram_chi2(text: str, language: str) -> float:
    """Chi-squared distance between observed letter frequencies in `text` and
    the reference distribution for `language`.

    Returns 0.0 if the language has no reference or the text has no letters.
    Lower values = better match to the target language.
    """
    ref = LANGUAGE_LETTER_FREQS.get(language)
    if ref is None:
        return 0.0
    letters = [c for c in text.upper() if c.isalpha()]
    if not letters:
        return 0.0
    total = len(letters)
    observed_pct = {c: (cnt / total) * 100.0 for c, cnt in Counter(letters).items()}
    return chi_squared(observed_pct, ref)
