from __future__ import annotations

from collections import Counter

ENGLISH_IC = 0.0667
RANDOM_IC_26 = 1.0 / 26.0  # ~0.0385


def random_ic(alphabet_size: int) -> float:
    return 1.0 / alphabet_size


def index_of_coincidence(tokens: list[int], alphabet_size: int) -> float:
    """Compute the index of coincidence for a token sequence.

    IC = sum(n_i * (n_i - 1)) / (N * (N - 1))
    where n_i is the count of each token and N is total tokens.
    """
    n = len(tokens)
    if n <= 1:
        return 0.0
    counts = Counter(tokens)
    numerator = sum(c * (c - 1) for c in counts.values())
    denominator = n * (n - 1)
    return numerator / denominator


def is_likely_monoalphabetic(ic: float, alphabet_size: int = 26) -> bool:
    """Heuristic: IC near English suggests monoalphabetic substitution."""
    expected = ENGLISH_IC
    rand = random_ic(alphabet_size)
    threshold = (expected + rand) / 2
    return ic >= threshold
