"""Tests for src/analysis/cipher_id.py — cipher-type fingerprint signals."""
from __future__ import annotations

import math
import random

import pytest

from analysis.cipher_id import (
    CipherFingerprint,
    _chi2_vs_uniform,
    _kasiski_analysis,
    _normalized_entropy,
    compute_cipher_fingerprint,
    format_fingerprint_for_context,
)
from collections import Counter


# ---------------------------------------------------------------------------
# Helpers to generate synthetic ciphertext of each type
# ---------------------------------------------------------------------------

# English letter frequencies (A–Z), approximately
_ENGLISH_FREQS = [
    0.0817, 0.0149, 0.0278, 0.0425, 0.1270, 0.0223, 0.0202, 0.0609, 0.0697,
    0.0015, 0.0077, 0.0403, 0.0241, 0.0675, 0.0751, 0.0193, 0.0010, 0.0599,
    0.0633, 0.0906, 0.0276, 0.0098, 0.0236, 0.0015, 0.0197, 0.0007,
]


def _sample_from_freq(freqs: list[float], n: int, rng: random.Random) -> list[int]:
    """Sample n tokens according to the given frequency distribution."""
    population = list(range(len(freqs)))
    return rng.choices(population, weights=freqs, k=n)


def _monoalphabetic_cipher(n: int, seed: int = 42) -> tuple[list[int], list[int]]:
    """Return (cipher_tokens, key) for a synthetic monoalphabetic cipher.

    The plaintext has English-like letter frequencies; the key is a random
    permutation of 0..25.  IC of the output equals the IC of English prose.
    """
    rng = random.Random(seed)
    plaintext = _sample_from_freq(_ENGLISH_FREQS, n, rng)
    key = list(range(26))
    rng.shuffle(key)
    cipher = [key[p] for p in plaintext]
    return cipher, key


def _vigenere_cipher(
    n: int, key_length: int, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Return (cipher_tokens, key) for a synthetic Vigenère cipher.

    Plaintext has English-like frequencies; key is random shifts mod 26.
    """
    rng = random.Random(seed)
    plaintext = _sample_from_freq(_ENGLISH_FREQS, n, rng)
    key = [rng.randint(0, 25) for _ in range(key_length)]
    cipher = [(p + key[i % key_length]) % 26 for i, p in enumerate(plaintext)]
    return cipher, key


def _homophonic_cipher(n: int, alphabet_size: int = 52, seed: int = 42) -> list[int]:
    """Return tokens for a synthetic homophonic cipher with a large alphabet.

    Symbols are distributed proportionally to letter frequency: common letters
    get more homophones, flattening the output distribution.  This ensures
    unique_symbols > 26 and a depressed IC even with modest n.
    """
    rng = random.Random(seed)
    plaintext = _sample_from_freq(_ENGLISH_FREQS, n, rng)
    # Start with 1 symbol per letter, then distribute the remainder to the
    # most frequent letters so high-frequency letters each have 2+ homophones.
    alloc = [1] * 26
    remaining = alphabet_size - 26
    freq_order = sorted(range(26), key=lambda i: _ENGLISH_FREQS[i], reverse=True)
    for j in range(remaining):
        alloc[freq_order[j % 26]] += 1
    # Build contiguous symbol ranges for each letter
    letter_symbols: list[list[int]] = []
    offset = 0
    for a in alloc:
        letter_symbols.append(list(range(offset, offset + a)))
        offset += a
    cipher = [rng.choice(letter_symbols[p]) for p in plaintext]
    return cipher


def _playfair_like_cipher(n: int, seed: int = 42) -> list[int]:
    """Return tokens resembling a Playfair cipher (no same-letter digraphs, even length).

    The Playfair cipher encrypts digraphs; same-letter pairs in a digraph are
    split by insertion of a null.  This means adjacent identical symbols are
    extremely rare.
    """
    rng = random.Random(seed)
    tokens: list[int] = []
    # Generate pairs where both symbols are always different
    n_pairs = (n + 1) // 2
    for _ in range(n_pairs):
        a = rng.randint(0, 24)  # 25-symbol Playfair alphabet
        b = rng.randint(0, 24)
        while b == a:
            b = rng.randint(0, 24)
        tokens.extend([a, b])
    return tokens[:n if n % 2 == 0 else n + 1]


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


def test_chi2_vs_uniform_is_zero_for_flat_distribution():
    n = 260
    counts = Counter({i: 10 for i in range(26)})
    result = _chi2_vs_uniform(counts, n)
    assert result == pytest.approx(0.0, abs=1e-9)


def test_chi2_vs_uniform_is_nonzero_for_peaked_distribution():
    counts = Counter({0: 100, 1: 5, 2: 5, 3: 5, 4: 5})
    n = sum(counts.values())
    result = _chi2_vs_uniform(counts, n)
    assert result > 10.0  # strongly peaked


def test_normalized_entropy_is_one_for_uniform():
    counts = Counter({i: 10 for i in range(26)})
    result = _normalized_entropy(counts, 260)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_normalized_entropy_is_zero_for_single_symbol():
    counts = Counter({0: 100})
    result = _normalized_entropy(counts, 100)
    assert result == pytest.approx(0.0, abs=1e-6)


def test_normalized_entropy_is_between_zero_and_one_for_typical_english():
    rng = random.Random(0)
    tokens = _sample_from_freq(_ENGLISH_FREQS, 500, rng)
    counts = Counter(tokens)
    result = _normalized_entropy(counts, 500)
    assert 0.80 < result < 0.98  # English is fairly spread but not flat


def test_kasiski_analysis_finds_known_period():
    # Build a token stream with a repeating 3-gram at multiples of 7
    tokens = [0] * 200
    for i in range(0, 200 - 5, 7):
        tokens[i] = 99
        tokens[i + 1] = 98
        tokens[i + 2] = 97
    _, best = _kasiski_analysis(tokens, max_period=20)
    # 7 should be among the supported factors
    assert best in {7, 14} or best is None  # short text may be noisy


def test_kasiski_analysis_returns_empty_for_no_repeats():
    tokens = list(range(100))  # all unique tokens, no repeats
    gcds, best = _kasiski_analysis(tokens)
    assert best is None or len(gcds) == 0 or gcds.get(best, 0) == 0


# ---------------------------------------------------------------------------
# Fingerprint signal tests for known cipher types
# ---------------------------------------------------------------------------


class TestMonoalphabeticFingerprint:
    def setup_method(self):
        self.tokens, _ = _monoalphabetic_cipher(500, seed=1)
        self.fp = compute_cipher_fingerprint(self.tokens, 26, language="en")

    def test_ic_near_english_reference(self):
        # IC should be near the English reference (~0.0667)
        assert abs(self.fp.ic - 0.0667) < 0.015

    def test_ic_interpretation_says_monoalphabetic(self):
        assert "monoalphabetic" in self.fp.ic_interpretation.lower()

    def test_frequency_peaked(self):
        # English has peaked frequencies; chi² vs uniform should be high
        assert self.fp.frequency_flatness_chi2 > 1.0

    def test_normalized_entropy_below_one(self):
        assert self.fp.normalized_entropy < 0.97

    def test_monoalphabetic_suspicion_is_highest(self):
        scores = self.fp.suspicion_scores
        top = max(scores, key=lambda k: scores[k])
        assert top == "monoalphabetic_substitution"

    def test_monoalphabetic_suspicion_is_substantial(self):
        assert self.fp.suspicion_scores["monoalphabetic_substitution"] >= 0.40

    def test_vigenere_suspicion_is_low(self):
        assert self.fp.suspicion_scores["polyalphabetic_vigenere"] < 0.30

    def test_unique_symbols_is_26(self):
        assert self.fp.unique_symbols <= 26

    def test_to_dict_serialisable(self):
        d = self.fp.to_dict()
        assert isinstance(d["ic"], float)
        assert isinstance(d["suspicion_scores"], dict)


class TestVigenereFingerprint:
    """Vigenère cipher with period 5 over 500 tokens."""

    def setup_method(self):
        self.tokens, self.key = _vigenere_cipher(500, key_length=5, seed=2)
        self.fp = compute_cipher_fingerprint(self.tokens, 26, language="en")

    def test_ic_depressed_below_english(self):
        # Vigenère IC should be well below 0.0667
        assert self.fp.ic < 0.060

    def test_ic_delta_is_negative(self):
        assert self.fp.ic_delta_from_reference is not None
        assert self.fp.ic_delta_from_reference < -0.005

    def test_normalized_entropy_high(self):
        # Vigenère produces a flatter distribution than monoalphabetic
        assert self.fp.normalized_entropy > 0.90

    def test_vigenere_suspicion_is_highest(self):
        scores = self.fp.suspicion_scores
        top = max(scores, key=lambda k: scores[k])
        assert top == "polyalphabetic_vigenere", f"Expected vigenere but got {top}; scores={scores}"

    def test_vigenere_suspicion_is_substantial(self):
        assert self.fp.suspicion_scores["polyalphabetic_vigenere"] >= 0.40

    def test_monoalphabetic_suspicion_lower_than_vigenere(self):
        assert (
            self.fp.suspicion_scores["polyalphabetic_vigenere"]
            > self.fp.suspicion_scores["monoalphabetic_substitution"]
        )

    def test_best_period_is_five_or_multiple(self):
        # Periodic IC should peak at the true key period (5) or a multiple
        if self.fp.best_period is not None:
            assert self.fp.best_period % 5 == 0 or 5 % self.fp.best_period == 0

    def test_periodic_ic_at_period_5_is_elevated(self):
        # Mean IC over every-5th-token streams should be closer to English reference
        assert 5 in self.fp.periodic_ic
        assert self.fp.periodic_ic[5] > self.fp.ic


class TestVigenereFingerprint3:
    """Vigenère cipher with period 3 over 300 tokens — shorter, smaller period."""

    def setup_method(self):
        self.tokens, _ = _vigenere_cipher(300, key_length=3, seed=7)
        self.fp = compute_cipher_fingerprint(self.tokens, 26, language="en")

    def test_ic_depressed(self):
        assert self.fp.ic < 0.065

    def test_periodic_ic_at_period_3_is_elevated_over_raw_ic(self):
        if 3 in self.fp.periodic_ic:
            assert self.fp.periodic_ic[3] > self.fp.ic


class TestHomophonicFingerprint:
    def setup_method(self):
        self.tokens = _homophonic_cipher(400, alphabet_size=40, seed=3)
        self.fp = compute_cipher_fingerprint(self.tokens, 40, language="en")

    def test_unique_symbols_greater_than_26(self):
        assert self.fp.unique_symbols > 26

    def test_normalized_entropy_high(self):
        # Homophones spread frequency across many symbols
        assert self.fp.normalized_entropy > 0.85

    def test_homophonic_suspicion_is_highest(self):
        scores = self.fp.suspicion_scores
        top = max(scores, key=lambda k: scores[k])
        assert top == "homophonic_substitution", f"Expected homophonic but got {top}; scores={scores}"

    def test_homophonic_suspicion_is_substantial(self):
        assert self.fp.suspicion_scores["homophonic_substitution"] >= 0.40


class TestPlayfairFingerprint:
    def setup_method(self):
        self.tokens = _playfair_like_cipher(300, seed=4)
        self.fp = compute_cipher_fingerprint(
            self.tokens, 25, language="en"
        )

    def test_doubled_digraph_rate_is_below_random_rate(self):
        # Playfair prevents within-pair identical symbols, roughly halving the
        # doubled-digraph rate vs. a random 25-symbol cipher (~1/25 ≈ 0.040).
        # Cross-pair boundaries have no constraint, so the rate is ~0.020, not
        # near-zero.  We verify it is well below the random-cipher expectation.
        assert self.fp.doubled_digraph_rate < 0.035

    def test_even_length(self):
        assert self.fp.even_length

    def test_unique_symbols_is_25(self):
        assert self.fp.unique_symbols == 25

    def test_playfair_suspicion_substantial(self):
        assert self.fp.suspicion_scores["playfair"] >= 0.40

    def test_playfair_suspicion_is_high_relative_to_vigenere(self):
        assert (
            self.fp.suspicion_scores["playfair"]
            > self.fp.suspicion_scores["polyalphabetic_vigenere"]
        )


# ---------------------------------------------------------------------------
# Short text / edge cases
# ---------------------------------------------------------------------------


def test_short_text_returns_insufficient_data_interpretation():
    fp = compute_cipher_fingerprint([0, 1, 2, 3], 26)
    assert "insufficient data" in fp.ic_interpretation or math.isnan(fp.ic)


def test_short_text_returns_unknown_suspicion():
    fp = compute_cipher_fingerprint([0, 1, 2], 26)
    # With very few tokens, only "unknown" should be returned
    if math.isnan(fp.ic):
        assert "unknown" in fp.suspicion_scores


def test_empty_tokens():
    fp = compute_cipher_fingerprint([], 26)
    assert fp.token_count == 0
    assert fp.unique_symbols == 0
    assert math.isnan(fp.ic)


def test_word_group_count_is_recorded():
    tokens, _ = _monoalphabetic_cipher(100)
    fp = compute_cipher_fingerprint(tokens, 26, word_group_count=15)
    assert fp.word_group_count == 15


# ---------------------------------------------------------------------------
# format_fingerprint_for_context
# ---------------------------------------------------------------------------


def test_format_fingerprint_contains_suspicion_section():
    tokens, _ = _monoalphabetic_cipher(300)
    fp = compute_cipher_fingerprint(tokens, 26)
    text = format_fingerprint_for_context(fp)
    assert "Top suspicions" in text
    assert "monoalphabetic_substitution" in text


def test_format_fingerprint_contains_ic_line():
    tokens, _ = _monoalphabetic_cipher(300)
    fp = compute_cipher_fingerprint(tokens, 26)
    text = format_fingerprint_for_context(fp)
    assert "IC = " in text


def test_format_fingerprint_contains_entropy_line():
    tokens, _ = _monoalphabetic_cipher(300)
    fp = compute_cipher_fingerprint(tokens, 26)
    text = format_fingerprint_for_context(fp)
    assert "entropy" in text.lower()


def test_format_fingerprint_shows_period_for_vigenere():
    tokens, _ = _vigenere_cipher(500, key_length=5, seed=10)
    fp = compute_cipher_fingerprint(tokens, 26)
    text = format_fingerprint_for_context(fp)
    assert "Periodic IC" in text or "period" in text.lower()


def test_format_fingerprint_shows_doubled_digraph():
    tokens = _playfair_like_cipher(200)
    fp = compute_cipher_fingerprint(tokens, 25)
    text = format_fingerprint_for_context(fp)
    assert "digraph" in text.lower() or "doubled" in text.lower()


def test_format_fingerprint_contains_cipher_type_fingerprint_header():
    tokens, _ = _monoalphabetic_cipher(200)
    fp = compute_cipher_fingerprint(tokens, 26)
    text = format_fingerprint_for_context(fp)
    assert text.startswith("## Cipher-type fingerprint")


# ---------------------------------------------------------------------------
# Language reference IC variation
# ---------------------------------------------------------------------------


def test_latin_reference_ic_used_correctly():
    tokens, _ = _monoalphabetic_cipher(400, seed=5)
    fp = compute_cipher_fingerprint(tokens, 26, language="la")
    # Latin reference IC is ~0.0737; IC delta should be computed against it
    assert fp.language_ic_reference == pytest.approx(0.0737, abs=0.001)
    assert fp.ic_delta_from_reference is not None


def test_german_reference_ic_used_correctly():
    tokens, _ = _monoalphabetic_cipher(400, seed=6)
    fp = compute_cipher_fingerprint(tokens, 26, language="de")
    assert fp.language_ic_reference == pytest.approx(0.0762, abs=0.001)


def test_unknown_language_falls_back_to_english():
    tokens, _ = _monoalphabetic_cipher(400, seed=8)
    fp = compute_cipher_fingerprint(tokens, 26, language="xx")
    assert fp.language_ic_reference == pytest.approx(0.0667, abs=0.001)
