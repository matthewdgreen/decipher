"""P8 numeric battery tests (INV-0 Part 4 / Part 9)."""
from __future__ import annotations

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.numeric_code import (
    alphabetical_run_report,
    assert_not_solution_bearing,
    benford_epsilon,
    is_numeric_ciphertext,
    last_digit_chi2,
    numeric_code_battery,
    parse_numeric_ciphertext,
)


def test_parse_and_detect():
    assert parse_numeric_ciphertext("1 2, 3\n4") == [1, 2, 3, 4]
    assert is_numeric_ciphertext("10 20 30")
    assert not is_numeric_ciphertext("ABC DEF")
    with pytest.raises(ValueError):
        parse_numeric_ciphertext("1 2 THREE 4")


def test_benford_math_vs_hand_values():
    # All values start with digit 1 -> obs law is a point mass at d=1.
    values = [1, 10, 11, 100, 123, 19]
    ben1 = math.log10(1 + 1 / 1)  # ~0.30103
    eps = benford_epsilon(values)
    # sup-norm deviation: obs(1)=1.0 -> |1.0 - 0.30103| = 0.69897 dominates.
    assert abs(eps - (1.0 - ben1)) < 1e-9


def test_last_digit_chi2_uniform_stream():
    # Perfectly uniform last digits -> chi2 == 0.
    values = list(range(0, 100))
    assert last_digit_chi2(values) == pytest.approx(0.0, abs=1e-9)


def test_word_position_book_cipher_flag_supported():
    # Synthetic word-position book cipher: front-loaded indices into a key text.
    from analysis.numeric_code import BEALE_DOI_KEY_PATH
    words = [w for w in open(BEALE_DOI_KEY_PATH).read().split() if not w.startswith("#")]
    booklen = len(words)
    rng = random.Random(7)
    values = []
    for _ in range(300):
        values.append(rng.randint(1, max(2, booklen // 5)) if rng.random() < 0.55
                      else rng.randint(1, booklen))
    battery = numeric_code_battery(values, n_shuffles=300)
    assert battery["flags"]["word_first_letter_book_cipher"]["plausibility"] == "supported"


def test_uniform_stream_random_like_not_hoax():
    # A seeded iid-uniform stream: independent_random_like supported AND
    # structured_hoax_artifact neutral (finding 8 fixture contradiction resolved).
    rng = random.Random(42)
    values = [rng.randint(1, 3000) for _ in range(300)]
    battery = numeric_code_battery(values, n_shuffles=300)
    assert battery["flags"]["independent_random_like"]["plausibility"] == "supported"
    assert battery["flags"]["structured_hoax_artifact"]["plausibility"] == "neutral"


def test_alphabetical_run_report_finds_planted_run():
    # A long ascending alphabetical run planted in random noise.
    rng = random.Random(1)
    noise = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(200))
    planted = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = noise + planted + noise
    report = alphabetical_run_report(text, n_shuffles=1000)
    assert report["longest_increasing_run"] >= 20
    assert report["longest_increasing_baseline"]["p_value"] < 1.0 / 1001 + 1e-12


def test_solution_bearing_guard():
    from analysis.numeric_code import BEALE_DOI_KEY_PATH
    content = open(BEALE_DOI_KEY_PATH).read()
    with pytest.raises(ValueError):
        assert_not_solution_bearing(content, source="doi")
    # ordinary ciphertext passes
    assert_not_solution_bearing("71 194 38 1701")
