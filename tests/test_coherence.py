"""island_report coherence tests (INV-0 Part 5 / Part 9)."""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.coherence import island_report

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CORPUS = os.path.join(_REPO, "corpus_data", "en")

# Beale-2 self-check plaintext (the review-2 finding-2 hard gate).
B2_PLAINTEXT = (
    "I HAVE DEPOSITED IN THE COUNTY OF BEDFORD ABOUT FOUR MILES FROM BUFORDS "
    "IN AN EXCAVATION OR VAULT SIX FEET BELOW THE SURFACE OF THE GROUND THE "
    "FOLLOWING ARTICLES BELONGING JOINTLY TO THE PARTIES WHOSE NAMES ARE GIVEN "
    "IN NUMBER THREE HEREWITH THE FIRST DEPOSIT CONSISTED OF ONE THOUSAND AND "
    "FOURTEEN POUNDS OF GOLD AND THREE THOUSAND EIGHT HUNDRED AND TWELVE POUNDS "
    "OF SILVER DEPOSITED NOVEMBER EIGHTEEN NINETEEN"
)


def _prose_words(pgid, skip_words=500, n_words=260):
    path = os.path.join(_CORPUS, f"pg{pgid}.txt")
    if not os.path.exists(path):
        pytest.skip(f"corpus page {pgid} not available")
    with open(path, encoding="utf-8", errors="ignore") as f:
        t = f.read()
    words = ["".join(c for c in w if c.isalpha()) for w in t.upper().split()]
    words = [w for w in words if w]
    return words[skip_words:skip_words + n_words]


def test_ordered_prose_is_coherent():
    # amendment #3: pages 84/11/98/120 clear dict_rate >= 0.75.
    words = _prose_words(84)
    r = island_report(" ".join(words), "en")
    assert r["verdict"] == "coherent"
    assert r["word_bigram_order_significant"] is True


def test_shuffled_prose_not_coherent():
    words = _prose_words(98)
    shuffled = list(words)
    random.Random(2024).shuffle(shuffled)
    r = island_report(" ".join(shuffled), "en")
    assert r["verdict"] != "coherent"
    # Order-significance is the anti-shuffle guard (measures ~0.365).
    assert r["word_bigram_order_p"] > 0.05


def test_b2_self_check_plaintext_coherent():
    r = island_report(B2_PLAINTEXT, "en")
    assert r["verdict"] == "coherent"
    assert r["longest_coherent_span"] >= 5
    assert r["word_bigram_order_p"] <= 0.05


def test_latin_never_coherent():
    # No word-bigram resource for la -> coherent is impossible.
    r = island_report(B2_PLAINTEXT, "la")
    assert r["word_bigram_available"] is False
    assert r["verdict"] != "coherent"


def test_borg_0077v_basin_not_coherent():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "borg_0077v_basin.txt")
    text = "".join(l for l in open(path) if not l.startswith("#")).strip()
    r = island_report(text, "la")
    assert r["verdict"] != "coherent"


def test_random_letters_gibberish():
    rng = random.Random(3)
    gib = " ".join(
        "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(6))
        for _ in range(60)
    )
    r = island_report(gib, "en")
    assert r["verdict"] == "gibberish"


def test_precomputed_segmented_matches_internal():
    from analysis.finalist_validation import _load_word_list, _load_word_set
    from analysis.segment import segment_text

    letters = "".join(c for c in B2_PLAINTEXT if c.isalpha())
    word_set = _load_word_set("en")
    freq_rank = {w.upper(): i for i, w in enumerate(_load_word_list("en"))}
    segmented = segment_text(letters, word_set, freq_rank=freq_rank)

    via_segmented = island_report(letters, "en", word_set=word_set,
                                  freq_rank=freq_rank, segmented=segmented)
    via_internal = island_report(letters, "en", word_set=word_set, freq_rank=freq_rank)
    assert via_segmented["verdict"] == via_internal["verdict"]
    assert via_segmented["word_count"] == via_internal["word_count"]
