"""Tests for signal panel + n-gram scoring."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis import ngram, signals


class TestNgram:
    def test_build_counts(self):
        counts = ngram.build_ngram_counts(["hello", "help"], n=2)
        # Both words contribute " H" and "H E" etc. (space padding)
        assert counts[" H"] == 2

    def test_log_probs_sum(self):
        counts = {"AB": 5, "CD": 5}
        lp = ngram.to_log_probs(counts)
        assert "_floor" in lp
        assert lp["AB"] == lp["CD"]

    def test_normalized_score_floors_unknown(self):
        lp = ngram.to_log_probs({"AB": 1})
        # "XY" is unknown — should use floor, not crash
        s = ngram.normalized_ngram_score("XY", lp, n=2)
        assert s < 0

    def test_higher_for_real_english(self):
        # Crude smoke test: real English should score higher than noise
        en_lp = ngram.NGRAM_CACHE.get("en", 4)
        real = ngram.normalized_ngram_score("THE QUICK BROWN FOX JUMPS", en_lp, n=4)
        noise = ngram.normalized_ngram_score("XZQP VWTR BKFG YJHM WSCD", en_lp, n=4)
        assert real > noise


class TestNormalizeForScoring:
    def test_collapses_spaced_letters(self):
        # "Q U E D A M" should collapse to QUEDAM
        out = signals.normalize_for_scoring("Q U E D A M")
        assert out == "QUEDAM"

    def test_replaces_canonical_separator(self):
        out = signals.normalize_for_scoring("HELLO | WORLD")
        assert out == "HELLO WORLD"

    def test_multisym_output(self):
        # Canonical-style multisym decryption
        out = signals.normalize_for_scoring("Q U E D A M | A N U S")
        assert out == "QUEDAM ANUS"


class TestConstraintSatisfaction:
    def test_one_to_one(self):
        key = {0: 0, 1: 1, 2: 2}
        assert signals._constraint_satisfaction(key) == 1.0

    def test_homophonic(self):
        key = {0: 5, 1: 5, 2: 5}  # three ct map to same pt
        assert signals._constraint_satisfaction(key) == 1 / 3

    def test_empty_is_satisfied(self):
        assert signals._constraint_satisfaction({}) == 1.0


class TestPanel:
    def test_panel_runs_on_english(self):
        # Simple monoalphabetic: ciphertext is "ABC" mapping to "THE"
        cipher_words = [[0, 1, 2]]
        key = {0: ord("T") - ord("A"), 1: ord("H") - ord("A"), 2: ord("E") - ord("A")}
        panel = signals.compute_panel(
            decrypted="THE",
            cipher_words=cipher_words,
            key=key,
            used_ct_ids={0, 1, 2},
            language="en",
        )
        assert panel.dictionary_rate == 1.0
        assert panel.mapped_count == 3
        assert panel.unmapped_count == 0
        assert panel.constraint_satisfaction == 1.0

    def test_panel_gibberish(self):
        cipher_words = [[0, 1, 2]]
        key = {0: 23, 1: 24, 2: 25}  # maps to XYZ
        panel = signals.compute_panel(
            decrypted="XYZ",
            cipher_words=cipher_words,
            key=key,
            used_ct_ids={0, 1, 2},
            language="en",
        )
        assert panel.dictionary_rate == 0.0
        assert panel.unrecognized_word_count == 1
