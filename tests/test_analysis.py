"""Tests for analysis tools."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis import frequency, ic, pattern


class TestFrequency:
    def test_mono_frequency(self):
        tokens = [0, 1, 0, 2, 0, 1]
        result = frequency.mono_frequency(tokens)
        assert result[0] == 3
        assert result[1] == 2
        assert result[2] == 1

    def test_mono_frequency_pct(self):
        tokens = [0, 0, 0, 0, 1]  # 80% zeros, 20% ones
        result = frequency.mono_frequency_pct(tokens)
        assert abs(result[0] - 80.0) < 0.01
        assert abs(result[1] - 20.0) < 0.01

    def test_bigram_frequency(self):
        tokens = [0, 1, 0, 1, 0]
        result = frequency.bigram_frequency(tokens)
        assert result[(0, 1)] == 2
        assert result[(1, 0)] == 2

    def test_trigram_frequency(self):
        tokens = [0, 1, 2, 0, 1, 2]
        result = frequency.trigram_frequency(tokens)
        assert result[(0, 1, 2)] == 2

    def test_sorted_frequency(self):
        tokens = [2, 2, 2, 1, 1, 0]
        result = frequency.sorted_frequency(tokens)
        assert result[0] == (2, 3)
        assert result[1] == (1, 2)
        assert result[2] == (0, 1)

    def test_chi_squared(self):
        observed = {0: 50.0, 1: 50.0}
        expected = {0: 50.0, 1: 50.0}
        assert frequency.chi_squared(observed, expected) == 0.0

        observed2 = {0: 60.0, 1: 40.0}
        result = frequency.chi_squared(observed2, expected)
        assert result > 0


class TestIC:
    def test_ic_uniform(self):
        # Perfectly uniform distribution
        tokens = list(range(26)) * 10  # each letter appears exactly 10 times
        result = ic.index_of_coincidence(tokens, 26)
        assert abs(result - ic.random_ic(26)) < 0.01

    def test_ic_english_like(self):
        # Repeated pattern with high IC
        tokens = [0] * 50 + [1] * 30 + [2] * 20
        result = ic.index_of_coincidence(tokens, 26)
        assert result > ic.random_ic(26)

    def test_ic_single_token(self):
        assert ic.index_of_coincidence([0], 26) == 0.0

    def test_ic_empty(self):
        assert ic.index_of_coincidence([], 26) == 0.0

    def test_is_likely_monoalphabetic(self):
        assert ic.is_likely_monoalphabetic(0.068) is True
        assert ic.is_likely_monoalphabetic(0.040) is False


class TestPattern:
    def test_word_pattern(self):
        # HELLO -> H.E.L.L.O -> 0.1.2.2.3
        tokens = [7, 4, 11, 11, 14]
        result = pattern.word_pattern(tokens)
        assert result == "0.1.2.2.3"

    def test_word_pattern_all_same(self):
        tokens = [5, 5, 5]
        assert pattern.word_pattern(tokens) == "0.0.0"

    def test_word_pattern_all_different(self):
        tokens = [1, 2, 3]
        assert pattern.word_pattern(tokens) == "0.1.2"

    def test_split_into_words(self):
        tokens = [0, 1, 99, 2, 3, 99, 4]
        words = pattern.split_into_words(tokens, separator_id=99)
        assert words == [[0, 1], [2, 3], [4]]

    def test_split_no_separator(self):
        tokens = [0, 1, 2]
        words = pattern.split_into_words(tokens, separator_id=None)
        assert words == [[0, 1, 2]]

    def test_find_isomorphs(self):
        words = [[0, 1, 0], [2, 3, 2], [4, 5, 6]]
        groups = pattern.find_isomorphs(words)
        # [0,1,0] and [2,3,2] have same pattern "0.1.0"
        assert len(groups["0.1.0"]) == 2
        assert len(groups["0.1.2"]) == 1

    def test_build_and_match_pattern_dict(self):
        word_list = ["HELLO", "WORLD", "TEETH", "APPLE"]
        pd = pattern.build_pattern_dictionary(word_list)
        # HELLO has pattern 0.1.2.2.3, TEETH has pattern 0.1.1.0.2,
        # APPLE has pattern 0.1.1.2.3
        matches = pattern.match_pattern("0.1.2.2.3", pd)
        assert "HELLO" in matches


class TestDictionary:
    def test_score_plaintext(self):
        from analysis.dictionary import score_plaintext
        word_set = {"THE", "CAT", "SAT", "ON", "MAT"}
        assert score_plaintext("THE CAT SAT ON THE MAT", word_set) == 1.0
        assert score_plaintext("XYZ ABC DEF", word_set) == 0.0
        assert score_plaintext("THE XYZ", word_set) == 0.5
