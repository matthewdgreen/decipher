"""Tests for core models: Alphabet, CipherText."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from models.alphabet import Alphabet, Token
from models.cipher_text import CipherText


class TestAlphabet:
    def test_standard_english(self):
        alpha = Alphabet.standard_english()
        assert alpha.size == 26
        assert alpha.symbols[0] == "A"
        assert alpha.symbols[25] == "Z"

    def test_encode_decode_roundtrip(self):
        alpha = Alphabet.standard_english()
        text = "HELLO"
        encoded = alpha.encode(text)
        decoded = alpha.decode(encoded)
        assert decoded == text

    def test_from_text(self):
        alpha = Alphabet.from_text("ABCABC")
        assert alpha.size == 3
        assert alpha.symbols == ["A", "B", "C"]

    def test_from_text_preserves_order(self):
        alpha = Alphabet.from_text("ZYXZYX")
        assert alpha.symbols == ["Z", "Y", "X"]

    def test_multisym_alphabet(self):
        alpha = Alphabet(["SYM_1", "SYM_2", "SYM_3"])
        assert alpha.size == 3
        assert alpha._multisym is True
        encoded = alpha.encode("SYM_1 SYM_3 SYM_2")
        assert encoded == [0, 2, 1]
        decoded = alpha.decode(encoded)
        assert decoded == "SYM_1 SYM_3 SYM_2"

    def test_from_text_multisym(self):
        alpha = Alphabet.from_text("A1 B2 A1 C3", multisym=True)
        assert alpha.size == 3
        assert alpha.symbols == ["A1", "B2", "C3"]

    def test_encode_skips_unknown(self):
        alpha = Alphabet(["A", "B", "C"])
        encoded = alpha.encode("ABXC")
        assert encoded == [0, 1, 2]  # X is skipped

    def test_unique_symbols_required(self):
        with pytest.raises(ValueError, match="unique"):
            Alphabet(["A", "A", "B"])

    def test_empty_symbols_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            Alphabet([])

    def test_has_symbol(self):
        alpha = Alphabet.standard_english()
        assert alpha.has_symbol("A") is True
        assert alpha.has_symbol("1") is False

    def test_id_for_and_symbol_for(self):
        alpha = Alphabet(["X", "Y", "Z"])
        assert alpha.id_for("X") == 0
        assert alpha.id_for("Z") == 2
        assert alpha.symbol_for(1) == "Y"

    def test_from_text_ignore_chars(self):
        alpha = Alphabet.from_text("AB CD EF", ignore_chars={" "})
        assert " " not in alpha.symbols
        assert alpha.size == 6  # A, B, C, D, E, F

    def test_from_text_with_spaces_excluded(self):
        alpha = Alphabet.from_text("HELLO WORLD", ignore_chars={" "})
        assert alpha.has_symbol(" ") is False
        assert alpha.has_symbol("H") is True


class TestCipherText:
    def test_basic_creation(self):
        alpha = Alphabet.standard_english()
        ct = CipherText(raw="HELLO WORLD", alphabet=alpha)
        assert len(ct) > 0
        assert ct.source == "manual"

    def test_tokens_encode_correctly(self):
        alpha = Alphabet(["A", "B", "C"])
        ct = CipherText(raw="ABCBA", alphabet=alpha, separator=None)
        assert ct.tokens == [0, 1, 2, 1, 0]

    def test_segment(self):
        alpha = Alphabet(["A", "B", "C"])
        ct = CipherText(raw="ABCBA", alphabet=alpha, separator=None)
        assert ct.segment(1, 3) == [1, 2]

    def test_display(self):
        alpha = Alphabet(["A", "B", "C"])
        ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
        assert ct.display() == "ABC"

    def test_words_with_space_separator(self):
        alpha = Alphabet.from_text("HELLO WORLD", ignore_chars={" "})
        ct = CipherText(raw="HELLO WORLD", alphabet=alpha, separator=" ")
        assert len(ct.words) == 2
        assert ct.display() == "HELLO WORLD"
        # Spaces should NOT be in the token list
        assert all(alpha.symbol_for(t) != " " for t in ct.tokens)

    def test_words_without_separator(self):
        alpha = Alphabet(["A", "B", "C"])
        ct = CipherText(raw="ABC", alphabet=alpha, separator=None)
        assert len(ct.words) == 1
        assert ct.words[0] == [0, 1, 2]

    def test_spaces_as_cipher_symbols(self):
        alpha = Alphabet.from_text("AB CD")  # space IS a symbol
        ct = CipherText(raw="AB CD", alphabet=alpha, separator=None)
        assert alpha.has_symbol(" ")
        assert len(ct.tokens) == 5  # A, B, space, C, D


