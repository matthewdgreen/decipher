"""Tests for cipher implementations."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.alphabet import Alphabet
from ciphers.substitution import SubstitutionCipher
from ciphers.caesar import CaesarCipher


class TestSubstitutionCipher:
    def test_encrypt_decrypt_roundtrip(self):
        alpha = Alphabet.standard_english()
        cipher = SubstitutionCipher()
        key = cipher.random_key(alpha)

        plaintext = alpha.encode("HELLOWORLD")
        encrypted = cipher.encrypt(plaintext, key, alpha)
        decrypted = cipher.decrypt(encrypted, key, alpha)
        assert decrypted == plaintext

    def test_known_key(self):
        alpha = Alphabet(["A", "B", "C"])
        cipher = SubstitutionCipher()
        # Key: A->C, B->A, C->B (shift by 2 basically)
        key = {0: 2, 1: 0, 2: 1}
        plaintext = [0, 1, 2]  # A B C
        encrypted = cipher.encrypt(plaintext, key, alpha)
        assert encrypted == [2, 0, 1]  # C A B

    def test_invert_key(self):
        key = {0: 2, 1: 0, 2: 1}
        inv = SubstitutionCipher.invert_key(key)
        assert inv == {2: 0, 0: 1, 1: 2}

    def test_random_key_is_permutation(self):
        alpha = Alphabet.standard_english()
        cipher = SubstitutionCipher()
        key = cipher.random_key(alpha)
        assert set(key.keys()) == set(range(26))
        assert set(key.values()) == set(range(26))


class TestCaesarCipher:
    def test_encrypt_decrypt_roundtrip(self):
        alpha = Alphabet.standard_english()
        cipher = CaesarCipher()
        plaintext = alpha.encode("HELLO")
        for shift in range(26):
            encrypted = cipher.encrypt(plaintext, shift, alpha)
            decrypted = cipher.decrypt(encrypted, shift, alpha)
            assert decrypted == plaintext

    def test_known_shift(self):
        alpha = Alphabet.standard_english()
        cipher = CaesarCipher()
        plaintext = alpha.encode("ABC")  # [0, 1, 2]
        encrypted = cipher.encrypt(plaintext, 3, alpha)
        assert encrypted == [3, 4, 5]  # DEF

    def test_wrap_around(self):
        alpha = Alphabet.standard_english()
        cipher = CaesarCipher()
        plaintext = alpha.encode("XYZ")  # [23, 24, 25]
        encrypted = cipher.encrypt(plaintext, 3, alpha)
        assert encrypted == [0, 1, 2]  # ABC

    def test_brute_force(self):
        alpha = Alphabet.standard_english()
        cipher = CaesarCipher()
        plaintext = alpha.encode("HELLO")
        encrypted = cipher.encrypt(plaintext, 7, alpha)
        results = cipher.brute_force(encrypted, alpha)
        # One of the 26 results should match the original
        decryptions = [tokens for _, tokens in results]
        assert plaintext in decryptions
