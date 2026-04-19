from __future__ import annotations

from ciphers.base import Cipher
from models.alphabet import Alphabet


class CaesarCipher(Cipher):
    """Caesar / ROT-N cipher (special case of substitution)."""

    def encrypt(
        self, plaintext: list[int], key: int, alphabet: Alphabet
    ) -> list[int]:
        n = alphabet.size
        return [(t + key) % n for t in plaintext]

    def decrypt(
        self, ciphertext: list[int], key: int, alphabet: Alphabet
    ) -> list[int]:
        n = alphabet.size
        return [(t - key) % n for t in ciphertext]

    def key_space_size(self, alphabet: Alphabet) -> int:
        return alphabet.size

    def brute_force(
        self, ciphertext: list[int], alphabet: Alphabet
    ) -> list[tuple[int, list[int]]]:
        """Return all possible decryptions with their shift values."""
        return [
            (shift, self.decrypt(ciphertext, shift, alphabet))
            for shift in range(alphabet.size)
        ]
