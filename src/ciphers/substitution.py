from __future__ import annotations

import random
from math import factorial

from ciphers.base import Cipher
from models.alphabet import Alphabet


class SubstitutionCipher(Cipher):
    """Monoalphabetic substitution cipher with arbitrary alphabets.

    Key is a dict mapping plaintext token ID -> ciphertext token ID.
    """

    def encrypt(
        self, plaintext: list[int], key: dict[int, int], alphabet: Alphabet
    ) -> list[int]:
        return [key.get(t, t) for t in plaintext]

    def decrypt(
        self, ciphertext: list[int], key: dict[int, int], alphabet: Alphabet
    ) -> list[int]:
        inv = self.invert_key(key)
        return [inv.get(t, t) for t in ciphertext]

    def key_space_size(self, alphabet: Alphabet) -> int:
        return factorial(alphabet.size)

    @staticmethod
    def invert_key(key: dict[int, int]) -> dict[int, int]:
        return {v: k for k, v in key.items()}

    @staticmethod
    def random_key(alphabet: Alphabet) -> dict[int, int]:
        ids = list(range(alphabet.size))
        shuffled = list(ids)
        random.shuffle(shuffled)
        return dict(zip(ids, shuffled))
