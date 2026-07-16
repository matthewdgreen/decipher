"""Playfair-family digraph ciphers: Playfair, Two-square, Four-square.

All use 5x5 keyed squares with the classical I/J merge (J folded to I, so the
recovered plaintext contains I where the source had J).  Standard digraph
rules (Kahn, *The Codebreakers*; ACA "Playfair", "Two Square", "Four Square").

Round-trip note: these ciphers pad/adjust the plaintext into digraphs before
enciphering (Playfair also inserts an ``X`` between repeated letters and pads
odd length).  ``decrypt(encrypt(pt))`` therefore reproduces the *prepared*
digraph text, which each cipher exposes via :meth:`prepare`.  Tests pin
``decrypt(encrypt(pt)) == prepare(pt)``.
"""

from __future__ import annotations

import random
from typing import Any

from ciphers.family_base import FamilyCipher
from ciphers.textutil import clean_ij, keyed_square_25, random_keyword

PLAIN_SQUARE = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # standard 5x5, J merged into I


def _positions(square: str) -> dict[str, tuple[int, int]]:
    return {ch: (i // 5, i % 5) for i, ch in enumerate(square)}


def _digraphs_playfair(text: str, filler: str = "X") -> list[tuple[str, str]]:
    """Split into Playfair digraphs, inserting ``filler`` between equal letters."""
    letters = clean_ij(text)
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(letters):
        a = letters[i]
        b = letters[i + 1] if i + 1 < len(letters) else None
        if b is None:
            pairs.append((a, filler if a != filler else "Z"))
            i += 1
        elif a == b:
            pairs.append((a, filler if a != filler else "Z"))
            i += 1  # consume only the first letter
        else:
            pairs.append((a, b))
            i += 2
    return pairs


def _even_digraphs(text: str, filler: str = "X") -> list[tuple[str, str]]:
    """Split into digraphs, padding odd length (no double-letter handling)."""
    letters = clean_ij(text)
    if len(letters) % 2:
        letters += filler
    return [(letters[i], letters[i + 1]) for i in range(0, len(letters), 2)]


class PlayfairCipher(FamilyCipher):
    """Playfair cipher (Wheatstone/Playfair; ACA "Playfair").

    Key is a keyword; the 5x5 square is ``keyed_square_25(keyword)``.  Digraph
    rules: same row -> letters to the right (wrap); same column -> letters below
    (wrap); rectangle -> swap to the column of the partner.
    """

    name = "playfair"

    def prepare(self, plaintext: str) -> str:
        return "".join(a + b for a, b in _digraphs_playfair(plaintext))

    def _square(self, key: str) -> str:
        return keyed_square_25(key)

    def _crypt_pair(self, a: str, b: str, pos, sq: str, step: int) -> tuple[str, str]:
        (ra, ca), (rb, cb) = pos[a], pos[b]
        if ra == rb:
            return sq[ra * 5 + (ca + step) % 5], sq[rb * 5 + (cb + step) % 5]
        if ca == cb:
            return sq[((ra + step) % 5) * 5 + ca], sq[((rb + step) % 5) * 5 + cb]
        return sq[ra * 5 + cb], sq[rb * 5 + ca]

    def encrypt(self, plaintext: str, key: str) -> str:
        sq = self._square(key)
        pos = _positions(sq)
        out = [self._crypt_pair(a, b, pos, sq, +1) for a, b in _digraphs_playfair(plaintext)]
        return "".join(a + b for a, b in out)

    def decrypt(self, ciphertext: str, key: str) -> str:
        sq = self._square(key)
        pos = _positions(sq)
        letters = clean_ij(ciphertext)
        pairs = [(letters[i], letters[i + 1]) for i in range(0, len(letters) - 1, 2)]
        out = [self._crypt_pair(a, b, pos, sq, -1) for a, b in pairs]
        return "".join(a + b for a, b in out)

    def random_key(self, rng: random.Random, **_: Any) -> str:
        return random_keyword(rng, min_len=5, max_len=9)

    def describe_key(self, key: str) -> str:
        return f"playfair keyword={clean_ij(key)!r} (5x5 I/J-merged square)"

    def key_space_size(self, **_: Any) -> int:
        # distinct 5x5 squares = 25! (keyword construction reaches a subset)
        import math
        return math.factorial(25)


class TwoSquareCipher(FamilyCipher):
    """Two-square (double Playfair), vertical arrangement (ACA "Two Square").

    Two keyed 5x5 squares stacked vertically.  For digraph ``(a, b)``: locate
    ``a`` in the top square at ``(r1, c1)`` and ``b`` in the bottom square at
    ``(r2, c2)``; the cipher pair is ``(top[r1][c2], bottom[r2][c1])`` (swap
    columns).  Same-column pairs pass through unchanged.  The operation is an
    involution, so ``decrypt == encrypt``.  No double-letter ``X`` insertion is
    needed (the two letters come from different squares); odd length is padded.
    """

    name = "two_square"

    def prepare(self, plaintext: str) -> str:
        return "".join(a + b for a, b in _even_digraphs(plaintext))

    def _squares(self, key) -> tuple[str, str]:
        k1, k2 = key
        return keyed_square_25(k1), keyed_square_25(k2)

    def _crypt(self, ciphertext_or_plaintext: str, key) -> str:
        top, bottom = self._squares(key)
        pos_t, pos_b = _positions(top), _positions(bottom)
        out: list[str] = []
        for a, b in _even_digraphs(ciphertext_or_plaintext):
            (r1, c1) = pos_t[a]
            (r2, c2) = pos_b[b]
            if c1 == c2:
                out.append(a + b)
            else:
                out.append(top[r1 * 5 + c2] + bottom[r2 * 5 + c1])
        return "".join(out)

    def encrypt(self, plaintext: str, key) -> str:
        return self._crypt(plaintext, key)

    def decrypt(self, ciphertext: str, key) -> str:
        # self-reciprocal
        return self._crypt(ciphertext, key)

    def random_key(self, rng: random.Random, **_: Any) -> tuple[str, str]:
        return (random_keyword(rng, min_len=5, max_len=9), random_keyword(rng, min_len=5, max_len=9))

    def describe_key(self, key) -> str:
        k1, k2 = key
        return f"two-square keywords=({clean_ij(k1)!r}, {clean_ij(k2)!r})"

    def key_space_size(self, **_: Any) -> int:
        import math
        return math.factorial(25) ** 2


class FourSquareCipher(FamilyCipher):
    """Four-square cipher (ACA "Four Square").

    Layout::

        plain1   cipher1
        cipher2  plain2

    ``plain1`` and ``plain2`` are the standard 5x5 square; ``cipher1`` and
    ``cipher2`` are keyed squares.  Encrypt ``(a, b)``: find ``a`` in ``plain1``
    at ``(r1, c1)`` and ``b`` in ``plain2`` at ``(r2, c2)``; cipher pair is
    ``(cipher1[r1][c2], cipher2[r2][c1])``.  Decrypt reverses through the cipher
    squares.  Not self-reciprocal.  Odd length is padded; no double-letter
    handling needed.
    """

    name = "four_square"

    def prepare(self, plaintext: str) -> str:
        return "".join(a + b for a, b in _even_digraphs(plaintext))

    def _squares(self, key) -> tuple[str, str]:
        k1, k2 = key
        return keyed_square_25(k1), keyed_square_25(k2)

    def encrypt(self, plaintext: str, key) -> str:
        c1sq, c2sq = self._squares(key)
        pos_p = _positions(PLAIN_SQUARE)
        out: list[str] = []
        for a, b in _even_digraphs(plaintext):
            (r1, ca) = pos_p[a]
            (r2, cb) = pos_p[b]
            out.append(c1sq[r1 * 5 + cb] + c2sq[r2 * 5 + ca])
        return "".join(out)

    def decrypt(self, ciphertext: str, key) -> str:
        c1sq, c2sq = self._squares(key)
        pos_c1, pos_c2 = _positions(c1sq), _positions(c2sq)
        letters = clean_ij(ciphertext)
        pairs = [(letters[i], letters[i + 1]) for i in range(0, len(letters) - 1, 2)]
        out: list[str] = []
        for a, b in pairs:
            (r1, cb) = pos_c1[a]
            (r2, ca) = pos_c2[b]
            out.append(PLAIN_SQUARE[r1 * 5 + ca] + PLAIN_SQUARE[r2 * 5 + cb])
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any) -> tuple[str, str]:
        return (random_keyword(rng, min_len=5, max_len=9), random_keyword(rng, min_len=5, max_len=9))

    def describe_key(self, key) -> str:
        k1, k2 = key
        return f"four-square keywords=({clean_ij(k1)!r}, {clean_ij(k2)!r})"

    def key_space_size(self, **_: Any) -> int:
        import math
        return math.factorial(25) ** 2
