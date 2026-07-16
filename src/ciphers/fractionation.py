"""Fractionation ciphers: Bifid, Trifid, ADFGX, ADFGVX.

These fractionate each plaintext letter into Polybius coordinates, mix the
coordinate streams, and recombine (ADFGX/ADFGVX add a keyed columnar
transposition of the fractionated text).  Delastelle's Bifid/Trifid and the
WWI German ADFGX/ADFGVX field ciphers (Kahn, *The Codebreakers*; ACA "Bifid",
"Trifid").

All operate on cleaned uppercase strings and round-trip exactly.  Bifid/ADFGX
merge I/J (J -> I); ADFGVX keeps all 26 letters plus 0-9.
"""

from __future__ import annotations

import random
from typing import Any

from ciphers.family_base import FamilyCipher
from ciphers.textutil import (
    DIGITS,
    clean_alnum,
    clean_ij,
    keyed_square_25,
    keyed_square_36,
    random_keyword,
)
from ciphers.transposition import columnar_decrypt, columnar_encrypt

SQUARE25 = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # J merged into I


# ---------------------------------------------------------------------------
# Bifid
# ---------------------------------------------------------------------------

def _blocks(seq, period):
    if period and period > 0:
        for i in range(0, len(seq), period):
            yield seq[i: i + period]
    else:
        yield seq


class BifidCipher(FamilyCipher):
    """Bifid (Delastelle).  Key is a 25-letter Polybius square (I/J merged).

    Coordinates (row, col) in 0-4.  Within each period-block: emit all rows then
    all columns, regroup the combined stream into pairs, and read each pair as a
    new (row, col).  ``period=0`` (default) treats the whole message as one
    block (the classic textbook form).
    """

    name = "bifid"

    def __init__(self, period: int = 0) -> None:
        self.period = period

    def _pos(self, square: str) -> dict[str, tuple[int, int]]:
        return {ch: (i // 5, i % 5) for i, ch in enumerate(square)}

    def encrypt(self, plaintext: str, key: str) -> str:
        square = key
        pos = self._pos(square)
        letters = clean_ij(plaintext)
        out: list[str] = []
        for block in _blocks(letters, self.period):
            rows = [pos[c][0] for c in block]
            cols = [pos[c][1] for c in block]
            seq = rows + cols
            for i in range(0, len(seq), 2):
                out.append(square[seq[i] * 5 + seq[i + 1]])
        return "".join(out)

    def decrypt(self, ciphertext: str, key: str) -> str:
        square = key
        pos = self._pos(square)
        letters = clean_ij(ciphertext)
        out: list[str] = []
        for block in _blocks(letters, self.period):
            coords: list[int] = []
            for c in block:
                r, col = pos[c]
                coords.append(r)
                coords.append(col)
            half = len(block)
            rows = coords[:half]
            cols = coords[half:]
            for r, col in zip(rows, cols):
                out.append(square[r * 5 + col])
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any) -> str:
        letters = list(SQUARE25)
        rng.shuffle(letters)
        return "".join(letters)

    def describe_key(self, key: str) -> str:
        period = f"period={self.period}" if self.period else "whole-message"
        return f"bifid square={key!r} ({period}, I/J-merged)"

    def key_space_size(self, **_: Any) -> int:
        import math
        return math.factorial(25)

    @staticmethod
    def square_from_keyword(keyword: str) -> str:
        return keyed_square_25(keyword)


# ---------------------------------------------------------------------------
# Trifid
# ---------------------------------------------------------------------------

TRIFID_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ+"  # 27 cells (26 letters + filler)


class TrifidCipher(FamilyCipher):
    """Trifid (Delastelle).  Key is a 27-symbol string (A-Z plus one filler).

    Each symbol maps to a 3-coordinate triple in a 3x3x3 cube.  Within each
    period-block: emit the three coordinate streams in order, regroup into
    triples, and read each as a new symbol.  Default ``period=5`` (Delastelle's
    original).
    """

    name = "trifid"

    def __init__(self, period: int = 5) -> None:
        self.period = period

    def _pos(self, cube: str) -> dict[str, tuple[int, int, int]]:
        return {ch: (i // 9, (i // 3) % 3, i % 3) for i, ch in enumerate(cube)}

    def _clean(self, text: str) -> str:
        return "".join(ch for ch in text.upper() if ch in TRIFID_SYMBOLS)

    def encrypt(self, plaintext: str, key: str) -> str:
        cube = key
        pos = self._pos(cube)
        letters = self._clean(plaintext)
        out: list[str] = []
        for block in _blocks(letters, self.period):
            a = [pos[c][0] for c in block]
            b = [pos[c][1] for c in block]
            c3 = [pos[c][2] for c in block]
            seq = a + b + c3
            for i in range(0, len(seq), 3):
                out.append(cube[seq[i] * 9 + seq[i + 1] * 3 + seq[i + 2]])
        return "".join(out)

    def decrypt(self, ciphertext: str, key: str) -> str:
        cube = key
        pos = self._pos(cube)
        letters = self._clean(ciphertext)
        out: list[str] = []
        for block in _blocks(letters, self.period):
            coords: list[int] = []
            for c in block:
                coords.extend(pos[c])
            n = len(block)
            a = coords[0:n]
            b = coords[n:2 * n]
            c3 = coords[2 * n:3 * n]
            for x, y, z in zip(a, b, c3):
                out.append(cube[x * 9 + y * 3 + z])
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any) -> str:
        symbols = list(TRIFID_SYMBOLS)
        rng.shuffle(symbols)
        return "".join(symbols)

    def describe_key(self, key: str) -> str:
        return f"trifid cube={key!r} (period={self.period}, 27-symbol)"

    def key_space_size(self, **_: Any) -> int:
        import math
        return math.factorial(27)


# ---------------------------------------------------------------------------
# ADFGX / ADFGVX
# ---------------------------------------------------------------------------

class _ADFGxBase(FamilyCipher):
    """Shared machinery for ADFGX (5x5) and ADFGVX (6x6)."""

    labels = "ADFGX"
    side = 5

    def _square(self, keyword: str) -> str:
        raise NotImplementedError

    def _clean(self, text: str) -> str:
        raise NotImplementedError

    def _fractionate(self, text: str, square: str) -> str:
        pos = {ch: (i // self.side, i % self.side) for i, ch in enumerate(square)}
        out: list[str] = []
        for ch in text:
            r, c = pos[ch]
            out.append(self.labels[r])
            out.append(self.labels[c])
        return "".join(out)

    def encrypt(self, plaintext: str, key) -> str:
        square = self._square(key[0])
        trans = "".join(ch for ch in key[1].upper() if ch.isalnum())
        frac = self._fractionate(self._clean(plaintext), square)
        return columnar_encrypt(frac, trans)

    def decrypt(self, ciphertext: str, key) -> str:
        square = self._square(key[0])
        trans = "".join(ch for ch in key[1].upper() if ch.isalnum())
        label_idx = {lab: i for i, lab in enumerate(self.labels)}
        frac = columnar_decrypt("".join(ch for ch in ciphertext.upper() if ch in self.labels), trans)
        out: list[str] = []
        for i in range(0, len(frac) - 1, 2):
            r = label_idx[frac[i]]
            c = label_idx[frac[i + 1]]
            out.append(square[r * self.side + c])
        return "".join(out)

    def describe_key(self, key) -> str:
        return f"{self.name} square_keyword={key[0]!r} transposition_keyword={key[1]!r}"

    def key_space_size(self, *, columns: int = 7, **_: Any) -> int:
        import math
        cells = self.side * self.side
        return math.factorial(cells) * math.factorial(columns)


class ADFGXCipher(_ADFGxBase):
    """ADFGX: 5x5 keyed Polybius (I/J merged) -> ADFGX pairs -> keyed columnar.

    Key is ``(square_keyword, transposition_keyword)``.
    """

    name = "adfgx"
    labels = "ADFGX"
    side = 5

    def _square(self, keyword: str) -> str:
        return keyed_square_25(keyword)

    def _clean(self, text: str) -> str:
        return clean_ij(text)

    def random_key(self, rng: random.Random, **_: Any):
        return (random_keyword(rng, min_len=5, max_len=9), random_keyword(rng, min_len=5, max_len=8))


class ADFGVXCipher(_ADFGxBase):
    """ADFGVX: 6x6 keyed Polybius (A-Z + 0-9) -> ADFGVX pairs -> keyed columnar.

    Key is ``(square_keyword, transposition_keyword)``.  Plaintext may include
    digits.
    """

    name = "adfgvx"
    labels = "ADFGVX"
    side = 6

    def _square(self, keyword: str) -> str:
        return keyed_square_36(keyword)

    def _clean(self, text: str) -> str:
        return clean_alnum(text)

    def random_key(self, rng: random.Random, **_: Any):
        return (random_keyword(rng, min_len=6, max_len=10), random_keyword(rng, min_len=5, max_len=8))
