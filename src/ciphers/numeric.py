"""Numeric ciphers: Nihilist substitution, Straddling checkerboard, Hill (2x2).

These emit numeric ciphertext (space-separated numbers or a digit string) and
are documented as such.  Standard definitions (Kahn, *The Codebreakers*; ACA
"Nihilist Sub", "Checkerboard"; Hill, 1929).
"""

from __future__ import annotations

import random
from typing import Any

from ciphers.family_base import FamilyCipher
from ciphers.textutil import ALPHABET, clean_ij, keyed_square_25, modinv, random_keyword, coprime


# ---------------------------------------------------------------------------
# Nihilist substitution
# ---------------------------------------------------------------------------

class NihilistSubstitutionCipher(FamilyCipher):
    """Nihilist substitution (Polybius + keyed addition).

    A 5x5 keyed Polybius square (I/J merged) maps each letter to a two-digit
    number ``10*row + col`` with row/col in 1-5 (values 11-55).  A repeated
    additive keyword (encoded through the same square) is added positionally
    (no modular reduction), so ciphertext numbers run roughly 22-110.

    Key is ``(square_keyword, additive_keyword)``.  Ciphertext is a
    space-separated string of integers; :meth:`decrypt` parses that form.
    """

    name = "nihilist_substitution"

    def _square(self, keyword: str) -> str:
        return keyed_square_25(keyword)

    def _numbers(self, text: str, square: str) -> list[int]:
        pos = {ch: (i // 5 + 1, i % 5 + 1) for i, ch in enumerate(square)}
        return [10 * pos[c][0] + pos[c][1] for c in text]

    def encrypt(self, plaintext: str, key) -> str:
        square = self._square(key[0])
        pt_nums = self._numbers(clean_ij(plaintext), square)
        key_nums = self._numbers(clean_ij(key[1]), square)
        if not key_nums:
            raise ValueError("additive keyword must contain letters")
        out = [pt_nums[i] + key_nums[i % len(key_nums)] for i in range(len(pt_nums))]
        return " ".join(str(x) for x in out)

    def decrypt(self, ciphertext: str, key) -> str:
        square = self._square(key[0])
        key_nums = self._numbers(clean_ij(key[1]), square)
        inv = {10 * (i // 5 + 1) + (i % 5 + 1): ch for i, ch in enumerate(square)}
        nums = [int(tok) for tok in ciphertext.split()]
        out: list[str] = []
        for i, n in enumerate(nums):
            plain_num = n - key_nums[i % len(key_nums)]
            out.append(inv[plain_num])
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any):
        return (random_keyword(rng, min_len=5, max_len=8), random_keyword(rng, min_len=4, max_len=7))

    def describe_key(self, key) -> str:
        return f"nihilist-sub square_keyword={key[0]!r} additive_keyword={key[1]!r}"

    def key_space_size(self, *, key_len: int = 6, **_: Any) -> int:
        import math
        return math.factorial(25) * (25 ** key_len)


# ---------------------------------------------------------------------------
# Straddling checkerboard
# ---------------------------------------------------------------------------

class StraddlingCheckerboardCipher(FamilyCipher):
    """Straddling checkerboard.

    A 3-row board: the top row (columns 0-9) holds 8 letters, leaving two blank
    columns whose digits index the two lower rows of 10 letters each.  Top-row
    letters encode to one digit (their column); lower-row letters encode to two
    digits (row-indicator digit + column).  The two indicator digits never occur
    as standalone codes, so the digit string is uniquely decodable.

    Key is ``(arrangement, blank1, blank2)`` -- a 26-letter arrangement and the
    two blank column positions (0-9).  Ciphertext is a digit string.
    """

    name = "straddling_checkerboard"

    def _codebook(self, key) -> dict[str, str]:
        arrangement, b1, b2 = key
        b1, b2 = sorted((int(b1), int(b2)))
        top_cols = [c for c in range(10) if c not in (b1, b2)]
        code: dict[str, str] = {}
        for i, c in enumerate(top_cols):
            code[arrangement[i]] = str(c)
        for c in range(10):
            idx = 8 + c
            if idx < 26:
                code[arrangement[idx]] = f"{b1}{c}"
        for c in range(10):
            idx = 18 + c
            if idx < 26:
                code[arrangement[idx]] = f"{b2}{c}"
        return code

    def encrypt(self, plaintext: str, key) -> str:
        code = self._codebook(key)
        return "".join(code[ch] for ch in clean_ij(plaintext) if ch in code)

    def decrypt(self, ciphertext: str, key) -> str:
        _arr, b1, b2 = key
        b1, b2 = sorted((int(b1), int(b2)))
        inv = {v: k for k, v in self._codebook(key).items()}
        digits = "".join(ch for ch in ciphertext if ch.isdigit())
        out: list[str] = []
        i = 0
        indicators = {str(b1), str(b2)}
        while i < len(digits):
            if digits[i] in indicators:
                out.append(inv[digits[i:i + 2]])
                i += 2
            else:
                out.append(inv[digits[i]])
                i += 1
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any):
        # 26-letter arrangement across the board; input folds J->I, so the 'J'
        # cell is simply never exercised (a harmless dead cell).
        letters = list(ALPHABET)
        rng.shuffle(letters)
        b1, b2 = sorted(rng.sample(range(10), 2))
        return ("".join(letters), b1, b2)

    def describe_key(self, key) -> str:
        arrangement, b1, b2 = key
        b1, b2 = sorted((int(b1), int(b2)))
        return f"straddling-checkerboard arrangement={arrangement!r} blanks=({b1},{b2})"

    def key_space_size(self, **_: Any) -> int:
        import math
        return math.factorial(26) * 45  # 45 = C(10, 2) blank-column choices


# ---------------------------------------------------------------------------
# Hill (2x2)
# ---------------------------------------------------------------------------

class Hill2x2Cipher(FamilyCipher):
    """Hill cipher with a 2x2 key matrix over Z26 (Hill, 1929).

    Key is ``(a, b, c, d)`` for the matrix ``[[a, b], [c, d]]`` whose
    determinant is invertible mod 26.  Plaintext is enciphered in digraphs
    ``[p0, p1] -> K @ [p0, p1] mod 26``; odd length is padded with ``X``.
    ``decrypt(encrypt(pt))`` reproduces the padded plaintext.
    """

    name = "hill2x2"

    def _det_inv(self, key) -> tuple[int, int]:
        a, b, c, d = key
        det = (a * d - b * c) % 26
        return det, modinv(det, 26)

    def _pairs(self, text: str) -> list[tuple[int, int]]:
        vals = [ord(ch) - 65 for ch in text]
        if len(vals) % 2:
            vals.append(ord("X") - 65)
        return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]

    def encrypt(self, plaintext: str, key) -> str:
        a, b, c, d = key
        out: list[str] = []
        for p0, p1 in self._pairs(clean_ij(plaintext)):
            out.append(chr((a * p0 + b * p1) % 26 + 65))
            out.append(chr((c * p0 + d * p1) % 26 + 65))
        return "".join(out)

    def decrypt(self, ciphertext: str, key) -> str:
        a, b, c, d = key
        _det, inv = self._det_inv(key)
        # inverse matrix = inv * [[d, -b], [-c, a]]
        ia, ib, ic, idd = (inv * d) % 26, (inv * -b) % 26, (inv * -c) % 26, (inv * a) % 26
        out: list[str] = []
        for c0, c1 in self._pairs(clean_ij(ciphertext)):
            out.append(chr((ia * c0 + ib * c1) % 26 + 65))
            out.append(chr((ic * c0 + idd * c1) % 26 + 65))
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any):
        while True:
            a, b, c, d = (rng.randrange(26) for _ in range(4))
            if coprime((a * d - b * c) % 26, 26):
                return (a, b, c, d)

    def describe_key(self, key) -> str:
        a, b, c, d = key
        return f"hill2x2 matrix=[[{a},{b}],[{c},{d}]]"

    def key_space_size(self, **_: Any) -> int:
        # |GL_2(Z_26)| = |GL_2(Z_2)| * |GL_2(Z_13)| = 6 * 26208 = 157248
        return 157248
