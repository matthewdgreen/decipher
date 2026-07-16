"""Periodic and running-key polyalphabetic ciphers.

Standard textbook / ACA definitions.  All families operate on cleaned
uppercase A-Z strings and round-trip exactly (no letters are folded, so
``decrypt(encrypt(pt)) == clean(pt)``).

Tableau conventions (letter value ``x`` = ``ord(x) - 65``):

* Vigenere        : c = (p + k) mod 26           (Kahn, *The Codebreakers*)
* Beaufort        : c = (k - p) mod 26           (self-reciprocal)
* Variant Beaufort: c = (p - k) mod 26           (Vigenere decryption used to encrypt)
* Gronsfeld       : Vigenere with a digit key (shifts 0-9)
* Porta           : 13 reciprocal alphabets keyed by letter pairs (della Porta;
                    ACA "Porta"); self-reciprocal
* Autokey         : keystream = primer ++ (plaintext | ciphertext)
                    (Wikipedia, "Autokey cipher": text-autokey vs key-autokey)
* Running key     : Vigenere whose key is a long non-repeating text passage

These mirror the shift conventions already used in ``analysis/polyalphabetic``
(``encode_values`` / ``decode_values``); the tableau helpers are re-implemented
here so ``ciphers/`` stays self-contained.
"""

from __future__ import annotations

import random
from typing import Any

from ciphers.family_base import FamilyCipher
from ciphers.textutil import clean, random_keyword

_A = ord("A")


def _vals(text: str) -> list[int]:
    return [ord(c) - _A for c in text]


def _chars(vals: list[int]) -> str:
    return "".join(chr(v % 26 + _A) for v in vals)


# ---------------------------------------------------------------------------
# Periodic Vigenere family
# ---------------------------------------------------------------------------

def _apply_periodic(text: str, shifts: list[int], variant: str, *, decrypt: bool) -> str:
    if not shifts:
        raise ValueError("periodic key is empty")
    out: list[int] = []
    for i, v in enumerate(_vals(text)):
        k = shifts[i % len(shifts)]
        if variant == "beaufort":
            # self-reciprocal: encrypt and decrypt are identical
            r = (k - v) % 26
        elif variant == "variant_beaufort":
            r = (v + k) % 26 if decrypt else (v - k) % 26
        else:  # vigenere / gronsfeld
            r = (v - k) % 26 if decrypt else (v + k) % 26
        out.append(r)
    return _chars(out)


class _PeriodicCipher(FamilyCipher):
    """Shared machinery for the periodic-key polyalphabetic ciphers."""

    variant = "vigenere"

    def encrypt(self, plaintext: str, key: str) -> str:
        return _apply_periodic(clean(plaintext), self._shifts(key), self.variant, decrypt=False)

    def decrypt(self, ciphertext: str, key: str) -> str:
        return _apply_periodic(clean(ciphertext), self._shifts(key), self.variant, decrypt=True)

    def _shifts(self, key: str) -> list[int]:
        letters = clean(key)
        if not letters:
            raise ValueError("key must contain A-Z letters")
        return _vals(letters)

    def random_key(self, rng: random.Random, *, period: int | None = None, **_: Any) -> str:
        if period is None:
            period = rng.randint(4, 8)
        return "".join(chr(rng.randrange(26) + _A) for _ in range(period))

    def describe_key(self, key: str) -> str:
        letters = clean(key)
        return f"{self.variant} keyword={letters!r} (period {len(letters)})"

    def key_space_size(self, *, period: int = 6, **_: Any) -> int:
        return 26 ** period


class VigenereCipher(_PeriodicCipher):
    """Vigenere: c = (p + k) mod 26."""

    name = "vigenere"
    variant = "vigenere"


class BeaufortCipher(_PeriodicCipher):
    """Beaufort: c = (k - p) mod 26; self-reciprocal."""

    name = "beaufort"
    variant = "beaufort"


class VariantBeaufortCipher(_PeriodicCipher):
    """Variant Beaufort: c = (p - k) mod 26."""

    name = "variant_beaufort"
    variant = "variant_beaufort"


class GronsfeldCipher(_PeriodicCipher):
    """Gronsfeld: Vigenere driven by a numeric (0-9) key.

    Key is a string of decimal digits; each digit is the shift for its phase.
    """

    name = "gronsfeld"
    variant = "gronsfeld"

    def _shifts(self, key: str) -> list[int]:
        digits = [ch for ch in str(key) if ch.isdigit()]
        if not digits:
            raise ValueError("Gronsfeld key must contain digits 0-9")
        return [int(ch) for ch in digits]

    def random_key(self, rng: random.Random, *, period: int | None = None, **_: Any) -> str:
        if period is None:
            period = rng.randint(4, 8)
        return "".join(str(rng.randrange(10)) for _ in range(period))

    def describe_key(self, key: str) -> str:
        digits = "".join(ch for ch in str(key) if ch.isdigit())
        return f"gronsfeld digits={digits!r} (period {len(digits)})"

    def key_space_size(self, *, period: int = 6, **_: Any) -> int:
        return 10 ** period


# ---------------------------------------------------------------------------
# Porta
# ---------------------------------------------------------------------------

def _porta_char(p_idx: int, key_idx: int) -> int:
    """One della-Porta reciprocal substitution (13 alphabets, keyed by pairs)."""
    row = key_idx // 2  # 0..12
    if p_idx < 13:
        return 13 + ((p_idx + row) % 13)
    return ((p_idx - 13) - row) % 13


class PortaCipher(FamilyCipher):
    """Porta cipher (della Porta / ACA "Porta").

    13 reciprocal cipher alphabets selected by key-letter pairs (A,B -> row 0;
    C,D -> row 1; ...; Y,Z -> row 12).  The substitution swaps the A-M and N-Z
    halves, which makes the cipher self-reciprocal: ``encrypt == decrypt``.
    """

    name = "porta"

    def encrypt(self, plaintext: str, key: str) -> str:
        shifts = self._shifts(key)
        out: list[int] = []
        for i, v in enumerate(_vals(clean(plaintext))):
            out.append(_porta_char(v, shifts[i % len(shifts)]))
        return _chars(out)

    def decrypt(self, ciphertext: str, key: str) -> str:
        # self-reciprocal
        return self.encrypt(ciphertext, key)

    def _shifts(self, key: str) -> list[int]:
        letters = clean(key)
        if not letters:
            raise ValueError("key must contain A-Z letters")
        return _vals(letters)

    def random_key(self, rng: random.Random, *, period: int | None = None, **_: Any) -> str:
        if period is None:
            period = rng.randint(4, 8)
        return "".join(chr(rng.randrange(26) + _A) for _ in range(period))

    def describe_key(self, key: str) -> str:
        letters = clean(key)
        return f"porta keyword={letters!r} (period {len(letters)})"

    def key_space_size(self, *, period: int = 6, **_: Any) -> int:
        # only the pair (row) matters, so effectively 13 alphabets per phase
        return 13 ** period


# ---------------------------------------------------------------------------
# Autokey (text-autokey and key/ciphertext-autokey)
# ---------------------------------------------------------------------------

class AutokeyCipher(FamilyCipher):
    """Autokey cipher (Wikipedia, "Autokey cipher").

    A short *primer* seeds the keystream, which is then extended by the message
    itself.  Two modes:

    * ``mode="text"``  -- text/plaintext-autokey: keystream = primer ++ plaintext.
    * ``mode="key"``   -- key/ciphertext-autokey: keystream = primer ++ ciphertext.

    Both use a Vigenere tableau (c = p + k mod 26) and round-trip exactly.
    The key is the primer keyword string; the mode is fixed per instance.
    """

    def __init__(self, mode: str = "text") -> None:
        if mode not in ("text", "key"):
            raise ValueError("mode must be 'text' or 'key'")
        self.mode = mode
        self.name = "autokey_text" if mode == "text" else "autokey_key"

    def encrypt(self, plaintext: str, key: str) -> str:
        primer = _vals(self._primer(key))
        pt = _vals(clean(plaintext))
        out: list[int] = []
        for i, p in enumerate(pt):
            if i < len(primer):
                k = primer[i]
            elif self.mode == "text":
                k = pt[i - len(primer)]
            else:  # ciphertext autokey
                k = out[i - len(primer)]
            out.append((p + k) % 26)
        return _chars(out)

    def decrypt(self, ciphertext: str, key: str) -> str:
        primer = _vals(self._primer(key))
        ct = _vals(clean(ciphertext))
        plain: list[int] = []
        for i, c in enumerate(ct):
            if i < len(primer):
                k = primer[i]
            elif self.mode == "text":
                k = plain[i - len(primer)]
            else:  # ciphertext autokey: keystream comes from known ciphertext
                k = ct[i - len(primer)]
            plain.append((c - k) % 26)
        return _chars(plain)

    def _primer(self, key: str) -> str:
        letters = clean(key)
        if not letters:
            raise ValueError("autokey primer must contain A-Z letters")
        return letters

    def random_key(self, rng: random.Random, *, period: int | None = None, **_: Any) -> str:
        if period is None:
            period = rng.randint(3, 7)
        return "".join(chr(rng.randrange(26) + _A) for _ in range(period))

    def describe_key(self, key: str) -> str:
        letters = clean(key)
        label = "text-autokey" if self.mode == "text" else "key-autokey"
        return f"{label} primer={letters!r} (length {len(letters)})"

    def key_space_size(self, *, period: int = 5, **_: Any) -> int:
        return 26 ** period


# ---------------------------------------------------------------------------
# Running key
# ---------------------------------------------------------------------------

class RunningKeyCipher(FamilyCipher):
    """Running-key cipher: Vigenere with a long non-repeating key text.

    The key is another passage of text (Slice C supplies a real corpus
    passage; :meth:`random_key` synthesises random letters for standalone use).
    The (cleaned) key must be at least as long as the (cleaned) plaintext.
    c = (p + k) mod 26.
    """

    name = "running_key"

    def encrypt(self, plaintext: str, key: str) -> str:
        pt = _vals(clean(plaintext))
        ks = _vals(clean(key))
        if len(ks) < len(pt):
            raise ValueError("running key must be at least as long as the plaintext")
        return _chars([(p + ks[i]) % 26 for i, p in enumerate(pt)])

    def decrypt(self, ciphertext: str, key: str) -> str:
        ct = _vals(clean(ciphertext))
        ks = _vals(clean(key))
        if len(ks) < len(ct):
            raise ValueError("running key must be at least as long as the ciphertext")
        return _chars([(c - ks[i]) % 26 for i, c in enumerate(ct)])

    def random_key(self, rng: random.Random, *, length: int = 200, **_: Any) -> str:
        return "".join(chr(rng.randrange(26) + _A) for _ in range(max(1, length)))

    def describe_key(self, key: str) -> str:
        letters = clean(key)
        preview = letters[:24] + ("..." if len(letters) > 24 else "")
        return f"running-key text length={len(letters)} preview={preview!r}"

    def key_space_size(self, *, length: int = 200, **_: Any) -> int:
        return 26 ** max(1, length)
