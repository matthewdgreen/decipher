"""Shared text helpers for the Slice-A classical cipher families.

Small, dependency-free utilities: alphabet cleaning, keyed alphabets, keyed
Polybius squares (I/J merged for 5x5, full 6x6 for ADFGVX), random keyword
generation, and modular-arithmetic helpers used by Hill.  Kept independent of
``analysis/`` so the cipher families are self-contained.
"""

from __future__ import annotations

import random
from math import gcd

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"


def clean(text: str) -> str:
    """Uppercase ``text`` and keep only A-Z letters."""
    return "".join(ch for ch in text.upper() if "A" <= ch <= "Z")


def clean_ij(text: str) -> str:
    """Clean to A-Z with the classical I/J merge (J -> I)."""
    return clean(text).replace("J", "I")


def clean_alnum(text: str) -> str:
    """Uppercase ``text`` and keep A-Z letters and 0-9 digits."""
    return "".join(ch for ch in text.upper() if ("A" <= ch <= "Z") or ch.isdigit())


def keyed_alphabet(keyword: str, *, base: str = ALPHABET) -> str:
    """Return a keyed alphabet: unique keyword letters, then the rest of ``base``.

    Standard construction used by keyword-substitution and keyed-square
    ciphers: take the letters of ``keyword`` in order (dropping repeats and any
    letter not in ``base``), then append the remaining ``base`` letters in
    order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for ch in keyword.upper() + base:
        if ch in base and ch not in seen:
            seen.add(ch)
            out.append(ch)
    return "".join(out)


def keyed_square_25(keyword: str) -> str:
    """25-letter keyed alphabet for a 5x5 Polybius square (I/J merged, J->I)."""
    base = ALPHABET.replace("J", "")  # 25 letters, I absorbs J
    seen: set[str] = set()
    out: list[str] = []
    for ch in keyword.upper().replace("J", "I") + base:
        if ch in base and ch not in seen:
            seen.add(ch)
            out.append(ch)
    if len(out) != 25:  # pragma: no cover - defensive
        raise ValueError("keyed 5x5 square must contain 25 distinct letters")
    return "".join(out)


def keyed_square_36(keyword: str) -> str:
    """36-symbol keyed alphabet for a 6x6 Polybius square (A-Z + 0-9)."""
    base = ALPHABET + DIGITS
    seen: set[str] = set()
    out: list[str] = []
    for ch in keyword.upper() + base:
        if ch in base and ch not in seen:
            seen.add(ch)
            out.append(ch)
    if len(out) != 36:  # pragma: no cover - defensive
        raise ValueError("keyed 6x6 square must contain 36 distinct symbols")
    return "".join(out)


def random_keyword(rng: random.Random, *, min_len: int = 5, max_len: int = 9) -> str:
    """Random A-Z keyword of a length in ``[min_len, max_len]`` (distinct letters)."""
    length = rng.randint(min_len, max_len)
    length = min(length, 26)
    letters = list(ALPHABET)
    rng.shuffle(letters)
    return "".join(letters[:length])


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm: return (g, x, y) with a*x + b*y = g."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def modinv(a: int, m: int) -> int:
    """Modular inverse of ``a`` mod ``m`` (raises if not coprime)."""
    a %= m
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse mod {m}")
    return x % m


def coprime(a: int, m: int) -> bool:
    return gcd(a % m, m) == 1
