"""Tests for the Slice-A benchmark cipher families (src/ciphers/).

Covers, per the generator spec:
  * round-trip per family (decrypt(encrypt(pt)) reproduces the prepared text),
  * known-answer vectors where a canonical example exists
    (Playfair, Bifid, columnar "ZEBRAS", Vigenere "LEMON", Hill 2x2),
  * key determinism (same seed -> identical key),
  * key_space_size sanity.
"""

from __future__ import annotations

import random

import pytest

from ciphers.textutil import clean, clean_alnum, clean_ij
from ciphers.polyalphabetic import (
    AutokeyCipher,
    BeaufortCipher,
    GronsfeldCipher,
    PortaCipher,
    RunningKeyCipher,
    VariantBeaufortCipher,
    VigenereCipher,
)
from ciphers.playfair import FourSquareCipher, PlayfairCipher, TwoSquareCipher
from ciphers.fractionation import (
    ADFGVXCipher,
    ADFGXCipher,
    BifidCipher,
    TrifidCipher,
)
from ciphers.transposition import (
    AmscoCipher,
    CadenusCipher,
    ColumnarCipher,
    MyszkowskiCipher,
    NihilistTranspositionCipher,
    RailfenceCipher,
    RedefenceCipher,
    RouteCipher,
)
from ciphers.numeric import (
    Hill2x2Cipher,
    NihilistSubstitutionCipher,
    StraddlingCheckerboardCipher,
)
from ciphers.encodings import (
    A1Z26Cipher,
    Base32Cipher,
    Base64Cipher,
    BaconianCipher,
    BinaryCipher,
    HexCipher,
    MorseCipher,
    Rot47Cipher,
    TapCodeCipher,
)

PT = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGANDTHEIDLEDUCK"
PT_DIGITS = "MEETATNOONBEHIND1200MAINSTREET"


def _expected_prepared(cipher, key, plaintext, *, folder=clean):
    """The text a round-trip should reproduce for ``cipher`` (handles padding)."""
    prep = getattr(cipher, "prepare", None)
    if prep is not None:
        try:
            return prep(plaintext, key)
        except TypeError:
            return prep(plaintext)
    return folder(plaintext)


# ---------------------------------------------------------------------------
# Polyalphabetic
# ---------------------------------------------------------------------------

POLY_CASES = [
    (VigenereCipher(), "LEMON"),
    (BeaufortCipher(), "FORTIFY"),
    (VariantBeaufortCipher(), "SECRET"),
    (GronsfeldCipher(), "31415"),
    (PortaCipher(), "CIPHER"),
    (AutokeyCipher("text"), "PRIMER"),
    (AutokeyCipher("key"), "PRIMER"),
]


@pytest.mark.parametrize("cipher,key", POLY_CASES, ids=lambda c: getattr(c, "name", str(c)))
def test_polyalphabetic_round_trip(cipher, key):
    ct = cipher.encrypt(PT, key)
    assert cipher.decrypt(ct, key) == clean(PT)
    assert set(ct) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_running_key_round_trip():
    cipher = RunningKeyCipher()
    key = cipher.random_key(random.Random(9), length=len(clean(PT)) + 5)
    ct = cipher.encrypt(PT, key)
    assert cipher.decrypt(ct, key) == clean(PT)


def test_running_key_requires_long_key():
    with pytest.raises(ValueError):
        RunningKeyCipher().encrypt(PT, "SHORT")


def test_vigenere_known_answer():
    # Wikipedia Vigenere example.
    assert VigenereCipher().encrypt("ATTACKATDAWN", "LEMON") == "LXFOPVEFRNHR"


def test_beaufort_and_porta_self_reciprocal():
    for cipher, key in [(BeaufortCipher(), "KEY"), (PortaCipher(), "KEY")]:
        ct = cipher.encrypt(PT, key)
        # applying encrypt again recovers the plaintext (involution)
        assert cipher.encrypt(ct, key) == clean(PT)


# ---------------------------------------------------------------------------
# Playfair family
# ---------------------------------------------------------------------------

def test_playfair_known_answer():
    pf = PlayfairCipher()
    ct = pf.encrypt("HIDETHEGOLDINTHETREESTUMP", "PLAYFAIREXAMPLE")
    assert ct == "BMODZBXDNABEKUDMUIXMMOUVIF"


@pytest.mark.parametrize(
    "cipher,key",
    [
        (PlayfairCipher(), "MONARCHY"),
        (TwoSquareCipher(), ("EXAMPLE", "KEYWORD")),
        (FourSquareCipher(), ("EXAMPLE", "KEYWORD")),
    ],
    ids=lambda c: getattr(c, "name", str(c)),
)
def test_playfair_family_round_trip(cipher, key):
    ct = cipher.encrypt(PT, key)
    assert cipher.decrypt(ct, key) == cipher.prepare(PT)


def test_two_square_self_reciprocal():
    ts = TwoSquareCipher()
    key = ("ALPHA", "OMEGA")
    ct = ts.encrypt(PT, key)
    assert ts.encrypt(ct, key) == ts.prepare(PT)


# ---------------------------------------------------------------------------
# Fractionation
# ---------------------------------------------------------------------------

def test_bifid_known_answer():
    # Wikipedia Bifid example (custom mixed square, whole-message period).
    square = "BGWKZQPNDSIOAXEFCLUMTHYVR"
    assert BifidCipher(period=0).encrypt("FLEEATONCE", square) == "UAEOLWRINS"


@pytest.mark.parametrize("period", [0, 5, 7])
def test_bifid_round_trip(period):
    cipher = BifidCipher(period=period)
    key = cipher.random_key(random.Random(11))
    ct = cipher.encrypt(PT, key)
    assert cipher.decrypt(ct, key) == clean_ij(PT)


@pytest.mark.parametrize("period", [0, 5])
def test_trifid_round_trip(period):
    cipher = TrifidCipher(period=period)
    key = cipher.random_key(random.Random(12))
    ct = cipher.encrypt(PT, key)
    assert cipher.decrypt(ct, key) == cipher._clean(PT)


def test_adfgx_round_trip():
    cipher = ADFGXCipher()
    key = cipher.random_key(random.Random(13))
    ct = cipher.encrypt(PT, key)
    assert set(ct) <= set("ADFGX")
    assert cipher.decrypt(ct, key) == clean_ij(PT)


def test_adfgvx_round_trip():
    cipher = ADFGVXCipher()
    key = cipher.random_key(random.Random(14))
    ct = cipher.encrypt(PT_DIGITS, key)
    assert set(ct) <= set("ADFGVX")
    assert cipher.decrypt(ct, key) == clean_alnum(PT_DIGITS)


# ---------------------------------------------------------------------------
# Transposition
# ---------------------------------------------------------------------------

def test_columnar_known_answer():
    # Wikipedia complete columnar example.
    ct = ColumnarCipher().encrypt("WEAREDISCOVEREDFLEEATONCEQKJEU", "ZEBRAS")
    assert ct == "EVLNEACDTKESEAQROFOJDEECUWIREE"


TRANSPOSITION_CIPHERS = [
    ColumnarCipher(),
    RailfenceCipher(),
    RedefenceCipher(),
    RouteCipher(),
    AmscoCipher(),
    MyszkowskiCipher(),
    CadenusCipher(),
    NihilistTranspositionCipher(),
]


@pytest.mark.parametrize("cipher", TRANSPOSITION_CIPHERS, ids=lambda c: c.name)
def test_transposition_round_trip(cipher):
    key = cipher.random_key(random.Random(15))
    ct = cipher.encrypt(PT, key)
    assert cipher.decrypt(ct, key) == _expected_prepared(cipher, key, PT)


def test_railfence_matches_redefence_identity_order():
    rails = 4
    rf = RailfenceCipher().encrypt(PT, rails)
    re = RedefenceCipher().encrypt(PT, (rails, tuple(range(rails))))
    assert rf == re


def test_incomplete_and_complete_columnar():
    col = ColumnarCipher()
    # complete: length divisible by key length
    complete = "ABCDEFGHIJKL"  # 12 / key len 4 -> exact
    assert col.decrypt(col.encrypt(complete, "DACB"), "DACB") == complete
    # incomplete: not divisible
    incomplete = "ABCDEFGHIJKLM"
    assert col.decrypt(col.encrypt(incomplete, "DACB"), "DACB") == incomplete


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------

def test_nihilist_substitution_round_trip():
    cipher = NihilistSubstitutionCipher()
    key = cipher.random_key(random.Random(21))
    ct = cipher.encrypt(PT, key)
    assert all(tok.isdigit() for tok in ct.split())
    assert cipher.decrypt(ct, key) == clean_ij(PT)


def test_straddling_checkerboard_round_trip():
    cipher = StraddlingCheckerboardCipher()
    key = cipher.random_key(random.Random(22))
    ct = cipher.encrypt(PT, key)
    assert ct.isdigit()
    assert cipher.decrypt(ct, key) == clean_ij(PT)


def test_hill_round_trip_even_and_odd():
    cipher = Hill2x2Cipher()
    key = cipher.random_key(random.Random(23))
    even = "FOURSCORE"[:8]
    assert cipher.decrypt(cipher.encrypt(even, key), key) == even
    # odd length pads with X
    odd = "HELLOWORLD"  # 10 already even; use 9-letter word
    nine = "CRYPTANAL"
    padded = nine + "X"
    assert cipher.decrypt(cipher.encrypt(nine, key), key) == padded


def test_hill_known_answer():
    # Hand-computable: matrix [[3,3],[2,5]], "HI" -> "TC".
    assert Hill2x2Cipher().encrypt("HI", (3, 3, 2, 5)) == "TC"


def test_hill_random_key_is_invertible():
    cipher = Hill2x2Cipher()
    for seed in range(20):
        a, b, c, d = cipher.random_key(random.Random(seed))
        from math import gcd
        assert gcd((a * d - b * c) % 26, 26) == 1


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------

ENCODINGS = [
    BaconianCipher(),
    A1Z26Cipher(),
    MorseCipher(),
    Base64Cipher(),
    Base32Cipher(),
    HexCipher(),
    BinaryCipher(),
    Rot47Cipher(),
    TapCodeCipher(),
]


@pytest.mark.parametrize("cipher", ENCODINGS, ids=lambda c: c.name)
def test_encoding_round_trip(cipher):
    key = cipher.random_key(random.Random(31))
    ct = cipher.encrypt(PT, key)
    expected = clean(PT).replace("K", "C") if cipher.name == "tap_code" else clean(PT)
    assert cipher.decrypt(ct, key) == expected


def test_rot47_self_reciprocal():
    r = Rot47Cipher()
    ct = r.encrypt(PT)
    assert r.decrypt(ct, None) == clean(PT)
    # the underlying rotation is its own inverse (47 * 2 == 94 == identity)
    assert r._rot(ct) == clean(PT)


# ---------------------------------------------------------------------------
# Key determinism (same seed -> identical key)
# ---------------------------------------------------------------------------

ALL_FAMILIES = (
    [c for c, _ in POLY_CASES]
    + [RunningKeyCipher(), PlayfairCipher(), TwoSquareCipher(), FourSquareCipher()]
    + [BifidCipher(), TrifidCipher(), ADFGXCipher(), ADFGVXCipher()]
    + TRANSPOSITION_CIPHERS
    + [NihilistSubstitutionCipher(), StraddlingCheckerboardCipher(), Hill2x2Cipher()]
    + ENCODINGS
)


@pytest.mark.parametrize("cipher", ALL_FAMILIES, ids=lambda c: c.name)
def test_key_determinism(cipher):
    k1 = cipher.random_key(random.Random(1234))
    k2 = cipher.random_key(random.Random(1234))
    assert k1 == k2
    # describe_key must not raise and returns a non-empty string
    assert isinstance(cipher.describe_key(k1), str)
    assert cipher.describe_key(k1)


# ---------------------------------------------------------------------------
# key_space_size sanity
# ---------------------------------------------------------------------------

def test_key_space_size_values():
    from math import factorial

    assert VigenereCipher().key_space_size(period=6) == 26 ** 6
    assert GronsfeldCipher().key_space_size(period=6) == 10 ** 6
    assert PortaCipher().key_space_size(period=6) == 13 ** 6
    assert PlayfairCipher().key_space_size() == factorial(25)
    assert BifidCipher().key_space_size() == factorial(25)
    assert TrifidCipher().key_space_size() == factorial(27)
    assert Hill2x2Cipher().key_space_size() == 157248
    assert ColumnarCipher().key_space_size(columns=7) == factorial(7)
    # every family returns a positive int
    for cipher in ALL_FAMILIES:
        assert isinstance(cipher.key_space_size(), int)
        assert cipher.key_space_size() >= 1


def _hill_prepare(pt):
    from ciphers.textutil import clean_ij
    s = clean_ij(pt)
    if len(s) % 2:
        s += "X"
    return s


def test_hill2x2_roundtrip_with_j_in_ciphertext():
    # Regression (found by the benchmark generator, 2026-07-15): Hill decrypt
    # must NOT fold J->I on the ciphertext (ciphertext spans full A-Z). Sweep
    # keys until one produces a J in the ciphertext, then require round-trip.
    import random as _r
    from ciphers.numeric import Hill2x2Cipher
    h = Hill2x2Cipher()
    pt = "THEQUICKBROWNFOXJUMPS"
    expected = _hill_prepare(pt)
    saw_j = False
    for seed in range(200):
        key = h.random_key(_r.Random(seed))
        ct = h.encrypt(pt, key)
        assert h.decrypt(ct, key) == expected, (seed, key, ct)
        if "J" in ct:
            saw_j = True
    assert saw_j, "test did not exercise a J-containing ciphertext"
