"""Detection-tier encodings (not cryptanalysis, but INV must triage them).

Baconian, A1Z26, Morse, Base64, Base32, Hex, Binary, ROT47, Tap code.

These have no meaningful secret key; the spec asks for "encrypt only + a decode
for the round-trip test".  Each therefore exposes ``encrypt``/``decrypt`` (with
``key`` ignored), a trivial ``random_key`` returning ``None``, and a
``describe_key`` naming the scheme.  ``encrypt`` operates on cleaned uppercase
A-Z text (Baconian uses the 26-letter distinct variant; Tap code folds K->C);
``decrypt(encrypt(pt))`` reproduces that cleaned text.
"""

from __future__ import annotations

import base64
import binascii
import random
from typing import Any

from ciphers.family_base import FamilyCipher
from ciphers.textutil import clean


class _Encoding(FamilyCipher):
    """Base for keyless detection-tier encodings."""

    def random_key(self, rng: random.Random, **_: Any) -> None:
        return None

    def describe_key(self, key: Any = None) -> str:
        return f"{self.name} (keyless encoding)"

    def key_space_size(self, **_: Any) -> int:
        return 1


class BaconianCipher(_Encoding):
    """Baconian cipher, 26-letter distinct variant: letter -> 5 A/B bits."""

    name = "baconian"

    def encrypt(self, plaintext: str, key: Any = None) -> str:
        out: list[str] = []
        for ch in clean(plaintext):
            bits = format(ord(ch) - 65, "05b")
            out.append(bits.replace("0", "A").replace("1", "B"))
        return "".join(out)

    def decrypt(self, ciphertext: str, key: Any = None) -> str:
        code = "".join(ch for ch in ciphertext.upper() if ch in "AB")
        out: list[str] = []
        for i in range(0, len(code) - 4, 5):
            bits = code[i:i + 5].replace("A", "0").replace("B", "1")
            out.append(chr(int(bits, 2) + 65))
        return "".join(out)


class A1Z26Cipher(_Encoding):
    """A1Z26: A=1 .. Z=26, space-separated numbers."""

    name = "a1z26"

    def encrypt(self, plaintext: str, key: Any = None) -> str:
        return " ".join(str(ord(ch) - 64) for ch in clean(plaintext))

    def decrypt(self, ciphertext: str, key: Any = None) -> str:
        return "".join(chr(int(tok) + 64) for tok in ciphertext.split())


_MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
}
_MORSE_INV = {v: k for k, v in _MORSE.items()}


class MorseCipher(_Encoding):
    """International Morse code; letters separated by spaces."""

    name = "morse"

    def encrypt(self, plaintext: str, key: Any = None) -> str:
        return " ".join(_MORSE[ch] for ch in clean(plaintext))

    def decrypt(self, ciphertext: str, key: Any = None) -> str:
        return "".join(_MORSE_INV[tok] for tok in ciphertext.split() if tok in _MORSE_INV)


class _ByteEncoding(_Encoding):
    """Shared base for byte-oriented encodings (Base64/Base32/Hex/Binary)."""

    def _encode_bytes(self, data: bytes) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _decode_bytes(self, text: str) -> bytes:  # pragma: no cover - overridden
        raise NotImplementedError

    def encrypt(self, plaintext: str, key: Any = None) -> str:
        return self._encode_bytes(clean(plaintext).encode("ascii"))

    def decrypt(self, ciphertext: str, key: Any = None) -> str:
        return self._decode_bytes(ciphertext).decode("ascii")


class Base64Cipher(_ByteEncoding):
    """Standard Base64 of the ASCII bytes of the cleaned plaintext."""

    name = "base64"

    def _encode_bytes(self, data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    def _decode_bytes(self, text: str) -> bytes:
        return base64.b64decode(text)


class Base32Cipher(_ByteEncoding):
    """Standard Base32."""

    name = "base32"

    def _encode_bytes(self, data: bytes) -> str:
        return base64.b32encode(data).decode("ascii")

    def _decode_bytes(self, text: str) -> bytes:
        return base64.b32decode(text)


class HexCipher(_ByteEncoding):
    """Hexadecimal (base16) encoding."""

    name = "hex"

    def _encode_bytes(self, data: bytes) -> str:
        return binascii.hexlify(data).decode("ascii").upper()

    def _decode_bytes(self, text: str) -> bytes:
        return binascii.unhexlify("".join(text.split()))


class BinaryCipher(_ByteEncoding):
    """8-bit binary encoding, one byte per character, no separator."""

    name = "binary"

    def _encode_bytes(self, data: bytes) -> str:
        return "".join(format(b, "08b") for b in data)

    def _decode_bytes(self, text: str) -> bytes:
        bits = "".join(ch for ch in text if ch in "01")
        return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


class Rot47Cipher(_Encoding):
    """ROT47: rotate printable ASCII 33-126 by 47; self-reciprocal."""

    name = "rot47"

    def _rot(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            o = ord(ch)
            if 33 <= o <= 126:
                out.append(chr(33 + (o - 33 + 47) % 94))
            else:
                out.append(ch)
        return "".join(out)

    def encrypt(self, plaintext: str, key: Any = None) -> str:
        return self._rot(clean(plaintext))

    def decrypt(self, ciphertext: str, key: Any = None) -> str:
        return self._rot(ciphertext)


# Tap code: 5x5 square with K folded to C (classic POW tap code); K omitted.
_TAP_SQUARE = "ABCDEFGHIJLMNOPQRSTUVWXYZ"  # 25 letters, K omitted (K taps as C)


class TapCodeCipher(_Encoding):
    """Tap code: 5x5 grid (K folded to C); each letter -> row/col tap counts.

    Output is space-separated ``<row><col>`` digit pairs (1-5 each).
    """

    name = "tap_code"

    def _pos(self) -> dict[str, tuple[int, int]]:
        return {ch: (i // 5 + 1, i % 5 + 1) for i, ch in enumerate(_TAP_SQUARE)}

    def encrypt(self, plaintext: str, key: Any = None) -> str:
        pos = self._pos()
        text = clean(plaintext).replace("K", "C")
        return " ".join(f"{pos[ch][0]}{pos[ch][1]}" for ch in text)

    def decrypt(self, ciphertext: str, key: Any = None) -> str:
        inv = {(i // 5 + 1, i % 5 + 1): ch for i, ch in enumerate(_TAP_SQUARE)}
        out: list[str] = []
        for tok in ciphertext.split():
            if len(tok) == 2 and tok.isdigit():
                out.append(inv[(int(tok[0]), int(tok[1]))])
        return "".join(out)
