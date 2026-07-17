"""Inline-ciphertext parsing for ``investigation_start`` (Part 5.1).

No filesystem paths, no benchmark ids, no benchmark splits (GT-1): the only
entry point takes an inline string plus a format hint and returns a
``CipherText``. ``canonical`` reuses the benchmark transcription parser;
``letters`` reproduces the ``cmd_crack`` non-canonical path exactly; ``auto``
detects canonical S-token transcriptions by shape.
"""
from __future__ import annotations

import re

from models.cipher_text import CipherText

# The canonical S-token shape (S001, S002, ...): three digits after an S.
_CANONICAL_RE = re.compile(r"\bS\d{3}\b")


def detect_format(text: str) -> str:
    """Return 'canonical' iff the text carries canonical S-tokens, else 'letters'."""
    return "canonical" if _CANONICAL_RE.search(text) else "letters"


def build_cipher_text(text: str, fmt: str) -> CipherText:
    """Parse inline ciphertext into a ``CipherText`` (never a path).

    ``fmt`` is one of 'auto' | 'letters' | 'canonical'. Raises on parse
    failure; the caller maps that to a typed ``parse_failed`` result.
    """
    from benchmark.loader import parse_canonical_transcription
    from models.alphabet import Alphabet

    resolved = detect_format(text) if fmt == "auto" else fmt
    if resolved == "canonical":
        return parse_canonical_transcription(text)
    # 'letters' — exactly the cmd_crack non-canonical path.
    ignore = {" ", "\t", "\n", "\r"}
    alphabet = Alphabet.from_text(text, ignore_chars=ignore)
    clean = " ".join(text.split())
    return CipherText(raw=clean, alphabet=alphabet, source="mcp", separator=" ")
