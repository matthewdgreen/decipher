"""Common contract for Slice-A benchmark cipher families.

The classical ciphers implemented under ``src/ciphers/`` (polyalphabetic,
Playfair, fractionation, transposition, numeric, encodings) operate most
naturally on cleaned uppercase A-Z strings rather than the ``list[int]`` /
``Alphabet`` token model used by :class:`ciphers.base.Cipher`.  Digraph
ciphers (Playfair), fractionation ciphers (Bifid/ADFGX), numeric ciphers
(Nihilist, Hill), and detection-tier encodings all fractionate, regroup, or
change the symbol space in ways that make the token-id-over-``Alphabet`` model
awkward.  The spec therefore permits a "self-contained encrypt/decrypt over
uppercase strings with a documented interface" for these families.

Every family here honours the same lightweight contract:

* ``encrypt(plaintext, key) -> str`` and ``decrypt(ciphertext, key) -> str``
  ROUND-TRIP: ``decrypt(encrypt(pt, key), key)`` reproduces the *prepared*
  plaintext (some families pad or fold letters; each documents its own
  preparation and the resulting round-trip invariant).
* ``random_key(rng, ...)`` derives a key deterministically from an injected
  :class:`random.Random`.  No global ``random`` state is ever touched.
* ``describe_key(key) -> str`` renders a human-readable ground-truth string.
* ``key_space_size(...) -> int`` returns an order-of-magnitude key-space count.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any


class FamilyCipher(ABC):
    """Uppercase-string cipher contract shared by the Slice-A families."""

    #: Stable family identifier (see docs/inv_family_roadmap.md / Slice C).
    name: str = "family"

    @abstractmethod
    def encrypt(self, plaintext: str, key: Any) -> str:
        """Encipher cleaned uppercase ``plaintext`` under ``key``."""

    @abstractmethod
    def decrypt(self, ciphertext: str, key: Any) -> str:
        """Recover the prepared plaintext from ``ciphertext`` under ``key``."""

    @abstractmethod
    def random_key(self, rng: random.Random, **kwargs: Any) -> Any:
        """Return a key derived deterministically from ``rng``."""

    @abstractmethod
    def describe_key(self, key: Any) -> str:
        """Human-readable ground-truth description of ``key``."""

    def key_space_size(self, **kwargs: Any) -> int:  # pragma: no cover - overridden
        """Approximate size of the key space (overridden per family)."""
        raise NotImplementedError
