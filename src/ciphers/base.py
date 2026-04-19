from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from models.alphabet import Alphabet


class Cipher(ABC):
    """Abstract base for all cipher implementations."""

    @abstractmethod
    def encrypt(self, plaintext: list[int], key: Any, alphabet: Alphabet) -> list[int]:
        ...

    @abstractmethod
    def decrypt(self, ciphertext: list[int], key: Any, alphabet: Alphabet) -> list[int]:
        ...

    @abstractmethod
    def key_space_size(self, alphabet: Alphabet) -> int:
        ...
