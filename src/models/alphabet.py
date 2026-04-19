from __future__ import annotations

from typing import NamedTuple


class Token(NamedTuple):
    id: int
    symbol: str


class Alphabet:
    """Bidirectional mapping between display symbols and integer token IDs.

    Supports both single-character alphabets (e.g. standard A-Z) and
    multi-character symbol labels (e.g. "SYM_1 SYM_2" from OCR output).
    """

    def __init__(self, symbols: list[str]) -> None:
        if len(symbols) != len(set(symbols)):
            raise ValueError("Alphabet symbols must be unique")
        if not symbols:
            raise ValueError("Alphabet must contain at least one symbol")
        self._symbols = list(symbols)
        self._id_to_symbol: dict[int, str] = {i: s for i, s in enumerate(symbols)}
        self._symbol_to_id: dict[str, int] = {s: i for i, s in enumerate(symbols)}
        # True if any symbol is multi-character (space-delimited OCR output)
        self._multisym = any(len(s) > 1 for s in symbols)

    @property
    def size(self) -> int:
        return len(self._symbols)

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def symbol_for(self, token_id: int) -> str:
        return self._id_to_symbol[token_id]

    def id_for(self, symbol: str) -> int:
        return self._symbol_to_id[symbol]

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._symbol_to_id

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs.

        For single-char alphabets, iterates characters.
        For multi-char symbol alphabets, splits on whitespace.
        Unknown symbols are silently skipped.
        """
        if self._multisym:
            parts = text.split()
        else:
            parts = list(text)
        return [self._symbol_to_id[p] for p in parts if p in self._symbol_to_id]

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to a display string."""
        sep = " " if self._multisym else ""
        return sep.join(self._id_to_symbol[i] for i in ids if i in self._id_to_symbol)

    @classmethod
    def from_text(
        cls,
        text: str,
        multisym: bool = False,
        ignore_chars: set[str] | None = None,
    ) -> Alphabet:
        """Auto-detect alphabet from ciphertext.

        If multisym is True, treats whitespace-separated tokens as symbols.
        Otherwise, each unique character becomes a symbol.
        Characters in ignore_chars are excluded from the alphabet.
        """
        if ignore_chars is None:
            ignore_chars = set()
        if multisym:
            parts = text.split()
        else:
            parts = list(text)
        seen: set[str] = set()
        symbols: list[str] = []
        for p in parts:
            if p not in seen and p not in ignore_chars:
                seen.add(p)
                symbols.append(p)
        return cls(symbols)

    @classmethod
    def standard_english(cls) -> Alphabet:
        return cls(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        if self.size <= 10:
            return f"Alphabet({self._symbols})"
        return f"Alphabet({self._symbols[:5]}... {self.size} symbols)"
