from __future__ import annotations

from dataclasses import dataclass, field

from models.alphabet import Alphabet


@dataclass
class CipherText:
    """Ciphertext with its associated alphabet and token encoding.

    The separator field stores the original separator string (e.g. " ")
    that was used to delimit words in the raw text. This is NOT part of
    the cipher alphabet — it represents word boundaries.
    """

    raw: str
    alphabet: Alphabet
    source: str = "manual"  # "manual", "ocr", "file"
    separator: str | None = " "  # word boundary character(s), None if no word boundaries
    tokens: list[int] = field(init=False)
    words: list[list[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.alphabet.encode(self.raw)
        self.words = self._split_words()

    def _split_words(self) -> list[list[int]]:
        """Split raw text into words using separator, encode each."""
        if self.separator is None:
            return [self.tokens]
        parts = self.raw.split(self.separator)
        result = []
        for part in parts:
            word_tokens = self.alphabet.encode(part)
            if word_tokens:
                result.append(word_tokens)
        return result

    def __len__(self) -> int:
        return len(self.tokens)

    def segment(self, start: int, end: int) -> list[int]:
        return self.tokens[start:end]

    def display(self) -> str:
        """Return the ciphertext as displayed symbols."""
        if self.separator is not None and not self.alphabet._multisym:
            # Re-insert word boundaries for display
            word_strs = [self.alphabet.decode(w) for w in self.words]
            return self.separator.join(word_strs)
        return self.alphabet.decode(self.tokens)
