from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from models.alphabet import Alphabet
from models.cipher_text import CipherText


class Session(QObject):
    """Central state for a cipher-cracking session.

    Holds the ciphertext, current key mapping, and plaintext alphabet.
    Emits Qt signals when state changes so the UI can react.
    """

    cipher_text_changed = Signal()
    key_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.cipher_text: CipherText | None = None
        self.plaintext_alphabet: Alphabet = Alphabet.standard_english()
        # key: ciphertext token ID -> plaintext token ID
        self._key: dict[int, int] = {}
        self.history: list[dict] = []

    @property
    def key(self) -> dict[int, int]:
        return dict(self._key)

    def set_cipher_text(self, ct: CipherText) -> None:
        self.cipher_text = ct
        self._key.clear()
        self.history.clear()
        self.cipher_text_changed.emit()
        self.key_changed.emit()

    def set_mapping(self, ct_id: int, pt_id: int) -> None:
        self._key[ct_id] = pt_id
        self.key_changed.emit()

    def clear_mapping(self, ct_id: int) -> None:
        self._key.pop(ct_id, None)
        self.key_changed.emit()

    def set_full_key(self, key: dict[int, int]) -> None:
        self._key = dict(key)
        self.key_changed.emit()

    def apply_key(self) -> str:
        """Decrypt ciphertext through the current partial key.

        Unmapped tokens render as '?'. Words are joined by the
        ciphertext's separator (typically a space for word boundaries).
        """
        if self.cipher_text is None:
            return ""

        def decode_word(word_tokens: list[int]) -> str:
            parts: list[str] = []
            for token_id in word_tokens:
                if token_id in self._key:
                    pt_id = self._key[token_id]
                    parts.append(self.plaintext_alphabet.symbol_for(pt_id))
                else:
                    parts.append("?")
            sep = " " if self.plaintext_alphabet._multisym else ""
            return sep.join(parts)

        decoded_words = [decode_word(w) for w in self.cipher_text.words]
        word_sep = self.cipher_text.separator or ""
        return word_sep.join(decoded_words)

    @property
    def is_complete(self) -> bool:
        if self.cipher_text is None:
            return False
        ct_symbols_used = set(self.cipher_text.tokens)
        return all(s in self._key for s in ct_symbols_used)

    @property
    def mapped_count(self) -> int:
        return len(self._key)

    @property
    def unmapped_cipher_ids(self) -> list[int]:
        if self.cipher_text is None:
            return []
        used = set(self.cipher_text.tokens)
        return sorted(used - set(self._key.keys()))

    @property
    def unmapped_plain_ids(self) -> list[int]:
        mapped_pt = set(self._key.values())
        return [i for i in range(self.plaintext_alphabet.size) if i not in mapped_pt]

    def invert_key(self) -> dict[int, int]:
        """Return plaintext->ciphertext mapping (for encoding)."""
        return {v: k for k, v in self._key.items()}
