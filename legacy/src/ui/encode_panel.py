from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ciphers.substitution import SubstitutionCipher
from models.session import Session


class EncodePanel(QWidget):
    """Panel for encoding plaintext using the derived substitution key."""

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._setup_ui()
        self.session.key_changed.connect(self._update_state)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("<b>Encode</b>")
        layout.addWidget(header)

        layout.addWidget(QLabel("Enter plaintext to encrypt:"))
        self.plaintext_edit = QTextEdit()
        self.plaintext_edit.setMaximumHeight(100)
        self.plaintext_edit.setPlaceholderText("Type plaintext here...")
        layout.addWidget(self.plaintext_edit)

        self.encode_btn = QPushButton("Encode")
        self.encode_btn.clicked.connect(self._encode)
        self.encode_btn.setEnabled(False)
        layout.addWidget(self.encode_btn)

        layout.addWidget(QLabel("Ciphertext output:"))
        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setMaximumHeight(100)
        layout.addWidget(self.output_edit)

        self.status_label = QLabel("Complete the key mapping to enable encoding.")
        layout.addWidget(self.status_label)

    def _update_state(self) -> None:
        complete = self.session.is_complete
        self.encode_btn.setEnabled(complete)
        if complete:
            self.status_label.setText("Key is complete. Ready to encode.")
        else:
            mapped = self.session.mapped_count
            ct = self.session.cipher_text
            total = len(set(ct.tokens)) if ct else 0
            self.status_label.setText(
                f"Key incomplete: {mapped}/{total} symbols mapped."
            )

    def _encode(self) -> None:
        plaintext_str = self.plaintext_edit.toPlainText().upper()
        if not plaintext_str:
            return

        ct = self.session.cipher_text
        if ct is None:
            return

        # Encode plaintext to token IDs using plaintext alphabet
        pt_alpha = self.session.plaintext_alphabet
        pt_tokens = pt_alpha.encode(plaintext_str)

        # Invert the decryption key to get an encryption key
        # Session key: ciphertext_id -> plaintext_id
        # We need: plaintext_id -> ciphertext_id
        encrypt_key = self.session.invert_key()

        cipher = SubstitutionCipher()
        ct_tokens = cipher.encrypt(pt_tokens, encrypt_key, pt_alpha)

        # Decode to cipher alphabet symbols
        result = ct.alphabet.decode(ct_tokens)
        self.output_edit.setText(result)
