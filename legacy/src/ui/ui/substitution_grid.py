from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.session import Session


class SubstitutionGrid(QWidget):
    """Interactive grid for viewing/editing the substitution key mapping."""

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._combos: dict[int, QComboBox] = {}
        self._updating = False
        self._setup_ui()
        self.session.cipher_text_changed.connect(self._rebuild_grid)
        self.session.key_changed.connect(self._refresh_decryption)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("<b>Substitution Key</b>")
        layout.addWidget(header)

        # Scrollable grid area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        scroll.setWidget(self.grid_container)
        layout.addWidget(scroll, 1)

        # Clear button
        clear_btn = QPushButton("Clear All Mappings")
        clear_btn.clicked.connect(self._clear_all)
        layout.addWidget(clear_btn)

        # Decryption preview
        layout.addWidget(QLabel("<b>Current Decryption:</b>"))
        self.decryption_view = QTextEdit()
        self.decryption_view.setReadOnly(True)
        self.decryption_view.setMaximumHeight(150)
        self.decryption_view.setFont(self.decryption_view.font())
        layout.addWidget(self.decryption_view)

    def _rebuild_grid(self) -> None:
        """Rebuild the mapping grid when ciphertext changes."""
        # Clear existing grid
        self._combos.clear()
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        ct = self.session.cipher_text
        if ct is None:
            return

        # Column headers
        self.grid_layout.addWidget(QLabel("<b>Cipher</b>"), 0, 0)
        self.grid_layout.addWidget(QLabel("<b>→</b>"), 0, 1)
        self.grid_layout.addWidget(QLabel("<b>Plain</b>"), 0, 2)

        pt_alpha = self.session.plaintext_alphabet
        plain_options = ["?"] + pt_alpha.symbols

        for row, (ct_id, symbol) in enumerate(
            sorted(
                [(i, ct.alphabet.symbol_for(i)) for i in range(ct.alphabet.size)],
                key=lambda x: x[1],
            ),
            start=1,
        ):
            label = QLabel(symbol)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid_layout.addWidget(label, row, 0)

            arrow = QLabel("→")
            arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid_layout.addWidget(arrow, row, 1)

            combo = QComboBox()
            combo.addItems(plain_options)
            combo.setCurrentIndex(0)
            combo.currentIndexChanged.connect(
                lambda idx, cid=ct_id: self._on_combo_changed(cid, idx)
            )
            self.grid_layout.addWidget(combo, row, 2)
            self._combos[ct_id] = combo

        self._refresh_decryption()

    def _on_combo_changed(self, ct_id: int, combo_index: int) -> None:
        if self._updating:
            return
        if combo_index == 0:
            self.session.clear_mapping(ct_id)
        else:
            pt_id = combo_index - 1  # offset for "?" at index 0
            self.session.set_mapping(ct_id, pt_id)

    def _refresh_decryption(self) -> None:
        """Update combo selections and decryption preview from session state."""
        self._updating = True
        key = self.session.key
        for ct_id, combo in self._combos.items():
            if ct_id in key:
                combo.setCurrentIndex(key[ct_id] + 1)
            else:
                combo.setCurrentIndex(0)
        self._updating = False

        decrypted = self.session.apply_key()
        self.decryption_view.setText(decrypted)

    def _clear_all(self) -> None:
        ct = self.session.cipher_text
        if ct is None:
            return
        for ct_id in list(self.session.key.keys()):
            self.session.clear_mapping(ct_id)
