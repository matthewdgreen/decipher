from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from analysis import frequency, ic, pattern
from models.session import Session


class AnalysisPanel(QWidget):
    """Tabbed panel showing frequency analysis, IC, and pattern matching."""

    def __init__(self, session: Session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._pattern_dict: dict[str, list[str]] = {}
        self._setup_ui()
        self.session.cipher_text_changed.connect(self.refresh)

    def set_pattern_dict(self, pd: dict[str, list[str]]) -> None:
        self._pattern_dict = pd

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("<b>Analysis</b>")
        layout.addWidget(header)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Frequency tab
        self.freq_table = QTableWidget()
        self.freq_table.setColumnCount(4)
        self.freq_table.setHorizontalHeaderLabels(
            ["Symbol", "Count", "%", "English Expected"]
        )
        self.freq_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tabs.addTab(self.freq_table, "Frequencies")

        # IC tab
        self.ic_widget = QWidget()
        ic_layout = QVBoxLayout(self.ic_widget)
        self.ic_label = QLabel("Load ciphertext to compute IC")
        self.ic_label.setTextFormat(Qt.TextFormat.RichText)
        ic_layout.addWidget(self.ic_label)
        ic_layout.addStretch()
        self.tabs.addTab(self.ic_widget, "IC")

        # Patterns tab
        self.pattern_table = QTableWidget()
        self.pattern_table.setColumnCount(3)
        self.pattern_table.setHorizontalHeaderLabels(
            ["Cipher Word", "Pattern", "Candidates"]
        )
        self.pattern_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tabs.addTab(self.pattern_table, "Patterns")

    def refresh(self) -> None:
        ct = self.session.cipher_text
        if ct is None:
            return
        self._refresh_frequency(ct)
        self._refresh_ic(ct)
        self._refresh_patterns(ct)

    def _refresh_frequency(self, ct) -> None:
        tokens = ct.tokens
        sorted_freq = frequency.sorted_frequency(tokens)
        total = len(tokens)

        # English reference (by frequency rank)
        english_sorted = sorted(
            frequency.ENGLISH_LETTER_FREQ.items(), key=lambda x: x[1], reverse=True
        )

        self.freq_table.setRowCount(len(sorted_freq))
        for row, (tid, count) in enumerate(sorted_freq):
            symbol = ct.alphabet.symbol_for(tid)
            pct = count / total * 100 if total > 0 else 0

            self.freq_table.setItem(row, 0, QTableWidgetItem(symbol))
            self.freq_table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.freq_table.setItem(row, 2, QTableWidgetItem(f"{pct:.1f}%"))

            # Show corresponding English letter by rank
            if row < len(english_sorted):
                eng_letter, eng_pct = english_sorted[row]
                self.freq_table.setItem(
                    row, 3, QTableWidgetItem(f"{eng_letter} ({eng_pct:.1f}%)")
                )

    def _refresh_ic(self, ct) -> None:
        ic_val = ic.index_of_coincidence(ct.tokens, ct.alphabet.size)
        eng_ic = ic.ENGLISH_IC
        rand_ic = ic.random_ic(ct.alphabet.size)
        likely = ic.is_likely_monoalphabetic(ic_val, ct.alphabet.size)

        color = "green" if likely else "orange"
        self.ic_label.setText(
            f"<h3>Index of Coincidence</h3>"
            f"<p style='font-size: 18px; color: {color};'><b>{ic_val:.4f}</b></p>"
            f"<p>English expected: {eng_ic:.4f}</p>"
            f"<p>Random expected ({ct.alphabet.size} symbols): {rand_ic:.4f}</p>"
            f"<p>{'Likely monoalphabetic substitution' if likely else 'May not be simple monoalphabetic'}</p>"
        )

    def _refresh_patterns(self, ct) -> None:
        # Use pre-split words from CipherText (respects separator setting)
        words = ct.words
        self.pattern_table.setRowCount(min(len(words), 50))

        for row, word_tokens in enumerate(words[:50]):
            word_display = "".join(ct.alphabet.symbol_for(t) for t in word_tokens)
            pat = pattern.word_pattern(word_tokens)
            candidates = pattern.match_pattern(pat, self._pattern_dict)

            self.pattern_table.setItem(row, 0, QTableWidgetItem(word_display))
            self.pattern_table.setItem(row, 1, QTableWidgetItem(pat))
            preview = ", ".join(candidates[:10])
            if len(candidates) > 10:
                preview += f"... ({len(candidates)} total)"
            self.pattern_table.setItem(row, 2, QTableWidgetItem(preview))
