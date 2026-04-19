from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.cipher_text import CipherText


class InputPanel(QWidget):
    """Panel for inputting ciphertext via text paste or image upload."""

    cipher_text_ready = Signal(CipherText)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_path: str | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel("<b>Input</b>")
        layout.addWidget(header)

        # Text input
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(
            "Paste ciphertext here, or load an image..."
        )
        self.text_edit.setMaximumHeight(200)
        layout.addWidget(self.text_edit)

        # Controls row
        controls = QHBoxLayout()

        self.load_image_btn = QPushButton("Load Image...")
        self.load_image_btn.clicked.connect(self._load_image)
        controls.addWidget(self.load_image_btn)

        self.image_label = QLabel("")
        controls.addWidget(self.image_label)
        controls.addStretch()

        # Alphabet mode
        controls.addWidget(QLabel("Alphabet:"))
        self.alphabet_mode = QComboBox()
        self.alphabet_mode.addItems(["Auto-detect", "Standard A-Z", "Multi-char symbols"])
        controls.addWidget(self.alphabet_mode)

        # Space treatment
        controls.addWidget(QLabel("Spaces:"))
        self.space_mode = QComboBox()
        self.space_mode.addItems(["Word boundaries", "Cipher symbols", "Ignore"])
        self.space_mode.setToolTip(
            "Word boundaries: spaces separate words (not part of cipher alphabet)\n"
            "Cipher symbols: spaces are cipher symbols like any other character\n"
            "Ignore: strip all spaces before processing"
        )
        controls.addWidget(self.space_mode)

        # Punctuation treatment
        controls.addWidget(QLabel("Punctuation:"))
        self.punct_mode = QComboBox()
        self.punct_mode.addItems(["Passthrough", "Cipher symbols", "Strip"])
        self.punct_mode.setToolTip(
            "Passthrough: punctuation is preserved but not part of the cipher alphabet\n"
            "Cipher symbols: punctuation characters are cipher symbols\n"
            "Strip: remove all punctuation before processing"
        )
        controls.addWidget(self.punct_mode)

        layout.addLayout(controls)

        # Process button
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self._process)
        layout.addWidget(self.process_btn)

        # Detected alphabet display
        self.alphabet_label = QLabel("")
        self.alphabet_label.setWordWrap(True)
        layout.addWidget(self.alphabet_label)

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Cipher Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)",
        )
        if path:
            self._image_path = path
            self.image_label.setText(path.split("/")[-1])
            self.text_edit.setPlaceholderText(
                f"Image loaded: {path.split('/')[-1]}\n"
                "Click Process to run OCR, or paste text to use text mode instead."
            )

    def _process(self) -> None:
        """Process input and emit cipher_text_ready signal.

        Actual OCR/text processing is handled by MainWindow which connects
        to this signal and has access to the OCR engine.
        """
        # This will be connected in MainWindow to handle the actual processing
        pass

    def get_text(self) -> str:
        return self.text_edit.toPlainText()

    def get_image_path(self) -> str | None:
        return self._image_path

    def get_alphabet_mode(self) -> str:
        return self.alphabet_mode.currentText()

    def get_space_mode(self) -> str:
        return self.space_mode.currentText()

    def get_punct_mode(self) -> str:
        return self.punct_mode.currentText()

    def set_alphabet_info(self, symbols: list[str]) -> None:
        self.alphabet_label.setText(
            f"Detected {len(symbols)} symbols: {', '.join(symbols[:30])}"
            + ("..." if len(symbols) > 30 else "")
        )
