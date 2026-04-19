from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QInputDialog,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from analysis.pattern import build_pattern_dictionary, load_word_list
from models.session import Session
from ocr.engine import OCREngine
from services.claude_api import ClaudeAPI, ClaudeAPIError
from services.settings import Settings
from ui.agent_panel import AgentPanel
from ui.analysis_panel import AnalysisPanel
from ui.encode_panel import EncodePanel
from ui.input_panel import InputPanel
from ui.substitution_grid import SubstitutionGrid


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(
        self,
        session: Session,
        claude_api: ClaudeAPI | None,
        settings: Settings,
    ) -> None:
        super().__init__()
        self.session = session
        self.api = claude_api
        self.settings = settings
        self.ocr_engine = OCREngine(claude_api) if claude_api else None

        # Load dictionary for pattern matching
        import os
        dict_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "resources", "dictionaries", "english_common.txt",
        )
        word_list = load_word_list(dict_path)
        self._pattern_dict = build_pattern_dictionary(word_list)

        self._setup_ui()
        self._setup_menu()
        self.setWindowTitle("Decipher - Classical Cipher Cryptanalysis")
        self.resize(1400, 900)

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left column: Input + Analysis
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.input_panel = InputPanel()
        self.input_panel.process_btn.clicked.connect(self._process_input)
        left_layout.addWidget(self.input_panel, 1)

        self.analysis_panel = AnalysisPanel(self.session)
        self.analysis_panel.set_pattern_dict(self._pattern_dict)
        left_layout.addWidget(self.analysis_panel, 2)

        splitter.addWidget(left)

        # Right column: Substitution Grid + Agent + Encode
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.sub_grid = SubstitutionGrid(self.session)
        right_layout.addWidget(self.sub_grid, 2)

        if self.api:
            self.agent_panel = AgentPanel(
                self.session, self.api, self.settings
            )
            right_layout.addWidget(self.agent_panel, 2)
        else:
            from PySide6.QtWidgets import QLabel
            no_api = QLabel(
                "Set API key (Settings menu) to enable AI agent."
            )
            no_api.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_layout.addWidget(no_api, 2)
            self.agent_panel = None

        self.encode_panel = EncodePanel(self.session)
        right_layout.addWidget(self.encode_panel, 1)

        splitter.addWidget(right)
        splitter.setSizes([600, 800])

    def _setup_menu(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("Quit", self.close, "Ctrl+Q")

        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        edit_menu.addAction("Clear Key", self._clear_key)
        edit_menu.addAction("Reset Session", self._reset_session)

        # Settings menu
        settings_menu = menu_bar.addMenu("Settings")
        settings_menu.addAction("Set API Key...", self._set_api_key)
        settings_menu.addAction("Set Model...", self._set_model)

    def _process_input(self) -> None:
        """Process text or image input into a CipherText."""
        image_path = self.input_panel.get_image_path()
        text = self.input_panel.get_text()
        mode = self.input_panel.get_alphabet_mode()
        space_mode_label = self.input_panel.get_space_mode()
        punct_mode_label = self.input_panel.get_punct_mode()

        # Map UI labels to internal values
        space_treatment = {
            "Word boundaries": "word_boundaries",
            "Cipher symbols": "cipher_symbols",
            "Ignore": "ignore",
        }.get(space_mode_label, "word_boundaries")

        punct_treatment = {
            "Passthrough": "passthrough",
            "Cipher symbols": "cipher_symbols",
            "Strip": "strip",
        }.get(punct_mode_label, "passthrough")

        try:
            if image_path and not text:
                # OCR mode
                if self.ocr_engine is None:
                    QMessageBox.warning(
                        self, "No API Key",
                        "Set your Anthropic API key in Settings to use OCR."
                    )
                    return
                ct = self.ocr_engine.process_image(image_path)
            elif text:
                # Text mode
                multisym = mode == "Multi-char symbols"
                if self.ocr_engine:
                    ct = self.ocr_engine.process_text(
                        text, multisym=multisym,
                        space_treatment=space_treatment,
                        punct_treatment=punct_treatment,
                    )
                else:
                    from models.alphabet import Alphabet
                    from models.cipher_text import CipherText
                    from ocr.engine import OCREngine
                    ignore: set[str] = set()
                    if space_treatment == "word_boundaries":
                        ignore.update({" ", "\t", "\n", "\r"})
                    if punct_treatment == "passthrough":
                        ignore.update(OCREngine.PUNCTUATION)
                    proc_text = text
                    if space_treatment == "ignore":
                        proc_text = proc_text.replace(" ", "")
                    if punct_treatment == "strip":
                        proc_text = "".join(c for c in proc_text if c not in OCREngine.PUNCTUATION)
                    if space_treatment != "cipher_symbols":
                        proc_text = " ".join(proc_text.split())
                    separator = " " if space_treatment == "word_boundaries" else None
                    alphabet = Alphabet.from_text(proc_text, multisym=multisym, ignore_chars=ignore)
                    ct = CipherText(raw=proc_text, alphabet=alphabet, source="manual", separator=separator)
            else:
                QMessageBox.information(
                    self, "No Input", "Paste ciphertext or load an image first."
                )
                return

            self.session.set_cipher_text(ct)
            self.input_panel.set_alphabet_info(ct.alphabet.symbols)

        except ClaudeAPIError as e:
            QMessageBox.critical(self, "API Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process input: {e}")

    def _clear_key(self) -> None:
        ct = self.session.cipher_text
        if ct:
            for ct_id in list(self.session.key.keys()):
                self.session.clear_mapping(ct_id)

    def _reset_session(self) -> None:
        self.session.set_cipher_text.__func__  # just to check it exists
        self.session.cipher_text = None
        self.session._key.clear()
        self.session.history.clear()
        self.session.cipher_text_changed.emit()
        self.session.key_changed.emit()

    def _set_api_key(self) -> None:
        current = self.settings.get_api_key() or ""
        masked = current[:8] + "..." if len(current) > 8 else current
        key, ok = QInputDialog.getText(
            self, "API Key",
            f"Enter your Anthropic API key (current: {masked}):",
        )
        if ok and key.strip():
            self.settings.set_api_key(key.strip())
            QMessageBox.information(
                self, "API Key",
                "API key saved. Restart the app to use the new key."
            )

    def _set_model(self) -> None:
        current = self.settings.model
        model, ok = QInputDialog.getText(
            self, "Model", f"Enter model name (current: {current}):",
            text=current,
        )
        if ok and model.strip():
            self.settings.model = model.strip()
