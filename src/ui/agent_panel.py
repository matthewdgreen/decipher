from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from agent.loop import AgentLoop
from models.session import Session
from services.claude_api import ClaudeAPI
from services.settings import Settings


class AgentPanel(QWidget):
    """Panel showing the AI agent's cracking progress and controls."""

    def __init__(
        self,
        session: Session,
        claude_api: ClaudeAPI,
        settings: Settings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.api = claude_api
        self.settings = settings
        self.language = "en"
        self._agent_loop: AgentLoop | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("<b>AI Agent</b>")
        layout.addWidget(header)

        # Controls
        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start Cracking")
        self.start_btn.clicked.connect(self._start)
        controls.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.stop_btn)

        controls.addStretch()

        controls.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Latin", "German"])
        self.language_combo.currentTextChanged.connect(self._on_language_changed)
        controls.addWidget(self.language_combo)

        self.status_label = QLabel("Idle")
        controls.addWidget(self.status_label)
        layout.addLayout(controls)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Score: %v%")
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

    def _start(self) -> None:
        if self.session.cipher_text is None:
            self._log("Error: No ciphertext loaded. Process input first.")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Running...")
        self.log_view.clear()
        self._log("Starting agent...")

        self._agent_loop = AgentLoop(self.session, self.api, self.settings, language=self.language)
        worker = self._agent_loop.start()

        worker.text_delta.connect(self._on_text)
        worker.tool_called.connect(self._on_tool)
        worker.iteration_complete.connect(self._on_iteration)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)

    def _on_language_changed(self, text: str) -> None:
        lang_map = {"English": "en", "Latin": "la", "German": "de"}
        self.language = lang_map.get(text, "en")

    def _stop(self) -> None:
        if self._agent_loop:
            self._agent_loop.stop()
        self.status_label.setText("Stopping...")

    def _on_text(self, text: str) -> None:
        self.log_view.insertPlainText(text)
        self.log_view.ensureCursorVisible()

    def _on_tool(self, name: str, result: str) -> None:
        self._log(f"\n🔧 {name}: {result}\n")

    def _on_iteration(self, iteration: int, summary: str) -> None:
        self._log(f"\n--- {summary} ---\n")

    def _on_progress(self, score: float) -> None:
        self.progress_bar.setValue(int(score * 100))

    def _on_finished(self, status: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Done: {status}")
        self._log(f"\n=== Agent finished: {status} ===\n")

    def _on_error(self, error: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Error")
        self._log(f"\n❌ {error}\n")

    def _log(self, text: str) -> None:
        self.log_view.append(text)
        self.log_view.ensureCursorVisible()
