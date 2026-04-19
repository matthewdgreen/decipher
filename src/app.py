from __future__ import annotations

from PySide6.QtWidgets import QApplication, QMessageBox

from models.session import Session
from services.claude_api import ClaudeAPI
from services.settings import Settings
from ui.main_window import MainWindow


class App:
    """Application controller. Wires together services, session, and UI."""

    def __init__(self, qt_app: QApplication) -> None:
        self.qt_app = qt_app
        self.settings = Settings()
        self.session = Session()

        # Try to create API client
        api_key = self.settings.get_api_key()
        self.api: ClaudeAPI | None = None
        if api_key:
            self.api = ClaudeAPI(api_key=api_key, model=self.settings.model)

        # Create main window
        self.window = MainWindow(
            session=self.session,
            claude_api=self.api,
            settings=self.settings,
        )

    def run(self) -> int:
        self.window.show()

        # Prompt for API key on first launch
        if self.api is None:
            QMessageBox.information(
                self.window,
                "Welcome to Decipher",
                "No Anthropic API key found.\n\n"
                "Go to Settings → Set API Key to enable AI features "
                "(OCR and agent cracking).\n\n"
                "You can still use manual text input and analysis tools.",
            )

        return self.qt_app.exec()
