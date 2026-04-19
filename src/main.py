"""Decipher - Classical Cipher Cryptanalysis Tool"""
from __future__ import annotations

import sys
import os

# Add src directory to path so imports work
sys.path.insert(0, os.path.dirname(__file__))


def main() -> None:
    from PySide6.QtWidgets import QApplication

    from app import App

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("Decipher")
    qt_app.setOrganizationName("Decipher")

    app = App(qt_app)
    sys.exit(app.run())


if __name__ == "__main__":
    main()
