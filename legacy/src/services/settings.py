from __future__ import annotations

import keyring
from PySide6.QtCore import QSettings

SERVICE_NAME = "decipher"
API_KEY_ACCOUNT = "anthropic_api_key"

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_ITERATIONS = 25


class Settings:
    """Application settings backed by QSettings (prefs) and macOS Keychain (secrets)."""

    def __init__(self) -> None:
        self._qs = QSettings("Decipher", "Decipher")

    # --- API key (stored in macOS Keychain) ---

    def get_api_key(self) -> str | None:
        key = keyring.get_password(SERVICE_NAME, API_KEY_ACCOUNT)
        return key if key else None

    def set_api_key(self, key: str) -> None:
        keyring.set_password(SERVICE_NAME, API_KEY_ACCOUNT, key)

    def delete_api_key(self) -> None:
        try:
            keyring.delete_password(SERVICE_NAME, API_KEY_ACCOUNT)
        except keyring.errors.PasswordDeleteError:
            pass

    # --- General preferences ---

    @property
    def model(self) -> str:
        return str(self._qs.value("model", DEFAULT_MODEL))

    @model.setter
    def model(self, value: str) -> None:
        self._qs.setValue("model", value)

    @property
    def max_iterations(self) -> int:
        val = self._qs.value("max_iterations", DEFAULT_MAX_ITERATIONS)
        return int(val)  # type: ignore[arg-type]

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self._qs.setValue("max_iterations", value)

    def get(self, key: str, default: object = None) -> object:
        return self._qs.value(key, default)

    def set(self, key: str, value: object) -> None:
        self._qs.setValue(key, value)
