from __future__ import annotations

from abc import ABC, abstractmethod


class ExternalTool(ABC):
    """Abstract wrapper for external CLI cryptanalysis tools."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the external tool is installed and accessible."""
        ...

    @abstractmethod
    def solve(self, ciphertext: str, **kwargs: object) -> dict:
        """Run the tool on ciphertext and return parsed results."""
        ...
