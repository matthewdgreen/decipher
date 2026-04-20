from __future__ import annotations

from external.base import ExternalTool


class AZdecrypt(ExternalTool):
    """Wrapper for the AZdecrypt cipher solving tool."""

    def is_available(self) -> bool:
        # Future integration: check for AZdecrypt on PATH or configured location.
        return False

    def solve(self, ciphertext: str, **kwargs: object) -> dict:
        raise NotImplementedError("AZdecrypt integration not yet implemented")
