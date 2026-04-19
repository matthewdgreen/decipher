from __future__ import annotations

from external.base import ExternalTool


class CryptoCrack(ExternalTool):
    """Wrapper for the CryptoCrack cipher solving tool."""

    def is_available(self) -> bool:
        # TODO: Check for CryptoCrack binary on PATH or configured location
        return False

    def solve(self, ciphertext: str, **kwargs: object) -> dict:
        raise NotImplementedError("CryptoCrack integration not yet implemented")
