"""Strategy base class and registry for the transform triage evaluation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from triage.types import CandidateRecord


class Strategy(ABC):
    """A function from (candidate_list, case_metadata) → ranked candidate_ids."""

    #: Short identifier used in reports and CLI.
    name: str = "unnamed"

    #: If True, the strategy uses decoded_text — only valid after label_case().
    requires_decoded_text: bool = False

    @abstractmethod
    def rank(
        self,
        candidates: list[CandidateRecord],
        case_metadata: dict[str, Any],
    ) -> list[str]:
        """Return candidate_ids ordered best-first."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def all_strategies() -> list[Strategy]:
    return [cls() for cls in _REGISTRY.values()]
