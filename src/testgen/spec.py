from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class DifficultyPreset(str, Enum):
    TINY    = "tiny"     # ~40 words,  word boundaries  (smoke test)
    MEDIUM  = "medium"   # ~200 words, word boundaries
    HARD    = "hard"     # ~250 words, NO word boundaries, simple substitution
    HARDEST = "hardest"  # ~200 words, NO word boundaries, homophonic substitution


_PRESET_PARAMS: dict[DifficultyPreset, dict] = {
    DifficultyPreset.TINY:    {"approx_length": 40,  "word_boundaries": True,  "homophonic": False},
    DifficultyPreset.MEDIUM:  {"approx_length": 200, "word_boundaries": True,  "homophonic": False},
    DifficultyPreset.HARD:    {"approx_length": 250, "word_boundaries": False, "homophonic": False},
    DifficultyPreset.HARDEST: {"approx_length": 200, "word_boundaries": False, "homophonic": True},
}

SUPPORTED_LANGUAGES = frozenset({"en", "it", "de", "fr", "la"})


@dataclass
class TestSpec:
    language: str = "en"
    approx_length: int = 100
    word_boundaries: bool = True
    topic: str = "general"
    seed: int | None = None
    frequency_style: Literal["normal", "unusual"] = "normal"
    homophonic: bool = False
    transform_pipeline: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language!r}. Use one of {sorted(SUPPORTED_LANGUAGES)}")

    @classmethod
    def from_preset(
        cls,
        preset: DifficultyPreset | str,
        language: str = "en",
        **overrides,
    ) -> TestSpec:
        p = DifficultyPreset(preset)
        params = dict(_PRESET_PARAMS[p])
        params.update(overrides)
        return cls(language=language, **params)
