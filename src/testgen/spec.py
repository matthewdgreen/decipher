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
POLYALPHABETIC_VARIANTS = frozenset({"vigenere", "beaufort", "variant_beaufort", "gronsfeld"})


@dataclass
class TestSpec:
    language: str = "en"
    approx_length: int = 100
    word_boundaries: bool = True
    topic: str = "general"
    seed: int | None = None
    frequency_style: Literal["normal", "unusual"] = "normal"
    homophonic: bool = False
    polyalphabetic_variant: str | None = None
    polyalphabetic_key: str | None = None
    polyalphabetic_period: int | None = None
    transform_pipeline: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language!r}. Use one of {sorted(SUPPORTED_LANGUAGES)}")
        if self.polyalphabetic_variant is not None:
            self.polyalphabetic_variant = self.polyalphabetic_variant.strip().lower()
            if self.polyalphabetic_variant not in POLYALPHABETIC_VARIANTS:
                raise ValueError(
                    f"Unsupported polyalphabetic variant: {self.polyalphabetic_variant!r}. "
                    f"Use one of {sorted(POLYALPHABETIC_VARIANTS)}"
                )
            if self.homophonic:
                raise ValueError("Synthetic specs cannot be both homophonic and polyalphabetic")
            if self.transform_pipeline:
                raise ValueError(
                    "Synthetic polyalphabetic specs do not yet support transform_pipeline; "
                    "keep transposition+homophonic ladder cases separate for now"
                )
        if self.polyalphabetic_period is not None and self.polyalphabetic_period < 1:
            raise ValueError("polyalphabetic_period must be >= 1")

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
