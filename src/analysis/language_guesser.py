"""IC-based plaintext language guesser.

Under monoalphabetic substitution the index of coincidence is invariant —
the cipher IC equals the plaintext IC. We exploit this to rank candidate
languages before any decipherment begins.

Design notes
------------
- Each language is described by a single reference IC (its expected IC for
  natural prose). More sophisticated features (bigram chi², letter-frequency
  correlation) can be added later by replacing or augmenting the scoring fn.
- Intentionally simple and data-driven so that adding a new language is a
  one-line change to LANGUAGE_IC_REFERENCES.
- Returns a ranked list so callers can show all candidates, not just the top.

References for IC values
------------------------
Sinkov, A. (1966). *Elementary Cryptanalysis*. MAA.
Practical Cryptography (practicalcryptography.com/cryptanalysis/text-characterisation/index-coincidence)
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Reference IC values for natural prose in each language.
# Add new languages here; no other code changes needed.
# Values are for standard modern alphabets; medieval / archaic text may vary.
# ---------------------------------------------------------------------------
LANGUAGE_IC_REFERENCES: dict[str, tuple[str, float]] = {
    # code  : (display_name, reference_ic)
    "en":   ("English",    0.0667),
    "la":   ("Latin",      0.0737),
    "de":   ("German",     0.0762),
    "fr":   ("French",     0.0778),
    "it":   ("Italian",    0.0738),
    "es":   ("Spanish",    0.0775),
    "nl":   ("Dutch",      0.0798),
    "pt":   ("Portuguese", 0.0745),
}


@dataclass
class LanguageGuess:
    """Ranking entry for one candidate language."""
    language: str        # ISO 639-1 code
    name: str            # display name
    reference_ic: float  # expected IC for natural prose
    ic_distance: float   # |measured_ic - reference_ic|  (lower = better match)


def rank_by_ic(measured_ic: float) -> list[LanguageGuess]:
    """Rank all reference languages by proximity to measured_ic.

    Returns a list sorted by ic_distance ascending (best match first).
    """
    guesses = [
        LanguageGuess(
            language=code,
            name=name,
            reference_ic=ref_ic,
            ic_distance=abs(measured_ic - ref_ic),
        )
        for code, (name, ref_ic) in LANGUAGE_IC_REFERENCES.items()
    ]
    guesses.sort(key=lambda g: g.ic_distance)
    return guesses


def best_guess(measured_ic: float) -> LanguageGuess:
    """Return the single best-matching language for the given IC."""
    return rank_by_ic(measured_ic)[0]


def format_ranking(measured_ic: float, top_n: int = 5) -> str:
    """Human-readable ranking string, e.g. for inclusion in agent context."""
    ranked = rank_by_ic(measured_ic)[:top_n]
    lines = [f"IC-based language ranking (measured IC = {measured_ic:.4f}):"]
    for i, g in enumerate(ranked, 1):
        lines.append(
            f"  {i}. {g.name:<12} reference IC={g.reference_ic:.4f}  "
            f"distance={g.ic_distance:.4f}"
        )
    return "\n".join(lines)
