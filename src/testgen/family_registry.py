"""Family registry for the on-demand cipher benchmark generator (Slice C).

This module is the single binding between the Slice-A cipher families
(``src/ciphers/``) and the generator orchestration
(``scripts/generate_cipher_benchmark.py``).  One :class:`FamilyGenSpec` entry
per *generatable* family declares:

* ``family_id``    — the stable ``cipher_system`` string emitted into the
  benchmark splits/manifest.  Where INV's ``investigation/families.py`` already
  names the *specific* family we REUSE its id string (only ``playfair`` collides
  today); every other Slice-A family is a NEW, more specific id that rolls up to
  a coarse INV family recorded in ``inv_family_id`` for cross-reference.
* ``factory``      — a zero-arg callable returning a fresh Slice-A cipher.
* ``canonical_form`` — how the ciphertext is serialised into the canonical
  transcription file, matching ``builder.py``'s conventions:
    - ``letters``  : an A-Z letter stream, stored as space-separated single
      letters (the no-boundary convention of ``_format_canonical``);
    - ``numeric``  : space-separated integer tokens, stored verbatim;
    - ``tokens``   : an opaque encoded string (Base64/hex/Morse/…), verbatim.
* ``make_key``     — a deterministic key sampler driven by an injected
  ``random.Random`` plus a :class:`KeyContext` (difficulty + selected plaintext
  + the language's corpus records, for running-key text keys).
* ``languages``    — applicable plaintext languages (all Slice-A families operate
  on cleaned uppercase A-Z, so every family applies to every shipped language).
* ``inv_family_id``, ``notes`` — documentation / the "NEW vs INV" deliverable.

Determinism: every key choice derives from the injected ``random.Random``; no
global ``random`` state is touched.  No network, no LLM.

"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from ciphers.encodings import (
    A1Z26Cipher,
    Base32Cipher,
    Base64Cipher,
    BaconianCipher,
    BinaryCipher,
    HexCipher,
    MorseCipher,
    Rot47Cipher,
    TapCodeCipher,
)
from ciphers.family_base import FamilyCipher
from ciphers.fractionation import (
    ADFGVXCipher,
    ADFGXCipher,
    BifidCipher,
    TrifidCipher,
)
from ciphers.numeric import (
    Hill2x2Cipher,
    NihilistSubstitutionCipher,
    StraddlingCheckerboardCipher,
)
from ciphers.playfair import FourSquareCipher, PlayfairCipher, TwoSquareCipher
from ciphers.polyalphabetic import (
    AutokeyCipher,
    BeaufortCipher,
    GronsfeldCipher,
    PortaCipher,
    RunningKeyCipher,
    VariantBeaufortCipher,
    VigenereCipher,
)
from ciphers.transposition import (
    AmscoCipher,
    CadenusCipher,
    ColumnarCipher,
    MyszkowskiCipher,
    NihilistTranspositionCipher,
    RailfenceCipher,
    RedefenceCipher,
    RouteCipher,
)
from ciphers.textutil import clean
from testgen.corpus_library import PassageRecord

__all__ = [
    "CanonicalForm",
    "LETTERS",
    "NUMERIC",
    "TOKENS",
    "DIFFICULTIES",
    "LANGUAGES",
    "KeyContext",
    "FamilyGenSpec",
    "FAMILY_GEN_REGISTRY",
    "all_family_ids",
    "get_spec",
    "length_window",
    "is_new_family",
    "new_family_ids",
    "known_family_ids",
    "family_cross_reference",
]

# Canonical-form tags (see module docstring).
LETTERS = "letters"
NUMERIC = "numeric"
TOKENS = "tokens"
CanonicalForm = str

DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

# The five shipped plaintext-library languages.
LANGUAGES: tuple[str, ...] = ("en", "de", "fr", "it", "la")

# Difficulty -> plaintext-length window (words), a MEASURED-property knob:
# more ciphertext is easier cryptanalysis, so ``easy`` picks longer passages.
# The shipped library ships graded lengths {40, 120, 250, 500} for every
# language, so each window resolves for every language.
_LENGTH_WINDOWS: dict[str, dict[str, int | None]] = {
    "easy": {"min_words": 200, "max_words": None},
    "medium": {"min_words": 100, "max_words": 200},
    "hard": {"min_words": None, "max_words": 80},
}

# Difficulty -> periodic-key length band (Vigenere relatives / autokey primer).
# Longer keys are harder, so ``hard`` uses longer periods.
_PERIOD_BANDS: dict[str, tuple[int, int]] = {
    "easy": (3, 4),
    "medium": (5, 6),
    "hard": (7, 9),
}


def length_window(difficulty: str) -> dict[str, int | None]:
    """Return the ``select()`` length filter for a difficulty (copy)."""
    return dict(_LENGTH_WINDOWS.get(difficulty, _LENGTH_WINDOWS["medium"]))


def _period(difficulty: str, rng: random.Random) -> int:
    lo, hi = _PERIOD_BANDS.get(difficulty, _PERIOD_BANDS["medium"])
    return rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# Key-sampling context + samplers
# ---------------------------------------------------------------------------


@dataclass
class KeyContext:
    """Everything a family's ``make_key`` may need, all deterministic.

    ``rng`` is the per-case ``random.Random`` (already used to draw ``plaintext``
    so key draws are sequenced after plaintext selection).  ``library_records``
    is the full language library (for running-key text keys).
    """

    language: str
    difficulty: str
    plaintext: str
    library_records: list[PassageRecord]
    rng: random.Random
    source_hash: str = ""


def _default_key(cipher: FamilyCipher, ctx: KeyContext) -> Any:
    return cipher.random_key(ctx.rng)


def _periodic_key(cipher: FamilyCipher, ctx: KeyContext) -> Any:
    return cipher.random_key(ctx.rng, period=_period(ctx.difficulty, ctx.rng))


def _running_key(cipher: FamilyCipher, ctx: KeyContext) -> Any:
    """Use a *different, long-enough* corpus passage as the running key.

    Falls back to synthesised random letters only when no shipped passage is at
    least as long as the plaintext (keeps the family self-contained/offline).
    """
    need = len(clean(ctx.plaintext))
    candidates = [
        r
        for r in ctx.library_records
        if r.language == ctx.language
        and r.content_hash != ctx.source_hash
        and len(clean(r.text)) >= need
    ]
    if candidates:
        return ctx.rng.choice(candidates).text
    return cipher.random_key(ctx.rng, length=max(1, need))


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyGenSpec:
    family_id: str
    display_name: str
    factory: Callable[[], FamilyCipher]
    canonical_form: CanonicalForm
    inv_family_id: str | None
    make_key: Callable[[FamilyCipher, KeyContext], Any] = _default_key
    languages: tuple[str, ...] = LANGUAGES
    notes: str = ""

    def new_cipher(self) -> FamilyCipher:
        return self.factory()


def _spec(
    family_id: str,
    display_name: str,
    factory: Callable[[], FamilyCipher],
    canonical_form: CanonicalForm,
    inv_family_id: str | None,
    *,
    make_key: Callable[[FamilyCipher, KeyContext], Any] = _default_key,
    languages: tuple[str, ...] = LANGUAGES,
    notes: str = "",
) -> FamilyGenSpec:
    return FamilyGenSpec(
        family_id=family_id,
        display_name=display_name,
        factory=factory,
        canonical_form=canonical_form,
        inv_family_id=inv_family_id,
        make_key=make_key,
        languages=languages,
        notes=notes,
    )


# INV coarse-family roll-ups (cross-reference only; the generated ``family_id``
# is the specific family). Ids validated against investigation.families at the
# bottom of this module.
_POLY = "polyalphabetic_periodic"
_TRANSP = "transposition"
_FRACT = "fractionation_transposition"
_POLYG = "polygraphic_substitution"


_SPECS: tuple[FamilyGenSpec, ...] = (
    # ---- polyalphabetic (roll up to polyalphabetic_periodic) ----
    _spec("vigenere", "Vigenere", VigenereCipher, LETTERS, _POLY,
          make_key=_periodic_key, notes="Periodic Vigenere tableau."),
    _spec("beaufort", "Beaufort", BeaufortCipher, LETTERS, _POLY,
          make_key=_periodic_key, notes="Self-reciprocal Vigenere relative."),
    _spec("variant_beaufort", "Variant Beaufort", VariantBeaufortCipher, LETTERS, _POLY,
          make_key=_periodic_key),
    _spec("gronsfeld", "Gronsfeld", GronsfeldCipher, LETTERS, _POLY,
          make_key=_periodic_key, notes="Vigenere with a 0-9 digit key."),
    _spec("porta", "Porta", PortaCipher, LETTERS, _POLY,
          make_key=_periodic_key, notes="13 reciprocal alphabets, self-reciprocal."),
    _spec("autokey_text", "Autokey (text)", lambda: AutokeyCipher("text"), LETTERS, _POLY,
          make_key=_periodic_key,
          notes="Text-autokey; non-periodic keystream (no INV-specific id)."),
    _spec("autokey_key", "Autokey (ciphertext)", lambda: AutokeyCipher("key"), LETTERS, _POLY,
          make_key=_periodic_key,
          notes="Ciphertext-autokey; non-periodic keystream."),
    _spec("running_key", "Running key", RunningKeyCipher, LETTERS, _POLY,
          make_key=_running_key,
          notes="Vigenere keyed by a full corpus passage; non-periodic."),
    # ---- digraphic / polygraphic ----
    _spec("playfair", "Playfair", PlayfairCipher, LETTERS, "playfair",
          notes="INV id reused (only family that collides with families.py)."),
    _spec("two_square", "Two-square", TwoSquareCipher, LETTERS, _POLYG,
          notes="Digraphic; two keyed 5x5 squares."),
    _spec("four_square", "Four-square", FourSquareCipher, LETTERS, _POLYG,
          notes="Digraphic; two keyed cipher squares."),
    _spec("hill2x2", "Hill 2x2", Hill2x2Cipher, LETTERS, _POLYG,
          notes="Linear polygraphic (mod-26 2x2). Uses corrected-decrypt wrapper."),
    # ---- fractionation ----
    _spec("bifid", "Bifid", BifidCipher, LETTERS, _FRACT,
          notes="Polybius fractionation (whole-message period)."),
    _spec("trifid", "Trifid", TrifidCipher, LETTERS, _FRACT,
          notes="3D Polybius fractionation, period 5."),
    _spec("adfgx", "ADFGX", ADFGXCipher, LETTERS, _FRACT,
          notes="5x5 fractionation + keyed columnar (ADFGX symbols)."),
    _spec("adfgvx", "ADFGVX", ADFGVXCipher, LETTERS, _FRACT,
          notes="6x6 fractionation + keyed columnar (ADFGVX symbols)."),
    # ---- transposition ----
    _spec("columnar_transposition", "Columnar transposition", ColumnarCipher, LETTERS, _TRANSP,
          notes="Complete/incomplete keyed columnar."),
    _spec("railfence", "Rail fence", RailfenceCipher, LETTERS, _TRANSP,
          notes="Zigzag rail-fence transposition."),
    _spec("redefence", "Redefence", RedefenceCipher, LETTERS, _TRANSP,
          notes="Rail-fence with keyed rail read-out order."),
    _spec("route_transposition", "Route transposition", RouteCipher, LETTERS, _TRANSP,
          notes="Grid fill, spiral/diagonal read-out."),
    _spec("amsco", "Amsco", AmscoCipher, LETTERS, _TRANSP,
          notes="Incomplete columnar with 1-2 letter cells."),
    _spec("myszkowski", "Myszkowski", MyszkowskiCipher, LETTERS, _TRANSP,
          notes="Columnar with repeated key letters."),
    _spec("cadenus", "Cadenus", CadenusCipher, LETTERS, _TRANSP,
          notes="25-row keyed columnar with cyclic column shifts (W->V)."),
    _spec("nihilist_transposition", "Nihilist transposition", NihilistTranspositionCipher,
          LETTERS, _TRANSP, notes="n x n block, rows+cols permuted by keyword."),
    # ---- numeric ----
    _spec("nihilist_substitution", "Nihilist substitution", NihilistSubstitutionCipher,
          NUMERIC, None,
          notes="Polybius + keyed additive; number pairs. No INV-specific id."),
    _spec("straddling_checkerboard", "Straddling checkerboard", StraddlingCheckerboardCipher,
          TOKENS, None,
          notes="Mixed 1/2-digit codes; digit string. No INV-specific id."),
    # ---- detection-tier encodings (roadmap item 14) ----
    _spec("baconian", "Baconian", BaconianCipher, LETTERS, None,
          notes="5-bit A/B encoding. Detection tier."),
    _spec("a1z26", "A1Z26", A1Z26Cipher, NUMERIC, None,
          notes="A=1..Z=26 numbers. Detection tier."),
    _spec("morse", "Morse", MorseCipher, TOKENS, None,
          notes="International Morse. Detection tier."),
    _spec("base64", "Base64", Base64Cipher, TOKENS, None,
          notes="Base64 of ASCII bytes. Detection tier."),
    _spec("base32", "Base32", Base32Cipher, TOKENS, None,
          notes="Base32 of ASCII bytes. Detection tier."),
    _spec("hex", "Hex", HexCipher, TOKENS, None,
          notes="Hexadecimal of ASCII bytes. Detection tier."),
    _spec("binary", "Binary", BinaryCipher, TOKENS, None,
          notes="8-bit binary of ASCII bytes. Detection tier."),
    _spec("rot47", "ROT47", Rot47Cipher, TOKENS, None,
          notes="ROT47 over printable ASCII. Detection tier."),
    _spec("tap_code", "Tap code", TapCodeCipher, NUMERIC, None,
          notes="5x5 tap pairs (K->C). Detection tier."),
)

FAMILY_GEN_REGISTRY: dict[str, FamilyGenSpec] = {s.family_id: s for s in _SPECS}


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def all_family_ids() -> list[str]:
    """Every generatable family id, in registry (declaration) order."""
    return [s.family_id for s in _SPECS]


def get_spec(family_id: str) -> FamilyGenSpec:
    try:
        return FAMILY_GEN_REGISTRY[family_id]
    except KeyError:
        raise KeyError(
            f"unknown family {family_id!r}; known: {', '.join(all_family_ids())}"
        ) from None


def _inv_family_ids() -> set[str]:
    """Ids present in INV's ``investigation.families`` registry (read-only)."""
    from investigation.families import FAMILY_REGISTRY as INV_REGISTRY

    return set(INV_REGISTRY)


def is_new_family(family_id: str) -> bool:
    """True if ``family_id`` is NOT already an INV ``families.py`` id.

    This is the "families not yet tested against INV/solver" test: only
    ``playfair`` currently collides, so every other generatable family is NEW.
    """
    return family_id not in _inv_family_ids()


def new_family_ids() -> list[str]:
    """Generatable family ids that are NEW vs INV ``families.py`` (registry order)."""
    inv = _inv_family_ids()
    return [s.family_id for s in _SPECS if s.family_id not in inv]


def known_family_ids() -> list[str]:
    """Generatable family ids that reuse an existing INV ``families.py`` id."""
    inv = _inv_family_ids()
    return [s.family_id for s in _SPECS if s.family_id in inv]


def family_cross_reference() -> list[dict[str, Any]]:
    """Per-family {family_id, inv_family_id, is_new, canonical_form, display} rows.

    ``inv_family_id`` is the coarse INV family the specific generated family
    rolls up to (or ``None``); ``is_new`` reflects whether the *specific*
    ``family_id`` string already exists in ``families.py``.
    """
    inv = _inv_family_ids()
    rows: list[dict[str, Any]] = []
    for s in _SPECS:
        rows.append(
            {
                "family_id": s.family_id,
                "display_name": s.display_name,
                "inv_family_id": s.inv_family_id,
                "inv_family_known": (s.inv_family_id in inv) if s.inv_family_id else False,
                "is_new": s.family_id not in inv,
                "canonical_form": s.canonical_form,
            }
        )
    return rows
