"""Family + discriminator registry for investigator mode (INV-0 Part 1).

The registry is the fixed vocabulary the diagnosis layer ranks over. It names
the cipher *families* (primaries, subtypes, and composable modifiers), the
*discriminators* (cheap tests that separate a confusable pair), and the mapping
from ``analysis.cipher_id`` suspicion keys onto family ids.

Solver-status table (set from the ACTUAL tool inventory at implementation time)
------------------------------------------------------------------------------
==============================  =================  =======================================
family                          solver_status      backing tool / status
==============================  =================  =======================================
monoalphabetic_substitution     solves_automated   automated hill-climb / anneal solver
homophonic_substitution         solves_automated   zenith_native (English, boundary-sep)
polyalphabetic_periodic         agent_assists      periodic-IC + Vigenere-family search
transposition                   agent_assists      transform_search columnar/route search
transposition_homophonic        diagnoses_only     no dedicated solver yet
fractionation_transposition     diagnoses_only     no dedicated solver yet
playfair                        diagnoses_only     no dedicated solver yet
polygraphic_substitution        diagnoses_only     no dedicated solver yet
nomenclator_codebook            diagnoses_only     needs key-text search (out of INV-0)
numeric_book_cipher             diagnoses_only     needs key-text search (out of INV-0)
plaintext_or_hoax               diagnoses_only     structural verdict only
unknown_custom                  unsupported        no model
quagmire_keyed (subtype)        agent_assists      search_quagmire3_keyword_alphabet
numeric_word_position (subtype) diagnoses_only     P8 signature only
numeric_char_position (subtype) diagnoses_only     P8 signature only
numeric_skip_nth_word (subtype) diagnoses_only     P8 signature only
nulls_noise_layer (modifier)    diagnoses_only     composes with any family
==============================  =================  =======================================

Import by module path (``from investigation import families``); do NOT touch
``investigation/__init__.py`` (M1-scoped, a three-spec merge point — finding 13).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "FamilySpec",
    "DiscriminatorSpec",
    "FAMILY_REGISTRY",
    "DISCRIMINATOR_REGISTRY",
    "SUSPICION_TO_FAMILY",
    "PRIMARY_IDS",
    "SUBTYPE_IDS",
    "MODIFIER_IDS",
    "SUBSTITUTION_PRIMARIES",
    "family_discriminators",
]


@dataclass(frozen=True)
class FamilySpec:
    id: str
    display_name: str
    parent: str | None
    role: str                       # primary | subtype | modifier
    solver_status: str              # solves_automated | agent_assists | diagnoses_only | planned | unsupported
    detectors: tuple[str, ...]      # atom observation names bearing on this family
    discriminators: tuple[str, ...]  # DiscriminatorSpec ids that name this family
    confusable_with: tuple[str, ...]  # symmetric confusable set
    notes: str
    min_tokens_diagnose: int
    # Optional client-facing sequencing hint (display-only; never a score input).
    # The panels cannot statically tell keyed from standard tableaux, so this is
    # a routing suggestion, not evidence. Default empty for every family.
    sequencing_hint: str = ""


@dataclass(frozen=True)
class DiscriminatorSpec:
    id: str
    description: str
    splits: tuple[str, str]
    depends_on_panels: tuple[str, ...]
    tool: str | None
    status: str                     # available | planned


# ---------------------------------------------------------------------------
# Discriminator inventory
# ---------------------------------------------------------------------------
# The first seven are the concrete AVAILABLE inventory (finding 2; required by
# the fixtures). The remaining four are `planned` covers so every primary lists
# at least one discriminator (registry invariant); they are never `run`, carry
# no confusable_with weight, and are invisible to verdict rules (c)/(d).
# RECORDED DEVIATION (see resources/reference/README.md, "INV-0 deviations
# record"): the four `planned` covers go beyond the spec's 7-entry inventory —
# the 7 leave 5 primaries uncovered, a spec-internal contradiction resolved via
# the spec's own `planned` status. Verified they cannot flip any verdict
# (`planned` -> `unavailable`, never `pending`, for rule (c); no confusable
# pair among their splits for rule (d)).
_DISCRIMINATORS: tuple[DiscriminatorSpec, ...] = (
    DiscriminatorSpec(
        "disc_mono_transp",
        "monoalphabetic vs transposition (letter order preserved or substituted)",
        ("monoalphabetic_substitution", "transposition"),
        ("order_layout",),
        "observe_transform_suspicion",
        "available",
    ),
    DiscriminatorSpec(
        "disc_mono_homophonic",
        "monoalphabetic vs homophonic (symbol inventory + flatness)",
        ("monoalphabetic_substitution", "homophonic_substitution"),
        ("shape", "frequency"),
        "observe_homophone_distribution",
        "available",
    ),
    DiscriminatorSpec(
        "disc_sub_periodic",
        "monoalphabetic vs polyalphabetic-periodic (periodic IC recovery)",
        ("monoalphabetic_substitution", "polyalphabetic_periodic"),
        ("periodicity",),
        "observe_periodic_ic",
        "available",
    ),
    DiscriminatorSpec(
        "disc_periodic_quagmire",
        "plain periodic vs keyed-tableau (Quagmire) polyalphabetic",
        ("polyalphabetic_periodic", "quagmire_keyed"),
        ("periodicity",),
        "search_quagmire3_keyword_alphabet",
        "available",
    ),
    DiscriminatorSpec(
        "disc_homo_transphomo",
        "homophonic vs transposition-homophonic (order + polygraphic structure)",
        ("homophonic_substitution", "transposition_homophonic"),
        ("order_layout", "polygraphic"),
        "observe_transform_suspicion",
        "available",
    ),
    DiscriminatorSpec(
        "disc_numeric_book_hoax",
        "numeric book cipher vs plaintext/hoax numeric artifact",
        ("numeric_book_cipher", "plaintext_or_hoax"),
        ("numeric_code",),
        None,
        "available",
    ),
    DiscriminatorSpec(
        "disc_book_word_char",
        "numeric word-position vs char-position book indexing",
        ("numeric_word_position", "numeric_char_position"),
        ("numeric_code",),
        None,
        "available",
    ),
    DiscriminatorSpec(
        "disc_sub_transp_composite",
        "plain substitution vs substitution+transposition (letter accuracy high, "
        "word structure absent)",
        ("monoalphabetic_substitution", "substitution_transposition"),
        ("order_layout",),
        "observe_transform_suspicion",       # existing tool; extended in 2.3
        "available",
    ),
    # --- planned covers (registry invariant only) ---
    DiscriminatorSpec(
        "disc_transp_fractionation",
        "plain transposition vs fractionation+transposition (planned)",
        ("transposition", "fractionation_transposition"),
        ("order_layout", "polygraphic"),
        None,
        "planned",
    ),
    DiscriminatorSpec(
        "disc_playfair_polygraphic",
        "Playfair vs general polygraphic substitution (planned)",
        ("playfair", "polygraphic_substitution"),
        ("polygraphic",),
        None,
        "planned",
    ),
    DiscriminatorSpec(
        "disc_nomenclator_book",
        "nomenclator/codebook vs numeric book cipher (planned; needs key-text search)",
        ("nomenclator_codebook", "numeric_book_cipher"),
        ("numeric_code",),
        None,
        "planned",
    ),
    DiscriminatorSpec(
        "disc_unknown_nomenclator",
        "unknown/custom construction vs nomenclator/codebook (planned)",
        ("unknown_custom", "nomenclator_codebook"),
        ("frequency",),
        None,
        "planned",
    ),
)

DISCRIMINATOR_REGISTRY: dict[str, DiscriminatorSpec] = {d.id: d for d in _DISCRIMINATORS}


def family_discriminators(family_id: str) -> list[str]:
    """Discriminator ids naming ``family_id``, in registry insertion order."""
    return [d.id for d in _DISCRIMINATORS if family_id in d.splits]


# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------
_FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "monoalphabetic_substitution", "Monoalphabetic substitution", None, "primary",
        "solves_automated",
        ("fingerprint_prior:monoalphabetic_substitution", "ic_near_language_reference",
         "peaked_monogram_shape", "letters_substituted"),
        ("disc_mono_transp", "disc_mono_homophonic", "disc_sub_periodic",
         "disc_sub_transp_composite"),
        ("transposition", "homophonic_substitution", "polyalphabetic_periodic",
         "substitution_transposition"),
        "Single fixed cipher-alphabet; peaked frequencies, language-reference IC.",
        40,
    ),
    FamilySpec(
        "homophonic_substitution", "Homophonic substitution", None, "primary",
        "solves_automated",
        ("fingerprint_prior:homophonic_substitution", "depressed_ic",
         "flat_high_entropy", "large_symbol_inventory"),
        ("disc_mono_homophonic", "disc_homo_transphomo"),
        ("monoalphabetic_substitution", "transposition_homophonic"),
        "Many symbols per plaintext letter; flattened frequencies, depressed IC.",
        40,
    ),
    FamilySpec(
        "polyalphabetic_periodic", "Polyalphabetic (periodic)", None, "primary",
        "agent_assists",
        ("fingerprint_prior:polyalphabetic_periodic", "depressed_ic", "periodic_ic_recovery"),
        ("disc_sub_periodic", "disc_periodic_quagmire"),
        ("monoalphabetic_substitution",),
        "Repeating key; depressed IC that recovers at the key period.",
        40,
    ),
    FamilySpec(
        "transposition", "Transposition", None, "primary",
        "agent_assists",
        ("fingerprint_prior:transposition", "ic_near_language_reference",
         "peaked_monogram_shape", "letters_unsubstituted"),
        ("disc_mono_transp", "disc_transp_fractionation"),
        ("monoalphabetic_substitution", "substitution_transposition"),
        "Letters unchanged but reordered; language frequencies, broken n-grams.",
        40,
    ),
    FamilySpec(
        "substitution_transposition", "Substitution + transposition", None, "primary",
        "agent_assists",
        ("fingerprint_prior:substitution_transposition",
         "residual_order_after_substitution", "peaked_monogram_shape"),
        ("disc_sub_transp_composite",),
        ("monoalphabetic_substitution", "transposition"),   # confusable_with
        "Substitution then transposition: peaked/displaced frequencies (substitution) "
        "with absent n-gram structure (order layer). High letter accuracy but no "
        "word structure when solved as plain substitution.",
        38,
        sequencing_hint=(
            "A plain-substitution solve with strong frequency/quadgram fit but zero "
            "dictionary words implies an unpeeled order layer — screen transpositions "
            "over the DECODED stream (peel-and-solve) before rejecting."
        ),
    ),
    FamilySpec(
        "transposition_homophonic", "Transposition + homophonic", None, "primary",
        "diagnoses_only",
        ("fingerprint_prior:transposition_homophonic", "large_symbol_inventory"),
        ("disc_homo_transphomo",),
        ("homophonic_substitution",),
        "Homophonic substitution layered under a transposition.",
        60,
    ),
    FamilySpec(
        "fractionation_transposition", "Fractionation + transposition", None, "primary",
        "diagnoses_only",
        (),
        ("disc_transp_fractionation",),
        (),
        "Symbols split into fractions (e.g. ADFGVX/bifid) then transposed.",
        60,
    ),
    FamilySpec(
        "playfair", "Playfair / digraphic", None, "primary",
        "diagnoses_only",
        ("fingerprint_prior:playfair",),
        ("disc_playfair_polygraphic",),
        (),
        "Digraphic substitution; even length, near-zero doubled digraphs.",
        40,
    ),
    FamilySpec(
        "polygraphic_substitution", "Polygraphic substitution", None, "primary",
        "diagnoses_only",
        (),
        ("disc_playfair_polygraphic",),
        (),
        "Block substitution over letter groups (Hill, larger polygraphic).",
        60,
    ),
    FamilySpec(
        "nomenclator_codebook", "Nomenclator / codebook", None, "primary",
        "diagnoses_only",
        ("numeric_token_stream",),
        ("disc_nomenclator_book", "disc_unknown_nomenclator"),
        (),
        "Mixed cipher + code list; numeric or symbol tokens map to words/phrases.",
        40,
    ),
    FamilySpec(
        "numeric_book_cipher", "Numeric book cipher", None, "primary",
        "diagnoses_only",
        ("numeric_token_stream", "book_keylength_plausible", "front_loading_present"),
        ("disc_numeric_book_hoax", "disc_nomenclator_book"),
        ("plaintext_or_hoax",),
        "Integers index a key text (word/char position); Beale family.",
        40,
    ),
    FamilySpec(
        "plaintext_or_hoax", "Plaintext / hoax artifact", None, "primary",
        "diagnoses_only",
        ("independent_random_like", "structured_hoax_artifact"),
        ("disc_numeric_book_hoax",),
        ("numeric_book_cipher",),
        "No decipherable structure: random-like or a fabricated artifact.",
        40,
    ),
    FamilySpec(
        "unknown_custom", "Unknown / custom", None, "primary",
        "unsupported",
        (),
        ("disc_unknown_nomenclator",),
        (),
        "No family matched; a bespoke or unrecognised construction.",
        40,
    ),
    # ---- subtypes ----
    FamilySpec(
        "quagmire_keyed", "Quagmire / keyed-tableau", "polyalphabetic_periodic", "subtype",
        "agent_assists",
        ("periodic_ic_recovery",),
        ("disc_periodic_quagmire",),
        (),
        "Periodic polyalphabetic over a keyed alphabet (Quagmire I-IV, Kryptos).",
        60,
        sequencing_hint=(
            "If a standard-tableau Vigenere-family search fails at the indicated "
            "period, test a keyed-tableau/Quagmire search next before rejecting "
            "the polyalphabetic family."
        ),
    ),
    FamilySpec(
        "numeric_word_position", "Numeric — word position", "numeric_book_cipher", "subtype",
        "diagnoses_only",
        ("book_word_position_signature",),
        ("disc_book_word_char",),
        (),
        "Index selects the Nth word of the key text (first-letter book cipher).",
        40,
    ),
    FamilySpec(
        "numeric_char_position", "Numeric — char position", "numeric_book_cipher", "subtype",
        "diagnoses_only",
        ("book_char_position_signature",),
        ("disc_book_word_char",),
        (),
        "Index selects the Nth character of the key text.",
        40,
    ),
    FamilySpec(
        "numeric_skip_nth_word", "Numeric — skip Nth word", "numeric_book_cipher", "subtype",
        "diagnoses_only",
        ("book_skip_nth_signature",),
        (),
        (),
        "Index selects words under a skip-N stride over the key text.",
        40,
    ),
    # ---- modifiers ----
    FamilySpec(
        "nulls_noise_layer", "Nulls / noise layer", None, "modifier",
        "diagnoses_only",
        (),
        (),
        (),
        "Composes with any family; inserted null symbols. Never ranked as a primary.",
        40,
    ),
)

FAMILY_REGISTRY: dict[str, FamilySpec] = {f.id: f for f in _FAMILIES}

PRIMARY_IDS: tuple[str, ...] = tuple(f.id for f in _FAMILIES if f.role == "primary")
SUBTYPE_IDS: tuple[str, ...] = tuple(f.id for f in _FAMILIES if f.role == "subtype")
MODIFIER_IDS: tuple[str, ...] = tuple(f.id for f in _FAMILIES if f.role == "modifier")

# The nine substitution-bearing primaries both numeric atoms weaken (finding 1;
# substitution_transposition added in composite Slice A — it carries a substitution
# layer, so a numeric stream is equally inconsistent with it).
SUBSTITUTION_PRIMARIES: tuple[str, ...] = (
    "monoalphabetic_substitution",
    "homophonic_substitution",
    "polyalphabetic_periodic",
    "transposition",
    "substitution_transposition",
    "transposition_homophonic",
    "fractionation_transposition",
    "playfair",
    "polygraphic_substitution",
)

# The six cipher_id.suspicion_scores keys mapped onto registry ids. The seventh
# key "unknown" (returned as {"unknown": 1.0} for <10 tokens) maps to nothing —
# dropped, emits no prior atom (finding 1).
SUSPICION_TO_FAMILY: dict[str, str] = {
    "monoalphabetic_substitution": "monoalphabetic_substitution",
    "homophonic_substitution": "homophonic_substitution",
    "polyalphabetic_vigenere": "polyalphabetic_periodic",
    "transposition": "transposition",
    "substitution_transposition": "substitution_transposition",
    "transposition_homophonic": "transposition_homophonic",
    "playfair": "playfair",
}


# ---------------------------------------------------------------------------
# Import-time validation (Part 1)
# ---------------------------------------------------------------------------

def _validate_registry() -> None:
    ids = list(FAMILY_REGISTRY) + list(DISCRIMINATOR_REGISTRY)
    if len(ids) != len(set(ids)):
        raise ValueError("family/discriminator ids are not unique")

    # parents reference existing family ids
    for fam in _FAMILIES:
        if fam.parent is not None and fam.parent not in FAMILY_REGISTRY:
            raise ValueError(f"family {fam.id} has unknown parent {fam.parent}")

    # discriminator splits reference existing family ids; family.discriminators
    # match the registry-derived set.
    for disc in _DISCRIMINATORS:
        for fam_id in disc.splits:
            if fam_id not in FAMILY_REGISTRY:
                raise ValueError(f"discriminator {disc.id} splits unknown family {fam_id}")
    for fam in _FAMILIES:
        derived = tuple(family_discriminators(fam.id))
        if tuple(fam.discriminators) != derived:
            raise ValueError(
                f"family {fam.id} discriminators {fam.discriminators} "
                f"disagree with registry {derived}"
            )

    # confusables symmetric + reference existing ids
    for fam in _FAMILIES:
        for other in fam.confusable_with:
            if other not in FAMILY_REGISTRY:
                raise ValueError(f"family {fam.id} confusable with unknown {other}")
            if fam.id not in FAMILY_REGISTRY[other].confusable_with:
                raise ValueError(
                    f"confusable relation not symmetric: {fam.id} -> {other}"
                )

    # every primary lists >= 1 discriminator
    for fam in _FAMILIES:
        if fam.role == "primary" and not fam.discriminators:
            raise ValueError(f"primary {fam.id} has no discriminator")

    # SUSPICION_TO_FAMILY targets exist
    for key, fam_id in SUSPICION_TO_FAMILY.items():
        if fam_id not in FAMILY_REGISTRY:
            raise ValueError(f"suspicion key {key} maps to unknown family {fam_id}")


_validate_registry()
