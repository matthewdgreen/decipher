"""Shared substrate for the INV model-diagnosis experiment harness.

This module is the non-throwaway core shared by the suite generator
(``scripts/build_inv_diagnosis_suite.py``), the thin diagnosis runner
(``scripts/run_inv_model_experiment.py``), and the scorer/report
(``scripts/report_inv_model_experiment.py``).

Dependency posture (see ``docs/specs/inv_model_diagnosis_experiment_spec.md``):
full INV-0 mode is not yet implemented, so the runner uses a *pluggable v1
diagnosis task* — a dedicated thin entry with an OBSERVE-ONLY tool subset plus
a new ``meta_declare_diagnosis`` tool as the parse target. Only the task adapter
changes between v1 (this file's runner) and the eventual v2 (INV-0's real
``DiagnosisReport``). The canonical family enum, truth/suspicion mappings,
firewall, and scoring here are reused unchanged by v2.

Nothing in this module imports ``agent.tools_v2`` (kept light so the schema /
firewall / scoring tests do not drag in the full tool executor). The runner
imports ``TOOL_DEFINITIONS`` directly for the observe-tool schemas.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Canonical family enum (F2)
# ---------------------------------------------------------------------------


class DiagnosisFamily(str, Enum):
    """The canonical cipher-family vocabulary for this experiment.

    Three separate vocabularies exist in the codebase and must be reconciled:
      * builder ``cipher_system`` strings (``simple_substitution`` etc.)
      * fingerprint suspicion-score keys (``polyalphabetic_vigenere`` etc.)
      * model free text (constrained here to the ``meta_declare_diagnosis`` enum)

    ``UNKNOWN`` is the honest "no single standard family fits" option. It is the
    correct answer only for the composite anchoring probe; for every true single
    family it is scored as a miss.
    """

    MONOALPHABETIC_SUBSTITUTION = "monoalphabetic_substitution"
    HOMOPHONIC_SUBSTITUTION = "homophonic_substitution"
    VIGENERE = "vigenere"
    COLUMNAR_TRANSPOSITION = "columnar_transposition"
    TRANSPOSITION_SUBSTITUTION = "transposition_substitution"
    UNKNOWN = "unknown"


#: The composite / off-menu true family (the anchoring probe).
COMPOSITE_FAMILY = DiagnosisFamily.TRANSPOSITION_SUBSTITUTION

#: Ordered list of family values offered to the model in the declare tool enum.
FAMILY_ENUM_VALUES: list[str] = [f.value for f in DiagnosisFamily]

#: Builder ``cipher_system`` string -> canonical family. Ground-truth labels.
TRUTH_LABEL_TO_FAMILY: dict[str, DiagnosisFamily] = {
    "simple_substitution": DiagnosisFamily.MONOALPHABETIC_SUBSTITUTION,
    "homophonic_substitution": DiagnosisFamily.HOMOPHONIC_SUBSTITUTION,
    "vigenere": DiagnosisFamily.VIGENERE,
    "pure_transposition": DiagnosisFamily.COLUMNAR_TRANSPOSITION,
    "transposition_substitution": DiagnosisFamily.TRANSPOSITION_SUBSTITUTION,
}

#: Fingerprint suspicion-key -> canonical family. Keys with no clean single-
#: family mapping (``transposition_homophonic``, ``playfair``, ``unknown``) fall
#: through to ``UNKNOWN`` so the automated baseline scores them as a miss for
#: every single-family truth (and, correctly, it can never *name* the composite).
SUSPICION_KEY_TO_FAMILY: dict[str, DiagnosisFamily] = {
    "monoalphabetic_substitution": DiagnosisFamily.MONOALPHABETIC_SUBSTITUTION,
    "homophonic_substitution": DiagnosisFamily.HOMOPHONIC_SUBSTITUTION,
    "polyalphabetic_vigenere": DiagnosisFamily.VIGENERE,
    "transposition": DiagnosisFamily.COLUMNAR_TRANSPOSITION,
}


def family_from_truth_label(cipher_system: str) -> DiagnosisFamily:
    """Map a builder ``cipher_system`` string to the canonical family enum."""
    try:
        return TRUTH_LABEL_TO_FAMILY[cipher_system]
    except KeyError as exc:  # pragma: no cover - guards suite drift
        raise ValueError(
            f"Unmapped builder cipher_system {cipher_system!r}; update "
            "TRUTH_LABEL_TO_FAMILY in experiments/inv_diagnosis.py"
        ) from exc


def family_from_suspicion_key(key: str) -> DiagnosisFamily:
    """Map a fingerprint suspicion key to the canonical family (default UNKNOWN)."""
    return SUSPICION_KEY_TO_FAMILY.get(key, DiagnosisFamily.UNKNOWN)


def map_argmax_suspicion(suspicion_scores: dict[str, float]) -> DiagnosisFamily:
    """Return the canonical family for the highest-weight suspicion key.

    Ties break by the ``SUSPICION_KEY_TO_FAMILY`` insertion order first, then
    alphabetically, so the mapping is fully deterministic. Empty input -> UNKNOWN.
    """
    if not suspicion_scores:
        return DiagnosisFamily.UNKNOWN
    order = {k: i for i, k in enumerate(SUSPICION_KEY_TO_FAMILY)}

    def sort_key(item: tuple[str, float]) -> tuple[float, float, str]:
        name, score = item
        return (-float(score), order.get(name, len(order)), name)

    argmax_key = sorted(suspicion_scores.items(), key=sort_key)[0][0]
    return family_from_suspicion_key(argmax_key)


def coerce_family(value: Any) -> DiagnosisFamily | None:
    """Best-effort coercion of a model-declared family string to the enum.

    Returns ``None`` when the value cannot be mapped (never guesses a family).
    """
    if value is None:
        return None
    if isinstance(value, DiagnosisFamily):
        return value
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    for fam in DiagnosisFamily:
        if text == fam.value:
            return fam
    # A couple of forgiving aliases the model might emit despite the enum.
    aliases = {
        "polyalphabetic_vigenere": DiagnosisFamily.VIGENERE,
        "polyalphabetic": DiagnosisFamily.VIGENERE,
        "vigenere_cipher": DiagnosisFamily.VIGENERE,
        "monoalphabetic": DiagnosisFamily.MONOALPHABETIC_SUBSTITUTION,
        "simple_substitution": DiagnosisFamily.MONOALPHABETIC_SUBSTITUTION,
        "substitution": DiagnosisFamily.MONOALPHABETIC_SUBSTITUTION,
        "homophonic": DiagnosisFamily.HOMOPHONIC_SUBSTITUTION,
        "columnar": DiagnosisFamily.COLUMNAR_TRANSPOSITION,
        "transposition": DiagnosisFamily.COLUMNAR_TRANSPOSITION,
        "pure_transposition": DiagnosisFamily.COLUMNAR_TRANSPOSITION,
        "composite": DiagnosisFamily.TRANSPOSITION_SUBSTITUTION,
        "substitution_transposition": DiagnosisFamily.TRANSPOSITION_SUBSTITUTION,
        "none": DiagnosisFamily.UNKNOWN,
        "no_single_family": DiagnosisFamily.UNKNOWN,
        "other": DiagnosisFamily.UNKNOWN,
    }
    return aliases.get(text)


# ---------------------------------------------------------------------------
# Scoring (F2 composite credit rule + Part 3 metrics)
# ---------------------------------------------------------------------------

#: Default confidence at/above which a wrong family call is treated as a
#: confident (pareidolia) assertion.
PAREIDOLIA_CONFIDENCE_THRESHOLD = 0.7


def is_diagnosis_correct(
    true_family: DiagnosisFamily | str,
    declared_family: DiagnosisFamily | str | None,
) -> bool:
    """Did the model name the true family?

    Composite credit rule (F2): for the composite case, credit either a literal
    ``transposition_substitution`` naming OR an honest ``unknown`` ("no single
    standard family fits"). For every single family, only an exact match counts.
    """
    true_fam = true_family if isinstance(true_family, DiagnosisFamily) else DiagnosisFamily(true_family)
    decl = declared_family
    if decl is not None and not isinstance(decl, DiagnosisFamily):
        decl = coerce_family(decl)
    if decl is None:
        return False
    if true_fam == COMPOSITE_FAMILY:
        return decl in (COMPOSITE_FAMILY, DiagnosisFamily.UNKNOWN)
    return decl == true_fam


def is_pareidolia(
    true_family: DiagnosisFamily | str,
    declared_family: DiagnosisFamily | str | None,
    confidence: float | None,
    *,
    threshold: float = PAREIDOLIA_CONFIDENCE_THRESHOLD,
) -> bool:
    """Did the model confidently assert a *wrong* family (the bad-basin failure)?

    An honest ``unknown`` is never pareidolia. A missing declaration is never
    pareidolia (nothing was asserted). Missing confidence is treated as high so
    an unqualified wrong assertion still counts.
    """
    decl = declared_family
    if decl is not None and not isinstance(decl, DiagnosisFamily):
        decl = coerce_family(decl)
    if decl is None or decl == DiagnosisFamily.UNKNOWN:
        return False
    if is_diagnosis_correct(true_family, decl):
        return False
    conf = 1.0 if confidence is None else float(confidence)
    return conf >= threshold


# ---------------------------------------------------------------------------
# Firewall (Part 4, F4)
# ---------------------------------------------------------------------------

#: Compound builder test_id family-flag tokens (F4). These are the exact
#: ``cipher_flag`` fragments the builder bakes into seeded test_ids
#: (honb/thonb/ptnb/tnb/q3nb/vignb/...). They encode the family and must never
#: reach a model prompt. Only COMPOUND flags are listed: bare fragments like
#: "vig"/"nb"/"wb" are deliberately excluded because they are substrings of the
#: legitimate candidate-family menu ("vigenere") and of ordinary text, which
#: would make the firewall false-positive. The full builder test_id is checked
#: separately and is the primary family-encoding leak vector.
FAMILY_FLAG_TOKENS: tuple[str, ...] = (
    "honb",
    "thonb",
    "ptnb",
    "tnb",
    "q3nb",
    "q3wb",
    "vignb",
    "vigwb",
    "bfnb",
    "vbfnb",
    "grnb",
)


class GroundTruthLeak(AssertionError):
    """Raised when a rendered prompt leaks firewalled ground truth."""


def assert_no_ground_truth_leak(
    rendered_prompt: str,
    *,
    true_family: str,
    plaintext: str,
    builder_test_id: str = "",
    builder_description: str = "",
    extra_tokens: tuple[str, ...] = (),
) -> None:
    """Assert a rendered model prompt contains no firewalled ground truth.

    Checks (case-insensitive) that the prompt does NOT contain:
      * the true family enum value,
      * the plaintext (full string and a representative long slice),
      * the builder test_id (which encodes the family) and its flag tokens,
      * the builder description string (which names the mechanism),
      * any caller-supplied extra tokens.

    Raises ``GroundTruthLeak`` on the first violation.
    """
    haystack = rendered_prompt.lower()
    violations: list[str] = []

    def _check(needle: str, label: str) -> None:
        n = (needle or "").strip().lower()
        if n and n in haystack:
            violations.append(f"{label}: {needle!r}")

    _check(true_family, "true_family")

    pt = (plaintext or "").strip()
    if pt:
        _check(pt, "plaintext(full)")
        # A representative interior slice catches accidental partial inclusion
        # without the false positives a tiny fragment would produce against a
        # transposition ciphertext (same letter multiset).
        if len(pt) >= 40:
            mid = len(pt) // 2
            _check(pt[mid : mid + 40], "plaintext(slice)")

    _check(builder_test_id, "builder_test_id")
    _check(builder_description, "builder_description")
    for token in (*FAMILY_FLAG_TOKENS, *extra_tokens):
        _check(token, "family_flag")

    if violations:
        raise GroundTruthLeak(
            "Ground-truth leak in rendered prompt: " + "; ".join(violations)
        )


# ---------------------------------------------------------------------------
# Observe-only tool subset + declare tool (F1, F8)
# ---------------------------------------------------------------------------

#: The OBSERVE-ONLY tool subset the v1 diagnosis task exposes. No decode/act/
#: search tools; no fingerprint injection. ``observe_cipher_id`` is the correct
#: tool name (the RunArtifact *field* is ``cipher_id_report`` — F8).
OBSERVE_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "observe_cipher_shape",
        "observe_frequency",
        "observe_ic",
        "observe_kasiski",
        "observe_periodic_ic",
        "observe_phase_frequency",
        "observe_periodic_shift_candidates",
        "observe_cipher_id",
    }
)

#: The new parse-target tool. Defined here (not in tools_v2) so the harness owns
#: it and the shared-file reconciliation surface stays minimal. The runner routes
#: this call itself rather than through ``WorkspaceToolExecutor``.
META_DECLARE_DIAGNOSIS_TOOL: dict[str, Any] = {
    "name": "meta_declare_diagnosis",
    "description": (
        "Declare your cipher-family diagnosis and STOP. Do not attempt a full "
        "decryption. `family` MUST be one of the enum values. Use "
        "`transposition_substitution` if you believe two mechanisms are combined, "
        "and `unknown` if no single standard family fits the evidence. "
        "`confidence` is your own 0.0-1.0 certainty in the named family."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "family": {
                "type": "string",
                "enum": FAMILY_ENUM_VALUES,
                "description": "The single most likely cipher family.",
            },
            "recommended_attack": {
                "type": "string",
                "description": "The concrete next attack you would run to solve it.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Your own certainty in the named family (0.0-1.0).",
            },
            "rationale": {
                "type": "string",
                "description": "Brief evidence for the diagnosis (optional).",
            },
        },
        "required": ["family", "recommended_attack", "confidence"],
    },
}


# ---------------------------------------------------------------------------
# Suite definition (Part 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteRow:
    """One curated suite row: the family, its build recipe, and neutral id.

    ``case_id`` is neutral and OVERRIDES the builder test_id (which encodes the
    family). ``spec_kwargs`` are passed straight to ``TestSpec``.
    """

    case_id: str
    true_family: str          # canonical DiagnosisFamily value
    expected_cipher_system: str
    alphabet_class: str       # "letters" | "digits"
    length_band: str
    topic: str
    seed: int
    notes: str
    spec_kwargs: dict[str, Any] = field(default_factory=dict)


def _columnar_pipeline(key: str) -> dict[str, Any]:
    return {"steps": [{"name": "Transposition", "data": {"key": key}}]}


#: The curated suite. Distinct ``topic`` per row => distinct plaintext (the
#: PlaintextCache key excludes seed and word_boundaries — F5). Fixed seeds keep
#: the cipher keys reproducible.
SUITE_ROWS: list[SuiteRow] = [
    SuiteRow(
        case_id="inv_diag_case_01",
        true_family=DiagnosisFamily.MONOALPHABETIC_SUBSTITUTION.value,
        expected_cipher_system="simple_substitution",
        alphabet_class="letters",
        length_band="medium",
        topic="medieval_scriptoria",
        seed=101,
        notes="Monoalphabetic substitution WITH word boundaries.",
        spec_kwargs={"approx_length": 150, "word_boundaries": True},
    ),
    SuiteRow(
        case_id="inv_diag_case_02",
        true_family=DiagnosisFamily.MONOALPHABETIC_SUBSTITUTION.value,
        expected_cipher_system="simple_substitution",
        alphabet_class="letters",
        length_band="medium",
        topic="alpine_botany",
        seed=102,
        notes="Monoalphabetic substitution NO word boundaries.",
        spec_kwargs={"approx_length": 150, "word_boundaries": False},
    ),
    SuiteRow(
        case_id="inv_diag_case_03",
        true_family=DiagnosisFamily.HOMOPHONIC_SUBSTITUTION.value,
        expected_cipher_system="homophonic_substitution",
        alphabet_class="digits",
        length_band="medium",
        topic="harbor_trade_ledgers",
        seed=103,
        notes="Homophonic substitution (2-digit tokens), no boundaries.",
        spec_kwargs={"approx_length": 160, "word_boundaries": False, "homophonic": True},
    ),
    SuiteRow(
        case_id="inv_diag_case_04",
        true_family=DiagnosisFamily.VIGENERE.value,
        expected_cipher_system="vigenere",
        alphabet_class="letters",
        length_band="medium",
        topic="clockmaking_guilds",
        seed=104,
        notes="Periodic polyalphabetic (Vigenere), period 6, no boundaries.",
        spec_kwargs={
            "approx_length": 155,
            "word_boundaries": False,
            "polyalphabetic_variant": "vigenere",
            "polyalphabetic_period": 6,
        },
    ),
    SuiteRow(
        case_id="inv_diag_case_05",
        true_family=DiagnosisFamily.COLUMNAR_TRANSPOSITION.value,
        expected_cipher_system="pure_transposition",
        alphabet_class="letters",
        length_band="medium",
        topic="river_navigation",
        seed=105,
        notes="Keyed columnar transposition, no boundaries.",
        spec_kwargs={
            "approx_length": 145,
            "word_boundaries": False,
            "transposition_only": True,
            "transform_pipeline": _columnar_pipeline("CIPHERK"),
        },
    ),
    SuiteRow(
        case_id="inv_diag_case_06",
        true_family=DiagnosisFamily.TRANSPOSITION_SUBSTITUTION.value,
        expected_cipher_system="transposition_substitution",
        alphabet_class="letters",
        length_band="medium",
        topic="observatory_records",
        seed=106,
        notes="Composite anchoring probe: substitution THEN columnar transposition.",
        spec_kwargs={
            "approx_length": 145,
            "word_boundaries": False,
            "transform_pipeline": _columnar_pipeline("VOYAGERS"),
        },
    ),
]


def build_spec(row: SuiteRow) -> Any:
    """Construct the ``TestSpec`` for a suite row (imported lazily)."""
    from testgen.spec import TestSpec

    return TestSpec(
        language="en",
        topic=row.topic,
        seed=row.seed,
        **row.spec_kwargs,
    )


def plaintext_hash(plaintext: str) -> str:
    """Stable short hash of a scored plaintext (F5 cache-regeneration detector)."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()[:16]


def build_case(
    row: SuiteRow,
    cache: Any,
    *,
    api_key: str = "",
    generator: Any = None,
    generator_provider: str = "openai",
    generator_model: str | None = None,
    offline: bool = False,
) -> Any:
    """Build (or load-from-cache) a suite row's ``TestData``, firewalled.

    Applies the neutral ``case_id`` as an OVERRIDE of the builder test_id (F4),
    and asserts the produced ``cipher_system`` matches the row's expectation so a
    silent builder change is caught. ``offline=True`` refuses to hit the LLM: it
    raises on a cache miss (used by CI/deterministic regen).
    """
    from testgen.builder import build_test_case

    spec = build_spec(row)
    if offline and generator is None and cache.get(spec) is None:
        raise RuntimeError(
            f"offline: plaintext cache miss for {row.case_id} (topic={row.topic!r}); "
            "run scripts/build_inv_diagnosis_suite.py once to populate the cache."
        )
    test_data = build_test_case(
        spec,
        cache,
        api_key=api_key,
        generator=generator,
        generator_provider=generator_provider,
        generator_model=generator_model,
    )
    if test_data.test.cipher_system != row.expected_cipher_system:
        raise RuntimeError(
            f"{row.case_id}: builder produced cipher_system "
            f"{test_data.test.cipher_system!r}, expected "
            f"{row.expected_cipher_system!r} — suite/builder drift."
        )
    # Neutral case_id overrides the family-encoding builder test_id (F4).
    test_data.test.test_id = row.case_id
    return test_data


def manifest_row(row: SuiteRow, test_data: Any) -> dict[str, Any]:
    """Build the JSONL manifest record for a suite row + its built ``TestData``."""
    spec = build_spec(row)
    synthetic_spec: dict[str, Any] = {
        "language": spec.language,
        "topic": spec.topic,
        "seed": spec.seed,
        "approx_length": spec.approx_length,
        "word_boundaries": spec.word_boundaries,
        "homophonic": spec.homophonic,
        "transposition_only": spec.transposition_only,
        "polyalphabetic_variant": spec.polyalphabetic_variant,
        "polyalphabetic_period": spec.polyalphabetic_period,
        "transform_pipeline": spec.transform_pipeline,
    }
    return {
        "case_id": row.case_id,
        "cipher_system": test_data.test.cipher_system,
        "seed": row.seed,
        "true_family": row.true_family,
        "alphabet_class": row.alphabet_class,
        "length_band": row.length_band,
        "notes": row.notes,
        "plaintext_sha256": plaintext_hash(test_data.plaintext),
        "topic": row.topic,
        "synthetic_spec": synthetic_spec,
    }


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

#: Required manifest keys, validated by the suite schema test.
MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "case_id",
    "cipher_system",
    "seed",
    "true_family",
    "alphabet_class",
    "length_band",
    "notes",
    "plaintext_sha256",
    "topic",
    "synthetic_spec",
)
