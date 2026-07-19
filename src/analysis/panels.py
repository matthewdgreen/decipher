"""Statistical diagnostic panels (INV-0 Part 2).

Nine panels turn a raw token stream into a list of evidence *atoms* (plain dicts,
NOT the M1 ``investigation.state.EvidenceEntry`` — that unification is INV-1).
Each panel adapts existing analysis code where possible and adds only the genuinely
new math its row in the spec's reuse table calls for. Panels never abort the
battery: :func:`run_battery` isolates per-panel exceptions into a
``not_computable`` result.

Atom weights, reliabilities, and thresholds are the FIXED-mode numbers derived by
``scripts/research/calibrate_inv0_scoring.py`` (the reconciled source of truth).
Changing any of them means re-running that script and the nine fixtures.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from analysis import ngram
from analysis.cipher_id import (
    _chi2_vs_uniform,
    _normalized_entropy,
    estimate_fundamental_period,
    kasiski_report,
)
from analysis.frequency import LANGUAGE_LETTER_FREQS
from analysis.ic import index_of_coincidence
from analysis.language_guesser import LANGUAGE_IC_REFERENCES
from analysis.null_baseline import null_percentile
from analysis.numeric_code import (
    BENFORD_INCONSISTENT_EPSILON,
    BOOK_KEYLEN_MAX,
    FRONTLOAD_MIN,
    numeric_code_battery,
)

# --- Calibrated thresholds (FIXED catalog) ---
IC_NEAR = 0.012            # |dIC| < this -> ic_near_language_reference
IC_DEPRESSED = 0.020       # dIC < -this -> depressed_ic
PEAK_FLAT_MIN = 0.28       # chi2-flatness > this AND unique<=28 -> peaked_monogram_shape
PEAK_UNIQUE_MAX = 28
FLAT_FLAT_MAX = 0.10       # chi2-flatness < this -> flat_high_entropy
PERIODIC_RECOVERY_MIN = 0.010   # best_period_ic - ic > this (structural pre-filter)
PERIODIC_ABS_MARGIN = 0.015     # best_period_ic >= lang_ic_ref - this
MONO_CHI2_LOW = 300.0      # monogram chi2 vs language below -> letters unchanged
MONO_CHI2_HIGH = 500.0     # above -> letters substituted
QUAD_NLL_SCRAMBLED = -6.0  # quadgram mean-loglik below -> not coherent plaintext
NUMERIC_UNIQUE_RATE = 0.40  # unique/token > this + numeric -> substitution inconsistency
# Substitution-INVARIANT residual-order gate (composite substitution+transposition).
# NEW constant (composite Slice A). The by-english quadgram score (QUAD_NLL_SCRAMBLED)
# CANNOT separate a plain monoalphabetic substitution from a substitution that was
# further transposed: a random substitution scrambles the by-letter quadgram score
# identically for both (empirically ~-6.6 for each). What DOES separate them is the
# substitution-invariant n-gram *structure* ratio (ngram.ngram_structure_ratio):
# a substitution PRESERVES adjacency structure (ratio well above 1.0), a transposition
# DESTROYS it (ratio toward 1.0). residual_order_after_substitution fires only when the
# structure ratio is below this threshold (order layer present). Calibrated from
# corpus renderings: coherent/plain-mono English bigram-structure ratio >= ~1.46,
# columnar-transposed (incl. composite) <= ~1.15 mean; 1.35 sits safely between and
# below the plain-mono floor so residual never fires on an unsubstituted-order mono.
STRUCTURE_RATIO_ABSENT = 1.35
# The bigram-structure ratio is noisy on short text (too few repeats to estimate
# adjacency structure): a 45-100 char plain mono can dip below the threshold. The
# residual/composite signal therefore requires enough letters. At >=150 the
# plain-mono ratio stays comfortably above STRUCTURE_RATIO_ABSENT in calibration.
STRUCTURE_MIN_TOKENS = 150

_MIN_TOKENS_PERIODICITY = 40
_PERIODIC_N_SHUFFLES = 1000

PANEL_NAMES = (
    "shape", "frequency", "repetition", "periodicity", "order_layout",
    "nulls_noise", "polygraphic", "numeric_code", "language",
)


@dataclass
class PanelResult:
    panel: str
    status: str                 # ok | not_computable
    reason: str | None = None
    measurements: dict[str, Any] = field(default_factory=dict)
    atoms: list[dict[str, Any]] = field(default_factory=list)
    reliability: str = "not_computable"   # high | low | not_computable

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel": self.panel,
            "status": self.status,
            "reason": self.reason,
            "measurements": self.measurements,
            "atoms": self.atoms,
            "reliability": self.reliability,
        }


def _atom(
    observation: str,
    panel: str,
    weight: float,
    supports: list[str],
    weakens: list[str],
    *,
    reliability: str = "high",
    measurement: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
    interpretation: str = "",
) -> dict[str, Any]:
    return {
        "observation": observation,
        "panel": panel,
        "weight": weight,
        "measurement": measurement or {},
        "baseline": baseline,
        "reliability": reliability,
        "interpretation": interpretation,
        "supports": list(supports),
        "weakens": list(weakens),
    }


def _lang_ic_ref(language: str) -> float:
    entry = LANGUAGE_IC_REFERENCES.get(language, LANGUAGE_IC_REFERENCES["en"])
    return entry[1]


def _english_monogram_prob(language: str) -> dict[str, float] | None:
    ref = LANGUAGE_LETTER_FREQS.get(language)
    if not ref:
        return None
    total = sum(ref.values())
    if total <= 0:
        return None
    return {L: ref[L] / total for L in ref}


def _periodic_ic_table(tokens: list[int], max_period: int = 26) -> dict[int, float]:
    """Per-period mean IC of every-k-th-token streams, k>=2.

    Mirrors the periodic-IC block of ``cipher_id.compute_cipher_fingerprint``;
    ``index_of_coincidence`` ignores the alphabet-size argument, so the values
    are identical to ``fingerprint.periodic_ic``.
    """
    n = len(tokens)
    uniq = len(set(tokens))
    table: dict[int, float] = {}
    for k in range(2, min(max_period + 1, n // 4 + 2)):
        streams = [tokens[i::k] for i in range(k)]
        ics = [index_of_coincidence(s, max(uniq, 26)) for s in streams if len(s) >= 6]
        if ics:
            table[k] = sum(ics) / len(ics)
    return table


def _best_period_ic(tokens: list[int], max_period: int = 26) -> float:
    """Max over periods k>=2 of the mean IC of every-k-th-token streams.

    This is the significance statistic for "is there ANY periodic structure"
    (the shuffle-null tests it); it is independent of which integer period the
    harmonic folder ultimately reports.
    """
    table = _periodic_ic_table(tokens, max_period=max_period)
    return max(table.values(), default=0.0)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_frequency(tokens, *, alphabet_size, alphabet_class, language,
                    word_group_count=0, line_lengths=None, numeric_values=None,
                    letter_rendering=None, related_profile=None, max_period=26,
                    rng_namespace="") -> PanelResult:
    n = len(tokens)
    if n < 10:
        return PanelResult("frequency", "not_computable", reason="too_few_tokens")
    counts = Counter(tokens)
    unique = len(counts)
    ic = index_of_coincidence(list(tokens), max(alphabet_size, unique))
    lang_ic = _lang_ic_ref(language)
    ic_delta = ic - lang_ic
    flat_chi = _chi2_vs_uniform(counts, n) / n
    norm_h = _normalized_entropy(counts, n)

    numeric = alphabet_class == "numeric"
    atoms: list[dict[str, Any]] = []
    if not numeric:
        if abs(ic_delta) < IC_NEAR:
            atoms.append(_atom(
                "ic_near_language_reference", "frequency", 0.25,
                ["monoalphabetic_substitution", "transposition"], [],
                measurement={"ic": ic, "ic_delta": ic_delta},
                interpretation="IC within 0.012 of the language reference (frequencies preserved).",
            ))
        if ic_delta < -IC_DEPRESSED:
            atoms.append(_atom(
                "depressed_ic", "frequency", 0.25,
                ["polyalphabetic_periodic", "homophonic_substitution"],
                ["monoalphabetic_substitution", "transposition"],
                measurement={"ic": ic, "ic_delta": ic_delta},
                interpretation="IC depressed >0.02 below the language reference.",
            ))
        peaked = (flat_chi > PEAK_FLAT_MIN) and (unique <= PEAK_UNIQUE_MAX)
        flat = flat_chi < FLAT_FLAT_MAX
        if peaked:
            # A peaked, small-alphabet frequency profile is preserved by BOTH a
            # substitution and a transposition, so it also bears on the composite
            # substitution_transposition family (composite Slice A: the family
            # declares peaked_monogram_shape among its detectors). It still weakens
            # homophonic. The residual_order_after_substitution atom is what makes
            # the composite SPECIFIC; this is one shared corroborating signal.
            # Gate the composite credit on STRUCTURE_MIN_TOKENS: the composite
            # hypothesis rests on the (order-panel) structure statistic, which is
            # unreliable below that length, so short peaked mono/transposition
            # fixtures must NOT surface the composite as a competitor.
            peaked_supports = ["monoalphabetic_substitution", "transposition"]
            if n >= STRUCTURE_MIN_TOKENS:
                peaked_supports.append("substitution_transposition")
            atoms.append(_atom(
                "peaked_monogram_shape", "frequency", 0.30,
                peaked_supports,
                ["homophonic_substitution"],
                measurement={"flatness": flat_chi, "unique": unique},
                interpretation="Per-token chi-square flatness >0.28 with <=28 symbols (peaked).",
            ))
        if flat:
            atoms.append(_atom(
                "flat_high_entropy", "frequency", 0.30,
                ["homophonic_substitution"], ["monoalphabetic_substitution"],
                measurement={"flatness": flat_chi},
                interpretation="Near-uniform per-token chi-square flatness (<0.10).",
            ))
    return PanelResult(
        "frequency", "ok",
        measurements={
            "ic": ic, "ic_delta": ic_delta, "language_ic_reference": lang_ic,
            "flatness": flat_chi, "normalized_entropy": norm_h, "unique": unique,
        },
        atoms=atoms,
        reliability="high",
    )


def panel_shape(tokens, *, alphabet_size, alphabet_class, language,
                word_group_count=0, line_lengths=None, numeric_values=None,
                letter_rendering=None, related_profile=None, max_period=26,
                rng_namespace="") -> PanelResult:
    n = len(tokens)
    if n < 2:
        return PanelResult("shape", "not_computable", reason="too_few_tokens")
    counts = Counter(tokens)
    unique = len(counts)

    # Symbol-inventory drift across halves (new): chi2 of first-half vs
    # second-half symbol counts, with a shuffle-null percentile.
    half = n // 2
    first = Counter(tokens[:half])
    second = Counter(tokens[half:])
    drift_chi2 = 0.0
    syms = set(first) | set(second)
    for s in syms:
        a = first.get(s, 0)
        b = second.get(s, 0)
        exp = (a + b) / 2.0
        if exp > 0:
            drift_chi2 += (a - exp) ** 2 / exp + (b - exp) ** 2 / exp

    atoms: list[dict[str, Any]] = []
    numeric = alphabet_class == "numeric"
    if not numeric and unique > 26:
        w = min(0.45, 0.45 * (unique - 26) / 20.0)
        if w > 0:
            atoms.append(_atom(
                "large_symbol_inventory", "shape", w,
                ["homophonic_substitution"], ["monoalphabetic_substitution"],
                measurement={"unique": unique},
                interpretation="More distinct symbols than a 26-letter alphabet.",
            ))
    return PanelResult(
        "shape", "ok",
        measurements={
            "token_count": n, "unique_symbols": unique,
            "symbol_inventory_drift_chi2": drift_chi2,
        },
        atoms=atoms,
        reliability="high",
    )


def panel_repetition(tokens, *, alphabet_size, alphabet_class, language,
                     word_group_count=0, line_lengths=None, numeric_values=None,
                     letter_rendering=None, related_profile=None, max_period=26,
                     rng_namespace="") -> PanelResult:
    n = len(tokens)
    if n < 4:
        return PanelResult("repetition", "not_computable", reason="too_few_tokens")
    toklist = list(tokens)
    kas = kasiski_report(toklist, max_period=max_period)

    def _rep_bigrams(vs):
        grams = Counter(tuple(vs[i:i + 2]) for i in range(len(vs) - 1))
        return sum(c for c in grams.values() if c >= 2)

    observed = _rep_bigrams(toklist)
    base = null_percentile(
        _rep_bigrams, toklist, tail="upper", n_shuffles=200,
        statistic_name="repeated_bigram_excess", namespace=rng_namespace,
    )
    excess = observed - base["null_mean"]
    return PanelResult(
        "repetition", "ok",
        measurements={
            "kasiski_best_period": kas.get("best_period"),
            "repeated_bigram_count": observed,
            "repeated_bigram_null_mean": base["null_mean"],
            "repeated_bigram_excess": excess,
            "repeated_bigram_p": base["p_value"],
        },
        reliability="high" if base["p_value"] <= 0.05 else "low",
    )


def panel_periodicity(tokens, *, alphabet_size, alphabet_class, language,
                      word_group_count=0, line_lengths=None, numeric_values=None,
                      letter_rendering=None, related_profile=None, max_period=26,
                      rng_namespace="") -> PanelResult:
    n = len(tokens)
    if n < _MIN_TOKENS_PERIODICITY:
        return PanelResult("periodicity", "not_computable", reason="too_few_tokens")
    toklist = list(tokens)
    ic = index_of_coincidence(toklist, max(alphabet_size, len(set(toklist))))
    table = _periodic_ic_table(toklist, max_period=max_period)
    bpi = max(table.values(), default=0.0)
    lang_ic = _lang_ic_ref(language)
    recovery = bpi - ic

    # Fundamental period via harmonic folding (raw table left untouched). Kasiski
    # factor counts reconcile the folded estimate; harmonic folding never
    # changes the significance statistic ``bpi`` or the atom-firing conditions.
    kas = kasiski_report(toklist, max_period=max_period)
    kas_factors = {int(k): v for k, v in kas.get("factor_counts", {}).items()}
    fundamental, fund_detail = estimate_fundamental_period(
        table, n, max_period=max_period, kasiski_factors=kas_factors or None,
    )

    atoms: list[dict[str, Any]] = []
    reliability = "low"
    baseline = None
    structural_ok = recovery > PERIODIC_RECOVERY_MIN and bpi >= lang_ic - PERIODIC_ABS_MARGIN
    if structural_ok:
        baseline = null_percentile(
            lambda vs: _best_period_ic(vs, max_period=max_period),
            toklist, tail="upper", n_shuffles=_PERIODIC_N_SHUFFLES,
            statistic_name="periodic_best_ic", namespace=rng_namespace,
        )
        if baseline["p_value"] <= 0.05:
            reliability = "high"
            atoms.append(_atom(
                "periodic_ic_recovery", "periodicity", 0.45,
                ["polyalphabetic_periodic"],
                ["monoalphabetic_substitution", "transposition"],
                reliability="high",
                measurement={"best_period_ic": bpi, "ic": ic, "recovery": recovery,
                             "fundamental_period": fundamental},
                baseline=baseline,
                interpretation="Best-period IC recovers toward the language reference; "
                               "shuffle-null rejects at p<=0.05.",
            ))
    return PanelResult(
        "periodicity", "ok",
        measurements={
            "ic": ic, "best_period_ic": bpi, "recovery": recovery,
            "structural_recovery": structural_ok,
            "shuffle_p": baseline["p_value"] if baseline else None,
            "periodic_ic_table": {str(k): v for k, v in table.items()},
            "fundamental_period": fundamental,
            "naive_best_period": fund_detail.get("naive_best_period"),
            "fundamental_detail": fund_detail,
        },
        atoms=atoms,
        reliability=reliability,
    )


def panel_order_layout(tokens, *, alphabet_size, alphabet_class, language,
                       word_group_count=0, line_lengths=None, numeric_values=None,
                       letter_rendering=None, related_profile=None, max_period=26,
                       rng_namespace="") -> PanelResult:
    if alphabet_class != "letters" or not letter_rendering:
        return PanelResult(
            "order_layout", "not_computable",
            reason="requires letters alphabet_class with letter_rendering",
        )
    rendering = "".join(ch for ch in letter_rendering.upper() if "A" <= ch <= "Z")
    if not rendering:
        return PanelResult("order_layout", "not_computable", reason="no_letters_in_rendering")
    prob = _english_monogram_prob(language)
    if prob is None:
        return PanelResult("order_layout", "not_computable", reason="no_language_reference")

    n = len(rendering)
    c = Counter(rendering)
    mchi = 0.0
    for i in range(26):
        L = chr(65 + i)
        exp = prob.get(L, 0.0) * n
        if exp > 0:
            mchi += (c.get(L, 0) - exp) ** 2 / exp
    qnll = ngram.normalized_ngram_score(rendering, ngram.NGRAM_CACHE.get(language, 4), 4)

    # peaked flag re-derived here (mirrors frequency panel's threshold).
    counts = Counter(tokens)
    flat_chi = _chi2_vs_uniform(counts, len(tokens)) / len(tokens) if tokens else 0.0
    peaked = (flat_chi > PEAK_FLAT_MIN) and (len(counts) <= PEAK_UNIQUE_MAX)

    # Substitution-invariant residual-order structure (composite Slice A). A plain
    # monoalphabetic substitution preserves adjacency structure (ratio >> 1); a
    # further transposition destroys it (ratio -> 1). See STRUCTURE_RATIO_ABSENT.
    structure_ratio = ngram.ngram_structure_ratio(rendering, 2)
    residual = (
        len(rendering) >= STRUCTURE_MIN_TOKENS
        and mchi > MONO_CHI2_HIGH
        and peaked
        and structure_ratio < STRUCTURE_RATIO_ABSENT
    )

    atoms: list[dict[str, Any]] = []
    if mchi < MONO_CHI2_LOW and qnll < QUAD_NLL_SCRAMBLED:
        atoms.append(_atom(
            "letters_unsubstituted", "order_layout", 0.55,
            ["transposition"],
            ["monoalphabetic_substitution", "homophonic_substitution"],
            measurement={"monogram_chi2": mchi, "quadgram_nll": qnll},
            interpretation="Language monogram frequencies but scrambled quadgrams "
                           "(letters reordered, not substituted).",
        ))
    if mchi > MONO_CHI2_HIGH and peaked:
        # When a residual order layer is detected, the composite explanation
        # (substitution THEN transposition) is live, so letters_substituted must
        # NOT push transposition down: drop its transposition-weaken.
        include_transposition_weaken = not residual
        atoms.append(_atom(
            "letters_substituted", "order_layout", 0.35,
            ["monoalphabetic_substitution", "homophonic_substitution"],
            ["transposition"] if include_transposition_weaken else [],
            measurement={"monogram_chi2": mchi, "quadgram_nll": qnll},
            interpretation="Monogram frequencies displaced from the language "
                           "reference (letters substituted).",
        ))
    if residual:
        # Supports the composite family, and WEAKENS plain monoalphabetic
        # substitution (a detected residual order layer means the cipher is not a
        # PLAIN mono). It deliberately does NOT weaken `transposition` (a composite
        # IS partly a transposition). The mono-weaken is what lets the diagnosis
        # escape the "confident plain-mono" anchoring the composite program targets;
        # it only fires on true composites (gated on displaced monograms + absent
        # structure + >=150 letters), so a plain mono is never demoted.
        atoms.append(_atom(
            "residual_order_after_substitution", "order_layout", 0.35,
            ["substitution_transposition"], ["monoalphabetic_substitution"],
            measurement={"monogram_chi2": mchi, "quadgram_nll": qnll,
                         "structure_ratio": structure_ratio},
            interpretation="Letters substituted (displaced monograms) but n-gram "
                           "structure absent (adjacency destroyed) — consistent with "
                           "a further order/transposition layer over a substitution.",
        ))
    return PanelResult(
        "order_layout", "ok",
        measurements={"monogram_chi2": mchi, "quadgram_nll": qnll, "peaked": peaked,
                      "structure_ratio": structure_ratio,
                      "residual_order_after_substitution": residual},
        atoms=atoms,
        reliability="high",
    )


def panel_nulls_noise(tokens, *, alphabet_size, alphabet_class, language,
                      word_group_count=0, line_lengths=None, numeric_values=None,
                      letter_rendering=None, related_profile=None, max_period=26,
                      rng_namespace="") -> PanelResult:
    n = len(tokens)
    if n < 10:
        return PanelResult("nulls_noise", "not_computable", reason="too_few_tokens")
    counts = Counter(tokens)
    top = [s for s, _ in counts.most_common(5)]
    top_set = set(top)
    # IC trajectory with the top-k frequent symbols masked out (minimal in INV-0).
    masked = [t for t in tokens if t not in top_set]
    ic_all = index_of_coincidence(list(tokens), max(alphabet_size, len(counts)))
    ic_masked = index_of_coincidence(masked, max(alphabet_size, len(set(masked)))) if len(masked) > 1 else 0.0
    return PanelResult(
        "nulls_noise", "ok",
        measurements={
            "top_symbols": top,
            "ic_all": ic_all,
            "ic_top5_masked": ic_masked,
            "ic_shift_when_masked": ic_masked - ic_all,
        },
        reliability="low",
    )


def panel_polygraphic(tokens, *, alphabet_size, alphabet_class, language,
                      word_group_count=0, line_lengths=None, numeric_values=None,
                      letter_rendering=None, related_profile=None, max_period=26,
                      rng_namespace="") -> PanelResult:
    n = len(tokens)
    if n < 4:
        return PanelResult("polygraphic", "not_computable", reason="too_few_tokens")
    toklist = list(tokens)
    doubled = sum(1 for i in range(n - 1) if toklist[i] == toklist[i + 1]) / (n - 1)
    mono_ic = index_of_coincidence(toklist, max(alphabet_size, len(set(toklist))))
    digraphs = [toklist[i] * 100003 + toklist[i + 1] for i in range(0, n - 1, 2)]
    digraph_ic = index_of_coincidence(digraphs, max(2, len(set(digraphs)))) if len(digraphs) > 1 else 0.0
    ratio = digraph_ic / mono_ic if mono_ic > 0 else 0.0
    unique = len(set(toklist))
    coord_check = unique in {25, 36}
    return PanelResult(
        "polygraphic", "ok",
        measurements={
            "doubled_digraph_rate": doubled,
            "monogram_ic": mono_ic,
            "digraph_ic": digraph_ic,
            "digraph_over_monogram_ic": ratio,
            "coordinate_alphabet_size": coord_check,
        },
        reliability="low",
    )


def panel_numeric_code(tokens, *, alphabet_size, alphabet_class, language,
                       word_group_count=0, line_lengths=None, numeric_values=None,
                       letter_rendering=None, related_profile=None, max_period=26,
                       rng_namespace="") -> PanelResult:
    if alphabet_class != "numeric":
        return PanelResult(
            "numeric_code", "not_computable", reason="requires numeric alphabet_class",
        )
    values = list(numeric_values) if numeric_values is not None else list(tokens)
    if not values:
        return PanelResult("numeric_code", "not_computable", reason="no_values")

    battery = numeric_code_battery(
        values, related_profile=related_profile, rng_namespace=rng_namespace,
        include_modulo=False, include_repeat_baselines=False,
    )
    m = battery["measurements"]
    from investigation.families import SUBSTITUTION_PRIMARIES  # local import: avoid cycle

    subst = list(SUBSTITUTION_PRIMARIES)
    atoms: list[dict[str, Any]] = []

    atoms.append(_atom(
        "numeric_token_stream", "numeric_code", 0.35,
        ["numeric_book_cipher", "nomenclator_codebook"], subst,
        measurement={"count": m["count"], "unique": m["unique"]},
        interpretation="Stream is decimal integer indices, not a substitution alphabet.",
    ))
    if m["unique_rate"] > NUMERIC_UNIQUE_RATE:
        atoms.append(_atom(
            "numeric_inconsistent_with_substitution", "numeric_code", 0.50,
            [], subst,
            measurement={"unique_rate": m["unique_rate"]},
            interpretation=(
                "unique/token %.3f > %.2f with a numeric alphabet — inconsistent "
                "with any substitution cipher." % (m["unique_rate"], NUMERIC_UNIQUE_RATE)
            ),
        ))
    if m["book_keylength_plausible"]:
        atoms.append(_atom(
            "book_keylength_plausible", "numeric_code", 0.30,
            ["numeric_book_cipher"], [],
            measurement={"min": m["min"], "max": m["max"]},
            interpretation="Value range is a feasible book key-text length.",
        ))
    if m["front_loading_significant"]:
        atoms.append(_atom(
            "front_loading_present", "numeric_code", 0.20,
            ["numeric_book_cipher"], ["plaintext_or_hoax"],
            reliability="high",
            measurement={"front_loading_index": m["front_loading_index"]},
            baseline=battery["front_loading"]["baseline"],
            interpretation="Front-loaded value distribution (book-order skew); "
                           "uniform-draw null rejects at p<=0.05.",
        ))
    if m["independent_random_like"]:
        atoms.append(_atom(
            "independent_random_like", "numeric_code", 0.36,
            ["plaintext_or_hoax"],
            ["numeric_book_cipher", "nomenclator_codebook"],
            measurement={
                "last_digit_uniform": m["last_digit_uniform"],
                "epsilon_benford_deviation": m["epsilon_benford_deviation"],
                "repeat_within_1sd": m["repeat_within_1sd"],
            },
            interpretation="Digit laws + repeat rate consistent with an i.i.d. "
                           "random-like stream (no book structure).",
        ))
    if m["structured_hoax_artifact"]:
        atoms.append(_atom(
            "structured_hoax_artifact", "numeric_code", 0.35,
            ["plaintext_or_hoax"], ["numeric_book_cipher"],
            reliability="high",
            measurement={
                "monotone_run_significant": m["monotone_run_significant"],
                "last_digit_uniform": m["last_digit_uniform"],
                "consecutive_run_significant": m["consecutive_run_significant"],
            },
            baseline=battery["runs"]["longest_monotone_run_baseline"],
            interpretation="Fabrication artifacts (monotone-run/digit-preference "
                           "excess) — Gillogly-style hoax signature.",
        ))
    if m["word_first_letter_book_cipher"]:
        atoms.append(_atom(
            "book_word_position_signature", "numeric_code", 0.40,
            ["numeric_word_position"], [],
            measurement={"front_loading_index": m["front_loading_index"]},
            interpretation="P8 word-position book-cipher hypothesis is supported.",
        ))
    if m["char_position_book_cipher"]:
        atoms.append(_atom(
            "book_char_position_signature", "numeric_code", 0.40,
            ["numeric_char_position"], [],
            interpretation="P8 char-position book-cipher hypothesis is supported.",
        ))
    if m["skip_nth_word"]:
        atoms.append(_atom(
            "book_skip_nth_signature", "numeric_code", 0.40,
            ["numeric_skip_nth_word"], [],
            interpretation="P8 skip-Nth-word book-cipher hypothesis is supported.",
        ))
    return PanelResult(
        "numeric_code", "ok",
        measurements=m,
        atoms=atoms,
        reliability="high",
    )


def panel_language(tokens, *, alphabet_size, alphabet_class, language,
                   word_group_count=0, line_lengths=None, numeric_values=None,
                   letter_rendering=None, related_profile=None, max_period=26,
                   rng_namespace="") -> PanelResult:
    if alphabet_class not in ("letters", "mixed"):
        return PanelResult(
            "language", "not_computable",
            reason="requires letters or mixed alphabet_class",
        )
    rendering = None
    if letter_rendering:
        rendering = "".join(ch for ch in letter_rendering.upper() if "A" <= ch <= "Z")
    atoms: list[dict[str, Any]] = []
    measurements: dict[str, Any] = {"assumed_language": language}
    if rendering:
        try:
            from analysis.language_guesser import best_guess
            ic = index_of_coincidence(list(tokens), max(alphabet_size, len(set(tokens))))
            measurements["ic_language_guess"] = best_guess(ic).language
        except Exception:  # noqa: BLE001
            pass
        atoms.append(_atom(
            "language_provisional", "language", 0.0, [], [],
            reliability="low",
            measurement=measurements,
            interpretation=f"Provisional language assumption: {language}.",
        ))
    return PanelResult(
        "language", "ok",
        measurements=measurements,
        atoms=atoms,
        reliability="low",
    )


_PANELS: dict[str, Callable[..., PanelResult]] = {
    "shape": panel_shape,
    "frequency": panel_frequency,
    "repetition": panel_repetition,
    "periodicity": panel_periodicity,
    "order_layout": panel_order_layout,
    "nulls_noise": panel_nulls_noise,
    "polygraphic": panel_polygraphic,
    "numeric_code": panel_numeric_code,
    "language": panel_language,
}


def run_battery(
    tokens,
    *,
    alphabet_size,
    alphabet_class,
    language,
    word_group_count=0,
    line_lengths=None,
    numeric_values=None,
    letter_rendering=None,
    related_profile=None,
    max_period=26,
    rng_namespace="",
) -> dict[str, PanelResult]:
    """Run all nine panels with per-panel exception isolation (Part 2)."""
    results: dict[str, PanelResult] = {}
    for name, fn in _PANELS.items():
        try:
            results[name] = fn(
                tokens, alphabet_size=alphabet_size, alphabet_class=alphabet_class,
                language=language, word_group_count=word_group_count,
                line_lengths=line_lengths, numeric_values=numeric_values,
                letter_rendering=letter_rendering, related_profile=related_profile,
                max_period=max_period, rng_namespace=rng_namespace,
            )
        except Exception as exc:  # noqa: BLE001 - a panel crash must not abort the battery
            results[name] = PanelResult(
                name, "not_computable", reason=f"panel_error:{type(exc).__name__}",
            )
    return results
