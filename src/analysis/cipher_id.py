"""Cipher-type fingerprint: cheap statistical signals for hypothesis generation.

Computes a structured fingerprint from raw cipher tokens that gives the LLM
agent a first-pass ranked hypothesis list before any solve tool is called.
No solver state, language models, or expensive analysis required — only token
ID arrays and a handful of classical statistical tests.

Signal summary
--------------
IC (index of coincidence)
    Near language reference  → monoalphabetic substitution
    Depressed               → polyalphabetic, homophonic, or random
Frequency flatness (chi² vs uniform / normalized entropy)
    Peaked                  → monoalphabetic
    Flat                    → homophonic or polyalphabetic
Periodic IC (mean IC of every-kth-token streams)
    Peak at period k, recovering to language IC → Vigenère key length k
Kasiski spacing GCDs
    Most common GCD of repeated-trigram spacings → corroborate period
Doubled-digraph rate
    Near zero               → Playfair (prohibits same-letter digraphs)
Alphabet/symbol counts
    unique_symbols >> 26    → homophonic substitution
Token count parity
    Even token count + near-zero doubles → Playfair further support
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from analysis.ic import index_of_coincidence
from analysis.language_guesser import LANGUAGE_IC_REFERENCES

_RANDOM_IC_26 = 1.0 / 26.0  # ~0.0385
_MIN_TOKENS_FOR_PERIODIC = 40   # minimum tokens to attempt periodic IC
_MIN_TOKENS_FOR_KASISKI = 30    # minimum tokens for Kasiski
_STANDARD_ALPHA_SIZE = 26       # standard monoalphabetic plaintext alphabet


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CipherFingerprint:
    """Structured cipher-type fingerprint derived from raw token analysis."""

    # ---- Basic statistics ----
    token_count: int
    unique_symbols: int       # number of distinct token IDs actually observed
    alphabet_size: int        # reported alphabet size (may differ from unique_symbols)
    word_group_count: int     # number of |-separated word groups (0 = no boundaries)
    even_length: bool

    # ---- IC-based signals ----
    ic: float                             # raw Index of Coincidence
    ic_interpretation: str                # prose label
    language_ic_reference: float | None   # reference IC for the target language
    ic_delta_from_reference: float | None # ic − language_ref (negative = depressed)

    # ---- Frequency distribution ----
    frequency_flatness_chi2: float  # chi² vs uniform; high = peaked (monoalphabetic)
    normalized_entropy: float       # H(X) / log2(unique_symbols); 1.0 = perfectly flat

    # ---- Periodic signals (Friedman-style) ----
    periodic_ic: dict[int, float]   # k → mean IC of every-kth-token streams
    best_period: int | None         # k ≥ 2 with highest periodic_ic
    best_period_ic: float | None

    # ---- Kasiski repeat analysis ----
    kasiski_spacing_gcds: dict[int, int]  # candidate_period → count of supporting spacings
    kasiski_best_period: int | None       # period with most Kasiski support

    # ---- Playfair-specific signals ----
    doubled_digraph_rate: float  # fraction of consecutive identical token pairs

    # ---- Suspicion scores (0.0–1.0, independent evidence weights) ----
    suspicion_scores: dict[str, float]

    # ---- Human-readable summary ----
    natural_language_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "unique_symbols": self.unique_symbols,
            "alphabet_size": self.alphabet_size,
            "word_group_count": self.word_group_count,
            "even_length": self.even_length,
            "ic": round(self.ic, 6) if not math.isnan(self.ic) else None,
            "ic_interpretation": self.ic_interpretation,
            "language_ic_reference": self.language_ic_reference,
            "ic_delta_from_reference": (
                round(self.ic_delta_from_reference, 6)
                if self.ic_delta_from_reference is not None else None
            ),
            "frequency_flatness_chi2": round(self.frequency_flatness_chi2, 6),
            "normalized_entropy": round(self.normalized_entropy, 4),
            "periodic_ic": {str(k): round(v, 6) for k, v in self.periodic_ic.items()},
            "best_period": self.best_period,
            "best_period_ic": (
                round(self.best_period_ic, 6)
                if self.best_period_ic is not None else None
            ),
            "kasiski_spacing_gcds": {str(k): v for k, v in self.kasiski_spacing_gcds.items()},
            "kasiski_best_period": self.kasiski_best_period,
            "doubled_digraph_rate": round(self.doubled_digraph_rate, 4),
            "suspicion_scores": {k: round(v, 4) for k, v in self.suspicion_scores.items()},
            "natural_language_summary": self.natural_language_summary,
        }


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def compute_cipher_fingerprint(
    tokens: list[int],
    alphabet_size: int,
    *,
    max_period: int = 26,
    language: str = "en",
    word_group_count: int = 0,
) -> CipherFingerprint:
    """Compute a comprehensive cipher-type fingerprint from raw token data.

    Parameters
    ----------
    tokens:
        Raw cipher token IDs (integers).
    alphabet_size:
        Size of the cipher symbol alphabet as declared by the transcription.
        May differ from len(set(tokens)) for sparse or multi-page ciphers.
    max_period:
        Maximum key period to test in Friedman and Kasiski analyses.
    language:
        Target language code (e.g. 'en', 'la', 'de'). Used to look up the
        reference IC for comparison.
    word_group_count:
        Number of |-separated word groups already parsed from the ciphertext.
        Zero means the cipher has no word-boundary structure.
    """
    n = len(tokens)
    unique_symbols = len(set(tokens))
    even_length = (n % 2 == 0)

    # Reference IC for target language
    lang_entry = LANGUAGE_IC_REFERENCES.get(language, LANGUAGE_IC_REFERENCES["en"])
    lang_ic_ref: float = lang_entry[1]

    # --- IC ---
    if n >= 10:
        ic = index_of_coincidence(tokens, max(alphabet_size, unique_symbols))
        ic_delta: float | None = ic - lang_ic_ref
    else:
        ic = float("nan")
        ic_delta = None
    ic_interpretation = _interpret_ic(ic, lang_ic_ref)

    # --- Frequency distribution ---
    counts = Counter(tokens)
    flatness_chi2 = _chi2_vs_uniform(counts, n)
    norm_entropy = _normalized_entropy(counts, n)

    # --- Periodic IC (Friedman) ---
    periodic_ic_dict: dict[int, float] = {}
    if n >= _MIN_TOKENS_FOR_PERIODIC:
        for k in range(2, min(max_period + 1, n // 4 + 2)):
            streams = [tokens[i::k] for i in range(k)]
            valid_ics = [
                index_of_coincidence(s, max(alphabet_size, unique_symbols))
                for s in streams
                if len(s) >= 6
            ]
            if valid_ics:
                periodic_ic_dict[k] = sum(valid_ics) / len(valid_ics)

    best_period: int | None = None
    best_period_ic: float | None = None
    if periodic_ic_dict:
        best_period = max(periodic_ic_dict, key=lambda k: periodic_ic_dict[k])
        best_period_ic = periodic_ic_dict[best_period]

    # --- Kasiski ---
    kasiski_gcds: dict[int, int] = {}
    kasiski_best: int | None = None
    if n >= _MIN_TOKENS_FOR_KASISKI:
        kasiski_gcds, kasiski_best = _kasiski_analysis(tokens, max_period=max_period)

    # --- Doubled-digraph rate ---
    if n >= 2:
        doubled = sum(1 for i in range(n - 1) if tokens[i] == tokens[i + 1])
        doubled_digraph_rate = doubled / (n - 1)
    else:
        doubled_digraph_rate = 0.0

    # --- Suspicion scores ---
    suspicion_scores = _compute_suspicion_scores(
        ic=ic,
        ic_delta=ic_delta,
        lang_ic_ref=lang_ic_ref,
        flatness_chi2=flatness_chi2,
        norm_entropy=norm_entropy,
        alphabet_size=alphabet_size,
        unique_symbols=unique_symbols,
        word_group_count=word_group_count,
        token_count=n,
        best_period=best_period,
        best_period_ic=best_period_ic,
        kasiski_best=kasiski_best,
        doubled_digraph_rate=doubled_digraph_rate,
        even_length=even_length,
    )

    summary = _format_natural_language_summary(
        token_count=n,
        unique_symbols=unique_symbols,
        alphabet_size=alphabet_size,
        ic=ic,
        ic_delta=ic_delta,
        lang_ic_ref=lang_ic_ref,
        ic_interpretation=ic_interpretation,
        norm_entropy=norm_entropy,
        best_period=best_period,
        best_period_ic=best_period_ic,
        kasiski_best=kasiski_best,
        kasiski_gcds=kasiski_gcds,
        doubled_digraph_rate=doubled_digraph_rate,
        suspicion_scores=suspicion_scores,
        word_group_count=word_group_count,
    )

    return CipherFingerprint(
        token_count=n,
        unique_symbols=unique_symbols,
        alphabet_size=alphabet_size,
        word_group_count=word_group_count,
        even_length=even_length,
        ic=ic,
        ic_interpretation=ic_interpretation,
        language_ic_reference=lang_ic_ref,
        ic_delta_from_reference=ic_delta,
        frequency_flatness_chi2=flatness_chi2,
        normalized_entropy=norm_entropy,
        periodic_ic=periodic_ic_dict,
        best_period=best_period,
        best_period_ic=best_period_ic,
        kasiski_spacing_gcds=kasiski_gcds,
        kasiski_best_period=kasiski_best,
        doubled_digraph_rate=doubled_digraph_rate,
        suspicion_scores=suspicion_scores,
        natural_language_summary=summary,
    )


def format_fingerprint_for_context(fp: CipherFingerprint) -> str:
    """Return a concise human-readable fingerprint block for initial_context.

    This is the text the agent sees before its first tool call. It should be
    informative but compact — the agent will call ``observe_cipher_fingerprint``
    for deeper analysis mid-run.
    """
    lines = ["## Cipher-type fingerprint", ""]
    lines.append(fp.natural_language_summary)
    lines.append("")

    # Ranked suspicions (only those above a minimum threshold)
    ranked = sorted(fp.suspicion_scores.items(), key=lambda x: x[1], reverse=True)
    notable = [(k, v) for k, v in ranked if v >= 0.05]
    if notable:
        lines.append("Top suspicions (ranked):")
        for i, (ctype, score) in enumerate(notable[:6], 1):
            bar = "█" * int(score * 10)
            lines.append(f"  {i}. {ctype:<35} {score:.2f}  {bar}")
    lines.append("")

    # IC line
    if not math.isnan(fp.ic):
        ref_str = ""
        if fp.language_ic_reference is not None:
            delta = fp.ic - fp.language_ic_reference
            ref_str = f" (ref {fp.language_ic_reference:.4f}, Δ{delta:+.4f})"
        lines.append(f"IC = {fp.ic:.4f}{ref_str}  →  {fp.ic_interpretation}")

    # Entropy
    lines.append(
        f"Normalized entropy: {fp.normalized_entropy:.3f} "
        f"(1.0 = flat)  |  Unique symbols: {fp.unique_symbols} / {fp.alphabet_size}"
    )

    # Periodic signals
    if fp.best_period is not None and fp.best_period_ic is not None:
        corr = ""
        if fp.kasiski_best_period is not None:
            corr = f"; Kasiski best: {fp.kasiski_best_period}"
        lines.append(
            f"Periodic IC peak: period={fp.best_period}  "
            f"(mean IC={fp.best_period_ic:.4f}){corr}"
        )
        top_kasiski = sorted(fp.kasiski_spacing_gcds.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_kasiski:
            gcd_str = ", ".join(f"{k}:{v}" for k, v in top_kasiski)
            lines.append(f"  Kasiski GCD counts (period:count): {gcd_str}")

    # Playfair signals
    lines.append(
        f"Doubled-digraph rate: {fp.doubled_digraph_rate:.3f}  "
        f"|  Even token count: {fp.even_length}"
    )

    return "\n".join(lines)


def kasiski_report(
    tokens: list[int],
    *,
    min_n: int = 3,
    max_n: int = 5,
    max_period: int = 40,
    top_n: int = 20,
    max_pairs_per_ngram: int = 20,
) -> dict[str, Any]:
    """Return repeated n-gram spacing evidence for periodic keys."""
    if len(tokens) < min_n:
        return {
            "status": "unsupported",
            "reason": "too_few_tokens",
            "token_count": len(tokens),
            "repeated_sequences": [],
            "factor_counts": {},
            "best_period": None,
        }

    factor_counts: Counter[int] = Counter()
    repeated: list[dict[str, Any]] = []
    for ngram_len in range(min_n, max_n + 1):
        positions: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i in range(len(tokens) - ngram_len + 1):
            positions[tuple(tokens[i : i + ngram_len])].append(i)

        for gram, pos_list in positions.items():
            if len(pos_list) < 2:
                continue
            spacings: list[int] = []
            local_factors: Counter[int] = Counter()
            for j in range(len(pos_list)):
                for k in range(j + 1, min(j + max_pairs_per_ngram, len(pos_list))):
                    spacing = pos_list[k] - pos_list[j]
                    if spacing < 2:
                        continue
                    spacings.append(spacing)
                    for factor in range(2, min(spacing, max_period) + 1):
                        if spacing % factor == 0:
                            factor_counts[factor] += 1
                            local_factors[factor] += 1
            if spacings:
                repeated.append({
                    "ngram": list(gram),
                    "ngram_length": ngram_len,
                    "positions": pos_list[:20],
                    "occurrences": len(pos_list),
                    "spacings": spacings[:30],
                    "top_factors": [
                        {"period": period, "count": count}
                        for period, count in local_factors.most_common(8)
                    ],
                })

    repeated.sort(
        key=lambda item: (
            item["occurrences"],
            len(item["spacings"]),
            item["ngram_length"],
        ),
        reverse=True,
    )
    best_period = factor_counts.most_common(1)[0][0] if factor_counts else None
    return {
        "status": "completed",
        "token_count": len(tokens),
        "min_ngram_length": min_n,
        "max_ngram_length": max_n,
        "max_period": max_period,
        "repeated_sequences": repeated[: max(1, top_n)],
        "factor_counts": {
            str(period): count for period, count in factor_counts.most_common(max_period)
        },
        "best_period": best_period,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interpret_ic(ic: float, lang_ic_ref: float) -> str:
    if math.isnan(ic):
        return "insufficient data (text too short)"
    random_ref = _RANDOM_IC_26
    if ic >= lang_ic_ref - 0.008:
        return "consistent with monoalphabetic substitution (IC near language reference)"
    if ic >= lang_ic_ref - 0.020:
        return "mildly depressed IC — possibly homophonic, short polyalphabetic, or noisy sample"
    if ic >= random_ref + 0.008:
        return "substantially depressed IC — suggests polyalphabetic (Vigenère) or homophonic cipher"
    return "near-random IC — likely polyalphabetic with short key, or strongly homophonic"


def _chi2_vs_uniform(counts: Counter, n: int) -> float:
    """Chi-squared of observed symbol frequencies vs. a uniform distribution."""
    if n <= 0 or not counts:
        return 0.0
    k = len(counts)
    if k <= 1:
        return 0.0
    expected = n / k
    return sum((c - expected) ** 2 / expected for c in counts.values())


def _normalized_entropy(counts: Counter, n: int) -> float:
    """Shannon entropy normalised to [0, 1] by log2(unique_symbols)."""
    if n <= 0 or not counts:
        return 0.0
    k = len(counts)
    if k <= 1:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / n
            entropy -= p * math.log2(p)
    max_entropy = math.log2(k)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _kasiski_analysis(
    tokens: list[int],
    *,
    min_n: int = 3,
    max_period: int = 26,
    max_pairs_per_ngram: int = 10,
) -> tuple[dict[int, int], int | None]:
    """Find candidate key periods via repeated-trigram spacing analysis.

    Returns
    -------
    (factor_counts, best_period)
        factor_counts: dict mapping each candidate period (2..max_period) to the
        number of repeated-trigram spacings that are divisible by it.
        best_period: the factor with the highest count, or None.
    """
    # Map each n-gram to the list of positions where it occurs
    positions: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i in range(len(tokens) - min_n + 1):
        key = tuple(tokens[i : i + min_n])
        positions[key].append(i)

    # Collect pairwise spacings from repeated n-grams
    spacings: list[int] = []
    for pos_list in positions.values():
        if len(pos_list) < 2:
            continue
        for j in range(len(pos_list)):
            for k in range(j + 1, min(j + max_pairs_per_ngram, len(pos_list))):
                spacing = pos_list[k] - pos_list[j]
                if spacing >= 2:
                    spacings.append(spacing)

    if not spacings:
        return {}, None

    # For each spacing, increment counts for all of its factors in [2, max_period]
    factor_counts: Counter[int] = Counter()
    for spacing in spacings:
        for f in range(2, min(spacing + 1, max_period + 1)):
            if spacing % f == 0:
                factor_counts[f] += 1

    if not factor_counts:
        return {}, None

    best_period = factor_counts.most_common(1)[0][0]
    return dict(factor_counts), best_period


def _compute_suspicion_scores(
    *,
    ic: float,
    ic_delta: float | None,
    lang_ic_ref: float,
    flatness_chi2: float,
    norm_entropy: float,
    alphabet_size: int,
    unique_symbols: int,
    word_group_count: int,
    token_count: int,
    best_period: int | None,
    best_period_ic: float | None,
    kasiski_best: int | None,
    doubled_digraph_rate: float,
    even_length: bool,
) -> dict[str, float]:
    """Compute independent evidence-weight scores for each cipher family.

    Scores are NOT probabilities and do NOT sum to 1. Each is a 0–1 evidence
    weight for that family. The agent should treat them as a ranked suspicion
    list, not a probability distribution.
    """
    if math.isnan(ic):
        return {"unknown": 1.0}

    scores: dict[str, float] = {}

    # ---- monoalphabetic_substitution ----
    mono = 0.0
    if ic_delta is not None and abs(ic_delta) < 0.012:
        mono += 0.50
    elif ic_delta is not None and abs(ic_delta) < 0.020:
        mono += 0.30
    # Peaked frequency distribution is expected for monoalphabetic
    if flatness_chi2 > 5.0 and unique_symbols <= 28:
        mono += 0.25
    elif flatness_chi2 > 1.0:
        mono += 0.10
    # Small alphabet expected
    if unique_symbols <= _STANDARD_ALPHA_SIZE + 2:
        mono += 0.15
    if token_count >= 50:
        mono += 0.10
    scores["monoalphabetic_substitution"] = min(mono, 1.0)

    # ---- homophonic_substitution ----
    homo = 0.0
    if unique_symbols > _STANDARD_ALPHA_SIZE:
        # Strong signal: alphabet is larger than 26
        extra = min((unique_symbols - _STANDARD_ALPHA_SIZE) / 20.0, 1.0)
        homo += 0.45 * extra + 0.20
    # Flat distribution (low chi², high entropy) supports homophonic
    if norm_entropy > 0.90:
        homo += 0.25
    elif norm_entropy > 0.80:
        homo += 0.15
    # No-boundary ciphers are more often homophonic
    if word_group_count == 0 and token_count >= 80:
        homo += 0.10
    scores["homophonic_substitution"] = min(homo, 1.0)

    # ---- polyalphabetic_vigenere ----
    vig = 0.0
    # Core signal: IC depressed below language reference.
    # Use a normalized IC fraction so large alphabets (e.g. 33-symbol Borg) are
    # not falsely penalised. random_ic = 1/alphabet_size; lang_ic_ref is the
    # target-language reference for 26 letters, which also applies to
    # monoalphabetic ciphers over larger symbol sets (the substitution
    # preserves the collision probability). norm_ic_frac ≈ 1.0 for
    # monoalphabetic, ≈ 0.0 for random/polyalphabetic.
    random_ic = 1.0 / max(alphabet_size, unique_symbols, 1)
    ic_range = lang_ic_ref - random_ic
    if ic_range > 0 and not math.isnan(ic):
        norm_ic_frac = (ic - random_ic) / ic_range
    else:
        norm_ic_frac = 0.5  # unknown; treat as neutral
    ic_poly_signal = 1.0 - max(0.0, min(1.0, norm_ic_frac))
    if ic_poly_signal > 0.80:
        vig += 0.40  # IC close to random → strong Vigenère signal
    elif ic_poly_signal > 0.55:
        vig += 0.20
    # Periodic IC recovering to language reference at some period k.
    # Only meaningful when columns are long enough for a reliable IC estimate;
    # require at least 25 tokens per column to avoid spurious peaks on short texts.
    _MIN_COLS_FOR_PERIODIC = 25
    periodic_ic_reliable = (
        best_period is not None
        and best_period_ic is not None
        and token_count // best_period >= _MIN_COLS_FOR_PERIODIC
    )
    if periodic_ic_reliable:
        recovery = best_period_ic - ic  # how much IC rises at best period
        if recovery > 0.010 and best_period_ic >= lang_ic_ref - 0.015:
            vig += 0.35  # strong recovery
        elif recovery > 0.005:
            vig += 0.15
    # Kasiski corroboration
    if kasiski_best is not None and best_period is not None and kasiski_best == best_period:
        vig += 0.20
    elif kasiski_best is not None:
        vig += 0.05
    # Standard 26-symbol alphabet expected
    if unique_symbols <= 26:
        vig += 0.05
    scores["polyalphabetic_vigenere"] = min(vig, 1.0)

    # ---- transposition ----
    # Transposition preserves letter frequencies (and hence IC) but disrupts
    # sequential patterns. This is hard to distinguish from monoalphabetic
    # using cheap signals alone, so cap the score conservatively.
    trans = 0.0
    if ic_delta is not None and abs(ic_delta) < 0.015:
        trans += 0.25  # IC preserved = possible transposition OR monoalphabetic
    if unique_symbols <= 28 and word_group_count == 0:
        trans += 0.10
    scores["transposition"] = min(trans, 0.40)  # capped — requires deeper analysis

    # ---- transposition_homophonic ----
    th = 0.0
    if unique_symbols > _STANDARD_ALPHA_SIZE and word_group_count == 0:
        th += 0.45
    if ic_delta is not None and ic_delta < -0.005 and unique_symbols > _STANDARD_ALPHA_SIZE:
        th += 0.20
    scores["transposition_homophonic"] = min(th, 1.0)

    # ---- playfair ----
    # Key signals: no same-letter digraphs (prohibits doubles), even token
    # count, 25-symbol alphabet (I/J merged), no word boundaries
    play = 0.0
    if token_count >= 20 and doubled_digraph_rate < 0.005:
        play += 0.45   # near-zero: very strong Playfair signal
    elif token_count >= 20 and doubled_digraph_rate < 0.030:
        play += 0.30   # halved-from-random: Playfair prohibits within-pair doubles
    elif token_count >= 20 and doubled_digraph_rate < 0.050:
        play += 0.15   # slightly below random: weak signal
    if even_length:
        play += 0.20
    if unique_symbols in {25, 26}:
        play += 0.15
    if word_group_count == 0:
        play += 0.10
    if token_count >= 40:
        play += 0.05
    # Playfair uses bigraph substitution: norm_entropy should be moderate
    if 0.70 < norm_entropy < 0.95:
        play += 0.05
    scores["playfair"] = min(play, 1.0)

    return scores


def _format_natural_language_summary(
    *,
    token_count: int,
    unique_symbols: int,
    alphabet_size: int,
    ic: float,
    ic_delta: float | None,
    lang_ic_ref: float,
    ic_interpretation: str,
    norm_entropy: float,
    best_period: int | None,
    best_period_ic: float | None,
    kasiski_best: int | None,
    kasiski_gcds: dict[int, int],
    doubled_digraph_rate: float,
    suspicion_scores: dict[str, float],
    word_group_count: int,
) -> str:
    parts: list[str] = []

    if math.isnan(ic):
        return (
            f"The cipher has {token_count} tokens, which is too short for reliable "
            "statistical analysis. Manual inspection is recommended."
        )

    # Token/symbol counts
    boundary_str = (
        f" with {word_group_count} word groups"
        if word_group_count > 0
        else " with no word-boundary structure"
    )
    sym_str = (
        f"{unique_symbols} unique symbols"
        f" (alphabet size {alphabet_size})"
    )
    parts.append(
        f"The cipher has {token_count} tokens{boundary_str}, using {sym_str}."
    )

    # IC interpretation
    ic_str = f"The IC ({ic:.4f}) is {ic_interpretation}."
    parts.append(ic_str)

    # Frequency distribution
    if norm_entropy > 0.90:
        parts.append("The frequency distribution is near-flat (high entropy), suggesting homophonic or polyalphabetic.")
    elif norm_entropy > 0.75:
        parts.append("The frequency distribution is moderately flat.")
    else:
        parts.append("The frequency distribution is peaked, consistent with monoalphabetic substitution.")

    # Periodic / Kasiski
    _MIN_COLS_FOR_PERIODIC_SUMMARY = 25
    if best_period is not None and best_period_ic is not None:
        col_size = token_count // best_period
        recovery = best_period_ic - ic
        if col_size < _MIN_COLS_FOR_PERIODIC_SUMMARY:
            parts.append(
                f"Periodic IC found a peak at period {best_period} "
                f"(mean IC {best_period_ic:.4f}), but only ~{col_size} tokens/column — "
                "unreliable; treat this as noise."
            )
        elif recovery > 0.010 and best_period_ic >= lang_ic_ref - 0.015:
            parts.append(
                f"Periodic IC analysis strongly suggests period {best_period}: "
                f"the mean IC of every-{best_period}th-token streams is {best_period_ic:.4f}, "
                f"recovering close to the language reference ({lang_ic_ref:.4f})."
            )
        else:
            parts.append(
                f"Periodic IC analysis found a tentative peak at period {best_period} "
                f"(mean IC {best_period_ic:.4f}), but the recovery is weak."
            )
        if kasiski_best is not None:
            if kasiski_best == best_period:
                top_count = kasiski_gcds.get(kasiski_best, 0)
                parts.append(
                    f"Kasiski analysis corroborates period {kasiski_best} "
                    f"({top_count} repeated-trigram spacing(s) divisible by this value)."
                )
            else:
                parts.append(
                    f"Kasiski analysis suggests a different period ({kasiski_best}); "
                    "results are mixed — treat both as hypotheses."
                )
    else:
        parts.append("No significant periodic structure detected in Friedman or Kasiski analysis.")

    # Playfair / doubled digraph
    if doubled_digraph_rate < 0.005 and token_count >= 30:
        parts.append(
            f"The doubled-digraph rate is near zero ({doubled_digraph_rate:.3f}), "
            "which is a strong Playfair indicator (Playfair prohibits same-letter digraphs)."
        )
    elif doubled_digraph_rate < 0.015 and token_count >= 30:
        parts.append(
            f"The doubled-digraph rate is low ({doubled_digraph_rate:.3f}), "
            "weakly suggesting Playfair or a structured selection process."
        )

    # Top suspicion
    top = sorted(suspicion_scores.items(), key=lambda x: x[1], reverse=True)
    if top and top[0][1] >= 0.4:
        parts.append(f"Best hypothesis: {top[0][0]} (score {top[0][1]:.2f}).")
    elif top and top[0][1] >= 0.2:
        parts.append(
            f"Weak leading hypothesis: {top[0][0]} (score {top[0][1]:.2f}); "
            "multiple cipher types remain plausible."
        )
    else:
        parts.append("Signals are ambiguous or the text is too short for a confident hypothesis.")

    return " ".join(parts)
