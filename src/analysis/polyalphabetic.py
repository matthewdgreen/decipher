"""Periodic polyalphabetic cipher helpers.

This is intentionally a first slice, not a full classical-cipher zoo.  It
supports A-Z Vigenere-family screens well enough for unknown-cipher diagnosis
and agent branch creation, while keeping the key representation distinct from
substitution mappings.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from analysis import ngram
from models.cipher_text import CipherText


ENGLISH_FREQS = [
    0.0817, 0.0149, 0.0278, 0.0425, 0.1270, 0.0223, 0.0202, 0.0609, 0.0697,
    0.0015, 0.0077, 0.0403, 0.0241, 0.0675, 0.0751, 0.0193, 0.0010, 0.0599,
    0.0633, 0.0906, 0.0276, 0.0098, 0.0236, 0.0015, 0.0197, 0.0007,
]

SUPPORTED_VARIANTS = {"vigenere", "beaufort", "variant_beaufort", "gronsfeld"}
AZ_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class PeriodicCandidate:
    variant: str
    period: int
    shifts: list[int]
    plaintext: str
    score: float
    selection_score: float
    init_score: float
    key: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "period": self.period,
            "shifts": list(self.shifts),
            "key": self.key,
            "score": round(self.score, 5),
            "selection_score": round(self.selection_score, 5),
            "init_score": round(self.init_score, 5),
            "plaintext": self.plaintext,
            "preview": self.plaintext[:240],
            "metadata": dict(self.metadata),
        }


def search_periodic_polyalphabetic(
    cipher_text: CipherText,
    *,
    language: str = "en",
    periods: list[int] | None = None,
    max_period: int = 20,
    variants: list[str] | None = None,
    top_n: int = 5,
    refine: bool = True,
) -> dict[str, Any]:
    """Run a bounded Vigenere-family search over A-Z ciphertext.

    The returned candidates are plaintext strings plus periodic key state.  No
    substitution mapping is produced because that would be the wrong state
    model for these ciphers.
    """
    values, skipped = cipher_values_from_text(cipher_text)
    if len(values) < 12:
        return {
            "status": "unsupported",
            "reason": "too_few_a_z_tokens",
            "token_count": len(values),
            "skipped_symbols": skipped[:20],
            "candidates": [],
        }

    variants = [v.strip().lower() for v in (variants or ["vigenere", "beaufort", "variant_beaufort"])]
    bad_variants = [v for v in variants if v not in SUPPORTED_VARIANTS]
    if bad_variants:
        raise ValueError(f"unsupported polyalphabetic variants: {bad_variants}")
    if "gronsfeld" in variants and len(variants) > 1:
        # Gronsfeld is just constrained Vigenere in this first slice.  Keep it
        # explicit so result provenance is clear.
        pass

    period_values = sorted(set(int(p) for p in (periods or range(1, max_period + 1)) if int(p) >= 1))
    period_values = [p for p in period_values if p <= max(1, len(values) // 4)]
    if not period_values:
        period_values = [1]

    quad = ngram.NGRAM_CACHE.get(language, 4)
    candidates: list[PeriodicCandidate] = []
    for variant in variants:
        for period in period_values:
            shifts = _initial_shifts(values, period, variant)
            plaintext = decode_values(values, shifts, variant=variant)
            init_score = ngram.normalized_ngram_score(plaintext, quad, n=4)
            score = init_score
            if refine:
                shifts, plaintext, score = _refine_shifts(
                    values,
                    shifts,
                    variant=variant,
                    log_probs=quad,
                    gronsfeld_only=(variant == "gronsfeld"),
                )
            candidates.append(
                PeriodicCandidate(
                    variant=variant,
                    period=period,
                    shifts=shifts,
                    plaintext=plaintext,
                    score=score,
                    selection_score=score - _period_complexity_penalty(period),
                    init_score=init_score,
                    key=_key_string(shifts, gronsfeld=(variant == "gronsfeld")),
                    metadata={
                        "refined": refine,
                        "score_model": "wordlist_quadgram",
                    },
                )
            )

    candidates.sort(key=lambda c: c.selection_score, reverse=True)
    best = candidates[0] if candidates else None
    return {
        "status": "completed" if best else "no_candidates",
        "solver": "periodic_polyalphabetic_screen",
        "language": language,
        "token_count": len(values),
        "skipped_symbols": skipped[:20],
        "skipped_symbol_count": _skipped_symbol_count(cipher_text),
        "periods_tested": period_values,
        "variants_tested": variants,
        "top_candidates": [c.to_dict() for c in candidates[: max(1, top_n)]],
        "best_candidate": best.to_dict() if best else None,
        "normalization_note": (
            "Non-A-Z symbols were skipped before periodic analysis; skipped "
            "symbols do not advance the periodic key."
        ) if skipped else None,
    }


def cipher_values_from_text(cipher_text: CipherText) -> tuple[list[int], list[str]]:
    """Return A-Z values from a CipherText, plus unsupported symbols."""
    values: list[int] = []
    skipped: list[str] = []
    for token_id in cipher_text.tokens:
        symbol = cipher_text.alphabet.symbol_for(token_id).upper()
        if len(symbol) == 1 and "A" <= symbol <= "Z":
            values.append(ord(symbol) - ord("A"))
        elif symbol not in skipped:
            skipped.append(symbol)
    return values, skipped


def cipher_values_with_positions(cipher_text: CipherText) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    """Return A-Z values, original token positions, and skipped token records."""
    values: list[int] = []
    positions: list[int] = []
    skipped: list[dict[str, Any]] = []
    for pos, token_id in enumerate(cipher_text.tokens):
        symbol = cipher_text.alphabet.symbol_for(token_id).upper()
        if len(symbol) == 1 and "A" <= symbol <= "Z":
            values.append(ord(symbol) - ord("A"))
            positions.append(pos)
        else:
            skipped.append({"position": pos, "symbol": symbol})
    return values, positions, skipped


def keyed_alphabet_from_keyword(keyword: str, *, base_alphabet: str = AZ_ALPHABET) -> str:
    """Build a keyed alphabet by taking unique keyword letters, then remaining A-Z."""
    seen: set[str] = set()
    out: list[str] = []
    for ch in (keyword or "").upper() + base_alphabet:
        if "A" <= ch <= "Z" and ch not in seen:
            seen.add(ch)
            out.append(ch)
    if len(out) != 26:
        raise ValueError("keyed alphabet must contain exactly 26 distinct A-Z letters")
    return "".join(out)


def normalize_keyed_alphabet(
    *,
    keyed_alphabet: str | None = None,
    alphabet_keyword: str | None = None,
) -> str:
    """Normalize an explicit keyed alphabet or derive one from a keyword."""
    if keyed_alphabet:
        letters = [ch for ch in keyed_alphabet.upper() if "A" <= ch <= "Z"]
        if len(letters) != 26 or len(set(letters)) != 26:
            raise ValueError("keyed_alphabet must contain exactly 26 distinct A-Z letters")
        return "".join(letters)
    if alphabet_keyword:
        return keyed_alphabet_from_keyword(alphabet_keyword)
    return AZ_ALPHABET


def decode_keyed_vigenere_values(
    values: list[int],
    *,
    key: str,
    keyed_alphabet: str | None = None,
    alphabet_keyword: str | None = None,
) -> str:
    """Decode values with a shared keyed alphabet and periodic key.

    This matches Zenith's custom Vigenere-square convention used by the
    Kryptos fixtures: rows are rotations of one keyed alphabet, and key letters
    select rows by their position in that keyed alphabet.
    """
    alphabet = normalize_keyed_alphabet(
        keyed_alphabet=keyed_alphabet,
        alphabet_keyword=alphabet_keyword,
    )
    index = {ch: i for i, ch in enumerate(alphabet)}
    key_letters = [ch for ch in key.upper() if "A" <= ch <= "Z"]
    if not key_letters:
        raise ValueError("key must contain A-Z letters")
    missing = [ch for ch in key_letters if ch not in index]
    if missing:
        raise ValueError(f"key contains letters missing from keyed alphabet: {missing}")
    out: list[str] = []
    for i, c in enumerate(values):
        key_shift = index[key_letters[i % len(key_letters)]]
        out.append(alphabet[(c - key_shift) % 26])
    return "".join(out)


def encode_keyed_vigenere_values(
    values: list[int],
    *,
    key: str,
    keyed_alphabet: str | None = None,
    alphabet_keyword: str | None = None,
) -> str:
    """Encode A-Z plaintext values with a keyed Vigenere tableau."""
    alphabet = normalize_keyed_alphabet(
        keyed_alphabet=keyed_alphabet,
        alphabet_keyword=alphabet_keyword,
    )
    index = {ch: i for i, ch in enumerate(alphabet)}
    key_letters = [ch for ch in key.upper() if "A" <= ch <= "Z"]
    if not key_letters:
        raise ValueError("key must contain A-Z letters")
    out: list[str] = []
    for i, p in enumerate(values):
        plain_letter = chr(ord("A") + p)
        if plain_letter not in index:
            raise ValueError(f"plaintext letter missing from keyed alphabet: {plain_letter}")
        key_shift = index[key_letters[i % len(key_letters)]]
        out.append(alphabet[(index[plain_letter] + key_shift) % 26])
    return "".join(out)


def encode_keyed_vigenere_plaintext(
    plaintext: str,
    *,
    key: str,
    keyed_alphabet: str | None = None,
    alphabet_keyword: str | None = None,
) -> str:
    values = [ord(ch) - ord("A") for ch in plaintext.upper() if "A" <= ch <= "Z"]
    return encode_keyed_vigenere_values(
        values,
        key=key,
        keyed_alphabet=keyed_alphabet,
        alphabet_keyword=alphabet_keyword,
    )


def replay_keyed_vigenere(
    cipher_text: CipherText,
    *,
    key: str,
    keyed_alphabet: str | None = None,
    alphabet_keyword: str | None = None,
) -> dict[str, Any]:
    """Decode a keyed-alphabet Vigenere case using supplied known parameters."""
    alphabet = normalize_keyed_alphabet(
        keyed_alphabet=keyed_alphabet,
        alphabet_keyword=alphabet_keyword,
    )
    index = {ch: i for i, ch in enumerate(alphabet)}
    values: list[int] = []
    positions: list[int] = []
    skipped: list[dict[str, Any]] = []
    for pos, token_id in enumerate(cipher_text.tokens):
        symbol = cipher_text.alphabet.symbol_for(token_id).upper()
        if len(symbol) == 1 and symbol in index:
            values.append(index[symbol])
            positions.append(pos)
        else:
            skipped.append({"position": pos, "symbol": symbol})
    if len(values) < 1:
        return {
            "status": "unsupported",
            "reason": "too_few_a_z_tokens",
            "plaintext": "",
            "skipped_symbols": skipped[:20],
        }
    plaintext = decode_keyed_vigenere_values(
        values,
        key=key,
        keyed_alphabet=alphabet,
    )
    key_letters = [ch for ch in key.upper() if "A" <= ch <= "Z"]
    return {
        "status": "completed",
        "solver": "keyed_vigenere_known_replay",
        "key_type": "PeriodicAlphabetKey",
        "variant": "keyed_vigenere",
        "period": len(key_letters),
        "key": "".join(key_letters),
        "keyed_alphabet": alphabet,
        "alphabet_keyword": alphabet_keyword,
        "token_count": len(values),
        "original_token_count": len(cipher_text.tokens),
        "skipped_symbol_count": len(skipped),
        "skipped_symbols": skipped[:20],
        "key_advances_over_skipped_symbols": False,
        "plaintext": plaintext,
        "positions": positions[:20],
        "note": (
            "Known-parameter keyed Vigenere replay. Non-A-Z symbols are omitted "
            "from the decoded stream and do not advance the periodic key."
        ),
    }


def search_keyed_vigenere(
    cipher_text: CipherText,
    *,
    language: str = "en",
    periods: list[int] | None = None,
    max_period: int = 20,
    keyed_alphabets: list[str] | None = None,
    alphabet_keywords: list[str] | None = None,
    include_standard_alphabet: bool = True,
    top_n: int = 5,
    refine: bool = True,
) -> dict[str, Any]:
    """Recover periodic shifts for candidate keyed Vigenere alphabets.

    This is unknown periodic-key recovery, not full alphabet discovery: callers
    supply one or more candidate keyed alphabets/tableau keywords, and the
    solver searches periods and shifts under each shared alphabet.
    """
    alphabet_candidates = _keyed_alphabet_candidates(
        keyed_alphabets=keyed_alphabets,
        alphabet_keywords=alphabet_keywords,
        include_standard_alphabet=include_standard_alphabet,
    )
    if not alphabet_candidates:
        return {
            "status": "unsupported",
            "reason": "no_keyed_alphabet_candidates",
            "candidates": [],
        }

    quad = ngram.NGRAM_CACHE.get(language, 4)
    candidates: list[PeriodicCandidate] = []
    tested_periods: set[int] = set()
    skipped_summary: list[dict[str, Any]] = []
    token_count = 0
    for alphabet_info in alphabet_candidates:
        alphabet = alphabet_info["keyed_alphabet"]
        values, _positions, skipped = _keyed_cipher_indices(cipher_text, alphabet)
        token_count = max(token_count, len(values))
        if not skipped_summary:
            skipped_summary = skipped[:20]
        if len(values) < 12:
            continue
        period_values = sorted(set(int(p) for p in (periods or range(1, max_period + 1)) if int(p) >= 1))
        period_values = [p for p in period_values if p <= max(1, len(values) // 4)]
        if not period_values:
            period_values = [1]
        tested_periods.update(period_values)
        for period in period_values:
            shifts = _initial_keyed_shifts(values, period, alphabet)
            plaintext = decode_keyed_vigenere_shifts(values, shifts, keyed_alphabet=alphabet)
            init_score = ngram.normalized_ngram_score(plaintext, quad, n=4)
            score = init_score
            if refine:
                shifts, plaintext, score = _refine_keyed_shifts(
                    values,
                    shifts,
                    keyed_alphabet=alphabet,
                    log_probs=quad,
                )
            candidates.append(
                PeriodicCandidate(
                    variant="keyed_vigenere",
                    period=period,
                    shifts=shifts,
                    plaintext=plaintext,
                    score=score,
                    selection_score=score - _period_complexity_penalty(period),
                    init_score=init_score,
                    key=_key_string_for_alphabet(shifts, alphabet),
                    metadata={
                        "refined": refine,
                        "score_model": "wordlist_quadgram",
                        "key_type": "PeriodicAlphabetKey",
                        "keyed_alphabet": alphabet,
                        "alphabet_keyword": alphabet_info.get("alphabet_keyword"),
                        "key_advances_over_skipped_symbols": False,
                    },
                )
            )

    candidates.sort(key=lambda c: c.selection_score, reverse=True)
    best = candidates[0] if candidates else None
    return {
        "status": "completed" if best else "no_candidates",
        "solver": "keyed_vigenere_periodic_key_search",
        "language": language,
        "token_count": token_count,
        "skipped_symbols": skipped_summary,
        "skipped_symbol_count": _skipped_symbol_count(cipher_text),
        "periods_tested": sorted(tested_periods),
        "alphabet_candidates_tested": alphabet_candidates,
        "top_candidates": [c.to_dict() for c in candidates[: max(1, top_n)]],
        "best_candidate": best.to_dict() if best else None,
        "normalization_note": (
            "Symbols absent from a candidate keyed alphabet were skipped and "
            "did not advance the periodic key."
        ),
        "scope_note": (
            "This recovers the periodic key for supplied candidate keyed "
            "alphabets. It does not yet discover an arbitrary keyed alphabet."
        ),
    }


def search_keyed_vigenere_alphabet_anneal(
    cipher_text: CipherText,
    *,
    language: str = "en",
    periods: list[int] | None = None,
    max_period: int = 20,
    initial_alphabets: list[str] | None = None,
    alphabet_keywords: list[str] | None = None,
    include_standard_alphabet: bool = True,
    steps: int = 2000,
    restarts: int = 4,
    seed: int = 1,
    top_n: int = 5,
) -> dict[str, Any]:
    """Experimental shared-tableau mutation search.

    This mutates the shared keyed alphabet, re-optimizes periodic shifts for
    each candidate alphabet, and scores the entire plaintext. It is intended as
    the first non-wordlist tableau-recovery scaffold; it is not yet a robust
    blind Kryptos solver.
    """
    starts = _keyed_alphabet_candidates(
        keyed_alphabets=initial_alphabets,
        alphabet_keywords=alphabet_keywords,
        include_standard_alphabet=include_standard_alphabet,
    )
    if not starts:
        return {
            "status": "unsupported",
            "reason": "no_initial_alphabets",
            "candidates": [],
        }
    rng = random.Random(seed)
    quad = ngram.NGRAM_CACHE.get(language, 4)
    all_candidates: list[PeriodicCandidate] = []
    tested_periods: set[int] = set()
    best_skipped: list[dict[str, Any]] = []
    best_token_count = 0
    for start_info in starts:
        start_alpha = start_info["keyed_alphabet"]
        for restart in range(max(1, restarts)):
            alpha = start_alpha
            if restart:
                alpha = _scramble_alphabet(alpha, rng, swaps=min(26, 3 + restart * 2))
            values, _positions, skipped = _keyed_cipher_indices(cipher_text, alpha)
            if not best_skipped:
                best_skipped = skipped[:20]
            best_token_count = max(best_token_count, len(values))
            if len(values) < 12:
                continue
            period_values = sorted(set(int(p) for p in (periods or range(1, max_period + 1)) if int(p) >= 1))
            period_values = [p for p in period_values if p <= max(1, len(values) // 4)]
            if not period_values:
                period_values = [1]
            tested_periods.update(period_values)
            for period in period_values:
                candidate = _anneal_keyed_alphabet_for_period(
                    cipher_text,
                    alphabet=alpha,
                    period=period,
                    log_probs=quad,
                    steps=max(0, steps),
                    rng=rng,
                    start_info=start_info,
                    restart=restart,
                )
                all_candidates.append(candidate)

    all_candidates.sort(key=lambda c: c.selection_score, reverse=True)
    best = all_candidates[0] if all_candidates else None
    return {
        "status": "completed" if best else "no_candidates",
        "solver": "keyed_vigenere_alphabet_anneal",
        "language": language,
        "token_count": best_token_count,
        "skipped_symbols": best_skipped,
        "skipped_symbol_count": _skipped_symbol_count(cipher_text),
        "periods_tested": sorted(tested_periods),
        "initial_alphabets_tested": starts,
        "steps_per_period": max(0, steps),
        "restarts_per_alphabet": max(1, restarts),
        "top_candidates": [c.to_dict() for c in all_candidates[: max(1, top_n)]],
        "best_candidate": best.to_dict() if best else None,
        "scope_note": (
            "Experimental shared-alphabet mutation search. It can refine or "
            "escape near-basin alphabets, but broad blind tableau discovery "
            "still needs better proposal priors and crib/context scoring."
        ),
    }


def decode_keyed_vigenere_shifts(values: list[int], shifts: list[int], *, keyed_alphabet: str) -> str:
    if not shifts:
        return ""
    out: list[str] = []
    for i, c in enumerate(values):
        shift = shifts[i % len(shifts)]
        out.append(keyed_alphabet[(c - shift) % 26])
    return "".join(out)


def decode_values(values: list[int], shifts: list[int], *, variant: str = "vigenere") -> str:
    if not shifts:
        return ""
    out: list[str] = []
    for i, c in enumerate(values):
        k = shifts[i % len(shifts)]
        if variant == "beaufort":
            p = (k - c) % 26
        elif variant == "variant_beaufort":
            p = (c + k) % 26
        else:
            p = (c - k) % 26
        out.append(chr(ord("A") + p))
    return "".join(out)


def encode_values(values: list[int], shifts: list[int], *, variant: str = "vigenere") -> str:
    """Encode A-Z plaintext values with a periodic Vigenere-family key."""
    if not shifts:
        return ""
    out: list[str] = []
    for i, p in enumerate(values):
        k = shifts[i % len(shifts)]
        if variant == "beaufort":
            c = (k - p) % 26
        elif variant == "variant_beaufort":
            c = (p - k) % 26
        else:
            c = (p + k) % 26
        out.append(chr(ord("A") + c))
    return "".join(out)


def encode_plaintext(plaintext: str, shifts: list[int], *, variant: str = "vigenere") -> str:
    """Encode an A-Z plaintext string, ignoring non-letter separators."""
    values = [ord(ch) - ord("A") for ch in plaintext.upper() if "A" <= ch <= "Z"]
    return encode_values(values, shifts, variant=variant)


def decode_cipher_text(
    cipher_text: CipherText,
    shifts: list[int],
    *,
    variant: str = "vigenere",
) -> dict[str, Any]:
    """Decode clean A-Z CipherText with a periodic key."""
    values, skipped = cipher_values_from_text(cipher_text)
    if not shifts:
        return {
            "status": "error",
            "reason": "empty_periodic_key",
            "plaintext": "",
        }
    return {
        "status": "completed",
        "variant": variant,
        "period": len(shifts),
        "shifts": list(shifts),
        "key": _key_string(shifts, gronsfeld=(variant == "gronsfeld")),
        "plaintext": decode_values(values, shifts, variant=variant),
        "skipped_symbols": skipped[:20],
        "skipped_symbol_count": _skipped_symbol_count(cipher_text),
    }


def phase_frequency_report(
    cipher_text: CipherText,
    *,
    period: int,
    top_n: int = 8,
) -> dict[str, Any]:
    """Return per-phase frequency profiles for clean A-Z periodic ciphers."""
    values, skipped = cipher_values_from_text(cipher_text)
    if period <= 0:
        raise ValueError("period must be >= 1")

    rows: list[dict[str, Any]] = []
    for phase in range(period):
        stream = values[phase::period]
        counts = Counter(stream)
        rows.append({
            "phase": phase,
            "length": len(stream),
            "ic": round(_phase_ic(stream), 6),
            "top_cipher_letters": [
                {
                    "letter": chr(ord("A") + value),
                    "count": count,
                    "pct": round(count / len(stream) * 100, 2) if stream else 0.0,
                }
                for value, count in counts.most_common(max(1, top_n))
            ],
        })
    return {
        "status": "completed",
        "token_count": len(values),
        "skipped_symbols": skipped[:20],
        "skipped_symbol_count": _skipped_symbol_count(cipher_text),
        "period": period,
        "phase_rows": rows,
    }


def periodic_shift_candidates(
    cipher_text: CipherText,
    *,
    period: int,
    variant: str = "vigenere",
    top_n: int = 5,
    sample: int = 80,
) -> dict[str, Any]:
    """Return likely Caesar-shift candidates for each periodic phase."""
    values, skipped = cipher_values_from_text(cipher_text)
    variant = variant.strip().lower()
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported polyalphabetic variant: {variant}")
    if period <= 0:
        raise ValueError("period must be >= 1")

    rows: list[dict[str, Any]] = []
    shift_range = range(10) if variant == "gronsfeld" else range(26)
    for phase in range(period):
        stream = values[phase::period]
        candidates = []
        for shift in shift_range:
            plain_values = _phase_plain_values(stream, shift, variant)
            chi2 = _chi2_english(plain_values)
            decoded_sample = "".join(chr(ord("A") + v) for v in plain_values[:sample])
            candidates.append({
                "shift": shift,
                "key_symbol": _key_string([shift], gronsfeld=(variant == "gronsfeld")),
                "chi2": round(chi2, 4),
                "decoded_sample": decoded_sample,
            })
        candidates.sort(key=lambda item: item["chi2"])
        rows.append({
            "phase": phase,
            "length": len(stream),
            "top_candidates": candidates[: max(1, top_n)],
        })
    return {
        "status": "completed",
        "token_count": len(values),
        "skipped_symbols": skipped[:20],
        "skipped_symbol_count": _skipped_symbol_count(cipher_text),
        "period": period,
        "variant": variant,
        "phase_rows": rows,
        "note": (
            "Lower chi2 is better for a single phase. Use these as shift "
            "hints, then validate the whole decoded text with n-gram and reading evidence."
        ),
    }


def _skipped_symbol_count(cipher_text: CipherText) -> int:
    return len(cipher_values_with_positions(cipher_text)[2])


def _keyed_alphabet_candidates(
    *,
    keyed_alphabets: list[str] | None = None,
    alphabet_keywords: list[str] | None = None,
    include_standard_alphabet: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    if include_standard_alphabet:
        seen.add(AZ_ALPHABET)
        out.append({
            "keyed_alphabet": AZ_ALPHABET,
            "alphabet_keyword": None,
            "candidate_type": "standard_alphabet",
        })
    for alphabet in keyed_alphabets or []:
        normalized = normalize_keyed_alphabet(keyed_alphabet=alphabet)
        if normalized not in seen:
            seen.add(normalized)
            out.append({
                "keyed_alphabet": normalized,
                "alphabet_keyword": None,
                "candidate_type": "explicit_alphabet",
            })
    for keyword in alphabet_keywords or []:
        normalized = normalize_keyed_alphabet(alphabet_keyword=keyword)
        if normalized not in seen:
            seen.add(normalized)
            out.append({
                "keyed_alphabet": normalized,
                "alphabet_keyword": keyword.upper(),
                "candidate_type": "keyword_alphabet",
            })
    return out


def _keyed_cipher_indices(
    cipher_text: CipherText,
    keyed_alphabet: str,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    index = {ch: i for i, ch in enumerate(keyed_alphabet)}
    values: list[int] = []
    positions: list[int] = []
    skipped: list[dict[str, Any]] = []
    for pos, token_id in enumerate(cipher_text.tokens):
        symbol = cipher_text.alphabet.symbol_for(token_id).upper()
        if len(symbol) == 1 and symbol in index:
            values.append(index[symbol])
            positions.append(pos)
        else:
            skipped.append({"position": pos, "symbol": symbol})
    return values, positions, skipped


def parse_periodic_key(
    *,
    key: str | None = None,
    shifts: list[int] | None = None,
    variant: str = "vigenere",
) -> list[int]:
    """Parse a periodic key from letters, Gronsfeld digits, or shift values."""
    if shifts is not None:
        parsed = [int(s) % 26 for s in shifts]
    else:
        raw = (key or "").strip().upper()
        if not raw:
            raise ValueError("periodic key is empty")
        if variant == "gronsfeld":
            digits = [ch for ch in raw if ch.isdigit()]
            if not digits or len(digits) != len(raw.replace(" ", "")):
                raise ValueError("Gronsfeld keys must contain only digits")
            parsed = [int(ch) % 10 for ch in digits]
        else:
            letters = [ch for ch in raw if "A" <= ch <= "Z"]
            if not letters:
                raise ValueError("periodic key must contain A-Z letters")
            parsed = [ord(ch) - ord("A") for ch in letters]
    if not parsed:
        raise ValueError("periodic key is empty")
    if variant == "gronsfeld" and any(s < 0 or s > 9 for s in parsed):
        raise ValueError("Gronsfeld shifts must be digits 0-9")
    return parsed


def _initial_shifts(values: list[int], period: int, variant: str) -> list[int]:
    shifts: list[int] = []
    for phase in range(period):
        stream = values[phase::period]
        best_shift = 0
        best_score = float("inf")
        shift_range = range(10) if variant == "gronsfeld" else range(26)
        for shift in shift_range:
            plain = _phase_plain_values(stream, shift, variant)
            score = _chi2_english(plain)
            if score < best_score:
                best_score = score
                best_shift = shift
        shifts.append(best_shift)
    return shifts


def _initial_keyed_shifts(values: list[int], period: int, keyed_alphabet: str) -> list[int]:
    shifts: list[int] = []
    for phase in range(period):
        stream = values[phase::period]
        best_shift = 0
        best_score = float("inf")
        for shift in range(26):
            plain_values = _keyed_phase_plain_values(stream, shift, keyed_alphabet)
            score = _chi2_english(plain_values)
            if score < best_score:
                best_score = score
                best_shift = shift
        shifts.append(best_shift)
    return shifts


def _phase_plain_values(stream: list[int], shift: int, variant: str) -> list[int]:
    if variant == "beaufort":
        return [(shift - c) % 26 for c in stream]
    if variant == "variant_beaufort":
        return [(c + shift) % 26 for c in stream]
    return [(c - shift) % 26 for c in stream]


def _keyed_phase_plain_values(stream: list[int], shift: int, keyed_alphabet: str) -> list[int]:
    return [
        ord(keyed_alphabet[(c - shift) % 26]) - ord("A")
        for c in stream
    ]


def _chi2_english(values: list[int]) -> float:
    if not values:
        return float("inf")
    counts = [0] * 26
    for v in values:
        counts[v] += 1
    n = len(values)
    chi2 = 0.0
    for i, expected_freq in enumerate(ENGLISH_FREQS):
        expected = expected_freq * n
        if expected > 0:
            chi2 += (counts[i] - expected) ** 2 / expected
    return chi2


def _phase_ic(values: list[int]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    counts = Counter(values)
    numerator = sum(c * (c - 1) for c in counts.values())
    return numerator / (n * (n - 1))


def _refine_shifts(
    values: list[int],
    shifts: list[int],
    *,
    variant: str,
    log_probs: dict[str, float],
    gronsfeld_only: bool = False,
) -> tuple[list[int], str, float]:
    best_shifts = list(shifts)
    best_plain = decode_values(values, best_shifts, variant=variant)
    best_score = ngram.normalized_ngram_score(best_plain, log_probs, n=4)
    allowed = range(10) if gronsfeld_only else range(26)
    improved = True
    passes = 0
    while improved and passes < 4:
        improved = False
        passes += 1
        for phase in range(len(best_shifts)):
            phase_best_shift = best_shifts[phase]
            phase_best_plain = best_plain
            phase_best_score = best_score
            for candidate_shift in allowed:
                if candidate_shift == best_shifts[phase]:
                    continue
                trial = list(best_shifts)
                trial[phase] = candidate_shift
                plain = decode_values(values, trial, variant=variant)
                score = ngram.normalized_ngram_score(plain, log_probs, n=4)
                if score > phase_best_score:
                    phase_best_shift = candidate_shift
                    phase_best_plain = plain
                    phase_best_score = score
            if phase_best_shift != best_shifts[phase]:
                best_shifts[phase] = phase_best_shift
                best_plain = phase_best_plain
                best_score = phase_best_score
                improved = True
    return best_shifts, best_plain, best_score


def _refine_keyed_shifts(
    values: list[int],
    shifts: list[int],
    *,
    keyed_alphabet: str,
    log_probs: dict[str, float],
) -> tuple[list[int], str, float]:
    best_shifts = list(shifts)
    best_plain = decode_keyed_vigenere_shifts(values, best_shifts, keyed_alphabet=keyed_alphabet)
    best_score = ngram.normalized_ngram_score(best_plain, log_probs, n=4)
    improved = True
    passes = 0
    while improved and passes < 4:
        improved = False
        passes += 1
        for phase in range(len(best_shifts)):
            phase_best_shift = best_shifts[phase]
            phase_best_plain = best_plain
            phase_best_score = best_score
            for candidate_shift in range(26):
                if candidate_shift == best_shifts[phase]:
                    continue
                trial = list(best_shifts)
                trial[phase] = candidate_shift
                plain = decode_keyed_vigenere_shifts(values, trial, keyed_alphabet=keyed_alphabet)
                score = ngram.normalized_ngram_score(plain, log_probs, n=4)
                if score > phase_best_score:
                    phase_best_shift = candidate_shift
                    phase_best_plain = plain
                    phase_best_score = score
            if phase_best_shift != best_shifts[phase]:
                best_shifts[phase] = phase_best_shift
                best_plain = phase_best_plain
                best_score = phase_best_score
                improved = True
    return best_shifts, best_plain, best_score


def _anneal_keyed_alphabet_for_period(
    cipher_text: CipherText,
    *,
    alphabet: str,
    period: int,
    log_probs: dict[str, float],
    steps: int,
    rng: random.Random,
    start_info: dict[str, Any],
    restart: int,
) -> PeriodicCandidate:
    current_alpha = alphabet
    values, _positions, _skipped = _keyed_cipher_indices(cipher_text, current_alpha)
    current_shifts = _initial_keyed_shifts(values, period, current_alpha)
    current_shifts, current_plain, current_score = _refine_keyed_shifts(
        values,
        current_shifts,
        keyed_alphabet=current_alpha,
        log_probs=log_probs,
    )
    best_alpha = current_alpha
    best_shifts = list(current_shifts)
    best_plain = current_plain
    best_score = current_score
    for step in range(max(0, steps)):
        trial_alpha = _mutate_alphabet(current_alpha, rng)
        trial_values, _positions, _skipped = _keyed_cipher_indices(cipher_text, trial_alpha)
        trial_shifts = _initial_keyed_shifts(trial_values, period, trial_alpha)
        trial_shifts, trial_plain, trial_score = _refine_keyed_shifts(
            trial_values,
            trial_shifts,
            keyed_alphabet=trial_alpha,
            log_probs=log_probs,
        )
        temperature = 0.08 * (1.0 - (step / max(1, steps))) + 0.004
        if trial_score > current_score or rng.random() < math.exp((trial_score - current_score) / temperature):
            current_alpha = trial_alpha
            current_shifts = trial_shifts
            current_plain = trial_plain
            current_score = trial_score
            if trial_score > best_score:
                best_alpha = trial_alpha
                best_shifts = list(trial_shifts)
                best_plain = trial_plain
                best_score = trial_score
    return PeriodicCandidate(
        variant="keyed_vigenere",
        period=period,
        shifts=best_shifts,
        plaintext=best_plain,
        score=best_score,
        selection_score=best_score - _period_complexity_penalty(period),
        init_score=current_score,
        key=_key_string_for_alphabet(best_shifts, best_alpha),
        metadata={
            "key_type": "PeriodicAlphabetKey",
            "keyed_alphabet": best_alpha,
            "alphabet_keyword": start_info.get("alphabet_keyword"),
            "initial_keyed_alphabet": alphabet,
            "initial_candidate_type": start_info.get("candidate_type"),
            "restart": restart,
            "mutation_search": "swap_move_reverse",
            "score_model": "wordlist_quadgram",
        },
    )


def _mutate_alphabet(alphabet: str, rng: random.Random) -> str:
    chars = list(alphabet)
    move = rng.random()
    if move < 0.70:
        i, j = rng.sample(range(26), 2)
        chars[i], chars[j] = chars[j], chars[i]
    elif move < 0.90:
        i, j = rng.sample(range(26), 2)
        ch = chars.pop(i)
        chars.insert(j, ch)
    else:
        i, j = sorted(rng.sample(range(26), 2))
        chars[i:j + 1] = reversed(chars[i:j + 1])
    return "".join(chars)


def _scramble_alphabet(alphabet: str, rng: random.Random, *, swaps: int) -> str:
    chars = list(alphabet)
    for _ in range(max(0, swaps)):
        i, j = rng.sample(range(26), 2)
        chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def _key_string(shifts: list[int], *, gronsfeld: bool = False) -> str:
    if gronsfeld:
        return "".join(str(s) for s in shifts)
    return "".join(chr(ord("A") + s) for s in shifts)


def _key_string_for_alphabet(shifts: list[int], keyed_alphabet: str) -> str:
    return "".join(keyed_alphabet[s % 26] for s in shifts)


def _period_complexity_penalty(period: int) -> float:
    """Small model-selection penalty to avoid overfitting with long periods."""
    return max(0, period - 1) * 0.015
