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

from analysis import dictionary, ngram, pattern
from models.cipher_text import CipherText


ENGLISH_FREQS = [
    0.0817, 0.0149, 0.0278, 0.0425, 0.1270, 0.0223, 0.0202, 0.0609, 0.0697,
    0.0015, 0.0077, 0.0403, 0.0241, 0.0675, 0.0751, 0.0193, 0.0010, 0.0599,
    0.0633, 0.0906, 0.0276, 0.0098, 0.0236, 0.0015, 0.0197, 0.0007,
]

SUPPORTED_VARIANTS = {"vigenere", "beaufort", "variant_beaufort", "gronsfeld"}
AZ_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ENGLISH_FREQ_ORDER = "ETAOINSHRDLUCMWFGYPBVKJXQZ"
QUAGMIRE_TYPES = {"quag1", "quag2", "quag3", "quag4"}


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


def normalize_quagmire_type(quagmire_type: str | int) -> str:
    """Normalize Quagmire variant names to ``quag1`` ... ``quag4``."""
    raw = str(quagmire_type or "").strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "1": "quag1",
        "i": "quag1",
        "quagmire1": "quag1",
        "quagmirei": "quag1",
        "quag1": "quag1",
        "2": "quag2",
        "ii": "quag2",
        "quagmire2": "quag2",
        "quagmireii": "quag2",
        "quag2": "quag2",
        "3": "quag3",
        "iii": "quag3",
        "quagmire3": "quag3",
        "quagmireiii": "quag3",
        "quag3": "quag3",
        "4": "quag4",
        "iv": "quag4",
        "quagmire4": "quag4",
        "quagmireiv": "quag4",
        "quag4": "quag4",
    }
    if raw not in aliases:
        raise ValueError(f"unsupported Quagmire type: {quagmire_type!r}")
    return aliases[raw]


def _normalize_alphabet_text(alphabet: str | None, *, label: str) -> str:
    if alphabet is None:
        return AZ_ALPHABET
    letters = [ch for ch in alphabet.upper() if "A" <= ch <= "Z"]
    if len(letters) != 26 or len(set(letters)) != 26:
        raise ValueError(f"{label} must contain exactly 26 distinct A-Z letters")
    return "".join(letters)


def quagmire_alphabets(
    *,
    quagmire_type: str | int = "quag3",
    plaintext_alphabet: str | None = None,
    ciphertext_alphabet: str | None = None,
    plaintext_keyword: str | None = None,
    ciphertext_keyword: str | None = None,
    alphabet_keyword: str | None = None,
    keyed_alphabet: str | None = None,
) -> tuple[str, str, str]:
    """Return ``(type, plaintext_alphabet, ciphertext_alphabet)``.

    The Quagmire naming and the first search-port roadmap follow Sam Blake's
    MIT-licensed ``polyalphabetic`` solver.  This replay helper is an
    independent implementation of the classical tableau semantics: encrypt by
    finding a plaintext letter's position in the plaintext alphabet, adding the
    cycleword letter's position in the ciphertext alphabet, then reading from
    the ciphertext alphabet.  Decryption subtracts that periodic offset.
    """
    qtype = normalize_quagmire_type(quagmire_type)
    shared = None
    if keyed_alphabet or alphabet_keyword:
        shared = normalize_keyed_alphabet(
            keyed_alphabet=keyed_alphabet,
            alphabet_keyword=alphabet_keyword,
        )

    if qtype == "quag1":
        pt_alpha = _normalize_alphabet_text(plaintext_alphabet, label="plaintext_alphabet")
        if plaintext_keyword and plaintext_alphabet is None:
            pt_alpha = keyed_alphabet_from_keyword(plaintext_keyword)
        if shared and plaintext_alphabet is None and plaintext_keyword is None:
            pt_alpha = shared
        ct_alpha = _normalize_alphabet_text(ciphertext_alphabet, label="ciphertext_alphabet")
        if ciphertext_keyword and ciphertext_alphabet is None:
            ct_alpha = keyed_alphabet_from_keyword(ciphertext_keyword)
    elif qtype == "quag2":
        pt_alpha = _normalize_alphabet_text(plaintext_alphabet, label="plaintext_alphabet")
        if plaintext_keyword and plaintext_alphabet is None:
            pt_alpha = keyed_alphabet_from_keyword(plaintext_keyword)
        ct_alpha = _normalize_alphabet_text(ciphertext_alphabet, label="ciphertext_alphabet")
        if ciphertext_keyword and ciphertext_alphabet is None:
            ct_alpha = keyed_alphabet_from_keyword(ciphertext_keyword)
        if shared and ciphertext_alphabet is None and ciphertext_keyword is None:
            ct_alpha = shared
    elif qtype == "quag3":
        if plaintext_alphabet or ciphertext_alphabet:
            pt_alpha = _normalize_alphabet_text(plaintext_alphabet or ciphertext_alphabet, label="plaintext_alphabet")
            ct_alpha = _normalize_alphabet_text(ciphertext_alphabet or plaintext_alphabet, label="ciphertext_alphabet")
        else:
            alpha_keyword = alphabet_keyword or plaintext_keyword or ciphertext_keyword
            pt_alpha = normalize_keyed_alphabet(
                keyed_alphabet=keyed_alphabet,
                alphabet_keyword=alpha_keyword,
            )
            ct_alpha = pt_alpha
    else:
        pt_alpha = _normalize_alphabet_text(plaintext_alphabet, label="plaintext_alphabet")
        ct_alpha = _normalize_alphabet_text(ciphertext_alphabet, label="ciphertext_alphabet")
        if plaintext_keyword and plaintext_alphabet is None:
            pt_alpha = keyed_alphabet_from_keyword(plaintext_keyword)
        if ciphertext_keyword and ciphertext_alphabet is None:
            ct_alpha = keyed_alphabet_from_keyword(ciphertext_keyword)
        if shared and plaintext_alphabet is None and ciphertext_alphabet is None:
            pt_alpha = shared
            ct_alpha = shared

    return qtype, pt_alpha, ct_alpha


def _cycleword_letters(cycleword: str) -> list[str]:
    letters = [ch for ch in (cycleword or "").upper() if "A" <= ch <= "Z"]
    if not letters:
        raise ValueError("cycleword must contain A-Z letters")
    return letters


def encode_quagmire_plaintext(
    plaintext: str,
    *,
    cycleword: str,
    quagmire_type: str | int = "quag3",
    plaintext_alphabet: str | None = None,
    ciphertext_alphabet: str | None = None,
    plaintext_keyword: str | None = None,
    ciphertext_keyword: str | None = None,
    alphabet_keyword: str | None = None,
    keyed_alphabet: str | None = None,
) -> str:
    """Encode A-Z plaintext under a Quagmire tableau with known parameters."""
    _qtype, pt_alpha, ct_alpha = quagmire_alphabets(
        quagmire_type=quagmire_type,
        plaintext_alphabet=plaintext_alphabet,
        ciphertext_alphabet=ciphertext_alphabet,
        plaintext_keyword=plaintext_keyword,
        ciphertext_keyword=ciphertext_keyword,
        alphabet_keyword=alphabet_keyword,
        keyed_alphabet=keyed_alphabet,
    )
    pt_index = {ch: i for i, ch in enumerate(pt_alpha)}
    ct_letters = _cycleword_letters(cycleword)
    ct_index = {ch: i for i, ch in enumerate(ct_alpha)}
    missing = [ch for ch in ct_letters if ch not in ct_index]
    if missing:
        raise ValueError(f"cycleword contains letters missing from ciphertext alphabet: {missing}")
    out: list[str] = []
    for i, ch in enumerate(plaintext.upper()):
        if not ("A" <= ch <= "Z"):
            continue
        if ch not in pt_index:
            raise ValueError(f"plaintext letter missing from plaintext alphabet: {ch}")
        shift = ct_index[ct_letters[i % len(ct_letters)]]
        out.append(ct_alpha[(pt_index[ch] + shift) % 26])
    return "".join(out)


def replay_quagmire(
    cipher_text: CipherText,
    *,
    cycleword: str,
    quagmire_type: str | int = "quag3",
    plaintext_alphabet: str | None = None,
    ciphertext_alphabet: str | None = None,
    plaintext_keyword: str | None = None,
    ciphertext_keyword: str | None = None,
    alphabet_keyword: str | None = None,
    keyed_alphabet: str | None = None,
) -> dict[str, Any]:
    """Decode a Quagmire case using supplied known parameters."""
    qtype, pt_alpha, ct_alpha = quagmire_alphabets(
        quagmire_type=quagmire_type,
        plaintext_alphabet=plaintext_alphabet,
        ciphertext_alphabet=ciphertext_alphabet,
        plaintext_keyword=plaintext_keyword,
        ciphertext_keyword=ciphertext_keyword,
        alphabet_keyword=alphabet_keyword,
        keyed_alphabet=keyed_alphabet,
    )
    ct_index = {ch: i for i, ch in enumerate(ct_alpha)}
    key_letters = _cycleword_letters(cycleword)
    missing = [ch for ch in key_letters if ch not in ct_index]
    if missing:
        raise ValueError(f"cycleword contains letters missing from ciphertext alphabet: {missing}")

    positions: list[int] = []
    skipped: list[dict[str, Any]] = []
    out: list[str] = []
    for pos, token_id in enumerate(cipher_text.tokens):
        symbol = cipher_text.alphabet.symbol_for(token_id).upper()
        if len(symbol) == 1 and symbol in ct_index:
            shift = ct_index[key_letters[len(out) % len(key_letters)]]
            out.append(pt_alpha[(ct_index[symbol] - shift) % 26])
            positions.append(pos)
        else:
            skipped.append({"position": pos, "symbol": symbol})

    if not out:
        return {
            "status": "unsupported",
            "reason": "too_few_a_z_tokens",
            "plaintext": "",
            "skipped_symbols": skipped[:20],
        }

    return {
        "status": "completed",
        "solver": f"{qtype}_known_replay",
        "key_type": "QuagmireKey",
        "variant": qtype,
        "quagmire_type": qtype,
        "period": len(key_letters),
        "cycleword": "".join(key_letters),
        "plaintext_alphabet": pt_alpha,
        "ciphertext_alphabet": ct_alpha,
        "alphabet_keyword": alphabet_keyword,
        "plaintext_keyword": plaintext_keyword,
        "ciphertext_keyword": ciphertext_keyword,
        "token_count": len(out),
        "original_token_count": len(cipher_text.tokens),
        "skipped_symbol_count": len(skipped),
        "skipped_symbols": skipped[:20],
        "key_advances_over_skipped_symbols": False,
        "plaintext": "".join(out),
        "positions": positions[:20],
        "attribution": (
            "Quagmire terminology and search-port roadmap reference Sam "
            "Blake's MIT-licensed polyalphabetic solver; this known-parameter "
            "replay implementation is independent Decipher code."
        ),
        "note": (
            "Known-parameter Quagmire replay. Unsupported symbols are omitted "
            "from the decoded stream and do not advance the cycleword."
        ),
    }


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


def search_quagmire3_keyword_alphabet(
    cipher_text: CipherText,
    *,
    language: str = "en",
    keyword_lengths: list[int] | None = None,
    cycleword_lengths: list[int] | None = None,
    initial_keywords: list[str] | None = None,
    steps: int = 500,
    restarts: int = 8,
    seed: int = 1,
    top_n: int = 5,
    refine: bool = True,
    screen_top_n: int = 128,
    word_weight: float = 0.25,
    slip_probability: float = 0.001,
    backtrack_probability: float = 0.15,
    dictionary_keyword_limit: int = 0,
    calibration_keyword: str | None = None,
) -> dict[str, Any]:
    """Search Quagmire III keyword-shaped alphabets plus derived cyclewords.

    This is the first Decipher scaffold inspired by Sam Blake's MIT-licensed
    ``polyalphabetic`` solver: it searches keyword-shaped alphabets instead of
    arbitrary 26! tableaux, and for each alphabet derives the best cycleword
    for one or more candidate lengths.  It is intentionally bounded and
    diagnostic; it does not yet port Blake's full shotgun/backtracking search.
    """
    initial_keywords = initial_keywords or []
    keyword_lengths = sorted(set(int(n) for n in (keyword_lengths or [7]) if int(n) >= 1))
    cycleword_lengths = sorted(set(int(n) for n in (cycleword_lengths or range(1, 13)) if int(n) >= 1))
    if not keyword_lengths:
        keyword_lengths = [7]
    if not cycleword_lengths:
        cycleword_lengths = [1]
    calibration_target = _normalize_keyword_state(
        calibration_keyword or "",
        max(1, len(calibration_keyword or "")),
    )

    rng = random.Random(seed)
    quad = ngram.NGRAM_CACHE.get(language, 4)
    candidates: list[PeriodicCandidate] = []
    word_set = dictionary.load_word_set(dictionary.get_dictionary_path(language) or "")
    token_count = 0
    skipped_summary: list[dict[str, Any]] = []
    dictionary_keywords = _dictionary_keyword_starts(
        language,
        keyword_lengths=keyword_lengths,
        limit=max(0, dictionary_keyword_limit),
    )

    def evaluate(
        keyword: str,
        period: int,
        *,
        metadata: dict[str, Any],
        refine_candidate: bool,
    ) -> PeriodicCandidate | None:
        nonlocal token_count, skipped_summary
        clean_keyword = _normalize_keyword_state(keyword, max(1, len(keyword)))
        if not clean_keyword:
            return None
        alphabet = keyed_alphabet_from_keyword(clean_keyword)
        values, _positions, skipped = _keyed_cipher_indices(cipher_text, alphabet)
        token_count = max(token_count, len(values))
        if not skipped_summary:
            skipped_summary = skipped[:20]
        if len(values) < 12:
            return None
        shifts = _initial_keyed_shifts(values, period, alphabet)
        plaintext = decode_keyed_vigenere_shifts(values, shifts, keyed_alphabet=alphabet)
        init_score = ngram.normalized_ngram_score(plaintext, quad, n=4)
        score = init_score
        if refine_candidate:
            shifts, plaintext, score = _refine_keyed_shifts(
                values,
                shifts,
                keyed_alphabet=alphabet,
                log_probs=quad,
            )
        word_score = (
            _continuous_word_hit_score(plaintext, word_set)
            if refine_candidate and word_set and word_weight > 0
            else 0.0
        )
        selection_score = (
            score
            + word_score * max(0.0, word_weight)
            - _period_complexity_penalty(period)
            - max(0, len(clean_keyword) - 1) * 0.002
        )
        return PeriodicCandidate(
            variant="quag3",
            period=period,
            shifts=shifts,
            plaintext=plaintext,
            score=score,
            selection_score=selection_score,
            init_score=init_score,
            key=_key_string_for_alphabet(shifts, alphabet),
            metadata={
                **metadata,
                "key_type": "QuagmireKey",
                "quagmire_type": "quag3",
                "cycleword": _key_string_for_alphabet(shifts, alphabet),
                "cycleword_shifts": list(shifts),
                "alphabet_keyword": clean_keyword,
                "plaintext_alphabet": alphabet,
                "ciphertext_alphabet": alphabet,
                "keyword_length": len(clean_keyword),
                "refined_cycleword": refine_candidate,
                "score_model": "wordlist_quadgram",
                "word_score": round(word_score, 5),
                "word_weight": max(0.0, word_weight),
                **(
                    {
                        "calibration_keyword": calibration_target,
                        "calibration_keyword_distance": _keyword_distance(
                            clean_keyword,
                            calibration_target,
                        ),
                    }
                    if calibration_target else {}
                ),
            },
        )

    start_keywords: list[tuple[str, dict[str, Any]]] = []
    for keyword in initial_keywords:
        clean = _normalize_keyword_state(keyword, max(1, len(keyword)))
        if clean:
            start_keywords.append((clean, {"start_type": "explicit_initial_keyword"}))
    for keyword in dictionary_keywords:
        clean = _normalize_keyword_state(keyword, max(1, len(keyword)))
        if clean:
            start_keywords.append((clean, {"start_type": "dictionary_keyword"}))
    for length in keyword_lengths:
        for restart in range(max(1, restarts)):
            start_keywords.append((
                _random_keyword_state(length, rng),
                {
                    "start_type": "random_keyword",
                    "restart": restart,
                    "target_keyword_length": length,
                },
            ))

    screen_candidates: list[PeriodicCandidate] = []
    seen_screen: set[tuple[str, int]] = set()
    accepted_screen_mutations = 0
    slipped_screen_mutations = 0
    backtrack_events = 0

    def add_screen_candidate(
        keyword: str,
        period: int,
        *,
        metadata: dict[str, Any],
    ) -> PeriodicCandidate | None:
        clean_keyword = _normalize_keyword_state(keyword, max(1, len(keyword)))
        if not clean_keyword:
            return None
        alphabet = keyed_alphabet_from_keyword(clean_keyword)
        state = (alphabet, period)
        if state in seen_screen:
            return None
        seen_screen.add(state)
        candidate = evaluate(
            clean_keyword,
            period,
            metadata={**metadata, "search_stage": "screen"},
            refine_candidate=False,
        )
        if candidate is not None:
            screen_candidates.append(candidate)
        return candidate

    for start_index, (start_keyword, start_meta) in enumerate(start_keywords):
        for period in cycleword_lengths:
            current_keyword = start_keyword
            current = add_screen_candidate(
                current_keyword,
                period,
                metadata={
                    **start_meta,
                    "start_index": start_index,
                    "step": 0,
                    "screen_stage": "start",
                },
            )
            if current is None:
                continue
            current_score = current.selection_score
            best_keyword = current_keyword
            best_score = current_score
            for step in range(1, max(0, steps) + 1):
                if rng.random() < max(0.0, backtrack_probability):
                    current_keyword = best_keyword
                    current_score = best_score
                    backtrack_events += 1
                trial_keyword = _mutate_keyword_state(current_keyword, rng)
                trial = add_screen_candidate(
                    trial_keyword,
                    period,
                    metadata={
                        **start_meta,
                        "start_index": start_index,
                        "step": step,
                        "screen_stage": "mutated_keyword",
                        "mutation_search": "blake_prefix_swap_or_tail_exchange",
                        "parent_keyword": current_keyword,
                    },
                )
                if trial is None:
                    continue
                delta = trial.selection_score - current_score
                slipped = False
                if delta >= 0 or rng.random() < max(0.0, slip_probability):
                    slipped = delta < 0
                    current_keyword = str(trial.metadata["alphabet_keyword"])
                    current_score = trial.selection_score
                    accepted_screen_mutations += 1
                    if slipped:
                        slipped_screen_mutations += 1
                if trial.selection_score > best_score:
                    best_keyword = str(trial.metadata["alphabet_keyword"])
                    best_score = trial.selection_score
                if slipped:
                    trial.metadata["accepted_by_slip"] = True
                if trial.selection_score >= best_score:
                    trial.metadata["best_screen_score_so_far"] = round(best_score, 5)

    screen_candidates.sort(key=lambda c: c.selection_score, reverse=True)
    finalist_count = max(1, min(len(screen_candidates), screen_top_n if refine else max(top_n, screen_top_n)))
    finalists = screen_candidates[:finalist_count]
    if refine:
        for rank, candidate in enumerate(finalists, start=1):
            keyword = str(candidate.metadata["alphabet_keyword"])
            refined = evaluate(
                keyword,
                candidate.period,
                metadata={
                    **candidate.metadata,
                    "search_stage": "refined_finalist",
                    "screen_rank": rank,
                    "screen_score": candidate.score,
                },
                refine_candidate=True,
            )
            if refined is not None:
                candidates.append(refined)
    else:
        candidates = finalists

    candidates.sort(key=lambda c: c.selection_score, reverse=True)
    best = candidates[0] if candidates else None
    calibration_rows = [
        {
            "rank": rank,
            "alphabet_keyword": str(candidate.metadata.get("alphabet_keyword")),
            "distance": int(candidate.metadata.get("calibration_keyword_distance")),
            "score": round(candidate.score, 5),
            "selection_score": round(candidate.selection_score, 5),
        }
        for rank, candidate in enumerate(candidates, start=1)
        if calibration_target and candidate.metadata.get("calibration_keyword_distance") is not None
    ]
    exact_calibration_rank = next(
        (row["rank"] for row in calibration_rows if row["distance"] == 0),
        None,
    )
    return {
        "status": "completed" if best else "no_candidates",
        "solver": "quagmire3_keyword_alphabet_search",
        "language": language,
        "token_count": token_count,
        "skipped_symbols": skipped_summary,
        "skipped_symbol_count": _skipped_symbol_count(cipher_text),
        "keyword_lengths": keyword_lengths,
        "cycleword_lengths": cycleword_lengths,
        "initial_keywords": [_normalize_keyword_state(k, max(1, len(k))) for k in initial_keywords],
        "dictionary_keyword_limit": max(0, dictionary_keyword_limit),
        "dictionary_keywords_loaded": len(dictionary_keywords),
        "calibration_keyword": calibration_target or None,
        "exact_calibration_keyword_rank": exact_calibration_rank,
        "best_calibration_keyword_distance": (
            min((row["distance"] for row in calibration_rows), default=None)
            if calibration_target else None
        ),
        "steps_per_start": max(0, steps),
        "restarts_per_length": max(1, restarts),
        "keyword_states_screened": len(seen_screen),
        "screen_top_n": max(1, screen_top_n),
        "refined_finalist_count": len(candidates) if refine else 0,
        "word_weight": max(0.0, word_weight),
        "screen_search": "score_guided_keyword_hill_climb",
        "slip_probability": max(0.0, slip_probability),
        "backtrack_probability": max(0.0, backtrack_probability),
        "accepted_screen_mutations": accepted_screen_mutations,
        "slipped_screen_mutations": slipped_screen_mutations,
        "backtrack_events": backtrack_events,
        "seed": seed,
        "top_candidates": [c.to_dict() for c in candidates[: max(1, top_n)]],
        "top_calibration_matches": calibration_rows[: max(1, top_n)] if calibration_target else [],
        "best_candidate": best.to_dict() if best else None,
        "attribution": (
            "Search strategy scaffold inspired by Sam Blake's MIT-licensed "
            "polyalphabetic solver: search keyword-shaped Quagmire alphabets "
            "and derive/optimize the cycleword for each candidate."
        ),
        "scope_note": (
            "Bounded Quagmire III keyword-alphabet search. This is not yet the "
            "full Blake shotgun/backtracking search; use it for calibration "
            "and candidate-generation experiments."
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
    guided: bool = True,
    guided_pool_size: int = 24,
    top_n: int = 5,
) -> dict[str, Any]:
    """Experimental shared-tableau mutation search.

    This mutates the shared keyed alphabet, re-optimizes periodic shifts for
    each candidate alphabet, and scores the entire plaintext. Guided mode adds
    frequency-pressure and per-phase common-letter swaps to the random
    swap/move/reverse mutations. It is intended as the first non-wordlist
    tableau-recovery scaffold; it is not yet a robust blind Kryptos solver.
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
                    guided=guided,
                    guided_pool_size=max(1, guided_pool_size),
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
        "guided": bool(guided),
        "guided_pool_size": max(1, guided_pool_size),
        "top_candidates": [c.to_dict() for c in all_candidates[: max(1, top_n)]],
        "best_candidate": best.to_dict() if best else None,
        "scope_note": (
            "Experimental shared-alphabet mutation search. Guided proposals "
            "use letter-frequency pressure and per-phase common-letter "
            "hypotheses, but broad blind tableau discovery still needs "
            "stronger structural priors, beam search, and crib/context scoring."
        ),
    }


def generate_keyed_vigenere_constraint_alphabets(
    cipher_text: CipherText,
    *,
    max_period: int = 12,
    beam_size: int = 64,
    limit: int = 128,
    random_limit: int = 0,
    random_seed: int = 1,
    top_shifts: int = 3,
    top_letters: int = 3,
    target_window: int = 3,
    random_phases: int | None = None,
) -> dict[str, Any]:
    """Generate keyed-tableau start alphabets from phase-frequency constraints.

    This is an experimental initializer, not a complete solver.  It uses
    per-phase Caesar-shift evidence under the current alphabet and asks:
    "what keyed alphabet would make the most frequent cipher letters in this
    phase land on common English letters under one of the plausible shifts?"

    The resulting alphabets are intended to seed the Rust/Python annealer so it
    can start from structured hypotheses instead of pure random walks.
    """
    values, skipped = cipher_values_from_text(cipher_text)
    if len(values) < 12:
        return {
            "status": "unsupported",
            "reason": "too_few_a_z_tokens",
            "token_count": len(values),
            "alphabets": [],
        }

    period_values = list(range(1, min(max_period, max(1, len(values) // 4)) + 1))
    all_states: list[tuple[float, str, dict[str, Any]]] = []
    option_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for period in period_values:
        states: list[tuple[float, str, list[dict[str, Any]]]] = [(0.0, AZ_ALPHABET, [])]
        for phase in range(period):
            phase_options = option_cache[(period, phase)] = _phase_constraint_options(
                values,
                period=period,
                phase=phase,
                top_shifts=max(1, top_shifts),
                top_letters=max(1, top_letters),
                target_window=max(1, target_window),
            )
            if not phase_options:
                continue
            expanded: list[tuple[float, str, list[dict[str, Any]]]] = []
            for state_score, alphabet, history in states:
                for option in phase_options:
                    candidate_alphabet = _apply_phase_constraint_option(alphabet, option)
                    penalty = _alphabet_disruption(candidate_alphabet)
                    score = state_score + option["score"] - penalty * 0.002
                    expanded.append((score, candidate_alphabet, [*history, option]))
            expanded.sort(key=lambda item: item[0], reverse=True)
            deduped: dict[str, tuple[float, str, list[dict[str, Any]]]] = {}
            for state in expanded:
                if state[1] not in deduped:
                    deduped[state[1]] = state
                if len(deduped) >= max(1, beam_size):
                    break
            states = list(deduped.values())
        for score, alphabet, history in states:
            all_states.append((
                score - _period_complexity_penalty(period),
                alphabet,
                {
                    "period": period,
                    "score": round(score, 5),
                    "constraints": history[:24],
                    "constraint_count": len(history),
                },
            ))

    if random_limit > 0:
        rng = random.Random(random_seed)
        for sample_index in range(random_limit):
            period = rng.choice(period_values)
            alphabet = AZ_ALPHABET
            history: list[dict[str, Any]] = []
            score = 0.0
            phase_count = period if random_phases is None else max(1, min(period, random_phases))
            phases = list(range(period))
            rng.shuffle(phases)
            for phase in phases[:phase_count]:
                options = option_cache.get((period, phase))
                if options is None:
                    options = _phase_constraint_options(
                        values,
                        period=period,
                        phase=phase,
                        top_shifts=max(1, top_shifts),
                        top_letters=max(1, top_letters),
                        target_window=max(1, target_window),
                    )
                    option_cache[(period, phase)] = options
                if not options:
                    continue
                # Bias toward good options but keep enough entropy to discover
                # starts outside the deterministic beam.
                pool = options[: min(len(options), max(4, top_shifts * target_window))]
                weights = [1.0 / (idx + 1) for idx, _option in enumerate(pool)]
                option = rng.choices(pool, weights=weights, k=1)[0]
                alphabet = _apply_phase_constraint_option(alphabet, option)
                history.append(option)
                score += float(option["score"])
            all_states.append((
                score - _period_complexity_penalty(period) - _alphabet_disruption(alphabet) * 0.001,
                alphabet,
                {
                    "period": period,
                    "score": round(score, 5),
                    "constraints": history[:24],
                    "constraint_count": len(history),
                    "sample_index": sample_index,
                    "randomized": True,
                },
            ))

    all_states.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for selection_score, alphabet, metadata in all_states:
        if alphabet == AZ_ALPHABET or alphabet in seen:
            continue
        seen.add(alphabet)
        rows.append({
            "keyed_alphabet": alphabet,
            "selection_score": round(selection_score, 5),
            "metadata": metadata,
        })
        if len(rows) >= max(1, limit):
            break

    return {
        "status": "completed" if rows else "no_candidates",
        "solver": "keyed_vigenere_phase_constraint_alphabet_generator",
        "token_count": len(values),
        "skipped_symbols": skipped[:20],
        "skipped_symbol_count": len(skipped),
        "periods_tested": period_values,
        "beam_size": max(1, beam_size),
        "limit": max(1, limit),
        "random_limit": max(0, random_limit),
        "random_seed": random_seed,
        "top_shifts": max(1, top_shifts),
        "top_letters": max(1, top_letters),
        "target_window": max(1, target_window),
        "random_phases": random_phases,
        "alphabets": rows,
        "scope_note": (
            "Experimental phase-frequency constraint initializer for "
            "keyed-tableau search. Use as start alphabets for anneal/screen, "
            "not as a solved-key claim."
        ),
    }


def generate_keyed_vigenere_offset_graph_alphabets(
    cipher_text: CipherText,
    *,
    max_period: int = 12,
    limit: int = 1024,
    random_seed: int = 1,
    samples: int = 4096,
    phase_count: int | None = None,
    top_cipher_letters: int = 5,
    target_letters: int = 8,
    target_window: int = 4,
) -> dict[str, Any]:
    """Generate keyed-tableau starts from modular offset graph constraints.

    Each sampled phase hypothesis asserts constraints of the form
    `pos(cipher_letter) - pos(plain_letter) = phase_shift (mod 26)` for the
    most frequent cipher letters in a phase. A weighted union-find keeps only
    mutually consistent modular constraints, then materializes complete keyed
    alphabets by placing connected components on the 26-cycle.

    This is intentionally a generator for experiments: it does not use known
    plaintext, and it does not claim a solved key/tableau.
    """
    values, skipped = cipher_values_from_text(cipher_text)
    if len(values) < 12:
        return {
            "status": "unsupported",
            "reason": "too_few_a_z_tokens",
            "token_count": len(values),
            "alphabets": [],
        }

    rng = random.Random(random_seed)
    period_values = list(range(1, min(max_period, max(1, len(values) // 4)) + 1))
    phase_tops: dict[tuple[int, int], list[int]] = {}
    for period in period_values:
        for phase in range(period):
            counts = Counter(values[phase::period])
            phase_tops[(period, phase)] = [
                value for value, _count in counts.most_common(max(1, top_cipher_letters))
            ]

    target_pool = [ord(ch) - ord("A") for ch in ENGLISH_FREQ_ORDER[: max(1, target_letters)]]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample_index in range(max(0, samples)):
        period = rng.choice(period_values)
        phases = list(range(period))
        rng.shuffle(phases)
        selected_phases = phases[: max(1, min(period, phase_count or period))]
        graph = _ModOffsetGraph()
        accepted = 0
        attempted = 0
        history: list[dict[str, Any]] = []
        for phase in selected_phases:
            top_cipher = phase_tops.get((period, phase), [])
            if not top_cipher:
                continue
            shift = rng.randrange(26)
            target_offset = rng.randrange(max(1, target_window))
            targets = target_pool[target_offset:] + target_pool[:target_offset]
            if rng.random() < 0.25:
                targets = list(targets)
                rng.shuffle(targets)
            for rank, cipher_value in enumerate(top_cipher):
                if rank >= len(targets):
                    break
                target_value = targets[rank]
                if cipher_value == target_value:
                    continue
                attempted += 1
                if graph.union(cipher_value, target_value, shift):
                    accepted += 1
                    if len(history) < 24:
                        history.append({
                            "phase": phase,
                            "shift": shift,
                            "cipher": chr(ord("A") + cipher_value),
                            "target": chr(ord("A") + target_value),
                            "rank": rank,
                        })
        alphabet = graph.materialize(rng)
        if not alphabet or alphabet == AZ_ALPHABET or alphabet in seen:
            continue
        seen.add(alphabet)
        rows.append({
            "keyed_alphabet": alphabet,
            "selection_score": round(accepted - 0.25 * (attempted - accepted), 5),
            "metadata": {
                "period": period,
                "sample_index": sample_index,
                "accepted_constraints": accepted,
                "attempted_constraints": attempted,
                "constraint_count": len(history),
                "constraints": history,
                "generator": "offset_graph",
            },
        })
        if len(rows) >= max(1, limit):
            break

    return {
        "status": "completed" if rows else "no_candidates",
        "solver": "keyed_vigenere_offset_graph_alphabet_generator",
        "token_count": len(values),
        "skipped_symbols": skipped[:20],
        "skipped_symbol_count": len(skipped),
        "periods_tested": period_values,
        "limit": max(1, limit),
        "random_seed": random_seed,
        "samples": max(0, samples),
        "phase_count": phase_count,
        "top_cipher_letters": max(1, top_cipher_letters),
        "target_letters": max(1, target_letters),
        "target_window": max(1, target_window),
        "alphabets": rows,
        "scope_note": (
            "Experimental modular offset-graph initializer for keyed-tableau "
            "search. It samples internally consistent relative-position "
            "constraints and materializes full alphabets for downstream scoring."
        ),
    }


def generate_keyed_vigenere_constraint_graph_alphabets(
    cipher_text: CipherText,
    *,
    max_period: int = 12,
    limit: int = 1024,
    random_seed: int = 1,
    beam_size: int = 128,
    phase_count: int | None = None,
    top_cipher_letters: int = 5,
    target_letters: int = 8,
    target_window: int = 4,
    options_per_phase: int = 96,
    materializations_per_state: int = 4,
) -> dict[str, Any]:
    """Generate keyed-tableau starts by beam-searching offset constraints.

    This is a more principled companion to the random offset-graph generator.
    For each candidate period it builds phase hypotheses of the form
    `pos(cipher_letter) - pos(plain_letter) = phase_shift mod 26`, keeps only
    mutually consistent modular graphs, and materializes several full tableaux
    from the best surviving graphs.
    """
    values, skipped = cipher_values_from_text(cipher_text)
    if len(values) < 12:
        return {
            "status": "unsupported",
            "reason": "too_few_a_z_tokens",
            "token_count": len(values),
            "alphabets": [],
        }

    rng = random.Random(random_seed)
    period_values = list(range(1, min(max_period, max(1, len(values) // 4)) + 1))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for period in period_values:
        phase_scores: list[tuple[float, int]] = []
        for phase in range(period):
            stream = values[phase::period]
            phase_scores.append((_phase_ic(stream) * len(stream), phase))
        phase_scores.sort(reverse=True)
        selected_phases = [
            phase
            for _score, phase in phase_scores[: max(1, min(period, phase_count or period))]
        ]
        states: list[tuple[float, _ModOffsetGraph, list[dict[str, Any]]]] = [
            (0.0, _ModOffsetGraph(), [])
        ]
        for phase in selected_phases:
            options = _phase_graph_options(
                values,
                period=period,
                phase=phase,
                top_cipher_letters=top_cipher_letters,
                target_letters=target_letters,
                target_window=target_window,
                options_per_phase=options_per_phase,
            )
            expanded: list[tuple[float, _ModOffsetGraph, list[dict[str, Any]]]] = []
            for state_score, graph, history in states:
                for option in options:
                    trial = graph.clone()
                    accepted = 0
                    attempted = 0
                    for constraint in option["constraints"]:
                        attempted += 1
                        if trial.union(
                            int(constraint["cipher_value"]),
                            int(constraint["target_value"]),
                            int(constraint["shift"]),
                        ):
                            accepted += 1
                    if accepted == 0:
                        continue
                    rejected = attempted - accepted
                    score = state_score + option["score"] + accepted * 0.5 - rejected * 0.75
                    expanded.append((
                        score,
                        trial,
                        [
                            *history,
                            {
                                "phase": phase,
                                "shift": option["shift"],
                                "target_offset": option["target_offset"],
                                "accepted": accepted,
                                "attempted": attempted,
                                "score": round(option["score"], 5),
                            },
                        ],
                    ))
            expanded.sort(key=lambda item: item[0], reverse=True)
            deduped: dict[tuple[tuple[int, int], ...], tuple[float, _ModOffsetGraph, list[dict[str, Any]]]] = {}
            for state in expanded:
                signature = state[1].signature()
                if signature not in deduped:
                    deduped[signature] = state
                if len(deduped) >= max(1, beam_size):
                    break
            states = list(deduped.values())
            if not states:
                break

        for state_rank, (score, graph, history) in enumerate(states[: max(1, beam_size)]):
            for materialization_index in range(max(1, materializations_per_state)):
                alphabet = graph.materialize(rng)
                if not alphabet or alphabet == AZ_ALPHABET or alphabet in seen:
                    continue
                seen.add(alphabet)
                rows.append({
                    "keyed_alphabet": alphabet,
                    "selection_score": round(score - _period_complexity_penalty(period), 5),
                    "metadata": {
                        "period": period,
                        "state_rank": state_rank,
                        "materialization_index": materialization_index,
                        "constraint_count": sum(int(h["accepted"]) for h in history),
                        "phase_count": len(history),
                        "constraints": history[:24],
                        "generator": "constraint_graph_beam",
                    },
                })
                if len(rows) >= max(1, limit):
                    break
            if len(rows) >= max(1, limit):
                break
        if len(rows) >= max(1, limit):
            break

    rows.sort(key=lambda row: row["selection_score"], reverse=True)
    return {
        "status": "completed" if rows else "no_candidates",
        "solver": "keyed_vigenere_constraint_graph_alphabet_generator",
        "token_count": len(values),
        "skipped_symbols": skipped[:20],
        "skipped_symbol_count": len(skipped),
        "periods_tested": period_values,
        "limit": max(1, limit),
        "random_seed": random_seed,
        "beam_size": max(1, beam_size),
        "phase_count": phase_count,
        "top_cipher_letters": max(1, top_cipher_letters),
        "target_letters": max(1, target_letters),
        "target_window": max(1, target_window),
        "options_per_phase": max(1, options_per_phase),
        "materializations_per_state": max(1, materializations_per_state),
        "alphabets": rows[: max(1, limit)],
        "scope_note": (
            "Experimental beam search over modular phase-frequency constraints "
            "for keyed-tableau candidate generation. This is a population "
            "initializer, not a solved-key claim."
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


def _phase_constraint_options(
    values: list[int],
    *,
    period: int,
    phase: int,
    top_shifts: int,
    top_letters: int,
    target_window: int,
) -> list[dict[str, Any]]:
    stream = values[phase::period]
    if not stream:
        return []
    counts = Counter(stream)
    top_cipher = [value for value, _count in counts.most_common(top_letters)]
    shift_rows: list[tuple[float, int]] = []
    for shift in range(26):
        score = _chi2_english(_phase_plain_values(stream, shift, "vigenere"))
        shift_rows.append((score, shift))
    shift_rows.sort(key=lambda item: item[0])

    options: list[dict[str, Any]] = []
    for shift_rank, (chi2, shift) in enumerate(shift_rows[:top_shifts]):
        for target_offset in range(target_window):
            constraints: list[dict[str, Any]] = []
            support = 0
            for rank, cipher_value in enumerate(top_cipher):
                target_index = min(rank + target_offset, len(ENGLISH_FREQ_ORDER) - 1)
                target = ENGLISH_FREQ_ORDER[target_index]
                constraints.append({
                    "cipher": chr(ord("A") + cipher_value),
                    "target": target,
                    "shift": shift,
                    "rank": rank,
                    "target_rank": target_index,
                    "count": counts[cipher_value],
                })
                support += counts[cipher_value]
            # Higher is better.  Chi-square dominates, support breaks ties.
            score = -(chi2 / max(1, len(stream))) + support / max(1, len(stream))
            options.append({
                "phase": phase,
                "shift": shift,
                "shift_rank": shift_rank,
                "chi2": round(chi2, 4),
                "target_offset": target_offset,
                "score": score,
                "constraints": constraints,
            })
    options.sort(key=lambda item: item["score"], reverse=True)
    return options


class _ModOffsetGraph:
    """Union-find with modular potentials over the 26-letter alphabet."""

    def __init__(self) -> None:
        self.parent = list(range(26))
        self.rank = [0] * 26
        # weight[x] = pos(x) - pos(parent[x]) mod 26
        self.weight = [0] * 26

    def clone(self) -> "_ModOffsetGraph":
        other = _ModOffsetGraph()
        other.parent = list(self.parent)
        other.rank = list(self.rank)
        other.weight = list(self.weight)
        return other

    def signature(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.find(node) for node in range(26))

    def find(self, x: int) -> tuple[int, int]:
        parent = self.parent[x]
        if parent == x:
            return x, 0
        root, root_weight = self.find(parent)
        self.parent[x] = root
        self.weight[x] = (self.weight[x] + root_weight) % 26
        return self.parent[x], self.weight[x]

    def union(self, a: int, b: int, delta: int) -> bool:
        """Assert pos(a) - pos(b) = delta (mod 26)."""
        root_a, weight_a = self.find(a)
        root_b, weight_b = self.find(b)
        delta %= 26
        if root_a == root_b:
            return (weight_a - weight_b) % 26 == delta
        if self.rank[root_a] < self.rank[root_b]:
            # parent[root_a] = root_b; need pos(root_a)-pos(root_b)
            self.parent[root_a] = root_b
            self.weight[root_a] = (delta - weight_a + weight_b) % 26
        else:
            # parent[root_b] = root_a; need pos(root_b)-pos(root_a)
            self.parent[root_b] = root_a
            self.weight[root_b] = (weight_a - weight_b - delta) % 26
            if self.rank[root_a] == self.rank[root_b]:
                self.rank[root_a] += 1
        return True

    def materialize(self, rng: random.Random) -> str | None:
        groups: dict[int, list[tuple[int, int]]] = {}
        for node in range(26):
            root, rel = self.find(node)
            groups.setdefault(root, []).append((node, rel))

        positions: list[int | None] = [None] * 26
        occupied: set[int] = set()
        components = sorted(groups.values(), key=len, reverse=True)
        for component in components:
            if len(component) == 1:
                continue
            bases = list(range(26))
            rng.shuffle(bases)
            placed = False
            for base in bases:
                candidate_positions = [(base + rel) % 26 for _node, rel in component]
                if len(set(candidate_positions)) != len(candidate_positions):
                    continue
                if any(pos in occupied for pos in candidate_positions):
                    continue
                for (node, _rel), pos in zip(component, candidate_positions):
                    positions[pos] = node
                    occupied.add(pos)
                placed = True
                break
            if not placed:
                return None

        remaining_letters = [node for node in range(26) if node not in set(p for p in positions if p is not None)]
        remaining_positions = [pos for pos, node in enumerate(positions) if node is None]
        rng.shuffle(remaining_letters)
        for pos, node in zip(remaining_positions, remaining_letters):
            positions[pos] = node
        if any(node is None for node in positions):
            return None
        return "".join(chr(ord("A") + int(node)) for node in positions)


def _phase_graph_options(
    values: list[int],
    *,
    period: int,
    phase: int,
    top_cipher_letters: int,
    target_letters: int,
    target_window: int,
    options_per_phase: int,
) -> list[dict[str, Any]]:
    stream = values[phase::period]
    if not stream:
        return []
    counts = Counter(stream)
    top_cipher = [value for value, _count in counts.most_common(max(1, top_cipher_letters))]
    target_pool = [
        ord(ch) - ord("A")
        for ch in ENGLISH_FREQ_ORDER[: max(1, target_letters)]
    ]
    options: list[dict[str, Any]] = []
    for shift in range(26):
        for target_offset in range(max(1, target_window)):
            constraints: list[dict[str, Any]] = []
            score = 0.0
            for rank, cipher_value in enumerate(top_cipher):
                target_index = (rank + target_offset) % len(target_pool)
                target_value = target_pool[target_index]
                if cipher_value == target_value:
                    continue
                count = counts[cipher_value]
                freq_weight = ENGLISH_FREQS[target_value]
                score += count * (1.0 + freq_weight * 10.0) / (rank + 1)
                constraints.append({
                    "cipher": chr(ord("A") + cipher_value),
                    "cipher_value": cipher_value,
                    "target": chr(ord("A") + target_value),
                    "target_value": target_value,
                    "shift": shift,
                    "rank": rank,
                    "target_rank": target_index,
                    "count": count,
                })
            if constraints:
                options.append({
                    "phase": phase,
                    "shift": shift,
                    "target_offset": target_offset,
                    "score": score / max(1, len(stream)),
                    "constraints": constraints,
                })
    options.sort(key=lambda item: item["score"], reverse=True)
    return options[: max(1, options_per_phase)]


def _apply_phase_constraint_option(alphabet: str, option: dict[str, Any]) -> str:
    chars = list(alphabet)
    for constraint in option.get("constraints", []):
        cipher = constraint["cipher"]
        target = constraint["target"]
        shift = int(constraint["shift"])
        if cipher == target:
            continue
        try:
            target_pos = chars.index(target)
            desired_pos = (target_pos + shift) % 26
            current_pos = chars.index(cipher)
        except ValueError:
            continue
        chars[current_pos], chars[desired_pos] = chars[desired_pos], chars[current_pos]
    return "".join(chars)


def _alphabet_disruption(alphabet: str) -> int:
    return sum(1 for a, b in zip(alphabet, AZ_ALPHABET) if a != b)


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
    guided: bool,
    guided_pool_size: int,
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
    start_score = current_score
    best_alpha = current_alpha
    best_shifts = list(current_shifts)
    best_plain = current_plain
    best_score = current_score
    random_proposals = 0
    guided_proposals = 0
    for step in range(max(0, steps)):
        guided_choices: list[str] = []
        if guided and rng.random() < 0.70:
            guided_choices = _guided_alphabet_mutations(
                cipher_text,
                alphabet=current_alpha,
                period=period,
                shifts=current_shifts,
                rng=rng,
                limit=guided_pool_size,
            )
        if guided_choices:
            trial_alpha = rng.choice(guided_choices)
            guided_proposals += 1
        else:
            trial_alpha = _mutate_alphabet(current_alpha, rng)
            random_proposals += 1
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
        init_score=start_score,
        key=_key_string_for_alphabet(best_shifts, best_alpha),
        metadata={
            "key_type": "PeriodicAlphabetKey",
            "keyed_alphabet": best_alpha,
            "alphabet_keyword": start_info.get("alphabet_keyword"),
            "initial_keyed_alphabet": alphabet,
            "initial_candidate_type": start_info.get("candidate_type"),
            "restart": restart,
            "mutation_search": "guided_frequency_phase_swap_move_reverse" if guided else "swap_move_reverse",
            "guided": guided,
            "guided_pool_size": guided_pool_size,
            "guided_proposals": guided_proposals,
            "random_proposals": random_proposals,
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


def _guided_alphabet_mutations(
    cipher_text: CipherText,
    *,
    alphabet: str,
    period: int,
    shifts: list[int],
    rng: random.Random,
    limit: int,
) -> list[str]:
    values, _positions, _skipped = _keyed_cipher_indices(cipher_text, alphabet)
    if not values or not shifts:
        return []

    candidates: list[str] = []
    candidates.extend(_frequency_pressure_swaps(alphabet, values, shifts, max(4, limit // 2)))
    candidates.extend(_phase_common_letter_swaps(alphabet, values, period, shifts, max(4, limit)))

    deduped = [candidate for candidate in dict.fromkeys(candidates) if candidate != alphabet]
    rng.shuffle(deduped)
    return deduped[: max(1, limit)]


def _frequency_pressure_swaps(
    alphabet: str,
    values: list[int],
    shifts: list[int],
    limit: int,
) -> list[str]:
    plaintext = decode_keyed_vigenere_shifts(values, shifts, keyed_alphabet=alphabet)
    if len(plaintext) < 20:
        return []
    counts = Counter(plaintext)
    n = len(plaintext)
    over: list[tuple[float, str]] = []
    under: list[tuple[float, str]] = []
    for i, expected_freq in enumerate(ENGLISH_FREQS):
        letter = chr(ord("A") + i)
        observed = counts.get(letter, 0)
        expected = expected_freq * n
        if expected <= 0:
            continue
        delta = (observed - expected) / math.sqrt(expected)
        if delta > 0:
            over.append((delta, letter))
        else:
            under.append((-delta, letter))
    over.sort(reverse=True)
    under.sort(reverse=True)

    proposals: list[str] = []
    for _delta_over, over_letter in over[:5]:
        for _delta_under, under_letter in under[:5]:
            proposal = _swap_letters_in_alphabet(alphabet, over_letter, under_letter)
            if proposal != alphabet:
                proposals.append(proposal)
                if len(proposals) >= limit:
                    return proposals
    return proposals


def _phase_common_letter_swaps(
    alphabet: str,
    values: list[int],
    period: int,
    shifts: list[int],
    limit: int,
) -> list[str]:
    proposals: list[str] = []
    common_targets = ENGLISH_FREQ_ORDER[:8]
    for phase in range(max(1, period)):
        stream = values[phase::period]
        if not stream:
            continue
        shift = shifts[phase % len(shifts)]
        counts = Counter(stream)
        for cipher_index, _count in counts.most_common(4):
            current_letter = alphabet[(cipher_index - shift) % 26]
            for target_letter in common_targets:
                if current_letter == target_letter:
                    continue
                proposal = _swap_letters_in_alphabet(alphabet, current_letter, target_letter)
                if proposal != alphabet:
                    proposals.append(proposal)
                    if len(proposals) >= limit:
                        return proposals
    return proposals


def _swap_letters_in_alphabet(alphabet: str, a: str, b: str) -> str:
    if a == b:
        return alphabet
    chars = list(alphabet)
    try:
        i = chars.index(a)
        j = chars.index(b)
    except ValueError:
        return alphabet
    chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def _normalize_keyword_state(keyword: str, length: int | None = None) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for ch in (keyword or "").upper():
        if "A" <= ch <= "Z" and ch not in seen:
            seen.add(ch)
            out.append(ch)
            if length is not None and len(out) >= max(1, length):
                break
    return "".join(out)


def _dictionary_keyword_starts(
    language: str,
    *,
    keyword_lengths: list[int],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    path = dictionary.get_dictionary_path(language)
    if not path:
        return []
    lengths = set(max(1, int(n)) for n in keyword_lengths)
    out: list[str] = []
    seen: set[str] = set()
    for word in pattern.load_word_list(path):
        clean = _normalize_keyword_state(word, max(lengths) if lengths else None)
        if len(clean) not in lengths or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= limit:
            break
    return out


def _random_keyword_state(length: int, rng: random.Random) -> str:
    letters = list(AZ_ALPHABET)
    rng.shuffle(letters)
    return "".join(letters[: max(1, min(26, length))])


def _mutate_keyword_state(keyword: str, rng: random.Random) -> str:
    chars = list(_normalize_keyword_state(keyword, max(1, len(keyword))))
    if not chars:
        return _random_keyword_state(1, rng)
    if rng.random() < 0.20 and len(chars) >= 2:
        i, j = rng.sample(range(len(chars)), 2)
        chars[i], chars[j] = chars[j], chars[i]
    else:
        idx = rng.randrange(len(chars))
        # Blake's perturbation swaps one prefix letter with one tail letter,
        # then restores the tail to A-Z order.  In prefix-state form this is
        # exactly a replacement by a currently unused letter.
        available = [ch for ch in AZ_ALPHABET if ch not in chars]
        if not available:
            return "".join(chars)
        chars[idx] = rng.choice(available)
    return _normalize_keyword_state("".join(chars), len(chars))


def _keyword_distance(keyword: str, target: str) -> int:
    if not target:
        return 0
    a = _normalize_keyword_state(keyword, max(1, len(keyword)))
    b = _normalize_keyword_state(target, max(1, len(target)))
    shared = sum(1 for x, y in zip(a, b) if x == y)
    return max(len(a), len(b)) - shared


def _continuous_word_hit_score(text: str, word_set: set[str]) -> float:
    """Strict word-evidence score for no-boundary finalist reranking.

    The generic segmenter is intentionally generous, which is useful for
    readable repair work but too easy for Quagmire gibberish to game.  For
    finalist reranking we instead count direct dictionary substrings of length
    four or more, weighted by length.
    """
    letters = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if not letters or not word_set:
        return 0.0
    score = 0.0
    for word in word_set:
        if 4 <= len(word) <= 12 and word in letters:
            score += len(word) * len(word)
    return min(1.0, score / max(1.0, len(letters) * 4.0))


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
