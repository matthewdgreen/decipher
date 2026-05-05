"""Copiale null/codeword candidate helpers.

These routines are ground-truth-free. They generate candidate null masks and
rank solver-produced finalists using German-ish readability and collapse
signals. Benchmark plaintext may be used by callers only after candidates have
already been produced, for calibration/reporting.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from models.cipher_text import CipherText


def diagnose_cipher_for_null_candidates(
    cipher_text: CipherText,
    *,
    key: dict[int, int] | None = None,
    id_to_letter: dict[int, str] | None = None,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return null/codeword diagnostics from ciphertext plus optional solver key."""
    symbols = [cipher_text.alphabet.decode([token]) for token in cipher_text.tokens]
    symbol_counts = Counter(symbols)
    solver_mapping = _solver_mapping(cipher_text, key or {}, id_to_letter or {})
    null_candidates = _null_candidates(
        symbols=symbols,
        words=cipher_text.words,
        alphabet=cipher_text.alphabet,
        symbol_counts=symbol_counts,
        solver_mapping=solver_mapping,
        quality=quality or {},
    )
    return {
        "token_count": len(symbols),
        "word_count": len(cipher_text.words),
        "unique_symbols": len(symbol_counts),
        "top_symbols": [
            {
                "symbol": symbol,
                "count": count,
                "frequency": count / len(symbols) if symbols else 0.0,
            }
            for symbol, count in symbol_counts.most_common(10)
        ],
        "homophone_families": _homophone_families(symbol_counts, solver_mapping),
        "null_codeword_candidates": null_candidates,
    }


def select_null_candidate_symbols(
    diagnostics: dict[str, Any],
    *,
    limit: int,
) -> list[str]:
    """Intermix rare/localized candidates with collapsed homophone families."""
    candidate_scores: dict[str, float] = {}
    for rank, item in enumerate(diagnostics.get("null_codeword_candidates") or []):
        symbol = str(item.get("symbol") or "")
        if not symbol:
            continue
        candidate_scores[symbol] = max(
            candidate_scores.get(symbol, 0.0),
            float(item.get("score") or 0.0) + max(0.0, 0.18 - rank * 0.012),
        )
    token_count = max(1, int(diagnostics.get("token_count") or 1))
    for family_rank, family in enumerate((diagnostics.get("homophone_families") or [])[:6]):
        family_token_count = max(1, int(family.get("token_count") or 1))
        for symbol_rank, symbol in enumerate(family.get("symbols") or []):
            if symbol_rank >= 5:
                break
            symbol = str(symbol)
            pressure = family_token_count / token_count
            score = 0.54 + pressure + max(0.0, 0.10 - family_rank * 0.015) - symbol_rank * 0.025
            candidate_scores[symbol] = max(candidate_scores.get(symbol, 0.0), score)
    return [
        symbol
        for symbol, _score in sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    ][:limit]


def generate_null_masks(candidates: list[str], max_mask_size: int) -> list[tuple[str, ...]]:
    """Generate bounded, deterministic null-mask candidates."""
    masks: list[tuple[str, ...]] = [()]
    if max_mask_size <= 0:
        return masks
    masks.extend((symbol,) for symbol in candidates)
    if max_mask_size >= 2:
        # Focus size-2 masks on nearby high-priority symbols rather than all
        # pairs; this keeps the production profile cheap and predictable.
        for idx, left in enumerate(candidates):
            for right in candidates[idx + 1: idx + 5]:
                masks.append((left, right))
    if max_mask_size >= 3:
        for idx, first in enumerate(candidates[:8]):
            for second in candidates[idx + 1: idx + 4]:
                for third in candidates[idx + 4: idx + 7]:
                    masks.append((first, second, third))
    return masks


def null_mask_validation_score(row: dict[str, Any], original_length: int) -> dict[str, Any]:
    """No-ground-truth finalist score for Copiale null-mask probes."""
    diagnostics = row.get("diagnostics") or {}
    quality = row.get("quality") or {}
    selection_score = float(row.get("selection_score") or 0.0)
    dict_rate = float(diagnostics.get("dict_rate") or 0.0)
    top_letter_fraction = float(quality.get("top_letter_fraction") or 0.0)
    unique_letters = float(quality.get("unique_letters") or diagnostics.get("unique_letters") or 0.0)
    letter_count = float(diagnostics.get("letter_count") or row.get("filtered_length") or 1)
    segmentation_cost = float(diagnostics.get("segmentation_cost") or 0.0)
    preview = str(row.get("preview") or diagnostics.get("segmented_preview") or "")
    fragment_score = german_fragment_score(preview)
    deletion_fraction = max(
        0.0,
        (original_length - int(row.get("filtered_length") or original_length)) / max(1, original_length),
    )
    mask_size = len(row.get("mask") or [])

    components = {
        "selection": selection_score,
        "dictionary": dict_rate * 4.0,
        "german_fragments": fragment_score * 1.35,
        "letter_diversity": min(unique_letters, 20.0) / 20.0 * 0.75,
        "segmentation_cost": -min(1.0, segmentation_cost / max(1.0, letter_count) / 6.0) * 0.45,
        "top_letter_penalty": -max(0.0, top_letter_fraction - 0.265) * 9.0,
        "length_penalty": -max(0.0, deletion_fraction - 0.12) * 4.0,
        "mask_penalty": -max(0, mask_size - 2) * 0.08,
    }
    score = sum(components.values())
    return {
        "score": round(score, 6),
        "components": {name: round(value, 6) for name, value in components.items()},
    }


def null_mask_validation_score_v2(row: dict[str, Any], original_length: int) -> dict[str, Any]:
    """Stricter no-ground-truth finalist score for Copiale null-mask probes."""
    diagnostics = row.get("diagnostics") or {}
    quality = row.get("quality") or {}
    selection_score = float(row.get("selection_score") or 0.0)
    dict_rate = float(diagnostics.get("dict_rate") or 0.0)
    top_letter_fraction = float(quality.get("top_letter_fraction") or 0.0)
    unique_letters = float(quality.get("unique_letters") or diagnostics.get("unique_letters") or 0.0)
    letter_count = float(diagnostics.get("letter_count") or row.get("filtered_length") or 1)
    segmentation_cost = float(diagnostics.get("segmentation_cost") or 0.0)
    preview = str(row.get("preview") or diagnostics.get("segmented_preview") or "")
    deletion_fraction = max(
        0.0,
        (original_length - int(row.get("filtered_length") or original_length)) / max(1, original_length),
    )
    mask_size = len(row.get("mask") or [])

    coherence = german_coherence_score(preview)
    repetition = repetitive_word_island_penalty(preview)
    components = {
        "selection": selection_score * 0.45,
        "dictionary": min(dict_rate, 0.64) * 2.35,
        "german_coherence": coherence * 1.65,
        "letter_diversity": min(unique_letters, 20.0) / 20.0 * 0.95,
        "segmentation_cost": -min(1.0, segmentation_cost / max(1.0, letter_count) / 6.0) * 0.55,
        "top_letter_penalty": -max(0.0, top_letter_fraction - 0.248) * 11.0,
        "length_penalty": -max(0.0, deletion_fraction - 0.12) * 5.0,
        "mask_penalty": -max(0, mask_size - 2) * 0.10,
        "repetition_penalty": -repetition * 1.1,
    }
    score = sum(components.values())
    return {
        "score": round(score, 6),
        "components": {name: round(value, 6) for name, value in components.items()},
    }


def german_fragment_score(text: str) -> float:
    """Cheap German readability signal for no-boundary finalist ranking."""
    cleaned = _az(text)
    if len(cleaned) < 20:
        return 0.0
    weighted_fragments = {
        "DER": 1.0,
        "DIE": 1.0,
        "DAS": 1.0,
        "UND": 1.15,
        "DEN": 0.9,
        "DEM": 0.9,
        "DES": 0.9,
        "EIN": 1.0,
        "EINE": 1.2,
        "EINER": 1.25,
        "IST": 0.9,
        "SICH": 1.2,
        "NICHT": 1.35,
        "MIT": 0.8,
        "ICH": 0.7,
        "SCH": 0.65,
        "CHT": 0.8,
        "UNG": 0.75,
        "EIT": 0.65,
        "ARBEIT": 1.5,
    }
    score = sum(cleaned.count(fragment) * weight for fragment, weight in weighted_fragments.items())
    expected_slots = max(1.0, len(cleaned) / 22.0)
    return min(1.0, score / expected_slots)


def german_coherence_score(text: str) -> float:
    """Less saturating German signal than raw fragment counts."""
    cleaned = _az(text)
    if len(cleaned) < 20:
        return 0.0
    long_fragments = {
        "WENIG": 1.2,
        "SICH": 1.0,
        "DASS": 1.0,
        "EINE": 0.9,
        "EINER": 1.1,
        "SEINE": 1.0,
        "SEINER": 1.1,
        "ARBEIT": 1.35,
        "BEWEG": 1.2,
        "GEORD": 1.2,
        "ANFANG": 1.2,
        "HEIM": 1.1,
        "BRUDER": 1.2,
        "ORDEN": 1.2,
        "NICHT": 1.0,
    }
    function_fragments = {
        "UND": 0.35,
        "DER": 0.30,
        "DIE": 0.30,
        "DAS": 0.30,
        "DEN": 0.25,
        "DES": 0.25,
        "EIN": 0.22,
    }
    score = 0.0
    distinct = 0
    for fragment, weight in long_fragments.items():
        count = cleaned.count(fragment)
        if count:
            distinct += 1
            score += min(2, count) * weight
    function_score = 0.0
    for fragment, weight in function_fragments.items():
        count = cleaned.count(fragment)
        if count:
            distinct += 1
            function_score += min(4, count) * weight
    score += min(function_score, 2.4)
    distinct_bonus = min(1.0, distinct / 8.0)
    expected_slots = max(1.0, len(cleaned) / 95.0)
    return min(1.0, (score / expected_slots) / 5.0 + distinct_bonus * 0.25)


def repetitive_word_island_penalty(text: str) -> float:
    """Penalty for repetitive German-ish islands that lack broader coherence."""
    cleaned = _az(text)
    if len(cleaned) < 24:
        return 0.0
    penalty = 0.0
    for n in (3, 4, 5):
        counts: dict[str, int] = {}
        for idx in range(0, max(0, len(cleaned) - n + 1)):
            gram = cleaned[idx: idx + n]
            counts[gram] = counts.get(gram, 0) + 1
        repeated = sum(max(0, count - 2) for count in counts.values())
        penalty += repeated / max(1.0, len(cleaned) / (n * 2.0))
    return min(1.0, penalty / 3.0)


def format_validation_components(components: dict[str, Any]) -> str:
    names = [
        "dictionary",
        "german_coherence",
        "letter_diversity",
        "segmentation_cost",
        "top_letter_penalty",
        "repetition_penalty",
    ]
    return ",".join(f"{name}={float(components.get(name) or 0.0):+.2f}" for name in names)


def _solver_mapping(
    cipher_text: CipherText,
    key: dict[int, int],
    id_to_letter: dict[int, str],
) -> dict[str, str]:
    mapping = {}
    for token in range(len(cipher_text.alphabet.symbols)):
        if token not in key:
            continue
        letter = id_to_letter.get(int(key[token]), "")
        if letter:
            mapping[cipher_text.alphabet.decode([token])] = letter
    return mapping


def _homophone_families(
    symbol_counts: Counter[str],
    solver_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    if not solver_mapping:
        return []
    families: dict[str, list[str]] = {}
    for symbol, letter in solver_mapping.items():
        families.setdefault(letter, []).append(symbol)
    rows = []
    for letter, symbols in families.items():
        symbols_sorted = sorted(symbols, key=lambda item: (-symbol_counts[item], item))
        rows.append({
            "letter": letter,
            "symbol_count": len(symbols_sorted),
            "token_count": sum(symbol_counts[item] for item in symbols_sorted),
            "symbols": symbols_sorted[:8],
        })
    return sorted(rows, key=lambda item: (-item["token_count"], -item["symbol_count"], item["letter"]))


def _null_candidates(
    *,
    symbols: list[str],
    words: list[list[int]],
    alphabet: Any,
    symbol_counts: Counter[str],
    solver_mapping: dict[str, str],
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    if not symbols:
        return []
    word_index_by_pos = []
    for word_index, word in enumerate(words):
        word_index_by_pos.extend([word_index] * len(word))

    left_neighbors: dict[str, set[str]] = {symbol: set() for symbol in symbol_counts}
    right_neighbors: dict[str, set[str]] = {symbol: set() for symbol in symbol_counts}
    word_spread: dict[str, set[int]] = {symbol: set() for symbol in symbol_counts}
    for idx, symbol in enumerate(symbols):
        if idx > 0:
            left_neighbors[symbol].add(symbols[idx - 1])
        if idx + 1 < len(symbols):
            right_neighbors[symbol].add(symbols[idx + 1])
        if idx < len(word_index_by_pos):
            word_spread[symbol].add(word_index_by_pos[idx])

    mapped_letter_counts: Counter[str] = Counter()
    for symbol, count in symbol_counts.items():
        letter = solver_mapping.get(symbol)
        if letter:
            mapped_letter_counts[letter] += count
    top_letter, top_letter_count = ("", 0)
    if mapped_letter_counts:
        top_letter, top_letter_count = mapped_letter_counts.most_common(1)[0]
    top_letter_fraction = top_letter_count / len(symbols) if symbols else 0.0
    collapsed = bool(quality.get("collapsed")) or top_letter_fraction >= 0.18

    rows = []
    for symbol, count in symbol_counts.items():
        frequency = count / len(symbols)
        rare_score = max(0.0, min(1.0, (8 - count) / 7.0))
        spread = len(word_spread[symbol])
        localization_score = 1.0 - min(1.0, spread / max(1, len(words)))
        neighbor_slots = max(1, min(count, 12) * 2)
        neighbor_diversity = (len(left_neighbors[symbol]) + len(right_neighbors[symbol])) / neighbor_slots
        context_specific_score = 1.0 - min(1.0, neighbor_diversity)
        mapped_letter = solver_mapping.get(symbol, "")
        collapse_score = 1.0 if collapsed and mapped_letter and mapped_letter == top_letter else 0.0
        score = (
            0.38 * rare_score
            + 0.25 * localization_score
            + 0.22 * context_specific_score
            + 0.15 * collapse_score
        )
        reasons = []
        if count <= 3:
            reasons.append("rare")
        if localization_score >= 0.65:
            reasons.append("localized")
        if context_specific_score >= 0.55:
            reasons.append("context-specific")
        if collapse_score:
            reasons.append(f"maps-to-collapsed-{top_letter}")
        if count > 12 and not collapse_score:
            score *= 0.55
            reasons.append("frequent-symbol-lower-priority")
        if score < 0.35 and count > 4:
            continue
        rows.append({
            "symbol": symbol,
            "count": count,
            "frequency": frequency,
            "mapped_letter": mapped_letter,
            "score": round(score, 3),
            "word_spread": spread,
            "left_neighbors": len(left_neighbors[symbol]),
            "right_neighbors": len(right_neighbors[symbol]),
            "reasons": reasons or ["weak-candidate"],
        })
    return sorted(rows, key=lambda item: (-item["score"], item["count"], item["symbol"]))[:12]


def _az(text: str) -> str:
    return "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
