"""Null/codeword candidate helpers for homophonic ciphers.

These routines are ground-truth-free. They identify symbols that are likely
nulls, abbreviation markers, or codewords rather than letter homophones,
generate candidate null masks, and rank solver-produced finalists using
readability and collapse signals. Benchmark plaintext may be used by callers
only after candidates have already been produced, for calibration/reporting.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from analysis.language_scoring import (
    LinearLanguageQualityModel,
    binary_ngram_fit_score,
    content_word_quality_score,
    function_overuse_penalty,
    language_coherence_score,
    language_fragment_score,
    language_quality_feature_dict,
    language_quality_solver_evidence_features,
    language_shape_score,
    repetitive_word_island_penalty,
    segmentation_shape_penalty,
    word_lattice_quality_score,
    word_island_template_penalty,
)
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
    rare_anchor_symbols: list[str] = []
    for rank, item in enumerate(diagnostics.get("null_codeword_candidates") or []):
        symbol = str(item.get("symbol") or "")
        if not symbol:
            continue
        reasons = set(item.get("reasons") or [])
        count = int(item.get("count") or 0)
        if count <= 2 and ({"rare", "localized", "context-specific"} & reasons):
            rare_anchor_symbols.append(symbol)
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
    score_sorted = [
        symbol
        for symbol, _score in sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    ]
    if limit <= 0:
        return []
    # A noisy first-pass key can make collapsed homophone families dominate the
    # score-sorted list. Reserve some early slots for ciphertext-only
    # rare/localized anchors so plausible null pairs remain reachable.
    rare_quota = min(max(4, limit // 2), len(rare_anchor_symbols), limit)
    final: list[str] = []
    for symbol in score_sorted[: max(2, limit // 3)]:
        if symbol not in final:
            final.append(symbol)
        if len(final) >= limit:
            return final
    added_rare = 0
    for symbol in rare_anchor_symbols:
        if symbol not in final:
            final.append(symbol)
            added_rare += 1
        if added_rare >= rare_quota:
            break
    for symbol in score_sorted:
        if symbol not in final:
            final.append(symbol)
        if len(final) >= limit:
            break
    return final


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


def null_mask_validation_score(
    row: dict[str, Any],
    original_length: int,
    *,
    language: str = "de",
) -> dict[str, Any]:
    """No-ground-truth finalist score for homophonic null-mask probes."""
    diagnostics = row.get("diagnostics") or {}
    quality = row.get("quality") or {}
    selection_score = float(row.get("selection_score") or 0.0)
    dict_rate = float(diagnostics.get("dict_rate") or 0.0)
    top_letter_fraction = float(quality.get("top_letter_fraction") or 0.0)
    unique_letters = float(quality.get("unique_letters") or diagnostics.get("unique_letters") or 0.0)
    letter_count = float(diagnostics.get("letter_count") or row.get("filtered_length") or 1)
    segmentation_cost = float(diagnostics.get("segmentation_cost") or 0.0)
    pseudo_word_fraction = float(diagnostics.get("pseudo_word_fraction") or 0.0)
    long_pseudo_word_fraction = float(diagnostics.get("long_pseudo_word_fraction") or 0.0)
    short_word_fraction = float(diagnostics.get("short_word_fraction") or 0.0)
    preview = str(row.get("decryption") or row.get("validation_text") or row.get("preview") or diagnostics.get("segmented_preview") or "")
    fragment_score = language_fragment_score(preview, language)
    deletion_fraction = max(
        0.0,
        (original_length - int(row.get("filtered_length") or original_length)) / max(1, original_length),
    )
    mask_size = len(row.get("mask") or [])

    components = {
        "selection": selection_score,
        "dictionary": dict_rate * 4.0,
        "language_fragments": fragment_score * 1.35,
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


def null_mask_validation_score_v2(
    row: dict[str, Any],
    original_length: int,
    *,
    language: str = "de",
) -> dict[str, Any]:
    """Stricter no-ground-truth finalist score for homophonic null-mask probes."""
    diagnostics = row.get("diagnostics") or {}
    quality = row.get("quality") or {}
    selection_score = float(row.get("selection_score") or 0.0)
    dict_rate = float(diagnostics.get("dict_rate") or 0.0)
    top_letter_fraction = float(quality.get("top_letter_fraction") or 0.0)
    unique_letters = float(quality.get("unique_letters") or diagnostics.get("unique_letters") or 0.0)
    letter_count = float(diagnostics.get("letter_count") or row.get("filtered_length") or 1)
    segmentation_cost = float(diagnostics.get("segmentation_cost") or 0.0)
    pseudo_word_fraction = float(diagnostics.get("pseudo_word_fraction") or 0.0)
    long_pseudo_word_fraction = float(diagnostics.get("long_pseudo_word_fraction") or 0.0)
    short_word_fraction = float(diagnostics.get("short_word_fraction") or 0.0)
    preview = str(row.get("decryption") or row.get("validation_text") or row.get("preview") or diagnostics.get("segmented_preview") or "")
    binary_ngram_mean = diagnostics.get("binary_ngram_mean_log_prob")
    deletion_fraction = max(
        0.0,
        (original_length - int(row.get("filtered_length") or original_length)) / max(1, original_length),
    )
    mask_size = len(row.get("mask") or [])

    coherence = language_coherence_score(preview, language)
    shape = language_shape_score(preview, language)
    repetition = repetitive_word_island_penalty(preview)
    template_islands = word_island_template_penalty(diagnostics, repetition=repetition)
    function_overuse = function_overuse_penalty(preview, language)
    binary_fit = binary_ngram_fit_score(binary_ngram_mean)
    lattice_quality = word_lattice_quality_score(diagnostics)
    content_quality = content_word_quality_score(diagnostics)
    components = {
        "selection": selection_score * 0.42,
        "dictionary": min(dict_rate, 0.64) * 1.65,
        "word_lattice_quality": lattice_quality * 0.95,
        "content_word_quality": content_quality * 1.15,
        "language_coherence": coherence * 1.20,
        "language_shape": shape * 0.90,
        "binary_ngram_fit": binary_fit * 0.9,
        "letter_diversity": min(unique_letters, 20.0) / 20.0 * 0.95,
        "segmentation_cost": -min(1.0, segmentation_cost / max(1.0, letter_count) / 6.0) * 0.55,
        "segmentation_shape": -segmentation_shape_penalty(
            pseudo_word_fraction=pseudo_word_fraction,
            long_pseudo_word_fraction=long_pseudo_word_fraction,
            short_word_fraction=short_word_fraction,
        ) * 0.65,
        "top_letter_penalty": -max(0.0, top_letter_fraction - 0.248) * 11.0,
        "length_penalty": -max(0.0, deletion_fraction - 0.12) * 5.0,
        "mask_penalty": -max(0, mask_size - 2) * 0.10,
        "repetition_penalty": -repetition * 1.1,
        "template_island_penalty": -template_islands * 0.45,
        "function_overuse_penalty": -function_overuse * 0.25,
    }
    score = sum(components.values())
    return {
        "score": round(score, 6),
        "components": {name: round(value, 6) for name, value in components.items()},
    }


def attach_null_mask_ensemble_scores(
    rows: list[dict[str, Any]],
    *,
    original_length: int,
    language: str = "de",
    language_quality_model: LinearLanguageQualityModel | None = None,
) -> None:
    """Attach pairwise/ensemble ranking scores to completed null-mask rows.

    The scalar v2 validator is useful, but p068-style candidate menus show that
    one fragment/coherence signal can dominate. This ranker compares candidates
    across several independent evidence families and records a vote-style
    score that is still ground-truth-free.
    """
    completed = [
        row for row in rows
        if row.get("status") in (None, "completed")
    ]
    if not completed:
        return
    if language_quality_model is not None:
        attach_null_mask_language_quality_scores(
            completed,
            original_length=original_length,
            language=language,
            model=language_quality_model,
        )
    for row in completed:
        if row.get("validation_score_v2") is None:
            validation = null_mask_validation_score_v2(
                row,
                original_length=original_length,
                language=language,
            )
            row["validation_score_v2"] = validation["score"]
            row["validation_components_v2"] = validation["components"]
        row["ensemble_features_v1"] = _null_mask_ensemble_features(row)

    wins = {id(row): 0.0 for row in completed}
    pair_count = {id(row): 0 for row in completed}
    for idx, left in enumerate(completed):
        for right in completed[idx + 1:]:
            left_points, right_points = _pairwise_null_mask_votes(
                left.get("ensemble_features_v1") or {},
                right.get("ensemble_features_v1") or {},
            )
            wins[id(left)] += left_points
            wins[id(right)] += right_points
            pair_count[id(left)] += left_points + right_points
            pair_count[id(right)] += left_points + right_points
    for row in completed:
        total = pair_count[id(row)]
        vote_rate = wins[id(row)] / total if total else 0.0
        features = row.get("ensemble_features_v1") or {}
        row["ensemble_vote_rate_v1"] = round(vote_rate, 6)
        row["ensemble_score_v1"] = round(
            vote_rate * 3.0
            + float(row.get("validation_score_v2") or 0.0) * 0.45
            + _bounded_language_quality_bonus(row) * 0.35
            + float(features.get("word_lattice_quality") or 0.0) * 0.55
            + float(features.get("damage_control") or 0.0) * 0.35,
            6,
        )
        if row.get("language_quality_raw_score") is not None:
            row["language_quality_rank_score"] = round(_language_quality_rank_score(row), 6)


def attach_null_mask_language_quality_scores(
    rows: list[dict[str, Any]],
    *,
    original_length: int,
    language: str,
    model: LinearLanguageQualityModel,
) -> None:
    """Attach trained language-quality scores to null-mask finalists.

    This consumes only solver-produced plaintext and diagnostics. The model may
    have been trained offline with solved calibration labels, but no benchmark
    plaintext is consulted here.
    """
    for row in rows:
        if row.get("status") not in (None, "completed"):
            continue
        text = str(
            row.get("decryption")
            or row.get("validation_text")
            or row.get("preview")
            or (row.get("diagnostics") or {}).get("segmented_preview")
            or ""
        )
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        features = language_quality_feature_dict(
            text,
            diagnostics=diagnostics,
            language=language,
            original_length=original_length,
            filtered_length=row.get("filtered_length"),
            mask_size=len(row.get("mask") or []),
        )
        features.update(language_quality_solver_evidence_features(row))
        row["language_quality_features"] = {
            name: round(float(value), 6)
            for name, value in features.items()
        }
        row["language_quality_raw_score"] = round(model.raw_score_features(features), 6)
        row["language_quality_score"] = round(model.score_features(features), 6)
        row["language_quality_model"] = {
            "language": model.language,
            "version": model.version,
            "feature_count": len(model.feature_names),
        }


def null_mask_rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    """Sorting key for ground-truth-free null-mask ranking."""
    return (
        float(row.get("ensemble_score_v1") or float("-inf")),
        float(row.get("validation_score_v2") or float("-inf")),
        float(row.get("selection_score") or float("-inf")),
    )


def null_mask_language_quality_rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Sorting key for trained language-quality null-mask ranking."""
    # The trained score is a noisy reading-quality vote, not a proof. Treat
    # differences below 0.01 as ties so scalar validation and ensemble evidence
    # can choose among near-identical LQ basins. The blended rank score already
    # includes scalar validation, so use a blended validation/ensemble tie-break
    # instead of counting validation alone twice.
    lq_score = float(row.get("language_quality_rank_score") or float("-inf"))
    lq_bucket = round(lq_score, 2) if lq_score != float("-inf") else lq_score
    validation = float(row.get("validation_score_v2") or float("-inf"))
    ensemble = float(row.get("ensemble_score_v1") or float("-inf"))
    if validation == float("-inf") or ensemble == float("-inf"):
        tie_break = float("-inf")
    else:
        tie_break = validation + ensemble * 0.05
    return (
        lq_bucket,
        tie_break,
        ensemble,
        validation,
        float(row.get("selection_score") or float("-inf")),
    )


def _null_mask_ensemble_features(row: dict[str, Any]) -> dict[str, float]:
    components = row.get("validation_components_v2") or {}
    diagnostics = row.get("diagnostics") or {}
    quality = row.get("quality") or {}
    damage_components = (
        components.get("top_letter_penalty"),
        components.get("repetition_penalty"),
        components.get("template_island_penalty"),
        components.get("function_overuse_penalty"),
        components.get("segmentation_shape"),
        components.get("length_penalty"),
        components.get("mask_penalty"),
    )
    damage = sum(abs(float(value or 0.0)) for value in damage_components)
    language_content = (
        min(float(components.get("language_coherence") or 0.0), 0.82)
        + float(components.get("language_shape") or 0.0)
        + float(components.get("content_word_quality") or 0.0) * 0.65
    )
    return {
        "validation_score_v2": float(row.get("validation_score_v2") or 0.0),
        "selection_score": float(row.get("selection_score") or 0.0),
        "word_lattice_quality": float(components.get("word_lattice_quality") or 0.0),
        "content_word_quality": float(components.get("content_word_quality") or 0.0),
        "language_content": language_content,
        "binary_ngram_fit": float(components.get("binary_ngram_fit") or 0.0),
        "letter_diversity": float(components.get("letter_diversity") or 0.0),
        "damage_control": max(0.0, 1.0 - damage),
        "dict_rate": float(diagnostics.get("dict_rate") or 0.0),
        "top_letter_control": max(0.0, 1.0 - float(quality.get("top_letter_fraction") or 0.0) * 2.8),
        "language_quality_raw_score": float(row.get("language_quality_raw_score") or 0.0),
        "language_quality_score": float(row.get("language_quality_score") or 0.0),
    }


def _bounded_language_quality_bonus(row: dict[str, Any]) -> float:
    if row.get("language_quality_raw_score") is None:
        return 0.0
    # Use raw score for ordering elsewhere, but cap its ensemble contribution
    # so an overconfident calibration model cannot drown out diagnostics.
    return max(0.0, min(1.0, float(row.get("language_quality_raw_score") or 0.0)))


def _language_quality_rank_score(row: dict[str, Any]) -> float:
    """Blend trained language quality with existing ground-truth-free evidence.

    The trained score is intentionally not primary: early Copiale calibration
    showed that a small model can saturate on word-island candidates. Treat it
    as a useful reading-quality vote beside scalar validation and ensemble
    diagnostics.
    """
    validation = float(row.get("validation_score_v2") or 0.0)
    ensemble = float(row.get("ensemble_score_v1") or 0.0)
    raw_quality = max(0.0, min(1.25, float(row.get("language_quality_raw_score") or 0.0)))
    return validation + raw_quality * 0.10 + ensemble * 0.04


def _pairwise_null_mask_votes(
    left: dict[str, float],
    right: dict[str, float],
) -> tuple[float, float]:
    votes = (
        ("word_lattice_quality", 1.35, 0.025),
        ("damage_control", 1.20, 0.045),
        ("binary_ngram_fit", 1.00, 0.025),
        ("letter_diversity", 0.85, 0.045),
        ("top_letter_control", 0.75, 0.045),
        ("dict_rate", 0.65, 0.025),
        # Language fragments are useful but easy to fool, so they are capped
        # and intentionally not allowed to dominate the vote.
        ("language_content", 0.60, 0.070),
        ("selection_score", 0.55, 0.080),
    )
    left_points = 0.0
    right_points = 0.0
    for name, weight, tolerance in votes:
        delta = float(left.get(name) or 0.0) - float(right.get(name) or 0.0)
        if abs(delta) <= tolerance:
            left_points += weight * 0.5
            right_points += weight * 0.5
        elif delta > 0:
            left_points += weight
        else:
            right_points += weight
    return left_points, right_points


def german_fragment_score(text: str) -> float:
    """Compatibility wrapper for German no-boundary fragment scoring."""
    return language_fragment_score(text, "de")


def german_coherence_score(text: str) -> float:
    """Compatibility wrapper for German no-boundary coherence scoring."""
    return language_coherence_score(text, "de")


def german_shape_score(text: str) -> float:
    """Compatibility wrapper for German no-boundary shape scoring."""
    return language_shape_score(text, "de")


def german_function_overuse_penalty(text: str) -> float:
    """Compatibility wrapper for German function-fragment overuse."""
    return function_overuse_penalty(text, "de")


def format_validation_components(components: dict[str, Any]) -> str:
    names = [
        "dictionary",
        "content_word_quality",
        "language_coherence",
        "language_shape",
        "binary_ngram_fit",
        "word_lattice_quality",
        "letter_diversity",
        "segmentation_cost",
        "segmentation_shape",
        "top_letter_penalty",
        "repetition_penalty",
        "template_island_penalty",
        "function_overuse_penalty",
    ]
    aliases = {
        "language_coherence": "german_coherence",
        "language_shape": "german_shape",
    }
    parts = []
    for name in names:
        value = components.get(name)
        if value is None and name in aliases:
            value = components.get(aliases[name])
        parts.append(f"{name}={float(value or 0.0):+.2f}")
    return ",".join(parts)


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
    # Keep more ciphertext-only rare/localized anchors than the production
    # mask budget will ultimately consume. p068-style pages can hide useful
    # one-off null/codeword candidates just below the first dozen, and the
    # later selector is responsible for mixing these with homophone-family
    # pressure under the configured candidate cap.
    return sorted(rows, key=lambda item: (-item["score"], item["count"], item["symbol"]))[:32]
