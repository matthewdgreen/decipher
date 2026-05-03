"""Cheap finalist validation for search candidate menus.

Fast search kernels are good at finding high-scoring candidates, but their
primary language-model score can still promote text with isolated word islands.
This module adds a second-pass, Python-side validator that is deliberately
cheap enough to run over top-N finalist menus before an LLM agent or automated
selector spends more work on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from analysis import dictionary, ngram
from analysis.segment import segment_text


@dataclass(frozen=True)
class FinalistValidationConfig:
    language: str = "en"
    min_strict_word_len: int = 4
    max_strict_word_len: int = 12


def validate_plaintext_finalist(
    text: str,
    *,
    language: str = "en",
    word_set: set[str] | None = None,
    word_list: list[str] | None = None,
    config: FinalistValidationConfig | None = None,
) -> dict[str, Any]:
    """Return cheap coherence evidence for a plaintext finalist.

    The validator is not a ground-truth scorer. It is meant to distinguish
    promising continuous plaintext from candidates whose high n-gram score is
    mostly scattered short words or accidental substrings.
    """

    cfg = config or FinalistValidationConfig(language=language)
    language = cfg.language or language
    letters = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if word_set is None:
        word_set = _load_word_set(language)
    if word_list is None:
        word_list = _load_word_list(language)
    freq_rank = {word.upper(): idx for idx, word in enumerate(word_list)} if word_list else None

    if not letters:
        return {
            "letter_count": 0,
            "validation_score": 0.0,
            "validation_label": "empty",
            "recommendation": "reject_or_review_more",
            "strict_word_hit_score": 0.0,
            "strict_word_hits": [],
        }

    quad_score = ngram.normalized_ngram_score(letters, ngram.NGRAM_CACHE.get(language, 4), n=4)
    bigram_score = ngram.normalized_ngram_score(letters, ngram.NGRAM_CACHE.get(language, 2), n=2)
    segmented = segment_text(letters, word_set, freq_rank=freq_rank)
    cost_per_char = segmented.cost / max(1, len(letters))
    pseudo_fraction = len(segmented.pseudo_words) / max(1, len(segmented.words))
    strict = _strict_continuous_word_hits(
        letters,
        word_set,
        min_len=cfg.min_strict_word_len,
        max_len=cfg.max_strict_word_len,
    )
    strict_score = strict["score"]
    segmentation_quality = max(0.0, min(1.0, 1.0 - (cost_per_char / 8.0)))
    integrity = _integrity_evidence(
        letters=letters,
        segmented_words=segmented.words,
        pseudo_words=segmented.pseudo_words,
        strict_char_coverage=strict["char_coverage"],
        strict_word_hit_score=strict_score,
    )
    validation_score = (
        1.55 * strict_score
        + 0.80 * segmented.dict_rate
        + 0.65 * segmentation_quality
        - 0.40 * pseudo_fraction
        - 0.35 * integrity["damage_score"]
    )
    label = _validation_label(validation_score, strict_score, segmented.dict_rate, pseudo_fraction)
    recommendation = _recommendation(label)
    return {
        "letter_count": len(letters),
        "quadgram_loglik_per_gram": round(quad_score, 5),
        "bigram_loglik_per_gram": round(bigram_score, 5),
        "segmentation": {
            "dict_rate": round(segmented.dict_rate, 4),
            "cost": round(segmented.cost, 3),
            "cost_per_char": round(cost_per_char, 4),
            "word_count": len(segmented.words),
            "pseudo_word_count": len(segmented.pseudo_words),
            "pseudo_word_fraction": round(pseudo_fraction, 4),
            "segmented_preview": segmented.segmented[:240],
            "pseudo_word_sample": segmented.pseudo_words[:12],
        },
        "strict_word_hit_score": round(strict_score, 4),
        "strict_word_hits": strict["hits"][:20],
        "strict_word_hit_char_coverage": round(strict["char_coverage"], 4),
        "integrity": integrity,
        "validation_score": round(validation_score, 5),
        "validation_label": label,
        "recommendation": recommendation,
        "scoring_note": (
            "Validation score is a cheap reranking signal, not ground truth. "
            "It combines strict continuous word hits, segmentation quality, "
            "dictionary coverage, and pseudo-word burden."
        ),
    }


def validation_adjustment(validation: dict[str, Any]) -> float:
    """Return a small adjustment suitable for reranking an n-gram finalist menu."""

    try:
        score = float(validation.get("validation_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    label = str(validation.get("validation_label") or "")
    integrity = validation.get("integrity") if isinstance(validation.get("integrity"), dict) else {}
    try:
        damage = float(integrity.get("damage_score") or 0.0)
    except (TypeError, ValueError):
        damage = 0.0
    bonus = 0.0
    if label == "coherent_candidate":
        bonus = 0.45
    elif label == "plausible_candidate":
        bonus = 0.2
    elif label == "weak_word_islands":
        bonus = -0.15
    elif label in {"gibberish_or_wrong_family", "empty"}:
        bonus = -0.35
    return round(0.35 * score + bonus - 0.20 * damage, 5)


def _load_word_set(language: str) -> set[str]:
    path = dictionary.get_dictionary_path(language)
    return dictionary.load_word_set(path) if path else set()


def _load_word_list(language: str) -> list[str]:
    path = dictionary.get_dictionary_path(language)
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return [line.strip().upper() for line in f if line.strip()]
    except OSError:
        return []


def _strict_continuous_word_hits(
    letters: str,
    word_set: set[str],
    *,
    min_len: int,
    max_len: int,
) -> dict[str, Any]:
    if not letters or not word_set:
        return {"score": 0.0, "hits": [], "char_coverage": 0.0}
    hits: list[dict[str, Any]] = []
    covered = [False] * len(letters)
    seen: set[tuple[str, int]] = set()
    for word in word_set:
        word = word.upper()
        if not (min_len <= len(word) <= max_len):
            continue
        start = letters.find(word)
        while start != -1:
            if (word, start) not in seen:
                seen.add((word, start))
                hits.append({"word": word, "start": start, "length": len(word)})
                for i in range(start, min(len(covered), start + len(word))):
                    covered[i] = True
            start = letters.find(word, start + 1)
    hits.sort(key=lambda item: (-int(item["length"]), int(item["start"]), str(item["word"])))
    weighted = sum(int(item["length"]) ** 2 for item in hits)
    score = min(1.0, weighted / max(1.0, len(letters) * 4.0))
    char_coverage = sum(1 for flag in covered if flag) / max(1, len(covered))
    return {
        "score": score,
        "hits": hits,
        "char_coverage": char_coverage,
    }


def _integrity_evidence(
    *,
    letters: str,
    segmented_words: list[str],
    pseudo_words: list[str],
    strict_char_coverage: float,
    strict_word_hit_score: float,
) -> dict[str, Any]:
    letter_count = max(1, len(letters))
    pseudo_chars = sum(len(word) for word in pseudo_words)
    pseudo_char_fraction = pseudo_chars / letter_count
    short_pseudo_words = [
        word for word in pseudo_words
        if 1 <= len(word) <= 3 and any(ch.isalpha() for ch in word)
    ]
    long_pseudo_words = [
        word for word in pseudo_words
        if len(word) >= 7 and any(ch.isalpha() for ch in word)
    ]
    singleton_count = sum(1 for word in pseudo_words if len(word) == 1)
    suspicious_short_fraction = len(short_pseudo_words) / max(1, len(segmented_words))
    uncovered_fraction = max(0.0, 1.0 - strict_char_coverage)
    # High strict scores prove many real substrings exist; remaining pseudo
    # fragments then become especially useful as local-damage evidence.
    scar_multiplier = 1.25 if strict_word_hit_score >= 0.8 else 1.0
    damage_score = min(
        1.0,
        scar_multiplier * (
            0.42 * pseudo_char_fraction
            + 0.24 * suspicious_short_fraction
            + 0.20 * min(1.0, singleton_count / 8.0)
            + 0.14 * max(0.0, uncovered_fraction - 0.20)
        ),
    )
    integrity_label = _integrity_label(
        damage_score=damage_score,
        pseudo_char_fraction=pseudo_char_fraction,
        suspicious_short_count=len(short_pseudo_words),
        long_pseudo_count=len(long_pseudo_words),
    )
    return {
        "integrity_label": integrity_label,
        "damage_score": round(damage_score, 4),
        "pseudo_char_fraction": round(pseudo_char_fraction, 4),
        "uncovered_char_fraction": round(uncovered_fraction, 4),
        "suspicious_short_pseudo_count": len(short_pseudo_words),
        "suspicious_short_pseudo_sample": short_pseudo_words[:12],
        "long_pseudo_count": len(long_pseudo_words),
        "long_pseudo_sample": long_pseudo_words[:8],
        "singleton_pseudo_count": singleton_count,
        "interpretation": _integrity_interpretation(integrity_label),
    }


def _integrity_label(
    *,
    damage_score: float,
    pseudo_char_fraction: float,
    suspicious_short_count: int,
    long_pseudo_count: int,
) -> str:
    if damage_score <= 0.08 and pseudo_char_fraction <= 0.06 and suspicious_short_count <= 2:
        return "clean_or_canonical"
    if damage_score <= 0.18 and pseudo_char_fraction <= 0.13 and suspicious_short_count <= 8:
        return "minor_local_damage"
    if damage_score <= 0.35:
        return "readable_with_local_scars"
    if long_pseudo_count >= 2:
        return "possible_wrong_family_or_large_order_error"
    return "damaged_word_island_basin"


def _integrity_interpretation(label: str) -> str:
    if label == "clean_or_canonical":
        return "Few visible segmentation scars; prefer this among equally readable candidates."
    if label == "minor_local_damage":
        return "Readable but has small local scars such as short pseudo-words or split fragments."
    if label == "readable_with_local_scars":
        return "Probably a readable basin, but not necessarily the clean/canonical transform."
    if label == "possible_wrong_family_or_large_order_error":
        return "Readable fragments coexist with larger unexplained pseudo-words; consider another transform family."
    return "Mostly word islands or heavily damaged text; review more finalists or broaden search."


def _validation_label(
    validation_score: float,
    strict_score: float,
    dict_rate: float,
    pseudo_fraction: float,
) -> str:
    if (
        validation_score >= 1.85
        and pseudo_fraction <= 0.25
        and (strict_score >= 0.25 or dict_rate >= 0.78)
    ):
        return "coherent_candidate"
    if (
        validation_score >= 1.25
        and pseudo_fraction <= 0.35
        and (strict_score >= 0.12 or dict_rate >= 0.55)
    ):
        return "plausible_candidate"
    if validation_score >= 0.55 or strict_score >= 0.08 or dict_rate >= 0.35:
        return "weak_word_islands"
    if pseudo_fraction >= 0.7:
        return "gibberish_or_wrong_family"
    return "weak_word_islands"


def _recommendation(label: str) -> str:
    if label == "coherent_candidate":
        return "install_or_promote"
    if label == "plausible_candidate":
        return "inspect_contextually"
    if label == "weak_word_islands":
        return "review_more_finalists_or_broaden_search"
    return "reject_or_switch_family"
