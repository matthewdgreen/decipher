"""Shared-alphabet multi-page cryptanalysis helpers.

A *page group* is a set of ciphertext pages that share one symbol alphabet and
(hypothetically) one substitution key. These helpers concatenate pages into a
combined ciphertext, project a shared key back onto each page, build a
per-symbol consensus assignment across finalist keys, and score each projected
page with runtime (ground-truth-free) language signals.

This module was extracted from ``scripts/research/copiale/
run_copiale_multipage_experiment.py`` and generalized from Copiale-specific
vocabulary to page-group/shared-alphabet vocabulary. It works for any
homophonic cipher whose pages share an alphabet, not just Copiale.

Ground-truth firewall
----------------------
Every function above the ``POST-HOC CALIBRATION`` banner is ground-truth-free:
it consumes ciphertext, keys, masks, and projected decryptions only, never the
benchmark plaintext. The single calibration helper below the banner
(``attach_page_scores``) grades projections against plaintext and must never be
called from a candidate-generation or ranking path.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.homophonic_nulls import null_mask_validation_score_v2
from analysis.language_scoring import language_quality_feature_dict
from automated.runner import (
    automated_candidate_diagnostics,
    cipher_text_from_tokens,
    load_word_list,
    plaintext_quality,
    run_homophonic_search,
)
from benchmark.scorer import score_decryption
from models.alphabet import Alphabet
from models.cipher_text import CipherText

_TOKEN_RE = re.compile(r"S\d+")

# Re-exported so downstream page-group tooling can build the runtime page
# ciphertext / solver call without importing the runner directly.
__all__ = [
    "PageBundle",
    "build_combined_cipher",
    "consensus_from_finalists",
    "consensus_summary",
    "project_pages",
    "project_page_with_sources",
    "score_page_runtime",
    "page_runtime_metrics",
    "fragment_illusion_penalty",
    "nested_mean",
    "mean",
    "stddev",
    "cipher_text_from_tokens",
    "run_homophonic_search",
    "attach_page_scores",
]


@dataclass(frozen=True)
class PageBundle:
    """One page of a shared-alphabet page group.

    ``test_id`` is the page/record identifier, ``symbols`` are the raw
    ciphertext symbols in order, and ``token_ids`` are their ids in the shared
    :class:`~models.alphabet.Alphabet`. ``plaintext`` is retained for post-hoc
    calibration only and must not be consumed by runtime scoring.
    """

    test_id: str
    canonical_transcription: str
    plaintext: str
    symbols: list[str]
    token_ids: list[int]


def build_combined_cipher(
    loader: Any,
    tests: dict[str, Any],
    test_ids: list[str],
    *,
    source: str = "multipage",
) -> tuple[CipherText, list[PageBundle]]:
    """Concatenate several pages into one shared-alphabet ciphertext.

    ``loader`` supplies ``load_test_data(test)`` returning an object with
    ``canonical_transcription`` (space-separated symbols) and ``plaintext``.
    The shared :class:`Alphabet` is the union of every page's symbols in
    first-seen order; each page's ``token_ids`` index into it. Pages are joined
    with a ``" | "`` word separator so page offsets/provenance stay recoverable.
    """
    token_re = _TOKEN_RE
    page_symbols: list[list[str]] = []
    page_plaintexts: list[str] = []
    canonical_pages: list[str] = []
    seen: set[str] = set()
    alphabet_symbols: list[str] = []
    for test_id in test_ids:
        data = loader.load_test_data(tests[test_id])
        symbols = token_re.findall(data.canonical_transcription)
        page_symbols.append(symbols)
        page_plaintexts.append(data.plaintext or "")
        canonical_pages.append(data.canonical_transcription)
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                alphabet_symbols.append(symbol)
    alphabet = Alphabet(alphabet_symbols)
    pages = [
        PageBundle(
            test_id=test_id,
            canonical_transcription=canonical_pages[index],
            plaintext=page_plaintexts[index],
            symbols=page_symbols[index],
            token_ids=[alphabet.id_for(symbol) for symbol in page_symbols[index]],
        )
        for index, test_id in enumerate(test_ids)
    ]
    raw = " | ".join(" ".join(page.symbols) for page in pages)
    return CipherText(raw=raw, alphabet=alphabet, source=source, separator=" | "), pages


def _selected_null_mask(artifact: dict[str, Any]) -> tuple[str, ...]:
    for step in reversed(artifact.get("steps") or []):
        if not isinstance(step, dict) or step.get("name") != "search_null_masks":
            continue
        selected = step.get("selected")
        if isinstance(selected, dict):
            return tuple(str(symbol) for symbol in (selected.get("mask") or []))
    return ()


def consensus_from_finalists(
    *,
    artifact: dict[str, Any],
    alphabet: Alphabet,
    top_n: int,
    min_agreement: float,
) -> dict[str, dict[str, Any]]:
    """Vote a per-symbol consensus assignment across a solver's finalist keys.

    Reads the ``search_null_masks`` step's selected key plus its
    ``top_finalists``, and for every symbol tallies the letter (or ``<null>``)
    that the finalist keys assign. A symbol is ``stable`` when its winning
    assignment reaches ``min_agreement``.
    """
    rows = []
    selected = None
    for step in reversed(artifact.get("steps") or []):
        if not isinstance(step, dict) or step.get("name") != "search_null_masks":
            continue
        selected = step.get("selected")
        if isinstance(selected, dict):
            rows.append(selected)
        rows.extend(
            row for row in (step.get("top_finalists") or [])
            if isinstance(row, dict)
        )
        break
    if not rows:
        rows = [{"key": artifact.get("key") or {}, "mask": _selected_null_mask(artifact)}]
    rows = rows[: max(1, top_n)]
    consensus: dict[str, dict[str, Any]] = {}
    for token_id in range(alphabet.size):
        symbol = alphabet.symbol_for(token_id)
        counts: dict[str, int] = {}
        for row in rows:
            mask = set(str(item) for item in (row.get("mask") or []))
            if symbol in mask:
                assignment = "<null>"
            else:
                key = row.get("key") if isinstance(row.get("key"), dict) else {}
                value = key.get(str(token_id), key.get(token_id))
                try:
                    value_int = int(value)
                except (TypeError, ValueError):
                    assignment = "?"
                else:
                    assignment = chr(ord("A") + value_int) if 0 <= value_int <= 25 else "?"
            counts[assignment] = counts.get(assignment, 0) + 1
        winner, winner_count = max(counts.items(), key=lambda item: (item[1], item[0]))
        agreement = winner_count / max(1, len(rows))
        consensus[symbol] = {
            "symbol": symbol,
            "token_id": token_id,
            "winner": winner,
            "agreement": round(agreement, 4),
            "stable": agreement >= min_agreement,
            "counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        }
    return consensus


def consensus_summary(consensus: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stable = [row for row in consensus.values() if row.get("stable")]
    stable_letters = [
        row for row in stable
        if isinstance(row.get("winner"), str)
        and len(str(row.get("winner"))) == 1
        and "A" <= str(row.get("winner")) <= "Z"
    ]
    stable_nulls = [row for row in stable if row.get("winner") == "<null>"]
    return {
        "symbol_count": len(consensus),
        "stable_symbol_count": len(stable),
        "stable_letter_count": len(stable_letters),
        "stable_null_count": len(stable_nulls),
        "disputed_symbol_count": len(consensus) - len(stable),
    }


def project_pages(
    *,
    pages: list[PageBundle],
    key: dict[int, int],
    mask: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Project a shared key/mask onto each page, returning per-page decryptions.

    Masked symbols are dropped; symbols whose key value is out of the A-Z range
    are skipped. Inverts :func:`build_combined_cipher` per page (round-trip
    identity when ``key`` maps each token id to a distinct letter).
    """
    masked = set(mask)
    rows = []
    for page in pages:
        chars: list[str] = []
        for symbol, token_id in zip(page.symbols, page.token_ids):
            if symbol in masked:
                continue
            value = key.get(token_id)
            if value is None or value < 0 or value > 25:
                continue
            chars.append(chr(ord("A") + value))
        rows.append({
            "test_id": page.test_id,
            "token_count": len(page.token_ids),
            "filtered_length": len(chars),
            "decryption": "".join(chars),
            "plaintext": page.plaintext,
        })
    return rows


def project_page_with_sources(
    *,
    page: PageBundle,
    key: dict[int, int],
    mask: tuple[str, ...],
    alphabet: Alphabet,
) -> tuple[str, list[str]]:
    """Project one page and return ``(text, sources)`` where ``sources[i]`` is
    the ciphertext symbol that produced ``text[i]`` (used for damage localization)."""
    masked = set(mask)
    chars: list[str] = []
    sources: list[str] = []
    for symbol, token_id in zip(page.symbols, page.token_ids):
        if symbol in masked:
            continue
        value = key.get(token_id)
        if value is None or value < 0 or value > 25:
            continue
        chars.append(chr(ord("A") + value))
        sources.append(alphabet.symbol_for(token_id))
    return "".join(chars), sources


def score_page_runtime(
    row: dict[str, Any],
    *,
    key: dict[int, int],
    mask: tuple[str, ...],
    language: str = "de",
) -> dict[str, Any]:
    """Score one projected page with runtime (ground-truth-free) signals.

    Consumes only the projected ``decryption`` text plus the key/mask; the
    ``plaintext`` field of ``row`` is never read. ``language`` selects the word
    list, dictionary, and binary n-gram model (``DECIPHER_NGRAM_MODEL_<LANG>``
    env override, default ``models/ngram5_<lang>.bin``).
    """
    text = row["decryption"]
    word_list = load_word_list(language)
    default_model = f"models/ngram5_{language}.bin"
    env_name = f"DECIPHER_NGRAM_MODEL_{language.upper()}"
    model_path = Path(os.environ.get(env_name, default_model))
    binary_model_path = model_path if model_path.exists() else None
    quality = plaintext_quality(text, key)
    diagnostics = automated_candidate_diagnostics(
        text,
        language=language,
        word_list=word_list,
        binary_model_path=binary_model_path,
    )
    validation_row = {
        "mask": list(mask),
        "filtered_length": row["filtered_length"],
        "selection_score": 0.0,
        "quality": quality,
        "diagnostics": diagnostics,
        "decryption": text,
        "preview": text,
    }
    validation = null_mask_validation_score_v2(
        validation_row,
        original_length=int(row["token_count"]),
        language=language,
    )
    features = language_quality_feature_dict(
        text,
        diagnostics=diagnostics,
        language=language,
        original_length=int(row["token_count"]),
        filtered_length=int(row["filtered_length"]),
        mask_size=len(mask),
    )
    return {
        "test_id": row["test_id"],
        "validation_score_v2": float(validation["score"]),
        "validation_components_v2": dict(validation["components"]),
        "language_quality_mean": mean(float(value) for value in features.values()),
        "language_quality_features": {key: float(value) for key, value in features.items()},
        "dict_rate": float(diagnostics.get("dict_rate") or 0.0),
        "diagnostics": {
            key: diagnostics.get(key)
            for key in (
                "dict_rate",
                "dictionary_content_word_fraction",
                "dictionary_content_char_fraction",
                "dictionary_long_content_word_count",
                "pseudo_word_fraction",
                "long_pseudo_word_fraction",
                "binary_ngram_mean_log_prob",
                "segmentation_cost",
            )
            if key in diagnostics
        },
    }


def page_runtime_metrics(runtime_scores: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-page runtime scores into page-group metrics.

    All inputs are ground-truth-free page runtime scores; the output is the
    aggregate signal used for cross-page ranking (robust score, balanced score,
    fragment-illusion penalty, and averaged language-quality components).
    """
    if not runtime_scores:
        return {
            "page_validation_avg": 0.0,
            "page_validation_min": 0.0,
            "page_validation_std": 0.0,
            "page_balanced_score": 0.0,
            "page_robust_score": 0.0,
            "fragment_illusion_penalty": 0.0,
            "page_language_quality_avg": 0.0,
            "page_dict_avg": 0.0,
            "page_content_char_avg": 0.0,
            "page_pseudo_word_avg": 0.0,
            "page_binary_component_avg": 0.0,
            "page_shape_component_avg": 0.0,
            "page_evidence_dispersion_avg": 0.0,
            "page_window_stability_avg": 0.0,
            "page_repetition_control_avg": 0.0,
            "page_content_word_quality_avg": 0.0,
            "page_language_coherence_avg": 0.0,
        }
    avg_validation = mean(item["validation_score_v2"] for item in runtime_scores)
    min_validation = min(item["validation_score_v2"] for item in runtime_scores)
    std_validation = stddev(item["validation_score_v2"] for item in runtime_scores)
    page_balanced_score = avg_validation + 0.20 * min_validation - 0.15 * std_validation
    avg_lq = mean(item["language_quality_mean"] for item in runtime_scores)
    avg_dict = mean(item["dict_rate"] for item in runtime_scores)
    avg_content_chars = nested_mean(runtime_scores, "diagnostics", "dictionary_content_char_fraction")
    avg_pseudo = nested_mean(runtime_scores, "diagnostics", "pseudo_word_fraction")
    avg_binary_component = nested_mean(runtime_scores, "validation_components_v2", "binary_ngram_fit")
    avg_shape = nested_mean(runtime_scores, "validation_components_v2", "language_shape")
    avg_coherence = nested_mean(runtime_scores, "validation_components_v2", "language_coherence")
    avg_content_quality = nested_mean(runtime_scores, "validation_components_v2", "content_word_quality")
    avg_dispersion = nested_mean(runtime_scores, "language_quality_features", "language_evidence_dispersion")
    avg_stability = nested_mean(runtime_scores, "language_quality_features", "language_window_stability")
    avg_repetition = nested_mean(runtime_scores, "language_quality_features", "repetition_control")
    fragment_illusion = fragment_illusion_penalty(
        content_word_quality=avg_content_quality,
        language_coherence=avg_coherence,
        language_shape=avg_shape,
        binary_ngram_fit=avg_binary_component,
        evidence_dispersion=avg_dispersion,
        window_stability=avg_stability,
        repetition_control=avg_repetition,
    )
    page_robust_score = (
        min_validation
        + 0.35 * avg_binary_component
        + 0.25 * avg_dispersion
        + 0.20 * avg_stability
        + 0.15 * avg_repetition
        - 0.15 * std_validation
        - 0.75 * fragment_illusion
    )
    return {
        "page_validation_avg": round(avg_validation, 6),
        "page_validation_min": round(min_validation, 6),
        "page_validation_std": round(std_validation, 6),
        "page_balanced_score": round(page_balanced_score, 6),
        "page_robust_score": round(page_robust_score, 6),
        "fragment_illusion_penalty": round(fragment_illusion, 6),
        "page_language_quality_avg": round(avg_lq, 6),
        "page_dict_avg": round(avg_dict, 6),
        "page_content_char_avg": round(avg_content_chars, 6),
        "page_pseudo_word_avg": round(avg_pseudo, 6),
        "page_binary_component_avg": round(avg_binary_component, 6),
        "page_shape_component_avg": round(avg_shape, 6),
        "page_evidence_dispersion_avg": round(avg_dispersion, 6),
        "page_window_stability_avg": round(avg_stability, 6),
        "page_repetition_control_avg": round(avg_repetition, 6),
        "page_content_word_quality_avg": round(avg_content_quality, 6),
        "page_language_coherence_avg": round(avg_coherence, 6),
    }


def nested_mean(rows: list[dict[str, Any]], nested_key: str, feature: str) -> float:
    return mean(
        float((row.get(nested_key) or {}).get(feature) or 0.0)
        for row in rows
    )


def fragment_illusion_penalty(
    *,
    content_word_quality: float,
    language_coherence: float,
    language_shape: float,
    binary_ngram_fit: float,
    evidence_dispersion: float,
    window_stability: float,
    repetition_control: float,
) -> float:
    """Penalize fragment-rich basins unsupported by broader evidence.

    This is ground-truth-free. It targets false positives where word islands
    and smooth letter shape outrun the slower, more global signals: binary
    n-grams, evidence dispersion, window stability, and repetition control.
    """
    fragment_side = mean([content_word_quality, language_coherence, language_shape])
    support_side = mean([binary_ngram_fit, evidence_dispersion, window_stability, repetition_control])
    return max(0.0, min(1.0, fragment_side - support_side))


def mean(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / max(1, len(items))


def stddev(values: Any) -> float:
    items = [float(value) for value in values]
    if len(items) < 2:
        return 0.0
    avg = mean(items)
    return (sum((value - avg) ** 2 for value in items) / len(items)) ** 0.5


# ===========================================================================
# POST-HOC CALIBRATION -- GROUND TRUTH REQUIRED
#
# Everything below this banner reads benchmark plaintext to *grade* a
# projection after it has already been produced. It must never be called from a
# candidate-generation or ranking path: doing so would leak ground truth into
# the solve. The runtime functions above do not import or call it.
# ===========================================================================


def attach_page_scores(rows: list[dict[str, Any]]) -> None:
    """POST-HOC ONLY. Grade projected pages against benchmark plaintext.

    Mutates each row in place with ``char_accuracy`` / ``word_accuracy`` /
    ``preview``. Reads ``row["plaintext"]`` (ground truth) and therefore must
    only run for post-hoc calibration/reporting, never during generation or
    ranking.
    """
    for row in rows:
        score = score_decryption(
            row["test_id"],
            row["decryption"],
            row["plaintext"],
            agent_score=0.0,
            status=str(row.get("status") or "completed"),
        )
        row["char_accuracy"] = score.char_accuracy
        row["word_accuracy"] = score.word_accuracy
        row["preview"] = row["decryption"][:180]
