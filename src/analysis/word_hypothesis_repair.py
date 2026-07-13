"""Word-hypothesis repair pipeline for shared-alphabet page groups.

This module extracts the ground-truth-free candidate generator that produced
the strongest Copiale multi-page basins and generalizes it to any homophonic
cipher whose pages share one alphabet and one substitution key. The pipeline
has five stages:

1. **Damaged-window detection** -- locate low-quality spans of a projected page
   (:func:`damaged_windows_for_text`, :func:`window_damage_score`).
2. **Hypothesis proposal** -- for each damaged span, propose same-length
   dictionary words within an edit budget (:func:`generate_word_hypotheses`).
3. **Edit-set conversion** -- turn a word hypothesis into a global symbol edit
   set on the shared key (:func:`implied_edits`, :func:`combined_edit_map`,
   :func:`apply_assignment`).
4. **Cross-page rescoring** -- apply the edit set and rescore every page with
   :func:`analysis.multipage.score_page_runtime`.
5. **Collateral adjudication** -- occurrence-level word-island checks that
   accept or reject a hypothesis by weighing target-word gain against
   collateral damage on other pages (:func:`adjudicate_repair`,
   :func:`annotate_acceptance`).

The public entry point is :func:`propose_word_repairs`, which runs all five
stages and returns :class:`~analysis.candidate_packet.CandidatePacket`
instances (``kind="word_repair"``). It is language-agnostic: the dictionary
path and language scoring profile are parameters; nothing German-specific
appears outside default arguments.

Ground-truth firewall
----------------------
Every function here is ground-truth-free. Generation, scoring, and
adjudication consume only ciphertext, keys, masks, projected decryptions, and
the supplied dictionary -- never benchmark plaintext. Word-repair packets set
``text=None`` and carry short previews only (Phase-1 review F3 deferral).
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.candidate_packet import CandidatePacket, packet_from_word_repair_row
from analysis.language_scoring import language_quality_feature_dict
from analysis.multipage import (
    PageBundle,
    page_runtime_metrics,
    project_page_with_sources,
    project_pages,
    score_page_runtime,
)
from models.alphabet import Alphabet


# ---------------------------------------------------------------------------
# Datamodel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WordHypothesis:
    """One same-length dictionary-word repair proposal for a damaged span."""

    test_id: str
    window_start: int
    start: int
    end: int
    observed: str
    target: str
    edits: tuple[tuple[str, str], ...]
    distance: int
    dictionary_rank: int
    local_score: float


@dataclass(frozen=True)
class WordEvidenceIndex:
    """Wildcard-indexed dictionary used for occurrence-level word-island checks."""

    dictionary: dict[int, list[tuple[str, int]]]
    patterns: dict[int, dict[str, list[tuple[str, int]]]]


@dataclass(frozen=True)
class WordRepairConfig:
    """Knobs for :func:`propose_word_repairs`.

    Defaults mirror the ``probe_copiale_word_hypothesis_repair.py`` CLI
    defaults (window 120 / step 40, min-len 5, max-len 14, max-edits 4,
    160 hypotheses / 16 per window, singleton hypothesis sets).
    """

    window_size: int = 120
    window_step: int = 40
    windows_per_page: int = 5
    min_word_len: int = 5
    max_word_len: int = 14
    max_edits: int = 4
    max_hypotheses: int = 160
    max_hypotheses_per_window: int = 16
    allow_stable_edits: bool = False
    max_hypothesis_set_size: int = 1
    combination_candidate_limit: int = 32
    max_combinations: int = 800
    max_combined_edits: int = 6
    acceptance_margin: float = 0.03
    min_page_drop: float = 0.02
    max_illusion_increase: float = 0.02
    allow_pair_acceptance: bool = True


# ---------------------------------------------------------------------------
# Alphabet helper
# ---------------------------------------------------------------------------


def alphabet_from_pages(pages: list[PageBundle]) -> Alphabet:
    """Build the shared :class:`Alphabet` from a page group's symbols."""
    symbols: list[str] = []
    seen: set[str] = set()
    for page in pages:
        for symbol in page.symbols:
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    return Alphabet(symbols)


# ---------------------------------------------------------------------------
# Stage 1 -- damaged-window detection
# ---------------------------------------------------------------------------


def window_damage_score(features: dict[str, float]) -> float:
    """Map language-quality features to a 0..1 damage score (1 == worst)."""
    good = (
        0.28 * float(features.get("language_coherence") or 0.0)
        + 0.22 * float(features.get("language_shape") or 0.0)
        + 0.15 * float(features.get("language_evidence_dispersion") or 0.0)
        + 0.12 * float(features.get("function_content_balance") or 0.0)
        + 0.10 * float(features.get("repetition_control") or 0.0)
        + 0.08 * float(features.get("function_overuse_control") or 0.0)
        + 0.05 * float(features.get("short_fragment_control") or 0.0)
    )
    return max(0.0, min(1.0, 1.0 - good))


def window_damage_for_text(text: str, *, language: str = "de") -> float:
    if not text:
        return 0.0
    return window_damage_score(language_quality_feature_dict(text, language=language))


def damaged_windows_for_text(
    *,
    text: str,
    sources: list[str],
    consensus: dict[str, dict[str, Any]],
    window_size: int,
    step: int,
    limit: int,
    language: str = "de",
) -> list[dict[str, Any]]:
    """Return the most-damaged windows of a projected page, worst first.

    Each window carries its damage score plus the disputed (non-stable per
    ``consensus``) symbols that produced it. ``consensus`` may be empty, in
    which case no symbol is treated as stable.
    """
    if not text:
        return []
    size = max(20, int(window_size))
    stride = max(1, int(step))
    if len(text) <= size:
        starts = [0]
    else:
        starts = list(range(0, max(1, len(text) - size + 1), stride))
        if starts[-1] != len(text) - size:
            starts.append(len(text) - size)
    windows = []
    for start in starts:
        end = min(len(text), start + size)
        snippet = text[start:end]
        features = language_quality_feature_dict(snippet, language=language)
        symbol_counts = Counter(sources[start:end])
        disputed = []
        for symbol, count in symbol_counts.most_common():
            info = consensus.get(symbol) or {}
            if info.get("stable"):
                continue
            disputed.append({
                "symbol": symbol,
                "count": count,
                "winner": info.get("winner"),
                "agreement": info.get("agreement"),
                "assignments": info.get("counts") or {},
            })
        windows.append({
            "start": start,
            "end": end,
            "damage_score": round(window_damage_score(features), 6),
            "disputed_symbol_count": len(disputed),
            "disputed_symbols": disputed[:10],
            "text": snippet,
        })
    return sorted(
        windows,
        key=lambda row: (
            float(row["damage_score"]),
            int(row["disputed_symbol_count"]),
        ),
        reverse=True,
    )[: max(0, int(limit))]


def build_page_windows(
    *,
    pages: list[PageBundle],
    alphabet: Alphabet,
    key: dict[int, int],
    mask: tuple[str, ...],
    consensus: dict[str, dict[str, Any]],
    window_size: int,
    window_step: int,
    windows_per_page: int,
    language: str = "de",
) -> list[dict[str, Any]]:
    """Project every page and detect its most-damaged windows."""
    rows = []
    for page in pages:
        text, sources = project_page_with_sources(page=page, key=key, mask=mask, alphabet=alphabet)
        windows = damaged_windows_for_text(
            text=text,
            sources=sources,
            consensus=consensus,
            window_size=window_size,
            step=window_step,
            limit=windows_per_page,
            language=language,
        )
        rows.append({
            "test_id": page.test_id,
            "text": text,
            "sources": sources,
            "windows": windows,
        })
    return rows


# ---------------------------------------------------------------------------
# Stage 2 -- hypothesis proposal
# ---------------------------------------------------------------------------


def normalize_word(value: str) -> str:
    """Uppercase and fold German umlauts (a no-op for other languages)."""
    value = value.strip().upper()
    return (
        value.replace("Ä", "AE")
        .replace("Ö", "OE")
        .replace("Ü", "UE")
        .replace("ß", "SS")
    )


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(1 for a, b in zip(left, right) if a != b)


def load_dictionary(path: Path, min_len: int, max_len: int) -> dict[int, list[tuple[str, int]]]:
    """Load a frequency-ordered word list bucketed by length (rank == line no.)."""
    words_by_len: dict[int, list[tuple[str, int]]] = {}
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        word = normalize_word(line)
        if min_len <= len(word) <= max_len and word.isalpha():
            words_by_len.setdefault(len(word), []).append((word, index))
    return words_by_len


def current_assignment(symbol: str, token_id: int, key: dict[int, int], mask: tuple[str, ...]) -> str:
    """Return the current plaintext letter (or ``<null>`` / ``?``) for a symbol."""
    if symbol in set(mask):
        return "<null>"
    value = key.get(token_id)
    if value is None or value < 0 or value > 25:
        return "?"
    return chr(ord("A") + value)


def apply_assignment(
    symbol: str,
    token_id: int,
    assignment: str,
    key: dict[int, int],
    mask: set[str],
) -> None:
    """Apply one symbol assignment in place (``<null>`` masks the symbol)."""
    if assignment == "<null>":
        mask.add(symbol)
        return
    mask.discard(symbol)
    if len(assignment) == 1 and "A" <= assignment <= "Z":
        key[token_id] = ord(assignment) - ord("A")


def parse_key(value: Any) -> dict[int, int]:
    """Coerce a serialized key (str/int keyed) into ``{token_id: letter_id}``."""
    if not isinstance(value, dict):
        return {}
    parsed = {}
    for key, item in value.items():
        try:
            parsed[int(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return parsed


def implied_edits(
    *,
    observed: str,
    target: str,
    sources: list[str],
    consensus: dict[str, dict[str, Any]],
    alphabet: Alphabet,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    allow_stable_edits: bool,
) -> tuple[tuple[str, str], ...]:
    """Convert an ``observed -> target`` word repair into per-symbol edits.

    Returns ``()`` if the repair would require a stable-symbol edit (unless
    allowed), touch a masked symbol, or assign one symbol two different letters.
    """
    symbol_targets: dict[str, str] = {}
    masked = set(baseline_mask)
    for observed_char, target_char, symbol in zip(observed, target, sources):
        if observed_char == target_char:
            continue
        if not symbol or not alphabet.has_symbol(symbol):
            return ()
        info = consensus.get(symbol) or {}
        if info.get("stable") and not allow_stable_edits:
            return ()
        if symbol in masked:
            return ()
        prior = symbol_targets.get(symbol)
        if prior is not None and prior != target_char:
            return ()
        token_id = alphabet.id_for(symbol)
        current = current_assignment(symbol, token_id, baseline_key, baseline_mask)
        if current == target_char:
            continue
        symbol_targets[symbol] = target_char
    edits = tuple(sorted(symbol_targets.items()))
    return edits


def generate_word_hypotheses(
    *,
    page_windows: list[dict[str, Any]],
    dictionary: dict[int, list[tuple[str, int]]],
    consensus: dict[str, dict[str, Any]],
    alphabet: Alphabet,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    min_word_len: int,
    max_word_len: int,
    max_edits: int,
    max_per_window: int,
    allow_stable_edits: bool,
) -> list[WordHypothesis]:
    """Propose same-length dictionary-word repairs for every damaged window."""
    hypotheses: list[WordHypothesis] = []
    seen: set[tuple[str, int, int, str, tuple[tuple[str, str], ...]]] = set()
    for page in page_windows:
        text = str(page["text"])
        sources = list(page["sources"])
        for window in page["windows"]:
            window_start = int(window.get("start") or 0)
            window_end = int(window.get("end") or window_start)
            local_rows: list[WordHypothesis] = []
            for start in range(window_start, max(window_start, window_end - min_word_len + 1)):
                for length in range(min_word_len, max_word_len + 1):
                    end = start + length
                    if end > window_end or end > len(text):
                        continue
                    observed = text[start:end]
                    if not observed.isalpha():
                        continue
                    for target, dictionary_rank in dictionary.get(length, []):
                        distance = hamming_distance(observed, target)
                        if distance <= 0 or distance > max_edits:
                            continue
                        edits = implied_edits(
                            observed=observed,
                            target=target,
                            sources=sources[start:end],
                            consensus=consensus,
                            alphabet=alphabet,
                            baseline_key=baseline_key,
                            baseline_mask=baseline_mask,
                            allow_stable_edits=allow_stable_edits,
                        )
                        if not edits:
                            continue
                        if len(edits) > max_edits:
                            continue
                        key = (str(page["test_id"]), start, end, target, edits)
                        if key in seen:
                            continue
                        seen.add(key)
                        rank_bonus = 1.0 / max(1.0, dictionary_rank ** 0.35)
                        score = length - distance * 1.7 + rank_bonus * 3.0 + float(window.get("damage_score") or 0.0)
                        local_rows.append(
                            WordHypothesis(
                                test_id=str(page["test_id"]),
                                window_start=window_start,
                                start=start,
                                end=end,
                                observed=observed,
                                target=target,
                                edits=edits,
                                distance=distance,
                                dictionary_rank=dictionary_rank,
                                local_score=score,
                            )
                        )
            local_rows.sort(
                key=lambda row: (
                    row.local_score,
                    len(row.target),
                    -row.distance,
                    -row.dictionary_rank,
                ),
                reverse=True,
            )
            hypotheses.extend(local_rows[: max(0, max_per_window)])
    hypotheses.sort(
        key=lambda row: (
            row.local_score,
            len(row.target),
            -row.distance,
            -row.dictionary_rank,
        ),
        reverse=True,
    )
    return dedupe_hypotheses_by_edits(hypotheses)


def dedupe_hypotheses_by_edits(hypotheses: list[WordHypothesis]) -> list[WordHypothesis]:
    best: dict[tuple[tuple[str, str], ...], WordHypothesis] = {}
    for row in hypotheses:
        old = best.get(row.edits)
        if old is None or row.local_score > old.local_score:
            best[row.edits] = row
    return sorted(
        best.values(),
        key=lambda row: (
            row.local_score,
            len(row.target),
            -row.distance,
            -row.dictionary_rank,
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Stage 3 -- edit-set conversion / hypothesis sets
# ---------------------------------------------------------------------------


def next_token_id(pages: list[PageBundle], symbol: str) -> int:
    for page in pages:
        for page_symbol, token_id in zip(page.symbols, page.token_ids):
            if page_symbol == symbol:
                return token_id
    raise KeyError(symbol)


def combined_edit_map(hypotheses: tuple[WordHypothesis, ...]) -> dict[str, str] | None:
    edit_map: dict[str, str] = {}
    for hypothesis in hypotheses:
        for symbol, target in hypothesis.edits:
            old = edit_map.get(symbol)
            if old is not None and old != target:
                return None
            edit_map[symbol] = target
    return edit_map


def compatible_hypothesis_set(
    hypotheses: tuple[WordHypothesis, ...], *, max_combined_edits: int
) -> bool:
    edit_map: dict[str, str] = {}
    for hypothesis in hypotheses:
        for symbol, target in hypothesis.edits:
            old = edit_map.get(symbol)
            if old is not None and old != target:
                return False
            edit_map[symbol] = target
    return 0 < len(edit_map) <= max_combined_edits


def pair_signature(hypotheses: tuple[WordHypothesis, ...]) -> str:
    return ";".join(
        sorted(f"{row.test_id}:{row.start}-{row.end}:{row.target}" for row in hypotheses)
    )


def combination_prescreen_key(hypotheses: tuple[WordHypothesis, ...]) -> tuple[float, int, int, str]:
    pages = {row.test_id for row in hypotheses}
    windows = {(row.test_id, row.window_start) for row in hypotheses}
    edits = combined_edit_map(hypotheses) or {}
    return (
        sum(row.local_score for row in hypotheses),
        len(pages) + len(windows),
        -len(edits),
        pair_signature(hypotheses),
    )


def build_hypothesis_sets(
    *,
    hypotheses: list[WordHypothesis],
    max_hypothesis_set_size: int,
    combination_candidate_limit: int,
    max_combinations: int,
    max_combined_edits: int,
) -> list[tuple[WordHypothesis, ...]]:
    """Build the singleton (and optionally multi-hypothesis) sets to evaluate.

    Always includes the empty baseline ``()`` and every singleton; when
    ``max_hypothesis_set_size > 1`` it adds compatible, deduplicated
    combinations prescreened by breadth and local score.
    """
    hypothesis_sets: list[tuple[WordHypothesis, ...]] = [()]
    hypothesis_sets.extend((row,) for row in hypotheses)
    max_size = max(1, int(max_hypothesis_set_size))
    if max_size <= 1:
        return hypothesis_sets

    pool = hypotheses[: max(0, int(combination_candidate_limit))]
    combinations: list[tuple[WordHypothesis, ...]] = []
    seen_edits: set[tuple[tuple[str, str], ...]] = set()
    for size in range(2, max_size + 1):
        for candidate_set in itertools.combinations(pool, size):
            if not compatible_hypothesis_set(candidate_set, max_combined_edits=max_combined_edits):
                continue
            edit_map = combined_edit_map(candidate_set)
            if not edit_map:
                continue
            edit_signature = tuple(sorted(edit_map.items()))
            if edit_signature in seen_edits:
                continue
            seen_edits.add(edit_signature)
            combinations.append(candidate_set)

    combinations.sort(key=combination_prescreen_key, reverse=True)
    if max_combinations >= 0:
        combinations = combinations[: max(0, int(max_combinations))]
    hypothesis_sets.extend(combinations)
    return hypothesis_sets


def build_edit_support(hypotheses: list[WordHypothesis]) -> dict[tuple[str, str], dict[str, Any]]:
    """Summarize how many hypotheses / pages / windows back each symbol edit."""
    support: dict[tuple[str, str], dict[str, Any]] = {}
    targets_by_symbol: dict[str, set[str]] = {}
    for hypothesis in hypotheses:
        for symbol, target in hypothesis.edits:
            key = (symbol, target)
            row = support.setdefault(
                key,
                {
                    "hypothesis_count": 0,
                    "local_score_sum": 0.0,
                    "pages": set(),
                    "windows": set(),
                    "examples": [],
                },
            )
            row["hypothesis_count"] += 1
            row["local_score_sum"] += float(hypothesis.local_score)
            row["pages"].add(hypothesis.test_id)
            row["windows"].add((hypothesis.test_id, hypothesis.window_start))
            if len(row["examples"]) < 6:
                row["examples"].append(f"{hypothesis.observed}->{hypothesis.target}")
            targets_by_symbol.setdefault(symbol, set()).add(target)
    normalized = {}
    for key, row in support.items():
        symbol, _target = key
        competing_targets = max(0, len(targets_by_symbol.get(symbol, set())) - 1)
        normalized[key] = {
            "hypothesis_count": int(row["hypothesis_count"]),
            "local_score_sum": round(float(row["local_score_sum"]), 6),
            "page_count": len(row["pages"]),
            "window_count": len(row["windows"]),
            "competing_target_count": competing_targets,
            "examples": row["examples"],
        }
    return normalized


# ---------------------------------------------------------------------------
# Stage 5 -- collateral (word-island) evidence
# ---------------------------------------------------------------------------


def project_text_sources(
    page: PageBundle,
    key: dict[int, int],
    mask: tuple[str, ...],
) -> tuple[str, list[str]]:
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
        sources.append(symbol)
    return "".join(chars), sources


def local_snippet(text: str, pos: int, radius: int = 14) -> str:
    start = max(0, pos - radius)
    end = min(len(text), pos + radius + 1)
    return text[start:end]


def local_window_quality(text: str, *, language: str = "de") -> float:
    if not text:
        return 0.0
    features = language_quality_feature_dict(text, language=language)
    return max(0.0, min(1.0, 1.0 - window_damage_score(features)))


def wildcard_patterns(word: str, max_wildcards: int) -> list[str]:
    positions = range(len(word))
    patterns = [word]
    for count in range(1, max(0, max_wildcards) + 1):
        for combo in itertools.combinations(positions, count):
            chars = list(word)
            for pos in combo:
                chars[pos] = "?"
            patterns.append("".join(chars))
    return patterns


def empty_word_evidence() -> dict[str, Any]:
    return {
        "score": 0.0,
        "span": [],
        "observed": "",
        "word": "",
        "distance": None,
        "dictionary_rank": None,
    }


def max_word_evidence_distance(length: int) -> int:
    if length <= 4:
        return 0
    if length <= 7:
        return 1
    return 2


def word_evidence_score(*, length: int, distance: int, rank: int) -> float:
    if distance > max_word_evidence_distance(length):
        return 0.0
    rank_bonus = 1.0 / max(1.0, rank ** 0.35)
    # Exact short function words should count, but near long content words
    # should be allowed to compete because these candidates are damaged basins.
    raw = length - 2.25 * distance
    return max(0.0, raw / max(1.0, length ** 0.5) + 0.65 * rank_bonus)


def word_evidence_reliability(evidence: dict[str, Any]) -> float:
    """Downweight fragile word-island evidence.

    Short exact words and long fuzzy words are useful hints but easy to
    hallucinate in damaged no-boundary candidates, so this keeps them from
    overpowering shared-key / global evidence.
    """
    word = str(evidence.get("word") or evidence.get("observed") or "")
    if not word:
        return 0.0
    length = len(word)
    distance = evidence.get("distance")
    distance = int(distance) if isinstance(distance, int) else 99
    if distance == 0:
        if length <= 3:
            return 0.22
        if length == 4:
            return 0.55
        return 1.0
    if distance == 1:
        if length >= 8:
            return 0.55
        if length >= 6:
            return 0.35
        return 0.18
    if distance == 2:
        if length >= 9:
            return 0.28
        if length >= 7:
            return 0.18
    return 0.08


def build_word_evidence_index(dictionary: dict[int, list[tuple[str, int]]]) -> WordEvidenceIndex:
    patterns: dict[int, dict[str, list[tuple[str, int]]]] = {}
    for length, words in dictionary.items():
        length_patterns: dict[str, list[tuple[str, int]]] = {}
        max_distance = max_word_evidence_distance(length)
        for word, rank in words:
            for pattern in wildcard_patterns(word, max_distance):
                length_patterns.setdefault(pattern, []).append((word, rank))
        patterns[length] = length_patterns
    return WordEvidenceIndex(dictionary=dictionary, patterns=patterns)


def near_dictionary_words(
    observed: str,
    word_index: WordEvidenceIndex,
    max_distance: int,
) -> list[tuple[str, int]]:
    by_word: dict[str, int] = {}
    for pattern in wildcard_patterns(observed, max_distance):
        for word, rank in word_index.patterns.get(len(observed), {}).get(pattern, []):
            old = by_word.get(word)
            if old is None or rank < old:
                by_word[word] = rank
    return [(word, rank) for word, rank in by_word.items()]


def best_covering_word_evidence(
    text: str,
    pos: int,
    word_index: WordEvidenceIndex,
) -> dict[str, Any]:
    """Return the strongest dictionary-like word island covering ``pos``.

    These candidates have no trustworthy spaces, so this is a local island
    signal rather than a segmentation claim: does a touched symbol participate
    in an exact or near dictionary word after an edit, and does the edit destroy
    such an island elsewhere?
    """
    if not text or pos < 0 or pos >= len(text):
        return empty_word_evidence()
    best = empty_word_evidence()
    for length in word_index.dictionary:
        if length < 3:
            continue
        start_min = max(0, pos - length + 1)
        start_max = min(pos, len(text) - length)
        for start in range(start_min, start_max + 1):
            observed = text[start:start + length]
            if not observed.isalpha():
                continue
            max_distance = max_word_evidence_distance(length)
            for target, rank in near_dictionary_words(observed, word_index, max_distance):
                distance = hamming_distance(observed, target)
                if distance > max_distance:
                    continue
                score = word_evidence_score(length=length, distance=distance, rank=rank)
                if score > float(best["score"]):
                    best = {
                        "score": round(score, 6),
                        "span": [start, start + length],
                        "observed": observed,
                        "word": target,
                        "distance": distance,
                        "dictionary_rank": rank,
                    }
    return best


def weighted_word_evidence_gain(impact: dict[str, Any]) -> float:
    delta = float(impact.get("word_evidence_delta") or 0.0)
    if delta <= 0.0:
        return 0.0
    after = impact.get("after_word_evidence") if isinstance(impact.get("after_word_evidence"), dict) else {}
    return delta * word_evidence_reliability(after)


def weighted_word_evidence_damage(impact: dict[str, Any]) -> float:
    delta = float(impact.get("word_evidence_delta") or 0.0)
    if delta >= 0.0:
        return 0.0
    before = impact.get("before_word_evidence") if isinstance(impact.get("before_word_evidence"), dict) else {}
    return -delta * word_evidence_reliability(before)


def target_only_penalty(
    *,
    edited_symbol_count: int,
    target_word_gain_sum: float,
    collateral_occurrences: int,
) -> float:
    if collateral_occurrences > 0:
        return 0.0
    # A repair that only affects the hand-picked target word is a weak shared
    # key claim. Keep it visible, but do not let pretty local words dominate.
    return min(8.0, 0.45 * target_word_gain_sum + 1.0 * max(1, edited_symbol_count))


def target_word_repairs(
    *,
    pages: list[PageBundle],
    before_key: dict[int, int],
    after_key: dict[int, int],
    mask: tuple[str, ...],
    hypotheses: tuple[WordHypothesis, ...],
) -> list[dict[str, Any]]:
    before_after_by_page: dict[str, tuple[str, str]] = {}
    for page in pages:
        before_text, _sources = project_text_sources(page, before_key, mask)
        after_text, _after_sources = project_text_sources(page, after_key, mask)
        before_after_by_page[page.test_id] = (before_text, after_text)
    rows = []
    for hypothesis in hypotheses:
        before_text, after_text = before_after_by_page.get(hypothesis.test_id, ("", ""))
        before = before_text[hypothesis.start:hypothesis.end]
        after = after_text[hypothesis.start:hypothesis.end]
        success = after == hypothesis.target
        rows.append({
            "test_id": hypothesis.test_id,
            "span": [hypothesis.start, hypothesis.end],
            "observed": hypothesis.observed,
            "before": before,
            "after": after,
            "target": hypothesis.target,
            "success": success,
            "distance": hypothesis.distance,
            "dictionary_rank": hypothesis.dictionary_rank,
            "target_word_gain": round(hypothesis.local_score if success else 0.0, 6),
        })
    return rows


def symbol_leverage_summary(
    *,
    edits: dict[str, str],
    impacts: list[dict[str, Any]],
    edit_support: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for symbol, target in sorted(edits.items()):
        symbol_impacts = [row for row in impacts if row.get("symbol") == symbol]
        collateral = [row for row in symbol_impacts if not row.get("in_hypothesis_target")]
        target_rows = [row for row in symbol_impacts if row.get("in_hypothesis_target")]
        support = edit_support.get((symbol, target), {})
        collateral_pages = {str(row.get("test_id")) for row in collateral}
        gain = sum(weighted_word_evidence_gain(row) for row in collateral)
        damage = sum(weighted_word_evidence_damage(row) for row in collateral)
        window_gain = sum(max(0.0, float(row.get("quality_delta") or 0.0)) for row in collateral)
        window_damage = sum(max(0.0, -float(row.get("quality_delta") or 0.0)) for row in collateral)
        support_score = (
            0.45 * min(3.0, float(support.get("hypothesis_count") or 0.0))
            + 0.025 * min(40.0, float(support.get("local_score_sum") or 0.0))
            + 0.35 * max(0.0, float(support.get("page_count") or 0.0) - 1.0)
            + 0.15 * max(0.0, float(support.get("window_count") or 0.0) - 1.0)
            - 0.22 * float(support.get("competing_target_count") or 0.0)
        )
        breadth_score = (
            0.10 * min(30.0, float(len(collateral)))
            + 0.45 * max(0.0, float(len(collateral_pages)) - 1.0)
            + 0.10 * min(10.0, float(len(target_rows)))
        )
        net_context = 0.85 * gain + 4.5 * window_gain - 0.65 * damage - 3.5 * window_damage
        target_only_discount = 1.8 if not collateral and target_rows else 0.0
        score = support_score + breadth_score + net_context - target_only_discount
        rows.append({
            "symbol": symbol,
            "target": target,
            "global_leverage_score": round(score, 6),
            "support_score": round(support_score, 6),
            "breadth_score": round(breadth_score, 6),
            "net_context_score": round(net_context, 6),
            "collateral_occurrences": len(collateral),
            "collateral_page_count": len(collateral_pages),
            "target_occurrences": len(target_rows),
            "weighted_word_gain": round(gain, 6),
            "weighted_word_damage": round(damage, 6),
            "window_gain": round(window_gain, 6),
            "window_damage": round(window_damage, 6),
            "support": support,
        })
    return rows


def adjudicate_repair(
    *,
    pages: list[PageBundle],
    before_key: dict[int, int],
    after_key: dict[int, int],
    mask: tuple[str, ...],
    hypotheses: tuple[WordHypothesis, ...],
    edits: dict[str, str],
    word_index: WordEvidenceIndex,
    edit_support: dict[tuple[str, str], dict[str, Any]],
    language: str = "de",
) -> dict[str, Any]:
    """Occurrence-level word-island adjudication of an edit set.

    For every occurrence of an edited symbol across all pages, compare the
    local snippet and covering word-island before and after the edit. Target
    occurrences (inside a proposed word span) count as gain; other occurrences
    count as collateral gain or damage. Ground-truth-free.
    """
    if not edits:
        return {
            "edited_symbol_count": 0,
            "occurrence_count": 0,
            "target_occurrences": 0,
            "collateral_occurrences": 0,
            "improved_occurrences": 0,
            "damaged_occurrences": 0,
            "target_gain_sum": 0.0,
            "collateral_gain_sum": 0.0,
            "collateral_damage_sum": 0.0,
            "occurrence_impacts": [],
        }
    target_ranges: dict[str, list[tuple[int, int]]] = {}
    for hypothesis in hypotheses:
        target_ranges.setdefault(hypothesis.test_id, []).append((hypothesis.start, hypothesis.end))
    target_repairs = target_word_repairs(
        pages=pages,
        before_key=before_key,
        after_key=after_key,
        mask=mask,
        hypotheses=hypotheses,
    )
    impacts = []
    for page in pages:
        before_text, sources = project_text_sources(page, before_key, mask)
        after_text, after_sources = project_text_sources(page, after_key, mask)
        if sources != after_sources:
            continue
        for pos, symbol in enumerate(sources):
            if symbol not in edits:
                continue
            before_snippet = local_snippet(before_text, pos)
            after_snippet = local_snippet(after_text, pos)
            before_quality = local_window_quality(before_snippet, language=language)
            after_quality = local_window_quality(after_snippet, language=language)
            delta = after_quality - before_quality
            before_word = best_covering_word_evidence(before_text, pos, word_index)
            after_word = best_covering_word_evidence(after_text, pos, word_index)
            word_delta = float(after_word["score"]) - float(before_word["score"])
            in_target = any(start <= pos < end for start, end in target_ranges.get(page.test_id, []))
            impacts.append({
                "test_id": page.test_id,
                "symbol": symbol,
                "position": pos,
                "target": edits[symbol],
                "in_hypothesis_target": in_target,
                "before": before_snippet,
                "after": after_snippet,
                "before_quality": round(before_quality, 6),
                "after_quality": round(after_quality, 6),
                "quality_delta": round(delta, 6),
                "before_word_evidence": before_word,
                "after_word_evidence": after_word,
                "word_evidence_delta": round(word_delta, 6),
            })
    target_impacts = [row for row in impacts if row["in_hypothesis_target"]]
    collateral_impacts = [row for row in impacts if not row["in_hypothesis_target"]]
    target_gain_sum = sum(float(row["quality_delta"]) for row in target_impacts)
    collateral_gain_sum = sum(max(0.0, float(row["quality_delta"])) for row in collateral_impacts)
    collateral_damage_sum = sum(max(0.0, -float(row["quality_delta"])) for row in collateral_impacts)
    target_word_evidence_gain_sum = sum(
        max(0.0, float(row["word_evidence_delta"])) for row in target_impacts
    )
    collateral_word_gain_sum = sum(
        max(0.0, float(row["word_evidence_delta"])) for row in collateral_impacts
    )
    collateral_word_damage_sum = sum(
        max(0.0, -float(row["word_evidence_delta"])) for row in collateral_impacts
    )
    collateral_word_gain_weighted_sum = sum(
        weighted_word_evidence_gain(row) for row in collateral_impacts
    )
    collateral_word_damage_weighted_sum = sum(
        weighted_word_evidence_damage(row) for row in collateral_impacts
    )
    target_word_gain_sum = sum(float(row["target_word_gain"]) for row in target_repairs if row["success"])
    symbol_leverage = symbol_leverage_summary(
        edits=edits,
        impacts=impacts,
        edit_support=edit_support,
    )
    global_leverage_score = sum(float(row["global_leverage_score"]) for row in symbol_leverage)
    word_improved_weighted = sum(
        1
        for row in impacts
        if weighted_word_evidence_gain(row) > 0.35
    )
    word_damaged_weighted = sum(
        1
        for row in impacts
        if weighted_word_evidence_damage(row) > 0.35
    )
    impacts.sort(
        key=lambda row: (
            not row["in_hypothesis_target"],
            -abs(float(row["word_evidence_delta"])),
            -abs(float(row["quality_delta"])),
            row["test_id"],
            row["position"],
        )
    )
    return {
        "edited_symbol_count": len(edits),
        "occurrence_count": len(impacts),
        "target_occurrences": len(target_impacts),
        "collateral_occurrences": len(collateral_impacts),
        "improved_occurrences": sum(1 for row in impacts if float(row["quality_delta"]) > 0.015),
        "damaged_occurrences": sum(1 for row in impacts if float(row["quality_delta"]) < -0.015),
        "target_gain_sum": round(target_gain_sum, 6),
        "target_word_gain_sum": round(target_word_gain_sum, 6),
        "target_word_evidence_gain_sum": round(target_word_evidence_gain_sum, 6),
        "collateral_gain_sum": round(collateral_gain_sum, 6),
        "collateral_damage_sum": round(collateral_damage_sum, 6),
        "collateral_word_gain_sum": round(collateral_word_gain_sum, 6),
        "collateral_word_damage_sum": round(collateral_word_damage_sum, 6),
        "collateral_word_gain_weighted_sum": round(collateral_word_gain_weighted_sum, 6),
        "collateral_word_damage_weighted_sum": round(collateral_word_damage_weighted_sum, 6),
        "target_gain_avg": round(target_gain_sum / max(1, len(target_impacts)), 6),
        "collateral_gain_avg": round(collateral_gain_sum / max(1, len(collateral_impacts)), 6),
        "collateral_damage_avg": round(collateral_damage_sum / max(1, len(collateral_impacts)), 6),
        "target_word_evidence_gain_avg": round(
            target_word_evidence_gain_sum / max(1, len(target_impacts)), 6
        ),
        "collateral_word_gain_avg": round(
            collateral_word_gain_sum / max(1, len(collateral_impacts)), 6
        ),
        "collateral_word_damage_avg": round(
            collateral_word_damage_sum / max(1, len(collateral_impacts)), 6
        ),
        "collateral_word_gain_weighted_avg": round(
            collateral_word_gain_weighted_sum / max(1, len(collateral_impacts)), 6
        ),
        "collateral_word_damage_weighted_avg": round(
            collateral_word_damage_weighted_sum / max(1, len(collateral_impacts)), 6
        ),
        "word_improved_occurrences": sum(
            1 for row in impacts if float(row["word_evidence_delta"]) > 0.35
        ),
        "word_damaged_occurrences": sum(
            1 for row in impacts if float(row["word_evidence_delta"]) < -0.35
        ),
        "word_improved_weighted_occurrences": word_improved_weighted,
        "word_damaged_weighted_occurrences": word_damaged_weighted,
        "global_leverage_score": round(global_leverage_score, 6),
        "symbol_leverage": symbol_leverage,
        "target_only_penalty": round(target_only_penalty(
            edited_symbol_count=len(edits),
            target_word_gain_sum=target_word_gain_sum,
            collateral_occurrences=len(collateral_impacts),
        ), 6),
        "target_repairs": target_repairs,
        "occurrence_impacts": impacts[:24],
    }


def adjudication_score(adjudication: dict[str, Any], hypothesis_score: float) -> float:
    target_word_gain = float(adjudication.get("target_word_gain_sum") or 0.0)
    target_word_evidence = float(adjudication.get("target_word_evidence_gain_avg") or 0.0)
    target_gain = float(adjudication.get("target_gain_avg") or 0.0)
    collateral_gain = float(adjudication.get("collateral_gain_avg") or 0.0)
    collateral_damage = float(adjudication.get("collateral_damage_avg") or 0.0)
    collateral_word_gain = float(adjudication.get("collateral_word_gain_weighted_avg") or 0.0)
    collateral_word_damage = float(adjudication.get("collateral_word_damage_weighted_avg") or 0.0)
    damaged = float(adjudication.get("damaged_occurrences") or 0.0)
    word_damaged = float(adjudication.get("word_damaged_weighted_occurrences") or 0.0)
    target_count = float(adjudication.get("target_occurrences") or 0.0)
    occurrence_count = float(adjudication.get("occurrence_count") or 0.0)
    target_only = float(adjudication.get("target_only_penalty") or 0.0)
    global_leverage = float(adjudication.get("global_leverage_score") or 0.0)
    support = min(1.0, target_count / max(1.0, occurrence_count))
    return (
        0.25 * float(hypothesis_score)
        + 0.45 * target_word_gain
        + 1.4 * target_word_evidence
        + 5.0 * target_gain
        + 2.0 * collateral_gain
        + 0.9 * collateral_word_gain
        + 0.42 * global_leverage
        + 0.35 * support
        - 6.0 * collateral_damage
        - 1.5 * collateral_word_damage
        - 0.12 * damaged
        - 0.18 * word_damaged
        - target_only
    )


def adjudication_no_target_score(adjudication: dict[str, Any]) -> float:
    """Score repair collateral only, excluding the target word span."""
    collateral_gain = float(adjudication.get("collateral_gain_avg") or 0.0)
    collateral_damage = float(adjudication.get("collateral_damage_avg") or 0.0)
    collateral_word_gain = float(adjudication.get("collateral_word_gain_weighted_avg") or 0.0)
    collateral_word_damage = float(adjudication.get("collateral_word_damage_weighted_avg") or 0.0)
    word_damaged = float(adjudication.get("word_damaged_weighted_occurrences") or 0.0)
    word_improved = float(adjudication.get("word_improved_weighted_occurrences") or 0.0)
    collateral_count = float(adjudication.get("collateral_occurrences") or 0.0)
    support = min(1.0, collateral_count / 12.0)
    global_leverage = float(adjudication.get("global_leverage_score") or 0.0)
    return (
        1.25 * collateral_word_gain
        + 3.0 * collateral_gain
        + 0.35 * global_leverage
        + 0.12 * word_improved
        + 0.25 * support
        - 2.25 * collateral_word_damage
        - 5.5 * collateral_damage
        - 0.16 * word_damaged
    )


# ---------------------------------------------------------------------------
# Stage 5 -- acceptance / evidence / ranking
# ---------------------------------------------------------------------------


def mask_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in (row.get("mask") or [])))


def numeric_delta(row: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(float(row.get(key) or 0.0) - float(baseline.get(key) or 0.0), 6)


def nested_delta(row: dict[str, Any], baseline: dict[str, Any], parent: str, key: str) -> float:
    return round(
        float((row.get(parent) or {}).get(key) or 0.0)
        - float((baseline.get(parent) or {}).get(key) or 0.0),
        6,
    )


def keyed_by_test_id(rows: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("test_id") or ""): row
        for row in rows
        if isinstance(row, dict)
    }


def changed_excerpt(before: str, after: str, *, radius: int = 28) -> dict[str, Any]:
    if before == after:
        return {"changed": False, "before": "", "after": "", "offset": None}
    limit = min(len(before), len(after))
    start = 0
    while start < limit and before[start] == after[start]:
        start += 1
    left = max(0, start - radius)
    right = min(max(len(before), len(after)), start + radius)
    return {
        "changed": True,
        "offset": start,
        "before": before[left:right],
        "after": after[left:right],
    }


def repair_evidence(row: dict[str, Any], *, baseline: dict[str, Any]) -> dict[str, Any]:
    """Per-page runtime/preview evidence for a variant vs the baseline.

    ``post_hoc_page_chars`` (ground-truth char accuracy) is consumed only if a
    caller populated it for calibration; in the runtime path it is absent and
    those deltas stay zero.
    """
    baseline_runtime = keyed_by_test_id(baseline.get("page_runtime_scores") or [])
    baseline_chars = keyed_by_test_id(baseline.get("post_hoc_page_chars") or [])
    baseline_previews = keyed_by_test_id(baseline.get("page_previews") or [])
    page_evidence = []
    runtime_improved = 0
    runtime_regressed = 0
    preview_changed = 0
    runtime_suspicious_pages = 0
    calibration_suspicious_pages = 0
    posthoc_improved = 0
    posthoc_regressed = 0
    for runtime in row.get("page_runtime_scores") or []:
        test_id = str(runtime.get("test_id") or "")
        base_runtime = baseline_runtime.get(test_id, {})
        base_char = baseline_chars.get(test_id, {})
        char = (keyed_by_test_id(row.get("post_hoc_page_chars") or []).get(test_id, {}))
        base_preview = str((baseline_previews.get(test_id) or {}).get("preview") or "")
        preview = str((keyed_by_test_id(row.get("page_previews") or {}).get(test_id, {})).get("preview") or "")
        val_delta = numeric_delta(runtime, base_runtime, "validation_score_v2")
        lq_delta = numeric_delta(runtime, base_runtime, "language_quality_mean")
        dict_delta = numeric_delta(runtime, base_runtime, "dict_rate")
        pseudo_delta = nested_delta(runtime, base_runtime, "diagnostics", "pseudo_word_fraction")
        binary_delta = nested_delta(runtime, base_runtime, "validation_components_v2", "binary_ngram_fit")
        coherence_delta = nested_delta(runtime, base_runtime, "validation_components_v2", "language_coherence")
        shape_delta = nested_delta(runtime, base_runtime, "validation_components_v2", "language_shape")
        char_delta = numeric_delta(char, base_char, "char_accuracy")
        changed = preview != base_preview
        if val_delta > 0.005:
            runtime_improved += 1
        elif val_delta < -0.005:
            runtime_regressed += 1
        if char_delta > 0.001:
            posthoc_improved += 1
        elif char_delta < -0.001:
            posthoc_regressed += 1
        if changed:
            preview_changed += 1
        runtime_flags = []
        if val_delta > 0.005 and lq_delta <= 0.0:
            runtime_flags.append("validation_up_without_lq_gain")
        if val_delta > 0.005 and dict_delta < -0.01:
            runtime_flags.append("validation_up_dictionary_down")
        if val_delta > 0.005 and pseudo_delta > 0.01:
            runtime_flags.append("validation_up_more_pseudowords")
        if val_delta > 0.005 and not changed:
            runtime_flags.append("validation_up_preview_unchanged")
        if runtime_flags:
            runtime_suspicious_pages += 1
        calibration_flags = []
        if val_delta > 0.005 and char_delta < -0.001:
            calibration_flags.append("runtime_up_posthoc_char_down")
        if calibration_flags:
            calibration_suspicious_pages += 1
        page_evidence.append({
            "test_id": test_id,
            "validation_delta": val_delta,
            "language_quality_delta": lq_delta,
            "dict_rate_delta": dict_delta,
            "pseudo_word_fraction_delta": pseudo_delta,
            "binary_ngram_fit_delta": binary_delta,
            "language_coherence_delta": coherence_delta,
            "language_shape_delta": shape_delta,
            "post_hoc_char_delta": char_delta,
            "preview_changed": changed,
            "changed_excerpt": changed_excerpt(base_preview, preview),
            "runtime_flags": runtime_flags,
            "calibration_flags": calibration_flags,
        })
    page_evidence.sort(
        key=lambda item: (
            -len(item["runtime_flags"]) - len(item["calibration_flags"]),
            -abs(float(item["validation_delta"])),
            item["test_id"],
        )
    )
    runtime_decision_flags = []
    if runtime_suspicious_pages:
        runtime_decision_flags.append(f"{runtime_suspicious_pages} page(s) improve by validation but have weak supporting signals")
    if runtime_improved <= runtime_regressed and row.get("edits") != ["baseline"]:
        runtime_decision_flags.append("runtime improvements are not page-majority")
    if not preview_changed and row.get("edits") != ["baseline"]:
        runtime_decision_flags.append("no preview changed")
    calibration_flags = []
    if calibration_suspicious_pages:
        calibration_flags.append(f"{calibration_suspicious_pages} page(s) improve by runtime score but lose post-hoc char")
    if posthoc_regressed > posthoc_improved and row.get("edits") != ["baseline"]:
        calibration_flags.append("post-hoc char regresses on more pages than it improves")
    return {
        "page_count": len(page_evidence),
        "runtime_pages_improved": runtime_improved,
        "runtime_pages_regressed": runtime_regressed,
        "preview_pages_changed": preview_changed,
        "runtime_suspicious_pages": runtime_suspicious_pages,
        "post_hoc_pages_improved": posthoc_improved,
        "post_hoc_pages_regressed": posthoc_regressed,
        "calibration_suspicious_pages": calibration_suspicious_pages,
        "runtime_decision_flags": runtime_decision_flags,
        "calibration_flags": calibration_flags,
        "pages": page_evidence,
    }


def repair_acceptance(
    row: dict[str, Any],
    *,
    baseline: dict[str, Any],
    robust_margin: float,
    min_page_drop: float,
    max_illusion_increase: float,
    allow_pair_acceptance: bool = False,
) -> dict[str, Any]:
    """Accept/reject a variant on page-group runtime metric deltas (GT-free)."""
    deltas = {
        "page_robust_score": numeric_delta(row, baseline, "page_robust_score"),
        "page_balanced_score": numeric_delta(row, baseline, "page_balanced_score"),
        "page_validation_avg": numeric_delta(row, baseline, "page_validation_avg"),
        "page_validation_min": numeric_delta(row, baseline, "page_validation_min"),
        "fragment_illusion_penalty": numeric_delta(row, baseline, "fragment_illusion_penalty"),
        "page_language_quality_avg": numeric_delta(row, baseline, "page_language_quality_avg"),
    }
    edits = row.get("edits") or []
    if edits == ["baseline"]:
        return {
            "accepted": False,
            "decision": "baseline",
            "deltas": deltas,
            "reasons": ["baseline candidate"],
        }
    reasons = []
    accepted = True
    edit_count = len(edits)
    if edit_count > 1 and not allow_pair_acceptance:
        accepted = False
        reasons.append("multi-edit variant is review-only unless pair acceptance is allowed")
    if deltas["page_robust_score"] < robust_margin:
        accepted = False
        reasons.append(f"robust gain below margin ({deltas['page_robust_score']:.3f} < {robust_margin:.3f})")
    else:
        reasons.append(f"robust gain clears margin ({deltas['page_robust_score']:.3f})")
    if deltas["page_balanced_score"] < 0.0:
        accepted = False
        reasons.append(f"balanced score regresses ({deltas['page_balanced_score']:.3f})")
    if deltas["page_validation_avg"] < 0.0:
        accepted = False
        reasons.append(f"average page validation regresses ({deltas['page_validation_avg']:.3f})")
    if deltas["page_validation_min"] < -min_page_drop:
        accepted = False
        reasons.append(f"worst page drops too much ({deltas['page_validation_min']:.3f})")
    if deltas["fragment_illusion_penalty"] > max_illusion_increase:
        accepted = False
        reasons.append(f"fragment illusion rises too much ({deltas['fragment_illusion_penalty']:.3f})")
    positive_support = sum(
        1
        for key in ("page_robust_score", "page_balanced_score", "page_validation_avg", "page_language_quality_avg")
        if deltas[key] > 0.0
    )
    if positive_support < 2:
        accepted = False
        reasons.append(f"only {positive_support} runtime signals improve")
    return {
        "accepted": accepted,
        "decision": "runtime_accept" if accepted else "hold_for_review",
        "deltas": deltas,
        "positive_signal_count": positive_support,
        "reasons": reasons,
    }


def annotate_acceptance(
    variants: list[dict[str, Any]],
    *,
    baseline: dict[str, Any],
    robust_margin: float,
    min_page_drop: float,
    max_illusion_increase: float,
    allow_pair_acceptance: bool = False,
) -> None:
    for row in variants:
        row["repair_acceptance"] = repair_acceptance(
            row,
            baseline=baseline,
            robust_margin=robust_margin,
            min_page_drop=min_page_drop,
            max_illusion_increase=max_illusion_increase,
            allow_pair_acceptance=allow_pair_acceptance,
        )


def annotate_repair_evidence(variants: list[dict[str, Any]], *, baseline: dict[str, Any]) -> None:
    for row in variants:
        row["repair_evidence"] = repair_evidence(row, baseline=baseline)


def variant_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("page_robust_score") or 0.0),
        float(row.get("page_balanced_score") or 0.0),
        float(row.get("page_validation_avg") or 0.0),
        -len(row.get("edits") or []),
    )


def variant_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "edits": row.get("edits") or [],
        "mask": row.get("mask") or [],
        "page_robust_score": row.get("page_robust_score"),
        "page_balanced_score": row.get("page_balanced_score"),
        "page_validation_avg": row.get("page_validation_avg"),
        "page_validation_min": row.get("page_validation_min"),
        "fragment_illusion_penalty": row.get("fragment_illusion_penalty"),
        "page_language_quality_avg": row.get("page_language_quality_avg"),
        "page_dict_avg": row.get("page_dict_avg"),
        "page_content_char_avg": row.get("page_content_char_avg"),
        "page_pseudo_word_avg": row.get("page_pseudo_word_avg"),
        "page_binary_component_avg": row.get("page_binary_component_avg"),
        "page_shape_component_avg": row.get("page_shape_component_avg"),
        "page_evidence_dispersion_avg": row.get("page_evidence_dispersion_avg"),
        "page_window_stability_avg": row.get("page_window_stability_avg"),
        "page_repetition_control_avg": row.get("page_repetition_control_avg"),
        "page_content_word_quality_avg": row.get("page_content_word_quality_avg"),
        "page_language_coherence_avg": row.get("page_language_coherence_avg"),
        "language_quality_rank_score": row.get("language_quality_rank_score"),
        "language_quality_rank_normalized": row.get("language_quality_rank_normalized"),
        "post_hoc_char_avg": row.get("post_hoc_char_avg"),
        "post_hoc_page_chars": row.get("post_hoc_page_chars") or [],
        "repair_acceptance": row.get("repair_acceptance") or {},
        "repair_evidence": row.get("repair_evidence") or {},
        "preview": row.get("preview"),
    }


def hypothesis_to_dict(row: WordHypothesis) -> dict[str, Any]:
    return {
        "test_id": row.test_id,
        "window_start": row.window_start,
        "start": row.start,
        "end": row.end,
        "observed": row.observed,
        "target": row.target,
        "edits": [f"{symbol}->{target}" for symbol, target in row.edits],
        "distance": row.distance,
        "dictionary_rank": row.dictionary_rank,
        "local_score": round(row.local_score, 6),
    }


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def _score_variant_row(
    *,
    pages: list[PageBundle],
    key: dict[int, int],
    mask: tuple[str, ...],
    edits: list[str],
    hypothesis_set: tuple[WordHypothesis, ...],
    language: str,
) -> dict[str, Any]:
    """Project + runtime-score all pages for one key/mask (ground-truth-free)."""
    page_rows = project_pages(pages=pages, key=key, mask=mask)
    runtime_scores = [score_page_runtime(row, key=key, mask=mask, language=language) for row in page_rows]
    metrics = page_runtime_metrics(runtime_scores)
    variant = {
        "edits": edits or ["baseline"],
        "mask": list(mask),
        "word_hypotheses": [hypothesis_to_dict(row) for row in hypothesis_set],
        "word_hypothesis_score": round(sum(row.local_score for row in hypothesis_set), 6),
        "page_runtime_scores": runtime_scores,
        "page_previews": [
            {
                "test_id": row["test_id"],
                # Runtime-only preview of the *projected* text; never plaintext.
                "preview": str(row.get("decryption") or "")[:180],
                "filtered_length": int(row.get("filtered_length") or 0),
            }
            for row in page_rows
        ],
        "preview": "; ".join(str(row.get("decryption") or "")[:70] for row in page_rows[:3]),
        **metrics,
    }
    return variant


def propose_word_repairs(
    *,
    pages: list[PageBundle],
    shared_key: dict[int, int],
    dictionary_path: str | Path,
    language: str = "de",
    config: WordRepairConfig | None = None,
    mask: tuple[str, ...] = (),
    consensus: dict[str, dict[str, Any]] | None = None,
    alphabet: Alphabet | None = None,
    source_branch: str | None = None,
) -> list[CandidatePacket]:
    """Run the word-hypothesis repair pipeline and return candidate packets.

    Stages: damaged-window detection -> hypothesis proposal -> edit-set
    conversion -> cross-page rescoring (:func:`multipage.score_page_runtime`) ->
    collateral adjudication. Every stage is ground-truth-free; the returned
    packets set ``text=None`` and carry projected-text previews only.

    Deviations from the spec sketch ``propose_word_repairs(pages, shared_key,
    dictionary_path, language, config)``: ``mask``, ``consensus``, ``alphabet``,
    and ``source_branch`` are additional (optional) parameters that the extracted
    pipeline genuinely needs. ``consensus`` may be omitted (no symbol treated as
    stable); ``alphabet`` defaults to the page group's shared alphabet.
    """
    config = config or WordRepairConfig()
    consensus = consensus or {}
    if alphabet is None:
        alphabet = alphabet_from_pages(pages)
    baseline_key = dict(shared_key)
    baseline_mask = tuple(sorted(str(symbol) for symbol in mask))

    dictionary = load_dictionary(dictionary_path, config.min_word_len, config.max_word_len)
    collateral_dictionary = load_dictionary(dictionary_path, 3, config.max_word_len)
    collateral_index = build_word_evidence_index(collateral_dictionary)

    page_windows = build_page_windows(
        pages=pages,
        alphabet=alphabet,
        key=baseline_key,
        mask=baseline_mask,
        consensus=consensus,
        window_size=config.window_size,
        window_step=config.window_step,
        windows_per_page=config.windows_per_page,
        language=language,
    )
    hypotheses = generate_word_hypotheses(
        page_windows=page_windows,
        dictionary=dictionary,
        consensus=consensus,
        alphabet=alphabet,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        min_word_len=config.min_word_len,
        max_word_len=config.max_word_len,
        max_edits=config.max_edits,
        max_per_window=config.max_hypotheses_per_window,
        allow_stable_edits=config.allow_stable_edits,
    )
    hypotheses = hypotheses[: max(0, config.max_hypotheses)]
    edit_support = build_edit_support(hypotheses)
    hypothesis_sets = build_hypothesis_sets(
        hypotheses=hypotheses,
        max_hypothesis_set_size=config.max_hypothesis_set_size,
        combination_candidate_limit=config.combination_candidate_limit,
        max_combinations=config.max_combinations,
        max_combined_edits=config.max_combined_edits,
    )

    variants: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = set()
    for hypothesis_set in hypothesis_sets:
        edit_map = combined_edit_map(hypothesis_set)
        if edit_map is None:
            continue
        key = dict(baseline_key)
        variant_mask = set(baseline_mask)
        edit_labels: list[str] = []
        for symbol, target in sorted(edit_map.items()):
            token_id = next_token_id(pages, symbol)
            before = current_assignment(symbol, token_id, key, tuple(sorted(variant_mask)))
            apply_assignment(symbol, token_id, target, key, variant_mask)
            edit_labels.append(f"{symbol}:{before}->{target}")
        identity = (tuple(sorted(variant_mask)), tuple(sorted(key.items())))
        if identity in seen:
            continue
        seen.add(identity)

        variant = _score_variant_row(
            pages=pages,
            key=key,
            mask=tuple(sorted(variant_mask)),
            edits=edit_labels,
            hypothesis_set=hypothesis_set,
            language=language,
        )
        if hypothesis_set:
            adjudication = adjudicate_repair(
                pages=pages,
                before_key=baseline_key,
                after_key=key,
                mask=tuple(sorted(variant_mask)),
                hypotheses=hypothesis_set,
                edits=edit_map,
                word_index=collateral_index,
                edit_support=edit_support,
                language=language,
            )
            adjudication["word_hypothesis_score"] = variant["word_hypothesis_score"]
            adjudication["adjudication_score"] = round(
                adjudication_score(adjudication, variant["word_hypothesis_score"]), 6
            )
            adjudication["adjudication_no_target_score"] = round(
                adjudication_no_target_score(adjudication), 6
            )
            adjudication["target_leverage_score"] = round(
                float(adjudication["adjudication_score"])
                - float(adjudication["adjudication_no_target_score"]),
                6,
            )
            variant["repair_adjudication"] = adjudication
        variants.append(variant)

    baseline_variant = next(
        (row for row in variants if row.get("edits") == ["baseline"]),
        variants[0] if variants else {},
    )
    annotate_acceptance(
        variants,
        baseline=baseline_variant,
        robust_margin=config.acceptance_margin,
        min_page_drop=config.min_page_drop,
        max_illusion_increase=config.max_illusion_increase,
        allow_pair_acceptance=config.allow_pair_acceptance,
    )
    annotate_repair_evidence(variants, baseline=baseline_variant)

    edited = [row for row in variants if row.get("edits") != ["baseline"]]
    edited.sort(key=variant_rank_key, reverse=True)

    packets: list[CandidatePacket] = []
    for rank, row in enumerate(edited, start=1):
        packets.append(packet_from_word_repair_row(row, rank=rank, source_branch=source_branch))
    return packets
