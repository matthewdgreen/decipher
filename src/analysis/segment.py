"""Dynamic-programming word segmentation for no-boundary plaintext.

When a cipher has been stripped of word separators (one continuous run of
letters), scoring by dictionary hit-rate is impossible without first
guessing where the word boundaries are. This module provides a Viterbi
segmenter: given a word set (optionally with frequency ranks), find the
single segmentation that minimises total log-cost.

Used by:
- analysis.signals.compute_panel     (no-boundary dictionary_rate)
- agent.tools_v2._tool_score_dictionary
- agent.tools_v2._tool_decode_diagnose
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SegmentResult:
    text: str
    segmented: str              # space-separated best segmentation
    words: list[str]
    dict_rate: float            # fraction of segmented words in dict
    pseudo_words: list[str]     # segments NOT in dict
    cost: float                 # total Viterbi log-cost (lower = better)


@dataclass
class TextRepairResult:
    original_text: str
    repaired_text: str
    before: SegmentResult
    after: SegmentResult
    rounds: int
    corrections: list[dict[str, str | int]]
    applied: bool
    reason: str | None = None


# Per-emitted-word constant. Discourages over-splitting — without it, the
# segmenter prefers many short dictionary hits ("MAR PA FOUND") over one
# long unknown pseudo-word ("MARPA FOUND"). Keeping long unknowns intact is
# essential for find_one_edit_corrections to spot residual letter errors.
_WORD_PENALTY = 6.0

# Unknown-span penalty per character. An L-length unknown span costs
# _WORD_PENALTY + L * _UNKNOWN_PER_CHAR. Tuned so a long unknown span
# (~7 chars) is cheaper than fragmenting it into short dict-entries.
_UNKNOWN_PER_CHAR = 5.0


def _word_cost(word: str, freq_rank: dict[str, int] | None) -> float:
    """Log-cost for a known dictionary word.

    Uses -log(p(word)) approximated from its rank (Zipf-style). If no rank
    table is provided, every known word gets cost 0 — ordering by dict-hit
    count only.
    """
    if freq_rank is None:
        return 0.0
    rank = freq_rank.get(word)
    if rank is None:
        return 0.0
    return math.log(rank + 1)


def _sequence_cost(words: list[str], word_set: set[str], freq_rank: dict[str, int] | None) -> float:
    total = 0.0
    for word in words:
        if word in word_set:
            total += _WORD_PENALTY + _word_cost(word, freq_rank)
        else:
            total += _WORD_PENALTY + _UNKNOWN_PER_CHAR * len(word)
    return total


def segment_text(
    text: str,
    word_set: set[str],
    freq_rank: dict[str, int] | None = None,
    min_word_len: int = 1,
    max_word_len: int = 20,
) -> SegmentResult:
    """Segment a continuous run of letters into the most likely word sequence.

    Non-letter characters are left untouched and used as natural boundaries —
    so text that already has some spaces (e.g. partial decodes) segments
    chunk-by-chunk rather than pathologically merging across whitespace.
    """
    if not text:
        return SegmentResult(text="", segmented="", words=[],
                             dict_rate=0.0, pseudo_words=[], cost=0.0)

    upper = text.upper()

    # Split on non-letter runs; segment each alpha chunk independently.
    chunks: list[str] = []
    current: list[str] = []
    for ch in upper:
        if ch.isalpha():
            current.append(ch)
        else:
            if current:
                chunks.append("".join(current))
                current = []
    if current:
        chunks.append("".join(current))

    all_words: list[str] = []
    total_cost = 0.0
    for chunk in chunks:
        words, cost = _viterbi_segment(
            chunk, word_set, freq_rank,
            min_word_len=min_word_len,
            max_word_len=max_word_len,
        )
        all_words.extend(words)
        total_cost += cost

    segmented = " ".join(all_words)
    pseudo = [w for w in all_words if w not in word_set]
    if all_words:
        hits = sum(1 for w in all_words if w in word_set)
        dict_rate = hits / len(all_words)
    else:
        dict_rate = 0.0

    return SegmentResult(
        text=text,
        segmented=segmented,
        words=all_words,
        dict_rate=dict_rate,
        pseudo_words=pseudo,
        cost=total_cost,
    )


def _viterbi_segment(
    chunk: str,
    word_set: set[str],
    freq_rank: dict[str, int] | None,
    min_word_len: int,
    max_word_len: int,
) -> tuple[list[str], float]:
    n = len(chunk)
    if n == 0:
        return [], 0.0

    INF = float("inf")
    # cost[i] = best cost to reach position i (0..n)
    cost = [INF] * (n + 1)
    cost[0] = 0.0
    back = [0] * (n + 1)  # index we came from

    for i in range(1, n + 1):
        lo = max(0, i - max_word_len)
        for j in range(lo, i):
            if cost[j] == INF:
                continue
            span = chunk[j:i]
            span_len = i - j
            if span in word_set:
                c = _WORD_PENALTY + _word_cost(span, freq_rank)
            else:
                # Unknown span — per-word penalty plus per-char cost.
                c = _WORD_PENALTY + _UNKNOWN_PER_CHAR * span_len
            total = cost[j] + c
            if total < cost[i]:
                cost[i] = total
                back[i] = j

    # Recover segmentation
    words: list[str] = []
    i = n
    while i > 0:
        j = back[i]
        words.append(chunk[j:i])
        i = j
    words.reverse()
    return words, cost[n]


def find_one_edit_corrections(
    pseudo_word: str,
    word_set: set[str],
) -> list[tuple[str, str, str]]:
    """Return (candidate_word, wrong_letter, correct_letter) triples for every
    dictionary word reachable from `pseudo_word` by a single substitution.

    Only substitutions at one position — not insertions/deletions — so we can
    attribute each correction to a concrete (wrong, correct) swap the agent
    could apply with act_swap_decoded.
    """
    w = pseudo_word.upper()
    if not w:
        return []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out: list[tuple[str, str, str]] = []
    for i, wrong in enumerate(w):
        if not wrong.isalpha():
            continue
        for correct in alphabet:
            if correct == wrong:
                continue
            candidate = w[:i] + correct + w[i + 1:]
            if candidate in word_set:
                out.append((candidate, wrong, correct))
    return out


def repair_no_boundary_text(
    text: str,
    word_set: set[str],
    freq_rank: dict[str, int] | None = None,
    *,
    max_rounds: int = 3,
    min_confidence_gap: int = 25,
    max_window_words: int = 3,
) -> TextRepairResult:
    """Try to improve a spaced or unspaced plaintext candidate.

    Strategy:
    1. segment the current text (or respect existing spacing)
    2. apply confident one-edit repairs on pseudo-words
    3. locally re-segment windows of 2-3 words when that reduces cost or
       pseudo-word count

    This is intentionally conservative: it only applies repairs that clearly
    improve the segmentation objective, so callers can use it in both
    automated and agentic paths without the repair becoming a source of noisy
    hallucinations.
    """
    normalized = text.upper().strip()
    if not normalized:
        empty = SegmentResult(text="", segmented="", words=[], dict_rate=0.0, pseudo_words=[], cost=0.0)
        return TextRepairResult(
            original_text=text,
            repaired_text=text,
            before=empty,
            after=empty,
            rounds=0,
            corrections=[],
            applied=False,
            reason="empty_text",
        )

    if any(ch.isspace() for ch in normalized):
        initial_words = [w for w in normalized.split() if w]
        before = SegmentResult(
            text=text,
            segmented=" ".join(initial_words),
            words=initial_words,
            dict_rate=(sum(1 for w in initial_words if w in word_set) / len(initial_words)) if initial_words else 0.0,
            pseudo_words=[w for w in initial_words if w not in word_set],
            cost=_sequence_cost(initial_words, word_set, freq_rank),
        )
    else:
        before = segment_text(normalized, word_set, freq_rank=freq_rank)
        initial_words = list(before.words)

    words = list(initial_words)
    corrections: list[dict[str, str | int]] = []
    rounds_used = 0

    for round_index in range(max_rounds):
        changed = False

        # First pass: conservative one-edit repairs on pseudo-words.
        for idx, word in enumerate(list(words)):
            if word in word_set or len(word) < 3:
                continue
            candidates = sorted(
                find_one_edit_corrections(word, word_set),
                key=lambda item: (freq_rank.get(item[0], 10**9) if freq_rank else 0, item[0]),
            )
            if not candidates:
                continue
            best = candidates[0]
            if len(candidates) > 1 and freq_rank is not None:
                best_rank = freq_rank.get(best[0], 10**9)
                next_rank = freq_rank.get(candidates[1][0], 10**9)
                if next_rank - best_rank < min_confidence_gap:
                    continue
            cand, wrong, correct = best
            words[idx] = cand
            corrections.append({
                "kind": "one_edit",
                "round": round_index + 1,
                "position": idx,
                "from": word,
                "to": cand,
                "wrong_letter": wrong,
                "correct_letter": correct,
            })
            changed = True

        # Second pass: local re-segmentation over suspicious windows.
        i = 0
        while i < len(words):
            window_replaced = False
            for window_size in range(min(max_window_words, len(words) - i), 1, -1):
                window = words[i : i + window_size]
                if all(w in word_set and len(w) > 2 for w in window):
                    continue
                joined = "".join(window)
                candidate = segment_text(joined, word_set, freq_rank=freq_rank)
                current_cost = _sequence_cost(window, word_set, freq_rank)
                current_pseudo = sum(1 for w in window if w not in word_set)
                candidate_pseudo = len(candidate.pseudo_words)
                better = (
                    candidate.cost + 1e-9 < current_cost
                    or (
                        math.isclose(candidate.cost, current_cost)
                        and candidate_pseudo < current_pseudo
                    )
                )
                if not better or candidate.words == window:
                    continue
                words[i : i + window_size] = candidate.words
                corrections.append({
                    "kind": "resegment",
                    "round": round_index + 1,
                    "position": i,
                    "from": " ".join(window),
                    "to": candidate.segmented,
                })
                changed = True
                window_replaced = True
                break
            if window_replaced:
                continue
            i += 1

        rounds_used = round_index + 1
        if not changed:
            break

    repaired_text = " ".join(words)
    after = segment_text("".join(words), word_set, freq_rank=freq_rank) if words else before
    applied = repaired_text != before.segmented and (
        after.dict_rate > before.dict_rate
        or (
            math.isclose(after.dict_rate, before.dict_rate)
            and (
                after.cost < before.cost
                or len(after.pseudo_words) < len(before.pseudo_words)
            )
        )
    )
    if not applied:
        repaired_text = before.segmented
        after = before

    return TextRepairResult(
        original_text=text,
        repaired_text=repaired_text,
        before=before,
        after=after,
        rounds=rounds_used,
        corrections=corrections,
        applied=applied,
        reason=None if applied else ("no_improving_repairs" if corrections else "no_confident_repairs"),
    )
