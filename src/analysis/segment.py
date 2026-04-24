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
from dataclasses import dataclass, field
from typing import Callable


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


@dataclass
class KeyRepairResult:
    """Outcome of a key-consistent (symbol-level) dictionary repair pass.

    Unlike :class:`TextRepairResult`, which only rewrites the decoded string,
    this result reports changes that have been propagated back through the
    cipher key, so every occurrence of the corrected cipher symbol reflects
    the new letter assignment.
    """

    original_key: dict[int, int]
    repaired_key: dict[int, int]
    before_plaintext: str
    after_plaintext: str
    before_hits: int
    after_hits: int
    before_words: int
    after_words: int
    rounds: int
    corrections: list[dict[str, object]] = field(default_factory=list)
    applied: bool = False
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


# ---------------------------------------------------------------------------
# Key-consistent repair
# ---------------------------------------------------------------------------

def _decode_words(
    cipher_words: list[list[int]],
    key: dict[int, int],
    id_to_letter: dict[int, str],
) -> list[str]:
    """Return the list of upper-cased plaintext words for the given key.

    Unknown mappings render as ``?`` so they don't accidentally collide with
    dictionary entries.
    """
    out: list[str] = []
    for word_tokens in cipher_words:
        if not word_tokens:
            continue
        letters: list[str] = []
        for tok in word_tokens:
            pt_id = key.get(tok)
            if pt_id is None:
                letters.append("?")
                continue
            letter = id_to_letter.get(pt_id, "?")
            letters.append(letter if letter else "?")
        out.append("".join(letters).upper())
    return out


def _dict_hits(words: list[str], word_set: set[str]) -> int:
    return sum(1 for w in words if w in word_set)


def repair_key_with_dictionary(
    cipher_words: list[list[int]],
    key: dict[int, int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    word_set: set[str],
    *,
    freq_rank: dict[str, int] | None = None,
    max_rounds: int = 6,
    min_word_len: int = 3,
    require_strict_improvement: bool = True,
    protected_symbols: set[int] | None = None,
    score_fn: Callable[[dict[int, int]], float] | None = None,
    max_score_drop: float = 0.0,
) -> KeyRepairResult:
    """Improve a homophonic key by propagating one-edit dictionary repairs.

    The existing :func:`repair_no_boundary_text` operates on the decoded string
    and cannot exploit the fact that fixing one wrong letter usually fixes
    **every** occurrence of the underlying cipher symbol. This helper does the
    opposite: it proposes one-letter substitutions on pseudo-words, projects
    each candidate back onto the cipher key, re-decodes the whole ciphertext,
    and keeps the change only if it strictly increases dictionary coverage
    (and, optionally, doesn't crater an external 5-gram score).

    Parameters
    ----------
    cipher_words:
        Tokens grouped by ciphertext word boundaries (``CipherText.words``).
        The cipher is assumed to already be word-delimited; feeding flat
        no-boundary tokens here would degrade to a single pseudo-word.
    key:
        Current cipher-symbol → plaintext-id mapping. Not mutated.
    id_to_letter / letter_to_id:
        Plaintext alphabet mappings. Letters are A–Z uppercase.
    word_set:
        Dictionary to check word membership against. Must be uppercase.
    freq_rank:
        Optional Zipf-style rank table for tie-breaking between correction
        candidates. When provided, lower rank (more common) wins.
    max_rounds:
        Upper bound on passes through the word list.
    min_word_len:
        Skip pseudo-words shorter than this — too many spurious matches.
    require_strict_improvement:
        When True (default), a candidate is only accepted when dictionary
        hit count strictly increases. Setting False allows equal-hit
        alternatives that might unlock future rounds, at the cost of risk.
    protected_symbols:
        Cipher symbols whose mappings must not change — useful for anchor
        words the caller trusts.
    score_fn:
        Optional callable returning a language-model score for a given key.
        Used as a guard: a repair is rejected if its score drops by more than
        ``max_score_drop`` relative to the current key, even when dictionary
        hits improve.
    max_score_drop:
        Tolerance for ``score_fn`` drops. 0.0 requires the score to be
        non-decreasing; set higher (e.g. 0.02) to allow small trade-offs.
    """
    protected = set(protected_symbols or ())

    current_key = dict(key)
    before_words = _decode_words(cipher_words, current_key, id_to_letter)
    before_hits = _dict_hits(before_words, word_set)
    before_plaintext = " ".join(before_words)

    corrections: list[dict[str, object]] = []

    current_score: float | None = None
    if score_fn is not None:
        try:
            current_score = float(score_fn(current_key))
        except Exception:  # noqa: BLE001 - treat scorer failure as "no guard"
            current_score = None

    rounds_used = 0
    for round_index in range(max_rounds):
        words_now = _decode_words(cipher_words, current_key, id_to_letter)
        hits_now = _dict_hits(words_now, word_set)

        changed = False
        for word_index, word in enumerate(words_now):
            if len(word) < min_word_len or word in word_set:
                continue
            # Candidate list: every single-substitution that lands in the dict.
            candidates = find_one_edit_corrections(word, word_set)
            if not candidates:
                continue
            # Deterministic ordering: prefer more common words first.
            candidates.sort(
                key=lambda item: (
                    freq_rank.get(item[0], 10**9) if freq_rank else 0,
                    item[0],
                )
            )

            accepted_candidate: tuple[str, str, str, int] | None = None
            for cand_word, wrong_letter, correct_letter in candidates:
                # Find the position within the cipher word where the
                # substitution applies.
                try:
                    pos = word.index(wrong_letter)
                except ValueError:
                    continue
                cipher_word = cipher_words[word_index]
                if pos >= len(cipher_word):
                    continue
                cipher_sym = cipher_word[pos]
                if cipher_sym in protected:
                    continue
                new_letter_id = letter_to_id.get(correct_letter.upper())
                if new_letter_id is None:
                    continue
                if current_key.get(cipher_sym) == new_letter_id:
                    continue

                trial_key = dict(current_key)
                trial_key[cipher_sym] = new_letter_id

                trial_words = _decode_words(cipher_words, trial_key, id_to_letter)
                trial_hits = _dict_hits(trial_words, word_set)

                if require_strict_improvement:
                    if trial_hits <= hits_now:
                        continue
                else:
                    if trial_hits < hits_now:
                        continue

                if score_fn is not None:
                    try:
                        trial_score = float(score_fn(trial_key))
                    except Exception:  # noqa: BLE001
                        trial_score = None
                    if trial_score is not None and current_score is not None:
                        if trial_score < current_score - max_score_drop:
                            continue
                        tentative_score = trial_score
                    else:
                        tentative_score = current_score
                else:
                    tentative_score = current_score

                # Accept.
                current_key = trial_key
                current_score = tentative_score
                hits_now = trial_hits
                words_now = trial_words
                corrections.append({
                    "kind": "symbol_remap",
                    "round": round_index + 1,
                    "word_index": word_index,
                    "from_word": word,
                    "to_word": cand_word,
                    "wrong_letter": wrong_letter,
                    "correct_letter": correct_letter,
                    "cipher_symbol": cipher_sym,
                    "dict_hits_after": trial_hits,
                })
                accepted_candidate = (cand_word, wrong_letter, correct_letter, cipher_sym)
                changed = True
                break

            if accepted_candidate is not None:
                # Re-decode is done; continue scanning words in this round so
                # other pseudo-words can also be fixed.
                continue

        rounds_used = round_index + 1
        if not changed:
            break

    after_words = _decode_words(cipher_words, current_key, id_to_letter)
    after_hits = _dict_hits(after_words, word_set)
    after_plaintext = " ".join(after_words)

    applied = after_hits > before_hits and current_key != key
    reason: str | None = None
    if not applied:
        if not corrections:
            reason = "no_confident_repairs"
        else:
            # Corrections happened but net dict_hits didn't improve — revert.
            reason = "no_net_dict_gain"
            current_key = dict(key)
            after_words = before_words
            after_hits = before_hits
            after_plaintext = before_plaintext

    return KeyRepairResult(
        original_key=dict(key),
        repaired_key=current_key,
        before_plaintext=before_plaintext,
        after_plaintext=after_plaintext,
        before_hits=before_hits,
        after_hits=after_hits,
        before_words=len(before_words),
        after_words=len(after_words),
        rounds=rounds_used,
        corrections=corrections,
        applied=applied,
        reason=reason,
    )
