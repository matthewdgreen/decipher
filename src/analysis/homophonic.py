"""Homophonic substitution search tools.

This module implements the useful core of Zenith's automatic solver in native
Python: each cipher symbol independently maps to a plaintext letter, and a
simulated annealer repeatedly proposes one-symbol remaps scored by a continuous
letter n-gram model.  Unlike the older Session-based annealer, this is built
for many-symbols-to-one-letter ciphers and no-boundary text.
"""
from __future__ import annotations

import csv
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable


ENGLISH_FREQUENCIES: dict[str, float] = {
    "E": 12.02, "T": 9.10, "A": 8.12, "O": 7.68, "I": 7.31, "N": 6.95,
    "S": 6.28, "R": 6.02, "H": 5.92, "D": 4.32, "L": 3.98, "U": 2.88,
    "C": 2.71, "M": 2.61, "F": 2.30, "Y": 2.11, "W": 2.09, "G": 2.03,
    "P": 1.82, "B": 1.49, "V": 1.11, "K": 0.69, "X": 0.17, "Q": 0.11,
    "J": 0.10, "Z": 0.07,
}


@dataclass(frozen=True)
class ContinuousNGramModel:
    order: int
    log_probs: dict[str, float]
    floor: float
    source: str = "unknown"

    def score(self, text: str) -> float:
        if len(text) < self.order:
            return float("-inf")
        total = 0.0
        for i in range(len(text) - self.order + 1):
            total += self.log_probs.get(text[i : i + self.order], self.floor)
        return total


@dataclass
class HomophonicAnnealResult:
    plaintext: str
    key: dict[int, int]
    score: float
    normalized_score: float
    epochs: int
    sampler_iterations: int
    accepted_moves: int
    improved_moves: int
    elapsed_seconds: float
    fixed_symbols: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


def build_continuous_ngram_model(
    words: Iterable[str],
    order: int = 5,
    floor_mass: float = 0.01,
) -> ContinuousNGramModel:
    """Build a continuous-letter n-gram model from a word list.

    The local dictionaries are not full corpora, so this deliberately mixes two
    signals: within-word n-grams and adjacent-word n-grams from the frequency
    ordered word list.  It is still much weaker than Zenith's corpus model, but
    it gives the native annealer a no-boundary objective.
    """
    cleaned_words = [
        "".join(ch for ch in word.upper() if "A" <= ch <= "Z")
        for word in words
    ]
    cleaned_words = [w for w in cleaned_words if w]
    counts: Counter[str] = Counter()

    for word in cleaned_words:
        if len(word) >= order:
            for i in range(len(word) - order + 1):
                counts[word[i : i + order]] += 3

    joined = "".join(cleaned_words)
    if len(joined) >= order:
        for i in range(len(joined) - order + 1):
            counts[joined[i : i + order]] += 1

    total = sum(counts.values())
    if total <= 0:
        floor = math.log(1e-12)
        return ContinuousNGramModel(
            order=order,
            log_probs={},
            floor=floor,
            source="word_list",
        )

    floor = math.log(floor_mass / total)
    log_probs = {gram: math.log(count / total) for gram, count in counts.items()}
    return ContinuousNGramModel(
        order=order,
        log_probs=log_probs,
        floor=floor,
        source="word_list",
    )


def load_zenith_csv_model(
    path: str | Path,
    order: int = 5,
    max_ngrams: int | None = 3_000_000,
) -> ContinuousNGramModel:
    """Load a Zenith-style continuous n-gram CSV model.

    Zenith's release model stores natural-log probabilities by n-gram. We use
    the same data shape without depending on Zenith at runtime: order-1 counts
    determine the unknown floor, and rows at the requested order provide the
    actual language model probabilities.
    """
    model_path = Path(path).expanduser().resolve()
    return _load_zenith_csv_model_cached(str(model_path), order, max_ngrams or 0)


@lru_cache(maxsize=4)
def _load_zenith_csv_model_cached(
    path: str,
    order: int,
    max_ngrams: int,
) -> ContinuousNGramModel:
    log_probs: dict[str, float] = {}
    first_order_total = 0
    loaded = 0

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                row_order = int(row[1])
            except ValueError:
                continue

            if row_order == 1:
                try:
                    first_order_total += int(row[2])
                except ValueError:
                    pass
                continue

            if row_order != order or not row[4]:
                continue

            if max_ngrams and loaded >= max_ngrams:
                continue

            gram = row[0].strip().upper()
            if not gram.isalpha() or len(gram) != order:
                continue
            try:
                log_probs[gram] = float(row[4])
            except ValueError:
                continue
            loaded += 1

    if first_order_total > 0:
        floor = math.log(1.0 / first_order_total)
    elif log_probs:
        floor = min(log_probs.values()) - 5.0
    else:
        floor = math.log(1e-12)

    return ContinuousNGramModel(
        order=order,
        log_probs=log_probs,
        floor=floor,
        source=f"zenith_csv:{path}",
    )


def homophonic_simulated_anneal(
    tokens: list[int],
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    model: ContinuousNGramModel,
    initial_key: dict[int, int] | None = None,
    fixed_cipher_ids: set[int] | None = None,
    epochs: int = 10,
    sampler_iterations: int = 5000,
    t_start: float = 0.012,
    t_end: float = 0.006,
    distribution_weight: float = 4.0,
    seed: int | None = None,
) -> HomophonicAnnealResult:
    """Run symbol-reassignment simulated annealing for homophonic ciphers.

    One sampler iteration visits every cipher symbol once. For each symbol it
    proposes a new plaintext letter sampled from a flattened English frequency
    distribution, updates only n-grams affected by that symbol, and accepts via
    a Boltzmann criterion.
    """
    rng = random.Random(seed)
    t0 = time.time()
    symbol_ids = sorted(set(tokens))
    fixed_cipher_ids = set(fixed_cipher_ids or set())
    mutable_symbol_ids = [sid for sid in symbol_ids if sid not in fixed_cipher_ids]
    occurrences = _occurrence_map(tokens)
    plaintext_letters = [id_to_letter[pid] for pid in plaintext_ids]
    proposal_letters = _proposal_bucket(
        plaintext_letters,
        flatten_weight=0.8,
    )

    best_key: dict[int, int] = {}
    best_chars: list[str] = []
    best_total = float("-inf")
    accepted_moves = 0
    improved_moves = 0
    ngram_count = max(1, len(tokens) - model.order + 1)

    for epoch in range(max(1, epochs)):
        key = _initial_key(
            symbol_ids,
            plaintext_ids,
            id_to_letter,
            letter_to_id,
            proposal_letters,
            rng,
            initial_key if epoch == 0 else None,
        )
        for sid in fixed_cipher_ids:
            if initial_key and sid in initial_key:
                key[sid] = initial_key[sid]

        chars = [id_to_letter[key[token]] for token in tokens]
        window_scores = _initial_window_scores(chars, model)
        char_counts = Counter(chars)
        current_ngram_total = sum(window_scores)
        current_distribution_score = _letter_distribution_score(
            char_counts,
            len(chars),
            plaintext_letters,
        )
        current_total = _combined_score(
            current_ngram_total,
            current_distribution_score,
            ngram_count,
            distribution_weight,
        )

        if current_total > best_total:
            best_total = current_total
            best_key = dict(key)
            best_chars = list(chars)

        temp_span = t_start - t_end
        for step in range(max(1, sampler_iterations)):
            ratio = (sampler_iterations - step) / max(1, sampler_iterations)
            temp = temp_end_safe(t_end + temp_span * ratio)

            for sid in mutable_symbol_ids:
                current_letter = id_to_letter[key[sid]]
                new_letter = rng.choice(proposal_letters)
                if new_letter == current_letter or new_letter not in letter_to_id:
                    continue

                affected = _affected_windows(occurrences[sid], len(chars), model.order)
                old_sum = sum(window_scores[i] for i in affected)
                changed_positions = occurrences[sid]
                for pos in changed_positions:
                    chars[pos] = new_letter
                new_scores = {
                    i: _score_window(chars, i, model)
                    for i in affected
                }
                new_sum = sum(new_scores.values())
                proposal_ngram_total = current_ngram_total - old_sum + new_sum
                proposal_counts = char_counts.copy()
                changed_count = len(changed_positions)
                proposal_counts[current_letter] -= changed_count
                proposal_counts[new_letter] += changed_count
                proposal_distribution_score = _letter_distribution_score(
                    proposal_counts,
                    len(chars),
                    plaintext_letters,
                )
                proposal_total = _combined_score(
                    proposal_ngram_total,
                    proposal_distribution_score,
                    ngram_count,
                    distribution_weight,
                )
                delta = proposal_total - current_total
                delta_normalized = delta / ngram_count

                if delta >= 0 or rng.random() < math.exp(delta_normalized / temp):
                    key[sid] = letter_to_id[new_letter]
                    for i, score in new_scores.items():
                        window_scores[i] = score
                    char_counts = proposal_counts
                    current_ngram_total = proposal_ngram_total
                    current_distribution_score = proposal_distribution_score
                    current_total = proposal_total
                    accepted_moves += 1
                    if delta > 0:
                        improved_moves += 1
                    if current_total > best_total:
                        best_total = current_total
                        best_key = dict(key)
                        best_chars = list(chars)
                else:
                    for pos in changed_positions:
                        chars[pos] = current_letter

    return HomophonicAnnealResult(
        plaintext="".join(best_chars),
        key=best_key,
        score=best_total,
        normalized_score=best_total / ngram_count,
        epochs=max(1, epochs),
        sampler_iterations=max(1, sampler_iterations),
        accepted_moves=accepted_moves,
        improved_moves=improved_moves,
        elapsed_seconds=time.time() - t0,
        fixed_symbols=len(fixed_cipher_ids),
        metadata={
            "order": model.order,
            "mutable_symbols": len(mutable_symbol_ids),
            "cipher_symbols": len(symbol_ids),
            "distribution_weight": distribution_weight,
        },
    )


def temp_end_safe(temp: float) -> float:
    return max(temp, 1e-9)


def _occurrence_map(tokens: list[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, token in enumerate(tokens):
        out.setdefault(token, []).append(i)
    return out


def _proposal_bucket(
    letters: list[str],
    flatten_weight: float = 0.8,
    scale: int = 1000,
) -> list[str]:
    available = [letter for letter in letters if letter in ENGLISH_FREQUENCIES]
    if not available:
        return list(letters)
    flat_mass = (1.0 / len(available)) * flatten_weight
    freq_total = sum(ENGLISH_FREQUENCIES.get(letter, 0.0) for letter in available)
    bucket: list[str] = []
    for letter in available:
        freq_mass = ENGLISH_FREQUENCIES.get(letter, 0.0) / max(freq_total, 1e-9)
        mass = flat_mass + freq_mass * (1.0 - flatten_weight)
        bucket.extend([letter] * max(1, int(scale * mass)))
    return bucket or available


def _initial_key(
    symbol_ids: list[int],
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    proposal_letters: list[str],
    rng: random.Random,
    initial_key: dict[int, int] | None,
) -> dict[int, int]:
    key = dict(initial_key or {})
    fallback = plaintext_ids[0]
    for sid in symbol_ids:
        if sid in key and key[sid] in plaintext_ids:
            continue
        letter = rng.choice(proposal_letters)
        key[sid] = letter_to_id.get(letter, fallback)
    return key


def _initial_window_scores(chars: list[str], model: ContinuousNGramModel) -> list[float]:
    if len(chars) < model.order:
        return []
    return [_score_window(chars, i, model) for i in range(len(chars) - model.order + 1)]


def _score_window(chars: list[str], start: int, model: ContinuousNGramModel) -> float:
    return model.log_probs.get("".join(chars[start : start + model.order]), model.floor)


def _combined_score(
    ngram_total: float,
    distribution_score: float,
    ngram_count: int,
    distribution_weight: float,
) -> float:
    return ngram_total + (ngram_count * distribution_weight * distribution_score)


def _letter_distribution_score(
    counts: Counter[str],
    total: int,
    letters: list[str],
) -> float:
    if total <= 0:
        return 0.0
    available = [letter for letter in letters if letter in ENGLISH_FREQUENCIES]
    if not available:
        return 0.0
    freq_total = sum(ENGLISH_FREQUENCIES[letter] for letter in available)
    chi = 0.0
    for letter in available:
        expected = ENGLISH_FREQUENCIES[letter] / freq_total
        observed = counts.get(letter, 0) / total
        chi += ((observed - expected) ** 2) / max(expected, 1e-9)
    return -chi / len(available)


def _affected_windows(positions: list[int], text_len: int, order: int) -> list[int]:
    starts: set[int] = set()
    max_start = text_len - order
    if max_start < 0:
        return []
    for pos in positions:
        lo = max(0, pos - order + 1)
        hi = min(pos, max_start)
        starts.update(range(lo, hi + 1))
    return sorted(starts)
