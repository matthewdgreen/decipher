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
from typing import Callable, Iterable


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
class HomophonicCandidate:
    plaintext: str
    key: dict[int, int]
    score: float
    normalized_score: float
    epoch: int


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
    candidates: list[HomophonicCandidate] = field(default_factory=list)


@dataclass
class EpochTelemetry:
    epoch: int
    initial_normalized_score: float
    final_normalized_score: float
    best_normalized_score: float
    accepted_moves: int
    improved_moves: int
    unique_letters: int
    top_letter_fraction: float
    index_of_coincidence: float


@dataclass
class MoveTelemetry:
    proposals: int = 0
    accepted: int = 0
    improved: int = 0
    affected_windows: int = 0
    score_time_seconds: float = 0.0


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
    diversity_weight: float = 0.0,
    ioc_weight: float = 0.0,
    score_formula: str = "additive",
    window_step: int = 1,
    move_profile: str = "single_symbol",
    seed: int | None = None,
    top_n: int = 1,
    epoch_callback: Callable[[dict[str, object]], bool] | None = None,
) -> HomophonicAnnealResult:
    """Run symbol-reassignment simulated annealing for homophonic ciphers.

    One sampler iteration visits every cipher symbol once. For each symbol it
    proposes a new plaintext letter sampled from a flattened English frequency
    distribution, updates only n-grams affected by that symbol, and accepts via
    a Boltzmann criterion.
    """
    rng = random.Random(seed)
    t0 = time.time()
    window_step = max(1, int(window_step))
    move_profile = (move_profile or "single_symbol").strip().lower()
    symbol_ids = sorted(set(tokens))
    fixed_cipher_ids = set(fixed_cipher_ids or set())
    mutable_symbol_ids = [sid for sid in symbol_ids if sid not in fixed_cipher_ids]
    occurrences = _occurrence_map(tokens)
    plaintext_letters = [id_to_letter[pid] for pid in plaintext_ids]
    proposal_letters = _proposal_bucket(
        plaintext_letters,
        flatten_weight=0.8,
    )
    target_refresh_interval = 500

    best_key: dict[int, int] = {}
    best_chars: list[str] = []
    best_total = float("-inf")
    accepted_moves = 0
    improved_moves = 0
    ngram_count = max(1, len(_window_starts(len(tokens), model.order, window_step)))
    candidate_pool: list[HomophonicCandidate] = []
    epoch_traces: list[EpochTelemetry] = []
    move_telemetry: dict[str, MoveTelemetry] = {}
    stopped_early = False
    stopped_after_epoch = 0
    stop_reason: str | None = None

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
        window_scores = _initial_window_scores(chars, model, step=window_step)
        char_counts = Counter(chars)
        current_ngram_total = sum(window_scores)
        current_distribution_score = _letter_distribution_score(
            char_counts,
            len(chars),
            plaintext_letters,
        )
        current_diversity_score = _letter_diversity_score(
            char_counts,
            len(chars),
            plaintext_letters,
        )
        current_ioc_score = _index_of_coincidence_score(char_counts, len(chars))
        current_total = _combined_score(
            current_ngram_total,
            current_distribution_score,
            current_diversity_score,
            current_ioc_score,
            ngram_count,
            distribution_weight,
            diversity_weight,
            ioc_weight,
            score_formula=score_formula,
        )
        epoch_initial_total = current_total
        epoch_best_total = current_total
        epoch_best_key = dict(key)
        epoch_best_chars = list(chars)
        epoch_accepted_moves = 0
        epoch_improved_moves = 0

        if current_total > best_total:
            best_total = current_total
            best_key = dict(key)
            best_chars = list(chars)

        temp_span = t_start - t_end
        targeted_symbol_ids = _targeted_symbol_ids(
            mutable_symbol_ids,
            occurrences,
            window_scores,
            len(chars),
            model.order,
            window_step,
        )
        for step in range(max(1, sampler_iterations)):
            ratio = (sampler_iterations - step) / max(1, sampler_iterations)
            temp = temp_end_safe(t_end + temp_span * ratio)
            if (
                move_profile == "mixed_v1_targeted"
                and step > 0
                and step % target_refresh_interval == 0
            ):
                targeted_symbol_ids = _targeted_symbol_ids(
                    mutable_symbol_ids,
                    occurrences,
                    window_scores,
                    len(chars),
                    model.order,
                    window_step,
                )

            for sid in mutable_symbol_ids:
                proposal_kind, proposal = _propose_homophonic_move(
                    sid,
                    mutable_symbol_ids,
                    key,
                    id_to_letter,
                    letter_to_id,
                    proposal_letters,
                    rng,
                    move_profile=move_profile,
                    targeted_symbol_ids=targeted_symbol_ids,
                )
                if proposal is None:
                    continue
                stats = move_telemetry.setdefault(proposal_kind, MoveTelemetry())
                stats.proposals += 1
                affected_positions: list[int] = []
                old_letters: dict[int, str] = {}
                for changed_sid in proposal:
                    affected_positions.extend(occurrences[changed_sid])
                    old_letters[changed_sid] = id_to_letter[key[changed_sid]]

                affected = _affected_windows(
                    affected_positions,
                    len(chars),
                    model.order,
                    step=window_step,
                )
                stats.affected_windows += len(affected)
                old_sum = sum(window_scores[i] for i in affected)
                proposal_counts = char_counts.copy()
                for changed_sid, new_letter in proposal.items():
                    changed_positions = occurrences[changed_sid]
                    for pos in changed_positions:
                        chars[pos] = new_letter
                    proposal_counts[old_letters[changed_sid]] -= len(changed_positions)
                    proposal_counts[new_letter] += len(changed_positions)
                score_started = time.perf_counter()
                new_scores = {
                    i: _score_window(chars, i, model)
                    for i in affected
                }
                new_sum = sum(new_scores.values())
                proposal_ngram_total = current_ngram_total - old_sum + new_sum
                proposal_distribution_score = _letter_distribution_score(
                    proposal_counts,
                    len(chars),
                    plaintext_letters,
                )
                proposal_diversity_score = _letter_diversity_score(
                    proposal_counts,
                    len(chars),
                    plaintext_letters,
                )
                proposal_ioc_score = _index_of_coincidence_score(
                    proposal_counts,
                    len(chars),
                )
                stats.score_time_seconds += time.perf_counter() - score_started
                proposal_total = _combined_score(
                    proposal_ngram_total,
                    proposal_distribution_score,
                    proposal_diversity_score,
                    proposal_ioc_score,
                    ngram_count,
                    distribution_weight,
                    diversity_weight,
                    ioc_weight,
                    score_formula=score_formula,
                )
                delta = proposal_total - current_total
                delta_normalized = delta / ngram_count

                if delta >= 0 or rng.random() < math.exp(delta_normalized / temp):
                    for changed_sid, new_letter in proposal.items():
                        key[changed_sid] = letter_to_id[new_letter]
                    for i, score in new_scores.items():
                        window_scores[i] = score
                    char_counts = proposal_counts
                    current_ngram_total = proposal_ngram_total
                    current_distribution_score = proposal_distribution_score
                    current_diversity_score = proposal_diversity_score
                    current_ioc_score = proposal_ioc_score
                    current_total = proposal_total
                    stats.accepted += 1
                    accepted_moves += 1
                    epoch_accepted_moves += 1
                    if delta > 0:
                        stats.improved += 1
                        improved_moves += 1
                        epoch_improved_moves += 1
                    if current_total > best_total:
                        best_total = current_total
                        best_key = dict(key)
                        best_chars = list(chars)
                    if current_total > epoch_best_total:
                        epoch_best_total = current_total
                        epoch_best_key = dict(key)
                        epoch_best_chars = list(chars)
                else:
                    for changed_sid, old_letter in old_letters.items():
                        for pos in occurrences[changed_sid]:
                            chars[pos] = old_letter

        candidate_pool.append(HomophonicCandidate(
            plaintext="".join(epoch_best_chars),
            key=epoch_best_key,
            score=epoch_best_total,
            normalized_score=epoch_best_total / ngram_count,
            epoch=epoch + 1,
        ))
        epoch_stats = _plaintext_stats(epoch_best_chars, len(plaintext_letters))
        epoch_traces.append(EpochTelemetry(
            epoch=epoch + 1,
            initial_normalized_score=epoch_initial_total / ngram_count,
            final_normalized_score=current_total / ngram_count,
            best_normalized_score=epoch_best_total / ngram_count,
            accepted_moves=epoch_accepted_moves,
            improved_moves=epoch_improved_moves,
            unique_letters=epoch_stats["unique_letters"],
            top_letter_fraction=epoch_stats["top_letter_fraction"],
            index_of_coincidence=epoch_stats["index_of_coincidence"],
        ))
        if epoch_callback is not None:
            callback_payload = {
                "epoch": epoch + 1,
                "plaintext": "".join(epoch_best_chars),
                "key": dict(epoch_best_key),
                "normalized_score": epoch_best_total / ngram_count,
                "final_normalized_score": current_total / ngram_count,
                "accepted_moves": epoch_accepted_moves,
                "improved_moves": epoch_improved_moves,
                "epoch_trace": epoch_traces[-1].__dict__,
            }
            callback_result = epoch_callback(callback_payload)
            if callback_result:
                stopped_early = True
                stopped_after_epoch = epoch + 1
                stop_reason = "epoch_callback"
                break

    candidate_pool.append(HomophonicCandidate(
        plaintext="".join(best_chars),
        key=best_key,
        score=best_total,
        normalized_score=best_total / ngram_count,
        epoch=0,
    ))
    candidates = _top_candidates(candidate_pool, max(1, top_n))

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
            "diversity_weight": diversity_weight,
            "ioc_weight": ioc_weight,
            "score_formula": score_formula,
            "window_step": window_step,
            "move_profile": move_profile,
            "target_refresh_interval": target_refresh_interval if move_profile == "mixed_v1_targeted" else None,
            "move_telemetry": {
                kind: {
                    "proposals": stats.proposals,
                    "accepted": stats.accepted,
                    "improved": stats.improved,
                    "affected_windows": stats.affected_windows,
                    "avg_affected_windows": round(
                        stats.affected_windows / max(1, stats.proposals),
                        3,
                    ),
                    "score_time_seconds": round(stats.score_time_seconds, 6),
                    "avg_score_time_ms": round(
                        (stats.score_time_seconds * 1000.0) / max(1, stats.proposals),
                        6,
                    ),
                }
                for kind, stats in sorted(move_telemetry.items())
            },
            "epoch_traces": [trace.__dict__ for trace in epoch_traces],
            "stopped_early": stopped_early,
            "stopped_after_epoch": stopped_after_epoch if stopped_early else None,
            "stop_reason": stop_reason,
        },
        candidates=candidates,
    )


def substitution_simulated_anneal(
    tokens: list[int],
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    model: ContinuousNGramModel,
    initial_key: dict[int, int] | None = None,
    epochs: int = 20,
    sampler_iterations: int = 8000,
    t_start: float = 0.018,
    t_end: float = 0.003,
    distribution_weight: float = 1.0,
    ioc_weight: float = 0.0,
    seed: int | None = None,
    top_n: int = 1,
) -> HomophonicAnnealResult:
    """Run bijective substitution annealing over continuous text.

    Unlike ``homophonic_simulated_anneal``, this keeps cipher-symbol mappings
    one-to-one. Moves either swap two currently assigned plaintext letters or
    replace one assignment with a currently unused plaintext letter, which is
    essential when the ciphertext uses fewer than 26 symbols.
    """
    rng = random.Random(seed)
    t0 = time.time()
    symbol_ids = sorted(set(tokens))
    occurrences = _occurrence_map(tokens)
    plaintext_letters = [id_to_letter[pid] for pid in plaintext_ids]

    best_key: dict[int, int] = {}
    best_chars: list[str] = []
    best_total = float("-inf")
    accepted_moves = 0
    improved_moves = 0
    ngram_count = max(1, len(tokens) - model.order + 1)
    candidate_pool: list[HomophonicCandidate] = []
    epoch_traces: list[EpochTelemetry] = []

    for epoch in range(max(1, epochs)):
        key = _initial_bijective_key(
            symbol_ids,
            plaintext_ids,
            rng,
            initial_key if epoch == 0 else None,
        )
        chars = [id_to_letter[key[token]] for token in tokens]
        window_scores = _initial_window_scores(chars, model)
        char_counts = Counter(chars)
        current_ngram_total = sum(window_scores)
        current_distribution_score = _letter_distribution_score(
            char_counts,
            len(chars),
            plaintext_letters,
        )
        current_ioc_score = _index_of_coincidence_score(char_counts, len(chars))
        current_total = _combined_score(
            current_ngram_total,
            current_distribution_score,
            0.0,
            current_ioc_score,
            ngram_count,
            distribution_weight,
            0.0,
            ioc_weight,
        )
        epoch_initial_total = current_total
        epoch_best_total = current_total
        epoch_best_key = dict(key)
        epoch_best_chars = list(chars)
        epoch_accepted_moves = 0
        epoch_improved_moves = 0

        if current_total > best_total:
            best_total = current_total
            best_key = dict(key)
            best_chars = list(chars)

        temp_span = t_start - t_end
        for step in range(max(1, sampler_iterations)):
            ratio = (sampler_iterations - step) / max(1, sampler_iterations)
            temp = temp_end_safe(t_end + temp_span * ratio)

            if len(symbol_ids) >= 2 and rng.random() < 0.75:
                sid_a, sid_b = rng.sample(symbol_ids, 2)
                old_a, old_b = key[sid_a], key[sid_b]
                proposal = {sid_a: old_b, sid_b: old_a}
            else:
                used = set(key.values())
                unused = [pid for pid in plaintext_ids if pid not in used]
                if not unused:
                    continue
                sid_a = rng.choice(symbol_ids)
                old_a = key[sid_a]
                new_a = rng.choice(unused)
                if new_a == old_a:
                    continue
                proposal = {sid_a: new_a}

            affected_positions: list[int] = []
            old_letters: dict[int, str] = {}
            for sid in proposal:
                affected_positions.extend(occurrences[sid])
                old_letters[sid] = id_to_letter[key[sid]]
            affected = _affected_windows(affected_positions, len(chars), model.order)
            old_sum = sum(window_scores[i] for i in affected)

            for sid, new_pt in proposal.items():
                new_letter = id_to_letter[new_pt]
                for pos in occurrences[sid]:
                    chars[pos] = new_letter

            new_scores = {i: _score_window(chars, i, model) for i in affected}
            new_sum = sum(new_scores.values())
            proposal_ngram_total = current_ngram_total - old_sum + new_sum

            proposal_counts = char_counts.copy()
            for sid, new_pt in proposal.items():
                old_letter = old_letters[sid]
                new_letter = id_to_letter[new_pt]
                changed_count = len(occurrences[sid])
                proposal_counts[old_letter] -= changed_count
                proposal_counts[new_letter] += changed_count
            proposal_distribution_score = _letter_distribution_score(
                proposal_counts,
                len(chars),
                plaintext_letters,
            )
            proposal_ioc_score = _index_of_coincidence_score(
                proposal_counts,
                len(chars),
            )
            proposal_total = _combined_score(
                proposal_ngram_total,
                proposal_distribution_score,
                0.0,
                proposal_ioc_score,
                ngram_count,
                distribution_weight,
                0.0,
                ioc_weight,
            )
            delta = proposal_total - current_total
            delta_normalized = delta / ngram_count

            if delta >= 0 or rng.random() < math.exp(delta_normalized / temp):
                for sid, new_pt in proposal.items():
                    key[sid] = new_pt
                for i, score in new_scores.items():
                    window_scores[i] = score
                char_counts = proposal_counts
                current_ngram_total = proposal_ngram_total
                current_distribution_score = proposal_distribution_score
                current_ioc_score = proposal_ioc_score
                current_total = proposal_total
                accepted_moves += 1
                epoch_accepted_moves += 1
                if delta > 0:
                    improved_moves += 1
                    epoch_improved_moves += 1
                if current_total > best_total:
                    best_total = current_total
                    best_key = dict(key)
                    best_chars = list(chars)
                if current_total > epoch_best_total:
                    epoch_best_total = current_total
                    epoch_best_key = dict(key)
                    epoch_best_chars = list(chars)
            else:
                for sid in proposal:
                    old_letter = old_letters[sid]
                    for pos in occurrences[sid]:
                        chars[pos] = old_letter

        candidate_pool.append(HomophonicCandidate(
            plaintext="".join(epoch_best_chars),
            key=epoch_best_key,
            score=epoch_best_total,
            normalized_score=epoch_best_total / ngram_count,
            epoch=epoch + 1,
        ))
        epoch_stats = _plaintext_stats(epoch_best_chars, len(plaintext_letters))
        epoch_traces.append(EpochTelemetry(
            epoch=epoch + 1,
            initial_normalized_score=epoch_initial_total / ngram_count,
            final_normalized_score=current_total / ngram_count,
            best_normalized_score=epoch_best_total / ngram_count,
            accepted_moves=epoch_accepted_moves,
            improved_moves=epoch_improved_moves,
            unique_letters=epoch_stats["unique_letters"],
            top_letter_fraction=epoch_stats["top_letter_fraction"],
            index_of_coincidence=epoch_stats["index_of_coincidence"],
        ))

    candidate_pool.append(HomophonicCandidate(
        plaintext="".join(best_chars),
        key=best_key,
        score=best_total,
        normalized_score=best_total / ngram_count,
        epoch=0,
    ))
    candidates = _top_candidates(candidate_pool, max(1, top_n))

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
        metadata={
            "order": model.order,
            "cipher_symbols": len(symbol_ids),
            "distribution_weight": distribution_weight,
            "ioc_weight": ioc_weight,
            "bijective": True,
            "epoch_traces": [trace.__dict__ for trace in epoch_traces],
        },
        candidates=candidates,
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


def _propose_homophonic_move(
    sid: int,
    mutable_symbol_ids: list[int],
    key: dict[int, int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    proposal_letters: list[str],
    rng: random.Random,
    move_profile: str = "single_symbol",
    targeted_symbol_ids: set[int] | None = None,
) -> tuple[str, dict[int, str] | None]:
    current_letter = id_to_letter[key[sid]]

    if move_profile == "single_symbol":
        new_letter = rng.choice(proposal_letters)
        if new_letter == current_letter or new_letter not in letter_to_id:
            return "single", None
        return "single", {sid: new_letter}

    if move_profile == "mixed_v1":
        roll = rng.random()
        if roll < 0.12 and len(mutable_symbol_ids) >= 2:
            others = [other for other in mutable_symbol_ids if other != sid]
            if others:
                sid_b = rng.choice(others)
                letter_a = rng.choice(proposal_letters)
                letter_b = rng.choice(proposal_letters)
                current_b = id_to_letter[key[sid_b]]
                if (
                    (letter_a == current_letter and letter_b == current_b)
                    or letter_a not in letter_to_id
                    or letter_b not in letter_to_id
                ):
                    return "double_reassign", None
                return "double_reassign", {sid: letter_a, sid_b: letter_b}
        if roll < 0.22 and len(mutable_symbol_ids) >= 2:
            others = [other for other in mutable_symbol_ids if other != sid]
            if others:
                sid_b = rng.choice(others)
                current_b = id_to_letter[key[sid_b]]
                if current_b != current_letter:
                    return "swap", {sid: current_b, sid_b: current_letter}
        new_letter = rng.choice(proposal_letters)
        if new_letter == current_letter or new_letter not in letter_to_id:
            return "single", None
        return "single", {sid: new_letter}

    if move_profile == "mixed_v1_targeted":
        if targeted_symbol_ids is None or sid not in targeted_symbol_ids:
            new_letter = rng.choice(proposal_letters)
            if new_letter == current_letter or new_letter not in letter_to_id:
                return "single", None
            return "single", {sid: new_letter}
        roll = rng.random()
        if roll < 0.12 and len(mutable_symbol_ids) >= 2:
            others = [other for other in mutable_symbol_ids if other != sid]
            if others:
                sid_b = rng.choice(others)
                letter_a = rng.choice(proposal_letters)
                letter_b = rng.choice(proposal_letters)
                current_b = id_to_letter[key[sid_b]]
                if (
                    (letter_a == current_letter and letter_b == current_b)
                    or letter_a not in letter_to_id
                    or letter_b not in letter_to_id
                ):
                    return "double_reassign", None
                return "double_reassign", {sid: letter_a, sid_b: letter_b}
        if roll < 0.22 and len(mutable_symbol_ids) >= 2:
            others = [other for other in mutable_symbol_ids if other != sid]
            if others:
                sid_b = rng.choice(others)
                current_b = id_to_letter[key[sid_b]]
                if current_b != current_letter:
                    return "swap", {sid: current_b, sid_b: current_letter}
        new_letter = rng.choice(proposal_letters)
        if new_letter == current_letter or new_letter not in letter_to_id:
            return "single", None
        return "single", {sid: new_letter}

    if move_profile == "mixed_v2":
        roll = rng.random()
        if roll < 0.1 and len(mutable_symbol_ids) >= 3:
            others = [other for other in mutable_symbol_ids if other != sid]
            if len(others) >= 2:
                sid_b = rng.choice(others)
                others = [other for other in others if other != sid_b]
                sid_c = rng.choice(others)
                letter_a = rng.choice(proposal_letters)
                letter_b = rng.choice(proposal_letters)
                letter_c = rng.choice(proposal_letters)
                current_b = id_to_letter[key[sid_b]]
                current_c = id_to_letter[key[sid_c]]
                if (
                    (letter_a == current_letter and letter_b == current_b and letter_c == current_c)
                    or letter_a not in letter_to_id
                    or letter_b not in letter_to_id
                    or letter_c not in letter_to_id
                ):
                    return "triple_reassign", None
                return "triple_reassign", {sid: letter_a, sid_b: letter_b, sid_c: letter_c}
        if roll < 0.22 and len(mutable_symbol_ids) >= 2:
            others = [other for other in mutable_symbol_ids if other != sid]
            if others:
                sid_b = rng.choice(others)
                shared_letter = rng.choice(proposal_letters)
                current_b = id_to_letter[key[sid_b]]
                if (
                    shared_letter not in letter_to_id
                    or (shared_letter == current_letter and shared_letter == current_b)
                ):
                    return "merge", None
                return "merge", {sid: shared_letter, sid_b: shared_letter}
        if roll < 0.32 and len(mutable_symbol_ids) >= 2:
            others = [other for other in mutable_symbol_ids if other != sid]
            if others:
                sid_b = rng.choice(others)
                current_b = id_to_letter[key[sid_b]]
                if current_b != current_letter:
                    return "swap", {sid: current_b, sid_b: current_letter}
        if roll < 0.42 and len(mutable_symbol_ids) >= 2:
            others = [other for other in mutable_symbol_ids if other != sid]
            if others:
                sid_b = rng.choice(others)
                letter_a = rng.choice(proposal_letters)
                letter_b = rng.choice(proposal_letters)
                current_b = id_to_letter[key[sid_b]]
                if (
                    (letter_a == current_letter and letter_b == current_b)
                    or letter_a not in letter_to_id
                    or letter_b not in letter_to_id
                ):
                    return "double_reassign", None
                return "double_reassign", {sid: letter_a, sid_b: letter_b}
        new_letter = rng.choice(proposal_letters)
        if new_letter == current_letter or new_letter not in letter_to_id:
            return "single", None
        return "single", {sid: new_letter}

    raise ValueError(
        f"unsupported homophonic move profile '{move_profile}' "
        "(expected one of: single_symbol, mixed_v1, mixed_v1_targeted, mixed_v2)"
    )


def _targeted_symbol_ids(
    mutable_symbol_ids: list[int],
    occurrences: dict[int, list[int]],
    window_scores: list[float],
    text_len: int,
    order: int,
    window_step: int,
) -> set[int]:
    if not mutable_symbol_ids or not window_scores:
        return set(mutable_symbol_ids)
    scored: list[tuple[float, int]] = []
    for sid in mutable_symbol_ids:
        affected = _affected_windows(
            occurrences[sid],
            text_len,
            order,
            step=window_step,
        )
        if not affected:
            continue
        avg_score = sum(window_scores[i] for i in affected) / len(affected)
        scored.append((avg_score, sid))
    if not scored:
        return set(mutable_symbol_ids)
    scored.sort()
    target_count = max(4, len(scored) // 3)
    return {sid for _, sid in scored[:target_count]}


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


def _initial_bijective_key(
    symbol_ids: list[int],
    plaintext_ids: list[int],
    rng: random.Random,
    initial_key: dict[int, int] | None,
) -> dict[int, int]:
    key: dict[int, int] = {}
    used: set[int] = set()
    if initial_key:
        for sid in symbol_ids:
            pt_id = initial_key.get(sid)
            if pt_id in plaintext_ids and pt_id not in used:
                key[sid] = pt_id
                used.add(pt_id)

    unused = [pid for pid in plaintext_ids if pid not in used]
    rng.shuffle(unused)
    for sid in symbol_ids:
        if sid in key:
            continue
        if not unused:
            raise ValueError("not enough plaintext ids for bijective key")
        key[sid] = unused.pop()
    return key


def _window_starts(text_len: int, order: int, step: int = 1) -> list[int]:
    if text_len < order:
        return []
    max_start = text_len - order
    starts = list(range(0, max_start + 1, max(1, step)))
    return starts or [0]


def _initial_window_scores(
    chars: list[str],
    model: ContinuousNGramModel,
    step: int = 1,
) -> list[float]:
    starts = _window_starts(len(chars), model.order, step=step)
    if not starts:
        return []
    return [_score_window(chars, start, model) for start in starts]


def _score_window(chars: list[str], start: int, model: ContinuousNGramModel) -> float:
    return model.log_probs.get("".join(chars[start : start + model.order]), model.floor)


def _combined_score(
    ngram_total: float,
    distribution_score: float,
    diversity_score: float,
    ioc_score: float,
    ngram_count: int,
    distribution_weight: float,
    diversity_weight: float = 0.0,
    ioc_weight: float = 0.0,
    score_formula: str = "additive",
) -> float:
    if score_formula == "multiplicative_ioc":
        base = (ngram_total / max(1, ngram_count)) * (
            ioc_score ** ioc_weight if ioc_weight > 0 else 1.0
        )
        return base + (
            distribution_weight * distribution_score
            + diversity_weight * diversity_score
        )
    if score_formula != "additive":
        raise ValueError(
            f"unsupported score formula '{score_formula}' "
            "(expected one of: additive, multiplicative_ioc)"
        )
    return ngram_total + (
        ngram_count
        * (
            distribution_weight * distribution_score
            + diversity_weight * diversity_score
            + ioc_weight * ioc_score
        )
    )


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


def _letter_diversity_score(
    counts: Counter[str],
    total: int,
    letters: list[str],
) -> float:
    if total < 50:
        return 0.0
    available = [letter for letter in letters if letter in ENGLISH_FREQUENCIES]
    if not available:
        return 0.0
    unique = sum(1 for letter in available if counts.get(letter, 0) > 0)
    top_fraction = max(counts.values()) / total if counts else 0.0
    if total >= 350:
        target_unique = min(18, len(available))
        max_top_fraction = 0.18
    elif total >= 150:
        target_unique = min(15, len(available))
        max_top_fraction = 0.22
    else:
        target_unique = min(11, len(available))
        max_top_fraction = 0.30

    penalty = 0.0
    if unique < target_unique:
        penalty += ((target_unique - unique) / max(1, target_unique)) ** 2
    if top_fraction > max_top_fraction:
        penalty += ((top_fraction - max_top_fraction) / max_top_fraction) ** 2
    return -penalty


def _index_of_coincidence_score(
    counts: Counter[str],
    total: int,
) -> float:
    if total <= 1:
        return 0.0
    numerator = sum(count * (count - 1) for count in counts.values())
    denominator = total * (total - 1)
    ic_value = numerator / denominator if denominator > 0 else 0.0
    return ic_value ** (1.0 / 6.0) if ic_value > 0 else 0.0


def _plaintext_stats(chars: list[str], alphabet_size: int) -> dict[str, float | int]:
    total = len(chars)
    if total <= 0:
        return {
            "unique_letters": 0,
            "top_letter_fraction": 0.0,
            "index_of_coincidence": 0.0,
        }
    counts = Counter(chars)
    numerator = sum(count * (count - 1) for count in counts.values())
    denominator = total * (total - 1) if total > 1 else 1
    return {
        "unique_letters": len(counts),
        "top_letter_fraction": max(counts.values()) / total,
        "index_of_coincidence": numerator / denominator if denominator > 0 else 0.0,
    }


def _affected_windows(
    positions: list[int],
    text_len: int,
    order: int,
    step: int = 1,
) -> list[int]:
    starts: set[int] = set()
    valid_starts = _window_starts(text_len, order, step=step)
    if not valid_starts:
        return []
    valid_start_set = set(valid_starts)
    for pos in positions:
        lo = max(0, pos - order + 1)
        hi = min(pos, valid_starts[-1])
        for start in range(lo, hi + 1):
            if start in valid_start_set:
                starts.add(start // max(1, step))
    return sorted(starts)


def _top_candidates(
    candidates: list[HomophonicCandidate],
    top_n: int,
) -> list[HomophonicCandidate]:
    by_plaintext: dict[str, HomophonicCandidate] = {}
    for candidate in candidates:
        existing = by_plaintext.get(candidate.plaintext)
        if existing is None or candidate.score > existing.score:
            by_plaintext[candidate.plaintext] = candidate
    return sorted(
        by_plaintext.values(),
        key=lambda c: c.score,
        reverse=True,
    )[:top_n]
