"""Algorithmic solvers for substitution ciphers.

Works for both simple (bijective) and homophonic ciphers without assuming
which kind the cipher is. The optimizer assigns each cipher symbol to the
best plaintext letter independently, so many-to-one mappings emerge
naturally when the scoring function favours them.
"""
from __future__ import annotations

import random
from typing import Callable

from analysis import dictionary, frequency, pattern
from models.session import Session


def hill_climb_reassign(
    session: Session,
    score_fn: Callable[[], float],
    max_rounds: int = 50,
) -> float:
    """Improve the current key by trying all 26 reassignments per cipher symbol.

    Each round: for every cipher symbol, try replacing its current plaintext
    assignment with each of the other pt_size alternatives.  Keep any that
    improve score_fn().  Stop when no round produces an improvement.

    Works for simple-substitution (bijective) and homophonic ciphers alike —
    homophones form naturally when scoring benefits from them.

    Returns the final score.
    """
    key = session.key
    if not key:
        return score_fn()

    pt_size = session.plaintext_alphabet.size
    best = score_fn()
    cipher_ids = sorted(key.keys())

    for _round in range(max_rounds):
        improved = False
        for ci in cipher_ids:
            old_pt = key[ci]
            local_best = best
            best_pt = old_pt
            for pt_id in range(pt_size):
                if pt_id == old_pt:
                    continue
                key[ci] = pt_id
                session.set_full_key(key)
                s = score_fn()
                if s > local_best:
                    local_best = s
                    best_pt = pt_id
            if best_pt != old_pt:
                key[ci] = best_pt
                session.set_full_key(key)
                best = local_best
                improved = True
            else:
                key[ci] = old_pt
                session.set_full_key(key)
        if not improved:
            break

    session.set_full_key(key)
    return best


def hill_climb_swaps(
    session: Session,
    word_set: set[str],
    max_rounds: int = 50,
) -> float:
    """Dictionary-scored hill climb.  Delegates to hill_climb_reassign."""
    if not session.key:
        return 0.0

    def score() -> float:
        return dictionary.score_plaintext(session.apply_key(), word_set)

    return hill_climb_reassign(session, score, max_rounds)


def random_restart_hill_climb(
    session: Session,
    word_set: set[str],
    restarts: int = 10,
    max_rounds_per_restart: int = 50,
) -> float:
    """Hill climbing with random restarts.

    Each restart assigns every cipher symbol to a random plaintext letter
    (independently, so homophones are possible) then runs hill_climb_reassign.
    Keeps the best key found across all restarts.
    """
    if not session.key:
        return 0.0

    pt_size = session.plaintext_alphabet.size
    cipher_ids = sorted(session.key.keys())

    def score() -> float:
        return dictionary.score_plaintext(session.apply_key(), word_set)

    best_score = score()
    best_key = dict(session.key)

    for _restart in range(restarts):
        new_key = {ci: random.randrange(pt_size) for ci in cipher_ids}
        session.set_full_key(new_key)
        s = hill_climb_reassign(session, score, max_rounds=max_rounds_per_restart)
        if s > best_score:
            best_score = s
            best_key = dict(session.key)

    session.set_full_key(best_key)
    return best_score


def simulated_anneal(
    session: Session,
    score_fn: Callable[[], float],
    max_steps: int = 5000,
    t_start: float = 1.0,
    t_end: float = 0.005,
    swap_fraction: float = 0.3,
    fixed_cipher_ids: set[int] | None = None,
) -> float:
    """Simulated annealing for substitution ciphers.

    Mixes single-symbol reassignment (1-opt) and 2-symbol swaps (2-opt).
    Accepts worse moves with Boltzmann probability exp(Δ/T), cooling
    geometrically from t_start to t_end over max_steps.

    When ``fixed_cipher_ids`` is supplied, those cipher symbols retain their
    current key mapping throughout the anneal; the search explores only the
    remaining symbols. This is how anchor-constrained refinement is wired
    from the automated runner.

    Always restores the best key seen — SA may end below its peak.
    Returns the best score achieved.
    """
    import math

    key = dict(session.key)
    if not key:
        return score_fn()

    pt_size = session.plaintext_alphabet.size
    frozen = set(fixed_cipher_ids or ())
    cipher_ids = [cid for cid in sorted(key.keys()) if cid not in frozen]
    n = len(cipher_ids)
    if n == 0:
        session.set_full_key(key)
        return score_fn()

    session.set_full_key(key)
    current_score = score_fn()
    best_score = current_score
    best_key = dict(key)

    if max_steps <= 1 or t_start <= t_end:
        return best_score

    alpha = (t_end / t_start) ** (1.0 / max_steps)
    temp = t_start

    for _ in range(max_steps):
        if n >= 2 and random.random() < swap_fraction:
            # 2-symbol swap
            i, j = random.sample(range(n), 2)
            ci, cj = cipher_ids[i], cipher_ids[j]
            old_pi, old_pj = key[ci], key[cj]
            key[ci], key[cj] = old_pj, old_pi
            is_swap = True
        else:
            # Single-symbol reassignment
            ci = random.choice(cipher_ids)
            old_pi = key[ci]
            new_pt = random.randrange(pt_size)
            if new_pt == old_pi:
                temp *= alpha
                continue
            key[ci] = new_pt
            is_swap = False

        session.set_full_key(key)
        new_score = score_fn()
        delta = new_score - current_score

        if delta >= 0 or random.random() < math.exp(delta / temp):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_key = dict(key)
        else:
            if is_swap:
                key[ci], key[cj] = old_pi, old_pj
            else:
                key[ci] = old_pi

        temp *= alpha

    session.set_full_key(best_key)
    return best_score


def pattern_seed_keys(
    session: Session,
    word_set: set[str],
    pattern_dict: dict[str, list[str]],
    n_seeds: int = 5,
) -> list[tuple[float, dict[int, int]]]:
    """Generate candidate initial keys using pattern matching on short words.

    Finds the most constrained cipher words (fewest pattern matches),
    uses their candidates to build partial keys, then fills remaining
    mappings with frequency-based assignments.

    Returns a list of (score, key) tuples sorted by score descending.
    """
    ct = session.cipher_text
    if ct is None:
        return []

    words = ct.words
    pt_alpha = session.plaintext_alphabet

    # Score each word by constraint level (fewer candidates = more constrained)
    word_candidates: list[tuple[int, str, list[str]]] = []
    for idx, word_tokens in enumerate(words):
        if len(word_tokens) < 2 or len(word_tokens) > 10:
            continue
        pat = pattern.word_pattern(word_tokens)
        matches = pattern.match_pattern(pat, pattern_dict)
        if matches and len(matches) <= 200:
            word_candidates.append((idx, pat, matches))

    if not word_candidates:
        return []

    # Sort by number of candidates (most constrained first)
    word_candidates.sort(key=lambda x: len(x[2]))

    # Try building keys from the top candidates of the most constrained words
    results: list[tuple[float, dict[int, int]]] = []

    # Take top 3 most constrained words
    seed_words = word_candidates[:3]

    for word_idx, _pat, candidates in seed_words:
        word_tokens = words[word_idx]
        # Try top candidates for this word
        for candidate in candidates[:n_seeds]:
            if len(candidate) != len(word_tokens):
                continue

            # Build partial key from this candidate
            partial_key: dict[int, int] = {}
            conflict = False
            for token_id, letter in zip(word_tokens, candidate):
                if not pt_alpha.has_symbol(letter):
                    conflict = True
                    break
                pt_id = pt_alpha.id_for(letter)
                if token_id in partial_key and partial_key[token_id] != pt_id:
                    conflict = True
                    break
                partial_key[token_id] = pt_id

            if conflict:
                continue

            # Fill remaining mappings with frequency-based guesses
            freq_data = frequency.sorted_frequency(ct.tokens)
            used_pt = set(partial_key.values())
            available_pt = [
                pt_alpha.id_for(pt_alpha.symbol_for(i))
                for i in range(pt_alpha.size)
                if i not in used_pt
            ]

            full_key = dict(partial_key)
            avail_idx = 0
            for ct_id, _count in freq_data:
                if ct_id not in full_key and avail_idx < len(available_pt):
                    full_key[ct_id] = available_pt[avail_idx]
                    avail_idx += 1

            session.set_full_key(full_key)
            score = dictionary.score_plaintext(session.apply_key(), word_set)
            results.append((score, dict(full_key)))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def auto_solve(
    session: Session,
    word_set: set[str],
    pattern_dict: dict[str, list[str]],
    verbose: bool = False,
) -> float:
    """Run automatic solving: pattern seeding + hill climbing.

    Returns the best score achieved.
    """
    ct = session.cipher_text
    if ct is None:
        return 0.0

    # Step 1: Try pattern-seeded keys
    seeds = pattern_seed_keys(session, word_set, pattern_dict, n_seeds=5)

    if verbose and seeds:
        print(f"    Pattern seeding: {len(seeds)} candidates, best={seeds[0][0]:.2%}")

    # Step 2: Also generate the plain frequency-based key
    freq_data = frequency.sorted_frequency(ct.tokens)
    from agent.prompts import FREQUENCY_ORDERS
    # Determine language from session context — default to English freq order
    freq_order = FREQUENCY_ORDERS.get("en")
    pt_alpha = session.plaintext_alphabet

    freq_key: dict[int, int] = {}
    for i, (ct_id, _count) in enumerate(freq_data):
        if i < len(freq_order) and pt_alpha.has_symbol(freq_order[i]):
            freq_key[ct_id] = pt_alpha.id_for(freq_order[i])

    session.set_full_key(freq_key)
    freq_score = dictionary.score_plaintext(session.apply_key(), word_set)
    all_candidates = [(freq_score, dict(freq_key))] + seeds

    # Step 3: Hill-climb the top candidates
    best_score = 0.0
    best_key: dict[int, int] = {}

    for i, (init_score, key) in enumerate(all_candidates[:5]):
        session.set_full_key(key)
        score = hill_climb_swaps(session, word_set, max_rounds=30)
        if verbose:
            print(f"    Candidate {i}: init={init_score:.2%} -> climbed={score:.2%}")
        if score > best_score:
            best_score = score
            best_key = session.key

    # Step 4: Random restart from best
    if best_key:
        session.set_full_key(best_key)
        final = random_restart_hill_climb(
            session, word_set, restarts=15, max_rounds_per_restart=30,
        )
        if verbose:
            print(f"    After random restarts: {final:.2%}")
        if final > best_score:
            best_score = final
            best_key = session.key

    session.set_full_key(best_key)
    return best_score
