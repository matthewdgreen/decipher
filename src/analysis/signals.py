"""Multi-signal scoring panel.

The v1 loop fed the agent a single dictionary-hit score. That's flat and has
a low ceiling on medieval Latin / archaic German. v2 exposes a panel of
independent signals; the agent picks which to consult and how to weight them.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from analysis import dictionary, ngram, pattern


@dataclass
class SignalPanel:
    """Scoring signals for one decryption."""

    dictionary_rate: float                    # fraction of alpha words in the wordlist
    quadgram_loglik_per_gram: float           # mean log10 p(quadgram), higher = better
    bigram_loglik_per_gram: float             # same, bigrams — more sensitive to short text
    bigram_chi2: float                        # chi-squared vs reference bigrams; lower = better
    pattern_consistency: float                # fraction of cipher words whose pattern matches SOME dictionary word
    constraint_satisfaction: float            # 1.0 if key is one-to-one; <1.0 indicates homophony
    unrecognized_word_count: int
    total_alpha_words: int
    mapped_count: int
    unmapped_count: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Round floats for readability in tool output
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


_SPACED_LETTERS_RE = re.compile(r"(?<!\S)([A-Z?] ){2,}[A-Z?](?!\S)")


def _collapse_spaced_letters(text: str) -> str:
    """Collapse 'Q U E D A M' → 'QUEDAM'.

    Multisym output from canonical S-token decryption produces letters
    separated by spaces. This makes it a single word for scoring purposes.
    """
    def repl(m: re.Match) -> str:
        return m.group(0).replace(" ", "")
    return _SPACED_LETTERS_RE.sub(repl, text.upper())


def normalize_for_scoring(decrypted: str) -> str:
    """Prepare a decryption string for scoring: uppercase, collapse spaced
    letters, normalize separators.

    Collapse runs before the " | " replacement so word boundaries don't
    merge two spaced-letter runs into one pseudo-word.
    """
    t = decrypted.upper()
    t = _collapse_spaced_letters(t)
    t = t.replace(" | ", " ")
    t = " ".join(t.split())
    return t


def compute_panel(
    decrypted: str,
    cipher_words: list[list[int]],
    key: dict[int, int],
    used_ct_ids: set[int],
    language: str = "en",
    word_set: set[str] | None = None,
    pattern_dict: dict[str, list[str]] | None = None,
) -> SignalPanel:
    """Compute all signals. Callers pass word_set / pattern_dict for caching;
    otherwise they're loaded from the language's dictionary file."""
    normalized = normalize_for_scoring(decrypted)

    # --- dictionary rate ---
    if word_set is None:
        word_set = dictionary.load_word_set(dictionary.get_dictionary_path(language))

    words = normalized.split()
    alpha_words = [w for w in words if _is_scored_word(w)]
    if alpha_words:
        hits = sum(1 for w in alpha_words if w in word_set)
        dict_rate = hits / len(alpha_words)
        unrecog = [w for w in alpha_words if w not in word_set]
    else:
        dict_rate = 0.0
        unrecog = []

    # --- n-gram log-likelihoods ---
    quad_lp = ngram.NGRAM_CACHE.get(language, 4)
    bi_lp = ngram.NGRAM_CACHE.get(language, 2)
    quad_score = ngram.normalized_ngram_score(normalized, quad_lp, n=4)
    bi_score = ngram.normalized_ngram_score(normalized, bi_lp, n=2)
    bi_chi2 = _bigram_chi_squared(normalized, bi_lp)

    # --- pattern consistency ---
    if pattern_dict is None:
        pattern_dict = pattern.build_pattern_dictionary(
            pattern.load_word_list(dictionary.get_dictionary_path(language))
        )
    pc = _pattern_consistency(cipher_words, pattern_dict)

    # --- constraint satisfaction ---
    cs = _constraint_satisfaction(key)

    unmapped = used_ct_ids - set(key.keys())

    return SignalPanel(
        dictionary_rate=dict_rate,
        quadgram_loglik_per_gram=quad_score,
        bigram_loglik_per_gram=bi_score,
        bigram_chi2=bi_chi2,
        pattern_consistency=pc,
        constraint_satisfaction=cs,
        unrecognized_word_count=len(unrecog),
        total_alpha_words=len(alpha_words),
        mapped_count=len(key),
        unmapped_count=len(unmapped),
    )


def _is_scored_word(w: str) -> bool:
    """Include words with at least one letter; skip pure separator tokens like '|'."""
    return any(c.isalpha() for c in w)


def _bigram_chi_squared(text: str, reference_log_probs: dict[str, float]) -> float:
    """Chi-squared between observed bigrams and reference probabilities.

    Uses only alphabetic bigrams (no space-padded ones) so the statistic is
    stable regardless of how the scorer tokenizes.
    """
    t = "".join(c for c in text.upper() if c.isalpha())
    if len(t) < 2:
        return float("inf")
    observed: Counter[str] = Counter()
    for i in range(len(t) - 1):
        observed[t[i : i + 2]] += 1
    total = sum(observed.values())
    if total == 0:
        return float("inf")
    floor_log = reference_log_probs.get("_floor", -10.0)
    chi2 = 0.0
    for bg, count in observed.items():
        obs_p = count / total
        ref_log = reference_log_probs.get(bg, floor_log)
        exp_p = 10 ** ref_log
        if exp_p > 0:
            chi2 += (obs_p - exp_p) ** 2 / exp_p
    return chi2


def _pattern_consistency(
    cipher_words: list[list[int]],
    pattern_dict: dict[str, list[str]],
) -> float:
    """Fraction of length-≥2 cipher words whose isomorph appears in the
    pattern dictionary."""
    if not cipher_words:
        return 0.0
    scored = 0
    matched = 0
    for w in cipher_words:
        if len(w) < 2:
            continue
        scored += 1
        pat = pattern.word_pattern(w)
        if pat in pattern_dict and pattern_dict[pat]:
            matched += 1
    return matched / scored if scored > 0 else 0.0


def _constraint_satisfaction(key: dict[int, int]) -> float:
    """1.0 if all ct→pt mappings are unique (proper substitution);
    fraction unique otherwise. Homophonic mappings legitimately score <1.0."""
    if not key:
        return 1.0
    values = list(key.values())
    return len(set(values)) / len(values)
