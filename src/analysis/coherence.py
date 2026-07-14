"""Order-sensitive word-island coherence report (INV-0 Part 5).

``island_report`` distinguishes genuine plaintext from high-n-gram word-island
gibberish. The naive "many dictionary words -> coherent" test is fooled by
shuffled prose (English is ~47% function words, so scattered valid words glue
into long runs). Two derived guards fix this (review-2 options 2 AND 3):

* an **attested-junction floor**: a coherent span must carry >= 2 attested
  high-frequency word-bigram junctions, which shuffled (attestation-sparse) prose
  almost never accumulates; and
* an **order-significance test**: the mean word-bigram log-likelihood must beat a
  frequency-preserving shuffle of the same words at p <= 0.05. A shuffled input is
  a draw from its own null (p ~ 0.5) and can never read `coherent`.

`coherent` additionally REQUIRES a real word-bigram resource, so languages without
one (la/de in INV-0) can never falsely read coherent.
"""
from __future__ import annotations

import hashlib
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

_RESOURCE_ROOT = Path(__file__).resolve().parents[2] / "resources"
_FUNCTION_WORDS_DIR = _RESOURCE_ROOT / "function_words"
_WORD_BIGRAMS_DIR = _RESOURCE_ROOT / "word_bigrams"

ATTESTED_PER_SPAN = 2               # finding-2 option-2 attested-junction floor
_ORDER_N_SHUFFLES = 1000            # production default (Part 3); calibration used 300
_FUNCTION_WORD_MIN = {"en": 0.25}   # default 0.10 for la/de (function-word poorer)
_COHERENT_DICT_MIN = 0.75
_COHERENT_SPAN_MIN = 5
_GIBBERISH_DICT_MAX = 0.35


@lru_cache(maxsize=None)
def _load_function_words(language: str) -> frozenset[str]:
    path = _FUNCTION_WORDS_DIR / f"{language}.txt"
    if not path.exists():
        return frozenset()
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.add(line.upper())
    return frozenset(words)


@lru_cache(maxsize=None)
def _load_word_bigrams(language: str):
    """Return (attested: frozenset[(w1,w2)], log_probs: dict, floor) or None."""
    path = _WORD_BIGRAMS_DIR / f"{language}_bigrams.txt"
    if not path.exists():
        return None
    counts: dict[tuple[str, str], int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        w1, w2, c = parts[0].upper(), parts[1].upper(), parts[2]
        try:
            counts[(w1, w2)] = int(c)
        except ValueError:
            continue
    if not counts:
        return None
    total = sum(counts.values())
    floor = math.log10(0.1 / total)
    lp = {pair: math.log10(c / total) for pair, c in counts.items()}
    return frozenset(counts), lp, floor


def _words_from_text(text: str) -> list[str]:
    """Letters-only word tokens; space-delimited input is treated as pre-tokenized."""
    words = ["".join(c for c in w if c.isalpha()) for w in text.upper().split()]
    return [w for w in words if w]


def _word_bigram_loglik(words: list[str], lp: dict, floor: float) -> float:
    if len(words) < 2:
        return floor
    return sum(lp.get((words[k], words[k + 1]), floor)
               for k in range(len(words) - 1)) / (len(words) - 1)


def _order_seed(words: list[str]) -> int:
    material = "\x1f".join(words).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _word_bigram_order_p(words: list[str], lp: dict, floor: float, *, n_shuffles: int) -> float:
    """Upper-tail shuffle-null p that observed word-bigram loglik exceeds a
    frequency-preserving permutation (the finding-3 order-sensitivity test)."""
    if len(words) < 2:
        return 1.0
    obs = _word_bigram_loglik(words, lp, floor)
    rng = random.Random(_order_seed(words))
    pool = list(words)
    ge = 0
    for _ in range(n_shuffles):
        rng.shuffle(pool)
        if _word_bigram_loglik(pool, lp, floor) >= obs:
            ge += 1
    return (ge + 1) / (n_shuffles + 1)


def _longest_coherent_span(words, dict_set, function_words, attested) -> int:
    """Longest run whose junctions are all adjacent_ok and which carries >=
    ATTESTED_PER_SPAN attested word-bigram junctions (finding 2, option 2).

    adjacent_ok = both words in the dictionary AND (attested_bigram OR a function
    word is involved). The function-word rescue keeps content words glued to their
    neighbours in real prose, but a run only *qualifies* as coherent once it also
    carries enough attested high-frequency bigrams — which shuffled prose
    (function-word-dense but attestation-sparse) does not.
    """
    if len(words) < 2:
        return len(words)

    def ok(k):
        a, b = words[k], words[k + 1]
        if a not in dict_set or b not in dict_set:
            return False, False
        att = (a, b) in attested
        adj = att or a in function_words or b in function_words
        return adj, att

    best = 1
    run_len = 1
    att_in_run = 0
    for k in range(len(words) - 1):
        adj, att = ok(k)
        if adj:
            run_len += 1
            att_in_run += 1 if att else 0
            if att_in_run >= ATTESTED_PER_SPAN:
                best = max(best, run_len)
        else:
            run_len = 1
            att_in_run = 0
    return best


def island_report(
    text,
    language: str = "en",
    *,
    word_set: set[str] | None = None,
    freq_rank: dict[str, int] | None = None,
    segmented=None,
    n_shuffles: int = _ORDER_N_SHUFFLES,
) -> dict[str, Any]:
    """Return coherence evidence for a plaintext candidate (Part 5).

    ``segmented`` (a :class:`analysis.segment.SegmentResult`) is used verbatim
    when supplied (finding 12 — no second segmentation on the finalist hot path);
    otherwise the text is tokenized (space-delimited input is treated as
    pre-tokenized words; a continuous letter run is segmented internally).
    """
    if segmented is not None:
        words = [w.upper() for w in segmented.words if w]
    elif isinstance(text, (list, tuple)):
        words = [str(w).upper() for w in text if str(w)]
    else:
        stripped = str(text)
        if any(ch.isspace() for ch in stripped.strip()):
            words = _words_from_text(stripped)
        else:
            words = _segment_internally(stripped, language, word_set, freq_rank)

    if word_set is None:
        word_set = _load_word_set(language)

    n = len(words)
    function_words = _load_function_words(language)
    bigram_res = _load_word_bigrams(language)
    has_bigram_resource = bigram_res is not None

    dict_hits = sum(1 for w in words if w in word_set)
    dict_rate = dict_hits / n if n else 0.0
    fn_rate = (sum(1 for w in words if w in function_words) / n) if n else 0.0
    pseudo_fraction = (1.0 - dict_rate) if n else 0.0

    if has_bigram_resource:
        attested, lp, floor = bigram_res
        wb_loglik = _word_bigram_loglik(words, lp, floor)
        order_p = _word_bigram_order_p(words, lp, floor, n_shuffles=n_shuffles)
        span = _longest_coherent_span(words, word_set, function_words, attested)
    else:
        wb_loglik = None
        order_p = None
        span = _longest_coherent_span(words, word_set, function_words, frozenset())

    order_significant = bool(order_p is not None and order_p <= 0.05)
    fn_min = _FUNCTION_WORD_MIN.get(language, 0.10)

    coherent = (
        dict_rate >= _COHERENT_DICT_MIN
        and fn_rate >= fn_min
        and span >= _COHERENT_SPAN_MIN
        and has_bigram_resource
        and order_significant
    )
    if coherent:
        verdict = "coherent"
    elif dict_rate < _GIBBERISH_DICT_MAX or (fn_rate < 0.08 and span <= 2):
        verdict = "gibberish"
    else:
        verdict = "word_islands"

    return {
        "dict_rate": round(dict_rate, 4),
        "function_word_rate": round(fn_rate, 4),
        "word_bigram_available": has_bigram_resource,
        "word_bigram_loglik": round(wb_loglik, 5) if wb_loglik is not None else None,
        "word_bigram_order_p": round(order_p, 5) if order_p is not None else None,
        "word_bigram_order_significant": order_significant,
        "longest_coherent_span": span,
        "word_count": n,
        "pseudo_word_fraction": round(pseudo_fraction, 4),
        "verdict": verdict,
    }


def _segment_internally(text: str, language: str, word_set, freq_rank):
    from analysis.segment import segment_text
    letters = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if word_set is None:
        word_set = _load_word_set(language)
    if freq_rank is None:
        freq_rank = _load_freq_rank(language)
    result = segment_text(letters, word_set, freq_rank=freq_rank)
    return [w.upper() for w in result.words if w]


def _load_word_set(language: str) -> set[str]:
    from analysis.finalist_validation import _load_word_set as loader
    return loader(language)


def _load_freq_rank(language: str) -> dict[str, int] | None:
    from analysis.finalist_validation import _load_word_list
    word_list = _load_word_list(language)
    if not word_list:
        return None
    return {w.upper(): i for i, w in enumerate(word_list)}
