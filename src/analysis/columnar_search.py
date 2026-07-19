"""Keyword / column-permutation columnar-transposition search with pluggable scoring.

Motivation (matrix finding F2, composite spec §1.1 item 3 / §4.1 / §7a):
``analysis.pure_transposition`` is a GEOMETRIC screen (matrix-rotate / route /
rail / mask / TransMatrix). It has no keyed-columnar coverage, so it fails blind
on a classic keyed columnar transposition (finding F2 measured ~0.10 char on the
round-4-class widths). This module fills that gap: it searches column *counts*
(widths) and column *permutations* — the exact inverse of
``ciphers.transposition.columnar_encrypt`` — and ranks candidates with a
**pluggable** scorer.

Why pluggable scoring is the point of the coordination decision (§7a):

* The composite substitution+transposition peel operates on the RAW cipher, whose
  letters are still substituted. A by-letter language model is blind to a
  substituted transposition (the substitution scrambles quadgram scores), so the
  peel passes a **substitution-invariant** scorer
  (:func:`make_structure_ratio_scorer`, wrapping ``ngram.ngram_structure_ratio``):
  the correct un-transposition restores adjacency structure (ratio >> 1) while
  every wrong permutation leaves it destroyed (ratio -> 1), regardless of which
  letters carry the distribution.
* A caller with un-substituted A-Z letters (a *plain* keyed columnar) passes the
  **language** scorer (:func:`make_language_scorer`): the correct un-transposition
  reads as real language and scores highest under the shipped n-gram model.
* The polygraphic PF-6 program (future work; do NOT build here) will pass a
  fractionation-stream scorer over non-language coordinate data to the SAME
  search loop. Nothing in the loop assumes language, so that reuse is free.

Convention (must exactly invert ``ciphers.transposition.columnar_encrypt``):
``columnar_encrypt`` fills columns round-robin (``cols[i % ncols] += ch``) and
reads them out in ``_rank_order(key)`` order. So a candidate is fully described
by ``(column_count, column_order)`` where ``column_order`` is the read-out
permutation (position -> source-column index), i.e. exactly ``_rank_order(key)``.
:func:`columnar_decode_by_order` inverts a candidate directly from that pair, and
:func:`keyword_for_order` recovers a canonical keyword that reproduces it.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from analysis import ngram

# A scorer maps a decoded A-Z stream to a float where HIGHER is better. It is the
# only knob the search loop needs; nothing else about the candidate matters to
# ranking, which is what keeps the loop scorer-agnostic (language, structure
# ratio, fractionation stream, ...).
Scorer = Callable[[str], float]


@dataclass(frozen=True)
class ColumnarFinalist:
    """One ranked keyed-columnar candidate.

    ``column_order`` is the read-out permutation (``_rank_order(key)`` in the
    ``ciphers.transposition`` convention): ``column_order[p]`` is the source
    column index whose block sits at read position ``p`` in the ciphertext.
    ``keyword`` is a canonical keyword that reproduces ``column_order`` via
    ``_rank_order`` (letters ``A, B, C, ...`` assigned in read order), so callers
    can record a human-readable transposition key alongside the raw permutation.
    """

    column_count: int
    column_order: tuple[int, ...]
    decoded_stream: str
    score: float
    keyword: str
    method: str = "exhaustive"  # "exhaustive" | "hill_climb"

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_count": self.column_count,
            "column_order": list(self.column_order),
            "keyword": self.keyword,
            "score": self.score,
            "method": self.method,
            "preview": self.decoded_stream[:80],
        }


# ---------------------------------------------------------------------------
# Candidate decode + key reconstruction (exact inverse of columnar_encrypt)
# ---------------------------------------------------------------------------

def columnar_decode_by_order(
    text: str, column_count: int, column_order: Iterable[int]
) -> str:
    """Invert a keyed columnar transposition from ``(column_count, column_order)``.

    Mirrors ``ciphers.transposition.columnar_decrypt`` exactly, but takes the
    read-out permutation directly instead of a keyword, so a blind search never
    has to materialise keys. ``columnar_decode_by_order(ct, len(key),
    _rank_order(key))`` equals ``columnar_decrypt(ct, key)`` for every key.
    Handles an incomplete final row (short columns) via the same ``divmod`` split.
    """
    order = list(column_order)
    ncols = int(column_count)
    if ncols <= 0:
        raise ValueError("column_count must be positive")
    if sorted(order) != list(range(ncols)):
        raise ValueError("column_order must be a permutation of range(column_count)")
    n = len(text)
    base, extra = divmod(n, ncols)
    col_len = [base + (1 if j < extra else 0) for j in range(ncols)]
    cols: list[str] = [""] * ncols
    idx = 0
    for j in order:
        cols[j] = text[idx: idx + col_len[j]]
        idx += col_len[j]
    out: list[str] = []
    maxlen = max(col_len) if col_len else 0
    for r in range(maxlen):
        for j in range(ncols):
            if r < len(cols[j]):
                out.append(cols[j][r])
    return "".join(out)


def keyword_for_order(column_order: Iterable[int]) -> str:
    """Return a canonical keyword whose ``_rank_order`` equals ``column_order``.

    Assigns ``A`` to the column read first, ``B`` to the column read second, and
    so on; sorting that keyword by ``(letter, index)`` then reproduces the read
    order. Widths above 26 fall back to a synthetic ``order=...`` string.
    """
    order = list(column_order)
    n = len(order)
    if n > 26:
        return "order=" + ",".join(str(c) for c in order)
    kw = ["A"] * n
    for pos, col in enumerate(order):
        kw[col] = chr(ord("A") + pos)
    return "".join(kw)


# ---------------------------------------------------------------------------
# Shipped scorer plugins
# ---------------------------------------------------------------------------

def make_language_scorer(language: str = "en", *, dict_weight: float = 0.0) -> Scorer:
    """Language-model scorer plugin (shipped models/dictionaries — never GT).

    Returns the mean quadgram log-probability of the decoded stream under the
    shipped n-gram model (higher = more language-like). This is valid whenever
    the stream is un-substituted A-Z language letters (e.g. a *plain* keyed
    columnar): the correct un-transposition reads as real language and wins.
    Optionally blends a dictionary-coverage term (``dict_weight`` > 0) for extra
    separation on boundary-bearing text.
    """
    quadgrams = ngram.NGRAM_CACHE.get(language, 4)
    word_set: set[str] = set()
    if dict_weight:
        from analysis.dictionary import get_dictionary_path, load_word_set

        path = get_dictionary_path(language)
        word_set = load_word_set(path) if path else set()

    def score(stream: str) -> float:
        value = ngram.normalized_ngram_score(stream, quadgrams, 4)
        if dict_weight and word_set:
            from analysis.dictionary import score_plaintext

            value += dict_weight * score_plaintext(stream, word_set)
        return value

    return score


def make_structure_ratio_scorer(n: int = 2) -> Scorer:
    """Substitution-INVARIANT scorer plugin (composite peel over substituted text).

    Wraps ``ngram.ngram_structure_ratio``: the ratio of observed n-gram index of
    coincidence to the value expected under independence. A monoalphabetic
    substitution only relabels letters, so this ratio is unchanged by it; a
    transposition shuffles positions and collapses the ratio toward 1.0. The
    correct un-transposition of a substituted-and-transposed stream therefore
    scores HIGHEST here even though the letters are still substituted — exactly
    what a by-letter language model cannot see.
    """

    def score(stream: str) -> float:
        return ngram.ngram_structure_ratio(stream, n)

    return score


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnarSearchConfig:
    """Bounded budget knobs for :func:`search_keyed_columnar` (sane defaults)."""

    min_width: int = 2
    max_width: int = 12
    # Widths whose permutation count is <= this are enumerated exhaustively
    # (guaranteed recovery); wider widths fall back to random-restart hill
    # climbing. 8! = 40320 scores in well under a second, so widths up to 8 are
    # exhaustive by default — this covers the round-4 class (5-8).
    exhaustive_max_factorial: int = 40320
    restarts: int = 64          # random restarts per hill-climbed width
    max_no_improve: int = 400   # local-search patience before a restart ends
    top_n: int = 10
    seed: int = 0


def search_keyed_columnar(
    stream: str,
    scorer: Scorer,
    *,
    config: ColumnarSearchConfig | None = None,
    min_width: int | None = None,
    max_width: int | None = None,
    exhaustive_max_factorial: int | None = None,
    restarts: int | None = None,
    top_n: int | None = None,
    seed: int | None = None,
) -> list[ColumnarFinalist]:
    """Search widths and column permutations for a keyed columnar transposition.

    ``scorer`` is the pluggable ranking callable (higher = better). The loop makes
    no *language* assumption — pass :func:`make_language_scorer` for plain (un-
    substituted) columnar, or :func:`make_structure_ratio_scorer` to peel a
    *substituted* transposition; a future caller (polygraphic PF-6) plugs in a
    fractionation-stream scorer. NOTE it is scorer-agnostic but NOT
    alphabet-agnostic: the stream is uppercased and filtered to A-Z internally,
    which is harmless for ADFGX/ADFGVX (their coordinate symbols ⊂ A-Z) but a
    non-letter coordinate scheme would need that normalization relaxed. Widths
    whose factorial is within budget are
    enumerated exhaustively (guaranteed to find the optimum for that width); wider
    widths use random-restart hill climbing over pairwise position swaps. Returns
    the ``top_n`` finalists across all widths, ranked by ``scorer`` (descending).

    Firewall: ciphertext-only. No ground truth enters here — the scorer ranks on
    shipped models/statistics alone.
    """
    cfg = config or ColumnarSearchConfig()
    if min_width is not None:
        cfg = _replace(cfg, min_width=min_width)
    if max_width is not None:
        cfg = _replace(cfg, max_width=max_width)
    if exhaustive_max_factorial is not None:
        cfg = _replace(cfg, exhaustive_max_factorial=exhaustive_max_factorial)
    if restarts is not None:
        cfg = _replace(cfg, restarts=restarts)
    if top_n is not None:
        cfg = _replace(cfg, top_n=top_n)
    if seed is not None:
        cfg = _replace(cfg, seed=seed)

    letters = "".join(ch for ch in stream.upper() if "A" <= ch <= "Z")
    n = len(letters)
    finalists: list[ColumnarFinalist] = []
    if n < 4:
        return finalists

    lo = max(2, cfg.min_width)
    # A width must leave at least 2 rows to be a meaningful transposition.
    hi = min(cfg.max_width, n - 1, n // 2)
    for width in range(lo, hi + 1):
        if math.factorial(width) <= cfg.exhaustive_max_factorial:
            best = _search_width_exhaustive(letters, width, scorer)
            method = "exhaustive"
        else:
            best = _search_width_hill_climb(
                letters, width, scorer,
                restarts=cfg.restarts,
                max_no_improve=cfg.max_no_improve,
                seed=cfg.seed + width,
            )
            method = "hill_climb"
        if best is None:
            continue
        order, decoded, score = best
        finalists.append(
            ColumnarFinalist(
                column_count=width,
                column_order=tuple(order),
                decoded_stream=decoded,
                score=score,
                keyword=keyword_for_order(order),
                method=method,
            )
        )

    finalists.sort(key=lambda f: f.score, reverse=True)
    return finalists[: cfg.top_n]


def _search_width_exhaustive(
    letters: str, width: int, scorer: Scorer
) -> tuple[tuple[int, ...], str, float] | None:
    best: tuple[tuple[int, ...], str, float] | None = None
    for order in itertools.permutations(range(width)):
        decoded = columnar_decode_by_order(letters, width, order)
        score = scorer(decoded)
        if best is None or score > best[2]:
            best = (order, decoded, score)
    return best


def _search_width_hill_climb(
    letters: str,
    width: int,
    scorer: Scorer,
    *,
    restarts: int,
    max_no_improve: int,
    seed: int,
) -> tuple[tuple[int, ...], str, float] | None:
    rng = random.Random(seed)
    best: tuple[tuple[int, ...], str, float] | None = None
    for _ in range(restarts):
        order = list(range(width))
        rng.shuffle(order)
        decoded = columnar_decode_by_order(letters, width, order)
        cur_score = scorer(decoded)
        no_improve = 0
        while no_improve < max_no_improve:
            i, j = rng.randrange(width), rng.randrange(width)
            if i == j:
                continue
            order[i], order[j] = order[j], order[i]
            cand = columnar_decode_by_order(letters, width, order)
            cand_score = scorer(cand)
            if cand_score > cur_score:
                cur_score = cand_score
                decoded = cand
                no_improve = 0
            else:
                order[i], order[j] = order[j], order[i]  # revert
                no_improve += 1
        if best is None or cur_score > best[2]:
            best = (tuple(order), decoded, cur_score)
    return best


def _replace(cfg: ColumnarSearchConfig, **changes: Any) -> ColumnarSearchConfig:
    import dataclasses

    return dataclasses.replace(cfg, **changes)
