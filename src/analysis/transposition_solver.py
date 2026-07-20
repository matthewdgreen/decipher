"""No-LLM transposition-family solver.

The existing transform / ``pure_transposition`` path already recovers *route*
transpositions (spiral/diagonal grid reads), but leaves the keyed columnar
families at the monoalphabetic floor because the transform search never
searches keyword-column ORDERINGS. This module closes that gap with a direct
permutation search:

* **Keyword columnar** (complete + incomplete): a greedy column-ordering
  hill-climb, scored by the language model. This is the biggest single win and
  the shared backbone for myszkowski / amsco / nihilist, which are keyed-column
  ciphers too.
* **Railfence / Redefence**: rail-count brute force (redefence additionally
  searches the rail read-order).
* **Myszkowski**: simulated annealing over the column-group vector (repeated
  key letters group columns that are read together).
* **Amsco**: column-ordering search over the alternating 1-2 cell fill.
* **Nihilist transposition**: keyword-permutation brute force over the
  ``n x n`` block.

Everything is scored by the *existing* language model — the dictionary word
coverage (``analysis.dictionary``) and n-gram log-likelihood
(``analysis.ngram``) that the substitution / route paths use. The search is
bounded by a deterministic per-strategy work budget plus a wall-clock safety
cap, so a run never hangs.

Routing: :func:`transposition_suspicion` gives a cheap content signal
(monogram frequencies match the target language by letter — preserved by a
transposition, destroyed by a substitution) so a plain-substitution or
homophonic cipher does not trigger a transposition sweep. The automated runner
also routes the ACA transposition family names here by name.

This module is part of the LLM-free cryptanalysis engine (``analysis`` /
``automated``); it does not touch the agentic system.
"""
from __future__ import annotations

import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from analysis import ngram
from analysis.dictionary import get_dictionary_path, score_plaintext

# ---------------------------------------------------------------------------
# Configuration (all env-overridable; deterministic iteration budgets drive the
# search, the wall-clock cap is only a safety net so a run never hangs)
# ---------------------------------------------------------------------------

#: Weight on dictionary word coverage when combining with the n-gram mean for
#: cross-family finalist ranking. The n-gram mean is a (negative) log10 and the
#: dictionary rate is 0..1; this weight puts them on a comparable scale.
_DICT_WEIGHT = 2.0

#: Dictionary word-rate above which the general cascade treats a candidate as a
#: solve and skips the expensive annealing strategies.
_SOLVED_DICT_RATE = 0.85

#: Minimum remaining wall-clock budget (seconds) required to attempt the
#: keyed-columnar F2 escalation. Measured F2 cost is ~7-9s on ~250 chars, so the
#: escalation is skipped entirely when less than this remains.
_F2_MIN_REMAINING_SECONDS = 5.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


#: Wall-clock safety cap for the whole solve (seconds). Generous — the
#: deterministic iteration budgets normally finish well inside it.
def _default_budget_seconds() -> float:
    return _env_float("DECIPHER_TRANSPOSITION_BUDGET_SECONDS", 60.0)


# ---------------------------------------------------------------------------
# Deadline helper
# ---------------------------------------------------------------------------


@dataclass
class _Deadline:
    """Monotonic wall-clock deadline shared across strategies."""

    limit: float

    @classmethod
    def after(cls, seconds: float) -> "_Deadline":
        return cls(limit=time.monotonic() + max(0.0, seconds))

    def expired(self) -> bool:
        return time.monotonic() >= self.limit

    def remaining(self) -> float:
        return max(0.0, self.limit - time.monotonic())


# ---------------------------------------------------------------------------
# Language scoring (reuse the existing dictionary + n-gram model)
# ---------------------------------------------------------------------------

_WORDSET_CACHE: dict[str, set[str]] = {}
_MONOGRAM_CACHE: dict[str, list[float]] = {}


def _word_set(language: str) -> set[str]:
    cached = _WORDSET_CACHE.get(language)
    if cached is None:
        from analysis.dictionary import load_word_set

        path = get_dictionary_path(language)
        cached = load_word_set(path) if path else set()
        _WORDSET_CACHE[language] = cached
    return cached


def _ngram_table(language: str, n: int) -> dict[str, float]:
    return ngram.NGRAM_CACHE.get(language, n)


def _ngram_mean(text: str, language: str, n: int) -> float:
    return ngram.normalized_ngram_score(text, _ngram_table(language, n), n)


def _dict_rate(text: str, language: str) -> float:
    ws = _word_set(language)
    if not ws:
        return 0.0
    return score_plaintext(text, ws)


def full_score(text: str, language: str = "en") -> float:
    """Cross-family finalist score: quadgram mean + weighted dictionary rate.

    This is the *existing* language model (``analysis.ngram`` +
    ``analysis.dictionary``); higher is better.
    """

    if not text:
        return float("-inf")
    return _ngram_mean(text, language, 4) + _DICT_WEIGHT * _dict_rate(text, language)


def _monogram_reference(language: str) -> list[float]:
    """Expected A-Z letter frequency for the language, from the wordlist."""

    cached = _MONOGRAM_CACHE.get(language)
    if cached is not None:
        return cached
    from analysis.pattern import load_word_list

    counts = Counter()
    path = get_dictionary_path(language)
    if path:
        for word in load_word_list(path):
            for ch in word.upper():
                if "A" <= ch <= "Z":
                    counts[ch] += 1
    total = sum(counts.values())
    if total == 0:
        vec = [1.0 / 26.0] * 26
    else:
        vec = [counts.get(chr(65 + i), 0) / total for i in range(26)]
    _MONOGRAM_CACHE[language] = vec
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da <= 0 or db <= 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


# ---------------------------------------------------------------------------
# Ciphertext extraction + suspicion
# ---------------------------------------------------------------------------


def az_string_from_cipher_text(cipher_text: Any) -> str | None:
    """Return the uppercase A-Z letter stream, or None if not A-Z-only.

    A transposition attack operates on the plaintext alphabet, so this solver
    only applies when every cipher symbol is a single A-Z letter.
    """

    letters: list[str] = []
    for token in cipher_text.tokens:
        symbol = cipher_text.alphabet.symbol_for(token).upper()
        if len(symbol) == 1 and "A" <= symbol <= "Z":
            letters.append(symbol)
        else:
            return None
    return "".join(letters)


# Substitution-invariant order-layer thresholds (composite Slice A §2.3).
_ORDER_LAYER_SHAPE_COS = 0.90     # sorted-magnitude cosine vs language -> language-like SHAPE
_ORDER_LAYER_STRUCTURE_ABSENT = 1.35  # bigram-structure ratio below -> n-gram structure scrambled
_ORDER_LAYER_MIN_TOKENS = 150     # structure ratio is unreliable below this


def transposition_suspicion(
    cipher_text: Any,
    language: str = "en",
    *,
    min_tokens: int = 40,
) -> dict[str, Any]:
    """Cheap content signal: does this look like a (mono-alphabet) transposition?

    A transposition preserves the plaintext letter multiset, so the observed
    A-Z monogram distribution matches the language distribution *by letter*
    (high cosine similarity). A monoalphabetic substitution permutes the
    letters, so the same-shaped distribution lands on the wrong letters (low
    cosine). Homophonic ciphers have >26 symbols and never reach here.

    Also returns ``order_layer_suspected`` (composite Slice A §2.3): a
    substitution-INVARIANT signal for a residual order/transposition layer. The
    by-letter cosine above catches an UNsubstituted transposition; this second
    signal fires when the monogram SHAPE is language-like by sorted magnitude
    (regardless of WHICH letters — so a substitution does not hide it) BUT the
    n-gram adjacency structure is scrambled (``ngram.ngram_structure_ratio`` near
    1.0). A plain substitution keeps its adjacency structure (ratio high), so this
    stays False for it; a substitution-then-transposition trips it. This is what
    the ``disc_sub_transp_composite`` discriminator (and, later, the router)
    consume.

    Returns ``{"suspicious", "score", "order_layer_suspected", "reasons"}``.
    """

    reasons: list[str] = []
    text = az_string_from_cipher_text(cipher_text)
    if text is None:
        return {
            "suspicious": False,
            "score": 0.0,
            "order_layer_suspected": False,
            "reasons": ["ciphertext is not single A-Z symbols"],
        }
    n = len(text)
    if n < min_tokens:
        return {
            "suspicious": False,
            "score": 0.0,
            "order_layer_suspected": False,
            "reasons": [f"too few letters ({n} < {min_tokens})"],
        }
    counts = Counter(text)
    observed = [counts.get(chr(65 + i), 0) / n for i in range(26)]
    reference = _monogram_reference(language)
    cos = _cosine(observed, reference)
    threshold = _env_float("DECIPHER_TRANSPOSITION_SUSPICION_COSINE", 0.82)
    suspicious = cos >= threshold
    reasons.append(
        f"monogram cosine vs {language} = {cos:.3f} "
        f"({'>=' if suspicious else '<'} {threshold:.2f})"
    )

    # --- substitution-invariant order-layer signal (§2.3) ---
    shape_cos = _cosine(sorted(observed, reverse=True), sorted(reference, reverse=True))
    structure_ratio = ngram.ngram_structure_ratio(text, 2)
    order_layer_suspected = (
        n >= _ORDER_LAYER_MIN_TOKENS
        and shape_cos >= _ORDER_LAYER_SHAPE_COS
        and 0.0 < structure_ratio < _ORDER_LAYER_STRUCTURE_ABSENT
    )
    reasons.append(
        f"language-like monogram SHAPE (sorted cosine {shape_cos:.3f}) with "
        f"n-gram structure ratio {structure_ratio:.3f} "
        f"({'<' if structure_ratio < _ORDER_LAYER_STRUCTURE_ABSENT else '>='} "
        f"{_ORDER_LAYER_STRUCTURE_ABSENT:.2f}) -> order layer "
        f"{'suspected' if order_layer_suspected else 'not suspected'}"
    )

    return {
        "suspicious": suspicious,
        "score": round(cos, 4),
        "order_layer_suspected": order_layer_suspected,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Decrypt primitives (explicit-key, so the search can drive them directly)
# ---------------------------------------------------------------------------


def decrypt_columnar_order(text: str, order: list[int]) -> str:
    """Columnar decrypt given the read order of the ORIGINAL column indices.

    ``order[k]`` is the original column index of the k-th ciphertext chunk
    (this is ``_rank_order(keyword)`` for a keyword-columnar cipher). Handles
    complete and incomplete final rows.
    """

    ncols = len(order)
    n = len(text)
    if ncols < 2 or ncols >= n:
        return text
    base, extra = divmod(n, ncols)
    col_len = [base + (1 if j < extra else 0) for j in range(ncols)]
    cols = [""] * ncols
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


def _rail_pattern(n: int, rails: int) -> list[int]:
    pattern: list[int] = []
    r, d = 0, 1
    for _ in range(n):
        pattern.append(r)
        if r == 0:
            d = 1
        elif r == rails - 1:
            d = -1
        r += d
    return pattern


def decrypt_railfence(text: str, rails: int) -> str:
    if rails < 2 or rails >= len(text):
        return text
    pattern = _rail_pattern(len(text), rails)
    counts = [pattern.count(k) for k in range(rails)]
    pos = 0
    rail_str: list[str] = []
    for k in range(rails):
        rail_str.append(text[pos: pos + counts[k]])
        pos += counts[k]
    idx = [0] * rails
    out: list[str] = []
    for r in pattern:
        out.append(rail_str[r][idx[r]])
        idx[r] += 1
    return "".join(out)


def decrypt_redefence(text: str, rails: int, order: list[int]) -> str:
    if rails < 2 or rails >= len(text):
        return text
    pattern = _rail_pattern(len(text), rails)
    counts = [pattern.count(k) for k in range(rails)]
    rail_str = [""] * rails
    pos = 0
    for k in order:
        rail_str[k] = text[pos: pos + counts[k]]
        pos += counts[k]
    idx = [0] * rails
    out: list[str] = []
    for r in pattern:
        out.append(rail_str[r][idx[r]])
        idx[r] += 1
    return "".join(out)


def decrypt_myszkowski(text: str, colnum: list[int]) -> str:
    """Myszkowski decrypt for an explicit column-group vector.

    ``colnum[j]`` is the reading rank of column ``j``'s group (columns sharing a
    value are read together, row by row). Unused (gap) group numbers are skipped
    so the search may propose non-contiguous vectors.
    """

    ncols = len(colnum)
    n = len(text)
    if ncols < 1 or ncols >= n:
        return text
    base, extra = divmod(n, ncols)
    col_len = [base + (1 if j < extra else 0) for j in range(ncols)]
    cols = [""] * ncols
    idx = 0
    for num in range(max(colnum) + 1):
        group = [j for j in range(ncols) if colnum[j] == num]
        if not group:
            continue
        if len(group) == 1:
            j = group[0]
            cols[j] = text[idx: idx + col_len[j]]
            idx += col_len[j]
        else:
            total = sum(col_len[j] for j in group)
            block = text[idx: idx + total]
            idx += total
            buffers: dict[int, list[str]] = {j: [] for j in group}
            bpos = 0
            maxlen = max(col_len[j] for j in group)
            for r in range(maxlen):
                for j in group:
                    if r < col_len[j] and bpos < len(block):
                        buffers[j].append(block[bpos])
                        bpos += 1
            for j in group:
                cols[j] = "".join(buffers[j])
    out: list[str] = []
    maxlen = max(col_len) if col_len else 0
    for r in range(maxlen):
        for j in range(ncols):
            if r < len(cols[j]):
                out.append(cols[j][r])
    return "".join(out)


def _amsco_cell_sizes(n: int, ncols: int, start: int) -> list[list[int]]:
    sizes: list[list[int]] = []
    remaining = n
    row = 0
    while remaining > 0:
        rowsizes: list[int] = []
        for col in range(ncols):
            if remaining <= 0:
                rowsizes.append(0)
                continue
            even = (row + col) % 2 == 0
            base = (1 if even else 2) if start == 1 else (2 if even else 1)
            take = min(base, remaining)
            rowsizes.append(take)
            remaining -= take
        sizes.append(rowsizes)
        row += 1
    return sizes


def decrypt_amsco_order(text: str, order: list[int], start: int) -> str:
    """Amsco decrypt given the read order of the columns and the fill start."""

    ncols = len(order)
    n = len(text)
    if ncols < 2 or ncols >= n:
        return text
    sizes = _amsco_cell_sizes(n, ncols, start)
    nrows = len(sizes)
    col_len = [sum(sizes[r][j] for r in range(nrows)) for j in range(ncols)]
    colstr = [""] * ncols
    idx = 0
    for j in order:
        colstr[j] = text[idx: idx + col_len[j]]
        idx += col_len[j]
    grid = [["" for _ in range(ncols)] for _ in range(nrows)]
    for j in range(ncols):
        pos = 0
        for r in range(nrows):
            s = sizes[r][j]
            grid[r][j] = colstr[j][pos: pos + s]
            pos += s
    out: list[str] = []
    for r in range(nrows):
        for j in range(ncols):
            out.append(grid[r][j])
    return "".join(out)


def decrypt_nihilist(text: str, perm: list[int]) -> str | None:
    """Nihilist-transposition decrypt for an explicit ``n x n`` permutation.

    Returns None when the text length is not a multiple of ``len(perm)**2``.
    """

    b = len(perm)
    block = b * b
    n = len(text)
    if b < 2 or block > n or n % block != 0:
        return None
    out: list[str] = []
    for base in range(0, n, block):
        chunk = text[base: base + block]
        grid = [["" for _ in range(b)] for _ in range(b)]
        k = 0
        for i in range(b):
            for j in range(b):
                grid[perm[i]][perm[j]] = chunk[k]
                k += 1
        for i in range(b):
            for j in range(b):
                out.append(grid[i][j])
    return "".join(out)


# ---------------------------------------------------------------------------
# Candidate container
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    plaintext: str
    score: float
    family: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "score": round(self.score, 6),
            "params": dict(self.params),
            "preview": self.plaintext[:120],
        }


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------


def _columnar_widths(n: int) -> range:
    hi = min(_env_int("DECIPHER_TRANSPOSITION_MAX_WIDTH", 15), max(2, n // 2))
    return range(2, hi + 1)


def _column_order_polish(
    text: str,
    order: list[int],
    score: Callable[[list[int]], float],
) -> tuple[list[int], float]:
    """Deterministic swap + insertion hill-climb to a local optimum."""

    width = len(order)
    best = list(order)
    best_score = score(best)
    order = list(best)
    cur = best_score
    improved = True
    while improved:
        improved = False
        for i in range(width):
            for j in range(i + 1, width):
                order[i], order[j] = order[j], order[i]
                s = score(order)
                if s > cur + 1e-9:
                    cur = s
                    improved = True
                    best = list(order)
                else:
                    order[i], order[j] = order[j], order[i]
        order = list(best)
        for i in range(width):
            col = order.pop(i)
            placed = False
            for j in range(width):
                if j == i:
                    continue
                order.insert(j, col)
                s = score(order)
                if s > cur + 1e-9:
                    cur = s
                    improved = True
                    best = list(order)
                    placed = True
                    break
                order.pop(j)
            if not placed:
                order.insert(i, col)
        order = list(best)
    return best, best_score if best_score >= cur else cur


def _columnar_search_width(
    text: str,
    width: int,
    language: str,
    *,
    decrypt: Callable[[str, list[int]], str],
    iters: int,
    seeds: int,
    reheat_period: int,
    seed_base: int,
    deadline: _Deadline,
    t0: float = 0.6,
    t_end: float = 0.02,
) -> tuple[list[int], float]:
    """Reheating simulated annealing over a column read order (swap + or-opt).

    SA (not plain hill-climbing) is needed because incomplete columnar has a
    sticky *rotation* local optimum — a cyclically shifted column order reads as
    near-fluent text yet is a full miss — that greedy restarts rarely escape.
    """

    table = _ngram_table(language, 4)

    def score(order: list[int]) -> float:
        return ngram.normalized_ngram_score(decrypt(text, order), table, 4)

    global_best = list(range(width))
    global_best_score = score(global_best)
    for s in range(seeds):
        if deadline.expired():
            break
        rng = random.Random(seed_base * 1000 + width * 31 + s)
        order = list(range(width))
        rng.shuffle(order)
        cur = score(order)
        best = list(order)
        best_score = cur
        for k in range(iters):
            if (k & 511) == 0 and deadline.expired():
                break
            frac = (k % reheat_period) / reheat_period
            temp = t0 * ((t_end / t0) ** frac)
            save = list(order)
            if rng.random() < 0.5:
                i, j = rng.sample(range(width), 2)
                order[i], order[j] = order[j], order[i]
            else:
                i = rng.randrange(width)
                col = order.pop(i)
                order.insert(rng.randrange(width), col)
            cand = score(order)
            delta = cand - cur
            if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-6)):
                cur = cand
                if cand > best_score:
                    best_score = cand
                    best = list(order)
            else:
                order = save
        polished, polished_score = _column_order_polish(text, best, score)
        if polished_score > global_best_score:
            global_best_score = polished_score
            global_best = polished
    return global_best, global_best_score


def search_columnar(
    text: str,
    language: str,
    *,
    deadline: _Deadline,
    seed: int = 0,
) -> list[_Candidate]:
    iters = _env_int("DECIPHER_TRANSPOSITION_COLUMNAR_ITERS", 2500)
    seeds = _env_int("DECIPHER_TRANSPOSITION_COLUMNAR_SEEDS", 3)
    reheat = _env_int("DECIPHER_TRANSPOSITION_COLUMNAR_REHEAT", 1200)
    out: list[_Candidate] = []
    for width in _columnar_widths(len(text)):
        if deadline.expired():
            break
        order, _ = _columnar_search_width(
            text,
            width,
            language,
            decrypt=decrypt_columnar_order,
            iters=iters,
            seeds=seeds,
            reheat_period=reheat,
            seed_base=seed,
            deadline=deadline,
        )
        pt = decrypt_columnar_order(text, order)
        out.append(
            _Candidate(pt, full_score(pt, language), "columnar", {"width": width, "order": order})
        )
    return out


def search_railfence(
    text: str,
    language: str,
    *,
    deadline: _Deadline,
) -> list[_Candidate]:
    n = len(text)
    hi = min(_env_int("DECIPHER_TRANSPOSITION_MAX_RAILS", 20), max(2, n // 2))
    out: list[_Candidate] = []
    for rails in range(2, hi + 1):
        if deadline.expired():
            break
        pt = decrypt_railfence(text, rails)
        out.append(_Candidate(pt, full_score(pt, language), "railfence", {"rails": rails}))
    return out


def search_redefence(
    text: str,
    language: str,
    *,
    deadline: _Deadline,
    seed: int = 0,
) -> list[_Candidate]:
    import itertools

    rng = random.Random(seed)
    n = len(text)
    hi = min(_env_int("DECIPHER_TRANSPOSITION_MAX_RAILS", 20), max(2, n // 2))
    brute_cap = _env_int("DECIPHER_TRANSPOSITION_REDEFENCE_BRUTE_RAILS", 7)
    tri = _ngram_table(language, 3)
    out: list[_Candidate] = []
    for rails in range(2, hi + 1):
        if deadline.expired():
            break
        best_order: list[int] | None = None
        best_inner = float("-inf")
        if rails <= brute_cap:
            for order in itertools.permutations(range(rails)):
                if deadline.expired():
                    break
                cand = decrypt_redefence(text, rails, list(order))
                s = ngram.normalized_ngram_score(cand, tri, 3)
                if s > best_inner:
                    best_inner = s
                    best_order = list(order)
        else:
            # Hill-climb the rail read-order for large rail counts.
            order = list(range(rails))
            best_order = list(order)
            best_inner = ngram.normalized_ngram_score(
                decrypt_redefence(text, rails, order), tri, 3
            )
            for _ in range(200):
                if deadline.expired():
                    break
                i, j = rng.sample(range(rails), 2)
                order[i], order[j] = order[j], order[i]
                s = ngram.normalized_ngram_score(
                    decrypt_redefence(text, rails, order), tri, 3
                )
                if s > best_inner + 1e-9:
                    best_inner = s
                    best_order = list(order)
                else:
                    order[i], order[j] = order[j], order[i]
        if best_order is not None:
            pt = decrypt_redefence(text, rails, best_order)
            out.append(
                _Candidate(pt, full_score(pt, language), "redefence", {"rails": rails, "order": best_order})
            )
    return out


def _myszkowski_polish(text: str, colnum: list[int], table: dict[str, float]) -> tuple[list[int], float]:
    colnum = list(colnum)
    cur = ngram.normalized_ngram_score(decrypt_myszkowski(text, colnum), table, 4)
    improved = True
    while improved:
        improved = False
        for i in range(len(colnum)):
            orig = colnum[i]
            best_ng = orig
            best_local = cur
            for ng in range(0, max(colnum) + 2):
                if ng == orig:
                    continue
                colnum[i] = ng
                s = ngram.normalized_ngram_score(decrypt_myszkowski(text, colnum), table, 4)
                if s > best_local + 1e-9:
                    best_local = s
                    best_ng = ng
            colnum[i] = best_ng
            if best_ng != orig:
                cur = best_local
                improved = True
    return colnum, cur


def _myszkowski_sa(
    text: str,
    width: int,
    language: str,
    *,
    iters: int,
    seed: int,
    reheat_period: int,
    prefix: int | None,
    deadline: _Deadline,
    t0: float = 1.3,
    t_end: float = 0.02,
) -> tuple[list[int], float]:
    rng = random.Random(seed)
    table = _ngram_table(language, 4)

    def score(colnum: list[int]) -> float:
        dec = decrypt_myszkowski(text, colnum)
        return ngram.normalized_ngram_score(dec if prefix is None else dec[:prefix], table, 4)

    colnum = [rng.randint(0, width - 1) for _ in range(width)]
    cur = score(colnum)
    best = list(colnum)
    best_score = cur
    for k in range(iters):
        if (k & 1023) == 0 and deadline.expired():
            break
        frac = (k % reheat_period) / reheat_period
        temp = t0 * ((t_end / t0) ** frac)
        r = rng.random()
        save = list(colnum)
        if r < 0.70:
            i = rng.randrange(width)
            colnum[i] = rng.randint(0, max(colnum) + 1)
        elif r < 0.85:
            mg = max(colnum)
            if mg >= 1:
                a, b = rng.sample(range(mg + 1), 2)
                colnum = [b if x == a else a if x == b else x for x in colnum]
        else:
            i, j = rng.sample(range(width), 2)
            colnum[i], colnum[j] = colnum[j], colnum[i]
        s = score(colnum)
        delta = s - cur
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-6)):
            cur = s
            if s > best_score:
                best_score = s
                best = list(colnum)
        else:
            colnum = save
    return best, best_score


def search_myszkowski(
    text: str,
    language: str,
    *,
    deadline: _Deadline,
    seed: int = 0,
) -> list[_Candidate]:
    """Two-phase SA: rank widths cheaply, then concentrate seeds on the best."""

    n = len(text)
    wlo = 2
    whi = min(_env_int("DECIPHER_TRANSPOSITION_MYSZKOWSKI_MAX_WIDTH", 10), max(2, n // 3))
    phase1_iters = _env_int("DECIPHER_TRANSPOSITION_MYSZKOWSKI_PHASE1_ITERS", 6000)
    phase2_iters = _env_int("DECIPHER_TRANSPOSITION_MYSZKOWSKI_PHASE2_ITERS", 15000)
    phase2_seeds = _env_int("DECIPHER_TRANSPOSITION_MYSZKOWSKI_PHASE2_SEEDS", 5)
    phase2_widths = _env_int("DECIPHER_TRANSPOSITION_MYSZKOWSKI_PHASE2_WIDTHS", 2)
    reheat = _env_int("DECIPHER_TRANSPOSITION_MYSZKOWSKI_REHEAT", 3000)
    prefix = _env_int("DECIPHER_TRANSPOSITION_MYSZKOWSKI_PREFIX", 350)
    table = _ngram_table(language, 4)

    ranked: list[tuple[float, int, list[int]]] = []
    for width in range(wlo, whi + 1):
        if deadline.expired():
            break
        colnum, _ = _myszkowski_sa(
            text, width, language,
            iters=phase1_iters, seed=1, reheat_period=reheat,
            prefix=prefix if n > prefix else None, deadline=deadline,
        )
        colnum, sc = _myszkowski_polish(text, colnum, table)
        ranked.append((sc, width, colnum))
    ranked.sort(key=lambda item: item[0], reverse=True)

    out: list[_Candidate] = []
    for sc, width, colnum in ranked:
        pt = decrypt_myszkowski(text, colnum)
        out.append(_Candidate(pt, full_score(pt, language), "myszkowski", {"width": width, "colnum": colnum}))

    for _sc, width, _cn in ranked[: max(1, phase2_widths)]:
        best_cn = None
        best_inner = float("-inf")
        for s in range(phase2_seeds):
            if deadline.expired():
                break
            colnum, _ = _myszkowski_sa(
                text, width, language,
                iters=phase2_iters, seed=s * 7 + 3, reheat_period=reheat,
                prefix=None, deadline=deadline,
            )
            colnum, sc = _myszkowski_polish(text, colnum, table)
            if sc > best_inner:
                best_inner = sc
                best_cn = colnum
        if best_cn is not None:
            pt = decrypt_myszkowski(text, best_cn)
            out.append(_Candidate(pt, full_score(pt, language), "myszkowski", {"width": width, "colnum": best_cn}))
    return out


def _amsco_sa(
    text: str,
    width: int,
    start: int,
    language: str,
    *,
    iters: int,
    seed: int,
    reheat_period: int,
    deadline: _Deadline,
    t0: float = 1.0,
    t_end: float = 0.02,
) -> tuple[list[int], float]:
    rng = random.Random(seed)
    table = _ngram_table(language, 3)

    def score(order: list[int]) -> float:
        return ngram.normalized_ngram_score(decrypt_amsco_order(text, order, start), table, 3)

    order = list(range(width))
    rng.shuffle(order)
    cur = score(order)
    best = list(order)
    best_score = cur
    for k in range(iters):
        if (k & 1023) == 0 and deadline.expired():
            break
        frac = (k % reheat_period) / reheat_period
        temp = t0 * ((t_end / t0) ** frac)
        i, j = rng.sample(range(width), 2)
        order[i], order[j] = order[j], order[i]
        s = score(order)
        delta = s - cur
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-6)):
            cur = s
            if s > best_score:
                best_score = s
                best = list(order)
        else:
            order[i], order[j] = order[j], order[i]
    # greedy polish
    order = list(best)
    improved = True
    while improved and not deadline.expired():
        improved = False
        for i in range(width):
            for j in range(i + 1, width):
                order[i], order[j] = order[j], order[i]
                s = score(order)
                if s > best_score + 1e-9:
                    best_score = s
                    improved = True
                    best = list(order)
                else:
                    order[i], order[j] = order[j], order[i]
        order = list(best)
    return best, best_score


def search_amsco(
    text: str,
    language: str,
    *,
    deadline: _Deadline,
    seed: int = 0,
) -> list[_Candidate]:
    n = len(text)
    whi = min(_env_int("DECIPHER_TRANSPOSITION_AMSCO_MAX_WIDTH", 10), max(2, n // 4))
    iters = _env_int("DECIPHER_TRANSPOSITION_AMSCO_ITERS", 4000)
    seeds = _env_int("DECIPHER_TRANSPOSITION_AMSCO_SEEDS", 2)
    reheat = _env_int("DECIPHER_TRANSPOSITION_AMSCO_REHEAT", 2000)
    out: list[_Candidate] = []
    for width in range(3, whi + 1):
        if deadline.expired():
            break
        for start in (1, 2):
            if deadline.expired():
                break
            best_order = None
            best_inner = float("-inf")
            for s in range(seeds):
                order, sc = _amsco_sa(
                    text, width, start, language,
                    iters=iters, seed=s * 11 + seed + 1, reheat_period=reheat,
                    deadline=deadline,
                )
                if sc > best_inner:
                    best_inner = sc
                    best_order = order
            if best_order is not None:
                pt = decrypt_amsco_order(text, best_order, start)
                out.append(
                    _Candidate(pt, full_score(pt, language), "amsco", {"width": width, "start": start, "order": best_order})
                )
    return out


def search_nihilist(
    text: str,
    language: str,
    *,
    deadline: _Deadline,
) -> list[_Candidate]:
    import itertools

    n = len(text)
    hi = min(_env_int("DECIPHER_TRANSPOSITION_NIHILIST_MAX_BLOCK", 7), 9)
    out: list[_Candidate] = []
    for b in range(2, hi + 1):
        if deadline.expired():
            break
        block = b * b
        if block > n or n % block != 0:
            continue
        best_perm = None
        best_inner = float("-inf")
        tri = _ngram_table(language, 3)
        for perm in itertools.permutations(range(b)):
            if deadline.expired():
                break
            dec = decrypt_nihilist(text, list(perm))
            if dec is None:
                continue
            s = ngram.normalized_ngram_score(dec, tri, 3)
            if s > best_inner:
                best_inner = s
                best_perm = list(perm)
        if best_perm is not None:
            pt = decrypt_nihilist(text, best_perm)
            if pt is not None:
                out.append(
                    _Candidate(pt, full_score(pt, language), "nihilist_transposition", {"block": b, "perm": best_perm})
                )
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Map an ACA family hint (from cipher_system) to its dedicated strategy.
_FAMILY_STRATEGIES: dict[str, str] = {
    "columnar": "columnar",
    "columnar_transposition": "columnar",
    "railfence": "railfence",
    "rail_fence": "railfence",
    "redefence": "redefence",
    "redefense": "redefence",
    "myszkowski": "myszkowski",
    "amsco": "amsco",
    "nihilist_transposition": "nihilist",
    "nihilist": "nihilist",
}

_CHEAP_STRATEGIES = ("railfence", "redefence", "columnar", "nihilist")
_EXPENSIVE_STRATEGIES = ("amsco", "myszkowski")


def _hint_strategy(family_hint: str) -> str | None:
    hint = (family_hint or "").strip().lower()
    if not hint:
        return None
    # Route hints are handled by the existing pure-transposition screen, not here.
    if "route" in hint or "cadenus" in hint:
        return None
    for key, strat in _FAMILY_STRATEGIES.items():
        if key in hint:
            return strat
    return None


def _run_strategy(
    name: str,
    text: str,
    language: str,
    *,
    deadline: _Deadline,
    seed: int,
) -> list[_Candidate]:
    if name == "columnar":
        return search_columnar(text, language, deadline=deadline, seed=seed)
    if name == "railfence":
        return search_railfence(text, language, deadline=deadline)
    if name == "redefence":
        return search_redefence(text, language, deadline=deadline, seed=seed)
    if name == "myszkowski":
        return search_myszkowski(text, language, deadline=deadline, seed=seed)
    if name == "amsco":
        return search_amsco(text, language, deadline=deadline, seed=seed)
    if name == "nihilist":
        return search_nihilist(text, language, deadline=deadline)
    raise ValueError(f"unknown transposition strategy: {name}")


def solve_transposition(
    cipher_text: Any,
    *,
    language: str = "en",
    family_hint: str = "",
    budget_seconds: float | None = None,
    seed: int = 0,
    top_n: int = 10,
) -> dict[str, Any]:
    """Solve an A-Z transposition cipher with a budget-bounded permutation search.

    ``family_hint`` (the cipher_system, when known) routes directly to the
    matching strategy; when absent, a general cascade runs the cheap families
    first and escalates to annealing only if nothing already reads as language.

    Returns a result dict with ``status`` one of ``completed`` /
    ``not_applicable`` / ``error``, the best ``plaintext`` and ``score``, the
    winning ``family`` + ``params``, the ranked ``candidates``, and timing.
    """

    started = time.monotonic()
    text = az_string_from_cipher_text(cipher_text)
    if text is None or len(text) < 12:
        return {
            "status": "not_applicable",
            "solver": "transposition_solver",
            "reason": "ciphertext is not a usable A-Z letter stream",
            "plaintext": "",
            "score": None,
            "family": None,
            "params": {},
            "candidates": [],
            "strategies_run": [],
            "elapsed_seconds": round(time.monotonic() - started, 4),
        }
    # Route (spiral/diagonal grid) transpositions are handled by the existing
    # grid/route transform screen, not by this keyed-column permutation search.
    if "route" in (family_hint or "").lower():
        return {
            "status": "not_applicable",
            "solver": "transposition_solver",
            "reason": "route transpositions are handled by the grid/route transform screen",
            "plaintext": "",
            "score": None,
            "family": None,
            "params": {},
            "candidates": [],
            "strategies_run": [],
            "elapsed_seconds": round(time.monotonic() - started, 4),
        }

    budget = budget_seconds if budget_seconds is not None else _default_budget_seconds()
    deadline = _Deadline.after(budget)

    strategy = _hint_strategy(family_hint)
    strategies_run: list[str] = []
    all_candidates: list[_Candidate] = []

    def record(name: str, cands: list[_Candidate]) -> None:
        strategies_run.append(name)
        all_candidates.extend(cands)

    if strategy is not None:
        record(strategy, _run_strategy(strategy, text, language, deadline=deadline, seed=seed))
        # For columnar/nihilist, myszkowski is a keyed-column cousin; if the
        # dedicated strategy did not clearly solve, try the cousin within budget.
        best = _best_candidate(all_candidates)
        if strategy in {"columnar", "nihilist"} and (best is None or _dict_rate(best.plaintext, language) < _SOLVED_DICT_RATE) and not deadline.expired():
            record("myszkowski", _run_strategy("myszkowski", text, language, deadline=deadline, seed=seed))
    else:
        # General cascade: cheap families first, escalate only if unsolved.
        for name in _CHEAP_STRATEGIES:
            if deadline.expired():
                break
            record(name, _run_strategy(name, text, language, deadline=deadline, seed=seed))
        best = _best_candidate(all_candidates)
        solved = best is not None and _dict_rate(best.plaintext, language) >= _SOLVED_DICT_RATE
        if not solved:
            for name in _EXPENSIVE_STRATEGIES:
                if deadline.expired():
                    break
                record(name, _run_strategy(name, text, language, deadline=deadline, seed=seed))

    # Keyed-columnar F2 escalation. The SA column-order search (the `columnar`
    # strategy) misses some keyed columnar orderings (measured: width-11 keyword
    # misses). When it ran but the incumbent is still below the solved dict-rate
    # threshold, run the dedicated keyed-columnar search (analysis.columnar_search)
    # with the language scorer and adopt its top finalist iff its full_score beats
    # the incumbent (same adopt-if-better convention as the runner's additive
    # block). Bounded by the shared deadline; skipped when too little budget
    # remains. Never raises: on error the incumbent is kept and a note recorded.
    keyed_columnar_f2: dict[str, Any] | None = None
    best = _best_candidate(all_candidates)
    if (
        "columnar" in strategies_run
        and (best is None or _dict_rate(best.plaintext, language) < _SOLVED_DICT_RATE)
        and deadline.remaining() >= _F2_MIN_REMAINING_SECONDS
    ):
        try:
            from analysis.columnar_search import (
                ColumnarSearchConfig,
                make_language_scorer,
                search_keyed_columnar,
            )
        except Exception as exc:  # pragma: no cover — packaging failure only
            keyed_columnar_f2 = {
                "ran": False,
                "adopted": False,
                "method": None,
                "score": None,
                "note": f"keyed_columnar_f2 unavailable (import failed): {exc}",
            }
        else:
            try:
                # The hill-climb has no internal deadline hook; its cost is
                # ~9s at the default 64 restarts on a ~250-char stream and
                # scales ~linearly with length (every local-search step scores
                # the full stream). Scale the restart budget to the remaining
                # wall-clock so the escalation cannot badly overshoot a
                # nearly-spent deadline; exhaustive widths (<= 8) are cheap
                # and unaffected by the restart knob.
                remaining = deadline.remaining()
                restarts = max(
                    8,
                    min(64, int(64 * remaining * 250.0 / (9.0 * max(1, len(text))))),
                )
                finalists = search_keyed_columnar(
                    text,
                    make_language_scorer(language),
                    config=ColumnarSearchConfig(seed=seed, restarts=restarts),
                )
                # The language scorer ranks finalists ngram-only; adoption is
                # judged on full_score (ngram + dict weight), so pick the
                # finalist that maximizes the ADOPTION metric (<= top_n of
                # them, cheap to score).
                scored = [
                    (full_score(f.decoded_stream, language), f) for f in finalists
                ]
                f2_score: float | None = None
                finalist = None
                if scored:
                    f2_score, finalist = max(scored, key=lambda pair: pair[0])
                adopted = False
                if finalist is not None:
                    incumbent = best.score if best is not None else float("-inf")
                    if f2_score > incumbent:
                        all_candidates.append(
                            _Candidate(
                                finalist.decoded_stream,
                                f2_score,
                                "columnar",
                                {
                                    "width": finalist.column_count,
                                    "order": list(finalist.column_order),
                                    "keyword": finalist.keyword,
                                    "engine": "columnar_search_f2",
                                    "method": finalist.method,
                                },
                            )
                        )
                        adopted = True
                keyed_columnar_f2 = {
                    "ran": True,
                    "adopted": adopted,
                    "method": finalist.method if finalist is not None else None,
                    "score": round(f2_score, 6) if f2_score is not None else None,
                }
            except Exception as exc:  # never let the escalation break the solve
                keyed_columnar_f2 = {
                    "ran": True,
                    "adopted": False,
                    "method": None,
                    "score": None,
                    "note": (
                        "keyed_columnar_f2 escalation raised, kept the "
                        f"incumbent: {exc}"
                    ),
                }

    all_candidates.sort(key=lambda c: c.score, reverse=True)
    best = all_candidates[0] if all_candidates else None
    if best is None:
        return {
            "status": "error",
            "solver": "transposition_solver",
            "reason": "no candidate produced (budget exhausted before any strategy ran)",
            "plaintext": "",
            "score": None,
            "family": None,
            "params": {},
            "candidates": [],
            "strategies_run": strategies_run,
            "elapsed_seconds": round(time.monotonic() - started, 4),
        }

    # De-duplicate identical plaintexts, keep best-scored.
    seen: set[str] = set()
    ranked: list[_Candidate] = []
    for c in all_candidates:
        if c.plaintext in seen:
            continue
        seen.add(c.plaintext)
        ranked.append(c)

    result: dict[str, Any] = {
        "status": "completed",
        "solver": "transposition_solver",
        "language": language,
        "family_hint": family_hint,
        "budget_seconds": budget,
        "plaintext": best.plaintext,
        "score": round(best.score, 6),
        "family": best.family,
        "params": dict(best.params),
        "dict_rate": round(_dict_rate(best.plaintext, language), 4),
        "strategies_run": strategies_run,
        "candidate_count": len(all_candidates),
        "candidates": [c.to_dict() for c in ranked[:top_n]],
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "note": (
            "Permutation-search transposition solver (columnar / railfence / "
            "redefence / myszkowski / amsco / nihilist) scored by the existing "
            "dictionary + n-gram language model."
        ),
    }
    if keyed_columnar_f2 is not None:
        result["keyed_columnar_f2"] = keyed_columnar_f2
    return result


def _best_candidate(cands: list[_Candidate]) -> Optional[_Candidate]:
    if not cands:
        return None
    return max(cands, key=lambda c: c.score)
