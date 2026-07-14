"""Shuffle-null and parametric baselines for order-sensitive cipher statistics.

INV-0 Part 3. Many diagnostic statistics (best-period IC peak, monotone-run
length, front-loading index, repeated-ngram excess) only mean something relative
to a null distribution. A raw "the best period-k IC is 0.066" is meaningless
until we know that a frequency-preserving reshuffle of the *same* multiset almost
never reaches 0.066. This module provides that null in one deterministic,
process-stable place.

Two null families are provided:

* :func:`null_percentile` — a **frequency-preserving permutation** null: the
  token multiset is fixed, only the ORDER is destroyed. Use for order-sensitive
  statistics (periodic IC recovery, monotone runs, repeated n-grams).
* :func:`parametric_percentile` — a **parametric** null where each draw is an
  i.i.d. sample from an explicit generator (e.g. uniform on ``[min, max]``). Use
  for value-multiset statistics that a permutation leaves invariant (the
  front-loading index, the ``values mod m`` chi-square).

Seeding is deterministic and identical across processes: it is derived by SHA-256
from the statistic name, an optional per-investigation namespace, and the token
stream itself (via :func:`encode_tokens`). Never ``hash()`` (salted per process),
never the global ``random`` module.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, Sequence

__all__ = [
    "encode_tokens",
    "tokens_digest",
    "derive_seed",
    "null_percentile",
    "parametric_percentile",
]


def encode_tokens(values: Sequence[int]) -> bytes:
    """Canonical byte encoding of an integer token stream (finding 6c).

    Each value is encoded as 8 big-endian bytes, so Beale's max value 2906 (and
    anything else that overflows a single byte) round-trips unambiguously. This
    single definition is reused for the seed hash, the Part-3 cache key, and the
    Part-6 ``view_hash``.

    Values must be non-negative (token ids and numeric ciphertext values always
    are); a negative value raises ``OverflowError`` from ``int.to_bytes``.
    """
    return b"".join(int(v).to_bytes(8, "big") for v in values)


def tokens_digest(values: Sequence[int]) -> str:
    """Hex SHA-256 of :func:`encode_tokens` — the cache/view identity of a stream."""
    return hashlib.sha256(encode_tokens(values)).hexdigest()


def derive_seed(statistic_name: str, namespace: str, values: Sequence[int]) -> int:
    """Deterministic 64-bit seed for a (statistic, namespace, stream) triple."""
    inner = hashlib.sha256(encode_tokens(values)).digest()
    material = statistic_name.encode() + b"|" + namespace.encode() + b"|" + inner
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


# Process-local cache keyed by (statistic_name, namespace, tokens_digest,
# n_shuffles, tail, kind). Values are the full baseline dicts.
_CACHE: dict[tuple, dict[str, Any]] = {}


def _summarise(
    observed: float,
    draws: list[float],
    *,
    tail: str,
    seed: int,
    kind: str,
) -> dict[str, Any]:
    n = len(draws)
    ge = sum(1 for d in draws if d >= observed)
    le = sum(1 for d in draws if d <= observed)
    lt = sum(1 for d in draws if d < observed)
    upper_p = (ge + 1) / (n + 1)
    lower_p = (le + 1) / (n + 1)
    if tail == "upper":
        p_value = upper_p
    elif tail == "lower":
        p_value = lower_p
    elif tail == "two_sided":
        p_value = min(1.0, 2.0 * min(upper_p, lower_p))
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown tail: {tail!r}")
    null_mean = sum(draws) / n if n else 0.0
    null_std = (sum((d - null_mean) ** 2 for d in draws) / n) ** 0.5 if n else 0.0
    percentile = 100.0 * lt / n if n else 0.0
    return {
        "observed": observed,
        "tail": tail,
        "percentile": percentile,
        "p_value": p_value,
        "n_shuffles": n,
        "null_mean": null_mean,
        "null_std": null_std,
        "seed": seed,
        "kind": kind,
    }


def null_percentile(
    statistic_fn: Callable[[list[int]], float],
    tokens: Sequence[int],
    *,
    tail: str = "upper",
    n_shuffles: int = 1000,
    statistic_name: str,
    namespace: str = "",
) -> dict[str, Any]:
    """Frequency-preserving permutation null for an order-sensitive statistic.

    ``observed = statistic_fn(list(tokens))``; each of ``n_shuffles`` draws
    reshuffles a copy of the token multiset and recomputes ``statistic_fn``.

    p-value (finding 6b): with ``n`` draws,
    ``upper = (#{draw >= observed} + 1)/(n+1)``,
    ``lower = (#{draw <= observed} + 1)/(n+1)``,
    ``two_sided = min(1, 2*min(upper, lower))``. The tightest reportable bound
    when the observation exceeds every draw is ``p < 1/(n+1)``; a "percentile
    >= 99.9" claim therefore requires ``n_shuffles >= 1000``.
    """
    values = list(tokens)
    digest = tokens_digest(values)
    key = (statistic_name, namespace, digest, n_shuffles, tail, "shuffle")
    if key in _CACHE:
        return dict(_CACHE[key])

    seed = derive_seed(statistic_name, namespace, values)
    rng = random.Random(seed)
    observed = float(statistic_fn(list(values)))
    pool = list(values)
    draws: list[float] = []
    for _ in range(n_shuffles):
        rng.shuffle(pool)
        draws.append(float(statistic_fn(pool)))

    result = _summarise(observed, draws, tail=tail, seed=seed, kind="shuffle")
    _CACHE[key] = dict(result)
    return result


def parametric_percentile(
    statistic_fn: Callable[[list[int]], float],
    tokens: Sequence[int],
    sampler: Callable[[random.Random, int], list[int]],
    *,
    tail: str = "upper",
    n_shuffles: int = 1000,
    statistic_name: str,
    namespace: str = "",
) -> dict[str, Any]:
    """Parametric i.i.d. null (finding 6, same-class gap).

    ``observed = statistic_fn(list(tokens))``; each draw is ``sampler(rng, n)``
    (an i.i.d. sample of the same length ``n``), e.g. uniform on ``[min, max]``.
    Used for value-multiset statistics a permutation leaves invariant
    (front-loading index, ``values mod m`` chi-square). Labelled
    ``kind="parametric"``.
    """
    values = list(tokens)
    n = len(values)
    digest = tokens_digest(values)
    key = (statistic_name, namespace, digest, n_shuffles, tail, "parametric")
    if key in _CACHE:
        return dict(_CACHE[key])

    seed = derive_seed(statistic_name, namespace, values)
    rng = random.Random(seed)
    observed = float(statistic_fn(list(values)))
    draws: list[float] = []
    for _ in range(n_shuffles):
        draws.append(float(statistic_fn(sampler(rng, n))))

    result = _summarise(observed, draws, tail=tail, seed=seed, kind="parametric")
    _CACHE[key] = dict(result)
    return result
