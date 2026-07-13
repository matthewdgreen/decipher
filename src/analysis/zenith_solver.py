"""Zenith-parity homophonic solver for Zodiac-style ciphers.

This module is a faithful Python port/adaptation of the Zenith homophonic
simulated annealing solver from the Zenith project by beldenge:
https://github.com/beldenge/Zenith

Decipher is distributed under GPLv3 so this derived implementation can be
redistributed ethically and with clear attribution. Please preserve this
attribution when moving or modifying this file.

The implementation is kept isolated from the existing ``homophonic.py`` so no
existing functionality is disturbed while we maintain a direct, auditable link
between the Zenith-derived algorithm and the rest of Decipher.

Key differences from the ``zenith_exact`` score profile in homophonic.py (which
scored 7.8% on Zodiac 408):

1. **Correct counterweight**: Zenith divides by ``entropy^(1/2.75)`` where entropy is
   Shannon entropy in bits.  The old profile multiplied by ``IoC^(1/6)`` — a
   completely different metric with a different gradient shape.

2. **Un-normalized acceptance criterion**: Zenith's Boltzmann criterion uses the raw
   score delta: ``exp(delta / temp)``.  The old code divided delta by ``ngram_count``
   (≈202 for Zodiac 408), making the effective temperature ~200× too low and
   rejecting almost every downhill move.

3. **Binary model loader**: loads the 45 MB ``zenith-model.array.bin`` file directly
   into a numpy float32 array for O(1) 5-gram lookups, instead of the slower CSV
   dict approach.
"""
from __future__ import annotations

import json
import math
import random
import struct
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from analysis.homophonic import HomophonicAnnealResult, HomophonicCandidate


# ---------------------------------------------------------------------------
# Model dataclass
# ---------------------------------------------------------------------------

# Default alphabet for v1 (26 lowercase letters) — kept as a module constant so
# both the loader and the dataclass default agree.
_V1_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class ZenithModel:
    """Loaded Zenith language model: ``base**order`` float32 log-prob array + metadata.

    For legacy v1 models this is the classic 26^5 lowercase-letter table.  The
    ``alphabet``/``order`` fields generalise the index math so ``zenith_binary_v2``
    models over an arbitrary symbol alphabet and order load and look up correctly.
    """

    log_probs: np.ndarray           # shape (base**order,), dtype float32, natural-log probs
    unknown_log_prob: float         # floor for any n-gram not in the training corpus
    letter_freq: dict[str, float]   # symbol → unigram probability (from corpus)
    order: int = 5
    alphabet: str = _V1_ALPHABET    # ordered symbol alphabet; index i → symbol alphabet[i]

    # Pre-computed powers for the index formula (derived from ``alphabet``/``order``
    # in ``__post_init__``).  ``_P4``/``_P3``/``_P2`` remain available for the
    # order-5 hot path used by the solver; for base=26 they equal 26**4/26**3/26**2.
    _base: int = field(default=26, init=False, repr=False)
    _P4: int = field(default=26**4, init=False, repr=False)
    _P3: int = field(default=26**3, init=False, repr=False)
    _P2: int = field(default=26**2, init=False, repr=False)
    _powers: tuple[int, ...] = field(default=(), init=False, repr=False)
    _index_of: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        base = len(self.alphabet) if self.alphabet else 26
        self._base = base
        # Order-5 hot-path powers (identical to the historical hardcoded values
        # when base == 26).
        self._P4 = base ** 4
        self._P3 = base ** 3
        self._P2 = base ** 2
        # Generic positional powers: big-endian, most-significant symbol first.
        self._powers = tuple(base ** (self.order - 1 - i) for i in range(self.order))
        self._index_of = {ch: i for i, ch in enumerate(self.alphabet)}

    def lookup_lo(self, a: int, b: int, c: int, d: int, e: int) -> float:
        """Return log-prob for the 5-gram encoded as five 0-(base-1) offsets."""
        idx = (a * self._P4 + b * self._P3 + c * self._P2 + d * self._base + e)
        return float(self.log_probs[idx])

    def lookup_indices(self, indices) -> float:
        """Return log-prob for an ``order``-length sequence of symbol offsets.

        Generic replacement for :meth:`lookup_lo` that works for any
        ``(alphabet, order)``.  Out-of-range offsets or a wrong-length sequence
        fall back to the floor.
        """
        if len(indices) != self.order:
            return self.unknown_log_prob
        idx = 0
        base = self._base
        for off, power in zip(indices, self._powers):
            if not (0 <= off < base):
                return self.unknown_log_prob
            idx += off * power
        return float(self.log_probs[idx])

    def lookup(self, gram: str) -> float:
        """Return log-prob for an ``order``-length symbol string.  Falls back to floor."""
        if len(gram) != self.order:
            return self.unknown_log_prob
        idx = 0
        index_of = self._index_of
        for ch, power in zip(gram, self._powers):
            off = index_of.get(ch)
            if off is None:
                return self.unknown_log_prob
            idx += off * power
        return float(self.log_probs[idx])


# ---------------------------------------------------------------------------
# Binary model loader
# ---------------------------------------------------------------------------

_MAGIC = 0x5A4D4D43       # "ZMMC" — v1 headerless-26^5 Zenith model
_MAGIC_V2 = 0x5A4D4332    # "ZMC2" — v2 (alphabet, order) container
_VERSION = 1
_VERSION_V2 = 2
_NGRAM_ARRAY_LEN = 26 ** 5  # 11_881_376
_DEFAULT_UNKNOWN_LOG_PROB = -30.0   # floor used when unknownProbability <= 0
_MAX_V2_ORDER = 12   # sanity bound on header order — avoids huge base**order on corrupt headers


def load_zenith_binary_model(path: str | Path) -> ZenithModel:
    """Load a Zenith binary model into a :class:`ZenithModel`.

    Two container formats are supported and distinguished by sniffing the 4-byte
    magic:

    **v1** (magic ``0x5A4D4D43``) — the classic 26^5 lowercase-letter table,
    Java ``DataOutputStream`` (big-endian).  Parsed byte-for-byte as before:

    .. code-block:: text

        4B  magic        = 0x5A4D4D43
        4B  version      = 1
        4B  order        = 5
        4B  maxNGramsToKeep
        4B  unknownProbability  (float32)
        4B  totalNodes
        4B  firstOrderCount     (= 26)
        per firstOrderNode:
            2B  char (UTF-16 BE)
            8B  count (int64)
            8B  logProbability (float64)
        4B  arrayLength = 26^5
        <arrayLength × 4B>  nGramLogProbabilities (float32)

    **v2** (magic ``0x5A4D4332``) — a generalised (alphabet, order) container
    (big-endian):

    .. code-block:: text

        4B  magic          = 0x5A4D4332
        4B  version        = 2
        4B  order          = K
        4B  alphabetLen    = N (number of symbols / code points)
        4B  alphabetBytes  = L (UTF-8 byte length of the alphabet string)
        LB  alphabet       (UTF-8)
        4B  unknownProbability  (float32)
        4B  arrayLength    = N^K
        <arrayLength × 4B>  nGramLogProbabilities (float32)

    In both cases, when a ``<path>.metadata.json`` sidecar carrying an
    ``unknown_log_prob`` value is present, that value overrides the floor parsed
    from the binary (see :func:`_read_sidecar_unknown_log_prob`).
    """
    return _load_cached(str(Path(path).expanduser().resolve()))


@lru_cache(maxsize=16)
def _load_cached(path: str) -> ZenithModel:
    """LRU-cached loader — the model is large, only load once per path."""
    with open(path, "rb") as fh:
        magic_bytes = fh.read(4)
        if len(magic_bytes) < 4:
            raise ValueError("Not a Zenith model file (truncated header)")
        magic = struct.unpack(">I", magic_bytes)[0]
        if magic == _MAGIC:
            model = _read_v1(fh)
        elif magic == _MAGIC_V2:
            model = _read_v2(fh)
        else:
            raise ValueError(
                f"Not a Zenith model file (bad magic 0x{magic:08X}); expected 0x{_MAGIC:08X}"
            )

    # Sidecar metadata may override the unknown-n-gram floor.
    override = _read_sidecar_unknown_log_prob(path)
    if override is not None:
        model.unknown_log_prob = override
    return model


def _read_v1(fh) -> ZenithModel:
    """Parse a v1 model body (``fh`` positioned right after the 4-byte magic)."""
    version, order, max_ngrams = struct.unpack(">III", fh.read(12))
    if version != _VERSION:
        raise ValueError(f"Unsupported Zenith model version {version}")
    if order != 5:
        raise ValueError(f"Only order-5 models are supported; got order={order}")

    unknown_prob: float = struct.unpack(">f", fh.read(4))[0]
    unknown_log_prob = math.log(unknown_prob) if unknown_prob > 0 else _DEFAULT_UNKNOWN_LOG_PROB

    total_nodes, first_order_count = struct.unpack(">II", fh.read(8))

    letter_freq: dict[str, float] = {}
    total_unigram_count = 0
    unigram_counts: dict[str, int] = {}

    for _ in range(first_order_count):
        # char: 2-byte UTF-16 BE
        char_code = struct.unpack(">H", fh.read(2))[0]
        count = struct.unpack(">q", fh.read(8))[0]   # int64
        log_prob = struct.unpack(">d", fh.read(8))[0]  # float64
        letter = chr(char_code).upper()
        if "A" <= letter <= "Z":
            unigram_counts[letter] = count
            total_unigram_count += count
            _ = log_prob  # not needed; we recompute from counts

    for letter, count in unigram_counts.items():
        letter_freq[letter] = count / max(1, total_unigram_count)

    array_length = struct.unpack(">I", fh.read(4))[0]
    if array_length != _NGRAM_ARRAY_LEN:
        raise ValueError(
            f"Unexpected array length {array_length}; expected {_NGRAM_ARRAY_LEN}"
        )

    raw_bytes = fh.read(array_length * 4)
    if len(raw_bytes) != array_length * 4:
        raise ValueError(
            f"Truncated model file: expected {array_length * 4} bytes, "
            f"got {len(raw_bytes)}"
        )

    # Big-endian float32 → native float32 numpy array
    log_probs = np.frombuffer(raw_bytes, dtype=">f4").astype(np.float32)

    return ZenithModel(
        log_probs=log_probs,
        unknown_log_prob=unknown_log_prob,
        letter_freq=letter_freq,
        order=5,
        alphabet=_V1_ALPHABET,
    )


def _read_v2(fh) -> ZenithModel:
    """Parse a v2 model body (``fh`` positioned right after the 4-byte magic)."""
    version, order, alphabet_len, alphabet_bytes_len = struct.unpack(">IIII", fh.read(16))
    if version != _VERSION_V2:
        raise ValueError(f"Unsupported Zenith v2 model version {version}")
    if not (1 <= order <= _MAX_V2_ORDER):
        raise ValueError(
            f"Invalid model order {order}; expected 1..{_MAX_V2_ORDER}"
        )
    if alphabet_len < 1:
        raise ValueError(f"Invalid v2 alphabet length {alphabet_len}")

    alphabet_bytes = fh.read(alphabet_bytes_len)
    if len(alphabet_bytes) != alphabet_bytes_len:
        raise ValueError("Truncated alphabet in v2 model header")
    alphabet = alphabet_bytes.decode("utf-8")
    if len(alphabet) != alphabet_len:
        raise ValueError(
            f"Alphabet length mismatch: header says {alphabet_len} symbols, "
            f"decoded {len(alphabet)}"
        )
    base = alphabet_len

    unknown_prob: float = struct.unpack(">f", fh.read(4))[0]
    unknown_log_prob = math.log(unknown_prob) if unknown_prob > 0 else _DEFAULT_UNKNOWN_LOG_PROB

    array_length = struct.unpack(">I", fh.read(4))[0]
    expected = base ** order
    if array_length != expected:
        raise ValueError(
            f"Unexpected array length {array_length}; expected {expected} "
            f"(base {base} ^ order {order})"
        )

    raw_bytes = fh.read(array_length * 4)
    if len(raw_bytes) != array_length * 4:
        raise ValueError(
            f"Truncated model file: expected {array_length * 4} bytes, "
            f"got {len(raw_bytes)}"
        )
    log_probs = np.frombuffer(raw_bytes, dtype=">f4").astype(np.float32)

    # v2 does not carry per-symbol counts; use a uniform prior keyed by the
    # alphabet symbols themselves.
    letter_freq = {ch: 1.0 / base for ch in alphabet} if base else {}

    return ZenithModel(
        log_probs=log_probs,
        unknown_log_prob=unknown_log_prob,
        letter_freq=letter_freq,
        order=order,
        alphabet=alphabet,
    )


def _read_sidecar_unknown_log_prob(path: str) -> float | None:
    """Return ``unknown_log_prob`` from a ``<path>.metadata.json`` sidecar, if present.

    The value in the sidecar is already a natural-log floor (matching the
    ``model_metadata`` pattern used elsewhere).  Returns ``None`` when there is no
    sidecar, it cannot be parsed, or it does not carry a numeric
    ``unknown_log_prob`` — in which case the loader keeps the floor parsed from
    the binary.
    """
    meta_path = Path(path + ".metadata.json")
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("unknown_log_prob")
    if isinstance(value, bool):  # guard: bool is a subclass of int
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


# ---------------------------------------------------------------------------
# Entropy (Zenith's exact counterweight)
# ---------------------------------------------------------------------------

def _precompute_entropy_table(cipher_len: int) -> np.ndarray:
    """Precomputed table: ``table[i] = |log2(i/N) * (i/N)|`` for i=0..N.

    This mirrors Zenith's ``EntropyEvaluator.precompute(cipher)``.  ``N`` is the
    total number of decoded characters (= ``len(tokens)``).
    """
    table = np.zeros(cipher_len + 1, dtype=np.float64)
    for i in range(1, cipher_len + 1):
        p = i / cipher_len
        table[i] = abs(math.log2(p) * p)
    return table


def _compute_entropy(
    letter_counts: np.ndarray,
    entropy_table: np.ndarray,
) -> float:
    """Shannon entropy (bits) from a 26-element letter-count array.

    Uses the precomputed table so no log calls are needed at solve time.
    """
    return float(entropy_table[letter_counts].sum())


def _zenith_score(sum_log_prob: float, ngram_count: int, entropy: float) -> float:
    """Zenith's objective: ``mean_log_prob / entropy^(1/2.75)``.

    Entropy is Shannon entropy in bits (always positive for non-uniform dists).
    Higher score is better.  Returns ``-inf`` when entropy ≤ 0 to avoid ZeroDivision.
    """
    if entropy <= 0.0:
        return float("-inf")
    mean = sum_log_prob / max(1, ngram_count)
    return mean / (entropy ** (1.0 / 2.75))


# ---------------------------------------------------------------------------
# Proposal bucket
# ---------------------------------------------------------------------------

def _build_biased_bucket(
    letter_freq: dict[str, float],
    flatten_weight: float = 0.8,
    scale: int = 1000,
) -> list[str]:
    """80% flat + 20% corpus frequency bucket (lowercase), mirroring Zenith's init.

    Uses the actual unigram frequencies loaded from the binary model rather than
    a hardcoded table.
    """
    available = sorted(k for k in letter_freq if "A" <= k <= "Z")
    if not available:
        return list("abcdefghijklmnopqrstuvwxyz")

    flat_mass = (1.0 / len(available)) * flatten_weight
    bucket: list[str] = []
    for letter in available:
        prob = letter_freq.get(letter, 0.0)
        scaled = prob * (1.0 - flatten_weight)
        bias = max(1, int(scale * (scaled + flat_mass)))
        bucket.extend([letter.lower()] * bias)
    return bucket


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def _make_window_starts(text_len: int, order: int, step: int) -> list[int]:
    """Return the sorted list of window start positions for the given step."""
    max_start = text_len - order
    if max_start < 0:
        return []
    return list(range(0, max_start + 1, step))


def _precompute_sym_affected(
    mutable_symbol_ids: list[int],
    occurrences: dict[int, list[int]],
    text_len: int,
    order: int,
    step: int,
) -> dict[int, list[int]]:
    """Precompute (once) the sorted list of window *indices* affected by each symbol.

    A window index ``wi`` corresponds to start position ``wi * step``.
    """
    max_start = text_len - order
    result: dict[int, list[int]] = {}
    for sid in mutable_symbol_ids:
        idx_set: set[int] = set()
        for pos in occurrences[sid]:
            lo = max(0, pos - order + 1)
            hi = min(pos, max_start)
            for s in range(lo, hi + 1):
                if s % step == 0:
                    idx_set.add(s // step)
        result[sid] = sorted(idx_set)
    return result


def _lookup_window_lo(chars_lo: list[int], s: int, model: ZenithModel) -> float:
    """Fast 5-gram lookup using pre-lowercased 0-25 integer array."""
    return model.lookup_lo(
        chars_lo[s], chars_lo[s + 1], chars_lo[s + 2],
        chars_lo[s + 3], chars_lo[s + 4],
    )


def _full_window_scores(
    chars_lo: list[int],
    model: ZenithModel,
    window_starts: list[int],
) -> np.ndarray:
    """Score every window from scratch; returns array of length ngram_count."""
    scores = np.empty(len(window_starts), dtype=np.float64)
    for k, s in enumerate(window_starts):
        scores[k] = _lookup_window_lo(chars_lo, s, model)
    return scores


# ---------------------------------------------------------------------------
# Occurrence map
# ---------------------------------------------------------------------------

def _occurrence_map(tokens: list[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, t in enumerate(tokens):
        out.setdefault(t, []).append(i)
    return out


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _random_key(
    symbol_ids: list[int],
    bucket: list[str],
    letter_to_id: dict[str, int],
    rng: random.Random,
    initial_key: dict[int, int] | None,
    plaintext_ids: list[int],
) -> dict[int, int]:
    """Random initialisation from biased bucket, optionally seeded from initial_key."""
    key = {}
    fallback = plaintext_ids[0]
    for sid in symbol_ids:
        if initial_key and sid in initial_key:
            key[sid] = initial_key[sid]
            continue
        letter = rng.choice(bucket).upper()
        key[sid] = letter_to_id.get(letter, fallback)
    return key


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def zenith_solve(
    tokens: list[int],
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    model: ZenithModel,
    initial_key: dict[int, int] | None = None,
    fixed_cipher_ids: set[int] | None = None,
    epochs: int = 10,
    sampler_iterations: int = 5000,
    t_start: float = 0.012,
    t_end: float = 0.006,
    seed: int | None = None,
    top_n: int = 1,
    epoch_callback: Callable[[dict], bool] | None = None,
) -> HomophonicAnnealResult:
    """Exact Zenith SA for homophonic ciphers.

    Correctness properties:

    * Score formula: ``mean_5gram_log_prob / entropy^(1/2.75)`` (Shannon entropy, bits)
    * Acceptance:  ``exp(delta / temp)`` — **no** normalization by ngram_count
    * Window step: ``order // 2 = 2``
    * Proposal bucket: 80% flat + 20% corpus frequency (from binary model)
    * Epoch structure: fresh random init each epoch, keep best overall
    """
    # The SA solver's proposal bucket, chars_lo encoding, and 26-slot letter
    # counts all assume the classic 26-letter order-5 (v1-shaped) model.  A v2
    # model with base > 26 would silently garbage-score (indices stay in
    # bounds), and base < 26 or order != 5 would crash mid-anneal — reject
    # cleanly up front instead.
    base = len(getattr(model, "alphabet", _V1_ALPHABET) or _V1_ALPHABET)
    if base != 26 or model.order != 5:
        raise ValueError(
            "zenith_solve requires a 26-letter order-5 (v1-shaped) model; "
            f"got base={base}, order={model.order}"
        )

    rng = random.Random(seed)
    t0 = time.time()

    order = model.order       # 5
    step = order // 2         # 2  (Zenith's stepSize)
    text_len = len(tokens)

    if text_len < order:
        raise ValueError(
            f"Cipher text too short for order-{order} model (length={text_len})"
        )

    fixed_cipher_ids = set(fixed_cipher_ids or ())
    symbol_ids = sorted(set(tokens))
    mutable_symbol_ids = [sid for sid in symbol_ids if sid not in fixed_cipher_ids]

    occurrences = _occurrence_map(tokens)
    bucket = _build_biased_bucket(model.letter_freq)

    window_starts = _make_window_starts(text_len, order, step)
    ngram_count = len(window_starts)

    # Precompute which window indices each symbol affects (constant for all epochs)
    sym_aff = _precompute_sym_affected(
        mutable_symbol_ids, occurrences, text_len, order, step
    )

    # Precompute entropy table (depends only on cipher_len, constant)
    entropy_table = _precompute_entropy_table(text_len)

    # Track best over all epochs
    best_key: dict[int, int] = {}
    best_chars_lo: list[int] = []
    best_total = float("-inf")
    accepted_moves = 0
    improved_moves = 0
    candidate_pool: list[HomophonicCandidate] = []
    stopped_early = False
    stop_reason: str | None = None

    for epoch in range(max(1, epochs)):
        # Fresh random initialisation each epoch (use initial_key only on epoch 0)
        key = _random_key(
            symbol_ids, bucket, letter_to_id, rng,
            initial_key if epoch == 0 else None,
            plaintext_ids,
        )
        # Apply fixed mappings
        for sid in fixed_cipher_ids:
            if initial_key and sid in initial_key:
                key[sid] = initial_key[sid]

        # Build chars_lo: list of 0-25 offsets (lowercase), parallel to tokens
        chars_lo: list[int] = [ord(id_to_letter[key[t]]) - ord("A") for t in tokens]

        # Initial window scores
        window_scores = _full_window_scores(chars_lo, model, window_starts)
        sum_log_prob = float(window_scores.sum())

        # letter_counts[i] = number of positions with letter i (0=A, ..., 25=Z)
        letter_counts = np.zeros(26, dtype=np.int64)
        for lo in chars_lo:
            letter_counts[lo] += 1

        entropy = _compute_entropy(letter_counts, entropy_table)
        current_score = _zenith_score(sum_log_prob, ngram_count, entropy)

        epoch_best_score = current_score
        epoch_best_key = dict(key)
        epoch_best_chars_lo = list(chars_lo)
        epoch_accepted = 0
        epoch_improved = 0

        if current_score > best_total:
            best_total = current_score
            best_key = dict(key)
            best_chars_lo = list(chars_lo)

        temp_span = t_start - t_end

        for si in range(max(1, sampler_iterations)):
            # Linear cooling: t_start → t_end
            ratio = (sampler_iterations - si) / max(1, sampler_iterations)
            temp = max(t_end + temp_span * ratio, 1e-12)

            for cipher_sym in mutable_symbol_ids:
                new_letter_lo = ord(rng.choice(bucket)) - ord("a")   # 0-25
                old_letter_lo = chars_lo[occurrences[cipher_sym][0]]  # same for all positions

                if new_letter_lo == old_letter_lo:
                    continue

                positions = occurrences[cipher_sym]
                aff = sym_aff[cipher_sym]

                if not aff:
                    # Symbol has no affected windows (very short text edge case)
                    continue

                # Old contribution to sum
                old_contrib = float(window_scores[aff].sum())

                # Tentatively apply change
                for pos in positions:
                    chars_lo[pos] = new_letter_lo
                letter_counts[old_letter_lo] -= len(positions)
                letter_counts[new_letter_lo] += len(positions)

                # Rescore affected windows
                new_window_vals = np.empty(len(aff), dtype=np.float64)
                for j, wi in enumerate(aff):
                    new_window_vals[j] = _lookup_window_lo(
                        chars_lo, wi * step, model
                    )

                new_contrib = float(new_window_vals.sum())
                proposal_sum = sum_log_prob - old_contrib + new_contrib

                proposal_entropy = _compute_entropy(letter_counts, entropy_table)
                proposal_score = _zenith_score(proposal_sum, ngram_count, proposal_entropy)

                # Boltzmann acceptance — NO division by ngram_count (Zenith does not normalize)
                delta = proposal_score - current_score
                if delta >= 0.0 or rng.random() < math.exp(delta / temp):
                    # Accept
                    key[cipher_sym] = letter_to_id.get(
                        chr(new_letter_lo + ord("A")),
                        plaintext_ids[0],
                    )
                    window_scores[aff] = new_window_vals
                    sum_log_prob = proposal_sum
                    entropy = proposal_entropy
                    current_score = proposal_score
                    accepted_moves += 1
                    epoch_accepted += 1
                    if delta > 0:
                        improved_moves += 1
                        epoch_improved += 1
                    if current_score > best_total:
                        best_total = current_score
                        best_key = dict(key)
                        best_chars_lo = list(chars_lo)
                    if current_score > epoch_best_score:
                        epoch_best_score = current_score
                        epoch_best_key = dict(key)
                        epoch_best_chars_lo = list(chars_lo)
                else:
                    # Reject — revert
                    for pos in positions:
                        chars_lo[pos] = old_letter_lo
                    letter_counts[new_letter_lo] -= len(positions)
                    letter_counts[old_letter_lo] += len(positions)

        epoch_plaintext = _chars_lo_to_str(epoch_best_chars_lo, id_to_letter)
        candidate_pool.append(HomophonicCandidate(
            plaintext=epoch_plaintext,
            key=epoch_best_key,
            score=epoch_best_score,
            normalized_score=epoch_best_score,  # already normalized (mean / entropy)
            epoch=epoch + 1,
        ))

        if epoch_callback is not None:
            payload = {
                "epoch": epoch + 1,
                "plaintext": epoch_plaintext,
                "key": dict(epoch_best_key),
                "normalized_score": epoch_best_score,
                "accepted_moves": epoch_accepted,
                "improved_moves": epoch_improved,
            }
            if epoch_callback(payload):
                stopped_early = True
                stop_reason = "epoch_callback"
                break

    best_plaintext = _chars_lo_to_str(best_chars_lo, id_to_letter) if best_chars_lo else ""
    candidate_pool.append(HomophonicCandidate(
        plaintext=best_plaintext,
        key=best_key,
        score=best_total,
        normalized_score=best_total,
        epoch=0,
    ))
    candidates = _top_candidates(candidate_pool, max(1, top_n))

    return HomophonicAnnealResult(
        plaintext=best_plaintext,
        key=best_key,
        score=best_total,
        normalized_score=best_total,
        epochs=max(1, epochs),
        sampler_iterations=max(1, sampler_iterations),
        accepted_moves=accepted_moves,
        improved_moves=improved_moves,
        elapsed_seconds=time.time() - t0,
        fixed_symbols=len(fixed_cipher_ids),
        metadata={
            "solver": "zenith_native",
            "order": order,
            "step": step,
            "ngram_count": ngram_count,
            "t_start": t_start,
            "t_end": t_end,
            "mutable_symbols": len(mutable_symbol_ids),
            "cipher_symbols": len(symbol_ids),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        },
        candidates=candidates,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chars_lo_to_str(chars_lo: list[int], id_to_letter: dict[int, str]) -> str:
    """Convert 0-25 offset list back to uppercase plaintext string."""
    return "".join(chr(lo + ord("A")) for lo in chars_lo)


def _top_candidates(
    candidates: list[HomophonicCandidate],
    top_n: int,
) -> list[HomophonicCandidate]:
    by_plaintext: dict[str, HomophonicCandidate] = {}
    for c in candidates:
        ex = by_plaintext.get(c.plaintext)
        if ex is None or c.score > ex.score:
            by_plaintext[c.plaintext] = c
    return sorted(
        by_plaintext.values(),
        key=lambda c: c.score,
        reverse=True,
    )[:top_n]
