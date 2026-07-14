from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from .normalize import normalize_text


ORDER = 5
ALPHABET_SIZE = 26
ARRAY_LEN = ALPHABET_SIZE ** ORDER
MAGIC = 0x5A4D4D43
VERSION = 1
MAGIC_V2 = 0x5A4D4332
VERSION_V2 = 2
DEFAULT_MAX_NGRAMS = 3_000_000

# Guard against accidentally allocating an enormous dense n-gram array for a
# large (alphabet, order) combination.  26^5 ≈ 11.9M is the reference size.
_MAX_V2_ARRAY_LEN = 60_000_000


@dataclass
class BuildStats:
    language: str
    raw_files: int
    normalized_characters: int
    distinct_seen_ngrams: int
    unknown_log_prob: float
    output_path: Path
    metadata_path: Path
    sha256: str


def _iter_text_files(corpus_dir: Path) -> Iterator[Path]:
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".txt":
            yield path


def _count_ngrams_and_letters(corpus_dir: Path, language: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    counts = np.zeros(ARRAY_LEN, dtype=np.int64)
    letter_counts = np.zeros(ALPHABET_SIZE, dtype=np.int64)
    raw_files = 0
    normalized_characters = 0

    for path in _iter_text_files(corpus_dir):
        raw_files += 1
        raw = path.read_text(encoding="utf-8", errors="ignore")
        norm = normalize_text(raw, language)
        if not norm:
            continue
        normalized_characters += len(norm)
        vals = [ord(ch) - 97 for ch in norm]
        for value in vals:
            if 0 <= value < 26:
                letter_counts[value] += 1
        if len(vals) < ORDER:
            continue
        for i in range(len(vals) - ORDER + 1):
            a, b, c, d, e = vals[i:i + ORDER]
            idx = a * 456976 + b * 17576 + c * 676 + d * 26 + e
            counts[idx] += 1

    return counts, letter_counts, raw_files, normalized_characters


def _compute_model_arrays(
    counts: np.ndarray,
    letter_counts: np.ndarray,
    floor_probability: float = 1e-9,
) -> tuple[np.ndarray, float, dict[str, float]]:
    total = int(counts.sum())
    unknown_log_prob = math.log(floor_probability)
    log_probs = np.full(ARRAY_LEN, unknown_log_prob, dtype=np.float32)
    if total > 0:
        seen = counts > 0
        log_probs[seen] = np.log(counts[seen] / total).astype(np.float32)

    total_letters = int(letter_counts.sum())
    if total_letters <= 0:
        letter_freq = {chr(65 + i): 1.0 / 26 for i in range(26)}
    else:
        letter_freq = {
            chr(65 + i): float(letter_counts[i] / total_letters)
            for i in range(26)
        }
    return log_probs, unknown_log_prob, letter_freq


def _count_ngrams_generic(
    corpus_dir: Path,
    alphabet: str,
    order: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Count order-``order`` n-grams over an arbitrary symbol ``alphabet``.

    Normalization for v2 builds is intentionally simple and alphabet-driven:
    lowercase the text, then keep only characters that appear in ``alphabet``.
    Each retained character maps to its 0-based index in ``alphabet``.
    """
    base = len(alphabet)
    array_len = base ** order
    counts = np.zeros(array_len, dtype=np.int64)
    symbol_counts = np.zeros(base, dtype=np.int64)
    index_of = {ch: i for i, ch in enumerate(alphabet)}
    powers = [base ** (order - 1 - i) for i in range(order)]
    raw_files = 0
    normalized_characters = 0

    for path in _iter_text_files(corpus_dir):
        raw_files += 1
        raw = path.read_text(encoding="utf-8", errors="ignore")
        vals = [index_of[ch] for ch in raw.lower() if ch in index_of]
        if not vals:
            continue
        normalized_characters += len(vals)
        for value in vals:
            symbol_counts[value] += 1
        if len(vals) < order:
            continue
        for i in range(len(vals) - order + 1):
            idx = 0
            for j in range(order):
                idx += vals[i + j] * powers[j]
            counts[idx] += 1

    return counts, symbol_counts, raw_files, normalized_characters


def _compute_model_arrays_generic(
    counts: np.ndarray,
    symbol_counts: np.ndarray,
    alphabet: str,
    floor_probability: float = 1e-9,
) -> tuple[np.ndarray, float, dict[str, float]]:
    array_len = counts.shape[0]
    total = int(counts.sum())
    unknown_log_prob = math.log(floor_probability)
    log_probs = np.full(array_len, unknown_log_prob, dtype=np.float32)
    if total > 0:
        seen = counts > 0
        log_probs[seen] = np.log(counts[seen] / total).astype(np.float32)

    total_symbols = int(symbol_counts.sum())
    if total_symbols <= 0:
        symbol_freq = {ch: 1.0 / len(alphabet) for ch in alphabet}
    else:
        symbol_freq = {
            ch: float(symbol_counts[i] / total_symbols)
            for i, ch in enumerate(alphabet)
        }
    return log_probs, unknown_log_prob, symbol_freq


def write_zenith_binary_model_v2(
    output_path: Path,
    *,
    alphabet: str,
    order: int,
    log_probs: np.ndarray,
    unknown_log_prob: float,
) -> None:
    """Write a ``zenith_binary_v2`` model file (big-endian).

    Layout mirrors :func:`analysis.zenith_solver._read_v2`:

    .. code-block:: text

        4B  magic          = 0x5A4D4332
        4B  version        = 2
        4B  order          = K
        4B  alphabetLen    = N
        4B  alphabetBytes  = L (UTF-8 byte length)
        LB  alphabet       (UTF-8)
        4B  unknownProbability  (float32)
        4B  arrayLength    = N^K
        <arrayLength × 4B>  nGramLogProbabilities (float32)
    """
    alphabet_bytes = alphabet.encode("utf-8")
    array_len = len(alphabet) ** order
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        fh.write(struct.pack(">I", MAGIC_V2))
        fh.write(struct.pack(">IIII", VERSION_V2, order, len(alphabet), len(alphabet_bytes)))
        fh.write(alphabet_bytes)
        fh.write(struct.pack(">f", math.exp(unknown_log_prob)))
        fh.write(struct.pack(">I", array_len))
        fh.write(log_probs.astype(">f4").tobytes())


def write_zenith_binary_model(
    output_path: Path,
    *,
    log_probs: np.ndarray,
    unknown_log_prob: float,
    letter_freq: dict[str, float],
    max_ngrams: int = DEFAULT_MAX_NGRAMS,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        fh.write(struct.pack(">IIII", MAGIC, VERSION, ORDER, max_ngrams))
        fh.write(struct.pack(">f", math.exp(unknown_log_prob)))
        fh.write(struct.pack(">II", ARRAY_LEN, 26))
        for letter in map(chr, range(ord("A"), ord("Z") + 1)):
            freq = float(letter_freq.get(letter, 0.0))
            count = int(freq * 10_000_000)
            log_prob = math.log(freq) if freq > 0 else unknown_log_prob
            fh.write(struct.pack(">H", ord(letter.lower())))
            fh.write(struct.pack(">q", count))
            fh.write(struct.pack(">d", log_prob))
        fh.write(struct.pack(">I", ARRAY_LEN))
        fh.write(log_probs.astype(">f4").tobytes())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_metadata(
    metadata_path: Path,
    *,
    language: str,
    output_path: Path,
    sha256: str,
    unknown_log_prob: float,
    raw_files: int,
    normalized_characters: int,
    distinct_seen_ngrams: int,
    sources: list[dict[str, str]] | None = None,
    fmt: str = "zenith_binary_v1",
    order: int = ORDER,
    array_length: int = ARRAY_LEN,
    alphabet: str | None = None,
    normalization: dict | None = None,
    variant: str | None = None,
    display_label: str | None = None,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": language,
        "order": order,
        "format": fmt,
        "output_file": output_path.name,
    }
    if variant is not None:
        payload["variant"] = variant
    if display_label is not None:
        payload["display_label"] = display_label
    payload |= {
        "sha256": sha256,
        "array_length": array_length,
        "unknown_log_prob": round(unknown_log_prob, 6),
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "builder_version": VERSION_V2 if alphabet is not None else 1,
        "sources": sources or [],
        "corpus_stats": {
            "raw_files": raw_files,
            "normalized_characters": normalized_characters,
            "distinct_seen_5grams": distinct_seen_ngrams,
        },
        "normalization": normalization or {
            "lowercase": True,
            "strip_non_alpha": True,
        },
        "redistribution_status": "redistributable",
    }
    if alphabet is not None:
        payload["alphabet"] = alphabet
        payload["alphabet_size"] = len(alphabet)
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_model(
    *,
    language: str,
    corpus_dir: Path,
    output_path: Path,
    sources: list[dict[str, str]] | None = None,
    floor_probability: float = 1e-9,
    alphabet: str | None = None,
    order: int | None = None,
    variant: str | None = None,
    display_label: str | None = None,
) -> BuildStats:
    """Build a Zenith binary n-gram model.

    With ``alphabet is None`` (the default) this emits the classic
    ``zenith_binary_v1`` 26^5 lowercase-letter model, byte-for-byte as before.

    When ``alphabet`` is provided a ``zenith_binary_v2`` model is emitted over
    that symbol alphabet (and optional ``order``, default 5).  v2 normalization
    is alphabet-driven (lowercase, then keep only symbols in ``alphabet``).
    """
    if alphabet is not None:
        return _build_model_v2(
            language=language,
            corpus_dir=corpus_dir,
            output_path=output_path,
            alphabet=alphabet,
            order=ORDER if order is None else order,
            sources=sources,
            floor_probability=floor_probability,
            variant=variant,
            display_label=display_label,
        )

    counts, letter_counts, raw_files, normalized_characters = _count_ngrams_and_letters(
        corpus_dir,
        language,
    )
    log_probs, unknown_log_prob, letter_freq = _compute_model_arrays(
        counts,
        letter_counts,
        floor_probability=floor_probability,
    )
    write_zenith_binary_model(
        output_path,
        log_probs=log_probs,
        unknown_log_prob=unknown_log_prob,
        letter_freq=letter_freq,
    )
    sha256 = _sha256(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    write_metadata(
        metadata_path,
        language=language,
        output_path=output_path,
        sha256=sha256,
        unknown_log_prob=unknown_log_prob,
        raw_files=raw_files,
        normalized_characters=normalized_characters,
        distinct_seen_ngrams=int((counts > 0).sum()),
        sources=sources,
        variant=variant,
        display_label=display_label,
    )
    return BuildStats(
        language=language,
        raw_files=raw_files,
        normalized_characters=normalized_characters,
        distinct_seen_ngrams=int((counts > 0).sum()),
        unknown_log_prob=unknown_log_prob,
        output_path=output_path,
        metadata_path=metadata_path,
        sha256=sha256,
    )


def _build_model_v2(
    *,
    language: str,
    corpus_dir: Path,
    output_path: Path,
    alphabet: str,
    order: int,
    sources: list[dict[str, str]] | None = None,
    floor_probability: float = 1e-9,
    variant: str | None = None,
    display_label: str | None = None,
) -> BuildStats:
    if len(alphabet) < 1:
        raise ValueError("alphabet must contain at least one symbol")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet must not contain duplicate symbols")
    if alphabet != alphabet.lower():
        raise ValueError(
            "alphabet must be lowercase — v2 normalization lowercases text before "
            f"filtering, so non-lowercase symbols would never match; got {alphabet!r}"
        )
    if order < 1:
        raise ValueError(f"order must be >= 1; got {order}")
    array_len = len(alphabet) ** order
    if array_len > _MAX_V2_ARRAY_LEN:
        raise ValueError(
            f"alphabet^order = {len(alphabet)}^{order} = {array_len} exceeds the "
            f"dense-array guard {_MAX_V2_ARRAY_LEN}"
        )

    counts, symbol_counts, raw_files, normalized_characters = _count_ngrams_generic(
        corpus_dir,
        alphabet,
        order,
    )
    log_probs, unknown_log_prob, _symbol_freq = _compute_model_arrays_generic(
        counts,
        symbol_counts,
        alphabet,
        floor_probability=floor_probability,
    )
    write_zenith_binary_model_v2(
        output_path,
        alphabet=alphabet,
        order=order,
        log_probs=log_probs,
        unknown_log_prob=unknown_log_prob,
    )
    sha256 = _sha256(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    write_metadata(
        metadata_path,
        language=language,
        output_path=output_path,
        sha256=sha256,
        unknown_log_prob=unknown_log_prob,
        raw_files=raw_files,
        normalized_characters=normalized_characters,
        distinct_seen_ngrams=int((counts > 0).sum()),
        sources=sources,
        fmt="zenith_binary_v2",
        order=order,
        array_length=array_len,
        alphabet=alphabet,
        normalization={"lowercase": True, "alphabet_filter": True},
        variant=variant,
        display_label=display_label,
    )
    return BuildStats(
        language=language,
        raw_files=raw_files,
        normalized_characters=normalized_characters,
        distinct_seen_ngrams=int((counts > 0).sum()),
        unknown_log_prob=unknown_log_prob,
        output_path=output_path,
        metadata_path=metadata_path,
        sha256=sha256,
    )
