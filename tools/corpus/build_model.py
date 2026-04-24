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
DEFAULT_MAX_NGRAMS = 3_000_000


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
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": language,
        "order": ORDER,
        "format": "zenith_binary_v1",
        "output_file": output_path.name,
        "sha256": sha256,
        "array_length": ARRAY_LEN,
        "unknown_log_prob": round(unknown_log_prob, 6),
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "builder_version": 1,
        "sources": sources or [],
        "corpus_stats": {
            "raw_files": raw_files,
            "normalized_characters": normalized_characters,
            "distinct_seen_5grams": distinct_seen_ngrams,
        },
        "normalization": {
            "lowercase": True,
            "strip_non_alpha": True,
        },
        "redistribution_status": "redistributable",
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_model(
    *,
    language: str,
    corpus_dir: Path,
    output_path: Path,
    sources: list[dict[str, str]] | None = None,
    floor_probability: float = 1e-9,
) -> BuildStats:
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
