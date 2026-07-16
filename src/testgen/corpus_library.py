"""Measured plaintext corpus library (Slice B).

This module is the local, offline, LLM-free plaintext source for the on-demand
cipher benchmark generator. It provides:

* the record schema (:class:`PassageRecord`) for one measured plaintext passage;
* the cleaning / sampling / measurement / tagging *core* used by the shipped
  builder (``scripts/build_plaintext_library.py``); and
* the query API — :func:`load_library` and :func:`select` — used by Slice C.

The shipped library lives at ``resources/plaintext_library/<lang>.jsonl`` (one
passage per line). Ground truth is the passage text itself (this *is* the
plaintext source), so there is no firewall concern at this layer — the
context/plaintext firewall lives at case assembly (Slice C).

Cleaning reuses the normalization convention of
``testgen.plaintext.PlaintextGenerator._normalize`` (NFD diacritic strip →
uppercase → A-Z-only tokens of length >= 2) so corpus plaintext matches the
LLM-generated plaintext path byte-for-byte in shape.

No network. No LLM. Determinism is driven by an injected ``random.Random``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY_DIR = _REPO_ROOT / "resources" / "plaintext_library"

# Canonical era tags (spec: derive from provenance).
ERA_HISTORICAL = "historical_1600_1899"
ERA_CLASSICAL = "classical"
ERA_LITERARY_19C = "literary_19c"
ERA_MODERN = "modern"

# Source corpora we recognise. ``bnc`` is intentionally excluded from the
# *shipped* library — its licence forbids redistributing the corpus text
# (corpus_data/en/corpus_manifest.json: redistributable=false). The builder
# skips it by default.
KNOWN_SOURCES = ("gutenberg", "oanc", "masc", "dta", "bnc")
NON_REDISTRIBUTABLE_SOURCES = frozenset({"bnc"})

ALPHABET_SIZE = 26

# Graded passage lengths (words), per spec (~40, 120, 250, 500).
DEFAULT_LENGTHS = (40, 120, 250, 500)

# Frequency-style percentile threshold: a passage whose unigram_chi2 is at or
# above this percentile *within its language* is tagged ``unusual`` (its letter
# distribution is unusually far from the language reference), else ``normal``.
UNUSUAL_CHI2_PERCENTILE = 80.0


# --------------------------------------------------------------------------
# Boilerplate / licence guard
# --------------------------------------------------------------------------

# Substrings that mark corpus boilerplate (Gutenberg headers/licences, OANC/DTA
# provenance lines, transcriber notes, URLs, ...). Matching is done on the
# UPPERCASED raw line. A line containing any of these is dropped before word
# extraction, and a fully-assembled passage is re-checked (belt and suspenders)
# so licence text can never leak into a shipped passage.
BOILERPLATE_MARKERS: tuple[str, ...] = (
    "PROJECT GUTENBERG",
    "GUTENBERG",
    "GUTENBERG-TM",
    "GUTENBERG EBOOK",
    "ETEXT",
    "EBOOK",
    "COPYRIGHT",
    "ALL RIGHTS RESERVED",
    "RIGHTS RESERVED",
    "PUBLIC DOMAIN",
    "CREATIVE COMMONS",
    "LICENSE",
    "LICENCE",
    "LICENSED",
    "TRADEMARK",
    "TRANSCRIBER",
    "PROOFREAD",
    "PRODUCED BY",
    "DISTRIBUTED PROOFREAD",
    "START OF TH",   # START OF THE / START OF THIS ...
    "END OF TH",     # END OF THE / END OF THIS ...
    "HTTP",
    "WWW.",
    ".ORG",
    ".COM",
    "ISBN",
    "DEUTSCHES TEXTARCHIV",
    "TEXTARCHIV",
    "@",
)

# The post-assembly guard checks the *normalized* passage text (A-Z + spaces,
# no punctuation), so only alphabetic markers are meaningful here.
_PASSAGE_GUARD_MARKERS: tuple[str, ...] = (
    "PROJECT GUTENBERG",
    "GUTENBERG",
    "ETEXT",
    "EBOOK",
    "COPYRIGHT",
    "CREATIVE COMMONS",
    "TRANSCRIBER",
    "PROOFREAD",
    "TEXTARCHIV",
    "DEUTSCHES TEXTARCHIV",
)


def line_is_boilerplate(line: str) -> bool:
    """True if a raw source line looks like corpus boilerplate/markup."""
    upper = line.upper()
    if not upper.strip():
        return False
    # Bracketed markup like ``[Illustration]`` / ``[Pg 12]``.
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    return any(marker in upper for marker in BOILERPLATE_MARKERS)


def contains_boilerplate(passage_text: str) -> bool:
    """True if an assembled (normalized, uppercase) passage still carries a
    licence/provenance marker. Used as a hard guard so nothing leaks."""
    upper = passage_text.upper()
    return any(marker in upper for marker in _PASSAGE_GUARD_MARKERS)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


def _normalize_words(text: str) -> list[str]:
    """Normalize a chunk of raw prose into a list of uppercase A-Z words.

    Reuses ``PlaintextGenerator._normalize`` (the existing testgen convention:
    NFD diacritic strip, uppercase, keep only A-Z within each whitespace token,
    drop tokens shorter than 2 chars) so the corpus plaintext path produces the
    same word shape as the LLM plaintext path.
    """
    # Lazy import keeps module import light and avoids any agent coupling
    # (PlaintextGenerator only touches agent code inside __init__, never in the
    # staticmethod we call here).
    from testgen.plaintext import PlaintextGenerator

    normalized = PlaintextGenerator._normalize(text)
    return normalized.split()


def _strip_gutenberg_envelope(lines: list[str]) -> list[str]:
    """Return only the body between Gutenberg START/END markers when present.

    Real Gutenberg files wrap the work in ``*** START OF THE PROJECT GUTENBERG
    EBOOK ... ***`` / ``*** END OF ...`` marker lines; everything outside is
    licence boilerplate. Slicing to the body removes the *entire* header and
    footer licence blocks (not just the individually-marked lines). Files
    without markers are returned unchanged (the line filter + sampler trim then
    handle any residual front/back matter).
    """
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        upper = line.upper()
        if start_idx is None and "START OF TH" in upper and "PROJECT GUTENBERG" in upper:
            start_idx = i
            continue
        if start_idx is not None and "END OF TH" in upper and "PROJECT GUTENBERG" in upper:
            end_idx = i
            break
    if start_idx is None and end_idx is None:
        # No START seen; still cut a stray END (defensive).
        for i, line in enumerate(lines):
            upper = line.upper()
            if "END OF TH" in upper and "PROJECT GUTENBERG" in upper:
                return lines[:i]
        return lines
    lo = (start_idx + 1) if start_idx is not None else 0
    hi = end_idx if end_idx is not None else len(lines)
    return lines[lo:hi]


def clean_to_words(raw_text: str) -> list[str]:
    """Turn a raw source document into a flat list of clean uppercase words.

    Pipeline: slice to the Gutenberg body (if envelope markers exist) → drop
    boilerplate/markup lines → normalize surviving lines to uppercase A-Z words.
    Front/back matter is *not* trimmed here — that is the sampler's job
    (:func:`sample_passages`).
    """
    body = _strip_gutenberg_envelope(raw_text.splitlines())
    kept: list[str] = []
    for line in body:
        if line_is_boilerplate(line):
            continue
        kept.append(line)
    return _normalize_words("\n".join(kept))


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SampledPassage:
    """A contiguous run of clean words drawn from a document's word pool."""

    start: int
    length_words: int
    text: str


def sample_passages(
    words: list[str],
    *,
    lengths: Iterable[int] = DEFAULT_LENGTHS,
    per_length: int = 1,
    rng,
    front_trim_frac: float = 0.08,
    back_trim_frac: float = 0.03,
    front_trim_min: int = 12,
) -> list[SampledPassage]:
    """Sample contiguous passages of the requested word lengths from ``words``.

    Front matter (title pages, tables of contents) and footer are trimmed by a
    small fraction so passages come from the body. Start offsets are chosen from
    the injected ``rng`` — deterministic given the same rng state. Passages
    that trip the post-assembly boilerplate guard are skipped.
    """
    n = len(words)
    if n == 0:
        return []
    front = max(front_trim_min, int(n * front_trim_frac))
    back = int(n * back_trim_frac)
    lo = min(front, n)
    hi = max(lo, n - back)
    body = words[lo:hi]
    m = len(body)
    if m == 0:
        return []

    out: list[SampledPassage] = []
    seen_starts: set[tuple[int, int]] = set()
    for length in lengths:
        if length <= 0 or length > m:
            continue
        max_start = m - length
        attempts = 0
        produced = 0
        while produced < per_length and attempts < per_length * 8:
            attempts += 1
            start = 0 if max_start == 0 else rng.randint(0, max_start)
            key = (start, length)
            if key in seen_starts:
                continue
            chunk = body[start : start + length]
            text = " ".join(chunk)
            if contains_boilerplate(text):
                continue
            seen_starts.add(key)
            out.append(SampledPassage(start=lo + start, length_words=length, text=text))
            produced += 1
    return out


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

# Measured fields, in a stable order (documented for the deliverable).
MEASURED_FIELDS: tuple[str, ...] = (
    "index_of_coincidence",
    "unigram_chi2",
    "dict_rate",
    "mean_word_length",
    "function_word_rate",
    "unique_symbol_ratio",
    "bigram_loglik",
    "quadgram_loglik",
)

_FUNCTION_WORDS_DIR = _REPO_ROOT / "resources" / "function_words"

# Small caches so the builder does not reload resources per passage.
_word_set_cache: dict[str, set[str]] = {}
_function_word_cache: dict[str, frozenset[str]] = {}


def _word_set(language: str) -> set[str]:
    if language not in _word_set_cache:
        from analysis.dictionary import get_dictionary_path, load_word_set

        try:
            path = get_dictionary_path(language)
        except ValueError:
            path = None
        _word_set_cache[language] = load_word_set(path) if path else set()
    return _word_set_cache[language]


def _function_words(language: str) -> frozenset[str]:
    if language not in _function_word_cache:
        path = _FUNCTION_WORDS_DIR / f"{language}.txt"
        words: set[str] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                words.add(line.upper())
        _function_word_cache[language] = frozenset(words)
    return _function_word_cache[language]


def measure_passage(text: str, language: str) -> dict[str, float]:
    """Measure a cleaned, uppercase, space-separated passage.

    Reuses ``analysis.frequency`` / ``analysis.ic`` / ``analysis.dictionary`` /
    ``analysis.ngram`` so measured values match the rest of the toolchain. All
    values are plain floats (JSON-serialisable).
    """
    from analysis.dictionary import score_plaintext
    from analysis.frequency import unigram_chi2
    from analysis.ic import index_of_coincidence
    from analysis.ngram import NGRAM_CACHE, normalized_ngram_score

    words = text.split()
    letters = [c for c in text if c.isalpha()]
    total_letters = len(letters)

    tokens = [ord(c) - 65 for c in letters]
    ioc = index_of_coincidence(tokens, ALPHABET_SIZE) if tokens else 0.0
    chi2 = unigram_chi2(text, language)
    dict_rate = score_plaintext(text, _word_set(language)) if words else 0.0
    mean_wl = (sum(len(w) for w in words) / len(words)) if words else 0.0
    fw = _function_words(language)
    fw_rate = (sum(1 for w in words if w in fw) / len(words)) if (words and fw) else 0.0
    distinct = len(set(letters))
    unique_symbol_ratio = (distinct / ALPHABET_SIZE) if total_letters else 0.0

    try:
        bigram_ll = normalized_ngram_score(text, NGRAM_CACHE.get(language, 2), 2)
    except Exception:
        bigram_ll = 0.0
    try:
        quad_ll = normalized_ngram_score(text, NGRAM_CACHE.get(language, 4), 4)
    except Exception:
        quad_ll = 0.0
    # normalized_ngram_score can return -inf on tiny inputs; clamp to a float.
    if bigram_ll == float("-inf"):
        bigram_ll = 0.0
    if quad_ll == float("-inf"):
        quad_ll = 0.0

    return {
        "index_of_coincidence": round(float(ioc), 6),
        "unigram_chi2": round(float(chi2), 4),
        "dict_rate": round(float(dict_rate), 4),
        "mean_word_length": round(float(mean_wl), 4),
        "function_word_rate": round(float(fw_rate), 4),
        "unique_symbol_ratio": round(float(unique_symbol_ratio), 4),
        "bigram_loglik": round(float(bigram_ll), 4),
        "quadgram_loglik": round(float(quad_ll), 4),
    }


# --------------------------------------------------------------------------
# Tagging
# --------------------------------------------------------------------------


def era_for(source: str, language: str) -> str:
    """Derive the era tag from provenance (spec ordering).

    dta -> historical; Latin -> classical (its corpus is classical literature,
    regardless of the Gutenberg wrapper); other Gutenberg -> literary 19c;
    modern corpora (bnc/oanc/masc) -> modern.
    """
    if source == "dta":
        return ERA_HISTORICAL
    if language == "la":
        return ERA_CLASSICAL
    if source == "gutenberg":
        return ERA_LITERARY_19C
    if source in ("bnc", "oanc", "masc"):
        return ERA_MODERN
    return ERA_MODERN


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Record schema
# --------------------------------------------------------------------------


@dataclass
class PassageRecord:
    """One measured plaintext passage (a line of ``<lang>.jsonl``)."""

    id: str
    language: str
    era: str
    provenance: str
    source_file: str
    text: str
    length_words: int
    length_chars: int
    measured: dict[str, float] = field(default_factory=dict)
    frequency_style: str = "normal"
    topic: str = "general"
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language": self.language,
            "era": self.era,
            "provenance": self.provenance,
            "source_file": self.source_file,
            "text": self.text,
            "length_words": self.length_words,
            "length_chars": self.length_chars,
            "measured": self.measured,
            "frequency_style": self.frequency_style,
            "topic": self.topic,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PassageRecord":
        return cls(
            id=d["id"],
            language=d["language"],
            era=d["era"],
            provenance=d["provenance"],
            source_file=d.get("source_file", ""),
            text=d["text"],
            length_words=int(d["length_words"]),
            length_chars=int(d["length_chars"]),
            measured=dict(d.get("measured", {})),
            frequency_style=d.get("frequency_style", "normal"),
            topic=d.get("topic", "general"),
            content_hash=d.get("content_hash", ""),
        )


def build_record(
    *,
    passage: SampledPassage,
    language: str,
    source: str,
    provenance: str,
    source_file: str,
    topic: str = "general",
    frequency_style: str = "normal",
) -> PassageRecord:
    """Assemble a fully-measured :class:`PassageRecord` for a sampled passage.

    ``frequency_style`` is passed in because it is a *within-language*
    percentile decision the builder makes after measuring every passage.
    """
    text = passage.text
    ch = content_hash(text)
    measured = measure_passage(text, language)
    rec_id = f"{language}-{source}-{Path(source_file).stem}-{passage.start}-{passage.length_words}-{ch[:8]}"
    return PassageRecord(
        id=rec_id,
        language=language,
        era=era_for(source, language),
        provenance=provenance,
        source_file=source_file,
        text=text,
        length_words=passage.length_words,
        length_chars=len(text),
        measured=measured,
        frequency_style=frequency_style,
        topic=topic,
        content_hash=ch,
    )


# --------------------------------------------------------------------------
# Query API
# --------------------------------------------------------------------------


class LibraryEmpty(Exception):
    """Raised when no passage matches the requested filters.

    The message names the unmet filter(s) so callers can relax them.
    """


def library_path(language: str, *, library_dir: str | os.PathLike | None = None) -> Path:
    base = Path(library_dir) if library_dir is not None else DEFAULT_LIBRARY_DIR
    return base / f"{language}.jsonl"


def load_library(
    language: str, *, library_dir: str | os.PathLike | None = None
) -> list[PassageRecord]:
    """Load all passage records for ``language`` from ``<lang>.jsonl``.

    Returns an empty list if the file is absent. Records preserve file order,
    which makes :func:`select` deterministic given a seeded rng.
    """
    path = library_path(language, library_dir=library_dir)
    if not path.exists():
        return []
    records: list[PassageRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(PassageRecord.from_dict(json.loads(line)))
    return records


def _matches(
    rec: PassageRecord,
    *,
    era: str | None,
    min_words: int | None,
    max_words: int | None,
    frequency_style: str | None,
    topic: str | None,
) -> bool:
    if era is not None and rec.era != era:
        return False
    if min_words is not None and rec.length_words < min_words:
        return False
    if max_words is not None and rec.length_words > max_words:
        return False
    if frequency_style is not None and rec.frequency_style != frequency_style:
        return False
    if topic is not None and rec.topic != topic:
        return False
    return True


def _describe_filters(
    *,
    language: str | None,
    era: str | None,
    min_words: int | None,
    max_words: int | None,
    frequency_style: str | None,
    topic: str | None,
) -> str:
    parts: list[str] = []
    if language is not None:
        parts.append(f"language={language!r}")
    if era is not None:
        parts.append(f"era={era!r}")
    if min_words is not None:
        parts.append(f"min_words={min_words}")
    if max_words is not None:
        parts.append(f"max_words={max_words}")
    if frequency_style is not None:
        parts.append(f"frequency_style={frequency_style!r}")
    if topic is not None:
        parts.append(f"topic={topic!r}")
    return ", ".join(parts) if parts else "(none)"


def filter_records(
    records: Iterable[PassageRecord],
    *,
    era: str | None = None,
    min_words: int | None = None,
    max_words: int | None = None,
    frequency_style: str | None = None,
    topic: str | None = None,
) -> list[PassageRecord]:
    """Return every record matching the given filters (order preserved)."""
    return [
        rec
        for rec in records
        if _matches(
            rec,
            era=era,
            min_words=min_words,
            max_words=max_words,
            frequency_style=frequency_style,
            topic=topic,
        )
    ]


def select(
    *,
    language: str | None = None,
    era: str | None = None,
    min_words: int | None = None,
    max_words: int | None = None,
    frequency_style: str | None = None,
    topic: str | None = None,
    rng,
    records: Iterable[PassageRecord] | None = None,
    library_dir: str | os.PathLike | None = None,
) -> PassageRecord:
    """Select one passage matching the filters, deterministically given ``rng``.

    Either pass an explicit ``records`` iterable (used by tests / Slice C when
    it already holds a library) or a ``language`` (the library is loaded from
    ``<library_dir>/<language>.jsonl``). Selection is ``rng``-driven over the
    filtered set in file order, so a fixed rng seed always yields the same pick.

    Raises :class:`LibraryEmpty` — naming the unmet filter(s) — when nothing
    matches.
    """
    if records is None:
        if language is None:
            raise ValueError("select() requires either `records` or `language`")
        pool: list[PassageRecord] = load_library(language, library_dir=library_dir)
        if not pool:
            raise LibraryEmpty(
                f"no library loaded for language={language!r} "
                f"(looked for {library_path(language, library_dir=library_dir)})"
            )
    else:
        pool = list(records)
        if language is not None:
            pool = [r for r in pool if r.language == language]

    candidates = filter_records(
        pool,
        era=era,
        min_words=min_words,
        max_words=max_words,
        frequency_style=frequency_style,
        topic=topic,
    )
    if not candidates:
        raise LibraryEmpty(
            "no passage matches filters: "
            + _describe_filters(
                language=language,
                era=era,
                min_words=min_words,
                max_words=max_words,
                frequency_style=frequency_style,
                topic=topic,
            )
        )
    return rng.choice(candidates)
