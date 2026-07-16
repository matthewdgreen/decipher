#!/usr/bin/env python3
"""Build the shipped, measured plaintext library (Slice B, offline, no LLM).

Samples graded-length passages of REAL prose from the local corpora, cleans
them to uppercase A-Z (stripping Gutenberg/OANC/DTA boilerplate + licence
text), MEASURES each passage (IoC, unigram chi2, dict-rate, mean word length,
function-word rate, unique-symbol ratio, bigram/quadgram log-likelihood), TAGS
each (language, era, provenance, frequency_style, topic), and writes one
``resources/plaintext_library/<lang>.jsonl`` per language plus a sampling
manifest (``manifest.json``).

Sources (redistributable only — BNC is skipped, its licence forbids
redistributing the corpus text):
  * ``corpus_data/en/pg*.txt``            gutenberg (English, literary_19c)
  * ``corpus_data/en/oanc/**``            oanc (English, modern)
  * ``corpus_data/en/masc/**``            masc (English, modern)
  * ``corpus_data/{de,fr,it,la}/gutenberg/pg*.txt``  gutenberg
  * ``corpora/dta/text/<period>/*.txt``   dta (German, historical_1600_1899)

Determinism: every random choice derives from ``--seed`` via a per-file
``random.Random(f"{seed}:{source_file}")`` and per-source subset selection via
``random.Random(f"{seed}:{lang}:{source}")`` — so a fixed seed reproduces the
library, and adding a source file does not perturb other files' passages.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_plaintext_library.py \
        --languages en de fr it la \
        --files-per-source 30 --per-length 1 --seed 20260716

    # Small smoke build into a temp dir:
    PYTHONPATH=src .venv/bin/python scripts/build_plaintext_library.py \
        --languages en --files-per-source 3 --out /tmp/plib --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from testgen.corpus_library import (  # noqa: E402
    DEFAULT_LENGTHS,
    DEFAULT_LIBRARY_DIR,
    NON_REDISTRIBUTABLE_SOURCES,
    UNUSUAL_CHI2_PERCENTILE,
    PassageRecord,
    build_record,
    clean_to_words,
    sample_passages,
)

DEFAULT_CORPUS_DATA = REPO_ROOT / "corpus_data"
DEFAULT_DTA_ROOT = REPO_ROOT / "corpora" / "dta"

# Module-level roots (overridable via CLI --corpus-root / --dta-root). Corpora
# are git-ignored and may live outside the worktree, so they are configurable.
CORPUS_DATA = DEFAULT_CORPUS_DATA
DTA_ROOT = DEFAULT_DTA_ROOT

DEFAULT_LANGUAGES = ("en", "de", "fr", "it", "la")


# --------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFile:
    path: Path
    language: str
    source: str          # gutenberg / oanc / masc / dta
    topic: str
    provenance_extra: str  # period (dta) or category (oanc/masc), else ""


def _rel(path: Path) -> str:
    """Portable, corpus-relative path string (independent of where corpora live).

    Anchors at the ``corpus_data`` or ``corpora`` component so shipped records
    never carry an absolute machine path.
    """
    parts = path.parts
    for anchor in ("corpus_data", "corpora"):
        if anchor in parts:
            i = parts.index(anchor)
            return "/".join(parts[i:])
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def _gutenberg_files(language: str) -> list[SourceFile]:
    if language == "en":
        # Root corpus_data/en/pg*.txt is the 500-file superset; the
        # corpus_data/en/gutenberg/ subdir duplicates a subset of it, so we
        # use only the root to avoid shipping duplicate passages.
        paths = sorted((CORPUS_DATA / "en").glob("pg*.txt"))
    else:
        paths = sorted((CORPUS_DATA / language / "gutenberg").glob("pg*.txt"))
    return [
        SourceFile(p, language, "gutenberg", "general", "")
        for p in paths
    ]


# OANC skip-list: metadata files that are not prose.
_OANC_SKIP = {"license.txt", "readme.txt", "annotations.txt"}


def _oanc_files() -> list[SourceFile]:
    root = CORPUS_DATA / "en" / "oanc" / "OANC" / "data"
    out: list[SourceFile] = []
    for p in sorted(root.rglob("*.txt")):
        if p.name.lower() in _OANC_SKIP:
            continue
        # data/{written_1,written_2,spoken}/<category>/...  -> topic=<category>
        rel_parts = p.relative_to(root).parts
        topic = rel_parts[1] if len(rel_parts) >= 2 else "general"
        out.append(SourceFile(p, "en", "oanc", topic, topic))
    return out


def _masc_files() -> list[SourceFile]:
    root = CORPUS_DATA / "en" / "masc" / "masc_500k_texts"
    out: list[SourceFile] = []
    for p in sorted(root.rglob("*.txt")):
        # {written,spoken}/<category>/file -> topic=<category>
        rel_parts = p.relative_to(root).parts
        topic = rel_parts[1] if len(rel_parts) >= 2 else "general"
        # Normalise whitespace in category names (e.g. "court transcript").
        topic = topic.replace(" ", "_")
        out.append(SourceFile(p, "en", "masc", topic, topic))
    return out


def _dta_files() -> list[SourceFile]:
    root = DTA_ROOT / "text"
    out: list[SourceFile] = []
    if not root.exists():
        return out
    for period_dir in sorted(root.iterdir()):
        if not period_dir.is_dir():
            continue
        for p in sorted(period_dir.glob("*.txt")):
            out.append(SourceFile(p, "de", "dta", "general", period_dir.name))
    return out


def discover_sources(language: str, *, include: set[str]) -> dict[str, list[SourceFile]]:
    """Return {source_name: [SourceFile, ...]} for a language."""
    by_source: dict[str, list[SourceFile]] = {}

    def add(name: str, files: list[SourceFile]) -> None:
        if name in NON_REDISTRIBUTABLE_SOURCES:
            return
        if include and name not in include:
            return
        if files:
            by_source[name] = files

    add("gutenberg", _gutenberg_files(language))
    if language == "en":
        add("oanc", _oanc_files())
        add("masc", _masc_files())
    if language == "de":
        add("dta", _dta_files())
    return by_source


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return float("inf")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def build_language(
    language: str,
    *,
    include_sources: set[str],
    files_per_source: int,
    lengths: Iterable[int],
    per_length: int,
    max_per_lang: int | None,
    seed: int,
    verbose: bool,
) -> tuple[list[PassageRecord], dict]:
    by_source = discover_sources(language, include=include_sources)
    records: list[PassageRecord] = []
    seen_hashes: set[str] = set()
    per_source_counts: dict[str, int] = {}

    for source, files in sorted(by_source.items()):
        # Deterministic subset of files for this (seed, language, source).
        picker = random.Random(f"{seed}:{language}:{source}")
        chosen = files if files_per_source <= 0 else picker.sample(
            files, k=min(files_per_source, len(files))
        )
        chosen = sorted(chosen, key=lambda sf: str(sf.path))
        count_before = len(records)

        for sf in chosen:
            try:
                raw = sf.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            words = clean_to_words(raw)
            if len(words) < min(lengths):
                continue
            file_rng = random.Random(f"{seed}:{_rel(sf.path)}")
            passages = sample_passages(
                words,
                lengths=lengths,
                per_length=per_length,
                rng=file_rng,
            )
            for passage in passages:
                rec = build_record(
                    passage=passage,
                    language=language,
                    source=source,
                    provenance=f"{source}:{sf.path.name}",
                    source_file=_rel(sf.path),
                    topic=sf.topic,
                    frequency_style="normal",  # set below by percentile
                )
                if rec.content_hash in seen_hashes:
                    continue
                seen_hashes.add(rec.content_hash)
                records.append(rec)
        per_source_counts[source] = len(records) - count_before

    # Frequency-style tagging: within-language chi2 percentile.
    chi2_vals = sorted(r.measured.get("unigram_chi2", 0.0) for r in records)
    threshold = (
        _percentile(chi2_vals, UNUSUAL_CHI2_PERCENTILE)
        if len(chi2_vals) >= 5
        else float("inf")
    )
    for r in records:
        if r.measured.get("unigram_chi2", 0.0) >= threshold:
            r.frequency_style = "unusual"

    # Deterministic global order (by id) and optional cap.
    records.sort(key=lambda r: r.id)
    capped = False
    if max_per_lang is not None and len(records) > max_per_lang:
        capper = random.Random(f"{seed}:cap:{language}")
        records = sorted(capper.sample(records, k=max_per_lang), key=lambda r: r.id)
        capped = True

    style_counts: dict[str, int] = {}
    era_counts: dict[str, int] = {}
    length_counts: dict[str, int] = {}
    for r in records:
        style_counts[r.frequency_style] = style_counts.get(r.frequency_style, 0) + 1
        era_counts[r.era] = era_counts.get(r.era, 0) + 1
        length_counts[str(r.length_words)] = length_counts.get(str(r.length_words), 0) + 1

    stats = {
        "language": language,
        "count": len(records),
        "per_source_sampled": per_source_counts,
        "sources": sorted(by_source.keys()),
        "unusual_chi2_threshold": (None if threshold == float("inf") else round(threshold, 4)),
        "by_frequency_style": style_counts,
        "by_era": era_counts,
        "by_length_words": length_counts,
        "capped": capped,
    }
    if verbose:
        print(f"[{language}] {len(records)} passages "
              f"(sources={stats['sources']}, styles={style_counts}, eras={era_counts})")
    return records, stats


def write_library(
    out_dir: Path, language: str, records: list[PassageRecord]
) -> tuple[Path, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{language}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True))
            f.write("\n")
    return path, path.stat().st_size


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    ap.add_argument(
        "--sources", nargs="*", default=[],
        help="Restrict to these source names (gutenberg/oanc/masc/dta). "
             "Empty = all redistributable sources for each language.",
    )
    ap.add_argument("--files-per-source", type=int, default=30,
                    help="Max source files sampled per (language, source). 0 = all.")
    ap.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS),
                    help="Graded passage lengths in words.")
    ap.add_argument("--per-length", type=int, default=1,
                    help="Passages sampled per length per file.")
    ap.add_argument("--max-per-lang", type=int, default=None,
                    help="Hard cap on passages per language (post-sample).")
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_DATA,
                    help="Root of corpus_data/ (git-ignored; may be outside the worktree).")
    ap.add_argument("--dta-root", type=Path, default=DEFAULT_DTA_ROOT,
                    help="Root of the DTA corpus (corpora/dta).")
    ap.add_argument("--out", type=Path, default=DEFAULT_LIBRARY_DIR,
                    help="Output directory for <lang>.jsonl + manifest.json.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and per-language counts; write nothing.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    # Point the module-level corpus roots at the (possibly overridden) paths.
    global CORPUS_DATA, DTA_ROOT
    CORPUS_DATA = args.corpus_root
    DTA_ROOT = args.dta_root

    include_sources = set(args.sources)
    lengths = tuple(sorted(set(args.lengths)))

    manifest = {
        "generator": "scripts/build_plaintext_library.py",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "config": {
            "languages": list(args.languages),
            "sources_filter": sorted(include_sources) or "all_redistributable",
            "files_per_source": args.files_per_source,
            "lengths": list(lengths),
            "per_length": args.per_length,
            "max_per_lang": args.max_per_lang,
            "unusual_chi2_percentile": UNUSUAL_CHI2_PERCENTILE,
        },
        "excluded_sources": sorted(NON_REDISTRIBUTABLE_SOURCES),
        "languages": {},
        "total_passages": 0,
        "total_bytes": 0,
    }

    total = 0
    total_bytes = 0
    for language in args.languages:
        records, stats = build_language(
            language,
            include_sources=include_sources,
            files_per_source=args.files_per_source,
            lengths=lengths,
            per_length=args.per_length,
            max_per_lang=args.max_per_lang,
            seed=args.seed,
            verbose=args.verbose or args.dry_run,
        )
        stats["bytes"] = 0
        if not args.dry_run and records:
            path, size = write_library(args.out, language, records)
            stats["file"] = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
            stats["bytes"] = size
            total_bytes += size
        manifest["languages"][language] = stats
        total += len(records)

    manifest["total_passages"] = total
    manifest["total_bytes"] = total_bytes

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\n[dry-run] {total} passages across {len(args.languages)} languages "
              f"(no files written)")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {total} passages ({total_bytes/1024:.1f} KiB) across "
          f"{len(args.languages)} languages to {args.out}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
