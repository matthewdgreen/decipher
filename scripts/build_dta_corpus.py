#!/usr/bin/env python3
"""Download and normalize the Deutsches Textarchiv (DTA) period-German corpus.

This builds a stronger 17th–19th-century German training corpus than the current
100-book Gutenberg ``de`` model. It downloads the DTA *Kernkorpus* published
plain-text export ("original" orthography variant, version 2020-10-23) for the
three period packages 1600–1699, 1700–1799, 1800–1899, records provenance (source
URLs, license, SHA256) in a corpus manifest, applies the historical-orthography
folds that the shared ``de`` normalizer does not handle, and writes cleaned
per-document ``.txt`` files ready for the existing ``tools.corpus`` builder.

Design notes
------------
* **Normalization parity.** The *modern* normalization (lowercase, ä→a ö→o ü→u
  ß→ss, strip non ``[a-z]``) is delegated entirely to
  ``tools.corpus.normalize.normalize_text(text, "de")`` at model-build time —
  byte-for-byte the same policy the existing ``ngram5_de.bin`` model used, so the
  two models are directly comparable. This script only applies the *historical*
  folds that the shared normalizer would otherwise drop (recovering real letters):

    - LATIN SMALL LETTER LONG S ``ſ`` (U+017F) → ``s``
    - LATIN SMALL LETTER R ROTUNDA ``ꝛ`` (U+A75B) → ``r``

  Combining historical diacritics — COMBINING LATIN SMALL LETTER E ``◌ͤ`` (U+0364,
  the historical umlaut, e.g. ``aͤ`` = ä) and COMBINING TILDE ``◌̃`` (U+0303, the
  nasal abbreviation bar) — need NO explicit fold: the shared normalizer strips
  the combining mark and keeps the base letter, which yields exactly the same
  ``[a-z]`` result the modern ``de`` policy would (ä→a etc.). Every fold and this
  delegation are recorded in the manifest.
* **Ground-truth firewall.** No Copiale or benchmark-derived text is included: the
  DTA Kernkorpus does not contain the Copiale manuscript, and a filename filter
  drops any member whose name matches ``copiale``/``benchmark`` as a safeguard.
  The manifest states this and records the excluded count (expected 0).
* **Streaming.** Documents are extracted and normalized one at a time; the model
  build (``tools.corpus``) counts n-grams per file, so no full-corpus load.
* **Resumable / idempotent.** A downloaded zip is reused if present and (when the
  manifest already records its SHA256) checksum-verified; extraction is skipped
  when the recorded SHA256 is unchanged and the period text dir is populated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# ``tools`` lives at the repo root; ``src`` holds the runtime packages.
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the shared HTTP helper so retry/SSL behavior matches the other sources.
from tools.corpus.sources.common import fetch_bytes  # noqa: E402

# --- DTA source description ------------------------------------------------

DTA_DOWNLOAD_PAGE = "https://www.deutschestextarchiv.de/download"
DTA_LICENSE = "CC BY-SA 4.0"
DTA_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
DTA_EXPORT_VERSION = "2020-10-23"
DTA_VARIANT = "original"  # original orthography (preserves ſ, historical umlauts)
_BASE = f"https://www.deutschestextarchiv.de/media/download/dtak/{DTA_EXPORT_VERSION}/{DTA_VARIANT}"

# The three period packages that together cover the DTA Kernkorpus 17th–19th c.
PERIODS: tuple[str, ...] = ("1600-1699", "1700-1799", "1800-1899")


def period_url(period: str) -> str:
    return f"{_BASE}/{period}.zip"


# --- Historical-orthography folds (see module docstring) -------------------

# Only letterform substitutions the shared de normalizer would otherwise DROP.
HISTORICAL_FOLDS: dict[str, str] = {
    "ſ": "s",   # ſ LATIN SMALL LETTER LONG S
    "ꝛ": "r",   # ꝛ LATIN SMALL LETTER R ROTUNDA
}
_FOLD_TABLE = {ord(k): v for k, v in HISTORICAL_FOLDS.items()}

HISTORICAL_FOLD_DOCS = [
    {
        "from": "ſ",
        "to": "s",
        "codepoint": "U+017F",
        "name": "LATIN SMALL LETTER LONG S",
        "note": "Positional variant of s; recovered as 's'.",
    },
    {
        "from": "ꝛ",
        "to": "r",
        "codepoint": "U+A75B",
        "name": "LATIN SMALL LETTER R ROTUNDA",
        "note": "Positional variant of r; recovered as 'r'.",
    },
]

# Combining marks the SHARED normalizer already handles (mark stripped, base kept).
SHARED_NORMALIZER_NOTE = {
    "delegated_to": "tools.corpus.normalize.normalize_text(text, 'de')",
    "policy": "lowercase; ä→a ö→o ü→u ß→ss; strip all non-[a-z]",
    "byte_identical_to": "models/ngram5_de.bin normalization",
    "combining_marks_auto_handled": [
        {
            "codepoint": "U+0364",
            "name": "COMBINING LATIN SMALL LETTER E",
            "note": "Historical umlaut (e.g. aͤ = ä). Mark stripped, base vowel kept "
            "-> same [a-z] result as ä→a.",
        },
        {
            "codepoint": "U+0303",
            "name": "COMBINING TILDE",
            "note": "Nasal abbreviation bar (e.g. vñ). Mark stripped, base letter kept.",
        },
    ],
}

# Ground-truth firewall: drop members whose name suggests benchmark provenance.
_EXCLUDE_NAME_TOKENS = ("copiale", "benchmark")


def apply_historical_folds(text: str) -> tuple[str, dict[str, int]]:
    """Fold historical letterforms to their modern ``a-z`` equivalents.

    Returns the folded text and a per-fold occurrence count (keyed by the source
    character) so the manifest can quantify what was changed.
    """
    counts = {ch: text.count(ch) for ch in HISTORICAL_FOLDS}
    counts = {ch: n for ch, n in counts.items() if n}
    return text.translate(_FOLD_TABLE), counts


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _member_excluded(name: str) -> bool:
    low = name.lower()
    return any(token in low for token in _EXCLUDE_NAME_TOKENS)


def download_zip(url: str, dest: Path, *, expected_sha: str | None = None) -> tuple[str, bool]:
    """Download ``url`` to ``dest`` unless a valid copy already exists.

    Returns ``(sha256, downloaded)``. A cached ``dest`` is reused when:

    * ``expected_sha`` is given and the file's checksum matches it; or
    * ``expected_sha`` is ``None`` and the file is a structurally valid zip
      (``zipfile.is_zipfile``). Without a checksum to verify against, a
      truncated/corrupt cache would otherwise be trusted silently, so the zip
      structure serves as the integrity check.

    The payload is written to a sibling ``.part`` temp file and atomically
    moved into place with ``os.replace`` so an interrupted run never leaves a
    partial ``dest`` that a later run would mistake for a complete cache.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        sha = sha256_file(dest)
        if expected_sha is not None:
            if sha == expected_sha:
                return sha, False
            # Recorded checksum mismatch -> stale/partial file: re-download.
        elif zipfile.is_zipfile(dest):
            return sha, False
        # No checksum and not a valid zip -> corrupt cache: re-download.
    data = fetch_bytes(url)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return sha256_file(dest), True


def extract_and_clean(zip_path: Path, period: str, text_dir: Path) -> dict[str, Any]:
    """Extract a period zip, apply historical folds, and write cleaned .txt files.

    Streams member-by-member. Returns per-period stats for the manifest.
    """
    out_dir = text_dir / period
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_count = 0
    excluded = 0
    raw_chars = 0
    fold_counts: dict[str, int] = {ch: 0 for ch in HISTORICAL_FOLDS}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".txt"):
                continue
            base = Path(info.filename).name
            if _member_excluded(info.filename):
                excluded += 1
                continue
            raw = zf.read(info).decode("utf-8", errors="ignore")
            folded, counts = apply_historical_folds(raw)
            for ch, n in counts.items():
                fold_counts[ch] += n
            raw_chars += len(folded)
            (out_dir / base).write_text(folded, encoding="utf-8")
            doc_count += 1
    return {
        "period": period,
        "documents": doc_count,
        "excluded_documents": excluded,
        "folded_characters": raw_chars,
        "fold_occurrences": {
            f"U+{ord(ch):04X}": fold_counts[ch] for ch in HISTORICAL_FOLDS if fold_counts[ch]
        },
    }


def _load_existing_manifest(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _recorded_sha(existing: dict[str, Any], period: str) -> str | None:
    for entry in existing.get("downloads", []) or []:
        if entry.get("period") == period:
            return entry.get("sha256")
    return None


def build_manifest(
    *,
    downloads: list[dict[str, Any]],
    period_stats: list[dict[str, Any]],
    corpus_dir: Path,
    text_dir: Path,
) -> dict[str, Any]:
    total_docs = sum(s["documents"] for s in period_stats)
    total_excluded = sum(s["excluded_documents"] for s in period_stats)
    total_chars = sum(s["folded_characters"] for s in period_stats)
    total_fold_occ: dict[str, int] = {}
    for s in period_stats:
        for cp, n in s["fold_occurrences"].items():
            total_fold_occ[cp] = total_fold_occ.get(cp, 0) + n
    # Builder-compatible `sources` list (one entry per period) -> model sidecar.
    sources = [
        {
            "name": f"dta_kernkorpus_{d['period']}",
            "language": "de",
            "license": DTA_LICENSE,
            "url": d["url"],
            "redistributable": True,
            "text_files": stats["documents"],
            "export_version": DTA_EXPORT_VERSION,
            "variant": DTA_VARIANT,
        }
        for d, stats in zip(downloads, period_stats)
    ]
    return {
        "schema_version": 1,
        "language": "de",
        "corpus": "deutsches_textarchiv_kernkorpus",
        "description": (
            "DTA Kernkorpus period plain-text (original orthography), 17th–19th c., "
            "for a period-German 5-gram model."
        ),
        "provider": "Deutsches Textarchiv (DTA)",
        "download_page": DTA_DOWNLOAD_PAGE,
        "license": DTA_LICENSE,
        "license_url": DTA_LICENSE_URL,
        "export_version": DTA_EXPORT_VERSION,
        "variant": DTA_VARIANT,
        "built_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "text_dir": str(text_dir.relative_to(corpus_dir)) if text_dir.is_relative_to(corpus_dir) else str(text_dir),
        "downloads": downloads,
        "sources": sources,
        "historical_folds": HISTORICAL_FOLD_DOCS,
        "shared_normalization": SHARED_NORMALIZER_NOTE,
        "ground_truth_firewall": {
            "excluded_name_tokens": list(_EXCLUDE_NAME_TOKENS),
            "excluded_documents": total_excluded,
            "statement": (
                "No Copiale or benchmark-derived text is included. The DTA Kernkorpus "
                "does not contain the Copiale manuscript; a filename filter is applied "
                "as a safeguard."
            ),
        },
        "corpus_stats": {
            "documents": total_docs,
            "folded_characters": total_chars,
            "fold_occurrences": total_fold_occ,
            "periods": period_stats,
        },
    }


def run(
    *,
    corpus_dir: Path,
    periods: tuple[str, ...],
    force: bool,
) -> dict[str, Any]:
    corpus_dir = corpus_dir.expanduser()
    downloads_dir = corpus_dir / "downloads"
    text_dir = corpus_dir / "text"
    manifest_path = corpus_dir / "corpus_manifest.json"
    existing = _load_existing_manifest(manifest_path)

    downloads: list[dict[str, Any]] = []
    period_stats: list[dict[str, Any]] = []
    for period in periods:
        url = period_url(period)
        dest = downloads_dir / f"{period}.zip"
        recorded = None if force else _recorded_sha(existing, period)
        print(f"[dta] {period}: fetching {url}", flush=True)
        sha, downloaded = download_zip(url, dest, expected_sha=recorded)
        print(
            f"[dta] {period}: {'downloaded' if downloaded else 'reused cached'} "
            f"{dest.name} ({dest.stat().st_size:,} bytes) sha256={sha[:16]}…",
            flush=True,
        )
        out_dir = text_dir / period
        already_extracted = (
            not force
            and recorded == sha
            and out_dir.is_dir()
            and any(out_dir.glob("*.txt"))
        )
        if already_extracted:
            existing_stats = next(
                (s for s in existing.get("corpus_stats", {}).get("periods", []) if s["period"] == period),
                None,
            )
            if existing_stats is not None:
                print(f"[dta] {period}: extraction unchanged, reusing cleaned text", flush=True)
                stats = existing_stats
            else:
                stats = extract_and_clean(dest, period, text_dir)
        else:
            print(f"[dta] {period}: extracting + folding…", flush=True)
            stats = extract_and_clean(dest, period, text_dir)
        print(
            f"[dta] {period}: {stats['documents']} docs, "
            f"{stats['folded_characters']:,} chars, "
            f"excluded={stats['excluded_documents']}",
            flush=True,
        )
        downloads.append(
            {
                "period": period,
                "url": url,
                "filename": dest.name,
                "sha256": sha,
                "bytes": dest.stat().st_size,
            }
        )
        period_stats.append(stats)

    manifest = build_manifest(
        downloads=downloads,
        period_stats=period_stats,
        corpus_dir=corpus_dir,
        text_dir=text_dir,
    )
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[dta] wrote manifest {manifest_path}", flush=True)
    return manifest


def maybe_build_model(corpus_dir: Path, output: Path, manifest: dict[str, Any]) -> None:
    """Invoke the existing tools.corpus v1 builder on the cleaned corpus."""
    from tools.corpus.build_model import build_model

    print(f"[dta] building v1 de model -> {output} (streaming n-gram count)…", flush=True)
    stats = build_model(
        language="de",
        corpus_dir=corpus_dir,
        output_path=output,
        sources=manifest.get("sources"),
    )
    print(
        f"[dta] built {stats.output_path}: {stats.raw_files} files, "
        f"{stats.normalized_characters:,} normalized chars, "
        f"{stats.distinct_seen_ngrams:,} distinct 5-grams",
        flush=True,
    )
    print(f"[dta] metadata: {stats.metadata_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=REPO_ROOT / "corpora" / "dta",
        help="Corpus root (gitignored). Default: corpora/dta",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        default=list(PERIODS),
        choices=list(PERIODS),
        help="Which DTA period packages to include. Default: all three.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached checksums and re-download + re-extract.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="After preparing the corpus, build models/ngram5_de_dta.bin via tools.corpus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "ngram5_de_dta.bin",
        help="Model output path when --build is given. Default: models/ngram5_de_dta.bin",
    )
    args = parser.parse_args(argv)

    manifest = run(
        corpus_dir=args.corpus_dir,
        periods=tuple(args.periods),
        force=args.force,
    )
    if args.build:
        maybe_build_model(args.corpus_dir, args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
