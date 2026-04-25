from __future__ import annotations

import csv
import re
import time
from pathlib import Path

from .common import fetch_bytes, update_manifest

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"

_LEADING_BOILERPLATE_PATTERNS = (
    re.compile(r"^\s*produced by\b", re.IGNORECASE),
    re.compile(r"^\s*thanks to\b", re.IGNORECASE),
    re.compile(r"^\s*this file was produced from images\b", re.IGNORECASE),
    re.compile(r"^\s*this e-?text is in\b", re.IGNORECASE),
    re.compile(r"^\s*this e-?text was produced\b", re.IGNORECASE),
    re.compile(r"^\s*we are releasing two versions of this e-?text\b", re.IGNORECASE),
    re.compile(r"^\s*this is the \d+-bit version\b", re.IGNORECASE),
    re.compile(r"^\s*this book content was graciously contributed\b", re.IGNORECASE),
    re.compile(r"^\s*project gutenberg\b", re.IGNORECASE),
    re.compile(r"^\s*project runeberg\b", re.IGNORECASE),
    re.compile(r"^\s*projekt-?de\b", re.IGNORECASE),
    re.compile(r"^\s*dieses buch wurde uns freundlicherweise\b", re.IGNORECASE),
    re.compile(r"^\s*this text was originally produced in html\b", re.IGNORECASE),
    re.compile(r"^\s*this text has been derived from html files\b", re.IGNORECASE),
    re.compile(r"^\s*(online )?distributed proofreading team\b", re.IGNORECASE),
    re.compile(r"^\s*transcriber[’']?s note\b", re.IGNORECASE),
    re.compile(r"^\s*[\{\[]?transcriber[’']?s note", re.IGNORECASE),
    re.compile(r"^\s*all material in\b", re.IGNORECASE),
    re.compile(r"^\s*material added by the transcriber\b", re.IGNORECASE),
    re.compile(r"^\s*dies ist ein zwischenstand\b", re.IGNORECASE),
    re.compile(r"^\s*die digitalisierung wird unter\b", re.IGNORECASE),
    re.compile(r"^\s*erarbeitet; dort kann man\b", re.IGNORECASE),
    re.compile(r"^\s*wenn korrekturen vornehmen wollen\b", re.IGNORECASE),
    re.compile(r"^\s*die html-formatierung ist bislang\b", re.IGNORECASE),
    re.compile(r"^\s*we need more volunteers like you\b", re.IGNORECASE),
    re.compile(r"^\s*learn more at https?://", re.IGNORECASE),
    re.compile(r"^\s*available at https?://", re.IGNORECASE),
    re.compile(r"^\s*this is a plain text file\b", re.IGNORECASE),
    re.compile(r'^\s*".+"\s+is\s+a\s+', re.IGNORECASE),
)

_LEADING_METADATA_CONTINUATION_PATTERNS = (
    re.compile(r"^\s*https?://", re.IGNORECASE),
    re.compile(r".+@.+"),
    re.compile(r"^\s*\(https?://", re.IGNORECASE),
    re.compile(r"^\s*\[in [^\]]+\]\s*$", re.IGNORECASE),
    re.compile(r"^\s*proofreaders\.?$", re.IGNORECASE),
    re.compile(r"^\s*biblioth[eè]que nationale de france\b", re.IGNORECASE),
    re.compile(r"^\s*prepared by\b", re.IGNORECASE),
    re.compile(r"^\s*from pages\b", re.IGNORECASE),
    re.compile(r"^\s*of Goethe's works published in\b", re.IGNORECASE),
    re.compile(r"^\s*hamburger ausgabe\b", re.IGNORECASE),
    re.compile(r"^\s*verlagshandlung\b", re.IGNORECASE),
    re.compile(r"^\s*isbn\b", re.IGNORECASE),
    re.compile(r'^\s*".+"\s+by\s+.+', re.IGNORECASE),
    re.compile(r"^\s*[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\.\s*$"),
    re.compile(r"^\s*die digitalisierung\b", re.IGNORECASE),
    re.compile(r"^\s*digitalisierung wird unter\b", re.IGNORECASE),
    re.compile(r"^\s*selbst an der arbeit teilnehmen\b", re.IGNORECASE),
    re.compile(r"^\s*ein korrigierten eintrag\b", re.IGNORECASE),
    re.compile(r"^\s*einarbeitung\b", re.IGNORECASE),
    re.compile(r"^\s*beachten sie\b", re.IGNORECASE),
    re.compile(r"^\s*rechtschreibung beibehalten!?$", re.IGNORECASE),
    re.compile(r"^\s*die html-formatierung ist bislang\b", re.IGNORECASE),
    re.compile(r"^\s*korrigieren nicht unn[oö]tig\b", re.IGNORECASE),
    re.compile(r"^\s*karl eichwalder\b", re.IGNORECASE),
    re.compile(r"^\s*contents\s*$", re.IGNORECASE),
)

_CONTENT_START_PATTERNS = (
    re.compile(r"^[A-ZÆŒ][A-ZÆŒ\s\.,;:!\?'’\"()\-\[\]&]{8,}$"),
    re.compile(r"^[A-ZÆŒ][a-zA-ZÆŒæœ\s\.,;:!\?'’\"()\-\[\]&]{12,}$"),
    re.compile(r"^(?:\s{2,})?[A-ZÆŒ][a-zA-ZÆŒæœ]+"),
)


def download_catalog(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "pg_catalog.csv"
    target.write_bytes(fetch_bytes(CATALOG_URL))
    return target


def iter_language_book_ids(catalog_path: Path, language: str = "en") -> list[int]:
    wanted = (language or "en").strip().lower()
    results: list[int] = []
    with catalog_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_language = (row.get("Language") or "").strip().lower()
            row_type = (row.get("Type") or "").strip().lower()
            text_no = (row.get("Text#") or "").strip()
            if wanted not in {part.strip() for part in row_language.split(",") if part.strip()}:
                continue
            if row_type != "text" or not text_no.isdigit():
                continue
            results.append(int(text_no))
    return results


def _strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = (
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    )
    end_markers = (
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    )
    lines = text.splitlines()
    start_idx = 0
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if any(marker in line for marker in start_markers):
            start_idx = i + 1
            break
    for i in range(start_idx, len(lines)):
        if any(marker in lines[i] for marker in end_markers):
            end_idx = i
            break
    body_lines = lines[start_idx:end_idx]

    # Older Gutenberg and Runeberg exports often prepend a short English
    # descriptive block or transcriber note without standard START markers.
    # Drop those leading lines until the first plausible content/title line.
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)

    dropped_any = False
    while body_lines:
        line = body_lines[0].strip()
        if not line:
            body_lines.pop(0)
            continue
        if any(pattern.search(line) for pattern in _LEADING_BOILERPLATE_PATTERNS):
            dropped_any = True
            body_lines.pop(0)
            continue
        if dropped_any and any(pattern.search(line) for pattern in _LEADING_METADATA_CONTINUATION_PATTERNS):
            body_lines.pop(0)
            continue
        if dropped_any and any(pattern.search(line) for pattern in _CONTENT_START_PATTERNS):
            break
        if dropped_any:
            body_lines.pop(0)
            continue
        break

    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)

    # Some multilingual Gutenberg files keep a lone English "Contents" line
    # immediately after the title block. Remove that narrow case without trying
    # to strip genuine language-native tables of contents.
    for i, line in enumerate(body_lines[:40]):
        if line.strip().lower() != "contents":
            continue
        next_nonempty: list[str] = []
        for candidate in body_lines[i + 1 : i + 8]:
            stripped = candidate.strip()
            if stripped:
                next_nonempty.append(stripped)
        if next_nonempty and any(
            stripped.lower().startswith("canto") or stripped.isupper()
            for stripped in next_nonempty
        ):
            del body_lines[i]
        break

    return "\n".join(body_lines).strip()


def download_book(book_id: int, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        raw = fetch_bytes(url)
    except Exception:
        return None
    text = _strip_gutenberg_boilerplate(raw.decode("utf-8", errors="ignore"))
    if not text:
        return None
    target = output_dir / f"pg{book_id}.txt"
    target.write_text(text, encoding="utf-8")
    time.sleep(1.0)
    return target


def download_books(
    *,
    catalog_path: Path,
    output_dir: Path,
    language: str = "en",
    max_books: int = 100,
) -> list[Path]:
    downloaded: list[Path] = []
    for book_id in iter_language_book_ids(catalog_path, language=language):
        if len(downloaded) >= max_books:
            break
        path = download_book(book_id, output_dir)
        if path is not None:
            downloaded.append(path)
    return downloaded


def download_gutenberg_books(
    *,
    corpus_dir: Path,
    language: str = "en",
    max_books: int = 100,
) -> dict:
    source_dir = corpus_dir / "gutenberg"
    source_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = download_catalog(source_dir)
    paths = download_books(
        catalog_path=catalog_path,
        output_dir=source_dir,
        language=language,
        max_books=max_books,
    )
    entry = {
        "name": "gutenberg",
        "language": language,
        "license": "Project Gutenberg terms",
        "url": CATALOG_URL,
        "redistributable": True,
        "text_files": len(paths),
        "max_books": max_books,
    }
    update_manifest(corpus_dir, language=language, source_entry=entry)
    return entry
