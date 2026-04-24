from __future__ import annotations

import csv
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"


def _fetch(url: str, *, retries: int = 3, delay_seconds: float = 1.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url) as response:  # noqa: S310
                return response.read()
        except (HTTPError, URLError) as exc:
            last_error = exc
            if attempt + 1 == retries:
                break
            time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def download_catalog(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "pg_catalog.csv"
    target.write_bytes(_fetch(CATALOG_URL))
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
    return "\n".join(lines[start_idx:end_idx]).strip()


def download_book(book_id: int, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        raw = _fetch(url)
    except HTTPError:
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
