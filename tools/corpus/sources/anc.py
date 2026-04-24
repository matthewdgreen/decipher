from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import (
    copy_text_file,
    extract_archive,
    stage_archive,
    update_manifest,
    with_temporary_dir,
)


OANC_XML_URL = "https://www.anc.org/OANC/OANC-1.0.1-UTF8.zip"
MASC_TEXT_URL = "https://www.anc.org/MASC/download/masc_500k_texts.tgz"


def _looks_like_oanc_primary_xml(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if any(token in parts for token in {"anc-header", "headers", "header", "annotations", "resourceannotations"}):
        return False
    return path.suffix.lower() == ".xml"


def _xml_to_text(src: Path, dst: Path) -> bool:
    try:
        root = ET.parse(src).getroot()
    except ET.ParseError:
        return False
    text = " ".join(chunk.strip() for chunk in root.itertext() if chunk and chunk.strip())
    text = " ".join(text.split())
    if not text:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return True


def download_masc_texts(*, corpus_dir: Path, language: str = "en") -> dict:
    source_dir = corpus_dir / "masc"
    archive_dir = corpus_dir / "_archives"
    archive_path = stage_archive(MASC_TEXT_URL, archive_dir, "masc_500k_texts.tgz")
    with with_temporary_dir() as tmp:
        extracted = extract_archive(archive_path, Path(tmp) / "masc")
        text_count = 0
        if source_dir.exists():
            shutil.rmtree(source_dir)
        for path in sorted(extracted.rglob("*.txt")):
            rel = path.relative_to(extracted)
            if copy_text_file(path, source_dir / rel):
                text_count += 1
    entry = {
        "name": "masc",
        "language": language,
        "license": "CC BY 3.0 US",
        "url": MASC_TEXT_URL,
        "redistributable": True,
        "text_files": text_count,
    }
    update_manifest(corpus_dir, language=language, source_entry=entry)
    return entry


def download_oanc_texts(*, corpus_dir: Path, language: str = "en") -> dict:
    source_dir = corpus_dir / "oanc"
    archive_dir = corpus_dir / "_archives"
    archive_path = stage_archive(OANC_XML_URL, archive_dir, "OANC-1.0.1-UTF8.zip")
    with with_temporary_dir() as tmp:
        extracted = extract_archive(archive_path, Path(tmp) / "oanc")
        text_count = 0
        if source_dir.exists():
            shutil.rmtree(source_dir)
        for path in sorted(extracted.rglob("*.txt")):
            rel = path.relative_to(extracted)
            if copy_text_file(path, source_dir / rel):
                text_count += 1
        for path in sorted(extracted.rglob("*.xml")):
            if not _looks_like_oanc_primary_xml(path):
                continue
            rel = path.relative_to(extracted).with_suffix(".txt")
            if _xml_to_text(path, source_dir / rel):
                text_count += 1
    entry = {
        "name": "oanc",
        "language": language,
        "license": "Open ANC open-data terms",
        "url": OANC_XML_URL,
        "redistributable": True,
        "text_files": text_count,
        "note": "Downloaded with relaxed TLS verification because anc.org currently serves an expired certificate.",
    }
    update_manifest(corpus_dir, language=language, source_entry=entry)
    return entry
