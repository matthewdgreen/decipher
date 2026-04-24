from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import copy_text_file, update_manifest


def _strip_namespaces(root: ET.Element) -> None:
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]


def _bnc_xml_to_text(src: Path, dst: Path) -> bool:
    try:
        root = ET.parse(src).getroot()
    except ET.ParseError:
        return False
    _strip_namespaces(root)
    text = " ".join(chunk.strip() for chunk in root.itertext() if chunk and chunk.strip())
    text = " ".join(text.split())
    if not text:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return True


def import_bnc_texts(
    *,
    corpus_dir: Path,
    source_dir: Path,
    language: str = "en",
) -> dict:
    """Import a licensed local BNC checkout into the plain-text corpus tree.

    The BNC corpus itself is not redistributed by Decipher. This command copies
    text extracted from a user-provided licensed BNC directory into a local
    build tree so a derived statistical model can be produced. Provenance is
    recorded in the corpus manifest for attribution and compliance tracking.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"BNC source directory does not exist: {source_dir}")

    dest_dir = corpus_dir / "bnc"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    text_count = 0
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        rel = path.relative_to(source_dir)
        if suffix == ".txt":
            if copy_text_file(path, dest_dir / rel):
                text_count += 1
        elif suffix == ".xml":
            if _bnc_xml_to_text(path, (dest_dir / rel).with_suffix(".txt")):
                text_count += 1

    entry = {
        "name": "bnc",
        "language": language,
        "license": "Licensed local BNC source; derived models only",
        "redistributable": False,
        "derived_products_redistributable": True,
        "text_files": text_count,
        "source_path": str(source_dir),
        "attribution": "British National Corpus (licensed local copy).",
        "licensing_note": (
            "Corpus text is not redistributed. This source is used to build "
            "derived statistical models only."
        ),
    }
    update_manifest(corpus_dir, language=language, source_entry=entry)
    return entry
