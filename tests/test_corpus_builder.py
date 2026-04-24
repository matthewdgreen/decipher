from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.corpus.__main__ import _resolve_sources
from tools.corpus.build_model import build_model
from tools.corpus.normalize import normalize_text
from tools.corpus.sources.anc import download_masc_texts, download_oanc_texts
from tools.corpus.sources.bnc import import_bnc_texts
from tools.corpus.sources.common import load_manifest
from tools.corpus.sources.gutenberg import _strip_gutenberg_boilerplate
from tools.corpus.verify_model import verify_model
from automated import runner as automated_runner


def test_normalize_english():
    assert normalize_text("Hello, World!", "en") == "helloworld"


def test_normalize_german_umlauts():
    assert normalize_text("Über straße", "de") == "uberstrasse"


def test_normalize_latin_diacritics():
    assert normalize_text("Aenēās", "la") == "aeneas"


def test_normalize_french_diacritics():
    assert normalize_text("château", "fr") == "chateau"


def test_strip_gutenberg_boilerplate():
    raw = (
        "intro\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Body text here\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\noutro"
    )
    assert _strip_gutenberg_boilerplate(raw) == "Body text here"


def test_build_model_roundtrip_and_metadata(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("hello world there hello world", encoding="utf-8")
    (corpus_dir / "b.txt").write_text("another there world hello", encoding="utf-8")

    output = tmp_path / "ngram5_en.bin"
    stats = build_model(language="en", corpus_dir=corpus_dir, output_path=output)

    assert output.exists()
    assert stats.metadata_path.exists()

    verified = verify_model(output)
    assert verified["shape_ok"] is True
    assert verified["sample_there"] > verified["unknown_log_prob"]

    metadata = json.loads(stats.metadata_path.read_text(encoding="utf-8"))
    assert metadata["language"] == "en"
    assert metadata["output_file"] == "ngram5_en.bin"
    assert metadata["corpus_stats"]["raw_files"] == 2


def test_zenith_native_model_path_prefers_repo_model(tmp_path: Path, monkeypatch):
    repo_root = tmp_path
    model_dir = repo_root / "models"
    model_dir.mkdir()
    expected = model_dir / "ngram5_en.bin"
    expected.write_bytes(b"model")
    monkeypatch.setattr(
        automated_runner,
        "__file__",
        str(repo_root / "src" / "automated" / "runner.py"),
    )

    assert automated_runner._zenith_native_model_path("en") == expected


def test_zenith_native_model_path_uses_language_env(tmp_path: Path, monkeypatch):
    expected = tmp_path / "ngram5_de.bin"
    expected.write_bytes(b"model")
    monkeypatch.setenv("DECIPHER_NGRAM_MODEL_DE", str(expected))

    assert automated_runner._zenith_native_model_path("de") == expected


def test_zenith_native_model_path_non_english_returns_none_without_model(tmp_path: Path, monkeypatch):
    repo_root = tmp_path
    monkeypatch.setattr(
        automated_runner,
        "__file__",
        str(repo_root / "src" / "automated" / "runner.py"),
    )

    assert automated_runner._zenith_native_model_path("de") is None


def test_build_model_only_counts_text_files(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("hello world", encoding="utf-8")
    (corpus_dir / "notes.json").write_text("{\"a\": 1}", encoding="utf-8")

    output = tmp_path / "ngram5_en.bin"
    stats = build_model(language="en", corpus_dir=corpus_dir, output_path=output)

    assert stats.raw_files == 1


def test_download_masc_texts_extracts_plain_text_and_updates_manifest(tmp_path: Path, monkeypatch):
    archive_dir = tmp_path / "corpus" / "_archives"
    archive_dir.mkdir(parents=True)
    archive_path = archive_dir / "masc_500k_texts.tgz"
    sample_txt = tmp_path / "sample.txt"
    sample_txt.write_text("MASC sample text", encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(sample_txt, arcname="masc_500k_texts/doc1.txt")

    entry = download_masc_texts(corpus_dir=tmp_path / "corpus")

    extracted = tmp_path / "corpus" / "masc" / "masc_500k_texts" / "doc1.txt"
    assert extracted.exists()
    assert extracted.read_text(encoding="utf-8") == "MASC sample text"
    assert entry["text_files"] == 1
    manifest = load_manifest(tmp_path / "corpus")
    assert manifest["sources"][0]["name"] == "masc"


def test_download_oanc_texts_extracts_xml_text_and_skips_headers(tmp_path: Path):
    archive_dir = tmp_path / "corpus" / "_archives"
    archive_dir.mkdir(parents=True)
    archive_path = archive_dir / "OANC-1.0.1-UTF8.zip"
    body_xml = "<root><p>Hello <w>world</w></p></root>"
    header_xml = "<header><title>Metadata only</title></header>"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("OANC/data/doc1.xml", body_xml)
        zf.writestr("OANC/anc-header/doc1.xml", header_xml)

    entry = download_oanc_texts(corpus_dir=tmp_path / "corpus")

    extracted = tmp_path / "corpus" / "oanc" / "OANC" / "data" / "doc1.txt"
    assert extracted.exists()
    assert extracted.read_text(encoding="utf-8") == "Hello world"
    assert entry["text_files"] == 1
    manifest = load_manifest(tmp_path / "corpus")
    names = {source["name"] for source in manifest["sources"]}
    assert "oanc" in names


def test_resolve_sources_defaults_to_gutenberg_for_non_english():
    assert _resolve_sources("de", None) == ["gutenberg"]
    assert _resolve_sources("la", None) == ["gutenberg"]


def test_resolve_sources_rejects_english_only_sources_for_non_english():
    with pytest.raises(ValueError, match="Unsupported source"):
        _resolve_sources("de", ["oanc"])


def test_resolve_sources_allows_bnc_for_english():
    assert _resolve_sources("en", ["bnc"]) == ["bnc"]


def test_import_bnc_texts_extracts_xml_and_records_attribution(tmp_path: Path):
    source_dir = tmp_path / "bnc_source"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("Plain BNC text", encoding="utf-8")
    (source_dir / "b.xml").write_text(
        "<root><s>Structured</s><s>BNC</s><s>text</s></root>",
        encoding="utf-8",
    )

    entry = import_bnc_texts(corpus_dir=tmp_path / "corpus", source_dir=source_dir)

    imported_txt = tmp_path / "corpus" / "bnc" / "a.txt"
    imported_xml = tmp_path / "corpus" / "bnc" / "b.txt"
    assert imported_txt.exists()
    assert imported_xml.exists()
    assert imported_xml.read_text(encoding="utf-8") == "Structured BNC text"
    assert entry["name"] == "bnc"
    assert entry["derived_products_redistributable"] is True
    manifest = load_manifest(tmp_path / "corpus")
    assert manifest["sources"][0]["attribution"].startswith("British National Corpus")
