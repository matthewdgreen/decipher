from __future__ import annotations

import json
from pathlib import Path

from tools.corpus.build_model import build_model
from tools.corpus.normalize import normalize_text
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
