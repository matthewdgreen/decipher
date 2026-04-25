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


def test_strip_gutenberg_boilerplate_removes_english_preface_without_markers():
    raw = (
        '"Nova analysis aquarum Medeviensium" is a scientific paper in Latin\n'
        "about water quality.\n"
        "\n"
        "This e-text was produced for Project Gutenberg from Project Runeberg's\n"
        "digital facsimile edition, available at http://runeberg.org/example/\n"
        "\n"
        "Project Runeberg publishes free digital editions.\n"
        "We need more volunteers like you. Learn more at http://runeberg.org/\n"
        "\n"
        "REGIO COLLEGIO MEDICO\n"
        "SACRUM.\n"
    )
    assert _strip_gutenberg_boilerplate(raw) == "REGIO COLLEGIO MEDICO\nSACRUM."


def test_strip_gutenberg_boilerplate_removes_transcriber_note_block():
    raw = (
        "Produced by Someone and the Online Distributed Proofreading Team\n"
        "\n"
        "{Transcriber's Note:\n"
        "Material added by the transcriber is in braces.}\n"
        "\n"
        "De\n"
        "M. TERENTI VARRONIS\n"
        "LIBRIS GRAMMATICIS\n"
    )
    assert _strip_gutenberg_boilerplate(raw) == "De\nM. TERENTI VARRONIS\nLIBRIS GRAMMATICIS"


def test_strip_gutenberg_boilerplate_removes_projekt_de_english_preface():
    raw = (
        "This Etext is in German.\n"
        "\n"
        "We are releasing two versions of this Etext, one in 7-bit format,\n"
        "known as Plain Vanilla ASCII.\n"
        "This is the 8-bit version.\n"
        "\n"
        "This book content was graciously contributed by the Gutenberg\n"
        "Projekt-DE. That project is reachable at the web site\n"
        "http://gutenberg.spiegel.de/.\n"
        "\n"
        "Dieses Buch wurde uns freundlicherweise vom \"Gutenberg Projekt-DE\"\n"
        "zur Verfugung gestellt.\n"
        "\n"
        "HAMBURGISCHE DRAMATURGIE\n"
        "von GOTTHOLD EPHRAIM LESSING\n"
    )
    assert _strip_gutenberg_boilerplate(raw) == "HAMBURGISCHE DRAMATURGIE\nvon GOTTHOLD EPHRAIM LESSING"


def test_strip_gutenberg_boilerplate_removes_derived_from_html_notice():
    raw = (
        "Produced by Gunther Olesch\n"
        "\n"
        "This text has been derived from HTML files at \"Projekt Gutenberg - DE\"\n"
        "(http://www.gutenberg2000.de/example), prepared by\n"
        "Gerd Bouillon.\n"
        "\n"
        "Johanna Spyri\n"
        "\n"
        "Heidi kann brauchen, was es gelernt hat\n"
    )
    assert _strip_gutenberg_boilerplate(raw) == "Johanna Spyri\n\nHeidi kann brauchen, was es gelernt hat"


def test_strip_gutenberg_boilerplate_removes_digitization_notice():
    raw = (
        "Dies ist ein Zwischenstand (Oktober 2003) der Digitalisierung von\n"
        "\"Meyers Konversationslexikon\".\n"
        "Die Digitalisierung wird unter\n"
        "http://www.meyers-konversationslexikon.de\n"
        "erarbeitet; dort kann man auch den jeweils aktuellen Stand einsehen.\n"
        "Wenn Korrekturen vornehmen wollen, melden Sie sich bitte.\n"
        "Die HTML-Formatierung ist bislang bewusst einfach gehalten.\n"
        "\n"
        "S.\n"
        "\n"
        "Das im laufenden Alphabet nicht Verzeichnete ist im Register des\n"
        "Schlußbandes aufzusuchen.\n"
    )
    assert _strip_gutenberg_boilerplate(raw) == (
        "Das im laufenden Alphabet nicht Verzeichnete ist im Register des\nSchlußbandes aufzusuchen."
    )


def test_strip_gutenberg_boilerplate_removes_html_edition_block():
    raw = (
        "Thanks to Andrew Sly.\n"
        "\n"
        "\"Satyros oder Der vergötterte Waldteufel\" by Johann Wolfgang Goethe\n"
        "[in German]\n"
        "\n"
        "This text was originally produced in HTML for Projekt-Gutenberg-DE by\n"
        "belmekhira@hotmail.com from pages 188 to 202 of \"Goethes Werke,\n"
        "Hamburger Ausgabe, Band 4 Dramen II\", the fourth volume of an edition\n"
        "of Goethe's works published in 1982 by \"C.H. Beck'sche\n"
        "Verlagshandlung, München\", ISBN 3-406-08484-2.\n"
        "\n"
        "Johann Wolfgang Goethe\n"
        "\n"
        "Satyros\n"
    )
    assert _strip_gutenberg_boilerplate(raw) == "Johann Wolfgang Goethe\n\nSatyros"


def test_strip_gutenberg_boilerplate_removes_gallica_image_preface():
    raw = (
        "This file was produced from images generously made available by the\n"
        "Bibliothèque nationale de France (BnF/Gallica) at http://gallica.bnf.fr.\n"
        "\n"
        "Produced by Carlo Traverso, Anne Dreze and the PG Online Distributed\n"
        "Proofreaders.\n"
        "\n"
        "LA VAMPIRE\n"
        "\n"
        "par\n"
        "\n"
        "PAUL FÉVAL\n"
    )
    assert _strip_gutenberg_boilerplate(raw) == "LA VAMPIRE\n\npar\n\nPAUL FÉVAL"


def test_strip_gutenberg_boilerplate_removes_english_contents_header():
    raw = (
        "LA DIVINA COMMEDIA\n"
        "\n"
        "di Dante Alighieri\n"
        "\n"
        "Contents\n"
        "\n"
        "INFERNO\n"
        "Canto I.\n"
    )
    assert _strip_gutenberg_boilerplate(raw) == (
        "LA DIVINA COMMEDIA\n\ndi Dante Alighieri\n\n\nINFERNO\nCanto I."
    )


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
