"""Tests for the DTA period-German corpus builder (``scripts/build_dta_corpus.py``).

No network access: the download helper is exercised only on the cached-file
(idempotent skip) path, and the builder glue runs on a tiny in-memory corpus.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_dta_corpus as dta  # noqa: E402
from tools.corpus.normalize import normalize_text  # noqa: E402


# --- historical folds -------------------------------------------------------


def test_apply_historical_folds_long_s_and_r_rotunda():
    text = "ſeines gleichen voꝛ dem"
    folded, counts = dta.apply_historical_folds(text)
    assert "ſ" not in folded
    assert "ꝛ" not in folded
    assert folded == "seines gleichen vor dem"
    assert counts == {"ſ": 1, "ꝛ": 1}


def test_apply_historical_folds_counts_only_present():
    folded, counts = dta.apply_historical_folds("no historical letters here")
    assert folded == "no historical letters here"
    assert counts == {}


def test_long_s_recovered_end_to_end_through_shared_normalizer():
    # ſ must survive as 's' after the shared de normalizer runs.
    folded, _ = dta.apply_historical_folds("Chriſtus")
    assert normalize_text(folded, "de") == "christus"


def test_combining_e_umlaut_handled_by_shared_normalizer():
    # Historical umlaut aͤ (a + U+0364) needs NO explicit fold: the shared
    # normalizer strips the combining mark and keeps the base vowel, matching ä→a.
    folded, counts = dta.apply_historical_folds("Taͤglich")  # = täglich
    assert counts == {}  # no explicit fold fired
    assert normalize_text(folded, "de") == "taglich"


def test_sharp_s_folds_to_ss_via_shared_normalizer():
    folded, _ = dta.apply_historical_folds("muß")
    assert normalize_text(folded, "de") == "muss"


def test_nasal_tilde_base_letter_preserved():
    # Combining tilde (nasal bar) base letter is kept by the shared normalizer.
    # DTA original text uses the COMBINING form (base + U+0303), e.g. "vn̄"/"vñ";
    # the precomposed ñ never occurs in the sampled corpus.
    text = "vñ"  # v + n + COMBINING TILDE
    folded, counts = dta.apply_historical_folds(text)
    assert counts == {}
    assert normalize_text(folded, "de") == "vn"


# --- ground-truth firewall --------------------------------------------------


@pytest.mark.parametrize(
    "name,excluded",
    [
        ("1700-1799/goethe_werther_1774.txt", False),
        ("1700-1799/copiale_leak.txt", True),
        ("some/benchmark_plaintext.txt", True),
        ("normal_document.txt", False),
    ],
)
def test_member_exclusion(name, excluded):
    assert dta._member_excluded(name) is excluded


# --- resumable / idempotent download ---------------------------------------


def test_download_zip_reuses_cached_file_without_network(tmp_path):
    dest = tmp_path / "cached.zip"
    dest.write_bytes(b"already here")
    expected = dta.sha256_file(dest)
    # expected_sha matches -> no network, reports not downloaded.
    sha, downloaded = dta.download_zip("https://example.invalid/x.zip", dest, expected_sha=expected)
    assert sha == expected
    assert downloaded is False


def _zip_bytes(members: dict[str, str] | None = None) -> bytes:
    import io

    members = members or {"x.txt": "payload"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_download_zip_reuses_cached_file_when_no_expected_sha(tmp_path):
    # With no checksum to verify, a structurally valid zip cache is trusted.
    dest = tmp_path / "cached.zip"
    dest.write_bytes(_zip_bytes())
    sha, downloaded = dta.download_zip("https://example.invalid/x.zip", dest, expected_sha=None)
    assert downloaded is False
    assert sha == dta.sha256_file(dest)


def test_download_zip_redownloads_corrupt_cache_without_sha(tmp_path, monkeypatch):
    # A cached file that is not a valid zip (and no checksum) must be replaced.
    dest = tmp_path / "cached.zip"
    dest.write_bytes(b"not a zip archive")
    payload = _zip_bytes()
    monkeypatch.setattr(dta, "fetch_bytes", lambda url: payload)

    sha, downloaded = dta.download_zip("https://example.invalid/x.zip", dest, expected_sha=None)
    assert downloaded is True
    assert zipfile.is_zipfile(dest)
    assert sha == dta.sha256_file(dest)
    assert not (tmp_path / "cached.zip.part").exists()


def test_download_zip_writes_atomically(tmp_path, monkeypatch):
    # The payload lands via a .part temp file + os.replace: dest never appears
    # until the write is complete, and no .part residue is left behind.
    dest = tmp_path / "1700-1799.zip"
    payload = _zip_bytes()
    seen = {}

    def fake_fetch(url):
        # Mid-download, no partial dest should be visible.
        seen["dest_during_fetch"] = dest.exists()
        return payload

    monkeypatch.setattr(dta, "fetch_bytes", fake_fetch)

    sha, downloaded = dta.download_zip("https://example.invalid/x.zip", dest)
    assert downloaded is True
    assert seen["dest_during_fetch"] is False
    assert dest.exists()
    assert zipfile.is_zipfile(dest)
    assert sha == dta.sha256_file(dest)
    assert not (tmp_path / "1700-1799.zip.part").exists()


# --- extraction + cleaning --------------------------------------------------


def _make_period_zip(zip_path: Path, period: str, members: dict[str, str]) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{period}/", "")  # directory entry, must be ignored
        for name, content in members.items():
            zf.writestr(f"{period}/{name}", content)


def test_extract_and_clean_folds_and_excludes(tmp_path):
    zip_path = tmp_path / "1700-1799.zip"
    _make_period_zip(
        zip_path,
        "1700-1799",
        {
            "arndt.txt": "Chriſtus muß leben voꝛ dem Volk",
            "copiale_secret.txt": "should be excluded",
        },
    )
    text_dir = tmp_path / "text"
    stats = dta.extract_and_clean(zip_path, "1700-1799", text_dir)

    assert stats["documents"] == 1
    assert stats["excluded_documents"] == 1
    assert stats["fold_occurrences"].get("U+017F") == 1  # long s
    assert stats["fold_occurrences"].get("U+A75B") == 1  # r rotunda

    cleaned = (text_dir / "1700-1799" / "arndt.txt").read_text(encoding="utf-8")
    assert "ſ" not in cleaned and "ꝛ" not in cleaned
    assert "Christus" in cleaned
    assert not (text_dir / "1700-1799" / "copiale_secret.txt").exists()


# --- manifest round-trip ----------------------------------------------------


def _sample_manifest(tmp_path: Path) -> dict:
    downloads = [
        {
            "period": "1700-1799",
            "url": dta.period_url("1700-1799"),
            "filename": "1700-1799.zip",
            "sha256": "a" * 64,
            "bytes": 12345,
        }
    ]
    period_stats = [
        {
            "period": "1700-1799",
            "documents": 3,
            "excluded_documents": 0,
            "folded_characters": 999,
            "fold_occurrences": {"U+017F": 42},
        }
    ]
    return dta.build_manifest(
        downloads=downloads,
        period_stats=period_stats,
        corpus_dir=tmp_path,
        text_dir=tmp_path / "text",
    )


def test_build_manifest_round_trips_and_records_provenance(tmp_path):
    manifest = _sample_manifest(tmp_path)
    # JSON round-trip is lossless.
    restored = json.loads(json.dumps(manifest, ensure_ascii=False))
    assert restored == manifest

    # Provenance: URLs, license, SHA256s.
    assert manifest["license"] == "CC BY-SA 4.0"
    assert manifest["license_url"].startswith("https://creativecommons.org/")
    assert manifest["download_page"] == dta.DTA_DOWNLOAD_PAGE
    dl = manifest["downloads"][0]
    assert dl["url"].startswith("https://www.deutschestextarchiv.de/")
    assert len(dl["sha256"]) == 64

    # Historical folds documented, including long-s.
    fold_from = {f["from"] for f in manifest["historical_folds"]}
    assert "ſ" in fold_from and "ꝛ" in fold_from
    assert any(f["codepoint"] == "U+017F" for f in manifest["historical_folds"])

    # Shared-normalizer delegation documented.
    assert "normalize_text" in manifest["shared_normalization"]["delegated_to"]

    # Ground-truth firewall statement + exclusion count.
    fw = manifest["ground_truth_firewall"]
    assert "copiale" in fw["excluded_name_tokens"]
    assert fw["excluded_documents"] == 0
    assert "No Copiale" in fw["statement"]


def test_manifest_sources_are_builder_compatible(tmp_path):
    manifest = _sample_manifest(tmp_path)
    sources = manifest["sources"]
    assert isinstance(sources, list) and sources
    entry = sources[0]
    # Shape the tools.corpus builder / model sidecar consumes.
    for key in ("name", "language", "license", "url", "text_files"):
        assert key in entry
    assert entry["language"] == "de"
    assert entry["text_files"] == 3


# --- builder invocation glue (no network) ----------------------------------


def test_builder_glue_produces_comparable_v1_de_model(tmp_path):
    """extract_and_clean -> tools.corpus build_model -> valid v1 de model.

    Confirms the DTA corpus feeds the existing v1 ``de`` builder and that the
    resulting model's normalization matches the existing ``de`` model's policy.
    """
    from tools.corpus.build_model import build_model
    from analysis.zenith_solver import load_zenith_binary_model

    # A tiny period corpus with period orthography (long-s, sharp-s, umlaut).
    zip_path = tmp_path / "1700-1799.zip"
    body = (
        "Chriſtus muß taͤglich leben und wirken/ alſo daß das Wort "
        "Gottes in uns Frucht bringe und wir im Glauben beſtehen "
    ) * 40
    _make_period_zip(zip_path, "1700-1799", {"doc1.txt": body, "doc2.txt": body})
    text_dir = tmp_path / "text"
    dta.extract_and_clean(zip_path, "1700-1799", text_dir)

    manifest = _sample_manifest(tmp_path)
    output = tmp_path / "ngram5_de_dta_test.bin"
    stats = build_model(
        language="de",
        corpus_dir=text_dir,
        output_path=output,
        sources=manifest["sources"],
    )
    assert output.exists()
    assert stats.raw_files == 2
    assert stats.normalized_characters > 0

    sidecar = json.loads((output.with_name(output.name + ".metadata.json")).read_text(encoding="utf-8"))
    assert sidecar["format"] == "zenith_binary_v1"
    assert sidecar["order"] == 5
    assert sidecar["array_length"] == 26 ** 5
    # Normalization policy must match the existing de model's sidecar policy.
    assert sidecar["normalization"] == {"lowercase": True, "strip_non_alpha": True}
    assert sidecar["sources"] == manifest["sources"]

    model = load_zenith_binary_model(output)
    assert model.order == 5
    assert len(model.alphabet) == 26
    # A 5-gram present in the (long-s folded) corpus scores above the floor.
    assert model.lookup("chris") > model.unknown_log_prob


def test_builder_normalization_matches_real_de_sidecar_if_present():
    """The real ngram5_de.bin sidecar (if built) uses the same normalization
    field the DTA builder relies on for comparability."""
    de_sidecar = _REPO_ROOT / "models" / "ngram5_de.bin.metadata.json"
    if not de_sidecar.exists():
        pytest.skip("models/ngram5_de.bin.metadata.json not present")
    meta = json.loads(de_sidecar.read_text(encoding="utf-8"))
    assert meta["normalization"] == {"lowercase": True, "strip_non_alpha": True}
    assert meta["language"] == "de"
