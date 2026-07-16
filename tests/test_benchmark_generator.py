"""Tests for the Slice-C cipher benchmark generator.

Covers: BenchmarkLoader compatibility (loads + iterates every generated case),
per-family round-trip (decrypt with the stored key reproduces the stored
plaintext), the context/plaintext firewall (a rich-tier context record leaks
neither plaintext nor key), context-tier gradation, determinism (same seed ->
byte-identical manifest), and dry-run (no files written).

Offline, no LLM: drives the shipped ``resources/plaintext_library`` via the
generator's own deterministic pipeline.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_generator():
    """Import scripts/generate_cipher_benchmark.py as a module (it is not a package)."""
    path = REPO_ROOT / "scripts" / "generate_cipher_benchmark.py"
    spec = importlib.util.spec_from_file_location("generate_cipher_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Register before exec so dataclass annotation resolution can find the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()

from benchmark.loader import BenchmarkLoader  # noqa: E402
from testgen.family_registry import (  # noqa: E402
    LETTERS,
    all_family_ids,
    get_spec,
    new_family_ids,
)

ALL_FAMILIES = all_family_ids()
TIERS = ["none", "language", "era_provenance", "rich"]


def _reconstruct_ciphertext(canonical: str, form: str) -> str:
    """Undo the canonical serialisation to recover the raw ciphertext."""
    if form == LETTERS:
        return canonical.replace(" ", "")
    return canonical


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    """A small but complete suite: every family x en x medium x 1 example."""
    out = tmp_path_factory.mktemp("gen_suite")
    result = gen.generate(
        families=ALL_FAMILIES,
        languages=["en"],
        difficulties=["medium"],
        tiers=TIERS,
        per_family=1,
        seed=987654,
        out_dir=out,
        dry_run=False,
    )
    return out, result


# ---------------------------------------------------------------------------
# Loader compatibility
# ---------------------------------------------------------------------------


def test_loader_loads_and_iterates_every_case(suite):
    out, result = suite
    loader = BenchmarkLoader(out)
    tests = loader.load_tests("all_generated.jsonl")
    assert len(tests) == result.num_tests
    assert result.num_records == len(ALL_FAMILIES)
    # every family x tier is present
    assert result.num_tests == len(ALL_FAMILIES) * len(TIERS)

    seen_records = set()
    for t in tests:
        data = loader.load_test_data(t)
        assert data.canonical_transcription.strip(), t.test_id
        assert data.plaintext.strip(), t.test_id
        assert data.plaintext_language == "en", t.test_id
        assert len(data.target_records) == 1
        seen_records.add(data.target_records[0].id)
    # every generated record is referenced by at least one test
    assert len(seen_records) == len(ALL_FAMILIES)


def test_every_family_has_its_own_split(suite):
    out, _ = suite
    loader = BenchmarkLoader(out)
    for family_id in ALL_FAMILIES:
        split = out / "splits" / f"gen_{family_id}.jsonl"
        assert split.exists(), family_id
        tests = loader.load_tests(split)
        assert tests, family_id
        assert all(t.cipher_system == family_id for t in tests)


# ---------------------------------------------------------------------------
# Round-trip per family
# ---------------------------------------------------------------------------


def test_round_trip_every_family(suite):
    out, _ = suite
    manifest = {
        json.loads(line)["family_id"]: json.loads(line)
        for line in (out / "manifest" / "records.jsonl").read_text().splitlines()
    }
    assert set(manifest) == set(ALL_FAMILIES)

    for family_id, rec in manifest.items():
        spec = get_spec(family_id)
        cipher = spec.new_cipher()
        key = json.loads((out / rec["key_file"]).read_text())["key"]
        canonical = (out / rec["transcription_canonical_file"]).read_text().strip()
        plaintext = (out / rec["plaintext_file"]).read_text().strip()

        ciphertext = _reconstruct_ciphertext(canonical, rec["canonical_form"])
        # decrypt with the stored key reproduces the stored (prepared) plaintext
        assert cipher.decrypt(ciphertext, key) == plaintext, family_id
        # and re-enciphering the stored plaintext reproduces the ciphertext
        assert cipher.encrypt(plaintext, key) == ciphertext, family_id


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------


def test_firewall_rich_tier_has_no_plaintext_or_key(suite):
    out, _ = suite
    n_checked = 0
    for line in (out / "manifest" / "records.jsonl").read_text().splitlines():
        rec = json.loads(line)
        rich = rec["context_layers"]["rich"]
        text = rich["text"]
        assert text, rec["id"]  # rich tier is non-empty
        plaintext = (out / rec["plaintext_file"]).read_text().strip()

        # full plaintext and the human key description must never appear in
        # context (a bare scalar key such as a rail count is not substring-
        # checked: a single digit is a meaningless coincidental match).
        assert plaintext not in text, rec["id"]
        assert rec["key_description"] not in text, rec["id"]
        assert rich["contains_solution"] is False, rec["id"]
        # keys live only under the firewalled ground_truth area
        assert rec["key_file"].startswith("ground_truth/"), rec["id"]
        n_checked += 1
    assert n_checked == len(ALL_FAMILIES)


def test_no_context_layer_leaks_plaintext(suite):
    out, _ = suite
    for line in (out / "manifest" / "records.jsonl").read_text().splitlines():
        rec = json.loads(line)
        plaintext = (out / rec["plaintext_file"]).read_text().strip()
        for tier, layer in rec["context_layers"].items():
            assert plaintext not in layer["text"], (rec["id"], tier)
            assert rec["key_description"] not in layer["text"], (rec["id"], tier)


# ---------------------------------------------------------------------------
# Context-tier gradation
# ---------------------------------------------------------------------------


def test_context_tier_gradation(suite):
    out, _ = suite
    for line in (out / "manifest" / "records.jsonl").read_text().splitlines():
        rec = json.loads(line)
        layers = rec["context_layers"]
        assert set(layers) == set(TIERS)

        none, lang, era, rich = (layers[t] for t in TIERS)
        # none -> empty, no hints of any kind
        assert none["text"] == ""
        assert none["contains_cipher_type_hint"] is False
        assert none["contains_plaintext_hint"] is False
        # language tier names the language but not the cipher family
        assert "English" in lang["text"]
        assert lang["contains_plaintext_hint"] is True
        assert lang["contains_cipher_type_hint"] is False
        # era_provenance adds era + provenance, still no cipher hint
        assert rec["era"] in era["text"]
        assert era["contains_cipher_type_hint"] is False
        # rich adds the cipher-family hint (monotonic increase in disclosure)
        assert rich["contains_cipher_type_hint"] is True
        assert rec["family_id"] in rich["text"]
        # strictly longer / more-informative as the tier rises
        assert len(none["text"]) < len(lang["text"]) < len(era["text"]) <= len(rich["text"])


def test_tier_tracks_distinct(suite):
    out, _ = suite
    loader = BenchmarkLoader(out)
    tests = loader.load_tests("all_generated.jsonl")
    tracks_by_tier = {}
    for t in tests:
        tier = t.test_id.rsplit("__", 1)[1]
        tracks_by_tier.setdefault(tier, set()).add(t.track)
    # each tier maps to exactly one, distinct track
    assert tracks_by_tier["none"] == {"transcription2plaintext"}
    assert tracks_by_tier["rich"] == {"transcription2plaintext_rich"}
    all_tracks = {track for s in tracks_by_tier.values() for track in s}
    assert len(all_tracks) == len(TIERS)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_byte_identical_manifest(tmp_path):
    kwargs = dict(
        families=["vigenere", "playfair", "columnar_transposition", "bifid", "hill2x2"],
        languages=["en", "de"],
        difficulties=["easy", "medium", "hard"],
        tiers=TIERS,
        per_family=2,
        seed=424242,
        dry_run=False,
    )
    a = tmp_path / "a"
    b = tmp_path / "b"
    gen.generate(out_dir=a, **kwargs)
    gen.generate(out_dir=b, **kwargs)

    manifest_a = (a / "manifest" / "records.jsonl").read_bytes()
    manifest_b = (b / "manifest" / "records.jsonl").read_bytes()
    assert manifest_a == manifest_b

    splits_a = (a / "splits" / "all_generated.jsonl").read_bytes()
    splits_b = (b / "splits" / "all_generated.jsonl").read_bytes()
    assert splits_a == splits_b


def test_different_seed_changes_manifest(tmp_path):
    kwargs = dict(
        families=["vigenere"],
        languages=["en"],
        difficulties=["medium"],
        tiers=["none"],
        per_family=2,
        dry_run=False,
    )
    a = tmp_path / "a"
    b = tmp_path / "b"
    gen.generate(out_dir=a, seed=1, **kwargs)
    gen.generate(out_dir=b, seed=2, **kwargs)
    assert (a / "manifest" / "records.jsonl").read_bytes() != (
        b / "manifest" / "records.jsonl"
    ).read_bytes()


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def test_dry_run_writes_no_files(tmp_path):
    out = tmp_path / "dry"
    result = gen.generate(
        families=["vigenere", "morse"],
        languages=["en"],
        difficulties=["medium"],
        tiers=TIERS,
        per_family=2,
        seed=7,
        out_dir=out,
        dry_run=True,
    )
    assert result.num_records == 4  # 2 families x 1 lang x 1 diff x 2 examples
    assert result.num_tests == 4 * len(TIERS)
    assert result.cases  # in-memory cases are populated
    # nothing on disk
    assert not out.exists() or not any(out.rglob("*"))


def test_cli_dry_run_returns_zero(capsys):
    rc = gen.main(
        ["--families", "vigenere", "--languages", "en", "--per-family", "1", "--dry-run"]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "DRY RUN" in captured


def test_cli_list_families(capsys):
    rc = gen.main(["--list-families"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "vigenere" in out
    assert "playfair" in out


# ---------------------------------------------------------------------------
# NEW-families deliverable
# ---------------------------------------------------------------------------


def test_new_families_are_the_expected_set():
    # Only playfair reuses an existing INV families.py id; everything else is
    # a NEW family not yet tested against INV/solver.
    news = set(new_family_ids())
    assert "playfair" not in news
    assert news == set(ALL_FAMILIES) - {"playfair"}
    assert len(news) == len(ALL_FAMILIES) - 1
