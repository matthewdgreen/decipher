"""Tests for the measured plaintext corpus library (Slice B).

Selection / filter / determinism / empty-filter tests are driven off a tiny
hand-written FIXTURE library (tests/fixtures/plaintext_library/) so they do not
depend on the multi-GB corpora. A separate cleaning test proves no Gutenberg
licence text leaks into a cleaned passage. One env-gated slow test exercises the
real builder over a tiny corpus sample.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from testgen.corpus_library import (
    ERA_CLASSICAL,
    ERA_HISTORICAL,
    ERA_LITERARY_19C,
    ERA_MODERN,
    MEASURED_FIELDS,
    LibraryEmpty,
    PassageRecord,
    clean_to_words,
    contains_boilerplate,
    era_for,
    filter_records,
    load_library,
    measure_passage,
    sample_passages,
    select,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "plaintext_library"
DIRTY_SAMPLE = Path(__file__).resolve().parent / "fixtures" / "dirty_gutenberg_sample.txt"


# --------------------------------------------------------------------------
# Loading + schema
# --------------------------------------------------------------------------


def test_load_fixture_library_en():
    recs = load_library("en", library_dir=FIXTURE_DIR)
    assert len(recs) == 6
    assert all(isinstance(r, PassageRecord) for r in recs)
    assert all(r.language == "en" for r in recs)
    # Every record carries the full measured battery.
    for r in recs:
        assert set(r.measured.keys()) == set(MEASURED_FIELDS)
        assert r.length_words == len(r.text.split())


def test_load_missing_library_returns_empty():
    assert load_library("xx", library_dir=FIXTURE_DIR) == []


def test_record_dict_round_trip():
    recs = load_library("en", library_dir=FIXTURE_DIR)
    rec = recs[0]
    assert PassageRecord.from_dict(rec.to_dict()) == rec


def test_fixture_spread_is_diverse():
    recs = load_library("en", library_dir=FIXTURE_DIR)
    eras = {r.era for r in recs}
    styles = {r.frequency_style for r in recs}
    topics = {r.topic for r in recs}
    assert eras == {ERA_LITERARY_19C, ERA_MODERN}
    assert styles == {"normal", "unusual"}
    assert {"general", "fiction", "letters"} <= topics


# --------------------------------------------------------------------------
# Filtering + selection
# --------------------------------------------------------------------------


def test_filter_by_era():
    recs = load_library("en", library_dir=FIXTURE_DIR)
    modern = filter_records(recs, era=ERA_MODERN)
    assert len(modern) == 3
    assert all(r.era == ERA_MODERN for r in modern)


def test_filter_by_length_window():
    recs = load_library("en", library_dir=FIXTURE_DIR)
    mid = filter_records(recs, min_words=100, max_words=200)
    assert mid, "expected some mid-length passages"
    assert all(100 <= r.length_words <= 200 for r in mid)


def test_filter_by_frequency_style_and_topic():
    recs = load_library("en", library_dir=FIXTURE_DIR)
    unusual = filter_records(recs, frequency_style="unusual")
    assert unusual and all(r.frequency_style == "unusual" for r in unusual)
    fiction = filter_records(recs, topic="fiction")
    assert fiction and all(r.topic == "fiction" for r in fiction)


def test_select_is_deterministic_given_rng():
    recs = load_library("en", library_dir=FIXTURE_DIR)
    a = select(records=recs, era=ERA_MODERN, rng=random.Random(1234))
    b = select(records=recs, era=ERA_MODERN, rng=random.Random(1234))
    assert a == b


def test_select_is_genuinely_rng_driven():
    # era=modern has 3 candidates; over many seeds we must see >1 distinct pick,
    # proving selection is rng-driven rather than "always first".
    recs = load_library("en", library_dir=FIXTURE_DIR)
    picks = {select(records=recs, era=ERA_MODERN, rng=random.Random(s)).id for s in range(50)}
    assert len(picks) > 1


def test_select_single_candidate_is_stable():
    recs = load_library("en", library_dir=FIXTURE_DIR)
    # Only one 'letters' passage exists — any rng returns it.
    for s in (0, 7, 99):
        rec = select(records=recs, topic="letters", rng=random.Random(s))
        assert rec.topic == "letters"


def test_select_by_language_loads_from_dir():
    rec = select(language="en", rng=random.Random(0), library_dir=FIXTURE_DIR)
    assert rec.language == "en"


def test_select_language_filter_on_records():
    en = load_library("en", library_dir=FIXTURE_DIR)
    de = load_library("de", library_dir=FIXTURE_DIR)
    combined = en + de
    rec = select(records=combined, language="de", rng=random.Random(3))
    assert rec.language == "de"


# --------------------------------------------------------------------------
# Empty-filter errors name the unmet filter
# --------------------------------------------------------------------------


def test_empty_filter_raises_named_error():
    recs = load_library("en", library_dir=FIXTURE_DIR)
    with pytest.raises(LibraryEmpty) as exc:
        select(records=recs, era=ERA_CLASSICAL, rng=random.Random(0))
    assert "classical" in str(exc.value)


def test_empty_combined_filter_names_filters():
    recs = load_library("en", library_dir=FIXTURE_DIR)
    with pytest.raises(LibraryEmpty) as exc:
        select(records=recs, era=ERA_MODERN, topic="letters",
               frequency_style="unusual", rng=random.Random(0))
    msg = str(exc.value)
    assert "topic='letters'" in msg
    assert "frequency_style='unusual'" in msg


def test_select_by_missing_language_raises():
    with pytest.raises(LibraryEmpty):
        select(language="xx", rng=random.Random(0), library_dir=FIXTURE_DIR)


def test_select_requires_records_or_language():
    with pytest.raises(ValueError):
        select(rng=random.Random(0))


# --------------------------------------------------------------------------
# Cleaning: no Gutenberg / licence leakage
# --------------------------------------------------------------------------

_FORBIDDEN = (
    "PROJECT GUTENBERG",
    "GUTENBERG",
    "COPYRIGHT",
    "LICENSE",
    "LICENCE",
    "PROOFREAD",
    "ILLUSTRATION",
    "ISBN",
    "EBOOK",
    "DISTRIBUTE",
    "TRADEMARK",
)


def test_cleaning_strips_gutenberg_boilerplate():
    raw = DIRTY_SAMPLE.read_text(encoding="utf-8")
    words = clean_to_words(raw)
    joined = " ".join(words)
    assert words, "expected a non-empty cleaned body"
    for term in _FORBIDDEN:
        assert term not in joined, f"licence/boilerplate term leaked: {term}"
    # Body should begin at the actual chapter text.
    assert joined.startswith("CHAPTER ONE THE MORNING BROKE")


def test_sampled_passages_are_boilerplate_free():
    raw = DIRTY_SAMPLE.read_text(encoding="utf-8")
    words = clean_to_words(raw)
    passages = sample_passages(words, lengths=[20, 40], per_length=3, rng=random.Random(7))
    assert passages
    for p in passages:
        assert not contains_boilerplate(p.text)
        for term in _FORBIDDEN:
            assert term not in p.text


def test_clean_words_are_uppercase_alpha():
    raw = DIRTY_SAMPLE.read_text(encoding="utf-8")
    for w in clean_to_words(raw):
        assert w.isalpha() and w.isupper() and len(w) >= 2


# --------------------------------------------------------------------------
# Measurement + tagging units
# --------------------------------------------------------------------------


def test_measure_passage_english_ranges():
    text = (
        "THE MORNING BROKE CLEAR AND COLD OVER THE LITTLE HARBOUR TOWN AND THE "
        "FISHERMEN WENT DOWN TO THEIR BOATS BEFORE THE SUN HAD FULLY RISEN GREY "
        "GULLS WHEELED ABOVE THE GREY WATER AND THE SMELL OF SALT AND TAR HUNG "
        "THICK IN THE NARROW STREETS"
    )
    m = measure_passage(text, "en")
    assert set(m.keys()) == set(MEASURED_FIELDS)
    assert all(isinstance(v, float) for v in m.values())
    # English prose: IoC well above random (0.038), dict-rate high.
    assert 0.05 <= m["index_of_coincidence"] <= 0.09
    assert m["dict_rate"] >= 0.7
    assert 3.0 <= m["mean_word_length"] <= 6.0
    assert 0.0 < m["unique_symbol_ratio"] <= 1.0


def test_measure_passage_empty_is_safe():
    m = measure_passage("", "en")
    assert set(m.keys()) == set(MEASURED_FIELDS)
    assert m["index_of_coincidence"] == 0.0
    assert m["dict_rate"] == 0.0


def test_era_for_tagging():
    assert era_for("dta", "de") == ERA_HISTORICAL
    assert era_for("gutenberg", "la") == ERA_CLASSICAL      # Latin overrides
    assert era_for("gutenberg", "de") == ERA_LITERARY_19C
    assert era_for("gutenberg", "en") == ERA_LITERARY_19C
    assert era_for("oanc", "en") == ERA_MODERN
    assert era_for("masc", "en") == ERA_MODERN


# --------------------------------------------------------------------------
# Slow / opt-in: exercise the real builder on a tiny corpus sample.
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("DECIPHER_RUN_CORPUS_BUILD") != "1",
    reason="set DECIPHER_RUN_CORPUS_BUILD=1 to run the real corpus builder",
)
def test_real_builder_small_sample(tmp_path):
    import importlib.util

    repo_root = Path(__file__).resolve().parent.parent
    corpus_root = repo_root / "corpus_data"
    if not corpus_root.exists():
        # Corpora are git-ignored; may live in the main checkout.
        corpus_root = Path.home() / "Dropbox" / "src2" / "decipher" / "corpus_data"
    if not corpus_root.exists():
        pytest.skip("no corpus_data available")

    import sys

    script = repo_root / "scripts" / "build_plaintext_library.py"
    spec = importlib.util.spec_from_file_location("build_plib", script)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's dataclass can resolve annotations.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    rc = mod.main([
        "--languages", "en",
        "--files-per-source", "1",
        "--sources", "gutenberg",
        "--corpus-root", str(corpus_root),
        "--out", str(tmp_path),
    ])
    assert rc == 0
    recs = load_library("en", library_dir=tmp_path)
    assert recs, "builder produced no passages"
    for r in recs:
        joined = r.text.upper()
        assert "PROJECT GUTENBERG" not in joined
        assert "GUTENBERG" not in joined
        assert set(r.measured.keys()) == set(MEASURED_FIELDS)
