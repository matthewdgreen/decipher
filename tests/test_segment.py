"""Tests for DP word segmentation."""
from __future__ import annotations

import pytest

from analysis.segment import find_one_edit_corrections, segment_text


# A small, plausibly-English fixture vocabulary.
VOCAB = {
    "THE", "QUICK", "BROWN", "FOX", "JUMPS", "OVER", "LAZY", "DOG",
    "HELLO", "THERE", "WORLD", "WINDOWS", "MARIA", "DECIDED", "MORNING",
    "AND", "IS", "IT", "A", "OF", "TO",
}


def test_segments_simple_run():
    result = segment_text("HELLOTHERE", VOCAB)
    assert result.words == ["HELLO", "THERE"]
    assert result.dict_rate == 1.0
    assert result.pseudo_words == []


def test_segments_longer_phrase():
    text = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
    result = segment_text(text, VOCAB)
    assert result.words == [
        "THE", "QUICK", "BROWN", "FOX", "JUMPS", "OVER", "THE", "LAZY", "DOG"
    ]
    assert result.dict_rate == 1.0


def test_detects_pseudo_words():
    # Embed the MARPA/WPNDOWS pattern — single-letter errors.
    text = "HELLOMARPAWPNDOWS"
    result = segment_text(text, VOCAB)
    # MARPA and WPNDOWS aren't in the dict; HELLO is.
    assert "HELLO" in result.words
    assert any("MARPA" in w for w in result.pseudo_words) or any(
        "WPNDOWS" in w for w in result.pseudo_words
    )


def test_empty_text():
    result = segment_text("", VOCAB)
    assert result.words == []
    assert result.dict_rate == 0.0


def test_dict_rate_on_all_unknown():
    result = segment_text("XYZQRSTVW", VOCAB)
    # Nothing recognisable; dict_rate is 0.
    assert result.dict_rate == 0.0
    assert len(result.pseudo_words) >= 1


def test_respects_existing_whitespace():
    # Chunks separated by spaces are segmented independently.
    result = segment_text("HELLOTHERE WORLD", VOCAB)
    assert "HELLO" in result.words
    assert "THERE" in result.words
    assert "WORLD" in result.words


def test_one_edit_corrections_finds_substitution():
    corrections = find_one_edit_corrections("MARPA", {"MARIA"})
    assert ("MARIA", "P", "I") in corrections


def test_one_edit_corrections_groups_by_position():
    # WPNDOWS -> WINDOWS is a single P→I at index 1
    corrections = find_one_edit_corrections("WPNDOWS", {"WINDOWS"})
    assert ("WINDOWS", "P", "I") in corrections


def test_one_edit_corrections_empty_when_unreachable():
    corrections = find_one_edit_corrections("ABCDE", {"ZZZZZ"})
    assert corrections == []


def test_freq_rank_tiebreaks_segmentations():
    # Both "AND" and "A" + "ND" could compete; with a freq_rank favouring AND,
    # the segmenter should prefer the single word.
    vocab = {"AND", "A", "ND"}
    freq_rank = {"AND": 1, "A": 2, "ND": 500}
    result = segment_text("AND", vocab, freq_rank=freq_rank)
    assert result.words == ["AND"]
