"""Tests for DP word segmentation."""
from __future__ import annotations

import pytest

from analysis.segment import (
    find_one_edit_corrections,
    repair_key_with_dictionary,
    repair_no_boundary_text,
    segment_text,
)


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


def test_repair_no_boundary_text_can_fix_one_edit_word():
    vocab = {"THERE", "CAT"}
    freq_rank = {"THERE": 1, "CAT": 2}

    result = repair_no_boundary_text("THERQ CAT", vocab, freq_rank=freq_rank)

    assert result.applied is True
    assert result.repaired_text == "THERE CAT"
    assert result.after.dict_rate == 1.0
    assert any(c["kind"] == "one_edit" for c in result.corrections)


def test_repair_no_boundary_text_can_resegment_local_window():
    vocab = {"THERE", "CAT"}
    freq_rank = {"THERE": 1, "CAT": 2}

    result = repair_no_boundary_text("THE RECAT", vocab, freq_rank=freq_rank)

    assert result.applied is True
    assert result.repaired_text == "THERE CAT"
    assert any(c["kind"] == "resegment" for c in result.corrections)


# ---------------------------------------------------------------------------
# Key-consistent repair tests
# ---------------------------------------------------------------------------


def _letter_maps():
    """Build id_to_letter / letter_to_id for a toy A-Z plaintext alphabet."""
    id_to_letter = {i: chr(ord("A") + i) for i in range(26)}
    letter_to_id = {v: k for k, v in id_to_letter.items()}
    return id_to_letter, letter_to_id


def test_repair_key_with_dictionary_single_symbol_fix():
    # Cipher symbols: 100, 101, 102, 103, 104.
    # Words: "CAT" = [100, 101, 102];  "DOG" = [103, 104, 102]
    # Ground truth: symbol 101 should map to 'A', symbol 104 to 'O'.
    # We start with 101 mapped to 'X' (so "CAT" decodes as "CXT") —
    # a single symbol remap to 'A' fixes CAT across every occurrence.
    id_to_letter, letter_to_id = _letter_maps()
    cipher_words = [[100, 101, 102], [103, 104, 102], [100, 101, 102]]
    # 100→C, 101→X(wrong, should be A), 102→T, 103→D, 104→O
    key = {
        100: letter_to_id["C"],
        101: letter_to_id["X"],
        102: letter_to_id["T"],
        103: letter_to_id["D"],
        104: letter_to_id["O"],
    }
    word_set = {"CAT", "DOG"}

    result = repair_key_with_dictionary(
        cipher_words=cipher_words,
        key=key,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        word_set=word_set,
    )

    assert result.applied is True
    # Every occurrence of symbol 101 is now 'A'.
    assert result.repaired_key[101] == letter_to_id["A"]
    assert result.after_hits > result.before_hits
    assert any(c["cipher_symbol"] == 101 for c in result.corrections)


def test_repair_key_with_dictionary_respects_protected_symbols():
    id_to_letter, letter_to_id = _letter_maps()
    cipher_words = [[100, 101, 102]]
    key = {
        100: letter_to_id["C"],
        101: letter_to_id["X"],  # would want to become 'A'
        102: letter_to_id["T"],
    }
    word_set = {"CAT"}

    result = repair_key_with_dictionary(
        cipher_words=cipher_words,
        key=key,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        word_set=word_set,
        protected_symbols={101},
    )

    # The only candidate fix touches a protected symbol; nothing should change.
    assert result.applied is False
    assert result.repaired_key[101] == letter_to_id["X"]


def test_repair_key_with_dictionary_rejects_score_regression():
    id_to_letter, letter_to_id = _letter_maps()
    cipher_words = [[100, 101, 102]]
    key = {
        100: letter_to_id["C"],
        101: letter_to_id["X"],
        102: letter_to_id["T"],
    }
    word_set = {"CAT"}

    # Pathological scorer that always reports a lower score after any change.
    def scorer(k):
        return 0.0 if k == key else -999.0

    result = repair_key_with_dictionary(
        cipher_words=cipher_words,
        key=key,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        word_set=word_set,
        score_fn=scorer,
        max_score_drop=0.0,
    )

    assert result.applied is False
    assert result.repaired_key == key


def test_repair_key_with_dictionary_multi_round_compounds_fixes():
    # Two distinct symbol errors; each round of repair should catch one.
    id_to_letter, letter_to_id = _letter_maps()
    # "CAT" = [100, 101, 102]; "DOG" = [103, 104, 105]
    cipher_words = [[100, 101, 102], [103, 104, 105]]
    key = {
        100: letter_to_id["C"],
        101: letter_to_id["X"],  # should be 'A'
        102: letter_to_id["T"],
        103: letter_to_id["D"],
        104: letter_to_id["Y"],  # should be 'O'
        105: letter_to_id["G"],
    }
    word_set = {"CAT", "DOG"}

    result = repair_key_with_dictionary(
        cipher_words=cipher_words,
        key=key,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        word_set=word_set,
    )

    assert result.applied is True
    assert result.repaired_key[101] == letter_to_id["A"]
    assert result.repaired_key[104] == letter_to_id["O"]
    assert result.after_hits == 2  # both CAT and DOG now hit
