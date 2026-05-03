from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.finalist_validation import validate_plaintext_finalist, validation_adjustment


WORD_LIST = [
    "THE",
    "QUICK",
    "BROWN",
    "FOX",
    "JUMPS",
    "OVER",
    "LAZY",
    "DOG",
    "RIVER",
    "STONE",
    "FIELD",
    "MOON",
    "RISE",
    "WATCHING",
]
WORD_SET = set(WORD_LIST)


def test_finalist_validation_marks_continuous_plaintext_as_promising():
    text = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWATCHINGTHEMOONRISE"

    result = validate_plaintext_finalist(
        text,
        language="en",
        word_set=WORD_SET,
        word_list=WORD_LIST,
    )

    assert result["validation_label"] in {"coherent_candidate", "plausible_candidate"}
    assert result["recommendation"] in {"install_or_promote", "inspect_contextually"}
    assert result["strict_word_hit_score"] > 0.5
    assert result["segmentation"]["dict_rate"] > 0.85
    assert result["integrity"]["integrity_label"] in {
        "clean_or_canonical",
        "minor_local_damage",
    }
    assert result["integrity"]["damage_score"] < 0.2
    assert validation_adjustment(result) > 0


def test_finalist_validation_flags_word_island_basin_as_weak():
    text = "THEQZXBROWNMMFIELDRRMOONQJXSTONEPPDOG"

    result = validate_plaintext_finalist(
        text,
        language="en",
        word_set=WORD_SET,
        word_list=WORD_LIST,
    )

    assert result["validation_label"] in {"weak_word_islands", "gibberish_or_wrong_family"}
    assert result["recommendation"] in {
        "review_more_finalists_or_broaden_search",
        "reject_or_switch_family",
    }
    assert result["segmentation"]["pseudo_word_fraction"] > 0.4
    assert result["integrity"]["integrity_label"] in {
        "readable_with_local_scars",
        "damaged_word_island_basin",
        "possible_wrong_family_or_large_order_error",
    }
    assert result["integrity"]["suspicious_short_pseudo_count"] >= 4


def test_finalist_validation_distinguishes_clean_from_scarred_readable_text():
    clean = "THEQUIETRIVERCARRIEDSMALLBOATSPASTTHEOLDMILL"
    scarred = "THEQUIETRIVERCARRIEDSMALLBOATSPASTTHOLDEMILL"

    clean_result = validate_plaintext_finalist(
        clean,
        language="en",
        word_set=WORD_SET | {"QUIET", "CARRIED", "SMALL", "BOATS", "PAST", "OLD", "MILL"},
        word_list=WORD_LIST + ["QUIET", "CARRIED", "SMALL", "BOATS", "PAST", "OLD", "MILL"],
    )
    scarred_result = validate_plaintext_finalist(
        scarred,
        language="en",
        word_set=WORD_SET | {"QUIET", "CARRIED", "SMALL", "BOATS", "PAST", "OLD", "MILL"},
        word_list=WORD_LIST + ["QUIET", "CARRIED", "SMALL", "BOATS", "PAST", "OLD", "MILL"],
    )

    assert clean_result["validation_label"] == "coherent_candidate"
    assert scarred_result["validation_label"] == "coherent_candidate"
    assert clean_result["integrity"]["damage_score"] < scarred_result["integrity"]["damage_score"]
    assert clean_result["integrity"]["integrity_label"] == "clean_or_canonical"
    assert scarred_result["integrity"]["integrity_label"] != "clean_or_canonical"
