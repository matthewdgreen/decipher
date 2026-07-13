"""Tests for src/analysis/word_hypothesis_repair.py (word-repair pipeline)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import Counter

from analysis.multipage import PageBundle
from analysis.word_hypothesis_repair import (
    WordRepairConfig,
    adjudication_score,
    build_page_windows,
    generate_word_hypotheses,
    load_dictionary,
    propose_word_repairs,
)
from models.alphabet import Alphabet
from tests.test_ground_truth_firewall import GROUND_TRUTH, assert_no_ground_truth_leak

DICTIONARY = os.path.join(
    os.path.dirname(__file__), "..", "resources", "dictionaries", "english_common.txt"
)


# ---------------------------------------------------------------------------
# Constructed-damage fixture
#
# Two pages projected (under the shared baseline key) to:
#   page1: THE HXUSE GRAEN WORLD   (18 chars)
#   page2: THE BRAND HOUSE         (13 chars)
# A single shared symbol S_a decodes to 'X' at page1's HXUSE span (nowhere
# else); repairing it (X->O) yields the dictionary word HOUSE with no
# collateral. A second shared symbol S_b decodes to 'A' both inside page1's
# GRAEN span AND inside page2's good word BRAND; the decoy repair (A->E) yields
# GREEN on page1 but corrupts BRAND->BREND on page2 (collateral damage).
# ---------------------------------------------------------------------------


def _build_fixture(page1_plaintext: str = "PG1", page2_plaintext: str = "PG2"):
    page1_text = "THEHXUSEGRAENWORLD"
    page2_text = "THEBRANDHOUSE"

    counter = [0]

    def fresh() -> str:
        counter[0] += 1
        return f"S{counter[0]:03d}"

    sym_a = fresh()  # the 'X' on page1 (unique)
    sym_b = fresh()  # the shared 'A' (page1 GRAEN + page2 BRAND)

    def build(text: str, specials: dict[int, str]) -> list[str]:
        return [specials.get(i) or fresh() for i, _ in enumerate(text)]

    p1_syms = build(page1_text, {4: sym_a, 10: sym_b})
    p2_syms = build(page2_text, {5: sym_b})

    all_syms: list[str] = []
    seen: set[str] = set()
    for sym in p1_syms + p2_syms:
        if sym not in seen:
            seen.add(sym)
            all_syms.append(sym)
    alphabet = Alphabet(all_syms)

    shared_key: dict[int, int] = {}
    for syms, text in ((p1_syms, page1_text), (p2_syms, page2_text)):
        for sym, ch in zip(syms, text):
            shared_key[alphabet.id_for(sym)] = ord(ch) - ord("A")

    pages = [
        PageBundle(
            test_id="pg1",
            canonical_transcription=" ".join(p1_syms),
            plaintext=page1_plaintext,
            symbols=p1_syms,
            token_ids=[alphabet.id_for(sym) for sym in p1_syms],
        ),
        PageBundle(
            test_id="pg2",
            canonical_transcription=" ".join(p2_syms),
            plaintext=page2_plaintext,
            symbols=p2_syms,
            token_ids=[alphabet.id_for(sym) for sym in p2_syms],
        ),
    ]
    return pages, shared_key, alphabet, sym_a, sym_b


def _small_config(**overrides) -> WordRepairConfig:
    base = dict(
        window_size=40,
        window_step=40,
        windows_per_page=3,
        min_word_len=5,
        max_word_len=5,
        max_edits=2,
        max_hypotheses=40,
        max_hypotheses_per_window=10,
    )
    base.update(overrides)
    return WordRepairConfig(**base)


def _find(packets, edits):
    return [p for p in packets if p.provenance.get("edits") == edits]


def test_constructed_damage_true_repair_accepted_decoy_rejected():
    pages, shared_key, alphabet, sym_a, sym_b = _build_fixture()
    packets = propose_word_repairs(
        pages=pages,
        shared_key=shared_key,
        dictionary_path=DICTIONARY,
        language="en",
        config=_small_config(),
        alphabet=alphabet,
    )
    assert packets, "pipeline produced no candidates"

    # --- The true repair is proposed, with the correct edit set. ---
    true_packets = _find(packets, [f"{sym_a}:X->O"])
    assert true_packets, "true repair HXUSE->HOUSE was not proposed"
    true = true_packets[0]
    assert true.kind == "word_repair"
    assert true.text is None  # F3 deferral: no full decryption in packet
    true_collateral = true.provenance["collateral_evidence"]
    # S_a occurs only in the target span -> zero collateral.
    assert true_collateral["collateral_occurrences"] == 0
    assert true.validation["accepted"] is True

    # --- The decoy is proposed but rejected because it damages another page. ---
    decoy_packets = _find(packets, [f"{sym_b}:A->E"])
    assert decoy_packets, "decoy repair GRAEN->GREEN was not proposed"
    decoy = decoy_packets[0]
    decoy_collateral = decoy.provenance["collateral_evidence"]
    # S_b also occurs inside page2's BRAND -> at least one collateral occurrence.
    assert decoy_collateral["collateral_occurrences"] >= 1
    assert decoy.validation["accepted"] is False
    # The collateral occurrence is on the *other* page (pg2).
    evidence_pages = {
        str(row.get("test_id"))
        for row in decoy.provenance.get("page_runtime_evidence") or []
    }
    assert "pg2" in evidence_pages


def test_config_bounds_are_respected():
    pages, shared_key, alphabet, _sym_a, _sym_b = _build_fixture()
    baseline_mask: tuple[str, ...] = ()
    dictionary = load_dictionary(DICTIONARY, 5, 5)
    page_windows = build_page_windows(
        pages=pages,
        alphabet=alphabet,
        key=shared_key,
        mask=baseline_mask,
        consensus={},
        window_size=40,
        window_step=40,
        windows_per_page=3,
        language="en",
    )

    def _generate(max_per_window: int):
        return generate_word_hypotheses(
            page_windows=page_windows,
            dictionary=dictionary,
            consensus={},
            alphabet=alphabet,
            baseline_key=shared_key,
            baseline_mask=baseline_mask,
            min_word_len=5,
            max_word_len=5,
            max_edits=2,
            max_per_window=max_per_window,
            allow_stable_edits=False,
        )

    # First prove the cap actually bites: with a huge per-window budget, every
    # window in this fixture produces far more than 3 candidates.
    uncapped = _generate(1000)
    uncapped_counts = Counter((h.test_id, h.window_start) for h in uncapped)
    assert len(uncapped_counts) >= 2  # both pages contribute a window
    assert min(uncapped_counts.values()) > 3, uncapped_counts

    # Per-window cap: no (test_id, window_start) may exceed max_per_window.
    # (Dedupe-by-edits runs after the slice, so counts may shrink, never grow.)
    capped = _generate(3)
    capped_counts = Counter((h.test_id, h.window_start) for h in capped)
    assert capped_counts, "capped generation produced no hypotheses"
    assert all(count <= 3 for count in capped_counts.values()), capped_counts
    assert len(capped) < len(uncapped)

    # Global cap: uncapped generation yields far more than 5 hypotheses, so
    # max_hypotheses=5 is a real truncation; the singleton packet count is then
    # bounded by it too.
    assert len(uncapped) > 5
    packets = propose_word_repairs(
        pages=pages,
        shared_key=shared_key,
        dictionary_path=DICTIONARY,
        language="en",
        config=_small_config(
            max_hypotheses=5,
            max_hypothesis_set_size=1,
            max_hypotheses_per_window=1000,
        ),
        alphabet=alphabet,
    )
    assert 0 < len(packets) <= 5


def test_adjudication_score_orientation():
    """Direct sign/weight orientation check on hand-built evidence dicts.

    An extraction that inverted the scorer's sign or flipped gain/damage
    weights would fail this test. (The end-to-end fixture cannot assert
    true > decoy on adjudication_score: the decoy earns real target-word gain
    there and is rejected by the page-group acceptance gate instead.)
    """
    neutral: dict = {}
    clean_gain = {
        "target_gain_avg": 0.2,
        "target_word_gain_sum": 5.0,
        "target_word_evidence_gain_avg": 0.5,
        "target_occurrences": 2,
        "occurrence_count": 2,
    }
    damaging = {
        "collateral_damage_avg": 0.2,
        "collateral_word_damage_weighted_avg": 0.5,
        "damaged_occurrences": 3,
        "word_damaged_weighted_occurrences": 2,
        "collateral_occurrences": 3,
        "occurrence_count": 3,
    }
    hypothesis_score = 1.0
    score_clean = adjudication_score(clean_gain, hypothesis_score)
    score_neutral = adjudication_score(neutral, hypothesis_score)
    score_damaging = adjudication_score(damaging, hypothesis_score)

    # Orientation: target gain raises the score, collateral damage lowers it.
    assert score_clean > score_neutral > score_damaging
    assert score_damaging < 0.0 < score_clean

    # Monotonicity in each direction.
    more_gain = dict(clean_gain, target_gain_avg=0.4)
    assert adjudication_score(more_gain, hypothesis_score) > score_clean
    more_damage = dict(damaging, collateral_damage_avg=0.4)
    assert adjudication_score(more_damage, hypothesis_score) < score_damaging

    # The hypothesis (local word) score contributes positively too.
    assert adjudication_score(clean_gain, 2.0) > adjudication_score(clean_gain, 1.0)


def test_propose_word_repairs_is_ground_truth_free():
    pages, shared_key, alphabet, _sym_a, _sym_b = _build_fixture(
        page1_plaintext=GROUND_TRUTH, page2_plaintext=GROUND_TRUTH
    )
    packets = propose_word_repairs(
        pages=pages,
        shared_key=shared_key,
        dictionary_path=DICTIONARY,
        language="en",
        config=_small_config(),
        alphabet=alphabet,
    )
    assert packets
    blob = json.dumps([p.to_dict() for p in packets])
    assert_no_ground_truth_leak([blob], GROUND_TRUTH)
    # No packet carries a full decryption (previews only).
    assert all(p.text is None for p in packets)
