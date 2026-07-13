"""Tests for src/analysis/multipage.py (shared-alphabet page-group helpers)."""
from __future__ import annotations

import inspect
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis import multipage
from analysis.multipage import (
    PageBundle,
    attach_page_scores,
    build_combined_cipher,
    consensus_from_finalists,
    page_runtime_metrics,
    project_pages,
    score_page_runtime,
)
from tests.test_ground_truth_firewall import GROUND_TRUTH, assert_no_ground_truth_leak


# ---------------------------------------------------------------------------
# Synthetic two-page, shared-alphabet fixture
# ---------------------------------------------------------------------------


class _FakeLoader:
    """Minimal stand-in for BenchmarkLoader: maps a test token to its data."""

    def __init__(self, data: dict[str, tuple[str, str]]) -> None:
        self._data = data

    def load_test_data(self, test):
        canonical, plaintext = self._data[test]
        return SimpleNamespace(canonical_transcription=canonical, plaintext=plaintext)


def _two_page_fixture():
    # Page A and Page B share symbols S002 and S003; A also has S001, B also S004.
    data = {
        "pgA": ("S001 S002 S003 S001", "AB PLAINTEXT A"),
        "pgB": ("S002 S003 S004", "CD PLAINTEXT B"),
    }
    tests = {"pgA": "pgA", "pgB": "pgB"}
    loader = _FakeLoader(data)
    combined, pages = build_combined_cipher(loader, tests, ["pgA", "pgB"])
    return combined, pages


def test_build_combined_cipher_offsets_and_provenance():
    combined, pages = _two_page_fixture()

    # Shared alphabet is the union of page symbols in first-seen order.
    assert list(combined.alphabet.symbols) == ["S001", "S002", "S003", "S004"]
    # Pages are joined with the word separator; provenance/source is generic.
    assert combined.raw == "S001 S002 S003 S001 | S002 S003 S004"
    assert combined.source == "multipage"
    assert combined.separator == " | "

    a, b = pages
    assert a.test_id == "pgA" and b.test_id == "pgB"
    assert a.symbols == ["S001", "S002", "S003", "S001"]
    assert b.symbols == ["S002", "S003", "S004"]
    # token_ids index into the shared alphabet (S002/S003 are shared ids 1/2).
    assert a.token_ids == [0, 1, 2, 0]
    assert b.token_ids == [1, 2, 3]


def test_project_pages_inverts_build_combined_cipher():
    combined, pages = _two_page_fixture()
    alphabet = combined.alphabet
    # A key that maps every used token id to a distinct letter is invertible.
    key = {token_id: token_id for token_id in range(alphabet.size)}  # 0->A,1->B,...
    rows = project_pages(pages=pages, key=key, mask=())

    assert [row["decryption"] for row in rows] == ["ABCA", "BCD"]

    # Round-trip identity: map each projected letter back through the inverse
    # key to a token id, then to a symbol, and recover the page's symbols.
    inverse = {letter_id: token_id for token_id, letter_id in key.items()}
    for page, row in zip(pages, rows):
        recovered = [
            alphabet.symbol_for(inverse[ord(ch) - ord("A")])
            for ch in row["decryption"]
        ]
        assert recovered == page.symbols
        assert row["filtered_length"] == len(page.symbols)


# ---------------------------------------------------------------------------
# consensus_from_finalists
# ---------------------------------------------------------------------------


def test_consensus_from_finalists_agreement_and_stability():
    from models.alphabet import Alphabet

    alphabet = Alphabet(["S001", "S002", "S003"])
    artifact = {
        "steps": [
            {
                "name": "search_null_masks",
                "selected": {"key": {"0": 0, "1": 1, "2": 2}, "mask": []},
                "top_finalists": [
                    {"key": {"0": 0, "1": 1, "2": 2}, "mask": []},
                    {"key": {"0": 0, "1": 3, "2": 2}, "mask": []},
                ],
            }
        ]
    }
    consensus = consensus_from_finalists(
        artifact=artifact, alphabet=alphabet, top_n=10, min_agreement=0.75
    )

    # 3 rows total (selected + 2 finalists).
    assert consensus["S001"]["winner"] == "A"
    assert consensus["S001"]["agreement"] == 1.0
    assert consensus["S001"]["stable"] is True

    # S002 is split 2x B, 1x D -> 0.667 agreement, below the 0.75 threshold.
    assert consensus["S002"]["winner"] == "B"
    assert round(consensus["S002"]["agreement"], 2) == 0.67
    assert consensus["S002"]["stable"] is False
    assert consensus["S002"]["counts"] == {"B": 2, "D": 1}

    assert consensus["S003"]["winner"] == "C"
    assert consensus["S003"]["stable"] is True


# ---------------------------------------------------------------------------
# Ground-truth firewall: runtime scoring never sees plaintext
# ---------------------------------------------------------------------------


def test_score_page_runtime_and_metrics_are_ground_truth_free():
    # A page whose *plaintext* is the distinctive ground truth, but whose
    # projected decryption is unrelated English text.
    symbols = [f"S{i:03d}" for i in range(21)]
    from models.alphabet import Alphabet

    alphabet = Alphabet(symbols)
    decryption_letters = "THEHOUSEGREENWORLD" + "XYZ"
    key = {alphabet.id_for(sym): (ord(ch) - ord("A")) for sym, ch in zip(symbols, decryption_letters)}
    page = PageBundle(
        test_id="firewall_pg",
        canonical_transcription=" ".join(symbols),
        plaintext=GROUND_TRUTH,  # solve-side input carries the ground truth
        symbols=symbols,
        token_ids=[alphabet.id_for(sym) for sym in symbols],
    )
    # project_pages copies plaintext into the row; runtime scoring must ignore it.
    row = project_pages(pages=[page], key=key, mask=())[0]
    assert row["decryption"] == decryption_letters
    assert row["plaintext"] == GROUND_TRUTH

    runtime = score_page_runtime(row, key=key, mask=(), language="en")
    metrics = page_runtime_metrics([runtime])

    assert_no_ground_truth_leak(
        [json.dumps(runtime), json.dumps(metrics)], GROUND_TRUTH
    )
    # The runtime payload also does not carry a plaintext field at all.
    assert "plaintext" not in runtime


# ---------------------------------------------------------------------------
# Post-hoc calibration is segregated and graded correctly
# ---------------------------------------------------------------------------


def test_attach_page_scores_grades_against_ground_truth():
    rows = [
        {"test_id": "t1", "decryption": "HELLOWORLD", "plaintext": "HELLOWORLD"},
        {"test_id": "t2", "decryption": "HELLOWORLX", "plaintext": "HELLOWORLD"},
    ]
    attach_page_scores(rows)
    # Exact match grades to full char accuracy; a single wrong char is lower.
    assert rows[0]["char_accuracy"] == 1.0
    assert rows[1]["char_accuracy"] < 1.0
    assert rows[0]["char_accuracy"] > rows[1]["char_accuracy"]
    assert rows[0]["preview"] == "HELLOWORLD"


def test_runtime_functions_do_not_reference_calibration():
    """Firewall audit: no runtime function references the calibration helper
    or the ground-truth scorer it uses."""
    runtime_functions = [
        multipage.build_combined_cipher,
        multipage.project_pages,
        multipage.project_page_with_sources,
        multipage.consensus_from_finalists,
        multipage.consensus_summary,
        multipage.score_page_runtime,
        multipage.page_runtime_metrics,
    ]
    for func in runtime_functions:
        src = inspect.getsource(func)
        assert "attach_page_scores" not in src, func.__name__
        assert "score_decryption" not in src, func.__name__
