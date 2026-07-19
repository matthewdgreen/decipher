"""Tests for the keyword-columnar search module (composite Slice C.1, PART 1).

This is the coordination-critical module (spec §1.1 item 3 / §7a / §4.1): the
geometric ``pure_transposition`` screen has no keyed-columnar coverage and fails
blind on classic keyed columnar (matrix finding F2, ~0.10 char). The direct test
below is the F2 closer — a blind keyed columnar over the round-4-class widths
(5-8) must be recovered at 100% by the language-score plugin.

Firewall: every scorer here is a shipped statistic/model — no ground truth ever
enters the search.
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.columnar_search import (
    ColumnarSearchConfig,
    columnar_decode_by_order,
    keyword_for_order,
    make_language_scorer,
    make_structure_ratio_scorer,
    search_keyed_columnar,
)
from ciphers.transposition import _rank_order, columnar_decrypt, columnar_encrypt

# A natural-English plaintext long enough for the language scorer to separate the
# correct un-transposition from every wrong permutation, and whose length avoids
# the rotational near-ties that a columnar grid can create at some (width, length)
# combinations (a cyclic rotation of English is itself fluent).
PLAINTEXT = (
    "THELAWSOFMATHEMATICSAREVERYCOMMENDABLEBUTTHEONLYLAWTHATAPPLIESINMY"
    "AREAISTHATOFSUPPLYANDDEMANDNEVERTHELESSWEMUSTCONTINUETOEXPLORETHEDEEPER"
    "STRUCTUREOFNUMBERSANDTHEIRHIDDENPATTERNS"
)


class TestColumnarDecodeConvention:
    """The decode-by-order helper must exactly invert columnar_encrypt."""

    def test_decode_by_order_matches_library_decrypt(self):
        for key in ("MASON", "SECRET", "MASONRY", "PASSWORD", "CRYPTOGRAM"):
            ct = columnar_encrypt(PLAINTEXT, key)
            by_order = columnar_decode_by_order(ct, len(key), _rank_order(key))
            assert by_order == columnar_decrypt(ct, key) == PLAINTEXT

    def test_keyword_for_order_reproduces_read_order(self):
        for key in ("MASON", "SECRET", "MASONRY", "PASSWORD"):
            order = _rank_order(key)
            recovered_kw = keyword_for_order(order)
            # The synthetic keyword must sort to the same read order.
            assert _rank_order(recovered_kw) == order

    def test_decode_rejects_non_permutation(self):
        import pytest

        with pytest.raises(ValueError):
            columnar_decode_by_order("ABCDEFGH", 4, [0, 1, 1, 3])


class TestBlindKeyedColumnarRecovery:
    """The F2 closer: blind keyed columnar over widths 5-8 -> 100% recovery."""

    def test_language_plugin_recovers_widths_5_to_8(self):
        scorer = make_language_scorer("en")
        cases = {
            5: "MASON",
            6: "SECRET",
            7: "MASONRY",
            8: "PASSWORD",
        }
        for width, key in cases.items():
            ct = columnar_encrypt(PLAINTEXT, key)
            finalists = search_keyed_columnar(
                ct, scorer, min_width=5, max_width=8, top_n=3
            )
            assert finalists, f"no finalist for width {width}"
            best = finalists[0]
            assert best.column_count == width
            assert best.column_order == tuple(_rank_order(key))
            assert best.decoded_stream == PLAINTEXT
            # The reported keyword hint round-trips to the same read order.
            assert _rank_order(best.keyword) == list(best.column_order)
            assert best.method == "exhaustive"

    def test_returns_ranked_finalists_descending(self):
        scorer = make_language_scorer("en")
        ct = columnar_encrypt(PLAINTEXT, "MASONRY")
        finalists = search_keyed_columnar(ct, scorer, min_width=5, max_width=8, top_n=4)
        scores = [f.score for f in finalists]
        assert scores == sorted(scores, reverse=True)
        assert len(finalists) <= 4


class TestPluggableScoring:
    """The loop is scorer-agnostic — a different plugin ranks the same search."""

    def test_structure_ratio_scorer_recovers_substituted_columnar(self):
        # Substitution THEN columnar. A by-letter language model is blind here,
        # but the substitution-INVARIANT structure ratio still recovers the order.
        rng = random.Random(1234)
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        perm = letters[:]
        rng.shuffle(perm)
        enc = {p: c for p, c in zip(letters, perm)}
        substituted = "".join(enc[ch] for ch in PLAINTEXT)
        ct = columnar_encrypt(substituted, "MASONRY")

        finalists = search_keyed_columnar(
            ct, make_structure_ratio_scorer(2), min_width=5, max_width=8, top_n=3
        )
        best = finalists[0]
        assert best.column_count == 7
        assert best.column_order == tuple(_rank_order("MASONRY"))
        # Peeled stream is the (still-substituted) monoalphabetic cipher.
        assert best.decoded_stream == substituted

    def test_scorer_receives_decoded_stream_only(self):
        # Firewall-by-construction: the scorer sees only a decoded A-Z stream.
        seen: list[str] = []

        def spy(stream: str) -> float:
            seen.append(stream)
            return float(stream.count("E"))

        ct = columnar_encrypt(PLAINTEXT, "MASON")
        search_keyed_columnar(ct, spy, min_width=5, max_width=5, top_n=1)
        assert seen
        for stream in seen:
            assert set(stream) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            assert len(stream) == len(ct)


class TestBudgetKnobs:
    def test_wide_width_uses_hill_climb(self):
        # Width 9 (9! > default exhaustive budget) falls back to hill climbing.
        scorer = make_language_scorer("en")
        ct = columnar_encrypt(PLAINTEXT, "CRYPTOGRA")  # width 9
        finalists = search_keyed_columnar(
            ct, scorer,
            config=ColumnarSearchConfig(
                min_width=9, max_width=9, exhaustive_max_factorial=40320, restarts=48
            ),
        )
        assert finalists
        assert finalists[0].method == "hill_climb"

    def test_short_text_returns_empty(self):
        assert search_keyed_columnar("ABC", make_language_scorer("en")) == []
