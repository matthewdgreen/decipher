from __future__ import annotations

from analysis.nomenclator import (
    render_token_views,
    render_with_expansions,
    reread_occurrences,
    suspicious_symbol_groups,
    symbol_context_packet,
)
from benchmark.loader import parse_canonical_transcription


def test_render_token_views_tracks_nulls_and_unknowns():
    cipher = parse_canonical_transcription("S001 S002 S003 S001")
    key = {
        cipher.alphabet.id_for("S001"): 0,
        cipher.alphabet.id_for("S003"): 2,
    }

    views, rendered = render_token_views(cipher, key=key, mask=("S003",))

    assert rendered == "AA"
    assert [(view.symbol, view.assignment, view.output_start, view.output_end) for view in views] == [
        ("S001", "A", 0, 1),
        ("S002", "?", 1, 1),
        ("S003", "<null>", 1, 1),
        ("S001", "A", 1, 2),
    ]


def test_render_with_expansions_projects_whole_word_symbol():
    cipher = parse_canonical_transcription("S001 S090 S002")
    key = {
        cipher.alphabet.id_for("S001"): 22,  # W
        cipher.alphabet.id_for("S002"): 18,  # S
    }

    rendered, views = render_with_expansions(
        cipher,
        key=key,
        expansions={"S090": "THE"},
    )

    assert rendered == "WTHES"
    assert views[1].symbol == "S090"
    assert views[1].rendered == "THE"
    assert views[1].output_start == 1
    assert views[1].output_end == 4


def test_suspicious_symbol_groups_and_context_packets():
    cipher = parse_canonical_transcription("S001 S002 S002 S003 S001")
    key = {
        cipher.alphabet.id_for("S001"): 0,
        cipher.alphabet.id_for("S002"): 1,
    }
    views, baseline = render_token_views(cipher, key=key)

    groups = suspicious_symbol_groups(
        views,
        include_one_letter_symbols=False,
        min_occurrences=1,
        max_symbols=5,
    )

    assert [symbol for symbol, _symbol_views in groups] == ["S003"]
    packet = symbol_context_packet(
        symbol="S003",
        views=groups[0][1],
        baseline=baseline,
        context_chars=3,
        recurrence_limit=3,
    )
    assert packet["signal"] == "missing_or_unmapped_symbol"
    assert packet["occurrences"][0]["context"] == "ABB⟦S003:?⟧A"


def test_reread_occurrences_after_expansion():
    cipher = parse_canonical_transcription("S001 S090 S002 S090 S003")
    key = {
        cipher.alphabet.id_for("S001"): 0,
        cipher.alphabet.id_for("S002"): 1,
        cipher.alphabet.id_for("S003"): 2,
    }
    expanded, views = render_with_expansions(
        cipher,
        key=key,
        expansions={"S090": "THE"},
    )

    rows = reread_occurrences(
        symbol="S090",
        views=views,
        expanded=expanded,
        context_chars=3,
        recurrence_limit=2,
    )

    assert expanded == "ATHEBTHEC"
    assert rows == ["A⟦THE⟧BTH", "HEB⟦THE⟧C"]
