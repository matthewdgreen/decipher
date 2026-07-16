"""Reading artifact tests (M3 Part 1): dataclasses, coercion, compilation."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from investigation.reading import (
    Reading,
    ReadingFragment,
    build_candidate_reading_packet,
    coerce_confidence,
    new_reading_id,
)
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


def test_confidence_coercion_and_clamp():
    assert coerce_confidence(0.7) == 0.7
    assert coerce_confidence("0.5") == 0.5
    assert coerce_confidence("1.4") == 1.0        # clamp high
    assert coerce_confidence("-2") == 0.0         # clamp low
    assert coerce_confidence("high") == 0.5       # unparseable -> default
    assert coerce_confidence(None) == 0.5


def test_reading_id_is_uuid12():
    rid = new_reading_id()
    assert len(rid) == 12
    int(rid, 16)  # hex


def test_reading_roundtrip_and_full_text_order():
    reading = Reading(
        branch="main",
        source="episode:abc123",
        created_turn=4,
        candidate_content_hash="abc123hash",
        candidate_renderer_id="decoded_text_v1",
        fragments=[
            ReadingFragment(text="WORLD.", repair_text="WORLD", start=6, end=11,
                            confidence=0.8, label="w2"),
            ReadingFragment(text="HELLO", start=0, end=5, confidence=0.9, label="w1"),
        ],
        holes=["gap@3"],
        overall_confidence=0.85,
    )
    # full_text joins in start order.
    assert reading.full_text == "HELLO WORLD."
    data = json.loads(json.dumps(reading.to_dict()))
    restored = Reading.from_dict(data)
    assert restored.branch == "main"
    assert restored.source == "episode:abc123"
    assert restored.created_turn == 4
    assert restored.candidate_content_hash == "abc123hash"
    assert restored.candidate_renderer_id == "decoded_text_v1"
    assert restored.holes == ["gap@3"]
    assert restored.overall_confidence == 0.85
    assert [f.text for f in restored.fragments] == ["WORLD.", "HELLO"]
    assert restored.fragments[0].start == 6
    assert restored.fragments[0].repair_text == "WORLD"
    assert restored.fragments[1].label == "w1"


def test_from_episode_result_carries_window_and_bounds():
    result = {
        "reading_text": "HELLO WORLD",
        "fragments": [
            {"window": "w1", "start": 0, "end": 5, "text": "Hello.",
             "repair_text": "HELLO", "confidence": "0.9"},
            {"window": "w2", "text": "WORLD", "confidence": 0.4},
        ],
        "holes": ["unresolved tail"],
        "overall_confidence": 0.7,
    }
    reading = Reading.from_episode_result(
        result, branch="hyp", source="episode:e1", created_turn=3
    )
    assert reading.branch == "hyp"
    assert reading.source == "episode:e1"
    assert len(reading.fragments) == 2
    # A8 start/end survive; window becomes label; string confidence coerced.
    assert reading.fragments[0].start == 0
    assert reading.fragments[0].end == 5
    assert reading.fragments[0].label == "w1"
    assert reading.fragments[0].confidence == 0.9
    assert reading.fragments[0].repair_text == "HELLO"
    assert reading.fragments[1].start is None
    assert reading.holes == ["unresolved tail"]
    assert reading.overall_confidence == 0.7


def test_from_episode_result_falls_back_to_reading_text():
    result = {"reading_text": "ONE FRAGMENT", "fragments": [], "holes": [],
              "overall_confidence": 0.5}
    reading = Reading.from_episode_result(
        result, branch="main", source="lead", created_turn=1
    )
    assert len(reading.fragments) == 1
    assert reading.fragments[0].text == "ONE FRAGMENT"
    assert reading.fragments[0].start is None


def test_from_episode_result_omitted_confidence_is_below_repair_threshold():
    # M5.1 softening: a worker fragment WITHOUT confidence compiles to 0.5 —
    # below MIN_REPAIR_FRAGMENT_CONFIDENCE (0.65) — visible but not
    # auto-actionable. An explicit confidence is preserved.
    from investigation.actions import MIN_REPAIR_FRAGMENT_CONFIDENCE
    r = Reading.from_episode_result(
        {
            "reading_text": "CAT ON",
            "fragments": [
                {"text": "CAT", "confidence": 0.9},
                {"text": "ON"},  # silent worker fragment
            ],
            "holes": [], "overall_confidence": 0.6,
        },
        branch="main", source="episode", created_turn=1, reading_id="rd_x",
    )
    assert r.fragments[0].confidence == 0.9
    assert r.fragments[1].confidence == 0.5
    assert r.fragments[1].confidence < MIN_REPAIR_FRAGMENT_CONFIDENCE


def test_candidate_packet_owns_repair_coordinates():
    alpha = Alphabet.from_text("ABCD EFGH", ignore_chars={" "})
    workspace = Workspace(CipherText(
        raw="ABCD EFGH", alphabet=alpha, separator=" "
    ))
    branch = workspace.get_branch("main")
    for token, letter in zip(workspace.effective_tokens("main"), "TESTWORD"):
        branch.key[token] = workspace.plaintext_alphabet.id_for(letter)

    packet = build_candidate_reading_packet(
        workspace, "main", window_tokens=4
    ).to_dict()
    assert packet["capability"] == "editable_key"
    assert [s["text"] for s in packet["spans"]] == ["TEST", "WORD"]
    span = packet["spans"][1]

    reading = Reading.from_episode_result(
        {
            "reading_text": "TEST WORD",
            "fragments": [{
                "span_id": span["span_id"],
                # These are deliberately wrong and must not be trusted.
                "start": 0,
                "end": 1,
                "text": "WORD",
                "repair_text": "WORD",
                "confidence": 0.9,
            }],
            "holes": [],
            "overall_confidence": 0.9,
        },
        branch="main",
        source="episode:e1",
        created_turn=2,
        candidate_packet=packet,
    )
    fragment = reading.fragments[0]
    assert fragment.span_id == span["span_id"]
    assert (fragment.start, fragment.end) == (4, 8)


def test_metadata_candidate_packet_is_explicitly_text_only():
    alpha = Alphabet.from_text("ABC", ignore_chars=set())
    workspace = Workspace(CipherText(raw="ABC", alphabet=alpha, separator=None))
    workspace.get_branch("main").metadata["decoded_text"] = "A READABLE OVERLAY"
    packet = build_candidate_reading_packet(workspace, "main").to_dict()
    assert packet["capability"] == "text_only"
    assert packet["spans"][0]["token_start"] is None
    assert packet["spans"][0]["text"] == "A READABLE OVERLAY"


def test_null_mask_candidate_packet_preserves_filtered_token_provenance():
    alpha = Alphabet.from_text("AXBC", ignore_chars=set())
    workspace = Workspace(CipherText(raw="AXBC", alphabet=alpha, separator=None))
    pt = workspace.plaintext_alphabet
    branch = workspace.get_branch("main")
    for symbol, letter in {"A": "C", "X": "Z", "B": "T", "C": "S"}.items():
        workspace.set_mapping("main", alpha.id_for(symbol), pt.id_for(letter))
    branch.metadata.update({
        "decoded_text": "CTS",
        "key_type": "homophonic_with_null_mask",
        "null_mask_finalist": {"mask": ["X"], "rank": 1},
    })

    packet = build_candidate_reading_packet(
        workspace, "main", window_tokens=20
    ).to_dict()
    assert packet["capability"] == "editable_null_mask"
    assert set(packet["capabilities"]) >= {"editable_key", "editable_null_mask"}
    assert packet["candidate_text"] == "CTS"
    assert packet["spans"][0]["token_indices"] == [0, 2, 3]
    assert packet["spans"][0]["text"] == "CTS"
