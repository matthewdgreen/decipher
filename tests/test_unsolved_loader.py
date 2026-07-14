"""Unsolved-record reader tests (INV-0 Part 6 / Part 9)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark.unsolved import list_unsolved_record_ids, load_unsolved_record


def _make_tree(tmp_path):
    root = tmp_path / "benchmark"
    unsolved = root / "unsolved"
    (unsolved / "manifest").mkdir(parents=True)
    (unsolved / "sources").mkdir(parents=True)
    (unsolved / "sources" / "num.canonical.txt").write_text("71 194 38 1701 89\n")
    (unsolved / "sources" / "held.canonical.txt").write_text("1 2 3 4 5\n")

    records = [
        {
            "id": "num_target",
            "source": "test",
            "cipher_type": ["numeric_code", "book_cipher"],
            "symbol_set": ["numeric_tokens"],
            "token_count": 5,
            "transcription_canonical_file": "sources/num.canonical.txt",
            "rights_class": "open",
            "plaintext_language": "en",
            "context_layers": {"secret": "PLAINTEXTLEAK"},
            "associated_documents": [{"content": "SOLUTION HINT"}],
            "notable_attempts": ["do not surface"],
            "related_records": [
                {"record_id": "companion_a", "relationship": "companion", "area": "unsolved"},
                {"record_id": "companion_b", "relationship": "companion", "area": "unsolved"},
            ],
        },
        {
            "id": "held_doc",
            "source": "test",
            "cipher_type": ["symbol_cipher"],
            "symbol_set": ["symbols"],
            "token_count": 5,
            "transcription_canonical_file": "sources/held.canonical.txt",
            "rights_class": "hold_for_review",
            "context_layers": {"secret": "WITHHELD"},
            "related_records": [],
        },
    ]
    with open(unsolved / "manifest" / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(root)


def test_related_record_ids_and_numeric_values(tmp_path):
    root = _make_tree(tmp_path)
    assert set(list_unsolved_record_ids(root)) == {"num_target", "held_doc"}
    rec = load_unsolved_record(root, "num_target")
    assert rec.related_record_ids == ["companion_a", "companion_b"]
    assert rec.numeric_values() == [71, 194, 38, 1701, 89]
    assert rec.content_withheld is False


def test_withheld_and_firewalled_fields_not_surfaced(tmp_path):
    root = _make_tree(tmp_path)
    rec = load_unsolved_record(root, "num_target")
    dumped = json.dumps(rec.to_dict())
    # Firewalled context / documents / attempts are never surfaced.
    assert "PLAINTEXTLEAK" not in dumped
    assert "SOLUTION HINT" not in dumped
    assert "context_layers" not in rec.metadata
    assert "associated_documents" not in rec.metadata
    assert "notable_attempts" not in rec.metadata


def test_hold_for_review_keys_off_rights_class(tmp_path):
    root = _make_tree(tmp_path)
    held = load_unsolved_record(root, "held_doc")
    assert held.content_withheld is True
    assert held.metadata.get("rights_class") == "hold_for_review"
    # Non-numeric hold_for_review target: transcription content withheld.
    assert held.canonical_text is None
    assert "WITHHELD" not in json.dumps(held.to_dict())
