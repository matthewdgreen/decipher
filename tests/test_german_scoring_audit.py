from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_german_scoring.py"
spec = importlib.util.spec_from_file_location("audit_german_scoring", SCRIPT_PATH)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def test_normalize_de_az_expands_german_letters():
    assert audit.normalize_de_az("Ärger, Öl, süß!") == "AERGEROELSUESS"


def test_aggregate_rows_measures_plaintext_control_margin():
    rows = [
        {
            "scores": {
                "plaintext": {"dict_rate": 0.8, "wordlist_quad": -2.0},
                "reversed": {"dict_rate": 0.2, "wordlist_quad": -6.0},
                "shuffled": {"dict_rate": 0.1, "wordlist_quad": -7.0},
                "rotated": {"dict_rate": 0.0, "wordlist_quad": -8.0},
            }
        },
        {
            "scores": {
                "plaintext": {"dict_rate": 0.5, "wordlist_quad": -3.0},
                "reversed": {"dict_rate": 0.55, "wordlist_quad": -5.0},
                "shuffled": {"dict_rate": 0.2, "wordlist_quad": -7.0},
                "rotated": {"dict_rate": 0.1, "wordlist_quad": -8.0},
            }
        },
    ]

    aggregate = audit.aggregate_rows(rows, models=[])

    assert aggregate["dict_rate"]["plaintext_beats_all_controls"] == 1
    assert aggregate["dict_rate"]["sample_count"] == 2
    assert aggregate["wordlist_quad"]["plaintext_beats_all_controls"] == 2
    assert aggregate["wordlist_quad"]["min_margin"] == 2.0


def test_score_text_variant_includes_binary_model_score():
    class FakeModel:
        order = 5

        def lookup(self, gram: str) -> float:
            return -1.0 if gram == "wenig" else -3.0

    row = audit.score_text_variant(
        "WENIG",
        word_set={"WENIG"},
        quad={"_floor": -10.0, "WENI": -1.0, "ENIG": -1.0},
        models=[
            {
                "metadata": {"path": "models/fake_de.bin"},
                "model": FakeModel(),
            }
        ],
    )

    assert row["dict_rate"] == 1.0
    assert row["fake_de_mean_log_prob"] == -1.0
