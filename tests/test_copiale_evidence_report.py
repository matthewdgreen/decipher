from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPORT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_copiale_evidence.py"
report_spec = importlib.util.spec_from_file_location("report_copiale_evidence", REPORT_PATH)
assert report_spec is not None and report_spec.loader is not None
copiale_report = importlib.util.module_from_spec(report_spec)
sys.modules[report_spec.name] = copiale_report
report_spec.loader.exec_module(copiale_report)


def test_diagnose_canonical_transcription_reports_cipher_side_features():
    diag = copiale_report.diagnose_canonical_transcription(
        "S001 S002 | S001 S003 | S001 S002 | S004"
    )

    assert diag["token_count"] == 7
    assert diag["word_count"] == 4
    assert diag["unique_symbols"] == 4
    assert diag["singleton_symbol_count"] == 2
    assert diag["top_symbols"][0]["symbol"] == "S001"
    assert diag["top_symbols"][0]["count"] == 3
    assert diag["repeated_cipher_words"] == [{"word": "S001 S002", "count": 2}]
    assert "short_page" in diag["diagnostic_flags"]
