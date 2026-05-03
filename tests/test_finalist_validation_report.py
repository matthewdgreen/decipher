from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_report_finalist_validation_summarizes_validation_rows(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({
            "test_id": "demo_case",
            "solver": "pure_transposition_screen_rust",
            "status": "completed",
            "steps": [{
                "name": "screen_pure_transposition",
                "selected": {"candidate_id": "route_1"},
                "top_candidates": [
                    {
                        "rank": 1,
                        "rust_rank": 4,
                        "candidate_id": "route_1",
                        "family": "route_columns_down",
                        "score": -1.2,
                        "selection_score": -1.2,
                        "validated_selection_score": -0.3,
                        "preview": "THEQUICKBROWNFOX",
                        "validation": {
                            "validation_label": "coherent_candidate",
                            "validation_score": 2.1,
                            "recommendation": "install_or_promote",
                            "strict_word_hit_score": 0.8,
                            "integrity": {
                                "integrity_label": "clean_or_canonical",
                                "damage_score": 0.02,
                                "pseudo_char_fraction": 0.0,
                                "suspicious_short_pseudo_count": 0,
                            },
                            "segmentation": {
                                "dict_rate": 1.0,
                                "pseudo_word_fraction": 0.0,
                                "cost_per_char": 1.2,
                            },
                        },
                    }
                ],
            }],
        }),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "report_finalist_validation.py"),
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )

    assert "Rows with validation evidence: 1" in result.stdout
    assert "`demo_case`" in result.stdout
    assert "`coherent_candidate`" in result.stdout
    assert "`clean_or_canonical`" in result.stdout
    assert "| selected rows | 1 |" in result.stdout
    assert "| rows with validation rank change | 1 |" in result.stdout
