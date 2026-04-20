from __future__ import annotations

import json

from scripts.validate_benchmark import validate_benchmark


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validate_benchmark_accepts_track_b_only_records(tmp_path):
    root = tmp_path
    (root / "manifest").mkdir()
    (root / "splits").mkdir()
    (root / "sources" / "synth" / "transcriptions").mkdir(parents=True)
    (root / "sources" / "synth" / "plaintext").mkdir(parents=True)
    (root / "sources" / "synth" / "transcriptions" / "a.canonical.txt").write_text("A B C")
    (root / "sources" / "synth" / "plaintext" / "a.txt").write_text("THE")
    (root / "manifest" / "schema.json").write_text(json.dumps({
        "required": ["id", "source", "status", "task_tracks"],
        "properties": {
            "id": {},
            "source": {},
            "status": {},
            "task_tracks": {},
            "transcription_canonical_file": {},
            "plaintext_file": {},
        },
        "additionalProperties": False,
    }))
    _write_jsonl(root / "manifest" / "records.jsonl", [{
        "id": "synth_001",
        "source": "synth",
        "status": "solved_verified",
        "task_tracks": ["transcription2plaintext"],
        "transcription_canonical_file": "sources/synth/transcriptions/a.canonical.txt",
        "plaintext_file": "sources/synth/plaintext/a.txt",
    }])
    _write_jsonl(root / "splits" / "parity.jsonl", [{
        "test_id": "parity_synth_001",
        "track": "transcription2plaintext",
        "target_records": ["synth_001"],
        "context_records": [],
    }])

    report = validate_benchmark(root)

    assert report.ok
    assert report.records == 1
    assert report.split_tests == 1


def test_validate_benchmark_reports_missing_files_and_bad_split_reference(tmp_path):
    root = tmp_path
    (root / "manifest").mkdir()
    (root / "splits").mkdir()
    (root / "manifest" / "schema.json").write_text(json.dumps({
        "required": ["id", "source", "status", "task_tracks"],
        "properties": {
            "id": {},
            "source": {},
            "status": {},
            "task_tracks": {},
            "image_files": {},
            "plaintext_file": {},
        },
        "additionalProperties": False,
    }))
    _write_jsonl(root / "manifest" / "records.jsonl", [{
        "id": "bad_001",
        "source": "bad",
        "status": "solved_verified",
        "task_tracks": ["image2plaintext"],
        "image_files": ["sources/bad/images/missing.png"],
        "plaintext_file": "sources/bad/plaintext/missing.txt",
    }])
    _write_jsonl(root / "splits" / "bad.jsonl", [{
        "test_id": "bad_split",
        "track": "transcription2plaintext",
        "target_records": ["unknown_record"],
        "context_records": [],
    }])

    report = validate_benchmark(root)

    assert not report.ok
    assert any("missing image file" in error for error in report.errors)
    assert any("missing plaintext_file" in error for error in report.errors)
    assert any("unknown record" in error for error in report.errors)
