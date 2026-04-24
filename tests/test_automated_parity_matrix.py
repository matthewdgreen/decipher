from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

from benchmark.loader import BenchmarkTest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_automated_parity_matrix.py"
spec = importlib.util.spec_from_file_location("run_automated_parity_matrix", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
matrix = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = matrix
spec.loader.exec_module(matrix)


def test_parse_int_ranges_supports_lists_and_ranges():
    assert matrix.parse_int_ranges("1-3,7,10-8") == [1, 2, 3, 7, 10, 9, 8]


def test_select_chunk_applies_shard_then_offset_limit():
    cases = list(range(10))

    selected = matrix.select_chunk(cases, offset=1, limit=2, shard_index=1, shard_count=3)

    assert selected == [4, 7]


def test_allow_generate_populates_missing_synthetic_case(monkeypatch, tmp_path):
    calls = {}

    def fake_generation_api_key():
        calls["api_key_requested"] = True
        return "test-key"

    def fake_build_test_case(spec, cache, api_key="", seed=None, generator=None):
        calls["api_key"] = api_key
        return matrix.TestData(
            test=BenchmarkTest(
                test_id="synth_demo",
                track="transcription2plaintext",
                cipher_system="simple_substitution",
                target_records=[],
                context_records=[],
                description="synthetic demo",
            ),
            canonical_transcription="A B C",
            plaintext="THE",
        )

    monkeypatch.setattr(matrix, "_generation_api_key", fake_generation_api_key)
    monkeypatch.setattr(matrix, "build_test_case", fake_build_test_case)

    args = argparse.Namespace(
        cache_dir=str(tmp_path / "cache"),
        allow_generate=True,
        fail_on_cache_miss=False,
        presets=[],
        families=["simple-wb"],
        lengths=[80],
        seeds="1",
        language="en",
        benchmark_split=[],
        benchmark_root="unused",
        track="transcription2plaintext",
    )

    cases = matrix.build_cases(args)

    assert len(cases) == 1
    assert cases[0].test_data.test.test_id == "synth_demo"
    assert calls["api_key_requested"] is True
    assert calls["api_key"] == "test-key"


def test_main_records_external_exception_and_continues(monkeypatch, tmp_path, capsys):
    case = matrix.MatrixCase(
        source="benchmark:test.jsonl",
        family="copiale",
        test_data=matrix.TestData(
            test=BenchmarkTest(
                test_id="copiale_demo",
                track="transcription2plaintext",
                cipher_system="copiale",
                target_records=[],
                context_records=[],
                description="copiale demo",
            ),
            canonical_transcription="A B C",
            plaintext="THE",
        ),
        spec=None,
    )

    class FakeRunner:
        def run_test(self, test_data, language=None):
            return SimpleNamespace(
                test_id=test_data.test.test_id,
                status="solved",
                final_decryption="THE",
                elapsed_seconds=0.01,
                char_accuracy=1.0,
                word_accuracy=1.0,
                artifact_path="decipher_artifact.json",
            )

    class FakeConfig:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(matrix, "build_cases", lambda args: [case])
    monkeypatch.setattr(matrix, "load_external_configs", lambda path, oracle: [FakeConfig("zkdecrypto-lite"), FakeConfig("zenith")])
    monkeypatch.setattr(matrix, "AutomatedBenchmarkRunner", lambda artifact_dir, **kw: FakeRunner())

    def fake_external(test_data, config, artifact_dir):
        if config.name == "zkdecrypto-lite":
            raise ValueError("symbol alphabet overflow")
        class Result:
            status = "completed"
            char_accuracy = 0.9
            word_accuracy = 0.0
            elapsed = 1.2
            artifact_path = "zenith_artifact.json"
            error = ""
            candidates_considered = 1
        return Result()

    monkeypatch.setattr(matrix, "run_external_baseline", fake_external)

    summary_jsonl = tmp_path / "summary.jsonl"
    summary_csv = tmp_path / "summary.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_automated_parity_matrix.py",
            "--solvers", "decipher", "external",
            "--external-config", "dummy.json",
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--summary-jsonl", str(summary_jsonl),
            "--summary-csv", str(summary_csv),
        ],
    )

    matrix.main()

    rows = [json.loads(line) for line in summary_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    by_solver = {row["solver"]: row for row in rows}
    assert by_solver["decipher-automated"]["status"] == "solved"
    assert by_solver["zkdecrypto-lite"]["status"] == "failed"
    assert "symbol alphabet overflow" in by_solver["zkdecrypto-lite"]["error"]
    assert by_solver["zenith"]["status"] == "completed"

    csv_rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    assert len(csv_rows) == 3
    out = capsys.readouterr().out
    assert "[zenith] running..." in out


def test_main_records_decipher_exception_and_still_writes_summary(monkeypatch, tmp_path):
    case = matrix.MatrixCase(
        source="synthetic",
        family="simple-wb",
        test_data=matrix.TestData(
            test=BenchmarkTest(
                test_id="synth_demo",
                track="transcription2plaintext",
                cipher_system="simple_substitution",
                target_records=[],
                context_records=[],
                description="synthetic demo",
            ),
            canonical_transcription="A B C",
            plaintext="THE",
        ),
        spec=None,
    )

    class ExplodingRunner:
        def run_test(self, test_data, language=None):
            raise RuntimeError("native solver exploded")

    monkeypatch.setattr(matrix, "build_cases", lambda args: [case])
    monkeypatch.setattr(matrix, "load_external_configs", lambda path, oracle: [])
    monkeypatch.setattr(matrix, "AutomatedBenchmarkRunner", lambda artifact_dir, **kw: ExplodingRunner())

    summary_jsonl = tmp_path / "summary.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_automated_parity_matrix.py",
            "--solvers", "decipher",
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--summary-jsonl", str(summary_jsonl),
            "--summary-csv", str(tmp_path / "summary.csv"),
        ],
    )

    matrix.main()

    rows = [json.loads(line) for line in summary_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["solver"] == "decipher-automated"
    assert row["status"] == "failed"
    assert "native solver exploded" in row["error"]
