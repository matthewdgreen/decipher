from __future__ import annotations

import argparse
import importlib.util
import sys
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
