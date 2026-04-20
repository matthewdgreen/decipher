from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cli
from automated.runner import AutomatedBenchmarkRunner, AutomatedRunResult
import automated.runner as automated_runner
from benchmark.loader import BenchmarkTest, TestData as BenchmarkTestData


def _test_data() -> BenchmarkTestData:
    return BenchmarkTestData(
        test=BenchmarkTest(
            test_id="auto_demo",
            track="transcription2plaintext",
            cipher_system="simple_substitution",
            target_records=[],
            context_records=[],
            description="automated demo",
        ),
        canonical_transcription="A B C",
        plaintext="THE",
    )


def test_automated_benchmark_runner_writes_no_llm_artifact(tmp_path, monkeypatch):
    def fake_run(cipher_text, language="en", cipher_id="cli", ground_truth=None):
        result = AutomatedRunResult(
            test_id=cipher_id,
            status="solved",
            final_decryption="THE",
            elapsed_seconds=0.01,
            char_accuracy=1.0,
            word_accuracy=1.0,
            solver="fake_automated",
            run_id="run123",
        )
        result.artifact = {
            "run_id": "run123",
            "run_mode": "automated_only",
            "automated_only": True,
            "test_id": cipher_id,
            "status": "solved",
            "solver": "fake_automated",
            "decryption": "THE",
            "char_accuracy": 1.0,
            "word_accuracy": 1.0,
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
        return result

    monkeypatch.setattr(automated_runner, "run_automated", fake_run)
    runner = AutomatedBenchmarkRunner(artifact_dir=tmp_path)

    result = runner.run_test(_test_data(), language="en")

    with open(result.artifact_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    assert result.char_accuracy == 1.0
    assert result.iterations_used == 0
    assert result.total_tokens == 0
    assert result.estimated_cost_usd == 0.0
    assert artifact["run_mode"] == "automated_only"
    assert artifact["estimated_cost_usd"] == 0.0
    assert artifact["total_input_tokens"] == 0


def test_cli_crack_automated_only_bypasses_api_key(tmp_path, monkeypatch, capsys):
    input_file = tmp_path / "cipher.txt"
    input_file.write_text("A B C\n", encoding="utf-8")

    def forbidden_api_key():
        raise AssertionError("get_api_key must not be called")

    def fake_run(cipher_text, language="en", cipher_id="cli", ground_truth=None):
        result = AutomatedRunResult(
            test_id=cipher_id,
            status="solved",
            final_decryption="THE",
            elapsed_seconds=0.01,
            solver="fake_automated",
            run_id="run123",
        )
        result.artifact = {
            "run_id": "run123",
            "run_mode": "automated_only",
            "automated_only": True,
            "test_id": cipher_id,
            "status": "solved",
            "solver": "fake_automated",
            "decryption": "THE",
            "steps": [],
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
        return result

    monkeypatch.setattr(cli, "get_api_key", forbidden_api_key)
    monkeypatch.setattr(automated_runner, "run_automated", fake_run)

    cli.cmd_crack(argparse.Namespace(
        file=str(input_file),
        canonical=True,
        max_iterations=25,
        model=None,
        language="en",
        artifact_dir=str(tmp_path / "artifacts"),
        cipher_id="auto_cli",
        verbose=False,
        automated_only=True,
    ))

    out = capsys.readouterr().out
    assert "Running automated-only solver" in out
    assert "THE" in out
