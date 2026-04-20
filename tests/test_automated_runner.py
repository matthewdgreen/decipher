from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cli
from automated.runner import (
    AutomatedBenchmarkRunner,
    AutomatedRunResult,
    format_automated_preflight_for_llm,
)
import automated.runner as automated_runner
import benchmark.runner_v2 as runner_v2
from benchmark.runner_v2 import BenchmarkRunnerV2
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


def test_preflight_summary_omits_ground_truth_accuracy():
    result = AutomatedRunResult(
        test_id="secret_case",
        status="solved",
        final_decryption="THEQUICKBROWNFOX",
        elapsed_seconds=1.0,
        char_accuracy=1.0,
        word_accuracy=1.0,
        solver="fake_native",
        run_id="run123",
    )
    result.artifact = {
        "solver": "fake_native",
        "cipher_alphabet_size": 26,
        "cipher_token_count": 16,
        "cipher_word_count": 1,
        "ground_truth": "THEQUICKBROWNFOX",
        "char_accuracy": 1.0,
        "word_accuracy": 1.0,
        "steps": [{"name": "search_anneal", "score": -1.23}],
    }

    summary = format_automated_preflight_for_llm(result)

    assert "THEQUICKBROWNFOX" in summary
    assert "ground_truth" not in summary
    assert "char_accuracy" not in summary
    assert "word_accuracy" not in summary
    assert "1.0" not in summary
    assert "$0.00 (no LLM access)" in summary


def test_benchmark_runner_preflight_runs_without_ground_truth(monkeypatch, tmp_path):
    seen = {}

    class FakeAPI:
        model = "fake-model"

    def fake_preflight(cipher_text, language, test_id):
        seen["language"] = language
        seen["test_id"] = test_id
        return {
            "enabled": True,
            "status": "solved",
            "solver": "fake_native",
            "summary": "preflight summary",
            "key": {"0": 19, "1": 7, "2": 4},
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

    def fake_run_v2(**kwargs):
        seen["automated_preflight"] = kwargs["automated_preflight"]
        from artifact.schema import BranchSnapshot, RunArtifact

        artifact = RunArtifact(
            run_id="run123",
            cipher_id=kwargs["cipher_id"],
            model="fake-model",
            language=kwargs["language"],
            status="solved",
        )
        artifact.branches = [
            BranchSnapshot(
                name="main",
                parent=None,
                created_iteration=0,
                key={},
                mapped_count=0,
                decryption="THE",
            )
        ]
        return artifact

    monkeypatch.setattr(runner_v2, "_run_automated_preflight", fake_preflight)
    monkeypatch.setattr(runner_v2, "run_v2", fake_run_v2)

    runner = BenchmarkRunnerV2(
        claude_api=FakeAPI(),  # type: ignore[arg-type]
        artifact_dir=tmp_path,
    )
    result = runner.run_test(_test_data(), language="en")

    assert result.final_decryption == "THE"
    assert seen["test_id"] == "auto_demo"
    assert seen["language"] == "en"
    assert seen["automated_preflight"]["summary"] == "preflight summary"


def test_collapsed_plaintext_detector_flags_single_letter_failure():
    assert automated_runner._is_collapsed_plaintext("E" * 100)
    assert not automated_runner._is_collapsed_plaintext(
        "THEOLDPHOTOGRAPHHADYELLOWEDATTHEEDGESITSCOLORSFADING"
    )
