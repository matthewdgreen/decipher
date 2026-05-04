from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_artifact.py"
spec = importlib.util.spec_from_file_location("inspect_artifact", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
inspect_artifact = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = inspect_artifact
spec.loader.exec_module(inspect_artifact)

import cli  # noqa: E402


def test_llm_summary_includes_analyzer_findings_and_runtime_evidence():
    artifact = {
        "model": "gpt-test",
        "provider": "openai",
        "test_id": "fixture_case",
        "status": "solved",
        "language": "la",
        "char_accuracy": 0.14,
        "word_accuracy": 0.07,
        "cipher_alphabet_size": 23,
        "cipher_word_count": 78,
        "solution": {"branch": "polish", "declared_at_iteration": 10},
        "branches": [
            {"name": "polish", "char_accuracy": 0.14, "word_accuracy": 0.07},
            {"name": "repair", "char_accuracy": 0.20, "word_accuracy": 0.12},
        ],
        "tool_calls": [
            {
                "iteration": 1,
                "tool_name": "search_automated_solver",
                "elapsed_ms": 42_000,
                "result": "{}",
            },
            {
                "iteration": 3,
                "tool_name": "act_set_mapping",
                "elapsed_ms": 6_500,
                "arguments": {"branch": "repair", "cipher_symbol": "P"},
                "result": {
                    "status": "ok",
                    "branch": "repair",
                    "score_delta": {"dict_rate_delta": -0.01},
                    "changed_words": [
                        {"before": "TREUITER", "after": "BREUITER"},
                        {"before": "MORIETANTU", "after": "MORIEBANTU"},
                    ],
                },
            },
        ],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I can read BREUITER here."},
                    {"type": "tool_use", "id": "u1", "name": "search_automated_solver", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "u1",
                        "content": '{"status":"ok","decoded_preview":"ETIAM QUOD"}',
                    }
                ],
            },
        ],
    }

    timeline = inspect_artifact.build_timeline(artifact)
    summary = inspect_artifact.build_llm_summary(artifact, timeline)

    assert summary["score_meaning"].startswith("char_accuracy")
    assert summary["artifact_tool_counts"]["act_set_mapping"] == 1
    assert summary["branch_scores"][0]["name"] == "repair"
    assert summary["analyzer_findings"]["labels"]["score_overrode_reading"] == 1
    assert summary["tool_timing"]["has_timing"] is True
    assert summary["tool_timing"]["unexpected_slow_tool_calls"][0]["tool"] == "act_set_mapping"


def test_analyze_tool_timing_flags_unexpected_slow_small_tool():
    artifact = {
        "tool_calls": [
            {
                "iteration": 1,
                "tool_name": "search_automated_solver",
                "elapsed_ms": 45_000,
                "arguments": {"branch": "auto"},
            },
            {
                "iteration": 2,
                "tool_name": "observe_branch",
                "elapsed_ms": 7_250,
                "arguments": {"branch": "auto"},
            },
            {
                "iteration": 3,
                "tool_name": "score_panel",
                "elapsed_ms": 300,
                "arguments": {"branch": "auto"},
            },
        ]
    }

    timing = inspect_artifact.analyze_tool_timing(artifact)
    text = inspect_artifact.format_timing_summary(timing)

    assert timing["has_timing"] is True
    assert timing["total_tool_seconds"] == 52.55
    assert timing["slow_tool_calls"][0]["tool"] == "search_automated_solver"
    assert timing["slow_tool_calls"][0]["expected_slow"] is True
    assert timing["unexpected_slow_tool_calls"] == [
        {
            "iteration": 2,
            "tool": "observe_branch",
            "elapsed_ms": 7_250,
            "elapsed_seconds": 7.25,
            "expected_slow": False,
            "arguments": {"branch": "auto"},
        }
    ]
    assert "Unexpectedly slow small tools" in text
    assert "observe_branch 7.2s" in text
    assert "search_automated_solver" in text


def test_analyze_tool_timing_handles_old_artifacts_without_elapsed_data():
    artifact = {
        "tool_calls": [
            {"iteration": 1, "tool_name": "observe_branch", "elapsed_ms": 0},
            {"iteration": 2, "tool_name": "score_panel"},
        ]
    }

    timing = inspect_artifact.analyze_tool_timing(artifact)
    text = inspect_artifact.format_timing_summary(timing)

    assert timing["has_timing"] is False
    assert "No nonzero per-tool elapsed_ms" in timing["message"]
    assert "older artifact" in text


def test_call_llm_uses_provider_layer_and_reports_cost(monkeypatch):
    class FakeProvider:
        def send(self, *, messages, tools=None, system="", max_tokens=4096):
            assert "Decipher artifact summary" in messages[0]["content"]
            return inspect_artifact.ModelResponse(
                content=[inspect_artifact.TextBlock(text="diagnosis")],
                usage=inspect_artifact.ModelUsage(input_tokens=1000, output_tokens=200),
            )

    import cli

    monkeypatch.setattr(cli, "_probe_api_key", lambda provider: "fake-key")
    monkeypatch.setattr(
        inspect_artifact,
        "make_model_provider",
        lambda provider, api_key, model: FakeProvider(),
    )

    result = inspect_artifact._call_llm(
        {"test_id": "fixture"},
        provider="openai",
        model="gpt-5.4-mini",
        max_tokens=200,
        analysis_mode="standard",
    )

    assert result.provider == "openai"
    assert result.model == "gpt-5.4-mini"
    assert result.text == "diagnosis"
    assert result.input_tokens == 1000
    assert result.output_tokens == 200
    assert result.estimated_cost_usd > 0


def test_cli_artifact_analysis_writes_sibling_markdown(tmp_path, monkeypatch):
    artifact = tmp_path / "run.json"
    artifact.write_text('{"status":"solved"}', encoding="utf-8")

    class FakeInspector:
        @staticmethod
        def inspect_one(path, analyze, provider, llm_model, max_tokens, analysis_mode):
            assert path == artifact
            assert analyze is True
            assert provider == "openai"
            assert llm_model == "gpt-5.4-mini"
            assert max_tokens == 123
            assert analysis_mode == "deep"
            print("Performing LLM analysis (openai/gpt-5.4-mini)...")
            print("diagnosis")

    monkeypatch.setattr(cli, "_load_inspect_artifact_module", lambda: FakeInspector)

    out = cli._maybe_write_artifact_analysis(
        artifact,
        SimpleNamespace(
            analyze=True,
            provider="openai",
            model="gpt-5.4-mini",
            analysis_max_tokens=123,
            analysis_mode="deep",
        ),
    )

    assert out == tmp_path / "run.analyzed.md"
    text = out.read_text(encoding="utf-8")
    assert "# Artifact Analysis: run.json" in text
    assert "diagnosis" in text


def test_cli_artifact_analysis_ignores_when_not_requested(tmp_path):
    artifact = tmp_path / "run.json"
    artifact.write_text("{}", encoding="utf-8")

    assert cli._maybe_write_artifact_analysis(artifact, SimpleNamespace(analyze=False)) is None
    assert not (tmp_path / "run.analyzed.md").exists()
