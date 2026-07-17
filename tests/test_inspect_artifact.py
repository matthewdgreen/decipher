from __future__ import annotations

import importlib.util
import io
import json
import sys
from types import SimpleNamespace
from pathlib import Path


FIXTURES = Path(__file__).resolve().parent / "fixtures"
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
    assert result.output_may_be_truncated is True


def test_call_llm_does_not_warn_when_under_token_cap(monkeypatch):
    class FakeProvider:
        def send(self, *, messages, tools=None, system="", max_tokens=4096):
            return inspect_artifact.ModelResponse(
                content=[inspect_artifact.TextBlock(text="diagnosis")],
                usage=inspect_artifact.ModelUsage(input_tokens=1000, output_tokens=199),
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

    assert result.output_may_be_truncated is False


def test_call_llm_retries_empty_length_response(monkeypatch):
    class FakeProvider:
        def __init__(self):
            self.calls = 0

        def send(self, *, messages, tools=None, system="", max_tokens=4096):
            self.calls += 1
            if self.calls == 1:
                return inspect_artifact.ModelResponse(
                    content=[],
                    usage=inspect_artifact.ModelUsage(
                        input_tokens=100,
                        output_tokens=max_tokens,
                    ),
                    raw=SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")]),
                )
            assert "no visible assistant text" in messages[0]["content"]
            assert "visible markdown only" in system
            return inspect_artifact.ModelResponse(
                content=[inspect_artifact.TextBlock(text="## Verdict\nRecovered.")],
                usage=inspect_artifact.ModelUsage(input_tokens=120, output_tokens=20),
                raw=SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")]),
            )

    provider = FakeProvider()

    import cli

    monkeypatch.setattr(cli, "_probe_api_key", lambda provider: "fake-key")
    monkeypatch.setattr(
        inspect_artifact,
        "make_model_provider",
        lambda provider, api_key, model: provider_obj,
    )
    provider_obj = provider

    result = inspect_artifact._call_llm(
        {"test_id": "fixture"},
        provider="openrouter",
        model="tencent/hy3-preview:free",
        max_tokens=200,
        analysis_mode="standard",
    )

    assert result.attempts == 2
    assert result.input_tokens == 220
    assert result.output_tokens == 220
    assert "first LLM call consumed completion tokens" in result.text
    assert "## Verdict" in result.text
    assert result.stop_reason == "stop"


def test_call_llm_reports_empty_retry_failure(monkeypatch):
    class FakeProvider:
        def send(self, *, messages, tools=None, system="", max_tokens=4096):
            return inspect_artifact.ModelResponse(
                content=[],
                usage=inspect_artifact.ModelUsage(input_tokens=100, output_tokens=max_tokens),
                raw=SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")]),
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
        provider="openrouter",
        model="tencent/hy3-preview:free",
        max_tokens=200,
        analysis_mode="standard",
    )

    assert result.attempts == 2
    assert "provider returned no visible assistant text" in result.text
    assert "Try a larger --analysis-max-tokens" in result.text


def test_extract_stop_reason_from_openai_like_response():
    raw = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")])

    assert inspect_artifact._extract_stop_reason(raw) == "length"


def test_cli_artifact_analysis_writes_sibling_markdown(tmp_path, monkeypatch):
    artifact = tmp_path / "run.json"
    artifact.write_text('{"status":"solved"}', encoding="utf-8")

    class FakeInspector:
        @staticmethod
        def inspect_one(
            path,
            analyze,
            provider,
            llm_model,
            max_tokens,
            analysis_mode,
            timeout_seconds=None,
            retry_empty_response=True,
        ):
            assert path == artifact
            assert analyze is True
            assert provider == "openai"
            assert llm_model == "gpt-5.4-mini"
            assert max_tokens == 123
            assert analysis_mode == "deep"
            assert timeout_seconds == 45
            assert retry_empty_response is False
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
            analysis_timeout=45,
            analysis_no_empty_retry=True,
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


# ---------------------------------------------------------------------------
# --narrative post-hoc transcript replay (CLI-2 spec Part 3)
# ---------------------------------------------------------------------------

def _render_narrative(artifact: dict, tmp_path: Path, *, verbose: bool = False) -> str:
    p = tmp_path / "art.json"
    p.write_text(json.dumps(artifact), encoding="utf-8")
    buf = io.StringIO()
    inspect_artifact.render_narrative(p, artifact, verbose=verbose, stream=buf)
    return buf.getvalue()


def test_narrative_replay_real_fixture_has_structure(tmp_path):
    # The stored v2 fixture has no loop_events but 8 tool_calls; the replay
    # reconstructs numbered tool lines + Part-1 glosses from them.
    artifact = json.loads(
        (FIXTURES / "v2_artifact_synth_en_40wb_s1.json").read_text(encoding="utf-8")
    )
    out = _render_narrative(artifact, tmp_path)
    # Header
    assert "▶" in out
    # Numbered tool lines (reconstructed from tool_calls)
    assert "1 │ observe_frequency" in out
    assert "2 │ observe_isomorph_clusters" in out
    # Part-1 gloss line applies automatically (same renderer)
    assert "· counts how often each symbol appears" in out
    # Result block + artifact path (= the input path)
    assert "── result ──" in out
    assert f"artifact: {tmp_path / 'art.json'}" in out
    # It says the artifact predates event capture (graceful reconstruction)
    assert "reconstructed from stored tool calls" in out


def test_narrative_empty_loop_events_degrades_to_header_and_result(tmp_path):
    artifact = {
        "cipher_id": "empty_case", "model": "m", "language": "en",
        "status": "exhausted", "max_iterations": 5,
        "loop_events": [], "tool_calls": [],
        "char_accuracy": None, "started_at": 0.0, "finished_at": 1.0,
    }
    out = _render_narrative(artifact, tmp_path)
    assert "▶ empty_case" in out
    assert "predates loop-event capture" in out
    assert "── result ──" in out
    # No numbered tool lines when there is nothing to replay.
    assert "1 │" not in out
    # No ground-truth signal -> accuracies suppressed.
    assert "char=" not in out


def test_narrative_v3_synthetic_events_replay_with_nesting_and_decode(tmp_path):
    artifact = {
        "cipher_id": "v3_synth", "model": "gpt", "language": "la",
        "loop_version": "v3", "max_iterations": 5, "status": "solved",
        "char_accuracy": 0.99, "word_accuracy": 0.9,
        "started_at": 0.0, "finished_at": 30.0, "estimated_cost_usd": 1.5,
        "branches": [{"name": "main", "decryption": "THE QUICK BROWN FOX"}],
        "solution": {"branch": "main"},
        "loop_events": [
            {"event": "iteration_start", "payload": {"iteration": 1}},
            {"event": "agent_text",
             "payload": {"text": "Forking to test a substitution hypothesis."}},
            {"event": "tool_start",
             "payload": {"tool": "episode_run",
                         "arguments": {"kind": "search", "branch": "main"}}},
            {"event": "episode_tool_call",
             "payload": {"episode_id": "e1", "kind": "search", "turn": 1,
                         "tool": "search_hill_climb", "arguments": {"branch": "main"}}},
            {"event": "episode_complete",
             "payload": {"episode_id": "e1", "kind": "search", "status": "ok", "calls": 3}},
            {"event": "tool_call",
             "payload": {"tool": "episode_run", "result_summary": {}}},
            {"event": "workspace_snapshot",
             "payload": {"branch": "main", "decryption_preview": "THE QUICK BROWN FOX",
                         "total_tokens": 1000, "estimated_cost_usd": 0.5}},
            {"event": "declared_solution",
             "payload": {"branch": "main", "confidence": "high"}},
        ],
    }
    out = _render_narrative(artifact, tmp_path)
    lines = out.splitlines()
    # Agent self-narration line rendered.
    assert "“ Forking to test a substitution hypothesis." in out
    # Episode nesting intact: parent above its gloss above its ↳ child.
    parent_idx = next(i for i, l in enumerate(lines) if "│ episode_run(" in l)
    gloss_idx = next(i for i, l in enumerate(lines) if "· spins off a helper to run a search" in l)
    child_idx = next(i for i, l in enumerate(lines) if "↳" in l and "search_hill_climb" in l)
    assert parent_idx < gloss_idx < child_idx
    # Decode-progress line intact.
    assert "decode [main]" in out and "THE QUICK BROWN FOX" in out
    # Declaration + result block with the real accuracy.
    assert "DECLARED solution on branch main" in out
    assert "── result ──" in out and "char=99.0%" in out


def test_narrative_cli_main_end_to_end(tmp_path, monkeypatch, capsys):
    artifact = json.loads(
        (FIXTURES / "v2_artifact_synth_en_40wb_s1.json").read_text(encoding="utf-8")
    )
    p = tmp_path / "art.json"
    p.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["inspect_artifact.py", str(p), "--narrative"])
    inspect_artifact.main()
    out = capsys.readouterr().out
    assert "▶" in out
    assert "1 │ observe_frequency" in out
    assert "── result ──" in out


def test_format_attestations_shows_verdict_and_new_fields():
    """Slice 6: the verdict column classifies positive/weak/negative; a legacy
    weak record renders n/a for the absent unit fields; the declared marker
    still fires."""
    positive_hash = "hpos"
    artifact = {
        "solution": {
            "branch": "cand",
            "attestation": {"content_hash": positive_hash},
        },
        "attestations": [
            {
                "branch": "cand", "content_hash": positive_hash,
                "coherence": 9, "reader_accepts": True,
                "reader_accepts_as_solution": True,
                "target_language_confidence": 0.9,
                "semantic_recoverability": 0.8,
                "damage_scope": "local", "repairability": "local_repair",
                "gloss": "reads", "anomalies": [],
            },
            {
                # Legacy weak record: no new keys, reader_accepts True but
                # coherence < 7 (legacy-non-positive) -> "weak".
                "branch": "old", "content_hash": "hlegacy",
                "coherence": 4, "reader_accepts": True,
                "gloss": "partial", "anomalies": ["broken"],
            },
        ],
    }
    out = inspect_artifact.format_attestations(artifact)
    assert "positive" in out
    assert "weak" in out
    # The positive row is marked as declared (hash match).
    assert "*declared" in out
    # The legacy row renders n/a for the absent unit fields.
    lines = out.splitlines()
    legacy_line = next(line for line in lines if line.strip().startswith("old"))
    assert "n/a" in legacy_line
