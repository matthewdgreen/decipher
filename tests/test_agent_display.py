from __future__ import annotations

import io
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.display import (
    JsonlAgentRenderer,
    RawAgentRenderer,
    _compact_final_summary,
    _compact_preview,
    summarize_tool_call,
)


def test_summarize_tool_call_prefers_readable_repair_fields():
    out = summarize_tool_call(
        "act_apply_word_repair",
        {
            "status": "ok",
            "branch": "main",
            "from": "TREUITER",
            "to": "BREUITER",
        },
    )

    assert "act_apply_word_repair" in out
    assert "[main]" in out
    assert "TREUITER -> BREUITER" in out
    assert "(ok)" in out


def test_raw_renderer_preserves_compact_event_stream():
    stream = io.StringIO()
    renderer = RawAgentRenderer(stream)

    renderer.start_test("case", "desc", model="model", max_iterations=2)
    renderer.event("preflight_start", {})
    renderer.event("preflight_result", {"status": "completed", "elapsed_seconds": 1.2})
    renderer.event("iteration_start", {"iteration": 1})
    renderer.event("tool_call", {"tool": "decode_show"})
    renderer.event("run_complete", {"status": "solved"})

    text = stream.getvalue()
    assert "[agentic] case" in text
    assert "preflight(no-LLM)" in text
    assert "iter 1" in text
    assert "." in text
    assert "[run_complete]" in text


def test_jsonl_renderer_emits_machine_readable_events():
    stream = io.StringIO()
    renderer = JsonlAgentRenderer(stream)

    renderer.start_test("case", "desc", model="model", max_iterations=2)
    renderer.event("iteration_start", {"iteration": 1})
    renderer.finish(SimpleNamespace(
        test_id="case",
        status="solved",
        char_accuracy=1.0,
        word_accuracy=0.5,
        artifact_path="artifacts/case/run.json",
        error_message="",
        final_summary="A short final reading summary.",
    ))

    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert rows[0]["event"] == "test_start"
    assert rows[1]["event"] == "iteration_start"
    assert rows[2]["event"] == "test_finish"
    assert rows[2]["char_accuracy"] == 1.0
    assert rows[2]["final_summary"] == "A short final reading summary."


def test_final_display_compacts_summary_sections_without_blank_lines():
    text = """Target language: Latin.
Reading note: this is an agent summary.

What it appears to say:
This is an archaic Latin text about poultry.

Process notes:
- One repair was held.
- One boundary retry failed.

Further iterations:
Likely helpful. Try local repair.
"""

    compact = _compact_final_summary(text, max_chars=500)

    assert "What it appears to say: This is an archaic Latin text" in compact
    assert "Process notes: One repair was held. One boundary retry failed." in compact
    assert "\n\n" not in compact


def test_final_display_compacts_decrypt_to_single_preview_line():
    compact = _compact_preview("THERE | FORE\nTHE   OLD", max_chars=100)

    assert compact == "THERE | FORE THE OLD"
