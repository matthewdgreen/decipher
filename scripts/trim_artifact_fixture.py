#!/usr/bin/env python3
"""trim_artifact_fixture.py — reproducible analyzer-fixture trimmer (M5.3 Slice 7).

Usage:
    python scripts/trim_artifact_fixture.py <src.json> <dst.json>

Produces a small, GROUND-TRUTH-FREE artifact for the inspect_artifact analyzer
regression tests. Pure stdlib, deterministic output
(``json.dump(..., indent=2, sort_keys=True)``).

What is removed and why: the ground-truth text (``ground_truth``), the benchmark
prompt / external context (``benchmark_context``, ``external_context``), alignment
material (never a stored key in this artifact shape; the banned-key firewall test
still guards the names), and every RAW MODEL BODY (assistant text, session
transcript, recent exchanges, loop-event ``agent_text``) are removed. Only the
structured signals the analyzer renders survive.

``char_accuracy`` / ``word_accuracy`` are post-hoc grading SCORES, not the answer;
the header renders them, so they stay. They are not the plaintext.

The result is ANALYZER-ONLY: because ``investigation_state`` is trimmed to a small
key allowlist, the fixture is NOT loadable via
``InvestigationState.from_artifact_dict``. It is intended solely for the
``scripts/inspect_artifact.py`` read-side sections.

The committed output (``tests/fixtures/v3_artifact_m5_2_smoke_trimmed.json``) is
roughly 720 KB after trimming the 1.46 MB M5.2 smoke artifact (the pretty-printed,
sort_keys output is larger than the compact source but still analyzer-sized).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_MAX_RESULT_CHARS = 16_384
_DECRYPTION_LIMIT = 1_200

# Top-level keys deleted wholesale (raw bodies, ground truth, benchmark prompt).
_DELETE_TOP_LEVEL = (
    "ground_truth",
    "session_transcript",
    "benchmark_context",
    "loop_events",
    "notebook",
    "subagent_runs",
)

# The ONLY investigation_state keys retained: exactly what the analyzer consumes
# plus small counters. Everything else (cipher, workspace, evidence_log,
# budget_ledger, recent_exchanges, external_context, episode_ledger,
# verify_attestations, readings, hypothesis_board, branch_aliases,
# finalist_sessions, experiment_queue, last_information_digest) is dropped.
_INV_STATE_ALLOWLIST = frozenset({
    "language",
    "turn",
    "repair_transactions",
    "repair_saturation",
    "repair_agenda",
    "workflow_hint_keys",
    "call_signature_counts",
    "no_new_information_streak",
    "model_variant",
})

_BLOCK_DROP_KEYS = frozenset({"provider_extra", "thinking", "reasoning", "signature"})


def _trim_long_result(text: str) -> str:
    """Replace an over-long tool result string with a compact placeholder."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        return json.dumps({"_trimmed": True, "status": parsed.get("status")})
    return '{"_trimmed": true}'


def _trim_message_block(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    btype = block.get("type")
    if btype == "tool_use":
        # Keep only structural identity + input.
        return {
            "type": "tool_use",
            "id": block.get("id"),
            "name": block.get("name"),
            "input": block.get("input", {}),
        }
    out = {k: v for k, v in block.items() if k not in _BLOCK_DROP_KEYS}
    if btype == "text":
        out["text"] = ""  # raw model prose removed
    elif btype == "tool_result":
        content = out.get("content")
        if isinstance(content, str) and len(content) > _MAX_RESULT_CHARS:
            out["content"] = _trim_long_result(content)
    return out


def _trim_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    out = dict(message)
    content = out.get("content")
    if isinstance(content, list):
        out["content"] = [_trim_message_block(b) for b in content]
    return out


def _trim_tool_call(call: Any) -> Any:
    if not isinstance(call, dict):
        return call
    out = dict(call)
    result = out.get("result")
    if isinstance(result, str) and len(result) > _MAX_RESULT_CHARS:
        out["result"] = _trim_long_result(result)
    return out


def _truncate_decryption(obj: Any) -> None:
    if isinstance(obj, dict) and isinstance(obj.get("decryption"), str):
        obj["decryption"] = obj["decryption"][:_DECRYPTION_LIMIT]


def trim_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    out = dict(artifact)

    # 1. Delete raw-body / ground-truth top-level keys; blank the plan.
    for key in _DELETE_TOP_LEVEL:
        out.pop(key, None)
    out["plan"] = ""

    # 2. messages: retained (timeline/section-4.8 read them) but stripped of raw
    #    prose and over-long tool results.
    if isinstance(out.get("messages"), list):
        out["messages"] = [_trim_message(m) for m in out["messages"]]

    # 3. tool_calls: same over-long-result truncation; structural args kept.
    if isinstance(out.get("tool_calls"), list):
        out["tool_calls"] = [_trim_tool_call(c) for c in out["tool_calls"]]

    # 4. investigation_state: keep ONLY the analyzer allowlist.
    inv = out.get("investigation_state")
    if isinstance(inv, dict):
        out["investigation_state"] = {
            k: v for k, v in inv.items() if k in _INV_STATE_ALLOWLIST
        }

    # 5. branches / automated_preflight: truncate the decryption strings.
    #    ``automated_preflight`` is a NESTED artifact of the same shape (it carries
    #    its own ground_truth/char_accuracy/... fields), so it receives the same
    #    raw-body / ground-truth key deletions as the top level — otherwise its
    #    nested ``ground_truth`` (null here — the preflight never grades against GT)
    #    would trip the §5.3 recursive banned-key firewall the exact top-level
    #    rules were written before. None of the deleted keys are analyzer inputs.
    if isinstance(out.get("branches"), list):
        for branch in out["branches"]:
            _truncate_decryption(branch)
    preflight = out.get("automated_preflight")
    if isinstance(preflight, dict):
        for key in _DELETE_TOP_LEVEL:
            preflight.pop(key, None)
    _truncate_decryption(preflight)

    # 6. episodes: drop raw worker bodies when present (none in this artifact,
    #    but keeps the trimmer reusable).
    if isinstance(out.get("episodes"), list):
        for episode in out["episodes"]:
            if isinstance(episode, dict):
                episode.pop("transcript", None)
                episode.pop("raw_text", None)

    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: python scripts/trim_artifact_fixture.py <src.json> <dst.json>",
            file=sys.stderr,
        )
        return 2
    src = Path(argv[1])
    dst = Path(argv[2])
    with open(src, encoding="utf-8") as f:
        artifact = json.load(f)
    trimmed = trim_artifact(artifact)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {dst} ({dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
