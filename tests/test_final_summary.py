from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.final_summary import build_final_summary
from artifact.schema import LoopEvent, RunArtifact, SolutionDeclaration, ToolCall


def test_final_summary_uses_agent_reading_for_non_english_run():
    artifact = RunArtifact(
        run_id="run",
        cipher_id="case",
        model="model",
        language="la",
        max_iterations=10,
        status="solved",
    )
    artifact.solution = SolutionDeclaration(
        branch="main",
        rationale="Best available branch.",
        self_confidence=0.91,
        declared_at_iteration=6,
        reading_summary="This is an archaic Latin medical passage about treating poultry.",
        further_iterations_helpful=False,
        further_iterations_note="The remaining differences are minor orthographic uncertainty.",
    )

    summary = build_final_summary(
        artifact,
        final_branch="main",
        final_decryption="ETIAM QUOD IN TALI CUR...",
    )

    assert "Target language: Latin." in summary
    assert "not a certified translation" in summary
    assert "archaic Latin medical passage" in summary
    assert "Probably not essential" in summary
    assert "minor orthographic uncertainty" in summary


def test_final_summary_flags_more_iterations_when_repairs_remain():
    artifact = RunArtifact(
        run_id="run",
        cipher_id="case",
        model="model",
        language="en",
        max_iterations=5,
        status="solved",
    )
    artifact.solution = SolutionDeclaration(
        branch="main",
        rationale="Partial branch with remaining boundary uncertainty.",
        self_confidence=0.62,
        declared_at_iteration=5,
    )
    artifact.repair_agenda.append({
        "id": 1,
        "branch": "main",
        "from": "RLURES",
        "to": "PLURES",
        "status": "held",
    })
    artifact.loop_events.append(LoopEvent(
        event="boundary_projection_count_retry",
        payload={"attempt": 1},
        outer_iteration=4,
    ))

    summary = build_final_summary(
        artifact,
        final_branch="main",
        final_decryption="PARTIAL TEXT",
    )

    assert "Held repair(s): RLURES -> PLURES." in summary
    assert "boundary/reading projection attempt" in summary
    assert "Likely helpful" in summary
    assert "agent confidence is below 0.85" in summary


def test_final_summary_recovers_reading_from_blocked_declaration_attempt():
    artifact = RunArtifact(
        run_id="run",
        cipher_id="case",
        model="model",
        language="la",
        max_iterations=10,
        status="solved",
    )
    artifact.solution = SolutionDeclaration(
        branch="main",
        rationale=(
            "Automatic fallback declaration at iteration limit; the agent did "
            "not call meta_declare_solution in time."
        ),
        self_confidence=0.0,
        declared_at_iteration=10,
    )
    artifact.tool_calls.append(ToolCall(
        iteration=8,
        tool_name="meta_declare_solution",
        tool_use_id="blocked",
        arguments={
            "branch": "main",
            "rationale": "The branch reads as a coherent Latin veterinary text.",
            "self_confidence": 0.78,
            "reading_summary": "This is a medieval Latin veterinary text about sick chickens.",
            "further_iterations_helpful": False,
            "further_iterations_note": "Remaining errors appear structural.",
        },
        result='{"status":"blocked"}',
    ))
    artifact.loop_events.append(LoopEvent(
        event="auto_declared_solution",
        payload={"branch": "main"},
    ))

    summary = build_final_summary(
        artifact,
        final_branch="main",
        final_decryption="ETIAM QUOD...",
    )

    assert "medieval Latin veterinary text about sick chickens" in summary
    assert "Probably not essential" in summary
    assert "Remaining errors appear structural" in summary
    assert "Automatic fallback declaration" not in summary.split("What it appears to say:")[1]
