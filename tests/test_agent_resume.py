from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.loop_v2 import run_v2
from agent.resume import (
    cipher_text_from_artifact,
    install_resume_branches,
    resume_context_from_artifact,
)
from workspace import Workspace


def _artifact(raw: str = "ABC") -> dict:
    return {
        "run_id": "parent123",
        "cipher_id": "resume_case",
        "model": "claude-sonnet-4-6",
        "language": "en",
        "status": "solved",
        "max_iterations": 5,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a digital humanities researcher analyzing a manuscript.\n\n"
                    "## The manuscript notation system\n"
                    "```\n"
                    f"{raw}\n"
                    "```\n"
                ),
            }
        ],
        "solution": {
            "branch": "main",
            "rationale": "Readable but one spelling remains uncertain.",
            "self_confidence": 0.72,
            "declared_at_iteration": 5,
            "reading_summary": "A short English test passage.",
            "further_iterations_helpful": True,
            "further_iterations_note": "Try one targeted spelling repair.",
        },
        "final_summary": "Previous final summary says more iterations may help.",
        "repair_agenda": [
            {
                "id": 3,
                "branch": "main",
                "from": "TEH",
                "to": "THE",
                "status": "held",
                "notes": "Needs rereading.",
            }
        ],
        "branches": [
            {
                "name": "main",
                "parent": None,
                "created_iteration": 0,
                "key": {"0": 19, "1": 7, "2": 4},
                "mapped_count": 3,
                "decryption": "THE",
                "tags": ["prior"],
            },
            {
                "name": "alt",
                "parent": "main",
                "created_iteration": 4,
                "key": {"0": 19},
                "mapped_count": 1,
                "decryption": "T??",
                "tags": ["candidate"],
            },
        ],
    }


def test_resume_context_tells_agent_not_to_restart():
    context = resume_context_from_artifact(
        _artifact(),
        branch=None,
        extra_iterations=4,
    )

    assert "continuation" in context
    assert "Do not restart from scratch" in context
    assert "Suggested starting branch: `main`" in context
    assert "Try one targeted spelling repair" in context
    assert "TEH -> THE" in context


def test_resume_restores_saved_branches_and_keys():
    prior = _artifact()
    ct = cipher_text_from_artifact(prior)
    ws = Workspace(ct)

    restored = install_resume_branches(ws, prior)

    assert restored == ["main", "alt"]
    assert ws.apply_key("main") == "THE"
    assert ws.apply_key("alt") == "T??"
    assert "candidate" in ws.get_branch("alt").tags


class _DeclareAPI:
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        self.messages_seen = []

    def send_message(self, messages, tools=None, system="", max_tokens=4096):
        self.messages_seen.append(messages)
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="declare",
                    name="meta_declare_solution",
                    input={
                        "branch": "main",
                        "rationale": "Continuing from the restored branch was sufficient.",
                        "self_confidence": 0.9,
                        "reading_summary": "The restored branch reads as THE.",
                        "further_iterations_helpful": False,
                        "further_iterations_note": "No further work is needed in this toy case.",
                    },
                )
            ],
        )


def test_run_v2_resume_uses_restored_state_without_preflight():
    prior = _artifact()
    prior["branches"] = prior["branches"][:1]
    ct = cipher_text_from_artifact(prior)
    api = _DeclareAPI()

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,  # type: ignore[arg-type]
        language="en",
        max_iterations=1,
        cipher_id="resume_case",
        prior_context=resume_context_from_artifact(
            prior,
            branch=None,
            extra_iterations=1,
        ),
        resume_from_artifact=prior,
        parent_artifact_path="/tmp/parent.json",
    )

    assert artifact.status == "solved"
    assert artifact.parent_run_id == "parent123"
    assert artifact.parent_artifact_path == "/tmp/parent.json"
    assert artifact.automated_preflight is None
    assert artifact.repair_agenda[0]["id"] == 3
    assert next(b for b in artifact.branches if b.name == "main").decryption == "THE"
    first_message = api.messages_seen[0][0]["content"]
    assert "Do not restart from scratch" in first_message
