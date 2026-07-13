"""Registry-consistency and dispatch-safety tests for the v2 tool executor.

Covers Phase-0 items 0.3 (tool-dispatch footgun) and 0.5 (allowlist drift).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.tools_v2 import (
    TOOL_DEFINITIONS,
    VALID_TOOL_NAMES,
    WorkspaceToolExecutor,
)
from agent import loop_v2
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


def _executor_for(raw: str = "ABCABC", separator: str | None = None) -> WorkspaceToolExecutor:
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=separator)
    return WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "AND"},
        word_list=["THE", "AND"],
        pattern_dict={},
    )


# ---------------------------------------------------------------------------
# 0.3 — dispatch footgun
# ---------------------------------------------------------------------------

def test_every_tool_definition_has_a_handler_method():
    """Every advertised tool name maps to a `_tool_<name>` method."""
    ex = _executor_for()
    missing = [
        d["name"]
        for d in TOOL_DEFINITIONS
        if not callable(getattr(ex, f"_tool_{d['name']}", None))
    ]
    assert missing == [], f"tool definitions without handlers: {missing}"


def test_every_tool_prefixed_method_is_a_real_tool():
    """No `_tool_`-prefixed helper exists that is not an advertised tool.

    Fails if someone reintroduces a `_tool_`-prefixed internal helper (the
    original footgun was `_tool_tried_for_branch`).
    """
    stray = [
        name[len("_tool_"):]
        for name in dir(WorkspaceToolExecutor)
        if name.startswith("_tool_")
        and name[len("_tool_"):] not in VALID_TOOL_NAMES
    ]
    assert stray == [], f"_tool_-prefixed methods not in TOOL_DEFINITIONS: {stray}"


def test_execute_rejects_internal_helper_name():
    """The old footgun name must be rejected as an unknown tool."""
    ex = _executor_for()
    result = json.loads(ex.execute("tried_for_branch", {}))
    assert result.get("error") == "Unknown tool: tried_for_branch"


def test_execute_rejects_arbitrary_unknown_name():
    ex = _executor_for()
    result = json.loads(ex.execute("definitely_not_a_tool", {}))
    assert result.get("error") == "Unknown tool: definitely_not_a_tool"


def test_gating_takes_precedence_over_unknown_tool_error():
    """When a turn is gated, an unknown name still reports gating, not unknown."""
    ex = _executor_for()
    ex.allowed_tool_names = {"decode_show"}
    result = json.loads(ex.execute("tried_for_branch", {}))
    assert result.get("reason") == "tool_gated"


def test_unknown_tool_call_is_recorded_in_call_log():
    ex = _executor_for()
    before = len(ex.call_log)
    ex.execute("tried_for_branch", {})
    assert len(ex.call_log) == before + 1
    assert ex.call_log[-1].tool_name == "tried_for_branch"


# ---------------------------------------------------------------------------
# 0.5 — allowlist drift
# ---------------------------------------------------------------------------

def test_hardcoded_allowlists_are_subsets_of_tool_definitions():
    valid = {d["name"] for d in TOOL_DEFINITIONS}
    allowlists = {
        "FULL_READING_WORKFLOW_TOOL_NAMES": loop_v2.FULL_READING_WORKFLOW_TOOL_NAMES,
        "FULL_READING_ACTUATOR_TOOL_NAMES": loop_v2.FULL_READING_ACTUATOR_TOOL_NAMES,
        "PENULTIMATE_ALLOWED_TOOL_NAMES": loop_v2.PENULTIMATE_ALLOWED_TOOL_NAMES,
        "FINAL_ALLOWED_TOOL_NAMES": loop_v2.FINAL_ALLOWED_TOOL_NAMES,
        "REPAIR_SANDBOX_TOOL_NAMES": loop_v2.REPAIR_SANDBOX_TOOL_NAMES,
        "INSPECTION_SANDBOX_TOOL_NAMES": loop_v2.INSPECTION_SANDBOX_TOOL_NAMES,
    }
    offenders: dict[str, list[str]] = {}
    for name, names in allowlists.items():
        stale = sorted(set(names) - valid)
        if stale:
            offenders[name] = stale
    assert offenders == {}, f"allowlist entries not in TOOL_DEFINITIONS: {offenders}"
