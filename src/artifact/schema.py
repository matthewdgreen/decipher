"""Dataclasses defining the shape of a v2 run artifact."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    """One tool invocation and its return value."""
    iteration: int
    tool_name: str
    tool_use_id: str
    arguments: dict[str, Any]
    result: str                # raw JSON string returned to the model
    elapsed_ms: int = 0
    parent_tool_use_id: str | None = None  # non-None if invoked from a subagent


@dataclass
class NotebookEntry:
    """A structured finding written by the agent."""
    id: int
    iteration: int
    claim: str
    evidence: str = ""
    confidence: float = 0.5                    # 0.0–1.0
    tags: list[str] = field(default_factory=list)
    status: str = "open"                        # open | confirmed | rejected
    # Agents may cross-link findings; ids here refer to other NotebookEntry.id
    supersedes: list[int] = field(default_factory=list)


@dataclass
class BranchSnapshot:
    """Final state of a single branch at the end of a run."""
    name: str
    parent: str | None
    created_iteration: int
    key: dict[int, int]                         # ct token id -> pt token id
    mapped_count: int
    decryption: str                             # result of applying key
    signals: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    char_accuracy: float | None = None   # vs ground truth (filled post-hoc by runner)
    word_accuracy: float | None = None


@dataclass
class SubagentRun:
    """Record of one sub-agent invocation from the parent."""
    id: str                                     # "sub_<n>"
    parent_iteration: int
    mission: str
    tool_whitelist: list[str]
    branch_scope: str | None                    # branch the subagent was told to work in
    iterations_used: int
    summary: str                                # returned to parent
    result: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass
class SolutionDeclaration:
    """Payload of meta_declare_solution."""
    branch: str
    rationale: str
    self_confidence: float                      # 0.0–1.0, agent's own assessment
    declared_at_iteration: int


@dataclass
class LoopEvent:
    """Structured agent-loop event for future inner-loop observability."""

    event: str
    payload: dict[str, Any]
    outer_iteration: int | None = None
    inner_step: int | None = None
    mode: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RunArtifact:
    """Everything about one run — the research datum."""
    run_id: str
    cipher_id: str                              # benchmark test_id or user-supplied
    model: str
    language: str
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    # Set-up information
    cipher_alphabet_size: int = 0
    cipher_token_count: int = 0
    cipher_word_count: int = 0
    max_iterations: int = 0
    automated_preflight: dict[str, Any] | None = None

    # What the agent produced
    plan: str = ""                              # first-turn text (or extended-thinking trace)
    notebook: list[NotebookEntry] = field(default_factory=list)
    branches: list[BranchSnapshot] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_requests: list[dict[str, Any]] = field(default_factory=list)  # meta_request_tool calls
    subagent_runs: list[SubagentRun] = field(default_factory=list)
    loop_events: list[LoopEvent] = field(default_factory=list)
    repair_agenda: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)  # full message history

    # Termination
    solution: SolutionDeclaration | None = None
    status: str = "running"                     # running | solved | exhausted | error | stopped
    error_message: str = ""

    # Token usage (accumulated across all API calls in this run)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Post-hoc scoring (filled by benchmark runner against ground truth)
    ground_truth: str | None = None
    char_accuracy: float | None = None
    word_accuracy: float | None = None
    preprocessing_applied: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        def convert(o: Any) -> Any:
            if hasattr(o, "__dataclass_fields__"):
                return {k: convert(getattr(o, k)) for k in o.__dataclass_fields__}
            if isinstance(o, dict):
                return {str(k): convert(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [convert(x) for x in o]
            return o
        return convert(self)  # type: ignore[no-any-return]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
