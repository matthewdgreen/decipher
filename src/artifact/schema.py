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
    episode_id: str | None = None  # set when the call ran inside a v3 episode (M2)


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
    metadata: dict[str, Any] = field(default_factory=dict)
    word_spans: list[tuple[int, int]] | None = None
    token_order: list[int] | None = None
    transform_pipeline: dict[str, Any] | None = None
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
    reading_summary: str = ""
    further_iterations_helpful: bool | None = None
    further_iterations_note: str = ""
    # M5 (v3): the verify AttestationRecord (dict) that gated this declaration.
    # None for v2 declarations and for fallback/auto declarations. A weak-but-
    # declared solve carries its coherence/anomalies here so it is visibly weak.
    attestation: dict[str, Any] | None = None


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
    # Configured v3 pre-send cost cutoff. None means uncapped. A provider call
    # already admitted below the cutoff can put final billed cost above it; the
    # artifact/analyzer expose that last-call overshoot explicitly.
    max_cost_usd: float | None = None
    automated_preflight: dict[str, Any] | None = None
    cipher_id_report: dict[str, Any] | None = None
    benchmark_context: dict[str, Any] | None = None
    parent_run_id: str = ""
    parent_artifact_path: str = ""

    # What the agent produced
    plan: str = ""                              # first-turn text (or extended-thinking trace)
    notebook: list[NotebookEntry] = field(default_factory=list)
    branches: list[BranchSnapshot] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_requests: list[dict[str, Any]] = field(default_factory=list)  # meta_request_tool calls
    subagent_runs: list[SubagentRun] = field(default_factory=list)
    loop_events: list[LoopEvent] = field(default_factory=list)
    repair_agenda: list[dict[str, Any]] = field(default_factory=list)
    cipher_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)  # full message history

    # Termination
    solution: SolutionDeclaration | None = None
    status: str = "running"                     # running | solved | unsolved | exhausted | error | stopped | fallback_declared (positive attestation only)
    auto_declared: bool = False                 # True when the solution was synthesized by the fallback path
    attested_fallback: bool = False              # v3 fallback selected a fresh positive attestation
    fallback_selection: dict[str, Any] | None = None  # v3 fallback tier/shortlist/rationale
    error_message: str = ""
    final_summary: str = ""

    # Token usage (accumulated across all API calls in this run)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # Served-model / safety-gate provenance (schema-additive; eval integrity).
    # ``model`` above is the REQUESTED model. ``served_models`` records the
    # distinct model ids actually served across all API calls (from
    # ``ModelResponse.raw.model``, when the provider exposes it), in first-seen
    # order. ``safety_gate_fired`` is True when any served model does not match
    # the requested model (e.g. requested claude-fable-5, served claude-opus-*)
    # — such runs are excluded from the model-comparison table as contaminated.
    served_models: list[str] = field(default_factory=list)
    safety_gate_fired: bool = False

    # Post-hoc scoring (filled by benchmark runner against ground truth)
    ground_truth: str | None = None
    char_accuracy: float | None = None
    word_accuracy: float | None = None
    preprocessing_applied: dict[str, Any] | None = None

    # Loop provenance and v3 additions (schema-additive; v2 artifacts default
    # to loop_version="v2" and empty/None for the rest).
    loop_version: str = "v2"                     # "v2" | "v3"
    budget_by_category: dict[str, Any] = field(default_factory=dict)
    session_transcript: dict[str, Any] | None = None
    investigation_state: dict[str, Any] | None = None
    episodes: list[dict[str, Any]] = field(default_factory=list)  # v3 episode ledger (M2)
    readings: list[dict[str, Any]] = field(default_factory=list)  # v3 stored Readings (M3)
    experiments: list[dict[str, Any]] = field(default_factory=list)  # v3 experiment queue (M4)
    attestations: list[dict[str, Any]] = field(default_factory=list)  # v3 verify attestations (M5)
    # M5.3 Slice 7: the four distinguished branch roles at termination
    # (best_scored_branch / workflow_branch / latest_installed_branch /
    # declared_or_selected_branch). Branch names only. None for v2 runs.
    branch_roles: dict[str, Any] | None = None

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
