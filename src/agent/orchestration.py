"""State and mode policy for the redesigned agent loop.

This module is deliberately behavior-light.  It names the concepts the new
loop will coordinate, while the existing v2 loop continues to own execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentMode(str, Enum):
    """High-level tool-use modes for the provider-neutral agent harness."""

    EXPLORE = "explore"
    READING_REPAIR = "reading_repair"
    BOUNDARY_PROJECTION = "boundary_projection"
    POLISH = "polish"
    DECLARE = "declare"


MODE_ALLOWED_TOOLS: dict[AgentMode, frozenset[str]] = {
    AgentMode.EXPLORE: frozenset({
        "observe_frequency",
        "observe_patterns",
        "decode_show",
        "decode_letter_stats",
        "score_panel",
        "score_dictionary",
        "workspace_list_branches",
        "workspace_branch_cards",
        "workspace_compare",
    }),
    AgentMode.READING_REPAIR: frozenset({
        "decode_show",
        "decode_letter_stats",
        "decode_plan_word_repair",
        "decode_plan_word_repair_menu",
        "decode_repair_no_boundary",
        "repair_agenda_list",
        "repair_agenda_update",
        "corpus_word_candidates",
        "act_set_mapping",
        "act_apply_word_repair",
        "act_bulk_set",
        "workspace_fork",
        "workspace_branch_cards",
        "workspace_compare",
        "score_panel",
        "score_dictionary",
    }),
    AgentMode.BOUNDARY_PROJECTION: frozenset({
        "decode_show",
        "score_panel",
        "score_dictionary",
        "workspace_branch_cards",
        "decode_plan_word_repair",
        "decode_plan_word_repair_menu",
        "act_apply_word_repair",
        "repair_agenda_list",
        "repair_agenda_update",
        "decode_validate_reading_repair",
        "act_resegment_by_reading",
        "act_resegment_from_reading_repair",
        "act_resegment_window_by_reading",
    }),
    AgentMode.POLISH: frozenset({
        "decode_show",
        "score_panel",
        "score_dictionary",
        "search_anneal",
        "search_homophonic_anneal",
        "search_automated_solver",
    }),
    AgentMode.DECLARE: frozenset({"meta_declare_solution"}),
}


@dataclass
class RepairAgendaItem:
    """One reading-driven repair hypothesis the loop is tracking."""

    branch: str
    hypothesis: str
    status: str = "open"
    evidence: str = ""
    tool_name: str | None = None
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStatus:
    """Per-branch workflow completion and notes."""

    name: str
    branch: str
    status: str = "not_started"
    started_outer_iteration: int | None = None
    finished_outer_iteration: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class AgentRunState:
    """Explicit run state beyond provider message history."""

    active_mode: AgentMode = AgentMode.EXPLORE
    active_branch: str = "main"
    outer_iteration: int = 0
    inner_step: int = 0
    repair_agenda: list[RepairAgendaItem] = field(default_factory=list)
    attempted_repairs: list[dict[str, Any]] = field(default_factory=list)
    held_repairs: list[dict[str, Any]] = field(default_factory=list)
    reverted_repairs: list[dict[str, Any]] = field(default_factory=list)
    unresolved_hypotheses: list[str] = field(default_factory=list)
    workflows: dict[str, WorkflowStatus] = field(default_factory=dict)
    latest_branch_cards: dict[str, dict[str, Any]] = field(default_factory=dict)

    def allowed_tools(self) -> frozenset[str]:
        return MODE_ALLOWED_TOOLS[self.active_mode]

    def set_mode(self, mode: AgentMode, *, branch: str | None = None) -> None:
        self.active_mode = mode
        if branch is not None:
            self.active_branch = branch
        self.inner_step = 0

    def workflow_key(self, name: str, branch: str | None = None) -> str:
        return f"{branch or self.active_branch}:{name}"

    def mark_workflow_started(self, name: str, branch: str | None = None) -> None:
        branch_name = branch or self.active_branch
        key = self.workflow_key(name, branch_name)
        status = self.workflows.get(key) or WorkflowStatus(name=name, branch=branch_name)
        status.status = "running"
        status.started_outer_iteration = self.outer_iteration
        self.workflows[key] = status

    def mark_workflow_finished(
        self,
        name: str,
        *,
        branch: str | None = None,
        status: str = "completed",
        note: str | None = None,
    ) -> None:
        branch_name = branch or self.active_branch
        key = self.workflow_key(name, branch_name)
        workflow = self.workflows.get(key) or WorkflowStatus(name=name, branch=branch_name)
        workflow.status = status
        workflow.finished_outer_iteration = self.outer_iteration
        if note:
            workflow.notes.append(note)
        self.workflows[key] = workflow
