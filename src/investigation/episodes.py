"""Episode runtime: fresh-context workers the lead delegates to (M2 spec C3).

An *episode* is a single-purpose worker (survey / search / reading / compare)
that runs in an ISOLATED copy of a few branches, with a HARD-ALLOWLISTED toolset,
its own budget, and no view of benchmark context or the hypothesis board. It
returns a structured result plus branch snapshots; nothing it does touches the
lead's live workspace until the lead explicitly installs a snapshot (A1).

Key pieces:
- ``EpisodeSpec`` / ``EpisodeBudget`` / ``EpisodeResult`` dataclasses.
- ``EPISODE_KINDS`` v1 registry (toolset, budget, result schema, contract).
- ``run_episode`` — the fresh-context worker loop with A9 failure semantics.
- ``build_episode_context`` — deterministic, capped worker context.
- ``episode_run`` / ``episode_install_branch`` lead tool defs and
  ``V3_LEAD_TOOL_DEFINITIONS``.
"""
from __future__ import annotations

import copy
import json
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.loop_shared import _branch_snapshot_for, _decoded_text_for_panel
from agent.model_provider import _collect_assistant_blocks
from agent.tools_v2 import (
    TOOL_DEFINITIONS,
    VALID_TOOL_NAMES,
    NoGatesPolicy,
    WorkspaceToolExecutor,
    _json,
)
from investigation.actions import (
    BRANCH_ADJUDICATE_TOOL,
    COMPOSITE_TOOL_DEFINITIONS,
    COMPOSITE_TOOL_NAMES,
    execute_composite,
)
from investigation.sessions import session_factory
from investigation.state import (
    BudgetEntry,
    InvestigationState,
    _restore_branch_into,
    _serialize_branch,
)
from workspace import Workspace


def _compact_episode_args(args: dict[str, Any]) -> dict[str, Any]:
    """Shrink a tool's args for the forwarded ``episode_tool_call`` event (F-3).

    The full args already live in ``artifact.tool_calls`` (the episode's
    ToolCall log, merged at finalize), so storing them again in
    ``artifact.loop_events`` is ~2x duplication that balloons for arg-heavy
    tools (``act_bulk_set`` mappings, ``run_python`` code). Spec decision 3
    pinned "name + compact arg summary" for this stream. This keeps a small,
    dict-shaped summary (the narrate renderer compacts further at render, so the
    on-screen output is unchanged): long strings truncated, big containers
    replaced by a size marker, at most a handful of keys.
    """
    if not isinstance(args, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in list(args.items())[:6]:
        if isinstance(value, str) and len(value) > 40:
            out[key] = value[:39] + "…"
        elif isinstance(value, (dict, list, tuple)) and len(value) > 8:
            out[key] = f"<{type(value).__name__}[{len(value)}]>"
        else:
            out[key] = value
    if len(args) > 6:
        out["…"] = f"+{len(args) - 6} more"
    return out


# ---------------------------------------------------------------------------
# Excluded tools (A9): long-running tools go to the M4 experiment queue, never
# to an episode (wall clock is checked only *between* tool calls).
# ---------------------------------------------------------------------------
EPISODE_EXCLUDED_TOOLS = frozenset({
    "search_automated_solver",
    "search_transform_candidates",
    "search_transform_homophonic",
    "search_quagmire3_keyword_alphabet",
})

# The five hypothesis handlers — episodes never write the board.
_HYPOTHESIS_HANDLERS = frozenset({
    "workspace_create_hypothesis_branch",
    "workspace_update_hypothesis",
    "workspace_reject_hypothesis",
    "workspace_hypothesis_cards",
    "workspace_hypothesis_next_steps",
})

# Tool-name prefixes an episode may never include.
_FORBIDDEN_PREFIXES = ("meta_", "inspect_", "list_")

# search_tool → its review/rate/install companions (module constant; Part 4).
_SEARCH_COMPANIONS: dict[str, list[str]] = {
    "search_word_repair_menu": [
        "search_review_word_repair_finalists",
        "act_install_word_repair_finalists",
    ],
    "search_pure_transposition": [
        "search_review_pure_transposition_finalists",
        "act_rate_transform_finalist",
        "act_install_pure_transposition_finalists",
    ],
}

_OBSERVE_TOOLS = frozenset(t for t in VALID_TOOL_NAMES if t.startswith("observe_"))
_SCORE_TOOLS = frozenset(t for t in VALID_TOOL_NAMES if t.startswith("score_"))


# ---------------------------------------------------------------------------
# Result schemas (local validator dialect: type/properties/required/items/enum,
# nullable via a type list e.g. ["string", "null"]).
# ---------------------------------------------------------------------------
_SURVEY_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": {"type": "string"}},
        "suspected_modes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "mode": {"type": "string"},
                "confidence": {"type": ["string", "number"]},
                "evidence": {"type": "string"},
            },
            "required": ["mode"],
        }},
        "recommended_next": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["findings", "suspected_modes", "recommended_next"],
}

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "improved": {"type": "boolean"},
        "best_branch": {"type": ["string", "null"]},
        "score_summary": {"type": "object"},
        "finalist_session_id": {"type": ["string", "null"]},
        "notes": {"type": "string"},
    },
    "required": ["improved", "best_branch", "score_summary", "notes"],
}

_READING_SCHEMA = {
    "type": "object",
    "properties": {
        "reading_text": {"type": "string"},
        "fragments": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "window": {"type": "string"},
                "span_id": {"type": ["string", "null"]},
                # A8: optional nullable-integer token indices alongside the M2
                # `window` string. `required` stays ["text"] (additive schema).
                "start": {"type": ["integer", "null"]},
                "end": {"type": ["integer", "null"]},
                "text": {"type": "string"},
                "repair_text": {"type": ["string", "null"]},
                "confidence": {"type": ["number", "string"]},
            },
            # M5.1 review fix, SOFTENED after Stage-1 forensics: schema-requiring
            # confidence would fail the WHOLE episode when a worker omits it —
            # a worse outcome than degrading one fragment. Omitted confidence
            # instead defaults to 0.5 in Reading.from_episode_result: BELOW the
            # MIN_REPAIR_FRAGMENT_CONFIDENCE threshold, so a silent worker's
            # fragment stays visible but cannot change the key automatically.
            "required": ["text"],
        }},
        "holes": {"type": "array", "items": {"type": "string"}},
        "overall_confidence": {"type": "number"},
    },
    "required": ["reading_text", "fragments", "holes", "overall_confidence"],
}

# M3 Part 6: the `repair` episode result schema.
_REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "applied": {"type": "boolean"},
        "best_branch": {"type": ["string", "null"]},
        "edits": {"type": "array", "items": {"type": "string"}},
        "verdicts": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "target": {"type": "string"},
                "verdict": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["action", "target", "verdict"],
        }},
        "collateral": {"type": "object"},
        "notes": {"type": "string"},
    },
    "required": ["applied", "best_branch", "edits", "verdicts", "notes"],
}

_COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "string"}},
        "verdicts": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "verdict": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["branch", "verdict"],
        }},
        "winner": {"type": ["string", "null"]},
    },
    "required": ["ranking", "verdicts", "winner"],
}

# M5 Part 1: the `verify` episode result schema. ``coherence`` is an int on a
# 0-10 scale. The local validator has no min/max support, so the scale is
# stated to the worker twice — in the contract prose and in the field
# ``description`` below (the validator ignores ``description``; providers
# render it in the tool schema) — and the dispatcher treats out-of-range
# values as scale violations (see loop_v3._clamp_coherence).
_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "coherence": {
            "type": "integer",
            "description": "0 (gibberish) to 10 (fluent, natural text)",
        },
        "reader_accepts": {"type": "boolean"},
        "gloss": {"type": "string"},
        "anomalies": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["coherence", "reader_accepts", "gloss", "anomalies", "confidence"],
}


@dataclass
class EpisodeBudget:
    max_tool_calls: int
    max_output_tokens: int
    wall_clock_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tool_calls": self.max_tool_calls,
            "max_output_tokens": self.max_output_tokens,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


# kind → registry entry. ``toolset`` is the STATIC part; the search kind extends
# it per ``inputs["search_tool"]``.
EPISODE_KINDS: dict[str, dict[str, Any]] = {
    "survey": {
        "toolset": frozenset({*_OBSERVE_TOOLS, *_SCORE_TOOLS, "decode_show"}),
        "budget": EpisodeBudget(10, 2048, 120.0),
        "result_schema": _SURVEY_SCHEMA,
        "tier": "diagnostic",
        "contract": (
            "You are a SURVEY worker. Diagnose this cipher: measure it with the "
            "observe_* and score_* tools and decode_show, then report findings, "
            "suspected cipher modes with evidence, and recommended next actions."
        ),
    },
    "search": {
        "toolset": frozenset({"decode_show", "score_panel"}),
        "budget": EpisodeBudget(8, 2048, 300.0),
        "result_schema": _SEARCH_SCHEMA,
        "tier": "search",
        "contract": (
            "You are a SEARCH worker. Run the assigned search tool (and its "
            "review/rate/install companions) on the given branch, read the "
            "decode, and report whether the branch improved, the best branch, a "
            "score summary, any finalist session id, and notes."
        ),
    },
    "reading": {
        "toolset": frozenset({
            "decode_show", "decode_letter_stats", "decode_ambiguous_letter",
            "decode_absent_letter_candidates", "decode_unmapped_report",
            "corpus_lookup_word", "corpus_word_candidates",
            "score_panel", "score_dictionary",
        }),
        # M5.1 reading envelope, restored by M5.3 Slice 1 (commit 1d01ef4 set
        # 16/8192/300; the M5.2 reduction to 4/4096/180 was never validated —
        # the per-batch budget-check defect let 4-call readings execute as many
        # as thirteen calls, so that smoke is not evidence the smaller budget is
        # usable). This is a MAXIMUM safety envelope, not a target: the
        # early-partial-reading contract stays in force, the duplicate/saturation
        # policy bounds how many readings run for unchanged evidence, and the
        # per-run cost ceiling is the final paid guard. Do not lower it below this
        # until a binding 4/8/12/16 calibration shows reading quality is
        # preserved.
        "budget": EpisodeBudget(16, 8192, 300.0),
        "result_schema": _READING_SCHEMA,
        "tier": "reading",
        "contract": (
            "You are a READING worker. Draft the best plain-language reading of "
            "the given branch's decode: report the reading text, per-window "
            "fragments with confidence, remaining holes, and an overall "
            "confidence. Your budget is small: prefer submitting a partial "
            "reading early over running more tools — a submitted reading with "
            "honest confidence beats an exhausted budget. "
            "The host gives you opaque span ids. Reference `span_id` exactly; "
            "never count or invent numeric offsets. Keep `text` human-readable. "
            "When a fragment is confident "
            "enough to drive key repair, also provide `repair_text` using only "
            "plaintext letters, spaces, and `?` for one unknown token. You do NOT "
            "change the key — you only read it."
        ),
    },
    "compare": {
        "toolset": frozenset({
            "decode_show", "score_panel", "score_dictionary", "workspace_compare",
            "branch_adjudicate",
        }),
        "budget": EpisodeBudget(8, 2048, 120.0),
        "result_schema": _COMPARE_SCHEMA,
        "tier": "compare",
        "contract": (
            "You are a COMPARE worker. Rank the given branches by how well their "
            "decodes read as the target language; report a ranking, a verdict and "
            "rationale per branch, and the winner (or null if none is convincing)."
        ),
    },
    "repair": {
        "toolset": frozenset({
            "hypothesis_test_word", "hypothesis_apply_reading", "branch_adjudicate",
            "decode_show", "decode_letter_stats", "decode_unmapped_report",
            "corpus_lookup_word", "corpus_word_candidates",
            "score_panel", "score_dictionary",
        }),
        "budget": EpisodeBudget(12, 4096, 240.0),
        "result_schema": _REPAIR_SCHEMA,
        "tier": "repair",
        "contract": (
            "You are a REPAIR worker. Compile the given reading / word hypotheses "
            "into applied edits on FORKS with hypothesis_apply_reading and "
            "hypothesis_test_word, verify collateral with branch_adjudicate and "
            "the score/decode tools, discard unsupported edits, keep only a small "
            "diverse finalist set, and compare it before naming exactly one best "
            "changed branch. Report which edits you applied, per-action verdicts, "
            "collateral summaries, and notes. Nothing you fork lands until the "
            "host repair transaction validates and installs it."
        ),
    },
    # M5 Part 1: the independent-reader `verify` episode. EMPTY toolset (text-only
    # judgement) and a one-send budget (wall clock is the real bound; F5). The
    # ``{language}`` placeholder is formatted in ``_episode_system_prompt`` after
    # the lead resolves the run language (contracts are otherwise static). No
    # cryptanalysis framing (avoids the refusal the INV experiment hit) and no
    # scores to anchor on — the whole value is independence.
    "verify": {
        "toolset": frozenset(),
        "budget": EpisodeBudget(1, 1024, 90.0),
        "result_schema": _VERIFY_SCHEMA,
        "tier": "verify",
        "contract": (
            "You are a fluent reader of {language}. Below is a candidate "
            "decipherment of a historical manuscript. Judge ONLY whether it "
            "reads as real (possibly damaged or partial) {language}: gloss what "
            "it says (a clause-level paraphrase), list anomalies (non-words, "
            "broken syntax spans, wrong-language runs), rate coherence as an "
            "integer from 0 (gibberish) to 10 (fluent, natural {language}), and "
            "state whether a fluent reader would accept it as genuine, if "
            "damaged, text. You have no other information and need none — do "
            "not ask for the cipher, the key, or any score."
        ),
    },
}


def episode_toolset_for(kind: str, inputs: dict[str, Any]) -> set[str]:
    """Return the concrete toolset for ``kind`` given ``inputs`` (Part 4)."""
    entry = EPISODE_KINDS.get(kind)
    if entry is None:
        raise ValueError(f"Unknown episode kind: {kind!r}")
    toolset = set(entry["toolset"])
    if kind == "search":
        search_tool = str(inputs.get("search_tool") or "").strip()
        if search_tool:
            toolset.add(search_tool)
            toolset.update(_SEARCH_COMPANIONS.get(search_tool, []))
    return toolset


def _validate_episode_toolset(toolset: set[str]) -> None:
    for name in sorted(toolset):
        # M3: v3 composite actions are valid episode tools (they clear the
        # _FORBIDDEN_PREFIXES / excluded / hypothesis-handler checks by
        # construction, so accept them before the v2 allowlist check).
        if name in COMPOSITE_TOOL_NAMES:
            continue
        if name not in VALID_TOOL_NAMES:
            raise ValueError(f"episode toolset has unknown tool: {name!r}")
        if name in EPISODE_EXCLUDED_TOOLS:
            raise ValueError(f"episode toolset includes an excluded tool: {name!r}")
        if name in _HYPOTHESIS_HANDLERS:
            raise ValueError(f"episode toolset includes a hypothesis handler: {name!r}")
        if name.startswith(_FORBIDDEN_PREFIXES):
            raise ValueError(f"episode toolset includes a forbidden tool: {name!r}")


@dataclass
class EpisodeSpec:
    kind: str
    goal: str
    inputs: dict[str, Any] = field(default_factory=dict)
    toolset: list[str] = field(default_factory=list)
    budget: EpisodeBudget | None = None
    result_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EPISODE_KINDS:
            raise ValueError(f"Unknown episode kind: {self.kind!r}")
        entry = EPISODE_KINDS[self.kind]
        if not self.toolset:
            self.toolset = sorted(episode_toolset_for(self.kind, self.inputs))
        if self.budget is None:
            self.budget = entry["budget"]
        if not self.result_schema:
            self.result_schema = entry["result_schema"]
        # Optional per-call override of the call cap (M5.3 Slice 1). The
        # model-facing `max_tool_calls` may LOWER the registered host budget but
        # must never RAISE it — a worker cannot vote itself a larger budget.
        # Clamp to 1..registered_max (the pre-override cap: the kind default, or
        # an explicitly-passed budget). Non-positive integers floor to 1 (via the
        # max(1, ...) below); UNPARSEABLE requests ("all", inf, a dict) are
        # IGNORED (keep the registered budget). Either way it never raises above
        # the registered cap.
        max_calls = self.inputs.get("max_tool_calls")
        if max_calls is not None:
            registered_max = self.budget.max_tool_calls
            try:
                requested = int(max_calls)
            except (TypeError, ValueError, OverflowError):
                requested = registered_max
            clamped = max(1, min(requested, registered_max))
            self.budget = EpisodeBudget(
                clamped, self.budget.max_output_tokens,
                self.budget.wall_clock_seconds,
            )
        _validate_episode_toolset(set(self.toolset))

    @property
    def contract(self) -> str:
        return EPISODE_KINDS[self.kind]["contract"]


@dataclass
class EpisodeResult:
    episode_id: str
    kind: str
    goal: str
    status: str  # "ok" | "episode_failed"
    failure_reason: str | None = None
    result: Any = None
    summary: str = ""
    branch_snapshots: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    # M5.3 Slice 1: count of over-budget tool_uses the host skipped (executed no
    # tool for, synthesizing a paired budget_exhausted result instead). Makes the
    # per-batch overshoot visible in the episode ledger / analyzer.
    suppressed_over_budget_calls: int = 0
    elapsed_seconds: float = 0.0
    budget_entries: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str | None = None
    transcript: dict[str, Any] | None = None
    # Episode-local repair-agenda items (A10): the episode executor's private
    # agenda rides in the packet; the lead merges it into state.repair_agenda
    # on episode_install_branch.
    agenda_additions: list[dict[str, Any]] = field(default_factory=list)
    # Full episode ToolCalls — merged into the artifact by the lead dispatcher
    # (F10), NOT serialized into the compact ledger dict.
    tool_calls: list[Any] = field(default_factory=list)

    def ledger_dict(self, *, include_transcript: bool) -> dict[str, Any]:
        out = {
            "episode_id": self.episode_id,
            "kind": self.kind,
            "goal": self.goal,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "result": self.result,
            "summary": self.summary,
            "branch_snapshots": self.branch_snapshots,
            "tool_call_count": self.tool_call_count,
            "suppressed_over_budget_calls": self.suppressed_over_budget_calls,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "budget_entries": self.budget_entries,
            "agenda_additions": self.agenda_additions,
        }
        if self.raw_text is not None:
            out["raw_text"] = self.raw_text
        if include_transcript and self.transcript is not None:
            out["transcript"] = self.transcript
        return out


# ---------------------------------------------------------------------------
# Local schema validator (no jsonschema dependency)
# ---------------------------------------------------------------------------
def _type_matches(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    return False


def _as_type_list(types: Any) -> list[str]:
    if types is None:
        return []
    return list(types) if isinstance(types, list) else [types]


def validate_against_schema(value: Any, schema: dict[str, Any], path: str = "result") -> list[str]:
    errors: list[str] = []
    types = _as_type_list(schema.get("type"))
    if types and not any(_type_matches(value, t) for t in types):
        errors.append(f"{path}: expected type {types}, got {type(value).__name__}")
        return errors
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        errors.append(f"{path}: {value!r} is not one of {enum}")
    if isinstance(value, dict) and ("object" in types or not types):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}.{key}: required field missing")
        for key, subschema in props.items():
            if key in value:
                errors.extend(validate_against_schema(value[key], subschema, f"{path}.{key}"))
    if isinstance(value, list) and ("array" in types or not types):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                errors.extend(validate_against_schema(item, item_schema, f"{path}[{i}]"))
    return errors


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Episode tool definitions
# ---------------------------------------------------------------------------
EPISODE_RUN_TOOL = {
    "name": "episode_run",
    "description": (
        "Delegate a focused sub-task to an isolated worker episode. Kinds: "
        "survey (diagnose), search (run one search tool on a branch), reading "
        "(draft a reading), compare (rank branches), repair (compile a "
        "reading/word-hypotheses into edits on a fork), verify (an independent "
        "fresh reader judges whether a branch's decode reads as real target-"
        "language text — required before declaring a solution). Runs "
        "synchronously in a copy of the named branches and returns a result "
        "summary + snapshots; install a snapshot with episode_install_branch to "
        "adopt it. A verify episode reads only the candidate plaintext and "
        "writes an attestation the lead needs to declare."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(EPISODE_KINDS.keys())},
            "goal": {"type": "string"},
            "branches": {
                "type": "array", "items": {"type": "string"},
                "description": (
                    "Branch name(s) the episode operates on. A `verify` episode "
                    "requires EXACTLY ONE existing branch (the candidate to attest)."
                ),
            },
            "search_tool": {"type": "string"},
            "context_note": {"type": "string"},
            "max_tool_calls": {"type": "integer"},
            # A1/M3: hand a stored Reading to a (repair) episode by id.
            "reading_id": {"type": "string"},
        },
        "required": ["kind", "goal", "branches"],
    },
}

EPISODE_INSTALL_TOOL = {
    "name": "episode_install_branch",
    "description": (
        "Adopt a branch snapshot produced by an episode into your workspace "
        "under a fresh name. Nothing from an episode changes your workspace "
        "until you install it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "episode_id": {"type": "string"},
            "branch": {"type": "string"},
            "as_name": {"type": "string"},
        },
        "required": ["episode_id", "branch"],
    },
}

REPAIR_TRANSACTION_TOOL = {
    "name": "repair_transaction",
    "description": (
        "Run one bounded repair transaction for a candidate. The host binds the "
        "candidate content and stored reading, delegates conservative edits to "
        "an isolated repair worker, validates its supported winner, installs the "
        "changed snapshot, and requires fresh verification."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "branch": {"type": "string"},
            "reading_id": {
                "type": "string",
                "description": (
                    "Stored reading to compile. Omit to use the newest reading "
                    "bound to this exact branch content."
                ),
            },
            "goal": {"type": "string"},
            "as_name": {"type": "string"},
        },
        "required": ["branch"],
    },
}

# The lead is a strategist, not an operator. Low-level observation, search, and
# mapping tools remain in the episode library but are intentionally absent from
# this surface. Benchmark-context reads are included dynamically by
# ``v3_lead_tool_definitions`` only when such context exists.
_V3_STRATEGIC_V2_TOOLS = frozenset({
    "workspace_create_hypothesis_branch",
    "workspace_update_hypothesis",
    "workspace_reject_hypothesis",
    "workspace_hypothesis_next_steps",
    "decode_show",
    "repair_agenda_list",
    "repair_agenda_update",
    "act_set_model_variant",
    "meta_request_tool",
    "meta_declare_solution",
    "meta_declare_unsolved",
})
_V3_CONTEXT_TOOLS = frozenset({
    "inspect_benchmark_context",
    "list_related_records",
    "inspect_related_transcription",
    "inspect_related_solution",
    "list_associated_documents",
    "inspect_associated_document",
})


def v3_lead_tool_definitions(
    *,
    include_context_tools: bool = False,
    allowed_episode_kinds: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Return the bounded lead surface; workers still receive scoped v2 tools."""
    names = set(_V3_STRATEGIC_V2_TOOLS)
    if include_context_tools:
        names.update(_V3_CONTEXT_TOOLS)
    selected = [
        definition
        for definition in TOOL_DEFINITIONS
        if definition["name"] in names
    ]
    episode_run = copy.deepcopy(EPISODE_RUN_TOOL)
    if allowed_episode_kinds is not None:
        valid = [kind for kind in allowed_episode_kinds if kind in EPISODE_KINDS]
        episode_run["input_schema"]["properties"]["kind"]["enum"] = valid
    selected.extend([
        episode_run,
        EPISODE_INSTALL_TOOL,
        REPAIR_TRANSACTION_TOOL,
        BRANCH_ADJUDICATE_TOOL,
    ])
    return selected


V3_LEAD_TOOL_DEFINITIONS = v3_lead_tool_definitions()


def _submit_result_tool_def(result_schema: dict[str, Any]) -> dict[str, Any]:
    """Virtual tool appended to every episode's tool defs (handled by the runner)."""
    return {
        "name": "episode_submit_result",
        "description": (
            "Submit your final result for this episode. `result` must match the "
            "result schema; `summary` is a <=800-char plain-language summary that "
            "becomes the ledger entry."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result": result_schema,
                "summary": {"type": "string"},
            },
            "required": ["result", "summary"],
        },
    }


# ---------------------------------------------------------------------------
# Episode context (deterministic, capped; firewall-covered)
# ---------------------------------------------------------------------------
_EPISODE_CONTEXT_CAP = 6000
_EPISODE_WINDOW_CHARS = 1500


def _episode_branch_card_text(executor: WorkspaceToolExecutor, name: str) -> str:
    card = executor._branch_card(name)  # side-effect-free
    scores = card.get("scores") or {}
    dict_rate = scores.get("dict_rate")
    quad = scores.get("quad")
    dr = f"{dict_rate:.3f}" if isinstance(dict_rate, (int, float)) else "n/a"
    qd = f"{quad:.3f}" if isinstance(quad, (int, float)) else "n/a"
    tags = ", ".join(card.get("tags") or [])
    excerpt = str(card.get("decoded_excerpt") or "").strip()
    line = (
        f"- `{name}` mapped={card.get('mapped_count')} dict_rate={dr} quad={qd}"
    )
    if tags:
        line += f" [{tags}]"
    if excerpt:
        line += f"\n    {excerpt}"
    return line


def _build_verify_context(spec: EpisodeSpec) -> str:
    """Verify-kind context: ONLY the candidate plaintext + language framing (F2).

    No branch cards, no dict_rate/quad, no decode windows — scores in a verify
    prompt defeat its independence and the ground-truth leak-assert cannot catch
    them. The candidate is the SOLVER's output (rendered by the lead dispatcher
    at dispatch time), never ground truth.

    ``spec.goal`` is deliberately IGNORED here (review F-3): the goal is
    lead-authored free text, and rendering it would let the lead shape the
    "independent" reader's judgement (e.g. "be lenient about transcription
    errors"). The task line is fixed; the goal still lands in the episode
    ledger for observability.
    """
    from analysis.dictionary import LANGUAGE_NAMES

    candidate = str(spec.inputs.get("candidate_text") or "")
    language = str(spec.inputs.get("language") or "en")
    language_name = LANGUAGE_NAMES.get(language, "the target language")
    parts = [
        (
            "## Task\nJudge whether the candidate text below reads as real "
            f"(possibly damaged or partial) {language_name}."
        ),
        (
            f"## Candidate decipherment (language: {language_name})\n"
            f"```\n{candidate}\n```"
        ),
    ]
    text = "\n\n".join(parts)
    if len(text) > _EPISODE_CONTEXT_CAP:
        text = text[:_EPISODE_CONTEXT_CAP] + "\n…[truncated]"
    return text


def build_episode_context(
    spec: EpisodeSpec,
    workspace: Workspace,
    executor: WorkspaceToolExecutor,
) -> str:
    """Deterministic worker context: branch card(s), a decode window, note."""
    if spec.kind == "verify":
        return _build_verify_context(spec)
    branches = list(spec.inputs.get("branches") or [])
    parts: list[str] = [f"## Task\n{spec.goal}"]
    packet = spec.inputs.get("candidate_packet")
    if spec.kind in {"reading", "repair"} and isinstance(packet, dict):
        spans = []
        for span in packet.get("spans") or []:
            if not isinstance(span, dict):
                continue
            spans.append(
                f"### {span.get('span_id')} "
                f"(words {span.get('word_start')}:{span.get('word_end')})\n"
                f"```\n{span.get('text') or ''}\n```"
            )
        parts.append(
            "## Candidate reading packet\n"
            f"- branch: `{packet.get('branch')}`\n"
            f"- content hash: `{packet.get('content_hash')}`\n"
            f"- capability: `{packet.get('capability')}`\n"
            "Use only the host provenance below for positioned fragments.\n\n"
            + "\n\n".join(spans)
        )
    if branches:
        cards = [_episode_branch_card_text(executor, b) for b in branches
                 if workspace.has_branch(b)]
        parts.append("## Branch cards\n" + "\n".join(cards))
        first = next((b for b in branches if workspace.has_branch(b)), None)
        if first is not None and not (
            spec.kind in {"reading", "repair"} and isinstance(packet, dict)
        ):
            decode = _decoded_text_for_panel(workspace, first)
            window = decode[:_EPISODE_WINDOW_CHARS]
            parts.append(
                f"## Decode window (branch `{first}`)\n```\n{window}\n```"
            )
    reading = spec.inputs.get("reading")
    if isinstance(reading, dict) and reading.get("reading_id"):
        # Part 6: a compiled reading summary for the repair worker (id,
        # confidence, holes, <=400-char text). Firewall-covered (proposed
        # plaintext only, never ground truth).
        from investigation.reading import Reading
        compiled = Reading.from_dict(reading)
        holes = compiled.holes
        holes_text = ("; ".join(holes)[:200]) if holes else "(none)"
        parts.append(
            "## Reading to apply\n"
            f"- id `{compiled.reading_id}` on branch `{compiled.branch}` "
            f"(overall_confidence={compiled.overall_confidence})\n"
            f"- holes: {holes_text}\n"
            f"- text: {compiled.full_text[:400]}"
        )
    note = str(spec.inputs.get("context_note") or "").strip()
    if note:
        parts.append("## Note\n" + note)
    text = "\n\n".join(parts)
    if len(text) > _EPISODE_CONTEXT_CAP:
        text = text[:_EPISODE_CONTEXT_CAP] + "\n…[truncated]"
    return text


def _episode_system_prompt(spec: EpisodeSpec, language: str | None = None) -> str:
    budget = spec.budget
    schema_json = json.dumps(spec.result_schema, ensure_ascii=False)
    # F5: EPISODE_KINDS contracts are static; the verify contract carries a
    # ``{language}`` placeholder formatted here after the lead resolves the run
    # language. Other contracts have no braces, so this is a no-op for them.
    contract = spec.contract
    if "{language}" in contract:
        from analysis.dictionary import LANGUAGE_NAMES

        language_name = LANGUAGE_NAMES.get(language or "en", "the target language")
        contract = contract.format(language=language_name)
    return (
        f"{contract}\n\n"
        f"Available tools (and no others): {', '.join(sorted(spec.toolset))}.\n"
        f"Budget: up to {budget.max_tool_calls} tool calls, "
        f"{budget.wall_clock_seconds:.0f}s wall clock.\n"
        f"When done, submit via `episode_submit_result` with a `result` object "
        f"matching this schema and a short `summary`:\n{schema_json}\n"
        "You cannot declare solutions, see benchmark context, or create "
        "hypothesis branches — stay within your task."
    )


# ---------------------------------------------------------------------------
# Episode workspace isolation (A1 + F5)
# ---------------------------------------------------------------------------
def _build_episode_workspace(state: InvestigationState, branches: list[str]) -> Workspace:
    ws = Workspace(
        cipher_text=state.cipher,
        plaintext_alphabet=state.workspace.plaintext_alphabet,
    )
    for name in branches:
        if not state.workspace.has_branch(name):
            continue
        # F5: _serialize_branch shallow-copies metadata and passes
        # transform_pipeline by reference — deep-copy the snapshot before restore
        # so a worker mutating nested metadata/pipeline cannot touch the lead.
        snap = copy.deepcopy(_serialize_branch(state.workspace, name))
        _restore_branch_into(ws, snap)
    return ws


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------
def _episode_transcripts_enabled() -> bool:
    return os.environ.get("DECIPHER_DEBUG_EPISODE_TRANSCRIPTS", "").strip() in {"1", "true", "yes", "on"}


def run_episode(
    spec: EpisodeSpec,
    state: InvestigationState,
    session: Any = None,
    *,
    provider: Any = None,
    language: str | None = None,
    word_set: set[str] | None = None,
    word_list: list[str] | None = None,
    pattern_dict: dict[str, list[str]] | None = None,
    launching_turn: int = 0,
    on_event: Any = None,
    max_cost_usd: float | None = None,
    outer_cost_usd: float = 0.0,
) -> EpisodeResult:
    """Run one fresh-context episode worker. Returns a structured EpisodeResult.

    Never raises for ordinary failures (A9): schema/budget/handler issues become
    ``episode_failed`` results. ``KeyboardInterrupt`` is re-raised after the
    ledger entry + budget are merged.

    ``max_cost_usd`` (M5.3 Slice 1, A4) is the per-run paid ceiling, checked
    before EVERY paid worker send (including mid-episode continuation and the
    submit-only reserve send). ``outer_cost_usd`` is the run's committed spend
    OUTSIDE this episode (lead + prior/completed episodes); the guard adds this
    episode's own session spend to it. A ceiling reached mid-episode ends the
    episode immediately with an ``episode_failed(cost_ceiling_reached)`` result
    (visible in the ledger) and makes no further send.

    ``on_event`` (default None) is an optional callback the v3 dispatcher passes
    so episode-internal progress is forwarded into the lead event stream: it
    receives ``episode_turn_start``, ``episode_tool_call`` and ``episode_submit``
    events, each carrying ``{episode_id, kind, turn, ...}`` (``turn`` is the
    episode-internal turn). Purely observational — never raises into the run.
    """
    episode_id = uuid.uuid4().hex[:12]
    kind = spec.kind
    start = time.time()

    def _emit_ep(event: str, payload: dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(event, {"episode_id": episode_id, "kind": kind, **payload})
        except Exception:  # noqa: BLE001 - observability must never crash a run
            pass

    # F12: language + word resources default from state.
    if language is None:
        language = state.language
    if word_set is None or word_list is None or pattern_dict is None:
        from analysis import dictionary, pattern
        dict_path = dictionary.get_dictionary_path(language)
        if word_set is None:
            word_set = dictionary.load_word_set(dict_path) if dict_path else set()
        if word_list is None:
            word_list = pattern.load_word_list(dict_path) if dict_path else []
        if pattern_dict is None:
            pattern_dict = pattern.build_pattern_dictionary(word_list)

    branches = list(spec.inputs.get("branches") or [])
    toolset = set(spec.toolset)

    # A1/M3: an episode-LOCAL readings map, seeded from the injected reading DICT
    # (Part 6). Workers never see or write state.readings; composites inside the
    # episode read/apply readings only from this local map.
    episode_readings: dict[str, dict[str, Any]] = {}
    injected_reading = spec.inputs.get("reading")
    if isinstance(injected_reading, dict) and injected_reading.get("reading_id"):
        episode_readings[str(injected_reading["reading_id"])] = injected_reading

    # A9: episode SETUP runs inside the crash guard too — a malformed branch
    # snapshot, a raising session builder (F6), or a _branch_card failure while
    # rendering the context must become episode_failed(runner_error), never a
    # lead crash.
    try:
        # (1) episode workspace + (2)/(3) fresh executor with the episode
        # toolset, NoGatesPolicy, an episode-local repair_agenda, the
        # state-owned finalist store, and NO shared board (A1).
        episode_ws = _build_episode_workspace(state, branches)
        executor = WorkspaceToolExecutor(
            workspace=episode_ws,
            language=language,
            word_set=word_set,
            word_list=word_list,
            pattern_dict=pattern_dict,
            benchmark_context=None,
            declaration_policy=NoGatesPolicy(),
            episode_toolset=toolset,
            finalist_sessions=state.finalist_sessions,
            repair_agenda=[],
            # The lead's act_set_model_variant selection is mirrored into state
            # by the lead loop; seed this fresh episode executor with it so
            # episode search tools resolve the same binary model.
            model_variant=state.model_variant,
        )
        executor.episode_id = episode_id
        executor.set_iteration(launching_turn)

        # (2) session from the kind registry (role episode:<kind>).
        system_prompt = _episode_system_prompt(spec, language)
        if session is None:
            session = session_factory(f"episode:{kind}", provider, system_prompt)

        # tool defs = toolset tools (v2 + v3 composites) + the virtual submit tool.
        tool_defs = [
            d for d in (TOOL_DEFINITIONS + COMPOSITE_TOOL_DEFINITIONS)
            if d["name"] in toolset
        ]
        tool_defs = tool_defs + [_submit_result_tool_def(spec.result_schema)]

        context_text = build_episode_context(spec, episode_ws, executor)
    except KeyboardInterrupt:
        raise
    except Exception:  # noqa: BLE001 - structured setup failure (A9)
        failure = EpisodeResult(
            episode_id=episode_id,
            kind=kind,
            goal=spec.goal,
            status="episode_failed",
            failure_reason="runner_error",
            raw_text=traceback.format_exc(),
            elapsed_seconds=time.time() - start,
        )
        state.episode_ledger.append(
            failure.ledger_dict(include_transcript=False)
        )
        return failure

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": context_text}]},
    ]

    budget = spec.budget
    tool_call_count = 0
    # M5.3 Slice 1: over-budget tool_uses the host skipped this episode.
    overbudget_skips = 0
    schema_retry_used = False
    turn_cap = max(budget.max_tool_calls * 2 + 3, 12)

    def _cost_ceiling_reached() -> bool:
        # M5.3 Slice 1 (A4): the per-run paid ceiling. True once the run's
        # committed spend outside this episode PLUS this episode's own session
        # spend so far reaches ``max_cost_usd``. Checked before every worker send.
        if max_cost_usd is None:
            return False
        episode_cost = sum(e.cost() for e in session.usage_entries())
        return (outer_cost_usd + episode_cost) >= max_cost_usd

    def _episode_budget_entries() -> list[BudgetEntry]:
        # A7: re-tag every send under category f"episode:{kind}" (per-entry cost;
        # mixed models correct). Real episode sessions already carry this
        # category; re-tagging also makes scripted fakes correct.
        return [
            BudgetEntry(
                category=f"episode:{kind}",
                provider=e.provider,
                model=e.model,
                input_tokens=e.input_tokens,
                output_tokens=e.output_tokens,
                cache_read_tokens=e.cache_read_tokens,
            )
            for e in session.usage_entries()
        ]

    def _finish(
        status: str,
        *,
        failure_reason: str | None = None,
        result: Any = None,
        summary: str = "",
        raw_text: str | None = None,
    ) -> EpisodeResult:
        requested = set(branches)
        snapshots = []
        for name in episode_ws.branch_names():
            if name == "main" and "main" not in requested:
                continue
            snapshots.append(_serialize_branch(episode_ws, name))
        budget_entries = _episode_budget_entries()
        return EpisodeResult(
            episode_id=episode_id,
            kind=kind,
            goal=spec.goal,
            status=status,
            failure_reason=failure_reason,
            result=result,
            summary=(summary or "")[:800],
            branch_snapshots=snapshots,
            tool_call_count=tool_call_count,
            suppressed_over_budget_calls=overbudget_skips,
            elapsed_seconds=time.time() - start,
            budget_entries=[e.to_dict() for e in budget_entries],
            raw_text=raw_text,
            transcript=(session.export_transcript() if _episode_transcripts_enabled() else None),
            agenda_additions=[dict(item) for item in executor.repair_agenda],
            tool_calls=list(executor.call_log),
        )

    def _commit(result: EpisodeResult) -> None:
        # Part 1: on completion append the ledger dict and extend the budget
        # ledger with the per-kind episode entries.
        state.episode_ledger.append(
            result.ledger_dict(include_transcript=_episode_transcripts_enabled())
        )
        state.budget_ledger.extend(_episode_budget_entries())

    def _final_result_send() -> EpisodeResult:
        """Budget exhausted: one submit-only send, preserving structured output."""
        # M5.3 Slice 1 (A4): the submit-only reserve is still a PAID send — the
        # per-run ceiling gates it too. If reached, end budget-class with no send.
        if _cost_ceiling_reached():
            return _finish(
                "episode_failed", failure_reason="cost_ceiling_reached"
            )
        nudge = {"role": "user", "content": [{"type": "text", "text": (
            "Budget reached. Call episode_submit_result now. Do not call any "
            "other tool. Submit the best partial result you can support."
        )}]}
        submit_def = _submit_result_tool_def(spec.result_schema)
        response = session.send(messages + [nudge], tools=[submit_def],
                                max_tokens=budget.max_output_tokens)
        _blocks, tool_uses, text_parts = _collect_assistant_blocks(response)
        for tu in tool_uses:
            if tu.get("name") != "episode_submit_result":
                continue
            args = tu.get("input") or {}
            proposed = args.get("result")
            errors = validate_against_schema(proposed, spec.result_schema)
            if not errors:
                return _finish(
                    "ok",
                    result=proposed,
                    summary=str(args.get("summary") or ""),
                )
        raw = "\n".join(text_parts).strip()
        parsed = _extract_json(raw)
        if parsed is not None and not validate_against_schema(parsed, spec.result_schema):
            return _finish("ok", result=parsed, summary=raw[:800])
        return _finish("episode_failed", failure_reason="budget_exhausted", raw_text=raw)

    def _run() -> EpisodeResult:
        nonlocal tool_call_count, schema_retry_used, overbudget_skips
        try:
            for _turn in range(turn_cap):
                # M5.3 Slice 1 (A4): the per-run paid ceiling is checked BEFORE
                # every worker send, including the submit-only reserve reached via
                # _final_result_send. If reached, end budget-class with NO send.
                if _cost_ceiling_reached():
                    return _finish(
                        "episode_failed", failure_reason="cost_ceiling_reached"
                    )
                # A9: wall clock is checked BETWEEN tool calls (never mid-tool).
                if time.time() - start > budget.wall_clock_seconds:
                    return _final_result_send()

                ep_turn = _turn + 1
                _emit_ep("episode_turn_start", {"turn": ep_turn})
                response = session.send(messages, tools=tool_defs,
                                        max_tokens=budget.max_output_tokens)
                assistant_blocks, tool_uses, _text = _collect_assistant_blocks(response)
                messages.append({"role": "assistant", "content": assistant_blocks})

                if not tool_uses:
                    # No tool call: nudge once to submit, else give up.
                    messages.append({"role": "user", "content": [{"type": "text", "text": (
                        "You made no tool call. Use an allowed tool, or submit "
                        "your result with episode_submit_result."
                    )}]})
                    continue

                tool_results: list[dict[str, Any]] = []
                for tu in tool_uses:
                    name = tu["name"]
                    args = tu.get("input") or {}
                    if name == "episode_submit_result":
                        proposed = args.get("result")
                        summary = str(args.get("summary") or "")
                        errors = validate_against_schema(proposed, spec.result_schema)
                        _emit_ep("episode_submit", {
                            "turn": ep_turn, "accepted": not errors,
                            "status": "ok" if not errors else "schema_error",
                        })
                        if not errors:
                            return _finish("ok", result=proposed, summary=summary)
                        if not schema_retry_used:
                            schema_retry_used = True
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tu["id"],
                                "content": json.dumps({
                                    "error": "result did not match the schema; fix and resubmit.",
                                    "schema_errors": errors[:12],
                                }, ensure_ascii=False),
                            })
                            continue
                        # Second mismatch → structured failure.
                        return _finish(
                            "episode_failed", failure_reason="schema_mismatch",
                            raw_text=json.dumps(proposed, ensure_ascii=False, default=str),
                        )
                    # M5.3 Slice 1: enforce the tool-call cap BEFORE each ordinary
                    # / composite call (not after the whole batch). A model may
                    # emit more calls than remain; execute only those that fit and
                    # synthesize a paired ``budget_exhausted`` result for every
                    # SKIPPED tool_use so the recorded exchange has exactly one
                    # tool_result per tool_use (an unpaired exchange 400s at
                    # resume). A valid ``episode_submit_result`` is handled above,
                    # so it is still accepted even in an over-budget batch.
                    if tool_call_count >= budget.max_tool_calls:
                        overbudget_skips += 1
                        _emit_ep("episode_tool_call_skipped", {
                            "turn": ep_turn, "tool": name,
                            "reason": "budget_exhausted",
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tu["id"],
                            "content": json.dumps({
                                "error": "budget_exhausted",
                                "reason": (
                                    "episode tool-call budget reached; this call "
                                    "was not executed."
                                ),
                                "max_tool_calls": budget.max_tool_calls,
                            }, ensure_ascii=False),
                        })
                        continue
                    _emit_ep("episode_tool_call", {
                        "turn": ep_turn, "tool": name,
                        "arguments": _compact_episode_args(args),
                    })
                    if name in COMPOSITE_TOOL_NAMES and name in toolset:
                        # Part 2/A4: dispatch a composite ONLY when it is in this
                        # episode's toolset; the ToolCall log + hint filtering
                        # happen inside execute_composite. Anything else falls
                        # through to executor.execute (the M2 neutral off-toolset
                        # rejection + _gate_hits telemetry apply for free).
                        result_obj = execute_composite(
                            name, args, executor=executor,
                            state_readings=episode_readings,
                            turn=launching_turn, tool_use_id=tu["id"],
                        )
                        tool_call_count += 1
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tu["id"],
                            "content": _json(result_obj),
                        })
                        continue
                    # Ordinary tool: executor turns handler exceptions into error
                    # JSON already (A9); call count is checked below.
                    result = executor.execute(name, args, tool_use_id=tu["id"])
                    tool_call_count += 1
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": result,
                    })

                messages.append({"role": "user", "content": tool_results})

                if tool_call_count >= budget.max_tool_calls:
                    return _final_result_send()

            # Ran out of turns without submitting → treat as budget exhausted.
            return _final_result_send()
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001 - the runner must never crash the lead.
            return _finish(
                "episode_failed", failure_reason="runner_error",
                raw_text=traceback.format_exc(),
            )

    try:
        result = _run()
    except KeyboardInterrupt:
        # Append the failure result + budget, then re-raise so the lead's R5
        # pairing synthesizes a `stopped` tool_result for the in-flight
        # episode_run (Part 6 dispatcher).
        interrupted = _finish("episode_failed", failure_reason="interrupted")
        _commit(interrupted)
        raise
    _commit(result)
    return result
