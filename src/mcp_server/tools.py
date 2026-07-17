"""MCP tool definitions (Part 5): 23 schemas + arg-validation helpers.

Schemas that mirror the v2/v3 surface are copied verbatim from the landed tool
definitions (never re-typed) and augmented with the shared envelope fields, so
they cannot drift from ``tools_v2``/``actions``/``experiments``. New surfaces
(candidate portfolio, records, repair v1, verification) are declared here per
the spec's exact JSON.
"""
from __future__ import annotations

import copy
from typing import Any

from agent.tools_v2 import TOOL_DEFINITIONS as _V2_DEFS
from investigation.actions import BRANCH_ADJUDICATE_TOOL, HYPOTHESIS_TEST_WORDS_TOOL
from investigation.experiments import EXPERIMENT_COLLECT_TOOL, EXPERIMENT_SUBMIT_TOOL

_V2 = {d["name"]: d for d in _V2_DEFS}

_INVESTIGATION_ID = {
    "type": "string",
    "description": "Id returned by investigation_start/list.",
}
_EXPECTED_REVISION = {
    "type": "integer",
    "description": (
        "Revision from your latest investigation_status; mismatch returns a "
        "conflict."
    ),
}


def _mutate_envelope(schema: dict) -> dict:
    """Deep-copy a base schema and add investigation_id + expected_revision."""
    schema = copy.deepcopy(schema)
    props = schema.setdefault("properties", {})
    props["investigation_id"] = dict(_INVESTIGATION_ID)
    props["expected_revision"] = dict(_EXPECTED_REVISION)
    req = list(schema.get("required") or [])
    schema["required"] = req + ["investigation_id", "expected_revision"]
    return schema


def _read_envelope(schema: dict) -> dict:
    """Deep-copy a base schema and add investigation_id (read class)."""
    schema = copy.deepcopy(schema)
    props = schema.setdefault("properties", {})
    props["investigation_id"] = dict(_INVESTIGATION_ID)
    req = list(schema.get("required") or [])
    if "investigation_id" not in req:
        req = req + ["investigation_id"]
    schema["required"] = req
    return schema


def _v2_schema(name: str) -> dict:
    return copy.deepcopy(_V2[name]["input_schema"])


# --- repair_hypotheses_test: HYPOTHESIS_TEST_WORDS_TOOL minus per-item install ---
def _repair_test_schema() -> dict:
    schema = copy.deepcopy(HYPOTHESIS_TEST_WORDS_TOOL["input_schema"])
    item_props = schema["properties"]["hypotheses"]["items"]["properties"]
    item_props.pop("install", None)  # the server forces install
    return _mutate_envelope(schema)


# --- hypothesis_branch_reject: workspace_reject_hypothesis minus acknowledge flag ---
def _reject_schema() -> dict:
    schema = _v2_schema("workspace_reject_hypothesis")
    schema["properties"].pop("acknowledge_pending_required_tools", None)
    return _mutate_envelope(schema)


MCP_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # 5.1 create
    {
        "name": "investigation_start",
        "description": (
            "Start a new cipher investigation from INLINE ciphertext (never a "
            "file path). Returns the investigation id, revision 1, and the full "
            "self-briefing. Formats: 'canonical' = space-separated S-tokens with "
            "' | ' word separators; 'letters' = plain text, whitespace separates "
            "words; 'auto' detects canonical S-token transcriptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ciphertext": {"type": "string"},
                "language": {
                    "type": "string",
                    "enum": ["en", "la", "de", "fr", "it", "es"],
                },
                "format": {
                    "type": "string",
                    "enum": ["auto", "letters", "canonical"],
                },
                "label": {"type": "string"},
                "context": {
                    "type": "string",
                    "description": (
                        "Optional prior/historical context; rendered every turn "
                        "as the external-context section. Never treated as ground "
                        "truth."
                    ),
                },
            },
            "required": ["ciphertext"],
        },
    },
    # 5.2 read (no investigation_id)
    {
        "name": "investigation_list",
        "description": (
            "List stored investigations (newest updated first): id, label, "
            "status, language, size, and whether a terminal record exists."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
    },
    # 5.3 read
    {
        "name": "investigation_status",
        "description": (
            "The full self-briefing rebuilt from server state: cipher facts, "
            "diagnostic fingerprint, branch cards, hypothesis board, readings, "
            "experiments, evidence, a rotating decode window, and advisory host "
            "guidance. One call fully briefs a bare session; the revision it "
            "returns is the one you pass with mutating calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"investigation_id": {"type": "string"}},
            "required": ["investigation_id"],
        },
    },
    # 5.4 read
    {
        "name": "observe_overview",
        "description": (
            "Compact measured facts: alphabet, token/word counts, IC, diagnostic "
            "fingerprint, model variants, branch and evidence counts. A subset of "
            "investigation_status for cheap re-orientation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"investigation_id": {"type": "string"}},
            "required": ["investigation_id"],
        },
    },
    # 5.5 read -> host.handle_tool("observe_diagnosis")
    {
        "name": "observe_diagnosis",
        "description": _V2["observe_diagnosis"]["description"],
        "input_schema": {
            "type": "object",
            "properties": {
                "investigation_id": {"type": "string"},
                "branch": {"type": "string"},
                "max_period": {"type": "integer"},
            },
            "required": ["investigation_id"],
        },
    },
    # 5.6 read -> host.handle_tool("decode_show")
    {
        "name": "decode_show",
        "description": _V2["decode_show"]["description"],
        "input_schema": {
            "type": "object",
            "properties": {
                "investigation_id": {"type": "string"},
                "branch": {"type": "string"},
                "start_word": {"type": "integer"},
                "count": {"type": "integer"},
            },
            "required": ["investigation_id", "branch"],
        },
    },
    # 5.7 mutate -> workspace_create_hypothesis_branch
    {
        "name": "hypothesis_branch_create",
        "description": _V2["workspace_create_hypothesis_branch"]["description"],
        "input_schema": _mutate_envelope(_v2_schema("workspace_create_hypothesis_branch")),
    },
    # 5.8 mutate -> workspace_update_hypothesis
    {
        "name": "hypothesis_branch_update",
        "description": _V2["workspace_update_hypothesis"]["description"],
        "input_schema": _mutate_envelope(_v2_schema("workspace_update_hypothesis")),
    },
    # 5.9 mutate -> workspace_reject_hypothesis (minus acknowledge flag)
    {
        "name": "hypothesis_branch_reject",
        "description": _V2["workspace_reject_hypothesis"]["description"],
        "input_schema": _reject_schema(),
    },
    # 5.10 read -> workspace_hypothesis_next_steps
    {
        "name": "hypothesis_next_steps",
        "description": (
            "Advisory next-step suggestions for a hypothesis branch (policy WF-1). "
            "Recommendations, not requirements."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "investigation_id": {"type": "string"},
                "branch": {"type": "string"},
            },
            "required": ["investigation_id"],
        },
    },
    # 5.11 read (new — WF-6 portfolio surface)
    {
        "name": "candidate_list",
        "description": (
            "The candidate portfolio: every active branch with provenance, "
            "labeled score signals, verification status, and derived roles. No "
            "single scalar defines 'the' candidate; the scalar-best branch is one "
            "labeled role (policy WF-6, pre-registered divergence)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "investigation_id": {"type": "string"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
                "include_rejected": {"type": "boolean"},
            },
            "required": ["investigation_id"],
        },
    },
    # 5.12 read (new)
    {
        "name": "candidate_show",
        "description": (
            "Full detail for ONE candidate branch: its portfolio entry plus the "
            "decoded text, attestation history, bound readings, and repair "
            "transactions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "investigation_id": {"type": "string"},
                "branch": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["investigation_id", "branch"],
        },
    },
    # 5.13 mutate -> experiment_submit
    {
        "name": "experiment_submit",
        "description": (
            EXPERIMENT_SUBMIT_TOOL["description"]
            + " Experiments run in the background inside the session that holds "
            "this investigation's writer lease; results surface in "
            "investigation_status and experiment_collect."
        ),
        "input_schema": _mutate_envelope(EXPERIMENT_SUBMIT_TOOL["input_schema"]),
    },
    # 5.14 mutate -> experiment_collect
    {
        "name": "experiment_collect",
        "description": EXPERIMENT_COLLECT_TOOL["description"],
        "input_schema": _mutate_envelope(EXPERIMENT_COLLECT_TOOL["input_schema"]),
    },
    # 5.15 mutate (new — client-authored reading)
    {
        "name": "reading_record",
        "description": (
            "Record YOUR reading of a branch's decode, hash-bound to its current "
            "content. Required before repair_transaction (evidence binding "
            "REP-1). One reading per (content, verifier-evidence) pair (SAT-4); "
            "duplicates return the existing reading_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "investigation_id": {"type": "string"},
                "expected_revision": {"type": "integer"},
                "branch": {"type": "string"},
                "reading_text": {"type": "string"},
                "fragments": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "repair_text": {"type": ["string", "null"]},
                        "span_id": {"type": ["string", "null"]},
                        "start": {"type": ["integer", "null"]},
                        "end": {"type": ["integer", "null"]},
                        "confidence": {"type": ["number", "string"]},
                    },
                    "required": ["text"],
                }},
                "holes": {"type": "array", "items": {"type": "string"}},
                "overall_confidence": {"type": "number"},
            },
            "required": [
                "investigation_id", "expected_revision", "branch",
                "reading_text", "overall_confidence",
            ],
        },
    },
    # 5.16 mutate (new — CMP-2 divergence)
    {
        "name": "comparison_record",
        "description": (
            "Record YOUR ranking of competing candidates, hash-bound to each "
            "branch's current content. best_partial and accepts_as_solution are "
            "SEPARATE fields (policy CMP-2): an honest 'best partial so far' never "
            "requires claiming a solution. accepts_as_solution here never unlocks "
            "declaration — only request_independent_verification can (DECL-1)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "investigation_id": {"type": "string"},
                "expected_revision": {"type": "integer"},
                "branches": {"type": "array", "items": {"type": "string"}},
                "ranking": {"type": "array", "items": {"type": "string"}},
                "best_partial": {"type": ["string", "null"]},
                "accepts_as_solution": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": [
                "investigation_id", "expected_revision", "branches", "ranking",
                "best_partial", "accepts_as_solution", "rationale",
            ],
        },
    },
    # 5.17 mutate (new — Part 8.2)
    {
        "name": "repair_hypotheses_test",
        "description": (
            "Compile a batch of word hypotheses against ONE branch in an ISOLATED "
            "scratch workspace (no LLM; deterministic host code). Builds the "
            "word-repair menu once, probes every hypothesis, installs each viable "
            "one on a scratch fork, and returns per-item verdicts plus a deduped "
            "finalist set with host-generated edit labels and collateral "
            "evidence. The result is a compile session; pass its compile_id and "
            "your chosen winner to repair_transaction for host-validated "
            "installation."
        ),
        "input_schema": _repair_test_schema(),
    },
    # 5.18 mutate (new — Part 8.3)
    {
        "name": "repair_transaction",
        "description": (
            "Host-validated acceptance and installation of ONE client-compiled "
            "repair winner (Slice-4 contract, REP-1..4): binds your stored "
            "reading to the branch's current content, suppresses duplicate "
            "source/interpretation pairs, enforces saturation, runs the eight "
            "acceptance checks against the compile evidence (winner "
            "named+changed, fork evidence, edit-claim binding, adjudication with "
            "multiple finalists, collateral limits, no-op probe, default-deny on "
            "any scalar decrease), and installs the winner under a fresh name "
            "requiring reverification. This is NOT the v3 API-billed internal "
            "repair episode."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "investigation_id": {"type": "string"},
                "expected_revision": {"type": "integer"},
                "branch": {"type": "string"},
                "compile_id": {"type": "string"},
                "winner": {"type": "string"},
                "reading_id": {
                    "type": "string",
                    "description": (
                        "Stored reading to bind (REP-1). Omit to use the newest "
                        "reading bound to this exact branch content."
                    ),
                },
                "as_name": {"type": "string"},
            },
            "required": [
                "investigation_id", "expected_revision", "branch",
                "compile_id", "winner",
            ],
        },
    },
    # 5.19 mutate (Part 7)
    {
        "name": "request_independent_verification",
        "description": (
            "Run one server-side INDEPENDENT verification of a branch: a fresh "
            "reader (no scores, no context you can shape) judges whether its "
            "decode reads as real target-language text and writes a hash-bound "
            "attestation. A positive attestation (reader_accepts_as_solution) is "
            "the only key that unlocks meta_declare_solution (DECL-1). "
            "Server-side and API-billed (Option A); returns a typed 'unavailable' "
            "result when the server has no API key."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "investigation_id": {"type": "string"},
                "expected_revision": {"type": "integer"},
                "branch": {"type": "string"},
            },
            "required": ["investigation_id", "expected_revision", "branch"],
        },
    },
    # 5.20 read -> branch_adjudicate
    {
        "name": "branch_adjudicate",
        "description": BRANCH_ADJUDICATE_TOOL["description"],
        "input_schema": _read_envelope(BRANCH_ADJUDICATE_TOOL["input_schema"]),
    },
    # 5.21 mutate -> act_set_model_variant
    {
        "name": "act_set_model_variant",
        "description": _V2["act_set_model_variant"]["description"],
        "input_schema": _mutate_envelope(_v2_schema("act_set_model_variant")),
    },
    # 5.22 mutate -> meta_declare_solution
    {
        "name": "meta_declare_solution",
        "description": (
            _V2["meta_declare_solution"]["description"]
            + " Declaration is hard-gated (DECL-1): it succeeds only when the "
            "newest independent-reader attestation matching this branch's current "
            "content hash is positive (reader_accepts_as_solution). Run "
            "request_independent_verification first."
        ),
        "input_schema": _mutate_envelope(_v2_schema("meta_declare_solution")),
    },
    # 5.23 mutate -> meta_declare_unsolved
    {
        "name": "meta_declare_unsolved",
        "description": _V2["meta_declare_unsolved"]["description"],
        "input_schema": _mutate_envelope(_v2_schema("meta_declare_unsolved")),
    },
]

# Tool class per Part 5.0 (create/read/mutate).
TOOL_CLASSES: dict[str, str] = {
    "investigation_start": "create",
    "investigation_list": "read",
    "investigation_status": "read",
    "observe_overview": "read",
    "observe_diagnosis": "read",
    "decode_show": "read",
    "hypothesis_next_steps": "read",
    "candidate_list": "read",
    "candidate_show": "read",
    "branch_adjudicate": "read",
    "hypothesis_branch_create": "mutate",
    "hypothesis_branch_update": "mutate",
    "hypothesis_branch_reject": "mutate",
    "experiment_submit": "mutate",
    "experiment_collect": "mutate",
    "reading_record": "mutate",
    "comparison_record": "mutate",
    "repair_hypotheses_test": "mutate",
    "repair_transaction": "mutate",
    "request_independent_verification": "mutate",
    "act_set_model_variant": "mutate",
    "meta_declare_solution": "mutate",
    "meta_declare_unsolved": "mutate",
}

TOOL_NAMES = frozenset(d["name"] for d in MCP_TOOL_DEFINITIONS)


def render_tool_list() -> list[dict]:
    """Render the tools/list payload (camelCase inputSchema)."""
    return [
        {
            "name": d["name"],
            "description": d["description"],
            "inputSchema": d["input_schema"],
        }
        for d in MCP_TOOL_DEFINITIONS
    ]


_SCHEMA_BY_NAME = {d["name"]: d["input_schema"] for d in MCP_TOOL_DEFINITIONS}


def schema_for(name: str) -> dict | None:
    return _SCHEMA_BY_NAME.get(name)
