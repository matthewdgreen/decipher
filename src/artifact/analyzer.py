"""Lightweight gap analysis for v2 agent run artifacts."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ArtifactFinding:
    label: str
    severity: str
    message: str
    iteration: int | None = None
    tool_name: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "severity": self.severity,
            "message": self.message,
            "iteration": self.iteration,
            "tool_name": self.tool_name,
            "evidence": self.evidence,
        }


def load_artifact(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyze_artifact(artifact: dict[str, Any]) -> list[ArtifactFinding]:
    """Return machine-readable parity/gap findings for one run artifact."""
    findings: list[ArtifactFinding] = []
    tool_calls = artifact.get("tool_calls", [])
    likely_homophonic_no_boundary = _likely_homophonic_no_boundary(artifact)

    if likely_homophonic_no_boundary:
        findings.extend(_check_homophonic_search_order(tool_calls))
        findings.extend(_check_homophonic_global_swaps(tool_calls))

    findings.extend(_check_worsening_mutations(tool_calls))
    findings.extend(_check_declaration_after_worse_mutation(tool_calls))
    findings.extend(_check_declared_branch_accuracy(artifact))
    findings.extend(_check_score_overrode_reading(tool_calls, artifact))
    findings.extend(_check_unattempted_reading_fix(artifact))
    findings.extend(_check_loop_events(artifact))
    findings.extend(_check_repair_agenda_unresolved(artifact))
    findings.extend(_check_projection_failures(tool_calls))
    return findings


# Reading-driven repair primitives — see prompts_v2.py "Reading-driven
# repair". These are the cipher-symbol-level mutation tools the agent should
# reach for when it recognises target-language words in the decoded text.
_READING_PRIMITIVES = frozenset(
    ("act_set_mapping", "act_bulk_set", "act_anchor_word", "act_apply_word_repair")
)
_READING_FOLLOWUP_TOOLS = _READING_PRIMITIVES | frozenset((
    "decode_plan_word_repair",
    "decode_plan_word_repair_menu",
    "repair_agenda_update",
))
# Textual signals that an assistant turn is voicing an actionable
# reading-driven fix hypothesis. Keep this stricter than a plain arrow search:
# reading narration often says "X -> likely Y" as analysis, not as a repair
# commitment.
_READING_HINTS = ("→", "->", "should be", "should map")
_READING_ACTION_RE = re.compile(
    r"(should\s+(?:be|map|stay)|need(?:s|ed)?\s+(?:to\s+)?(?:be\s+)?"
    r"|fix(?:es|ing)?|repair(?:s|ing)?|apply|change"
    r"|key repairs? i can read|clearest (?:word )?repairs?)",
    re.IGNORECASE,
)
_READING_SPECULATIVE_RE = re.compile(
    r"(likely|probably|may be|could be|unclear|\?|or similar|or [A-Z]{2,})",
    re.IGNORECASE,
)
_READING_RETROSPECTIVE_RE = re.compile(
    r"(?:\b(?:was|were)\s+applied\b|\bapplied\b|conflicted|conflict"
    r"|problematic|broke|worse|badly|revert(?:ed)?|reject(?:ed)?"
    r"|declare now|declaring|declared)",
    re.IGNORECASE,
)
_READING_FORWARD_ACTION_RE = re.compile(
    r"(?:let me|i(?:'ll| will| need)|need(?:s)?\s+to"
    r"|should\s+(?:be|map)|apply(?:ing)?\s+(?:the|this)?"
    r"|change\s+\S+\s+to|fix\s+\S+\s+to|repair\s+\S+\s+to)",
    re.IGNORECASE,
)


def summarize_findings(findings: list[ArtifactFinding]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.label] = counts.get(finding.label, 0) + 1
    return {
        "finding_count": len(findings),
        "labels": counts,
        "findings": [finding.to_dict() for finding in findings],
    }


def _likely_homophonic_no_boundary(artifact: dict[str, Any]) -> bool:
    alphabet_size = int(artifact.get("cipher_alphabet_size") or 0)
    word_count = int(artifact.get("cipher_word_count") or 0)
    return alphabet_size > 26 and word_count <= 1


def _check_homophonic_search_order(
    tool_calls: list[dict[str, Any]],
) -> list[ArtifactFinding]:
    findings: list[ArtifactFinding] = []
    search_calls = [
        call for call in tool_calls
        if str(call.get("tool_name", "")).startswith("search_")
    ]
    if not search_calls:
        findings.append(ArtifactFinding(
            label="tool_missing",
            severity="error",
            message=(
                "Likely homophonic no-boundary cipher did not call any search "
                "tool."
            ),
        ))
        return findings

    first_search = search_calls[0]
    if first_search.get("tool_name") != "search_homophonic_anneal":
        findings.append(ArtifactFinding(
            label="agent_wrong_tool",
            severity="error",
            message=(
                "Likely homophonic no-boundary cipher used a non-specialized "
                "search tool before search_homophonic_anneal."
            ),
            iteration=first_search.get("iteration"),
            tool_name=first_search.get("tool_name"),
            evidence={"first_search_tool": first_search.get("tool_name")},
        ))

    first_homophonic = next(
        (
            call for call in tool_calls
            if call.get("tool_name") == "search_homophonic_anneal"
        ),
        None,
    )
    if first_homophonic and int(first_homophonic.get("iteration") or 0) > 2:
        findings.append(ArtifactFinding(
            label="agent_wrong_tool",
            severity="warning",
            message=(
                "search_homophonic_anneal was delayed despite opening facts "
                "being sufficient to infer a likely homophonic no-boundary "
                "cipher."
            ),
            iteration=first_homophonic.get("iteration"),
            tool_name="search_homophonic_anneal",
            evidence={"first_homophonic_search_iteration": first_homophonic.get("iteration")},
        ))
    return findings


def _check_homophonic_global_swaps(
    tool_calls: list[dict[str, Any]],
) -> list[ArtifactFinding]:
    findings = []
    for call in tool_calls:
        if call.get("tool_name") != "act_swap_decoded":
            continue
        findings.append(ArtifactFinding(
            label="agent_wrong_tool",
            severity="warning",
            message=(
                "Likely homophonic no-boundary run used act_swap_decoded; "
                "targeted act_set_mapping is usually safer for homophones."
            ),
            iteration=call.get("iteration"),
            tool_name="act_swap_decoded",
            evidence={"arguments": call.get("arguments", {})},
        ))
    return findings


def _check_worsening_mutations(
    tool_calls: list[dict[str, Any]],
) -> list[ArtifactFinding]:
    findings: list[ArtifactFinding] = []
    for call in tool_calls:
        result = _result_dict(call)
        delta = result.get("score_delta") if isinstance(result, dict) else None
        if not isinstance(delta, dict) or delta.get("verdict") != "worse":
            continue
        status = result.get("status")
        label = "premature_declaration" if status != "reverted" else "scoring_false_positive"
        severity = "error" if status != "reverted" else "info"
        message = (
            "Mutating tool made the branch score worse and did not revert."
            if status != "reverted"
            else "Mutating tool tried a worsening repair but reverted it."
        )
        findings.append(ArtifactFinding(
            label=label,
            severity=severity,
            message=message,
            iteration=call.get("iteration"),
            tool_name=call.get("tool_name"),
            evidence={"score_delta": delta, "status": status},
        ))
    return findings


def _check_declaration_after_worse_mutation(
    tool_calls: list[dict[str, Any]],
) -> list[ArtifactFinding]:
    last_worse: dict[str, Any] | None = None
    for call in tool_calls:
        result = _result_dict(call)
        delta = result.get("score_delta") if isinstance(result, dict) else None
        if isinstance(delta, dict) and delta.get("verdict") == "worse":
            if result.get("status") != "reverted":
                last_worse = call
        if call.get("tool_name") != "meta_declare_solution" or last_worse is None:
            continue
        return [ArtifactFinding(
            label="premature_declaration",
            severity="error",
            message=(
                "Solution was declared after a non-reverted worsening repair. "
                "The run should restore the earlier branch or re-run repair "
                "before declaring."
            ),
            iteration=call.get("iteration"),
            tool_name="meta_declare_solution",
            evidence={
                "worsening_iteration": last_worse.get("iteration"),
                "worsening_tool": last_worse.get("tool_name"),
            },
        )]
    return []


def _check_declared_branch_accuracy(artifact: dict[str, Any]) -> list[ArtifactFinding]:
    solution = artifact.get("solution")
    branches = artifact.get("branches", [])
    if not isinstance(solution, dict) or not branches:
        return []
    declared = solution.get("branch")
    scored = [
        b for b in branches
        if isinstance(b, dict) and isinstance(b.get("char_accuracy"), (int, float))
    ]
    if not scored:
        return []
    best = max(scored, key=lambda b: float(b.get("char_accuracy", 0.0)))
    declared_branch = next((b for b in scored if b.get("name") == declared), None)
    if not declared_branch:
        return []
    best_acc = float(best.get("char_accuracy", 0.0))
    declared_acc = float(declared_branch.get("char_accuracy", 0.0))
    if best.get("name") != declared and best_acc - declared_acc > 0.01:
        return [ArtifactFinding(
            label="premature_declaration",
            severity="error",
            message="Declared branch was not the best final branch by character accuracy.",
            iteration=solution.get("declared_at_iteration"),
            tool_name="meta_declare_solution",
            evidence={
                "declared_branch": declared,
                "declared_char_accuracy": declared_acc,
                "best_branch": best.get("name"),
                "best_char_accuracy": best_acc,
            },
        )]
    return []


def _result_dict(call: dict[str, Any]) -> dict[str, Any]:
    result = call.get("result", {})
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _check_score_overrode_reading(
    tool_calls: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> list[ArtifactFinding]:
    """Flag reading-driven mapping changes that were dropped despite
    producing multiple word-level edits.

    A reading-driven primitive (`act_set_mapping`, `act_bulk_set`,
    `act_anchor_word`) that touches >= 2 decoded words but reports a
    negative `dictionary_rate` delta is the textbook score-vs-reading
    conflict on boundary-preserved ciphers. If the run later declares a
    *different* branch, the agent overrode its own reading on the strength
    of the score signal — the failure pattern this label exists to track.
    """
    findings: list[ArtifactFinding] = []
    solution = artifact.get("solution", {})
    declared_branch = (
        solution.get("branch") if isinstance(solution, dict) else None
    )

    for call in tool_calls:
        name = call.get("tool_name")
        if name not in _READING_PRIMITIVES:
            continue
        result = _result_dict(call)
        delta = result.get("score_delta") if isinstance(result, dict) else None
        if not isinstance(delta, dict):
            continue
        dict_delta = delta.get("dict_rate_delta")
        if dict_delta is None or dict_delta >= 0:
            continue
        changed = result.get("changed_words")
        if not isinstance(changed, list) or len(changed) < 2:
            continue
        branch = result.get("branch") or (
            (call.get("arguments") or {}).get("branch")
        )
        if declared_branch and branch == declared_branch:
            # Change was kept on the declared branch — no override pattern.
            continue
        findings.append(ArtifactFinding(
            label="score_overrode_reading",
            severity="warning",
            message=(
                "Reading-driven mapping change moved >= 2 decoded words "
                "but lowered dictionary_rate; the run later declared a "
                "different branch. Likely score-vs-reading conflict — see "
                "AGENTS.md → 'Reading-driven repair discipline'."
            ),
            iteration=call.get("iteration"),
            tool_name=name,
            evidence={
                "branch": branch,
                "declared_branch": declared_branch,
                "score_delta": delta,
                "changed_words_count": len(changed),
                "changed_words_sample": changed[:4],
            },
        ))
    return findings


def _check_unattempted_reading_fix(
    artifact: dict[str, Any],
) -> list[ArtifactFinding]:
    """Flag iterations where assistant reasoning voices a reading-driven
    fix hypothesis but no mutation primitive follows within two iterations.

    Looks for explicit action cues, not every arrow in ordinary reading
    narration. Iterations are tracked by counting assistant turns; tool calls
    in the next two assistant turns are checked for any reading-driven
    primitive or durable repair-agenda follow-up.
    """
    findings: list[ArtifactFinding] = []
    turns = list(_walk_assistant_turns(artifact))
    for i, (iter_num, text, tool_names) in enumerate(turns):
        if not _voices_actionable_reading_fix(text):
            continue
        # Hypothesis voiced in this turn — does any mutation follow within
        # this or the next two assistant turns?
        followed = any(
            tn in _READING_FOLLOWUP_TOOLS
            for j in range(i, min(i + 3, len(turns)))
            for tn in turns[j][2]
        )
        if followed:
            continue
        snippet = text.strip().replace("\n", " ")
        findings.append(ArtifactFinding(
            label="unattempted_reading_fix",
            severity="warning",
            message=(
                "Assistant reasoning voiced a reading-driven fix but no "
                "mapping, word-repair plan, or agenda update followed within "
                "two iterations."
            ),
            iteration=iter_num,
            evidence={"sample": snippet[:300]},
        ))
    return findings


def _voices_actionable_reading_fix(text: str) -> bool:
    if not text or not any(p in text for p in _READING_HINTS):
        return False
    if re.search(
        r"(cipher\s+\S+\s+should\s+(?:be|map)|currently\s*\S*\s*→?\s*\S*"
        r"\s+should\s+(?:be|map)|should\s+map|should\s+stay)",
        text,
        re.IGNORECASE,
    ):
        return True
    for match in re.finditer(r"→|->", text):
        start = max(0, match.start() - 140)
        end = min(len(text), match.end() + 180)
        window = text[start:end]
        if re.search(r"\bboundar(?:y|ies)\b", window, re.IGNORECASE):
            continue
        if (
            _READING_RETROSPECTIVE_RE.search(window)
            and (
                not _READING_FORWARD_ACTION_RE.search(window)
                or re.search(r"(?:let me\s+declare|declare now)", window, re.IGNORECASE)
            )
        ):
            continue
        if not _READING_ACTION_RE.search(window):
            continue
        if _READING_SPECULATIVE_RE.search(window) and not re.search(
            r"(key repairs? i can read|clearest)",
            window,
            re.IGNORECASE,
        ):
            continue
        return True
    return False


def _walk_assistant_turns(
    artifact: dict[str, Any],
):
    """Yield (iteration_number, text_content, tool_names) for each
    assistant turn in the run.

    Iteration numbers are derived from sequence order — assistant turn N
    corresponds to iteration N. This matches v2 loop semantics where each
    iteration sends one assistant message, optionally including tool_use
    blocks.
    """
    iter_num = 0
    for msg in artifact.get("messages", []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        iter_num += 1
        text_parts: list[str] = []
        tool_names: list[str] = []
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(str(block.get("text", "")))
                elif btype == "tool_use":
                    tool_names.append(str(block.get("name", "")))
        elif isinstance(content, str):
            text_parts.append(content)
        yield iter_num, "\n".join(text_parts), tool_names


def _check_loop_events(artifact: dict[str, Any]) -> list[ArtifactFinding]:
    findings: list[ArtifactFinding] = []
    for event in artifact.get("loop_events", []) or []:
        if not isinstance(event, dict):
            continue
        name = event.get("event")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if name == "gated_tool_retry":
            findings.append(ArtifactFinding(
                label="gated_tool_retry",
                severity="info",
                message=(
                    "The loop caught a disallowed tool choice and retried "
                    "inside the same outer iteration."
                ),
                iteration=event.get("outer_iteration") or payload.get("iteration"),
                evidence={
                    "inner_step": event.get("inner_step") or payload.get("inner_step"),
                    "mode": event.get("mode"),
                    "attempted_tools": payload.get("attempted_tools", []),
                    "allowed_tools": payload.get("allowed_tools", []),
                },
            ))
        elif name == "boundary_projection_count_retry":
            findings.append(ArtifactFinding(
                label="same_length_projection_failed",
                severity="warning",
                message=(
                    "A full-reading/boundary projection proposal had the wrong "
                    "normalized character count and required an immediate retry."
                ),
                iteration=event.get("outer_iteration") or payload.get("iteration"),
                evidence={
                    "inner_step": event.get("inner_step") or payload.get("inner_step"),
                    "mode": event.get("mode"),
                    "attempted_tools": payload.get("attempted_tools", []),
                    "allowed_tools": payload.get("allowed_tools", []),
                },
            ))
    return findings


def _check_repair_agenda_unresolved(artifact: dict[str, Any]) -> list[ArtifactFinding]:
    agenda = artifact.get("repair_agenda", []) or []
    if not isinstance(agenda, list):
        return []
    unresolved_statuses = {"open", "blocked"}
    unresolved = [
        item for item in agenda
        if isinstance(item, dict) and item.get("status") in unresolved_statuses
    ]
    if not unresolved:
        return []
    solution = artifact.get("solution") or {}
    iteration = (
        solution.get("declared_at_iteration")
        if isinstance(solution, dict)
        else None
    )
    return [ArtifactFinding(
        label="repair_agenda_unresolved",
        severity="warning",
        message=(
            "Run ended with unresolved reading-repair agenda items. The agent "
            "should apply, reject, or explicitly hold these before declaration."
        ),
        iteration=iteration,
        evidence={
            "unresolved_count": len(unresolved),
            "items": unresolved[:5],
        },
    )]


def _check_projection_failures(
    tool_calls: list[dict[str, Any]],
) -> list[ArtifactFinding]:
    findings: list[ArtifactFinding] = []
    for call in tool_calls:
        name = call.get("tool_name")
        if name not in {
            "decode_validate_reading_repair",
            "act_resegment_by_reading",
            "act_resegment_from_reading_repair",
            "act_resegment_window_by_reading",
        }:
            continue
        result = _result_dict(call)
        if not result:
            continue
        if result.get("same_character_count") is not False:
            projection = result.get("boundary_projection")
            if not (
                isinstance(projection, dict)
                and projection.get("applicable") is False
                and "character counts differ" in str(projection.get("reason", ""))
            ):
                continue
        findings.append(ArtifactFinding(
            label="same_length_projection_failed",
            severity="warning",
            message=(
                "A reading/projection tool received a proposal whose normalized "
                "character count did not match the current branch."
            ),
            iteration=call.get("iteration"),
            tool_name=name,
            evidence={
                "current_char_count": result.get("current_char_count"),
                "proposed_char_count": result.get("proposed_char_count"),
                "error": result.get("error"),
            },
        ))
    return findings
