"""Lightweight gap analysis for v2 agent run artifacts."""
from __future__ import annotations

import json
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
    return findings


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
