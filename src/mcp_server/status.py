"""The status brief + advisory rendering (Part 6).

``build_status_brief`` composes ONE text blob by reusing the factual section
renderers from ``investigation/context.py`` and REPLACING the binding workflow
menu with an advisory block whose lines carry policy ids (proposal 4.3). The
brief is the self-briefing a bare session reads once to be fully oriented.
"""
from __future__ import annotations

from typing import Any

from investigation import context
from investigation.context import (
    DEFAULT_BRANCH_CARDS,
    DEFAULT_EVIDENCE_ENTRIES,
    DEFAULT_WINDOW_TOKENS,
    _attestation_route,
    _exhausted_entry_for,
    _fresh_attestation,
    _render_branch_cards,
    _render_cipher,
    _render_episode_ledger,
    _render_evidence,
    _render_experiment_queue,
    _render_external_context,
    _render_fingerprint,
    _render_framing,
    _render_hypothesis_board,
    _render_readings,
    _render_window,
    workflow_state,
)
from investigation.state import attestation_is_positive

_ADVISORY_HEADER = (
    "## Host guidance (advisory unless marked ENFORCED; policy ids per "
    "docs/mcp_policy_provenance_ledger.md)"
)


def _render_comparisons(records: dict) -> str:
    """Last 3 sidecar client comparisons, one line each (Part 6 section 11)."""
    comparisons = list(records.get("comparisons") or [])[-3:]
    if not comparisons:
        return ""
    lines = []
    for record in comparisons:
        ranking = " > ".join(str(b) for b in record.get("ranking") or [])
        best = record.get("best_partial")
        best_str = f"`{best}`" if best else "None"
        accepts = "true" if record.get("accepts_as_solution") else "false"
        rationale = str(record.get("rationale") or "")[:120]
        lines.append(
            f"- [t{record.get('created_turn')}] ranking: {ranking}; "
            f"best_partial={best_str}; accepts_as_solution={accepts} — {rationale}"
        )
    return "## Client comparisons\n" + "\n".join(lines)


def _render_advisory(
    state: Any, executor: Any, verification_available: bool
) -> str:
    menu = workflow_state(state, executor)
    branch = menu.get("branch")
    lines = [_ADVISORY_HEADER]

    # 1. Workflow recommendation (WF-1, advisory).
    focus = f"- [WF-1 advisory] Suggested focus: {menu['state']}"
    if branch:
        focus += f" on `{branch}`"
    lines.append(focus)
    for action in menu.get("actions") or []:
        lines.append(f"    - {action}")

    # The workflow branch's fresh attestation drives WF-4 / WF-5 / SAT-3.
    workflow_att = _fresh_attestation(state, branch) if branch else None

    # 2. Verifier route (WF-4, advisory): fresh NON-positive attestation.
    if workflow_att is not None and not attestation_is_positive(workflow_att):
        route = _attestation_route(workflow_att)
        tlc = float(workflow_att.get("target_language_confidence") or 0.0)
        recov = float(workflow_att.get("semantic_recoverability") or 0.0)
        scope = str(workflow_att.get("damage_scope") or "basin_wide")
        lines.append(
            f"- [WF-4 advisory] Independent-reader route for `{branch}`: {route} "
            f"(target_language_confidence={tlc:.2f}, "
            f"semantic_recoverability={recov:.2f}, damage_scope={scope})"
        )

    # 3. Declare hint (WF-5, advisory): any branch with a fresh POSITIVE attestation.
    positive_branch = next(
        (
            name for name in state.workspace.branch_names()
            if attestation_is_positive(_fresh_attestation(state, name))
        ),
        None,
    )
    if positive_branch is not None:
        lines.append(
            f"- [WF-5 advisory] `{positive_branch}` has a fresh positive "
            "verification; declare it unless you hold concrete contradictory "
            "evidence."
        )

    # 4. Repair hint (WF-5, advisory): workflow branch, fresh non-positive, route repair.
    if (
        workflow_att is not None
        and not attestation_is_positive(workflow_att)
        and _attestation_route(workflow_att) == "repair"
    ):
        lines.append(
            f"- [WF-5 advisory] Record a reading of `{branch}` (reading_record), "
            "compile hypotheses (repair_hypotheses_test), then repair_transaction; "
            "reverify the changed content."
        )

    # 5. Saturation (SAT-3, ENFORCED).
    if _exhausted_entry_for(state, menu.get("branch"), workflow_att) is not None:
        lines.append(
            "- [SAT-3 ENFORCED] Repair is exhausted for this candidate content "
            "and verifier evidence; repair_transaction will be blocked until "
            "content or verifier evidence changes. Alternate search "
            "(experiment_submit), compare distinct finalists, or declare "
            "honestly unsolved."
        )

    # 6. Declaration gate (DECL-1, ENFORCED): always.
    lines.append(
        "- [DECL-1 ENFORCED] meta_declare_solution requires a fresh POSITIVE "
        "independent verification of the branch's current content. "
        "meta_declare_unsolved is never gated (DECL-8)."
    )

    # 7. Keyless (only when verification unavailable).
    if not verification_available:
        lines.append(
            "- [DECL-1 ENFORCED] Independent verification is UNAVAILABLE (no "
            "server-side API key). The solved-declaration gate stays closed; "
            "your strongest candidate remains available via candidate_list, "
            "labeled 'promising but not independently verified'. Configure a "
            "provider key (OPENAI_API_KEY / ANTHROPIC_API_KEY / "
            ".decipher_keys/<provider>_api_key or keychain service 'decipher'), "
            "restart the MCP server, and resume — state is preserved."
        )

    # 8. Portfolio note (WF-6, divergence): always.
    lines.append(
        "- [WF-6] Candidate attention is yours: candidate_list shows every "
        "branch with labeled signals; no scalar silently selects 'the' candidate."
    )
    return "\n".join(lines)


def build_status_brief(
    state: Any,
    executor: Any,
    host: Any,
    *,
    turn: int,
    revision: int,
    verification_available: bool,
    max_cost_usd: float,
    records: dict | None = None,
) -> str:
    """Compose the full self-briefing text blob (pure; deterministic)."""
    records = records or {}
    investigation_line = (
        f"## Investigation state (turn {turn}, revision {revision})\n"
        f"Budget: ${host.committed_cost():.4f} of ${max_cost_usd:.2f} "
        "server-side ceiling (BUD-1).\n"
        "Independent verification: "
        + (
            "available"
            if verification_available
            else "UNAVAILABLE — no API key (see advisory below)"
        )
        + "."
    )
    sections = [
        _render_framing(state),
        _render_cipher(state),
        _render_fingerprint(state),
        _render_external_context(state),
        investigation_line,
        _render_branch_cards(state, executor, DEFAULT_BRANCH_CARDS),
        _render_hypothesis_board(state),
        _render_episode_ledger(state),
        _render_experiment_queue(state),
        _render_readings(state),
        _render_comparisons(records),
        _render_evidence(state, DEFAULT_EVIDENCE_ENTRIES),
        _render_window(state, executor, turn, DEFAULT_WINDOW_TOKENS),
        _render_advisory(state, executor, verification_available),
    ]
    return "\n\n".join(section for section in sections if section)
