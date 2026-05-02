"""Human-readable final summaries for agentic decipherment runs."""
from __future__ import annotations

import re
from typing import Any

from artifact.schema import RunArtifact


LANGUAGE_LABELS = {
    "en": "English",
    "la": "Latin",
    "de": "German",
    "fr": "French",
    "it": "Italian",
}


def build_final_summary(
    artifact: RunArtifact,
    *,
    final_branch: str,
    final_decryption: str,
) -> str:
    """Build a compact final-screen summary from the run artifact.

    This is deliberately deterministic and artifact-local. The agent can
    provide the best summary through meta_declare_solution/meta_declare_unsolved,
    but fallback text should still be useful for auto-declared/error runs.
    """
    solution = artifact.solution
    language = LANGUAGE_LABELS.get(artifact.language, artifact.language or "unknown")
    lines: list[str] = []
    lines.append(f"Target language: {language}.")
    if artifact.language and artifact.language != "en":
        lines.append(
            "Reading note: this is an agent summary of the target-language "
            "text, not a certified translation."
        )

    declaration_attempt = _latest_declaration_attempt(artifact)
    reading_summary = (solution.reading_summary if solution else "") or ""
    if not reading_summary and declaration_attempt:
        reading_summary = str(declaration_attempt.get("reading_summary") or "")
    rationale = (solution.rationale if solution else "") or ""
    attempted_rationale = str((declaration_attempt or {}).get("rationale") or "")
    if reading_summary.strip():
        lines.append("")
        lines.append("What it appears to say:")
        lines.append(_clean_sentence(reading_summary))
    elif attempted_rationale.strip() and _is_fallback_rationale(rationale):
        lines.append("")
        lines.append("What it appears to say:")
        lines.append(_summary_from_rationale(attempted_rationale))
    elif rationale.strip():
        lines.append("")
        lines.append("What it appears to say:")
        lines.append(_summary_from_rationale(rationale))
    elif final_decryption.strip():
        lines.append("")
        lines.append("What it appears to say:")
        lines.append("No separate reading summary was provided; inspect the final decrypt.")

    lines.append("")
    confidence = solution.self_confidence if solution else None
    status_label = "Declared branch" if artifact.status != "unsolved" else "Best inspected branch"
    status_bits = [f"{status_label}: {final_branch or '-'}."]
    if confidence is not None:
        status_bits.append(f"Agent confidence: {confidence:.2f}.")
    status_bits.append(f"Run status: {artifact.status}.")
    lines.append("Status:")
    lines.append(" ".join(status_bits))

    process_notes = _process_notes(artifact)
    if process_notes:
        lines.append("")
        lines.append("Process notes:")
        lines.extend(f"- {note}" for note in process_notes[:5])

    helpful, help_note = _further_iterations_assessment(artifact)
    if solution and solution.further_iterations_helpful is not None:
        helpful = solution.further_iterations_helpful
    elif declaration_attempt and declaration_attempt.get("further_iterations_helpful") is not None:
        helpful = bool(declaration_attempt["further_iterations_helpful"])
    if solution and solution.further_iterations_note.strip():
        help_note = solution.further_iterations_note.strip()
    elif declaration_attempt and str(declaration_attempt.get("further_iterations_note") or "").strip():
        help_note = str(declaration_attempt["further_iterations_note"]).strip()
    lines.append("")
    lines.append("Further iterations:")
    lines.append(("Likely helpful. " if helpful else "Probably not essential. ") + help_note)
    return "\n".join(lines).strip()


def _clean_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:900] + ("..." if len(text) > 900 else "")


def _summary_from_rationale(rationale: str) -> str:
    text = _clean_sentence(rationale)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    useful = [
        s for s in sentences
        if re.search(
            r"\b(text|passage|decodes|appears|about|coherent|subject|latin|german|french|italian)\b",
            s,
            re.IGNORECASE,
        )
    ]
    chosen = useful[:3] or sentences[:2]
    return _clean_sentence(" ".join(chosen))


def _latest_declaration_attempt(artifact: RunArtifact) -> dict[str, Any] | None:
    """Return the latest attempted declaration args, including blocked calls."""
    for call in reversed(artifact.tool_calls):
        if call.tool_name in {"meta_declare_solution", "meta_declare_unsolved"}:
            return dict(call.arguments)
    return None


def _is_fallback_rationale(rationale: str) -> bool:
    return "Automatic fallback declaration" in rationale


def _process_notes(artifact: RunArtifact) -> list[str]:
    notes: list[str] = []
    if artifact.status in {"error", "exhausted", "unsolved"}:
        notes.append(f"The run ended as {artifact.status}; treat the result as partial.")
    if any(e.event == "auto_declared_solution" for e in artifact.loop_events):
        notes.append("The branch was auto-declared by fallback rather than explicitly declared by the agent.")
    unresolved = [
        item for item in artifact.repair_agenda
        if item.get("status") in {"open", "blocked"}
    ]
    held = [
        item for item in artifact.repair_agenda
        if item.get("status") == "held"
    ]
    if unresolved:
        notes.append(f"{len(unresolved)} repair agenda item(s) remained open or blocked.")
    if held:
        examples = ", ".join(
            f"{item.get('from')} -> {item.get('to')}" for item in held[:3]
        )
        notes.append(f"Held repair(s): {examples}.")
    projection_failures = [
        e for e in artifact.loop_events
        if e.event == "boundary_projection_count_retry"
    ]
    if projection_failures:
        notes.append(
            f"{len(projection_failures)} boundary/reading projection attempt(s) had character-count mismatches."
        )
    if artifact.tool_requests:
        notes.append(f"The agent requested {len(artifact.tool_requests)} missing or better tool capability item(s).")
    return notes


def _further_iterations_assessment(artifact: RunArtifact) -> tuple[bool, str]:
    solution = artifact.solution
    confidence = solution.self_confidence if solution else 0.0
    unresolved = any(
        item.get("status") in {"open", "blocked"} for item in artifact.repair_agenda
    )
    held = any(item.get("status") == "held" for item in artifact.repair_agenda)
    projection_failures = any(
        e.event == "boundary_projection_count_retry" for e in artifact.loop_events
    )
    auto_declared = any(e.event == "auto_declared_solution" for e in artifact.loop_events)
    declared_at_limit = bool(solution and solution.declared_at_iteration >= artifact.max_iterations)
    rationale = (solution.rationale if solution else "").lower()
    says_partial = any(
        word in rationale for word in
        ("partial", "remaining", "blocked", "uncertain", "likely", "boundary", "further")
    )

    helpful = (
        artifact.status != "solved"
        or confidence < 0.85
        or unresolved
        or held
        or projection_failures
        or auto_declared
        or declared_at_limit
        or says_partial
    )
    if helpful:
        reasons: list[str] = []
        if held or unresolved:
            reasons.append("resolve held/open reading repairs")
        if projection_failures:
            reasons.append("retry boundary/reading projection with corrected character counts")
        if confidence < 0.85:
            reasons.append("agent confidence is below 0.85")
        if declared_at_limit:
            reasons.append("the declaration happened at the iteration limit")
        if not reasons:
            reasons.append("remaining uncertainty was noted in the declaration")
        return True, "Next work should " + "; ".join(reasons) + "."
    return False, "The agent did not report major unresolved repair work, and confidence was high."
