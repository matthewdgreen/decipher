"""Helpers for continuing an agentic run from a saved artifact."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from benchmark.loader import parse_canonical_transcription
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


def load_artifact_dict(path: str | Path) -> dict[str, Any]:
    with open(Path(path), encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Artifact must be a JSON object: {path}")
    return data


def cipher_text_from_artifact(artifact: dict[str, Any]) -> CipherText:
    """Reconstruct the original cipher text from the artifact's first prompt."""
    raw = _extract_cipher_block(artifact)
    if not raw:
        raise ValueError(
            "Could not find the manuscript notation block in the artifact. "
            "Use a newer agentic artifact or resume through the benchmark path."
        )
    if "|" in raw or _looks_like_space_delimited_tokens(raw):
        return parse_canonical_transcription(raw)

    ignore = {" ", "\t", "\n", "\r"}
    alphabet = Alphabet.from_text(raw, ignore_chars=ignore)
    clean = " ".join(raw.split())
    separator = " " if " " in clean else None
    return CipherText(raw=clean, alphabet=alphabet, source="artifact_resume", separator=separator)


def install_resume_branches(
    workspace: Workspace,
    artifact: dict[str, Any],
    *,
    branch: str | None = None,
) -> list[str]:
    """Restore branch snapshots from a prior artifact into a workspace."""
    branches = artifact.get("branches") or []
    if not isinstance(branches, list) or not branches:
        raise ValueError("Prior artifact contains no branch snapshots to resume from.")

    restored: list[str] = []
    for snap in branches:
        if not isinstance(snap, dict):
            continue
        name = str(snap.get("name") or "")
        if not name:
            continue
        key = _int_key_dict(snap.get("key") or {})
        spans = _word_spans_from_snapshot(snap)
        workspace.restore_branch(
            name,
            key=key,
            parent=snap.get("parent"),
            created_iteration=int(snap.get("created_iteration") or 0),
            tags=[str(t) for t in snap.get("tags") or []],
            word_spans=spans,
        )
        restored.append(name)

    if branch and branch not in restored:
        available = ", ".join(_branch_names(artifact))
        raise ValueError(f"Branch {branch!r} not found in artifact. Available: {available}")
    return restored


def resume_context_from_artifact(
    artifact: dict[str, Any],
    *,
    branch: str | None = None,
    extra_iterations: int,
) -> str:
    """Build a compact prior-run briefing for the next agent call."""
    solution = artifact.get("solution") or {}
    declared_branch = branch or solution.get("branch") or _best_branch_name(artifact) or "main"
    branch_snap = _branch_snapshot(artifact, declared_branch)
    final_summary = str(artifact.get("final_summary") or "").strip()
    if not final_summary and solution:
        final_summary = _solution_fallback_summary(solution)

    lines = [
        "This is a continuation of a previous Decipher agent run.",
        f"Previous artifact run_id: {artifact.get('run_id', '-')}.",
        f"Previous status: {artifact.get('status', '-')}.",
        f"Previous model: {artifact.get('model', '-')}.",
        f"New continuation budget: {extra_iterations} additional outer iteration(s).",
        "",
        "Do not restart from scratch. The workspace has been restored from the "
        "saved branch snapshots. Start by inspecting the restored branch cards, "
        "then continue the unresolved reading or repair work.",
        "",
        f"Suggested starting branch: `{declared_branch}`.",
    ]

    if solution:
        lines.extend([
            "",
            "Previous declaration:",
            f"- Branch: {solution.get('branch', '-')}",
            f"- Confidence: {solution.get('self_confidence', '-')}",
            f"- Rationale: {_short(solution.get('rationale', ''), 900)}",
        ])
        reading_summary = str(solution.get("reading_summary") or "").strip()
        if reading_summary:
            lines.append(f"- Reading summary: {_short(reading_summary, 700)}")
        helpful = solution.get("further_iterations_helpful")
        note = str(solution.get("further_iterations_note") or "").strip()
        if helpful is not None or note:
            lines.append(f"- Further iterations helpful: {helpful}. {note}")

    if final_summary:
        lines.extend(["", "Previous final summary:", _short(final_summary, 1400)])

    if branch_snap:
        lines.extend([
            "",
            f"Restored branch `{declared_branch}`:",
            f"- Mapped symbols: {branch_snap.get('mapped_count', '-')}",
            f"- Prior char accuracy, if known: {branch_snap.get('char_accuracy', '-')}",
            f"- Prior word accuracy, if known: {branch_snap.get('word_accuracy', '-')}",
            "- Decryption preview:",
            "```",
            _short(str(branch_snap.get("decryption") or ""), 1600),
            "```",
        ])

    agenda = artifact.get("repair_agenda") or []
    if isinstance(agenda, list) and agenda:
        active = [
            item for item in agenda
            if isinstance(item, dict)
            and item.get("status") in {"open", "blocked", "held"}
        ]
        if active:
            lines.extend(["", "Prior repair agenda items to revisit or explicitly close:"])
            for item in active[:8]:
                lines.append(
                    f"- #{item.get('id')}: {item.get('from')} -> {item.get('to')} "
                    f"on `{item.get('branch')}` ({item.get('status')}); "
                    f"{_short(item.get('notes', ''), 180)}"
                )

    tool_requests = artifact.get("tool_requests") or []
    if isinstance(tool_requests, list) and tool_requests:
        lines.extend(["", "Prior missing-tool requests:"])
        for req in tool_requests[:5]:
            if isinstance(req, dict):
                lines.append(f"- {_short(req.get('description', ''), 220)}")

    lines.extend([
        "",
        "Continuation rule: prefer low-cost inspection and targeted repair on "
        "restored branches. Only rerun broad automated search if the restored "
        "state is clearly unusable or you need a fresh comparison branch.",
    ])
    return "\n".join(lines)


def inherited_repair_agenda(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    agenda = artifact.get("repair_agenda") or []
    if not isinstance(agenda, list):
        return []
    return [dict(item) for item in agenda if isinstance(item, dict)]


def _extract_cipher_block(artifact: dict[str, Any]) -> str:
    messages = artifact.get("messages") or []
    if not messages:
        return ""
    content = messages[0].get("content") if isinstance(messages[0], dict) else ""
    if isinstance(content, list):
        text = "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text = str(content or "")
    match = re.search(
        r"## The manuscript notation system\s*```(?:\w+)?\n(.*?)\n```",
        text,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    blocks = re.findall(r"```(?:\w+)?\n(.*?)\n```", text, flags=re.DOTALL)
    return blocks[0].strip() if blocks else ""


def _looks_like_space_delimited_tokens(raw: str) -> bool:
    tokens = raw.replace("\n", " ").split()
    return len(tokens) > 1 and all(len(tok) > 1 for tok in tokens[:20])


def _int_key_dict(raw: Any) -> dict[int, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, int] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _word_spans_from_snapshot(snap: dict[str, Any]) -> list[tuple[int, int]] | None:
    raw = snap.get("word_spans")
    if raw is None:
        return None
    spans: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            spans.append((int(item[0]), int(item[1])))
    return spans or None


def _branch_names(artifact: dict[str, Any]) -> list[str]:
    return [
        str(b.get("name"))
        for b in artifact.get("branches") or []
        if isinstance(b, dict) and b.get("name")
    ]


def _branch_snapshot(artifact: dict[str, Any], name: str) -> dict[str, Any] | None:
    for branch in artifact.get("branches") or []:
        if isinstance(branch, dict) and branch.get("name") == name:
            return branch
    return None


def _best_branch_name(artifact: dict[str, Any]) -> str:
    branches = [b for b in artifact.get("branches") or [] if isinstance(b, dict)]
    if not branches:
        return ""
    scored = [
        b for b in branches
        if isinstance(b.get("char_accuracy"), (int, float))
    ]
    if scored:
        return str(max(scored, key=lambda b: float(b.get("char_accuracy") or 0.0)).get("name") or "")
    return str(branches[0].get("name") or "")


def _solution_fallback_summary(solution: dict[str, Any]) -> str:
    parts = []
    if solution.get("reading_summary"):
        parts.append(str(solution["reading_summary"]))
    if solution.get("rationale"):
        parts.append(str(solution["rationale"]))
    if solution.get("further_iterations_note"):
        parts.append(str(solution["further_iterations_note"]))
    return "\n".join(parts)


def _short(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
