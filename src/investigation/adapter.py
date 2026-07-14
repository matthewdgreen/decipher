"""Adapter: build a v3 ``InvestigationState`` from a stored v2 ``RunArtifact``.

M6 Part A / F2 (the hard step). A v2 RunArtifact stores no structured cipher —
the cipher is reconstructed from the first prompt's manuscript-notation block,
using the SAME extraction ``agent/resume.py`` used. Those helpers are RELOCATED
(copied) here so this adapter carries no dependency on ``agent/resume`` (that
module is deletion-listed at v2 retirement).

Design decisions (F2, binding):
  (b) Plaintext alphabet = ``Alphabet.standard_english()`` — v2 never stored one;
      this pairing is what makes a branch snapshot's integer key values
      meaningful in the v3 state.
  (c) A STRUCTURED error (``V2ArtifactAdapterError``) is raised when the notation
      block cannot be extracted (old/compressed artifacts).
  (d) The cipher identity (run_id / cipher_id / model / prior status) lands in
      ``external_context`` — ``InvestigationState`` has no cipher_id field.
  branches restore via ``state._restore_branch_into`` (F9a: v2 snapshot dicts —
      which lack ``word_spans`` / ``token_order`` / ``metadata`` — restore
      cleanly; extra fields are ignored).
  Key v2 findings (the declaration and the final scores) become append-only
      evidence entries.

The adapter is READ-ONLY over the artifact; unknown/missing fields default (old
artifacts predate many fields). Purpose: ``resume-artifact`` keeps working on old
v2 artifacts after v2 retirement. Per F2e the scope here is the function + tests
ONLY — the ``cmd_resume_artifact`` CLI wiring ships with the retirement release.
"""
from __future__ import annotations

import re
from typing import Any

from benchmark.loader import parse_canonical_transcription
from investigation.state import InvestigationState, _restore_branch_into
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


class V2ArtifactAdapterError(ValueError):
    """Raised when a v2 artifact cannot be adapted to a v3 InvestigationState."""


def state_from_v2_artifact(artifact: dict[str, Any]) -> InvestigationState:
    """Reconstruct a v3 ``InvestigationState`` from a stored v2 RunArtifact dict.

    Raises ``V2ArtifactAdapterError`` if the artifact is a v3 artifact (which
    should load directly via ``InvestigationState.from_artifact_dict``) or if the
    cipher notation block cannot be extracted.
    """
    if not isinstance(artifact, dict):
        raise V2ArtifactAdapterError("Artifact must be a JSON object (dict).")
    if artifact.get("loop_version") == "v3":
        raise V2ArtifactAdapterError(
            "This is a v3 (investigation) artifact; adapt is for v2 artifacts "
            "only. Load it directly via InvestigationState.from_artifact_dict."
        )

    # (F2a) reconstruct the cipher from the first-prompt notation block, then
    # (F2b) pair it with the standard-English plaintext alphabet.
    cipher = _cipher_text_from_artifact(artifact)
    workspace = Workspace(
        cipher_text=cipher, plaintext_alphabet=Alphabet.standard_english()
    )

    # (F9a) restore every branch snapshot the v2 artifact carried. v2 snapshots
    # lack the three v3 fields (word_spans / token_order / transform_pipeline) and
    # a metadata dict; _restore_branch_into defaults them.
    for snap in artifact.get("branches") or []:
        if not isinstance(snap, dict) or not snap.get("name"):
            continue
        _restore_branch_into(workspace, snap)

    language = str(artifact.get("language") or "en")

    # (F2d) cipher identity -> external_context (never ground truth; the firewall
    # covers this surface, and it is rendered as its own stable context section).
    external_context = _external_context(artifact)

    state = InvestigationState(
        workspace=workspace,
        language=language,
        external_context=external_context,
    )

    # Key v2 findings become append-only evidence (turn 0 — they predate the run).
    _add_declaration_evidence(state, artifact)
    _add_final_scores_evidence(state, artifact)
    return state


# ---------------------------------------------------------------------------
# Evidence + context construction
# ---------------------------------------------------------------------------


def _external_context(artifact: dict[str, Any]) -> str:
    bits = [
        f"run_id={artifact.get('run_id', '-')}",
        f"cipher_id={artifact.get('cipher_id', '-')}",
        f"model={artifact.get('model', '-')}",
        f"prior_status={artifact.get('status', '-')}",
    ]
    return (
        "Resumed from a stored v2 (workspace-loop) artifact. Cipher identity: "
        + ", ".join(bits)
        + "."
    )


def _add_declaration_evidence(
    state: InvestigationState, artifact: dict[str, Any]
) -> None:
    solution = artifact.get("solution")
    if not isinstance(solution, dict):
        return
    state.add_evidence(
        "v2_declaration",
        turn=0,
        summary=(
            f"v2 declared branch {solution.get('branch', '-')!r} "
            f"(confidence {solution.get('self_confidence', '-')})"
        ),
        branch=solution.get("branch"),
        self_confidence=solution.get("self_confidence"),
        rationale=solution.get("rationale"),
        declared_at_iteration=solution.get("declared_at_iteration"),
    )


def _add_final_scores_evidence(
    state: InvestigationState, artifact: dict[str, Any]
) -> None:
    branch_scores = [
        {
            "name": snap.get("name"),
            "char_accuracy": snap.get("char_accuracy"),
            "word_accuracy": snap.get("word_accuracy"),
            "mapped_count": snap.get("mapped_count"),
        }
        for snap in (artifact.get("branches") or [])
        if isinstance(snap, dict) and snap.get("name")
    ]
    state.add_evidence(
        "v2_final_scores",
        turn=0,
        summary=(
            f"v2 final status={artifact.get('status', '-')} "
            f"char_acc={artifact.get('char_accuracy', '-')} "
            f"word_acc={artifact.get('word_accuracy', '-')}"
        ),
        status=artifact.get("status"),
        char_accuracy=artifact.get("char_accuracy"),
        word_accuracy=artifact.get("word_accuracy"),
        branches=branch_scores,
    )


# ---------------------------------------------------------------------------
# Cipher reconstruction (RELOCATED from agent/resume.py — F2a; do NOT import
# agent/resume here, which is deletion-listed at v2 retirement).
# ---------------------------------------------------------------------------


def _cipher_text_from_artifact(artifact: dict[str, Any]) -> CipherText:
    """Reconstruct the original cipher text from the artifact's first prompt.

    Raises ``V2ArtifactAdapterError`` (a STRUCTURED error, F2c) when the notation
    block cannot be extracted (old/compressed artifacts).
    """
    raw = _extract_cipher_block(artifact)
    if not raw:
        raise V2ArtifactAdapterError(
            "Could not find the manuscript notation block in the v2 artifact; "
            "it is too old/compressed to adapt. Resume through the benchmark "
            "path, or use a newer artifact."
        )
    if "|" in raw or _looks_like_space_delimited_tokens(raw):
        return parse_canonical_transcription(raw)

    ignore = {" ", "\t", "\n", "\r"}
    alphabet = Alphabet.from_text(raw, ignore_chars=ignore)
    clean = " ".join(raw.split())
    separator = " " if " " in clean else None
    return CipherText(
        raw=clean, alphabet=alphabet, source="artifact_resume", separator=separator
    )


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
