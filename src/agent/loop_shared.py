"""Shared, provider-neutral helpers used by both the v2 loop and the v3
investigation modules.

These functions were originally defined in ``agent.loop_v2``; the v3 packages
imported them from there, coupling the new investigation loop to the legacy v2
loop. They are hoisted here verbatim (behavior-preserving) so that:

- ``agent.loop_v2`` imports them from this module (its internal callers are
  unchanged), and
- ``investigation.*`` imports them from this module and no longer imports
  ``agent.loop_v2`` at all.

Only self-contained helpers with no dependency on v2 loop internals live here.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from analysis import ngram
from analysis import signals as sig
from analysis.segment import segment_text
from workspace import Workspace


# The named renderer used for the M5 verify-attestation content hash (A6/F1).
# ``_decoded_text_for_panel`` is the string that fills ``BranchSnapshot.decryption``
# (the exact text the benchmark scores), so it is the string an attestation
# certifies. Both attest-time (verify dispatch) and declare-time
# (AttestationPolicy) call it and hash the result with ``_candidate_content_hash``.
DECODED_TEXT_RENDERER_ID = "decoded_text_v1"


def _candidate_content_hash(text: str) -> str:
    """sha256 (hex) of the candidate string encoded as utf-8 (A6/F11)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Decoded-text helpers (metadata-aware)
# ------------------------------------------------------------------
def _metadata_decoded_text(workspace: Workspace, branch_name: str) -> str | None:
    decoded = workspace.get_branch(branch_name).metadata.get("decoded_text")
    if isinstance(decoded, str) and decoded.strip():
        return decoded.strip()
    return None


def _metadata_decoded_words(workspace: Workspace, branch_name: str) -> list[str] | None:
    metadata_text = _metadata_decoded_text(workspace, branch_name)
    if metadata_text is None:
        return None
    if any(ch.isspace() for ch in metadata_text):
        return metadata_text.split()
    branch = workspace.get_branch(branch_name)
    if branch.word_spans is None:
        return [metadata_text]
    spans = workspace.effective_word_spans(branch_name)
    return [metadata_text[start:end] for start, end in spans]


def _decoded_text_for_panel(workspace: Workspace, branch_name: str) -> str:
    metadata_words = _metadata_decoded_words(workspace, branch_name)
    if metadata_words is not None:
        return " ".join(metadata_words)
    return workspace.apply_key(branch_name)


# ------------------------------------------------------------------
# Result parsing / summaries
# ------------------------------------------------------------------
def _parse_json_result(result: str) -> Any:
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return None


def _tool_result_summary(result: str) -> dict[str, Any]:
    parsed = _parse_json_result(result)
    if not isinstance(parsed, dict):
        return {}
    keys = (
        "status",
        "error",
        "branch",
        "mapping",
        "from",
        "to",
        "mappings",
        "mappings_set",
        "score_delta",
        "old_word_count",
        "new_word_count",
        "old_window_word_count",
        "new_window_word_count",
        "old_total_word_count",
        "new_total_word_count",
        "current_char_count",
        "proposed_char_count",
        "same_character_count",
        "unresolved_count",
    )
    summary = {key: parsed[key] for key in keys if key in parsed}
    changed = parsed.get("changed_words")
    if isinstance(changed, list):
        summary["changed_words"] = changed[:4]
    risks = parsed.get("orthography_risks")
    if isinstance(risks, list) and risks:
        summary["orthography_risks"] = risks[:2]
    agenda_item = parsed.get("agenda_item")
    if isinstance(agenda_item, dict):
        summary["agenda_item"] = {
            key: agenda_item[key]
            for key in ("id", "branch", "from", "to", "status")
            if key in agenda_item
        }
    return summary


# ------------------------------------------------------------------
# Branch scoring / selection (ground-truth free)
# ------------------------------------------------------------------
def _score_branch_for_panel(
    workspace: Workspace,
    branch_name: str,
    language: str,
    word_set: set[str],
    freq_rank: dict[str, int] | None = None,
) -> tuple[float | None, float | None]:
    """Return (dict_rate, quad_loglik_per_gram) for panel display.

    dict_rate uses the DP segmenter for no-boundary text, otherwise whitespace
    split. quad_loglik_per_gram is the normalised quadgram log-likelihood.
    Returns (None, None) if the branch has no mappings (nothing to score).
    """
    branch = workspace.get_branch(branch_name)
    decrypted = _decoded_text_for_panel(workspace, branch_name)
    if not branch.key and _metadata_decoded_text(workspace, branch_name) is None:
        return None, None
    normalized = sig.normalize_for_scoring(decrypted)
    if not normalized.strip():
        return None, None

    # dict_rate
    if not any(c.isspace() for c in normalized.strip()):
        seg = segment_text(normalized, word_set, freq_rank)
        dict_rate = seg.dict_rate
    else:
        words = [w for w in normalized.split() if any(c.isalpha() for c in w)]
        if words:
            hits = sum(1 for w in words if w in word_set)
            dict_rate = hits / len(words)
        else:
            dict_rate = 0.0

    # quadgram
    quad_lp = ngram.NGRAM_CACHE.get(language, 4)
    quad = ngram.normalized_ngram_score(normalized, quad_lp, n=4)
    if quad == float("-inf"):
        quad = None

    return dict_rate, quad


def _best_branch_for_auto_declare(
    workspace: Workspace,
    language: str,
    word_set: set[str],
    freq_rank: dict[str, int] | None,
) -> tuple[str, dict[str, float | None]]:
    """Pick the best available branch without ground truth."""
    best_name = workspace.branch_names()[0]
    best_scores: dict[str, float | None] = {"dict_rate": None, "quad": None}
    best_key: tuple[float, float, int] = (float("-inf"), float("-inf"), -1)

    for name in workspace.branch_names():
        dr, quad = _score_branch_for_panel(
            workspace, name, language, word_set, freq_rank
        )
        branch = workspace.get_branch(name)
        rank_key = (
            dr if dr is not None else float("-inf"),
            quad if quad is not None else float("-inf"),
            len(branch.key),
        )
        if rank_key > best_key:
            best_name = name
            best_key = rank_key
            best_scores = {
                "dict_rate": round(dr, 4) if dr is not None else None,
                "quad": round(quad, 4) if quad is not None else None,
            }
    return best_name, best_scores


# ------------------------------------------------------------------
# Artifact snapshot helpers
# ------------------------------------------------------------------
def _branch_snapshot_for(workspace: Workspace, name: str) -> Any:
    """Build a BranchSnapshot dataclass for a single branch."""
    from artifact.schema import BranchSnapshot
    branch = workspace.get_branch(name)
    return BranchSnapshot(
        name=name,
        parent=branch.parent,
        created_iteration=branch.created_iteration,
        key=dict(branch.key),
        mapped_count=len(branch.key),
        decryption=_decoded_text_for_panel(workspace, name),
        signals={},  # panel not computed here; caller can add post-hoc
        tags=list(branch.tags),
        metadata=dict(branch.metadata),
        word_spans=list(branch.word_spans) if branch.word_spans is not None else None,
        token_order=list(branch.token_order) if branch.token_order is not None else None,
        transform_pipeline=(
            dict(branch.transform_pipeline)
            if branch.transform_pipeline is not None
            else None
        ),
    )


def _hypothesis_cards_for_artifact(workspace: Workspace) -> list[dict[str, Any]]:
    """Persist the final cipher-mode hypothesis trail in a compact form."""
    cards: list[dict[str, Any]] = []
    for name in workspace.branch_names():
        branch = workspace.get_branch(name)
        metadata = branch.metadata
        mode = metadata.get("cipher_mode")
        if not mode and "hypothesis" not in branch.tags:
            continue
        cards.append({
            "branch": name,
            "parent": branch.parent,
            "created_iteration": branch.created_iteration,
            "cipher_mode": mode or "unknown",
            "mode_status": metadata.get("mode_status", "active"),
            "mode_confidence": metadata.get("mode_confidence"),
            "mode_evidence": metadata.get("mode_evidence") or metadata.get("hypothesis_notes"),
            "mode_counter_evidence": metadata.get("mode_counter_evidence"),
            "next_recommended_action": metadata.get("next_recommended_action"),
            "rejection_reason": metadata.get("rejection_reason"),
            "tags": list(branch.tags),
        })
    return cards


def _install_automated_preflight_branch(
    workspace: Workspace,
    automated_preflight: dict[str, Any],
) -> None:
    key = automated_preflight.get("key") or {}
    decryption = str(automated_preflight.get("decryption") or "").strip()
    has_key = isinstance(key, dict) and bool(key)
    has_decryption = bool(decryption)
    if not has_key and not has_decryption:
        return
    try:
        workspace.fork("automated_preflight", from_branch="main")
        if has_key:
            parsed_key = {int(ct_id): int(pt_id) for ct_id, pt_id in key.items()}
            workspace.set_full_key("automated_preflight", parsed_key)
        else:
            # Pure transposition: no substitution key, but we may have a pipeline.
            # Extract the pipeline from the most recent step's selected candidate.
            pipeline = None
            for step in reversed(automated_preflight.get("steps") or []):
                selected = step.get("selected") or {}
                if selected.get("pipeline"):
                    pipeline = selected["pipeline"]
                    break
            if pipeline is not None:
                workspace.apply_transform_pipeline("automated_preflight", pipeline)
            # Store the decryption string in branch metadata so that
            # _decoded_text_for_panel and BranchSnapshot.decryption return the
            # correct plaintext even though the substitution key is empty.
            if has_decryption:
                branch = workspace.get_branch("automated_preflight")
                branch.metadata["decoded_text"] = decryption
        workspace.tag("automated_preflight", "automated_preflight")
        workspace.tag("automated_preflight", "no_llm")
    except Exception:  # noqa: BLE001
        # A malformed native preflight should never prevent the LLM run.
        return
