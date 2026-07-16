"""Unified v3 candidate representation derived from a workspace branch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.loop_shared import (
    DECODED_TEXT_RENDERER_ID,
    _candidate_content_hash,
    _decoded_text_for_panel,
)


def null_mask_symbols(branch: Any) -> tuple[str, ...]:
    """Return the normalized null mask carried by a branch, if any."""
    metadata = branch.metadata
    block = metadata.get("null_mask_finalist")
    if not isinstance(block, dict):
        block = metadata.get("null_mask_selected")
    if not isinstance(block, dict):
        return ()
    return tuple(str(symbol) for symbol in (block.get("mask") or []))


def rendered_token_indices(workspace: Any, branch_name: str) -> tuple[int, ...]:
    """Map rendered candidate characters back to effective token positions."""
    branch = workspace.get_branch(branch_name)
    mask = set(null_mask_symbols(branch))
    if not mask:
        return tuple(range(len(workspace.effective_tokens(branch_name))))
    cipher_alpha = workspace.cipher_text.alphabet
    return tuple(
        index
        for index, token in enumerate(workspace.effective_tokens(branch_name))
        if cipher_alpha.symbol_for(token) not in mask
    )


@dataclass(frozen=True)
class BranchCandidatePacket:
    """Ground-truth-free candidate state with explicit repair capabilities."""

    branch: str
    text: str
    content_hash: str
    renderer_id: str
    capabilities: tuple[str, ...]
    provenance: dict[str, Any]
    key: dict[int, int]
    null_mask: tuple[str, ...]
    word_spans: tuple[tuple[int, int], ...] | None
    token_order: tuple[int, ...] | None
    transform_pipeline: dict[str, Any] | None
    rendered_token_indices: tuple[int, ...]

    @property
    def primary_capability(self) -> str:
        for name in (
            "editable_null_mask",
            "editable_key",
            "editable_transform",
            "editable_boundaries",
        ):
            if name in self.capabilities:
                return name
        return "text_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "text": self.text,
            "content_hash": self.content_hash,
            "renderer_id": self.renderer_id,
            "capability": self.primary_capability,
            "capabilities": list(self.capabilities),
            "provenance": dict(self.provenance),
            "key": {str(k): v for k, v in self.key.items()},
            "null_mask": list(self.null_mask),
            "word_spans": (
                [list(span) for span in self.word_spans]
                if self.word_spans is not None
                else None
            ),
            "token_order": list(self.token_order) if self.token_order is not None else None,
            "transform_pipeline": self.transform_pipeline,
            "rendered_token_indices": list(self.rendered_token_indices),
        }


def candidate_packet_for_branch(workspace: Any, branch_name: str) -> BranchCandidatePacket:
    branch = workspace.get_branch(branch_name)
    text = _decoded_text_for_panel(workspace, branch_name)
    mask = null_mask_symbols(branch)
    capabilities: list[str] = []
    if branch.key:
        capabilities.append("editable_key")
    if branch.key and mask:
        capabilities.append("editable_null_mask")
    if branch.word_spans is not None or branch.key:
        capabilities.append("editable_boundaries")
    if branch.token_order is not None or branch.transform_pipeline is not None:
        capabilities.append("editable_transform")
    if not capabilities:
        capabilities.append("text_only")

    metadata = branch.metadata
    provenance = {
        "decoded_text_source": metadata.get("decoded_text_source"),
        "cipher_mode": metadata.get("cipher_mode"),
        "key_type": metadata.get("key_type"),
        "null_mask_finalist": metadata.get("null_mask_finalist"),
        "null_mask_selected": metadata.get("null_mask_selected"),
        "parent": branch.parent,
        "tags": list(branch.tags),
    }
    return BranchCandidatePacket(
        branch=branch_name,
        text=text,
        content_hash=_candidate_content_hash(text),
        renderer_id=DECODED_TEXT_RENDERER_ID,
        capabilities=tuple(dict.fromkeys(capabilities)),
        provenance=provenance,
        key=dict(branch.key),
        null_mask=mask,
        word_spans=(
            tuple(tuple(span) for span in branch.word_spans)
            if branch.word_spans is not None
            else None
        ),
        token_order=(tuple(branch.token_order) if branch.token_order is not None else None),
        transform_pipeline=branch.transform_pipeline,
        rendered_token_indices=rendered_token_indices(workspace, branch_name),
    )
