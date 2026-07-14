"""HypothesisBoard: the single writer for cipher-mode hypothesis cards (A10).

Design (agent_v3_design A10 + M2 spec Part 5 / F3 / F4):

- All hypothesis *card-field* mutations go through the board. The board mirrors
  each card into ``branch.metadata`` under exactly the keys the v2 handlers wrote,
  so every metadata *read* site (``_mode_scoped_suggestions``,
  ``_context_cipher_family_tool_block``, ``workspace_hypothesis_cards``, guards)
  keeps working with no sweep.
- The gating / validation / tag / context-prior writes stay in the tool handlers
  (F3): the board routes ONLY the card fields listed in ``CARD_MIRROR_KEYS``.
- ``tried``/``pending`` are call_log-derived in v2 and have no M2 writer, so they
  are dropped from the M2 card (F3).

When the executor is given no board (v2 path) it builds a private board; the
mirror keeps the handler outputs and metadata byte-identical. The v3 lead injects
a state-owned board so it survives resume and renders into the lead context.
"""
from __future__ import annotations

from typing import Any

# The card fields that are mirrored verbatim into ``branch.metadata`` — exactly
# the keys the v2 hypothesis handlers wrote for these fields.
CARD_MIRROR_KEYS = (
    "cipher_mode",
    "mode_status",
    "mode_confidence",
    "mode_evidence",
    "mode_counter_evidence",
    "evidence_source",
    "next_recommended_action",
    "rejection_reason",
)

# Full card field order (id + branch + the mirrored fields).
_CARD_FIELDS = ("id", "branch", *CARD_MIRROR_KEYS)


def _empty_card(card_id: int, branch: str) -> dict[str, Any]:
    card = {key: None for key in _CARD_FIELDS}
    card["id"] = card_id
    card["branch"] = branch
    return card


class HypothesisBoard:
    """Single-writer store of cipher-mode hypothesis cards."""

    def __init__(self) -> None:
        self._cards: dict[str, dict[str, Any]] = {}
        self._next_id: int = 1

    # --- id minting ------------------------------------------------------
    def _mint_id(self) -> int:
        card_id = self._next_id
        self._next_id += 1
        return card_id

    # --- internal write --------------------------------------------------
    def _upsert(
        self, workspace: Any, branch: str, fields: dict[str, Any]
    ) -> dict[str, Any]:
        """Set ``fields`` on the branch's card and mirror them into metadata.

        Only non-None card-mirror fields are written to metadata, matching the
        conditional writes the v2 handlers performed (so v2 output is
        byte-identical). The branch must already exist in the workspace.
        """
        card = self._cards.get(branch)
        if card is None:
            card = _empty_card(self._mint_id(), branch)
            self._cards[branch] = card
        branch_obj = workspace.get_branch(branch)
        for key, value in fields.items():
            if value is None:
                continue
            card[key] = value
            if key in CARD_MIRROR_KEYS:
                branch_obj.metadata[key] = value
        return card

    # --- public mutators (called by the hypothesis-tool adapters) --------
    def create(
        self,
        workspace: Any,
        branch: str,
        *,
        cipher_mode: str,
        mode_confidence: str,
        mode_status: str,
        mode_evidence: str,
        evidence_source: str,
    ) -> dict[str, Any]:
        return self._upsert(workspace, branch, {
            "cipher_mode": cipher_mode,
            "mode_confidence": mode_confidence,
            "mode_status": mode_status,
            "mode_evidence": mode_evidence,
            "evidence_source": evidence_source,
        })

    def update(self, workspace: Any, branch: str, **fields: Any) -> dict[str, Any]:
        """Update arbitrary card fields (only the provided, non-None ones)."""
        return self._upsert(workspace, branch, fields)

    def reject(
        self,
        workspace: Any,
        branch: str,
        *,
        mode_status: str,
        rejection_reason: str,
    ) -> dict[str, Any]:
        return self._upsert(workspace, branch, {
            "mode_status": mode_status,
            "rejection_reason": rejection_reason,
        })

    # --- reads -----------------------------------------------------------
    def get(self, branch: str) -> dict[str, Any] | None:
        card = self._cards.get(branch)
        return dict(card) if card is not None else None

    def cards(self) -> list[dict[str, Any]]:
        """Return all cards (id-ordered), each a fresh dict."""
        return [dict(card) for card in sorted(
            self._cards.values(), key=lambda c: c.get("id") or 0
        )]

    # --- adoption of workspace hypothesis metadata (F4/F11) --------------
    @staticmethod
    def _is_hypothesis_branch(branch_obj: Any) -> bool:
        metadata = branch_obj.metadata
        return bool(metadata.get("cipher_mode")) or ("hypothesis" in branch_obj.tags)

    def _card_from_metadata(self, card_id: int, name: str, branch_obj: Any) -> dict[str, Any]:
        metadata = branch_obj.metadata
        card = _empty_card(card_id, name)
        card.update({
            "cipher_mode": metadata.get("cipher_mode") or "unknown",
            "mode_status": metadata.get("mode_status", "active"),
            "mode_confidence": metadata.get("mode_confidence"),
            "mode_evidence": metadata.get("mode_evidence")
            or metadata.get("hypothesis_notes"),
            "mode_counter_evidence": metadata.get("mode_counter_evidence"),
            "evidence_source": metadata.get("evidence_source"),
            "next_recommended_action": metadata.get("next_recommended_action"),
            "rejection_reason": metadata.get("rejection_reason"),
        })
        return card

    def sync_from_workspace(self, workspace: Any) -> None:
        """Adopt hypothesis branches that have metadata but no card yet (F4).

        Run once per lead turn so a hypothesis branch forked INSIDE an episode
        gains a card after ``episode_install_branch``. Existing cards are left
        untouched (the board remains the writer of record for anything it made).
        """
        for name in workspace.branch_names():
            if name in self._cards:
                continue
            branch_obj = workspace.get_branch(name)
            if not self._is_hypothesis_branch(branch_obj):
                continue
            self._cards[name] = self._card_from_metadata(
                self._mint_id(), name, branch_obj
            )

    @classmethod
    def from_workspace(cls, workspace: Any) -> "HypothesisBoard":
        """Build a board by projecting existing branch hypothesis metadata.

        Replaces M1's read-only ``hypothesis_board_from_workspace`` projection
        (kept only as this import path). Used by ``InvestigationState`` when an
        artifact predates the board field (F11).
        """
        board = cls()
        board.sync_from_workspace(workspace)
        return board

    # --- serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "cards": self.cards(),
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HypothesisBoard":
        board = cls()
        if not data:
            return board
        for raw in data.get("cards") or []:
            card = _empty_card(int(raw.get("id") or board._next_id), str(raw.get("branch") or ""))
            for key in CARD_MIRROR_KEYS:
                if key in raw:
                    card[key] = raw[key]
            branch = card["branch"]
            if branch:
                board._cards[branch] = card
        next_id = data.get("next_id")
        if next_id is not None:
            board._next_id = int(next_id)
        else:
            board._next_id = max(
                (int(c.get("id") or 0) for c in board._cards.values()), default=0
            ) + 1
        return board
