"""Finalist session store: one home for the three wide-search session kinds.

Historically the ``WorkspaceToolExecutor`` carried three parallel, copy-pasted
in-memory session subsystems (transform search, pure transposition, null mask),
each with its own dict and monotonic counter. :class:`FinalistSessionStore`
unifies them behind a single per-kind keyed API while preserving the exact
externally-visible session-id scheme (``f"{kind}_{n}"`` with a per-kind counter
starting at 1).

The store has *no* dependency on ``agent.tools_v2`` or ``agent.loop_v2``: the v3
episode architecture will later move ownership to ``InvestigationState``, so the
store must be self-contained. It imports only :mod:`analysis.candidate_packet`.
"""

from __future__ import annotations

from typing import Any, Iterable

from analysis.candidate_packet import CandidatePacket


class FinalistSessionStore:
    """In-memory finalist sessions keyed by ``kind`` then session id.

    Payload dicts keep their current per-kind keys verbatim; the store adds a
    single ``"packets"`` key holding ``[packet.to_dict() for packet in packets]``.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self._counters: dict[str, int] = {}

    def new_session(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        packets: list[CandidatePacket],
    ) -> str:
        """Register ``payload`` under a freshly minted ``f"{kind}_{n}"`` id.

        ``payload`` is stored by reference (callers rely on later in-place
        mutation, e.g. rating refresh). The returned id embeds a per-kind
        counter that starts at 1 and increments on every call for that kind.
        """
        count = self._counters.get(kind, 0) + 1
        self._counters[kind] = count
        session_id = f"{kind}_{count}"
        payload["packets"] = [packet.to_dict() for packet in packets]
        self._sessions.setdefault(kind, {})[session_id] = payload
        return session_id

    def get(self, kind: str, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(kind, {}).get(str(session_id))

    def find_id(self, kind: str, payload: dict[str, Any]) -> str | None:
        """Return the session id whose stored payload *is* ``payload`` (identity)."""
        for session_id, stored in self._sessions.get(kind, {}).items():
            if stored is payload:
                return session_id
        return None

    def sessions(self, kind: str) -> Iterable[tuple[str, dict[str, Any]]]:
        """(session_id, payload) pairs for ``kind`` in insertion order."""
        return list(self._sessions.get(kind, {}).items())

    def last_sessions(self, kind: str) -> Iterable[tuple[str, dict[str, Any]]]:
        """(session_id, payload) pairs for ``kind`` in reverse-insertion order."""
        return list(reversed(list(self._sessions.get(kind, {}).items())))
