"""Candidate packet: the interchange format for finalist candidates.

A :class:`CandidatePacket` is a normalized, artifact-side representation of a
single finalist candidate produced by one of the three finalist generators
(null-mask refinement, transform+homophonic search, pure-transposition screen).
Packets are stored server-side (in finalist sessions and run artifacts) and are
*never* added to model-visible review rows -- keeping them out of the token
budget is a hard requirement of the surrounding tools.

This module has no dependencies on ``agent.tools_v2`` or ``automated.runner``;
the import direction is one-way (those modules import this one) so that the v3
episode architecture can later move packet ownership to ``InvestigationState``.

The three ``packet_from_*`` adapters are pure functions. They never mutate the
input row, they never raise ``KeyError`` on a missing field (missing keys map to
``None`` / empty dict), and any row key they do not explicitly consume is swept
into ``extras`` so no provenance is silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The canonical set of top-level packet keys. Used by ``from_dict`` to decide
# which incoming keys are known fields versus unknown extras.
_FIELD_NAMES: tuple[str, ...] = (
    "packet_version",
    "candidate_id",
    "kind",
    "source",
    "rank",
    "text",
    "preview",
    "solver_scores",
    "validation",
    "language_features",
    "provenance",
    "rating",
    "extras",
)


@dataclass
class CandidatePacket:
    """Normalized finalist candidate. See module docstring for semantics."""

    candidate_id: str = ""
    kind: str = ""
    packet_version: int = 1
    source: dict[str, Any] = field(default_factory=dict)
    rank: int | None = None
    text: str | None = None
    preview: str | None = None
    solver_scores: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] | None = None
    language_features: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    rating: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_version": self.packet_version,
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "source": self.source,
            "rank": self.rank,
            "text": self.text,
            "preview": self.preview,
            "solver_scores": self.solver_scores,
            "validation": self.validation,
            "language_features": self.language_features,
            "provenance": self.provenance,
            "rating": self.rating,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidatePacket":
        data = dict(data)
        extras = dict(data.get("extras") or {})
        kwargs: dict[str, Any] = {}
        for key in _FIELD_NAMES:
            if key in ("extras",):
                continue
            if key in data:
                kwargs[key] = data[key]
        for key, value in data.items():
            if key not in _FIELD_NAMES:
                extras[key] = value
        kwargs["extras"] = extras
        return cls(**kwargs)


def _present(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Return ``{k: row[k]}`` for present keys only (never a KeyError)."""
    return {key: row[key] for key in keys if key in row}


def _sweep_extras(
    row: dict[str, Any],
    consumed: set[str],
    extras: dict[str, Any],
) -> dict[str, Any]:
    """Copy every row key not already consumed into ``extras``.

    ``packet`` is always treated as consumed: rows finalized on the runner side
    may already carry a serialized packet, and re-packeting them would nest a
    packet inside a packet.
    """
    consumed = set(consumed)
    consumed.add("packet")
    for key, value in row.items():
        if key not in consumed and key not in extras:
            extras[key] = value
    return extras


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

_NULL_MASK_SOLVER_SCORE_KEYS: tuple[str, ...] = (
    "anneal_score",
    "selection_score",
    "validation_score_v2",
    "confirmed_validation_score_v2",
    "promoted_validation_score_v2",
    "ensemble_score_v1",
    "ensemble_vote_rate_v1",
)

_NULL_MASK_LANGUAGE_QUALITY_KEYS: tuple[str, ...] = (
    "language_quality_raw_score",
    "language_quality_score",
    "language_quality_rank_score",
    "language_quality_model",
)


def packet_from_null_mask_row(
    row: dict[str, Any],
    *,
    source_branch: str | None = None,
    rank: int | None = None,
) -> CandidatePacket:
    """Adapt a null-mask finalist row (full or compact) into a packet.

    Null-mask rows have no ``finalist_validation`` block, so ``validation`` is
    ``None``; the ``validation_components_v2`` / ``diagnostics`` / ``quality``
    blocks and any ``language_quality_*`` scalars are preserved under
    ``extras``. ``rank`` is the 1-based menu position at creation time; when
    omitted it falls back to the row's own ``rank`` field.
    """
    mask = list(row.get("mask") or [])
    candidate_id = row.get("candidate_id")
    if candidate_id is None:
        candidate_id = "mask:" + ",".join(str(part) for part in mask)

    solver_scores = _present(row, _NULL_MASK_SOLVER_SCORE_KEYS)

    extras: dict[str, Any] = {}
    for key in ("validation_components_v2", "diagnostics", "quality"):
        if key in row:
            extras[key] = row[key]
    language_quality = _present(row, _NULL_MASK_LANGUAGE_QUALITY_KEYS)
    if language_quality:
        extras["language_quality"] = language_quality

    provenance = {
        "mask": mask,
        "mask_size": row.get("mask_size"),
        "filtered_length": row.get("filtered_length"),
        "key": row.get("key"),
    }

    consumed: set[str] = {
        "candidate_id",
        "mask",
        "mask_size",
        "filtered_length",
        "key",
        "solver",
        "decryption",
        "preview",
        "agent_readability_judgment",
        "rank",
        "language_features",
        "validation_components_v2",
        "diagnostics",
        "quality",
    }
    consumed.update(_NULL_MASK_SOLVER_SCORE_KEYS)
    consumed.update(_NULL_MASK_LANGUAGE_QUALITY_KEYS)

    return CandidatePacket(
        candidate_id=str(candidate_id),
        kind="null_mask",
        source={"solver": row.get("solver"), "source_branch": source_branch},
        rank=rank if rank is not None else row.get("rank"),
        text=row.get("decryption"),
        preview=row.get("preview"),
        solver_scores=solver_scores,
        validation=None,
        language_features=row.get("language_features"),
        provenance=provenance,
        rating=row.get("agent_readability_judgment"),
        extras=_sweep_extras(row, consumed, extras),
    )


def packet_from_transform_row(
    row: dict[str, Any],
    *,
    rank: int | None = None,
) -> CandidatePacket:
    """Adapt a transform+homophonic ranked row into a packet.

    Handles both shapes observed in the codebase:

    * the agent-tool row (``tools_v2.py``): a nested ``candidate`` dict plus
      ``candidate_index`` and ``decoded_preview``;
    * the automated-runner row (``runner.py`` ``_rank_transform_candidates``):
      flat ``candidate_id`` / ``family`` / ``params`` plus ``decryption_preview``.

    See the module/implementation notes: the spec's field mapping targets the
    tool shape only; the runner shape is a documented mismatch handled here so
    the runner-side artifact attachment cannot raise. ``rank`` is the 1-based
    menu position at creation time; when omitted it falls back to the row's own
    ``rank`` field.
    """
    candidate = row.get("candidate")
    if not isinstance(candidate, dict):
        candidate = None

    if candidate is not None:
        candidate_id = candidate.get("candidate_id")
    else:
        candidate_id = row.get("candidate_id")
    if candidate_id is None:
        if "candidate_index" in row:
            candidate_id = f"cand_{row.get('candidate_index')}"
        else:
            candidate_id = "cand_unknown"

    family = candidate.get("family") if candidate is not None else row.get("family")
    params = candidate.get("params") if candidate is not None else row.get("params")
    preview = row.get("decoded_preview")
    if preview is None:
        preview = row.get("decryption_preview")

    provenance = {
        "pipeline": row.get("pipeline"),
        "key": row.get("key"),
        "family": family,
        "params": params,
    }

    consumed: set[str] = {
        "candidate",
        "candidate_index",
        "candidate_id",
        "family",
        "params",
        "pipeline",
        "key",
        "status",
        "solver",
        "anneal_score",
        "elapsed_seconds",
        "decoded_preview",
        "decryption_preview",
        "validation",
        "agent_readability_judgment",
        "rank",
    }

    return CandidatePacket(
        candidate_id=str(candidate_id),
        kind="transform",
        source={"solver": row.get("solver"), "status": row.get("status")},
        rank=rank if rank is not None else row.get("rank"),
        text=None,
        preview=preview,
        solver_scores=_present(row, ("anneal_score",)),
        validation=row.get("validation"),
        language_features=row.get("language_features"),
        provenance=provenance,
        rating=row.get("agent_readability_judgment"),
        extras=_sweep_extras(row, consumed, {}),
    )


_PURE_TRANSPOSITION_SOLVER_SCORE_KEYS: tuple[str, ...] = (
    "score",
    "selection_score",
    "validated_selection_score",
    "rust_rank",
)

_PURE_TRANSPOSITION_PROVENANCE_KEYS: tuple[str, ...] = (
    "candidate_id",
    "family",
    "params",
    "pipeline",
    "inverse_mode",
    "grid",
    "provenance",
)


def packet_from_pure_transposition_row(
    row: dict[str, Any],
    *,
    rank: int | None = None,
) -> CandidatePacket:
    """Adapt a pure-transposition finalist row into a packet.

    The row is ``TransformCandidate.to_dict()`` merged with the ranking and
    plaintext fields, plus ``validation`` / ``validated_selection_score`` from
    the finalist-menu evaluation. ``rank`` is the 1-based menu position at
    creation time; when omitted it falls back to the row's own ``rank`` field
    (which pure-transposition rows carry natively).
    """
    provenance = {key: row.get(key) for key in _PURE_TRANSPOSITION_PROVENANCE_KEYS}

    consumed: set[str] = {
        "candidate_id",
        "rank",
        "rust_rank",
        "score",
        "selection_score",
        "validated_selection_score",
        "plaintext",
        "preview",
        "validation",
        "agent_readability_judgment",
    }
    consumed.update(_PURE_TRANSPOSITION_PROVENANCE_KEYS)

    return CandidatePacket(
        candidate_id=str(row.get("candidate_id") or ""),
        kind="pure_transposition",
        source={"family": row.get("family")},
        rank=rank if rank is not None else row.get("rank"),
        text=row.get("plaintext"),
        preview=row.get("preview"),
        solver_scores=_present(row, _PURE_TRANSPOSITION_SOLVER_SCORE_KEYS),
        validation=row.get("validation"),
        language_features=row.get("language_features"),
        provenance=provenance,
        rating=row.get("agent_readability_judgment"),
        extras=_sweep_extras(row, consumed, {}),
    )
