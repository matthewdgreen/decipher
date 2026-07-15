"""The Reading artifact (v3 M3 spec Part 1; design C4 + A8).

A *Reading* is a proposed plain-language reading of a branch's decode. It is
produced by a ``reading``-kind episode (compiled lead-side from the worker's
result dict) or written directly by the lead, stored in
``InvestigationState.readings`` (reading_id -> ``Reading.to_dict()``), and later
consumed by ``hypothesis_apply_reading`` to compile key edits + boundary changes
onto a fork.

Readings never carry ground truth: fragment text is the model's proposed
plaintext, and the firewall covers this surface (Part 8).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def new_reading_id() -> str:
    """A 12-hex-char reading id (same convention as episode ids)."""
    return uuid.uuid4().hex[:12]


def coerce_confidence(value: Any, default: float = 0.5) -> float:
    """Coerce a confidence to a float in [0, 1] (A5b).

    Strings arriving from a scripted/real worker are parsed with ``float()`` and
    clamped to [0, 1]; unparseable strings become ``default`` (0.5).
    """
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:  # NaN
        return default
    return max(0.0, min(1.0, parsed))


def _coerce_span(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ReadingFragment:
    """One positioned (or whole-text) fragment of a Reading.

    ``start``/``end`` are token indices into the branch's effective token stream
    (``None`` = the whole text). ``text`` is the human-facing reading and may
    contain ordinary prose punctuation. ``repair_text`` is the optional
    machine-actionable form (plaintext alphabet symbols, spaces, and ``?``
    wildcards). ``label`` is an optional window tag (the M2 reading proto's
    ``window`` string survives here).
    """

    text: str
    repair_text: str | None = None
    start: int | None = None
    end: int | None = None
    confidence: float = 1.0
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "repair_text": self.repair_text,
            "confidence": self.confidence,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReadingFragment":
        return cls(
            text=str(data.get("text") or ""),
            repair_text=(
                str(data["repair_text"])
                if data.get("repair_text") is not None
                else None
            ),
            start=_coerce_span(data.get("start")),
            end=_coerce_span(data.get("end")),
            confidence=coerce_confidence(data.get("confidence")),
            label=(str(data["label"]) if data.get("label") is not None else None),
        )


@dataclass
class Reading:
    """A stored reading of a branch (Part 1)."""

    branch: str
    source: str  # "episode:<episode_id>" or "lead"
    created_turn: int = 0
    reading_id: str = field(default_factory=new_reading_id)
    fragments: list[ReadingFragment] = field(default_factory=list)
    holes: list[str] = field(default_factory=list)
    overall_confidence: float = 0.5

    @property
    def full_text(self) -> str:
        """Fragment texts joined in ``start`` order (whole-text fragments first)."""
        ordered = sorted(
            self.fragments,
            key=lambda f: (f.start if f.start is not None else -1),
        )
        return " ".join(f.text for f in ordered if f.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reading_id": self.reading_id,
            "branch": self.branch,
            "source": self.source,
            "created_turn": self.created_turn,
            "fragments": [f.to_dict() for f in self.fragments],
            "holes": list(self.holes),
            "overall_confidence": self.overall_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reading":
        return cls(
            reading_id=str(data.get("reading_id") or new_reading_id()),
            branch=str(data.get("branch") or ""),
            source=str(data.get("source") or "lead"),
            created_turn=int(data.get("created_turn") or 0),
            fragments=[
                ReadingFragment.from_dict(f)
                for f in (data.get("fragments") or [])
                if isinstance(f, dict)
            ],
            holes=[str(h) for h in (data.get("holes") or [])],
            overall_confidence=coerce_confidence(data.get("overall_confidence")),
        )

    @classmethod
    def from_episode_result(
        cls,
        result: dict[str, Any],
        *,
        branch: str,
        source: str,
        created_turn: int,
        reading_id: str | None = None,
    ) -> "Reading":
        """Compile a ``reading``-kind episode result dict into a Reading (Part 1).

        The result matches ``episodes._READING_SCHEMA``: ``reading_text``,
        ``fragments`` (each with optional ``window``/``start``/``end``, ``text``,
        ``confidence``), ``holes``, ``overall_confidence``. A fragment carries its
        ``window`` string across as ``label``; A8's optional ``start``/``end``
        token indices survive when the worker reported them.
        """
        result = result or {}
        raw_fragments = result.get("fragments") or []
        fragments: list[ReadingFragment] = []
        for item in raw_fragments:
            if not isinstance(item, dict):
                continue
            fragments.append(
                ReadingFragment(
                    text=str(item.get("text") or ""),
                    repair_text=(
                        str(item["repair_text"])
                        if item.get("repair_text") is not None
                        else None
                    ),
                    start=_coerce_span(item.get("start")),
                    end=_coerce_span(item.get("end")),
                    confidence=coerce_confidence(item.get("confidence")),
                    label=(
                        str(item["window"])
                        if item.get("window") is not None
                        else None
                    ),
                )
            )
            if item.get("confidence") is None:
                # M5.1 review fix (softened after Stage-1 forensics): a worker
                # that omits confidence gets the conservative 0.5 — BELOW the
                # MIN_REPAIR_FRAGMENT_CONFIDENCE threshold, so the fragment
                # stays visible in the Reading (and in skipped_fragments when
                # application is attempted) but cannot change the key
                # automatically. Legacy STORED readings load via from_dict,
                # not through this compiler, and are unaffected. Schema-
                # requiring confidence was tried and reverted: it failed the
                # WHOLE episode on one silent fragment.
                fragments[-1].confidence = 0.5
        # When the worker gave no fragments but did give reading_text, keep the
        # whole-text reading as a single fragment so the reading is applicable.
        if not fragments:
            text = str(result.get("reading_text") or "")
            if text:
                fragments.append(
                    ReadingFragment(
                        text=text,
                        confidence=coerce_confidence(
                            result.get("overall_confidence")
                        ),
                    )
                )
        return cls(
            reading_id=reading_id or new_reading_id(),
            branch=branch,
            source=source,
            created_turn=created_turn,
            fragments=fragments,
            holes=[str(h) for h in (result.get("holes") or [])],
            overall_confidence=coerce_confidence(result.get("overall_confidence")),
        )
