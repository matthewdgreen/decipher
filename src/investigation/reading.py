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

import hashlib
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

    ``span_id`` is a host-owned opaque reference to a candidate-reading span.
    ``start``/``end`` are token indices into the branch's effective token stream
    (``None`` = the whole text). ``text`` is the human-facing reading and may
    contain ordinary prose punctuation. ``repair_text`` is the optional
    machine-actionable form (plaintext alphabet symbols, spaces, and ``?``
    wildcards). ``label`` is an optional window tag (the M2 reading proto's
    ``window`` string survives here).
    """

    text: str
    repair_text: str | None = None
    span_id: str | None = None
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
            "span_id": self.span_id,
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
            span_id=(str(data["span_id"]) if data.get("span_id") else None),
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
        candidate_packet: dict[str, Any] | None = None,
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
        packet_spans = {
            str(item.get("span_id")): item
            for item in (candidate_packet or {}).get("spans") or []
            if isinstance(item, dict) and item.get("span_id")
        }
        fragments: list[ReadingFragment] = []
        for item in raw_fragments:
            if not isinstance(item, dict):
                continue
            span_id = str(item.get("span_id") or "") or None
            bound_span = packet_spans.get(span_id or "")
            # Episode workers refer to host-owned span ids. Their own numeric
            # offsets are not trusted when a packet is present: real M6 workers
            # sometimes returned word/window positions while the repair code
            # interpreted them as token positions.
            if candidate_packet is not None:
                start = _coerce_span((bound_span or {}).get("token_start"))
                end = _coerce_span((bound_span or {}).get("token_end"))
            else:
                start = _coerce_span(item.get("start"))
                end = _coerce_span(item.get("end"))
            fragments.append(
                ReadingFragment(
                    text=str(item.get("text") or ""),
                    repair_text=(
                        str(item["repair_text"])
                        if item.get("repair_text") is not None
                        else None
                    ),
                    span_id=span_id if bound_span is not None else None,
                    start=start,
                    end=end,
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


@dataclass(frozen=True)
class CandidateReadingSpan:
    """One host-owned window in a candidate-reading packet."""

    span_id: str
    text: str
    token_start: int | None
    token_end: int | None
    word_start: int | None = None
    word_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "text": self.text,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "word_start": self.word_start,
            "word_end": self.word_end,
        }


@dataclass(frozen=True)
class CandidateReadingPacket:
    """Exact candidate text plus stable spans prepared by the host.

    The worker never invents repair coordinates. Key-backed candidates expose
    token-bound spans; metadata-only candidates remain readable but are marked
    ``text_only`` and therefore cannot accidentally drive key edits.
    """

    branch: str
    content_hash: str
    renderer_id: str
    capability: str
    candidate_text: str
    spans: tuple[CandidateReadingSpan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "content_hash": self.content_hash,
            "renderer_id": self.renderer_id,
            "capability": self.capability,
            "candidate_text": self.candidate_text,
            "spans": [span.to_dict() for span in self.spans],
        }


def build_candidate_reading_packet(
    workspace: Any,
    branch_name: str,
    *,
    window_tokens: int = 120,
) -> CandidateReadingPacket:
    """Build a deterministic packet for a reading worker.

    The general renderer is imported lazily so packet identity exactly matches
    attestation and benchmark rendering without adding an import-time cycle.
    """
    from agent.loop_shared import DECODED_TEXT_RENDERER_ID, _decoded_text_for_panel

    branch = workspace.get_branch(branch_name)
    metadata_text = branch.metadata.get("decoded_text")
    if isinstance(metadata_text, str) and metadata_text.strip():
        candidate = _decoded_text_for_panel(workspace, branch_name)
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        span = CandidateReadingSpan(
            span_id=f"span_text_{digest[:10]}",
            text=candidate,
            token_start=None,
            token_end=None,
        )
        return CandidateReadingPacket(
            branch=branch_name,
            content_hash=digest,
            renderer_id=DECODED_TEXT_RENDERER_ID,
            capability="text_only",
            candidate_text=candidate,
            spans=(span,),
        )

    tokens = list(workspace.effective_tokens(branch_name))
    word_spans = list(workspace.effective_word_spans(branch_name))
    word_ends = {end for _start, end in word_spans}
    pt_alpha = workspace.plaintext_alphabet

    chars: list[str] = []
    token_char_offsets: list[int] = []
    for index, token in enumerate(tokens):
        token_char_offsets.append(len(chars))
        if token in branch.key:
            chars.append(pt_alpha.symbol_for(branch.key[token]))
        else:
            chars.append("?")
        if index + 1 in word_ends and index + 1 < len(tokens):
            chars.append(" ")
    candidate = _decoded_text_for_panel(workspace, branch_name)
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()

    spans: list[CandidateReadingSpan] = []
    size = max(1, int(window_tokens))
    for token_start in range(0, len(tokens), size):
        token_end = min(len(tokens), token_start + size)
        char_start = token_char_offsets[token_start]
        char_end = (
            token_char_offsets[token_end]
            if token_end < len(tokens)
            else len(candidate)
        )
        overlapping = [
            i for i, (start, end) in enumerate(word_spans)
            if start < token_end and end > token_start
        ]
        spans.append(CandidateReadingSpan(
            span_id=f"span_{token_start}_{token_end}_{digest[:10]}",
            text=candidate[char_start:char_end].strip(),
            token_start=token_start,
            token_end=token_end,
            word_start=min(overlapping) if overlapping else None,
            word_end=(max(overlapping) + 1) if overlapping else None,
        ))
    if not spans:
        spans.append(CandidateReadingSpan(
            span_id=f"span_0_0_{digest[:10]}",
            text="",
            token_start=0,
            token_end=0,
        ))
    return CandidateReadingPacket(
        branch=branch_name,
        content_hash=digest,
        renderer_id=DECODED_TEXT_RENDERER_ID,
        capability="editable_key",
        candidate_text=candidate,
        spans=tuple(spans),
    )
