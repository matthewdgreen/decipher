"""InvestigationState: the serializable source of truth for a v3 run.

Design: state is the source of truth; context is a view (agent_v3_design C1,
amendments A3/A7, and the M1 spec Part 1 + F1/F5). Every turn reconstructs the
lead context from this object, so loading a serialized state and continuing IS
the resume path — there is no separate resume machinery in v3.

M1 scope: the cipher, the live Workspace (full branch snapshots), an
append-only evidence log, a per-entry budget ledger, the last-N native-format
exchanges (F1), and the relocated repair agenda (F5). ``episode_ledger`` and
``experiment_queue`` are present as empty schema only (M2/M4).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from agent.finalist_sessions import FinalistSessionStore
from investigation.board import HypothesisBoard
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


# How many whole exchanges (assistant turn + its tool_result turn) to keep in
# ``recent_exchanges``. The context builder renders these verbatim for local
# coherence and reasoning-passback integrity (F1); the budget rule is to drop
# the OLDEST WHOLE exchange, never split one.
DEFAULT_RECENT_EXCHANGES = 2


@dataclass
class BudgetEntry:
    """One metered model call (A7).

    Cost is derived per entry via ``estimate_provider_cost`` and never
    recomputed from run totals, so mixed-model runs account correctly.
    """

    category: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def cost(self) -> float:
        from agent.model_provider import estimate_provider_cost

        return estimate_provider_cost(
            self.provider,
            self.model,
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cost_usd": self.cost(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetEntry":
        return cls(
            category=str(data.get("category") or "lead"),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            cache_read_tokens=int(data.get("cache_read_tokens") or 0),
        )


@dataclass
class EvidenceEntry:
    """One append-only, typed observation with provenance.

    M1 writes ``diagnostic_preflight`` and ``turn_summary`` entries.
    """

    kind: str
    turn: int
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "turn": self.turn,
            "summary": self.summary,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceEntry":
        return cls(
            kind=str(data.get("kind") or ""),
            turn=int(data.get("turn") or 0),
            summary=str(data.get("summary") or ""),
            data=dict(data.get("data") or {}),
        )


# M5.3 Slice 6: the pre-Slice-6 positive-attestation coherence threshold.
# FROZEN migration constant — used ONLY to derive `reader_accepts_as_solution`
# for legacy serialized records (master spec 472-475). Never tune; live gating
# reads `reader_accepts_as_solution` alone.
LEGACY_DECLARE_COHERENCE = 7

DAMAGE_SCOPES = ("local", "distributed", "basin_wide")
REPAIRABILITIES = ("local_repair", "broaden", "none")


def clamp_unit_interval(value: Any) -> float:
    """Coerce a verifier 0..1 field: unparseable/NaN -> 0.0; clamp to [0, 1]."""
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return 0.0
    if coerced != coerced:  # NaN
        return 0.0
    return min(1.0, max(0.0, coerced))


def normalize_damage_scope(value: Any) -> str:
    """Out-of-enum -> conservative 'basin_wide'."""
    return value if value in DAMAGE_SCOPES else "basin_wide"


def normalize_repairability(value: Any) -> str:
    """Out-of-enum -> conservative 'none'."""
    return value if value in REPAIRABILITIES else "none"


@dataclass
class AttestationRecord:
    """An independent-reader verdict on a branch's decode (M5 spec Part 2, A6).

    Produced by a ``verify`` episode and written to state by the lead DISPATCHER
    (workers never write state, A1). ``content_hash`` is
    ``_candidate_content_hash`` over ``renderer_id`` (``decoded_text_v1`` =
    ``_decoded_text_for_panel``) applied to the branch at attest time; the
    AttestationPolicy recomputes the same hash at declare time and requires a
    match. ``branch`` is recorded for observability only — matching is by
    ``content_hash`` (F11 branch-rename edge). ``coherence`` is clamped to 0-10
    by the dispatcher (F5: the range is advisory).

    Slice 6 adds the diplomatic verifier fields; ``reader_accepts_as_solution``
    alone gates declaration (C6 reversed), ``coherence`` is report-only legacy.
    """

    branch: str
    content_hash: str
    renderer_id: str
    episode_id: str
    coherence: int = 0
    reader_accepts: bool = False
    gloss: str = ""
    anomalies: list[str] = field(default_factory=list)
    created_turn: int = 0
    # M5.3 Slice 6 — diplomatic verifier fields. Conservative defaults: a
    # record that never states them can neither declare nor route to repair.
    target_language_confidence: float = 0.0
    semantic_recoverability: float = 0.0
    damage_scope: str = "basin_wide"
    repairability: str = "none"
    reader_accepts_as_solution: bool = False
    uncertainty_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "content_hash": self.content_hash,
            "renderer_id": self.renderer_id,
            "episode_id": self.episode_id,
            "coherence": self.coherence,
            "reader_accepts": self.reader_accepts,
            "gloss": self.gloss,
            "anomalies": list(self.anomalies),
            "created_turn": self.created_turn,
            "target_language_confidence": self.target_language_confidence,
            "semantic_recoverability": self.semantic_recoverability,
            "damage_scope": self.damage_scope,
            "repairability": self.repairability,
            "reader_accepts_as_solution": self.reader_accepts_as_solution,
            "uncertainty_note": self.uncertainty_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttestationRecord":
        if "reader_accepts_as_solution" in data:
            # Strict: only a JSON true counts. Any other value (string "true",
            # 1, None) is conservative False.
            accepts_as_solution = data.get("reader_accepts_as_solution") is True
        else:
            # Legacy (pre-Slice-6) record: positive iff the prior
            # _is_positive_attestation condition held (master spec 472-475).
            accepts_as_solution = bool(data.get("reader_accepts")) and int(
                data.get("coherence") or 0
            ) >= LEGACY_DECLARE_COHERENCE
        return cls(
            branch=str(data.get("branch") or ""),
            content_hash=str(data.get("content_hash") or ""),
            renderer_id=str(data.get("renderer_id") or ""),
            episode_id=str(data.get("episode_id") or ""),
            coherence=int(data.get("coherence") or 0),
            reader_accepts=bool(data.get("reader_accepts")),
            gloss=str(data.get("gloss") or ""),
            anomalies=[str(a) for a in (data.get("anomalies") or [])],
            created_turn=int(data.get("created_turn") or 0),
            target_language_confidence=clamp_unit_interval(
                data.get("target_language_confidence")
            ),
            semantic_recoverability=clamp_unit_interval(
                data.get("semantic_recoverability")
            ),
            damage_scope=normalize_damage_scope(data.get("damage_scope")),
            repairability=normalize_repairability(data.get("repairability")),
            reader_accepts_as_solution=accepts_as_solution,
            uncertainty_note=str(data.get("uncertainty_note") or ""),
        )


def attestation_key(attestation: dict[str, Any] | None) -> str:
    """Identity of one verifier-evidence unit (M5.3 Slice 2).

    Primary component is the verify episode id (AttestationRecord has no id
    field; episode_id is unique per verify episode). Fallback for records
    without an episode_id (hand-seeded tests, legacy data): a digest over the
    verdict content (anomalies + coherence + reader_accepts). "none" when no
    attestation exists for the candidate.

    Known seam (documented, not built): keying by episode_id means a re-verify
    of unchanged content mints new verifier evidence and resets saturation.
    Re-verification of unchanged content is already discouraged by the system
    prompt and workflow hints; if paid runs show saturation laundering via
    re-verification, switch the primary component to the content digest below
    (a one-line change; both forms are specified here).
    """
    if not attestation:
        return "none"
    episode_id = str(attestation.get("episode_id") or "")
    if episode_id:
        return f"ep:{episode_id}"
    payload = {
        "anomalies": [str(a) for a in attestation.get("anomalies") or []],
        "coherence": int(attestation.get("coherence") or 0),
        "reader_accepts": bool(attestation.get("reader_accepts")),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"digest:{digest}"


def saturation_key(candidate_content_hash: str, att_key: str) -> str:
    """Key of one repair-saturation entry: (candidate content, verifier evidence).

    A new candidate hash OR genuinely new verifier evidence yields a new key —
    saturation reset is automatic (a fresh entry). A reworded goal changes
    neither component, so it never resets saturation.
    """
    return hashlib.sha1(
        f"{candidate_content_hash}|{att_key}".encode("utf-8")
    ).hexdigest()


def pair_digest(source_content_hash: str, interp_digest: str) -> str:
    """Identity of one evaluated source/interpretation pair.

    DIGEST-based (not reading_id-based): re-running a byte-identical reading
    under a fresh reading_id or a new as_name is the same pair.
    """
    return hashlib.sha1(
        f"{source_content_hash}|{interp_digest}".encode("utf-8")
    ).hexdigest()


def latest_attestation_for_hash(
    attestations: list[dict[str, Any]], content_hash: str
) -> dict[str, Any] | None:
    """Newest attestation matching ``content_hash`` by (created_turn, episode_id).

    Single shared selection rule — context._fresh_attestation and the
    repair-transaction dispatcher must both use THIS function (the dispatcher
    previously tie-broke on created_turn only)."""
    matches = [a for a in attestations if a.get("content_hash") == content_hash]
    return max(
        matches,
        key=lambda a: (int(a.get("created_turn") or 0), str(a.get("episode_id") or "")),
        default=None,
    )


def attestation_is_positive(attestation: dict[str, Any] | None) -> bool:
    """M5.3 Slice 6 SINGLE positive-attestation predicate (reverses C6).

    Positive == the independent reader accepts the candidate AS A SOLUTION.
    Every gating/routing/fallback consumer (AttestationPolicy, context
    workflow/hints, fallback tiering, agenda seeding, bakeoff telemetry) must
    call THIS function — the pre-Slice-6 pair (context._positive /
    loop_v3._is_positive_attestation) is deleted so the definition cannot
    drift again. Legacy dicts (no `reader_accepts_as_solution` key — e.g. a
    pre-Slice-6 artifact read raw, or a hand-seeded test record) fall back to
    the frozen pre-Slice-6 condition, mirroring AttestationRecord.from_dict.
    `coherence` appears ONLY in that legacy branch; it never gates a
    new-format record.
    """
    if not attestation:
        return False
    if "reader_accepts_as_solution" in attestation:
        return attestation.get("reader_accepts_as_solution") is True
    return bool(attestation.get("reader_accepts")) and int(
        attestation.get("coherence") or 0
    ) >= LEGACY_DECLARE_COHERENCE


def new_saturation_entry(
    candidate_content_hash: str, att_key: str, turn: int
) -> dict[str, Any]:
    """Fresh repair-saturation entry (all JSON-native; see field table)."""
    return {
        "candidate_content_hash": candidate_content_hash,
        "attestation_key": att_key,
        "evidence_failures": 0,
        "process_failures": {},
        "evidence_failed_pairs": [],
        "finalist_hashes": [],
        "readings": 0,
        "exhausted": False,
        "pending_experiment_id": None,
        "created_turn": turn,
        "updated_turn": turn,
    }


def get_or_create_saturation_entry(
    state: "InvestigationState",
    candidate_content_hash: str,
    att_key: str,
    turn: int,
) -> dict[str, Any]:
    key = saturation_key(candidate_content_hash, att_key)
    entry = state.repair_saturation.get(key)
    if entry is None:
        entry = new_saturation_entry(candidate_content_hash, att_key, turn)
        state.repair_saturation[key] = entry
    return entry


def _serialize_cipher(state: "InvestigationState") -> dict[str, Any]:
    """Serialize the full CipherText + plaintext alphabet (A3).

    Resume NEVER parses prompt text; it reconstructs the CipherText from these
    fields. ``tokens`` / ``word_lengths`` are stored for verification, but the
    authoritative reconstruction re-runs ``CipherText`` over (raw, alphabet,
    separator), which reproduces identical tokens by construction.
    """
    ct = state.cipher
    ws = state.workspace
    return {
        "raw": ct.raw,
        "separator": ct.separator,
        "source": ct.source,
        "cipher_symbols": list(ct.alphabet.symbols),
        "plaintext_symbols": list(ws.plaintext_alphabet.symbols),
        "tokens": list(ct.tokens),
        "word_lengths": [len(w) for w in ct.words],
    }


def _serialize_branch(ws: Workspace, name: str) -> dict[str, Any]:
    """Full branch snapshot including the three fields v2 resume drops (A3)."""
    branch = ws.get_branch(name)
    return {
        "name": name,
        "parent": branch.parent,
        "created_iteration": branch.created_iteration,
        "key": {str(k): int(v) for k, v in branch.key.items()},
        "tags": list(branch.tags),
        "metadata": dict(branch.metadata),
        "word_spans": (
            [[int(s), int(e)] for (s, e) in branch.word_spans]
            if branch.word_spans is not None
            else None
        ),
        "token_order": (
            list(branch.token_order) if branch.token_order is not None else None
        ),
        "transform_pipeline": branch.transform_pipeline,
    }


@dataclass
class InvestigationState:
    """Everything durable about one v3 run (C1)."""

    workspace: Workspace
    language: str = "en"
    # Optional external / benchmark context (prior-context, benchmark
    # prompt). Set once at run start; rendered every turn as its OWN stable
    # context section (in the cacheable prefix), NOT a turn-0 evidence entry
    # that scrolls out of the recent-evidence window (R3). Never ground
    # truth — the firewall covers this surface.
    external_context: str = ""
    evidence_log: list[EvidenceEntry] = field(default_factory=list)
    budget_ledger: list[BudgetEntry] = field(default_factory=list)
    # Last-N whole exchanges stored as NATIVE-format message dicts (assistant
    # blocks incl. provider_extra, and their tool_result messages). Rendered
    # UNTRANSFORMED by the context builder (F1).
    recent_exchanges: list[dict[str, Any]] = field(default_factory=list)
    # Relocated durable repair agenda (F5). The executor is constructed to share
    # this list object so agenda edits are reflected in state.
    repair_agenda: list[dict[str, Any]] = field(default_factory=list)
    # M4 schema-only (present, empty).
    experiment_queue: list[dict[str, Any]] = field(default_factory=list)
    # M2: append-only ledger of completed episodes (one dict per episode).
    episode_ledger: list[dict[str, Any]] = field(default_factory=list)
    # M5: verify-attestation records (AttestationRecord.to_dict()). Written by
    # the lead dispatcher on a completed ``verify`` episode; read by the
    # AttestationPolicy at declare time (matched by content_hash). Named
    # distinctly from v2's executor._reading_attestations (F11). Absent from
    # pre-M5 artifacts -> empty on load (extends resume identity).
    verify_attestations: list[dict[str, Any]] = field(default_factory=list)
    # M3: stored Readings (reading_id -> Reading.to_dict()). The lead compiles a
    # reading-kind episode result into a Reading here; workers never write it
    # (A1). Absent from M2 artifacts -> empty on load (extends resume identity).
    readings: dict[str, dict[str, Any]] = field(default_factory=dict)
    # M5.2: host-owned repair transactions. Each record binds source/result
    # content hashes, the worker episode, installed branch, and anomalies the
    # transaction attempted to address. This is durable across resume.
    repair_transactions: list[dict[str, Any]] = field(default_factory=list)
    # M5.3 Slice 2: durable repair-saturation counters, keyed by
    # saturation_key(candidate_content_hash, attestation_key). Keying by
    # (content, evidence) makes reset-on-new-content/new-evidence automatic
    # (a new pair is a fresh entry) and makes reworded goals irrelevant (the
    # goal is not in the key). Entries are created/mutated ONLY by loop_v3;
    # context.py reads them. Absent from pre-M5.3 artifacts -> empty on load.
    repair_saturation: dict[str, dict[str, Any]] = field(default_factory=dict)
    branch_aliases: list[dict[str, Any]] = field(default_factory=list)
    call_signature_counts: dict[str, int] = field(default_factory=dict)
    no_new_information_streak: int = 0
    last_information_digest: str | None = None
    # M5.1: workflow hints already shown/emitted, keyed by event + candidate
    # content hash. This keeps guidance useful without repeating it every turn
    # and survives resume.
    workflow_hint_keys: list[str] = field(default_factory=list)
    # A1/A10: the state-owned finalist-session store (shared with the lead and
    # every episode executor) and the single-writer hypothesis board. Both
    # survive resume so a search episode's finalist session and the hypothesis
    # trail are reviewable/installable after reload.
    finalist_sessions: FinalistSessionStore = field(default_factory=FinalistSessionStore)
    hypothesis_board: HypothesisBoard = field(default_factory=HypothesisBoard)
    # Active language-model variant selection (act_set_model_variant). The lead
    # loop mirrors the lead executor's selection here after each dispatched tool
    # call; run_episode seeds each fresh episode executor from this field; it
    # serializes/restores so a v3 resume keeps the selection. ``None`` = default.
    model_variant: str | None = None
    turn: int = 0
    max_recent_exchanges: int = DEFAULT_RECENT_EXCHANGES

    def __post_init__(self) -> None:
        # Adopt any hypothesis branches already present in the workspace (a
        # directly-populated workspace, or one restored from an artifact whose
        # board did not carry every card — F4/F11). Existing cards are untouched.
        self.hypothesis_board.sync_from_workspace(self.workspace)

    # --- convenience ---
    @property
    def cipher(self) -> CipherText:
        return self.workspace.cipher_text

    def hypothesis_cards(self) -> list[dict[str, Any]]:
        """Board cards for context rendering (A10).

        Cards whose branch has since been deleted from the workspace are
        filtered out here (stale-card handling); the board itself keeps them so
        the full hypothesis trail still serializes into the artifact.
        """
        return [
            card for card in self.hypothesis_board.cards()
            if self.workspace.has_branch(str(card.get("branch") or ""))
        ]

    # --- mutation helpers ---
    def add_evidence(self, kind: str, turn: int, summary: str = "", **data: Any) -> None:
        self.evidence_log.append(
            EvidenceEntry(kind=kind, turn=turn, summary=summary, data=dict(data))
        )

    def add_budget(self, entry: BudgetEntry) -> None:
        self.budget_ledger.append(entry)

    def record_exchange(
        self,
        assistant_message: dict[str, Any],
        tool_result_message: dict[str, Any] | None,
    ) -> None:
        """Append one whole exchange and trim to ``max_recent_exchanges``.

        An exchange is the assistant turn plus its tool_result turn. Trimming
        drops the OLDEST WHOLE exchange, never splitting one — a tool_result
        without its tool_use, or a reasoning item without its siblings, 400s on
        the Responses API (F1).
        """
        self.recent_exchanges.append(assistant_message)
        if tool_result_message is not None:
            self.recent_exchanges.append(tool_result_message)
        self._trim_recent_exchanges()

    def _trim_recent_exchanges(self) -> None:
        # Group the flat message list into whole exchanges. Each exchange starts
        # at an assistant message and includes any immediately-following
        # non-assistant messages (its tool_result). Keep the last N.
        exchanges: list[list[dict[str, Any]]] = []
        for message in self.recent_exchanges:
            role = message.get("role")
            if role == "assistant" or not exchanges:
                exchanges.append([message])
            else:
                exchanges[-1].append(message)
        kept = exchanges[-self.max_recent_exchanges:] if self.max_recent_exchanges else []
        self.recent_exchanges = [m for group in kept for m in group]

    def budget_by_category(self) -> dict[str, dict[str, Any]]:
        """Aggregate the budget ledger by category (A7)."""
        out: dict[str, dict[str, Any]] = {}
        for entry in self.budget_ledger:
            bucket = out.setdefault(
                entry.category,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cost_usd": 0.0,
                    "calls": 0,
                },
            )
            bucket["input_tokens"] += entry.input_tokens
            bucket["output_tokens"] += entry.output_tokens
            bucket["cache_read_tokens"] += entry.cache_read_tokens
            bucket["cost_usd"] += entry.cost()
            bucket["calls"] += 1
        # M4/A7: experiments are token-free solver compute — no BudgetEntry rows.
        # Append, additively, one ``experiment:<type>`` bucket per type present in
        # the queue: ``calls`` counts records that reached running (started_at
        # set); ``elapsed_seconds`` sums completed/failed records only (F6); token
        # fields and cost stay 0.
        for record in self.experiment_queue:
            cat = f"experiment:{record.get('type') or 'unknown'}"
            bucket = out.setdefault(
                cat,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cost_usd": 0.0,
                    "calls": 0,
                    "elapsed_seconds": 0.0,
                },
            )
            if record.get("started_at") is not None:
                bucket["calls"] += 1
            if record.get("status") in {"completed", "failed"}:
                bucket["elapsed_seconds"] += float(record.get("elapsed_seconds") or 0.0)
        return out

    def total_cost(self) -> float:
        return sum(entry.cost() for entry in self.budget_ledger)

    # --- serialization (resume identity) ---
    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "cipher": _serialize_cipher(self),
            "language": self.language,
            "external_context": self.external_context,
            "workspace": {
                "plaintext_symbols": list(self.workspace.plaintext_alphabet.symbols),
                "branches": [
                    _serialize_branch(self.workspace, name)
                    for name in self.workspace.branch_names()
                ],
            },
            "hypothesis_board": self.hypothesis_board.to_dict(),
            "evidence_log": [entry.to_dict() for entry in self.evidence_log],
            "budget_ledger": [entry.to_dict() for entry in self.budget_ledger],
            "recent_exchanges": self.recent_exchanges,
            "repair_agenda": [dict(item) for item in self.repair_agenda],
            "episode_ledger": [dict(item) for item in self.episode_ledger],
            "verify_attestations": [dict(item) for item in self.verify_attestations],
            "readings": {rid: dict(r) for rid, r in self.readings.items()},
            "repair_transactions": [
                dict(item) for item in self.repair_transactions
            ],
            "repair_saturation": {
                str(key): {
                    **entry,
                    "process_failures": dict(entry.get("process_failures") or {}),
                    "evidence_failed_pairs": list(entry.get("evidence_failed_pairs") or []),
                    "finalist_hashes": list(entry.get("finalist_hashes") or []),
                }
                for key, entry in self.repair_saturation.items()
            },
            "branch_aliases": [dict(item) for item in self.branch_aliases],
            "call_signature_counts": dict(self.call_signature_counts),
            "no_new_information_streak": self.no_new_information_streak,
            "last_information_digest": self.last_information_digest,
            "workflow_hint_keys": list(self.workflow_hint_keys),
            "experiment_queue": [dict(item) for item in self.experiment_queue],
            "finalist_sessions": self.finalist_sessions.to_dict(),
            "model_variant": self.model_variant,
            "turn": self.turn,
        }

    @classmethod
    def from_artifact_dict(cls, data: dict[str, Any]) -> "InvestigationState":
        cipher_data = data["cipher"]
        cipher_alpha = Alphabet(list(cipher_data["cipher_symbols"]))
        pt_alpha = Alphabet(list(cipher_data["plaintext_symbols"]))
        cipher = CipherText(
            raw=cipher_data["raw"],
            alphabet=cipher_alpha,
            source=cipher_data.get("source", "manual"),
            separator=cipher_data.get("separator"),
        )
        workspace = Workspace(cipher_text=cipher, plaintext_alphabet=pt_alpha)
        for branch_data in data.get("workspace", {}).get("branches", []):
            _restore_branch_into(workspace, branch_data)

        # F11: the M2 board serializes as ``{"cards": [...], "next_id": N}``.
        # An M1-era artifact stored a projection LIST (or nothing) there; fall
        # back to projecting from the restored workspace so the resume-identity
        # test keeps passing unchanged.
        board_data = data.get("hypothesis_board")
        if isinstance(board_data, dict):
            board = HypothesisBoard.from_dict(board_data)
        else:
            board = HypothesisBoard.from_workspace(workspace)

        state = cls(
            workspace=workspace,
            language=str(data.get("language") or "en"),
            external_context=str(data.get("external_context") or ""),
            evidence_log=[
                EvidenceEntry.from_dict(e) for e in data.get("evidence_log") or []
            ],
            budget_ledger=[
                BudgetEntry.from_dict(b) for b in data.get("budget_ledger") or []
            ],
            recent_exchanges=[dict(m) for m in data.get("recent_exchanges") or []],
            repair_agenda=[dict(item) for item in data.get("repair_agenda") or []],
            episode_ledger=[dict(item) for item in data.get("episode_ledger") or []],
            # Slice 6: normalize every stored attestation through
            # AttestationRecord.from_dict so legacy records gain the new
            # fields (conservative defaults; positivity derived from the
            # frozen legacy condition) and resume behaves like a live run.
            verify_attestations=[
                AttestationRecord.from_dict(dict(item)).to_dict()
                for item in data.get("verify_attestations") or []
            ],
            readings={
                str(rid): dict(r)
                for rid, r in (data.get("readings") or {}).items()
            },
            repair_transactions=[
                dict(item) for item in data.get("repair_transactions") or []
            ],
            repair_saturation={
                str(key): _normalize_saturation_entry(value)
                for key, value in (data.get("repair_saturation") or {}).items()
                if isinstance(value, dict)
            },
            branch_aliases=[dict(item) for item in data.get("branch_aliases") or []],
            call_signature_counts={
                str(key): int(value)
                for key, value in (data.get("call_signature_counts") or {}).items()
            },
            no_new_information_streak=int(data.get("no_new_information_streak") or 0),
            last_information_digest=(
                str(data["last_information_digest"])
                if data.get("last_information_digest") is not None else None
            ),
            workflow_hint_keys=[
                str(key) for key in (data.get("workflow_hint_keys") or [])
            ],
            experiment_queue=_normalize_loaded_experiment_records(
                data.get("experiment_queue") or []
            ),
            finalist_sessions=FinalistSessionStore.from_dict(
                data.get("finalist_sessions")
            ),
            hypothesis_board=board,
            model_variant=(
                str(data["model_variant"])
                if data.get("model_variant") is not None
                else None
            ),
            turn=int(data.get("turn") or 0),
        )
        return state


def _normalize_saturation_entry(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_content_hash": str(value.get("candidate_content_hash") or ""),
        "attestation_key": str(value.get("attestation_key") or "none"),
        "evidence_failures": int(value.get("evidence_failures") or 0),
        "process_failures": {
            str(k): int(v)
            for k, v in (value.get("process_failures") or {}).items()
        },
        "evidence_failed_pairs": sorted(
            str(p) for p in value.get("evidence_failed_pairs") or []
        ),
        "finalist_hashes": sorted(
            str(h) for h in value.get("finalist_hashes") or []
        ),
        "readings": int(value.get("readings") or 0),
        "exhausted": bool(value.get("exhausted")),
        "pending_experiment_id": (
            str(value["pending_experiment_id"])
            if value.get("pending_experiment_id") else None
        ),
        "created_turn": int(value.get("created_turn") or 0),
        "updated_turn": int(value.get("updated_turn") or 0),
    }


def _normalize_loaded_experiment_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """A9 load transition: any ``pending|running`` experiment record loaded from
    an artifact becomes ``orphaned(loaded)`` (its worker thread did not survive
    serialization). Completed/failed/orphaned records load verbatim — results,
    sessions, and dedup keys stay live so resume-by-resubmission is cheap.

    Implemented inline here (a helper in ``experiments.py`` would import-cycle:
    experiments imports ``_serialize_branch``/``_restore_branch_into`` from this
    module).
    """
    out: list[dict[str, Any]] = []
    for item in records:
        record = dict(item)
        if record.get("status") in {"pending", "running"}:
            record["status"] = "orphaned"
            record["orphan_reason"] = "loaded"
        out.append(record)
    return out


def _restore_branch_into(ws: Workspace, branch_data: dict[str, Any]) -> None:
    key = {int(k): int(v) for k, v in (branch_data.get("key") or {}).items()}
    word_spans = branch_data.get("word_spans")
    spans = (
        [(int(s), int(e)) for (s, e) in word_spans]
        if word_spans is not None
        else None
    )
    token_order = branch_data.get("token_order")
    ws.restore_branch(
        str(branch_data["name"]),
        key=key,
        parent=branch_data.get("parent"),
        created_iteration=int(branch_data.get("created_iteration") or 0),
        tags=[str(t) for t in branch_data.get("tags") or []],
        word_spans=spans,
        token_order=list(token_order) if token_order is not None else None,
        transform_pipeline=branch_data.get("transform_pipeline"),
        metadata=dict(branch_data.get("metadata") or {}),
    )
