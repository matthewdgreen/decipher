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
    # M3: stored Readings (reading_id -> Reading.to_dict()). The lead compiles a
    # reading-kind episode result into a Reading here; workers never write it
    # (A1). Absent from M2 artifacts -> empty on load (extends resume identity).
    readings: dict[str, dict[str, Any]] = field(default_factory=dict)
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
            "readings": {rid: dict(r) for rid, r in self.readings.items()},
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
            readings={
                str(rid): dict(r)
                for rid, r in (data.get("readings") or {}).items()
            },
            experiment_queue=[dict(item) for item in data.get("experiment_queue") or []],
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
