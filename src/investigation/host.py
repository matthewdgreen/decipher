"""InvestigationHost: the loop-independent v3 lead dispatch layer.

Extracted verbatim from ``run_v3``'s nested closures (MCP plan, host-
extraction slice; see docs/mcp_dual_harness_proposal.md §7.0). The host
owns everything that turns one lead ``tool_use`` into one tool-result
string — episode dispatch (including verify attestation writes), episode
install, the bounded repair transaction and its acceptance checks,
duplicate suppression, the lead read cache, the information digest, and
the per-run budget seam — while ``run_v3`` remains the turn-loop driver
(context build, session sends, truncation/nudge/429 handling, termination,
artifact finalize). A future MCP server is a second thin driver over this
same class; behavior here must stay identical for both entry points.

Epistemic gates (M5.3): the verification-gated declaration policy lives in
the executor's AttestationPolicy; the host enforces repair acceptance
(default-deny), saturation, duplicate suppression, and episode-kind gating
exactly as the loop closures did.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from typing import Any, Callable

from agent.loop_shared import (
    DECODED_TEXT_RENDERER_ID,
    _candidate_content_hash,
    _decoded_text_for_panel,
)
from agent.tools_v2 import WorkspaceToolExecutor
from artifact.schema import ToolCall
from investigation.actions import COMPOSITE_TOOL_NAMES, execute_composite
from investigation.board import CARD_MIRROR_KEYS
from investigation.context import allowed_episode_kinds, workflow_state
from investigation.episodes import EpisodeSpec, run_episode
from investigation.experiments import (
    EXPERIMENT_TOOL_NAMES,
    ExperimentQueue,
    dispatch_experiment_collect,
    dispatch_experiment_submit,
)
from investigation.reading import Reading, build_candidate_reading_packet
from investigation.sessions import ModelSession
from investigation.state import (
    AttestationRecord,
    BudgetEntry,
    InvestigationState,
    attestation_is_positive,
    clamp_unit_interval,
    latest_attestation_for_hash,
    normalize_damage_scope,
    normalize_repairability,
)
from workspace import Workspace


# emit(event, payload, **extra) — extra carries LoopEvent kwargs such as
# ``outer_iteration``. Provided by the driver (run_v3 today; MCP later).
EmitFn = Callable[..., None]


def _branch_hash(workspace: Workspace, branch: str) -> str:
    return _candidate_content_hash(_decoded_text_for_panel(workspace, branch))


def _active_branch(workspace: Workspace, branch: str) -> bool:
    return workspace.has_branch(branch) and workspace.get_branch(branch).metadata.get(
        "mode_status", "active"
    ) not in {"rejected", "superseded"}


def _resync_attestation_branch_on_rename(
    workspace: Workspace,
    attestations: list[dict[str, Any]],
    target: str,
    name: str,
) -> None:
    """M6 F5: re-point verify-attestation ``branch`` labels after a rename.

    ``episode_install_branch`` renames the branch it installs from its intended
    ``target`` to ``name`` when ``target`` is already taken (collision). The
    dispatcher owns both that rename and the verify-attestation writes; those
    records are matched at declare time PRIMARILY by ``content_hash`` — the
    ``branch`` field is the observability label and the source of the
    stale-vs-required message.

    When a rename moved attested content onto ``name``, re-point ONLY the records
    whose hash matches the freshly-installed branch AND no longer matches the
    (different, or now-absent) branch still holding ``target``. This never steals
    a label from a branch that legitimately still owns its attested content — so
    the stale-vs-required label and the declaration-carried attestation keep
    naming a live branch, with no mislabel. Pure over its arguments (mutates the
    passed-in records list in place).
    """
    if name == target:
        return
    installed_hash = _candidate_content_hash(
        _decoded_text_for_panel(workspace, name)
    )
    target_hash = (
        _candidate_content_hash(_decoded_text_for_panel(workspace, target))
        if workspace.has_branch(target)
        else None
    )
    if installed_hash == target_hash:
        return
    for record in attestations:
        if (
            record.get("branch") == target
            and record.get("content_hash") == installed_hash
        ):
            record["branch"] = name


def _episode_result_digest(
    kind: str,
    status: str,
    failure_reason: str | None,
    result: Any,
) -> str:
    """CLI-3: one-line plain-English outcome for a finished episode.

    Built lead-side from the episode's structured result so the narrate display
    can render an outcome (`↳ verify: NEGATIVE — …`) instead of a bare status.
    Pure + module-level so it is unit-testable in isolation. Tolerant of missing
    keys (old/partial result dicts).
    """
    kind = str(kind or "episode")
    if status != "ok":
        reason = str(failure_reason or status or "no result")
        return f"{kind} failed: {reason}"
    data = result if isinstance(result, dict) else {}

    if kind == "verify":
        from investigation.state import attestation_is_positive

        positive = attestation_is_positive(data)
        verdict = "POSITIVE" if positive else "NEGATIVE"
        gloss = _truncate_text(data.get("gloss"), 240)
        lang = _fmt_unit(data.get("target_language_confidence"))
        recov = _fmt_unit(data.get("semantic_recoverability"))
        scope = str(data.get("damage_scope") or "?")
        repairability = str(data.get("repairability") or "?")
        head = f'{verdict} — "{gloss}"' if gloss else verdict
        return (
            f"{head} (lang {lang}, recoverability {recov}, "
            f"damage {scope} → {repairability})"
        )

    if kind == "search":
        best = data.get("best_branch")
        notes = _truncate_text(data.get("notes"), 240)
        if data.get("improved") and best:
            head = f"improved → {best}"
        else:
            head = "no improvement"
        return f"{head} — {notes}" if notes else head

    if kind == "reading":
        text = _truncate_text(data.get("reading_text"), 200)
        conf = data.get("overall_confidence")
        conf_str = _fmt_unit(conf) if conf is not None else "?"
        return f'proposed "{text}" (confidence {conf_str})'

    if kind == "compare":
        winner = data.get("winner")
        ranking = [str(b) for b in (data.get("ranking") or [])][:3]
        head = f"winner {winner}" if winner else "no winner"
        if ranking:
            head += f" (ranked: {', '.join(ranking)})"
        return head

    if kind == "repair":
        if data.get("applied"):
            n = len(data.get("edits") or [])
            best = data.get("best_branch") or "?"
            return f"applied {n} edit(s) → {best}"
        return "did not apply"

    if kind == "survey":
        findings = data.get("findings") or []
        if findings:
            return _truncate_text(findings[0], 240)
        return "survey complete"

    return f"{kind} complete"


def _truncate_text(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _fmt_unit(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "?"


# M5.3 Slice 4: the scalar-decrease acceptance policy hook. ``None`` = default
# deny (reject any net scalar decrease). No worker-improvisable policy is
# implemented in M5.3; a tested, ground-truth-free policy object is an M5.4
# non-goal represented by this stub.
REPAIR_ACCEPTANCE_POLICY: Any = None  # M5.4 hook; None = default deny


# M5.3 Slice 2: the enumerated evidence-class failure reasons. Everything else
# (precondition/process failure modes, provider error strings, new episode
# failure modes) classifies as ``process`` so an unknown reason never silently
# consumes saturation budget on first occurrence.
_EVIDENCE_FAILURE_REASONS = frozenset({
    "no_changed_finalists",
    "no_op",
    "all_finalists_rejected",
    "materially_non_improving",
})


def _classify_failure_reason(reason: Any) -> str:
    """'evidence' for the enumerated evidence reasons; 'process' otherwise.

    Default-process is deliberate: an unknown reason (new episode failure
    modes, provider error strings) must not silently consume saturation
    budget on first occurrence — the second-process-failure rule converts
    repeats into evidence failures."""
    return "evidence" if str(reason or "") in _EVIDENCE_FAILURE_REASONS else "process"


_REPAIR_COMPOSITE_NAMES = frozenset({
    "hypothesis_apply_reading", "hypothesis_test_word",
    "hypothesis_test_words", "branch_adjudicate",
})


def _extract_repair_evidence(tool_calls: list[Any]) -> dict[str, Any]:
    """Parse the repair episode's composite ToolCall results into the
    evidence sets the acceptance checks bind against. ``tool_calls`` is the
    episode-filtered list; each item has .tool_name and .result (JSON str).

    A result is SUCCESSFUL iff it parses to a dict with status == "ok" and no
    top-level "error" key. Failed results contribute nothing to
    supported_forks / edit_evidence (this, plus the failed_result_forks scan,
    realizes "no unresolved error invalidates a claimed edit")."""
    supported_forks: set[str] = set()
    edit_evidence: set[str] = set()
    adjudicated_sets: list[set[str]] = []
    batch_finalist_forks: set[str] = set()
    failed_result_forks: set[str] = set()
    for call in tool_calls:
        if getattr(call, "tool_name", None) not in _REPAIR_COMPOSITE_NAMES:
            continue
        try:
            parsed = json.loads(getattr(call, "result", "") or "")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        ok = parsed.get("status") == "ok" and "error" not in parsed
        fork_fields: list[str] = []
        if isinstance(parsed.get("fork"), str):
            fork_fields.append(parsed["fork"])
        if isinstance(parsed.get("installed_fork"), str):
            fork_fields.append(parsed["installed_fork"])
        for name in parsed.get("installed") or []:
            if isinstance(name, str):
                fork_fields.append(name)
        items = [i for i in parsed.get("items") or [] if isinstance(i, dict)]
        for item in items:
            if isinstance(item.get("installed_fork"), str):
                fork_fields.append(item["installed_fork"])
        if not ok:
            failed_result_forks.update(fork_fields)
            continue
        if call.tool_name == "branch_adjudicate":
            adjudicated_sets.append({
                str(row.get("branch") or "")
                for row in parsed.get("rows") or []
                if isinstance(row, dict)
            })
            continue
        supported_forks.update(fork_fields)
        edit_evidence.update(
            str(e).strip() for e in parsed.get("edits") or []
        )
        for item in items:
            if item.get("status") == "ok":
                edit_evidence.update(
                    str(e).strip() for e in item.get("edits") or []
                )
        for finalist in parsed.get("finalists") or []:
            if isinstance(finalist, dict) and isinstance(
                finalist.get("installed_fork"), str
            ):
                batch_finalist_forks.add(finalist["installed_fork"])
    return {
        "supported_forks": supported_forks,
        "edit_evidence": edit_evidence,
        "adjudicated_sets": adjudicated_sets,
        "batch_finalist_forks": batch_finalist_forks,
        "failed_result_forks": failed_result_forks,
    }


def _worker_rejected_targets(result: dict[str, Any]) -> set[str]:
    """Fork names the worker's own verdicts explicitly reject."""
    rejected: set[str] = set()
    for verdict in result.get("verdicts") or []:
        if not isinstance(verdict, dict):
            continue
        word = str(verdict.get("verdict") or "").strip().lower()
        if "reject" in word or word in {"discard", "discarded", "invalid", "not viable"}:
            rejected.add(str(verdict.get("target") or ""))
    return rejected


_EDIT_LABEL_IN_PROSE_RE = re.compile(
    r"(?:[A-Za-z0-9_]+:[A-Z?]->[A-Z?]|[A-Za-z0-9_]+=[A-Z?])"
)


def _unbound_edit_claims(
    claims: list[str], edit_evidence: set[str]
) -> tuple[list[str], dict[str, list[str]]]:
    """Bind worker edit claims to host-produced labels, failing closed.

    Exact labels are preferred. A prose-bearing legacy claim is accepted only
    when it contains mapping-shaped labels and every such label is present in
    successful composite evidence. This handles ``Applied ... W:D->F ...``
    without allowing invented mappings or unstructured prose through the gate.
    """
    unbound: list[str] = []
    normalized: dict[str, list[str]] = {}
    for claim in claims:
        if claim in edit_evidence:
            normalized[claim] = [claim]
            continue
        labels = _EDIT_LABEL_IN_PROSE_RE.findall(claim)
        if labels and all(label in edit_evidence for label in labels):
            normalized[claim] = labels
            continue
        unbound.append(claim)
    return sorted(unbound), normalized


def _winner_adjudication_summary(
    tool_calls: list[Any], winner: str
) -> dict[str, Any] | None:
    """The adjudication_summary of the LAST successful composite result that
    produced ``winner`` (top-level for the singleton, item-level for the
    batch; ``hypothesis_apply_reading`` has none → None)."""
    summary: dict[str, Any] | None = None
    for call in tool_calls:
        if getattr(call, "tool_name", None) not in _REPAIR_COMPOSITE_NAMES:
            continue
        try:
            parsed = json.loads(getattr(call, "result", "") or "")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        if not (parsed.get("status") == "ok" and "error" not in parsed):
            continue
        if parsed.get("fork") == winner or parsed.get("installed_fork") == winner:
            summary = parsed.get("adjudication_summary")
        for item in parsed.get("items") or []:
            if isinstance(item, dict) and item.get("installed_fork") == winner:
                summary = item.get("adjudication_summary")
    return summary if isinstance(summary, dict) else None


def _clamp_coherence(value: Any) -> int:
    """Coerce the verify coherence to the 0-10 int scale (F5 + review F-1).

    A value ABOVE 10 is a scale violation (the contract states 0-10; a
    reader answering e.g. 12 is using some other scale, likely 0-100 —
    which would make 12 a LOW reading). Recording it as 10/10 would mint a
    top-coherence attestation on a decode the reader may have rejected, so
    out-of-scale values are recorded as the conservative floor 0 instead —
    the verdict signal lives in reader_accepts/anomalies either way, and a
    low coherence can never make a weak solve look strong. Negative and
    unparseable values also floor to 0.
    """
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        try:
            coerced = int(float(value))
        except (TypeError, ValueError):
            return 0
    if coerced > 10:
        return 0  # scale violation -> conservative floor, never maximum
    return max(0, coerced)


READ_ONLY_LEAD_TOOLS = frozenset({
    "decode_show",
    "repair_agenda_list",
    "branch_adjudicate",
    "inspect_benchmark_context",
    "list_related_records",
    "inspect_related_transcription",
    "inspect_related_solution",
    "list_associated_documents",
    "inspect_associated_document",
})


class InvestigationHost:
    """Owns the v3 lead dispatch layer over (state, workspace, executor,
    queue) plus the run's budget accounting. One instance per run/driver.
    The driver must call ``set_available_tools`` before ``handle_tool``
    each turn (an empty set blocks every tool, matching the loop's
    per-turn tool-list gate)."""

    def __init__(
        self,
        *,
        state: InvestigationState,
        workspace: Workspace,
        executor: WorkspaceToolExecutor,
        queue: ExperimentQueue,
        emit: EmitFn,
        session: ModelSession,
        model_provider: Any = None,
        language: str,
        word_set: set[str],
        word_list: list[str],
        pattern_dict: dict[str, Any],
        episode_models: dict[str, str] | None = None,
        max_cost_usd: float | None = None,
        prior_budget: list[BudgetEntry] | None = None,
    ) -> None:
        if workspace is not state.workspace:
            raise ValueError(
                "InvestigationHost requires workspace is state.workspace"
            )
        self.state = state
        self.workspace = workspace
        self.executor = executor
        self.queue = queue
        self._emit = emit
        self._session = session
        self._model_provider = model_provider
        self.language = language
        self._word_set = word_set
        self._word_list = word_list
        self._pattern_dict = pattern_dict
        self._episode_models = episode_models
        self.max_cost_usd = max_cost_usd
        self._prior_budget: list[BudgetEntry] = list(prior_budget or [])
        # Mutable dispatch-layer state (was run_v3 closure variables):
        # Episode spend accumulates here across turns so it survives the
        # per-turn ledger rebuild (run_episode also extends state.budget_ledger
        # for direct callers; the rebuild is the single source of truth for the
        # lead run).
        self.episode_budget: list[BudgetEntry] = []       # was L700
        self.episode_tool_calls: list[Any] = []           # was L774
        self._available_tools: set[str] = set()           # was current_lead_tool_names, L775
        self._read_call_cache: dict[
            tuple[str, str, tuple[tuple[str, str], ...]], str
        ] = {}                                            # was read_call_cache, L777

    def handle_tool(self, tu: dict[str, Any], turn: int) -> str:
        name = tu["name"]
        if name not in self._available_tools:
            self._emit("lead_tool_rejected", {
                "tool": name,
                "reason": "lead_tool_not_available",
                "workflow_state": workflow_state(self.state, self.executor)["state"],
            }, outer_iteration=turn)
            return self._record_dispatch_result(
                name=name,
                tu=tu,
                turn=turn,
                payload={
                    "status": "blocked",
                    "reason": "lead_tool_not_available",
                    "tool": name,
                    "note": (
                        "This operator tool is not available to the v3 lead. "
                        "Delegate the work through an episode or experiment."
                    ),
                },
            )
        args = tu.get("input") or {}
        signature_payload = {
            "tool": name,
            "arguments": args,
            "branch_hashes": self._lead_read_cache_key(name, args)[2],
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        self.state.call_signature_counts[signature] = (
            self.state.call_signature_counts.get(signature, 0) + 1
        )
        if self.state.call_signature_counts[signature] > 1:
            self._emit("repeated_call", {
                "tool": name,
                "count": self.state.call_signature_counts[signature],
                "signature": signature,
            }, outer_iteration=turn)
        if name == "episode_run":
            requested_kind = str(args.get("kind") or "")
            live_kinds = set(allowed_episode_kinds(self.state, self.executor))
            if requested_kind not in live_kinds:
                return self._record_dispatch_result(
                    name=name,
                    tu=tu,
                    turn=turn,
                    payload={
                        "status": "blocked",
                        "reason": "episode_kind_not_available",
                        "requested_kind": requested_kind,
                        "allowed_kinds": sorted(live_kinds),
                        "workflow_state": workflow_state(self.state, self.executor)["state"],
                    },
                )
        cache_key = None
        if name in READ_ONLY_LEAD_TOOLS:
            cache_key = self._lead_read_cache_key(name, args)
            if cache_key in self._read_call_cache:
                self._emit("duplicate_read_suppressed", {
                    "tool": name,
                    "workflow_state": workflow_state(self.state, self.executor)["state"],
                }, outer_iteration=turn)
                return self._record_dispatch_result(
                    name=name,
                    tu=tu,
                    turn=turn,
                    payload={
                        "status": "duplicate_suppressed",
                        "tool": name,
                        "note": (
                            "The same read was already performed against "
                            "unchanged content; use the investigation state."
                        ),
                    },
                )
        if name == "episode_run":
            result = self._dispatch_episode_run(tu, turn)
            return result
        if name == "episode_install_branch":
            return self._dispatch_episode_install(tu, turn)
        if name == "repair_transaction":
            phase = workflow_state(self.state, self.executor)["state"]
            if phase not in {"candidate_reading", "repair_required"}:
                return self._record_dispatch_result(
                    name=name, tu=tu, turn=turn,
                    payload={
                        "status": "blocked",
                        "reason": "repair_transaction_not_ready",
                        "workflow_state": phase,
                    },
                )
            try:
                return self._dispatch_repair_transaction(tu, turn)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - preserve the run artifact
                return self._record_dispatch_result(
                    name=name, tu=tu, turn=turn,
                    payload={
                        "status": "failed",
                        "reason": "transaction_error",
                        "error": str(exc),
                    },
                )
        # M4: experiment-queue lead tools are routed before executor.execute (the
        # episode_* pattern); they cannot change the model variant, so they
        # early-return WITHOUT the mirror below.
        if name in EXPERIMENT_TOOL_NAMES:
            args = tu.get("input") or {}
            # F-4: an unexpected exception (e.g. collect(install=true) over a
            # corrupted loaded snapshot) must become a structured tool error, not
            # propagate out of run_v3 and lose the artifact — matching
            # executor.execute and the episode_* dispatchers. KeyboardInterrupt
            # still propagates so the R5 interrupt pairing runs.
            try:
                if name == "experiment_submit":
                    result_obj = dispatch_experiment_submit(
                        self.queue, self.state, self.workspace, self.executor, args, turn
                    )
                else:
                    result_obj = dispatch_experiment_collect(
                        self.queue, self.state, self.workspace, self.executor, args, turn
                    )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                result_obj = {"error": f"experiment tool {name} failed: {exc}"}
            if (
                name == "experiment_submit"
                and isinstance(result_obj, dict)
                and result_obj.get("experiment_id")
            ):
                menu = workflow_state(self.state, self.executor)
                if menu.get("state") == "repair_exhausted":
                    entry = self.state.repair_saturation.get(
                        str(menu.get("saturation_key") or "")
                    )
                    if entry is not None:
                        entry["pending_experiment_id"] = str(
                            result_obj["experiment_id"]
                        )
                        entry["updated_turn"] = turn
            return json.dumps(result_obj, ensure_ascii=False)
        # M3: the lead hosts the composites too (Part 2). state.readings is the
        # lead's readings map; execute_composite logs the ToolCall + returns the
        # dict, which we serialize for the tool_result. Composites cannot change
        # the model variant, so they early-return WITHOUT the mirror below
        # (consistent with the episode_run / episode_install early returns).
        if name in COMPOSITE_TOOL_NAMES:
            result_obj = execute_composite(
                name, tu.get("input") or {}, executor=self.executor,
                state_readings=self.state.readings, turn=turn, tool_use_id=tu["id"],
            )
            result = json.dumps(result_obj, ensure_ascii=False)
            if cache_key is not None:
                self._read_call_cache[cache_key] = result
            return result
        result = self.executor.execute(name, args, tool_use_id=tu["id"])
        # Mirror the executor's model-variant selection into state so episodes
        # inherit it and it serializes for resume (act_set_model_variant is the
        # only writer; mirroring unconditionally keeps them in lock-step).
        self.state.model_variant = self.executor._model_variant
        if cache_key is not None:
            self._read_call_cache[cache_key] = result
        return result


    def _dispatch_verify_run(self, args: dict[str, Any], turn: int) -> str:
        # F2: verify is a special episode kind. Render the candidate for the
        # named branch AT DISPATCH TIME with the pinned renderer (the exact
        # string BranchSnapshot.decryption / the benchmark score), compute the
        # sha256 here, and build a spec whose inputs are ONLY the candidate text
        # + language (branches=[] -> empty episode workspace). No scores reach
        # the verify prompt.
        #
        # M6 F6: a verify episode attests EXACTLY ONE branch — the attestation it
        # writes is keyed to that branch's decoded content. Silently picking the
        # first EXISTING name out of a multi-name list (the old
        # ``next(has_branch)`` behavior) mislabels the attestation when the lead
        # meant a different, mistyped, or not-yet-created branch (e.g.
        # ``["typo", "main"]`` would attest ``main``). Require arity 1 and
        # existence, both as structured errors.
        branches = list(args.get("branches") or [])
        if len(branches) != 1:
            return json.dumps(
                {"error": (
                    "verify requires exactly one branch in `branches`; got "
                    f"{len(branches)}: {branches}"
                )}
            )
        branch = branches[0]
        if not self.workspace.has_branch(branch):
            return json.dumps(
                {"error": f"verify branch {branch!r} does not exist"}
            )
        candidate = _decoded_text_for_panel(self.workspace, branch)
        content_hash = _candidate_content_hash(candidate)
        try:
            spec = EpisodeSpec(
                kind="verify", goal=str(args.get("goal") or ""),
                inputs={"candidate_text": candidate, "language": self.language},
            )
        except Exception as exc:  # noqa: BLE001 - bad spec -> structured error
            return json.dumps({"error": f"invalid verify episode: {exc}"})
        ep_provider = self._provider_for_model((self._episode_models or {}).get("verify"))
        result = run_episode(
            spec, self.state, provider=ep_provider, language=self.language,
            word_set=self._word_set, word_list=self._word_list, pattern_dict=self._pattern_dict,
            launching_turn=turn,
            on_event=self._episode_event_forwarder(turn),
            max_cost_usd=self.max_cost_usd, outer_cost_usd=self.committed_cost(),
        )
        self.episode_tool_calls.extend(result.tool_calls)
        self.episode_budget.extend(
            BudgetEntry.from_dict(b) for b in result.budget_entries
        )
        spend_usd = round(
            sum(b.get("cost_usd", 0.0) for b in result.budget_entries), 6
        )
        self._emit("episode_complete", {
            "episode_id": result.episode_id, "kind": result.kind,
            "status": result.status, "calls": result.tool_call_count,
            "spend_usd": spend_usd,
            "digest": _episode_result_digest(
                result.kind, result.status, result.failure_reason, result.result
            ),
        }, outer_iteration=turn)
        payload = {
            "episode_id": result.episode_id,
            "kind": result.kind,
            "status": result.status,
            "failure_reason": result.failure_reason,
            "result": result.result,
            "summary": result.summary,
            "branch": branch,
            "spend_usd": spend_usd,
        }
        # On success the DISPATCHER (not the lead model) writes the
        # AttestationRecord with the pre-computed hash (A1 — workers never write
        # state), mirroring the reading-compile precedent above.
        if result.status == "ok" and isinstance(result.result, dict):
            record = AttestationRecord(
                branch=branch,
                content_hash=content_hash,
                renderer_id=DECODED_TEXT_RENDERER_ID,
                episode_id=result.episode_id,
                coherence=_clamp_coherence(result.result.get("coherence")),
                reader_accepts=bool(result.result.get("reader_accepts")),
                gloss=str(result.result.get("gloss") or ""),
                anomalies=[str(a) for a in (result.result.get("anomalies") or [])],
                created_turn=turn,
                target_language_confidence=clamp_unit_interval(
                    result.result.get("target_language_confidence")
                ),
                semantic_recoverability=clamp_unit_interval(
                    result.result.get("semantic_recoverability")
                ),
                damage_scope=normalize_damage_scope(
                    result.result.get("damage_scope")
                ),
                repairability=normalize_repairability(
                    result.result.get("repairability")
                ),
                reader_accepts_as_solution=(
                    result.result.get("reader_accepts_as_solution") is True
                ),
                uncertainty_note=str(result.result.get("uncertainty_note") or ""),
            )
            record_dict = record.to_dict()
            self.state.verify_attestations.append(record_dict)
            # Slice 6: the agenda seeds from the verifier's REPAIRABILITY
            # verdict, not from coherence. Only a non-positive attestation
            # whose reader says targeted local fixes are worthwhile mints
            # open repair items; broaden/none verdicts route elsewhere
            # (context workflow) and must not queue local repair work.
            if (
                not attestation_is_positive(record_dict)
                and record.repairability == "local_repair"
            ):
                for anomaly in record.anomalies:
                    if any(
                        item.get("status", "open") == "open"
                        and item.get("source") == "verify_attestation"
                        and item.get("content_hash") == content_hash
                        and item.get("anomaly") == anomaly
                        for item in self.state.repair_agenda
                    ):
                        continue
                    numeric_ids = []
                    for existing_item in self.state.repair_agenda:
                        try:
                            numeric_ids.append(int(existing_item.get("id") or 0))
                        except (TypeError, ValueError):
                            continue
                    self.state.repair_agenda.append({
                        "id": max(numeric_ids, default=0) + 1,
                        "kind": "verify_anomaly",
                        "source": "verify_attestation",
                        "branch": branch,
                        "content_hash": content_hash,
                        "anomaly": anomaly,
                        "status": "open",
                        "created_turn": turn,
                        "episode_id": result.episode_id,
                        "damage_scope": record.damage_scope,
                        "repairability": record.repairability,
                    })
            payload["attestation"] = {
                "branch": branch,
                "coherence": record.coherence,
                "reader_accepts": record.reader_accepts,
                "reader_accepts_as_solution": record.reader_accepts_as_solution,
                "target_language_confidence": record.target_language_confidence,
                "semantic_recoverability": record.semantic_recoverability,
                "damage_scope": record.damage_scope,
                "repairability": record.repairability,
                "uncertainty_note": record.uncertainty_note,
                "anomalies": record.anomalies,
            }
        return json.dumps(payload, ensure_ascii=False)


    def _dispatch_episode_run(self, tu: dict[str, Any], turn: int) -> str:
        args = tu.get("input") or {}
        kind = str(args.get("kind") or "")
        if kind == "verify":
            return self._dispatch_verify_run(args, turn)
        inputs: dict[str, Any] = {
            "branches": list(args.get("branches") or []),
            "search_tool": args.get("search_tool"),
            "context_note": args.get("context_note"),
            "max_tool_calls": args.get("max_tool_calls"),
        }
        _reading_att_key = "none"
        if kind in {"reading", "repair"}:
            reading_branches = [
                name for name in inputs["branches"] if self.workspace.has_branch(name)
            ]
            if len(reading_branches) != 1:
                return json.dumps({
                    "error": f"{kind} requires exactly one existing branch"
                })
            if kind == "reading":
                from investigation.state import (
                    attestation_key, latest_attestation_for_hash, saturation_key,
                )
                reading_branch = reading_branches[0]
                content_hash = _branch_hash(self.workspace, reading_branch)
                _reading_att_key = attestation_key(latest_attestation_for_hash(
                    self.state.verify_attestations, content_hash
                ))
                sat_entry = self.state.repair_saturation.get(
                    saturation_key(content_hash, _reading_att_key)
                )
                if sat_entry is not None and int(sat_entry.get("readings") or 0) >= 1:
                    existing = max(
                        (
                            r for r in self.state.readings.values()
                            if r.get("candidate_content_hash") == content_hash
                        ),
                        key=lambda r: (
                            int(r.get("created_turn") or 0),
                            str(r.get("reading_id") or ""),
                        ),
                        default=None,
                    )
                    return json.dumps({
                        "status": "blocked",
                        "reason": "duplicate_reading_suppressed",
                        "branch": reading_branch,
                        "content_hash": content_hash,
                        "existing_reading_id": (
                            str(existing.get("reading_id")) if existing else None
                        ),
                        "note": (
                            "A reading already exists for this exact content "
                            "and verifier evidence. Reuse it via "
                            "repair_transaction, or produce new content or new "
                            "verifier evidence first."
                        ),
                    }, ensure_ascii=False)
            inputs["candidate_packet"] = build_candidate_reading_packet(
                self.workspace, reading_branches[0]
            ).to_dict()
        compare_branch_hashes = (
            {
                name: _branch_hash(self.workspace, name)
                for name in inputs["branches"]
                if self.workspace.has_branch(name)
            }
            if kind == "compare"
            else None
        )
        # A1/M3: hand a stored Reading to the episode by id (repair kind). The
        # reading DICT is injected as inputs["reading"]; unknown id → error.
        reading_id = args.get("reading_id")
        if reading_id is not None:
            stored = self.state.readings.get(str(reading_id))
            if stored is None:
                return json.dumps({"error": f"unknown reading_id: {reading_id!r}"})
            inputs["reading"] = stored
        try:
            spec = EpisodeSpec(kind=kind, goal=str(args.get("goal") or ""), inputs=inputs)
        except Exception as exc:  # noqa: BLE001 - bad spec → structured error
            payload: dict[str, Any] = {
                "error": f"invalid episode_run: {exc}",
            }
            if kind == "search":
                from investigation.episodes import SEARCH_EPISODE_TOOL_NAMES
                payload.update({
                    "valid_search_tools": list(SEARCH_EPISODE_TOOL_NAMES),
                    "episode_corrected_example": {
                        "kind": "search",
                        "goal": str(args.get("goal") or "Search this branch."),
                        "branches": list(args.get("branches") or ["main"]),
                        "search_tool": "search_anneal",
                    },
                    "automated_solver_example": {
                        "tool": "experiment_submit",
                        "input": {
                            "type": "automated_solver",
                            "branch": str(next(iter(args.get("branches") or []), "main")),
                            "config": {},
                        },
                    },
                })
            return json.dumps(payload)
        ep_provider = self._provider_for_model((self._episode_models or {}).get(kind))
        result = run_episode(
            spec, self.state, provider=ep_provider, language=self.language,
            word_set=self._word_set, word_list=self._word_list, pattern_dict=self._pattern_dict,
            launching_turn=turn,
            on_event=self._episode_event_forwarder(turn),
            max_cost_usd=self.max_cost_usd, outer_cost_usd=self.committed_cost(),
        )
        self.episode_tool_calls.extend(result.tool_calls)
        self.episode_budget.extend(
            BudgetEntry.from_dict(b) for b in result.budget_entries
        )
        spend_usd = round(
            sum(b.get("cost_usd", 0.0) for b in result.budget_entries), 6
        )
        self._emit("episode_complete", {
            "episode_id": result.episode_id, "kind": result.kind,
            "status": result.status, "calls": result.tool_call_count,
            "spend_usd": spend_usd,
            "digest": _episode_result_digest(
                result.kind, result.status, result.failure_reason, result.result
            ),
        }, outer_iteration=turn)
        payload = {
            "episode_id": result.episode_id,
            "kind": result.kind,
            "status": result.status,
            "failure_reason": result.failure_reason,
            "result": result.result,
            "summary": result.summary,
            "snapshots": [s.get("name") for s in result.branch_snapshots],
            "spend_usd": spend_usd,
        }
        if kind == "compare" and isinstance(result.result, dict):
            winner = result.result.get("winner")
            binding = {
                "branch_hashes": compare_branch_hashes or {},
                "winner": winner,
                "winner_hash": (
                    (compare_branch_hashes or {}).get(str(winner))
                    if winner is not None else None
                ),
                "created_turn": turn,
                "episode_id": result.episode_id,
            }
            ledger_entry = next(
                (entry for entry in reversed(self.state.episode_ledger)
                 if entry.get("episode_id") == result.episode_id),
                None,
            )
            if ledger_entry is not None:
                ledger_entry["comparison_binding"] = binding
            payload["comparison_binding"] = binding
        ledger_entry = next(
            (entry for entry in reversed(self.state.episode_ledger)
             if entry.get("episode_id") == result.episode_id),
            None,
        )
        if ledger_entry is not None:
            ledger_entry["launching_turn"] = turn
            ledger_entry["input_branches"] = list(inputs.get("branches") or [])
        # Part 1: the lead compiles a reading-kind result into a stored Reading
        # (A1 — workers never write state.readings) and returns its id.
        if result.kind == "reading" and result.status == "ok" and isinstance(result.result, dict):
            branch = next(iter(spec.inputs.get("branches") or []), "main")
            reading = Reading.from_episode_result(
                result.result,
                branch=str(branch),
                source=f"episode:{result.episode_id}",
                created_turn=turn,
                candidate_packet=inputs.get("candidate_packet"),
            )
            self.state.readings[reading.reading_id] = reading.to_dict()
            payload["reading_id"] = reading.reading_id
            from investigation.state import get_or_create_saturation_entry
            packet_hash = str(
                (inputs.get("candidate_packet") or {}).get("content_hash") or ""
            )
            if packet_hash:
                sat_entry = get_or_create_saturation_entry(
                    self.state, packet_hash, _reading_att_key, turn
                )
                sat_entry["readings"] = int(sat_entry.get("readings") or 0) + 1
                sat_entry["updated_turn"] = turn
        self.state.add_evidence(
            f"episode:{result.kind}",
            turn=turn,
            summary=result.summary or f"{result.kind} episode {result.status}",
            episode_id=result.episode_id,
            status=result.status,
            input_branches=list(inputs.get("branches") or []),
            result=(dict(result.result) if isinstance(result.result, dict) else result.result),
        )
        if result.status == "ok" and isinstance(result.result, dict):
            input_branch = next(
                (
                    name for name in inputs.get("branches") or []
                    if self.workspace.has_branch(name)
                ),
                None,
            )
            if input_branch and result.kind == "survey":
                modes = result.result.get("suspected_modes") or []
                top = modes[0] if modes and isinstance(modes[0], dict) else None
                if top and top.get("mode"):
                    self.state.hypothesis_board.update(
                        self.workspace,
                        input_branch,
                        cipher_mode=str(top["mode"]),
                        mode_status="active",
                        mode_confidence=str(top.get("confidence") or "unknown"),
                        mode_evidence=str(top.get("evidence") or result.summary or ""),
                        evidence_source=f"episode:{result.episode_id}",
                        next_recommended_action=(
                            str((result.result.get("recommended_next") or [""])[0]) or None
                        ),
                    )
        return json.dumps(payload, ensure_ascii=False)


    def _dispatch_episode_install(self, tu: dict[str, Any], turn: int) -> str:
        args = tu.get("input") or {}
        ep_id = str(args.get("episode_id") or "")
        branch = str(args.get("branch") or "")
        as_name = args.get("as_name")
        entry = next(
            (e for e in reversed(self.state.episode_ledger) if e.get("episode_id") == ep_id),
            None,
        )
        if entry is None:
            return json.dumps({"error": f"unknown episode_id: {ep_id}"})
        snap = next(
            (s for s in (entry.get("branch_snapshots") or []) if s.get("name") == branch),
            None,
        )
        if snap is None:
            return json.dumps({"error": f"episode {ep_id} has no branch {branch!r}"})
        # Deep-copy the ledger snapshot before restore: restore_branch shallow-
        # copies metadata/pipeline, so without this the installed live branch
        # would alias nested dicts inside state.episode_ledger (the mirror-image
        # of the F5 episode-side aliasing).
        snap = copy.deepcopy(snap)
        # F4 (spec-author amendment): strip the board-mirrored card keys from the
        # restored metadata, then route their VALUES through board.update so the
        # board re-mirrors them into metadata — single-writer preserved AND the
        # episode branch's mode/status/evidence survive the install.
        raw_metadata = snap.get("metadata") or {}
        card_fields = {
            k: v for k, v in raw_metadata.items() if k in CARD_MIRROR_KEYS
        }
        metadata = {
            k: v for k, v in raw_metadata.items() if k not in CARD_MIRROR_KEYS
        }
        target = str(as_name) if as_name else f"{entry.get('kind')}_{ep_id[:6]}_{branch}"
        snapshot_hash = self._snapshot_content_hash(snap)
        existing = next(
            (
                name for name in self.workspace.branch_names()
                if _branch_hash(self.workspace, name) == snapshot_hash
            ),
            None,
        )
        if existing is not None:
            alias = {
                "requested_name": target,
                "existing_branch": existing,
                "content_hash": snapshot_hash,
                "from_episode": ep_id,
                "from_branch": branch,
                "created_turn": turn,
            }
            self.state.branch_aliases.append(alias)
            for item in entry.get("agenda_additions") or []:
                if str(item.get("branch") or "") != branch:
                    continue
                merged = dict(item)
                merged["branch"] = existing
                self.state.repair_agenda.append(merged)
            return json.dumps({
                "status": "deduplicated",
                "installed": existing,
                "alias": alias,
            }, ensure_ascii=False)
        name = target
        suffix = 2
        while self.workspace.has_branch(name):
            name = f"{target}_{suffix}"
            suffix += 1
        install_snap = dict(snap)
        install_snap["name"] = name
        install_snap["metadata"] = metadata
        install_snap["created_iteration"] = turn
        from investigation.state import _restore_branch_into
        _restore_branch_into(self.workspace, install_snap)
        if card_fields:
            self.state.hypothesis_board.update(self.workspace, name, **card_fields)
        else:
            source_branch = next(iter(entry.get("input_branches") or []), None)
            source_card = (
                self.state.hypothesis_board.get(str(source_branch))
                if source_branch is not None else None
            )
            if source_card is not None:
                inherited = {
                    key: source_card.get(key)
                    for key in CARD_MIRROR_KEYS
                    if source_card.get(key) is not None
                }
                inherited["mode_status"] = "active"
                inherited["evidence_source"] = f"episode:{ep_id}"
                if entry.get("summary"):
                    inherited["mode_evidence"] = str(entry["summary"])
                self.state.hypothesis_board.update(self.workspace, name, **inherited)
        # M6 F5: this dispatcher owns BOTH the collision-rename above AND the
        # verify-attestation writes; keep attestation branch labels in sync when a
        # rename moved attested content to a new name (see the helper's docstring).
        _resync_attestation_branch_on_rename(
            self.workspace, self.state.verify_attestations, target, name
        )
        # Episode-local repair-agenda additions ride in the packet (A10): merge
        # ONLY the residuals targeting the branch being installed (F5), remapped
        # to the installed name. A blanket re-append + relabel would duplicate an
        # episode's residuals across every install and mislabel residuals that
        # belong to a different fork the same episode produced.
        for item in entry.get("agenda_additions") or []:
            if str(item.get("branch") or "") != branch:
                continue
            merged = dict(item)
            merged["branch"] = name
            self.state.repair_agenda.append(merged)
        return json.dumps({
            "status": "ok",
            "installed": name,
            "from_episode": ep_id,
            "from_branch": branch,
        }, ensure_ascii=False)


    def _snapshot_content_hash(self, snapshot: dict[str, Any]) -> str:
        """Render one isolated episode snapshot with the canonical renderer."""
        from investigation.state import _restore_branch_into

        scratch = Workspace(
            cipher_text=self.workspace.cipher_text,
            plaintext_alphabet=self.workspace.plaintext_alphabet,
        )
        snap = copy.deepcopy(snapshot)
        _restore_branch_into(scratch, snap)
        return _branch_hash(scratch, str(snap["name"]))


    def _settle_repair_outcome(self, *, record: dict[str, Any], entry_args: tuple[str, str, str], changed_hashes: list[str], turn: int) -> dict[str, Any]:
        """Update the saturation entry for one FINISHED transaction record
        (failed or installed), stamp failure_class/counted_evidence_failure on
        failures, append the record to state.repair_transactions, and attach a
        compact ``saturation`` summary to the returned payload."""
        from investigation.state import get_or_create_saturation_entry

        source_hash, att_key, pair = entry_args
        entry = get_or_create_saturation_entry(self.state, source_hash, att_key, turn)
        entry["updated_turn"] = turn
        if changed_hashes:
            entry["finalist_hashes"] = sorted(
                set(entry.get("finalist_hashes") or []) | set(changed_hashes)
            )
        if record.get("status") == "failed":
            failure_class = _classify_failure_reason(record.get("reason"))
            record["failure_class"] = failure_class
            counted = False
            if failure_class == "process":
                prior = int((entry.get("process_failures") or {}).get(pair, 0))
                entry.setdefault("process_failures", {})[pair] = prior + 1
                if prior >= 1:
                    # Second process failure for the pair counts as evidence.
                    counted = True
            else:
                counted = True
            if counted:
                entry["evidence_failures"] = int(entry.get("evidence_failures") or 0) + 1
                failed_pairs = set(entry.get("evidence_failed_pairs") or [])
                failed_pairs.add(pair)
                entry["evidence_failed_pairs"] = sorted(failed_pairs)
                if entry["evidence_failures"] >= 2:
                    entry["exhausted"] = True
            record["counted_evidence_failure"] = counted
        self.state.repair_transactions.append(record)
        record_with_summary = {
            **record,
            "saturation": {
                "evidence_failures": int(entry.get("evidence_failures") or 0),
                "remaining_before_exhausted": max(
                    0, 2 - int(entry.get("evidence_failures") or 0)
                ),
                "exhausted": bool(entry.get("exhausted")),
            },
        }
        return record_with_summary


    def _probe_snapshot_scores(self, snapshot: dict[str, Any], transaction_id: str) -> tuple[str, dict[str, float | None]]:
        """Restore the winner snapshot into the live workspace under a
        reserved probe name, hash + quick-score it with the SAME renderer and
        scoring the branch cards use, and always delete the probe."""
        from investigation.state import _restore_branch_into

        probe_name = f"__repair_probe_{transaction_id}"
        snap = copy.deepcopy(snapshot)
        snap["name"] = probe_name
        try:
            _restore_branch_into(self.workspace, snap)
            probe_hash = _branch_hash(self.workspace, probe_name)
            scores = self.executor._compute_quick_scores(probe_name)
        finally:
            if self.workspace.has_branch(probe_name):
                self.workspace.delete(probe_name)
        return probe_hash, scores


    def check_repair_preconditions(
        self, *, branch: str, reading_id_arg: str, turn: int
    ) -> dict[str, Any]:
        """Bounded-repair precondition checks (Part 8.1 extraction).

        Returns ``{"blocked": <payload>}`` for any early-return gate (payloads
        byte-identical to the pre-split dispatcher) or ``{"ok": True, ...}`` with
        the resolved bindings the caller needs to run and install the repair.
        Zero behavior change: this is code motion out of
        ``_dispatch_repair_transaction``.
        """
        if not self.workspace.has_branch(branch):
            return {"blocked": {"status": "failed", "reason": "unknown_branch", "branch": branch}}
        source_hash = _branch_hash(self.workspace, branch)
        reading_id = reading_id_arg
        reading_data = self.state.readings.get(reading_id) if reading_id else None
        if reading_data is None and not reading_id:
            candidates = [
                item for item in self.state.readings.values()
                if str(item.get("branch") or "") == branch
                and item.get("candidate_content_hash") == source_hash
            ]
            if candidates:
                reading_data = max(
                    candidates, key=lambda item: int(item.get("created_turn") or 0)
                )
                reading_id = str(reading_data.get("reading_id") or "")
        if reading_data is None:
            return {"blocked": {
                "status": "failed",
                "reason": "fresh_reading_required",
                "branch": branch,
                "content_hash": source_hash,
            }}
        reading = Reading.from_dict(reading_data)
        if reading.branch != branch:
            return {"blocked": {
                "status": "failed", "reason": "reading_branch_mismatch",
                "branch": branch, "reading_branch": reading.branch,
            }}
        if reading.candidate_content_hash != source_hash:
            return {"blocked": {
                "status": "failed", "reason": "stale_or_unbound_reading",
                "branch": branch,
                "source_content_hash": source_hash,
                "reading_content_hash": reading.candidate_content_hash,
            }}
        from investigation.reading import interpretation_digest as _interp_digest
        from investigation.state import (
            attestation_key, get_or_create_saturation_entry,
            latest_attestation_for_hash, pair_digest, saturation_key,
        )

        interp_digest = _interp_digest(reading_data)
        duplicate = next(
            (
                item for item in reversed(self.state.repair_transactions)
                if item.get("status") == "installed"
                and item.get("source_content_hash") == source_hash
                and (
                    item.get("interpretation_id", item.get("reading_id")) == reading_id
                    or (
                        item.get("interpretation_digest")
                        and item.get("interpretation_digest") == interp_digest
                    )
                )
            ),
            None,
        )
        if duplicate is not None:
            return {"blocked": {
                "status": "duplicate_suppressed",
                "reason": "source_and_reading_already_handled",
                "transaction_id": duplicate.get("transaction_id"),
                "installed": duplicate.get("installed_branch"),
            }}

        latest_attestation = latest_attestation_for_hash(
            self.state.verify_attestations, source_hash
        )
        att_key = attestation_key(latest_attestation)
        sat_key = saturation_key(source_hash, att_key)
        pair = pair_digest(source_hash, interp_digest)
        entry = self.state.repair_saturation.get(sat_key)

        if entry is not None and entry.get("exhausted"):
            return {"blocked": {
                "status": "blocked",
                "reason": "repair_saturated",
                "branch": branch,
                "saturation_key": sat_key,
                "evidence_failures": int(entry.get("evidence_failures") or 0),
                "note": (
                    "Repair is exhausted for this candidate content and "
                    "verifier evidence. Run one alternate search/basin "
                    "experiment, compare genuinely distinct finalists, or "
                    "declare honestly unsolved."
                ),
            }}
        if entry is not None and pair in (entry.get("evidence_failed_pairs") or []):
            return {"blocked": {
                "status": "blocked",
                "reason": "pair_evidence_failed",
                "branch": branch,
                "pair_digest": pair,
                "note": (
                    "This source/interpretation pair was already evidence-"
                    "evaluated and failed; it cannot be rerun under a new "
                    "name. Provide genuinely new content or evidence."
                ),
            }}
        retry_of = None
        if entry is not None and int((entry.get("process_failures") or {}).get(pair, 0)) >= 1:
            retry_of = next(
                (
                    str(item.get("transaction_id") or "")
                    for item in reversed(self.state.repair_transactions)
                    if item.get("pair_digest") == pair
                    and item.get("status") == "failed"
                    and item.get("failure_class") == "process"
                ),
                None,
            )
        anomalies = [
            str(item) for item in (latest_attestation or {}).get("anomalies") or []
        ]
        note = (
            "Address these independent-reader anomalies conservatively: "
            + "; ".join(anomalies)
            if anomalies else
            "Use only the stored reading's supported fragments; avoid speculative edits."
        )
        return {
            "ok": True, "source_hash": source_hash, "reading_id": reading_id,
            "interp_digest": interp_digest, "att_key": att_key, "sat_key": sat_key,
            "pair": pair, "retry_of": retry_of, "anomalies": anomalies, "note": note,
        }


    def validate_and_install_repair(
        self, *, tu: dict[str, Any], turn: int, branch: str, source_hash: str,
        att_key: str, pair: str, base_record: dict[str, Any],
        episode_payload: dict[str, Any], as_name: str,
    ) -> str:
        """Validate the repair worker's forks against the eight acceptance
        checks and install the accepted winner (Part 8.1 extraction; verbatim
        code motion of the post-episode dispatcher body with three parameter
        substitutions)."""
        episode_id = str(episode_payload.get("episode_id") or "")
        ledger_entry = next(
            (
                entry for entry in reversed(self.state.episode_ledger)
                if entry.get("episode_id") == episode_id
            ),
            None,
        )
        result = episode_payload.get("result") or {}
        snapshots = list((ledger_entry or {}).get("branch_snapshots") or [])
        changed: dict[str, str] = {}
        for snapshot in snapshots:
            name = str(snapshot.get("name") or "")
            if not name:
                continue
            digest = self._snapshot_content_hash(snapshot)
            if digest != source_hash:
                changed[name] = digest
        requested = str(result.get("best_branch") or "")
        snapshot_names = {
            str(snapshot.get("name") or "")
            for snapshot in snapshots
            if snapshot.get("name")
        }
        rejected_targets = _worker_rejected_targets(result)
        episode_calls = [
            tc for tc in self.episode_tool_calls
            if getattr(tc, "episode_id", None) == episode_id
        ]
        evidence = _extract_repair_evidence(episode_calls)

        acceptance_checks: list[dict[str, Any]] = []
        before = self.executor._compute_quick_scores(branch)
        after: dict[str, Any] | None = None
        score_deltas: dict[str, Any] | None = None
        adjudicated_flag: bool | None = None if len(changed) <= 1 else False

        def _acceptance() -> dict[str, Any]:
            return {
                "policy": "default_deny_v1",
                "checks": acceptance_checks,
                "supported_forks": sorted(evidence["supported_forks"]),
                "edit_evidence_count": len(evidence["edit_evidence"]),
                "adjudicated": adjudicated_flag,
                "scores_before": before,
                "scores_after": after,
                "score_deltas": score_deltas,
            }

        def _fail(reason: str) -> str:
            record = {
                **base_record, "status": "failed", "reason": reason,
                "claimed_winner": requested or None,
                "changed_finalists": sorted(changed),
                "finalist_hashes": sorted(changed.values()),
                "acceptance": _acceptance(),
            }
            payload = self._settle_repair_outcome(
                record=record, entry_args=(source_hash, att_key, pair),
                changed_hashes=sorted(changed.values()), turn=turn,
            )
            return self._record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn, payload=payload
            )

        # Check 1 — winner_named: resolve the authoritative winner.
        winner = ""
        if requested:
            if requested not in snapshot_names:
                acceptance_checks.append({
                    "check": "winner_named", "passed": False,
                    "requested": requested, "detail": "not_a_snapshot",
                })
                return _fail("unsupported_winner")
            if requested not in changed:
                acceptance_checks.append({
                    "check": "winner_named", "passed": False,
                    "requested": requested, "detail": "named_but_unchanged",
                })
                return _fail("no_op")
            winner = requested
            acceptance_checks.append({
                "check": "winner_named", "passed": True, "winner": winner,
            })
        else:
            if len(changed) == 0:
                acceptance_checks.append({
                    "check": "winner_named", "passed": False,
                    "detail": "no_changed_finalists",
                })
                return _fail("no_changed_finalists")
            if all(name in rejected_targets for name in changed):
                acceptance_checks.append({
                    "check": "winner_named", "passed": False,
                    "detail": "all_finalists_rejected",
                })
                return _fail("all_finalists_rejected")
            if len(changed) == 1:
                winner = next(iter(changed))
                acceptance_checks.append({
                    "check": "winner_named", "passed": True, "winner": winner,
                })
            else:
                acceptance_checks.append({
                    "check": "winner_named", "passed": False,
                    "detail": "no_winner_named", "changed": sorted(changed),
                })
                return _fail("no_winner_named_with_multiple_changed_finalists")

        # Check 2 — worker_applied.
        applied = bool(result.get("applied"))
        acceptance_checks.append({"check": "worker_applied", "passed": applied})
        if not applied:
            return _fail("worker_did_not_apply")

        # Check 3 — winner_fork_evidence.
        fork_ok = (
            winner in evidence["supported_forks"]
            and winner not in evidence["failed_result_forks"]
        )
        acceptance_checks.append({"check": "winner_fork_evidence", "passed": fork_ok})
        if not fork_ok:
            return _fail("winner_fork_from_failed_call")

        # Check 4 — edit_claims_bound.
        claimed_edits = [str(e).strip() for e in result.get("edits") or []]
        unbound, normalized_claims = _unbound_edit_claims(
            claimed_edits, evidence["edit_evidence"]
        )
        acceptance_checks.append({
            "check": "edit_claims_bound", "passed": not unbound,
            "claimed": len(claimed_edits), "unbound": unbound,
            "normalized_claims": normalized_claims,
        })
        if unbound:
            return _fail("unsupported_edit_claim")

        # Check 5 — winner_adjudicated (only material with >=2 changed finalists).
        if len(changed) >= 2:
            adjudicated_flag = any(
                winner in s and any(m in s for m in changed if m != winner)
                for s in evidence["adjudicated_sets"]
            ) or winner in evidence["batch_finalist_forks"]
            acceptance_checks.append({
                "check": "winner_adjudicated", "passed": adjudicated_flag,
            })
            if not adjudicated_flag:
                return _fail("no_winner_named_with_multiple_changed_finalists")
        else:
            acceptance_checks.append({
                "check": "winner_adjudicated", "passed": True, "trivial": True,
            })

        # Check 6 — collateral_within_limits (appended only when it can compare).
        adj_summary = _winner_adjudication_summary(episode_calls, winner) or {}
        damaged = adj_summary.get("damaged_occurrences")
        improved = adj_summary.get("improved_occurrences")
        if isinstance(damaged, (int, float)) and isinstance(improved, (int, float)):
            collateral_ok = damaged <= improved
            acceptance_checks.append({
                "check": "collateral_within_limits", "passed": collateral_ok,
                "damaged_occurrences": damaged, "improved_occurrences": improved,
            })
            if not collateral_ok:
                return _fail("materially_non_improving")

        # Checks 7 + 8 — one probe restore serves both.
        winner_snapshot = next(
            s for s in snapshots if str(s.get("name") or "") == winner
        )
        probe_hash, after = self._probe_snapshot_scores(winner_snapshot, base_record["transaction_id"])
        dict_before, dict_after = before.get("dict_rate"), after.get("dict_rate")
        quad_before, quad_after = before.get("quad"), after.get("quad")
        dict_delta = (
            dict_after - dict_before
            if isinstance(dict_before, (int, float)) and isinstance(dict_after, (int, float))
            else None
        )
        quad_delta = (
            quad_after - quad_before
            if isinstance(quad_before, (int, float)) and isinstance(quad_after, (int, float))
            else None
        )
        score_deltas = {"dict_rate_delta": dict_delta, "quad_delta": quad_delta}

        # Check 7 — no_op_probe (defensive re-check with the live renderer).
        no_op_ok = probe_hash != source_hash
        acceptance_checks.append({
            "check": "no_op_probe", "passed": no_op_ok, "probe_hash": probe_hash,
        })
        if not no_op_ok:
            return _fail("no_op")

        # Check 8 — scalar_non_decrease (default deny on any measured decrease).
        decreased = (
            (dict_delta is not None and dict_delta < 0)
            or (quad_delta is not None and quad_delta < 0)
        )
        # Default deny: any measured scalar decrease rejects. REPAIR_ACCEPTANCE_POLICY
        # is the M5.4 hook; no allow-policy branch is implemented yet, so the guard
        # asserts the invariant rather than carrying a dead alternative.
        assert REPAIR_ACCEPTANCE_POLICY is None
        scalar_ok = not decreased
        acceptance_checks.append({
            "check": "scalar_non_decrease", "passed": scalar_ok,
            "deltas": dict(score_deltas),
        })
        if not scalar_ok:
            return _fail("materially_non_improving")

        acceptance = _acceptance()

        install_payload = json.loads(self._dispatch_episode_install({
            "id": f"{tu.get('id')}:install",
            "name": "episode_install_branch",
            "input": {
                "episode_id": episode_id,
                "branch": winner,
                "as_name": as_name,
            },
        }, turn))
        installed = str(install_payload.get("installed") or "")
        if install_payload.get("status") not in {"ok", "deduplicated"} or not installed:
            record = {
                **base_record, "status": "failed", "reason": "install_failed",
                "install": install_payload,
                "acceptance": acceptance,
            }
            payload = self._settle_repair_outcome(
                record=record, entry_args=(source_hash, att_key, pair),
                changed_hashes=sorted(changed.values()), turn=turn,
            )
            return self._record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn, payload=payload
            )
        result_hash = _branch_hash(self.workspace, installed)
        self.workspace.get_branch(installed).metadata["repair_transaction"] = {
            "transaction_id": base_record["transaction_id"],
            "source_branch": branch,
            "source_content_hash": source_hash,
            "reading_id": base_record["reading_id"],
            "episode_id": episode_id,
            "addressed_anomalies": base_record["addressed_anomalies"],
        }
        record = {
            **base_record,
            "status": "installed",
            "worker_winner": winner,
            "installed_branch": installed,
            "result_content_hash": result_hash,
            "changed": result_hash != source_hash,
            "reverification_required": True,
            "edits": [str(item) for item in result.get("edits") or []],
            "collateral": dict(result.get("collateral") or {}),
            "acceptance": acceptance,
        }
        payload = self._settle_repair_outcome(
            record=record, entry_args=(source_hash, att_key, pair),
            changed_hashes=sorted(changed.values()), turn=turn,
        )
        for item in self.state.repair_agenda:
            if (
                item.get("status", "open") == "open"
                and item.get("source") == "verify_attestation"
                and item.get("content_hash") == source_hash
            ):
                item["status"] = "addressed"
                item["addressed_by_transaction"] = base_record["transaction_id"]
                item["addressed_turn"] = turn
        self.state.add_evidence(
            "repair_transaction", turn=turn,
            summary=f"Installed {installed} from bounded repair transaction.",
            **record,
        )
        self._emit("repair_transaction_complete", dict(record), outer_iteration=turn)
        return self._record_dispatch_result(
            name="repair_transaction", tu=tu, turn=turn, payload=payload
        )


    def _dispatch_repair_transaction(self, tu: dict[str, Any], turn: int) -> str:
        """Run, validate, install, and record one bounded repair operation."""
        args = tu.get("input") or {}
        branch = str(args.get("branch") or "")
        pre = self.check_repair_preconditions(
            branch=branch, reading_id_arg=str(args.get("reading_id") or ""), turn=turn
        )
        if "blocked" in pre:
            return self._record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn, payload=pre["blocked"],
            )
        episode_payload = json.loads(self._dispatch_episode_run({
            "id": f"{tu.get('id')}:repair",
            "name": "episode_run",
            "input": {
                "kind": "repair",
                "goal": str(args.get("goal") or "Repair the bound candidate conservatively."),
                "branches": [branch],
                "reading_id": pre["reading_id"],
                "context_note": pre["note"],
            },
        }, turn))
        transaction_id = uuid.uuid4().hex[:12]
        base_record = {
            "transaction_id": transaction_id,
            "source_branch": branch,
            "source_content_hash": pre["source_hash"],
            "reading_id": pre["reading_id"],                 # operational pointer into state.readings
            "interpretation_id": pre["reading_id"],          # B2 identity component (== reading_id in M5.3)
            "interpretation_digest": pre["interp_digest"],   # B2 identity component
            "attestation_key": pre["att_key"],
            "saturation_key": pre["sat_key"],
            "pair_digest": pre["pair"],
            "retry_of": pre["retry_of"],
            "episode_id": episode_payload.get("episode_id"),
            "addressed_anomalies": pre["anomalies"],
            "created_turn": turn,
        }
        if episode_payload.get("status") != "ok":
            record = {
                **base_record, "status": "failed",
                "reason": episode_payload.get("failure_reason") or episode_payload.get("error"),
            }
            settled = self._settle_repair_outcome(
                record=record, entry_args=(pre["source_hash"], pre["att_key"], pre["pair"]),
                changed_hashes=[], turn=turn,
            )
            return self._record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn,
                payload={**settled, "episode": episode_payload},
            )
        return self.validate_and_install_repair(
            tu=tu, turn=turn, branch=branch, source_hash=pre["source_hash"],
            att_key=pre["att_key"], pair=pre["pair"], base_record=base_record,
            episode_payload=episode_payload,
            as_name=str(args.get("as_name") or f"repair_tx_{turn}_{branch}"),
        )


    def _record_dispatch_result(self, *, name: str, tu: dict[str, Any], turn: int, payload: dict[str, Any]) -> str:
        rendered = json.dumps(payload, ensure_ascii=False)
        self.executor.call_log.append(ToolCall(
            iteration=turn,
            tool_name=name,
            tool_use_id=str(tu.get("id") or ""),
            arguments=dict(tu.get("input") or {}),
            result=rendered,
            elapsed_ms=0,
        ))
        return rendered


    def _provider_for_model(self, model_id: str | None) -> Any:
        """Return a provider for an episode model id (same client, swapped model)."""
        if self._model_provider is None or not model_id:
            return self._model_provider
        if getattr(self._model_provider, "model", None) == model_id:
            return self._model_provider
        clone = copy.copy(self._model_provider)
        clone.model = model_id
        return clone


    def _lead_read_cache_key(self, name: str, args: dict[str, Any]) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        normalized = json.dumps(args or {}, sort_keys=True, default=str)
        branch_names: list[str] = []
        if isinstance(args.get("branch"), str):
            branch_names.append(str(args["branch"]))
        for branch_name in args.get("branches") or []:
            if isinstance(branch_name, str):
                branch_names.append(branch_name)
        if name == "decode_show" and not branch_names:
            branch_names.append("main")
        hashes = tuple(
            sorted(
                (branch_name, _branch_hash(self.workspace, branch_name))
                for branch_name in set(branch_names)
                if self.workspace.has_branch(branch_name)
            )
        )
        return name, normalized, hashes


    def information_digest(self) -> str:
        payload = {
            "branch_hashes": sorted(
                {_branch_hash(self.workspace, name) for name in self.workspace.branch_names()}
            ),
            "readings": sorted(self.state.readings),
            "attestations": sorted(
                (
                    str(item.get("content_hash") or ""),
                    int(item.get("coherence") or 0),
                    bool(item.get("reader_accepts")),
                    bool(item.get("reader_accepts_as_solution")),
                    f"{float(item.get('target_language_confidence') or 0.0):.4f}",
                    f"{float(item.get('semantic_recoverability') or 0.0):.4f}",
                    str(item.get("damage_scope") or ""),
                    str(item.get("repairability") or ""),
                    tuple(str(a) for a in item.get("anomalies") or []),
                )
                for item in self.state.verify_attestations
            ),
            "episode_results": sorted(
                json.dumps(item.get("result"), sort_keys=True, default=str)
                for item in self.state.episode_ledger
                if item.get("status") == "ok"
            ),
            "experiment_results": sorted(
                str(item.get("dedup_key") or "")
                for item in self.state.experiment_queue
                if item.get("status") == "completed"
            ),
            "repair_results": sorted(
                str(item.get("result_content_hash") or "")
                for item in self.state.repair_transactions
                if item.get("status") == "installed"
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


    def _episode_event_forwarder(self, turn: int) -> Any:
        """F6: forward an episode's internal progress events into the lead
        event stream (nested context preserved via ``outer_iteration``)."""
        def _forward(event: str, payload: dict) -> None:
            self._emit(event, payload, outer_iteration=turn)
        return _forward


    def sync_budget(self) -> None:
        self.state.budget_ledger = (
            self._prior_budget + self.episode_budget + list(self._session.usage_entries())
        )

    def committed_cost(self) -> float:
        # M5.3 Slice 1: cost of every COMPLETED send (lead + prior + finished
        # episodes), computed straight from the live ledgers so it is correct
        # regardless of sync_budget() timing. It excludes any in-flight episode,
        # whose own session spend the episode's cost guard adds on top; this is
        # therefore the correct ``outer_cost_usd`` base to hand an episode.
        return sum(
            entry.cost()
            for entry in (self._prior_budget + self.episode_budget + list(self._session.usage_entries()))
        )


    def cost_ceiling_reached(self) -> bool:
        # The lead-side per-run paid ceiling: refuse the next paid lead send.
        return self.max_cost_usd is not None and self.committed_cost() >= self.max_cost_usd


    def set_available_tools(self, names: set[str]) -> None:
        self._available_tools = set(names)
