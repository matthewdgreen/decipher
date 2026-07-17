"""run_v3: the v3 lead loop (M1 spec Part 4).

A lead loop over context REBUILT from ``InvestigationState`` each turn, using
the existing tool executor directly (no episodes — that is M2), talking to
models through the ``ModelSession`` seam. The declaration policy is disabled
(``NoGatesPolicy``): declaration is always allowed, confidence recorded.

Declaration tools terminate the run. Exhaustion preserves a best-effort branch
for inspection but is honestly unsolved unless a fresh positive independent
attestation supports a synthesized fallback declaration.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from typing import Any

from agent.loop_shared import (
    DECODED_TEXT_RENDERER_ID,
    _best_branch_for_auto_declare,
    _branch_snapshot_for,
    _candidate_content_hash,
    _decoded_text_for_panel,
    _hypothesis_cards_for_artifact,
    _install_automated_preflight_branch,
    _tool_result_summary,
    _workspace_snapshot_payload,
)
from agent.model_provider import (
    ModelProviderError,
    ensure_model_provider,
    _collect_assistant_blocks,
)
from agent.tools_v2 import AttestationPolicy, WorkspaceToolExecutor
from investigation.board import CARD_MIRROR_KEYS
from investigation.actions import COMPOSITE_TOOL_NAMES, execute_composite
from investigation.episodes import (
    EPISODE_KINDS,
    v3_lead_tool_definitions,
    EpisodeSpec,
    run_episode,
)
from investigation.experiments import (
    EXPERIMENT_TOOL_DEFINITIONS,
    EXPERIMENT_TOOL_NAMES,
    ExperimentQueue,
    dispatch_experiment_collect,
    dispatch_experiment_submit,
)
from investigation.reading import Reading, build_candidate_reading_packet
from analysis import cipher_id as cipher_id_analysis
from analysis import dictionary, pattern
from artifact.schema import LoopEvent, RunArtifact, SolutionDeclaration, ToolCall
from investigation.context import (
    allowed_episode_kinds,
    build_lead_context,
    build_v3_system_prompt,
    workflow_state,
    workflow_hint_candidates,
    DECLARE_COHERENCE,
)
from investigation.sessions import ModelSession, session_factory
from investigation.state import AttestationRecord, BudgetEntry, InvestigationState
from models.cipher_text import CipherText
from workspace import Workspace


def _is_positive_attestation(attestation: dict[str, Any]) -> bool:
    return bool(attestation.get("reader_accepts")) and int(
        attestation.get("coherence") or 0
    ) >= DECLARE_COHERENCE


def _branch_hash(workspace: Workspace, branch: str) -> str:
    return _candidate_content_hash(_decoded_text_for_panel(workspace, branch))


def _active_branch(workspace: Workspace, branch: str) -> bool:
    return workspace.has_branch(branch) and workspace.get_branch(branch).metadata.get(
        "mode_status", "active"
    ) not in {"rejected", "superseded"}


def _fresh_compare_winner(
    state: InvestigationState,
) -> tuple[str, dict[str, Any]] | None:
    """Return the newest compare winner whose complete hash binding is fresh."""
    for entry in reversed(state.episode_ledger):
        if entry.get("kind") != "compare" or entry.get("status") != "ok":
            continue
        binding = entry.get("comparison_binding")
        if not isinstance(binding, dict):
            continue
        hashes = binding.get("branch_hashes") or {}
        winner = str(binding.get("winner") or "")
        if not winner or winner not in hashes or not _active_branch(state.workspace, winner):
            continue
        if any(
            not state.workspace.has_branch(name)
            or _branch_hash(state.workspace, name) != expected_hash
            for name, expected_hash in hashes.items()
        ):
            continue
        if binding.get("winner_hash") != _branch_hash(state.workspace, winner):
            continue
        result = entry.get("result") or {}
        verdict = next(
            (
                str(item.get("verdict") or "")
                for item in (result.get("verdicts") or [])
                if str(item.get("branch") or "") == winner
            ),
            "",
        ).lower()
        if "reject" in verdict or verdict in {"invalid", "not viable"}:
            continue
        return winner, binding
    return None


def _select_v3_fallback(
    state: InvestigationState,
    executor: WorkspaceToolExecutor,
) -> tuple[str, dict[str, Any]]:
    """Select a fallback using only solver-derived, hash-bound evidence."""
    workspace = state.workspace
    shortlist: list[dict[str, Any]] = []
    for name in workspace.branch_names():
        if not _active_branch(workspace, name):
            continue
        content_hash = _branch_hash(workspace, name)
        attestations = [
            a for a in state.verify_attestations
            if a.get("content_hash") == content_hash and _is_positive_attestation(a)
        ]
        latest_positive = max(
            attestations,
            key=lambda a: (int(a.get("coherence") or 0), int(a.get("created_turn") or 0), str(a.get("episode_id") or "")),
            default=None,
        )
        shortlist.append({
            "branch": name,
            "content_hash": content_hash,
            "positive_attestation": dict(latest_positive) if latest_positive else None,
            "scores": executor._compute_quick_scores(name),
        })

    positively_attested = [item for item in shortlist if item["positive_attestation"]]
    if positively_attested:
        chosen = max(
            positively_attested,
            key=lambda item: (
                int(item["positive_attestation"].get("coherence") or 0),
                int(item["positive_attestation"].get("created_turn") or 0),
                str(item["branch"]),
            ),
        )
        return str(chosen["branch"]), {
            "tier": "fresh_positive_attestation",
            "rationale": "Selected a branch with a fresh positive independent-reader attestation.",
            "attestation": chosen["positive_attestation"],
            "shortlist": shortlist,
        }

    compare = _fresh_compare_winner(state)
    if compare is not None:
        winner, binding = compare
        return winner, {
            "tier": "fresh_compare_winner",
            "rationale": "Selected the winner of the newest hash-fresh compare episode.",
            "comparison_binding": dict(binding),
            "shortlist": shortlist,
        }

    branch, scores = _best_branch_for_auto_declare(
        workspace, state.language, executor.word_set, executor._freq_rank
    )
    return branch, {
        "tier": "scalar_fallback",
        "rationale": "No fresh positive attestation or fresh compare winner was available.",
        "scores": scores,
        "shortlist": shortlist,
    }


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


def _tool_status(result: str) -> str:
    """Compact status token for a tool result, for the turn-summary evidence."""
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return "ok"
    if not isinstance(parsed, dict):
        return "ok"
    if parsed.get("error"):
        return "error"
    status = parsed.get("status")
    if isinstance(status, str):
        return status
    if parsed.get("accepted") is False:
        return "blocked"
    return "ok"


def run_v3(
    cipher_text: CipherText,
    claude_api: Any = None,
    language: str = "en",
    max_iterations: int = 25,
    cipher_id: str = "unknown",
    *,
    session: ModelSession | None = None,
    prior_context: str | None = None,
    automated_preflight: dict[str, Any] | None = None,
    benchmark_context: Any = None,
    resume_state: InvestigationState | None = None,
    verbose: bool = False,
    on_event: Any = None,
    token_budget: int = 20000,
    max_tokens: int = 8192,
    episode_models: dict[str, str] | None = None,
    experiment_queue: ExperimentQueue | None = None,
    max_cost_usd: float | None = None,
) -> RunArtifact:
    """Run one v3 lead session against a cipher. Returns a full RunArtifact.

    ``max_cost_usd`` (M5.3 Slice 1) is the per-run paid ceiling, enforced before
    EVERY paid provider send — lead turns and every worker send inside an episode,
    including mid-episode continuation and the submit-only reserve. Once the
    committed spend reaches it, the loop makes no further paid call: any active
    episode ends budget-class, and the run terminates honestly (best supported
    branch; ``unsolved`` preserved when no fresh positive attestation exists).
    This is distinct from the bake-off runner's matrix-level launch guard."""

    run_id = uuid.uuid4().hex[:12]

    # F6: episode_models maps kind → model id, restricted to the same provider as
    # the lead. Validate the kinds up front (same-provider is enforced by reusing
    # the lead provider's client when cloning per model below).
    if episode_models:
        bad = sorted(set(episode_models) - set(EPISODE_KINDS))
        if bad:
            raise ValueError(f"episode_models has unknown kinds: {bad}")

    # R8(a): on resume the serialized state owns the language. Preferring the
    # `language` param would load English word/pattern resources for a resumed
    # `de` run. Resolve the effective language BEFORE loading resources or
    # building the system prompt.
    if resume_state is not None:
        language = resume_state.language

    # --- session / provider ---
    model_provider = None
    if session is None:
        if claude_api is None:
            raise ValueError("run_v3 requires either `session` or `claude_api`.")
        model_provider = ensure_model_provider(claude_api)
        system_prompt = build_v3_system_prompt(language)
        session = session_factory("lead", model_provider, system_prompt)
    model_name = getattr(session, "model", None) or (
        getattr(model_provider, "model", "") if model_provider else ""
    )

    # --- language resources ---
    dict_path = dictionary.get_dictionary_path(language)
    word_set = dictionary.load_word_set(dict_path) if dict_path else set()
    word_list = pattern.load_word_list(dict_path) if dict_path else []
    pattern_dict = pattern.build_pattern_dictionary(word_list)

    # --- state / workspace ---
    if resume_state is not None:
        state = resume_state
        workspace = state.workspace
    else:
        workspace = Workspace(cipher_text=cipher_text)
        if automated_preflight:
            _install_automated_preflight_branch(workspace, automated_preflight)
        state = InvestigationState(workspace=workspace, language=language)

    executor = WorkspaceToolExecutor(
        workspace=workspace,
        language=language,
        word_set=word_set,
        word_list=word_list,
        pattern_dict=pattern_dict,
        benchmark_context=benchmark_context,
        # M5 (A2/F4): declaration is gated on a fresh verify attestation. The
        # policy holds a LIVE reference to state.verify_attestations (matched by
        # content_hash at declare time); it subclasses NoGatesPolicy so the v3
        # neutral finalize-phase guard is preserved and only check_declare_solution
        # is overridden. meta_declare_unsolved / fallback are NOT gated.
        declaration_policy=AttestationPolicy(attestations=state.verify_attestations),
        # F5/R8: the repair agenda, the hypothesis board (A10 single writer), and
        # the finalist-session store (A1) all live in state; inject them at
        # construction so the lead shares the state-owned objects (no attribute
        # pokes) and they serialize on resume.
        repair_agenda=state.repair_agenda,
        hypothesis_board=state.hypothesis_board,
        finalist_sessions=state.finalist_sessions,
        # The model-variant selection lives in state (serialized for resume);
        # seed the lead executor from it and mirror it back after each
        # dispatched tool call (see _dispatch_tool).
        model_variant=state.model_variant,
    )
    executor.set_max_iterations(max_iterations)

    artifact = RunArtifact(
        run_id=run_id,
        cipher_id=cipher_id,
        model=model_name,
        language=language,
        cipher_alphabet_size=cipher_text.alphabet.size,
        cipher_token_count=len(cipher_text.tokens),
        cipher_word_count=len(cipher_text.words),
        max_iterations=max_iterations,
        automated_preflight=automated_preflight,
        benchmark_context=(
            benchmark_context.to_artifact_dict()
            if hasattr(benchmark_context, "to_artifact_dict")
            else benchmark_context
        ),
    )
    artifact.loop_version = "v3"
    artifact.messages = []  # logical transcript: per-turn NEW content only

    def emit(event: str, payload: dict, **extra: Any) -> None:
        artifact.loop_events.append(
            LoopEvent(event=event, payload=payload, **extra)
        )
        if on_event is not None:
            try:
                on_event(event, payload)
            except Exception:  # noqa: BLE001
                pass
        if verbose:
            print(f"[{event}] {payload}")

    def _episode_event_forwarder(turn: int) -> Any:
        """F6: forward an episode's internal progress events into the lead
        event stream (nested context preserved via ``outer_iteration``)."""
        def _forward(event: str, payload: dict) -> None:
            emit(event, payload, outer_iteration=turn)
        return _forward

    prior_budget = list(state.budget_ledger)
    # Episode spend accumulates here across turns so it survives the per-turn
    # ledger rebuild (run_episode also extends state.budget_ledger for direct
    # callers; the rebuild below is the single source of truth for the lead run).
    episode_budget: list[BudgetEntry] = []

    def sync_budget() -> None:
        state.budget_ledger = (
            prior_budget + episode_budget + list(session.usage_entries())
        )
        artifact.total_input_tokens = sum(e.input_tokens for e in state.budget_ledger)
        artifact.total_output_tokens = sum(e.output_tokens for e in state.budget_ledger)
        artifact.total_cache_read_tokens = sum(
            e.cache_read_tokens for e in state.budget_ledger
        )
        artifact.estimated_cost_usd = state.total_cost()

    def _committed_cost() -> float:
        # M5.3 Slice 1: cost of every COMPLETED send (lead + prior + finished
        # episodes), computed straight from the live ledgers so it is correct
        # regardless of sync_budget() timing. It excludes any in-flight episode,
        # whose own session spend the episode's cost guard adds on top; this is
        # therefore the correct ``outer_cost_usd`` base to hand an episode.
        return sum(
            entry.cost()
            for entry in (prior_budget + episode_budget + list(session.usage_entries()))
        )

    def _cost_ceiling_reached() -> bool:
        # The lead-side per-run paid ceiling: refuse the next paid lead send.
        return max_cost_usd is not None and _committed_cost() >= max_cost_usd

    # --- diagnostic-preflight evidence entry (M1 writes this) ---
    fingerprint = cipher_id_analysis.compute_cipher_fingerprint(
        cipher_text.tokens,
        cipher_text.alphabet.size,
        language=language,
        word_group_count=len(cipher_text.words),
    )
    artifact.cipher_id_report = fingerprint.to_dict()
    if not resume_state:
        top_modes = fingerprint.to_dict().get("suspicion_scores") or {}
        state.add_evidence(
            "diagnostic_preflight",
            turn=0,
            summary=(
                f"alphabet={cipher_text.alphabet.size} tokens="
                f"{len(cipher_text.tokens)} words={len(cipher_text.words)}"
            ),
            fingerprint=artifact.cipher_id_report,
            suspicion_scores=top_modes,
        )
        # Optional external / benchmark context becomes its OWN stable
        # context section, rendered every turn from state (R3) — not a
        # turn-0 evidence entry that scrolls out of the 6-entry evidence
        # window after ~6 turns. Never ground truth — firewall-covered.
        context_parts = [
            part for part in [
                prior_context,
                benchmark_context.prompt
                if hasattr(benchmark_context, "prompt")
                else None,
            ] if part
        ]
        if context_parts:
            state.external_context = "\n\n".join(context_parts)

    start = time.time()
    # F7: assemble the FINAL lead tool list here (never append into episodes.py's
    # constant — import cycle). The v3 lead sees the episode tools plus the two
    # experiment-queue tools.
    # M4: the experiment queue is constructed per run over the state records
    # (never serialized). On resume, loaded pending|running records are already
    # orphaned(loaded); this fresh queue owns no threads.
    queue = experiment_queue if experiment_queue is not None else ExperimentQueue()

    # Full episode ToolCalls accumulate here and merge into artifact.tool_calls
    # at finalize (F10). The compact ledger summary lives in state.episode_ledger.
    episode_tool_calls: list[Any] = []
    current_lead_tool_names: set[str] = set()
    current_episode_kinds: set[str] = set()
    read_call_cache: dict[tuple[str, str, tuple[tuple[str, str], ...]], str] = {}

    read_only_lead_tools = frozenset({
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

    def _lead_read_cache_key(
        name: str, args: dict[str, Any]
    ) -> tuple[str, str, tuple[tuple[str, str], ...]]:
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
                (branch_name, _branch_hash(workspace, branch_name))
                for branch_name in set(branch_names)
                if workspace.has_branch(branch_name)
            )
        )
        return name, normalized, hashes

    def _information_digest() -> str:
        payload = {
            "branch_hashes": sorted(
                {_branch_hash(workspace, name) for name in workspace.branch_names()}
            ),
            "readings": sorted(state.readings),
            "attestations": sorted(
                (
                    str(item.get("content_hash") or ""),
                    int(item.get("coherence") or 0),
                    bool(item.get("reader_accepts")),
                    tuple(str(a) for a in item.get("anomalies") or []),
                )
                for item in state.verify_attestations
            ),
            "episode_results": sorted(
                json.dumps(item.get("result"), sort_keys=True, default=str)
                for item in state.episode_ledger
                if item.get("status") == "ok"
            ),
            "experiment_results": sorted(
                str(item.get("dedup_key") or "")
                for item in state.experiment_queue
                if item.get("status") == "completed"
            ),
            "repair_results": sorted(
                str(item.get("result_content_hash") or "")
                for item in state.repair_transactions
                if item.get("status") == "installed"
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _record_dispatch_result(
        *, name: str, tu: dict[str, Any], turn: int, payload: dict[str, Any]
    ) -> str:
        rendered = json.dumps(payload, ensure_ascii=False)
        executor.call_log.append(ToolCall(
            iteration=turn,
            tool_name=name,
            tool_use_id=str(tu.get("id") or ""),
            arguments=dict(tu.get("input") or {}),
            result=rendered,
            elapsed_ms=0,
        ))
        return rendered

    def _provider_for_model(model_id: str | None) -> Any:
        """Return a provider for an episode model id (same client, swapped model)."""
        if model_provider is None or not model_id:
            return model_provider
        if getattr(model_provider, "model", None) == model_id:
            return model_provider
        clone = copy.copy(model_provider)
        clone.model = model_id
        return clone

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

    def _dispatch_verify_run(args: dict[str, Any], turn: int) -> str:
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
        if not workspace.has_branch(branch):
            return json.dumps(
                {"error": f"verify branch {branch!r} does not exist"}
            )
        candidate = _decoded_text_for_panel(workspace, branch)
        content_hash = _candidate_content_hash(candidate)
        try:
            spec = EpisodeSpec(
                kind="verify", goal=str(args.get("goal") or ""),
                inputs={"candidate_text": candidate, "language": language},
            )
        except Exception as exc:  # noqa: BLE001 - bad spec -> structured error
            return json.dumps({"error": f"invalid verify episode: {exc}"})
        ep_provider = _provider_for_model((episode_models or {}).get("verify"))
        result = run_episode(
            spec, state, provider=ep_provider, language=language,
            word_set=word_set, word_list=word_list, pattern_dict=pattern_dict,
            launching_turn=turn,
            on_event=_episode_event_forwarder(turn),
            max_cost_usd=max_cost_usd, outer_cost_usd=_committed_cost(),
        )
        episode_tool_calls.extend(result.tool_calls)
        episode_budget.extend(
            BudgetEntry.from_dict(b) for b in result.budget_entries
        )
        spend_usd = round(
            sum(b.get("cost_usd", 0.0) for b in result.budget_entries), 6
        )
        emit("episode_complete", {
            "episode_id": result.episode_id, "kind": result.kind,
            "status": result.status, "calls": result.tool_call_count,
            "spend_usd": spend_usd,
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
            )
            state.verify_attestations.append(record.to_dict())
            if not _is_positive_attestation(record.to_dict()):
                for anomaly in record.anomalies:
                    if any(
                        item.get("status", "open") == "open"
                        and item.get("source") == "verify_attestation"
                        and item.get("content_hash") == content_hash
                        and item.get("anomaly") == anomaly
                        for item in state.repair_agenda
                    ):
                        continue
                    numeric_ids = []
                    for existing_item in state.repair_agenda:
                        try:
                            numeric_ids.append(int(existing_item.get("id") or 0))
                        except (TypeError, ValueError):
                            continue
                    state.repair_agenda.append({
                        "id": max(numeric_ids, default=0) + 1,
                        "kind": "verify_anomaly",
                        "source": "verify_attestation",
                        "branch": branch,
                        "content_hash": content_hash,
                        "anomaly": anomaly,
                        "status": "open",
                        "created_turn": turn,
                        "episode_id": result.episode_id,
                    })
            payload["attestation"] = {
                "branch": branch,
                "coherence": record.coherence,
                "reader_accepts": record.reader_accepts,
                "anomalies": record.anomalies,
            }
        return json.dumps(payload, ensure_ascii=False)

    def _dispatch_episode_run(tu: dict[str, Any], turn: int) -> str:
        args = tu.get("input") or {}
        kind = str(args.get("kind") or "")
        if kind == "verify":
            return _dispatch_verify_run(args, turn)
        inputs: dict[str, Any] = {
            "branches": list(args.get("branches") or []),
            "search_tool": args.get("search_tool"),
            "context_note": args.get("context_note"),
            "max_tool_calls": args.get("max_tool_calls"),
        }
        if kind in {"reading", "repair"}:
            reading_branches = [
                name for name in inputs["branches"] if workspace.has_branch(name)
            ]
            if len(reading_branches) != 1:
                return json.dumps({
                    "error": f"{kind} requires exactly one existing branch"
                })
            inputs["candidate_packet"] = build_candidate_reading_packet(
                workspace, reading_branches[0]
            ).to_dict()
        compare_branch_hashes = (
            {
                name: _branch_hash(workspace, name)
                for name in inputs["branches"]
                if workspace.has_branch(name)
            }
            if kind == "compare"
            else None
        )
        # A1/M3: hand a stored Reading to the episode by id (repair kind). The
        # reading DICT is injected as inputs["reading"]; unknown id → error.
        reading_id = args.get("reading_id")
        if reading_id is not None:
            stored = state.readings.get(str(reading_id))
            if stored is None:
                return json.dumps({"error": f"unknown reading_id: {reading_id!r}"})
            inputs["reading"] = stored
        try:
            spec = EpisodeSpec(kind=kind, goal=str(args.get("goal") or ""), inputs=inputs)
        except Exception as exc:  # noqa: BLE001 - bad spec → structured error
            return json.dumps({"error": f"invalid episode_run: {exc}"})
        ep_provider = _provider_for_model((episode_models or {}).get(kind))
        result = run_episode(
            spec, state, provider=ep_provider, language=language,
            word_set=word_set, word_list=word_list, pattern_dict=pattern_dict,
            launching_turn=turn,
            on_event=_episode_event_forwarder(turn),
            max_cost_usd=max_cost_usd, outer_cost_usd=_committed_cost(),
        )
        episode_tool_calls.extend(result.tool_calls)
        episode_budget.extend(
            BudgetEntry.from_dict(b) for b in result.budget_entries
        )
        spend_usd = round(
            sum(b.get("cost_usd", 0.0) for b in result.budget_entries), 6
        )
        emit("episode_complete", {
            "episode_id": result.episode_id, "kind": result.kind,
            "status": result.status, "calls": result.tool_call_count,
            "spend_usd": spend_usd,
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
                (entry for entry in reversed(state.episode_ledger)
                 if entry.get("episode_id") == result.episode_id),
                None,
            )
            if ledger_entry is not None:
                ledger_entry["comparison_binding"] = binding
            payload["comparison_binding"] = binding
        ledger_entry = next(
            (entry for entry in reversed(state.episode_ledger)
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
            state.readings[reading.reading_id] = reading.to_dict()
            payload["reading_id"] = reading.reading_id
        state.add_evidence(
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
                    if workspace.has_branch(name)
                ),
                None,
            )
            if input_branch and result.kind == "survey":
                modes = result.result.get("suspected_modes") or []
                top = modes[0] if modes and isinstance(modes[0], dict) else None
                if top and top.get("mode"):
                    state.hypothesis_board.update(
                        workspace,
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

    def _dispatch_episode_install(tu: dict[str, Any], turn: int) -> str:
        args = tu.get("input") or {}
        ep_id = str(args.get("episode_id") or "")
        branch = str(args.get("branch") or "")
        as_name = args.get("as_name")
        entry = next(
            (e for e in reversed(state.episode_ledger) if e.get("episode_id") == ep_id),
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
        snapshot_hash = _snapshot_content_hash(snap)
        existing = next(
            (
                name for name in workspace.branch_names()
                if _branch_hash(workspace, name) == snapshot_hash
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
            state.branch_aliases.append(alias)
            for item in entry.get("agenda_additions") or []:
                if str(item.get("branch") or "") != branch:
                    continue
                merged = dict(item)
                merged["branch"] = existing
                state.repair_agenda.append(merged)
            return json.dumps({
                "status": "deduplicated",
                "installed": existing,
                "alias": alias,
            }, ensure_ascii=False)
        name = target
        suffix = 2
        while workspace.has_branch(name):
            name = f"{target}_{suffix}"
            suffix += 1
        install_snap = dict(snap)
        install_snap["name"] = name
        install_snap["metadata"] = metadata
        install_snap["created_iteration"] = turn
        from investigation.state import _restore_branch_into
        _restore_branch_into(workspace, install_snap)
        if card_fields:
            state.hypothesis_board.update(workspace, name, **card_fields)
        else:
            source_branch = next(iter(entry.get("input_branches") or []), None)
            source_card = (
                state.hypothesis_board.get(str(source_branch))
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
                state.hypothesis_board.update(workspace, name, **inherited)
        # M6 F5: this dispatcher owns BOTH the collision-rename above AND the
        # verify-attestation writes; keep attestation branch labels in sync when a
        # rename moved attested content to a new name (see the helper's docstring).
        _resync_attestation_branch_on_rename(
            workspace, state.verify_attestations, target, name
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
            state.repair_agenda.append(merged)
        return json.dumps({
            "status": "ok",
            "installed": name,
            "from_episode": ep_id,
            "from_branch": branch,
        }, ensure_ascii=False)

    def _snapshot_content_hash(snapshot: dict[str, Any]) -> str:
        """Render one isolated episode snapshot with the canonical renderer."""
        from investigation.state import _restore_branch_into

        scratch = Workspace(
            cipher_text=workspace.cipher_text,
            plaintext_alphabet=workspace.plaintext_alphabet,
        )
        snap = copy.deepcopy(snapshot)
        _restore_branch_into(scratch, snap)
        return _branch_hash(scratch, str(snap["name"]))

    def _dispatch_repair_transaction(tu: dict[str, Any], turn: int) -> str:
        """Run, validate, install, and record one bounded repair operation."""
        args = tu.get("input") or {}
        branch = str(args.get("branch") or "")
        if not workspace.has_branch(branch):
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn,
                payload={"status": "failed", "reason": "unknown_branch", "branch": branch},
            )
        source_hash = _branch_hash(workspace, branch)
        reading_id = str(args.get("reading_id") or "")
        reading_data = state.readings.get(reading_id) if reading_id else None
        if reading_data is None and not reading_id:
            candidates = [
                item for item in state.readings.values()
                if str(item.get("branch") or "") == branch
                and item.get("candidate_content_hash") == source_hash
            ]
            if candidates:
                reading_data = max(
                    candidates, key=lambda item: int(item.get("created_turn") or 0)
                )
                reading_id = str(reading_data.get("reading_id") or "")
        if reading_data is None:
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn,
                payload={
                    "status": "failed",
                    "reason": "fresh_reading_required",
                    "branch": branch,
                    "content_hash": source_hash,
                },
            )
        reading = Reading.from_dict(reading_data)
        if reading.branch != branch:
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn,
                payload={
                    "status": "failed", "reason": "reading_branch_mismatch",
                    "branch": branch, "reading_branch": reading.branch,
                },
            )
        if reading.candidate_content_hash != source_hash:
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn,
                payload={
                    "status": "failed", "reason": "stale_or_unbound_reading",
                    "branch": branch,
                    "source_content_hash": source_hash,
                    "reading_content_hash": reading.candidate_content_hash,
                },
            )
        duplicate = next(
            (
                item for item in reversed(state.repair_transactions)
                if item.get("status") == "installed"
                and item.get("source_content_hash") == source_hash
                and item.get("reading_id") == reading_id
            ),
            None,
        )
        if duplicate is not None:
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn,
                payload={
                    "status": "duplicate_suppressed",
                    "reason": "source_and_reading_already_handled",
                    "transaction_id": duplicate.get("transaction_id"),
                    "installed": duplicate.get("installed_branch"),
                },
            )

        matching_attestations = [
            item for item in state.verify_attestations
            if item.get("content_hash") == source_hash
        ]
        latest_attestation = max(
            matching_attestations,
            key=lambda item: int(item.get("created_turn") or 0),
            default=None,
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
        episode_payload = json.loads(_dispatch_episode_run({
            "id": f"{tu.get('id')}:repair",
            "name": "episode_run",
            "input": {
                "kind": "repair",
                "goal": str(args.get("goal") or "Repair the bound candidate conservatively."),
                "branches": [branch],
                "reading_id": reading_id,
                "context_note": note,
            },
        }, turn))
        transaction_id = uuid.uuid4().hex[:12]
        base_record = {
            "transaction_id": transaction_id,
            "source_branch": branch,
            "source_content_hash": source_hash,
            "reading_id": reading_id,
            "episode_id": episode_payload.get("episode_id"),
            "addressed_anomalies": anomalies,
            "created_turn": turn,
        }
        if episode_payload.get("status") != "ok":
            record = {
                **base_record, "status": "failed",
                "reason": episode_payload.get("failure_reason") or episode_payload.get("error"),
            }
            state.repair_transactions.append(record)
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn,
                payload={**record, "episode": episode_payload},
            )

        episode_id = str(episode_payload.get("episode_id") or "")
        ledger_entry = next(
            (
                entry for entry in reversed(state.episode_ledger)
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
            digest = _snapshot_content_hash(snapshot)
            if digest != source_hash:
                changed[name] = digest
        requested = str(result.get("best_branch") or "")
        if requested in changed:
            winner = requested
        elif not requested and len(changed) == 1:
            winner = next(iter(changed))
        else:
            reason = "unsupported_winner" if requested else "ambiguous_or_unchanged_finalists"
            record = {
                **base_record, "status": "failed", "reason": reason,
                "claimed_winner": requested or None,
                "changed_finalists": sorted(changed),
            }
            state.repair_transactions.append(record)
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn, payload=record
            )
        if not bool(result.get("applied")):
            record = {
                **base_record, "status": "failed", "reason": "worker_did_not_apply",
                "claimed_winner": winner,
            }
            state.repair_transactions.append(record)
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn, payload=record
            )

        install_payload = json.loads(_dispatch_episode_install({
            "id": f"{tu.get('id')}:install",
            "name": "episode_install_branch",
            "input": {
                "episode_id": episode_id,
                "branch": winner,
                "as_name": str(args.get("as_name") or f"repair_tx_{turn}_{branch}"),
            },
        }, turn))
        installed = str(install_payload.get("installed") or "")
        if install_payload.get("status") not in {"ok", "deduplicated"} or not installed:
            record = {
                **base_record, "status": "failed", "reason": "install_failed",
                "install": install_payload,
            }
            state.repair_transactions.append(record)
            return _record_dispatch_result(
                name="repair_transaction", tu=tu, turn=turn, payload=record
            )
        result_hash = _branch_hash(workspace, installed)
        workspace.get_branch(installed).metadata["repair_transaction"] = {
            "transaction_id": transaction_id,
            "source_branch": branch,
            "source_content_hash": source_hash,
            "reading_id": reading_id,
            "episode_id": episode_id,
            "addressed_anomalies": anomalies,
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
        }
        state.repair_transactions.append(record)
        for item in state.repair_agenda:
            if (
                item.get("status", "open") == "open"
                and item.get("source") == "verify_attestation"
                and item.get("content_hash") == source_hash
            ):
                item["status"] = "addressed"
                item["addressed_by_transaction"] = transaction_id
                item["addressed_turn"] = turn
        state.add_evidence(
            "repair_transaction", turn=turn,
            summary=f"Installed {installed} from bounded repair transaction.",
            **record,
        )
        emit("repair_transaction_complete", dict(record), outer_iteration=turn)
        return _record_dispatch_result(
            name="repair_transaction", tu=tu, turn=turn, payload=record
        )

    def _dispatch_tool(tu: dict[str, Any], turn: int) -> str:
        name = tu["name"]
        if name not in current_lead_tool_names:
            emit("lead_tool_rejected", {
                "tool": name,
                "reason": "lead_tool_not_available",
                "workflow_state": workflow_state(state, executor)["state"],
            }, outer_iteration=turn)
            return _record_dispatch_result(
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
            "branch_hashes": _lead_read_cache_key(name, args)[2],
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        state.call_signature_counts[signature] = (
            state.call_signature_counts.get(signature, 0) + 1
        )
        if state.call_signature_counts[signature] > 1:
            emit("repeated_call", {
                "tool": name,
                "count": state.call_signature_counts[signature],
                "signature": signature,
            }, outer_iteration=turn)
        if name == "episode_run":
            requested_kind = str(args.get("kind") or "")
            if requested_kind not in current_episode_kinds:
                return _record_dispatch_result(
                    name=name,
                    tu=tu,
                    turn=turn,
                    payload={
                        "status": "blocked",
                        "reason": "episode_kind_not_available",
                        "requested_kind": requested_kind,
                        "allowed_kinds": sorted(current_episode_kinds),
                        "workflow_state": workflow_state(state, executor)["state"],
                    },
                )
        cache_key = None
        if name in read_only_lead_tools:
            cache_key = _lead_read_cache_key(name, args)
            if cache_key in read_call_cache:
                emit("duplicate_read_suppressed", {
                    "tool": name,
                    "workflow_state": workflow_state(state, executor)["state"],
                }, outer_iteration=turn)
                return _record_dispatch_result(
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
            result = _dispatch_episode_run(tu, turn)
            return result
        if name == "episode_install_branch":
            return _dispatch_episode_install(tu, turn)
        if name == "repair_transaction":
            phase = workflow_state(state, executor)["state"]
            if phase not in {"candidate_reading", "repair_required"}:
                return _record_dispatch_result(
                    name=name, tu=tu, turn=turn,
                    payload={
                        "status": "blocked",
                        "reason": "repair_transaction_not_ready",
                        "workflow_state": phase,
                    },
                )
            try:
                return _dispatch_repair_transaction(tu, turn)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - preserve the run artifact
                return _record_dispatch_result(
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
                        queue, state, workspace, executor, args, turn
                    )
                else:
                    result_obj = dispatch_experiment_collect(
                        queue, state, workspace, executor, args, turn
                    )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                result_obj = {"error": f"experiment tool {name} failed: {exc}"}
            return json.dumps(result_obj, ensure_ascii=False)
        # M3: the lead hosts the composites too (Part 2). state.readings is the
        # lead's readings map; execute_composite logs the ToolCall + returns the
        # dict, which we serialize for the tool_result. Composites cannot change
        # the model variant, so they early-return WITHOUT the mirror below
        # (consistent with the episode_run / episode_install early returns).
        if name in COMPOSITE_TOOL_NAMES:
            result_obj = execute_composite(
                name, tu.get("input") or {}, executor=executor,
                state_readings=state.readings, turn=turn, tool_use_id=tu["id"],
            )
            result = json.dumps(result_obj, ensure_ascii=False)
            if cache_key is not None:
                read_call_cache[cache_key] = result
            return result
        result = executor.execute(name, args, tool_use_id=tu["id"])
        # Mirror the executor's model-variant selection into state so episodes
        # inherit it and it serializes for resume (act_set_model_variant is the
        # only writer; mirroring unconditionally keeps them in lock-step).
        state.model_variant = executor._model_variant
        if cache_key is not None:
            read_call_cache[cache_key] = result
        return result

    # R1: a resume continues from where the serialized state left off; a
    # fresh run starts at turn 1. ``turn`` is pre-initialized so the
    # post-loop bookkeeping stays valid even if the resume point is already
    # at/after the turn limit (empty range).
    first_turn = (state.turn + 1) if resume_state is not None else 1
    turn = first_turn - 1
    cost_ceiling_hit = False
    for turn in range(first_turn, max_iterations + 1):
        state.turn = turn
        workspace.set_iteration(turn)
        executor.set_iteration(turn)
        # M5.3 Slice 1: the per-run paid ceiling gates the next paid LEAD send.
        # An episode last turn may have pushed committed spend over the ceiling;
        # refuse to send and fall through to honest termination (the fallback
        # block below selects the best supported branch and preserves unsolved
        # without a positive attestation). No further paid call is made.
        if _cost_ceiling_reached():
            cost_ceiling_hit = True
            artifact.status = "exhausted"
            artifact.error_message = (
                f"Per-run cost ceiling reached before turn {turn}: "
                f"${_committed_cost():.4f} >= ${max_cost_usd:.4f}. "
                "No further paid call."
            )
            emit("cost_ceiling_reached", {
                "turn": turn,
                "total_cost_usd": _committed_cost(),
                "max_cost_usd": max_cost_usd,
            })
            break
        # F4: adopt hypothesis branches that gained metadata via an install last
        # turn (or a resumed workspace) into the board once per turn.
        state.hypothesis_board.sync_from_workspace(workspace)
        # M4: harvest completed experiments each turn so they surface as evidence
        # + context without the lead having to ask (promotes pending -> running).
        queue.poll(state, turn)
        emit("iteration_start", {"iteration": turn}, outer_iteration=turn)

        messages = build_lead_context(
            state, executor, turn, token_budget, max_iterations
        )
        current_episode_kinds = set(allowed_episode_kinds(state, executor))
        tools = v3_lead_tool_definitions(
            include_context_tools=benchmark_context is not None,
            allowed_episode_kinds=sorted(current_episode_kinds),
        ) + EXPERIMENT_TOOL_DEFINITIONS
        current_lead_tool_names = {definition["name"] for definition in tools}

        # M6 F7: make the F9 late-turn attestation hint artifact-visible for the
        # bake-off. The context builder RENDERS the reminder (render-only, no
        # emitter); the LOOP re-evaluates the SAME predicate (shared pure helper —
        # no context->loop back-channel) and emits the LoopEvent itself so the
        # analysis can see when the lead was reminded to verify with turns to
        # spare.
        for hint in workflow_hint_candidates(state, executor, turn, max_iterations):
            state.workflow_hint_keys.append(str(hint["key"]))
            emit(
                str(hint["event"]),
                {key: value for key, value in hint.items() if key not in {"event", "key", "message"}},
                outer_iteration=turn,
            )

        try:
            response = session.send(messages, tools=tools, max_tokens=max_tokens)
        except KeyboardInterrupt:
            artifact.status = "stopped"
            artifact.error_message = (
                f"Interrupted by user during model call on turn {turn}; "
                "partial artifact preserved."
            )
            emit("interrupted", {"message": artifact.error_message})
            break
        except ModelProviderError as exc:
            artifact.status = "error"
            artifact.error_message = f"API error on turn {turn}: {exc}"
            emit("error", {"message": artifact.error_message})
            break

        sync_budget()
        assistant_blocks, tool_uses, text_parts = _collect_assistant_blocks(response)
        assistant_message = {"role": "assistant", "content": assistant_blocks}
        artifact.messages.append(assistant_message)

        if turn == 1 and text_parts:
            artifact.plan = "\n\n".join(text_parts)
        if text_parts:
            emit(
                "agent_text",
                {"iteration": turn, "text": text_parts[0][:400]},
                outer_iteration=turn,
            )

        if not tool_uses:
            artifact.status = "exhausted"
            state.add_evidence(
                "turn_summary", turn=turn, summary="no tool calls (exhausted)"
            )
            emit("no_tool_calls", {"iteration": turn}, outer_iteration=turn)
            break

        tool_results_blocks: list[dict[str, Any]] = []
        summary_items: list[str] = []
        interrupted = False
        for idx, tu in enumerate(tool_uses):
            emit(
                "tool_start",
                {"tool": tu["name"], "arguments": tu.get("input") or {}},
                outer_iteration=turn,
            )
            try:
                result = _dispatch_tool(tu, turn)
            except KeyboardInterrupt:
                artifact.status = "stopped"
                artifact.error_message = (
                    f"Interrupted by user during tool `{tu['name']}` on turn "
                    f"{turn}; partial artifact preserved."
                )
                # R5: synthesize a `stopped` tool_result for the interrupted
                # tool AND every remaining tool_use in this batch, so the
                # recorded exchange has one tool_result per tool_use. An
                # unpaired assistant exchange 400s at resume on both
                # providers.
                stopped_content = json.dumps({
                    "status": "stopped",
                    "error": artifact.error_message,
                })
                for pending in tool_uses[idx:]:
                    tool_results_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": pending["id"],
                        "content": stopped_content,
                    })
                emit("interrupted", {"message": artifact.error_message,
                                     "tool": tu["name"]})
                interrupted = True
                break
            tool_results_blocks.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result,
            })
            summary_items.append(f"{tu['name']}:{_tool_status(result)}")
            emit(
                "tool_call",
                {
                    "tool": tu["name"],
                    "result_preview": result[:160],
                    "result_summary": _tool_result_summary(result),
                },
                outer_iteration=turn,
            )
            # Review F-2: an ACCEPTED declaration terminates the run — the
            # remaining tool_uses in this same batch must NOT execute. Without
            # this early-exit, [meta_declare_solution, act_set_mapping] in one
            # turn would mutate the declared branch AFTER the AttestationPolicy
            # allowed it, silently voiding attested == declared == scored (the
            # attestation attach below would find no hash match). Synthesize a
            # `run_terminated` tool_result for each skipped tool_use so the
            # recorded exchange stays one-result-per-use (the R5 pairing rule).
            if executor.terminated and idx + 1 < len(tool_uses):
                terminated_content = json.dumps({
                    "status": "run_terminated",
                    "error": (
                        f"Run terminated by `{tu['name']}` earlier in this "
                        "turn; this tool was not executed."
                    ),
                })
                for pending in tool_uses[idx + 1:]:
                    tool_results_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": pending["id"],
                        "content": terminated_content,
                    })
                    summary_items.append(f"{pending['name']}:run_terminated")
                emit("post_terminate_tools_skipped", {
                    "terminating_tool": tu["name"],
                    "skipped": [p["name"] for p in tool_uses[idx + 1:]],
                }, outer_iteration=turn)
                break

        tool_result_message = {"role": "user", "content": tool_results_blocks}
        artifact.messages.append(tool_result_message)
        # Section 6: keep this exchange for the next turn's rebuilt context.
        state.record_exchange(assistant_message, tool_result_message)
        state.add_evidence(
            "turn_summary", turn=turn, summary=", ".join(summary_items)
        )
        information_digest = _information_digest()
        if state.last_information_digest == information_digest:
            state.no_new_information_streak += 1
        else:
            state.no_new_information_streak = 0
        state.last_information_digest = information_digest
        if state.no_new_information_streak:
            emit("no_new_information", {
                "streak": state.no_new_information_streak,
                "digest": information_digest,
            }, outer_iteration=turn)

        # F5: end-of-turn observability parity with v2. A fresh sync_budget()
        # first so this turn's lead + episode spend is counted, THEN a workspace
        # snapshot of the best branch and a category budget breakdown. Emitting
        # here (post-dispatch) — not the pre-dispatch :675 site — shows THIS
        # turn's workspace and includes any episode spend.
        sync_budget()
        _turn_tokens = artifact.total_input_tokens + artifact.total_output_tokens
        emit(
            "workspace_snapshot",
            _workspace_snapshot_payload(
                workspace,
                language,
                word_set,
                executor._freq_rank,
                turn,
                max_iterations,
                total_tokens=_turn_tokens,
                estimated_cost_usd=artifact.estimated_cost_usd,
            ),
            outer_iteration=turn,
        )
        emit(
            "budget_update",
            {
                "budget_by_category": state.budget_by_category(),
                "total_cost_usd": state.total_cost(),
                "total_tokens": _turn_tokens,
            },
            outer_iteration=turn,
        )

        if interrupted:
            break

        if executor.terminated:
            if getattr(executor, "unsolved_declaration", None) is not None:
                artifact.status = "unsolved"
                emit("declared_unsolved", {
                    "best_branch": executor.unsolved_declaration.get("best_branch"),
                })
            else:
                artifact.status = "solved"
                # M5: carry the matching verify attestation into the declaration
                # so a weak-but-declared solve is visibly weak in the artifact.
                # The declared branch is unchanged since the AttestationPolicy
                # allowed it, so its rendered hash still matches an attestation.
                sol = executor.solution
                if sol is not None and workspace.has_branch(sol.branch):
                    declared_hash = _candidate_content_hash(
                        _decoded_text_for_panel(workspace, sol.branch)
                    )
                    match = max(
                        (
                            a for a in state.verify_attestations
                            if a.get("content_hash") == declared_hash
                        ),
                        key=lambda a: (
                            int(a.get("created_turn") or 0),
                            str(a.get("episode_id") or ""),
                        ),
                        default=None,
                    )
                    if match is not None:
                        sol.attestation = dict(match)
                _declared_attestation = (
                    getattr(sol, "attestation", None) if sol is not None else None
                )
                emit("declared_solution", {
                    "branch": executor.solution.branch if executor.solution else None,
                    "confidence": (
                        executor.solution.self_confidence
                        if executor.solution else None
                    ),
                    "attestation": _declared_attestation,
                })
            break
    else:
        artifact.status = "exhausted"
        emit("max_iterations_reached", {"iterations": max_iterations})

    sync_budget()

    # --- attested fallback or honest best-effort termination ---
    if (
        artifact.status in {"exhausted", "error"}
        and executor.solution is None
        and getattr(executor, "unsolved_declaration", None) is None
    ):
        best_branch, fallback_selection = _select_v3_fallback(state, executor)
        best_scores = next(
            (item.get("scores") for item in fallback_selection.get("shortlist", [])
             if item.get("branch") == best_branch),
            fallback_selection.get("scores") or {},
        )
        original_status = artifact.status
        if original_status == "error":
            reason = (
                "Automatic fallback declaration after agent/API error; "
                "preserving the best available branch for inspection. "
                f"Original error: {artifact.error_message}. "
            )
        elif cost_ceiling_hit:
            reason = (
                "Automatic fallback termination at the per-run cost ceiling; "
                "preserving the best supported branch for inspection. "
            )
        else:
            reason = (
                "Automatic fallback declaration at turn limit; the agent did "
                "not call meta_declare_solution in time. "
            )
        artifact.fallback_selection = fallback_selection
        if fallback_selection["tier"] == "fresh_positive_attestation":
            executor.solution = SolutionDeclaration(
                branch=best_branch,
                rationale=(
                    f"{reason}Selected the positively attested branch: "
                    f"{fallback_selection['rationale']} Scores: {best_scores}."
                ),
                self_confidence=float(
                    fallback_selection["attestation"].get("coherence") or 0
                ) / 10.0,
                declared_at_iteration=max_iterations,
                attestation=dict(fallback_selection["attestation"]),
            )
            artifact.status = "fallback_declared"
            artifact.auto_declared = True
            artifact.attested_fallback = True
            emit("auto_declared_solution", {
                "branch": best_branch,
                "scores": best_scores,
                "selection_tier": fallback_selection["tier"],
                "attested_fallback": True,
            })
        else:
            artifact.status = "error" if original_status == "error" else "unsolved"
            artifact.auto_declared = False
            artifact.attested_fallback = False
            artifact.final_summary = (
                f"Best-effort branch `{best_branch}` retained for inspection; "
                "no fresh positive independent-reader attestation supported a "
                "solution declaration."
            )
            emit("best_effort_selected", {
                "branch": best_branch,
                "scores": best_scores,
                "selection_tier": fallback_selection["tier"],
                "terminal_status": artifact.status,
            })

    # --- finalize ---
    artifact.finished_at = time.time()

    # M4/A9 experiment-queue finalize — ordering pinned (F6). (1) one last poll
    # WITHOUT promotion (harvest already-completed results — this makes
    # post-resume resubmission cheap), (2) flip every remaining pending|running
    # record to orphaned, (3) guarded env restore — ALL before artifact.experiments
    # and artifact.investigation_state are set.
    queue.poll(state, turn, promote=False)
    orphan_reason = "interrupted" if artifact.status == "stopped" else "run_ended"
    for record in state.experiment_queue:
        if record.get("status") in {"pending", "running"}:
            record["status"] = "orphaned"
            record["orphan_reason"] = orphan_reason
    env_warning = queue.finalize_env_restore()
    if env_warning:
        emit("experiment_env_override_retained", {"message": env_warning})
    artifact.experiments = [dict(record) for record in state.experiment_queue]

    # F10: full episode tool calls are merged into the artifact (with episode_id
    # set and iteration = the launching lead turn); the compact ledger summary
    # rides in artifact.episodes / state.
    artifact.tool_calls = list(executor.call_log) + list(episode_tool_calls)
    artifact.episodes = [dict(entry) for entry in state.episode_ledger]
    artifact.attestations = [dict(a) for a in state.verify_attestations]
    artifact.readings = [dict(r) for r in state.readings.values()]
    artifact.tool_requests = list(executor.tool_requests)
    artifact.repair_agenda = [dict(item) for item in state.repair_agenda]
    artifact.cipher_hypotheses = _hypothesis_cards_for_artifact(workspace)
    artifact.solution = executor.solution
    if getattr(executor, "unsolved_declaration", None) is not None:
        artifact.final_summary = str(
            executor.unsolved_declaration.get("reading_summary") or ""
        )
        artifact.error_message = str(
            executor.unsolved_declaration.get("rationale") or ""
        )
    artifact.branches = [
        _branch_snapshot_for(workspace, name) for name in workspace.branch_names()
    ]
    artifact.budget_by_category = state.budget_by_category()
    artifact.session_transcript = session.export_transcript()
    artifact.investigation_state = state.to_artifact_dict()

    emit("run_complete", {
        "status": artifact.status,
        "iterations": min(turn, max_iterations),
        "elapsed_seconds": round(artifact.finished_at - start, 1),
    })
    return artifact
