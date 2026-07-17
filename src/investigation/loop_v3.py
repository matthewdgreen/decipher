"""run_v3: the v3 lead loop (M1 spec Part 4).

A lead loop over context REBUILT from ``InvestigationState`` each turn, using
the existing tool executor directly (no episodes — that is M2), talking to
models through the ``ModelSession`` seam. The declaration policy is disabled
(``NoGatesPolicy``): declaration is always allowed, confidence recorded.

Declaration tools terminate the run. Exhaustion preserves a best-effort branch
for inspection but is honestly unsolved unless a fresh positive independent
attestation supports a synthesized fallback declaration.

The dispatch layer lives in ``investigation.host`` (host-extraction
slice); ``run_v3`` is the turn-loop driver over an ``InvestigationHost``.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from agent.loop_shared import (
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
    call_with_rate_limit_retry,
    ensure_model_provider,
    _collect_assistant_blocks,
)
from agent.tools_v2 import AttestationPolicy, WorkspaceToolExecutor
from investigation.episodes import (
    EPISODE_KINDS,
    v3_lead_tool_definitions,
)
from investigation.experiments import (
    EXPERIMENT_TOOL_DEFINITIONS,
    ExperimentQueue,
)
from analysis import cipher_id as cipher_id_analysis
from analysis import dictionary, pattern
from artifact.schema import LoopEvent, RunArtifact, SolutionDeclaration
from investigation.context import (
    allowed_episode_kinds,
    build_lead_context,
    build_v3_system_prompt,
    workflow_state,
    workflow_hint_candidates,
)
from investigation.host import InvestigationHost, _active_branch, _branch_hash
from investigation.sessions import ModelSession, session_factory
from investigation.state import (
    BudgetEntry,
    InvestigationState,
    attestation_is_positive,
    latest_attestation_for_hash,
)
from models.cipher_text import CipherText
from workspace import Workspace


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
        # Slice 6: the LATEST verdict on the current content governs (same
        # rule as the declare gate). An older positive superseded by a newer
        # negative on identical content does NOT qualify.
        latest = latest_attestation_for_hash(state.verify_attestations, content_hash)
        positive = latest if attestation_is_positive(latest) else None
        shortlist.append({
            "branch": name,
            "content_hash": content_hash,
            "positive_attestation": dict(positive) if positive else None,
            "scores": executor._compute_quick_scores(name),
        })

    positively_attested = [item for item in shortlist if item["positive_attestation"]]
    if positively_attested:
        # Slice 6 ordering [FIXED]: coherence no longer sorts the tier. Order
        # by the reader's meaning-recovery estimate, then language confidence,
        # then recency, then name (fully deterministic; legacy-derived
        # positives carry 0.0/0.0 and sort last).
        chosen = max(
            positively_attested,
            key=lambda item: (
                float(item["positive_attestation"].get("semantic_recoverability") or 0.0),
                float(item["positive_attestation"].get("target_language_confidence") or 0.0),
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


def _compute_branch_roles(
    state: InvestigationState,
    executor: WorkspaceToolExecutor,
    declared_or_selected: str | None = None,
) -> dict[str, str | None]:
    """The four distinguished branch roles (M5.3 Slice 7, master 504-509).

    Derived, never stored in InvestigationState: a resume recomputes them
    from the restored state + a fresh executor. Branch NAMES only —
    firewall-safe by construction.

    - best_scored_branch: the internal scalar-best branch
      (_best_branch_for_auto_declare — the value M5.2 telemetry mislabeled
      as plain `branch`).
    - workflow_branch: the branch the workflow state machine is focused on
      (workflow_state(...)["branch"]; None when the menu names no branch).
    - latest_installed_branch: the newest `installed` repair transaction
      whose installed branch still exists in the workspace.
    - declared_or_selected_branch: the declared / honest-unsolved /
      fallback-selected branch. None until termination resolves it
      (mid-run snapshots always carry None).
    """
    best_scored_branch = _best_branch_for_auto_declare(
        state.workspace, state.language, executor.word_set, executor._freq_rank
    )[0]
    workflow_branch = workflow_state(state, executor).get("branch")
    latest_installed_branch = next(
        (
            str(item.get("installed_branch") or "")
            for item in reversed(state.repair_transactions)
            if item.get("status") == "installed"
            and state.workspace.has_branch(str(item.get("installed_branch") or ""))
        ),
        None,
    )
    return {
        "best_scored_branch": best_scored_branch,
        "workflow_branch": workflow_branch,
        "latest_installed_branch": latest_installed_branch,
        "declared_or_selected_branch": declared_or_selected,
    }


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
        max_cost_usd=max_cost_usd,
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

    prior_budget = list(state.budget_ledger)

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

    host = InvestigationHost(
        state=state, workspace=workspace, executor=executor, queue=queue,
        emit=emit, session=session, model_provider=model_provider,
        language=language, word_set=word_set, word_list=word_list,
        pattern_dict=pattern_dict, episode_models=episode_models,
        max_cost_usd=max_cost_usd, prior_budget=prior_budget,
    )

    def sync_budget() -> None:
        # Host rebuilds state.budget_ledger; the artifact mirror stays
        # loop-side (the host is artifact-free by design).
        host.sync_budget()
        artifact.total_input_tokens = sum(
            e.input_tokens for e in state.budget_ledger
        )
        artifact.total_output_tokens = sum(
            e.output_tokens for e in state.budget_ledger
        )
        artifact.total_cache_read_tokens = sum(
            e.cache_read_tokens for e in state.budget_ledger
        )
        artifact.estimated_cost_usd = state.total_cost()

    # R1: a resume continues from where the serialized state left off; a
    # fresh run starts at turn 1. ``turn`` is pre-initialized so the
    # post-loop bookkeeping stays valid even if the resume point is already
    # at/after the turn limit (empty range).
    first_turn = (state.turn + 1) if resume_state is not None else 1
    turn = first_turn - 1
    cost_ceiling_hit = False
    # Tool-less-turn resilience (K3 incident, 2026-07-17): a single turn with
    # zero tool calls must not silently end the run. Two bounded recoveries:
    # - TRUNCATION: a reasoning-heavy model (e.g. moonshotai/kimi-k3) can spend
    #   the whole output budget thinking and get cut off BEFORE emitting its
    #   tool call (observed: output == max_tokens exactly, no text, no tools).
    #   Retry with a doubled budget, at most twice per run, capped at 32k.
    #   Reasoning EFFORT is deliberately not capped — that is a quality knob;
    #   this is a robustness bug. Spend stays guarded by max_cost_usd.
    # - TEXT-ONLY: a model that narrates without acting gets ONE evidence-log
    #   nudge (the rebuilt context surfaces it next turn) before the honest
    #   `exhausted` terminal. Retries consume normal turn numbers, so the
    #   max_iterations bound is unchanged.
    lead_max_tokens = max_tokens
    truncation_retries_left = 2
    no_tool_nudges_left = 1
    # CLI-3: surface workflow-phase transitions live. Compared turn-over-turn;
    # a change emits `workflow_state_changed` BEFORE the send so the transition
    # appears where it happened (e.g. searching → repair_exhausted).
    prev_workflow_phase: str | None = None
    for turn in range(first_turn, max_iterations + 1):
        state.turn = turn
        workspace.set_iteration(turn)
        executor.set_iteration(turn)
        # M5.3 Slice 1: the per-run paid ceiling gates the next paid LEAD send.
        # An episode last turn may have pushed committed spend over the ceiling;
        # refuse to send and fall through to honest termination (the fallback
        # block below selects the best supported branch and preserves unsolved
        # without a positive attestation). No further paid call is made.
        if host.cost_ceiling_reached():
            cost_ceiling_hit = True
            artifact.status = "exhausted"
            artifact.error_message = (
                f"Per-run cost ceiling reached before turn {turn}: "
                f"${host.committed_cost():.4f} >= ${max_cost_usd:.4f}. "
                "No further paid call."
            )
            emit("cost_ceiling_reached", {
                "turn": turn,
                "total_cost_usd": host.committed_cost(),
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
        host.set_available_tools({definition["name"] for definition in tools})

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

        # CLI-3: emit a workflow-phase transition when the phase changed since
        # the previous turn (the menu is the same shared helper the context
        # builder uses). Emitted before the send so the transition renders where
        # it occurred; `repair_exhausted` and other mid-run states become live.
        menu = workflow_state(state, executor)
        current_workflow_phase = str(menu.get("state") or "")
        if current_workflow_phase != prev_workflow_phase:
            emit("workflow_state_changed", {
                "from": prev_workflow_phase,
                "to": current_workflow_phase,
                "branch": menu.get("branch"),
            }, outer_iteration=turn)
            prev_workflow_phase = current_workflow_phase

        try:
            # Transient 429s get a few short, bounded retries (Retry-After
            # honored); quota exhaustion and persistent limits still land in
            # the ModelProviderError arm below — the honest error terminal.
            response = call_with_rate_limit_retry(
                lambda: session.send(
                    messages, tools=tools, max_tokens=lead_max_tokens
                ),
                on_retry=lambda attempt, delay, exc: emit(
                    "rate_limit_retry", {
                        "iteration": turn,
                        "attempt": attempt,
                        "delay_seconds": delay,
                        "error": str(exc)[:200],
                    }, outer_iteration=turn,
                ),
            )
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
            # CLI-3: the narration is now the primary human display line; join
            # ALL text blocks and raise the emit cap from 400 to 4000 chars so
            # multi-sentence turns are not visibly truncated. Full text remains
            # in artifact.messages.
            emit(
                "agent_text",
                {"iteration": turn, "text": "\n\n".join(text_parts)[:4000]},
                outer_iteration=turn,
            )

        if not tool_uses:
            usage = getattr(response, "usage", None)
            out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            truncated = out_tokens >= lead_max_tokens
            if truncated and truncation_retries_left > 0:
                truncation_retries_left -= 1
                previous_cap = lead_max_tokens
                lead_max_tokens = min(lead_max_tokens * 2, 32768)
                state.add_evidence(
                    "turn_summary", turn=turn,
                    summary=(
                        f"turn {turn} hit the {previous_cap}-token output cap "
                        "before emitting a tool call (long reasoning); "
                        f"retrying with max_tokens={lead_max_tokens}"
                    ),
                )
                emit("lead_truncation_retry", {
                    "iteration": turn,
                    "output_tokens": out_tokens,
                    "previous_max_tokens": previous_cap,
                    "new_max_tokens": lead_max_tokens,
                }, outer_iteration=turn)
                continue
            if not truncated and no_tool_nudges_left > 0:
                no_tool_nudges_left -= 1
                state.add_evidence(
                    "turn_summary", turn=turn,
                    summary=(
                        f"turn {turn} produced no tool call. A tool call is "
                        "REQUIRED every turn: act via a tool now, or call "
                        "meta_declare_unsolved to end the run honestly."
                    ),
                )
                emit("no_tool_calls_nudge", {"iteration": turn},
                     outer_iteration=turn)
                continue
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
                result = host.handle_tool(tu, turn)
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
        information_digest = host.information_digest()
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
        snapshot_payload = _workspace_snapshot_payload(
            workspace,
            language,
            word_set,
            executor._freq_rank,
            turn,
            max_iterations,
            total_tokens=_turn_tokens,
            estimated_cost_usd=artifact.estimated_cost_usd,
        )
        # M5.3 Slice 7: mid-run the declared_or_selected role is always None.
        snapshot_payload["branch_roles"] = _compute_branch_roles(state, executor)
        emit("workspace_snapshot", snapshot_payload, outer_iteration=turn)
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
    # M5.3 Slice 7: hoisted so the finalize block can resolve the
    # declared-or-selected branch role for the honest best-effort tier (the
    # attested-fallback tier sets executor.solution, covered separately).
    fallback_best_branch: str | None = None
    if (
        artifact.status in {"exhausted", "error"}
        and executor.solution is None
        and getattr(executor, "unsolved_declaration", None) is None
    ):
        best_branch, fallback_selection = _select_v3_fallback(state, executor)
        fallback_best_branch = best_branch
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
                self_confidence=round(
                    (
                        float(fallback_selection["attestation"].get(
                            "target_language_confidence") or 0.0)
                        + float(fallback_selection["attestation"].get(
                            "semantic_recoverability") or 0.0)
                    ) / 2.0,
                    4,
                ),
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
    artifact.tool_calls = list(executor.call_log) + list(host.episode_tool_calls)
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

    # M5.3 Slice 7: stamp the four distinguished branch roles at termination.
    # Resolve declared-or-selected: a declaration/honest-unsolved sets it
    # directly; the honest best-effort fallback tier counts as "selected"
    # (master 509). An interrupted/exhausted run that never reached the
    # fallback block leaves it None (nothing was declared or selected).
    declared_or_selected: str | None = None
    if executor.solution is not None:
        declared_or_selected = executor.solution.branch
    elif getattr(executor, "unsolved_declaration", None) is not None:
        declared_or_selected = executor.unsolved_declaration.get("best_branch")
    elif artifact.fallback_selection is not None:
        declared_or_selected = fallback_best_branch
    artifact.branch_roles = _compute_branch_roles(
        state, executor, declared_or_selected
    )

    artifact.investigation_state = state.to_artifact_dict()

    emit("run_complete", {
        "status": artifact.status,
        "iterations": min(turn, max_iterations),
        "elapsed_seconds": round(artifact.finished_at - start, 1),
    })
    return artifact
