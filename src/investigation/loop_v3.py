"""run_v3: the v3 lead loop (M1 spec Part 4).

A lead loop over context REBUILT from ``InvestigationState`` each turn, using
the existing tool executor directly (no episodes — that is M2), talking to
models through the ``ModelSession`` seam. The declaration policy is disabled
(``NoGatesPolicy``): declaration is always allowed, confidence recorded.

Phase 0 termination semantics are preserved: declaration tools terminate the
run; exhaustion / provider error fall back to the best available branch
(``fallback_declared`` + ``auto_declared``).
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from agent.loop_v2 import (
    _best_branch_for_auto_declare,
    _branch_snapshot_for,
    _hypothesis_cards_for_artifact,
    _install_automated_preflight_branch,
    _tool_result_summary,
)
from agent.model_provider import (
    ModelProviderError,
    ensure_model_provider,
    _collect_assistant_blocks,
)
from agent.tools_v2 import TOOL_DEFINITIONS, NoGatesPolicy, WorkspaceToolExecutor
from analysis import cipher_id as cipher_id_analysis
from analysis import dictionary, pattern
from artifact.schema import LoopEvent, RunArtifact, SolutionDeclaration
from investigation.context import build_lead_context, build_v3_system_prompt
from investigation.sessions import ModelSession, session_factory
from investigation.state import BudgetEntry, InvestigationState
from models.cipher_text import CipherText
from workspace import Workspace


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
) -> RunArtifact:
    """Run one v3 lead session against a cipher. Returns a full RunArtifact."""

    run_id = uuid.uuid4().hex[:12]

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
        declaration_policy=NoGatesPolicy(),
    )
    executor.set_max_iterations(max_iterations)
    # F5: the repair agenda lives in state; share the list object so agenda
    # edits are reflected in state and serialize on resume.
    executor.repair_agenda = state.repair_agenda
    max_agenda_id = max(
        (int(item.get("id") or 0) for item in state.repair_agenda),
        default=0,
    )
    executor._next_repair_agenda_id = max_agenda_id + 1

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

    prior_budget = list(state.budget_ledger)

    def sync_budget() -> None:
        state.budget_ledger = prior_budget + list(session.usage_entries())
        artifact.total_input_tokens = sum(e.input_tokens for e in state.budget_ledger)
        artifact.total_output_tokens = sum(e.output_tokens for e in state.budget_ledger)
        artifact.total_cache_read_tokens = sum(
            e.cache_read_tokens for e in state.budget_ledger
        )
        artifact.estimated_cost_usd = state.total_cost()

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
    tools = TOOL_DEFINITIONS

    # R1: a resume continues from where the serialized state left off; a
    # fresh run starts at turn 1. ``turn`` is pre-initialized so the
    # post-loop bookkeeping stays valid even if the resume point is already
    # at/after the turn limit (empty range).
    first_turn = (state.turn + 1) if resume_state is not None else 1
    turn = first_turn - 1
    for turn in range(first_turn, max_iterations + 1):
        state.turn = turn
        workspace.set_iteration(turn)
        executor.set_iteration(turn)
        emit("iteration_start", {"iteration": turn}, outer_iteration=turn)

        messages = build_lead_context(
            state, executor, turn, token_budget, max_iterations
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
                result = executor.execute(
                    tu["name"], tu["input"], tool_use_id=tu["id"]
                )
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

        tool_result_message = {"role": "user", "content": tool_results_blocks}
        artifact.messages.append(tool_result_message)
        # Section 6: keep this exchange for the next turn's rebuilt context.
        state.record_exchange(assistant_message, tool_result_message)
        state.add_evidence(
            "turn_summary", turn=turn, summary=", ".join(summary_items)
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
                emit("declared_solution", {
                    "branch": executor.solution.branch if executor.solution else None,
                    "confidence": (
                        executor.solution.self_confidence
                        if executor.solution else None
                    ),
                })
            break
    else:
        artifact.status = "exhausted"
        emit("max_iterations_reached", {"iterations": max_iterations})

    sync_budget()

    # --- fallback auto-declare (Phase 0 semantics preserved) ---
    if (
        artifact.status in {"exhausted", "error"}
        and executor.solution is None
        and getattr(executor, "unsolved_declaration", None) is None
    ):
        best_branch, best_scores = _best_branch_for_auto_declare(
            workspace, language, word_set, executor._freq_rank
        )
        if artifact.status == "error":
            reason = (
                "Automatic fallback declaration after agent/API error; "
                "preserving the best available branch for inspection. "
                f"Original error: {artifact.error_message}. "
            )
        else:
            reason = (
                "Automatic fallback declaration at turn limit; the agent did "
                "not call meta_declare_solution in time. "
            )
        executor.solution = SolutionDeclaration(
            branch=best_branch,
            rationale=(
                f"{reason}Selected the highest-scoring available branch by "
                f"internal dictionary and quadgram signals: {best_scores}."
            ),
            self_confidence=0.0,
            declared_at_iteration=max_iterations,
        )
        artifact.status = "fallback_declared"
        artifact.auto_declared = True
        emit("auto_declared_solution", {"branch": best_branch, "scores": best_scores})

    # --- finalize ---
    artifact.finished_at = time.time()
    artifact.tool_calls = list(executor.call_log)
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
