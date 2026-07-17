"""Repair v1: client-compiled finalists (Part 8.2 + 8.3).

``repair_hypotheses_test`` deterministically compiles word hypotheses against a
branch in an ISOLATED scratch workspace (no LLM) and records a compile session
plus a synthetic ``repair_compile`` episode-ledger entry. ``repair_transaction``
then validates and installs ONE chosen winner through the SAME host acceptance
code the v3 internal repair uses — via ``check_repair_preconditions`` +
``validate_and_install_repair`` (Part 8.1 split), bypassing ``handle_tool``'s
phase gate (REP-7 advisory).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor
from artifact.schema import ToolCall
from investigation import episodes
from investigation import state as state_module
from investigation.actions import execute_composite
from investigation.host import _branch_hash

_KNOB_KEYS = ("window_size", "max_edits", "max_hypotheses", "max_hypotheses_per_window")
_MAX_COMPILES = 8


def _mcp_tu(name: str, inp: dict) -> dict:
    return {"id": f"mcp_{uuid.uuid4().hex[:8]}", "name": name, "input": inp}


def compile_hypotheses(runtime: Any, records: dict, args: dict, turn: int) -> dict:
    """Deterministic word-hypothesis compile in an isolated scratch workspace."""
    state = runtime.state
    host = runtime.host
    workspace = runtime.workspace
    branch = str(args.get("branch") or "")
    if not workspace.has_branch(branch):
        return {"status": "error", "reason": "unknown_branch", "branch": branch}
    source_hash = _branch_hash(workspace, branch)

    # Duplicate suppression by (source content, args digest).
    digest_src = {
        "branch": branch,
        "hypotheses": args.get("hypotheses"),
        "window_size": args.get("window_size"),
        "max_edits": args.get("max_edits"),
        "max_hypotheses": args.get("max_hypotheses"),
        "max_hypotheses_per_window": args.get("max_hypotheses_per_window"),
    }
    args_digest = hashlib.sha256(
        json.dumps(digest_src, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    for stored in records.get("repair_compiles") or []:
        if (
            stored.get("source_content_hash") == source_hash
            and stored.get("args_digest") == args_digest
        ):
            return {
                "status": "duplicate_suppressed",
                "compile_id": stored.get("compile_id"),
                "note": (
                    "Identical compile already exists for this content; reuse "
                    "it or change the hypotheses."
                ),
            }

    compile_id = "rc" + uuid.uuid4().hex[:10]

    # Scratch isolation: deep-copied snapshots (EPI-1's exact mechanism).
    scratch = episodes._build_episode_workspace(state, [branch])
    scratch_executor = WorkspaceToolExecutor(
        workspace=scratch,
        language=state.language,
        word_set=runtime._word_set,
        word_list=runtime._word_list,
        pattern_dict=runtime._pattern_dict,
        benchmark_context=None,
        declaration_policy=NoGatesPolicy(),
        episode_toolset=set(episodes.EPISODE_KINDS["repair"]["toolset"]),
        model_variant=state.model_variant,
    )
    scratch_executor.episode_id = compile_id
    scratch_executor.set_iteration(turn)

    # Force install on every hypothesis (server owns installation).
    hyps = [dict(h, install=True) for h in args["hypotheses"]]
    composite_args: dict[str, Any] = {"branch": branch, "hypotheses": hyps}
    for key in _KNOB_KEYS:
        if key in args and args[key] is not None:
            composite_args[key] = args[key]

    result_obj = execute_composite(
        "hypothesis_test_words", composite_args,
        executor=scratch_executor, state_readings=state.readings, turn=turn,
        tool_use_id=f"mcp_{compile_id}",
    )
    if isinstance(result_obj, dict) and "error" in result_obj:
        return {
            "status": "failed", "reason": "compile_error",
            "error": result_obj["error"],
        }

    # Snapshots, episode-format: requested branch + created forks; "main" only
    # if it was the source (the same rule run_episode uses).
    snapshots = [
        state_module._serialize_branch(scratch, name)
        for name in scratch.branch_names()
        if name != "main" or "main" == branch
    ]
    changed: dict[str, str] = {}
    for snap in snapshots:
        name = str(snap.get("name") or "")
        digest = host._snapshot_content_hash(snap)
        if digest != source_hash:
            changed[name] = digest

    # Synthetic episode-ledger entry: lets validate_and_install_repair +
    # _dispatch_episode_install run unchanged.
    state.episode_ledger.append({
        "episode_id": compile_id, "kind": "repair_compile",
        "goal": f"client-compiled word hypotheses on {branch}",
        "status": "ok", "failure_reason": None, "result": None,
        "summary": (
            f"{len(hyps)} hypotheses probed; {len(changed)} changed finalist fork(s)"
        ),
        "branch_snapshots": snapshots, "tool_call_count": 1,
        "budget_entries": [], "agenda_additions": [],
        "launching_turn": turn, "input_branches": [branch],
    })

    # Extend the host evidence stream with the compile ToolCalls; persist them in
    # the sidecar for cross-process/restart transactions.
    compile_calls = [
        tc for tc in scratch_executor.call_log
        if getattr(tc, "episode_id", None) == compile_id
    ]
    host.episode_tool_calls.extend(compile_calls)
    stored_calls = [
        {
            "tool_name": tc.tool_name, "result": tc.result,
            "episode_id": tc.episode_id, "iteration": tc.iteration,
            "tool_use_id": tc.tool_use_id,
        }
        for tc in compile_calls
    ]

    session = {
        "compile_id": compile_id, "branch": branch,
        "source_content_hash": source_hash, "args_digest": args_digest,
        "created_turn": turn, "result": result_obj, "tool_calls": stored_calls,
        "changed_finalists": dict(changed),
    }
    compiles = records.setdefault("repair_compiles", [])
    compiles.append(session)
    del compiles[:-_MAX_COMPILES]  # cap at 8, drop oldest first

    return {
        "status": "ok", "compile_id": compile_id, "branch": branch,
        "source_content_hash": source_hash,
        "changed_finalists": [
            {"branch": name, "content_hash": h} for name, h in changed.items()
        ],
        "result": result_obj,
        "next": (
            "Pick a winner from changed_finalists (with >1 changed it must be one "
            "of result.finalists) and call repair_transaction with this compile_id."
        ),
    }


def _winner_edits(session_result: Any, winner: str) -> list:
    if not isinstance(session_result, dict):
        return []
    for finalist in session_result.get("finalists") or []:
        if isinstance(finalist, dict) and finalist.get("installed_fork") == winner:
            return list(finalist.get("edits") or [])
    for item in session_result.get("items") or []:
        if isinstance(item, dict) and item.get("installed_fork") == winner:
            return list(item.get("edits") or [])
    return []


def dispatch_repair_transaction(runtime: Any, records: dict, args: dict, turn: int) -> dict:
    """Host-validated acceptance + install of ONE client-compiled winner."""
    state = runtime.state
    host = runtime.host
    branch = str(args.get("branch") or "")
    compile_id = str(args.get("compile_id") or "")
    tu = _mcp_tu("repair_transaction", {
        k: v for k, v in args.items()
        if k not in {"investigation_id", "expected_revision"}
    })

    # 1. Preconditions (REP-1/REP-2/SAT-1/SAT-3) — record blocked results too.
    pre = host.check_repair_preconditions(
        branch=branch, reading_id_arg=str(args.get("reading_id") or ""), turn=turn
    )
    if "blocked" in pre:
        rendered = host._record_dispatch_result(
            name="repair_transaction", tu=tu, turn=turn, payload=pre["blocked"]
        )
        return json.loads(rendered)

    # 2. Compile-session lookup + freshness.
    session = next(
        (s for s in records.get("repair_compiles") or [] if s.get("compile_id") == compile_id),
        None,
    )
    if session is None:
        return {"status": "error", "reason": "unknown_compile_id"}
    if session.get("branch") != branch:
        return {"status": "error", "reason": "compile_branch_mismatch"}
    if session.get("source_content_hash") != pre["source_hash"]:
        return {
            "status": "failed", "reason": "stale_compile",
            "note": (
                "Branch content changed since this compile; rerun "
                "repair_hypotheses_test."
            ),
        }

    # 3. Re-materialize compile evidence if this process didn't run the compile.
    if not any(
        getattr(tc, "episode_id", None) == compile_id for tc in host.episode_tool_calls
    ):
        for call in session.get("tool_calls") or []:
            host.episode_tool_calls.append(ToolCall(
                iteration=call.get("iteration") or turn,
                tool_name=call.get("tool_name") or "",
                tool_use_id=call.get("tool_use_id") or "",
                arguments={},
                result=call.get("result") or "",
                episode_id=call.get("episode_id"),
            ))

    # 4. Synthesize the worker-result equivalent.
    winner = str(args["winner"])
    winner_edits = _winner_edits(session.get("result"), winner)
    worker_result = {
        "applied": bool(session.get("changed_finalists")),
        "best_branch": winner,
        "edits": [str(e) for e in winner_edits],
        "verdicts": [], "collateral": {},
        "notes": "client-compiled finalists (repair_hypotheses_test)",
    }
    episode_payload = {
        "episode_id": compile_id, "status": "ok", "result": worker_result,
    }

    # 5. base_record exactly as the v3 dispatcher builds it, plus two additive keys.
    base_record = {
        "transaction_id": uuid.uuid4().hex[:12],
        "source_branch": branch,
        "source_content_hash": pre["source_hash"],
        "reading_id": pre["reading_id"],
        "interpretation_id": pre["reading_id"],
        "interpretation_digest": pre["interp_digest"],
        "attestation_key": pre["att_key"],
        "saturation_key": pre["sat_key"],
        "pair_digest": pre["pair"],
        "retry_of": pre["retry_of"],
        "episode_id": compile_id,
        "addressed_anomalies": pre["anomalies"],
        "created_turn": turn,
        "compile_id": compile_id,
        "mode": "client_compiled",
    }

    # 6-7. Validate + install through the shared host code; parse for the envelope.
    payload_str = host.validate_and_install_repair(
        tu=tu, turn=turn, branch=branch, source_hash=pre["source_hash"],
        att_key=pre["att_key"], pair=pre["pair"], base_record=base_record,
        episode_payload=episode_payload,
        as_name=str(args.get("as_name") or f"repair_tx_{turn}_{branch}"),
    )
    return json.loads(payload_str)
