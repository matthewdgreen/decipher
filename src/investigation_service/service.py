"""InvestigationService: the transport-neutral investigation service layer.

The dispatch pipeline (validate → resolve → terminal-check → lease →
revision-check → dispatch → commit → revision-inject) and every domain helper
were extracted VERBATIM from ``mcp_server.server.DecipherMCPServer`` (interface
lockstep contract, spec §6/§7.1). Both transport skins call into this one layer:

- the MCP server (``mcp_server.server``) keeps only JSON-RPC framing, session
  lifecycle, and a call into ``dispatch``; and
- the structured investigation CLI (I-1+) keeps only argv↔dict mapping.

This is a pure byte-parity extraction: the same dicts flow, the same gates fire,
the same revision injection happens. No behavior change.

Lease-lifetime policy: the service accepts a :class:`LeasePolicy`. MCP passes
``SESSION_HELD`` — runtimes and their writer leases live for the process
lifetime and are finalized in :meth:`shutdown`, exactly as before. The CLI's
invocation-held policy is deferred to I-2 (it needs the registry's explicit
``release_lease``/``close`` API, which does not exist yet); only ``SESSION_HELD``
is implemented here.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
import traceback
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel
from analysis import cipher_id as cipher_id_analysis
from investigation import context, episodes, loop_v3
from investigation import host as host_module
from investigation.reading import Reading, build_candidate_reading_packet
from investigation.state import (
    InvestigationState,
    attestation_is_positive,
    attestation_key,
    get_or_create_saturation_entry,
    latest_attestation_for_hash,
    saturation_key,
)
from workspace import Workspace

from investigation_service import manifest
from mcp_server import intake, repair, verify
from mcp_server.registry import (
    InvestigationNotFound,
    InvestigationRegistry,
)
from mcp_server.runtime import InvestigationRuntime
from mcp_server.status import build_status_brief

# MCP tool name -> the v2/v3 name handed to host.handle_tool.
_HANDLE_TOOL_MAP = {
    "observe_diagnosis": "observe_diagnosis",
    "decode_show": "decode_show",
    "hypothesis_branch_create": "workspace_create_hypothesis_branch",
    "hypothesis_branch_update": "workspace_update_hypothesis",
    "hypothesis_branch_reject": "workspace_reject_hypothesis",
    "hypothesis_next_steps": "workspace_hypothesis_next_steps",
    "experiment_submit": "experiment_submit",
    "experiment_collect": "experiment_collect",
    "branch_adjudicate": "branch_adjudicate",
    "act_set_model_variant": "act_set_model_variant",
    "meta_declare_solution": "meta_declare_solution",
    "meta_declare_unsolved": "meta_declare_unsolved",
}
_MAX_CIPHERTEXT = 200_000

_SERVER_CODE_INFO: dict | None = None


class LeasePolicy(Enum):
    """Lease-lifetime policy the service dispatches under.

    ``SESSION_HELD`` (MCP): once acquired, a writer lease and its live runtime
    persist for the process lifetime; finalized in :meth:`InvestigationService.shutdown`.
    The CLI's ``INVOCATION_HELD`` policy (acquire at dispatch, release in a
    ``finally``) is deferred to I-2 and is NOT implemented here.
    """

    SESSION_HELD = "session_held"


def _server_code_info() -> dict:
    """Git revision of the code this server process is actually running.

    Computed once per process (the answer cannot change without a restart —
    that is the point). Lets a client verify, from inside a session, that a
    freshly pulled fix is live: compare ``git_head`` against
    ``git rev-parse --short HEAD`` in the working clone; a mismatch or a
    stale ``started_at`` means the server predates the pull and the session
    must be restarted. Degrades to nulls outside a git checkout.
    """
    global _SERVER_CODE_INFO
    if _SERVER_CODE_INFO is None:
        import subprocess
        repo_root = Path(__file__).resolve().parents[2]
        head = dirty = None
        try:
            head = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or None
            dirty = bool(subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root, capture_output=True, text=True, timeout=5,
            ).stdout.strip())
        except Exception:  # noqa: BLE001 - non-git installs are fine
            pass
        _SERVER_CODE_INFO = {
            "git_head": head,
            "dirty": dirty,
            "started_at": time.time(),
        }
    return _SERVER_CODE_INFO


class InvestigationService:
    """Transport-neutral investigation service: the dispatch pipeline + domain."""

    def __init__(
        self,
        *,
        registry: InvestigationRegistry,
        verify_provider: Any = None,
        verify_model: str | None = None,
        max_cost_usd: float = 5.0,
        synchronous_experiments: bool = False,
        client_name: str = "unknown",
        lease_policy: LeasePolicy = LeasePolicy.SESSION_HELD,
    ) -> None:
        if lease_policy is not LeasePolicy.SESSION_HELD:
            raise NotImplementedError(
                "only SESSION_HELD is implemented in I-0; the CLI's "
                "invocation-held policy lands in I-2"
            )
        self.registry = registry
        self.verify_provider = verify_provider
        self.verify_model = verify_model
        self.verification_available = verify_provider is not None
        self.max_cost_usd = max_cost_usd
        self.synchronous_experiments = synchronous_experiments
        self.client_name = client_name
        self.lease_policy = lease_policy
        # Live runtimes for investigations this process holds the writer lease on.
        self._runtimes: dict[str, InvestigationRuntime] = {}

    # ---------------------------------------------------------------- lifecycle
    def _build_runtime(self, document: dict) -> InvestigationRuntime:
        investigation_id = document["meta"]["investigation_id"]
        cell: dict[str, Any] = {}

        def emit(event: str, payload: dict, **extra: Any) -> None:
            rt = cell.get("runtime")
            turn = rt.state.turn if rt is not None else 0
            self.registry.append_event(investigation_id, {
                "ts": time.time(), "event": event, "payload": payload, "turn": turn,
            })

        runtime = InvestigationRuntime(
            document=document, emit=emit,
            verify_provider=self.verify_provider, verify_model=self.verify_model,
            max_cost_usd=self.max_cost_usd,
            synchronous_experiments=self.synchronous_experiments,
        )
        cell["runtime"] = runtime
        return runtime

    def shutdown(self) -> None:
        """Finalize + commit every leased runtime (Part 4.4)."""
        for investigation_id, runtime in list(self._runtimes.items()):
            try:
                runtime.finalize(runtime.state.turn)
                document = {"schema_version": 1, **runtime.persist_dict()}
                self.registry.commit(investigation_id, document)
            except Exception:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, name: str, arguments: dict) -> dict:
        # Step 2: validate arguments against the tool schema.
        schema = manifest.schema_for(name)
        errors = episodes.validate_against_schema(arguments, schema or {})
        if errors:
            return {"status": "error", "reason": "invalid_arguments", "errors": errors}

        if name == "investigation_start":
            return self._start(arguments)
        if name == "investigation_list":
            return self._list(arguments)

        tool_class = manifest.TOOL_CLASSES[name]
        is_mutate = tool_class == "mutate"
        investigation_id = str(arguments.get("investigation_id") or "")

        # Step 3: resolve. Holder trusts its live runtime; others load fresh.
        holder = self.registry.holds_lease(investigation_id)
        if holder:
            runtime = self._runtimes[investigation_id]
            meta = runtime.meta
        else:
            try:
                document = self.registry.load(investigation_id)
            except InvestigationNotFound:
                return {
                    "status": "error", "reason": "investigation_not_found",
                    "investigation_id": investigation_id,
                }
            meta = document["meta"]
            runtime = None

        # Step 4: terminal check (mutations only).
        if is_mutate and meta.get("status") != "active":
            return {
                "status": "blocked", "reason": "investigation_terminal",
                "terminal_status": meta.get("status"),
            }

        # Step 5: mutations acquire the lease, check the revision, bump the turn.
        if is_mutate:
            if not self.registry.acquire_lease(investigation_id):
                return {
                    "status": "blocked", "reason": "writer_lease_held",
                    "holder": self.registry.lease_holder_hint(investigation_id),
                    "note": (
                        "Another live session owns writes for this investigation. "
                        "Continue there, or retry after it exits."
                    ),
                }
            if investigation_id not in self._runtimes:
                runtime = self._build_runtime(self.registry.load(investigation_id))
                self._runtimes[investigation_id] = runtime
            else:
                runtime = self._runtimes[investigation_id]
            meta = runtime.meta
            expected = arguments.get("expected_revision")
            current = int(meta.get("revision") or 0)
            if int(expected) != current:
                return {
                    "status": "conflict", "reason": "revision_mismatch",
                    "expected_revision": expected, "current_revision": current,
                    "note": (
                        "State changed since your last brief. Call "
                        "investigation_status and retry."
                    ),
                }
            runtime.state.turn += 1
            runtime.workspace.set_iteration(runtime.state.turn)
            runtime.executor.set_iteration(runtime.state.turn)
        elif not holder:
            runtime = self._build_runtime(document)

        # Step 6: experiment poll (holder only).
        transitioned: list[str] = []
        if self.registry.holds_lease(investigation_id) and runtime is self._runtimes.get(investigation_id):
            transitioned = runtime.queue.poll(runtime.state, runtime.state.turn) or []

        # Step 7: execute.
        if name == "investigation_status":
            body: dict = {}
            did_mutate = False
        elif name == "request_independent_verification":
            body, did_mutate = verify.run_independent_verification(
                runtime, self.verification_available, str(arguments.get("branch") or "")
            )
        else:
            body, did_mutate = self._execute(name, arguments, runtime)

        # Step 8: commit (mutation OR a poll transition), then inject revision.
        needs_commit = did_mutate or bool(transitioned)
        if needs_commit:
            runtime.host.sync_budget()
            document = {"schema_version": 1, **runtime.persist_dict()}
            final_rev = self.registry.commit(investigation_id, document)
        else:
            final_rev = int(runtime.meta.get("revision") or 0)

        if name == "investigation_status":
            body = self._status_body(runtime, final_rev)
        if is_mutate:
            body["revision"] = final_rev
        return body

    # --------------------------------------------------------------- execution
    def _execute(self, name: str, arguments: dict, runtime: InvestigationRuntime) -> tuple[dict, bool]:
        if name in _HANDLE_TOOL_MAP:
            body = self._via_host(runtime, _HANDLE_TOOL_MAP[name], arguments)
            if name == "hypothesis_next_steps":
                return ({
                    "advisory": True, "policy_ids": ["WF-1"], "suggestions": body,
                    "note": "Recommendations, not requirements (policy WF-1, advisory in MCP v1).",
                }, False)
            if name == "meta_declare_solution":
                self._finalize_solution(runtime, body)
                return (body, True)
            if name == "meta_declare_unsolved":
                self._finalize_unsolved(runtime, body)
                return (body, True)
            return (body, manifest.TOOL_CLASSES[name] == "mutate")
        if name == "observe_overview":
            return (self._overview(runtime), False)
        if name == "candidate_list":
            return (self._candidate_list(runtime, arguments), False)
        if name == "candidate_show":
            return (self._candidate_show(runtime, arguments), False)
        if name == "reading_record":
            return (self._reading_record(runtime, arguments), True)
        if name == "comparison_record":
            return (self._comparison_record(runtime, arguments), True)
        if name == "repair_hypotheses_test":
            return (
                repair.compile_hypotheses(runtime, runtime.records, arguments, runtime.state.turn),
                True,
            )
        if name == "repair_transaction":
            return (
                repair.dispatch_repair_transaction(runtime, runtime.records, arguments, runtime.state.turn),
                True,
            )
        raise RuntimeError(f"unrouted tool {name}")

    def _via_host(self, runtime: InvestigationRuntime, v2_name: str, arguments: dict) -> dict:
        inp = {
            k: v for k, v in arguments.items()
            if k not in {"investigation_id", "expected_revision"}
        }
        tu = {"id": f"mcp_{uuid.uuid4().hex[:8]}", "name": v2_name, "input": inp}
        return json.loads(runtime.host.handle_tool(tu, runtime.state.turn))

    # --------------------------------------------------------------- start/list
    def _start(self, arguments: dict) -> dict:
        text = arguments.get("ciphertext")
        if not text or not str(text).strip():
            return {"status": "error", "reason": "empty_ciphertext"}
        if len(str(text)) > _MAX_CIPHERTEXT:
            return {"status": "error", "reason": "ciphertext_too_large"}
        fmt = arguments.get("format") or "auto"
        try:
            ct = intake.build_cipher_text(str(text), fmt)
        except Exception as exc:  # noqa: BLE001 - typed parse failure
            return {"status": "error", "reason": "parse_failed", "error": str(exc)}

        language = arguments.get("language") or "en"
        investigation_id = uuid.uuid4().hex[:12]
        workspace = Workspace(cipher_text=ct)
        state = InvestigationState(workspace=workspace, language=language)
        state.external_context = arguments.get("context") or ""
        fp = cipher_id_analysis.compute_cipher_fingerprint(
            ct.tokens, ct.alphabet.size, language=language,
            word_group_count=len(ct.words),
        )
        state.add_evidence(
            "diagnostic_preflight", turn=0,
            summary=(
                f"alphabet={ct.alphabet.size} tokens="
                f"{len(ct.tokens)} words={len(ct.words)}"
            ),
            fingerprint=fp.to_dict(),
            suspicion_scores=fp.to_dict().get("suspicion_scores") or {},
        )
        state.turn = 0

        now = time.time()
        meta = {
            "investigation_id": investigation_id,
            "label": arguments.get("label") or "",
            "created_at": now, "updated_at": now, "revision": 1,
            "status": "active", "language": language,
            "alphabet_size": ct.alphabet.size, "tokens": len(ct.tokens),
            "words": len(ct.words), "client_name": self.client_name,
            "terminal": None,
        }
        self.registry.create(meta=meta, state_dict=state.to_artifact_dict())
        self.registry.acquire_lease(investigation_id)
        runtime = self._build_runtime(self.registry.load(investigation_id))
        self._runtimes[investigation_id] = runtime
        brief = build_status_brief(
            runtime.state, runtime.executor, runtime.host,
            turn=0, revision=1, verification_available=self.verification_available,
            max_cost_usd=self.max_cost_usd, records=runtime.records,
        )
        return {
            "investigation_id": investigation_id, "revision": 1,
            "language": language, "alphabet_size": ct.alphabet.size,
            "tokens": len(ct.tokens), "words": len(ct.words), "brief": brief,
        }

    def _list(self, arguments: dict) -> dict:
        limit = arguments.get("limit")
        limit = 50 if limit is None else min(int(limit), 50)
        stubs = []
        for meta in self.registry.list(limit=limit):
            stub = {k: v for k, v in meta.items() if k != "terminal"}
            stub["has_terminal"] = meta.get("terminal") is not None
            stubs.append(stub)
        return {"investigations": stubs, "server_code": _server_code_info()}

    # ------------------------------------------------------------------- reads
    def _status_body(self, runtime: InvestigationRuntime, revision: int) -> dict:
        brief = build_status_brief(
            runtime.state, runtime.executor, runtime.host,
            turn=runtime.state.turn, revision=revision,
            verification_available=self.verification_available,
            max_cost_usd=self.max_cost_usd, records=runtime.records,
        )
        return {
            "investigation_id": runtime.meta["investigation_id"],
            "revision": revision, "status": runtime.meta.get("status"),
            "turn": runtime.state.turn,
            "budget": {
                "total_cost_usd": runtime.host.committed_cost(),
                "max_cost_usd": self.max_cost_usd,
            },
            "verification_available": self.verification_available,
            "brief": brief,
        }

    def _overview(self, runtime: InvestigationRuntime) -> dict:
        state = runtime.state
        ws = state.workspace
        final = (
            f"Branches: {len(ws.branch_names())}   Readings: {len(state.readings)}   "
            f"Attestations: {len(state.verify_attestations)}   "
            f"Evidence entries: {len(state.evidence_log)}"
        )
        text = (
            context._render_cipher(state) + "\n\n"
            + context._render_fingerprint(state) + "\n\n" + final
        )
        return {"overview": text}

    # ------------------------------------------------------------- candidates
    def _att_summary(self, att: dict | None) -> dict | None:
        if att is None:
            return None
        return {
            "reader_accepts_as_solution": att.get("reader_accepts_as_solution"),
            "target_language_confidence": att.get("target_language_confidence"),
            "semantic_recoverability": att.get("semantic_recoverability"),
            "damage_scope": att.get("damage_scope"),
            "repairability": att.get("repairability"),
            "created_turn": att.get("created_turn"),
        }

    def _verification_label(self, att: dict | None) -> str:
        if att is not None:
            return "positive" if attestation_is_positive(att) else "negative"
        return "none" if self.verification_available else "unavailable"

    def _candidate_entry(self, runtime: InvestigationRuntime, name: str, roles: dict) -> dict:
        state = runtime.state
        ws = runtime.workspace
        branch = ws.get_branch(name)
        content_hash = host_module._branch_hash(ws, name)
        att = latest_attestation_for_hash(state.verify_attestations, content_hash)
        readings_ct = sum(
            1 for r in state.readings.values()
            if r.get("candidate_content_hash") == content_hash
        )
        return {
            "branch": name,
            "content_hash": content_hash,
            "mapped_count": len(branch.key),
            "scores": runtime.executor._compute_quick_scores(name),
            "roles": [role for role, b in roles.items() if b == name],
            "tags": list(branch.tags),
            "parent": branch.parent,
            "created_turn": branch.created_iteration,
            "attestation": self._att_summary(att),
            "verification": self._verification_label(att),
            "readings": readings_ct,
            "excerpt": _decoded_text_for_panel(ws, name)[:120],
        }

    def _candidate_list(self, runtime: InvestigationRuntime, arguments: dict) -> dict:
        state = runtime.state
        ws = runtime.workspace
        roles = loop_v3._compute_branch_roles(state, runtime.executor, None)
        include_rejected = bool(arguments.get("include_rejected"))
        limit = arguments.get("limit")
        limit = 12 if limit is None else min(int(limit), 50)
        offset = int(arguments.get("offset") or 0)
        entries = []
        for name in ws.branch_names():
            card = state.hypothesis_board.get(name)
            mode_status = card.get("mode_status") if card else None
            if mode_status in {"rejected", "superseded"} and not include_rejected:
                continue
            entries.append(self._candidate_entry(runtime, name, roles))
        total = len(entries)
        return {
            "candidates": entries[offset: offset + limit],
            "total": total, "offset": offset,
            "note": "Scores are individual signals (dict_rate, quad), not a ranking (WF-6).",
        }

    def _candidate_show(self, runtime: InvestigationRuntime, arguments: dict) -> dict:
        state = runtime.state
        ws = runtime.workspace
        branch = str(arguments.get("branch") or "")
        if not ws.has_branch(branch):
            return {"status": "error", "reason": "unknown_branch"}
        roles = loop_v3._compute_branch_roles(state, runtime.executor, None)
        entry = self._candidate_entry(runtime, branch, roles)
        content_hash = entry["content_hash"]
        max_chars = arguments.get("max_chars")
        max_chars = 4000 if max_chars is None else min(int(max_chars), 20000)
        decoded = context._truncate(_decoded_text_for_panel(ws, branch), max_chars)
        attestation_history = [
            dict(a) for a in state.verify_attestations
            if a.get("content_hash") == content_hash
        ]
        readings = []
        for r in state.readings.values():
            if r.get("candidate_content_hash") != content_hash:
                continue
            frags = r.get("fragments") or []
            preview = " ".join(
                str(f.get("text") or "") for f in frags if isinstance(f, dict)
            ).strip()[:120]
            readings.append({
                "reading_id": r.get("reading_id"),
                "confidence": r.get("overall_confidence"),
                "preview": preview,
            })
        repair_txs = [
            {
                "transaction_id": t.get("transaction_id"), "status": t.get("status"),
                "reason": t.get("reason"), "installed_branch": t.get("installed_branch"),
            }
            for t in state.repair_transactions
            if t.get("source_branch") == branch or t.get("installed_branch") == branch
        ]
        return {
            **entry, "decoded_text": decoded,
            "attestation_history": attestation_history,
            "readings": readings, "repair_transactions": repair_txs,
        }

    # --------------------------------------------------------------- records
    def _reading_record(self, runtime: InvestigationRuntime, arguments: dict) -> dict:
        state = runtime.state
        ws = runtime.workspace
        branch = str(arguments.get("branch") or "")
        if not ws.has_branch(branch):
            return {"status": "error", "reason": "unknown_branch"}
        content_hash = host_module._branch_hash(ws, branch)
        att_key = attestation_key(
            latest_attestation_for_hash(state.verify_attestations, content_hash)
        )
        sat_entry = state.repair_saturation.get(saturation_key(content_hash, att_key))
        if sat_entry is not None and int(sat_entry.get("readings") or 0) >= 1:
            existing = max(
                (
                    r for r in state.readings.values()
                    if r.get("candidate_content_hash") == content_hash
                ),
                key=lambda r: (int(r.get("created_turn") or 0), str(r.get("reading_id") or "")),
                default=None,
            )
            return {
                "status": "blocked", "reason": "duplicate_reading_suppressed",
                "branch": branch, "content_hash": content_hash,
                "existing_reading_id": (str(existing.get("reading_id")) if existing else None),
                "note": (
                    "A reading already exists for this exact content and verifier "
                    "evidence. Reuse it via repair_transaction, or produce new "
                    "content or new verifier evidence first."
                ),
            }
        result = {
            "reading_text": arguments.get("reading_text"),
            "fragments": arguments.get("fragments") or [],
            "holes": arguments.get("holes") or [],
            "overall_confidence": arguments.get("overall_confidence"),
        }
        errors = episodes.validate_against_schema(result, episodes._READING_SCHEMA)
        if errors:
            return {"status": "error", "reason": "invalid_reading", "errors": errors}
        packet = build_candidate_reading_packet(ws, branch).to_dict()
        reading = Reading.from_episode_result(
            result, branch=branch, source=f"client:{self.client_name}",
            created_turn=state.turn, candidate_packet=packet,
        )
        state.readings[reading.reading_id] = reading.to_dict()
        sat = get_or_create_saturation_entry(state, content_hash, att_key, state.turn)
        sat["readings"] = int(sat.get("readings") or 0) + 1
        sat["updated_turn"] = state.turn
        return {
            "status": "ok", "reading_id": reading.reading_id, "branch": branch,
            "content_hash": content_hash, "fragment_count": len(reading.fragments),
        }

    def _comparison_record(self, runtime: InvestigationRuntime, arguments: dict) -> dict:
        state = runtime.state
        ws = runtime.workspace
        branches = list(arguments.get("branches") or [])
        ranking = list(arguments.get("ranking") or [])
        best_partial = arguments.get("best_partial")
        if not 2 <= len(branches) <= 8:
            return {"status": "error", "reason": "invalid_branch_count"}
        for b in branches:
            if not ws.has_branch(str(b)):
                return {"status": "error", "reason": "unknown_branch", "branch": b}
        if len(set(ranking)) != len(ranking) or any(r not in branches for r in ranking):
            return {"status": "error", "reason": "invalid_ranking"}
        if best_partial is not None and best_partial not in branches:
            return {"status": "error", "reason": "invalid_best_partial"}
        branch_hashes = {name: host_module._branch_hash(ws, name) for name in branches}
        record = {
            "comparison_id": uuid.uuid4().hex[:12], "created_turn": state.turn,
            "client": self.client_name, "branches": branches,
            "branch_hashes": branch_hashes, "ranking": ranking,
            "best_partial": best_partial,
            "best_partial_hash": (branch_hashes.get(best_partial) if best_partial else None),
            "accepts_as_solution": bool(arguments.get("accepts_as_solution")),
            "rationale": str(arguments.get("rationale") or ""),
        }
        runtime.records.setdefault("comparisons", []).append(record)
        state.add_evidence(
            "client_comparison", turn=state.turn,
            summary=f"client ranked {ranking[:3]}...",
            comparison_id=record["comparison_id"],
        )
        return dict(record)

    # ------------------------------------------------------------ terminals
    def _finalize_solution(self, runtime: InvestigationRuntime, body: dict) -> None:
        executor = runtime.executor
        if not (executor.terminated and executor.solution is not None):
            return
        sol = executor.solution
        if runtime.workspace.has_branch(sol.branch):
            declared_hash = _candidate_content_hash(
                _decoded_text_for_panel(runtime.workspace, sol.branch)
            )
            match = max(
                (
                    a for a in runtime.state.verify_attestations
                    if a.get("content_hash") == declared_hash
                ),
                key=lambda a: (int(a.get("created_turn") or 0), str(a.get("episode_id") or "")),
                default=None,
            )
            if match is not None:
                sol.attestation = dict(match)
        runtime.meta["status"] = "solved"
        runtime.meta["terminal"] = {
            "kind": "solution", "declared_turn": runtime.state.turn,
            "solution": dataclasses.asdict(executor.solution),
        }
        body["terminal_status"] = "solved"

    def _finalize_unsolved(self, runtime: InvestigationRuntime, body: dict) -> None:
        executor = runtime.executor
        if not (executor.terminated and getattr(executor, "unsolved_declaration", None) is not None):
            return
        runtime.meta["status"] = "unsolved"
        runtime.meta["terminal"] = {
            "kind": "unsolved", "declared_turn": runtime.state.turn,
            "declaration": dict(executor.unsolved_declaration),
        }
        body["terminal_status"] = "unsolved"
