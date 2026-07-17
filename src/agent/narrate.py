"""NarrateAgentRenderer — a human-first, Claude-Code-style transcript renderer.

This is the default agentic CLI display (see ``cli._resolve_agent_display``). It
is a plain-text SCROLLING transcript (not a live dashboard). CLI-3 reversed the
priority: the DEFAULT output is a clear human NARRATIVE of what the agent is
doing, why, what happened, and where the problems are.

Default (non-verbose) line vocabulary:
- ``“ `` full agent narration (never truncated in the renderer), cyan.
- ``⏺ `` plain-English action line per lead tool call (``describe_tool_action``).
- ``  ↳ `` plain-English result/outcome line under the action (dim; yellow when
  the result is a problem).
- ``! `` problem lines — ALWAYS shown (never verbose-gated).
- ``⤷ `` workflow-phase transitions.
- the changed-only decode preview and the cumulative ``[$cost | tokens]`` ticker.
- ``finish`` prints an analyzer-grade run digest computed purely from streamed
  events (the renderer never reads the artifact file).

Verbose (``-v`` / ``renderer_verbose``) ADDS, strictly additively, the numbered
precise ``N │ tool(args)`` line under each action, the episode-internal per-tool
lines, snapshot/budget/roles lines, and full decode previews. Verbose never
REPLACES a default line.

Optional ANSI color is applied only when the output stream is an interactive
TTY. The renderer reuses ``describe_tool_action`` + ``summarize_tool_call`` from
display.py. ``pretty``/``raw``/``jsonl`` displays are untouched; v2 event streams
render through the same handlers (all new payload reads use ``.get`` fallbacks).
"""
from __future__ import annotations

import sys
import textwrap
from collections import Counter
from typing import Any

from agent.display import describe_tool_action, summarize_tool_call


# Lead tools that spawn forwarded episode_* children (nested ↳ lines). Their
# action line is printed at tool_start so the children nest UNDER it (F-1).
# Everything else prints its action line at tool_call (with its result).
_PARENT_TOOLS = {"episode_run", "experiment_submit", "experiment_collect"}

# Problem events whose count feeds the finish-digest ``problems`` line, mapped to
# a singular human label. (best_effort/declared terminals feed ``outcome``.)
_PROBLEM_LABELS = {
    "rate_limit_retry": "rate-limit retry",
    "lead_truncation_retry": "truncation retry",
    "lead_tool_rejected": "rejected lead tool",
    "no_tool_calls_nudge": "no-tool nudge",
    "no_tool_calls": "tool-less terminal",
    "boundary_projection_count_retry": "reading-length retry",
    "gated_tool_retry": "gated-tool retry",
    "cost_ceiling_reached": "cost ceiling reached",
    "max_iterations_reached": "turn limit reached",
    "error": "provider error",
    "interrupted": "interruption",
}


# --- ANSI helpers (no-op unless the stream is a real TTY) --------------------
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
}


def _stream_is_tty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001
        return False


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    if seconds >= 60:
        minutes = int(seconds // 60)
        rem = int(round(seconds - minutes * 60))
        return f"{minutes}m{rem:02d}s"
    return f"{seconds:.1f}s"


def _format_tokens(total_tokens: int) -> str:
    tokens = max(0, int(total_tokens or 0))
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.2f}M"
    if tokens >= 1000:
        return f"{tokens / 1000:.0f}k"
    return str(tokens)


def _pluralize(label: str, n: int) -> str:
    if n <= 1:
        return label
    if label.endswith("y"):
        return label[:-1] + "ies"
    return label + "s"


def _compact_args(args: dict[str, Any] | None, *, verbose: bool) -> str:
    """A short ``key=val, key=val`` rendering of tool arguments.

    Non-verbose: at most three keys, each value truncated. Verbose: every key,
    values truncated less aggressively. Deterministic key order (insertion).
    """
    if not isinstance(args, dict) or not args:
        return ""
    items = list(args.items())
    limit = len(items) if verbose else 3
    val_cap = 60 if verbose else 24
    parts: list[str] = []
    for key, value in items[:limit]:
        text = _stringify(value)
        if len(text) > val_cap:
            text = text[: val_cap - 1] + "…"
        parts.append(f"{key}={text}")
    if not verbose and len(items) > limit:
        parts.append("…")
    return ", ".join(parts)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stringify(v) for v in list(value)[:4]) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(str(k) for k in list(value.keys())[:4]) + "}"
    return str(value)


class NarrateAgentRenderer:
    """Human-first scrolling transcript renderer (default agentic display)."""

    def __init__(self, stream: Any = None, *, verbose: bool = False) -> None:
        self.stream = stream or sys.stdout
        self.verbose = verbose
        self._color = _stream_is_tty(self.stream)
        # Running state.
        self._tool_index = 0
        self._iteration = 0
        self._last_tokens = 0
        self._last_cost = 0.0
        self._last_decode = ""
        self._pending_tool: dict[str, Any] | None = None
        self._episode_labels: dict[str, str] = {}
        self._episode_counter = 0
        # -- finish-digest accumulators (computed purely from streamed events) --
        self._problem_counts: dict[str, int] = {}
        self._episode_records: list[tuple[str, str]] = []
        self._repairs_installed = 0
        self._repairs_rejected = 0
        self._experiments_submitted = 0
        self._last_verify_digest = ""
        self._last_branch_roles: dict[str, Any] | None = None
        self._workflow_phase = ""
        self._declared_branch: Any = None
        self._auto_declared: Any = None
        self._best_effort: dict[str, Any] | None = None
        self._declared_unsolved_branch: Any = None
        self._declared_unsolved_seen = False

    # -- protocol -------------------------------------------------------------
    def start_test(
        self,
        test_id: str,
        description: str,
        *,
        model: str,
        max_iterations: int,
        language: str | None = None,
        source: str | None = None,
        agent_loop: str | None = None,
    ) -> None:
        tags = ", ".join(t for t in (source, language) if t)
        header = self._c(f"▶ {test_id}", "bold")
        if tags:
            header += f"  ({tags})"
        if model:
            header += f"  model={model}"
        if agent_loop:
            header += f"  loop={agent_loop}"
        header += f"  max_iter={max_iterations}"
        self._line(header)
        if description:
            self._line(self._c(f"  {description}", "dim"))

    def event(self, event: str, payload: dict[str, Any]) -> None:
        payload = payload or {}
        self._accumulate(event, payload)
        handler = getattr(self, f"_on_{event}", None)
        if handler is not None:
            handler(payload)

    def finish(self, result: Any) -> None:
        status = str(getattr(result, "status", "") or "")
        char = float(getattr(result, "char_accuracy", 0.0) or 0.0)
        word = float(getattr(result, "word_accuracy", 0.0) or 0.0)
        iters = getattr(result, "iterations_used", None)
        cost = float(getattr(result, "estimated_cost_usd", 0.0) or 0.0)
        elapsed = getattr(result, "elapsed_seconds", 0.0)
        artifact_path = getattr(result, "artifact_path", "") or ""
        error_message = str(getattr(result, "error_message", "") or "")
        # F-4: `crack` has no ground truth — suppress the misleading char/word
        # 0.0% accuracy line for it. Benchmark/resume results omit this flag
        # (getattr default True) and keep showing accuracies.
        has_ground_truth = bool(getattr(result, "has_ground_truth", True))

        self._line(self._c("  ── result ──", "dim"))
        parts = [f"status={status}"]
        if has_ground_truth:
            parts.append(f"char={char:.1%}")
            parts.append(f"word={word:.1%}")
        if iters is not None:
            parts.append(f"iters={iters}")
        parts.append(f"${cost:.2f}")
        parts.append(f"({_format_elapsed(elapsed)})")
        self._line("  " + "  ".join(parts))
        # Always show the final plaintext (the run's actual product). First
        # 200 chars at non-verbose; full text when verbose. `crack` prints its
        # own final-decryption block after finish and passes no
        # final_decryption here, so there is no duplication.
        final_decryption = str(getattr(result, "final_decryption", "") or "").strip()
        if final_decryption:
            shown = (
                final_decryption
                if self.verbose
                else final_decryption[:200]
                + ("…" if len(final_decryption) > 200 else "")
            )
            self._line(self._c("  plaintext:", "cyan"))
            for i in range(0, len(shown), 100):
                self._line(self._c(f"    {shown[i:i + 100]}", "cyan"))
        if artifact_path:
            self._line(f"  artifact: {artifact_path}")
        if error_message:
            self._line(self._c(f"  ! {error_message}", "red"))
        if self.verbose:
            summary = str(getattr(result, "final_summary", "") or "").strip()
            if summary:
                self._line(self._c("  final summary:", "dim"))
                for ln in summary.splitlines():
                    self._line(f"    {ln}")
        # CLI-3: the analyzer-grade run digest (default AND verbose).
        self._render_digest(result)

    # -- event handlers -------------------------------------------------------
    def _on_preflight_start(self, payload: dict[str, Any]) -> None:
        self._line(self._c("  · preflight (no-LLM) …", "dim"))

    def _on_preflight_result(self, payload: dict[str, Any]) -> None:
        solver = payload.get("solver") or "automated"
        status = payload.get("status", "unknown")
        elapsed = float(payload.get("elapsed_seconds", 0.0) or 0.0)
        self._line(
            self._c(
                f"  · preflight: {solver} → {status}  "
                f"({elapsed:.1f}s, $0.00)",
                "dim",
            )
        )

    def _on_iteration_start(self, payload: dict[str, Any]) -> None:
        self._iteration = int(payload.get("iteration") or 0)

    def _on_agent_text(self, payload: dict[str, Any]) -> None:
        # CLI-3: FULL narration in both modes — this is the primary display
        # line. Wrap at ~100 cols; preserve paragraph breaks. Never truncated
        # here (the emit-site cap is 4000 chars).
        text = str(payload.get("text") or "").strip()
        if not text:
            return
        for para in text.split("\n"):
            para = para.strip()
            if not para:
                continue
            for ln in textwrap.wrap(para, width=100) or [para]:
                self._line(self._c(f"  “ {ln}", "cyan"))

    def _on_tool_start(self, payload: dict[str, Any]) -> None:
        tool = str(payload.get("tool") or "tool")
        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        self._pending_tool = {"tool": tool, "arguments": args, "printed": False}
        # F-1: a PARENT tool (one that spawns forwarded episode_* children) must
        # print its action line NOW — the ↳ children arrive between tool_start
        # and tool_call, so deferring the action to tool_call would invert the
        # nesting (children above their parent). Regular tools print at
        # tool_call (with their result).
        if tool in _PARENT_TOOLS:
            self._print_action_line(tool, args)
            self._pending_tool["printed"] = True

    def _on_tool_call(self, payload: dict[str, Any]) -> None:
        tool = str(payload.get("tool") or "tool")
        pending = None
        if self._pending_tool and self._pending_tool.get("tool") == tool:
            pending = self._pending_tool
        args = (pending or {}).get("arguments") or {}
        already_printed = bool(pending and pending.get("printed"))
        self._pending_tool = None
        result_summary = payload.get("result_summary") or {}
        summary = self._result_summary(tool, result_summary)
        is_problem = str(result_summary.get("status") or "") == "blocked"
        if not already_printed:
            self._print_action_line(tool, args)
        # Result/outcome line under the action. For parent tools the
        # episode_complete ↳ digest already carries the outcome, so an empty
        # tool_call result prints nothing extra.
        if summary:
            color = "yellow" if is_problem else "dim"
            self._line(self._c(f"    ↳ {summary}{self._ticker()}", color))

    def _print_action_line(self, tool: str, args: dict[str, Any]) -> None:
        """Default ``⏺ action`` line; verbose ADDS the numbered precise line."""
        action = describe_tool_action(tool, args)
        self._line(f"  ⏺ {action}")
        if self.verbose:
            self._tool_index += 1
            arg_str = _compact_args(args, verbose=True)
            self._line(self._c(f"  {self._tool_index} │ {tool}({arg_str})", "dim"))

    def _on_workspace_snapshot(self, payload: dict[str, Any]) -> None:
        self._last_tokens = int(payload.get("total_tokens") or self._last_tokens)
        self._last_cost = float(payload.get("estimated_cost_usd") or self._last_cost)
        if self.verbose:
            branch = payload.get("branch")
            scores = payload.get("scores") or {}
            self._line(
                self._c(
                    f"      snapshot: best={branch} "
                    f"dict={scores.get('dict_rate')} quad={scores.get('quad')}"
                    f"{self._ticker()}",
                    "dim",
                )
            )
            # M5.3 Slice 7: distinguish the four branch roles when any of the
            # non-best roles is present and diverges from the best-scored branch.
            roles = payload.get("branch_roles")
            if isinstance(roles, dict):
                best = roles.get("best_scored_branch")
                workflow = roles.get("workflow_branch")
                installed = roles.get("latest_installed_branch")
                selected = roles.get("declared_or_selected_branch")
                if any(
                    value is not None and value != best
                    for value in (workflow, installed, selected)
                ):
                    self._line(
                        self._c(
                            f"      roles: workflow={workflow or '-'} "
                            f"installed={installed or '-'} "
                            f"selected={selected or '-'}",
                            "dim",
                        )
                    )
        # Live progress: render the current best decode whenever it CHANGES
        # (the user's core "show me the work" signal). One line at
        # non-verbose (first 140 chars); the full preview when verbose.
        decode = str(
            payload.get("decryption_preview") or payload.get("decryption") or ""
        ).strip()
        if decode and decode != self._last_decode:
            self._last_decode = decode
            branch = payload.get("branch") or "?"
            if self.verbose:
                self._line(self._c(f"      decode [{branch}]:", "cyan"))
                for i in range(0, min(len(decode), 1000), 100):
                    self._line(self._c(f"        {decode[i:i + 100]}", "cyan"))
            else:
                preview = decode[:140] + ("…" if len(decode) > 140 else "")
                self._line(self._c(f"      decode [{branch}] {preview}", "cyan"))

    def _on_budget_update(self, payload: dict[str, Any]) -> None:
        total = payload.get("total_cost_usd")
        if total is not None:
            self._last_cost = float(total or 0.0)
        tokens = payload.get("total_tokens")
        if tokens is not None:
            self._last_tokens = int(tokens or 0)
        if self.verbose:
            cats = payload.get("budget_by_category") or {}
            names = ", ".join(sorted(cats.keys()))
            self._line(self._c(f"      budget: {names}{self._ticker()}", "dim"))

    # -- episode internals (forwarded from run_episode) — VERBOSE ONLY --------
    def _on_episode_turn_start(self, payload: dict[str, Any]) -> None:
        if self.verbose:
            self._line(self._episode_prefix(payload) + " turn start")

    def _on_episode_tool_call(self, payload: dict[str, Any]) -> None:
        if not self.verbose:
            return
        tool = str(payload.get("tool") or "tool")
        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        arg_str = _compact_args(args, verbose=True)
        self._line(f"{self._episode_prefix(payload)} │ {tool}({arg_str})")

    def _on_episode_submit(self, payload: dict[str, Any]) -> None:
        if not self.verbose:
            return
        accepted = payload.get("accepted")
        status = "ok" if accepted else str(payload.get("status") or "retry")
        self._line(
            f"{self._episode_prefix(payload)} │ episode_submit → {status}"
        )

    def _on_episode_complete(self, payload: dict[str, Any]) -> None:
        kind = str(payload.get("kind") or "episode")
        digest = payload.get("digest")
        calls = payload.get("calls")
        spend = payload.get("spend_usd")
        extra = []
        if calls is not None:
            extra.append(f"calls={calls}")
        if spend is not None:
            extra.append(f"${float(spend):.2f}")
        suffix = f"  ({', '.join(extra)})" if extra else ""
        if digest:
            body = f"{kind}: {digest}"
        else:
            # v2 / older artifacts without a digest: fall back to the status.
            body = f"{kind} → {payload.get('status', 'ok')}"
        failed = bool(digest) and str(digest).startswith(f"{kind} failed")
        self._line(
            self._c(f"    ↳ {body}{suffix}", "yellow" if failed else "dim")
        )

    # -- declaration ----------------------------------------------------------
    def _on_declared_solution(self, payload: dict[str, Any]) -> None:
        branch = payload.get("branch")
        conf = payload.get("confidence")
        self._line(
            self._c(
                f"  ✓ DECLARED solution on branch {branch} "
                f"(confidence: {conf})",
                "green",
            )
        )
        attestation = payload.get("attestation")
        if isinstance(attestation, dict):
            coherence = attestation.get("coherence")
            anomalies = attestation.get("anomalies") or []
            n = len(anomalies) if isinstance(anomalies, list) else anomalies
            if "reader_accepts_as_solution" in attestation:
                accept_word = (
                    "reader accepts as solution"
                    if attestation.get("reader_accepts_as_solution")
                    else "reader does not accept as solution"
                )
            else:
                accepts = attestation.get("reader_accepts")
                accept_word = "reader accepts" if accepts else "reader rejects"
            self._line(
                self._c(
                    f"      attestation: coherence {coherence}/10, "
                    f"{accept_word}, {n} anomaly(ies)",
                    "dim",
                )
            )

    def _on_declared_unsolved(self, payload: dict[str, Any]) -> None:
        branch = payload.get("best_branch")
        self._line(self._c(f"  ✗ declared UNSOLVED (best branch: {branch})", "yellow"))

    def _on_auto_declared_solution(self, payload: dict[str, Any]) -> None:
        branch = payload.get("branch")
        self._line(
            self._c(f"  ✗ auto-declared fallback on branch {branch}", "yellow")
        )

    # -- problem handlers (ALWAYS shown; yellow unless noted) -----------------
    def _on_rate_limit_retry(self, payload: dict[str, Any]) -> None:
        self._line(self._c(
            f"  ! rate limit — waiting {float(payload.get('delay_seconds') or 0):.0f}s "
            f"before retry (attempt {payload.get('attempt')})",
            "yellow",
        ))

    def _on_lead_tool_rejected(self, payload: dict[str, Any]) -> None:
        tool = payload.get("tool") or "?"
        self._line(self._c(
            f"  ! tool '{tool}' not available here — delegate via an "
            "episode/experiment",
            "yellow",
        ))

    def _on_no_tool_calls_nudge(self, payload: dict[str, Any]) -> None:
        self._line(self._c(
            "  ! agent narrated without acting — nudged to use a tool",
            "yellow",
        ))

    def _on_no_tool_calls(self, payload: dict[str, Any]) -> None:
        self._line(self._c(
            "  ! agent produced no tool call — ending run (exhausted)",
            "yellow",
        ))

    def _on_lead_truncation_retry(self, payload: dict[str, Any]) -> None:
        self._line(self._c(
            "  ! output budget hit mid-reasoning — retrying with "
            f"{payload.get('new_max_tokens')} tokens",
            "yellow",
        ))

    def _on_cost_ceiling_reached(self, payload: dict[str, Any]) -> None:
        try:
            ceiling = float(payload.get("max_cost_usd") or 0.0)
        except (TypeError, ValueError):
            ceiling = 0.0
        self._line(self._c(
            f"  ! cost ceiling ${ceiling:.2f} reached — ending run honestly",
            "red",
        ))

    def _on_max_iterations_reached(self, payload: dict[str, Any]) -> None:
        self._line(self._c(
            "  ! turn limit reached without a declaration", "yellow"
        ))

    def _on_best_effort_selected(self, payload: dict[str, Any]) -> None:
        tier = payload.get("selection_tier") or "best"
        branch = payload.get("branch")
        self._line(self._c(
            f"  ✗ best-effort fallback: {tier} on {branch}", "yellow"
        ))

    def _on_no_new_information(self, payload: dict[str, Any]) -> None:
        self._line(self._c("  · no new information this turn", "dim"))

    def _on_repeated_call(self, payload: dict[str, Any]) -> None:
        if self.verbose:
            self._line(self._c(
                f"      · repeated call: {payload.get('tool')} "
                f"(x{payload.get('count')})",
                "dim",
            ))

    def _on_duplicate_read_suppressed(self, payload: dict[str, Any]) -> None:
        if self.verbose:
            self._line(self._c(
                f"      · duplicate read suppressed: {payload.get('tool')}",
                "dim",
            ))

    def _on_post_terminate_tools_skipped(self, payload: dict[str, Any]) -> None:
        if self.verbose:
            skipped = ", ".join(payload.get("skipped") or [])
            self._line(self._c(
                f"      · skipped after termination: {skipped}", "dim"
            ))

    def _on_repair_transaction_complete(self, payload: dict[str, Any]) -> None:
        status = str(payload.get("status") or "")
        failed = status not in {"installed", "ok"}
        if status == "installed":
            detail = f"installed {payload.get('installed_branch') or '?'}"
        else:
            detail = str(payload.get("reason") or status or "failed")
            failure_class = payload.get("failure_class")
            if failure_class:
                detail += f" ({failure_class})"
        self._line(self._c(
            f"    ↳ repair {status}: {detail}",
            "yellow" if failed else "dim",
        ))

    def _on_workflow_state_changed(self, payload: dict[str, Any]) -> None:
        frm = payload.get("from") or "-"
        to = payload.get("to") or "-"
        branch = payload.get("branch") or "?"
        self._line(self._c(
            f"    ⤷ workflow: {frm} → {to} (on {branch})", "dim"
        ))

    def _on_error(self, payload: dict[str, Any]) -> None:
        self._line(self._c(f"  ! ERROR: {payload.get('message', 'error')}", "red"))

    def _on_interrupted(self, payload: dict[str, Any]) -> None:
        self._line(self._c(f"  ! interrupted: {payload.get('message', '')}", "yellow"))

    def _on_gated_tool_retry(self, payload: dict[str, Any]) -> None:
        attempted = ", ".join(payload.get("attempted_tools") or [])
        self._line(self._c(f"  ! gated tool rejected ({attempted}); retrying", "yellow"))

    def _on_boundary_projection_count_retry(self, payload: dict[str, Any]) -> None:
        self._line(self._c("  ! reading proposal length mismatch; retrying", "yellow"))

    # -- finish digest --------------------------------------------------------
    def _accumulate(self, event: str, payload: dict[str, Any]) -> None:
        """Track events for the finish digest (purely from the stream)."""
        if event == "episode_complete":
            kind = str(payload.get("kind") or "episode")
            self._episode_records.append((kind, str(payload.get("status") or "ok")))
            if kind == "verify" and payload.get("digest"):
                self._last_verify_digest = str(payload.get("digest"))
        elif event == "repair_transaction_complete":
            status = str(payload.get("status") or "")
            if status == "installed":
                self._repairs_installed += 1
            else:
                self._repairs_rejected += 1
        elif event == "workspace_snapshot":
            roles = payload.get("branch_roles")
            if isinstance(roles, dict):
                self._last_branch_roles = roles
        elif event == "workflow_state_changed":
            self._workflow_phase = str(payload.get("to") or "")
        elif event == "tool_call" and str(payload.get("tool")) == "experiment_submit":
            self._experiments_submitted += 1
        elif event == "declared_solution":
            self._declared_branch = payload.get("branch")
        elif event == "auto_declared_solution":
            self._auto_declared = payload.get("branch")
        elif event == "best_effort_selected":
            self._best_effort = dict(payload)
        elif event == "declared_unsolved":
            self._declared_unsolved_seen = True
            self._declared_unsolved_branch = payload.get("best_branch")
        if event in _PROBLEM_LABELS:
            self._problem_counts[event] = self._problem_counts.get(event, 0) + 1

    def _render_digest(self, result: Any) -> None:
        self._line(self._c("  ── digest ──", "dim"))
        self._line(f"  outcome    : {self._digest_outcome(result)}")
        branch_line = self._digest_branch_line()
        if branch_line:
            self._line(f"  branch     : {branch_line}")
        self._line(f"  attestation: {self._last_verify_digest or 'none'}")
        self._line(f"  episodes   : {self._digest_episodes()}")
        self._line(
            f"  repairs    : {self._repairs_installed} installed, "
            f"{self._repairs_rejected} rejected"
        )
        problems = self._digest_problems()
        if problems:
            self._line(f"  problems   : {problems}")
        artifact_path = getattr(result, "artifact_path", "") or ""
        if artifact_path:
            self._line(
                f"  artifact   : {artifact_path}   (inspect with: python "
                f"scripts/inspect_artifact.py {artifact_path})"
            )

    def _digest_outcome(self, result: Any) -> str:
        status = str(getattr(result, "status", "") or "") or "unknown"
        if self._declared_branch:
            return f"solved — declared on {self._declared_branch}"
        if self._auto_declared:
            return f"{status} — auto-declared fallback on {self._auto_declared}"
        if self._best_effort is not None:
            branch = self._best_effort.get("branch")
            return (
                f"{status} — best-effort branch '{branch}' retained, no "
                "positive attestation (honest terminal)"
            )
        if self._declared_unsolved_seen:
            return f"{status} — declared unsolved (best: {self._declared_unsolved_branch})"
        if "cost_ceiling_reached" in self._problem_counts:
            return f"{status} — cost ceiling reached (honest terminal)"
        if "max_iterations_reached" in self._problem_counts:
            return f"{status} — turn limit reached without a declaration"
        return status

    def _digest_branch_line(self) -> str:
        roles = self._last_branch_roles
        if not isinstance(roles, dict):
            return ""
        best = roles.get("best_scored_branch")
        workflow = roles.get("workflow_branch")
        installed = roles.get("latest_installed_branch")
        selected = roles.get("declared_or_selected_branch")
        extras = []
        if workflow is not None and workflow != best:
            extras.append(f"workflow={workflow}")
        if installed is not None and installed != best:
            extras.append(f"installed={installed}")
        if selected is not None and selected != best:
            extras.append(f"selected={selected}")
        if not extras:
            return ""
        return f"{best}   ({', '.join(extras)})"

    def _digest_episodes(self) -> str:
        if not self._episode_records:
            base = "0"
        else:
            total = len(self._episode_records)
            by_kind: Counter[str] = Counter()
            ok_by_kind: Counter[str] = Counter()
            for kind, status in self._episode_records:
                by_kind[kind] += 1
                if status == "ok":
                    ok_by_kind[kind] += 1
            parts = []
            for kind in sorted(by_kind):
                count = by_kind[kind]
                ok = ok_by_kind[kind]
                if ok == count:
                    parts.append(f"{kind} {count} ok")
                else:
                    parts.append(f"{kind} {count} ({ok} ok)")
            base = f"{total} ({', '.join(parts)})"
        if self._experiments_submitted:
            base += f"   experiments: {self._experiments_submitted} queued"
        return base

    def _digest_problems(self) -> str:
        parts = []
        for event, label in _PROBLEM_LABELS.items():
            n = self._problem_counts.get(event, 0)
            if n:
                parts.append(f"{n} {_pluralize(label, n)}")
        if self._repairs_rejected:
            parts.append(
                f"{self._repairs_rejected} "
                f"{_pluralize('rejected repair', self._repairs_rejected)}"
            )
        return "; ".join(parts)

    # -- helpers --------------------------------------------------------------
    def _result_summary(self, tool: str, result_summary: dict[str, Any]) -> str:
        summary = summarize_tool_call(tool, result_summary)
        # summarize_tool_call prefixes the tool name; strip it (already shown).
        if summary.startswith(tool):
            summary = summary[len(tool):].strip()
        return summary

    def _ticker(self) -> str:
        return self._c(
            f"   [${self._last_cost:.2f} | {_format_tokens(self._last_tokens)} tok]",
            "dim",
        )

    def _episode_label(self, episode_id: Any) -> str:
        key = str(episode_id or "")
        if key not in self._episode_labels:
            self._episode_counter += 1
            self._episode_labels[key] = f"ep{self._episode_counter}"
        return self._episode_labels[key]

    def _episode_prefix(self, payload: dict[str, Any]) -> str:
        kind = str(payload.get("kind") or "episode")
        label = self._episode_label(payload.get("episode_id"))
        turn = payload.get("turn")
        base = f"      ↳ {kind} {label}"
        if turn is not None:
            base += f" │ turn {turn}"
        return self._c(base, "dim")

    def _c(self, text: str, style: str) -> str:
        if not self._color:
            return text
        code = _ANSI.get(style)
        if not code:
            return text
        return f"{code}{text}{_ANSI['reset']}"

    def _line(self, text: str) -> None:
        print(text, file=self.stream, flush=True)
