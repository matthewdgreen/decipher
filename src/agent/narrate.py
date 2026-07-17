"""NarrateAgentRenderer — a scrolling, Claude-Code-style transcript renderer.

This is the default agentic CLI display (see ``cli._resolve_agent_display``). It
is a plain-text SCROLLING transcript (not a live dashboard): one line per lead
tool call, indented ``↳`` lines for episode internals, a bracketed cumulative
cost/token ticker, and distinct declaration / attestation / result blocks. It
works in scrollback, pipes, and CI logs. Optional ANSI color is applied only
when the output stream is an interactive TTY.

The renderer implements the ``AgentRunRenderer`` protocol (display.py). It reuses
``summarize_tool_call`` from display.py for lead-tool result summaries and
``describe_tool_gloss`` for the first-use plain-English tool glosses (CLI-2 Part 1).
"""
from __future__ import annotations

import sys
from typing import Any

from agent.display import describe_tool_gloss, summarize_tool_call


# Lead tools that spawn forwarded episode_* children (nested ↳ lines). Their
# numbered launching line is printed at tool_start so the children nest UNDER it
# (F-1). Everything else stays single-line (printed at tool_call with a result).
_PARENT_TOOLS = {"episode_run", "experiment_submit", "experiment_collect"}


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
    """Scrolling structured transcript renderer (default agentic display)."""

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
        # CLI-2 Part 1: tool names (or episode_run:<kind>) whose plain-English
        # gloss has already been printed this run — first-use-per-run only.
        self._glossed: set[str] = set()

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
        handler = getattr(self, f"_on_{event}", None)
        if handler is not None:
            handler(payload or {})

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
        text = str(payload.get("text") or "").strip()
        if not text:
            return
        if self.verbose:
            for ln in text.splitlines():
                self._line(self._c(f"  “ {ln}", "cyan"))
        else:
            first = " ".join(text.split())
            if len(first) > 120:
                first = first[:119] + "…"
            self._line(self._c(f"  “ {first}", "cyan"))

    def _on_tool_start(self, payload: dict[str, Any]) -> None:
        tool = str(payload.get("tool") or "tool")
        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        self._pending_tool = {"tool": tool, "arguments": args, "printed": False}
        # F-1: a PARENT tool (one that spawns forwarded episode_* children) must
        # print its numbered launching line NOW — the ↳ children arrive between
        # tool_start and tool_call, so deferring the parent line to tool_call
        # would invert the nesting (children above their parent). Regular tools
        # stay single-line (printed at tool_call with their result).
        if tool in _PARENT_TOOLS:
            self._tool_index += 1
            arg_str = _compact_args(args, verbose=self.verbose)
            self._line(f"  {self._tool_index} │ {tool}({arg_str})")
            # Gloss goes directly under the parent line, ABOVE its ↳ children,
            # so the episode nesting is preserved.
            self._emit_gloss(tool, args)
            self._pending_tool["printed"] = True

    def _on_tool_call(self, payload: dict[str, Any]) -> None:
        tool = str(payload.get("tool") or "tool")
        pending = None
        if self._pending_tool and self._pending_tool.get("tool") == tool:
            pending = self._pending_tool
        args = (pending or {}).get("arguments") or {}
        already_printed = bool(pending and pending.get("printed"))
        self._pending_tool = None
        summary = self._result_summary(tool, payload.get("result_summary") or {})
        if already_printed:
            # Parent line + its ↳ children are already on screen; show the result
            # only as a compact continuation UNDER the parent (never a new
            # numbered line), so nesting is preserved. The episode_complete ↳
            # line already reports calls/spend, so keep this minimal.
            if summary:
                self._line(self._c(f"      → {summary}{self._ticker()}", "dim"))
            return
        self._tool_index += 1
        arg_str = _compact_args(args, verbose=self.verbose)
        line = f"  {self._tool_index} │ {tool}({arg_str})"
        if summary:
            line += f" → {summary}"
        line += self._ticker()
        self._line(line)
        self._emit_gloss(tool, args)

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

    # -- episode (forwarded from run_episode via the v3 dispatcher) -----------
    def _on_episode_turn_start(self, payload: dict[str, Any]) -> None:
        # Tracked implicitly; the turn number rides on the tool-call/submit
        # lines below (matches the pinned narrate shape). No standalone line at
        # non-verbose to keep episode blocks compact.
        if self.verbose:
            self._line(self._episode_prefix(payload) + " turn start")

    def _on_episode_tool_call(self, payload: dict[str, Any]) -> None:
        tool = str(payload.get("tool") or "tool")
        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        arg_str = _compact_args(args, verbose=self.verbose)
        self._line(f"{self._episode_prefix(payload)} │ {tool}({arg_str})")

    def _on_episode_submit(self, payload: dict[str, Any]) -> None:
        accepted = payload.get("accepted")
        status = "ok" if accepted else str(payload.get("status") or "retry")
        self._line(
            f"{self._episode_prefix(payload)} │ episode_submit → {status}"
        )

    def _on_episode_complete(self, payload: dict[str, Any]) -> None:
        kind = str(payload.get("kind") or "episode")
        label = self._episode_label(payload.get("episode_id"))
        status = payload.get("status", "ok")
        calls = payload.get("calls")
        spend = payload.get("spend_usd")
        extra = []
        if calls is not None:
            extra.append(f"calls={calls}")
        if spend is not None:
            extra.append(f"${float(spend):.2f}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        self._line(
            self._c(f"      ↳ {kind} {label} → {status}{suffix}", "dim")
        )

    # -- declaration / errors -------------------------------------------------
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

    def _on_error(self, payload: dict[str, Any]) -> None:
        self._line(self._c(f"  ! ERROR: {payload.get('message', 'error')}", "red"))

    def _on_interrupted(self, payload: dict[str, Any]) -> None:
        self._line(self._c(f"  ! interrupted: {payload.get('message', '')}", "yellow"))

    def _on_gated_tool_retry(self, payload: dict[str, Any]) -> None:
        attempted = ", ".join(payload.get("attempted_tools") or [])
        self._line(self._c(f"  ! gated tool rejected ({attempted}); retrying", "yellow"))

    def _on_boundary_projection_count_retry(self, payload: dict[str, Any]) -> None:
        self._line(self._c("  ! reading proposal length mismatch; retrying", "yellow"))

    # -- helpers --------------------------------------------------------------
    def _emit_gloss(self, tool: str, args: dict[str, Any] | None) -> None:
        """Print the tool's plain-English gloss the FIRST time it appears (CLI-2
        Part 1). A dim, indented ``·`` line under the numbered tool line. Keyed
        per tool name, except episode_run which is keyed per ``kind`` so each
        kind's distinct gloss shows once."""
        key = self._gloss_key(tool, args)
        if key in self._glossed:
            return
        self._glossed.add(key)
        gloss = describe_tool_gloss(tool, args or {})
        if gloss:
            self._line(self._c(f"      · {gloss}", "dim"))

    @staticmethod
    def _gloss_key(tool: str, args: dict[str, Any] | None) -> str:
        if tool == "episode_run":
            kind = str((args or {}).get("kind") or "").strip().lower()
            if kind:
                return f"episode_run:{kind}"
        return tool

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
