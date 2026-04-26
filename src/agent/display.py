"""Terminal renderers for agentic CLI runs."""
from __future__ import annotations

import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Protocol


class AgentRunRenderer(Protocol):
    def start_test(self, test_id: str, description: str, *, model: str, max_iterations: int) -> None:
        ...

    def event(self, event: str, payload: dict[str, Any]) -> None:
        ...

    def finish(self, result: Any) -> None:
        ...


def make_agent_renderer(
    mode: str,
    *,
    stream: Any = None,
) -> AgentRunRenderer | None:
    stream = stream or sys.stdout
    if mode == "off":
        return None
    if mode == "jsonl":
        return JsonlAgentRenderer(stream)
    if mode == "pretty":
        return PrettyAgentRenderer(stream)
    return RawAgentRenderer(stream)


def summarize_tool_call(tool: str, result: dict[str, Any]) -> str:
    parts = [tool]
    if result.get("branch"):
        parts.append(f"[{result['branch']}]")
    if result.get("from") and result.get("to"):
        parts.append(f"{result['from']} -> {result['to']}")
    elif result.get("mapping"):
        parts.append(str(result["mapping"]))
    elif result.get("mappings"):
        mappings = result["mappings"]
        if isinstance(mappings, dict):
            parts.append(", ".join(f"{k}->{v}" for k, v in list(mappings.items())[:3]))
    status = result.get("status")
    if status:
        parts.append(f"({status})")
    agenda_item = result.get("agenda_item")
    if isinstance(agenda_item, dict):
        item_id = agenda_item.get("id")
        item_status = agenda_item.get("status")
        if item_id is not None:
            parts.append(f"agenda#{item_id}:{item_status}")
    if result.get("unresolved_count") is not None:
        parts.append(f"unresolved={result['unresolved_count']}")
    risks = result.get("orthography_risks")
    if isinstance(risks, list) and risks:
        first = risks[0]
        if isinstance(first, dict):
            parts.append(f"orthography-risk={first.get('from')}->{first.get('to')}")
        else:
            parts.append("orthography-risk")
    if result.get("error"):
        parts.append(f"ERROR: {result['error']}")
    return " ".join(parts)


class RawAgentRenderer:
    """Current compact event stream for scripts and debugging."""

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream or sys.stdout

    def start_test(self, test_id: str, description: str, *, model: str, max_iterations: int) -> None:
        print(f"[agentic] {test_id} — {description}", file=self.stream)

    def event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "preflight_start":
            print("  preflight(no-LLM)...", end="", flush=True, file=self.stream)
        elif event == "preflight_result":
            print(
                f" [{payload.get('status', 'unknown')}, "
                f"{payload.get('elapsed_seconds', 0):.0f}s, $0.00 no LLM]",
                flush=True,
                file=self.stream,
            )
        elif event == "iteration_start":
            print(f"  iter {payload['iteration']}...", end="", flush=True, file=self.stream)
        elif event == "tool_call":
            print(".", end="", flush=True, file=self.stream)
        elif event in {"declared_solution", "run_complete", "error", "max_iterations_reached"}:
            print(f" [{event}]", flush=True, file=self.stream)

    def finish(self, result: Any) -> None:
        final_summary = str(getattr(result, "final_summary", "") or "").strip()
        if final_summary:
            print(file=self.stream)
            print("Final summary:", file=self.stream)
            print(final_summary, file=self.stream)
        else:
            print(file=self.stream)


class JsonlAgentRenderer:
    """Machine-readable renderer for GUI wrappers and external tools."""

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream or sys.stdout

    def start_test(self, test_id: str, description: str, *, model: str, max_iterations: int) -> None:
        self._write({
            "event": "test_start",
            "test_id": test_id,
            "description": description,
            "model": model,
            "max_iterations": max_iterations,
        })

    def event(self, event: str, payload: dict[str, Any]) -> None:
        self._write({"event": event, "payload": payload})

    def finish(self, result: Any) -> None:
        self._write({
            "event": "test_finish",
            "test_id": result.test_id,
            "status": result.status,
            "char_accuracy": result.char_accuracy,
            "word_accuracy": result.word_accuracy,
            "artifact_path": result.artifact_path,
            "error_message": result.error_message,
            "final_summary": getattr(result, "final_summary", ""),
        })

    def _write(self, obj: dict[str, Any]) -> None:
        print(json.dumps(obj, ensure_ascii=False), file=self.stream, flush=True)


@dataclass
class _PrettyState:
    test_id: str = ""
    description: str = ""
    model: str = ""
    max_iterations: int = 0
    iteration: int = 0
    mode: str = ""
    branch: str = ""
    mapped_count: int = 0
    scores: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    decryption: str = ""
    previous_decryption: str = ""
    log: list[str] = field(default_factory=list)
    commentary: str = ""
    error: str = ""


class PrettyAgentRenderer:
    """Human-readable live renderer.

    Uses Rich when installed. Without Rich, it degrades to readable line output
    so the CLI still works in minimal environments.
    """

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream or sys.stdout
        self.state = _PrettyState()
        self._rich = self._load_rich()
        self._live = None

    def start_test(self, test_id: str, description: str, *, model: str, max_iterations: int) -> None:
        self.state = _PrettyState(
            test_id=test_id,
            description=description,
            model=model,
            max_iterations=max_iterations,
        )
        if self._rich:
            self._live = self._rich["Live"](
                self._render(),
                console=self._rich["Console"](file=self.stream),
                refresh_per_second=6,
                transient=False,
            )
            self._live.start()
        else:
            print(f"\n=== {test_id} ===", file=self.stream)
            print(description, file=self.stream)

    def event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "iteration_start":
            self.state.iteration = int(payload.get("iteration") or 0)
            self.state.mode = str(payload.get("mode") or "")
            self._add_log(f"iter {self.state.iteration}/{self.state.max_iterations}  mode={self.state.mode}")
        elif event == "preflight_start":
            self._add_log("preflight(no-LLM) started")
        elif event == "preflight_result":
            self._add_log(
                f"preflight {payload.get('status')} solver={payload.get('solver')} "
                f"{payload.get('elapsed_seconds', 0):.1f}s"
            )
        elif event == "agent_text":
            text = _clean_text(str(payload.get("text", "")))
            self.state.commentary = text
            self._add_log(f"agent: {text[:140]}")
        elif event == "tool_call":
            tool = str(payload.get("tool", "tool"))
            summary = summarize_tool_call(tool, payload.get("result_summary") or {})
            self._add_log(f"tool: {summary}")
            changed = (payload.get("result_summary") or {}).get("changed_words")
            if changed:
                pieces = [
                    f"{c.get('before')}->{c.get('after')}"
                    for c in changed[:3]
                    if isinstance(c, dict)
                ]
                self._add_log("  changed: " + ", ".join(pieces))
        elif event == "workspace_snapshot":
            self.state.previous_decryption = self.state.decryption
            self.state.decryption = str(payload.get("decryption") or "")
            self.state.branch = str(payload.get("branch") or "")
            self.state.mapped_count = int(payload.get("mapped_count") or 0)
            self.state.scores = payload.get("scores") or {}
            self.state.total_tokens = int(payload.get("total_tokens") or 0)
            self.state.estimated_cost_usd = float(payload.get("estimated_cost_usd") or 0.0)
        elif event == "boundary_projection_count_retry":
            self._add_log("warning: reading proposal length mismatch; retrying in-place")
        elif event == "gated_tool_retry":
            attempted = ", ".join(payload.get("attempted_tools") or [])
            self._add_log(f"warning: gated tool rejected ({attempted}); retrying in-place")
        elif event == "error":
            self.state.error = str(payload.get("message", "API/provider error"))
            self._add_log("ERROR: " + self.state.error)
        elif event == "auto_declared_solution":
            branch = payload.get("branch")
            self._add_log(f"fallback auto-declared {branch}")
        elif event == "declared_solution":
            self._add_log(f"declared {payload.get('branch')} conf={payload.get('confidence')}")
        elif event == "run_complete":
            self._add_log(f"complete status={payload.get('status')} time={payload.get('elapsed_seconds')}s")
        self._refresh()

    def finish(self, result: Any) -> None:
        if self._live:
            self._live.update(self._render_final(result))
            self._live.stop()
        else:
            print(
                f"Status: {result.status}  char={result.char_accuracy:.1%} "
                f"word={result.word_accuracy:.1%}  artifact={result.artifact_path}",
                file=self.stream,
            )
            if result.error_message:
                print("ERROR:", result.error_message, file=self.stream)
            final_summary = str(getattr(result, "final_summary", "") or "").strip()
            if final_summary:
                print("\nFinal summary:", file=self.stream)
                print(final_summary, file=self.stream)

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _add_log(self, line: str) -> None:
        self.state.log.append(line)
        self.state.log = self.state.log[-12:]

    def _render(self) -> Any:
        if not self._rich:
            return ""
        Layout = self._rich["Layout"]
        Panel = self._rich["Panel"]
        Text = self._rich["Text"]
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="decrypt", ratio=2),
            Layout(name="agent", ratio=3),
        )
        header = (
            f"{self.state.test_id}  iter {self.state.iteration}/{self.state.max_iterations}  "
            f"mode={self.state.mode or '-'}  branch={self.state.branch or '-'}  "
            f"dict={self.state.scores.get('dict_rate')} quad={self.state.scores.get('quad')}  "
            f"{_format_live_usage(self.state.total_tokens, self.state.estimated_cost_usd)}"
        )
        layout["header"].update(Panel(header, title="Decipher agentic run"))
        layout["decrypt"].update(Panel(self._decrypt_text(), title="Current decrypt"))
        agent_lines = []
        if self.state.commentary:
            agent_lines.append(self.state.commentary)
            agent_lines.append("")
        agent_lines.extend(self.state.log)
        if self.state.error:
            agent_lines.append("")
            agent_lines.append("[API ERROR] " + self.state.error)
        layout["agent"].update(Panel("\n".join(agent_lines) or "Waiting...", title="Agent"))
        return layout

    def _render_final(self, result: Any) -> Any:
        if not self._rich:
            return ""
        Layout = self._rich["Layout"]
        Panel = self._rich["Panel"]
        layout = Layout()
        layout.split_column(
            Layout(name="summary", size=8),
            Layout(name="body", ratio=1),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="final_summary", ratio=2),
        )
        layout["left"].split_column(
            Layout(name="decrypt", ratio=2),
            Layout(name="alignment", ratio=5),
        )
        body = [
            f"Status: {result.status}",
            f"Branch: {getattr(result, 'final_branch', '') or self.state.branch or '-'}",
            f"Char: {result.char_accuracy:.1%}   Word: {result.word_accuracy:.1%}",
            f"Iterations: {result.iterations_used}   Time: {result.elapsed_seconds:.1f}s",
            f"Tokens: {result.total_tokens}   Cost: ${result.estimated_cost_usd:.2f}",
            f"Artifact: {result.artifact_path}",
        ]
        branch_scores = getattr(result, "branch_scores", None) or []
        if branch_scores:
            score_bits = []
            for row in branch_scores[:5]:
                word = row.get("word_accuracy")
                word_part = "N/A" if word is None else f"{word:.1%}"
                score_bits.append(
                    f"{row.get('branch')}: char={row.get('char_accuracy', 0.0):.1%} "
                    f"word={word_part}"
                )
            body.append("Branches: " + "; ".join(score_bits))
        if result.error_message:
            body.append("")
            body.append("API/provider error:")
            body.append(result.error_message)
            body.append("")
            body.append("This run may have fallback auto-declared; do not treat it as a capability result.")
        layout["summary"].update(Panel("\n".join(body), title=f"{result.test_id} complete"))

        final_decryption = str(getattr(result, "final_decryption", "") or self.state.decryption)
        layout["decrypt"].update(
            Panel(
                _compact_preview(final_decryption, max_chars=1800),
                title="Final decrypt",
            )
        )
        final_summary = str(getattr(result, "final_summary", "") or "").strip()
        if not final_summary:
            final_summary = "No final reading summary was provided for this run."
        layout["final_summary"].update(
            Panel(
                _compact_final_summary(final_summary, max_chars=2400),
                title="Reading / Process Summary",
            )
        )
        alignment = str(getattr(result, "alignment_report", "") or "")
        if not alignment:
            alignment = "No ground-truth alignment available for this run."
        layout["alignment"].update(
            Panel(
                alignment,
                title="Matched / Unmatched Words",
            )
        )
        return layout

    def _decrypt_text(self) -> Any:
        if not self._rich:
            return ""
        Text = self._rich["Text"]
        text = Text()
        current = self.state.decryption[:900] or "(no decrypt yet)"
        previous = self.state.previous_decryption
        for i, ch in enumerate(current):
            if ch == "?":
                style = "yellow dim"
            elif i < len(previous) and previous[i] != ch:
                style = "bold bright_white"
            elif ch == "|":
                style = "dim"
            else:
                style = "white"
            text.append(ch, style=style)
        return text

    @staticmethod
    def _load_rich() -> dict[str, Any] | None:
        try:
            from rich.console import Console
            from rich.layout import Layout
            from rich.live import Live
            from rich.panel import Panel
            from rich.text import Text
        except Exception:
            return None
        return {
            "Console": Console,
            "Layout": Layout,
            "Live": Live,
            "Panel": Panel,
            "Text": Text,
        }


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return textwrap.shorten(text, width=900, placeholder="...")


def _compact_preview(text: str, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    return text or "(no decrypt available)"


def _format_live_usage(total_tokens: int, estimated_cost_usd: float) -> str:
    tokens = max(0, int(total_tokens or 0))
    cost = max(0.0, float(estimated_cost_usd or 0.0))
    if tokens >= 1_000_000:
        token_text = f"{tokens / 1_000_000:.2f}M"
    elif tokens >= 10_000:
        token_text = f"{tokens / 1000:.0f}K"
    elif tokens >= 1000:
        token_text = f"{tokens / 1000:.1f}K"
    else:
        token_text = str(tokens)
    return f"tokens={token_text} cost=${cost:.2f}"


def _compact_final_summary(text: str, *, max_chars: int) -> str:
    """Make the final summary fit a terminal panel without losing sections."""
    lines = [line.strip() for line in text.splitlines()]
    sections: list[str] = []
    current_heading = ""
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_parts
        if current_heading or current_parts:
            body = " ".join(current_parts).strip()
            if current_heading and body:
                sections.append(f"{current_heading}: {body}")
            elif current_heading:
                sections.append(current_heading)
            elif body:
                sections.append(body)
        current_heading = ""
        current_parts = []

    for line in lines:
        if not line:
            continue
        if line.endswith(":") and len(line) <= 40:
            flush()
            current_heading = line[:-1]
            continue
        if line.startswith("- "):
            current_parts.append(line[2:])
        else:
            current_parts.append(line)
    flush()

    compact = "\n".join(sections)
    compact = re.sub(r"[ \t]+", " ", compact).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + " ..."
    return compact or "No final reading summary was provided for this run."
