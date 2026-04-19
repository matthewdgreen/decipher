"""v2 agent loop: workspace-based, agent-driven termination.

Differences from v1:
- No phased prompt; brief-style system prompt.
- Workspace replaces Session for the agent's purposes.
- Multi-signal scoring is a tool the agent calls, not a pushed signal.
- No auto-warnings, no auto-rollback. Agent drives.
- Termination via meta_declare_solution or iteration cap.
- Produces a RunArtifact capturing the full trajectory.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from agent.prompts_v2 import get_system_prompt, initial_context
from agent.tools_v2 import TOOL_DEFINITIONS, WorkspaceToolExecutor
from analysis import dictionary, ic, pattern
from artifact.schema import RunArtifact
from models.cipher_text import CipherText
from services.claude_api import ClaudeAPI, ClaudeAPIError
from workspace import Workspace


# ------------------------------------------------------------------
# Workspace panel: rendered into every user turn so the LLM always
# has the raw ciphertext + current decode in front of it.
# ------------------------------------------------------------------

PANEL_WORDS_START = 30
PANEL_WORDS_TAIL = 10
MAX_BRANCHES_IN_PANEL = 3


def _render_word_tokens(tokens, alphabet) -> str:
    sep = " " if alphabet._multisym else ""
    return sep.join(alphabet.symbol_for(t) for t in tokens)


def _render_decoded_word(tokens, key, pt_alphabet) -> str:
    sep = " " if pt_alphabet._multisym else ""
    parts = []
    for t in tokens:
        if t in key:
            parts.append(pt_alphabet.symbol_for(key[t]))
        else:
            parts.append("?")
    return sep.join(parts)


def _select_word_indices(n_words: int) -> list[int]:
    if n_words <= PANEL_WORDS_START + PANEL_WORDS_TAIL:
        return list(range(n_words))
    head = list(range(PANEL_WORDS_START))
    tail = list(range(n_words - PANEL_WORDS_TAIL, n_words))
    return head + tail


def _pick_panel_branches(workspace: Workspace) -> list[str]:
    """Pick up to MAX_BRANCHES_IN_PANEL branches to show. Prefer most-mapped;
    always include main if it has mappings or is the only branch."""
    entries = []
    for name in workspace.branch_names():
        b = workspace.get_branch(name)
        entries.append((name, len(b.key)))
    # Sort by mapped count desc, with main as a tiebreaker priority
    entries.sort(key=lambda e: (-e[1], 0 if e[0] == "main" else 1, e[0]))
    selected = [name for name, _ in entries[:MAX_BRANCHES_IN_PANEL]]
    # Always include main if not selected and has mappings
    if "main" not in selected:
        main_mapped = len(workspace.get_branch("main").key)
        if main_mapped > 0 and len(selected) >= MAX_BRANCHES_IN_PANEL:
            selected[-1] = "main"
        elif main_mapped > 0:
            selected.append("main")
    return selected


PANEL_HEADER_MARKER = "## Workspace panel — iteration "
PANEL_STUB_TEXT = "[earlier workspace panel omitted — see current panel below]"
TOOL_RESULT_STUB = "[result omitted from history — see current workspace panel]"

# Keep the last N iterations of full tool results; older ones are stubbed out.
TOOL_RESULT_HISTORY_DEPTH = 4


def _compress_history(messages: list[dict]) -> list[dict]:
    """Trim the message history to stay within token budgets.

    Two passes:
    1. Replace all workspace panel text blocks except the most recent with a stub.
    2. Replace tool_result content blocks from iterations more than
       TOOL_RESULT_HISTORY_DEPTH turns ago with a short stub, so long tool
       outputs (e.g. corpus_word_candidates with 50 candidates) don't
       accumulate into a 30k+ token context.
    """
    # --- pass 1: stale panels ---
    last_panel_msg_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(c, dict)
            and c.get("type") == "text"
            and PANEL_HEADER_MARKER in c.get("text", "")
            for c in content
        ):
            last_panel_msg_idx = i
            break

    # --- pass 2: identify user messages carrying tool results, oldest-first ---
    # Collect indices of user messages that contain tool_result blocks
    tool_result_msg_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(
            isinstance(c, dict) and c.get("type") == "tool_result"
            for c in m["content"]
        )
    ]
    # Stub everything except the most recent TOOL_RESULT_HISTORY_DEPTH turns
    stale_tool_indices = set(tool_result_msg_indices[:-TOOL_RESULT_HISTORY_DEPTH])

    out: list[dict] = []
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            out.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue

        new_content = []
        for c in content:
            if not isinstance(c, dict):
                new_content.append(c)
                continue
            # Stub stale workspace panels
            if (
                c.get("type") == "text"
                and PANEL_HEADER_MARKER in c.get("text", "")
                and i != last_panel_msg_idx
            ):
                new_content.append({"type": "text", "text": PANEL_STUB_TEXT})
            # Stub stale tool results
            elif c.get("type") == "tool_result" and i in stale_tool_indices:
                new_content.append({
                    "type": "tool_result",
                    "tool_use_id": c["tool_use_id"],
                    "content": TOOL_RESULT_STUB,
                })
            else:
                new_content.append(c)
        out.append({**m, "content": new_content})
    return out


def build_workspace_panel(workspace: Workspace, iteration: int) -> str:
    """Render ciphertext + current partial decode(s) for this turn.

    The LLM sees this on every iteration so its judgement always centers on
    reading the text, not on scores.
    """
    ct = workspace.cipher_text
    alpha = ct.alphabet
    pt_alpha = workspace.plaintext_alphabet
    n_words = len(ct.words)
    indices = _select_word_indices(n_words)

    # Render ciphertext tokens in those indices
    cipher_tokens = [_render_word_tokens(ct.words[i], alpha) for i in indices]
    ct_line = " | ".join(cipher_tokens)

    branch_names = _pick_panel_branches(workspace)

    lines: list[str] = []
    lines.append(f"## Workspace panel — iteration {iteration}")
    lines.append("")
    span_note = (
        f"words 0..{n_words - 1} (all)"
        if n_words <= PANEL_WORDS_START + PANEL_WORDS_TAIL
        else f"first {PANEL_WORDS_START} + last {PANEL_WORDS_TAIL} of {n_words} words"
    )
    lines.append(f"### Ciphertext ({span_note})")
    lines.append("```")
    lines.append(ct_line)
    lines.append("```")

    for name in branch_names:
        branch = workspace.get_branch(name)
        decoded = [
            _render_decoded_word(ct.words[i], branch.key, pt_alpha) for i in indices
        ]
        lines.append("")
        lines.append(
            f"### Branch `{name}` — {len(branch.key)}/{alpha.size} symbols mapped"
        )
        lines.append("```")
        lines.append(" | ".join(decoded))
        lines.append("```")

    lines.append("")

    no_boundaries = (n_words <= 1)
    if no_boundaries:
        lines.append(
            "**Read the decoded stream above.** This cipher has no word "
            "boundaries — the plaintext is one continuous run of letters. "
            "**Before declaring, segment the stream into words** and look "
            "for any obviously wrong segments (non-words). A single wrong "
            "cipher-symbol mapping corrupts every occurrence of that symbol "
            "throughout the text, so one fix can raise accuracy significantly. "
            "Use `act_swap_decoded(branch, 'X', 'Y')` to fix a decoded letter X "
            "that should be Y (this is safer than `act_set_mapping` for decoded-text "
            "corrections). "
            "After one or two fix attempts, **declare your best branch** — "
            "a 95% solution is excellent; don't exhaust all iterations "
            "chasing the last few errors."
        )
    else:
        lines.append(
            "**Read the decoded text above.** Your primary judgement instrument "
            "is semantic reading, not scores. "
            "If a branch's decode looks like coherent text in the target language, "
            "call `meta_declare_solution` on that branch now — don't keep "
            "optimizing a solved cipher. "
            "**IMPORTANT:** Declaring a partial solution is always better than "
            "not declaring. If you can read even a few correct words "
            "(ET, PER, IN, EST…) call `meta_declare_solution` with your best "
            "branch immediately — the benchmark records whatever accuracy your "
            "branch achieves at the moment you declare, so 5% beats 0% every "
            "time. Only exhaust all iterations if you truly cannot read a single "
            "recognisable word in the target language."
        )
    return "\n".join(lines)


def run_v2(
    cipher_text: CipherText,
    claude_api: ClaudeAPI,
    language: str = "en",
    max_iterations: int = 50,
    cipher_id: str = "unknown",
    prior_context: str | None = None,
    verbose: bool = False,
    on_event: Any = None,  # optional callback(event_type: str, payload: dict)
) -> RunArtifact:
    """Run one v2 agent session against a cipher. Returns a full RunArtifact."""

    run_id = uuid.uuid4().hex[:12]

    artifact = RunArtifact(
        run_id=run_id,
        cipher_id=cipher_id,
        model=claude_api.model,
        language=language,
        cipher_alphabet_size=cipher_text.alphabet.size,
        cipher_token_count=len(cipher_text.tokens),
        cipher_word_count=len(cipher_text.words),
        max_iterations=max_iterations,
    )

    def emit(event: str, payload: dict) -> None:
        if on_event is not None:
            try:
                on_event(event, payload)
            except Exception:  # noqa: BLE001
                pass
        if verbose:
            print(f"[{event}] {payload}")

    # --- set up resources ---
    dict_path = dictionary.get_dictionary_path(language)
    word_set = dictionary.load_word_set(dict_path) if dict_path else set()
    word_list = pattern.load_word_list(dict_path) if dict_path else []
    pattern_dict = pattern.build_pattern_dictionary(word_list)

    workspace = Workspace(cipher_text=cipher_text)
    executor = WorkspaceToolExecutor(
        workspace=workspace,
        language=language,
        word_set=word_set,
        word_list=word_list,
        pattern_dict=pattern_dict,
    )

    # --- initial context ---
    ic_value = ic.index_of_coincidence(cipher_text.tokens, cipher_text.alphabet.size)
    context_msg = initial_context(
        cipher_display=cipher_text.raw,
        alphabet_symbols=cipher_text.alphabet.symbols,
        total_tokens=len(cipher_text.tokens),
        total_words=len(cipher_text.words),
        ic_value=ic_value,
        language=language,
        prior_context=prior_context,
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": context_msg}]
    artifact.messages = messages  # reference, will grow

    system = get_system_prompt(language)

    start = time.time()

    for iteration in range(1, max_iterations + 1):
        workspace.set_iteration(iteration)
        executor.set_iteration(iteration)
        emit("iteration_start", {"iteration": iteration})

        # Compress history before sending: stub stale panels and old tool
        # results to keep total input tokens well under rate-limit ceilings.
        send_messages = _compress_history(messages)

        try:
            response = claude_api.send_message(
                messages=send_messages,
                tools=TOOL_DEFINITIONS,
                system=system,
                max_tokens=8192,
            )
        except ClaudeAPIError as e:
            artifact.status = "error"
            artifact.error_message = f"API error on iteration {iteration}: {e}"
            emit("error", {"message": artifact.error_message})
            break

        # Collect assistant blocks to add to the message history
        assistant_blocks: list[dict[str, Any]] = []
        tool_uses: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                assistant_blocks.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            elif block.type == "tool_use":
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_uses.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        messages.append({"role": "assistant", "content": assistant_blocks})

        # Capture plan from the first iteration's text
        if iteration == 1 and text_parts:
            artifact.plan = "\n\n".join(text_parts)

        if text_parts:
            emit("agent_text", {"iteration": iteration, "text": text_parts[0][:400]})

        # No tool uses → agent has no more to do
        if not tool_uses:
            artifact.status = "exhausted"
            emit("no_tool_calls", {"iteration": iteration})
            break

        # Execute tool calls
        tool_results_blocks: list[dict[str, Any]] = []
        for tu in tool_uses:
            result = executor.execute(tu["name"], tu["input"], tool_use_id=tu["id"])
            tool_results_blocks.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result,
            })
            emit("tool_call", {"tool": tu["name"], "result_preview": result[:160]})

        # Append the workspace panel as a trailing text block in the same
        # user turn, so the LLM always sees the raw ciphertext and current
        # partial decode before choosing its next action. Skip if the agent
        # just declared a solution — no further iterations will run.
        if not executor.terminated:
            panel_text = build_workspace_panel(workspace, iteration)
            tool_results_blocks.append({"type": "text", "text": panel_text})

        messages.append({"role": "user", "content": tool_results_blocks})

        if executor.terminated:
            artifact.status = "solved"
            emit("declared_solution", {
                "branch": executor.solution.branch if executor.solution else None,
                "confidence": executor.solution.self_confidence if executor.solution else None,
            })
            break

    else:
        # Loop completed without break — hit max_iterations
        artifact.status = "exhausted"
        emit("max_iterations_reached", {"iterations": max_iterations})

    # --- finalize ---
    artifact.finished_at = time.time()
    artifact.tool_calls = list(executor.call_log)
    artifact.solution = executor.solution
    artifact.branches = [
        _branch_snapshot_for(workspace, name) for name in workspace.branch_names()
    ]

    elapsed = artifact.finished_at - start
    emit("run_complete", {
        "status": artifact.status,
        "iterations": min(iteration, max_iterations),
        "elapsed_seconds": round(elapsed, 1),
    })

    return artifact


def _branch_snapshot_for(workspace: Workspace, name: str) -> Any:
    """Build a BranchSnapshot dataclass for a single branch."""
    from artifact.schema import BranchSnapshot
    branch = workspace.get_branch(name)
    return BranchSnapshot(
        name=name,
        parent=branch.parent,
        created_iteration=branch.created_iteration,
        key=dict(branch.key),
        mapped_count=len(branch.key),
        decryption=workspace.apply_key(name),
        signals={},  # panel not computed here; caller can add post-hoc
        tags=list(branch.tags),
    )
