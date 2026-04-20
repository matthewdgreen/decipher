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
from analysis import dictionary, ic, ngram, pattern
from analysis import signals as sig
from analysis.segment import segment_text
from artifact.schema import RunArtifact, SolutionDeclaration
from models.cipher_text import CipherText
from services.claude_api import ClaudeAPI, ClaudeAPIError, estimate_cost
from workspace import Workspace


# ------------------------------------------------------------------
# Workspace panel: rendered into every user turn so the LLM always
# has the raw ciphertext + current decode in front of it.
# ------------------------------------------------------------------

PANEL_WORDS_START = 30
PANEL_WORDS_TAIL = 10
MAX_BRANCHES_IN_PANEL = 3

# Floor-trigger thresholds — when a branch crosses BOTH signals simultaneously,
# the panel renders an explicit "DECLARE NOW" block.
# Calibration notes:
#   - No-boundary dict_rate 0.85: the Viterbi segmenter inflates dict_rate by
#     finding short spurious dict words (DE, PA, ED…); 0.70 fired on 69.5%-
#     accurate text. 0.85 requires a genuinely solved decode.
#   - quad ≥ -2.2: English-like quadgram log-likelihood per gram. A 69%-accurate
#     text scores ~-5.6, well below the trigger.  Requiring BOTH signals prevents
#     false positives from either alone.
DECLARE_FLOOR_DICT_RATE_NO_BOUNDARY = 0.85
DECLARE_FLOOR_DICT_RATE_WITH_BOUNDARY = 0.50
DECLARE_FLOOR_QUAD = -2.2


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

FINAL_ITERATION_PREFLIGHT = (
    "🚨 **THIS IS YOUR FINAL ACTION TURN.** "
    "You are on the last iteration. You MUST call "
    "`meta_declare_solution(branch='...', rationale='...', self_confidence=...)` "
    "now. Do not call diagnostic or repair tools on this turn. An imperfect "
    "declared branch scores; no declaration is treated as a failed run."
)


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


def _score_branch_for_panel(
    workspace: Workspace,
    branch_name: str,
    language: str,
    word_set: set[str],
    freq_rank: dict[str, int] | None = None,
) -> tuple[float | None, float | None]:
    """Return (dict_rate, quad_loglik_per_gram) for panel display.

    dict_rate uses the DP segmenter for no-boundary text, otherwise whitespace
    split. quad_loglik_per_gram is the normalised quadgram log-likelihood.
    Returns (None, None) if the branch has no mappings (nothing to score).
    """
    branch = workspace.get_branch(branch_name)
    if not branch.key:
        return None, None
    decrypted = workspace.apply_key(branch_name)
    normalized = sig.normalize_for_scoring(decrypted)
    if not normalized.strip():
        return None, None

    # dict_rate
    if not any(c.isspace() for c in normalized.strip()):
        seg = segment_text(normalized, word_set, freq_rank)
        dict_rate = seg.dict_rate
    else:
        words = [w for w in normalized.split() if any(c.isalpha() for c in w)]
        if words:
            hits = sum(1 for w in words if w in word_set)
            dict_rate = hits / len(words)
        else:
            dict_rate = 0.0

    # quadgram
    quad_lp = ngram.NGRAM_CACHE.get(language, 4)
    quad = ngram.normalized_ngram_score(normalized, quad_lp, n=4)
    if quad == float("-inf"):
        quad = None

    return dict_rate, quad


def build_workspace_panel(
    workspace: Workspace,
    iteration: int,
    language: str = "en",
    word_set: set[str] | None = None,
    freq_rank: dict[str, int] | None = None,
    max_iterations: int | None = None,
    tokens_used: int = 0,
    estimated_cost_usd: float = 0.0,
    run_python_calls: int = 0,
) -> str:
    """Render ciphertext + current partial decode(s) for this turn.

    The LLM sees this on every iteration so its judgement always centers on
    reading the text, not on scores. Per-branch scores and a floor-trigger
    block give it a mechanical signal of when to declare.
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
    if word_set is None:
        word_set = set()

    iters_left = (max_iterations - iteration) if max_iterations else None
    iter_str = (
        f"iteration {iteration}/{max_iterations}"
        if max_iterations else f"iteration {iteration}"
    )
    cost_str = ""
    if tokens_used:
        tok_k = tokens_used / 1000
        if estimated_cost_usd:
            cost_str = f" — {tok_k:,.0f}K tokens (≈${estimated_cost_usd:.2f})"
        else:
            cost_str = f" — {tok_k:,.0f}K tokens"

    lines: list[str] = []
    lines.append(f"## Workspace panel — {iter_str}{cost_str}")
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

    # Precompute scores once; reuse for per-branch header + floor trigger.
    branch_scores: dict[str, tuple[float | None, float | None]] = {
        name: _score_branch_for_panel(
            workspace, name, language, word_set, freq_rank
        )
        for name in branch_names
    }

    for name in branch_names:
        branch = workspace.get_branch(name)
        decoded = [
            _render_decoded_word(ct.words[i], branch.key, pt_alpha) for i in indices
        ]
        dict_rate, quad = branch_scores[name]
        score_bits = []
        if dict_rate is not None:
            score_bits.append(f"dict={dict_rate:.2f}")
        if quad is not None:
            score_bits.append(f"quad={quad:.2f}")
        score_str = f"  {'  '.join(score_bits)}" if score_bits else ""
        lines.append("")
        lines.append(
            f"### Branch `{name}` — {len(branch.key)}/{alpha.size} symbols mapped"
            f"{score_str}"
        )
        lines.append("```")
        lines.append(" | ".join(decoded))
        lines.append("```")

    lines.append("")

    no_boundaries = (n_words <= 1)

    # Floor trigger: branch must cross BOTH dict AND quad thresholds to fire.
    # Requiring both signals prevents false positives from the Viterbi dict_rate
    # being inflated by short spurious dict words on a poorly-decoded text.
    floor_dict = (
        DECLARE_FLOOR_DICT_RATE_NO_BOUNDARY if no_boundaries
        else DECLARE_FLOOR_DICT_RATE_WITH_BOUNDARY
    )
    triggered: list[tuple[str, float | None, float | None]] = []
    for name, (dr, qd) in branch_scores.items():
        if (
            dr is not None and dr >= floor_dict
            and qd is not None and qd >= DECLARE_FLOOR_QUAD
        ):
            triggered.append((name, dr, qd))
    if triggered:
        best_name, best_dr, best_qd = triggered[0]
        dr_part = f"dict={best_dr:.2f}" if best_dr is not None else "dict=?"
        qd_part = f" quad={best_qd:.2f}" if best_qd is not None else ""
        lines.append(
            f"**⚑ DECLARE NOW: branch `{best_name}` has {dr_part}{qd_part} — "
            f"this is a solved cipher.** Call "
            f"`meta_declare_solution(branch='{best_name}', rationale='...', "
            f"self_confidence=0.9)`. Further manual swaps from this point "
            f"risk *regression*. Declaration records the CURRENT accuracy; "
            f"chasing perfection and running out of iterations scores ZERO."
        )
        lines.append("")

    if run_python_calls >= 3:
        lines.append(
            f"⚠ **Token efficiency**: you have called `run_python` "
            f"{run_python_calls} time(s) this run. Many of those computations "
            f"are already built into the dedicated tools — which are faster, "
            f"cheaper, and don't consume an iteration for boilerplate code:\n"
            f"- Letter frequencies / absent letters → `decode_letter_stats`\n"
            f"- Segmentation + pseudo-word diagnosis → `decode_diagnose`\n"
            f"- Dictionary rate (works for no-boundary text) → `score_panel` "
            f"or `score_dictionary`\n"
            f"- Quadgram score → `score_quadgram`\n"
            f"Reserve `run_python` for genuinely novel computations."
        )
        lines.append("")

    if no_boundaries:
        lines.append(
            "**Read the decoded stream above.** This cipher has no word "
            "boundaries — the plaintext is one continuous run of letters. "
            "Call `decode_diagnose(branch)` to see which segments are likely "
            "wrong and which single swap fixes the most errors. "
            "If one decoded letter seems to stand for several true letters, "
            "call `decode_ambiguous_letter(branch, decoded_letter)` to separate "
            "the cipher symbols before changing anything. "
            "Use `act_swap_decoded(branch, 'X', 'Y')` to fix a decoded letter X "
            "that should be Y — the result includes `score_delta` so you can "
            "immediately confirm the swap helped. "
            "Use `decode_diagnose_and_fix(branch)` instead of applying many "
            "one-at-a-time repairs. "
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

    # Last-iteration hard warning — must appear AFTER the main text block.
    if max_iterations is not None and iteration >= max_iterations:
        lines.append("")
        lines.append(
            "🚨 **THIS IS THE FINAL ITERATION.** "
            "You have NO more turns after this. "
            "You MUST call `meta_declare_solution` RIGHT NOW on your best branch, "
            "or the run ends with score=0.00. "
            "An imperfect declaration always beats no declaration."
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
    freq_rank = {w.upper(): i + 1 for i, w in enumerate(word_list)}
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

        if iteration == max_iterations:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": FINAL_ITERATION_PREFLIGHT}],
            })

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

        # Accumulate token usage
        usage = response.usage
        artifact.total_input_tokens += usage.input_tokens
        artifact.total_output_tokens += usage.output_tokens
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        artifact.total_cache_read_tokens += cache_read
        artifact.estimated_cost_usd = estimate_cost(
            claude_api.model,
            artifact.total_input_tokens,
            artifact.total_output_tokens,
            artifact.total_cache_read_tokens,
        )

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
            run_python_calls = sum(
                1 for tc in executor.call_log if tc.tool_name == "run_python"
            )
            panel_text = build_workspace_panel(
                workspace, iteration,
                language=language, word_set=word_set,
                freq_rank=freq_rank,
                max_iterations=max_iterations,
                tokens_used=artifact.total_input_tokens + artifact.total_output_tokens,
                estimated_cost_usd=artifact.estimated_cost_usd,
                run_python_calls=run_python_calls,
            )
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

    if artifact.status in {"exhausted", "error"} and executor.solution is None:
        best_branch, best_scores = _best_branch_for_auto_declare(
            workspace, language, word_set, freq_rank
        )
        if artifact.status == "error":
            reason = (
                "Automatic fallback declaration after agent/API error; a useful "
                "partial branch still scores better than no declaration. "
                f"Original error: {artifact.error_message}. "
            )
        else:
            reason = (
                "Automatic fallback declaration at iteration limit; the agent "
                "did not call meta_declare_solution in time. "
            )
        executor.solution = SolutionDeclaration(
            branch=best_branch,
            rationale=(
                f"{reason}Selected the "
                "highest-scoring available branch by internal dictionary and "
                f"quadgram signals: {best_scores}."
            ),
            self_confidence=0.0,
            declared_at_iteration=max_iterations,
        )
        artifact.status = "solved"
        emit("auto_declared_solution", {
            "branch": best_branch,
            "scores": best_scores,
        })

    # --- finalize ---
    artifact.finished_at = time.time()
    artifact.tool_calls = list(executor.call_log)
    artifact.tool_requests = list(executor.tool_requests)
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


def _best_branch_for_auto_declare(
    workspace: Workspace,
    language: str,
    word_set: set[str],
    freq_rank: dict[str, int] | None,
) -> tuple[str, dict[str, float | None]]:
    """Pick the best available branch without ground truth."""
    best_name = workspace.branch_names()[0]
    best_scores: dict[str, float | None] = {"dict_rate": None, "quad": None}
    best_key: tuple[float, float, int] = (float("-inf"), float("-inf"), -1)

    for name in workspace.branch_names():
        dr, quad = _score_branch_for_panel(
            workspace, name, language, word_set, freq_rank
        )
        branch = workspace.get_branch(name)
        rank_key = (
            dr if dr is not None else float("-inf"),
            quad if quad is not None else float("-inf"),
            len(branch.key),
        )
        if rank_key > best_key:
            best_name = name
            best_key = rank_key
            best_scores = {
                "dict_rate": round(dr, 4) if dr is not None else None,
                "quad": round(quad, 4) if quad is not None else None,
            }
    return best_name, best_scores
