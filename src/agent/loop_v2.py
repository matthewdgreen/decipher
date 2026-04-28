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
import json
from typing import Any

from agent.model_provider import (
    ModelProviderError,
    ModelResponse,
    ensure_model_provider,
    estimate_provider_cost,
)
from agent.orchestration import AgentMode
from agent.prompts_v2 import get_system_prompt, initial_context
from agent.resume import inherited_repair_agenda, install_resume_branches
from agent.tools_v2 import TOOL_DEFINITIONS, WorkspaceToolExecutor
from analysis import dictionary, ic, ngram, pattern
from analysis import signals as sig
from analysis.segment import segment_text
from artifact.schema import LoopEvent, RunArtifact, SolutionDeclaration
from models.cipher_text import CipherText
from workspace import Workspace


# ------------------------------------------------------------------
# Workspace panel: rendered into every user turn so the LLM always
# has the raw ciphertext + current decode in front of it.
# ------------------------------------------------------------------

PANEL_WORDS_START = 30
PANEL_WORDS_TAIL = 10
PANEL_WORDS_FULL_LIMIT = 90
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
    if n_words <= PANEL_WORDS_FULL_LIMIT:
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
READING_WORKFLOW_GATE_TURNS = 2
BOUNDARY_PROJECTION_MAX_INNER_RETRIES = 3
REPAIR_SANDBOX_MAX_INNER_RETRIES = 4
INSPECTION_SANDBOX_MAX_INNER_RETRIES = 3
FINAL_DECLARATION_MAX_INNER_RETRIES = 3

FULL_READING_WORKFLOW_TOOL_NAMES = {
    "decode_validate_reading_repair",
    "act_resegment_by_reading",
    "act_resegment_from_reading_repair",
    "act_resegment_window_by_reading",
}

FULL_READING_ACTUATOR_TOOL_NAMES = {
    "act_resegment_by_reading",
    "act_resegment_from_reading_repair",
    "act_resegment_window_by_reading",
}

PENULTIMATE_ALLOWED_TOOL_NAMES = {
    "workspace_fork_best",
    "decode_show",
    "score_panel",
    "score_dictionary",
    "observe_transform_pipeline",
    "workspace_branch_cards",
    "decode_plan_word_repair",
    "decode_plan_word_repair_menu",
    "act_apply_word_repair",
    "repair_agenda_list",
    "repair_agenda_update",
    "decode_validate_reading_repair",
    "act_resegment_by_reading",
    "act_resegment_from_reading_repair",
    "act_resegment_window_by_reading",
    "search_transform_homophonic",
    "search_review_transform_finalists",
    "act_install_transform_finalists",
    "act_rate_transform_finalist",
    "act_apply_transform_pipeline",
    "meta_declare_solution",
}

FINAL_ALLOWED_TOOL_NAMES = {
    "workspace_branch_cards",
    "repair_agenda_list",
    "repair_agenda_update",
    "meta_declare_solution",
}

REPAIR_SANDBOX_TOOL_NAMES = {
    "decode_show",
    "score_panel",
    "score_dictionary",
    "workspace_branch_cards",
    "decode_plan_word_repair",
    "decode_plan_word_repair_menu",
    "act_apply_word_repair",
    "repair_agenda_list",
    "repair_agenda_update",
    "decode_validate_reading_repair",
    "act_resegment_by_reading",
    "act_resegment_from_reading_repair",
    "act_resegment_window_by_reading",
}

INSPECTION_SANDBOX_TOOL_NAMES = {
    "workspace_fork_best",
    "observe_frequency",
    "observe_patterns",
    "observe_ic",
    "observe_homophone_distribution",
    "observe_transform_pipeline",
    "decode_show",
    "decode_unmapped",
    "decode_heatmap",
    "decode_letter_stats",
    "decode_ambiguous_letter",
    "decode_absent_letter_candidates",
    "decode_diagnose",
    "score_panel",
    "score_quadgram",
    "score_dictionary",
    "corpus_lookup_word",
    "corpus_word_candidates",
    "workspace_list_branches",
    "workspace_branch_cards",
    "workspace_compare",
    "repair_agenda_list",
}

FINAL_ITERATION_PREFLIGHT = (
    "🚨 **THIS IS YOUR FINAL ACTION TURN.** "
    "You are on the last iteration. You MUST end by calling "
    "`meta_declare_solution(branch='...', rationale='...', self_confidence=..., "
    "reading_summary='...', further_iterations_helpful=..., "
    "further_iterations_note='...')` now. Only final bookkeeping tools are available. "
    "If there are multiple "
    "branches, call `workspace_branch_cards` first. If any repair agenda item "
    "is open but you have decided not to apply it, call `repair_agenda_update` "
    "with status `held` or `rejected`, then declare in the same response. An "
    "imperfect declared branch scores; no declaration is treated as a failed "
    "run. The reading_summary should explain what the text appears to be about "
    "in plain language, especially for non-English target languages, and the "
    "further_iterations fields should say whether more work would likely help."
)

PENULTIMATE_READING_WORKFLOW_PREFLIGHT = (
    "⚠ **ONE TURN LEFT AFTER THIS.** If any branch is readable but still has "
    "word-boundary, segmentation, or alignment issues, this is your last safe "
    "turn to run the reading repair workflow. Do NOT spend this turn on "
    "another local split/merge, bulk mapping, or free search. Tool access is "
    "restricted on this turn: some tools are no longer available, and you must "
    "only use the allowed tools shown to you. If you can name a same-length "
    "word repair such as TREUITER -> BREUITER, call "
    "`decode_plan_word_repair`; if several readings are plausible or the word "
    "contains repeated letters, call `decode_plan_word_repair_menu` before "
    "`act_apply_word_repair`. If the main problem is boundaries/alignment, "
    "write your best target-language reading and prefer the actuator now. For "
    "a local boundary repair, use "
    "`act_resegment_window_by_reading`; for a whole-stream repair, "
    "call `act_resegment_by_reading` if it is character-preserving, or "
    "`act_resegment_from_reading_repair` if it changes letters but has the same "
    "character count. Use `decode_validate_reading_repair` first only if you "
    "truly cannot tell which actuator applies. Then use the final turn to "
    "declare the best branch."
)

READING_WORKFLOW_GATE_PREFLIGHT = (
    "⚠ **FULL-READING WORKFLOW WINDOW.** The run is near its final turns. "
    "If a branch is readable but has boundary/alignment drift, local edits and "
    "free search are now gated off. Some tools are no longer allowed; only use "
    "the tools available on this turn. If you have specific same-length word "
    "repairs, use `decode_plan_word_repair_menu` when several readings are "
    "plausible, or `decode_plan_word_repair` / `act_apply_word_repair` when "
    "one reading is clearly safe, so the repair agenda records them. For local "
    "boundary drift, use "
    "`act_resegment_window_by_reading` on just the affected word window. For "
    "global boundary drift, draft the best complete target-language reading "
    "and use `act_resegment_by_reading` or "
    "`act_resegment_from_reading_repair`; use `decode_validate_reading_repair` "
    "only if you need to choose between those two."
)

BOUNDARY_PROJECTION_RETRY_PREFLIGHT = (
    "The previous tool call was rejected by the boundary-projection gate. "
    "This does not consume another outer iteration, but you must recover now. "
    "Some tools are no longer allowed in this window. Use only the allowed "
    "tools shown to you. If you voiced a same-length word repair, use "
    "`decode_plan_word_repair_menu` or `decode_plan_word_repair` before "
    "`act_apply_word_repair`. Otherwise prefer a "
    "local boundary action with `act_resegment_window_by_reading`, or a "
    "full-reading action: `act_resegment_by_reading`, "
    "`act_resegment_from_reading_repair`, or "
    "`decode_validate_reading_repair` if you need to choose between them."
)

BOUNDARY_PROJECTION_COUNT_RETRY_PREFLIGHT = (
    "The full-reading proposal could not be applied because its normalized "
    "character count did not match the current branch. This does not consume "
    "another outer iteration. Revise the proposal now inside this same "
    "boundary-projection workflow. You must write a complete reading, not a "
    "summary or excerpt. Its normalized letters must have exactly the "
    "`current_char_count` reported by the tool result. If you intend only word "
    "boundary changes, preserve the decoded character stream exactly and call "
    "`act_resegment_by_reading`. If you intend spelling/key repairs too, keep "
    "the same character count and call `act_resegment_from_reading_repair`. "
    "For small split/merge fixes, avoid rewriting the whole stream: call "
    "`act_resegment_window_by_reading` with the affected word window. If the "
    "failed tool result includes `nearby_compatible_windows`, use one of those "
    "exact suggested retry calls before drafting a new proposal from scratch."
)

REPAIR_SANDBOX_CONTINUE_PREFLIGHT = (
    "You are still inside the low-cost repair sandbox for this same outer "
    "iteration. The previous tool results did not consume another benchmark "
    "iteration. Use this chance for one more small experiment: inspect the "
    "result, try a nearby suggested resegmentation window, compare a word "
    "repair menu, apply a clearly safe word repair, or declare if the branch "
    "is good enough. Avoid free search and broad mapping changes here; keep "
    "this to local reading, word-repair, and boundary-repair tools."
)

INSPECTION_SANDBOX_CONTINUE_PREFLIGHT = (
    "You are still inside the same outer iteration. The previous tool call(s) "
    "were read-only inspection, so they did not consume another benchmark "
    "iteration. Use this chance to act on what you just learned: apply a "
    "targeted repair, try a boundary tool, start a search/polish step, or "
    "declare. Call more read-only inspection only if it is genuinely needed "
    "to choose the next action."
)

FINAL_DECLARATION_RETRY_PREFLIGHT = (
    "You used final-turn bookkeeping tools but have not declared yet. This "
    "does not consume another outer iteration. You must now call "
    "`meta_declare_solution` on the best branch. If a declaration was blocked, "
    "use only the suggested bookkeeping tool needed to clear that blocker, "
    "then call `meta_declare_solution` in this same response."
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


def _filter_tool_definitions(allowed_names: set[str]) -> list[dict[str, Any]]:
    return [tool for tool in TOOL_DEFINITIONS if tool["name"] in allowed_names]


def _has_used_full_reading_workflow(executor: WorkspaceToolExecutor) -> bool:
    return any(
        call.tool_name in FULL_READING_ACTUATOR_TOOL_NAMES
        for call in executor.call_log
    )


def _is_reading_workflow_gate_turn(
    executor: WorkspaceToolExecutor,
    iteration: int,
    max_iterations: int,
) -> bool:
    return (
        max_iterations > READING_WORKFLOW_GATE_TURNS
        and max_iterations - READING_WORKFLOW_GATE_TURNS <= iteration < max_iterations
        and not _has_used_full_reading_workflow(executor)
    )


def _tool_definitions_for_turn(
    executor: WorkspaceToolExecutor,
    iteration: int,
    max_iterations: int,
) -> list[dict[str, Any]]:
    if iteration == max_iterations:
        return _filter_tool_definitions(FINAL_ALLOWED_TOOL_NAMES)
    if _is_reading_workflow_gate_turn(executor, iteration, max_iterations):
        return _filter_tool_definitions(PENULTIMATE_ALLOWED_TOOL_NAMES)
    return TOOL_DEFINITIONS


def _mode_for_turn(
    executor: WorkspaceToolExecutor,
    iteration: int,
    max_iterations: int,
) -> AgentMode:
    if iteration == max_iterations:
        return AgentMode.DECLARE
    if _is_reading_workflow_gate_turn(executor, iteration, max_iterations):
        return AgentMode.BOUNDARY_PROJECTION
    return AgentMode.EXPLORE


def _collect_assistant_blocks(
    response: ModelResponse,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
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
    return assistant_blocks, tool_uses, text_parts


def _parse_json_result(result: str) -> Any:
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return None


def _is_tool_gated_result(result: str) -> bool:
    parsed = _parse_json_result(result)
    if not isinstance(parsed, dict):
        return False
    return parsed.get("reason") == "tool_gated"


def _is_boundary_projection_count_failure(tool_name: str, result: str) -> bool:
    if tool_name not in FULL_READING_WORKFLOW_TOOL_NAMES:
        return False
    parsed = _parse_json_result(result)
    if not isinstance(parsed, dict):
        return False
    if parsed.get("same_character_count") is False:
        return True
    if not parsed.get("error"):
        return False
    projection = parsed.get("boundary_projection")
    return (
        isinstance(projection, dict)
        and projection.get("applicable") is False
        and "character counts differ" in str(projection.get("reason", ""))
    )


def _tool_result_summary(result: str) -> dict[str, Any]:
    parsed = _parse_json_result(result)
    if not isinstance(parsed, dict):
        return {}
    keys = (
        "status",
        "error",
        "branch",
        "mapping",
        "from",
        "to",
        "mappings",
        "mappings_set",
        "score_delta",
        "old_word_count",
        "new_word_count",
        "old_window_word_count",
        "new_window_word_count",
        "old_total_word_count",
        "new_total_word_count",
        "current_char_count",
        "proposed_char_count",
        "same_character_count",
        "unresolved_count",
    )
    summary = {key: parsed[key] for key in keys if key in parsed}
    changed = parsed.get("changed_words")
    if isinstance(changed, list):
        summary["changed_words"] = changed[:4]
    risks = parsed.get("orthography_risks")
    if isinstance(risks, list) and risks:
        summary["orthography_risks"] = risks[:2]
    agenda_item = parsed.get("agenda_item")
    if isinstance(agenda_item, dict):
        summary["agenda_item"] = {
            key: agenda_item[key]
            for key in ("id", "branch", "from", "to", "status")
            if key in agenda_item
        }
    return summary


def _workspace_snapshot_payload(
    workspace: Workspace,
    language: str,
    word_set: set[str],
    freq_rank: dict[str, int],
    iteration: int,
    max_iterations: int,
    total_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
) -> dict[str, Any]:
    branch, scores = _best_branch_for_auto_declare(
        workspace, language, word_set, freq_rank
    )
    b = workspace.get_branch(branch)
    decryption = workspace.apply_key(branch)
    branches = []
    for name in workspace.branch_names():
        candidate = workspace.get_branch(name)
        dr, quad = _score_branch_for_panel(
            workspace, name, language, word_set, freq_rank
        )
        branches.append({
            "name": name,
            "mapped_count": len(candidate.key),
            "dict_rate": round(dr, 4) if dr is not None else None,
            "quad": round(quad, 4) if quad is not None else None,
        })
    return {
        "iteration": iteration,
        "max_iterations": max_iterations,
        "branch": branch,
        "mapped_count": len(b.key),
        "scores": scores,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "decryption": decryption,
        "decryption_preview": decryption[:1000],
        "branches": branches,
    }


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
    repair_agenda: list[dict[str, Any]] | None = None,
) -> str:
    """Render ciphertext + current partial decode(s) for this turn.

    The LLM sees this on every iteration so its judgement always centers on
    reading the text, not on scores. Per-branch scores and a floor-trigger
    block give it a mechanical signal of when to declare.
    """
    ct = workspace.cipher_text
    alpha = ct.alphabet
    pt_alpha = workspace.plaintext_alphabet
    canonical_words = ct.words
    n_words = len(canonical_words)
    indices = _select_word_indices(n_words)

    # Render ciphertext tokens in those indices
    cipher_tokens = [_render_word_tokens(canonical_words[i], alpha) for i in indices]
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
        if n_words <= PANEL_WORDS_FULL_LIMIT
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
        branch_words = workspace.effective_words(name)
        branch_word_count = len(branch_words)
        branch_indices = _select_word_indices(branch_word_count)
        decoded = [
            _render_decoded_word(branch_words[i], branch.key, pt_alpha) for i in branch_indices
        ]
        dict_rate, quad = branch_scores[name]
        score_bits = []
        if dict_rate is not None:
            score_bits.append(f"dict={dict_rate:.2f}")
        if quad is not None:
            score_bits.append(f"quad={quad:.2f}")
        score_str = f"  {'  '.join(score_bits)}" if score_bits else ""
        boundary_note = ""
        if branch.word_spans is not None:
            boundary_note = f"  custom_boundaries={branch_word_count}"
        lines.append("")
        lines.append(
            f"### Branch `{name}` — {len(branch.key)}/{alpha.size} symbols mapped"
            f"{score_str}{boundary_note}"
        )
        lines.append("```")
        lines.append(" | ".join(decoded))
        lines.append("```")

    lines.append("")

    no_boundaries = all(len(workspace.effective_words(name)) <= 1 for name in branch_names)

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
            f"self_confidence=0.9, reading_summary='...', "
            f"further_iterations_helpful=false, further_iterations_note='...')`. "
            f"Further manual swaps from this point "
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

    if repair_agenda:
        unresolved = [
            item for item in repair_agenda
            if item.get("status") in {"open", "blocked"}
        ]
        lines.append("### Reading repair agenda")
        for item in repair_agenda[-6:]:
            lines.append(
                f"- #{item.get('id')} `{item.get('branch')}` "
                f"{item.get('from')} -> {item.get('to')} "
                f"[{item.get('status')}]"
            )
        if unresolved:
            lines.append(
                "**Resolve open/blocked repair agenda items before "
                "declaring, or explicitly explain why they remain unresolved.**"
            )
        lines.append("")

    if iters_left == 1:
        lines.append(PENULTIMATE_READING_WORKFLOW_PREFLIGHT)
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
            "If you want to repair the current best branch, especially an "
            "`automated_preflight` baseline, create the repair branch with "
            "`workspace_fork_best` rather than plain `workspace_fork`; plain "
            "`workspace_fork` defaults to empty `main` unless you explicitly "
            "set `from_branch`. "
            "If a branch's decode looks like coherent text in the target language, "
            "fix any obvious residual errors, then do one anchored polish call "
            "with `search_anneal(..., preserve_existing=true, score_fn='combined')` "
            "or `search_homophonic_anneal(..., preserve_existing=true, "
            "solver_profile='zenith_native')`, then read again. If a diagnose "
            "tool returns `boundary_candidates` or a `recommended_next_tool` "
            "using split/merge or `act_apply_boundary_candidate`, do that "
            "before another free anneal. If the character stream reads as "
            "solved but words are visibly split or merged (for example "
            "`THERE | FORE`, `AP | PLY`, `UN | TO`, `WITH | OUT`), apply "
            "those boundary edits before declaring. Prefer one complete "
            "`act_resegment_by_reading` proposal when you can read the whole "
            "stream; use `act_merge_decoded_words` for smaller manual merges "
            "so earlier edits cannot make later numeric word indices stale. "
            "If your best reading changes letters as well as spaces, validate "
            "it first with `decode_validate_reading_repair`; if the character "
            "count matches, apply its boundary pattern with "
            "`act_resegment_from_reading_repair`, then apply targeted "
            "cipher-symbol repairs. Merge likely suffix/piece splits too, "
            "even if the merged spelling is archaic or not in the dictionary, "
            "when leaving the split would shift all following words. "
            "**IMPORTANT:** A partial declaration is not better than continued "
            "work while useful tools remain. A few correct words "
            "(ET, PER, IN, EST…) are not enough to stop early. Before any "
            "non-final partial declaration, you must have either (a) produced "
            "a coherent paraphrasable reading, or (b) actually tried the next "
            "high-leverage tools you would otherwise name as future work: "
            "branch-card comparison, a targeted repair pass, anchored polish, "
            "and, for word islands or suspected ordering problems, transform "
            "suspicion/search. Only declare a low-confidence partial in the "
            "final stretch or after those bigger swings have failed."
        )

    lines.append("")
    lines.append(
        "**Transform check:** If the best branches show only scattered word "
        "islands, low confidence, or you are thinking about columnar/"
        "transposition/period/Vigenere explanations, do not declare early. "
        "Call `observe_transform_pipeline`, then try "
        "`search_transform_homophonic(branch=..., homophonic_budget='screen')` "
        "before giving up on this run. If that screen does not help, say so in "
        "the declaration rationale."
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
    claude_api: Any,
    language: str = "en",
    max_iterations: int = 50,
    cipher_id: str = "unknown",
    prior_context: str | None = None,
    automated_preflight: dict[str, Any] | None = None,
    benchmark_context: Any = None,
    resume_from_artifact: dict[str, Any] | None = None,
    resume_branch: str | None = None,
    parent_artifact_path: str = "",
    verbose: bool = False,
    on_event: Any = None,  # optional callback(event_type: str, payload: dict)
) -> RunArtifact:
    """Run one v2 agent session against a cipher. Returns a full RunArtifact."""

    run_id = uuid.uuid4().hex[:12]
    model_provider = ensure_model_provider(claude_api)

    artifact = RunArtifact(
        run_id=run_id,
        cipher_id=cipher_id,
        model=model_provider.model,
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
        parent_run_id=str((resume_from_artifact or {}).get("run_id") or ""),
        parent_artifact_path=parent_artifact_path,
    )

    def emit(
        event: str,
        payload: dict,
        *,
        outer_iteration: int | None = None,
        inner_step: int | None = None,
        mode: AgentMode | None = None,
    ) -> None:
        artifact.loop_events.append(
            LoopEvent(
                event=event,
                payload=payload,
                outer_iteration=outer_iteration,
                inner_step=inner_step,
                mode=mode.value if mode is not None else None,
            )
        )
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
    if resume_from_artifact:
        install_resume_branches(
            workspace,
            resume_from_artifact,
            branch=resume_branch,
        )
    elif automated_preflight:
        _install_automated_preflight_branch(workspace, automated_preflight)
    executor = WorkspaceToolExecutor(
        workspace=workspace,
        language=language,
        word_set=word_set,
        word_list=word_list,
        pattern_dict=pattern_dict,
        benchmark_context=benchmark_context,
    )
    executor.set_max_iterations(max_iterations)
    if resume_from_artifact:
        executor.repair_agenda = inherited_repair_agenda(resume_from_artifact)
        max_id = max(
            (int(item.get("id") or 0) for item in executor.repair_agenda),
            default=0,
        )
        executor._next_repair_agenda_id = max_id + 1

    # --- initial context ---
    ic_value = ic.index_of_coincidence(cipher_text.tokens, cipher_text.alphabet.size)
    context_parts = [part for part in [prior_context, _preflight_context(automated_preflight)] if part]
    context_msg = initial_context(
        cipher_display=cipher_text.raw,
        alphabet_symbols=cipher_text.alphabet.symbols,
        total_tokens=len(cipher_text.tokens),
        total_words=len(cipher_text.words),
        ic_value=ic_value,
        language=language,
        prior_context="\n\n".join(context_parts) if context_parts else None,
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": context_msg}]
    artifact.messages = messages  # reference, will grow
    if benchmark_context is not None and hasattr(benchmark_context, "to_artifact_dict"):
        artifact.benchmark_context = benchmark_context.to_artifact_dict()

    system = get_system_prompt(language)

    start = time.time()

    def record_usage(response: ModelResponse) -> None:
        usage = response.usage
        artifact.total_input_tokens += usage.input_tokens
        artifact.total_output_tokens += usage.output_tokens
        artifact.total_cache_read_tokens += usage.cache_read_input_tokens
        artifact.estimated_cost_usd = estimate_provider_cost(
            model_provider.provider_name,
            model_provider.model,
            artifact.total_input_tokens,
            artifact.total_output_tokens,
            artifact.total_cache_read_tokens,
        )

    for iteration in range(1, max_iterations + 1):
        workspace.set_iteration(iteration)
        executor.set_iteration(iteration)
        turn_mode = _mode_for_turn(executor, iteration, max_iterations)
        emit(
            "iteration_start",
            {"iteration": iteration, "mode": turn_mode.value},
            outer_iteration=iteration,
            mode=turn_mode,
        )

        if _is_reading_workflow_gate_turn(executor, iteration, max_iterations):
            gate_text = (
                PENULTIMATE_READING_WORKFLOW_PREFLIGHT
                if iteration == max_iterations - 1
                else READING_WORKFLOW_GATE_PREFLIGHT
            )
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": gate_text,
                }],
            })

        if iteration == max_iterations:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": FINAL_ITERATION_PREFLIGHT}],
            })

        # Compress history before sending: stub stale panels and old tool
        # results to keep total input tokens well under rate-limit ceilings.
        send_messages = _compress_history(messages)
        tool_definitions = _tool_definitions_for_turn(
            executor,
            iteration,
            max_iterations,
        )
        executor.set_allowed_tool_names({tool["name"] for tool in tool_definitions})

        try:
            response: ModelResponse = model_provider.send(
                messages=send_messages,
                tools=tool_definitions,
                system=system,
                max_tokens=8192,
            )
        except ModelProviderError as e:
            artifact.status = "error"
            artifact.error_message = f"API error on iteration {iteration}: {e}"
            emit("error", {"message": artifact.error_message})
            break

        record_usage(response)
        assistant_blocks, tool_uses, text_parts = _collect_assistant_blocks(response)
        messages.append({"role": "assistant", "content": assistant_blocks})

        # Capture plan from the first iteration's text
        if iteration == 1 and text_parts:
            artifact.plan = "\n\n".join(text_parts)

        if text_parts:
            emit(
                "agent_text",
                {"iteration": iteration, "text": text_parts[0][:400]},
                outer_iteration=iteration,
                inner_step=0,
                mode=turn_mode,
            )

        # No tool uses → agent has no more to do
        if not tool_uses:
            artifact.status = "exhausted"
            emit(
                "no_tool_calls",
                {"iteration": iteration},
                outer_iteration=iteration,
                inner_step=0,
                mode=turn_mode,
            )
            break

        # Execute tool calls
        tool_results_blocks: list[dict[str, Any]] = []
        inner_step = 1
        inner_retries = 0
        while True:
            tool_results_blocks = []
            gated_tools: list[str] = []
            boundary_count_failures: list[str] = []
            final_declare_needed = False
            final_declare_blocked = False
            for tu in tool_uses:
                emit(
                    "tool_start",
                    {
                        "tool": tu["name"],
                        "arguments": tu.get("input") or {},
                        "inner_step": inner_step,
                        "mode": turn_mode.value,
                    },
                    outer_iteration=iteration,
                    inner_step=inner_step,
                    mode=turn_mode,
                )
                result = executor.execute(tu["name"], tu["input"], tool_use_id=tu["id"])
                if _is_tool_gated_result(result):
                    gated_tools.append(tu["name"])
                if _is_boundary_projection_count_failure(tu["name"], result):
                    boundary_count_failures.append(tu["name"])
                if iteration == max_iterations and tu["name"] == "meta_declare_solution":
                    parsed_result = _parse_json_result(result)
                    if isinstance(parsed_result, dict) and parsed_result.get("accepted") is False:
                        final_declare_blocked = True
                tool_results_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result,
                })
                emit(
                    "tool_call",
                    {
                        "tool": tu["name"],
                        "result_preview": result[:160],
                        "result_summary": _tool_result_summary(result),
                        "inner_step": inner_step,
                        "mode": turn_mode.value,
                    },
                    outer_iteration=iteration,
                    inner_step=inner_step,
                    mode=turn_mode,
                )

            retry_kind = ""
            retry_preflight = ""
            retry_payload: dict[str, Any] = {}
            if iteration == max_iterations and not executor.terminated:
                called_meta = any(tu["name"] == "meta_declare_solution" for tu in tool_uses)
                final_declare_needed = (not called_meta) or final_declare_blocked
            if gated_tools and iteration == max_iterations and not executor.terminated:
                retry_kind = "final_declare_retry"
                retry_preflight = FINAL_DECLARATION_RETRY_PREFLIGHT
                retry_payload = {
                    "iteration": iteration,
                    "inner_step": inner_step,
                    "attempted_tools": gated_tools,
                    "allowed_tools": sorted(executor.allowed_tool_names or []),
                    "declare_blocked": final_declare_blocked,
                    "reason": "gated_tool_on_final_iteration",
                }
            elif gated_tools:
                retry_kind = "gated_tool_retry"
                retry_preflight = BOUNDARY_PROJECTION_RETRY_PREFLIGHT
                retry_payload = {
                    "iteration": iteration,
                    "inner_step": inner_step,
                    "attempted_tools": gated_tools,
                    "allowed_tools": sorted(executor.allowed_tool_names or []),
                }
            elif boundary_count_failures:
                retry_kind = "boundary_projection_count_retry"
                retry_preflight = BOUNDARY_PROJECTION_COUNT_RETRY_PREFLIGHT
                retry_payload = {
                    "iteration": iteration,
                    "inner_step": inner_step,
                    "attempted_tools": boundary_count_failures,
                    "allowed_tools": sorted(executor.allowed_tool_names or []),
                }
            elif final_declare_needed:
                retry_kind = "final_declare_retry"
                retry_preflight = FINAL_DECLARATION_RETRY_PREFLIGHT
                retry_payload = {
                    "iteration": iteration,
                    "inner_step": inner_step,
                    "attempted_tools": [tu["name"] for tu in tool_uses],
                    "allowed_tools": sorted(executor.allowed_tool_names or []),
                    "declare_blocked": final_declare_blocked,
                }
            elif (
                turn_mode is AgentMode.EXPLORE
                and not executor.terminated
                and tool_uses
                and all(tu["name"] in INSPECTION_SANDBOX_TOOL_NAMES for tu in tool_uses)
            ):
                retry_kind = "inspection_sandbox_continue"
                retry_preflight = INSPECTION_SANDBOX_CONTINUE_PREFLIGHT
                retry_payload = {
                    "iteration": iteration,
                    "inner_step": inner_step,
                    "attempted_tools": [tu["name"] for tu in tool_uses],
                    "allowed_tools": sorted(executor.allowed_tool_names or []),
                }
            elif (
                turn_mode is AgentMode.BOUNDARY_PROJECTION
                and not executor.terminated
                and not any(tu["name"] == "meta_declare_solution" for tu in tool_uses)
                and any(tu["name"] in REPAIR_SANDBOX_TOOL_NAMES for tu in tool_uses)
            ):
                retry_kind = "repair_sandbox_continue"
                retry_preflight = REPAIR_SANDBOX_CONTINUE_PREFLIGHT
                retry_payload = {
                    "iteration": iteration,
                    "inner_step": inner_step,
                    "attempted_tools": [tu["name"] for tu in tool_uses],
                    "allowed_tools": sorted(executor.allowed_tool_names or []),
                }

            can_retry_boundary_projection = (
                retry_kind in {"gated_tool_retry", "boundary_projection_count_retry"}
                and turn_mode is AgentMode.BOUNDARY_PROJECTION
                and not executor.terminated
                and inner_retries < BOUNDARY_PROJECTION_MAX_INNER_RETRIES
            )
            can_continue_repair_sandbox = (
                retry_kind == "repair_sandbox_continue"
                and turn_mode is AgentMode.BOUNDARY_PROJECTION
                and not executor.terminated
                and inner_retries < REPAIR_SANDBOX_MAX_INNER_RETRIES
            )
            can_continue_inspection_sandbox = (
                retry_kind == "inspection_sandbox_continue"
                and turn_mode is AgentMode.EXPLORE
                and not executor.terminated
                and inner_retries < INSPECTION_SANDBOX_MAX_INNER_RETRIES
            )
            can_retry_final_declare = (
                retry_kind == "final_declare_retry"
                and turn_mode is AgentMode.DECLARE
                and not executor.terminated
                and inner_retries < FINAL_DECLARATION_MAX_INNER_RETRIES
            )
            if not (
                can_retry_boundary_projection
                or can_continue_repair_sandbox
                or can_continue_inspection_sandbox
                or can_retry_final_declare
            ):
                break

            inner_retries += 1
            emit(
                retry_kind,
                retry_payload,
                outer_iteration=iteration,
                inner_step=inner_step,
                mode=turn_mode,
            )
            messages.append({
                "role": "user",
                "content": [
                    *tool_results_blocks,
                    {"type": "text", "text": retry_preflight},
                ],
            })

            try:
                retry_response = model_provider.send(
                    messages=_compress_history(messages),
                    tools=tool_definitions,
                    system=system,
                    max_tokens=8192,
                )
            except ModelProviderError as e:
                artifact.status = "error"
                artifact.error_message = f"API error on iteration {iteration}: {e}"
                emit("error", {"message": artifact.error_message})
                break

            record_usage(retry_response)
            assistant_blocks, tool_uses, text_parts = _collect_assistant_blocks(
                retry_response
            )
            messages.append({"role": "assistant", "content": assistant_blocks})
            if text_parts:
                emit(
                    "agent_text",
                    {"iteration": iteration, "text": text_parts[0][:400]},
                    outer_iteration=iteration,
                    inner_step=inner_step,
                    mode=turn_mode,
                )
            if not tool_uses:
                artifact.status = "exhausted"
                tool_results_blocks = []
                emit(
                    "no_tool_calls",
                    {"iteration": iteration},
                    outer_iteration=iteration,
                    inner_step=inner_step,
                    mode=turn_mode,
                )
                break
            inner_step += 1

        if artifact.status == "error":
            break
        if artifact.status == "exhausted":
            messages.append({"role": "user", "content": tool_results_blocks})
            break

        emit(
            "workspace_snapshot",
            _workspace_snapshot_payload(
                workspace,
                language,
                word_set,
                freq_rank,
                iteration,
                max_iterations,
                total_tokens=artifact.total_input_tokens + artifact.total_output_tokens,
                estimated_cost_usd=artifact.estimated_cost_usd,
            ),
            outer_iteration=iteration,
            inner_step=inner_step,
            mode=turn_mode,
        )

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
                repair_agenda=executor.repair_agenda,
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
                "Automatic fallback declaration after agent/API error; preserving "
                "the best available branch for inspection rather than losing "
                "run state. "
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
    artifact.repair_agenda = [dict(item) for item in executor.repair_agenda]
    if benchmark_context is not None and hasattr(benchmark_context, "to_artifact_dict"):
        artifact.benchmark_context = benchmark_context.to_artifact_dict()
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
        word_spans=list(branch.word_spans) if branch.word_spans is not None else None,
        token_order=list(branch.token_order) if branch.token_order is not None else None,
        transform_pipeline=(
            dict(branch.transform_pipeline)
            if branch.transform_pipeline is not None
            else None
        ),
    )


def _preflight_context(automated_preflight: dict[str, Any] | None) -> str | None:
    if not automated_preflight:
        return None
    summary = automated_preflight.get("summary")
    if isinstance(summary, str) and summary.strip():
        body = summary
    else:
        body = "Automated native solver preflight ran before the agent loop."
    return (
        f"{body}\n\n"
        "Protected baseline rule: when an `automated_preflight` branch exists, "
        "it is a no-LLM branch. Inspect it before fresh search. If it already "
        "reads coherently, fork from it for experiments and keep the original "
        "branch unchanged for comparison. If that branch is absent, the "
        "preflight did not produce an installable key; continue from `main` "
        "and use the preflight summary only as diagnostic context. Do not "
        "declare an edited branch over a readable preflight branch just because "
        "the spelling looks more modern/classical; prefer the manuscript-"
        "faithful transcription style unless the edited branch clearly reads "
        "better."
    )


def _install_automated_preflight_branch(
    workspace: Workspace,
    automated_preflight: dict[str, Any],
) -> None:
    key = automated_preflight.get("key") or {}
    if not isinstance(key, dict) or not key:
        return
    try:
        workspace.fork("automated_preflight", from_branch="main")
        parsed_key = {int(ct_id): int(pt_id) for ct_id, pt_id in key.items()}
        workspace.set_full_key("automated_preflight", parsed_key)
        workspace.tag("automated_preflight", "automated_preflight")
        workspace.tag("automated_preflight", "no_llm")
    except Exception:  # noqa: BLE001
        # A malformed native preflight should never prevent the LLM run.
        return


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
