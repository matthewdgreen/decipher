"""Terminal renderers for agentic CLI runs."""
from __future__ import annotations

import json
import re
import sys
import threading
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class AgentRunRenderer(Protocol):
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
        ...

    def event(self, event: str, payload: dict[str, Any]) -> None:
        ...

    def finish(self, result: Any) -> None:
        ...


def make_agent_renderer(
    mode: str,
    *,
    stream: Any = None,
    verbose: bool = False,
) -> AgentRunRenderer | None:
    stream = stream or sys.stdout
    if mode == "off":
        return None
    if mode == "jsonl":
        return JsonlAgentRenderer(stream)
    if mode == "pretty":
        return PrettyAgentRenderer(stream)
    if mode == "narrate":
        # Imported lazily to avoid a module import cycle (narrate reuses helpers
        # defined in this module).
        from agent.narrate import NarrateAgentRenderer

        return NarrateAgentRenderer(stream, verbose=verbose)
    return RawAgentRenderer(stream)


# CLI-3 Part 2: plain-English phrasing for blocked tool results. The renderer
# strips the leading tool name (see NarrateAgentRenderer._result_summary), so a
# blocked summary reads as "blocked: <reason in words> — <guidance>". Unknown
# reason codes fall back to the raw reason string.
_BLOCKED_REASON_WORDS = {
    "attestation_not_positive": (
        "declaration needs a fresh positive attestation — reader does not "
        "accept as solution"
    ),
    "attestation_required": (
        "declaration needs a fresh verify attestation on this branch first"
    ),
    "attestation_stale": (
        "the attestation is stale — the branch text changed since it was "
        "written, so reverify before declaring"
    ),
    "repair_transaction_not_ready": (
        "the repair transaction is not ready — no fresh reading is bound to "
        "this branch content yet"
    ),
    "episode_kind_not_available": (
        "that episode kind is not available in the current workflow state"
    ),
    "lead_tool_not_available": (
        "that operator tool is not available to the lead — delegate it "
        "through an episode or experiment"
    ),
    "repair_saturated": (
        "repair is saturated on this candidate — local fixes are exhausted; "
        "broaden or compare instead"
    ),
    "pair_evidence_failed": (
        "the repair evidence did not hold up for this source/reading pair"
    ),
}


def _first_sentence(text: Any) -> str:
    """First sentence of a free-text field (blocked-result guidance)."""
    collapsed = " ".join(str(text or "").split())
    if not collapsed:
        return ""
    for sep in (". ", "! ", "? "):
        idx = collapsed.find(sep)
        if idx != -1:
            return collapsed[: idx + 1]
    return collapsed


def _blocked_phrase(result: dict[str, Any]) -> str:
    reason = str(result.get("reason") or "")
    words = _BLOCKED_REASON_WORDS.get(reason, reason or "action not permitted here")
    phrase = f"blocked: {words}"
    guidance = _first_sentence(result.get("how") or result.get("note"))
    if guidance and guidance.lower() not in phrase.lower():
        phrase += f" — {guidance}"
    return phrase


def summarize_tool_call(tool: str, result: dict[str, Any]) -> str:
    status_early = result.get("status")
    if status_early == "blocked":
        return f"{tool} {_blocked_phrase(result)}"
    if status_early == "duplicate_suppressed":
        return f"{tool} duplicate — already done against unchanged content"
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


def describe_tool_process(tool: str, args: dict[str, Any] | None = None) -> str:
    """Human-sized label for a running tool call."""
    args = args or {}
    branch = args.get("branch")
    suffix = f" on `{branch}`" if branch else ""
    if tool == "search_transform_homophonic":
        profile = args.get("profile", "small")
        budget = args.get("homophonic_budget", "screen")
        program = " + program search" if args.get("include_program_search") else ""
        return f"transform+homophonic search ({profile}, {budget}{program}){suffix}"
    if tool == "search_transform_candidates":
        breadth = args.get("breadth", "broad")
        return f"structural transform candidate screen ({breadth}){suffix}"
    if tool == "search_homophonic_anneal":
        solver = args.get("solver_profile") or args.get("homophonic_solver") or "zenith_native"
        return f"homophonic anneal ({solver}){suffix}"
    if tool == "search_automated_solver":
        budget = args.get("homophonic_budget", "full")
        return f"automated local solver ({budget}){suffix}"
    if tool == "search_anneal":
        return f"substitution anneal{suffix}"
    if tool == "decode_diagnose_and_fix":
        return f"diagnose and apply safe text repairs{suffix}"
    if tool == "decode_validate_reading_repair":
        return f"validate proposed reading repair{suffix}"
    if tool == "act_resegment_by_reading":
        return f"apply reading-based resegmentation{suffix}"
    if tool == "meta_declare_solution":
        return "declare final/partial solution"
    return f"{tool}{suffix}"


# ---------------------------------------------------------------------------
# Plain-English tool glosses (narrate renderer only; CLI-2 spec Part 1).
#
# describe_tool_process (above) is a SHORT running-action LABEL reused by the
# raw/pretty renderers — its output must not change (spec: raw/pretty/jsonl
# untouched). describe_tool_gloss is a SEPARATE, one-line, plain-English
# EXPLANATION of what a tool does and why, aimed at a reader who is not a
# cryptographer. The narrate renderer prints it once per run under the first
# use of each tool. Anything unmapped resolves via the namespace-prefix table
# and finally a generic fallback, so every tool always yields a gloss.
# ---------------------------------------------------------------------------

# Per-kind glosses for the v3 lead's episode_run tool (varies by `kind` arg).
_EPISODE_KIND_GLOSSES = {
    "survey": "spins off a helper to diagnose the cipher and suggest which methods to try",
    "search": "spins off a helper to run a search on a branch and report its best result",
    "reading": "spins off a helper to draft the best plain-language reading of a branch",
    "compare": "spins off a helper to rank the competing branches",
    "repair": "spins off a helper to turn a reading into concrete edits on a fresh copy",
    "verify": "sends the candidate text to a fresh, independent reader who judges whether it reads as real language",
}

# Exact-name glosses for high-value / commonly-seen tools.
_TOOL_GLOSSES = {
    # observe_* — measurements that characterize the cipher
    "observe_frequency": "counts how often each symbol appears; frequent symbols usually map to common letters like E, T, A",
    "observe_ic": "measures how 'clumpy' the letter distribution is — a quick test of what kind of cipher this is",
    "observe_isomorph_clusters": "finds repeated symbol patterns (words that share a shape), which hint at real word structure",
    "observe_cipher_id": "sizes up the cipher — alphabet size, repeats, overall shape — to guess its type",
    "observe_cipher_shape": "sizes up the cipher — alphabet size, repeats, overall shape — to guess its type",
    "observe_diagnosis": "runs a battery of statistical tests to name the most likely cipher family",
    "observe_homophone_distribution": "checks whether several different symbols seem to stand for the same letter (a homophonic cipher)",
    "observe_kasiski": "looks for a repeating key period, the signature of a polyalphabetic cipher",
    "observe_periodic_ic": "looks for a repeating key period, the signature of a polyalphabetic cipher",
    "observe_periodic_shift_candidates": "looks for a repeating key period, the signature of a polyalphabetic cipher",
    "observe_phase_frequency": "counts symbol frequencies within each slot of a repeating key",
    "observe_language_models": "lists the language models used to score how word-like a decode is",
    "observe_transform_suspicion": "checks whether the text was scrambled (transposed) before substitution",
    "observe_transform_pipeline": "shows any un-scrambling steps currently applied to a branch",
    # search_* — automated hunts for a better key
    "search_hill_climb": "hunts for a better symbol-to-letter key by repeatedly trying small swaps and keeping improvements",
    "search_anneal": "searches for a good key using simulated annealing — bold guesses early, careful refinements later",
    "search_homophonic_anneal": "same idea as annealing, but for homophonic ciphers where one letter has many symbols",
    "search_automated_solver": "runs the built-in automatic solver end to end and reports its best decode",
    "search_quagmire3_keyword_alphabet": "tries to crack a keyword-based polyalphabetic (Quagmire) cipher",
    "search_transform_candidates": "tries undoing a scrambling/transposition step to see if the text then reads",
    "search_transform_homophonic": "tries undoing a scramble AND solving a homophonic key together",
    "search_pure_transposition": "tries reordering the letters (transposition) to recover readable text",
    "search_periodic_polyalphabetic": "searches for a repeating-key cipher solution across candidate periods",
    "search_null_masks": "tries dropping 'null' filler symbols that carry no meaning",
    "search_word_repair_menu": "proposes fixes for words that are close to real language but not quite right",
    # act_* — edits to the working key / decode
    "act_set_mapping": "commits one symbol→letter guess to the working key",
    "act_bulk_set": "commits several symbol→letter guesses to the key at once",
    "act_anchor_word": "locks in a guessed word, fixing all of its letters in the key",
    "act_clear_mapping": "removes a symbol→letter guess from the key",
    "act_swap_decoded": "swaps two letters in the current decode to test an alternative",
    "act_split_cipher_word": "splits one run-together word into two",
    "act_merge_cipher_words": "joins two words into one",
    "act_merge_decoded_words": "joins two decoded words into one",
    "act_resegment_by_reading": "re-splits the text into words to match a proposed reading",
    "act_resegment_from_reading_repair": "re-splits the text into words to match a proposed reading",
    "act_resegment_window_by_reading": "re-splits one stretch of text into words to match a proposed reading",
    "act_apply_word_repair": "applies a proposed fix to a specific word",
    "act_apply_transform_pipeline": "installs an un-scrambling step onto a branch",
    "act_set_periodic_key": "sets the repeating key for a polyalphabetic decode",
    "act_set_periodic_shift": "sets one slot of the repeating key for a polyalphabetic decode",
    "act_adjust_periodic_shift": "nudges one slot of the repeating key",
    "act_set_model_variant": "switches which language model is used to score decodes",
    # decode_* — read or repair the current text
    "decode_show": "shows the current decoded text so you can read what it says so far",
    "decode_show_phases": "shows the decode split out by each slot of a repeating key",
    "decode_unmapped_report": "lists the symbols that still have no letter assigned",
    "decode_ngram_heatmap": "highlights which parts of the decode look like real language and which look like gibberish",
    "decode_letter_stats": "reports letter-level statistics of the current decode",
    "decode_ambiguous_letter": "flags a letter whose mapping looks shaky and suggests alternatives",
    "decode_absent_letter_candidates": "points out common letters that are missing from the decode",
    "decode_diagnose": "diagnoses problems in the current decode",
    "decode_diagnose_and_fix": "diagnoses problems in the current decode and applies safe repairs",
    "decode_repair_no_boundary": "re-splits a run-together decode into words",
    "decode_validate_reading_repair": "checks a proposed reading fix before it is applied",
    "decode_plan_word_repair": "plans candidate word fixes for the reader to choose from",
    "decode_plan_word_repair_menu": "plans candidate word fixes for the reader to choose from",
    # score_* — how language-like is the decode
    "score_panel": "scores the current decode on several measures of how language-like it is",
    "score_dictionary": "measures how many decoded words are real dictionary words",
    "score_quadgram": "scores the decode by how natural its four-letter sequences look",
    "score_delta": "estimates how much a proposed change would improve or hurt the decode",
    "score_delta_if_remapped": "estimates how much a proposed remapping would improve or hurt the decode",
    # corpus_* — consult the word list
    "corpus_lookup_word": "looks up a word in the target-language word list",
    "corpus_word_candidates": "suggests real words that could fill a gap in the decode",
    # workspace_* — manage exploration branches
    "workspace_fork": "makes a copy of a branch so an idea can be explored without disturbing the original",
    "workspace_fork_best": "copies the best-scoring branch to build on it",
    "workspace_create_hypothesis_branch": "starts a new branch to test a specific hypothesis about the cipher",
    "workspace_merge": "merges progress from one branch into another",
    "workspace_compare": "compares branches side by side to see which decode reads better",
    "workspace_branch_cards": "lists the current branches and what each one is exploring",
    "workspace_list_branches": "lists the current branches and what each one is exploring",
    "workspace_hypothesis_cards": "lists the current hypotheses and their status",
    "workspace_hypothesis_next_steps": "suggests the next step for a hypothesis",
    "workspace_update_hypothesis": "updates the status of a hypothesis on a branch",
    "workspace_reject_hypothesis": "marks a hypothesis as ruled out",
    "workspace_delete": "removes a branch that is no longer needed",
    # hypothesis_* / composite v3 actions
    "hypothesis_apply_reading": "turns a proposed reading of a branch into concrete key and word-boundary edits on a fresh copy",
    "hypothesis_test_word": "tests whether a specific word guess holds up against the evidence",
    "branch_adjudicate": "ranks the competing branches to pick the most promising one",
    # repair_* — outstanding-fix bookkeeping
    "repair_agenda_list": "lists the outstanding fixes the decode still needs",
    "repair_agenda_update": "updates an item on the list of outstanding fixes",
    "repair_agenda_unresolved": "lists the fixes that are still unresolved",
    # inspect_* / list_* — benchmark reference material
    "inspect_related_solution": "reads a related solved cipher provided for context",
    "inspect_related_transcription": "reads a related transcription provided for context",
    "inspect_associated_document": "reads an associated reference document",
    "inspect_benchmark_context": "reads the benchmark context notes for this cipher",
    "list_related_records": "lists related reference records available for this cipher",
    "list_associated_documents": "lists associated reference documents available for this cipher",
    # run_python — escape hatch
    "run_python": "runs a small custom Python snippet as an escape hatch for a one-off calculation",
    # meta_* — run-level decisions
    "meta_declare_solution": "declares the current decode as the final solution",
    "meta_declare_unsolved": "gives up and records the best partial result with what was tried",
    "meta_request_tool": "asks for a tool that isn't currently available",
    "meta_attest_reading_comprehensibility": "records a judgement of how readable the decode is",
    # v3 lead-only helpers
    "episode_install_branch": "brings a helper's result branch into the main workspace",
    "experiment_submit": "queues a long-running solver to work in the background while you continue",
    "experiment_collect": "checks on background solver runs and installs a finished one if it looks good",
}

# Namespace-prefix fallbacks: used when an exact name is not listed above.
_TOOL_GLOSS_PREFIXES = (
    ("observe_", "measures a statistical property of the cipher to guide the next move"),
    ("search_review_", "reviews the candidates a previous search produced before installing one"),
    ("search_", "searches for a better decode with an automated solver"),
    ("act_install_", "installs a candidate solution onto a branch so it becomes the working decode"),
    ("act_", "changes the working decode or key on a branch"),
    ("decode_", "inspects or repairs the current decoded text"),
    ("score_", "scores how language-like the current decode is"),
    ("corpus_", "consults the target-language word list"),
    ("workspace_", "manages the branches used to explore competing ideas"),
    ("hypothesis_", "works with a hypothesis about the cipher"),
    ("repair_", "manages the list of outstanding decode fixes"),
    ("inspect_", "reads reference material provided for context"),
    ("list_", "lists reference material available for this cipher"),
    ("experiment_", "manages background solver runs"),
    ("episode_", "manages a focused helper sub-task"),
    ("meta_", "records a run-level decision"),
)


def describe_tool_gloss(tool: str, args: dict[str, Any] | None = None) -> str:
    """One-line, plain-English explanation of what a tool does and why.

    Narrate-only (Part 1 of the CLI-2 narrative spec). Always returns a
    non-empty string: an exact-name gloss when known, else a namespace-prefix
    gloss, else a generic fallback.
    """
    args = args or {}
    if tool == "episode_run":
        kind = str(args.get("kind") or "").strip().lower()
        return _EPISODE_KIND_GLOSSES.get(
            kind, "spins off a focused helper to work a sub-task in isolation"
        )
    exact = _TOOL_GLOSSES.get(tool)
    if exact:
        return exact
    for prefix, gloss in _TOOL_GLOSS_PREFIXES:
        if tool.startswith(prefix):
            return gloss
    return f"runs the {tool} tool"


# ---------------------------------------------------------------------------
# Plain-English ACTION lines (CLI-3 spec Part 2). describe_tool_action is the
# args-aware, present-progressive sentence the default narrate display prints as
# an `⏺` line — "what the agent is DOING", not the tool name. Specific patterns
# exist for every v3 lead tool and the common v2 tools; anything else falls back
# to describe_tool_gloss's table, and finally to "Running <tool>".
# ---------------------------------------------------------------------------
def _truncate(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def describe_tool_action(tool: str, args: dict[str, Any] | None = None) -> str:
    """One-line, present-progressive description of what a lead tool call does."""
    args = args or {}

    if tool == "episode_run":
        kind = str(args.get("kind") or "").strip() or "focused"
        goal = _truncate(args.get("goal"), 90)
        base = f"Launching a {kind} episode"
        return f"{base}: {goal}" if goal else base
    if tool == "episode_install_branch":
        branch = str(args.get("branch") or "?")
        as_name = str(args.get("as_name") or branch)
        return f"Installing episode branch '{branch}' as '{as_name}'"
    if tool == "repair_transaction":
        branch = str(args.get("branch") or "?")
        reading_id = str(args.get("reading_id") or "")
        if reading_id:
            return (
                f"Attempting a validated repair of '{branch}' bound to reading "
                f"{reading_id[:8]}"
            )
        return f"Attempting a validated repair of '{branch}' (newest bound reading)"
    if tool == "branch_adjudicate":
        branches = [str(b) for b in (args.get("branches") or [])]
        joined = ", ".join(branches) if branches else "the current branches"
        return f"Comparing branches: {joined}"
    if tool == "experiment_submit":
        exp_type = str(args.get("type") or "automated_solver")
        branch = str(args.get("branch") or "?")
        if args.get("resubmit"):
            return f"Resubmitting experiment {str(args['resubmit'])[:8]}"
        return f"Queuing a {exp_type} experiment on '{branch}'"
    if tool == "experiment_collect":
        experiment_id = str(args.get("experiment_id") or "")
        if experiment_id:
            return f"Collecting experiment {experiment_id[:8]} results"
        return "Collecting background experiment results"
    if tool == "meta_declare_solution":
        return f"Declaring the solution on '{args.get('branch') or '?'}'"
    if tool == "meta_declare_unsolved":
        best = args.get("best_branch")
        if best:
            return f"Declaring the run unsolved (best: '{best}')"
        return "Declaring the run unsolved"
    if tool == "workspace_create_hypothesis_branch":
        new_name = str(args.get("new_name") or "?")
        mode = str(args.get("cipher_mode") or "hypothesis")
        return f"Opening hypothesis branch '{new_name}' ({mode})"
    if tool == "decode_show":
        return f"Reading the decode of '{args.get('branch') or '?'}'"
    if tool == "repair_agenda_list":
        return "Reviewing the repair agenda"
    if tool == "repair_agenda_update":
        item_id = args.get("item_id", args.get("id"))
        return f"Updating repair-agenda item {item_id}"
    if tool == "act_set_model_variant":
        variant = str(args.get("variant") or "?")
        language = str(args.get("language") or "").strip()
        prefix = f"{language} " if language else ""
        return f"Switching the {prefix}language model to '{variant}'"

    # Fallback: reuse the gloss table; a generic gloss ("runs the X tool")
    # degrades to a present-progressive "Running X".
    gloss = describe_tool_gloss(tool, args)
    if gloss and gloss != f"runs the {tool} tool":
        return gloss
    return f"Running {tool}"


class RawAgentRenderer:
    """Current compact event stream for scripts and debugging."""

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream or sys.stdout

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
        elif event == "tool_start":
            tool = str(payload.get("tool", "tool"))
            args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
            print(f" [{describe_tool_process(tool, args)}", end="", flush=True, file=self.stream)
        elif event == "tool_call":
            print(" done]", end="", flush=True, file=self.stream)
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
        self._write({
            "event": "test_start",
            "test_id": test_id,
            "description": description,
            "model": model,
            "max_iterations": max_iterations,
            "language": language,
            "source": source,
            "agent_loop": agent_loop,
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
    active_tool: str = ""
    active_tool_started_at: float = 0.0
    active_tool_ticks: int = 0


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
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None

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
        elif event == "tool_start":
            tool = str(payload.get("tool", "tool"))
            args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
            self.state.active_tool = describe_tool_process(tool, args)
            self.state.active_tool_started_at = time.monotonic()
            self.state.active_tool_ticks = 0
            self._add_log(f"running: {self.state.active_tool}")
            self._start_heartbeat()
        elif event == "tool_call":
            self._stop_heartbeat()
            tool = str(payload.get("tool", "tool"))
            summary = summarize_tool_call(tool, payload.get("result_summary") or {})
            elapsed = ""
            if self.state.active_tool_started_at:
                elapsed = f" {time.monotonic() - self.state.active_tool_started_at:.1f}s"
            self._add_log(f"done{elapsed}: {summary}")
            self.state.active_tool = ""
            self.state.active_tool_started_at = 0.0
            self.state.active_tool_ticks = 0
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
            self._stop_heartbeat()
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
        self._stop_heartbeat()
        if self._live:
            self._live.update(self._render_final(result))
            self._live.stop()
        else:
            print(
                f"Status: {result.status}  comparison to known ground-truth plaintext: "
                f"char={result.char_accuracy:.1%} word={result.word_accuracy:.1%}  "
                f"artifact={result.artifact_path}",
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

    def _start_heartbeat(self) -> None:
        if not self._rich or not self._live:
            return
        self._stop_heartbeat()
        stop = threading.Event()
        self._heartbeat_stop = stop

        def run() -> None:
            while not stop.wait(1.0):
                self.state.active_tool_ticks += 1
                self._refresh()

        self._heartbeat_thread = threading.Thread(
            target=run,
            name="decipher-pretty-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        stop = self._heartbeat_stop
        if stop is not None:
            stop.set()
        self._heartbeat_stop = None
        self._heartbeat_thread = None

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
        if self.state.active_tool:
            elapsed = max(0.0, time.monotonic() - self.state.active_tool_started_at)
            dots = "." * (1 + (self.state.active_tool_ticks % 6))
            agent_lines.append(
                f"[bold]Running:[/bold] {self.state.active_tool}  "
                f"[dim]{elapsed:.0f}s {dots}[/dim]"
            )
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
            "Comparison to known ground-truth plaintext:",
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
