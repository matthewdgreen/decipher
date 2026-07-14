"""Lead context builder for the v3 investigation loop (C2, M1 spec Part 2).

``build_lead_context`` is a PURE function: it renders the lead's per-turn view
entirely from ``InvestigationState`` (+ the executor for the side-effect-free
branch-card renderer). Rebuilt every turn, so the transcript is never
load-bearing. Middle-blindness is fixed by a rotating full-decode window whose
offset advances each turn.

The returned value is an Anthropic-style ``messages`` list. Sessions convert it
to their native format with the existing provider converters, so the last-2
exchanges (section 6) are re-emitted verbatim as native message dicts —
flattening them to text would silently kill OpenAI reasoning passback and break
function_call pairing (F1).
"""
from __future__ import annotations

import json
from typing import Any

from agent.loop_shared import _best_branch_for_auto_declare
from agent.prompts_v2 import LANGUAGE_NOTES
from analysis import cipher_id as cipher_id_analysis
from analysis import ic
from analysis.dictionary import LANGUAGE_NAMES
from investigation.state import InvestigationState

CHARS_PER_TOKEN = 4  # crude estimator for budget bookkeeping

# Per-section character caps (before the global budget clamp). Deterministic.
_CIPHER_RENDER_CAP = 6000
_FINGERPRINT_CAP = 2500
_BRANCH_CARDS_CAP = 6000
_BOARD_CAP = 2000
_EPISODE_LEDGER_CAP = 2000
_EVIDENCE_CAP = 2500
_WINDOW_CAP = 4000
_EXTERNAL_CAP = 4000
_PREFIX_FLOOR = 2000

# Rotating full-decode window size (in cipher tokens).
DEFAULT_WINDOW_TOKENS = 400
# How many trailing evidence-log entries to render.
DEFAULT_EVIDENCE_ENTRIES = 6
# How many branch cards to render (top-K by internal scoring).
DEFAULT_BRANCH_CARDS = 4


# ---------------------------------------------------------------------------
# v3 system prompt — a NEW, much shorter brief (~1/3 of v2). No cipher-mode
# playbooks, no reading-repair discipline essay: the hypothesis board and tool
# results carry state, and M1 targets substitution-family synthetics.
# ---------------------------------------------------------------------------

_V3_SYSTEM_TEMPLATE = """\
You are a cryptanalyst working a manuscript cipher. Target language: \
**{language_name}**.

How this loop works
- Every turn you are handed a freshly rebuilt view of the investigation: the \
cipher, a diagnostic fingerprint, the current branch cards, the hypothesis \
board, recent evidence, and a rotating window over the full decode. The view \
is the source of truth — you do not need to re-derive facts it already states.
- You act through tools. A `main` branch exists; fork branches to explore \
competing hypotheses, set mappings, run searches, and read the decode. Tool \
results are your instruments; trust the decoded text over any single score.
- Work at the hypothesis level: form a hypothesis about the cipher, test it, \
keep evidence for and against on the branch, and move on when it is settled.

Reading the evidence
- The measured facts (alphabet size, IC, fingerprint) are already computed — \
use them to pick a method rather than re-measuring them. A cipher alphabet \
much larger than 26 is homophonic (several symbols per letter); its \
constraint-satisfaction score is naturally below 1.0 and is not an error. \
Judge a branch by whether its decode reads as language, not by any single \
number.

Delegating
- If an existing branch already reads as solved (for example an installed \
`automated_preflight`), verify it with a quick read and declare it directly — \
do not spend budget on episodes first.
- You can spin off a focused worker with `episode_run(kind, goal, branches, …)`. \
Four kinds exist: `survey` (diagnose the cipher and suspected modes), `search` \
(run one search tool and its review/install companions on a branch), `reading` \
(draft a best reading of a branch), and `compare` (rank competing branches). An \
episode runs in an isolated copy of the branches you name and returns a result \
summary plus branch snapshots; nothing it does touches your workspace until you \
call `episode_install_branch`. Integrate an episode's findings and snapshots \
explicitly — they do not apply themselves.

Finishing
- When a branch reads as coherent {language_name}, call \
`meta_declare_solution` with that branch and your confidence. If the cipher \
resists every hypothesis you can mount, call `meta_declare_unsolved` with your \
best branch and what you tried. There are no gate prerequisites — declare when \
the reading justifies it, and say so honestly if it is only a partial.
{language_notes}"""


def build_v3_system_prompt(language: str = "en") -> str:
    language_name = LANGUAGE_NAMES.get(language, "Unknown")
    notes = LANGUAGE_NOTES.get(language, "")
    notes_section = f"\n{notes}" if notes.strip() else ""
    return _V3_SYSTEM_TEMPLATE.format(
        language_name=language_name,
        language_notes=notes_section,
    )


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n…[truncated {len(text) - cap} chars]"


# ---------------------------------------------------------------------------
# Section renderers (each deterministic, each capped)
# ---------------------------------------------------------------------------


def _render_framing(state: InvestigationState) -> str:
    # Kept turn-INDEPENDENT so this + the cipher form a byte-stable, cacheable
    # prefix across turns. The dynamic turn marker lives in the view section.
    language_name = LANGUAGE_NAMES.get(state.language, "Unknown")
    return (
        f"## Task\n"
        f"Decipher the manuscript below. Target language: {language_name}. "
        f"The sections after the cipher are rebuilt from the investigation "
        f"state each turn; act on them directly."
    )


def _render_cipher(state: InvestigationState) -> str:
    ct = state.cipher
    ic_value = ic.index_of_coincidence(ct.tokens, ct.alphabet.size)
    symbols = ct.alphabet.symbols
    preview = ", ".join(symbols[:40])
    if len(symbols) > 40:
        preview += f", … ({len(symbols) - 40} more)"
    body = _truncate(ct.display(), _CIPHER_RENDER_CAP)
    return (
        "## Cipher\n"
        f"- Symbol alphabet: {ct.alphabet.size} symbols — {preview}\n"
        f"- Tokens: {len(ct.tokens)}   Words: {len(ct.words)}   "
        f"Index of coincidence: {ic_value:.4f}\n"
        "```\n"
        f"{body}\n"
        "```"
    )


def _render_fingerprint(state: InvestigationState) -> str:
    ct = state.cipher
    fp = cipher_id_analysis.compute_cipher_fingerprint(
        ct.tokens,
        ct.alphabet.size,
        language=state.language,
        word_group_count=len(ct.words),
    )
    text = cipher_id_analysis.format_fingerprint_for_context(fp)
    return "## Diagnostic fingerprint\n" + _truncate(text, _FINGERPRINT_CAP)


def _render_external_context(state: InvestigationState) -> str:
    """External / benchmark context as its OWN stable section (R3).

    Rendered in the cacheable prefix every turn so it never scrolls out of
    the recent-evidence window. Returns "" when there is no external context
    so the caller can omit the section entirely.
    """
    text = (state.external_context or "").strip()
    if not text:
        return ""
    return "## External context\n" + _truncate(text, _EXTERNAL_CAP)


def _branch_card_line(card: dict[str, Any]) -> str:
    scores = card.get("scores") or {}
    dict_rate = scores.get("dict_rate")
    quad = scores.get("quad")
    dr = f"{dict_rate:.3f}" if isinstance(dict_rate, (int, float)) else "n/a"
    qd = f"{quad:.3f}" if isinstance(quad, (int, float)) else "n/a"
    excerpt = str(card.get("decoded_excerpt") or "").strip()
    tags = ", ".join(card.get("tags") or [])
    header = (
        f"- `{card.get('branch')}` mapped={card.get('mapped_count')} "
        f"dict_rate={dr} quad={qd}"
    )
    if tags:
        header += f" [{tags}]"
    if card.get("protected_baseline"):
        header += " (protected baseline)"
    return header + (f"\n    {excerpt}" if excerpt else "")


def _render_branch_cards(
    state: InvestigationState, executor: Any, top_k: int
) -> str:
    ws = state.workspace
    names = ws.branch_names()
    cards = [executor._branch_card(name) for name in names]

    def score_key(card: dict[str, Any]) -> tuple[float, float, int]:
        scores = card.get("scores") or {}
        dict_rate = scores.get("dict_rate")
        quad = scores.get("quad")
        return (
            dict_rate if isinstance(dict_rate, (int, float)) else float("-inf"),
            quad if isinstance(quad, (int, float)) else float("-inf"),
            int(card.get("mapped_count") or 0),
        )

    cards.sort(key=score_key, reverse=True)
    lines = [_branch_card_line(card) for card in cards[:top_k]]
    hidden = len(cards) - len(lines)
    body = "\n".join(lines) if lines else "(no branches)"
    if hidden > 0:
        body += f"\n- … {hidden} more branch(es) not shown"
    return _truncate("## Branch cards (top-K by internal score)\n" + body, _BRANCH_CARDS_CAP)


def _render_hypothesis_board(state: InvestigationState) -> str:
    board = state.hypothesis_cards()
    if not board:
        return "## Hypothesis board\n(no cipher-mode hypotheses recorded)"
    lines = []
    for card in board:
        lines.append(
            f"- `{card['branch']}`: {card['cipher_mode']} "
            f"[{card['mode_status']}]"
            + (
                f" — next: {card['next_recommended_action']}"
                if card.get("next_recommended_action")
                else ""
            )
        )
    return _truncate("## Hypothesis board\n" + "\n".join(lines), _BOARD_CAP)


def _render_episode_ledger(state: InvestigationState, n: int = 3) -> str:
    """Render the last ``n`` completed episodes (C2 section between board and
    recent evidence). Kind, goal, status, summary, and installed snapshot names.
    """
    ledger = state.episode_ledger[-n:]
    if not ledger:
        return ""
    lines = []
    for entry in ledger:
        kind = entry.get("kind", "?")
        status = entry.get("status", "?")
        goal = str(entry.get("goal") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        snaps = entry.get("branch_snapshots") or []
        snap_names = ", ".join(
            s.get("name", "?") for s in snaps if isinstance(s, dict)
        )
        head = f"- [{kind} {status}] {goal}".rstrip()
        if summary:
            head += f"\n    {summary}"
        if snap_names:
            head += f"\n    snapshots: {snap_names}"
        lines.append(head)
    return _truncate("## Recent episodes\n" + "\n".join(lines), _EPISODE_LEDGER_CAP)


def _render_evidence(state: InvestigationState, n: int) -> str:
    entries = state.evidence_log[-n:]
    if not entries:
        return "## Evidence log\n(empty)"
    lines = []
    for entry in entries:
        summary = entry.summary or ""
        lines.append(f"- [t{entry.turn} {entry.kind}] {summary}".rstrip())
    return _truncate("## Recent evidence\n" + "\n".join(lines), _EVIDENCE_CAP)


def _snap_offset_to_span(raw_offset: int, spans: list[tuple[int, int]]) -> int:
    """Snap ``raw_offset`` down to the nearest word-span start (F6)."""
    best = 0
    for start, _end in spans:
        if start <= raw_offset:
            best = start
        else:
            break
    return best


def rotating_window_bounds(
    total_tokens: int, turn: int, window_tokens: int,
    spans: list[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Return (offset, end) for the rotating window (deterministic).

    Offset is ``(turn * window) mod len`` snapped to a word-span start when
    spans are supplied. Over ``ceil(len / window)`` turns the whole cipher is
    seen.
    """
    if total_tokens <= 0:
        return 0, 0
    window = min(max(1, window_tokens), total_tokens)
    raw_offset = (turn * window) % total_tokens
    offset = raw_offset
    if spans:
        offset = _snap_offset_to_span(raw_offset, spans)
    end = min(offset + window, total_tokens)
    return offset, end


def _render_window(
    state: InvestigationState, executor: Any, turn: int, window_tokens: int
) -> str:
    ws = state.workspace
    best, _scores = _best_branch_for_auto_declare(
        ws, state.language, executor.word_set, executor._freq_rank
    )
    tokens = ws.effective_tokens(best)
    total = len(tokens)
    if total == 0:
        return "## Full-decode window\n(empty cipher)"
    spans = ws.effective_word_spans(best)
    offset, end = rotating_window_bounds(total, turn, window_tokens, spans)
    branch = ws.get_branch(best)
    pt = ws.plaintext_alphabet

    def _decode(t: int) -> str:
        return pt.symbol_for(branch.key[t]) if t in branch.key else "?"

    # R6: insert word separators at effective_word_spans boundaries so
    # word-structured ciphers read as words, not character soup. Within a
    # word, multi-symbol plaintext alphabets keep a space between symbols;
    # single-char alphabets concatenate.
    sym_sep = " " if pt._multisym else ""
    if spans:
        word_starts = {start for start, _e in spans}
        word_sep = " | " if pt._multisym else " "
        parts: list[str] = []
        for idx in range(offset, end):
            if parts:
                parts.append(word_sep if idx in word_starts else sym_sep)
            parts.append(_decode(tokens[idx]))
        rendered = "".join(parts)
    else:
        rendered = sym_sep.join(_decode(t) for t in tokens[offset:end])
    header = (
        f"## Full-decode window (branch `{best}`, tokens {offset}–{end} of "
        f"{total}; rotates each turn)\n"
    )
    return _truncate(header + rendered, _WINDOW_CAP)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _group_exchanges(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group a flat native-message list into whole exchanges.

    Each exchange starts at an assistant message and absorbs the immediately
    following non-assistant (tool_result) messages. The F1 budget rule drops
    whole exchanges — never splitting a tool_use from its tool_result
    (unpaired exchanges 400 on the Responses API).
    """
    groups: list[list[dict[str, Any]]] = []
    for message in messages:
        if message.get("role") == "assistant" or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def _group_chars(groups: list[list[dict[str, Any]]]) -> int:
    return sum(len(json.dumps(m, default=str)) for g in groups for m in g)


def _copy_message(message: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of a native message dict with a copied content list.

    Prevents the returned context from aliasing ``state.recent_exchanges`` so a
    session that transforms the context in place cannot corrupt durable state
    (R8b). Inner content blocks are shared (they are treated as immutable).
    """
    content = message.get("content")
    if isinstance(content, list):
        return {**message, "content": list(content)}
    return dict(message)


def build_lead_context(
    state: InvestigationState,
    executor: Any,
    turn: int,
    token_budget: int = 20000,
    max_turns: int | None = None,
    *,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    evidence_entries: int = DEFAULT_EVIDENCE_ENTRIES,
    branch_cards: int = DEFAULT_BRANCH_CARDS,
) -> list[dict[str, Any]]:
    """Render the lead's per-turn messages from state (pure, deterministic).

    Returns an Anthropic-style messages list that preserves strict role
    alternation (so the Anthropic path via GenericChatSession is valid too):
      1. a STABLE-prefix user message (framing + language notes + cipher +
         fingerprint) — the cacheable prefix (turn >= 2);
      2. the last-2 exchanges verbatim (native message dicts, section 6);
      3. a DYNAMIC view (branch cards, hypothesis board, recent evidence,
         rotating decode window). To avoid two consecutive user turns, the view
         text is appended to the last exchange's user turn (mirroring v2's
         panel-in-tool-result pattern); with no exchanges it is merged into the
         single stable-prefix user turn.
    The tool_result blocks inside recent exchanges are preserved verbatim
    (untransformed) — only an extra text block is appended (F1).
    """
    total_char_budget = max(1, token_budget) * CHARS_PER_TOKEN

    # --- stable prefix (sections 1, 2 + external context) ---
    prefix_sections = [
        _render_framing(state),
        _render_cipher(state),
        _render_fingerprint(state),
    ]
    external_section = _render_external_context(state)
    if external_section:
        prefix_sections.append(external_section)
    prefix_text = "\n\n".join(prefix_sections)
    prefix_cap = max(_PREFIX_FLOOR, total_char_budget // 2)
    prefix_text = _truncate(prefix_text, prefix_cap)

    # --- dynamic view (sections 3, 4, 5, 7) ---
    # R4: show the turn budget so the model knows its remaining runway.
    turn_marker = f"turn {turn} of {max_turns}" if max_turns else f"turn {turn}"
    view_sections = [
        f"## Investigation state ({turn_marker})",
        _render_branch_cards(state, executor, branch_cards),
        _render_hypothesis_board(state),
    ]
    episode_section = _render_episode_ledger(state)
    if episode_section:
        view_sections.append(episode_section)
    view_sections += [
        _render_evidence(state, evidence_entries),
        _render_window(state, executor, turn, window_tokens),
    ]
    view_text = "\n\n".join(view_sections)

    # Global clamp: keep the stable prefix (cacheable) and truncate the view to
    # fit the remaining budget so total rendered text respects token_budget.
    remaining = max(0, total_char_budget - len(prefix_text))
    view_text = _truncate(view_text, remaining)
    view_block = {"type": "text", "text": view_text}
    # C2 stable-prefix breakpoint: mark the cacheable prefix text block. An
    # AnthropicSession converts this ``cache_hint`` into a ``cache_control``
    # breakpoint; every other send path strips it (no provider sees the field).
    prefix_block = {"type": "text", "text": prefix_text, "cache_hint": True}

    # F1 budget rule (R2): the recent exchanges are rendered verbatim as
    # native dicts, so they carry their own weight. While the fully rendered
    # context exceeds token_budget, drop the OLDEST WHOLE exchange — never
    # split one (a tool_result without its tool_use, or a reasoning item
    # without its siblings, 400s on the Responses API).
    groups = _group_exchanges(state.recent_exchanges)
    fixed_chars = len(prefix_text) + len(view_text)
    while groups and fixed_chars + _group_chars(groups) > total_char_budget:
        groups.pop(0)
    exchanges = [m for group in groups for m in group]
    if not exchanges:
        # No prior exchanges: one user turn carrying prefix + view.
        return [
            {"role": "user", "content": [prefix_block, view_block]},
        ]

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [prefix_block]},
    ]
    # Section 6: exchanges verbatim, except the view is appended to the final
    # user turn to keep roles alternating. R8(b): return COPIES of the state's
    # exchange dicts — the caller/session must never alias (and thus mutate)
    # ``state.recent_exchanges`` while transforming the context.
    messages.extend(_copy_message(m) for m in exchanges[:-1])
    last = exchanges[-1]
    if last.get("role") == "user":
        last_content = last.get("content")
        base = list(last_content) if isinstance(last_content, list) else [last_content]
        messages.append({**last, "content": [*base, view_block]})
    else:
        messages.append(_copy_message(last))
        messages.append({"role": "user", "content": [view_block]})
    return messages
