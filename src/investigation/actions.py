"""Composite hypothesis actions for the v3 surfaces (M3 spec Part 2-5).

Three composite tools live ONLY here (never in ``agent/tools_v2.py``), so v2's
``TOOL_DEFINITIONS`` / ``VALID_TOOL_NAMES`` and TOOLS.md are unchanged and a v2
run can never see or call them:

- ``hypothesis_apply_reading`` — compile an accepted Reading into key edits +
  boundary changes in ONE step, on a fork (Part 3; absorbs plan Phase 6).
- ``hypothesis_test_word`` — same-length word probe with a menu-backed and an
  injected path, both parity-by-construction with the word-repair library
  (Part 4).
- ``branch_adjudicate`` — read-only, packet-based branch comparison (Part 5).

``execute_composite`` is the single dispatcher both hosts call — the v3 lead
(``loop_v3._dispatch_tool`` checks ``COMPOSITE_TOOL_NAMES`` before
``executor.execute``) and the episode runner's tool loop (which dispatches a
composite only when the name is in ``spec.toolset``). The dispatch site appends
one ToolCall to ``executor.call_log`` with the serialized result and the M2
``episode_id`` / ``iteration`` stamping; episode-side results pass through
``_filter_next_tool_hints`` against the episode toolset (A4).

v3-only coupling (documented per Part 2): composites reuse the executor's
side-effect-free private helpers (``_branch_card``, ``_compute_quick_scores``,
``_decoded_preview``, ``_decoded_words``, ``_decoded_words_with_key``,
``_branch_word_repair_mask``, ``_word_repair_menu_config``,
``_reading_validation``) and operate ONLY on ``executor.workspace`` (the episode
copy inside workers, so isolation is inherited from M2). They never touch v2
loop discipline (``_seen_resegment_proposals``, ``_pending_declare_*``,
gate/panel state).
"""
from __future__ import annotations

import math
import time
from typing import Any

from artifact.schema import ToolCall

# ---------------------------------------------------------------------------
# Tool definitions (defined HERE, not in agent/tools_v2.py)
# ---------------------------------------------------------------------------
HYPOTHESIS_APPLY_READING_TOOL = {
    "name": "hypothesis_apply_reading",
    "description": (
        "Compile an accepted reading of a branch into key edits + word-boundary "
        "changes in ONE step, on a NEW fork (never in place). Give a stored "
        "reading_id, or an inline reading_text / fragments. `window` {start,end} "
        "scopes the whole call to a token range; each fragment can scope itself "
        "with its own start/end. Human-facing `text` may contain prose; use "
        "`repair_text` (letters, spaces, and ? wildcards only) for conservative "
        "machine repair. Low-confidence or ambiguous fragments are skipped. "
        "Auto-detects letter fixes from the reading; "
        "when the letters already match it is boundaries-only. Use dry_run=true "
        "to preview edits/conflicts/holes without creating anything."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "branch": {"type": "string"},
            "reading_id": {"type": "string"},
            "reading_text": {"type": "string"},
            "fragments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": ["integer", "null"]},
                        "end": {"type": ["integer", "null"]},
                        "text": {"type": "string"},
                        "repair_text": {"type": ["string", "null"]},
                        "token_indices": {
                            "type": ["array", "null"],
                            "items": {"type": "integer"},
                        },
                        "confidence": {"type": ["number", "string"]},
                        "label": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
            "window": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
            },
            "as_name": {"type": "string"},
            "dry_run": {"type": "boolean"},
        },
        "required": ["branch"],
    },
}

HYPOTHESIS_TEST_WORD_TOOL = {
    "name": "hypothesis_test_word",
    "description": (
        "Probe a single same-length word hypothesis on a branch: does reading a "
        "given span as `word` improve the decode? Locate the span with "
        "word_index or char_start (its length must equal len(word)). Returns the "
        "word-repair library's edits, adjudication, and accept/hold verdict. "
        "With install=true and a non-empty edit set, forks and applies it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "branch": {"type": "string"},
            "word": {"type": "string"},
            "word_index": {"type": "integer"},
            "char_start": {"type": "integer"},
            "install": {"type": "boolean"},
        },
        "required": ["branch", "word"],
    },
}

BRANCH_ADJUDICATE_TOOL = {
    "name": "branch_adjudicate",
    "description": (
        "Read-only comparison table over 2-8 branches: mapped count, "
        "dict_rate/quad scores, tags, board status, latest reading, and a decode "
        "excerpt per branch, plus deltas against the first-listed branch and a "
        "deterministic ranking. No verdict is imposed — you adjudicate."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "branches": {"type": "array", "items": {"type": "string"}},
            "include_window": {"type": "boolean"},
        },
        "required": ["branches"],
    },
}

COMPOSITE_TOOL_DEFINITIONS = [
    HYPOTHESIS_APPLY_READING_TOOL,
    HYPOTHESIS_TEST_WORD_TOOL,
    BRANCH_ADJUDICATE_TOOL,
]
COMPOSITE_TOOL_NAMES = frozenset(d["name"] for d in COMPOSITE_TOOL_DEFINITIONS)


# ---------------------------------------------------------------------------
# Dispatcher (single choke point for both hosts; A4)
# ---------------------------------------------------------------------------
def execute_composite(
    name: str,
    args: dict[str, Any],
    *,
    executor: Any,
    state_readings: dict[str, dict[str, Any]],
    turn: int,
    tool_use_id: str = "",
) -> dict[str, Any]:
    """Run one composite action and log it, returning the result dict.

    Both hosts serialize the returned dict for their tool_result (both loops
    consume JSON strings). This function itself appends the ToolCall to
    ``executor.call_log`` (serialized result + ``elapsed_ms``, ``iteration`` =
    ``turn``, ``episode_id`` from the executor) and, when an episode toolset is
    active, filters every next-tool hint down to that toolset.
    """
    # Lazy import (binding constraint): agent.tools_v2 is heavy and imports the
    # runner; keep it off this module's import path.
    from agent.tools_v2 import _filter_next_tool_hints, _json

    started = time.time()
    try:
        if name == "hypothesis_apply_reading":
            result_obj = _hypothesis_apply_reading(executor, args, state_readings, turn)
        elif name == "hypothesis_test_word":
            result_obj = _hypothesis_test_word(executor, args, turn)
        elif name == "branch_adjudicate":
            result_obj = _branch_adjudicate(executor, args, state_readings)
        else:
            result_obj = {"error": f"Unknown composite tool: {name}"}
    except Exception as exc:  # noqa: BLE001 - structured error, never crash a host
        result_obj = {"error": f"{type(exc).__name__}: {exc}"}

    episode_toolset = getattr(executor, "episode_toolset", None)
    if episode_toolset is not None:
        result_obj = _filter_next_tool_hints(result_obj, episode_toolset)

    elapsed_ms = int((time.time() - started) * 1000)
    executor.call_log.append(
        ToolCall(
            iteration=turn,
            tool_name=name,
            tool_use_id=tool_use_id,
            arguments=dict(args),
            result=_json(result_obj),
            elapsed_ms=elapsed_ms,
            episode_id=getattr(executor, "episode_id", None),
        )
    )
    return result_obj


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _decoded_char(pt_alpha: Any, key: dict[int, int], token_id: int) -> str:
    """One decoded char for a token in decode coordinates (`?` if unmapped)."""
    if token_id in key:
        return pt_alpha.symbol_for(key[token_id])
    return "?"


def _unsupported_branch_state(branch_obj: Any) -> str | None:
    """Return an ``unsupported``-reason for branch states the composites can't
    safely handle (F6), else ``None``.

    The word-repair menu / projection and the reading alignment reason in
    key-decode coordinates over the BASE-order ciphertext; a transform branch
    (``token_order`` set) reorders tokens, so ``effective_tokens`` spans no
    longer align with the base-order projection. A branch whose decode came from
    ``metadata["decoded_text"]`` (transposition / polyalphabetic overlay) is read
    from that string by ``_decoded_words``, not from the key — the composites'
    symbol edits would not correspond to it. Both are v2 boundary/transform
    territory, out of scope here.
    """
    if branch_obj.token_order is not None:
        return (
            "transform branches (token_order set) are unsupported; composites "
            "reason in base-order key-decode coordinates."
        )
    if branch_obj.metadata.get("decoded_text"):
        from investigation.candidates import null_mask_symbols

        if branch_obj.key and null_mask_symbols(branch_obj):
            return None
        return (
            "branches carrying decoded_text metadata (transposition / "
            "polyalphabetic overlay) are unsupported; the composites edit the "
            "key, not that string."
        )
    return None


MIN_REPAIR_FRAGMENT_CONFIDENCE = 0.65
_SAFE_LEGACY_READING_PUNCTUATION = frozenset(".,;:!\n\r\t")


def _normalize_reading_words(
    text: str,
    pt_alpha: Any,
    *,
    explicit_repair_text: bool = False,
) -> tuple[list[str] | None, str | None]:
    """Normalize one fragment into machine-actionable words.

    Returns ``(words, None)`` on success or ``(None, offending_char)`` if any
    character is unsafe. Explicit ``repair_text`` accepts alphabet symbols,
    whitespace, and ``?`` (a token-consuming wildcard). Legacy human text
    additionally treats conservative sentence punctuation as whitespace;
    editorial notation remains unsafe — and so does ``?``: a prose question
    mark is punctuation, and giving it wildcard (token-consuming) semantics
    would shift the alignment of everything after it, exactly the
    false-global-mapping hazard the M5.1 spec bans. Wildcards exist only in
    explicit ``repair_text``.
    """
    raw = str(text).upper()
    if not explicit_repair_text and ".." in raw:
        return None, "."
    normalized: list[str] = []
    for ch in raw:
        if ch.isspace():
            normalized.append(" ")
        elif pt_alpha.has_symbol(ch):
            normalized.append(ch)
        elif ch == "?" and explicit_repair_text:
            normalized.append(ch)
        elif not explicit_repair_text and ch in _SAFE_LEGACY_READING_PUNCTUATION:
            normalized.append(" ")
        else:
            return None, ch
    words = "".join(normalized).split()
    for word in words:
        for ch in word:
            if ch != "?" and not pt_alpha.has_symbol(ch):
                return None, ch
    return words, None


def _banded_alignment(
    proposed: str, decoded: str, band: int
) -> list[tuple[int | None, int | None]] | None:
    """Banded global alignment (M3 Part 3 / A3).

    Aligns ``proposed`` (chars) against ``decoded`` (chars, `?` = unmapped, which
    matches anything at cost 0). Mismatch cost 1, gap cost 2. Only cells with
    ``|i - j| <= band`` are computed. Traceback is deterministic and emits gaps
    as LATE as possible (A3): walking backward from the end, a gap move on a
    min-cost path is taken before the diagonal, so forced gaps are consumed
    first (= placed latest in forward order); proposed-gaps ("up") are preferred
    over decoded-gaps ("left"). Returns a forward-order op list where each op is
    ``(p_index|None, d_index|None)``: a match/mismatch has both indices; a
    proposed-only gap has ``d_index=None``; a decoded-only gap has
    ``p_index=None``. ``None`` if no DP path exists within the band.
    """
    m, n = len(proposed), len(decoded)
    inf = math.inf
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for i in range(m + 1):
        lo = max(0, i - band)
        hi = min(n, i + band)
        for j in range(lo, hi + 1):
            if i == 0 and j == 0:
                continue
            best = inf
            if i > 0 and j > 0 and dp[i - 1][j - 1] != inf:
                sub = 0 if (
                    proposed[i - 1] == decoded[j - 1]
                    or proposed[i - 1] == "?"
                    or decoded[j - 1] == "?"
                ) else 1
                best = min(best, dp[i - 1][j - 1] + sub)
            if i > 0 and dp[i - 1][j] != inf:
                best = min(best, dp[i - 1][j] + 2)
            if j > 0 and dp[i][j - 1] != inf:
                best = min(best, dp[i][j - 1] + 2)
            dp[i][j] = best
    if dp[m][n] == inf:
        return None

    ops: list[tuple[int | None, int | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        cur = dp[i][j]
        took = False
        # Gaps-late: prefer gap moves over the diagonal on min-cost ties (a gap
        # consumed early in this backward walk lands late in forward order).
        if i > 0 and dp[i - 1][j] != inf and cur == dp[i - 1][j] + 2:
            ops.append((i - 1, None))
            i -= 1
            took = True
        if not took and j > 0 and dp[i][j - 1] != inf and cur == dp[i][j - 1] + 2:
            ops.append((None, j - 1))
            j -= 1
            took = True
        if not took and i > 0 and j > 0 and dp[i - 1][j - 1] != inf:
            sub = 0 if (
                proposed[i - 1] == decoded[j - 1]
                or proposed[i - 1] == "?"
                or decoded[j - 1] == "?"
            ) else 1
            if cur == dp[i - 1][j - 1] + sub:
                ops.append((i - 1, j - 1))
                i -= 1
                j -= 1
                took = True
        if not took:  # defensive: should be unreachable
            break
    ops.reverse()
    return ops


def _word_boundaries(words: list[str]) -> list[int]:
    """Internal char-position boundaries between consecutive words (cum lengths)."""
    boundaries: list[int] = []
    pos = 0
    for word in words[:-1]:
        pos += len(word)
        boundaries.append(pos)
    return boundaries


# ---------------------------------------------------------------------------
# hypothesis_apply_reading (Part 3; absorbs plan Phase 6)
# ---------------------------------------------------------------------------
def _err(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "error", "kind": "reading_application", "error": reason, **extra}


def _hypothesis_apply_reading(
    executor: Any, args: dict[str, Any], state_readings: dict[str, dict[str, Any]], turn: int
) -> dict[str, Any]:
    from investigation.reading import Reading, ReadingFragment, new_reading_id

    ws = executor.workspace
    branch = str(args.get("branch") or "")
    if not branch or not ws.has_branch(branch):
        return _err(f"unknown branch: {branch!r}")

    pt_alpha = ws.plaintext_alphabet
    if pt_alpha._multisym:
        return {
            "status": "unsupported",
            "kind": "reading_application",
            "error": (
                "hypothesis_apply_reading supports single-character plaintext "
                "alphabets only."
            ),
        }
    cipher_alpha = ws.cipher_text.alphabet

    # --- resolve the reading source (exactly one) ---
    reading_id = args.get("reading_id")
    reading_text = args.get("reading_text")
    inline_fragments = args.get("fragments")
    has_inline = reading_text is not None or inline_fragments is not None
    if reading_id is not None and has_inline:
        return _err("provide exactly one of reading_id or inline reading_text/fragments")
    if reading_id is None and not has_inline:
        return _err("one of reading_id, reading_text, or fragments is required")

    if reading_id is not None:
        stored = state_readings.get(str(reading_id))
        if stored is None:
            return _err(f"unknown reading_id: {reading_id!r}")
        reading = Reading.from_dict(stored)
        fragments = list(reading.fragments)
        id6 = str(reading_id)[:6]
    else:
        fragments = []
        if inline_fragments is not None:
            for item in inline_fragments:
                if not isinstance(item, dict):
                    continue
                fragment = ReadingFragment.from_dict(item)
                # Lead-authored inline fragments are deliberate commands:
                # omitted confidence means full confidence (1.0). Worker
                # fragments cannot reach this branch without confidence —
                # the reading episode schema REQUIRES it (M5.1 review fix);
                # only LEGACY stored Readings get the 1.0 bump in
                # Reading.from_episode_result.
                if item.get("confidence") is None:
                    fragment.confidence = 1.0
                fragments.append(fragment)
        if reading_text is not None:
            fragments.append(ReadingFragment(text=str(reading_text), confidence=1.0))
        # A5d: an inline reading gets an ephemeral id for fork naming only.
        id6 = new_reading_id()[:6]

    if not fragments:
        return _err("reading has no fragments to apply")

    eff_tokens = ws.effective_tokens(branch)
    total = len(eff_tokens)
    branch_obj = ws.get_branch(branch)
    unsupported = _unsupported_branch_state(branch_obj)
    if unsupported is not None:
        return {"status": "unsupported", "kind": "reading_application",
                "error": unsupported}
    key = branch_obj.key

    # --- window scoping ---
    window = args.get("window")
    if window is not None:
        try:
            win_start = int(window.get("start"))
            win_end = int(window.get("end"))
        except (TypeError, ValueError):
            return _err("window must be {start, end} integer token indices")
        if not (0 <= win_start < win_end <= total):
            return _err(
                "window out of bounds", window=[win_start, win_end], total=total
            )
    else:
        win_start, win_end = 0, total

    # --- per-fragment alignment ---
    edit_votes: dict[str, dict[str, int]] = {}
    holes: list[str] = []
    boundary_token_indices: set[int] = set()
    fragment_edges: set[int] = set()
    fragment_spans: list[tuple[int, int]] = []
    total_gaps = 0
    total_mismatches = 0
    fragment_reports: list[dict[str, Any]] = []
    skipped_fragments: list[dict[str, Any]] = []

    for idx, frag in enumerate(fragments):
        if frag.confidence < MIN_REPAIR_FRAGMENT_CONFIDENCE:
            skipped_fragments.append({
                "fragment_index": idx,
                "reason": "confidence_below_threshold",
                "confidence": frag.confidence,
                "minimum_confidence": MIN_REPAIR_FRAGMENT_CONFIDENCE,
            })
            continue
        if frag.token_indices is not None:
            fragment_token_indices = [int(value) for value in frag.token_indices]
            if not fragment_token_indices:
                skipped_fragments.append({
                    "fragment_index": idx,
                    "reason": "empty_token_provenance",
                })
                continue
            if fragment_token_indices != sorted(set(fragment_token_indices)):
                return _err(
                    "fragment token provenance must be unique and ordered",
                    fragment_index=idx,
                )
            f_start = fragment_token_indices[0]
            f_end = fragment_token_indices[-1] + 1
        else:
            f_start = frag.start if frag.start is not None else win_start
            f_end = frag.end if frag.end is not None else win_end
            fragment_token_indices = list(range(f_start, f_end))
        if any(index < win_start or index >= win_end for index in fragment_token_indices):
            return _err(
                "fragment extends outside the window",
                fragment_index=idx,
                fragment_bounds=[f_start, f_end],
                window=[win_start, win_end],
            )
        if not (0 <= f_start < f_end <= total):
            return _err(
                "fragment span out of bounds",
                fragment_index=idx,
                fragment_bounds=[f_start, f_end],
                total=total,
            )
        source_text = frag.repair_text if frag.repair_text is not None else frag.text
        words, bad = _normalize_reading_words(
            source_text,
            pt_alpha,
            explicit_repair_text=frag.repair_text is not None,
        )
        if words is None:
            skipped_fragments.append({
                "fragment_index": idx,
                "reason": "unsafe_repair_text",
                "character": bad,
                "used_explicit_repair_text": frag.repair_text is not None,
            })
            continue
        proposed = "".join(words)
        decoded = "".join(
            _decoded_char(pt_alpha, key, eff_tokens[t])
            for t in fragment_token_indices
        )
        span_len = len(fragment_token_indices)
        delta = len(proposed) - span_len
        tolerance = max(2, math.ceil(0.02 * span_len))

        if delta == 0:
            ops = [(t, t) for t in range(span_len)]
            mode = "direct"
        elif abs(delta) <= tolerance:
            band = max(3, abs(delta) + 1)
            ops = _banded_alignment(proposed, decoded, band)
            mode = "banded"
            if ops is None:
                skipped_fragments.append({
                    "fragment_index": idx,
                    "reason": "no_banded_alignment_path",
                    "proposed_len": len(proposed),
                    "span_len": span_len,
                })
                continue
        else:
            first_div = _first_divergence(proposed, decoded)
            skipped_fragments.append({
                "fragment_index": idx,
                "reason": "count_mismatch_too_large",
                "proposed_len": len(proposed),
                "span_len": span_len,
                "delta": delta,
                "tolerance": tolerance,
                "first_divergence": first_div,
            })
            continue

        # p_to_token[i] = token index where proposed char i starts.
        p_to_token: dict[int, int] = {}
        d_consumed = 0
        frag_gaps = 0
        frag_mismatches = 0
        for pi, dj in ops:
            if pi is not None and dj is not None:
                token_index = fragment_token_indices[dj]
                p_to_token[pi] = token_index
                d_consumed = dj + 1
                proposed_char = proposed[pi]
                decoded_char = decoded[dj]
                if proposed_char != "?" and proposed_char != decoded_char:
                    frag_mismatches += 1
                    symbol = cipher_alpha.symbol_for(eff_tokens[token_index])
                    edit_votes.setdefault(symbol, {})
                    edit_votes[symbol][proposed_char] = (
                        edit_votes[symbol].get(proposed_char, 0) + 1
                    )
            elif pi is not None and dj is None:
                # proposed char with no decoded token -> hole
                insertion_position = (
                    fragment_token_indices[d_consumed]
                    if d_consumed < len(fragment_token_indices)
                    else f_end
                )
                p_to_token[pi] = insertion_position
                frag_gaps += 1
                holes.append(
                    f"frag{idx}: inserted '{proposed[pi]}' at token "
                    f"{insertion_position}"
                )
            else:
                # decoded token with no proposed char -> hole
                d_consumed = dj + 1
                frag_gaps += 1
                holes.append(
                    f"frag{idx}: unread token {fragment_token_indices[dj]} (decoded "
                    f"'{decoded[dj]}')"
                )

        total_gaps += frag_gaps
        total_mismatches += frag_mismatches

        # boundaries -> token indices
        for b in _word_boundaries(words):
            token_index = p_to_token.get(b)
            if token_index is not None:
                boundary_token_indices.add(token_index)
        if f_start != win_start:
            fragment_edges.add(f_start)
        if f_end != win_end:
            fragment_edges.add(f_end)
        fragment_spans.append((f_start, f_end))

        fragment_reports.append({
            "fragment_index": idx,
            "mode": mode,
            "span": [f_start, f_end],
            "token_indices": (
                list(fragment_token_indices)
                if frag.token_indices is not None
                else None
            ),
            "proposed_len": len(proposed),
            "gaps": frag_gaps,
            "mismatches": frag_mismatches,
            "used_explicit_repair_text": frag.repair_text is not None,
        })

    if not fragment_reports:
        return {
            "status": "ok",
            "kind": "reading_application",
            "branch": branch,
            "fork": None,
            "dry_run": bool(args.get("dry_run", False)),
            "no_actionable_fragments": True,
            "actionable_fragment_count": 0,
            "skipped_fragments": skipped_fragments,
            "edits": [],
            "conflicts": [],
            "holes": [],
        }

    # --- resolve edit votes (majority; tie -> drop) ---
    edits: list[tuple[str, str]] = []
    conflicts: list[dict[str, Any]] = []
    for symbol in sorted(edit_votes):
        counter = edit_votes[symbol]
        ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            conflicts.append({
                "symbol": symbol,
                "reason": "tie",
                "candidates": dict(counter),
            })
            continue
        winner = ranked[0][0]
        if len(ranked) > 1:
            conflicts.append({
                "symbol": symbol,
                "reason": "majority",
                "chosen": winner,
                "candidates": dict(counter),
            })
        edits.append((symbol, winner))

    edit_labels = sorted(f"{symbol}={letter}" for symbol, letter in edits)
    character_preserving = not edits

    # --- boundary compilation (token-index cut points) ---
    # F1: a fragment REPLACES only the boundaries strictly inside its own span.
    # Old cuts outside the window (byte-identical region) AND old cuts inside the
    # window but outside every fragment span (untouched words) are KEPT — only
    # cuts strictly inside a fragment span are dropped and re-derived from the
    # reading, so a narrow fragment never silently deletes a neighbour's
    # boundary.
    def _inside_a_fragment(cut: int) -> bool:
        return any(fs < cut < fe for (fs, fe) in fragment_spans)

    current_spans = ws.effective_word_spans(branch)
    old_cuts = {s for (s, _e) in current_spans if s != 0}
    new_cuts: set[int] = set()
    for c in old_cuts:
        if c <= win_start or c >= win_end:
            new_cuts.add(c)               # outside the window -> byte-identical
        elif not _inside_a_fragment(c):
            new_cuts.add(c)               # untouched word inside window -> keep
        # else: inside a fragment span -> replaced by the fragment's boundaries
    if 0 < win_start < total:
        new_cuts.add(win_start)
    if 0 < win_end < total:
        new_cuts.add(win_end)
    for c in fragment_edges:
        if 0 < c < total and win_start <= c <= win_end:
            new_cuts.add(c)
    for c in boundary_token_indices:
        if win_start < c < win_end and 0 < c < total:
            new_cuts.add(c)
    new_spans: list[tuple[int, int]] = []
    prev = 0
    for c in sorted(new_cuts) + [total]:
        if c > prev:
            new_spans.append((prev, c))
            prev = c
    boundary_change_count = len(
        {s for (s, _e) in current_spans} ^ {s for (s, _e) in new_spans}
    )

    # --- before scores + diff preview inputs ---
    before_scores = executor._compute_quick_scores(branch)
    before_words = executor._decoded_words(branch)
    before_stream = "".join(before_words)

    dry_run = bool(args.get("dry_run", False))

    # --- create the fork and apply ---
    as_name = args.get("as_name")
    target_name = str(as_name) if as_name else f"reading_{id6}_{branch}"
    fork_name = target_name
    suffix = 2
    while ws.has_branch(fork_name):
        fork_name = f"{target_name}_{suffix}"
        suffix += 1

    ws.fork(fork_name, from_branch=branch)
    # F7: on dry_run the fork is a scratch copy that MUST be deleted even if
    # scoring / diffing raises, so a dry-run probe never leaks a branch.
    try:
        for symbol, letter in edits:
            ws.set_mapping(fork_name, cipher_alpha.id_for(symbol), pt_alpha.id_for(letter))
        ws.set_word_spans(fork_name, new_spans)

        after_scores = executor._compute_quick_scores(fork_name)
        after_words = executor._decoded_words(fork_name)
        after_stream = "".join(after_words)

        from analysis.word_hypothesis_repair import changed_excerpt
        diff_preview = changed_excerpt(before_stream, after_stream)

        # Part 6: a composite MAY append a repair-agenda follow-up. Record
        # residual work (unresolved holes / dropped conflicts) against the fork
        # so it rides out of an episode as agenda_additions (merged on install)
        # and shows on the lead's branch card. Skipped on dry_run (nothing kept).
        if not dry_run and (holes or conflicts) and hasattr(executor, "repair_agenda"):
            executor.repair_agenda.append({
                "id": getattr(executor, "_next_repair_agenda_id", 1),
                "branch": fork_name,
                "kind": "reading_residual",
                "reading_source": id6,
                "holes": len(holes),
                "conflicts": len(conflicts),
                "status": "open",
            })
            if hasattr(executor, "_next_repair_agenda_id"):
                executor._next_repair_agenda_id += 1
    finally:
        if dry_run and ws.has_branch(fork_name):
            ws.delete(fork_name)
    if dry_run:
        fork_name = None

    return {
        "status": "ok",
        "kind": "reading_application",
        "branch": branch,
        "fork": fork_name,
        "dry_run": dry_run,
        "character_preserving": character_preserving,
        "edits": edit_labels,
        "conflicts": conflicts,
        "holes": holes,
        "boundary_change_count": boundary_change_count,
        "alignment": {"gaps": total_gaps, "mismatches": total_mismatches},
        "fragments": fragment_reports,
        "actionable_fragment_count": len(fragment_reports),
        "skipped_fragments": skipped_fragments,
        "no_actionable_fragments": False,
        "scores_before": before_scores,
        "scores_after": after_scores,
        "diff_preview": diff_preview,
    }


def _first_divergence(proposed: str, decoded: str) -> dict[str, Any]:
    limit = min(len(proposed), len(decoded))
    pos = 0
    while pos < limit and (proposed[pos] == decoded[pos] or decoded[pos] == "?"):
        pos += 1
    radius = 20
    return {
        "offset": pos,
        "proposed": proposed[max(0, pos - radius):pos + radius],
        "decoded": decoded[max(0, pos - radius):pos + radius],
    }


# ---------------------------------------------------------------------------
# hypothesis_test_word (Part 4)
# ---------------------------------------------------------------------------
def _hypothesis_test_word(executor: Any, args: dict[str, Any], turn: int) -> dict[str, Any]:
    # Lazy imports (binding constraint): the runner + the word-repair library
    # pull heavy dependencies; keep them off this module's import path.
    from analysis import dictionary as dictionary_module
    from analysis.word_hypothesis_repair import (
        changed_excerpt,
        score_injected_word_hypothesis,
    )
    from automated import runner as automated_runner

    ws = executor.workspace
    branch = str(args.get("branch") or "")
    if not branch or not ws.has_branch(branch):
        return {"status": "error", "kind": "word_hypothesis", "error": f"unknown branch: {branch!r}"}
    word = str(args.get("word") or "").upper()
    if not word:
        return {"status": "error", "kind": "word_hypothesis", "error": "word is required"}

    branch_obj = ws.get_branch(branch)
    base_key = dict(branch_obj.key)
    if not base_key:
        return {
            "status": "skipped",
            "kind": "word_hypothesis",
            "branch": branch,
            "reason": "empty_base_key",
        }
    pt_alpha = ws.plaintext_alphabet
    if pt_alpha._multisym:
        return {
            "status": "unsupported",
            "kind": "word_hypothesis",
            "error": "hypothesis_test_word supports single-character plaintext alphabets only.",
        }
    unsupported = _unsupported_branch_state(branch_obj)
    if unsupported is not None:
        return {"status": "unsupported", "kind": "word_hypothesis",
                "error": unsupported}

    eff_tokens = ws.effective_tokens(branch)
    total = len(eff_tokens)
    spans = ws.effective_word_spans(branch)

    # --- locate the span in decode coordinates ---
    word_index = args.get("word_index")
    char_start = args.get("char_start")
    if word_index is not None:
        wi = int(word_index)
        if wi < 0 or wi >= len(spans):
            return {"status": "error", "kind": "word_hypothesis",
                    "error": f"word_index {wi} out of range (0..{len(spans) - 1})"}
        char_start, char_end = spans[wi]
    elif char_start is not None:
        char_start = int(char_start)
        char_end = char_start + len(word)
    else:
        return {"status": "error", "kind": "word_hypothesis",
                "error": "one of word_index or char_start is required"}
    if not (0 <= char_start < char_end <= total):
        return {"status": "error", "kind": "word_hypothesis",
                "error": "span out of bounds", "span": [char_start, char_end], "total": total}
    span_len = char_end - char_start
    if span_len != len(word):
        return {
            "status": "error",
            "kind": "word_hypothesis",
            "error": "span length does not equal len(word)",
            "span_length": span_len,
            "word_length": len(word),
            "span": [char_start, char_end],
        }

    mask = executor._branch_word_repair_mask(branch_obj)
    masked = set(mask)

    # --- map decode-coordinate span into projection coordinates (A2) ---
    offending: list[int] = []
    for t in range(char_start, char_end):
        token_id = eff_tokens[t]
        value = base_key.get(token_id)
        symbol = ws.cipher_text.alphabet.symbol_for(token_id)
        if value is None or value < 0 or value > 25 or symbol in masked:
            offending.append(t)
    if offending:
        return {
            "status": "error",
            "kind": "word_hypothesis",
            "error": "span contains unmapped or masked tokens",
            "offending_token_positions": offending,
        }
    proj_start = 0
    for t in range(0, char_start):
        token_id = eff_tokens[t]
        value = base_key.get(token_id)
        symbol = ws.cipher_text.alphabet.symbol_for(token_id)
        if value is not None and 0 <= value <= 25 and symbol not in masked:
            proj_start += 1
    proj_end = proj_start + len(word)

    dictionary_path = dictionary_module.get_dictionary_path(executor.language)
    if not dictionary_path:
        return {
            "status": "skipped",
            "kind": "word_hypothesis",
            "branch": branch,
            "reason": "no_dictionary_for_language",
            "language": executor.language,
        }
    config = executor._word_repair_menu_config(args)
    # F2: score with the SAME model variant the menu tool uses (main's
    # search_word_repair_menu passes variant=self._model_variant; DTA is the
    # German default), so the menu-backed and injected paths cannot drift.
    resolved_model = automated_runner.zenith_native_model_path(
        executor.language, variant=getattr(executor, "_model_variant", None)
    )

    # --- build the menu exactly as _tool_search_word_repair_menu does ---
    try:
        menu = automated_runner.build_word_repair_menu(
            cipher_text=ws.cipher_text,
            base_key=base_key,
            mask=mask,
            language=executor.language,
            config=config,
            dictionary_path=dictionary_path,
            model_path=resolved_model,
            source_branch=branch,
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "kind": "word_hypothesis",
                "branch": branch, "error": f"{type(exc).__name__}: {exc}"}

    menu_packet = None
    for packet in menu.packets:
        prov = packet.provenance or {}
        for hyp in prov.get("word_hypotheses") or []:
            if (
                hyp.get("start") == proj_start
                and hyp.get("end") == proj_end
                and hyp.get("target") == word
            ):
                menu_packet = packet
                break
        if menu_packet is not None:
            break

    span = [char_start, char_end]
    if menu_packet is not None:
        return _word_hypothesis_result(
            executor,
            branch=branch,
            span=span,
            word=word,
            menu_backed=True,
            in_dictionary=True,
            verdict=_packet_verdict(menu_packet),
            packet=menu_packet,
            edits=list((menu_packet.provenance or {}).get("edits") or []),
            install=bool(args.get("install", False)),
            mask=mask,
            base_key=base_key,
            changed_excerpt=changed_excerpt,
            automated_runner=automated_runner,
            turn=turn,
        )

    # --- injected path (word the menu did not propose) ---
    pages, alphabet = automated_runner._single_page_group(ws.cipher_text)
    injected = score_injected_word_hypothesis(
        pages=pages,
        shared_key=base_key,
        dictionary_path=dictionary_path,
        start=proj_start,
        end=proj_end,
        target=word,
        language=executor.language,
        config=config,
        mask=mask,
        alphabet=alphabet,
        source_branch=branch,
        model_path=resolved_model,
    )
    if injected.get("verdict") == "no_valid_edits":
        return {
            "status": "ok",
            "kind": "word_hypothesis",
            "branch": branch,
            "span": span,
            "word": word,
            "menu_backed": False,
            "in_dictionary": bool(injected.get("in_dictionary")),
            "verdict": "no_valid_edits",
            "observed": injected.get("observed"),
            "edits": [],
            "installed_fork": None,
            "note": (
                "No valid edit set: the span already reads as the word, or the "
                "fix would touch a masked/stable symbol or conflict."
            ),
        }
    return _word_hypothesis_result(
        executor,
        branch=branch,
        span=span,
        word=word,
        menu_backed=False,
        in_dictionary=bool(injected.get("in_dictionary")),
        verdict=injected.get("verdict"),
        packet=injected.get("packet"),
        edits=list(injected.get("edits") or []),
        install=bool(args.get("install", False)),
        mask=mask,
        base_key=base_key,
        changed_excerpt=changed_excerpt,
        automated_runner=automated_runner,
        turn=turn,
    )


def _packet_verdict(packet: Any) -> str:
    validation = packet.validation or {}
    if bool(validation.get("accepted")):
        return "accept"
    return str(validation.get("decision") or "hold_for_review")


def _word_hypothesis_result(
    executor: Any,
    *,
    branch: str,
    span: list[int],
    word: str,
    menu_backed: bool,
    in_dictionary: bool,
    verdict: str | None,
    packet: Any,
    edits: list[str],
    install: bool,
    mask: tuple[str, ...],
    base_key: dict[int, int],
    changed_excerpt: Any,
    automated_runner: Any,
    turn: int,
) -> dict[str, Any]:
    ws = executor.workspace
    solver_scores = dict(packet.solver_scores or {}) if packet is not None else {}
    validation = packet.validation or {} if packet is not None else {}
    provenance = packet.provenance or {} if packet is not None else {}
    collateral = provenance.get("collateral_evidence") or {}

    # after-decode for the changed-excerpt preview (does not require install).
    new_key, applied, reason = automated_runner.apply_word_repair_edits(
        base_key=base_key, edits=edits, alphabet=ws.cipher_text.alphabet, mask=mask
    )
    before_words = executor._decoded_words(branch)
    if new_key is not None:
        after_words = executor._decoded_words_with_key(branch, new_key)
        changed = changed_excerpt("".join(before_words), "".join(after_words))
    else:
        changed = {"changed": False, "before": "", "after": "", "offset": None}

    installed_fork: str | None = None
    install_note: str | None = None
    if install and verdict != "no_valid_edits":
        if new_key is None:
            install_note = (
                f"whole-candidate rejected ({reason}); edits={edits} not applied"
            )
        else:
            target = f"wordtest_{turn}_{branch}"
            name = target
            n = 2
            while ws.has_branch(name):
                name = f"{target}_{n}"
                n += 1
            ws.fork(name, from_branch=branch)
            ws.set_full_key(name, new_key)
            installed_fork = name

    return {
        "status": "ok",
        "kind": "word_hypothesis",
        "branch": branch,
        "span": span,
        "word": word,
        "provenance": {
            "branch": branch,
            "span": span,
            "word": word,
            "menu_backed": menu_backed,
        },
        "menu_backed": menu_backed,
        "in_dictionary": in_dictionary,
        "verdict": verdict,
        "edits": list(edits),
        "solver_scores": {
            "adjudication_score": solver_scores.get("adjudication_score"),
            "page_validation_avg": solver_scores.get("page_validation_avg"),
            "page_robust_score": solver_scores.get("page_robust_score"),
        },
        "solver_score_deltas": (validation.get("deltas") or {}) if isinstance(validation, dict) else {},
        "adjudication_summary": {
            "collateral_occurrences": collateral.get("collateral_occurrences"),
            "improved_occurrences": collateral.get("improved_occurrences"),
            "damaged_occurrences": collateral.get("damaged_occurrences"),
            "adjudication_score": solver_scores.get("adjudication_score"),
        },
        "acceptance": {
            "accepted": bool(validation.get("accepted")) if isinstance(validation, dict) else False,
            "decision": validation.get("decision") if isinstance(validation, dict) else None,
            "reasons": list((validation.get("reasons") or [])[:3]) if isinstance(validation, dict) else [],
        },
        "changed_excerpt": changed,
        "installed_fork": installed_fork,
        "install_note": install_note,
    }


# ---------------------------------------------------------------------------
# branch_adjudicate (Part 5)
# ---------------------------------------------------------------------------
def _branch_adjudicate(
    executor: Any, args: dict[str, Any], state_readings: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ws = executor.workspace
    branches = list(args.get("branches") or [])
    if not (2 <= len(branches) <= 8):
        return {
            "status": "error",
            "kind": "branch_adjudication",
            "error": "branch_adjudicate compares 2-8 branches",
            "branch_count": len(branches),
        }
    unknown = [b for b in branches if not ws.has_branch(str(b))]
    if unknown:
        return {
            "status": "error",
            "kind": "branch_adjudication",
            "error": "unknown branch(es)",
            "unknown": unknown,
        }
    include_window = bool(args.get("include_window", False))
    in_episode = getattr(executor, "episode_id", None) is not None

    # latest stored reading per branch (id + overall_confidence).
    latest_reading: dict[str, dict[str, Any]] = {}
    for rid, rdict in (state_readings or {}).items():
        rbranch = str(rdict.get("branch") or "")
        if rbranch not in branches:
            continue
        prev = latest_reading.get(rbranch)
        this_turn = int(rdict.get("created_turn") or 0)
        if prev is None or this_turn >= prev["_turn"]:
            latest_reading[rbranch] = {
                "reading_id": str(rdict.get("reading_id") or rid),
                "overall_confidence": rdict.get("overall_confidence"),
                "_turn": this_turn,
            }

    rows: list[dict[str, Any]] = []
    for name in branches:
        name = str(name)
        branch_obj = ws.get_branch(name)
        scores = executor._compute_quick_scores(name)
        excerpt = executor._decoded_preview(name, max_words=30)[:160]
        row: dict[str, Any] = {
            "branch": name,
            "mapped_count": len(branch_obj.key),
            "dict_rate": scores.get("dict_rate"),
            "quad": scores.get("quad"),
            "tags": list(branch_obj.tags),
            "decoded_excerpt": excerpt,
        }
        if not in_episode:
            card = executor.hypothesis_board.get(name)
            if card is not None:
                row["board_card"] = {
                    "cipher_mode": card.get("cipher_mode"),
                    "mode_status": card.get("mode_status"),
                }
        reading = latest_reading.get(name)
        if reading is not None:
            row["reading"] = {
                "reading_id": reading["reading_id"],
                "overall_confidence": reading["overall_confidence"],
            }
        if include_window:
            row["window_text"] = executor._decoded_preview(name, max_words=200)
        rows.append(row)

    # deltas vs the first-listed branch.
    def _num(value: Any) -> float | None:
        return value if isinstance(value, (int, float)) else None

    base = rows[0]
    deltas: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        d_delta = q_delta = None
        if _num(row["dict_rate"]) is not None and _num(base["dict_rate"]) is not None:
            d_delta = round(row["dict_rate"] - base["dict_rate"], 4)
        if _num(row["quad"]) is not None and _num(base["quad"]) is not None:
            q_delta = round(row["quad"] - base["quad"], 4)
        deltas[row["branch"]] = {"dict_rate_delta": d_delta, "quad_delta": q_delta}

    def _rank_key(row: dict[str, Any]) -> tuple[float, float]:
        dr = row["dict_rate"]
        qd = row["quad"]
        return (
            dr if isinstance(dr, (int, float)) else float("-inf"),
            qd if isinstance(qd, (int, float)) else float("-inf"),
        )

    ranking = [r["branch"] for r in sorted(rows, key=_rank_key, reverse=True)]

    return {
        "status": "ok",
        "kind": "branch_adjudication",
        "baseline_branch": base["branch"],
        "rows": rows,
        "deltas": deltas,
        "ranking": ranking,
    }
