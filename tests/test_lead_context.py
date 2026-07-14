"""Lead context builder tests (M1 Part 2/5): determinism, budgets, rotation."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor
from investigation.context import (
    CHARS_PER_TOKEN,
    build_lead_context,
    build_v3_system_prompt,
    rotating_window_bounds,
)
from investigation.state import InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


def _state(raw: str = "ABCDEFG HIJKLMN OPQRSTU", separator: str | None = " ") -> InvestigationState:
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=separator)
    return InvestigationState(workspace=Workspace(ct), language="en")


def _executor(state: InvestigationState) -> WorkspaceToolExecutor:
    return WorkspaceToolExecutor(
        workspace=state.workspace,
        language=state.language,
        word_set={"THE"},
        word_list=["THE"],
        pattern_dict={},
        declaration_policy=NoGatesPolicy(),
    )


def _texts(messages):
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    out.append(b["text"])
    return out


def _blocks(m):
    content = m.get("content")
    return content if isinstance(content, list) else []


def test_context_is_deterministic():
    state = _state()
    ex = _executor(state)
    a = build_lead_context(state, ex, turn=3, token_budget=8000)
    b = build_lead_context(state, ex, turn=3, token_budget=8000)
    assert a == b


def test_context_sections_present():
    state = _state()
    state.add_evidence("diagnostic_preflight", 0, "seeded evidence entry")
    ex = _executor(state)
    joined = "\n".join(_texts(build_lead_context(state, ex, turn=1, token_budget=8000)))
    assert "## Task" in joined
    assert "## Cipher" in joined
    assert "## Diagnostic fingerprint" in joined
    assert "## Branch cards" in joined
    assert "## Hypothesis board" in joined
    assert "## Recent evidence" in joined
    assert "seeded evidence entry" in joined
    assert "## Full-decode window" in joined


def test_context_respects_token_budget():
    state = _state("S001 S002 S003 S004 S005 S006 S007 S008 S009 S010")
    ex = _executor(state)
    for budget in (500, 1500, 4000):
        msgs = build_lead_context(state, ex, turn=1, token_budget=budget)
        total = sum(len(t) for t in _texts(msgs))
        assert total <= budget * CHARS_PER_TOKEN


def test_recent_exchanges_rendered_verbatim():
    state = _state()
    # Two exchanges so the first is emitted fully verbatim and the second's
    # user turn carries the appended dynamic view.
    a0 = {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t0", "name": "decode_show", "input": {"branch": "main"}}]}
    r0 = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t0", "content": "{}"}]}
    a1 = {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "decode_show", "input": {"branch": "main"}},
        {"type": "provider_extra", "provider": "openai", "kind": "reasoning",
         "items": [{"encrypted_content": "xyz"}]},
    ]}
    r1 = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}
    state.record_exchange(a0, r0)
    state.record_exchange(a1, r1)
    msgs = build_lead_context(state, _executor(state), turn=3, token_budget=8000)
    # First exchange appears verbatim (untransformed, incl. provider_extra).
    assert a0 in msgs
    assert r0 in msgs
    assert a1 in msgs
    # The last user turn carries the tool_result verbatim plus the view block.
    last = msgs[-1]
    assert last["role"] == "user"
    assert last["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "{}"}
    assert last["content"][-1]["type"] == "text"
    assert "## Full-decode window" in last["content"][-1]["text"]


def test_context_roles_strictly_alternate():
    """No two consecutive messages share a role (valid for the Anthropic path)."""
    state = _state()
    a = {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "decode_show", "input": {}}]}
    r = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}
    ex = _executor(state)
    # Turn 1: no exchanges — a single user message.
    m1 = build_lead_context(state, ex, turn=1, token_budget=8000)
    assert [x["role"] for x in m1] == ["user"]
    # Turn 2+: prefix user, then alternating exchange, ending on user.
    state.record_exchange(a, r)
    m2 = build_lead_context(state, ex, turn=2, token_budget=8000)
    roles = [x["role"] for x in m2]
    assert roles[0] == "user" and roles[-1] == "user"
    for prev, nxt in zip(roles, roles[1:]):
        assert prev != nxt, f"consecutive same-role turns: {roles}"


def test_rotating_window_bounds_cover_full_cipher():
    total = 1000
    window = 400
    seen = set()
    n_turns = -(-total // window)  # ceil
    for turn in range(1, n_turns + 3):
        offset, end = rotating_window_bounds(total, turn, window)
        assert 0 <= offset <= total
        assert offset <= end <= total
        seen.update(range(offset, end))
    # Over enough turns the window sweeps the whole cipher.
    assert seen == set(range(total))


def test_rotating_window_snaps_to_word_spans():
    spans = [(0, 3), (3, 7), (7, 12)]
    # raw offset lands inside the second span → snaps down to its start (3).
    offset, _end = rotating_window_bounds(12, turn=1, window_tokens=5, spans=spans)
    assert offset in {s for s, _ in spans}


def test_window_offset_advances_with_turn():
    state = _state("S001 S002 S003 S004 S005 S006 S007 S008 S009 S010")
    ex = _executor(state)
    # Window (3 tokens) smaller than the 10-token cipher, so the offset rotates.
    t1 = "\n".join(_texts(
        build_lead_context(state, ex, turn=1, token_budget=8000, window_tokens=3)))
    t2 = "\n".join(_texts(
        build_lead_context(state, ex, turn=2, token_budget=8000, window_tokens=3)))

    def window_header(text: str) -> str:
        for line in text.splitlines():
            if line.startswith("## Full-decode window"):
                return line
        return ""

    h1, h2 = window_header(t1), window_header(t2)
    assert h1 and h2
    # The window offset advances between turns (rotation over the cipher).
    assert h1 != h2
    assert t1 != t2


def test_v3_system_prompt_is_short_and_language_aware():
    en = build_v3_system_prompt("en")
    de = build_v3_system_prompt("de")
    # Much shorter than the v2 full system prompt (~32k chars).
    assert len(en) < 11000
    assert "meta_declare_solution" in en
    # German notes are woven in.
    assert "German" in de or "BRUDER" in de


def test_view_header_shows_turn_of_max():
    """R4: the view header shows 'turn N of M' so the model knows its runway."""
    state = _state()
    ex = _executor(state)
    joined = "\n".join(_texts(
        build_lead_context(state, ex, turn=3, token_budget=8000, max_turns=20)))
    assert "turn 3 of 20" in joined
    # Without max_turns, only the bare turn number is shown.
    bare = "\n".join(_texts(build_lead_context(state, ex, turn=3, token_budget=8000)))
    header = next(
        l for l in bare.splitlines() if l.startswith("## Investigation state"))
    assert header.strip() == "## Investigation state (turn 3)"


def test_external_context_is_stable_prefix_section():
    """R3: external context is its own stable prefix section, every turn."""
    state = _state()
    state.external_context = "Related manuscript: an 18th-century lodge register."
    ex = _executor(state)
    j1 = "\n".join(_texts(build_lead_context(state, ex, turn=1, token_budget=8000)))
    j6 = "\n".join(_texts(build_lead_context(state, ex, turn=6, token_budget=8000)))
    # Present every turn (it does not scroll out like an evidence entry would).
    assert "## External context" in j1
    assert "an 18th-century lodge register" in j1
    assert "## External context" in j6
    # It lives in the first (stable-prefix) user message, not the dynamic view.
    m1 = build_lead_context(state, ex, turn=1, token_budget=8000)
    prefix_text = m1[0]["content"][0]["text"]
    assert "## External context" in prefix_text
    # Omitted entirely when there is no external context.
    state.external_context = ""
    assert "## External context" not in "\n".join(
        _texts(build_lead_context(state, ex, turn=1, token_budget=8000)))


def test_decode_window_inserts_word_separators():
    """R6: a word-structured cipher renders as words, not character soup."""
    state = _state("ABC DEF GHI", separator=" ")
    ws = state.workspace
    ct = ws.cipher_text
    pt = ws.plaintext_alphabet
    for sym in "ABCDEFGHI":
        ws.set_mapping("main", ct.alphabet.id_for(sym), pt.id_for(sym))
    ex = _executor(state)
    joined = "\n".join(_texts(build_lead_context(
        state, ex, turn=1, token_budget=8000, window_tokens=100)))
    window_text = joined[joined.index("## Full-decode window"):]
    assert "ABC DEF GHI" in window_text     # word separators inserted
    assert "ABCDEFGHI" not in window_text    # not run together


def test_context_budget_drops_oldest_whole_exchange():
    """R2/F1: over budget, drop the OLDEST WHOLE exchange (never split one)."""
    state = _state()
    ex = _executor(state)
    big_old = "0" * 6000
    big_new = "1" * 6000
    a0 = {"role": "assistant", "content": [
        {"type": "tool_use", "id": "old", "name": "decode_show", "input": {}}]}
    r0 = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "old", "content": big_old}]}
    a1 = {"role": "assistant", "content": [
        {"type": "tool_use", "id": "new", "name": "decode_show", "input": {}}]}
    r1 = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "new", "content": big_new}]}
    state.record_exchange(a0, r0)
    state.record_exchange(a1, r1)
    # Budget fits prefix/view + exactly ONE big exchange.
    msgs = build_lead_context(state, ex, turn=3, token_budget=3000)
    blob = json.dumps(msgs)
    assert big_new in blob      # newest exchange kept
    assert big_old not in blob  # oldest exchange dropped as a whole unit
    # Whatever survives is paired: every tool_use has its tool_result.
    uses = {b["id"] for m in msgs for b in _blocks(m) if b.get("type") == "tool_use"}
    results = {b["tool_use_id"] for m in msgs for b in _blocks(m)
               if b.get("type") == "tool_result"}
    assert uses == results == {"new"}
