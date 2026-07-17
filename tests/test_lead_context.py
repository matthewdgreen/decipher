"""Lead context builder tests (M1 Part 2/5): determinism, budgets, rotation."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.loop_shared import _best_branch_for_auto_declare
from agent.tools_v2 import NoGatesPolicy, WorkspaceToolExecutor
from investigation.context import (
    CHARS_PER_TOKEN,
    build_lead_context,
    build_v3_system_prompt,
    rotating_window_bounds,
    workflow_state,
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
    assert "## Workflow state: searching" in joined
    assert "## Hypothesis board" in joined
    assert "## Recent evidence" in joined
    assert "seeded evidence entry" in joined
    assert "## Full-decode window" in joined


def test_negative_partial_attestation_creates_repair_action_menu():
    state = _state()
    ex = _executor(state)
    from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel

    text = _decoded_text_for_panel(state.workspace, "main")
    state.verify_attestations.append({
        "branch": "main",
        "content_hash": _candidate_content_hash(text),
        "coherence": 4,
        "reader_accepts": False,
        "gloss": "partly readable",
        "anomalies": ["broken middle"],
        "created_turn": 2,
    })
    menu = workflow_state(state, ex)
    assert menu["state"] == "repair_required"
    assert any("repair episode" in action for action in menu["actions"])


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


def test_v3_brief_has_self_narration_line_v2_untouched():
    """CLI-2 Part 2: the v3 lead brief instructs one self-narration sentence per
    turn, and the v2 system prompt is left byte-untouched (no such line)."""
    from agent.prompts_v2 import get_system_prompt

    narration_marker = "one short plain-language sentence"
    en = build_v3_system_prompt("en")
    de = build_v3_system_prompt("de")
    assert narration_marker in en
    assert narration_marker in de
    # v3-only: v2's brief (both styles) must not have grown the line.
    assert narration_marker not in get_system_prompt("en", "full")
    assert narration_marker not in get_system_prompt("en", "compact")
    assert narration_marker not in get_system_prompt("la", "full")


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


def test_truncate_never_exceeds_cap_and_reports_honest_count():
    from investigation.context import _truncate

    text = "x" * 500
    # Ordinary truncation: result fits the cap and the marker reports the
    # actual number of dropped characters (len(text) - kept prefix).
    for cap in (400, 100, 50):
        out = _truncate(text, cap)
        assert len(out) <= cap
        kept = out.index("\n") if "\n" in out else len(out)
        reported = int(out.split("truncated ")[1].split(" ")[0])
        assert reported == len(text) - kept

    # Tiny caps (cap < marker length): the marker head only, never overshoot.
    for cap in (10, 3, 1):
        out = _truncate(text, cap)
        assert len(out) <= cap
    # cap=0 → empty.
    assert _truncate(text, 0) == ""
    # No-op path unchanged.
    assert _truncate("short", 10) == "short"


def test_late_turn_attestation_hint():
    """F9: with ≤2 turns left and no fresh attestation on the best branch, the
    view carries a one-line reminder to run verify now."""
    from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel

    state = _state()
    ex = _executor(state)

    def _joined(turn, max_turns):
        return "\n".join(_texts(build_lead_context(state, ex, turn, 8000, max_turns)))

    # Early turn: no hint.
    assert "Attestation reminder" not in _joined(1, 20)
    # Late turn (≤2 remaining), no attestation: hint present.
    late = _joined(19, 20)
    assert "Attestation reminder" in late
    assert "run verify now" in late
    # A fresh matching attestation on the best branch suppresses the hint.
    best, _scores = _best_branch_for_auto_declare(
        state.workspace, state.language, ex.word_set, ex._freq_rank
    )
    state.verify_attestations.append({
        "branch": best,
        "content_hash": _candidate_content_hash(
            _decoded_text_for_panel(state.workspace, best)
        ),
    })
    assert "Attestation reminder" not in _joined(19, 20)
    # No max_turns -> no hint (nothing to bound against).
    assert "Attestation reminder" not in _joined(19, None)


# ---------------------------------------------------------------------------
# M5.3 Slice 2 — repair_exhausted workflow phase (pure context tests)
# ---------------------------------------------------------------------------
import pytest

from agent.loop_shared import _candidate_content_hash, _decoded_text_for_panel
from investigation.context import allowed_episode_kinds
from investigation.state import (
    attestation_key, new_saturation_entry, saturation_key,
)


def _seed_exhausted(state, *, pending=None, evidence_failures=2):
    """Seed a negative attestation on main + an exhausted saturation entry
    keyed on main's (content, verifier evidence). Returns (hash, att_key)."""
    h = _candidate_content_hash(_decoded_text_for_panel(state.workspace, "main"))
    att = {
        "branch": "main", "content_hash": h, "renderer_id": "decoded_text_v1",
        "episode_id": "prior", "coherence": 4, "reader_accepts": False,
        "gloss": "partial", "anomalies": ["broken word"], "created_turn": 1,
    }
    state.verify_attestations.append(att)
    att_key = attestation_key(att)
    entry = new_saturation_entry(h, att_key, 1)
    entry["evidence_failures"] = evidence_failures
    entry["exhausted"] = True
    if pending is not None:
        entry["pending_experiment_id"] = pending
    state.repair_saturation[saturation_key(h, att_key)] = entry
    return h, att_key


def test_s2_new_content_returns_to_candidate_reading():
    state = _state()
    ex = _executor(state)
    _seed_exhausted(state)
    state.readings["r1"] = {"reading_id": "r1", "branch": "main"}
    assert workflow_state(state, ex)["state"] == "repair_exhausted"
    # Changing the best branch content re-keys the saturation entry, so the
    # candidate returns to candidate_reading (new content requires verification).
    ct_alpha = state.workspace.cipher_text.alphabet
    pt = state.workspace.plaintext_alphabet
    state.workspace.set_mapping("main", ct_alpha.id_for("A"), pt.id_for("Z"))
    menu = workflow_state(state, ex)
    assert menu["state"] != "repair_exhausted"
    assert menu["state"] == "candidate_reading"


def test_s2_pending_experiment_offers_collect_and_excludes_repair_kinds():
    state = _state()
    ex = _executor(state)
    _seed_exhausted(state, pending="exp123")
    state.experiment_queue.append({
        "experiment_id": "exp123", "type": "automated_solver",
        "status": "running", "collected": False,
    })
    menu = workflow_state(state, ex)
    assert menu["state"] == "repair_exhausted"
    assert any("exp123" in a and "experiment_collect" in a for a in menu["actions"])
    assert allowed_episode_kinds(state, ex) == ["search", "compare", "verify"]
    # Once collected, the collect action drops (the other three remain).
    state.experiment_queue[0]["collected"] = True
    menu2 = workflow_state(state, ex)
    assert not any("exp123" in a for a in menu2["actions"])
    assert len(menu2["actions"]) == 3


def test_s2_unknown_phase_fails_closed_to_verify_with_warning(monkeypatch):
    import investigation.context as ctx_mod

    state = _state()
    ex = _executor(state)
    monkeypatch.setattr(
        ctx_mod, "workflow_state",
        lambda *a, **k: {"state": "someday_phase", "branch": None, "actions": []},
    )
    with pytest.warns(RuntimeWarning):
        kinds = allowed_episode_kinds(state, ex)
    assert kinds == ["verify"]
