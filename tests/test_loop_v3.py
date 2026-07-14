"""run_v3 lead-loop tests (M1 Part 4/5) with scripted fake sessions."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.model_provider import ModelProviderError, ModelResponse, ModelUsage, TextBlock, ToolUseBlock
from investigation.loop_v3 import run_v3
from investigation.sessions import SessionCapabilities
from investigation.state import BudgetEntry
from models.alphabet import Alphabet
from models.cipher_text import CipherText


def _caesar(text: str, shift: int) -> str:
    return "".join(
        chr((ord(c) - 65 + shift) % 26 + 65) if c.isalpha() else c for c in text
    )


def _caesar_cipher(plaintext: str, shift: int = 3):
    raw = _caesar(plaintext, shift)
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    return CipherText(raw=raw, alphabet=alpha, separator=" "), alpha


class ScriptedSession:
    """A ModelSession fake driven by a list of per-turn content-block lists."""

    def __init__(self, scripts, *, model="fake-openai", provider_name="openai"):
        self.model = model
        self.provider_name = provider_name
        self.capabilities = SessionCapabilities()
        self._scripts = list(scripts)
        self._budget: list[BudgetEntry] = []
        self.blocks_seen: list = []
        self._n = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self.blocks_seen.append(blocks)
        self._budget.append(
            BudgetEntry("lead", self.provider_name, self.model, 1000, 50, 100)
        )
        content = self._scripts[min(self._n, len(self._scripts) - 1)]
        self._n += 1
        return ModelResponse(content=content, usage=ModelUsage(1000, 50, 100))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": self.provider_name, "model": self.model,
                "exchanges": [{"n": self._n}]}


class ErrorSession(ScriptedSession):
    def send(self, blocks, tools=None, max_tokens=8192):
        raise ModelProviderError("simulated API overload")


def _solve_scripts(alpha):
    mapping = {
        alpha.symbol_for(i): _caesar(alpha.symbol_for(i), -3)
        for i in range(alpha.size)
    }
    return [
        [TextBlock(text="Looks like a Caesar; setting the whole key."),
         ToolUseBlock(id="t1", name="act_bulk_set",
                      input={"branch": "main", "mappings": mapping})],
        [ToolUseBlock(id="t2", name="decode_show", input={"branch": "main"})],
        [TextBlock(text="Reads as English."),
         ToolUseBlock(id="t3", name="meta_declare_solution",
                      input={"branch": "main",
                             "rationale": "Caesar shift 3 recovered; reads as English.",
                             "self_confidence": 0.95})],
    ]


def test_run_v3_scripted_solve_and_declare_no_gates():
    ct, alpha = _caesar_cipher("THE DOG")
    scripts = _solve_scripts(alpha)
    # Fork an extra branch on turn 1 so a v2 branch_cards gate WOULD block the
    # declaration; under NoGatesPolicy it must be accepted immediately.
    scripts[0].insert(
        1,
        ToolUseBlock(id="t0", name="workspace_fork",
                     input={"new_name": "alt", "from_branch": "main"}),
    )
    session = ScriptedSession(scripts)
    art = run_v3(ct, session=session, language="en", max_iterations=10,
                 cipher_id="v3_caesar")

    assert art.status == "solved"
    assert art.loop_version == "v3"
    assert art.solution is not None and art.solution.branch == "main"
    assert abs(art.solution.self_confidence - 0.95) < 1e-9
    # The declaration was accepted on the FIRST call (no gate bounce).
    declare_results = [
        json.loads(tc.result) for tc in art.tool_calls
        if tc.tool_name == "meta_declare_solution"
    ]
    assert len(declare_results) == 1
    assert declare_results[0]["accepted"] is True
    # Decode is correct.
    main_decode = next(b.decryption for b in art.branches if b.name == "main")
    assert main_decode == "THE DOG"


def test_run_v3_tool_iteration_is_lead_turn():
    ct, alpha = _caesar_cipher("THE DOG")
    session = ScriptedSession(_solve_scripts(alpha))
    art = run_v3(ct, session=session, language="en", max_iterations=10,
                 cipher_id="v3_iter")
    by_name = {tc.tool_name: tc.iteration for tc in art.tool_calls}
    assert by_name["act_bulk_set"] == 1
    assert by_name["decode_show"] == 2
    assert by_name["meta_declare_solution"] == 3


def test_run_v3_records_state_budget_and_transcript():
    ct, alpha = _caesar_cipher("THE DOG")
    session = ScriptedSession(_solve_scripts(alpha))
    art = run_v3(ct, session=session, language="en", max_iterations=10,
                 cipher_id="v3_state")
    # Budget accounting flows from the session's per-send entries.
    assert art.budget_by_category["lead"]["calls"] == 3
    assert art.total_input_tokens == 3000
    assert art.total_output_tokens == 150
    assert art.total_cache_read_tokens == 300
    # Artifact carries the v3 additions.
    assert art.session_transcript["provider"] == "openai"
    assert art.investigation_state is not None
    kinds = [e["kind"] for e in art.investigation_state["evidence_log"]]
    assert "diagnostic_preflight" in kinds
    assert "turn_summary" in kinds
    # Logical transcript stores only new per-turn content (no rebuilt contexts).
    assert all(m["role"] in {"assistant", "user"} for m in art.messages)


def test_run_v3_fallback_declared_on_provider_error():
    ct, _alpha = _caesar_cipher("THE DOG")
    art = run_v3(ct, session=ErrorSession([]), language="en", max_iterations=5,
                 cipher_id="v3_error")
    assert art.status == "fallback_declared"
    assert art.auto_declared is True
    assert art.solution is not None
    assert art.solution.self_confidence == 0.0
    assert "API error" in art.error_message


def test_run_v3_exhaustion_falls_back():
    ct, _alpha = _caesar_cipher("THE DOG")
    # Text-only responses → no tool calls → exhausted → fallback.
    session = ScriptedSession([[TextBlock(text="I have no idea.")]])
    art = run_v3(ct, session=session, language="en", max_iterations=3,
                 cipher_id="v3_exhaust")
    assert art.status == "fallback_declared"
    assert art.auto_declared is True
    assert art.solution is not None


def test_run_v3_resume_from_state_continues():
    ct, alpha = _caesar_cipher("THE DOG")
    session = ScriptedSession(_solve_scripts(alpha))
    art = run_v3(ct, session=session, language="en", max_iterations=10,
                 cipher_id="v3_resume")
    # Reload state from the artifact and confirm it is a valid resume seed.
    from investigation.state import InvestigationState
    reloaded = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(art.investigation_state))
    )
    assert reloaded.workspace.get_branch("main").key  # key survived
    assert reloaded.language == "en"


def test_run_v3_resume_continues_from_state_turn_monotonic():
    """R1: resume continues from state.turn + 1 with monotonic turn numbers."""
    from investigation.state import InvestigationState
    ct, alpha = _caesar_cipher("THE DOG")
    # Leg 1: two non-declaring turns, exhausts at max_iterations=2.
    noop = [[ToolUseBlock(id="n1", name="decode_show", input={"branch": "main"})],
            [ToolUseBlock(id="n2", name="decode_show", input={"branch": "main"})]]
    first = run_v3(ct, session=ScriptedSession(noop), language="en",
                   max_iterations=2, cipher_id="v3_resume_leg1")
    assert first.investigation_state["turn"] == 2

    reloaded = InvestigationState.from_artifact_dict(
        json.loads(json.dumps(first.investigation_state)))
    # Leg 2: resume and solve. Turns continue strictly past the pre-resume turns.
    seen = []
    second = run_v3(
        ct, session=ScriptedSession(_solve_scripts(alpha)), language="en",
        max_iterations=8, cipher_id="v3_resume_leg2", resume_state=reloaded,
        on_event=lambda e, p: seen.append(p["iteration"])
        if e == "iteration_start" else None,
    )
    assert seen == sorted(seen)      # monotonic
    assert seen and seen[0] == 3     # resumed at state.turn + 1
    assert second.status == "solved"
    # The turn-0 diagnostic preflight is not re-added on resume.
    kinds = [e["kind"] for e in second.investigation_state["evidence_log"]]
    assert kinds.count("diagnostic_preflight") == 1


def test_run_v3_interrupt_pairs_all_tool_results(monkeypatch):
    """R5: a mid-batch interrupt pairs every tool_use with a stopped result."""
    from agent import tools_v2
    ct, _alpha = _caesar_cipher("THE DOG")
    # One turn emitting TWO tool calls; the executor is interrupted on the 1st.
    scripts = [[
        ToolUseBlock(id="x1", name="decode_show", input={"branch": "main"}),
        ToolUseBlock(id="x2", name="decode_show", input={"branch": "main"}),
    ]]

    def boom(self, name, args, tool_use_id=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(tools_v2.WorkspaceToolExecutor, "execute", boom)
    art = run_v3(ct, session=ScriptedSession(scripts), language="en",
                 max_iterations=1, cipher_id="v3_interrupt")
    assert art.status == "stopped"

    def pair_sets(messages):
        uses, results = set(), set()
        for m in messages:
            for b in (m.get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    uses.add(b["id"])
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results.add(b["tool_use_id"])
        return uses, results

    # Both the logical transcript and the serialized state exchange are paired.
    assert pair_sets(art.messages) == ({"x1", "x2"}, {"x1", "x2"})
    assert pair_sets(art.investigation_state["recent_exchanges"]) == (
        {"x1", "x2"}, {"x1", "x2"})
