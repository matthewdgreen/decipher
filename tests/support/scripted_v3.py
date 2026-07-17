"""Scripted fake sessions + seed helpers for v3 lead-loop tests.

Hoisted from ``tests/test_loop_v3.py`` (M5.3 Slice 7, Part F) so the
Sequence-B replay test and any future v3 loop test can share one copy of the
scripted-session / worker-fake machinery instead of re-declaring it. Behavior
is identical to the originals; the only additions are the Sequence-B fakes at
the bottom (``make_verify_builder``, ``NEGATIVE_LOCAL_REPAIR_VERDICT``,
``OverBudgetReadingWorkerFake``).

Not a cassette framework — plain deterministic fakes only.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent.model_provider import (  # noqa: E402
    ModelProviderError,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolUseBlock,
)
from agent.loop_shared import (  # noqa: E402
    _candidate_content_hash,
    _decoded_text_for_panel,
)
from investigation import sessions as sessions_mod  # noqa: E402
from investigation.reading import (  # noqa: E402
    Reading,
    build_candidate_reading_packet,
)
from investigation.sessions import SessionCapabilities  # noqa: E402
from investigation.state import BudgetEntry, InvestigationState  # noqa: E402
from models.alphabet import Alphabet  # noqa: E402
from models.cipher_text import CipherText  # noqa: E402
from workspace import Workspace  # noqa: E402


# ---------------------------------------------------------------------------
# Scripted lead/worker sessions
# ---------------------------------------------------------------------------
class ScriptedSession:
    """A ModelSession fake driven by a list of per-turn content-block lists."""

    def __init__(self, scripts, *, model="fake-openai", provider_name="openai"):
        self.model = model
        self.provider_name = provider_name
        self.capabilities = SessionCapabilities()
        self._scripts = list(scripts)
        self._budget: list[BudgetEntry] = []
        self.blocks_seen: list = []
        self.tools_seen: list = []
        self._n = 0

    def send(self, blocks, tools=None, max_tokens=8192):
        self.blocks_seen.append(blocks)
        self.tools_seen.append(tools)
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


class VerifyWorkerFake:
    """M5: a verify episode worker that accepts the candidate as coherent English.

    Registered per test via the ``verify_fake`` fixture; the run's
    AttestationPolicy needs an attestation before meta_declare_solution is
    allowed. Remains the positive-verdict shorthand — see ``make_verify_builder``
    for arbitrary verdicts.
    """

    def __init__(self, provider=None, system="", role="episode:verify"):
        self.model = "fake-luna"
        self.provider_name = "openai"
        self.capabilities = SessionCapabilities()
        self._budget: list[BudgetEntry] = []

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(
            BudgetEntry("episode:verify", "openai", "fake-luna", 50, 10, 0)
        )
        result = {"coherence": 9, "reader_accepts": True,
                  "reader_accepts_as_solution": True,
                  "target_language_confidence": 0.9,
                  "semantic_recoverability": 0.8,
                  "damage_scope": "local", "repairability": "local_repair",
                  "uncertainty_note": "",
                  "gloss": "reads as clear English", "anomalies": [],
                  "confidence": "high"}
        return ModelResponse(
            content=[ToolUseBlock(id="v1", name="episode_submit_result",
                                  input={"result": result, "summary": "reads well"})],
            usage=ModelUsage(50, 10, 0))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": self.model, "exchanges": []}


# ---------------------------------------------------------------------------
# Programmable repair worker (Slice 2/4 FIFO)
# ---------------------------------------------------------------------------
REPAIR_PROGRAMS: list = []


class ProgrammableRepairWorker:
    """Repair worker driven by a FIFO of per-episode programs.

    Each program is ``{"apply": [apply-arg dicts], "result": callable(forks)->dict}``.
    ``apply`` entries lacking reading_text/fragments/reading_id get the injected
    reading id; ``result`` receives the created fork names (in call order)."""

    def __init__(self, provider, system, role):
        self.model = "fake-repair"; self.provider_name = "openai"
        self.capabilities = SessionCapabilities(); self._budget = []
        prog = REPAIR_PROGRAMS.pop(0) if REPAIR_PROGRAMS else {
            "apply": [],
            "result": (lambda forks: {
                "applied": True, "best_branch": None, "edits": [],
                "verdicts": [], "collateral": {}, "notes": "",
            }),
        }
        self._applies = list(prog.get("apply") or [])
        self._result_fn = prog["result"]
        self._step = 0
        self._reading_id = None

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(BudgetEntry("episode:repair", "openai", self.model, 10, 5, 0))
        if self._reading_id is None:
            m = re.search(r"id `([0-9a-f]{12})`", json.dumps(blocks, default=str))
            self._reading_id = m.group(1) if m else None
        if self._step < len(self._applies):
            apply_args = dict(self._applies[self._step])
            apply_args.setdefault("branch", "main")
            if not any(k in apply_args for k in ("reading_text", "fragments", "reading_id")):
                apply_args["reading_id"] = self._reading_id
            self._step += 1
            return ModelResponse(
                content=[ToolUseBlock(id=f"ap{self._step}",
                                      name="hypothesis_apply_reading", input=apply_args)],
                usage=ModelUsage(10, 5, 0))
        forks: list[str] = []
        for message in blocks:
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                try:
                    payload = json.loads(block.get("content") or "")
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and isinstance(payload.get("fork"), str):
                    if payload["fork"] not in forks:
                        forks.append(payload["fork"])
        result = self._result_fn(forks)
        return ModelResponse(
            content=[ToolUseBlock(id="sub", name="episode_submit_result",
                                  input={"result": result, "summary": "programmed"})],
            usage=ModelUsage(10, 5, 0))

    def usage_entries(self): return list(self._budget)
    def export_transcript(self): return {"provider": "openai", "exchanges": []}


def register_programmable_repair(programs):
    REPAIR_PROGRAMS[:] = list(programs)
    sessions_mod.register_session_builder("episode:repair", ProgrammableRepairWorker)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------
def keyed_catton_state():
    raw = "abcde"  # single word, decodes to CATON
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ws = Workspace(ct)
    pt = ws.plaintext_alphabet
    for sym, letter in {"a": "C", "b": "A", "c": "T", "d": "O", "e": "N"}.items():
        ws.set_mapping("main", alpha.id_for(sym), pt.id_for(letter))
    state = InvestigationState(workspace=ws, language="en")
    state.turn = 0
    return ct, state


def seed_reading(state, text, *, reading_id=None, created_turn=0):
    packet = build_candidate_reading_packet(state.workspace, "main").to_dict()
    reading = Reading.from_episode_result(
        {"reading_text": text,
         "fragments": [{"text": text, "repair_text": text, "confidence": 0.95}],
         "holes": [], "overall_confidence": 0.9},
        branch="main", source="lead", created_turn=created_turn,
        reading_id=reading_id, candidate_packet=packet,
    )
    state.readings[reading.reading_id] = reading.to_dict()
    return reading.reading_id


def seed_negative_attestation(state, *, episode_id="prior_verify"):
    h = _candidate_content_hash(_decoded_text_for_panel(state.workspace, "main"))
    state.verify_attestations.append({
        "branch": "main", "content_hash": h, "renderer_id": "decoded_text_v1",
        "episode_id": episode_id, "coherence": 4, "reader_accepts": False,
        "reader_accepts_as_solution": False,
        "target_language_confidence": 0.8, "semantic_recoverability": 0.7,
        "damage_scope": "local", "repairability": "local_repair",
        "gloss": "partly readable", "anomalies": ["damaged middle word"],
        "created_turn": 0,
    })
    return h


# ---------------------------------------------------------------------------
# M5.3 Slice 7 (Sequence B) additions
# ---------------------------------------------------------------------------
NEGATIVE_LOCAL_REPAIR_VERDICT = {
    "coherence": 4, "reader_accepts": False,
    "reader_accepts_as_solution": False,
    "target_language_confidence": 0.8, "semantic_recoverability": 0.7,
    "damage_scope": "local", "repairability": "local_repair",
    "uncertainty_note": "", "gloss": "partly readable",
    "anomalies": ["damaged middle word"], "confidence": "medium",
}


class _ConfigurableVerifyWorker:
    """A verify worker submitting a caller-supplied verdict dict."""

    def __init__(self, result, provider=None, system="", role="episode:verify"):
        self.model = "fake-luna"
        self.provider_name = "openai"
        self.capabilities = SessionCapabilities()
        self._budget: list[BudgetEntry] = []
        self._result = result

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(
            BudgetEntry("episode:verify", "openai", "fake-luna", 50, 10, 0)
        )
        return ModelResponse(
            content=[ToolUseBlock(id="v1", name="episode_submit_result",
                                  input={"result": dict(self._result),
                                         "summary": "verified"})],
            usage=ModelUsage(50, 10, 0))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": self.model, "exchanges": []}


def make_verify_builder(result: dict):
    """Session-builder factory: a verify worker submitting ``result``.

    (VerifyWorkerFake remains the positive-verdict shorthand.)"""
    def _builder(provider=None, system="", role="episode:verify"):
        return _ConfigurableVerifyWorker(result, provider, system, role)
    return _builder


class OverBudgetReadingWorkerFake:
    """Reading worker that emits FOUR decode_show tool_uses in one batch on its
    first send (over a 2-call effective budget → 2 executed + 2 synthesized
    budget_exhausted skips), never volunteers a submit, and on the submit-only
    reserve send (the 'Budget reached' nudge, exposing only
    episode_submit_result) submits a valid reading:
    reading_text/fragments[0].text = repair_text = CLS.READING_TEXT
    (default "CATON"), holes=[], overall_confidence=0.8.
    Model name "gpt-5.5" so BudgetEntry costs are nonzero-deterministic."""

    READING_TEXT = "CATON"

    def __init__(self, provider=None, system="", role="episode:reading"):
        self.model = "gpt-5.5"
        self.provider_name = "openai"
        self.capabilities = SessionCapabilities()
        self._budget: list[BudgetEntry] = []

    def send(self, blocks, tools=None, max_tokens=8192):
        self._budget.append(
            BudgetEntry("episode:reading", "openai", self.model, 10, 5, 0)
        )
        tool_names = {t.get("name") for t in (tools or []) if isinstance(t, dict)}
        if tool_names == {"episode_submit_result"}:
            # Submit-only reserve send: submit the valid reading.
            text = type(self).READING_TEXT
            content = [ToolUseBlock(id="rsub", name="episode_submit_result", input={
                "result": {"reading_text": text,
                           "fragments": [{"text": text, "repair_text": text,
                                          "confidence": 0.9}],
                           "holes": [], "overall_confidence": 0.8},
                "summary": f"read as {text}",
            })]
        else:
            # First send: over-emit FOUR decode_show calls in one batch.
            content = [
                ToolUseBlock(id=f"od{i}", name="decode_show",
                             input={"branch": "main"})
                for i in range(4)
            ]
        return ModelResponse(content=content, usage=ModelUsage(10, 5, 0))

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "exchanges": []}
