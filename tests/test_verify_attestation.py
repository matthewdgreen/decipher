"""M5 verify-attestation tests (Part 6 + amendments F1/F4/F8).

Covers the pinned renderer + hash determinism, the F1 renderer-trap negative
test, the KEY-rendered stale case + its metadata-frozen companion, the
AttestationPolicy gate (absent/match/weak), the F8 would-have-passed-v2-gates
control, the verify episode zero-tool submit path, and state round-trip.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.loop_shared import (
    DECODED_TEXT_RENDERER_ID,
    _candidate_content_hash,
    _decoded_text_for_panel,
)
from agent.model_provider import ModelResponse, ModelUsage, ToolUseBlock
from agent.tools_v2 import (
    AttestationPolicy,
    NoGatesPolicy,
    V2GatePolicy,
    WorkspaceToolExecutor,
)
from investigation.episodes import EPISODE_KINDS, EpisodeSpec, run_episode
from investigation.sessions import SessionCapabilities
from investigation.state import (
    AttestationRecord,
    BudgetEntry,
    InvestigationState,
)
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _cat_on_ws():
    """A keyed monoalphabetic workspace whose `main` apply_key decodes to CAT ON.

    No metadata decoded_text -> `_decoded_text_for_panel` is KEY-rendered, so key
    mutations change the rendered string (the stale-detection case, F1).
    """
    raw = "abc de"
    alpha = Alphabet.from_text(raw, ignore_chars={" "})
    ct = CipherText(raw=raw, alphabet=alpha, separator=" ")
    ws = Workspace(ct)
    pt = ws.plaintext_alphabet
    for sym, letter in {"a": "C", "b": "A", "c": "T", "d": "O", "e": "N"}.items():
        ws.set_mapping("main", alpha.id_for(sym), pt.id_for(letter))
    return ws, alpha, pt


def _simple_executor(ws, policy):
    return WorkspaceToolExecutor(
        workspace=ws, language="en", word_set={"CAT", "ON", "CAR"},
        word_list=["CAT", "ON", "CAR"], pattern_dict={},
        declaration_policy=policy,
    )


def _attestation_for(ws, branch, *, coherence=9, reader_accepts=True,
                     anomalies=None, episode_id="ep1", turn=1,
                     reader_accepts_as_solution=True,
                     target_language_confidence=0.9,
                     semantic_recoverability=0.8, damage_scope="local",
                     repairability="local_repair", uncertainty_note=""):
    """Build an AttestationRecord dict over the current rendered text.

    Slice 6: positive-shaped defaults (reader_accepts_as_solution True + the
    routing fields), so a bare call produces a positive attestation."""
    text = _decoded_text_for_panel(ws, branch)
    return AttestationRecord(
        branch=branch,
        content_hash=_candidate_content_hash(text),
        renderer_id=DECODED_TEXT_RENDERER_ID,
        episode_id=episode_id,
        coherence=coherence,
        reader_accepts=reader_accepts,
        gloss="reads as a short clause",
        anomalies=list(anomalies or []),
        created_turn=turn,
        reader_accepts_as_solution=reader_accepts_as_solution,
        target_language_confidence=target_language_confidence,
        semantic_recoverability=semantic_recoverability,
        damage_scope=damage_scope,
        repairability=repairability,
        uncertainty_note=uncertainty_note,
    ).to_dict()


def _declare_args(branch="main"):
    return {"branch": branch, "rationale": "reads well", "self_confidence": 0.9}


# ---------------------------------------------------------------------------
# Renderer determinism + hash match (Part 6)
# ---------------------------------------------------------------------------
def test_renderer_deterministic_and_attest_declare_hash_match():
    ws, _alpha, _pt = _cat_on_ws()
    a = _decoded_text_for_panel(ws, "main")
    b = _decoded_text_for_panel(ws, "main")
    assert a == b == "CAT ON"
    # The attest-time hash (dispatch) and the declare-time hash (policy) are the
    # SAME renderer over the same unchanged text.
    assert _candidate_content_hash(a) == _candidate_content_hash(
        _decoded_text_for_panel(ws, "main")
    )


def test_branch_decoded_text_matches_panel_renderer():
    """All candidate consumers use the attestation renderer.

    Metadata-backed candidates with host word spans used to produce different
    text in tools and verification, which made content hashes impossible to
    reason about.  The unified candidate path keeps the rendered text exact.
    """
    raw = "ABCDEF"
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    ws = Workspace(ct)
    ws.set_word_spans("main", [(0, 3), (3, 6)])
    # A no-whitespace metadata decode is re-spaced by the shared renderer.
    ws.get_branch("main").metadata["decoded_text"] = "HELLOX"

    executor = _simple_executor(ws, NoGatesPolicy())
    panel = _decoded_text_for_panel(ws, "main")
    tool_text = executor._branch_decoded_text("main")
    assert panel == "HEL LOX"
    assert tool_text == panel
    assert _candidate_content_hash(panel) == _candidate_content_hash(tool_text)


# ---------------------------------------------------------------------------
# Stale detection (F1: KEY-rendered branch) + metadata-frozen companion
# ---------------------------------------------------------------------------
def test_attestation_stale_on_key_rendered_branch():
    ws, alpha, pt = _cat_on_ws()
    attestations = [_attestation_for(ws, "main")]
    executor = _simple_executor(ws, AttestationPolicy(attestations))
    # Fresh: hash matches -> allowed.
    assert executor._declaration_policy.check_declare_solution(
        executor, _declare_args()
    ) is None

    # Mutate the KEY (no metadata decoded_text -> apply_key changes) -> stale.
    ws.set_mapping("main", alpha.id_for("c"), pt.id_for("R"))  # CAT ON -> CAR ON
    assert _decoded_text_for_panel(ws, "main") == "CAR ON"
    block = executor._declaration_policy.check_declare_solution(
        executor, _declare_args()
    )
    assert block is not None
    assert block["reason"] == "attestation_stale"
    assert block["status"] == "blocked" and block["accepted"] is False
    # And the handler returns the block verbatim (does not terminate the run).
    handler_out = executor._tool_meta_declare_solution(_declare_args())
    assert handler_out["reason"] == "attestation_stale"
    assert executor.terminated is False and executor.solution is None


def test_metadata_frozen_branch_attestation_stays_fresh():
    """Companion to the stale test: on a branch whose decode is driven by
    metadata['decoded_text'], key edits do NOT change the rendered string, so
    the attestation legitimately stays fresh (attested == declared == scored)."""
    ws, alpha, pt = _cat_on_ws()
    # Freeze the decode in metadata (with whitespace so it renders verbatim).
    ws.get_branch("main").metadata["decoded_text"] = "CAT ON"
    attestations = [_attestation_for(ws, "main")]
    executor = _simple_executor(ws, AttestationPolicy(attestations))

    # A key edit that WOULD change apply_key output, but metadata is frozen.
    ws.set_mapping("main", alpha.id_for("c"), pt.id_for("R"))
    assert _decoded_text_for_panel(ws, "main") == "CAT ON"  # unchanged
    assert executor._declaration_policy.check_declare_solution(
        executor, _declare_args()
    ) is None  # still fresh -> allowed


# ---------------------------------------------------------------------------
# AttestationPolicy: absent / match / weak (Part 6)
# ---------------------------------------------------------------------------
def test_attestation_policy_absent_blocks():
    ws, _alpha, _pt = _cat_on_ws()
    executor = _simple_executor(ws, AttestationPolicy([]))
    out = executor._tool_meta_declare_solution(_declare_args())
    assert out["reason"] == "attestation_required"
    assert out["status"] == "blocked" and out["accepted"] is False
    assert "verify" in out["how"]
    assert executor.solution is None and executor.terminated is False


def test_attestation_policy_positive_match_allows_and_accepts():
    ws, _alpha, _pt = _cat_on_ws()
    attestations = [_attestation_for(ws, "main")]
    executor = _simple_executor(ws, AttestationPolicy(attestations))
    out = executor._tool_meta_declare_solution(_declare_args())
    assert out["status"] == "ok" and out["accepted"] is True
    assert executor.terminated is True
    assert executor.solution is not None and executor.solution.branch == "main"


def test_attestation_policy_weak_blocks_declaration():
    """Weak (reader_accepts False / low coherence / reader_accepts_as_solution
    False) BLOCKS declaration (C6 reversed); the block echoes the verdict."""
    ws, _alpha, _pt = _cat_on_ws()
    weak = _attestation_for(
        ws, "main", coherence=2, reader_accepts=False,
        reader_accepts_as_solution=False,
        anomalies=["non-word GHXQ", "broken syntax"],
    )
    executor = _simple_executor(ws, AttestationPolicy([weak]))
    out = executor._tool_meta_declare_solution(_declare_args())
    assert out["status"] == "blocked"
    assert out["reason"] == "attestation_not_positive"
    assert executor.solution is None
    assert executor.terminated is False
    echo = out["attestation"]
    assert echo["reader_accepts_as_solution"] is False
    assert echo["anomalies"] == ["non-word GHXQ", "broken syntax"]


def test_attestation_policy_high_recoverability_alone_does_not_unlock():
    """B3: high semantic_recoverability + language confidence + local damage +
    reader_accepts True + coherence 10, but reader_accepts_as_solution False ->
    STILL blocked. Only reader_accepts_as_solution unlocks declaration."""
    ws, _alpha, _pt = _cat_on_ws()
    att = _attestation_for(
        ws, "main", coherence=10, reader_accepts=True,
        reader_accepts_as_solution=False,
        target_language_confidence=1.0, semantic_recoverability=1.0,
        damage_scope="local", repairability="local_repair",
    )
    executor = _simple_executor(ws, AttestationPolicy([att]))
    out = executor._tool_meta_declare_solution(_declare_args())
    assert out["status"] == "blocked"
    assert out["reason"] == "attestation_not_positive"


def test_attestation_policy_latest_verdict_governs():
    """Two attestations on the SAME current hash: the NEWEST verdict governs."""
    ws, _alpha, _pt = _cat_on_ws()
    older_positive = _attestation_for(
        ws, "main", episode_id="ep_old", turn=1,
        reader_accepts_as_solution=True,
    )
    newer_negative = _attestation_for(
        ws, "main", episode_id="ep_new", turn=2,
        reader_accepts=False, reader_accepts_as_solution=False,
    )
    executor = _simple_executor(
        ws, AttestationPolicy([older_positive, newer_negative])
    )
    out = executor._tool_meta_declare_solution(_declare_args())
    assert out["status"] == "blocked"
    assert out["reason"] == "attestation_not_positive"

    # Reversed order (newest is positive) -> allowed.
    ws2, _a2, _p2 = _cat_on_ws()
    older_negative = _attestation_for(
        ws2, "main", episode_id="ep_old", turn=1,
        reader_accepts=False, reader_accepts_as_solution=False,
    )
    newer_positive = _attestation_for(
        ws2, "main", episode_id="ep_new", turn=2,
        reader_accepts_as_solution=True,
    )
    executor2 = _simple_executor(
        ws2, AttestationPolicy([older_negative, newer_positive])
    )
    assert executor2._declaration_policy.check_declare_solution(
        executor2, _declare_args()
    ) is None


def test_attestation_policy_legacy_record_positive_via_old_condition():
    """A pre-Slice-6 dict (no new keys) is positive iff reader_accepts and
    coherence>=7 (the frozen legacy condition), pinned at the gate."""
    ws, _alpha, _pt = _cat_on_ws()
    chash = _candidate_content_hash(_decoded_text_for_panel(ws, "main"))
    legacy_positive = {
        "branch": "main", "content_hash": chash,
        "renderer_id": DECODED_TEXT_RENDERER_ID, "episode_id": "ep_legacy",
        "coherence": 7, "reader_accepts": True, "gloss": "g",
        "anomalies": [], "created_turn": 1,
    }
    executor = _simple_executor(ws, AttestationPolicy([legacy_positive]))
    assert executor._declaration_policy.check_declare_solution(
        executor, _declare_args()
    ) is None

    ws2, _a2, _p2 = _cat_on_ws()
    legacy_weak = dict(legacy_positive, coherence=6)
    executor2 = _simple_executor(ws2, AttestationPolicy([legacy_weak]))
    out = executor2._tool_meta_declare_solution(_declare_args())
    assert out["status"] == "blocked"
    assert out["reason"] == "attestation_not_positive"


def test_attestation_matched_by_content_hash_not_branch_name():
    """F11 branch-rename edge: a hash match on a DIFFERENT branch name allows."""
    ws, _alpha, _pt = _cat_on_ws()
    att = _attestation_for(ws, "main")
    att["branch"] = "renamed_elsewhere"  # branch recorded for observability only
    executor = _simple_executor(ws, AttestationPolicy([att]))
    assert executor._declaration_policy.check_declare_solution(
        executor, _declare_args()
    ) is None


# ---------------------------------------------------------------------------
# F8: the new check is the operative control
# ---------------------------------------------------------------------------
def test_no_attestation_blocks_branch_that_passes_v2_gates():
    """A single-branch fresh executor with no pending agenda/finalist passes
    EVERY v2 declare gate. The M5 AttestationPolicy blocks the same declaration
    with no attestation, proving the attestation check is the operative control."""
    ws, _alpha, _pt = _cat_on_ws()

    # V2GatePolicy allows it (all quick prerequisites are satisfied).
    v2_exec = _simple_executor(ws, V2GatePolicy())
    v2_out = v2_exec._tool_meta_declare_solution(_declare_args())
    assert v2_out["status"] == "ok" and v2_out["accepted"] is True

    # Same branch, AttestationPolicy, no attestation -> blocked.
    ws2, _a2, _p2 = _cat_on_ws()
    att_exec = _simple_executor(ws2, AttestationPolicy([]))
    att_out = att_exec._tool_meta_declare_solution(_declare_args())
    assert att_out["status"] == "blocked"
    assert att_out["reason"] == "attestation_required"


# ---------------------------------------------------------------------------
# meta_declare_unsolved is NOT gated
# ---------------------------------------------------------------------------
def test_declare_unsolved_not_gated_by_attestation():
    ws, _alpha, _pt = _cat_on_ws()
    executor = _simple_executor(ws, AttestationPolicy([]))
    out = executor._tool_meta_declare_unsolved({
        "rationale": "resisted", "branches_considered": ["main"],
        "best_branch": "main", "reading_summary": "partial",
        "further_iterations_helpful": True, "further_iterations_note": "n",
    })
    assert out["status"] == "ok" and out["accepted"] is True
    assert executor.terminated is True


# ---------------------------------------------------------------------------
# verify episode: zero-tool-call submit path (Part 1, confirmed by review)
# ---------------------------------------------------------------------------
class _VerifyFake:
    def __init__(self, result, *, role="episode:verify"):
        self.model = "fake-luna"
        self.provider_name = "openai"
        self.capabilities = SessionCapabilities()
        self._result = result
        self._budget: list[BudgetEntry] = []
        self.blocks_seen: list = []

    def send(self, blocks, tools=None, max_tokens=8192):
        self.blocks_seen.append(blocks)
        self._budget.append(BudgetEntry("episode:verify", "openai", "fake-luna", 40, 8, 0))
        return ModelResponse(
            content=[ToolUseBlock(id="v1", name="episode_submit_result",
                                  input={"result": self._result, "summary": "judged"})],
            usage=ModelUsage(40, 8, 0),
        )

    def usage_entries(self):
        return list(self._budget)

    def export_transcript(self):
        return {"provider": "openai", "model": self.model, "exchanges": []}


def _simple_state(raw="ABCDEFGHIJKL"):
    alpha = Alphabet.from_text(raw, ignore_chars=set())
    ct = CipherText(raw=raw, alphabet=alpha, separator=None)
    return InvestigationState(workspace=Workspace(ct), language="en")


def test_verify_episode_zero_tool_submit():
    state = _simple_state()
    good = {"coherence": 7, "reader_accepts": True,
            "reader_accepts_as_solution": True, "gloss": "reads",
            "anomalies": [], "confidence": "medium"}
    spec = EpisodeSpec("verify", "judge this",
                       inputs={"candidate_text": "SOME PROPOSED TEXT", "language": "en"})
    fake = _VerifyFake(good)
    res = run_episode(spec, state, session=fake)
    assert res.status == "ok"
    assert res.result == good
    # Verify submitted on its first (and only) send: zero prior tool calls.
    assert res.tool_call_count == 0
    assert len(fake.blocks_seen) == 1
    # No branch snapshots ride out of a verify episode.
    assert res.branch_snapshots == []


def test_verify_episode_budget_tagged():
    state = _simple_state()
    good = {"coherence": 5, "reader_accepts": False,
            "reader_accepts_as_solution": False,
            "gloss": "words but not sentences",
            "anomalies": ["fragmented"], "confidence": "low"}
    spec = EpisodeSpec("verify", "judge",
                       inputs={"candidate_text": "WORD ISLANDS HERE", "language": "en"})
    res = run_episode(spec, state, session=_VerifyFake(good))
    assert res.status == "ok"
    assert res.budget_entries and res.budget_entries[0]["category"] == "episode:verify"


# ---------------------------------------------------------------------------
# State round-trip + old-artifact default (F11)
# ---------------------------------------------------------------------------
def test_verify_attestations_roundtrip_and_old_artifact_default():
    ws, _alpha, _pt = _cat_on_ws()
    state = InvestigationState(workspace=ws, language="en")
    state.verify_attestations.append(_attestation_for(ws, "main"))
    data = json.loads(json.dumps(state.to_artifact_dict()))
    assert data["verify_attestations"][0]["content_hash"]
    restored = InvestigationState.from_artifact_dict(data)
    assert restored.verify_attestations == state.verify_attestations

    # A pre-M5 artifact has no verify_attestations key -> empty on load.
    del data["verify_attestations"]
    restored2 = InvestigationState.from_artifact_dict(data)
    assert restored2.verify_attestations == []


def test_attestation_record_from_dict_defaults():
    rec = AttestationRecord.from_dict({"content_hash": "abc"})
    assert rec.content_hash == "abc"
    assert rec.coherence == 0 and rec.reader_accepts is False
    assert rec.anomalies == []
    # Slice 6 conservative defaults on an old-shape dict.
    assert rec.target_language_confidence == 0.0
    assert rec.semantic_recoverability == 0.0
    assert rec.damage_scope == "basin_wide"
    assert rec.repairability == "none"
    assert rec.reader_accepts_as_solution is False
    assert rec.uncertainty_note == ""


def test_attestation_record_legacy_load_derivation():
    """Legacy (no reader_accepts_as_solution key) positivity derivation +
    strict handling of the explicit key + routing-field clamping."""
    assert AttestationRecord.from_dict(
        {"reader_accepts": True, "coherence": 9}
    ).reader_accepts_as_solution is True
    assert AttestationRecord.from_dict(
        {"reader_accepts": True, "coherence": 6}
    ).reader_accepts_as_solution is False
    assert AttestationRecord.from_dict(
        {"reader_accepts": False, "coherence": 10}
    ).reader_accepts_as_solution is False
    # The key, when present, wins over the legacy fields.
    assert AttestationRecord.from_dict({
        "reader_accepts_as_solution": False,
        "reader_accepts": True, "coherence": 10,
    }).reader_accepts_as_solution is False
    clamped = AttestationRecord.from_dict({
        "reader_accepts_as_solution": True,
        "target_language_confidence": 1.7,
        "semantic_recoverability": -0.2,
        "damage_scope": "LOCAL", "repairability": "spolish",
    })
    assert clamped.target_language_confidence == 1.0
    assert clamped.semantic_recoverability == 0.0
    assert clamped.damage_scope == "basin_wide"
    assert clamped.repairability == "none"
