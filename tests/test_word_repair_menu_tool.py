"""Agent-facing word-repair menu tool tests (Improvement Phase 2.5).

Covers the composite-action triplet built on the shared FinalistSessionStore:
``search_word_repair_menu`` (generate + install session), the paging review
tool, the folded rate branch of ``act_rate_transform_finalist``, and
``act_install_word_repair_finalists`` (whole-candidate fork install). The heavy
``build_word_repair_menu`` pipeline is stubbed so these stay fast and
deterministic; the install path exercises the REAL shared
``apply_word_repair_edits`` guard.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent.tools_v2 as tools_v2
from agent.tools_v2 import WorkspaceToolExecutor
from analysis.candidate_packet import CandidatePacket
from automated.runner import WordRepairMenu
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace


def _executor(mode: str = "homophonic_substitution") -> WorkspaceToolExecutor:
    alpha = Alphabet(["01", "02", "03"])
    ct = CipherText(raw="01 02 03 01 02 03", alphabet=alpha, separator=None)
    ex = WorkspaceToolExecutor(
        workspace=Workspace(ct),
        language="en",
        word_set={"THE", "CATS"},
        word_list=["THE", "CATS"],
        pattern_dict={},
    )
    ex.set_iteration(3)
    ex.workspace.set_full_key("main", {0: 0, 1: 1, 2: 2})
    ex.workspace.get_branch("main").metadata["cipher_mode"] = mode
    return ex


def _packet(rank: int, edits: list[str], val: float, adj: float, accepted: bool = True) -> CandidatePacket:
    return CandidatePacket(
        candidate_id="edits:" + ",".join(edits),
        kind="word_repair",
        rank=rank,
        source={"solver": "word_hypothesis_repair"},
        text=None,  # F3 deferral: full decryptions never travel in packets
        preview="THE CATS EAT",
        solver_scores={
            "page_validation_avg": val,
            "page_robust_score": val,
            "page_balanced_score": val,
            "adjudication_score": adj,
        },
        validation={
            "accepted": accepted,
            "decision": "runtime_accept" if accepted else "hold_for_review",
            "reasons": ["robust gain clears margin"],
            "deltas": {"page_validation_avg": round(val - 1.0, 6)},
        },
        provenance={
            "edits": edits,
            "mask": [],
            "word_hypotheses": [{"observed": "CXTS", "target": "CATS"}],
            "collateral_evidence": {
                "collateral_occurrences": 2,
                "improved_occurrences": 1,
                "damaged_occurrences": 0,
            },
            "page_runtime_evidence": [],
        },
    )


def _patch_menu(monkeypatch, packets, *, baseline: float = 1.0) -> None:
    alpha = Alphabet(["01", "02", "03"])

    def fake_build(**_kwargs):
        return WordRepairMenu(
            packets=list(packets),
            baseline_validation=baseline,
            page=None,
            alphabet=alpha,
        )

    monkeypatch.setattr(tools_v2.automated_runner, "build_word_repair_menu", fake_build)
    monkeypatch.setattr(tools_v2.automated_runner, "zenith_native_model_path", lambda _lang, variant=None: None)


# ---------------------------------------------------------------------------
# Full flow: search -> review page 2 -> rate -> install
# ---------------------------------------------------------------------------

def test_word_repair_menu_full_flow(monkeypatch):
    ex = _executor()
    _patch_menu(
        monkeypatch,
        [
            _packet(1, ["01:A->X"], 2.5, 4.0, accepted=True),
            _packet(2, ["02:B->Y"], 2.0, 1.5, accepted=True),
        ],
    )

    out = ex._tool_search_word_repair_menu({"branch": "main", "top_n": 8})
    assert out["status"] == "ok"
    # Session id scheme: FinalistSessionStore per-kind counter starts at 1.
    session_id = out["search_session_id"]
    assert session_id == "word_repair_1"
    assert out["candidate_count"] == 2
    assert out["baseline_validation"] == 1.0
    # Compact row shape (spec's required fields).
    row = out["finalist_review"][0]
    for field in (
        "rank",
        "edits",
        "adjudication_score",
        "validation_delta",
        "acceptance",
        "collateral_occurrence_count",
        "decoded_preview",
        "word_context",
        "would_adopt",
    ):
        assert field in row, f"missing compact row field: {field}"
    assert row["edits"] == ["01:A->X"]
    assert row["validation_delta"] == 1.5
    assert row["acceptance"]["accepted"] is True
    assert row["collateral_occurrence_count"] == 2
    assert row["word_context"] == [{"observed": "CXTS", "target": "CATS"}]
    assert len(row["decoded_preview"]) <= 120
    assert out["primary_ranking_signal"] == "agent_contextual_readability"
    assert out["numeric_scores_role"] == "supporting_evidence"
    # would_adopt summary picks the best gate-passing candidate.
    assert out["would_adopt"]["edits"] == ["01:A->X"]

    # Review paging: page 2 shows rank 2 only.
    page2 = ex._tool_search_review_word_repair_finalists(
        {"search_session_id": session_id, "start_rank": 2, "count": 1}
    )
    assert page2["finalist_review_count"] == 1
    assert page2["finalist_review"][0]["rank"] == 2
    assert page2["finalist_review"][0]["edits"] == ["02:B->Y"]
    assert page2["has_more_finalists"] is False

    # Rating flows through the SHARED act_rate_transform_finalist tool.
    rating = ex._tool_act_rate_transform_finalist({
        "search_session_id": session_id,
        "rank": 1,
        "readability_score": 4,
        "label": "coherent_plaintext",
        "rationale": "Reads as THE CATS EAT.",
        "coherent_clause": "the cats eat",
    })
    assert rating["status"] == "ok"
    assert rating["session_type"] == "word_repair"
    assert rating["finalist"]["agent_readability_judgment"]["label"] == "coherent_plaintext"

    # Packet-rating refresh: the stored server-side packet carries the rating.
    session = ex._finalist_sessions.get("word_repair", session_id)
    assert session["packets"][0]["rating"]["label"] == "coherent_plaintext"

    # Install forks the source branch and applies the full edit set.
    installed = ex._tool_act_install_word_repair_finalists({
        "search_session_id": session_id,
        "ranks": [1],
    })
    assert installed["status"] == "ok"
    assert installed["installed_count"] == 1
    assert "main_wr_rank1" in ex.workspace.branch_names()
    branch = ex.workspace.get_branch("main_wr_rank1")
    # 01 (token 0) reassigned A(0) -> X(23); base branch key is unchanged.
    assert branch.key[0] == ord("X") - ord("A")
    assert ex.workspace.get_branch("main").key[0] == 0
    assert "word_repair_finalist" in branch.tags
    assert "wr_rank_1" in branch.tags
    meta = branch.metadata["word_repair_finalist"]
    assert meta["search_session_id"] == session_id
    assert meta["rank"] == 1
    assert meta["edits"] == ["01:A->X"]
    assert meta["adjudication_score"] == 4.0
    assert meta["acceptance"]["accepted"] is True
    assert meta["agent_readability_judgment"]["label"] == "coherent_plaintext"


# ---------------------------------------------------------------------------
# No-leak guard + row-count bound
# ---------------------------------------------------------------------------

def test_word_repair_menu_tools_never_leak_packets(monkeypatch):
    ex = _executor()
    _patch_menu(
        monkeypatch,
        [_packet(rank, [f"0{rank}:A->X"], 2.0 + rank * 0.1, 1.0) for rank in (1, 2, 3)],
    )

    # top_n=2 must return at most 2 rows even though 3 candidates exist.
    out = ex._tool_search_word_repair_menu({"branch": "main", "top_n": 2})
    assert "packet" not in json.dumps(out, default=str)
    assert out["finalist_review_count"] <= 2
    session_id = out["search_session_id"]

    review = ex._tool_search_review_word_repair_finalists(
        {"search_session_id": session_id, "count": 2}
    )
    assert "packet" not in json.dumps(review, default=str)
    assert review["finalist_review_count"] <= 2

    rating = ex._tool_act_rate_transform_finalist({
        "search_session_id": session_id,
        "rank": 1,
        "readability_score": 3,
        "label": "partial_clause",
        "rationale": "Partly readable.",
    })
    assert "packet" not in json.dumps(rating, default=str)

    installed = ex._tool_act_install_word_repair_finalists(
        {"search_session_id": session_id, "ranks": [1]}
    )
    assert "packet" not in json.dumps(installed, default=str)


# ---------------------------------------------------------------------------
# Mode guardrail
# ---------------------------------------------------------------------------

def test_word_repair_menu_blocked_on_periodic_branch(monkeypatch):
    ex = _executor(mode="periodic_polyalphabetic")
    _patch_menu(monkeypatch, [_packet(1, ["01:A->X"], 2.5, 4.0)])

    blocked = ex._tool_search_word_repair_menu({"branch": "main"})
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "mode_mismatch_substitution_repair_blocked"
    assert blocked["attempted_tool"] == "search_word_repair_menu"

    # The override flag lets it through.
    allowed = ex._tool_search_word_repair_menu(
        {"branch": "main", "allow_mode_mismatch_repair": True}
    )
    assert allowed["status"] == "ok"


# ---------------------------------------------------------------------------
# Whole-candidate + masked-symbol install rejection
# ---------------------------------------------------------------------------

def test_word_repair_install_rejects_whole_candidate_on_bad_label(monkeypatch):
    ex = _executor()
    _patch_menu(
        monkeypatch,
        [_packet(1, ["01:A->X", "BOGUS LABEL"], 2.5, 4.0, accepted=True)],
    )
    out = ex._tool_search_word_repair_menu({"branch": "main"})
    session_id = out["search_session_id"]

    installed = ex._tool_act_install_word_repair_finalists(
        {"search_session_id": session_id, "ranks": [1]}
    )
    assert installed["status"] == "error"
    assert installed["installed_count"] == 0
    assert installed["errors"] and "no_applicable_edits" in installed["errors"][0]["reason"]
    # No partial fork was created.
    assert "main_wr_rank1" not in ex.workspace.branch_names()


def test_word_repair_install_rejects_masked_symbol(monkeypatch):
    ex = _executor()
    # The branch carries an active null mask that includes the edited symbol.
    ex.workspace.get_branch("main").metadata["null_mask_selected"] = {"mask": ["01"]}
    _patch_menu(
        monkeypatch,
        [_packet(1, ["01:A->X"], 2.5, 4.0, accepted=True)],
    )
    out = ex._tool_search_word_repair_menu({"branch": "main"})
    assert out["mask"] == ["01"]
    session_id = out["search_session_id"]

    installed = ex._tool_act_install_word_repair_finalists(
        {"search_session_id": session_id, "ranks": [1]}
    )
    assert installed["status"] == "error"
    assert installed["installed_count"] == 0
    assert "no_applicable_edits" in installed["errors"][0]["reason"]
    assert "main_wr_rank1" not in ex.workspace.branch_names()


def test_word_repair_menu_skips_empty_key_branch(monkeypatch):
    ex = _executor()
    ex.workspace.fork("blank", "main")
    ex.workspace.set_full_key("blank", {})
    _patch_menu(monkeypatch, [_packet(1, ["01:A->X"], 2.5, 4.0)])

    out = ex._tool_search_word_repair_menu({"branch": "blank"})
    assert out["status"] == "skipped"
    assert out["reason"] == "empty_base_key"
