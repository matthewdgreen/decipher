"""R2 fixed-candidate lifecycle and preference tests; no solver/provider calls."""
import copy
import json
from types import SimpleNamespace

import pytest

from agent.tools_v2 import AttestationPolicy
from investigation import portfolio
from investigation.candidates import candidate_packet_for_branch
from investigation.context import _render_window
from investigation.loop_v3 import _select_v3_fallback
from investigation.state import InvestigationState
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace import Workspace, WorkspaceError


def fixture():
    ws = Workspace(CipherText(raw="AB | CD | AB", alphabet=Alphabet(list("ABCD")), separator=" | "))
    state = InvestigationState(workspace=ws, language="en")
    ex = SimpleNamespace(workspace=ws, word_set={"THE", "CAT", "DOG"}, _freq_rank={},
                         _compute_quick_scores=lambda _: {"dict_rate": 0.5, "quad": -5})
    return state, ex


def add(state, name, text, family="substitution"):
    branch = state.workspace.fork(name)
    branch.metadata.update(decoded_text=text, cipher_mode=family)
    return branch


def comparison(state, best="partial", *, accept=False, turn=1):
    hashes = {name: portfolio.content_hash(state, name) for name in ("partial", "other")}
    return {"kind": "compare", "status": "ok", "turn": turn,
            "result": {"best_candidate": best, "accepts_as_solution": accept,
                       "ranking": ["partial", "other"], "rationale": "prefer partial",
                       "verdicts": [{"branch": b, "verdict": "partial"} for b in hashes]},
            "comparison_binding": {"branch_hashes": hashes, "best_candidate": best,
                                   "best_candidate_hash": hashes.get(best), "accepts_as_solution": accept}}


def test_best_partial_survives_null_comparison_but_not_stale_shortlist():
    state, ex = fixture()
    add(state, "partial", "THE CAT")
    add(state, "other", "ZZZ ZZZ")
    state.episode_ledger = [comparison(state), comparison(state, None, turn=2)]
    name, selected = _select_v3_fallback(state, ex)
    assert name == "partial" and selected["tier"] == "fresh_compare_best_partial"
    assert selected["comparison_binding"]["accepts_as_solution"] is False
    state.workspace.get_branch("other").metadata["decoded_text"] = "CHANGED"
    assert portfolio.fresh_compare_best_candidate(state) is None


@pytest.mark.parametrize("invalid", ["rejected", "short_ranking", "missing_verdict"])
def test_invalid_worker_comparison_never_becomes_preference(invalid):
    state, _ = fixture()
    add(state, "partial", "THE CAT")
    add(state, "other", "THE DOG")
    entry = comparison(state)
    if invalid == "rejected":
        entry["result"]["verdicts"][0]["verdict"] = "rejected"
    elif invalid == "short_ranking":
        entry["result"]["ranking"] = ["partial"]
    else:
        entry["result"]["verdicts"] = []
    state.episode_ledger = [entry]
    assert portfolio.fresh_compare_best_candidate(state) is None


def test_fluent_preference_and_retention_do_not_authorize_declaration():
    state, ex = fixture()
    add(state, "partial", "THE CAT")
    add(state, "other", "THE DOG")
    state.episode_ledger = [comparison(state, accept=True)]
    assert portfolio.fresh_compare_best_candidate(state)[0] == "partial"
    portfolio.refresh_portfolio(state, ex)
    gate = AttestationPolicy(state.verify_attestations)
    assert gate.check_declare_solution(ex, {"branch": "partial"})["reason"] == "attestation_required"
    state.verify_attestations.append({"branch": "partial", "content_hash": portfolio.content_hash(state, "partial"),
                                      "reader_accepts_as_solution": True})
    assert gate.check_declare_solution(ex, {"branch": "partial"}) is None
    state.workspace.get_branch("partial").metadata["decoded_text"] = "THE RAT"
    assert gate.check_declare_solution(ex, {"branch": "partial"}) is not None


def test_portfolio_keeps_third_ranked_distinct_family_and_baseline(monkeypatch):
    state, ex = fixture()
    for i in range(9):
        add(state, f"candidate_{i}", f"TEXT {chr(65+i)}", "substitution")
    null = state.workspace.get_branch("candidate_2")
    null.metadata["null_mask_selected"] = {"mask": ["B"]}
    baseline = add(state, "automated_preflight", "BASELINE", "baseline")
    baseline.tags.append("automated_preflight")
    add(state, "duplicate", "TEXT C", "substitution").metadata["null_mask_selected"] = {"mask": ["B"]}
    def scores(ws, name, *args):
        return (0.9 - int(name[-1]) * 0.01, -5) if name.startswith("candidate_") else (0.01, -9)
    monkeypatch.setattr(portfolio, "_score_branch_for_panel", scores)
    rows = portfolio.refresh_portfolio(state, ex)
    assert len(rows) == 6 and len({r["content_hash"] for r in rows}) == 6
    by_name = {row["branch"]: row for row in rows}
    assert "candidate_2" in by_name and "family_finalist" in by_name["candidate_2"]["roles"]
    assert "automated_preflight" in by_name
    assert "duplicate" in by_name["candidate_2"]["aliases"]
    assert portfolio.candidate_portfolio(state, ex) == rows
    with pytest.raises(WorkspaceError, match="retained"):
        state.workspace.delete("candidate_2")
    restored = InvestigationState.from_artifact_dict(json.loads(json.dumps(state.to_artifact_dict())))
    assert restored.candidate_portfolio == rows
    with pytest.raises(WorkspaceError, match="retained"):
        restored.workspace.delete("candidate_2")
    state.workspace.get_branch("candidate_2").metadata["mode_status"] = "rejected"
    portfolio.refresh_portfolio(state, ex)
    state.workspace.delete("candidate_2")
    assert not state.workspace.has_branch("candidate_2")


@pytest.mark.parametrize("kind", ["substitution", "null_mask", "custom_boundaries", "periodic", "transform"])
def test_fixed_candidate_preserves_key_render_and_structure_on_resume(kind):
    state, ex = fixture()
    branch = state.workspace.fork("candidate")
    branch.key = dict(enumerate(range(4)))
    if kind == "null_mask":
        branch.metadata["null_mask_selected"] = {"mask": ["B"]}
    if kind == "custom_boundaries":
        branch.word_spans = [(0, 3), (3, 6)]
    if kind == "periodic":
        branch.metadata.update(decoded_text="THECAT", key_type="QuagmireKey", cipher_mode="quagmire3",
                               alphabet_keyword="EXAMPLE", cycleword="KEY")
    if kind == "transform":
        branch.token_order = [5, 4, 3, 2, 1, 0]
        branch.transform_pipeline = {"operations": []}
    before = copy.deepcopy(candidate_packet_for_branch(state.workspace, "candidate").to_dict())
    portfolio.refresh_portfolio(state, ex)
    restored = InvestigationState.from_artifact_dict(json.loads(json.dumps(state.to_artifact_dict())))
    assert candidate_packet_for_branch(restored.workspace, "candidate").to_dict() == before


def test_window_and_key_inspection_use_actual_metadata_candidate(monkeypatch):
    from investigation import context
    from investigation_service.service import InvestigationService
    state, ex = fixture()
    branch = add(state, "periodic", "THECAT", "quagmire3")
    branch.metadata.update(alphabet_keyword="EXAMPLE", cycleword="KEY", key_type="QuagmireKey")
    monkeypatch.setattr(context, "_best_branch_for_auto_declare", lambda *a: ("periodic", {}))
    assert "THECAT" in _render_window(state, ex, 0, 400)
    assert "???" not in _render_window(state, ex, 0, 400)
    key = InvestigationService._candidate_key_state(state.workspace, "periodic")
    assert key["mode_key_state"]["alphabet_keyword"] == "EXAMPLE"
    assert key["mode_key_state"]["cycleword"] == "KEY"
    assert key["content_hash"] == portfolio.content_hash(state, "periodic")


def test_corrected_null_render_has_verification_debt_not_old_rejection():
    state, ex = fixture()
    b = state.workspace.fork("masked")
    b.key = dict(enumerate(range(4)))
    b.metadata["null_mask_selected"] = {"mask": ["B"]}
    state.verify_attestations.append({"content_hash": portfolio._candidate_content_hash("ACDA"),
                                      "reader_accepts_as_solution": False})
    row = portfolio.refresh_portfolio(state, ex)[0]
    assert row["verification"] == "missing" and row["verification_priority"] is True


def test_client_comparison_and_explicit_automated_baseline_survive_resume():
    from investigation.context import workflow_state
    state, ex = fixture()
    add(state, "partial", "THE CAT")
    add(state, "other", "THE DOG")
    state.experiment_queue = [{"type": "automated_solver", "status": "completed", "installed_as": "other"}]
    hashes = {name: portfolio.content_hash(state, name) for name in ("partial", "other")}
    state.comparison_records = [{"best_partial": "partial", "best_partial_hash": hashes["partial"],
                                 "branch_hashes": hashes, "accepts_as_solution": False, "created_turn": 4}]
    state.repair_transactions = [{"status": "installed", "installed_branch": "other",
                                   "result_content_hash": hashes["other"]}]
    assert workflow_state(state, ex)["branch"] == "other"
    rows = portfolio.refresh_portfolio(state, ex)
    assert rows[0]["branch"] == "other" and "automated_baseline" in rows[0]["roles"]
    restored = InvestigationState.from_artifact_dict(json.loads(json.dumps(state.to_artifact_dict())))
    assert workflow_state(restored, ex)["branch"] == "other"
    assert portfolio.fresh_compare_best_candidate(restored)[0] == "partial"
    assert portfolio.candidate_portfolio(restored, ex) == rows
