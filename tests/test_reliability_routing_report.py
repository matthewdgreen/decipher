from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import report_reliability_routing as report
from reliability_routing_worker import digest
from run_reliability_routing import RESERVATION_SECONDS, write_once


@pytest.fixture
def evidence(tmp_path):
    preparation, campaign = tmp_path / "prep", tmp_path / "run"
    grading = tmp_path / "grading.json"
    case_id = "r000000000001"
    write_once(grading, {"cases": [{"case_id": case_id, "family": "vigenere", "role": "held_out", "plaintext": "THE DOG"}]})
    jobs = []
    for arm in ("blind_family", "family_supplied"):
        request = {"case_id": case_id, "ciphertext": "ABCDEF", "ciphertext_sha256": digest("ABCDEF"),
                   "format": "letters", "language": "en", "arm": arm, "seed": 61001,
                   "cipher_system": "" if arm == "blind_family" else "vigenere"}
        job_id = "j" + digest(request)[:16]
        jobs.append({"job_id": job_id, "request": request, "request_sha256": digest(request)})
        write_once(campaign / "attempts" / (job_id + ".started.json"), {
            "job_id": job_id, "request_sha256": digest(request), "reserved_seconds": RESERVATION_SECONDS})
        text = "THE CAT" if arm == "blind_family" else "THE DOG"
        result = {"request_sha256": digest(request), "status": "completed", "delivered_text": text,
                  "delivered_sha256": digest(text), "artifact": {"decryption": text, "top_candidates": [{"plaintext": "THE DOG"}]},
                  "routes_attempted": [{"route": "homophonic" if arm == "blind_family" else "periodic_polyalphabetic"}],
                  "solver": arm, "delivery_matches_artifact": True}
        write_once(campaign / "results" / (job_id + ".json"), {
            "execution_status": "completed", "request_sha256": digest(request), "result": result,
            "guard_wall_seconds": 1, "wall_seconds": 0.9})
    plan = {"jobs": jobs, "grading_packet_sha256": report.file_digest(grading),
            "provenance": {"solver_revision": "frozen"}}
    write_once(preparation / "control/plan.json", plan)
    write_once(campaign / "campaign.json", {"plan_sha256": report.file_digest(preparation / "control/plan.json")})
    write_once(campaign / "execution_context.json", {"segments": [{"first_attempt": 1, "last_attempt": 2, "context": "test"}]})
    return preparation, campaign, grading


def test_report_keeps_selection_diagnostic_separate_from_delivery(evidence):
    result = report.build_report(*evidence)
    assert result["attempted"] == 2
    pair = result["pairs"][0]
    assert pair["supplied_minus_blind_char"] > 0.02
    assert "family_supplied_material_gain" in pair["observations"]
    assert pair["different_route_or_solver"]
    assert pair["blind"]["saved_minus_delivered_char"] > 0
    assert pair["blind"]["best_generated_quality"] is None
    assert pair["blind"]["best_saved"]["path"].endswith("top_candidates[0].plaintext")
    assert result["primary_synthetic_arm_near_exact"] == {"blind": 0, "family_supplied": 1}
    assert result["primary_synthetic_matched_complete_pairs"] == 1
    assert "THE DOG" not in json.dumps(result)  # post-hoc scores/hashes, no copied reference prose
    assert "Primary synthetic comparisons" in report.markdown(result)


def test_missing_result_is_unknown_and_preserves_pair_denominator(evidence):
    preparation, campaign, grading = evidence
    plan = report.read_json(preparation / "control/plan.json")
    missing = campaign / "results" / (plan["jobs"][0]["job_id"] + ".json")
    missing.unlink()
    result = report.build_report(*evidence)
    assert result["execution_counts"] == {"interrupted": 1, "completed": 1}
    assert result["primary_synthetic_pair_count"] == 1
    pair = result["pairs"][0]
    assert pair["supplied_minus_blind_char"] is None
    assert pair["blind"]["post_hoc_delivered"] is None
    assert result["wall_unknown_attempts"] == 1


def test_changed_grading_packet_cannot_silently_replace_frozen_labels(evidence):
    evidence[2].write_text('{}')
    with pytest.raises(ValueError, match="grading packet changed"):
        report.build_report(*evidence)


def test_historical_and_controls_not_pooled_as_primary_synthetic(evidence):
    result = report.build_report(*evidence)
    pair = result["pairs"][0]
    rows = [dict(pair[arm], role="historical_diagnostic") for arm in ("blind", "family_supplied")]
    assert "provisional_pending_R3" in report.paired_rows(rows)[0]["observations"]
    for row in rows:
        row.update(role="unsupported_control", expected_solve=False)
    observations = report.paired_rows(rows)[0]["observations"]
    assert observations == ["control_no_success_expectation"]


def test_report_never_enters_runtime_imports():
    import ast
    for name in ("run_reliability_routing.py", "reliability_routing_worker.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert "report_reliability_routing" not in imports
        assert "benchmark.scorer" not in imports


def test_cross_context_pair_remains_visible_but_not_a_clean_comparison(evidence):
    path = evidence[1] / "execution_context.json"
    path.write_text(json.dumps({"segments": [
        {"first_attempt": 1, "last_attempt": 1, "context": "sandbox"},
        {"first_attempt": 2, "last_attempt": 2, "context": "host"},
    ]}))
    result = report.build_report(*evidence)
    assert result["primary_synthetic_pair_count"] == 1
    assert result["primary_synthetic_matched_complete_pairs"] == 0
    assert "execution_context_not_matched_or_unrecorded" in result["pairs"][0]["observations"]
