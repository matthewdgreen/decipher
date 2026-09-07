from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_reliability_routing as campaign
from reliability_routing_worker import digest


@pytest.fixture
def prepared(tmp_path):
    preparation = tmp_path / "preparation"
    jobs = []
    for i in range(4):
        request = {"case_id": f"r{i // 2:012x}", "ciphertext": "ABCDEF",
                   "format": "letters", "language": "en", "ciphertext_sha256": digest("ABCDEF"),
                   "arm": "blind_family" if i % 2 == 0 else "family_supplied",
                   "cipher_system": "" if i % 2 == 0 else "vigenere", "seed": 61001}
        job = {"job_id": "j" + digest(request)[:16], "request": request, "request_sha256": digest(request)}
        jobs.append(job)
        campaign.write_once(preparation / "runtime" / (job["job_id"] + ".json"), request)
    plan = {"jobs": jobs, "provenance": {"model_files": {lang: {"path": "/model"} for lang in ("en", "la", "de")}}}
    campaign.write_once(preparation / "control/plan.json", plan)
    return preparation, plan, tmp_path / "campaign"


def start(out, job):
    campaign.write_once(out / "attempts" / (job["job_id"] + ".started.json"), {
        "job_id": job["job_id"], "request_sha256": job["request_sha256"],
        "reserved_seconds": campaign.RESERVATION_SECONDS,
    })


def finish(out, job, *, status="timeout", seconds=180):
    campaign.write_once(out / "results" / (job["job_id"] + ".json"), {
        "request_sha256": job["request_sha256"], "execution_status": status,
        "guard_wall_seconds": seconds, "wall_seconds": seconds, "result": None,
    })


def test_atomic_evidence_is_idempotent_not_overwritten(tmp_path):
    path = tmp_path / "record.json"
    campaign.write_once(path, {"a": 1})
    campaign.write_once(path, {"a": 1})
    with pytest.raises(ValueError, match="replace evidence"):
        campaign.write_once(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}
    assert not list(tmp_path.glob(".pending-*"))


def test_interrupted_and_failed_attempts_spend_budget_and_are_never_retried(prepared):
    _, plan, out = prepared
    a, b, c, _ = plan["jobs"]
    start(out, a)  # controller died before or during launch
    start(out, b)
    finish(out, b, status="worker_error", seconds=0.5)
    ledger = campaign.read_attempts(out, plan)
    assert ledger[a["job_id"]]["charged_seconds"] == 181
    assert ledger[b["job_id"]]["charged_seconds"] == 1.5
    assert campaign.next_job(plan, ledger) == c


def test_cap_exhaustion_does_not_shorten_last_arm(prepared):
    _, plan, _ = prepared
    assert campaign.next_job(plan, {"prior": {"charged_seconds": 10800 - 181}}) == plan["jobs"][0]
    assert campaign.next_job(plan, {"prior": {"charged_seconds": 10800 - 180}}) is None
    assert campaign.next_job(plan, {i: {"charged_seconds": 0} for i in range(60)}) is None


@pytest.mark.parametrize("kind", ["orphan", "wrong_hash", "bad_time"])
def test_corrupt_ledger_fails_closed(prepared, kind):
    _, plan, out = prepared
    job = plan["jobs"][0]
    if kind != "orphan":
        start(out, job)
    altered = dict(job, request_sha256="wrong") if kind == "wrong_hash" else job
    finish(out, altered, seconds=-1 if kind == "bad_time" else 0.2)
    with pytest.raises(ValueError):
        campaign.read_attempts(out, plan)


def test_campaign_is_serial_records_before_launch_and_resumes_without_repeats(prepared, monkeypatch):
    preparation, plan, out = prepared
    monkeypatch.setattr(campaign, "verify_launch_inputs", lambda *args: None)
    launched = []
    active = []

    class FakeGuard:
        def __init__(self, args, **kwargs):
            assert not active
            request_path = Path(args[args.index("--request") + 1])
            job = next(j for j in plan["jobs"] if j["job_id"] == request_path.stem)
            assert (out / "attempts" / (job["job_id"] + ".started.json")).exists()
            assert "OPENAI_API_KEY" not in kwargs["env"]
            assert len(kwargs["pass_fds"]) == 1
            launched.append(job["job_id"])
            active.append(job)

        def wait(self):
            finish(out, active.pop(), status="completed", seconds=0.01)
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", FakeGuard)
    plan_hash = campaign.file_digest(preparation / "control/plan.json")
    with pytest.raises(ValueError, match="authority"):
        campaign.run_campaign(preparation, out, plan_hash)
    with pytest.raises(ValueError, match="plan hash"):
        campaign.run_campaign(preparation, out, "wrong", allow_r3_pending=True)
    first = campaign.run_campaign(preparation, out, plan_hash, allow_r3_pending=True)
    second = campaign.run_campaign(preparation, out, plan_hash, allow_r3_pending=True)
    assert first == second and first["attempted"] == 4
    assert launched == [j["job_id"] for j in plan["jobs"]]
    assert first["unattempted"] == first["missing_results"] == []


def test_guard_inherits_lock_until_it_exits(tmp_path):
    out = tmp_path / "campaign"
    with campaign.campaign_lock(out) as fd:
        guard = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=(fd,))
    try:
        with pytest.raises(ValueError, match="still running"):
            with campaign.campaign_lock(out):
                pass
    finally:
        guard.terminate()
        guard.wait(timeout=5)
    with campaign.campaign_lock(out):
        pass


def test_launch_preflight_is_read_only_and_compares_all_runtime_files(prepared, monkeypatch):
    import prepare_reliability_routing as prep
    preparation, plan, _ = prepared
    provenance = {key: {} for key in ("tracked_input_hashes", "auxiliary_resource_hashes", "native_files",
                                     "python", "executable", "harness_files", "model_files")}
    plan["provenance"] = provenance
    monkeypatch.setattr(campaign, "verify_prepared_inputs", lambda p: None)
    monkeypatch.setattr(prep, "capture_provenance", lambda m: provenance)
    campaign.verify_launch_inputs(plan, preparation)
    path = preparation / "runtime" / (plan["jobs"][0]["job_id"] + ".json")
    path.write_text("{}")
    with pytest.raises(ValueError, match="differs from prepared"):
        campaign.verify_launch_inputs(plan, preparation)
