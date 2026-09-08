from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_reliability_periodic", ROOT / "scripts/run_reliability_periodic.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
import reliability_periodic_worker as worker


def request(arm="prototype"):
    return {"case_id": "r0123456789ab", "ciphertext": "A" * 200,
            "ciphertext_sha256": campaign.digest("A" * 200), "language": "en",
            "format": "letters", "arm": arm, "seed": 61001}


def preflight(now):
    return {"checked_utc": now.isoformat(), "host": socket.gethostname(), "decision": "proceed",
            "cpu_load_observations": [{"utc": (now - timedelta(seconds=3)).isoformat(), "summary": "idle"},
                                      {"utc": now.isoformat(), "summary": "idle"}],
            "active_evaluations": [], "visibility": "complete", "memory_summary": "no pressure",
            "process_summary": "no competing runs", "cleanup_authority_checked": True}


def test_worker_projection_and_identical_environment_except_switch(monkeypatch):
    models = {lang: {"path": f"/{lang}.bin"} for lang in ("en", "la", "de")}
    baseline = worker.worker_environment(request("baseline"), models)
    prototype = worker.worker_environment(request(), models)
    assert baseline.pop("DECIPHER_PERIODIC_ROUTING") == "off"
    assert prototype.pop("DECIPHER_PERIODIC_ROUTING") == "probe_v1"
    assert baseline == prototype
    assert worker.projected_request(request()) == worker.projected_request(request("baseline"))
    assert worker.projected_request(request())["cipher_system"] == ""
    for key in ("ground_truth", "family", "period", "role", "source_file"):
        with pytest.raises(ValueError, match="allowlist"):
            worker.projected_request({**request(), key: "forbidden"})
    with pytest.raises(ValueError, match="environment"):
        worker.solve_request(request())
    monkeypatch.setenv("DECIPHER_PERIODIC_ROUTING", "probe_v1")
    calls = []
    def fake_run(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(steps=[], artifact={"decryption": ""}, status="error",
                               solver="fake", final_decryption="")
    result = worker.solve_request(request(), run_solver=fake_run, intake=lambda *a: "cipher")
    assert result["request_sha256"] == campaign.digest(request())
    assert not any(key in calls[0] for key in ("ground_truth", "arm", "period"))


@pytest.mark.parametrize("change", [
    {"active_evaluations": ["another run"]}, {"host": "different-host"}, {"visibility": "limited"},
    {"decision": "defer"}, {"cleanup_authority_checked": False}, {"memory_summary": ""},
    {"cpu_load_observations": []},
])
def test_machine_preflight_blocks_unsafe_or_incomplete_checks(change):
    now = datetime.now(timezone.utc)
    campaign.validate_preflight(preflight(now), now=now)
    with pytest.raises(ValueError):
        campaign.validate_preflight({**preflight(now), **change}, now=now)
    with pytest.raises(ValueError, match="stale"):
        campaign.validate_preflight(preflight(now - timedelta(minutes=3)), now=now)


def test_fixed_slots_and_budget_no_retry():
    plan = {"jobs": [{"job_id": "a"}, {"job_id": "b"}]}
    attempts = {"a": {"charged_seconds": 181, "execution": {"execution_status": "worker_error", "cleanup_confirmed": True}}}
    assert campaign.next_job(plan, attempts)["job_id"] == "b"
    attempts["a"]["charged_seconds"] = 5700
    assert campaign.next_job(plan, attempts) is None
    attempts["a"]["execution"] = None
    with pytest.raises(ValueError, match="unresolved"):
        campaign.next_job(plan, attempts)
    attempts["a"]["execution"] = {"cleanup_confirmed": False}
    with pytest.raises(ValueError, match="unresolved"):
        campaign.next_job(plan, attempts)


def test_guard_does_not_promote_timeout_result(tmp_path, monkeypatch):
    req_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    campaign.write_once(req_path, request())
    def fake(*a, **kw):
        assert kw["wall_seconds"] == 180 and kw["lock_fd"] == 10
        return {"execution_status": "timeout", "cleanup_confirmed": True, "wall_seconds": 180.,
                "events": [{"event": "result", "result": {"request_sha256": campaign.digest(request())}}]}
    monkeypatch.setattr(campaign, "run_guarded", fake)
    campaign.guard(req_path, result_path, 10)
    result = campaign.read_json(result_path)
    assert result["execution_status"] == "timeout" and result["result"] is None


def test_fake_campaign_durable_start_single_arm_and_resume(tmp_path, monkeypatch):
    req = request()
    job = {"job_id": "j1", "request": req, "request_sha256": campaign.digest(req)}
    plan = {"jobs": [job], "model_files": {}, "protocol": {"max_runs": 32}}
    pin = {"preparation_sha256": campaign.PREPARATION_HASH, "snapshot": {"pinned": True}}
    pin_path = tmp_path / "pin.json"
    campaign.write_once(pin_path, pin)
    monkeypatch.setattr(campaign, "load_preparation", lambda *a: plan)
    monkeypatch.setattr(campaign, "snapshot", lambda *a: {"pinned": True})
    monkeypatch.setattr(campaign, "worker_environment", lambda *a: {})
    out, calls = tmp_path / "campaign", []
    def fake_popen(command, **kwargs):
        start = campaign.read_json(out / "attempts/j1.started.json")
        assert start["request_sha256"] == job["request_sha256"]
        assert start["machine_preflight"]["decision"] == "proceed"
        assert kwargs["pass_fds"]
        calls.append(command)
        campaign.write_once(out / "results/j1.json", {"request_sha256": job["request_sha256"],
            "guard_wall_seconds": .1, "execution_status": "worker_error", "cleanup_confirmed": True})
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(campaign.subprocess, "Popen", fake_popen)
    args = (tmp_path, pin_path, campaign.digest(pin), out, preflight(datetime.now(timezone.utc)))
    result = campaign.run_one(*args)
    assert result["attempt"] == 1 and len(calls) == 1
    result = campaign.run_one(*args)
    assert result["attempted"] == 1 and len(calls) == 1
    assert (out / "summary.json").exists()
    monkeypatch.setattr(campaign, "snapshot", lambda *a: {"pinned": False})
    with pytest.raises(ValueError, match="drift"):
        campaign.run_one(*args)


def test_pin_mismatch_launches_nothing(tmp_path, monkeypatch):
    path = tmp_path / "pin.json"
    campaign.write_once(path, {})
    monkeypatch.setattr(campaign, "load_preparation", lambda *a: {})
    with pytest.raises(ValueError, match="identity"):
        campaign.run_one(tmp_path, path, "wrong", tmp_path / "out", {})
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("failure", ["orphan", "hash", "time", "reservation"])
def test_r5_ledger_rejects_corruption(tmp_path, failure):
    req = request()
    job = {"job_id": "j1", "request_sha256": campaign.digest(req)}
    if failure != "orphan":
        campaign.write_once(tmp_path / "attempts/j1.started.json", {
            "job_id": "j1", "request_sha256": job["request_sha256"],
            "reserved_seconds": 181 if failure == "reservation" else campaign.RESERVATION_SECONDS})
    campaign.write_once(tmp_path / "results/j1.json", {
        "request_sha256": "wrong" if failure == "hash" else job["request_sha256"],
        "guard_wall_seconds": -1 if failure == "time" else .2, "cleanup_confirmed": True})
    with pytest.raises(ValueError):
        campaign.read_attempts(tmp_path, {"jobs": [job]})
