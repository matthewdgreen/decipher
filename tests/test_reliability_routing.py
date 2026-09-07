"""R4 preparation tests: fake workers only; never run the frozen solver packet."""
from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_reliability_routing as prep
import reliability_routing_worker as worker


@pytest.fixture
def packet():
    runtime, labels = [], []
    families = sorted(worker.FAMILIES - {"bifid"})
    membership = [(family, role) for family in families for role in ("development", "held_out")]
    membership += [("simple_substitution", "historical_diagnostic")] * 4
    membership += [("bifid", "unsupported_control"), (None, "random_control")]
    for index, (family, role) in enumerate(membership):
        text = "ABC DEF " + "G" * (index + 1)
        case_id = f"r{index:012x}"
        runtime.append(dict(case_id=case_id, ciphertext=text, format="letters", language="en",
                            ciphertext_sha256=worker.digest(text)))
        labels.append(dict(case_id=case_id, family=family, role=role,
                           plaintext="SECRET LABEL NEVER SENT TO SOLVER", key={"secret": 42}))
    return runtime, {"packet_version": "reliability-r0-v1", "cases": labels,
                     "protocol": copy.deepcopy(prep.PROTOCOL), "unavailable": [],
                     "runtime_manifest_sha256": worker.digest(runtime)}


@pytest.fixture
def job_request(packet):
    return prep.prepare_schedule(*packet)["jobs"][0]["request"]


def test_schedule_is_frozen_paired_and_runtime_safe(packet):
    plan = prep.prepare_schedule(*packet)
    assert plan == prep.prepare_schedule(*packet)
    assert plan["planned_runs"] == 60 and plan["available_cases"] == 18
    assert plan["missing_roles"] == {} and not plan["execution_authorized"]
    counts = Counter(job["role"] for job in plan["jobs"])
    assert counts == {"development": 36, "held_out": 12, "historical_diagnostic": 8,
                      "unsupported_control": 2, "random_control": 2}
    for first, second in zip(plan["jobs"][::2], plan["jobs"][1::2]):
        a, b = first["request"], second["request"]
        assert set(a) == set(b) == worker.REQUEST_FIELDS
        assert {a["arm"], b["arm"]} == set(worker.ARMS)
        assert {k: v for k, v in a.items() if k not in {"arm", "cipher_system"}} == {
            k: v for k, v in b.items() if k not in {"arm", "cipher_system"}}
        blind = a if a["arm"] == "blind_family" else b
        assert blind["cipher_system"] == ""
        if first["role"] != "development":
            assert a["seed"] == 61001
    serialized = json.dumps(plan)
    assert "SECRET LABEL" not in serialized and '"key"' not in serialized
    # Grading outcomes cannot steer the schedule or metadata projection.
    packet[1]["cases"][0].update(plaintext="DIFFERENT", key={"other": 1}, char_accuracy=1.0)
    assert prep.prepare_schedule(*packet) == plan


@pytest.mark.parametrize("field,value", [
    ("plaintext", "LEAK"), ("ground_truth", "LEAK"), ("key", {"a": "b"}),
    ("solver_hints", {"known_params": "LEAK"}), ("role", "held_out"),
    ("ciphertext_sha256", "wrong"), ("case_id", "borg_0109v"), ("seed", 61004),
    ("seed", True), ("language", "xx"), ("arm", "other"), ("format", "unknown"),
])
def test_worker_rejects_extra_or_invalid_fields(job_request, field, value):
    with pytest.raises(ValueError):
        worker.validate_request(dict(job_request, **{field: value}))


def test_blind_family_is_rejected_not_silently_removed(job_request):
    with pytest.raises(ValueError, match="blind arm"):
        worker.validate_request(dict(job_request, arm="blind_family", cipher_system="quagmire3"))


def test_missing_cases_are_explicit_not_passing(packet):
    runtime, labels = packet
    missing = labels["cases"].pop(12)
    runtime[:] = [r for r in runtime if r["case_id"] != missing["case_id"]]
    labels["runtime_manifest_sha256"] = worker.digest(runtime)
    labels["unavailable"] = [{"case_id": missing["case_id"], "reason": "missing_source"}]
    plan = prep.prepare_schedule(runtime, labels)
    assert plan["planned_runs"] == 58
    assert plan["missing_roles"] == {"historical_diagnostic": 1}
    assert plan["unavailable_cases"] == labels["unavailable"]


@pytest.mark.parametrize("change", ["protocol", "hash", "duplicate", "extra", "role", "family"])
def test_preparation_fails_closed_on_packet_drift(packet, change):
    runtime, labels = packet
    if change == "protocol":
        labels["protocol"]["max_runs"] = 61
    elif change == "hash":
        runtime[0]["ciphertext"] += "A"
    elif change == "duplicate":
        labels["cases"][1]["case_id"] = labels["cases"][0]["case_id"]
    elif change == "extra":
        runtime[0]["source"] = "family_bearing_name"
        labels["runtime_manifest_sha256"] = worker.digest(runtime)
    elif change == "role":
        labels["cases"][0]["role"] = "held_out"
    else:
        labels["cases"][0]["family"] = "vigenere"
    with pytest.raises(ValueError):
        prep.prepare_schedule(runtime, labels)


def test_prepared_files_are_immutable_and_requests_do_not_contain_roles(tmp_path, packet):
    plan = prep.prepare_schedule(*packet)
    prep.write_preparation(tmp_path, plan)
    prep.write_preparation(tmp_path, plan)
    assert len(list((tmp_path / "runtime").glob("*.json"))) == 60
    for path in (tmp_path / "runtime").glob("*.json"):
        worker.validate_request(json.loads(path.read_text()))
    before = (tmp_path / "control/plan.json").read_bytes()
    plan["jobs"][0]["request"]["ciphertext"] = "changed"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        prep.write_preparation(tmp_path, plan)
    assert (tmp_path / "control/plan.json").read_bytes() == before


def test_environment_drops_credentials_and_unfrozen_solver_settings(job_request):
    models = {lang: {"path": "/frozen/model-" + lang} for lang in ("en", "la", "de")}
    hostile = {"PATH": "/usr/bin", "OPENAI_API_KEY": "secret", "ANTHROPIC_API_KEY": "secret",
               "PYTHONPATH": "/injection", "DECIPHER_KEYED_VIGENERE_MODE": "replay",
               "DECIPHER_QUAGMIRE_INITIAL_KEYWORDS": "ANSWER", "DECIPHER_PARALLEL_WORKERS": "99"}
    env = worker.worker_environment(job_request, models, inherited=hostile)
    assert env["PATH"] == "/usr/bin" and env["PYTHONPATH"] == str(ROOT / "src")
    assert not any("API_KEY" in k for k in env)
    assert "DECIPHER_KEYED_VIGENERE_MODE" not in env
    assert "DECIPHER_QUAGMIRE_INITIAL_KEYWORDS" not in env
    assert env["DECIPHER_PARALLEL_WORKERS"] == "4"
    assert env["DECIPHER_QUAGMIRE_SEARCH_SEED"] == str(job_request["seed"])
    assert env["DECIPHER_NGRAM_MODEL_DE"] == "/frozen/model-de"


def test_adapter_calls_existing_api_and_preserves_delivery_evidence(job_request):
    from automated.runner import AutomatedRunResult, _StepList

    events = []

    def fake_solver(**kwargs):
        assert kwargs == dict(worker.solver_kwargs(job_request), cipher_text="parsed", on_step=kwargs["on_step"])
        assert "ground_truth" not in kwargs and "solver_hints" not in kwargs
        steps = _StepList(kwargs["on_step"])
        steps.append({"name": "route_automated_solver", "route": "test"})
        return AutomatedRunResult(
            test_id=job_request["case_id"], status="completed", final_decryption="ABC", elapsed_seconds=0,
            solver="fake", steps=steps, artifact={"decryption": "ABC", "steps": list(steps),
                                                "ground_truth": None, "char_accuracy": 0, "word_accuracy": 0})

    result = worker.solve_request(job_request, run_solver=fake_solver, intake=lambda *args: "parsed", emit=events.append)
    assert events[0]["name"] == "route_automated_solver"
    assert result["routes_attempted"][0]["route"] == "test"
    assert result["delivery_matches_artifact"]
    assert "ground_truth" not in result["artifact"] and "char_accuracy" not in result["artifact"]
    assert result["seed_evidence"]["independent_replication_established"] is False
    assert worker.digest(json.loads(json.dumps(result))["delivered_text"]) == result["delivered_sha256"]


def test_seed_echo_is_not_an_independence_claim():
    steps = [{"name": "search_quagmire3_keyword_alphabet", "seed": 61001}]
    assert worker.seed_evidence(steps, 61001)["quagmire_seed_echo_matches"]
    assert not worker.seed_evidence(steps, 61002)["quagmire_seed_echo_matches"]
    assert not worker.seed_evidence(steps, 61001)["independent_replication_established"]
    assert worker.seed_evidence([], 61001)["quagmire_seed_echo_matches"] is None


@pytest.mark.parametrize("count,elapsed,allowed", [
    (0, 0, True), (59, 10620, True), (60, 0, False), (1, 10620.01, False),
    (-1, 0, False), (1, -1, False),
])
def test_campaign_limits_reserve_full_arms(count, elapsed, allowed):
    assert worker.launch_allowed(count, elapsed) is allowed


def _fake_env():
    # Subprocesses import the worker but replace the solver before worker_main.
    return {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "scripts")}


def test_real_process_cpu_limit_and_result_transport_with_fake_solver(job_request, tmp_path):
    code = """
import reliability_routing_worker as w
def fake(job_request, **kwargs):
    return {"request_sha256": w.digest(job_request), "status": "completed"}
w.solve_request = fake
w.worker_main()
"""
    output = worker.run_bounded([sys.executable, "-c", code], job_request, env=_fake_env(), cwd=tmp_path, wall_seconds=5)
    assert output["execution_status"] == "completed"
    assert output["cpu_usage"]["cpu_limit_per_process"] == [720, 720]
    assert output["cpu_usage"]["self_user_seconds"] >= 0


@pytest.mark.parametrize("mode", ["bad_json", "nonzero", "wrong_hash", "duplicate", "missing"])
def test_fake_worker_errors_cannot_be_completion(job_request, tmp_path, mode):
    code = f"""
import json, sys
import reliability_routing_worker as w
r = json.load(sys.stdin)
mode = {mode!r}
event = {{"event": "result", "result": {{"request_sha256": w.digest(r)}}}}
if mode == "bad_json": print("not json")
if mode == "wrong_hash": event["result"]["request_sha256"] = "wrong"
if mode != "missing": print(json.dumps(event))
if mode == "duplicate": print(json.dumps(event))
if mode == "nonzero": sys.exit(2)
"""
    output = worker.run_bounded([sys.executable, "-c", code], job_request, env=_fake_env(), cwd=tmp_path, wall_seconds=5)
    assert output["execution_status"] == "worker_error" and output["result"] is None


def test_timeout_preserves_progress_but_rejects_early_final_and_kills_descendants(job_request, tmp_path):
    marker = tmp_path / "escaped_child"
    code = f"""
import json, subprocess, sys, time
import reliability_routing_worker as w
r = json.load(sys.stdin)
subprocess.Popen([sys.executable, "-c", {('import time; from pathlib import Path; time.sleep(1); Path(' + repr(str(marker)) + ').touch()')!r}])
print(json.dumps({{"event": "progress", "name": "started"}}), flush=True)
print(json.dumps({{"event": "result", "result": {{"request_sha256": w.digest(r)}}}}), flush=True)
time.sleep(30)
"""
    output = worker.run_bounded([sys.executable, "-c", code], job_request, env=_fake_env(), cwd=tmp_path, wall_seconds=0.4)
    assert output["execution_status"] == "timeout" and output["result"] is None
    assert output["events"][0]["event"] == "progress"
    assert output["cpu_usage"] is None
    # Wait beyond the child's scheduled write to prove it did not survive.
    subprocess.run([sys.executable, "-c", "import time; time.sleep(1)"], check=True)
    assert not marker.exists()


def test_grading_only_joins_completed_hash_bound_candidates(job_request):
    label = dict(case_id=job_request["case_id"], plaintext="ABC", role="held_out")
    candidate = dict(request_sha256=worker.digest(job_request), status="completed",
                     delivered_text="ABC", delivered_sha256=worker.digest("ABC"))
    execution = dict(request_sha256=worker.digest(job_request), execution_status="completed", result=candidate)
    scored = prep.grade_completed(job_request, execution, label)
    assert scored["post_hoc_delivered"]["char_accuracy"] == 1.0
    assert scored["best_generated_quality"] is None
    label["role"] = "unsupported_control"
    assert not prep.grade_completed(job_request, execution, label)["expected_solve"]
    execution.update(execution_status="timeout", result=None)
    assert prep.grade_completed(job_request, execution, label)["post_hoc_delivered"] is None
    execution.update(execution_status="completed", result=candidate)
    candidate["delivered_text"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        prep.grade_completed(job_request, execution, label)


def test_prepare_has_no_solver_or_campaign_entry_point():
    import ast

    tree = ast.parse(Path(prep.__file__).read_text())
    names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not names & {"run_automated", "solve_request", "run_bounded", "build_packet"}
    worker_tree = ast.parse(Path(worker.__file__).read_text())
    imported = {n.module for n in ast.walk(worker_tree) if isinstance(n, ast.ImportFrom)}
    assert not imported & {"benchmark.loader", "benchmark.scorer", "prepare_reliability_routing", "build_reliability_packet"}


def test_prelaunch_rechecks_frozen_files_and_requests(tmp_path, packet, monkeypatch):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    source = tmp_path / "source.py"
    source.write_text("original")
    plan = prep.prepare_schedule(*packet)
    plan["provenance"] = dict(solver_revision=prep.BASELINE_REVISION,
                              tracked_input_hashes={"source.py": prep.file_digest(source)},
                              auxiliary_resource_hashes={}, native_files={}, model_files={}, harness_files={})
    prep.verify_prepared_inputs(plan)
    source.write_text("changed")
    with pytest.raises(ValueError, match="input absent or changed"):
        prep.verify_prepared_inputs(plan)
    source.write_text("original")
    plan["jobs"][0]["request"]["seed"] = 61003
    with pytest.raises(ValueError, match="prepared request changed"):
        prep.verify_prepared_inputs(plan)


@pytest.mark.parametrize("changed", ["tracked", "untracked", "model", "native"])
def test_provenance_rejects_solver_model_and_native_drift(tmp_path, monkeypatch, changed):
    def fake_git(args, **kwargs):
        if args[1] == "diff":
            return "changed" if changed == "tracked" else ""
        if args[1:3] == ["ls-files", "--others"]:
            return "extra.py" if changed == "untracked" else ""
        return ""

    monkeypatch.setattr(prep.subprocess, "check_output", fake_git)
    models = {}
    for language in ("en", "la", "de"):
        path = tmp_path / language
        path.write_text("model")
        models[language] = {"path": str(path), "sha256": prep.file_digest(path)}
    if changed == "model":
        Path(models["en"]["path"]).write_text("drift")
    monkeypatch.setattr(prep.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match={
        "tracked": "differ", "untracked": "untracked", "model": "model is absent or changed",
        "native": "native module is absent",
    }[changed]):
        prep.capture_provenance(models)
