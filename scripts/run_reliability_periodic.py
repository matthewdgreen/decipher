#!/usr/bin/env python3
"""R5 execution pinning and single-arm durable launcher. No grading reads.

Each invocation launches at most one arm and requires a recent machine
preflight. This makes between-arm coordination explicit, not a background
batch that can silently overlap a newly started evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from run_reliability_routing import campaign_lock, read_json, write_once
from prepare_reliability_routing import file_digest
from reliability_routing_worker import digest
from reliability_periodic_worker import projected_request, worker_environment
from automated.bounded_process import run_guarded

PREPARATION_HASH = "4e9a057e479c9ac1b45a36db4488735742ac00fe1ff33f256cd96749a43c75bf"
ALLOWED_SOLVER_CHANGES = {"src/automated/runner.py", "src/automated/periodic_probe.py", "src/automated/bounded_process.py"}
HARNESS = ("run_reliability_periodic.py", "reliability_periodic_worker.py", "run_reliability_routing.py",
           "prepare_reliability_routing.py", "reliability_routing_worker.py")
RESERVATION_SECONDS = 195  # 180-second arm plus bounded nested cleanup/controller allowance


def read_attempts(out, plan):
    """R4 ledger rules with R5's longer nested-cleanup reservation."""
    jobs, attempts = {j["job_id"]: j for j in plan["jobs"]}, {}
    for path in sorted((out / "attempts").glob("*.started.json")):
        start = read_json(path)
        job_id = start["job_id"]
        if (job_id not in jobs or path.name != job_id + ".started.json"
                or start["request_sha256"] != jobs[job_id]["request_sha256"]
                or start["reserved_seconds"] != RESERVATION_SECONDS):
            raise ValueError("invalid R5 attempt ledger")
        result_path = out / "results" / (job_id + ".json")
        execution = read_json(result_path) if result_path.exists() else None
        charge = RESERVATION_SECONDS
        if execution:
            wall = execution["guard_wall_seconds"]
            if (execution["request_sha256"] != start["request_sha256"]
                    or type(wall) not in (float, int) or not 0 <= wall < float("inf")):
                raise ValueError("invalid R5 attempt/result identity or time")
            charge = wall + 1
        attempts[job_id] = {"start": start, "execution": execution, "charged_seconds": charge}
    if any(p.stem not in attempts for p in (out / "results").glob("*.json")):
        raise ValueError("result without a durable start")
    return attempts


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_preparation(directory):
    plan = read_json(directory / "control/plan.json")
    if digest(plan) != PREPARATION_HASH:
        raise ValueError("R5a preparation differs from frozen logical hash")
    if len(plan["jobs"]) != 32:
        raise ValueError("R5 requires exactly 32 scheduled slots")
    for job in plan["jobs"]:
        projected_request(job["request"])
        if (read_json(directory / "runtime" / (job["job_id"] + ".json")) != job["request"]
                or digest(job["request"]) != job["request_sha256"]):
            raise ValueError("R5 runtime request drift")
    return plan


def snapshot(plan):
    scopes = ("src", "rust", "models", "resources", "pyproject.toml")
    if git("diff", "HEAD", "--", *scopes) or git("ls-files", "--others", "--exclude-standard", "--", *scopes):
        raise ValueError("commit solver implementation before freezing/executing R5")
    base = plan["provenance"]["baseline_revision"]
    changed = set(git("diff", "--name-only", base, "--", *scopes).splitlines())
    if not changed <= ALLOWED_SOLVER_CHANGES:
        raise ValueError(f"unrelated solver/resource changes: {sorted(changed - ALLOWED_SOLVER_CHANGES)}")
    native = importlib.util.find_spec("decipher_fast")
    if native is None or native.origin not in plan["provenance"]["native_files"]:
        raise ValueError("native import resolution changed")
    for group in ("auxiliary_resource_hashes", "native_files"):
        for name, expected in plan["provenance"][group].items():
            path = ROOT / name
            if file_digest(path) != expected:
                raise ValueError(f"R5a provenance drift: {name}")
    resources = {str(p.relative_to(ROOT)): file_digest(p) for p in sorted((ROOT / "resources").rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts}
    if resources != plan["provenance"]["auxiliary_resource_hashes"]:
        raise ValueError("auxiliary resource inventory changed")
    if sys.version != plan["provenance"]["python"] or sys.executable != plan["provenance"]["executable"]:
        raise ValueError("interpreter drift")
    for model in plan["model_files"].values():
        if file_digest(Path(model["path"])) != model["sha256"]:
            raise ValueError("model drift")
    files = git("ls-files", "--", *scopes).splitlines()
    return {"baseline_revision": base, "solver_diff": git("diff", base, "--", *scopes),
            "source_hashes": {p: file_digest(ROOT / p) for p in files},
            "harness_hashes": {p: file_digest(ROOT / "scripts" / p) for p in HARNESS},
            "native_files": plan["provenance"]["native_files"], "resources": resources,
            "model_files": plan["model_files"], "python": sys.version, "executable": sys.executable}


def freeze(preparation, output):
    plan = load_preparation(preparation)
    pin = {"schema": "reliability-r5-execution-pin-v1", "preparation_sha256": PREPARATION_HASH,
           "implementation_revision": git("rev-parse", "HEAD"), "snapshot": snapshot(plan)}
    # Snapshot hashes include harness content, whose committed identity is required too.
    harness_paths = ["scripts/" + p for p in HARNESS]
    if (git("diff", "HEAD", "--", *harness_paths)
            or git("ls-files", "--others", "--exclude-standard", "--", *harness_paths)):
        raise ValueError("commit harness before pinning")
    write_once(output, pin)
    return {"execution_pin_sha256": digest(pin), "implementation_revision": pin["implementation_revision"]}


def validate_preflight(value, *, now=None):
    now = now or datetime.now(timezone.utc)
    observed = datetime.fromisoformat(value["checked_utc"])
    if observed.tzinfo is None or not 0 <= (now - observed).total_seconds() <= 120:
        raise ValueError("machine preflight is stale or lacks timezone")
    samples = value.get("cpu_load_observations", [])
    if len(samples) < 2 or not all(isinstance(s, dict) and s.get("summary") for s in samples):
        raise ValueError("two timestamped CPU/load observations required")
    times = [datetime.fromisoformat(s["utc"]) for s in samples]
    if any(t.tzinfo is None for t in times) or (times[-1] - times[0]).total_seconds() < 1:
        raise ValueError("CPU/load observations must be separated in time")
    if any(not 0 <= (observed - t).total_seconds() <= 120 for t in times):
        raise ValueError("CPU/load observations are not recent")
    if (value.get("host") != socket.gethostname() or value.get("decision") != "proceed"
            or value.get("active_evaluations") != [] or value.get("visibility") != "complete"
            or not value.get("memory_summary") or not value.get("process_summary")
            or value.get("cleanup_authority_checked") is not True):
        raise ValueError("preflight must confirm host visibility, no active evaluations and cleanup authority")


def next_job(plan, attempts):
    if any(a["execution"] is None or not a["execution"].get("cleanup_confirmed") for a in attempts.values()):
        raise ValueError("unresolved prior cleanup/result; reconcile before further launches (never retry)")
    if len(attempts) >= 32 or sum(a["charged_seconds"] for a in attempts.values()) + RESERVATION_SECONDS > 5760:
        return None
    return next((j for j in plan["jobs"] if j["job_id"] not in attempts), None)


def guard(request_path, result_path, lock_fd):
    started = time.monotonic()
    request = read_json(request_path)
    projected_request(request)
    with tempfile.TemporaryDirectory(prefix="decipher-r5-arm-") as directory:
        execution = run_guarded([sys.executable, str(ROOT / "scripts/reliability_periodic_worker.py")],
            request, env=dict(os.environ), cwd=directory, wall_seconds=180, lock_fd=lock_fd)
    events = execution.get("events", [])
    finals = [e for e in events if e.get("event") == "result"]
    valid = (len(finals) == 1 and events[-1] == finals[0] and isinstance(finals[0].get("result"), dict)
             and finals[0]["result"].get("request_sha256") == digest(request))
    if execution["execution_status"] == "completed" and not valid:
        execution["execution_status"] = "worker_error"
    execution.update(schema="reliability-r5-execution-v1", request_sha256=digest(request),
        guard_wall_seconds=time.monotonic() - started,
        result=finals[0]["result"] if execution["execution_status"] == "completed" else None,
        cpu_usage=finals[0].get("cpu_usage") if valid else None)
    write_once(result_path, execution)


def run_one(preparation, pin_path, expected_pin_hash, out, preflight):
    plan, pin = load_preparation(preparation), read_json(pin_path)
    if digest(pin) != expected_pin_hash or pin["preparation_sha256"] != PREPARATION_HASH:
        raise ValueError("execution pin identity mismatch")
    with campaign_lock(out) as lock_fd:
        if snapshot(plan) != pin["snapshot"]:
            raise ValueError("execution provenance drift")
        attempts = read_attempts(out, plan)
        job = next_job(plan, attempts)
        if job is None:
            summary = {"attempted": len(attempts), "planned": 32,
                       "charged_seconds": sum(a["charged_seconds"] for a in attempts.values()),
                       "unattempted": [j["job_id"] for j in plan["jobs"] if j["job_id"] not in attempts]}
            write_once(out / "summary.json", summary)
            return summary
        validate_preflight(preflight)  # Immediately before the durable launch.
        write_once(out / "campaign.json", {"schema": "reliability-r5-campaign-v1",
            "execution_pin_sha256": expected_pin_hash, "preparation_sha256": PREPARATION_HASH,
            "protocol": plan["protocol"], "provider_calls": "none"})
        write_once(out / "attempts" / (job["job_id"] + ".started.json"), {
            "job_id": job["job_id"], "request_sha256": job["request_sha256"],
            "reserved_seconds": RESERVATION_SECONDS, "started_utc": datetime.now(timezone.utc).isoformat(),
            "machine_preflight": preflight})
        result_path = out / "results" / (job["job_id"] + ".json")
        with (out / "guard.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "guard",
                "--request", str(preparation / "runtime" / (job["job_id"] + ".json")),
                "--result", str(result_path), "--lock-fd", str(lock_fd)],
                env=worker_environment(job["request"], plan["model_files"]), cwd=ROOT,
                stdout=log, stderr=log, start_new_session=True, pass_fds=(lock_fd,))
            process.wait()
        return {"job_id": job["job_id"], "attempt": len(attempts) + 1,
                "execution_status": read_json(result_path)["execution_status"] if result_path.exists() else "interrupted",
                "next": "fresh machine check before the next run-one invocation"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest="command", required=True)
    freeze_ap = subs.add_parser("freeze")
    freeze_ap.add_argument("--preparation", type=Path, required=True)
    freeze_ap.add_argument("--out", type=Path, required=True)
    run = subs.add_parser("run-one")
    for name in ("preparation", "pin", "out", "preflight"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--expected-pin-sha256", required=True)
    g = subs.add_parser("guard", help=argparse.SUPPRESS)
    g.add_argument("--request", type=Path, required=True)
    g.add_argument("--result", type=Path, required=True)
    g.add_argument("--lock-fd", type=int, required=True)
    args = ap.parse_args()
    if args.command == "freeze":
        result = freeze(args.preparation.resolve(), args.out.resolve())
    elif args.command == "guard":
        guard(args.request, args.result, args.lock_fd)
        return
    else:
        result = run_one(args.preparation.resolve(), args.pin, args.expected_pin_sha256,
                         args.out.resolve(), read_json(args.preflight))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
