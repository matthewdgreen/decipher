#!/usr/bin/env python3
"""Serial, resumable R4 campaign controller. No grading inputs or paid calls.

An immutable start record spends an attempt before launching it. Failed or
interrupted attempts are never rerun. A separate guard retains the campaign
lock and enforces worker timeout even if the controller is killed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_reliability_routing import encode, file_digest, verify_prepared_inputs
from reliability_routing_worker import (
    MAX_RUNS, MAX_WALL_SECONDS, WALL_SECONDS, digest, run_bounded,
    validate_request, worker_environment,
)

RESERVATION_SECONDS = WALL_SECONDS + 1  # conservative per-attempt controller allowance


def write_once(path, value):
    """Atomic publication, never replace an existing evidence record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = encode(value)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"refusing to replace evidence: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic and refuses concurrent replacement
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def campaign_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "campaign.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("campaign controller or guard is still running") from exc
        # A guard inherits this open-file description, keeping the lock even
        # after controller SIGKILL. Closing the parent copy must not LOCK_UN it.
        yield stream.fileno()


def read_json(path):
    return json.loads(path.read_text())


def read_attempts(out, plan):
    jobs = {j["job_id"]: j for j in plan["jobs"]}
    attempts = {}
    for path in sorted((out / "attempts").glob("*.started.json")):
        record = read_json(path)
        job_id = record["job_id"]
        if (job_id not in jobs or path.name != job_id + ".started.json"
                or record["request_sha256"] != jobs[job_id]["request_sha256"]
                or record["reserved_seconds"] != RESERVATION_SECONDS):
            raise ValueError("invalid attempt ledger")
        execution_path = out / "results" / (job_id + ".json")
        execution = read_json(execution_path) if execution_path.exists() else None
        if execution and execution["request_sha256"] != record["request_sha256"]:
            raise ValueError("attempt/result identity mismatch")
        charge = RESERVATION_SECONDS
        if execution:
            wall = execution["guard_wall_seconds"]
            if not isinstance(wall, (float, int)) or not 0 <= wall < float("inf"):
                raise ValueError("invalid recorded time")
            charge = wall + 1
        attempts[job_id] = {"start": record, "execution": execution, "charged_seconds": charge}
    # An orphan result is not permission to skip or retry a job.
    if any(p.stem not in attempts for p in (out / "results").glob("*.json")):
        raise ValueError("result without a durable start record")
    return attempts


def next_job(plan, attempts):
    charge = sum(a["charged_seconds"] for a in attempts.values())
    if len(attempts) >= MAX_RUNS or charge + RESERVATION_SECONDS > MAX_WALL_SECONDS:
        return None
    return next((j for j in plan["jobs"] if j["job_id"] not in attempts), None)


def guard_main(request_path, result_path):
    started = time.monotonic()
    request = read_json(request_path)
    validate_request(request)
    with tempfile.TemporaryDirectory(prefix="decipher-r4-") as workdir:
        execution = run_bounded(
            [sys.executable, str(ROOT / "scripts/reliability_routing_worker.py")],
            request, env=dict(os.environ), cwd=workdir,
        )
    execution["guard_wall_seconds"] = time.monotonic() - started
    write_once(result_path, execution)


def verify_launch_inputs(plan, preparation):
    verify_prepared_inputs(plan)
    # Also detect new untracked source and auxiliary files, native module
    # replacement, and interpreter drift, not just changes to known paths.
    from prepare_reliability_routing import capture_provenance
    current = capture_provenance(plan["provenance"]["model_files"])
    for field in ("tracked_input_hashes", "auxiliary_resource_hashes", "native_files",
                  "python", "executable", "model_files", "harness_files"):
        if current[field] != plan["provenance"][field]:
            raise ValueError(f"prepared provenance drift: {field}")
    for job in plan["jobs"]:
        request = read_json(preparation / "runtime" / (job["job_id"] + ".json"))
        if request != job["request"]:
            raise ValueError("runtime request differs from prepared schedule")


def run_campaign(preparation, out, expected_plan_hash, *, allow_r3_pending=False):
    if not allow_r3_pending:
        raise ValueError("R3 remains open; explicit sequencing-override authority is required")
    plan_path = preparation / "control/plan.json"
    if file_digest(plan_path) != expected_plan_hash:
        raise ValueError("prepared plan hash mismatch")
    plan = read_json(plan_path)
    with campaign_lock(out) as lock_fd:
        verify_launch_inputs(plan, preparation)
        metadata_path = out / "campaign.json"
        metadata = {
            "schema": "reliability-r4-campaign-v1", "plan_sha256": expected_plan_hash,
            "launcher_sha256": file_digest(Path(__file__)),
            "r3_review": "pending; historical conclusions provisional",
            "authority": "User authorized proceeding with R4 while R3 source review continues",
            "provider_calls": "not authorized; local automated only",
            "budget": {"max_runs": MAX_RUNS, "max_wall_seconds": MAX_WALL_SECONDS,
                       "wall_seconds_per_run": WALL_SECONDS, "reservation_seconds": RESERVATION_SECONDS},
        }
        write_once(metadata_path, metadata)
        while True:
            attempts = read_attempts(out, plan)
            job = next_job(plan, attempts)
            if job is None:
                break
            # The same solver/native/model revision must serve every arm.
            verify_launch_inputs(plan, preparation)
            request_path = preparation / "runtime" / (job["job_id"] + ".json")
            result_path = out / "results" / (job["job_id"] + ".json")
            write_once(out / "attempts" / (job["job_id"] + ".started.json"), {
                "job_id": job["job_id"], "request_sha256": job["request_sha256"],
                "reserved_seconds": RESERVATION_SECONDS,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(json.dumps({"event": "launch", "attempt": len(attempts) + 1,
                              "job_id": job["job_id"], "case_id": job["request"]["case_id"],
                              "arm": job["request"]["arm"], "seed": job["request"]["seed"]}), flush=True)
            env = worker_environment(job["request"], plan["provenance"]["model_files"])
            with (out / "guard.log").open("ab") as log:
                guard = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "guard",
                     "--request", str(request_path), "--result", str(result_path)],
                    env=env, cwd=ROOT, stdout=log, stderr=log,
                    start_new_session=True, pass_fds=(lock_fd,),
                )
                guard.wait()
            # Crashes spend the reserved slot. Never retry an invalid result.
            result = read_json(result_path) if result_path.exists() else None
            print(json.dumps({"event": "finished", "job_id": job["job_id"],
                              "execution_status": result["execution_status"] if result else "interrupted",
                              "wall_seconds": result.get("wall_seconds") if result else None}), flush=True)
        summary = {"planned": len(plan["jobs"]), "attempted": len(attempts),
                   "charged_seconds": sum(a["charged_seconds"] for a in attempts.values()),
                   "unattempted": [j["job_id"] for j in plan["jobs"] if j["job_id"] not in attempts],
                   "missing_results": [key for key, a in attempts.items() if a["execution"] is None]}
        write_once(out / "summary.json", summary)
        print(json.dumps({"event": "campaign_finished", **summary}), flush=True)
        return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--preparation", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--expected-plan-sha256", required=True)
    run.add_argument("--allow-r3-pending", action="store_true")
    guard = commands.add_parser("guard", help=argparse.SUPPRESS)
    guard.add_argument("--request", type=Path, required=True)
    guard.add_argument("--result", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "guard":
        guard_main(args.request, args.result)
    else:
        run_campaign(args.preparation.resolve(), args.out.resolve(), args.expected_plan_sha256,
                     allow_r3_pending=args.allow_r3_pending)


if __name__ == "__main__":
    main()
