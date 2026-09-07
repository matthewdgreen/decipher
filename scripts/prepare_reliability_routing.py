#!/usr/bin/env python3
"""Prepare (never execute) the frozen R4 paired routing measurement.

This is a grading-side controller. Only each allowlisted request, not this
module or its private schedule, may be supplied to a worker. No benchmark
loader, plaintext generation, solver search, or provider call is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reliability_routing_worker import (
    ARMS, FAMILIES, MAX_RUNS, MAX_WALL_SECONDS, RUNTIME_FIELDS,
    digest, validate_request,
)

BASELINE_REVISION = "9eaf309fee3ece258e202ddd186e2ee290874b8c"
PROTOCOL = {
    "arms": list(ARMS), "language": "same supplied language in both arms",
    "wall_seconds_per_run": 180, "cpu_seconds_per_process": 720,
    "workers": 4, "concurrent_runs": 1, "solver_profile": "shipped_defaults",
    "homophonic_budget": "screen", "transform_search": "off",
    "primary_seed": 61001, "development_seeds": [61001, 61002, 61003],
    "replication": "all six development cases; both arms; no adaptive repeats",
    "held_out_seeds": [61001], "additional_repeats": "none",
    "max_runs": MAX_RUNS, "max_wall_seconds": MAX_WALL_SECONDS,
    "timeouts": "report timeout without treating partial output as completion",
    "thresholds": {"near_exact_char": 0.99, "material_char_gap": 0.02},
    "interpretation": "small descriptive pilot; no family-wide or live-agent success claims",
    "seed_limitation": "R4 must record which engines honor injected seeds; fixed internal seeds are not independent replicates",
}


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def prepare_schedule(runtime, grading):
    """Project only role/family metadata; plaintext and keys never enter jobs."""
    if grading.get("packet_version") != "reliability-r0-v1" or grading.get("protocol") != PROTOCOL:
        raise ValueError("frozen R0 protocol changed; record an amendment before preparing R4")
    if digest(runtime) != grading.get("runtime_manifest_sha256"):
        raise ValueError("runtime manifest hash mismatch")
    runtime_ids = [r.get("case_id") for r in runtime]
    rows = grading["cases"]
    grading_ids = [r["case_id"] for r in rows]
    if (len(set(runtime_ids)) != len(runtime_ids) or len(set(grading_ids)) != len(grading_ids)
            or set(runtime_ids) != set(grading_ids)):
        raise ValueError("duplicate or unmatched packet ids")
    expected_roles = {"development": 6, "held_out": 6, "historical_diagnostic": 4,
                      "unsupported_control": 1, "random_control": 1}
    roles = Counter(row["role"] for row in rows)
    if any(role not in expected_roles or count > expected_roles[role] for role, count in roles.items()):
        raise ValueError("unexpected role membership")
    supported = FAMILIES - {"bifid"}
    for role in ("development", "held_out"):
        if Counter(r["family"] for r in rows if r["role"] == role) != Counter(supported):
            raise ValueError("development/held-out family coverage changed")
    by_id = {r["case_id"]: r for r in rows}
    jobs = []
    for case in sorted(runtime, key=lambda r: r["case_id"]):
        if set(case) != RUNTIME_FIELDS:
            raise ValueError("runtime case violates allowlist")
        row = by_id[case["case_id"]]
        if row["role"] == "random_control" and row["family"] is not None:
            raise ValueError("random control must not invent a family")
        if row["role"] == "unsupported_control" and row["family"] != "bifid":
            raise ValueError("unsupported control changed")
        seeds = PROTOCOL["development_seeds"] if row["role"] == "development" else [61001]
        for seed in seeds:
            # Adjacent matched pairs; deterministic counterbalancing, no result input.
            arms = ARMS if int(digest(f"{case['case_id']}:{seed}")[:8], 16) % 2 == 0 else ARMS[::-1]
            for arm in arms:
                request = dict(case, arm=arm, seed=seed,
                               cipher_system=(row["family"] or "") if arm == "family_supplied" else "")
                validate_request(request)
                jobs.append({"job_id": "j" + digest(request)[:16], "request": request,
                             "request_sha256": digest(request), "role": row["role"]})
    if len(jobs) > MAX_RUNS:
        raise ValueError("schedule exceeds frozen ceiling")
    return {
        "schema": "reliability-r4-preparation-v1", "status": "prepared_not_run",
        "execution_authorized": False, "pending_gate": "R3 human source review and closure",
        "packet_version": grading["packet_version"], "protocol": PROTOCOL,
        "runtime_manifest_sha256": digest(runtime), "jobs": jobs,
        "available_cases": len(runtime), "planned_runs": len(jobs),
        "missing_roles": {r: n - roles[r] for r, n in expected_roles.items() if n != roles[r]},
        "unavailable_cases": grading.get("unavailable", []),
        "seed_independence": "not established; repeats with fixed internal seeds are not independent trials",
    }


def capture_provenance(model_files):
    """Fail on solver drift; hash native code and actual auxiliary inputs as well."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    scopes = ["src", "rust", "resources", "models", "pyproject.toml"]
    if git("diff", BASELINE_REVISION, "--", *scopes):
        raise ValueError("solver/resources differ from the R2 checkpoint")
    if git("ls-files", "--others", "--exclude-standard", "--", *scopes):
        raise ValueError("untracked solver/resources require a provenance decision")
    verified_models = {}
    for language in ("en", "la", "de"):
        evidence = model_files.get(language)
        if not evidence or file_digest(evidence["path"]) != evidence["sha256"]:
            raise ValueError(f"frozen {language} model is absent or changed")
        verified_models[language] = evidence
    spec = importlib.util.find_spec("decipher_fast")
    if spec is None or not spec.origin:
        raise ValueError("required native module is absent")
    native_path = Path(spec.origin)
    native_files = [native_path] if native_path.suffix != ".py" else [native_path, *sorted(native_path.parent.glob("*.so"))]
    if not any(p.suffix in {".so", ".pyd"} for p in native_files):
        raise ValueError("native extension binary is absent")
    tracked = git("ls-files", "--", "src", "rust", "resources", "pyproject.toml").splitlines()
    inputs = {p: file_digest(ROOT / p) for p in tracked}
    # Runtime language scorers can use ignored/downloaded data too. Record it
    # explicitly; omit __pycache__ and model binaries already pinned above.
    auxiliary = {str(p.relative_to(ROOT)): file_digest(p)
                 for p in sorted((ROOT / "resources").rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts}
    return {
        "solver_revision": BASELINE_REVISION, "preparation_revision": git("rev-parse", "HEAD"),
        "model_files": verified_models, "tracked_input_hashes": inputs,
        "auxiliary_resource_hashes": auxiliary,
        "native_files": {str(p): file_digest(p) for p in native_files},
        "native_build_source_correspondence": "not independently attested; installed binary hash is pinned",
        "python": sys.version, "executable": sys.executable,
        "harness_files": {name: file_digest(ROOT / "scripts" / name) for name in
                          ("prepare_reliability_routing.py", "reliability_routing_worker.py")},
    }


def verify_prepared_inputs(plan):
    """Read-only recheck for a future launcher; never re-freeze drift silently."""
    provenance = plan["provenance"]
    if provenance["solver_revision"] != BASELINE_REVISION or plan["protocol"] != PROTOCOL:
        raise ValueError("prepared protocol or solver revision changed")
    jobs = plan["jobs"]
    if len(jobs) > MAX_RUNS or len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("invalid prepared schedule")
    for job in jobs:
        validate_request(job["request"])
        if (job["request_sha256"] != digest(job["request"])
                or job["job_id"] != "j" + digest(job["request"])[:16]):
            raise ValueError("prepared request changed")
    paths = {}
    for group in ("tracked_input_hashes", "auxiliary_resource_hashes"):
        paths.update({ROOT / name: sha for name, sha in provenance[group].items()})
    paths.update({Path(name): sha for name, sha in provenance["native_files"].items()})
    paths.update({Path(row["path"]): row["sha256"] for row in provenance["model_files"].values()})
    paths.update({ROOT / "scripts" / name: sha for name, sha in provenance["harness_files"].items()})
    for path, sha in paths.items():
        if not path.is_file() or file_digest(path) != sha:
            raise ValueError(f"prepared input absent or changed: {path}")


def write_preparation(out, plan):
    """Idempotent immutable preparation; validate every conflict before writes."""
    outputs = {out / "control/plan.json": encode(plan)}
    for job in plan["jobs"]:
        outputs[out / "runtime" / (job["job_id"] + ".json")] = encode(job["request"])
    for path, content in outputs.items():
        if path.exists() and path.read_text() != content:
            raise ValueError(f"refusing to overwrite prepared output: {path}")
    for path, content in outputs.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


def grade_completed(request, execution, label):
    """Post-hoc only: never used by scheduling, worker inputs, or retries.

    Character recovery is descriptive even for Bifid; controls do not count as
    expected successes. Missing candidates/timeout scores remain unknown.
    """
    validate_request(request)
    if label["case_id"] != request["case_id"] or execution["request_sha256"] != digest(request):
        raise ValueError("post-hoc join identity mismatch")
    score = None
    result = execution.get("result")
    if execution["execution_status"] == "completed" and result is not None:
        if (result["request_sha256"] != digest(request)
                or result["delivered_sha256"] != digest(result["delivered_text"])):
            raise ValueError("delivered candidate hash mismatch")
        truth = label.get("plaintext_spaced") or label.get("plaintext")
        if truth and result["delivered_text"]:
            from benchmark.scorer import score_decryption
            scored = score_decryption(request["case_id"], result["delivered_text"], truth,
                                      agent_score=0.0, status=result["status"])
            score = {"char_accuracy": scored.char_accuracy, "word_accuracy": scored.word_accuracy}
    return {
        "case_id": request["case_id"], "arm": request["arm"], "seed": request["seed"],
        "role": label["role"], "execution_status": execution["execution_status"],
        "solver_status": result.get("status") if result else None,
        "post_hoc_delivered": score,
        "expected_solve": label["role"] not in {"unsupported_control", "random_control"},
        "best_generated_quality": None,
        "best_generated_limitation": "saved menus are not an exhaustive generated-candidate trace",
        "causal_attribution": "requires paired route/candidate evidence; no automatic wrong-route label",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, default=ROOT / "artifacts/reliability_r0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runtime = [json.loads(line) for line in (args.packet / "runtime/cases.jsonl").read_text().splitlines() if line.strip()]
    grading_path = args.packet / "grading/packet.json"
    grading = json.loads(grading_path.read_text())
    plan = prepare_schedule(runtime, grading)
    plan["provenance"] = capture_provenance(grading["model_files"])
    plan["grading_packet_sha256"] = file_digest(grading_path)
    write_preparation(args.out, plan)
    print(json.dumps({k: plan[k] for k in ("status", "planned_runs", "available_cases", "pending_gate", "missing_roles")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
