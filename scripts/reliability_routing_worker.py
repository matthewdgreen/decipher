"""Runtime-only R4 adapter and bounded subprocess mechanics.

No grading-store imports or paths belong here. This is not a security sandbox:
the firewall is an explicit input contract, clean environment, and one-way
results. The preparation CLI does not call this worker or start a campaign.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FIELDS = {"case_id", "ciphertext", "format", "language", "ciphertext_sha256"}
REQUEST_FIELDS = RUNTIME_FIELDS | {"arm", "seed", "cipher_system"}
FAMILIES = {
    "simple_substitution", "homophonic_substitution", "vigenere", "quagmire3",
    "columnar_transposition", "substitution_transposition", "bifid",
}
ARMS = ("blind_family", "family_supplied")
WALL_SECONDS = 180
CPU_SECONDS = 720
WORKERS = 4
MAX_RUNS = 60
MAX_WALL_SECONDS = 10800


def digest(value):
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_request(request):
    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
        raise ValueError("worker request must use the exact runtime allowlist")
    if not isinstance(request["case_id"], str) or not re.fullmatch(r"r[0-9a-f]{12}", request["case_id"]):
        raise ValueError("case id must be opaque")
    if not isinstance(request["ciphertext"], str) or not request["ciphertext"].strip():
        raise ValueError("empty ciphertext")
    if request["ciphertext_sha256"] != digest(request["ciphertext"]):
        raise ValueError("ciphertext hash mismatch")
    if request["format"] not in {"canonical", "letters"} or request["language"] not in {"en", "la", "de"}:
        raise ValueError("unsupported input convention")
    if request["arm"] not in ARMS:
        raise ValueError("unknown arm")
    if type(request["seed"]) is not int or request["seed"] not in (61001, 61002, 61003):
        raise ValueError("seed is outside the frozen protocol")
    if request["cipher_system"] not in FAMILIES | {""}:
        raise ValueError("unsupported family metadata")
    if request["arm"] == "blind_family" and request["cipher_system"]:
        raise ValueError("blind arm cannot receive family metadata")


def solver_kwargs(request):
    validate_request(request)
    return {
        "language": request["language"], "cipher_id": request["case_id"],
        "cipher_system": request["cipher_system"], "homophonic_budget": "screen",
        "homophonic_refinement": "none", "homophonic_solver": "zenith_native",
        "transform_search": "off",
    }


def worker_environment(request, model_files, *, inherited=None):
    """Discard caller solver knobs, Python injection, and provider credentials."""
    validate_request(request)
    inherited = os.environ if inherited is None else inherited
    env = {key: inherited[key] for key in ("PATH", "TMPDIR", "SYSTEMROOT") if key in inherited}
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONHASHSEED=str(request["seed"]),
               PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", LC_ALL="C", LANG="C")
    for key in (
        "DECIPHER_PARALLEL_WORKERS", "DECIPHER_QUAGMIRE_THREADS",
        "DECIPHER_PURE_TRANSPOSITION_THREADS", "DECIPHER_NULL_MASK_THREADS",
        "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(WORKERS)
    env["DECIPHER_QUAGMIRE_SEARCH_SEED"] = str(request["seed"])
    for language in ("en", "la", "de"):
        env[f"DECIPHER_NGRAM_MODEL_{language.upper()}"] = model_files[language]["path"]
    return env


def launch_allowed(completed_or_started, elapsed_seconds):
    """Reserve a full arm; never shorten a late run or retry by result quality."""
    return (0 <= completed_or_started < MAX_RUNS
            and 0 <= elapsed_seconds <= MAX_WALL_SECONDS - WALL_SECONDS)


def seed_evidence(steps, requested):
    quagmire = [s for s in steps if s.get("name") == "search_quagmire3_keyword_alphabet"]
    honored = bool(quagmire and all(s.get("seed") == requested for s in quagmire))
    return {
        "requested_seed": requested,
        "quagmire_seed_echo_matches": honored if quagmire else None,
        "independent_replication_established": False,
        "limitation": (
            "Quagmire exposes an injected seed; an echoed seed alone is not an independence proof. "
            "Other shipped routes use fixed internal seeds and/or mixed RNGs. "
            "Treat repeated calls as repeated executions, not independent seed trials."
        ),
    }


def solve_request(request, *, run_solver=None, intake=None, emit=None):
    """Call the existing API without labels, hints, known keys, or route overrides."""
    kwargs = solver_kwargs(request)
    if run_solver is None:
        from automated.runner import run_automated
        run_solver = run_automated
    if intake is None:
        from mcp_server.intake import build_cipher_text
        intake = build_cipher_text
    random.seed(request["seed"])
    cipher = intake(request["ciphertext"], request["format"])
    def on_step(name, status, elapsed_seconds):
        if emit:
            emit({"event": "progress", "name": name, "status": status,
                  "elapsed_seconds": elapsed_seconds})

    result = run_solver(cipher_text=cipher, on_step=on_step, **kwargs)
    steps = result.steps
    artifact = result.artifact
    # The ordinary artifact has empty post-hoc score placeholders; do not
    # misrepresent these zeros as measured failures in the runtime result.
    artifact = {k: v for k, v in artifact.items()
                if k not in {"ground_truth", "char_accuracy", "word_accuracy"}}
    return {
        "request_sha256": digest(request), "status": result.status,
        "solver": result.solver, "artifact": artifact,
        "delivered_text": result.final_decryption,
        "delivered_sha256": digest(result.final_decryption),
        "artifact_text_sha256": digest(artifact.get("decryption", "")),
        "delivery_matches_artifact": result.final_decryption == artifact.get("decryption"),
        "routes_attempted": [s for s in steps if s.get("name") == "route_automated_solver"],
        "seed_evidence": seed_evidence(steps, request["seed"]),
        "generated_menu_completeness": "unknown; saved steps are not every generated candidate",
        "verification": "not_run; completed does not mean solved",
    }


def run_bounded(command, request, *, env, cwd, wall_seconds=WALL_SECONDS):
    """Sequential caller primitive; kill the entire child session on every exit.

    CPU limits are installed by worker_main before solver imports and inherited
    by descendants. Wall limits include imports/startup. Smaller limits are for
    fake-worker mechanics tests only. Logs preserve progress on a timeout, which
    never becomes a completed result even if the child emitted a final packet.
    """
    validate_request(request)
    if not 0 < wall_seconds <= WALL_SECONDS:
        raise ValueError("wall limit exceeds frozen protocol")
    started = time.monotonic()
    timed_out = False
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                   env=env, cwd=cwd, start_new_session=True)
        try:
            try:
                process.communicate(json.dumps(request).encode(), timeout=wall_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        stdout.seek(0)
        stderr.seek(0)
        raw = stdout.read().decode("utf-8", errors="replace")
        errors = stderr.read().decode("utf-8", errors="replace")
    events = []
    invalid = False
    for line in raw.splitlines():
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("event must be an object")
            events.append(item)
        except ValueError:
            invalid = True
    finals = [e for e in events if e.get("event") == "result"]
    valid = (not invalid and len(finals) == 1 and events[-1] == finals[0]
             and isinstance(finals[0].get("result"), dict)
             and finals[0]["result"].get("request_sha256") == digest(request))
    status = "timeout" if timed_out else "completed" if process.returncode == 0 and valid else "worker_error"
    return {
        "request_sha256": digest(request), "execution_status": status,
        "returncode": process.returncode, "wall_seconds": time.monotonic() - started,
        "events": events, "stderr": errors, "invalid_stdout": raw if invalid else None,
        "result": finals[0]["result"] if status == "completed" else None,
        "cpu_usage": finals[0].get("cpu_usage") if valid and not timed_out else None,
        "cpu_usage_scope": "worker plus reaped children; unavailable after forced termination",
    }


def worker_main():
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    request = json.load(sys.stdin)
    validate_request(request)
    # Imports can print diagnostics; stdout remains a JSONL evidence stream.
    from contextlib import redirect_stdout
    output = sys.stdout

    def emit(event):
        print(json.dumps(event, ensure_ascii=False), file=output, flush=True)

    with redirect_stdout(sys.stderr):
        result = solve_request(request, emit=emit)
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    emit({"event": "result", "result": result, "cpu_usage": {
        "self_user_seconds": own.ru_utime, "self_system_seconds": own.ru_stime,
        "children_user_seconds": children.ru_utime, "children_system_seconds": children.ru_stime,
        "self_maxrss": own.ru_maxrss, "maxrss_units": "bytes on macOS; KiB on Linux",
        "cpu_limit_per_process": list(resource.getrlimit(resource.RLIMIT_CPU)),
    }})


if __name__ == "__main__":
    worker_main()
