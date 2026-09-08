"""Runtime-only R5 adapter. Both arms receive identical family-blind input."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reliability_routing_worker as r4

FIELDS = r4.RUNTIME_FIELDS | {"arm", "seed"}


def projected_request(request):
    if (not isinstance(request, dict) or set(request) != FIELDS
            or request["arm"] not in {"baseline", "prototype"} or request["seed"] != 61001):
        raise ValueError("R5 worker request violates frozen allowlist")
    value = {k: request[k] for k in r4.RUNTIME_FIELDS}
    value.update(arm="blind_family", seed=request["seed"], cipher_system="")
    r4.validate_request(value)
    return value


def worker_environment(request, models):
    env = r4.worker_environment(projected_request(request), models)
    env["DECIPHER_PERIODIC_ROUTING"] = "off" if request["arm"] == "baseline" else "probe_v1"
    return env


def solve_request(request, **kwargs):
    projected = projected_request(request)
    expected = "off" if request["arm"] == "baseline" else "probe_v1"
    if os.environ.get("DECIPHER_PERIODIC_ROUTING") != expected:
        raise ValueError("worker environment does not match the recorded R5 arm")
    result = r4.solve_request(projected, **kwargs)
    result["request_sha256"] = r4.digest(request)
    result["periodic_routing_mode"] = expected
    initial = result["routes_attempted"]
    probes = [s for s in result["artifact"].get("steps", []) if s.get("name") == "probe_periodic_routing"]
    if probes:
        probe = probes[-1]
        result["initial_route_selection"] = initial
        attempted = [{"route": "periodic_probe", "engine": stage} for stage in probe.get("stages", [])]
        if probe["decision"] in {"fallback", "skipped"}:
            attempted += initial
        result["routes_attempted"] = attempted
    return result


if __name__ == "__main__":
    import resource
    from contextlib import redirect_stdout
    resource.setrlimit(resource.RLIMIT_CPU, (720, 720))
    output = sys.stdout
    def emit(event):
        print(json.dumps(event), file=output, flush=True)
    with redirect_stdout(sys.stderr):
        result = solve_request(json.load(sys.stdin), emit=emit)
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    emit({"event": "result", "result": result, "cpu_usage": {
        "self_user_seconds": own.ru_utime, "self_system_seconds": own.ru_stime,
        "children_user_seconds": children.ru_utime, "children_system_seconds": children.ru_stime,
        "cpu_limit_per_process": [720, 720],
        "scope": "nested guard CPU is included only when reaped; timeout descendants may be unaccounted"}})
