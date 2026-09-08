"""POSIX worker guard: one deadline, parent-death watch, confirmed group cleanup.

The independent guard survives loss of its caller's process group. Its lease
pipe reaches EOF when the caller dies; only the guard inherits the read end.
No provider or grading imports. Commands are internal trusted worker programs.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time

CLEANUP_SECONDS = 2.0
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
REGISTRY_ENV = "DECIPHER_BOUNDED_REGISTRY"
ANCESTORS_ENV = "DECIPHER_BOUNDED_ANCESTORS"


def write_ticket(path, value):
    """Publish an acknowledgement atomically in our private guard directory."""
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".ack-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def descendants_clean(registry, ancestors):
    """A nested out-of-group guard must acknowledge cleanup independently."""
    deadline = time.monotonic() + CLEANUP_SECONDS + 1
    while True:
        pending = []
        for name in os.listdir(registry):
            if not name.endswith(".json") or name in ancestors:
                continue
            try:
                with open(os.path.join(registry, name)) as stream:
                    ticket = json.load(stream)
                if ticket.get("cleanup_confirmed") is True:
                    continue
            except (OSError, ValueError):
                pass
            pending.append(name)
        if not pending:
            return True, None
        if time.monotonic() >= deadline:
            return False, f"nested guards have no clean acknowledgement: {pending}"
        time.sleep(.02)


def cleanup_group(process):
    """Kill only our newly created session, then bound reaping/confirmation."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    deadline = time.monotonic() + CLEANUP_SECONDS
    try:
        process.wait(timeout=CLEANUP_SECONDS)
    except subprocess.TimeoutExpired:
        return False, "worker did not reap after SIGKILL"
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True, None
        except OSError as exc:
            return False, f"cannot confirm process-group exit: {exc}"
        if time.monotonic() >= deadline:
            return False, "process group still visible after cleanup deadline"
        time.sleep(0.02)


def supervise(command, payload, *, lease_fd, deadline, cwd):
    """Guard-side primitive. Valid partial JSONL events survive crashes/timeouts."""
    started = time.monotonic()
    process = None
    status, clean, error = "worker_error", True, None
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            if started >= deadline:
                status = "timeout"
            else:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output,
                    stderr=errors, cwd=cwd, start_new_session=True, close_fds=True)
                process.stdin.write((json.dumps(payload) + "\n").encode())
                process.stdin.close()
                while True:
                    if select.select([lease_fd], [], [], 0)[0] and not os.read(lease_fd, 1):
                        status = "interrupted"
                        break
                    if time.monotonic() >= deadline:
                        status = "timeout"
                        break
                    if process.poll() is not None:
                        status = "completed" if process.returncode == 0 else "worker_error"
                        break
                    time.sleep(0.02)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if process is not None:
                clean, cleanup_error = cleanup_group(process)
                if not clean:
                    status, error = "cleanup_failed", cleanup_error
            registry = os.environ.get(REGISTRY_ENV)
            if registry:
                nested_clean, nested_error = descendants_clean(
                    registry, json.loads(os.environ.get(ANCESTORS_ENV, "[]")))
                if not nested_clean:
                    clean, status, error = False, "cleanup_failed", nested_error
        events, invalid = [], False
        output.seek(0)
        raw = output.read(MAX_OUTPUT_BYTES + 1)
        if len(raw) > MAX_OUTPUT_BYTES:
            invalid = True
        for line in raw[:MAX_OUTPUT_BYTES].splitlines():
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("event must be an object")
                events.append(item)
            except (ValueError, UnicodeDecodeError):
                invalid = True
        if invalid and status == "completed":
            status = "worker_error"
        errors.seek(0)
        return {"execution_status": status, "cleanup_confirmed": clean,
                "error": error, "returncode": process.returncode if process else None,
                "worker_pid": process.pid if process else None,
                "wall_seconds": time.monotonic() - started,
                "events": events, "invalid_output": invalid,
                "stderr": errors.read(16000).decode(errors="replace")}


def run_guarded(command, payload, *, env, cwd, wall_seconds, lock_fd=None):
    """Caller primitive; no fallback is safe if cleanup_confirmed is false."""
    if os.name != "posix" or not 0 < wall_seconds <= 180:
        raise ValueError("bounded worker requires POSIX and a 0–180 second limit")
    started = time.monotonic()
    read_fd, write_fd = os.pipe()
    guard = None
    ticket_path = None
    env = dict(env)
    try:
        if env.get(REGISTRY_ENV):
            fd, ticket_path = tempfile.mkstemp(dir=env[REGISTRY_ENV], suffix=".json", prefix="guard-")
            with os.fdopen(fd, "w") as stream:
                json.dump({"cleanup_confirmed": None, "parent_pid": os.getpid()}, stream)
            ancestors = json.loads(env.get(ANCESTORS_ENV, "[]"))
            env[ANCESTORS_ENV] = json.dumps([*ancestors, os.path.basename(ticket_path)])
        try:
            guard = subprocess.Popen(
                [sys.executable, "-m", "automated.bounded_process", str(read_fd)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True, pass_fds=(read_fd,) if lock_fd is None else (read_fd, lock_fd), env=env, cwd=cwd)
        except OSError as exc:
            if ticket_path:
                write_ticket(ticket_path, {"cleanup_confirmed": True, "not_started": True})
            return {"execution_status": "worker_error", "cleanup_confirmed": True,
                    "events": [], "error": f"guard launch failed: {exc}"}
        os.close(read_fd)
        read_fd = None
        message = {"command": command, "payload": payload, "cwd": str(cwd),
                   "deadline": started + wall_seconds, "ticket_path": ticket_path}
        try:
            stdout, stderr = guard.communicate(json.dumps(message).encode(),
                                               timeout=wall_seconds + 2 * CLEANUP_SECONDS + 4)
        except subprocess.TimeoutExpired:
            # Closing the lease requests cleanup, not an unsafe guard SIGKILL.
            return {"execution_status": "cleanup_failed", "cleanup_confirmed": False,
                    "guard_pid": guard.pid, "events": [],
                    "error": "guard did not acknowledge cleanup before deadline"}
        try:
            result = json.loads(stdout)
            if not isinstance(result, dict) or type(result.get("cleanup_confirmed")) is not bool:
                raise ValueError("invalid guard response")
        except (ValueError, UnicodeDecodeError):
            return {"execution_status": "cleanup_failed", "cleanup_confirmed": False,
                    "guard_pid": guard.pid, "events": [],
                    "error": "guard exited without a valid cleanup acknowledgement",
                    "stderr": stderr.decode(errors="replace")[:16000]}
        if guard.returncode:
            result.update(execution_status="cleanup_failed", cleanup_confirmed=False)
        result["wall_seconds"] = time.monotonic() - started
        return result
    finally:
        if read_fd is not None:
            os.close(read_fd)
        os.close(write_fd)
        if guard is not None and guard.poll() is None:
            try:
                guard.wait(timeout=CLEANUP_SECONDS + 1)
            except subprocess.TimeoutExpired:
                pass  # Guard still owns its deadline; never claim clean fallback.


if __name__ == "__main__":
    request = json.load(sys.stdin)
    owned_registry = not os.environ.get(REGISTRY_ENV)
    if owned_registry:
        os.environ[REGISTRY_ENV] = tempfile.mkdtemp(prefix="decipher-guards-")
    result = supervise(request["command"], request["payload"], lease_fd=int(sys.argv[1]),
                       deadline=request["deadline"], cwd=request["cwd"])
    if request.get("ticket_path"):
        write_ticket(request["ticket_path"], {"cleanup_confirmed": result["cleanup_confirmed"],
                                             "guard_pid": os.getpid(), "worker_pid": result["worker_pid"]})
    if owned_registry:
        if result["cleanup_confirmed"]:
            shutil.rmtree(os.environ[REGISTRY_ENV])
        else:
            result["cleanup_registry"] = os.environ[REGISTRY_ENV]
    print(json.dumps(result), flush=True)
