"""Investigation registry: on-disk store, revisions, writer lease (Part 3).

Per-investigation directory ``<root>/<investigation_id>/``:

    investigation.json   authoritative document (single atomic file, 3.3)
    meta.json            denormalized listing stub (one commit may lag)
    events.jsonl         append-only emit stream (observability only)
    lease.lock           writer-lease flock file (3.4)

Writes use tmp-file + ``os.replace`` (atomic rename), so readers always see a
complete document. The revision protocol itself lives in the server (3.5); the
registry only stores and bumps.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import fcntl  # POSIX-only; the project targets macOS/Linux.

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX fallback (import safety only)
    fcntl = None  # type: ignore
    _HAVE_FCNTL = False


class InvestigationNotFound(Exception):
    """Raised by ``load`` for an unknown investigation id."""


def default_registry_dir() -> Path:
    """``--registry-dir`` → ``$DECIPHER_MCP_REGISTRY`` → the config default (3.1)."""
    env = os.environ.get("DECIPHER_MCP_REGISTRY")
    if env:
        return Path(env).expanduser()
    return Path("~/.config/decipher/investigations").expanduser()


def _atomic_write(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class InvestigationRegistry:
    """One instance per server process (constructed with the root path)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        # investigation_id -> open lease fd (kept for the process lifetime).
        self._leases: dict[str, int] = {}

    # --- paths ---
    def _dir(self, investigation_id: str) -> Path:
        return self.root / investigation_id

    def _doc_path(self, investigation_id: str) -> Path:
        return self._dir(investigation_id) / "investigation.json"

    def _meta_path(self, investigation_id: str) -> Path:
        return self._dir(investigation_id) / "meta.json"

    # --- store ---
    def create(self, *, meta: dict, state_dict: dict) -> None:
        """Write revision 1 for a new investigation (both atomic renames)."""
        investigation_id = str(meta["investigation_id"])
        self._dir(investigation_id).mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": 1,
            "meta": meta,
            "state": state_dict,
            "records": {"comparisons": [], "repair_compiles": []},
        }
        _atomic_write(self._doc_path(investigation_id), document)
        _atomic_write(self._meta_path(investigation_id), meta)

    def load(self, investigation_id: str) -> dict:
        path = self._doc_path(investigation_id)
        if not path.exists():
            raise InvestigationNotFound(investigation_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, limit: int = 50) -> list[dict]:
        """Return metas, newest ``updated_at`` first (best-effort reads)."""
        metas: list[dict] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            meta_path = child / "meta.json"
            if not meta_path.exists():
                continue
            try:
                metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        metas.sort(key=lambda m: float(m.get("updated_at") or 0.0), reverse=True)
        return metas[: max(0, int(limit))]

    def commit(self, investigation_id: str, document: dict) -> int:
        """Bump revision + updated_at, write atomically, return the new revision."""
        meta = document.setdefault("meta", {})
        meta["revision"] = int(meta.get("revision") or 0) + 1
        meta["updated_at"] = time.time()
        _atomic_write(self._doc_path(investigation_id), document)
        _atomic_write(self._meta_path(investigation_id), meta)
        return int(meta["revision"])

    def current_revision(self, investigation_id: str) -> int:
        return int(self.load(investigation_id)["meta"].get("revision") or 0)

    # --- events (observability only; failures swallowed) ---
    def append_event(self, investigation_id: str, event: dict) -> None:
        try:
            path = self._dir(investigation_id) / "events.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 - observability must never break a call
            pass

    # --- writer lease (3.4) ---
    def acquire_lease(self, investigation_id: str) -> bool:
        """Acquire (or confirm) the process-held writer lease for this id.

        Returns True on success (or if this process already holds it), False if
        another live process holds it (``BlockingIOError`` from ``flock``).
        """
        if investigation_id in self._leases:
            return True
        self._dir(investigation_id).mkdir(parents=True, exist_ok=True)
        lock_path = self._dir(investigation_id) / "lease.lock"
        if _HAVE_FCNTL:
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                return False
            except OSError:
                os.close(fd)
                return False
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps(
                {"pid": os.getpid(), "acquired_at": time.time()}
            ).encode("utf-8"))
            os.fsync(fd)
            self._leases[investigation_id] = fd
            return True
        # Non-POSIX fallback: O_CREAT|O_EXCL pid-file with stale-pid takeover.
        return self._acquire_lease_pidfile(investigation_id, lock_path)

    def _acquire_lease_pidfile(self, investigation_id: str, lock_path: Path) -> bool:  # pragma: no cover - non-POSIX
        for _attempt in range(2):
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
            except FileExistsError:
                try:
                    prior = json.loads(lock_path.read_text(encoding="utf-8"))
                    pid = int(prior.get("pid") or 0)
                    os.kill(pid, 0)
                    return False  # holder alive
                except (OSError, ValueError, json.JSONDecodeError):
                    try:
                        lock_path.unlink()
                    except OSError:
                        return False
                    continue
            os.write(fd, json.dumps(
                {"pid": os.getpid(), "acquired_at": time.time()}
            ).encode("utf-8"))
            self._leases[investigation_id] = fd
            return True
        return False

    def holds_lease(self, investigation_id: str) -> bool:
        return investigation_id in self._leases

    def lease_holder_hint(self, investigation_id: str) -> dict | None:
        """Best-effort ``{"pid": ...}`` read of a foreign lease file."""
        lock_path = self._dir(investigation_id) / "lease.lock"
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("pid") is not None:
                return {"pid": data.get("pid")}
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        return None
