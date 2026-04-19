from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from testgen.spec import TestSpec


class PlaintextCache:
    """Disk-backed cache for LLM-generated plaintexts.

    Each entry is a single JSON file at {cache_dir}/{language}_{hash12}.json.
    Writes are atomic (tmp → rename). The cache key covers only the fields
    that determine what the LLM generates (language, approx_length, topic,
    frequency_style); seed is intentionally excluded because it governs the
    cipher key, not the plaintext.
    """

    def __init__(self, cache_dir: str | Path = "testgen_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hash(self, spec: TestSpec) -> str:
        canonical = json.dumps({
            "language": spec.language,
            "approx_length": spec.approx_length,
            "topic": spec.topic,
            "frequency_style": spec.frequency_style,
        }, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def _path(self, spec: TestSpec) -> Path:
        return self.cache_dir / f"{spec.language}_{self._hash(spec)}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, spec: TestSpec) -> str | None:
        """Return cached plaintext, or None on miss."""
        p = self._path(spec)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data["plaintext"]

    def put(self, spec: TestSpec, plaintext: str) -> None:
        """Store plaintext atomically."""
        p = self._path(spec)
        tmp = p.with_suffix(".tmp")
        data = {
            "language": spec.language,
            "approx_length": spec.approx_length,
            "topic": spec.topic,
            "frequency_style": spec.frequency_style,
            "plaintext": plaintext,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    def flush(self, spec: TestSpec | None = None) -> int:
        """Delete cached entries. If spec is None, flush all. Returns count deleted."""
        if spec is not None:
            p = self._path(spec)
            if p.exists():
                p.unlink()
                return 1
            return 0
        count = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink()
            count += 1
        return count

    def list_entries(self) -> list[dict]:
        """Return summary dicts for all cached entries, sorted by filename."""
        result = []
        for p in sorted(self.cache_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                result.append({
                    "file": p.name,
                    "language": data.get("language", "?"),
                    "approx_length": data.get("approx_length", 0),
                    "topic": data.get("topic", "?"),
                    "frequency_style": data.get("frequency_style", "normal"),
                    "generated_at": data.get("generated_at", "?"),
                    "word_count": len(data.get("plaintext", "").split()),
                })
            except Exception:  # noqa: BLE001
                result.append({"file": p.name, "error": "unreadable"})
        return result
