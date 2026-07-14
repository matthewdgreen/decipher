"""Lightweight reader for the benchmark *unsolved* area (INV-0 Part 6).

``BenchmarkLoader`` reads only ``manifest/records.jsonl`` and is blind to the
unsolved manifest (the zodiac340 incident). Rather than extend it, INV-0 adds
this separate reader that reads ``unsolved/manifest/records.jsonl`` directly and
resolves ``transcription_canonical_file`` relative to ``unsolved/``.

FIREWALL: only ciphertext-safe structural fields are loaded. Solution-bearing or
context-bearing fields (``context_layers``, ``associated_documents``,
``notable_attempts``, ``partial_solution_evidence``, ``curation_notes``, images)
are NEVER loaded. A record or document whose ``rights_class == "hold_for_review"``
has its content withheld; the canonical numeric ciphertext stream still loads for
numeric targets (it is indices, not plaintext).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

# Fields that may carry solution hints or long context — never surfaced.
_WITHHELD_FIELDS = frozenset({
    "context_layers",
    "associated_documents",
    "notable_attempts",
    "partial_solution_evidence",
    "curation_notes",
    "image_files",
    "image_provenance",
})
# Safe structural metadata surfaced to the diagnosis layer.
_SAFE_METADATA_FIELDS = (
    "id", "source", "cipher_type", "symbol_set", "symbol_count", "token_count",
    "transcription_canonical_file", "word_boundaries", "rights_class",
    "plaintext_language", "status", "length_chars", "date_or_century",
    "provenance", "page_count", "manuscript_page", "synthetic", "source_url",
    "source_record_id", "task_tracks",
)
_HOLD_FOR_REVIEW = "hold_for_review"
_NUMERIC_TYPES = frozenset({"numeric_code", "book_cipher"})


@dataclass
class UnsolvedRecord:
    id: str
    source: str
    cipher_type: list[str]
    symbol_set: Any
    token_count: int
    canonical_text: str | None
    metadata: dict[str, Any]
    related_record_ids: list[str]
    content_withheld: bool = False

    def numeric_values(self) -> list[int]:
        """Parse the canonical numeric stream into integer values (numeric targets)."""
        if not self.canonical_text:
            return []
        from analysis.numeric_code import parse_numeric_ciphertext
        return parse_numeric_ciphertext(self.canonical_text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "cipher_type": self.cipher_type,
            "symbol_set": self.symbol_set,
            "token_count": self.token_count,
            "canonical_text_present": self.canonical_text is not None,
            "related_record_ids": self.related_record_ids,
            "content_withheld": self.content_withheld,
            "metadata": self.metadata,
        }


def _unsolved_dir(benchmark_root: str) -> str:
    return os.path.join(benchmark_root, "unsolved")


def _manifest_path(benchmark_root: str) -> str:
    return os.path.join(_unsolved_dir(benchmark_root), "manifest", "records.jsonl")


def _iter_manifest(benchmark_root: str):
    path = _manifest_path(benchmark_root)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def list_unsolved_record_ids(benchmark_root: str) -> list[str]:
    """All record ids in ``unsolved/manifest/records.jsonl`` (order preserved)."""
    ids: list[str] = []
    for rec in _iter_manifest(benchmark_root):
        rid = rec.get("id")
        if rid:
            ids.append(rid)
    return ids


def _is_numeric_target(cipher_type: list[str]) -> bool:
    return any(t in _NUMERIC_TYPES for t in cipher_type)


def load_unsolved_record(benchmark_root: str, record_id: str) -> UnsolvedRecord:
    """Load one unsolved record by id, with the firewall applied."""
    manifest = None
    for rec in _iter_manifest(benchmark_root):
        if rec.get("id") == record_id:
            manifest = rec
            break
    if manifest is None:
        raise KeyError(f"unsolved record not found: {record_id}")

    cipher_type = list(manifest.get("cipher_type") or [])
    rights_class = manifest.get("rights_class")
    content_withheld = rights_class == _HOLD_FOR_REVIEW

    related_record_ids = [
        r["record_id"]
        for r in manifest.get("related_records", [])
        if isinstance(r, dict) and "record_id" in r
    ]

    metadata = {
        k: manifest[k]
        for k in _SAFE_METADATA_FIELDS
        if k in manifest and k not in _WITHHELD_FIELDS
    }
    metadata["related_records"] = [
        {
            "record_id": r.get("record_id"),
            "relationship": r.get("relationship"),
            "area": r.get("area"),
        }
        for r in manifest.get("related_records", [])
        if isinstance(r, dict)
    ]

    # Load the canonical ciphertext. For a hold_for_review NON-numeric target the
    # transcription content is withheld; the numeric ciphertext (integer indices,
    # not plaintext) still loads for numeric targets.
    canonical_text: str | None = None
    canonical_rel = manifest.get("transcription_canonical_file")
    numeric_target = _is_numeric_target(cipher_type)
    if canonical_rel and (not content_withheld or numeric_target):
        canonical_path = os.path.join(_unsolved_dir(benchmark_root), canonical_rel)
        try:
            with open(canonical_path, encoding="utf-8") as f:
                canonical_text = f.read().strip()
        except OSError:
            canonical_text = None

    return UnsolvedRecord(
        id=manifest.get("id"),
        source=manifest.get("source"),
        cipher_type=cipher_type,
        symbol_set=manifest.get("symbol_set"),
        token_count=int(manifest.get("token_count") or 0),
        canonical_text=canonical_text,
        metadata=metadata,
        related_record_ids=related_record_ids,
        content_withheld=content_withheld,
    )
