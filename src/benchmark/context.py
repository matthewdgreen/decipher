"""Benchmark context policy and scoped agent access helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.loader import BenchmarkRecord, TestData


BENCHMARK_CONTEXT_POLICIES = (
    "none",
    "minimal",
    "standard",
    "historical",
    "related_metadata",
    "related_solutions",
    "max",
)

_PROMPT_LAYERS_BY_POLICY = {
    "none": [],
    "minimal": ["minimal"],
    "standard": ["minimal", "standard"],
    "historical": ["minimal", "standard", "historical"],
    "related_metadata": ["minimal", "standard", "historical"],
    "related_solutions": ["minimal", "standard", "historical"],
    "max": ["minimal", "standard", "historical"],
}

_RELATED_METADATA_POLICIES = {"related_metadata", "related_solutions", "max"}
_RELATED_SOLUTION_POLICIES = {"related_solutions", "max"}


@dataclass
class ScopedBenchmarkContext:
    """Context made available to the agent for a single benchmark run."""

    policy: str
    prompt: str = ""
    injected_layers: list[dict[str, Any]] = field(default_factory=list)
    target_record_ids: list[str] = field(default_factory=list)
    context_record_ids: list[str] = field(default_factory=list)
    related_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    associated_documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    benchmark_root: str = ""
    access_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def related_solution_allowed(self) -> bool:
        return self.policy in _RELATED_SOLUTION_POLICIES

    @property
    def related_metadata_allowed(self) -> bool:
        return self.policy in _RELATED_METADATA_POLICIES

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "prompt_injected": bool(self.prompt),
            "injected_layers": self.injected_layers,
            "target_record_ids": self.target_record_ids,
            "context_record_ids": self.context_record_ids,
            "related_records_available": [
                _public_related_entry(entry)
                for entry in self.related_records.values()
            ],
            "associated_documents_available": [
                _public_document_entry(entry)
                for entry in self.associated_documents.values()
            ],
            "related_solution_allowed": self.related_solution_allowed,
            "access_log": list(self.access_log),
        }

    def log_access(
        self,
        tool_name: str,
        *,
        record_id: str | None = None,
        document_id: str | None = None,
        content_type: str = "metadata",
        allowed: bool = True,
        error: str = "",
    ) -> None:
        self.access_log.append(
            {
                "tool": tool_name,
                "record_id": record_id,
                "document_id": document_id,
                "content_type": content_type,
                "allowed": allowed,
                "error": error,
            }
        )


def build_benchmark_context(
    test_data: TestData,
    policy: str = "max",
) -> ScopedBenchmarkContext | None:
    """Build prompt text and scoped tool access for a benchmark test."""

    if policy not in BENCHMARK_CONTEXT_POLICIES:
        raise ValueError(f"unknown benchmark context policy: {policy}")

    root = Path(test_data.benchmark_root).resolve() if test_data.benchmark_root else None
    context = ScopedBenchmarkContext(
        policy=policy,
        target_record_ids=[record.id for record in test_data.target_records],
        context_record_ids=[record.id for record in test_data.context_records],
        benchmark_root=str(root) if root else "",
    )
    if policy == "none":
        return context

    records: dict[str, dict[str, Any]] = {}
    for record in [*test_data.target_records, *test_data.context_records]:
        records[record.id] = _record_entry(record, area="benchmark", root=root)

    related_entries: dict[str, dict[str, Any]] = {}
    associated_documents: dict[str, dict[str, Any]] = {}

    if root and context.related_metadata_allowed:
        main_records = _load_manifest(root / "manifest" / "records.jsonl")
        unsolved_records = _load_manifest(root / "unsolved" / "manifest" / "records.jsonl")
        for record in [*test_data.target_records, *test_data.context_records]:
            for rel in record.raw.get("related_records", []) or []:
                record_id = rel.get("record_id")
                if not record_id:
                    continue
                area = rel.get("area", "benchmark")
                source_manifest = main_records if area == "benchmark" else unsolved_records
                target = source_manifest.get(record_id)
                if not target:
                    continue
                entry = _record_entry_from_raw(target, area=area, root=root)
                entry["relationship"] = rel
                records.setdefault(record_id, entry)
                related_entries[record_id] = entry

    for record in [*test_data.target_records, *test_data.context_records]:
        for document in record.raw.get("associated_documents", []) or []:
            doc_id = document.get("id")
            if not doc_id:
                continue
            safe_layers = set(document.get("safe_context_layers") or [])
            if safe_layers and policy not in safe_layers and policy != "max":
                continue
            if document.get("contains_solution") and not context.related_solution_allowed:
                continue
            associated_documents[doc_id] = {
                "document": dict(document),
                "record_id": record.id,
                "area": "benchmark",
            }

    context.records = records
    context.related_records = related_entries
    context.associated_documents = associated_documents
    context.injected_layers = _injected_layers(test_data, policy)
    context.prompt = _format_prompt_context(context, test_data)
    return context


def _injected_layers(test_data: TestData, policy: str) -> list[dict[str, Any]]:
    names = _PROMPT_LAYERS_BY_POLICY.get(policy, [])
    rows: list[dict[str, Any]] = []
    for record in [*test_data.target_records, *test_data.context_records]:
        layers = record.raw.get("context_layers", {}) or {}
        for name in names:
            layer = layers.get(name)
            if not layer:
                continue
            rows.append(
                {
                    "record_id": record.id,
                    "layer": name,
                    "label": layer.get("label", name),
                    "contains_solution": bool(layer.get("contains_solution")),
                    "contains_plaintext_hint": bool(layer.get("contains_plaintext_hint")),
                    "contains_cipher_type_hint": bool(layer.get("contains_cipher_type_hint")),
                    "text": str(layer.get("text", "")),
                }
            )
    return rows


def _format_prompt_context(
    context: ScopedBenchmarkContext,
    test_data: TestData,
) -> str:
    if context.policy == "none":
        return ""

    lines = [
        "## Benchmark Context",
        f"Context policy: `{context.policy}`.",
        (
            "The benchmark may expose auxiliary context in controlled layers. "
            "Use it as research context, but do not assume it is a solution."
        ),
    ]
    if context.injected_layers:
        lines.append("")
        lines.append("### Context Layers Injected")
        # Group layers by record_id to avoid repeating the record ID on every line.
        from collections import defaultdict
        layers_by_record: dict[str, list[dict]] = defaultdict(list)
        record_order: list[str] = []
        for item in context.injected_layers:
            rid = item["record_id"]
            if rid not in layers_by_record:
                record_order.append(rid)
            layers_by_record[rid].append(item)
        for rid in record_order:
            record_layers = layers_by_record[rid]
            layer_tags = []
            for item in record_layers:
                flags = []
                if item["contains_solution"]:
                    flags.append("contains solution")
                if item["contains_plaintext_hint"]:
                    flags.append("plaintext hint")
                if item["contains_cipher_type_hint"]:
                    flags.append("cipher-type hint")
                tag = item["label"]  # human-readable label, e.g. "Basic provenance"
                if flags:
                    tag += f" ({', '.join(flags)})"
                layer_tags.append(tag)
            header = f"- {rid} [{' | '.join(layer_tags)}]:"
            texts = " ".join(item["text"] for item in record_layers if item.get("text"))
            lines.append(f"{header} {texts}")
        if any(item.get("contains_cipher_type_hint") for item in context.injected_layers):
            lines.append("")
            lines.append("### Agent-Declared Context Assumptions")
            lines.append(
                "If you read these layers as identifying the cipher family, record "
                "that inference explicitly with `workspace_create_hypothesis_branch` "
                "or `workspace_update_hypothesis` and set "
                "`evidence_source='benchmark_context'`. The executor will only "
                "enforce context as a hard working assumption after you make that "
                "declaration; it does not infer cipher type from this prose itself."
            )

    if test_data.context_records:
        ids = ", ".join(record.id for record in test_data.context_records[:20])
        if len(test_data.context_records) > 20:
            ids += f", ... ({len(test_data.context_records)} total)"
        lines.append("")
        lines.append("### Auxiliary Context Records")
        lines.append(
            f"Context records listed by the split: {ids}. "
            "Use `inspect_related_transcription` to read their transcription "
            "instead of relying on a large initial prompt dump."
        )

    if context.related_records:
        lines.append("")
        lines.append("### Related Records Available By Tool")
        for entry in context.related_records.values():
            rel = entry.get("relationship", {})
            solution_note = (
                "solution available"
                if bool(rel.get("solution_available")) else "no related solution"
            )
            lines.append(
                f"- {entry['id']} ({rel.get('relationship', 'related')}): "
                f"{solution_note}. Use tools to inspect if useful."
            )

    if context.associated_documents:
        lines.append("")
        lines.append("### Associated Documents Available By Tool")
        for entry in context.associated_documents.values():
            doc = entry["document"]
            lines.append(
                f"- {doc.get('id')}: {doc.get('title', 'Untitled')} "
                f"({doc.get('document_type', 'document')})."
            )

    if context.related_records or context.associated_documents or test_data.context_records:
        lines.append("")
        lines.append(
            "Scoped benchmark tools are available: `inspect_benchmark_context`, "
            "`list_related_records`, `inspect_related_transcription`, "
            "`inspect_related_solution`, `list_associated_documents`, and "
            "`inspect_associated_document`. They only expose records/documents "
            "declared by this benchmark test and its manifest metadata."
        )

    return "\n".join(lines)


def _record_entry(
    record: BenchmarkRecord,
    area: str,
    root: Path | None = None,
) -> dict[str, Any]:
    return _record_entry_from_raw(record.raw, area=area, root=root, record=record)


def _record_entry_from_raw(
    raw: dict[str, Any],
    *,
    area: str,
    root: Path | None,
    record: BenchmarkRecord | None = None,
) -> dict[str, Any]:
    record_id = record.id if record is not None else raw.get("id", "")
    entry = {
        "id": record_id,
        "area": area,
        "source": raw.get("source", ""),
        "cipher_type": raw.get("cipher_type", []),
        "plaintext_language": raw.get("plaintext_language", ""),
        "status": raw.get("status", ""),
        "provenance": raw.get("provenance", ""),
        "date_or_century": raw.get("date_or_century", ""),
        "notes": raw.get("notes") or raw.get("curation_notes", ""),
        "transcription_canonical_file": raw.get("transcription_canonical_file", ""),
        "plaintext_file": raw.get("plaintext_file", ""),
        "has_key": raw.get("has_key", False),
        "raw": dict(raw),
    }
    if root is not None:
        entry["root"] = str(root if area == "benchmark" else root / "unsolved")
    return entry


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows[record["id"]] = record
    return rows


def _public_related_entry(entry: dict[str, Any]) -> dict[str, Any]:
    rel = entry.get("relationship", {})
    return {
        "record_id": entry.get("id"),
        "area": entry.get("area"),
        "relationship": rel.get("relationship"),
        "solution_available": bool(rel.get("solution_available")),
        "safe_context_layers": rel.get("safe_context_layers", []),
        "status": entry.get("status"),
        "source": entry.get("source"),
    }


def _public_document_entry(entry: dict[str, Any]) -> dict[str, Any]:
    doc = entry.get("document", {})
    return {
        "document_id": doc.get("id"),
        "record_id": entry.get("record_id"),
        "document_type": doc.get("document_type"),
        "title": doc.get("title"),
        "summary": doc.get("summary", ""),
        "contains_solution": bool(doc.get("contains_solution")),
        "contains_plaintext_hint": bool(doc.get("contains_plaintext_hint")),
        "safe_context_layers": doc.get("safe_context_layers", []),
    }


def safe_read_benchmark_file(root: str | Path, relative_path: str) -> str:
    """Read a benchmark-relative file after path containment checks."""

    base = Path(root).resolve()
    path = (base / relative_path).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path escapes benchmark root: {relative_path}") from exc
    if not path.exists():
        raise FileNotFoundError(relative_path)
    return path.read_text(encoding="utf-8").strip()
