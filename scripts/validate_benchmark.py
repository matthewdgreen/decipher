#!/usr/bin/env python3
"""Validate a classical-cipher benchmark checkout.

This is intentionally lightweight: it checks the project schema shape, record
file references, track-specific data layers, and split references without
requiring an external JSON Schema package.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TRACKS = {"image2transcription", "transcription2plaintext", "image2plaintext"}


@dataclass
class ValidationReport:
    records: int = 0
    splits: int = 0
    split_tests: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    by_source: Counter[str] = field(default_factory=Counter)
    by_track: Counter[str] = field(default_factory=Counter)
    by_status: Counter[str] = field(default_factory=Counter)
    by_language: Counter[str] = field(default_factory=Counter)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_benchmark(root: str | Path, strict: bool = False) -> ValidationReport:
    root = Path(root).expanduser().resolve()
    report = ValidationReport()

    manifest_path = root / "manifest" / "records.jsonl"
    schema_path = root / "manifest" / "schema.json"
    splits_dir = root / "splits"

    if not manifest_path.exists():
        report.errors.append(f"missing manifest: {manifest_path}")
        return report
    if not schema_path.exists():
        report.errors.append(f"missing schema: {schema_path}")
        return report

    schema = _load_json(schema_path, report)
    if not isinstance(schema, dict):
        return report

    records = _load_manifest(manifest_path, report)
    properties = set((schema.get("properties") or {}).keys())
    required = set(schema.get("required") or [])
    allow_extra = not (schema.get("additionalProperties") is False)

    seen_ids: set[str] = set()
    for line_no, record in records:
        record_id = record.get("id", f"<line {line_no}>")
        report.records += 1
        if record_id in seen_ids:
            report.errors.append(f"{record_id}: duplicate record id")
        seen_ids.add(record_id)

        _validate_schema_shape(
            record=record,
            record_id=record_id,
            line_no=line_no,
            required=required,
            properties=properties,
            allow_extra=allow_extra,
            report=report,
        )
        _validate_record_semantics(root, record, record_id, report, strict=strict)

        report.by_source[record.get("source", "<missing>")] += 1
        report.by_status[record.get("status", "<missing>")] += 1
        report.by_language[record.get("plaintext_language", "<missing>")] += 1
        for track in record.get("task_tracks") or []:
            report.by_track[track] += 1

    if splits_dir.exists():
        _validate_splits(splits_dir, seen_ids, report)
    else:
        report.warnings.append(f"missing splits directory: {splits_dir}")

    return report


def _load_json(path: Path, report: ValidationReport) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"invalid JSON in {path}: {exc}")
        return None


def _load_manifest(path: Path, report: ValidationReport) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            report.errors.append(f"manifest line {line_no}: invalid JSON: {exc}")
            continue
        if not isinstance(record, dict):
            report.errors.append(f"manifest line {line_no}: record is not an object")
            continue
        records.append((line_no, record))
    return records


def _validate_schema_shape(
    record: dict[str, Any],
    record_id: str,
    line_no: int,
    required: set[str],
    properties: set[str],
    allow_extra: bool,
    report: ValidationReport,
) -> None:
    for field in sorted(required - set(record)):
        report.errors.append(f"{record_id}: missing required field {field!r} (line {line_no})")
    if not allow_extra:
        for field in sorted(set(record) - properties):
            report.errors.append(f"{record_id}: field {field!r} is not in schema (line {line_no})")


def _validate_record_semantics(
    root: Path,
    record: dict[str, Any],
    record_id: str,
    report: ValidationReport,
    strict: bool = False,
) -> None:
    tracks = record.get("task_tracks") or []
    unknown_tracks = sorted(set(tracks) - TRACKS)
    for track in unknown_tracks:
        report.errors.append(f"{record_id}: unknown task track {track!r}")

    if "image2transcription" in tracks:
        _require_field(record, record_id, "image_files", report)
        if strict:
            _require_field(record, record_id, "transcription_diplomatic_file", report)
        elif not record.get("transcription_diplomatic_file"):
            report.warnings.append(
                f"{record_id}: Track A record has no diplomatic transcription yet"
            )
    if "transcription2plaintext" in tracks:
        _require_field(record, record_id, "transcription_canonical_file", report)
        _require_field(record, record_id, "plaintext_file", report)
    if "image2plaintext" in tracks:
        _require_field(record, record_id, "image_files", report)
        _require_field(record, record_id, "plaintext_file", report)

    for field in (
        "transcription_diplomatic_file",
        "transcription_canonical_file",
        "plaintext_file",
    ):
        value = record.get(field)
        if value and not (root / value).exists():
            report.errors.append(f"{record_id}: missing {field} file {value}")

    for image in record.get("image_files") or []:
        image_path = root / image
        if not image_path.exists():
            report.errors.append(f"{record_id}: missing image file {image}")
        elif not _looks_like_image(image_path):
            report.errors.append(f"{record_id}: image file does not look like an image: {image}")


def _require_field(
    record: dict[str, Any],
    record_id: str,
    field: str,
    report: ValidationReport,
) -> None:
    if not record.get(field):
        report.errors.append(f"{record_id}: track requires {field}")


def _looks_like_image(path: Path) -> bool:
    try:
        prefix = path.read_bytes()[:12]
    except OSError:
        return False
    return (
        prefix.startswith(b"\xff\xd8\xff")
        or prefix.startswith(b"\x89PNG\r\n\x1a\n")
        or prefix.startswith(b"GIF87a")
        or prefix.startswith(b"GIF89a")
        or prefix.startswith(b"II*\x00")
        or prefix.startswith(b"MM\x00*")
    )


def _validate_splits(
    splits_dir: Path,
    record_ids: set[str],
    report: ValidationReport,
) -> None:
    for split in sorted(splits_dir.glob("*.jsonl")):
        report.splits += 1
        for line_no, line in enumerate(split.read_text().splitlines(), 1):
            if not line.strip():
                continue
            report.split_tests += 1
            try:
                test = json.loads(line)
            except json.JSONDecodeError as exc:
                report.errors.append(f"{split.name}:{line_no}: invalid JSON: {exc}")
                continue
            test_id = test.get("test_id", f"{split.name}:{line_no}")
            if test.get("track") not in TRACKS:
                report.errors.append(f"{test_id}: invalid track {test.get('track')!r}")
            for field in ("target_records", "context_records"):
                for record_id in test.get(field) or []:
                    if record_id not in record_ids:
                        report.errors.append(
                            f"{test_id}: {field} references unknown record {record_id!r}"
                        )


def format_report(report: ValidationReport) -> str:
    lines = [
        "BENCHMARK VALIDATION",
        f"  records: {report.records}",
        f"  split files: {report.splits}",
        f"  split tests: {report.split_tests}",
        "",
        "By source:",
    ]
    for key, value in report.by_source.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("By track:")
    for key, value in report.by_track.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("By status:")
    for key, value in report.by_status.most_common():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("By language:")
    for key, value in report.by_language.most_common():
        lines.append(f"  {key}: {value}")

    if report.warnings:
        lines.append("")
        lines.append(f"Warnings ({len(report.warnings)}):")
        for warning in report.warnings[:50]:
            lines.append(f"  - {warning}")
        if len(report.warnings) > 50:
            lines.append(f"  ... {len(report.warnings) - 50} more")

    if report.errors:
        lines.append("")
        lines.append(f"Errors ({len(report.errors)}):")
        for error in report.errors[:80]:
            lines.append(f"  - {error}")
        if len(report.errors) > 80:
            lines.append(f"  ... {len(report.errors) - 80} more")
    else:
        lines.append("")
        lines.append("No errors.")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark_root",
        nargs="?",
        default="../cipher_benchmark/benchmark",
        help="Path to benchmark root containing manifest/, splits/, and sources/.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing Track A diplomatic transcriptions as errors.",
    )
    args = parser.parse_args(argv)
    report = validate_benchmark(args.benchmark_root, strict=args.strict)
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
