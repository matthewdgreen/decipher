from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from models.alphabet import Alphabet
from models.cipher_text import CipherText


@dataclass
class BenchmarkRecord:
    """A single page-level record from the benchmark manifest."""

    id: str
    source: str
    cipher_type: list[str]
    plaintext_language: str
    transcription_canonical_file: str
    plaintext_file: str
    has_key: bool
    # Raw manifest data for any extra fields
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class BenchmarkTest:
    """A single test case from a benchmark split."""

    test_id: str
    track: str
    cipher_system: str
    target_records: list[str]
    context_records: list[str]
    description: str


@dataclass
class TestData:
    """Loaded data for a single test, ready to run."""

    test: BenchmarkTest
    canonical_transcription: str  # full canonical transcription text
    plaintext: str  # ground truth plaintext
    plaintext_language: str = ""
    context_canonical_transcription: str = ""
    context_plaintext: str = ""
    context_records: list[BenchmarkRecord] = field(default_factory=list)
    symbol_map: dict | None = None  # optional symbol map metadata
    transform_pipeline: dict | None = None


class BenchmarkLoader:
    """Loads benchmark tests and their associated data files."""

    def __init__(self, benchmark_root: str | Path) -> None:
        self.root = Path(benchmark_root).expanduser().resolve()
        self._manifest: dict[str, BenchmarkRecord] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        manifest_path = self.root / "manifest" / "records.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                record = BenchmarkRecord(
                    id=data["id"],
                    source=data.get("source", ""),
                    cipher_type=data.get("cipher_type", []),
                    plaintext_language=data.get("plaintext_language", ""),
                    transcription_canonical_file=data.get("transcription_canonical_file", ""),
                    plaintext_file=data.get("plaintext_file", ""),
                    has_key=data.get("has_key", False),
                    raw=data,
                )
                self._manifest[record.id] = record

    def load_tests(
        self,
        split_file: str | Path,
        track: str | None = None,
        source: str | None = None,
    ) -> list[BenchmarkTest]:
        """Load test definitions from a split JSONL file.

        Optionally filter by track (e.g. 'transcription2plaintext')
        and/or source (e.g. 'borg').
        """
        path = Path(split_file)
        if not path.is_absolute():
            path = self.root / "splits" / path
        tests = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                t = BenchmarkTest(
                    test_id=data["test_id"],
                    track=data["track"],
                    cipher_system=data.get("cipher_system", ""),
                    target_records=data.get("target_records", []),
                    context_records=data.get("context_records", []),
                    description=data.get("description", ""),
                )
                if track and t.track != track:
                    continue
                if source and not t.test_id.startswith(source):
                    continue
                tests.append(t)
        return tests

    def load_test_data(self, test: BenchmarkTest) -> TestData:
        """Load the actual transcription and plaintext data for a test."""
        canonical, plaintext, target_records = self._load_record_texts(test.target_records)
        context_canonical, context_plaintext, context_records = self._load_record_texts(
            test.context_records
        )

        # Try to load symbol map
        source = test.test_id.split("_")[0]  # e.g. "borg" from "borg_single_B_..."
        symbol_map = self._load_symbol_map(source)

        return TestData(
            test=test,
            canonical_transcription=canonical,
            plaintext=plaintext,
            plaintext_language=_resolve_plaintext_language(target_records),
            context_canonical_transcription=context_canonical,
            context_plaintext=context_plaintext,
            context_records=context_records,
            symbol_map=symbol_map,
        )

    def _load_record_texts(
        self,
        record_ids: list[str],
    ) -> tuple[str, str, list[BenchmarkRecord]]:
        canonical_parts = []
        plaintext_parts = []
        records = []

        for record_id in record_ids:
            record = self._manifest.get(record_id)
            if record is None:
                raise ValueError(f"Record '{record_id}' not found in manifest")
            records.append(record)

            if record.transcription_canonical_file:
                canon_path = self.root / record.transcription_canonical_file
                if canon_path.exists():
                    canonical_parts.append(canon_path.read_text().strip())

            if record.plaintext_file:
                pt_path = self.root / record.plaintext_file
                if pt_path.exists():
                    plaintext_parts.append(pt_path.read_text().strip())

        return "\n".join(canonical_parts), "\n".join(plaintext_parts), records

    def _load_symbol_map(self, source: str) -> dict | None:
        """Load symbol map metadata if available."""
        map_path = self.root / "sources" / source / "metadata" / f"{source}_symbol_map.json"
        if map_path.exists():
            with open(map_path) as f:
                return json.load(f)
        return None

    def get_record(self, record_id: str) -> BenchmarkRecord | None:
        return self._manifest.get(record_id)


def parse_canonical_transcription(canonical_text: str) -> CipherText:
    """Parse a canonical transcription (space-separated S-tokens, | word separators)
    into a CipherText object.

    Format: S025 S012 S006 | S003 S007 S012 S019 | S005 S009 S010 S009
    """
    lines = canonical_text.strip().split("\n")
    full = " | ".join(line.strip() for line in lines if line.strip())
    word_strings = full.split(" | ")

    seen: set[str] = set()
    symbols: list[str] = []
    for word_str in word_strings:
        for token in word_str.split():
            if token not in seen:
                seen.add(token)
                symbols.append(token)

    alphabet = Alphabet(symbols)
    raw = " | ".join(word_strings)
    return CipherText(raw=raw, alphabet=alphabet, source="benchmark", separator=" | ")


def resolve_test_language(test_data: TestData, default_language: str | None = None) -> str:
    """Resolve the intended plaintext language for a test.

    Preference order:
    1. Explicit caller override
    2. Benchmark manifest language on target records
    3. Legacy test-id/source heuristic
    """
    if default_language:
        return default_language
    if test_data.plaintext_language:
        return test_data.plaintext_language
    source = test_data.test.test_id.split("_")[0]
    return {
        "borg": "la",
        "copiale": "de",
        "it": "it",
        "fr": "fr",
    }.get(source, "en")


def _resolve_plaintext_language(records: list[BenchmarkRecord]) -> str:
    languages = [record.plaintext_language for record in records if record.plaintext_language]
    if not languages:
        return ""
    counts: dict[str, int] = {}
    for language in languages:
        counts[language] = counts.get(language, 0) + 1
    return max(sorted(counts), key=lambda language: counts[language])
