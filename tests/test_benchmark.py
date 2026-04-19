"""Tests for benchmark loader, runner utilities, and scorer."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from benchmark.loader import BenchmarkLoader, BenchmarkTest, parse_canonical_transcription
from benchmark.scorer import (
    ScoreResult,
    _collapse_spaced_letters,
    format_report,
    normalize_text,
    score_decryption,
)


# --- Scorer tests ---

class TestNormalizeText:
    def test_uppercase(self):
        assert normalize_text("hello world") == "HELLO WORLD"

    def test_strip_brackets(self):
        assert normalize_text("ret[ulit]") == "RET"

    def test_strip_angle_annotations(self):
        assert normalize_text("dolo<=r>is") == "DOLOIS"

    def test_collapse_whitespace(self):
        assert normalize_text("  foo   bar  ") == "FOO BAR"

    def test_strip_question_mark(self):
        assert normalize_text("word(?) other") == "WORD OTHER"


class TestCollapseSpacedLetters:
    def test_basic(self):
        assert _collapse_spaced_letters("Q U E D A M") == "QUEDAM"

    def test_mixed(self):
        assert _collapse_spaced_letters("Q U E D A M something A N U S") == "QUEDAM something ANUS"

    def test_no_change(self):
        assert _collapse_spaced_letters("HELLO WORLD") == "HELLO WORLD"

    def test_two_letters_not_collapsed(self):
        # Only 2 single letters — not collapsed (need 3+)
        assert _collapse_spaced_letters("A B") == "A B"

    def test_three_letters_collapsed(self):
        assert _collapse_spaced_letters("A B C") == "ABC"


class TestScoreDecryption:
    def test_perfect_match(self):
        result = score_decryption(
            "t1", "HELLO WORLD", "hello world", 1.0, "solved",
        )
        assert result.char_accuracy == 1.0
        assert result.word_accuracy == 1.0

    def test_multisym_output(self):
        result = score_decryption(
            "t2", "Q U E D A M | A N U S", "quedam anus", 0.5, "solved",
        )
        assert result.char_accuracy == 1.0
        assert result.word_accuracy == 1.0

    def test_partial_match(self):
        result = score_decryption(
            "t3", "HELLX WORLD", "hello world", 0.5, "stuck",
        )
        assert result.char_accuracy < 1.0
        assert result.word_accuracy < 1.0

    def test_with_annotations(self):
        result = score_decryption(
            "t4", "R E T", "ret[ulit]", 0.5, "solved",
        )
        assert result.char_accuracy == 1.0


class TestFormatReport:
    def test_basic(self):
        scores = [
            ScoreResult("test_1", 0.9, 0.8, 100, 90, 10, 8, 0.7, "solved"),
            ScoreResult("test_2", 0.5, 0.3, 100, 50, 10, 3, 0.4, "stuck"),
        ]
        report = format_report(scores)
        assert "BENCHMARK RESULTS" in report
        assert "test_1" in report
        assert "test_2" in report
        assert "AVERAGE" in report


# --- Parser tests ---

class TestParseCanonicalTranscription:
    def test_basic(self):
        text = "S001 S002 S003 | S004 S005"
        ct = parse_canonical_transcription(text)
        assert ct.alphabet.size == 5
        assert len(ct.words) == 2
        assert len(ct.words[0]) == 3
        assert len(ct.words[1]) == 2
        assert ct.separator == " | "

    def test_multiline(self):
        text = "S001 S002 | S003\nS004 | S005 S006"
        ct = parse_canonical_transcription(text)
        # Lines joined with |: "S001 S002 | S003 | S004 | S005 S006"
        assert len(ct.words) == 4
        assert ct.alphabet.size == 6

    def test_token_encoding(self):
        text = "S001 S002 | S001 S003"
        ct = parse_canonical_transcription(text)
        assert ct.alphabet.size == 3  # S001, S002, S003
        assert len(ct.tokens) == 4  # S001, S002, S001, S003


# --- Loader tests (with temp fixture) ---

@pytest.fixture
def benchmark_dir():
    """Create a minimal benchmark directory structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure
        os.makedirs(os.path.join(tmpdir, "manifest"))
        os.makedirs(os.path.join(tmpdir, "splits"))
        os.makedirs(os.path.join(tmpdir, "sources", "test", "transcriptions"))
        os.makedirs(os.path.join(tmpdir, "sources", "test", "plaintext"))

        # Write manifest
        record = {
            "id": "test_page1",
            "source": "test",
            "cipher_type": ["monoalphabetic_substitution"],
            "plaintext_language": "en",
            "transcription_canonical_file": "sources/test/transcriptions/page1.canonical.txt",
            "plaintext_file": "sources/test/plaintext/page1.txt",
            "has_key": True,
        }
        with open(os.path.join(tmpdir, "manifest", "records.jsonl"), "w") as f:
            f.write(json.dumps(record) + "\n")

        # Write split
        test_def = {
            "test_id": "test_single_B_page1",
            "track": "transcription2plaintext",
            "cipher_system": "test_cipher",
            "target_records": ["test_page1"],
            "context_records": [],
            "description": "Test B: page1",
        }
        with open(os.path.join(tmpdir, "splits", "test_tests.jsonl"), "w") as f:
            f.write(json.dumps(test_def) + "\n")

        # Write canonical transcription
        with open(os.path.join(tmpdir, "sources", "test", "transcriptions", "page1.canonical.txt"), "w") as f:
            f.write("S001 S002 S003 | S004 S005 S006\n")

        # Write plaintext
        with open(os.path.join(tmpdir, "sources", "test", "plaintext", "page1.txt"), "w") as f:
            f.write("hello world\n")

        yield tmpdir


class TestBenchmarkLoader:
    def test_load_manifest(self, benchmark_dir):
        loader = BenchmarkLoader(benchmark_dir)
        record = loader.get_record("test_page1")
        assert record is not None
        assert record.source == "test"

    def test_load_tests(self, benchmark_dir):
        loader = BenchmarkLoader(benchmark_dir)
        tests = loader.load_tests("test_tests.jsonl")
        assert len(tests) == 1
        assert tests[0].test_id == "test_single_B_page1"

    def test_filter_by_track(self, benchmark_dir):
        loader = BenchmarkLoader(benchmark_dir)
        tests = loader.load_tests("test_tests.jsonl", track="image2transcription")
        assert len(tests) == 0

    def test_load_test_data(self, benchmark_dir):
        loader = BenchmarkLoader(benchmark_dir)
        tests = loader.load_tests("test_tests.jsonl")
        td = loader.load_test_data(tests[0])
        assert "S001" in td.canonical_transcription
        assert "hello" in td.plaintext
