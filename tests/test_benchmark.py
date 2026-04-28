"""Tests for benchmark loader, runner utilities, and scorer."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from benchmark.context import build_benchmark_context, safe_read_benchmark_file
from benchmark.loader import BenchmarkLoader, BenchmarkTest, parse_canonical_transcription
from benchmark.scorer import (
    ScoreResult,
    _collapse_spaced_letters,
    align_char_sequences,
    align_word_sequences,
    format_char_diff,
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

    def test_char_score_resynchronizes_after_insertion(self):
        result = score_decryption(
            "t5",
            "ABCDEFXGHIJK",
            "ABCDEFGHIJK",
            0.5,
            "completed",
        )

        assert result.correct_chars == 11
        assert result.total_chars == 12
        assert result.char_accuracy == pytest.approx(11 / 12)

    def test_char_alignment_keeps_substitutions_as_errors(self):
        rows = align_char_sequences("ABCXDEF", "ABCYDEF")

        assert [row.op for row in rows] == [
            "match",
            "match",
            "match",
            "substitute",
            "match",
            "match",
            "match",
        ]

    def test_char_diff_reports_gaps_from_alignment(self):
        diff = format_char_diff("ABCDEFXGHIJK", "ABCDEFGHIJK", context=2)

        assert "character alignment error" in diff
        assert "[X]" in diff
        assert "[-]" in diff

    def test_word_score_resynchronizes_after_extra_words(self):
        result = score_decryption(
            "t6",
            "ETIAM QUOD IN TALI CUR PLICARE U UEL AC PULLO ET BREUITER",
            "ETIAM QUOD IN TALI CUR PLICARE UEL PULLO ET BREUITER",
            0.5,
            "completed",
        )

        assert result.correct_words == 10
        assert result.total_words == 12
        assert result.word_accuracy == pytest.approx(10 / 12)

    def test_word_alignment_marks_insertions_and_resyncs(self):
        rows = align_word_sequences(
            ["PULLO", "ET", "AC", "BREUITER", "UT"],
            ["PULLO", "ET", "BREUITER", "UT"],
        )

        assert [row.op for row in rows] == [
            "match",
            "match",
            "insert",
            "match",
            "match",
        ]


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
        os.makedirs(os.path.join(tmpdir, "sources", "test", "documents"))
        os.makedirs(os.path.join(tmpdir, "sources", "test", "metadata"))
        os.makedirs(os.path.join(tmpdir, "sources", "test", "transcriptions"))
        os.makedirs(os.path.join(tmpdir, "sources", "test", "plaintext"))
        os.makedirs(os.path.join(tmpdir, "sources", "zodiac", "metadata"))

        # Write manifest
        record = {
            "id": "test_page1",
            "source": "test",
            "cipher_type": ["monoalphabetic_substitution"],
            "plaintext_language": "en",
            "transcription_canonical_file": "sources/test/transcriptions/page1.canonical.txt",
            "plaintext_file": "sources/test/plaintext/page1.txt",
            "has_key": True,
            "context_layers": {
                "minimal": {
                    "label": "Basic provenance",
                    "text": "A short handwritten test cipher.",
                },
                "standard": {
                    "label": "Standard context",
                    "text": "The note is from the test archive.",
                },
                "historical": {
                    "label": "Historical context",
                    "text": "The writer also produced a solved companion note.",
                    "contains_plaintext_hint": True,
                },
            },
            "related_records": [
                {
                    "record_id": "test_page2",
                    "relationship": "same_writer_solved_example",
                    "solution_available": True,
                    "safe_context_layers": ["related_solutions", "max"],
                }
            ],
            "associated_documents": [
                {
                    "id": "test_note",
                    "document_type": "letter",
                    "title": "Plaintext covering note",
                    "summary": "A short note accompanying the cipher.",
                    "text_file": "sources/test/documents/test_note.txt",
                    "contains_plaintext_hint": True,
                    "safe_context_layers": ["historical", "related_metadata", "max"],
                },
                {
                    "id": "test_solution_note",
                    "document_type": "solution_note",
                    "title": "Solution-bearing note",
                    "summary": "A deliberately gated note.",
                    "text_file": "sources/test/documents/test_solution_note.txt",
                    "contains_solution": True,
                    "safe_context_layers": ["related_solutions", "max"],
                },
            ],
        }
        context_record = {
            "id": "test_page2",
            "source": "test",
            "cipher_type": ["monoalphabetic_substitution"],
            "plaintext_language": "en",
            "transcription_canonical_file": "sources/test/transcriptions/page2.canonical.txt",
            "plaintext_file": "sources/test/plaintext/page2.txt",
            "has_key": True,
        }
        with open(os.path.join(tmpdir, "manifest", "records.jsonl"), "w") as f:
            f.write(json.dumps(record) + "\n")
            f.write(json.dumps(context_record) + "\n")

        # Write split
        test_def = {
            "test_id": "test_single_B_page1",
            "track": "transcription2plaintext",
            "cipher_system": "test_cipher",
            "target_records": ["test_page1"],
            "context_records": ["test_page2"],
            "description": "Test B: page1",
        }
        with open(os.path.join(tmpdir, "splits", "test_tests.jsonl"), "w") as f:
            f.write(json.dumps(test_def) + "\n")

        # Write canonical transcription
        with open(os.path.join(tmpdir, "sources", "test", "transcriptions", "page1.canonical.txt"), "w") as f:
            f.write("S001 S002 S003 | S004 S005 S006\n")
        with open(os.path.join(tmpdir, "sources", "test", "transcriptions", "page2.canonical.txt"), "w") as f:
            f.write("S007 S008 S009\n")

        # Write plaintext
        with open(os.path.join(tmpdir, "sources", "test", "plaintext", "page1.txt"), "w") as f:
            f.write("hello world\n")
        with open(os.path.join(tmpdir, "sources", "test", "plaintext", "page2.txt"), "w") as f:
            f.write("context page\n")
        with open(os.path.join(tmpdir, "sources", "test", "documents", "test_note.txt"), "w") as f:
            f.write("The note says this was copied from a handwritten source.\n")
        with open(os.path.join(tmpdir, "sources", "test", "documents", "test_solution_note.txt"), "w") as f:
            f.write("The solution-bearing note is only for max-context runs.\n")
        with open(os.path.join(tmpdir, "sources", "test", "metadata", "test_symbol_map.json"), "w") as f:
            json.dump({"source": "test", "symbols": {"S001": {"glyph_id": "TEST_A"}}}, f)
        with open(os.path.join(tmpdir, "sources", "zodiac", "metadata", "zodiac_symbol_map.json"), "w") as f:
            json.dump({"source": "zodiac", "symbols": {"S001": {"glyph_id": "ZODIAC_A"}}}, f)

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
        assert "S007" in td.context_canonical_transcription
        assert "context" in td.context_plaintext
        assert [r.id for r in td.context_records] == ["test_page2"]
        assert [r.id for r in td.target_records] == ["test_page1"]
        assert td.benchmark_root == str(Path(benchmark_dir).resolve())
        assert td.symbol_map["source"] == "test"

    def test_symbol_map_uses_record_source_not_test_id_prefix(self, benchmark_dir):
        loader = BenchmarkLoader(benchmark_dir)
        test = BenchmarkTest(
            test_id="zodiac_named_fixture_but_test_source",
            track="transcription2plaintext",
            cipher_system="test_cipher",
            target_records=["test_page1"],
            context_records=[],
            description="test source under a misleading test id",
        )

        td = loader.load_test_data(test)

        assert td.symbol_map["source"] == "test"
        assert td.symbol_map["symbols"]["S001"]["glyph_id"] == "TEST_A"

    def test_build_benchmark_context_max_policy(self, benchmark_dir):
        loader = BenchmarkLoader(benchmark_dir)
        test = loader.load_tests("test_tests.jsonl")[0]
        td = loader.load_test_data(test)

        context = build_benchmark_context(td, policy="max")

        assert context is not None
        assert context.policy == "max"
        assert context.target_record_ids == ["test_page1"]
        assert context.context_record_ids == ["test_page2"]
        assert "test_page2" in context.related_records
        assert "test_note" in context.associated_documents
        assert "test_solution_note" in context.associated_documents
        assert "Basic provenance" in context.prompt
        assert "Scoped benchmark tools are available" in context.prompt
        assert context.related_solution_allowed is True

    def test_build_benchmark_context_none_policy_is_empty_but_audited(self, benchmark_dir):
        loader = BenchmarkLoader(benchmark_dir)
        test = loader.load_tests("test_tests.jsonl")[0]
        td = loader.load_test_data(test)

        context = build_benchmark_context(td, policy="none")

        assert context is not None
        assert context.prompt == ""
        assert context.injected_layers == []
        assert context.related_records == {}
        assert context.to_artifact_dict()["policy"] == "none"

    def test_build_benchmark_context_hides_unavailable_documents_by_policy(self, benchmark_dir):
        loader = BenchmarkLoader(benchmark_dir)
        test = loader.load_tests("test_tests.jsonl")[0]
        td = loader.load_test_data(test)

        minimal = build_benchmark_context(td, policy="minimal")
        historical = build_benchmark_context(td, policy="historical")

        assert minimal.associated_documents == {}
        assert set(historical.associated_documents) == {"test_note"}

    def test_safe_read_benchmark_file_rejects_path_escape(self, benchmark_dir):
        with pytest.raises(ValueError, match="escapes benchmark root"):
            safe_read_benchmark_file(benchmark_dir, "../outside.txt")

    def test_english_borg_analog_fixture_loads_and_has_boundary_drift(self):
        root = Path(__file__).resolve().parent.parent / "fixtures" / "benchmarks" / "english_borg_analog"
        loader = BenchmarkLoader(root)
        tests = loader.load_tests("english_borg_analog.jsonl")
        assert [t.test_id for t in tests] == ["english_borg_analog_001"]

        td = loader.load_test_data(tests[0])
        ct = parse_canonical_transcription(td.canonical_transcription)
        assert td.plaintext_language == "en"
        assert len(ct.tokens) == 125
        assert len(ct.words) == 35

        source_boundary_decode = (
            "THERE | FORE | THE | OLD | PHYSICK | ER | DID | AP | PLY | A | "
            "SALVE | UN | TO | THE | SORE | HEN | AND | THE | FOWL | DID | "
            "LIVE | AFTER | WARD | HE | WROTE | THAT | MANY | SUCH | CURES | "
            "WERE | SWEET | AND | WITH | OUT | PAIN"
        )
        score = score_decryption(
            "english_borg_analog_001",
            source_boundary_decode,
            td.plaintext,
            agent_score=0.0,
            status="completed",
        )
        assert score.char_accuracy == 1.0
        assert 0.5 < score.word_accuracy < 1.0
