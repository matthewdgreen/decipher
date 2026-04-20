from __future__ import annotations

import json
import sys

from benchmark.loader import BenchmarkTest, TestData as BenchmarkTestData
from external_baselines.harness import (
    ExternalBaselineConfig,
    extract_candidates,
    prepare_external_case,
    run_external_baseline,
)


def _test_data(plaintext: str = "THEQUICKBROWNFOX") -> BenchmarkTestData:
    return BenchmarkTestData(
        test=BenchmarkTest(
            test_id="external_demo",
            track="transcription2plaintext",
            cipher_system="homophonic_substitution",
            target_records=[],
            context_records=[],
            description="external baseline demo",
        ),
        canonical_transcription="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16",
        plaintext=plaintext,
        symbol_map=None,
    )


def test_prepare_external_case_writes_neutral_inputs(tmp_path):
    prepared = prepare_external_case(_test_data(), tmp_path, "zenith")

    assert prepared.tokens_file.read_text(encoding="utf-8").strip().startswith("01 02 03")
    assert prepared.compact_file.read_text(encoding="utf-8").strip().startswith("010203")
    assert prepared.letters_file.read_text(encoding="utf-8").strip().startswith("ABC")
    metadata = json.loads(prepared.metadata_file.read_text(encoding="utf-8"))
    assert metadata["cipher_system"] == "homophonic_substitution"
    assert metadata["symbol_count"] == 16


def test_extract_candidates_prefers_labeled_solution():
    candidates = extract_candidates("noise\nPlaintext: THE QUICK BROWN FOX\n")

    assert candidates == ["THE QUICK BROWN FOX"]


def test_run_external_baseline_scores_stdout_solution(tmp_path):
    code = "print('SOLUTION: THEQUICKBROWNFOX')"
    config = ExternalBaselineConfig(
        name="dummy",
        command=[sys.executable, "-c", code],
        timeout_seconds=5,
    )

    result = run_external_baseline(_test_data(), config, artifact_dir=tmp_path)

    assert result.status == "solved"
    assert result.char_accuracy == 1.0
    assert result.candidates_considered == 1
    assert result.artifact_path


def test_run_external_baseline_supports_output_file_placeholder(tmp_path):
    code = (
        "import pathlib, sys; "
        "pathlib.Path(sys.argv[1]).write_text('PLAINTEXT: THEQUICKBROWNFOX\\n')"
    )
    config = ExternalBaselineConfig(
        name="dummy-file",
        command=[sys.executable, "-c", code, "{output_file}"],
        timeout_seconds=5,
    )

    result = run_external_baseline(_test_data(), config, artifact_dir=tmp_path)

    assert result.status == "solved"
    assert result.char_accuracy == 1.0
