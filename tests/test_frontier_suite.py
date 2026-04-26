from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmark.loader import BenchmarkLoader, BenchmarkTest
from frontier.suite import (
    evaluate_frontier_rows,
    load_frontier_suite,
    nominate_frontier_candidates,
    resolve_frontier_case,
)
from testgen.cache import PlaintextCache
from testgen.spec import TestSpec as SyntheticSpec


RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_frontier_suite.py"
runner_spec = importlib.util.spec_from_file_location("run_frontier_suite", RUNNER_PATH)
assert runner_spec is not None and runner_spec.loader is not None
frontier_runner = importlib.util.module_from_spec(runner_spec)
sys.modules[runner_spec.name] = frontier_runner
runner_spec.loader.exec_module(frontier_runner)


def test_load_frontier_suite_accepts_benchmark_and_synthetic_cases(tmp_path):
    suite = tmp_path / "frontier.jsonl"
    suite.write_text(
        json.dumps({
            "test_id": "bench_case",
            "track": "transcription2plaintext",
            "cipher_system": "simple_substitution",
            "target_records": ["rec1"],
            "context_records": [],
            "description": "benchmark-backed",
            "frontier_class": "known_good",
        })
        + "\n"
        + json.dumps({
            "test_id": "synth_case",
            "track": "transcription2plaintext",
            "cipher_system": "homophonic_substitution",
            "target_records": [],
            "context_records": [],
            "description": "synthetic",
            "frontier_class": "slow_result",
            "synthetic_spec": {
                "language": "en",
                "approx_length": 80,
                "word_boundaries": False,
                "homophonic": True,
                "seed": 2,
                "topic": "general",
                "frequency_style": "normal",
            },
        })
        + "\n",
        encoding="utf-8",
    )

    cases = load_frontier_suite(suite)

    assert len(cases) == 2
    assert cases[0].synthetic_spec is None
    assert cases[1].synthetic_spec is not None
    assert cases[1].synthetic_spec.seed == 2


def test_load_frontier_suite_rejects_invalid_entries(tmp_path):
    suite = tmp_path / "bad.jsonl"
    suite.write_text(
        json.dumps({
            "test_id": "bad_case",
            "track": "transcription2plaintext",
            "cipher_system": "simple_substitution",
            "target_records": [],
            "context_records": [],
            "description": "invalid",
            "frontier_class": "unknown",
        })
        + "\n",
        encoding="utf-8",
    )

    try:
        load_frontier_suite(suite)
    except ValueError as exc:
        assert "invalid frontier_class" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_homophonic_profile_ablation_suite_loads():
    suite = Path(__file__).resolve().parents[1] / "frontier" / "homophonic_profile_ablation.jsonl"

    cases = load_frontier_suite(suite)

    assert [case.test.test_id for case in cases] == [
        "synth_en_150honb_s1",
        "synth_en_150honb_s3",
        "synth_en_80honb_s1",
        "synth_en_80honb_s6",
        "parity_tool_zenith_zodiac408",
    ]
    assert all("ablation_packet" in case.frontier_tags for case in cases)
    assert sum(1 for case in cases if case.synthetic_spec is not None) == 4
    assert cases[-1].synthetic_spec is None


def test_english_model_comparison_suite_loads():
    suite = Path(__file__).resolve().parents[1] / "frontier" / "english_model_comparison.jsonl"

    cases = load_frontier_suite(suite)

    assert [case.test.test_id for case in cases] == [
        "synth_en_150honb_s1",
        "synth_en_150honb_s3",
        "synth_en_80honb_s1",
        "synth_en_80honb_s2",
        "synth_en_80honb_s3",
        "synth_en_80honb_s5",
        "synth_en_80honb_s6",
        "parity_tool_zenith_zodiac408",
    ]
    assert all("english_model_comparison" in case.frontier_tags for case in cases)
    assert sum(1 for case in cases if case.synthetic_spec is not None) == 7
    assert cases[-1].synthetic_spec is None


def test_real_english_challenge_suite_loads():
    suite = Path(__file__).resolve().parents[1] / "frontier" / "real_english_challenge.jsonl"

    cases = load_frontier_suite(suite)

    assert [case.test.test_id for case in cases] == [
        "parity_tool_zenith_goldbug",
        "parity_tool_zenith_horacemann",
        "parity_tool_zenith_zodiac408",
    ]
    assert all("real_english_challenge" in case.frontier_tags for case in cases)
    assert all(case.synthetic_spec is None for case in cases)
    assert cases[0].min_char_accuracy_by_solver["decipher-automated"] == 0.95
    assert cases[1].min_char_accuracy_by_solver["decipher-automated"] == 0.70
    assert cases[2].min_char_accuracy_by_solver["decipher-automated"] == 0.95


def test_automated_solver_frontier_suite_loads_shared_hard_cases():
    suite = Path(__file__).resolve().parents[1] / "frontier" / "automated_solver_frontier.jsonl"

    cases = load_frontier_suite(suite)
    by_test = {case.test.test_id: case for case in cases}

    assert "synth_en_80honb_s2" in by_test
    assert "synth_en_200honb_s3" in by_test
    assert "synth_en_200honb_s6" in by_test
    assert by_test["synth_en_80honb_s2"].frontier_class == "shared_hard"
    assert by_test["synth_en_200honb_s3"].frontier_class == "shared_hard"
    assert by_test["synth_en_200honb_s6"].frontier_class == "shared_hard"
    assert "shared_hard" in by_test["synth_en_80honb_s2"].frontier_tags


def test_evaluate_frontier_rows_applies_thresholds_and_gap_checks():
    rows = [
        {
            "test_id": "case1",
            "solver": "decipher-automated",
            "status": "solved",
            "char_accuracy": 0.7,
            "elapsed_seconds": 40.0,
            "expected_status_by_solver": {},
            "min_char_accuracy_by_solver": {"decipher-automated": 0.9},
            "max_elapsed_seconds_by_solver": {"decipher-automated": 20.0},
            "max_gap_vs_solver": {"zenith": 0.05},
        },
        {
            "test_id": "case1",
            "solver": "zenith",
            "status": "completed",
            "char_accuracy": 0.95,
            "elapsed_seconds": 10.0,
            "expected_status_by_solver": {},
            "min_char_accuracy_by_solver": {},
            "max_elapsed_seconds_by_solver": {},
            "max_gap_vs_solver": {},
        },
    ]

    evaluated = evaluate_frontier_rows(rows)
    decipher = next(row for row in evaluated if row["solver"] == "decipher-automated")

    assert decipher["meets_expectations"] is False
    assert any("below" in item for item in decipher["expectation_failures"])
    assert any("above" in item for item in decipher["expectation_failures"])
    assert any("gap_vs_zenith" in item for item in decipher["expectation_failures"])


def test_nominate_frontier_candidates_deduplicates_and_classifies():
    rows = [
        {
            "test_id": "parity_tool_zenith_zodiac408",
            "solver": "decipher-automated",
            "status": "solved",
            "char_accuracy": 0.66,
            "elapsed_seconds": 460.0,
            "family": "zodiac408",
        },
        {
            "test_id": "parity_tool_zenith_zodiac408",
            "solver": "zenith",
            "status": "completed",
            "char_accuracy": 0.99,
            "elapsed_seconds": 10.0,
            "family": "zodiac408",
        },
        {
            "test_id": "synth_en_80honb_s1",
            "solver": "decipher-automated",
            "status": "solved",
            "char_accuracy": 0.98,
            "elapsed_seconds": 480.0,
            "family": "homophonic_substitution",
        },
        {
            "test_id": "synth_en_80honb_s1",
            "solver": "decipher-automated",
            "status": "solved",
            "char_accuracy": 0.97,
            "elapsed_seconds": 800.0,
            "family": "homophonic_substitution",
        },
    ]

    nominations = nominate_frontier_candidates(rows)
    by_test = {item["test_id"]: item for item in nominations}

    assert by_test["parity_tool_zenith_zodiac408"]["frontier_class"] == "bad_result"
    assert by_test["synth_en_80honb_s1"]["frontier_class"] == "slow_result"
    assert by_test["synth_en_80honb_s1"]["decipher_elapsed_seconds"] == 480.0


def test_nominate_frontier_candidates_marks_shared_hard_band():
    rows = [
        {
            "test_id": "synth_en_200honb_s3",
            "solver": "decipher-automated",
            "status": "completed",
            "char_accuracy": 0.992063,
            "elapsed_seconds": 70.0,
            "family": "homophonic_substitution",
        },
        {
            "test_id": "synth_en_200honb_s3",
            "solver": "zenith",
            "status": "completed",
            "char_accuracy": 0.989087,
            "elapsed_seconds": 13.1,
            "family": "homophonic_substitution",
        },
    ]

    nominations = nominate_frontier_candidates(rows)

    assert nominations[0]["test_id"] == "synth_en_200honb_s3"
    assert nominations[0]["frontier_class"] == "shared_hard"
    assert "shared_hard" in nominations[0]["frontier_tags"]


def test_frontier_runner_writes_summary_and_continues_after_external_exception(monkeypatch, tmp_path):
    suite = tmp_path / "frontier.jsonl"
    suite.write_text(
        json.dumps({
            "test_id": "synth_demo",
            "track": "transcription2plaintext",
            "cipher_system": "simple_substitution",
            "target_records": [],
            "context_records": [],
            "description": "synthetic demo",
            "frontier_class": "known_good",
            "expected_solvers": ["decipher-automated", "zkdecrypto-lite", "zenith"],
            "synthetic_spec": {
                "language": "en",
                "approx_length": 80,
                "word_boundaries": True,
                "homophonic": False,
                "seed": 1,
                "topic": "general",
                "frequency_style": "normal",
            },
        })
        + "\n",
        encoding="utf-8",
    )

    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "manifest").mkdir(parents=True)
    (benchmark_root / "manifest" / "records.jsonl").write_text("", encoding="utf-8")
    cache = PlaintextCache(tmp_path / "cache")
    cache.put(
        SyntheticSpec(language="en", approx_length=80, word_boundaries=True, homophonic=False, seed=1),
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    )

    seen: dict[str, str] = {}

    class FakeRunner:
        def run_test(self, test_data, language=None):
            return SimpleNamespace(
                status="solved",
                char_accuracy=1.0,
                word_accuracy=1.0,
                artifact_path="decipher.json",
                error_message="",
            )

    class FakeConfig:
        def __init__(self, name):
            self.name = name

    def fake_automated_runner(artifact_dir, homophonic_budget="full", homophonic_refinement="none", homophonic_solver="zenith_native"):
        seen["homophonic_budget"] = homophonic_budget
        return FakeRunner()

    monkeypatch.setattr(frontier_runner, "AutomatedBenchmarkRunner", fake_automated_runner)
    monkeypatch.setattr(frontier_runner, "_load_external_configs", lambda path, oracle: [FakeConfig("zkdecrypto-lite-quick"), FakeConfig("zenith-quick")])

    def fake_external(test_data, config, artifact_dir):
        if config.name == "zkdecrypto-lite":
            raise AssertionError("canonical name should not reach wrapper")
        if config.name == "zkdecrypto-lite-quick":
            raise ValueError("symbol alphabet overflow")
        return SimpleNamespace(
            status="completed",
            char_accuracy=0.9,
            word_accuracy=0.0,
            elapsed=1.5,
            artifact_path="zenith.json",
            error="",
            candidates_considered=1,
        )

    monkeypatch.setattr(frontier_runner, "run_external_baseline", fake_external)
    dummy_config = tmp_path / "dummy.json"
    dummy_config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_suite.py",
            "--suite-file", str(suite),
            "--solvers", "decipher", "external",
            "--benchmark-root", str(benchmark_root),
            "--cache-dir", str(tmp_path / "cache"),
            "--external-config", str(dummy_config),
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--summary-jsonl", str(tmp_path / "summary.jsonl"),
            "--summary-csv", str(tmp_path / "summary.csv"),
        ],
    )

    frontier_runner.main()

    rows = [
        json.loads(line)
        for line in (tmp_path / "summary.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 3
    by_solver = {row["solver"]: row for row in rows}
    assert by_solver["decipher-automated"]["status"] == "solved"
    assert by_solver["zkdecrypto-lite-quick"]["status"] == "failed"
    assert by_solver["zenith-quick"]["status"] == "completed"
    assert by_solver["zkdecrypto-lite-quick"]["solver_key"] == "zkdecrypto-lite"
    assert by_solver["zenith-quick"]["solver_key"] == "zenith"
    assert seen["homophonic_budget"] == "full"


def test_dashboard_accepts_local_metadata_paths(tmp_path):
    benchmark = tmp_path / "benchmark"
    (benchmark / "splits").mkdir(parents=True)
    (benchmark / "splits" / "parity.jsonl").write_text("", encoding="utf-8")

    frontier = tmp_path / "frontier.jsonl"
    frontier.write_text(
        json.dumps({
            "test_id": "case1",
            "track": "transcription2plaintext",
            "cipher_system": "simple_substitution",
            "target_records": ["rec1"],
            "context_records": [],
            "description": "frontier",
            "frontier_class": "known_good",
            "frontier_tags": ["quick"],
        })
        + "\n",
        encoding="utf-8",
    )

    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({
            "cipher_id": "case1",
            "cipher_alphabet_size": 26,
            "cipher_word_count": 1,
            "status": "solved",
            "char_accuracy": 1.0,
            "branches": [{"name": "main", "char_accuracy": 1.0}],
            "solution": {"branch": "main", "declared_at_iteration": 1},
            "tool_calls": [],
        }),
        encoding="utf-8",
    )

    from artifact.dashboard import build_dashboard

    rows = build_dashboard([artifact], benchmark_root=benchmark, metadata_paths=[frontier])
    assert rows[0].family == "known_good"
    assert rows[0].metadata["frontier_tags"] == ["quick"]


def test_frontier_runner_passes_manifest_language_for_benchmark_backed_case(monkeypatch, tmp_path):
    suite = tmp_path / "frontier.jsonl"
    suite.write_text(
        json.dumps({
            "test_id": "parity_borg_demo",
            "track": "transcription2plaintext",
            "cipher_system": "borg_lat_898",
            "target_records": ["rec1"],
            "context_records": [],
            "description": "historical benchmark-backed case",
            "frontier_class": "bad_result",
            "expected_solvers": ["decipher-automated"],
        })
        + "\n",
        encoding="utf-8",
    )

    benchmark = tmp_path / "benchmark"
    (benchmark / "manifest").mkdir(parents=True)
    (benchmark / "data").mkdir(parents=True)
    (benchmark / "manifest" / "records.jsonl").write_text(
        json.dumps({
            "id": "rec1",
            "source": "borg",
            "cipher_type": ["substitution"],
            "plaintext_language": "la",
            "transcription_canonical_file": "data/rec1.txt",
            "plaintext_file": "data/rec1_plain.txt",
            "has_key": False,
        })
        + "\n",
        encoding="utf-8",
    )
    (benchmark / "data" / "rec1.txt").write_text("S001 S002 S003", encoding="utf-8")
    (benchmark / "data" / "rec1_plain.txt").write_text("EST", encoding="utf-8")

    seen: dict[str, str | None] = {}

    class FakeRunner:
        def run_test(self, test_data, language=None):
            seen["language"] = language
            return SimpleNamespace(
                status="completed",
                char_accuracy=0.1,
                word_accuracy=0.0,
                artifact_path="decipher.json",
                error_message="",
            )

        monkeypatch.setattr(
            frontier_runner,
            "AutomatedBenchmarkRunner",
            lambda artifact_dir, homophonic_budget="full", homophonic_refinement="none", homophonic_solver="zenith_native": FakeRunner(),
        )
    monkeypatch.setattr(frontier_runner, "_load_external_configs", lambda path, oracle: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_suite.py",
            "--suite-file", str(suite),
            "--solvers", "decipher",
            "--benchmark-root", str(benchmark),
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--summary-jsonl", str(tmp_path / "summary.jsonl"),
            "--summary-csv", str(tmp_path / "summary.csv"),
        ],
    )

    frontier_runner.main()

    assert seen["language"] == "la"


def test_frontier_runner_passes_homophonic_budget(monkeypatch, tmp_path):
    suite = tmp_path / "frontier.jsonl"
    suite.write_text(
        json.dumps({
            "test_id": "synth_demo",
            "track": "transcription2plaintext",
            "cipher_system": "homophonic_substitution",
            "target_records": [],
            "context_records": [],
            "description": "synthetic demo",
            "frontier_class": "known_good",
            "expected_solvers": ["decipher-automated"],
            "synthetic_spec": {
                "language": "en",
                "approx_length": 80,
                "word_boundaries": False,
                "homophonic": True,
                "seed": 1,
                "topic": "general",
                "frequency_style": "normal",
            },
        })
        + "\n",
        encoding="utf-8",
    )

    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "manifest").mkdir(parents=True)
    (benchmark_root / "manifest" / "records.jsonl").write_text("", encoding="utf-8")
    cache = PlaintextCache(tmp_path / "cache")
    cache.put(
        SyntheticSpec(language="en", approx_length=80, word_boundaries=False, homophonic=True, seed=1),
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG",
    )

    seen: dict[str, str] = {}

    class FakeRunner:
        def run_test(self, test_data, language=None):
            return SimpleNamespace(
                status="completed",
                char_accuracy=1.0,
                word_accuracy=0.0,
                artifact_path="decipher.json",
                error_message="",
            )

    def fake_automated_runner(artifact_dir, homophonic_budget="full", homophonic_refinement="none", homophonic_solver="zenith_native"):
        seen["homophonic_budget"] = homophonic_budget
        return FakeRunner()

    monkeypatch.setattr(frontier_runner, "AutomatedBenchmarkRunner", fake_automated_runner)
    monkeypatch.setattr(frontier_runner, "_load_external_configs", lambda path, oracle: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_suite.py",
            "--suite-file", str(suite),
            "--solvers", "decipher",
            "--benchmark-root", str(benchmark_root),
            "--cache-dir", str(tmp_path / "cache"),
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--summary-jsonl", str(tmp_path / "summary.jsonl"),
            "--summary-csv", str(tmp_path / "summary.csv"),
            "--homophonic-budget", "screen",
        ],
    )

    frontier_runner.main()

    assert seen["homophonic_budget"] == "screen"


def test_frontier_runner_passes_homophonic_refinement(monkeypatch, tmp_path):
    suite = tmp_path / "frontier.jsonl"
    suite.write_text(
        json.dumps({
            "test_id": "synth_demo",
            "track": "transcription2plaintext",
            "cipher_system": "homophonic_substitution",
            "target_records": [],
            "context_records": [],
            "description": "synthetic demo",
            "frontier_class": "known_good",
            "expected_solvers": ["decipher-automated"],
            "synthetic_spec": {
                "language": "en",
                "approx_length": 80,
                "word_boundaries": False,
                "homophonic": True,
                "seed": 1,
                "topic": "general",
                "frequency_style": "normal",
            },
        })
        + "\n",
        encoding="utf-8",
    )

    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "manifest").mkdir(parents=True)
    (benchmark_root / "manifest" / "records.jsonl").write_text("", encoding="utf-8")
    cache = PlaintextCache(tmp_path / "cache")
    cache.put(
        SyntheticSpec(language="en", approx_length=80, word_boundaries=False, homophonic=True, seed=1),
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG",
    )

    seen: dict[str, str] = {}

    class FakeRunner:
        def run_test(self, test_data, language=None):
            return SimpleNamespace(
                status="completed",
                char_accuracy=1.0,
                word_accuracy=0.0,
                artifact_path="decipher.json",
                error_message="",
            )

    def fake_automated_runner(artifact_dir, homophonic_budget="full", homophonic_refinement="none", homophonic_solver="zenith_native"):
        seen["homophonic_refinement"] = homophonic_refinement
        return FakeRunner()

    monkeypatch.setattr(frontier_runner, "AutomatedBenchmarkRunner", fake_automated_runner)
    monkeypatch.setattr(frontier_runner, "_load_external_configs", lambda path, oracle: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_suite.py",
            "--suite-file", str(suite),
            "--solvers", "decipher",
            "--benchmark-root", str(benchmark_root),
            "--cache-dir", str(tmp_path / "cache"),
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--summary-jsonl", str(tmp_path / "summary.jsonl"),
            "--summary-csv", str(tmp_path / "summary.csv"),
            "--homophonic-refinement", "targeted_repair",
        ],
    )

    frontier_runner.main()

    assert seen["homophonic_refinement"] == "targeted_repair"


def test_frontier_runner_passes_legacy_homophonic_flag(monkeypatch, tmp_path):
    suite = tmp_path / "frontier.jsonl"
    suite.write_text(
        json.dumps({
            "test_id": "synth_demo",
            "track": "transcription2plaintext",
            "cipher_system": "homophonic_substitution",
            "target_records": [],
            "context_records": [],
            "description": "synthetic demo",
            "frontier_class": "known_good",
            "expected_solvers": ["decipher-automated"],
            "synthetic_spec": {
                "language": "en",
                "approx_length": 80,
                "word_boundaries": False,
                "homophonic": True,
                "seed": 1,
                "topic": "general",
                "frequency_style": "normal",
            },
        })
        + "\n",
        encoding="utf-8",
    )

    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "manifest").mkdir(parents=True)
    (benchmark_root / "manifest" / "records.jsonl").write_text("", encoding="utf-8")
    cache = PlaintextCache(tmp_path / "cache")
    cache.put(
        SyntheticSpec(language="en", approx_length=80, word_boundaries=False, homophonic=True, seed=1),
        "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG",
    )

    seen: dict[str, str] = {}

    class FakeRunner:
        def run_test(self, test_data, language=None):
            return SimpleNamespace(
                status="completed",
                char_accuracy=1.0,
                word_accuracy=0.0,
                artifact_path="decipher.json",
                error_message="",
            )

    def fake_automated_runner(artifact_dir, homophonic_budget="full", homophonic_refinement="none", homophonic_solver="zenith_native"):
        seen["homophonic_solver"] = homophonic_solver
        return FakeRunner()

    monkeypatch.setattr(frontier_runner, "AutomatedBenchmarkRunner", fake_automated_runner)
    monkeypatch.setattr(frontier_runner, "_load_external_configs", lambda path, oracle: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_suite.py",
            "--suite-file", str(suite),
            "--solvers", "decipher",
            "--benchmark-root", str(benchmark_root),
            "--cache-dir", str(tmp_path / "cache"),
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--summary-jsonl", str(tmp_path / "summary.jsonl"),
            "--summary-csv", str(tmp_path / "summary.csv"),
            "--legacy-homophonic",
        ],
    )

    frontier_runner.main()

    assert seen["homophonic_solver"] == "legacy"


def test_frontier_runner_defaults_external_config_to_zenith_only(monkeypatch, tmp_path):
    suite = tmp_path / "frontier.jsonl"
    suite.write_text(
        json.dumps({
            "test_id": "synth_demo",
            "track": "transcription2plaintext",
            "cipher_system": "simple_substitution",
            "target_records": [],
            "context_records": [],
            "description": "synthetic demo",
            "frontier_class": "known_good",
            "expected_solvers": ["zenith"],
            "synthetic_spec": {
                "language": "en",
                "approx_length": 80,
                "word_boundaries": True,
                "homophonic": False,
                "seed": 1,
                "topic": "general",
                "frequency_style": "normal",
            },
        })
        + "\n",
        encoding="utf-8",
    )

    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "manifest").mkdir(parents=True)
    (benchmark_root / "manifest" / "records.jsonl").write_text("", encoding="utf-8")
    cache = PlaintextCache(tmp_path / "cache")
    cache.put(
        SyntheticSpec(language="en", approx_length=80, word_boundaries=True, homophonic=False, seed=1),
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    )

    seen: dict[str, str] = {}

    class FakeConfig:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(frontier_runner, "AutomatedBenchmarkRunner", lambda *args, **kwargs: SimpleNamespace())

    def fake_load_external_configs(path, oracle):
        seen["external_config"] = path
        return [FakeConfig("zenith")]

    monkeypatch.setattr(frontier_runner, "_load_external_configs", fake_load_external_configs)
    monkeypatch.setattr(
        frontier_runner,
        "run_external_baseline",
        lambda test_data, config, artifact_dir: SimpleNamespace(
            status="completed",
            char_accuracy=1.0,
            word_accuracy=1.0,
            elapsed=1.0,
            artifact_path="zenith.json",
            error="",
            candidates_considered=1,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frontier_suite.py",
            "--suite-file", str(suite),
            "--solvers", "external",
            "--benchmark-root", str(benchmark_root),
            "--cache-dir", str(tmp_path / "cache"),
            "--artifact-dir", str(tmp_path / "artifacts"),
            "--summary-jsonl", str(tmp_path / "summary.jsonl"),
            "--summary-csv", str(tmp_path / "summary.csv"),
        ],
    )

    frontier_runner.main()

    assert seen["external_config"].endswith("external_baselines/zenith_only.json")


def test_transposition_homophonic_ladder_suite_loads():
    suite = Path(__file__).resolve().parents[1] / "frontier" / "transposition_homophonic_ladder.jsonl"

    cases = load_frontier_suite(suite)

    assert [case.test.test_id for case in cases] == [
        "synth_en_120honb_s11",
        "synth_en_120thonb_reverse_s12",
        "synth_en_120thonb_ranges_s13",
        "synth_en_120thonb_columnar_s14",
        "synth_en_180thonb_ndown_s15",
    ]
    assert cases[0].synthetic_spec is not None
    assert cases[0].synthetic_spec.transform_pipeline is None
    assert all("transposition_homophonic_ladder" in case.frontier_tags for case in cases)
    assert cases[1].synthetic_spec is not None
    assert cases[1].synthetic_spec.transform_pipeline["steps"][0]["name"] == "Reverse"


def test_zodiac340_known_replay_suite_loads_fixture():
    repo_root = Path(__file__).resolve().parents[1]
    suite = repo_root / "frontier" / "zodiac340_known_replay.jsonl"
    benchmark_root = repo_root / "fixtures" / "benchmarks" / "zodiac340_known_replay"

    cases = load_frontier_suite(suite)
    test_data = resolve_frontier_case(
        cases[0],
        benchmark_loader=BenchmarkLoader(benchmark_root),
        cache=PlaintextCache(repo_root / "testgen_cache"),
    )

    assert cases[0].test.test_id == "zodiac340_known_replay"
    assert cases[0].raw["transform_pipeline"]["columns"] == 17
    assert len(cases[0].raw["transform_pipeline"]["steps"]) == 10
    assert test_data.transform_pipeline["steps"][0]["name"] == "NDownMAcross"
    assert len(test_data.plaintext) == 340
    assert len(test_data.canonical_transcription.split()) == 340
