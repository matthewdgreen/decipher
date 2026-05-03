from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cli
from automated.runner import (
    AutomatedBenchmarkRunner,
    AutomatedRunResult,
    format_automated_preflight_for_llm,
)
import automated.runner as automated_runner
import analysis.zenith_solver as zenith_solver
import benchmark.runner_v2 as runner_v2
from benchmark.runner_v2 import BenchmarkRunnerV2
from benchmark.loader import BenchmarkTest, TestData as BenchmarkTestData, parse_canonical_transcription


def _test_data() -> BenchmarkTestData:
    return BenchmarkTestData(
        test=BenchmarkTest(
            test_id="auto_demo",
            track="transcription2plaintext",
            cipher_system="simple_substitution",
            target_records=[],
            context_records=[],
            description="automated demo",
        ),
        canonical_transcription="A B C",
        plaintext="THE",
        plaintext_language="",
    )


def test_automated_benchmark_runner_writes_no_llm_artifact(tmp_path, monkeypatch):
    def fake_run(cipher_text, language="en", cipher_id="cli", ground_truth=None, cipher_system="", homophonic_budget="full", homophonic_refinement="none", homophonic_solver="zenith_native"):
        result = AutomatedRunResult(
            test_id=cipher_id,
            status="solved",
            final_decryption="THE",
            elapsed_seconds=0.01,
            char_accuracy=1.0,
            word_accuracy=1.0,
            solver="fake_automated",
            run_id="run123",
        )
        result.artifact = {
            "run_id": "run123",
            "run_mode": "automated_only",
            "automated_only": True,
            "test_id": cipher_id,
            "status": "solved",
            "solver": "fake_automated",
            "decryption": "THE",
            "char_accuracy": 1.0,
            "word_accuracy": 1.0,
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
        return result

    monkeypatch.setattr(automated_runner, "run_automated", fake_run)
    runner = AutomatedBenchmarkRunner(artifact_dir=tmp_path)

    result = runner.run_test(_test_data(), language="en")

    with open(result.artifact_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    assert result.char_accuracy == 1.0
    assert result.iterations_used == 0
    assert result.total_tokens == 0
    assert result.estimated_cost_usd == 0.0
    assert artifact["run_mode"] == "automated_only"
    assert artifact["estimated_cost_usd"] == 0.0
    assert artifact["total_input_tokens"] == 0


def test_cli_crack_defaults_to_automated_solver_and_bypasses_api_key(tmp_path, monkeypatch, capsys):
    input_file = tmp_path / "cipher.txt"
    input_file.write_text("A B C\n", encoding="utf-8")

    def forbidden_api_key():
        raise AssertionError("get_api_key must not be called")

    def fake_run(cipher_text, language="en", cipher_id="cli", ground_truth=None, cipher_system="", homophonic_budget="full", homophonic_refinement="none", homophonic_solver="zenith_native"):
        result = AutomatedRunResult(
            test_id=cipher_id,
            status="solved",
            final_decryption="THE",
            elapsed_seconds=0.01,
            solver="fake_automated",
            run_id="run123",
        )
        result.artifact = {
            "run_id": "run123",
            "run_mode": "automated_only",
            "automated_only": True,
            "test_id": cipher_id,
            "status": "solved",
            "solver": "fake_automated",
            "decryption": "THE",
            "steps": [],
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
        return result

    monkeypatch.setattr(cli, "get_api_key", forbidden_api_key)
    monkeypatch.setattr(automated_runner, "run_automated", fake_run)

    cli.cmd_crack(argparse.Namespace(
        file=str(input_file),
        canonical=True,
        max_iterations=25,
        model=None,
        language="en",
        artifact_dir=str(tmp_path / "artifacts"),
        cipher_id="auto_cli",
        verbose=False,
        agentic=False,
    ))

    out = capsys.readouterr().out
    assert "Running automated solver" in out
    assert "THE" in out


def test_cli_doctor_reports_required_rust_kernel(capsys):
    cli.cmd_doctor(argparse.Namespace(json=False))

    out = capsys.readouterr().out
    assert "Decipher environment check" in out
    assert "Rust fast kernels" in out
    assert "quagmire3_shotgun_search" in out


def test_cli_preflight_exits_early_when_required_rust_kernel_missing(monkeypatch, capsys):
    import analysis.polyalphabetic_fast as poly_fast

    monkeypatch.setattr(poly_fast, "FAST_AVAILABLE", False)
    monkeypatch.setattr(
        poly_fast,
        "fast_kernel_unavailable_message",
        lambda *, feature="the Rust fast kernel": f"{feature} unavailable; build it",
    )

    with pytest.raises(SystemExit) as exc:
        cli._require_rust_fast_kernel()

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Decipher runtime unavailable" in err


def test_cli_subprocess_reports_missing_rust_kernel_before_running_tool(tmp_path):
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        """
import importlib.abc


class _BlockDecipherFast(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "decipher_fast" or fullname.startswith("decipher_fast."):
            raise ModuleNotFoundError("blocked decipher_fast for CLI missing-kernel test")
        return None


import sys
sys.meta_path.insert(0, _BlockDecipherFast())
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(tmp_path),
        str(Path(__file__).resolve().parents[1] / "src"),
    ])

    completed = subprocess.run(
        [sys.executable, "-m", "cli", "crack"],
        input="ABC\n",
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Decipher runtime requires the Python module `decipher_fast`" in completed.stderr
    assert "blocked decipher_fast for CLI missing-kernel test" in completed.stderr
    assert "scripts/build_rust_fast.sh" in completed.stderr
    assert "PYTHONPATH=src .venv/bin/decipher doctor" in completed.stderr


def test_preflight_summary_omits_ground_truth_accuracy():
    result = AutomatedRunResult(
        test_id="secret_case",
        status="solved",
        final_decryption="THEQUICKBROWNFOX",
        elapsed_seconds=1.0,
        char_accuracy=1.0,
        word_accuracy=1.0,
        solver="fake_native",
        run_id="run123",
    )
    result.artifact = {
        "solver": "fake_native",
        "cipher_alphabet_size": 26,
        "cipher_token_count": 16,
        "cipher_word_count": 1,
        "ground_truth": "THEQUICKBROWNFOX",
        "char_accuracy": 1.0,
        "word_accuracy": 1.0,
        "steps": [{"name": "search_anneal", "score": -1.23}],
    }

    summary = format_automated_preflight_for_llm(result)

    assert "THEQUICKBROWNFOX" in summary
    assert "ground_truth" not in summary
    assert "char_accuracy" not in summary
    assert "word_accuracy" not in summary
    assert "1.0" not in summary
    assert "$0.00 (no LLM access)" in summary


def test_preflight_summary_includes_cipher_id_report_without_accuracy():
    result = AutomatedRunResult(
        test_id="vigenere_hint",
        status="completed",
        final_decryption="SOMEPLAINTEXT",
        elapsed_seconds=1.0,
        char_accuracy=1.0,
        word_accuracy=1.0,
        solver="periodic_polyalphabetic_screen",
        run_id="run123",
    )
    result.artifact = {
        "solver": "periodic_polyalphabetic_screen",
        "cipher_alphabet_size": 26,
        "cipher_token_count": 100,
        "cipher_word_count": 1,
        "cipher_id_report": {
            "natural_language_summary": "Depressed IC with periodic recovery.",
            "best_period": 5,
            "best_period_ic": 0.064,
            "suspicion_scores": {
                "polyalphabetic_vigenere": 0.72,
                "homophonic_substitution": 0.31,
            },
        },
        "ground_truth": "SOMEPLAINTEXT",
        "char_accuracy": 1.0,
        "word_accuracy": 1.0,
        "steps": [{"name": "search_periodic_polyalphabetic", "score": -5.0}],
    }

    summary = format_automated_preflight_for_llm(result)

    assert "Cipher-type fingerprint" in summary
    assert "polyalphabetic_vigenere=0.72" in summary
    assert "Periodic IC best period: 5" in summary
    assert "observe_cipher_shape" in summary
    assert "ground_truth" not in summary
    assert "char_accuracy" not in summary
    assert "word_accuracy" not in summary


def test_automated_transform_search_screen_records_router_artifact(monkeypatch):
    cipher_text = parse_canonical_transcription("S001 S002 S003 S004 S005 S006")

    def fake_substitution(cipher_text, language):
        return (
            "fake_substitution",
            {0: 0, 1: 1, 2: 2},
            "ABCDEF",
            {"name": "search_anneal", "score": -1.0},
        )

    monkeypatch.setattr(automated_runner, "_run_substitution", fake_substitution)
    result = automated_runner.run_automated(
        cipher_text=cipher_text,
        language="en",
        cipher_id="transform_screen_demo",
        cipher_system="simple_substitution",
        transform_search="screen",
    )

    screen_step = next(
        step for step in result.artifact["steps"]
        if step["name"] == "screen_transform_candidates"
    )
    assert result.status == "completed"
    assert result.artifact["transform_search"]["status"] == "screened"
    assert screen_step["screen"]["top_candidates"]
    assert screen_step["suspicion"]["recommendation"] in {
        "stay_baseline",
        "consider_screen",
        "run_screen",
    }


def test_automated_transform_search_rank_can_select_candidate(monkeypatch):
    cipher_text = parse_canonical_transcription("S001 S002 S003 S004 S005 S006")
    monkeypatch.setenv("DECIPHER_ZENITH_NATIVE_ENGINE", "python")
    monkeypatch.setenv("DECIPHER_TRANSFORM_RANK_ENGINE", "python")

    def fake_homophonic(cipher_text, language, budget, refinement, solver_profile, ground_truth, seed_offset=0):
        return (
            "fake_homophonic",
            {0: 0, 1: 1},
            "TRANSFORMED",
            {"name": "search_homophonic_anneal", "anneal_score": -0.5},
        )

    monkeypatch.setattr(automated_runner, "_run_homophonic", fake_homophonic)
    result = automated_runner.run_automated(
        cipher_text=cipher_text,
        language="en",
        cipher_id="transform_rank_demo",
        cipher_system="zodiac_transposition_homophonic",
        transform_search="rank",
        homophonic_budget="screen",
    )

    assert result.status == "completed"
    assert result.solver == "transform_search_homophonic"
    assert result.final_decryption == "TRANSFORMED"
    rank = result.artifact["transform_search"]["rank"]
    assert rank["top_ranked_candidates"]
    assert rank["evaluation"]["stage"] == "transform_homophonic_finalist_menu_evaluation"
    assert rank["validation"]["policy"]
    assert rank["confirmation"]["policy"]
    assert rank["finalists"]["policy"]
    assert rank["selection"]["selected"] is True
    assert rank["diagnostics"]["policy"]
    assert "validated_selection_score" in rank["top_ranked_candidates"][0]
    assert "confirmed_selection_score" in rank["top_ranked_candidates"][0]
    assert "finalist_label" in rank["top_ranked_candidates"][0]
    assert any(step["name"] == "select_transform_candidate" for step in result.artifact["steps"])


def test_automated_wide_transform_search_is_structural_only():
    cipher_text = parse_canonical_transcription(" ".join(f"S{i:03d}" for i in range(1, 121)))

    result = automated_runner.run_automated(
        cipher_text=cipher_text,
        language="en",
        cipher_id="wide_structural_demo",
        cipher_system="zodiac_transposition_homophonic",
        transform_search="wide",
        transform_search_profile="wide",
    )

    assert result.status == "completed"
    assert result.solver == "transform_search_structural_only"
    assert result.final_decryption == ""
    report = result.artifact["transform_search"]
    assert report["rank"] is None
    assert report["screen"]["profile"] == "wide"
    assert report["screen"]["top_candidates"]
    assert any(step["name"] == "transform_search_structural_only" for step in result.artifact["steps"])


def test_automated_transform_promote_artifact_feeds_rank_stage(tmp_path, monkeypatch):
    cipher_text = parse_canonical_transcription("S001 S002 S003 S004 S005 S006")
    promoted_candidate = {
        "candidate_id": "cand_reverse",
        "family": "reverse_whole",
        "provenance": "unit_test",
        "params": {},
        "pipeline": {
            "steps": [
                {"transformer_name": "reverse_ranges", "data": {"ranges": [[0, 6]]}},
            ],
        },
        "score": 1.25,
        "metrics": {"matrix_rank_score": 1.0},
    }
    artifact_path = tmp_path / "wide.json"
    artifact_path.write_text(
        json.dumps({
            "transform_search": {
                "mode": "wide",
                "profile": "wide",
                "status": "screened",
                "screen": {
                    "profile": "wide",
                    "candidate_count": 2,
                    "deduped_candidate_count": 2,
                    "identity_candidate": {
                        "candidate_id": "000_identity",
                        "family": "identity",
                        "pipeline": {"steps": []},
                        "score": 0.0,
                    },
                    "top_candidates": [
                        promoted_candidate,
                        {
                            "candidate_id": "cand_other",
                            "family": "route_rows",
                            "pipeline": {"steps": []},
                            "score": 0.5,
                        },
                    ],
                    "anchor_candidates": [],
                },
            },
        }),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_rank(**kwargs):
        screen = kwargs["screen"]
        seen["screen"] = screen
        seen["max_candidates"] = kwargs["max_candidates"]
        candidate = dict(screen["top_candidates"][0])
        candidate.update({
            "status": "completed",
            "key": {"0": 0},
            "decryption": "PROMOTED",
            "anneal_score": -0.1,
            "validated_selection_score": 1.0,
            "confirmed_selection_score": 1.0,
            "finalist_label": "robust_candidate",
        })
        return {
            "budget": kwargs["budget"],
            "selection": {
                "selected": True,
                "selected_candidate_id": candidate["candidate_id"],
            },
            "top_ranked_candidates": [candidate],
        }

    monkeypatch.setattr(automated_runner, "_rank_transform_candidates", fake_rank)

    result = automated_runner.run_automated(
        cipher_text=cipher_text,
        language="en",
        cipher_id="promote_demo",
        cipher_system="zodiac_transposition_homophonic",
        transform_search="promote",
        transform_promote_artifact=str(artifact_path),
        transform_promote_candidate_ids=["cand_reverse"],
        homophonic_budget="screen",
    )

    assert result.status == "completed"
    assert result.solver == "transform_search_homophonic"
    assert result.final_decryption == "PROMOTED"
    report = result.artifact["transform_search"]
    assert report["status"] == "promoted"
    assert report["screen"]["promotion"]["source_artifact"].endswith("wide.json")
    assert report["screen"]["promotion"]["promoted_candidate_ids"] == ["cand_reverse"]
    assert seen["screen"]["top_candidates"][0]["candidate_id"] == "cand_reverse"
    assert seen["max_candidates"] >= 2
    assert result.artifact["transform_pipeline"] == {
        "steps": [{"name": "reverse_ranges", "data": {"ranges": [[0, 6]]}}],
    }
    assert result.artifact["input_transform_pipeline"] is None
    assert result.artifact["transform_selection"]["source"] == "transform_search_rank"
    assert result.artifact["transform_selection"]["candidate_id"] == "cand_reverse"
    assert result.artifact["transform_selection"]["promotion"]["source_artifact"].endswith("wide.json")


def test_transform_search_fast_profile_trims_rank_work():
    fast = automated_runner._transform_search_profile_params("rank", "fast")
    broad = automated_runner._transform_search_profile_params("rank", "broad")
    promote = automated_runner._transform_search_profile_params("promote", "fast")
    wide_screen = automated_runner._transform_search_profile_params("wide", "wide")
    wide_50k = automated_runner._transform_search_profile_params(
        "wide",
        "wide",
        max_generated_candidates=50000,
    )

    assert fast["screen_profile"] == "medium"
    assert fast["include_mutations"] is False
    assert fast["top_n"] < broad["top_n"]
    assert fast["max_candidates"] < broad["max_candidates"]
    assert fast["adaptive_confirmations"] == 0
    assert fast["include_program_search"] is False
    assert broad["include_mutations"] is True
    assert broad["include_program_search"] is True
    assert broad["program_max_depth"] >= 5
    assert promote["max_candidates"] == fast["max_candidates"]
    assert wide_screen["screen_profile"] == "wide"
    assert wide_screen["max_candidates"] == 0
    assert wide_screen["max_generated_candidates"] > broad["max_generated_candidates"]
    assert wide_screen["include_program_search"] is True
    assert wide_50k["max_generated_candidates"] == 50000


def test_transform_rank_triage_preserves_family_breadth():
    def candidate(candidate_id, family, score=1.0, matrix=1.0):
        return {
            "candidate_id": candidate_id,
            "family": family,
            "pipeline": {"steps": []},
            "score": score,
            "metrics": {
                "matrix_rank_score": matrix,
                "periodic_redundancy": matrix,
                "inverse_periodic_redundancy": 0.0,
                "position_nontriviality": 1.0,
            },
        }

    screen = {
        "identity_candidate": candidate("000_identity", "identity", score=0.0, matrix=0.0),
        "top_candidates": [
            candidate(f"diag_{i}", "route_diagonal_down_right", score=2.0 - i * 0.01, matrix=2.0)
            for i in range(8)
        ] + [
            candidate("columns", "route_columns_down", score=0.5, matrix=0.5),
            candidate("ndown", "ndownmacross_1_2", score=0.4, matrix=0.4),
            candidate("split", "split_horizontal_10_swap", score=0.3, matrix=0.3),
        ],
    }

    selected, report = automated_runner._two_stage_transform_rank_candidates(
        screen,
        max_candidates=6,
    )
    selected_ids = {item["candidate_id"] for item in selected}

    assert "000_identity" in selected_ids
    assert "columns" in selected_ids
    assert "ndown" in selected_ids
    assert "split" in selected_ids
    assert report["class_counts"]["diagonal_route"] < 8


def test_transform_rank_triage_keeps_route_rows_when_programs_dominate():
    def candidate(candidate_id, family, score=1.0):
        return {
            "candidate_id": candidate_id,
            "family": family,
            "pipeline": {"steps": []},
            "score": score,
            "metrics": {
                "matrix_rank_score": score,
                "periodic_redundancy": score,
                "inverse_periodic_redundancy": 0.0,
                "position_nontriviality": 1.0,
            },
        }

    screen = {
        "identity_candidate": candidate("000_identity", "identity", score=0.0),
        "top_candidates": [
            candidate(f"program_{i:04d}", "program_banded_ndown_constructed", score=2.0 - i * 0.01)
            for i in range(12)
        ],
        "anchor_candidates": [
            candidate("route_rows_17", "route_rows_boustrophedon", score=0.2),
            candidate("route_rows_19", "route_rows_boustrophedon", score=0.19),
            candidate("route_rows_20", "route_rows_boustrophedon", score=0.18),
            candidate("route_columns_up", "route_columns_up", score=0.5),
        ],
    }

    selected, report = automated_runner._two_stage_transform_rank_candidates(
        screen,
        max_candidates=10,
    )
    selected_ids = {item["candidate_id"] for item in selected}

    assert "route_rows_17" in selected_ids
    assert "route_columns_up" in selected_ids
    assert report["selection_reasons"]["route_rows_17"] == "family_diverse:route_rows"


def test_program_diverse_candidates_reserve_route_repair_shape():
    def candidate(candidate_id, family, template, score):
        return {
            "candidate_id": candidate_id,
            "family": family,
            "pipeline": {"steps": []},
            "params": {"template": template},
            "score": score,
            "metrics": {
                "matrix_rank_score": score,
                "periodic_redundancy": score,
                "inverse_periodic_redundancy": 0.0,
                "position_nontriviality": 1.0,
            },
        }

    bucket = [
        candidate(f"banded_{i}", "program_banded_ndown_constructed", "banded_ndown_constructed", 2.0 - i * 0.01)
        for i in range(8)
    ] + [
        candidate("route_repair", "program_route_repair_constructed", "route_repair_constructed", 0.5)
    ]

    selected = automated_runner._program_diverse_transform_candidates(bucket, limit=3)

    assert any(item["candidate_id"] == "route_repair" for item in selected)
    assert len(selected) == 3


def test_transform_rank_triage_prefers_constructed_program_before_generic_banded_template():
    def candidate(candidate_id, family, params):
        return {
            "candidate_id": candidate_id,
            "family": family,
            "pipeline": {"steps": []},
            "params": params,
            "score": 0.90,
            "metrics": {
                "matrix_rank_score": 0.70,
                "periodic_redundancy": 0.70,
                "inverse_periodic_redundancy": 0.70,
                "position_nontriviality": 1.0,
            },
        }

    screen = {
        "identity_candidate": candidate("000_identity", "identity", {}),
        "top_candidates": [
            candidate(
                "generic_banded",
                "banded_ndown_lock_shift_across_2",
                {"template": "banded_ndown_lock_shift"},
            ),
            candidate(
                "constructed",
                "program_banded_ndown_constructed",
                {
                    "template": "banded_ndown_constructed",
                    "constructed_template_match": True,
                    "template_source": "program_grammar",
                },
            ),
        ],
    }

    selected, _report = automated_runner._two_stage_transform_rank_candidates(
        screen,
        max_candidates=2,
    )

    assert selected[1]["candidate_id"] == "constructed"


def test_transform_rank_triage_keeps_banded_program_when_route_repairs_dominate():
    def candidate(candidate_id, family, params=None, score=1.0):
        return {
            "candidate_id": candidate_id,
            "family": family,
            "pipeline": {"steps": []},
            "params": params or {},
            "score": score,
            "metrics": {
                "matrix_rank_score": score,
                "periodic_redundancy": score,
                "inverse_periodic_redundancy": score,
                "position_nontriviality": 1.0,
            },
        }

    screen = {
        "identity_candidate": candidate("000_identity", "identity", score=0.0),
        "top_candidates": [
            candidate(
                f"route_repair_{i}",
                "program_route_repair_constructed",
                {"template": "route_repair_constructed"},
                score=1.0 - i * 0.01,
            )
            for i in range(8)
        ],
        "anchor_candidates": [
            candidate(
                "banded_program",
                "program_banded_ndown_constructed",
                {"template": "banded_ndown_constructed"},
                score=0.60,
            ),
        ],
    }

    selected, report = automated_runner._two_stage_transform_rank_candidates(
        screen,
        max_candidates=5,
    )
    selected_ids = {item["candidate_id"] for item in selected}

    assert "banded_program" in selected_ids
    assert report["selection_reasons"]["banded_program"] == "family_diverse:program_search"


def test_transform_rank_triage_keeps_multiple_banded_program_variants():
    def candidate(candidate_id, labels, score):
        return {
            "candidate_id": candidate_id,
            "family": "program_banded_ndown_constructed",
            "pipeline": {"steps": []},
            "params": {
                "template": "banded_ndown_constructed",
                "operation_labels": labels,
            },
            "score": score,
            "metrics": {
                "matrix_rank_score": score,
                "periodic_redundancy": score,
                "inverse_periodic_redundancy": score,
                "position_nontriviality": 1.0,
            },
        }

    screen = {
        "identity_candidate": {
            "candidate_id": "000_identity",
            "family": "identity",
            "pipeline": {"steps": []},
            "params": {},
            "score": 0.0,
            "metrics": {},
        },
        "top_candidates": [
            candidate(
                "banded_a1_left",
                ["ndown_top_a1", "late_shift_left", "mid_late_lock", "ndown_lower_a1", "tail_repair_pack"],
                0.63,
            ),
            candidate(
                "banded_a2_right",
                ["ndown_top_a2", "late_shift_right", "mid_late_lock", "ndown_lower_a2", "tail_repair_pack"],
                0.61,
            ),
            candidate(
                "banded_a2_left",
                ["ndown_top_a2", "late_shift_left", "mid_late_lock", "ndown_lower_a2", "tail_repair_pack"],
                0.60,
            ),
        ],
    }

    selected, _report = automated_runner._two_stage_transform_rank_candidates(
        screen,
        max_candidates=6,
    )
    selected_ids = {item["candidate_id"] for item in selected}

    assert {"banded_a1_left", "banded_a2_right", "banded_a2_left"} <= selected_ids


def test_transform_rank_triage_prefers_full_tail_repair_pack():
    def candidate(candidate_id, labels, *, matrix, periodic, structural, nontrivial=0.95):
        return {
            "candidate_id": candidate_id,
            "family": "program_banded_ndown_constructed",
            "pipeline": {"steps": []},
            "params": {
                "template": "banded_ndown_constructed",
                "operation_labels": labels,
            },
            "score": structural,
            "metrics": {
                "matrix_rank_score": matrix,
                "periodic_redundancy": periodic,
                "inverse_periodic_redundancy": 0.0,
                "position_nontriviality": nontrivial,
            },
        }

    screen = {
        "identity_candidate": {
            "candidate_id": "000_identity",
            "family": "identity",
            "pipeline": {"steps": []},
            "params": {},
            "score": 0.0,
            "metrics": {},
        },
        "top_candidates": [
            candidate(
                "banded_short",
                ["ndown_top_s24_a2", "late_shift_right", "mid_late_lock", "ndown_lower_s24_a2", "tail_repair_short"],
                matrix=0.679,
                periodic=0.818,
                structural=0.670,
                nontrivial=0.942,
            ),
            candidate(
                "banded_pack",
                ["ndown_top_s24_a2", "late_shift_right", "mid_late_lock", "ndown_lower_s24_a2", "tail_repair_pack"],
                matrix=0.682,
                periodic=0.799,
                structural=0.671,
                nontrivial=0.959,
            ),
        ],
    }

    selected, _report = automated_runner._two_stage_transform_rank_candidates(
        screen,
        max_candidates=2,
    )
    selected_ids = [item["candidate_id"] for item in selected]

    assert "banded_pack" in selected_ids
    assert "banded_short" not in selected_ids


def test_transform_rank_triage_keeps_banded_across_shift_cells():
    def candidate(candidate_id, labels, score):
        return {
            "candidate_id": candidate_id,
            "family": "program_banded_ndown_constructed",
            "pipeline": {"steps": []},
            "params": {
                "template": "banded_ndown_constructed",
                "operation_labels": labels,
            },
            "score": score,
            "metrics": {
                "matrix_rank_score": score,
                "periodic_redundancy": score,
                "inverse_periodic_redundancy": score,
                "position_nontriviality": 1.0,
            },
        }

    screen = {
        "identity_candidate": {
            "candidate_id": "000_identity",
            "family": "identity",
            "pipeline": {"steps": []},
            "params": {},
            "score": 0.0,
            "metrics": {},
        },
        "top_candidates": [
            candidate(
                f"a1_left_{i}",
                ["ndown_top_a1", "late_shift_left", "mid_late_lock", "ndown_lower_a1", "tail_repair_pack"],
                0.80 - i * 0.01,
            )
            for i in range(3)
        ] + [
            candidate(
                "a1_right",
                ["ndown_top_a1", "late_shift_right", "mid_late_lock", "ndown_lower_a1", "tail_repair_pack"],
                0.70,
            ),
            candidate(
                "a2_left",
                ["ndown_top_a2", "late_shift_left", "mid_late_lock", "ndown_lower_a2", "tail_repair_pack"],
                0.69,
            ),
            candidate(
                "a2_right",
                ["ndown_top_a2", "late_shift_right", "mid_late_lock", "ndown_lower_a2", "tail_repair_pack"],
                0.68,
            ),
        ],
    }

    selected, _report = automated_runner._two_stage_transform_rank_candidates(
        screen,
        max_candidates=5,
    )
    selected_ids = {item["candidate_id"] for item in selected}

    assert {"a1_left_0", "a1_right", "a2_left", "a2_right"} <= selected_ids
    assert "a1_left_1" not in selected_ids


def test_transform_rank_triage_prefers_mid_split_banded_programs():
    def candidate(candidate_id, *, split, matrix, periodic, structural):
        return {
            "candidate_id": candidate_id,
            "family": "program_banded_ndown_constructed",
            "pipeline": {
                "steps": [
                    {"name": "NDownMAcross", "data": {"rangeStart": 0, "rangeEnd": split - 1, "down": 1, "across": 2}},
                    {"name": "ShiftCharactersLeft", "data": {"rangeStart": 478, "rangeEnd": 493}},
                ],
                "columns": 19,
                "rows": 36,
            },
            "grid": {"columns": 19, "rows": 36},
            "params": {
                "template": "banded_ndown_constructed",
                "operation_labels": ["ndown_top_a2", "late_shift_left", "mid_late_lock", "ndown_lower_a2", "tail_repair_pack"],
            },
            "score": structural,
            "metrics": {
                "matrix_rank_score": matrix,
                "periodic_redundancy": periodic,
                "inverse_periodic_redundancy": 0.0,
                "position_nontriviality": 0.96,
            },
        }

    screen = {
        "identity_candidate": {
            "candidate_id": "000_identity",
            "family": "identity",
            "pipeline": {"steps": []},
            "params": {},
            "score": 0.0,
            "metrics": {},
        },
        "top_candidates": [
            candidate("late_split", split=24, matrix=0.682, periodic=0.799, structural=0.671),
            candidate("mid_split", split=17, matrix=0.652, periodic=0.715, structural=0.649),
        ],
    }

    selected, _report = automated_runner._two_stage_transform_rank_candidates(
        screen,
        max_candidates=2,
    )
    selected_ids = [item["candidate_id"] for item in selected]

    assert "mid_split" in selected_ids


def test_transform_family_gates_reject_unstable_false_positive():
    ranked = [
        {
            "candidate_id": "candidate_unwrap",
            "family": "unwrap_transposition_reverse_key",
            "status": "completed",
            "confirmed_selection_score": -7.70,
            "confirmation": {
                "status": "completed",
                "stability_score": 0.10,
            },
        },
        {
            "candidate_id": "000_identity",
            "family": "identity",
            "status": "completed",
            "confirmed_selection_score": -7.90,
            "confirmation": {
                "status": "completed",
                "stability_score": 0.50,
            },
        },
    ]

    report = automated_runner._label_transform_finalists(ranked)
    ranked.sort(
        key=lambda item: (
            bool(item.get("selectable_transform_candidate")),
            item.get("confirmed_selection_score", float("-inf")),
        ),
        reverse=True,
    )
    selection = automated_runner._choose_transform_candidate(ranked)

    assert report["label_counts"]["unstable_false_positive"] == 1
    assert ranked[0]["candidate_id"] == "000_identity"
    assert selection["selected_candidate_id"] == "000_identity"
    assert selection["selects_transform"] is False


def test_banded_program_gate_allows_moderate_stability_with_clear_margin():
    ranked = [
        {
            "candidate_id": "banded",
            "family": "program_banded_ndown_constructed",
            "params": {
                "template": "banded_ndown_constructed",
                "constructed_template_match": True,
                "template_source": "program_grammar",
            },
            "status": "completed",
            "confirmed_selection_score": -7.55,
            "confirmation": {
                "status": "completed",
                "stability_score": 0.50,
            },
        },
        {
            "candidate_id": "000_identity",
            "family": "identity",
            "status": "completed",
            "confirmed_selection_score": -7.89,
            "confirmation": {
                "status": "completed",
                "stability_score": 0.45,
            },
        },
    ]

    automated_runner._label_transform_finalists(ranked)
    ranked.sort(
        key=lambda item: (
            bool(item.get("selectable_transform_candidate")),
            item.get("confirmed_selection_score", float("-inf")),
        ),
        reverse=True,
    )
    selection = automated_runner._choose_transform_candidate(ranked)

    assert ranked[0]["candidate_id"] == "banded"
    assert selection["selected_candidate_id"] == "banded"
    assert selection["selects_transform"] is True


def test_selected_transform_candidate_gets_full_final_refinement(monkeypatch):
    cipher_text = parse_canonical_transcription("A B C")
    selected = {
        "candidate_id": "reverse_candidate",
        "family": "whole_reverse",
        "status": "completed",
        "decryption": "SCREEN",
        "key": {"0": 0, "1": 1, "2": 2},
        "pipeline": {"steps": [{"name": "Reverse", "data": {}}]},
        "anneal_score": -8.0,
        "validated_selection_score": -8.0,
        "confirmed_selection_score": -8.0,
        "elapsed_seconds": 1.0,
    }
    calls = []

    monkeypatch.setattr(
        automated_runner,
        "inspect_transform_suspicion",
        lambda **_kwargs: {"recommendation": "run_screen"},
    )
    monkeypatch.setattr(
        automated_runner,
        "screen_transform_candidates",
        lambda *_args, **_kwargs: {"identity_candidate": {}, "top_candidates": []},
    )
    monkeypatch.setattr(
        automated_runner,
        "_rank_transform_candidates",
        lambda **_kwargs: {
            "budget": "screen",
            "selection": {
                "selected": True,
                "selected_candidate_id": "reverse_candidate",
            },
            "top_ranked_candidates": [selected],
        },
    )

    def fake_homophonic(cipher_text, language, budget, refinement, solver_profile, ground_truth, seed_offset=0):
        calls.append({"tokens": list(cipher_text.tokens), "budget": budget})
        return (
            "fake_homophonic",
            {0: 0, 1: 1, 2: 2},
            "FULL",
            {"name": "search_homophonic_anneal", "solver": "fake", "budget": budget},
        )

    monkeypatch.setattr(automated_runner, "_run_homophonic", fake_homophonic)

    result = automated_runner.run_automated(
        cipher_text=cipher_text,
        language="en",
        cipher_id="refine_transform",
        cipher_system="transposition_homophonic",
        transform_search="rank",
        homophonic_budget="full",
    )

    assert result.solver == "transform_search_homophonic_refined"
    assert result.final_decryption == "FULL"
    assert calls == [{"tokens": [2, 1, 0], "budget": "full"}]
    assert any(
        step["name"] == "refine_selected_transform_candidate_homophonic"
        for step in result.artifact["steps"]
    )


def test_transform_rank_and_zenith_fast_engines_default_to_rust(monkeypatch):
    monkeypatch.delenv("DECIPHER_ZENITH_NATIVE_ENGINE", raising=False)
    monkeypatch.delenv("DECIPHER_TRANSFORM_RANK_ENGINE", raising=False)

    assert automated_runner._zenith_native_engine() == "rust"
    assert automated_runner._transform_rank_engine() == "rust"


def test_rust_rank_auto_escalates_to_full_when_screen_has_no_robust_candidate(monkeypatch):
    cipher_text = parse_canonical_transcription("A B C")
    full_candidate = {
        "candidate_id": "full_candidate",
        "family": "program_banded_ndown_constructed",
        "status": "completed",
        "decryption": "FULL_RANK",
        "key": {"0": 0, "1": 1, "2": 2},
        "pipeline": {"steps": [{"name": "Reverse", "data": {}}]},
        "anneal_score": -6.8,
        "validated_selection_score": -6.8,
        "confirmed_selection_score": -6.8,
        "elapsed_seconds": 1.0,
        "finalist_label": "robust_candidate",
        "selectable_transform_candidate": True,
    }
    calls = []

    monkeypatch.setenv("DECIPHER_ZENITH_NATIVE_ENGINE", "rust")
    monkeypatch.setenv("DECIPHER_TRANSFORM_RANK_ENGINE", "rust")
    monkeypatch.setattr(
        automated_runner,
        "inspect_transform_suspicion",
        lambda **_kwargs: {"recommendation": "run_screen"},
    )
    monkeypatch.setattr(
        automated_runner,
        "screen_transform_candidates",
        lambda *_args, **_kwargs: {"identity_candidate": {}, "top_candidates": []},
    )

    def fake_rank(**kwargs):
        calls.append(kwargs["budget"])
        if kwargs["budget"] == "screen":
            return {
                "budget": "screen",
                "evaluated_candidates": 10,
                "selection": {
                    "selected": False,
                    "selected_candidate_id": None,
                    "reason": "no_confirmed_candidate_passed_family_specific_gates",
                },
                "diagnostics": {"conclusion": "no_robust_transform_unstable_false_positives"},
                "top_ranked_candidates": [
                    {
                        "candidate_id": "unstable",
                        "family": "program_banded_ndown_constructed",
                        "finalist_label": "unstable_false_positive",
                        "confirmed_selection_score": -7.8,
                        "anneal_score": -7.7,
                    }
                ],
            }
        return {
            "budget": "full",
            "evaluated_candidates": 12,
            "selection": {
                "selected": True,
                "selected_candidate_id": "full_candidate",
            },
            "diagnostics": {"conclusion": "robust_transform_selected"},
            "top_ranked_candidates": [full_candidate],
        }

    monkeypatch.setattr(automated_runner, "_rank_transform_candidates", fake_rank)

    result = automated_runner.run_automated(
        cipher_text=cipher_text,
        language="en",
        cipher_id="rank_auto_escalate",
        cipher_system="transposition_homophonic",
        transform_search="rank",
        homophonic_budget="full",
    )

    assert calls == ["screen", "full"]
    assert result.solver == "transform_search_homophonic"
    assert result.final_decryption == "FULL_RANK"
    report = result.artifact["transform_search"]
    assert report["rank"]["budget"] == "full"
    assert report["rank_escalation"]["status"] == "escalated"
    assert report["rank_escalation"]["initial_budget"] == "screen"
    assert result.artifact["transform_selection"]["source"] == "transform_search_rank"
    assert result.artifact["transform_selection"]["candidate_id"] == "full_candidate"
    assert not any(
        step["name"] == "refine_selected_transform_candidate_homophonic"
        for step in result.artifact["steps"]
    )


def test_rust_transform_confirmation_uses_batch_seed_offsets(monkeypatch, tmp_path):
    import analysis.zenith_fast as zenith_fast

    ranked = [
        {
            "candidate_id": "candidate_a",
            "family": "program_banded_ndown_constructed",
            "status": "completed",
            "pipeline": {"steps": []},
            "selection_score": -7.0,
            "validated_selection_score": -7.0,
            "structural_score": 0.0,
            "decryption": "ABCDEF",
        },
        {
            "candidate_id": "candidate_b",
            "family": "whole_reverse",
            "status": "completed",
            "pipeline": {"steps": [{"name": "Reverse", "data": {}}]},
            "selection_score": -9.0,
            "validated_selection_score": -9.0,
            "structural_score": 0.0,
            "decryption": "FEDCBA",
        },
    ]
    calls = []

    monkeypatch.setenv("DECIPHER_ZENITH_NATIVE_ENGINE", "rust")
    monkeypatch.setenv("DECIPHER_TRANSFORM_RANK_ENGINE", "rust")
    monkeypatch.setattr(
        automated_runner,
        "_zenith_native_model_path",
        lambda _language: tmp_path / "fake_model.bin",
    )
    monkeypatch.setattr(
        automated_runner,
        "_homophonic_budget_params",
        lambda *_args, **_kwargs: {
            "epochs": 1,
            "sampler_iterations": 1,
            "seeds": [3, 4],
        },
    )

    def fake_batch(**kwargs):
        calls.append(kwargs)
        rows = []
        for candidate in kwargs["candidates"]:
            rows.append({
                "candidate_id": candidate["candidate_id"],
                "family": candidate["family"],
                "status": "completed",
                "seed_offset": candidate["seed_offset"],
                "best_seed": candidate["seed_offset"] + kwargs["seeds"][0],
                "decryption": "ABCDEF",
                "normalized_score": -6.5,
                "elapsed_seconds": 0.1,
                "key": {"0": 0},
            })
        return {"results": rows}

    monkeypatch.setattr(zenith_fast, "zenith_transform_candidates_batch_fast", fake_batch)

    report = automated_runner._confirm_transform_finalists(
        cipher_text=parse_canonical_transcription("A B C D E F"),
        language="en",
        ranked=ranked,
        budget="screen",
        solver_profile="zenith_native",
        confirm_count=1,
        adaptive_confirmations=0,
    )

    assert len(calls) == 1
    assert calls[0]["seeds"] == [3, 4]
    assert calls[0]["candidates"][0]["seed_offset"] == 10000
    assert ranked[0]["confirmation"]["engine"] == "rust_batch"
    assert ranked[0]["confirmation"]["best_seed"] == 10003
    assert ranked[1]["confirmation"]["status"] == "not_run"
    assert report["engine"] == "rust_batch"
    assert report["confirmed_candidate_count"] == 1


def test_full_final_refinement_bakeoff_can_replace_screen_winner(monkeypatch):
    cipher_text = parse_canonical_transcription("A B C")
    screen_winner = {
        "candidate_id": "screen_winner",
        "family": "program_banded_ndown_constructed",
        "status": "completed",
        "decryption": "SCREEN_WINNER",
        "key": {"0": 0},
        "pipeline": {"steps": [{"name": "Reverse", "data": {"rangeStart": 0, "rangeEnd": 2}}]},
        "anneal_score": -7.50,
        "validated_selection_score": -7.50,
        "confirmed_selection_score": -7.50,
        "finalist_label": "robust_candidate",
        "selectable_transform_candidate": True,
    }
    close_neighbor = {
        "candidate_id": "close_neighbor",
        "family": "program_banded_ndown_constructed",
        "status": "completed",
        "decryption": "SCREEN_NEIGHBOR",
        "key": {"0": 0},
        "pipeline": {"steps": [{"name": "Reverse", "data": {"rangeStart": 0, "rangeEnd": 1}}]},
        "anneal_score": -7.52,
        "validated_selection_score": -7.52,
        "confirmed_selection_score": -7.52,
        "finalist_label": "robust_candidate",
        "selectable_transform_candidate": True,
    }

    monkeypatch.setattr(
        automated_runner,
        "inspect_transform_suspicion",
        lambda **_kwargs: {"recommendation": "run_screen"},
    )
    monkeypatch.setattr(
        automated_runner,
        "screen_transform_candidates",
        lambda *_args, **_kwargs: {"identity_candidate": {}, "top_candidates": []},
    )
    monkeypatch.setattr(
        automated_runner,
        "_rank_transform_candidates",
        lambda **_kwargs: {
            "budget": "screen",
            "selection": {
                "selected": True,
                "selected_candidate_id": "screen_winner",
            },
            "top_ranked_candidates": [screen_winner, close_neighbor],
        },
    )

    def fake_homophonic(cipher_text, language, budget, refinement, solver_profile, ground_truth, seed_offset=0):
        if cipher_text.source.endswith(":1"):
            return (
                "fake_homophonic",
                {0: 0},
                "FULL_NEIGHBOR",
                {"name": "search_homophonic_anneal", "anneal_score": -6.0},
            )
        return (
            "fake_homophonic",
            {0: 0},
            "FULL_SCREEN_WINNER",
            {"name": "search_homophonic_anneal", "anneal_score": -7.0},
        )

    monkeypatch.setattr(automated_runner, "_run_homophonic", fake_homophonic)

    result = automated_runner.run_automated(
        cipher_text=cipher_text,
        language="en",
        cipher_id="refine_bakeoff",
        cipher_system="transposition_homophonic",
        transform_search="rank",
        homophonic_budget="full",
    )

    assert result.solver == "transform_search_homophonic_refined"
    assert result.final_decryption == "FULL_NEIGHBOR"
    refine_step = next(
        step for step in result.artifact["steps"]
        if step["name"] == "refine_selected_transform_candidate_homophonic"
    )
    assert refine_step["screen_selected_candidate_id"] == "screen_winner"
    assert refine_step["candidate_id"] == "close_neighbor"
    assert refine_step["bakeoff"]["selected_candidate_changed"] is True
    assert refine_step["bakeoff"]["refined_candidate_count"] == 2
    assert result.artifact["transform_pipeline"] == close_neighbor["pipeline"]
    assert result.artifact["transform_selection"]["source"] == "transform_search_full_refinement"
    assert result.artifact["transform_selection"]["candidate_id"] == "close_neighbor"
    assert result.artifact["transform_selection"]["screen_selected_candidate_id"] == "screen_winner"
    assert result.artifact["transform_selection"]["selected_candidate_changed"] is True


def test_full_final_refinement_cannot_replace_robust_winner_with_unstable_false_positive(monkeypatch):
    cipher_text = parse_canonical_transcription("A B C")
    screen_winner = {
        "candidate_id": "screen_winner",
        "family": "program_banded_ndown_constructed",
        "status": "completed",
        "decryption": "SCREEN_WINNER",
        "key": {"0": 0},
        "pipeline": {"steps": [{"name": "Reverse", "data": {"rangeStart": 0, "rangeEnd": 2}}]},
        "anneal_score": -7.50,
        "validated_selection_score": -7.50,
        "confirmed_selection_score": -7.50,
        "finalist_label": "robust_candidate",
        "selectable_transform_candidate": True,
    }
    unstable_neighbor = {
        "candidate_id": "unstable_neighbor",
        "family": "wide_route_repair_columns_boustrophedon_Reverse_row0_full",
        "status": "completed",
        "decryption": "SCREEN_FALSE_POSITIVE",
        "key": {"0": 0},
        "pipeline": {"steps": [{"name": "Reverse", "data": {"rangeStart": 0, "rangeEnd": 1}}]},
        "anneal_score": -7.48,
        "validated_selection_score": -7.48,
        "confirmed_selection_score": -7.48,
        "finalist_label": "unstable_false_positive",
        "selectable_transform_candidate": False,
    }

    monkeypatch.setattr(
        automated_runner,
        "inspect_transform_suspicion",
        lambda **_kwargs: {"recommendation": "run_screen"},
    )
    monkeypatch.setattr(
        automated_runner,
        "screen_transform_candidates",
        lambda *_args, **_kwargs: {"identity_candidate": {}, "top_candidates": []},
    )
    monkeypatch.setattr(
        automated_runner,
        "_rank_transform_candidates",
        lambda **_kwargs: {
            "budget": "screen",
            "selection": {
                "selected": True,
                "selected_candidate_id": "screen_winner",
            },
            "top_ranked_candidates": [unstable_neighbor, screen_winner],
        },
    )

    def fake_homophonic(cipher_text, language, budget, refinement, solver_profile, ground_truth, seed_offset=0):
        if cipher_text.source.endswith(":1"):
            return (
                "fake_homophonic",
                {0: 0},
                "FULL_FALSE_POSITIVE",
                {"name": "search_homophonic_anneal", "anneal_score": -6.0},
            )
        return (
            "fake_homophonic",
            {0: 0},
            "FULL_ROBUST",
            {"name": "search_homophonic_anneal", "anneal_score": -7.0},
        )

    monkeypatch.setattr(automated_runner, "_run_homophonic", fake_homophonic)

    result = automated_runner.run_automated(
        cipher_text=cipher_text,
        language="en",
        cipher_id="refine_unstable_guard",
        cipher_system="transposition_homophonic",
        transform_search="rank",
        homophonic_budget="full",
    )

    assert result.final_decryption == "FULL_ROBUST"
    refine_step = next(
        step for step in result.artifact["steps"]
        if step["name"] == "refine_selected_transform_candidate_homophonic"
    )
    assert refine_step["candidate_id"] == "screen_winner"
    assert refine_step["bakeoff"]["selected_candidate_changed"] is False
    refined = refine_step["bakeoff"]["refined_candidates"]
    assert {item["candidate_id"] for item in refined} == {"screen_winner", "unstable_neighbor"}
    assert next(
        item for item in refined if item["candidate_id"] == "unstable_neighbor"
    )["refinement_selectable"] is False


def test_transform_diagnostics_summarize_false_positive_evidence():
    ranked = [
        {
            "candidate_id": "candidate_unwrap",
            "family": "unwrap_transposition_reverse_key",
            "status": "completed",
            "confirmed_selection_score": -7.70,
            "anneal_score": -7.65,
            "plaintext_quality_score": 0.8,
            "structural_score": 1.0,
            "structural_delta_vs_identity": 0.5,
            "confirmation": {
                "status": "completed",
                "stability_score": 0.10,
            },
        },
        {
            "candidate_id": "000_identity",
            "family": "identity",
            "status": "completed",
            "confirmed_selection_score": -7.90,
            "anneal_score": -7.85,
            "plaintext_quality_score": 0.2,
            "structural_score": 0.4,
            "structural_delta_vs_identity": 0.0,
            "confirmation": {
                "status": "completed",
                "stability_score": 0.50,
            },
        },
    ]

    automated_runner._label_transform_finalists(ranked)
    ranked.sort(
        key=lambda item: (
            bool(item.get("selectable_transform_candidate")),
            item.get("confirmed_selection_score", float("-inf")),
        ),
        reverse=True,
    )
    selection = automated_runner._choose_transform_candidate(ranked)
    diagnostics = automated_runner._diagnose_transform_finalists(ranked, selection)

    assert diagnostics["conclusion"] == "identity_baseline_preferred_over_transform_candidates"
    assert diagnostics["unstable_false_positive_count"] == 1
    false_positive = next(
        item for item in diagnostics["top_evidence"]
        if item["candidate_id"] == "candidate_unwrap"
    )
    assert false_positive["quality_signal"] is True
    assert false_positive["stability_pass"] is False
    assert "failed_stability_gate" in false_positive["diagnostic_reasons"]


def test_transform_diagnostic_candidate_prefers_identity_baseline():
    report = {
        "rank": {
            "top_ranked_candidates": [
                {
                    "candidate_id": "unstable",
                    "status": "completed",
                    "decryption": "FALSEPOSITIVE",
                },
                {
                    "candidate_id": "000_identity",
                    "status": "completed",
                    "decryption": "BASELINE",
                },
            ],
        },
    }

    candidate = automated_runner._diagnostic_ranked_transform_candidate(report)

    assert candidate["candidate_id"] == "000_identity"


def test_transform_confirmation_honors_zero_adaptive_budget(monkeypatch):
    monkeypatch.setenv("DECIPHER_TRANSFORM_RANK_ENGINE", "python")
    ranked = [
        {
            "candidate_id": "000_identity",
            "family": "identity",
            "status": "completed",
            "pipeline": {"steps": []},
            "selection_score": -10.0,
            "validated_selection_score": -10.0,
            "structural_score": 0.0,
            "decryption": "BASELINE",
        },
        {
            "candidate_id": "near",
            "family": "route_columns_down",
            "status": "completed",
            "pipeline": {"steps": []},
            "selection_score": -10.01,
            "validated_selection_score": -10.01,
            "structural_score": 0.0,
            "decryption": "NEAR",
        },
    ]
    ct = parse_canonical_transcription("S001 S002 S003")
    calls = []

    def fake_homophonic(cipher_text, language, budget, refinement, solver_profile, ground_truth, seed_offset=0):
        calls.append(seed_offset)
        return (
            "fake_homophonic",
            {0: 0},
            "BASELINE",
            {"name": "search_homophonic_anneal", "anneal_score": -10.0},
        )

    monkeypatch.setattr(automated_runner, "_run_homophonic", fake_homophonic)

    report = automated_runner._confirm_transform_finalists(
        cipher_text=ct,
        language="en",
        ranked=ranked,
        budget="screen",
        solver_profile="zenith_native",
        confirm_count=1,
        adaptive_confirmations=0,
    )

    assert calls == [10000]
    assert report["confirmed_candidate_count"] == 1
    assert report["adaptive_confirmed_candidate_count"] == 0
    assert ranked[1]["confirmation"]["status"] == "not_run"


def test_benchmark_runner_preflight_runs_without_ground_truth(monkeypatch, tmp_path):
    seen = {}

    class FakeAPI:
        model = "fake-model"

    def fake_preflight(cipher_text, language, test_id, cipher_system):
        seen["language"] = language
        seen["test_id"] = test_id
        seen["cipher_system"] = cipher_system
        return {
            "enabled": True,
            "status": "completed",
            "solver": "fake_native",
            "summary": "preflight summary",
            "key": {"0": 19, "1": 7, "2": 4},
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

    def fake_run_v2(**kwargs):
        seen["automated_preflight"] = kwargs["automated_preflight"]
        seen["benchmark_context"] = kwargs["benchmark_context"]
        from artifact.schema import BranchSnapshot, RunArtifact

        artifact = RunArtifact(
            run_id="run123",
            cipher_id=kwargs["cipher_id"],
            model="fake-model",
            language=kwargs["language"],
            status="solved",
        )
        artifact.branches = [
            BranchSnapshot(
                name="main",
                parent=None,
                created_iteration=0,
                key={},
                mapped_count=0,
                decryption="THE",
            )
        ]
        return artifact

    monkeypatch.setattr(runner_v2, "_run_automated_preflight", fake_preflight)
    monkeypatch.setattr(runner_v2, "run_v2", fake_run_v2)

    runner = BenchmarkRunnerV2(
        claude_api=FakeAPI(),  # type: ignore[arg-type]
        artifact_dir=tmp_path,
    )
    result = runner.run_test(_test_data(), language="en")

    assert result.final_decryption == "THE"
    assert seen["test_id"] == "auto_demo"
    assert seen["language"] == "en"
    assert seen["cipher_system"] == "simple_substitution"
    assert seen["automated_preflight"]["summary"] == "preflight summary"
    assert seen["benchmark_context"].policy == "max"


def test_collapsed_plaintext_detector_flags_single_letter_failure():
    assert automated_runner._is_collapsed_plaintext("E" * 100)
    assert not automated_runner._is_collapsed_plaintext(
        "THEOLDPHOTOGRAPHHADYELLOWEDATTHEEDGESITSCOLORSFADING"
    )


def test_automated_runner_prefers_manifest_language_over_test_id_prefix():
    test_data = BenchmarkTestData(
        test=BenchmarkTest(
            test_id="parity_borg_latin_demo",
            track="transcription2plaintext",
            cipher_system="borg_lat_898",
            target_records=["rec1"],
            context_records=[],
            description="historical demo",
        ),
        canonical_transcription="A B C",
        plaintext="THE",
        plaintext_language="la",
    )

    runner = AutomatedBenchmarkRunner()

    assert runner._resolve_language(test_data) == "la"


def test_routing_prefers_homophonic_when_cipher_system_or_alphabet_demands_it():
    canonical_dense = " ".join(f"S{i:03d}" for i in range(1, 33))
    ct_dense = automated_runner.parse_canonical_transcription(canonical_dense)
    route = automated_runner._select_solver_path(ct_dense, "la", "borg_lat_898")
    assert route["route"] == "homophonic"

    canonical_simple = "A B C | D E F"
    ct_simple = automated_runner.parse_canonical_transcription(canonical_simple)
    route = automated_runner._select_solver_path(ct_simple, "en", "simple_substitution")
    assert route["route"] == "substitution"


def test_run_automated_marks_success_as_completed(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("A B C")

    def fake_run_substitution(cipher_text, language):
        return (
            "native_substitution_anneal",
            {0: 19},
            "THE",
            {"name": "search_anneal", "solver": "native_substitution_anneal"},
        )

    monkeypatch.setattr(automated_runner, "_run_substitution", fake_run_substitution)

    result = automated_runner.run_automated(
        cipher_text=ct,
        language="en",
        cipher_id="demo",
    )

    assert result.status == "completed"
    assert result.final_decryption == "THE"


def test_homophonic_score_profile_reads_env_and_exposes_weights(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_SCORE_PROFILE", "ioc_ngram")

    weights = automated_runner._homophonic_score_weights(
        automated_runner._homophonic_score_profile(),
        short_homophonic=False,
    )

    assert weights == {
        "distribution_weight": 0.0,
        "diversity_weight": 0.0,
        "ioc_weight": 1.0,
    }


def test_homophonic_score_profile_defaults_to_zenith_native_without_env(monkeypatch):
    monkeypatch.delenv("DECIPHER_HOMOPHONIC_SCORE_PROFILE", raising=False)

    assert automated_runner._homophonic_score_profile() == "zenith_native"


def test_homophonic_score_profile_can_default_to_legacy_balanced(monkeypatch):
    monkeypatch.delenv("DECIPHER_HOMOPHONIC_SCORE_PROFILE", raising=False)

    assert automated_runner._homophonic_score_profile("legacy") == "balanced"


def test_homophonic_score_config_supports_zenith_like_profile(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_SCORE_PROFILE", "zenith_like")

    config = automated_runner._homophonic_score_config(
        automated_runner._homophonic_score_profile(),
        short_homophonic=False,
    )

    assert config == {
        "distribution_weight": 0.0,
        "diversity_weight": 0.0,
        "ioc_weight": 1.0,
        "score_formula": "multiplicative_ioc",
        "window_step": 2,
    }


def test_homophonic_score_config_supports_zenith_exact_profile(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_SCORE_PROFILE", "zenith_exact")

    config = automated_runner._homophonic_score_config(
        automated_runner._homophonic_score_profile(),
        short_homophonic=False,
    )

    assert config["distribution_weight"] == 0.0
    assert config["diversity_weight"] == 0.0
    assert config["score_formula"] == "multiplicative_ioc"
    assert config["window_step"] == 2
    assert config["ioc_weight"] == 1.0 / 6.0


def test_homophonic_selection_profile_reads_env(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_SELECTION_PROFILE", "pool_rerank_v1")

    assert automated_runner._homophonic_selection_profile() == "pool_rerank_v1"


def test_homophonic_move_profile_reads_env(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_MOVE_PROFILE", "mixed_v1")

    assert automated_runner._homophonic_move_profile() == "mixed_v1"


def test_homophonic_move_profile_reads_mixed_v2(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_MOVE_PROFILE", "mixed_v2")

    assert automated_runner._homophonic_move_profile() == "mixed_v2"


def test_homophonic_move_profile_reads_mixed_v1_targeted(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_MOVE_PROFILE", "mixed_v1_targeted")

    assert automated_runner._homophonic_move_profile() == "mixed_v1_targeted"


def test_homophonic_use_early_stop_reads_env(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_EARLY_STOP", "1")

    assert automated_runner._homophonic_use_early_stop() is True


def test_homophonic_search_profile_reads_env(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_SEARCH_PROFILE", "dev")

    assert automated_runner._homophonic_search_profile() == "dev"


def test_homophonic_repair_profile_reads_env(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_REPAIR_PROFILE", "dev")

    assert automated_runner._homophonic_repair_profile() == "dev"


def test_homophonic_parallel_seed_workers_reads_env(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS", "4")

    assert automated_runner._homophonic_parallel_seed_workers() == 4


def test_homophonic_parallel_seed_workers_defaults_from_cpu_count(monkeypatch):
    monkeypatch.delenv("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS", raising=False)
    monkeypatch.setattr(automated_runner.os, "cpu_count", lambda: 8)

    assert automated_runner._homophonic_parallel_seed_workers() == 7


def test_homophonic_parallel_seed_workers_caps_to_seed_count(monkeypatch):
    monkeypatch.delenv("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS", raising=False)
    monkeypatch.setattr(automated_runner.os, "cpu_count", lambda: 8)

    assert automated_runner._homophonic_parallel_seed_workers(seed_count=4) == 4


def test_homophonic_parallel_seed_workers_env_caps_to_seed_count(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS", "8")

    assert automated_runner._homophonic_parallel_seed_workers(seed_count=3) == 3


def test_homophonic_budget_screen_reduces_search_parameters():
    full_short = automated_runner._homophonic_budget_params("full", short_homophonic=True)
    screen_short = automated_runner._homophonic_budget_params("screen", short_homophonic=True)

    assert screen_short["budget"] == "screen"
    assert len(screen_short["seeds"]) < len(full_short["seeds"])
    assert screen_short["epochs"] < full_short["epochs"]
    assert screen_short["sampler_iterations"] < full_short["sampler_iterations"]


def test_homophonic_budget_dev_profile_reduces_full_search_parameters():
    full_short = automated_runner._homophonic_budget_params(
        "full",
        short_homophonic=True,
        search_profile="full",
    )
    dev_short = automated_runner._homophonic_budget_params(
        "full",
        short_homophonic=True,
        search_profile="dev",
    )

    assert full_short["search_profile"] == "full"
    assert dev_short["search_profile"] == "dev"
    assert len(dev_short["seeds"]) < len(full_short["seeds"])
    assert dev_short["epochs"] < full_short["epochs"]
    assert dev_short["sampler_iterations"] < full_short["sampler_iterations"]


def test_homophonic_refinement_params_dev_profile_reduces_family_repair_budget():
    full = automated_runner._homophonic_refinement_params(
        "family_repair",
        "full",
        short_homophonic=True,
        repair_profile="full",
    )
    dev = automated_runner._homophonic_refinement_params(
        "family_repair",
        "full",
        short_homophonic=True,
        repair_profile="dev",
    )

    assert full["repair_profile"] == "full"
    assert dev["repair_profile"] == "dev"
    assert len(dev["repair_plans"]) < len(full["repair_plans"])
    assert dev["repair_plans"][0]["beam_limit"] < full["repair_plans"][0]["beam_limit"]
    assert dev["repair_plans"][0]["sampler_iterations"] < full["repair_plans"][0]["sampler_iterations"]
    assert dev["repair_plans"][0]["screen_limit"] == 1
    assert dev["repair_plans"][0]["min_branch_score"] == 0.0


def test_run_homophonic_records_candidate_diagnostics_and_epoch_traces(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")

    fake_result = SimpleNamespace(
        plaintext="THERETHERETHERE",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.23,
        epochs=2,
        sampler_iterations=10,
        candidates=[
            SimpleNamespace(
                plaintext="THERETHERETHERE",
                key={0: 19, 1: 7, 2: 4},
                normalized_score=-1.23,
                epoch=1,
            )
        ],
        metadata={
            "move_telemetry": {
                "single": {
                    "proposals": 10,
                    "accepted": 4,
                    "improved": 2,
                    "affected_windows": 20,
                    "avg_affected_windows": 2.0,
                    "score_time_seconds": 0.01,
                    "avg_score_time_ms": 1.0,
                }
            },
            "epoch_traces": [
                {
                    "epoch": 1,
                    "initial_normalized_score": -2.0,
                    "final_normalized_score": -1.4,
                    "best_normalized_score": -1.23,
                    "accepted_moves": 4,
                    "improved_moves": 2,
                    "unique_letters": 5,
                    "top_letter_fraction": 0.2,
                    "index_of_coincidence": 0.07,
                }
            ]
        },
    )

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(["THERETHERETHERE"], order=3),
            "test model",
        ),
    )
    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        lambda **kwargs: fake_result,
    )

    _solver, _key, _text, step = automated_runner._run_homophonic(ct, "en", solver_profile="legacy")

    assert step["homophonic_budget"] == "full"
    assert step["score_profile"] == "balanced"
    assert step["score_weights"]["distribution_weight"] == 5.0
    assert step["diagnostics"]["dict_rate"] >= 0.0
    assert step["seed_attempts"][0]["epoch_traces"][0]["epoch"] == 1
    assert step["seed_attempts"][0]["move_telemetry"]["single"]["proposals"] == 10
    assert step["move_telemetry"]["single"]["accepted"] == 4
    assert step["candidates"][0]["diagnostics"]["segmented_preview"]


def test_run_homophonic_screen_budget_passes_reduced_params(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")
    seen = {}

    fake_result = SimpleNamespace(
        plaintext="THERETHERETHERE",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[],
        metadata={"epoch_traces": []},
    )

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(["THERETHERETHERE"], order=3),
            "test model",
        ),
    )

    def fake_homophonic_simulated_anneal(**kwargs):
        seen.update({
            "epochs": kwargs["epochs"],
            "sampler_iterations": kwargs["sampler_iterations"],
        })
        return fake_result

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, _text, step = automated_runner._run_homophonic(ct, "en", budget="screen", solver_profile="legacy")

    assert seen["epochs"] == 5
    assert seen["sampler_iterations"] == 1500
    assert step["homophonic_budget"] == "screen"
    assert step["budget_params"]["budget"] == "screen"


def test_run_homophonic_passes_epoch_callback_when_early_stop_enabled(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")
    seen = {}

    fake_result = SimpleNamespace(
        plaintext="THERETHERETHERE",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[],
        metadata={"epoch_traces": [], "stopped_early": False},
    )

    monkeypatch.setenv("DECIPHER_HOMOPHONIC_EARLY_STOP", "1")
    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(["THERETHERETHERE"], order=3),
            "test model",
        ),
    )

    def fake_homophonic_simulated_anneal(**kwargs):
        seen["epoch_callback"] = kwargs.get("epoch_callback")
        return fake_result

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, _text, step = automated_runner._run_homophonic(ct, "en", solver_profile="legacy")

    assert callable(seen["epoch_callback"])
    assert step["early_stop_enabled"] is True


def test_run_homophonic_zenith_like_profile_passes_structural_scoring_args(
    monkeypatch,
):
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")
    seen = {}

    fake_result = SimpleNamespace(
        plaintext="THERETHERETHERE",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[],
        metadata={"epoch_traces": []},
    )

    monkeypatch.setenv("DECIPHER_HOMOPHONIC_SCORE_PROFILE", "zenith_like")
    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THERETHERETHERE"], order=3
            ),
            "test model",
        ),
    )

    def fake_homophonic_simulated_anneal(**kwargs):
        seen.update({
            "score_formula": kwargs["score_formula"],
            "window_step": kwargs["window_step"],
            "distribution_weight": kwargs["distribution_weight"],
            "diversity_weight": kwargs["diversity_weight"],
            "ioc_weight": kwargs["ioc_weight"],
        })
        return fake_result

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, _text, step = automated_runner._run_homophonic(ct, "en", solver_profile="legacy")

    assert seen["score_formula"] == "multiplicative_ioc"
    assert seen["window_step"] == 2
    assert seen["distribution_weight"] == 0.0
    assert seen["diversity_weight"] == 0.0
    assert seen["ioc_weight"] == 1.0
    assert step["score_profile"] == "zenith_like"
    assert step["score_formula"] == "multiplicative_ioc"
    assert step["window_step"] == 2


def test_run_homophonic_zenith_exact_profile_passes_exact_ioc_exponent(
    monkeypatch,
):
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")
    seen = {}

    fake_result = SimpleNamespace(
        plaintext="THERETHERETHERE",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[],
        metadata={"epoch_traces": []},
    )

    monkeypatch.setenv("DECIPHER_HOMOPHONIC_SCORE_PROFILE", "zenith_exact")
    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THERETHERETHERE"], order=3
            ),
            "test model",
        ),
    )

    def fake_homophonic_simulated_anneal(**kwargs):
        seen.update({
            "score_formula": kwargs["score_formula"],
            "window_step": kwargs["window_step"],
            "distribution_weight": kwargs["distribution_weight"],
            "diversity_weight": kwargs["diversity_weight"],
            "ioc_weight": kwargs["ioc_weight"],
        })
        return fake_result

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, _text, step = automated_runner._run_homophonic(ct, "en", solver_profile="legacy")

    assert seen["score_formula"] == "multiplicative_ioc"
    assert seen["window_step"] == 2
    assert seen["distribution_weight"] == 0.0
    assert seen["diversity_weight"] == 0.0
    assert seen["ioc_weight"] == 1.0 / 6.0
    assert step["score_profile"] == "zenith_exact"
    assert step["score_formula"] == "multiplicative_ioc"
    assert step["window_step"] == 2


def test_run_homophonic_move_profile_is_passed_to_annealer(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")
    seen = {}

    fake_result = SimpleNamespace(
        plaintext="THERETHERETHERE",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[],
        metadata={"epoch_traces": []},
    )

    monkeypatch.setenv("DECIPHER_HOMOPHONIC_MOVE_PROFILE", "mixed_v1")
    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THERETHERETHERE"], order=3
            ),
            "test model",
        ),
    )

    def fake_homophonic_simulated_anneal(**kwargs):
        seen["move_profile"] = kwargs["move_profile"]
        return fake_result

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, _text, step = automated_runner._run_homophonic(ct, "en", solver_profile="legacy")

    assert seen["move_profile"] == "mixed_v1"
    assert step["move_profile"] == "mixed_v1"


def test_run_homophonic_zenith_native_can_parallelize_across_seeds(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS", "2")
    monkeypatch.setattr(automated_runner, "_zenith_native_model_path", lambda language: "fake-model.bin")

    class FakeCandidate(SimpleNamespace):
        pass

    def make_candidate(seed):
        return FakeCandidate(
            plaintext=f"THERE{seed}",
            key={0: 19, 1: 7, 2: 4},
            normalized_score=-1.0 + (0.1 * seed),
            epochs=5,
            sampler_iterations=1500,
        )

    class FakeFuture:
        def __init__(self, seed):
            self.seed = seed

        def result(self):
            return make_candidate(self.seed)

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, **kwargs):
            return FakeFuture(kwargs["seed"])

    monkeypatch.setattr(automated_runner.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        automated_runner.concurrent.futures,
        "as_completed",
        lambda futures: list(reversed(list(futures))),
    )
    monkeypatch.setattr(
        automated_runner,
        "_word_list",
        lambda language: ["THERE"],
    )
    monkeypatch.setattr(
        automated_runner,
        "_plaintext_quality",
        lambda plaintext, key: {
            "ok": True,
            "collapsed": False,
            "penalty": 0.0,
            "reasons": [],
            "letter_count": len(plaintext),
            "unique_letters": len(set(plaintext)),
            "top_letter_fraction": 0.2,
            "key_plaintext_letters": len(set(key.values())),
            "monogram_chi_per_letter": 0.1,
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_automated_candidate_diagnostics",
        lambda plaintext, language, word_list: {
            "letter_count": len(plaintext),
            "unique_letters": len(set(plaintext)),
            "top_letter_fraction": 0.2,
            "index_of_coincidence": 0.07,
            "dict_rate": 0.9,
            "segmentation_cost": 20.0,
            "segmented_preview": plaintext,
            "pseudo_word_count": 1,
        },
    )

    _solver, _key, text, step = automated_runner._run_homophonic_zenith_native(
        cipher_text=ct,
        language="en",
        budget="screen",
        ground_truth=None,
        pt_alpha=automated_runner._plaintext_alphabet("en"),
        plaintext_ids=list(range(automated_runner._plaintext_alphabet("en").size)),
        id_to_letter={i: automated_runner._plaintext_alphabet("en").symbol_for(i).upper() for i in range(automated_runner._plaintext_alphabet("en").size)},
        letter_to_id={automated_runner._plaintext_alphabet("en").symbol_for(i).upper(): i for i in range(automated_runner._plaintext_alphabet("en").size)},
        short_homophonic=True,
        budget_params={"seeds": [0, 1], "epochs": 5, "sampler_iterations": 1500, "budget": "screen"},
        started=0.0,
    )

    assert text == "THERE1"
    assert step["parallel_seed_workers"] == 2
    assert [attempt["seed"] for attempt in step["seed_attempts"]] == [0, 1]


def test_collect_anchor_symbols_returns_symbols_of_matching_words():
    # CAT = [100, 101, 102], DOG = [103, 104, 105], XYZ = [200, 201, 202]
    cipher_words = [[100, 101, 102], [103, 104, 105], [200, 201, 202]]
    id_to_letter = {i: chr(ord("A") + i) for i in range(26)}
    # CAT=2,0,19; DOG=3,14,6; XYZ=23,24,25
    key = {
        100: 2, 101: 0, 102: 19,
        103: 3, 104: 14, 105: 6,
        200: 23, 201: 24, 202: 25,
    }
    anchors, words = automated_runner._collect_anchor_symbols(
        cipher_words, key, id_to_letter, word_set={"CAT", "DOG"}
    )
    assert anchors == {100, 101, 102, 103, 104, 105}
    assert sorted(words) == ["CAT", "DOG"]


def test_collect_anchor_symbols_skips_short_words():
    cipher_words = [[100, 101]]  # length 2 — ignored by default min_word_len=3
    id_to_letter = {i: chr(ord("A") + i) for i in range(26)}
    key = {100: 8, 101: 13}  # IN
    anchors, words = automated_runner._collect_anchor_symbols(
        cipher_words, key, id_to_letter, word_set={"IN"}
    )
    assert anchors == set()
    assert words == []


def test_maybe_repair_zenith_native_key_no_op_without_word_boundaries(monkeypatch):
    # Single word group (no '|') means repair is skipped.
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")
    id_to_letter = {i: chr(ord("A") + i) for i in range(26)}
    letter_to_id = {v: k for k, v in id_to_letter.items()}
    info = automated_runner._maybe_repair_zenith_native_key(
        cipher_text=ct,
        bin_path=Path("/nonexistent-bin"),
        key={0: 0, 1: 1, 2: 2},
        plaintext="ABCABC",
        language="en",
        word_list=[],
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
    )
    assert info["applied"] is False
    assert info["reason"] == "no_word_boundaries"


def test_maybe_repair_zenith_native_key_fixes_wrong_symbol_mapping(monkeypatch):
    # Word-delimited ciphertext: two words "CAT" and "DOG".
    ct = automated_runner.parse_canonical_transcription("01 02 03 | 04 05 06")
    # Token ids in ct.alphabet: 01→0, 02→1, 03→2, 04→3, 05→4, 06→5
    monkeypatch.setattr(
        automated_runner.dictionary, "get_dictionary_path", lambda language: "fake"
    )
    monkeypatch.setattr(
        automated_runner.dictionary, "load_word_set", lambda path: {"CAT", "DOG"}
    )

    id_to_letter = {i: chr(ord("A") + i) for i in range(26)}
    letter_to_id = {v: k for k, v in id_to_letter.items()}
    # Start with a wrong mapping for the middle of "CAT": symbol 1 → 'X' (should be 'A')
    key = {
        0: letter_to_id["C"],
        1: letter_to_id["X"],
        2: letter_to_id["T"],
        3: letter_to_id["D"],
        4: letter_to_id["O"],
        5: letter_to_id["G"],
    }
    info = automated_runner._maybe_repair_zenith_native_key(
        cipher_text=ct,
        bin_path=Path("/nonexistent-bin"),  # forces score_fn guard to no-op
        key=key,
        plaintext="CXTDOG",
        language="en",
        word_list=["CAT", "DOG"],
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        min_word_len=3,  # test uses 3-char words; production default is 5
    )
    assert info["applied"] is True
    assert info["key"][1] == letter_to_id["A"]
    assert info["plaintext"] == "CATDOG"
    assert info["after_dict_hits"] > info["before_dict_hits"]


def test_run_key_consistent_repair_rejects_fix_that_drops_model_score(monkeypatch):
    """The language-model guard should block greedy dict-only fixes.

    Regression scenario from Borg 0109v: a candidate repair raises dict-hit
    count but cuts the n-gram score. Without the guard the repair was applied
    and the final decryption got worse. With the guard, it is rejected.
    """
    ct = automated_runner.parse_canonical_transcription("01 02 03 04 05 | 06 07 08 09 10")
    monkeypatch.setattr(
        automated_runner.dictionary, "get_dictionary_path", lambda language: "fake"
    )
    # LATER is in the dict; current key decodes word 1 as "LATEE" (1-edit away).
    monkeypatch.setattr(
        automated_runner.dictionary, "load_word_set", lambda path: {"LATER", "WORLD"}
    )
    id_to_letter = {i: chr(ord("A") + i) for i in range(26)}
    letter_to_id = {v: k for k, v in id_to_letter.items()}
    # Initial key: first word decodes "LATEE" (pseudo), second decodes "WORLD" (dict).
    key = {
        0: letter_to_id["L"],
        1: letter_to_id["A"],
        2: letter_to_id["T"],
        3: letter_to_id["E"],
        4: letter_to_id["E"],  # shared cipher symbol 4 → E
        5: letter_to_id["W"],
        6: letter_to_id["O"],
        7: letter_to_id["R"],
        8: letter_to_id["L"],
        9: letter_to_id["D"],
    }

    def falling_score_fn(k: dict[int, int]) -> float:
        # Any deviation from initial key scores worse than baseline.
        return 0.0 if k == key else -10.0

    info = automated_runner._run_key_consistent_repair(
        cipher_text=ct,
        key=key,
        language="en",
        word_list=["LATER", "WORLD"],
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        score_fn=falling_score_fn,
        min_word_len=3,
    )
    # Dict-only, LATEE → LATER would remap symbol 4 (E→R), but that tanks the
    # model score — the guard rejects it and the repair is a no-op overall.
    assert info["applied"] is False


def test_maybe_polish_zenith_native_plaintext_repairs_segmented_words(monkeypatch):
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_POLISH", "1")
    monkeypatch.setattr(automated_runner.dictionary, "get_dictionary_path", lambda language: "fake-dict")
    monkeypatch.setattr(
        automated_runner.dictionary,
        "load_word_set",
        lambda path: {"THERE", "CAT"},
    )

    info = automated_runner._maybe_polish_zenith_native_plaintext(
        "THERQCAT",
        language="en",
        word_list=["THERE", "CAT"],
    )

    assert info["applied"] is True
    assert info["plaintext"] == "THERE CAT"
    assert info["before"]["pseudo_word_count"] >= 1
    assert info["after"]["dict_rate"] == 1.0
    assert info["corrections"][0]["from"] == "THERQ"
    assert info["corrections"][0]["to"] == "THERE"


def test_run_homophonic_zenith_native_can_adopt_polished_plaintext(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")
    monkeypatch.setenv("DECIPHER_ZENITH_NATIVE_ENGINE", "python")
    monkeypatch.delenv("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS", raising=False)
    monkeypatch.setattr(automated_runner, "_zenith_native_model_path", lambda language: "fake-model.bin")

    candidate = SimpleNamespace(
        plaintext="THERQCAT",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.0,
        epochs=5,
        sampler_iterations=1500,
    )

    monkeypatch.setattr(zenith_solver, "load_zenith_binary_model", lambda path: object())
    monkeypatch.setattr(zenith_solver, "zenith_solve", lambda **kwargs: candidate)
    monkeypatch.setattr(automated_runner, "_word_list", lambda language: ["THERE", "CAT"])
    monkeypatch.setattr(
        automated_runner,
        "_plaintext_quality",
        lambda plaintext, key: {
            "ok": True,
            "collapsed": False,
            "penalty": 0.0,
            "reasons": [],
            "letter_count": len(plaintext),
            "unique_letters": len(set(plaintext)),
            "top_letter_fraction": 0.2,
            "key_plaintext_letters": len(set(key.values())),
            "monogram_chi_per_letter": 0.1,
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_automated_candidate_diagnostics",
        lambda plaintext, language, word_list: {
            "letter_count": len(plaintext),
            "unique_letters": len(set(plaintext)),
            "top_letter_fraction": 0.2,
            "index_of_coincidence": 0.07,
            "dict_rate": 1.0 if plaintext == "THERE CAT" else 0.5,
            "segmentation_cost": 10.0 if plaintext == "THERE CAT" else 20.0,
            "segmented_preview": plaintext,
            "pseudo_word_count": 0 if plaintext == "THERE CAT" else 1,
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_maybe_polish_zenith_native_plaintext",
        lambda plaintext, language, word_list: {
            "enabled": True,
            "applied": True,
            "mode": "segment_one_edit_local",
            "rounds": 1,
            "corrections": [{"from": "THERQ", "to": "THERE"}],
            "plaintext": "THERE CAT",
            "key_consistent_with_output": False,
        },
    )

    _solver, _key, text, step = automated_runner._run_homophonic_zenith_native(
        cipher_text=ct,
        language="en",
        budget="screen",
        ground_truth=None,
        pt_alpha=automated_runner._plaintext_alphabet("en"),
        plaintext_ids=list(range(automated_runner._plaintext_alphabet("en").size)),
        id_to_letter={i: automated_runner._plaintext_alphabet("en").symbol_for(i).upper() for i in range(automated_runner._plaintext_alphabet("en").size)},
        letter_to_id={automated_runner._plaintext_alphabet("en").symbol_for(i).upper(): i for i in range(automated_runner._plaintext_alphabet("en").size)},
        short_homophonic=True,
        budget_params={"seeds": [0], "epochs": 5, "sampler_iterations": 1500, "budget": "screen"},
        started=0.0,
    )

    assert text == "THERE CAT"
    assert step["postprocess"]["applied"] is True
    assert step["diagnostics"]["dict_rate"] == 1.0


def test_run_homophonic_two_stage_refinement_adopts_only_better_candidate(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")
    calls = []

    stage1 = SimpleNamespace(
        plaintext="THERETHERETHERE",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[],
        metadata={"epoch_traces": []},
    )
    stage2 = SimpleNamespace(
        plaintext="THEREBETTERTHERE",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.0,
        epochs=1,
        sampler_iterations=900,
        candidates=[],
        metadata={"epoch_traces": [{"epoch": 1}]},
    )

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THERETHERETHERE", "THEREBETTERTHERE"], order=3
            ),
            "test model",
        ),
    )
    monkeypatch.setattr(
        automated_runner,
        "_automated_candidate_diagnostics",
        lambda plaintext, language, word_list: {"dict_rate": 1.0, "segmented_preview": plaintext},
    )

    def fake_quality(plaintext, key):
        if plaintext == "THERETHERETHERE":
            return {"ok": True, "collapsed": False, "penalty": 0.0, "reasons": []}
        return {"ok": True, "collapsed": False, "penalty": 0.0, "reasons": []}

    monkeypatch.setattr(automated_runner, "_plaintext_quality", fake_quality)

    def fake_homophonic_simulated_anneal(**kwargs):
        calls.append(kwargs)
        return stage1 if len(calls) < 5 else stage2

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, text, step = automated_runner._run_homophonic(
        ct,
        "en",
        budget="screen",
        refinement="two_stage",
    solver_profile="legacy",
    )

    assert len(calls) == 5
    assert calls[-1]["initial_key"] == stage1.key
    assert calls[-1]["epochs"] == 1
    assert step["refinement"]["mode"] == "two_stage"
    assert step["refinement"]["adopted"] is True
    assert text == "THEREBETTERTHERE"


def test_run_homophonic_targeted_repair_freezes_non_suspicious_symbols(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 04 01 02 03 04")
    calls = []

    stage1 = SimpleNamespace(
        plaintext="THEREXXX",
        key={0: 19, 1: 7, 2: 4, 3: 17},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[],
        metadata={"epoch_traces": [], "move_telemetry": {}},
    )
    repaired = SimpleNamespace(
        plaintext="THEREBET",
        key={0: 19, 1: 7, 2: 4, 3: 1},
        normalized_score=-1.0,
        epochs=3,
        sampler_iterations=900,
        candidates=[],
        metadata={"epoch_traces": [{"epoch": 1}], "move_telemetry": {}},
    )

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THEREXXX", "THEREBET"], order=3
            ),
            "test model",
        ),
    )
    monkeypatch.setattr(
        automated_runner,
        "_automated_candidate_diagnostics",
        lambda plaintext, language, word_list: {"dict_rate": 1.0, "segmented_preview": plaintext},
    )
    monkeypatch.setattr(
        automated_runner,
        "_suspicious_homophonic_symbols",
        lambda cipher_text, key, id_to_letter, model, limit, window_step=1: [2, 3][:limit],
    )

    def fake_quality(plaintext, key):
        return {"ok": True, "collapsed": False, "penalty": 0.0, "reasons": []}

    monkeypatch.setattr(automated_runner, "_plaintext_quality", fake_quality)

    def fake_homophonic_simulated_anneal(**kwargs):
        calls.append(kwargs)
        return stage1 if len(calls) < 5 else repaired

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, text, step = automated_runner._run_homophonic(
        ct,
        "en",
        budget="screen",
        refinement="targeted_repair",
    solver_profile="legacy",
    )

    assert calls[-1]["fixed_cipher_ids"] == {0, 1}
    assert calls[-1]["move_profile"] == "mixed_v1_targeted"
    assert step["refinement"]["mode"] == "targeted_repair"
    assert step["refinement"]["selected_plan"] == "targeted8"
    assert step["refinement"]["adopted"] is True
    assert text == "THEREBET"


def test_run_homophonic_family_repair_skips_unreadable_candidates(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 04 01 02 03 04")

    stage1 = SimpleNamespace(
        plaintext="QZXWQZXW",
        key={0: 19, 1: 7, 2: 4, 3: 17},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[],
        metadata={"epoch_traces": [], "move_telemetry": {}},
    )

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THEREXXX", "THEREBET"], order=3
            ),
            "test model",
        ),
    )
    monkeypatch.setattr(
        automated_runner,
        "_automated_candidate_diagnostics",
        lambda plaintext, language, word_list: {
            "dict_rate": 0.2,
            "segmentation_cost": 80.0,
            "letter_count": max(1, len(plaintext)),
            "segmented_preview": plaintext,
        },
    )

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        lambda **kwargs: stage1,
    )

    _solver, _key, _text, step = automated_runner._run_homophonic(
        ct,
        "en",
        budget="screen",
        refinement="family_repair",
    solver_profile="legacy",
    )

    assert step["refinement"]["mode"] == "family_repair"
    assert step["refinement"]["skipped"] is True
    assert step["refinement"]["gate"]["ok"] is False
    assert "dictionary_rate_too_low" in step["refinement"]["gate"]["reasons"]


def test_run_homophonic_family_repair_freezes_non_family_symbols(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 04 01 02 03 04")
    calls = []

    stage1 = SimpleNamespace(
        plaintext="THEREXXX",
        key={0: 19, 1: 7, 2: 4, 3: 17},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[],
        metadata={"epoch_traces": [], "move_telemetry": {}},
    )
    repaired = SimpleNamespace(
        plaintext="THEREBET",
        key={0: 19, 1: 7, 2: 1, 3: 4},
        normalized_score=-1.0,
        epochs=3,
        sampler_iterations=900,
        candidates=[],
        metadata={"epoch_traces": [{"epoch": 1}], "move_telemetry": {}},
    )

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THEREXXX", "THEREBET"], order=3
            ),
            "test model",
        ),
    )

    def fake_diagnostics(plaintext, language, word_list):
        if plaintext == "THEREXXX":
            return {
                "dict_rate": 0.9,
                "segmentation_cost": 24.0,
                "letter_count": 8,
                "segmented_preview": plaintext,
            }
        return {
            "dict_rate": 1.0,
            "segmentation_cost": 16.0,
            "letter_count": 8,
            "segmented_preview": plaintext,
        }

    monkeypatch.setattr(automated_runner, "_automated_candidate_diagnostics", fake_diagnostics)
    monkeypatch.setattr(
        automated_runner,
        "_homophonic_family_diagnostics",
        lambda *args, **kwargs: {
            "window_step": 1,
            "family_count": 3,
            "families": [
                {"letter": "L", "symbol_ids": [2], "suspicion_score": 9.0},
                {"letter": "B", "symbol_ids": [3], "suspicion_score": 8.0},
                {"letter": "T", "symbol_ids": [0, 1], "suspicion_score": 1.0},
            ],
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_family_competition_proposals",
        lambda *args, **kwargs: [
            {
                "kind": "single_symbol_reassign",
                "score": 1.5,
                "source_letter": "L",
                "target_letter": "B",
                "trigger_symbol": 2,
                "mutable_symbols": [2, 3],
                "branch_updates": {2: 1},
                "description": "L -> B via symbol 2",
            }
        ],
    )

    def fake_quality(plaintext, key):
        return {"ok": True, "collapsed": False, "penalty": 0.0, "reasons": []}

    monkeypatch.setattr(automated_runner, "_plaintext_quality", fake_quality)

    def fake_homophonic_simulated_anneal(**kwargs):
        calls.append(kwargs)
        return stage1 if len(calls) < 5 else repaired

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, text, step = automated_runner._run_homophonic(
        ct,
        "en",
        budget="screen",
        refinement="family_repair",
    solver_profile="legacy",
    )

    assert calls[-1]["fixed_cipher_ids"] == {0, 1}
    assert calls[-1]["initial_key"][2] == 1
    assert calls[-1]["move_profile"] == "mixed_v1_targeted"
    assert step["refinement"]["mode"] == "family_repair"
    assert step["refinement"]["selected_plan"] == "family2"
    assert step["refinement"]["selected_branch"]["source_letter"] == "L"
    assert step["refinement"]["selected_branch"]["target_letter"] == "B"
    assert step["refinement"]["gate"]["ok"] is True
    assert step["refinement"]["adopted"] is True
    assert text == "THEREBET"


def test_run_homophonic_family_repair_requires_epsilon_to_adopt(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 04 01 02 03 04")
    calls = []

    stage1 = SimpleNamespace(
        plaintext="THEREXXX",
        key={0: 19, 1: 7, 2: 4, 3: 17},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[],
        metadata={"epoch_traces": [], "move_telemetry": {}},
    )
    repaired = SimpleNamespace(
        plaintext="THEREXXX",
        key={0: 19, 1: 7, 2: 4, 3: 17},
        normalized_score=-1.22995,
        epochs=3,
        sampler_iterations=900,
        candidates=[],
        metadata={"epoch_traces": [{"epoch": 1}], "move_telemetry": {}},
    )

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THEREXXX", "THEREBET"], order=3
            ),
            "test model",
        ),
    )
    monkeypatch.setattr(
        automated_runner,
        "_automated_candidate_diagnostics",
        lambda plaintext, language, word_list: {
            "dict_rate": 0.9,
            "segmentation_cost": 24.0,
            "letter_count": 8,
            "segmented_preview": plaintext,
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_homophonic_family_diagnostics",
        lambda *args, **kwargs: {
            "window_step": 1,
            "family_count": 2,
            "families": [
                {"letter": "L", "symbol_ids": [2], "suspicion_score": 9.0},
                {"letter": "B", "symbol_ids": [3], "suspicion_score": 8.0},
            ],
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_family_competition_proposals",
        lambda *args, **kwargs: [
            {
                "kind": "single_symbol_reassign",
                "score": 0.1,
                "source_letter": "L",
                "target_letter": "B",
                "trigger_symbol": 2,
                "mutable_symbols": [2, 3],
                "branch_updates": {2: 1},
                "description": "L -> B via symbol 2",
            }
        ],
    )
    monkeypatch.setattr(
        automated_runner,
        "_plaintext_quality",
        lambda plaintext, key: {"ok": True, "collapsed": False, "penalty": 0.0, "reasons": []},
    )

    def fake_homophonic_simulated_anneal(**kwargs):
        calls.append(kwargs)
        return stage1 if len(calls) < 5 else repaired

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, text, step = automated_runner._run_homophonic(
        ct,
        "en",
        budget="screen",
        refinement="family_repair",
    solver_profile="legacy",
    )

    assert step["refinement"]["adoption_epsilon"] == 1e-4
    assert step["refinement"]["adopted"] is False
    assert text == "THEREXXX"


def test_family_competition_proposals_can_emit_two_symbol_split(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 04 01 02 03 04")
    family_report = {
        "families": [
            {
                "letter": "S",
                "symbol_ids": [2, 3],
                "observed_fraction": 0.20,
                "overuse": 0.12,
                "score_spread": 1.5,
            },
            {
                "letter": "L",
                "symbol_ids": [0],
                "observed_fraction": 0.01,
                "overuse": 0.0,
                "score_spread": 0.1,
            },
            {
                "letter": "F",
                "symbol_ids": [1],
                "observed_fraction": 0.01,
                "overuse": 0.0,
                "score_spread": 0.1,
            },
        ]
    }
    key = {0: 11, 1: 5, 2: 18, 3: 18}
    id_to_letter = {i: chr(ord("A") + i) for i in range(26)}
    letter_to_id = {v: k for k, v in id_to_letter.items()}

    def fake_alts(cipher_text, key, id_to_letter, letter_to_id, model, sid, candidate_letters, window_step=1, top_k=4):
        if sid == 2:
            return [
                {"symbol_id": 2, "from_letter": "S", "to_letter": "L", "local_delta": -0.5},
                {"symbol_id": 2, "from_letter": "S", "to_letter": "F", "local_delta": -0.7},
            ]
        return [
            {"symbol_id": 3, "from_letter": "S", "to_letter": "F", "local_delta": -0.4},
            {"symbol_id": 3, "from_letter": "S", "to_letter": "L", "local_delta": -0.6},
        ]

    monkeypatch.setattr(automated_runner, "_symbol_letter_alternatives", fake_alts)

    proposals = automated_runner._family_competition_proposals(
        ct,
        key,
        family_report,
        id_to_letter,
        letter_to_id,
        model=SimpleNamespace(order=3),
        window_step=1,
        family_limit=1,
        top_k_per_symbol=2,
        beam_limit=6,
    )

    assert any(proposal["kind"] == "two_symbol_split" for proposal in proposals)
    assert any(proposal["kind"] == "single_symbol_reassign" for proposal in proposals)


def test_select_diverse_homophonic_elites_prefers_distinct_plaintexts_and_signatures():
    candidates = [
        {
            "plaintext": "AAAAABBBBB",
            "selection_score": 10.0,
            "anneal_score": 10.0,
            "key": {0: 0, 1: 0, 2: 1},
        },
        {
            "plaintext": "AAAAABBBBC",
            "selection_score": 9.9,
            "anneal_score": 9.9,
            "key": {0: 0, 1: 0, 2: 1},
        },
        {
            "plaintext": "XYZXYZXYZX",
            "selection_score": 9.8,
            "anneal_score": 9.8,
            "key": {0: 2, 1: 3, 2: 4},
        },
    ]

    elites = automated_runner._select_diverse_homophonic_elites(
        candidates,
        limit=3,
        min_plaintext_distance=0.15,
    )

    assert len(elites) == 2
    assert elites[0]["plaintext"] == "AAAAABBBBB"
    assert elites[1]["plaintext"] == "XYZXYZXYZX"


def test_select_diverse_homophonic_elites_prefers_distinct_seeds_first():
    candidates = [
        {
            "plaintext": "THERETHERETHERE",
            "selection_score": 10.0,
            "anneal_score": 10.0,
            "key": {0: 19, 1: 7, 2: 4},
            "seed": 0,
        },
        {
            "plaintext": "THEREBETTHERE",
            "selection_score": 9.95,
            "anneal_score": 9.95,
            "key": {0: 19, 1: 7, 2: 1},
            "seed": 0,
        },
        {
            "plaintext": "OTHERBASINTEST",
            "selection_score": 9.8,
            "anneal_score": 9.8,
            "key": {0: 14, 1: 19, 2: 7},
            "seed": 1,
        },
    ]

    elites = automated_runner._select_diverse_homophonic_elites(
        candidates,
        limit=2,
        min_plaintext_distance=0.05,
    )

    assert len(elites) == 2
    assert [elite["seed"] for elite in elites] == [0, 1]


def test_run_homophonic_family_repair_can_select_non_primary_elite(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 04 01 02 03 04")
    calls = []

    stage1 = SimpleNamespace(
        plaintext="THEREXXX",
        key={0: 19, 1: 7, 2: 4, 3: 17},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[
            SimpleNamespace(
                plaintext="THEREXXX",
                key={0: 19, 1: 7, 2: 4, 3: 17},
                normalized_score=-1.23,
                epoch=1,
            ),
            SimpleNamespace(
                plaintext="THEREALT",
                key={0: 19, 1: 7, 2: 11, 3: 19},
                normalized_score=-1.231,
                epoch=2,
            ),
        ],
        metadata={"epoch_traces": [], "move_telemetry": {}},
    )
    repaired_bad = SimpleNamespace(
        plaintext="THEREXXX",
        key={0: 19, 1: 7, 2: 4, 3: 17},
        normalized_score=-1.23,
        epochs=3,
        sampler_iterations=900,
        candidates=[],
        metadata={"epoch_traces": [{"epoch": 1}], "move_telemetry": {}},
    )
    repaired_good = SimpleNamespace(
        plaintext="THEREBET",
        key={0: 19, 1: 7, 2: 1, 3: 4},
        normalized_score=-1.0,
        epochs=3,
        sampler_iterations=900,
        candidates=[],
        metadata={"epoch_traces": [{"epoch": 1}], "move_telemetry": {}},
    )

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THEREXXX", "THEREALT", "THEREBET"], order=3
            ),
            "test model",
        ),
    )

    monkeypatch.setattr(
        automated_runner,
        "_automated_candidate_diagnostics",
        lambda plaintext, language, word_list: {
            "dict_rate": 0.9,
            "segmentation_cost": 24.0,
            "letter_count": 8,
            "segmented_preview": plaintext,
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_homophonic_family_diagnostics",
        lambda *args, **kwargs: {
            "window_step": 1,
            "family_count": 2,
            "families": [
                {"letter": "L", "symbol_ids": [2], "suspicion_score": 9.0},
                {"letter": "B", "symbol_ids": [3], "suspicion_score": 8.0},
            ],
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_family_competition_proposals",
        lambda cipher_text, key, *args, **kwargs: [
            {
                "kind": "single_symbol_reassign",
                "score": 1.0,
                "source_letter": "L",
                "target_letter": "B",
                "trigger_symbol": 2,
                "mutable_symbols": [2, 3],
                "branch_updates": {2: 1},
                "description": "L -> B via symbol 2",
            }
        ],
    )
    monkeypatch.setattr(
        automated_runner,
        "_plaintext_quality",
        lambda plaintext, key: {"ok": True, "collapsed": False, "penalty": 0.0, "reasons": []},
    )

    def fake_homophonic_simulated_anneal(**kwargs):
        calls.append(kwargs)
        if len(calls) < 5:
            return stage1
        initial_key = kwargs.get("initial_key", {})
        if initial_key.get(2) == 1 and initial_key.get(3) == 19:
            return repaired_good
        return repaired_bad

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, text, step = automated_runner._run_homophonic(
        ct,
        "en",
        budget="screen",
        refinement="family_repair",
    solver_profile="legacy",
    )

    assert len(step["elite_candidates"]) >= 2
    assert step["refinement"]["selected_elite_rank"] is not None
    assert step["refinement"]["adopted"] is True
    assert text == "THEREBET"


def test_run_homophonic_dev_repair_profile_limits_elite_pool(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 04 01 02 03 04")
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_REPAIR_PROFILE", "dev")

    stage1 = SimpleNamespace(
        plaintext="THEREXXX",
        key={0: 19, 1: 7, 2: 4, 3: 17},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[
            SimpleNamespace(
                plaintext="THEREXXX",
                key={0: 19, 1: 7, 2: 4, 3: 17},
                normalized_score=-1.23,
                epoch=1,
            ),
            SimpleNamespace(
                plaintext="THEREALT",
                key={0: 19, 1: 7, 2: 11, 3: 19},
                normalized_score=-1.231,
                epoch=2,
            ),
            SimpleNamespace(
                plaintext="THEREBET",
                key={0: 19, 1: 7, 2: 1, 3: 4},
                normalized_score=-1.232,
                epoch=3,
            ),
        ],
        metadata={"epoch_traces": [], "move_telemetry": {}},
    )

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THEREXXX", "THEREALT", "THEREBET"], order=3
            ),
            "test model",
        ),
    )
    monkeypatch.setattr(
        automated_runner,
        "_automated_candidate_diagnostics",
        lambda plaintext, language, word_list: {
            "dict_rate": 0.9,
            "segmentation_cost": 24.0,
            "letter_count": 8,
            "segmented_preview": plaintext,
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_homophonic_family_diagnostics",
        lambda *args, **kwargs: {
            "window_step": 1,
            "family_count": 2,
            "families": [
                {"letter": "L", "symbol_ids": [2], "suspicion_score": 9.0},
                {"letter": "B", "symbol_ids": [3], "suspicion_score": 8.0},
            ],
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_family_competition_proposals",
        lambda cipher_text, key, *args, **kwargs: [
            {
                "kind": "single_symbol_reassign",
                "score": 1.0,
                "source_letter": "L",
                "target_letter": "B",
                "trigger_symbol": 2,
                "mutable_symbols": [2, 3],
                "branch_updates": {2: 1},
                "description": "L -> B via symbol 2",
            }
        ],
    )
    monkeypatch.setattr(
        automated_runner,
        "_plaintext_quality",
        lambda plaintext, key: {"ok": True, "collapsed": False, "penalty": 0.0, "reasons": []},
    )
    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        lambda **kwargs: stage1,
    )

    _solver, _key, _text, step = automated_runner._run_homophonic(
        ct,
        "en",
        budget="full",
        refinement="family_repair",
    solver_profile="legacy",
    )

    assert step["repair_profile"] == "dev"
    assert len(step["elite_candidates"]) == 2
    assert step["elite_seed_count"] >= 1
    assert step["refinement"]["repair_plans"][0]["beam_limit"] == 2


def test_run_homophonic_dev_repair_profile_screens_weak_branches(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 04 01 02 03 04")
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_REPAIR_PROFILE", "dev")
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_SEARCH_PROFILE", "dev")

    stage1 = SimpleNamespace(
        plaintext="THEREXXX",
        key={0: 19, 1: 7, 2: 4, 3: 17},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[
            SimpleNamespace(
                plaintext="THEREXXX",
                key={0: 19, 1: 7, 2: 4, 3: 17},
                normalized_score=-1.23,
                epoch=1,
            ),
            SimpleNamespace(
                plaintext="THEREALT",
                key={0: 19, 1: 7, 2: 11, 3: 19},
                normalized_score=-1.231,
                epoch=2,
            ),
        ],
        metadata={"epoch_traces": [], "move_telemetry": {}},
    )
    calls = []

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THEREXXX", "THEREALT", "THEREBET"], order=3
            ),
            "test model",
        ),
    )
    monkeypatch.setattr(
        automated_runner,
        "_automated_candidate_diagnostics",
        lambda plaintext, language, word_list: {
            "dict_rate": 0.9,
            "segmentation_cost": 24.0,
            "letter_count": 8,
            "segmented_preview": plaintext,
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_homophonic_family_diagnostics",
        lambda *args, **kwargs: {
            "window_step": 1,
            "family_count": 2,
            "families": [
                {"letter": "L", "symbol_ids": [2], "suspicion_score": 9.0},
                {"letter": "B", "symbol_ids": [3], "suspicion_score": 8.0},
            ],
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_family_competition_proposals",
        lambda *args, **kwargs: [
            {
                "kind": "single_symbol_reassign",
                "score": 1.5,
                "source_letter": "L",
                "target_letter": "B",
                "trigger_symbol": 2,
                "mutable_symbols": [2, 3],
                "branch_updates": {2: 1},
                "description": "L -> B via symbol 2",
            },
            {
                "kind": "single_symbol_reassign",
                "score": -0.5,
                "source_letter": "L",
                "target_letter": "F",
                "trigger_symbol": 2,
                "mutable_symbols": [2, 3],
                "branch_updates": {2: 5},
                "description": "L -> F via symbol 2",
            },
        ],
    )
    monkeypatch.setattr(
        automated_runner,
        "_plaintext_quality",
        lambda plaintext, key: {"ok": True, "collapsed": False, "penalty": 0.0, "reasons": []},
    )

    def fake_homophonic_simulated_anneal(**kwargs):
        calls.append(kwargs)
        return stage1

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, _text, step = automated_runner._run_homophonic(
        ct,
        "en",
        budget="full",
        refinement="family_repair",
    solver_profile="legacy",
    )

    assert len(step["elite_candidates"]) == 2
    assert len(step["refinement"]["screening"]) == 2
    assert step["refinement"]["screening"][0]["candidate_branch_count"] == 2
    assert step["refinement"]["screening"][0]["screened_branch_count"] == 1
    assert step["refinement"]["screening"][0]["min_branch_score"] == 0.0
    assert step["refinement"]["screening"][0]["screen_limit"] == 1
    assert len(calls) == 6


def test_run_homophonic_dev_search_profile_reduces_seed_budget(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 04 01 02 03 04")
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_SEARCH_PROFILE", "dev")

    stage1 = SimpleNamespace(
        plaintext="THEREXXX",
        key={0: 19, 1: 7, 2: 4, 3: 17},
        normalized_score=-1.23,
        epochs=5,
        sampler_iterations=1500,
        candidates=[
            SimpleNamespace(
                plaintext="THEREXXX",
                key={0: 19, 1: 7, 2: 4, 3: 17},
                normalized_score=-1.23,
                epoch=1,
            )
        ],
        metadata={"epoch_traces": [], "move_telemetry": {}},
    )
    calls = []

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(["THEREXXX"], order=3),
            "test model",
        ),
    )
    monkeypatch.setattr(
        automated_runner,
        "_automated_candidate_diagnostics",
        lambda plaintext, language, word_list: {
            "dict_rate": 0.9,
            "segmentation_cost": 24.0,
            "letter_count": 8,
            "segmented_preview": plaintext,
        },
    )
    monkeypatch.setattr(
        automated_runner,
        "_plaintext_quality",
        lambda plaintext, key: {"ok": True, "collapsed": False, "penalty": 0.0, "reasons": []},
    )

    def fake_homophonic_simulated_anneal(**kwargs):
        calls.append(kwargs)
        return stage1

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, _text, step = automated_runner._run_homophonic(
        ct,
        "en",
        budget="full",
        refinement="none",
    solver_profile="legacy",
    )

    assert step["search_profile"] == "dev"
    assert step["budget_params"]["search_profile"] == "dev"
    assert step["budget_params"]["seeds"] == [0, 1, 2, 3]
    assert step["budget_params"]["epochs"] == 5
    assert step["budget_params"]["sampler_iterations"] == 1800
    assert len(calls) == 4


def test_run_homophonic_pool_rerank_can_override_best_anneal_seed(monkeypatch):
    ct = automated_runner.parse_canonical_transcription("01 02 03 01 02 03")
    monkeypatch.setenv("DECIPHER_HOMOPHONIC_SELECTION_PROFILE", "pool_rerank_v1")

    candidate_a = SimpleNamespace(
        plaintext="NOISENOISENOISE",
        key={0: 13, 1: 14, 2: 8},
        normalized_score=-1.0,
        epoch=1,
    )
    candidate_b = SimpleNamespace(
        plaintext="THERETHERETHERE",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.2,
        epoch=2,
    )

    seed0 = SimpleNamespace(
        plaintext="NOISENOISENOISE",
        key={0: 13, 1: 14, 2: 8},
        normalized_score=-1.0,
        epochs=2,
        sampler_iterations=10,
        candidates=[candidate_a],
        metadata={"epoch_traces": []},
    )
    seed1 = SimpleNamespace(
        plaintext="THERETHERETHERE",
        key={0: 19, 1: 7, 2: 4},
        normalized_score=-1.2,
        epochs=2,
        sampler_iterations=10,
        candidates=[candidate_b],
        metadata={"epoch_traces": []},
    )

    monkeypatch.setattr(
        automated_runner,
        "_homophonic_model",
        lambda language, word_list: (
            automated_runner.homophonic.build_continuous_ngram_model(
                ["THERETHERETHERE", "NOISENOISENOISE"], order=3
            ),
            "test model",
        ),
    )

    def fake_quality(plaintext, key):
        return {
            "ok": True,
            "collapsed": False,
            "penalty": 0.0,
            "reasons": [],
            "letter_count": len(plaintext),
            "unique_letters": len(set(plaintext)),
            "top_letter_fraction": 0.2,
            "key_plaintext_letters": len(set(key.values())),
            "monogram_chi_per_letter": 0.1,
        }

    def fake_diagnostics(plaintext, language, word_list):
        if plaintext == "NOISENOISENOISE":
            return {
                "letter_count": len(plaintext),
                "unique_letters": 5,
                "top_letter_fraction": 0.45,
                "index_of_coincidence": 0.18,
                "dict_rate": 0.1,
                "segmentation_cost": 120.0,
                "segmented_preview": "NOISE NOISE NOISE",
                "pseudo_word_count": 8,
            }
        return {
            "letter_count": len(plaintext),
            "unique_letters": 4,
            "top_letter_fraction": 0.2,
            "index_of_coincidence": 0.07,
            "dict_rate": 0.9,
            "segmentation_cost": 20.0,
            "segmented_preview": "THERE THERE THERE",
            "pseudo_word_count": 1,
        }

    monkeypatch.setattr(automated_runner, "_plaintext_quality", fake_quality)
    monkeypatch.setattr(automated_runner, "_automated_candidate_diagnostics", fake_diagnostics)

    def fake_homophonic_simulated_anneal(**kwargs):
        return seed0 if kwargs["seed"] == 0 else seed1

    monkeypatch.setattr(
        automated_runner.homophonic,
        "homophonic_simulated_anneal",
        fake_homophonic_simulated_anneal,
    )

    _solver, _key, text, step = automated_runner._run_homophonic(ct, "en", budget="screen", solver_profile="legacy")

    assert text == "THERETHERETHERE"
    assert step["selection_profile"] == "pool_rerank_v1"
    assert step["selection"]["selected_seed"] == 1
    assert step["selection"]["top_candidates"][0]["preview"].startswith("THERETHERETHERE")
