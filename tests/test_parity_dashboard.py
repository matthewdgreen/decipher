from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from artifact.dashboard import build_dashboard, render_markdown


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_dashboard_joins_agent_external_and_split_metadata(tmp_path):
    benchmark = tmp_path / "benchmark"
    split = benchmark / "splits" / "parity.jsonl"
    split.parent.mkdir(parents=True)
    split.write_text(
        json.dumps({
            "test_id": "case1",
            "track": "transcription2plaintext",
            "cipher_system": "homophonic_substitution",
            "target_records": [],
            "context_records": [],
            "description": "demo",
            "parity_family": "homophonic_en",
        }) + "\n",
        encoding="utf-8",
    )

    agent = tmp_path / "artifacts" / "case1" / "agent.json"
    _write_json(agent, {
        "cipher_id": "case1",
        "cipher_alphabet_size": 57,
        "cipher_word_count": 1,
        "status": "solved",
        "char_accuracy": 0.99,
        "word_accuracy": 0.0,
        "estimated_cost_usd": 0.05,
        "branches": [{"name": "main", "char_accuracy": 0.99}],
        "solution": {"branch": "main", "declared_at_iteration": 2},
        "tool_calls": [
            {"iteration": 1, "tool_name": "search_homophonic_anneal", "result": "{}"},
            {"iteration": 2, "tool_name": "meta_declare_solution", "result": "{}"},
        ],
    })
    external = tmp_path / "artifacts" / "external_baselines" / "case1" / "zenith" / "artifact.json"
    _write_json(external, {
        "test_id": "case1",
        "solver": "zenith",
        "status": "completed",
        "char_accuracy": 0.95,
        "word_accuracy": 0.0,
        "elapsed": 12.0,
        "command": ["zenith"],
    })
    automated = tmp_path / "artifacts" / "automated_only" / "case1" / "artifact.json"
    _write_json(automated, {
        "test_id": "case1",
        "run_mode": "automated_only",
        "solver": "automated_native",
        "status": "solved",
        "char_accuracy": 0.98,
        "word_accuracy": 0.0,
        "elapsed_seconds": 4.0,
    })

    rows = build_dashboard(
        agent_paths=[agent],
        external_paths=[external],
        automated_paths=[automated],
        benchmark_root=benchmark,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.family == "homophonic_en"
    assert row.full_agent_best is not None
    assert row.full_agent_best.char_accuracy == 0.99
    assert row.native_best is not None
    assert row.native_best.solver == "native_homophonic_anneal"
    assert row.automated_best is not None
    assert row.automated_best.solver == "automated_native"
    assert row.external_best is not None
    assert row.external_best.solver == "zenith"
    assert row.next_action == "parity ok"


def test_dashboard_recommends_classifying_gap_when_external_beats_agent(tmp_path):
    agent = tmp_path / "artifacts" / "case2" / "agent.json"
    _write_json(agent, {
        "cipher_id": "case2",
        "cipher_alphabet_size": 26,
        "cipher_word_count": 1,
        "status": "solved",
        "char_accuracy": 0.70,
        "branches": [{"name": "main", "char_accuracy": 0.70}],
        "solution": {"branch": "main", "declared_at_iteration": 5},
        "tool_calls": [],
    })
    external = tmp_path / "external" / "artifact.json"
    _write_json(external, {
        "test_id": "case2",
        "solver": "external",
        "status": "completed",
        "char_accuracy": 0.95,
        "command": ["solver"],
    })

    automated = tmp_path / "automated" / "artifact.json"
    _write_json(automated, {
        "test_id": "case2",
        "run_mode": "automated_only",
        "solver": "automated",
        "status": "solved",
        "char_accuracy": 0.80,
    })

    rows = build_dashboard([agent], [external], [automated])

    assert rows[0].next_action == "classify automated parity gap"
    assert "case2" in render_markdown(rows)


def test_dashboard_uses_best_agent_run_labels_not_stale_failures(tmp_path):
    bad = tmp_path / "artifacts" / "case3" / "bad.json"
    _write_json(bad, {
        "cipher_id": "case3",
        "cipher_alphabet_size": 57,
        "cipher_word_count": 1,
        "status": "solved",
        "char_accuracy": 0.80,
        "branches": [{"name": "main", "char_accuracy": 0.80}],
        "solution": {"branch": "main", "declared_at_iteration": 5},
        "tool_calls": [
            {"iteration": 4, "tool_name": "search_homophonic_anneal", "result": "{}"},
        ],
    })
    good = tmp_path / "artifacts" / "case3" / "good.json"
    _write_json(good, {
        "cipher_id": "case3",
        "cipher_alphabet_size": 57,
        "cipher_word_count": 1,
        "status": "solved",
        "char_accuracy": 1.0,
        "branches": [{"name": "main", "char_accuracy": 1.0}],
        "solution": {"branch": "main", "declared_at_iteration": 2},
        "tool_calls": [
            {"iteration": 1, "tool_name": "search_homophonic_anneal", "result": "{}"},
            {"iteration": 2, "tool_name": "meta_declare_solution", "result": "{}"},
        ],
    })

    rows = build_dashboard([tmp_path / "artifacts"])

    assert rows[0].full_agent_best is not None
    assert rows[0].full_agent_best.path.endswith("good.json")
    assert rows[0].gap_labels == {}


def test_dashboard_requests_automated_baseline_before_claiming_parity(tmp_path):
    agent = tmp_path / "artifacts" / "case4" / "agent.json"
    _write_json(agent, {
        "cipher_id": "case4",
        "cipher_alphabet_size": 26,
        "cipher_word_count": 1,
        "status": "solved",
        "char_accuracy": 1.0,
        "branches": [{"name": "main", "char_accuracy": 1.0}],
        "solution": {"branch": "main", "declared_at_iteration": 2},
        "tool_calls": [
            {"iteration": 1, "tool_name": "search_anneal", "result": "{}"},
            {"iteration": 2, "tool_name": "meta_declare_solution", "result": "{}"},
        ],
    })
    external = tmp_path / "external" / "artifact.json"
    _write_json(external, {
        "test_id": "case4",
        "solver": "external",
        "status": "completed",
        "char_accuracy": 1.0,
        "command": ["solver"],
    })

    rows = build_dashboard([agent], [external])

    assert rows[0].next_action == "run automated-only baseline"
    assert "Automated Only" in render_markdown(rows)
