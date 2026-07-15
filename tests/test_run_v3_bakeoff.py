"""Unit tests for the M6 bake-off runner (scripts/run_v3_bakeoff.py).

Pure-logic only: cell enumeration, resume, checkpoint math, report aggregation,
and summary-row construction. NO live LLM calls, NO benchmark/testgen loading.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_v3_bakeoff as bake  # noqa: E402


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------
def test_enumerate_cells_matrix_shape():
    cells = bake.enumerate_cells()
    assert len(cells) == 42
    on = [c for c in cells if c.preflight]
    off = [c for c in cells if not c.preflight]
    assert len(on) == 30
    assert len(off) == 12
    # Preflight-OFF only for the two designated cases.
    assert {c.case for c in off} == {"borg_single_B_borg_0109v", "synth_en_250nb_s4"}
    # Every case has exactly 2 loops x 3 replicates preflight-ON.
    for case in bake.CASE_BY_NAME:
        got = [c for c in on if c.case == case]
        assert len(got) == 6
        assert {c.loop for c in got} == {"v2", "v3"}
        assert sorted(c.replicate for c in got) == [1, 1, 2, 2, 3, 3]
    # Keys are unique (preflight disambiguates the on/off pairs).
    assert len({c.key() for c in cells}) == 42


def test_cell_artifact_subdir_layout():
    on = bake.Cell("v3", "borg_single_B_borg_0109v", True, 2)
    off = bake.Cell("v3", "borg_single_B_borg_0109v", False, 2)
    assert on.artifact_subdir() == "v3/borg_single_B_borg_0109v/2"
    assert off.artifact_subdir() == "v3/borg_single_B_borg_0109v__noprefl/2"
    # The on/off pair keeps distinct trees AND distinct resume keys.
    assert on.key() != off.key()


def test_per_cell_estimate_and_total():
    cells = bake.enumerate_cells()
    total = sum(bake.per_cell_estimate(c) for c in cells)
    # Estimates recalibrated 2026-07-14 to observed cost (Borg $4, Copiale $3.5,
    # synth $1.5): full 3-replicate matrix now totals $120.
    assert abs(total - 120.0) < 1e-6
    assert bake.per_cell_estimate("synth_en_250nb_s4") == 1.5
    assert bake.per_cell_estimate("borg_single_B_borg_0045v") == 4.0


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
def test_resume_skips_completed_keys(tmp_path):
    cells = bake.enumerate_cells()
    # Mark the first 5 cells complete in a summary file.
    summary = tmp_path / "summary.jsonl"
    for c in cells[:5]:
        bake.append_summary_row(summary, {
            "loop": c.loop, "case": c.case,
            "preflight": c.preflight, "replicate": c.replicate,
            "cost_usd": 1.0,
        })
    rows = bake.load_summary_rows(summary)
    assert len(rows) == 5
    done = bake.completed_keys(rows)
    todo = bake.pending_cells(cells, done)
    assert len(todo) == 37
    assert all(c.key() not in done for c in todo)
    # A completed ON cell does NOT skip its OFF sibling.
    on = bake.Cell("v2", "borg_single_B_borg_0109v", True, 1)
    off = bake.Cell("v2", "borg_single_B_borg_0109v", False, 1)
    done2 = {on.key()}
    remaining = bake.pending_cells(cells, done2)
    assert off in remaining and on not in remaining


# ---------------------------------------------------------------------------
# Checkpoint math (F9c)
# ---------------------------------------------------------------------------
def _rows_for_checkpoint(n, cost_each, case="borg_single_B_borg_0109v"):
    return [{"case": case, "cost_usd": cost_each} for _ in range(n)]


def test_checkpoint_no_stop_before_run_21():
    # 20 runs at 10x estimate -> still no check (trigger is after run 21).
    rows = _rows_for_checkpoint(20, 25.0)  # borg est 2.5, actual 25 = 10x
    stop, msg = bake.checkpoint_should_stop(rows)
    assert stop is False and msg == ""


def test_checkpoint_stops_when_over_30pct_after_21():
    # 21 borg runs; estimate 21*4.0=84; actual 21*5.5=115.5 = +37% -> stop.
    rows = _rows_for_checkpoint(21, 5.5)
    stop, msg = bake.checkpoint_should_stop(rows)
    assert stop is True
    assert "HALFWAY CHECKPOINT" in msg and "Stopping launches" in msg


def test_checkpoint_no_stop_when_within_estimate():
    # 21 borg runs at exactly the estimate -> within threshold, no stop.
    rows = _rows_for_checkpoint(21, 2.5)
    stop, _msg = bake.checkpoint_should_stop(rows)
    assert stop is False
    # 21 runs at +20% -> still under the 30% bar.
    rows2 = _rows_for_checkpoint(21, 3.0)  # 3.0/2.5 = +20%
    assert bake.checkpoint_should_stop(rows2)[0] is False


# ---------------------------------------------------------------------------
# Summary-row construction
# ---------------------------------------------------------------------------
def _fake_result(**kw):
    base = dict(
        test_id="t", status="solved", char_accuracy=0.9, word_accuracy=0.8,
        estimated_cost_usd=2.1, iterations_used=12, elapsed_seconds=88.0,
        self_confidence=0.7,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_build_summary_row_v3_extracts_attestation_and_hint():
    cell = bake.Cell("v3", "borg_single_B_borg_0109v", True, 2)
    artifact = {
        "run_id": "abc123",
        "total_input_tokens": 5000,
        "total_output_tokens": 800,
        "total_cache_read_tokens": 300,
        "solution": {"branch": "main", "attestation": {
            "reader_accepts": True, "coherence": 8, "created_turn": 19}},
        "loop_events": [
            {"event": "iteration_start"},
            {"event": "late_turn_attestation_hint", "payload": {"branch": "main"}},
        ],
    }
    row = bake.build_summary_row(cell, _fake_result(), artifact)
    assert row["loop"] == "v3" and row["preflight"] is True and row["replicate"] == 2
    assert row["run_id"] == "abc123"
    assert row["prompt_tokens"] == 5000 and row["completion_tokens"] == 800
    assert row["cache_read_tokens"] == 300
    assert row["declared"] is True and row["fallback"] is False
    assert row["attested"] is True
    assert row["verify_reader_accepts"] is True and row["verify_coherence"] == 8
    assert row["verify_created_turn"] == 19
    assert row["late_turn_hint_fired"] is True


def test_build_summary_row_v2_has_no_attestation_and_flags_fallback():
    cell = bake.Cell("v2", "borg_single_B_borg_0045v", False, 1)
    artifact = {"run_id": "z", "total_input_tokens": 10, "total_output_tokens": 2}
    row = bake.build_summary_row(
        cell, _fake_result(status="fallback_declared"), artifact)
    assert row["attested"] is None
    assert row["verify_reader_accepts"] is None
    assert row["late_turn_hint_fired"] is False
    assert row["declared"] is False and row["fallback"] is True


# ---------------------------------------------------------------------------
# Report aggregation (Part C)
# ---------------------------------------------------------------------------
_V2_CHAR = {
    "borg_single_B_borg_0109v": 0.90, "borg_single_B_borg_0045v": 0.85,
    "copiale_single_B_copiale_p017": 0.0, "synth_en_250nb_s4": 0.95,
    "synth_en_200honb_s6": 0.30,
}
_V2_WORD = {"borg_single_B_borg_0109v": 0.80, "borg_single_B_borg_0045v": 0.75}
_COST = {"borg": 2.5, "copiale": 2.5, "synth": 1.0}


def _cost_for(case):
    return _COST[bake.CASE_BY_NAME[case].kind]


def _make_decision_rows(v3_char_delta=0.02, v3_word_delta=0.02, v3_cost_factor=0.7):
    """30 preflight-ON rows (5 cases x 2 loops x 3 replicates)."""
    rows = []
    for case in bake.CASE_BY_NAME:
        for rep in (1, 2, 3):
            v2c = _V2_CHAR[case]
            v2w = _V2_WORD.get(case, 0.0)
            cost = _cost_for(case)
            rows.append(_row("v2", case, True, rep, v2c, v2w, cost))
            rows.append(_row(
                "v3", case, True, rep,
                v2c + v3_char_delta, v2w + v3_word_delta, cost * v3_cost_factor))
    return rows


def _row(loop, case, preflight, rep, char, word, cost, **extra):
    row = {
        "loop": loop, "case": case, "preflight": preflight, "replicate": rep,
        "char_accuracy": char, "word_accuracy": word, "cost_usd": cost,
        "declared": True, "fallback": False,
        "prompt_tokens": 1000, "completion_tokens": 200, "cache_read_tokens": 50,
        "iterations": 10, "wall_clock_seconds": 60.0,
    }
    row.update(extra)
    return row


def test_decision_aggregate_case_weighting_and_word_over_borg_only():
    rows = _make_decision_rows()
    report = bake.aggregate(rows)
    d = report["decision"]
    assert d["n_preflight_on_rows"] == 30
    v2 = d["per_loop"]["v2"]
    v3 = d["per_loop"]["v3"]
    # char = mean of 5 per-case means (equal weight).
    assert abs(v2["char_case_weighted"] - (0.90 + 0.85 + 0.0 + 0.95 + 0.30) / 5) < 1e-9
    assert abs(v3["char_case_weighted"] - (0.92 + 0.87 + 0.02 + 0.97 + 0.32) / 5) < 1e-9
    # word aggregates ONLY over the two Borg pages.
    assert abs(v2["word_case_weighted"] - (0.80 + 0.75) / 2) < 1e-9
    assert abs(v3["word_case_weighted"] - (0.82 + 0.77) / 2) < 1e-9


def test_decision_ignores_preflight_off_rows():
    rows = _make_decision_rows()
    # Add wild preflight-OFF rows; they must NOT move the decision aggregate.
    for rep in (1, 2, 3):
        rows.append(_row("v3", "borg_single_B_borg_0109v", False, rep, 0.0, 0.0, 99.0))
    d = bake.aggregate(rows)["decision"]
    assert d["n_preflight_on_rows"] == 30
    v3 = d["per_loop"]["v3"]
    assert abs(v3["char_case_weighted"] - (0.92 + 0.87 + 0.02 + 0.97 + 0.32) / 5) < 1e-9


def test_verdict_switch_true_when_v3_better_and_cheaper():
    rows = _make_decision_rows(v3_cost_factor=0.7)  # ~30% cheaper
    v = bake.aggregate(rows)["decision"]["verdict"]
    assert v["accuracy_ok"] is True
    assert v["cost_ok"] is True
    assert v["switch"] is True


def test_verdict_no_switch_when_accuracy_wins_but_cost_ties():
    rows = _make_decision_rows(v3_cost_factor=1.0)  # same cost
    v = bake.aggregate(rows)["decision"]["verdict"]
    assert v["accuracy_ok"] is True
    assert v["cost_ok"] is False
    assert v["switch"] is False
    assert any("not cost" in r for r in v["reasons"])


def test_verdict_no_switch_when_v3_less_accurate():
    rows = _make_decision_rows(v3_char_delta=-0.10, v3_cost_factor=0.5)
    v = bake.aggregate(rows)["decision"]["verdict"]
    assert v["accuracy_ok"] is False
    assert v["switch"] is False


def test_per_cell_stats_mean_min_max():
    rows = [
        _row("v3", "borg_single_B_borg_0109v", True, 1, 0.80, 0.70, 2.0,
             attested=True, verify_reader_accepts=True, verify_coherence=8,
             verify_created_turn=20, late_turn_hint_fired=True),
        _row("v3", "borg_single_B_borg_0109v", True, 2, 0.90, 0.75, 2.5,
             attested=True, verify_reader_accepts=False, verify_coherence=4,
             verify_created_turn=None, late_turn_hint_fired=False),
        _row("v3", "borg_single_B_borg_0109v", True, 3, 1.00, 0.80, 3.0,
             attested=False, verify_reader_accepts=None, verify_coherence=None,
             verify_created_turn=None, late_turn_hint_fired=False),
    ]
    cells = bake.aggregate(rows)["per_cell"]
    assert len(cells) == 1
    c = cells[0]
    assert c["n"] == 3
    assert abs(c["char_mean"] - 0.90) < 1e-9
    assert c["char_min"] == 0.80 and c["char_max"] == 1.00
    assert c["attested_n"] == 2
    assert c["reader_accepts_n"] == 1
    assert abs(c["coherence_mean"] - 6.0) < 1e-9  # mean of {8,4}
    assert c["verify_turn_mean"] == 20            # only the one non-None
    assert c["late_turn_hint_n"] == 1


def test_format_report_and_matrix_render_without_error():
    rows = _make_decision_rows()
    text = bake.format_report(bake.aggregate(rows))
    assert "Decision aggregate" in text and "VERDICT" in text
    matrix = bake.format_matrix_table(bake.enumerate_cells())
    assert "42 runs" in matrix and "120.00" in matrix


# ---------------------------------------------------------------------------
# Review fix 1 — a "stopped" run records nothing and exits 130
# ---------------------------------------------------------------------------
def test_replicates_flag_narrows_enumeration_for_breadth():
    # Breadth-first pass: 1 replicate/cell covers every case x loop x preflight
    # once (5 cases x 2 loops = 10 ON + 2 preflight-off cases x 2 loops = 4 OFF).
    cells = bake.enumerate_cells(replicates=range(1, 2))
    assert len(cells) == 14
    assert {c.replicate for c in cells} == {1}
    # every case is represented on both loops
    covered = {(c.case, c.loop) for c in cells}
    for case in bake.CASES:
        assert (case.name, "v2") in covered and (case.name, "v3") in covered
    # full 3-replicate matrix is still 42
    assert len(bake.enumerate_cells(replicates=range(1, 4))) == 42


def test_funding_probe_ok_returns_none():
    from agent.model_provider import ModelResponse, ModelUsage

    class _OKApi:
        def send(self, **kw):
            return ModelResponse(content=[], usage=ModelUsage())

    assert bake.probe_model_funding(_OKApi()) is None


def test_funding_probe_reports_provider_error():
    from agent.model_provider import ModelProviderError

    class _BrokeApi:
        def send(self, **kw):
            raise ModelProviderError(
                "Error code: 429 - insufficient_quota: exceeded your current quota"
            )

    reason = bake.probe_model_funding(_BrokeApi())
    assert reason is not None and "insufficient_quota" in reason


def test_process_completed_run_stopped_records_nothing_and_exits_130(tmp_path, capsys):
    cell = bake.Cell("v3", "borg_single_B_borg_0109v", True, 1)
    summary = tmp_path / "summary.jsonl"
    completed: list[dict] = []
    code = bake.process_completed_run(
        cell, _fake_result(status="stopped"), {}, summary, completed
    )
    assert code == bake.EXIT_INTERRUPTED == 130
    # NOTHING recorded: no summary file, no in-memory row -> --resume retries.
    assert not summary.exists()
    assert completed == []
    assert "--resume" in capsys.readouterr().err


def test_process_completed_run_records_row_and_returns_none(tmp_path):
    cell = bake.Cell("v2", "synth_en_250nb_s4", True, 2)
    summary = tmp_path / "summary.jsonl"
    completed: list[dict] = []
    code = bake.process_completed_run(
        cell, _fake_result(status="solved"), {"run_id": "r1"}, summary, completed
    )
    assert code is None
    rows = bake.load_summary_rows(summary)
    assert len(rows) == 1 and rows[0]["status"] == "solved"
    assert len(completed) == 1


def test_process_completed_run_checkpoint_stop_after_21(tmp_path):
    cell = bake.Cell("v2", "borg_single_B_borg_0109v", True, 3)
    summary = tmp_path / "summary.jsonl"
    # 20 prior over-budget borg rows; the 21st completes then trips the check.
    # Borg estimate is $4.0; $5.5/run is +37% -> over the 30% threshold.
    completed = [
        {"case": "borg_single_B_borg_0109v", "cost_usd": 5.5} for _ in range(20)
    ]
    code = bake.process_completed_run(
        cell, _fake_result(estimated_cost_usd=5.5), {}, summary, completed
    )
    assert code == bake.EXIT_CHECKPOINT_STOP
    # The 21st run IS recorded (it completed) — only launching stops.
    assert len(bake.load_summary_rows(summary)) == 1
    assert len(completed) == 21


# ---------------------------------------------------------------------------
# Review fix 2 — incomplete/asymmetric matrix can never switch
# ---------------------------------------------------------------------------
def test_verdict_incomplete_matrix_blocks_switch():
    """The demonstrated hazard: all copiale v3 rows missing (the post-checkpoint
    -stop state) with v3 winning everywhere else must NOT switch."""
    rows = [
        r for r in _make_decision_rows(v3_cost_factor=0.5)
        if not (r["loop"] == "v3" and r["case"] == "copiale_single_B_copiale_p017")
    ]
    assert len(rows) == 27
    d = bake.aggregate(rows)["decision"]
    assert d["complete"] is False
    assert d["completeness"]["v3"]["missing"] == [
        "copiale_single_B_copiale_p017#r1",
        "copiale_single_B_copiale_p017#r2",
        "copiale_single_B_copiale_p017#r3",
    ]
    v = d["verdict"]
    assert v["switch"] is False
    assert v["complete"] is False
    assert any("incomplete matrix" in r for r in v["reasons"])


def test_verdict_complete_matrix_reports_complete():
    d = bake.aggregate(_make_decision_rows())["decision"]
    assert d["complete"] is True
    assert d["verdict"]["complete"] is True
    assert all(not info["missing"] for info in d["completeness"].values())


# ---------------------------------------------------------------------------
# Review fix 3 — zero-cost guard is symmetric (both polarities)
# ---------------------------------------------------------------------------
def test_cost_guard_zero_v3_cost_cannot_switch():
    rows = _make_decision_rows(v3_cost_factor=0.0)  # v3 cost 0.0 = extraction bug
    v = bake.aggregate(rows)["decision"]["verdict"]
    assert v["cost_ok"] is None
    assert v["switch"] is False
    assert any("positive mean cost" in r for r in v["reasons"])


def test_cost_guard_zero_v2_cost_cannot_switch():
    rows = []
    for case in bake.CASE_BY_NAME:
        for rep in (1, 2, 3):
            rows.append(_row("v2", case, True, rep, 0.5, 0.5, 0.0))  # v2 cost 0.0
            rows.append(_row("v3", case, True, rep, 0.9, 0.9, 1.0))
    v = bake.aggregate(rows)["decision"]["verdict"]
    assert v["cost_ok"] is None
    assert v["switch"] is False


# ---------------------------------------------------------------------------
# Review fix 4a — upfront validation exits before any run
# ---------------------------------------------------------------------------
def test_validate_all_cases_reports_every_broken_case(tmp_path):
    empty_cache = tmp_path / "empty_cache"       # created empty by PlaintextCache
    bogus_bench = tmp_path / "no_benchmark_here"
    errors = bake.validate_all_cases(str(bogus_bench), str(empty_cache))
    # ALL 5 cases fail: 2 synth cache misses + 3 benchmark lookups.
    assert len(errors) == 5
    blob = "\n".join(errors)
    assert "cache MISS" in blob
    assert "synth_en_250nb_s4" in blob and "synth_en_200honb_s6" in blob
    assert "borg_single_B_borg_0109v" in blob


def test_main_exits_before_any_run_on_missing_cache(tmp_path, capsys):
    summary = tmp_path / "summary.jsonl"
    rc = bake.main([
        "--summary", str(summary),
        "--cache-dir", str(tmp_path / "empty_cache"),
        "--benchmark-root", str(tmp_path / "no_benchmark"),
        "--artifact-root", str(tmp_path / "artifacts"),
    ])
    assert rc == bake.EXIT_CACHE_MISS
    # Exited BEFORE any run: no summary row was ever written.
    assert not summary.exists()
    err = capsys.readouterr().err
    assert "validation failed" in err and "no runs launched" in err


# ---------------------------------------------------------------------------
# Review fix 5 — existing summary without --resume is refused
# ---------------------------------------------------------------------------
def test_main_refuses_existing_summary_without_resume(tmp_path, capsys):
    summary = tmp_path / "summary.jsonl"
    summary.write_text('{"loop":"v2"}\n', encoding="utf-8")
    rc = bake.main(["--summary", str(summary)])
    assert rc == bake.EXIT_SUMMARY_EXISTS
    assert "--resume" in capsys.readouterr().err
    # File untouched (no double-count risk).
    assert summary.read_text(encoding="utf-8") == '{"loop":"v2"}\n'


# ---------------------------------------------------------------------------
# Review fix 6 — ~ expansion on every path argument
# ---------------------------------------------------------------------------
def test_normalize_args_expands_tilde():
    args = bake.build_parser().parse_args([
        "--benchmark-root", "~/bench",
        "--artifact-root", "~/arts",
        "--summary", "~/sum.jsonl",
        "--cache-dir", "~/cache",
    ])
    args = bake._normalize_args(args)
    home = os.path.expanduser("~")
    assert args.benchmark_root == os.path.join(home, "bench")
    assert args.artifact_root == os.path.join(home, "arts")
    assert args.summary == os.path.join(home, "sum.jsonl")
    assert args.cache_dir == os.path.join(home, "cache")
