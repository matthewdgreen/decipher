#!/usr/bin/env python
"""Agent-loop v3 bake-off matrix runner (M6 Part A.3 + Part B/C).

Drives the M6 bake-off matrix PROGRAMMATICALLY through ``BenchmarkRunnerV2``
(the ``agent_loop`` param selects v2/v3; ``automated_preflight`` is the
constructor flag for the preflight-OFF pair) — no shelling to the CLI. Two data
sources are mixed: ``BenchmarkLoader`` for the Borg/Copiale pages and ``testgen``
for the two synthetics (F3).

Matrix (F1 — "seed" = REPLICATE index 1..3; NO seed plumbing, per-run variance
comes from agent-side sampling):

    5 cases x 2 loops x 3 replicates                     = 30 preflight-ON runs
    + 2 cases x 2 loops x 3 replicates (preflight-OFF)   = 12 runs
                                                          -> 42 runs total

Robustness: an incremental JSONL summary row is appended after EVERY run (so a
mid-matrix network death never loses completed cells); ``--resume`` skips
completed (loop, case, preflight, replicate) keys; ``--dry-run`` prints the
matrix + per-cell cost estimate; ``--report`` re-aggregates the summary JSONL
into the Part-C decision table (deterministic re-aggregation).

Halfway checkpoint (F9c): after run 21, cumulative actual spend vs cumulative
per-completed-cell estimate; >30% over -> STOP LAUNCHING and exit with a
distinct code (``EXIT_CHECKPOINT_STOP``).

The Part-B/C RUNS are the orchestrator's to launch; this module is the harness +
its aggregation logic. The pure helpers (cell enumeration, resume, checkpoint
math, aggregation) are unit-tested with SYNTHETIC rows — no live LLM calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Distinct exit codes so the orchestrator can react programmatically.
EXIT_OK = 0
EXIT_CHECKPOINT_STOP = 3
EXIT_CACHE_MISS = 4          # in-run cache miss OR upfront data validation failure
EXIT_SUMMARY_EXISTS = 5      # summary file already exists and --resume not passed
EXIT_API_UNFUNDED = 6        # funding/reachability probe failed before any run
EXIT_BUDGET_STOP = 7         # focused run stopped before exceeding its cost ceiling
EXIT_INTERRUPTED = 130       # a run returned status="stopped" (Ctrl-C mid-run)

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_BENCHMARK_ROOT = "~/Dropbox/src2/cipher_benchmark/benchmark"
DEFAULT_ARTIFACT_ROOT = "artifacts/m6_bakeoff"
DEFAULT_CACHE_DIR = "testgen_cache"

LOOPS = ("v2", "v3")
REPLICATES = (1, 2, 3)

CHECKPOINT_AFTER = 21          # F9c: "after run 21"
CHECKPOINT_OVER_FRACTION = 0.30  # >30% over estimate -> stop launching


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    name: str                # also the benchmark test_id (synth: the built id)
    kind: str                # "borg" | "copiale" | "synth"
    est_cost: float          # per-run cost estimate (USD)
    word_meaningful: bool    # word accuracy is meaningful (the two Borg pages)
    preflight_off: bool      # also run a preflight-OFF pair for this case
    split_file: str | None = None    # benchmark split (borg/copiale)
    preset: str | None = None        # testgen preset (synth)
    seed: int | None = None          # testgen seed/replicate id (synth)


# F8: Copiale case pinned to copiale_p017 (lowest DTA-default floor).
# est_cost recalibrated 2026-07-14 to OBSERVED run-1 costs: Borg agentic runs
# averaged ~$4/run (0109v $2.2-3.3, 0045v up to $8.7 when v2 explores before
# giving up) — the original $2.5 was low; Copiale set Borg-like pending data;
# synth kept ~$1 (Part B) with headroom. These feed the checkpoint estimate.
CASES: tuple[Case, ...] = (
    Case("borg_single_B_borg_0109v", "borg", 4.0, True, True,
         split_file="borg_tests.jsonl"),
    Case("borg_single_B_borg_0045v", "borg", 4.0, True, False,
         split_file="borg_tests.jsonl"),
    Case("copiale_single_B_copiale_p017", "copiale", 3.5, False, False,
         split_file="copiale_tests.jsonl"),
    Case("synth_en_250nb_s4", "synth", 1.5, False, True,
         preset="hard", seed=4),
    Case("synth_en_200honb_s6", "synth", 1.5, False, False,
         preset="hardest", seed=6),
)
CASE_BY_NAME: dict[str, Case] = {c.name: c for c in CASES}


# ---------------------------------------------------------------------------
# Cells + enumeration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    loop: str          # "v2" | "v3"
    case: str          # Case.name
    preflight: bool    # automated preflight ON/OFF
    replicate: int     # 1..3

    def key(self) -> tuple[str, str, bool, int]:
        """Resume key. Preflight is part of the key because the two
        preflight-OFF cases have BOTH an ON and an OFF cell at the same
        (loop, case, replicate)."""
        return (self.loop, self.case, self.preflight, self.replicate)

    def case_label(self) -> str:
        """Directory label folding preflight into ``<case>`` so the on/off pairs
        land in separate trees while keeping F9b's `<loop>/<case>/<replicate>`
        layout shape."""
        return self.case if self.preflight else f"{self.case}__noprefl"

    def artifact_subdir(self) -> str:
        return f"{self.loop}/{self.case_label()}/{self.replicate}"


def enumerate_cells(
    cases: Iterable[Case] = CASES,
    loops: Iterable[str] = LOOPS,
    replicates: Iterable[int] = REPLICATES,
) -> list[Cell]:
    """All 42 cells: the 30 preflight-ON runs first, then the 12 preflight-OFF."""
    cases = list(cases)
    loops = list(loops)
    replicates = list(replicates)
    cells: list[Cell] = []
    for case in cases:
        for loop in loops:
            for rep in replicates:
                cells.append(Cell(loop, case.name, True, rep))
    for case in cases:
        if not case.preflight_off:
            continue
        for loop in loops:
            for rep in replicates:
                cells.append(Cell(loop, case.name, False, rep))
    return cells


def select_cells(
    *,
    case_names: Iterable[str] | None = None,
    loops: Iterable[str] | None = None,
    replicates: int = len(REPLICATES),
    preflight: str = "both",
) -> list[Cell]:
    """Build a safely filtered matrix for focused acceptance runs."""
    names = list(case_names or CASE_BY_NAME)
    unknown = [name for name in names if name not in CASE_BY_NAME]
    if unknown:
        raise ValueError(f"unknown case(s): {', '.join(unknown)}")
    selected_loops = list(loops or LOOPS)
    invalid_loops = [loop for loop in selected_loops if loop not in LOOPS]
    if invalid_loops:
        raise ValueError(f"unknown loop(s): {', '.join(invalid_loops)}")
    if preflight not in {"on", "off", "both"}:
        raise ValueError("preflight must be on, off, or both")
    cells = enumerate_cells(
        cases=[CASE_BY_NAME[name] for name in names],
        loops=selected_loops,
        replicates=range(1, replicates + 1),
    )
    if preflight == "on":
        cells = [cell for cell in cells if cell.preflight]
    elif preflight == "off":
        cells = [cell for cell in cells if not cell.preflight]
    return cells


def per_cell_estimate(cell_or_case: Cell | str) -> float:
    name = cell_or_case.case if isinstance(cell_or_case, Cell) else cell_or_case
    case = CASE_BY_NAME.get(name)
    return case.est_cost if case else 0.0


# ---------------------------------------------------------------------------
# Summary JSONL I/O + resume
# ---------------------------------------------------------------------------
def load_summary_rows(summary_path: str | Path) -> list[dict[str, Any]]:
    p = Path(summary_path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, bool, int]:
    return (
        str(row.get("loop")),
        str(row.get("case")),
        bool(row.get("preflight")),
        int(row.get("replicate")),
    )


def completed_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str, bool, int]]:
    return {row_key(r) for r in rows}


def pending_cells(
    all_cells: Iterable[Cell],
    done_keys: set[tuple[str, str, bool, int]],
) -> list[Cell]:
    return [c for c in all_cells if c.key() not in done_keys]


def append_summary_row(summary_path: str | Path, row: dict[str, Any]) -> None:
    p = Path(summary_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Halfway checkpoint (F9c)
# ---------------------------------------------------------------------------
def checkpoint_should_stop(
    completed_rows: list[dict[str, Any]],
    *,
    trigger_after: int = CHECKPOINT_AFTER,
    over_fraction: float = CHECKPOINT_OVER_FRACTION,
    estimate_fn: Callable[[str], float] = per_cell_estimate,
) -> tuple[bool, str]:
    """After ``trigger_after`` completed runs, compare cumulative ACTUAL cost to
    the cumulative per-completed-cell ESTIMATE. Returns (stop, message)."""
    if len(completed_rows) < trigger_after:
        return False, ""
    actual = sum(float(r.get("cost_usd") or 0.0) for r in completed_rows)
    estimate = sum(estimate_fn(str(r.get("case"))) for r in completed_rows)
    if estimate > 0 and actual > (1.0 + over_fraction) * estimate:
        pct = (actual / estimate - 1.0) * 100.0
        msg = (
            f"HALFWAY CHECKPOINT (F9c): after {len(completed_rows)} runs, "
            f"cumulative actual ${actual:.2f} exceeds cumulative estimate "
            f"${estimate:.2f} by {pct:.0f}% (>{int(over_fraction * 100)}%). "
            "Stopping launches; report and await go-ahead."
        )
        return True, msg
    return False, ""


# ---------------------------------------------------------------------------
# Summary-row construction from a completed run
# ---------------------------------------------------------------------------
def build_summary_row(
    cell: Cell,
    result: Any,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one summary JSONL row from a RunResultV2 + its saved artifact."""
    status = getattr(result, "status", artifact.get("status", "unknown"))
    declared = status in {"solved", "unsolved"}
    fallback = status in {"fallback_declared", "error"}
    row: dict[str, Any] = {
        "loop": cell.loop,
        "case": cell.case,
        "preflight": cell.preflight,
        "replicate": cell.replicate,
        "test_id": getattr(result, "test_id", artifact.get("cipher_id")),
        "run_id": artifact.get("run_id"),
        "status": status,
        "char_accuracy": float(getattr(result, "char_accuracy", 0.0) or 0.0),
        "word_accuracy": float(getattr(result, "word_accuracy", 0.0) or 0.0),
        "declared": declared,
        "fallback": fallback,
        "prompt_tokens": int(artifact.get("total_input_tokens") or 0),
        "completion_tokens": int(artifact.get("total_output_tokens") or 0),
        "cache_read_tokens": int(artifact.get("total_cache_read_tokens") or 0),
        "cost_usd": float(getattr(result, "estimated_cost_usd", 0.0) or 0.0),
        "iterations": int(getattr(result, "iterations_used", 0) or 0),
        "wall_clock_seconds": float(getattr(result, "elapsed_seconds", 0.0) or 0.0),
        "self_confidence": getattr(result, "self_confidence", None),
        "telemetry_version": 2,
        # v3 attestation fields (None for v2). ``attested`` remains the legacy
        # declaration-attached meaning; verify telemetry below is sourced from
        # the complete top-level attestation ledger.
        "attested": None,
        "declaration_attested": None,
        "verify_ran": None,
        "verify_count": None,
        "verify_branch": None,
        "verify_reader_accepts": None,
        "verify_coherence": None,
        "verify_created_turn": None,
        "positive_attestation_available": None,
        "attested_fallback": None,
        "late_turn_hint_fired": False,
    }
    if cell.loop == "v3":
        _apply_v3_telemetry(row, artifact)
    return row


def _attestation_sort_key(attestation: dict[str, Any]) -> tuple[int, str]:
    return (
        int(attestation.get("created_turn") or 0),
        str(attestation.get("episode_id") or ""),
    )


def _is_positive_attestation(attestation: dict[str, Any]) -> bool:
    return bool(attestation.get("reader_accepts")) and int(
        attestation.get("coherence") or 0
    ) >= 7


def _apply_v3_telemetry(row: dict[str, Any], artifact: dict[str, Any]) -> None:
    """Populate v3 verification fields without changing outcome metrics."""
    solution = artifact.get("solution") or {}
    solution = solution if isinstance(solution, dict) else {}
    attached = solution.get("attestation")
    declaration_attested = isinstance(attached, dict)
    attestations = [
        a for a in (artifact.get("attestations") or []) if isinstance(a, dict)
    ]
    selected_branch = str(solution.get("branch") or "")
    selected_attestations = [
        a for a in attestations if str(a.get("branch") or "") == selected_branch
    ]
    latest = max(selected_attestations or attestations, key=_attestation_sort_key, default=None)

    row["attested"] = declaration_attested
    row["declaration_attested"] = declaration_attested
    row["verify_ran"] = bool(attestations) or declaration_attested
    row["verify_count"] = len(attestations) or (1 if declaration_attested else 0)
    row["positive_attestation_available"] = any(
        _is_positive_attestation(a) for a in attestations
    ) or (declaration_attested and _is_positive_attestation(attached))
    row["attested_fallback"] = bool(artifact.get("attested_fallback"))
    if latest is not None:
        row["verify_branch"] = latest.get("branch")
        row["verify_reader_accepts"] = latest.get("reader_accepts")
        row["verify_coherence"] = latest.get("coherence")
        row["verify_created_turn"] = latest.get("created_turn")
    elif declaration_attested:
        # Older artifacts may carry only the declaration-attached attestation.
        row["verify_branch"] = attached.get("branch")
        row["verify_reader_accepts"] = attached.get("reader_accepts")
        row["verify_coherence"] = attached.get("coherence")
        row["verify_created_turn"] = attached.get("created_turn")
    events = artifact.get("loop_events") or []
    row["late_turn_hint_fired"] = any(
        isinstance(e, dict) and e.get("event") == "late_turn_attestation_hint"
        for e in events
    )


def refresh_summary_rows_from_artifacts(
    rows: list[dict[str, Any]], artifact_root: str | Path
) -> list[dict[str, Any]]:
    """Refresh additive v3 telemetry from saved artifacts only (no API use)."""
    root = Path(artifact_root)
    refreshed: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        row["telemetry_version"] = 2
        for field, default in {
            "declaration_attested": None,
            "verify_ran": None,
            "verify_count": None,
            "verify_branch": None,
            "positive_attestation_available": None,
            "attested_fallback": None,
        }.items():
            row.setdefault(field, default)
        if row.get("loop") != "v3" or not row.get("run_id"):
            refreshed.append(row)
            continue
        matches = sorted(root.glob(f"**/{row['run_id']}.json"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one artifact for run_id={row['run_id']!r}; "
                f"found {len(matches)} under {root}"
            )
        artifact = json.loads(matches[0].read_text(encoding="utf-8"))
        _apply_v3_telemetry(row, artifact)
        refreshed.append(row)
    return refreshed


def write_summary_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Aggregation (Part C)
# ---------------------------------------------------------------------------
def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _min(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return min(xs) if xs else None


def _max(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def _per_case_means(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, float | None]]:
    """{case: {loop: mean-over-replicates}} for one numeric field."""
    buckets: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        buckets.setdefault((r["case"], r["loop"]), []).append(
            float(r.get(field) or 0.0)
        )
    out: dict[str, dict[str, float | None]] = {}
    for (case, loop), vals in buckets.items():
        out.setdefault(case, {})[loop] = _mean(vals)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce the Part-C report: per-cell stats + the decision aggregate.

    Decision aggregate (F4): the 30 preflight-ON runs ONLY, case-level equal
    weighting (mean of per-case means). The word-accuracy clause aggregates ONLY
    over cases where word accuracy is meaningful (the two Borg pages).
    """
    per_cell = _aggregate_per_cell(rows)
    decision = _aggregate_decision(rows)
    return {"per_cell": per_cell, "decision": decision, "n_rows": len(rows)}


def _aggregate_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, bool], list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault((r["loop"], r["case"], bool(r["preflight"])), []).append(r)
    out: list[dict[str, Any]] = []
    for (loop, case, preflight), group in sorted(buckets.items()):
        char = [float(r.get("char_accuracy") or 0.0) for r in group]
        word = [float(r.get("word_accuracy") or 0.0) for r in group]
        cost = [float(r.get("cost_usd") or 0.0) for r in group]
        cell = {
            "loop": loop,
            "case": case,
            "preflight": preflight,
            "n": len(group),
            "char_mean": _mean(char), "char_min": _min(char), "char_max": _max(char),
            "word_mean": _mean(word), "word_min": _min(word), "word_max": _max(word),
            "declared_n": sum(1 for r in group if r.get("declared")),
            "fallback_n": sum(1 for r in group if r.get("fallback")),
            "cost_mean": _mean(cost), "cost_min": _min(cost), "cost_max": _max(cost),
            "prompt_tokens_mean": _mean(
                [float(r.get("prompt_tokens") or 0) for r in group]),
            "completion_tokens_mean": _mean(
                [float(r.get("completion_tokens") or 0) for r in group]),
            "cache_read_mean": _mean(
                [float(r.get("cache_read_tokens") or 0) for r in group]),
            "iterations_mean": _mean(
                [float(r.get("iterations") or 0) for r in group]),
            "wall_mean": _mean(
                [float(r.get("wall_clock_seconds") or 0.0) for r in group]),
        }
        if loop == "v3":
            cell["attested_n"] = sum(1 for r in group if r.get("attested"))
            cell["reader_accepts_n"] = sum(
                1 for r in group if r.get("verify_reader_accepts"))
            cell["coherence_mean"] = _mean(
                [r.get("verify_coherence") for r in group
                 if r.get("verify_coherence") is not None])
            cell["verify_turn_mean"] = _mean(
                [r.get("verify_created_turn") for r in group
                 if r.get("verify_created_turn") is not None])
            cell["late_turn_hint_n"] = sum(
                1 for r in group if r.get("late_turn_hint_fired"))
        out.append(cell)
    return out


def _aggregate_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decision_rows = [r for r in rows if bool(r.get("preflight"))]
    char_means = _per_case_means(decision_rows, "char_accuracy")
    word_means = _per_case_means(decision_rows, "word_accuracy")
    cost_means = _per_case_means(decision_rows, "cost_usd")

    word_cases = {c.name for c in CASES if c.word_meaningful}

    def loop_agg(loop: str) -> dict[str, Any]:
        char_case_means = [
            m[loop] for m in char_means.values() if m.get(loop) is not None
        ]
        word_case_means = [
            m[loop] for case, m in word_means.items()
            if case in word_cases and m.get(loop) is not None
        ]
        cost_case_means = [
            m[loop] for m in cost_means.values() if m.get(loop) is not None
        ]
        per_run_cost = [
            float(r.get("cost_usd") or 0.0)
            for r in decision_rows if r["loop"] == loop
        ]
        return {
            "char_case_weighted": _mean(char_case_means),
            "word_case_weighted": _mean(word_case_means),
            "cost_case_weighted": _mean(cost_case_means),
            "cost_per_run": _mean(per_run_cost),
            "n_runs": len(per_run_cost),
        }

    v2 = loop_agg("v2")
    v3 = loop_agg("v3")
    complete, completeness = _matrix_completeness(decision_rows)
    verdict = _decision_verdict(v2, v3, complete=complete,
                                completeness=completeness)
    return {
        "n_preflight_on_rows": len(decision_rows),
        "per_loop": {"v2": v2, "v3": v3},
        "complete": complete,
        "completeness": completeness,
        "verdict": verdict,
    }


def _matrix_completeness(
    decision_rows: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Reviewer fix 2: the decision aggregate averages whatever cases each loop
    HAS, so an asymmetric/incomplete matrix (exactly the post-checkpoint-stop
    state) can flip the verdict — e.g. all copiale v3 rows missing drops v3's
    hardest case from its mean. The verdict may only fire when BOTH loops have
    all 15 preflight-ON runs over IDENTICAL case sets (5 cases x 3 replicates,
    checked as exact (case, replicate) coverage per loop)."""
    expected = {(c.name, rep) for c in CASES for rep in REPLICATES}
    per_loop: dict[str, Any] = {}
    for loop in LOOPS:
        got = {
            (str(r.get("case")), int(r.get("replicate") or 0))
            for r in decision_rows
            if r.get("loop") == loop
        }
        missing = sorted(expected - got)
        per_loop[loop] = {
            "n_expected": len(expected),
            "n_present": len(got & expected),
            "missing": [f"{case}#r{rep}" for case, rep in missing],
        }
    complete = all(not v["missing"] for v in per_loop.values())
    return complete, per_loop


def _decision_verdict(
    v2: dict[str, Any],
    v3: dict[str, Any],
    *,
    complete: bool = True,
    completeness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Design decision rule: switch to v3 iff (1) v3 mean accuracy >= v2 (char AND
    word) and (2) v3 cost materially lower (>=20% lower mean cost per run at equal
    or better accuracy). NEVER switches on an incomplete/asymmetric matrix."""
    reasons: list[str] = []

    def ge(a: float | None, b: float | None) -> bool | None:
        if a is None or b is None:
            return None
        return a >= b - 1e-9

    char_ok = ge(v3["char_case_weighted"], v2["char_case_weighted"])
    word_ok = ge(v3["word_case_weighted"], v2["word_case_weighted"])
    accuracy_ok = bool(char_ok) and bool(word_ok)

    # Reviewer fix 3: a 0.0 mean cost is a cost-extraction bug, not a free loop —
    # the cost clause needs a POSITIVE mean on BOTH sides to be evaluable.
    cost_ok: bool | None = None
    v2_cost = v2.get("cost_per_run")
    v3_cost = v3.get("cost_per_run")
    if (
        isinstance(v2_cost, (int, float)) and v2_cost > 0
        and isinstance(v3_cost, (int, float)) and v3_cost > 0
    ):
        cost_ok = v3_cost <= 0.80 * v2_cost

    switch = complete and bool(accuracy_ok) and bool(cost_ok)
    if not complete:
        detail = ""
        if completeness:
            missing_bits = [
                f"{loop}: missing {len(info['missing'])} of {info['n_expected']}"
                for loop, info in completeness.items()
                if info["missing"]
            ]
            detail = f" ({'; '.join(missing_bits)})"
        reasons.append(
            "incomplete matrix: both loops need all 15 preflight-ON runs over "
            f"identical case sets before the verdict can fire{detail}"
        )
    if char_ok is None or word_ok is None:
        reasons.append("insufficient data to evaluate accuracy clause")
    else:
        reasons.append(
            f"accuracy: v3 char {'>=' if char_ok else '<'} v2, "
            f"v3 word {'>=' if word_ok else '<'} v2"
        )
    if cost_ok is None:
        reasons.append(
            "insufficient data to evaluate cost clause "
            "(need a positive mean cost/run on both loops)"
        )
    else:
        reasons.append(
            f"cost: v3 mean/run {'<= 0.8x' if cost_ok else '> 0.8x'} v2 "
            "(>=20% lower required)"
        )
    if complete and not switch and accuracy_ok and cost_ok is False:
        reasons.append("v3 wins accuracy but not cost -> DO NOT switch; escalate")
    return {
        "switch": switch,
        "complete": complete,
        "accuracy_ok": accuracy_ok,
        "cost_ok": cost_ok,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def format_matrix_table(cells: list[Cell]) -> str:
    lines = ["loop  preflight  replicate  case                              est($)"]
    lines.append("-" * 72)
    total = 0.0
    for c in cells:
        est = per_cell_estimate(c)
        total += est
        pf = "ON " if c.preflight else "OFF"
        lines.append(f"{c.loop:<5} {pf:<9}  {c.replicate:<9}  {c.case:<32}  {est:>5.2f}")
    lines.append("-" * 72)
    lines.append(
        f"{len(cells)} runs; total estimate ${total:.2f} "
        f"(halfway checkpoint after run {CHECKPOINT_AFTER})"
    )
    return "\n".join(lines)


def _fmt(x: float | None, spec: str = ".3f") -> str:
    return format(x, spec) if isinstance(x, (int, float)) else "-"


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"=== M6 bake-off report ({report['n_rows']} summary rows) ===\n")
    lines.append("Per-cell (loop/case/preflight):")
    header = (
        "loop case                             pf  n  char(mean/min/max)      "
        "word(mean)  decl/fb  cost   iters  att/rdr"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for c in report["per_cell"]:
        att = ""
        if c["loop"] == "v3":
            att = f"{c.get('attested_n', 0)}/{c.get('reader_accepts_n', 0)}"
        lines.append(
            f"{c['loop']:<4} {c['case']:<32} "
            f"{'ON' if c['preflight'] else 'off':<3} {c['n']:<2} "
            f"{_fmt(c['char_mean'])}/{_fmt(c['char_min'])}/{_fmt(c['char_max'])}  "
            f"{_fmt(c['word_mean']):<9}  {c['declared_n']}/{c['fallback_n']:<4}  "
            f"{_fmt(c['cost_mean'], '.2f'):<5}  {_fmt(c['iterations_mean'], '.1f'):<5}  {att}"
        )
    d = report["decision"]
    lines.append("")
    lines.append(
        f"Decision aggregate ({d['n_preflight_on_rows']} preflight-ON runs, "
        "case-level equal weighting):"
    )
    for loop in ("v2", "v3"):
        pl = d["per_loop"][loop]
        lines.append(
            f"  {loop}: char={_fmt(pl['char_case_weighted'])}  "
            f"word[borg]={_fmt(pl['word_case_weighted'])}  "
            f"cost/run=${_fmt(pl['cost_per_run'], '.2f')}  "
            f"cost[case-wt]=${_fmt(pl['cost_case_weighted'], '.2f')}  "
            f"(n={pl['n_runs']})"
        )
    v = d["verdict"]
    lines.append(f"  VERDICT: switch_to_v3={v['switch']}")
    for reason in v["reasons"]:
        lines.append(f"    - {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live data loading (lazy imports; exercised by the orchestrator, not the tests)
# ---------------------------------------------------------------------------
def _load_test_data_for_case(case: Case, benchmark_root: str, cache_dir: str) -> Any:
    """Return a ``TestData`` for a case. Synth cases ASSERT a warm cache hit
    (F3) — a miss raises (never calls a generator)."""
    if case.kind == "synth":
        from testgen.builder import build_test_case
        from testgen.cache import PlaintextCache
        from testgen.spec import DifficultyPreset, TestSpec

        spec = TestSpec.from_preset(
            DifficultyPreset(case.preset), language="en", seed=case.seed
        )
        cache = PlaintextCache(cache_dir)
        if cache.get(spec) is None:
            raise _CacheMiss(
                f"testgen cache MISS for {case.name} (preset={case.preset}, "
                f"seed={case.seed}) in {cache_dir!r}. Warm the cache first — this "
                "runner never calls a plaintext generator."
            )
        # api_key="" is safe: the asserted cache hit means no generator call.
        return build_test_case(spec, cache, api_key="")

    from benchmark.loader import BenchmarkLoader

    loader = BenchmarkLoader(benchmark_root)
    tests = loader.load_tests(case.split_file)
    match = [t for t in tests if t.test_id == case.name]
    if not match:
        raise ValueError(
            f"benchmark test {case.name!r} not found in split {case.split_file!r}"
        )
    return loader.load_test_data(match[0])


class _CacheMiss(RuntimeError):
    pass


def _build_model_api(model: str) -> Any:
    """Build the provider-neutral model API for ``model`` (gpt-5.5 -> OpenAI)."""
    from agent.model_provider import infer_provider_from_model, make_model_provider

    # Reuse the CLI's key resolution so provider/key wiring stays identical.
    from cli import get_api_key  # noqa: E402

    provider = infer_provider_from_model(model, None)
    return make_model_provider(
        provider=provider, api_key=get_api_key(provider), model=model
    )


def probe_model_funding(api: Any) -> str | None:
    """Cheap 1-token send to confirm the account is funded/reachable BEFORE the
    matrix spends real money. Returns None on success, else a short reason.
    Motivated by the 2026-07-14 run where an unfunded OpenAI account
    (insufficient_quota) failed all 39 remaining cells on turn 1 — a probe
    here fails at run 0 instead. Any provider error (quota, auth, network)
    counts as a failed probe."""
    from agent.model_provider import ModelProviderError

    try:
        api.send(
            messages=[{"role": "user", "content": "Reply with the word OK."}],
            system="",
            max_tokens=5,
        )
    except ModelProviderError as exc:
        return str(exc)[:300]
    except Exception as exc:  # noqa: BLE001 - a probe failure must not be fatal-by-crash
        return f"{type(exc).__name__}: {str(exc)[:280]}"
    return None


def validate_all_cases(
    benchmark_root: str,
    cache_dir: str,
    cases: Iterable[Case] = CASES,
) -> list[str]:
    """Reviewer fix 4a: validate EVERY case's data source up front (zero LLM
    cost) so a testgen cache miss or benchmark lookup failure surfaces BEFORE
    run 1, not at run 19 after real spend. Returns one error string per broken
    case (empty list = all good)."""
    errors: list[str] = []
    for case in cases:
        try:
            _load_test_data_for_case(case, benchmark_root, cache_dir)
        except Exception as exc:  # noqa: BLE001 - collect ALL problems, then exit
            errors.append(f"{case.name}: {exc}")
    return errors


def process_completed_run(
    cell: Cell,
    result: Any,
    artifact: dict[str, Any],
    summary_path: str | Path,
    completed_rows: list[dict[str, Any]],
) -> int | None:
    """Record one finished run; return an exit code to stop the matrix, or None.

    Reviewer fix 1: both loops swallow KeyboardInterrupt during a model call and
    return a status="stopped" partial artifact. Recording a summary row for it
    would mark the cell complete FOREVER (--resume skips it) — so a stopped run
    records NOTHING and exits with the conventional SIGINT code so the operator
    can rerun with --resume to retry the cell.
    """
    row = build_summary_row(cell, result, artifact)
    if row["status"] == "stopped":
        print(
            f"[interrupted] loop={cell.loop} case={cell.case} "
            f"preflight={'ON' if cell.preflight else 'OFF'} rep={cell.replicate} "
            "returned status=stopped (interrupted mid-run). NOT recording a "
            "summary row for this cell; rerun with --resume to retry it.",
            file=sys.stderr,
        )
        return EXIT_INTERRUPTED
    append_summary_row(summary_path, row)   # incremental, after EVERY run
    completed_rows.append(row)
    print(
        f"    -> {row['status']}  char={row['char_accuracy']:.1%}  "
        f"cost=${row['cost_usd']:.2f}  (total rows={len(completed_rows)})"
    )
    stop, msg = checkpoint_should_stop(completed_rows)
    if stop:
        print(msg, file=sys.stderr)
        return EXIT_CHECKPOINT_STOP
    return None


def _run_matrix(args: argparse.Namespace) -> int:
    summary_path = Path(args.summary)

    # Reviewer fix 5: an existing summary without --resume would double-count
    # rows (and blind the checkpoint math) — refuse and hint.
    if summary_path.exists() and not args.resume:
        print(
            f"ERROR: summary file {summary_path} already exists. Pass --resume "
            "to skip its completed cells and continue, or move the file aside "
            "for a fresh matrix.",
            file=sys.stderr,
        )
        return EXIT_SUMMARY_EXISTS

    # Reviewer fix 4a: validate all data sources BEFORE any run / any spend.
    try:
        all_cells = select_cells(
            case_names=args.cases,
            loops=args.loops,
            replicates=args.replicates,
            preflight=args.preflight,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CACHE_MISS
    selected_cases = [CASE_BY_NAME[name] for name in dict.fromkeys(
        cell.case for cell in all_cells
    )]
    problems = validate_all_cases(
        args.benchmark_root, args.cache_dir, cases=selected_cases
    )
    if problems:
        print(
            "ERROR: upfront data-source validation failed; no runs launched:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_CACHE_MISS

    artifact_root = Path(args.artifact_root)

    done_keys: set[tuple[str, str, bool, int]] = set()
    completed_rows: list[dict[str, Any]] = []
    if args.resume:
        existing = load_summary_rows(summary_path)
        completed_rows = list(existing)
        done_keys = completed_keys(existing)
        print(f"[resume] {len(done_keys)} cells already complete; skipping them.")

    todo = pending_cells(all_cells, done_keys)
    print(f"[matrix] {len(todo)} cell(s) to run (of {len(all_cells)}), "
          f"{args.replicates} replicate(s)/cell.")

    api = _build_model_api(args.model)

    # Funding pre-check: one cheap send before any matrix spend. Skippable via
    # --skip-funding-probe (tests / offline). Fails at "run 0" if the account
    # is unfunded/unreachable, instead of contaminating N cells on turn 1.
    if not getattr(args, "skip_funding_probe", False):
        reason = probe_model_funding(api)
        if reason is not None:
            print(
                f"ERROR: model funding/reachability probe failed; no runs "
                f"launched. Provider said: {reason}",
                file=sys.stderr,
            )
            return EXIT_API_UNFUNDED

    for idx, cell in enumerate(todo, 1):
        spent = sum(float(row.get("cost_usd") or 0.0) for row in completed_rows)
        if (
            args.max_total_cost is not None
            and spent + args.budget_reserve_per_run > args.max_total_cost
        ):
            print(
                f"[budget-stop] spent=${spent:.2f}; launching another run with "
                f"the ${args.budget_reserve_per_run:.2f} reserve would exceed "
                f"the ${args.max_total_cost:.2f} ceiling. Resume later with a "
                "higher ceiling if authorized.",
                file=sys.stderr,
            )
            return EXIT_BUDGET_STOP
        case = CASE_BY_NAME[cell.case]
        print(
            f"[{idx}/{len(todo)}] loop={cell.loop} case={cell.case} "
            f"preflight={'ON' if cell.preflight else 'OFF'} rep={cell.replicate}"
        )
        # Reviewer fix 4b: an unexpected exception in one cell must not kill the
        # matrix — print, record NOTHING for that cell (so --resume retries it),
        # and continue. A cache miss stays a hard error (validation makes it
        # near-impossible mid-run; if it happens, the cache was mutated under
        # us). KeyboardInterrupt is a BaseException and still propagates.
        try:
            test_data = _load_test_data_for_case(
                case, args.benchmark_root, args.cache_dir
            )

            from benchmark.runner_v2 import BenchmarkRunnerV2

            runner = BenchmarkRunnerV2(
                claude_api=api,
                max_iterations=args.max_iterations,
                verbose=args.verbose,
                artifact_dir=artifact_root / cell.artifact_subdir(),
                automated_preflight=cell.preflight,
                agent_loop=cell.loop,
            )
            language = "en" if case.kind == "synth" else None
            result = runner.run_test(test_data, language=language)

            artifact: dict[str, Any] = {}
            if result.artifact_path and Path(result.artifact_path).exists():
                artifact = json.loads(
                    Path(result.artifact_path).read_text(encoding="utf-8")
                )
        except _CacheMiss as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_CACHE_MISS
        except Exception as exc:  # noqa: BLE001
            print(
                f"[error] cell loop={cell.loop} case={cell.case} "
                f"preflight={'ON' if cell.preflight else 'OFF'} "
                f"rep={cell.replicate} failed unexpectedly: {exc!r}. Recording "
                "NOTHING for this cell (--resume will retry it); continuing.",
                file=sys.stderr,
            )
            continue

        code = process_completed_run(
            cell, result, artifact, summary_path, completed_rows
        )
        if code is not None:
            return code

    print("[matrix] complete.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Agent-loop v3 bake-off matrix runner.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    p.add_argument("--benchmark-root", default=DEFAULT_BENCHMARK_ROOT)
    p.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    p.add_argument("--summary", default=str(Path(DEFAULT_ARTIFACT_ROOT) / "summary.jsonl"))
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--resume", action="store_true",
                   help="Skip completed (loop,case,preflight,replicate) cells.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the matrix + per-cell cost estimate and exit.")
    p.add_argument("--report", action="store_true",
                   help="Re-aggregate the summary JSONL into the Part-C table.")
    p.add_argument("--refresh-summary", action="store_true",
                   help="Refresh v3 verification telemetry from artifacts only; "
                        "does not initialize a model provider or run cases.")
    p.add_argument("--refresh-output",
                   help="Output JSONL for --refresh-summary (required unless "
                        "--refresh-overwrite is used).")
    p.add_argument("--refresh-overwrite", action="store_true",
                   help="Allow --refresh-summary to replace --summary in place.")
    p.add_argument("--replicates", type=int, default=len(REPLICATES),
                   help="Replicates per cell (default 3). Lower for a "
                        "breadth-first pass that covers every case×loop×"
                        "preflight before adding replicates; combine with "
                        "--resume to fill in only the uncovered cells.")
    p.add_argument("--case", dest="cases", action="append",
                   choices=sorted(CASE_BY_NAME),
                   help="Run only this case (repeatable). Default: all cases.")
    p.add_argument("--loop", dest="loops", action="append", choices=LOOPS,
                   help="Run only this loop (repeatable). Default: v2 and v3.")
    p.add_argument("--preflight", choices=("on", "off", "both"), default="both",
                   help="Select automated-preflight cells. Default: both.")
    p.add_argument("--max-total-cost", type=float,
                   help="Stop before another run when spend plus the per-run "
                        "reserve would exceed this dollar ceiling.")
    p.add_argument("--budget-reserve-per-run", type=float, default=0.0,
                   help="Conservative next-run reserve used with "
                        "--max-total-cost (default 0).")
    p.add_argument("--skip-funding-probe", action="store_true",
                   help="Skip the pre-run 1-token funding/reachability probe.")
    p.add_argument("--verbose", action="store_true")
    return p


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Reviewer fix 6: expand ~ in every path argument."""
    args.benchmark_root = str(Path(args.benchmark_root).expanduser())
    args.artifact_root = str(Path(args.artifact_root).expanduser())
    args.summary = str(Path(args.summary).expanduser())
    args.cache_dir = str(Path(args.cache_dir).expanduser())
    if args.refresh_output:
        args.refresh_output = str(Path(args.refresh_output).expanduser())
    return args


def main(argv: list[str] | None = None) -> int:
    args = _normalize_args(build_parser().parse_args(argv))

    if args.dry_run:
        print(format_matrix_table(select_cells(
            case_names=args.cases, loops=args.loops,
            replicates=args.replicates, preflight=args.preflight,
        )))
        return EXIT_OK
    if args.report:
        rows = load_summary_rows(args.summary)
        print(format_report(aggregate(rows)))
        return EXIT_OK
    if args.refresh_summary:
        if args.refresh_overwrite:
            output = args.summary
        elif args.refresh_output:
            output = args.refresh_output
        else:
            raise SystemExit(
                "--refresh-summary requires --refresh-output or --refresh-overwrite"
            )
        rows = load_summary_rows(args.summary)
        refreshed = refresh_summary_rows_from_artifacts(rows, args.artifact_root)
        write_summary_rows(output, refreshed)
        print(f"Wrote {output} ({len(refreshed)} rows; no model calls).")
        return EXIT_OK
    return _run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
