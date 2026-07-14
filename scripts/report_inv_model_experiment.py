#!/usr/bin/env python3
"""Scorer + report for the INV model-diagnosis experiment.

Deterministic aggregation from artifacts. Produces:

  * the FREE automated (non-LLM) diagnosis baseline over the suite — the floor
    the models should beat (run-policy step 1); it maps the fingerprint
    argmax-suspicion into the canonical family enum, and literally cannot name
    the composite (scored as a miss, expected);
  * a per-model table: diagnosis accuracy overall + per family, mean efficiency,
    pareidolia rate, and cost;
  * the EXCLUDED gate-fired bucket (Anthropic runs served by a different model),
    reported separately as a contaminated bucket, never mixed into the table;
  * the canonical family enum mappings, for transparency.

Ground truth (true_family) comes only from the suite manifest — never from a
model prompt.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/report_inv_model_experiment.py \
        [--artifact-root artifacts/inv_model_experiment] \
        [--suite frontier/inv_diagnosis_suite.jsonl] \
        [--cache-dir testgen_cache] [--no-baseline]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.inv_diagnosis import (  # noqa: E402
    SUITE_ROWS,
    SUSPICION_KEY_TO_FAMILY,
    TRUTH_LABEL_TO_FAMILY,
    DiagnosisFamily,
    is_diagnosis_correct,
    is_pareidolia,
    map_argmax_suspicion,
)

DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "inv_model_experiment"
DEFAULT_SUITE = REPO_ROOT / "frontier" / "inv_diagnosis_suite.jsonl"
DEFAULT_CACHE_DIR = REPO_ROOT / "testgen_cache"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_true_families(suite_path: Path) -> dict[str, str]:
    """case_id -> true_family (canonical enum value) from the suite manifest."""
    out: dict[str, str] = {}
    if not suite_path.exists():
        # Fall back to the in-code suite definition.
        return {r.case_id: r.true_family for r in SUITE_ROWS}
    for line in suite_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["case_id"]] = row["true_family"]
    return out


def load_artifacts(artifact_root: Path) -> list[dict[str, Any]]:
    """Load every diagnosis artifact JSON under artifact_root/<model>/*.json."""
    artifacts: list[dict[str, Any]] = []
    if not artifact_root.exists():
        return artifacts
    for path in sorted(artifact_root.glob("*/*.json")):
        try:
            artifacts.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return artifacts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_artifact(artifact: dict[str, Any], true_family: str) -> dict[str, Any]:
    """Score one diagnosis artifact against ground truth (post-hoc only)."""
    diagnosis = artifact.get("diagnosis") or {}
    declared_family = diagnosis.get("family")
    confidence = diagnosis.get("confidence")
    correct = is_diagnosis_correct(true_family, declared_family)
    pareidolia = is_pareidolia(true_family, declared_family, confidence)
    declared_iteration = artifact.get("declared_iteration")
    turns_to_first_correct = (
        declared_iteration if (correct and declared_iteration is not None) else math.inf
    )
    return {
        "case_id": artifact.get("case_id"),
        "model": artifact.get("model"),
        "true_family": true_family,
        "declared_family": declared_family,
        "confidence": confidence,
        "diagnosis_correct": correct,
        "pareidolia_flag": pareidolia,
        "turns_to_first_correct": turns_to_first_correct,
        "tool_call_count": artifact.get("tool_call_count", 0),
        "final_char_accuracy": artifact.get("final_char_accuracy"),
        "estimated_cost_usd": artifact.get("estimated_cost_usd", 0.0),
        "cost_priced": artifact.get("cost_priced", True),
        "total_input_tokens": artifact.get("total_input_tokens", 0),
        "total_output_tokens": artifact.get("total_output_tokens", 0),
        "safety_gate_fired": bool(artifact.get("safety_gate_fired", False)),
        "served_models": artifact.get("served_models", []),
        "status": artifact.get("status"),
    }


def aggregate(
    artifacts: list[dict[str, Any]],
    true_families: dict[str, str],
) -> dict[str, Any]:
    """Aggregate scored artifacts into per-model rows + the excluded bucket."""
    included: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        case_id = artifact.get("case_id")
        if case_id not in true_families:
            continue
        scored = score_artifact(artifact, true_families[case_id])
        model = str(scored["model"])
        if scored["safety_gate_fired"]:
            excluded[model].append(scored)
        else:
            included[model].append(scored)

    per_model = {model: _summarize_model(rows) for model, rows in included.items()}
    excluded_summary = {
        model: {
            "n": len(rows),
            "served_models": sorted({s for r in rows for s in r["served_models"]}),
            "cases": [r["case_id"] for r in rows],
        }
        for model, rows in excluded.items()
    }
    return {"per_model": per_model, "excluded": excluded_summary}


def _summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = [r for r in rows if r["diagnosis_correct"]]
    per_family: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        fam = r["true_family"]
        per_family[fam]["n"] += 1
        per_family[fam]["correct"] += int(r["diagnosis_correct"])
    correct_turns = [
        r["turns_to_first_correct"]
        for r in correct
        if r["turns_to_first_correct"] != math.inf
    ]
    correct_calls = [r["tool_call_count"] for r in correct]
    priced_costs = [r["estimated_cost_usd"] for r in rows if r["cost_priced"]]
    any_unpriced = any(not r["cost_priced"] for r in rows)
    return {
        "n": n,
        "diagnosis_accuracy": (len(correct) / n) if n else 0.0,
        "pareidolia_rate": (sum(r["pareidolia_flag"] for r in rows) / n) if n else 0.0,
        "mean_turns_to_correct": (sum(correct_turns) / len(correct_turns)) if correct_turns else None,
        "mean_tool_calls_to_correct": (sum(correct_calls) / len(correct_calls)) if correct_calls else None,
        "total_cost_usd": sum(priced_costs),
        "mean_cost_usd": (sum(priced_costs) / len(priced_costs)) if priced_costs else None,
        "any_unpriced": any_unpriced,
        "per_family": {fam: dict(v) for fam, v in sorted(per_family.items())},
        "cases": {
            r["case_id"]: {
                "true_family": r["true_family"],
                "declared_family": r["declared_family"],
                "correct": r["diagnosis_correct"],
                "pareidolia": r["pareidolia_flag"],
                "confidence": r["confidence"],
                "status": r["status"],
            }
            for r in sorted(rows, key=lambda x: str(x["case_id"]))
        },
    }


# ---------------------------------------------------------------------------
# Free automated (non-LLM) baseline
# ---------------------------------------------------------------------------


def automated_baseline(
    true_families: dict[str, str],
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Compute the free fingerprint-argmax diagnosis over the suite.

    Rebuilds each case offline (warm plaintext cache), runs
    ``compute_cipher_fingerprint``, maps argmax-suspicion into the canonical
    enum, and scores it. Cases whose plaintext is not cached are skipped with a
    note (baseline needs the warm cache that step 0 populates).
    """
    from analysis.cipher_id import compute_cipher_fingerprint
    from benchmark.loader import parse_canonical_transcription
    from experiments.inv_diagnosis import build_case
    from testgen.cache import PlaintextCache

    cache = PlaintextCache(cache_dir)
    per_case: list[dict[str, Any]] = []
    skipped: list[str] = []
    for row in SUITE_ROWS:
        if row.case_id not in true_families:
            continue
        try:
            test_data = build_case(row, cache, offline=True)
        except RuntimeError:
            skipped.append(row.case_id)
            continue
        ct = parse_canonical_transcription(test_data.canonical_transcription)
        fp = compute_cipher_fingerprint(
            ct.tokens,
            ct.alphabet.size,
            language="en",
            word_group_count=len(ct.words),
        )
        declared = map_argmax_suspicion(fp.suspicion_scores)
        true_family = true_families[row.case_id]
        correct = is_diagnosis_correct(true_family, declared)
        per_case.append({
            "case_id": row.case_id,
            "true_family": true_family,
            "declared_family": declared.value,
            "diagnosis_correct": correct,
            "suspicion_scores": {k: round(v, 3) for k, v in sorted(
                fp.suspicion_scores.items(), key=lambda x: -x[1]
            )},
        })

    per_family: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for c in per_case:
        per_family[c["true_family"]]["n"] += 1
        per_family[c["true_family"]]["correct"] += int(c["diagnosis_correct"])
    n = len(per_case)
    return {
        "n": n,
        "accuracy": (sum(c["diagnosis_correct"] for c in per_case) / n) if n else 0.0,
        "per_family": {fam: dict(v) for fam, v in sorted(per_family.items())},
        "per_case": per_case,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    return f"{x:.0%}" if isinstance(x, (int, float)) else "n/a"


def _fmt_num(x: float | None) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"


def format_report(
    aggregation: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("INV MODEL-DIAGNOSIS EXPERIMENT REPORT")
    lines.append("=" * 74)

    # Family enum mappings (transparency).
    lines.append("\nCanonical family enum + mappings")
    lines.append("-" * 74)
    lines.append("  truth-label (builder cipher_system) -> canonical family:")
    for k, v in TRUTH_LABEL_TO_FAMILY.items():
        lines.append(f"    {k:28s} -> {v.value}")
    lines.append("  suspicion-key (fingerprint) -> canonical family "
                 "(others -> unknown):")
    for k, v in SUSPICION_KEY_TO_FAMILY.items():
        lines.append(f"    {k:28s} -> {v.value}")
    lines.append("  composite credit: transposition_substitution OR unknown "
                 "counts as correct for the composite case.")

    # Free automated baseline.
    if baseline is not None:
        lines.append("\nFREE automated (non-LLM) diagnosis baseline")
        lines.append("-" * 74)
        lines.append(f"  overall accuracy: {_fmt_pct(baseline['accuracy'])} "
                     f"(n={baseline['n']})")
        lines.append(f"  {'family':30s} {'acc':>6s}  n")
        for fam, v in baseline["per_family"].items():
            acc = v["correct"] / v["n"] if v["n"] else 0.0
            lines.append(f"  {fam:30s} {_fmt_pct(acc):>6s}  {v['n']}")
        lines.append("  per case:")
        for c in baseline["per_case"]:
            mark = "OK " if c["diagnosis_correct"] else "MISS"
            lines.append(
                f"    [{mark}] {c['case_id']:18s} true={c['true_family']:28s} "
                f"argmax->{c['declared_family']}"
            )
        if baseline.get("skipped"):
            lines.append(f"  skipped (cold cache): {baseline['skipped']}")

    # Per-model table.
    per_model = aggregation.get("per_model", {})
    lines.append("\nPer-model diagnosis comparison (gate-fired runs excluded)")
    lines.append("-" * 74)
    if not per_model:
        lines.append("  (no model artifacts found)")
    else:
        header = (
            f"  {'model':22s} {'acc':>5s} {'pareid':>7s} "
            f"{'turns':>6s} {'calls':>6s} {'cost':>9s}  n"
        )
        lines.append(header)
        for model, s in sorted(per_model.items()):
            cost_s = "n/a" if s["any_unpriced"] else f"${s['total_cost_usd']:.4f}"
            lines.append(
                f"  {model:22s} {_fmt_pct(s['diagnosis_accuracy']):>5s} "
                f"{_fmt_pct(s['pareidolia_rate']):>7s} "
                f"{_fmt_num(s['mean_turns_to_correct']):>6s} "
                f"{_fmt_num(s['mean_tool_calls_to_correct']):>6s} "
                f"{cost_s:>9s}  {s['n']}"
            )
        # Per-family accuracy per model.
        lines.append("\n  Per-family accuracy:")
        for model, s in sorted(per_model.items()):
            lines.append(f"    {model}:")
            for fam, v in s["per_family"].items():
                acc = v["correct"] / v["n"] if v["n"] else 0.0
                lines.append(f"      {fam:30s} {_fmt_pct(acc):>6s}  (n={v['n']})")
        # Per-case detail.
        lines.append("\n  Per-case diagnoses:")
        for model, s in sorted(per_model.items()):
            lines.append(f"    {model}:")
            for case_id, c in s["cases"].items():
                mark = "OK " if c["correct"] else "MISS"
                flag = " [PAREIDOLIA]" if c["pareidolia"] else ""
                lines.append(
                    f"      [{mark}] {case_id:18s} true={c['true_family']:28s} "
                    f"decl={c['declared_family']} conf={c['confidence']}"
                    f"{flag}"
                )

    # Excluded gate-fired bucket.
    excluded = aggregation.get("excluded", {})
    lines.append("\nEXCLUDED — safety-gate-fired (contaminated) bucket")
    lines.append("-" * 74)
    if not excluded:
        lines.append("  (none — no run was served by a different model)")
    else:
        for model, info in sorted(excluded.items()):
            lines.append(
                f"  {model}: {info['n']} run(s) excluded; served={info['served_models']}; "
                f"cases={info['cases']}"
            )
    lines.append("=" * 74)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--json-out", default=None, help="Also write the raw aggregation JSON here.")
    args = parser.parse_args()

    true_families = load_true_families(Path(args.suite))
    artifacts = load_artifacts(Path(args.artifact_root))
    aggregation = aggregate(artifacts, true_families)
    baseline = None if args.no_baseline else automated_baseline(true_families, args.cache_dir)

    print(format_report(aggregation, baseline))

    if args.json_out:
        payload = {"aggregation": aggregation, "baseline": baseline}
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nRaw aggregation JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
