#!/usr/bin/env python3
"""Post-hoc R4 report. This module must never be imported by a solver/launcher."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from benchmark.scorer import score_decryption
from prepare_reliability_routing import file_digest, grade_completed
from reliability_routing_worker import digest
from run_reliability_routing import read_attempts, read_json, write_once


def saved_texts(value, path="artifact"):
    """Saved text evidence only; never claim this is the complete search menu."""
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}.{key}"
            if key in {"plaintext", "decryption", "decoded_text"} and isinstance(child, str) and child:
                yield here, child
            elif isinstance(child, (dict, list)):
                yield from saved_texts(child, here)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from saved_texts(child, f"{path}[{i}]")


def candidate_evidence(result, label):
    truth = label.get("plaintext_spaced") or label.get("plaintext")
    if result is None or not truth:
        return {"best_saved": None, "saved_distinct_texts": None}
    candidates = {}
    for path, text in saved_texts(result["artifact"]):
        sha = digest(text)
        if sha not in candidates:
            score = score_decryption(label["case_id"], text, truth, agent_score=0, status="post_hoc")
            candidates[sha] = {"path": path, "sha256": sha, "char_accuracy": score.char_accuracy,
                               "word_accuracy": score.word_accuracy}
    best = max(candidates.values(), key=lambda c: (c["char_accuracy"], c["word_accuracy"]), default=None)
    return {"best_saved": best, "saved_distinct_texts": len(candidates)}


def paired_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = row["case_id"], row["seed"]
        if row["arm"] in grouped[key]:
            raise ValueError("duplicate paired arm")
        grouped[key][row["arm"]] = row
    pairs = []
    for (case_id, seed), arms in sorted(grouped.items()):
        if set(arms) != {"blind_family", "family_supplied"}:
            raise ValueError("report must include missing-arm placeholders")
        blind, supplied = arms["blind_family"], arms["family_supplied"]
        a, b = blind["post_hoc_delivered"], supplied["post_hoc_delivered"]
        delta = b["char_accuracy"] - a["char_accuracy"] if a and b else None
        observations = []
        if not blind["expected_solve"]:
            observations.append("control_no_success_expectation")
        elif delta is None:
            observations.append("quality_comparison_unavailable")
        else:
            if delta >= 0.02:
                observations.append("family_supplied_material_gain")
            if delta <= -0.02:
                observations.append("blind_material_gain")
            if a["char_accuracy"] >= 0.99 and b["char_accuracy"] >= 0.99:
                observations.append("both_near_exact")
            if a["char_accuracy"] < 0.99 and b["char_accuracy"] < 0.99:
                observations.append("both_below_near_exact")
        if blind["role"] == "historical_diagnostic":
            observations.append("provisional_pending_R3")
        matched_context = (blind.get("execution_context") not in (None, "unrecorded")
                           and blind.get("execution_context") == supplied.get("execution_context"))
        if not matched_context:
            observations.append("execution_context_not_matched_or_unrecorded")
        pairs.append({
            "case_id": case_id, "family": blind["family"], "role": blind["role"], "seed": seed,
            "blind": blind, "family_supplied": supplied,
            "supplied_minus_blind_char": delta,
            "different_route_or_solver": (blind["routes"], blind["solver"]) != (supplied["routes"], supplied["solver"]),
            "matched_execution_context": matched_context,
            "observations": observations,
        })
    return pairs


def build_report(preparation, campaign, grading_path):
    plan_path = preparation / "control/plan.json"
    plan = read_json(plan_path)
    meta = read_json(campaign / "campaign.json")
    if meta["plan_sha256"] != file_digest(plan_path):
        raise ValueError("campaign/preparation hash mismatch")
    if plan["grading_packet_sha256"] != file_digest(grading_path):
        raise ValueError("grading packet changed; use the frozen labels, not silently updated references")
    labels = {r["case_id"]: r for r in read_json(grading_path)["cases"]}
    attempts = read_attempts(campaign, plan)
    context_path = campaign / "execution_context.json"
    contexts = read_json(context_path) if context_path.exists() else {"segments": []}
    rows = []
    for index, job in enumerate(plan["jobs"], start=1):
        attempt = attempts.get(job["job_id"])
        execution = attempt["execution"] if attempt else None
        if execution is None:
            execution = {"request_sha256": job["request_sha256"], "result": None,
                         "execution_status": "interrupted" if attempt else "unattempted"}
        result = execution.get("result")
        label = labels[job["request"]["case_id"]]
        row = grade_completed(job["request"], execution, label)
        artifact = (result or {}).get("artifact") or {}
        evidence = candidate_evidence(result, label)
        delivered = row["post_hoc_delivered"]
        best = evidence["best_saved"]
        row.update(evidence)
        row.update({
            "job_id": job["job_id"], "family": label["family"],
            "execution_context": next((s["context"] for s in contexts["segments"]
                                       if s["first_attempt"] <= index <= s["last_attempt"]), "unrecorded"),
            "request_sha256": job["request_sha256"],
            "result_file_sha256": file_digest(campaign / "results" / (job["job_id"] + ".json")) if attempt and attempt["execution"] else None,
            "solver": (result or {}).get("solver"),
            "routes": [s["route"] for s in (result or {}).get("routes_attempted", [])],
            "wall_seconds": execution.get("wall_seconds"), "cpu_usage": execution.get("cpu_usage"),
            "charged_seconds": attempt["charged_seconds"] if attempt else None,
            "seed_evidence": (result or {}).get("seed_evidence"),
            "delivery_matches_artifact": (result or {}).get("delivery_matches_artifact"),
            "delivered_sha256": (result or {}).get("delivered_sha256"),
            "solver_error": artifact.get("error"),
            "worker_stderr": execution.get("stderr", "")[-2000:],
            "saved_minus_delivered_char": best["char_accuracy"] - delivered["char_accuracy"] if best and delivered else None,
        })
        rows.append(row)
    pairs = paired_rows(rows)
    primary_synthetic = [p for p in pairs if p["seed"] == 61001 and p["role"] in {"development", "held_out"}]
    matched_complete = [p for p in primary_synthetic if p["matched_execution_context"]
                        and p["supplied_minus_blind_char"] is not None]
    return {
        "schema": "reliability-r4-report-v1", "r3_review": "pending",
        "campaign": str(campaign.relative_to(ROOT)) if campaign.is_relative_to(ROOT) else str(campaign),
        "plan_sha256": file_digest(plan_path), "campaign_metadata_sha256": file_digest(campaign / "campaign.json"),
        "grading_packet_sha256": plan["grading_packet_sha256"],
        "report_script_sha256": file_digest(Path(__file__)),
        "execution_context": contexts,
        "execution_context_sha256": file_digest(context_path) if context_path.exists() else None,
        "solver_revision": plan["provenance"]["solver_revision"],
        "attempted": len(attempts), "planned": len(plan["jobs"]),
        "execution_counts": dict(Counter(r["execution_status"] for r in rows)),
        "solver_status_counts": dict(Counter(r["solver_status"] for r in rows if r["solver_status"] is not None)),
        "known_worker_wall_seconds": sum(r["wall_seconds"] or 0 for r in rows),
        "wall_unknown_attempts": sum(r["wall_seconds"] is None and r["execution_status"] != "unattempted" for r in rows),
        "charged_seconds": sum(a["charged_seconds"] for a in attempts.values()),
        "primary_synthetic_pair_count": len(primary_synthetic),
        "primary_synthetic_observations": dict(Counter(x for p in primary_synthetic for x in p["observations"])),
        "primary_synthetic_matched_complete_pairs": len(matched_complete),
        "matched_complete_primary_observations": dict(Counter(x for p in matched_complete for x in p["observations"])),
        "primary_synthetic_arm_near_exact": {arm: sum(
            bool(p[arm]["post_hoc_delivered"] and p[arm]["post_hoc_delivered"]["char_accuracy"] >= 0.99)
            for p in primary_synthetic) for arm in ("blind", "family_supplied")},
        "pairs": pairs,
        "limitations": [
            "Primary synthetic cases, development repeats, historical diagnostics and controls are separate strata.",
            "Repeated request seeds are not automatically independent engine seeds.",
            "Missing/timeout output is unknown quality, never a zero-accuracy score.",
            "Saved-text maxima are post-hoc diagnostics, not exhaustive generated-menu quality or a selectable oracle.",
            "Historical scores use unreviewed frozen references; no source/repair conclusions until R3 review.",
            "No provider verification or live-agent advantage was measured.",
            "The sandbox-to-host cleanup change is recorded; any boundary-straddling pair is descriptive, not a clean causal/timing comparison.",
        ],
    }


def markdown(report):
    def percent(value):
        return "unknown" if value is None else f"{100 * value:.1f}%"

    lines = ["# R4 paired routing measurement", "", "R3 source review remains open; historical scores are provisional.", "",
             f"Solver revision: `{report['solver_revision']}`. Local automated only; no paid calls.", "",
             f"Attempted {report['attempted']}/{report['planned']} scheduled arms. Execution counts: `{report['execution_counts']}`.",
             f"Known worker wall: {report['known_worker_wall_seconds']:.1f}s; unknown duration for {report['wall_unknown_attempts']} attempts. Conservative charged budget: {report['charged_seconds']:.1f}s / 10,800s.",
             f"Complete, execution-context-matched primary synthetic pairs: {report['primary_synthetic_matched_complete_pairs']}/{report['primary_synthetic_pair_count']}. The tables retain incomplete and cross-context pairs descriptively, rather than dropping them.", ""]
    for title, select in (
        ("Primary synthetic comparisons", lambda p: p["seed"] == 61001 and p["role"] in {"development", "held_out"}),
        ("Predeclared development repeats (not necessarily independent)", lambda p: p["seed"] != 61001),
        ("Historical diagnostics — provisional", lambda p: p["role"] == "historical_diagnostic"),
        ("Controls — no expected solve", lambda p: p["role"] in {"unsupported_control", "random_control"}),
    ):
        lines += [f"## {title}", "", "| Case | Family / role | Seed | Blind char | Supplied char | Delta | Blind → supplied execution |", "|---|---|---:|---:|---:|---:|---|"]
        for p in filter(select, report["pairs"]):
            a, b = p["blind"], p["family_supplied"]
            ac, bc = a["post_hoc_delivered"], b["post_hoc_delivered"]
            lines.append(f"| `{p['case_id']}` | {p['family']} / {p['role']} | {p['seed']} | {percent(ac['char_accuracy'] if ac else None)} | {percent(bc['char_accuracy'] if bc else None)} | {percent(p['supplied_minus_blind_char'])} | {a['execution_status']} → {b['execution_status']} |")
        lines += [""]
    lines += ["## Route and saved-candidate evidence", "", "| Case / seed | Blind route / solver | Supplied route / solver | Saved-minus-delivered char gap (blind / supplied) |", "|---|---|---|---|"]
    for p in report["pairs"]:
        a, b = p["blind"], p["family_supplied"]
        lines.append(f"| `{p['case_id']}` / {p['seed']} | {', '.join(a['routes']) or 'unknown'} / {a['solver']} | {', '.join(b['routes']) or 'unknown'} / {b['solver']} | {percent(a['saved_minus_delivered_char'])} / {percent(b['saved_minus_delivered_char'])} |")
    lines += ["", "## Limitations", "", *[f"- {s}" for s in report["limitations"]], "",
              "The JSON companion binds each result file, request and delivered candidate by hash, and records timing, routes, seed evidence, delivery checks and available saved-text maxima.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preparation", type=Path, required=True)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--grading", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    report = build_report(args.preparation.resolve(), args.campaign.resolve(), args.grading.resolve())
    # Grade only once all scheduled jobs were attempted, or the controller
    # declared a terminal budget stop. Never write a misleading final report mid-run.
    if not (args.campaign / "summary.json").exists():
        raise ValueError("campaign has not reached its terminal ledger summary")
    text = markdown(report)
    if args.report.exists() and args.report.read_text() != text:
        raise ValueError("refusing to overwrite frozen report")
    write_once(args.report.with_suffix(".json"), report)
    if not args.report.exists():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(json.dumps({k: report[k] for k in ("attempted", "execution_counts", "primary_synthetic_observations")}))


if __name__ == "__main__":
    main()
