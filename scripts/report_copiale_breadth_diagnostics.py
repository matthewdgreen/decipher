#!/usr/bin/env python3
"""Explain Copiale breadth candidates that fool the current rankers.

This is an offline analysis script. It may use benchmark plaintext to label
already-produced finalists, but it never feeds those labels back into a solve.
The goal is to compare the generated best basin against the candidate selected
by the language-quality ranker and identify feature gaps to improve future
ground-truth-free scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.language_scoring import LinearLanguageQualityModel  # noqa: E402
from benchmark.scorer import score_decryption  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    test_id: str
    run_label: str
    artifact: str
    rank_in_artifact: int
    mask: tuple[str, ...]
    source: str
    char_accuracy: float
    language_quality_raw_score: float | None
    language_quality_rank_score: float | None
    validation_score_v2: float | None
    ensemble_score_v1: float | None
    selection_score: float | None
    decryption: str
    features: dict[str, float]
    validation_components: dict[str, float]
    ensemble_features: dict[str, float]

    @property
    def mask_label(self) -> str:
        return ",".join(self.mask) if self.mask else "(none)"

    @property
    def preview(self) -> str:
        return self.decryption[:180]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report why Copiale breadth rankers selected or missed strong basins."
    )
    parser.add_argument(
        "--experiment-dir",
        default="artifacts/copiale_breadth_experiment/four_page_wide",
        help="Breadth experiment directory containing ranker artifacts and models.",
    )
    parser.add_argument("--output", default="", help="Markdown report path.")
    parser.add_argument("--json-output", default="", help="JSON report path.")
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    experiment_dir = resolve_path(Path(args.experiment_dir))
    artifacts_by_test = discover_artifacts(experiment_dir)
    if not artifacts_by_test:
        raise SystemExit(f"No breadth artifacts found under {experiment_dir}")

    reports = []
    for test_id, artifacts in sorted(artifacts_by_test.items()):
        candidates = load_candidates(test_id, artifacts)
        if not candidates:
            continue
        model = load_model(experiment_dir, test_id)
        reports.append(analyze_test(test_id, candidates, model, top_n=args.top_n))

    payload = {
        "experiment_dir": str(experiment_dir),
        "tests": reports,
        "summary": summarize_reports(reports),
    }
    markdown = render_markdown(payload)
    output = resolve_path(Path(args.output)) if args.output else experiment_dir / "diagnostics.md"
    json_output = (
        resolve_path(Path(args.json_output))
        if args.json_output
        else experiment_dir / "diagnostics.json"
    )
    output.write_text(markdown, encoding="utf-8")
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")


def discover_artifacts(root: Path) -> dict[str, list[Path]]:
    artifacts: dict[str, list[Path]] = {}
    for path in sorted(root.glob("*/automated_only/*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        test_id = str(payload.get("test_id") or path.parent.name)
        artifacts.setdefault(test_id, []).append(path)
    return artifacts


def load_candidates(test_id: str, artifacts: list[Path]) -> list[Candidate]:
    rows: list[Candidate] = []
    seen: set[str] = set()
    for artifact in artifacts:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        ground_truth = str(payload.get("ground_truth") or "")
        null_step = next(
            (
                step for step in payload.get("steps") or []
                if isinstance(step, dict) and step.get("name") == "search_null_masks"
            ),
            None,
        )
        if not isinstance(null_step, dict):
            continue
        ranker = str(null_step.get("ranker") or artifact.parent.parent.parent.name)
        candidate_rows = list(null_step.get("top_finalists") or [])
        selected = null_step.get("selected")
        if isinstance(selected, dict):
            candidate_rows.insert(0, selected)
        for rank, row in enumerate(candidate_rows, start=1):
            if not isinstance(row, dict):
                continue
            text = str(row.get("decryption") or row.get("plaintext") or "")
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            scored = score_decryption(
                test_id,
                text,
                ground_truth,
                agent_score=0.0,
                status="completed",
            )
            rows.append(
                Candidate(
                    test_id=test_id,
                    run_label=ranker,
                    artifact=str(artifact),
                    rank_in_artifact=rank,
                    mask=tuple(str(symbol) for symbol in (row.get("mask") or [])),
                    source=str(row.get("source") or ""),
                    char_accuracy=float(scored.char_accuracy),
                    language_quality_raw_score=float_or_none(row.get("language_quality_raw_score")),
                    language_quality_rank_score=float_or_none(row.get("language_quality_rank_score")),
                    validation_score_v2=float_or_none(row.get("validation_score_v2")),
                    ensemble_score_v1=float_or_none(row.get("ensemble_score_v1")),
                    selection_score=float_or_none(row.get("selection_score")),
                    decryption=text,
                    features=float_dict(row.get("language_quality_features")),
                    validation_components=float_dict(row.get("validation_components_v2")),
                    ensemble_features=float_dict(row.get("ensemble_features_v1")),
                )
            )
    return rows


def load_model(experiment_dir: Path, test_id: str) -> LinearLanguageQualityModel | None:
    path = experiment_dir / "models" / f"{test_id}_candidate_only.json"
    if not path.exists():
        return None
    try:
        return LinearLanguageQualityModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return None


def analyze_test(
    test_id: str,
    candidates: list[Candidate],
    model: LinearLanguageQualityModel | None,
    *,
    top_n: int,
) -> dict[str, Any]:
    by_char = sorted(candidates, key=lambda row: row.char_accuracy, reverse=True)
    by_lq = sorted(
        candidates,
        key=lambda row: (
            row.language_quality_rank_score if row.language_quality_rank_score is not None else float("-inf"),
            row.language_quality_raw_score if row.language_quality_raw_score is not None else float("-inf"),
        ),
        reverse=True,
    )
    by_validation = sorted(
        candidates,
        key=lambda row: (
            row.validation_score_v2 if row.validation_score_v2 is not None else float("-inf"),
            row.selection_score if row.selection_score is not None else float("-inf"),
        ),
        reverse=True,
    )
    by_ensemble = sorted(
        candidates,
        key=lambda row: (
            row.ensemble_score_v1 if row.ensemble_score_v1 is not None else float("-inf"),
            row.validation_score_v2 if row.validation_score_v2 is not None else float("-inf"),
        ),
        reverse=True,
    )
    best = by_char[0]
    lq_pick = by_lq[0]
    return {
        "test_id": test_id,
        "candidate_count": len(candidates),
        "best_char": compact_candidate(best),
        "lq_pick": compact_candidate(lq_pick),
        "validation_pick": compact_candidate(by_validation[0]),
        "ensemble_pick": compact_candidate(by_ensemble[0]),
        "lq_gap": round(best.char_accuracy - lq_pick.char_accuracy, 6),
        "best_lq_rank": by_lq.index(best) + 1,
        "best_validation_rank": by_validation.index(best) + 1,
        "best_ensemble_rank": by_ensemble.index(best) + 1,
        "top_by_char": [compact_candidate(row) for row in by_char[:top_n]],
        "top_by_lq": [compact_candidate(row) for row in by_lq[:top_n]],
        "feature_deltas_lq_minus_best": feature_delta_rows(
            lq_pick,
            best,
            candidates,
            model=model,
            feature_source="features",
        ),
        "validation_component_deltas_lq_minus_best": feature_delta_rows(
            lq_pick,
            best,
            candidates,
            model=None,
            feature_source="validation_components",
        ),
        "model_contribution_deltas_lq_minus_best": model_contribution_rows(
            lq_pick,
            best,
            model,
        ),
    }


def compact_candidate(row: Candidate) -> dict[str, Any]:
    return {
        "mask": list(row.mask),
        "source": row.source,
        "run": row.run_label,
        "artifact": row.artifact,
        "rank_in_artifact": row.rank_in_artifact,
        "char_accuracy": round(row.char_accuracy, 6),
        "language_quality_raw_score": rounded(row.language_quality_raw_score),
        "language_quality_rank_score": rounded(row.language_quality_rank_score),
        "validation_score_v2": rounded(row.validation_score_v2),
        "ensemble_score_v1": rounded(row.ensemble_score_v1),
        "selection_score": rounded(row.selection_score),
        "preview": row.preview,
    }


def feature_delta_rows(
    picked: Candidate,
    best: Candidate,
    candidates: list[Candidate],
    *,
    model: LinearLanguageQualityModel | None,
    feature_source: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    pick_features = getattr(picked, feature_source)
    best_features = getattr(best, feature_source)
    names = sorted(set(pick_features) | set(best_features))
    scales = feature_scales(candidates, names, feature_source=feature_source)
    model_names = set(model.feature_names) if model else set()
    rows = []
    for name in names:
        pick_value = float(pick_features.get(name) or 0.0)
        best_value = float(best_features.get(name) or 0.0)
        delta = pick_value - best_value
        scale = scales.get(name) or 1.0
        rows.append({
            "feature": name,
            "lq_pick": round(pick_value, 6),
            "best_char": round(best_value, 6),
            "delta_lq_minus_best": round(delta, 6),
            "z_delta": round(delta / scale, 6),
            "in_model": name in model_names,
        })
    rows.sort(key=lambda row: abs(float(row["z_delta"])), reverse=True)
    return rows[:limit]


def model_contribution_rows(
    picked: Candidate,
    best: Candidate,
    model: LinearLanguageQualityModel | None,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    if model is None:
        return []
    rows = []
    for idx, name in enumerate(model.feature_names):
        pick_value = float(picked.features.get(name) or 0.0)
        best_value = float(best.features.get(name) or 0.0)
        scale = model.scales[idx] or 1.0
        weight = model.weights[idx]
        contribution = weight * ((pick_value - best_value) / scale)
        rows.append({
            "feature": name,
            "weight": round(weight, 6),
            "lq_pick": round(pick_value, 6),
            "best_char": round(best_value, 6),
            "delta_lq_minus_best": round(pick_value - best_value, 6),
            "contribution_lq_minus_best": round(contribution, 6),
        })
    rows.sort(key=lambda row: abs(float(row["contribution_lq_minus_best"])), reverse=True)
    return rows[:limit]


def feature_scales(
    candidates: list[Candidate],
    names: list[str],
    *,
    feature_source: str,
) -> dict[str, float]:
    scales = {}
    for name in names:
        values = [
            float(getattr(row, feature_source).get(name) or 0.0)
            for row in candidates
        ]
        if not values:
            scales[name] = 1.0
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scales[name] = math.sqrt(variance) or 1.0
    return scales


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [float(report["lq_gap"]) for report in reports]
    return {
        "test_count": len(reports),
        "mean_lq_gap": round(sum(gaps) / len(gaps), 6) if gaps else None,
        "exact_lq_picks": sum(1 for report in reports if float(report["lq_gap"]) <= 0.000001),
        "top3_lq_captures": sum(1 for report in reports if int(report["best_lq_rank"]) <= 3),
        "top10_lq_captures": sum(1 for report in reports if int(report["best_lq_rank"]) <= 10),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Copiale Breadth Diagnostics",
        "",
        "Ground truth is used only to label candidates after all solves completed.",
        "",
        f"Experiment: `{payload['experiment_dir']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    summary = payload["summary"]
    lines.extend([
        f"| tests | {summary.get('test_count')} |",
        f"| exact LQ picks | {summary.get('exact_lq_picks')} |",
        f"| top-3 LQ captures | {summary.get('top3_lq_captures')} |",
        f"| top-10 LQ captures | {summary.get('top10_lq_captures')} |",
        f"| mean LQ char gap | {format_percent(summary.get('mean_lq_gap'))} |",
        "",
        "## Tests",
        "",
        "| Test | Candidates | Best Char | LQ Pick | Gap | Best LQ Rank | Best Val Rank | Best Ens Rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for report in payload["tests"]:
        lines.append(
            "| {test} | {count} | {best} | {pick} | {gap} | {lq_rank} | {val_rank} | {ens_rank} |".format(
                test=report["test_id"],
                count=report["candidate_count"],
                best=format_percent(report["best_char"]["char_accuracy"]),
                pick=format_percent(report["lq_pick"]["char_accuracy"]),
                gap=format_percent(report["lq_gap"]),
                lq_rank=report["best_lq_rank"],
                val_rank=report["best_validation_rank"],
                ens_rank=report["best_ensemble_rank"],
            )
        )
    for report in payload["tests"]:
        lines.extend(render_test(report))
    return "\n".join(lines) + "\n"


def render_test(report: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"## {report['test_id']}",
        "",
        "Key picks:",
        "",
        "| Pick | Mask | Run | Source | Char | LQ Raw | LQ Rank | Val2 | Ens | Preview |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, key in (
        ("Best char", "best_char"),
        ("LQ pick", "lq_pick"),
        ("Validation pick", "validation_pick"),
        ("Ensemble pick", "ensemble_pick"),
    ):
        lines.append(candidate_line(label, report[key]))
    lines.extend([
        "",
        "Largest LQ-feature deltas, LQ pick minus best-char candidate:",
        "",
        delta_table(report["feature_deltas_lq_minus_best"], contribution=False),
        "",
        "Largest validation-component deltas, LQ pick minus best-char candidate:",
        "",
        delta_table(report["validation_component_deltas_lq_minus_best"], contribution=False),
    ])
    if report.get("model_contribution_deltas_lq_minus_best"):
        lines.extend([
            "",
            "Largest trained-model contribution deltas, LQ pick minus best-char candidate:",
            "",
            contribution_table(report["model_contribution_deltas_lq_minus_best"]),
        ])
    lines.extend([
        "",
        "Top by post-hoc character accuracy:",
        "",
        candidate_rank_table(report["top_by_char"]),
        "",
        "Top by language-quality rank:",
        "",
        candidate_rank_table(report["top_by_lq"]),
    ])
    return lines


def candidate_line(label: str, row: dict[str, Any]) -> str:
    return (
        f"| {label} | {mask_label(row)} | {row.get('run')} | {row.get('source')} | "
        f"{format_percent(row.get('char_accuracy'))} | {format_number(row.get('language_quality_raw_score'))} | "
        f"{format_number(row.get('language_quality_rank_score'))} | {format_number(row.get('validation_score_v2'))} | "
        f"{format_number(row.get('ensemble_score_v1'))} | {escape_cell(str(row.get('preview') or '')[:120])} |"
    )


def candidate_rank_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Rank | Mask | Run | Source | Char | LQ Raw | LQ Rank | Val2 | Ens | Preview |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {mask_label(row)} | {row.get('run')} | {row.get('source')} | "
            f"{format_percent(row.get('char_accuracy'))} | {format_number(row.get('language_quality_raw_score'))} | "
            f"{format_number(row.get('language_quality_rank_score'))} | {format_number(row.get('validation_score_v2'))} | "
            f"{format_number(row.get('ensemble_score_v1'))} | {escape_cell(str(row.get('preview') or '')[:120])} |"
        )
    return "\n".join(lines)


def delta_table(rows: list[dict[str, Any]], *, contribution: bool) -> str:
    del contribution
    lines = [
        "| Feature | LQ Pick | Best Char | Delta | Z-Delta | In Model |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {pick:.3f} | {best:.3f} | {delta:+.3f} | {z:+.3f} | {in_model} |".format(
                feature=row["feature"],
                pick=float(row["lq_pick"]),
                best=float(row["best_char"]),
                delta=float(row["delta_lq_minus_best"]),
                z=float(row["z_delta"]),
                in_model=1 if row.get("in_model") else 0,
            )
        )
    return "\n".join(lines)


def contribution_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Feature | Weight | LQ Pick | Best Char | Delta | Contribution |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {weight:+.4f} | {pick:.3f} | {best:.3f} | {delta:+.3f} | {contribution:+.4f} |".format(
                feature=row["feature"],
                weight=float(row["weight"]),
                pick=float(row["lq_pick"]),
                best=float(row["best_char"]),
                delta=float(row["delta_lq_minus_best"]),
                contribution=float(row["contribution_lq_minus_best"]),
            )
        )
    return "\n".join(lines)


def mask_label(row: dict[str, Any]) -> str:
    return ",".join(str(symbol) for symbol in (row.get("mask") or [])) or "(none)"


def float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, raw in value.items():
        parsed = float_or_none(raw)
        if parsed is not None:
            out[str(key)] = parsed
    return out


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def format_percent(value: Any) -> str:
    parsed = float_or_none(value)
    return "n/a" if parsed is None else f"{parsed * 100:.1f}%"


def format_number(value: Any) -> str:
    parsed = float_or_none(value)
    return "n/a" if parsed is None else f"{parsed:.3f}"


def escape_cell(value: str) -> str:
    return value.replace("|", "/").replace("\n", " ")


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    main()
