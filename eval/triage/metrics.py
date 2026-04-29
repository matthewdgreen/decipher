"""Metrics computation for the transform triage evaluation.

Given a set of CapturedCase objects (with labels) and a list of strategies,
compute per-case and aggregate metrics.

Metrics
-------
  recall_at_k     — fraction of cases where at least one rescuable candidate
                    appears in the strategy's top-K.
  mrr_rescuable   — mean reciprocal rank of the first rescuable candidate.
  top1_rescuable  — fraction of cases where strategy's top-1 is rescuable.
  regression_rate — fraction of cases where the strategy *drops* a rescuable
                    candidate that baseline_ngram kept in its top-K.
  coverage        — fraction of cases with at least one labeled rescuable
                    candidate (those without are excluded from recall).

Usage
-----
    from triage.metrics import evaluate_all, MetricsRow

    rows = evaluate_all(strategies, cases, k_values=[1, 3, 10, 25])
    for row in rows:
        print(row)
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from triage.strategies.base import Strategy
from triage.types import CandidateRecord, CapturedCase


# ---------------------------------------------------------------------------
# Per-(strategy × case) result
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    strategy_name: str
    case_id: str
    language: str
    transform_family: str
    columns: int | None
    pipeline_template: str        # from case_id or metadata
    approx_length: int

    labeled_count: int            # candidates with rescuable != None
    rescuable_count: int          # candidates with rescuable == True

    # recall@K: 1 if any rescuable in top-K, 0 otherwise, -1 if no labels
    recall_at: dict[int, int] = field(default_factory=dict)
    # rank of first rescuable candidate (1-indexed), None if none in ranked list
    rank_first_rescuable: int | None = None
    # did the strategy's top-1 happen to be rescuable?
    top1_rescuable: bool = False
    # did the strategy remove a rescuable that baseline had in top-K?
    regression_vs_baseline: dict[int, bool] = field(default_factory=dict)


@dataclass
class MetricsRow:
    strategy_name: str
    language: str
    transform_family: str
    columns: str          # "all" or specific value
    approx_length: str    # "all" or specific value
    k: int
    recall_at_k: float
    mrr_rescuable: float
    top1_rescuable: float
    regression_rate: float
    n_cases: int
    n_labeled_cases: int   # cases with at least one rescuable label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ranked_ids(
    strategy: Strategy,
    case: CapturedCase,
) -> list[str]:
    entry = case.entry()
    return strategy.rank(case.candidates, {"language": entry.language})


def _rescuable_ids(candidates: list[CandidateRecord]) -> set[str]:
    return {c.candidate_id for c in candidates if c.rescuable is True}


def _recall_at(ranked: list[str], rescuable: set[str], k: int) -> int:
    """1 if any rescuable in ranked[:k], 0 otherwise."""
    return 1 if any(cid in rescuable for cid in ranked[:k]) else 0


def _rank_of_first_rescuable(ranked: list[str], rescuable: set[str]) -> int | None:
    for i, cid in enumerate(ranked):
        if cid in rescuable:
            return i + 1   # 1-indexed
    return None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_cases(
    strategies: list[Strategy],
    cases: list[CapturedCase],
    k_values: list[int] | None = None,
    baseline_name: str = "baseline_ngram",
) -> list[CaseResult]:
    """Return per-(strategy × case) results."""
    if k_values is None:
        k_values = [1, 3, 10, 25]

    # Precompute baseline rankings for regression comparison.
    baseline_strategy = next(
        (s for s in strategies if s.name == baseline_name), None
    )
    baseline_ranked: dict[str, list[str]] = {}
    if baseline_strategy is not None:
        for case in cases:
            baseline_ranked[case.case_id] = _ranked_ids(baseline_strategy, case)

    results = []
    for strategy in strategies:
        for case in cases:
            entry = case.entry()
            rescuable = _rescuable_ids(case.candidates)
            labeled_count = sum(1 for c in case.candidates if c.rescuable is not None)

            if labeled_count == 0:
                # Nothing labeled — skip this case entirely.
                continue

            ranked = _ranked_ids(strategy, case)
            rank_first = _rank_of_first_rescuable(ranked, rescuable)

            recall_at = {k: _recall_at(ranked, rescuable, k) for k in k_values}
            top1 = (len(ranked) > 0 and ranked[0] in rescuable)

            # Regression: did baseline have a rescuable in top-K that we don't?
            regression = {}
            if case.case_id in baseline_ranked:
                bl = baseline_ranked[case.case_id]
                for k in k_values:
                    bl_has = _recall_at(bl, rescuable, k) == 1
                    we_have = recall_at[k] == 1
                    regression[k] = bl_has and not we_have

            # Extract template from case_id heuristically.
            parts = case.case_id.split("_")
            template = "_".join(parts[5:]).rstrip("_s0123456789") if len(parts) > 5 else "unknown"

            results.append(CaseResult(
                strategy_name=strategy.name,
                case_id=case.case_id,
                language=entry.language,
                transform_family=entry.transform_family,
                columns=entry.columns,
                pipeline_template=template,
                approx_length=entry.approx_length,
                labeled_count=labeled_count,
                rescuable_count=len(rescuable),
                recall_at=recall_at,
                rank_first_rescuable=rank_first,
                top1_rescuable=top1,
                regression_vs_baseline=regression,
            ))

    return results


def aggregate_metrics(
    case_results: list[CaseResult],
    k_values: list[int] | None = None,
    baseline_name: str = "baseline_ngram",
) -> list[MetricsRow]:
    """Aggregate per-case results into summary MetricsRow objects."""
    if k_values is None:
        k_values = [1, 3, 10, 25]

    def _rows_for_slice(
        subset: list[CaseResult],
        language: str,
        transform_family: str,
        columns: str,
        approx_length: str,
    ) -> list[MetricsRow]:
        rows = []
        by_strategy: dict[str, list[CaseResult]] = {}
        for r in subset:
            by_strategy.setdefault(r.strategy_name, []).append(r)

        for strategy_name, results in by_strategy.items():
            n_labeled = sum(1 for r in results if r.rescuable_count > 0)
            labeled_results = [r for r in results if r.rescuable_count > 0]

            for k in k_values:
                if not labeled_results:
                    recall = mrr = top1 = regression = 0.0
                else:
                    recall = sum(
                        r.recall_at.get(k, 0) for r in labeled_results
                    ) / len(labeled_results)
                    rr_values = []
                    for r in labeled_results:
                        rank = r.rank_first_rescuable
                        rr_values.append(1.0 / rank if rank is not None else 0.0)
                    mrr = sum(rr_values) / len(rr_values) if rr_values else 0.0
                    top1 = sum(r.top1_rescuable for r in labeled_results) / len(labeled_results)
                    regression = sum(
                        r.regression_vs_baseline.get(k, False) for r in labeled_results
                    ) / len(labeled_results)

                rows.append(MetricsRow(
                    strategy_name=strategy_name,
                    language=language,
                    transform_family=transform_family,
                    columns=columns,
                    approx_length=approx_length,
                    k=k,
                    recall_at_k=round(recall, 4),
                    mrr_rescuable=round(mrr, 4),
                    top1_rescuable=round(top1, 4),
                    regression_rate=round(regression, 4),
                    n_cases=len(results),
                    n_labeled_cases=n_labeled,
                ))
        return rows

    all_rows: list[MetricsRow] = []

    # Overall slice.
    all_rows.extend(
        _rows_for_slice(case_results, "all", "all", "all", "all")
    )

    # Per-language slice.
    langs = sorted({r.language for r in case_results})
    for lang in langs:
        sub = [r for r in case_results if r.language == lang]
        all_rows.extend(_rows_for_slice(sub, lang, "all", "all", "all"))

    # Per transform_family slice.
    families = sorted({r.transform_family for r in case_results})
    for fam in families:
        sub = [r for r in case_results if r.transform_family == fam]
        all_rows.extend(_rows_for_slice(sub, "all", fam, "all", "all"))

    # Per-columns slice.
    col_values = sorted({str(r.columns) for r in case_results})
    for col in col_values:
        sub = [r for r in case_results if str(r.columns) == col]
        all_rows.extend(_rows_for_slice(sub, "all", "all", col, "all"))

    return all_rows


def evaluate_all(
    strategies: list[Strategy],
    cases: list[CapturedCase],
    k_values: list[int] | None = None,
    baseline_name: str = "baseline_ngram",
) -> list[MetricsRow]:
    """Full pipeline: per-case evaluation → aggregation."""
    case_results = evaluate_cases(strategies, cases, k_values=k_values, baseline_name=baseline_name)
    return aggregate_metrics(case_results, k_values=k_values, baseline_name=baseline_name)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_metrics_csv(rows: list[MetricsRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        if not rows:
            return
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
