"""Report generation for the transform triage evaluation.

Produces a markdown summary and CSV detail file from MetricsRow results.

Usage
-----
    from triage.report import write_report

    write_report(rows, out_dir=Path("eval/artifacts/reports/run_001"))
"""
from __future__ import annotations

from pathlib import Path

from triage.metrics import MetricsRow, write_metrics_csv


def write_report(
    rows: list[MetricsRow],
    out_dir: Path,
    run_id: str = "",
    notes: str = "",
) -> Path:
    """Write metrics.csv + summary.md to out_dir; return the markdown path."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics.csv"
    write_metrics_csv(rows, csv_path)

    md_path = out_dir / "summary.md"
    md_path.write_text(_render_markdown(rows, run_id=run_id, notes=notes), encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _render_markdown(
    rows: list[MetricsRow],
    run_id: str = "",
    notes: str = "",
) -> str:
    lines = []

    title = f"Transform Triage Evaluation — {run_id}" if run_id else "Transform Triage Evaluation"
    lines.append(f"# {title}\n")
    if notes:
        lines.append(f"{notes}\n")

    # Overall summary table (all slices, K=10).
    overall = [r for r in rows if r.language == "all" and r.transform_family == "all"
               and r.columns == "all" and r.k == 10]
    if overall:
        lines.append("## Overall — recall@10, MRR, regression (all cases, K=10)\n")
        lines.append(_table(
            ["strategy", "recall@10", "MRR", "top1", "regression", "n_cases", "n_labeled"],
            [
                [
                    r.strategy_name,
                    f"{r.recall_at_k:.3f}",
                    f"{r.mrr_rescuable:.3f}",
                    f"{r.top1_rescuable:.3f}",
                    f"{r.regression_rate:.3f}",
                    str(r.n_cases),
                    str(r.n_labeled_cases),
                ]
                for r in sorted(overall, key=lambda r: r.recall_at_k, reverse=True)
            ],
        ))

    # Per-K breakdown for the best strategy.
    if overall:
        best = sorted(overall, key=lambda r: r.recall_at_k, reverse=True)[0].strategy_name
        per_k = [r for r in rows if r.strategy_name == best
                 and r.language == "all" and r.transform_family == "all"
                 and r.columns == "all"]
        if per_k:
            lines.append(f"\n## Best strategy ({best}) — per-K breakdown\n")
            lines.append(_table(
                ["K", "recall@K", "MRR", "regression"],
                [
                    [str(r.k), f"{r.recall_at_k:.3f}", f"{r.mrr_rescuable:.3f}",
                     f"{r.regression_rate:.3f}"]
                    for r in sorted(per_k, key=lambda r: r.k)
                ],
            ))

    # Per-transform-family breakdown at K=10.
    family_rows = [r for r in rows if r.language == "all" and r.transform_family != "all"
                   and r.columns == "all" and r.k == 10]
    if family_rows:
        lines.append("\n## Per transform family — recall@10 (K=10)\n")
        families = sorted({r.transform_family for r in family_rows})
        for fam in families:
            lines.append(f"\n### {fam}\n")
            subset = [r for r in family_rows if r.transform_family == fam]
            lines.append(_table(
                ["strategy", "recall@10", "MRR", "regression", "n_labeled"],
                [
                    [r.strategy_name, f"{r.recall_at_k:.3f}", f"{r.mrr_rescuable:.3f}",
                     f"{r.regression_rate:.3f}", str(r.n_labeled_cases)]
                    for r in sorted(subset, key=lambda r: r.recall_at_k, reverse=True)
                ],
            ))

    # Per-columns breakdown at K=10.
    col_rows = [r for r in rows if r.language == "all" and r.transform_family == "all"
                and r.columns != "all" and r.k == 10]
    if col_rows:
        lines.append("\n## Per grid width (columns) — recall@10\n")
        lines.append(_table(
            ["columns", "strategy", "recall@10", "regression", "n_labeled"],
            [
                [r.columns, r.strategy_name, f"{r.recall_at_k:.3f}",
                 f"{r.regression_rate:.3f}", str(r.n_labeled_cases)]
                for r in sorted(col_rows, key=lambda r: (r.columns, -r.recall_at_k))
            ],
        ))

    lines.append(f"\nFull metrics: `{Path('metrics.csv').name}`\n")
    return "\n".join(lines)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def _fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [_fmt_row(headers), sep] + [_fmt_row(row) for row in rows]
    return "\n".join(lines) + "\n"
