# Script Surface

Keep root-level scripts stable and broadly useful.  Research probes should live
under `scripts/research/` unless they have become part of the regular
development or evaluation workflow.

## Stable Root Scripts

| Script | Purpose |
|---|---|
| `setup_dev.sh` | Install/editable dev setup. |
| `build_rust_fast.sh` | Build mandatory Rust fast modules. |
| `validate_benchmark.py` | Validate a benchmark checkout. |
| `run_frontier_suite.py` | Main no-LLM frontier/regression suite runner. |
| `run_testgen_suite.py` | Agentic/automated benchmark packet runner. |
| `run_external_baselines.py` | External baseline harness. |
| `run_automated_parity_matrix.py` | Automated parity matrix runner. |
| `build_frontier_report.py` | Frontier report generation. |
| `build_parity_dashboard.py` | Parity dashboard generation. |
| `inspect_artifact.py` | Standard artifact inspector and optional LLM analyzer. |
| `count_prompt_tokens.py` | Prompt cost/token accounting. |
| `generate_synthetic_benchmark.py` | General synthetic benchmark generation. |
| `generate_transform_stress_suite.py` | Broad synthetic transform stress-suite generation. |
| `train_language_quality_scorer.py` | Generic language-quality scorer training/calibration. |
| `report_finalist_validation.py` | Generic finalist-validation report. |
| `report_language_candidate_ranker.py` | Generic language-ranker candidate report. |
| `report_language_ranker_failure_family.py` | Generic language-ranker failure-family report. |

## Domain-Specific Root Scripts

These are still root-level because they are part of recurring evidence packets,
not one-off probes:

| Script | Purpose |
|---|---|
| `report_copiale_evidence.py` | Compact Copiale evidence packet report. |
| `run_copiale_breadth_experiment.py` | Current broad Copiale candidate-breadth experiment wrapper. |
| `report_copiale_breadth_curve.py` | Saved broad-run breadth curve. |
| `report_copiale_breadth_diagnostics.py` | Saved broad-run diagnostics. |
| `report_copiale_mask_stability.py` | Null-mask stability report. |

## Research Scripts

Exploratory scripts live under `scripts/research/`. They may have more flags,
weaker compatibility promises, and calibration-only ground-truth columns. Do
not wire research scripts into agent tools or automated solver defaults without
first promoting the reusable logic into `src/` and adding focused tests.

Current Copiale reading/logogram experiments live in:

```text
scripts/research/copiale/
```

## Promotion Rule

Most Copiale probes, repair experiments, selector sweeps, and logogram-reading
harnesses now live under `scripts/research/copiale/`. Promote one back to the
root only when it has a stable command contract, non-experimental output, and
reusable logic has moved into `src/` with focused tests.
