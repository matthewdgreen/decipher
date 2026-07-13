# Experimental Consolidation Plan

This ledger tracks which successful research techniques have been promoted
into the main Decipher codebase and which remain diagnostic harnesses.  The
goal is to keep solver behavior reproducible while still preserving useful
research probes.

Ground-truth rule: benchmark plaintext may appear in calibration reports and
tests only after candidates have already been produced.  It must not influence
candidate generation, branch routing, solver selection, or runtime ranking.

## Promoted Core

| Capability | Main code location | Status |
|---|---|---|
| Broad null-mask finalist search and review | `src/analysis/homophonic_nulls.py`, `src/automated/runner.py`, `src/agent/tools_v2.py` | Production opt-in via `--homophonic-refinement null_masks`; default null-mask profile is broad/wide. |
| Rust batched null-mask solves | Rust fast module plus Python orchestration in automated runner | Production path for broad Copiale-style screens; Python remains orchestration/reporting. |
| Language-quality feature extraction | `src/analysis/language_scoring.py` | Main reusable scoring subsystem. Current trainable models remain calibration aids unless explicitly configured. |
| Diagnostic unknown-gap/logogram scoring | `src/benchmark/scorer.py` | Post-hoc reporting only; default benchmark char/word metrics are unchanged. |
| Nomenclator token rendering and recurrence packets | `src/analysis/nomenclator.py` | Promoted in this cleanup pass. Provides null-mask rendering, token-position views, symbol grouping, recurrence snippets, and whole-word/codeword expansions. |

## Research Harnesses To Keep, But Not Promote Yet

| Harness | Purpose | Reason it remains experimental |
|---|---|---|
| `scripts/research/copiale/probe_reading_holes.py` | Reading-first missing-word/logogram probe, optionally with LLM semantic rereader. | Useful diagnostic packets, but current local+LLM passes do not reliably surface true Copiale logograms. |
| `scripts/research/copiale/probe_logogram_hypotheses.py` and `scripts/research/copiale/run_copiale_logogram_repair_experiment.py` | Symbol-first logogram/codeword hypothesis calibration. | Recognition signal is too weak for solver routing. |
| `scripts/research/copiale/probe_copiale_phrase_hypotheses.py` | Word/phrase-level repair hypotheses. | Candidate generation and scoring were not strong enough to improve basins reliably. |
| `scripts/research/copiale/run_copiale_iterative_repair_tree.py` | Iterative local repair tree. | Behaves like a slow weak annealer; useful negative evidence, not production strategy. |
| `scripts/research/copiale/rank_candidate_texts_with_llm.py` | LLM ranking of candidate texts. | Helpful for diagnostics, but not reliable enough as a solver-ranker. |
| English Copiale analog fixture/scripts | Human-interpretability analog for debugging. | Useful explanatory fixture; not a benchmark parity case. |

## Next Promotion Candidates

1. Refactor remaining scripts to consume `analysis.nomenclator` instead of
   defining their own token-view renderers.
2. Move generic artifact/finalist extraction helpers out of Copiale-specific
   scripts if they are reused by more than one report.
3. Keep root-level `scripts/` small. Stable and recurring interfaces are
   documented in `scripts/README.md`; exploratory runners belong under
   `scripts/research/`.
4. Keep trainable language-quality models behind explicit configuration until
   held-out tests show stable gains across more than Copiale.
5. Keep logogram/reading-first tools report-only until they can recognize true
   missing-word/codeword evidence without swamping it with ordinary boundary
   letters.
