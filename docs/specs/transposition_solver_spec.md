# Spec: Transposition-Family Solver (no-LLM)

Spec author: Fable (2026-07-15). First solver from the coverage-matrix plan
(`docs/solver_coverage_matrix.md`). Goal: crack the transposition families
the coverage sweep left at the ~0.38 mono-floor, reusing the route-solving
infrastructure that already works.

## Diagnosis (established, do not re-litigate)

The no-LLM `AutomatedBenchmarkRunner` solves `route_transposition` at 0.988
but leaves columnar/railfence/redefence/myszkowski/amsco/cadenus/
nihilist_transposition at ~0.20–0.42 EVEN with `transform_search="full"`.
Root cause: `analysis/transformers.py:candidate_transform_pipelines` only
generates identity/reverse/shift/NDownMAcross/route-reads and
`UnwrapTransposition` with a **natural-order** key (`"abc…"`). It never
searches **keyword-column permutations** (real columnar) or **railfence
zigzag** patterns. The apply/score/screen path (`transform_search.py`,
`transform_evaluation.py`, `pure_transposition.py`) is sound — route proves
it. This is a candidate-COVERAGE gap, not a scoring gap.

## SCOPE BOUNDARY (Codex owns the agentic v3 system)

Edit ONLY the no-LLM cryptanalysis engine: `src/analysis/` (transformers,
transform_search, pure_transposition, or a NEW module) and, if needed, the
no-LLM `src/automated/runner.py` transposition dispatch. DO NOT touch
`src/agent/**`, `src/investigation/**`, tools_v2, loop/episodes/context/
actions/reading/sessions. Prefer additive new functions/modules.

## What to build

A transposition solver that searches the permutation space for these
families, scored by an existing language model (dictionary + n-gram
log-likelihood — reuse `analysis/dictionary.py`, `analysis/ngram.py`,
`analysis/signals.py`; the same scoring the mono/route path uses). Cover,
in priority order:

1. **Keyword columnar** (complete + incomplete): for a candidate column
   count W (try a bounded range, e.g. 2..12 or до token_count//4), search
   over column ORDERINGS. W is small, so either (a) enumerate permutations
   for W ≤ ~8 with beam/branch-and-bound on partial column-adjacency
   n-gram score, or (b) SA/hill-climb over the ordering for larger W. Invert
   via the existing `_columnar_transposition` apply (unwrap=True) with the
   discovered ordering. This is the biggest win (columnar + myszkowski +
   nihilist_transposition + amsco share the keyed-column structure).
2. **Railfence / Redefence**: enumerate rail counts (2..~12) and offset;
   deterministic inverse per rail count — cheap brute force + language score.
3. **Myszkowski** (repeated key letters → grouped columns): a keyword search
   variant of (1); may fall out of (1) if the ordering search allows equal
   ranks. Best-effort.
4. **Amsco** (alternating 1–2 char cell fill + columnar): (1) plus the
   1–2 fill pattern; best-effort.
5. **Cadenus / Nihilist transposition**: stretch — the 25-row/n²-block keyed
   forms. Attempt if (1) generalizes; otherwise leave as documented gaps.

Route already works — keep it. Do NOT regress the existing transform search
or the substitution/homophonic paths.

Integration: the runner already screens for transposition via cipher_id
(IC preserved). Route this solver in on that signal (transposition
suspicion), bounded by a time/candidate budget so it never hangs a run.
Keep it behind the existing transposition dispatch; a plain-substitution or
homophonic case must NOT trigger a full transposition sweep.

## Tests

- `tests/test_transposition_solver.py`: for keyword-columnar, railfence,
  redefence (+ myszkowski/amsco if landed), GENERATE a case with the Slice-A
  cipher (`src/ciphers/transposition.py`) at a known key, run the solver,
  assert char accuracy ≥ 0.90 (a real solve) on ≥ a few seeds/lengths.
  Reuse the benchmark generator or the cipher classes directly.
- A determinism test (same input → same result) and a budget/time-bound test
  (returns within the cap on a hard/for-this-family-unsupported case without
  hanging).
- Regression: `route_transposition` still solves; a plain substitution case
  still solves and does NOT spend the transposition budget.
- Full suite zero-regression (baseline: current HEAD; note the pre-existing
  `test_lead_context` failure is Codex's, not ours).

## Acceptance (local, no LLM)

Re-run the transposition subset of the coverage sweep
(`scripts/solver_coverage_sweep.py` over a generated suite) and report the
before/after matrix: columnar/railfence/redefence/myszkowski should move
from ~0.38 to ≥0.90 (or report which families genuinely resisted and why).
Report per-family char accuracy over ≥3 fresh generated cases each (n=1 is
not enough to claim a solver).

## Deliverables

Files changed, the before/after coverage table for the transposition
families (≥3 cases each), suite counts, which families landed vs remain
gaps, time budget per family, deviations. No commits.
