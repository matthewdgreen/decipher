# Keyed-columnar exposure (workplan step 3) — corrected scope

**Status:** spec (2026-07-19). Supersedes the workplan/frontier framing of the
gap, which measurement showed is stale.

## 0. Corrected problem statement (measured 2026-07-19, current main 39b889a)

The frontier docs (`docs/evidence/frontier_extension_2026_07.md` §4,
`agentic_frontier_suite_results.md` fs3) claim a PLAIN keyed columnar
"auto-routes to the geometric transposition screen, which fails," with
`analysis/columnar_search.py` (F2) reachable only inside the composite peel.
**That is no longer true.** Since `8e40744` (2026-07-16, "Add
transposition-family solver"), the automated route for a
content-suspicious/named transposition is `_run_transposition_solver` →
`analysis.transposition_solver.solve_transposition`, whose SA column-order
search covers keyed columnar. Verified on the actual fs3 ciphertext:
`run_automated` solves it end-to-end (solver
`transposition_permutation_search`, dict_rate 0.94, ~6s). The fs3 *agent*
failure was tool-selection: it escalated the geometric transform screens
(which genuinely lack keyed-columnar coverage) and never submitted an
`automated_solver` experiment.

**The real residual gap is width robustness.** Fresh per-family sweep
(11 cases, keyword widths 4–11, ~190-char no-boundary English): the SA
search solved 9/11; both misses were width-11 (`LIGHTHOUSEX`). The F2
module (`search_keyed_columnar`, hill-climb mode, language scorer) solves
both width-11 misses in ~7s each.

## 1. Change — F2 escalation inside `solve_transposition`

**File:** `src/analysis/transposition_solver.py`.

After the existing strategy loop produces its best candidate (the same seam
as the myszkowski-cousin escalation at ~line 1148), add a keyed-columnar F2
escalation:

- Trigger: the run included the `columnar` strategy (explicitly hinted or via
  the cheap-strategy default), AND (`best is None` OR
  `_dict_rate(best.plaintext, language) < _SOLVED_DICT_RATE`), AND the
  deadline has not expired.
- Action: call `analysis.columnar_search.search_keyed_columnar` over the
  letters-only stream with `make_language_scorer(language)`. Wrap the top
  finalist as a `_Candidate` with `family="columnar"` and params
  `{"width": column_count, "order": list(column_order),
  "keyword": keyword, "engine": "columnar_search_f2",
  "method": finalist.method}`; adopt it iff its `full_score` beats the
  incumbent best (same adopt-if-better convention as the runner's additive
  block).
- Budget: respect the existing `deadline` object — pass whatever bound the
  F2 config supports and skip the escalation entirely when the remaining
  budget is under ~5 seconds (measured F2 cost ≈ 7–9s on ~250 chars). The
  escalation must never raise: wrap in the module's usual try/except-to-note
  pattern; on error record a `keyed_columnar_f2` note in the result and keep
  the incumbent.
- Surface in the result dict: when the escalation ran, add
  `"keyed_columnar_f2": {"ran": true, "adopted": <bool>, "method": ...,
  "score": ...}` so artifacts/steps show it (the runner step already copies
  the solver result wholesale via named keys — add this key to the step in
  `_run_transposition_solver` too).

No routing changes, no new tools, no TOOLS.md changes. The MCP/agent
surface reaches this through the existing `automated_solver` experiment
(cipher_system naming a transposition/columnar family, or the content
auto-route), and the P1 decoded-branch installer (39b889a) already makes
the empty-key transposition result verifiable/declarable.

## 2. Docs corrections (same commit)

- `docs/evidence/frontier_extension_2026_07.md` §4: append a dated
  CORRECTION paragraph: the standalone route has covered keyed columnar
  since `8e40744`; fs3's agent-surface failure was transform-screen
  tool-selection; the measured residual gap was width-11 SA misses, closed
  by this spec's F2 escalation.
- `docs/evidence/agentic_frontier_suite_results.md` fs3 row: append a
  one-line dated correction note (do not rewrite the original grading).

## 3. Tests

In the existing transposition-solver test module (follow current layout):
- width-11 regression: `columnar_encrypt(<~190-char English plaintext>,
  "LIGHTHOUSEX")` → `solve_transposition` returns the exact plaintext with
  `keyed_columnar_f2.adopted == true` (use a fixed plaintext; the F2
  hill-climb is seeded — `search_keyed_columnar` takes a seed via its
  config; pin it if flakiness appears).
- non-regression: a width-6 case the SA already solves returns the same
  plaintext and does NOT report `keyed_columnar_f2.adopted` (the escalation
  is skipped because dict_rate ≥ threshold).
- deadline safety: with an already-expired/tiny deadline the escalation is
  skipped and the result still completes.

Baseline: suite is 1861 passed / 2 skipped at `dd8331d`.

## 4. Review adjudication (2026-07-19, Fable review: LAND WITH FIXES)

- **Finding 1 (fixed):** the F2 config has no deadline hook, so a long stream
  could overshoot a nearly-spent budget — the restart budget is now scaled to
  `deadline.remaining()` (64 restarts ≈ 9s at ~250 chars, linear in length;
  floor 8, cap 64). Exhaustive widths (≤8) are unaffected.
- **Finding 2 (fixed):** both non-deadline tests now pass an explicit
  `budget_seconds=240.0` so machine load cannot starve (or un-skip) the
  escalation.
- **Finding 3 (fixed):** import failure now reports `ran: false`; both error
  shapes carry `method`/`score` keys for shape stability.
- **Finding 4 (taken):** the adopted finalist is now chosen by maximizing
  `full_score` over all (≤ top_n) finalists rather than trusting the
  ngram-only ranking's top-1 — the ranking scorer and the adoption metric
  differ (dict_weight), so the top-ranked finalist is not always the best
  candidate under the adoption metric.
- Accepted deviations: recovery test uses `THUNDERBOLT` (width 11) rather
  than the parenthetical `LIGHTHOUSEX`; the tiny-deadline test tolerates
  `kc is None or not adopted` (the ≥5s guard makes the skip structural).

## 5. Out of scope

- Transform-screen keyed-columnar coverage (the screens stay geometric; the
  automated route is the designated path).
- Agent-guidance text changes (revisit only if a future dogfood shows agents
  still miss the automated route after this lands).
- The stale `docs/frontier_solver_comparison.md` regeneration (workplan
  "automated frontier JSONL" item — separate task).
