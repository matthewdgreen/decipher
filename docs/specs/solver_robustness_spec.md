# Solver robustness: F2 trigger, order-layer language guard, peel outcome honesty

**Status:** spec (2026-07-20). Implements tasks #8 and #9, both found during
agentic-suite-v2 validation (`36e057e`); repro ciphertexts are public in
`docs/evidence/agentic_frontier_suite_v2.md` (answers sealed). Base: `36e057e`,
suite baseline **1912 passed / 2 skipped**.

## S1 — F2 escalation trigger hardening (task #8)

**Evidence.** gs6 (width-11 keyed columnar, 201 chars): the SA converged to
pseudo-English whose greedy-segmentation dict_rate crossed
`_SOLVED_DICT_RATE`, so the `keyed_columnar_f2` escalation was skipped as
"already solved" (`keyed_columnar_f2` absent even at `budget_seconds=240`)
and the case failed at char 0.035–0.070.

**Analysis.** The trigger's dict-rate conjunct assumes dict_rate separates
solved from garbage; SA pseudo-English defeats it. No existing strategy in
`solve_transposition` is provably exhaustive (the SA column search is
stochastic; the exhaustive guarantee exists only inside
`analysis/columnar_search` for widths ≤8), so no incumbent is ever certain.

**Required change** (`src/analysis/transposition_solver.py`, the escalation
block landed in `7f3f09f`): drop the dict-rate skip entirely. The escalation
fires whenever `"columnar" in strategies_run` AND the deadline guard passes
(`remaining >= _F2_MIN_REMAINING_SECONDS`), unconditionally of the
incumbent's score. Adopt-if-better (strict `>` on `full_score`) already
guarantees no regression; the cost is a bounded ~7–9s (budget-scaled
restarts, unchanged) added to columnar-scope runs that the SA already
solved — accepted, and the deadline guard still bounds it. Update the block
comment to state the rationale (SA pseudo-English defeats dict-rate
gating; nothing upstream is exhaustive).

**Test changes:**
- `test_keyed_columnar_f2_skipped_when_sa_solves` becomes
  `..._runs_but_does_not_adopt_when_sa_solves`: `keyed_columnar_f2` is now
  PRESENT with `adopted is False` (equal/lower score never displaces the
  incumbent — strict `>`), and the returned plaintext is unchanged.
- The tiny-deadline test is unaffected (the ≥5s guard is the remaining skip
  path) — verify, don't change.
- New unit: with the SA producing a high-dict-rate WRONG candidate
  (monkeypatch `_dict_rate` to return 0.99, or patch the SA strategy to
  return a fixed garbage candidate) and `search_keyed_columnar`
  monkeypatched to return the true decode at a better `full_score`, the
  escalation runs and adopts. (Do NOT embed gs6's plaintext/keyword — that
  would unseal a live suite case.)

## S2 — order-layer signal gated to calibrated languages (task #9, fr misroute)

**Evidence.** gs15 (FRENCH mono, spaced): `run_automated(..., "fr")` routed
into the composite peel (`peeled_and_solved`, char 0.077). Mechanism, from
`transposition_suspicion` (`transposition_solver.py:223`): the shape-cosine
uses the fr reference correctly, but `ngram_structure_ratio` is compared
against `_ORDER_LAYER_STRUCTURE_ABSENT` — a threshold calibrated on ENGLISH
(composite Slice A). Real French adjacency sits below it, so a plain
substituted French text reads as "structure scrambled" → order layer
suspected → composite peel.

**Required change** (`transposition_solver.py`, inside
`transposition_suspicion`): compute `order_layer_suspected` only when
`language == "en"`; for any other language force it False and append a
reason string, e.g. `"order-layer structure threshold uncalibrated for
'fr'; signal disabled"`. The by-letter `suspicious` cosine (which IS
language-parameterized via `_monogram_reference`) is unchanged. Effect: a
non-en mono cipher falls through to the substitution default (which has
language-correct models/dictionaries — la measured 0.93 on this path); the
composite CONTENT auto-route becomes en-only until the threshold is
calibrated per language (recorded as future work in the spec, not code).

**Tests:** `transposition_suspicion(ct, "fr")` on letters that trip the en
signal → `order_layer_suspected is False` + the reason present;
`language="en"` behavior unchanged (existing tests must pass unmodified);
one routing-level test: `run_automated`-routing for a fr-labeled ≤26-symbol
text does NOT select the composite route (stub/routing-table level, no
heavy solve).

## S3 — composite peel outcome honesty (task #9, wrong-basin label)

**Evidence.** gs1-original/gs11-original and gs16: the peel reported
`outcome="peeled_and_solved"` while the substitution anneal had landed
wrong-basin pseudo-English (dict_rate far below any solved bar; char
0.04–0.09). Downstream verifiers catch it, but the label misleads agents,
artifacts, and the experiment installer's `mode_evidence`.

**Required change** (`src/automated/runner.py`,
`_run_composite_substitution_transposition` — where the step's `outcome`
is set): when the peel succeeded structurally but the substitution result's
`dict_rate` is below **0.5** (comment: real solves measure ≥0.85 on this
route — round-4 0.93; wrong-basin garbage measures ≤0.35 — gs16 0.09; 0.5
splits the measured populations with margin), set
`outcome="peeled_low_confidence"` instead of `"peeled_and_solved"`. Nothing
else changes: the decryption is still returned, the experiment installer
still installs (its evidence string simply carries the honest outcome), no
gate behavior changes.

**Tests:** monkeypatch the substitution anneal (or the C.1 sub-solver) to
return a low-dict-rate decode → `outcome == "peeled_low_confidence"`; a
real round-4-class solve keeps `peeled_and_solved` (existing composite
tests must pass unmodified — if any pins the outcome on a garbage fixture,
update it and say so).

## Constraints

- No routing changes beyond S2's language gate; no firewall changes; no
  TOOLS.md changes; the experiment surface is untouched (installer reads
  whatever outcome string arrives).
- The basin fragility itself (peel picking a wrong width/order that anneals
  to pseudo-English at moderate row counts) is NOT fixed here — S3 makes it
  honest, the verifier remains the backstop, and the limitation is recorded
  in this spec as accepted.
- Landing bar: suite baseline 1912/2 plus the new/updated tests, zero
  failures.

## Review adjudication (2026-07-20, Fable review: LAND WITH FIXES)

- **Finding 1 (MAJOR, fixed):** F2's cost was NOT bounded — the dominant cost
  is the exhaustive width sweep (~46k full-stream scorer calls for widths
  2..8), measured at 13 MINUTES on one width-6 test; the earlier "~9s"
  calibration and the restart-scaling heuristic were wrong.
  `ColumnarSearchConfig` gains `time_budget_seconds` (checked between
  widths, between restarts, every 512 exhaustive orders / 256 hill-climb
  steps; returns best-so-far on expiry), and the escalation passes
  `min(max(5, remaining-2), 30)`. The spec's S1 "bounded ~7-9s" cost claim
  is superseded by this hard 30s cap. The composite peel's own unbudgeted
  `search_keyed_columnar` call is out of scope here (pre-existing; noted).
- **Finding 2 (fixed):** the non-adopt test now pins `ran is True` and exact
  plaintext equality, and its budget drops 240s→60s.
- **Finding 3 (fixed):** the language gate folds case/None:
  `(language or "").lower() != "en"`.
- **Finding 4 (fixed):** shape-cos and structure-ratio computed only on the
  en branch.
- Consumer analysis verified clean: `order_layer_suspected` has exactly one
  control-flow consumer (the composite content auto-route — sanctioned);
  INV-0 panels compute their own atom; nothing branches on
  `peeled_and_solved`; the experiment installer installs low-confidence
  results unchanged.

## Post-land validation (orchestrator, not coder scope)

Re-run the v2 suite validation (`scripts/gen_agentic_frontier_suite_v2.py
--validate`): gs15 should now route substitution and score well above 0.5;
gs6's blind-path failure should flip to a solve (F2 now fires); gs16 should
report `peeled_low_confidence`. Expectations in the sealed answers are then
updated accordingly (gs6 note simplifies; gs15 may upgrade).
