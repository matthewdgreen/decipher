# Spec: Improvement Program Phase 2.4 — Multipage Shared-Key Benchmark Route

Parent plan: `docs/improvement_program_plan.md` (Phase 2.4; runs after
2.5 per the reprioritization). Spec author: Fable. Implementer: coding
sub-agent. Depends on: 2a libraries (`637a281`), 2b refinement
(`a9d3c2c`).

## Goal

Make multi-page shared-alphabet solving a first-class benchmark route:
combine a page group into one cipher, run the homophonic stack (+
refinements) once, project the shared key back per page, and emit
per-page plus group artifacts. This is where the research's strongest
Copiale basins (~79%) came from, and it supplies the cross-page evidence
that 2b showed single-page word-repair adoption lacks.

## Required behavior

1. **Group definitions.** `frontier/groups/<name>.json`:
   `{"name", "benchmark_split", "test_ids": [...], "language",
   "description"}`. Ship `copiale_evidence.json` with the five evidence
   packet Track-B test ids. Loader validates all pages share one symbol
   inventory policy (S-token space) before combining.
2. **Runner route.** `run_automated_multipage(pages, ...)` in
   `src/automated/` (new module `multipage_route.py` is fine; lazy
   imports of the analysis libraries per the standing constraint):
   build the combined cipher via `analysis.multipage.build_combined_cipher`,
   run the existing homophonic route ONCE on it (same budget/refinement
   options as single-page — reuse `run_automated`'s internals via the
   public wrappers, do not fork solver logic), then `project_pages` back.
   Word-repair refinement runs on the GROUP (this is the multipage
   evidence case the library was built for).
3. **CLI.** `decipher benchmark <root> --multipage-group
   frontier/groups/copiale_evidence.json [--automated-only ...]`
   (agentic multipage is out of scope). Also a
   `scripts/run_frontier_suite.py` passthrough is NOT required this
   slice.
4. **Artifacts.** One artifact per page (same schema as single-page
   automated artifacts, plus `multipage_group` metadata: group name,
   combined cipher hash, page index/offsets) and one group artifact
   (combined solve steps, refinement steps, per-page projections +
   post-hoc scores, aggregate char/word). Ground truth: post-hoc scoring
   only, per page, existing scorer.
5. **Adoption on multipage evidence.** With the group route, run
   word_repair in BOTH modes for the acceptance evidence: menu-only
   default, and `DECIPHER_WORD_REPAIR_ADOPT=1`. Cross-page collateral is
   now real (the library's adjudication was designed for it) — but do
   NOT change the default; report both.
6. **Tests.** Group loader validation (mismatched alphabets rejected);
   two-page synthetic group end-to-end with a stubbed solver (offsets/
   projection round-trip, artifact shapes, group metadata); firewall
   extension (group route leak-checked); lazy-import guard extended to
   the new module.

## Acceptance evidence (compute, report)

1. Five-page Copiale group, screen budget, `null_masks+word_repair`,
   both adoption modes. Report per-page char vs: single-page baseline
   (74.4/76.4/69.1/59.9/69.9), and note the research multipage reference
   (~79% best basins) — closing that gap fully is not required this
   slice; direction and adoption-safety are the questions. Explicitly
   report whether ADOPT=1 on group evidence avoids the single-page harm
   pattern (that's the key measurement).
2. Runtime: combined solve time vs 5× single-page.

## Out of scope

Agentic multipage runs; changing the adoption default (orchestrator
decision from your report); Borg groups (structure permits later).

## Review follow-ups (deferred; from the Fable review — LAND WITH FIXES applied)

- **Group-mean gate baseline (2.6-adjacent)**: the composed gate's
  validation leg anchors on page 0's projection, making the group gate
  order-dependent (near-vacuous when page 0 is weak, over-strict when
  strong; bounded by the group-native verdict leg). Follow-up: group-mean
  baseline — identical for single pages, changes only the group gate.
  Evaluate together with any adoption-default revisit.
- Silently-ignored benchmark flags under `--multipage-group` should warn
  or reject (only `--agentic` is guarded).
- Route's skip step diverges from `_skip_step` schema (cosmetic).
- `_selected_null_mask` now exists in three places — consolidate.
- Add a non-empty-mask route test (mask projection unpinned on the
  group path).

## Deliverables

Files changed, suite counts (baseline: record first), acceptance tables
(both modes), deviations. No commits.
