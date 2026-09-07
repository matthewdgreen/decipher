# R2 — Candidate preservation and delivery

Date: 2026-09-07. Base revision `c4554c5`; implementation is in the working tree.
Scope: the current improvement plan's R2 and M5.3a comparison/retention subset.
No new solver, learned ranker, sparse-null routing, provider calls, budget
increase, or declaration-gate weakening.

Status: **complete; R3 is next**.

## Verification

- Full default suite: **2,048 passed, three skipped, one failed** in 506.55 s.
  The failure was the fresh-clone packaging guard: Git did not yet track the
  newly added `src/investigation/portfolio.py`, so the clone omitted it.
  That module is now staged (no commit, no unrelated files staged).
- After that fix, the onboarding/fresh-clone, inspector, retention, CLI, and
  MCP checks passed: **129 passed in 11.84 s**. No failure remains unresolved;
  the entire eight-minute suite was not rerun after the staging-only fix.
- A separate native-inclusive run of CLI acceptance, retention, inspector,
  CLI, and MCP tests passed: **129 passed in 18.60 s**. This executes the real
  round-6 Rust solve and fake-reader gate lifecycle, not a paid verification.
- Broader focused comparison/episode/experiment/interface checks:
  **349 passed in 37.69 s**. Final inspector/retention checks after the last
  test refinement: **41 passed in 0.30 s**. `git diff --check` passed.

These runs overlap; counts are not additive. The three default-suite skips
retain their opt-in semantics; native CLI acceptance was enabled separately.

## Changes

- V3 comparisons now require `best_candidate`, `accepts_as_solution`, and a
  rationale, alongside ranking/verdicts. A supported partial can be preferred
  when none is solved. Legacy `winner` artifacts remain readable conservatively;
  new bindings do not write that overloaded contract.
- Fallback keeps the existing hierarchy: fresh positive verification, then a
  hash-fresh supported comparison partial, then existing scalar fallback. A
  null/unrankable comparison does not erase an earlier fresh preference.
  Changing **any** compared candidate invalidates the full-shortlist binding.
- Shared CLI/MCP and V3 state carries a deterministic portfolio of at most six
  distinct text hashes. Roles reserve attention for the automated baseline,
  compare best partial, positively verified candidate, scalar best, newest
  repair, and diverse family/refinement finalists. Existing scalar signals
  order candidates within a role; no grading label or new scorer participates.
  CLI/MCP's first installed automated experiment substitutes for V3 preflight.
- Portfolio entries expose selection reasons, aliases, source, creation turn,
  existing scores, and fresh-verification debt. Representative branches are
  protected from deletion while retained; rejection/supersession followed by
  refresh removes protection. Portfolio eviction never itself deletes a branch.
  The portfolio is **not an immutable version archive**: intentional in-place
  edits change identity and invalidate evidence; fork to preserve both versions.
- Status uses the shared reading renderer for metadata/null-mask candidates.
  Candidate inspection exposes recovered mode key state and structural fields
  through the existing candidate packet. Solver menus label exact duplicate
  text without renumbering finalists or discarding alternative recovered keys.
- CLI `--wait` now reports committed terminal status/slots/summary, not stale
  submit-time values. MCP and detached submission still return immediate state.
- The standard artifact inspector's human and LLM packets distinguish preference,
  advisory acceptance, shortlist freshness, and retained roles. Candidate
  reliability inspection also accepts CLI/MCP registry state and migrates older
  client comparison records read-only.

## Evidence and limitations

The [live Astra observation](codex_astra_r1_observation.md) supplied three
concrete lifecycle/reporting defects, not proof of a new solving frontier.
Read-only replay under R2 yields three distinct portfolio readings: the
automated baseline, the preferred period-8 recovery (three aliases), and the
period-4 partial. The status window now contains the actual recovered text;
candidate inspection exposes its recovered keyword/cycleword. Its exact
content hash, unsolved status, zero attestations, and source-document SHA-256
are unchanged. No live verifier was called.

All eight R0 saved artifacts were also replayed read-only after R2. Every
candidate's content/structure round-trip check passed; all source artifact
checksums still match the frozen R0 report.

Fixed fixtures cover substitution, null masking, custom boundaries, periodic
metadata decoding, and transforms through packet rendering and JSON resume;
existing experiment tests cover their installation paths. Other regressions
cover a third-ranked distinct null-mask finalist, deterministic bounded
deduplication, protected deletion, changed-text verification debt, legacy/client
comparisons, invalid verdicts, and fluent but unverified candidates. Comparison
acceptance and portfolio membership cannot satisfy the attestation gate.

R0 did not reproduce a current install/save/resume content loss in its eight
saved artifacts. Its historical rank-three null-mask case therefore motivates
diverse retention, not a fabricated claim that today's serializer lost the
delivered solution. These checks prove fixed-candidate mechanics; they do not
measure improved solve rate or independent-reader quality. R3 audits verification
and residuals; R4 measures routing before choosing later solver/routing changes.
