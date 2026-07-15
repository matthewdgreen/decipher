# Spec: Agent Loop v3 - M5.1 (Recovery, Adjudication, and Declaration)

Status: REVISED AFTER M6 FORENSICS - READY FOR IMPLEMENTATION IN ORDERED
SLICES. This revision supersedes the original declaration-trigger-only M5.1
spec. The original positive-declaration work remains, but it is no longer the
first implementation step.

Parent documents:

- `docs/specs/agent_v3_design.md`
- `docs/specs/agent_v3_m3_spec.md`
- `docs/specs/agent_v3_m5_spec.md`
- `docs/specs/agent_v3_m6_spec.md`
- `docs/improvement_program_plan.md`

Evidence:

- `artifacts/m6_bakeoff/summary.jsonl`
- the v3 Borg artifacts under `artifacts/m6_bakeoff/v3/borg_*/`

## Purpose

M6 showed that v3 is much cheaper than v2 and approximately tied on the
incomplete aggregate character metric, but underperforms on Borg. The first
M5.1 hypothesis was that M5's fresh-attestation gate worked but lacked a
positive trigger that converted a good verification into a declaration.

Artifact forensics show that declaration timing is only one part of the
problem. The decisive failures are:

1. reading episodes produced useful target-language reconstructions, but the
   reading-to-repair composite rejected realistic punctuation and hole markers;
2. negative verification arrived at the end of several runs and was not turned
   into a focused repair cycle;
3. one fallback selected a much weaker branch while a substantially better
   generated branch was available;
4. the bake-off summary omitted negative verification episodes, obscuring the
   actual control flow;
5. the M6 matrix is incomplete, so single-replicate Copiale/synthetic results
   are directional rather than a final default-switch result.

M5.1 therefore repairs the complete path:

`reading -> conservative repair proposal -> repaired branch -> verification ->
declaration or honest partial/fallback`

It must do so without restoring the v2 coercion cascade and without weakening
the ground-truth firewall.

## Binding Forensic Findings

These findings are post-hoc diagnostics. Character and word accuracy below are
never available to the solver or model during a run.

### Borg 0109v

All three preflight-on v3 runs ended on the same 91.0% character / 66.7% word
basin. That basin was the post-hoc-best generated branch in each run. Verify
ran on turns 24, 25, and 24; every result was `reader_accepts=false`, coherence
2. The lead therefore did not have a positive attestation waiting to be
declared.

Two runs attempted `hypothesis_apply_reading`. Both calls failed with:

`reading contains a non-plaintext-alphabet character`

The offending character was ordinary punctuation from the Reading. The useful
reading was never converted into edits.

### Borg 0045v

- Replicate 1 generated an 83.6% branch, verified a same-decode branch
  negatively at turn 11 (coherence 1), and later fell back to the 83.6% branch.
- Replicate 2 ran no verify episode and selected a 34.1% transform finalist at
  fallback even though a 68.9% branch was present.
- Replicate 3 generated an 83.6% branch, verified it negatively at turn 22
  (coherence 1), and explicitly declared the case unsolved.

An early negative verification alone did not produce repair progress in
replicate 1. Fallback selection, not just declaration timing, caused the
replicate-2 collapse.

### Bake-off telemetry

`scripts/run_v3_bakeoff.py:build_summary_row` currently reads verification
fields only from `solution.attestation`. A negative or unattached attestation
lives in the artifact's top-level `attestations` list, so five of the six
preflight-on Borg runs that actually verified appear in the summary as if they
did not.

The current summary contains 22 rows, not the M6 design's 42. The official
aggregator correctly marks the decision matrix incomplete. M5.1 must not turn
that incomplete result into a default-switch decision.

## Non-Negotiable Constraints

1. Ground truth remains post-hoc grading only. It cannot influence prompts,
   branch selection, repair derivation, retry triggers, verification, fallback,
   or declaration.
2. Verify episodes continue to receive only candidate plaintext and language.
   They receive no cipher, key, branch scores, benchmark metadata, or lead-authored
   goal text.
3. Worker episodes remain isolated and single-writer rules remain intact.
4. V2 behavior stays byte-for-byte unchanged.
5. M5.1 adds a small number of state-derived hints and structural preferences;
   it does not add inner retries, hidden tool calls, or gate-bounce prompts.
6. A human-readable reconstruction is not automatically a key-repair command.
   Only explicitly machine-actionable, position-bound text may generate edits.

## Implementation Order

The slices below land and test in order. Do not begin live acceptance until all
local slices pass review.

## Slice A - Correct M6 Verification Telemetry

Update `scripts/run_v3_bakeoff.py` before using the old summary as diagnostic
evidence.

### Summary fields

Preserve existing fields for compatibility, with precise meanings:

- `attested`: declaration/fallback solution carries an attached attestation;
- `verify_reader_accepts`, `verify_coherence`, `verify_created_turn`: latest
  completed verification for the selected branch when available, otherwise the
  latest completed verification in the artifact;
- `late_turn_hint_fired`: unchanged.

Add:

- `verify_ran: bool`;
- `verify_count: int`;
- `verify_branch: str | null`;
- `declaration_attested: bool` (explicit alias that removes ambiguity);
- `positive_attestation_available: bool`;
- `attested_fallback: bool` (false for old artifacts).

Source verification history from top-level `artifact["attestations"]`. Use
`solution.attestation` only to determine declaration attachment. Sort multiple
attestations deterministically by `(created_turn, episode_id)`.

### Existing-artifact refresh

Add a report-only refresh path that reconstructs summary rows from artifacts
without making API calls. It must write a new path unless the caller explicitly
requests overwrite. Rebuild the 22-row report after implementation and record
the corrected verification counts/timing in the implementation report.

### Tests

- negative unattached attestation is reported as `verify_ran=true` and
  `attested=false`;
- positive attached attestation sets both `attested` and
  `declaration_attested`;
- multiple attestations choose the deterministic latest record;
- v2 rows keep null verification details and unchanged declaration semantics;
- refresh performs no provider/model construction.

## Slice B - Make Readings Repairable Without Treating Prose as a Key

The current producer and consumer contracts disagree: the reading worker is
asked for a plain-language reconstruction, while `hypothesis_apply_reading`
accepts only plaintext-alphabet characters and spaces. Resolve this with an
additive machine-actionable field.

### Reading schema

Extend `ReadingFragment` and the reading episode result schema with:

`repair_text: string | null`

Meanings:

- `text` is human-readable interpretation. It may contain punctuation,
  editorial brackets, alternatives, and prose-style holes.
- `repair_text` is an optional conservative proposal bound to the fragment's
  `start`/`end` token span. It may contain only plaintext-alphabet letters,
  spaces, and `?`.
- A letter proposes a plaintext symbol at that aligned position.
- Space proposes a word boundary and consumes no token.
- `?` consumes exactly one token position but proposes no mapping.
- If the worker does not know the position or unknown-run length, it must set
  `repair_text=null`; it must not invent enough question marks to fit.

The reading-worker contract must explicitly request `repair_text` only for
high-confidence, position-bound fragments. Naturalized spelling, expanded
abbreviations, punctuation, and glosses remain in `text` unless the worker can
bind them conservatively to the cipher positions.

### Legacy and inline normalization

For existing Readings and inline `reading_text`:

- clean plaintext-alphabet text plus whitespace remains supported;
- a narrow safe punctuation set (`. , ; : !` and line breaks) may be converted
  to spaces;
- brackets, slashes, parentheses, ellipses used as unknown spans, editorial
  alternatives, digits, and other ambiguous notation make that fragment
  non-actionable rather than guessed;
- invalidity in one fragment must not abort all other fragments.

Do not silently strip ambiguous material and concatenate the surrounding text;
that would shift alignment and create false global mappings.

### `hypothesis_apply_reading`

Apply fragments independently:

1. prefer `repair_text` when present;
2. otherwise attempt only the safe legacy normalization above;
3. skip ambiguous/non-actionable fragments and record why;
4. derive edits only from letters; `?` aligns but casts no mapping vote;
5. preserve existing conflict, majority, boundary, dry-run, and fork behavior;
6. return `skipped_fragments`, `actionable_fragment_count`, and
   `no_actionable_fragments` when appropriate.

Default to fragments with confidence >= `MIN_REPAIR_FRAGMENT_CONFIDENCE`
(named constant, default 0.65). A lower-confidence fragment remains visible in
the Reading but cannot change the key automatically. Do not expose a prompt
argument that casually bypasses this threshold in M5.1.

### Regression fixtures

Add artifact-derived, ground-truth-free fixtures covering:

- `ET BREUITER UT PLURES MANERENT VIVI;` (safe punctuation);
- a fragment containing `[---]` (skipped, not fatal);
- a fragment containing an editorial alternative such as
  `[experi-/examin-?]` (skipped, not flattened);
- a multi-fragment Reading with one actionable and one skipped fragment;
- `?` consuming a token without producing a mapping;
- the two exact `hypothesis_apply_reading` failure shapes from the 0109v
  artifacts now producing either a useful dry-run or a precise
  `no_actionable_fragments` result, never the old single-character error.

## Slice C - Turn Verification Into a Mid-Run Decision Point

Retain M5's fresh-attestation declaration gate. Add concise v3-only guidance
and state-derived hints.

### Lead brief

Add approximately four lines:

1. Verify the current best branch by mid-budget once it contains a sustained
   target-language reading; do not reserve verify for the final turn.
2. Positive verify (`reader_accepts=true`, coherence >= 7) is a signal to
   declare now unless concrete evidence identifies a better branch.
3. A negative but partially readable verify is a repair signal: obtain or reuse
   a Reading, apply its conservative fragments on a fork, adjudicate, and
   reverify changed content.
4. Do not verify unchanged branch content twice.

### Attestation response bands

Named constants:

- `DECLARE_COHERENCE = 7`;
- `REPAIRABLE_COHERENCE_MIN = 2`;
- `LATE_VERIFY_TURNS = 4`;
- `POST_ATTEST_PATIENCE = 2`.

Guidance:

- positive: `reader_accepts=true` and coherence >= 7 -> declare;
- repairable partial: coherence 2-6, or a nonempty gloss with localized
  anomalies -> reading/repair cycle;
- collapsed: coherence 0-1 with no sustained reading -> broaden/reject rather
  than polishing the same branch.

These are workflow hints, not declaration gates. The attestation gate continues
to require freshness, not positivity, so an agent may honestly declare a
partial solution with a negative attestation when appropriate.

### Hints and deduplication

Add artifact-visible LoopEvents, emitted at most once per branch content hash:

- `mid_budget_verify_hint`;
- `negative_verify_repair_hint`;
- `positive_attestation_declare_hint`;
- retain/extend `late_turn_attestation_hint` at <=4 turns.

The negative hint fires only when no reading-application or repair episode has
addressed that attested content. A changed branch hash resets eligibility.

## Slice D - Positive Declaration and Attested Fallback

Implement the useful parts of the original M5.1 proposal.

1. If a fresh positive attestation exists and the lead has not declared within
   `POST_ATTEST_PATIENCE` turns, name the branch in the context and emit the
   positive hint event.
2. On exhaustion/error, before generic fallback, select a branch carrying a
   fresh positive attestation. Attach it to `SolutionDeclaration`, set
   `auto_declared=true`, `attested_fallback=true`, and retain status
   `fallback_declared`.
3. Map fallback self-confidence conservatively from coherence (`coherence / 10`).
4. Never promote a stale or negative attestation as if it were positive.
5. Never prefer a branch merely because it has *any* attestation. A negative
   reader verdict is evidence, not a generic tie-break bonus.

Tests cover positive, negative, stale-hash, renamed-branch, and v2-no-change
paths.

## Slice E - Robust Fallback Adjudication

The 0045v replicate-2 failure shows that the scalar dictionary/quadgram fallback
can choose a word-island branch over a substantially better reading. Fix this
without using ground truth and without making an unbudgeted model call after
the lead loop exits.

### Fresh compare records

When a `compare` episode completes, the dispatcher records, additively in its
episode-ledger entry:

- compared branch names;
- content hash for each compared decode at dispatch time;
- winner (or null);
- winner content hash;
- per-branch verdict/rationale already returned by the episode.

A compare result is fresh only while every referenced branch hash still
matches.

### Late adjudication hint

At <=4 turns remaining, when there is no positive attestation and the generic
fallback shortlist contains multiple materially distinct decode hashes, render
one concise suggestion to compare the top branches before termination. Emit
`late_branch_adjudication_hint` once for that shortlist hash.

This is advisory. Do not launch a hidden compare episode.

### Fallback precedence

Use this deterministic order:

1. fresh positive-attested branch (Slice D);
2. fresh compare winner, if its verdict is not explicit rejection;
3. existing ground-truth-free scalar fallback.

Record the shortlist, internal scores, selected precedence tier, and rationale
in the artifact. Preserve the old v2 helper behavior by implementing v3
selection at the v3 call site or behind optional v3-only inputs.

Do not add family-specific Borg/Latin/transform penalties in M5.1. If a generic
compare cannot distinguish the 0045v candidates, that is scorer/reader evidence
for a later milestone rather than permission to hard-code the fixture.

## Local Test Gate

Before any paid run:

1. all new unit and artifact-regression tests pass;
2. the focused v3/INV/firewall/bake-off suite remains green;
3. full repository suite is zero-regression against the recorded baseline;
4. existing 22-row summary can be refreshed locally with correct verify counts;
5. no prompt, tool result, attestation, compare record, or fallback input contains
   benchmark plaintext or post-hoc accuracy;
6. v2 artifact behavior and v2 fallback selection are unchanged.

## Live Acceptance - Stage 1 (Focused Borg Diagnostic)

Run v3 only, `gpt-5.5`, preflight on, 25 iterations, three replicates each:

- `borg_single_B_borg_0109v`;
- `borg_single_B_borg_0045v`.

Report every run, including verify timing/verdict, reading creation,
repair-application outcome, installed repair branches, compare usage, fallback
precedence, cost, and post-hoc grading.

### 0109v targets

- no `non-plaintext-alphabet character` reading-application failures;
- verify by turn 15 in at least 2/3 runs;
- at least 2/3 runs attempt a machine-actionable reading repair after a
  repairable negative verify;
- mean character accuracy >= 93%;
- mean word accuracy >= 70%;
- every run that obtains a fresh positive attestation declares explicitly or
  via positive-attested fallback within two turns; a run with only negative
  attestations may remain honestly unsolved and must not be forced to satisfy a
  declaration-count target.

### 0045v targets

- mean character accuracy >= 80%;
- no selected result below 60%;
- post-hoc selected-vs-best-generated branch gap <=5 character points in every
  run (grading/report only, never runtime selection);
- any declaration must carry a fresh attestation; an honest unsolved result is
  preferable to a false declaration.

### Cost target

V3's mean cost on the two focused cells remains <=70% of the stored matched v2
mean. If the new reading/repair/compare episodes exceed this, report the cost
source before tuning budgets.

If the accuracy targets fail, stop and inspect artifacts. Do not tune
declaration thresholds to disguise a repair/search failure.

## Live Acceptance - Stage 2 (Default-Switch Evidence)

Passing Stage 1 reopens M6; it does not switch the default by itself.

Before switching:

1. run a complete paired M6 matrix with three replicates per cell, including the
   preflight-off analysis cells;
2. do not mix pre-fix v3 rows into the post-fix decision aggregate;
3. report case-level means/min/max and verification/repair/fallback behavior;
4. apply the original M6 rule: v3 character and meaningful-word accuracy >= v2,
   with >=20% lower cost;
5. keep v2 as default if the complete matrix does not pass both clauses.

## Deliverables

1. corrected bake-off telemetry and local refresh report;
2. additive machine-actionable Reading schema and repair implementation;
3. mid-budget verification and negative-repair guidance/events;
4. positive declaration trigger and attested fallback;
5. fresh compare binding and v3 fallback precedence;
6. artifact-derived regression tests;
7. focused Stage-1 report;
8. only after Stage 1 passes, a separately authorized complete M6 rerun and
   default-switch recommendation.

## Out of Scope

- any v2 behavior change or v2 deletion;
- changing the underlying classical solvers;
- hidden ground-truth-assisted branch choice;
- automatic application of speculative prose, editorial alternatives, or
  unknown-length holes;
- family-specific Borg rules;
- automatic background compare/verify calls not requested by the lead;
- the default switch itself before complete M6 evidence;
- unrelated INV-1 investigator work.

---

## Forensic verification record (Fable, 2026-07-15 — BINDING)

The four load-bearing forensic claims above were independently verified
against the local artifacts before this revision was adopted:

1. **0109v verifies at turns 24/25/24, all reader_accepts=false,
   coherence 2** — CONFIRMED (all three preflight-ON artifacts; the
   noprefl run likewise: turn 24, coherence 2). There were NO positive
   attestations in any 0109v run — the original spec's positive-trigger
   hypothesis alone would have fired on nothing.
2. **`hypothesis_apply_reading` punctuation rejections** — CONFIRMED:
   the exact string `non-plaintext-alphabet character` appears in the
   tool results of 0109v replicates 1 and 2.
3. **0045v replicate 2 fallback mis-pick** — CONFIRMED by post-hoc
   scoring of every stored branch decryption: `main` = 0.689 char was
   available; the fallback declared `main_transform_rank5` = 0.341.
   Replicate 2 ran no verify episode; replicates 1/3 verified negatively
   (turns 11/22, coherence 1), matching the table above.
4. **Telemetry blindness** — CONFIRMED: `build_summary_row` in
   scripts/run_v3_bakeoff.py contains zero references to the top-level
   `artifact["attestations"]` list; verify fields derive only from
   `solution.attestation`, so unattached (incl. all negative) verifies
   are invisible in the current 22-row summary.

Small amendments (binding, none structural):

A. **Suite baseline**: the local test gate's zero-regression baseline is
   1386 passed / 1 skipped at main `bf63b14`.
B. **Slice A semantics shift**: repointing `verify_*` columns from
   `solution.attestation` to the attestations list changes the meaning
   of existing columns for old rows — the refresh output must carry a
   `telemetry_version: 2` marker so old and new summaries are not mixed
   silently.
C. **Slice B fixtures**: derive the two 0109v failure-shape fixtures
   from the stored artifacts' Reading fragments only (tool inputs), never
   from benchmark plaintext — same firewall as all fixtures.
D. **Orchestration**: standard pipeline applies (Opus coder from this
   spec in a worktree, Fable review, fixes, land, then Stage-1
   acceptance as a separately-launched step). Slices A+B may land
   together; C+D+E together; telemetry refresh report accompanies the
   first landing.
