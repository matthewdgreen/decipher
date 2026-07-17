# Agent Loop v3 M5.3a - Candidate Retention and Sparse-Null Reliability

Status: PLANNED. This is the next bounded v3 follow-up after M5.3. It does not
reopen M5.3 worker budgets, the repair mechanism, M5.4, or the M6 bake-off.

Motivating evidence:

- `docs/reports/m5_3_targeted_smokes_2026_07_17.md`
- `docs/reports/m5_3_borg_candidate_selection_2026_07_17.md`
- Borg artifact `8d5bce9769b1`, which generated a boundary-corrected
  96.5% character / 82.1% word null-mask candidate
- Borg artifact `ac129831aebc`, which topped out at 91.0% / 66.7% and never
  attempted the successful sparse-null configuration

Ground truth in the percentages above is post-hoc diagnostic evidence only.
It must not enter candidate generation, routing, retention, comparison,
verification, declaration, or retry decisions.

## 1. Objective

Fix two tightly related failures before spending on another agent run:

1. v3 can generate a strong partial candidate and then discard it because a
   compare worker uses `winner=null` to mean both "no candidate is solved" and
   "no candidate is preferable."
2. v3 can fail to attempt a sparse-null refinement even after a strong
   substitution basin has persistent distributed damage.

The intended end state is deliberately modest:

- the best available partial and solved acceptance are separate facts;
- a small, diverse finalist portfolio survives scalar/workflow changes;
- changed promising finalists receive fresh-verification priority;
- sparse-null coverage debt is explicit and paid by a real completed search;
- the exact Borg search envelope has measured generation and retention rates;
- matched synthetic controls quantify useful routing and false detours; and
- only then may one separately approved paid Borg smoke run.

## 2. Non-Negotiable Constraints

1. **Ground-truth firewall.** Runtime logic sees no benchmark plaintext,
   alignment, accuracy, or solution key. Post-hoc reports may grade completed
   candidate sets.
2. **No declaration weakening.** A compare result, high scalar, protected
   finalist, or strong post-hoc score never substitutes for a fresh positive
   independent-reader attestation.
3. **No Borg or Latin special case.** Routing is based on cipher structure,
   solver evidence, verifier damage reports, and completed coverage.
4. **No broader budgets.** Keep current lead, reading, repair, and search
   budgets unless a local test proves a separate mechanical defect.
5. **No new general language scorer.** Existing ground-truth-free signals are
   enough to test retention and coverage behavior.
6. **No paid model call before Slices 1-5 pass.** The paid smoke is a separate
   user-approved gate, not part of implementation verification.
7. **No M6 or Stage-1 packet.** This milestone ends after at most one approved
   Borg smoke and its diagnosis.

## 3. Slice 1 - Split Best Candidate from Solved Acceptance

### 3.1 Compare contract

Replace the overloaded compare `winner` concept with two independent fields:

- `best_candidate: string | null`: the strongest supported partial among the
  compared branches, even when none is a solution;
- `accepts_as_solution: bool`: whether the compare worker believes that best
  candidate is solution-grade. This is advisory and does not satisfy the
  declaration gate.

The result still carries a complete ranking and per-branch verdicts. New
compare workers must return both fields. Legacy artifacts with only `winner`
remain readable through an explicit compatibility adapter; new artifacts must
not write the old ambiguous contract.

### 3.2 Hash binding and fallback

Bind `best_candidate` to the hashes of the complete compared shortlist.
`_fresh_compare_winner` should become a semantically named helper such as
`_fresh_compare_best_candidate` and may select a non-rejected best partial for
honest-unsolved fallback. It must not create a solved declaration.

If `accepts_as_solution=true`, the candidate still requires a fresh positive
verify attestation on the same content hash. If `best_candidate=null`, the
comparison must explain why no branch was rankable.

### 3.3 Acceptance tests

- A compare can return branch A as `best_candidate` while
  `accepts_as_solution=false`; fallback retains A.
- A null best candidate does not erase an earlier hash-fresh supported partial.
- A stale comparison is ignored after any compared branch content changes.
- A rejected/invalid verdict cannot become the best candidate.
- Compare acceptance alone cannot pass `meta_declare_solution`.
- Artifact and analyzer output show ranking, best candidate, solved acceptance,
  and hash freshness separately.

## 4. Slice 2 - Protected Finalist Portfolio and Verification Priority

### 4.1 Portfolio

Add a bounded durable finalist portfolio, deduped by candidate content hash.
It should normally contain at most six entries and preserve distinct roles:

- current scalar-best candidate;
- newest hash-fresh compare best candidate;
- best fresh positively verified candidate, if any;
- strongest candidate from each materially distinct solver family/refinement,
  including null-mask search;
- newest installed repair candidate when materially distinct.

Each entry records branch, content hash, source, family/refinement, creation
turn, solver-derived score summary, verification freshness, and protection
rationale. Protection means the candidate remains available for comparison,
verification, fallback, resume, and artifact analysis. It does not mean that
the branch becomes workflow focus or wins automatically.

Eviction is deterministic: stale or rejected content first, then exact hash
duplicates, then redundant same-family entries. A distinct supported finalist
must not be evicted merely because its aggregate scalar ranks third.

### 4.2 Fresh-verification priority

When a compare best candidate or protected family finalist is changed or has
never been verified at its current hash, status/context should identify that
candidate as the next verification priority before another repair on an
already-rejected hash. Verification remains an explicit paid operation in live
runs and a scripted worker in local tests; this slice changes priority, not
verification budgets.

### 4.3 Acceptance tests

- Reproduce the `8d5bce9769b1` ordering where a null-mask candidate ranks below
  two scalar candidates; the distinct null-mask finalist remains protected.
- Correcting rendering changes the content hash, invalidates the historical
  rejection, and creates fresh-verification debt.
- A protected candidate survives workflow focus changes and resume.
- Portfolio order and eviction are deterministic.
- The fallback hierarchy is fresh positive verify, fresh compare best partial,
  then scalar fallback, with all tiers visible in the artifact.

## 5. Slice 3 - Sparse-Null Coverage Debt and Routing

### 5.1 Why the existing heuristic is insufficient

The v2 structural hint primarily looks for an overcomplete cipher alphabet.
Borg 0109v has 23 cipher symbols, so a rule requiring more cipher symbols than
plaintext letters cannot discover its sparse-null case. Language and record id
must not be used as substitutes.

### 5.2 Debt evidence

Create `sparse_null_coverage_debt` for an active simple/homophonic substitution
hypothesis when all of the following hold:

1. a credible substitution basin exists under solver-native signals;
2. a fresh independent reading rejects solution acceptance and reports
   distributed/basin-wide residual damage or recommends broader search, or
   focused repair has saturated without resolving distributed anomalies;
3. no completed sparse-null/null-mask experiment is bound to this hypothesis
   and source content hash; and
4. the candidate is not already explained by an active incompatible family
   with stronger measured evidence.

Overcomplete alphabet, coarse boundaries, and word-island shape remain useful
supporting signals, but none is required alone and target language contributes
no evidence.

### 5.3 Routing behavior

When debt is present:

- expose the exact debt, evidence, source hash, and bounded experiment config;
- prioritize a screen/full null-mask experiment before unrelated transform or
  polyalphabetic exploration;
- do not mark debt paid on submission, failure, invalid config, or an empty
  finalist result;
- mark it paid only after a completed result is collected and its finalist
  portfolio is installed/reviewable; and
- permit an explicit, recorded override when contradictory evidence makes the
  null hypothesis unreasonable.

This is coverage discipline, not an assertion that nulls exist.

### 5.4 Acceptance tests

- A scripted Borg-shaped strong substitution basin plus a negative
  basin-wide/broaden verifier result creates debt.
- Debt routes to the typed null-mask experiment before off-family work.
- A completed collected null-mask result pays debt; a failed or orphaned job
  does not.
- German/Latin language alone never creates debt.
- A clean no-null substitution with a positive verification creates no debt.
- An explicit override is artifacted with rationale and does not erase the
  underlying evidence trail.

## 6. Slice 4 - Repeated No-LLM Borg Replays

Add a focused runner and report that freeze the successful artifact's search
envelope. The canonical replay records at least:

- case `borg_single_B_borg_0109v` and language `la`;
- `cipher_system=homophonic_substitution`;
- `homophonic_solver=zenith_native`;
- outer `homophonic_budget=full`;
- `homophonic_refinement=null_masks`;
- `transform_search=off`;
- Rust null-mask batch, `profile=wide`, `budget=screen`;
- candidate limit 48, max mask size 3, max masks 1500;
- current Latin model SHA-256
  `b2b92f631982e5dbabf0946ed8aa59268a31f8b08568ec689836953e2c73890c`;
- all effective environment/profile values, thread count, and seed schedule.

Run two local phases:

1. **Exact reproducibility:** at least three runs with the artifact's original
   seed schedule and configuration.
2. **Basin frequency:** at least twenty runs with pre-recorded independent seed
   blocks while holding every non-seed setting fixed.

The report distinguishes:

- strongest candidate generated;
- candidate selected by each ground-truth-free ranker;
- candidates retained in the protected portfolio;
- the declared/fallback candidate, if the scripted host path is replayed;
- post-hoc char/word accuracy for grading only; and
- elapsed time and seed/model/config fingerprints.

The primary questions are measured rather than assumed:

1. How often does the strong sparse-null basin exist in generated candidates?
2. Conditional on its existence, how often do rankers and the portfolio retain
   it?
3. Does host comparison/fallback preserve the strongest supported partial?

If exact-seed runs disagree unexpectedly, stop and diagnose nondeterminism
before interpreting basin frequency.

## 7. Slice 5 - Synthetic Null/No-Null Controls

Generate opaque-id matched controls from non-famous plaintexts. At minimum:

- sparse-null substitution analogs at several low insertion rates and with one
  or multiple null symbols;
- matched no-null ciphers using the same plaintext, substitution key, length,
  boundaries, and search budget;
- no-null homophonic analogs with a larger symbol inventory, so
  "overcomplete" is not accidentally treated as proof of nulls; and
- damaged/transcription analogs without nulls, to test whether generic
  residual damage produces excessive false debt.

Use multiple deterministic seeds per condition. The runtime route and solver
remain ground-truth blind; a separate report grades:

- debt true-positive and false-positive rates;
- null-mask generation and protected-retention rates;
- final candidate quality relative to the matched baseline;
- wasted null-mask runtime on negative controls; and
- whether any baseline strong candidate was lost.

Pre-register thresholds in the runner config before looking at aggregate
ground-truth results. Initial engineering gates:

- all expected debt transitions and completed-result semantics pass scripted
  tests;
- no-null controls are not blocked indefinitely by null-mask policy;
- protected retention is at least 95% conditional on a materially stronger
  candidate having been generated; and
- any observed false-routing rate is reported, not hidden by changing the
  synthetic generator after results are visible.

Thresholds for true-positive/false-positive rates should be finalized after a
small generator dry run establishes that the cases are neither trivial nor
impossible. That dry run is for case calibration only and may not tune routing
against individual plaintext answers.

## 8. Gate 6 - One Targeted Paid Borg Smoke

Only after Slices 1-5 pass, request explicit user approval for one run:

- provider/model: OpenAI `gpt-5.5`;
- case: `borg_single_B_borg_0109v`;
- loop: v3 only;
- automated preflight: on;
- maximum iterations: 25;
- target cost: below $3;
- hard per-run cutoff: $5.

In addition to the existing M5.3 control checks, acceptance requires:

1. sparse-null coverage debt is surfaced after the qualifying negative
   evidence and is either paid or explicitly overridden;
2. no unrelated expensive family search jumps ahead of unresolved debt without
   recorded contradictory evidence;
3. compare records `best_candidate` independently of solved acceptance;
4. a materially distinct null-mask finalist, if generated, remains protected;
5. the highest-priority unverified protected candidate receives fresh
   verification before termination;
6. declaration remains blocked without fresh positive verification;
7. character accuracy is at least 93% and word accuracy at least 70%; and
8. artifact/analyzer output agrees on generation, portfolio retention,
   comparison, verification freshness, selected branch, and cost.

Honest unsolved remains acceptable for declaration integrity, but missing the
accuracy or coverage targets means the follow-up has not cleared the Stage-1
blocker. Stop and diagnose the single artifact. Do not launch a replicate,
Stage-1 packet, M6 bake-off, or larger paid run without another user decision.

## 9. Implementation and Commit Order

Land each slice separately with focused tests and a commit checkpoint:

1. compare contract split;
2. finalist portfolio and verification priority;
3. sparse-null debt and routing;
4. Borg no-LLM replay runner/report;
5. synthetic controls and report;
6. separately approved paid smoke report only.

Slices 1-3 change host behavior and require scripted v3 tests plus artifact
analyzer coverage. Slices 4-5 are local evaluation infrastructure and must not
modify runtime policy based on ground-truth results. The paid gate is evidence,
not an implementation slice.
