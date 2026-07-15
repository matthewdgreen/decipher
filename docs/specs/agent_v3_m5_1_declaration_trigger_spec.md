# Spec: Agent Loop v3 — M5.1 (Declaration Trigger / Run-to-Fallback Fix)

Spec author: Fable (2026-07-15). Status: SPEC READY FOR IMPLEMENTATION —
self-contained; written so a fresh session can execute it without the
originating conversation. Follows the standard pipeline: Opus coder from
this spec (worktree), Fable review, fixes, land, acceptance.

## Why (the problem, with the M6 evidence inline)

The M6 bake-off (2026-07-15, `artifacts/m6_bakeoff/summary.jsonl`, 22
rows, gitignored/local — key numbers reproduced here for durability)
returned: **do not switch the default to v3 yet**. v3 is ~60% cheaper
than v2 on every cell ($1.94 vs $4.80 mean per preflight-ON run) and
ties v2 on aggregate char accuracy (86.7% vs 86.8%; v3 WINS copiale_p017
75.4% vs 54.6%; synthetics even at ~100%). Its ONE deficit: Borg.

| cell (gpt-5.5, 25 iters) | v2 | v3 |
|---|---|---|
| borg_0109v ON char (n=3) | 96.1% (solved x3) | 91.0% (fallback x3) |
| borg_0109v ON word | 75.4% | 66.7% |
| borg_0109v OFF char (n=1) | 96.8% (solved) | 91.0% (unsolved) |
| borg_0045v ON char (n=3) | 83.1% (unsolved x3) | 67.1% (fallback x2, unsolved x1) |

**Every v3 Borg run ended `fallback_declared`/`unsolved`; every v2
borg_0109v run explicitly solved.** On synthetics (where preflight hands
the lead a near-perfect decode) v3 declares fine. The M5 acceptance run
showed the mechanism: the lead deferred its `verify` episode to the
final turn (20/20), got its attestation with zero turns left to act, and
fell back. M5's gate (declaration requires a fresh attestation) is
working as designed; what's missing is the POSITIVE half — nothing
converts a good reading + cheap verify into a timely declaration, and
nothing pushes the lead to verify early.

CAUTION (honesty about scope): the Borg char gap (91 vs 96) may not be
pure timidity — v3's exploration may genuinely converge less far on Borg
in 25 turns. Part 0 settles this before the fix is tuned.

## Part 0 — Forensics (LOCAL, no LLM; do this first)

Read the six v3 Borg artifacts under `artifacts/m6_bakeoff/v3/borg_*/`
(and the M5 acceptance artifact if present). For each, extract and
report:
1. Best branch by internal signals at end vs the fallback-picked branch
   — did the fallback pick the lead's true best? (If not, fix #4 gains
   weight.)
2. Was a `verify` episode run; at what turn (vs max 25); verdict.
3. Turn timeline shape: turns spent observing vs acting vs episodes;
   did the lead have a >90%-char branch by mid-run (compare
   per-turn `workspace_snapshot` events if present — note: pre-CLI-slice
   artifacts may lack them; fall back to branch snapshots).
4. Did the F9 late-turn hint fire (loop_events, post-M6-Part-A runs
   only) and did the lead act on it?
Write the findings into the implementation report; they calibrate
thresholds below and are the baseline for acceptance.

## The fix (four small, additive pieces — no coercion cascade)

Surfaces: `src/investigation/context.py` (brief + hint),
`src/investigation/loop_v3.py` (fallback + hint emission),
`src/agent/loop_shared.py` / `tools_v2.py` only if #4 needs the shared
best-branch helper extended. v2 untouched. All thresholds named
constants.

1. **Brief: verify-by-mid-budget + declare-on-positive.** Edit the v3
   lead brief (context.py): (a) instruct running `verify` on the current
   best branch BY MID-BUDGET ("verify is cheap; run it as soon as a
   branch reads mostly as <language> — do not save it for the end");
   (b) add the positive-signal rule: "a verify with reader_accepts=true
   and coherence >= 7 is your signal to declare that branch NOW unless
   you have concrete evidence of a better branch in progress." Keep it
   to ~3 lines; this is guidance, not a gate.
2. **Late-turn hint: earlier + positive-aware.** The existing F9 hint
   (fires at turns-remaining <= 2 when the best branch lacks a fresh
   attestation) becomes two-stage: (a) at turns-remaining <=
   LATE_VERIFY_TURNS (default 4): if NO fresh attestation on the best
   branch — current behavior, earlier; (b) at ANY turn: if a fresh
   POSITIVE attestation (reader_accepts=true, coherence >=
   DECLARE_COHERENCE, default 7) exists and the lead has not declared
   within POST_ATTEST_PATIENCE turns (default 2) of receiving it, the
   context appends one line naming the branch: "branch X carries a
   positive reader attestation — declare it or state why not." Loop
   emits a LoopEvent both times (extend the M6-Part-A event). Render-only
   in context.py; predicate shared with the loop (the
   late_turn_attestation_target pattern from M6 Part A).
3. **Attested fallback promotion (the structural piece).** At the
   exhaustion/error fallback (loop_v3.py, `_best_branch_for_auto_declare`
   call site): BEFORE the generic best-branch pick, check whether any
   branch carries a FRESH (content-hash-matching, recomputed via
   `_decoded_text_for_panel`) attestation with reader_accepts=true and
   coherence >= DECLARE_COHERENCE. If yes, declare THAT branch instead:
   `SolutionDeclaration` with the attestation attached,
   `auto_declared=True`, a rationale naming the attestation, and
   `self_confidence` mapped from coherence (coherence/10). Artifact
   status: keep `fallback_declared` PLUS a new additive boolean
   `attested_fallback=True` — do NOT relabel as "solved" (the lead did
   not declare; honesty in reporting outweighs the optics). Scoring is
   unchanged (the scorer reads the decryption, not the status).
4. **Fallback tie-break prefers attested branches.** In the generic
   fallback path (no positive attestation), if the top branches are
   within FALLBACK_TIE_EPS (default 0.02 dict-rate) of each other and
   one carries any fresh attestation (even weak), pick it — the reader
   signal breaks internal-score ties. Implement in/next to
   `_best_branch_for_auto_declare` WITHOUT changing v2 (pass the
   attestation list in from the v3 call site only; v2 call sites pass
   none and behave identically — pin with a test).

## Tests

- Brief: the three lines render (and only for v3).
- Hint stage (a) fires at <=4; stage (b) fires only after
  POST_ATTEST_PATIENCE turns with a positive attestation and names the
  branch; neither fires in v2; LoopEvents recorded.
- Attested fallback: scripted run — lead gets a positive attestation,
  never declares, exhausts → the attested branch is declared with
  attestation attached + attested_fallback=True; a stale (hash-mismatch)
  or negative (reader_accepts=false / coherence<7) attestation does NOT
  promote (falls through to generic).
- Tie-break: attested branch wins within eps; outside eps the better
  branch wins regardless; v2 path byte-identical (explicit test).
- Suite zero-regression (baseline 1386 passed / 1 skipped @ 98be6c9).

## Acceptance (live, ~$10-12, needs OpenAI credit)

Re-run the two decisive bake-off cells with the fix, n=3 each, via the
EXISTING runner (`scripts/run_v3_bakeoff.py` machinery or direct
BenchmarkRunnerV2 calls, gpt-5.5, 25 iters, preflight ON):
- v3 borg_single_B_borg_0109v x3 — targets: explicit or attested
  declaration >= 2/3 (was 0/3); char >= 93% mean (was 91.0); word >= 70%
  mean (was 66.7).
- v3 borg_single_B_borg_0045v x3 — directional: any declaration; char >
  67.1% mean.
Compare against the stored bake-off rows (they remain in
`artifacts/m6_bakeoff/summary.jsonl`); report per-run numbers. If
targets are met, propose re-opening the M6 switch decision (a focused
v2-vs-v3 re-run on the Borg cells only, ~$25, rather than the full
matrix). If Part 0 shows the gap is exploration (not declaration), stop
after the forensics report and bring the evidence back instead of
tuning blindly.

## Out of scope

Any v2 change; new gates/coercion; multi-branch verify orchestration;
the M6 default switch itself (separate decision after acceptance);
Fable-refusal framing work.
