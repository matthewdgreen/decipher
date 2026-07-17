# Agent Loop v3 M5.3 - Control and Repair Reliability

Status: proposed 2026-07-16. No implementation has landed. This milestone is
the targeted follow-on to M5.2; it must complete before another paid acceptance
packet or a full M6 bake-off.

**Revision 3 amendment, 2026-07-16.** The reliability clarifications and
forward-compatible seams in `docs/repair_reframe_m53_comments.md` are now
normative for this milestone. The sections below incorporate the executable
requirements directly. The future `InterpretationPacket`, deterministic
compiler, annotation ledger, and recovered-reading terminal remain deferred to
M5.4 and gated by `docs/repair_mechanism_rethink.md`; M5.3 must preserve those
seams without implementing them prematurely.

## Motivation

M5.2 established the intended strategist/worker architecture:

- the lead verified the automated preflight on turn 1;
- a negative verification produced concrete reading and repair work;
- repair transactions isolated edits and installed only explicit winners;
- changed candidates required fresh verification; and
- the run ended honestly `unsolved` rather than declaring damaged Latin solved.

The first paid M5.2 smoke also showed that the host still gives the model too
much freedom inside that architecture. A reading nominally configured for four
calls could execute a larger batched tool response before the budget check, so
the smoke did not test whether four calls can produce a useful reading. The
lead could request a new reading and repair transaction repeatedly against
essentially unchanged content. Each `hypothesis_test_word` rebuilt an
expensive menu. An otherwise reasonable attempt to escape into a fresh
automated search failed because the model-facing experiment schema did not
expose the actual configuration contract. The run therefore spent 25 turns
and $4.01 circling a useful but damaged basin.

M5.3 is a host-control milestone, not a prompt-tuning milestone. The host must
enforce bounded work, recognize saturation, make the efficient operation the
obvious operation, and preserve honest termination.

## Prior Results

### M6 bake-off baseline

The stored M6 bake-off contains 22 completed rows. On the three preflight-on
`borg_single_B_borg_0109v` replicates:

| Loop | Mean char | Mean word | Mean cost | Outcome pattern |
|---|---:|---:|---:|---|
| v2 | 96.1% | 75.4% | $3.31 | 3/3 solved |
| v3 | 91.0% | 66.7% | $2.17 | 3/3 unattested fallback declarations |

V3 was cheaper but selected the same preflight basin in all three replicates
and did not verify or repair it successfully. On `borg_0045v`, v3 was also
cheaper but unstable: two runs retained the 83.6% basin while one selected a
34.1% result.

The synthetic controls were encouraging. With preflight enabled, v3 solved
`synth_en_250nb_s4` exactly in two turns for $0.17 and solved
`synth_en_200honb_s6` at 99.9% in two turns for $0.20. The weakness was not the
basic architecture; it was recovery when the first candidate was strong but
imperfect.

### M5.1 focused acceptance

M5.1 added earlier verification and honest declaration behavior. Its three
preflight-on `borg_0109v` replicates all verified on turn 13, but all retained
the 91.0%/66.7% basin. Costs were $2.74, $2.59, and $3.21. Two completed
`borg_0045v` replicates retained 83.6% character accuracy and cost $3.39 and
$3.31. The six-run packet stopped before the final replicate under its budget
ceiling.

This proved that declaration timing was not the main accuracy limitation. V3
needed a reliable path from negative verification to either a better candidate
or a timely decision to broaden.

M5.1 forensics also found a distinct reading-worker exhaustion mode. With the
original twelve-call budget, `gpt-5.5` commonly spent ten or eleven calls
exploring and then failed to submit a structured reading. Commit `1d01ef4`
raised the reading envelope to 16 calls, 8,192 output tokens, and 300 seconds,
and added an explicit instruction to submit a useful partial reading early.
M5.2 later reduced the registered reading budget to four calls, 4,096 output
tokens, and 180 seconds. Because the batch-check bug allowed four-call readings
to execute as many as thirteen calls, the M5.2 smoke is not evidence that this
smaller production budget is usable.

### M5.2 targeted smoke

Artifact:
`artifacts/m5_2_targeted_smoke_20260716/v3/borg_single_B_borg_0109v/1/borg_single_B_borg_0109v/d3eccab14a40.json`

Configuration: OpenAI `gpt-5.5`, v3, automated preflight enabled, 25 turns,
one replicate, $5 launch ceiling.

| Metric | Result |
|---|---:|
| status | honestly unsolved |
| runtime | 1,644.6 seconds |
| actual cost | $4.0089 |
| preflight char / word | 89.6% / 62.8% |
| selected char / word | 91.3% / 67.9% |
| verification | first ran on turn 1; no positive attestation |
| episode count | 20 |
| lead `episode_run` calls | 12 |
| repair transactions | 8 |
| `hypothesis_test_word` calls | 35 |
| cumulative `hypothesis_test_word` time | 456.3 seconds |

The run made real progress over preflight and produced a useful semantic
summary about treatment of chicks, mortality, night-time work, and a painless
care procedure. It nevertheless fell short of the focused 93% character / 70%
word target and spent most of the extra budget revisiting the same basin.

The smoke exposed these concrete failures:

1. Episode tool-call limits were checked after a complete model-emitted batch.
   Readings configured for four calls executed up to thirteen calls; repairs
   executed up to fifteen. This proves an enforcement defect, not the
   sufficiency of a four-call reading budget.
2. The model could raise `max_tool_calls` above the host default.
3. Rewording a reading or repair goal allowed effectively duplicate work on
   unchanged candidate content and unchanged verifier evidence.
4. Each singleton word hypothesis rebuilt the complete repair menu.
5. The lead correctly attempted an alternate automated search on turn 4, but
   guessed unsupported config keys (`target_language`, `allow_homophones`, and
   `max_runtime_seconds`) because the tool schema exposed only an unstructured
   `config` object.
6. The host validated that a repair worker named a changed branch, but did not
   independently enforce the full collateral-evidence acceptance contract.
7. Workspace telemetry displayed an internally best-scored branch as
   `branch`, which obscured the distinct workflow-focused and latest-installed
   branches.
8. `inspect_artifact.py` did not parse the v3 top-level shape correctly: model
   provider, iteration count, declaration state, and final branch appeared
   unknown or false even though the detailed episode sections were present.

## Diplomatic-Text Complication

The post-hoc alignment revealed a separate evaluation issue. The benchmark
plaintext is corrected/diplomatic Latin with editorial insertions and lacunae,
for example:

```text
cur[a ap] plicare [5, 4] uel [6] pullo[s]
hi[nc] pro certo eger libe[ra] bitur
```

The solver-facing candidate is a flat decoded token stream. Editorial brackets
and omissions do not survive that representation cleanly, so even exact or
near-exact spans can look like broken continuous Latin. The verifier was right
not to declare the candidate solved, but a single `coherence` value conflated
four questions:

- Is the target language recognizable?
- Is the broad meaning recoverable?
- Is the damage local or basin-wide?
- Is the candidate complete enough to declare solved?

Runtime must not receive benchmark plaintext or post-hoc alignment. The general
fix is to separate these judgments in the verifier contract, not to expose the
answer or add Borg-specific prompting.

## Design Principles

1. Ground truth remains post-hoc grading only. It must not affect routing,
   repair, scoring, retries, verification, or declaration.
2. Host policy, not model obedience, enforces budgets and workflow transitions.
3. Content hashes and evidence hashes define whether work is new. Reworded
   goals do not create new evidence.
4. Expensive common setup is built once and reused across candidate probes.
5. Verification distinguishes a promising damaged basin from a solved text.
6. A failed repair cycle creates pressure to broaden or stop, not to restate
   the same reading indefinitely.
7. Paid tests remain single-cell and explicitly capped until local acceptance
   is complete.
8. A worker budget may be reduced only after a binding-cap test shows that the
   worker still returns a usable result. Lower cost or earlier transition alone
   is not evidence of usability.

## Slice 1 - Hard Episode and Cost Budgets

### Episode call enforcement

Enforce the tool-call cap before each ordinary or composite call. If a model
emits more calls than remain in the budget:

- execute only the calls that fit;
- synthesize a paired `budget_exhausted` result for every skipped tool use;
- still accept a valid `episode_submit_result`; and
- make the overshoot visible in the episode ledger and analyzer.

The model-facing `max_tool_calls` argument may lower the registered host budget
but may never raise it. Clamp it to `1..default_max_tool_calls`, or remove it
from the model-facing schema and select overrides only in host code.

### Reading budget and submission reserve

Do not make the current four-call reading budget the production default while
fixing enforcement. Restore the M5.1 reading envelope provisionally:

- at most 16 exploratory tool calls;
- up to 8,192 output tokens for structured submission; and
- a 300-second wall-clock ceiling.

Treat these as a maximum safety envelope, not a target. The early-partial-
reading contract remains in force, duplicate/saturation policy limits how many
readings may run for unchanged evidence, and the per-run cost ceiling remains
the final paid guard.

This envelope applies only to the legacy M5.3 prose-reading episode. It is not
a default for the deferred M5.4 `InterpretationPacket` producer, whose design
allows one tool-free structured response by default and at most one batched
lookup plus one final response.

Exploration and submission must have separate accounting. When exploration
ends or reaches its cap, preserve one submit-only completion attempt with the
full structured-output allowance. Exploratory calls may not consume this
submission reserve. The submit-only phase receives the candidate packet and
the worker's accumulated notes, exposes only `episode_submit_result`, and may
return either a useful partial reading or an explicit `no_safe_reading` result.

Four calls remains a useful enforcement fixture and a candidate future budget,
but it must not become the default until a binding 4/8/12/16 calibration shows
that reading quality is preserved. The batch/cache work in Slice 3 improves
repair episodes; it does not justify reducing reading budgets because reading
workers do not run word-hypothesis repair menus.

### Per-run paid ceiling

Add a v3 `max_cost_usd` option enforced before **every** paid provider send:
lead turns and worker sends, including mid-episode continuation and submit-only
sends. Once reached, the active episode ends with an explicit budget-class
result and the loop must make no further paid call. It should select the best
supported branch and terminate honestly, preserving `unsolved` when no
positive attestation exists. This is distinct from the bake-off runner's
matrix-level launch guard.

### Local acceptance

- A response containing ten ordinary calls under a four-call budget executes
  exactly four.
- Every emitted tool use receives exactly one tool result.
- A requested budget of twenty on a sixteen-call reading remains sixteen.
- A submit tool in an over-budget batch can still finish the episode.
- Exhausting exploratory calls still leaves one submit-only attempt with an
  8,192-token output allowance.
- A cost ceiling prevents the next paid call and produces a complete artifact.
- A ceiling reached between two worker sends marks the episode
  budget-terminated and prevents all subsequent lead and worker sends.

### Reading-usability acceptance

Budget enforcement is not sufficient by itself. A reading counts as usable
only when all of the following are true:

- status is `ok`, not `budget_exhausted` or an empty-output fallback;
- `reading_text` or semantic gloss is non-empty and bound to the exact
  candidate content hash;
- cited fragments use valid stable span ids or explicitly identify an
  unresolved hole;
- the result contains at least one evidence-linked anomaly, actionable repair
  clue, or a reasoned `no_safe_repair` conclusion; and
- a repair transaction can consume the result without schema failure,
  `fresh_reading_required`, or a no-op caused by a malformed reading packet.

Local scripted tests prove shape, binding, and fallback behavior. They cannot
establish model reading quality. The first paid smoke must therefore report
both actual exploratory-call use and the usability fields above before the
default reading budget can be reduced.

The paid-smoke report must also include, per reading, requested versus executed
calls and the report-only fraction of usable readings containing at least one
non-empty `repair_text` fragment. This statistic does not gate acceptance, but
it distinguishes readings useful only for interpretation from compile-ready
legacy readings.

## Slice 2 - Repair Saturation and Workflow Escape

Track repair-cycle identity with:

- candidate content hash;
- latest verifier-attestation id or anomaly digest;
- `interpretation_id` and `interpretation_digest` (M5.3 values are the legacy
  Reading id/digest; generic names avoid a later state-format split);
- generated finalist content hashes; and
- transaction outcome and failure reason.

For one unchanged candidate and one unchanged attestation, permit by default:

- one fresh reading;
- up to two distinct repair transactions; and
- no repeated evidence evaluation with the same source/interpretation pair.

Classify failures before consuming saturation budget:

- **process failure**: evidence was not adjudicated, including
  `no_winner_named_with_multiple_changed_finalists`, `worker_did_not_apply`,
  unsupported/fabricated winner names, episode runner errors, and schema
  failures. Permit exactly one linked retry (`retry_of`), which does not count
  toward saturation. A second process failure for the pair counts as evidence
  failure;
- **evidence failure**: the pair was evaluated and did not support installation,
  including `unsupported`, `no_changed_finalists`, `all_finalists_rejected`,
  no-op, and materially non-improving adjudication. It counts toward saturation
  and the pair cannot be rerun.

Do not emit a combined `ambiguous_or_unchanged_finalists` reason; it erases the
process/evidence distinction required by this policy.

After two **evidence-failed** repair transactions (no-op, unsupported, all
finalists rejected, or materially non-improving) for the same candidate and
verifier evidence, enter a durable `repair_exhausted` workflow state. Process
failures advance this counter only through the second-process-failure rule
above. Its menu is:

1. run one alternate search/basin experiment;
2. compare genuinely distinct existing finalists; or
3. declare honestly unsolved.

A new candidate hash or genuinely new verifier evidence resets saturation. A
newly worded goal does not. Resume must preserve the counters.

`repair_exhausted` must be explicit in the episode-kind phase map and exclude
reading and repair. Its allowed episode kinds are `search`, `compare`, and
`verify`. Unknown workflow phases fail closed to `verify` (or raise in tests)
with a warning rather than defaulting to all kinds. While an alternate
experiment is pending, the state remains `repair_exhausted` and offers collect;
it resets only on a new candidate hash or genuinely new verifier evidence.

### Local acceptance

- Repeated readings on unchanged content and attestation are suppressed.
- An evidence-failed source/interpretation pair cannot be rerun under a new
  `as_name`; one process-failure retry can succeed and install.
- Two distinct evidence-failed repairs move the state to `repair_exhausted`.
- New changed content returns to `candidate_reading` and requires verification.
- Serialization/resume preserves the same next action.
- With an experiment pending, allowed kinds exclude reading/repair and the menu
  names the pending experiment.

## Slice 3 - Batched and Cached Word Hypotheses

Add a composite `hypothesis_test_words` operation that accepts a bounded list
of word/span hypotheses. It must:

1. accept only `claim_type="word"` in M5.3; future `boundary`/`op` fields are
   documented as reserved but rejected with `unsupported_reserved_field` until
   implemented, never silently ignored;
2. resolve granular host-owned anchors and reject malformed hypotheses cheaply:
   one `word_id` or a contiguous `{start_token_id, end_token_id}` run inside a
   content-hash-bound window. Existing 120-token window ids alone are too
   coarse; Slice 3 mints these `word_id`/token anchors host-side and exposes
   them in the host-built candidate packet, so models never need raw numeric
   offsets. Raw `word_index`/`char_start` remain legacy singleton-parity inputs
   and are discouraged on model-facing surfaces;
3. return `not_expressible_as_key_edit` as typed data, including span and word
   lengths, when one-character-per-token compilation is impossible;
4. build or retrieve the branch's repair menu once;
5. evaluate menu-backed hypotheses from that shared packet;
6. score only surviving injected hypotheses;
7. deduplicate equivalent edit sets;
8. return a small diverse finalist set with collateral evidence; and
9. optionally install only the explicitly selected forks.

Cache the expensive menu using digests of the exact resolved builder inputs:
base key, null mask, boundary spans, language, resolved repair config,
dictionary path, and language-model path. A proven-sufficient proxy is allowed
only if its sufficiency is documented. The existing singleton
`hypothesis_test_word` becomes a wrapper over the batch core.

This batch tool is a performance substrate for M5.4, not the proposed
deterministic assertion compiler itself. The deferred oracle experiment tests a
small reference implementation of that compiler directly.

Repair episodes should receive the batch operation as the preferred interface.
Once batch parity is proven, reduce the repair episode's ordinary tool-call
budget from twelve to approximately six.

### Performance acceptance

- A batch of eight hypotheses constructs the repair menu exactly once.
- Batched and singleton results agree for the same hypothesis and configuration.
- Cache invalidates on key, mask, boundary, model, or config changes.
- Identically rendered branches with different resolved per-call configs do
  not share a cache entry.
- The replay fixture's cumulative word-hypothesis time falls by at least 70%.

## Slice 4 - Host-Validated Repair Acceptance

The host must not accept a changed snapshot merely because the worker reports
`applied=true`. Before installation, bind the claimed winner to episode tool
evidence and require:

- every claimed edit appears in a successful composite-action result;
- the winner snapshot is one of the recorded changed finalists;
- no unresolved error invalidates a claimed edit;
- the winner was included in final adjudication when multiple finalists exist;
- deterministic collateral limits are satisfied; and
- the acceptance decision and component deltas are stored in the transaction.

Small scalar decreases may be allowed for strong local-language repairs, but
only under an explicit, tested, ground-truth-free policy. The policy must not
be improvised by the worker; absent such a policy, any net scalar decrease
rejects the installation (default deny). Unsupported or ambiguous outcomes
remain review evidence and do not install.

### Local acceptance

- A fabricated best-branch name is rejected.
- A changed fork produced by a failed tool call is rejected.
- An unadjudicated winner among multiple changed finalists is rejected with
  process-failure reason `no_winner_named_with_multiple_changed_finalists`;
  genuinely unchanged or rejected finalist sets use distinct evidence-failure
  reasons.
- A supported singleton with bounded collateral installs and requires fresh
  verification.
- Acceptance records contain enough evidence for artifact review.

## Slice 5 - Typed Experiments and Alternate Search

Replace the opaque model-facing `experiment_submit.config` object with the
registered experiment type's actual schema. For `automated_solver`, expose the
supported keys, enums, defaults, and concise descriptions. Make these facts
explicit:

- language is derived by the host and must not be supplied;
- `cipher_system` is the family hint;
- homophonic behavior is selected through supported solver/refinement fields;
- unsupported runtime controls are not accepted; and
- the lead should collect/install the experiment result rather than submitting
  duplicate jobs.

When a repair cycle saturates, the workflow should recommend one alternate
search experiment with a valid family-consistent configuration. A structured
validation error should return a corrected example and keep the search escape
prominent on the next turn instead of silently returning the workflow to
repair.

### Local acceptance

- Provider-visible schema rejects unknown config keys before dispatch.
- A scripted lead can submit a valid simple-substitution rerun without guessing
  language or unsupported homophonic flags.
- A validation error produces a valid corrected configuration example.
- Saturated repair transitions to alternate search, then compare/verify.

## Slice 6 - Diplomatic Verification Contract

Extend the text-only verifier result with separate fields:

- `target_language_confidence` (0..1);
- `semantic_recoverability` (0..1);
- `damage_scope` (`local`, `distributed`, or `basin_wide`);
- `repairability` (`local_repair`, `broaden`, or `none`);
- `reader_accepts_as_solution` (boolean); and
- a concise gloss, anomaly list, and uncertainty note.

The M5.3 solution verifier still receives only canonical key-derived candidate
plaintext and generic permitted context. It receives no benchmark plaintext,
key, score, alignment, annotated editorial reconstruction, or
post-hoc accuracy. Its instructions should explain generally that historical
decipherments may preserve lacunae, abbreviation scars, uncertain boundaries,
and editorial omissions. It must quote or point to the candidate evidence for
its judgments.

**Explicit policy decision:** this intentionally reverses documented design C6.
A fresh **positive** attestation with `reader_accepts_as_solution=true` is now
required for `meta_declare_solution`; an absent, stale, weak, or negative
attestation cannot satisfy the solved gate. Weak fresh attestations remain
routing evidence. Ship the schema change together with migrations for
`AttestationPolicy`, positive-attestation/fallback tiering, context routing,
repair-agenda seeding, state serialization/resume defaults, analyzer output,
and the verify contract. Legacy serialized attestations load conservatively as
not positive unless their old fields establish the new condition (for legacy
records: positive iff `reader_accepts` was true and `coherence >= 7`, the
prior `_is_positive_attestation` condition). The fallback tier
`fresh_positive_attestation` re-keys on the same new positive condition.

The other fields route work:

- high language confidence + high recoverability + local damage -> one bounded
  repair cycle;
- recognizable language + distributed damage -> compare or alternate search;
- low language confidence or basin-wide damage -> broaden;
- `reader_accepts_as_solution=true` -> declare promptly.

Initial routing thresholds are host constants and calibration defaults,
tunable only with paid-smoke or equivalent targeted evidence: high (and
recognizable) language confidence is `target_language_confidence >= 0.7`;
high recoverability is `semantic_recoverability >= 0.5`. `coherence` remains
in the verifier schema as a clamped 0-10 report-only legacy field during
M5.3; it no longer gates declaration, routing, or fallback-tier selection.

This lets the system say "likely correct basin with recoverable meaning, but
not a complete solution" without either rejecting all useful evidence or
declaring a damaged text solved.

Do not overload this solved gate with future recovered-reading semantics. M5.4
may add `meta_declare_recovered_reading` with separate accounting and a
composite attestation hash over key text, interpretation packet, annotated
reading, and annotation ledger. That action remains out of M5.3.

## Slice 7 - Observability and Analyzer Parity

Workspace snapshots and human output must distinguish:

- `best_scored_branch`;
- `workflow_branch`;
- `latest_installed_branch`; and
- `declared_or_selected_branch` when present.

Update `scripts/inspect_artifact.py` for the v3 artifact shape. The summary
must correctly show provider/model, loop version, iterations, solved/unsolved
declaration, final branch, attestation status, and cost. Add sections for:

- episode budget requested vs executed;
- suppressed over-budget calls;
- repair cycles grouped by content hash;
- saturation transitions;
- installed and rejected repair transactions;
- experiment validation failures;
- workflow branch vs score-selected branch; and
- cumulative time spent rebuilding or evaluating repair hypotheses.

Commit only a **trimmed** M5.2 analyzer regression fixture. Remove benchmark
ground truth, expected plaintext, alignment blocks, and raw model message
bodies; retain only the structural fields the analyzer consumes. Add a test
that rejects fixture keys associated with ground truth or expected answers.

## Verification Sequence

### A. Focused local tests after each slice

Run the episode, workflow, experiment, repair, state/resume, analyzer, and
ground-truth-firewall tests touched by the slice. Do not run a paid model.
Claim and update the pre-existing stale assertion in
`tests/test_lead_context.py::test_negative_partial_attestation_creates_repair_action_menu`
to the Slice-2 menu contract; record that it predated M5.3 rather than leaving
it ambient red or reverting the improved menu wording.

### B. Scripted end-to-end replay

Use a scripted provider and a stored/trimmed version of the M5.2 sequence to
prove:

- verification occurs early;
- one reading feeds a bounded repair transaction;
- reading exploration cannot consume the reserved submit-only attempt;
- repeated no-op repair saturates;
- alternate search is offered and accepts a valid config;
- no worker exceeds its call budget;
- termination is honest; and
- the analyzer reports the full path correctly.

### C. One paid targeted smoke

Only after local acceptance, request user approval for one run:

- provider/model: OpenAI `gpt-5.5`;
- case: `borg_single_B_borg_0109v`;
- loop: v3 only;
- automated preflight: on;
- maximum iterations: 25;
- target cost: below $3;
- hard per-run ceiling: $5.

Paid acceptance targets:

1. verification by turn 3;
2. no episode exceeds its registered call cap;
3. the first negative verification yields a usable reading by turn 5;
4. no reading ends `budget_exhausted`, empty, or malformed;
5. at most two readings and two repair transactions per candidate hash;
6. at least one usable reading feeds a valid repair transaction, or records a
   reasoned `no_safe_repair` conclusion;
7. alternate search after repair saturation;
8. no invalid experiment configuration;
9. no repeated unchanged reading/repair cycle;
10. character accuracy >=93%;
11. word accuracy >=70%;
12. honest `unsolved` is allowed without positive attestation; and
13. artifact/analyzer agree on status, branch, verification, cost, reading
    usability, and actual reading-call consumption.

If the run misses the control or reading-usability targets, stop and inspect
locally. Do not spend on additional replicates. If those targets pass but
accuracy misses, diagnose search/repair quality before changing budgets. Do
not lower the reading default from the provisional 16-call envelope on the
basis of one successful run; first compare the stored reading packet under
binding 4/8/12/16 limits or collect equivalent targeted evidence.

### D. Focused Stage-1 packet

Only after the single paid smoke passes may the prior M5.1 Stage-1 packet be
reopened: `borg_0109v` and `borg_0045v`, three replicates each, v3 only,
preflight on. Obtain separate user approval for that spend.

### E. Full M6 bake-off

The paired M6 rerun and default-loop decision remain explicitly out of scope
until the focused Stage-1 packet passes and the user approves the larger spend.

## Non-Goals

- No ground-truth-guided routing, repair, scoring, retry, or declaration.
- No Borg-specific plaintext hints or verifier rules.
- No larger worker budgets as a substitute for host control.
- No broad prompt expansion.
- No automatic default switch from v2 to v3.
- No full bake-off during this milestone without explicit user approval.

## Deliverables

1. Hard per-call episode budgeting and a true per-run paid ceiling.
2. Durable repair-saturation state and duplicate suppression.
3. Batched/cached word-hypothesis evaluation.
4. Host-validated repair acceptance.
5. Typed experiment schemas and a reliable alternate-search transition.
6. Diplomatic-text-aware, ground-truth-free verification fields.
7. V3-aware workspace telemetry and artifact analysis.
8. Focused local/replay test evidence.
9. Reading-usability telemetry and a justified post-calibration default.
10. One separately approved paid smoke report before any larger evaluation.

---

## Revision 3 Review Rationale

The detailed review history, rejected alternatives, and forward-compatible M5.4
seams are preserved in `docs/repair_reframe_m53_comments.md`. The executable
M5.3 requirements are integrated above; if the companion rationale conflicts
with this specification, this specification controls.
