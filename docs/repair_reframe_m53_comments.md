# M5.3 Consolidated Amendment Block — Reliability Clarifications + Repair-Reframe Seams

Status: proposed 2026-07-16, **revision 3**, pending user review; retained as
the detailed rationale for requirements now integrated directly into
`docs/specs/agent_v3_m5_3_control_reliability_spec.md`. Author:
Fable design synthesis, read-only pass. All file/line references verified
against `main @ 4b85e20`. Sources reconciled here: the M5.3 spec review's
eight required clarifications, GPT-5.6-Sol's design review of the repair
rethink (ten comments), and the revised design note
`docs/repair_mechanism_rethink.md` (revision 3), which carries the full
rationale and the adjudication of each Sol comment.

Revision-3 edits are incorporated directly below: legacy reading budgets are
explicitly nonprecedential for M5.4, failure reasons are split before
saturation, the positive solution-attestation decision is explicit, batch
seams use granular anchors and reject unused future fields, and the deferred
annotation/recovered-reading contracts use separate support levels and
composite hash binding.

Structure:

- **Part A** — amendments to the M5.3 slices as specced; land in M5.3.
- **Part B** — small forward-compatibility seams for the repair reframe;
  land in M5.3 (hours of work, no behavior change beyond what is stated).
- **Part C** — explicitly deferred to a follow-on milestone (working name
  M5.4 "typed repair"); do **not** build in M5.3. Listed so Slice work does
  not accidentally foreclose it.

Priorities: P0 = the spec is wrong or unsafe without it; P1 = required before
the paid smoke is meaningful; P2 = hygiene, land with the relevant slice.

---

## Part A — amendments to M5.3 slices

### A1 (P1 | Slice 1) Reading budget and usability — mostly landed; two residuals

Status: the substance of this clarification was already incorporated into the
spec by commit `4b85e20` (provisional restoration of the 16-call / 8,192-token
/ 300-second envelope, the submit-only reserve, the reading-usability
acceptance section, design principle 8, and the binding 4/8/12/16 calibration
requirement). Note the code today is still at 4/4,096/180
(`src/investigation/episodes.py:299`) — both the call-count and the
output-token reversion are real and Slice 1 restores both.

Residuals to fold in:

1. The paid-smoke report (Verification C) must cite, per reading episode, the
   requested-vs-executed call ledger (Slice 7 already specifies collecting
   it) — i.e. the smoke report is not complete without actual
   exploratory-call consumption numbers next to the usability fields.
2. Add one report-only (not gated) usability statistic: the fraction of
   usable readings that carried at least one non-empty `repair_text`
   fragment. A reading that is `ok` and bound but contains zero compile-ready
   fragments is usable for verification but useless for repair; the
   calibration decision needs to see that rate.

Scope clarification from repair-rethink revision 3: this 16-call envelope is a
temporary safety setting for the **legacy M5.3 prose-reading episode only**. It
must not become the default for M5.4's `InterpretationPacket` producer, which
uses one tool-free structured response by default and at most one batched
lookup plus one final response.

### A2 (P0 | Slice 2 × Slice 4) Distinguish process failure from evidence failure before burning a (source, reading) pair

Slice 2 currently says: "no repeated transaction with the same source/reading
pair, regardless of whether the prior transaction installed a winner or
failed", and two failed repairs move the state to `repair_exhausted`. As
written, this burns saturation budget on failures that never evaluated the
evidence.

Amendment — classify transaction failures:

- **Process failures** — the worker or harness failed before the evidence was
  actually adjudicated: `no_winner_named_with_multiple_changed_finalists`,
  `worker_did_not_apply`, `fabricated/unsupported winner name`, episode runner
  errors, and schema failures. These permit **exactly one retry** of the same
  (source, reading)
  pair, recorded with a `retry_of: <transaction_id>` link, and do **not**
  count toward the two-transaction saturation threshold. A second process
  failure on the same pair counts as an evidence failure (the pipeline is
  telling you something).
- **Evidence failures** — the pair was genuinely evaluated and did not support
  an install: `unsupported`, `no_changed_finalists`,
  `all_finalists_rejected`, no-op, or materially non-improving adjudication.
  These count toward saturation and the pair may not be rerun. Do not use one
  combined `ambiguous_or_unchanged_finalists` reason: it erases the distinction
  this policy depends on.

Code note: today's duplicate suppression
(`src/investigation/loop_v3.py:1010-1028`) matches only
`status == "installed"` transactions, so *any* failed pair is currently
rerunnable — the spec's stricter rule is correct and needed; it just needs
this carve-out so Slice 2 and Slice 4 compose instead of conflict.

Acceptance addition: a Slice-4
`no_winner_named_with_multiple_changed_finalists` rejection followed by a
retried, successful adjudication of the same pair installs; the same pair
failing twice on evidence saturates.

### A3 (P0 | Slice 6) Explicit decision: replace design C6 with a positive-attestation solution gate and ship a consumer migration map

Slice 6 states: "Only `reader_accepts_as_solution` can satisfy the
declaration gate." Today's gate is **content-hash-match-only**:
`AttestationPolicy` (`src/agent/tools_v2.py:2337-2389`) explicitly documents
(design C6) that a weak attestation — `reader_accepts` false, low coherence —
does **not** block declaration; only an absent or stale attestation blocks,
and the weakness is carried into the artifact. Revision 3 makes the decision
explicit: `meta_declare_solution` requires a fresh **positive** solution
attestation. This intentionally reverses design C6 rather than landing as a
side effect. Weak but fresh attestations remain useful routing evidence and may
support the future recovered-reading gate, but cannot satisfy solved
declaration. Ship this consumer migration map with the change:

1. `AttestationPolicy.check_declare_solution` — the gate itself
   (`src/agent/tools_v2.py:2337`).
2. `_is_positive_attestation` — `reader_accepts` AND `coherence >= 7`
   (`src/investigation/loop_v3.py:71-74`), used by the fallback tiering.
3. Fallback tier selection — `fresh_positive_attestation` tier
   (`src/investigation/loop_v3.py:151-166`).
4. Context routing — `DECLARE_COHERENCE` / `REPAIRABLE_COHERENCE_MIN`
   constants and the coherence-threshold routing
   (`src/investigation/context.py:54-57` and `:120-136`) must be re-expressed
   against the new fields (`semantic_recoverability`, `damage_scope`,
   `repairability`), or explicitly kept on `coherence` with a stated reason.
   Resolved in the spec: routing is re-expressed against the new fields with
   initial calibration constants (language confidence >= 0.7, recoverability
   >= 0.5); `coherence` stays in the schema as a clamped report-only legacy
   field in M5.3 and no longer gates declaration, routing, or fallback
   tiering.
5. Repair-agenda seeding — `verify_anomaly` items are created from
   attestation anomalies (`src/investigation/loop_v3.py:642-668`).
6. `AttestationRecord` serialization — new fields must round-trip through
   `to_dict`/state save/load (`src/investigation/state.py:112-133`);
   previously serialized states carrying single-`coherence` attestations must
   either load with mapped defaults or be declared format-incompatible.
   Resolved in the spec: load with the conservative mapping (positive iff the
   old `reader_accepts` was true and `coherence >= 7`).
7. `_VERIFY_SCHEMA` (`src/investigation/episodes.py:232-245`) and the verify
   contract prose.

Also see B3: whatever the gate decision, `reader_accepts_as_solution` must
not absorb "recovered reading" semantics.

### A4 (P1 | Slice 1) Cost-ceiling scope: every paid send, including mid-episode

"Enforced between provider calls" is ambiguous about worker episodes.
Amendment: `max_cost_usd` is checked before **every** paid provider send —
lead turns *and* episode-worker sends, including mid-episode continuation
sends inside reading/repair/verify workers. A ceiling reached mid-episode
ends that episode immediately with an explicit budget-class result (the
episode ledger shows it), after which the loop performs its honest
termination path (best supported branch, `unsolved` preserved without
positive attestation). The bake-off runner's matrix-level launch guard
remains a separate mechanism.

Acceptance addition: a ceiling that trips between two worker sends produces a
complete artifact with the episode marked budget-terminated and no further
paid call from either lead or worker.

### A5 (P0 | Slice 2) `repair_exhausted` must not silently un-gate episode kinds; make the phase map fail-closed

`allowed_episode_kinds` falls back to **all** kinds for an unknown phase:
`by_phase.get(str(phase), list(EPISODE_KINDS_FOR_CONTEXT))`
(`src/investigation/context.py:167-177`). Introducing the new
`repair_exhausted` workflow state without touching this map would therefore
*allow* reading and repair episodes in exactly the state whose purpose is to
stop them.

Amendment:

1. Add an explicit `repair_exhausted` entry. Suggested set:
   `["search", "compare", "verify"]` — matching the Slice 2 escape menu
   (alternate search experiment, compare finalists, honest termination);
   the hard requirement is that `reading` and `repair` are excluded.
2. Make the fallback fail-closed: an unknown phase returns the most
   restrictive useful set (suggest `["verify"]`, or raise in tests) plus a
   logged warning, so the *next* new state cannot repeat this bug.
3. Specify the pending-experiment window: when saturation recommends an
   alternate search experiment and it has been submitted but not collected,
   the workflow **stays** in `repair_exhausted` with a "collect the
   experiment" action; it must not flap back to a repair-family state until
   a new candidate hash or genuinely new verifier evidence exists (the
   spec's own reset rule). Resume must preserve this too (the counters are
   already specced to survive resume; the pending-experiment linkage must as
   well).

Acceptance addition: with saturation reached and an experiment pending,
`allowed_episode_kinds` contains neither `reading` nor `repair`, and the
rendered workflow menu names the pending experiment.

### A6 (P2 | Slice 2/7) Claim the pre-existing failing `test_lead_context` test

`tests/test_lead_context.py::test_negative_partial_attestation_creates_repair_action_menu`
fails on `main` today (verified: it asserts `"repair episode"` appears in the
`repair_required` action menu; the current menu wording is "Run or reuse one
reading episode … / Run one repair_transaction …"). It is a stale assertion
that predates M5.3, not an M5.3 regression — but M5.3's Slice 2 rewrites the
workflow menu, so this milestone must claim it: update the assertion to the
menu contract Slice 2 defines, and note in the slice's test evidence that the
failure predated the milestone. Do not leave it ambient red through the
milestone, and do not "fix" it by reverting menu wording.

### A7 (P2 | Slice 7) The analyzer regression fixture must be trimmed of ground truth and message bodies

Slice 7 allows "a frozen analyzer regression fixture or a trimmed
equivalent". Tighten to: the committed fixture **must** be trimmed. Strip
benchmark `ground_truth` / expected plaintext / alignment blocks and raw
model message bodies; keep the structural fields the analyzer reads (episode
ledger shapes, budgets requested/executed, repair transactions, workflow
transitions, costs, declaration state). Add a test asserting the fixture
contains no ground-truth keys — same class as the existing runtime firewall
tests. Rationale: committed fixtures are read by future sessions and tools;
a fixture carrying the answer key for a live benchmark page is a standing
firewall leak in the repo itself.

### A8 (P1 | Slice 3) Cache key: exact builder inputs, or a written sufficiency proof

The specced cache key (candidate content hash, null mask, language, LM
variant, repair config) is a proxy. The menu builder actually consumes
`(cipher_text, base_key, mask, language, config, dictionary_path,
model_path)` (`src/investigation/actions.py:926-935`), and the per-call
config comes from `_word_repair_menu_config(args)` — i.e., partially
model-visible knobs. Either:

1. key the cache on digests of the **exact builder inputs** (base-key digest,
   mask digest, boundary-spans digest, resolved config digest, resolved
   dictionary/model paths); or
2. include a short written proof in the spec that the proxy is sufficient —
   e.g.: the rendered candidate text plus boundaries determine the decode
   projection of every token *present* in the ciphertext; base-key entries
   for absent tokens cannot affect the menu; `dictionary_path`/`model_path`
   are pure functions of (language, variant); and the "repair config"
   component is defined as the **resolved** config digest, not the raw args.

Either way, the acceptance test "cache invalidates on key, mask, boundary,
model, or config changes" should include the adversarial case: two branches
whose rendered text is identical but whose configs differ per-call.

---

## Part B — repair-reframe seams to cut in M5.3

Context (reconciling Sol's P1-8 with the frozen in-flight scope): the design
review concluded that the repair *interface* is mis-framed — prose readings
as the mandatory compile source, vote-on-mismatch key inference — and that a
typed-packet interface should replace it in a follow-on milestone (Part C),
gated on experiments. Sol argued for landing only the orthogonal M5.3
controls now. The adjudicated position: **all M5.3 slices land**, because
Slices 2 and 4 are interface-agnostic guardrails (saturation and
host-validated acceptance are what make *any* repair interface safe, and they
transfer wholesale to the packet model), and the entrenchment risk is
concentrated in three specific spots, each neutralized by a small seam
amendment below rather than by re-scoping in-flight work. Each seam is
schema/naming-level; none changes M5.3 runtime behavior beyond what is
stated.

### B1 (P1 | Slice 3) `hypothesis_test_words` input headroom: typed word claims, granular anchors, typed rejection

1. Each batch item carries a `claim_type` field whose only currently accepted
   value is `"word"`. Future values such as `"boundary"` and an `op` field are
   reserved in documentation, but a caller that supplies them before
   implementation receives `unsupported_reserved_field`; no field is accepted
   and silently ignored. M5.3 evaluates word claims exactly as specced.
2. Existing 120-token reading-window ids are too coarse for exact word repair.
   Slice 3 mints granular host-owned `word_id` values and opaque per-token
   anchors within each window. A batch item may use one `word_id` or a
   contiguous `{start_token_id, end_token_id}` run. The host validates
   contiguity and content-hash binding. Raw positional refs remain accepted
   only for singleton parity and are marked legacy/model-discouraged; the
   future packet surface exposes only host-owned anchors.
3. A hypothesis whose word length does not match the span's token count is
   rejected as **typed data** — reason `not_expressible_as_key_edit`, with the
   span/word lengths — not as a schema or validation error. Same outcome in
   M5.3 (nothing installs), but the follow-on milestone turns that typed
   rejection into a pointer at the annotation channel without changing the
   wire shape, and models get a legible reason instead of a generic error.

### B2 (P1 | Slice 2) Name the saturation-identity component generically

The repair-cycle identity tuple's "reading id and reading content digest"
components should be stored as `interpretation_id` / `interpretation_digest`
(today these are always a Reading's id/digest). Pure naming in the state
format; no logic change. The follow-on milestone's interpretation packets
then join the saturation machinery without a serialized-state migration or a
parallel counter system.

### B3 (P2 | Slice 6) Do not overload the solved gate with "recovered reading" semantics

Land the multi-field verifier contract and its routing table as specced. But
keep the boundary clean: `reader_accepts_as_solution` answers exactly "is
this candidate complete enough to declare **solved**". The honest terminal
for "correct basin, meaning recovered, residual inexpressible by any key"
is a **separate** future action (`meta_declare_recovered_reading`, Part C)
with its own gate and its own accounting, excluded from solved statistics.
Slice 6 should therefore route the high-recoverability/incomplete case to
"one bounded repair cycle, then compare/hold or honest unsolved" — never to
declaration — and the verifier fields should be treated as evidence usable by
*both* future gates. No Slice 6 wording should imply that high
`semantic_recoverability` plus carried weakness can satisfy
`meta_declare_solution`; that would recreate the C6 ambiguity of A3 in a new
form.

---

## Part C — deferred to the follow-on milestone (M5.4 "typed repair"); do not build in M5.3

Full definitions and rationale: `docs/repair_mechanism_rethink.md`
(revision 3), §4.4 and §7. Summary of what is deliberately **out** of M5.3:

1. **`InterpretationPacket`** — the reading episode's result extended so one
   worker call returns editorial reading + typed span assertions + typed
   non-key annotations + holes; the packet (not prose) becomes the repair
   precondition; no second episode translates prose. The producer defaults to
   one tool-free structured response and permits at most one batched lookup
   plus one final response. Assertions use granular host-owned word/token
   anchors, not coarse window ids or raw numeric positions.
2. **Deterministic compiler** — assertions compile to key edits only at
   1 proposed char ↔ 1 ciphertext token; vote-on-mismatch inference and
   banded alignment retired as key-inference mechanisms
   (`src/investigation/actions.py:539-585` is the code being replaced).
3. **Typed positional annotations** — explicit proposals with separate support
   levels. The deterministic occurrence-conflict signature proves only that a
   global key edit is unsupported; it does not prove the replacement or a
   transcription scar. Independent linguistic review remains an editorial
   support level; source support requires manuscript/OCR or corroborating
   transcription evidence. Annotations never
   enter canonical render, content hashes, or internal scores. Solver-supported
   symbol-level null masks remain canonical structural state, not annotations.
4. **`meta_declare_recovered_reading`** — a separate hash-bound terminal
   action, excluded from solved-result statistics. Its attestation binds to a
   composite of key content, packet, annotated reading, and annotation-ledger
   hashes; ordinary solution attestation remains key-text-only.

Gates before M5.4 is scoped/adopted (in order):

- **Gate 1 — residual-composition measurement**: grading-side classification
  of stored borg_0109v/0045v candidates' residual errors (genuine key error /
  isolated occurrence conflict / editorial-lacuna / boundary / grading
  artifact / ambiguous). Labels carry confidence, with every proposed scar,
  every low-confidence item, and at least a 10% sample human-reviewed. Sizes
  the milestone; no runtime code.
- **Gate 2 — oracle compiler test** (design note §7.1–7.5): LLM-free, $0,
  perfect-proposal and adversarial-proposal arms. A small reference
  implementation tests the proposed deterministic compiler directly; the
  production menu-backed batch tool is a secondary comparator after Slice 3,
  not a substitute for the compiler. Includes negative controls and a
  pre-registered false-support gate with per-claim and affected-token rates.
- **Gate 3 — small mandatory live Phase 1** with separate spend approval,
  regardless of how decisive Gate 2 is (Gate 2 grants the very premise —
  perfect interpretation quality — that live runs must supply).

## Firewall notes

- Nothing in Parts A or B routes ground truth into routing, repair, scoring,
  retries, verification, or declaration. A7 *strengthens* the firewall
  (committed fixtures may not carry answer keys).
- Gate 2 (Part C) is an **oracle compiler test**: it deliberately synthesizes
  proposals from ground truth and feeds them into the repair mechanism's
  production input, in an isolated harness. Required isolation: dedicated
  script, dedicated artifact directory (`artifacts/oracle_compiler_test/`),
  an explicit `"oracle_compiler_test": true` marker in every artifact,
  exclusion from benchmark loaders/summaries/capability tables, and no
  description of it as ground-truth-free. This corrects the prior design
  note's inaccurate firewall wording (Sol P1-5, accepted).
- Gate 1 is grading-side only, the same class as `benchmark/scorer.py`.
- Gate 3's recovered-reading attestation must bind to the packet, annotated
  reading, and annotation-ledger hashes in addition to the canonical key-text
  hash; changing noncanonical interpretation state must make the attestation
  stale even when the key-derived text is unchanged.
