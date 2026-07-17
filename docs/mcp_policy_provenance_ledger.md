# MCP Phase 0 — Policy Provenance Ledger

Status: Phase-0 deliverable per `docs/mcp_dual_harness_proposal.md` §3.3, §6,
§8, expanding the §11 seed table. Written 2026-07-17 against the post-M5.3
tree (HEAD `533f60d`). No code changes accompany this document.

Scope: every host-side control in the v3 investigation host — the lead loop
(`src/investigation/loop_v3.py`), the context/workflow layer
(`src/investigation/context.py`), the declaration policy
(`src/agent/tools_v2.py::AttestationPolicy`), the episode runtime
(`src/investigation/episodes.py`), and the experiment queue
(`src/investigation/experiments.py`) — plus the two v2-inherited executor
policies that remain active inside v3 workers. v2-only controls
(`V2GatePolicy` cascade) are out of scope: the MCP server never constructs
them.

Method: each control was located in source and traced to its introducing
commit and/or spec section. Where a commit or spec records the motivating
failure, that is cited. Where it does not, the entry is flagged
**UNKNOWN-PROVENANCE** rather than given an invented history.

Classification key (§3.3):

- **INV** — invariant: stays hard in MCP v1.
- **EVID** — evidence-management mechanism: stays active in MCP v1.
- **POL** — investigative policy: candidate for advisory form in MCP v1
  (pre-registered divergence, never silent).
- Straddles are noted per entry (§3.3's default-deny example).

Provenance shorthand — commits:
`ce478db` (Slice 5, typed experiment config), `f0818cb` (Slice 1, budgets +
cost ceiling), `24541d2` (Slice 3, batched hypotheses), `f371bba` (Slices
2+B2+4, saturation + acceptance), `a0ba63c` (Slice 6+B3, diplomatic verify /
C6 reversal), `2ca2c6d` (Slice 7, observability + Sequence-B), `22b6656`
(post-smoke acceptance hardening), `10af31f` (K3 truncation resilience),
`d7e1376` (429 retry), `616f297` (max_cost_usd wiring), `e725d27` (null-mask
renderer fix), `f543d28` / `f62dcee` / `9da17ce` / `12201b2` (M5.2 slices),
`55404f5` (M5.1), `e699773` (M2 episodes), `96f2fbe` (M4 experiments),
`ef4ac9a` (improvement Phase 0, first firewall tests), `d41d444`
(polyalphabetic tooling, context-family gate). Specs:
`docs/specs/agent_v3_m5_3_control_reliability_spec.md` (M5.3),
`agent_v3_m5_2_strategy_repair_spec.md` (M5.2),
`agent_v3_m5_1_declaration_trigger_spec.md` (M5.1), `agent_v3_m5_spec.md`
(M5), `agent_v3_m2_spec.md` (M2), `agent_v3_m4_spec.md` (M4). Key artifacts:
M5.2 smoke `artifacts/m5_2_targeted_smoke_20260716/.../d3eccab14a40.json`
(the eight numbered failures in M5.3 spec "M5.2 targeted smoke"), M5.3 smoke
report `docs/reports/m5_3_targeted_smokes_2026_07_17.md`, M6 bake-off
`artifacts/m6_bakeoff/summary.jsonl`, Borg null-mask investigation (proposal
§2.4).

Default pre-registered evidence bar to change ANY form (referenced below as
"§6 bar"): a targeted ablation of that one control inside the §6 experiment
frame — pilot + ≥2 confirmation replicates per arm on the discriminating case
subset, pre-registered pass/fail thresholds, policy-intervention counts
recorded (not just terminal compliance). Single-run anecdotes explicitly do
not qualify (proposal §6 staging rule, risk 10). Entries note additional
control-specific evidence where relevant.

---

## 1. Ground truth and verifier independence (GT)

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| GT-1 | Ground-truth firewall: host constructs every model-visible byte; solver runner hardcodes `ground_truth=None` (`experiments.py::_automated_solver_runner`); episodes get `benchmark_context=None` (`episodes.py::run_episode`); artifact fixtures GT-trimmed with a recursive key-rejection test (`2ca2c6d`) | Pre-Phase-0 solver workflow consumed grading data (§11 seed); first regression tests landed in `ef4ac9a` (`tests/test_ground_truth_firewall.py`), extended by M5 Part 5 and Slice 7 Part E | INV | Scored results become unfalsifiable; the benchmark instrument is dead | **Hard** — server/capsule boundary (§4.4); sandbox escape = failed run (§9.5) |
| GT-2 | Verify-reader independence: `verify` episode has an EMPTY toolset, one send; context is candidate text + language ONLY — no scores, no branch cards; lead-authored `goal` deliberately ignored (M5 review F-3) (`episodes.py::_build_verify_context`, `EPISODE_KINDS["verify"]`) | M5 spec "Why": v2/v3 leads declared confident readable WRONG basins (borg_0077v); self-attestation cannot catch it; contract made diplomatic in `a0ba63c` after the Borg diplomatic-text complication (M5.3 spec) | INV | Gate degrades to self-attestation with paperwork (proposal §3.4 Option B) | **Hard** — server-side, API-billed verify (§3.4 Option A) |
| GT-3 | Host-derived experiment `language`: never a model-suppliable key; supplied `language` rejected as unknown (`experiments.py::validate_experiment_config`) | `ce478db`: language is host-derived; part of the Slice-5 firewalled config surface | INV | A config channel for smuggling target facts into solver runs | **Hard** |

Change-evidence: none pre-registered for GT-1/GT-2/GT-3. Proposal §1: no
invariant is removed in v1; the §6 experiment does not test firewall removal.

## 2. Declaration gating (DECL)

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| DECL-1 | `meta_declare_solution` requires the NEWEST attestation matching the branch's CURRENT rendered hash to be POSITIVE (`reader_accepts_as_solution=true`); absent/stale/weak/negative all block, with the verdict echoed for routing (`tools_v2.py::AttestationPolicy.check_declare_solution`, ~line 2337) | M5 (borg_0077v wrong-basin declares); weak-allows→weak-blocks was the deliberate **C6 reversal** in `a0ba63c` (+20 tests); B3: high `semantic_recoverability` alone never unlocks | INV | Reverts to declared-junk terminal states; the strongest observed v3 failure class returns | **Hard**, hash-bound server gate (§3.3 item 1, §11) |
| DECL-2 | Attestation records are written by the DISPATCHER only, with the content hash computed at dispatch time under the pinned renderer; workers never write state (`loop_v3.py::_dispatch_verify_run`; A1/A6) | M5 A6 binding; M2 A1 single-writer discipline (`e699773`) | INV | Attestations become forgeable / mislabelable by workers or clients | **Hard** |
| DECL-3 | Verify arity: exactly ONE existing branch per verify episode; multi-name lists rejected (M6 F6) (`loop_v3.py::_dispatch_verify_run`) | M6 review F6: the old `next(has_branch)` silently attested `main` on `["typo","main"]`, mislabeling the attestation | EVID | Mislabeled attestations corrupt the declare gate's hash matching | **Hard** (input validation) |
| DECL-4 | Conservative verdict coercion: coherence >10 or unparseable floors to 0, never 10 (`loop_v3.py::_clamp_coherence`, M5 review F-1); unit-interval clamps + non-positive defaults (0.0/0.0/basin_wide/none) for omitted routing fields (`state.clamp_unit_interval`, Slice 6) | `a0ba63c`: a reader answering on a 0-100 scale must not mint a top-coherence attestation | EVID | Scale-violating readers mint spurious positives | **Hard** (validation) |
| DECL-5 | Accepted declaration terminates the batch: remaining tool_uses get synthesized `run_terminated` results, preserving attested == declared == scored (M5 review F-2) (`loop_v3.py` post-dispatch early-exit, ~line 2356) | M5 review: `[meta_declare_solution, act_set_mapping]` in one turn would mutate the declared branch after the gate allowed it | INV | Declared branch can silently diverge from the attested content | **Hard** (transactional consistency; MCP analog: revision commit on terminal) |
| DECL-6 | Fallback tiering at exhaustion/error: fresh POSITIVE attestation → fresh hash-bound compare winner → scalar best-effort; only tier 1 synthesizes `fallback_declared`; otherwise honestly `unsolved`/`error` with best branch preserved (`loop_v3.py::_select_v3_fallback`; latest-verdict-governs per Slice 6) | M5.1 Slices D/E (`55404f5`); honest termination in `f62dcee` (M5.2 Slice 5: "negative-only exhaustion terminates honestly unsolved"); ordering fixed in `a0ba63c` (coherence no longer sorts the tier) | INV (the no-positive-no-declare half) / POL (ordering within the attested tier) | Unattested basins masquerade as solutions again (the pre-Phase-0 `fallback_declared`-as-solved bug, `ef4ac9a`) | **Hard** for the invariant half; tier-internal ordering: telemetry |
| DECL-7 | Attestation branch-label resync on install rename (M6 F5): hash-primary matching preserved; labels re-pointed only when content moved (`loop_v3.py::_resync_attestation_branch_on_rename`) | M6 review F5: collision renames left stale-vs-required messages naming dead branches | EVID | Misleading stale/required diagnostics; hash matching itself unaffected | **Active** |
| DECL-8 | `meta_declare_unsolved` is deliberately NOT gated (base-class default) | M5 Part 3: honest surrender must never be blocked | INV (by omission) | Coercion to keep polishing; dishonest terminals | **Hard** (keep ungated) |

Change-evidence: DECL-1/2/5/6 — none in v1 (§3.3 item 1 "declaration gating"
stays hard). DECL-6 tier ordering: §6 bar comparing retention quality of
alternative orderings.

## 3. Repair transaction and host-validated acceptance (REP)

All in `loop_v3.py::_dispatch_repair_transaction` unless noted. Provenance:
M5.2 Slice 4 introduced the transactional shape (`12201b2`); M5.3 Slice 4
(`f371bba`) added the 8-check host-validated acceptance after M5.2 smoke
failure 6 ("host validated that a worker named a changed branch, but did not
independently enforce the full collateral-evidence acceptance contract");
`22b6656` hardened edit-claim binding after the paid targeted smokes
(`docs/reports/m5_3_targeted_smokes_2026_07_17.md`).

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| REP-1 | Evidence binding preconditions: a stored Reading must exist, name the same branch, and match the branch's CURRENT content hash (`fresh_reading_required` / `reading_branch_mismatch` / `stale_or_unbound_reading`) | §11 seed: readers proposed prose or unsupported mappings as edits; M5.1 Slice B ("readings repairable without treating prose as a key") | INV | Free-form prose becomes key edits; unbound repairs re-enter | **Hard** compiler/install validation (§11) |
| REP-2 | Duplicate-transaction suppression by `(source_content_hash, interpretation_digest)` — a re-named reading or new `as_name` with identical content is the SAME pair (`duplicate_suppressed`, `pair_evidence_failed`) | M5.3 B2 (`f371bba`): M5.2 smoke failure 3 — rewording a goal allowed duplicate work on unchanged content/evidence | EVID | Budget burns on re-running settled repairs under new names | **Hard** exact-duplicate block; near-duplicate telemetry (§11) |
| REP-3 | Eight ordered acceptance checks binding the worker's claims to its OWN successful composite ToolCall results: winner_named, worker_applied, winner_fork_evidence (failed-call forks disqualify), edit_claims_bound (fail-closed label binding, `_unbound_edit_claims`), winner_adjudicated (≥2 changed finalists), collateral_within_limits (damaged ≤ improved), no_op_probe (scratch-probe re-render), scalar_non_decrease | Slice 4 (`f371bba`); prose-claim fail-closed binding added by `22b6656` after paid smokes showed `Applied ... W:D->F ...` style claims | INV (evidence binding, reject-before-install) with one POL-assumption inside (see REP-4) | Workers install unsupported edits; the M5.2 failure class returns | **Hard** — reused as-is for MCP's `repair_transaction` over client-compiled finalists (§3.5) |
| REP-4 | Default-deny on ANY measured `dict_rate`/`quad` decrease; `REPAIR_ACCEPTANCE_POLICY = None` is the guarded M5.4 hook (`loop_v3.py` check 8, module constant ~line 388) | Slice 4; the §3.3 straddle example verbatim: default-deny is invariant, "any scalar decrease is materially bad" is a scoring-policy assumption | INV (default-deny) / POL (the scalar test) | A weak scalar becomes an accidental permanent axiom — or, removed entirely, degrading installs pass | **Hard**, unchanged for clean comparison; record every scalar-denied install as telemetry so the assumption is auditable |
| REP-5 | Repair-agenda seeding gated on verifier `repairability == "local_repair"`; broaden/none verdicts must not queue local repair work (`_dispatch_verify_run` agenda block) | Slice 6 (`a0ba63c`): agenda seeds from the repairability verdict, not coherence | EVID (mechanism) carrying a POL routing judgment | Local-repair queues fill with unfixable basin-wide damage → polishing gibberish | **Active** mechanism; the routing judgment inherits WF-4's advisory status |
| REP-6 | Installed transaction auto-closes the open agenda items for its source content; installs stamp `reverification_required` | M5.2 Slice 5 (`f62dcee`) | EVID | Stale agenda items misdirect later turns; repaired content escapes reverification | **Active** |
| REP-7 | Repair-transaction phase gate: dispatch blocked unless workflow phase is `candidate_reading`/`repair_required` (`_dispatch_tool`, `repair_transaction_not_ready`) | M5.2 Slice 2 (`f543d28`) | POL | Repairs launched with no verified evidence context; wasted spend, not corrupted evidence (REP-1 still binds) | **Advisory** with policy id (§3.3 named divergence: phase/action restrictions) |

Change-evidence: REP-3/REP-4 hard forms — §6 bar plus the M5.4
interpretation-packet work (a tested, ground-truth-free policy object may
replace default-deny; proposal §10). REP-7 advisory — pre-registered in §3.3;
re-hardening requires §6 telemetry showing clients repeatedly running
evidence-free repairs.

## 4. Saturation and duplicate-work suppression (SAT / DUP)

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| SAT-1 | Durable saturation entries keyed `saturation_key(content_hash, attestation_key)`; reset ONLY by new content or new verifier evidence — reworded goals never reset (`state.repair_saturation`; `loop_v3.py::_settle_repair_outcome`) | M5.3 Slice 2 (`f371bba`); M5.2 smoke: 25 turns/$4.01 circling one damaged basin (artifact `d3eccab14a40`) | EVID + resource (§11: "mixed evidence/resource policy") | Indefinite repair retries on unchanged evidence | **Preserve in v1** (§11); distinguish exact repeated evidence from genuinely new hypotheses via interpretation digests |
| SAT-2 | Process-vs-evidence failure taxonomy: enumerated evidence reasons (`_EVIDENCE_FAILURE_REASONS`); unknown reasons default to `process` so novel failures never consume saturation budget on first occurrence; one linked `retry_of` per pair; a second process failure reclassifies as evidence (`_classify_failure_reason`, `_settle_repair_outcome`) | Slice 2 (`f371bba`), conservative-default rationale in code | EVID | Provider hiccups exhaust repair budget; or infinite process-failure retries | **Active** |
| SAT-3 | `repair_exhausted` latch after 2 evidence failures per (content, evidence); blocks further `repair_transaction` (`repair_saturated`); menu narrows to alternate search / compare / declare-unsolved (+ collect pending experiment); exhaustion short-circuit runs FIRST in every attested route (`context.py::_repair_exhausted_menu`, `_attested_menu`) | Slice 2; §11 seed "agents retried ineffective repairs indefinitely" | EVID/POL straddle: the latch is an evidence fact; the narrowed menu is policy | Return of the M5.2 basin-circling failure | **Preserve in v1** (§11). The latch and the `repair_transaction` block stay hard; the narrowed episode-kind menu inherits WF-2's advisory status |
| SAT-4 | Reading suppression: at most 1 reading per (content hash, attestation key); duplicates return the existing `reading_id` (`loop_v3.py::_dispatch_episode_run`, `duplicate_reading_suppressed`) | Slice 2 (`f371bba`): repeated readings of unchanged content were the other half of the duplicate-work bug | EVID + resource | Paid reading episodes re-run on unchanged evidence | **Hard** exact-duplicate block |
| DUP-1 | Lead read-call cache: duplicate read-only calls keyed (tool, normalized args, branch content hashes) suppressed with a structured pointer to state (`loop_v3.py` `read_only_lead_tools` / `read_call_cache`) | M5.2 Slice 2 (`f543d28`); §11 seed "agents repeated unchanged searches and reads" | EVID + resource | Transcript/context bloat; re-derived facts drift from state | **Hard** exact-duplicate block; near-duplicate telemetry (§11) |
| DUP-2 | Repeated-call signature counting: every dispatch hashes (tool, args, branch hashes); repeats emit `repeated_call` events; counts survive resume (`state.call_signature_counts`) | M5.2 Slice 5 (`f62dcee`) | EVID (telemetry) | Blind spot for disciplined-looking clients that thrash | **Telemetry** (feeds §6 policy-intervention metrics) |
| DUP-3 | No-new-information streak: per-turn digest over branch hashes, readings, attestations, episode/experiment/repair results; streaks emit events and survive resume (`loop_v3.py::_information_digest`) | M5.2 Slice 5 (`f62dcee`) | EVID (telemetry) | Stagnation invisible until exhaustion | **Telemetry** |
| DUP-4 | Content-identical episode installs deduplicate onto the existing branch with a durable provenance alias; agenda residuals remapped (`loop_v3.py::_dispatch_episode_install`) | M5.2 Slice 5 (`f62dcee`) | EVID | Branch namespace floods with identical content; compare/portfolio diversity illusory | **Hard** |

Change-evidence: SAT-1/SAT-3 softening — §6 bar with the specific
pre-registered metric "maximum repeated expensive actions" (§6) and evidence
that a mature harness self-terminates repair loops without the latch; §11
requires distinguishing new-hypothesis submissions before any relaxation.

## 5. Workflow routing and focus (WF) — the named advisory candidates

These are the §3.3 pre-registered divergences: hard blocks in v3, advisory in
MCP v1, each carrying its policy id in `next_steps`/status output.

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| WF-1 | Workflow state machine (`searching` / `candidate_reading` / `repair_required` / `broaden_required` / `repair_exhausted` / `verified`) with a binding action menu rendered every turn (`context.py::workflow_state`) | M5.2 Slices 1-2 (`9da17ce`, `f543d28`): the lead behaved like a low-cost v2 operator and did not turn negative verification into bounded repair | POL | Loss of the deterministic route from verification verdicts to next actions; M6 evidence says v3's weakness was recovery, not architecture | **Advisory**: §4.3 — status must not smuggle the binding menu back as neutral fact; recommendations labeled with policy ids |
| WF-2 | `allowed_episode_kinds` per phase; `episode_run.kind` enum narrowed per turn; off-menu kinds blocked (`episode_kind_not_available`); unknown phase fails closed to `["verify"]` via `warnings.warn` (`context.py::allowed_episode_kinds`; `loop_v3.py::_dispatch_tool`) | M5.2 Slice 2 (`f543d28`); fail-closed default + `repair_exhausted` row added in Slice 2 of M5.3 (`f371bba`) | POL (the restriction); EVID (the fail-closed default) | Clients may run any episode kind anytime; wasted spend, gated elsewhere for integrity | **Advisory** recommendation + telemetry on off-menu choices. Fail-closed default is moot server-side (no phase-gated enum), noted for parity audits |
| WF-3 | Lead tool-surface restriction: 16-tool strategist surface (22 with benchmark context); hidden v2 operator tools rejected with `lead_tool_not_available` (`episodes.py::v3_lead_tool_definitions`; `loop_v3.py::_dispatch_tool`) | M5.2 Slice 1 (102-tool effective surface → 16; schema 66k→10.9k chars); Slice 2 enforcement (`f543d28`) | POL (surface design) | In v3: operator regression. In MCP: n/a — the §3.5 surface IS the exposed tool set | **Structural**: enforced by surface composition, not a gate; the raw v2 operator surface stays unexposed (§3.5 "deliberately NOT exposed") |
| WF-4 | Verifier-to-route thresholds: `TARGET_LANGUAGE_CONFIDENCE_HIGH=0.7`, `SEMANTIC_RECOVERABILITY_HIGH=0.5`; `_attestation_route` maps (tlc, recoverability, damage_scope) → repair / compare_or_search / broaden; residual cell (recognizable + local + low recoverability) conservatively routes AWAY from repair/declare; legacy records route broaden (`context.py` lines 55-140) | Slice 6 (`a0ba63c`), completing the master routing table; §11 seed "agents polished basin-wide gibberish or abandoned useful local damage". Thresholds are host constants, "tunable only with paid-smoke or equivalent targeted evidence" (code comment) | POL | Clients polish unfixable text or abandon repairable candidates — a spend/quality risk, not an integrity risk (DECL-1 still gates) | **Advisory** route with policy id; record client choice and outcome (§11) |
| WF-5 | Workflow hints: `mid_budget_verify`, `late_turn_attestation` (≤`LATE_VERIFY_TURNS`=4), `positive_attestation_declare` (after `POST_ATTEST_PATIENCE`=2), `negative_verify_repair`, `late_branch_adjudication`; deduplicated by content-bound hint keys; emitted as artifact events (M6 F7) (`context.py::workflow_hint_candidates`) | M5.1 Slices C/E (`55404f5`): leads that first attempt declaration on the final turn are blocked with zero turns left; late compare exists so fallback can use a fresh winner | POL (already advisory in v3) | Late-turn blocked-declare failure mode returns unprompted | **Advisory** (carried over as-is) + telemetry |
| WF-6 | Scalar-selected workflow focus: `_best_branch_for_auto_declare` (dict_rate/quad scalar) selects THE branch the workflow menu, decode window, and hints center on (`context.py` throughout; `agent/loop_shared.py`) | §11 seed: "needed a deterministic branch when agents failed to choose". Inherited from the v2 auto-declare fallback into M1+. **UNKNOWN-PROVENANCE (artifact)**: no single motivating artifact/test is recorded for the focus-selection behavior itself, only for the fallback declare it descends from | POL | Nothing breaks; candidate attention becomes client-owned. Proposal §2.4/§3.5: the scalar silently defining "the" candidate is itself an identified failure contributor (Borg null-mask ranking discarded) | **Diverge**: diverse candidate portfolio; scalar is one labeled signal (§3.5 `candidate_list`, §11). Slice-7 branch roles (`_compute_branch_roles`) already de-conflate the labels |

Change-evidence for WF-1/2/4/6 re-hardening: §6 policy-intervention telemetry
showing clients ignoring advisories AND a measured accuracy/spend regression
beyond pre-registered thresholds, on ≥2 replicates. For WF-4 threshold
retuning: paid-smoke or equivalent targeted evidence (already pinned in code).

## 6. Budgets and cost ceilings (BUD)

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| BUD-1 | Per-run paid ceiling `max_cost_usd`, checked before EVERY paid send — lead turns, every worker send, mid-episode continuation, the submit-only reserve; mid-episode hit ends the episode `cost_ceiling_reached`; run terminates honestly (`loop_v3.py::_cost_ceiling_reached`, `episodes.py::run_episode`) | M5.3 Slice 1 (`f0818cb`); wired through the benchmark runner in `616f297`; distinct from the bake-off matrix guard | INV (§3.3 item 1: cost ceilings) | Unbounded paid spend on a wedged run | **Hard** for server-side spend (verify episodes, experiments); client-side lead spend is the client harness's own metering — recorded as telemetry |
| BUD-2 | Per-episode tool-call caps enforced BEFORE each call (not per batch); skipped over-budget calls get paired synthesized `budget_exhausted` results; overshoot ledger-visible (`suppressed_over_budget_calls`) (`episodes.py::_run`) | M5.2 smoke failure 1: 4-call readings executed up to 13 calls, repairs up to 15 — an enforcement defect (`f0818cb`) | INV (resource) | Episode budgets are fiction; per-batch overshoot returns | **Hard** |
| BUD-3 | `max_tool_calls` clamp: a worker/lead request may LOWER the registered cap, never raise it; clamp 1..registered; unparseable requests ignored (`episodes.py::EpisodeSpec.__post_init__`) | M5.2 smoke failure 2: the model raised `max_tool_calls` above the host default (`f0818cb`) | INV (resource) | Workers vote themselves budgets | **Hard** |
| BUD-4 | Reading envelope pinned at (16 calls, 8192 tokens, 300 s) — a MAXIMUM safety envelope; M5.2's 4/4096/180 reduction explicitly invalidated (the batch-check defect meant it was never actually tested); "do not lower below this until a binding 4/8/12/16 calibration shows reading quality is preserved" (`episodes.py::EPISODE_KINDS["reading"]` comment) | M5.1 forensics (commit `1d01ef4` raised it after 10-11-call exploration with no submission); restored by Slice 1; design principle 8: reduction requires a binding-cap usability test | POL (a calibrated budget, not an integrity gate) | Reading workers exhaust without submitting (the M5.1 failure) or budgets are lowered on non-evidence | **Hard cap** carried as-is; change requires the pre-registered binding 4/8/12/16 calibration — already the recorded evidence bar |
| BUD-5 | Repair budget 12 → 6 calls, reduced ONLY after batch/singleton parity was proven (`hypothesis_test_words`, tests in `tests/test_hypothesis_actions.py`) (`episodes.py::EPISODE_KINDS["repair"]`) | M5.3 Slice 3 (`24541d2`); M5.2 smoke failure 4 (each singleton rebuilt the expensive menu) | POL (calibrated budget) | Slow, menu-rebuilding repair workers | **Hard cap**, same evidence rule as BUD-4 |
| BUD-6 | Submit-only reserve on exhaustion: one final send exposing only `episode_submit_result`, preserving a structured partial (`episodes.py::_final_result_send`) | M5.1 forensics ("submit a useful partial early"); protected as a distinct reserve in Slice 1 | EVID | Exhausted episodes return nothing usable | **Active** |
| BUD-7 | Episode wall-clock checked between calls; hard turn cap `max(2*calls+3, 12)` (`episodes.py::run_episode`) | M2 A9 (`e699773`) | INV (resource) | Runaway worker loops | **Hard** |

## 7. Episode isolation and toolsets (EPI)

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| EPI-1 | Full episode isolation: deep-copied snapshot workspaces; nothing an episode does touches the lead workspace until explicit `episode_install_branch`; ledger snapshots deep-copied again before restore (aliasing fixes, F5 both directions) (`episodes.py::_build_episode_workspace`; `loop_v3.py::_dispatch_episode_install`) | M2 A1/F5 (`e699773`) | INV (immutable provenance, transactional consistency) | Worker mutations corrupt lead state invisibly | **Hard** (server-side episode runner keeps it) |
| EPI-2 | Per-kind hard tool allowlists + `_validate_episode_toolset`: unknown tools, `EPISODE_EXCLUDED_TOOLS`, hypothesis handlers, and `meta_`/`inspect_`/`list_` prefixes all rejected (`episodes.py` lines 84-135, 521-535) | M2 (`e699773`): workers must not declare (meta_), see benchmark context (inspect_/list_), or write the board (A10) | INV for the firewall/declaration parts; EVID for board single-writer | Workers self-declare or read benchmark context; board multi-writer races | **Hard** |
| EPI-3 | `EPISODE_EXCLUDED_TOOLS`: long-running solver searches barred from episodes and routed to the experiment queue (wall clock only checked between calls) | M2 A9; M4 provides the queue (`96f2fbe`) | EVID + resource | Episodes blow wall-clock budgets mid-tool | **Hard** (structural: MCP exposes experiments, not raw solver tools) |
| EPI-4 | Search episodes must name a valid `search_tool` from the registry-derived enum; companions auto-added; invalid choices get a structured error + corrected example + explicit pointer to `experiment_submit` (`episodes.py::episode_toolset_for`; `loop_v3.py::_dispatch_episode_run` error payload) | M2/M4; error-payload guidance hardened alongside Slice 5's corrected-example pattern | EVID (usability/structure) | Workers guess tool names; long searches sneak into episodes | **Active** |
| EPI-5 | Result-schema validation with ONE schema-retry (error echoed) then `episode_failed(schema_mismatch)`; required-set minimalism — omitted reading confidence defaults BELOW the auto-apply threshold instead of failing the episode (`episodes.py::validate_against_schema`, `_READING_SCHEMA` note) | M2; minimalism philosophy from M5.1 Stage-1 forensics (a required-confidence schema failure destroyed whole readings) | EVID | Unstructured worker output; or over-strict schemas destroying usable partials | **Active** |
| EPI-6 | A9 crash guard: episode SETUP and run failures become structured `episode_failed(runner_error)` results, never lead crashes; KeyboardInterrupt commits the ledger + budget then re-raises into R5 pairing (`episodes.py::run_episode`) | M2 A9 (`e699773`) | INV (artifact preservation) | One malformed snapshot loses the whole run artifact | **Hard** |
| EPI-7 | Hypothesis-board single writer (A10): only the lead writes the board; installs route mirrored card fields through `board.update`; survey results update the board via the dispatcher (`loop_v3.py::_dispatch_episode_install`, `_dispatch_episode_run`) | M2 (`e699773`) + F4 spec-author amendment | EVID | Board state forks; coverage/next-steps evidence corrupts | **Active** |

## 8. Experiment queue and config validation (EXP)

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| EXP-1 | Typed `experiment_submit` config: registered per-type schema with enums/defaults/docs surfaced to the model; two-layer unknown-key rejection (provider `additionalProperties:false` + the HOST whitelist in `validate_experiment_config` — the local validator dialect ignores additionalProperties, so the whitelist is the guarantee); validation errors return a guaranteed-valid, firewalled `corrected_example` (`experiments.py` lines 60-390) | M5.3 Slice 5 (`ce478db`); M5.2 smoke failure 5: the lead's correct turn-4 escape attempt failed because it guessed `target_language` / `allow_homophones` / `max_runtime_seconds` against an opaque `{"type":"object"}` schema (artifact `d3eccab14a40`) | EVID (structure) + INV for the `language` firewall key (GT-3) | The one observed GOOD escape behavior (alternate search) fails on schema opacity; unknown keys silently alter solver runs | **Hard** validation; identical schema exposure through MCP `experiment_submit` |
| EXP-2 | Semantic `dedup_key` (type, language, effective cipher, canonical config): duplicate submissions vs pending/running/completed records return `already_queued`; `resubmit=<id>` re-runs an orphaned/failed record with `superseded_by` stamped (`experiments.py::dedup_key`, `dispatch_experiment_submit`) | M4 (`96f2fbe`) A9 | EVID + resource | Identical background solver runs stack up | **Hard** exact-duplicate; telemetry for near-duplicates |
| EXP-3 | Queue lifecycle honesty: per-turn `poll` harvest; finalize does one last no-promotion poll, flips pending/running → `orphaned` with reason, guarded env restore; loaded records on resume are orphaned — a fresh queue never adopts another process's threads (`experiments.py::ExperimentQueue`; `loop_v3.py` finalize block) | M4 A9/F6 (`96f2fbe`) | EVID | Phantom "running" experiments after crashes; leaked env overrides | **Hard**; this rule is the §9.1 single-owner concurrency rule in miniature |
| EXP-4 | `repair_exhausted` ↔ experiment linkage: a submit while exhausted stamps `pending_experiment_id` on the saturation entry; the menu then offers collect (`loop_v3.py::_dispatch_tool`; `context.py::_repair_exhausted_menu`) | Slice 2/5 workflow half (`f371bba`) | EVID | Exhausted state offers no visible way out | **Active** |

## 9. Provider/process resilience (RES)

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| RES-1 | Bounded 429 retry at all three send sites: 3 retries at 2/5/10 s stretched to parsed Retry-After (clamped 60 s); `insufficient_quota` fails fast; persistent limits end the run honestly with fallback preserved (`agent/model_provider.py::call_with_rate_limit_retry`) | `d7e1376`: an upstream OpenRouter 429 (kimi-k3, Retry-After: 1) killed a whole run on turn 2 (K3 incident) | EVID (process reliability) | Single transient 429s kill paid runs | **Hard** for server-side sends (verify); client lead sends are the client harness's concern — convergent-evolution item the proposal expects the mature harness to own (§2.1) |
| RES-2 | Truncation retry: a tool-less turn with output == cap retries with doubled budget (8192→16384→32768, max 2/run); reasoning effort deliberately NOT capped (`loop_v3.py` turn loop) | `10af31f`: kimi-k3 spent the entire 8192-token budget thinking on turn 1; the single `if not tool_uses` exit declared exhaustion (K3 incident, borg_0109v) | EVID (process reliability) | One reasoning-heavy turn ends a run | Client-harness-owned in MCP; **telemetry** only server-side (n/a to server) |
| RES-3 | Text-only-turn nudge: ONE evidence-log nudge before the honest `exhausted` terminal (`loop_v3.py`) | `10af31f` | POL (a nudge, already advisory-shaped) | Narrating-without-acting turns end runs silently | Client-harness-owned; **telemetry** |
| RES-4 | R5 pairing: interrupts and terminations synthesize one tool_result per outstanding tool_use (`stopped` / `run_terminated`) so the recorded exchange never desyncs — an unpaired exchange 400s at resume on both providers (`loop_v3.py`; `episodes.py`) | M1/M2 R5; reiterated in Slice 1 (paired `budget_exhausted` results) | INV (artifact/resume validity) | Artifacts unusable for resume/replay | **Hard** — MCP analog is the atomic `expected_revision` commit protocol (§3.2) |
| RES-5 | Lead-tool availability gate: tool_uses not in the CURRENT turn's definition set are blocked (`lead_tool_not_available`) rather than executed from stale context (`loop_v3.py::_dispatch_tool` head) | M5.2 Slice 2 (`f543d28`): "historical exchanges remain context only and do not restore legacy tool authority" | INV (dispatch hygiene) | Stale-context tool calls execute hidden operator tools | **Hard** (server rejects unknown tools by construction) |

## 10. Candidate, compare, and renderer contract (CMP)

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| CMP-1 | Compare hash binding: dispatch records `comparison_binding` (per-branch hashes, winner, winner_hash, turn); `_fresh_compare_winner` demands ALL bound hashes still current, winner active, verdict not reject — else the record is dead for fallback (`loop_v3.py`) | M5.1 Slice E (fresh compare records, `55404f5`); §11 seed "needed a hash-bound terminal fallback" | EVID | Stale comparisons pick fallback winners over changed content | **Hard** hash-bound storage (`comparison_record`, §3.5) |
| CMP-2 | `winner` conflates "best partial" with "solved winner": the compare schema's nullable `winner` caused the host to discard a first-ranked branch when the worker returned `winner=null` (proposal §2.4, the Borg null-mask contract bug) | Live M5.3-week Borg artifact investigation; fix pre-registered for MCP (§3.5, §11): `comparison_record` records `best_partial` SEPARATELY from `accepts_as_solution`, both hash-bound | POL (contract policy, per §11) | Rankings of honest partials keep being discarded | **Diverge in v1**: split fields, bind both — a ledger-recorded contract change, not a silent one |
| CMP-3 | Pinned canonical renderer: all content hashes computed via `DECODED_TEXT_RENDERER_ID` (`_decoded_text_for_panel`); snapshot hashing restores into a scratch workspace and renders canonically (`loop_v3.py::_snapshot_content_hash`; `loop_shared.py`) | `e725d27`: the host renderer erased canonical word boundaries on a null-mask candidate, so the verifier judged a fused string (proposal §2.4 — a HOST bug the hash/renderer pinning now guards) | INV (hash freshness is only meaningful under a stable renderer) | Attestations/comparisons bind to phantom renderings | **Hard** |
| CMP-4 | Branch-role de-conflation: four derived roles (best_scored / workflow / latest_installed / declared_or_selected), never stored, recomputed on resume, stamped on the artifact (`loop_v3.py::_compute_branch_roles`) | M5.3 Slice 7 (`2ca2c6d`); M5.2 smoke failure 7 (telemetry displayed the scalar-best branch as `branch`) | EVID (telemetry) | Analyzers conflate focus, score, and outcome again | **Telemetry**; feeds the §3.5 candidate-portfolio labels |

## 11. Context assembly and state mechanisms (CTX)

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| CTX-1 | Rebuilt-context rule: the lead view is rendered purely from `InvestigationState` every turn; loading serialized state IS the resume path; language resolves from state on resume (R8a) (`context.py::build_lead_context`; `loop_v3.py` head) | M1 core design (`agent_v3_m1_spec.md`); the proposal relocates exactly this into the server (§3.2) | INV (state model) | Transcript becomes load-bearing; resume and cross-client continuation break | **Hard** — this is `investigation_status` (§4.3) |
| CTX-2 | Deterministic per-section caps + global budget clamp; whole-exchange drops only (F1 — splitting a tool_use from its result 400s on the Responses API); copied messages so sessions cannot mutate durable state (R8b) | M1/M2 reviews | EVID + provider validity | Context bloat (proposal risk 8) and provider 400s | **Active**: paginated/compact status packets (§9.8) |
| CTX-3 | External/benchmark context is its own stable rendered section, never a scroll-away evidence entry, never ground truth (R3) (`loop_v3.py` preflight block) | M1 R3 | EVID | Externally supplied context silently vanishes after ~6 turns | **Active** |
| CTX-4 | Evidence/board/ledger surfacing: hypothesis cards, episode ledger, experiment queue, readings, evidence log, rotating full-decode window rendered every turn (`context.py` renderers) | M1-M4; §3.3 class-2 items "hypothesis coverage, stale evidence, pending experiments, candidate diversity, research history" | EVID | The §4.3 self-briefing loses its substance | **Active** — becomes the `investigation_status` brief |
| CTX-5 | Model-variant single writer: only `act_set_model_variant` changes it; lead mirrors to state; episodes/experiments inherit; serialized for resume (`loop_v3.py::_dispatch_tool` mirror) | Model-variant registry work (`model_variant_registry_spec.md`) | EVID | Episode/experiment scoring silently diverges from the lead's model selection | **Active** (`act_set_model_variant` is in the §3.5 surface) |

## 12. v2-inherited executor policies active in v3 workers (POL)

| ID | Control + enforcement point | Original failure mode → provenance | Class | If removed | MCP v1 form |
|---|---|---|---|---|---|
| POL-1 | Context-family prior: benchmark context naming a cipher family blocks off-family search tools (`context_cipher_family_mismatch`) unless `override_context_cipher_family=true` with a ≥40-char rationale; overrides recorded with a compare-separately warning (`tools_v2.py` ~line 3440, `context_family_overrides`) | Commit `d41d444` (2026-05-02, "Add fast polyalphabetic agent tooling"; no body). §11 seed rationale: "agents wandered into unrelated families despite explicit context". **UNKNOWN-PROVENANCE (artifact)**: the specific motivating run/artifact is not recorded in the commit or a spec | POL with budget consequences (§11) | Budget wandering on context-contradicted families; note the M5.2-smoke lesson cuts BOTH ways (the lead's escape attempt was correct) | **Strong advisory + coverage telemetry**; retain a bounded-spend guard until ablated (§11) |
| POL-2 | Finalize-phase guard (neutralized v3 form): re-running heavy search on a branch already rated readable requires an explicit `justification`; the v2 coercive text and tool suggestions removed (F8) (`tools_v2.py::NoGatesPolicy.finalize_guard`) | v2-era guard; neutral rewording in M2 A4/F8 (`e699773`). **UNKNOWN-PROVENANCE (artifact)** for the original v2 introduction | POL | Readable candidates overwritten by late searches — recoverable via branches, so spend/annoyance, not integrity | **Advisory** with policy id |

---

## Summary counts

Total enumerated controls: **66 rows** (GT 3, DECL 8, REP 7, SAT/DUP 8, WF 6,
BUD 7, EPI 7, EXP 4, RES 5, CMP 4, CTX 5, POL 2).

Counting each row once by its PRIMARY classification (straddles carry their
secondary class in-row — DECL-6, REP-3, REP-4, EPI-2, SAT-3, REP-5, WF-2):

| Classification | Count | Rows |
|---|---:|---|
| Invariant (INV) | 22 | GT-1, GT-2, GT-3, DECL-1, DECL-2, DECL-5, DECL-6, DECL-8, REP-1, REP-3, REP-4, BUD-1, BUD-2, BUD-3, BUD-7, EPI-1, EPI-2, EPI-6, RES-4, RES-5, CMP-3, CTX-1 |
| Evidence mechanism (EVID) | 31 | DECL-3, DECL-4, DECL-7, REP-2, REP-5, REP-6, SAT-1, SAT-2, SAT-3, SAT-4, DUP-1, DUP-2, DUP-3, DUP-4, BUD-6, EPI-3, EPI-4, EPI-5, EPI-7, EXP-1, EXP-2, EXP-3, EXP-4, RES-1, RES-2, CMP-1, CMP-4, CTX-2, CTX-3, CTX-4, CTX-5 |
| Investigative policy (POL) | 13 | REP-7, WF-1, WF-2, WF-3, WF-4, WF-5, WF-6, BUD-4, BUD-5, RES-3, CMP-2, POL-1, POL-2 |

MCP v1 form totals:

- **Hard / active: 51** — every INV row, every EVID row except the four
  telemetry-class ones below, plus BUD-4 and BUD-5 (hard caps whose CHANGE
  path, not enforcement, is policy-shaped).
- **Advisory: 7** — REP-7, WF-1, WF-2, WF-4, WF-5, POL-1, POL-2.
- **Pre-registered contract divergences: 2** — WF-6 (scalar focus → diverse
  candidate portfolio), CMP-2 (`winner` → `best_partial` +
  `accepts_as_solution` split).
- **Telemetry-only: 5** — DUP-2, DUP-3, RES-2, RES-3, CMP-4.
- **Structural (not a runtime gate in MCP): 1** — WF-3 (the exposed tool
  surface itself).

This matches §3.3's named divergence set exactly: everything softened in v1
is on the pre-registered list (scalar-selected workflow focus, phase/action
restrictions, verifier-to-route thresholds, `next_steps`/menus, family
sequencing, the compare-winner contract split); nothing else changes form.

## UNKNOWN-PROVENANCE register

1. **WF-6 (scalar-selected workflow focus)** — the mechanism is traceable to
   the v2 auto-declare fallback lineage (`agent/loop_shared.py::
   _best_branch_for_auto_declare`) and the §11 seed rationale, but no specific
   motivating artifact/test for making the scalar the FOCUS selector (as
   opposed to the fallback selector) is recorded anywhere. Do not invent one;
   the §6 experiment's candidate-retention metric is the first controlled
   evidence this control will ever have.
2. **POL-1 (context-family prior)** — introducing commit `d41d444` has no
   body and no linked artifact; the failure mode is asserted only by the §11
   seed. Treat the bounded-spend guard as unvalidated-but-preserved; its
   ablation belongs in the §6 wrong-family-lure case, which was designed for
   exactly this control.
3. **POL-2 (finalize-phase guard)** — the v2-era introduction predates the
   current provenance discipline; only the M2-era neutralization (A4/F8) is
   documented.

All other 63 rows trace to a named commit, spec section, test, or artifact.

## Coverage check against §6 minimum

Required by §6: context-family discipline (POL-1), verification gating
(DECL-1..8, GT-2), repeated-call suppression (DUP-1..4, SAT-4, EXP-2),
repair acceptance (REP-1..7), saturation (SAT-1..3), workflow routing
(WF-1..5), scalar candidate focus (WF-6). All covered.
