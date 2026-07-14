# Spec: Agent Loop v3 — Milestone M4 (Experiment Queue)

Parent design: `docs/specs/agent_v3_design.md` (C5; the M4 milestone paragraph;
amendments **A5, A9 binding here**; A7 for accounting, C8 for artifacts). M2 is
landed at `e699773`; M3 (`agent_v3_m3_spec.md`) may land first — integrate
with whatever `V3_LEAD_TOOL_DEFINITIONS` assembly is then current. The
model-variant-registry slice (incl. its F1a thread-local fix round, Scope) is
a REQUIRED predecessor. Spec author: Fable; implementer: coding sub-agent;
Fable-reviewed before implementation.

Conventions (binding): baseline is **1086 passed / 1 skipped** (`PYTHONPATH=src
.venv/bin/python -m pytest tests/ -q`) as of `e699773` — re-record before
starting (report actual); no pre-existing test may change outcome. Cited
`file:line` may be stale — locate constructs by the quoted identifiers/strings,
never by line. **No commits.** Report deviations (differences, skips,
contradictions found, with rationale). Acceptance compute is local solver CPU;
no live LLM spend beyond an optional informational lead run (OpenAI key at
`.decipher_keys/openai_api_key`, under **$1**, report actual). Anthropic has no
credits — fakes only.

## Scope

New `src/investigation/experiments.py` (experiment registry, queue, lead tools
`experiment_submit`/`experiment_collect`); lead-loop and context wiring; A9
orphan/resume semantics in `state.py`; the A5 worker-budget arbiter. **v2
untouched**: no change to `TOOL_DEFINITIONS`, `VALID_TOOL_NAMES`,
`automated/runner.py` behavior, or any v2 loop path — with ONE named
exception, the F1a entry precondition (BLOCKING): the model-variant-registry
slice's `_ACTIVE_MODEL_VARIANT` must become a `threading.local` slot, landing
in THAT slice's fix round before M4 work starts. `threading.local` is
SUFFICIENT (reviewer-verified): every deep consumer resolves the model path
on the thread that entered `run_automated`, and process-pool seed workers
take an explicit `model_path`. The v2 synchronous long-running tools
(`search_automated_solver`, the transform searches, the Quagmire shotgun)
stay available to the v3 lead unchanged; the queue is the sanctioned overlap
path, nudged by prompt and tool descriptions, not forced (M6 revisits).

## Part 1 — Experiments as pure functions + registry (C5 + A5)

- **Purity contract (A5)**: an experiment is `(cipher: CipherText,
  branch_snapshot: dict, config: dict) -> result dict`. The snapshot is a deep
  copy of `_serialize_branch` output (the M2 F5 pattern from
  `episodes._build_episode_workspace`). The runner receives NO executor, NO
  workspace, NO state, and never mutates its inputs; results integrate on the
  lead thread only.
- **`EXPERIMENT_TYPES` registry**: type → `{config_schema, config_defaults,
  runner, description}`. Config validated with
  `episodes.validate_against_schema` PLUS an unknown-key whitelist check
  (structured error; also a firewall surface). `register_experiment_type(name,
  entry)` is the test seam (mirrors `sessions.register_session_builder`).
- **v1 type: `automated_solver`** — the `run_automated` surface. Allowed config
  keys (subset of `run_automated` kwargs, same enums/defaults):
  `homophonic_budget`, `homophonic_refinement`, `homophonic_solver`,
  `transform_search`, `transform_search_profile`,
  `transform_search_max_generated_candidates`, `cipher_system`, and
  `model_variant` (F1b; validation in Part 3). The `transform_search` enum
  EXCLUDES `"promote"` (F5): it reads prior artifacts from disk — a purity
  break — and raises when its companion arg is absent. Transform screens
  (`transform_search="screen"|"rank"|"full"`) and null-mask bakeoffs
  (`homophonic_refinement="null_masks"`) are THIS type via its existing options
  — no separate types; `run_automated` routes both internally. The Quagmire
  shotgun is a different surface (`analysis.polyalphabetic_fast`) — a follow-on
  registry entry, out of M4 scope.
- **Runner implementation**: deep-copy the stored snapshot AGAIN before
  `_restore_branch_into` (F5 — the record's copy must stay pristine; restore
  shallow-copies nested metadata), reconstruct a one-branch throwaway
  `Workspace` over `cipher`, derive `workspace.effective_cipher_text(branch)`
  exactly as `_tool_search_automated_solver` does, set the registry's
  thread-local active variant to `config["model_variant"]` for the duration
  (F1a makes this per-thread-safe; clear on exit), call `run_automated(...,
  ground_truth=None, cipher_id="experiment")`; return `{status, solver,
  error_message, elapsed_seconds, key (str→int), final_decryption, steps}` from
  the `AutomatedRunResult` (artifact `key`/`steps` copied out; nothing else).
  Lazy-import `automated.runner` inside the function.
- **Dedup key (A9)**: `sha256` of `json.dumps({type, language, effective: token
  list + word lengths of the DERIVED effective cipher, config:
  defaults-applied}, sort_keys=True)`. Hashing the derived effective text (not
  branch name/key) makes dedup semantic: same transform state + same config →
  same experiment (the branch key is ignored on this path). The
  defaults-applied config includes the resolved `model_variant` (F1b), so
  dedup is variant-aware: two experiments differing only by model must NOT
  dedup.
- **Record schema** (plain dicts in the existing empty `state.experiment_queue`
  field): `{experiment_id (uuid12), dedup_key, type, branch (source name),
  snapshot, config (defaults-applied), note, status:
  "pending"|"running"|"completed"|"failed"|"orphaned", submitted_turn,
  started_at, completed_at (wall-clock timestamps — the F8 overlap evidence),
  elapsed_seconds, inner_workers, result, error, summary (≤400 chars, built
  lead-side), collected: bool, null_mask_session_id, installed_as,
  superseded_by, orphan_reason}`. `result` holds the runner dict verbatim
  (artifact-bound); everything shown to the MODEL passes
  `tools_v2._strip_packet_keys` over `steps` first (the v2 handler's "must
  never reach the model" discipline).

## Part 2 — Async execution + worker-budget arbiter (A5)

- **Adjudication: daemon threads, not a process pool.** (a) `run_automated`'s
  CPU fan-out is already process-based — `_run_homophonic_zenith_native` opens
  its own `ProcessPoolExecutor` sized by `_homophonic_parallel_seed_workers()`
  — so the queue slot mostly waits; a process pool would nest pools (the A5 N×N
  warning) and add a pickle boundary for no CPU gain. (b) In-process caches
  (lazy ngram models, the ~47 MB zenith binary, pattern dictionaries) stay warm
  and shared. (c) Plain `threading.Thread(daemon=True)` per running experiment
  — NOT `ThreadPoolExecutor` (its exit hook would add yet another join).
  Honest exit note (F3, module docstring): daemon-ness does not make exit
  free — the inner `ProcessPoolExecutor`'s NON-daemon management threads are
  joined at interpreter exit regardless, so a `run_ended` orphan can delay a
  non-interrupt exit until its solver finishes. Ctrl-C exits promptly because
  the terminal SIGINTs the whole process group (seed children die) — not
  because of daemon threads. Orphaning (Part 4) is a STATE transition only.
  Risk note, not a work item: inner pools spawned from a worker thread are
  safe under spawn (macOS default); flag fork flakiness on Linux.
- **Single-writer discipline**: ALL record mutations, env changes, and thread
  starts happen on the lead thread — in `experiment_submit`,
  `queue.poll(state)`, and finalize. A worker thread writes only into a private
  lock-guarded result box in the `ExperimentQueue` object (`{experiment_id:
  (result | exception, elapsed)}`); `poll` harvests the box, flips
  running→completed/failed, stamps `elapsed_seconds` and `summary`, appends an
  `experiment_complete` evidence entry, and promotes pending→running while
  slots are free. The queue object is NEVER serialized — constructed per
  `run_v3` call over the state records.
- **Arbiter math (A5, precise)**:
  - `W = int(DECIPHER_PARALLEL_WORKERS)` if set, else `max(1, cpu_count - 1)`
    (matching `_homophonic_parallel_seed_workers`'s auto-size).
  - `S = clamp(int(DECIPHER_EXPERIMENT_SLOTS or 2), 1, W)` — slots.
  - `I = max(1, W // S)` — inner solver parallelism per experiment. `S × I ≤ W`
    by floor division. STATIC split (not W // running): predictable, race-free;
    an already-spawned pool cannot shrink anyway. Documented trade: a lone
    experiment uses only `I`.
  - **Mechanism**: while `running > 0`, set the SPECIFIC override vars
    `DECIPHER_HOMOPHONIC_PARALLEL_SEEDS`, `DECIPHER_NULL_MASK_THREADS`,
    `DECIPHER_TRANSFORM_RANK_THREADS` to `str(I)` — these are what the runner
    helpers read at call time (quote-locate
    `_homophonic_parallel_seed_workers`, `_null_mask_batch_threads`). A var the
    user explicitly set is NOT overridden (user wins; record
    `arbiter_overridden: false` provenance + a warning in the submit result).
    Save prior values (incl. absent) on the 0→1 running transition — the env
    set MUST complete before `Thread.start()` in `experiment_submit` (F2, pin
    by test); restore on 1→0 inside `poll`/finalize, guarded per Part 4 F2.
    Intentional side effect: a busy-queue synchronous lead solver call also
    reads `I` — the global budget behaving.
- **Synchronous mode (MANDATORY)**: `ExperimentQueue(synchronous=True)`
  (constructor arg; default from env `DECIPHER_EXPERIMENT_SYNC` truthy per the
  `_episode_transcripts_enabled` value set). Submit runs inline on the lead
  thread through the SAME record transitions; the arbiter env machinery is a
  no-op (matches today's v2 sync tool behavior). Tests default to sync except
  the concurrency tests.

## Part 3 — Lead surface (tools, dispatch, context)

- **Two tool defs in `experiments.py`** (v2 never sees them). Assembly
  location (F7): the FINAL `V3_LEAD_TOOL_DEFINITIONS` list is assembled in
  `loop_v3` (or `experiments.py` re-exports the combined list) — never append
  into the constant defined in `episodes.py` (import cycle):
  - `experiment_submit(type, branch, config?, note?, resubmit?)` — validates
    type/branch/config; snapshots the branch. `model_variant` (F1b) is
    validated at SUBMIT time on the lead thread via
    `model_registry.resolve_language_model` (`ModelVariantError` → structured
    error listing available slugs), defaults to the lead executor's current
    `_model_variant`, and is stamped into the defaults-applied config (so
    dedup is variant-aware). Dedup check: same-key `completed` → return its id
    + summary with `"deduplicated": true`, create nothing (the cheap
    resubmission path); same-key `pending|running` → return that id with
    `"already_queued": true`; same-key `orphaned` → new record, stamp the old
    one's `superseded_by`. `resubmit=<experiment_id>` (F6): creates a NEW
    record — fresh id, copied spec+snapshot from the named orphaned/failed
    record, old record stamped `superseded_by` — after first dedup-checking
    against same-key completed records; works after resume even if the source
    branch was deleted; unknown id → structured error. Returns
    `{experiment_id, status, slots}`.
  - `experiment_collect(experiment_id?, install?, as_name?)` — no id: runs
    `poll`, returns queue status (one line per non-collected record) plus full
    summaries of newly completed ones. With id: returns the result packet —
    `{experiment_id, type, status, config, elapsed_seconds, inner_workers,
    solver, decoded preview (≤800 chars), route/primary step (packet-stripped),
    error}` — and marks `collected`. `install=true` (completed runs only):
    deep-copy the stored snapshot, restore under a fresh name (default
    `exp_<id6>_<branch>`, collision-suffixed like `_dispatch_episode_install`),
    `set_full_key` with the result key, `created_iteration` = current turn, and
    mirror the v2 null-mask metadata block (quote-locate `"null_mask_selected"`
    in `_tool_search_automated_solver`) when the result carries a completed
    `search_null_masks` step. The SOURCE branch is never mutated (unlike the v2
    in-place tool).
  - **Null-mask finalist session**: created lead-side at FIRST collect of a
    result with a completed `search_null_masks` step — reuse
    `executor._new_null_mask_session` + `_null_mask_finalist_review`
    (state-owned store, so review/rate/install triplets work as today);
    `null_mask_session_id` on the record prevents duplicates. The collect
    packet MUST surface it (F7): `null_mask_search_session_id`, a capped
    initial finalist review, and the v2 suggested-next-tools triplet
    (`search_review_null_mask_finalists`, `act_rate_null_mask_finalist`,
    `act_install_null_mask_finalists`) — else the session exists but the lead
    cannot find it. Finalist-install source-branch rule (F7): at session
    creation, `source_branch` = the record's source branch if it still
    exists, else the freshly installed `exp_` branch when `install=true`;
    document the constraint that installing finalists from the session
    requires that branch to be live.
- **Dispatch + turn-start poll**: `loop_v3._dispatch_tool` routes
  `experiment_*` before `executor.execute` (the `episode_*` pattern), R5
  interrupt pairing unchanged; implementations live in `experiments.py` as
  functions taking `(queue, state, workspace, executor, args, turn)`. The lead
  loop also calls `queue.poll(state)` each turn next to
  `hypothesis_board.sync_from_workspace`, so completions surface as evidence +
  context without the lead asking.
- **Context**: new `context._render_experiment_queue(state)` after the episode
  ledger, rendered when NON-COLLECTED records exist (F7): one line per such
  record — id, type, status, source branch, config digest (non-default keys),
  elapsed, summary for completed ones; capped ~1200 chars (module constant).
  The v3 system prompt's "Delegating" paragraph gains two sentences: long
  solver runs go through `experiment_submit` and run in the background while
  you keep working; check and adjudicate with `experiment_collect`.
- **Episodes stay experiment-free by construction**: `experiment_*` is not in
  `VALID_TOOL_NAMES`, so `_validate_episode_toolset` rejects it (pin with a
  test). `EPISODE_EXCLUDED_TOOLS` unchanged — the queue is where those
  workloads now go (lead-facing note only).

## Part 4 — Interrupt, orphaning, resume (A9)

- **Finalize orphaning** — ordering pinned (F6): at loop finalize (every exit
  path — declared, exhausted, error, interrupted), (1) one last `poll`
  (harvest already-completed results — this is what makes post-resume
  resubmission cheap), (2) flip every remaining `pending|running` record to
  `orphaned` with `orphan_reason` = `"interrupted"` when `artifact.status ==
  "stopped"` else `"run_ended"`, (3) env restore — ALL before
  `artifact.experiments` and `artifact.investigation_state` are set.
  Orphan-alive guard (F2): step (3) restores the overrides ONLY if no orphan
  thread `is_alive()`; otherwise leave them in place (a live solver is still
  reading them) and record a warning in the artifact.
- **Load transition**: `InvestigationState.from_artifact_dict` normalizes
  loaded records — any `pending|running` becomes `orphaned(loaded)`.
  Implemented inline in `state.py` (a helper in `experiments.py` would
  import-cycle: experiments imports `_serialize_branch` from state). Completed
  records load verbatim: results, sessions, and dedup keys stay live.
- **Resume-by-resubmission**: submitting the same spec after resume dedups to
  completed records (zero compute); `resubmit=<id>` re-runs an orphaned record
  from its stored snapshot. No reattachment machinery.

## Part 5 — Budget/ledger + artifact (A7, C8)

- Experiments are token-free solver compute: NO `BudgetEntry` rows (those are
  model calls). Instead `InvestigationState.budget_by_category()` appends,
  additively, one `experiment:<type>` bucket per type present in the queue —
  `{calls: N (records that reached running), elapsed_seconds: summed over
  completed/failed records only (F6), token fields 0, cost_usd 0.0}`.
  Wall-clock/config/provenance live on the records.
- Artifact (additive): `RunArtifact.experiments: list[dict]` = the queue
  records at finalize (like `episodes`); `scripts/inspect_artifact.py` renders
  a minimal experiments table (id, type, status, elapsed, summary) — the
  standing C8 requirement.

## Part 6 — Firewall

`run_automated` is called with `ground_truth=None` hardcoded; the config
whitelist rejects smuggled keys. Extend `tests/test_ground_truth_firewall.py`:
for a benchmark-backed cipher, (a) the rendered queue section, (b) an
`experiment_submit` result, (c) an `experiment_collect` packet (incl. an
installed null-mask case), and (d) the raw record dicts all pass
`assert_no_ground_truth_leak` (design C6: experiment specs and results are
leak-checked surfaces).

## Part 7 — Tests

New `tests/test_experiments.py`; extend `test_loop_v3.py`,
`test_investigation_state.py`, `test_ground_truth_firewall.py`. Cover every
MUST above; called out:

- **Purity/isolation**: the runner mutates nothing — nested metadata/pipeline
  of the source branch byte-identical after a run; registry validation errors
  (unknown type/config key, enum violation) are structured, never raised.
- **Arbiter math**: unit-test `W/S/I` across (env set/unset, S>W, W=1);
  overrides set on 0→1 and restored on 1→0 (monkeypatched environ); an
  explicitly-set user var is not clobbered, provenance says so. F2 pins: the
  0→1 env set completes BEFORE `Thread.start()`; finalize with a still-alive
  fake orphan leaves the overrides in place + artifact warning, and restores
  when no orphan is alive.
- **Concurrency + sync fallback**: a registered fake experiment type whose
  runner blocks on a `threading.Event` — submit two, both `running`, a third
  stays `pending` at S=2; release, `poll` harvests both on the lead thread and
  promotes the third; the same script with `synchronous=True` passes with
  identical record shapes. Events only — no sleeps.
- **Variant concurrency (F1c)**: two fake-blocked experiments submitted with
  DIFFERENT `model_variant` values; a monkeypatched
  `resolve_language_model` records thread→variant — assert each worker thread
  saw its own variant and the records' configs (and dedup keys) differ.
- **Rust-pool verify (F4)**, a report item not a unit test: determine whether
  the Rust extension holds a global Rayon pool sized by the FIRST `threads=`
  value it sees (a shared-pool sizing wrinkle, not corruption); the
  acceptance-3 overlap demo doubles as this smoke — report findings in the
  deviations report.
- **Orphan/resume round-trip**: serialize state mid-run (one running, one
  pending, one completed-uncollected) → `from_artifact_dict` → running/pending
  are `orphaned(loaded)`, completed intact; resubmitting the completed spec
  returns `deduplicated: true` with zero runner invocations (counting fake);
  `resubmit` of the orphaned id re-runs from the stored snapshot after the
  source branch is deleted. Finalize orphaning: `run_ended` AND `stopped`.
- **Scripted lead flow (fakes, no network)**: a `ScriptedSession` lead submits
  two fake experiments, does other work for two turns while the queue section
  renders both running, then collects both, installs one, declares. Assert:
  `experiment_complete` evidence entries; queue section content;
  `experiment:<fake>` bucket in `budget_by_category`; installed branch name;
  `artifact.experiments`.
- **Pins**: `TOOL_DEFINITIONS` length unchanged;
  `executor.execute("experiment_submit")` → unknown-tool error; episode toolset
  validation rejects `experiment_*`; collect output has packet-stripped steps
  (assert on a result carrying a `packet` key).

## Acceptance (compute, report)

1. Full suite green: re-recorded baseline (1086 / 1 at `e699773`; report
   actual), zero regressions, plus new tests. Report final counts.
2. Scripted lead flow (Part 7) passes; report the assertion list.
3. **Real overlap demonstration** (no LLM): a script (under `scripts/` or a
   marked slow test) on a synthetic cipher queues one transform screen
   (`config={"transform_search": "screen"}`) and one anneal (`config={}`,
   homophonic path) concurrently at S=2, then the same pair sequentially in
   sync mode. The GATE (F8) is INTERVAL OVERLAP: the records'
   `started_at`/`completed_at` intervals must overlap. The wall-clock ratio
   is reported informationally, not gated; if any hard ratio is asserted,
   keep it tolerant (`concurrent / sequential < 1.15`) or configure the
   anneal with `seed_count < I` so the win is structural. Report both
   wall-clocks, the ratio, `W/S/I`, and per-experiment elapsed.
4. Optional, informational (not gating): one v3 lead run on a synthetic
   (`--provider openai --model gpt-5.5`) reporting whether the lead used the
   experiment tools unprompted; spend under $1, report actual.

## Out of scope

Verify episodes, attestation, declaration wiring (M5/A6); any v2 behavior
change, incl. removing/redirecting the synchronous long-running lead tools (M6
decides) — except the F1a thread-local precondition, which lands in the
variant-registry slice, not here; Quagmire-shotgun or any second experiment
type; process-pool execution and cross-process cancellation; experiments
inside episodes; v2-artifact → v3-state adapter (M6).

## Post-review amendments (BINDING — from the Fable spec review, READY WITH AMENDMENTS)

All eight findings are folded into the Parts above; this section records the
mapping and is normative where wording differs.

1. **(F1, BLOCKING) Model-variant threading.** (a) Entry precondition in
   Scope: `_ACTIVE_MODEL_VARIANT` becomes a `threading.local` slot via the
   variant-registry slice's fix round BEFORE M4 starts (reviewer-verified
   sufficient: deep consumers resolve the model path on the thread that
   entered `run_automated`; seed workers take explicit `model_path`). (b)
   `model_variant` config key: Part 1 whitelist + Part 3 submit-time
   lead-thread validation via `model_registry.resolve_language_model`
   (`ModelVariantError` → structured error listing slugs), default from the
   lead executor's `_model_variant`, stamped into defaults-applied config —
   dedup is variant-aware. (c) Part 7 variant-concurrency test.
2. **(F2) Orphan-alive env guard.** Part 4 step (3): restore overrides only
   if no orphan thread `is_alive()`, else leave them + artifact warning; Part
   2 pins that the 0→1 env set completes before `Thread.start()`.
3. **(F3) Honest daemon-thread rationale.** Part 2(c) rewritten: the inner
   pool's non-daemon management threads join at interpreter exit, so
   `run_ended` orphans can delay non-interrupt exit; Ctrl-C works via
   process-group SIGINT, not daemon-ness.
4. **(F4) Rayon shared-pool verify.** Part 7 report item: does the Rust
   extension hold a global Rayon pool sized by the first `threads=` value?
   The acceptance-3 demo doubles as the smoke; findings go in deviations.
5. **(F5) Purity guards.** `"promote"` excluded from the `transform_search`
   enum (disk reads + companion-arg raise); the worker deep-copies the stored
   snapshot again before `_restore_branch_into`.
6. **(F6) Ordering + resubmit + bucket precision.** Finalize order pinned
   (poll → orphan flips → guarded env restore, all before the artifact
   fields); `resubmit` creates a NEW record (fresh id, copied spec+snapshot,
   old stamped `superseded_by`, dedup-checked against completed first);
   `elapsed_seconds` buckets sum completed/failed records only.
7. **(F7) Discoverability + wiring pins.** Collect packet surfaces
   `null_mask_search_session_id` + capped initial review + the v2
   suggested-next-tools triplet; finalist-session `source_branch` rule
   (record's source branch if live, else the installed `exp_` branch on
   `install=true`, constraint documented); final tool-def assembly lives in
   `loop_v3` (or an `experiments.py` re-export), never appended into
   `episodes.py`'s constant (import cycle); queue section renders when
   non-collected records exist.
8. **(F8) Overlap gate.** Acceptance 3 gates on interval overlap of
   `started_at`/`completed_at` (fields added to the record schema);
   wall-clock ratio informational; any hard ratio tolerant (`< 1.15`) or the
   anneal sized `seed_count < I`.

## Deliverables

Files changed; suite counts (baseline vs final); scripted-flow assertion list;
overlap-demo table (interval overlap shown, concurrent vs sequential
wall-clock, W/S/I); the F4 Rayon-pool finding; deviations report. No commits.
