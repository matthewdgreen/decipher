# Improvement Program Plan

Status: opened 2026-07-13 from a full architecture review of the agent loop,
tool surface, solver stack, and Copiale research scripts. This is the
cross-cutting engineering/capability program that sits alongside
`docs/copiale_generalization_plan.md` (Copiale capability track) and
`docs/experimental_consolidation_plan.md` (promotion ledger). When an item
here lands, update the ledger and this file together.

## Goal

Turn the strongest results currently stranded in research scripts —
especially the Copiale word-hypothesis and multipage shared-key repair work —
into production runner/agent capabilities, while fixing the evaluation,
cost, and robustness debts that make every future benchmark number less
trustworthy or more expensive than it should be.

## Priority Overview

| Phase | Title | Why this order | Size | Depends on |
|---|---|---|---|---|
| 0 | Evaluation integrity + hygiene | Small, protects every number collected afterward | S (1–2 days) | ✅ landed `ef4ac9a` |
| 1 | Shared candidate packet + generic finalist sessions | Foundation for promotion, ranking, and v3 episodes | M (3–5 days) | — |
| 2 | **Copiale research promotion** | ~79% basins live in orphaned scripts; solver-side, survives the v3 redesign | L (1.5–3 weeks) | 1 (for 2.4+) |
| 3 | German model + solver objective upgrades | Cheapest accuracy lever; parallelizable with 2 | M–L (1–2 weeks) | — |
| **V3** | **Agent loop v3 redesign** | The v2 loop's failures are architectural; episodes + state-rebuilt context replace the coercion machinery. Design: `docs/specs/agent_v3_design.md` | XL (3–5 weeks, milestoned M1–M6) | 1; 2.5 built v3-shaped |
| 4 | LLM reader (runtime selection) | 4.1/4.2 attack the selection wall on the runner side; 4.3 is subsumed by V3-M5 | M (~1 week) | 1 |
| 5 | Agent loop cost + robustness | Shrunk: telemetry + tool output caps only; the rest is deleted-by-design in v3 | S | — |
| 6 | ~~Boundary actuator consolidation~~ | **Subsumed by V3-M3** (Reading artifact + `hypothesis_apply_reading`); do not execute separately | — | — |

Phases 2 and 3 run in parallel (different files). The V3 redesign is the
centerpiece of the agent-side program: the v2 loop has not produced
successful historical solves, and its failure modes (wrong-basin
declarations, gate bounces, actuator misuse, token bloat) are properties of
the single-long-conversation architecture. V3 proceeds in six milestones
with a v2-vs-v3 bake-off gate (M6) before the default switches and the v2
coercion machinery is deleted. Until M6, v2 receives no new features except
Phase 2's tool exposure, which is built in the v3 composite-action shape
from the start.

## Landing Discipline (applies to every phase)

Each phase lands as its own commit only after implementation, tests, and the
Fable code review (plus the Fable-verification metadata check) are complete.
Pre-existing unrelated WIP is checkpointed separately first. See the
"Claude Code Orchestration Strategy" section in CLAUDE.md.

## Ground-Truth Firewall (applies to every phase)

Benchmark plaintext may be used only for post-hoc grading and offline
calibration after candidates exist. It must never influence candidate
generation, routing, ranking, repair adoption, branch selection, or any
agent tool output. Phase 0.7 adds the regression tests that enforce this;
every later phase adds its new surfaces to those tests.

## Baseline Snapshot (do before Phase 0 lands)

Capture one clean run of each smoke so later phases have a comparison point:

- [ ] Zodiac 408 `zenith_native` (expect ≥99% char).
- [ ] Z340 hidden-transform rank smoke (expect 96.2%).
- [ ] Transform ladder (expect 8/8) and pure-transposition ladder (11 rows).
- [ ] Copiale five-page evidence packet with `null_masks` refinement
      (by-selection baseline: p017 75.3%, p035 61.2%, p052 66.1%,
      p068 54.0%, p084 63.0%).
- [ ] Borg packet + `synth_en_200honb_s6` agentic smoke.
- [ ] One agentic run with per-category token counts recorded (for Phase 5).

---

## Phase 0: Evaluation Integrity + Hygiene

Small isolated fixes; land as one or two PRs.

- [ ] **0.1 Stop reporting give-ups as solves.** `run_v2` auto-declares the
      best branch on `exhausted`/`error` and sets `artifact.status="solved"`
      with `self_confidence=0.0` (`src/agent/loop_v2.py:1632–1666`, status
      set at `:1662`, branch pick at `:1805`). Introduce a distinct status
      (`fallback_declared`) or an `auto_declared: true` artifact field that
      benchmark reporting, `scripts/inspect_artifact.py`, and summary tables
      all surface. Audit every consumer that branches on `status=="solved"`.
      Add a fake-provider test that exhausts iterations and asserts the new
      status.
- [ ] **0.2 Delete `src/agent/state.py`.** `AgentState`/`Checkpoint` are v1
      dead code with zero references; their auto-rollback design is exactly
      what the v2 loop docstring says was removed.
- [ ] **0.3 Close the tool-dispatch footgun.** Dispatch is
      `getattr(self, f"_tool_{name}")` (`src/agent/tools_v2.py:2130`), so the
      internal helper `_tool_tried_for_branch` (`:3024`) is reachable as a
      tool named `tried_for_branch`. Rename it (`_was_tool_tried`), and make
      `execute()` reject any name not present in `TOOL_DEFINITIONS`.
- [ ] **0.4 Deep-copy branch metadata on fork.** `Branch.copy_as`
      (`src/workspace/branch.py:21`) does `metadata=dict(self.metadata)`;
      nested mutables (transform structures, decoded-text blocks) alias
      across forked branches. Use a structured deep copy and add a test that
      mutates nested metadata on a fork.
- [ ] **0.5 Allowlist drift test.** Six hardcoded tool-name sets
      (`FINAL_ALLOWED_TOOL_NAMES` etc., `src/agent/loop_v2.py:158–245`) can
      silently drift from `TOOL_DEFINITIONS`. Add a test asserting each is a
      subset of real tool names.
- [ ] **0.6 Fix the penultimate-preflight double injection.**
      `PENULTIMATE_READING_WORKFLOW_PREFLIGHT` appears to be appended both to
      the prior turn's panel (`loop_v2.py:972`) and as the gate user message
      (`:1227–1239`). Verify with a fake-provider transcript; dedupe if real.
- [ ] **0.7 Ground-truth firewall regression tests** (from TODO). Assert the
      automated runner, benchmark runner, and agent loop never read ground
      truth before a candidate exists: e.g. loader wrapper that records
      access order, or fixtures whose ground truth is poisoned/absent until
      scoring time. This is the enforcement mechanism the firewall currently
      lacks.
- [ ] **0.8 Refresh CLAUDE.md.** Tool count (88, not 32/78), current status
      of "Remaining Challenges", pointer to this plan.

Acceptance: full test suite green; a deliberately exhausted run reports
`fallback_declared`; firewall tests fail if a probe reads ground truth early.

---

## Phase 1: Shared Candidate Packet + Generic Finalist Sessions

This is the "Candidate Scoring Architecture" TODO made concrete. Everything
in Phases 2 and 4 flows through it.

- [ ] **1.1 `src/analysis/candidate_packet.py`.** A `CandidatePacket`
      dataclass with: `candidate_id`, `source` (solver family + generator +
      config hash), `text` (or per-page texts), `page_scope`,
      `solver_native_scores`, `language_features` (the
      `LANGUAGE_QUALITY_FEATURES` dict from
      `src/analysis/language_scoring.py:182`), `validation` (the existing
      `analysis.finalist_validation` block), `provenance` (masks, symbol
      edits, transform pipeline, seed), and optional `ranker` outputs
      (model id, raw, calibrated, component explanation). JSON round-trip
      helpers. Solver-native scores and language-quality scores stay
      separate fields, per the existing plan.
- [ ] **1.2 Adapters for existing menus.** Convert null-mask finalist rows,
      transform finalist rows, and pure-transposition finalist rows into
      packets. Embed the packet under existing artifact rows first
      (`"packet": {...}`) so nothing downstream breaks; migrate readers
      later.
- [ ] **1.3 One finalist-session store.** `tools_v2.py` holds three
      copy-pasted session subsystems with their own counters and
      review/rate/install triplets (`_transform_search_sessions`,
      `_null_mask_sessions`, `_pure_transposition_sessions`,
      `src/agent/tools_v2.py:2100–2112`). Extract a single
      `FinalistSessionStore` keyed by session kind holding packets. Keep the
      existing tool names as thin aliases initially (prompts and tests
      reference them); new candidate sources get sessions for free.
- [ ] **1.4 Branch cards / artifact rows consume packets.** Wherever a
      finalist is displayed (branch cards, review tools, artifact rows), the
      packet is the payload, so a better scorer or an LLM reader can be
      switched on globally without per-source rewiring.

Acceptance: Z340 rank smoke and null-mask five-page packet reproduce
baseline numbers exactly; the three review flows share one implementation;
artifacts carry packets for all three menu kinds.

---

## Phase 2: Copiale Research Promotion (centerpiece)

Template: the null-mask promotion (research probes →
`src/analysis/homophonic_nulls.py` → runner `--homophonic-refinement
null_masks` → agent review tools → ledger row). Follow the same path for the
word-hypothesis and multipage shared-key work. The renaming precedent is
commit `f1b8925` (Copiale-specific identifiers → generic).

**What gets promoted now:** multipage shared-key machinery and
word-hypothesis global-edit repair (the source of the ~79% char-accuracy
basins, e.g. `KUNGER→JUNGER`-class repairs reaching 79.3–79.4%).

**What stays research (per the consolidation ledger):** logogram/codeword
probes, reading-holes, phrase hypotheses, iterative repair tree — their
recognition signal is still too weak. Revisit logogram/reading-holes after
Phase 4, since an LLM rereader may be the missing recognizer.

### 2.1 Extract multipage machinery → `src/analysis/multipage.py`

- [ ] Move from `scripts/research/copiale/run_copiale_multipage_experiment.py`:
      `PageBundle`, `build_combined_cipher`, `project_pages`,
      `project_page_with_sources`, `consensus_from_finalists`,
      `page_runtime_metrics`, `score_page_runtime`, `attach_page_scores`.
      Rename Copiale-specific identifiers to generic (shared-alphabet page
      groups, not "Copiale pages").
- [ ] The script currently imports private runner helpers
      (`_run_homophonic`, `_cipher_text_from_tokens`,
      `_automated_candidate_diagnostics`, `_plaintext_quality`,
      `_word_list`). Give these public, stable homes (either exported from
      `automated.runner` or moved into the new module) — promotion must not
      leave `src/` code importing underscore-private runner internals.
- [ ] Unit tests with a small synthetic two-page shared-alphabet fixture:
      combine, solve stub, project back, consensus.

### 2.2 Extract word-hypothesis repair core → `src/analysis/word_hypothesis_repair.py`

- [ ] Move from `probe_copiale_word_hypothesis_repair.py`,
      `report_copiale_repair_agenda.py`, and
      `probe_copiale_multipage_global_repair.py`: damaged-window detection
      (`damaged_windows_for_text`, `window_damage_score`), same-length
      dictionary hypothesis proposal (min/max word length, `max_edits`,
      per-window caps), hypothesis → global symbol-edit-set conversion
      (`current_assignment`, `parse_key`, `apply_assignment`), cross-page
      rescoring, collateral word-island adjudication
      (`annotate_acceptance`, `annotate_repair_evidence`), and ranking
      (`variant_rank_key`, `variant_summary`).
- [ ] Output type: `list[CandidatePacket]` (Phase 1.1), each carrying the
      proposed edit set, per-page score deltas, and collateral evidence in
      `provenance`.
- [ ] Language-agnostic API: dictionary path + language scoring profile are
      parameters (works for Borg Latin and synthetic English analogs too,
      not just German).
- [ ] Unit tests: synthetic damaged text where the true repair is known by
      construction (no benchmark ground truth needed).

### 2.3 Runner integration

- [ ] New `homophonic_refinement` value **`word_repair`**, plus composite
      `null_masks+word_repair` (null-mask bakeoff produces the finalist
      portfolio; word repair refines the selected/consensus basin). Existing
      values (`none`, `two_stage`, `targeted_repair`, `family_repair`,
      `null_masks`) unchanged.
- [ ] Adoption policy mirrors the existing post-solve chain: a repair is
      adopted only on strict improvement of the ground-truth-free objective
      (language-quality/validation score, never raw anneal score alone —
      p068 shows raw scores mis-rank readability).
- [ ] Config surface mirrors `DECIPHER_NULL_MASK_*`:
      `DECIPHER_WORD_REPAIR_{WINDOW_SIZE,WINDOW_STEP,MAX_EDITS,MAX_HYPOTHESES,MAX_HYPOTHESES_PER_WINDOW,MIN_WORD_LEN,MAX_WORD_LEN,CONSENSUS_TOP_N,...}`
      with defaults taken from the batch-runner sweet spots.
- [ ] Artifacts record: the candidate menu (packets), the adopted edits,
      per-page deltas, and rejected-hypothesis counts. `inspect_artifact.py`
      learns the new shapes (per the standing TODO).

### 2.4 Multipage shared-key benchmark route

- [ ] Add a first-class multipage mode: given a page group with a shared
      symbol alphabet, build the combined cipher, run the homophonic stack
      (+ refinements) once, project the shared key back per page, and emit
      one artifact per page plus a group artifact. CLI shape:
      `decipher benchmark ... --multipage-group copiale_evidence` (group
      definitions live next to the frontier/evidence JSONL files).
- [ ] This is where the strongest basins came from — cross-page evidence is
      the real unlock, and today it exists only inside the experiment
      script.
- [ ] Scoring: reuse `score_decryption` per page; group summary aggregates.

### 2.5 Agent exposure

- [ ] Through Phase 1.3's generic sessions: `search_word_repair_menu`
      (bounded generation on the current branch's basin, installs a finalist
      session) plus the shared review/rate/install tools. Keep it in the
      homophonic/nomenclator mode playbooks only (mode-filtered).
- [ ] Feed accepted/rejected word hypotheses into the existing
      `repair_agenda_*` bookkeeping (`tools_v2.py:1725,1742`) so agent-side
      durable repair state and the automated route share one ledger.
- [ ] Fake-provider test: agent requests a word-repair menu, rates, installs.

### 2.6 Calibration, acceptance, and ledger

- [ ] Rewrite the research scripts as thin wrappers importing the promoted
      modules (generation/ranking logic deleted from `scripts/`); the
      offline ground-truth calibration reports stay in
      `scripts/research/copiale/`.
- [ ] Acceptance gate on the five-page evidence packet (by-selection,
      GT-free): `null_masks+word_repair` ≥ `null_masks` baseline on at least
      4/5 pages, no page regresses by more than 1 point, and the multipage
      route reproduces (or beats) the ~79% best-basin result that today only
      the scripts reach.
- [ ] Also run the Borg packet and the English Copiale analog to check the
      machinery is not German-overfit.
- [ ] Update `docs/experimental_consolidation_plan.md` (rows move to
      "Promoted Core") and `docs/copiale_generalization_plan.md`.

---

## Phase 3: German Model + Solver Objective Upgrades

Runs in parallel with Phase 2 (different files). The current `de` binary
model is ~8× smaller than English (100 Gutenberg books, 23.2M chars,
475,932 distinct 5-grams vs 1,402,934 for `en`) and the solver objective is
structurally blind to word boundaries and umlauts.

- [ ] **3.1 Generalize the binary model format.** The loader hardcodes
      order-5 over exactly 26 lowercase letters
      (`src/analysis/zenith_solver.py:129–130`, index math at `:65–82`).
      Add a `zenith_binary_v2` header carrying the alphabet string and
      order; keep v1 reading unchanged. Mirror in the Rust engine. Raise
      `lru_cache(maxsize=2)` (`:118`) to cover the model inventory (12+
      files) and load `unknown_log_prob` from sidecar metadata instead of a
      single hardcoded floor.
- [ ] **3.2 Period-appropriate German model.** Train from the Deutsches
      Textarchiv (1600–1900 German, openly licensed) with an explicit,
      documented normalization policy — first model folds umlauts for
      compatibility; a second 30-symbol variant (ä/ö/ü/ß) rides on 3.1.
      Sidecar metadata records corpus, checksum, normalization,
      redistribution status (existing `model_metadata` pattern). Add the
      A/B packet from the Copiale plan's Milestone 2 and run
      `scripts/audit_german_scoring.py` against both models.
- [ ] **3.3 Space-aware scoring variant.** Train a 27-symbol (space
      included) model from boundary-preserved text and score 5-grams across
      boundaries, so the anneal objective itself rewards boundary-consistent
      words instead of leaving all word structure to post-hoc repair
      (`_make_window_starts` is boundary-blind, `zenith_solver.py:251–256`).
      Gate behind a profile value (e.g.
      `DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native_space`); English
      first as proof (word-delimited synthetics), then German/Latin.
      Note: anchor-refine currently no-ops without boundaries
      (`runner.py:6680–6682`) — the space-aware objective is the
      complementary fix for delimited text.
- [ ] **3.4 SA proposal mix.** Add a configurable fraction of 2-symbol swap
      moves to the zenith inner loop (`zenith_solver.py:454–521` is
      single-symbol only; the hill-climber already mixes swaps at
      `solver.py:171`). Validate on Zodiac 408 (must stay ≥99%) and the
      homophonic synthetic ladder before enabling by default.

Acceptance: Zodiac 408 regression unchanged; documented A/B showing the DTA
model's effect on the Copiale evidence packet; space-aware profile beats
flat-stream on word-delimited homophonic synthetics.

---

## Phase V3: Agent Loop v3 Redesign

Full design: `docs/specs/agent_v3_design.md`. Summary of the architecture:

- **InvestigationState** — one serializable state object (workspace,
  hypothesis board, evidence log, episode ledger, experiment queue, budget
  ledger); every turn's context is *rebuilt* from it; resume-from-artifact
  becomes the normal code path.
- **Episodes** — a lead context makes strategic decisions only; bounded
  fresh-context worker episodes (survey/search/reading/repair/verify/
  compare) do the work with task-scoped toolsets and structured results.
  The v2 executor's tool handlers are retained as the worker tool library.
- **Hypothesis-level actions** — composite tools
  (`hypothesis_test_word`, `hypothesis_apply_reading`,
  `branch_adjudicate`) returning evidence packets; `Reading` becomes a
  first-class artifact, replacing the seven boundary actuators.
- **Experiment queue** — long searches run async; the lead adjudicates
  results instead of babysitting synchronous calls.
- **Verification-gated declaration** — a fresh-context attestation
  (candidate text only) replaces self-attestation and the declare-gate
  bounce machinery.
- **Provider-native model sessions** (2026-07-13 user decision) — v3
  drops the lowest-common-denominator provider layer: each live context
  owns a native `ModelSession` (OpenAI Responses with reasoning passback
  and server-side chaining inside episodes; Anthropic with cache
  breakpoints and extended thinking; generic chat for Ollama/OpenRouter),
  with neutrality only at the event/usage/transcript seam the loop needs.
  Design section C7.

Milestones (each gets its own implementation spec + review cycle + commit):

- [ ] **M1** State + lead loop (no episodes); `run_v3` entry; token parity
      measurement vs v2 on synthetics.
- [ ] **M2** Episode runtime + fake-provider multi-context test harness.
- [ ] **M3** Composite actions + Reading artifact (absorbs Phase 6's test
      matrix; boundary misuse impossible by construction).
- [ ] **M4** Experiment queue (async + mandatory sync mode).
- [ ] **M5** Verify episodes + attestation-carrying declaration; firewall
      extension; wrong-basin fixture must be caught.
- [ ] **M6** v2-vs-v3 benchmark bake-off (accuracy/tokens/cost/wall-clock
      on the baseline matrix); default switch iff accuracy ≥ v2 at
      materially lower cost; delete the v2 coercion machinery.

---

## Phase 4: LLM Reader Scout (runtime selection)

The measured wall is selection: post-hoc-best beats GT-free selection on
4/5 Copiale pages, and the linear ranker's clustered holdout collapses
(mean best-label rank 16.0 → 6.14 only with trap features). An LLM reading
candidate texts is the strongest available selector; today it exists only
as the offline `rank_candidate_texts_with_llm.py` harness.

- [ ] **4.1 Promote the ranking core** → `src/analysis/llm_reader.py`,
      using `agent/model_provider.py` (provider-neutral), with a strict
      input firewall: candidate id + text excerpt only — no ground truth, no
      solver scores, no filenames that leak labels. Budget caps (max
      candidates, max chars each, max calls) are constructor parameters.
- [ ] **4.2 Runner opt-in**: `--finalist-reader llm[:model]` reranks the
      top-N packets of any finalist menu (null-mask, word-repair, transform)
      before selection. Votes, rationale snippets, model id, and cost land
      in the artifact. Ground-truth firewall tests from 0.7 extend to this
      path. Default off; cheap-model default when on.
- [ ] **4.3 Agent-side scout — SUBSUMED by V3-M5.** The candidate-reader
      role is the `verify`/`compare` episode kinds in the v3 architecture;
      do not build a separate v2-side scout. 4.1's `llm_reader` library is
      still the shared engine those episodes call.
- [ ] **4.4 Calibration bake-off.** Re-run the clustered-holdout ranker
      evaluation with the LLM reader vs `LinearLanguageQualityModel` vs the
      v2/ensemble validators on the same held-out families; publish the
      comparison in `docs/non_llm_candidate_ranker_plan.md` and set the
      adoption rule (e.g. reader wins ≥6/7 held-out families before default-on
      for Copiale-class runs).

---

## Phase 5: Agent Loop Cost + Robustness

**Shrunk by the V3 redesign.** Items 5.1 (telemetry — lands in the v3
budget ledger) and 5.2 (central tool output caps — applies to the shared
tool library either way) proceed. Items 5.3 (in-place pruning), 5.4
(mode-gated prompt), 5.5 (panel dedup), 5.6 (proactive declaration
guidance), 5.7 (retry budget), and 5.8 (panel rotation) are
deleted-by-design in v3 — implement them only if a v2 stopgap is needed
before V3-M6 lands.

Original rationale below, retained for reference.
Precedent: `docs/prompt_reduction_strategy.md` (27k tokens/turn baseline
measurement).

- [ ] **5.1 Cost telemetry per category** (from TODO): opening context, tool
      schemas, tool results by tool name, panels, retries. Emit into the
      artifact and `inspect_artifact.py` summaries.
- [ ] **5.2 Central tool-output cap.** `execute()` serializes results with
      no size bound (`tools_v2.py:2183`); only a few handlers self-truncate.
      Add a per-tool byte budget with overrides, truncation markers, and an
      artifact counter for truncations.
- [ ] **5.3 In-place history pruning.** `_compress_history` copies and
      rescans the unbounded `messages` list every turn
      (`loop_v2.py:359–433`, applied at `:1249`) — O(n²) over a 50-iteration
      run. Stub old tool results destructively once they pass
      `TOOL_RESULT_HISTORY_DEPTH`; the artifact call log already preserves
      full fidelity.
- [ ] **5.4 Mode-gated system prompt.** The static prompt ships every
      cipher-mode playbook on every run. Assemble playbook sections from the
      fingerprint (same mechanism that already filters tools, kept stable
      across the run for cache friendliness). Measure with 5.1.
- [ ] **5.5 Panel deduplication.** `build_workspace_panel`
      (`loop_v2.py:976–1045`) restates system-prompt discipline nearly
      verbatim each turn. Single source of truth; the panel references
      rather than restates.
- [ ] **5.6 Proactive pre-declaration guidance** (from TODO): when reading
      is strong (attested comprehensibility ≥8 or dict_rate >0.85), inject
      the declaration checklist before the agent hits the gate, collapsing
      the solve-at-N/declare-at-N+3 bounce.
- [ ] **5.7 Global inner-retry budget.** The inner `while True`
      (`loop_v2.py:1312`) can multiply one iteration into ~4 model calls via
      stacked gated/boundary/final retries exactly at end-of-run. Add a
      per-run cap on total extra calls, recorded in the artifact.
- [ ] **5.8 Rotate the panel window.** The panel shows only the first 30 +
      last 10 words of >90-word ciphers (`_select_word_indices`,
      `loop_v2.py:77`), leaving the model blind to manuscript middles.
      Rotate the visible window across iterations, or add an explicit
      "middle unseen — use decode_show" marker.

Acceptance: tokens/turn and cost/run measurably down on the baseline
snapshot runs with no accuracy regression on the smoke matrix.

---

## Phase 6: Boundary Actuator Consolidation — SUBSUMED BY V3-M3

Do not execute this phase separately. The Reading artifact and
`hypothesis_apply_reading` in `docs/specs/agent_v3_design.md` (milestone
M3) absorb this phase's goal and its test matrix (char-preserving,
char-changing, window-scoped, and miscounted readings). The v2 actuators
are deleted at V3-M6. Original analysis retained below for reference.

Seven–eight overlapping boundary/reading actuators (`act_split_cipher_word`,
`act_merge_cipher_words`, `act_merge_decoded_words`,
`act_apply_boundary_candidate`, `act_resegment_by_reading`,
`act_resegment_from_reading_repair`, `act_resegment_window_by_reading`) plus
the `boundary_projection` retry machinery exist to manage model confusion —
this is the single most failure-prone region for weaker models (see the
Llama/DeepSeek artifacts in CLAUDE.md).

- [ ] **6.1 Design one `act_project_reading`.** Input: branch, span (whole
      text or window), proposed reading. The tool itself detects
      char-preserving vs char-changing edits, aligns internally (tolerant of
      small count mismatches — this deletes the count-retry loop), returns a
      diff preview, and applies. Absorb `decode_validate_reading_repair` as
      a `dry_run=true` flag.
- [ ] **6.2 Measured migration.** Use Phase 5.1 telemetry to record
      boundary-tool failure/retry rates before and after. Ship the new tool
      alongside the old ones for one benchmark cycle, then remove the old
      ones from `TOOL_DEFINITIONS`, update prompts/playbooks, and delete the
      `boundary_projection` retry subsystem
      (`loop_v2.py:302–329, 1408–1416`).
- [ ] **6.3 Fake-provider tests** for the new actuator: char-preserving,
      char-changing, window-scoped, and deliberately miscounted readings.

Acceptance: boundary-related inner retries near zero across a benchmark
sweep; weaker-model runs (the OpenRouter matrix) show fewer actuator
misuse failures; net tool count drops by ~6.

---

## Investigator-mode structured model experiments (user-requested 2026-07-14)

When INV mode reaches its structured/ablation experiments (the
playbook-vs-no-playbook and model-tier study — see the investigator
design's calibration layer and the "scaffold as structure" analysis),
compare three models: **`claude-fable-5`, `gpt-5.6-sol`, `gpt-5.5`**.
This tests the model-tier-dependence hypothesis (playbook value likely
scales inversely with capability) on fresh testgen analogs, not
memorization-poisoned famous ciphers.

**Fable prerequisites — a hard gate before any Fable numbers are
trustworthy:**

1. **Verify Anthropic support end-to-end in the decipher harness.** The
   whole program has run on OpenAI; the Anthropic path (v2
   `ClaudeModelProvider`, v3 `AnthropicSession` — the latter was
   fake-test-only through M2 for lack of credits) must be exercised
   live. Blocker: the Anthropic account had no credits this session —
   confirm credits + that `claude-fable-5` routes to the Anthropic
   provider and completes a real agentic run.
2. **Served-model safety-gate detection (evaluation-integrity
   requirement).** The Anthropic safety gate can silently downgrade
   Fable→Opus; a contaminated "Fable" run must never enter the bake-off
   as Fable. Mechanism: capture the *served* model from each Anthropic
   response (`response.model` reflects what actually served, not what we
   requested) through `ModelResponse` into the artifact — per-call and as
   a run-level `safety_gate_fired`/`served_model_mismatch` flag set when
   served ≠ requested (e.g. requested `claude-fable-5`, served
   `claude-opus-*`). This is the harness-internal automation of the
   manual sub-agent transcript grep (`grep '"model"' … | uniq -c`) the
   orchestration strategy already uses. Then: benchmark/experiment
   summaries surface the flag prominently, and the model bake-off
   EXCLUDES gate-fired runs (or reports them as a separate
   contaminated-Fable bucket) so the comparison measures Fable, not a
   mix. Add a test: a fake Anthropic response whose `model` differs from
   the request sets the flag; a matching one does not. Reusable for any
   future Fable-in-harness run, not just INV.

Scope: this is a prerequisite slice for the INV model experiments, not
part of INV-0 itself; sequence it when the INV experiment harness is
built.

## Deferred / explicitly out of scope for this program

- Logogram/reading-holes/phrase-hypothesis promotion (revisit after Phase 4).
- Legacy homophonic path retirement and legacy parallelism env removal
  (already tracked in TODO Engineering Cleanup; unaffected here).
- The full investigator-mode/live-presentation work
  (`docs/unknown_cipher_investigator_mode.md`) — this program feeds it
  (packets, reader scout, cost work) but does not start it.
- Splitting `tools_v2.py` into modules for its own sake. Do it
  opportunistically as Phases 1.3/2.5/6 touch regions (finalist sessions,
  word-repair tools, boundary actuators move into their own files), not as
  a big-bang refactor.

## Suggested Landing Order

1. ✅ Phase 0 (landed `ef4ac9a`).
2. Baseline snapshot + Phase 1 packet + session store.
3. Phase 2.1–2.3 extraction and runner integration, with Phase 3.1–3.2
   (format generalization + DTA corpus build) as the parallel track.
4. V3-M1/M2 (state + lead loop, episode runtime) — may overlap late
   Phase 2, different files.
5. Phase 2.5 agent exposure BEFORE 2.4 (reprioritized 2026-07-13: the
   2b acceptance runs proved no GT-free scalar safely auto-adopts
   single-page repairs — word_repair landed menu-only, and the menu's
   intended selector is the agent-as-reader; trace evidence shows ~half
   of a successful Borg run's tool calls go to manual repair mechanics,
   the program's costliest remaining agent-side friction). Then 2.4
   multipage route; V3-M3 through M5.
6. Phase 4.1/4.2 runner-side reader (feeds V3-M5's verify episodes).
7. V3-M6 bake-off, default switch, coercion-machinery deletion.
   **GATED (2026-07-14): M6's live bake-off runs only on an explicit
   "go" from Matthew** — network stability + plan review first. Spec
   drafting/review for M6 may proceed; no live matrix runs.
7b. **DTA default switch — APPROVED by Matthew (2026-07-14).** Lands as
   a small slice immediately after the variant-registry slice: de's
   default resolution becomes the `historical_1600_1899` variant
   (ngram5_de_dta.bin), expressed through the registry; pin tests
   updated (de default = DTA, all other languages unchanged); local
   re-baseline of the Copiale packet recorded as the new reference.
8. **CLI observability revamp (user-requested 2026-07-14):** the agentic
   live display is hard to parse; target the Claude Code display
   pattern: turn-by-turn narration surfacing the agent's stated
   reasoning/decisions (not just state panels), tool calls rendered
   one-per-line with compact args + status + elapsed, EPISODES rendered
   as clearly nested sub-agent blocks (kind, goal, model tier, live
   status, one-line result summary on completion), experiment-queue
   status lines, a running cost/token ticker, and clear
   declaration/fallback rendering. Applies to both loops but designed
   around v3's turn/episode/experiment structure. Display-only — the
   artifact stays the source of truth; the existing pretty/raw/verbose
   modes gain a narrated default. ALWAYS print the artifact path at run
   end in every mode (agentic v2/v3 pretty display currently omits it —
   user-hit paper cut), plus a ready-to-paste
   `scripts/inspect_artifact.py <path>` hint line and a compact end-of-
   run summary (status, accuracy when scorable, turns, tokens, cost) so
   casual debugging rarely needs the artifact at all. Sequenced after M5
   (so episode + experiment + verification surfaces are final), before
   the docs slice (which will screenshot it). Reviewed like any slice.
9. **Docs & onboarding refresh (final slice, user-requested 2026-07-14):**
   bring a new user from clone to a successful run with current reality.
   README: fresh-checkout setup (venv, `pip install -e .`,
   `scripts/setup_dev.sh` incl. the Rust build, API keys —
   `.decipher_keys/` + keychain conventions, OpenAI as the billed
   account), a worked quick-start (the benchmark `--split`/`--test-id`
   + `--agentic --agent-loop v3` command shape — note `--split` is
   required for synthetic ids, a discovered paper cut), model setup
   (gpt-5.5 default, variant registry / `--model-variant`, DTA model),
   the multipage group route, and `--homophonic-refinement` values incl.
   `word_repair`. CLAUDE.md full refresh (the Phase 0.8 deferral):
   current tool count, v3 architecture summary, retire stale
   "Remaining Challenges". TOOLS.md count audit. Improve the
   "No matching tests found" error to name the split searched (the UX
   paper cut from live testing). Reviewed like any slice.

---

## Program status (2026-07-15): PLANNED SCOPE COMPLETE — follow-ups queued

All planned slices landed and reviewed (ledger = git log 7c086b4..98be6c9;
suite 1386). M6 bake-off verdict: NO default switch yet — v3 ~60% cheaper
and char-tied with v2 (wins Copiale), but trails on Borg word accuracy
because every v3 Borg run ended in best-branch fallback instead of an
explicit declaration (data: artifacts/m6_bakeoff/summary.jsonl, local).

**Implemented locally; paid validation deferred: revised M5.1 recovery,
adjudication, and declaration** -
`docs/specs/agent_v3_m5_1_declaration_trigger_spec.md`. M6 artifact
forensics showed that declaration timing is not the only Borg deficit:
the Reading-to-repair handoff rejected realistic punctuation/hole markers,
negative verification did not trigger repair, one fallback selected a much
weaker available branch, and the bake-off summary omitted unattached negative
attestations. M5.1 now fixes those surfaces in ordered local slices before its
focused paid acceptance. Passing the focused gate reopens M6 but does not
switch the default; a complete post-fix paired matrix is still required. After
a switch: v2 retirement one release later (deletion list in
agent_v3_design.md). The local targeted gate is green; no focused Stage-1 or
full bake-off was run during implementation in order to control token cost.

---

## External comparison tools — noted for later (not scheduled)

- **Ciphey / ciphey** (https://github.com/bee-san/Ciphey) — noted 2026-07-15
  for a LATER benchmark comparison; no work requested yet. `ciphey` is the
  Rust rewrite of the original Python `Ciphey`. A* search over a decoder/cracker
  tree + plaintext detection (n-gram/dict/regex + optional BERT gibberish
  model). Handles simple classical ciphers (Caesar/Vigenere/Beaufort/Atbash/
  Braille) + ~16 encodings (Base64 …) + hash lookup. **No transposition, no
  homophonic** — so it's a comparator for the ENCODING / easy-classical end,
  NOT a rival on the homophonic historical-manuscript frontier (Borg/Copiale/
  Zodiac) this project centers on. Belongs with the existing external-comparison
  lineage: `src/external/{azdecrypt,cryptocrack}.py` stubs +
  `scripts/run_automated_parity_matrix.py`. Install: `cargo install ciphey`.
