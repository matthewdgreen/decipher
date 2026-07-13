# Agent Loop v3 — Design

Status: design document, opened 2026-07-13 after the Phase 0 landing.
This is the architecture reference for the v3 redesign phase of
`docs/improvement_program_plan.md`. Per-milestone implementation specs will
be derived from it under `docs/specs/agent_v3_m*.md` as each milestone
starts. It replaces no shipped behavior by itself.

## Why a redesign, not more patches

The v2 loop is a single 50-iteration conversation steered by coercion
machinery. Inventory of that machinery, all of it added rationally in
response to observed failures:

- six hardcoded tool allowlists (`loop_v2.py:158–245`);
- five inner-retry subsystems (gated-tool, boundary-projection,
  final-declare, inspection-sandbox, repair-sandbox; `loop_v2.py:1312+`);
- ~200 lines of preflight/gate prompt constants (`loop_v2.py:247–356`);
- per-turn panel nags restating system-prompt discipline
  (`loop_v2.py:976–1045`);
- declaration prerequisite gates (hypothesis next-steps, branch-cards,
  coverage checks) that agents discover by bouncing off them;
- executor-level mode guardrails and context-prior blocks;
- history stubbing (`_compress_history`) fighting unbounded conversation
  growth.

Failure evidence this machinery has not fixed: wrong-basin confident
declarations (borg_0077v: dict_rate 0.35, wrong text), multi-iteration
gate bounces on every solved run, 20-iteration single-branch fixation
(llama artifact `a7cba7261bac`), boundary-actuator misuse severe enough to
need its own retry subsystem, ~27k tokens of prompt per turn before any
output, and structural blindness to the middle of long ciphers.

Diagnosis: the failures are properties of the *architecture* — one
long-lived context asked to be strategist, operator, reader, and
bookkeeper at once — not of the models. v3 restructures so the desired
behavior is the path of least resistance, then deletes the coercion.

## Design principles

1. **State is the source of truth; context is a view.** Everything durable
   lives in a serializable investigation state. Prompts are rebuilt from it
   every turn. The conversation transcript is never load-bearing.
2. **No context lives long enough to drift.** Strategy runs in a lead
   context whose transcript stays small because work happens in bounded
   worker episodes with fresh contexts.
3. **Actions at the hypothesis level.** The model proposes and adjudicates;
   compiled machinery does bookkeeping. Ritual tool sequences become single
   composite actions returning evidence packets.
4. **Verification is independent.** No context attests to its own output.
   Declaration requires a fresh-context reading attestation.
5. **Expensive contexts think only on new information.** Long searches run
   async in an experiment queue; the lead adjudicates results instead of
   babysitting calls.
6. **Ground-truth firewall unchanged and extended** to every new surface
   (episode inputs, experiment specs, attestations).

## Architecture

### C1. InvestigationState (`src/investigation/state.py`)

One serializable object owning everything durable about a run:

- `cipher`: immutable CipherText reference (as today, via Workspace).
- `workspace`: the existing `Workspace` (branches are unchanged — they
  remain the checkpoint/fork mechanism).
- `hypothesis_board`: evolution of today's hypothesis cards
  (`workspace_*_hypothesis` tools) into first-class state: mode, status,
  evidence for/against, tried/pending playbook items, linked branches.
- `evidence_log`: append-only typed observations (diagnostic results,
  episode findings, attestations) with provenance.
- `repair_agenda`: as today (`executor.repair_agenda`), relocated.
- `episode_ledger`: per-episode record — kind, goal, budget spent,
  structured result packet, ≤200-token summary.
- `experiment_queue`: submitted/running/completed experiment records.
- `budget_ledger`: tokens/cost/wall-clock by category (lead, per episode
  kind, per experiment) — Phase 5 telemetry lands here natively.

Serialization: `to_artifact_dict()` / `from_artifact_dict()`. Resume is
the identity operation: every turn already reconstructs from state, so
loading an artifact IS the normal code path (replaces `agent/resume.py`
special-casing).

### C2. Lead context builder (`src/investigation/context.py`)

`build_lead_context(state, token_budget) -> list[Message]` renders, in
priority order with per-section budgets:

1. Task framing + language notes (static, small).
2. Cipher summary + diagnostic fingerprint (as today's opening context).
3. Branch cards for the top-K branches (existing card renderer).
4. Hypothesis board (open/rejected, evidence counts).
5. Pending/completed-unadjudicated experiment results.
6. Last N episode summaries (structured, not transcripts).
7. Open questions / next-step suggestions (from playbooks, advisory only).
8. The last 1–2 lead tool exchanges verbatim (for local coherence).

Properties: deterministic given state; middle-blindness fixed by rotating
decode windows across turns within the budget; no stubbing, no O(n²)
rescans — `_compress_history` has no v3 equivalent. Stable prefix ordering
(1–2 rarely change) preserves provider prompt caching.

### C3. Episode runtime (`src/investigation/episodes.py`)

`EpisodeSpec`: `kind`, `goal` (short text), `inputs` (branch names,
hypothesis ids, candidate packets, curated context slices), `toolset`
(explicit tool-name list), `budget` (max tool calls, max output tokens,
wall clock), `result_schema`.

`run_episode(spec, state, model) -> EpisodeResult`: a *small* fresh-context
tool loop — system prompt is the episode contract (goal, inputs, toolset,
result schema), no panels, no gates, no mode filters. The toolset IS the
allowlist. Result is validated against `result_schema` (retry once on
mismatch), plus a bounded free-text summary. The worker writes nothing to
shared state; the lead integrates results explicitly.

Episode kinds (v1 set):

| kind | purpose | typical toolset | default model tier |
|---|---|---|---|
| `survey` | diagnostics on a branch/cipher | observe_*, score_*, decode_show | cheap |
| `search` | run one search tool, review finalists | one search_* + finalist review | mid |
| `reading` | produce a Reading for a branch | decode_show, corpus_*, score_* | strong |
| `repair` | compile a Reading/hypothesis into edits + collateral evidence | composite hypothesis tools | mid |
| `verify` | independent attestation of candidate plaintext | none (text-only) | cheap/mid |
| `compare` | adjudicate N branches/candidates | decode_show, score_panel | mid |

Model assignment per kind is config via the session-factory registry
(see C7); single-model operation (all kinds = one model) must work for
Ollama parity.

Lead tool surface (replaces most of the 88 for the lead): episode
launch/collect, experiment submit/collect, hypothesis board CRUD, branch
adjudication/promotion, a small set of direct cheap reads
(decode_show, score_panel, branch cards), and declare/unsolved. Workers
get task-scoped slices of the existing 89-tool library — **the v2 executor
handlers are retained as the tool library**; v3 replaces the loop and
prompts, not the analysis code. (`meta_attest_reading_comprehensibility`,
the v2 self-attestation surface, maps to the `verify` episode kind and is
retired with v2.)

### C4. Hypothesis-level composite actions (`src/investigation/actions.py`)

New tools built on existing primitives, returning evidence packets
(CandidatePacket-compatible, Phase 1):

- `hypothesis_test_word(branch, window, word)` — fork, derive symbol
  edits, global rescore, collateral word-island check, verdict. Reference
  implementation: Phase 2's `analysis/word_hypothesis_repair.py`; this
  tool is its agent-facing face.
- `hypothesis_apply_reading(branch, reading)` — compile an accepted
  Reading into key edits/boundary changes in one step.
- `branch_adjudicate(branches)` — packet-based comparison table.

**Reading as a first-class artifact** (`src/investigation/reading.py`):
`Reading` = proposed plaintext (or per-window fragments), span
confidences, unresolved holes, produced by `reading` episodes and stored
in state. `hypothesis_apply_reading` does its own alignment, tolerant of
small count mismatches, auto-detecting char-preserving vs char-changing
edits. This subsumes plan Phase 6: the seven v2 boundary actuators and
the `boundary_projection` retry subsystem have no v3 equivalents (they
remain in the v2 path until deletion at M6).

### C5. Experiment queue (`src/investigation/experiments.py`)

`Experiment` = a solver/search invocation spec (tool name + args + budget
profile) executed asynchronously (thread/process pool, honoring
`DECIPHER_PARALLEL_WORKERS`). Results land in `state.experiment_queue`
with the same finalist/packet shapes the synchronous tools produce.
Lead flow: submit → continue other work → adjudicate on completion.
A synchronous execution mode is mandatory for tests and simple runs.
Long-running v2 tools (`search_automated_solver`, transform screens,
null-mask bakeoffs, Quagmire shotgun) become the first queue-able
experiments; cheap tools stay synchronous.

### C6. Verification-gated declaration

`meta_declare_solution` (lead) requires an `attestation` argument that
must reference a `verify` episode result recorded in state for the same
branch content hash. The verify episode receives ONLY the candidate
plaintext and language — no scores, no history, no cipher — and returns:
coherence score (0–10), a clause-level gloss/translation, anomalies
(non-words, broken syntax spans), and a "would a fluent reader accept
this?" verdict. Weak attestation doesn't block declaration (the lead may
declare a partial), but the attestation is recorded and the declaration
carries it. This replaces the v2 declare-gate bounce machinery and
implements the proactive-guidance TODO as structure instead of prompt
injection. Phase 0's `fallback_declared` semantics are preserved
unchanged for exhaustion/error.

Firewall: verify episodes are the most leak-sensitive surface (text-only
input); `assert_no_ground_truth_leak` extends to episode inputs,
experiment specs, and attestations.

### C7. Provider-native model sessions (amendment, 2026-07-13 — user decision)

v3 abandons the lowest-common-denominator provider abstraction. Matthew's
directive: agent loops should be as native and powerful as each API
allows; a limiting neutral layer is not worth maintaining. Motivating
failure: the neutral message schema silently dropped OpenAI reasoning
items between tool calls, measurably handicapping gpt-5.6 tiers.

**Design: neutrality moves up to the seam the loop actually needs.**
A `ModelSession` (one per live context: the lead, each episode) OWNS its
conversation state in the provider's native format and exposes only:

```python
class ModelSession(Protocol):
    def send(self, blocks: list[ContextBlock], tools: list[dict],
             budget: TokenBudget) -> AgentEvents   # text / tool_calls / usage
    def add_tool_results(self, results: list[ToolResult]) -> None
    def export_transcript(self) -> dict  # provider-tagged, for artifacts
    capabilities: SessionCapabilities    # declared, see below
```

The loop never sees or stores provider message formats; it supplies
semantic context blocks (from the C2 builder) and consumes events.
Per-provider implementations exploit everything native:

- **OpenAISession** (primary — current preferred models are gpt-5.5/5.6):
  Responses API native. Within an episode, `previous_response_id`
  server-side chaining (safe because episodes are bounded and never
  history-stubbed); for the lead and anything resumable, stateless mode
  with `reasoning.encrypted_content` round-tripped so chain-of-thought
  survives tool calls. Strict tool schemas. Native usage/cache fields.
- **AnthropicSession**: native messages with `cache_control` breakpoints
  placed at the C2 builder's stable-prefix boundaries; extended thinking
  where configured; fine-grained tool-result blocks.
- **GenericChatSession**: OpenAI-compatible chat completions — covers
  Ollama, OpenRouter, and legacy models. This is the old neutral behavior
  demoted to one implementation among several.

`SessionCapabilities` declares what each implementation supports
(server_state, reasoning_passback, cache_breakpoints, strict_tools) so
episode scheduling can select worker tiers by capability + price, and
tests can assert capability-dependent behavior per session type.

**What stays common:** the event surface, `TokenBudget`, per-entry cost
accounting (A7), and `export_transcript()` — artifacts store the
provider-tagged native transcript; `inspect_artifact.py` renders per
provider. Fake sessions implement the same protocol per provider shape,
replacing the single fake-provider script style.

**Migration:** `agent/model_provider.py` remains untouched for the v2
loop until retirement. The session layer is v3-only, introduced at M1
with OpenAISession + GenericChatSession (AnthropicSession lands with M2).
The interim v2 reasoning-passback hotfix
(`docs/specs/hotfix_openai_reasoning_passback.md`) builds the
encrypted-content mechanics OpenAISession will reuse.

### C8. Artifacts, observability, benchmark integration

- `RunArtifact` gains `episodes`, `experiments`, `attestations`,
  `budget_by_category` (schema additive; v2 artifacts stay readable).
- `scripts/inspect_artifact.py` learns episode/experiment/attestation
  rendering (standing TODO requirement).
- Entry: `run_v3()` alongside `run_v2()`; CLI/benchmark flag
  `--agent-loop v3` (default remains v2 until M6).
- `tests/test_agent_reliability.py` fake-provider patterns extend to
  scripted lead + scripted workers (the fake provider needs per-context
  scripts keyed by episode kind).

## Milestones

**M1 — State + lead loop, no episodes.** `InvestigationState`, context
builder, minimal lead loop using direct tools only (existing handlers),
`run_v3` entry, artifact round-trip, resume-as-load. Introduces the
`ModelSession` seam (C7) with OpenAISession (Responses-native, stateless
encrypted-reasoning mode) and GenericChatSession; AnthropicSession lands
with M2. Acceptance:
fake-provider parity tests; solves `synth_en_250nb_s4`-class synthetics at
≥ v2 accuracy with materially fewer prompt tokens per turn (measure).

**M2 — Episode runtime.** Worker loop, `survey`/`search`/`reading`/
`compare` kinds, episode ledger, per-kind model config, episode isolation
model (A1 below), handler-output discipline sweep (A5). Acceptance:
fake-provider episode tests; a scripted survey → search → reading →
compare workflow passes (reading *application* is M3); one real-model
smoke on a synthetic.

**M3 — Hypothesis actions + Reading.** Composite tools, Reading artifact,
`hypothesis_apply_reading` alignment (absorbs Phase 6 test matrix:
char-preserving, char-changing, window-scoped, miscounted readings).
Acceptance: boundary-actuator misuse rate zero by construction in v3
fake-provider suite; word-hypothesis composite matches
`word_hypothesis_repair` library outputs exactly.

**M4 — Experiment queue.** Async execution, adjudication flow, sync mode.
Acceptance: a lead run that overlaps two experiments with other work in
fake-provider tests; wall-clock win demonstrated on one real transform
screen + anneal pair.

**M5 — Verification-gated declaration.** Verify episodes, attestation
records, declaration wiring, firewall extension. Acceptance: firewall
tests cover episode inputs; scripted wrong-basin fixture (borg_0077v-like
readable-but-wrong text) shows attestation flags anomalies where v2
self-attestation declared.

**M6 — Bake-off and default switch.** v2 vs v3 on the benchmark matrix
(synthetic ladder cases, Borg packet, Copiale evidence packet,
`synth_en_200honb_s6`) comparing accuracy, tokens, cost, iterations,
wall-clock. Switch default to v3 iff accuracy ≥ v2 on the matrix and cost
is materially lower. **Deletion is NOT part of M6** (review finding F3):
the coercion machinery — allowlist sets, inner-retry subsystems, preflight
constants, panel nags, `_compress_history`, boundary actuators +
`boundary_projection`, declare-gate wiring, `agent/resume.py` — is
module-level v2-loop code and cannot be removed while v2 remains runnable.
It is all deleted in one operation at v2 retirement (one release after the
default switch), together with the v2 loop itself. M6 does deliver the
v2-artifact → v3-state adapter so `resume-artifact` keeps working on old
artifacts after retirement.

## Risks and mitigations

- **Workers lack context the lead had.** Mitigate: episode inputs carry
  curated slices (branch card, relevant evidence); lead can re-launch with
  more context; `reading` episodes default to the strong model tier.
- **Lead thrash (launching episodes without integrating).** Mitigate:
  budget ledger surfaces spend per kind in the lead context; episode
  results require explicit adjudication notes on the hypothesis board.
- **Cost regression from episode overhead.** Each episode re-pays a small
  system prompt but avoids dragging 50-turn history; M1/M2 acceptance
  includes token measurements, and M6 gates the default switch on cost.
- **Fake-provider test complexity.** Per-kind scripted workers are more
  machinery than v2's single script; build the harness in M2 before any
  real-model dependence.
- **Two loops in parallel during migration.** Contained by: shared tool
  library (no logic forks), additive artifact schema, M6 deletion
  deadline. No new features land on v2 during the redesign except
  Phase 2 tool exposure, which is built episode-compatible from the start.

## Post-review amendments (2026-07-13)

Adopted from the Fable design review (verdict: SOUND WITH AMENDMENTS;
finding numbers F1–F15 refer to that review). These amendments are binding
on the milestone specs.

- **A1 (F1) Episode isolation model — blocks M2.** Workers never touch the
  lead's live Workspace or executor. Each episode gets a fresh
  `WorkspaceToolExecutor` over a workspace reconstructed from deep-copied
  branch snapshots (Phase 0.4's `copy_as` semantics). Episode results are
  snapshot packets; the lead integrates explicitly via `restore_branch`.
  Search-tool sessions live in the Phase 1 `FinalistSessionStore` owned by
  `InvestigationState`, not by any executor, so review/rate/install
  triplets work across episode boundaries.
- **A2 (F2) Declaration policy is injected — blocks M1.** The declare-gate
  cascade inside `_tool_meta_declare_solution` is extracted into a policy
  object supplied by the loop: v2 injects the full gate set (behavior
  unchanged); v3-M1 injects none; v3-M5 injects the attestation check.
  The M1 spec defines the v3 mapping for `executor.set_iteration`/
  `set_max_iterations` (lead turn number) since several shared guards key
  off them.
- **A3 (F4) State carries the cipher — blocks M1.**
  `InvestigationState.to_artifact_dict()` serializes the CipherText itself
  (symbols, tokens, separators, word structure, plaintext alphabet) and
  full branch state including `token_order`, `transform_pipeline`, and
  `metadata`. Resume never parses prompt text (v2's resume regex-scrapes
  `messages[0]` and silently drops those three branch fields — do not
  inherit either behavior).
- **A4 (F5) Handler-output discipline sweep — M2 work item.** Retained
  handlers embed v2 loop discipline in their outputs
  (`suggested_next_tools`/`recommended_next_tool` naming boundary
  actuators, finalize-phase guards saying "prefer meta_declare_solution",
  the gated-call rejection text naming v2 workflows). M2 sweeps these:
  next-tool hints computed against the active episode toolset; guard
  blocks become policy-injected; off-toolset rejection text becomes a
  neutral one-liner.
- **A5 (F6) Experiments are pure functions — blocks M4.** An experiment is
  (cipher, branch snapshot, config) → result packet, executed through the
  automated-runner layer; no executor state crosses the boundary; results
  integrate on the lead thread only. A single worker-budget arbiter splits
  `DECIPHER_PARALLEL_WORKERS` between concurrent experiments and inner
  solver parallelism (no N×N process blowup).
- **A6 (F7) Attestation binding.** Attestation hash = sha256 of the exact
  candidate string sent to the verify episode, produced by one named
  renderer; the record is {branch, hash, renderer_id, episode_id};
  declaration recomputes with the same renderer.
- **A7 (F8) Mixed-model accounting.** Budget-ledger entries carry
  (category, provider, model, input/output/cache tokens); cost is summed
  per entry, never recomputed from run totals. `run_episode` takes its
  session from a kind→session-factory registry (C7); the fake harness
  registers one scripted fake session per kind/provider shape.
- **A8 (F9) Ordering.** M2 acceptance ends at `compare` (application is
  M3). M3 has an explicit entry gate: Phase 2.2
  (`analysis/word_hypothesis_repair.py`) must have landed.
- **A9 (F10) Failure semantics.** Episode double schema-failure → the
  episode returns a structured `episode_failed` result (raw text attached)
  for the lead to adjudicate; handler exceptions inside episodes surface in
  the result packet, never crash the lead; wall-clock budgets are checked
  between tool calls only, so long-running tools are excluded from episode
  toolsets and routed to the experiment queue; on interrupt, in-flight
  episodes persist as `episode_failed(interrupted)` in the ledger and
  running experiments are marked orphaned (resumable by resubmission).
- **A10 (F11, F12) Single writers.** The hypothesis board is the sole
  hypothesis store; the v2 `workspace_*_hypothesis` handlers become
  adapters over it. `repair_agenda` moves to `InvestigationState` and is
  constructor-injected into executors; episode-local agenda additions
  arrive in the result packet and are merged by the lead.
- **A11 (F13) Artifact semantics.** `ToolCall.iteration` = lead turn
  number; episode tool calls additionally carry `episode_id`.
  `artifact.messages` stores the lead transcript only (episode transcripts
  summarized in the ledger; full text behind a debug flag) — rebuilt
  contexts are NOT persisted per turn.
- **A12 (F14) Honest cache accounting.** Rebuilt contexts cache only the
  static prefix; M1's token measurement reports cache-read vs uncached
  input explicitly, and the M1 acceptance criterion is total *cost*, not
  raw token counts.

## Interactions with the improvement program

- **Phase 1 (packets/sessions): prerequisite.** Packets are the episode
  and experiment interchange format; finalist sessions become episode
  results.
- **Phase 2 (Copiale promotion): unchanged**, with 2.5's agent exposure
  built as the M3 composite-action shape (hypothesis_test_word) rather
  than a bare v2 tool.
- **Phase 3 (German model/solver): independent, parallel.**
- **Phase 4 (LLM reader): 4.1/4.2 (library + runner-side reranker)
  unchanged; 4.3 (agent-side scout) becomes the `verify`/`compare`
  episode kinds (M5).**
- **Phase 5 (cost): telemetry (5.1) lands in the budget ledger; 5.2
  (output caps) still applies to worker tools; 5.3–5.5, 5.7, 5.8 are
  deleted-by-design in v3 and stay v2-only stopgaps if needed.**
- **Phase 6 (boundary consolidation): subsumed by M3.** Do not execute
  Phase 6 separately.
