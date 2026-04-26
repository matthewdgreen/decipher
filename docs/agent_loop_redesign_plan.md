# Agent Loop Redesign Plan

Status: Milestones 1-3 are implemented in the existing `run_v2` benchmark
path. Milestone 4 generalization is partly validated for Borg and remains open
for Copiale/German. The English Borg analog plus Borg `0109v`, `0045v`,
`0140v`, `0171v`, and `0077v` have been exercised; Copiale `p068` showed that
Copiale needs a separate capability track rather than only Borg-style repair.
The loop now runs through a provider-neutral response adapter, records loop
events, supports bounded same-iteration inspection/repair sandboxes, can retry
gated or wrong-length boundary-projection attempts inside the same outer
iteration, and writes final reading/process summaries.

## Why This Exists

The current v2 agent loop is good enough for benchmarkable tool use, but it is
too turn-based for reading-driven repair. Each "iteration" is a full model
call followed by tool execution and a large workspace panel. That makes small
inspect/apply/inspect cycles expensive, encourages broad `act_bulk_set` or
`search_anneal` moves, and makes it hard for the agent to recover from one bad
local repair.

The Borg `0109v` work made the limitation concrete:

- The English analog was solved at 100%/100% once the agent could state a
  whole reading and apply word boundaries safely.
- The Latin Borg case improved in one run to 13.9% char / 20.8% word, but the
  behavior was not reliable.
- Prompt nudges and late-turn tool gating made the agent attempt the
  full-reading workflow, but it still struggled to produce a character-count
  preserving proposal and sometimes spent turns on local churn.

The next step should be a more modern agentic harness, not more Borg-specific
prompt pressure.

## Design Goals

1. Keep provider neutrality.
   The loop should support Claude, OpenAI, and future local/hosted models
   behind a small provider adapter. Do not bind core state management to one
   provider SDK.

2. Separate outer reasoning from inner tool work.
   A benchmark "turn" should not be the same thing as a single small tool
   action. The agent needs a cheap inner loop for reversible repair steps.

3. Make workflows first-class.
   Known sequences such as "draft reading -> validate -> project boundaries ->
   inspect mismatches" should be structured workflows with state and recovery,
   not just prose instructions.

4. Preserve benchmark observability.
   Every model call, tool call, rejection, rollback, workflow state change,
   and declaration must still be recorded in artifacts.

5. Keep partial-solution safety.
   The final action must still declare the best available branch rather than
   risk a zero-score exhausted run.

## Proposed Architecture

### Provider Adapter

Define a narrow model-provider interface:

- `send(messages, tools, system, max_tokens) -> ModelResponse`
- normalized tool-use blocks
- normalized token/cost metadata when available
- provider name and model name

The rest of the harness should not know whether the backend is Claude,
OpenAI, or another API. Provider-specific Agent SDKs can be used behind this
adapter later, but should not own the benchmark loop.

### Outer Loop

The outer loop remains the benchmark-facing loop:

- initialize workspace and automated preflight
- send initial context
- allow high-level planning
- run one or more inner workflows
- request declaration
- write complete artifact

Outer loop iterations are for strategy changes and declaration, not for every
single letter repair.

### Inner Tool Loop

Add an inner loop that can execute several small tool steps before returning
to the outer loop. It should support:

- tool gating by mode
- immediate retry on disallowed tools without consuming an outer iteration
- compact feedback after each tool call
- automatic stop conditions
- max inner steps to prevent loops

Example modes:

- `explore`: read-only inspection, scoring, branch comparison
- `reading_repair`: mapping and reading-driven repair tools only
- `boundary_projection`: full-reading validation and resegmentation only
- `polish`: anchored search and quality checks only
- `declare`: declaration only

### Workflow Macros

Add structured workflows that combine multiple tools and enforce invariants.

#### Full Reading Repair Workflow

Input:

- branch
- proposed best reading
- optional uncertainty notes

Steps:

1. Normalize current branch and proposed reading.
2. Check character counts and character-preservation.
3. If character-preserving, apply `act_resegment_by_reading`.
4. If same length but letter-different, apply boundary projection and return
   mismatch spans.
5. Group mismatch spans by cipher symbol where possible.
6. Ask the model for explicit accept/reject decisions on proposed symbol
   repairs.
7. Apply accepted repairs one at a time with immediate changed-word feedback.
8. Stop with a branch summary and remaining uncertainty list.

Key invariant:

- Boundary projection must never change the key or decoded character stream.

#### Reading-Driven Mapping Workflow

Input:

- branch
- list of proposed repairs, such as `TREUITER -> BREUITER`

Steps:

1. Locate each proposed repair in the current decoded words.
2. Identify the cipher symbol(s) responsible for differing letters.
3. Apply one candidate mapping at a time.
4. Report changed words and score deltas.
5. Auto-hold changes that add multiple readable words unless clear damage is
   detected.
6. If a change is ambiguous, fork and compare rather than overwrite.

#### Declaration Workflow

Before declaration, build a compact "best branch card":

- decoded preview
- branch scores
- mappings changed from preflight
- custom boundary status
- unresolved reading hypotheses
- artifact analyzer warnings

If no branch has gone through a needed workflow, route back to that workflow
unless this is the final action.

## Tool Gating Policy

Tool gating should happen in two places:

1. Model-facing tool list: hide unavailable tools.
2. Executor enforcement: reject unavailable tools with a clear `tool_gated`
   result.

Rejected gated tools should trigger an immediate retry inside the same outer
iteration when possible. The retry message should include:

- attempted disallowed tool
- only allowed tools
- the mode goal
- a concise instruction for the next valid action

This prevents a single stale tool choice from consuming the penultimate turn.

## State Model

Add explicit run state beyond message history:

- active mode
- active branch
- repair agenda
- attempted repairs
- held repairs
- reverted repairs
- unresolved hypotheses
- whether required workflows have run per branch
- latest branch quality card

This state should be serialized into artifacts and shown compactly to the
model.

## Artifact Changes

Artifacts should record:

- outer iteration number
- inner step number
- mode
- available tools
- gated tool rejections
- workflow start/finish events
- workflow inputs and normalized outputs
- repair agenda snapshots

The analyzer should learn labels for:

- `workflow_skipped_before_declaration`
- `gated_tool_retry`
- `same_length_projection_failed`
- `repair_agenda_unresolved`
- `inner_loop_exhausted`

## SDK Strategy

Do not move the core loop into a provider-specific SDK yet.

Recommended path:

- Build a Decipher-native orchestrator with a provider adapter.
- Borrow concepts from modern agent frameworks: tracing, guardrails, durable
  state, handoffs/workflows, and tool gating.
- Keep tools as plain JSON-schema functions so Claude/OpenAI/local backends
  can all use them.
- Consider MCP later as a way to expose Decipher tools to external agents, not
  as the internal state machine.
- Consider LangGraph-style durable graph execution if the loop grows complex,
  but only after the workflow boundaries are clear.

Provider-specific Agent SDKs may still be useful for experiments, especially
for tracing or sandboxed execution, but they should live behind the adapter
instead of defining the Decipher architecture.

## Milestones

### Milestone 1: Design Freeze

- [x] Write provider adapter interface.
- [x] Define outer loop vs inner loop responsibilities.
- [x] Define mode/tool-gating table.
- [x] Define artifact schema additions.
- [x] Decide how the next loop entry point coexists with `run_v2`: defer a
  separate entry point until the workflow boundaries settle; keep the
  prototype in `run_v2` so existing CLI/benchmark paths exercise it.
- Behavior change is limited to the `run_v2` prototype path.

### Milestone 2: Inner Loop Prototype

- [x] Add a narrow inner-loop path for one mode: `boundary_projection`.
- [x] Support same-iteration retry for gated tools.
- [x] Support same-iteration retry when a full-reading proposal has the wrong
  normalized character count.
- [x] Defer a separate next-generation harness entry point; for now, the
  prototype is wired into `run_v2` so existing CLI tests exercise it
  immediately.
- [x] Run English analog.
- [x] Run fair post-tightening Borg `0109v` trials.
- [x] Compare tool counts, token cost, char/word accuracy, and analyzer
  labels in `docs/agent_loop_milestone2_comparison.md`.

Early validation notes:

- English analog: `artifacts/english_borg_analog_001/f8c8ead3e9b2.json`
  reached 100.0% character / 100.0% word accuracy in 6 iterations. The agent
  used whole-reading resegmentation and one mapping repair from a 99.2% /
  2.9% automated preflight.
- Borg `0109v`: `artifacts/parity_borg_latin_borg_0109v/048448b15ebb.json`
  completed before the actuator-only gate tightening and reached 13.9%
  character / 7.5% word. It showed why validation alone must not satisfy the
  late full-reading gate: after `decode_validate_reading_repair`, the next
  turn dropped back to general explore tools.
- Borg `0109v`: `artifacts/parity_borg_latin_borg_0109v/5b16b17ac4c1.json`
  is not a capability measurement; it stopped after an API credit error and
  auto-declared a weak partial branch.
- Borg `0109v`: `artifacts/parity_borg_latin_borg_0109v/bb419814f0c3.json`
  reached 13.9% character / 7.7% word after actuator-only gate tightening.
- Borg `0109v`: `artifacts/parity_borg_latin_borg_0109v/0827d5850034.json`
  reached 13.9% character / 6.4% word and exercised two
  `boundary_projection_count_retry` events.

Milestone 2 closeout: complete as an inner-loop prototype. The result is
mechanically useful and well-observed, but not sufficient for Borg by itself;
Milestone 3 owns durable reading-repair capability.

### Milestone 3: Reading Repair Workflow

- [x] Add a first low-friction word repair planner/action pair:
  `decode_plan_word_repair` and `act_apply_word_repair`.
- [x] Add structured repair agenda.
- [x] Add one-at-a-time repair application with immediate feedback.
- [x] Add branch cards and unresolved hypothesis summaries.
- [x] Add bounded low-cost inspection/repair sandboxes so read-only checks,
  repair menus, local resegmentation attempts, branch cards, and declaration
  can happen without burning additional outer benchmark iterations.
- [x] Add artifact resume/continuation support for follow-up iterations
  without replaying the whole run from scratch.
- [x] Add final reading/process summaries and render them in the pretty
  terminal final screen.

Early implementation note:

- The first slice gives the agent a direct tool path for a same-length
  reading hypothesis such as "this decoded word should be that word": plan
  the responsible cipher-symbol changes, preview collateral `changed_words`,
  then apply the repair if the preview reads better.
- Word-repair plans now populate a durable `repair_agenda` in the run
  artifact. `act_apply_word_repair` marks matching agenda items applied, and
  `repair_agenda_list` / `repair_agenda_update` let the agent resolve held,
  rejected, or blocked repairs before declaration.
- `workspace_branch_cards` now exposes compact branch cards with internal
  scores, mapped count, decoded excerpt, applied/held/open repairs, and
  orthography-risk warnings.
- Reading repair actions now flag broad Latin `U/V` or `I/J` shifts as
  orthography risks, so the agent can distinguish manuscript-faithful repair
  from modernized/classicized spelling.
- Declaration is now gated on two bits of explicit run discipline: open or
  blocked repair-agenda items must be applied/held/rejected before declaring,
  and runs with multiple branches must call `workspace_branch_cards` before
  declaring. The final action turn exposes only bookkeeping tools plus
  `meta_declare_solution`, so the agent can perform cards/update/declare in
  one response. If it performs final bookkeeping but forgets to declare, the
  loop now retries inside the same final iteration with an explicit
  declaration nudge.
- First Borg trial with these tools:
  `artifacts/parity_borg_latin_borg_0109v/495a27b339ba.json` reached 13.9%
  character / 9.1% word accuracy. The agent used `act_apply_word_repair` for
  `NREUITER -> BREUITER` and `SIMALITER -> SIMILITER`, tried
  `RLURES -> PLURES`, then reverted after seeing broad collateral damage.
- Follow-up Borg trial after agenda prompting:
  `artifacts/parity_borg_latin_borg_0109v/38e3d02d7c7a.json` confirmed the
  agenda behavior, but regressed to 11.6% character / 5.1% word because the
  agent classicized Latin `U` forms into `V` forms (`BREVITER`, `QVOD`, etc.).
  This motivated the orthography-risk guard.
- Latest Borg trial before declaration gating:
  `artifacts/parity_borg_latin_borg_0109v/260a15ce6778.json` recovered to
  13.9% character / 7.7% word and avoided broad `U/V` classicization, but
  left an open `RLURES -> PLURES` agenda item while explaining the rejection
  only in prose. This motivated the repair-agenda declaration gate.
- First Borg trial after declaration bookkeeping gates:
  `artifacts/parity_borg_latin_borg_0109v/fca17fd203a6.json` reached 13.9%
  character / 7.7% word. It used substantially richer repair tooling
  (`act_apply_word_repair` on four items, `repair_agenda_list`, and
  `workspace_branch_cards`) but spent the final turn on bookkeeping and failed
  to call `meta_declare_solution`, causing fallback declaration. This
  motivated the same-iteration final declaration retry.
- Later follow-up added edit-aware word and character scoring, same-iteration
  inspection/repair sandboxes, final-summary recovery from blocked
  declarations, artifact continuation, and the pretty live terminal display.
  Borg `0109v` now reliably produces readable partial Latin with a high
  insertion-friendly local word score, though it still has inserted words and
  unresolved spelling/boundary errors.
- Borg `0045v` was exercised as an out-of-family Latin page and reached a
  readable but still partial branch:
  `artifacts/parity_borg_latin_borg_0045v/4353961ccb34.json` improved over
  automated preflight but retained significant errors. This is useful evidence
  that the workflow generalizes beyond a single page, but not enough to close
  Milestone 4.

### Milestone 4: Generalization

- [x] Run Borg `0109v`.
- [x] Run Borg `0045v`.
- [x] Run the English Borg analog.
- [x] Run additional Borg pages outside the current parity trio, especially
  `borg_single_B_borg_0140v`, `borg_single_B_borg_0171v`, and/or
  `parity_borg_latin_borg_0077v`.
- [x] Confirm improvements are not just prompt overfitting by comparing
  behavior and artifact labels across those additional pages.
- [x] Decide whether to expand the immediate loop-work branch to
  Copiale/German, after at least one stretch run such as
  `copiale_single_B_copiale_p068`: do not treat Copiale as just another Borg
  page. Create a separate Copiale/German capability track.
- [ ] Add a full-agent parity smoke suite so core agent-loop behavior can be
  regression-tested without manually inspecting long artifacts every time.

Latest generalization checkpoint:

- Borg `0140v`, `artifacts/borg_single_B_borg_0140v/47df72a4da8b.json`:
  automated preflight was weak (`36.9%` char / `0.0%` word), while the agent
  found a fresh readable branch at `85.5%` char / `54.8%` word.
- Borg `0171v`, `artifacts/borg_single_B_borg_0171v/a43a53111e26.json`:
  automated preflight was already strong (`90.9%` char / `72.7%` word), but
  the agent over-repaired it to `85.8%` char / `50.8%` word by favoring more
  classicized Latin-looking forms. This motivates the protected-preflight
  rule: a readable `automated_preflight` branch should be preserved as the
  baseline, and broad Latin `U/V` or `I/J` edits must be treated skeptically.
- Borg `0077v`, `artifacts/parity_borg_latin_borg_0077v/c9d17916d17f.json`:
  automated preflight was weak (`37.2%` char / `2.8%` word), while the agent
  reached a readable partial branch at `84.1%` char / `53.5%` word.
- Copiale `p068`, `artifacts/copiale_single_B_copiale_p068/7d795a0ae0a9.json`:
  the agent did not improve over preflight (`45.3%` char / `0.0%` word). It
  identified German-looking islands but not coherent sentence-level German.
  Copiale needs separate work on German homophonic/nomenclator modeling,
  context use, and declaration discipline.

## Open Questions

- Should workflow macros be tools visible to the model, or internal modes
  selected by the harness?
- How many inner steps should be allowed before another model call is required?
- Should branch comparison be automatic after every held repair?
- How should the loop detect "readable enough" without ground truth?
- What is the minimum provider abstraction needed for OpenAI, Claude, and a
  local model without losing useful model-specific capabilities?
