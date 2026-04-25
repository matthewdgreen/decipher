# Agent Loop Redesign Plan

Status: design note, no implementation yet.

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

- Write provider adapter interface.
- Define outer loop vs inner loop responsibilities.
- Define mode/tool-gating table.
- Define artifact schema additions.
- No behavior change yet.

### Milestone 2: Inner Loop Prototype

- Add inner loop for one mode: `boundary_projection`.
- Support same-iteration retry for gated tools.
- Run English analog and Borg `0109v`.
- Compare tool counts, token cost, char/word accuracy, and analyzer labels.

### Milestone 3: Reading Repair Workflow

- Add structured repair agenda.
- Add one-at-a-time repair application with immediate feedback.
- Add branch cards and unresolved hypothesis summaries.

### Milestone 4: Generalization

- Run Borg `0109v`, Borg `0045v`, and the English analog.
- Confirm improvements are not just prompt overfitting.
- Decide whether to expand to Copiale/German.

## Open Questions

- Should workflow macros be tools visible to the model, or internal modes
  selected by the harness?
- How many inner steps should be allowed before another model call is required?
- Should branch comparison be automatic after every held repair?
- How should the loop detect "readable enough" without ground truth?
- What is the minimum provider abstraction needed for OpenAI, Claude, and a
  local model without losing useful model-specific capabilities?
