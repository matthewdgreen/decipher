# Spec: Agent Loop v3 — Milestone M1 (State + Lead Loop + Session Seam)

Parent design: `docs/specs/agent_v3_design.md` (C1, C2, C7; amendments
A2, A3, A7, A11, A12 are binding here). Spec author: Fable. Implementer:
coding sub-agent. This spec receives its own Fable review before
implementation (the C7 scrutiny gate).

## Scope

`run_v3`: a lead loop over context REBUILT from state each turn, using
the existing 92-tool executor directly (no episodes — M2), talking to
models through the new `ModelSession` seam. v2 is untouched in behavior;
the one shared-surface change is the declaration-policy extraction (A2),
which must be behavior-preserving for v2.

New package `src/investigation/`: `state.py`, `context.py`,
`sessions.py`, `loop_v3.py`. No file in `src/agent/` changes except the
A2 extraction in `tools_v2.py` (+ its wiring in `loop_v2.py` if needed).

## Part 1 — `InvestigationState` (`state.py`, per C1 + A3)

- Fields: `cipher` (full CipherText serialization: cipher/plaintext
  alphabet symbols, tokens, separators, word structure — resume NEVER
  parses prompts), `workspace` (existing Workspace object; serialized as
  full branch snapshots incl. `token_order`, `transform_pipeline`,
  `metadata`), `hypothesis_board` (this milestone: a thin typed wrapper
  over the existing branch-metadata hypothesis cards — single-writer
  refactor is M2/A10; M1 only READS them for context), `evidence_log`
  (append-only typed entries; M1 writes diagnostic-preflight and
  turn-summary entries), `episode_ledger` + `experiment_queue` (present,
  empty — schema only), `budget_ledger` (A7: entries of {category,
  provider, model, input_tokens, output_tokens, cache_read_tokens};
  cost derived per entry via `estimate_provider_cost`).
- `to_artifact_dict()` / `from_artifact_dict()` round-trip; loading a
  serialized state and continuing IS the resume path (test: serialize
  mid-run, reload, next turn's rebuilt context is identical).

## Part 2 — Context builder (`context.py`, per C2)

- `build_lead_context(state, token_budget) -> list[Message]` rendering,
  in stable order with per-section char budgets: (1) task framing +
  language notes (reuse `prompts_v2` language notes; the v3 system
  prompt is a NEW, much shorter brief — write it, ~1/3 of v2's, no
  cipher-mode playbooks, no reading-repair discipline essay: the
  hypothesis board and tool results carry state, and M1 targets
  substitution-family synthetics), (2) cipher rendering + diagnostic
  fingerprint, (3) top-K branch cards (reuse the existing card
  renderer via the executor), (4) hypothesis board summary, (5) last N
  evidence-log entries, (6) the last 2 turns' tool exchanges verbatim,
  (7) a rotating full-decode window (rotate offset per turn so long
  ciphers are fully seen over ⌈len/window⌉ turns — the middle-blindness
  fix).
- Deterministic given (state, turn); pure function; unit-tested with
  golden-ish structural assertions (section presence, budget respect,
  rotation coverage).

## Part 3 — `ModelSession` seam (`sessions.py`, per C7 + A7 + A12)

- `SessionCapabilities` dataclass {server_state, reasoning_passback,
  cache_breakpoints, strict_tools}; `ModelSession` Protocol with
  `send(blocks, tools, max_tokens) -> ModelResponse-like events` and
  `usage_entries() -> list[BudgetEntry]`; `export_transcript() -> dict`
  (provider-tagged native transcript for the artifact).
- `OpenAISession`: wraps the existing `OpenAIModelProvider` paths —
  responses-native for gpt-5.6*, chat otherwise; stateless
  encrypted-reasoning passback REUSED from the landed provider_extra
  mechanics (do not duplicate the converters — the session owns history
  and calls the existing module-level converters). Server-side chaining
  (`previous_response_id`) is M2 (episodes); M1 sessions are stateless.
- `GenericChatSession`: wraps chat-completions providers (anthropic via
  ClaudeModelProvider, ollama, openrouter) — the neutral behavior as
  one implementation.
- `session_factory(kind_or_role, config) -> ModelSession` registry; M1
  has a single role ("lead"). Fake sessions for tests implement the
  Protocol per provider shape.
- Budget entries recorded per send; cost never recomputed from totals.

## Part 4 — Lead loop (`loop_v3.py`) + A2 extraction

- `run_v3(cipher_text, language, model_provider_or_session, ...)
  -> RunArtifact`: per turn — rebuild context, send, execute tool calls
  through a `WorkspaceToolExecutor` owned by the lead (v2 handlers
  unchanged), append results to the evidence log (turn summary =
  compact tool-name + status list), loop. Termination: declaration
  tools, max_turns, provider error → Phase 0 semantics preserved
  (`fallback_declared` + `auto_declared`; reuse
  `_best_branch_for_auto_declare` or an extracted equivalent).
- **A2 declaration-policy extraction (the shared-surface change)**: the
  gate cascade inside `_tool_meta_declare_solution`
  (`tools_v2.py:11553–11855` region) moves behind a
  `DeclarationPolicy` object on the executor: v2 loop injects
  `V2GatePolicy()` (the cascade verbatim — behavior-preserving, pinned
  by the existing declare-gate tests passing unchanged); v3 injects
  `NoGatesPolicy()` (declaration always allowed; confidence recorded).
  Executor default = V2GatePolicy so nothing changes for existing
  callers/tests that construct executors directly.
- Iteration mapping (A2): `executor.set_iteration(lead_turn)`,
  `set_max_iterations(max_turns)`.
- `ToolCall.iteration` = lead turn (A11); `artifact.messages` = the
  lead's logical transcript (rebuilt contexts NOT persisted; store the
  per-turn NEW content only: model output + tool results + the turn's
  evidence entry). Artifact gains `loop_version: "v3"`,
  `budget_by_category`, `session_transcript` (from
  `export_transcript()`).
- CLI: `--agent-loop v3` on `crack` and `benchmark` (default v2);
  benchmark runner passes through.

## Part 5 — Tests

- State round-trip + resume-identity; context builder determinism,
  budgets, rotation; session fakes (routing, budget entries, transcript
  export); A2: full existing declare-gate test set green (V2GatePolicy
  default), plus NoGatesPolicy unit test; fake-session v3 end-to-end on
  a tiny substitution cipher (scripted lead solves and declares; no
  gates fire); fallback_declared path on provider error; firewall: v3
  opening context leak-checked (reuse the helper).

## Acceptance (compute, report)

1. Suite green (record baseline first; zero failures; the A2
   extraction must not change any existing test's outcome).
2. Real-model check: `synth_en_250nb_s4`-class case (testgen medium/
   hard preset, cached) with gpt-5.5 on BOTH loops (`--agent-loop v2`
   then `v3`), 15 iterations/turns. Report: solved?, char accuracy,
   total cost, input tokens split cached/uncached (A12 — v3's win must
   show in COST, not raw tokens; report honestly if it doesn't),
   turns used, and the v3 system-prompt + rebuilt-context sizes vs
   v2's ~27k/turn.
3. One Borg page (borg_0109v, gpt-5.5, 20 turns) on v3 — report
   accuracy vs the session baseline (95.9/84.8-79.7 band) and the
   repair-workflow call fraction (the 2.5 menu tools are available to
   v3 automatically via the shared executor).

## Out of scope

Episodes/workers (M2), hypothesis-board single-writer (M2/A10),
composite hypothesis tools (M3), experiment queue execution (M4),
verify episodes (M5), v2-artifact resume adapter (M6), AnthropicSession
cache breakpoints (M2), deletion of anything v2.

## Post-review amendments (BINDING — from the Fable spec review, READY WITH AMENDMENTS)

1. **(F1/F2) `recent_exchanges` state field + passback integrity.**
   `InvestigationState` gains `recent_exchanges`: the last 2 turns'
   exchanges stored as NATIVE-format message dicts (assistant blocks
   incl. `provider_extra`, and their tool_result messages). Section 6 of
   the context builder renders these dicts UNTRANSFORMED (they are what
   the session converters consume — flattening to text would silently
   kill reasoning passback and break function_call pairing). Budget rule:
   drop the oldest WHOLE exchange, never split one (a tool_result
   without its tool_use, or a reasoning item without its siblings,
   400s on the Responses API). Capture-side reuse: hoist
   `_collect_assistant_blocks`/`_opaque_block_to_dict` from `loop_v2`
   to `model_provider.py` as a behavior-preserving move (add both files
   to the allowed-changes list) so v3 does not import from `loop_v2`.
2. **(F3) A2 extraction, corrected.** The cascade is at
   `tools_v2.py:12292–12607` (~10 gates), NOT the stale citation.
   `DeclarationPolicy.check(executor, args) -> block|None` is PURE LOGIC
   over executor-resident state — gate state (`_pending_declare_*`)
   stays on the executor because other tools discharge it
   (`workspace_branch_cards` ~5157, `hypothesis_next_steps` ~5071,
   resegment ~8549) and `loop_v2` reads it (~1621) for the panel; zero
   changes to discharge sites. The policy governs BOTH
   `meta_declare_solution` AND `meta_declare_unsolved` (its own 3-gate
   cascade at 12609–12716); `NoGatesPolicy` bypasses both. Before
   extracting, add one pin test each for `full_reading_workflow_required`
   and the multi-prereq `prerequisites_required` batch shape (currently
   unpinned).
3. **(F4) Acceptance item 2 harness.** Use `decipher crack
   --provider openai --model gpt-5.5 --agent-loop {v2,v3}` on the cached
   testgen ciphertext plus an explicit `score_decryption` comparison
   step (do NOT touch run_testgen_suite.py this slice).
4. **(F5) `repair_agenda` joins M1 state** (serialized + re-injected on
   load; v2 precedent: `inherited_repair_agenda`). Note in tests:
   `FinalistSessionStore` is executor-owned and does NOT survive resume
   until M2/A1 — the resume-identity test must not exercise
   review/rate/install sessions.
5. **(F6/F7) Context builder corrections.** Signature:
   `build_lead_context(state, executor, turn, token_budget)`. Branch
   cards via the side-effect-free `_branch_card(name)` renderer — NEVER
   the `workspace_branch_cards` tool handler (it discharges gate state).
   Rotating window defined in TOKENS: offset `(turn*window) mod len`,
   snapped to `effective_word_spans` boundaries when present, rendered
   for the branch chosen by `_best_branch_for_auto_declare` scoring.
6. **(F8) Session contract pins.** `send(blocks, ...)`: blocks are the
   complete logical context for the call; a `server_state`-capable
   session (M2) may transmit only the unseen suffix — additive detail,
   no protocol change. `SessionCapabilities` computed per
   (provider, MODEL): `reasoning_passback` true only on the Responses
   path (gpt-5.6*), false for gpt-5.5. Add a 2-turn gpt-5.6-sol live
   smoke to acceptance (cheap) so the Responses-native session path is
   exercised once for real.
7. **(F9/F10/F11) Artifact + scoping.** v3 artifacts carry
   `investigation_state` (the `to_artifact_dict()` output); v2
   `agent/resume.py` gets a one-line guard refusing
   `loop_version=="v3"` artifacts. Part 5's "no gates fire" claim is
   scoped to DECLARATION gates (context-family/null-mask/transform
   executor blocks still fire; v3 never calls `set_allowed_tool_names`).
   Acceptance 3: record the Borg baseline band in the report itself
   (95.9/84.8 and 95.9/79.7 from `artifacts/agentic_model_comparison/`)
   and define the repair-fraction numerator as calls matching
   `repair|resegment|reading|attest|word` over total tool calls.

## Deliverables

Files changed, suite counts, acceptance comparisons (the v2-vs-v3 table
+ prompt-size numbers), deviations. No commits.

## Review follow-ups (deferred to M2 — recorded post-hoc; the M2 spec
## reconstructed these from code before this note landed)

- R7 test gaps: token_order/transform_pipeline state round-trip pin;
  resume.py v3-refusal guard test; built-context -> Responses-converter
  end-to-end test (merged [tool_results..., text] user-message shape).
- R8 nits: max_iterations=0 NameError at run_complete emit; run_v3 uses
  the language param over state.language on resume; live references from
  messages.extend(exchanges[:-1]); resumed artifacts' totals include
  prior-run spend (documented, accepted).
- export_transcript upgrade: neutral blocks -> provider-native items.
- Rehome the five loop_v2 helper imports used by loop_v3 (to
  src/agent/loop_shared.py or similar) before v2 deletion.
