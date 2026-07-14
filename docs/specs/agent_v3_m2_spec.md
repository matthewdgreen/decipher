# Spec: Agent Loop v3 — Milestone M2 (Episode Runtime)

Parent design: `docs/specs/agent_v3_design.md` (C3; C7's AnthropicSession +
within-episode server-state chaining; amendments **A1, A4, A7, A8, A9, A10,
A11 binding here**; the design's milestone list mislabels the discipline
sweep "(A5)" — it is A4; A5 proper is M4). M1 is landed at `b6262f6`
(`src/investigation/`). Spec author: Fable; implementer: coding sub-agent;
Fable-reviewed before implementation.

Conventions (binding): baseline is **1043 passed / 1 skipped**
(`PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`) — re-record before
starting; no pre-existing test may change outcome. Cited `file:line` may be
stale — locate constructs by the quoted identifiers/strings, never by line.
**No commits.** Report deviations (differences, skips, contradictions
found, with rationale). Acceptance compute uses the OpenAI key
(`.decipher_keys/openai_api_key`); keep live spend under ~$2, report actual
cost per run. Anthropic has no credits — fakes only.

## Scope

`src/investigation/episodes.py` (`EpisodeSpec`/`run_episode`, four kinds,
lead tools `episode_run`/`episode_install_branch`); A1 isolation; the A4
discipline sweep; A10 board single-writer; `AnthropicSession` + OpenAI
server-state chaining (C7); deferred M1 follow-ups (Part 7). v2 behavior
unchanged: the shared surfaces touched (`tools_v2.py`
constructor/dispatch, `loop_v2.py` rehoming, `model_provider.py`) must be
behavior-preserving under defaults, pinned by the existing suite.

## Part 1 — `EpisodeSpec` / `run_episode` (`episodes.py`; C3 + A9 + A7)

- `EpisodeBudget`: `max_tool_calls`, `max_output_tokens` (per send),
  `wall_clock_seconds`. `EpisodeSpec`: `kind`, `goal`, `inputs` (dict:
  `branches: list[str]`, optional `search_tool`, `context_note`),
  `toolset: list[str]`, `budget`, `result_schema`. Construction validates:
  toolset ⊆ `VALID_TOOL_NAMES`; disjoint from `EPISODE_EXCLUDED_TOOLS` =
  `{search_automated_solver, search_transform_candidates,
  search_transform_homophonic, search_quagmire3_keyword_alphabet}` (A9:
  wall clock is checked only *between* tool calls; long-running tools go
  to the M4 queue); free of all `meta_*`, `inspect_*`, `list_*` tools and
  the five hypothesis handlers (`workspace_create_hypothesis_branch`,
  `workspace_update_hypothesis`, `workspace_reject_hypothesis`,
  `workspace_hypothesis_cards`, `workspace_hypothesis_next_steps`) —
  episodes never declare, never see benchmark context, never write the
  board.
- `run_episode(spec, state, session=None, *, language, word_set, word_list,
  pattern_dict, verbose=False) -> EpisodeResult`, a fresh-context worker
  loop: (1) episode workspace + fresh executor (Part 2); (2) session from
  the kind registry, role `f"episode:{kind}"` (A7), system prompt = the
  episode contract (goal, rendered inputs, toolset, result schema, budget,
  "submit via `episode_submit_result`"); (3) `build_episode_context(spec,
  workspace, executor)` — deterministic, capped ~6k chars: branch card(s)
  via the side-effect-free `_branch_card`, a decode window, `context_note`;
  firewall-covered; (4) loop: send → execute tools → append tool results
  (native message dicts, lead-loop pairing rules), checking call count and
  wall clock between tool calls. `episode_submit_result` is a **virtual
  tool** handled by the runner (never the executor), appended to every
  episode's tool defs; input validated against `result_schema` by a small
  local validator (`type`, `properties`, `required`, `items`, `enum` — no
  jsonschema dependency).
- **A9 failure semantics** (structured results, never a lead crash):
  schema mismatch → tool result listing errors, one retry; second mismatch
  → `episode_failed(schema_mismatch)` with `raw_text`. Budget exhausted →
  one final tool-less send ("emit the result JSON now"); validation failure
  → `episode_failed(budget_exhausted)`. Handler exceptions already become
  error JSON in `executor.execute`; wrap the runner loop so unexpected
  exceptions → `episode_failed(runner_error)` with traceback.
  `KeyboardInterrupt` → append `episode_failed(interrupted)` to the ledger,
  merge budget entries, re-raise — the Part 6 dispatcher preserves the
  lead's R5 pairing of the in-flight `episode_run` tool_use with a
  `stopped` tool_result.
- `EpisodeResult`: `episode_id` (uuid12), `kind`, `goal`, `status`
  ("ok"|"episode_failed"), `failure_reason`, `result`, `summary` (hard cap
  800 chars ≈ 200 tokens), `branch_snapshots` (Part 2), `tool_call_count`,
  `elapsed_seconds`, `budget_entries`, `raw_text` (failures only),
  `transcript` (native; only under `DECIPHER_DEBUG_EPISODE_TRANSCRIPTS=1`
  — A11). On completion: append the ledger dict to `state.episode_ledger`;
  extend `state.budget_ledger` with `category == f"episode:{kind}"` entries
  (A7 — per-entry cost, mixed models correct). Episode tool calls log with
  `ToolCall.episode_id` (new optional field, additive) and `iteration` =
  the launching lead turn (A11).

## Part 2 — Episode isolation (A1)

- Workers never touch the lead's live Workspace or executor. Episode
  workspace = fresh `Workspace` over `state.cipher` holding ONLY
  `inputs["branches"]`, each deep-copied via the existing serialize/restore
  pair (`state._serialize_branch` → `_restore_branch_into`; Phase 0.4
  `copy_as` semantics); parent links to non-copied branches stay dangling.
  Fresh `WorkspaceToolExecutor` per episode, constructor-injected with the
  episode toolset (Part 3), `NoGatesPolicy()`, an episode-local
  `repair_agenda`, the **state-owned `FinalistSessionStore`**, NO board.
- **FinalistSessionStore moves to `InvestigationState`**: new field
  `finalist_sessions` (default factory). Executor gains param
  `finalist_sessions: ... | None = None`; `None` → fresh store (v2
  byte-identical). v3 lead and every episode executor share
  `state.finalist_sessions`, so a search episode's finalist session is
  reviewable/installable by the lead or a later episode. Add store
  `to_dict()`/`from_dict()` (payloads are JSON-safe by construction — they
  render into tool results), wire into state serialization, and update the
  M1 resume-identity test note (finalist sessions now survive resume).
- Integration is explicit: `branch_snapshots` = every episode branch
  created or changed (key/spans/order/pipeline/metadata), via
  `_serialize_branch`. The lead installs via `episode_install_branch` →
  `workspace.restore_branch` under a fresh name (default
  `f"{kind}_{episode_id[:6]}_{branch}"`, collision-suffixed); nothing
  auto-merges. Episode-local agenda additions ride in the packet
  (`agenda_additions`); the lead merges into `state.repair_agenda` (A10).

## Part 3 — Handler-output discipline sweep (A4)

Three surfaces, each behind NEW executor constructor params (v2 defaults
untouched):

- **Toolset-as-allowlist, neutral rejection.** Param `episode_toolset:
  set[str] | None = None`. When set, `execute()` rejects off-toolset calls
  with `{"error": "`<name>` is not in this episode's toolset.",
  "allowed_tools": [...]}` — none of the v2 gated-window essay (locate by
  `"Do not use local split/merge"`). Keep `_gate_hits` telemetry.
  `allowed_tool_names`/`set_allowed_tool_names` and their v2 text are
  untouched (v2 gated windows use them; v3 never calls them).
- **Next-tool hints computed against the active toolset.** One choke point
  in `execute()` (not a 52-site edit): when `episode_toolset` is set,
  post-filter the result recursively — drop from any `suggested_next_tools`
  list, and null any `recommended_next_tool`/`fix_tool` string, every name
  not in the toolset (drop emptied keys). Unset → byte-identical output.
- **Guard blocks policy-injected.** Extract the two finalize-phase guards
  (locate by `"Prefer meta_declare_solution. "` in
  `_tool_search_hill_climb`/`_tool_search_anneal`) into a
  `DeclarationPolicy` hook `finalize_guard(executor, branch, tool, args) ->
  dict | None`: `V2GatePolicy` returns the current dicts byte-identical
  (add a pin test first if none exists); `NoGatesPolicy` keeps the guard
  *logic* with neutral text — no `meta_declare_solution`, no
  `suggested_next_tools`; ends "re-call with justification=<reason> if
  further search is needed." Sweep `_search_declare_note` likewise via a
  `search_note(executor, search_kind) -> str` hook: V2 text unchanged;
  v3/episode text drops declare-tool references and off-toolset names.

## Part 4 — Episode kinds v1, session registry, AnthropicSession (A7, C7)

- `EPISODE_KINDS` registry: kind → {toolset, default budget (overridable),
  result_schema, contract template, tier tag}. v1 (`repair`/`verify` are
  M3/M5):
  - `survey` (10 calls/120 s): all `observe_*` + all `score_*` +
    `decode_show`. Result `{findings: [str], suspected_modes: [{mode,
    confidence, evidence}], recommended_next: [str]}`.
  - `search` (8/300): `inputs["search_tool"]` + its review/rate/install
    companions where they exist (a module-constant mapping, e.g.
    `search_word_repair_menu` → `search_review_word_repair_finalists` +
    `act_install_word_repair_finalists`; likewise the pure-transposition
    triplet) + `decode_show`, `score_panel`. Result `{improved: bool,
    best_branch: str|null, score_summary: object, finalist_session_id:
    str|null, notes: str}`.
  - `reading` (12/180): `decode_show`, `decode_letter_stats`,
    `decode_ambiguous_letter`, `decode_absent_letter_candidates`,
    `decode_unmapped_report`, `corpus_lookup_word`,
    `corpus_word_candidates`, `score_panel`, `score_dictionary`. Result
    (proto-Reading dict — the `Reading` artifact and its *application* are
    M3, A8) `{reading_text: str, fragments: [{window, text, confidence}],
    holes: [str], overall_confidence: number}`.
  - `compare` (8/120): `decode_show`, `score_panel`, `score_dictionary`,
    `workspace_compare`. Result `{ranking: [str], verdicts: [{branch,
    verdict, rationale}], winner: str|null}`.
- **kind→session-factory registry (A7)**: reuse
  `sessions.register_session_builder` with roles `episode:<kind>`. Default
  builder uses the lead's provider config (single-model operation — Ollama
  parity). `run_v3` gains `episode_models: dict[str, str] | None` (kind →
  model id). The fake harness registers one scripted fake per kind.
- **`AnthropicSession`**: routed by `make_lead_session` for provider
  "anthropic" (GenericChatSession keeps ollama/openrouter). Native messages
  with `cache_control` breakpoints at the C2 stable-prefix boundary:
  `build_lead_context` marks the stable-prefix text block
  `"cache_hint": True`; AnthropicSession converts it to `"cache_control":
  {"type": "ephemeral"}`; every send path strips `cache_hint` so no
  provider sees an unknown field (system + last tool are already cached by
  `ClaudeAPI.send_message`). `capabilities.cache_breakpoints = True`.
- **OpenAI within-episode server-state chaining** (C7; deferred from M1
  F8): `OpenAISession` gains a `server_state` mode used ONLY for episode
  roles on the Responses path (`_requires_responses_api`): first send full
  context with `store=True`, record response id + transmitted-block count;
  later sends pass `previous_response_id` + only the unseen block suffix.
  The lead stays stateless `store=False` + encrypted reasoning.
  `capabilities.server_state = True` for episode-role Responses sessions.
  Plumb optional kwargs (`previous_response_id`, `store`) through
  `OpenAIModelProvider._send_responses`; defaults byte-identical.

## Part 5 — Hypothesis-board single-writer (A10)

- New `HypothesisBoard` (`state.py` or `investigation/board.py`): cards
  `{id, branch, cipher_mode, mode_status, mode_confidence, mode_evidence,
  mode_counter_evidence, evidence_source, next_recommended_action,
  rejection_reason, tried: [str], pending: [str]}`; methods
  `create/update/reject/get/cards`. **Single writer**: all hypothesis
  mutations go through board methods; the board mirrors each card into
  `branch.metadata` under exactly the current keys (locate in
  `_tool_workspace_create_hypothesis_branch`) so metadata *read* sites
  (`_mode_scoped_suggestions`, `_context_cipher_family_tool_block`, cards,
  guards) need no sweep.
- The five `workspace_*_hypothesis` handlers become adapters over the
  board. Executor param `hypothesis_board: ... | None = None`; `None` →
  private board (v2 path — handler outputs and metadata writes
  byte-identical; existing hypothesis-tool tests pin this). v3 injects a
  state-owned board, replacing M1's read-only
  `hypothesis_board_from_workspace` projection (kept only as the board's
  `from_workspace` import path). State serialization round-trips the
  board; `context._render_hypothesis_board` renders from it.
  Constructor-inject `repair_agenda` at the same time: param
  `repair_agenda: list | None = None` replaces `loop_v3.py`'s attribute
  pokes (`executor.repair_agenda = ...`, `_next_repair_agenda_id`) — R8.

## Part 6 — Lead-side wiring (`loop_v3.py`, `context.py`)

- `V3_LEAD_TOOL_DEFINITIONS` = `TOOL_DEFINITIONS` + two defs (in
  `episodes.py`; v2 never sees them): `episode_run(kind, goal, branches,
  search_tool?, context_note?, max_tool_calls?)` and
  `episode_install_branch(episode_id, branch, as_name?)`. The lead loop
  dispatches `episode_*` itself before `executor.execute`, with R5-style
  interrupt pairing. `episode_run` is synchronous (async = M4); returns the
  ledger entry (status, result, summary, snapshot names, spend).
- Context builder: new section `_render_episode_ledger` (last 3 episodes —
  kind, goal, status, summary, snapshot names; capped ~2000 chars) between
  the hypothesis board and recent evidence (C2 section 6). v3 system
  prompt gains a short "Delegating" paragraph (the four kinds;
  results/snapshots must be explicitly integrated).
- Artifact (C8, additive): `RunArtifact.episodes: list[dict]` = the ledger;
  `budget_by_category` gains `episode:<kind>` rows;
  `scripts/inspect_artifact.py` renders a minimal episodes table. Firewall:
  extend `tests/test_ground_truth_firewall.py` — rendered episode context +
  contract prompt for a benchmark-backed cipher contain no ground truth
  (reuse `assert_no_ground_truth_leak`).

## Part 7 — Deferred M1 review follow-ups folded into M2

(Authoritative record: the "Review follow-ups (deferred to M2)" note at
the end of `docs/specs/agent_v3_m1_spec.md`; restated here as binding
work items.)

- **loop_v2 helper rehoming**: `loop_v3.py` imports
  `_best_branch_for_auto_declare`, `_branch_snapshot_for`,
  `_hypothesis_cards_for_artifact`, `_install_automated_preflight_branch`,
  `_tool_result_summary` from `agent.loop_v2`; `context.py` imports
  `_best_branch_for_auto_declare`. Move all five, behavior-preserving, to
  new `src/agent/loop_shared.py`; `loop_v2.py` imports from there;
  investigation modules import `loop_shared` — v3 no longer imports
  `agent.loop_v2`.
- **`export_transcript` native-items upgrade**: `_BaseSession.send` records
  only neutral `_collect_assistant_blocks` output. Upgrade: each exchange
  records the provider-native response items (Responses: raw output items
  incl. reasoning `encrypted_content`; Anthropic/chat: native content
  blocks) plus usage; export the system prompt once. Episode transcripts
  stay out of the artifact except under the A11 flag.
- **R7 test gaps** (three): (i) state round-trip pin for `token_order` +
  `transform_pipeline` branch fields; (ii) `agent/resume.py` v3-refusal
  guard test (refuses `loop_version=="v3"` artifacts); (iii) built-context
  → Responses-converter end-to-end test covering the merged
  `[tool_results..., text]` user-message shape the builder emits.
- **R8 fixes, each with a test**: (a) REAL BUG — on resume `run_v3`
  prefers the `language` param over `state.language`, so resuming a `de`
  state loads English word resources (the `dictionary`/`pattern` loading
  block at the top of `run_v3`); use `state.language` when `resume_state`
  is given. (b) `build_lead_context` returns live references — the final
  `messages.extend(exchanges[:-1])` assembly aliases
  `state.recent_exchanges` dicts; return copies. (c) Verify the
  `max_iterations=0` NameError at the `run_complete` emit is already
  fixed by the `turn` pre-init; pin with a test or record why dropped.
  Plus the Part 5 constructor-injection cleanups and dead imports left by
  the rehoming.

## Part 8 — Fake-session multi-context harness + tests

- Harness (extends `tests/test_loop_v3.py`'s `ScriptedSession`): scripted
  fakes registered per role via `sessions.register_session_builder` — one
  per episode kind plus the lead — so one test drives lead + N workers with
  distinct scripted contexts (design's M2 risk item: build before any
  real-model use). Fakes assert received blocks (contract present, toolset
  listed, no lead-transcript bleed-through).
- Tests (new `tests/test_episodes.py`; extend `test_loop_v3.py`,
  `test_model_sessions.py`, `test_investigation_state.py`) cover every
  MUST in Parts 1–7. Non-obvious ones called out: isolation proven by
  mutating an episode branch and asserting the lead workspace unchanged
  until install; every A9 path (schema double failure, both budget kinds
  via monkeypatched clock, handler exception in packet, interrupt → ledger
  entry + lead pairing); finalist-session sharing across an episode
  boundary + store round-trip; A4/A10 v2 paths byte-identical with the new
  params unset (pin tests first), plus suggestion filtering on nested
  dicts; OpenAI server-state suffix-only second send against a fake
  client; `episode:<kind>` budget categories under two different fake
  models (A7); the scripted survey→search→reading→compare end-to-end;
  firewall episode-surface test; the R7 gap tests.

## Acceptance (compute, report)

1. Full suite green: baseline 1043 passed / 1 skipped, zero regressions,
   plus new tests. Report final counts.
2. Scripted workflow (fakes, no network): lead script runs
   survey → search → reading → compare on a synthetic substitution cipher,
   installs the search snapshot, declares. Assert: 4 "ok" ledger entries;
   four `episode:<kind>` budget rows; compare names the installed branch;
   the reading result is stored, NOT applied (A8).
3. Real-model smoke (report cost, tokens, ledger): (a) direct `run_episode`
   for `survey` and `compare` on the cached testgen medium synthetic with
   `gpt-5.6-luna` workers — exercises Responses + server-state chaining
   live; verify the second send carries `previous_response_id`; report
   cached/uncached split; (b) one `decipher crack --provider openai
   --model gpt-5.5 --agent-loop v3` on the same synthetic — report whether
   the lead launched episodes (informational, not gating), solve status,
   and cost vs the M1 baseline ($0.07).

## Out of scope

`repair`/`verify` kinds (M3/M5); `Reading` artifact +
`hypothesis_apply_reading` (M3, entry-gated on Phase 2.2 — A8); experiment
queue + worker-budget arbiter (M4/A5); attestation/declaration wiring
(M5/A6); `meta_attest_reading_comprehensibility` retirement and any v2
behavior change (v2 retirement); v2-artifact → v3-state adapter (M6).

## Post-review amendments (BINDING — from the Fable spec review, READY WITH AMENDMENTS)

1. **(F2) Server-state protocol precision.** "Seen" = the input messages
   transmitted by a send PLUS the assistant message that send returned;
   the next suffix = messages after the last seen index. Before each
   chained send, check prefix consistency (the seen list must be an exact
   prefix of the new block list); on mismatch OR any mid-episode
   `ModelProviderError`, fall back to a full stateless resend
   (`store=False`) and continue stateless. Keep
   `include=["reasoning.encrypted_content"]` even with `store=True` (it
   enables the fallback and keeps transcripts self-contained). "Defaults
   byte-identical" means `store=None`/`previous_response_id=None`
   reproduce today's `_send_responses` kwargs exactly, pinned by
   fake-client kwargs assertions. Acceptance 3(a) additionally reports
   the response ids and the send-2 suffix item count. A9's schema retry
   and final tool-less send are just more chained turns; an episode
   re-launch starts a fresh chain.
2. **(F3) Full metadata contract for the board.** Adapters keep ALL
   gating/validation/tag/context-prior writes verbatim — workspace tags
   `hypothesis`/`mode:<mode>`/`rejected`/`superseded`, and metadata keys
   `hypothesis_notes`, `context_supported_mode`, `context_mode_prior`,
   `context_assumption_note`, `required_tools_before_rejection` alongside
   the Part 5 keys — routing ONLY the card fields through the board with
   its mirror. Pre-extraction pin test: capture the full metadata dict
   produced by create + update with `evidence_source="benchmark_context"`.
   `tried`/`pending` are call_log-derived in v2 and have no M2 writer —
   DROPPED from the M2 card (a later milestone may add them with a
   defined writer).
3. **(F4) Fork/install reconciliation.** `episode_install_branch` strips
   the board-mirrored card keys from restored metadata, and the lead loop
   runs a `board.sync_from_workspace()` adopt pass once per turn so
   hypothesis branches forked INSIDE an episode gain cards after install.
   Test: fork a hypothesis branch in an episode, install it, assert the
   board and metadata surfaces agree.
4. **(F5) TRUE deep copies for episode snapshots.** `_serialize_branch`
   shallow-copies `metadata` and passes `transform_pipeline` by
   reference — episode workspace construction must deep-copy the snapshot
   dicts (JSON round-trip or `copy.deepcopy`) before restore. The Part 8
   isolation test must mutate NESTED metadata and `transform_pipeline`
   inside the episode and assert the lead branch unchanged.
5. **(F6) Episode provider plumbing.** The lead's provider is passed
   explicitly into episode session construction; `episode_models` is
   restricted to same-provider model ids (validated at `run_v3` entry).
   `session_factory` lookup order: exact registered builder for
   `episode:<kind>` → default builder → `episode_failed(runner_error)`.
   Default routing: anthropic → AnthropicSession, openai → OpenAISession,
   else GenericChatSession.
6. **(F7) Store serialization details.** `FinalistSessionStore.to_dict()`
   persists the per-kind counters (session ids must not restart after
   resume); the round-trip test covers all four kinds; `find_id` stays
   identity-based (documented: payload identity does not survive resume).
7. **(F8) Filter coverage + lead neutral text.** The Part 3 suggestion
   filter applies on EVERY `execute()` return path — handler results,
   context-family blocks, gated rejections, and error dicts. The
   `NoGatesPolicy` neutral guard/note text applies to the v3 LEAD as well
   as to episodes.
8. **(F10) Dispatcher + result plumbing.** The lead dispatcher merges
   episode ToolCalls into `artifact.tool_calls` with `episode_id` set and
   `iteration` = the launching lead turn. Three visibility tiers,
   explicit: full episode tool calls in the artifact; ledger summary in
   state/context; snapshot packets integrated only on install.
   `episode_submit_result` gains a REQUIRED `summary` argument (truncated
   to 800 chars — it becomes the ledger summary). Per-kind
   `max_output_tokens` defaults live in `EPISODE_KINDS`. The local schema
   validator supports nullable types (`"type": ["string", "null"]`).
9. **(F11) `hypothesis_board` field transition.**
   `InvestigationState.from_artifact_dict` falls back to
   `HypothesisBoard.from_workspace(...)` when the artifact predates the
   board field (M1-era artifacts); the M1 resume-identity test keeps
   passing unchanged.
10. **(F12) Defaults + scope notes.** `run_episode`'s `language` (and the
    word resources) default from `state`; AnthropicSession is
    fake-test-only this milestone (no Anthropic credits).

## Deliverables

Files changed; suite counts (baseline vs final); scripted-workflow
assertion list; real-model smoke table (episodes run, cached/uncached
tokens, cost per run); deviations report. No commits.
