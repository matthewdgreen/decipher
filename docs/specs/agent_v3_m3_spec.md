# Spec: Agent Loop v3 — Milestone M3 (Hypothesis Actions + Reading)

Parent design: `docs/specs/agent_v3_design.md` (C4; the M3 milestone
paragraph; amendments **A1, A8, A9, A10 binding here**). M2 is landed at
`e699773`; the A8 entry gate is satisfied (Phase 2.2's
`src/analysis/word_hypothesis_repair.py` landed at `637a281`). M2-review
notes feeding this milestone: `EpisodeResult.agenda_additions` is exported
on every episode and merged by the lead on install — the repair kind
reuses that path unchanged; the `repair` kind itself was explicitly
deferred from M2 to here. Spec author: Fable; implementer: coding
sub-agent; Fable-reviewed before implementation.

Conventions (binding): baseline is **1086 passed / 1 skipped**
(`PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`) — re-record before
starting; no pre-existing test may change outcome. Cited `file:line` may
be stale — locate constructs by the quoted identifiers/strings, never by
line. **No commits.** Report deviations (differences, skips,
contradictions found, with rationale). Acceptance compute uses the OpenAI
key (`.decipher_keys/openai_api_key`); keep live spend under **$5**,
report actual cost per run. Anthropic has no credits — fakes only.

## Scope

`src/investigation/reading.py` (the Reading artifact) and
`src/investigation/actions.py` (three composite actions +
`execute_composite` dispatcher); the `repair` episode kind; the A8
reading-kind result-schema upgrade; lead/episode wiring. **v2 is
untouched**: no change to `TOOL_DEFINITIONS`, no new executor handlers, no
change to the seven v2 boundary actuators (`act_split_cipher_word`,
`act_merge_cipher_words`, `act_merge_decoded_words`,
`act_apply_boundary_candidate`, `act_resegment_by_reading`,
`act_resegment_from_reading_repair`, `act_resegment_window_by_reading`) or
the `boundary_projection` retry subsystem — they stay on the v2 path until
v2 retirement (M6+). Composite actions live ONLY on v3 surfaces:
`V3_LEAD_TOOL_DEFINITIONS` and episode toolsets; the one shared-surface
touch is additive (Part 2: toolset validation learns the composite
names). This milestone absorbs plan Phase 6 ("SUBSUMED BY V3-M3"): do
not build `act_project_reading` — `hypothesis_apply_reading` IS that
consolidation, and Phase 6.3's test matrix lands here (Part 8).

## Part 1 — Reading artifact (`reading.py`; C4 + A8)

- `ReadingFragment`: `start`/`end` (token indices into the branch's
  effective stream; `None` = whole text), `text` (proposed plaintext,
  A–Z + spaces), `confidence` (0–1), `label` (optional window tag — the
  M2 proto's `window` string survives here). `Reading`: `reading_id`
  (uuid12), `branch`, `source` (`"episode:<episode_id>"` or `"lead"`),
  `created_turn`, `fragments`, `holes: list[str]` (unresolved-span
  notes), `overall_confidence`. `full_text` property = fragment texts
  joined in `start` order. `to_dict()`/`from_dict()`.
- **State storage**: `InvestigationState` gains `readings:
  dict[str, dict]` (reading_id → `Reading.to_dict()`), serialized in
  `to_artifact_dict()`/`from_artifact_dict()` (absent key → empty — M2
  artifacts keep loading; extend the resume-identity test).
- **A8 schema upgrade** (`episodes.py` `_READING_SCHEMA`): fragment
  objects gain optional nullable-integer `start`/`end` fields alongside
  the existing `window` string; `required` stays `["text"]`. Extend the
  reading kind's contract text: report `start`/`end` token indices per
  fragment when known. Existing scripted-reading tests pass unchanged
  (additive schema).
- **Lead-side compilation**: in `loop_v3.py` `_dispatch_episode_run`,
  when a `reading`-kind episode returns `status=="ok"`, the lead compiles
  the result dict into a `Reading` (source=`episode:<id>`, branch = first
  input branch), stores it in `state.readings`, and adds `"reading_id"`
  to the returned JSON. Workers never write `state.readings` (A1).
- **Context**: new `_render_readings` section in `context.py`, after
  `_render_episode_ledger`: last 3 readings — id, branch,
  overall_confidence, hole count, ≤120-char preview; capped ~1200 chars.
  The v3 system prompt's "Delegating" paragraph gains one sentence:
  readings are applied with `hypothesis_apply_reading`, words tested
  with `hypothesis_test_word` (a nudge at the M1 Borg finding that
  repair machinery went unused, 2/56).

## Part 2 — Composite action surface (`actions.py`; C4)

- Three tool defs, `COMPOSITE_TOOL_DEFINITIONS`/`COMPOSITE_TOOL_NAMES`,
  defined in `actions.py` — NOT in `agent/tools_v2.py`, so v2's
  `TOOL_DEFINITIONS`/`VALID_TOOL_NAMES` and TOOLS.md are unchanged and a
  v2 run can never see or call them (pin test:
  `executor.execute("hypothesis_test_word")` → unknown-tool error).
- `execute_composite(name, args, *, executor, state_readings, turn,
  tool_use_id) -> dict` — the single dispatcher both hosts call
  (amendment 4): the lead (`loop_v3._dispatch_tool` checks
  `COMPOSITE_TOOL_NAMES` before `executor.execute`) and the episode
  runner's tool loop, which dispatches a composite ONLY when the name is
  in `spec.toolset` — anything else falls through to `executor.execute`,
  so the M2 neutral off-toolset rejection + `_gate_hits` telemetry apply
  for free. The dispatch site appends a ToolCall to `executor.call_log`
  with `elapsed_ms` and the SERIALIZED result string (`tools_v2._json` —
  both loops consume JSON strings), counting against `max_tool_calls`,
  with M2's `episode_id`/`iteration` stamping unchanged. Episode-side
  composite results pass through `_filter_next_tool_hints` against
  `spec.toolset` (menu packets carry `suggested_next_tools` naming
  v2-only tools). `state_readings` = `state.readings` for the lead; for
  episodes an episode-LOCAL dict seeded from `inputs["reading"]` (A1 —
  Part 6). Composites operate ONLY on `executor.workspace` (the episode
  copy inside workers), so isolation is inherited from M2.
- Composites may reuse the executor's side-effect-free private helpers
  (`_branch_card`, `_compute_quick_scores`, `_decoded_preview`,
  `_branch_word_repair_mask`, `_word_repair_menu_config`,
  `_reading_validation`) — v3-only coupling, documented in the module
  docstring. They must NOT touch v2 loop discipline
  (`_seen_resegment_proposals`, `_pending_declare_prerequisites`,
  gate/panel state).
- Lead wiring: `V3_LEAD_TOOL_DEFINITIONS = TOOL_DEFINITIONS +
  [EPISODE_RUN_TOOL, EPISODE_INSTALL_TOOL] + COMPOSITE_TOOL_DEFINITIONS`.
  Episode wiring: `_validate_episode_toolset` accepts names in
  `VALID_TOOL_NAMES ∪ COMPOSITE_TOOL_NAMES`; the episode tool-def filter
  (`[d for d in TOOL_DEFINITIONS if d["name"] in toolset]`) draws from
  `TOOL_DEFINITIONS + COMPOSITE_TOOL_DEFINITIONS`. Composite names clear
  `_FORBIDDEN_PREFIXES` by construction.

## Part 3 — `hypothesis_apply_reading` (absorbs Phase 6)

`hypothesis_apply_reading(branch, reading_id? | reading_text? |
fragments?, window?, as_name?, dry_run=false)` — compile an accepted
Reading into key edits + boundary changes in ONE step, on a **fork**
(never in place; default name `reading_<id6>_<branch>`,
collision-suffixed). Exactly one of `reading_id` (looked up in
`state_readings`) or inline `reading_text`/`fragments` is required;
`window` = `{start, end}` token indices scoping the whole call, and each
fragment additionally scopes itself via its own `start`/`end`.

- **Normalization**: uppercase, collapse whitespace, words =
  whitespace-split; non-plaintext-alphabet characters → structured
  error. Multisym plaintext alphabets → structured `unsupported` error
  (v2 resegment's restriction; quote-locate by `"single-character
  plaintext alphabets only"`).
- **Alignment** (per fragment, against the branch's effective decoded
  stream for that span — the amendment-2 shared definition; `?` for
  unmapped). Alignment runs over SPACE-STRIPPED character streams;
  proposed word-boundary positions are carried separately and
  re-projected through the alignment. Proposed char count equal to span
  length → direct positional alignment. Else if `abs(delta) <= max(2,
  ceil(0.02 * span_len))` → banded global alignment, band
  `max(3, abs(delta) + 1)` (a fixed band of 3 leaves NO valid DP path
  once the tolerance exceeds it, at span_len ≥ 151): `?` matches
  anything, mismatch cost 1, gap cost 2; on the designed fixtures gaps
  appear only where counts force them. Traceback is deterministic: on
  cost ties prefer diagonal over gap, emitting gaps as late as possible;
  a boundary landing at a gap column maps to the token index after the
  last aligned token to its left. Larger deltas → structured
  `count_mismatch_too_large` error with both counts and a
  first-divergence preview (the model re-scopes with `window`; NO
  v2-style partial safe-prefix apply and NO count-retry loop in v3).
- **Edit derivation (auto-detect)**: each aligned position where
  proposed differs from decoded (incl. decoded `?` → new mapping)
  accumulates symbol→letter on the effective cipher symbol there. One
  symbol assigned two letters → majority wins, ties → symbol dropped;
  every conflict reported. Gap positions produce no edit and are
  recorded as `holes`. `character_preserving = (edits == [])` — then the
  call is boundaries-only (cross-check the read-only
  `_reading_validation` verdict in tests).
- **Boundary compilation**: proposed word boundaries map through the
  alignment to token indices; spans inside the scoped window are
  replaced (`workspace.set_word_spans` on the fork), spans outside stay
  byte-identical (window-scoped case).
- **Result packet** (CandidatePacket-compatible dict, kind
  `"reading_application"`): status, fork name (`null` on `dry_run`),
  `character_preserving`, `edits` (sorted `SYMBOL=X` labels),
  `conflicts`, `holes`, `boundary_change_count`, `alignment` {gaps,
  mismatches}, before/after `_compute_quick_scores`, and a diff preview
  (reuse `word_hypothesis_repair.changed_excerpt`). `dry_run=true`
  computes everything, creates nothing — absorbing
  `decode_validate_reading_repair`'s role on the v3 path.

## Part 4 — `hypothesis_test_word` (the Phase-2 library's agent face)

`hypothesis_test_word(branch, word, word_index? | char_start?,
install=false)` — same-length word probe; the located span's length must
equal `len(word)`, else a structured error naming both. Span location is
in decode coordinates and maps into projection coordinates per
amendment 2 (spans containing unmapped/masked tokens → structured
error). **Reuse the Phase 2.5 shared helpers — do not reimplement**:
build the page group and baseline exactly as
`_tool_search_word_repair_menu` does
(`automated_runner.build_word_repair_menu` + `_branch_word_repair_mask`,
`_word_repair_menu_config`, `zenith_native_model_path`), then:

- **Menu-backed path**: if a menu packet proposes the same
  (span, target), return THAT packet verbatim (plus composite envelope),
  `menu_backed: true` — the design's acceptance ("composite matches
  `word_hypothesis_repair` library outputs exactly") holds by
  construction.
- **Injected path** (agent's word not menu-proposed — parity by
  construction here too, amendment 1): the composite calls a small
  ADDITIVE library entry point in `analysis/word_hypothesis_repair.py` —
  either `propose_word_repairs(..., injected_candidates=[(start, end,
  target)])` (injected candidates appended after
  `generate_word_hypotheses`, then flowing through the IDENTICAL
  edit-set / rescore / adjudication / acceptance / packet plumbing) or a
  public `score_injected_word_hypothesis(...)` wrapper reusing
  `_score_variant_row` and the same baseline/adjudication/acceptance/
  packet code. The composite never re-composes scoring outside the
  library. An injected candidate whose `implied_edits` come back empty
  yields the structured `no_valid_edits` verdict (masked/stable symbol
  or conflicting assignment). Flag `menu_backed: false` and
  `in_dictionary: false` when the word is not in the repair dictionary;
  non-dictionary rank semantics live inside the library (amendment 1).
- **Install**: when `install=true` and the verdict is not
  `no_valid_edits`, fork (`wordtest_<n>_<branch>`) and apply via
  `automated_runner.apply_word_repair_edits` — inheriting its
  whole-candidate masked-symbol rejection (reported, not raised).
  `install=false` creates nothing. Packet: CandidatePacket-compatible,
  kind `"word_hypothesis"`, provenance {branch, span, word, menu_backed},
  `solver_scores` deltas, adjudication summary, verdict, changed-excerpt
  preview, installed fork name or null.

## Part 5 — `branch_adjudicate`

`branch_adjudicate(branches, include_window=false)` — read-only,
packet-based comparison table (2–8 branches; structured error outside
that range or on unknown names). One row per branch: mapped count,
`_compute_quick_scores` (dict_rate/quad), tags, board card status via
the executor's injected board (omitted inside episodes — no board
there), latest stored Reading for the branch (id + overall_confidence)
when `state_readings` has one, ≤160-char `_decoded_preview` excerpt,
optional window text. Plus a `deltas` block against the first-listed
branch and a deterministic `ranking` by (dict_rate, quad). No verdict is
imposed — the model adjudicates; the rows are the packet.

## Part 6 — The `repair` episode kind

- `EPISODE_KINDS["repair"]`: toolset `{hypothesis_test_word,
  hypothesis_apply_reading, branch_adjudicate, decode_show,
  decode_letter_stats, decode_unmapped_report, corpus_lookup_word,
  corpus_word_candidates, score_panel, score_dictionary}`; budget
  `EpisodeBudget(12, 4096, 240.0)`; tier `"repair"`; contract: compile
  the given Reading / word hypotheses into applied edits on forks,
  verify collateral, report. Also add `branch_adjudicate` to the
  `compare` kind's toolset (additive; existing compare tests unchanged).
- Session-factory registration needs NO new code:
  `session_factory("episode:repair", …)` default-routes (M2 F6) and
  `run_v3`'s `episode_models` validation reads `EPISODE_KINDS`; the fake
  harness registers a scripted `episode:repair` session.
- **Reading hand-off (A1)**: `EPISODE_RUN_TOOL` schema gains optional
  `reading_id`; `_dispatch_episode_run` resolves it against
  `state.readings` (unknown id → structured error) and injects the
  reading DICT as `inputs["reading"]`. `run_episode` seeds the
  episode-local readings map passed to `execute_composite`;
  `build_episode_context` renders a reading summary section when present
  (id, confidence, holes, ≤400-char text).
- **Result schema** `_REPAIR_SCHEMA`: `{applied: boolean, best_branch:
  string|null, edits: [string], verdicts: [{action, target, verdict,
  rationale?}] (required: action, target, verdict), collateral: object
  (adjudication summaries of applied sets), notes: string}`; required:
  applied, best_branch, edits, verdicts, notes. `agenda_additions` need
  no schema slot — the M2 export carries them; composites inside an
  episode may append follow-ups to `executor.repair_agenda`. Repaired
  branches leave as `branch_snapshots`; nothing lands until
  `episode_install_branch` (A1).

## Part 7 — Artifact / observability

Readings ride in `artifact.investigation_state` automatically; also add
a top-level `artifact.readings` list (additive, like `episodes`) and a
minimal readings table + composite-call rendering to
`scripts/inspect_artifact.py` (standing C8 requirement). Composite
ToolCalls: lead-side stamp `iteration` = lead turn, episode-side stamp
`episode_id` (A11).

## Part 8 — Tests

New `tests/test_hypothesis_actions.py` + `tests/test_reading.py`; extend
`test_episodes.py`, `test_loop_v3.py`, `test_investigation_state.py`,
`test_ground_truth_firewall.py`. Cover every MUST above; called out:

- **Phase 6 matrix** on `hypothesis_apply_reading` (synthetic cipher,
  known key): (a) char-preserving → boundaries-only fork, zero edits,
  `_reading_validation` agrees; (b) char-changing same-count → correct
  symbol edits incl. `?`-fills, conflict majority/tie handling; (c)
  window-scoped → spans and key outside the window byte-identical; (d)
  deliberately miscounted (±1 doubled/dropped letter) → banded alignment
  applies the rest, gap reported as hole, no crash, no position cascade;
  (e) too-large mismatch → `count_mismatch_too_large`; (f) multisym
  guard; (g) a `?`-heavy stream (several unmapped symbols) through the
  banded path — deterministic result, `?`-fills where aligned, holes
  where gapped.
- **Parity** (design M3 acceptance): a menu-representable word →
  `hypothesis_test_word` packet equals the `search_word_repair_menu`
  packet for the same span/target (edits, acceptance verdict,
  adjudication numbers), compared MODULO the `_filter_next_tool_hints`-
  filtered hint fields (amendment 4). Plus an injected-path
  (non-dictionary) case, a masked-symbol `no_valid_edits` case, an
  unmapped/masked-span error case (amendment 2), and `install=true`
  fork-and-apply incl. the whole-candidate rejection surfaced.
- **Boundary misuse zero by construction**: the seven Scope-listed v2
  actuators appear in NO episode kind's toolset and are never called
  across the v3 scripted suite (assert over fake-run call logs);
  `COMPOSITE_TOOL_NAMES ∩ VALID_TOOL_NAMES == ∅` and the current
  `TOOL_DEFINITIONS` count are pinned by test (amendment 5a).
- **Repair episode end-to-end (scripted)**: reading episode → Reading
  stored → repair episode with `reading_id` → `hypothesis_apply_reading`
  in the worker → lead workspace UNCHANGED until `episode_install_branch`
  → install merges `agenda_additions`; both ToolCall stampings verified.
- **Firewall**: extend `test_ground_truth_firewall.py` — for a
  benchmark-backed cipher, the repair-episode contract + rendered context
  (incl. the injected reading section) and all three composites' result
  packets pass `assert_no_ground_truth_leak`.
- State: readings round-trip; an M2 artifact (no `readings` key) loads.

## Acceptance (compute, report)

1. Full suite green: baseline 1086 passed / 1 skipped, zero regressions,
   plus new tests. Report final counts.
2. Scripted workflow (fakes, no network): survey → reading (Reading
   stored) → repair (applies it) → install → `branch_adjudicate` →
   declare; assert ledger, readings, agenda merge, zero
   boundary-actuator calls.
3. Real-model run: one `borg_0109v` v3 run, `--provider openai --model
   gpt-5.5`, 20 turns. Report: char/word accuracy vs the session
   baselines (v2 band 95.9/84.8–79.7; v3-M1 91.0/66.7); the repair-turn
   tax — repair-fraction per the M1 definition (calls matching
   `repair|resegment|reading|attest|word` over total; baselines: v2
   19/40 → 11/35 with the 2.5 menus, v3-M1 2/56), composite-action call
   counts, repair/reading episodes launched; total cost and cache-read
   split. Informational, not gating (borg word accuracy swings ±20
   run-to-run; multi-seed is M6) — but state plainly whether the
   composites + repair episodes were USED and whether they reclaimed
   repair turns. Total live spend across all attempts under $5.

## Out of scope

Experiment queue (M4/A5); verify episodes, attestation, declaration
wiring (M5/A6); any v2 behavior change, incl. boundary-actuator deletion
and `decode_validate_reading_repair` retirement (v2 retirement);
v2-artifact → v3-state adapter (M6); exposing composites to v2; multipage
page groups in composites (single-page only, as the 2.5 menu).

## Post-review amendments (BINDING — from the Fable spec review, READY WITH AMENDMENTS)

Bundles A-1, A-3, A-4 are folded into Parts 2–4 and 8 above; this
section records their residue and is normative for A-2 and A-5.

1. **(A-1) Parity by construction on both paths — library details.** The
   entry-point choice (`injected_candidates` kwarg vs.
   `score_injected_word_hypothesis`) is the implementer's; either way it
   is ADDITIVE (existing `propose_word_repairs` callers byte-identical,
   pinned). Non-dictionary rank semantics are defined ONCE inside the
   library: when `dictionary_rank < 1`, `rank_bonus = 0.0` and
   `damage_score = 0.0` wherever rank feeds a fractional power — the
   review proved `(-1)**0.35` yields a complex number and crashes
   `max()`. The collateral word-island index is built from
   `load_dictionary(path, 3, max_word_len)` (min_len 3), NOT the
   hypothesis dictionary. The menu path passes EMPTY `consensus` — the
   injected path must match. `adjudicate_repair` already calls
   `target_word_repairs` internally — the composite makes NO separate
   call (Part 4's earlier draft implied one; this supersedes it).
2. **(A-2) Decode↔projection coordinate mapping.** The "effective
   decoded stream" (Parts 3–4) is the branch's effective token stream
   decoded per current key, `?` for unmapped — decode coordinates.
   `project_text_sources` SKIPS masked and unmapped tokens, so
   projection positions diverge from decode positions whenever ANY token
   is unmapped: map a decode-coordinate span into projection coordinates
   by counting only mapped, unmasked tokens before span start. A located
   span that CONTAINS unmapped or masked tokens returns a structured
   error naming the offending token positions (test pinned in Part 8's
   parity bullet).
3. **(A-3) Alignment precision — folded into Part 3.** Band
   `max(3, abs(delta)+1)`; space-stripped alignment alphabet with
   boundary positions carried separately; deterministic traceback
   (diagonal preferred over gap on ties, gaps as late as possible);
   boundary-at-gap mapping rule; the shared stream definition is
   amendment 2's; the "gaps only where counts force them" claim is
   scoped to the designed fixtures, not asserted universally.
4. **(A-4) Dispatcher contract — folded into Part 2.** `tool_use_id`
   parameter; call_log append with `elapsed_ms` + `tools_v2._json`-
   serialized result; episode-side dispatch only when the name is in
   `spec.toolset` (else executor fall-through); `_filter_next_tool_hints`
   on episode-side composite results. Part 8's parity assertion compares
   packets MODULO the filtered hint fields.
5. **(A-5) Small pins.** (a) v2-untouched operationalized: a test
   asserts `COMPOSITE_TOOL_NAMES` is disjoint from `VALID_TOOL_NAMES`
   and pins the current `TOOL_DEFINITIONS` length. (b) Reading
   confidences arriving as strings are coerced with `float()` and
   clamped to [0, 1]; unparseable strings become 0.5. (c) A fragment
   whose `start`/`end` extends outside the call-level `window` is a
   structured error naming the fragment index and both bounds — no
   silent clipping. (d) An inline reading (no `reading_id`) gets an
   ephemeral uuid12 for fork naming only, never stored in
   `state.readings`. (e) `actions.py` imports `automated.runner` and
   `analysis.*` lazily inside functions, with a comment matching the
   runner's "Lazy imports (binding constraint)" pattern. (f) The Part 4
   menu build MAY be cached per (branch, key-hash) within one episode or
   lead turn; without the cache, the episode wall-clock budget is the
   accepted backstop for repeated probes.

## Deliverables

Files changed; suite counts (baseline vs final); Phase 6 matrix + parity
assertion list; the Borg run table (accuracy, repair fraction, composite
usage, cost); deviations report. No commits.
