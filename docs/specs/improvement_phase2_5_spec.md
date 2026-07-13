# Spec: Improvement Program Phase 2.5 — Agent-Facing Word-Repair Menus

Parent plan: `docs/improvement_program_plan.md` (Phase 2.5, reprioritized
before 2.4). Spec author: Fable. Implementer: coding sub-agent.
Depends on: Phase 2b (runner integration, menu-only) — landed before this
starts. Design shape: the V3-M3 composite-action pattern
(`docs/specs/agent_v3_design.md` C4) built on the Phase 1
`FinalistSessionStore`.

## Why (evidence)

In a successful gpt-5.5 Borg run (95.9% char), 19 of 40 tool calls were
manual repair mechanics (plan→apply→merge/split→validate cycles with two
actuator errors). Meanwhile 2b proved the mechanical pipeline generates
good candidate menus whose *safe selection* requires a reader. This
phase hands the menu to the reading agent: one call generates and
evidence-tags repairs; the agent reviews and applies with judgment.

## Required behavior

### 1. `search_word_repair_menu` (new agent tool)

- Args: `branch` (required); optional `window_size`, `max_edits`,
  `max_hypotheses`, `max_hypotheses_per_window` (bounded overrides of the
  library defaults; validate ranges).
- Behavior: build a one-page group from the branch's effective state
  (same construction as the runner's 2b path — reuse, do not
  reimplement), call `propose_word_repairs` (LAZY import per the 2b
  constraint; model path via the runner's `zenith_native_model_path`
  resolver), install a `FinalistSessionStore` session
  (`kind="word_repair"`, id scheme `word_repair_N`) holding the packets.
- Returns a COMPACT review (token discipline is a hard rule — full
  packets stay server-side): session id, counts, `would_adopt` summary,
  and top-N rows (default 8) each with: rank, edit labels,
  adjudication_score, validation delta, acceptance verdict + reasons
  (short), collateral occurrence count, decoded preview (≤120 chars),
  and a short local-context snippet per edited word if the library
  provides one. Include the standard
  `primary_ranking_signal="agent_contextual_readability"` /
  `numeric_scores_role="supporting_evidence"` framing used by the other
  finalist reviews.
- Executor guardrails: the tool is substitution-family — respect the
  existing mode guardrails (blocked on periodic/transform/fractionation
  hypothesis branches unless `allow_mode_mismatch_repair=true`), same as
  other repair tools.

### 2. Review / rate / install triplet (via the shared machinery)

- `search_review_word_repair_finalists(search_session_id, start_rank,
  count)` — paging over the same compact row shape.
- Rating: extend the existing shared rate pattern (either
  `act_rate_word_repair_finalist` or fold into the shared rate tool the
  way pure-transposition shares `act_rate_transform_finalist` — pick
  whichever preserves the existing tools' JSON exactly; report choice).
  Ratings refresh the session packets (Phase 1 `_refresh_...` pattern).
- `act_install_word_repair_finalists(search_session_id, ranks,
  branch_prefix?)` — for each rank, FORK the source branch as
  `{source}_wr_rank{N}` and apply the candidate's full edit set to the
  fork's key (whole-candidate application; reject with a clear error if
  any label fails — mirror 2b's no-partial rule and masked-symbol
  guard). Metadata block `word_repair_finalist` {session id, rank,
  edits, adjudication_score, acceptance, agent rating}; tags
  `word_repair_finalist`, `wr_rank_{n}`.

### 3. Surface integration

- Add the three tools to `TOOL_DEFINITIONS` with tight descriptions;
  registry/consistency tests (Phase 0's `test_tool_dispatch.py`) will
  enforce handler naming automatically.
- `_mode_tool_menu`: add `search_word_repair_menu` to the foreground
  tools of the homophonic/nomenclator/substitution modes (exact mode
  keys per the existing menus; NOT periodic/transform modes).
- Toolkit prompt section (`prompts_v2.py` `_TOOLKIT_FULL/_COMPACT`): one
  or two lines under the search/act namespaces describing the menu →
  review → rate → install flow as the PREFERRED route when decoded text
  is mostly readable but locally damaged. Do not add new gates,
  preflights, or panel nags.
- **TOOLS.md**: add entries for the new tools (name, description, param
  table, usage notes) and update the tool count — CLAUDE.md mandates
  this. Also update the `tools_v2.py` count in CLAUDE.md's Key Files
  line (it is stale at "78"; count the real number).

### 4. Tests

- Fake-provider flow test (pattern: the three existing finalist-flow
  tests): search → review page 2 → rate → install; assert session id
  scheme, compact row fields, fork branch name/metadata/tags, and the
  packet-rating refresh.
- No-leak guard: `"packet" not in json.dumps(...)` for every new tool's
  result (Phase 1 F2 pattern), plus row count ≤ requested count.
- Guardrail test: tool blocked on a periodic-tagged branch without the
  override flag.
- Whole-candidate + masked-symbol rejection tests on install.
- Firewall: extend the leak coverage to the new tools' results.

## Acceptance evidence (compute, report)

One real agentic run (user-approved spend class):
`decipher benchmark <root> --test-id borg_single_B_borg_0109v --agentic
--model gpt-5.5 --max-iterations 20 --verbose`, artifacts kept. Report:
whether the agent used `search_word_repair_menu`, the repair-workflow
tool-call count vs today's baseline (19 of 40 calls; count the same
"repair/resegment/reading/attest/word" name filter), char/word accuracy,
tokens/cost. One run is anecdote, not proof — report it as such; the
multi-seed comparison belongs to the program's later gates.

## Out of scope

Multipage groups (2.4), auto-adoption changes (2b's menu-only default
stands), packet size trimming (F3), v3 episode packaging (M3 will
consume these tools' library layer, not their v2 JSON).

## Review follow-ups (deferred; from the Fable review — LAND AS-IS)

- Add `search_word_repair_menu` to `_KEYED_TABLEAU_OFF_FAMILY_TOOLS`
  and/or make the shared rate tool's context block session-kind-aware
  (narrow inconsistency under declared keyed-tableau priors).
- Install-time key drift: snapshot the base key (or its hash) in the
  session and warn on mismatch if the branch was hand-edited between
  menu and install.
- Registry test pinning `_mode_tool_menu` foreground strings to
  `TOOL_DEFINITIONS` (pre-existing gap; would have caught the
  observe_patterns staleness class automatically).
- Cosmetic: install's `cipher_mode` default mislabels mode-less mono
  branches (same pattern as null-mask install).

## Deliverables

Files changed, suite counts (record baseline first), the acceptance-run
comparison, deviations. No commits.
