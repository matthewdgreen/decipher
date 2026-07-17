# CLI-3 — Human-First Narrate Display

Status: final. Authority for the CLI-3 implementation. Base: HEAD `d7e1376`.

## Motivation (user directive, 2026-07-17)

The default CLI output truncates human-readable explanations and prioritizes
precise tool calls. Reverse that, Claude-Code-style: the DEFAULT display is a
clear human narrative of what the agent is doing, why, what happened, and
where the problems are — live. Precise tool calls (exact names + args) move
behind the existing verbose flag. A second, standing problem: run artifacts
must be interpreted after the fact (by an LLM or the analyzer script) to learn
what happened; that is backwards. The CLI itself must tell the human what
happened, ending with an analyzer-grade digest so post-hoc interpretation is
the exception, not the routine.

## Scope

`src/agent/narrate.py` (the default renderer), `src/agent/display.py`
(glosses/summaries), minimal emit-site changes in
`src/investigation/loop_v3.py`, and test updates. The `pretty`, `raw`, and
`jsonl` displays are UNCHANGED. v2 runs share this renderer: every handler
must keep working when v2-only events arrive (no new required payload keys —
all new payload reads use `.get` with fallbacks).

## Part 1 — Behavior contract

### Default (non-verbose) — the human narrative

Line vocabulary (order of a typical turn):

1. **Narration** — the agent's own turn-opening sentence(s), FULL text, cyan,
   `“ ` prefix, wrapped at ~100 cols. Never truncated in the renderer.
   (Emit-site cap raised — Part 3.)
2. **Action lines** — one per lead tool call, plain English, `⏺ ` prefix:
   what the agent is DOING, not the tool name. Built by a new
   `describe_tool_action(tool, args)` (Part 2). Examples:
   - `⏺ Launching a verify episode: fresh reader judging whether 'automated_preflight' reads as real Latin`
   - `⏺ Queuing an automated_solver experiment on branch 'main'`
   - `⏺ Attempting a validated repair of 'automated_preflight' bound to reading 8c1a…`
   - `⏺ Comparing branches: automated_preflight vs alt_targeted_auto`
   - `⏺ Declaring the solution on branch 'automated_preflight'`
   No numbered index, no `tool(args)` at default.
3. **Result lines** — `  ↳ ` under the action, plain English, dim unless it
   is a problem. Use the existing `summarize_tool_call` output as the base,
   EXTENDED (Part 2) so that:
   - blocked results read as problems with guidance:
     `  ↳ blocked: declaration needs a fresh positive attestation — reader does not accept as solution; route repair/compare/broaden` (yellow)
   - episode completions read as outcomes (Part 3 digest):
     `  ↳ verify: NEGATIVE — "Latin-like but not acceptable as real Latin" (lang 0.70, recoverability 0.25, damage distributed → broaden)`
     `  ↳ search: no improvement — no better Latin substitution branch found`
     `  ↳ reading: proposed "PLERISQUE VERO …" (confidence 0.7)`
     `  ↳ repair: applied 2 edits, winner alt_repair_3`
   - experiment results summarize in words, not config dumts.
4. **Problem lines** — always shown, never verbose-gated (Part 4 list), with
   `! ` prefix and yellow/red color.
5. **Workflow transitions** — new event (Part 3):
   `  ⤷ workflow: searching → repair_required (on automated_preflight)` (dim).
   Emitted only when the phase CHANGES between turns.
6. **Decode preview** — keep the existing changed-only decode line.
7. **Ticker** — keep the existing cumulative `[$cost | tokens]` ticker, but
   attach it to result lines (as today) rather than adding new line types.
8. **Preflight, declaration, finish blocks** — keep, with Part 5's digest
   appended to finish.

Suppressed at default (moved to verbose): numbered `N │ tool(args)` lines,
episode-internal per-tool `↳ kind ep1 │ tool(args)` lines, snapshot/budget
lines, first-use glosses (the action line replaces the gloss role).

### Verbose (`-v` / existing `renderer_verbose`) — everything above PLUS

- The numbered `N │ tool(args)` line (current format, full args) directly
  UNDER each action line.
- Episode internals: the current `↳ kind epN │ tool(args)` lines and
  `episode_submit → status` lines.
- Snapshot lines (`best=… dict=… quad=…`, branch roles) and budget lines.
- Full multi-line decode previews (current verbose behavior).

The default:verbose relationship is strictly additive — verbose never
REPLACES a default line, it interleaves detail under it. (This keeps the
narrative readable in both modes and makes tests composable.)

## Part 2 — `src/agent/display.py`

1. New `describe_tool_action(tool: str, args: dict) -> str`: an args-aware,
   plain-English present-progressive sentence for EVERY v3 lead tool and the
   common v2 tools. Reuse `describe_tool_gloss`'s table as the fallback: when
   no specific pattern exists, fall back to the gloss text, else to
   `f"Running {tool}"`. Specific patterns required (v3 lead set):
   - `episode_run`: "Launching a {kind} episode: {goal}" (goal truncated 90).
   - `episode_install_branch`: "Installing episode branch '{branch}' as '{as_name}'".
   - `repair_transaction`: "Attempting a validated repair of '{branch}' bound to reading {reading_id[:8]}".
   - `branch_adjudicate`: "Comparing branches: {', '.join(branches)}".
   - `experiment_submit`: "Queuing a {experiment_type} experiment on '{branch}'".
   - `experiment_collect`: "Collecting experiment {experiment_id[:8]} results".
   - `meta_declare_solution` / `meta_declare_unsolved`: "Declaring the
     solution on '{branch}'" / "Declaring the run unsolved (best: '{best_branch}')".
   - `workspace_create_hypothesis_branch`: "Opening hypothesis branch '{new_name}' ({cipher_mode})".
   - `decode_show`: "Reading the decode of '{branch}'".
   - `repair_agenda_list`/`repair_agenda_update`: "Reviewing the repair agenda" / "Updating repair-agenda item {id}".
   - `act_set_model_variant`: "Switching the {language} language model to '{variant}'".
2. `summarize_tool_call` additions (keep all existing outputs working):
   - When the result dict has `status: "blocked"`, produce
     `"blocked: {reason_in_words}"` where reason_in_words maps the known
     reason codes to sentences (table in code, fallback = the raw reason):
     `attestation_not_positive`, `attestation_required`, `attestation_stale`,
     `repair_transaction_not_ready`, `episode_kind_not_available`,
     `lead_tool_not_available`, `repair_saturated`, `pair_evidence_failed`.
     Include the result's `how`/`note` first sentence when present.
   - When the result dict has `status: "duplicate_suppressed"`:
     `"duplicate — already done against unchanged content"`.

## Part 3 — emit-site changes (`src/investigation/loop_v3.py`, minimal)

1. `agent_text`: raise the emit cap from `text_parts[0][:400]` to the FULL
   first text block capped at 4000 chars, and include ALL text blocks
   (`"\n\n".join(text_parts)[:4000]`). Rationale: the narration is now the
   primary display line; 400 chars visibly truncates multi-sentence turns.
   (Artifact loop_events size impact is bounded and acceptable; full text is
   already stored in artifact.messages.)
2. New `workflow_state_changed` event: in the turn loop, after
   `workflow_state(state, executor)` is (already) computed for the context
   build, compare the phase string against the previous turn's; when changed
   emit `{"from": prev, "to": new, "branch": menu.get("branch")}`. Keep a
   loop-local `prev_workflow_phase: str | None`. Emit BEFORE the send so the
   transition appears where it happened. (`repair_exhausted` transitions thus
   become visible live — a Slice-2 state the display currently never shows.)
3. `episode_complete` payload: add `"digest": <str>` — a one-line plain-
   English outcome built lead-side from the episode's structured result by a
   new pure helper `_episode_result_digest(kind, status, failure_reason,
   result) -> str` (module-level, unit-testable):
   - non-ok: `"{kind} failed: {failure_reason}"`.
   - verify: `VERDICT — "gloss[:90]" (lang X.XX, recoverability X.XX, damage
     {scope} → {repairability})` where VERDICT is POSITIVE/NEGATIVE from
     `reader_accepts_as_solution` (legacy fallback predicate for old dicts).
   - search: `"improved → {best_branch}" | "no improvement"` + `notes[:80]`.
   - reading: `proposed "{reading_text[:60]}" (confidence {overall_confidence})`.
   - compare: `winner {winner}` + first ranking entries.
   - repair: `applied {len(edits)} edit(s) → {best_branch}` | `did not apply`.
   - survey: first `findings` entry truncated 90, else "survey complete".
4. `repair_transaction_complete` already carries the record; no change — the
   renderer formats it (Part 4).

## Part 4 — `src/agent/narrate.py` rework

Rewrite the handler layer to the Part-1 contract. Key handler changes:

- `_on_agent_text`: full text both modes (wrap at ~100 cols; keep `“ `).
- `_on_tool_start` / `_on_tool_call`: default prints `⏺ action` (+ result
  `↳` on tool_call); verbose ALSO prints the numbered `tool(args)` line under
  the action. Parent-tool nesting rule (episode children between start/call)
  is preserved: the action line prints at tool_start for `_PARENT_TOOLS`, at
  tool_call otherwise (same structure as today, so the F-1 ordering pin
  stays).
- `_on_episode_tool_call` / `_on_episode_turn_start` / `_on_episode_submit`:
  verbose-only now.
- `_on_episode_complete`: default prints
  `  ↳ {kind}: {digest}  (calls=N, $X.XX)` using the Part-3 digest (fallback
  to the current `status` line when digest absent — v2/old artifacts).
- New problem handlers (always shown; yellow unless noted):
  - `lead_tool_rejected` → `! tool '{tool}' not available here — delegate via an episode/experiment`
  - `no_tool_calls_nudge` → `! agent narrated without acting — nudged to use a tool`
  - `lead_truncation_retry` → `! output budget hit mid-reasoning — retrying with {new_max_tokens} tokens`
  - `cost_ceiling_reached` → `! cost ceiling ${max:.2f} reached — ending run honestly` (red)
  - `no_new_information` → dim `· no new information this turn`
  - `repeated_call` / `duplicate_read_suppressed` → verbose-only, dim.
  - `max_iterations_reached` → `! turn limit reached without a declaration`
  - `best_effort_selected` → `✗ best-effort fallback: {tier} on {branch}` (yellow)
  - `repair_transaction_complete` → default `  ↳ repair {status}: {reason or
    'installed ' + installed_branch}` (yellow when failed; include
    `failure_class` when present).
  - `workflow_state_changed` → `  ⤷ workflow: {from} → {to} (on {branch})` (dim).
- `_emit_gloss`: delete (the action line subsumes it); keep
  `describe_tool_gloss` in display.py (other consumers) but narrate no longer
  calls it.
- `finish()`: after the existing result block, print the **run digest**
  (default AND verbose; ≤ 14 lines, plain text):
  ```
  ── digest ──
  outcome    : unsolved — no positive attestation (honest terminal)
  branch     : automated_preflight   (workflow=…, installed=…, selected=…) [only when divergent]
  attestation: NEGATIVE — "gloss[:70]" (lang 0.70, recov 0.25, distributed → broaden)
  episodes   : 2 (verify 1 ok, compare 1 ok)   experiments: 5 (4 completed, 1 orphaned)
  repairs    : 0 installed, 0 rejected
  problems   : 1 rate-limit retry; 1 truncation retry     [omit line when none]
  artifact   : <path>   (inspect with: python scripts/inspect_artifact.py <path>)
  ```
  Data source: a `finish`-side accumulator inside the renderer — count the
  events it has SEEN (problems, episodes by kind/status from
  episode_complete, repairs from repair_transaction_complete, last
  attestation from declared/verify payloads is NOT available → instead track
  the latest `episode_complete` verify digest string and reuse it). Branch
  roles come from the final `workspace_snapshot` payload (track latest).
  The digest must be computed purely from events already streamed — the
  renderer must NOT read the artifact file.

## Part 5 — tests

Baseline: 1689 passed / 2 skipped at `d7e1376`.

1. `tests/test_cli_observability.py` pins the CURRENT default shape
   (`test_narrate_renders_full_transcript_shape`,
   `test_narrate_nested_episode_block_orders_parent_above_children`,
   `test_narrate_verbose_shows_full_args_nonverbose_truncates`,
   `test_narrate_renders_decode_on_change_only`, and the cmd_* stdout tests).
   Update them to the NEW contract deliberately:
   - default-shape test asserts: `⏺ ` action lines present, `tool(` absent,
     full narration text present (no `…` truncation), episode-internal tool
     names absent, `↳` outcome lines present.
   - verbose test asserts BOTH the action line and the numbered
     `tool(args)` line, and episode internals present.
   - nesting pin: action line for `episode_run` still precedes its
     children's lines (F-1 preserved; children only visible at verbose — the
     ordering assert moves to a verbose-mode fixture).
2. New unit tests:
   - `describe_tool_action` patterns (each v3 lead tool; fallback path).
   - blocked-reason phrasing in `summarize_tool_call` (each reason code).
   - `_episode_result_digest` per kind (incl. non-ok + legacy verify dict).
   - `workflow_state_changed` emitted exactly on transitions (scripted run
     that moves searching → verified or similar; assert no event on
     unchanged turns).
   - agent_text full-text emit (≤4000) replacing the 400 cap.
   - finish digest renders the problems/episodes/attestation lines from a
     synthetic event stream (drive the renderer directly with `event()`
     calls; no run needed).
3. `tests/test_agent_display.py` narrate-roles test: unchanged expectation
   (roles line is verbose-only — still true).
4. Sequence-B (`test_v3_sequence_b.py`) and loop tests do not assert narrate
   output; they must stay green (event payload ADDITIONS only — no renames).

## Non-goals

- No changes to `pretty`/`raw`/`jsonl` displays or their CLI flags.
- No new CLI flags: the existing `-v`/`renderer_verbose` is the verbose
  switch (`--display` values unchanged).
- No v2 loop-code changes; v2 events render with the same handlers (missing
  new payload keys must degrade to the current lines).
- No artifact-schema changes (event payload additions only).
- The LLM analyzer script and `inspect_artifact.py` are unchanged (the CLI
  digest complements, not replaces, deep analysis).
