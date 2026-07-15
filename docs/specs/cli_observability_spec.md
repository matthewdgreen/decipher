# Spec: CLI Observability Revamp (improvement plan item 8)

Spec author: Fable. User request (2026-07-14): "the current output
visualization is very hard to parse… improve the clarity and verbosity
of the CLI output" (Claude-Code-style narrated turns, nested episode
blocks, cost ticker, clear declarations) and "the current version
doesn't output the artifact filename" (artifact path must always be
printed; the CLI should be self-sufficient for humans debugging runs).
Factual basis: the output-surface map (agent recon 2026-07-14) — all
file:line refs below verified against main @ eaef2d8.

## Design decisions (binding)

1. **New default renderer: `narrate`** — a SCROLLING structured
   transcript (Claude-Code style), not a live dashboard. Plain text with
   optional ANSI color when `sys.stdout.isatty()`; no Rich dependency;
   works in scrollback, pipes, and CI logs. The existing `pretty` (Rich
   Live), `raw`, `jsonl` renderers remain selectable; `--display`
   default changes `pretty` → `narrate` (cli.py:1513-1521, :1666).
   `--verbose` no longer forces `off` (cli.py:18-24): it means MORE
   DETAIL WITHIN the active renderer (full tool args, full agent text,
   per-event payloads in narrate; the old raw firehose stays available
   as `--display raw --verbose`).
2. **v2/v3 event parity** — v3 gains the missing emissions so ANY
   renderer works identically for both loops:
   - `workspace_snapshot` after each lead turn (decryption preview of
     the current best branch + total tokens + estimated_cost_usd —
     mirror v2 loop_v2.py:1457-1472; v3 has the data at sync_budget,
     loop_v3.py:257-266/:675, it just never emits).
   - `budget_update` event after every sync_budget: payload =
     `state.budget_by_category()` compact dict + total cost. (Additive
     event kind; JSONL renderer passes it through — additive is
     backward-compatible for tooling.)
3. **Episode narration (the silence killer)** — `run_episode`
   (src/investigation/episodes.py:750-762; currently ZERO prints,
   `verbose` param unused) gains an optional `on_event` callback
   (default None), emitting: `episode_turn_start`, `episode_tool_call`
   (name + compact arg summary), `episode_submit`, each carrying
   `{episode_id, kind}`. The v3 dispatcher forwards these into the lead
   `emit` stream so renderers see them (nested context preserved).
   Remove the dead `verbose` param or wire it to the callback — pick
   one, state it.
4. **Artifact path: always, everywhere** — the single user complaint
   with a one-line root cause (cli.py:734 gated behind
   `not quiet_structured_display`). Fix: the per-test artifact path and
   the final artifact path print UNCONDITIONALLY to stdout scrollback in
   every mode (benchmark agentic pretty/jsonl included), and
   `cmd_benchmark` ends with a consolidated `Artifacts:` list (one line
   per run) + the run-dir root. `cmd_crack` already echoes (cli.py:826,
   :964) — keep.
5. **Final per-test result block always printed** — status, char/word
   accuracy, iterations, tokens, cost, artifact path — to scrollback in
   every display mode (today it's skipped when quiet_structured_display,
   cli.py:719-739; the pretty panel is transient). Wire
   `scorer.format_report` (src/benchmark/scorer.py:757-795 — currently
   dead code) as the end-of-benchmark per-test table replacing the
   inline AVERAGE-only line (cli.py:744-759).
6. **No prints inside an active Rich Live** — cmd_crack's plain prints
   at cli.py:940-966 move AFTER `renderer.finish` (cli.py:971) or route
   through the renderer. General rule: while a renderer is active, ALL
   terminal output goes through it.
7. **Automated-path progress** — AutomatedBenchmarkRunner.verbose is
   dead (runner.py:117, zero prints in module). Smallest honest fix: an
   optional `on_step(name, status, elapsed)` callback threaded from
   cmd_benchmark/cmd_crack, called at each pipeline step boundary the
   runner already tracks; narrate renders `  · step: transform_rank …
   done (3.2s)` lines. No behavior change when None.
8. **Newline discipline** — narrate mode never uses `end=""` dot
   streams (cli.py:906-910, runner_v2.py:184-191 remain for the
   legacy no-renderer fallback only).

## The narrate format (pin the shape, not the exact strings)

```
▶ borg_single_B_borg_0109v  (borg, la)  model=gpt-5.5  loop=v3
  · preflight: zenith_native … solved 91.2% dict  (12.4s, $0.00)
  1 │ observe_frequency(branch=main) → top: E7.1% …   [$0.14 | 41k tok]
  2 │ episode_run(kind=search, branch=main)
      ↳ search ep_1 │ turn 1 │ search_hill_climb(…)
      ↳ search ep_1 │ turn 2 │ episode_submit → ok (calls=3, $0.21)
  3 │ act_set_mapping(S12→E, …) → 14 mapped
  …
  ✓ DECLARED solution on branch main (confidence: high)
      attestation: coherence 8/10, reader accepts, 1 anomaly
  ── result ──
  status=solved  char=95.7%  word=73.1%  iters=9  $2.31  (4m12s)
  artifact: artifacts/…/borg_single_B_borg_0109v/6d7885ca379d.json
```

One line per lead tool call (`summarize_tool_call`, display.py:40-72,
reused); indented `↳` lines for episode internals; a bracketed cumulative
cost/token ticker on iteration lines (from `budget_update` /
workspace_snapshot payloads — no live repainting, just line suffixes);
declaration/attestation rendered as a distinct block (✓/✗ prefixes);
errors and retries as `!`-prefixed lines. Verbose adds full args and
full agent text; non-verbose truncates to one line each.

## Implementation notes

- New module `src/agent/narrate.py` (NarrateAgentRenderer implementing
  the AgentRenderer protocol in display.py:25-37); registered in
  `make_agent_renderer`. Reuse `summarize_tool_call`,
  `describe_tool_process`, `_compact_preview`.
- v3 emissions in loop_v3.py (snapshot after sync_budget; forward
  episode events); episodes.py callback plumbing. All additive to the
  event stream; LoopEvent schema unchanged (event/payload fields).
- JSONL renderer: new event kinds flow through automatically (it dumps
  every event) — note in the spec that consumers must tolerate unknown
  kinds (they already must).
- Tests: renderer unit tests over a scripted event stream (golden-ish:
  assert structural markers, not exact spacing); v3 snapshot/budget
  emission tests (fake provider, assert events present with correct
  payload fields); episode on_event forwarding test; artifact-path
  always-printed tests for benchmark (agentic pretty + jsonl + narrate)
  and crack; format_report wiring test; cmd_crack no-print-during-Live
  regression (assert the moved prints happen after finish); automated
  on_step callback test. Suite zero-regression.
- **v2 and v3 LLM-facing surfaces untouched** (build_workspace_panel
  etc. are prompt text, not terminal output — do not modify).
- **Do not touch scripts/run_v3_bakeoff.py** (its own reporting; a live
  matrix depends on it).

## Out of scope

Rich dashboard redesign (pretty stays as-is, selectable); logging-module
migration; inspect_artifact.py changes (post-hoc tool stays);
web/HTML output.

## Acceptance (local, no live LLM needed)

1. Suite green, zero regressions (baseline ~1359 + new).
2. Scripted fake-provider v2 AND v3 runs through the narrate renderer:
   capture stdout, show the transcript renders turns, a nested episode
   block (v3), the cost ticker advancing, a declaration block with
   attestation, and the artifact path — paste the two transcripts in
   the report.
3. `decipher benchmark --dry`-equivalent not required; a real automated
   (non-LLM) benchmark run on one page showing the always-printed
   result block + artifact path + format_report table.

## Deliverables

Files changed, suite counts, the two captured transcripts, the
automated-run output sample, deviations. No commits.

## Post-review amendments (BINDING — Fable review: READY WITH AMENDMENTS)

**F1 (jsonl purity).** JsonlAgentRenderer's stdout is machine-readable
(its documented purpose) and its `test_finish` event already carries
artifact_path/status/accuracies. Decisions 4/5's unconditional human
prints are EXEMPT in jsonl mode (or routed to stderr — pick stderr).
The pretty-mode interleaving worry does NOT apply in cmd_benchmark (the
renderer lifecycle is fully inside runner_v2.run_test, finish at
:352-353); cmd_crack is the only real interleaving site (decision 6).

**F2 (format_report shape).** `format_report(scores: list[ScoreResult])`
needs total_words/agent_score which RunResultV2 lacks, and it counts
status=="solved" while automated runs use "completed". Decision:
option (b) — change `format_report` to take a lightweight row type
(dataclass or dict: test_id, status, char_accuracy, word_accuracy,
duration, cost) built from RunResultV2; DROP the Agent% column; count
solved = status in {"solved","completed"}. Update its existing tests.

**F3 (protocol additions — named).** The protocol is `AgentRunRenderer`
(display.py:14-22; :25-37 is the factory). Additive changes: (a)
`start_test` gains keyword-only optional `language`, `source`,
`agent_loop` (default None — old callers fine); (b)
`make_agent_renderer(mode, verbose=False)` gains the verbose flag,
consumed by narrate (others ignore it). Update the 3 call sites:
runner_v2.py:98-105, cli.py:843-850, cli.py:1025-1032.

**F4 (verbose plumbing rule).** Loop/runner `verbose` remains
display-off-only — KEEP the `args.verbose and display_mode == "off"`
conjunctions at all five sites (cli.py:697, 923, 935, 1066, 1278) and
the raw emit-prints (loop_v2.py:1000-1001, loop_v3.py:248-249,
runner_v2's self.verbose blocks) unchanged. ONLY the renderer gets the
new verbose flag. (Naively passing args.verbose through would fire
runner_v2's branch-score dump inside an active Live — the decision-6
bug reintroduced.)

**F5 (v3 emission point).** Emit `workspace_snapshot` + `budget_update`
at END-OF-TURN after the tool-dispatch loop (near state.record_exchange,
loop_v3.py:777-783), preceded by a fresh sync_budget() — the :675 site
is pre-dispatch and would show the previous turn's workspace and miss
episode spend. MOVE `_workspace_snapshot_payload` from loop_v2.py:577
to loop_shared.py; both loops import it from there. Best-branch notion:
`_best_branch_for_auto_declare` + `_decoded_text_for_panel` (already
imported in loop_v3:22-28); freq_rank via `executor._freq_rank`
(precedent :833). Per-turn cost is ms-scale (v2 already pays it).

**F6 (episode plumbing pins).** (a) REMOVE the dead `verbose` param from
run_episode (keyword-only, no caller passes it). (b) Add `turn` to
`episode_turn_start`/`episode_tool_call` payloads (the pinned format
renders it). (c) Per-episode cost renders from the existing
`episode_complete` payload (spend is not available mid-episode).

**F7 (--display completeness).** FOUR `--display` definitions:
benchmark :1514, crack :1667, resume :1781, testgen :1857 — add
`narrate` to all four choice lists and flip all four defaults
consistently. `auto` resolves to narrate (pipe-safe). Update the
`getattr(args, "display", "pretty")` fallback in _resolve_agent_display
to "narrate".

**F8 (automated-path rendering ownership).** No renderer exists on the
automated path (display forced "off" for non-agentic, cli.py:634) —
`on_step` is a cli.py-defined callback that PRINTS DIRECTLY (safe: no
Live in automated mode). Step boundaries = the ~14 `steps.append` sites
in `_run_automated_impl` (each has a "name" key); plumbing chain:
AutomatedBenchmarkRunner.run_test → run_automated → _run_automated_impl
(optional kwarg, default None). Per-step elapsed: compute in the
callback wrapper (not all sites track it).

**F9 (landing order).** Implement in a worktree; MERGE ONLY AFTER the
running M6 matrix completes (new loop_events kinds + per-turn snapshot
strings would blur artifact-shape uniformity mid-matrix).

**F10 (wording fix).** The pretty panel is NOT transient
(Live(transient=False), display.py:231, persists the final frame incl.
artifact path at :424). Decision 5's motivation is jsonl coverage,
interrupted runs, and resized-terminal garbling — not panel transience.

**F11 (narrate clarifications).** The cost ticker on an iteration line
is the LAST-KNOWN cumulative value (a scrolling renderer cannot
retro-annotate; v2's snapshot arrives after the iteration's tools).
Agent commentary renders as one truncated line at non-verbose, full
text at verbose.
