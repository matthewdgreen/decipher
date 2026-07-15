# Spec: CLI-2 — Human-Friendly Narration & Transcripts

Spec author: Fable (2026-07-15). Follow-up to the CLI observability slice
(947f8c6, narrate renderer) driven by user feedback: "commands are opaque
and confusing sometimes, and I don't see any reasoning or comments about
why the agent is making its decisions, the way Claude Code gives
feedback" — plus "convert the transcript into something human-friendly"
for stored runs. Baseline: main @ a5b0be9, suite 1401 passed / 1 skipped.

Three parts. Parts 1 and 3 change NO agent behavior (rendering only).
Part 2 is a v3-brief change (visible reasoning), deliberately scoped.

## Part 1 — Synthesized tool glosses in narrate (rendering only)

The narrate renderer explains each tool in plain English the FIRST time
it appears in a run: a dim, indented line under the numbered tool line,
e.g.

```
  1 │ observe_frequency(branch=main) → top: E7.1% …
      · counts how often each symbol appears; frequent symbols usually map to E/T/A
  2 │ episode_run(kind=verify, branches=[main])
      · sends the candidate text to a fresh reader who judges if it reads as real language
```

- Source: `describe_tool_process` (src/agent/display.py:75-104). AUDIT
  its coverage against the tools that actually appear in v2/v3 runs
  (observe_*, search_*, act_*, decode_*, score_*, hypothesis_*,
  episode_run per-kind, experiment_submit/collect, meta_*,
  workspace_*) and fill gaps with one-line plain-English glosses a
  non-cryptographer understands. Keep the existing generic fallback for
  anything unmapped; episode_run's gloss should vary by `kind` argument.
- First-use-per-run only (both verbose and non-verbose; a repeat of the
  same tool renders compactly as today). Track seen tool names on the
  renderer instance.
- narrate only — raw/jsonl/pretty untouched.

## Part 2 — Agent self-narration (v3 brief, ~1 line)

Add ONE line to the v3 lead brief (src/investigation/context.py, the
system prompt): instruct the lead to begin each turn with one short
sentence stating what it is doing and why (it is shown to a human
observer watching the run). The narrate renderer already displays
`agent_text` (the `“ …` lines) — this makes gpt-5.5-class models, whose
visible text is otherwise sparse, actually produce the commentary.

- v3 ONLY. v2's brief is frozen (comparability + v2-untouched
  discipline). No CLI flag — the line is part of the v3 brief from now
  on; the post-M5.1 M6 rerun measures v3 as it now is (its spec already
  forbids mixing pre-fix rows). Token cost ≈ 10-25 output tokens/turn —
  negligible against tool traffic.
- Test: the brief contains the line for v3 (and v2's system prompt is
  byte-unchanged — extend the existing v2-parity pin if one exists).

## Part 3 — `inspect_artifact --narrative` (post-hoc transcript replay)

Convert ANY stored artifact into the same human-friendly transcript the
live narrate renderer produces, by REPLAY: construct a
`NarrateAgentRenderer` and feed it the artifact's stored events — do not
build a second formatter.

- New flag on scripts/inspect_artifact.py: `--narrative` (with
  `--verbose` for full agent text/args). Mechanics:
  1. `start_test` from artifact fields (test_id/description-ish header,
     language, model, agent_loop/loop_version).
  2. Replay `artifact.loop_events` in order: each LoopEvent's
     (event, payload) goes straight to `renderer.event()`. The stored
     stream already includes agent_text, tool_start/tool_call, the
     episode_* kinds, workspace_snapshot (decode progress),
     budget_update, hints, declarations.
  3. Synthesize the `finish` result from artifact fields (status,
     accuracies when present, iterations, cost, elapsed, artifact path =
     the input path, final_decryption from the declared branch's stored
     decryption if available; `has_ground_truth` False when accuracies
     are absent/zero-with-no-benchmark).
- Graceful degradation, explicitly: old artifacts missing newer event
  kinds (pre-947f8c6 v3 runs have no workspace_snapshot/episode_* events;
  v2 artifacts have no episode events at all) simply render fewer lines —
  no errors, no fabricated content. If `loop_events` is empty, print the
  header + finish block and say the artifact predates event capture.
- Part-1 glosses apply automatically (same renderer).
- LLM-assisted narrative annotation is OUT OF SCOPE (recorded future
  option; this slice stays LLM-free).

## Tests

- Gloss: first use renders the gloss line, second use doesn't; unmapped
  tool falls back generically; episode_run gloss varies by kind.
- Coverage audit pinned: every tool name appearing in
  tests/fixtures artifacts + the common namespaces above resolves to a
  non-generic gloss (table-driven test naming any gaps).
- Part 2: v3 system prompt contains the narration line; v2 prompt
  unchanged.
- Replay: feed the real stored fixture artifact
  (tests/fixtures/v2_artifact_synth_en_40wb_s1.json) through
  `--narrative` → structural markers assert (header, numbered tool
  lines, result block, artifact path); an artifact with empty
  loop_events degrades to header+result; a v3-shaped synthetic event
  list (agent_text + episode_* + workspace_snapshot + declaration)
  replays with nesting and decode lines intact.
- Suite zero-regression (baseline 1401 / 1 skipped).

## Acceptance (local, no LLM)

1. Suite green.
2. `--narrative` replay of a stored M6 bake-off v3 Borg artifact
   (artifacts/m6_bakeoff/v3/borg_single_B_borg_0109v/1/**) — paste an
   excerpt showing tool lines + glosses + (if present in that artifact
   vintage) decode/attestation lines.
3. One real automated (non-LLM) benchmark run with narrate showing
   first-use glosses live.

## Out of scope

v2 brief changes; any LLM calls; pretty/raw/jsonl changes; TOOLS.md
per-tool doc rewrites (glosses live in code); the M6 rerun itself.

## Deliverables

Files changed, suite counts, the replay excerpt + live gloss sample,
gloss-coverage list (which tools got new glosses), deviations. No
commits.
