# Spec: Improvement Program Phase 0 — Evaluation Integrity + Hygiene

Parent plan: `docs/improvement_program_plan.md` (Phase 0).
Spec author: Fable (main session). Implementer: coding sub-agent.

## Scope and constraints

- Implement items 0.1–0.7 below. Item 0.8 (CLAUDE.md content refresh) is out
  of scope except where noted (the `state.py` line in Key Files).
- The working tree contains unrelated uncommitted work in progress (e.g. in
  `src/agent/tools_v2.py`, `src/benchmark/scorer.py`, `TODO.md`, docs).
  Make targeted edits only. Do not revert, reformat, or "clean up" anything
  outside the edits specified here. Do not run `git checkout`/`restore`/
  `stash` on any file. Do not commit.
- All line numbers below were verified against the current working tree on
  2026-07-13. If a line has drifted, locate the construct by the quoted code,
  not the number.
- Run the test suite with:
  `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
  Record any pre-existing failures before you start; you are responsible only
  for not adding new ones.
- Behavior outside the specified changes must be preserved exactly.

---

## 0.1 Fallback declarations must not report status "solved"

### Current behavior

`run_v2` in `src/agent/loop_v2.py` auto-declares when the run ends
`exhausted` or `error` without an agent declaration (block at
`loop_v2.py:1632–1666`): it picks the best branch
(`_best_branch_for_auto_declare`, `:1805`), synthesizes a
`SolutionDeclaration` with `self_confidence=0.0`, emits
`auto_declared_solution`, and then sets `artifact.status = "solved"`
(`:1662`). Downstream, a give-up is indistinguishable from a real solve by
status.

Known status values (comment at `src/artifact/schema.py:131`):
`running | solved | unsolved | exhausted | error | stopped`.

### Required behavior

1. In the auto-declare block, set `artifact.status = "fallback_declared"`
   instead of `"solved"`. Keep the `auto_declared_solution` event, the
   synthesized `SolutionDeclaration`, `self_confidence=0.0`, and (for the
   error path) `artifact.error_message` exactly as they are.
2. Add a field to `RunArtifact` in `src/artifact/schema.py`:
   `auto_declared: bool = False`. Set it to `True` in the auto-declare
   block. Update the status comment at `schema.py:131` to include
   `fallback_declared`.
3. Scoring must be unchanged: `src/benchmark/runner_v2.py` grades
   `artifact.solution` (see `runner_v2.py:229`), which is still populated.
   Verify no code path in `runner_v2.py` gates scoring on
   `status == "solved"`.
4. Success counting: `src/cli.py:648–650` computes
   `success_status = "completed" if not agentic else "solved"` and counts
   `r.status == success_status`. Keep that as the definition of *success*
   (fallback runs are not successes). In the same summary output, add a
   separate count of fallback runs (`status == "fallback_declared"`) so they
   are visible rather than silently lumped into failures, e.g.
   `Solved: N/M  (fallback declarations: K)`.
5. Sweep for other consumers: `grep -rn '"solved"' src/ scripts/ tests/` and
   inspect each hit. For every place that branches on or counts
   `status == "solved"`, decide whether `fallback_declared` needs explicit
   handling, and record the decision as a one-line note in your final report.
   Known consumers to check: `src/cli.py` (several status prints — plain
   passthrough is fine), `src/benchmark/runner_v2.py:305` (passthrough,
   fine), `scripts/inspect_artifact.py`, `scripts/run_testgen_suite.py`,
   `scripts/run_automated_parity_matrix.py` (if they aggregate agentic
   statuses).
6. `scripts/inspect_artifact.py`: the summary currently derives
   `declared` from solution presence (`:280`, `:531`). Keep that, but make
   the human summary line (`:286`) and the LLM analysis packet (`:524–532`)
   distinguish fallbacks: when `artifact.get("status") == "fallback_declared"`
   or `artifact.get("auto_declared")` is true, render
   `declared=True (fallback)` and include `"auto_declared": true` in the
   packet.
7. Backward compatibility: old artifacts on disk have `status == "solved"`
   with `self_confidence == 0.0` for fallbacks. Do not add heuristics for
   them; new artifacts carry the new status.

### Tests

- `tests/test_agent_reliability.py` already exercises the fallback path with
  a fake provider. Update any assertion expecting `status == "solved"` after
  a fallback to expect `"fallback_declared"`, and assert
  `artifact.auto_declared is True` and `artifact.solution is not None`.
- Add/keep a test for a real declaration (fake provider calls
  `meta_declare_solution`): `status == "solved"` and
  `artifact.auto_declared is False`.
- Add a test for the error path (fake provider raises `ModelProviderError`
  mid-run): status `"fallback_declared"`, `error_message` preserved.

---

## 0.2 Delete dead `src/agent/state.py`

`AgentState` and `Checkpoint` in `src/agent/state.py` are v1 leftovers.
Verified: zero references anywhere in `src/`, `tests/`, or `scripts/`
(grep for `agent.state`, `AgentState`, `Checkpoint`).

- Delete the file.
- Remove the `state.py` line from the Key Files listing in `CLAUDE.md`
  (`    state.py              — AgentState, Checkpoint ...`).
- Re-run the grep after deletion to confirm nothing breaks; run the suite.

---

## 0.3 Close the tool-dispatch footgun

### Current behavior

`WorkspaceToolExecutor.execute` (`src/agent/tools_v2.py:2124–2190`)
dispatches via `getattr(self, f"_tool_{tool_name}")` (`:2130`). The internal
helper `_tool_tried_for_branch` (`:3024`) matches that naming pattern, so a
model calling a tool named `tried_for_branch` would invoke it. Today only
the per-turn allowlist prevents this.

### Required behavior

1. Rename `_tool_tried_for_branch` to `_was_tool_tried` and update all call
   sites (`grep -n '_tool_tried_for_branch' src/agent/tools_v2.py`).
2. In `execute()`, validate the name against the real tool registry before
   `getattr`: build a module-level
   `VALID_TOOL_NAMES = frozenset(d["name"] for d in TOOL_DEFINITIONS)` and
   return the existing `{"error": f"Unknown tool: {tool_name}"}` result for
   any name not in it. Keep the `handler is None` branch as a backstop.
   Preserve ordering so gating (`tool_gated`) still takes precedence over
   unknown-tool errors, and the call is still recorded in `call_log`.
3. Add a consistency test (new or in an existing suitable test module):
   - every name in `TOOL_DEFINITIONS` has a `_tool_<name>` method on
     `WorkspaceToolExecutor`;
   - every attribute of `WorkspaceToolExecutor` whose name starts with
     `_tool_` corresponds to a name in `TOOL_DEFINITIONS` (this fails if
     someone reintroduces a `_tool_`-prefixed helper);
   - `execute("tried_for_branch", {})` returns an unknown-tool error.

---

## 0.4 Deep-copy nested mutables on branch fork

### Current behavior

`Branch.copy_as` (`src/workspace/branch.py:21–32`) copies one level:
`metadata=dict(self.metadata)` and
`transform_pipeline=dict(self.transform_pipeline)`. Nested mutables (lists
of steps inside `transform_pipeline`, dicts/lists stored in `metadata` such
as decoded-text blocks or finalist evidence) alias between parent and fork.

### Required behavior

1. In `copy_as`, use `copy.deepcopy` for `metadata` and
   `transform_pipeline`. Leave `key`, `word_spans`, `token_order`, `tags`
   as they are (flat containers of immutables).
2. Do not change `snapshot_dict`.

### Tests

Extend `tests/test_workspace.py`: fork a branch whose `metadata` contains a
nested dict and whose `transform_pipeline` contains a list of step dicts;
mutate the nested structures on the fork; assert the parent is unchanged
(and vice versa).

---

## 0.5 Allowlist drift test

Six hardcoded tool-name sets live in `src/agent/loop_v2.py:158–245`:
`FULL_READING_WORKFLOW_TOOL_NAMES`, `FULL_READING_ACTUATOR_TOOL_NAMES`,
`PENULTIMATE_ALLOWED_TOOL_NAMES`, `FINAL_ALLOWED_TOOL_NAMES`,
`REPAIR_SANDBOX_TOOL_NAMES`, `INSPECTION_SANDBOX_TOOL_NAMES`.

Add a test asserting each set is a subset of
`{d["name"] for d in TOOL_DEFINITIONS}` (import both modules; on failure,
print the offending names). This catches renamed/removed tools leaving
stale allowlist entries.

---

## 0.6 Deduplicate the penultimate reading-workflow preflight

### Current behavior (verified)

`PENULTIMATE_READING_WORKFLOW_PREFLIGHT` can reach the model twice in
adjacent positions:

- appended to the workspace panel when `iters_left == 1`
  (`loop_v2.py:972–974`, inside `build_workspace_panel`), i.e. at the end of
  the turn *before* the penultimate turn;
- appended as a gate user message at the start of the penultimate turn when
  `_is_reading_workflow_gate_turn(...)` is true and
  `iteration == max_iterations - 1` (`loop_v2.py:1227–1239`).

The gate message only fires when the full-reading workflow has not been
used; the panel copy is unconditional on that.

### Required behavior

The model must see exactly one copy per turn. Change the panel-side
injection (`:972–974`) so it fires only when the gate turn will NOT inject
(i.e. when the full-reading workflow has already been used — mirror the
condition inside `_is_reading_workflow_gate_turn`, `loop_v2.py:527`). The
gate user message is the primary carrier; the panel copy is the reminder
for the already-compliant case. Keep `FINAL_ITERATION_PREFLIGHT` behavior
unchanged.

### Tests

Fake-provider run with a small `max_iterations` (e.g. 4) where the workflow
is never used: capture the exact message list sent to the provider on the
penultimate turn (the fake provider records inputs) and assert
`PENULTIMATE_READING_WORKFLOW_PREFLIGHT` text appears exactly once across
the visible (non-stubbed) messages. A second case where the workflow HAS
been used (fake provider calls one of the resegmentation actuators
earlier): assert the panel copy appears and the gate message does not.

---

## 0.7 Ground-truth firewall regression tests

Goal: an automated guard that benchmark ground truth cannot influence a run
— it may be read only for post-hoc grading. First version = leak detection
on everything that flows toward the model/solver.

Add `tests/test_ground_truth_firewall.py`:

1. **Agent path.** Using the existing fake-provider infrastructure from
   `tests/test_agent_reliability.py`, run a small benchmark-style test
   end-to-end through `src/benchmark/runner_v2.py` (or `run_v2` directly
   with a `benchmark_context` if runner wiring is impractical in-test) on a
   fixture whose ground-truth plaintext is a distinctive string (e.g.
   `WOMBATFESTIVALQUARTZ...`, long enough that no tool would produce it by
   chance). Capture:
   - every message the provider receives (system prompt + all user/assistant
     turns),
   - every tool result string recorded in `executor.call_log` /
     `artifact.tool_calls`,
   - the rendered `benchmark_context` / opening context.
   Assert the normalized ground truth (uppercase, spaces stripped) appears
   in none of them — check both the full string and its first 30
   characters. It MAY appear in the artifact's post-hoc scoring fields only.
2. **Automated path.** Call `automated.runner.run_automated` on a small
   synthetic substitution case and assert by construction that no argument
   passed to it contains the plaintext (build the call the way
   `runner_v2`/`cli` build it; the test documents that the runner API has no
   ground-truth-bearing parameter).
3. Structure the leak check as a reusable helper
   (`assert_no_ground_truth_leak(haystacks: Iterable[str], plaintext: str)`)
   so later phases (finalist menus, LLM reader inputs) can reuse it.

Notes: normalization matters — compare case-insensitively and
whitespace-insensitively; also check a no-boundary variant of the ground
truth. Keep runtime under ~30s (small cipher, low max_iterations, no real
solver budget).

---

## Review follow-ups (deferred to a later phase spec)

Recorded from the Fable review of the Phase 0 implementation (2026-07-13).
Fixed in-phase: stale `observe_patterns` entries in `_mode_tool_menu`
(`tools_v2.py`) and a non-vacuous-run assertion in the automated firewall
test. Deferred:

- 0.6 residual duplicate: if the penultimate gate message fires and the model
  complies during that turn, the end-of-turn panel re-emits the preflight
  (condition is computed at panel-build time). Fix by passing a loop-side
  "gate message already injected" flag into `build_workspace_panel` instead
  of recomputing workflow-used.
- 0.6 edge: for `max_iterations <= READING_WORKFLOW_GATE_TURNS` (≤2) the gate
  can never fire and the panel copy is now also suppressed — zero copies
  where old behavior correctly showed one. Mirror the full gate condition.
- Delete the dead `ground_truth` parameters threaded through
  `automated/runner.py` internals (`_refine_transform_finalist_bakeoff`,
  `_run_homophonic`, `_run_homophonic_zenith_native`) — currently inert
  plumbing below the firewall.
- Add a fixture-driven `runner_v2.run_test` leak test covering
  `build_benchmark_context` (`benchmark/context.py` plaintext-bearing
  layers), the real leak surface the agent-path test cannot reach.
- Document that `assert_no_ground_truth_leak` false-positives on genuinely
  solved runs before reusing it on finalist menus.
- Cosmetic: `fallback_declared` overflows fixed-width status columns
  (`benchmark/scorer.py` `{s.status:<8}`, `scripts/run_testgen_suite.py`
  `{sr.status:<10}`).
- Extend the 0.5 drift test to cover tool-name strings in `_mode_tool_menu`
  and `_mode_playbook` data, not just the six loop allowlists.

## Deliverables / final report

- All edits above, full test suite run, with:
  - list of pre-existing failures (if any) that you did not introduce;
  - the 0.1 consumer-sweep decisions (one line each);
  - any spec ambiguity you hit and how you resolved it (or flag it as open
    rather than inventing behavior).
- Do not commit. Leave changes in the working tree.
