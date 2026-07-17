# M5.3 Slice 7 — Implementation Sub-Spec (Observability and Analyzer Parity)

Status: ready to implement. Authored 2026-07-17 against HEAD `a0ba63c`
(Slices 1/2/3/5/6 + B1/B2/B3 all landed; Slice 7 is the last M5.3 slice).

Authority: `docs/specs/agent_v3_m5_3_control_reliability_spec.md` —
Slice 7 (lines 502–527), Verification Sequence A (531–538) and B (540–552),
Deliverables 7 & 8 (611–620). The paid Sequence-C smoke, Stage-1 packet, and
M6 bake-off are user-gated and OUT OF SCOPE for this slice.

Baseline: full suite **1644 passed / 2 skipped** at `a0ba63c`. Run
`PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`. No paid model call
anywhere in this slice.

---

## 0. Blast radius and design summary

Files touched (production):

| File | Change |
|---|---|
| `src/investigation/loop_v3.py` | new pure helper `_compute_branch_roles`; per-turn `branch_roles` in the `workspace_snapshot` payload; final `artifact.branch_roles` |
| `src/artifact/schema.py` | new additive `RunArtifact.branch_roles` field |
| `src/investigation/episodes.py` | episode ledger gains `budget`, `requested_max_tool_calls`, `registered_max_tool_calls` (minimal instrumentation for analyzer section 1) |
| `src/agent/narrate.py` | verbose snapshot line renders branch roles when present |
| `scripts/inspect_artifact.py` | header/summary parity fix + 8 new `format_*` sections |
| `scripts/trim_artifact_fixture.py` | NEW — reproducible fixture trimmer |

Files touched (tests):

| File | Change |
|---|---|
| `tests/support/__init__.py`, `tests/support/scripted_v3.py` | NEW — hoisted scripted-session/worker-fake helpers |
| `tests/test_loop_v3.py` | import hoisted helpers (delete local copies); add branch-role tests |
| `tests/test_v3_sequence_b.py` | NEW — the Sequence-B end-to-end replay test |
| `tests/test_inspect_artifact.py` | new-section unit tests + trimmed-fixture regression test |
| `tests/test_ground_truth_firewall.py` | fixture ground-truth-rejection test |
| `tests/test_agent_display.py` | narrate verbose roles-line test |
| `tests/fixtures/v3_artifact_m5_2_smoke_trimmed.json` | NEW — committed trimmed M5.2 analyzer fixture |

Everything else in the analyzer work is **read-side**. The only new
production instrumentation is (a) the three episode-ledger budget keys and
(b) the derived `branch_roles` dict — both additive, both branch-NAMES /
integers only (firewall-safe; no ground truth anywhere on these surfaces).

---

## 1. Part A — Four distinguished branch roles

### 1.1 Where each role is computed (all existing machinery)

All four are **derived** values. They are computed only in
`src/investigation/loop_v3.py` by ONE new pure helper; nothing else in the
codebase computes or caches them.

```python
def _compute_branch_roles(
    state: InvestigationState,
    executor: WorkspaceToolExecutor,
    declared_or_selected: str | None = None,
) -> dict[str, str | None]:
    """The four distinguished branch roles (M5.3 Slice 7, master 504-509).

    Derived, never stored in InvestigationState: a resume recomputes them
    from the restored state + a fresh executor. Branch NAMES only —
    firewall-safe by construction.

    - best_scored_branch: the internal scalar-best branch
      (_best_branch_for_auto_declare — the value M5.2 telemetry mislabeled
      as plain `branch`).
    - workflow_branch: the branch the workflow state machine is focused on
      (workflow_state(...)["branch"]; None when the menu names no branch).
    - latest_installed_branch: the newest `installed` repair transaction
      whose installed branch still exists in the workspace.
    - declared_or_selected_branch: the declared / honest-unsolved /
      fallback-selected branch. None until termination resolves it
      (mid-run snapshots always carry None).
    """
```

Body (exact logic):

1. `best_scored_branch`:
   `_best_branch_for_auto_declare(state.workspace, state.language, executor.word_set, executor._freq_rank)[0]`
   (`src/agent/loop_shared.py:197`). Note the loop already computes this
   inside `_workspace_snapshot_payload` (loop_shared.py:244); the duplicate
   per-turn computation matches the existing per-turn cost profile
   (`workflow_state` also calls it) — do not restructure.
2. `workflow_branch`: `workflow_state(state, executor).get("branch")`
   (`src/investigation/context.py:205`). May be `None`.
3. `latest_installed_branch`:
   ```python
   next(
       (
           str(item.get("installed_branch") or "")
           for item in reversed(state.repair_transactions)
           if item.get("status") == "installed"
           and state.workspace.has_branch(str(item.get("installed_branch") or ""))
       ),
       None,
   )
   ```
   This is deliberately WITHOUT the content-hash freshness check that
   `workflow_state`'s `latest_transaction` lookup applies (context.py:210-223):
   the role answers "which installed branch is newest", not "is it unchanged".
4. `declared_or_selected_branch`: the passed-through argument (see 1.3).

Place the helper in `loop_v3.py` directly after `_select_v3_fallback`
(after line 189). It must not import anything new (all names already
imported at module top).

### 1.2 Per-turn: the `workspace_snapshot` event payload

Anchor: `loop_v3.py:2117-2137` (the post-dispatch `emit("workspace_snapshot", ...)`).

Do **not** change `_workspace_snapshot_payload`'s signature (it is shared
with v2, `loop_v2.py:1419`). Instead, in `run_v3`, build the payload, then
merge the roles before emitting:

```python
snapshot_payload = _workspace_snapshot_payload(...existing args...)
snapshot_payload["branch_roles"] = _compute_branch_roles(state, executor)
emit("workspace_snapshot", snapshot_payload, outer_iteration=turn)
```

Mid-run, `declared_or_selected_branch` is always `None`. The existing
`branch` key stays (renderer compatibility); `branch_roles` is additive.
v2 snapshots simply never carry the key.

### 1.3 End-of-run: `RunArtifact.branch_roles`

`src/artifact/schema.py` — add to `RunArtifact` (with the other v3
additions, after `attestations`, line 174):

```python
# M5.3 Slice 7: the four distinguished branch roles at termination
# (best_scored_branch / workflow_branch / latest_installed_branch /
# declared_or_selected_branch). Branch names only. None for v2 runs.
branch_roles: dict[str, Any] | None = None
```

In `run_v3`'s finalize block: resolve `declared_or_selected` and stamp the
artifact **before** `artifact.investigation_state = state.to_artifact_dict()`
(loop_v3.py:2315), i.e. in the `# --- finalize ---` region:

```python
declared_or_selected: str | None = None
if executor.solution is not None:
    declared_or_selected = executor.solution.branch
elif getattr(executor, "unsolved_declaration", None) is not None:
    declared_or_selected = executor.unsolved_declaration.get("best_branch")
elif artifact.fallback_selection is not None:
    declared_or_selected = _fallback_best_branch  # see below
artifact.branch_roles = _compute_branch_roles(
    state, executor, declared_or_selected
)
```

`_fallback_best_branch`: the fallback block (loop_v3.py:2200-2271) computes
`best_branch, fallback_selection = _select_v3_fallback(...)` inside its
`if`. Hoist a module-local variable: initialize
`fallback_best_branch: str | None = None` just before that block and assign
`fallback_best_branch = best_branch` inside it (both tiers — the attested
fallback AND the best-effort/unsolved tier count as "selected"; master 509
says "declared **or selected**"). An interrupted (`stopped`) or
`no_tool_calls`-exhausted run that never reaches the fallback block gets
`declared_or_selected_branch=None` — correct: nothing was declared or
selected.

Ordering note: the attested-fallback tier sets `executor.solution`, so the
`executor.solution` check above already covers it; the explicit
`fallback_best_branch` arm only fires for the honest best-effort tier.

### 1.4 Serialization, resume defaults, firewall

- Serialization: `RunArtifact.to_dict()` handles the plain dict natively.
  Exact shape, all four keys ALWAYS present:
  `{"best_scored_branch": str|None, "workflow_branch": str|None,
    "latest_installed_branch": str|None, "declared_or_selected_branch": str|None}`.
- **NOT** stored in `InvestigationState` / `to_artifact_dict`
  (state.py:581) — the roles are derived views; state stores only sources
  of truth. `from_artifact_dict` (state.py:624) is untouched. Resume
  default: recompute — restoring `investigation_state` and calling
  `_compute_branch_roles(restored_state, fresh_executor)` reproduces the
  first three roles; `declared_or_selected_branch` is termination-scoped
  and passed explicitly (None mid-run).
- Firewall: values are branch names or None. No decode text, no scores, no
  ground truth. No change needed to `tests/test_ground_truth_firewall.py`'s
  v3 context test.

### 1.5 Narrate (human output, workspace-snapshot surface)

`src/agent/narrate.py::_on_workspace_snapshot` (line 273). In the existing
`if self.verbose:` branch, after the current `snapshot: best=...` line,
render one additional dim line **only when** `payload.get("branch_roles")`
is a dict AND any of `workflow_branch` / `latest_installed_branch` /
`declared_or_selected_branch` is non-None and different from
`best_scored_branch`:

```
      roles: workflow=<w> installed=<i> selected=<s>
```

(render `-` for None). Non-verbose output is unchanged; v2 payloads (no
key) are unchanged. This plus the analyzer section (4.9) is the "human
output must distinguish" requirement; the primary machine surface is the
snapshot payload + artifact field.

---

## 2. Part B — Episode-ledger budget keys (minimal new instrumentation)

Analyzer section 1 ("episode budget requested vs executed") needs the
per-episode budget, which the ledger does not carry today
(`EpisodeResult.ledger_dict`, episodes.py:579-599, records only
`tool_call_count` / `suppressed_over_budget_calls`). Smallest addition:

### 2.1 `EpisodeSpec` — capture the pre-override registered cap

In `EpisodeSpec.__post_init__` (episodes.py:515-545): the clamp block
already computes `registered_max = self.budget.max_tool_calls` before
overriding. Stash it as an attribute in BOTH paths:

```python
# before the `max_calls` override block:
self.registered_max_tool_calls = self.budget.max_tool_calls
```

(then the override block reads `registered_max = self.registered_max_tool_calls`).
Plain attribute set in `__post_init__` — do not add a dataclass field (it
is derived, and adding a field would change the constructor surface).

### 2.2 `EpisodeResult` + ledger

Add two fields to `EpisodeResult` (after `suppressed_over_budget_calls`,
episodes.py:566):

```python
# M5.3 Slice 7: effective budget + the raw model-facing override request,
# so the analyzer can render requested vs registered vs executed without
# reconstructing kind registries.
budget: dict[str, Any] = field(default_factory=dict)
requested_max_tool_calls: int | None = None
registered_max_tool_calls: int | None = None
```

`ledger_dict` adds all three keys (right after
`suppressed_over_budget_calls`).

### 2.3 `run_episode` — populate

- In `_finish` (episodes.py:1175-1207), pass:
  `budget=budget.to_dict()`,
  `requested_max_tool_calls=_requested_calls(spec)`,
  `registered_max_tool_calls=getattr(spec, "registered_max_tool_calls", None)`.
- In the setup-failure `EpisodeResult` (episodes.py:1125-1133), pass
  `budget=spec.budget.to_dict()` and the same two values (spec is in scope).
- `_requested_calls(spec)`: module-level helper — the RAW model-facing
  request, `int`-coerced when parseable, else `None`:
  ```python
  def _requested_calls(spec: EpisodeSpec) -> int | None:
      raw = spec.inputs.get("max_tool_calls")
      if raw is None:
          return None
      try:
          return int(raw)
      except (TypeError, ValueError, OverflowError):
          return None
  ```

Additive keys flow through `state.episode_ledger` →
`artifact.episodes` and state serialization automatically (`dict(item)`
copies). No test currently asserts exact ledger key sets (verified at
`a0ba63c`). Pre-Slice-7 artifacts simply lack the keys; the analyzer
renders `n/a`.

---

## 3. Part C — nothing else changes in production

Explicitly UNCHANGED: `workflow_state` / `allowed_episode_kinds`
(context.py), `_settle_repair_outcome` / `_dispatch_repair_transaction`
(loop_v3.py:1217-1748), experiment dispatch (loop_v3.py:1856-1894 — its
results intentionally reach only `artifact.messages`; the analyzer reads
them from the timeline, see 4.8), `execute_composite` ToolCall timing
(actions.py:236-266 — already records `elapsed_ms` per composite call,
which is the read-side source for section 4.10; **no new timing
instrumentation is required**), `_workspace_snapshot_payload` signature,
all tool schemas, all budgets, all prompts.

---

## 4. Part D — `scripts/inspect_artifact.py`

### 4.1 Shared run-facts helper (header/summary parity)

The current `format_header` (line 274) reads keys that do not exist on the
v2/v3 `RunArtifact` shape (`provider`, `iterations_used`, `best_branch`,
`test_id`, `cipher_system`) — master failure 8. Fix once, in one place:

```python
def derive_run_facts(artifact: dict) -> dict:
    """Provider/model/iterations/branch/declaration/attestation/cost facts
    correct for the v2+v3 RunArtifact shape, with graceful fallbacks for the
    older ad-hoc shapes (automated-runner artifacts carry explicit
    provider/test_id/iterations_used keys — those take precedence)."""
```

Fields and derivations:

| fact | derivation (first non-empty wins) |
|---|---|
| `model` | `artifact["model"]` |
| `provider` | `artifact.get("provider")` → `infer_provider_from_model(model, None)` (already imported) → `"?"` |
| `cipher` | `artifact.get("cipher_id")` → `artifact.get("test_id")` → `artifact.get("cipher_system")` → `"?"` |
| `loop_version` | `artifact.get("loop_version") or "v2"` |
| `status` | `artifact.get("status") or "?"` |
| `iterations` | `artifact.get("iterations_used")` → `_iterations_used(artifact, artifact.get("loop_events") or [])` (existing helper, line 1199; falls back to max `tool_calls[].iteration`) → `"?"` |
| `declared` | `(artifact.get("solution") or artifact.get("declared_solution")) is not None` |
| `fallback` | existing rule (`status == "fallback_declared"` or `auto_declared`) |
| `final_branch` | `(artifact.get("branch_roles") or {}).get("declared_or_selected_branch")` → `_declared_branch(artifact)` (existing helper, line 1179) → `"?"` |
| `attestation_status` | see below |
| `cost_usd` | `float(artifact.get("estimated_cost_usd") or 0.0)` |

`attestation_status` string:
- if the solution carries an attestation dict:
  `"positive (declared)"` / `"weak (declared)"` / `"negative (declared)"`
  via `attestation_is_positive` (already imported) and `reader_accepts`
  (same verdict rule as `format_attestations`, line 377-382);
- else if `artifact.get("attestations")`:
  `f"{len(attestations)} recorded ({n_positive} positive)"`;
- else `"none"`.

### 4.2 `format_header` — rewrite body

Keep the banner style. New content:

```
======================================================================
  Model   : gpt-5.5 (openai)   loop=v3
  Cipher  : borg_single_B_borg_0109v  language=la
  Iters   : 25   status=unsolved   declared=False
  Accuracy: char=91.3%  word=67.9%          (line only when char is not None)
  Branch  : <final_branch>   attestation: 5 recorded (0 positive)
  Cost    : $4.0089
======================================================================
```

All values from `derive_run_facts`. `build_llm_summary` (line 705) merges
the same facts dict (replace its hand-rolled `provider`/`iterations_used`/
`best_branch`/`declared`/`auto_declared` entries with the helper's values,
keeping its existing keys' names so the LLM prompt shape is stable; add
`loop_version`, `attestation_status`, `final_branch`).

### 4.3 New sections — general contract

Each is a top-level `def format_<name>(artifact: dict) -> str` (section
4.8 additionally takes `timeline`), returns `""` when it has nothing to
show, and is wired into `inspect_one` (line 1343) with the same
`if text: print(text); print()` pattern. Assembly order in `inspect_one`:

1. header
2. `format_episodes` (existing)
3. **`format_episode_budgets`** (new)
4. **`format_suppressed_calls`** (new)
5. `format_readings`, `format_attestations`, `format_experiments` (existing)
6. **`format_experiment_validation_failures`** (new)
7. **`format_repair_cycles`** (new)
8. **`format_saturation`** (new)
9. **`format_repair_transactions`** (new)
10. **`format_branch_roles`** (new)
11. **`format_repair_hypothesis_time`** (new)
12. `format_composite_calls`, `format_tool_summary`, timing, automated
    steps, timeline (existing, unchanged order)

Shared helper: `_inv_state(artifact) -> dict` returning
`artifact.get("investigation_state") or {}`; and
`_short_hash(h) -> str` returning `str(h)[:12]` (`""` → `"-"`).

### 4.4 `format_episode_budgets` — master 515

Data: `artifact["episodes"]` — per entry `kind`, `status`,
`requested_max_tool_calls` (Slice 7 key; absent → `-`),
`registered_max_tool_calls` (absent → `n/a`),
`budget.max_tool_calls` (effective; absent → `n/a`),
`tool_call_count`, `suppressed_over_budget_calls` (absent → 0),
`elapsed_seconds`. Returns `""` when there are no episodes.

```
Episode budgets (requested → effective / registered, executed):
  kind      requested  registered  effective  executed  skipped  elapsed
  reading           2          16          2         2        2     0.1s
  verify            -           1          1         0        0     0.0s
```

### 4.5 `format_suppressed_calls` — master 516

Data: `artifact["episodes"]` entries with
`int(suppressed_over_budget_calls or 0) > 0`. `""` when none. One line per
episode: `episode_id`, kind, suppressed count, effective cap, executed.
(The synthesized `budget_exhausted` tool_results also exist in
`messages`, but the ledger count is authoritative — Slice 1 writes it.)

### 4.6 `format_repair_cycles` — master 517 ("grouped by content hash")

Data: `_inv_state(artifact).get("repair_transactions") or []` (this list is
NOT top-level on RunArtifact — it lives inside `investigation_state`;
written by `_settle_repair_outcome`, loop_v3.py:1254). Group by
`source_content_hash`. Per group render: short hash, transaction count,
distinct `pair_digest` count (missing on pre-Slice-2 records → count
distinct `reading_id` instead and mark `~`), status sequence in
`created_turn` order (e.g. `failed(no_op), installed, failed(unsupported)`),
and distinct `attestation_key`s (absent → `n/a`). `""` when no
transactions.

### 4.7 `format_saturation` — master 518 ("saturation transitions")

Data: `_inv_state(artifact).get("repair_saturation") or {}` (Slice 2 field;
absent on pre-Slice-2 artifacts → `""`). Per entry (state.py:327-343
field table): short candidate hash, `attestation_key`,
`evidence_failures`, total process failures (`sum(process_failures.values())`),
`readings`, `exhausted`, `pending_experiment_id` (or `-`),
`created_turn`→`updated_turn`. Additionally derive the **transition turn**:
scan that entry's transactions (match `saturation_key` on Slice-2+ records)
in `created_turn` order and report the turn of the transaction whose
`counted_evidence_failure` made the count reach 2, as
`exhausted at turn N` (omit when not derivable — e.g. pre-Slice-2 records
or seeded entries). `""` when the map is empty.

### 4.8 `format_experiment_validation_failures` — master 520

Data source: the **timeline** (i.e. `artifact["messages"]`), because
successful and failed `experiment_submit` dispatches return their JSON
directly without a `ToolCall` record (loop_v3.py:1859-1894). Signature:
`format_experiment_validation_failures(timeline: list[dict]) -> str`.
Scan every timeline tool call whose `name` is `experiment_submit` (or
`experiment_collect`) and whose parsed `result` dict has either
`config_errors` or an `error` string starting with
`"invalid experiment config"` / `"unknown experiment type"`. Per row:
iteration, tool, first 2 `config_errors` entries (truncated to ~90 chars),
and `corrected_example: yes/no` (key present and non-None). `""` when no
matches. (The committed M5.2 fixture contains exactly one such failure —
`target_language`/`cipher_hint` unknown keys — verified in the stored
artifact.)

### 4.9 `format_branch_roles` — master 521 ("workflow branch vs score-selected branch")

Data: `artifact.get("branch_roles")` (Part A). `""` when absent (v2 / old
artifacts). Render the four roles, one per line, with `-` for None, plus a
divergence marker:

```
Branch roles:
  best_scored_branch        : main
  workflow_branch           : transaction_repaired   [differs from best-scored]
  latest_installed_branch   : transaction_repaired
  declared_or_selected_branch: transaction_repaired
```

### 4.10 `format_repair_hypothesis_time` — master 522

Data: `artifact["tool_calls"]` rows whose `tool_name` is one of
`{"hypothesis_test_words", "hypothesis_test_word", "hypothesis_apply_reading"}`
(lead + episode; composite calls carry real `elapsed_ms` via
`execute_composite`, actions.py:255-266 — the same source that produced the
M5.2 smoke's 456.3 s figure). Per tool: count, total seconds, max seconds.
For the two `hypothesis_test_word*` tools, parse each `result` JSON and
tally `menu_source` values (`built` / `cache` / `not_built`;
`_finalize_word_batch`, actions.py:1452-1462) so menu REBUILDS vs cache
hits are visible. Headline line: cumulative seconds over the three tools.
`""` when no such calls exist. Note in the docstring that lead
`repair_transaction` ToolCalls carry `elapsed_ms=0` by design
(`_record_dispatch_result`, loop_v3.py:691-703): the real hypothesis work
is timed on the inner composite calls, so this section neither
double-counts nor misses it.

```
Repair hypothesis time: 456.3s cumulative
  hypothesis_test_word    35 calls  total=456.3s  max=32.1s  menus: built=34 cache=1
  hypothesis_apply_reading 3 calls  total=12.0s   max=6.2s
```

Master 519 ("installed and rejected repair transactions") is section 4.11:

### 4.11 `format_repair_transactions`

Data: `_inv_state(artifact).get("repair_transactions") or []`, in list
order. One row per transaction: short `transaction_id`, `status`,
`reason` (failures; `-` otherwise), `failure_class` +
`counted_evidence_failure` (absent on pre-Slice-2 records → `n/a`),
`worker_winner`→`installed_branch` (installs), `retry_of` (short, or `-`),
and an acceptance summary when the record carries `acceptance`
(Slice 4): `checks passed x/y`, plus `dict_rate_delta`/`quad_delta` from
`acceptance.score_deltas` when numeric. `""` when no transactions.

### 4.12 Explicitly UNCHANGED in the analyzer

`build_timeline`, `format_timeline`, `format_tool_summary`,
`analyze_tool_timing`, `format_automated_steps`, the LLM-analysis path
(`_call_llm` etc.), and `render_narrative` are unchanged except that
`build_llm_summary` consumes `derive_run_facts` (4.2).

---

## 5. Part E — Trimmed M5.2 analyzer regression fixture

### 5.1 Source and location

- Source artifact (local only; `artifacts/` is gitignored):
  `artifacts/m5_2_targeted_smoke_20260716/v3/borg_single_B_borg_0109v/1/borg_single_B_borg_0109v/d3eccab14a40.json`
  (1.46 MB; the exact artifact named in the master spec, line 88-89).
- Committed output: `tests/fixtures/v3_artifact_m5_2_smoke_trimmed.json`
  (follows the existing `v2_artifact_synth_en_40wb_s1.json` naming
  convention). Expected size after trimming: roughly 400–700 KB; pin the
  actual value informally in the trim-script docstring, not in a test.

### 5.2 Producer: `scripts/trim_artifact_fixture.py` (NEW)

CLI: `python scripts/trim_artifact_fixture.py <src.json> <dst.json>`.
Pure stdlib. Deterministic (`json.dump(..., indent=2, sort_keys=True)`).
Trim rules (exact):

1. **Delete top-level keys**: `ground_truth`, `session_transcript`,
   `benchmark_context`, `loop_events`, `notebook`, `subagent_runs`.
   Set `plan` to `""`.
2. **`messages`** (retained — the analyzer's timeline/tool-summary and
   section 4.8 consume it): for every content block,
   - `type == "text"` → replace `text` with `""` (raw model prose removed);
   - drop `provider_extra`, `thinking`, `reasoning`, `signature` keys from
     any block dict;
   - `tool_use` blocks: keep `type`/`id`/`name`/`input` only;
   - `tool_result` blocks: keep, but when `content` is a string longer
     than 16 384 chars, replace with
     `json.dumps({"_trimmed": True, "status": parsed.get("status")})`
     (parse first; unparseable long strings become
     `'{"_trimmed": true}'`).
3. **`tool_calls`**: apply the same >16 384-char `result` truncation rule.
   Keep `arguments` (structural inputs).
4. **`investigation_state`**: keep ONLY the allowlist
   `{"language", "turn", "repair_transactions", "repair_saturation",
   "repair_agenda", "workflow_hint_keys", "call_signature_counts",
   "no_new_information_streak", "model_variant"}` — exactly the keys the
   analyzer consumes plus small counters. Everything else (cipher,
   workspace, evidence_log, budget_ledger, recent_exchanges,
   external_context, episode_ledger, verify_attestations, readings,
   hypothesis_board, branch_aliases, finalist_sessions, experiment_queue,
   last_information_digest) is dropped. The fixture is analyzer-only; it
   is NOT loadable via `InvestigationState.from_artifact_dict` — say so in
   the script docstring and the fixture test.
5. **`branches`**: truncate each `decryption` to 1 200 chars.
   **`automated_preflight`**: truncate its `decryption` to 1 200 chars.
6. **`episodes`**: drop `transcript` and `raw_text` keys when present
   (none in this artifact, but the rule keeps the script reusable). Keep
   everything else verbatim (results/summaries are structured worker
   submissions the analyzer renders, not raw message bodies).
7. Keep verbatim: `run_id`, `cipher_id`, `model`, timestamps, counters,
   `status`, `char_accuracy`, `word_accuracy`, `estimated_cost_usd`,
   `loop_version`, `budget_by_category`, `solution`,
   `fallback_selection`, `error_message`, `final_summary`,
   `cipher_id_report`, `cipher_hypotheses`, `repair_agenda`,
   `tool_requests`, `served_models`, `safety_gate_fired`,
   `preprocessing_applied`, `attestations`, `readings`, `experiments`,
   `max_iterations`, `cipher_alphabet_size`, `cipher_token_count`,
   `cipher_word_count`, `parent_run_id`, `parent_artifact_path`.

Rationale line to keep in the script docstring: `char_accuracy` /
`word_accuracy` are post-hoc grading SCORES, not the answer; the header
renders them, so they stay. The GT text itself (`ground_truth`), the
benchmark prompt (`benchmark_context`, `external_context`), alignment
material (none is stored in this artifact shape — alignments were post-hoc
session work, never artifact keys; the banned-key test still guards the
names), and every raw model body (assistant text, session transcript,
recent exchanges, loop-event `agent_text`) are removed.

### 5.3 Firewall test over the committed fixture

`tests/test_ground_truth_firewall.py::test_trimmed_m5_2_fixture_contains_no_ground_truth_keys`

- Load `tests/fixtures/v3_artifact_m5_2_smoke_trimmed.json`.
- Recursive sweep of every dict key in the JSON tree; assert no key equals
  (exact match, case-sensitive) any of:
  `{"ground_truth", "expected", "expected_plaintext", "plaintext",
  "alignment", "alignments", "alignment_blocks", "solution_key",
  "session_transcript", "recent_exchanges", "external_context",
  "benchmark_context"}`.
  (Exact-match deliberately: `plaintext_symbols` is a legitimate alphabet
  key elsewhere, but the fixture's allowlist already excludes the
  structures that carry it — the sweep must still pass.)
- Assert `artifact["plan"] == ""` and every `messages[*]` text block has
  `text == ""` (raw-body removal is structural, not just key-based).
- Assert the `investigation_state` key set is a subset of the 5.2
  allowlist.

Coordinate: this file already exposes `assert_no_ground_truth_leak`; the
new test is key-structural (the Borg GT text must NOT be embedded in the
test file, so a content sweep is impossible by design — key + emptiness
assertions are the right tool).

### 5.4 Analyzer regression test over the fixture

`tests/test_inspect_artifact.py::test_trimmed_m5_2_fixture_analyzer_regression`

- Load the fixture; `facts = derive_run_facts(artifact)`; assert
  `model == "gpt-5.5"`, `provider == "openai"` (inferred),
  `loop_version == "v3"`, `status == "unsolved"`, `declared is False`,
  `cost_usd == pytest.approx(4.0089, abs=0.01)`, `iterations` equals the
  value observed from the fixture's tool_calls (pin the literal once the
  fixture is built — deterministic).
- `format_header` output contains `v3`, `unsolved`, `openai`.
- `format_episodes` renders (20 episode rows).
- `format_episode_budgets` renders with `n/a` in the
  registered/effective columns (pre-Slice-7 ledger).
- `format_suppressed_calls(artifact) == ""` (no Slice-1 counts in this
  artifact).
- `format_repair_cycles` and `format_repair_transactions` render 8
  transactions (statuses observed: 6 failed / 2 installed) with `n/a`
  failure_class (pre-Slice-2 records).
- `format_saturation(artifact) == ""` (`repair_saturation` absent).
- `format_experiment_validation_failures(build_timeline(artifact))`
  contains `target_language`.
- `format_branch_roles(artifact) == ""` (pre-Slice-7).
- `format_repair_hypothesis_time` renders with a cumulative total > 400 s
  and a `hypothesis_test_word` row with 35 calls.
- End-to-end: `inspect_one(fixture_path, analyze=False, ...)` via
  `capsys` completes without exception.

---

## 6. Part F — Shared scripted-v3 test support module (hoist)

### 6.1 Decision

Hoist into `tests/support/scripted_v3.py` (new package dir with an empty
`tests/support/__init__.py`). Rationale: `ScriptedSession` is currently
defined THREE times (`tests/test_loop_v3.py:34`,
`tests/test_m6_m5_note_fixes.py:34`, `tests/test_cli_observability.py:313`)
and Sequence B would be a fourth. Hoist + convert **only**
`tests/test_loop_v3.py` (the file whose fakes Sequence B reuses); leave the
other two files' private copies untouched (converting them is optional
cleanup, not Slice-7 scope — do not churn them). The `tests` directory is
already a package with cross-file imports
(`from tests.test_ground_truth_firewall import ...` in `test_multipage.py`),
so `from tests.support.scripted_v3 import ...` is the established pattern.
No conftest.py exists and none is added. Do NOT build any record/replay
cassette framework.

### 6.2 Contents (moved verbatim from `tests/test_loop_v3.py`, plus two small additions)

Moved (delete from `test_loop_v3.py`, import instead):

- `ScriptedSession` (line 34) and `ErrorSession` (line 65);
- `VerifyWorkerFake` (line 70) — keep as the positive-verdict default;
- `ProgrammableRepairWorker` + the `_REPAIR_PROGRAMS` FIFO (lines
  1143-1202) + `_register_programmable_repair` (line 1243) — rename the
  registry/registrar to `REPAIR_PROGRAMS` / `register_programmable_repair`
  (public names in the support module);
- `_keyed_catton_state` (line 660) → `keyed_catton_state`;
- `_seed_reading` (line 1209) → `seed_reading`;
- `_seed_negative_attestation` (line 1222) → `seed_negative_attestation`.

`tests/test_loop_v3.py` keeps thin aliases at its import site
(`_keyed_catton_state = keyed_catton_state`, etc.) so its ~40 call sites
need no edits beyond deleting the moved definitions and adding one import
block. Behavior identical; suite must stay green.

New in the support module (needed by Sequence B):

```python
NEGATIVE_LOCAL_REPAIR_VERDICT = {
    "coherence": 4, "reader_accepts": False,
    "reader_accepts_as_solution": False,
    "target_language_confidence": 0.8, "semantic_recoverability": 0.7,
    "damage_scope": "local", "repairability": "local_repair",
    "uncertainty_note": "", "gloss": "partly readable",
    "anomalies": ["damaged middle word"], "confidence": "medium",
}

def make_verify_builder(result: dict):
    """Session-builder factory: a verify worker submitting `result`.
    (VerifyWorkerFake remains the positive-verdict shorthand.)"""

class OverBudgetReadingWorkerFake:
    """Reading worker that emits FOUR decode_show tool_uses in one batch on
    its first send (over a 2-call effective budget → 2 executed + 2
    synthesized budget_exhausted skips), never volunteers a submit, and on
    the submit-only reserve send (the 'Budget reached' nudge, exposing only
    episode_submit_result) submits a valid reading:
    reading_text/fragments[0].text = repair_text = CLS.READING_TEXT
    (default "CATON"), holes=[], overall_confidence=0.8.
    Model name "gpt-5.5" so BudgetEntry costs are nonzero-deterministic."""
```

(Implementation mirrors `ReadingWorkerFake`, test_loop_v3.py:687, with the
batch + reserve behavior above; class attribute `READING_TEXT` so tests can
subclass.)

---

## 7. Part G — Sequence-B scripted end-to-end replay test

### 7.1 File and harness

`tests/test_v3_sequence_b.py` (NEW). Uses the support module (Part F),
`run_v3` with a custom lead session object (the `TransactionLead` pattern,
test_loop_v3.py:896), per-kind worker fakes registered via
`sessions_mod.register_session_builder` (the seam `episodes.py:1112`
consumes), a **synchronous** `ExperimentQueue(synchronous=True)` passed
through `run_v3(..., experiment_queue=...)`, and
`monkeypatch.setitem(experiments.EXPERIMENT_TYPES["automated_solver"], "runner", _fake_runner)`
so the REAL Slice-5 typed validation runs while the solver compute is
trivial:

```python
def _fake_runner(cipher, snapshot, config):
    return {"status": "no_solution", "solver": "fake", "error_message": "",
            "elapsed_seconds": 0.01, "key": {}, "final_decryption": "",
            "steps": []}
```

No paid model, no network, no real solver compute.

### 7.2 Setup

- `ct, state = keyed_catton_state()` (main decodes to `CATON`).
- Pre-seed ONE auxiliary stored reading:
  `rid_x = seed_reading(state, "COTON")`. Rationale (documented in the
  test): Slice-2 policy allows ONE fresh reading per (content, evidence)
  pair, so the second distinct interpretation that drives the second
  evidence failure stands in for the M5.2 run's pre-suppression duplicate
  readings — exactly the "stored version of the M5.2 sequence" the master
  names. (The Slice-2 tests `test_s2_two_evidence_failures_enter_repair_exhausted`
  seed both readings; Sequence B seeds only the auxiliary one and produces
  the first live.)
- Register workers:
  - `episode:verify` → `make_verify_builder(NEGATIVE_LOCAL_REPAIR_VERDICT)`;
  - `episode:reading` → `OverBudgetReadingWorkerFake` (submits "CATON" —
    equal to the current decode, so applying it later is a no-op);
  - `episode:repair` → `register_programmable_repair([prog1, prog2])`:
    - `prog1 = {"apply": [{}], "result": NO_FORK_APPLIED}` — one
      `hypothesis_apply_reading` call with the injected reading id (fork
      created but content-identical → not a changed finalist), then submit
      `applied=True, best_branch=None` → host classifies
      `no_changed_finalists` (evidence failure 1). The apply call also
      guarantees section 4.10 has at least one timed hypothesis call.
    - `prog2 = {"apply": [], "result": NO_FORK_APPLIED}` →
      `no_changed_finalists` (evidence failure 2 → exhausted).
- Lead: `SequenceBLead` class (in the test file, not the support module —
  it is sequence-specific). Records every `blocks` it is sent
  (`self.blocks_seen`) and every `tools` list; extracts prior tool_result
  payloads the way `TransactionLead` does. Model name `"gpt-5.5"`.

### 7.3 Lead turn program (one tool call per turn)

| turn | tool call | drives master bullet |
|---|---|---|
| 1 | `episode_run {kind: "verify", goal, branches: ["main"]}` | verification occurs early |
| 2 | `episode_run {kind: "reading", goal, branches: ["main"], max_tool_calls: 2}` | reading budget + reserved submit-only attempt |
| 3 | `repair_transaction {branch: "main", reading_id: <rid_A from turn-2 result>, as_name: "tx1"}` | one reading feeds a bounded repair transaction |
| 4 | `repair_transaction {branch: "main", reading_id: rid_x, as_name: "tx2"}` | second distinct pair → saturation |
| 5 | `repair_transaction {branch: "main", reading_id: rid_x, as_name: "tx3"}` | blocked: `repair_transaction_not_ready` (workflow `repair_exhausted`) |
| 6 | `experiment_submit {type: "automated_solver", branch: "main", config: {"target_language": "en", "allow_homophones": True, "max_runtime_seconds": 60}}` | structured validation error + corrected example (mirrors M5.2 failure 5) |
| 7 | `experiment_submit {type: "automated_solver", branch: "main", config: <the corrected_example extracted from turn 6's tool_result>}` | alternate search accepts a valid config |
| 8 | `meta_declare_unsolved {best_branch: "main", rationale, reading_summary}` | honest termination |

`rid_A` extraction: turn-3's send sees turn-2's tool_result payload
(`kind == "reading"`, `reading_id` present) — reuse the
`_episode_id_from_blocks` scanning pattern (test_loop_v3.py:674).
`max_iterations=10`.

### 7.4 Assertions (mapped 1:1 to master 540-552)

Let `art = run_v3(...)`, `inv = art.investigation_state`,
`h = content hash of main` (compute with
`_candidate_content_hash(_decoded_text_for_panel(...))` before the run),
and `att_key = "ep:" + <verify episode_id from art.attestations[0]>`.

1. **Verification occurs early**: `art.episodes[0]["kind"] == "verify"`,
   `status == "ok"`, `art.attestations[0]["created_turn"] == 1`, and the
   attestation is non-positive
   (`attestation_is_positive(art.attestations[0]) is False`).
2. **One reading feeds a bounded repair transaction**:
   `inv["repair_transactions"][0]["reading_id"] == rid_A`;
   the repair episode ledger entries (`kind == "repair"`) have
   `registered_max_tool_calls == 6` and `tool_call_count <= 6`;
   `art.readings` contains exactly the compiled reading A plus seeded X.
3. **Reading exploration cannot consume the reserved submit-only
   attempt**: the reading ledger entry has
   `requested_max_tool_calls == 2`, `budget["max_tool_calls"] == 2`,
   `registered_max_tool_calls == 16`, `tool_call_count == 2`,
   `suppressed_over_budget_calls == 2`, `status == "ok"`, and a non-empty
   `result["reading_text"]` — i.e. the over-emitted batch was clipped and
   the submission still succeeded through the reserve.
4. **Repeated no-op repair saturates**: both transactions have
   `status == "failed"`, `reason == "no_changed_finalists"`,
   `failure_class == "evidence"`, `counted_evidence_failure is True`;
   `inv["repair_saturation"][saturation_key(h, att_key)]` has
   `evidence_failures == 2` and `exhausted is True`; turn-5's
   `repair_transaction` lead result is
   `{"status": "blocked", "reason": "repair_transaction_not_ready",
   "workflow_state": "repair_exhausted"}`.
5. **Alternate search is offered and accepts a valid config**:
   (a) offered — some lead context text block sent on turn ≥ 6 contains
   `"Workflow state: repair_exhausted"` and `"experiment_submit"`
   (scan `lead.blocks_seen`);
   (b) turn-6 result parses to
   `error == "invalid experiment config"` with `config_errors` naming
   `target_language` and a `corrected_example` key;
   (c) turn-7 result carries an `experiment_id` and status
   `completed` (synchronous queue + fake runner), and
   `art.experiments` contains exactly one record (the invalid attempt never
   entered the queue);
   (d) the saturation entry's `pending_experiment_id` equals that
   experiment id (loop_v3.py:1879-1893).
6. **No worker exceeds its call budget**: for EVERY entry in
   `art.episodes`: `tool_call_count <= budget["max_tool_calls"]` (all
   Slice-7 ledger keys present on every entry).
7. **Termination is honest**: `art.status == "unsolved"`,
   `art.solution is None`, `art.auto_declared is False`,
   `art.attested_fallback is False`,
   `art.branch_roles == {"best_scored_branch": "main",
   "workflow_branch": "main", "latest_installed_branch": None,
   "declared_or_selected_branch": "main"}`. Also assert exchange pairing:
   every tool_use id in `art.messages` has exactly one tool_result (reuse
   the `pair_sets` pattern, test_loop_v3.py:1012).
8. **The analyzer reports the full path correctly**: round-trip
   `data = json.loads(json.dumps(art.to_dict()))`, then:
   - `derive_run_facts(data)`: status `unsolved`, loop `v3`, provider
     `openai`, model `gpt-5.5`, declared False, final_branch `main`;
   - `format_episode_budgets(data)` contains a reading row with
     `2`, `16`, `2`, `2`, `2` in the requested/registered/effective/
     executed/skipped columns;
   - `format_suppressed_calls(data)` names the reading episode;
   - `format_repair_cycles(data)` shows ONE group (short hash of `h`) with
     2 transactions and 2 distinct pairs;
   - `format_saturation(data)` shows `exhausted=True`,
     `evidence_failures=2`, the pending experiment id, and
     `exhausted at turn 4`;
   - `format_repair_transactions(data)` shows two `failed` rows with
     `evidence` class;
   - `format_experiment_validation_failures(build_timeline(data))`
     names `target_language` and `corrected_example: yes`;
   - `format_branch_roles(data)` renders all four roles;
   - `format_repair_hypothesis_time(data)` shows
     `hypothesis_apply_reading` with 1 call.

The analyzer module is loaded the way `tests/test_inspect_artifact.py`
does (importlib over `scripts/inspect_artifact.py`); import that module's
loader helper or replicate the 6-line loader locally.

Cleanup: `finally:` pop all three registered session builders
(the established pattern).

### 7.5 Known constraint recorded (not a Slice-7 change)

A single (content, evidence) pair cannot reach two evidence failures with
only ONE reading, because Slice 2 caps fresh readings at one per pair and
an evidence-failed pair cannot be rerun. Live exhaustion therefore
requires a second interpretation minted under earlier/no verifier evidence
(the `att_key="none"` epoch) or an installed-and-reverified cycle. This is
Slice-2 behavior, exercised as such by its tests; Sequence B mirrors those
tests by seeding the auxiliary reading. FLAGGED as an observation for the
maintainer, NOT resolved here (no master contradiction: master 287-291
permits "up to two distinct repair transactions" without promising two
fresh readings).

---

## 8. Part H — Sequence-A housekeeping (master 531-538)

`tests/test_lead_context.py::test_negative_partial_attestation_creates_repair_action_menu`
(line 82) was already claimed and updated to the Slice-2 menu contract by
the Slice 2+4 work (see `agent_v3_m5_3_slice2_4_impl_spec.md` §U1, line
1221) and further reshaped by Slice 6 (it now seeds the diplomatic verdict
fields and asserts `repair_required` + "repair episode" wording). Verified
green at `a0ba63c` (in the 1644-passed baseline). Slice 7 actions:

1. Re-run the test and confirm green (no residual stale assertion — the
   assertions reference only the live `_attested_menu` wording).
2. Add one provenance comment line to the test docstring:
   `"""Pre-M5.3 test, claimed+updated to the Slice-2 menu contract (M5.3);
   see agent_v3_m5_3_slice2_4_impl_spec.md §U1."""` — this is the
   master-required "record that it predated M5.3".
3. No other change.

---

## 9. Required tests (file::name, setup → assertions)

Production-behavior tests:

1. `tests/test_loop_v3.py::test_branch_roles_in_snapshot_and_artifact`
   — setup: the `TransactionLead` flow (reading → repair_transaction
   installs `transaction_repaired` → verify → declare), i.e. reuse
   `test_repair_transaction_runs_validates_installs_and_requires_reverify`'s
   fixtures with the hoisted helpers. Assert:
   `art.branch_roles` has exactly the four keys;
   `latest_installed_branch == "transaction_repaired"`;
   `declared_or_selected_branch == "transaction_repaired"`;
   `workflow_branch` is a str; at least one `workspace_snapshot` loop
   event payload carries `branch_roles` with
   `declared_or_selected_branch is None` (mid-run).
2. `tests/test_loop_v3.py::test_branch_roles_recomputable_after_resume`
   — setup: take the artifact from (1);
   `restored = InvestigationState.from_artifact_dict(art.investigation_state)`;
   build an executor (`_slice_executor` pattern, test_loop_v3.py:1236);
   `roles = _compute_branch_roles(restored, ex)`. Assert the three
   non-termination roles equal the artifact's, and
   `roles["declared_or_selected_branch"] is None` (resume default);
   also assert `"branch_roles" not in art.investigation_state` (derived,
   not state).
3. `tests/test_loop_v3.py::test_branch_roles_honest_unsolved_fallback`
   — setup: text-only lead (`test_run_v3_exhaustion_is_honestly_unsolved`
   shape). Assert `art.status == "unsolved"` and
   `art.branch_roles["declared_or_selected_branch"]` equals the
   best-effort selected branch named in `art.final_summary` /
   `best_effort_selected` event (fallback-selected counts as "selected").
4. `tests/test_episodes.py::test_episode_ledger_records_budget_requested_and_registered`
   — setup: run a scripted survey episode (existing `EpisodeFake`
   machinery, test_episodes.py:37) with `inputs={"max_tool_calls": 3}`.
   Assert the ledger entry has `budget["max_tool_calls"] == 3`,
   `requested_max_tool_calls == 3`, `registered_max_tool_calls == 10`
   (survey kind cap); a second run WITHOUT the override records
   `requested_max_tool_calls is None` and effective == registered.

Analyzer-section tests (all in `tests/test_inspect_artifact.py`; each
asserts both the rendered case and the `""`-when-empty case using small
synthetic artifact dicts):

5. `test_derive_run_facts_v3_artifact_shape` — synthetic v3 dict with
   `model="gpt-5.5"`, no `provider` key, `loop_version="v3"`,
   `status="unsolved"`, tool_calls with iterations 1..7,
   `branch_roles={"declared_or_selected_branch": "best", ...}`,
   `attestations` with one negative record → provider inferred `openai`,
   iterations 7, final_branch `best`, attestation summary
   `1 recorded (0 positive)`; header string contains all of them.
   Also: an automated-runner-shaped dict with explicit
   `provider`/`iterations_used`/`best_branch` keys still wins (backward
   compatibility).
6. `test_format_episode_budgets_renders_and_empty` — one episode with the
   Slice-7 keys renders requested/registered/effective/executed/skipped;
   one WITHOUT them renders `-`/`n/a`; `format_episode_budgets({"episodes": []}) == ""`.
7. `test_format_suppressed_calls_renders_and_empty`.
8. `test_format_repair_cycles_groups_by_content_hash` — two transactions
   sharing `source_content_hash` + one on another hash → two groups;
   empty → `""`.
9. `test_format_saturation_shows_transition_turn` — a saturation entry
   with `exhausted=True` plus two matching counted transactions
   (`created_turn` 3 and 4, same `saturation_key`) → renders
   `exhausted at turn 4`; empty map → `""`.
10. `test_format_repair_transactions_installed_and_rejected` — one
    installed record with a full Slice-4 `acceptance` dict (checks 7/7,
    deltas) and one failed record with
    `failure_class`/`counted_evidence_failure`, plus one pre-Slice-2
    record (no class) → renders rows with `n/a` where absent; empty →
    `""`.
11. `test_format_experiment_validation_failures_from_timeline` — messages
    with one `experiment_submit` tool_use whose result carries
    `config_errors` + `corrected_example`, and one successful submit →
    exactly one row; no failures → `""`.
12. `test_format_branch_roles_renders_divergence_and_empty` — roles dict
    with `workflow_branch != best_scored_branch` renders the divergence
    marker; artifact without the key → `""`.
13. `test_format_repair_hypothesis_time_totals_and_menu_counts` —
    synthetic tool_calls: two `hypothesis_test_words` (results with
    `menu_source: "built"` / `"cache"`, elapsed_ms 2000/500) and one
    `hypothesis_apply_reading` (1000) → cumulative `3.5s`, menu counts
    `built=1 cache=1`; no hypothesis calls → `""`.

Fixture tests:

14. `tests/test_inspect_artifact.py::test_trimmed_m5_2_fixture_analyzer_regression`
    — as specified in 5.4.
15. `tests/test_ground_truth_firewall.py::test_trimmed_m5_2_fixture_contains_no_ground_truth_keys`
    — as specified in 5.3.

Display test:

16. `tests/test_agent_display.py::test_verbose_snapshot_renders_branch_roles`
    — feed `_on_workspace_snapshot`-reaching event with a `branch_roles`
    payload where `workflow_branch != best`; verbose renderer output
    contains `roles: workflow=`; a payload without the key renders no
    roles line (and non-verbose renders neither).

Sequence B:

17. `tests/test_v3_sequence_b.py::test_sequence_b_m5_2_shaped_replay_end_to_end`
    — as specified in Part G (all 8 master bullets in ONE test; keep it a
    single test so the flow is one artifact, with helper functions for the
    assertion clusters).

Suite hygiene: after the Part-F hoist, the whole existing suite must stay
green (`1644 + new` passed / 2 skipped); `test_loop_v3.py`'s existing tests
run unmodified against the imported helpers.

---

## 10. Local acceptance / verification (1:1 to master Slice 7 + Sequences A/B)

| Master requirement | Where satisfied |
|---|---|
| 504-509 snapshots + human output distinguish the four roles | Parts A.2 (snapshot payload), A.3 (artifact), A.5 (narrate), 4.9 (analyzer); tests 1-3, 12, 16, 17.7 |
| 511-513 summary shows provider/model, loop version, iterations, solved/unsolved declaration, final branch, attestation status, cost | 4.1/4.2 `derive_run_facts` + header; tests 5, 14, 17.8 |
| 515 episode budget requested vs executed | Part B + 4.4; tests 4, 6, 14, 17.3/17.8 |
| 516 suppressed over-budget calls | 4.5; tests 7, 17.8 |
| 517 repair cycles grouped by content hash | 4.6; tests 8, 17.8 |
| 518 saturation transitions | 4.7; tests 9, 17.8 |
| 519 installed and rejected repair transactions | 4.11; tests 10, 14, 17.8 |
| 520 experiment validation failures | 4.8; tests 11, 14, 17.8 |
| 521 workflow branch vs score-selected branch | 4.9; tests 12, 17.8 |
| 522 cumulative repair-hypothesis time | 4.10 (read-side over composite `elapsed_ms` + `menu_source`); tests 13, 14, 17.8 |
| 524-527 trimmed fixture, GT/plaintext/alignment/raw-bodies removed, structural fields retained, GT-key rejection test | Part E; tests 14, 15 |
| Sequence A (531-538): focused local tests, no paid model; stale-assertion claim recorded | §9 test list (all scripted/fake); Part H |
| Sequence B (540-552): all 8 replay proofs + analyzer agreement | Part G; test 17 |

Run for acceptance:
`PYTHONPATH=src .venv/bin/python -m pytest tests/test_loop_v3.py tests/test_episodes.py tests/test_inspect_artifact.py tests/test_ground_truth_firewall.py tests/test_agent_display.py tests/test_lead_context.py tests/test_v3_sequence_b.py -q`
then the full suite.

---

## 11. Non-goals / explicitly unchanged

- No Sequence-C paid smoke, Stage-1 packet, or M6 rerun (user-gated).
- No changes to budgets, saturation policy, acceptance checks, verify
  contract, experiment schemas, prompts, or tool surfaces.
- No cassette/record-replay framework; the support module is fakes only.
- No conversion of `test_m6_m5_note_fixes.py` / `test_cli_observability.py`
  to the shared `ScriptedSession` (optional later cleanup).
- No `InvestigationState` schema change (branch roles are derived).
- `_workspace_snapshot_payload` signature and v2 loop untouched.
- Docs: **no TOOLS.md change** (no model-facing tool changed) and **no
  CLAUDE.md change** (the analyzer is developer-facing and CLAUDE.md has
  no inspect_artifact section; do not add one).

## 12. Commit

One phase commit after review:
`M5.3 Slice 7: observability + analyzer parity, trimmed M5.2 fixture, Sequence-B replay`
— message body notes the Sequence-A claim (Part H) as predating M5.3 and
already handled in Slice 2.
