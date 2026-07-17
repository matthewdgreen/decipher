# Spec: InvestigationHost extraction (MCP plan, first code phase)

Status: READY FOR IMPLEMENTATION. Authored 2026-07-17 by the spec author
(Fable) from a full read of `src/investigation/loop_v3.py` at commit
`7715614` ("Harden M5.3 targeted acceptance harness"; working tree clean).
Authority: `docs/mcp_dual_harness_proposal.md` §7.0–§7.1 (seam analysis,
no-breakage rule) and §3.3 (gates stay host-enforced). This slice is the
"one real refactor" named there: extract `run_v3`'s lead **dispatch layer**
into a loop-independent `InvestigationHost` so a later MCP server can be a
second thin driver over the same implementation.

**This slice has ZERO behavior change.** No new features, no new events, no
new state fields, no artifact schema change, no tool change, v2 untouched.
The MCP server itself, the client-compiled `repair_transaction` variant
(proposal §3.5), the revision store, and the capsule launcher are ALL later
phases and explicitly out of scope here (§10).

All line numbers below refer to `src/investigation/loop_v3.py` as of commit
`7715614` (2,622 lines). The implementer should re-verify anchors with the
quoted code rather than trusting raw numbers if the file has drifted.

---

## 1. Objective and hard constraints

Create `src/investigation/host.py` containing class `InvestigationHost`,
which owns the v3 lead **dispatch layer** — everything that turns one
`tool_use` block into one tool-result string, including episode dispatch,
verify attestation writes, episode install, the repair-transaction
acceptance machinery, duplicate suppression, the read cache, the
information digest, and the budget seam. `run_v3` becomes a thin driver:
build context → `session.send` → for each tool_use → `host.handle_tool` →
turn bookkeeping.

Hard constraints (each is an acceptance criterion):

1. **No behavior change.** Every moved function body moves **verbatim**
   except the mechanical renames in §4.1 (closure variable → attribute).
   Comments and docstrings move with their code, unedited. Lazy `import`
   statements inside function bodies stay inside the method bodies,
   verbatim. All `json.dumps(...)` calls keep their exact flags
   (`ensure_ascii=False` where present today, absent where absent) so
   tool-result strings are byte-identical.
2. **All 1,710 existing tests pass.** They drive this code through
   `run_v3` with `ScriptedSession` fakes and must pass **unmodified**,
   except the four edits enumerated in §7.2, each of which changes ONLY an
   import path or a monkeypatch target (never an assertion, fixture value,
   or scenario).
3. **Artifact/event-stream byte compatibility**, proven by the scripted-run
   byte-parity procedure in §9.3 (zero diff between a pre-change and
   post-change capture with uuid/time frozen).
4. **v2 untouched.** No edit to `src/agent/loop_v2.py`, `loop_shared.py`,
   `tools_v2.py`, or anything outside the files listed in §2.
5. **The API-billed repair episode inside `repair_transaction` stays.**
   `_dispatch_repair_transaction` moves as-is, including its internal
   `_dispatch_episode_run` call that runs a paid repair episode. The MCP
   variant that validates client-compiled finalists is a LATER phase.
6. **`run_v3`'s public signature and return type are unchanged.** Its two
   production importers (`src/cli.py:1004`, `src/benchmark/runner_v2.py:224`)
   need no edits.
7. No new dependencies; Python 3.11; existing style (`from __future__
   import annotations`, `Any`-typed dict payloads).

## 2. Files touched

| File | Action |
|---|---|
| `src/investigation/host.py` | NEW (~1,350 lines, almost all moved code) |
| `src/investigation/loop_v3.py` | Shrinks to ~1,300 lines: module helpers that stay (§6.1) + `run_v3` rewired per §5 |
| `tests/test_loop_v3.py` | 2 edits (§7.2: one import, one monkeypatch target) |
| `tests/test_m6_m5_note_fixes.py` | 1 import edit (§7.2) |
| `tests/test_cli_observability.py` | 1 import edit (§7.2) |
| `src/investigation/__init__.py` | Docstring: add one `host` line to the module list |
| `CLAUDE.md` | Key Files: add `host.py` line under `src/investigation/` (see §8) |

Nothing else. In particular: NO edits to `context.py`, `episodes.py`,
`experiments.py`, `actions.py`, `state.py`, `reading.py`, `board.py`,
`sessions.py`, `adapter.py`, `artifact/schema.py`, `agent/*`, `cli.py`,
`benchmark/*`, `TOOLS.md`.

## 3. The new module: `src/investigation/host.py`

### 3.1 Module header

```python
"""InvestigationHost: the loop-independent v3 lead dispatch layer.

Extracted verbatim from ``run_v3``'s nested closures (MCP plan, host-
extraction slice; see docs/mcp_dual_harness_proposal.md §7.0). The host
owns everything that turns one lead ``tool_use`` into one tool-result
string — episode dispatch (including verify attestation writes), episode
install, the bounded repair transaction and its acceptance checks,
duplicate suppression, the lead read cache, the information digest, and
the per-run budget seam — while ``run_v3`` remains the turn-loop driver
(context build, session sends, truncation/nudge/429 handling, termination,
artifact finalize). A future MCP server is a second thin driver over this
same class; behavior here must stay identical for both entry points.

Epistemic gates (M5.3): the verification-gated declaration policy lives in
the executor's AttestationPolicy; the host enforces repair acceptance
(default-deny), saturation, duplicate suppression, and episode-kind gating
exactly as the loop closures did.
"""
```

### 3.2 Imports (exact)

```python
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from typing import Any, Callable

from agent.loop_shared import (
    DECODED_TEXT_RENDERER_ID,
    _candidate_content_hash,
    _decoded_text_for_panel,
)
from agent.tools_v2 import WorkspaceToolExecutor
from artifact.schema import ToolCall
from investigation.actions import COMPOSITE_TOOL_NAMES, execute_composite
from investigation.board import CARD_MIRROR_KEYS
from investigation.context import allowed_episode_kinds, workflow_state
from investigation.episodes import EpisodeSpec, run_episode
from investigation.experiments import (
    EXPERIMENT_TOOL_NAMES,
    ExperimentQueue,
    dispatch_experiment_collect,
    dispatch_experiment_submit,
)
from investigation.reading import Reading, build_candidate_reading_packet
from investigation.sessions import ModelSession
from investigation.state import (
    AttestationRecord,
    BudgetEntry,
    InvestigationState,
    attestation_is_positive,
    clamp_unit_interval,
    latest_attestation_for_hash,
    normalize_damage_scope,
    normalize_repairability,
)
from workspace import Workspace
```

(No `models.*` import is needed. `ModelSession` is imported for typing
only; there is no import cycle: `sessions`, `episodes`, `experiments`,
`actions`, `context`, `state`, `board`, `reading` do not import `loop_v3`
or `host`. The lazy in-function imports listed in §4.3 remain inside
method bodies.)

Type alias, directly after imports:

```python
# emit(event, payload, **extra) — extra carries LoopEvent kwargs such as
# ``outer_iteration``. Provided by the driver (run_v3 today; MCP later).
EmitFn = Callable[..., None]
```

### 3.3 Module-level constants and pure helpers (moved from loop_v3.py)

In this order, **verbatim** (bodies, docstrings, comments unchanged):

| Name | Source lines | Notes |
|---|---|---|
| `_branch_hash(workspace, branch)` | 81–82 | shared with loop_v3, which now imports it from host |
| `_active_branch(workspace, branch)` | 85–89 | ditto |
| `_resync_attestation_branch_on_rename(...)` | 237–277 | pure over args |
| `_episode_result_digest(kind, status, failure_reason, result)` | 298–369 | keeps its lazy `from investigation.state import attestation_is_positive` |
| `_truncate_text(text, limit)` | 372–374 | |
| `_fmt_unit(value)` | 377–380 | |
| `REPAIR_ACCEPTANCE_POLICY: Any = None` | 384–388 | module attribute incl. its comment block (M5.4 hook) |
| `_EVIDENCE_FAILURE_REASONS` | 391–400 | incl. comment block |
| `_classify_failure_reason(reason)` | 403–410 | |
| `_REPAIR_COMPOSITE_NAMES` | 413–416 | |
| `_extract_repair_evidence(tool_calls)` | 419–485 | |
| `_worker_rejected_targets(result)` | 488–497 | |
| `_EDIT_LABEL_IN_PROSE_RE` | 500–502 | |
| `_unbound_edit_claims(claims, edit_evidence)` | 505–526 | |
| `_winner_adjudication_summary(tool_calls, winner)` | 529–552 | |
| `_clamp_coherence(value)` | 876–897 | was a closure but touches nothing in scope; becomes a module function, body verbatim |
| `READ_ONLY_LEAD_TOOLS` | 779–789 | was the local `read_only_lead_tools` frozenset; becomes a module constant, same members, same order |

### 3.4 `class InvestigationHost`

Class docstring:

```python
class InvestigationHost:
    """Owns the v3 lead dispatch layer over (state, workspace, executor,
    queue) plus the run's budget accounting. One instance per run/driver.
    The driver must call ``set_available_tools`` before ``handle_tool``
    each turn (an empty set blocks every tool, matching the loop's
    per-turn tool-list gate)."""
```

Constructor — exact signature (all keyword-only):

```python
def __init__(
    self,
    *,
    state: InvestigationState,
    workspace: Workspace,
    executor: WorkspaceToolExecutor,
    queue: ExperimentQueue,
    emit: EmitFn,
    session: ModelSession,
    model_provider: Any = None,
    language: str,
    word_set: set[str],
    word_list: list[str],
    pattern_dict: dict[str, Any],
    episode_models: dict[str, str] | None = None,
    max_cost_usd: float | None = None,
    prior_budget: list[BudgetEntry] | None = None,
) -> None:
    if workspace is not state.workspace:
        raise ValueError(
            "InvestigationHost requires workspace is state.workspace"
        )
    self.state = state
    self.workspace = workspace
    self.executor = executor
    self.queue = queue
    self._emit = emit
    self._session = session
    self._model_provider = model_provider
    self.language = language
    self._word_set = word_set
    self._word_list = word_list
    self._pattern_dict = pattern_dict
    self._episode_models = episode_models
    self.max_cost_usd = max_cost_usd
    self._prior_budget: list[BudgetEntry] = list(prior_budget or [])
    # Mutable dispatch-layer state (was run_v3 closure variables):
    self.episode_budget: list[BudgetEntry] = []       # was L700
    self.episode_tool_calls: list[Any] = []           # was L774
    self._available_tools: set[str] = set()           # was current_lead_tool_names, L775
    self._read_call_cache: dict[
        tuple[str, str, tuple[tuple[str, str], ...]], str
    ] = {}                                            # was read_call_cache, L777
```

Rationale for parameters beyond the proposal's sketch `(state, workspace,
executor, queue, emit)`: the moved closures demonstrably also close over
`session` (budget math), `model_provider` (episode-model cloning, L866–874),
`language`/`word_set`/`word_list`/`pattern_dict` (every `run_episode` call,
L937–943 and L1159–1165), `episode_models` (L936, L1158), `max_cost_usd`
(episode cost guard + ceiling), and `prior_budget` (budget seam, L696).
They are injected explicitly rather than scraped off `executor` internals.
The `workspace is state.workspace` check can never fire from `run_v3`
(which always passes the same object) — it exists to stop a future driver
from silently forking the two.

Note the sketch in proposal §7.0 says `handle_tool(name, args)`; this spec
deliberately uses `handle_tool(tool_use, turn)` instead (§4.2 row 1)
because the `tool_use` **id** is load-bearing — it is recorded in
`ToolCall.tool_use_id` and used to synthesize the `:repair` / `:install`
sub-call ids inside `repair_transaction` (L1623, L1876). A name/args-only
surface would have to invent ids and change artifacts. The MCP phase wraps
this with its own id scheme; not this slice's concern.

### 3.5 Methods

Methods appear in the class in the order of §4.2 (same relative order as
the closures appear today, for diff-reviewability).

## 4. Migration table

### 4.1 Mechanical rename map (the ONLY permitted body edits)

Applied uniformly to every moved body:

| Closure-scope name (today) | In host methods |
|---|---|
| `state` | `self.state` |
| `workspace` | `self.workspace` |
| `executor` | `self.executor` |
| `queue` | `self.queue` |
| `language` | `self.language` |
| `word_set` / `word_list` / `pattern_dict` | `self._word_set` / `self._word_list` / `self._pattern_dict` |
| `episode_models` | `self._episode_models` |
| `max_cost_usd` | `self.max_cost_usd` |
| `model_provider` | `self._model_provider` |
| `session` | `self._session` |
| `emit(` | `self._emit(` |
| `prior_budget` | `self._prior_budget` |
| `episode_budget` | `self.episode_budget` |
| `episode_tool_calls` | `self.episode_tool_calls` |
| `read_call_cache` | `self._read_call_cache` |
| `current_lead_tool_names` | `self._available_tools` |
| `read_only_lead_tools` | `READ_ONLY_LEAD_TOOLS` (module const) |
| `_committed_cost()` | `self.committed_cost()` |
| `_information_digest()` | (host-internal callers: none) |
| calls to sibling closures `_dispatch_*`, `_settle_repair_outcome`, `_probe_snapshot_scores`, `_snapshot_content_hash`, `_record_dispatch_result`, `_provider_for_model`, `_lead_read_cache_key`, `_episode_event_forwarder` | `self._<same name>(...)` |
| calls to `_clamp_coherence`, `_episode_result_digest`, `_branch_hash`, `_active_branch`, `_resync_attestation_branch_on_rename`, `_extract_repair_evidence`, `_worker_rejected_targets`, `_unbound_edit_claims`, `_winner_adjudication_summary`, `_classify_failure_reason` | unchanged (module-level in host.py) |

Nothing else in any body changes. If the implementer finds a line that
seems to need any other edit, that is a spec bug: STOP and report it
rather than improvising.

### 4.2 Closure → method table (exact signatures)

| # | Today (lines) | Host method (exact signature) | Notes |
|---|---|---|---|
| 1 | `_dispatch_tool(tu, turn)` (1942–2108) | `def handle_tool(self, tu: dict[str, Any], turn: int) -> str:` | The router. PUBLIC. Renamed per proposal §7.0; body verbatim incl. the `lead_tool_not_available` block, repeated-call signature counting, `episode_kind_not_available` gate, duplicate-read suppression, the `repair_transaction` workflow-phase gate + its `try/except` (KeyboardInterrupt re-raised, other exceptions → structured `transaction_error`), the experiment-tool arm (F-4 exception wrapper), the composite arm, the final `executor.execute` arm, and the `state.model_variant` mirror. |
| 2 | `_dispatch_verify_run(args, turn)` (899–1051) | `def _dispatch_verify_run(self, args: dict[str, Any], turn: int) -> str:` | Attestation write + repair-agenda seeding stay exactly here. |
| 3 | `_dispatch_episode_run(tu, turn)` (1053–1275) | `def _dispatch_episode_run(self, tu: dict[str, Any], turn: int) -> str:` | Keeps the internal `kind == "verify"` route to method 2; keeps lazy state/episodes imports. |
| 4 | `_dispatch_episode_install(tu, turn)` (1277–1392) | `def _dispatch_episode_install(self, tu: dict[str, Any], turn: int) -> str:` | Calls module-level `_resync_attestation_branch_on_rename`. |
| 5 | `_snapshot_content_hash(snapshot)` (1394–1404) | `def _snapshot_content_hash(self, snapshot: dict[str, Any]) -> str:` | Uses `self.workspace.cipher_text` / `.plaintext_alphabet` for the scratch Workspace. |
| 6 | `_settle_repair_outcome(...)` (1406–1454) | `def _settle_repair_outcome(self, *, record: dict[str, Any], entry_args: tuple[str, str, str], changed_hashes: list[str], turn: int) -> dict[str, Any]:` | |
| 7 | `_probe_snapshot_scores(...)` (1456–1474) | `def _probe_snapshot_scores(self, snapshot: dict[str, Any], transaction_id: str) -> tuple[str, dict[str, float | None]]:` | |
| 8 | `_dispatch_repair_transaction(tu, turn)` (1476–1940) | `def _dispatch_repair_transaction(self, tu: dict[str, Any], turn: int) -> str:` | The internal API-billed repair episode call (`self._dispatch_episode_run({... "kind": "repair" ...}, turn)`, L1622–1632) and the internal install call (L1875–1883) are PRESERVED, synthesized `tu` dicts and `:repair`/`:install` id suffixes included. Inner closures `_acceptance()` and `_fail()` remain inner closures of the method, verbatim. The `assert REPAIR_ACCEPTANCE_POLICY is None` (L1864) now reads host.py's module attribute. |
| 9 | `_record_dispatch_result(...)` (852–864) | `def _record_dispatch_result(self, *, name: str, tu: dict[str, Any], turn: int, payload: dict[str, Any]) -> str:` | Appends to `self.executor.call_log`. |
| 10 | `_provider_for_model(model_id)` (866–874) | `def _provider_for_model(self, model_id: str | None) -> Any:` | |
| 11 | `_lead_read_cache_key(name, args)` (791–810) | `def _lead_read_cache_key(self, name: str, args: dict[str, Any]) -> tuple[str, str, tuple[tuple[str, str], ...]]:` | |
| 12 | `_information_digest()` (812–850) | `def information_digest(self) -> str:` | PUBLIC (run_v3 calls it per turn at L2384). Only rename; body verbatim. |
| 13 | `_episode_event_forwarder(turn)` (689–694) | `def _episode_event_forwarder(self, turn: int) -> Any:` | Returns the same `_forward` closure over `self._emit`. |
| 14 | `sync_budget()` (702–711) | `def sync_budget(self) -> None:` — rebuilds ONLY `self.state.budget_ledger = self._prior_budget + self.episode_budget + list(self._session.usage_entries())` | The four `artifact.*` mirror lines move to the loop-side wrapper (§5 site S2). The L697–699 comment ("Episode spend accumulates here across turns…") rides with `self.episode_budget` in `__init__`, not with this method. |
| 15 | `_committed_cost()` (713–722) | `def committed_cost(self) -> float:` | PUBLIC (run_v3 uses it in the ceiling event/message). Body + comment verbatim. |
| 16 | `_cost_ceiling_reached()` (724–726) | `def cost_ceiling_reached(self) -> bool:` | PUBLIC. `self.max_cost_usd is not None and self.committed_cost() >= self.max_cost_usd`. |
| 17 | (new, trivial) | `def set_available_tools(self, names: set[str]) -> None:` — `self._available_tools = set(names)` | Replaces the per-turn `current_lead_tool_names = {...}` assignment (L2175). |

### 4.3 Lazy in-body imports that must move verbatim (checklist)

- `_dispatch_episode_run`: `from investigation.state import (attestation_key,
  latest_attestation_for_hash, saturation_key)` (reading gate);
  `from investigation.episodes import SEARCH_EPISODE_TOOL_NAMES` (bad-spec
  error payload); `from investigation.state import
  get_or_create_saturation_entry` (reading saturation).
- `_dispatch_episode_install`: `from investigation.state import
  _restore_branch_into`.
- `_snapshot_content_hash`, `_probe_snapshot_scores`: `from
  investigation.state import _restore_branch_into`.
- `_settle_repair_outcome`: `from investigation.state import
  get_or_create_saturation_entry`.
- `_dispatch_repair_transaction`: `from investigation.reading import
  interpretation_digest as _interp_digest`; `from investigation.state
  import (attestation_key, get_or_create_saturation_entry,
  latest_attestation_for_hash, pair_digest, saturation_key)`.
- `_episode_result_digest` (module fn): `from investigation.state import
  attestation_is_positive`.

## 5. `run_v3` rewiring (site-by-site)

`run_v3` keeps its signature, docstring, and every retained line
unchanged. The following are the ONLY changes, keyed to current lines:

- **S1 (delete 700–726):** remove `episode_budget`, `sync_budget`,
  `_committed_cost`, `_cost_ceiling_reached` closures. Keep line 696
  (`prior_budget = list(state.budget_ledger)`) exactly where it is — its
  capture timing is load-bearing (before any ledger rebuild).
- **S2 (insert after 770, the `queue = ...` line):** construct the host and
  the thin budget wrapper:

  ```python
  host = InvestigationHost(
      state=state, workspace=workspace, executor=executor, queue=queue,
      emit=emit, session=session, model_provider=model_provider,
      language=language, word_set=word_set, word_list=word_list,
      pattern_dict=pattern_dict, episode_models=episode_models,
      max_cost_usd=max_cost_usd, prior_budget=prior_budget,
  )

  def sync_budget() -> None:
      # Host rebuilds state.budget_ledger; the artifact mirror stays
      # loop-side (the host is artifact-free by design).
      host.sync_budget()
      artifact.total_input_tokens = sum(
          e.input_tokens for e in state.budget_ledger
      )
      artifact.total_output_tokens = sum(
          e.output_tokens for e in state.budget_ledger
      )
      artifact.total_cache_read_tokens = sum(
          e.cache_read_tokens for e in state.budget_ledger
      )
      artifact.estimated_cost_usd = state.total_cost()
  ```

  The three existing call sites (`sync_budget()` at 2236, 2401, 2475) are
  untouched. The math is identical to today's L702–711 because
  `host.sync_budget()` writes the same ledger the mirror sums.
- **S3 (delete 774–777):** `episode_tool_calls`, `current_lead_tool_names`,
  `current_episode_kinds`, `read_call_cache` closure variables. (`current_
  episode_kinds` survives only as a plain local inside the turn loop —
  see S7.)
- **S4 (delete 779–810, 812–850, 852–864, 866–874, 876–897):** the
  read-only set, `_lead_read_cache_key`, `_information_digest`,
  `_record_dispatch_result`, `_provider_for_model`, `_clamp_coherence`
  (all now in host.py).
- **S5 (delete 899–2108):** the five dispatchers + settle/probe/snapshot
  helpers + `_dispatch_tool` (all now host methods).
- **S6 (2145, 2150, 2155):** `_cost_ceiling_reached()` →
  `host.cost_ceiling_reached()`; both `_committed_cost()` occurrences (the
  f-string in `artifact.error_message` and the `cost_ceiling_reached`
  event payload) → `host.committed_cost()`.
- **S7 (2170):** `current_episode_kinds = set(allowed_episode_kinds(state,
  executor))` stays, as a plain loop-local (it feeds only the tool-schema
  build on 2171–2173).
- **S8 (2175):** `current_lead_tool_names = {definition["name"] for
  definition in tools}` → `host.set_available_tools({definition["name"]
  for definition in tools})`.
- **S9 (2307):** `result = _dispatch_tool(tu, turn)` →
  `result = host.handle_tool(tu, turn)`. The surrounding
  KeyboardInterrupt handling (R5 pairing) is untouched — `handle_tool`
  re-raises KeyboardInterrupt exactly as `_dispatch_tool` does today.
- **S10 (2384):** `information_digest = _information_digest()` →
  `information_digest = host.information_digest()`.
- **S11 (2578):** `artifact.tool_calls = list(executor.call_log) +
  list(episode_tool_calls)` → `... + list(host.episode_tool_calls)`.

Everything else in `run_v3` — session/provider setup, language resources,
state/workspace/executor/artifact construction, `emit` (677–687), the
fingerprint/preflight block (728–761), the whole turn loop (context build,
workflow hints, `workflow_state_changed`, 429 retry, truncation retry,
no-tool nudge, tool_result pairing, post-terminate skip, exchange
recording, workspace snapshot + budget events, termination handling), the
fallback/honest-termination block (2477–2554), and finalize (2556–2622) —
stays byte-identical.

## 6. Final shape of `loop_v3.py`

### 6.1 Retained members

- Module docstring (append one sentence: "The dispatch layer lives in
  ``investigation.host`` (host-extraction slice); ``run_v3`` is the
  turn-loop driver over an ``InvestigationHost``.").
- `_fresh_compare_winner` (91–125), `_select_v3_fallback` (128–191),
  `_compute_branch_roles` (194–234), `_tool_status` (280–295) — verbatim,
  still module-level in loop_v3 (they are termination/fallback logic, not
  dispatch; tests import the first three from `investigation.loop_v3` and
  must keep working WITHOUT edits).
- `run_v3` per §5.

### 6.2 Import list for the revised loop_v3.py (exact)

Keep: `json`, `time`, `uuid`; from `agent.loop_shared`:
`_best_branch_for_auto_declare`, `_branch_snapshot_for`,
`_candidate_content_hash`, `_decoded_text_for_panel`,
`_hypothesis_cards_for_artifact`, `_install_automated_preflight_branch`,
`_tool_result_summary`, `_workspace_snapshot_payload`; from
`agent.model_provider`: `ModelProviderError`, `call_with_rate_limit_retry`,
`ensure_model_provider`, `_collect_assistant_blocks`; from
`agent.tools_v2`: `AttestationPolicy`, `WorkspaceToolExecutor`; from
`investigation.episodes`: `EPISODE_KINDS`, `v3_lead_tool_definitions`;
from `investigation.experiments`: `EXPERIMENT_TOOL_DEFINITIONS`,
`ExperimentQueue`; from `artifact.schema`: `LoopEvent`, `RunArtifact`,
`SolutionDeclaration`; from `analysis`: `cipher_id as cipher_id_analysis`,
`dictionary`, `pattern`; from `investigation.context`:
`allowed_episode_kinds`, `build_lead_context`, `build_v3_system_prompt`,
`workflow_state`, `workflow_hint_candidates`; from
`investigation.sessions`: `ModelSession`, `session_factory`; from
`investigation.state`: `BudgetEntry`, `InvestigationState`,
`attestation_is_positive`, `latest_attestation_for_hash`; from
`models.cipher_text`: `CipherText`; from `workspace`: `Workspace`.

Add: `from investigation.host import InvestigationHost, _active_branch,
_branch_hash`.

Drop (moved to host or now unused): `copy`, `hashlib`, `re`,
`DECODED_TEXT_RENDERER_ID`, `CARD_MIRROR_KEYS`, `COMPOSITE_TOOL_NAMES`,
`execute_composite`, `EpisodeSpec`, `run_episode`, `EXPERIMENT_TOOL_NAMES`,
`dispatch_experiment_collect`, `dispatch_experiment_submit`, `Reading`,
`build_candidate_reading_packet`, `ToolCall`, `AttestationRecord`,
`clamp_unit_interval`, `normalize_damage_scope`, `normalize_repairability`.

(Verify with a linter pass that nothing retained still needs a dropped
import; `_select_v3_fallback` keeps needing `attestation_is_positive` +
`latest_attestation_for_hash`, which is why they stay.)

**No compatibility re-exports.** loop_v3 does NOT re-export moved names
(e.g. `_unbound_edit_claims`); the three test imports are edited instead
(§7.2). A stale `loop_v3._episode_result_digest` reference should fail
loudly, not silently resolve to a second copy.

## 7. Test impact

### 7.1 Audit (complete, verified by repo-wide grep at spec time)

Tests reaching `loop_v3` at all:

| File | What it uses | Impact |
|---|---|---|
| `tests/test_loop_v3.py` | imports `run_v3`, `_fresh_compare_winner` (stays), `_unbound_edit_claims` (**moves**); local imports `_compute_branch_roles` (:994, stays), `_select_v3_fallback` (:1661, stays); monkeypatches `loop_v3_mod.dispatch_experiment_submit` (:1439–1443) (**call site moves to host**) | 2 edits |
| `tests/test_m6_m5_note_fixes.py` | `from investigation.loop_v3 import _resync_attestation_branch_on_rename, run_v3` (:23) (**helper moves**) | 1 edit |
| `tests/test_cli_observability.py` | `from investigation.loop_v3 import _episode_result_digest, run_v3` (:33) (**helper moves**) | 1 edit |
| `tests/test_benchmark.py` | monkeypatches `loop_v3.run_v3` (:569); works because `runner_v2` imports `run_v3` inside the method at call time; `run_v3` stays in loop_v3 | none |
| `tests/test_v3_sequence_b.py`, `tests/test_episodes.py`, `tests/test_experiments.py`, `tests/test_ground_truth_firewall.py` | import `run_v3` only | none |
| `tests/support/scripted_v3.py` | no loop_v3 import | none |

No test references `_dispatch_*`, `_settle_repair_outcome`,
`_information_digest`, `_lead_read_cache_key`, `_clamp_coherence`,
`REPAIR_ACCEPTANCE_POLICY`, `_branch_hash`, or `_active_branch` by
attribute access on the module (verified). No `scripts/*` file imports
loop_v3 (comment mentions only). Production imports (`src/cli.py:1004`,
`src/benchmark/runner_v2.py:224`) import `run_v3` only.

### 7.2 The four permitted test edits (exact)

1. `tests/test_loop_v3.py` top-of-file import block: remove
   `_unbound_edit_claims` from the `investigation.loop_v3` import; add
   `from investigation.host import _unbound_edit_claims`.
   *Justification: pure import-path change; the helper moved with the
   repair-acceptance cluster it serves.*
2. `tests/test_loop_v3.py` :1439–1443 (inside
   `test_s2_experiment_submit_records_pending_pointer_when_exhausted`,
   defined at :1429):

   ```python
   import investigation.loop_v3 as loop_v3_mod
   monkeypatch.setattr(
       loop_v3_mod, "dispatch_experiment_submit", ...
   ```

   becomes

   ```python
   import investigation.host as host_mod
   monkeypatch.setattr(
       host_mod, "dispatch_experiment_submit", ...
   ```

   *Justification: monkeypatch-target change only; the experiment-tool arm
   of the router (which binds this name at module level) moved to
   `investigation.host`. The lambda and every assertion are unchanged.*
3. `tests/test_m6_m5_note_fixes.py` :23 →
   `from investigation.host import _resync_attestation_branch_on_rename`
   and `from investigation.loop_v3 import run_v3`.
   *Justification: import path only.*
4. `tests/test_cli_observability.py` :33 →
   `from investigation.host import _episode_result_digest` and
   `from investigation.loop_v3 import run_v3`.
   *Justification: import path only.*

Any OTHER test edit — however small — is out of contract for this slice
and must go back to the spec author.

### 7.3 New tests

None required, deliberately: the slice's claim is behavior preservation,
which the existing 1,710 tests + §9.3 parity check assert. Do not add
host-only unit tests in this slice (they would pin the extraction shape
before the MCP phase settles the driver contract).

## 8. Documentation edits

1. `src/investigation/__init__.py` module docstring: add
   `- ``host``     — ``InvestigationHost``: the loop-independent lead
   dispatch layer shared by run_v3 (and, later, the MCP server).`
   alongside the existing `loop_v3` line.
2. `CLAUDE.md`, Key Files, under `src/investigation/`: insert
   `host.py               — InvestigationHost: loop-independent lead dispatch
   layer (episodes, installs, repair transactions, budget seam) extracted
   from loop_v3` immediately after the `loop_v3.py` line, and reword the
   `loop_v3.py` line's description to `run_v3 lead turn loop; dispatch
   lives in host.py (M1)`.
3. No TOOLS.md change (no tool surface change).

## 9. Acceptance

### 9.1 Full suite

`PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` — all 1,710 tests
pass. (Same count as pre-change; this slice adds none and removes none.)

### 9.2 Sequence-B replay

`PYTHONPATH=src .venv/bin/python -m pytest tests/test_v3_sequence_b.py -q`
— green. (Subset of 9.1; called out because the proposal names it a pin.)

### 9.3 Scripted-run artifact byte-parity (zero diff)

Byte-level proof that the event stream, tool calls, attestations, budget
ledger, and serialized state are unchanged. Two scripted scenarios are
captured with `uuid.uuid4` and `time.time` frozen, so the pre- and
post-change artifacts must be **byte-identical** (plain `diff`, no
normalization step).

**Step 0 — write the capture script** to the session scratchpad (NOT into
the repo; it must not be committed) as `capture_artifact.py`, exactly:

```python
"""Scripted-run artifact capture for the host-extraction byte-parity check.

Usage (from the decipher repo root, same venv, same env both times):
    PYTHONPATH=src:. .venv/bin/python <scratchpad>/capture_artifact.py out.json

uuid4/time.time are frozen BEFORE any run so two captures of identical
code are byte-identical; a diff between the pre- and post-extraction
captures is therefore a pure behavior diff.
"""
import itertools
import json
import sys

from _pytest.monkeypatch import MonkeyPatch


def _freeze(mp):
    import time as time_mod
    import uuid as uuid_mod

    ticks = itertools.count()
    mp.setattr(time_mod, "time", lambda: 1_000_000.0 + 0.001 * next(ticks))
    ids = itertools.count(1)

    class _FakeUUID:
        def __init__(self, n):
            # First 12 hex chars must be unique: every consumer slices
            # uuid4().hex[:12].
            self.hex = f"{n:012x}" + "0" * 20

        def __str__(self):
            return self.hex

    mp.setattr(uuid_mod, "uuid4", lambda: _FakeUUID(next(ids)))


def _scenario_sequence_b(mp):
    import tests.test_v3_sequence_b as seq

    art, _lead, _content_hash, _rid = seq._run_sequence_b(mp)
    return art


def _scenario_solve(mp):
    from agent.model_provider import TextBlock, ToolUseBlock
    from investigation import sessions as sessions_mod
    import tests.test_loop_v3 as tl
    from tests.support.scripted_v3 import ScriptedSession, VerifyWorkerFake

    ct, alpha = tl._caesar_cipher("THE DOG")
    scripts = [
        [ToolUseBlock(id="t1", name="decode_show", input={"branch": "main"})],
        # Identical read against unchanged content -> duplicate-read
        # suppression + no_new_information paths.
        [ToolUseBlock(id="t2", name="decode_show", input={"branch": "main"})],
        [ToolUseBlock(id="tv", name="episode_run",
                      input={"kind": "verify", "goal": "verify main",
                             "branches": ["main"]})],
        [TextBlock(text="Reads as English."),
         ToolUseBlock(id="t3", name="meta_declare_solution",
                      input={"branch": "main",
                             "rationale": "Caesar shift 3 recovered.",
                             "self_confidence": 0.95})],
    ]
    sessions_mod.register_session_builder("episode:verify", VerifyWorkerFake)
    try:
        from investigation.loop_v3 import run_v3

        art = run_v3(
            ct, session=ScriptedSession(scripts), language="en",
            max_iterations=10, cipher_id="parity_solve",
            resume_state=tl._seeded_caesar_state(ct, alpha, with_alt=True),
        )
    finally:
        sessions_mod._SESSION_BUILDERS.pop("episode:verify", None)
    return art


def main(out_path):
    payload = {}
    for name, fn in [
        ("sequence_b", _scenario_sequence_b),
        ("solve", _scenario_solve),
    ]:
        mp = MonkeyPatch()
        try:
            _freeze(mp)
            payload[name] = fn(mp).to_dict()
        finally:
            mp.undo()
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
```

Coverage of the two scenarios: Sequence B exercises negative verify +
agenda seeding, reading episodes, two failed repair transactions →
saturation/exhaustion, experiment submit/collect (synchronous queue),
workflow transitions, and honest termination; the solve scenario exercises
the read cache/duplicate suppression, `repeated_call`, positive verify +
attestation write, the declaration gate accept path, and the budget seam
on the solved terminal.

**Step 1 — baseline, BEFORE touching any source:** from the clean tree at
the starting commit, run the capture TWICE and diff the two outputs to
prove harness determinism; keep one as `baseline.json` in the scratchpad:

```bash
cd ~/Dropbox/src2/decipher
PYTHONPATH=src:. .venv/bin/python <scratchpad>/capture_artifact.py <scratchpad>/baseline.json
PYTHONPATH=src:. .venv/bin/python <scratchpad>/capture_artifact.py <scratchpad>/baseline2.json
diff <scratchpad>/baseline.json <scratchpad>/baseline2.json   # MUST be empty
```

If this determinism diff is non-empty, STOP and report (do not proceed to
the refactor on a nondeterministic harness).

**Step 2 — after the extraction (suite already green):**

```bash
PYTHONPATH=src:. .venv/bin/python <scratchpad>/capture_artifact.py <scratchpad>/candidate.json
diff <scratchpad>/baseline.json <scratchpad>/candidate.json   # MUST be empty
```

Zero diff is the acceptance bar. Any diff, however cosmetic-looking, is a
behavior change: STOP and fix (or report if the fix is unclear).

*Fallback if the baseline was not captured before editing:* create a
pristine checkout with `git worktree add <scratchpad>/base_tree <start-commit>`,
run Step 1 inside it (`cd <scratchpad>/base_tree` with `PYTHONPATH=src:.`),
then `git worktree remove` it afterwards.

### 9.4 Sanity greps (post-change)

- `grep -n "def _dispatch" src/investigation/loop_v3.py` → no matches.
- `grep -rn "from investigation.host import" src/` → exactly one file
  (`loop_v3.py`).
- `grep -c "def " src/investigation/host.py` roughly matches the §4.2/§3.3
  inventory (no dropped or invented functions).
- `git diff --stat` touches ONLY the files in §2.

### 9.5 Landing

Per repo convention: this phase lands as ONE commit after adversarial
review passes ("MCP host-extraction slice: extract InvestigationHost from
run_v3 (zero behavior change)"). The working tree is currently clean, so
no WIP-checkpoint commit is needed first. The optional ~$1.3 paid parity
smoke from proposal §7.1(5) is a post-land, pre-Phase-A step owned by the
main session, not part of this slice's acceptance.

## 10. Out of scope (do NOT build here)

- The MCP server, `decipher mcp-serve`, capsule launcher, registry,
  revision store (Phases A–C).
- The client-compiled `repair_transaction` variant (proposal §3.5): here,
  `repair_transaction` keeps running its internal API-billed repair
  episode, unchanged.
- Any advisory-mode softening of `_dispatch_tool` policies (§3.3 ledger
  work — Phase 0 owns that; this slice moves the hard blocks verbatim).
- Any change to episode internals, experiment queue, context building,
  prompts, tool schemas, or the v2 loop.
- Host-only unit tests, docstring "improvements" to moved bodies, dead-code
  cleanups spotted along the way (report them; do not do them).

## 11. Reviewer checklist (for the adversarial review pass)

1. Diff `host.py` method bodies against the deleted loop_v3 closures:
   verbatim modulo the §4.1 rename map — flag ANY other token change.
2. Confirm `prior_budget` capture site (old L696) is unchanged and feeds
   the host ctor; confirm the loop-side `sync_budget` wrapper reproduces
   old L702–711 exactly (ledger first, then the four artifact mirrors).
3. Confirm `handle_tool` re-raises KeyboardInterrupt (both the
   repair-transaction arm and the experiment arm keep their explicit
   `except KeyboardInterrupt: raise`).
4. Confirm the verify dispatcher still writes the AttestationRecord and
   agenda items host-side (A1: workers never write state), and
   `_resync_attestation_branch_on_rename` is still called from the install
   path.
5. Confirm `state.model_variant` mirroring still happens ONLY on the
   `executor.execute` arm (not on episode/experiment/composite arms).
6. Confirm the four test edits match §7.2 exactly and nothing else in
   tests changed (`git diff tests/` shows only those hunks).
7. Confirm §9.3 ran and both diffs (determinism + parity) were empty.
8. Confirm no import cycle: `python -c "import investigation.host,
   investigation.loop_v3"` under `PYTHONPATH=src`.
