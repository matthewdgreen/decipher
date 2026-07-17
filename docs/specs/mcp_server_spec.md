# MCP Server Implementation Spec — Phases A–C

Status: IMPLEMENTATION SPEC, ready for a coding agent. Written 2026-07-17
against HEAD `5f269ac` (post host-extraction). Author: Fable (spec role).

Authority (in precedence order where they overlap):

1. `docs/mcp_dual_harness_proposal.md` — §3 architecture, §3.5 v1 tool
   surface (with §7.2 v1.1 deferrals), §4 prompting/self-briefing, §8.1
   zero-effort onboarding (commit `1171cb7`), §9.1 server operations.
2. `docs/mcp_policy_provenance_ledger.md` — the 66-control v1 forms. The
   conformance map in Part 9 of this spec is the binding translation of that
   ledger for the server; nothing here may soften a control the ledger keeps
   hard, and the 7 advisory softenings + 2 divergences are implemented
   exactly as advisory text / contract splits, never as new hard gates.
3. The landed host: `src/investigation/host.py` (`InvestigationHost`),
   `state.py`, `context.py`, `episodes.py`, `experiments.py`, `actions.py`,
   `agent/tools_v2.py` (`AttestationPolicy`, `WorkspaceToolExecutor`).

Baseline: **1708 passed / 2 skipped** at `5f269ac`
(`PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`). Every part of this
spec is additive except the one behavior-preserving host refactor in Part 8.1;
the full suite must remain green after every part lands.

Hard rules (restating the proposal §7):

- **v3 must not break.** `src/investigation/loop_v3.py` is NOT edited by this
  work (imports from it are fine). `host.py` is edited ONLY as Part 8.1
  specifies (mechanical extraction, zero behavior change, pinned by the
  existing suite). `state.py`, `context.py`, `episodes.py`, `experiments.py`,
  `actions.py`, `tools_v2.py` are NOT edited at all.
- **No code duplication.** The server imports the host, the context
  renderers, the state helpers, and the composite dispatcher. Where this spec
  says "replicate", it names the exact helper functions to call, not logic to
  re-derive.
- **No paths through MCP.** No tool accepts a filesystem path, benchmark id,
  or benchmark split (proposal §3.5, §4.4, §9.1). `investigation_start`
  takes inline ciphertext only. Benchmark-backed/capsule investigations are
  Phase-D scope and are NOT part of this spec.
- **Firewall surface.** The server never constructs a
  `ScopedBenchmarkContext`; `benchmark_context=None` everywhere. The v2
  `inspect_*`/`list_*` context tools are not exposed.

Non-goals (explicitly out of scope for Phases A–C):

- The sanitized run-capsule launcher, container isolation, and the §6
  experiment (Phase D).
- `candidate_compare_signals` and the richer `comparison_record` integration
  extras (deferred to v1.1 per proposal §7.2).
- A generic server-side `episode_run` cognitive loop (§3.5 "deliberately NOT
  exposed").
- A standalone daemon process; v1 uses the per-investigation writer lease
  (Part 3.4) to satisfy the §9.1 single-owner rule.
- The full §8.1.6 acceptance matrix (Phase C completion gate, run later);
  this spec ships the files and the automated subset in Part 11.

---

## Part 0 — Architecture summary

One new package, `src/mcp_server/`, exposes the investigation surface as an
MCP stdio server. Per client process:

```
Claude Code / Codex  ──stdio JSON-RPC──►  mcp_server.protocol
                                             │ tools/call
                                             ▼
                                    mcp_server.server (DecipherMCPServer)
                                             │ per-investigation
                                             ▼
                          mcp_server.registry (on-disk store, revisions, lease)
                                             │
                                             ▼
                          mcp_server.runtime (InvestigationRuntime)
                
                   state = InvestigationState.from_artifact_dict(...)
                   executor = WorkspaceToolExecutor(..., AttestationPolicy)
                   host = InvestigationHost(...)      ← LANDED, shared with v3
                   queue = ExperimentQueue()
```

The MCP tools map onto `host.handle_tool` (for the tools that already exist
on the v3 surface), onto `host._dispatch_verify_run` (verification), onto the
Part-8 repair glue, and onto new thin read/record layers. All state persists
through `InvestigationState.to_artifact_dict()` / `from_artifact_dict()` —
reused as-is, no schema change.

---

## Part 1 — Package layout and CLI subcommand

### 1.1 New files

```
src/mcp_server/__init__.py     — empty docstring module
src/mcp_server/__main__.py     — `python -m mcp_server` entry point
src/mcp_server/protocol.py     — MCP stdio JSON-RPC framing + method dispatch
src/mcp_server/server.py       — DecipherMCPServer: tool table, routing, envelope
src/mcp_server/registry.py     — InvestigationRegistry: disk store, revisions, lease
src/mcp_server/runtime.py      — InvestigationRuntime: state/executor/host/queue
src/mcp_server/status.py       — build_status_brief + advisory rendering
src/mcp_server/tools.py        — MCP_TOOL_DEFINITIONS (23 schemas) + arg validation
src/mcp_server/intake.py       — inline-ciphertext parsing for investigation_start
src/mcp_server/verify.py       — request_independent_verification glue
src/mcp_server/repair.py       — repair_hypotheses_test + repair_transaction glue
```

`src/` is already the package root (`[tool.setuptools.packages.find] where =
["src"]`), so `mcp_server` installs with the existing `pip install -e .` and
imports resolve under `PYTHONPATH=src` exactly like `investigation`.

### 1.2 CLI subcommand

In `src/cli.py`:

- Add `cmd_mcp_serve(args)`:

  ```python
  def cmd_mcp_serve(args: argparse.Namespace) -> None:
      from mcp_server.server import serve_stdio
      serve_stdio(
          registry_dir=args.registry_dir,
          verify_provider=args.verify_provider,
          verify_model=args.verify_model,
          max_cost_usd=args.max_cost_usd,
      )
  ```

- Register the subparser (after `diagnose`, same style):

  ```python
  mcp = subparsers.add_parser(
      "mcp-serve",
      help="Run the Decipher MCP stdio server (tools for Claude Code / Codex)",
  )
  mcp.add_argument("--registry-dir", default=None,
      help="Investigation registry directory "
           "(default: $DECIPHER_MCP_REGISTRY or ~/.config/decipher/investigations)")
  mcp.add_argument("--verify-provider", default="auto",
      choices=["auto", "anthropic", "openai", "gemini", "openrouter", "ollama", "none"],
      help="Provider for server-side verify episodes (default: auto-detect; "
           "'none' forces keyless degradation)")
  mcp.add_argument("--verify-model", default=None,
      help="Model id for verify episodes (default: the provider's default model)")
  mcp.add_argument("--max-cost-usd", type=float, default=5.0,
      help="Per-investigation paid ceiling for server-side verify spend (BUD-1)")
  ```

- Wire it in `main()`'s command dispatch exactly like the other subcommands.

`src/mcp_server/__main__.py` parses the same four flags with `argparse` and
calls the same `serve_stdio` — this gives tests and bare clones a stable
`python -m mcp_server` entry that does not require the console script.

Logging: the server writes **nothing to stdout except protocol frames**.
Diagnostics go to stderr via `print(..., file=sys.stderr)`.

---

## Part 2 — MCP protocol layer (`protocol.py`)

Pure stdlib (`json`, `sys`, `typing`). **No new dependency.** Justification:
the tools-only MCP subset is three request methods plus notifications over
newline-delimited JSON-RPC; an SDK would add a dependency for ~150 lines and
complicate the fresh-clone bootstrap.

### 2.1 Framing

MCP stdio transport: one JSON-RPC 2.0 message per line, UTF-8,
newline-delimited (no Content-Length headers). Implementation:

```python
def serve_loop(handler: "DecipherMCPServer",
               stdin=sys.stdin, stdout=sys.stdout) -> None
```

- Read `stdin` line by line (`for line in stdin`). Skip blank lines. EOF →
  clean return (client closed; process exits 0).
- `json.loads(line)` failure → write JSON-RPC error `-32700` with
  `id: null`, continue.
- A parsed dict without `"method"` → error `-32600` (use the message's `id`
  if present, else null).
- Messages **with an `id`** are requests → dispatch and write exactly one
  response line. Messages **without an `id`** are notifications → dispatch,
  write nothing (including for unknown notification methods).
- Every response is written as `json.dumps(obj) + "\n"` followed by
  `stdout.flush()`.
- Any exception escaping a handler → JSON-RPC error `-32603` with
  `"message": "internal error"` and the traceback printed to stderr. The
  loop continues (one bad call must not kill the server).

### 2.2 Methods

| method | behavior |
|---|---|
| `initialize` | Return `{"protocolVersion": v, "capabilities": {"tools": {"listChanged": false}}, "serverInfo": {"name": "decipher", "version": "0.1.0"}}`. `v` = the client's requested `params.protocolVersion` if it is in `SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")`, else `"2025-06-18"`. Record `params.clientInfo.name` (string, default `"unknown"`) on the server object as `client_name`. |
| `notifications/initialized` | No-op notification. |
| `ping` | Return `{}`. |
| `tools/list` | Return `{"tools": [...]}` from `mcp_server.tools.MCP_TOOL_DEFINITIONS`, each rendered as `{"name", "description", "inputSchema"}` (note camelCase `inputSchema`; the definitions store `input_schema` and the protocol layer renames). No pagination (23 tools). |
| `tools/call` | `params = {"name", "arguments"}`. Route to `DecipherMCPServer.call_tool(name, arguments)`. Result: `{"content": [{"type": "text", "text": <json string>}], "isError": <bool>}`. |
| anything else with an id | JSON-RPC error `-32601` method not found. |

`isError: true` is used ONLY for: unknown tool name, and internal exceptions
escaping a tool handler (message `"internal error running <tool>"`, traceback
to stderr). Every domain-level outcome — blocked gates, revision conflicts,
validation errors, keyless degradation — is `isError: false` with a
structured JSON body, exactly like v3 tool results are strings the model
reads. This keeps gate behavior client-visible data, not protocol failures.

---

## Part 3 — Investigation registry (`registry.py`)

### 3.1 Location (fixes proposal §9.1 "storage and retention")

Default root: `~/.config/decipher/investigations/` (sibling of the existing
`~/.config/decipher/openrouter_pricing.json`). Override precedence:
`--registry-dir` flag → `DECIPHER_MCP_REGISTRY` env → default. The directory
is created with `parents=True, exist_ok=True` on first use.

Retention: investigations persist until the user deletes their directory;
the server never garbage-collects (v1). Each investigation directory is
self-contained and auditable.

Per-investigation layout, `<root>/<investigation_id>/`:

```
investigation.json   — authoritative document (single atomic file, §3.3)
meta.json            — denormalized listing stub (best-effort copy of meta)
events.jsonl         — append-only emit stream (observability only)
lease.lock           — writer-lease flock file (§3.4)
```

### 3.2 Document format

`investigation.json`:

```json
{
  "schema_version": 1,
  "meta": {
    "investigation_id": "9f2c01ab34cd",
    "label": "user label or ''",
    "created_at": 1789000000.0,
    "updated_at": 1789000123.4,
    "revision": 7,
    "status": "active",            // "active" | "solved" | "unsolved"
    "language": "en",
    "alphabet_size": 26,
    "tokens": 312,
    "words": 61,
    "client_name": "claude-code",  // from initialize, at creation
    "terminal": null               // or the Part-5.22/5.23 terminal record
  },
  "state": { ... },                // InvestigationState.to_artifact_dict()
  "records": {                     // server-side sidecar (NOT in state)
    "comparisons": [ ... ],        // Part 5.16
    "repair_compiles": [ ... ]     // Part 8.2 (capped at 8, oldest dropped)
  }
}
```

`meta.json` is `meta` alone, written in the same commit (second atomic
rename). Readers of `meta.json` tolerate it being one commit behind.

**No InvestigationState schema change.** Client comparisons and repair
compiles live in the sidecar precisely so `state.py` stays untouched. Client
readings go into the existing `state.readings` map (Part 5.15) — that is
existing schema.

### 3.3 Atomic commit + revision protocol

`InvestigationRegistry` (one instance per server process, constructed with
the root path):

```python
class InvestigationRegistry:
    def __init__(self, root: Path) -> None
    def create(self, *, meta: dict, state_dict: dict) -> None          # revision 1
    def load(self, investigation_id: str) -> dict                      # full document
    def list(self, limit: int = 50) -> list[dict]                      # metas, newest first
    def commit(self, investigation_id: str, document: dict) -> int     # bump + write, returns new revision
    def append_event(self, investigation_id: str, event: dict) -> None # best-effort jsonl append
    def acquire_lease(self, investigation_id: str) -> bool             # §3.4
    def holds_lease(self, investigation_id: str) -> bool
    def lease_holder_hint(self, investigation_id: str) -> dict | None  # {"pid":..} best-effort
```

Write path (`create`/`commit`): serialize to `investigation.json.tmp` in the
same directory, `os.replace` onto `investigation.json`, then the same for
`meta.json`. `commit` sets `meta.revision += 1` and `meta.updated_at` before
writing and returns the new revision. `load` for an unknown id raises
`InvestigationNotFound` (a registry-local exception the server maps to the
typed `investigation_not_found` result).

Revision checking is the SERVER's job (Part 5.0), not the registry's: the
registry only stores and bumps.

### 3.4 Writer lease — the §9.1 single-owner rule

Concurrency model (implements proposal §3.2 + §9.1 without a daemon):

- **Mutations require the lease.** Before executing any mutating tool
  (Part 5.0 table), the server calls `acquire_lease(id)`:
  `open(<dir>/lease.lock, "a+")` then
  `fcntl.flock(fd, LOCK_EX | LOCK_NB)`. Success → keep the fd open for the
  process lifetime (store in a dict `id → fd`), write `{"pid": os.getpid(),
  "acquired_at": time.time()}` into the file (truncate first; best-effort),
  return True. Already held by this process → True. Held elsewhere
  (`BlockingIOError`) → False, and the tool returns
  `{"status": "blocked", "reason": "writer_lease_held", "holder": <hint>,
  "note": "Another live session owns writes for this investigation. Continue
  there, or retry after it exits."}`.
- **flock releases automatically on process death**, so a crashed holder
  never wedges an investigation: the next mutating call in any process
  acquires the lease and loads the on-disk state (whose `pending|running`
  experiment records were already flipped to `orphaned(loaded)` by
  `InvestigationState.from_artifact_dict` — the existing EXP-3 rule).
- **Experiment threads run only in the lease holder.** The
  `ExperimentQueue` lives in the holder's `InvestigationRuntime`; no other
  process ever starts or adopts threads (§9.1 verbatim). Non-holders see
  experiment records read-only through the registry.
- **Reads never need the lease.** A non-holder read-only tool loads a fresh
  snapshot via `load()` (atomic rename ⇒ readers always see a complete
  document) and builds an ephemeral runtime for rendering.
- The lease holder trusts its in-memory runtime between commits (no other
  process can write). On FIRST acquiring a lease the server always builds
  the runtime from a fresh `load()`.

`fcntl` is POSIX-only; the project targets macOS/Linux (env states darwin).
Guard the import: on platforms without `fcntl`, `acquire_lease` falls back
to an `os.open(..., O_CREAT | O_EXCL)` pid-file with stale-pid takeover
(check `os.kill(pid, 0)`); this branch needs no test coverage beyond import
safety.

### 3.5 Revision conflict semantics

Every mutating tool carries required `expected_revision` (integer). The
server compares it against the CURRENT revision (in-memory for the holder,
on-disk otherwise) BEFORE executing. Mismatch →

```json
{"status": "conflict", "reason": "revision_mismatch",
 "expected_revision": <client value>, "current_revision": <server value>,
 "note": "State changed since your last brief. Call investigation_status and retry."}
```

No mutation occurs. On success the server executes, calls
`registry.commit`, and includes `"revision": <new>` in the result. This is
the §3.2 optimistic-concurrency contract; last-writer-wins is impossible by
construction (lease + revision).

---

## Part 4 — Investigation runtime (`runtime.py`)

### 4.1 Construction recipe

`InvestigationRuntime` reproduces exactly the object graph `run_v3` builds,
minus the model session and loop:

```python
class InvestigationRuntime:
    def __init__(self, *, document: dict, emit, verify_provider, verify_model,
                 max_cost_usd: float, synchronous_experiments: bool = False) -> None
```

Steps (all imports are existing symbols; do not re-implement):

1. `state = InvestigationState.from_artifact_dict(document["state"])`.
2. Language resources exactly as `run_v3` lines 293–296:
   `dict_path = dictionary.get_dictionary_path(state.language)`;
   `word_set = dictionary.load_word_set(dict_path) if dict_path else set()`;
   `word_list = pattern.load_word_list(dict_path) if dict_path else []`;
   `pattern_dict = pattern.build_pattern_dictionary(word_list)`.
3. `executor = WorkspaceToolExecutor(workspace=state.workspace,
   language=state.language, word_set=..., word_list=..., pattern_dict=...,
   benchmark_context=None,
   declaration_policy=AttestationPolicy(attestations=state.verify_attestations),
   repair_agenda=state.repair_agenda,
   hypothesis_board=state.hypothesis_board,
   finalist_sessions=state.finalist_sessions,
   model_variant=state.model_variant)` — byte-for-byte the `run_v3`
   construction (lines 308–332) except `benchmark_context=None`.
   Do NOT call `set_max_iterations` (stays `None`; MCP has no turn budget).
4. `queue = ExperimentQueue(synchronous=synchronous_experiments or None)`
   (None ⇒ env default, preserving `DECIPHER_EXPERIMENT_SYNC` for tests).
5. `provider = <verify provider or None>` (Part 7.1).
6. `session = _NullLeadSession()` (§4.2).
7. `host = InvestigationHost(state=state, workspace=state.workspace,
   executor=executor, queue=queue, emit=emit, session=session,
   model_provider=provider, language=state.language, word_set=word_set,
   word_list=word_list, pattern_dict=pattern_dict, episode_models=None,
   max_cost_usd=max_cost_usd, prior_budget=list(state.budget_ledger))`.
8. `host.set_available_tools(V3_MAPPED_TOOL_NAMES)` once at construction,
   where

   ```python
   V3_MAPPED_TOOL_NAMES = {
       "observe_diagnosis", "decode_show",
       "workspace_create_hypothesis_branch", "workspace_update_hypothesis",
       "workspace_reject_hypothesis", "workspace_hypothesis_next_steps",
       "experiment_submit", "experiment_collect",
       "branch_adjudicate", "act_set_model_variant",
       "meta_declare_solution", "meta_declare_unsolved",
   }
   ```

   (module constant in `runtime.py`). This is the RES-5 gate surface: any
   other name reaching `host.handle_tool` is rejected
   `lead_tool_not_available` — defense in depth behind the server's own
   routing table.

Provide `runtime.persist_dict() -> dict` returning
`{"meta": ..., "state": state.to_artifact_dict(), "records": records}` after
`host.sync_budget()` (so episode spend lands in `state.budget_ledger` before
serialization).

### 4.2 `_NullLeadSession`

```python
class _NullLeadSession:
    capabilities = SessionCapabilities()
    model = "mcp-null"
    def send(self, blocks, tools=None, max_tokens=8192):
        raise RuntimeError("the MCP server makes no lead model calls")
    def usage_entries(self):  return []
    def export_transcript(self): return {"provider": "mcp", "messages": []}
```

Satisfies the `ModelSession` protocol; `host.sync_budget()` then rebuilds
`state.budget_ledger = prior + episode_budget + []`, i.e. prior spend plus
any server-side verify spend — exactly BUD-1's accounting with zero lead
spend.

### 4.3 Turn semantics

`state.turn` is the bookkeeping clock for `created_turn` stamps, saturation
entries, and evidence entries. Rule: **increment `state.turn` by 1 at the
start of every MUTATING tool call** (and set
`state.workspace.set_iteration(turn)`, `executor.set_iteration(turn)` — same
as the loop's per-turn calls). Read-only calls do not increment (so
unpersisted drift is impossible). Freshness everywhere in the gate machinery
is hash-based, not turn-based, so this is safe by construction.

### 4.4 emit / events

The server's `emit(event, payload, **extra)` appends
`{"ts": time.time(), "event": event, "payload": payload, "turn": state.turn}`
via `registry.append_event`. Failures are swallowed (observability only).
This preserves the host's telemetry events (`repeated_call`,
`duplicate_read_suppressed`, `episode_complete`,
`repair_transaction_complete`, …) as an auditable stream — the ledger's
telemetry-class controls (DUP-2, DUP-3 n/a, CMP-4) report here.

### 4.5 Experiment polling

At the START of every tool call for an investigation where this process
holds the lease: `transitioned = queue.poll(state, state.turn)`. If
`transitioned` is non-empty, the server commits (revision bump) even for a
read-only tool — new information arrived and stale clients must refresh.
This mirrors `run_v3`'s per-turn poll (M4) and keeps EXP-3's lifecycle
honesty. Non-holders never poll (they own no threads).

Server shutdown (EOF in the protocol loop): for each leased runtime, run the
`run_v3` finalize sequence verbatim — `queue.poll(state, state.turn,
promote=False)`; flip remaining `pending|running` records to
`orphaned(run_ended)`; `queue.finalize_env_restore()`; `host.sync_budget()`;
commit. (Copy the semantics of `loop_v3.py` lines 890–898; this is ~10 lines
against public queue methods, not duplication of host logic.)

---

## Part 5 — Tool surface

### 5.0 Conventions

23 tools (the §3.5 list minus the two v1.1 deferrals). Classification:

| class | tools | expected_revision | lease |
|---|---|---|---|
| create | `investigation_start` | no (returns revision 1) | acquires |
| read | `investigation_status`, `investigation_list`, `observe_overview`, `observe_diagnosis`, `decode_show`, `hypothesis_next_steps`, `candidate_list`, `candidate_show`, `branch_adjudicate` | no | no |
| mutate | `hypothesis_branch_create`, `hypothesis_branch_update`, `hypothesis_branch_reject`, `experiment_submit`, `experiment_collect`, `reading_record`, `comparison_record`, `repair_hypotheses_test`, `repair_transaction`, `request_independent_verification`, `act_set_model_variant`, `meta_declare_solution`, `meta_declare_unsolved` | REQUIRED | required |

Server-side dispatch pipeline for every `tools/call`:

1. Unknown tool → protocol `isError: true`.
2. Validate `arguments` against the tool's `input_schema` using the existing
   `episodes.validate_against_schema` (it supports
   type/properties/required/items/enum — the same dialect the schemas below
   restrict themselves to). Errors →
   `{"status": "error", "reason": "invalid_arguments", "errors": [...]}`.
3. Resolve the investigation (`investigation_not_found` on miss).
4. Terminal check: if `meta.status != "active"` and the tool is class
   mutate → `{"status": "blocked", "reason": "investigation_terminal",
   "terminal_status": <status>}`. Read tools still work on terminal
   investigations.
5. Class mutate: acquire lease (else `writer_lease_held`); check
   `expected_revision` (else `revision_mismatch`, §3.5); `state.turn += 1`
   + iteration mirrors (§4.3).
6. Poll experiments if holder (§4.5).
7. Execute the mapped operation (below).
8. Class mutate (and any poll-transition): `host.sync_budget()`;
   `registry.commit`; add `"revision"` to the result.
9. Serialize the result dict with `json.dumps(..., ensure_ascii=False)` into
   the single text content block.

Where a tool maps to `host.handle_tool`, the server builds
`tu = {"id": f"mcp_{uuid.uuid4().hex[:8]}", "name": <v2/v3 name>,
"input": <arguments minus investigation_id/expected_revision>}` and calls
`host.handle_tool(tu, state.turn)`; the returned JSON string is parsed and
re-wrapped so the envelope can add `revision`. DUP-1 (read cache), DUP-2
(signature counts), and the model-variant mirror all come for free from
`handle_tool`.

All schemas below share:

```json
"investigation_id": {"type": "string", "description": "Id returned by investigation_start/list."}
"expected_revision": {"type": "integer", "description": "Revision from your latest investigation_status; mismatch returns a conflict."}
```

### 5.1 `investigation_start` (create)

Description text: "Start a new cipher investigation from INLINE ciphertext
(never a file path). Returns the investigation id, revision 1, and the full
self-briefing. Formats: 'canonical' = space-separated S-tokens with ' | '
word separators; 'letters' = plain text, whitespace separates words; 'auto'
detects canonical S-token transcriptions."

```json
{"type": "object",
 "properties": {
   "ciphertext": {"type": "string"},
   "language": {"type": "string", "enum": ["en", "la", "de", "fr", "it", "es"]},
   "format": {"type": "string", "enum": ["auto", "letters", "canonical"]},
   "label": {"type": "string"},
   "context": {"type": "string",
     "description": "Optional prior/historical context; rendered every turn as the external-context section. Never treated as ground truth."}
 },
 "required": ["ciphertext"]}
```

Behavior (`intake.py::build_cipher_text(text, fmt) -> CipherText`):

- Reject empty/whitespace ciphertext → `{"status":"error","reason":"empty_ciphertext"}`.
  Reject `len(ciphertext) > 200_000` → `ciphertext_too_large`.
- `fmt == "canonical"` → `benchmark.loader.parse_canonical_transcription(text)`.
- `fmt == "letters"` → exactly the `cmd_crack` non-canonical path
  (`cli.py` lines 850–855): `Alphabet.from_text(text, ignore_chars={" ",
  "\t", "\n", "\r"})`, `clean = " ".join(text.split())`,
  `CipherText(raw=clean, alphabet=alphabet, source="mcp", separator=" ")`.
- `fmt == "auto"` (default): canonical iff
  `re.search(r"\bS\d{3}\b", text)` — the canonical S-token shape; else
  letters.
- Parse exceptions → `{"status":"error","reason":"parse_failed","error":str(exc)}`.

Then: `investigation_id = uuid.uuid4().hex[:12]`;
`workspace = Workspace(cipher_text=ct)`;
`state = InvestigationState(workspace=workspace,
language=arguments.get("language") or "en")`;
`state.external_context = arguments.get("context") or ""`; write the
diagnostic-preflight evidence entry with the same fields as `run_v3`
(lines 370–388): compute
`cipher_id_analysis.compute_cipher_fingerprint(ct.tokens, ct.alphabet.size,
language=..., word_group_count=len(ct.words))` and
`state.add_evidence("diagnostic_preflight", turn=0, summary=..., fingerprint=fp.to_dict(),
suspicion_scores=fp.to_dict().get("suspicion_scores") or {})` (copy the
summary f-string verbatim). `state.turn = 0`. No automated preflight solver
runs at start (clients use `experiment_submit`; keeps start fast and
§6-parity honest).

Persist via `registry.create` (revision 1, meta filled per §3.2). Result:

```json
{"investigation_id": "...", "revision": 1, "language": "en",
 "alphabet_size": 26, "tokens": 312, "words": 61,
 "brief": "<the Part-6 status brief text>"}
```

### 5.2 `investigation_list` (read; no investigation_id)

```json
{"type": "object", "properties": {"limit": {"type": "integer"}}}
```

Returns `{"investigations": [meta stubs, newest updated first]}`, each stub =
meta minus `terminal` details plus `has_terminal: bool`. Default/max limit
50.

### 5.3 `investigation_status` (read)

```json
{"type": "object", "properties": {"investigation_id": {"type": "string"}},
 "required": ["investigation_id"]}
```

Returns:

```json
{"investigation_id": "...", "revision": N, "status": "active",
 "turn": T, "budget": {"total_cost_usd": ..., "max_cost_usd": ...},
 "verification_available": true,
 "brief": "<Part-6 text blob>"}
```

This is the §4.3 self-briefing: one call fully briefs a bare session.

### 5.4 `observe_overview` (read)

Description: "Compact measured facts: alphabet, token/word counts, IC,
diagnostic fingerprint, model variants, branch and evidence counts. A subset
of investigation_status for cheap re-orientation."

Schema: same as 5.3. Result: `{"overview": "<text>"}` where the text is
`context._render_cipher(state)` + `"\n\n"` +
`context._render_fingerprint(state)` + a final line
`f"Branches: {len(ws.branch_names())}   Readings: {len(state.readings)}   "
f"Attestations: {len(state.verify_attestations)}   Evidence entries: {len(state.evidence_log)}"`.
(Both renderers are module-level functions in `investigation/context.py`;
import them directly.)

### 5.5 `observe_diagnosis` (read → `host.handle_tool("observe_diagnosis")`)

```json
{"type": "object",
 "properties": {"investigation_id": {"type": "string"},
   "branch": {"type": "string"}, "max_period": {"type": "integer"}},
 "required": ["investigation_id"]}
```

Pass-through of the v2 tool (default branch "main" handled by the executor).

### 5.6 `decode_show` (read → `host.handle_tool`)

```json
{"type": "object",
 "properties": {"investigation_id": {"type": "string"},
   "branch": {"type": "string"},
   "start_word": {"type": "integer"}, "count": {"type": "integer"}},
 "required": ["investigation_id", "branch"]}
```

DUP-1: a second identical call against unchanged content returns the host's
`duplicate_suppressed` payload — pass it through unchanged.

### 5.7–5.9 `hypothesis_branch_create` / `hypothesis_branch_update` / `hypothesis_branch_reject` (mutate → `host.handle_tool`)

Map to `workspace_create_hypothesis_branch` / `workspace_update_hypothesis`
/ `workspace_reject_hypothesis`. Schemas: copy the v2 `input_schema`
property blocks verbatim from `tools_v2.py` (including enums and defaults)
and add `investigation_id` + `expected_revision` to `properties` and
`required`. (For `hypothesis_branch_reject`, drop the
`acknowledge_pending_required_tools` property — it discharges v2 gate state
that never arises under `NoGatesPolicy`/`AttestationPolicy`.)

### 5.10 `hypothesis_next_steps` (read → `host.handle_tool("workspace_hypothesis_next_steps")`)

```json
{"type": "object",
 "properties": {"investigation_id": {"type": "string"}, "branch": {"type": "string"}},
 "required": ["investigation_id"]}
```

The server wraps the executor result:
`{"advisory": true, "policy_ids": ["WF-1"], "suggestions": <parsed executor result>,
"note": "Recommendations, not requirements (policy WF-1, advisory in MCP v1)."}`
— the §3.5 requirement that `next_steps` returns policy ids and rationale,
not a binding whitelist.

### 5.11 `candidate_list` (read; new — the WF-6 divergence surface)

Description: "The candidate portfolio: every active branch with provenance,
labeled score signals, verification status, and derived roles. No single
scalar defines 'the' candidate; the scalar-best branch is one labeled role
(policy WF-6, pre-registered divergence)."

```json
{"type": "object",
 "properties": {"investigation_id": {"type": "string"},
   "limit": {"type": "integer"}, "offset": {"type": "integer"},
   "include_rejected": {"type": "boolean"}},
 "required": ["investigation_id"]}
```

Implementation: `roles = loop_v3._compute_branch_roles(state, executor,
None)` (import from `investigation.loop_v3`; importing is allowed, editing
is not). For each branch name (workspace order; skip
`mode_status in {"rejected","superseded"}` unless `include_rejected`):

```json
{"branch": name,
 "content_hash": host_module._branch_hash(workspace, name),
 "mapped_count": len(branch.key),
 "scores": executor._compute_quick_scores(name),        // {"dict_rate", "quad"} — labeled signals
 "roles": [r for r, b in roles.items() if b == name],
 "tags": branch.tags,
 "parent": branch.parent,
 "created_turn": branch.created_iteration,
 "attestation": <summary of latest_attestation_for_hash(state.verify_attestations, hash)>,
 "verification": "positive" | "negative" | "none" | "unavailable",
 "readings": <count of state.readings bound to this hash>,
 "excerpt": <first 120 chars of _decoded_text_for_panel(workspace, name)>}
```

`attestation` summary keys: `reader_accepts_as_solution`,
`target_language_confidence`, `semantic_recoverability`, `damage_scope`,
`repairability`, `created_turn`; `null` when none. `verification` is
`"unavailable"` when the server has no verify provider AND no attestation
exists (the §8.1.4 label surface); with no provider but an existing
attestation, report the attestation verdict. Result:
`{"candidates": [...], "total": N, "offset": ..., "note":
"Scores are individual signals (dict_rate, quad), not a ranking (WF-6)."}`
Default limit 12, max 50.

### 5.12 `candidate_show` (read; new)

```json
{"type": "object",
 "properties": {"investigation_id": {"type": "string"},
   "branch": {"type": "string"}, "max_chars": {"type": "integer"}},
 "required": ["investigation_id", "branch"]}
```

Unknown branch → `{"status":"error","reason":"unknown_branch"}`. Result: the
5.11 entry for the branch, plus `"decoded_text"` =
`_decoded_text_for_panel(workspace, branch)` truncated to `max_chars`
(default 4000, cap 20000) via `context._truncate`, plus
`"attestation_history"` (every attestation whose `content_hash` matches the
current hash, full records), `"readings"` (ids + confidence + 120-char
preview for readings bound to the hash), and `"repair_transactions"` (the
`state.repair_transactions` records whose `source_branch` or
`installed_branch` equals the branch; compact: transaction_id, status,
reason, installed_branch).

### 5.13 `experiment_submit` (mutate → `host.handle_tool`)

Schema: `EXPERIMENT_SUBMIT_TOOL["input_schema"]` from `experiments.py`
verbatim (the typed config schema INCLUDED — EXP-1's exact model-facing
schema exposure), plus `investigation_id`/`expected_revision` in
properties+required. Description: reuse `EXPERIMENT_SUBMIT_TOOL["description"]`
with one appended sentence: "Experiments run in the background inside the
session that holds this investigation's writer lease; results surface in
investigation_status and experiment_collect."

All EXP-1/EXP-2/GT-3 validation, dedup, and the corrected-example error
payloads flow through `dispatch_experiment_submit` unchanged via
`host.handle_tool`.

### 5.14 `experiment_collect` (mutate → `host.handle_tool`)

Schema: `EXPERIMENT_COLLECT_TOOL["input_schema"]` + the two envelope fields;
description reused verbatim.

### 5.15 `reading_record` (mutate; new — client-authored reading)

Description: "Record YOUR reading of a branch's decode, hash-bound to its
current content. Required before repair_transaction (evidence binding
REP-1). One reading per (content, verifier-evidence) pair (SAT-4); duplicates
return the existing reading_id."

```json
{"type": "object",
 "properties": {
   "investigation_id": {"type": "string"}, "expected_revision": {"type": "integer"},
   "branch": {"type": "string"},
   "reading_text": {"type": "string"},
   "fragments": {"type": "array", "items": {"type": "object",
     "properties": {"text": {"type": "string"},
       "repair_text": {"type": ["string", "null"]},
       "span_id": {"type": ["string", "null"]},
       "start": {"type": ["integer", "null"]}, "end": {"type": ["integer", "null"]},
       "confidence": {"type": ["number", "string"]}},
     "required": ["text"]}},
   "holes": {"type": "array", "items": {"type": "string"}},
   "overall_confidence": {"type": "number"}},
 "required": ["investigation_id", "expected_revision", "branch",
              "reading_text", "overall_confidence"]}
```

Implementation (server.py helper, using existing symbols only):

1. Unknown branch → `unknown_branch`. `content_hash =
   host_module._branch_hash(workspace, branch)`.
2. SAT-4 duplicate suppression — replicate the host's reading-suppression
   check by calling the SAME helpers it calls
   (`host.py::_dispatch_episode_run`, the `kind == "reading"` block):
   `att_key = attestation_key(latest_attestation_for_hash(state.verify_attestations, content_hash))`;
   `sat_entry = state.repair_saturation.get(saturation_key(content_hash, att_key))`;
   if `sat_entry` exists with `int(sat_entry.get("readings") or 0) >= 1`,
   return the host's exact payload shape:
   `{"status": "blocked", "reason": "duplicate_reading_suppressed", "branch": ...,
   "content_hash": ..., "existing_reading_id": <newest reading for the hash or null>,
   "note": <the host's note string verbatim>}`.
3. Build `result = {"reading_text": ..., "fragments": arguments.get("fragments") or [],
   "holes": arguments.get("holes") or [], "overall_confidence": ...}` and
   validate against `episodes._READING_SCHEMA` with
   `validate_against_schema`; errors → `invalid_reading` with the error
   list.
4. `packet = build_candidate_reading_packet(workspace, branch).to_dict()`;
   `reading = Reading.from_episode_result(result, branch=branch,
   source=f"client:{server.client_name}", created_turn=state.turn,
   candidate_packet=packet)`;
   `state.readings[reading.reading_id] = reading.to_dict()`.
5. Saturation accounting exactly as the host does after compiling a
   reading: `sat = get_or_create_saturation_entry(state, content_hash,
   att_key, state.turn)`; `sat["readings"] += 1`;
   `sat["updated_turn"] = state.turn`.
6. Result: `{"status": "ok", "reading_id": ..., "branch": ...,
   "content_hash": ..., "fragment_count": ..., "revision": N}`.

### 5.16 `comparison_record` (mutate; new — the CMP-2 divergence)

Description: "Record YOUR ranking of competing candidates, hash-bound to
each branch's current content. best_partial and accepts_as_solution are
SEPARATE fields (policy CMP-2): an honest 'best partial so far' never
requires claiming a solution. accepts_as_solution here never unlocks
declaration — only request_independent_verification can (DECL-1)."

```json
{"type": "object",
 "properties": {
   "investigation_id": {"type": "string"}, "expected_revision": {"type": "integer"},
   "branches": {"type": "array", "items": {"type": "string"}},
   "ranking": {"type": "array", "items": {"type": "string"}},
   "best_partial": {"type": ["string", "null"]},
   "accepts_as_solution": {"type": "boolean"},
   "rationale": {"type": "string"}},
 "required": ["investigation_id", "expected_revision", "branches", "ranking",
              "best_partial", "accepts_as_solution", "rationale"]}
```

Validation: 2 ≤ len(branches) ≤ 8; every branch exists; `ranking` is a
permutation of a subset of `branches`; `best_partial` ∈ branches or null.
Store in `records["comparisons"]`:

```json
{"comparison_id": uuid.uuid4().hex[:12], "created_turn": T,
 "client": server.client_name, "branches": [...],
 "branch_hashes": {name: _branch_hash(...)},
 "ranking": [...], "best_partial": ..., 
 "best_partial_hash": <hash or null>,
 "accepts_as_solution": bool, "rationale": "..."}
```

Also `state.add_evidence("client_comparison", turn=T,
summary=f"client ranked {ranking[:3]}...", comparison_id=...)` so the brief
surfaces it. Result: the stored record + revision. (v1 stores hash-bound
records only; integration extras are v1.1 per proposal §7.2.)

### 5.17 `repair_hypotheses_test` (mutate; new — Part 8.2)

### 5.18 `repair_transaction` (mutate; new — Part 8.3)

Schemas and behavior are specified in Part 8 (they depend on the host
refactor in 8.1).

### 5.19 `request_independent_verification` (mutate; Part 7)

### 5.20 `branch_adjudicate` (read → `host.handle_tool`)

Schema: `BRANCH_ADJUDICATE_TOOL["input_schema"]` from `actions.py` +
`investigation_id` (read class: no expected_revision). Description reused
verbatim. Routed through `handle_tool` so the composite dispatcher logs the
ToolCall and DUP-1 caches repeats.

### 5.21 `act_set_model_variant` (mutate → `host.handle_tool`)

Schema: the v2 `input_schema` + envelope fields. The host mirrors
`executor._model_variant → state.model_variant` (CTX-5); the commit persists
it; `meta.model_variant` is NOT stored (state owns it).

### 5.22 `meta_declare_solution` (mutate → `host.handle_tool`)

Schema: the v2 `input_schema` properties verbatim (branch, rationale,
self_confidence, reading_summary, further_iterations_helpful,
further_iterations_note, forced_partial) + envelope fields; required list =
v2 required + envelope. Description: v2 description + " Declaration is
hard-gated (DECL-1): it succeeds only when the newest independent-reader
attestation matching this branch's current content hash is positive
(reader_accepts_as_solution). Run request_independent_verification first."

Flow: `host.handle_tool` → executor → `AttestationPolicy.check_declare_solution`.
After the call, if `executor.terminated and executor.solution is not None`:
attach the matching attestation exactly as `run_v3` does at lines 767–784
(same `max(...)` selection over `state.verify_attestations` by
`(created_turn, episode_id)` for the declared branch's current hash; set
`sol.attestation`); build
`terminal = {"kind": "solution", "declared_turn": state.turn,
"solution": dataclasses.asdict(executor.solution)}`;
set `meta.status = "solved"`, `meta.terminal = terminal`. Result: the
executor's accept payload (parsed) + `{"terminal_status": "solved",
"revision": N}`. A blocked declaration returns the `AttestationPolicy`
payload untouched (reason `attestation_required` / `attestation_stale` /
`attestation_not_positive` with the echoed verdict — the exact keyless-user
surface, Part 7.3).

DECL-5 analog: MCP calls are one-per-request, so a same-batch post-terminal
mutation cannot occur; the terminal check in the dispatch pipeline (step 4)
enforces the same invariant across subsequent calls.

### 5.23 `meta_declare_unsolved` (mutate → `host.handle_tool`)

Schema: v2 properties + envelope; required = v2 required + envelope.
DELIBERATELY ungated (DECL-8) — the policy base class allows it. On
`executor.terminated` with `executor.unsolved_declaration`: `terminal =
{"kind": "unsolved", "declared_turn": T, "declaration":
dict(executor.unsolved_declaration)}`; `meta.status = "unsolved"`. Result:
executor payload + `{"terminal_status": "unsolved", "revision": N}`.

---

## Part 6 — The status brief (`status.py`)

`build_status_brief(state, executor, host, *, turn, verification_available,
max_cost_usd) -> str` — a pure function composing ONE text blob. It reuses
the factual section renderers from `investigation/context.py` (module-level
functions; import them by name) and REPLACES the binding workflow rendering
with an advisory block (proposal §4.3: "the status tool must not smuggle
the ... binding action menu back ... as if they were neutral facts";
Phase A note: "reuse factual context renderers selectively rather than
importing the binding workflow menu verbatim").

Section order (drop any renderer's empty-string section):

1. `_render_framing(state)`
2. `_render_cipher(state)`
3. `_render_fingerprint(state)`
4. `_render_external_context(state)`
5. `f"## Investigation state (turn {turn}, revision {revision})"` — one
   line, plus `f"Budget: ${host.committed_cost():.4f} of ${max_cost_usd:.2f} server-side ceiling (BUD-1)."`
   and `f"Independent verification: {'available' if verification_available else 'UNAVAILABLE — no API key (see advisory below)'}."`
6. `_render_branch_cards(state, executor, DEFAULT_BRANCH_CARDS)`
7. `_render_hypothesis_board(state)`
8. `_render_episode_ledger(state)`
9. `_render_experiment_queue(state)`
10. `_render_readings(state)`
11. Client comparisons (new, local renderer `_render_comparisons(records)`):
    last 3 sidecar comparison records, one line each:
    `- [t{created_turn}] ranking: a > b > c; best_partial=`x`; accepts_as_solution=false — <rationale ≤120 chars>`;
    omitted when none.
12. `_render_evidence(state, DEFAULT_EVIDENCE_ENTRIES)`
13. `_render_window(state, executor, turn, DEFAULT_WINDOW_TOKENS)`
14. The advisory block (below).

### 6.1 The advisory block — `_render_advisory(state, executor, verification_available)`

Header line:

```
## Host guidance (advisory unless marked ENFORCED; policy ids per docs/mcp_policy_provenance_ledger.md)
```

Lines, in order (each conditional on its predicate):

1. Workflow recommendation (WF-1, advisory): compute
   `menu = context.workflow_state(state, executor)`. Render
   `- [WF-1 advisory] Suggested focus: {menu['state']}` (+
   `` on `{branch}` `` when present), then one indented `- ` line per
   `menu["actions"]` item. Never the word "must"; the actions text is
   reused verbatim from the menu (it is already imperative but the header
   marks it advisory).
2. Verifier route (WF-4, advisory): when the workflow branch has a fresh
   NON-positive attestation, add
   `- [WF-4 advisory] Independent-reader route for `{branch}`: {route}`
   where `route = context._attestation_route(attestation)`; include
   `(target_language_confidence={tlc:.2f}, semantic_recoverability={recov:.2f}, damage_scope={scope})`.
3. Declare hint (WF-5, advisory): if any branch has a fresh POSITIVE
   attestation —
   `- [WF-5 advisory] `{branch}` has a fresh positive verification; declare it unless you hold concrete contradictory evidence.`
4. Repair hint (WF-5, advisory): if the workflow branch has a fresh
   non-positive attestation whose route is `repair` —
   `- [WF-5 advisory] Record a reading of `{branch}` (reading_record), compile hypotheses (repair_hypotheses_test), then repair_transaction; reverify the changed content.`
   (The budget-relative v3 hints — mid-budget/late-turn/late-adjudication —
   are structurally n/a: MCP has no turn budget. Recorded in Part 9.)
5. Saturation (SAT-3, ENFORCED): when
   `context._exhausted_entry_for(state, menu.get("branch"), <fresh attestation>)`
   is non-None —
   `- [SAT-3 ENFORCED] Repair is exhausted for this candidate content and verifier evidence; repair_transaction will be blocked until content or verifier evidence changes. Alternate search (experiment_submit), compare distinct finalists, or declare honestly unsolved.`
6. Declaration gate (DECL-1, ENFORCED): always —
   `- [DECL-1 ENFORCED] meta_declare_solution requires a fresh POSITIVE independent verification of the branch's current content. meta_declare_unsolved is never gated (DECL-8).`
7. Keyless (only when `not verification_available`):
   `- [DECL-1 ENFORCED] Independent verification is UNAVAILABLE (no server-side API key). The solved-declaration gate stays closed; your strongest candidate remains available via candidate_list, labeled 'promising but not independently verified'. Configure a provider key (OPENAI_API_KEY / ANTHROPIC_API_KEY / .decipher_keys/<provider>_api_key or keychain service 'decipher'), restart the MCP server, and resume — state is preserved.`
8. Portfolio note (WF-6, divergence): always —
   `- [WF-6] Candidate attention is yours: candidate_list shows every branch with labeled signals; no scalar silently selects 'the' candidate.`

The brief is NOT re-clamped beyond each section's own cap (the section caps
in `context.py` already bound it; total worst case ≈ 30k chars — acceptable
for a deliberate status call, per proposal risk 8's "compact packets +
explicit detail calls" resolved via `observe_overview` for cheap reads).

---

## Part 7 — `request_independent_verification` (`verify.py`)

### 7.1 Provider resolution (server start, once)

`resolve_verify_provider(verify_provider_flag, verify_model_flag)`:

- `"none"` → `(None, None)` (forced keyless; for tests and offline use).
- `"auto"` → iterate `("anthropic", "openai", "gemini", "openrouter")` with
  `cli._probe_api_key(provider)` (import from `cli`; it is silent and
  side-effect-free) and take the first with a key. No key anywhere →
  `(None, None)`.
- Explicit provider → `_probe_api_key(provider)`; empty → print a stderr
  warning and `(None, None)` (the server still boots — §8.1.4: absent keys
  must not block installation or investigation).
- Model: `verify_model_flag or default_model_for_provider(provider)`
  (existing helper in `agent/model_provider.py`).
- Build: `make_model_provider(provider=..., api_key=key, model=model)`.

The provider object is passed to every `InvestigationRuntime` as
`model_provider` (host ctor). `verification_available = provider is not
None`. The credential boundary (§9.1) holds by construction: the key lives
only in the provider object inside the server process; no tool result ever
contains it.

### 7.2 Tool

Description: "Run one server-side INDEPENDENT verification of a branch: a
fresh reader (no scores, no context you can shape) judges whether its decode
reads as real target-language text and writes a hash-bound attestation. A
positive attestation (reader_accepts_as_solution) is the only key that
unlocks meta_declare_solution (DECL-1). Server-side and API-billed
(Option A); returns a typed 'unavailable' result when the server has no
API key."

```json
{"type": "object",
 "properties": {"investigation_id": {"type": "string"},
   "expected_revision": {"type": "integer"},
   "branch": {"type": "string"}},
 "required": ["investigation_id", "expected_revision", "branch"]}
```

Flow (after the standard mutate pipeline):

1. `verification_available` false →

   ```json
   {"status": "unavailable", "reason": "no_verification_provider",
    "branch": "...",
    "declaration_gate": "closed",
    "note": "The server has no API key for verify episodes. The declaration gate (DECL-1) remains closed; work is preserved. Add a key (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY, .decipher_keys/<provider>_api_key, or macOS keychain service 'decipher'), restart the MCP server, and rerun this tool."}
   ```

   (Committed anyway? NO — no state changed; skip the commit and return the
   current revision. This is the one mutate-class tool that may exit without
   a revision bump.)
2. `host.cost_ceiling_reached()` →
   `{"status": "blocked", "reason": "cost_ceiling_reached",
   "committed_cost_usd": ..., "max_cost_usd": ...}` (BUD-1). No commit.
3. Call `payload_str = host._dispatch_verify_run({"branches": [branch],
   "goal": ""}, state.turn)`. This existing dispatcher: validates arity and
   existence (DECL-3), renders the candidate with the pinned renderer at
   dispatch time and hashes it (CMP-3), builds the empty-toolset spec
   (GT-2), runs `run_episode` (BUD-2/3/6/7, RES-1 retries, EPI-1/2 all
   inside), writes the `AttestationRecord` via the dispatcher (DECL-2), and
   seeds the repair agenda only on `repairability == "local_repair"`
   (REP-5). The lead-authored `goal` is deliberately empty — the MCP client
   gets no channel to shape the reader (stronger than v3, allowed: GT-2 is
   INV).
4. `host.sync_budget()`; commit. Result: the parsed dispatcher payload
   (episode id/status, `attestation` summary when written, `spend_usd`) +
   `{"revision": N}`.

### 7.3 What a keyless user sees (normative walkthrough)

1. `decipher mcp-serve` boots normally; stderr notes
   `verify provider: none (verification unavailable)`.
2. `investigation_status` brief line 5 shows
   `Independent verification: UNAVAILABLE — no API key (see advisory below)`
   and advisory line 7 (Part 6.1) explains the closed gate + recovery.
3. `request_independent_verification` returns the 7.2-step-1 typed result.
4. `meta_declare_solution` returns the AttestationPolicy block verbatim:
   `{"status": "blocked", "accepted": false, "branch": ..., "reason":
   "attestation_required", "how": "run a verify episode on this branch, then
   declare if the reader accepts it as a solution"}`.
5. `candidate_list` labels candidates `"verification": "unavailable"`.
6. `meta_declare_unsolved` works (DECL-8). Adding a key and restarting the
   server, then rerunning `request_independent_verification`, requires no
   other migration — state is on disk.

---

## Part 8 — Repair v1: client-compiled finalists

### 8.1 Host refactor (the ONLY edit to `host.py`; zero behavior change)

Split `InvestigationHost._dispatch_repair_transaction` into three parts.
The two new methods are public (no leading underscore) because the MCP glue
calls them.

**(a) `check_repair_preconditions(self, *, branch: str, reading_id_arg:
str, turn: int) -> dict[str, Any]`** — move the body of
`_dispatch_repair_transaction` from `source_hash = _branch_hash(...)`
(current line 1223) through the `note = (...)` assignment (line 1359),
INCLUDING the unknown-branch check currently above it. Every early return
`self._record_dispatch_result(...)` inside the moved code becomes
`return {"blocked": <payload dict>}` with the payload dicts byte-identical
(`unknown_branch`, `fresh_reading_required`, `reading_branch_mismatch`,
`stale_or_unbound_reading`, `duplicate_suppressed`/`source_and_reading_already_handled`,
`repair_saturated`, `pair_evidence_failed`). The success path returns:

```python
{"ok": True, "source_hash": source_hash, "reading_id": reading_id,
 "interp_digest": interp_digest, "att_key": att_key, "sat_key": sat_key,
 "pair": pair, "retry_of": retry_of, "anomalies": anomalies, "note": note}
```

**(b) `validate_and_install_repair(self, *, tu: dict, turn: int, branch:
str, source_hash: str, att_key: str, pair: str, base_record: dict,
episode_payload: dict, as_name: str) -> str`** — move the body from
`episode_id = str(episode_payload.get("episode_id") or "")` (line 1401)
through the final `return self._record_dispatch_result(...)` (line 1678),
with exactly three mechanical substitutions:

- `str(args.get("as_name") or f"repair_tx_{turn}_{branch}")` (line 1619) →
  the `as_name` parameter.
- `"reading_id": reading_id` in the installed-branch metadata (line 1641) →
  `base_record["reading_id"]`.
- `"addressed_anomalies": anomalies` (line 1643) →
  `base_record["addressed_anomalies"]`.

Everything else — ledger lookup by episode_id, snapshot hashing, the
`_worker_rejected_targets` / `_extract_repair_evidence` evidence build over
`self.episode_tool_calls` filtered by episode_id, the eight ordered
acceptance checks with their exact payloads and failure reasons, the
`_dispatch_episode_install` call, `_settle_repair_outcome`, the agenda
auto-close, the evidence entry, and the `repair_transaction_complete` emit —
moves verbatim.

**(c) `_dispatch_repair_transaction`** becomes:

```python
def _dispatch_repair_transaction(self, tu, turn):
    args = tu.get("input") or {}
    branch = str(args.get("branch") or "")
    pre = self.check_repair_preconditions(
        branch=branch, reading_id_arg=str(args.get("reading_id") or ""), turn=turn)
    if "blocked" in pre:
        return self._record_dispatch_result(
            name="repair_transaction", tu=tu, turn=turn, payload=pre["blocked"])
    episode_payload = json.loads(self._dispatch_episode_run({... unchanged,
        "reading_id": pre["reading_id"], "context_note": pre["note"] ...}, turn))
    transaction_id = uuid.uuid4().hex[:12]
    base_record = { ... unchanged, built from pre[...] ... }
    if episode_payload.get("status") != "ok":
        ... unchanged failed-episode settle-and-return ...
    return self.validate_and_install_repair(
        tu=tu, turn=turn, branch=branch, source_hash=pre["source_hash"],
        att_key=pre["att_key"], pair=pre["pair"], base_record=base_record,
        episode_payload=episode_payload,
        as_name=str(args.get("as_name") or f"repair_tx_{turn}_{branch}"))
```

No test edits are expected: the entire M5.3 repair suite drives this path
through `run_v3`/`handle_tool` and the payloads are unchanged. If any test
imports a moved local by line number (none known), fix the test reference,
never the behavior. Run the full suite before and after; both runs must
report 1708/2.

### 8.2 `repair_hypotheses_test` (`repair.py`) — deterministic compile

Description: "Compile a batch of word hypotheses against ONE branch in an
ISOLATED scratch workspace (no LLM; deterministic host code). Builds the
word-repair menu once, probes every hypothesis, installs each viable one on
a scratch fork, and returns per-item verdicts plus a deduped finalist set
with host-generated edit labels and collateral evidence. The result is a
compile session; pass its compile_id and your chosen winner to
repair_transaction for host-validated installation."

Schema: the `HYPOTHESIS_TEST_WORDS_TOOL["input_schema"]` properties
(`branch`, `hypotheses` items with word/claim_type/op/word_id/span_id/
start_token_id/end_token_id/word_index/char_start/label — DROP the per-item
`install` flag; the server forces install), the four shared menu knobs
(`window_size`, `max_edits`, `max_hypotheses`, `max_hypotheses_per_window`),
plus the envelope fields. Required: `investigation_id`, `expected_revision`,
`branch`, `hypotheses`.

`compile_hypotheses(runtime, records, args, turn) -> dict`:

1. `branch` must exist → else `unknown_branch`. `source_hash =
   _branch_hash(workspace, branch)`.
2. Duplicate suppression: `args_digest = sha256(json.dumps({"branch":...,
   "hypotheses":..., <the four knobs>}, sort_keys=True, default=str))`. If a
   stored compile in `records["repair_compiles"]` has the same
   `(source_content_hash, args_digest)`, return
   `{"status": "duplicate_suppressed", "compile_id": <existing>,
   "note": "Identical compile already exists for this content; reuse it or change the hypotheses."}`
   (no new session).
3. `compile_id = "rc" + uuid.uuid4().hex[:10]`.
4. Scratch isolation: `scratch = episodes._build_episode_workspace(state,
   [branch])` (deep-copied snapshots — EPI-1's exact mechanism).
5. Scratch executor: `WorkspaceToolExecutor(workspace=scratch,
   language=state.language, word_set=..., word_list=..., pattern_dict=...,
   benchmark_context=None, declaration_policy=NoGatesPolicy(),
   episode_toolset=set(episodes.EPISODE_KINDS["repair"]["toolset"]),
   model_variant=state.model_variant)`; `scratch_executor.episode_id =
   compile_id`; `scratch_executor.set_iteration(turn)`. (Fresh private
   board/agenda/finalists — episode parity.)
6. Force install: `hyps = [dict(h, install=True) for h in
   arguments["hypotheses"]]`.
7. `result_obj = execute_composite("hypothesis_test_words",
   {"branch": branch, "hypotheses": hyps, <the four knobs if supplied>},
   executor=scratch_executor, state_readings=state.readings, turn=turn,
   tool_use_id=f"mcp_{compile_id}")`. A top-level `"error"` in `result_obj`
   → return `{"status": "failed", "reason": "compile_error",
   "error": result_obj["error"]}` (no session stored).
8. Snapshots, episode-format: `snapshots =
   [state_module._serialize_branch(scratch, name) for name in
   scratch.branch_names() if name != "main" or "main" == branch]` — the
   same rule `run_episode` uses (requested branches + created forks; "main"
   only if it was the source).
9. `changed = {name: host._snapshot_content_hash(snap) for snap in
   snapshots ...}` filtered to hashes ≠ `source_hash` (reuse the host
   method; it restores into a scratch `Workspace` with the pinned
   renderer — CMP-3).
10. **Write the synthetic episode-ledger entry** (this is what lets
    `validate_and_install_repair` + `_dispatch_episode_install` run
    unchanged):

    ```python
    state.episode_ledger.append({
        "episode_id": compile_id, "kind": "repair_compile",
        "goal": f"client-compiled word hypotheses on {branch}",
        "status": "ok", "failure_reason": None, "result": None,
        "summary": f"{len(hyps)} hypotheses probed; {len(changed)} changed finalist fork(s)",
        "branch_snapshots": snapshots, "tool_call_count": 1,
        "budget_entries": [], "agenda_additions": [],
        "launching_turn": turn, "input_branches": [branch]})
    ```

11. Extend the host's evidence stream with the compile ToolCalls:
    `compile_calls = [tc for tc in scratch_executor.call_log if
    getattr(tc, "episode_id", None) == compile_id]`;
    `host.episode_tool_calls.extend(compile_calls)`. Persist them in the
    sidecar for cross-process/restart transactions:
    `stored_calls = [{"tool_name": tc.tool_name, "result": tc.result,
    "episode_id": tc.episode_id, "iteration": tc.iteration,
    "tool_use_id": tc.tool_use_id} for tc in compile_calls]`.
12. Store the compile session in `records["repair_compiles"]` (append; cap
    the list at 8 — drop oldest first):

    ```json
    {"compile_id": ..., "branch": ..., "source_content_hash": ...,
     "args_digest": ..., "created_turn": T,
     "result": <result_obj>, "tool_calls": <stored_calls>,
     "changed_finalists": {name: hash}}
    ```

    (Snapshots live in the episode-ledger entry inside state; not
    duplicated in the sidecar.)
13. Result to the client:

    ```json
    {"status": "ok", "compile_id": ..., "branch": ...,
     "source_content_hash": ...,
     "changed_finalists": [{"branch": name, "content_hash": h}],
     "result": <result_obj — per-item verdicts, finalists with
                installed_fork/edits/adjudication_summary>,
     "next": "Pick a winner from changed_finalists (with >1 changed it must be one of result.finalists) and call repair_transaction with this compile_id.",
     "revision": N}
    ```

### 8.3 `repair_transaction` (MCP variant, `repair.py`)

Description: "Host-validated acceptance and installation of ONE
client-compiled repair winner (Slice-4 contract, REP-1..4): binds your
stored reading to the branch's current content, suppresses duplicate
source/interpretation pairs, enforces saturation, runs the eight acceptance
checks against the compile evidence (winner named+changed, fork evidence,
edit-claim binding, adjudication with multiple finalists, collateral
limits, no-op probe, default-deny on any scalar decrease), and installs the
winner under a fresh name requiring reverification. This is NOT the v3
API-billed internal repair episode."

```json
{"type": "object",
 "properties": {
   "investigation_id": {"type": "string"}, "expected_revision": {"type": "integer"},
   "branch": {"type": "string"},
   "compile_id": {"type": "string"},
   "winner": {"type": "string"},
   "reading_id": {"type": "string",
     "description": "Stored reading to bind (REP-1). Omit to use the newest reading bound to this exact branch content."},
   "as_name": {"type": "string"}},
 "required": ["investigation_id", "expected_revision", "branch",
              "compile_id", "winner"]}
```

`dispatch_repair_transaction(runtime, records, args, turn) -> dict`:

1. `pre = host.check_repair_preconditions(branch=branch,
   reading_id_arg=str(args.get("reading_id") or ""), turn=turn)`. Blocked →
   return `pre["blocked"]` AND record it via
   `host._record_dispatch_result(name="repair_transaction", tu=<mcp tu>,
   turn=turn, payload=pre["blocked"])` (parse the returned string back for
   the envelope). This preserves REP-1/REP-2/SAT-1/SAT-3 exactly.
2. Look up the compile session by `compile_id` →
   `{"status":"error","reason":"unknown_compile_id"}` on miss.
   `session["branch"] != branch` → `compile_branch_mismatch`.
   `session["source_content_hash"] != pre["source_hash"]` →
   `{"status": "failed", "reason": "stale_compile",
   "note": "Branch content changed since this compile; rerun repair_hypotheses_test."}`.
3. Re-materialize evidence if this process didn't run the compile: if no
   entry in `host.episode_tool_calls` has `episode_id == compile_id`,
   rebuild `ToolCall(iteration=c["iteration"], tool_name=c["tool_name"],
   tool_use_id=c["tool_use_id"], arguments={}, result=c["result"],
   episode_id=c["episode_id"])` for each stored call and extend
   `host.episode_tool_calls`. (The ledger entry with snapshots is already in
   state — it persisted.)
4. Synthesize the worker-result equivalent:

   ```python
   winner = str(args["winner"])
   winner_edits = <the "edits" list of the entry in
       session["result"].get("finalists", []) whose installed_fork == winner;
       else of the item in session["result"].get("items", []) whose
       installed_fork == winner; else []>
   worker_result = {"applied": bool(session["changed_finalists"]),
                    "best_branch": winner, "edits": [str(e) for e in winner_edits],
                    "verdicts": [], "collateral": {},
                    "notes": "client-compiled finalists (repair_hypotheses_test)"}
   episode_payload = {"episode_id": compile_id, "status": "ok",
                      "result": worker_result}
   ```

   Binding property: `winner_edits` are copied from host-generated
   successful-composite output, so check 4 (`edit_claims_bound`) binds by
   construction; a winner that is not a compile fork fails check 3; a
   non-finalist winner with ≥2 changed forks fails check 5 — all through the
   UNMODIFIED check code.
5. `transaction_id`-bearing `base_record` exactly as the v3 dispatcher
   builds it (same keys, from `pre[...]`), plus two additive keys:
   `"compile_id": compile_id, "mode": "client_compiled"`.
6. `payload_str = host.validate_and_install_repair(tu=<mcp tu>, turn=turn,
   branch=branch, source_hash=pre["source_hash"], att_key=pre["att_key"],
   pair=pre["pair"], base_record=base_record,
   episode_payload=episode_payload,
   as_name=str(args.get("as_name") or f"repair_tx_{turn}_{branch}"))`.
7. Parse, add `"revision"`, return. Success payloads carry
   `status: "installed"`, the installed branch,
   `reverification_required: true`, the acceptance block, and the
   saturation summary — all from the shared implementation. Failures settle
   saturation through `_settle_repair_outcome` exactly as in v3 (evidence
   failures count toward the SAT-3 latch; process-class failures get the
   one-retry taxonomy).

---

## Part 9 — Policy-ledger conformance map (binding)

Enforcement locus for every ledger row that touches the server surface.
"host" = unmodified shared code reached through the server.

| Rows | v1 form | Server realization |
|---|---|---|
| GT-1 | hard | No path/benchmark params anywhere; `benchmark_context=None`; no `inspect_*`/`list_*` tools; registry creates inline INV investigations only. |
| GT-2 | hard | `host._dispatch_verify_run` with empty goal; empty toolset; candidate+language only. |
| GT-3 | hard | `validate_experiment_config` via host (`language` host-derived). |
| DECL-1/2/4/8 | hard | `AttestationPolicy` on the runtime executor; dispatcher-written records; clamps; unsolved ungated. |
| DECL-3 | hard | verify arity: server schema requires ONE `branch`; host re-checks. |
| DECL-5 | hard (analog) | one-call-per-request + terminal check in dispatch step 4. |
| DECL-6 | n/a | the server never synthesizes fallback declarations (no exhaustion terminal exists). |
| DECL-7 | active | `_resync_attestation_branch_on_rename` inside `_dispatch_episode_install` (reached via repair install). |
| REP-1..6 | hard/active | `check_repair_preconditions` + `validate_and_install_repair` + `_settle_repair_outcome` (shared implementation, Part 8). |
| REP-7 | **advisory** | No phase gate on the MCP repair path (glue calls the host methods directly, bypassing `handle_tool`'s phase check); WF-1 advisory block carries the workflow recommendation instead. |
| SAT-1/2/3 | preserved | shared saturation code; SAT-3 rendered ENFORCED in the brief. |
| SAT-4 | hard | reading_record duplicate suppression (Part 5.15 step 2) + host path for any episode-compiled reading. |
| DUP-1 | hard | `handle_tool` read cache for decode_show/adjudicate/diagnosis reads. |
| DUP-2 | telemetry | `handle_tool` signature counts → `repeated_call` events → events.jsonl (persisted with the next commit). |
| DUP-3 | telemetry | n/a structurally (no lead turns); recorded here as the parity note. |
| DUP-4 | hard | `_dispatch_episode_install` dedup-by-content (repair install path). |
| WF-1 | **advisory** | Part 6.1 line 1 — labeled recommendation, never a block. |
| WF-2 | **advisory/structural** | no `episode_run` tool exists; verify is callable in every state. |
| WF-3 | structural | the 23-tool surface IS the restriction; `V3_MAPPED_TOOL_NAMES` + RES-5 backstop. |
| WF-4 | **advisory** | Part 6.1 line 2 with the threshold-derived route + policy id. |
| WF-5 | **advisory** | Part 6.1 lines 3–4; budget-relative hints structurally n/a (no turn budget). |
| WF-6 | **divergence** | candidate_list/candidate_show portfolio; roles labeled; scalar is one signal (Part 5.11). |
| BUD-1 | hard (server spend) | host `max_cost_usd` + ceiling check in Part 7.2; client-side lead spend is the client harness's meter (ledger note). |
| BUD-2/3/4/5/6/7 | hard | inside `run_episode` (verify path). |
| EPI-1/2/3/6 | hard | episode runner + `_build_episode_workspace` (verify + repair compile isolation). EPI-4/5: verify-only surface; schema retry inside runner. |
| EXP-1/2/3/4 | hard/active | host experiment dispatchers; EXP-3's no-thread-adoption extended by the Part-3.4 lease (the §9.1 rule). |
| RES-1 | hard (server sends) | `call_with_rate_limit_retry` inside `run_episode`. RES-2/3: client-harness-owned, n/a. |
| RES-4 | analog | atomic commit + revision protocol (§3.3/§3.5), per ledger note. |
| RES-5 | hard | `set_available_tools(V3_MAPPED_TOOL_NAMES)` + server routing table. |
| CMP-1 | hard | comparison_record stores per-branch hashes (Part 5.16); v3 compare-binding untouched. |
| CMP-2 | **divergence** | best_partial vs accepts_as_solution split, both hash-bound (Part 5.16). |
| CMP-3 | hard | all hashes via `_branch_hash`/`_snapshot_content_hash` (pinned renderer). |
| CMP-4 | telemetry | roles in candidate_list + events. |
| CTX-1 | hard | status brief rebuilt from state every call; disk state IS resume (§3.2). |
| CTX-2/3/4 | active | section renderers with caps; external context its own section; evidence/board/queue rendered (Part 6). |
| CTX-5 | active | act_set_model_variant via handle_tool mirror; serialized. |
| POL-1 | **advisory / inert** | no benchmark context exists in v1 investigations, so the executor gate cannot fire; recorded as structurally inert until capsule provisioning (Phase D). |
| POL-2 | **advisory** | `NoGatesPolicy.finalize_guard` neutral form retained inside the executor (search tools only run inside experiments/compiles). |

This matches the ledger's totals: nothing hard is softened; exactly the 7
advisory rows and 2 divergences change form; telemetry rows land in
events.jsonl.

---

## Part 10 — Phase C onboarding files

All files below are new (or single-line additions) at the repo root unless
noted. Contents are normative — copy them verbatim modulo whitespace.

### 10.1 `.mcp.json` (Claude Code project discovery)

```json
{
  "mcpServers": {
    "decipher": {
      "command": "sh",
      "args": ["scripts/mcp_launch.sh"]
    }
  }
}
```

### 10.2 `.codex/config.toml` (Codex trusted-project discovery)

```toml
# Loaded by Codex after the user trusts this project.
# Personal-config fallback (troubleshooting): copy this block into
# ~/.codex/config.toml with an absolute path in args.

[mcp_servers.decipher]
command = "sh"
args = ["scripts/mcp_launch.sh"]
```

### 10.3 `scripts/mcp_launch.sh` (dependency-free launcher, §8.1.1)

```sh
#!/bin/sh
# Decipher MCP launcher. Dependency-free: starts the server when the
# environment is healthy, otherwise fails fast with a machine-readable
# bootstrap_required diagnostic on stderr (never a long build on stdio).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo '{"decipher_bootstrap_required": true, "reason": "missing_venv", "run": "sh scripts/bootstrap.sh"}' >&2
  exit 1
fi
if ! "$PY" -c "import mcp_server" >/dev/null 2>&1; then
  if ! PYTHONPATH="$ROOT/src" "$PY" -c "import mcp_server" >/dev/null 2>&1; then
    echo '{"decipher_bootstrap_required": true, "reason": "package_not_importable", "run": "sh scripts/bootstrap.sh"}' >&2
    exit 1
  fi
fi
cd "$ROOT"
exec env PYTHONPATH="$ROOT/src" "$PY" -m mcp_server "$@"
```

(`python -m mcp_server` rather than the console script so a source checkout
whose editable install predates the package still works.)

### 10.4 `scripts/bootstrap.sh` (idempotent clean-clone bootstrap, §8.1.1)

POSIX sh. Behavior contract (implement exactly; ~90 lines):

1. `ROOT` = repo root (as in 10.3). Serialize concurrent runs with
   `mkdir "$ROOT/.bootstrap.lock"` (atomic); if it exists and is older than
   30 minutes, remove and retry once; else exit 1 with
   `{"bootstrap": "locked"}` on stderr. `trap 'rmdir ...' EXIT`.
2. Find Python: first of `python3.12 python3.11 python3` whose
   `sys.version_info >= (3, 11)`; none → print
   `{"bootstrap": "failed", "layer": "prerequisite", "missing": "python>=3.11",
   "install": "https://www.python.org/downloads/ or brew install python@3.11"}`
   to stderr, exit 1. NEVER invoke sudo or a package manager.
3. Fingerprint = `shasum -a 256` (fall back `sha256sum`) over
   `pyproject.toml` and `rust/decipher_fast/Cargo.lock`, concatenated. If
   `.venv/.decipher_build_fingerprint` matches AND
   `.venv/bin/decipher --help` succeeds → print
   `{"bootstrap": "ok", "cached": true}` and exit 0.
4. `[ -d .venv ] || "$PYBIN" -m venv .venv`.
5. `.venv/bin/python -m pip install -q -e ".[providers,dev]"` — failure →
   `{"bootstrap": "failed", "layer": "python_deps"}` + the LAST 15 lines of
   pip output to stderr, exit 1.
6. Rust kernels (required accelerator per `fast_kernel_status`): if
   `cargo` is on PATH → `.venv/bin/python -m pip install -q maturin` and
   `(cd rust/decipher_fast && ../../.venv/bin/python -m maturin develop
   --release)`; failure → `{"bootstrap": "failed", "layer": "rust_build"}` +
   last 15 lines, exit 1. If `cargo` is MISSING → print
   `{"bootstrap": "degraded", "layer": "prerequisite", "missing": "cargo",
   "install": "https://rustup.rs (curl https://sh.rustup.rs -sSf | sh)",
   "note": "continuing without Rust kernels; some solvers are slow/unavailable"}`
   to stderr and continue (approval-first rule: report, don't install).
7. Write the fingerprint file; run
   `.venv/bin/decipher doctor --json > /dev/null` (failure →
   `{"bootstrap": "failed", "layer": "health_check"}`, exit 1).
8. Print `{"bootstrap": "ok"}` to stdout, exit 0.

### 10.5 `docs/mcp_onboarding.md` (the closed onboarding set, §8.1.2)

One document with five titled sections (the §8.1.2 roles):

1. **Operator quickstart** — the three-line experience, what the launcher
   does, `sh scripts/bootstrap.sh`, the reconnect note ("after the first
   bootstrap, reconnect MCP or start a new session"), health check
   (`.venv/bin/decipher doctor --json`), where the registry lives, and the
   recovery table: missing venv → bootstrap; project not trusted (Codex) →
   trust then reload; no API key → Part-7.3 behavior; stale build →
   bootstrap (fingerprint); interrupted investigation → it is on disk,
   `investigation_list` + `investigation_status` resume it from either
   client.
2. **Investigation methodology** (canonical; both client adapters point
   here) — the adapted v3 brief. Normative text:

   > You are the strategist of a cipher investigation; the Decipher MCP
   > server is your instrument bench and evidence store.
   > - **The brief is the source of truth.** `investigation_status` rebuilds
   >   the whole picture from server state every call — cipher, measured
   >   fingerprint, branch cards, hypothesis board, readings, experiments,
   >   evidence, and a rotating decode window. Never re-derive facts it
   >   already states; call it again whenever a result changes state (the
   >   `revision` you pass with mutating calls comes from it).
   > - **Work at the hypothesis level.** Form a hypothesis about the cipher
   >   family, create a branch for it (`hypothesis_branch_create`), test it,
   >   record evidence for and against, and reject it when settled. Keep
   >   several live candidates; `candidate_list` shows every branch with
   >   labeled signals — no single score defines the leader. Trust decoded
   >   text over any single number.
   > - **Measure before solving.** `observe_diagnosis` runs the LLM-free
   >   family diagnosis; an alphabet much larger than 26 means homophonic,
   >   and its constraint scores naturally sit below 1.0.
   > - **Long solver work is an experiment.** `experiment_submit` runs the
   >   no-LLM solver stack in the background; collect and, if good, install
   >   with `experiment_collect`. Never resubmit an identical config —
   >   duplicates are suppressed.
   > - **Read, then repair, then reverify.** When a candidate partly reads:
   >   record your reading (`reading_record`), compile concrete word
   >   hypotheses (`repair_hypotheses_test`), then ask the host to validate
   >   and install one winner (`repair_transaction`). The host rejects
   >   unsupported edits and any scoring regression; after two failed
   >   repair rounds on the same evidence it latches exhausted — broaden
   >   instead of polishing.
   > - **Verify before declaring.** `request_independent_verification` has
   >   a fresh reader judge your branch. Declaration
   >   (`meta_declare_solution`) is hard-gated on a positive fresh
   >   verification of the exact current content. Honest surrender
   >   (`meta_declare_unsolved`) is always available and never blocked.
   > - The host guidance block in the brief is advisory (policy ids shown);
   >   you may deviate with reason, except lines marked ENFORCED.
3. **MCP capability reference** — the 23 tools, one line each (name, class,
   one-sentence purpose). Generated by hand from Part 5; keep in sync with
   `tools.py`.
4. **Privacy and publication** — private ciphertext stays local (registry
   under `~/.config/decipher/`); nothing is published or sent anywhere
   except verify-episode candidate text to the configured model provider;
   note that `request_independent_verification` sends the candidate decode
   to the provider and how to run keyless if that is unacceptable.
5. **Contributing** — condensed §8.1.5: preserve the investigation
   directory + commit id, redact ciphertext/keys/candidates by default, ask
   the user before any issue/PR, use a separate branch/worktree for code
   changes.

### 10.6 `AGENTS.md` addition (doctrine pointer)

Insert immediately after the `## What This Is` section:

```markdown
## Cracking a cipher (MCP quick path)

This repo ships an MCP server exposing the investigation surface. A checked-in
`.codex/config.toml` / `.mcp.json` wires it up once the project is trusted;
`sh scripts/bootstrap.sh` prepares a fresh clone. Methodology, tool
reference, and recovery live in **`docs/mcp_onboarding.md`** — read that
(not this file's development notes) when the task is *cracking a cipher*
rather than developing Decipher. After an investigation exists,
`investigation_status` is the authoritative briefing; do not treat onboarding
prose as live investigation state.
```

### 10.7 `CLAUDE.md` pointer line

Insert after the `## What This Is` block, one paragraph:

```markdown
**Cracking a cipher via MCP:** a checked-in `.mcp.json` exposes the
`decipher` MCP server (bootstrap: `sh scripts/bootstrap.sh`). Doctrine and
tool reference: `docs/mcp_onboarding.md`.
```

### 10.8 Codex prompt — `docs/prompts/decipher-investigate.md`

Canonical file (checked in), installed by the user (or the agent, with
approval) via
`mkdir -p ~/.codex/prompts && cp docs/prompts/decipher-investigate.md ~/.codex/prompts/`.
Content:

```markdown
Investigate a cipher with the Decipher MCP server.

If the `decipher` MCP server is not connected: run `sh scripts/bootstrap.sh`,
then reconnect (trust the project if prompted) and retry.

If I did not paste a cipher yet, ask me to paste it (or name a local text
file for you to read and pass inline — the server takes no file paths).
Then: call `investigation_start` with the inline ciphertext; read
`docs/mcp_onboarding.md` §Investigation methodology; and drive the
investigation from `investigation_status`, following its advisory guidance
until a verified declaration or an honest unsolved.
```

### 10.9 `README.md` quickstart insert

Add under `## Quick Start` (before existing content), verbatim:

```markdown
### Zero-effort agent quickstart (Claude Code / Codex)

    git clone <this-repository>
    cd decipher
    codex          # or: claude
    > I would like to crack a cipher.

The checked-in `.mcp.json` (Claude Code) and `.codex/config.toml` (Codex,
after you trust the project) launch the Decipher MCP server via
`scripts/mcp_launch.sh`. On a fresh clone the launcher reports
`bootstrap_required`; the agent (or you) runs `sh scripts/bootstrap.sh`
once — it creates `.venv`, installs Decipher, builds the Rust kernels if
`cargo` is present, and health-checks with `decipher doctor --json`. Then
reconnect and paste your cipher. Without any API key everything works
except independent verification: you still get diagnosis, background
solvers, repairs, and your best candidate labeled "promising but not
independently verified" (see `docs/mcp_onboarding.md`).
```

Exact user steps on a bare clone (normative walkthrough the docs must
match): (1) clone, (2) `cd decipher`, (3) start `codex`/`claude` and trust
the project, (4) say "I would like to crack a cipher", (5) approve the
agent running `sh scripts/bootstrap.sh` when MCP reports
bootstrap_required, (6) reconnect MCP / new session, (7) paste ciphertext;
the agent calls `investigation_start` and proceeds. No config editing, no
build-command discovery, no architecture docs.

### 10.10 `TOOLS.md` appendix

Append a short section "MCP server surface (v1)" listing the 23 tool names
with one-line descriptions and a pointer to `docs/mcp_onboarding.md` §3 and
this spec. (CLAUDE.md's TOOLS.md-currency rule extends to this surface.)

---

## Part 11 — Tests

New files; all runnable in the standard env
(`PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`); zero API spend
(scripted sessions via `register_session_builder`, synchronous experiment
queue via ctor flag). Existing baseline 1708/2 must be preserved untouched.

Shared fixture (in `tests/support/mcp.py` or top of the main test file):
`make_server(tmp_path, *, verify="none") -> DecipherMCPServer` constructing
the server core directly (no subprocess) with
`registry_dir=tmp_path/"registry"`, `synchronous_experiments=True`, and a
tiny letters cipher (reuse the style of existing loop_v3 fixtures, e.g. a
short English substitution text). Helper `call(server, name, **args)`
returning the parsed JSON body.

### 11.1 `tests/test_mcp_protocol.py` (subprocess; scripted stdio client)

Spawn `[sys.executable, "-m", "mcp_server", "--registry-dir", tmp,
"--verify-provider", "none"]` with `env={**os.environ, "PYTHONPATH": "src"}`,
pipes for stdio, text mode. A tiny client helper writes one JSON line and
reads one response line (with a 30 s timeout guard).

1. `initialize` → protocolVersion echoed for "2025-06-18"; serverInfo.name
   == "decipher"; tools capability present.
2. Unsupported requested version ("1999-01-01") → server responds with
   "2025-06-18".
3. `notifications/initialized` → no response line (verify by immediately
   sending `ping` and getting the ping response next).
4. `tools/list` → exactly 23 tools; every entry has name/description/
   inputSchema; names match the Part-5 list; `investigation_start` schema
   requires `ciphertext`.
5. `tools/call` `investigation_list` → `isError` false, JSON body with
   `investigations: []`.
6. `tools/call` unknown tool → `isError` true.
7. Malformed JSON line → `-32700` error with id null; server still answers
   a subsequent `ping`.
8. Unknown method with id → `-32601`.
9. stdout purity: every line the server emitted parses as JSON.
10. EOF (close stdin) → process exits 0 within the timeout.

### 11.2 `tests/test_mcp_registry.py` (in-process)

1. create → load round-trips meta+state; revision 1; files exist;
   `meta.json` matches.
2. commit bumps revision and `updated_at`; `investigation.json` replaced
   atomically (no `.tmp` residue).
3. Revision conflict: server-level — a mutating call with
   `expected_revision` one behind returns
   `status=conflict/revision_mismatch` and does NOT mutate (state file
   unchanged, revision unchanged).
4. Lease: two `InvestigationRegistry` instances on the same dir —
   `acquire_lease` succeeds in the first, fails in the second;
   server-level mutating call through a second server instance returns
   `writer_lease_held`. After closing the first's fd, the second acquires.
5. `list` orders by `updated_at` desc and honors limit.
6. Unknown id → `investigation_not_found` typed result.
7. Terminal blocking: after a declared-unsolved investigation, a mutating
   call returns `investigation_terminal`; `investigation_status` still
   works.

### 11.3 `tests/test_mcp_tools.py` (in-process; the scripted investigation)

Intake:
1. letters format: start with plain text → alphabet/tokens/words match the
   `cmd_crack` path for the same input; brief contains "## Cipher".
2. canonical: S-token input parses via `parse_canonical_transcription`;
   auto-detection picks canonical for S-token text and letters otherwise.
3. empty / oversized ciphertext → typed errors.

Status and reads:
4. `investigation_status` brief contains the section headers of Part 6 and
   the advisory header with "[WF-1 advisory]" and "[DECL-1 ENFORCED]"
   lines; `verification_available` false under verify="none" and the
   UNAVAILABLE advisory line present.
5. `observe_overview` contains fingerprint text; `observe_diagnosis`
   returns a parsed diagnosis dict.
6. DUP-1: two identical `decode_show` calls → second returns
   `duplicate_suppressed`.
7. `hypothesis_branch_create` → revision bumps; card appears in the next
   brief; `hypothesis_next_steps` result carries `advisory: true` and
   `policy_ids: ["WF-1"]`.
8. `candidate_list`: entries carry scores/roles/verification fields;
   `verification == "unavailable"` keyless; `candidate_show` returns
   decoded_text and attestation_history.

Records:
9. `reading_record` happy path → reading_id; stored in state.readings with
   `source == "client:unknown"`; hash bound. Second reading on unchanged
   content+evidence → `duplicate_reading_suppressed` with the existing id
   (SAT-4).
10. `comparison_record` stores hashes; `best_partial` may be null while
    `accepts_as_solution` false (CMP-2 split); invalid branch → error;
    evidence entry appears in the brief.

Experiments (synchronous queue):
11. `experiment_submit` invalid config → the EXP-1 corrected_example error
    payload passes through; valid submit runs inline and
    `experiment_collect` returns the packet; duplicate submit →
    deduplicated.

Verification and gates:
12. keyless: `request_independent_verification` → `unavailable`,
    revision UNCHANGED; `meta_declare_solution` → blocked
    `attestation_required` (the exact keyless surface of Part 7.3).
13. scripted verify: `register_session_builder("episode:verify", <fake
    returning a positive _VERIFY_SCHEMA result>)` (pattern from
    `tests/test_verify_attestation.py`) + a runtime built with a dummy
    provider object → attestation written with the branch's current hash;
    `meta_declare_solution` now succeeds; meta.status == "solved";
    subsequent mutation blocked `investigation_terminal`.
14. negative scripted verify → declare blocked `attestation_not_positive`
    with the echoed verdict; repair agenda seeded only when
    repairability == "local_repair".
15. `meta_declare_unsolved` always succeeds (DECL-8), terminalizes.

Repair (the Part-8 path; craft a small cipher where a word hypothesis
produces a changed fork — reuse the fixture style of
`tests/test_hypothesis_actions.py`):
16. `repair_hypotheses_test` returns compile_id + changed_finalists;
    identical re-compile → duplicate_suppressed; the synthetic
    `repair_compile` ledger entry exists in state.
17. `repair_transaction` without a prior `reading_record` →
    `fresh_reading_required` (REP-1). With a reading: happy path installs
    the winner (`status installed`, `reverification_required` true,
    acceptance checks all passed) and the installed branch exists.
18. Stale compile: mutate the branch (e.g. a second installed repair or a
    hypothesis fork changing content) → `stale_compile`.
19. Winner not a compile fork → failed `unsupported_winner` /
    `winner_fork_from_failed_call` per check order; duplicate
    source+interpretation pair → `duplicate_suppressed` (REP-2); two
    evidence failures latch SAT-3 → third attempt blocked
    `repair_saturated` and the brief shows the `[SAT-3 ENFORCED]` line.

Host refactor pin (belt and braces; the real pin is the untouched existing
suite):
20. `check_repair_preconditions` on a branch with no reading returns
    `{"blocked": {... "reason": "fresh_reading_required" ...}}` — called
    directly on a v3-style host from an existing loop_v3 test fixture.

### 11.4 `tests/test_mcp_onboarding.py` (fresh-clone simulation)

Skip (`pytest.mark.skipif`) when `git` is unavailable.

1. Build the "clone": `git -C <repo> ls-files -z` → copy each tracked file
   into `tmp/clone` (tracked files only — simulates a bare clone of the
   working tree deterministically, including in-flight changes).
2. Assert onboarding files exist in the clone and parse: `.mcp.json`
   (json; `mcpServers.decipher.args == ["scripts/mcp_launch.sh"]`),
   `.codex/config.toml` (contains `[mcp_servers.decipher]`),
   `scripts/mcp_launch.sh`, `scripts/bootstrap.sh`,
   `docs/mcp_onboarding.md`, `docs/prompts/decipher-investigate.md`.
3. Launcher degradation: run `sh scripts/mcp_launch.sh` in the clone (no
   `.venv` there) → exit nonzero fast (<10 s), stderr contains
   `bootstrap_required`.
4. Server boots from the clone source: spawn
   `[sys.executable, "-m", "mcp_server", "--registry-dir", tmp_registry,
   "--verify-provider", "none"]` with `cwd=tmp/clone`,
   `PYTHONPATH=<clone>/src` (only) — the test venv supplies third-party
   deps; imports of decipher code must resolve from the clone.
5. Through the stdio client: initialize → tools/list (23) →
   `investigation_start` with a small inline letters cipher →
   `investigation_status` (brief non-empty, mentions the cipher stats) →
   `decode_show` on "main" → `meta_declare_solution` on "main" with the
   required fields → assert blocked with reason `attestation_required`
   (the gate holds on a bare clone with zero keys). Close stdin; exit 0.

Estimated new tests: ~55. Expected post-land suite: ≥1763 passed / 2
skipped, with the original 1708 unchanged.

---

## Part 12 — Implementation order and acceptance

Land as one branch, reviewed in this order (each step keeps the suite
green):

1. Part 8.1 host refactor alone; run the FULL suite → must be 1708/2 with
   zero test edits (any needed retarget is a review flag, not a silent
   fix).
2. Parts 1–4 (package, protocol, registry, runtime) + tests 11.1/11.2.
3. Part 5 tools + Part 6 status + Part 7 verify + Part 8.2/8.3 repair +
   tests 11.3.
4. Part 10 onboarding files + Part 11.4 + TOOLS.md/README/AGENTS/CLAUDE
   edits.

Acceptance checklist (all must hold):

- [ ] Full suite green; baseline tests untouched.
- [ ] `loop_v3.py`, `state.py`, `context.py`, `episodes.py`,
      `experiments.py`, `actions.py`, `tools_v2.py` have zero diffs.
- [ ] `host.py` diff is exactly the Part-8.1 extraction (mechanical; a
      reviewer can verify by code motion).
- [ ] `grep -r "benchmark" src/mcp_server/` shows no benchmark loader or
      path parameter usage (intake imports `parse_canonical_transcription`
      only).
- [ ] Every tool result and the whole brief contain no API key material
      (spot-check via test 11.3 by asserting the key string never appears —
      trivially true keyless).
- [ ] Part 9 conformance table cross-checked against the diff in review:
      the 7 advisory + 2 divergence rows and NOTHING else changed form.
- [ ] The fresh-clone simulation passes on macOS (darwin is the dev
      platform; Linux CI parity is expected but not gating in this slice).
