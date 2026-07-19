# Spec — `decipher investigation`: the structured investigation CLI

Status: DRAFT for implementation; external interface review incorporated
2026-07-19. Author: Fable (main loop), 2026-07-19.
Origin: external agent evaluation (GPT-5.6-sol, fresh session, 2026-07-19)
reviewed the repo independently and concluded the missing piece is "first-class,
machine-readable access to the investigation state machine that MCP already
exposes." Its property list is adopted below as binding requirements. Sibling
context: `docs/mcp_dual_harness_proposal.md`, `docs/mcp_onboarding.md`.

---

## 1. Why

- **Shell-first agents are a significant client class.** Both the naive-Codex control
  (cloned cDecryptor rather than discover our MCP surface) and the Sol
  evaluation (CLI-first, MCP only for extended investigations) show that
  agents with shells often reach for CLIs. The sample is too small for a
  population claim, but it is enough to justify a structured surface with zero
  session setup.
- **It structurally eliminates the staleness class of failures.** Every CLI
  invocation loads current code: no long-lived stdio server, no cached tool
  schemas, no restart-after-pull, no `server_code` handshake. The entire
  2026-07-19 stale-server incident class disappears on this surface.
- **It is cheap because of the host extraction.** The MCP server is a thin
  JSON-RPC skin over `InvestigationHost` + the registry. This CLI is a second
  thin skin over the same objects. No new capability, no new state, no new
  gates — transports only.
- **Phase-D evidence.** If agents prefer this surface, the durable asset is
  confirmed as host+gates+registry, with transports as commodity skins.

## 2. Non-goals

- No new solving or investigative capability. Verbs map 1:1 onto existing
  host operations.
- No replacement of `decipher crack`/`benchmark`/`diagnose` (the one-shot
  paths stay as they are).
- No daemon. The registry on disk IS the persistence; processes are
  short-lived.
- No interactive/TTY affordances (this surface is for programs; humans have
  the existing CLI and the narrate display).
- No second domain implementation. CLI-only work is limited to transport
  concerns: argv/JSON decoding, local file intake, process lifecycle, privacy
  opt-in, and exit-code mapping.

## 3. Surface

One verb per state transition, `investigation` namespace, JSON out always
(`--json` is not a flag; structured output is the contract — a `--pretty`
flag MAY add human formatting later, JSON stays the default):

```
decipher investigation start   --ciphertext-file F | --ciphertext "..." [--language X] [--label L]
decipher investigation list    [--limit N]
decipher investigation status  ID
decipher investigation overview ID
decipher investigation diagnose ID [--branch B] [--max-period N]
decipher investigation decode  ID --branch B [...]
decipher investigation next-steps ID
decipher investigation candidates ID            # candidate_list
decipher investigation candidate ID --branch B  # candidate_show
decipher investigation adjudicate ID --branches-json '["B1","B2"]'
decipher investigation branch-create ID --revision R ...
decipher investigation branch-update ID --revision R ...
decipher investigation branch-reject ID --revision R ...
decipher investigation experiment-submit ID --revision R --type T --config-json JSON [--wait|--detach]
decipher investigation experiment-collect ID --revision R --experiment E ...
decipher investigation reading-record ID --revision R ...
decipher investigation comparison-record ID --revision R ...
decipher investigation repair-test ID --revision R ...
decipher investigation repair-transaction ID --revision R ...
decipher investigation verify ID --revision R --branch B
decipher investigation set-model-variant ID --revision R ...
decipher investigation declare-solution ID --revision R ...
decipher investigation declare-unsolved ID --revision R ...
```

That is the 23-operation MCP surface, verb-for-tool. Argument names mirror the
MCP tool schemas exactly (§6 makes this mechanical, not conventional).

The friendly verb is explicit public metadata in the operation manifest; it
is NOT inferred from underscores. Several names deliberately differ from a
mechanical conversion (`investigation_start` → `start`, `observe_overview` →
`overview`, `request_independent_verification` → `verify`, and
`meta_declare_solution` → `declare-solution`). Renaming a friendly verb is a
CLI breaking change. The canonical MCP operation name is permanent and is
also callable through the lossless escape hatch:

```
decipher investigation call OPERATION --input-json JSON
decipher investigation call OPERATION --input-file FILE
```

`call` guarantees immediate coverage for schemas that are awkward to express
as flags and gives tests a transport-neutral path. `--input-file -` reads one
JSON object from stdin. Friendly verbs and `call` dispatch the same operation;
neither may contain operation-specific business logic.

Global transport options precede the verb and are not operation arguments:

```
decipher investigation [--registry-dir DIR] [--verify-provider P]
  [--verify-model M] [--max-cost-usd N] [--allow-external] VERB ...
```

`--registry-dir` follows the existing explicit flag →
`DECIPHER_MCP_REGISTRY` → config-default precedence. A verification provider
is NEVER inferred merely because an API key happens to be present: the
structured CLI is keyless unless `--verify-provider` is explicit. Provider
credentials retain the existing provider-specific lookup behavior after that
explicit selection.

### 3.1 Binding input contract

1. `--input-json`, `--input-file`, and friendly flags all build one object
   that is validated against the operation's manifest schema by the shared
   service. Supplying more than one input mode is an argument error; there is
   no precedence rule that could silently discard data.
2. Scalar schema properties become `--kebab-case` flags. Booleans use
   `--flag` and `--no-flag` only when the schema has a default; required
   booleans require an explicit value. Enums remain constrained by argparse
   and by service validation.
3. Arrays, objects, union-typed values, and nested structures accept JSON
   values (`--fragments-json`, `--hypotheses-json`, etc.) or the canonical
   `call --input-file` path. Do not invent lossy comma splitting. Convenience
   repeated flags MAY be added only when they round-trip every legal value.
4. Positional `ID` and friendly names such as `--revision` are aliases for the
   exact schema fields `investigation_id` and `expected_revision`. The
   translated object is what validation sees.
5. `start --ciphertext-file F` is a transport-only alias that reads UTF-8 and
   supplies the schema's `ciphertext` property. It is mutually exclusive with
   `--ciphertext` and JSON-object input. `--ciphertext-file -` reads the raw
   ciphertext from stdin. File-read/UTF-8 failures are typed CLI input errors;
   ciphertext size and format failures remain domain results from the service.

### Binding output/exit contract (adopted from the Sol checklist)

1. **Stable JSON** on stdout: exactly the MCP tool-result body (the same dict
   the server would wrap), plus `revision` injection identical to the server.
   Diagnostics/progress go to stderr, never stdout. Every normal invocation,
   including failures, emits exactly one JSON object followed by `\n`. Internal
   errors emit `{"status":"error","reason":"internal_error"}` without
   secrets or a traceback on stdout; a traceback MAY go to stderr in debug
   mode.
2. **Stable exit codes**: `0` operation completed (including an honest
   `declare-unsolved`); `1` domain/lookup/unavailable failure; `2` invalid
   CLI or schema arguments; `3` blocked
   (`investigation_terminal`, `writer_lease_held`, gate refusals — the JSON
   `reason` is the identifier, exit code just classifies); `4` revision
   conflict; `5` internal error. Conventional signal exits remain conventional
   (for example `130` for SIGINT) after best-effort state finalization. Exit
   codes classify; the machine-readable
   identifier is always the JSON `reason` field. Existing domain outcomes
   preserve the server's strings verbatim (`attestation_required`,
   `revision_mismatch`, ...); transport policy may add a documented shared
   reason such as `external_call_not_authorized`.
   A single shared `result_to_exit_code` table is exhaustive over service
   status classes; tests cover at least not-found, parse failure, unavailable
   verification, blocked gate, lease-held, conflict, and internal exception.
3. **Optimistic concurrency**: every mutating verb REQUIRES `--revision R`
   (maps to `expected_revision`); reads never take it. Same conflict semantics
   as MCP.
4. **Keyless/local by default**: `verify` with no provider configured returns
   the same structured `no_verification_provider` refusal as the server;
   sending candidate text to an external provider requires an explicitly
   selected provider AND `--allow-external` on that invocation (explicit
   per-call opt-in — stricter than MCP, deliberately, because CLI callers may
   be scripts). This applies to EVERY external-call path, including
   `repair-transaction --verifier-arbitration`, not only `verify`. The
   operation manifest marks possible external effects. For `verify`, the CLI
   privacy guard runs before provider construction or domain dispatch. For
   conditionally external operations such as repair, provider construction is
   lazy and the same guard runs immediately before the actual provider call;
   a mechanically accepted repair does not require irrelevant external-call
   authority. Resolution order is stable: no explicitly selected provider →
   `no_verification_provider`; provider selected but no `--allow-external` →
   `external_call_not_authorized`; both present → the provider may be called.
   Do not silently downgrade a requested arbitration or verification after an
   external call becomes necessary.
5. **Persistent IDs**: the registry's investigation ids; nothing session-scoped.

## 4. Concurrency and lease semantics

Reuse the registry exactly. A mutating verb acquires the per-investigation
flock lease at dispatch and releases it explicitly in a `finally` block before
process exit. Process teardown is a fallback, not the release API. The shared
service accepts a lease-lifetime policy: MCP retains its current session-held
lease while CLI uses invocation-held leases. Consequences to document and test:

- CLI mutation while a live MCP server holds the lease → `writer_lease_held`
  + holder hint, exit 3. Correct: same investigation, one writer.
- CLI↔CLI races → the lease serializes; the loser gets the same block.
- Reads never need the lease (unchanged).

Before joining the path, validate `investigation_id` against the generated-id
grammar and verify the resolved directory remains under the registry root.
This closes path traversal for BOTH transports. Treat that as a deliberate
shared hardening change rather than hiding it inside the nominally
behavior-preserving I-0 extraction.

## 5. Experiments without a resident process

The MCP server runs the experiment queue inside its long-lived process; a CLI
process is short-lived. Two modes, explicit:

- `--wait` (default): submit through the normal asynchronous queue, COMMIT the
  pending/running record first, then wait in this process, harvest, and COMMIT
  the terminal experiment record before releasing the lease. The command emits
  the ordinary `experiment_submit` result body once and exits after the second
  commit, with `revision` set to the FINAL commit so the caller can immediately
  collect without an avoidable conflict. This two-commit rule makes `status`
  useful while a caller has put the command in the shell background with `&`;
  do not use the current single synchronous dispatch transaction, which would
  hide the running record until completion. The two transport commits are the
  sole intentional temporal difference from MCP's immediate async response;
  all domain fields and gates remain shared.
- `--detach`: the parent does not acquire the lease or mutate state. It spawns a
  private `decipher investigation _run-experiment` worker in a new session and
  receives a one-shot readiness message over a private pipe. The CHILD
  acquires the lease, revision-checks, submits, commits the running record, and
  only then sends the exact submit result/error body to the parent. The parent
  prints that body and exits; the child keeps the lease, harvests and commits
  the terminal result, then exits. Redirect the child's stdio and put
  diagnostics in the investigation event log. This handshake avoids the
  fork/flock race and prevents reporting an experiment id that was never
  durably submitted. The detached response contains the submission revision;
  because the child later commits completion, callers MUST refresh with
  `status` before a later mutation such as `experiment-collect`.

SIGINT/SIGTERM in a waiting worker performs a best-effort poll, marks unfinished
records `orphaned` with a typed reason, commits, and releases the lease. SIGKILL
cannot do that, so shared runtime startup must reconcile a persisted
`pending`/`running` record when the lease is free but the newly built queue has
no corresponding live worker: mark it `orphaned` before accepting a new
mutation. This reconciliation applies to MCP crash recovery too and prevents
deduplication from making a dead experiment permanently unresubmittable.

Cross-surface invariant: an experiment submitted by one surface must be
collectable from the other (the registry record format is already shared).
Test this explicitly.

## 6. THE LOCKSTEP CONTRACT (policy, mechanical)

Owner decision 2026-07-19: **both interfaces extend in lockstep, enforced by
construction, not by convention.**

- **I-0 extracts a transport-neutral service layer.** The dispatch pipeline in
  `src/mcp_server/server.py::_dispatch` (validate → resolve → terminal-check →
  lease → revision-check → dispatch → commit → revision-inject) moves to a new
  `src/investigation_service/` module (or `mcp_server/service.py`; implementer
  chooses, spec review ratifies) with ZERO behavior change — the same
  byte-parity discipline as the original host extraction (`5f269ac`). The MCP
  server and the CLI both call it. The MCP server keeps only JSON-RPC framing;
  the CLI keeps only argv↔dict mapping.
- **One operation manifest.** `src/mcp_server/tools.py`'s tool list becomes
  the single OPERATION MANIFEST consumed by BOTH skins: the MCP server derives
  its tools/list from it (as today), and the CLI auto-registers one verb per
  manifest entry. Each `OperationSpec` contains the canonical name, schema,
  description, operation class (`create`/`read`/`mutate`), stable friendly CLI
  verb, and external-effect metadata (including whether the effect is
  unconditional or depends on input/runtime outcome). This folds the current
  parallel `TOOL_CLASSES` map into the manifest so class metadata cannot drift. Schema
  → argparse conversion follows §3.1 and always retains the canonical `call`
  path. Adding an
  operation to the manifest EXTENDS BOTH SURFACES in the same commit, with no
  second implementation site.
- **Parity test**: `tests/test_interface_parity.py` asserts (a) every manifest
  operation has a registered friendly CLI verb and vice versa, excluding the
  reserved transport verbs `call` and private `_run-experiment`; (b) the CLI's
  generated
  argument set covers every schema property OR losslessly routes it through
  the canonical JSON input path; (c) friendly flags and canonical `call`
  produce identical service input; and (d) for a scripted investigation, the
  JSON body returned via the CLI equals the body returned via MCP. Stateful
  mutation sequences use separate registries with injected deterministic
  clock/id/provider/runner seams, or explicitly normalize declared volatile
  fields; do not call the same mutation twice against one registry and call
  that parity. Extending one surface without the other fails CI.
- **Scope clarification (experiment types are NOT operations).** The parity
  contract is at the OPERATION level (the ~23 verbs). New solver
  `EXPERIMENT_TYPES` values are NOT new operations — `experiment_submit` is a
  single operation whose `type` is a string arg (and the CLI passes the whole
  config as one `--config-json JSON` blob). A new experiment type therefore
  reaches
  BOTH surfaces automatically because both dispatch `experiment_submit` through
  the shared service layer that reads `EXPERIMENT_TYPES`; it is NOT gated by
  the manifest or per-type parity assertions. So the composite/polygraphic
  programs add `EXPERIMENT_TYPES` entries (dual-surface for free post-I-0), not
  manifest entries.
- **Doctrine line** (add in this program, small docs commit): AGENTS.md and
  CLAUDE.md each get one sentence under their MCP/architecture sections:
  "The MCP tool list is the operation manifest for BOTH the MCP server and
  `decipher investigation`; surface changes happen in the manifest and are
  enforced by tests/test_interface_parity.py."

## 7. Gates and invariants (unchanged, verified)

All provenance-ledger controls ride the service layer, which rides the host:
DECL-1 hash-bound declaration, GT-1..3 firewall, REP-* repair validation,
lease/revision integrity. The CLI adds NO new bypass: `declare-solution`
without a fresh positive attestation gets the same structured refusal.
Ledger impact: no gate changes form; add one ledger note that the surface set
grew (transport row, not a control row).

### 7.1 Implementation topology

The implementation is expected to touch these ownership points:

- `src/mcp_server/server.py`: reduce `DecipherMCPServer` to protocol framing,
  session lifecycle, and a call into the shared service. Preserve MCP's
  session-held runtime/lease behavior.
- `src/mcp_server/tools.py` (or a new transport-neutral manifest module):
  replace the parallel definitions/`TOOL_CLASSES` structures with
  `OperationSpec` records and keep `render_tool_list()` as an MCP projection.
- `src/mcp_server/registry.py`: add an idempotent explicit
  `release_lease(investigation_id)`/`close()` API and root-containment/id
  validation. Do not make CLI code close private registry file descriptors
  directly.
- `src/mcp_server/runtime.py`: expose the lifecycle seams needed to submit,
  persist, wait/harvest, reconcile orphaned records, and finalize without
  duplicating `ExperimentQueue` transitions in CLI code.
- `src/mcp_server/verify.py` and `src/mcp_server/repair.py`: accept the shared
  external-call authorization/provider-factory policy at the last responsible
  moment. MCP passes its current permissive server policy; CLI passes explicit
  per-invocation authority.
- `src/investigation_service/` (preferred over placing the shared layer under
  an MCP-named package): own validation, resolution, terminal/revision checks,
  dispatch, commit, result revision, execution policy, and typed outcomes.
- `src/cli.py`: register the `investigation` namespace and delegate to a small
  CLI adapter module; do not add 23 hand-written domain handlers to this
  already-large file. The top-level Rust-kernel preflight must exempt the
  `investigation` namespace, as it already exempts `mcp-serve`; read/start
  operations remain usable without Rust, while an experiment that truly needs
  a missing kernel returns its existing typed capability failure.
- `tests/test_interface_parity.py`: manifest, argv/JSON-input, body parity,
  privacy, exit-code, and deterministic cross-transport assertions.
- Focused registry/experiment tests (existing files or new dedicated files):
  explicit release, containment, two-process lease collision, wait/detach
  handshakes, signal cleanup, crash reconciliation, and cross-surface collect.

Argparse normally prints usage text and raises `SystemExit`. The investigation
adapter must catch/replace that behavior so parser errors also honor the
one-JSON-object stdout contract; human usage text may go to stderr.

## 8. Milestones

- **I-0**: service-layer extraction, byte-parity, suite green. (The risk
  milestone; everything else is mechanical.) **PURE extraction only** — no
  behavior change on the MCP path, no new tests beyond what proves the existing
  MCP suite still passes through the extracted layer. The shared-hardening
  items in §4/§5/§7.1 are DEFERRED to their assigned later milestones and must
  NOT ride I-0's byte-parity claim: id-containment/path-traversal → I-2,
  crash/orphan reconciliation → I-3, explicit `release_lease`/`close` API → I-2
  (needed first by the CLI's invocation-held lease). I-0 acceptance = the full
  MCP test suite is byte-identical-green over the extracted service layer.
- **I-1**: read verbs (`list/status/overview/diagnose/decode/next-steps/
  candidates/candidate/adjudicate`) via manifest auto-registration, plus
  canonical JSON input and the `call` escape hatch.
- **I-2**: mutations + `--revision` + lease-per-command semantics + exhaustive
  output/exit-code contract + investigation-id containment hardening.
- **I-3**: experiments (`--wait` two-commit lifecycle, crash reconciliation,
  cross-surface collect test).
- **I-4**: detached-worker handshake and signal cleanup.
- **I-5**: `verify` and verifier-arbitrated repair (keyless refusal,
  `--allow-external` opt-in, explicit provider selection) + declarations
  through the gates.
- **I-6**: parity test hardening, onboarding §"CLI mode" (the operator recipe:
  no restart ever needed on this surface), README three-ways update (the CLI
  row gains "or the structured investigation CLI"), AGENTS.md/CLAUDE.md
  doctrine lines, ledger note.

## 9. Acceptance

- **$0 scripted end-to-end (fake verifier)**: the round-6 quagmire cipher
  solved through the CLI ONLY: `start → diagnose → experiment-submit --type
  quagmire3_shotgun
  --wait → experiment-collect (install) → verify (explicit fake provider plus
  --allow-external) →
  declare-solution` — declaration accepted through the gate. Keyless variant
  asserts `verify` returns `no_verification_provider` and `declare-solution`
  stays blocked (`attestation_required`).
- **Cross-surface**: investigation started via CLI, continued via MCP session
  (and the reverse); experiment submitted on one surface, collected on the
  other; lease collision produces `writer_lease_held` not corruption; killed
  detached worker becomes recoverably `orphaned`, not permanently `running`.
- **Parity**: test_interface_parity green; a deliberately-added dummy manifest
  entry appears on both surfaces with no per-surface code.
- **Input/contract**: every operation accepts a canonical JSON object; nested
  repair/read payloads round-trip without comma parsing; stdout remains one
  JSON object for success and every classified failure; the exit-code matrix
  is covered.
- **Privacy**: an ambient API key alone never enables a call; both `verify` and
  a repair that actually needs verifier arbitration are blocked before the
  provider call without explicit provider selection plus `--allow-external`.

## 10. Orchestration

Per CLAUDE.md: this spec (Fable) → per-milestone Opus/Sonnet coders (I-0 and
the I-3/I-4 lifecycle work require the most care; I-1/I-2/I-6 are more
mechanical) → Fable review per
milestone → one commit per reviewed milestone. Coordinate with the composite
program (`composite_substitution_transposition_spec.md` Slice C.2 adds an
experiment type — after I-0 lands, that type arrives on both surfaces via the
manifest for free) and the polygraphic program (PF-7 must target the manifest,
not the MCP tool list directly; its spec review already flags surface
coordination).
