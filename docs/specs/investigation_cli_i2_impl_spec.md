# I-2 implementation sub-spec — mutations, invocation-held leases, exit-code contract

**Parent:** `docs/specs/investigation_cli_spec.md` (binding: §3/§3.1, the
output/exit contract, §4 concurrency/lease semantics, §8 milestone I-2).
Builds on I-1 (`55226d7`); base commit `3a8894c`.

## 0. Scope and milestone boundaries

I-2 delivers: mutating verbs on the CLI, `--revision`, invocation-held
leases with an explicit release API, exit codes 3/4 live (full table
exhaustive over service status classes), and investigation-id containment
hardening on BOTH transports.

**Operations registered in I-2** (manifest classes `create` + `mutate`),
with THREE exclusions that keep their I-1-style typed error
(`operation_not_yet_available`, exit 2, naming the milestone):
- `experiment_submit`, `experiment_collect` → **I-3** (a short-lived CLI
  process would orphan async experiments; the two-commit lifecycle,
  `--wait/--detach`, and crash reconciliation are I-3's whole subject).
- `request_independent_verification` → **I-5** (the external-call privacy
  guard, `--verify-provider`/`--allow-external`, is I-5's subject; I-2
  constructs the service with NO provider).

`meta_declare_solution` / `meta_declare_unsolved` ARE registered in I-2:
they are pure domain operations. `declare-solution` without a fresh
attestation returns the server's own gate refusal (`blocked`, exit 3) —
correct and useful; the full verify→declare flow is exercised in I-5.
`repair-test`/`repair-transaction` are registered; with no provider
configured, any arbitration-needing path returns the service's own
no-provider domain result (keyless-by-default holds with zero new guard
code; the explicit `--allow-external` machinery is I-5).

## 1. Registry — explicit release + id containment (`src/mcp_server/registry.py`)

1. **`release_lease(investigation_id)`**: releases the flock and removes
   this instance's holder bookkeeping. Explicit release becomes the API;
   process teardown stays the documented fallback (parent §4). Idempotent
   (releasing a lease you don't hold is a no-op returning False).
2. **Id containment (BOTH transports, deliberate shared hardening —
   parent §4):** before any path join, validate `investigation_id` against
   the generated-id grammar (read the actual generator in
   `registry.create`/the id mint — pin the grammar it produces, e.g.
   `^[0-9a-f]{12}$`; derive from code, do not guess) AND verify the
   resolved directory is under the registry root (`Path.resolve()` +
   `is_relative_to`). Violations raise/return a typed
   `invalid_investigation_id` failure that `InvestigationService.dispatch`
   maps to `{"status": "error", "reason": "invalid_investigation_id"}`.
   Applies to every id-taking entry point (`load`, `acquire_lease`,
   `commit`, `append_event`, ...) via one shared validator called at the
   `_dir` seam. The MCP suite must stay green — well-formed ids are
   unaffected; a traversal attempt (`"../x"`, absolute paths, empty)
   never touches the filesystem outside the root.

## 2. Service — `LeasePolicy.INVOCATION_HELD` (`src/investigation_service/service.py`)

1. Remove the I-0 `NotImplementedError`; implement the policy in
   `dispatch`: under INVOCATION_HELD, a mutating dispatch wraps steps 5–8
   so that AFTER the commit (or any early return/exception past the
   acquire), the lease is released and the runtime dropped from
   `_runtimes` in a `finally`. SESSION_HELD behavior (MCP) is untouched —
   same code path, policy-gated release only.
2. Read dispatches never acquire; under INVOCATION_HELD they must also
   never RETAIN a runtime (today's read path already builds throwaway
   runtimes for non-holders — verify, don't change).
3. `investigation_start` (the `create` op, `_start`) under
   INVOCATION_HELD: commits the new document and holds nothing afterward.
4. Failure semantics preserved verbatim: `writer_lease_held` (+ holder
   hint), `revision_mismatch` (`status: "conflict"`), terminal block —
   the CLI adds no wording.

## 3. CLI (`src/investigation_cli.py`)

1. Auto-register `create`+`mutate` verbs from the manifest exactly like
   I-1's reads (same flag-generation rules; the three excluded ops keep
   the typed error and are NOT filtered out of `call`'s known-op set —
   `call` on them returns the same typed error). The I-1 `call` guard
   ("class is not read") is replaced by: dispatch any operation EXCEPT the
   three exclusions.
2. `--revision R` is the friendly alias for `expected_revision` on every
   verb whose schema requires it (positional-`ID` rule unchanged).
   Required-ness stays service-enforced (I-1 convention): a missing
   revision is the service's `invalid_arguments` → exit 2.
3. `start`: `--ciphertext "..."` XOR `--ciphertext-file PATH` (`-` =
   stdin, RAW ciphertext bytes decoded UTF-8 — NOT a JSON object) XOR
   JSON-object input modes; the file alias fills the schema's
   `ciphertext` property. File-read/UTF-8 failures are typed CLI input
   errors (exit 2); size/format failures remain domain results.
4. Service construction gains `lease_policy=LeasePolicy.INVOCATION_HELD`.
   Additionally, `run_investigation_command` releases via
   `service.shutdown()`-equivalent finalization in a `finally` so a
   SIGINT mid-verb still releases before the conventional exit (parent:
   best-effort state finalization, then conventional signal exit).
5. Exit-code table extension (same single table):
   `status == "blocked"` → 3; `status == "conflict"` → 4;
   `invalid_investigation_id` → 2; everything else as I-1. The table's
   docstring enumerates the service status classes and asserts
   exhaustiveness (unknown statuses fall to 1 with the reason preserved).

## 4. Tests

Extend `tests/test_investigation_cli.py` (CLI) + the service/registry test
homes (follow existing layout). All $0, tmp registries, no providers.

- **Lease lifecycle:** after a CLI mutation completes, the lease file is
  released (a fresh registry instance can `acquire_lease` immediately);
  two sequential CLI mutations on the same investigation both succeed;
  `_runtimes` empty after each invocation.
- **Lease collision:** while another registry instance holds the lease,
  a CLI mutation returns the verbatim `writer_lease_held` body (holder
  hint included), exit 3, and the document is unchanged.
- **Revision conflict:** stale `--revision` → verbatim `revision_mismatch`
  body, exit 4.
- **Terminal block:** mutation on a declared/closed investigation →
  `investigation_terminal`, exit 3.
- **Id containment:** `status "../x"`, an absolute path, and an
  overlong/malformed id each → `invalid_investigation_id`, exit 2, with a
  filesystem sentinel proving nothing outside the registry root was
  touched or created; same malformed id through a SESSION_HELD service
  (the MCP path) gets the same reason (shared hardening pinned on both
  transports).
- **start e2e:** `start --ciphertext ...` → exit 0, id in body; then
  `status ID` reads it back; `--ciphertext-file -` (stdin) variant; file
  + inline conflict → exit 2; unreadable file → exit 2.
- **Mutation e2e:** `branch-create` (+ `--revision`) through the CLI
  equals the same dispatch through a SESSION_HELD service on a copy
  (body parity, I-1 convention); a follow-up read sees the new revision.
- **Exclusions:** `experiment-submit`/`experiment-collect`/`verify` (and
  `call` on their canonical names) → `operation_not_yet_available`
  naming I-3/I-5 respectively, exit 2, registry untouched.
- **Exit-code matrix:** one parametrized test covering the parent's
  required classes reachable in I-2: not-found (1), schema failure (2),
  CLI parse failure (2), blocked gate (3), lease-held (3), terminal (3),
  conflict (4), internal exception (5).
- **Declare-unsolved smoke:** `declare-unsolved ID --revision R ...` →
  exit 0 (DECL-8: never gated), terminal afterwards (next mutation → 3).

Baseline: suite is **1885 passed / 2 skipped** at `3a8894c`; landing bar =
that plus these, zero failures. The MCP suite must pass unmodified except
tests that (correctly) gain the shared id-containment behavior.

## 5. Review adjudication (2026-07-20, Fable review: LAND WITH FIXES)

- **Grammar deviation RULED: containment-only enforcement accepted.** §1.2 as
  originally written was internally contradictory with §4's "MCP suite
  unmodified" bar: the strict `^[0-9a-f]{12}$` charset reclassifies benign
  CONTAINED ids ("nope", "id0", "aaaa0000") that five existing MCP tests
  exercise (verified empirically — exactly 5 fail under strict enforcement,
  including the pinned `test_unknown_id_typed_result`), and the parent spec
  frames the deliberate hardening as closing PATH TRAVERSAL. Enforced
  validator: non-empty, ≤128 chars, single non-relative path component (no
  separators/`..`/NUL), resolved directory under the registry root. The
  12-hex grammar stands as generator documentation
  (`_GENERATED_ID_GRAMMAR`), not an input filter. Residual accepted: on
  case-insensitive APFS an aliased contained id can reference another id's
  directory; the flock is on the same underlying lease file so single-writer
  exclusion holds (local-trust nuisance only).
- **Finding 1 (fixed):** `_start` raising after its acquire leaked the flock
  fd + runtime to library callers under INVOCATION_HELD (the CLI's
  shutdown() mop-up masked it). The create path now releases keyed on the
  held-lease-set DELTA in a finally, not on the success result.
- **Finding 2 (fixed):** `result_to_exit_code` docstring corrected — unknown
  reasons under `status=="error"` → 1; non-error statuses (incl. domain
  statuses in read bodies) → 0. The sub-spec §3.5 sentence "unknown statuses
  fall to 1" was unimplementable as written and is superseded by this.
- **Finding 3 (fixed):** direct-service revision-conflict test added — pins
  the dispatch-level finally release without the CLI shutdown() mask.
- **Finding 4 (fixed):** unused test imports removed.
- Verified clean at review: all 10 dispatch return paths under
  INVOCATION_HELD (loser never releases the winner's lease); SESSION_HELD
  byte-compatibility; validator coverage at every id-taking registry entry;
  exclusion short-circuits before registry construction; collision tests
  NOT vacuous (second open-file-description flock conflicts confirmed on
  this platform).

## 6. Out of scope (binding)

Experiments on the CLI (`--wait`/`--detach`, reconciliation) → I-3/I-4;
`--verify-provider`/`--verify-model`/`--max-cost-usd`/`--allow-external`
and every external-call path → I-5; parity-test hardening + docs → I-6;
README → I-7; `--pretty`; any change to MCP lease policy or wording.
