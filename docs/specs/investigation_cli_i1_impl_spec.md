# I-1 implementation sub-spec — read verbs via the operation manifest

**Parent:** `docs/specs/investigation_cli_spec.md` (binding: §3 surface, §3.1
input contract, §3 output/exit contract, §8 milestone I-1). This sub-spec
resolves I-1-scope decisions so the coder invents nothing.

## 0. Scope

Exactly milestone I-1: the `decipher investigation` subcommand with the NINE
read verbs, canonical JSON input, and the `call` escape hatch — reads only.
Mutations, leases-per-invocation, `--revision`, id-containment hardening, the
exhaustive exit-code matrix, verify/external-call flags: all later milestones.

The manifest already carries everything needed (verified on main @ `7f3f09f`):
`investigation_service/manifest.py` `OperationSpec` records with
`operation class` (`TOOL_CLASSES`) and explicit `cli_verb` (`_CLI_VERB`); the
read-class set is exactly the spec's I-1 list:

| operation | verb |
|---|---|
| `investigation_list` | `list` |
| `investigation_status` | `status` |
| `observe_overview` | `overview` |
| `observe_diagnosis` | `diagnose` |
| `decode_show` | `decode` |
| `hypothesis_next_steps` | `next-steps` |
| `candidate_list` | `candidates` |
| `candidate_show` | `candidate` |
| `branch_adjudicate` | `adjudicate` |

## 1. Files

- **New:** `src/investigation_cli.py` — all CLI logic:
  `add_investigation_subparser(subparsers)` (registers the verb tree from the
  manifest) and `run_investigation_command(args) -> int` (returns the exit
  code). `src/cli.py` only wires these two calls into `main()` (mirror how the
  `mcp` subcommand stays thin).
- **New:** `tests/test_investigation_cli.py`.
- No TOOLS.md / onboarding changes (docs are I-6).

## 2. Verb registration (auto, from the manifest)

Iterate the manifest's operations; register a subparser for every entry whose
class is `read`, named by its `cli_verb`. NO hand-written verb list anywhere —
the parity test pins today's set of nine. Per operation:

- `investigation_id` (when the schema requires it) is the positional `ID`;
  every other scalar property (`string`/`integer`/`number`, incl. enums)
  becomes `--kebab-case` with the argparse type/choices derived from the
  schema. Array/object/union properties become `--<kebab-name>-json` flags
  whose values are parsed with `json.loads` (parse failure = CLI input error,
  §4). No comma-splitting convenience flags in I-1.
- Boolean properties *(corrected at review — the original "reads have no
  booleans" claim was false: `candidate_list.include_rejected` and
  `branch_adjudicate.include_window` are boolean read properties)*: an
  optional boolean with no schema default registers as
  `--flag`/`--no-flag` (`BooleanOptionalAction`, `default=None`) and the key
  is OMITTED from the built object when unset, so the service-visible
  arguments are identical to MCP's for all three states (absent, explicit
  true, explicit false; the service treats explicit false == absent).
  Byte-parity of all three states was verified against the MCP serializer
  at review.
- Additionally each verb accepts `--input-json JSON` and `--input-file PATH`
  (`-` = stdin, one JSON object). Supplying more than one input mode — any two
  of {friendly flags, `--input-json`, `--input-file`} — is a CLI input error
  (positional `ID` counts as a friendly flag for this rule, EXCEPT that `ID`
  may be combined with `--input-json`/`--input-file` when the JSON object does
  not itself contain `investigation_id`; a duplicate is the error). The built
  object is handed to the service untouched — validation lives in
  `InvestigationService.dispatch` (it already validates against
  `manifest.schema_for`), and the CLI adds ZERO schema logic.

`call OPERATION --input-json|--input-file` dispatches any manifest operation
by canonical name — but in I-1, if the named operation's class is not `read`,
return `{"status": "error", "reason": "operation_not_yet_available",
"detail": "mutating operations arrive with milestone I-2
(invocation-held lease semantics)"}` with exit 2, WITHOUT constructing the
registry/service. Unknown operation name → same shape with reason
`unknown_operation`, exit 2. (Rationale: `InvestigationService` itself
hard-rejects non-SESSION_HELD lease policies until I-2; the CLI must not
mutate under a session-held policy it cannot honor.)

## 3. Global options and service construction

- `decipher investigation [--registry-dir DIR] VERB ...` — precedence:
  explicit flag → `$DECIPHER_MCP_REGISTRY` → `default_registry_dir()`
  (reuse the registry module's helper; do not re-implement).
- Per invocation: `InvestigationRegistry(<dir>)` +
  `InvestigationService(registry=..., client_name="cli")` (defaults
  otherwise: no verify provider — reads never verify). Constructed lazily
  AFTER argument parsing and the `call` class check.
- **Reads must not acquire the writer lease.** Verify this holds in the I-0
  service read path (spec §4 "reads never need the lease"); if you find that
  read dispatch does acquire/hold a lease, STOP and report it as a spec gap —
  do not change lease behavior inside I-1.

## 4. Output and exit codes (I-1 subset of the parent contract)

- stdout: exactly one JSON object + `\n` per invocation — the dispatch result
  body byte-identical to what the MCP server would return for the same
  operation+arguments (including any revision injection the server layer
  performs — locate where `mcp_server` injects `revision` and reuse the same
  code path/helper; if the injection lives inside `dispatch` already, nothing
  to add). `json.dumps(..., ensure_ascii=False)`, no indentation. Diagnostics
  to stderr only.
- Exit codes via ONE shared table `result_to_exit_code(result) -> int` in
  `investigation_cli.py`, written to be extended in I-2:
  - result `status` != "error" → `0`
  - reason `invalid_arguments` → `2`
  - post-parse CLI input errors (unparseable `--*-json`/`--input-json`,
    unreadable/non-UTF-8 `--input-file`, input-mode conflict, `call` class
    check) → the CLI emits `{"status": "error", "reason":
    "invalid_cli_arguments", "detail": ...}` (or the §2 reasons) → `2`
  - any other domain error result → `1`
  - unexpected exception → stdout `{"status": "error", "reason":
    "internal_error"}`, traceback to stderr ONLY when `DECIPHER_CLI_DEBUG=1`,
    exit `5`
  - `3`/`4` are reserved (documented in the table as I-2 classes; no read
    produces them).
- argparse-native failures (unknown verb, missing required flag, bad enum)
  keep argparse's conventional behavior: usage on stderr, exit 2, empty
  stdout. The one-JSON-object guarantee applies to invocations that parse.

## 5. Tests (`tests/test_investigation_cli.py`)

Use a tmp registry dir; seed state by calling
`InvestigationService.dispatch("investigation_start", {...})` directly with a
short ciphertext (synchronous, $0, no provider). Invoke the CLI in-process
(argv → parser → `run_investigation_command`), capturing stdout/stderr.

1. **Auto-registration parity**: the registered verb set == the manifest's
   read-class `cli_verb` set == the nine-verb table above (pins both
   directions: no missing, no extra, no hand-list drift).
2. **E2E reads**: `list` (finds the seeded id), `status ID`,
   `decode ID --branch main`, `candidates ID` — each exits 0, stdout parses
   as exactly one JSON object ending in `\n`, and the body equals
   `service.dispatch(...)` called directly with the same arguments (parity
   assertion, ignoring nothing).
3. **call escape hatch**: `call investigation_status --input-json
   '{"investigation_id": ID}'` == the `status ID` body; `call
   meta_declare_solution --input-json '{}'` → reason
   `operation_not_yet_available`, exit 2, and the registry directory is
   untouched (no lease file created, no event appended); `call nope_op` →
   `unknown_operation`, exit 2.
4. **Input modes**: friendly flag + `--input-json` duplicating
   `investigation_id` → `invalid_cli_arguments` exit 2; malformed
   `--input-json` → exit 2; `--input-file -` reads the object from stdin
   (monkeypatched) and succeeds; `ID` positional + `--input-json` WITHOUT a
   duplicate id → succeeds (merge rule §2).
5. **Exit-code table**: unknown investigation id → exit 1 with the service's
   own reason string; `decode ID` missing required `--branch` →
   argparse exit 2 if the schema marks it required, else the service's
   `invalid_arguments` → 2 (assert whichever path the schema actually takes,
   with a comment); monkeypatched `dispatch` raising → exit 5, stdout
   `internal_error`, no traceback on stdout.
6. **stdout purity**: for every invocation above, stdout is a single JSON
   document (`json.loads` consumes it fully after stripping the trailing
   newline).

Baseline: suite is 1864 passed / 2 skipped at `7f3f09f`; the landing bar is
that plus these tests, zero failures.

## 6. Review adjudication (2026-07-20, Fable review: LAND)

No code findings required fixes. Doc correction applied (§2 booleans, above).
Low-severity notes deferred to their milestones:
- **I-6 parity test**: add one byte-level CLI-stdout vs `call_tool`-text
  assertion (today's tests use `service.dispatch` as the oracle; byte parity
  was verified at review but is not pinned against serializer drift).
- **I-6 (optional)**: registry-untouched test should also assert the
  directory set, not just file hashes; consider gating the manifest import
  in `cli.py` on the subcommand if `decipher` startup latency (~0.1s added)
  ever matters.
- Latent drift note: a future read op with an OPTIONAL `investigation_id`
  would get neither positional nor flag (reachable via `--input-json` only —
  lossless path exists, contract satisfied).

## 7. Out of scope (repeat, binding)

Mutations/`start` via CLI; `--revision`; lease acquisition or release APIs;
`--verify-provider`/`--allow-external`/`--max-cost-usd`; exit codes 3/4 in
practice; id-grammar/path-containment hardening (I-2); `--pretty`; docs.
