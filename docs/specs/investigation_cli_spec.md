# Spec — `decipher investigation`: the structured investigation CLI

Status: DRAFT for implementation. Author: Fable (main loop), 2026-07-19.
Origin: external agent evaluation (GPT-5.6-sol, fresh session, 2026-07-19)
reviewed the repo unprompted and concluded the missing piece is "first-class,
machine-readable access to the investigation state machine that MCP already
exposes." Its property list is adopted below as binding requirements. Sibling
context: `docs/mcp_dual_harness_proposal.md`, `docs/mcp_onboarding.md`.

---

## 1. Why

- **Shell-first agents are the majority client.** Both the naive-Codex control
  (cloned cDecryptor rather than discover our MCP surface) and the Sol
  evaluation (CLI-first, MCP only for extended investigations) show that
  agents with shells reach for CLIs. A structured CLI captures them with zero
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
decipher investigation adjudicate ID --branches B1,B2[,...]
decipher investigation branch-create ID --revision R ...
decipher investigation branch-update ID --revision R ...
decipher investigation branch-reject ID --revision R ...
decipher investigation experiment-submit ID --revision R --type T --config JSON [--wait|--detach]
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

### Binding output/exit contract (adopted from the Sol checklist)

1. **Stable JSON** on stdout: exactly the MCP tool-result body (the same dict
   the server would wrap), plus `revision` injection identical to the server.
   Diagnostics/progress go to stderr, never stdout.
2. **Stable exit codes**: `0` ok; `2` invalid arguments (schema); `3` blocked
   (`investigation_terminal`, `writer_lease_held`, gate refusals — the JSON
   `reason` is the identifier, exit code just classifies); `4` revision
   conflict; `5` internal error. Exit codes classify; the machine-readable
   identifier is always the JSON `reason` field, which reuses the server's
   strings verbatim (`attestation_required`, `revision_mismatch`, ...).
3. **Optimistic concurrency**: every mutating verb REQUIRES `--revision R`
   (maps to `expected_revision`); reads never take it. Same conflict semantics
   as MCP.
4. **Keyless/local by default**: `verify` with no provider configured returns
   the same structured `no_verification_provider` refusal as the server;
   sending candidate text to an external provider requires the provider to be
   configured AND `--allow-external` on the verb (explicit per-call opt-in —
   stricter than MCP, deliberately, because CLI callers may be scripts).
5. **Persistent IDs**: the registry's investigation ids; nothing session-scoped.

## 4. Concurrency and lease semantics

Reuse the registry exactly. A mutating verb acquires the per-investigation
flock lease at dispatch and releases it at process exit (flock's fd semantics
give this for free). Consequences to document and test:

- CLI mutation while a live MCP server holds the lease → `writer_lease_held`
  + holder hint, exit 3. Correct: same investigation, one writer.
- CLI↔CLI races → the lease serializes; the loser gets the same block.
- Reads never need the lease (unchanged).

## 5. Experiments without a resident process

The MCP server runs the experiment queue inside its long-lived process; a CLI
process is short-lived. Two modes, explicit:

- `--wait` (default): `experiment-submit` runs the experiment synchronously in
  this process and exits when it completes (the submit-record and result land
  in the registry exactly as the server would write them; `experiment-collect`
  then works from any surface). Callers who want shell-level backgrounding use
  `&` — that is their business.
- `--detach`: fork a worker process that owns the lease for the duration,
  writes the same registry state, and exits; `experiment-submit` returns
  immediately with the experiment id. `status`/`experiment-collect` poll as
  usual. Implementation: `subprocess.Popen` of a private
  `decipher investigation _run-experiment` helper, stdin/stdout detached,
  never a daemon framework.

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
  manifest entry (name → verb via a fixed `s/_/-/` mapping table, schema →
  argparse arguments mechanically: string/int/bool/JSON-blob). Adding an
  operation to the manifest EXTENDS BOTH SURFACES in the same commit, with no
  second implementation site.
- **Parity test**: `tests/test_interface_parity.py` asserts (a) every manifest
  operation has a registered CLI verb and vice versa; (b) the CLI's generated
  argument set covers every schema property; (c) for a scripted investigation,
  the JSON body returned by the service layer via the CLI equals the body the
  MCP server returns for the same call sequence (transport-independence
  proof). Extending one surface without the other fails CI.
- **Scope clarification (experiment types are NOT operations).** The parity
  contract is at the OPERATION level (the ~23 verbs). New solver
  `EXPERIMENT_TYPES` values are NOT new operations — `experiment_submit` is a
  single operation whose `type` is a string arg (and the CLI passes the whole
  config as one `--config JSON` blob). A new experiment type therefore reaches
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

## 8. Milestones

- **I-0**: service-layer extraction, byte-parity, suite green. (The risk
  milestone; everything else is mechanical.)
- **I-1**: read verbs (`list/status/overview/diagnose/decode/next-steps/
  candidates/candidate/adjudicate`) via manifest auto-registration.
- **I-2**: mutations + `--revision` + lease-per-command semantics + exit-code
  contract.
- **I-3**: experiments (`--wait`/`--detach`) + cross-surface collect test.
- **I-4**: `verify` (keyless refusal, `--allow-external` opt-in) +
  declarations through the gates.
- **I-5**: parity test hardening, onboarding §"CLI mode" (the operator recipe:
  no restart ever needed on this surface), README three-ways update (the CLI
  row gains "or the structured investigation CLI"), AGENTS.md/CLAUDE.md
  doctrine lines, ledger note.

## 9. Acceptance

- **$0 scripted end-to-end**: the round-6 quagmire cipher solved through the
  CLI ONLY: `start → diagnose → experiment-submit --type quagmire3_shotgun
  --wait → experiment-collect (install) → verify (provider configured) →
  declare-solution` — declaration accepted through the gate. Keyless variant
  asserts `verify` returns `no_verification_provider` and `declare-solution`
  stays blocked (`attestation_required`).
- **Cross-surface**: investigation started via CLI, continued via MCP session
  (and the reverse); experiment submitted on one surface, collected on the
  other; lease collision produces `writer_lease_held` not corruption.
- **Parity**: test_interface_parity green; a deliberately-added dummy manifest
  entry appears on both surfaces with no per-surface code.

## 10. Orchestration

Per CLAUDE.md: this spec (Fable) → per-milestone Opus/Sonnet coders (I-0 is
Opus; I-1..I-5 are largely mechanical, Sonnet-eligible) → Fable review per
milestone → one commit per reviewed milestone. Coordinate with the composite
program (`composite_substitution_transposition_spec.md` Slice C.2 adds an
experiment type — after I-0 lands, that type arrives on both surfaces via the
manifest for free) and the polygraphic program (PF-7 must target the manifest,
not the MCP tool list directly; its spec review already flags surface
coordination).
