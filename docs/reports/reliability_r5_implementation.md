# R5b — Opt-in periodic probes and bounded execution

2026-09-08. **R5b implementation complete; R5c evaluation and the default
decision remain open.** No fresh packet was solved, no provider calls were
made, and the production switch remains off. R3 and the separate model
comparison are not advanced by this checkpoint.

## Implemented behavior

`DECIPHER_PERIODIC_ROUTING=probe_v1` enables the frozen R5 rules in the
automated runner. Unset/`off` preserves the existing path; unknown values fail
clearly. The only changes under `src/` are the runner integration and two new
automated helpers. No native kernel, scorer, language model, dictionary,
investigation service, agent-tool schema or verification gate changed.

- Eligibility is English, entirely A–Z input, 180–1200 tokens, without explicit
  family, transform, solver hints or custom periodic-search configuration.
  Explicit model variants are conservatively left on the existing route too.
  The campaign's fixed seed/thread settings are permitted; different custom
  Quagmire seed/thread settings preserve their directed search behavior.
- Existing IC/Kasiski/harmonic diagnostics select at most two qualifying
  periods, with the thresholds frozen in the [R5a spec](../specs/reliability_r5_periodic_routing_spec.md).
  No R5 fresh-case diagnostics or outcomes were inspected during development.
- The ordinary periodic screen runs first. If it has no eligible candidate,
  Rust Quagmire runs once across those periods and keyword lengths 6–8,
  with the specified fixed proposal budget. Both stages share a 45-second
  deadline including interpreter startup and language validation.
- Candidate eligibility requires full-length exact mode-specific replay and
  every frozen language gate. Conflicting displayed keys, shifts, alphabets
  or cyclewords are rejected. Ranking uses common validation and quadgram
  evidence, not incomparable engine scores.
- At most six distinct texts survive in the artifact menu. Complete menu
  publications survive later native failures/timeouts as diagnostic evidence.
  Only a clean completed probe may deliver an eligible candidate. Other clean
  outcomes fall back to the original route; unresolved cleanup aborts instead.
- An adopted text is `completed`, **not verified or solved**. Its artifact has
  an empty substitution mapping and preserved periodic/Quagmire key state.
  This native artifact menu does not automatically create six workspace branches.

The artifact inspector's human and LLM summaries expose eligibility, selected
periods, attempted stages, retained candidate hashes/replay/language evidence,
timeout/error, cleanup, adoption and fallback. The automated preflight briefing
also identifies probe outcomes without exposing benchmark labels. R5 execution
envelopes are distinguished from R4 envelopes in inspection.

## Process lifetime and budget

The native call executes in a worker session supervised by an independent
guard, not in the controlling Python thread. A private pipe tells the guard
when its caller dies. The guard enforces a monotonic deadline, kills only its
own worker group and bounds reaping/exit confirmation. Permission failures,
missing acknowledgements and still-visible groups are not called successful
cleanup.

Nested probes register private cleanup acknowledgements. The outer evaluation
guard waits for those acknowledgements too; killing the outer worker must not
let an out-of-group probe survive unnoticed into the next arm. The independent
outer guard inherits the campaign lock even if the launcher exits. Unresolved
groups retain their diagnostic registry path and require reconciliation; the
launcher does not automatically proceed or retry that slot.

The probe allowance remains 45 seconds plus bounded cleanup/controller work.
An evaluation arm retains its 180-second search deadline, including the probe;
fallback does not receive another 180 seconds. R5 ledger reservations are
**195 seconds** (180 plus nested cleanup/controller allowance), rather than
reusing R4's 181-second assumption. Completed attempts are charged measured
guard wall plus one second; unknown outcomes consume the full reservation.
The 5760-second campaign ceiling and maximum 32 slots remain unchanged. Budget
exhaustion can leave slots unattempted and cannot count as passing acceptance.

Quagmire may exhaust its deadline before returning a menu, particularly across
multiple lengths/periods. This implementation does not establish recovery or
latency gains. Those are the purpose of R5c; no thresholds or search budgets
were adjusted using fresh-case results.

## Execution adapter and machine preflight

`scripts/reliability_periodic_worker.py` projects both arms to the same existing
family-blind worker input. Their environments differ only in the opt-in switch.
Labels, keys, source IDs, roles and expected periods never enter that worker.

`scripts/run_reliability_periodic.py` reuses R4's immutable publication and
inherited campaign locking, with an R5-specific ledger reservation and strict
baseline/prototype provenance checks. It will not start another arm after an
unknown prior result or unconfirmed cleanup. No R4 preparation or campaign was
modified.

Each `run-one` invocation launches **at most one arm**. It requires a machine
preflight no older than two minutes, including two timestamped CPU/load
observations, process and memory summaries, host identity, complete visibility,
no active competing evaluations and confirmed cleanup authority. This record
is stored in the durable attempt before process creation. The launcher validates
the record; it does **not** itself inspect the host or reserve the whole machine.
The operator must perform the [shared machine check](../evaluation_machine_preflight.md).
The strict R5 adapter does not offer an overlap override.

Freeze the committed implementation/harness without running a solver:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_reliability_periodic.py freeze \
  --preparation artifacts/reliability_r5/preparation_v1 \
  --out artifacts/reliability_r5/execution_v1/pin.json
```

The execution pin binds the immutable R5a plan, baseline `e1536c1`, full solver
diff, implementation revision, source and harness hashes, native import path
and binary, models/resources and interpreter. Unrelated solver changes or
uncommitted runtime files block pinning. Later documentation-only commits do
not alter the execution snapshot. Recreating the pin at a different revision
is not an implicit replacement of its recorded implementation revision.

R5c will supply that pin's logical SHA-256 and a fresh preflight to `run-one`.
No `run-one` invocation or evaluation output exists at this checkpoint. After
the final slot, another invocation writes the campaign summary without launching
anything. Post-hoc report/acceptance joins must be finalized and tested before
the R5c run; do not import grading into the launcher to implement them.

## Verification and remaining limits

**355 focused tests passed in 13.95 seconds**, covering routing/eligibility,
all replay/language conditions, bounded retention, fixed native parameters,
fake worker completion/crash/timeout, parent-lease loss, nested guard cleanup,
permission failure, unchanged default fallback, artifact/LLM diagnostics,
R5 arm projection, preflight rejection, immutable/no-retry ledger and provenance
rejection, plus existing runner, candidate-retention, firewall and interface
tests. This is a focused regression pass, not the full repository suite or a
performance evaluation.

Before the broader pass, a host process check classified the idle Python
processes as MCP servers and found no evaluation runner. Two CPU observations
were about 72–74% idle, but memory was heavily compressed with swap activity.
Only focused regression checks proceeded. This was not clearance for R5c;
its launch requires a fresh check including active campaign/worker records.

The fake guard tests demonstrate local lifetime mechanics, not portability or
native search effectiveness. The current guard is POSIX-only. The installed
native binary is hash-pinned, not independently proven to match its source.
Fixed seeds are not automatically independent trials, and incomplete outcomes
must stay explicit in the paired report. No default adoption is warranted
until the predeclared R5c acceptance experiment passes.
