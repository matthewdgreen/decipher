# I-3 implementation sub-spec — experiments without a resident process

**Parent:** `docs/specs/investigation_cli_spec.md` §5 (binding design:
two-commit `--wait`, `--detach` handshake, signal semantics, orphan
reconciliation, cross-surface invariant), §6 (the private `_run-experiment`
verb is excluded from parity), §8 milestone I-3. Builds on I-2 (`c8ca2e7`).

## 0. Scope

`experiment-submit` and `experiment-collect` become live CLI verbs (the I-2
exclusions drop to ONE: `request_independent_verification` → I-5).
Deliverables: the `--wait` two-commit lifecycle, the `--detach` worker
handshake, SIGINT/SIGTERM orphan-marking in the waiting path, shared
startup orphan reconciliation (MCP crash recovery too), and the
cross-surface submit/collect tests.

## 1. Service — two-commit wait flow (`src/investigation_service/service.py`)

New method used ONLY by the CLI (MCP untouched):

```
run_experiment_to_completion(name="experiment_submit", arguments) ->
    (submit_body, final_revision)
```

Behavior (parent §5 `--wait`, byte-parity discipline):
1. Same validate/resolve/terminal/lease/revision pipeline as a mutating
   dispatch (reuse the existing helpers — do NOT duplicate the pipeline;
   factor a shared internal if needed, keeping `dispatch` byte-identical).
2. Submit through the normal ASYNC queue (`synchronous_experiments` must be
   False on this path even if the service was constructed otherwise —
   parent explicitly bans the single synchronous transaction because
   `status` must see the running record).
3. **COMMIT #1** with the pending/running record, exactly as a normal
   mutation commit.
4. Poll the queue in-process (existing `queue.poll` cadence; a short sleep
   loop) until the submitted experiment id reaches a terminal status.
5. **COMMIT #2** with the terminal record; release the lease (I-2
   INVOCATION_HELD machinery); return the ORIGINAL submit result body with
   `revision` set to the FINAL commit's revision (parent: caller can
   immediately collect without an avoidable conflict).
6. Early failures (validation, lease-held, revision conflict, terminal
   investigation) return the ordinary dispatch result untouched — the CLI
   maps them through the existing exit table.

**Signals:** while waiting, SIGINT/SIGTERM triggers best-effort: one final
poll; any non-terminal record for this submission marked
`orphaned` with typed reason `"interrupted"`; COMMIT; release; then
re-raise/exit conventionally (130 for SIGINT). Install the handler only for
the duration of the wait loop; restore afterwards.

## 2. Shared startup orphan reconciliation (service/runtime seam)

Parent §5: a persisted pending/running experiment is stale only when the
current process has proof that no live writer owns its worker. A read-only
runtime has no such proof: another process may hold the lease and be running
the experiment. Requirements:
1. Preserve pending/running records when rebuilding a runtime for a read.
   After a mutation successfully acquires the writer lease, rebuild through
   the same runtime seam and mark pending/running records `orphaned` with
   typed reason `"no_live_worker_at_startup"` IN MEMORY; they persist with
   the mutation's next commit. This applies to BOTH transports because both
   mutation paths use the shared lease/revision pipeline. Generic artifact
   resume retains the pre-existing conservative `orphaned(loaded)` default.
2. FIRST inventory the existing orphan machinery (`orphaned:run_ended`
   exists; `resubmit=<experiment_id>` re-runs orphaned/failed records). The
   new reason must compose with resubmit and must NOT reclassify records
   the existing machinery already handles. If the existing semantics
   conflict with the startup rule (e.g. the queue already restores workers
   for persisted running records), STOP that part and report — do not
   guess.
3. Dedup interaction (parent §5's motivation): a dead experiment must not
   be permanently unresubmittable via the duplicate-spec dedup — verify the
   dedup path skips orphaned records (it advertises dedup "to a prior
   completed run"; confirm and pin with a test).
4. Runtime construction after lease acquisition is transactional with respect
   to lease ownership: if load/build raises before the runtime is installed,
   release a lease acquired by that call under either lease policy.

## 3. CLI (`src/investigation_cli.py`)

1. Drop `experiment_submit`/`experiment_collect` from the I-2 exclusion set
   (the `verify` exclusion and its message stay). `experiment-collect` is
   an ordinary mutating verb — no special lifecycle.
2. `experiment-submit` gains mutually exclusive `--wait` (DEFAULT) and
   `--detach`. `--wait` routes through `run_experiment_to_completion`;
   stdout is still exactly ONE JSON object (the submit body with the final
   revision), emitted after the second commit.
3. `--detach` (parent §5): the parent process does NOT construct a lease or
   mutate. It spawns `decipher investigation _run-experiment` (private
   verb, hidden from help, excluded from parity per parent §6) via
   `subprocess.Popen(start_new_session=True)` with stdio redirected to
   DEVNULL and a dedicated handshake pipe fd passed as
   `--handshake-fd N` (`pass_fds`). The CHILD acquires the lease,
   revision-checks, submits, COMMITS the running record, writes the exact
   submit result/error body as one JSON line to the handshake fd, closes
   it, then keeps the lease, harvests, commits the terminal record
   (reusing §1's wait flow), and exits. The PARENT blocks on the handshake
   line, prints it verbatim to stdout, exits with the ordinary exit-code
   mapping for that body. Parent-side timeout for the handshake: 60s → if
   exceeded, print `{"status":"error","reason":"detach_handshake_timeout"}`
   exit 5 (the child may still be running; say so in `detail`).
   Child diagnostics go to the investigation event log, not stdio.
4. `_run-experiment` argument surface is private and minimal:
   `--registry-dir`, `--handshake-fd`, `--input-json` (the full canonical
   submit arguments). It is registered but not listed in help output
   (argparse `add_parser(..., help=argparse.SUPPRESS)` or equivalent).

## 4. Tests

All $0, tmp registries. For fast experiments, inject a stub entry into
`EXPERIMENT_TYPES` (monkeypatch) whose runner returns instantly, plus one
slow-stub (sleeps ~2s) for the lifecycle-visibility tests. Follow the
existing test layout (`tests/test_investigation_cli.py` + the service test
homes).

- **Two-commit visibility:** with the slow stub and `--wait` run in a
  thread/subprocess, a concurrent `status` read (separate registry
  instance) observes the running record between the two commits; after
  exit, the record is terminal and stdout's `revision` equals the final
  registry revision; an immediate `experiment-collect --revision <that>`
  succeeds with no conflict.
- **`--wait` result parity:** the emitted body equals a SESSION_HELD
  `dispatch("experiment_submit")` body for the same arguments (modulo the
  documented final-revision difference — assert that difference
  explicitly).
- **SIGINT during wait:** send SIGINT to the waiting process → registry
  shows the record `orphaned`/`interrupted`, lease free, exit 130, still
  exactly one JSON object on stdout (the submit body was NOT yet printed —
  verify the contract: on interrupt before completion print the orphan
  outcome body; pin whatever §1 implements and document it in the
  sub-spec adjudication if this needed a decision).
- **Detach handshake:** `--detach` returns promptly with the submit body
  while the child completes in the background; after child exit the record
  is terminal; parent never held the lease (assert via lease file during
  the handshake window).
- **Kill reconciliation:** SIGKILL the detached child mid-run → record
  stuck `running`, lease free (flock died with the process); a read preserves
  that persisted state because it has no writer proof → the NEXT mutation
  on either surface (CLI mutation AND a SESSION_HELD dispatch)
  reconciles it to `orphaned`/`no_live_worker_at_startup`; `resubmit`
  then re-runs it successfully; the duplicate-spec dedup does NOT return
  the orphaned record.
- **Cross-surface:** submit via CLI `--wait`, collect via a SESSION_HELD
  service (MCP path); and submit via SESSION_HELD (synchronous), collect
  via CLI. Both directions produce installable results (parent §5
  invariant).
- **Exclusion narrowing:** `experiment-submit`/`experiment-collect` no
  longer return `operation_not_yet_available`; `verify` still does.

## 5. Out of scope (binding)

`request_independent_verification` and all external-call flags → I-5;
parity-test hardening and docs → I-6; README → I-7; no timeout knob for
`--wait` (Ctrl-C is the mechanism); no changes to MCP lease policy; no new
experiment types.

Landing bar: main suite baseline at the base commit plus the new tests,
zero failures; the MCP suite unmodified except tests that (correctly) gain
the shared startup-reconciliation behavior.

## 6. Follow-up review findings (2026-09-07)

An adversarial review of the landed I-3 work (`b9353ae`) produced seven
findings against `src/investigation_cli.py`,
`src/investigation_service/service.py`, and the runtime seam in
`src/mcp_server/runtime.py`. All seven are FIXED; no contract in §§1–5 changed.

- **F1 (medium, fixed) — detached-child diagnostics never reached the event
  log.** §3.3 requires child diagnostics in the investigation event log, but
  `_run_experiment_worker`'s `except Exception` handler swallowed everything
  (stderr is `DEVNULL`, so the `DECIPHER_CLI_DEBUG` traceback was unreachable)
  and returned 0, leaving a crashed child completely silent and
  `<registry>/<id>/events.jsonl` never created. The registry is now built
  BEFORE the service call so the handler can reach it; the worker emits
  `detached_worker_started` / `detached_worker_finished` around the lifecycle
  and `detached_worker_error` (payload `{"error": <format_exception_only>}`,
  `turn` 0) on a crash, and returns a NON-ZERO exit code. Pinned by
  `test_detached_worker_crash_reaches_the_event_log`.
- **F2 (medium, fixed) — handshake EOF was misreported as a 60s timeout.**
  `_read_handshake_line` returns `None` for both a real timeout and an EOF, so
  a child that died in milliseconds still produced
  `detach_handshake_timeout` claiming the worker "may still be running". The
  parent now consults the child's own exit status after a `None` result and
  returns `{"status":"error","reason":"detach_worker_exited", ...}` (exit 5)
  when it has exited; `detach_handshake_timeout` is retained for a genuine
  timeout with the child still alive. Because the exit status may not be
  reaped at the instant of EOF, `poll()` is followed by a bounded
  `wait(timeout=_DETACH_EXIT_GRACE_S)` (0.5s) before concluding "still
  running". Pinned by
  `test_detach_worker_exit_before_handshake_is_not_reported_as_timeout`.
- **F3 (low, fixed) — handshake fd double-close.** `_on_running` and the
  `except` path both closed the same integer; if an intervening `os.open`
  reused the descriptor, the second close hit an unrelated fd. The fd now
  lives in a single-slot holder set to `-1` on the first close, so every
  later close is a no-op.
- **F4 (low, fixed) — potential busy-spin.** `_wait_experiment_terminal`'s
  `pending` branch polled and `continue`d with no sleep. Unreachable while a
  slot is free, but a hot loop if slots are ever exhausted. It now sleeps the
  shared `_WAIT_POLL_INTERVAL_S` (0.1s, the same interval the running branch
  waits on the settle event).
- **F5 (low, fixed) — signal-restore `TypeError`.** `signal.signal(sig, prev)`
  raises `TypeError` when `getsignal` returned `None` (a handler installed
  outside Python). `TypeError` joined the restore `except` clause.
- **F6 (low, fixed) — unconditional release.** `run_experiment_to_completion`
  released the lease on paths that ignored whether THIS call acquired it, so a
  SESSION_HELD or pre-held INVOCATION_HELD lease would have been released by a
  call that did not take it. Not CLI-reachable today (every caller is
  INVOCATION_HELD with no prior lease), but it broke the I-2 guard discipline
  `dispatch` honors. Every release now goes through one `_release()` helper
  gated on the `acquired` flag, matching `dispatch`'s `finally`.
- **F7 (info -> fixed) — reason string overclaimed.**
  `InvestigationRuntime.__init__` defaulted `reconcile_stale_experiments=True`
  with the hardcoded reason `no_live_worker_at_startup`, which any direct
  library constructor (tests included) triggered without lease proof. The
  constructor gained `reconcile_reason`, defaulting to the conservative
  `"loaded"`; `InvestigationService._acquire_and_check_revision` — the one
  path holding writer proof — passes `no_live_worker_at_startup` explicitly.
  Generic artifact resume keeps `"loaded"` (§2.1) and the service-level typed
  reason after a lease acquire is unchanged.

### Accepted and documented gaps (no code change)

- **SIGINT outside the guarded window.** The §1 signal handler is installed
  only for the duration of the wait loop. A SIGINT landing between COMMIT #1
  and handler installation, or between the wait returning and COMMIT #2,
  leaves the record persisted `running` with no orphan marker. It is not lost
  state: the next mutation's lease acquire reconciles it to
  `orphaned`/`no_live_worker_at_startup` (§2), i.e. it self-heals. Widening
  the guard would mean handling signals across the commit itself, which is a
  larger change than the window it closes.
- **`status` after SIGKILL.** A SIGKILLed worker leaves the record `running`,
  and `status` reports `running` until the next mutation reconciles it. §2.1
  chooses this deliberately: a read has no proof that no live writer owns the
  worker, so reporting the persisted state honestly is preferred over a read
  that guesses.

### Second-review adjudication (2026-09-07, Fable: LAND WITH FIXES)

No correctness defect; the reviewer independently reproduced the F3
double-close (pre-fix code fails `os.fstat` with EBADF on a recycled
descriptor; post-fix survives). Applied:

- **#1 (fixed):** `_read_handshake_line` now returns `(line, outcome)` with
  `line | eof | deadline`, so an EOF from a STILL-RUNNING child reports the
  new `detach_handshake_closed` reason instead of falsely claiming a 60s
  timeout elapsed. `detach_worker_exited` also names the signal on a negative
  returncode.
- **#2 (fixed):** added `test_handshake_fd_is_closed_exactly_once`, which
  recycles the freed descriptor and would fail on a double close.
- **#3 (fixed):** added `test_detach_handshake_timeout_requires_a_live_child`
  pinning the genuine-timeout branch (a child that never writes and outlives
  the deadline).
- **#4 (fixed):** `_close_handshake()` moved into the worker's `finally`,
  covering BaseException paths.
- **#5 (fixed):** `detached_worker_finished` now carries status/reason/
  experiment_id/final_revision rather than an unconditional "completed".
- **#6 (accepted):** the `/usr/bin/false` skipif is correct as written.
- **#7 (fixed):** this section's date.

Out-of-scope note recorded for a future slice: on Linux with the fork start
method, an `automated_solver` pool worker forked during the window between
`_execute` and `_on_running`'s close would inherit the handshake write end
and delay the parent's EOF on the crash-before-handshake path (`pass_fds`/
`close_fds` guard `subprocess`, not `fork`). It does not affect the normal
path, which waits for a line rather than EOF.
