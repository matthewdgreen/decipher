# Shared-machine evaluation preflight

Effective 2026-09-08, at the user's request. Applies to Codex, Claude Code and
manual evaluation launches on the same execution host, across repositories,
worktrees and sessions.

**Before starting or resuming any evaluation, check for other active runs and
machine contention.** This includes solver benchmarks, frontier sweeps, timed
acceptance runs, live solving smokes and provider-backed verifier/model
experiments. A provider run can be active while local CPU usage is low.

## Required check

1. Immediately before launch, inspect the host's process tree and at least two
   CPU/load observations a few seconds apart. Check memory pressure as well.
   Look for evaluation controllers, detached workers, solver/native children,
   test sweeps, builds and other sustained resource-heavy jobs. Do not rely on
   a single low-CPU snapshot or the processes in the current terminal alone.
2. Check known campaign ledgers, worker/lease status and active-run records
   where available. Resolve whether suspicious processes belong to a running
   evaluation; an idle-looking controller may be waiting on an API or child.
   A stale record alone does not establish a live run or permission to delete it.
3. If another evaluation is active, **defer by default and coordinate** with its
   owner/user before overlapping runs. Also defer CPU/timing-sensitive work
   under substantial competing CPU activity or memory pressure. Lightweight
   editing and fake-provider/unit checks can continue if they do not materially
   compete. Do not infer an exception from an apparently idle CPU.
4. If visibility is insufficient or ownership is unclear, report that limitation
   and resolve it before launch. Never kill, pause or change the priority of
   another session's processes to make room without authorization.
5. Record the preflight with the campaign/run provenance: timestamp and host,
   relevant PIDs/parent PIDs, run identities/status, CPU/load and memory summary,
   visibility limitations, and the proceed/defer decision. Record any explicit
   approval for overlap and qualify timing claims accordingly. Save sanitized
   summaries, not credentials, environment dumps or unrelated command arguments.

On macOS, useful read-only starting points are:

```sh
date -u
uptime
ps -axo pid,ppid,etime,%cpu,%mem,comm
top -l 2 -s 2 -n 10 -o cpu
vm_stat
```

The process listing uses executable names rather than full arguments to reduce
accidental credential disclosure. Inspect narrowly scoped process arguments
only when needed to identify a relevant run, and redact sensitive information
before saving or sharing. Use equivalent host-native commands on other systems;
inspect the actual execution host, not just the machine initiating a remote run.

## Launch and campaign discipline

- Announce the intended evaluation, resource cap and preflight result to the
  user. Recheck after a material delay or immediately before a resumed launch.
- This check is not an atomic reservation: two sessions can both observe an idle
  host. Keep existing campaign locks and coordinate launch ownership; do not
  describe the machine as reserved solely because the check passed.
- If competing work starts during a campaign, record the boundary and assess
  contention before launching the next arm. Let an already-authorized bounded
  arm finish unless its own safety rules require stopping. Do not hide affected
  timings, silently discard outcomes, or rerun spent slots. Resume under the
  campaign's existing budget/no-retry rules after a fresh check.
- Ordinary lightweight unit tests and offline report generation are not new
  evaluation campaigns, but large test suites, native builds or other heavy
  work must not knowingly compete with a timed evaluation without coordination.

This is an operational policy, not a new evaluation authority or automatic
enforcement mechanism. It does not authorize paid calls, alter frozen inputs,
change solver budgets or loosen acceptance gates. Existing frozen protocols
(including R5a) remain intact; new launchers and run reports must apply and
record this additional pre-launch check.
