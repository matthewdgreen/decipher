# R4 preparation — 2026-09-07

Status: **preparation complete; measurement not run; R3 source review remains
open.** This checkpoint prepares the experiment while the user reviews R3.
It is not R4 closure, a solver performance result, or provider-run authority.

## Prepared evidence

`scripts/prepare_reliability_routing.py` validates the frozen R0 packet and
writes an immutable, deterministic schedule. The local preparation contains
18 cases and 60 requests, with no missing cases:

| Cohort | Cases | Seeds per case | Arms | Requests |
|---|---:|---:|---:|---:|
| Development | 6 | 3 | 2 | 36 |
| Held-out | 6 | 1 | 2 | 12 |
| Historical diagnostics | 4 | 1 | 2 | 8 |
| Unsupported/random controls | 2 | 1 | 2 | 4 |

The local outputs are gitignored:

- `artifacts/reliability_r4/preparation_v1/control/plan.json`: private schedule,
  role membership, protocol and provenance; never passed to a solver.
- `artifacts/reliability_r4/preparation_v1/runtime/j*.json`: individual worker
  requests. Each worker receives one request, not the directory or schedule.

Evidence hashes:

- Prepared plan: `baca3c3452c84e7695e48d8662bb6e10edf64f9bce10f7fffbcbd177bbbf5c36`
- Original R0 runtime manifest (logical JSON hash):
  `ea22bc8a4a791275f03919eca6b450e80ab98c1d694d1208307886e18f2ae357`
- Original private grading packet (file hash):
  `9941e6714cc3e42017ee1c527b8b7d59a4350c8cdae2bb619d349eccdbf92d5b`
- Solver baseline: `9eaf309fee3ece258e202ddd186e2ee290874b8c`, the checkpoint
  containing R2's completed code. Preparation changes no `src/` code.

All three R0 language-model checksums still match. Provenance also records
163 tracked source/resource/build inputs, 18 auxiliary resource files, the
Python interpreter, both harness files, and the installed native package and
extension hashes. The native binary hash binds the installed build, but does
not independently attest that it was built from the recorded Rust sources.
Future launches must recheck the pins; changed inputs require an explicit
new preparation/amendment, not a silent re-freeze.

## Experiment contract

Worker inputs allow only opaque case ID, ciphertext, format, language,
ciphertext hash, arm, requested seed and family metadata. Extra fields fail
closed. Blind requests have an empty family string. Family-supplied requests
receive only the frozen family name, not keys, plaintext, cribs, solver hints,
source names, cohort membership or known transform parameters. Language and
all other inputs are identical within each pair. Random controls receive no
invented family. Plaintext and keys cannot influence scheduling or retries.

Pairs are adjacent with a deterministic, hash-based counterbalanced arm order.
All development repeats are scheduled before results exist; no adaptive
repeats are added. The existing `run_automated` API consumes family metadata;
the harness does not force a chosen route or repair the router during measurement.
In particular, supplied metadata is not proof that the appropriate engine was
selected. Actual route records remain necessary for interpretation.

`scripts/reliability_routing_worker.py` provides the runtime-only adapter and
bounded subprocess mechanics. Each isolated worker uses a clean environment:
caller provider credentials, Python-path injection, solver overrides and
answer-bearing keyword/replay settings are not forwarded. Model paths are
pinned; the configured worker/thread pools are capped at four. This is an
input firewall, not an OS filesystem/network security sandbox, and thread
settings are not a hard aggregate process/thread quota.

The frozen limits remain 180 seconds wall per arm, 720 CPU seconds per process,
one concurrent arm, at most 60 arms, and 10,800 seconds total wall. The launch
predicate reserves the entire 180 seconds rather than shortening a late arm.
The bounded subprocess helper kills the child's process group on every exit,
including timeouts and interruption. CPU limits are installed before solver
imports and inherited by descendants. On normal completion, worker and reaped
child CPU use is recorded. After a forced timeout, CPU use is **unknown**, not
zero. Escaped sessions and hard aggregate resource accounting are not covered.

The existing progress callback exposes step name/status/time, not full route
or candidate details. Those details are saved on completed artifacts; a
timeout may leave only progress names. Timeout evidence is retained but never
promoted to a completed result, even if a child printed a final packet before
hanging. Malformed, duplicate, nonzero-exit, or wrong-request results fail.

## Measurement caveats caught during preparation

1. **Requested seeds are not uniformly honored.** Quagmire uses
   `DECIPHER_QUAGMIRE_SEARCH_SEED`; other shipped paths have fixed internal
   seeds or mixed random-number sources. The global Python RNG is seeded too,
   but this is not a universal engine seed. Artifacts distinguish requested
   seeds and Quagmire's returned seed. Repeated executions must not be counted
   as independent stochastic trials merely because request seeds differ.
2. **Completed is not solved.** No verifier or provider is called. Empty
   scoring placeholders from runtime artifacts are removed before handoff;
   post-hoc grading joins completed, hash-bound candidates separately. Controls
   have no fictional solve expectation; missing or timed-out quality remains
   unknown. Historical recovery metrics remain subject to R3's source review.
3. **Best-generated quality is not observable from every route.** Returned
   steps/menus and delivered text are preserved, including delivery-versus-
   artifact text equality. They do not prove that every generated candidate
   was retained. The grading scaffold leaves best-generated quality unknown
   and does not automatically attribute a quality gap to wrong routing.

## Validation and reproduction

Preparation was run twice against the unchanged R0 packet; output was
idempotent, and the read-only prepared-input recheck passed. No benchmark
solver, live Codex session, or paid provider was run for this preparation.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_reliability_routing.py \
  --out artifacts/reliability_r4/preparation_v1

PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_reliability_routing.py tests/test_reliability_packet.py \
  tests/test_ground_truth_firewall.py tests/test_verification_residual_audit.py -q
```

The preparation/firewall group passed **80 tests**, including 45 new R4 tests.
The neighboring automated-runner, candidate-retention and interface-parity
group passed **201 tests** (281 passing focused tests total).
Fake-worker process tests exercise CPU-limit installation, timeout cleanup of
descendants, progress retention, early-final rejection, malformed output and
request/result identity. An API-spy test uses the real progress callback
contract without running a solver. Provenance tests reject changed source,
models, missing native code and changed requests. These checks establish
mechanics, not actual matrix runtime or solving quality.

Preparation records the creating revision as well as file hashes. After a
revision/input change, use a new output directory; the old preparation is
immutable and must not be overwritten. A new preparation is not run authority.

## Remaining after R3 closes

1. Incorporate the human source-review findings and record whether they alter
   interpretation or require a pre-run amendment to R4.
2. Wire the prepared worker into a serial campaign launcher with a durable
   attempt ledger, total-budget accounting across interruption/resume, and
   no silent reruns. Recheck source/model/native/request pins before launch.
   The current preparation CLI intentionally has **no campaign-run option**;
   the low-level worker is not an operator launch recipe.
3. Execute the frozen paired measurement. Join labels only after completion;
   preserve failures, timeouts, actual route/resource evidence and seed caveats.
4. Produce the paired report and select at most two bounded follow-up slices.
   Do not infer live-agent/interface superiority from this automated packet.

The K4 mechanism-recovery proposals remain post-R4 choices. Packaging remains
parked; the separate session's current packaging-plan edit was not changed.
