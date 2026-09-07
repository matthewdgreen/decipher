# R4 decision — 2026-09-07

**R4 closes with a bounded periodic-routing slice selected. R3 remains open.**
The user authorized running this local automated measurement while the human
source review continues. This decision relies on synthetic and mechanical
evidence, not provisional historical residual labels. No solver default,
verification gate, paid-run authority or live-agent claim changes here.

## What the measurement establishes

The [paired report](reliability_r4_measurement.md) and its hash-bound JSON
companion record all 60 scheduled attempts: 59 complete artifacts and one
interrupted artifact-publication attempt. Eleven of twelve primary synthetic
pairs have complete results under matched execution contexts. Among those
eleven pairs, near-exact character recovery (at least 99%) rises from **5/11
blind to 9/11 family-supplied**. These are descriptive pilot counts, not a
population success rate.

The four gains are substantial, with the same language, models and limits:

| Case | Family / role | Blind route | Blind char | Supplied char |
|---|---|---|---:|---:|
| `r2364d0069826` | Quagmire III / held-out | Homophonic | 39.8% | 100.0% |
| `r2714bba5ed4f` | Vigenère / held-out | Homophonic | 41.7% | 100.0% |
| `r766311e4367b` | Vigenère / development | Transposition | 32.1% | 100.0% |
| `r96f9e4d54195` | Quagmire III / development | Substitution+transposition | 31.5% | 100.0% |

The existing periodic screen and Quagmire engine recover these texts when
selected. The development repeats show the same recovery pattern, but do not
turn two development cases into six independent examples. Quagmire returned
the four requested seed echoes; other engines' fixed internal seeds remain
an explicit limitation. These fresh-key/source-split synthetics demonstrate
an automated routing opportunity, **not** Astra-versus-Sol superiority or
live-agent advantage.

Other observations:

- Both simple-substitution and substitution+transposition source pairs recover
  exactly in both arms. They are regression anchors, not a reason to rewrite
  those engines.
- The held-out columnar case stays at 69.5% in both arms. Supplied metadata
  changes the initial route but both end with the permutation search solver.
  Better family identification alone does not resolve this engine/search gap.
- The homophonic development case stays at 95.6% in both arms. Its held-out
  counterpart has one missing arm; it cannot establish a paired homophonic
  effect. Neither observation justifies a broad scorer/default change.
- All 59 delivered texts match their serialized artifact text. No superior
  text was found among the saved text evidence in the 57 reference-labeled
  completed outputs. This is evidence against a delivery defect in **this
  packet**, not proof that all generated candidates were retained or that
  candidate selection is solved generally.
- Controls produced no verified solution claims. Bifid was nevertheless
  routed to transposition even when its label was supplied; its 32.6% reference
  overlap is not a solve. A separate unsupported-family signaling issue remains
  visible. Random-text recovery quality is unknown because no plaintext exists.
- Historical scores remain provisional. They do not determine manuscript
  repair, editorial policy, source tools, or changes to the verifier.

## Next selected slice: R5 — bounded periodic-family routing/probes

**Observed problem:** family-blind routing sends Vigenère/Quagmire inputs to
homophonic or transposition/composite engines despite effective existing
periodic solvers. The four matched primary gains above span two source-separated
families; the benefit is not confined to a famous memorized cipher.

**User benefit:** improve blind recovery using existing engines, without
requiring the user or an agent to supply the cipher family correctly first.

**Bounded scope:** introduce a cheap, ciphertext-only periodic diagnostic and
bounded probe path before committing to an expensive non-periodic branch.
Reuse existing diagnostics, periodic search and candidate-preservation
mechanics. Keep uncertainty and fallback explicit; do not route every unknown
cipher straight to Quagmire. No new cipher family, model training, K4-specific
cribs, editorial repair or verifier change is included. The code specification
must freeze probe limits and selection rules before fresh evaluations.

**Acceptance experiment:** treat the four already-inspected R4 periodic cases
as engineering regressions, no longer unseen holdouts. Freeze an additional
12-case packet before implementation/evaluation: six varied periodic positives
and six matched non-periodic/random/unsupported controls, with fresh sources
and keys, lengths/periods recorded, and no solve-based rerolls. Compare unchanged
baseline versus prototype under identical 180-second per-arm limits. Labels
remain grading-side; candidate selection and retry decisions are solver-blind
to reference plaintext.

Acceptance requires near-exact recovery on all four R4 periodic regression
anchors, material gains (at least two percentage points) on at least three
of the six fresh positives, no material regression on any of the six controls,
and no unsupported/verified-solution claim created by routing. Report every
timeout and seed limitation. A missed threshold is a failed acceptance result,
not permission to filter examples or add retries.

**Proposed resource ceiling:** at most 32 local automated arms (eight anchor
arms plus 24 fresh-case arms), 180 seconds per arm, 720 CPU seconds per process,
four configured workers, one arm at a time, and 96 minutes total wall. No
provider calls. This is the next slice's proposed envelope, not extra authority
to rerun or extend the completed R4 campaign.

**Stop/change direction:** if the positive gains do not reproduce, controls
regress materially, or bounded probes consume the budget without improving
delivery, do not adopt the new routing default. Record whether the limiting
factor is diagnosis, probe ranking, period/search coverage or engine strength;
choose a discriminating measurement before expanding the implementation.

Only **one** program-development slice is selected. The second slot remains
open; do not automatically schedule the columnar gap, homophonic tuning,
MR0–MR4, packaging, or an agent-interface/model bake-off. The separately
reported CLI detached-worker diagnostics/EOF defects are maintenance findings,
not an R4 cryptanalytic conclusion; they were confirmed but not changed here.

## Execution, recovery and reproducibility

The immutable R0/R4 preparation was not rewritten. A separate campaign
metadata record records the user-authorized R3 sequencing exception and
launcher hash. Model checksums, native binary, source inputs and worker
request hashes were rechecked before each launch. The serial controller
records a durable start before spawning, excludes all previously started jobs
on resume, and retains the campaign lock in an independent timeout guard if
the main controller exits. Unknown outcomes consume a conservative reservation.

During attempt 9 (`j9ed34fc5d36bf0e0`, homophonic held-out family-supplied),
the sandbox denied `os.killpg` during guard cleanup. No final artifact was
published; solver quality is unknown. New launches were paused with SIGINT
during attempt 15, whose existing bounded guard completed. A process-tree
check found no remaining R4 workers. Attempts 16–60 then ran with host cleanup
permission. No attempt was rerun, and no solver code changed. The raw traceback,
durable starts, and execution-context boundary remain in the ignored campaign
directory. The Bifid pair (attempts 15–16) crosses that boundary and is not a
clean timing/causal comparison; it is retained descriptively. All eleven
complete primary synthetic pairs have matched contexts.

Known worker wall totals **877.6 seconds**; one attempt's actual worker time
is unknown. Conservative charged budget totals **1,117.7 seconds** against
10,800. CPU use is recorded per completed worker and its reaped children.
The wall numbers are shared-workstation observations, not a dedicated speed
benchmark; development checks and diagnostics were occasionally concurrent.
All 60 slots are spent. The missing outcome is not a zero score or an implicit
authorization for a replacement run.

The local campaign lives at `artifacts/reliability_r4/campaign_v1`. Its
`campaign.json`, `attempts/*.started.json`, `results/*.json`, `guard.log`,
`execution_context.json` and `summary.json` are retained. The result envelopes
can be inspected directly with `scripts/inspect_artifact.py`; both human and
LLM analysis packets now expose execution status, CPU/timeout uncertainty,
requested-seed caveats and delivery equality. No `--analyze` provider call was
made. The paired report can be regenerated without a solve:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_reliability_routing.py \
  --preparation artifacts/reliability_r4/preparation_v1 \
  --campaign artifacts/reliability_r4/campaign_v1 \
  --grading artifacts/reliability_r0/grading/packet.json \
  --report docs/reports/reliability_r4_measurement.md
```

Validation: **324 focused tests pass**, including fake-worker concurrency,
durable-attempt/no-retry recovery, inherited locking, budget limits, provenance,
post-hoc joins, cross-context reporting and artifact inspection. Benchmark
validation found 905 records and no errors (154 existing layer-availability
warnings). No paid calls, live Codex solving session or production solver/gate
changes were made. The unrelated packaging-plan edit remains untouched.
