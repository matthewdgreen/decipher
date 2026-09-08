# R5a — Frozen periodic-routing preparation

2026-09-08. **Preparation complete; R5 implementation and evaluation remain
open.** No production solver code, routing default, verification gate or paid
run authority changed. R3 human source review remains open.

## What is frozen

The [implementation specification](../specs/reliability_r5_periodic_routing_spec.md)
defines three checkpoints: R5a preparation, R5b opt-in implementation and
fake-worker checks, then R5c one measured baseline/prototype campaign and a
default-adoption decision. This checkpoint closes R5a only.

`scripts/prepare_reliability_periodic.py` produced sixteen cases and 32
deterministically ordered paired requests, without running a solver:

- Four previously inspected R4 periodic cases are regression anchors.
- Six fresh periodic positives vary Vigenère periods (4, 7, 11) and Quagmire
  periods/keyword lengths (6/6, 8/7, 10/8).
- Six fresh controls cover two no-boundary simple substitutions, columnar,
  homophonic, Bifid and random text.
- All eleven plaintext-bearing fresh cases use distinct source documents
  excluded from R0. Actual fresh lengths are 240–708 tokens. All known-key
  roundtrips and intake token counts match; random text has no reference.

The packet uses published local English corpus passages, not new model-written
text. Freshness refers to ciphertext/keys and disjoint source documents, not
an absence of language-model training overlap. It is not automatically a
packet for the deferred GPT-5.5/Astra model comparison.

Worker inputs contain only opaque ID, ciphertext, format, language, ciphertext
hash, arm and shared seed. No plaintext, source, key, family, period, role or
control label is supplied. The controller's grading records are separate and
gitignored. The baseline/prototype comparison is family-blind in both arms.

## Prototype and experiment boundaries

The spec reuses existing harmonic-period diagnostics and native engines. It
freezes eligibility, signal thresholds, at most two periods, a shared
45-second child-process probe deadline, common replay/language validation,
deterministic candidate ranking and fallback to the original route. Quagmire
keyword lengths 6–8 are searched rather than assuming R4's length 7 / period 8
default. No probe candidate becomes independently verified merely by passing
the language gate.

The opt-in prototype remains unimplemented. The eventual 32-arm experiment
retains R4's selected ceiling: 180 seconds wall per entire arm, 720 CPU seconds
per process, four workers, one active arm, and 96 minutes total wall. Probe
overhead counts inside the prototype arm's 180 seconds. All slots, including
failed/interrupted attempts, count; no outcome-dependent retry is permitted.

Acceptance requires all four regression anchors near-exact, material gains on
at least three fresh positives, no material loss on a labeled control, and no
new verified-solution claims. Missing or cross-context pairs cannot establish
a pass. Random text has no character-accuracy claim. Failure leaves the switch
off and informs a new decision rather than an expanded search budget.

## Provenance and reproducibility

Baseline: `e1536c145542e812b5191515d0d061f1b80b6947`, Claude's landed
CLI observability/handshake/lease follow-up on top of R4. Its source changes
are not duplicated here. Only the separate packaging-plan document was dirty
before this work; it remains untouched and excluded from this checkpoint.

Local immutable preparation: `artifacts/reliability_r5/preparation_v1`.
The control plan pins baseline source, models, resources, installed native
binary, interpreter, generator and spec. Native build/source correspondence
is not independently attested. R5b must pin its implementation diff and
campaign adapter before any real arm; this is not yet an executable campaign.

Logical SHA-256 values (the shared `digest` canonical JSON representation,
not hashes of indented file bytes):

| Object | SHA-256 |
|---|---|
| Runtime manifest | `148a50aa27e91d555e9e1c0657b7e8aba025c0134f311e0fe7cd7a85a4226885` |
| Grading packet | `ccc3e4a53cd8838a29e9641fa5ed9815e2eac58196d35d7cfeed185399586aae` |
| Control plan | `4e9a057e479c9ac1b45a36db4488735742ac00fe1ff33f256cd96749a43c75bf` |

Construction was executed twice and returned identical hashes. Outputs are
atomic, immutable and reject differing replacements. To reproduce this
checkpoint before modifying solver/resources:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_reliability_periodic.py \
  --out artifacts/reliability_r5/preparation_v1
```

After implementation, preserve this preparation rather than rerunning its
baseline-only provenance check against the modified solver tree.

## Validation and next checkpoint

**153 focused tests passed in 12.16 seconds**, covering new preparation tests,
R0 construction, R4 worker/campaign/report, ground-truth firewall and the
investigation CLI including Claude's follow-up tests. Additional read-only
construction inspection confirmed every fresh intake token count. No fresh
case IC, candidate, solve score or outcome was inspected.

Next is **R5b**: implement the opt-in probe and its cancellation, retention,
fallback and artifact diagnostics; test with fake workers; pin the campaign
adapter. R5 remains open until the subsequent evaluation and decision. No
live Codex solving session or model/provider comparison is needed at this
checkpoint.
