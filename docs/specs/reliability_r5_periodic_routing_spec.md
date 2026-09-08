# R5 — Bounded periodic-family probes

2026-09-08. Preparation specification, before prototype implementation or
evaluation. R3 human source review remains open. No paid/model calls, new
cipher family, or verification-policy changes are included.

## Checkpoints

1. **R5a — Freeze:** this specification, fresh packet, provenance and tests of
   construction/firewall. No solver execution or production changes.
2. **R5b — Implement:** opt-in bounded probe, diagnostics, candidate retention,
   fallback and fake-worker acceptance. Commit before real evaluation.
3. **R5c — Measure and decide:** one frozen 32-arm local campaign; publish all
   results and either adopt the default or leave it off. No outcome-dependent
   retries, replacement cases, or threshold changes.

The user's 2026-09-08 request to continue starts this selected R5 sequence.
It does not authorize the deferred model comparison or paid R3 experiment.

## Problem and scope

R4's four Vigenère/Quagmire synthetic pairs improved from 31–42% blind
character recovery to 100% with family metadata. The content router currently
chooses transposition/composite/homophonic before testing periodic structure.
Existing `compute_cipher_fingerprint` and `estimate_fundamental_period` already
provide cheap, ciphertext-only periodic evidence. Reuse those functions and
the native periodic engines; do not introduce another cipher classifier.

R4's Quagmire cases used keyword length 7 and cycleword period 8, matching the
label-aware search defaults. Those results do not establish arbitrary-period
or arbitrary-keyword recovery. The new packet varies both dimensions. Search
coverage and language ranking are separate possible failure causes.

Initial scope is English, entirely A–Z input, 180–1200 tokens, unknown family,
with no explicit transform, known parameters, solver hints or custom periodic
search mode. Explicit family labels and existing directed searches retain
their behavior. Other languages, shorter/longer inputs, symbol encodings and
supported inputs without periodic evidence retain the existing route.

## Frozen prototype rules

These are falsifiable engineering choices, not calibrated success claims.
Do not adjust them against fresh packet outcomes within this campaign.

### Diagnosis

- Compute periodic IC for periods 2 through `min(20, token_count // 25)`.
  Use the existing harmonic-fold estimator with the median periodic IC as
  its null and the existing Kasiski support.
- A proposed period must have mean phase IC at least 0.055, exceed both
  global IC and median periodic IC by at least 0.010, and leave at least
  25 tokens in every phase. A harmonic estimate alone does not pass this gate.
- Probe at most two qualifying periods: the qualifying harmonic fundamental
  first, then remaining periods in descending excess over median, smaller
  period breaking ties. Do not use source labels, expected periods or keys.
- Record input eligibility, the full bounded IC table, the median/global IC,
  harmonic evidence, selected periods and skip reasons. Diagnostics do not
  claim a cipher identification is certain.

### Bounded search and selection

- Prototype is opt-in (`DECIPHER_PERIODIC_ROUTING=probe_v1`); unset/`off`
  preserves current behavior during development and evaluation. Unknown values
  fail clearly. Default adoption is a separate R5c decision.
- Run probes in a disposable child process group with a **45-second total wall
  deadline**, including startup and validation. A native kernel without a
  cancellation API must not be called on the controlling process/thread.
  Cleanup must finish before fallback starts; report cleanup failures honestly.
- First run the existing refined Vigenère/Beaufort/Variant Beaufort screen at
  only the selected periods, retaining at most six distinct finalist texts.
- A candidate is eligible for delivery only if it consumes all input tokens,
  has exact length-preserving mode-specific key replay, and the existing
  finalist validator reports `coherent_candidate` with strict word-hit score
  at least 0.25, dictionary rate at least 0.78 and pseudo-word fraction at
  most 0.25. These are unverified language signals, not solved acceptance.
- Rank eligible finalists by existing validation score, then common
  normalized quadgram score, then smaller period, then text hash. Do not
  compare the raw native scores of different solver families.
- If no ordinary-periodic candidate is eligible, invoke Rust Quagmire III
  shotgun once across selected periods and keyword lengths **[6, 7, 8]**:
  5000 hillclimbs, 250 restarts per combination, four threads, seed 61001,
  slip 0.001, backtrack 0.15, no initial keywords and top six finalists.
  The shared 45-second deadline is authoritative even if the proposal budget
  has not finished. Do not silently fall back to the Python search engine.
- Apply the same replay/validation gate and deterministic ranking. If a
  candidate qualifies, deliver it as `completed`, never automatically
  verified/solved. Preserve mode-specific key state and bounded finalist
  evidence in the artifact rather than inventing a substitution key.
- Otherwise retain any fully published finalist evidence, record rejection,
  timeout or exception, and run the unchanged original route. Failed probe
  evidence must not replace the fallback result. The 45-second probe budget
  is shared across both engines, not renewed per call or candidate.

The real evaluation gives the entire prototype arm the same 180 seconds as
baseline, including probe overhead; it does not grant 180 seconds anew to
fallback. Outside the campaign, the opt-in adds at most 45 seconds plus
bounded cleanup to existing runner behavior, not a new general solve deadline.
Document that latency tradeoff before any default adoption.

## Frozen evaluation packet

Four R4 periodic inputs become **engineering regression anchors**, no longer
unseen holdouts. Twelve fresh cases follow this construction schedule:

| Pair | Periodic positive | Letter target | Non-periodic/control |
|---|---|---:|---|
| 1 | Vigenère, period 4 | 240 | No-boundary simple substitution |
| 2 | Vigenère, period 7 | 360 | No-boundary simple substitution |
| 3 | Vigenère, period 11 | 600 | Columnar transposition, width 9 |
| 4 | Quagmire III, period 6 / keyword length 6 | 360 | Homophonic substitution |
| 5 | Quagmire III, period 8 / keyword length 7 | 500 | Unsupported Bifid, period 7 |
| 6 | Quagmire III, period 10 / keyword length 8 | 700 | Uniform random A–Z |

Targets use whole-word prefixes; record actual token counts. Random length
matches its positive exactly; other pairs share the target, not a plaintext.
All eleven plaintext-bearing fresh cases use distinct source documents,
excluding every R0 source document and duplicate normalized passage content.
Select deterministically from the shipped English plaintext library: sort
records by ID, keep the first sufficiently long record per source, exclude
R0 sources/content, then shuffle documents with seed 20260908. Never select or
reject by solver result, IC, difficulty score, candidate fluency or known keys.
Generation seeds are 62000 through 62011 in interleaved pair order.

Exact encryption/decryption roundtrips are construction checks only. Plaintext,
keys, family/role, periods, source IDs and expected outcomes stay in the
gitignored grading store. Runtime input has only opaque ID, ciphertext, format,
language and ciphertext hash. Worker arm configuration adds only baseline vs
prototype and the shared search seed; both arms are family-blind. The control
label must not suppress probes at runtime.

Published corpus material can overlap language-model training. Freshness means
new ciphertext/keys and disjoint source documents relative to R0 and within
this packet, not an LLM contamination-free benchmark. Do not use this packet
as the deferred GPT-5.5/Astra comparison automatically.

## Campaign and acceptance

Exactly 16 cases × two arms = **32 slots maximum**, no development repeats.
Use seed 61001, homophonic `screen`, transform search off, same models and
resources, four workers and one active arm. Each arm receives 180 seconds
wall and 720 CPU seconds per process; campaign wall ceiling is 5760 seconds
(96 minutes). Schedule adjacent pairs with deterministic counterbalancing
from opaque IDs, then freeze the complete order. Record fixed internal RNG
limitations; seed echoes are not proof of independent replication.

The baseline is the unchanged source tree at `e1536c1` (Claude's CLI follow-up
on top of R4). Baseline and prototype must differ only by the opt-in routing
switch in the committed R5b tree; verify all unrelated solver code against
that baseline and record both revisions plus complete diff and file hashes.
Pin installed native binary, models, dictionaries/resources, interpreter,
generation code/spec, packet, request and harness hashes. A differing native
build cannot silently enter one arm.

Use R4's durable attempt ledger, inherited campaign lock and no-retry rules,
but do not mutate its frozen artifacts or relax its baseline provenance check.
Install verified process-group cleanup authority before the first real arm;
do not repeat the R4 mid-campaign sandbox-context transition. Freeze and test
the R5-specific campaign adapter before evaluation. Per-attempt outcomes must
distinguish timeout, worker error, interruption, cleanup failure and complete
artifact. No missing outcome is silently scored zero or removed from a pair.

Grade only after the campaign ends. Report all sixteen pairs, route/probe
decisions, token equality/replay, delivered and best-saved quality, elapsed
time, failure/timeout and status semantics. A valid empty delivery on a
labeled case is 0% recovery, not missing evidence; random text has no accuracy
label. Random/unsupported controls require no new verified-solution claims.

Acceptance follows the R4 decision: all four anchors at least 99% character
recovery; at least three of six fresh positives gain at least two percentage
points over baseline; no labeled control loses at least two points; no new
unsupported verified/solved claim. All required pairs must be complete and
context-matched to claim a pass; otherwise acceptance is unresolved or failed,
not an invitation to replace a case. Random quality is explicitly not scored.

On failure, leave the switch off and attribute the gap using saved diagnostic
evidence: eligibility/period diagnosis, probe timeout, engine coverage,
candidate ranking or fallback loss. Any new experiment or changed threshold
requires a separate recorded decision and fresh acceptance evidence.

## Required tests before real evaluation

- Deterministic construction, source exclusion/duplicate rejection, all varied
  periods/key lengths, independent arithmetic inverses, exact intake lengths.
- Strict runtime allowlist, immutable/hash-bound packet, negative labels never
  reaching worker, complete counterbalanced 32-slot schedule.
- Eligibility/IC/harmonic gates, noise/high-global-IC rejection, ties and
  at-most-two periods; explicit labels/transforms/hints and off switch unchanged.
- Both engines consume one deadline; fake hung/crashed child and descendants
  are reaped; uncertain cleanup prevents starting overlapping fallback.
- No eligible candidate, invalid replay, weak validation, missing native
  module, timeout and empty output all preserve fallback semantics.
- Bounded distinct candidate evidence, exact delivered artifact text, preserved
  key/pipeline replay and no new solution/verification assertions.
- Artifact inspector human and LLM summaries expose diagnostic/probe/fallback
  evidence without requiring a provider call.
- Campaign no-retry/resume locking, provenance drift, budget reservations,
  post-hoc grading joins and missing-result acceptance cannot pass.
