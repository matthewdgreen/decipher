# Spec: INV Model-Diagnosis Experiment Harness

Parent: the investigator-mode structured model experiments
(docs/improvement_program_plan.md) and the playbook-vs-no-playbook
analysis. Spec author: Fable. Implementer: coding sub-agent.

## Goal

A reproducible harness that runs candidate models over a curated
synthetic multi-family cipher suite and scores each model's cipher-family
**diagnosis** (not full solve), producing a comparison table. It answers:
does a frontier model reliably identify how to attack an unknown cipher,
and how does that vary by model tier? It is the reusable substrate for
(a) the model bake-off (Fable vs gpt-5.6-sol vs gpt-5.5) and (b) the
later playbook ablation.

## Dependency posture (READ FIRST)

Full INV mode (INV-0: family registry, panels, DiagnosisReport) is not
yet implemented — INV-0 is in its second spec review. This harness is
therefore built INV-0-independent, with a **pluggable diagnosis task**:

- **v1 task (interim, buildable now):** the model is given the ciphertext
  plus the EXISTING diagnostic tools (`observe_cipher_shape`,
  `observe_frequency`/`ic`/`kasiski`/`periodic_*`, `cipher_id_report`)
  and asked to (i) name the most likely cipher family and (ii) recommend
  the next attack, in a bounded short run. Scored against the synthetic's
  known family.
- **v2 task (post-INV-0):** swap the task for INV-0's real diagnosis mode
  and score the `DiagnosisReport` (top-family correctness, uncertainty
  honesty). Same suite, runner, models, gate-detection, metrics, report.

Only the task adapter changes between v1 and v2; everything else is
shared and non-throwaway. State this in the module docstring.

## Part 1 — Synthetic suite (`frontier/inv_diagnosis_suite.jsonl` + generator)

A fixed, reproducible suite via `testgen` (fresh plaintexts — famous
ciphers are memorization-poisoned and must NOT be used here). Rows span
families, each with a manifest recording the ground-truth family + seed:

- monoalphabetic substitution (boundaries + no-boundary)
- homophonic substitution
- periodic polyalphabetic (Vigenere)
- columnar transposition
- a composite / off-menu case whose true mechanism is deliberately NOT a
  standard single family (e.g. substitution + transposition, or a periodic
  null pattern) — the anchoring probe: does a model over-commit to a
  standard family?
- (optional) a plaintext/near-random control

Each row: `{case_id, cipher_system, seed, true_family, alphabet_class,
length_band, notes}`. Generation is one-time-cached (testgen convention);
a `scripts/build_inv_diagnosis_suite.py` regenerates deterministically.
Ground truth (family, plaintext) is FIREWALLED — used only in post-hoc
scoring, never in any model prompt.

## Part 2 — Model runner (`scripts/run_inv_model_experiment.py`)

For each (model, case): run the pluggable diagnosis task in a bounded
config (low max-iterations, e.g. 6–8; automated preflight OFF so the
model actually does the work — this is the lesson from M1/M2/M3
acceptance where a solved preflight left the agent nothing to do). Save
one artifact per run under `artifacts/inv_model_experiment/<model>/`.

**Model set:** `gpt-5.5`, `gpt-5.6-sol` (OpenAI), `claude-fable-5`
(Anthropic). Provider auto-routes by model id.

**Served-model safety-gate detection (REQUIRED, eval-integrity):** for
Anthropic runs, capture the SERVED model from each response
(`ModelResponse.raw.model` — verified 2026-07-14 to expose it) and set a
run-level `safety_gate_fired` flag when served ≠ requested (e.g.
requested `claude-fable-5`, served `claude-opus-*`). The gate is
context-dependent (fires on some content, not others), so this is a
per-call runtime check, not a one-time preflight. Thread the served model
+ flag into the artifact. **Gate-fired runs are EXCLUDED from the
comparison table (reported separately as a contaminated bucket)** so the
bake-off measures Fable, not a Fable/Opus mix. If this requires a small
`model_provider`/artifact change to surface `served_model`, make it
(additive; also benefits any future Fable-in-harness run) and pin it with
a test: a fake Anthropic response whose `.model` differs from the request
sets the flag; a matching one does not.

## Part 3 — Scoring + report

Per (model, case) metrics, ground-truth-scored post-hoc only:

- **diagnosis_correct**: did the model name the true family (for the
  composite case: did it detect that no single standard family fits /
  propose the composite — credit either).
- **turns_to_first_correct**: efficiency (turns/tool-calls to first
  correct family statement; ∞ if never).
- **pareidolia_flag**: did it confidently assert a wrong family or a
  readable-but-wrong solve (the bad-basin failure).
- **final_char_accuracy**: if it attempted a solve (informational).
- **cost, tokens, wall_clock**.

Report: `scripts/report_inv_model_experiment.py` → a per-model table
(diagnosis accuracy overall + per family, mean efficiency, pareidolia
rate, cost) + the excluded gate-fired bucket. Deterministic aggregation
from artifacts.

## Part 4 — Tests

- Suite manifest schema + firewall (`assert_no_ground_truth_leak` over
  the rendered task prompt — true_family/plaintext never present).
- Served-model gate-detection unit test (Part 2).
- Runner with a FAKE provider (scripted diagnosis responses) end-to-end:
  produces artifacts, scorer reads them, report aggregates — NO network,
  runs in CI.
- Scoring unit tests (correct/wrong/never-diagnosed cases).

## Run policy (network + budget aware)

Anthropic budget is modest ($20) and the network is unstable. Therefore:

1. **Free, now:** an automated (non-LLM) diagnosis baseline over the
   suite using the existing `cipher_id`/fingerprint — establishes how
   well the current diagnostic code identifies each family (a floor the
   models should beat) and validates the suite/scorer end-to-end at zero
   cost.
2. **Small live smoke, now:** 3 models × 2 cases (one easy family, one
   composite) = 6 runs, bounded iterations — validates the live
   multi-model path, exercises the Fable gate-detection for real, and
   yields a FIRST comparison signal. Est. < $10.
3. **Full sweep: GATED on explicit "go" from Matthew** (like M6) — all
   models × all cases, and later the playbook arms. Est. $20–40; run when
   network is stable.

Report smoke results with the explicit caveat that n is tiny and the
diagnosis task is the v1 proxy, not full INV-0.

## Post-review amendments (BINDING — Fable review, READY WITH AMENDMENTS)

1. **(F1) v1 task = a DEDICATED thin diagnosis entry, NOT `run_v2`.**
   `run_v2` injects the automated `compute_cipher_fingerprint`
   suspicion-ranking into the opening prompt (loop_v2.py:1036-1072,
   independent of `--no-automated-preflight`) and mode-filters the tool
   list — so reusing it hands the baseline's answer to the models
   (circular). Build a thin runner: a `WorkspaceToolExecutor` with an
   OBSERVE-ONLY tool subset (`observe_cipher_shape`, `observe_frequency`,
   `observe_ic`, `observe_kasiski`, `observe_periodic_ic`,
   `observe_phase_frequency`, `observe_periodic_shift_candidates`,
   `observe_cipher_id`), its own short initial context (NO fingerprint
   injection, NO mode filter), and a NEW `meta_declare_diagnosis` tool
   (family enum + recommended_attack + confidence) as the parse target.
   Whether an arm ALSO gets the fingerprint context is the later
   playbook-ablation lever — keep it a config flag, default off.
2. **(F2) Canonical family enum + mappings.** Three vocabularies don't
   match (builder `cipher_system`, fingerprint suspicion keys, model free
   text). Define a canonical family enum in the suite manifest + the
   truth-label→enum and suspicion-key→enum mappings + the composite
   credit rule. The automated baseline maps argmax-suspicion into the
   enum (and literally cannot name the composite — expected, scored as a
   miss).
3. **(F3) Composite = `transposition_substitution`, drop periodic-null.**
   No transformer inserts nulls (all steps are permutations). Build the
   composite as plain-sub `TestSpec` + `transform_pipeline`
   (`cipher_system="transposition_substitution"`), precedent
   `frontier/transposition_homophonic_ladder.jsonl`. The suite builder
   constructs `TestSpec` objects directly (the CLI `--cipher-system` path
   + `TestSpec` reject poly+pipeline).
4. **(F4) Firewall specifics.** Runner builds `TestData` via
   `build_test_case` and calls the task adapter DIRECTLY (no
   `BenchmarkLoader`); neutral `case_id`s OVERRIDE builder test_ids
   (which encode family: honb/vig/tnb/ptnb/thonb/q3nb);
   `assert_no_ground_truth_leak` over the rendered prompt checks
   true_family, plaintext, family-flag tokens, AND the builder
   description string; assert preflight is off.
5. **(F5) Cache honesty.** `PlaintextCache` excludes seed and
   `testgen_cache/` is gitignored: vary `topic` per row for distinct
   plaintexts; record a plaintext hash per case in the manifest (detect
   silent cache regeneration); add "populate plaintext cache" as an
   explicit step-0 of the run policy (tiny LLM cost, one-time).
6. **(F6) Fable cost reporting.** `_PRICING` has no `claude-fable-5`
   prefix → cost reports $0.00. Add a `claude-fable-5` pricing entry
   (additive) OR the report shows raw tokens with cost=n/a for unpriced
   models. State the choice.
7. **(F7) Served-model detection details.** Capture for ALL THREE models
   (OpenAI response objects also expose `.model`), not Anthropic-only.
   loop-side capture must cover BOTH send sites (the main send AND the
   retry — hook the `record_usage` pattern). `RunArtifact.model` is the
   REQUESTED model; add `served_models`/`safety_gate_fired` additively.
   The fake-response test uses a SimpleNamespace whose `.model` differs.
8. **(F8) Naming**: the tool is `observe_cipher_id` (the RunArtifact
   FIELD is `cipher_id_report`) — fix the tool list. Structural template:
   `scripts/run_agent_model_packet.py` (but it shells to `decipher
   benchmark` — this harness must NOT; call the adapter directly).
9. **(F9) Smoke graceful degradation.** If the Anthropic (Fable) call
   fails (credits/network), the smoke degrades to the two OpenAI arms +
   the free baseline rather than blocking; report which arms ran.

## Deliverables

Suite + generator, runner, scorer/report, served-model detection + its
artifact plumbing, all tests; the free automated baseline table and the
small live-smoke comparison table; deviations. No commits (orchestrator
lands).
