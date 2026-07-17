# Investigator Mode (INV) — File Index & Continuation Guide

Everything needed to continue the INV work, 2026-07-16. INV = the local,
LLM-free cipher-family DIAGNOSIS system (`decipher diagnose`) + the
model-diagnosis experiment harness + the long-range family roadmap.

## Boundary-safety note (important while Codex reworks the v3 agent loop)
The INV diagnosis engine (families/diagnosis/panels/coherence/numeric_code/
null_baseline/unsolved) is NOT among the files Codex is reworking
(loop_v3/episodes/context/actions/reading/sessions). **INV extension is a
safe parallel track.** Exceptions: `observe_diagnosis` lives in
`src/agent/tools_v2.py` and `diagnose` in `src/cli.py` — shared surfaces;
coordinate before editing those.

## 1. Design & vision (read first)
- `docs/specs/investigator_mode_v3_design.md` — THE design (mode architecture,
  family-registry philosophy, the "Recorded design notes": cross-run case-file
  memory, the playbook ablation axes, the family-roadmap pointer).
- `docs/unknown_cipher_investigator_mode.md` — predecessor/broader investigator
  vision (background).
- `docs/unknown_cipher_agent_plan.md` — earlier agent plan (background).

## 2. INV-0 (the LANDED diagnosis core) — spec + reviews + calibration
- `docs/specs/investigator_inv0_spec.md` — THE implementation spec, incl. ALL
  binding post-review amendments (cleared after THREE review cycles). Authority
  for INV-0 behavior/scoring.
- `docs/specs/investigator_inv0_review_findings.md` — review 1.
- `docs/specs/investigator_inv0_review2_findings.md` — review 2 (the
  "recalibrate by EXECUTION, not on paper" mandate).
- `scripts/research/calibrate_inv0_scoring.py` — the REPRODUCIBLE scoring source
  of truth (imports the real cipher_id priors + Beale streams). Also runs
  in-suite as `tests/test_calibration_inv0.py`. **Any INV-0 scoring change must
  re-verify here.**
- Landed at commit `b2a2f65`.

## 3. INV model-diagnosis experiment (model comparison harness)
- `docs/specs/inv_model_diagnosis_experiment_spec.md` — harness spec (9 binding
  amendments; served-model safety-gate detection; playbook-ablation posture).
- `scripts/build_inv_diagnosis_suite.py` — suite generator (step-0 cache).
- `scripts/run_inv_model_experiment.py` — the runner (observe-only executor,
  meta_declare_diagnosis, served-model capture, --max-tokens).
- `scripts/report_inv_model_experiment.py` — deterministic aggregator/report.
- `frontier/inv_diagnosis_suite.jsonl` — the 6-case suite manifest.
- `artifacts/inv_model_experiment/full_sweep_report.json` + `full_sweep.log` —
  RESULTS (GITIGNORED, local only). Headline: gpt-5.5 & gpt-5.6-sol 83%,
  claude-fable-5 50% (mono-task refusals; gate never fired), free baseline 67%;
  ALL THREE models missed the substitution+transposition composite (universal
  anchoring). Landed at `433e8c1`.

## 4. Long-range queue + enabling work
- `docs/inv_family_roadmap.md` — the tiered list of families not yet diagnosed
  plus the revised implementation sequence: canonical taxonomy/coverage,
  diagnosis calibration, Tier-0 representations, transposition probes,
  layered/composite diagnosis, unknown-language handling, then the broader
  family tail. This is the concrete discriminator queue.
- `docs/specs/cipher_benchmark_generator_spec.md` + the generator
  (`scripts/generate_cipher_benchmark.py`, `src/ciphers/*`, `src/testgen/*`) —
  contains 35 registered generators (34 new relative to the coarse INV
  registry). Convention: every diagnosable mechanism gets generated
  calibration evidence at the appropriate hierarchy level.
- `docs/solver_coverage_matrix.md` + `docs/specs/transposition_solver_spec.md` —
  the solver track that pairs with INV families (what we can SOLVE vs DIAGNOSE).
- `docs/improvement_program_plan.md` — sections: "Investigator-mode structured
  model experiments", "Benchmark generation on demand" + generator pipeline,
  "INV family roadmap", "External comparison tools" (CipherLens = the natural
  aligned-data diagnosis comparison after INV-0.5; Ciphey overlaps primarily
  with Tier-0 representations).

## 5. Code entry points
- Engine: `src/investigation/{families,diagnosis}.py`,
  `src/analysis/{panels,coherence,numeric_code,null_baseline}.py`,
  `src/benchmark/unsolved.py`.
- Interfaces: `decipher diagnose` (`src/cli.py`), `observe_diagnosis` tool
  (`src/agent/tools_v2.py`).
- Tests (9, all green): `tests/test_{diagnosis,calibration_inv0,panels,coherence,
  families_registry,numeric_code,null_baseline,unsolved_loader,inv_model_experiment}.py`.

## 6. Status tracking (outside the repo)
The session auto-memory (`.../memory/project_state.md`,
`.../memory/ciphey_comparison_tool.md`) tracks live INV status, the full-sweep
numbers, CipherLens, and the gate-lift decisions. Not in the repo.

## How to continue (concrete)
- Diagnose: `.venv/bin/decipher diagnose path/to/ct.txt` or
  `echo "S001 S002 | S003" | .venv/bin/decipher diagnose - --json`.
- Re-verify scoring: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_calibration_inv0.py -q`.
- Rebuild the model suite / run / report: the three `*_inv_*` scripts in §3.
- **Highest-value next INV work (LLM-free, Codex-safe):** execute INV-0.5 from
  the design: first generate the canonical support matrix and expand the
  held-out diagnosis/calibration benchmark; then add Tier-0 representation
  detection, broad transposition evidence with solver-backed variant probes,
  and composition-aware diagnosis for substitution+transposition. In parallel,
  INV-1 should begin as a thin persistent case file, not the full presentation
  surface. Run the report/playbook model ablation only after the expanded local
  benchmark exists. Run CipherLens only after aligning labels and test data.
