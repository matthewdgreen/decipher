# Spec: Agent Loop v3 — Milestone M6 (Bake-off + Default-Switch Decision)

Parent design: `docs/specs/agent_v3_design.md` (M6 milestone). Spec
author: Fable. This is the LAST v3 milestone. User authorization for the
live spend is on record (2026-07-14, "good to go on all the big
LLM-based test runs and full sweeps"); the run matrix below is posted to
the user when launched (visibility, not a blocking ask).

M6 has three parts: (A) a small code slice, (B) the live bake-off matrix,
(C) the decision + (iff passed) the default switch. **v2 deletion is NOT
part of M6** (design F3): the coercion machinery is deleted in one
operation at v2 retirement, one release after the default switch.

## Part A — Code slice (implement + review BEFORE the runs)

1. **v2-artifact → v3-state adapter** (design deliverable): a function
   (e.g. `src/investigation/adapter.py: state_from_v2_artifact`) that
   builds an `InvestigationState` from a stored v2 RunArtifact — branches
   from `branch_snapshots` (via the same serialize shapes
   `experiments.py` uses), evidence entries from key findings
   (declaration, final scores), `language`, cipher identity. Purpose:
   `resume-artifact` keeps working on old artifacts after v2 retirement.
   Tests: round-trip on a real stored v2 artifact fixture (one exists
   under `artifacts/`; copy a small one into `tests/fixtures/`), plus a
   synthetic-artifact unit test. The adapter is READ-ONLY over the
   artifact; unknown/missing fields default (old artifacts predate many
   fields).
2. **M5-review note fixes** (small, from the M5 review's findings 4–6):
   (a) `attestation_stale` vs `attestation_required` after branch rename
   — infer staleness by content-hash history, not branch-name presence
   (tools_v2.py:~2364); (b) verify dispatch: error on a named branch
   that doesn't exist (currently silently picks the first existing;
   loop_v3.py:~299) — structured error naming the missing branch;
   (c) emit a `LoopEvent` when the F9 late-turn hint fires so the
   behavior is artifact-visible for the bake-off analysis.
3. **Matrix runner** `scripts/run_v3_bakeoff.py`: drives the Part-B
   matrix via the existing benchmark runner APIs (no shelling to the
   CLI), one artifact per cell-run under
   `artifacts/m6_bakeoff/<loop>/<case>/<seed>/`, incremental JSONL
   summary append after EVERY run (network-robust), `--resume` skips
   completed cells (keyed by loop/case/seed), `--dry-run` prints the
   matrix + cost estimate. Aggregation script or flag producing the
   Part-C table from the summary JSONL (deterministic re-aggregation
   from artifacts).

Suite green (baseline: main after M5 lands); zero regressions.

## Part B — The matrix

Model: `gpt-5.5` for BOTH loops (the confirmed agent model; same model
both sides — this measures the LOOP, not the model). Provider: OpenAI.
`--max-iterations 25` both loops. Seeds: 3 per cell via
`DECIPHER_SEED`-style variation (the runner passes seed into the
solver/SA paths; agent-side sampling varies naturally per run).
Multi-seed is MANDATORY — single-run Borg word accuracy swings ±20 pts.

Cases (5):
- `borg_single_B_borg_0109v` (Borg, well-characterized reference page)
- `borg_single_B_borg_0045v` (Borg, second page)
- one Copiale Track-B page from the five-page packet group (pick the
  page whose automated floor is LOWEST in
  `artifacts/baseline_dta_default/` — the hard one)
- `synth_en_250nb_s4` (M1 parity case, no-boundary synthetic)
- `synth_en_200honb_s6` (hardest homophonic/no-boundary stress case)

Rows per case: v2 default-config, v3 default-config (preflight ON both —
the production comparison), PLUS a preflight-OFF pair for
`borg_single_B_borg_0109v` and `synth_en_250nb_s4` only (the
agent-capability comparison; the M1–M5 lesson is that preflight-solved
pages never exercise the loop machinery).

Total: 5 cases × 2 loops × 3 seeds = 30 runs, + 2 cases × 2 loops × 3
seeds preflight-off = 12 runs → **42 runs**. Cost estimate: Borg/Copiale
~$2.5/run, synthetics ~$1/run → **~$70 ± 20**. Report actual spend per
cell in the summary. If spend tracks >30% over estimate at the halfway
checkpoint, pause and report before continuing.

## Part C — Metrics + decision rule

Per cell (mean over seeds, plus min/max): char accuracy, word accuracy,
declared-vs-fallback (and for v3: was the declaration attested, verify
verdict, verify timing — turn index relative to budget), total tokens
(prompt/completion split), cache reads, cost, iterations used,
wall-clock. Report per-case table + aggregate.

**Decision rule (design):** switch the default loop to v3 iff
(1) v3 mean accuracy ≥ v2 on the matrix aggregate (char AND word), and
(2) v3 cost is materially lower (≥20% lower mean cost per run at equal
or better accuracy). If v3 wins accuracy but not cost (or vice versa),
report and DO NOT switch — bring the trade-off to the user.
The default switch itself (if passed): `--agent-loop` default v2→v3 in
cli.py + benchmark runner, docs updated (CLAUDE.md Running section), v2
remains selectable. One commit, after the decision is reported.

Analysis notes the bake-off must speak to (from M5 acceptance):
- Does v3 verify-then-declare convert fallbacks into declarations
  (run-to-fallback rate v2 vs v3)?
- Verify-at-last-turn: does the F9 hint (now a LoopEvent) fire, and does
  the lead act on it with turns to spare?
- Wrong-basin: any confident-but-wrong declarations on either side (char
  accuracy < 50% with a declared solution)? v3's attestation should make
  these visible/rarer.

## Out of scope

v2 deletion; multipage/packet routes (single pages only — the multipage
route is v2-automated-runner territory); Fable rows (Fable refuses
plain-substitution tasks in the thin runner — recorded 2026-07-14; a
framed-prompt Fable arm is future work); playbook arms (INV track).

## Deliverables

Part A: files + tests + suite counts (reviewed before runs). Part B:
the 42-run summary JSONL + artifacts. Part C: the decision table, the
recommendation, and (iff passed) the default-switch commit. Deviations
reported. No commits except as specified.

## Post-review amendments (BINDING — Fable review: READY WITH AMENDMENTS)

**F1 (seeds → replicates).** `DECIPHER_SEED` does not exist; no runner/
solver seed plumbing is added for the bake-off (it would change the thing
measured). "Seed" = REPLICATE INDEX (1..3), recorded in the summary
JSONL. Per-run variation comes from agent-side sampling (real for
gpt-5.5). Preflight is deterministic across replicates — a feature: both
loops start from identical preflight per case, so replicate variance
isolates loop+agent.

**F2 (adapter reconstruction — the hard step).** A v2 RunArtifact stores
no structured cipher. (a) RELOCATE (copy) `_extract_cipher_block` +
`cipher_text_from_artifact` from `agent/resume.py` (deletion-listed at
v2 retirement) into `src/investigation/adapter.py`; adapter must not
import agent/resume. (b) Plaintext alphabet = `Alphabet.
standard_english()` (v2 never stored one; that pair makes snapshot key
ints meaningful). (c) Structured error when the notation block can't be
extracted (old/compressed artifacts). (d) Cipher identity lands in
`external_context` (InvestigationState has no cipher_id field).
(e) Scope = function + tests ONLY; `cmd_resume_artifact` CLI wiring
ships with the retirement release (`load_artifact_dict` currently
hard-rejects v3 artifacts; cli.py:992ff is v2-only).

**F3 (synth construction path).** `synth_en_250nb_s4` /
`synth_en_200honb_s6` are in NO benchmark split. Construct via testgen:
`TestSpec.from_preset(HARD, en, seed=4)` / `from_preset(HARDEST, en,
seed=6)` (s6 is a deliberate non-default; HARDEST default suite seed is
5) + `testgen.builder.build_test_case` + `PlaintextCache
("testgen_cache")` with `api_key=""` and an ASSERTED cache hit (both
verified warm locally; no generator call). The runner mixes two data
sources: BenchmarkLoader for Borg/Copiale, testgen for synths.
Programmatic-drive precedent: `scripts/run_testgen_suite.py:548-556`.

**F4 (decision aggregate pins).** The switch aggregate uses the 30
preflight-ON runs ONLY (the production comparison; preflight-OFF rows
are analysis, not decision — else borg_0109v/250nb double-weight).
Case-level equal weighting: mean of per-case means. The word-accuracy
clause aggregates ONLY over cases where word accuracy is meaningful
(the two Borg pages; no-boundary synths lack word boundaries and
Copiale Track B scores 0.0 structurally on both sides).

**F5 (stale/required label).** Drop the hash-history idea (no such
primitive). Fix = the dispatcher updates attestation records' `branch`
field when `episode_install_branch` renames on collision (dispatcher
owns both; records are live in state). The stale-vs-required distinction
stays message-level.

**F6 (verify branch arity).** Verify episodes require EXACTLY ONE
branch: structured error on `len(branches) != 1` and on a missing
branch (currently `["typo", "main"]` silently verifies main — the
mislabeled-attestation risk).

**F7 (F9-hint event placement).** The hint renders in context.py which
has no emitter. Mechanism: the LOOP re-evaluates the same predicate
(turns-remaining ≤ 2 AND best branch lacks fresh attestation) and emits
the LoopEvent itself; context.py stays render-only. No context→loop
back-channel.

**F8 (Copiale case).** Pinned: `copiale_single_B_copiale_p017` (lowest
DTA-default floor, 73.9%, per
artifacts/baseline_dta_default/REFERENCE_copiale_default_77.6pct.log).

**F9 (mechanical).** (a) Artifact field is `branches` (not
branch_snapshots); restore via `state.py:_restore_branch_into` (v2
snapshot dicts restore cleanly; extra fields ignored). Fixture: copy
`artifacts/synth_en_40wb_s1/2a13e2895ed7.json` (April-era, no
loop_version, notation block present) into tests/fixtures/. "Round-trip"
= adapter → to_artifact_dict → from_artifact_dict fixpoint (v2→v3 is
one-way). (b) Runner layout: per-cell artifact_dir yields
`.../<replicate>/<test_id>/<run_id>.json` — accepted. (c) Halfway
checkpoint pinned: after run 21, cumulative actual vs cumulative
per-completed-cell estimate; >30% over → stop launching, report, await
go-ahead.

**Confirmed by review (no change):** preflight-off is one constructor
flag identical for both loops; both loops drive programmatically through
one BenchmarkRunnerV2 (agent_loop param); both Borg ids exist in
splits/borg_tests.jsonl; cost decomposition lands ~$60-80 inside the
estimate; the default-switch surface is cli.py:1529/1679 + runner
default; scope contains no covert v2-deletion.
