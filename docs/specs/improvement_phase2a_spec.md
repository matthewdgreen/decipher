# Spec: Improvement Program Phase 2a — Extract Multipage + Word-Hypothesis Libraries

Parent plan: `docs/improvement_program_plan.md` (Phase 2, items 2.1–2.2).
Spec author: Fable (main session). Implementer: coding sub-agent.
Promotion template: the null-mask precedent (research scripts →
`src/analysis/homophonic_nulls.py`); renaming precedent: commit `f1b8925`
(Copiale-specific identifiers → generic).

## Scope

Extract the two research subsystems that produced the ~79% Copiale basins
into importable, tested library modules. NO runner integration and NO
agent tools in this slice (those are 2.3/2.5). The research scripts keep
working as thin wrappers. Ground-truth firewall applies throughout:
generation/ranking code must never consume benchmark plaintext; post-hoc
calibration helpers must be clearly segregated.

Sources (all under `scripts/research/copiale/`):
- `run_copiale_multipage_experiment.py` (~70KB) — multipage machinery.
- `probe_copiale_word_hypothesis_repair.py` (~91KB) — word-hypothesis
  repair core.
- `probe_copiale_multipage_global_repair.py` (~48KB) — key/assignment
  helpers + acceptance annotation.
- `report_copiale_repair_agenda.py` — `window_damage_score`.

Baseline suite: run before starting (expect ~837 passed / 1 skipped in
the main checkout after the passback slice lands; record the actual
number). Zero new failures. Do not commit.

## Part A — `src/analysis/multipage.py`

Move, generalizing names from Copiale-specific to generic
(page-group/shared-alphabet vocabulary):

- `PageBundle` (dataclass), `build_combined_cipher`, `project_pages`,
  `project_page_with_sources`, `consensus_from_finalists`,
  `page_runtime_metrics`, `score_page_runtime` — from
  `run_copiale_multipage_experiment.py`.
- Post-hoc calibration: `attach_page_scores` uses ground truth. Place it
  in a clearly marked section at the bottom of the module with a
  firewall docstring ("post-hoc grading only — must not be called from
  candidate generation/ranking paths"), or in a sibling
  `multipage_calibration.py` if cleaner. Runtime functions must not
  import or call it.

**Private-runner-import cleanup (required).** The experiment script
imports underscore-private helpers from `automated.runner`:
`_run_homophonic`, `_cipher_text_from_tokens`,
`_automated_candidate_diagnostics`, `_plaintext_quality`, `_word_list`.
Promoted `src/` code must not import underscore-private names across
modules. Add thin public wrappers in `automated/runner.py`
(`run_homophonic_search`, `cipher_text_from_tokens`,
`automated_candidate_diagnostics`, `plaintext_quality_score`,
`load_word_list` — keeping the private names as aliases so nothing else
breaks), and have `multipage.py` use the public names. Do not change the
helpers' behavior.

### Tests — `tests/test_multipage.py`

- Synthetic two-page shared-alphabet fixture (build it in-test: one
  Alphabet, two short token sequences with overlapping symbols):
  `build_combined_cipher` concatenates with correct page offsets/
  provenance; `project_pages` inverts it exactly (round-trip identity).
- `consensus_from_finalists` on hand-built finalist rows with known
  agreement structure → expected consensus assignments and agreement
  rates.
- `page_runtime_metrics`/`score_page_runtime` are ground-truth-free:
  reuse `assert_no_ground_truth_leak` from
  `tests/test_ground_truth_firewall.py` — feed a distinctive plaintext
  through the fixture's solve-side inputs and assert it appears in no
  runtime metric payload.
- Calibration function: separate test proving it grades correctly, and a
  test asserting `multipage` runtime functions do not reference it
  (e.g., no runtime code path calls it — a simple import/attribute
  audit).

## Part B — `src/analysis/word_hypothesis_repair.py`

Extract the generation → conversion → scoring → adjudication pipeline:

1. **Damaged-window detection**: `damaged_windows_for_text` (experiment
   script) + `window_damage_score` (`report_copiale_repair_agenda.py`).
2. **Hypothesis proposal**: same-length dictionary-word candidates for a
   damaged window (min/max word length, `max_edits`,
   `max_hypotheses[_per_window]` knobs — mirror the probe's CLI
   defaults: window 120/step 40, min-len 5, max-len 14, max-edits 3,
   max 40 hypotheses / 6 per window).
3. **Edit-set conversion**: word hypothesis → global symbol edit set —
   `parse_key`, `current_assignment`, `apply_assignment` (from
   `probe_copiale_multipage_global_repair.py`).
4. **Cross-page rescoring**: apply the edit set to the shared key,
   rescore all pages via `multipage.score_page_runtime` (Part A
   dependency).
5. **Collateral adjudication**: `annotate_acceptance`,
   `annotate_repair_evidence`, `variant_rank_key`, `variant_summary` —
   occurrence-level word-island checks that accept/reject a hypothesis.

Public API sketch (adjust to the code's reality; report deviations):

```python
def propose_word_repairs(pages, shared_key, dictionary_path, language,
                         config: WordRepairConfig) -> list[CandidatePacket]
```

- Returns `CandidatePacket`s (`kind="word_repair"`) whose `provenance`
  carries the edit set, per-page score deltas, and collateral evidence;
  `solver_scores` carries the rank-key components. Add the packet-kind
  adapter to `analysis/candidate_packet.py` following the three existing
  adapters (and their tests' patterns). Per the Phase 1 review's F3
  deferral: word-repair packets set `text=None` and carry previews only
  — full decryptions stay out of packets for this kind.
- Language-agnostic: dictionary path + language scoring profile are
  parameters. Nothing German-specific outside defaults.

### Tests — `tests/test_word_hypothesis_repair.py`

- Constructed-damage case: build a synthetic page set where a known
  symbol corruption damages exactly one dictionary word; assert the
  pipeline proposes the true repair, its edit set is correct, and
  adjudication accepts it while rejecting a decoy hypothesis that
  damages other pages (collateral check exercised).
- Config bounds respected (max hypotheses, per-window caps).
- Firewall: `propose_word_repairs` output contains no ground-truth
  plaintext (reuse the leak helper; the fixture's GT never enters the
  call).

## Part C — Scripts become wrappers

- `run_copiale_multipage_experiment.py` and
  `probe_copiale_word_hypothesis_repair.py`: replace the moved function
  bodies with imports from the new modules (keep CLI arg parsing and
  report formatting in the scripts). The other research scripts that
  import from these two keep working (they import via the script module
  — verify `run_copiale_word_hypothesis_batch.py` and
  `probe_copiale_multipage_global_repair.py` still run their `--help`
  at minimum).
- Do not refactor the remaining research-only logic (logograms,
  reading-holes, phrase hypotheses) — out of scope per the consolidation
  ledger.

## Review follow-ups → Phase 2b spec constraints

From the Fable review (LAND WITH FIXES; extraction fidelity verified over
~40 functions with zero unsanctioned logic changes; required test fixes
applied in-phase). Binding on the Phase 2b spec:

- **Circular-import hazard**: `analysis/multipage.py` imports
  `automated.runner` at module top, and the public wrappers sit at the
  END of runner.py. Any future top-of-file `import analysis.multipage`
  (or `word_hypothesis_repair`) in runner.py hits a
  partially-initialized-module ImportError. 2b must import the new
  libraries lazily inside functions, or relocate the wrappers above
  runner's heavy body.
- **CWD-relative model default**: `multipage.py` falls back to
  `models/ngram5_<lang>.bin` relative to CWD; the runner's own resolver
  (`_zenith_native_model_path`) is repo-root-anchored with expanduser +
  language normalization. 2b must route model resolution through a
  public wrapper of the runner's resolver so library callers from any
  CWD score identically to calibration.
- Provenance semantics: word-repair packets' group-level metric deltas
  vs per-page evidence (`page_runtime_evidence`) — 2.5's agent tools
  must present these correctly.

## Acceptance

- Full suite green with the new test modules; no new failures.
- `grep -rn 'from automated.runner import _' src/` returns nothing.
- Both wrapper scripts still execute end-to-end on a small input
  (`--help` plus, if cheap, a smoke invocation documented in your
  report).
- Report: files changed, suite counts, API deviations from the sketch,
  and the exact function-to-module mapping table.
