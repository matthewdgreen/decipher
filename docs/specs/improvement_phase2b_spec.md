# Spec: Improvement Program Phase 2b — `word_repair` Runner Integration

Parent plan: `docs/improvement_program_plan.md` (Phase 2, item 2.3).
Spec author: Fable. Implementer: coding sub-agent.
Depends on: Phase 2a (landed `637a281`). Binding constraints from the 2a
review are in `docs/specs/improvement_phase2a_spec.md` ("Review
follow-ups → Phase 2b spec constraints") — repeated below because they
are load-bearing.

## Goal

`--homophonic-refinement word_repair` (and the composite
`null_masks+word_repair`) becomes a real automated-runner option: after
the homophonic solve (and null-mask bakeoff when enabled), run the
Phase 2a `propose_word_repairs` pipeline on the solved basin, adopt
repairs only on strict ground-truth-free improvement, and record the
full candidate menu in the artifact.

## Binding constraints (from the 2a review)

1. **Lazy imports only.** `analysis/multipage.py` imports
   `automated.runner` at module top; runner's public wrappers sit at EOF.
   The runner must import `analysis.word_hypothesis_repair` /
   `analysis.multipage` INSIDE the refinement function bodies, never at
   module top. Add a comment at the import site explaining why.
2. **Model resolution through the runner's resolver.** Never let the
   library fall back to its CWD-relative `models/ngram5_<lang>.bin`
   default. Add a public alias `zenith_native_model_path` for
   `_zenith_native_model_path` (same alias pattern as `637a281`'s five)
   and pass the resolved path explicitly into every library call that
   scores with the binary model (extend `WordRepairConfig` or the
   function signatures with an explicit `model_path` if the plumbing
   requires it — report which).

## Required behavior

1. **Refinement values.** Extend `HOMOPHONIC_REFINEMENT_CHOICES` (cli)
   and the runner's refinement dispatch with `word_repair` and
   `null_masks+word_repair`. Existing values unchanged. The composite
   runs the null-mask bakeoff first, then word repair on the
   bakeoff-selected result; plain `word_repair` runs on the zenith
   solve + existing repair-chain output.
2. **Page-group construction.** Single-test runs form a one-page group
   (`PageBundle` from the run's cipher + solved key). Multi-page groups
   are Phase 2.4 — do NOT build group plumbing beyond the one-page case,
   but keep the call shapes group-native so 2.4 is additive.
3. **Adoption policy (AMENDED 2026-07-13 after first acceptance run).**
   The original single-signal gate (validation_score_v2 strict
   improvement alone) adopted a candidate on all five packet pages and
   regressed char accuracy on four — while the library's
   adjudication_score correctly ordered all five outcomes (+6.67 on the
   only improving page; −3.00 on the worst regression). Spec error: the
   research pipeline's acceptance mechanism is the collateral
   adjudication, and the gate must compose it. Adopt iff BOTH:
   (a) the library's repair-acceptance verdict accepts the candidate
   (`annotate_acceptance`/`repair_acceptance` — the extracted research
   mechanism with its own margins), AND
   (b) `validation_score_v2` strictly improves.
   Record both signals (verdict + adjudication_score) on the adopted
   and rejected entries in the artifact step.

   **FINAL (menu-only default, 2026-07-13 after composed-gate rerun).**
   The composed-gate rerun reproduced the run-1 adoptions exactly: the
   library's repair-acceptance verdict accepted all five packet-page
   candidates (`runtime_accept`), including the worst regressor with
   adjudication_score −3.00; and adjudication sign does not separate the
   remaining harm either (p035 +1.84, p052 +2.57, p068 +0.87 all
   regressed char accuracy). Conclusion: no available ground-truth-free
   signal safely auto-adopts single-page repairs. Therefore
   `word_repair` (and the word_repair stage of the composite) is
   **menu-only by default**: it computes the menu and records the full
   step (candidate_menu, gate_decisions, counts) plus `would_adopt` —
   the candidate the composed gate would have selected (or none), with
   both signals — but does NOT modify the key/decryption
   (`adopted_reason: "menu_only_default"`). Setting
   `DECIPHER_WORD_REPAIR_ADOPT=1` opts in to the composed-gate adoption
   behavior unchanged (research + multipage experiments). Adoption
   decisions are deferred to multipage evidence (Phase 2.4) and agent
   review of these menus (Phase 2.5), per the parent plan.
4. **Config surface.** `DECIPHER_WORD_REPAIR_*` env vars mapping 1:1 to
   `WordRepairConfig` fields (window size/step, min/max word len,
   max_edits, max_hypotheses, max_hypotheses_per_window, plus the
   acceptance margins). Defaults = the library's (probe-CLI) defaults.
   Parse once, log the effective config into the artifact step.
5. **Artifact step.** New step `search_word_repair` carrying: effective
   config, candidate menu (the `CandidatePacket` dicts from
   `propose_word_repairs` — they already have `text=None`), adopted
   edit set (or explicit `adopted: none` + reason), validation
   before/after, and counts (proposed/prescreened/adjudicated/rejected).
   Step lands via the same append path as `search_null_masks`.
   `scripts/inspect_artifact.py`: render the step compactly (counts +
   adopted edits + validation delta) — follow how `search_null_masks`
   is rendered.
6. **Firewall.** `propose_word_repairs` inputs come only from the solve
   side. Extend `tests/test_ground_truth_firewall.py`'s automated-path
   test (or add a sibling) so the leak assertion covers a run with
   `word_repair` enabled (stub the solver to keep it fast).

## Tests

- Unit: refinement dispatch reaches the word-repair path for both new
  values (mock `propose_word_repairs` to return a canned accepted
  packet; assert adoption applies edits and the artifact step is
  shaped as specified; assert strict-improvement gating rejects a
  non-improving canned packet).
- Config env parsing round-trip (including bad values → error or
  documented fallback, matching how DECIPHER_NULL_MASK_* handles them).
- Lazy-import guard: a test asserting `automated.runner` module does NOT
  import `analysis.multipage`/`analysis.word_hypothesis_repair` at
  module level (inspect `sys.modules` after fresh import, or scan the
  module source for top-level imports).
- Firewall extension per item 6.

## Acceptance evidence (compute, report — do not gate the suite on it)

After the suite is green, run the five-page Copiale evidence packet:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/copiale_evidence_packet.jsonl \
  --solvers decipher --benchmark-root ../cipher_benchmark/benchmark \
  --homophonic-budget screen \
  --homophonic-refinement null_masks+word_repair \
  --artifact-dir artifacts/phase2b_acceptance
```

Report per-page char accuracy vs the baseline in
`artifacts/baseline_20260713/copiale_null_masks/` (by-selection:
74.4 / 76.4 / 69.1 / 59.9 / 69.9). The plan's gate — ≥ baseline on ≥4/5
pages, no page regressing by more than 1 point — is adjudicated by the
orchestrator from your report; your job is accurate numbers plus any
anomalies (e.g., zero repairs proposed on some pages is a valid result —
report it, don't force adoption). Also run one Borg page
(`--suite-file` equivalent or the benchmark CLI) with `word_repair` to
confirm non-German operation, and note runtime overhead per page.

## Out of scope

Multi-page benchmark route (2.4), agent tools (2.5), packet size
trimming (2a follow-up F3 — but if the artifact step balloons past a few
hundred KB per page in the acceptance run, report the number).

## Deliverables

Files changed, suite counts (baseline ~855 passed / 1 skipped), the
acceptance-run table, config/plumbing deviations, runtime overhead.
No commits.
