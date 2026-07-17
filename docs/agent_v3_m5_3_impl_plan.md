# M5.3 Implementation Plan + Codex-INV Coordination Contract

**STATUS 2026-07-17:** M5.3 has landed. Its next bounded follow-up is M5.3a,
specified in
`docs/specs/agent_v3_m5_3a_candidate_reliability_spec.md`. That plan addresses
compare semantics, protected finalist retention, sparse-null coverage, local
replays, and synthetic controls before any further paid Borg smoke. The
historical implementation sequence below remains as the record of completed
M5.3 work.

2026-07-16. Fable implements M5.3 (v3 control reliability, spec
`docs/specs/agent_v3_m5_3_control_reliability_spec.md`).

**HISTORICAL UPDATE 2026-07-17: Codex's INV work was paused while M5.3 landed;
this ownership pause is no longer active. Fable
had full ownership of the whole repo for that task; there was no live conflict.
The file-ownership / shared-file guardrails below are therefore RELAXED (touch
whatever the slice needs, incl. tools_v2.py/cli.py freely). The PHASE SEQUENCE
still stands — it is driven by intra-M5.3 coupling on `loop_v3.py`, not by
Codex. The ownership map below is retained only as the implementation record.**

## File ownership

**M5.3 (Fable) edits:**
`src/investigation/{loop_v3,episodes,context,actions,reading,state,experiments}.py`,
`scripts/inspect_artifact.py`, and — in `src/agent/tools_v2.py` — ONLY the
`AttestationPolicy` region (~2337) and the `experiment_submit` schema. Tests:
`tests/test_{loop_v3,episodes,lead_context,experiments,verify_attestation,
hypothesis_actions,reading,lead_context,m6_m5_note_fixes}.py` and new M5.3 tests.

**INV (Codex) edits:**
`src/investigation/{families,diagnosis}.py`,
`src/analysis/{panels,coherence,numeric_code,null_baseline}.py`,
`src/benchmark/unsolved.py`, the INV scripts
(`scripts/{build,run,report}_inv_*.py`, `scripts/research/calibrate_inv0_scoring.py`),
`src/cli.py` (the `diagnose` subcommand), and — in `src/agent/tools_v2.py` —
ONLY the `observe_diagnosis` tool def + handler (~6382). INV tests per
`docs/inv_index.md` §5.

## The one genuinely shared file: `src/agent/tools_v2.py`

M5.3 touches `AttestationPolicy` (~2337) + the `experiment_submit` schema.
INV touches `observe_diagnosis` (~6382). These regions are far apart, so
additive edits should not textually conflict. Protocol:
- Each side edits ONLY its own region; do not reformat/move unrelated code.
- Landing is by diff-apply with a collision check (the established pattern);
  if a hunk conflicts, stop and coordinate rather than clobber.
- The tool-count assertion / `TOOL_DEFINITIONS` list is a shared line — whoever
  adds a tool updates it; the other rebases. (M5.3's `hypothesis_test_words`
  batch lives in `actions.py` COMPOSITE defs, NOT tools_v2 — so M5.3 adds no
  new tools_v2 executor tool; INV's observe_diagnosis already landed. Low risk.)

`src/cli.py`: INV owns the `diagnose` subcommand; M5.3 does not need cli.py
(Slice 7's analyzer work is in `scripts/inspect_artifact.py`, not cli). If
M5.3 unexpectedly needs cli.py, coordinate.

## M5.3 phase sequence (each: worktree off prior HEAD → Fable review → land)

Files are heavily coupled on `loop_v3.py` (phases 1/3/4/5) so phases land
SEQUENTIALLY, not in parallel.

1. **Slice 1 — hard episode budgets + per-run cost ceiling.** episodes.py
   (clamp `max_tool_calls` to 1..default; per-call enforcement with synthesized
   `budget_exhausted`; submit-only reserve; restore reading envelope
   16/8192/300), loop_v3.py (`max_cost_usd` checked before every paid send incl.
   mid-episode). Foundational safety — first.
2. **Slice 3 + B1 — batched/cached `hypothesis_test_words`.** actions.py (batch
   core; menu cache keyed on EXACT builder inputs; mint+expose word_id/token
   anchors), tools_v2 untouched (composite lives in actions). Perf substrate.
3. **Slice 2 + B2 + Slice 4 — saturation state machine + host-validated
   acceptance.** state.py (`interpretation_id`/`interpretation_digest`, counters,
   resume), context.py (`repair_exhausted` phase, fail-closed phase map,
   pending-experiment latch), loop_v3.py (process-vs-evidence classification +
   `retry_of`; host-validated repair acceptance, default-deny). Tightly coupled.
4. **Slice 6 + B3 — multi-field verifier + C6 gate reversal + 7-consumer
   migration.** episodes.py (`_VERIFY_SCHEMA` new fields), tools_v2.py
   (`AttestationPolicy` positivity gate), state.py (`AttestationRecord`
   round-trip + legacy mapping), loop_v3.py (`_is_positive`, fallback tier),
   context.py (routing on new fields + flagged threshold defaults). Cross-cutting.
5. **Slice 5 — typed experiments.** experiments.py (expose the real per-type
   config schema model-facing), + the `experiment_submit` schema in tools_v2.py.
6. **Slice 7 — observability + record-replay harness.** inspect_artifact.py (v3
   shape fix, new sections, TRIMMED fixture + firewall test); plus generalize
   Sequence B into a reusable transcript→scripted-provider replay harness (the
   $0 way to regression-test host logic against recorded model behavior).

## Testing boundary (per user: "continue until you need major testing")

Each phase must pass **Sequence A** (focused local tests, no paid model) before
landing. After all phases, run **Sequence B** (scripted end-to-end replay, $0).
**STOP before Sequence C** (the one paid targeted smoke) — that is "major
testing" and requires explicit user approval. Do NOT run a paid model during
implementation.

Baseline note: `tests/test_lead_context.py::test_negative_partial_attestation_
creates_repair_action_menu` is currently red (stale assertion, predates M5.3);
Slice 2 (phase 3) claims and fixes it per Verification A.
