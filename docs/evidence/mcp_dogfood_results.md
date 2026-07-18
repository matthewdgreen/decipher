# MCP dogfood — Codex-over-MCP crack results (July 2026)

Running log of real Codex sessions cracking generated ciphers through the
`decipher` MCP server (client_name `codex-mcp-client`), from a bare clone.
Purpose: dogfood the server end-to-end AND gather falsifiable evidence for the
§6 tooling-value question. Sealed answers live in the session scratchpad
(`codex_test*_answer.json`); each cipher is original prose (not memorizable).

Char-accuracy is measured by me against the sealed plaintext from the
investigation registry (`~/.config/decipher/investigations/<id>/`).

## Difficulty ladder

| # | Family | Boundaries | Tokens/symbols | New challenge |
|---|--------|-----------|----------------|---------------|
| 1 | monoalphabetic substitution | yes | 223 / 22 | baseline |
| 2 | homophonic substitution | yes | 276 / 51 | multiple symbols per letter |
| 3 | homophonic substitution | **no** | 327 / 50 | + segmentation (repo's hardest homophonic) |
| 4 | substitution + columnar transposition | no | 287 letters | + reorder layer; the "anchoring trap" |
| 5 (planned) | homophonic + transposition | no | — | top of the synthetic ladder |

## Results

### Round 1 — mono + boundaries — investigation (my dogfood, session MCP)
- Verdict: **100% char.** Automated solver cracked instantly; declaration gate
  correctly BLOCKED without verification (`attestation_required`). Gate open
  path not yet exercised.
- Note: my own dogfood used a too-short flat near-pangram first (near-random IC,
  solver couldn't crack a tiny flat sample — an honest statistical-attack limit,
  handled gracefully with ranked candidates + no false confidence).

### Round 2 — homophonic + boundaries — `1b7232f85f11` (Codex)
- Verdict: **100% recovery** (exact sealed plaintext present in state).
- 25 revisions, genuine codex-mcp-client session.
- **Verifier earned its keep:** at verify time one residual key error remained
  (`BRESS`→`BRASS`); independent reader scored lang 0.99 / recov 0.97 /
  coherence 9, LOCALIZED the damage ("likely misread for BRASS"), and REFUSED
  to accept as complete over one wrong word — the diplomatic C6 behavior working
  live. Codex then fixed it (exact plaintext in later state).
- Gap: no final re-verify + declaration recorded (`terminal: null`).

### Round 3 — homophonic, NO boundaries — `c56de7e6c600` (Codex)
- Verdict: **99.7% char** (closest branch; single residual: pos 197 W→I).
  Excellent on a hard class.
- **FAILURE MODE A (repair guard false-reject).** Codex's reading was 100%
  correct (all 5 damaged words identified: OUST→JUST, RESTHE→RESCUE,
  IATTHED→WATCHED, TRAILER→TRAWLER, DETK→DECK). Repair transaction passed
  5/6 acceptance checks and FAILED check 6 `collateral_within_limits` →
  `materially_non_improving`. Cause: the deterministic occurrence-counter judged
  OBJECTIVELY-CORRECT fixes as damage (TRAWLER not in the 5000-word common list;
  TRAILER is). Scalar probe never ran (score_deltas null). Meanwhile an
  independent reader scored the CORRECTED reading 0.97 language confidence.
  → A counting heuristic silently overruled the strongest evidence source.
  Evidence: `docs/evidence/c56de7e6c600_repair_guard_false_reject.json`.

### Round 3b — homophonic, NO boundaries (test 3 cipher) — `776221457325` (Codex)
- Verdict: **98.2% char** on the repo's HARDEST synthetic class (no-boundary
  homophonic, 50 symbols). Strong cryptanalysis.
- **FAILURE MODE B (repair not attempted).** Reader identified 6 correct
  single-symbol fixes (HOLLOWED→FOLLOWED, DUST→DUSK, THANT→THANK, DEAR→YEAR,
  AND ONE→ANYONE, HESTIVAL→FESTIVAL) but **0 repair transactions**: the verify
  verdict was `damage_scope=distributed` (6 scattered errors), which routes AWAY
  from local repair toward broaden — even though each error is an
  individually-simple batch-repairable key edit. attestation: lang 0.93 /
  recov 0.82 / distributed / coherence 6, not accepted.

### Round 4 — substitution + columnar transposition — PENDING (Codex + naive control)
- Cipher issued (287 letters, keyword MASONRY). Naive-Codex control (no MCP)
  running in parallel — the clean tooling-isolation experiment.
- Prediction: naive arm falls into the anchoring trap (IC ~English → misdiagnose
  as mono → solve → scramble). MCP arm has observe_diagnosis + transform_search.
- (fill in results)

## Cross-cutting findings

1. **The system reads correctly but cannot persist the fix** — reproduced TWICE
   (modes A + B), on independent runs. This is exactly the M5.4 repair-reframe
   motivation, now with in-the-wild evidence artifacts. Root: "understanding" is
   solved; "key modification" is the wall.
2. **The independent verifier works and matters** — caught BRESS and the residual
   homophonic errors; refused to over-declare near-perfect-but-wrong decodes.
   This is the clearest thing the tooling buys vs a naive session (which has no
   independent check and may declare a wrong decode confidently — watch for this
   in the control).
3. **Solver strength is high across the ladder** — 98–100% char even on
   no-boundary homophonic. The differentiation between arms is expected to live
   in DIAGNOSIS (round 4) and the CLOSE (verify/gate/repair), not the raw crack.

## In-flight fix (from this evidence)
`docs/specs/verifier_arbitrated_repair_spec.md` (being authored): when the
mechanical repair checks would reject, an OPT-IN path runs a fresh SERVER-SIDE
independent verify on the repaired fork (hash-bound); if the independent reader
prefers it, the repair installs. Preserves every invariant (reader independent,
nothing installs on model say-so, mechanical checks stay default, worse repairs
still fail). Directly fixes mode A; mode B gets a batch-repair doctrine line +
confirmation the arbitration path handles scattered-edit batches.
