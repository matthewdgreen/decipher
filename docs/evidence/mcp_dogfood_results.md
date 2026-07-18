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

### Round 4 — substitution + columnar transposition — SELF-TEST (no LLM, $0)
I ran the composite through the MCP server myself (automated solver only) to
answer "would v3 handle it?" **Verdict: NO as-is — but it's a detection/routing
gap, not a capability gap. Proven three ways:**
1. Quick fingerprint: `monoalphabetic 1.00`, transposition 0.25 (anchoring trap:
   transposition preserves letter freqs → IC 0.0603 reads as clean mono).
2. Deep INV-0 diagnosis (9 panels, `observe_diagnosis`): `CONFIDENT: mono 1.00`.
   The `order_layout` panel's `letters_substituted` atom ACTIVELY WEAKENS
   transposition ("freqs displaced → substituted, not transposed"). It treats
   substituted vs transposed as COMPETING, not COMPOSABLE. There is a
   `transposition_homophonic` family but **NO substitution+transposition
   composite family**.
3. `transform_search: full` experiment: the router chose route=homophonic
   ("single word-group + dense symbols → no-boundary homophonic"),
   transform_pipeline=None — NEVER tried a transposition. Result 6.7% garbage.

ISOLATION (decisive): I de-transposed by hand (known key) and ran the plain
substitution solver on the correct-order stream → **exact plaintext, 100%,
5s, dict_rate 0.85** (BENEATH THE ABBEY A FORGOTTEN CRYPT...). So the
substitution/no-boundary stack is fully capable; the ENTIRE gap is:
nothing detects/peels the transposition layer.

THE GAP (backlog item — "composite sub+transposition"): the stack needs to
recognize that a high-letter-accuracy-but-no-word-structure solve implies an
ORDER layer, and screen columnar/other transpositions around the substitution
solve. Currently (a) diagnosis has no composite family + letters_substituted
suppresses transposition, and (b) the solver router's "no boundaries →
homophonic" heuristic hijacks transform_search away from transposition. This
CONFIRMS the INV "composite universally missed" finding at the TOOLING level
(not just the LLMs). A v3 LLM agent likely can't rescue it either, because its
transform_search tool routes away from the transposition.

Implication for the ladder: Round 5 (homophonic + transposition) would hit the
SAME gap and fail worse — NOT a useful next test until the composite
detection/routing is fixed. Better next rungs that avoid this gap: Vigenère
(polyalphabetic, different attack) or non-English (Latin/German, historical
manuscript direction).

### Round 4 — LIVE ARMS (MCP-Codex vs naive-Codex control)

**MCP-Codex on the composite (round 4) — `14f780f21699`: HONEST FAIL, validates the gap.**
A real Codex session WITH the full MCP surface + repo access tried: family
diagnosis, direct bijective substitution annealing, affine/Caesar, and a broad
**364-candidate transform search** with solver confirmation — and could not peel
the transposition. It explicitly diagnosed the homophonic MISROUTE ("negative
evidence for the route, not the diagnosis") — the same routing bug I found in
the $0 self-test — and then **recorded it honestly as UNRESOLVED rather than
inventing a reading**. So: (a) a live LLM agent confirms the composite defeats
the stack at detection/routing (my isolation conclusion holds under a real
agent, not just automation), and (b) the epistemic discipline WORKED — faced
with a cipher the stack genuinely can't do, it declared honestly-unsolved
instead of hallucinating a plaintext. That refusal is the single clearest
"tooling matters" signal in the experiment.

**Naive-Codex control on the homophonic (round 3, 327/50) — FULL SOLVE, via an
external tool.** With a shell + web and NO MCP, Codex frequency-profiled it,
recognized homophonic, computed the unicity distance (~77 vs 327 → solvable),
then **cloned `github.com/freichmann/cDecryptor`** (a C++ homophonic SA solver),
wrote prep scripts, fixed a compile error, and recovered the EXACT plaintext
100%. Humbling for the "our solver is the moat" thesis: a frontier model with a
coding environment will FETCH OR BUILD the solver it needs. BUT two caveats:
(1) ~a dozen build/debug steps vs one `experiment_submit` (effort/latency), and
(2) **no independent verifier** — it declared "Cracked" with zero external
check. It was right here, but the BRESS/TRAWLER near-misses (rounds 2-3, which
our verifier CAUGHT and refused to over-declare) are exactly where a naive
"trust the solver output" declares a wrong reading undetected.

**Refined tooling-value conclusion (the honest §6 read):** the durable moat is
NOT raw solving power (contestable — they clone cDecryptor). It is (a)
effort/latency, (b) EPISTEMIC DISCIPLINE — the independent verifier + gate that
convert "the solver emitted X" into "an independent reader accepts X," catching
near-misses and refusing to over-declare (round-4 honest-unsolved + round-2/3
near-miss catches are the evidence), and (c) reproducible provenance for scored
eval. This matches the earlier harness retrospective: the tooling is the
measurement instrument + discipline, less the raw solver. The naive arm cloning
cDecryptor is almost a perfect illustration of "the pile of tools is
contestable; the epistemic scaffolding is the moat." Caveat: the two live arms
ran DIFFERENT ciphers (naive=round3 homophonic, MCP=round4 composite), so this
is not a same-cipher head-to-head; a clean A/B would run both arms on both.
