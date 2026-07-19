# v3 vs Codex/MCP — pre-registered comparison matrix

Systematic follow-up to the ad-hoc dogfood rounds (`mcp_dogfood_results.md`).
Question: **where does the v3 agent loop outperform an MCP-driven external
agent, and vice versa — and is the difference the harness, the client model,
or repo access?** Created 2026-07-18.

## Arms

| Arm label | Client | Harness | Solver access | Cost channel |
|---|---|---|---|---|
| `v3-gpt5.5` | gpt-5.5 (OpenAI API) | v3 investigation lead loop | episode/operator tools + experiment queue | API $ |
| `codex-mcp` | Codex (ChatGPT sub) | 23-tool MCP surface | MCP tools; repo access = escape hatch (recorded) | flat-rate |
| `claude-code-mcp` *(confirmed 2026-07-18; runs deferred — Codex-only for now)* | Claude Code (Max sub) | same 23-tool MCP surface | same | flat-rate |

The third arm is the cell that separates "the MCP surface is limiting" from
"Codex's model/style differs" — without it those stay confounded. It is the
missing Phase-D leg. The same brief files under
`docs/evidence/codex_run_briefs/` serve both MCP arms verbatim.

## Two scoring axes (every cell gets both)

1. **Outcome**: char/word accuracy vs sealed answer or benchmark GT,
   graded post-hoc by `scripts/grade_dual_harness_run.py` (appends to
   `docs/evidence/v3_vs_mcp_results.jsonl`). Convention: unmapped symbols
   count as WRONG (`?`→`x` before scoring); historical pages score
   edit-aware via the benchmark scorer.
2. **Process**: terminal kind (verified declare / honest unsolved / silent
   stop), attestation scalars, verifier spend, WF-7 compliance, escape-hatch
   usage, turns, cost.

Rounds 5–6 proved these dissociate: round 6 was an outcome WIN + process
FAIL (gate unreachable); round 5 was outcome-close + process gap (verifier
never called).

## Protocol

- **v3 arm**: `decipher crack -f <cipher> [--canonical] --language unknown
  --agentic --agent-loop v3 --model gpt-5.5 --max-iterations 25` — no
  `--context`, no cipher-id hints. (Benchmark-backed cases use the benchmark
  CLI with `--benchmark-context none`.)
- **MCP arms**: fresh session, `docs/prompts/decipher-investigate.md` entry,
  ciphertext pasted with NO family/language hints. Escape-hatch use is
  allowed but must be recorded (`--escape-hatch` at grading).
- Language identification is part of the task (C4): neither arm is told the
  language. EXCEPTION — Tier A v3 completions use `--language en` to match
  the conditions their historical Codex counterparts actually ran under;
  Tier B briefs state the language because the v3 benchmark arms received
  it from the loader. Symmetry per cell, not one global rule.
- **MCP-arm run briefs** live at `docs/evidence/codex_run_briefs/` — one
  self-contained file per instance (ciphertext + permitted context +
  WF-7 instructions). Paste/point a fresh session at one file per run.
- **Replication**: n≥2 on discriminating cells; n≥3 on Borg pages (word acc
  swings ±20 pts run-to-run — never compare single Borg runs).
- **Pre-registration**: every cell has a prediction written BEFORE its arms
  run. Deviations are findings.
- Sealed answers live OUTSIDE the repo:
  `~/.config/decipher/dogfood_answers/` (`round6_q3nb_answer.json`,
  `tier_c_pack_answers.json`). Do not paste answers or this file's
  prediction column into an arm session.

## Contamination policy

| Class | Cases | Use |
|---|---|---|
| None (fresh original prose, this program) | rounds 1–6, Tier C pack | capability aggregates |
| Medium (published academic solutions; S-tokens anonymize) | Borg, Copiale pages | capability with flag |
| High (famous, in training data) | Z340, Kryptos | behavioral probes ONLY — a quality jump without tool evidence = contaminated run |

## Tier A — same-cipher pairs (complete/completed)

| Case | v3-gpt5.5 | codex-mcp | Prediction (pre-registered) | Status |
|---|---|---|---|---|
| Borg 0077v (medium contam.) | 77.8/35.2 honest-unsolved, 2 neg attestations, $2.36 (`640e959623f4`) | 74.5/19.0 best branch, silent stop turn 11, 0 attestations (`f58515b7263f`) | — (ran pre-registration) | **DONE (round 5)** |
| Round-6 Quagmire III nb | **38.9%/0 honest fail** at the preflight basin, 1 neg attestation, 15 episodes + 10 experiments, $3.30 (`5029af2de3db`). **RE-RUN post-quagmire-fix (9f4ed28, actuator only, no signpost yet): SOLVED 100% char through the gate** — positive attestation coh 10, verified declare, 2 experiments, $1.87 (`9f547bfcf55a`) | 100% exact via escape hatch; honest declare-unsolved; gate unreachable (`bbd8eabb899b`). Post-fix MCP re-run (`e1dd2d78a0aa`, 2026-07-19): **SOLVED 1.0 IN-SURFACE**, no escape hatch, positive attestation coh 10, declared through the gate (one excluded attempt `3fb8ef066464` ran pre-fix code from an unsynced clone) | v3 solves in-surface via quagmire tool — **FALSIFIED pre-fix (F4); ACCEPTED post-fix on BOTH arms: actuator alone sufficed** | **F4 CLOSED both arms, 2026-07-19** |
| Z340 (behavioral probe) | 40.3%/0 honest fail, 2 neg attestations (basin_wide, coh 1), $3.15 (`407e29ec7c70`) | 38.2%/0, stopped turn 6, 0 attestations (pre-WF-7 run) (`c0f2cb982acf`) | v3 also fails (composite gap); watch memorization — **CONFIRMED; no memorization jump on either arm** | **DONE 2026-07-18** |
| Round-4 composite sub+transposition (PRE-FIX) | 8.7% honest fail, 1 neg attestation (basin_wide, coh 0), tried route work, $3.64 (`1d6d78083226`) | honest fail, correct gap diagnosis (`14f780f21699`) | v3 honest-fails identically — **CONFIRMED** (gap is shared tooling, not harness) | **DONE 2026-07-18** |
| Round-4 composite (POST-FIX ACCEPTANCE) | local CLI + peel route: **100% char, auto-routed** (verified 2026-07-19; the composite program A/C.1/B/C.2). v3 agent re-run PENDING | **UNSOLVED in one Codex session** (`9f3329da4125`): peel found the CORRECT 7-col order `[1,0,4,3,5,2,6]`, but the substitution came out gibberish (coherence 1, honest-unsolved). NOT REPRODUCIBLE — 12/12 local runs (direct fn, MCP runner, 3 model variants) solve this exact cipher 100%. Session artifact (Codex self-reported dedup: 'repeat searches were deduplicated'); recommend a clean re-run. Surfaced a real limitation: composite branches expose no base key, so word-repair can't polish a near-miss composite. | v3 AND codex-mcp both solve in-surface (F4-shaped acceptance: found→fix→two-arm close) | **AWAITING DOGFOOD** |
| Fresh composite (LANTERN, generalization) | local peel: **100%, auto-routed** | **SOLVED 100% IN-SURFACE** (`794b93900137`): diagnose→composite experiment→peel 7-col→substitution→verify→declare, coherence 10, 13.1s, no escape hatch | not overfit to round-4 — **CONFIRMED** | **DONE 2026-07-19** |

**Tier-A read (n=1/cell except 0109v):** neither harness dominates. v3 wins
0077v narrowly; Codex+repo wins round-6 outright (escape hatch); the two gap
probes fail symmetrically. The Tier-B 0109v cell (codex-mcp 95.9 char vs v3's
91.0 basin, via verifier-routed broaden) is the strongest MCP-arm outcome win
so far. Process discipline flipped with WF-7: both post-doctrine Codex runs
closed with verdicts; both pre-doctrine runs did not.

## Tier B — recorded v3/v2 data, Codex arm pending (subscription-only cost)

| Case | Recorded v3 (n) | Recorded v2 (n) | codex-mcp result (2026-07-18, n=1 each) |
|---|---|---|---|
| borg_0109v (preflight-ON equiv.) | 91.0/66.7 ×3 fallback_declared | 95.7–96.8 char ×3 solved | **n=3 COMPLETE**: 95.9/55.9 (`3d4adc42b5f9`, mono→homophonic broaden), 90.7/41.9 (`2feb01efec53`, stayed mono), 91.0/43.0 (`738fa6209d99`, stayed mono). REPLICATION VERDICT: the basin escape is a 1/3 stochastic client choice, not a reliable behavior — median char matches v3's 91.0 basin; and at IDENTICAL 91.0 char, v3 holds a large word edge (66.7 vs ~43), suggesting the v3/preflight key preserves word alignment better. The round-5-era "codex beats v3 char here" headline is deflated; the durable deltas on this page are v3's word accuracy and codex's occasional escape. All three runs WF-7-compliant; replicate 2 logged the first successful in-the-wild MCP repair install on a historical page |
| borg_0109v preflight-OFF | 91.0/66.7 ×1 unsolved $1.43 | 96.8/82.3 ×1 solved | n/a over MCP (experiments ARE preflight-equivalent) — treat ON as comparator |
| borg_0045v | 83.6/0.0 ×2 + 34.1 ×1 | 80.5–85.2/≤32 ×3 unsolved | **83.5/23.3** (`326b796e0331`) — char ties v3's best; word beats every v3 row; honest unsolved |
| copiale_p017 (German; v3's clearest win) | 75.4/0.0 ×1 | 54.6/0.0 ×1 | **73.9/0.0** (`ed95982c2a8a`) — found the same null-mask route; effective tie. v3's p017 edge = reaching the route, and any client that finds experiment_submit inherits it |

**Tier-B read (n=1 codex cells):** char parity across the board; word mixed
(codex better on 0045v, worse on 0109v); all three codex runs WF-7-compliant
with verify episodes. Replication priority: 0109v ×2 more (is the basin
escape reliable or lucky? our own n≥3 Borg rule applies). A dangling second
0109v investigation (`46a5f9ae1306`, active, no terminal) exists from a
restarted session — not graded.
| ~~synth_en_200honb_s6 (control)~~ | 99.9 ×1 | 100 ×1 | **DROPPED 2026-07-18**: the testgen cache regenerated since M6 (F5 detector fired 2026-07-14), so today's cipher ≠ the one the recorded rows measured; also flagged non-discriminative (zenith preflight ~100%) |

## Tier C — fresh contamination-free pack (`docs/evidence/tier_c_pack/`)

All plaintexts are original prose composed 2026-07-18 for this pack; answers
sealed at `~/.config/decipher/dogfood_answers/tier_c_pack_answers.json`.
$0-verified = the named in-repo engine solves the exact ciphertext locally.

| Case | Family / tier | $0 verification | Prediction v3-gpt5.5 | Prediction codex-mcp |
|---|---|---|---|---|
| `tier_c1_vigenere_nb` (509) | Vigenère, no boundaries / control | `search_periodic_polyalphabetic` → **1.0** | solves | solves via experiment `cipher_system:"vigenere"`; NOTE blind router hijacks to homophonic (finding F1) |
| `tier_c2_matrix_rotate_nb` (170) | MatrixRotate cw / agent_assists | pure-transposition screen → **0.9765** (shift-repair variant; exact needs agent repair) | ≥0.97, likely 1.0 after reading repair | uncertain: transform_search enum may not reach the pure-transposition screen; escape hatch likely |
| `tier_c3_quag3_nb` (537) | Quagmire III / agent_assists | Rust shotgun → **1.0** (top-5 all exact) | solves in-surface | blocked in-surface until quagmire experiment route lands (task in flight); replicates round 6; becomes the acceptance pair after the fix |
| `tier_c4_latin_sub_wb` (276) | Latin simple sub, boundaries / language routing | `crack --language la` → **~0.99** (two 1-letter slips, reading-repairable) | solves if it identifies Latin; repair closes B/R slips | same; tests language ID + non-English verifier |
| `tier_c5_bifid_nb` (377) | Bifid p7 / **anchoring-trap probe** | diagnosis says **CONFIDENT mono 0.81 — a misdiagnosis**; mono solvers produce garbage | honest unsolved after mono failure; any confident mono declaration = discipline failure | same; also tests whether the C6 verify gate blocks a pareidolia declare |

## $0 findings from pack construction (2026-07-18)

- **F1 — router hijack extends to periodic ciphers**: blind
  `decipher crack` sent the 509-char no-boundary Vigenère to
  `search_homophonic_anneal` (garbage), even though `observe_diagnosis`
  rates polyalphabetic_periodic strong and `search_periodic_polyalphabetic`
  solves it at 1.0. Same class as the round-4 "no boundaries → homophonic"
  hijack. Backlog: route on the diagnosis, not the boundary heuristic.
- **F2 — pure-transposition screen has a cross-implementation coverage
  gap**: it fails blind (~0.10–0.26) on classic keyed columnar (widths 5–8)
  and rail fence generated by `src/ciphers/transposition.py`, while solving
  its own `TransformPipeline`-library conventions (ladder families). The
  ladder's "all pass" therefore overstates blind coverage of classical
  transposition forms. Backlog: add keyed-columnar orders + ciphers/-package
  conventions to the candidate plan, or reconcile the two implementations.
- **F4 — CLOSED 2026-07-19 (both arms).** CONFIRMED AGAIN 2b3390539458 (Codex, in-surface, 1.0): first live run AFTER the harmonic-fold landing — diagnosis reported fundamental period 8 (not the naive 24), then quagmire3_shotgun → verified declare. Both fix halves (actuator + signpost/fold) now proven together in one live MCP session, 6 turns, $0.015. Post-fix round-6 acceptance:
  v3 re-run solved 1.0 through the gate (`9f547bfcf55a`: quagmire3_solution
  branch, positive attestation coherence 10, explicit declaration, $1.87 vs
  38.9% pre-fix); Codex re-run solved 1.0 IN-SURFACE under a no-escape-hatch
  constraint (`e1dd2d78a0aa`: diagnosis → failed Vigenère experiment →
  quagmire3_shotgun → install → positive verification coherence 10 →
  declared; 4 finalists converged; $0.014 server-side). Both arms succeeded
  on the ACTUATOR half alone (experiment type + misroute guard + playbook
  doc) — the harmonic/signpost half was not yet live in either arm's server,
  and both found the sequencing themselves after plain Vigenère failed. One
  earlier re-run attempt (`3fb8ef066464`) is EXCLUDED: it ran pre-fix code
  from an unsynced clone (see ledger note; produced the sync/freshness
  `server_code` mechanism). Original finding below for the record.
- **F4 (original) — v3 never sequences the Quagmire search (round-6 v3 run,
  `5029af2de3db`)**: episodes called `search_periodic_polyalphabetic` 6×
  (correct family, fails on the keyed tableau) but never
  `search_quagmire3_keyword_alphabet` — a tool IN their own toolset that
  solves this exact cipher at 1.0. Experiments cycled 5 family hints; no
  quagmire route exists there. The v2 loop's structured prior ("plain
  Vigenère failure → run the Quagmire keyword search before rejecting the
  family") never migrated to v3's lean brief, so BOTH the v3 surface and the
  MCP surface share the same hole. The two in-flight tasks (quagmire
  experiment kind; quagmire_keyed sequencing signpost) fix both arms at
  once; re-run this cell afterward as the acceptance test.
- **F3 — Bifid presents as confident mono** (0.81, `peaked_monogram_shape` +
  `letters_substituted`): INV-0 has no fractionation discriminator that
  fires here. Tier-C5 turns this into a live discipline probe.

## Results ledger

Append-only: `docs/evidence/v3_vs_mcp_results.jsonl` via
`scripts/grade_dual_harness_run.py --append`. Narrative findings continue in
`mcp_dogfood_results.md`.
