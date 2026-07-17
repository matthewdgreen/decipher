# No-LLM Solver Coverage Matrix (2026-07-15)

Ran the existing `AutomatedBenchmarkRunner` (no LLM) against one generated
English case per family (35 families, `scripts/solver_coverage_sweep.py`,
`artifacts/coverage_sweep/matrix.json`). Char accuracy vs ground truth.
**This is a coverage probe, n=1/family — directional, not a leaderboard.**

**Correction (2026-07-16):** the transposition-solver acceptance (n≥2, fresh
seeds) showed `route_transposition` at ~0.36–0.75 on medium cases, NOT the
0.988 originally recorded here — that figure was an easier n=1 probe. Route
remains a GAP (deferred to the existing Rust route screen). columnar/railfence/
redefence/myszkowski/amsco/nihilist_transposition are now SOLVED (~1.000) by
the new transposition_solver (landed 2026-07-16); route + cadenus remain gaps.

## Genuinely solved (real cryptanalysis)

| family | char | why it works |
|---|---|---|
| vigenere / beaufort / variant_beaufort / gronsfeld | 1.000 | periodic polyalphabetic solver |

| a1z26 / morse / tap_code | ~0.99 | **each is a 1:1 per-token substitution** — our symbol-general mono solver cracks any 1:1 encoding regardless of symbol shape |

## NOT solved — the ~0.40 "mono floor" (solver applied the WRONG attack)

hill2x2 0.42, playfair 0.41, two_square 0.41, nihilist_substitution 0.40,
four_square 0.39. These are FAILURES: the solver falls back to a
monoalphabetic hill-climb and scores ~0.40 by chance letter overlap.
Treat as GAPS, not partial solves.

## Gaps needing new solvers

- **Transposition variants** (~0.37–0.39): columnar_transposition, railfence,
  redefence, myszkowski, amsco, cadenus, nihilist_transposition. We crack
  `route` but not these — the transform search doesn't cover keyword-columnar
  permutations. **Investigate the route-works/columnar-doesn't gap first —
  likely a tractable extension of existing machinery.**
- **Vigenère relatives** (~0.37–0.39): porta, autokey_text, autokey_key,
  running_key. Periodic near-misses; small adaptations of the attack we have
  (running_key hardest — no period).
- **Digraphic** (~0.40 floor): playfair, two_square, four_square — need a
  Playfair-family SA over keyed 5×5 squares.
- **Fractionation** (0.18–0.38): bifid, trifid, adfgx, adfgvx.
- **Hill 2×2** (0.42 floor): only ~157k invertible 2×2 mod-26 keys →
  **brute-forceable with a language score** (easiest real solver).
- **Nihilist substitution** (0.40): Polybius + keyword addition.
- **Multi-char encodings** (~0.00): base64, hex, binary, rot47, base32,
  straddling_checkerboard, baconian. Not 1:1 substitutions — these need
  DETECT-and-DECODE (deterministic once identified), which is INV-diagnosis
  territory, not SA cracking.

## Prioritized solver-build plan (value × tractability)

1. **Transposition-family solver** — 7 families at once; we already crack
   route, so start by diagnosing why columnar fails and extend the transform
   search. Biggest single win.
2. **Vigenère-relatives** (porta, autokey ×2) — adapt the periodic attack;
   small deltas. running_key separately (harder).
3. **Hill 2×2 brute-force** — smallest, self-contained; a quick real solver.
4. **Playfair-family SA** — the flagship digraphic solver (roadmap Tier-1 #1).
5. **Fractionation SA** (bifid/trifid/adfgx/adfgvx).
6. **Nihilist substitution.**
7. **Encodings detect+decode** (base64/hex/binary/rot47/…) — deterministic;
   overlaps INV encoding-detection tier. The 1:1 ones (a1z26/morse/tap) are
   already cracked.

All of this is LLM-free and lives in analysis/automated/ciphers (NOT the
Codex-owned agentic system). Solvers should be validated against the
generated benchmark (the generator produces unlimited fresh cases per
family).
