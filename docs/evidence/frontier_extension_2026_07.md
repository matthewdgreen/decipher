# Frontier extension — July 2026 (what recently crossed over)

Answer to "have we extended the frontier?": **yes, three capabilities moved from
unsolved → solved this cycle, and one new gap surfaced.** The existing automated
frontier (`frontier/automated_solver_frontier.jsonl`) predates all of it.

## Crossed over (unsolved → solved)
1. **Substitution + transposition composite** (round-4 class). Was the "composite
   universally missed" gap (failed on every arm; no solver). Now: diagnosed
   (`substitution_transposition` family), auto-routed (content heuristic), solved
   (peel-and-solve), reachable in-surface (`composite_substitution_transposition`
   experiment). Verified: fresh 332-char case → 1.0. **No frontier row exists** —
   add as `known_good`.
2. **Blind Quagmire III, no boundaries** (round-6 class). Now solved via the
   `quagmire3_shotgun` experiment (Rust keyword-alphabet search). Verified: fresh
   449-char case → 1.0. `polyalphabetic_ladder.jsonl` has quag rows but not a
   blind no-boundary anchor — add one.
3. **Keyed columnar (F2) — the MODULE.** `analysis/columnar_search.py` recovers
   blind keyed columnar 100% where the geometric pure-transposition screen scored
   ~0.10. Built during the composite program.

## New gap surfaced (the current edge)
4. **Standalone keyed columnar exposure.** The F2 module (3) lives ONLY inside
   the composite peel (`_peel_order_layer`). A PLAIN keyed columnar (no
   substitution) auto-routes to the geometric transposition screen, which fails.
   So the capability EXISTS but is not reachable for standalone keyed columnar.
   **Backlog:** expose `search_keyed_columnar` as a standalone transposition
   route/experiment. This is fs3 in the agentic suite (an honest open-frontier
   probe).

## Still-open frontier (unchanged)
- Homophonic + transposition, no boundaries (Z340 class) — genuine open frontier.
- Bifid/fractionation — `diagnoses_only`; diagnosis mis-calls mono (F3 anchoring
  trap). Honest-fail discipline probe.
- Italian simple substitution, Copiale p068 — the two `bad_result` rows, not
  addressed this cycle.

## Deliverables
- **Agentic frontier suite** (pasteable, one sub-agent per crack):
  `docs/evidence/agentic_frontier_suite.md`. 9 fresh cases spanning
  extended-frontier solves (1,2), standing solves (mono/vigenere/latin/
  homophonic), and open-frontier / honest-fail probes (3,4,bifid,homo+transp).
  Generator: `scripts/gen_agentic_frontier_suite.py`; sealed answers +
  pre-registered expectations at `~/.config/decipher/dogfood_answers/
  agentic_frontier_answers.json` (NOT in the pasteable file — Codex is unbiased).
- **Automated frontier JSONL update (follow-up):** add `known_good` rows for (1)
  and (2) and a `bad_result` row for (4), once per-family testgen `synthetic_spec`
  support is confirmed (composite sub+transposition may need a new testgen mode).
