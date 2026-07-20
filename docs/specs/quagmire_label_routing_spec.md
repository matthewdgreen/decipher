# Label-aware blind-Quagmire routing in the automated runner

**Status:** spec (2026-07-20). Follow-up to the frontier-suite validation
(`3eb6368`, row `synth_en_540q3nb_s42` recorded `bad_result`) and the same
principle as the P3 fractionation guard in `39b889a`: an explicit family
label must not silently degrade to an incapable engine.

## 0. Evidence

- Frontier row `synth_en_540q3nb_s42` (blind Quagmire III, tableau keyword
  length 7, cycleword period 8, labeled `cipher_system="quagmire3"`):
  **char 0.29** through the automated route. Root cause read directly from
  `src/automated/runner.py`: `_run_periodic_polyalphabetic` gates its blind
  quag search branch (line ~3295) on
  `DECIPHER_KEYED_VIGENERE_MODE ∈ {quagmire_search, quag3_search,
  quagmire3_search}`, but the env default is `"replay"` (line ~3198), so a
  labeled quag cipher with no known parameters falls through to the generic
  periodic screen — which cannot solve a keyed tableau.
- The experiment-surface misroute guard (`src/investigation/experiments.py`
  ~line 605) rejects quag hints for `automated_solver` precisely because of
  this fallthrough ("it would silently misroute to the generic periodic
  screen"). This spec fixes the runner; the guard's wording is revisited
  LATER (out of scope here).
- Capability + budget measured 2026-07-20 on the regenerated s42-class
  cipher: `search_quagmire3_shotgun_fast` with the experiment defaults
  (keyword_lengths [7], cycleword_lengths [8], hillclimbs 5000,
  restarts 250, threads 0) solves it **exactly (char 1.0) in 21.3s**
  (consistent with round-6's 22.4s).

## 1. Required behavior — `src/automated/runner.py`, `_run_periodic_polyalphabetic`

1. Read the RAW env: `env_mode_raw = os.environ.get("DECIPHER_KEYED_VIGENERE_MODE")`
   (`None` when unset). Any EXPLICITLY SET value keeps today's semantics
   byte-for-byte — including an explicit `"replay"` (which suppresses blind
   search exactly as now) and the search modes with their CURRENT defaults
   (hillclimbs 500, restarts 8, cycleword_lengths 1..max_period). Zero
   behavior change for anyone who sets the env.
2. **Label-aware default (the new rule):** when the env is UNSET, the
   (lowercased) `cipher_system` contains `"quag"`, and the known-params
   replay block above did not apply, enter the blind quag SEARCH branch.
3. In this label-aware entry ONLY, the search uses **experiment-parity
   defaults** — the same numbers as
   `EXPERIMENT_TYPES["quagmire3_shotgun"]`: `keyword_lengths` default `[7]`,
   `cycleword_lengths` default `[8]`, `hillclimbs` default `5000`,
   `restarts` default `250`. Every one remains overridable by the existing
   `DECIPHER_QUAGMIRE_*` env knobs (an explicitly-set knob wins over the
   parity default), and `DECIPHER_QUAGMIRE_ENGINE` still selects
   rust_shotgun (default) vs the Python fallback, unchanged.
4. The step dict records how the search was entered:
   `"routing": "label_aware_default"` vs `"routing": "env_mode"` (add the
   key to both entries of the existing step construction; no other step
   shape changes).
5. No firewall changes; nothing reads ground truth.

Implementation note: the cleanest cut is computing
`effective_mode = env_mode_raw if env_mode_raw is not None else
("quagmire3_search" if <label rule> else "replay")` plus a
`label_aware = env_mode_raw is None and <label rule>` flag that selects the
parity defaults; do not duplicate the search-call block.

## 2. Frontier row flip (same commit)

`frontier/automated_solver_frontier.jsonl`, row `synth_en_540q3nb_s42`:
`frontier_class` → `"known_good"`,
`min_char_accuracy_by_solver` → `{"decipher-automated": 0.99}`,
`max_elapsed_seconds_by_solver` → `{"decipher-automated": 240.0}`, and the
notes rewritten to: the 2026-07-20 bad_result (char 0.29, env-gated
non-label-aware routing) was fixed by this spec (label-aware default at
experiment-parity budget, measured 1.0 in ~21s); tableau keyword remains
pinned to length 7 = the default sweep; an unswept keyword length remains
out of budget by design. Post-land validation (orchestrator, not coder
scope): `run_frontier_suite.py --test-id synth_en_540q3nb_s42` gates green.

Also update the addendum row for s42 in `docs/frontier_solver_comparison.md`
(the table cell and its trailing sentence) to record the fix and measured
result.

## 3. Tests

In the module that already tests automated-runner routing (find the existing
home for `_run_periodic_polyalphabetic`/routing tests; follow its
conventions). All engine calls are monkeypatched stubs — no real shotgun
compute in tests:

- **Label-aware trigger:** env unset (`monkeypatch.delenv`), labeled
  `quagmire3`, no known params → the shotgun stub is called with
  keyword_lengths [7], cycleword_lengths [8], hillclimbs 5000, restarts 250;
  the step records `routing == "label_aware_default"`; the returned
  plaintext is the stub's.
- **Explicit replay respected:** `DECIPHER_KEYED_VIGENERE_MODE=replay` set
  explicitly + the same label → the search branch is NOT entered (generic
  periodic screen stub reached instead).
- **Env-mode defaults unchanged:** `DECIPHER_KEYED_VIGENERE_MODE=
  quagmire3_search` → stub sees hillclimbs 500, restarts 8, and the
  1..max_period cycleword default; step records `routing == "env_mode"`
  (pins zero drift on the explicit path).
- **Non-quag label untouched:** env unset, `cipher_system="vigenere"` →
  generic screen stub reached, shotgun stub never called.
- **Knob override in label-aware mode:** `DECIPHER_QUAGMIRE_HILLCLIMBS=123`
  → stub sees hillclimbs 123 with the other parity defaults intact.

Baseline: suite is **1879 passed / 2 skipped** at `3eb6368`; landing bar is
that plus these tests, zero failures.

## 4. Review adjudication (2026-07-20, Fable review: LAND)

- **Finding 1 (accepted as intended):** with the env unset, a quag-labeled
  run on a machine without the Rust kernel now fails LOUDLY at import
  instead of silently completing at char ~0.29 through the generic screen.
  This matches the round-6 fail-loudly doctrine (`decipher doctor` treats
  the kernel as required); no change.
- **Finding 2 (taken):** added a Python-fallback-engine stub test pinning
  the label-aware parity defaults on that branch (restarts 250,
  cycleword [8]; hillclimbs is rust-only).
- **Finding 3 (taken):** the knob-override test now pins all four
  `DECIPHER_QUAGMIRE_*` overrides, not just hillclimbs.
- **Finding 4 (no-op):** single shared step construction with a conditional
  routing value satisfies the "both entries" wording.
- Ladder-test edit adjudicated CORRECT and minimal (the base version pinned
  the misroute this spec fixes; the edit stays accuracy-agnostic per that
  test's convention).

## 5. Out of scope

- The experiment-surface misroute guard wording (`experiments.py` ~605) —
  revisit in a later slice once this default has soaked.
- Deriving cycleword lengths from periodic-IC/Kasiski evidence (a future
  refinement; the pinned [8] default mirrors the experiment type).
- Transform screens, diagnosis panels, TOOLS.md (no tool surface changes).
