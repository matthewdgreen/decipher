# INV/solver spec — composite substitution + transposition detection & routing

Status: DRAFT for implementation. Author: Fable (main loop), 2026-07-19.
Motivation and acceptance evidence: `docs/evidence/mcp_dogfood_results.md`
(Round 4 self-test + live honest-fail `14f780f21699`), `docs/evidence/
v3_vs_mcp_matrix.md` (round-4 composite cell, findings F1/F2), and the INV
model sweep's "composite universally missed" result (`project-state` memory).

---

## 0. The problem, precisely

A cipher that applies a **monoalphabetic substitution and then a transposition**
(or the reverse) defeats the entire stack — every arm, not just the LLMs. The
round-4 cipher (287 letters, substitution + columnar keyword MASONRY, no word
boundaries) is the canonical instance. The failure is NOT capability: the
isolation test proved that hand-peeling the transposition and running the plain
substitution solver on the correct-order stream recovers the exact plaintext
(100%, dict_rate 0.85). The failure is **detection and routing** — nothing in
the stack recognizes the composite or peels the order layer. Three concrete
mechanisms, all confirmed:

1. **Diagnosis suppresses transposition when substitution is present.** The
   `order_layout` panel emits at most one of two mutually-exclusive atoms:
   `letters_unsubstituted` (→ transposition) OR `letters_substituted`
   (→ substitution/homophonic, and explicitly `weakens=["transposition"]`).
   A composite is substituted *and* transposed, so it trips
   `letters_substituted`, which actively pushes transposition DOWN. The two
   layers are modeled as competing hypotheses, never composable. There is a
   `transposition_homophonic` family but **no substitution+transposition
   composite family**.

2. **The solver router hijacks to homophonic on no-boundary text.** In
   `_select_solver_path` (`src/automated/runner.py:2802`), the rule
   `if word_groups <= 1 and alphabet_size > 20: route=homophonic` and, earlier,
   `if alphabet_size > pt_alpha.size: route=homophonic`. A 26-symbol
   no-boundary composite with an A-Z alphabet actually falls THROUGH to the
   substitution route — but finding F1 shows the same class of boundary-driven
   misroute sends no-boundary **Vigenère** to the homophonic annealer, and any
   composite whose substitution inflates the effective inventory misroutes too.
   The `content_suspicious` transposition probe (`transposition_suspicion`)
   cannot help: it keys on the monogram distribution matching the language
   **by letter** (cosine similarity), which a substitution DESTROYS. So the one
   existing transposition-detection signal is blind to any *substituted*
   transposition by construction.

3. **`transform_search` routes away from transpositions on dense/no-boundary
   text.** Confirmed in the round-4 self-test: `transform_search: full` chose
   `route=homophonic`, `transform_pipeline=None`, and never tried a
   transposition (6.7% garbage).

The unifying insight the fix is built around: **a solve with HIGH letter
accuracy but NO word structure implies a residual order layer.** A plain
substitution that lands in a good frequency/quadgram basin but produces zero
dictionary words is the signature of an unpeeled transposition. This is the
detectable, family-agnostic signal the stack currently ignores.

---

## 1. Scope and non-goals

**In scope (this program):**
- A `substitution_transposition` composite family in the INV-0 registry, with a
  discriminator and the harmonic-style signposting.
- An `order_layout` panel change so `letters_substituted` no longer
  unconditionally suppresses transposition when a residual-order signal is
  present.
- A router fix so no-boundary / dense text routes on the DIAGNOSIS, not the
  boundary heuristic (closes F1 for periodic AND composite).
- A **peel-and-solve pipeline**: after a substitution solve that is
  high-letter-quality but word-empty, screen transpositions over the DECODED
  letter stream and re-score. Exposed as an automated route and an
  `experiment_submit` type so BOTH the v3 lead and the MCP surface reach it.

**Out of scope (future work, do not build here):**
- Word-alignment / GT-free key selection ([[word-alignment-future-work]]).
- transposition-then-substitution with a HOMOPHONIC substitution layer (Z340
  class) — the peel pipeline targets monoalphabetic substitution first; Z340
  remains the documented stretch case and its acceptance target is "honest
  fail with correct composite diagnosis," not a solve.
- Reworking `transposition_homophonic`.

---

## 2. Slice A — diagnosis: composite family + residual-order atom

### 2.1 New `order_layout` atom (`src/analysis/panels.py:356`)

Today `panel_order_layout` emits `letters_unsubstituted` XOR
`letters_substituted` from `(monogram_chi2, quadgram_nll)`. Add a THIRD,
composable observation computed from the same rendering:

- **`residual_order_after_substitution`**: fires when the monogram distribution
  is displaced from the language reference (substitution present: `mchi >
  MONO_CHI2_HIGH`) AND the quadgram score of the rendering is scrambled-poor
  (`qnll` above the scrambled threshold `QUAD_NLL_SCRAMBLED`, i.e. no language
  n-gram structure). Interpretation: "letters substituted, but n-gram structure
  absent — consistent with a further order (transposition) layer." It
  `supports=["substitution_transposition"]` and, crucially, does NOT weaken
  `transposition` (unlike `letters_substituted`).

Keep `letters_substituted` as-is EXCEPT: when `residual_order_after_substitution`
also fires, `letters_substituted` must not carry its `weakens=["transposition"]`
(the composite explanation makes the suppression wrong). Implement by computing
the residual flag first and passing `include_transposition_weaken=not residual`
into the `letters_substituted` atom construction.

Thresholds: reuse existing `MONO_CHI2_HIGH`, `QUAD_NLL_SCRAMBLED`,
`PEAK_FLAT_MIN`, `PEAK_UNIQUE_MAX`. Do NOT introduce new magic numbers without
recording them at panel top with the others. The residual atom weight: 0.35
(same as `letters_substituted`; it is one corroborating signal, not decisive).

### 2.2 New family (`src/investigation/families.py:193`)

Add a `primary` FamilySpec:

```
FamilySpec(
    "substitution_transposition", "Substitution + transposition", None, "primary",
    "agent_assists",
    ("fingerprint_prior:substitution_transposition",
     "residual_order_after_substitution", "peaked_monogram_shape"),
    ("disc_sub_transp_composite",),          # new discriminator, below
    ("monoalphabetic_substitution", "transposition"),   # confusable_with
    "Substitution then transposition: peaked/displaced frequencies (substitution) "
    "with absent n-gram structure (order layer). High letter accuracy but no "
    "word structure when solved as plain substitution.",
    38,                                       # display order: below the base primaries
    sequencing_hint=(
        "A plain-substitution solve with strong frequency/quadgram fit but zero "
        "dictionary words implies an unpeeled order layer — screen transpositions "
        "over the DECODED stream (peel-and-solve) before rejecting."
    ),
)
```

Add to `SUBSTITUTION_PRIMARIES` (it is a substitution-bearing family). Add the
`SUSPICION_TO_FAMILY` mapping key if `cipher_id` emits a matching suspicion
prior (2.4). Register a new `DiscriminatorSpec`:

```
DiscriminatorSpec(
    "disc_sub_transp_composite",
    "plain substitution vs substitution+transposition (letter accuracy high, "
    "word structure absent)",
    ("monoalphabetic_substitution", "substitution_transposition"),
    ("order_layout",),
    "observe_transform_suspicion",       # existing tool; extended in 2.3
    "available",
)
```

`_validate_registry()` must still pass (every primary lists ≥1 discriminator;
the composite lists `disc_sub_transp_composite`). Add the mono→composite
discriminator to `monoalphabetic_substitution.discriminators` too, so a strong
mono verdict surfaces the composite as the thing to rule out.

### 2.3 Extend `transposition_suspicion` (`src/analysis/transposition_solver.py:212`)

Add a second signal that survives substitution. The by-letter cosine stays (it
catches UNsubstituted transposition). Add a **substitution-invariant** check:
compute the quadgram-NLL of the raw stream; if the monogram distribution is
peaked/language-like in SHAPE (sorted magnitudes match a language profile,
regardless of which letters) but the quadgram structure is scrambled, flag
`order_layer_suspected=True` with a distinct reason. Return both fields:
`{"suspicious", "score", "order_layer_suspected", "reasons"}`. This is what the
new discriminator and the router (Slice B) consume. Keep the existing return
keys for back-compat; `min_tokens` guard unchanged.

### 2.4 Optional fingerprint prior (`src/analysis/cipher_id.py`)

If low-effort: emit a small `substitution_transposition` suspicion in
`_compute_suspicion_scores` when the peaked-monogram + scrambled-quadgram
combination holds on an A-Z-sized alphabet. If it complicates the existing
suspicion logic, SKIP it — the `residual_order_after_substitution` atom already
carries the family; the fingerprint prior is corroboration only. Record the
choice in the impl notes.

---

## 3. Slice B — router: route on diagnosis, not boundaries

### 3.1 `_select_solver_path` (`src/automated/runner.py:2802`)

The boundary heuristics (`word_groups <= 1 and alphabet_size > 20 → homophonic`)
are the F1/F2 hijack. Change the precedence so a **content signal** can override
the boundary default:

1. Explicit `cipher_system` name routing stays first (unchanged).
2. Before the `word_groups <= 1` homophonic default, consult
   `transposition_suspicion`: if `order_layer_suspected` is True on an
   A-Z-sized alphabet, route to a new `composite_substitution_transposition`
   route (Slice C solver) instead of homophonic. Guard on `alphabet_size <=
   pt_alpha.size` — a genuinely dense homophonic inventory (>26) still routes
   homophonic (the composite peel targets monoalphabetic substitution).
3. The `alphabet_size > pt_alpha.size → homophonic` rule stays for truly
   overcomplete alphabets.

Do NOT remove the homophonic default; narrow it. A no-boundary Vigenère
(finding F1) should route periodic via the existing name/periodic path when the
diagnosis periodic-IC signal is strong — add a periodic-IC consult symmetric to
the transposition one if it is cheap; otherwise leave F1's periodic half to the
diagnosis-driven agent path and note it. The MUST-fix here is the composite
route; the Vigenère half is a documented secondary.

### 3.2 Regression guard

A plain monoalphabetic substitution (peaked freqs, GOOD quadgram structure,
`order_layer_suspected=False`) must STILL route to `substitution`. A plain
homophonic (>26 symbols) must STILL route homophonic. Add explicit tests
(Slice D) — this router is load-bearing and every existing family must be
unchanged.

---

## 4. Slice C — peel-and-solve pipeline

The capability. After a substitution solve produces a high-letter-quality,
word-empty stream, screen transpositions over the DECODED stream.

### 4.1 Automated route (`src/automated/runner.py`)

New solver path `composite_substitution_transposition`:
1. Run the plain bijective substitution anneal (existing
   `native_substitution_*`) on the raw cipher → decoded letter stream S and its
   solver score.
2. Detect residual order: if S has strong quadgram/frequency fit but
   `dict_rate(S)` is near zero (the "high letter accuracy, no words" signature),
   proceed to peel; else return the substitution result as-is (it was not a
   composite).
3. Screen transpositions over S. **CORRECTION (2026-07-19, from review #2 +
   finding F2):** the round-4 acceptance cipher is KEYWORD-COLUMNAR (keyword
   MASONRY), and `screen_pure_transposition` is the GEOMETRIC screen (matrix-
   rotate/route/rail/mask/TransMatrix) — finding F2 proved it fails blind on
   classic keyed columnar (~0.10 char). So reusing it wholesale would NOT solve
   round-4. Instead:
   - The geometric screen is still consulted (cheap, catches route/rail
     composites).
   - The keyword-columnar layer is peeled by a NEW shared module
     `analysis/columnar_search.py` (per the coordination decision, §7a /
     polygraphic §1a): a keyword/column-permutation search with PLUGGABLE
     scoring. THIS program supplies the LANGUAGE-score plugin (S is A-Z
     letters after substitution, so language scoring is valid here). Building
     this module in Slice C.1 is REQUIRED for round-4 AND closes matrix
     finding F2 (keyed-columnar blind coverage) for the transposition side.
     The polygraphic program's PF-6 later adds a fractionation-stream-score
     plugin to the SAME module (coordinate data has no language structure).
4. Return the best (substitution-key, transposition-key) pair with the combined
   decode. Artifact records BOTH layers.

Ordering note: substitution-then-transposition-encryption means DEcryption is
transposition-peel THEN substitution-invert. But peeling transposition on
CIPHER letters is equivalent to peeling on DECODED letters (substitution is
position-independent), so solving substitution first on the raw stream is valid
and lets the frequency attack work normally. Document this in the code.

### 4.2 Experiment type (`src/investigation/experiments.py`)

Add `composite_substitution_transposition` to the experiment schema (mirror the
`quagmire3_shotgun` pattern landed at `9f4ed28`): host-derived `language`,
bounded budget knobs (substitution anneal restarts; transposition screen
breadth via the existing `PureTranspositionSearchConfig`), unknown-key
rejection with a family-consistent `corrected_example`, results installable via
`experiment_collect` so the branch-bound verify→declare gate is reachable.
Reuse the install machinery; the branch carries both the substitution key and
the transform pipeline in metadata (the grader already handles metadata
`decoded_text`, per `81f0afb`). Update the misroute guard: an
`automated_solver` submit whose `cipher_system` names a sub+transposition
composite gets redirected here, exactly as `quag` redirects to
`quagmire3_shotgun`.

### 4.3 Onboarding + TOOLS.md

Add the new experiment type to `docs/mcp_onboarding.md` §2 (the experiment-types
paragraph, next to `quagmire3_shotgun`) and to TOOLS.md. Keep the playbook line
generic: "strong letter fit but no words → try the composite peel."

---

## 5. Firewall (ENFORCED)

Ground truth must never enter detection, routing, selection, or install. The
peel pipeline ranks transposition finalists by language score, NOT by benchmark
plaintext. `language` stays host-derived in the experiment (GT-3). The dict_rate
residual check uses the shipped dictionaries, not GT. Any new `ground_truth`
parameter is a review red flag. This is a hard gate per AGENTS.md.

---

## 6. Slice D — tests (required)

- **Panel**: `residual_order_after_substitution` fires on a synthetic
  sub+transposition, does NOT fire on plain substitution (good quadgrams) or
  plain transposition (unsubstituted); `letters_substituted` drops its
  transposition-weaken exactly when the residual atom co-fires.
- **Diagnosis**: the round-4 cipher (build it in-test from
  `src/ciphers/transposition.py` columnar + a substitution; fixed seed; the
  sealed answer at `~/.config/decipher/dogfood_answers/round4_composite_answer.json`
  is for the ACCEPTANCE assertion on OUTPUT only, ciphertext-as-input) yields
  `substitution_transposition` in the ranked top-2, not a CONFIDENT plain-mono
  verdict. This is the anti-anchoring assertion — the exact failure the INV
  sweep documented.
- **Router**: composite → `composite_substitution_transposition`; plain mono →
  `substitution` (regression); plain homophonic → `homophonic` (regression);
  no-boundary Vigenère → periodic or at least NOT homophonic (F1 guard).
- **Peel pipeline**: end-to-end on the round-4 cipher → exact plaintext (the
  isolation test already proved 100% is reachable; this makes it automatic).
  Assert both layers recorded.
- **Experiment**: submit `composite_substitution_transposition`, collect,
  install → branch carries both layers; misroute guard redirects a composite
  `cipher_system` hint; unknown keys rejected.
- **Firewall**: no GT reachable from the new route (extend the existing
  `test_ground_truth_firewall.py` pattern).
- Full suite green.

---

## 7. Acceptance

- **$0**: round-4 cipher solved 100% end-to-end through the automated composite
  route AND through `experiment_submit`→`experiment_collect`→verify→declare in
  a scripted MCP-host test.
- **Live (both arms, matrix)**: re-run the round-4 composite cell. v3 lead and a
  fresh Codex-MCP session should now solve in-surface (diagnosis names the
  composite → experiment peels it → verified declare). Grade via
  `scripts/grade_dual_harness_run.py`; this closes the round-4 matrix cell the
  way F4/round-6 closed.
- **Stretch (documented, not required)**: Z340 diagnosis should name a composite
  (transposition + homophonic) rather than CONFIDENT homophonic; a solve is NOT
  expected (homophonic substitution layer is out of scope).

---

## 7a. Cross-program coordination (RESOLVED 2026-07-19)

This program shares surfaces with two same-week specs; binding decisions live
in `polygraphic_fractionation_solver_spec.md` §1a. Reciprocal summary:
- **Land order: this program lands FIRST.** It owns the `order_layout` panel
  change (`residual_order_after_substitution`) and the
  `substitution_transposition` family + `disc_sub_transp_composite`. The
  polygraphic program lands after and edits `families.py`/`panels.py`
  sequentially (no conflict).
- **solver_status**: use the live enum; this family is `agent_assists`.
- **pure_transposition reuse is legitimate HERE** (the peel yields A-Z letters
  the screen ranks by language model) and is NOT shared with the polygraphic
  ADFGX/ADFGVX peel (which yields non-language coordinate data). Distinct.
- **Experiment type** `composite_substitution_transposition` follows the
  `quagmire3_shotgun` contract. Mechanism correction (review #2 N1): a new
  `EXPERIMENT_TYPES` value is NOT an operation-manifest entry —
  `experiment_submit` is a SINGLE operation whose `type` arg is a string. The
  new type reaches both the MCP surface and `decipher investigation`
  automatically because both skins dispatch through the shared service layer
  that reads `EXPERIMENT_TYPES` (post CLI I-0); parity coverage is of the
  `experiment_submit` operation, not of each type. Misroute-guard convention
  is the shared one (distinct type names, same pattern).

## 8. Slice order & orchestration

Per CLAUDE.md: Fable specs (this doc + per-slice impl specs as needed), Opus/
Sonnet coders implement, Fable reviews the diff, one commit per reviewed slice,
served-model grep each Fable agent. Coupling order:

1. Slice A (diagnosis) — self-contained; land first.
2. Slice C.1 (automated peel route) — depends on nothing in A but shares the
   residual-signal helper; extract that helper so A and C share it.
3. Slice B (router) — routes INTO C's solver; land after C.1 exists.
4. Slice C.2 (experiment type) — mirrors quagmire3_shotgun; land after C.1.
5. Slice D tests land WITH each slice, not at the end.

Each slice keeps the full suite green. The round-4 cipher is the through-line
acceptance fixture for every slice.
