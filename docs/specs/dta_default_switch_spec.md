# Spec: DTA German Model as the Default `de` Model

Parent: user-approved decision (2026-07-14). Spec author: Fable.
Implementer: coding sub-agent. Depends on: the variant registry
(`dcf7200`). Small slice — the switch itself plus pins plus a re-baseline.

## Decision (evidence, for the record)

DTA (`ngram5_de_dta.bin`, variant `historical_1600_1899`) beats the old
Gutenberg model (`ngram5_de.bin`, `literary_19c`) on every measured German
workload: Copiale packet +7.4 mean char (all 5 pages, largest on the
weakest); modern-German synthetics +9.0 char / +15.4 word; stacks with
multipage to 79.9%. Content-probe clean (zero verbatim ground-truth
windows in the corpus). Matthew approved making it the `de` default.

## Change

`src/analysis/model_registry.py` `resolve_language_model`: introduce a
`_DEFAULT_VARIANTS: dict[str, str]` table, `{"de": "historical_1600_1899"}`
(all other languages absent → today's behavior). In the default branch
(c), BEFORE the `ngram5_<lang>.bin` filename fallback: if
`_DEFAULT_VARIANTS.get(lang_key)` is set, resolve that variant via the
same registry scan used by branch (b); on a miss (model absent), fall
through to the filename fallback (never raise for a default). Precedence
is UNCHANGED and load-bearing:

1. `DECIPHER_NGRAM_MODEL_<LANG>` env — still wins (a de user pinning the
   old model via env is unaffected).
2. explicit `variant=` arg — still wins over the default (an agent or
   `--model-variant literary_19c` can select the old model).
3. default → now the `_DEFAULT_VARIANTS` variant when present, else the
   filename fallback.

Net effect: with no env and no explicit variant, `de` resolves to
`ngram5_de_dta.bin`; every other language is byte-identical to today.

## Tests (this is where the care goes)

- **Flip pin**: `resolve_language_model("de")` (no env, no variant) →
  `ngram5_de_dta.bin`; every other language still →
  `ngram5_<lang>.bin`. This REPLACES the prior "de default =
  ngram5_de.bin" assertion in `tests/test_model_registry.py` — update it,
  don't leave a contradicting pin.
- **Precedence preserved**: `DECIPHER_NGRAM_MODEL_DE=<old>` still wins
  over the new default; `variant="literary_19c"` still selects the old
  model; `variant="historical_1600_1899"` and the default now resolve to
  the same path.
- **Runner parity**: `_zenith_native_model_path("de")` (delegating to the
  registry) returns the DTA path; the `active_selection("de")` source is
  reported as `default` (or a new `default_variant` source — pick one and
  make the preflight line/tool honest about WHY de resolves to DTA, so an
  agent isn't confused that "default" points at a non-obvious file).
- **Missing-model fallback**: if `ngram5_de_dta.bin` were absent, default
  resolution falls through to `ngram5_de.bin` without raising (temporarily
  rename in-test or monkeypatch the scan).

## Re-baseline (compute, record — this becomes the new reference)

Run locally (no LLM), record under `artifacts/baseline_dta_default/`:
- Five-page Copiale evidence packet, screen budget, `null_masks`
  refinement, DEFAULT model resolution (no env var now needed) — confirm
  it matches the DTA A/B numbers (75.3/77.7/78.2/74.8/80.5).
- Note in the report that `artifacts/baseline_20260713/copiale_null_masks/`
  (69.9 mean) is now the OLD-model historical reference, superseded.
- Update any doc/table that cited 69.9 as "the" Copiale baseline to say
  "old `literary_19c` model" and add the 77.3 DTA-default figure.

## Out of scope

Dictionary variants (still language-keyed); modern-German model build;
touching the env/explicit-variant precedence.

## Deliverables

Files changed, suite counts (baseline 1123/1), the precedence
demonstration (env/variant/default on de all shown), the re-baseline
table, deviations. No commits.
