# Spec: Language-Model Variant Registry

Parent: user decision-support for the DTA-default question. Spec author:
Fable. Implementer: coding sub-agent. Goal: remove the hard-coded
one-model-per-language rule and make variants ("modern X", "archaic X")
first-class — labeled, listable, and selectable by CLI, benchmark
context, the agent, and the preflight — WITHOUT changing any current
default resolution in this slice (the de-default switch stays a separate
one-line decision).

Baseline: main at `e699773` + uncommitted docs. Suite 1086 passed /
1 skipped (record first; zero new failures). No commits. Stale-line
rule: locate constructs by quoted code.

## Part 1 — Sidecar variant fields + registry

1. Sidecar metadata (models/*.bin.metadata.json) gains OPTIONAL fields
   `variant` (machine slug) and `display_label` (human). Update the
   existing sidecars in-place (data-only):
   - ngram5_de_dta: `historical_1600_1899`, "German (1600–1899, DTA
     Kernkorpus)"
   - ngram5_de: `literary_19c`, "German (19th-c literary, Gutenberg)"
   - ngram5_de_500: `literary_19c_small`, "German (Gutenberg, 500-book)"
   - en/fr/it/la + their _500s: `gutenberg` / `gutenberg_small` with
     matching labels; en_mixed `mixed`; en_parity `parity`.
   Also teach `tools/corpus/build_model.py`'s metadata writer an
   optional `--variant`/`--display-label` passthrough.
2. `src/analysis/model_registry.py` (new):
   `list_language_models(language, models_dir=None) ->
   list[ModelInfo{path, language, variant, display_label,
   distinct_ngrams, chars, sha256}]` — scans sidecars (repo-root
   anchored, expanduser; missing/invalid sidecars yield a ModelInfo with
   `variant=None` and a filename-derived label, never a crash);
   `resolve_language_model(language, variant=None, models_dir=None) ->
   path` with precedence:
   (a) `DECIPHER_NGRAM_MODEL_<LANG>` env — unchanged, always wins
   (back-compat pin);
   (b) explicit `variant` arg → registry match (exact slug; error
   listing available variants if absent);
   (c) default: `models/ngram5_<lang>.bin` exactly as today (pin test:
   with no env and no variant, resolution is byte-identical to the
   current `_zenith_native_model_path` for every language).

## Part 2 — Plumbing (no behavior change by default)

1. `_zenith_native_model_path` delegates to the registry (keeping its
   public alias) and gains an optional `variant=None` param threaded
   from: a new `model_variant` parameter on `run_automated` /
   `AutomatedBenchmarkRunner`, a `--model-variant` CLI flag (crack +
   benchmark, default None), and `run_automated_multipage`.
2. Benchmark-context mapping: a small table
   `SOURCE_MODEL_VARIANTS = {"copiale": ("de", "historical_1600_1899")}`
   consulted ONLY when the new `--model-variant auto` value is passed
   (explicit opt-in; default None keeps today's behavior). Record the
   chosen variant + path + sha in artifacts wherever `binary_ngram_model`
   is already recorded.
3. `analysis/multipage.py` + `word_hypothesis_repair` already take
   explicit `model_path` — no change beyond the runner passing the
   variant-resolved path.

## Part 3 — Agent + preflight surface

1. New tool `observe_language_models` (no args beyond optional
   `language`): returns the registry list (label, variant, distinct
   n-grams, chars — NO paths/shas in model-visible output beyond what
   branch cards already show) plus which model is currently active and
   why (env/variant/default).
2. New tool `act_set_model_variant(variant)`: sets an executor-level
   selection consumed by every search tool that resolves the binary
   model (`search_automated_solver`, `search_homophonic_anneal`,
   transform/null-mask paths — thread through the existing model-path
   resolution choke points; grep for `_zenith_native_model_path` /
   `zenith_native_model_path` call sites). Validates against the
   registry; records in artifacts. Cleared only by calling it again.
3. Preflight: the diagnostic preflight/cipher_id_report section that
   introduces the run gains one line listing available variants for the
   run language and the active selection, so both v2 and v3 agents can
   see the choice exists. (v3: the context builder's fingerprint
   section — additive.)
4. TOOLS.md entries + tool count update (92 → 94) and the CLAUDE.md
   Key Files count line; the Phase 0 dispatch/consistency tests cover
   the handlers automatically.

## Part 4 — Tests

- Registry: scan/labels/missing-sidecar tolerance; resolution
  precedence (env > variant > default) with the back-compat pin.
- Runner: `model_variant` threading (artifact records the resolved
  variant); `--model-variant auto` maps copiale→historical; default
  None byte-identical (pin against a no-variant run's artifact fields).
- Tools: flow test (observe → set → search resolves the variant path —
  stub the solver); invalid variant → structured error naming
  available slugs; no-leak (paths/shas absent from tool JSON as
  specified).
- Firewall: registry/selection surfaces leak-checked (reuse helper).

## Out of scope

Changing the de default (separate decision); building a true modern-
German model; v3 episode-kind toolset additions (the two new tools are
v2-surface; v3 leads inherit them via the shared executor — note this);
dictionary variants (get_dictionary_path stays language-keyed — record
as a follow-up note).

## Deliverables

Files changed, suite counts, the resolution-precedence demonstration
(env/variant/default on de), deviations. No commits.
