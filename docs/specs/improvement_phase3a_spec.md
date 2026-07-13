# Spec: Improvement Program Phase 3a — Binary Model Format v2 + Period-German Model

Parent plan: `docs/improvement_program_plan.md` (Phase 3, items 3.1–3.2).
Spec author: Fable. Implementer: coding sub-agent. Parallel track to
Phase 2 (different files). Items 3.3 (space-aware scoring) and 3.4 (SA
proposal mix) are LATER slices — out of scope here.

## Part A — `zenith_binary_v2` format (item 3.1)

### Current constraints (verified)

`src/analysis/zenith_solver.py`: loader raises unless order == 5
(`:129–130`); index math assumes exactly 26 lowercase letters `a-z`
(`:65–82`); `lru_cache(maxsize=2)` on the loader (`:118`) thrashes across
the 12-model inventory; a single hardcoded `unknown_log_prob` floor
(`:132–133`).

### Required

1. A `zenith_binary_v2` container: header carrying magic/version, the
   alphabet string (N symbols, UTF-8), and order K; payload = N^K float32
   log-probs in the same big-endian layout. v1 files (headerless 26^5)
   keep loading exactly as today — sniff the magic to distinguish.
2. Generalize the Python loader + lookup index math to (alphabet, order)
   from the header; v1 path keeps its current hardcoded behavior
   byte-for-byte. Raise `lru_cache` to `maxsize=16`. Read
   `unknown_log_prob` from the sidecar metadata JSON when present
   (existing `model_metadata` pattern), falling back to the current
   constant.
3. Mirror v2 reading in the Rust engine (`rust/decipher_fast`), with a
   shared-semantics test between Python and Rust loaders on a tiny
   fixture model (build a 3-symbol order-2 fixture in-test).
4. The model-BUILDING tooling: locate the existing builder that produced
   `models/ngram5_*.bin` + sidecars (search `scripts/` for the generator;
   the `*_500` variants and metadata sidecars imply one exists — report
   if it lives outside the repo). Extend it to emit v2 with an
   `--alphabet` option. Do not regenerate any existing v1 model.

### Tests

Round-trip build→load for a tiny v2 fixture (Python and Rust); v1
regression (existing `tests/test_zenith_solver.py` untouched and green);
lookup-index property test for a non-26 alphabet; metadata
`unknown_log_prob` override.

## Part B — DTA period-German model (item 3.2)

Goal: a stronger 17th–19th-century German model than the current
100-book Gutenberg `de` model (23.2M chars, 475,932 distinct 5-grams vs
1.4M for `en`).

1. **Corpus**: Deutsches Textarchiv (DTA) core corpus — openly licensed
   (CC BY-SA). Add `scripts/build_dta_corpus.py` that downloads the
   published plain-text/TCF export, records source URLs, license, and
   SHA256s into a corpus manifest JSON, and normalizes text with the SAME
   normalization policy as the existing models (lowercase, strip
   non-alpha, fold umlauts/ß: ä→a ö→o ü→u ß→s — match whatever the
   existing `de` sidecar's `normalization` field documents; verify and
   follow it exactly so models are comparable). Long-s (ſ) → s and other
   historical-orthography folds must be handled and documented in the
   manifest. Corpus files land under a gitignored `corpora/dta/` dir.
2. **Model**: build `models/ngram5_de_dta.bin` (v1 format is fine for
   this model — 26 letters, order 5 — so it works in Rust immediately)
   plus the metadata sidecar (source corpus, char count, distinct
   5-grams, normalization, license, checksum). Do NOT replace
   `ngram5_de.bin`; selection stays explicit.
3. **Selection hook**: the runner resolves the model per language today
   (`DECIPHER_NGRAM_MODEL_DE` env / `models/ngram5_<lang>.bin` default —
   Phase 2a generalized this). No code change needed beyond confirming
   `DECIPHER_NGRAM_MODEL_DE=models/ngram5_de_dta.bin` works end to end.
4. **A/B packet** (the plan's Milestone-2 requirement): a small runnable
   comparison — `scripts/research/copiale/run_german_model_ab.py` or
   extend `scripts/audit_german_scoring.py` — that scores the Copiale
   evidence-packet pages' known plaintext + degraded controls under both
   models and reports separation quality; plus one solver-backed run of
   the five-page packet with the DTA model selected, artifacts under
   `artifacts/german_model_ab/`. Report the numbers; do NOT change any
   default based on them (that decision is the orchestrator's).

### Constraints

- Network downloads: DTA only, URLs recorded, resumable/idempotent
  (skip if checksums match). If the DTA download shape has changed and
  no stable plain-text export is found, STOP and report options rather
  than scraping ad hoc.
- Ground-truth firewall: the A/B calibration report may read benchmark
  plaintext post-hoc (it is a calibration artifact); the built model
  obviously must not embed benchmark plaintext — exclude any Copiale/
  benchmark-derived text from the corpus, and say so in the manifest.
- Training a 26^5 model from a multi-hundred-MB corpus must stream
  (no full-corpus memory load); note expected runtime in the report.

## Acceptance

- Full suite green (baseline: record before starting; ~850 passed /
  1 skipped expected).
- v2 fixture round-trips in Python and Rust; all existing zenith tests
  untouched and green.
- `ngram5_de_dta.bin` + sidecar + corpus manifest exist; A/B report
  produced with numbers for both models.
- Report: files changed, suite counts, corpus size/5-gram stats vs the
  old model, A/B summary table, builder-tooling findings, deviations.
