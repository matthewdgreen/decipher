# Transform Generator Optimization — Testing Framework Plan

## Background

Transform search (`src/analysis/transform_search.py`) produces a large
candidate list per cipher: each candidate is a (transform pipeline,
post-transform decoding) pair, scored by an internal n-gram/IC ranker. The
top page of these candidates is exposed to the agent loop, which picks one or
more to continue solving from.

Two operational concerns motivate this plan:

1. **The current scorer leaves "word soup"** in the top page. We do not have
   a measured view of how often a rescuable candidate is hidden below
   non-rescuable ones, or how often the agent picks a wrong-but-high-scoring
   candidate and burns iterations on it.
2. **We optimize against ~3 stress cases.** Conclusions drawn from this small
   set are not statistically meaningful. We need hundreds of cases, and we
   need to be able to evaluate triage/scoring changes without running the
   agentic loop end-to-end.

This plan describes a **frozen-set / replayable-strategy** evaluation
framework: run the expensive transform search once per case, cache the full
candidate list, and replay alternative ranking and triage strategies over the
cache cheaply.

---

## Goals

- Quantify how often the current top-K contains a rescuable candidate, by
  cipher family / language / length.
- Make scorer / triage changes evaluable in seconds against a frozen
  population of hundreds of cases.
- Enable fair comparison of LLM-based ranking (Haiku, perplexity from a small
  local model, etc.) against non-LLM strategies, with cost accounted for.
- Keep historical cases (Borg, Copiale, Zodiac) as held-out validation,
  reported separately from synthetic results.

## Non-goals

- Not a replacement for the existing frontier or parity benchmark suites.
  Those evaluate end-to-end solver performance; this evaluates **candidate
  triage in isolation**.
- Not a tool for tuning the transform-search-internal scoring during search.
  This evaluates the *output* candidate list. Improvements to in-search
  scoring are a downstream consequence of what this framework reveals.
- Not a system for online (live-run) candidate filtering. That comes after
  this framework has identified a strategy worth deploying.

---

## Architecture

```
┌─────────────────────────────┐
│ 1. Population generator     │  one-time per population
│    (synthetic + held-out)   │  — versioned config
└──────────────┬──────────────┘
               │ cases.jsonl
               ▼
┌─────────────────────────────┐
│ 2. Candidate capture        │  expensive (CPU only, no LLM)
│    transform_search dump    │  — run once per case, cached
└──────────────┬──────────────┘
               │ artifacts/transform_triage/<case_id>.jsonl
               ▼
┌─────────────────────────────┐
│ 3. Ground-truth labeler     │  expensive (per candidate)
│    transform-correct,       │  — run once per case, cached
│    rescuable, readable-now  │
└──────────────┬──────────────┘
               │ adds labels to cached candidate JSONL
               ▼
┌─────────────────────────────┐
│ 4. Strategy harness         │  cheap (seconds per strategy)
│    pluggable rankers        │  — iterate freely
└──────────────┬──────────────┘
               │ ranked lists
               ▼
┌─────────────────────────────┐
│ 5. Metrics + reporting      │  cheap
│    recall@K, MRR, regression│
└─────────────────────────────┘
```

The architectural commitment: **strategy iteration is decoupled from data
generation.** Once a population is captured and labeled, every new strategy
is a few seconds of replay over fixed inputs.

---

## Component 1 — Population generator

### Synthetic sweep

Extend `src/testgen/builder.py` with a `transform_compose` mode that produces
ciphertexts from a parameterized transform pipeline applied to a known
plaintext. Sweep dimensions:

| Dimension | Values |
|-----------|--------|
| Language | `en`, `la`, `de` |
| Transform family | `pure_transposition`, `t_homophonic`, `t_substitution`, `t_periodic` |
| Grid columns | `[5, 7, 9, 11]` (where applicable) |
| Plaintext length (chars) | `[200, 500, 1000]` |
| Homophone factor | `[1, 2, 3]` (homophonic only) |
| Periodic key length | `[3, 5, 7]` (periodic only) |
| Seed | 10 seeds per cell |

Order of magnitude: ~3 langs × 4 families × ~3 sizes × ~3 sub-params × 10 seeds
≈ ~1000 cases. Trim to the cells that actually exercise transform search
(skip combinations that are not meaningful).

Each generated case carries metadata sufficient to label ground truth later:
the exact transform pipeline applied, the plaintext, the homophone map (if
any), and a per-case stable `case_id`.

Population manifest stored as
`frontier/transform_triage_population.jsonl` — versioned in git so the
population itself is auditable.

### Held-out historical anchors

A small file `frontier/transform_triage_holdout.jsonl` referencing existing
benchmark records:
- Borg samples (3-5 representative pages)
- Copiale samples (3-5 representative pages)
- Zodiac 408 / 340
- Any active frontier transform cases

These do not have ground-truth transform pipelines in all cases, so labeling
falls back to "rescuable via downstream solver" (see Component 3).

**Held-out cases are never used for tuning.** Report metrics on held-out
separately; if a strategy wins on synthetic but underperforms on held-out,
the strategy is overfit to the generator.

---

## Component 2 — Candidate capture

### Required code change

`src/analysis/transform_search.py` currently surfaces a top page. Add an
opt-in flag (e.g., `dump_full_candidates: int = 0`) that, when set, returns
the top-N (default N=200) candidates as a structured list rather than only
the page-sized output. This is the only invasive code change to existing
search code.

### Capture script

`scripts/capture_transform_candidates.py`:

For each case in the population manifest:
1. Load ciphertext and metadata.
2. Run transform search with `dump_full_candidates=200`.
3. For each candidate, persist:
   - `candidate_id` (stable, e.g., `case_id::rank`)
   - `transform_pipeline` (full pipeline metadata)
   - `decoded_text` (post-transform, post-substitution string)
   - Internal scores: `ngram_score`, `ic`, `dict_rate`, anything else the
     ranker computes
   - `original_rank` (where this candidate sat in the top-N pre-replay)
4. Write `artifacts/transform_triage/<case_id>.jsonl`.

This is the expensive step. Budget: probably a few hours of CPU for the full
population on a multi-core machine. Cache invalidation is by population
manifest version; if a case is unchanged, the cached candidate file is
reused.

Parallelize with the existing process-pool pattern used in the automated
parity matrix.

---

## Component 3 — Ground-truth labeler

### Three label tiers

Per candidate:

1. **`transform_correct`** (boolean). The candidate's transform pipeline
   matches the generator's ground-truth pipeline. Cheap; compute from
   metadata. For held-out cases without known pipelines, this label is
   `null`.

2. **`readable_now`** (float in [0, 1]). Character-level alignment score of
   the candidate's decoded text against the ground-truth plaintext. Use
   the same edit-aware alignment already in `src/benchmark/scorer.py`.
   Cheap proxy; computable for any case where plaintext is known.

3. **`rescuable`** (boolean). Starting from this candidate's decoding as the
   initial branch, run a fixed-budget downstream solver (homophonic anneal,
   simple-substitution anneal, or hill-climb depending on family) and check
   whether the result reaches a threshold (e.g., ≥ 0.85 character accuracy).
   This is the operational target metric — it answers *"would the agent
   succeed if it picked this one?"* Expensive: requires running a downstream
   solver per labeled candidate.

### Sampling strategy for `rescuable`

Computing `rescuable` for all 200 candidates per case is too expensive. Per
case:
- Always label the top 50 candidates by `original_rank`.
- Always label the top 25 candidates by `readable_now`.
- Always label any candidate with `transform_correct=true`.
- Label a random sample of 25 from the remaining tail (so strategies that
  reach into the tail can be evaluated, even with noise).

Document the sampling strategy in the artifact metadata so per-strategy
metrics know which candidates have a real label vs. a missing one.

### Implementation

`scripts/label_transform_candidates.py`:
- Reads cached candidate JSONL.
- For `transform_correct` and `readable_now`: pure function over metadata.
- For `rescuable`: invokes the appropriate solver from
  `src/automated/runner.py` with a fixed seed budget and time cap.
  Annotate the cached file in place with label fields.

Cache: once a candidate has been labeled, the label is persistent. Re-running
the labeler skips already-labeled candidates unless an explicit
`--relabel` flag is passed.

Budget: this is the second expensive step. Probably a CPU-day on a multi-core
machine for the full population. Parallelize aggressively. Held-out cases
should be labeled separately and stored in a different cache directory so
they cannot be accidentally tuned against.

---

## Component 4 — Strategy harness

### Strategy interface

A `Strategy` is a pure function:

```
def rank(candidates: list[CachedCandidate],
         case_metadata: dict) -> list[str]:
    """Return candidate_ids in ranked order (best first)."""
```

Implementations live in `src/triage/strategies/`. The harness loads them by
name from a registry.

### Strategies to implement initially

Non-LLM:
- `baseline_ngram` — current behavior; replay using `original_rank`.
- `dict_rate_weighted` — `ngram_score + α · dict_rate` with α tuned on a
  held-in subset.
- `consecutive_dict_words` — bonus for runs of k+ consecutive dictionary
  words; penalizes scattered hits.
- `binary_ngram_perplexity` — score using the loaded `zenith_native` 5-gram
  binary model. Microseconds per candidate, no API cost.
- `cluster_dedupe_then_rank` — cluster by edit distance on decoded_text,
  keep one representative per cluster, then rank within representatives.
- `mmr_diversity` — Maximal Marginal Relevance: rank by `λ · score − (1−λ) ·
  max_similarity_to_already_ranked`.

LLM-based (offline-cached):
- `haiku_score` — call Haiku (or another lightweight model) on each
  candidate's decoded text with a fixed prompt requesting a 0–1 readability
  score. Cache scores to disk keyed by `candidate_id` + prompt hash.
  After the first run, replay is free.
- `haiku_top_pick` — show Haiku the top-K candidates from a non-LLM
  pre-rank and ask it to pick the most readable. (Tests whether
  comparison-based scoring beats independent scoring.)

Hybrid:
- `binary_ngram_then_haiku` — non-LLM gate to top-25, then Haiku rerank.
  Accounts for cost-conscious deployment.

### Replay engine

`scripts/evaluate_triage_strategies.py`:

```
for strategy in registered_strategies:
    for case in population:
        candidates = load_cached_candidates(case)
        ranked = strategy.rank(candidates, case.metadata)
        record_metrics(strategy, case, ranked)
emit summary CSV + per-cell breakdown
```

Runs in seconds for non-LLM strategies. LLM strategies have a one-time
scoring cost on the first run; subsequent runs are also free.

---

## Component 5 — Metrics

Per (strategy × case), compute:

- **`recall@K`** for K ∈ {1, 3, 10, 25}: did at least one rescuable candidate
  appear in the strategy's top-K? Operational metric.
- **`mrr_rescuable`**: 1 / rank of first rescuable candidate. Captures how
  deep the agent must dig.
- **`top1_is_rescuable`**: stricter version of recall@1.
- **`regression_vs_baseline`**: did this strategy *remove* a rescuable
  candidate that `baseline_ngram` had in its top-K? Catches silent-failure
  modes — the most important gate for whether to ship a strategy.
- **`cost_per_1k_candidates`**: total seconds and total LLM tokens
  consumed. Necessary for fair Haiku comparison.

Aggregate by:
- Overall (all synthetic).
- Per language (en/la/de).
- Per transform family.
- Per length bucket.
- Held-out (separate report).

Report format: CSV table + a small markdown summary (`reports/triage/<run-id>.md`)
per evaluation, similar to existing parity matrix outputs.

---

## Files and modules

### New

| Path | Purpose |
|------|---------|
| `src/triage/__init__.py` | Module marker |
| `src/triage/types.py` | `CachedCandidate`, `Strategy`, `LabelTiers` dataclasses |
| `src/triage/labeler.py` | Ground-truth labeling logic |
| `src/triage/strategies/__init__.py` | Strategy registry |
| `src/triage/strategies/baseline.py` | `baseline_ngram` |
| `src/triage/strategies/non_llm.py` | dict-rate, consecutive-dict, perplexity, dedupe, MMR |
| `src/triage/strategies/llm.py` | Haiku-based strategies + score cache |
| `src/triage/metrics.py` | Recall@K, MRR, regression, cost accounting |
| `scripts/build_transform_triage_population.py` | Generate synthetic cases |
| `scripts/capture_transform_candidates.py` | Run transform search, dump candidate lists |
| `scripts/label_transform_candidates.py` | Label cached candidates |
| `scripts/evaluate_triage_strategies.py` | Replay strategies, emit metrics |
| `frontier/transform_triage_population.jsonl` | Versioned population manifest |
| `frontier/transform_triage_holdout.jsonl` | Held-out historical anchors |
| `tests/test_triage_strategies.py` | Unit tests for each strategy |
| `tests/test_triage_metrics.py` | Unit tests for metrics with hand-built fixtures |

### Modified

| Path | Change |
|------|--------|
| `src/analysis/transform_search.py` | Add `dump_full_candidates: int` parameter that returns full top-N rather than only the agent-page output |
| `src/testgen/builder.py` | Add `transform_compose` mode for synthetic transformed ciphertexts |

### Reused as-is

- `src/automated/runner.py` — drives the downstream solver inside the
  labeler's `rescuable` computation.
- `src/analysis/zenith_solver.py` — provides the binary 5-gram perplexity
  baseline.
- `src/benchmark/scorer.py` — edit-aware alignment for `readable_now`.

---

## Implementation order

1. **Population generator + held-out manifest** (`scripts/build_transform_triage_population.py`,
   `transform_compose` mode in `testgen/builder.py`). Verify generator
   reproducibility: same config → same population, byte-identical.

2. **`dump_full_candidates` flag in transform_search.py** + a small unit
   test that confirms the flag returns a strict superset of the agent-page
   output.

3. **Capture script** for a small subset (e.g. 20 cases) end-to-end. Confirm
   cache format, parallelism, and resumability.

4. **Labeler — cheap labels first** (`transform_correct`, `readable_now`).
   These let us start computing approximate metrics before the expensive
   `rescuable` labels are ready. Keeps early iteration fast.

5. **Strategy harness scaffolding + `baseline_ngram`** strategy. Confirm
   replay works against the small subset and that `baseline_ngram` reproduces
   the candidate list as currently produced.

6. **Add the cheap non-LLM strategies** (`dict_rate_weighted`,
   `consecutive_dict_words`, `binary_ngram_perplexity`, `cluster_dedupe`,
   `mmr_diversity`). Compare against baseline on `readable_now` recall.

7. **Expensive `rescuable` labeling** on the full population. This is when
   the framework starts producing operationally meaningful metrics.

8. **Full population capture + labeling** (the long-running job).

9. **Add LLM strategies** (`haiku_score`, `haiku_top_pick`,
   `binary_ngram_then_haiku`) with score caching. One-time API cost; replays
   are free thereafter.

10. **Held-out evaluation** + report generation. First end-to-end answer to
    "does any strategy beat baseline on held-out historical anchors with an
    acceptable cost?"

Each step lands as a separate commit and a small write-up in
`reports/triage/`. Stop and reassess after step 7 — that is the earliest
point where the framework can answer the original motivating question.

---

## Success criteria / falsifiability

This framework is worth building if and only if it produces actionable
findings of one of the following shapes:

- **"Cheap fix wins":** one of the non-LLM strategies (likely
  `cluster_dedupe_then_rank` or `binary_ngram_perplexity`) beats baseline by
  a meaningful margin on `recall@10` with no `regression_vs_baseline`. We
  ship it without any LLM gate.
- **"LLM gate wins on hard slice only":** Haiku-based strategies beat
  non-LLM only on a specific per-cell slice (e.g., long ciphertexts with
  many near-duplicate candidates). We deploy the gate conditionally.
- **"No strategy beats baseline on held-out":** baseline is fine; the
  perceived problem was an artifact of the small stress-case set. We do not
  ship a triage change. This is also a valid outcome — it would have saved
  the time we'd otherwise spend tuning the wrong thing.

If after step 10 none of these three shapes is clearly true, the framework
itself is suspect and needs review (likely the labeler, the population
distribution, or both).

### Quantitative targets

- Population size: ≥ 500 synthetic cases, ≥ 10 held-out.
- Per-cell statistical power: ≥ 30 cases per (language × family) cell.
- Strategy iteration time: a new non-LLM strategy must be evaluable across
  the full population in under 60 seconds.
- Reproducibility: re-running the full pipeline from a fixed population
  manifest must produce byte-identical metric CSVs.

---

## Open questions

1. **Threshold for `rescuable`.** Is 0.85 char accuracy the right bar? Too
   high and we under-label good candidates; too low and we over-label
   marginal ones. Calibrate against existing parity-matrix results before
   committing.

2. **Prompt for `haiku_score`.** Different prompts will give different
   rankings. Need a small prompt-ablation step before declaring `haiku_score`
   results meaningful. Probably 3-5 variants compared on a small held-in
   subset.

3. **What about candidates that are "almost transform-correct"?** A
   candidate with the right transform but wrong substitution may decode to
   nonsense even though it is structurally correct. Rescuable labeling
   handles this — a downstream homophonic anneal from there will recover.
   But the `transform_correct` label may visually disagree with
   `readable_now`. Worth instrumenting both and noting the divergence.

4. **Population drift.** When the transform search code itself is changed,
   cached candidate lists become stale. Treat the population manifest +
   transform_search code version as a joint cache key, and expire caches
   when either changes. Document this clearly so old metrics are not
   silently compared against new code.

---

## Future extensions

Once the framework exists, several follow-ons become cheap:

- **In-search scoring tuning.** If `binary_ngram_perplexity` consistently
  beats the current internal scorer, plug the binary model into the search
  itself, not just the post-search triage.
- **Candidate-list shape diagnostics.** Per-case features (variance of
  `original_rank` scores, diversity of decoded texts, presence of
  near-duplicates) may predict when triage matters most. Useful for
  conditional deployment.
- **Cross-family transfer.** Does a strategy tuned on `t_homophonic`
  generalize to `t_periodic`? Per-cell metrics answer this directly.
- **Online deployment.** Once a strategy is chosen, expose it through the
  agent loop as a re-ranker on the agent-facing top page, behind a config
  flag. The framework continues to validate the strategy as the agent loop
  evolves.
