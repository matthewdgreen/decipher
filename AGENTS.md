# Decipher — AGENTS.md

Project context for Codex sessions. Keep this file updated as the project evolves.

---

## What This Is

A CLI research tool for classical cipher cryptanalysis. Primary focus:
- **Monoalphabetic substitution ciphers** with arbitrary symbol alphabets
- **Historical manuscripts** (Borg cipher in Latin, Copiale cipher in German)
- **AI-assisted decipherment** using Codex tool-use API
- **Benchmark evaluation** against a dataset of solved historical ciphers

Licensing note:
- Decipher is now GPLv3-licensed.
- `src/analysis/zenith_solver.py` is derived from the Zenith project by
  beldenge and must retain explicit attribution in code comments and docs.
- The original Zenith English binary model is still not redistributed in this
  repo. Current understanding is that BNC itself is probably not the blocking
  issue; the remaining redistribution uncertainty is primarily the Blog
  Authorship Corpus, which Zenith documents as part of its training mix.
  Replacing that model with Decipher-built models remains active work.

---

## Key Files

```
src/
  cli.py                  — CLI entry point (benchmark, crack, testgen subcommands)
  models/
    alphabet.py           — Alphabet class (symbol↔integer mapping, multisym support)
    cipher_text.py        — CipherText dataclass (raw text + alphabet + word structure)
    session.py            — Headless Session: cipher text, key dict, apply_key()
  analysis/
    frequency.py          — mono/bigram/trigram frequency, chi-squared
    ic.py                 — Index of Coincidence
    pattern.py            — Word isomorphs, pattern dictionary, match_pattern()
    dictionary.py         — load_word_set(), score_plaintext(), get_dictionary_path(lang)
    solver.py             — Algorithmic solver: hill_climb_swaps(), auto_solve()
    ngram.py              — N-gram language models with lazy caching
    signals.py            — Multi-signal scoring panel (6 metrics)
    segment.py            — Rank-aware no-boundary word segmentation
    zenith_solver.py      — Zenith-parity SA for homophonic ciphers: exact entropy score,
                            un-normalized acceptance, binary model loader (26^5 float32)
  agent/
    prompts_v2.py         — V2 brief-style system prompt (no rigid phases)
    tools_v2.py           — V2: 32 tools across 9 namespaces + WorkspaceToolExecutor
    loop_v2.py            — V2 agent loop with workspace integration
    state.py              — AgentState, Checkpoint (checkpointing + rollback)
  workspace/
    __init__.py           — Branch and Workspace classes for v2 agent
  preprocessing/
    s_token_converter.py  — S-token to letter normalization for API compatibility
  artifact/
    schema.py             — RunArtifact, BranchSnapshot, ToolCall dataclasses
  benchmark/
    loader.py             — BenchmarkLoader: reads JSONL manifest + splits + data files
    runner_v2.py          — V2 BenchmarkRunner: with artifacts and preprocessing
    scorer.py             — score_decryption(), format_report() (char/word accuracy)
  automated/
    runner.py             — Automated-only/no-LLM runner using native solving techniques
  services/
    claude_api.py         — ClaudeAPI: send_message(), estimate_cost(), retry/error helpers
  ocr/
    engine.py             — OCREngine: process_image(), process_text()
    vision.py             — VisionOCR: Codex Vision for symbol extraction
  ciphers/
    substitution.py       — SubstitutionCipher: encrypt/decrypt/random_key
    caesar.py             — CaesarCipher: brute_force()
  external/
    azdecrypt.py          — Stub for AZdecrypt integration (not implemented)
    cryptocrack.py        — Stub for CryptoCrack integration (not implemented)
resources/
  dictionaries/
    english_common.txt    — 5000 common English words (uppercase, freq-ordered)
    latin_common.txt      — 4440 Latin words (medical/pharmaceutical focus)
    german_common.txt     — 3057 German words (18th-century Masonic focus)
tests/
  test_models.py          — model and session tests
  test_analysis.py        — frequency, IC, pattern, dictionary tests
  test_ciphers.py         — cipher primitive tests
  test_benchmark.py       — loader, runner, scorer tests
  test_workspace.py       — branch workspace tests
  test_signals.py         — scoring panel tests
  test_segment.py         — no-boundary segmentation tests
  test_automated_runner.py — no-LLM automated runner and CLI bypass tests
  test_agent_reliability.py — loop fallback and reliability behavior tests
  test_zenith_solver.py   — binary model loading, entropy/score formula, SA recovery (23 tests)
```

---

## Architecture Decisions

### Token model
All analysis works on `list[int]` token IDs, not strings. `Alphabet` is the bidirectional mapping. This supports both single-char (A-Z) and multi-char (S001, S002 OCR-style) symbol sets uniformly.

### Session and workspace state
`Session` is a lightweight headless container used by solver algorithms. V2 agent runs use `Workspace`, which holds the immutable cipher text plus named branch keys for hypothesis exploration. There are no Qt signal dependencies in the active CLI path.

### Automated-only mode
`--automated-only` is a no-LLM path for `benchmark`, `crack`, and cached `testgen` runs. It lives in `src/automated/runner.py` so native parity work can evolve separately from `agent/loop_v2.py`. Artifacts are marked `run_mode: automated_only`, `automated_only: true`, with zero tokens and zero estimated cost. Testgen automated-only requires cached plaintext because generating fresh synthetic prose would require an LLM call.

Automated homophonic runs now also support `--homophonic-budget {full,screen}`.
Use `screen` for comparative scorer/strategy sweeps and `full` for headline
parity claims. The budget choice is recorded in automated artifacts.
The default automated homophonic solver path is now `zenith_native` with the
bundled parity model when available. Use `--legacy-homophonic` on CLI/scripts
to force the older pre-`zenith_native` homophonic path for comparison runs.
For iterative native-solver work, the automated homophonic path also supports
two opt-in env profiles:
- `DECIPHER_HOMOPHONIC_SEARCH_PROFILE=dev|full`
- `DECIPHER_HOMOPHONIC_REPAIR_PROFILE=dev|full`

Both default to `full`. `dev` is for faster local experimentation only and
should not be used for parity claims; artifacts record both values so runs are
not silently mixed together.
For `DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native`, seed exploration can
also be parallelized across local CPU cores with
`DECIPHER_HOMOPHONIC_PARALLEL_SEEDS=<N>`. This is opt-in and preserves the
same seed set/budget, but it evaluates multiple seeds concurrently instead of
serially. Use it for faster wall-clock experimentation; artifacts record the
parallel worker count.
If `DECIPHER_HOMOPHONIC_PARALLEL_SEEDS` is not set, the solver now auto-sizes
to `max(1, os.cpu_count() - 1)` capped by the seed count, so modern
`zenith_native` runs will usually parallelize by default on multi-core local
machines.
There is also an experimental opt-in postprocess for raw no-boundary
`zenith_native` output:
- `DECIPHER_HOMOPHONIC_POLISH=1`

This currently runs a conservative segmentation + one-edit local repair pass
after `zenith_native` and records a `postprocess` block in the artifact. Keep
it off for headline parity/frontier claims until it has been evaluated more
thoroughly.

Benchmark-backed automated runs should use the benchmark manifest's
`plaintext_language` when available. Do not rely on `test_id` prefixes alone
for parity/frontier cases such as `parity_borg_*` or `parity_copiale_*`;
those prefixes previously caused historical runs to fall back to English by
accident.

### Automated preflight for LLM runs
LLM-enabled `benchmark`, `crack`, and testgen-suite runs now execute the native automated solver before iteration 1 by default. The result is stored in `artifact.automated_preflight`, exposed to the agent as an `automated_preflight` workspace branch, and summarized in the first context message without benchmark ground-truth accuracy. Use `--no-automated-preflight` to disable this when measuring the unaided agent loop.

### Key representation
`dict[int, int]` — cipher token ID → plaintext token ID. Partial keys are fine; unmapped tokens show as `?`. `apply_key()` uses the plaintext alphabet's `_multisym` flag to determine output spacing (not the cipher alphabet's flag — important fix).

### Multisym alphabets
Canonical benchmark transcriptions use space-separated S-tokens (S001 S002 ...) with ` | ` as word separator. `parse_canonical_transcription()` handles this. Newlines in source files are also word boundaries.

### Language support
`analysis/dictionary.py` has `get_dictionary_path(language)` for `en`, `la`, `de`.
`agent/prompts.py` has language-specific `FREQUENCY_ORDERS`, `LANGUAGE_NOTES`, and `get_system_prompt(language)`.
Benchmark auto-detects: borg→`la`, copiale→`de`.

### Benchmark dataset
Decipher expects a local checkout of the benchmark repository, typically at a
path like `/path/to/cipher_benchmark/benchmark/`.
- Benchmark repo: `https://github.com/matthewdgreen/cipher_benchmark`
- `manifest/records.jsonl` — currently 896 records: Borg, Copiale, DECODE/Gallica, multilingual synthetic substitution, and tool-bundled parity records
- `splits/borg_tests.jsonl` — 45 tests (15 Track B: transcription→plaintext)
- `splits/copiale_tests.jsonl` — 45 tests (15 Track B)
- `splits/*_ss_synth*_tests.jsonl` — multilingual synthetic simple-substitution Track B tests
- Track B (transcription2plaintext) = canonical S-token transcription → plaintext
- Borg: monoalphabetic, 33 symbols, Latin pharmaceutical text
- Copiale: homophonic, 86 symbols, German Masonic text
- Use `scripts/validate_benchmark.py` before relying on a benchmark checkout for parity work.
- Benchmark curation lives in `../cipher_benchmark`; Decipher solver/harness code lives here.
- Keep parity tests distinct from agentic-advantage tests. Parity asks whether the agent can match non-agentic solvers; advantage asks whether agentic context, OCR, diagnosis, or hypothesis management improves beyond them.
- Decipher also keeps a local frontier automated suite in `frontier/automated_solver_frontier.jsonl`. This is intentionally separate from benchmark parity splits: it is a compact regression/frontier pack for automated-only runs, with explicit classes (`known_good`, `shared_hard`, `bad_result`, `slow_result`) and optional synthetic case definitions.
- The frontier suite now also includes a `shared_hard` class for synthetic
  cases that are meaningfully challenging for both Decipher and Zenith without
  being collapse cases. Current anchors are `synth_en_80honb_s2`,
  `synth_en_200honb_s3`, and `synth_en_200honb_s6`.
- There is now also a small English model evaluation packet in
  `frontier/english_model_eval.jsonl`. Use this when comparing
  `DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native` across different English
  binary models (for example, proprietary Zenith vs Decipher-built Gutenberg
  models). It is intentionally narrow: two known-good English homophonic
  cases, two short runtime/frontier cases, and Zodiac 408.
- There is also a broader English model comparison packet in
  `frontier/english_model_comparison.jsonl`. Use this when checking whether a
  proprietary or newly built English binary model is broadly stronger on
  Zodiac-like English homophonic ciphers, rather than merely better on Zodiac
  408 itself. It expands the synthetic side of the packet to seven English
  no-boundary homophonic cases plus Zodiac 408 and is meant for paired A/B
  runs with the same `zenith_native` solver settings and only the model path
  changed.
- The main automated frontier suite now defaults external runs to Zenith only
  through `external_baselines/zenith_only.json`. This keeps routine frontier
  runs fast and avoids dragging `zkdecrypto-lite` through every comparison
  unless explicitly requested with `--external-config external_baselines/local_tools.json`.

### Parity evaluation modes
When comparing Decipher against Zenith, zkdecrypto-lite, or future baselines,
distinguish two evaluation modes:

- **Blind parity**: ciphertext-derived routing only. No benchmark metadata such
  as language or source family is passed to any solver.
- **Context-aware parity**: benign metadata such as language or source family
  may be used, but artifacts and summary rows must record which solvers
  actually received and consumed that context.

Do not silently compare Decipher in a context-aware mode against external
wrappers that only received raw ciphertext. If wrapper capabilities differ,
make that asymmetry explicit in reporting.

---

## Major Achievements (April 2026)

### ✅ **V2 Agentic Framework Completed**
Successfully implemented state-of-the-art agent-driven cryptanalysis system:
- **Branching workspace** with fork/merge/compare operations (src/workspace/)
- **32 specialized tools** across 9 namespaces (src/agent/tools_v2.py)
- **Multi-signal scoring** with 6 different metrics (src/analysis/signals.py)
- **Agent-driven termination** via meta_declare_solution (no rigid phases)
- **Full observability** via comprehensive run artifacts (src/artifact/schema.py)
- **Synthetic hard benchmark solved exactly**: synth_en_250nb_s4 reached 100% in 7 iterations

### ✅ **API Compatibility Layer Implemented**
Robust preprocessing and framing for reliable API interaction:
- **Automatic S-token normalization** (src/preprocessing/s_token_converter.py)
- **Manuscript-analysis framing** for academic historical research tasks
- **Model selection**: Codex Sonnet 4.6 recommended for decipherment tasks
- **Transparent artifact tracking** of preprocessing applied

### ✅ **Advanced Cryptanalytic Capabilities**
V2 system demonstrates sophisticated reasoning:
- **Constraint propagation**: "AMAMUS → H=A, C=M, I=U, G=S"
- **Conflict detection**: "K=A but H=A from AMAMUS - conflict!"
- **Strategic progression**: Overview → patterns → word candidates → constraints
- **Latin domain expertise**: Identifies pharmaceutical vocabulary (CARERE, etc.)
- **Multi-hypothesis testing** across branching workspace

### ✅ **Reliability and Homophonic Guardrails Added**
Recent testgen work turned failure logs into tool-design improvements:
- **Final-iteration preflight**: the loop can declare a strong branch before an avoidable last API call
- **Best-branch fallback**: API overloads/errors preserve the best candidate instead of losing the run
- **Rank-aware segmentation**: no-boundary English is segmented using frequency-ranked dictionary costs
- **Homophonic diagnostics**: tools identify ambiguous letters, absent letters, and likely split homophones
- **`run_python` audit trail**: Python remains allowed, but every use records a justification and is highlighted in reports as a tool-design signal
- **Automated-only baseline**: `--automated-only` runs native solvers without LLM API calls and writes comparable zero-cost artifacts
- **Automated preflight**: LLM runs receive a no-LLM native candidate before iteration 1 and can adopt, repair, or reject it
- **Simple-substitution parity path**: English simple substitution now uses bijective continuous n-gram annealing, preserving one-to-one mappings while allowing unused plaintext letters to enter the key
- **Homophonic quality gates**: automated homophonic runs now retry low-diversity/collapsed outputs across seeds and rank candidates by anneal score adjusted for plaintext quality
- **Diversity-aware homophonic search**: `search_homophonic_anneal` supports a plaintext-diversity objective to reduce ETAOIN-ish collapsed false positives on short no-boundary ciphers
- **Historical routing fix**: automated-only and automated preflight runs now honor benchmark plaintext language metadata and route overcomplete symbol alphabets to non-bijective/homophonic search instead of failing immediately on bijective assumptions
- **Status semantics fix**: automated-only successful runs report `completed` rather than `solved`, so garbage-looking plaintext is not implicitly treated as a declared solution

### ✅ **Zenith-Parity Homophonic Solver — 99.3% on Zodiac 408**
Implemented `src/analysis/zenith_solver.py`: a faithful Python port of Zenith's SA
algorithm for Zodiac-style homophonic ciphers. Achieves **99.3% character accuracy**
on the Zodiac 408 in ~160 s, closing the gap from the previous best of 83.6%.

Root cause of the former ceiling — two independent bugs in the old `zenith_exact` profile:

1. **Wrong counterweight metric**: used `IoC^(1/6)` as a multiplier.
   Zenith uses **Shannon entropy** as a divisor:
   `score = mean_5gram_log_prob / entropy^(1/2.75)`.
   IoC and entropy are correlated but have very different gradient shapes.

2. **Over-normalized acceptance**: used `exp(delta / (ngram_count * temp))`.
   Zenith uses `exp(delta / temp)` — **no** `ngram_count` normalization.
   With `ngram_count ≈ 202`, the effective temperature was ~202× too cold, so
   nearly all downhill moves were rejected.

Key implementation details:
- Binary model loader: Java `DataOutputStream` big-endian format, magic `0x5A4D4D43`,
  26^5 float32 array → numpy for O(1) lookup (~47 MB RAM).
- Entropy table precomputed once per cipher length; updated incrementally per proposal.
- Biased proposal bucket: 80% flat + 20% corpus letter frequencies from binary model.
- `zenith_native` score profile dispatched from `automated/runner.py` via
  `DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native`.
- No modifications to `homophonic.py`; imports `HomophonicAnnealResult` for compatibility.
- Requires numpy (already used in `signals.py`); binary model at
  `other_tools/zenith-2026.2/zenith-model.array.bin` (git-ignored, see provenance note).

---

## Remaining Challenges

### 1. ⏳ **Hardest homophonic/no-boundary tests**
The hardest synthetic preset (`synth_en_200honb_s6`) is the current stress case. The tool now exposes homophonic evidence explicitly, but the next run should confirm whether the agent uses those tools instead of ad hoc Python.

### 2. ✅ **Homophonic search quality — Zodiac 408 solved (99.3%)**
The `zenith_native` score profile (`DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native`) now
achieves **99.3% character accuracy** on the Zodiac 408 in ~160 s. This closes the main
gap vs. Zenith. See the "Zenith-Parity Homophonic Solver" achievement above for the two
root-cause bugs that were fixed.

Remaining open questions in this area:
- **Bundled redistributable model**: the repo now ships a redistributable English binary
  model at `models/ngram5_en.bin`, and `zenith_native` should prefer it by default unless
  `DECIPHER_NGRAM_MODEL_EN` overrides it. This removes the immediate clone-and-run blocker,
  but the bundled model is still a work in progress and should be improved with broader corpora.
- **Corpus tooling expansion**: `python -m tools.corpus` now supports mixed English-source
  downloads from Gutenberg, OANC, and MASC, with a corpus manifest recording provenance.
  OANC/MASC access works around the current expired TLS certificate on `anc.org`; keep that
  exception scoped to ANC hosts only.
- **BNC planning and attribution**: corpus tooling now also supports BNC as a local licensed
  import path for English (`--source bnc --bnc-source-dir ...`). Decipher must never
  redistribute raw BNC text; only derived models plus explicit provenance/attribution.
- **Non-English model builds**: corpus tooling now supports Gutenberg-backed automated model
  builds for `de`, `fr`, `it`, and `la` as well as `en`. Additional non-English sources and
  better corpus mixes remain future work.
- **Broader cipher class coverage**: non-English homophonic ciphers (Copiale/German) still
  fall back to `homophonic.py` profiles unless a corresponding `models/ngram5_<lang>.bin`
  exists. A German equivalent binary model does not yet exist.
- **Agent tool exposure**: `search_homophonic_anneal` in the agentic tool set still uses
  the old `homophonic.py` profiles. The `zenith_native` path is automated-runner-only for
  now; exposing it as an agent tool is future work.
- **No-boundary homophonic ciphers**: `synth_en_200honb_s6` (English, no word boundaries)
  is still the hardest stress case. The `zenith_native` path runs on no-boundary ciphers
  but has not been benchmarked against these yet.

Historical ablation record (kept for reference — superseded by `zenith_native`):
- `balanced + mixed_v1`: best prior result, ~83.6% Zodiac.
- `zenith_exact` (old): ~7.8% with `single_symbol`; ~2.5% with `mixed_v1`. The score
  formula was correct but the acceptance normalization bug (~202× too cold) destroyed it.
- `ioc_ngram`: ~34.1%; collapsed into high-IoC garbage.
- `ngram_distribution`: ~30.1%; over-compressed repetitive text.
- All these failures were search-dynamics failures, not just score-function failures.
  The decisive fix was correcting the acceptance temperature, not just the score.

### 3. 📚 **Continuous n-gram model provenance**
High-quality English homophonic solving can still auto-discover Zenith's local English model from `other_tools/`, but treat it as an optional local dependency until provenance is fully resolved. Current understanding: BNC-derived products appear acceptable, Leipzig downloadable corpora appear permissive, and OANC/MASC are open; the main remaining redistribution uncertainty is the Blog Authorship Corpus, which Zenith documents as part of its training mix and which appears to be limited to non-commercial research use. Do not bundle or publish Zenith model files from this repo until that remaining issue is resolved.

Decipher does not currently have comparable continuous corpus n-gram files for Latin, German, French, or Italian. It has word-list fallbacks and dictionaries, but those are weaker than the Zenith-style English model. Future work should add a model registry with language, order, source, checksum, license/provenance, row count, and redistribution status, and should record the selected model in run artifacts.

Current April 2026 state for non-English bundled models:
- Latin now has both `models/ngram5_la.bin` (100 Gutenberg books) and
  `models/ngram5_la_500.bin` (500 Gutenberg books) available locally for
  focused experiments.
- On the focused Borg `parity_borg_latin_borg_0109v` probe, forcing
  `zenith_native` with the Latin binary model reaches a reproducible
  no-boundary partial-Latin basin at about `13.9%` character accuracy with the
  full 8-seed budget, but still `0.0%` word accuracy.
- The 500-book Latin model slightly improves the anneal score on that case
  versus the 100-book model but does not yet move headline accuracy.
- The first conservative post-`zenith_native` polish loop does not yet improve
  `0109v`; its local one-edit repairs were too weak to change the candidate.

### 4. 🎭 **Historical Copiale/Borg generalization**
Synthetic tests are useful for controlled iteration, but the historical benchmark still needs broader runs to separate synthetic overfitting from durable cryptanalytic progress. The first correctness pass is now in place: historical automated runs use benchmark language metadata instead of English fallback, and routing no longer assumes that word boundaries imply a simple bijective substitution.

Additional current read:
- Borg `0077v` routes to `zenith_native` because its symbol inventory exceeds
  the Latin plaintext alphabet size.
- Borg `0109v` still routes to the substitution path by default because its
  symbol inventory does not trip the current overcomplete/homophonic heuristic.
- Focused probes show that forcing `zenith_native` on `0109v` produces a more
  fluent no-boundary Latin stream than the substitution path, but not a clear
  win yet. This now looks like a cleanup/segmentation problem at least as much
  as a raw search problem.

### 5. 🧭 **Context capability audit**
Blind vs. context-aware parity is now the intended evaluation framework.
Current local evidence suggests:
- **zkdecrypto-lite**: likely English-only with a very thin CLI surface; it may
  genuinely lack practical context-awareness hooks.
- **Zenith**: richer API surface than our current wrapper uses. It clearly
  supports optimizer settings and transformation steps, and it may support some
  context-adjacent capabilities that should be audited before claiming parity
  comparisons are fully fair.

Treat "wrapper omission" and "upstream tool limitation" as separate findings.

### 5. 🔧 **Model selection**
Sonnet 4.6 performs best on historical manuscript analysis tasks. Opus 4.7 is more
conservative with encoded historical text. See Model Selection section for guidance.

---

## V2 Architecture (✅ Implemented)

Successfully replaced rigid v1 agent with sophisticated v2 framework:

### Core principle: Agent drives, tools assist
✅ **Implemented features:**
1. **Full visibility** — observe/decode/score tools for comprehensive analysis
2. **Rich tool set** — 32 tools across 9 namespaces (workspace, observe, decode, score, corpus, act, search, run_python, meta)
3. **Agent freedom** — No phases, agent plans own strategy
4. **Hypothesis tracking** — Branching workspace preserves exploration history

### Tool Arsenal (32 tools implemented)
✅ **workspace_* (5 tools)** — fork, list, delete, compare, merge
✅ **observe_* (4 tools)** — frequency, isomorph_clusters, ic, homophone_distribution
✅ **decode_* (8 tools)** — show, unmapped, heatmap, letter_stats, ambiguous_letter, absent_letter_candidates, diagnose, diagnose_and_fix
✅ **score_* (3 tools)** — panel, quadgram, dictionary
✅ **corpus_* (2 tools)** — lookup_word, word_candidates
✅ **act_* (5 tools)** — set_mapping, bulk_set, anchor_word, clear, swap_decoded
✅ **search_* (3 tools)** — hill_climb, anneal, homophonic_anneal
✅ **run_python (1 tool)** — allowed escape hatch with required justification
✅ **meta_* (2 tools)** — request_tool, declare_solution

### Termination criteria
✅ **Implemented:**
- Agent calls `meta_declare_solution` when confident
- Natural exhaustion at max_iterations
- No arbitrary score thresholds

### Advanced capabilities demonstrated
✅ **Constraint reasoning**: Detects mapping conflicts
✅ **Strategic thinking**: Plans multi-step analysis
✅ **Domain expertise**: Recognizes Latin pharmaceutical vocabulary
✅ **Hypothesis management**: Uses workspace branches effectively

---

## Running

```bash
# V2 Benchmark (recommended)
.venv/bin/decipher benchmark /path/to/cipher_benchmark/benchmark \
  --source borg --model claude-sonnet-4-6 --verbose

# V2 Single test with full analysis
.venv/bin/decipher benchmark /path/to/cipher_benchmark/benchmark \
  --test-id borg_single_B_borg_0045v --model claude-sonnet-4-6 --max-iterations 15

# V2 crack from text (automatic S-token preprocessing)
echo "S025 S012 S006 | S003 S007" | .venv/bin/decipher crack \
  --language la --model claude-sonnet-4-6 --canonical

# Hardest synthetic regression only
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --preset hardest --model claude-sonnet-4-6 --max-iterations 25 --verbose

# No-LLM automated baseline on the hardest synthetic cached instance
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --preset hardest --automated-only

# Chunkable no-LLM automated parity matrix
PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py \
  --presets tiny medium hard hardest \
  --artifact-dir artifacts/automated_parity_matrix/default

# Frontier automated suite
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers decipher external \
  --artifact-dir artifacts/frontier_suite/default

# To include slower external wrappers such as zkdecrypto-lite explicitly:
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers decipher external \
  --external-config external_baselines/local_tools.json \
  --artifact-dir artifacts/frontier_suite/all_external

# Homophonic ablation packet, quick screening budget
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/homophonic_profile_ablation.jsonl \
  --solvers decipher \
  --homophonic-budget screen \
  --artifact-dir artifacts/homophonic_ablation/balanced_screen

# Zenith-parity native solver on Zodiac 408 (99.3% in ~160s)
DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native \
  PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py \
  --solvers decipher \
  --benchmark-split /path/to/cipher_benchmark/benchmark/splits/parity_zodiac.jsonl \
  --benchmark-root /path/to/cipher_benchmark/benchmark \
  --artifact-dir artifacts/zenith_native \
  --summary-jsonl artifacts/zenith_native/summary.jsonl \
  --summary-csv artifacts/zenith_native/summary.csv

# Example small chunk: seed sweep, first shard only
PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py \
  --families simple-wb simple-nb homophonic-nb \
  --lengths 40 200 \
  --seeds 1-20 \
  --shard-count 4 --shard-index 0 \
  --artifact-dir artifacts/automated_parity_matrix/shard0

# No-LLM crack from canonical text
echo "S025 S012 S006 | S003 S007" | .venv/bin/decipher crack \
  --language la --canonical --automated-only

# Legacy V1 commands
.venv/bin/decipher benchmark /path/to/cipher_benchmark/benchmark --source borg -v
.venv/bin/decipher crack -f input.txt --language la

# Run tests
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q

# Validate benchmark checkout
PYTHONPATH=src .venv/bin/python scripts/validate_benchmark.py \
  /path/to/cipher_benchmark/benchmark
```

---

## Development Setup

```bash
cd ~/Dropbox/src2/decipher
source .venv/bin/activate   # Python 3.11 venv
pip install -e .             # Install with entry points
```

Python 3.11 at `/opt/homebrew/bin/python3.11`. Venv at `.venv/`.

---

## Model Selection

**Recommended for V2**: `claude-sonnet-4-6` — Best results on historical manuscript analysis
**Default**: `claude-opus-4-7` — Configurable via `--model` flag

### Model Notes
- **Claude Sonnet 4.6**: Strong performance on S-token sequences and Latin/German manuscript analysis
- **Claude Opus 4.7**: More conservative with historical encoded text; use Sonnet 4.6 for decipherment
- **Preprocessing**: S-token normalization (letter substitution) improves API compatibility across models

### Configuration
Models configurable via `--model` CLI flag.
API key stored in macOS Keychain under service `decipher`, account `anthropic_api_key`.
Also reads `ANTHROPIC_API_KEY` env var.

### Performance
Sonnet 4.6 on `synth_en_250nb_s4`: exact match in 7 iterations after reliability and segmentation fixes.
`synth_en_200honb_s6` is the active hardest homophonic/no-boundary stress test.
