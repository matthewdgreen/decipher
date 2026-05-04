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
    polyalphabetic.py     — Vigenere/Beaufort/Gronsfeld solvers, keyed
                            Vigenere/Quagmire replay/search, and experimental
                            shared-tableau mutation search
    signals.py            — Multi-signal scoring panel (6 metrics)
    segment.py            — Rank-aware no-boundary word segmentation
    transform_evaluation.py — Shared transform finalist-menu validation,
                            scoring adjustment, and sorting helpers
    transform_homophonic_batch.py — Rust batch payload helpers for
                            transform+homophonic finalist probes
    zenith_solver.py      — Zenith-parity SA for homophonic ciphers: exact entropy score,
                            un-normalized acceptance, binary model loader (26^5 float32)
  agent/
    prompts_v2.py         — V2 brief-style system prompt (no rigid phases)
    tools_v2.py           — V2: 85 tools across 11 namespaces + WorkspaceToolExecutor
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
    scorer.py             — score_decryption(), format_report(); char/word accuracy
                            use edit-aware alignment so local drift can resync
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
  test_polyalphabetic.py  — periodic polyalphabetic and keyed-Vigenere tests
  test_automated_runner.py — no-LLM automated runner and CLI bypass tests
  test_agent_reliability.py — loop fallback and reliability behavior tests
  test_zenith_solver.py   — binary model loading, entropy/score formula, SA recovery (23 tests)
```

**TOOLS.md** is the canonical human-readable reference for all agent tools.
When adding, removing, or significantly changing tools in
`src/agent/tools_v2.py`, update `TOOLS.md` to match: tool name, description,
parameter table, and usage notes. The tool count in the `tools_v2.py` line
above should also be kept current.

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

The v2 agent tool surface now also exposes the modern automated stack
directly:
- `search_automated_solver` runs the same local no-LLM solver stack used by
  frontier/parity evaluation and installs the result on a branch.
- `search_homophonic_anneal` now supports `solver_profile=zenith_native|legacy`.
- `decode_repair_no_boundary` exposes the shared text-repair helper used by
  automated postprocessing: segmentation, one-edit local repairs, and local
  re-segmentation over suspicious windows. It is text-only and does not mutate
  branch keys.
- Branches now also support local word-boundary overlays. The main agent tools
  for this are:
  - `act_split_cipher_word`
  - `act_merge_cipher_words`
  - `act_apply_boundary_candidate`
  These are intended for Borg-style cases where the text is locally readable
  but globally shifted by likely transcription boundary errors.

This closes the earlier gap where the agent had automated preflight context
but could only actively invoke the older homophonic annealer from inside the
tool loop.

Agentic declarations now also carry a final reading/process summary. The
`meta_declare_solution` tool requires the agent to summarize what the text
appears to be about, record status/uncertainty notes, and say whether further
iterations would likely help. Pretty terminal mode shows this on the final
screen, and artifacts persist it as `final_summary` for follow-up analysis.

Unknown-cipher agent runs now also persist first-pass cipher identification
state. `run_v2` computes `cipher_id_report` before the first turn, includes a
compact diagnostic block in the opening context, and records final
`cipher_hypotheses` plus branch `metadata` in artifacts. Use
`workspace_create_hypothesis_branch`, `workspace_update_hypothesis`,
`workspace_reject_hypothesis`, `workspace_hypothesis_cards`, and
`workspace_hypothesis_next_steps` to keep the mode trail explicit instead of
burying cipher-type decisions in free text.
For keyed-Vigenere/Kryptos-style context, hypothesis branches now carry a
structured context prior: ordinary Vigenere-family failure must be followed by
`search_quagmire3_keyword_alphabet` before rejecting the broader
polyalphabetic family. In zero-context K2-like cases, the next-step tool also
reports family coverage debt after plain Vigenere search if the statistical
fingerprint still supports periodic polyalphabetic alternatives.
When exposed benchmark context names a cipher family, Decipher treats that as
a controlling working assumption rather than a weak hint. For example, a
keyed-tableau/Kryptos-style context prior blocks off-family transform and
homophonic search tools unless the agent explicitly passes
`override_context_cipher_family=true` with a concrete
`context_override_rationale`. Declaration tool results record the context
assumption and any override, so an artifact can be read with the caveat that
wrong context may have affected the solve.

Artifact continuation is now available through
`decipher resume-artifact <artifact.json> --extra-iterations N [--branch BRANCH]`.
It restores saved branch keys, branch tags, repair agenda items, and, for new
artifacts, custom word-boundary spans. The resumed prompt uses a compact
prior-run briefing (final summary, declaration, branch preview, open/held
repairs, and missing-tool requests) rather than blindly replaying the entire
old provider transcript. Continued artifacts record `parent_run_id` and
`parent_artifact_path`.

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
- `manifest/records.jsonl` — currently 898 records: Borg, Copiale, DECODE/Gallica, multilingual synthetic substitution, tool-bundled parity records, and curated Zodiac records
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
- As of May 2026, the main frontier suite has been rebalanced into a
  multi-family automated frontier packet. It now includes regression anchors
  for simple substitution, English homophonic, periodic polyalphabetic,
  Kryptos K1/K2 known replay, Kryptos K3 pure transposition, and known
  transposition+homophonic replay; `shared_hard` anchors such as Zodiac 408,
  short homophonic, Z340-lite replay, hidden-route transform search, and Borg
  0171v/0109v; and explicit `bad_result` rows for current gaps such as Italian
  simple substitution and Copiale p068. Some rows may carry
  `decipher_runner_options` to enable per-case search settings such as hidden
  transform ranking without requiring a global CLI flag.
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

### Benchmark context for agentic runs
Benchmark records may now carry tiered `context_layers`, `related_records`, and
`associated_documents`. Agentic benchmark runs default to
`--benchmark-context max`, which gives the agent the richest permitted
non-target context available from the record. Use `--benchmark-context none`
for blind/no-context evaluations, or narrower tiers such as `minimal`,
`standard`, `historical`, `related_metadata`, and `related_solutions` for
controlled ablations.

The v2 agent receives an initial scoped context briefing and can inspect more
through benchmark tools:
- `inspect_benchmark_context`
- `list_related_records`
- `inspect_related_transcription`
- `inspect_related_solution`
- `list_associated_documents`
- `inspect_associated_document`

These tools are intentionally scoped to records and documents explicitly listed
by the benchmark JSON. They do not provide arbitrary filesystem access. The
target record's solution is never exposed through these tools; related
solutions are only exposed under `related_solutions` or `max`.

The sibling benchmark repo also has a parallel unsolved area at
`../cipher_benchmark/benchmark/unsolved`. Decipher can load its manifest and
splits for exploratory runs, but those artifacts are hypothesis/qualitative
evidence unless a solved ground truth is available. Current unsolved curation
includes Voynich seed records, Zodiac diagnostic variants, and Scorpion S1/S5
with tentative v0.2 transcriptions.

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
- **85 specialized tools** across 11 namespaces (src/agent/tools_v2.py)
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

### ✅ **Periodic Polyalphabetic First Slice**
Implemented `src/analysis/polyalphabetic.py` and routed explicit
Vigenere-family metadata through the automated runner. Current scope:
- Clean A-Z Vigenere, Beaufort, Variant Beaufort, and Gronsfeld search.
- Synthetic ladder at `frontier/polyalphabetic_ladder.jsonl`; current
  automated run is 4/4 at 100% character accuracy.
- Agent diagnostics for periodic ciphers: `observe_kasiski`,
  `observe_phase_frequency`, and `observe_periodic_shift_candidates`.
- Keyed-Vigenere calibration and research modes:
  - `keyed_vigenere_known_replay`: known key + known tableau replay.
  - `quag3_known_replay`: known-parameter Quagmire III replay using explicit
    cycleword and tableau alphabets. Quagmire naming and the search-port
    roadmap reference Sam Blake's MIT-licensed `polyalphabetic` solver; the
    current replay implementation is independent Decipher code.
  - `DECIPHER_KEYED_VIGENERE_MODE=quagmire_search`: first bounded Quagmire III
    keyword-alphabet search scaffold. It searches keyword-shaped shared
    alphabets with a cheap score-guided screen hill climb, then
    derives/optimizes the cycleword for a finalist set. The screen supports
    Blake-style exploration knobs through `DECIPHER_QUAGMIRE_SLIP_PROB`
    (default `0.001`) and `DECIPHER_QUAGMIRE_BACKTRACK_PROB` (default `0.15`).
    Finalist ranking includes a strict
    no-boundary word-hit signal (`DECIPHER_QUAGMIRE_WORD_WEIGHT`, default
    `0.25`) because the generic segmenter is too permissive for gibberish
    basins. `DECIPHER_QUAGMIRE_DICTIONARY_STARTS=<N>` adds the first N
    language-dictionary keyword-shaped starts for ordinary keyword recovery.
    `DECIPHER_QUAGMIRE_CALIBRATION_KEYWORD=<WORD>` records distance/rank
    diagnostics for solved calibration experiments without steering the search.
    This is inspired by Sam Blake's strategy but not yet a full
    shotgun/backtracking port.
  - `DECIPHER_KEYED_VIGENERE_MODE=search`: recover periodic key over supplied
    candidate keyed alphabets/tableau keywords.
  - `DECIPHER_KEYED_VIGENERE_MODE=tableau_search`: test standard A-Z first,
    then keyword-derived tableaux from
    `DECIPHER_KEYED_VIGENERE_TABLEAU_KEYWORDS`.
  - `DECIPHER_KEYED_VIGENERE_MODE=alphabet_anneal`: experimental shared-tableau
    mutation search. Guided mode is on by default and adds frequency-pressure
    plus per-phase common-letter swaps before falling back to random
    swap/move/reverse mutations; useful as a scaffold/diagnostic, not yet
    robust blind Kryptos recovery.
  - `eval/scripts/capture_keyed_vigenere_candidates.py` now also supports
    phase-constraint, offset-graph, and constraint-graph starts for blind K2
    population studies. It also supports explicit tableau perturbation
    controls; those are solution-bearing scorer/ranker calibrations, not
    blind evidence. These diagnostics are not yet production-grade Kryptos
    solvers.
- Rust fast-kernel work has started under `rust/decipher_fast`, with Python
  wrapper code in `src/analysis/polyalphabetic_fast.py`.
  `search_quagmire3_shotgun_fast` implements the first Blake-style Quagmire
  III restart/hillclimb loop in compiled code and parallelizes independent
  restart jobs across local CPU cores. The agent-facing
  `search_quagmire3_keyword_alphabet` tool exposes this through
  `engine="rust_shotgun"` plus `hillclimbs`, `restarts`, `threads`,
  `slip_probability`, and `backtrack_probability`. Automated no-LLM
  Quagmire runs can use the same engine through
  `DECIPHER_QUAGMIRE_ENGINE=rust_shotgun`,
  `DECIPHER_QUAGMIRE_HILLCLIMBS`, and `DECIPHER_QUAGMIRE_THREADS`. Keep
  Python as the
  reference path for semantics, but use the Rust engine for broad candidate
  searches after `tests/test_polyalphabetic_fast.py` passes locally. New
  users should run `scripts/setup_dev.sh` after activating the virtualenv; it
  performs `pip install -e .` and builds the Rust module. Use
  `scripts/build_rust_fast.sh` to rebuild only the Rust module, and
  `decipher doctor` to check availability. Agent tools should not silently
  replace `rust_shotgun` with `python_screen`: the Python screen is
  reference/diagnostic scaffolding, not an equivalent large-scale search.
- The Rust fast-kernel module also exposes the first Zenith-native
  homophonic/transform acceleration path. Rust is now the default for the
  one-seed Zenith solver and the transform-candidate batch evaluator used by
  solver-backed rank/full transform validation. Use
  `DECIPHER_ZENITH_NATIVE_ENGINE=python` and
  `DECIPHER_TRANSFORM_RANK_ENGINE=python` only for reference/regression
  comparisons against the older Python implementation. Stage C transform
  confirmation supports per-candidate seed offsets through the same batch API.
  The Rust module also owns pure-transposition scoring:
  `pure_transposition_score_batch` applies provenance-bearing transform
  pipelines and scores the resulting A-Z text directly. The automated
  pure-transposition route now uses `analysis.pure_transposition` to generate
  a broad screen of grid/route/columnar families, direct `MatrixRotate`
  candidates, rail/fence-style routes, route+orientation-repair composites,
  shifted/offset route reads, bounded mask/grille-like routes, and K3-style
  `TransMatrix` candidates. Python handles candidate
  generation/orchestration/artifacts for this path, including branch-side
  `MatrixRotate`/`TransMatrix` application and a small identical-call
  pure-transposition screen cache. `PureTranspositionSearchConfig` is the
  shared candidate-plan object for this path, and artifacts/tool results
  record its `candidate_plan` metadata. Keep older Python implementations as
  reference/regression paths, not as feature-parity obligations for new
  Rust-scale search work.
- Transform finalist menus now share a Python-side evaluation skeleton in
  `analysis.transform_evaluation`. Pure-transposition direct-score candidates
  and transform+homophonic solver finalists both flow through
  `evaluate_finalist_menu`: attach plaintext validation, sort finalists,
  optionally run confirmation callbacks, apply labels/gates, final-sort,
  select, diagnose, and return a normalized artifact shape. The expensive
  probe engines still differ by cipher family, but downstream finalist
  reporting and agent review now use the same menu structure. Rust
  transform+homophonic rank/confirmation payload construction is factored
  into `analysis.transform_homophonic_batch`, along with rank-row and
  confirmation-record shaping, so `automated.runner` no longer owns candidate
  dedupe, rank-batch row normalization, confirmation batch-id/seed-offset
  mechanics, or the low-level Rust batch artifact row formats. The runner
  still supplies the actual quality/selection scoring callbacks. Rank probes
  now call one helper for dedupe, request construction, Rust execution, and
  row normalization. The same helper also owns the common
  `ZenithTransformBatchContext` / `ZenithTransformBatchRequest` shapes and
  Rust batch call wrapper; the runner still chooses model path, plaintext
  alphabet, budget policy, and candidate scoring policy.

Kryptos status:
- K1/K2/K3 are imported in `../cipher_benchmark` as solved calibration records.
- K2 can recover `ABSCISSA` and plaintext when `KRYPTOS` is supplied as a
  candidate tableau keyword.
- K3 is now a solved pure-transposition fixture
  (`kryptos_k3_transmatrix`). The automated-only Decipher route detects
  pure transposition/TransMatrix metadata and runs the broad Rust-scored
  pure-transposition screen. Latest validation searched 165,291 candidates
  (grid/route/columnar plus TransMatrix), selected a TransMatrix equivalent
  (`w1=4`, `w2=48`, `cw`), and recovered the local K3 fixture at 100%
  char/word accuracy in the normal benchmark harness.
- A synthetic pure-transposition ladder now lives at
  `frontier/pure_transposition_ladder.jsonl`. It uses the `transposition_only`
  testgen mode to create no-substitution route/order cases for MatrixRotate,
  diagonal route reads, split-grid routes, spiral/rail/boustrophedon routes,
  route+repair composites including route+rotate+reverse, shifted spiral
  routes, border/checkerboard mask routes, repeated-block turning-mask routes,
  turning-mask routes with shifted starting orientations, block-level route
  transpositions with optional local block reversal and cyclic block-order
  offsets,
  short/long no-boundary calibration rows, and non-Kryptos TransMatrix.
- Agentic runs can invoke that same screen through
  `search_pure_transposition`. Pure-transposition searches now create
  `search_session_id` menus analogous to transform+homophonic searches:
  `search_review_pure_transposition_finalists` pages through previews,
  `act_rate_transform_finalist` records the agent's contextual readability
  score, and `act_install_pure_transposition_finalists` installs selected
  ranks as readable transform branches with decoded-text metadata. Pure
  finalist menus now include `analysis.finalist_validation` evidence:
  strict continuous word hits, segmentation cost/rate, pseudo-word burden,
  edge cleanliness for beginnings/endings, n-gram scores, and a
  `validated_selection_score` rerank. Mixed
  transform+homophonic finalist previews carry the same validation block as
  supporting evidence.
- K1/K2 can now be represented explicitly as Quagmire III cases for
  known-parameter replay; unknown Quagmire search is still pending.
- The first Quagmire search scaffold can recover K2's `ABSCISSA` cycleword
  when `KRYPTOS` is present as an initial keyword-shaped alphabet. It does not
  yet recover `KRYPTOS` blindly.
- Dictionary-derived starts work on non-solution-bearing synthetic Quagmire III
  cases with ordinary keywords, but the built-in English dictionary does not
  contain `KRYPTOS`; context-derived or custom keyword sources are still needed
  for that family.
- K1 remains a short-text stress case; current scorer can prefer false
  English-ish basins without crib/context-aware scoring.

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
- **Agent tool exposure**: `search_automated_solver` gives the agent access to the
  modern no-LLM solver stack, including the default `zenith_native` homophonic
  route when the selected language/model supports it. `search_homophonic_anneal`
  remains available for focused legacy/profile experiments.
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
  `models/ngram5_la_500.bin` (a `max_books=500` Gutenberg run that currently
  resolves to all 101 catalog-tagged Latin texts) available locally for
  focused experiments.
- On the focused Borg `parity_borg_latin_borg_0109v` probe, forcing
  `zenith_native` with the Latin binary model reaches a reproducible
  no-boundary partial-Latin basin at about `13.9%` character accuracy with the
  full 8-seed budget, but still `0.0%` word accuracy.
- The larger "all available Gutenberg Latin" model slightly improves the
  anneal score on that case versus the 100-book model but does not yet move
  headline accuracy.
- The shared no-boundary repair helper is now available to both automated and
  agentic paths. On a focused forced-`zenith_native` probe of `0109v` using
  `models/ngram5_la_500.bin`, the repair pass now applies a few conservative
  local edits and lifts the result to about `13.9%` character accuracy and
  `3.9%` word accuracy. That is a modest but real gain, though the repaired
  text is still only a text-level preview and is not yet key-consistent.

### 4. 🎭 **Historical Copiale/Borg generalization**
Synthetic tests are useful for controlled iteration, but the historical benchmark still needs broader runs to separate synthetic overfitting from durable cryptanalytic progress. The first correctness pass is now in place: historical automated runs use benchmark language metadata instead of English fallback, and routing no longer assumes that word boundaries imply a simple bijective substitution.

Additional current read:
- The Agent Loop Redesign plan is now considered complete through Milestone 4
  smoke coverage. New Copiale/German and broader generalization work should
  start from `docs/copiale_generalization_plan.md`.
- Agent-loop Milestone 4 now has broader Borg evidence:
  - Borg `0140v`
    (`artifacts/borg_single_B_borg_0140v/47df72a4da8b.json`) improved from
    weak automated preflight (`36.9%` char / `0.0%` word) to a readable agent
    branch (`85.5%` char / `54.8%` word).
  - Borg `0077v`
    (`artifacts/parity_borg_latin_borg_0077v/c9d17916d17f.json`) improved
    from weak `zenith_native` preflight (`37.2%` char / `2.8%` word) to a
    readable partial agent branch (`84.1%` char / `53.5%` word).
  - Borg `0171v`
    (`artifacts/borg_single_B_borg_0171v/a43a53111e26.json`) exposed a
    do-no-harm failure: automated preflight was already strong (`90.9%` char /
    `72.7%` word), but the agent declared a more classicized repair branch at
    `85.8%` char / `50.8%` word. The prompt, preflight context, and branch
    cards now tell the agent to treat `automated_preflight` as a protected
    no-LLM baseline and avoid broad manuscript-orthography drift.
- Copiale `p068`
  (`artifacts/copiale_single_B_copiale_p068/7d795a0ae0a9.json`) did not
  improve over preflight (`45.3%` char / `0.0%` word). The agent found
  German-looking islands but not coherent sentence-level German. Treat
  Copiale/German as a separate capability track: stronger German models,
  context-aware modes, nomenclator/codeword behavior, and stricter declaration
  discipline are needed before comparing it to Borg progress.
- A no-LLM automated baseline smoke packet for these Milestone 4 cases now
  lives at `frontier/agentic_milestone4_smoke.jsonl`. The opt-in pytest
  command `DECIPHER_RUN_MILESTONE4_SMOKE=1 .venv/bin/python -m pytest
  tests/test_milestone4_smoke.py` runs `AutomatedBenchmarkRunner` across the
  packet, verifies zero LLM tokens/cost, and checks baseline thresholds.
- The same smoke test file also includes fake-provider LLM-agent coverage.
  These tests do not call a live API; they check that the loop can inspect and
  declare a protected `automated_preflight` branch, and can make a small
  reading-driven `act_set_mapping` repair before declaration.
- Borg `0077v` routes to `zenith_native` because its symbol inventory exceeds
  the Latin plaintext alphabet size.
- Borg `0109v` still routes to the substitution path by default because its
  symbol inventory does not trip the current overcomplete/homophonic heuristic.
- Focused probes show that forcing `zenith_native` on `0109v` produces a more
  fluent no-boundary Latin stream than the substitution path, but not a clear
  win yet. The new shared repair helper improves that stream a little, which
  strengthens the case that this is a cleanup/segmentation problem at least as
  much as a raw search problem.
- Focused agentic `0109v` work now has a better boundary-edit path too:
  - diagnosis returns `boundary_candidates` plus `recommended_next_tool`
  - prompt emphasis alone was not enough to make the agent use boundary tools
  - the one-step wrapper `act_apply_boundary_candidate` did improve compliance
  - best recent agentic `0109v` run reached about `14.2%` char / `8.9%` word
  - working lesson: for agentic parity, a low-friction actuator can matter
    more than an increasingly forceful recommendation
  - Important current repair limitation on the best recent `0109v` text:
    - the shared diagnosis/repair path only auto-proposes one-letter
      substitutions that land on words already present in the loaded Latin
      dictionary
    - `RLURES -> PLURES` is detectable under that rule, but weak
    - `TREUITER`, `SIMULITER`, and `MORIETANTUR` are not currently cleaned by
      the automatic path because they are not all reachable as one-edit fixes
      to dictionary words in the shipped Latin word list
    - so the remaining miss is partly an agent-policy issue and partly a real
      capability ceiling in the underlying repair primitive
    - Recent `8f2993dc03a6` follow-up clarified a second repair primitive
      problem: direct word repair must respect repeated cipher symbols inside
      the same word. A target like `RLURES -> PLURES` is not a safe one-letter
      repair when the cipher symbol producing the first `R` also produces a
      later `R`; forcing it yields `PLUPES`-style collateral damage. The tool
      surface now has a read-only `decode_plan_word_repair_menu` for comparing
      candidate readings and flagging these conflicts before mutation.
  - Planned next split for Borg/Latin follow-up:
    - one branch should focus on automated repair capability:
      stronger Latin dictionary coverage, richer cleanup primitives, and
      broader guarded repair logic
    - a separate branch should focus on agent behavior:
      the agent should be encouraged to use its own reading of partial Latin
      to hypothesize spelling and boundary repairs, with tools acting as
      helpers and validators rather than the outer limit of what it may infer
  - English analog fixture for agent-autonomy debugging:
    - `fixtures/benchmarks/english_borg_analog` contains one hand-shaped
      English simple-substitution case with archaic vocabulary and deliberately
      misleading cipher word boundaries (`THERE | FORE`, `PHYSICK | ER`,
      `UN | TO`, `WITH | OUT`, etc.)
    - This lets us inspect the same "locally readable but word-alignment
      scoring is poor" failure mode without needing to read Latin.
    - The source-boundary rendering is character-perfect against the ground
      truth but has very low word accuracy, by design.
    - Milestone result: after adding whole-stream reading validation and
      resegmentation tools (`decode_validate_reading_repair` and
      `act_resegment_by_reading`), the agent solved this analog at
      `100%` character / `100%` word accuracy. The key behavioral change was
      letting the agent state a complete best reading and have tooling
      validate/apply boundaries, instead of forcing many fragile local
      split/merge edits. Reference artifact:
      `artifacts/english_borg_analog_001/919b40a14b6f.json`.
  - Immediate Borg transfer result:
    - Focused `parity_borg_latin_borg_0109v` rerun with the same tool surface
      reached `13.9%` character / `20.8%` word accuracy on the declared
      `repair` branch, versus automated preflight `14.2%` character / `6.4%`
      word accuracy. Reference artifact:
      `artifacts/parity_borg_latin_borg_0109v/b15432813fde.json`.
    - This is a meaningful word-alignment/reading gain, not a full solve.
      The agent made targeted reading-driven mappings (`F -> Q`, `P -> B`)
      and some local boundary edits, but did not yet use
      `act_resegment_by_reading` as a whole-text Latin operation. The artifact
      analyzer still reports `unattempted_reading_fix` warnings, especially
      around `RLURES -> PLURES`-style repairs.
    - Follow-up tool improvement: `decode_validate_reading_repair` now returns
      a same-length `boundary_projection` for proposed readings that change
      letters, and `act_resegment_from_reading_repair` applies that projected
      word-boundary pattern without changing the key or decoded letters. This
      is meant for the hard middle case where the agent can state a plausible
      full Latin reading, but some spans still require later cipher-symbol
      repair.
    - `meta_declare_solution` now has a pre-final guard for this workflow: if
      the rationale still mentions boundary/alignment issues and no full-
      reading validation/resegmentation tool has been used on the branch, the
      declaration is blocked with instructions to run the validation/projection
      workflow first. Final-turn declarations remain accepted to avoid losing
      a partial solution.
    - The loop now also emits a penultimate-turn warning that explicitly
      prioritizes `decode_validate_reading_repair` and the resegmentation
      tools. This is necessary because the final turn is reserved for
      declaration; the full-reading workflow has to happen one turn earlier.
    - Late-window tool gating now hides and executor-blocks local edit/search
      tools. This did cause the agent to attempt the full-reading workflow on
      `0109v`, but the proposed readings were not length-preserving
      (`345` current chars vs `307`/`378` proposed chars), so no projection
      was applied. Latest gated-run artifact:
      `artifacts/parity_borg_latin_borg_0109v/8776c6231842.json`
      (`13.9%` char / `8.9%` word).
    - Current conclusion: pause Borg-specific prompt/tool work until Decipher
      has a more modern agentic loop with inner tool steps, same-iteration
      gated-tool retry, explicit workflow state, and provider-neutral model
      adapters. Detailed design note:
      `docs/agent_loop_redesign_plan.md`.
    - Current active nudge: once a branch is readable enough to notice word
      fragments or boundary drift, the agent should use
      `act_resegment_window_by_reading` for local phrases or the full-reading
      projection tools for global drift before declaring. Merely describing
      the boundary problem is now treated as unfinished work.
    - Latest loop/tool refinement: boundary-projection count failures now get
      multiple same-iteration retries, and late reading repair has a bounded
      low-cost sandbox. The sandbox lets the agent run small repair/resegment
      experiments, inspect branch cards, and declare without consuming extra
      outer benchmark iterations. `act_resegment_window_by_reading` also
      suggests nearby compatible windows when a proposed local reading has the
      wrong character count, which helps recover from stale word indices after
      earlier boundary edits.

### 6. 📖 **Reading-driven repair discipline (open)**

Once a branch decodes into recognisable target-language text, the agent's
reading must dominate over scoring signals — not the other way around. The
current v2 prompt already says this but does not back it up consistently in
either prompt structure or tool-output design, and recent Borg artifacts show
the agent identifying correct cipher-symbol fixes, applying them, and then
reverting because a tool-reported `verdict: worse` (driven by a `dict_rate`
delta) contradicted its own reading.

This is a general agentic-decipherment problem and should not be framed in
terms of any single language or manuscript. The shape of the failure is:

1. Agent identifies a fix from reading (`cipher word XYZ decodes as FOO but
   should be BAR`).
2. Agent applies `act_set_mapping(cipher_symbol=…, plain_letter=…)` — the
   correct primitive for cipher-symbol-level repair.
3. The change makes 2+ previously-broken decoded words read as real
   target-language words, but happens to drop one or two short accidentally-
   in-dictionary fragments.
4. Tool returns `verdict: worse` based on the score delta.
5. Agent defers to the verdict, reverts, and either declares the unfixed
   branch or burns iterations re-investigating what it already knew.

Two structural reasons the score conflicts with the reading on
boundary-preserved ciphers in particular:

- The dictionary scorer counts whole-word hits. When the cipher's `|`
  boundaries don't match target-language word boundaries (so single
  target-language words are split across multiple cipher words, or vice
  versa), short fragments can be accidentally valid dictionary words. Fixing
  one cipher symbol can replace four wrong fragments with four correct
  fragments **and** lose a couple of short accidental hits — net dict_rate
  goes down even though the underlying decryption improved.
- Per-token quad-gram score is similarly insensitive to single-letter cipher-
  symbol moves at a fixed key-mass; it can move slightly the wrong direction
  on a correct fix without that fix being wrong.

The remediation is a mix of prompt discipline and tool-output discipline. See
`TODO.md` ("Priority 4 → Reading-driven repair discipline") for the actionable
items. Headline points:

- The prompt must establish an explicit hierarchy: **when the agent can read
  coherent target-language words, its reading is authoritative and a
  `verdict: worse` from a single score delta does not override it.** Scores
  are decision support, not adjudication.
- Worked examples in the prompt must lead with the cipher-symbol mental model
  (`cipher symbol Sxx currently → A, change to B`), not the decoded-letter
  mental model (`decoded T should be B`). The decoded-letter framing primes
  `act_swap_decoded` (a bidirectional tool that is a footgun for
  reading-driven fixes) when the correct primitive is `act_set_mapping`.
- `act_swap_decoded` should be marked clearly in the toolkit as a
  bidirectional-letter operation, not a fix-this-word operation. For
  reading-driven repairs, `act_set_mapping` is always the right primitive.
- Tool surfaces returning a verdict (`improved` / `worse`) on individual
  mapping changes should report deltas as data, not as a quality judgement,
  and should include a was→now sample of changed words so the agent reads
  rather than scores the change.
- "Boundary edit recommended" suggestions must not outrank candidate
  letter-level repairs in `recommended_next_tool`. On boundary-preserved
  ciphers with letter-level errors, boundary edits are at best a 2–5%
  cleanup; letter-level fixes are 10–20%.
- The "anchored polish" rule must require at least one applied
  `act_set_mapping` (or pre-existing anchored mapping) before
  `search_anneal(preserve_existing=true)` is called; otherwise the polish has
  nothing meaningful to anchor and just re-confirms the existing local
  optimum.

A regression test for this discipline is the existing focused Borg
follow-up: an agent that respects its own reading should not declare a
reading-driven fix as `worse` when it produces additional readable words. The
artifact gap analyzer should grow a `score_overrode_reading` label so this
failure mode is detectable across runs.

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
2. **Rich tool set** — 49 tools across 10 namespaces (workspace, observe, decode, score, corpus, act, search, repair_agenda, run_python, meta)
3. **Agent freedom** — No phases, agent plans own strategy
4. **Hypothesis tracking** — Branching workspace preserves exploration history

### Tool Arsenal (49 tools implemented)
✅ **workspace_* (6 tools)** — fork, list, branch_cards, delete, compare, merge
✅ **observe_* (4 tools)** — frequency, isomorph_clusters, ic, homophone_distribution
✅ **decode_* (12 tools)** — show, unmapped, heatmap, letter_stats, ambiguous_letter, absent_letter_candidates, diagnose, diagnose_and_fix, repair_no_boundary, validate_reading_repair, plan_word_repair, plan_word_repair_menu
✅ **score_* (3 tools)** — panel, quadgram, dictionary
✅ **corpus_* (2 tools)** — lookup_word, word_candidates
✅ **act_* (13 tools)** — set_mapping, bulk_set, anchor_word, clear, swap_decoded, split/merge words, decoded-word merge, boundary candidate, word repair, and reading resegmentation tools
✅ **search_* (4 tools)** — hill_climb, anneal, automated_solver, homophonic_anneal
✅ **repair_agenda_* (2 tools)** — list, update
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
scripts/setup_dev.sh         # pip install -e . plus required Rust fast kernels
pip install -e '.[providers]' # Optional: OpenAI/Gemini agent providers
```

Python 3.11 at `/opt/homebrew/bin/python3.11`. Venv at `.venv/`.

---

## Model Selection

**Recommended for V2**: `claude-sonnet-4-6` — Best results on historical manuscript analysis
**Default**: `claude-opus-4-7` — Configurable via `--model` flag

### Model Notes
- **Claude Sonnet 4.6**: Strong performance on S-token sequences and Latin/German manuscript analysis
- **Claude Opus 4.7**: More conservative with historical encoded text; use Sonnet 4.6 for decipherment
- **OpenAI/Gemini**: Agentic runs now support `--provider openai` and
  `--provider gemini` through the provider adapter. Use the smoke packet before
  treating any non-Claude model as parity-quality for historical decipherment.
- **Preprocessing**: S-token normalization (letter substitution) improves API compatibility across models

### Configuration
Models are configurable via `--provider` and `--model` CLI flags. If
`--provider` is omitted, Decipher infers it from common model prefixes
(`claude-*`, `gpt-*`, `gemini-*`) and otherwise defaults to Anthropic.

API keys may be supplied by environment variable, gitignored local files, or
macOS Keychain:
- Anthropic: `ANTHROPIC_API_KEY`, `.decipher_keys/anthropic_api_key`,
  keychain account `anthropic_api_key`
- OpenAI: `OPENAI_API_KEY`, `.decipher_keys/openai_api_key`, keychain account
  `openai_api_key`
- Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`,
  `.decipher_keys/gemini_api_key`, keychain account `gemini_api_key`

The repo gitignores `.env`, `.env.*`, and `.decipher_keys/`.

### Performance
Sonnet 4.6 on `synth_en_250nb_s4`: exact match in 7 iterations after reliability and segmentation fixes.
`synth_en_200honb_s6` is the active hardest homophonic/no-boundary stress test.
