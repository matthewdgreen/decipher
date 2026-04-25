# Decipher TODO

## Agentic Parity Program

Goal: the agentic solver should have at least the same practical solving
capability as the best available non-agentic solvers on clean benchmark tasks,
and failures should identify a missing tool, a weak tool, a wrong agent choice,
or a benchmark data issue.

### Priority 1: Benchmark Hygiene

- [x] Add a benchmark validator for `../cipher_benchmark/benchmark`.
  - Validate manifest/schema compatibility.
  - Validate referenced files.
  - Validate split references.
  - Validate track-specific required layers.
  - Summarize source/track/status/language counts.
- [x] Resolve benchmark schema drift.
  - Track-B-only synthetic records may omit images.
  - `word_boundaries`, `token_count`, `word_count`, and `notes` are valid record fields.
  - Synthetic records need explicit `rights_class` and `source_record_id`, or schema rules must allow generated records.
- [x] Fix or explicitly mark missing benchmark assets.
  - Known current issue: `sources/decode_gallica/images/decode_2732_f403.jpg`.
- [x] Refresh benchmark docs and Decipher docs with current source counts.

### Priority 2: Parity Coverage

- [x] Import external-tool bundled ciphers as a separate benchmark source.
  - Zenith: Zodiac 408, Goldbug, Horace Mann, Kryptos sections, Hampton/J. Hampton as appropriate.
  - zkdecrypto-lite: Zodiac files and classical test ciphers such as Playfair, Bifid, Trifid, ADFGX, ADFGVX, columnar, running key.
  - Keep provenance/license notes clear and do not mix these with historical manuscript records.
- [x] Add parity split files.
  - `parity_homophonic_en.jsonl`
  - `parity_simple_substitution_multilang.jsonl`
  - `parity_borg_latin.jsonl`
  - `parity_copiale_german.jsonl`
  - `parity_tool_builtins.jsonl`
  - `parity_zodiac.jsonl`
- [x] Add optional parity metadata to split/test definitions.
  - `parity_family`
  - `recommended_agent_tool`
  - `baseline_solvers`
  - `expected_baseline_status`
  - `expected_min_char_accuracy`
  - `known_cipher_type`
  - `word_boundaries`
- [x] Teach Decipher's benchmark loader to load `context_records` separately from scored targets.
- [ ] Formalize parity evaluation modes.
  - `blind`: ciphertext-derived routing only; no benchmark metadata passed to any solver.
  - `context-aware`: allow benign context such as language/source family, but record exactly which solvers actually consumed it.
  - Default comparative parity reporting should make the evaluation mode explicit in summaries, dashboards, and artifacts.
  - Add a per-solver `context_capabilities` / `context_used` record so comparisons stay honest when wrappers differ.

### Priority 3: Native Tool Parity

- [ ] Harden `search_homophonic_anneal` with seed sweeps against Zenith and zkdecrypto-lite.
  - [x] Add low-diversity/collapse quality gates for automated homophonic output.
  - [x] Retry automated homophonic runs across multiple seeds when quality gates fail.
  - [x] Select homophonic candidates by anneal score adjusted for plaintext quality, not raw n-gram score alone.
  - [x] Add a diversity objective to homophonic annealing and expose it in the agent tool.
  - [ ] Close the remaining Zodiac-class gap where native search can produce readable but sub-par candidates.
  - Current April 2026 state:
    - Broad automated parity is now strong on synthetic/simple families (`simple-wb`, `simple-nb`, `preset:*`, and most `homophonic-nb` rows are near or at parity).
    - The clearest remaining native-search quality gap is still `parity_tool_zenith_zodiac408`.
    - Historical `borg`/`copiale` rows from older matrices should be treated cautiously until rerun under the newer language-propagation and routing fixes.
  - Scorer-ablation findings recorded so far:
    - `balanced` remains the strongest tested objective on Zodiac (`65.7%`, full budget) even though it still converges to a readable-but-wrong basin.
    - `ioc_ngram` performed much worse on Zodiac (`34.1%`, full budget), frequently collapsing toward high-IoC low-diversity garbage such as near-all-`E` states.
    - `ngram_distribution` also underperformed badly on Zodiac (`30.1%`, screen budget), producing over-compressed repetitive text.
    - Working conclusion: the distribution/diversity terms are protective guardrails, not the main cause of the Zodiac gap. The next likely gains are structural: staged search, better move sets, or stronger candidate aggregation/reranking.
  - Zenith-audit findings recorded so far:
    - Current Zenith parity appears fair: our wrapper submits raw ciphertext/config through the GraphQL solve path rather than invoking a built-in named Zodiac shortcut.
    - Zenith's bundled `zodiac408.json` includes a `knownSolutionKey`, but the API solve path reconstructs a fresh cipher from request input and does not populate that key into the optimization run.
    - The `knownSolutionKey` appears to be used for evaluation/proximity reporting, not as a hidden search oracle.
    - Working conclusion: Zenith's advantage on Zodiac is unlikely to be a fairness bug. The more plausible causes are stronger 5-gram data, sparse/stride-2 incremental scoring, tighter array-based implementation, and higher effective proposal throughput.
  - Experiment harness status:
    - [x] Add homophonic score-profile ablation support (`balanced`, `ngram_only`, `ngram_distribution`, `ioc_ngram`) with per-seed diagnostics in automated artifacts.
    - [x] Add a fixed homophonic ablation packet in `frontier/homophonic_profile_ablation.jsonl`.
    - [x] Add a `--homophonic-budget {full,screen}` flag so comparative experiments can run quickly without changing default solver behavior.
    - [ ] Try opt-in two-stage homophonic refinement before broader candidate aggregation/reranking.
  - Additional Zodiac experiments after the first ablation packet:
    - [x] Add experimental score profile `zenith_like`.
    - [x] Add experimental score profile `zenith_exact`.
    - [x] Add experimental selection profile `pool_rerank_v1`.
    - [x] Add experimental move profiles `mixed_v1`, `mixed_v2`, and `mixed_v1_targeted`.
    - [x] Add move-telemetry instrumentation to homophonic annealing and thread it into automated artifacts.
    - [x] Add targeted local refinement mode `targeted_repair`.
  - Zodiac experiment findings after those changes:
    - `pool_rerank_v1` alone did not improve over baseline `balanced`; it selected the same wrong basin (`65.7%`), suggesting the missing solution was not present in the candidate pool.
    - `zenith_like` scoring was actively harmful on Zodiac (`14.2%` with `pool_rerank_v1`), producing pseudo-English soup rather than a stronger basin.
    - `zenith_exact` plus `single_symbol` was also strongly harmful on Zodiac (`7.8%`, full budget). This is useful negative evidence: the remaining gap is not just "copy Zenith's score formula."
    - `mixed_v1` was the first meaningful search-neighborhood improvement, lifting Zodiac from the long-standing `65.7%` plateau to `83.6%`.
    - `mixed_v2` regressed to `74.3%`; telemetry suggests the problem is proposal mix quality, not simply that larger moves are too expensive.
    - `mixed_v1_targeted` matched `mixed_v1` at `83.6%` but ran slower, so the current targeting heuristic is not yet paying for itself.
    - `two_stage` refinement adopted small score improvements but did not materially improve final accuracy, including when layered on top of `mixed_v1`.
    - `targeted_repair` also failed to improve the `mixed_v1` Zodiac result: it completed at the same `83.6%` accuracy. The repair step ran, targeted 8/12-symbol subsets, accepted local moves, but returned the same rounded selection score and unchanged plaintext preview.
    - `zenith_exact + mixed_v1` was even worse (`2.5%`, full budget). Together with `zenith_exact + single_symbol`, this is now strong negative evidence against simple Zenith-score portability.
    - `family_repair` now has family diagnostics, family-competition proposals, two-symbol split branches, multi-elite repair, and epsilon-gated adoption. It is useful for diagnosis, but repeated runs still return the same `83.6%` Zodiac basin.
    - `DECIPHER_HOMOPHONIC_EARLY_STOP=1` is implemented and honest, but it has not yet fired on Zodiac because the losing seeds are still coherent enough under the current thresholds.
    - `DECIPHER_HOMOPHONIC_REPAIR_PROFILE=dev` successfully reduced repair breadth, but Zodiac runtime stayed high because the main search, not repair, remained the dominant cost center.
    - `DECIPHER_HOMOPHONIC_SEARCH_PROFILE=dev` is now implemented to shrink the broad search during development runs while leaving `full` as the benchmark default.
    - Working conclusion: the main recent gain came from richer move proposals (`mixed_v1`), not scorer mimicry or late reranking/polish. Exact Zenith-style score semantics do not transfer cleanly into the current Decipher search loop by themselves.
  - Current best-known Zodiac state:
    - Best current automated stack is `balanced` + `anneal_quality` + `mixed_v1`.
    - Best observed Zodiac result so far is `83.6%` character accuracy, still materially below Zenith's near-perfect solve.
    - The best decoded text is readable but contains systematic family confusions rather than random noise.
    - Most common residual confusions in the best `mixed_v1` output were `L -> T`, `L -> S`, `L -> N`, `B -> H`, `W -> T`, `K -> D`, and `F -> S`, which points toward unresolved homophone-cluster/letter-family errors.
    - The first `targeted_repair` heuristic only hit part of the true error structure: it included one `L -> S` symbol and the `F -> S` symbol, but missed other high-error families such as `L -> T`, `L -> N`, `B -> H`, `W -> T`, and `K -> D`. This suggests the suspicious-symbol selector is too score-local and not sufficiently driven by global homophone-family inconsistency.
  - Next informed Zodiac steps:
    - [x] Evaluate `targeted_repair` on Zodiac after `mixed_v1` search to see whether focused local repair can resolve the remaining letter-family confusions.
    - [x] Evaluate exact Zenith-style scoring with `single_symbol` search as a portability check.
    - [x] Add broader candidate-pool persistence and structured local repair over top candidate families rather than just the selected winner.
    - [x] Evaluate `zenith_exact + mixed_v1` before deciding whether exact score semantics are only harmful or merely dependent on a stronger move neighborhood.
    - [x] Replace the first suspicious-symbol heuristic with a family-consistency repair heuristic: identify plaintext letters with too many/few homophones, symbols whose decoded local contexts are inconsistent with their assigned plaintext family, and high-confusion clusters such as the current `L` family.
    - [x] Add development-vs-confirmation runtime controls so iteration work can run faster without changing headline parity defaults.
    - [ ] Audit the best new dev-profile runs on Zodiac to see whether they preserve the same dominant basins and ranking order as full runs.
    - [ ] Add a cheap pre-screen for repair branches/elites so obviously weak repair attempts are discarded before full local annealing.
    - [ ] Tighten early-stop thresholds using the latest Zodiac seed telemetry so clearly dominated coherent seeds can stop earlier without suppressing promising near-miss basins.
    - [ ] Explore multi-basin search improvements rather than more local repair variants on the same winner basin; current evidence says we repeatedly rediscover the same `83.6%` structure.
- [x] Add top-N candidate support for homophonic annealing.
- [x] Add an automated-only/no-LLM CLI mode that runs native techniques and writes zero-cost artifacts.
- [x] Run no-LLM automated preflight before LLM iteration 1 by default, with a branch and prompt summary.
- [x] Upgrade English simple-substitution automated solving with bijective continuous n-gram annealing.
- [x] Add a chunkable automated-only parity matrix runner for seed, length, family, benchmark split, and external-tool comparisons.
- [ ] Add model provenance and acquisition docs for continuous n-gram models.
  - Current high-quality continuous model support relies on a local Zenith English `zenith-model.csv` under `other_tools/`, which is git-ignored and not redistributed with Decipher.
  - Document that the Zenith model is optional but strongly recommended for English homophonic parity, and that fallback word-list models are weaker.
  - Record expected model path(s), version/source, order, row count, checksum, and artifact metadata.
  - Update the Zenith-model redistribution note: BNC no longer appears to be the blocking issue, but Zenith's documented use of the Blog Authorship Corpus still leaves redistribution legally unresolved for Decipher.
  - Resolve redistribution status before bundling any Zenith model files: confirm whether `zenith-model.csv`/`.array.bin` are covered by Zenith GPLv3 and whether the Blog Authorship Corpus training component permits redistributing a derived commercial/open-source model.
  - Decide whether Decipher should bundle models, download them, or require users to provide them locally.
- [ ] Add a model registry/config layer for continuous n-gram models.
  - Support named models such as `en_zenith_5gram`.
  - Track language, order, source, license/provenance, path, checksum, row count, and redistribution status.
  - Ensure run artifacts record exactly which model was used.
- [ ] Add or train language-specific continuous models for Latin, German, French, and Italian.
  - Identify corpus sources and licenses for each language.
  - Normalize corpora consistently with benchmark plaintext rules.
  - Produce reproducible build configs and metadata sidecars.
  - Compare trained models against current word-list fallbacks.
  - Current Latin status:
    - [x] Build `models/ngram5_la.bin` from 100 Gutenberg books.
    - [x] Build a larger probe model `models/ngram5_la_500.bin` from a `max_books=500` Gutenberg run; this currently resolves to all 101 catalog-tagged Latin texts.
    - [ ] Compare the larger all-available-Gutenberg Latin model against the default Latin model on a small Borg packet, not just `0109v`.
    - Finding so far: on `parity_borg_latin_borg_0109v`, the larger all-available-Gutenberg Latin model slightly improves `zenith_native` anneal score but does not improve headline char/word accuracy over the 100-book Latin model.
- [ ] Add Decipher-native tooling to train/export continuous n-gram models.
  - Input: plain-text corpus directory.
  - Output: supported CSV or compact binary format plus metadata sidecar.
  - Include tests for loader compatibility and scoring reproducibility.
- [ ] Benchmark simple substitution with and without word boundaries across English/French/German/Italian.
- [ ] Assess Borg and Copiale failures by tool gap: language model, context loading, nomenclator/codeword behavior, historical spelling, or prompt choice.
- [ ] Borg focused follow-up: `parity_borg_latin_borg_0109v`.
  - Current state:
    - Default routing sends `0109v` down the substitution path because its symbol inventory does not exceed the Latin plaintext alphabet size.
    - Forcing `zenith_native` with the full 8-seed budget is reproducible and reaches a distinct no-boundary Latin basin at about `13.9%` character accuracy and `0.0%` word accuracy.
    - With only one seed, the forced `zenith_native` basin is much worse (~`9.0%`), so this case is notably multi-seed-sensitive.
    - The substitution path lands in a different partial-Latin basin with similar character accuracy but slightly better word-level structure.
    - The new shared no-boundary repair helper now gives a small lift on the forced `zenith_native` path with `models/ngram5_la_500.bin` (about `13.9%` char / `3.9%` word), but the repaired plaintext is still only a text-level preview and is not yet key-consistent.
    - Agentic branch-local boundary editing is now available through workspace word-boundary overlays plus:
      - `act_split_cipher_word`
      - `act_merge_cipher_words`
      - `act_apply_boundary_candidate`
    - `decode_diagnose` and `decode_diagnose_and_fix` now surface `boundary_candidates` and a `recommended_next_tool`.
    - Important tool-design finding: recommendation text alone was not enough to make the agent use split/merge; a one-shot wrapper tool (`act_apply_boundary_candidate`) materially improved compliance.
    - Best recent agentic `0109v` run after that wrapper change reached about `14.2%` character accuracy and `8.9%` word accuracy, modestly improving over earlier `7.7%` word-accuracy runs while still staying in the same broadly partial-Latin basin.
  - Next likely useful experiments:
    - [ ] Try a stronger Latin cleanup pass on `zenith_native` output: segmentation plus phrase-level split/merge and repair, beyond the current conservative shared helper.
    - [ ] Re-run the improved agentic boundary-tool path on a second small Borg case to see whether the `act_apply_boundary_candidate` gain generalizes beyond `0109v`.
    - [ ] Compare default substitution vs forced `zenith_native` on a second small Borg subset to see whether `0109v` is representative or a special case.
    - [ ] Revisit routing later if a cleanup step starts making the `zenith_native` basin consistently more useful.
- [x] Fix automated historical language propagation for benchmark-backed parity/frontier runs.
  - Benchmark target-record `plaintext_language` now overrides brittle `test_id` prefix heuristics.
  - This closes the accidental `parity_borg_*`/`parity_copiale_*` → English fallback in automated-only and automated preflight paths.
- [x] Tighten automated routing so overcomplete symbol alphabets and known homophonic benchmark families use non-bijective search instead of erroring immediately.
  - Route by ciphertext evidence first (alphabet size vs plaintext alphabet size), with benchmark `cipher_system` as a compatibility hint.
  - Keep this distinct from ground-truth leakage: do not pass solution-derived structure.
- [x] Stop labeling any non-empty automated output as `solved`.
  - Automated-only status is now `completed` on successful local runs, aligned with external baseline reporting.
  - Preserve pass/fail evaluation through accuracy thresholds instead of status inflation.
- [x] Reclassify frontier cases that are no longer genuine Decipher bad-result failures.
  - `synth_en_150honb_s3` moved out of `bad_result`; it is now a known-good/cross-tool divergence case.
- [ ] Improve automated homophonic candidate selection beyond collapse checks.
  - Current Zodiac-class gap is mostly "readable but wrong" output, not low-diversity collapse.
  - Next step: add segmentation/dictionary/plausibility reranking or local repair across top-N candidates.
  - Defer full reranking work until after the first two-stage refinement experiment, so we can separate "better search trajectory" from "better final pick".
  - Update: the first reranking and repair experiments are now in place, and the more urgent remaining work is multi-basin search quality plus cheap triage of weak seeds/repair branches.
- [ ] Assess external-tool cipher families and explicitly mark unsupported families rather than letting the agent fake confidence.
- [ ] Verify external solver context capabilities rather than assuming wrappers are complete.
  - `zkdecrypto-lite`: current source/build appears English-only and CLI-thin; verify whether any non-English or family-specific context hooks exist beyond the present wrapper.
  - `Zenith`: current API definitely supports optimizer configuration and plaintext/ciphertext transformations; verify whether it supports practical context cues such as language models, segmentation, or cipher-family hints that our wrapper is not yet exposing.
  - Separate "upstream tool cannot use context" from "our wrapper does not pass supported context".
- [x] Add a Decipher-local frontier automated solver suite.
  - Benchmark-split-compatible JSONL format with `frontier_class`, tags, thresholds, and optional `synthetic_spec`.
  - Curated initial cases live in `frontier/automated_solver_frontier.jsonl`.
  - `scripts/run_frontier_suite.py` runs the suite with the same automated/external solver lanes as parity.
  - `scripts/nominate_frontier_cases.py` proposes candidate cases from parity summaries/artifacts.
  - `scripts/build_frontier_report.py` summarizes pass/fail, regressions, slow cases, unsupported wrappers, and frontier movement.
  - Current April 2026 state:
    - [x] Add a `shared_hard` class so the suite tracks cases that are meaningfully challenging for both Decipher and Zenith, not just regressions and one Zodiac row.
    - [x] Make external frontier runs default to Zenith-only via `external_baselines/zenith_only.json`.
    - [ ] Revisit thresholds for a few rows now that `zenith_native` is the default automated homophonic path (`synth_en_80nb_s1` timing, multilingual simple-substitution expectations, etc.).

### Priority 4: Full-Agent Parity

- [ ] Add a full-agent parity smoke suite.
- [x] Add artifact checks for wrong-tool use.
  - Homophonic no-boundary should call `search_homophonic_anneal` before generic `search_anneal`.
  - High dictionary rate on no-boundary homophonic text must not trigger declaration without coherent reading.
  - Final-turn declarations should pick the best available branch, not the most rationalized branch.
- [x] Add an artifact gap analyzer with labels:
  - `tool_missing`
  - `tool_underpowered`
  - `agent_wrong_tool`
  - `premature_declaration`
  - `scoring_false_positive`
  - `external_unsolved`
  - `benchmark_data_issue`
- [x] Produce a parity dashboard:
  - family
  - external best
  - native best
  - full agent
  - gap type
  - next action

## Agentic Advantage Program

Only start claiming agentic advantage after parity failures are classified.
Candidate benchmark families:

- Context scaling: single-page vs 10-page vs full-document Borg/Copiale.
- Image-to-plaintext Track C runs where transcription and decipherment interact.
- Nomenclator/codeword stress cases.
- Dirty transcription variants with controlled symbol confusion.
- Short or underdetermined ciphers where multiple candidates and uncertainty matter.
- Historical-language/domain tests where metadata, spelling, and manuscript conventions matter.
