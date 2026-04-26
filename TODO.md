# Decipher TODO

## Agentic Parity Program

Goal: the agentic solver should have at least the same practical solving
capability as the best available non-agentic solvers on clean benchmark tasks,
and failures should identify a missing tool, a weak tool, a wrong agent choice,
or a benchmark data issue.

Current planning split:
- The Agent Loop Redesign plan is considered complete after Milestone 4 smoke
  coverage.
- New Copiale/generalization work is tracked in
  `docs/copiale_generalization_plan.md`.

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
    - Case-state note: `docs/borg_0109v_case_state.md`.
    - Latest high-level conclusion: pause intensive work on `0109v` for now.
      The best branch is readable and reaches `95.36%` edit-aware character
      accuracy / `77.92%` edit-aware word accuracy, but remaining issues are
      local historical-Latin cleanup, odd inserted/split words, and
      benchmark-alignment nuance rather than first-pass decipherment.
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
    - Focused repair diagnosis from the current best `0109v` text:
      - The current `decode_diagnose` / `decode_diagnose_and_fix` path only proposes **single-letter substitutions** that land on a word already present in the loaded Latin dictionary.
      - `RLURES -> PLURES` is detectable by that machinery, but it is still a weak case because it is a single local correction and `decode_diagnose_and_fix` defaults to `min_evidence=2` before auto-applying repairs.
      - Several visually obvious near-Latin forms in the current output are **not** reachable by the existing automatic repair path:
        - `TREUITER` would like `BREUITER`, but `BREUITER` is not present in the current Latin dictionary.
        - `SIMULITER` would like `SIMILITER`, but `SIMILITER` is not present in the current Latin dictionary.
        - `MORIETANTUR` is not a one-substitution miss of the historical target form and also is not repairable through the current dictionary lookup.
      - So the present limitation is not only "the agent failed to notice the error"; it is also that the shared repair primitive is intentionally narrow: one-edit, dictionary-backed, and conservative.
  - Next likely useful experiments:
    - [ ] Try a stronger Latin cleanup pass on `zenith_native` output: segmentation plus phrase-level split/merge and repair, beyond the current conservative shared helper.
    - [ ] Re-run the improved agentic boundary-tool path on a second small Borg case to see whether the `act_apply_boundary_candidate` gain generalizes beyond `0109v`.
    - [ ] Compare default substitution vs forced `zenith_native` on a second small Borg subset to see whether `0109v` is representative or a special case.
    - [ ] Revisit routing later if a cleanup step starts making the `zenith_native` basin consistently more useful.
  - Planned branch split:
    - [ ] **Branch A: automated Latin cleanup / dictionary expansion**
      - Goal: improve the shared automated repair primitives without changing the core agent posture.
      - Candidate tasks:
        - expand the Latin dictionary with missing high-value forms observed in Borg output
        - add historical/variant Latin spellings where they materially help repair
        - broaden repair primitives beyond pure one-substitution matches
        - evaluate whether `decode_diagnose_and_fix` / automated repair should allow lower-evidence fixes when a global score guard agrees
        - compare stronger automated cleanup on both substitution and forced-`zenith_native` paths
    - [ ] **Branch B: agent native-reading autonomy**
      - Goal: make the agent use its own linguistic judgment more directly, with tools as helpers rather than as the outer limit of what it can think.
      - Problem statement:
        - the current agent often treats tool output as the boundary of permissible reasoning
        - on partial Latin text, it does not reliably propose spelling repairs or boundary diagnoses from first-principles reading when the dictionary-backed tools stay silent
      - Candidate tasks:
        - strengthen prompt language so the agent treats tools as support, not permission
        - explicitly allow manual spelling-repair proposals from semantic/contextual reading even when no dictionary tool endorses them
        - explicitly instruct the agent to diagnose likely word-boundary drift from reading continuity, not only from `boundary_candidates`
        - [x] add a workflow where the agent can state a manual repair hypothesis first and then use tools to test/refine it
          - Added `decode_validate_reading_repair` for read-only validation of a proposed best reading, including whether it is character-preserving or requires letter repairs.
          - Added `act_resegment_by_reading` for one-shot character-preserving word-boundary overlays from a complete proposed reading.
          - Added `act_resegment_from_reading_repair` for the harder Borg-like case: the proposed reading may change letters, but if its character count matches the current branch, the tool applies only the implied word-boundary pattern and preserves the current key/letters.
        - evaluate whether the agent begins making manual Latin spelling and boundary proposals on Borg `0109v` and neighboring cases
      - English readability analog:
        - Added `fixtures/benchmarks/english_borg_analog`, a tiny benchmark root with one hand-shaped English simple-substitution case.
        - The plaintext is archaic/readable English, while the canonical cipher word boundaries intentionally split words such as `THEREFORE`, `PHYSICKER`, `APPLY`, `UNTO`, `AFTERWARD`, and `WITHOUT`.
        - A perfect source-boundary decrypt scores `100%` character accuracy but under `10%` word accuracy, making the Borg-style word-alignment problem easy to inspect in English.
        - Use this fixture when debugging whether the agent is reading the text directly or over-deferring to tool scores and word-alignment artifacts.
        - Milestone: the first run after the whole-stream reading/resegmentation tools reached `100%` character / `100%` word accuracy on the analog (`artifacts/english_borg_analog_001/919b40a14b6f.json`). This strongly suggests the missing abstraction was not another local boundary primitive, but a tool-mediated way for the agent to state and validate a full reading.
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

- [ ] **Reading-driven repair discipline.**
  - Once a branch decodes into recognisable target-language words, the
    agent's reading must dominate over scoring signals. Recent Borg
    artifacts show the agent making the correct cipher-symbol fix, the
    tool reporting `verdict: worse` from a single dict_rate delta, and the
    agent reverting. See AGENTS.md → "Reading-driven repair discipline" for
    the full diagnosis. Keep all prompt language language-agnostic; this is
    a general agentic-decipherment failure mode, not a Borg- or Latin-
    specific one.
  - **Prompt changes** (`src/agent/prompts_v2.py`):
    - [x] Add an explicit reading-vs-score hierarchy near the top of
      "Reading-driven repair":
      - When you cannot read the decoded text → scores are your only signal.
      - When you can partially read it → reading and scores are co-equal;
        prefer the change that produces more recognisable words.
      - When you can read coherent target-language words →
        **your reading is authoritative.** A score delta that disagrees with
        a reading-driven fix that produces additional real words is not a
        reason to revert.
    - [x] Replace the `TREUITER → BREUITER` worked example with a
      language-agnostic, cipher-symbol-framed example. Lead with cipher
      symbols, not decoded letters: `cipher word XXXXX decodes as FOO
      but reads as BAR; cipher symbol Sxx is currently → A and should
      be → B; call act_set_mapping(cipher_symbol='Sxx', plain_letter='B');
      this changes every occurrence of cipher Sxx — read the resulting
      decode for the full effect`.
    - [x] Add an explicit warning that `act_swap_decoded` is **bidirectional
      across the entire decoded text** and is a footgun for reading-driven
      repairs. For "this letter should be that letter in this word" fixes,
      always use `act_set_mapping` on the cipher symbol producing the wrong
      letter.
    - [x] Make the anchored-polish sequencing rule non-negotiable: do not
      call `search_anneal(preserve_existing=true)` (or
      `search_homophonic_anneal(preserve_existing=true)`) on a branch until
      at least one `act_set_mapping` reading-driven anchor has been
      applied. Otherwise the polish has nothing meaningful to anchor and
      just re-confirms the prior local optimum.
    - [x] Generalise the dict_rate threshold guidance away from
      language-specific numbers ("Latin caps near 0.20"). State the rule in
      generic form: `dict_rate` has language-and-cipher-specific ceilings;
      on boundary-preserving ciphers where cipher word boundaries don't
      align with target-language word boundaries, `dict_rate` can be
      inflated by short accidental fragments. **Do not declare on
      `dict_rate` alone — declare on reading.**
    - [x] Add explicit tool-output discipline: a `verdict: worse` reported
      by `act_set_mapping` is a score delta, not an authoritative quality
      verdict. If the change produced two or more decoded words that now
      read as real target-language words (or fragments of real words),
      accept it.
  - **Tool-surface changes** (`src/agent/tools_v2.py`):
    - [x] Stop surfacing a `verdict: improved/worse` field on
      `act_set_mapping`/`act_bulk_set` results. Report the score deltas as
      data only, and include a `changed_words` sample (was → now, up to ~8
      examples) so the agent reads the change rather than scoring it.
      `act_swap_decoded` may keep its verdict because it operates at the
      decoded-letter layer where the score is more meaningful.
    - [x] Tighten the `act_swap_decoded` tool description: state up front
      that it swaps two decoded letters bidirectionally across the entire
      branch and is rarely the right primitive for reading-driven repairs.
      When it auto-reverts, the result note should suggest the specific
      `act_set_mapping(cipher_symbol=…)` calls that would have made the
      same single-direction change.
    - [x] Downrank boundary candidates in `recommended_next_tool` and the
      `decode_diagnose` `note` whenever there are unattempted letter-level
      candidate corrections. Boundary edits should only be the top
      recommendation when there are no plausible letter-level fixes and the
      diagnostics indicate a likely missed split/merge.
    - [ ] Consider adding a `decode_letter_culprit(branch, word_index,
      position)` helper that returns "the decoded letter at this position
      is produced by cipher symbol X (currently → Y); other words
      containing cipher X: …". This shifts the agent's frame from decoded-
      letter to cipher-symbol without requiring the agent to do the lookup
      manually each time.
  - **Telemetry / artifact gap analyzer**:
    - [x] Add a `score_overrode_reading` label: agent identified a fix from
      reading, applied it via `act_set_mapping`, the score reported a
      regression, and the agent reverted or declared a strictly worse
      branch.
    - [x] Add an `unattempted_reading_fix` label: agent reasoning text
      proposes a specific `cipher symbol X → letter Y` fix but no
      `act_set_mapping`/`act_bulk_set`/`act_anchor_word` call follows
      within the next two iterations.
  - **Verification**:
    - [ ] Re-run `borg_single_B_borg_0109v` and `borg_single_B_borg_0045v`
      with the updated prompt + tool surfaces. Acceptance signals:
      - At least 2 reading-driven `act_set_mapping` calls per run.
      - No `search_anneal(preserve_existing=true)` call before any
        `act_set_mapping` on the same branch.
      - When a reading-driven fix produces additional readable words but
        score delta is negative, the agent keeps the fix.
      - Final declared branch ≥ automated_preflight on both char and word
        accuracy.
      - 2026-04-25 focused `parity_borg_latin_borg_0109v` rerun after the
        whole-stream reading tools: artifact
        `artifacts/parity_borg_latin_borg_0109v/b15432813fde.json`.
        Result: declared `repair` branch reached `13.9%` char / `20.8%`
        word, versus automated preflight `14.2%` char / `6.4%` word. This
        is a real word-alignment/reading improvement, but not a full solve:
        the analyzer still reports six `unattempted_reading_fix` warnings,
        and the agent did not yet use `act_resegment_by_reading` on the
        Latin case. Next prompt/tool iteration should encourage full-reading
        resegmentation when the branch contains many visible boundary drifts,
        while continuing to require targeted `act_set_mapping` for letter
        repairs such as `RLURES -> PLURES`.
      - Follow-up implementation: `decode_validate_reading_repair` now returns
        a `boundary_projection` whenever the proposed reading has the same
        character count, and `act_resegment_from_reading_repair` can apply
        that projection in one call. This is intended to make the Latin case
        behave more like the English analog: draft the best whole reading,
        apply safe boundaries, then repair the listed letter mismatches.
      - Added a pre-final declaration guard: if an agent rationale still
        mentions boundary/alignment issues and the branch has not used
        `decode_validate_reading_repair`, `act_resegment_by_reading`, or
        `act_resegment_from_reading_repair`, `meta_declare_solution` blocks
        once and points the agent to that workflow. Final-turn declarations
        are still accepted so partial readable work is not lost.
      - Added a penultimate-turn warning in the loop/panel. This is the
        important scheduling fix: the final turn must remain a declaration
        turn, so the agent is now told one turn earlier to run the full-reading
        validation/projection workflow instead of spending that turn on more
        local edits or search.
      - Late-window tool gating now hides and executor-blocks local edit/search
        tools. This successfully caused the agent to attempt the full-reading
        workflow on `0109v`, but the proposed readings failed character-count
        checks (`345` current chars vs `307`/`378` proposed chars), so no
        boundary projection was applied. Latest gated-run artifact:
        `artifacts/parity_borg_latin_borg_0109v/8776c6231842.json`
        (`13.9%` char / `8.9%` word).
      - Current conclusion: pause Borg-specific prompt/tool iteration until a
        more modern agentic loop exists. Prompt nudges and late gating can make
        the agent attempt the right workflow, but the current turn-based loop
        is too clunky for exact length-preserving reading repair.
      - Next design note: `docs/agent_loop_redesign_plan.md`.
    - [x] Add an automated check to `tests/test_agent_reliability.py` (or a
      new fixture-driven test) that simulates a "verdict: worse but
      reading-positive" tool result and asserts the prompt-derived loop
      keeps the change. This protects the discipline against future prompt
      drift.
- [ ] Design a provider-neutral modern agent loop before further Borg-specific
  work.
  - [x] Keep the core Decipher harness independent of any one LLM provider.
  - [x] Use a small provider adapter so Claude, OpenAI, and future local/hosted
    models can share the same tool/workspace/artifact state machine.
    - Cross-provider CLI support now exists for `--provider anthropic|openai|gemini`.
      OpenAI and Gemini adapters translate Decipher's existing Anthropic-style
      message/tool transcript into each provider's function-calling shape and
      normalize responses back into `ModelResponse`.
    - API keys are read from provider-specific env vars, `.env`, gitignored
      `.decipher_keys/*_api_key` files, or macOS Keychain.
    - Remaining work: run the live smoke packet across candidate models and
      calibrate provider-specific reliability/cost notes.
    - Add periodic checkpoint artifact saving during long live-provider runs.
      Current artifacts are written only after `run_v2` returns, so an
      interrupted or hung provider call can lose partial post-fix state, as
      seen with the Gemini Pro packet rerun.
  - [x] Separate outer benchmark iterations from inner tool steps.
  - [x] Add same-iteration retry for gated/disallowed tools so a stale tool choice
    does not consume a scarce late-turn action.
  - [x] Add same-iteration retry for boundary-projection count mismatches so a
    truncated or overlong full-reading proposal can be revised before the
    outer iteration advances.
  - [ ] Make workflows first-class, especially:
    - full-reading validation and boundary projection
    - reading-driven mapping repair
    - declaration preparation
    - First reading-repair slice added:
      `decode_plan_word_repair` plans same-length word repairs and
      `act_apply_word_repair` applies them with `changed_words` feedback.
      Follow-up repair-menu slice added:
      `decode_plan_word_repair_menu` compares multiple candidate readings for
      one word without mutating the branch, flags repeated-symbol conflicts,
      summarizes collateral changed words, and tells the agent when a direct
      word repair should not be applied. This addresses the `RLURES -> PLURES`
      failure mode where the agent knew a plausible target word but one
      naive mapping changed another position in the same cipher word.
      Planned repairs are now also tracked in a durable `repair_agenda`
      artifact field, with `repair_agenda_list` / `repair_agenda_update`
      available to keep open hypotheses from vanishing into transcript memory.
      Added `workspace_branch_cards` for compact branch state before
      declaration, plus orthography-risk warnings for broad Latin `U/V` or
      `I/J` shifts introduced by reading repairs.
      Declarations now require open/blocked repair agenda items to be resolved
      with `repair_agenda_update`, and multi-branch runs must call
      `workspace_branch_cards` before declaring. The final action turn exposes
      those bookkeeping tools alongside `meta_declare_solution`; if the agent
      performs final bookkeeping but forgets to declare, the loop retries
      inside the same final iteration with an explicit declaration nudge.
      First Borg trial:
      `artifacts/parity_borg_latin_borg_0109v/495a27b339ba.json`, 13.9%
      char / 9.1% word. The agent used the wrapper for two useful repairs
      and correctly backed away from `RLURES -> PLURES` after collateral
      damage.
      Follow-up agenda trial:
      `artifacts/parity_borg_latin_borg_0109v/38e3d02d7c7a.json`, 11.6%
      char / 5.1% word. This confirmed agenda use but exposed the Latin
      orthography trap: the agent improved its Latin-looking reading by
      classicizing `U` to `V`, hurting the benchmark transcription.
      Later trial:
      `artifacts/parity_borg_latin_borg_0109v/260a15ce6778.json`, 13.9%
      char / 7.7% word. This avoided the orthography trap but left
      `RLURES -> PLURES` open in the agenda despite rejecting it in prose,
      motivating the declaration gate.
      Post-gate trial:
      `artifacts/parity_borg_latin_borg_0109v/fca17fd203a6.json`, 13.9%
      char / 7.7% word. This showed richer tool behavior but also exposed a
      final-turn failure: the agent called `repair_agenda_list` and
      `workspace_branch_cards` but forgot `meta_declare_solution`, so fallback
      declaration fired. A same-iteration final declaration retry now covers
      that case.
      Later trial:
      `artifacts/parity_borg_latin_borg_0109v/8f2993dc03a6.json`, 13.9%
      char / 7.7% word. Final declaration retry worked and the agent recovered
      from a bad `RLURES -> PLURES` repair by restoring `M -> R`, but the run
      still did not use `act_resegment_window_by_reading`. This motivates two
      active rules: uncertain word repairs should go through the repair menu,
      and once words are readable the agent should perform local/global
      boundary repair rather than merely describing boundary drift.
      Scoring follow-up: word accuracy now uses edit-aware exact-word alignment
      instead of positional zip comparison. Local extra/missing/split words are
      shown as alignment edits, and later exact word runs can resynchronize
      instead of being counted wrong for the rest of the text.
      Character accuracy now uses the same principle at character level:
      exact-character alignment with penalties for substitutions and gaps.
      This preserves substitutions as errors while allowing local
      insertions/deletions to resynchronize later matching text.
      Loop/tool follow-up:
      boundary-projection count failures now get up to three same-iteration
      retries, and the late reading workflow has a bounded low-cost repair
      sandbox. Within that sandbox the agent can inspect, compare repair
      menus, apply small word repairs, try local resegmentation windows, call
      branch cards, and declare without burning additional outer benchmark
      iterations. Failed local resegmentation windows now also return
      `nearby_compatible_windows`, which catches stale-index cases such as
      asking for `BITUR | SI -> LIBEBITUR` when the matching window is really
      `LIBE | BITUR`.
      Continuation follow-up from
      `artifacts/parity_borg_latin_borg_0109v/6a1b5bd9137b.json`: the agent
      used most of its additional outer iterations on read-only inspection.
      The loop now has a bounded read-only inspection sandbox, so
      `decode_show`, branch cards, dictionary/corpus probes, diagnostics, and
      score panels can continue inside the same outer iteration before the
      agent chooses a repair/search/declaration action. Compact pages also
      show all words in the workspace panel (currently up to 90 words), which
      should reduce unnecessary `decode_show` calls caused by panel truncation.
      Milestone 4 generalization checkpoint:
      - Borg `0140v`
        (`artifacts/borg_single_B_borg_0140v/47df72a4da8b.json`) improved
        from weak automated preflight (`36.9%` char / `0.0%` word) to a
        readable agent branch (`85.5%` char / `54.8%` word).
      - Borg `0077v`
        (`artifacts/parity_borg_latin_borg_0077v/c9d17916d17f.json`) improved
        from weak `zenith_native` preflight (`37.2%` char / `2.8%` word) to a
        readable partial agent branch (`84.1%` char / `53.5%` word).
      - Borg `0171v`
        (`artifacts/borg_single_B_borg_0171v/a43a53111e26.json`) exposed a
        do-no-harm failure: preflight was already strong (`90.9%` char /
        `72.7%` word), but the agent declared a more classicized repair branch
        at `85.8%` char / `50.8%` word. The prompt, preflight context, and
        branch cards now tell the agent to treat `automated_preflight` as a
        protected no-LLM baseline and avoid broad manuscript-orthography drift.
      - Copiale `p068`
        (`artifacts/copiale_single_B_copiale_p068/7d795a0ae0a9.json`) did not
        improve over preflight (`45.3%` char / `0.0%` word). The agent found
        German-looking islands but not coherent sentence-level German, so
        Copiale should become a separate capability track rather than a Borg
        repair follow-up.
  - [x] Track explicit run state: active mode, branch, repair agenda, held/reverted
    repairs, unresolved hypotheses, per-branch workflow completion.
  - [x] Preserve complete artifact observability for every model call, tool call,
    workflow event, gate rejection, retry, and declaration.
  - [x] Validate the boundary-projection prototype on the English Borg analog
    and Borg `0109v`, then compare artifacts against the prior runs.
    - English analog rerun:
      `artifacts/english_borg_analog_001/f8c8ead3e9b2.json`, 100.0% char /
      100.0% word in 6 iterations.
    - Borg completed once before actuator-only gate tightening:
      `artifacts/parity_borg_latin_borg_0109v/048448b15ebb.json`, 13.9%
      char / 7.5% word; showed that validation-only should not end the gate.
    - Post-tightening Borg retry was interrupted by Anthropic credit-balance
      API error on iteration 4:
      `artifacts/parity_borg_latin_borg_0109v/5b16b17ac4c1.json`.
    - Milestone 2 comparison:
      `docs/agent_loop_milestone2_comparison.md`.
  - [x] Improve agent-loop command-line output.
    - Added `--display {auto,pretty,raw,jsonl}` for agentic benchmark/crack/
      testgen paths.
    - `auto` uses pretty mode on an interactive terminal, raw mode when
      piped, and legacy verbose output with `-v`.
    - Pretty mode renders a live decrypt panel, agent commentary, concise tool
      summaries, branch scores, changed-word previews, and loud API/fallback
      errors.
    - Final screens now include a human-readable reading/process summary built
      from `meta_declare_solution`, including what the text appears to be
      about, status/uncertainty notes, and whether further iterations would
      likely help.
    - Raw mode preserves the compact event/tool stream; JSONL mode exposes
      structured events for GUI wrappers.
    - Output is driven by structured `loop_events` plus `workspace_snapshot`
      events; artifact JSON remains the source of truth.
  - [ ] Design a richer artifact/run review GUI.
    - The pretty terminal display is useful for live runs, but long final
      word alignments, decrypts, and reading/process summaries need scrollable
      panes.
    - Desired affordances: scroll through full word/character alignment,
      inspect branch cards and repair agenda history, compare parent/resume
      artifacts, and copy exact tool calls or branch diffs for follow-up runs.
  - [ ] Update the README for the current agentic interface and capabilities.
    - Add a short testing section that points to `docs/test_inventory.md`.
    - Include the main default test command, the opt-in Milestone 4 smoke
      command, and the live cross-model packet command.
    - Document cross-provider agentic configuration:
      `--provider anthropic|openai|gemini`, model inference from `--model`,
      and provider-specific API key locations/env vars.
    - Include gitignored local key-file locations:
      `.decipher_keys/anthropic_api_key`, `.decipher_keys/openai_api_key`,
      and `.decipher_keys/gemini_api_key`.
    - Note that `.env`, `.env.*`, `.decipher_keys/`, and `*_api_key.txt` are
      ignored so credentials should not be committed.
    - Document `--display {auto,pretty,raw,jsonl}`, the live decrypt view,
      token/cost reporting, final reading/process summary, and raw/JSONL modes
      for wrappers.
    - Document automated preflight, the protected `automated_preflight`
      baseline branch, branch cards, repair agenda, reading-driven repair
      tools, boundary/resegmentation tools, and `resume-artifact`.
    - Include current benchmark command examples for clean no-extra-context
      agentic runs, artifact continuation, and the planned
      `--benchmark-context` modes once implemented.
  - [x] Add first-class artifact resume/continuation.
    - Command shape: `decipher resume-artifact <artifact.json>
      --extra-iterations N [--branch BRANCH]`.
    - Resumes from workspace state and branch snapshots, not by replaying the
      entire old provider transcript verbatim.
    - Includes prior final summary, repair agenda, held/open repairs, branch
      previews, missing-tool requests, and compact declaration context in the
      new first prompt.
    - Writes a chained artifact with `parent_run_id`/`parent_artifact_path` so
      continued runs are auditable and comparable without overwriting the
      original run.
    - Current limitation: old artifacts did not persist custom branch
      `word_spans`, so exact boundary overlays can only be restored for
      artifacts produced after the new `word_spans` snapshot field landed.
  - [ ] Add explicit benchmark-context modes for agentic runs.
    - Proposed command shape:
      `--benchmark-context none|metadata|ciphertext|metadata+ciphertext`,
      defaulting to `none`.
    - `metadata` should pass benign manifest context such as source family,
      plaintext language, date/century, provenance, manuscript page, cipher
      type, symbol count, curation notes, and source URL. Do not include
      solution plaintext or solution-derived hints.
    - `ciphertext` should pass same-cipher context records from benchmark
      split definitions, as the current `10page`/`full` cases already can,
      but make this explicit in artifacts instead of implicit.
    - Artifacts must record the selected context mode and a compact summary of
      the exact context supplied, so clean parity runs and context-aware runs
      are never mixed silently.
    - Keep this off until the current no-extra-context generalization pass is
      documented for Borg `0140v`, Borg `0171v`, Borg `0077v`, and at least
      one Copiale/German stretch case.
  - Detailed no-code plan: `docs/agent_loop_redesign_plan.md`.
- [ ] Add a Copiale/German capability track.
  - Do not treat Copiale as "Borg in German." Current `p068` evidence shows
    scattered German word islands are not enough for declaration.
  - Near-term work:
    - Add stronger declaration discipline for Copiale-like no-boundary
      homophonic/nomenclator ciphers: require coherent sentence-level German,
      not just common short words.
    - Audit whether benchmark metadata/context should be opt-in for Copiale,
      especially source family, Masonic/fraternal domain, page context, and
      same-cipher neighboring pages.
    - Improve German continuous n-gram/model support and consider a
      Copiale-specific or 18th-century German model.
    - Investigate nomenclator/codeword behavior and whether some symbols or
      clusters should be treated as nulls, abbreviations, or multi-letter
      tokens rather than simple letter homophones.
    - Add a small Copiale-focused artifact packet once the above exists:
      `p017`, `p035`, `p052`, `p068`, and `p084`, with clean
      no-extra-context and context-aware modes separated.
- [x] Add a full-agent parity smoke suite.
  - [x] Add no-LLM automated baseline packet for the Milestone 4 cases:
    `frontier/agentic_milestone4_smoke.jsonl`.
  - [x] Add opt-in pytest coverage for the automated-only baseline pass:
    `DECIPHER_RUN_MILESTONE4_SMOKE=1 .venv/bin/python -m pytest
    tests/test_milestone4_smoke.py`.
    This runs `AutomatedBenchmarkRunner` on every packet case, checks
    `run_mode=automated_only`, zero tokens, zero estimated cost, and baseline
    char-accuracy/elapsed expectations.
  - [x] Add the actual agentic smoke layer separately.
    `tests/test_milestone4_smoke.py` now includes fake-provider LLM-agent
    coverage for declaring a protected `automated_preflight` branch and for
    making a small reading-driven mapping repair before declaration. Keep any
    live-provider smoke packet opt-in if we add one later.
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
