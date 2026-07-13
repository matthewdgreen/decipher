# Copiale And Generalization Plan

Status: new plan opened after completing the Agent Loop Redesign milestones and
Milestone 4 smoke coverage. Treat this as the next capability track, not as a
continuation of Borg-specific repair work.

## Goal

Make Decipher's agentic and automated paths generalize beyond readable Borg
Latin pages, starting with Copiale/German and then widening to other benchmark
cipher families. The goal is not just higher scores; each failure should point
to a missing model, cipher-system assumption, context hook, agent workflow, or
benchmark issue.

## Current Read

- Borg progress is real: multiple Latin pages now reach readable partial or
  strong outputs, and the English analog confirms the agent can exploit full
  readings when the right actuator exists.
- Copiale `p068` did not improve meaningfully over preflight. The agent found
  German-looking islands but not coherent sentence-level German.
- That failure is structurally different from Borg `0109v`: Copiale likely
  needs stronger German language modeling, better homophonic/nomenclator
  handling, and stricter declaration discipline around isolated words.
- The new Milestone 4 smoke suite gives us a regression base before beginning
  this broader work.

## Principles

1. Separate blind parity from context-aware runs.
   If the solver uses source family, language, or manuscript notes, artifacts
   and summaries must say so.

2. Avoid Borg-shaped overfitting.
   A workflow that helps word-boundary drift in Latin should not be assumed to
   solve German homophonic manuscripts.

3. Build diagnostics before broad fixes.
   For each failing Copiale page, we want to know whether the bottleneck is
   search, scoring, language model quality, symbol inventory, nulls/codewords,
   segmentation, or agent behavior.

4. Keep automated and agentic paths comparable.
   The agent may receive richer tools and optional context, but the no-LLM
   baseline must remain easy to run and artifact-compatible.

## Milestone 1: Copiale Evidence Packet

- [x] Select a compact Copiale packet:
  `copiale_single_B_copiale_p017`, `p035`, `p052`, `p068`, and `p084` if
  available in the local benchmark checkout. The Decipher-local evidence suite
  is `frontier/copiale_evidence_packet.jsonl`.
- [ ] Run automated-only baselines for the packet and store/compare artifacts.
- [ ] Run agentic baselines without extra manuscript context.
- [ ] Add an artifact comparison note summarizing:
  - first readable branch
  - final declared branch
  - German word/sentence coherence
  - edit-aware character and word scores
  - whether the agent declared on isolated islands
- [ ] Add the packet to a frontier/smoke file only after thresholds are stable.

## Milestone 2: German Model And Scoring Audit

- [x] Inventory the currently bundled German resources:
  `models/ngram5_de.bin` and `models/ngram5_de_500.bin` are redistributable
  Gutenberg-derived Zenith-format binaries; `resources/dictionaries/german_common.txt`
  is a 3,057-word support dictionary. This is a starting point, not evidence
  that the model fits 18th-century Copiale prose.
- [x] Verify the bundled German corpus/model provenance and normalization
  against model metadata during automated runs. `zenith_native` homophonic
  steps now include `model_metadata` with path, checksum, source corpus,
  normalization, and redistribution fields when a sidecar metadata file is
  available.
- [x] Compare German dictionary scoring, quad scoring, and continuous n-gram
  scoring on known-good German plaintext samples. Use
  `scripts/audit_german_scoring.py` to compare Copiale plaintext against
  deterministic degraded controls under dictionary, wordlist quadgram, and
  Zenith-format binary model signals.
- [ ] Build or select a stronger German continuous model if the current model
  is weak for 18th-century or Masonic-style prose.
- [ ] Ensure automated artifacts record the selected German model name, path,
  checksum, and provenance.
- [ ] Add a small model A/B packet so future model changes are visible.

## Milestone 3: Copiale Cipher-System Diagnostics

- [x] Add a first Copiale-focused diagnostic report that does not require
  ground truth but can attach post-hoc summary scores when available:
  `scripts/report_copiale_evidence.py`.
- [x] Measure first-pass symbol inventory, flatness, rare-symbol pressure, and
  coarse/missing word-boundary structure per evidence-packet page.
- [x] Deepen the report with solver-basin homophone-family behavior,
  null/codeword candidates, and repeated-symbol n-grams. These diagnostics use
  the solver-produced key/decrypt when an artifact is supplied, not the
  ground-truth plaintext.
- [x] Add a separate offline calibration report that can compare null/codeword
  candidates against ground truth after the solve, so we can tune heuristics
  without leaking solution data into runtime tools. Invoke it with
  `scripts/report_copiale_evidence.py --ground-truth-calibration`.
- [ ] Tune the null/codeword heuristic against the offline calibration. First
  p052 read: insertion-heavy symbols are not merely rare/localized; several
  frequent symbols inside collapsed homophone families are overused by the
  solver, so a null-aware profile should consider family overuse and insertion
  pressure, not just low-frequency glyphs.
  - First p052 pair-mask probe: baseline was `64.4%`; best post-hoc mask was
    `S005,S014` at `70.6%`, and the solver's own no-ground-truth selection
    ranked it second. The top selection mask `S020,S050` reached `66.1%`.
    This is enough signal to continue probing, but not yet enough to promote
    null masks into the default automated route.
  - First five-page pair-mask packet: best-by-selection improved all pages
    materially (`p017` 75.3%, `p035` 61.2%, `p052` 66.1%, `p068` 54.0%,
    `p084` 63.0%). Best post-hoc masks were higher on four of five pages
    (`p035` 68.7%, `p052` 70.6%, `p068` 55.3%, `p084` 63.4%), so generation
    is clearly useful while selection still needs calibration.
  - The probe now reports no-ground-truth validation scores in addition to raw
    solver selection and post-hoc character accuracy. The current v2 score
    reduces raw anneal-selection weight and adds stricter German coherence,
    letter-diversity, top-letter, and repetition controls.
  - Current v3-ish slice: the v2 validator now also consumes a mild
    `binary_ngram_fit` component when a Zenith-format German binary model is
    available. The component is derived from ground-truth-free mean 5-gram log
    probability of each finalist decrypt and is intended as a tie-breaker
    against repetitive word-island basins, not a declaration criterion.
  - The validation report now also surfaces `language_shape`, binary fit, and
    repetition/overuse components directly, so misses can be read as scoring
    failures rather than as a wall of German-looking text.
  - The probe now scores the full candidate plaintext for validation, not only
    the first preview slice. This matters for no-boundary German where a
    candidate can have a plausible first line and collapse later. Saved
    `--include-all-rows` probe JSONL now carries `validation_text` so the
    report can faithfully rerank rows without rerunning the solver.
  - Segmentation diagnostics now include segmented word count, pseudo-word
    fraction, short-word fraction, and long-pseudo-word fraction. The v2
    validation score includes a mild `segmentation_shape` penalty for
    candidates held together by pseudo-word glue.
  - Candidate selection now reserves room for rare/localized ciphertext
    anchors in addition to solver-key homophone-family pressure. This prevents
    a noisy first-pass key from burying plausible null/codeword symbols below
    the candidate cap. Focused p068 testing now includes formerly-missed rare
    pairs such as `S090,S091`.
  - On the original full-row five-page packet, v2 still picks the exact post-hoc best
    mask on only `1/5` pages, but it captures the post-hoc best within the
    top-3 validation finalists on `5/5` pages. Treat top-N finalist promotion
    as the next design target rather than single-winner selection.
  - On the binary-model candidate-blend probe
    `artifacts/copiale_evidence_packet/null_probe_pair_masks_binary_rows.jsonl`,
    validation picked the exact post-hoc best on `3/5` pages. p068 remained
    the calibration failure: before candidate blending, the best post-hoc row
    was outside top-8; after candidate blending, focused p068 probing found a
    `59.4%` candidate and placed it near the top of the validation menu. The
    remaining p068 problem is readability discrimination: some lower-accuracy
    rows look more sentence-like to the ground-truth-free scorer than the
    post-hoc best row.
  - Focused p068 promotion testing also showed that useful one-off symbols can
    sit just below the old diagnostic cap. The null/codeword diagnostic helper
    now keeps up to 32 rare/localized candidates before the production selector
    applies its budget, so symbols such as p068 `S075` remain reachable.
  - The production null-mask path now includes a bounded beam extension stage:
    after the initial mask screen, `DECIPHER_NULL_MASK_BEAM*` settings extend
    the best masks with additional candidate symbols. This permits targeted
    size-3 hypotheses without brute-forcing all triples.
  - The automated path now also has a bounded top-k neighborhood stage:
    `DECIPHER_NULL_MASK_NEIGHBORHOOD*` tries add/remove/swap variants around
    the strongest scalar-validation masks. This keeps close p068-style basins
    alive for correction without committing to one null hypothesis too early.
    The neighborhood generator now prioritizes removals and same-size swaps
    before larger expansions, so pair-level alternatives get tested before the
    budget is spent on triples.
  - The automated path now adds a consensus-polish stage:
    `DECIPHER_NULL_MASK_CONSENSUS_*` derives a stable key from the strongest
    scalar-validation finalists, freezes only symbol mappings that those
    finalists agree on, and reruns the same masks with disputed symbols still
    mutable. This is intended for cases where the solver has found a readable
    basin but needs local correction rather than another full restart.
  - The ranking code now uses an extensible language-scoring subsystem in
    `src/analysis/language_scoring.py`. German is the first real calibration
    target, but the null-mask validator is wired to generic
    `language_coherence`, `language_shape`, and function-overuse signals so
    future Latin/English/French/Italian profiles can share the same workflow.
  - The validator now also includes a generic `word_lattice_quality` component
    built from dictionary segmentation diagnostics (`dict_rate`,
    segmentation cost, pseudo-word fraction, long pseudo-word pressure, and
    short-word pressure). This is deliberately language-independent once a
    dictionary/segmenter is available.
  - The first hand-scored sentence-quality pass is now general rather than
    Copiale-specific. Automated candidate diagnostics include content-word
    evidence (non-stopword dictionary hits, long content words, and
    content-character fraction), and scalar v2 validation uses
    `content_word_quality` together with lattice, binary n-gram, language
    shape/coherence, repetition, template-island pressure, and damage-control
    features. This is still a heuristic scorer; the next calibration step is
    to build real-corpus positives and solver-generated bad-basin negatives
    across languages.
  - A first trainable fast-scorer scaffold now exists. `language_scoring.py`
    provides bounded features and a transparent linear model, while
    `scripts/train_language_quality_scorer.py` trains JSON models from solved
    artifacts, probe rows, and positive corpus text. Treat same-case training
    as diagnostic only; the next meaningful Copiale test should be held-out by
    page. The first held-out p068 run trained on the other Copiale evidence
    pages and ranked the best p068 candidate in the top 3. A first synthetic
    word-island negative pass was too aggressive and demoted the real strong
    p068 basin, so synthetic negatives should stay experimental until they are
    generated from realistic solver failures rather than blunt fragment soup.
  - Runtime integration is now opt-in: set
    `DECIPHER_NULL_MASK_LANGUAGE_QUALITY_MODEL=<model.json>` and
    `DECIPHER_NULL_MASK_RANKER=language_quality` to attach trained
    `language_quality_*` fields to null-mask finalists and sort by a blended
    rank score. The blend keeps scalar validation as the backbone and treats
    the trained raw score as a modest extra vote, because the first p068 smoke
    showed the linear model can saturate on German-looking word islands. This
    is ready for calibration runs, not yet for default Copiale claims.
  - Candidate-only training is currently the most promising objective. The
    trainer has `--candidate-only`, which excludes clean plaintext/corpus rows
    and trains only on finalist menus. A held-out p068 candidate-only model
    trained on other Copiale pages selected the `S003,S021,S030`
    consensus-polish basin in a live automated p068 run and reached `59.0%`
    character accuracy. This is a real selection improvement, but still a
    damaged basin rather than a final Copiale solve.
  - Null-mask finalist reporting now computes an ensemble/pairwise score
    (`ensemble_score_v1`) alongside the scalar v2 validator. It votes across
    lattice quality, damage controls, binary n-gram fit, diversity, top-letter
    control, dictionary rate, language content, and solver selection. Language
    content is deliberately capped so fragment-rich damaged basins do not
    swamp cleaner but less sentence-like candidates. This is currently a
    calibration ranker; production selection still defaults to the scalar v2
    validator unless `DECIPHER_NULL_MASK_RANKER=ensemble` is set.
  - The automated null-mask confirmation stage now records independent-rerun
    stability evidence but does not control selection. Confirmation means were
    too noisy: they could demote good initial finalists when one rerun landed
    in a worse basin. Focused p084 automated reruns remain around `64.0%`;
    focused p068 automated reruns remain weak (`47.7%` in the latest run), so
    p068 should be treated as the next agent-review/ranking calibration case,
    not as an automated single-winner success.
  - The automated path now has a separate promotion stage before confirmation:
    the initial screen ranks candidate masks, then promotes the top finalists
    through a stronger same-mask solve controlled by
    `DECIPHER_NULL_MASK_PROMOTE_TOP_N`,
    `DECIPHER_NULL_MASK_PROMOTE_RERUNS`, and
    `DECIPHER_NULL_MASK_PROMOTE_BUDGET`. Promotion can update the
    representative decrypt/key and final validation rank; confirmation remains
    stability evidence only. This is the current automated answer to p068-like
    close damaged-basin menus. The default promotion width is now 12, matching
    the default finalist menu, because p068 showed that useful basins can sit
    around ranks 8-12 when the scorer over-rewards short German word islands.
  - Multi-page selection now has an explicit robustness audit:
    `scripts/research/copiale/report_copiale_selector_robustness.py`. Current saved artifacts
    support a two-stage policy: use the balanced page score for the broad
    elite menu, then use the anti-fragment robust score for the smaller
    refined/local-repair portfolio. This avoids a raw elite-menu regression
    while preserving the robust score's observed gains after repair.
  - Breadth diagnostics now preserve candidate identity more reliably. New
    null-mask artifacts attach `candidate_id` and `evaluated_index` to
    evaluated rows and top finalists, while
    `DECIPHER_NULL_MASK_STORE_EVALUATED_TEXT=1` provides an opt-in full-text
    mode for heavy calibration runs. `report_copiale_breadth_curve.py` uses
    those IDs first and falls back to signature matching for legacy artifacts.
    Promotion adoption now also requires a material validation gain
    (`DECIPHER_NULL_MASK_PROMOTE_ADOPTION_MARGIN`, default `0.08`), so a
    same-mask rerun cannot overwrite a useful basin merely by improving the
    noisy scalar score by a tiny amount.
  - A first global shared-key repair probe now exists:
    `scripts/research/copiale/probe_copiale_multipage_global_repair.py`. It starts from a
    multi-page finalist, finds disputed symbols in damaged windows across
    pages, and evaluates bounded shared-key/null edits with the same
    page-aware runtime scores used by the multi-page selector. On the current
    five-page `portfolio_local_repair` artifact, the single-edit probe found
    only tiny post-hoc gains (`79.2%` baseline to `79.3%` for `S084:E->T`)
    and the robust score slightly preferred a non-improving edit. Treat this
    as a useful diagnostic surface, not yet an automated acceptance policy.
    Pairwise global repair now has explicit pruning (`--pair-candidate-limit`,
    `--max-pairs`) and liveness output (`--progress`). It also emits
    ground-truth-free `accept/review/baseline` diagnostics, but multi-edit
    variants are review-only by default unless `--allow-pair-acceptance` is
    supplied. The current pruned pair run found no default-accepted repair;
    this is the desired conservative behavior until the scorer can distinguish
    tiny real repairs from globally plausible but harmful edits.
    The report now includes per-variant repair evidence: page-by-page runtime
    deltas, language-quality/dictionary/pseudo-word deltas, changed preview
    excerpts, and explicit post-hoc calibration flags. On the same five-page
    artifact, the top robust-ranked pair (`S084:E->T; S049:S->N`) is now easy
    to diagnose: runtime scores rise on most pages, but post-hoc char drops on
    four pages. Smaller `S084:E->T`-family edits remain plausible review items
    rather than accepted repairs, which is the right posture until the runtime
    scorer has a stronger reading-level signal.
- [ ] Add reading-driven logogram/codeword hypothesis tools.
  - A high-probability logogram signal is: readable damaged plaintext plus a
    whole-word hole at a position occupied by an unmapped/null-rendered symbol.
    This should be treated as evidence for a nomenclator/codeword test, not as
    an ordinary single-letter typo.
  - Prototype status: `scripts/research/copiale/probe_reading_holes.py` now performs a first
    reading-first pass. It segments a damaged candidate into word islands,
    reports broken words, identifies reader-visible missing-word slots, then
    aggregates those slots by cipher symbol and recurrence contexts. A second
    recurrence-level table asks whether the same symbol repeatedly behaves like
    a missing-word marker and renders rereads with
    `<MISSING_WORD?>` placeholders.
  - Current calibration result: the local segmenter-only layer is useful for
    exposing the evidence chain, but it does **not** yet reliably surface true
    Copiale logograms. On the four-page smoke run, ordinary one-letter symbols
    at word boundaries dominate the shortlist, while true logograms remain
    low-rank or unrevealed. The script now has an opt-in `--rereader llm`
    semantic pass that sends recurrence packets to the configured API model
    and asks it to distinguish real missing whole-word/codeword slots from
    ordinary damaged letters. This is still a diagnostic harness, not default
    solver behavior.
  - Candidate generation must check all recurrences of the suspicious symbol
    and reread the text with the proposed expansion installed. A repair should
    be promoted only when the expansion is globally plausible, not merely cute
    in one local phrase.
  - Unknown-but-suspicious is a valid output state. Tooling should support
    rendered placeholders such as
    `THE BIG BROWN <possible logogram:S123> JUMPED OVER` or
    `<unknown logogram:S123>` when the context proves a missing unit exists
    but does not pin down the exact plaintext.
  - Keep the evidence chain explicit for the future agent cleanup pass:
    identify homophonic structure from alphabet/frequency evidence; identify
    true null candidates from recurrence/readability evidence; identify
    logogram candidates from repeated missing-word holes; then test expansions
    by rereading all occurrences globally.
- [x] Add a prototype null-mask search script:
  `scripts/research/copiale/probe_copiale_null_masks.py`. It generates candidate null masks
  without plaintext, reruns the homophonic solver on filtered token streams,
  and reports ground-truth accuracy only after each candidate is produced.
  Keep it as a calibration tool until the selection signal is strong enough to
  promote into `AutomatedBenchmarkRunner`.
- [x] Add a cheap saved-probe reporter:
  `scripts/research/copiale/report_copiale_null_probe.py`. Run the probe with
  `--include-all-rows` when tuning validation; the report can then compare raw
  solver selection, no-ground-truth validation ranking, and post-hoc character
  accuracy without rerunning the native solver. It also reports aggregate
  exact-hit/gap metrics, top-N capture rates, and component-level miss
  analysis.
- [x] Add a first opt-in top-N null-mask automated profile. Use
  `--homophonic-refinement null_masks` to run the baseline homophonic
  solver, generate candidate null/codeword masks, solve filtered streams, rank
  a finalist menu with ground-truth-free language coherence signals, and record
  `search_null_masks` in the artifact. This is deliberately not the default
  route yet. `null_masks` is the preferred public name; `copiale_nulls`
  remains accepted only as a backward-compatible alias from the first
  Copiale-focused experiments.
- [ ] Decide whether the default automated route should use:
  - plain homophonic substitution
  - homophonic plus null handling
  - nomenclator/codeword hypotheses
  - segmentation/transcription normalization first
- [x] Surface the null-mask finalist menu to the agent in concise branch cards
  so it can compare several German-ish branches instead of inheriting a single
  brittle automated selection. `search_automated_solver` now creates a
  `null_mask_*` review session when run with
  `homophonic_refinement=null_masks`; the agent can page candidates with
  `search_review_null_mask_finalists`, record contextual readability with
  `act_rate_null_mask_finalist`, and install selected branches with
  `act_install_null_mask_finalists`. The default review page now shows eight
  finalists and includes segmentation-shape, binary n-gram, promotion, and
  confirmation diagnostics so close damaged basins are easier to compare.

## Milestone 4: Agent Workflows For German Manuscripts

- [ ] Add a "German coherence" workflow that distinguishes isolated dictionary
  hits from sentence-level German.
- [ ] Teach the agent to hold, not declare, branches that only contain islands
  such as articles, particles, or short common words.
- [ ] Add a reading-driven null/logogram workflow for homophonic/nomenclator
  basins.
  - The agent should see explicit evidence packets for each investigative
    step: homophonic diagnosis, null evidence, missing-word/logogram evidence,
    and recurrence-tested repair hypotheses.
  - The agent should call logogram hypothesis/review tools when a missing-word
    hole plus an unmapped/null-rendered symbol is present. This should be
    justified by the evidence pattern rather than by random availability of a
    new tool.
  - The workflow must allow marked uncertainty in the candidate plaintext:
    `<possible logogram:S123>` is a useful result when the exact expansion is
    unknown.
- [ ] Add optional context-loading support for benchmark `context_records`,
  gated by a CLI flag so blind and context-aware runs remain separate.
- [ ] Add prompt notes for historical German spelling and Copiale-specific
  caution, but keep them concise and evidence-driven.
- [ ] Add fake-provider smoke tests for any new workflow gate before live runs.

## Milestone 5: Generalize Beyond Copiale

- [ ] Classify benchmark cases by cipher family and current solver capability:
  simple substitution, homophonic, nomenclator, transposition, polyalphabetic,
  OCR/transcription-heavy, and mixed/unknown.
- [ ] For each unsupported family, decide whether Decipher should:
  - implement a native baseline,
  - wrap an external baseline,
  - expose diagnosis-only tooling,
  - or mark the family out of scope for now.
- [ ] Extend frontier packets by family only when there is a meaningful
  baseline and a clear expected behavior.
- [ ] Keep live-agent smoke small; use fake-provider tests for loop mechanics
  and opt-in live runs for capability checks.

## Open Questions

- Is Copiale primarily a better-search problem, or are nulls/codewords the
  dominant obstacle?
- Does German continuous n-gram scoring help on Copiale text, or does
  historical spelling/transcription mismatch overwhelm it?
- What context is fair to provide in context-aware parity, and can external
  baselines consume comparable context?
- Which non-Borg benchmark family should become the second generalization
  target after Copiale?
