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

### Priority 3: Native Tool Parity

- [ ] Harden `search_homophonic_anneal` with seed sweeps against Zenith and zkdecrypto-lite.
- [x] Add top-N candidate support for homophonic annealing.
- [x] Add an automated-only/no-LLM CLI mode that runs native techniques and writes zero-cost artifacts.
- [x] Run no-LLM automated preflight before LLM iteration 1 by default, with a branch and prompt summary.
- [x] Upgrade English simple-substitution automated solving with bijective continuous n-gram annealing.
- [x] Add a chunkable automated-only parity matrix runner for seed, length, family, benchmark split, and external-tool comparisons.
- [ ] Add model provenance and acquisition docs for continuous n-gram models.
- [ ] Add or train language-specific continuous models for Latin, German, French, and Italian.
- [ ] Benchmark simple substitution with and without word boundaries across English/French/German/Italian.
- [ ] Assess Borg and Copiale failures by tool gap: language model, context loading, nomenclator/codeword behavior, historical spelling, or prompt choice.
- [ ] Assess external-tool cipher families and explicitly mark unsupported families rather than letting the agent fake confidence.

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
