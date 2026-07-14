# Spec: Investigator Mode — INV-0 (Compiled Analysis Pack)

Parent design: `docs/specs/investigator_mode_v3_design.md` (sections **I1, I2 (P8), I5**,
milestone **INV-0**, decisions D5/D9/D10). Background:
`docs/unknown_cipher_investigator_mode.md` (battery categories),
`docs/beale_statistical_fingerprinting.md` (P8 reference analyses). Review addressed:
`docs/specs/investigator_inv0_review_findings.md` (16 findings; dispositions in the **Revision
notes** section — every finding is fixed or explicitly deferred). Spec author: Fable; implementer:
coding sub-agent; this revision goes to a SECOND spec-review pass before a coder starts. INV-0 has
**no v3 dependency** — LLM-free local compute, consumable by v2; no API keys or model spend
anywhere.

Conventions (binding): baseline suite = re-record the actual counts of
`PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` **at implementation start** (slices are
landing on main; do not trust counts in older specs); no pre-existing test may change outcome.
Cited `file:line` may be stale — locate constructs by quoted identifiers/strings, never by line
number. **No commits.** Report deviations rather than silently improvising. The working tree
carries unrelated in-flight changes (`CLAUDE.md`, `TOOLS.md`, model metadata): touch only files
this spec names, additively where it says additive. **Do not modify
`src/investigation/__init__.py`** (M1-scoped, a three-spec merge point — finding 13); import
`families`/`diagnosis` by module path.

Six deliverables: (1) family registry + discriminator inventory; (2) panel layer + P8 numeric
battery; (3) shuffle-null baselines; (4) `island_report`; (5) ranked `DiagnosisReport` with an
executable scoring formula, CLI `decipher diagnose`, an unsolved-record reader, and v2 wiring;
(6) `scripts/research/beale_report.py`. Non-goals: the investigator loop/state/ledger/notes/
episodes/plans/human channel (INV-1+); corpus key-text search (a separate ledger-gated layer);
`BenchmarkLoader` changes.

**Calibration provenance (binding).** Every atom weight, shape threshold, numeric threshold, and
fixture number in Parts 5/6/9 and the acceptance criteria was DERIVED by executable simulation
against the real `analysis.cipher_id` (fingerprint priors, `_normalized_entropy`,
`_chi2_vs_uniform`) and the real `beale_1`/`beale_3` streams, not hand-computed. The reproducible
source of truth is `scripts/research/calibrate_inv0_scoring.py` (stdlib + repo imports only, no
network/LLM, ~6 s, deterministic across processes). It implements the Part-6 catalog + scoring
formula and the Part-5 island rule literally in two modes — `ORIGINAL` (reproduces the
second-review failures) and `FIXED` (the numbers below) — and evaluates all nine Part-9 fixtures +
acceptance 2/5 + the Part-5 Monte-Carlo. The implementer MUST keep this script green: any change to
a weight/threshold/atom re-runs it and all nine fixtures + acceptance 2/5 + Part-5 checks must
still pass. Run: `PYTHONPATH=src .venv/bin/python scripts/research/calibrate_inv0_scoring.py`.

## Part 0 — Data availability findings (verified 2026-07-14; binding)

- Beale records exist in the **unsolved area only**:
  `~/Dropbox/src2/cipher_benchmark/benchmark/unsolved/manifest/records.jsonl` has `beale_1` (520
  tokens, 298 unique, max value 2906) and `beale_3` (618 tokens, 263 unique), `cipher_type:
  ["book_cipher", "numeric_code", "unknown"]`; splits in `unsolved/splits/beale_tests.jsonl`.
- Data shape: `transcription_canonical_file` (path relative to `unsolved/`) is **single-line
  whitespace-separated decimal numeric tokens, no word boundaries**; sidecar metadata JSON has
  counts + sha256. Manifest fields the reader needs: `related_records` (list of dicts with
  `record_id`, `relationship`, `area`) and `rights_class` (the value `"hold_for_review"` gates
  content out — Part 6).
- `BenchmarkLoader` (`src/benchmark/loader.py`) reads only `manifest/records.jsonl` — the
  unsolved manifest is invisible to it (the zodiac340 incident). **Decision: do NOT extend
  BenchmarkLoader**; INV-0 adds a separate lightweight reader (Part 6). `loader.py` is untouched.
- Beale 2 is in NO manifest, but a local public-domain copy exists:
  `other_tools/azdecrypt-src/AZdecrypt/Ciphers/Substitution/Beale 2.txt` (762 tokens as shipped —
  commonly cited count is 763; treat as a transcription variant; record actual count + sha256).
  No Declaration of Independence key text exists anywhere locally (Part 8 bundles one).
- A stored borg_0077v bad-basin decode exists for the anti-pareidolia fixture:
  `artifacts/automated_only/borg_single_B_borg_0077v/457acea3e481.json`, field `decryption`
  (Latin word-island gibberish, `word_accuracy` 0.0).

## Part 1 — Family registry + discriminator inventory (`src/investigation/families.py`, new)

- `@dataclass(frozen=True) FamilySpec`: `id`, `display_name`, `parent: str|None`, `role` ∈
  {`primary`, `subtype`, `modifier`} (drives hierarchy-aware ranking, Part 6), `solver_status` ∈
  {`solves_automated`, `agent_assists`, `diagnoses_only`, `planned`, `unsupported`},
  `detectors: tuple[str, ...]` (atom observation names bearing on it), `discriminators:
  tuple[str, ...]` (DiscriminatorSpec ids), `confusable_with: tuple[str, ...]` (symmetric),
  `notes: str`, `min_tokens_diagnose: int`.
- `@dataclass(frozen=True) DiscriminatorSpec`: `id`, `description`, `splits: tuple[str, str]`,
  `depends_on_panels: tuple[str, ...]` (panels whose `ok` status marks the discriminator "run" —
  Part 6), `tool: str|None` (existing v2 tool when one exists), `status` ∈ {`available`,
  `planned`}.
- `FAMILY_REGISTRY` families: primaries `monoalphabetic_substitution`, `homophonic_substitution`,
  `polyalphabetic_periodic`, `transposition`, `transposition_homophonic`,
  `fractionation_transposition`, `playfair`, `polygraphic_substitution`, `nomenclator_codebook`,
  `numeric_book_cipher`, `plaintext_or_hoax`, `unknown_custom`; subtypes `quagmire_keyed` (parent
  `polyalphabetic_periodic`), `numeric_word_position` / `numeric_char_position` /
  `numeric_skip_nth_word` (parent `numeric_book_cipher`); modifier `nulls_noise_layer` (`notes`:
  composes with any family; never in primary ranking).
- `solver_status` set from the ACTUAL tool inventory at implementation time (homophonic en →
  `solves_automated` via `zenith_native`; `quagmire_keyed` → `agent_assists` via
  `search_quagmire3_keyword_alphabet`; all `numeric_*` and `plaintext_or_hoax` →
  `diagnoses_only`; `unknown_custom` → `unsupported`); record the table in the module docstring.
- **`confusable_with` — enumerated, symmetric (finding 4; load-bearing for strong-margin and
  verdict rule (d)).** Each listed pair has a covering DiscriminatorSpec, so `confusable_with` and
  `DISCRIMINATOR_REGISTRY` splits are the same relation (a pair is listed iff a discriminator can
  tell them apart) — this prevents rule (d) from firing vacuously on a pair with no discriminator.
  `monoalphabetic_substitution ↔ {transposition, homophonic_substitution, polyalphabetic_periodic}`;
  `homophonic_substitution ↔ {monoalphabetic_substitution, transposition_homophonic}`;
  `polyalphabetic_periodic ↔ {monoalphabetic_substitution}`;
  `transposition ↔ {monoalphabetic_substitution}`;
  `transposition_homophonic ↔ {homophonic_substitution}`;
  `numeric_book_cipher ↔ {plaintext_or_hoax}`. All other primaries have empty `confusable_with`.
  DERIVED consequence: `polyalphabetic_periodic` does NOT list `homophonic_substitution` (no
  poly↔homo discriminator exists) — so fixtures B and C stay `confident` (rule (d) cannot fire on
  their top-2), and `numeric_book_cipher` does NOT list `nomenclator_codebook` (no cheap
  discriminator; separation needs key-text search, out of INV-0 scope) — so fixture E and the
  Beale acceptance stay `confident`/ranked-correctly. The genuine poly-vs-homo and book-vs-codebook
  ambiguities are handled by evidence margin, not by forcing an uncertain verdict.
- `DISCRIMINATOR_REGISTRY` — the concrete inventory (finding 2; required by fixtures):

  | id | splits | depends_on_panels | tool | status |
  |---|---|---|---|---|
  | `disc_mono_transp` | mono ↔ transposition | `order_layout` | `observe_transform_suspicion` | available |
  | `disc_mono_homophonic` | mono ↔ homophonic | `shape`,`frequency` | `observe_homophone_distribution` | available |
  | `disc_sub_periodic` | mono ↔ polyalphabetic_periodic | `periodicity` | `observe_periodic_ic` | available |
  | `disc_periodic_quagmire` | polyalphabetic_periodic ↔ quagmire_keyed | `periodicity` | `search_quagmire3_keyword_alphabet` | available |
  | `disc_homo_transphomo` | homophonic ↔ transposition_homophonic | `order_layout`,`polygraphic` | `observe_transform_suspicion` | available |
  | `disc_numeric_book_hoax` | numeric_book_cipher ↔ plaintext_or_hoax | `numeric_code` | None | available |
  | `disc_book_word_char` | numeric_word_position ↔ numeric_char_position | `numeric_code` | None | available |

- `SUSPICION_TO_FAMILY` maps the six `cipher_id.py` `suspicion_scores` keys onto registry ids
  (`polyalphabetic_vigenere` → `polyalphabetic_periodic`; rest 1:1). The seventh key `"unknown"`
  (returned as `{"unknown":1.0}` for <10 tokens) maps to nothing — dropped, emits no prior atom
  (finding 1).
- `_validate_registry()` at import: parents/splits reference existing ids; confusables symmetric;
  every primary lists ≥ 1 discriminator; ids unique.

## Part 2 — Statistical panel layer (`src/analysis/panels.py`, new)

- `@dataclass PanelResult`: `panel`, `status` ∈ {`ok`, `not_computable`}, `reason: str|None`,
  `measurements: dict`, `atoms: list[dict]`, `reliability` ∈ {`high`, `low`, `not_computable`}.
- **Atom dict** (plain dicts; NOT the M1 `investigation.state.EvidenceEntry` — that unification
  is INV-1): `{observation, panel, weight: float, measurement: dict, baseline: dict|None,
  reliability, interpretation, supports: [family_id], weakens: [family_id]}`. `baseline` (for
  order-dependent atoms) = `{tail: "upper"|"lower"|"two_sided", percentile, p_value, n_shuffles,
  null_mean, null_std, seed}` (Part 3). **Reliability rule** (replaces the old
  "null_percentile<95→low", which mislabeled significant low-tail statistics — finding 6a): a
  *baselined* atom is `high` iff its `baseline.p_value ≤ 0.05` in its declared tail, else `low`;
  a *structural* atom (no baseline) declares `high`/`low` from a documented threshold margin.
- **Signature (findings 4, 10)**: `panel_<name>(tokens, *, alphabet_size, alphabet_class,
  language, word_group_count=0, line_lengths=None, numeric_values=None, letter_rendering=None,
  related_profile=None, max_period=26, rng_namespace="") -> PanelResult`. `alphabet_class` ∈
  {`letters`, `numeric`, `symbols`, `mixed`}; `letter_rendering: str|None` = the normalized A–Z
  string (source alphabet is letters), produced by the CLI/tool layer. Unused kwargs ignored.
  Token-count gates → `not_computable` with a `reason`.
- Reuse inventory (adapters, not reimplementations; "NEW" = genuinely new math):

  | Panel | Reuses | NEW delta | Gate |
  |---|---|---|---|
  | `shape` | inputs | symbol-inventory drift χ² across halves/lines + Part-3 percentile | — |
  | `frequency` | `frequency.py`; `cipher_id._chi2_vs_uniform`/`_normalized_entropy` | — | — |
  | `repetition` | `cipher_id.kasiski_report` | expected repeat count under multinomial null + excess (baselined) | — |
  | `periodicity` | periodic-IC block of `cipher_id.compute_cipher_fingerprint`, Kasiski | shuffle-null percentile of best-period IC peak (baselined:upper) | ≥40 tok |
  | `order_layout` | `transform_search.inspect_transform_suspicion` | monogram-identity χ² vs language + quadgram loglik via `ngram.py` → `letters_unsubstituted`/`letters_substituted` atoms | `letters` + `letter_rendering`, else `not_computable` |
  | `nulls_noise` | `homophonic_nulls.py` mask helpers | top-k frequent-symbol IC trajectory (minimal in INV-0) | — |
  | `polygraphic` | doubled-digraph rate from `cipher_id` | digraph-IC/monogram-IC ratio; 25/36-symbol coord check | — |
  | `numeric_code` | — | NEW module, Part 4 | `numeric`, else `not_computable` |
  | `language` | `language_guesser.py` | `language_provisional` atom | `letters`/`mixed`, else `not_computable` (finding 15) |

- `run_battery(...) -> dict[str, PanelResult]` runs all nine with per-panel exception isolation
  (a panel crash → `not_computable`, `reason="panel_error:<Exc>"`; never aborts the battery).
  Panel names above are the `depends_on_panels` keys used by the Part-1 discriminators.

## Part 3 — Shuffle-null baselines (`src/analysis/null_baseline.py`, new)

- **Canonical integer encoding (finding 6c)**: `encode_tokens(values) = b"".join(v.to_bytes(8,
  "big") for v in values)`. Used for the seed hash, the Part-3 cache key, and Part-6 `view_hash`
  — one definition, reused (Beale max 2906 no longer overflows a byte encoding).
- `null_percentile(statistic_fn, tokens, *, tail="upper", n_shuffles=1000, statistic_name,
  namespace="") -> dict` (findings 6a/6b): frequency-preserving permutation of a copy; returns
  `{observed, tail, percentile, p_value, n_shuffles, null_mean, null_std, seed}`. `p_value`:
  upper = (#draws ≥ observed + 1)/(n+1); lower = (#draws ≤ observed + 1)/(n+1); two_sided =
  min(1, 2·min(upper,lower)). **Default `n_shuffles=1000`**; any atom or acceptance claim of
  "percentile ≥ 99.9" MUST use `n_shuffles ≥ 1000` (with n draws the tightest reportable bound is
  `p < 1/(n+1)`; when observed exceeds all draws, report `p_value` and that bound, not a bare
  99.9). Callers pass the tail matching the atom's expected direction.
- **Deterministic seeding**: `seed = int.from_bytes(sha256(statistic_name.encode() + b"|" +
  namespace.encode() + b"|" + sha256(encode_tokens(tokens)).digest()).digest()[:8], "big")` fed
  to a local `random.Random(seed)`. Never salted `hash()`, never global `random`. Identical
  across processes.
- Cache keyed `(statistic_name, namespace, sha256(encode_tokens(tokens)), n_shuffles, tail)`;
  `namespace` = the future D10 per-investigation segregation hook, defaulted `""`.
- The Part-4 `modulo` Monte-Carlo (a *parametric* uniform-draw null, not a shuffle) uses the same
  seed scheme with `statistic_name=f"modulo_m{m}"` and labels `baseline.kind="parametric"`
  (finding 6, same-class gap).

## Part 4 — P8 numeric battery (`src/analysis/numeric_code.py`, new)

- `parse_numeric_ciphertext(text) -> list[int]` (whitespace/comma-separated decimal tokens;
  `ValueError` naming the first bad token) and `is_numeric_ciphertext(text) -> bool`.
- `numeric_code_battery(values, *, related_profile=None, rng_namespace="") -> dict` with groups:
  - `basic`: count, unique, min, max, repeated-token rate, top-10 values with counts, gap
    histogram over sorted unique values.
  - `digits` + `benford`: first-/last-digit distributions; first-digit chi² vs uniform and vs
    Benford (`p(d) = log10(1 + 1/d)`, d = 1..9); last-digit chi² vs uniform (0..9);
    `epsilon_benford_deviation` = sup-norm `max_d |p_obs(d) − p_benford(d)|` (smallest ε with the
    empirical first-digit law ε-close to Benford — our operationalization of Wase 2021);
    total-variation distance. **Deviation rule**: if the implementer can consult Wase 2021 and
    its operative statistic differs materially, add the paper's variant as an extra field and
    report a deviation.
  - `repeats` + `runs`: repeated numeric n-grams (n = 2..3) with positions + expected count under
    a multinomial null (baselined:upper); longest monotone run and consecutive-integer-pair count
    (baselined:upper).
  - `modulo`: for m ∈ 2..12 and 26: χ² of `values mod m` vs uniform with the Part-3 parametric
    percentile (uniform draws on [min, max], same n — no scipy).
  - `key_length` + `front_loading`: required key-text length — word-position → `max(values)`
    words, char-position → `max(values)` chars, skip-Nth feasibility for N ∈ 2..5
    (`N × max(values)` words); quintile histogram of `values/max`; `front_loading_index` =
    (share ≤ max/5) / 0.2 (> 1 ⇒ book-order skew; the excess is baselined:upper vs a uniform draw).
  - `flags` (finding 8 — split, each with an explicit k-of-n rule; `plausibility` ∈
    {`supported`, `neutral`, `counterindicated`}):
    - `word_first_letter_book_cipher` / `char_position_book_cipher` / `skip_nth_word`: supported
      iff the corresponding `key_length` hypothesis is feasible AND (front-loading present OR
      repeat-rate below multinomial expectation) — 2-of-2.
    - `shared_key_with_companion` (only if `related_profile`): supported iff value-inventory
      Jaccard ≥ t AND frequency-profile correlation ≥ r — 2-of-2.
    - `independent_random_like`: supported iff ≥ 2 of {last-digit χ² p > 0.05,
      `epsilon_benford_deviation` ≥ the Benford-inconsistency threshold (NO Benford decay),
      repeated-token rate within ±1σ of multinomial expectation}. **Monotone-run excess is NOT a
      condition here** — a seeded iid uniform stream is distributionally its own shuffle → no run
      excess → this flag correctly fires while `structured_hoax_artifact` stays `neutral`
      (resolves finding 8's fixture contradiction).
    - `structured_hoax_artifact`: supported iff ≥ 2 of {monotone-run excess baselined p ≤ 0.05,
      digit-preference χ² p ≤ 0.05, consecutive-run excess p ≤ 0.05}. Monotone runs are
      supporting-only here (the Gillogly fabrication signature), never for random-like.
  - `related_profile_distance` when `related_profile` given (this function's own
    `basic`/`digits`/`benford` measurements for the reference — Beale 2 only, never ground truth).
- `alphabetical_run_report(letters, *, n_shuffles=1000, rng_namespace="") -> dict`: longest
  non-decreasing and strictly-increasing alphabetical runs, offsets, baselined:upper percentiles
  at `n_shuffles=1000` (the Gillogly-1980 artifact detector; used by Part 8 on decoded
  first-letter streams).

## Part 5 — `island_report` (`src/analysis/coherence.py`, new + resources)

- `island_report(text, language="en", *, word_set=None, freq_rank=None, segmented=None) -> dict`
  (finding 12: accepts a precomputed `segmented` from `analysis.segment.segment_text`; when
  `None`, segments internally via the `finalist_validation._load_word_set`/`_load_word_list`
  path — keeps the standalone signature intact). Fields: `dict_rate`, `function_word_rate`
  (closed-class fraction of segmented dictionary words), `word_bigram_available: bool`,
  `word_bigram_loglik: float|None`, `word_bigram_order_p: float|None`,
  `word_bigram_order_significant: bool`, `longest_coherent_span` (see below), `word_count`,
  `pseudo_word_fraction`, `verdict` ∈ {`coherent`, `word_islands`, `gibberish`}.
- **Order-sensitive span (finding 3 / review-2 finding 2 — DERIVED, calibration §PART 5).** The
  v1 "either word is a function word" OR-rescue is fatal at English function-word density (~47%):
  Monte-Carlo of shuffled prose gave longest spans 14–33 with 240/240 shuffles ≥ 5. TWO derived
  fixes are applied (review-2 options 2 **and** 3, belt-and-suspenders):
  - **Attested-junction floor (option 2).** Walking the segmented sequence, a *run* extends across
    `(w_k, w_{k+1})` while both words are dictionary words AND `adjacent_ok` = `attested_bigram(w_k,
    w_{k+1})` OR `function_word(w_k)` OR `function_word(w_{k+1})`; a run only *qualifies* (contributes
    to `longest_coherent_span`) once it carries **≥ 2 attested word-bigram junctions**
    (`ATTESTED_PER_SPAN = 2`). Shuffled prose is function-word-dense but attestation-sparse → its
    long function-glued runs almost never accumulate 2 attested pairs → spans collapse; ordered prose
    (`IN THE COUNTY OF …`) carries several attested function bigrams and qualifies. Derived spans:
    ordered prose 8–22, B2 self-check 8, shuffled fixed fixture 25 (but see order test).
  - **Order-significance test (option 3, the primary anti-shuffle guard).**
    `word_bigram_order_p` = Part-3 upper-tail shuffle-null p that the mean word-bigram loglik
    (from the attested-bigram log-prob resource, floored) exceeds a frequency-preserving
    permutation of the same words; `word_bigram_order_significant` = `word_bigram_order_p ≤ 0.05`.
    Ordered prose/B2 → p ≈ 0.003–0.03 (significant); a shuffled input is a draw from its own null →
    p ≈ 0.5 (not significant). Calibration `n_shuffles = 300`; production default 1000 per Part 3.
  - When no word-bigram resource exists for `language`, `word_bigram_available = false`,
    `word_bigram_order_significant = false`, and span uses the function-word path only.
- **Verdict** (per-language constants; `en` defaults): `coherent` iff `dict_rate ≥ 0.75` AND
  `function_word_rate ≥ function_word_min` AND `longest_coherent_span ≥ 5` AND
  `word_bigram_available` is True AND `word_bigram_order_significant` is True — the `coherent`
  verdict **requires a real transition resource AND passes the order-significance test**, so
  `la`/`de` (no bigram resource in INV-0) and shuffled English (order p ≈ 0.5) can never falsely
  read `coherent`; `gibberish` iff `dict_rate < 0.35` OR (`function_word_rate < 0.08` AND
  `longest_coherent_span ≤ 2`); else `word_islands`. `function_word_min` = **0.25** for `en`,
  **0.10** for `la`/`de` (function-word-poorer; DERIVED to pass Part-9). **Derived checks
  (calibration §PART 5):** ordered English prose → `coherent` (dict 0.78–0.88, fn 0.50–0.55, span
  8–22, order_p 0.003); the B2 self-check plaintext → `coherent` (dict 0.78, fn 0.41, span 8,
  order_p 0.003 — the review-2 finding-2 hard gate holds under the stricter rule); shuffled English
  → not `coherent` (the fixed test fixture reads `word_islands`: span 25 but order_p 0.365; the
  Monte-Carlo false-coherent rate falls from 240/240 to ≈ 3–5 %, the nominal α floor); `la` (no
  bigram resource) → `word_islands`.
- Resources: `resources/function_words/{en,la,de}.txt` (~100–200 uppercase closed-class words
  each; one per line, `#` comments; curated by the implementer);
  `resources/word_bigrams/en_bigrams.txt` (`WORD1 WORD2 weight`, ordered high-frequency pairs —
  the count must be large enough that ordered prose reaches the ≥ 2-attested-junction floor; the
  calibration built a **4000-pair** model from the local corpus and that is the recommended floor,
  not the v1 "~300"; `la`/`de` omitted in INV-0 → those verdicts cannot be `coherent`, the
  intended conservative behavior). The same pairs feed BOTH the span attestation test AND the
  order-significance log-prob model (counts → log10 with a `0.1/total` floor). Loading is
  `lru_cache`d; missing files degrade gracefully. (This admits the bigram/transition term to the
  verdict — a deliberate change from the v1 spec's "enrichment only," required by finding 3 and
  hardened by review-2 finding 2/3.)

## Part 6 — DiagnosisReport, unsolved reader, CLI

**`src/investigation/diagnosis.py`** (new):
- `view_hash(tokens, *, transform_pipeline=(), null_mask=(), segmentation=()) -> str` = sha256 of
  `encode_tokens(tokens) + b"\x1f" + json.dumps([transform_pipeline, null_mask, segmentation],
  sort_keys=True).encode()` (finding 6 — the D5 dedup key defined once; INV-0 calls it with empty
  extras).
- `diagnose(tokens, *, alphabet_size, alphabet_class, language="en", word_group_count=0,
  line_lengths=None, numeric_values=None, letter_rendering=None, related_profile=None,
  max_period=26) -> DiagnosisReport` (findings 4, 10). Ciphertext-derived inputs ONLY (firewall by
  signature — no record objects, no context text). Embeds `cipher_id.compute_cipher_fingerprint
  (..., max_period=max_period)` verbatim (`fingerprint` field); `run_battery` supplies atoms;
  `max_period` forwarded to fingerprint + P3/P4.
- **Scoring formula (finding 1 — executable).** Atoms carry `weight` and reliability; let
  `rel(a) = 1.0` if `a.reliability=="high"` else `0.4`. For each family F:
  `raw(F) = Σ_{a: F∈supports} weight(a)·rel(a) − Σ_{a: F∈weakens} weight(a)·rel(a)`;
  `score(F) = clamp(raw(F), 0, 1)`. Primary ranking sorts by `(−score, family_id)` (deterministic
  tie-break). `fingerprint_prior:<family>` atoms (one per mapped suspicion key) carry
  `weight = 0.5·suspicion`, `reliability="low"` (⇒ effective 0.4·weight); the `"unknown"`
  suspicion key emits none.

  **Shape statistic (DERIVED, review-2 finding 3).** `_normalized_entropy` (the real
  `cipher_id.py` plug-in estimator, normalized by log2 of OBSERVED unique symbols) does NOT
  separate the regimes at n≈300: English letter streams measure 0.90–0.95 and Vigenère 0.95–0.98,
  and the tails OVERLAP (mono reaches 0.949, period-5 Vigenère dips to 0.9475), so the v1
  thresholds (`<0.85` peaked, `>0.90` flat) are dead / coin-flip on real data (confirmed by the
  calibration `ORIGINAL` mode). The shape atoms therefore use a **per-token χ² flatness** statistic
  `flatness = _chi2_vs_uniform(counts, n) / n` (identically `≈ k_obs·IC − 1`, the excess collision
  probability over uniform — sample-size robust). Measured regimes (calibration, 6 texts × 3
  seeds): mono/transposition letters `0.30–0.74`; period-5 Vigenère `0.12–0.36`; homophonic
  (realistic random homophone assignment) `0.13–0.21`; iid-uniform `≈0.05`. Thresholds: `peaked` =
  `flatness > 0.28 ∧ unique ≤ 28`; `flat_high_entropy` = `flatness < 0.10`. NOTE (DERIVED):
  realistic homophonic and Vigenère OVERLAP on `flatness` (~0.13–0.21), so `flat_high_entropy` is
  NOT the homophonic discriminator — `large_symbol_inventory` is; `flatness < 0.10` isolates only
  genuinely near-uniform streams (not load-bearing for any fixture; suppressed for numeric class).
  `peaked`'s `unique ≤ 28` guard keeps it off homophonic; it may fire on Vigenère harmlessly
  (Vigenère's `depressed_ic` + `periodic_ic_recovery` crush the mono/transposition it feeds).

  **Numeric-class gate (DERIVED, review-2 finding 1).** For `alphabet_class == "numeric"` the
  substitution-family frequency/shape atoms (`ic_near_language_reference`, `depressed_ic`,
  `peaked_monogram_shape`, `flat_high_entropy`, `large_symbol_inventory`) are **NOT emitted** —
  treating integer indices as a substitution alphabet is the category error that let dense numeric
  streams score homophonic (`ORIGINAL` mode: beale_1 homophonic 0.85 ties numeric_book 0.85). Their
  counter-signal is carried by the new scored atom `numeric_inconsistent_with_substitution`, so
  substitution rows still render explicit numeric counterevidence.

  **Baselined-atom emission (DERIVED, review-2 finding 6).** A baselined atom
  (`periodic_ic_recovery`, `front_loading_present`, `structured_hoax_artifact`) is **emitted ONLY
  when significant** (its Part-3 baseline rejects at `p ≤ 0.05` in its declared tail); when emitted
  it is `high`. It is NOT emitted-and-graded-low — otherwise an insignificant
  `periodic_ic_recovery` silently adds `0.45·0.4 = 0.18` to every stream's poly score (the
  `ORIGINAL` defect: fixture A then ranks poly, not mono). The periodicity structural pattern
  (`best_period_ic ≥ lang_ic_ref − 0.015 ∧ best_period_ic − ic > 0.010`) is a necessary pre-filter,
  but a **frequency-preserving shuffle null is the emission gate**: it suppresses the
  multiple-testing IC peak on monoalphabetic English (p ≈ 0.26) yet fires on real period-5 Vigenère
  (p ≈ 0.005). Do NOT gate periodicity on the v1 `cols ≥ 25` heuristic — the detected best period is
  often a harmonic (a period-5 signal detected at k=20 → 15 columns) and the heuristic wrongly kills
  it. `front_loading_present` uses a **parametric** iid-uniform-draw null (a frequency-preserving
  shuffle is order-invariant for this value-multiset statistic): uniform stream index 1.08 → p ≈ 0.3
  (suppressed), Beale index 3.7–4.3 → p < 0.005 (emitted).

  **Atom catalog** (weights + thresholds DERIVED by `calibrate_inv0_scoring.py`; the FIXED-mode
  numbers reproduce all nine fixtures + acceptance 2/5 — see Revision notes):

  | observation | panel | weight | rel basis | supports | weakens |
  |---|---|---|---|---|---|
  | `ic_near_language_reference` | frequency | 0.25 | structural (\|Δic\|<0.012); not numeric | mono, transposition | — |
  | `depressed_ic` | frequency | 0.25 | structural (Δic<−0.02); not numeric | polyalphabetic_periodic, homophonic | mono, transposition |
  | `peaked_monogram_shape` | frequency | 0.30 | structural (χ²-flatness>0.28 ∧ unique≤28); not numeric | mono, transposition | homophonic |
  | `flat_high_entropy` | frequency | 0.30 | structural (χ²-flatness<0.10); not numeric | homophonic | mono |
  | `large_symbol_inventory` | shape | min(0.45, 0.45·(unique−26)/20) | structural (unique>26); not numeric | homophonic | mono |
  | `periodic_ic_recovery` | periodicity | 0.45 | baselined:upper — emit only when shuffle-null p≤0.05 (then high) | polyalphabetic_periodic | mono, transposition |
  | `letters_unsubstituted` | order_layout | 0.55 | structural (monogram χ² vs lang <300 ∧ quadgram mean-loglik <−6.0) | transposition | mono, homophonic |
  | `letters_substituted` | order_layout | 0.35 | structural (monogram χ² vs lang >500 ∧ shape peaked) | mono, homophonic | transposition |
  | `numeric_token_stream` | numeric_code | 0.35 | structural (alphabet_class numeric) | numeric_book_cipher, nomenclator_codebook | ALL 8 substitution primaries † |
  | `numeric_inconsistent_with_substitution` | numeric_code | 0.50 | structural (numeric ∧ unique/token>0.4) | — | ALL 8 substitution primaries † |
  | `book_keylength_plausible` | numeric_code | 0.30 | structural (1≤min ∧ max≤100000) | numeric_book_cipher | — |
  | `front_loading_present` | numeric_code | 0.20 | baselined:upper (uniform-draw null) — emit only when p≤0.05 (then high) | numeric_book_cipher | plaintext_or_hoax |
  | `independent_random_like` | numeric_code | 0.36 | structural — ≥2 of: last-digit χ² p>0.05; ε_benford≥0.12; repeat-rate within ±1σ of iid-uniform null | plaintext_or_hoax | numeric_book_cipher, nomenclator_codebook |
  | `structured_hoax_artifact` | numeric_code | 0.35 | baselined — ≥2 of: monotone-run excess p≤0.05; LAST-digit χ² p≤0.05; consecutive-run excess p≤0.05 | plaintext_or_hoax | numeric_book_cipher |
  | `book_word_position_signature` | numeric_code | 0.40 | structural (P8 word_first_letter flag supported) | numeric_word_position | — |
  | `book_char_position_signature` | numeric_code | 0.40 | structural (P8 char_position flag supported) | numeric_char_position | — |
  | `book_skip_nth_signature` | numeric_code | 0.40 | structural (P8 skip_nth flag supported) | numeric_skip_nth_word | — |

  † **ALL 8 substitution primaries** = `monoalphabetic_substitution`, `homophonic_substitution`,
  `polyalphabetic_periodic`, `transposition`, `transposition_homophonic`,
  `fractionation_transposition`, `playfair`, `polygraphic_substitution` (finding 1: the v1
  `numeric_token_stream` weakens list omitted the last three; both numeric atoms now weaken all
  eight). `book_keylength_plausible` (DERIVED) supports `numeric_book_cipher` only and no longer
  weakens `plaintext_or_hoax`: feasibility of a key-text length does not argue against a hoax; only
  positive book-usage evidence (`front_loading_present`) does — required so uniform-random fixture
  (iv) keeps `plaintext_or_hoax` above `numeric_book_cipher`. The three `book_*_signature` subtype
  atoms (finding 7) make sibling margins non-vacuous — without them every `numeric_*` subtype scores
  0.

  `peaked_monogram_shape` and `ic_near_language_reference` support BOTH mono and transposition
  (both preserve letter frequencies); only `order_layout` (needs `letter_rendering`) separates
  them — the mechanism that makes fixture (iii) uncertain when rendering is withheld. For
  `alphabet_class=="numeric"` every substitution-family row renders
  `numeric_inconsistent_with_substitution` (+ `numeric_token_stream`) as the mandatory
  counterevidence naming the unique-count/max-value inconsistency.
- **Hierarchy-aware ranking (finding 7)**: `ranked` sorts **primaries only**; modifiers →
  separate `modifiers` list; subtypes scored the same way and reported under their parent's
  `subtypes` (sibling margin), never in primary ranking — the parent represents the family.
  Top-two/margin/confusable comparisons use primaries only, so a cleanly-diagnosed book cipher is
  no longer forced uncertain by its own parent+child both ranking high.
- Per-family `confidence`: `strong` iff `score ≥ 0.70` AND (# high-reliability supporting atoms)
  ≥ 2 AND `margin(F) ≥ 0.25`, where `margin(F) = score(F) − max score over
  F.confusable_with ∩ primaries`; `moderate` iff `score ≥ 0.45` AND ≥ 1 high-rel atom; `weak` iff
  `score > 0`; else `none`.
- **Discriminator-state predicate (finding 2 — fixed, not deferred)**: `discriminator_status(d,
  battery)` = `run` iff every panel in `d.depends_on_panels` has status `ok`; `unavailable` iff
  `d.status=="planned"`; else `pending`. Because `diagnose()` runs the full battery every turn, a
  discriminator is `pending` exactly when a depended panel was `not_computable` (e.g.
  `order_layout` with no `letter_rendering`, or a token-gated panel) — this is the INV-0 mechanism
  that fires the all-pending verdict rule without an INV-1 ledger.
- **Report verdict**: with `top1, top2` the two highest primaries, `uncertain` iff ANY:
  (a) `token_count < 60`; (b) `top2` exists AND `score(top1) − score(top2) < 0.15`; (c)
  `families[top1].discriminators` is non-empty AND every one is `pending` (the non-empty guard
  prevents the vacuous-`all()` bug — finding 2); (d) `top1, top2` mutually confusable AND every
  DiscriminatorSpec with `splits == {top1, top2}` has status `!= run`. Else `confident`. **Rule (c)
  is near-dead-code with the enumerated registry (review-2 finding 5): a `mono` top1 owns three
  discriminators (`disc_mono_transp`, `disc_mono_homophonic`, `disc_sub_periodic`), so a single
  `not_computable` panel never makes *all* of them pending — e.g. fixture (iii) (rendering withheld)
  goes uncertain via (b)+(d), not (c). Pin (iii)'s test to the observable outcome (`uncertain` AND
  `recommended_next[0] == disc_mono_transp`), not to mechanism (c).** An `uncertain` report leads
  `recommended_next` with a discriminator whose `splits` covers `{top1, top2}`; **FALLBACK when no
  discriminator covers the actual top-2 (finding 5): recommend the highest-priority discriminator
  that NAMES `top2` (the challenger) and is actionable now (all depended panels `ok` / status
  `run`), else one naming `top1`, else any naming either** — this surfaces the panel that best
  separates the pair (fixture (ii)'s homophonic-vs-poly top-2 has no covering discriminator, so the
  fallback returns `disc_sub_periodic`, whose periodicity panel is exactly the homophonic/poly
  test). Fixtures are parameterized so ≥ 2 primaries are live.
- `@dataclass DiagnosisReport`: `view_hash`, `fingerprint: dict`, `atoms`, `ranked:
  list[FamilyDiagnosis]` (primaries), `modifiers: list[FamilyDiagnosis]`, `verdict` ∈
  {`confident`, `uncertain`}, `battery_coverage: dict[str, str]` (panel → `ran` |
  `skipped(<reason>)`), `recommended_next: list[dict]` (`{discriminator_id, description, tool,
  splits}`), `token_count`, `language`, `alphabet_class`, `to_dict()`. `FamilyDiagnosis`:
  `family`, `score` (0–1 evidence weight, NOT a probability), `confidence` ∈ {`strong`,
  `moderate`, `weak`, `none`}, `evidence: list[str]` (atom observation names),
  `counterevidence: list[str]` (**mandatory, always rendered — `(none recorded)` when empty**),
  `subtypes: list[FamilyDiagnosis]`, `pending_discriminators: list[str]`, `solver_status`,
  `coverage_status: str` (hardcoded `"untested"` in INV-0; ledger is INV-1).
- `format_diagnosis(report) -> str` — ranked table with evidence AND counterevidence, verdict,
  coverage, recommended next; ≤ ~2500 chars.

**`src/benchmark/unsolved.py`** (new — the Part-0 decision): `@dataclass UnsolvedRecord`: `id`,
`source`, `cipher_type: list[str]`, `symbol_set`, `token_count`, `canonical_text`,
`metadata: dict`, `related_record_ids: list[str]`. `load_unsolved_record(benchmark_root,
record_id)` and `list_unsolved_record_ids(benchmark_root)` read `unsolved/manifest/records.jsonl`,
resolve `transcription_canonical_file` relative to `unsolved/`, and set `related_record_ids =
[r["record_id"] for r in manifest.get("related_records", [])]` (finding 14). NOT loaded:
`context_layers` text, `associated_documents`, `notable_attempts`, images. Any record or document
whose `rights_class == "hold_for_review"` has its content withheld (the canonical numeric stream
still loads for numeric targets; the loader test keys off `rights_class`).

**CLI** (`src/cli.py`, additive subcommand `diagnose`): `decipher diagnose [input]` (file or `-`
stdin) OR `--unsolved-id <record_id> --benchmark-root <path>`. Flags `--language` (default `en`),
`--json`, `--related-profile <numeric-file>` (profiled via P8), `--max-period` (default 26,
forwarded to `diagnose`). The CLI computes `alphabet_class` (numeric via `is_numeric_ciphertext`;
`letters` when the parsed alphabet is A–Z; else `symbols`/`mixed`) and `letter_rendering` (the
normalized A–Z string when `letters`). Numeric input keeps `numeric_values` for P8 and
dense-factorized token ids (sorted-unique rank) for generic panels; letter/S-token input uses the
existing crack-path parsers. Prints `format_diagnosis`; exit 0 even when `uncertain`.

## Part 7 — v2 wiring (all additive)

- `analysis/finalist_validation.validate_plaintext_finalist` gains ONE new top-level key
  `island_report` (Part 5), passing its already-computed `segmented` into
  `island_report(segmented=...)` (finding 12 — no second segmentation on the
  `transform_evaluation` finalist-menu hot path). **The validation block shape is pinned by many
  consumers** (`transform_evaluation.py`, `candidate_packet.py` pass-through, the `tools_v2.py`
  call site, `scripts/report_finalist_validation.py`, tests): every pre-existing key/value and the
  `validation_score`/`validation_label`/`recommendation` formulas stay byte-identical — write the
  pin test FIRST (capture full result dicts for one coherent and one word-island input; assert
  unchanged minus the new key).
- **CandidatePacket contract (finding 11)**: INV-0 keeps `island_report` nested inside the
  validation block and makes NO `CandidatePacket` schema change. This diverges from design D9
  ("CandidatePacket gains an `island_report` field"): null-mask packets carry `validation=None`
  and therefore get no island data under INV-0. **Recorded deviation** — a top-level packet field
  is deferred to the slice that first needs island data on null-mask packets; the design's D9 line
  should be amended to "validation-nested, with the null-mask gap noted." Flag to the design owner.
- New v2 tool `observe_diagnosis` in `src/agent/tools_v2.py`: definition + handler
  `_tool_observe_diagnosis(args)` — param `branch` (default active); runs `diagnose()` over the
  branch's current token order with the executor's language/alphabet; returns `report.to_dict()`
  plus the formatted render. Follow the existing `observe_cipher_id` handler's wiring pattern.
  Update `TOOLS.md` (name, description, param table, usage notes) and the `CLAUDE.md` tool-count
  line per the standing rule — both files carry unrelated in-flight edits; make surgical
  additions only.

## Part 8 — Beale reference resources + report script

- `resources/reference/beale2_numbers.txt`: copied from the local AZdecrypt file (Part 0),
  normalized to whitespace-separated tokens. `resources/reference/README.md` records provenance,
  actual token count, max value, and sha256 for every file in the directory.
- `resources/reference/beale_doi_key_words.txt` (finding 9): the Declaration of Independence as a
  one-word-per-line list in the Beale-2 numbering convention. **Marked solution-bearing**: a
  README banner + a module-level constant/guard so `diagnose()` and any candidate ranking refuse
  to load it (test-pinned — it is a key, not diagnostic input). Construction constraints: state
  the source edition, the normalization rules, and a sha256; iteration to satisfy the B2
  self-check is bounded to an **enumerated, individually-cited list of documented pamphlet
  numbering quirks** (no free numbering search). **B2 self-check gate**: first letters of the
  words indexed by the Beale-2 numbers decode to English whose first 14 letters are
  `IHAVEDEPOSITED` and whose `island_report` verdict is `coherent`; document every quirk applied.
  **Out-of-range rule (mandatory)**: B1 max 2906 exceeds any DoI word count (~1320); an index past
  the list end is **skipped, its position recorded**, and the skipped count reported in the
  Gillogly panel. If the self-check cannot be reached offline within the cited quirk list, stop
  and report a deviation — do NOT ship a Gillogly claim from a key that fails it.
- `scripts/research/beale_report.py` (stdlib + repo imports only; no network, no LLM — must not
  import from `services/`): loads beale_1/beale_3 via `load_unsolved_record` and Beale 2 from
  resources; emits (stdout + `--json`): (a) comparative P8 table for B1/B2/B3 (all Part-4 groups;
  B2 as `related_profile` for B1/B3); (b) Wase panel — Benford + epsilon-Benford deviations per
  cipher; (c) Campanelli panel with **concrete operationalizations (finding 16)**:
  `last_digit_uniform` = last-digit χ² p > 0.05; `unique_token_growth` = cumulative unique count
  at 10 positional deciles; `gap_structure` = histogram of consecutive sorted-unique gaps + max
  gap + count of gaps == 1; a **divergence call** for statistic S = the boolean `bool(Bn) !=
  bool(B2)` (or |value gap| past a stated threshold), reported per statistic; (d) Gillogly panel —
  B2 self-check result first (with skipped-index count), then decode B1/B3 first-letters under the
  DoI key and run `alphabetical_run_report` (`n_shuffles=1000`) + `island_report` on each. Zero
  access to any claimed B1/B3 plaintext (none exists in the benchmark).

## Part 9 — Tests (new files; extensions marked)

Fixture parameterization (finding 5; every winner DERIVED by `calibrate_inv0_scoring.py` FIXED
mode, not on paper — the score column below is the script's computed `score(top1)`/`score(top2)`).
The confident set and near-miss set are disjoint; fixture D (transposition, rendering present) and
near-miss (iii) (transposition, rendering withheld) differ only by `letter_rendering`. Streams are
built from local English corpus (mono/Vigenère/columnar/homophonic — homophonic uses **random**
homophone assignment, never cyclic, which would inject a spurious period) and seeded iid ints; the
book cipher is **early-biased** (front-loaded) so `front_loading_present` fires and
`numeric_book_cipher` reaches `strong` (≥0.70).

| fixture | params | class / rendering | derived top-2 (score) | verdict | trigger / winning atom |
|---|---|---|---|---|---|
| A mono | 26 letters, ~300 tok | letters / yes | mono 1.00 ≫ transposition 0.27 | confident/strong | `ic_near`+`peaked`+`letters_substituted` |
| B periodic | Vigenère period 5, 26 sym, ~300 | letters / yes | polyalphabetic 0.80 ≫ homophonic 0.32 | confident/strong | `depressed_ic`+`periodic_ic_recovery` |
| C homophonic | 52 sym, flat, ~300 | symbols / no | homophonic 0.90 ≫ polyalphabetic 0.34 | confident/strong | `large_symbol_inventory`+`depressed_ic` (NOT `flat` — realistic homophonic isn't flat enough) |
| D transposition | columnar, ~400 letters | letters / yes | transposition 1.00 ≫ mono 0.20 | confident/strong | `letters_unsubstituted`+`peaked`+`ic_near` |
| E numeric book | early-biased word-position, ~300 | numeric | numeric_book 0.85 ≫ nomenclator 0.35 | confident/strong | `numeric_token_stream`+`book_keylength`+`front_loading`; subtype `numeric_word_position` 0.40 |
| (i) short mono | 26 letters, 45 tok | letters / yes | mono 0.79, homophonic 0.10 | uncertain | (a) token_count<60 |
| (ii) light homophonic | 29 sym, ~150 tok | mixed / no | homophonic 0.44, polyalphabetic 0.34 | uncertain | (b) margin 0.10<0.15; top-2 is homophonic-vs-**poly** (both fire `depressed_ic`; a light homophonic depresses IC without large inventory — genuinely poly-confusable); `recommended_next[0]=disc_sub_periodic` (fallback) |
| (iii) transposition no-render | columnar, ~120, rendering withheld | symbols / no | mono 0.46, transposition 0.37 | uncertain | (b) margin 0.09<0.15 AND (d) `disc_mono_transp` pending (order_layout not_computable); `recommended_next[0]=disc_mono_transp` |
| (iv) uniform random | seeded iid uniform ints, ~300 | numeric | plaintext_or_hoax 0.36, numeric_book 0.29 | uncertain | (b) margin 0.07<0.15; `independent_random_like` fires, `structured_hoax_artifact` NEUTRAL |

- `tests/test_families_registry.py` (invariants; every primary ≥ 1 discriminator; `unknown` maps
  to nothing), `tests/test_panels.py` (per-panel `not_computable` gates incl. `order_layout`
  without rendering and P9 for numeric; adapter parity vs `cipher_id`; drift on a spliced stream;
  battery exception isolation).
- `tests/test_null_baseline.py`: determinism across two fresh processes (subprocess → identical
  dict); `encode_tokens` handles values > 255 (Beale 2906); a real periodic signal p ≤ 0.05
  upper-tail while its shuffle is null; a significantly LOW statistic gets p ≤ 0.05 lower-tail (not
  mislabeled — finding 6a); `n_shuffles=1000` supports the ≥ 99.9 bound.
- `tests/test_numeric_code.py`: Benford math vs hand values; epsilon sup-norm; parse errors; a
  synthetic word-position book cipher (built in-test from `beale_doi_key_words.txt`) →
  `word_first_letter_book_cipher=supported`; a seeded iid-uniform stream →
  `independent_random_like=supported` AND `structured_hoax_artifact=neutral` (finding 8);
  `alphabetical_run_report` finds a planted run at p < 1/1001.
- `tests/test_coherence.py`: coherent English prose → `coherent`; **shuffled** English prose (same
  words, permuted, fixed seed) → `verdict != coherent` (fails `word_bigram_order_significant`:
  order_p ≈ 0.5 — finding 3 / review-2 finding 2); the checked-in fixture
  `tests/fixtures/borg_0077v_basin.txt` (Part-0 `decryption` + provenance header; `la`) →
  `verdict != coherent`; random non-word letters → `gibberish`; `la` never returns `coherent` (no
  bigram resource → `word_bigram_available=false`); the B2 self-check plaintext stays `coherent`
  under the stricter rule (span ≥ 5 AND order_p ≤ 0.05); the precomputed-`segmented` path matches
  internal segmentation.
- `tests/test_diagnosis.py`: the five confident fixtures → correct top primary, `confident`, top
  family `strong`, evidence cites the intended atom (right reason); the four near-misses →
  `uncertain` via the tabulated trigger with `recommended_next[0]` either splitting the actual
  top-2 or (fallback, (ii)) naming an actionable discriminator for the top-2 ((iii)→`disc_mono_transp`,
  (ii)→`disc_sub_periodic`, (iv)→`disc_numeric_book_hoax`); hierarchy — E reports
  `numeric_word_position` under `subtypes` (score 0.40), not in `ranked` (finding 7);
  counterevidence always present (numeric fixtures' substitution rows cite
  `numeric_inconsistent_with_substitution`); every primary in `ranked`.
- `tests/test_unsolved_loader.py`: mini `unsolved/` tree in `tmp_path` (manifest line + canonical
  file + a `rights_class:"hold_for_review"` document); assert `related_record_ids` from
  `related_records`, and hold-for-review content withheld (keyed off `rights_class`).
- Extensions: `tests/test_finalist_validation.py` (Part-7 pin test + `island_report` presence +
  precomputed-segmentation reuse), `tests/test_cli.py` (`diagnose` numeric stdin `--json`),
  `tests/test_ground_truth_firewall.py` (`format_diagnosis` + `to_dict` for a solved benchmark
  cipher pass `assert_no_ground_truth_leak`), `observe_diagnosis` dispatch.

## Acceptance (all local compute; report numbers)

1. Full suite green: recorded baseline + new tests, zero pre-existing regressions; report
   before/after counts.
2. `decipher diagnose --unsolved-id beale_1 --benchmark-root
   ~/Dropbox/src2/cipher_benchmark/benchmark` (and `beale_3`): `numeric_book_cipher` ranks above
   ALL substitution primaries; P8 `ran` in `battery_coverage`; substitution rows carry the numeric
   counterevidence; runtime < 60 s. Paste both ranked tables. DERIVED targets (FIXED calibration):
   beale_1 → `numeric_book_cipher` 0.85, `nomenclator_codebook` 0.35, every substitution primary
   0.00 (each with `numeric_token_stream` + `numeric_inconsistent_with_substitution`
   counterevidence); beale_3 → `numeric_book_cipher` 0.50 (its `structured_hoax_artifact` fires — a
   genuine Gillogly-style digit/run signature — knocking 0.35 off, consistent with the Part-8
   hoax-ambiguity finding), `nomenclator_codebook` 0.35, `plaintext_or_hoax` 0.15, every
   substitution primary 0.00. Both remain top1 `numeric_book_cipher` above all substitution.
3. `beale_report.py` reproduces ≥ 1 headline statistic per paper: (a) **Gillogly 1980** — B2
   self-check passes (report skipped-index count) AND the B1-under-DoI decode contains an
   alphabetical non-decreasing run ≥ 10 letters at upper-tail `p < 1/1001` (print run text +
   offset); (b) **Wase 2021** — the Benford/epsilon-Benford table B1/B2/B3 (all deviations
   printed) supporting that B1/B3 differ from B2; (c) **Campanelli 2023** — the operationalized
   last-digit / growth / gap divergence calls showing B1/B3 unlike B2 under the standard B2
   method. Qualitative agreement is the bar; note edition/tokenization dependence in the output.
4. borg_0077v fixture `verdict != coherent` via `island_report`; the same text through
   `validate_plaintext_finalist` carries the new `island_report` key with all pre-existing keys
   byte-identical (pin test green). Additionally (finding 3 hard gate): the B2 self-check plaintext
   → `coherent` and a fixed-seed shuffle of a coherent-prose sample → `!= coherent`, both under the
   stricter attested-density + order-significance rule.
5. All four near-misses → `uncertain` with the tabulated trigger + the correct discriminator first
   (fixtures (iii)/(iv) via a covering discriminator, (ii) via the finding-5 fallback naming
   `disc_sub_periodic`); the five confident fixtures → `confident`, top family `strong`,
   right-reason evidence. DERIVED per-fixture scores are tabulated in Part 9 and re-derived by
   `calibrate_inv0_scoring.py`; a change to any weight/threshold must keep all nine rows passing.
6. Determinism: acceptance runs 2–3 repeated in a fresh process produce byte-identical JSON.

## Out of scope

InvestigationState/ledger/plans/inbox/notes (INV-1+); battery-as-episodes and `theorize` (INV-2);
admission policy (INV-3); verify wiring, `lead_verdict`, decoy floor, brittleness probe, lineup
(INV-4/5); corpus key-text search of any kind; `CandidatePacket` schema changes;
`BenchmarkLoader` changes; any v2 behavior change beyond the named additive surfaces.

## Revision notes (dispositions for review findings 1–16 and SECOND-review findings 1–8)

**DERIVED score table** — computed by `scripts/research/calibrate_inv0_scoring.py` FIXED mode
against the real `cipher_id.py` + real Beale streams (NOT on paper; the two prior paper passes were
numerically wrong — that is why review-2 mandated executable recalibration). `score(top1)` /
`score(top2)` (0–1 evidence weight, clamped):

| fixture | top1 | top2 | margin | verdict | conf |
|---|---|---|---|---|---|
| A mono | mono 1.00 | transposition 0.27 | 0.73 | confident | strong |
| B periodic | polyalphabetic 0.80 | homophonic 0.32 | 0.48 | confident | strong |
| C homophonic | homophonic 0.90 | polyalphabetic 0.34 | 0.56 | confident | strong |
| D transposition | transposition 1.00 | mono 0.20 | 0.80 | confident | strong |
| E numeric book | numeric_book 0.85 | nomenclator 0.35 | 0.50 | confident | strong |
| (i) short mono | mono 0.79 | homophonic 0.10 | 0.69 | uncertain (a) | — |
| (ii) light homophonic | homophonic 0.44 | polyalphabetic 0.34 | 0.10 | uncertain (b) | — |
| (iii) transp no-render | mono 0.46 | transposition 0.37 | 0.09 | uncertain (b)+(d) | — |
| (iv) uniform random | plaintext_or_hoax 0.36 | numeric_book 0.29 | 0.07 | uncertain (b) | — |

Beale (acceptance 2, DERIVED): beale_1 numeric_book 0.85 vs all-substitution 0.00 (nomenclator
0.35); beale_3 numeric_book 0.50 vs all-substitution 0.00 (nomenclator 0.35, hoax 0.15). The
`ORIGINAL`-catalog mode of the same script reproduces every second-review failure verbatim
(fixture A ranks poly 0.54; B homophonic 0.62 uncertain; E and beale_1 homophonic 0.85 == numeric
0.85 tie; ii/iii/iv all mis-verdicted).

**Second-review (calibration) dispositions:**
- **R2-1 numeric ties/beats** — FIXED by executable recalibration: numeric-class gate suppresses
  the substitution frequency/shape atoms, `numeric_inconsistent_with_substitution` (0.50) + the
  widened `numeric_token_stream` weakens-list drive every substitution primary to 0.00 on beale;
  the book cipher is early-biased so `numeric_book_cipher` reaches `strong`.
- **R2-2 shuffled reads coherent** — FIXED: attested-junction floor (≥2) + word-bigram
  order-significance (shuffle-null p ≤ 0.05); shuffled Monte-Carlo drops 240/240 → ≈3–5 % (α floor);
  B2 self-check stays `coherent` under the stricter rule.
- **R2-3 shape thresholds** — FIXED: switched from `_normalized_entropy` (overlapping tails) to
  per-token χ² flatness with derived thresholds 0.28 / 0.10; fixture A `strong` and (iii) top-2 now
  hold.
- **R2-4 confusable_with** — FIXED: enumerated symmetric sets tied to discriminators; B/C stay
  confident because poly does not list homophonic.
- **R2-5 (iii) trigger / rule (c)** — FIXED: reworded to (b)+(d), rule (c) noted near-dead;
  `recommended_next` fallback defined and exercised by (ii).
- **R2-6 baselined emission** — FIXED: emit-only-when-significant via shuffle/parametric nulls;
  removes the spurious 0.18 poly boost.
- **R2-7 subtype scoring** — FIXED: `book_*_signature` atoms; E's `numeric_word_position` scores
  0.40 under subtypes.
- **R2-8 definitional gaps** — FIXED: ε_benford threshold 0.12; digit-preference = LAST-digit χ²;
  book_keylength bound `max ≤ 100000`; p-value parenthesized `(#{≥obs}+1)/(n+1)`; encode_tokens
  nonnegative note. **One derived deviation flagged:** review-2's fixture (ii) expected top-2
  "homophonic, mono" is empirically unreachable — a realistic light homophonic depresses IC (fires
  `depressed_ic`), which crushes mono and lifts poly, so the honest near-miss is
  **homophonic-vs-polyalphabetic** (forcing mono would need homophones on rare letters, an
  unrealistic cipher). The Part-9 row and the fallback discriminator were updated accordingly rather
  than overfitting the construction.

- **1 (P0) family scoring** — FIXED: explicit weight formula + atom catalog + clamp + `(−score,
  id)` tie-break + `unknown`-key drop; transposition no longer relies on the capped 0.40 suspicion
  (its score comes from `letters_unsubstituted`). Numeric-vs-substitution calibration completed
  under R2-1 (see above).
- **2 (P0) discriminator state** — FIXED (not deferred): `DISCRIMINATOR_REGISTRY` enumerated;
  "run" = a panel-`ok` predicate over `depends_on_panels`; the all-pending trigger requires a
  non-empty discriminator set (kills the vacuous-`all()` bug).
- **3 (P0) island span** — FIXED: order-sensitive `adjacent_ok`; `coherent` additionally requires
  a word-bigram resource. The v1 either-word function rescue was found to regress (shuffled reads
  coherent) and was replaced under R2-2 (attested-junction floor + order-significance); shuffled/`la`
  now cannot read `coherent` and B2 still can.
- **4 (P0) panel signature** — FIXED: `alphabet_class` + `letter_rendering` threaded through
  `diagnose`/`run_battery`; P5/P9 gated on them.
- **5 (P0) fixture consistency** — FIXED: fixture table parameterized and disjoint (D vs (iii)
  distinguished by rendering present/withheld); each winner DERIVED by the calibration script (see
  the derived score table above), not on paper.
- **6 (P1) shuffle stats** — FIXED: `tail`+`p_value`, reliability keyed to the declared tail,
  `n_shuffles ≥ 1000` for ≥ 99.9 claims, 8-byte-BE `encode_tokens` for seed/cache/`view_hash`,
  modulo labeled parametric.
- **7 (P1) flat ranking** — FIXED: `role`-based primaries/subtypes/modifiers; margin over
  primaries only.
- **8 (P1) hoax split** — FIXED: `independent_random_like` vs `structured_hoax_artifact`, k-of-n
  rules, monotone runs supporting-only for the structured flag.
- **9 (P1) DoI resource** — FIXED: solution-labeled + guarded + test-pinned; cited-quirk-bounded
  iteration; edition/normalization/sha256; skip-with-position out-of-range rule.
- **10 (P2) max_period** — FIXED: added to `diagnose`/`run_battery`, forwarded to fingerprint +
  P3/P4.
- **11 (P2) CandidatePacket** — RESOLVED by choosing the validation-nested contract; the D9
  divergence and null-mask gap are a recorded deviation to flag to the design owner (no top-level
  field in INV-0).
- **12 (P2) duplicate segmentation** — FIXED: `island_report(segmented=...)`; Part-7 passes the
  existing segmentation.
- **13 (P2) __init__ ownership** — FIXED: conventions bar edits to
  `src/investigation/__init__.py`.
- **14 (P2) reader field mapping** — FIXED: `related_records[].record_id`; hold-for-review keyed
  off `rights_class`.
- **15 (P3) P9 numeric gate** — FIXED: P9 `not_computable` for non-letter classes.
- **16 (P3) Campanelli defs** — FIXED: concrete growth/gap/divergence operationalizations in
  Part 8.

No finding was dropped; all 16 first-review findings (15 fixed, #11 resolved-with-deviation) and
all 8 second-review (calibration) findings are addressed above and reproduced by the calibration
script.

## Deliverables

Files changed/added; suite counts (recorded baseline vs final); the `diagnose` ranked tables for
beale_1/beale_3; the full `beale_report.py` output (or `--json` path); the fixture-verdict table;
deviations report (epsilon-Benford, word-bigram-in-verdict, DoI-numbering, CandidatePacket/D9, and
the derived fixture-(ii) homophonic-vs-poly retargeting). **`scripts/research/calibrate_inv0_scoring.py`
is a required deliverable — kept in the repo as the reproducible source of truth for every derived
number; the implementer must keep its FIXED-mode checks green and its numbers in sync with the
shipped catalog.** No commits.

## Post-third-review amendments (BINDING — wording/comment only, no recalibration)

Third review verdict: READY WITH AMENDMENTS (the empirical calibration
holds under independent re-execution + new-seed sweep). None change a
weight, threshold, or fixture number. The calibration script
`scripts/research/calibrate_inv0_scoring.py` is the reproducible source of
truth and MUST land with this spec as its regression test.

1. **Part-9 preamble self-contradiction.** The claim that fixture D and
   near-miss (iii) "differ only by `letter_rendering`" is false — they
   also differ in length/seed (D ~400 letters seed 5; (iii) ~120 seed 6).
   Build them per the FIXTURE TABLE, not the preamble. Reword to: "same
   construction family (columnar, page 84); (iii) is additionally shorter
   and has rendering withheld." A coder must build (iii) at ~120/seed-6 or
   the derived 0.46/0.37 scores won't reproduce.
2. **`recommended_next` fallback priority** — pin the EXACT order the
   script uses (a non-runnable top-2 discriminator beats a runnable top-1
   one): top2-runnable → top2-any → top1-runnable → top1-any.
3. **Ordered-prose→coherent test sensitivity.** The "coherent English
   prose → coherent" fixture must use a sample that clears dict_rate ≥
   0.75 (pages 84/11/98/120 at the calibration offsets work; some prose
   samples read `word_islands` under the 5000-word proxy dictionary —
   false-negative direction only, the anti-shuffle direction is safe).
   Assert the shuffled fixture's `order_p > 0.05` (measures 0.365), not
   "≈ 0.5".
4. **Stale comment in the calibration script** (~lines 63–65): the
   "homophonic flatness ~0.03–0.08" comment contradicts the spec (0.13–
   0.21) and the script's own behavior (fixture C correctly does not fire
   `flat_high_entropy`). Fix the comment since the script is the declared
   source of truth a coder reads.
5. **Recorded limitation for INV-1+** (not INV-0-blocking): at small n
   (~150 tok) ~5–10% of light-homophonic streams draw a spurious
   `periodic_ic_recovery` and classify confident polyalphabetic — the
   documented cost of the "margin, not forced-uncertain" decision for the
   poly↔homo pair. Acceptable for a diagnosis layer; note it.
