# Spec: Polygraphic and Fractionation Solver Program

Status: proposed future program. This document is intentionally standalone so
it can be reviewed while MCP and agent-host work proceeds elsewhere.

Date: 2026-07-19

Primary roadmap references:

- `docs/inv_family_roadmap.md`
- `docs/solver_coverage_matrix.md`
- `docs/automated_tool_comparison.md`
- `docs/polyalphabetic_capability_plan.md`

## 1. Executive Summary

Decipher has strong character-substitution, homophonic, transposition, and
periodic-polyalphabetic capabilities. Its next major solver program should
cover ciphers whose natural units are digraphs, matrices, Polybius
coordinates, or composed coordinate/transposition pipelines.

The program lands eight generated families in this order:

1. Hill 2x2
2. Playfair
3. Two-Square
4. Four-Square
5. Bifid
6. Trifid
7. ADFGX
8. ADFGVX

Hill 2x2 is a bounded engineering probe. Playfair is the first flagship
search solver. The remaining families reuse square/cube key models,
coordinate representations, search machinery, scoring, and the existing
transposition stack.

This is a no-LLM solver program first. Investigator and MCP exposure follows
only after local solvers and solver-backed discriminators have measured
behavior on fresh generated cases.

<!-- FABLE REVIEW [MAJOR] Cross-program coordination — RESOLVED in §1a below
(land order, solver_status decision, id namespaces, pure_transposition reuse,
experiment-type convention, and the operation-manifest consequence from the
investigation-CLI program). -->

## 1a. Cross-program coordination — SEE §1.1 (canonical)

The canonical binding coordination block is **§1.1** below. An earlier hand-
edit added a duplicate here; it is merged into §1.1 (which resolves the
review #2 N2 divergence). §1.1's decisions govern; two corrections from
review #2 are folded there: (N1) experiment types reach both surfaces via the
shared SERVICE-LAYER dispatch over `EXPERIMENT_TYPES`, NOT via the operation
manifest (`experiment_submit` is one operation with a string `type` arg;
`test_interface_parity` covers that operation, not each type); (N2) the
keyword-columnar search is ONE shared module `analysis/columnar_search.py`
with pluggable scoring — built by the composite program (Slice C.1, closing
matrix finding F2) and EXTENDED by this program's PF-6 with a
fractionation-stream scoring plugin. Neither program duplicates the search.

<!-- FABLE R2: OK on (a),(b),(c). Verified against code:
  (a) FamilySpec.solver_status enum is exactly {solves_automated, agent_assists,
      diagnoses_only, planned, unsupported} (families.py:59); composite uses
      agent_assists (composite spec §2.2). Match confirmed.
  (b) fractionation_transposition is an existing primary (families.py:247);
      disc_transp_fractionation is the existing `planned` discriminator
      (families.py:152-159); substitution_transposition/disc_sub_transp_composite
      are composite-owned (composite §2.2). No id collision.
  (c) pure_transposition.screen_pure_transposition IS geometric + language-scored
      with no keyword-columnar enumerator (verified: only matrix_rotate/transmatrix/
      rail/route/mask/block candidates); ADFGX/ADFGVX peel = keyed columnar over a
      2-symbol coord stream (fractionation.py: columnar_encrypt(frac, trans)).
      Distinct attacks confirmed. BUT see the §1.1(3) reconciliation finding —
      "writes its own columnar search" here vs "built ONCE as a shared module
      analysis/columnar_search.py closing F2" in §1.1 is an unresolved divergence. -->
<!-- FABLE R2: [MINOR] (d) mechanism mis-stated. An experiment TYPE is NOT an
operation-manifest entry. Verified: MCP_TOOL_DEFINITIONS (mcp_server/tools.py:75)
is the flat tool list; `experiment_submit` is ONE tool (tools.py:259) whose `type`
arg is a bare string, not even an enum. A new EXPERIMENT_TYPES value is dispatched
by the SHARED service layer that reads EXPERIMENT_TYPES — it reaches both surfaces
post-I-0 automatically, NOT by "registering in the manifest," and it is NOT gated
by test_interface_parity.py (which checks operation↔verb parity + top-level schema
props; the CLI passes `config` as one `--config JSON` blob, so per-type config
fields are not individual argparse args either). FIX: reword (d) and PF-7 to:
"reaches both surfaces because both skins dispatch through the shared service layer
over EXPERIMENT_TYPES (post-I-0); no manifest entry, and parity coverage is of the
`experiment_submit` operation, not of the type." Same wording error is echoed in
composite §7a and CLI §10 — consistent across the three, but consistently loose. -->
<!-- FABLE R2: [MODERATE] §1a and §1.1 are TWO near-identically-titled binding
coordination blocks ("RESOLVED 2026-07-19" vs "BINDING decisions — Fable
2026-07-19"). The R2 §1a rewrite did not state its relationship to §1.1, and the
two DIVERGE on columnar-search ownership: §1a(c) + PF-6 say "this program writes
its own columnar search"; §1.1(3) says it is "built ONCE as a shared module
(analysis/columnar_search.py) ... also closes matrix finding F2 ... Neither program
duplicates the other's search." FIX: make §1a explicitly supersede/merge §1.1, and
pick one columnar-search story — recommended: keep §1.1's shared pluggable-scoring
module (PF supplies the fractionation-stream scorer, transposition side supplies
the language scorer), and change PF-6's "writes its own" to "adds the
fractionation-stream scoring plugin to the shared analysis/columnar_search.py." As
written, a PF-6 coder builds a PF-private search and never learns to close F2,
which §1.1 forbids. -->


### 1.1 Cross-program coordination (BINDING decisions — Fable, 2026-07-19)

Resolves the [MAJOR] coordination finding above. Three concurrent programs
touch the same surfaces: THIS program, the composite
(`composite_substitution_transposition_spec.md`), and the investigation CLI
(`investigation_cli_spec.md`). The following decisions bind all three; each
spec's implementers treat them as constraints, and deviations go back to the
spec authors, not into code.

1. **Land order.** Composite Slice A (families/panels registry churn) lands
   first (already in implementation). This program's registry changes (finer
   family ids, detectors — see PF-7 finding) rebase on it. CLI milestone I-0
   (transport-neutral SERVICE LAYER) lands before this program's PF-7 surface
   work. **Mechanism (review #2 N1):** a new experiment type is an
   `EXPERIMENT_TYPES` value, NOT an operation-manifest entry —
   `experiment_submit` is a single operation whose `type` is a string arg.
   Both skins (MCP server, `decipher investigation`) dispatch it through the
   shared service layer that reads `EXPERIMENT_TYPES`, so a new type reaches
   both surfaces automatically post-I-0. `test_interface_parity` covers the
   `experiment_submit` OPERATION, not each type. PF-7 adds `EXPERIMENT_TYPES`
   entries (not manifest entries) and does not hand-wire the MCP tool list.
2. **Experiment-type convention.** All new types follow the
   `quagmire3_shotgun` pattern (`9f4ed28`): bounded budget knobs, host-derived
   `language` (GT-3), unknown-key rejection with a family-consistent
   `corrected_example`, results installable via `experiment_collect` so the
   verify→declare gate is reachable. **Misroute-guard convention**: an
   `automated_solver` submit whose `cipher_system` names an
   accepted-but-unsolvable family gets a structured validation error
   redirecting to the dedicated type (the `quag` precedent). PF families
   adopt this as each dedicated type lands (e.g. `playfair`, `bifid`,
   `adfgvx` hints redirect).
3. **Transposition search ownership.** `analysis/pure_transposition` stays
   GEOMETRIC-only (matrix-rotate/route/rail/mask/TransMatrix). Keyword-columnar
   search is a SEPARATE shared module `analysis/columnar_search.py` — over
   arbitrary token streams with PLUGGABLE scoring. Built ONCE by the composite
   program (its Slice C.1 needs it: the round-4 acceptance cipher is keyword
   columnar, which the geometric screen cannot do — this is matrix finding F2,
   and building the module closes it). The composite supplies the LANGUAGE
   scorer plugin (its peeled stream is A-Z letters); PF-6 (ADFGX/ADFGVX)
   EXTENDS the same module with a FRACTIONATION-STREAM scorer plugin (its
   peeled stream is 2-symbol coordinate data with no language structure).
   Neither program duplicates the other's search; the composite's peel does
   NOT "reuse pure_transposition as-is" (that was the pre-review error).
4. **`solver_status` vocabulary.** Existing enum ONLY
   (`solves_automated | agent_assists | diagnoses_only | planned |
   unsupported`) — option (b) of the §12 finding; the three research statuses
   become a separate `rollout_status` report field (§12, as revised). The
   composite family uses `agent_assists`; PF families map: probe →
   `diagnoses_only` (until measured recovery) or `agent_assists`,
   experimental solver → `agent_assists`, strong-gate pass →
   `solves_automated`.
5. **Namespaces.** Composite claims family id `substitution_transposition` +
   `disc_sub_transp_composite`. This program claims: the existing `playfair`
   id (status flips only), subtypes under `polygraphic_substitution`
   (`hill_2x2`, `two_square`, `four_square`), subtypes under
   `fractionation_transposition` (`bifid`, `trifid`, `adfgx`, `adfgvx`),
   discriminator prefix `disc_polygraphic_*` / `disc_fractionation_*`.

<!-- FABLE REVIEW [MINOR] detector atoms — RESOLVED as explicit scope:
verified against cipher_id.py the ONLY polygraphic fingerprint suspicion today
is `playfair`; `polygraphic_substitution` and `fractionation_transposition`
FamilySpecs have EMPTY detectors=(), so they rank with no fingerprint-prior
atoms. §3.2's "distinguish broad polygraphic vs fractionation evidence" needs
new DETECTOR ATOMS wired into panels/cipher_id, not just discriminators. Scoped
as the PF-7 registry-granularity sub-slice (subtype ids + their detector atoms
land together). -->
<!-- FABLE R2: OK — verified cipher_id.py emits NO polygraphic/fractionation
suspicion (only the `playfair` block, cipher_id.py:747-768) and both primaries
carry detectors=() (families.py:249,267). Note for the coder: _validate_registry
does NOT validate detector strings at all, so a detector atom name is inert until
panels.py/cipher_id.py actually EMIT it — the sub-slice must wire the emitter, not
just name the string in the FamilySpec. -->


## 2. Current Baseline

<!-- FABLE: verified every baseline claim below against the repo — all accurate.
playfair.py has PlayfairCipher/TwoSquareCipher/FourSquareCipher; numeric.py has
Hill2x2Cipher (2x2 only); fractionation.py has Bifid/Trifid/ADFGX/ADFGVX modeled
as fractionation-then-columnar; family_registry.py has generators for all eight;
cipher_id.py emits a `playfair` suspicion (Playfair-only — see §3.2 note);
families.py has the coarse diagnosis-only entries. No path/name corrections
needed in this section. -->
Already present and treated as the rule-level source of truth:

- `src/ciphers/playfair.py`: Playfair, Two-Square, Four-Square primitives.
- `src/ciphers/numeric.py`: Hill 2x2 primitive.
- `src/ciphers/fractionation.py`: Bifid, Trifid, ADFGX, ADFGVX primitives.
- `src/testgen/family_registry.py`: generators for all eight families.
- `tests/test_cipher_families.py`: known-answer and round-trip tests.
- `src/investigation/families.py`: coarse Playfair, polygraphic, and
  fractionation registry entries, currently diagnosis-only.
- `src/analysis/cipher_id.py`: preliminary Playfair suspicion signal.
- Existing n-gram/language scoring and transposition search infrastructure.

The no-LLM coverage sweep currently applies the wrong monoalphabetic attack to
most of these families and lands near the approximately 0.40 chance-overlap
floor. That is a failure, not partial support.

## 3. Goals

### 3.1 Solver goals

- Recover readable plaintext and an explicit family-native key model without
  benchmark plaintext entering search or selection.
- Search arbitrary square/cube arrangements, not only dictionary keywords.
- Enumerate or infer family parameters such as Bifid/Trifid period and Hill
  matrix.
- Preserve multiple finalists and family hypotheses when evidence is
  ambiguous.
- Expose bounded `screen`, `full`, and `overnight` profiles with recorded
  seeds, proposal counts, threads, and elapsed time.
- Keep Python responsible for orchestration, candidate records, artifacts,
  routing, and reporting. Move measured hot loops to Rust only after a Python
  reference search establishes semantics.

### 3.2 Investigator goals

- Distinguish broad `polygraphic_substitution` and
  `fractionation_transposition` evidence from exact-family claims.
- Use bounded inversion as a probe when static statistics cannot distinguish
  Playfair, Two-Square, Four-Square, Hill, Bifid, or Trifid reliably.
- Represent ADFGX/ADFGVX as compositions rather than flattening them into an
  unrelated family name.
- Report abstention and external referral when length, language, or budget is
  outside measured support.

### 3.3 Scientific goals

- Measure family confusion, not just within-family recovery.
- Quantify candidate recall separately from finalist ranking.
- Establish how performance changes with length, language, key shape, period,
  noise, and missing boundaries.
- Prevent exact-family diagnosis from being inferred solely from a registry
  label or suggestive low-order statistic.

## 4. Non-Goals

- No changes to agent-v3, MCP host extraction, or client prompts in the early
  solver milestones.
- No general bigram-substitution solver beyond the named square/matrix
  families.
- No Nihilist substitution, straddling checkerboard, VIC, Fractionated Morse,
  Morbit, or Pollux in this program. The architecture should permit them later.
- No OCR or manuscript-layout work.
- No claim that exact key recovery is always identifiable. Plaintext recovery
  is primary; equivalent keys must be reported honestly.
- No use of benchmark plaintext, generator keys, test identifiers, or hidden
  family metadata in blind search and candidate selection.

## 5. Binding Invariants

### 5.1 Ground-truth firewall

Generator family, key, prepared plaintext, and solution alignment are visible
only to post-hoc tests and reports. Runtime solvers may consume ciphertext,
explicitly permitted language/context, public solver configuration, and their
own candidate scores.

An explicit-family benchmark route may name the family, as in ordinary
context-aware parity. A blind route must infer a broad family or launch a
bounded family probe without reading the generated label. Reports must say
which mode was used.

### 5.2 Family-native key semantics

Do not force these keys into `dict[cipher_token, plaintext_token]`:

- Hill uses a modular matrix.
- Playfair/Bifid use a 25-cell square.
- Two-Square/Four-Square use two independent squares.
- Trifid uses a 27-cell cube.
- ADFGX/ADFGVX use a square plus a transposition key/order.

Each result must serialize a typed key and enough convention metadata to
replay decryption exactly.

### 5.3 Honest preparation and scoring

Playfair inserts fillers and splits repeated letters; 5x5 families merge I/J;
Trifid has a 27th symbol; ADFGVX includes digits. Runtime language scoring and
post-hoc accuracy must distinguish:

- source plaintext;
- family-prepared plaintext;
- raw decrypted output;
- optionally normalized human reading.

Solver selection never uses post-hoc character accuracy. Acceptance reports
must state which normalized target is being measured.

### 5.4 No false support from fallback

If a dedicated solver does not run or returns no accepted candidate, the
artifact must say unsupported/unsolved. A monoalphabetic fallback at roughly
40% overlap cannot be labeled a polygraphic or fractionation partial solve.

## 6. Shared Architecture

Names below are recommendations, not mandatory filenames. Keep modules
additive and outside agent/MCP code until the integration milestone.

### 6.1 Typed models

Provide serializable immutable models resembling:

```python
SquareKey25(cells: str, ij_merged: bool = True)
SquareKey36(cells: str, symbols: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
SquarePairKey25(first: SquareKey25, second: SquareKey25)
CubeKey27(cells: str, filler_symbol: str)
MatrixKey2x2(a: int, b: int, c: int, d: int, modulus: int = 26)
FractionationKey(grid: SquareKey25 | CubeKey27, period: int)
ADFGXKey(grid: SquareKey25 | SquareKey36, column_order: tuple[int, ...],
         labels: str, incomplete_mode: str)
```

Validation must reject duplicate/missing cells, non-invertible Hill matrices,
invalid periods, and malformed column orders before decryption.

<!-- FABLE REVIEW: [MINOR] Grounding vs the existing primitives (verified in
src/ciphers/{playfair,numeric,fractionation}.py). The cipher layer has NO key
dataclasses today — keys are bare tuples/strings on FamilyCipher subclasses:
Playfair = a single 25-char keyed-square string (I/J merge BAKED IN via a J-less
alphabet + clean_ij; there is no `ij_merged` flag, and ADFGVX/Trifid do NOT
merge); Two-Square/Four-Square = a `(k1, k2)` keyword tuple; Hill2x2 = an
`(a,b,c,d)` int tuple; Bifid = 25-char string with `period` a CONSTRUCTOR arg
(not part of the key); Trifid = 27-symbol cube string, period also a ctor arg;
ADFGX/ADFGVX = a `(square_keyword, transposition_keyword)` tuple. Introducing
these typed solver-layer models is fine, but the spec must state that each model
serializes to/from the primitive's existing tuple/string convention so
`describe_key`/replay stays bit-exact against the round-trip tests in
tests/test_cipher_families.py. In particular `FractionationKey(period=...)` here
moves period INTO the key, diverging from the primitive where period is a ctor
arg — reconcile so a serialized key replays without a side-channel period. -->


### 6.2 Candidate/result contract

Every solver returns the same outer record:

```python
FamilyCandidate(
    family: str,
    plaintext: str,
    key: typed_key,
    score: float,
    score_components: dict[str, float],
    parameters: dict,
    seed: int,
    source_stage: str,
)

FamilySearchResult(
    family: str,
    candidates: list[FamilyCandidate],
    profile: str,
    budget: dict,
    diagnostics: dict,
    elapsed_seconds: float,
    engine: str,
)
```

Candidates are ordered by ground-truth-free score. Keep at least a configurable
top K plus structurally diverse finalists; do not retain only global best.

**Contract boundary (resolved).** `FamilySearchResult`/`FamilyCandidate` is the
SOLVER-INTERNAL shape only. It is NOT a new persistence/install format. When
surfaced through the host (PF-7), the runner maps it onto the existing
EXPERIMENT_TYPES flat result dict — `(cipher, branch_snapshot, config) → {status,
solver, error_message, elapsed_seconds, key, final_decryption, steps,
optionally top_candidates}` — and install happens via `experiment_collect` onto
a branch whose metadata carries `decoded_text` for the grader, with the richer
typed key stored in the branch `metadata` slim-record exactly like the quagmire
install. No new install path.

<!-- FABLE REVIEW [MAJOR] parallel result/install contract — RESOLVED: internal
shape maps onto the flat EXPERIMENT_TYPES result + experiment_collect install;
typed key in branch metadata, not a new persistence format. -->
<!-- FABLE R2: OK — verified the runner contract is the flat dict
`{status, solver, error_message, elapsed_seconds, key, final_decryption, steps,
[top_candidates]}` (experiments.py:271-281, 389-399) and install writes
`decoded_text`/`decoded_text_source` onto branch metadata via the installer
(experiments.py:1361,1492-1493), with the slim typed record in metadata exactly
like _quagmire3_candidate_record. Mapping is faithful; no new install path. One
concrete note: the flat result `key` field is coerced to `{str: int}`
(experiments.py:278) — a cipher-token→plaintext-token dict. The family-native
typed keys here (SquareKey25, MatrixKey2x2, ADFGXKey) do NOT fit that shape, so
they MUST ride in `top_candidates[*].metadata` / branch metadata (as the spec
says), and `key` stays `{}` for these families (mirror quagmire3, which returns
`key: {}` at experiments.py:394). Make that explicit so a coder doesn't try to
serialize a matrix into the int-map `key`. -->



### 6.3 Scoring

The initial scorer should reuse Decipher's strongest applicable continuous
language score. It must:

- score no-boundary letter streams;
- support prepared I/J-merged text;
- avoid dictionary segmentation as the sole objective;
- return stable component diagnostics;
- support incremental or windowed updates in hot search loops;
- have an explicit language/profile identifier in every artifact.

Before tuning search, run an oracle-ranking test: score true-key plaintext,
near-key mutations, random-key candidates, and wrong-family candidates. If the
true basin is not reliably preferred when present, fix scoring before adding
more search breadth.

<!-- FABLE REVIEW: [MINOR] The "strongest applicable continuous language score"
in this repo is the zenith_native 5-gram English binary model (models/ngram5_*.
bin via analysis.model_registry). Two concrete risks the oracle test MUST cover,
or it will pass on the wrong target: (1) The scorer is trained on NATURAL text.
The candidates this program optimizes are FAMILY-PREPARED text carrying X-fillers
(Playfair doubled-letter splits + odd-length padding) and I/J merges. A 5-gram
model penalizes those artifacts, so score the *prepared* target the solver
actually produces (per §5.3), not clean source — otherwise the true key can rank
BELOW a filler-free near-miss. (2) Trifid/ADFGVX inner streams (pre-square)
are coordinate/label symbols with no language structure; the language score is
meaningless there (see PF-6). Review Question 1 already names this; make the
prepared-text + non-language-stream cases MANDATORY rows in the PF-0 oracle
packet, not optional. -->

<!-- FABLE REVIEW: [MINOR] Firewall on the oracle-ranking harness: it consumes
the TRUE key/plaintext, so state plainly (as the composite spec does for its
sealed answers) that this is a TEST/CALIBRATION path only, never reachable from a
runtime solve or selection. Keep it in tests/ or a scripts/ calibration tool with
`ground_truth` confined to the assertion, mirroring AGENTS.md GT-1 and
tests/test_ground_truth_firewall.py. -->


### 6.4 Search mutations

Shared square/cube mutations should include:

- swap two cells;
- swap rows/columns or cube planes;
- rotate/reverse a row or column;
- move one cell or a contiguous run;
- keyword-shaped seed squares plus arbitrary random permutations;
- family-specific composite mutations over two squares;
- period changes for Bifid/Trifid;
- column-order changes for ADFGX/ADFGVX.

Record accepted/proposed mutation counts by type so dead proposal families are
visible.

### 6.5 Reproducibility and parallelism

- A seed plus profile must reproduce a candidate population modulo documented
  thread scheduling differences.
- Parallelize independent restarts through the global worker configuration.
- Record effective threads, restarts, iterations, candidate count, and model.
- Never create nested process pools from an already parallel outer suite.

<!-- FABLE REVIEW: [NIT] This exact hazard is already solved for the experiment
queue by the W/S/I arbiter (`compute_arbiter` / `ExperimentQueue` in
src/investigation/experiments.py), which caps inner parallelism (I) so
S x I <= W across background solver slots. When these solvers run under
experiment_submit (PF-7), inherit that arbiter rather than adding independent
worker-governance; reference DECIPHER_PARALLEL_WORKERS / DECIPHER_EXPERIMENT_SLOTS
so the two mechanisms don't fight. -->


## 7. Milestones

### PF-0: Contracts, calibration packet, and score audit

Deliverables:

- Typed key/result contracts.
- Common normalization and prepared-plaintext scoring helpers.
- A generated packet with fresh seeds for all eight families plus confusable
  monoalphabetic, homophonic, transposition, and periodic-poly controls.
- Oracle-ranking reports for each family.
- `screen`, `full`, and `overnight` budget schema, even if only `screen` is
  implemented initially.

Gate:

- Existing cipher primitive tests remain unchanged and pass.
- No generated solution data is present in runtime solver payloads.
- The true-key candidate ranks above random candidates in at least 95% of
  long-form English calibration cases. Failures are analyzed before PF-1.

### PF-1: Hill 2x2 exhaustive solver

Algorithm:

<!-- FABLE: verified 157,248 = |GL(2,Z_26)| = |GL(2,Z_2)|*|GL(2,Z_13)| = 6*26208.
numeric.py::Hill2x2Cipher.key_space_size returns exactly this. Count is correct. -->
- Enumerate all 157,248 invertible matrices in `GL(2, Z_26)`.
- Decrypt and score each matrix.
- Retain top K candidates and equivalent/near-equivalent keys.
<!-- FABLE REVIEW: [QUESTION] "equivalent/near-equivalent keys" — unlike Playfair
(which has a large symmetry class per solution), a Hill 2x2 decryption matrix is
essentially unique; there is no big equivalence class to preserve. Clarify what
this means here (adjacent-basin near-misses for the diversity guarantee? true
matrix symmetries, of which there are essentially none mod 26?). If it's just
"keep structurally diverse runners-up", say that; if you believe there are real
equivalent Hill keys, name the symmetry. This wording may confuse the coder. -->

- Start in Python if fast enough; otherwise use a small Rust exhaustive kernel
  with a Python reference on reduced key sets.

Gate:

- At least 20 fresh cases across multiple lengths and seeds.
- Exact prepared plaintext on at least 95% of cases of 80 or more letters.
- Deterministic top result for fixed input/profile.
- Negative controls do not trigger a Hill solved verdict merely because one of
  157,248 candidates has modest language score.

PF-1 validates the contracts; it does not by itself establish that square-key
annealing works.

### PF-2: Playfair flagship solver

Algorithm:

- Simulated annealing/hill climbing over arbitrary 25-cell squares.
- Many independent restarts with temperature/reheat schedules.
- Incremental rescoring for mutations where practical.
- Dictionary-keyword squares may seed the population but cannot define the
  entire search space.
- Preserve candidates from distinct square basins.

Diagnostics:

- Prepared-text handling and likely filler locations.
- Key-square replay.
- Score trajectory and restart diversity.
- Near-key mutation capture report.

Gate:

- Fresh generated English cases at short, medium, and long lengths.
- `full` profile reaches median prepared-text character accuracy >= 0.90 on at
  least 20 medium/long cases, with at least 80% of cases >= 0.80.
- Short cases are reported separately and may remain frontier cases.
- Playfair probes beat wrong-family probes on a held-out confusion packet
  without forcing confident exact-family labels on ambiguous short texts.

Do not move the hot loop to Rust until the Python/reference candidate behavior
and scoring gate are established.

### PF-3: Two-Square and Four-Square

Algorithm:

- Reuse square mutations and scoring.
- Search two squares with alternating coordinate descent plus occasional joint
  mutations/restarts.
- Preserve square-swap and other equivalent-key symmetries in diagnostics.
<!-- FABLE REVIEW: [MINOR] Keyspace clarification, verified against
playfair.py::FourSquareCipher: Four-Square's two PLAINTEXT squares are the fixed
standard square (PLAIN_SQUARE), NOT searched — only the two mixed CIPHER squares
are the key `(k1, k2)`. So `SquarePairKey25` is the right model for BOTH families,
but state that for Four-Square the search space is the two mixed squares only.
Also: Two-Square is self-reciprocal (decrypt == encrypt in the primitive), Four-
Square is not; the square-swap symmetry you want to preserve differs between the
two, so treat them as separate symmetry sets in diagnostics rather than one. -->


Gate:

- At least 20 fresh cases per family at medium/long lengths.
- Each family reaches experimental support: median prepared-text accuracy >=
  0.80 and top-5 capture >= 0.90 under `full`.
- A cross-family probe report measures Playfair/Two-Square/Four-Square
  confusion and abstains where the margin is uncalibrated.
- Promotion to `solves_automated` requires the stronger PF-2 support threshold,
  not merely the experimental threshold.

### PF-4: Bifid

Algorithm:

- Anneal the 25-cell square while enumerating or jointly mutating period.
- Include whole-message mode and bounded periods supported by message length.
- Cache coordinate transforms and update affected scoring windows where
  possible.

Gate:

- At least 20 fresh cases spanning several periods, including whole-message.
- `full` profile reaches median prepared-text accuracy >= 0.90 on medium/long
  cases and identifies a compatible period set.
- Period aliases and non-identifiability are reported rather than hidden.
- Playfair and ordinary transposition controls do not receive false Bifid
  solved verdicts.

### PF-5: Trifid

Algorithm:

- Generalize fractionation search to a 27-cell cube and three coordinate
  streams.
- Enumerate/mutate period and cube arrangement.
- Treat the filler symbol explicitly in keys and rendering.

Gate:

- At least 20 fresh cases across periods and lengths.
- First acceptance may be experimental (median >= 0.75, top-5 capture >= 0.85)
  because the search space is materially harder than Bifid.
- `solves_automated` status requires the common >= 0.90 median support gate.
- Bifid/Trifid confusion and insufficient-length abstention are measured.

### PF-6a: ADFGX (5×5) and PF-6b: ADFGVX (6×6 + digits)

SPLIT into two milestones per the review — 5×5 is materially more tractable
than 6×6-with-digits, and a single gate over both was optimistic.

Architecture (`fractionation → keyword-columnar transposition`), corrected:

- These are NOT solvable by reusing `pure_transposition.screen_pure_transposition`.
  That screen enumerates GEOMETRIC transforms (matrix-rotate/route/rail/mask/
  TransMatrix) and scores by LANGUAGE model on A-Z text. ADFGX/ADFGVX
  transposition is KEYWORD-COLUMNAR (`fractionation.py` → `ciphers.transposition.
  columnar_*`), and the peeled intermediate is a 2-symbol coordinate stream over
  {A,D,F,G,(V,)X} with NO language structure until the square is also inverted.
  Reuse gives the finalist-MENU skeleton and the Rust batch-scorer harness only.
- This program EXTENDS the shared `analysis/columnar_search.py` module (built by
  the composite program's Slice C.1, §1.1 item 3) with a fractionation-stream
  scoring PLUGIN — it does not build a private search. Concretely it adds the
  keyword-columnar (column-permutation) inversion driven by that plugin,
  driven by a FRACTIONATION-STREAM statistic — column-coincidence / digram
  regularity on the coordinate stream — NOT a language score. The two layers are
  coupled: score candidate column orders by fractionation-stream statistics,
  then jointly optimize the square; alternate and retain diverse joint finalists.
  Language scoring re-enters ONLY after both layers are inverted (final A-Z text).

<!-- FABLE REVIEW [MAJOR] PF-6 reuse/scoring/frontier — RESOLVED: split 6a/6b,
named the fractionation-stream statistic as the driver (not language score),
own keyword-columnar search acknowledged as new work, depth mode + honest gates
below. -->
<!-- FABLE R2: OK on the reuse/scoring/frontier CORE — verified fractionation.py
peels a keyed columnar over a 2-symbol coordinate stream (columnar_encrypt/decrypt,
lines 210/216), the screen has no keyword-columnar enumerator, and the depth-mode /
honest-gate split (calibration modes as primary deliverable; ADFGVX single-message
blind → probe maturity) is sound. BUT this PF-6 rewrite says "this program WRITES
a keyword-columnar inversion search" and lists only "finalist-MENU skeleton + Rust
batch-scorer harness" as reuse — it does NOT mention §1.1(3)'s decision that this
search is "built ONCE as a shared module (analysis/columnar_search.py) [that] also
closes matrix finding F2 ... Neither program duplicates the other's search." A
coder implementing PF-6 from this text builds a PF-private search and never closes
F2. FIX (tie to the §1a/§1.1 reconciliation above): state whether PF-6 (a) adds a
fractionation-stream scoring plugin to a shared columnar_search.py that also serves
the F2 transposition side, or (b) is genuinely PF-private and §1.1(3) is withdrawn.
Do not leave both live. -->


- Support staged diagnostic modes for engineering only:
  - known column order, recover square;
  - known square, recover column order;
  - blind alternating/joint recovery.
- The first two modes are labeled calibration and are not blind capability
  claims.

Blind algorithm:

- Generate a portfolio of plausible columnar inversions.
- For each, optimize the square against the resulting coordinate stream.
- Alternate column-order and square refinement while retaining diverse joint
  finalists.
- For ADFGVX, preserve digits and use a compatible alphanumeric language
  profile rather than silently dropping them.

Depth mode (per review): a supported multi-message DEPTH path (same key,
aligned columns — the historical Painvin setting) is a first-class deliverable
and the realistic route to reliable recovery. Single-message ciphertext-only
blind recovery (unknown square AND unknown column order) is genuinely near the
classical frontier and is marked frontier/experimental, NOT a headline gate.

Gate (honest, per family):
- PRIMARY deliverable: the two calibration modes (known column order → recover
  square; known square → recover column order) recover their unknown component
  on fresh cases. This is what clears the milestone.
- DEPTH mode (multi-message, same key) is a supported blind path; gate it on
  recovery given ≥K aligned messages (K a measured knob).
- SINGLE-MESSAGE blind `full` is tested (≥20 fresh cases per family) but its
  bar is `agent_assists`/probe maturity, not `solves_automated`: ADFGX may
  reach experimental recovery; ADFGVX single-message blind may land only at
  probe_available and that is an acceptable, honestly-labeled outcome.
- Artifacts show both layer keys and replay the complete forward/decrypt
  pipeline; single-layer controls are never reported as composed solutions.

### PF-7: Automated routing and investigator/MCP probes

This milestone starts only after the relevant solver acceptance gates pass and
the MCP host surface is stable.

Deliverables:

- Explicit-family automated routes for accepted solvers.
- A bounded broad-family probe that can compare candidate families without
  reading benchmark labels.
- **Registry granularity FIRST (resolved).** Per-family status is impossible
  today: only `playfair` has its own INV id; Hill/Two-Square/Four-Square roll
  into `polygraphic_substitution`, and Bifid/Trifid/ADFGX/ADFGVX into
  `fractionation_transposition`. Before any family-by-family status change, this
  program introduces finer ids as SUBTYPES under those two primaries (mirroring
  the `numeric_*` subtype pattern), each with a detector atom, so a solver can
  flip one family without over-claiming its siblings. All new ids/subtypes must
  satisfy `_validate_registry()` (every primary ≥1 discriminator, symmetric
  confusables, derived-consistent discriminators) or import fails — treat this
  as an early PF-7 sub-slice, not an afterthought.
- **Experiment schemas follow the EXPERIMENT_TYPES contract EXACTLY (resolved,
  provenance invariant EXP-1).** Each new type in
  `src/investigation/experiments.py` = `{config_schema, config_defaults, runner,
  description, installer?}` + per-field docs, with the two-layer unknown-key
  rejection (provider `additionalProperties:false` + the
  `validate_experiment_config` whitelist; `language` host-derived per GT-3), a
  guaranteed-valid `corrected_example` on error, and a MISROUTE GUARD redirect
  (like `quag → quagmire3_shotgun`). Install via `experiment_collect` through
  the verify→declare gate. Per §1a: register in the OPERATION MANIFEST (post
  CLI I-0) so the type reaches both surfaces; coordinate the misroute-guard
  convention with the composite program (same pattern, distinct type names).
  Do not expose raw mutation controls as a giant agent tool menu.

<!-- FABLE REVIEW [MAJOR] PF-7 registry granularity + [MAJOR] experiment-type
pattern — both RESOLVED above (subtype-first registry work; EXPERIMENT_TYPES /
EXP-1 contract named explicitly; manifest + misroute coordination per §1a). -->
<!-- FABLE R2: OK on registry-granularity and the EXPERIMENT_TYPES/EXP-1 contract,
with two precisions verified against code:
  (1) subtype-first is IMPLEMENTABLE and renders for free — diagnosis.py reads
      solver_status from FAMILY_REGISTRY[fam] and folds SUBTYPE_IDS under parent
      generically (diagnosis.py:301,305-315). Precision on "_validate_registry":
      the "every primary ≥1 discriminator" invariant applies to PRIMARIES, not
      subtypes — subtypes MAY have discriminators=() (numeric_skip_nth_word does,
      families.py:343-350). The real constraint on a new subtype is the
      derived-consistency check (families.py:414-420): if the subtype lists a
      discriminator, a DiscriminatorSpec whose splits name that subtype id must
      exist, else import fails. The two parents already satisfy the ≥1 rule.
  (2) experiment-schema shape matches (config_schema/config_defaults/runner/
      description/[installer], additionalProperties:false in the model-facing
      schema + the validate_experiment_config whitelist, host-derived language,
      corrected_example, misroute guard) — all verified in experiments.py.
      HOWEVER the "register in the OPERATION MANIFEST ... so the type reaches
      both surfaces ... must pass tests/test_interface_parity.py" clause is the
      category error flagged at §1a(d): an experiment TYPE is not a manifest
      operation and is not parity-gated. Apply the §1a(d) FIX wording here. -->


- Coverage-debt reporting when evidence supports a family whose solver was not
  run or remains unsupported.
- Comparison records that separate `best_partial` from `accepts_as_solution`.

The model should choose whether a bounded probe is worth running. The compiled
host owns budgets, schemas, provenance, and artifact persistence.

Gate:

- Blind generated confusion suite reports hierarchical top-1/top-k,
  abstention, false-confident rate, probe cost, and family coverage.
- Fake-client MCP tests can request, poll, inspect, compare, and preserve a
  finalist without accessing generator truth.
- No change to existing v3 behavior is required to expose the new host
  capabilities.

### PF-8: Rust kernels and stress suite

Port only profiles that measurement shows are CPU-bound. Likely kernels:

- square/cube decrypt under a mutation;
- incremental n-gram score delta;
- restart-level annealing;
- Hill exhaustive scoring if Python is not already sufficient;
- joint ADFGX/ADFGVX candidate evaluation.

Requirements:

- Python remains orchestration and artifact authority.
- Reduced seeded searches have reference parity tests.
- Runtime never silently falls back to a scale-inferior Python path when the
  selected profile requires Rust.
- Build/doctor output names the missing module and exact repair command.
- Overnight suite runs fresh generated seeds and records hardware/thread data.

## 8. Generated Evaluation Design

### 8.1 Dataset axes

For each family vary:

- length: short, medium, long;
- seed and key shape;
- language, beginning with English and adding languages only when a compatible
  runtime scorer exists;
- preserved versus removed word boundaries in source presentation;
- filler/padding pressure;
- family-specific period or column count;
- optional controlled transcription damage in a later packet.

Final acceptance cases must use a wholly withheld seed set not consulted
during implementation tuning. Keep tuning, regression, and final-acceptance
seed manifests distinct in reports.

<!-- FABLE REVIEW: [MINOR] This design is consistent with the contamination
policy in docs/evidence/v3_vs_mcp_matrix.md ("None (fresh original prose) ->
capability aggregates; High (famous, in training data) -> behavioral probes
ONLY"). Good. Two concretions to add: (1) all eight families already have fresh
generators in src/testgen/family_registry.py — say the withheld acceptance seeds
are drawn from those, so "fresh synthetic" is mechanized, not hand-built. (2)
Follow the established SEALED-ANSWER convention: sealed answers live OUTSIDE the
repo (e.g. ~/.config/decipher/dogfood_answers/), never committed, so the
acceptance harness can't leak into runtime. §11 already says "no famous historical
cipher is sufficient acceptance evidence" — cross-reference that here. -->


### 8.2 Confusion controls

Every packet includes:

- monoalphabetic substitution;
- homophonic substitution;
- periodic polyalphabetic;
- pure transposition;
- random frequency-preserving permutations;
- plaintext/no-encryption controls;
- neighboring polygraphic/fractionation families.

The purpose is to measure whether a successful-looking score is specific to
the proposed mechanism.

### 8.3 Metrics

Report separately:

- prepared-plaintext character accuracy;
- normalized reading accuracy where meaningful;
- top-1/top-3/top-5 candidate capture;
- score rank of the post-hoc best candidate;
- exact or equivalent key recovery;
- broad-family and exact-family probe accuracy;
- abstention and false-confident rates;
- runtime, proposals, threads, memory, and candidate count;
- performance by length and parameter regime.

Do not average an unsupported 40% fallback into successful cases.

## 9. Profiles and Budgets

Every solver exposes the same conceptual profiles:

| Profile | Purpose | Expected behavior |
|---|---|---|
| `screen` | diagnosis/probe | small restart set, strict wall-clock cap, preserve a few finalists |
| `full` | ordinary solve | multi-core restarts and accepted family budget |
| `overnight` | frontier/stress | broad periods/parameters, many seeds, checkpointed candidates |

Exact numeric defaults are set from measured calibration, not guessed in this
document. Artifacts always record effective values, so profile defaults can
evolve without making runs incomparable.

## 10. Artifacts and Human Reports

Each run should make the following obvious:

- which family was assumed or inferred;
- whether the run was blind or context-aware;
- which solver/profile/model/engine ran;
- normalized ciphertext conventions;
- top candidates with plaintext previews and typed keys;
- replay verification status;
- score components and candidate diversity;
- period/column/filler assumptions;
- why the run accepted, abstained, or referred;
- known limitations for the input length and language.

`scripts/inspect_artifact.py` must learn any new result fields when integration
lands. That analyzer work belongs to PF-7, not the isolated solver slices.

## 11. Testing Layers

1. Primitive tests: existing known-answer and round-trip tests remain green.
2. Key-contract tests: serialization, validation, replay, equivalence.
3. Scorer tests: true/near/random/wrong-family ranking packets.
4. Search unit tests: seeded reduced-budget recovery and budget termination.
5. Fresh synthetic acceptance: multiple unseen seeds per family.
6. Confusion tests: wrong-family probes and honest abstention.
7. Automated-runner tests: explicit and blind routes, zero LLM usage.
8. MCP/fake-client tests after PF-7.
9. Overnight stochastic suite after PF-8.

No famous historical cipher is sufficient acceptance evidence. Historical and
external-tool examples are compatibility checks after synthetic acceptance.

## 12. Rollout and Status Semantics

<!-- FABLE REVIEW [MAJOR] status-vocabulary mismatch — RESOLVED via §1a(a):
mapped onto the live enum, maturity ladder moved to the eval artifact, and a
_validate_registry enum check proposed below. Original review text retained for
audit trail: -->
<!-- ORIGINAL: FamilySpec.solver_status takes one of {solves_automated,
agent_assists, diagnoses_only, planned, unsupported}; the section invented
probe_available/experimental_solver. Options were (a) extend the enum or (b) map
onto it + separate report field. Chose (b) per §1a(a) — the composite spec uses
agent_assists, so (b) is the lower-friction coordinated choice, and we deliberately do not
extend the shared enum for both programs. -->

Per coordination decision §1a(a), the registry `solver_status` uses ONLY the
live enum `{solves_automated, agent_assists, diagnoses_only, planned,
unsupported}`. A family's registry status progresses:

- `diagnoses_only` → the starting point (detector/probe may exist, no reliable
  recovery);
- `agent_assists` → a solver exists and a fresh-evaluation report shows it
  recovers with agent/experiment help (the maturity a bounded probe or a
  narrow solver reaches);
- `solves_automated` → fresh held-out evaluation clears the common strong gate
  AND runtime is operationally acceptable.

The richer research distinction — `probe_available` (bounded inversion gives
evidence, recovery unreliable), `experimental_solver` (measured recovery clears
the milestone's LOWER gate), `solves_automated` (clears the STRONG gate) — is a
per-milestone MATURITY signal recorded in the eval/calibration ARTIFACT
(§8 report schema), never in `FamilySpec.solver_status`. Registry status
changes only after the corresponding report exists. A family may ship a strong
generator and detector while remaining `diagnoses_only`.

<!-- FABLE REVIEW [MAJOR] status-vocabulary mismatch — RESOLVED: mapped onto the
live enum per §1a(a); the three-way maturity ladder now lives in the eval
artifact, not solver_status. -->

Note (no _validate check today): `solver_status` strings are unvalidated in
`families.py` — a typo would pass silently. This program SHOULD add a
one-line enum check to `_validate_registry()` when it first touches the
registry (cheap insurance now that three programs write statuses).

<!-- FABLE R2: OK — verified the live enum is exactly {solves_automated,
agent_assists, diagnoses_only, planned, unsupported} (families.py:59) and that
_validate_registry() (families.py:398-440) does NOT check solver_status today, so
the "no _validate check" note is accurate and the maturity ladder correctly lives
in the §8 eval artifact, not solver_status. The proposed enum check is sound; when
adding it, seed the allowed set from ALL five values in the FamilySpec docstring
(NOT only the values in current use — `planned` is presently used by
DiscriminatorSpec.status, not any FamilySpec, but must stay legal for families).
Coordinate landing with the composite program: whichever of the three specs
touches _validate_registry first adds the check; the others must not re-add it. -->


## 13. Review Questions Before Implementation

1. Does the existing no-boundary language scorer reliably rank true and
   near-true candidates for every family, especially Trifid and ADFGVX?
2. Should square/cube state use strings for reference simplicity or compact
   integer arrays from the start?
3. Which key symmetries must be canonicalized for each family?
4. What medium/long length bands produce meaningful but nontrivial acceptance
   cases?
5. Can the existing transposition solver expose a reusable finalist API for
   ADFGX/ADFGVX without duplicating search?
<!-- FABLE REVIEW: [QUESTION -> partially answered] Yes for the finalist MENU
skeleton (analysis/pure_transposition.py: screen_pure_transposition returns a
ranked top_candidates dict; generate_pure_transposition_candidates +
PureTranspositionSearchConfig give candidate-only generation). NO for the search
FAMILY you need: its candidates are geometric transforms, it has no keyword-
columnar enumerator, and it requires A-Z + a language score (see PF-6 [MAJOR]).
So the honest answer is "reuse the finalist plumbing, write the columnar-
inversion search and its non-language scoring yourself." -->

6. Which solver stage becomes CPU-bound first, and what evidence justifies a
   Rust port?
7. Should Hill remain a separate matrix family in investigator reports rather
   than being grouped under square-based polygraphics?
<!-- FABLE REVIEW: [QUESTION -> this is really a BLOCKER for PF-7, not an open
musing]. Today Hill has NO family id — it rolls up under `polygraphic_substitution`
(see PF-7 [MAJOR]). Whatever the answer, the registry cannot express a Hill-
specific solver_status until a Hill id/subtype exists. Resolve this before PF-7
so "family by family" status updates are possible. My recommendation: give Hill
its own primary (its key model, attack, and keyspace are unrelated to the 5x5
square families), and split `fractionation_transposition` into per-family subtypes
too. -->

8. What calibrated margin permits exact-family confidence versus only broad
   `polygraphic` or `fractionation` suspicion?

## 14. Recommended First Implementation Packet

When work begins, implement only PF-0 and PF-1 in the first change set:

- shared typed contracts;
- prepared-text normalization;
- scorer oracle report;
- generated held-out packet definition;
- exhaustive Hill 2x2 solver;
- explicit automated Hill route;
- fresh-seed Hill and negative-control report.

Then review the scorer and contracts before starting Playfair annealing. This
keeps the first slice small enough to expose architectural mistakes without
committing the entire program to them.
