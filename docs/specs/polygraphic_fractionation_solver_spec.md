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

## 2. Current Baseline

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

- Enumerate all 157,248 invertible matrices in `GL(2, Z_26)`.
- Decrypt and score each matrix.
- Retain top K candidates and equivalent/near-equivalent keys.
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

### PF-6: ADFGX and ADFGVX composition

Architecture:

- Represent each as `fractionation -> columnar transposition`.
- Reuse the accepted transposition solver rather than writing an opaque
  monolithic attack.
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

Gate:

- Component calibration modes recover their unknown component on fresh cases.
- Blind `full` mode is tested on at least 20 cases per family across column
  counts and incomplete-grid shapes.
- Experimental support requires median >= 0.75 and top-5 capture >= 0.85;
  automated-solved status requires median >= 0.90.
- Artifacts show both layer keys and replay the complete forward/decrypt
  pipeline.
- Single-layer controls are not falsely reported as composed solutions.

### PF-7: Automated routing and investigator/MCP probes

This milestone starts only after the relevant solver acceptance gates pass and
the MCP host surface is stable.

Deliverables:

- Explicit-family automated routes for accepted solvers.
- A bounded broad-family probe that can compare candidate families without
  reading benchmark labels.
- Update registry solver statuses from measured evidence, family by family.
- Add experiment schemas and concise result packets through the shared
  investigation host/MCP surface. Do not expose raw mutation controls as a
  giant agent tool menu.
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

Use three statuses independently for each family:

- `probe_available`: bounded inversion can provide evidence but recovery is
  not yet reliable.
- `experimental_solver`: measured recovery clears the milestone's lower gate.
- `solves_automated`: fresh held-out evaluation clears the common strong gate
  and runtime is operationally acceptable.

Documentation and registry status change only after the corresponding report
exists. A family may have a strong generator and detector while remaining
unsupported for solving.

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
6. Which solver stage becomes CPU-bound first, and what evidence justifies a
   Rust port?
7. Should Hill remain a separate matrix family in investigator reports rather
   than being grouped under square-based polygraphics?
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
