# K4 mechanism-search tooling: external project review

Reviewed 2026-09-07. This review recommends capabilities; it does not attempt a K4 solve or change Decipher's development queue.

The strongest opportunity is to implement Decipher's already-planned **known-plaintext mechanism-recovery mode**, then equip it with algebraic constraint tests, calibrated statistical experiments, and explicit accounting for fitted parameters. The reviewed projects offer useful experiments, reference implementations, and audit practices. None of the reviewed material establishes a recovered K4 mechanism that Decipher should adopt as fact.

Existing repository files were treated as read-only. This report is the only repository file created by this review. External checkouts, downloads, compilation, and test caches were placed under `/private/tmp/k4-capability-review.BUVGhz`. No paid solver runs or external submissions were made.

## Scope and evidence

| Source | Snapshot inspected | Main value |
|---|---|---|
| [doranchak/kryptos](https://github.com/doranchak/kryptos/tree/d192cc72bf2019f043804cbac66fd389cb4ef7bd) | `d192cc72bf2019f043804cbac66fd389cb4ef7bd`, dated 2026-09-07 | Synthetic ciphers, independent reference vectors, mechanism traces, anomaly exploration |
| [RichardBean/k4testing](https://github.com/RichardBean/k4testing/tree/6a5e3cb200d5ab72a62bb7f5124b4fdf163faf8c) | `6a5e3cb200d5ab72a62bb7f5124b4fdf163faf8c`, dated 2023-09-12 | Permutation tests and Gromark primer constraints |
| [glthr/K4nundrum](https://github.com/glthr/K4nundrum/tree/e5356d6ec8c1e93845542e3be1faaf55b7a90400) | `e5356d6ec8c1e93845542e3be1faaf55b7a90400`, dated 2024-03-31 | Separator/group hypotheses and substitution-invariant frequency shapes |
| [SolveKryptos](https://solvekryptos.com/) | Live pages and canonical bundle labeled v15 | Reproducible reconstruction, provenance labels, falsification records, physical-coordinate hypotheses |

The downloaded [SolveKryptos bundle](https://solvekryptos.com/downloads/solvekryptos-canonical-bundle.zip) had SHA-256 `3c87e6ab9976df562accf6cd31f58c3dec1bfc3517ff9602e6294b5b86b2c5af`. Its files were dated through July 2, 2026. The site and bundle are separately versioned evidence; neither should be assumed immutable.

Local comparison used the working tree based on `c4554c5323a64d0a0662220ad379f07dc427dc2a`, including existing uncommitted work. Other work changed files during the review, so local references describe the inspected implementation, not a frozen release.

An important distinction for the proposed workflow: supplied plaintext can legitimately be input to mechanism recovery. Its evidential status still matters. SolveKryptos explicitly identifies only the 24 anchor positions as artist-confirmed and labels its remaining plaintext a reconstruction. Its text should therefore be a candidate hypothesis, not the source of authenticated full plaintext. If the user supplies stronger evidence, record that source separately. [Site statement](https://solvekryptos.com/), [claim-history explanation](https://solvekryptos.com/about#what-changed-on-this-site).

## What Decipher already has

There is substantial overlap, which changes the adoption recommendation:

- [The capability plan](/Users/mgreen/Dropbox/src2/decipher/docs/polyalphabetic_capability_plan.md:464) already specifies supplied-plaintext mechanism recovery, correspondence tests, effective keystreams, degrees of freedom, held-out prediction, and separate artifacts. [TODO](/Users/mgreen/Dropbox/src2/decipher/TODO.md:410) records it as unfinished. I did not find a first-class implementation of this mode in the inspected CLI or investigation surface.
- [Polyalphabetic analysis](/Users/mgreen/Dropbox/src2/decipher/src/analysis/polyalphabetic.py:248) implements Quagmire I–IV alphabet/replay semantics, ordinary periodic variants, and Quagmire III search. It also has offset/constraint-graph helpers. Broader Quagmire replay is already available even though the main specialized search experiment is Quagmire III.
- [Transform pipelines](/Users/mgreen/Dropbox/src2/decipher/src/analysis/transformers.py:42), pure-transposition search, and transposition solvers already cover many grid, route, columnar, and composite operations. Applying a pipeline to index tokens provides a permutation trace. [Candidate packets](/Users/mgreen/Dropbox/src2/decipher/src/investigation/candidates.py:45) already carry order, rendering, and provenance fields.
- [The family generator registry](/Users/mgreen/Dropbox/src2/decipher/src/testgen/family_registry.py:242) already includes running key, both autokey variants, Porta, Playfair, Hill, Bifid/Trifid, ADFGX/ADFGVX, and transposition families. Generator coverage should not be confused with solver coverage.
- [Investigation experiments](/Users/mgreen/Dropbox/src2/decipher/src/investigation/experiments.py:546) already provide background execution, typed configuration, deduplication, and result collection. These are useful foundations for bounded mechanism experiments.

The [current program](/Users/mgreen/Dropbox/src2/decipher/docs/improvement_program_plan.md:3) governs implementation order through R0–R4. The priorities below are recommendations for a subsequent K4 capability slice, not an instruction to interrupt that work.

## Project findings

### 1. doranchak/kryptos: strongest source of calibration and inspection tools

The substantial component is `ciphers/cipher-generator`, beyond the small top-level README. It offers 30 cipher types, manual replay, bulk generation, and exports. Of particular interest are running-key alphabet variants I–IV, the ACA contiguous-passage running-key construction, and both orders of running key plus transposition. Many other cipher primitives overlap Decipher. [Generator documentation](https://github.com/doranchak/kryptos/blob/d192cc72bf2019f043804cbac66fd389cb4ef7bd/ciphers/cipher-generator/README.md).

Its visualizer traces letters through key tables and intermediate stages. This suggests a useful Decipher inspection operation: given a candidate and a position, explain the original ciphertext coordinate, intermediate positions, alphabet indices, key values, and final character. Existing transform index maps can supply much of this; the missing layer is a uniform explanation across composed mechanisms. [Visualizer implementation](https://github.com/doranchak/kryptos/blob/d192cc72bf2019f043804cbac66fd389cb4ef7bd/ciphers/cipher-generator/js/visualizer.js).

I ran the supplied Node self-test: **773 checks, zero failures**. It includes round trips and external/hand-worked reference vectors. This is useful compatibility evidence, not proof that all supported variants or the browser UI are correct. [Test source](https://github.com/doranchak/kryptos/blob/d192cc72bf2019f043804cbac66fd389cb4ef7bd/ciphers/cipher-generator/scripts/test_ciphers.js).

Two headerless fixture files each contain **19,185** 97-character plaintext/ciphertext pairs, with keys supplied and both K4 crib spans preserved. I replayed **all 38,370 rows through Decipher**, obtaining exact ciphertext agreement throughout. Each file contains 19,181 unique plaintexts, so a benchmark importer must group duplicates; it should also check overlap between files. These are positive controls for specified Quagmire mechanisms, not independent evidence that K4 uses those mechanisms. [Quagmire III fixtures](https://github.com/doranchak/kryptos/blob/d192cc72bf2019f043804cbac66fd389cb4ef7bd/ciphers/test-ciphers/k4-quagmire3-19185-test-ciphers-with-cribs.csv), [Quagmire IV fixtures](https://github.com/doranchak/kryptos/blob/d192cc72bf2019f043804cbac66fd389cb4ef7bd/ciphers/test-ciphers/k4-quagmire4-19185-test-ciphers-with-cribs.csv).

The generator samples exact-length whole-word passages and rejects some low-entropy/repetitive samples. Most randomness uses `Math.random`; the CSV export omits the source location and seed, even though some source information exists internally. For Decipher evaluation, preserve corpus provenance, sampling filters, normalized plaintext hashes, structured keys, and deterministic seeds. Keep ordinary generated texts alongside the K4-crib-conditioned set. [Sampling/export code](https://github.com/doranchak/kryptos/blob/d192cc72bf2019f043804cbac66fd389cb4ef7bd/ciphers/cipher-generator/js/generator.js).

**Concrete interoperability finding:** Decipher's Quagmire I replay does not reproduce the ACA `SPRINGFEVER` / `FLOWER` example with those parameters unchanged: all 83 characters differ by +9 modulo 26. Doranchak subtracts the plaintext alphabet's A-position, which is 9; Decipher's generic replay adds the cycleword index without that adjustment. Translating the cycleword to `WCFNVI` makes Decipher reproduce the ACA ciphertext exactly. Thus this is an indicator-alignment convention gap, not inability to represent the cipher. Add an explicit indicator-under-letter/origin convention and conversion tests; preserve existing K1 semantics. The ACA permits placement under different plaintext letters. [ACA definition](https://www.cryptogram.org/downloads/aca.info/ciphers/QuagmireI.pdf), [Doranchak implementation](https://github.com/doranchak/kryptos/blob/d192cc72bf2019f043804cbac66fd389cb4ef7bd/ciphers/cipher-generator/js/ciphers.js#L484-L516), [Decipher encoder](/Users/mgreen/Dropbox/src2/decipher/src/analysis/polyalphabetic.py:325).

Two other tools suggest reusable experiment types. The Graham-Cumming explorer scans pairs of text windows and rotations, ranks the modulo-26 **sum** by entropy, and exports results. The difference simulator instead measures the maximum frequency of a **signed, non-modular difference** across rotations of random strings. Preserve these definitions explicitly: these tools compute different statistics and cannot be treated as interchangeable keystream tests. Generalize their window/rotation scans into a bounded operation whose entire search is also run on controls. [Window experiment](https://github.com/doranchak/kryptos/blob/d192cc72bf2019f043804cbac66fd389cb4ef7bd/john-graham-cumming-pattern/index.html#L326-L395), [Difference simulation](https://github.com/doranchak/kryptos/blob/d192cc72bf2019f043804cbac66fd389cb4ef7bd/difference-simulator/index.html#L269-L329).

**Adoption:** reference vectors, fixture evaluation, and trace design are high value. Build on Decipher's primitives; a second embedded cipher engine would add maintenance without a demonstrated solving gain. No repository-level license file was present in the inspected tree. Obtain explicit permission before copying code or redistributing datasets, and review bundled corpus rights separately.

### 2. RichardBean/k4testing: strongest source of new cryptanalytic probes

This is research code for specific statistical and algebraic questions. Bean's HistoCrypt paper explains the permutation-testing approach and proposes Gromark as a possible explanation; that proposal is a hypothesis, not a recovered K4 method. [Paper abstract](https://ecp.ep.liu.se/index.php/histocrypt/article/view/153).

The useful diagnostics include repeated digrams at a specified lag, distances between ciphertext letters associated with repeated/nearby crib letters, and repeat counts after row/column rearrangements. Different experiments shuffle ciphertext, alphabet order, rows, or generate uniform letters. These alternatives illustrate why a reported rarity must identify what was randomized and what stayed fixed. [Crib/lag tests](https://github.com/RichardBean/k4testing/blob/6a5e3cb200d5ab72a62bb7f5124b4fdf163faf8c/k4-testy.c), [Alphabet controls](https://github.com/RichardBean/k4testing/blob/6a5e3cb200d5ab72a62bb7f5124b4fdf163faf8c/hannon-test.c), [Transposition tests](https://github.com/RichardBean/k4testing/blob/6a5e3cb200d5ab72a62bb7f5124b4fdf163faf8c/k4-perm-test.c).

`gt.c` enumerates primers for the recurrence `k[i] = (k[i-L] + k[i-L+1]) mod base`, then applies hardcoded equalities/inequalities derived from the two K4 crib spans. Compiled with `clang -std=gnu89 -O2`, `gt 10 5` reproduced the documented **39 survivors**. This confirms the filter's output, not 39 completed decryption mechanisms: it prints primers/streams, not complete consistent alphabet assignments and decoded texts. [Primer filter](https://github.com/RichardBean/k4testing/blob/6a5e3cb200d5ab72a62bb7f5124b4fdf163faf8c/gt.c).

The accompanying Sage equations reveal the more general opportunity: derive constraints from arbitrary supplied crib positions and shared plaintext/ciphertext alphabet variables. Replace hand-maintained K4 index equations with a generic constraint compiler. Use cheap necessary tests first, then a complete feasibility check that enforces distinct alphabet positions before accepting a candidate. Gromark digit-class IC is another cheap ranking signal; retain its grouping/averaging definition in results. [Equation scaffold](https://github.com/RichardBean/k4testing/blob/6a5e3cb200d5ab72a62bb7f5124b4fdf163faf8c/kryptos-k4-sage.txt), [Grouped IC](https://github.com/RichardBean/k4testing/blob/6a5e3cb200d5ab72a62bb7f5124b4fdf163faf8c/grom-ic-list.c).

Most executables have embedded ciphertext, fixed large trial counts, and time-based seeds. Several report reciprocal hit rates without safe zero-hit handling. I did not rerun their million/billion-trial statistical claims. A production port needs bounded budgets, recorded seeds, completed-trial counts, uncertainty intervals, and full experiment-family correction. The Python minor-distance statistic depends on alphabet order; its embedded LLM explanation suggesting generic simple-substitution diagnosis should not become Decipher's interpretation. [Python experiment](https://github.com/RichardBean/k4testing/blob/6a5e3cb200d5ab72a62bb7f5124b4fdf163faf8c/k4testing.py).

**Adoption:** generalize the algebra and statistical tests. A bounded Gromark search is a worthwhile later family addition, not a reason to route K4 to Gromark by default. The repository contains a [GPLv3 license](https://github.com/RichardBean/k4testing/blob/6a5e3cb200d5ab72a62bb7f5124b4fdf163faf8c/LICENSE); retain attribution and review component origins if reusing code.

### 3. K4nundrum: useful structural observation, limited statistical implementation

The program splits on each candidate separator, enumerates segment permutations and equal-length groupings, and compares sorted letter-count distributions. This forgets letter identity: two groups can have identical count shapes under different substitutions. It additionally checks segment lengths and alternating group membership. [Grouping code](https://github.com/glthr/K4nundrum/blob/e5356d6ec8c1e93845542e3be1faaf55b7a90400/groups/groups.go), [Frequency comparison](https://github.com/glthr/K4nundrum/blob/e5356d6ec8c1e93845542e3be1faaf55b7a90400/frequencies/frequencies.go).

All three Go package test suites passed. Running the default analysis reproduced the W result: five separators, six segments, and two 46-letter groups with equal frequency shapes. This is a reproducible observation, but equal count shapes do not establish aligned text, a shared key, or a separator's cryptographic role. [Program](https://github.com/glthr/K4nundrum/blob/e5356d6ec8c1e93845542e3be1faaf55b7a90400/main.go).

There are two material limitations to adopting its reported simulation percentages:

- Its random strings use independent uniform A–Z draws, rather than preserving K4's histogram. Both null models can be useful, but answer different questions. [Generator](https://github.com/glthr/K4nundrum/blob/e5356d6ec8c1e93845542e3be1faaf55b7a90400/helpers/generators.go).
- `Record` increments once for each matching collection, while the denominator counts generated ciphertexts. There is no trial-ID deduplication at the recorder, and generation can run ahead of completed analysis. Consequently the counters do not by construction estimate the fraction of fully tested random ciphertexts having at least one match. Recompute those probabilities with one completed-trial record per sample before citing them. [Recorder](https://github.com/glthr/K4nundrum/blob/e5356d6ec8c1e93845542e3be1faaf55b7a90400/helpers/statistics.go#L213-L241).

**Adoption:** add a bounded separator/group diagnostic with positional provenance, then test each hypothesis with a model that predicts more than frequency shape. Use canonical partitions, cached count vectors, and symmetry reduction instead of generating all segment permutations. Keep repeated segment occurrences distinct by position. A proposed null deletion must retain original crib coordinates and pass length/correspondence checks; dropping five symbols cannot silently remain a 97-to-97 one-letter mapping. This is useful beyond K4 for interleaved messages and state-changing separators. The code is [MIT-licensed](https://github.com/glthr/K4nundrum/blob/e5356d6ec8c1e93845542e3be1faaf55b7a90400/LICENSE).

### 4. SolveKryptos: most valuable as a mechanism-audit case

The bundle supplies a runnable forward decoder, an arithmetic verifier, public anchors, reconstructed tables, physical-side transcriptions, a file-status index, negative-test summaries, and preregistered field/prediction protocols. Those audit artifacts are useful models for Decipher output. The site acknowledges that its cards were fitted using the proposed plaintext. [Downloads and changelog](https://solvekryptos.com/resources), [Verification explanation](https://solvekryptos.com/verify).

Both bundled Python programs ran successfully. The forward decoder passed its anchor checks. The verifier reported 97/97 arithmetic matches, 97/97 decomposition matches, and 31/31 footer-handoff matches. These are replication results for the supplied reconstruction. In `verify.py`, the decomposition check computes `r = R - gate` and then adds the gate back, so that particular check is an identity; it does not independently validate the helper-card mechanism. [Bundle containing both scripts](https://solvekryptos.com/downloads/solvekryptos-canonical-bundle.zip).

I independently inspected the additive dependency graph of `forward_decode.py`. Its operative formula is:

`P[i] = C[i] - F[C[i]] - G[state(i), helper(i)] - gate[i]  (mod 26)`

Treating active F and G entries as free parameters gives **108 parameter nodes, 97 observation edges, 11 connected components, and no cycles**. Each component is a tree. With the gate and geometry fixed, one can select a root value and solve outward to fit any assigned 97-element shift stream. There are no surplus cycle constraints to reject an arbitrary stream in this relaxed model. This independently reproduces the site's stated forest limitation. It does not establish the freedom remaining after imposing every additional constraint claimed by the site. [Decoder in bundle](https://solvekryptos.com/downloads/solvekryptos-canonical-bundle.zip), [site's July 10 discussion](https://solvekryptos.com/resources).

This is a particularly good negative control: a program can have no plaintext input at execution time while its constants encode a plaintext-dependent fit. Verification must trace the origin of constants and model choices, not merely inspect the runtime function signature. Also, a stream's high empirical entropy is not a lower bound on the size of its generating program; avoid turning the bundle's descriptive entropy arguments into mechanism exclusions.

**Adoption:** parameter/dependency audits, explicit claim grades, position-by-position traces, negative results, and prediction commitments. Treat this particular reconstruction as a hypothesis and test fixture. Do not import its full plaintext, helper cards, gate geometry, or “Quagmire III variant” label as established K4 facts. The inspected bundle contained no explicit reuse license; copying its artifacts requires a separate rights decision.

## Recommended capability package

These are proposed operations, not currently available commands. Most computation can be exposed through the existing experiment queue; inexpensive inspection can remain synchronous. Any new public operations should flow through the shared investigation manifest to maintain CLI/MCP parity.

| Priority within a future mechanism slice | Capability | Minimum useful output |
|---|---|---|
| 1 | Explicit supplied-plaintext evidence and correspondence model | Source, confidence/mask per position, index map, alphabet/arithmetic convention, fit/holdout split |
| 2 | Effective-stream and algebraic constraint tests | Derived values, contradictions, surviving parameter domains, equivalent keys, feasibility status |
| 3 | Mechanism identifiability and prediction audit | Free versus fixed parameters, provenance dependencies, constraint redundancy, held-out predictions, alternatives |
| 4 | Reproducible statistical experiment harness | Statistic definition, null generator, seed, completed trials, exceedances, intervals, search multiplicity |
| 5 | Separator/segment and layered-transform experiments | Candidate grouping/order, full coordinate trace, constraints consumed, downstream predictive test |
| 6 | External calibration fixtures and mechanism explanations | Independent replay vectors, mechanism-recovery metrics, per-position human/agent explanation |

### Evidence and correspondence first

Implement the existing plan's separate run mode. Distinguish artist-confirmed cribs, user-supplied plaintext, tentative reconstructions, contextual statements, and parameters fitted from any of them. Unknown letters need masks; competing proposed readings need distinct hypotheses. Preserve the original blind investigation and mark new results as non-blind mechanism work.

For each hypothesis, specify ciphertext-to-plaintext correspondence before computing a stream: direct order, offsets, reversal, supported transpositions, or explicitly modeled omissions/expansions. Retain inscription/source positions through every step. For ordinary numeric alphabets, derive Vigenere `K=C-P`, Beaufort `K=C+P`, and Variant Beaufort `K=P-C`; keyed alphabets require their own index maps and indicator origin. Equal-looking numerical streams under different conventions are not automatically equal mechanisms.

### Constraint and prediction tests

Begin with periodic keys, affine/progressive rules, and supplied keyed alphabets; then add bounded Gromark recurrences. Use crib contradictions to reject models cheaply before language scoring. For unknown alphabets, enforce bijections where the family requires them. Modular arithmetic modulo 26 needs appropriate treatment—such as solving modulo 2 and 13 with reconciliation, or a finite-domain solver—not ordinary real-valued rank calculations.

Require fit partitions and model-family limits to be fixed before prediction tests. Predict held-out ciphertext/plaintext positions without consulting their labels. Track model selection on holdouts too: a split repeatedly used to choose mechanisms has become tuning data. Group splits by repeated key/helper state where needed, and report uncovered states as underdetermined. Exact agreement on a reused fitted lookup is weaker than predicting an unobserved relation.

Report fit consistency, holdout performance, complexity, equivalence classes, and historical/authorship evidence separately. A successful predictive mechanism can still be one of multiple compatible mechanisms. When models disagree only at unknown positions, identify the positions or source observations that would best distinguish them.

### Statistical controls that reflect the search

Support histogram-preserving ciphertext shuffles, independently randomized alphabets, geometry-preserving row/block permutations, uniform strings, and synthetic encryptions from declared families. Choose the null according to the claim. For a search over offsets, widths, separators, thresholds, or alphabets, repeat that whole bounded search for every control and compare the best statistic, rather than comparing a selected winner against one fixed test.

Record completed trials and one any-hit/best-statistic result per trial, with seeds and deterministic sharding. Report raw exceedance counts and Monte Carlo uncertainty; zero hits are not zero probability. Keep explored variants and unsuccessful experiments in a searchable ledger with their exact scope. “No survivor in these 100,000 primers under these alphabets” is useful; “Gromark disproved” is not warranted by that result.

### Calibration and reuse

Use the independently implemented Quagmire fixtures to test replay, key recovery, search recall, and candidate preservation separately. Add corpus/source-grouped splits and fresh generated texts without K4 anchors; exclude solution fields from blind runs. Add running-key alphabet variants and both transposition orders only where they fill gaps in the existing generator/composition system. Full running-key brute force is not implied by adding its generator.

Useful negative controls include random keystreams, deliberately overparameterized lookup mechanisms, and proposed plaintexts with correct cribs but incompatible remaining structure. The SolveKryptos additive forest is an unusually clear test that the audit can distinguish fitted replication from constrained prediction. Reimplement a generic mathematical fixture if artifact reuse rights remain unresolved.

## Verification record and limits

| Check performed | Result | What it establishes |
|---|---|---|
| Doranchak Node self-tests | 773 checks, zero failures | Supplied implementation passes its included suite in this environment |
| Both external Quagmire III/IV CSVs through Decipher | 38,370/38,370 exact replays | Compatible encryption semantics for those rows and supplied keys |
| ACA Quagmire I vector through Decipher | 83/83 differ by +9; translated indicator matches | Specific indicator-origin interoperability gap |
| Bean `gt 10 5` | 39 surviving primers | Reproduction of hardcoded filter result; no completed mechanism proof |
| K4nundrum package tests and default analysis | Tests pass; W grouping reproduced | Implementation reproduces the structural observation |
| SolveKryptos forward decoder and verifier | Supplied checks pass | Internal reconstruction replication |
| Independent F/G graph construction | 108 nodes, 97 edges, 11 trees | Unconstrained card model can fit arbitrary shifts |

The Node suite includes randomness; the reported count is from one run. Go emitted a sandbox-denied telemetry-token warning, but its tests and analysis exited successfully. Bean's C compilation emitted legacy-prototype and printf-type warnings. No browser interaction, statistical mega-run, general solver-performance benchmark, complete license audit, or independent physical/archival validation was performed.

Recommended first deliverable after the governing reliability work: a small non-blind mechanism command that accepts masked supplied plaintext, derives convention-labeled streams, tests periodic/affine models, and produces a frozen-fit prediction report. Include the Quagmire indicator convention check and an overfitting negative control from the outset. Extend it with Bean-style generic constraints and calibrated group/geometry experiments once that evidence contract works.
