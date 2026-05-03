# Automated Solver Comparison

Status: 2026-05-03

This document compares Decipher's current automated and agent-assisted
cryptanalysis capabilities with other classical-cipher tools. It is meant to
be candid rather than promotional: Decipher has become genuinely useful on
several hard families, but the mature tools still define important parity
targets, especially for breadth, speed, and open-ended search.

## Scope

This comparison covers classical-cipher automation and related workflows:

- no-LLM automated solving,
- solver-assisted candidate search,
- cipher-type diagnosis,
- benchmark/reproducibility support,
- agentic or human-in-the-loop workflows.

The tools below are not all trying to be the same thing. AZdecrypt is a
high-powered hillclimbing workbench. CryptoCrack and CrypTool are broad GUI
toolkits. Ciphey, dCode, Boxentriq, and quipqiup optimize for accessibility and
unknown-input triage. Decipher is currently a research CLI with benchmark
artifacts, historical-cipher support, and an LLM-agent layer.

## Executive Summary

Decipher's current strengths:

- Reproducible benchmark runs with artifacts, provenance, costs, branch state,
  and solver settings.
- Strong homophonic substitution support through the Zenith-compatible native
  scorer and Rust acceleration.
- Emerging transform + homophonic search: known replay, broad structural
  screening, finalist validation, and agent-facing transform tools.
- Periodic polyalphabetic support, including Vigenere-family solvers and a
  Rust Quagmire III search path that can solve Kryptos K2-style cases when the
  correct tableau family is in scope.
- Pure-transposition screening, including a broad Rust-scored path that solves
  the local Kryptos K3 fixture.
- Multisymbol benchmark support for historical manuscripts and non-Latin glyph
  alphabets.
- Agentic context handling: tiered benchmark context, related records,
  associated documents, branch hypotheses, and scoped inspection tools.

Where Decipher still trails mature tools:

- Breadth of cipher families. CryptoCrack, CrypTool, dCode, and Boxentriq cover
  many more classical types.
- Mature open-ended transposition search. AZdecrypt remains the clearest
  parity target for route/columnar/periodic transposition breadth and staged
  search ergonomics.
- Long-established GUI workflows and interactive solver controls.
- Packaged model/language resources and turnkey installation on non-developer
  machines.
- Unknown-cipher identification across a broad taxonomy.
- Exhaustive production-grade search budgets without requiring ad hoc scripts
  or environment variables.

## Tool Landscape

| Tool | Best Current Use | Relative Strengths | Decipher Parity Gaps |
| --- | --- | --- | --- |
| Decipher | Research-grade benchmarked automated and agentic decipherment | Artifacts, context tiers, multisymbol alphabets, agent tools, reproducible no-LLM and LLM runs, Rust kernels for selected families | Broader cipher taxonomy, faster search kernels, better UI, more automatic cipher identification |
| AZdecrypt | High-powered hillclimbing workbench for substitution, homophonic, and transposition mixtures | Mature solver families, strong n-gram scoring, many transform/transposition modes, detailed matrix/stat output, very relevant Zodiac heritage | Decipher needs broader route/columnar/periodic transposition families, better staged ranking, and fair external regression wrappers |
| CryptoCrack | Broad GUI solver for classical ciphers | Very wide classical-cipher coverage, multitasking GUI, language/dictionary data for English plus other languages | Decipher needs breadth, language resources, and family-specific UX before it can claim comparable generality |
| Zenith | Zodiac-style homophonic and transform-aware solver/reference | Strong homophonic scorer, transform vocabulary, useful external baseline for Zodiac-family work | Decipher has narrowed the scoring gap, but still needs more polish, speed, model management, and transform-search maturity |
| zkdecrypto-lite | Command-line Zodiac/homophonic baseline | Lightweight external baseline, useful bundled Zodiac and classical examples | Narrower than Decipher's current architecture, but still useful for regression and Zodiac-family sanity checks |
| CrypTool 2 | Educational visual cryptanalysis workbench | Broad classic and modern cryptography toolkit, visual workflows, extensive help, open source | Decipher is not an educational GUI; parity means better automated routing and coverage, not replicating the visual workbench |
| dCode | Web-first cipher collection and identifier | Huge online solver/identifier surface; recognizes many cipher families; low-friction UX | Closed implementation; Decipher should match the triage experience for research cases while retaining reproducibility |
| Boxentriq | Web cipher identifier and puzzle toolkit | ML/statistical cipher identification, local-browser tools, clear guidance to decoders | Decipher needs a better first-pass unknown-cipher hypothesis system and user-facing diagnostic reports |
| Ciphey | Automatic decode/decrypt pipeline for CTF-style inputs | A* search over decoding paths, cipher/encoding detection, caching, fast Rust implementation | Ciphey is less historical-cipher focused; Decipher should borrow search orchestration ideas for unknown-input workflows |
| quipqiup | Fast simple-substitution solver | Excellent low-friction cryptogram/patristocrat solving | Decipher already covers this family but should match the simplicity of the user experience |

## Detailed Notes

### AZdecrypt

AZdecrypt is the most important technical comparison for Decipher's current
transform-search work. Its public README lists many solver families beyond
plain substitution, including substitution combined with columnar
rearrangement, columnar transposition, crib grids/lists, nulls/skips,
polyphones, row-bound modes, sequential homophones, simple transposition,
sparse polyalphabetism, units, Vigenere, bigram substitution, higher-order
homophonic, and non-substitution scoring. It also includes dedicated columnar,
grid, periodic, and simple transposition modes.

This matters because Decipher's current broad transform and pure-transposition
screens are now plausible, but still comparatively young. Decipher should
borrow AZdecrypt's shape:

- enumerate many transform families explicitly,
- screen cheaply before expensive solving,
- persist matrix/stat artifacts,
- promote only strong finalists to full solver budgets,
- make search breadth a named, reproducible profile.

Parity target: Decipher should be able to run a benchmark packet where its
candidate families, top-N rankings, and solve outcomes can be compared directly
against AZdecrypt on substitution, homophonic, transposition-only, and
substitution+transposition cases.

### CryptoCrack

CryptoCrack is a breadth target. Its site describes a freeware classical
cipher-solving program that can solve more than 60 classical cipher types, often
without known plaintext or key length, and includes tools to assist manual
decipherment. It also provides English language data and optional data for
other languages.

Decipher's strength is not breadth yet. Its current advantage is a more modern
artifacted research workflow and agent integration. The parity path is to use
CryptoCrack's supported-family list as a capability map: explicitly mark which
families Decipher can solve, diagnose, partially assist, or does not support.

### Zenith

Zenith remains the closest homophonic reference. Decipher now has a
Zenith-compatible native scoring path and can use both Decipher-built and
external Zenith-format n-gram models. On the current frontier comparisons,
Decipher is often near parity or better in accuracy when using comparable
models, while external Zenith is usually faster and still slightly ahead on
some Zodiac-family cases.

Parity target: keep Decipher's scorer and search semantics faithful enough that
Zenith comparisons remain meaningful, while improving Decipher's distinct
strengths: artifacts, model provenance, benchmark context, branchable agent
repair, and transform-search orchestration.

### zkdecrypto-lite

zkdecrypto-lite is still useful as a thin command-line baseline for
Zodiac-style and general homophonic work. It is not the richest modern search
architecture, but its bundled examples have been valuable for importing real
challenge cases such as Zodiac variants, Dorabella, and related qualitative
tests.

Parity target: keep zkdecrypto-lite in the external baseline harness for
selected fast checks, but do not let it dominate every frontier run. It is best
used as a regression/reference lane rather than as the main design model.

### CrypTool 2

CrypTool 2 is a broad educational workbench with visual workflows, integrated
help, and tools for both classic and modern cryptanalysis. It is not directly
comparable to Decipher's CLI-first research workflow, but it is a useful
reminder that users benefit from explanation, visualization, and progressive
diagnostics.

Parity target: Decipher should not try to clone CrypTool's GUI. Instead, it
should improve its own terminal and artifact reporting so an analyst can see
why a cipher-family hypothesis was chosen, what tests were run, and what
evidence remains ambiguous.

### dCode and Boxentriq

dCode and Boxentriq are broad web-first references. dCode advertises an
identifier covering more than 200 ciphers and many online solvers. Boxentriq
offers a cipher identifier that uses statistical features and machine learning
to suggest likely cipher/encoding types, with better reliability on longer
texts.

These tools are especially relevant to Decipher's unknown-cipher plan.
Decipher currently has individual diagnostics and hypothesis branches, but not
yet a polished first-pass report that behaves like a good cipher identifier.

Parity target: Decipher should produce a ranked hypothesis card for unknown
inputs, with recommended next tools, confidence caveats, and an explicit record
of which cipher families have not yet been tested.

### Ciphey

Ciphey is not primarily a historical-cipher solver; it is an automated
decode/decrypt/crack pipeline for CTF-style inputs. Its useful design ideas are
search orchestration, caching, path search, timeout handling, and plaintext
detection. Its current Rust-oriented implementation also highlights the benefit
of moving hot loops and candidate evaluation out of Python.

Parity target: Decipher should borrow orchestration ideas for unknown inputs:
cheap triage first, cached intermediate states, bounded path search, clear
timeouts, and "why this path was tried" reporting.

### quipqiup

quipqiup is a narrow but excellent baseline for simple substitution
cryptograms, including word-boundary and no-boundary forms. Decipher can solve
the same general class, but quipqiup's value is its frictionless interface.

Parity target: for simple substitution, Decipher should make the happy path as
easy as quipqiup while retaining artifacts and benchmark discipline.

## Capability Matrix

| Capability | Decipher Today | Stronger Reference |
| --- | --- | --- |
| Simple substitution | Strong, including no-boundary and benchmarks | quipqiup for UX, AZdecrypt for solver workbench |
| Homophonic substitution | Strong, Zenith-compatible scorer, Rust path | Zenith, AZdecrypt, zkdecrypto-lite |
| Homophonic + transposition | Promising: known replay, broad/rank/search profiles, finalist validation | AZdecrypt and Zodiac-specific research tooling |
| Pure transposition | Emerging: broad Rust screen and K3 fixture | AZdecrypt, CryptoCrack, CrypTool |
| Periodic polyalphabetic | Strong for standard Vigenere-family synthetics | CryptoCrack, CrypTool, dCode/Boxentriq for breadth |
| Quagmire / keyed tableau | Emerging: Rust Quagmire III search can solve K2-style cases in scope | Specialized Kryptos/Quagmire solvers and Blake-style search |
| Fractionation / Polybius | Mostly planned | CryptoCrack, CrypTool, dCode, Boxentriq |
| Digraphic/polygraphic substitution | Mostly planned | CryptoCrack, CrypTool, dCode, Boxentriq |
| Nulls/errors/noisy transcription | Partial support through repair/agent tools | AZdecrypt has explicit null/skip families |
| Nomenclators/codebook hybrids | Planned only | AZdecrypt units, CryptoCrack/manual tools |
| Unknown-cipher identification | Early hypothesis scaffolding | Boxentriq, dCode, Ciphey |
| Historical benchmark artifacts | Strong Decipher advantage | No close equivalent found in this pass |
| LLM-agent context use | Strong Decipher advantage | No close equivalent found in this pass |

## Priority Roadmap To Parity

### P0: Make Current Claims Harder To Fool

- Keep the expanded frontier suite up to date with every newly supported family.
- Add comparison reports that make speed/accuracy/search-budget tradeoffs
  visible across Decipher, Zenith, zkdecrypto-lite, and future wrappers.
- Preserve successful Z340, K2, and K3 agentic/non-agentic runs as smoke or
  regression fixtures.
- Document when benchmark context, related records, or known cipher-type
  metadata was provided.

### P1: AZdecrypt-Style Search Breadth

- Expand pure-transposition families: route variants, grilles, rail/fence
  variants, periodic rule families, columnar variants, and keyed/unkeyed
  profiles.
- Expand transform+homophonic families beyond the current Z340-shaped search
  space.
- Improve finalist validation so top candidates are judged by structure,
  language score, local readability, branch diversity, and full-solve
  confirmation.
- Add explicit search profiles such as `screen`, `broad`, `deep`, and
  `overnight`, with estimated candidate counts and expected runtime.

### P2: Unknown-Cipher Triage

- Produce a first-pass ranked cipher-hypothesis report for every unknown input.
- Give the agent family-specific tool menus after it commits to a hypothesis,
  while allowing explicit hypothesis changes.
- Track coverage debt: families strongly suggested by diagnostics but not yet
  tested.
- Add a compact "why not solved yet" final report for failed or partial runs.

### P3: Breadth Against CryptoCrack/CrypTool/dCode

- Add fractionation/Polybius families: Bifid, Trifid, ADFGX, ADFGVX.
- Add digraphic/polygraphic substitution: Playfair, Two-Square, Four-Square,
  bigram substitution.
- Add null/error/noisy-transcription profiles.
- Add nomenclator/codebook-hybrid scaffolding.
- Expand multilingual language models and dictionaries.

### P4: Packaging And Usability

- Make Rust fast modules easy to build and diagnose.
- Add a clearer terminal summary for long search jobs and failed runs.
- Provide a one-command "unknown cipher" workflow that is useful without
  knowing Decipher's internal flags.
- Keep raw artifacts machine-readable, but make human reports attractive enough
  for iterative research.

## Broad Search Pass Notes

The 2026-05-03 broad search pass did not uncover a clearly superior automated
classical-cipher tool missing from our map. The main additions or reminders
were:

- Boxentriq and dCode are important references for cipher identification UX.
- Ciphey is worth studying for automated path search, caching, timeouts, and
  plaintext detection, even though its focus is CTF-style decoding more than
  historical manuscript cryptanalysis.
- quipqiup remains the cleanest UX reference for simple substitution.
- AZdecrypt remains the strongest technical search-harness reference for our
  immediate transform/transposition work.

## Sources

- [AZdecrypt GitHub](https://github.com/doranchak/azdecrypt)
- [CryptoCrack site](https://sites.google.com/site/cryptocrackprogram/)
- [Zodiac Killer Ciphers Wiki software tools](https://zodiackillerciphers.com/wiki/index.php?title=Software_Tools)
- [CrypTool 2 overview](https://www.cryptool.org/en/ct2/)
- [Ciphey GitHub](https://github.com/bee-san/Ciphey)
- [quipqiup](https://www.quipqiup.com/)
- [dCode](https://www.dcode.fr/en)
- [Boxentriq cipher identifier](https://www.boxentriq.com/analysis/cipher-identifier)
- Local Decipher notes:
  - `docs/frontier_solver_comparison.md`
  - `docs/transposition_homophonic_capability.md`
  - `docs/polyalphabetic_capability_plan.md`
  - `docs/unknown_cipher_agent_plan.md`
