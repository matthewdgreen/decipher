# INV Cipher-Family Roadmap

Updated 2026-09-01 after review of the landed INV-0 system, the 35-family
generator, the no-LLM solver coverage sweep, the six-case model-diagnosis
experiment, and the source-indexed mechanism reported for Urquhart's Cyphral
Distich.

The original request was to list cipher families Decipher does not yet address,
starting with families supported by other tools. That inventory remains useful,
but family count is not the primary progress measure. INV must distinguish what
the available evidence can support without manufacturing false precision.

Reference tools: AZdecrypt, CrypTool 2, the ACA cipher-type list, Ciphey, and
CipherLens.

## Support levels

Every entry is tracked at the correct level of the mechanism hierarchy:

1. **Representation**: text, digits, glyph tokens, Base64, hex, Morse, and
   similar deterministic encodings.
2. **Broad family**: substitution, transposition, periodic polyalphabetic,
   polygraphic, fractionating, numeric/codebook, or unknown/custom.
3. **Variant**: columnar vs railfence, Playfair vs two-square, Vigenere vs
   Beaufort, and similar close relatives.
4. **Composition**: substitution+transposition,
   transposition+homophonic, fractionation+transposition, and other layered
   mechanisms.
5. **Modifier/view**: nulls, noise, segmentation, transcription variants,
   layout, language, and page grouping.

The support matrix must distinguish:

- `detect`: cheap deterministic or statistical evidence for a broad family;
- `discriminate`: a calibrated test separating confusable hypotheses;
- `probe`: a bounded solver-backed inversion used when ciphertext statistics
  alone cannot identify the exact variant;
- `generate`: a parameterized benchmark counterpart;
- `solve`: an automated attack with measured recovery performance;
- `refer`: an honest external-tool recommendation when Decipher cannot solve.

An exact variant must not be reported confidently merely because it is listed
in the registry. Many close transposition variants preserve the same
low-order statistics and are distinguishable only by successful inversion.

## Current coverage

This is the planning-level summary. The next slice creates a generated,
machine-readable matrix so these facts stop drifting across documents.

| mechanism | detect/discriminate | generate | solve/probe |
|---|---|---|---|
| monoalphabetic substitution | landed | landed | landed |
| homophonic substitution | landed | landed | landed (`zenith_native`) |
| periodic polyalphabetic | landed broadly | 8 generated relatives | Vigenere/Beaufort/Variant/Gronsfeld landed; Porta/autokey/running-key gaps |
| broad transposition | generic suspicion only | 8 generated variants | columnar/railfence/redefence/Myszkowski/Amsco/nihilist-transposition landed; route and Cadenus remain gaps |
| substitution+transposition | coarse suspicion; model experiment missed it | suite-builder cases | transform search exists, composition diagnosis weak |
| transposition+homophonic | registry entry | ladder cases | transform-homophonic search exists, composition diagnosis weak |
| Playfair/polygraphic | registry cover only; discriminator planned | Playfair/two-square/four-square/Hill generated | not solved |
| fractionation+transposition | registry cover only; discriminator planned | Bifid/Trifid/ADFGX/ADFGVX generated | not solved |
| numeric book cipher | P8 diagnosis landed; structured-source matching absent | no general generator | diagnosis only; MCP numeric intake does not yet preserve multi-digit values |
| nomenclator/codebook | registry only | not generated | not solved |
| deterministic encodings | not first-class in INV | Base64/Base32/hex/binary/ROT47/Baconian/A1Z26/Morse/tap generated | detect-and-decode missing; some 1:1 forms happen to fall to substitution |
| plaintext/random/fabrication models | provisional hypothesis | random controls needed | never a conventional solve |

## Immediate enabling lane

These slices precede broad family expansion.

### A. Canonical taxonomy and coverage matrix

- Generate one authoritative matrix from the INV registry, generator registry,
  discriminator registry, and solver acceptance artifacts.
- Record hierarchy level, support mode, implementation status, measured power,
  applicable lengths/languages, and external referral.
- Make hand-written documents consume or summarize that matrix rather than
  maintaining independent status claims.

### B. Diagnosis calibration benchmark

- Expand beyond the current six-case model suite using fresh, held-out
  generator seeds.
- Cover multiple lengths, languages, key parameters, boundary conditions,
  noise levels, and deliberately confusable pairs.
- Include simple, composite, unsupported, and actually-random controls.
- Measure hierarchical top-1/top-k accuracy, abstention quality,
  false-confident rate, calibration, and discriminator power.
- Keep generator parameters and ground truth outside runtime diagnosis. They
  are post-hoc evaluation data only.
- A new discriminator cannot support a strong rule-out until its power and
  false-rule-out rate are measured on this suite.

### C. Unknown-language and transcription axes

Language, segmentation, transcription quality, and cipher family are separate
unknowns. They must be represented as orthogonal hypotheses/views rather than
silently fixing `language=en` and interpreting language-model failure as
cipher-family evidence. Prefer language-neutral family evidence first, then
compare plausible language-conditioned views.

## Tier 0: deterministic representation preflight

Implement cheap detect-and-decode checks for Base64/32/85, hex, binary, Morse,
Baconian, A1Z26, ROT47, tap code, and obvious mixed-length numeric encodings.
This runs before cryptanalytic diagnosis, prevents wasted investigations, and
provides the fair overlap needed for a later Ciphey comparison.

## Tier 1: highest-value diagnosis work

### 1. Broad transposition and variant probes

The next LLM-free discriminator work remains transposition, but with an honest
split:

- Static/order-layout evidence decides whether transposition is a live broad
  family.
- Bounded inverse screens for columnar, railfence, redefence, Myszkowski,
  Amsco, nihilist transposition, route, and Cadenus act as solver-backed
  variant probes.
- Exact subtype confidence comes from calibrated probe separation and readable
  inversions, not monogram statistics.
- Route and Cadenus remain solver gaps; the other six variants are regression
  anchors for diagnosis/probe behavior.

### 2. Layered/composite diagnosis

This is the highest demonstrated reasoning gap: every tested frontier model
missed substitution+transposition. Add compositional hypotheses over mechanism
layers and update them across `view_hash` transformations. Initial acceptance
must cover:

- substitution+transposition;
- transposition+homophonic;
- fractionation+transposition;
- null/noise overlays;
- negative controls where a single family is sufficient.

The report should say which layer is supported, which remains uncertain, and
which transformed view produced the evidence. Do not force every composition
into a permanently flat family enum.

### 3. Periodic and non-periodic Vigenere relatives

Generators already exist. Add calibrated distinction/probes for Porta,
text-autokey, ciphertext-autokey, and running key. Beaufort, Variant Beaufort,
Gronsfeld, and ordinary Vigenere are solved regression anchors. Absence of a
Kasiski peak is weak evidence, not a standalone autokey diagnosis.

### 4. Polygraphic and fractionating families

Proceed in confusable groups rather than isolated names:

- Playfair, two-square, and four-square;
- Hill 2x2 first, then larger Hill variants if justified;
- Bifid and Trifid;
- ADFGX and ADFGVX as explicit fractionation+transposition compositions;
- Nihilist substitution and straddling-checkerboard/VIC-style numeric systems.

Each group gets generated calibration cases, broad-family evidence, a
solver-backed discriminator where needed, and a solver or referral note.

### 5. Remaining common families

- Fractionated Morse, Morbit, and Pollux;
- grille and turning-grille transpositions;
- progressive-key/Gromark relatives;
- Base85 and other representation variants not covered in Tier 0.

## Tier 2: historical research frontier

### 1. Source-indexed numeric and book ciphers

Split this family into two engineering problems rather than treating every
book cipher as an open-ended corpus search:

1. **Local structured-reference ciphers**: the key text is adjacent, explicitly
   associated, or strongly suggested by the source. Position may select a page,
   paragraph, section, or line; the numeric token then selects a word or
   character. This is the first implementation target because the mechanism is
   bounded, falsifiable, and can be verified out of sample.
2. **Open-corpus key search**: the key text is unknown and must be searched over
   documented candidate corpora. This remains a later, much larger retrieval
   and indexing problem.

The local structured-reference lane is implemented as reusable machinery, not
as an Urquhart-specific solver:

- **N1 - numeric intake**: preserve each decimal integer as one token, retain
  line/group boundaries, and carry the original numeric value through CLI,
  MCP, case-file, artifact, and experiment interfaces. Add malformed/mixed
  numeric controls. Do not force numeric values through S-token identities or
  digit-by-digit letter parsing.
- **N2 - structured reference documents**: ingest user-approved source text as
  named pages/sections/paragraphs/lines while preserving edition, source URI,
  rights, tokenization policy, and stable unit ids. Host agents may read local
  files or retrieve public sources, but MCP receives inline content or opaque
  stored-document ids, never arbitrary filesystem paths.
- **N3 - contextual structure evidence**: compare ciphertext line/position
  counts and numeric ranges against available document-unit counts and unit
  lengths. Report cardinality matches, out-of-range rates, nearby textual
  clues, and counterevidence. A match proposes an experiment; it is not proof.
- **N4 - indexed-reference experiment**: test compact declared rules such as
  `position i -> unit i`, `value n -> word n`, then first/last/full-word or
  character extraction, with a small explicit set of global offsets and
  tokenization conventions. Emit every coordinate, selected source span,
  extracted value, failure, and searched degree of freedom.
- **N5 - predictive verification**: prefer mechanisms learned on one line or
  prefix and applied unchanged to a held-out line or suffix. Measure exception
  count, source-edition robustness, parameter sensitivity, language quality,
  and null/multiple-search baselines. A readable fit with many local offsets is
  a fitted replay, not a solved mechanism.
- **N6 - generated evidence**: add synthetic analogs with structured reference
  documents, distractor documents, multiple languages, OCR/tokenization damage,
  coincidental cardinality matches, and negative controls. Keep rule parameters
  and plaintext in the grading layer only. Public historical examples become
  compatibility tests, not agentic-generalization evidence once their solutions
  are online.

Historical acceptance should include the Cyphral Distich as a newly public
compatibility record only after independent source/transcription and rights
review. The larger Cyphral Octastich should remain a provisional/partial record
until its edition-dependent offsets and unreadable positions are independently
checked. The primary benchmark evidence must be held-out synthetic analogs that
cannot be solved from model memory.

The completed source-aware slice must demonstrate that a simple rule decodes a
held-out region without new tuning, that plausible distractor documents do not,
and that reports distinguish exact coordinates from exceptions. Only after this
lane is measured should Decipher attempt broad Gutenberg-scale or archival
corpus search.

### 2. Remaining historical frontier

- Open-corpus numeric book-cipher search over documented candidate collections;
- nomenclator solving and synthetic code-list generation;
- polyphonic substitution;
- syllabaries and large-alphabet historical systems;
- rotor machines;
- abjad, non-Latin-script, and shorthand systems;
- Voynich-adjacent unknown-language and transcription research.

For these targets, diagnosis-first and an explicit engineering frontier are
valuable outcomes even when no solver exists.

## Investigator-state and model sequencing

Family work is only one track. In parallel with the LLM-free enabling lane:

1. Land a thin INV-1 case file: canonical evidence/coverage/experiment state,
   atomic resume, generated research note, and basic human suggestions.
2. Defer richer `watch` and presentation surfaces until the state is exercised
   on real multi-session investigations.
3. Run the model playbook ablation only after the expanded local diagnosis
   benchmark exists. Compare raw ciphertext, DiagnosisReport, static playbook,
   and report+playbook as separate arms.
4. Use LLMs primarily for experiment selection, adjudication, and testable
   custom mechanisms, not as replacements for compiled family diagnosis.
5. Compare with CipherLens only on an aligned family set and fresh shared test
   distribution. Its published aggregate is a reference point, not directly
   comparable to INV's current six-case suite.

## Definition of a completed family slice

A family or variant lands only when:

1. Its hierarchy level and confusable set are explicit.
2. A generator counterpart exists, or the record explains why generation is
   not meaningful.
3. Its detector/discriminator/probe is calibrated on fresh held-out cases.
4. Reports state uncertainty and counterevidence honestly.
5. Solver status is measured, with a referral when unsupported.
6. Composite behavior and language dependence are tested where applicable.
7. Ground truth is used only after diagnosis/search for evaluation.

Priority follows this document unless a target cipher exposes a more valuable
missing instrument.
