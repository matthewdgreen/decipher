# AZdecrypt Example Corpus Catalog

This is a working catalog of cipher examples bundled with the local AZdecrypt
checkout at `other_tools/azdecrypt-src/AZdecrypt/Ciphers`. It is intended to
answer two practical questions:

- Which bundled examples do we already cover in `cipher_benchmark` or local
  Decipher fixtures?
- Which missing examples would make good Decipher tests, especially unsolved or
  mixed-technique cases?

The AZdecrypt checkout itself is under `other_tools/` and is not redistributed
with Decipher. Before importing any named example into `cipher_benchmark`,
verify provenance, rights, and current solution status from primary or
reputable public sources. The label `Unsolved` below means "file appears in
AZdecrypt's `Ciphers/Unsolved` folder", not an independently revalidated
claim.

## Current Benchmark Coverage

Decipher's main benchmark currently emphasizes:

| Area | Current state |
|---|---|
| Borg | 397 benchmark records, Latin simple/nomenclator-like historical material. |
| Copiale | 101 benchmark records, German homophonic/nomenclator historical material. |
| DECODE/Gallica | 155 benchmark records. |
| Synthetic simple substitution | 300 records across English, German, French, Italian, with and without word boundaries. |
| Zodiac | Z408 and Z340 are present with global Zodiac glyph IDs. Z340 transform replay/search fixtures also exist in Decipher. |
| Kryptos | K1, K2, and K3 are present as solved calibration records. K4 is present in the unsolved benchmark area. |
| Voynich | ZL3b transliteration records are present in the unsolved benchmark area. |
| Scorpion | S1/S5 tentative transcriptions and a hypothetical combined S1+S5 synthetic case are present in the unsolved benchmark area. |
| Neutral synthetic probes | At least one Quagmire III neutral probe is present for agentic unknown-cipher testing. |

## AZdecrypt Top-Level Families

| AZdecrypt folder | Approximate contents | Decipher coverage | Import value |
|---|---|---|---|
| `Substitution` | Named monoalphabetic examples, ACA-like exercises, Beale 2, Gold-Bug, quotes, short challenges. | Mostly covered by Decipher's solvers and synthetic tests, but not by these exact examples. | Medium for regression breadth; low urgency unless provenance is useful. |
| `Substitution + defects` | Substitution examples with deliberate errors, omissions, or noisy text. | Partly planned under noisy-transcription/null support; not well covered as concrete fixtures. | High, because noisy historical text is central to Borg/Copiale-like work. |
| `Substitution + nomenclator` | Numeric substitution plus codeword/nomenclator elements. | Conceptually overlaps Borg/Copiale, but exact synthetic/numeric examples are missing. | High, especially for testing codebook/nomenclator tools. |
| `Bigram substitution` | Klaus Schmeh bigram challenges, Navy test problems, smokie example. | Largely unsupported. | High long-term; this is a real gap for digraphic/polygraphic substitution. |
| `Higher-order homophonic` | Second-order homophonic challenge material. | Unsupported or diagnostic-only. | High long-term; would stress homophonic assumptions beyond current Zenith-style solver. |
| `Homophonic substitution cycle types` | Many Zodiac-like cycle-structure variants. | Not directly covered; Decipher has Zodiac and synthetic homophonic cases but not cycle-family tests. | Medium/high for improving homophonic diagnostics and cycle-aware search. |
| `Substitution + transposition` | Z340-like mixed substitution/transposition challenges and many synthetic route/period variants. | Partly covered by the transform ladder and stress suite, but exact examples are mostly missing. | High for transform breadth regression, once provenance is acceptable. |
| `Transposition` | Pure transposition examples including Feynman 1, DCT, K3, columnar/route challenges. | K3 covered; many others missing. | High for the pure-transposition screen. |
| `Substitution + polyalphabetic` | Mixed substitution/polyalphabetic/wildcard/interlaced examples. | Mostly unsupported or only partially covered by Quagmire/Vigenere work. | High for the unknown-cipher router and future cipher-mode work. |
| `Substitution + vigenère` / `Substitution + trithemius` | Hybrid substitution with periodic/progressive polyalphabetic layers. | Mostly unsupported. | Medium/high after polyalphabetic mode separation matures. |
| `Vigenère` | Kryptos K1/K2. | Covered in benchmark. | Low; already imported. |
| `Vowel substitution` | Specialized vowel-only substitution example. | Unsupported/specialized. | Low/medium; useful as identifier oddity. |
| `Various` | Fractionated Morse, Alberti, Langrenus, D'Agapeyeff Polybius transforms, periodic nulls. | Mostly unsupported except adjacent TODOs. | High as a roadmap sampler; import selectively. |
| `Zodiac ciphers` | Zodiac 13/32/408/340 and Fairfield/Fake Zodiac variants. | Z408/Z340 covered; Z13/Z32 and minor variants are only partially covered in unsolved area/docs. | High for qualitative Zodiac-family tests. |
| `Unsolved` | Famous and challenge unsolved items. | Voynich, Scorpion, K4 present elsewhere; most others missing. | High, but provenance/status must be checked. |
| `Plaintext` and `Batch` | Support/generated plaintexts and batch files. | Not benchmark targets. | Low, except for generating synthetic stress packets. |

## Named Examples Already Covered Or Mostly Covered

| AZdecrypt example | Current Decipher status | Notes |
|---|---|---|
| `Zodiac ciphers/Zodiac 408*.txt` | Covered in `cipher_benchmark` as `zodiac408_zenith_global` and parity splits. | Good homophonic regression anchor. |
| `Zodiac ciphers/Zodiac 340*.txt` and `Substitution + transposition/Zodiac 340.txt` | Covered in benchmark/fixtures with transform replay/search. | Still useful for apples-to-apples import checks if glyph mapping differs. |
| `Vigenère/Kryptos 1.txt` | Covered as `kryptos_k1`. | Solved calibration; famous/memorization caveat for agents. |
| `Vigenère/Kryptos 2.txt` | Covered as `kryptos_k2`. | Quagmire III/keyed-Vigenere benchmark anchor. |
| `Transposition/Kryptos 3.txt` | Covered as `kryptos_k3`. | Pure-transposition calibration. |
| `Unsolved/Kryptos 4*.txt` | Present in unsolved benchmark area according to benchmark docs. | Verify split/record names before running. |
| `Unsolved/Scorpio 5.txt` | Adjacent to current Scorpion S5 import, but naming/transcription should be cross-checked. | AZdecrypt calls it "Scorpio"; our benchmark uses Scorpion. |
| `Unsolved/Dorabella.txt` | Imported in unsolved benchmark as `dorabella_cipher`. | Uses AZdecrypt letter-surrogate transcription; not a final glyph ontology. |
| `Substitution/The Gold Bug.txt` | Covered indirectly through Zenith/tool parity fixtures. | Exact AZdecrypt file not necessarily imported. |

## Missing Unsolved / Disputed Examples Worth Considering

These are the best candidates to curate first because they are either famous,
large enough to analyze, or directly exercise solver gaps.

| Priority | AZdecrypt file | Size signal | Why it is interesting for Decipher | Current capability fit |
|---|---|---:|---|---|
| 1 | `Unsolved/D'Agapeyeff.txt` | 392 nonspace chars | Classic unresolved numeric cipher; AZdecrypt also ships Polybius-derived variants in `Various`. Good for fractionation/Polybius TODOs. | Imported in `../cipher_benchmark` as `dagapeyeff_cipher`; diagnostic only. |
| 1 | `Unsolved/Beale 1.txt` and `Unsolved/Beale 3.txt` | 1336 / 1505 nonspace tokens | Famous book/numeric cipher material; good for codebook/nomenclator/unknown-cipher diagnosis. | Not currently solved; import as qualitative/unsolved. |
| 1 | `Unsolved/Feynman 2.txt` and `Unsolved/Feynman 3.txt` | 261 / 231 chars | Compact famous alphabetic challenges with strong claimed solutions. Good for unknown-cipher hypothesis discipline and calibration caveats. | Imported in `../cipher_benchmark` main benchmark as `solved_probable`; claimed plaintext and method metadata are stored, but solution-bearing details stay out of blind/standard context. |
| 1 | `Unsolved/DCT Reloaded 3.txt` | 1100 chars | German double-columnar/transposition-style challenge with explicit keyword length hints. | Imported in `../cipher_benchmark` as `dct_reloaded_3`; good target for pure-transposition breadth and agent context use. |
| 1 | `Unsolved/Ricky McCormick page 1/2.txt` | 406 / 425 nonspace chars | Real-world ambiguous notation, likely not a standard substitution. Good for unknown-language/noisy-symbol diagnostics. | Imported in `../cipher_benchmark` as `ricky_mccormick_note_1/2`; qualitative stress tests. |
| 1 | `Unsolved/Blitz cipher page 7/8.txt` | 470 / 159 chars | Historical symbolic manuscript material; good OCR/transcription/unknown-family target. | Imported in `../cipher_benchmark` as `blitz_cipher_p7/p8`; rights/transcription certainty remain caveats. |
| 2 | `Unsolved/Copenhagen cryptogram.txt` | 101 chars | Short classic challenge; useful as "too short, be honest" diagnostic. | Current agents should likely avoid overclaiming. |
| 2 | `Unsolved/Helen Fouché Gaines.txt` | 125 chars | Short challenge with cryptanalytic-history flavor. | Import after provenance check. |
| 2 | `Unsolved/IKLP long.txt` / `IKLP short.txt` | 358 / 19 chars | Good for testing extremely short/long contrast and declaration restraint. | Diagnostics only. |
| 2 | `Unsolved/Lawrence Public Library Cryptogram part 1/2.txt` | 451 / 395 chars | Two related pages could test associated-record context tools. | Imported in `../cipher_benchmark` as `lawrence_public_library_cryptogram_p1/p2`; good agentic context exercise. |
| 2 | `Unsolved/Moustier St Martin.txt` and `Moustier Virgin.txt` | 76 / 80 chars | Short symbolic/historical examples. | Needs provenance; likely diagnostic only. |
| 2 | `Unsolved/Nick Pelling challenge 2-7.txt` | 222-354 chars | Challenge set from the historical-cipher community; multiple related items. | Imported in `../cipher_benchmark` as hold-for-review placeholders; exact source pages still need verification. |
| 2 | `Unsolved/Paul Rubin.txt` | 459 chars | Medium mixed-stream challenge. | Imported in `../cipher_benchmark` as `paul_rubin_cipher`; useful early unknown-cipher run. |
| 2 | `Unsolved/Powers cryptogram.txt` | 96 chars | Short real/disputed challenge. | Diagnostics only. |
| 2 | `Unsolved/Taman Shud.txt` | 44 chars | Famous but extremely short; likely not solvable by normal ciphers. | Imported in `../cipher_benchmark` as `taman_shud_code`; useful for "no overclaiming" tests. |
| 3 | `Unsolved/1916 train station robbery cryptogram.txt` | 94 chars | Short historical case. | Import only if provenance is easy. |
| 3 | `Unsolved/GUN WA 1889.txt` | 77 chars | Short challenge. | Low solving leverage. |
| 3 | `Unsolved/Allen Benjy 2010 challenge.txt` | 708 chars | Longer challenge with unknown provenance/status. | Potentially useful after source check. |
| 3 | `Unsolved/Glurk (Beale 3 emulation challenge).txt` | 1591 chars | Synthetic/emulation challenge; likely good for codebook/book-cipher tooling. | Imported in `../cipher_benchmark` as `glurk_beale3_emulation`; explicitly marked synthetic/placeholder. |

## Missing Solved / Calibration Examples Worth Importing

These are not as glamorous as unsolved ciphers, but they would be extremely
useful for honest capability testing.

| Priority | AZdecrypt area | Examples | Why import |
|---|---|---|---|
| 1 | Bigram substitution | Klaus Schmeh Bigram 600/750/1000/1346, Navy test problems | Gives us concrete digraphic/polygraphic substitution tests, a known gap. |
| 1 | Substitution + nomenclator | CrypTool example, `daikon 5/6` | Directly supports the long-range nomenclator/codebook roadmap and Copiale-like codeword handling. |
| 1 | Transposition | Feynman 1, Klaus DCT 456/480/599, Norbert Roche | Broadens pure-transposition regression beyond K3 and synthetic ladders. |
| 1 | Substitution + transposition | Jarlve/Largo/smokie challenge families | Gives a non-Zodiac mixed-transform corpus to test whether the transform solver generalizes. |
| 2 | Substitution + defects | W.B. Tyler 2, doranchak multiobjective set | Good for nulls, transcription errors, and noisy repair workflows. |
| 2 | Higher-order homophonic | Jarlve second-order set, doranchak 8-letter plaintexts | Forces us to distinguish first-order homophonic success from higher-order variants. |
| 2 | Homophonic cycle types | perfect/anti/random/palindromic/top-bottom cycles | Useful for cycle-structure diagnostics; less important as plaintext benchmarks. |
| 2 | Various | Langrenus, Fractionated Morse, Alberti, periodic nulls | Good "future cipher family sampler" once each family has at least diagnostic tooling. |
| 3 | Vowel substitution | `VV-1.txt` | Small specialized identifier test. |

## Suggested Import Order

1. **Curate missing famous unsolved records**: K4 split verification and any
   remaining short famous records worth no-overclaim testing. D'Agapeyeff,
   Beale 1/3, Dorabella, Taman Shud, Ricky McCormick pages, DCT Reloaded 3,
   Lawrence Public Library, Blitz pages 7/8, Nick Pelling 2-7, Paul Rubin, and
   Glurk are already seeded in the unsolved area; Feynman 2/3 are seeded as
   main-benchmark `solved_probable` calibration records.
   Mark all unsolved rows as qualitative/no-ground-truth unless a reliable
   accepted plaintext exists.
2. **Curate pure-transposition calibrations**: Feynman 1, Klaus DCT challenges,
   Norbert Roche, Geoffrey Rochefort. These are the cleanest next tests for
   the expanded transposition screen.
3. **Curate mixed substitution+transposition calibrations**: a small
   representative set from Jarlve/Largo/smokie plus Z340. Avoid importing
   dozens before the first 5-10 are running cleanly.
4. **Curate bigram/polygraphic substitution tests**: Klaus Schmeh bigram set and
   Navy examples, then add diagnostics/solvers.
5. **Curate nomenclator/codebook examples**: CrypTool and daikon examples, then
   compare with Copiale/Borg codeword behavior.
6. **Use AZdecrypt synthetic families as generator inspiration** rather than
   importing every file. For long-term robustness, Decipher should generate
   fresh analogs with recorded seeds, not overfit to AZdecrypt's bundled set.

## Immediate Interesting Runs

Without new import work, the best current qualitative runs are:

- Existing unsolved benchmark area: Voynich folios, Scorpion S1/S5, K4/Zodiac
  unsolved splits if present.
- Existing fixtures/benchmark: K3 and Z340 for transform behavior.

After light curation, the highest-value AZdecrypt-derived targets would be:

- D'Agapeyeff: fractionation/Polybius and unknown-cipher diagnosis. Imported
  as `dagapeyeff_cipher` in the unsolved benchmark area.
- DCT Reloaded 3: German transposition with explicit keyword-length context.
  Imported as `dct_reloaded_3` in the unsolved benchmark area.
- Feynman 2/3: compact alphabetic solved-probable tests for agent hypothesis
  discipline and context-control calibration. Imported as `feynman_2` and
  `feynman_3` in the main benchmark.
- Beale 1/3: book/numeric/codebook hypothesis workflow.
- Ricky McCormick pages: unknown notation/noisy real-world text, likely
  qualitative only. Imported as `ricky_mccormick_note_1/2` in the unsolved
  benchmark area.
- Lawrence Public Library cryptogram: related-page unknown/book/substitution
  candidate. Imported as `lawrence_public_library_cryptogram_p1/p2`.
- Blitz Cipher pages 7/8: symbolic unknown-family pages with provisional
  transcription caveats. Imported as `blitz_cipher_p7/p8`.
- Dorabella and Taman Shud: imported as famous short/no-overclaim diagnostics.
- Nick Pelling challenges 2-7: imported as related numeric challenge
  placeholders.
- Paul Rubin: imported as a mixed-stream unknown-cipher diagnostic.
- Glurk: imported as a synthetic Beale-emulation/codebook-style placeholder.
