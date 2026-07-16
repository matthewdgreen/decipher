# INV Cipher-Family Roadmap (long-range support list)

Requested by Matthew (2026-07-15): "a list of cipher families that we have
not yet addressed, so that we can eventually add them to INV. Start with
families that other tools support and we don't, then the list can get
longer." Companion to the "Benchmark generation on demand" plan item
(docs/improvement_program_plan.md): full support for a family means all
three columns — **Diagnose** (INV registry entry + discriminators),
**Generate** (benchmark generator counterpart), **Solve** (an automated
attack). A family can usefully land diagnosis-first (INV names it and
recommends the right external tool even before we can break it).

Reference tools for "who else supports it": AZdecrypt (the community
standard for Zodiac-class work), CrypTool 2, the ACA cipher-type list
(~60 types), ciphey/Ciphey (encodings + simple classical).

## Current coverage (2026-07-15)

| family | diagnose | generate | solve |
|---|---|---|---|
| monoalphabetic substitution (incl. Caesar/Atbash as degenerate keys) | ✅ | ✅ testgen | ✅ hill_climb/anneal |
| homophonic substitution | ✅ | ✅ testgen | ✅ zenith_native (99.3% Z408) |
| periodic polyalphabetic (Vigenère; Quagmire III keyword) | ✅ | ✅ (suite builder) | ✅ periodic + quagmire3 searches |
| transposition (columnar/pure) | ✅ | ✅ (transform pipeline) | ✅ pure_transposition + transform searches |
| substitution+transposition composite | ✅ (transform suspicion) | ✅ (suite builder) | ✅ transform_candidates |
| transposition+homophonic composite | ✅ registry | ✅ ladder split | ✅ transform_homophonic |
| numeric book cipher (Beale-class) | ✅ P8 battery | ➖ | ❌ (diagnosis only) |
| nomenclator | ✅ registry | ❌ | ❌ |
| plaintext/hoax | ✅ | n/a | n/a |
| playfair / polygraphic / fractionation | ⚠️ registry entries exist but discriminators are `planned` (INV-0 cover set) | ❌ | ❌ |

## Tier 1 — other tools support these; we do not (add first)

Ordered roughly by (a) how commonly other tools crack them, (b) fit with
machinery we already have.

1. **Playfair** (digraphic) — AZdecrypt, CrypTool, ACA staple. Registry
   entry exists; needs a real discriminator (even-length digraph stats, no
   doubled digraph letters), a generator, and an SA solver.
2. **Bifid / Trifid** (Polybius fractionation) — AZdecrypt, ACA. Pairs
   naturally with the existing fractionation registry entry.
3. **ADFGX / ADFGVX** (fractionation + columnar) — CrypTool; historically
   major (WWI). Composite of two things we partly have.
4. **Two-square / Four-square** (digraphic) — AZdecrypt, ACA.
5. **Autokey (Vigenère autokey)** — CrypTool, ACA. Small extension of our
   periodic machinery; distinct diagnosis signature (no Kasiski repeats).
6. **Beaufort / Variant / Porta / Gronsfeld** (Vigenère relatives) — nearly
   free on the solve side given our periodic stack; diagnosis merges with
   polyalphabetic_periodic; generators trivial.
7. **Running key** — AZdecrypt. Diagnosis: polyalphabetic with no period.
8. **Transposition variants: Railfence/Redefence, Route, Amsco,
   Myszkowski, Cadenus** — ACA staples, several in AZdecrypt/CrypTool. Our
   generic transposition search may already crack some; needs per-variant
   generators + verification, then targeted solvers where generic fails.
9. **Hill cipher (2x2/3x3 linear)** — CrypTool. Distinct algebraic
   diagnosis signature.
10. **Nihilist substitution / transposition** — ACA; numeric-pair
    signature overlaps our P8 numeric battery (good discriminator fit).
11. **Straddling checkerboard / VIC-style** — mixed-length numeric;
    extends the numeric battery.
12. **Fractionated Morse / Morbit / Pollux** — ACA; needs a morse layer.
13. **Grille / turning grille** — CrypTool; transposition family.
14. **Encodings tier** (ciphey's home turf): Base64/32/85, hex, binary,
    Morse, Baconian, A1Z26, ROT47, tap code. Not cryptanalysis — but INV
    diagnosis (P0) should DETECT and name them so an "unknown input"
    triages correctly; generators are trivial; needed for any fair
    external-tool comparison (see Ciphey note in the improvement plan).

## Tier 2 — historically important; weak or no support anywhere (longer range)

- **Numeric book-cipher SOLVING** (key-text search over corpora) — we
  diagnose (Beale); nobody solves well. High-prestige target.
- **Nomenclator solving** (mixed code+cipher, historical archives) — the
  Borg/Copiale lineage generalized; generator = code-list synthesis.
- **Polyphonic substitution** (one symbol → several letters) — rare in
  tools; appears in historical material.
- **Syllabary / large-alphabet historical systems** — extends multisym.
- **Rotor machines (Enigma-class)** — CrypTool has it; long-range and a
  different compute profile.
- **Running-key with non-book keys, progressive-key (Gromark family)** —
  ACA tail.
- **Abjad / non-Latin-script and shorthand systems** (Voynich-adjacent) —
  research frontier; diagnosis-first.

## How this list gets consumed

Adding a family = one slice: registry entry + discriminator (with
shuffle-null calibration per INV-0 conventions) + generator (per the
on-demand benchmark plan item) + solver-or-referral note. Diagnosis-first
is acceptable; generators are required with the entry (they are how the
discriminator gets calibrated). Priority follows this document's order
unless a target cipher demands otherwise.
