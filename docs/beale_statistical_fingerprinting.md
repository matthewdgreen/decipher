# Beale Statistical Fingerprinting Plan

Status: planning note. This is not a solve plan. Beale 1 and Beale 3 should be
used as unsolved numeric-cipher diagnostics for Decipher's unknown-cipher
fingerprinting work.

## Why Beale Belongs Here

Beale is useful because it is famous, numeric, probably codebook/book-cipher
shaped, and still unsolved. A good tool should be able to say useful things
about it even when it cannot recover plaintext.

The benchmark now has unsolved records for:

- `beale_1`
- `beale_3`

These records live in the sibling benchmark repo under
`benchmark/unsolved/sources/famous_short/` and should be treated as
diagnosis-only unless a candidate can be independently validated. There is no
ground truth for solver routing, candidate ranking, or agent prompting.

## Prior Statistical Work To Reproduce

Decipher should eventually reproduce, or at least closely approximate, the
main statistical evidence reported in:

- Viktor Wase, "Benford's law in the Beale ciphers", Cryptologia 45(3),
  2021. DOI: `10.1080/01611194.2020.1821409`.
- Leonardo Campanelli, "A statistical cryptanalysis of the Beale ciphers",
  Cryptologia 47(5), 2023. DOI: `10.1080/01611194.2022.2116614`.
- Jim Gillogly, "The Beale Cipher: a dissenting opinion", Cryptologia 4(2),
  1980. This is important for the non-random DOI-derived alphabet-like
  sequence in Beale 1.

Useful public context:

- Cipher Foundation Beale overview:
  <https://cipherfoundation.org/older-ciphers/beale-papers/>
- Cipher Foundation transcription:
  <https://cipherfoundation.org/older-ciphers/beale-papers/beale-papers-transcription/>
- Simon Ayrinhac's Beale search writeup, which describes systematic internet
  and Project Gutenberg key-text trials:
  <https://simon.ayrinhac.free.fr/chiffre_de_beale.html>

## Diagnostic Features

Add numeric-code/book-cipher panels to the unknown-cipher fingerprint report:

- Numeric token count, unique count, max token, min token, repeated-token rate.
- First-digit and last-digit distributions.
- Benford and epsilon-Benford deviation scores.
- Comparison against known Beale 2 / Declaration-of-Independence book-cipher
  profile when explicitly available as a public related reference.
- Repeated n-gram counts over numeric tokens.
- Consecutive numeric runs and monotone runs.
- Gap and modulo structure.
- Front-loading: how much of the cipher references the early portion of a
  hypothetical key text.
- Required key-text length under word-position and character-position
  hypotheses.
- Plausibility flags for:
  - word-first-letter book cipher;
  - character-position book cipher;
  - skip/every-Nth-word variants;
  - shared-key relationship between Beale 1 and Beale 3;
  - fake/random-number or hoax-like generation.

## Corpus Search Is A Separate Layer

A brute-force key-text scan is valuable, but it should sit behind the
diagnostic layer:

1. Diagnose whether the numeric stream resembles a book/code cipher.
2. Estimate what kind of key text would be required.
3. Run bounded public-corpus trials only when the diagnostic evidence justifies
   them.
4. Report negative evidence honestly, including edition/tokenization
   dependence.

Potential corpus-search variants:

- Gutenberg/public-domain English texts.
- Founding-era legal and political texts.
- Virginia/Masonic/geographic texts suggested by historical context.
- Word-position first-letter mapping.
- Word-position last-letter or nth-letter mapping.
- Character-position mapping.
- Offset and tokenization variants.
- Every-Nth-word or parity-based numbering.

The artifact should record the corpus snapshot, source licenses, tokenization
rules, offsets tried, and scoring thresholds. Do not conflate "no match found"
with proof that no book cipher exists.

## Acceptance Criteria

- `observe_cipher_id` or a successor tool can identify Beale 1/3 as
  numeric-code/book-cipher-like rather than trying ordinary substitution first.
- A standalone report script can produce Beale 1/2/3 comparative statistics
  without using Beale 1/3 plaintext.
- The report can reproduce the broad conclusion from Wase/Campanelli: Beale 1
  and 3 do not look statistically like Beale 2 under the standard Beale 2
  method, unless a different method or keying convention is assumed.
- Agentic runs can declare a structural/diagnostic result without pretending
  to have a plaintext decipherment.
