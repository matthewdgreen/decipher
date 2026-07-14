# Beale reference resources (INV-0 Part 8)

Provenance and integrity for every file in `resources/reference/`. All files are
public-domain historical material. **No accepted plaintext for Beale 1 or Beale 3
exists in this repository or the benchmark** — none is bundled here.

## beale2_numbers.txt
- Content: the Beale cipher #2 numeric stream (the SOLVED companion), whitespace-
  separated decimal tokens, wrapped 20 per line.
- Source: `other_tools/azdecrypt-src/AZdecrypt/Ciphers/Substitution/Beale 2.txt`
  (local public-domain AZdecrypt distribution), normalized to whitespace-separated
  tokens.
- Token count: 762 (commonly cited as 763; this transcription variant
  ships 762 tokens).
- Max value: 1005   Min value: 1
- sha256 (file): `26e5e5928007e0d5ec027fbc626a84415af130d1bb96bb37e690e5975696bd85`

## beale_doi_key_words.txt  —  SOLUTION-BEARING KEY TEXT
- Content: the U.S. Declaration of Independence as a one-word-per-line KEY for the
  Beale word-first-letter book cipher (1-based word numbering).
- Source edition: standard public-domain engrossed Declaration of Independence
  (U.S. National Archives transcription); punctuation stripped, apostrophes and
  hyphens removed, uppercased, reading order preserved.
- Word count (body, excluding banner): 1165
- sha256 (file, incl. banner): `e2d64aaca60f944983fd758f536ff326a15d41ba9c98b8f6c4038588ae6cdc2f`
- **This is a decryption KEY, not diagnostic input.** It carries the marker
  `SOLUTION-BEARING KEY TEXT`; `analysis.numeric_code.assert_not_solution_bearing`
  refuses to load it as ciphertext, and the CLI `diagnose` path enforces this.
- Out-of-range rule: a Beale index past word 1165 is SKIPPED and its
  position recorded (Beale 1 max 2906 exceeds the DoI word count).
- **Recorded deviation (B2 self-check):** decoding Beale 2 via this standard
  public-domain DoI numbering yields `IHAREDEPOSCTED...` (12/14 of the opening;
  477/762 letters overall), NOT a clean `IHAVEDEPOSITED`. The exact 1885
  Ward-pamphlet numbering quirks needed for a full offline self-check could not be
  reproduced or individually cited from memory. Per the INV-0 spec, the Gillogly
  B2 self-check is therefore **reported, not asserted as passed**.
- **Gillogly headline provenance:** the B1 nondecreasing-run statistic (14-letter
  run `DEFGHIIJKLMMNO`, upper-tail p = 1/1001) reproduces **under the standard
  public-domain DoI numbering**; the exact-1885-Ward-pamphlet form of the claim is
  **provisional** pending a pamphlet-quirk-corrected key (B2 self-check 12/14).
  Robustness note: the run statistic tolerates the 2-of-14 key-word error rate the
  self-check exposes — isolated first-letter substitutions rarely break a
  14-letter monotone run. `scripts/research/beale_report.py` prints the statistic
  with this provenance attached.

## INV-0 deviations record (registry)
- **Planned cover discriminators:** four `planned` discriminators
  (`disc_transp_fractionation`, `disc_playfair_polygraphic`,
  `disc_nomenclator_book`, `disc_unknown_nomenclator`) were added beyond the
  spec's 7-entry inventory to satisfy `_validate_registry` (the 7 available
  entries leave 5 primaries with no discriminator — a spec-internal contradiction
  resolved via the spec's own `planned` status). Verified they cannot flip any
  verdict: `planned` maps to `unavailable` (never `pending`) for rule (c), and
  none of their split pairs is in `confusable_with` for rule (d).
