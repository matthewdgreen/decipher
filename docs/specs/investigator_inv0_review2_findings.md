# INV-0 Spec — SECOND Review Findings (VERDICT: NEEDS REWORK)

Reviewer: Fable (verified), method = EXECUTABLE re-derivation (implemented
the Part-6 catalog+formula literally, ran it against the real
`src/analysis/cipher_id.py` fingerprint priors and the real
`beale_1`/`beale_3` streams; Monte-Carloed the Part-5 span rule on
shuffled prose). The rework's STRUCTURE is genuinely good and confirmed
sound (scoring formula executable, discriminator predicate decidable and
non-vacuous, shuffle stats/encode_tokens/tail correct, hoax split
correct, DoI resource sound, Part-0 data all verified, scope clean). What
fails is the CALIBRATION beneath it — 5 of 9 fixtures' hand-computed
numbers are wrong. **The mandated fix method: recalibrate by EXECUTABLE
SIMULATION against cipher_id.py + real Beale data, not on paper.** The
reviewer's scripts took minutes.

## P0 — provably unpassable as written

**1. Numeric fixtures + acceptance-2 broken: homophonic ties/beats
numeric_book on real Beale data.** Dense numeric streams fire
large_symbol_inventory (0.45) + flat_high_entropy (0.30) + depressed_ic
(0.25) + homophonic prior (0.20) = 1.20, vs only numeric_token_stream's
−0.35. VERIFIED: beale_1 → homophonic 0.850 == numeric_book 0.850, and
the spec's own (−score, family_id) tie-break ranks homophonic first
(alphabetical). Same for beale_3. Acceptance 2 ("numeric_book ranks above
ALL substitution primaries") is unpassable; fixture E and near-miss (iv)
also fail. FIX: turn the existing "unique-count/max-value inconsistency"
counterevidence note into a SCORED weakening atom (condition
alphabet_class=="numeric" ∧ unique/token > 0.4, weight ≥ ~0.5, weakens
ALL substitution primaries — and add transposition_homophonic/
fractionation/polygraphic to numeric_token_stream's weakens list). E's
index sampling must be early-biased or numeric_book = 0.65 < 0.70 →
`strong` unreachable (acceptance 5 fails).

**2. Finding-3 regressed: Part-5 span rule STILL reads shuffled prose as
coherent.** The adjacent_ok OR-rescue ("either word is a function word")
is fatal at English function-word density (~47%). Monte-Carlo: shuffled
prose → longest spans 14–33; 40/40 shuffles ≥ 5; dict_rate ~1.0 and
function_word_rate ~0.47 are shuffle-invariant → all four `coherent`
conjuncts pass → Part-9 "shuffled prose → != coherent" unpassable. FIX:
make the junction predicate attested-bigram-driven (rescue only when the
function-word PAIR is attested; or require ≥k attested junctions per
qualifying span; or use word_bigram_loglik vs the text's own shuffle null
via Part-3 infra). THEN re-verify the Part-8 B2 self-check stays coherent
under the stricter rule (hard gate — currently passes only because the
rule is too weak).

**3. Shape-atom thresholds miscalibrated for real English → fixture A's
`strong` and (iii)'s top-2 fail.** With cipher_id's actual
_normalized_entropy (normalizes by log2 of OBSERVED unique symbols),
English letter streams measure 0.8975 (300 tok) / 0.8982 (400) / 0.9037
(120). So peaked_monogram_shape (<0.85) NEVER fires on letters, and
flat_high_entropy (>0.90) coin-flips (fired spuriously on the 120-tok
transposition → (iii) real top-2 = homophonic 0.37/transposition 0.32,
not tabulated). Fixture A → mono 0.45 vs transposition 0.30, margin
exactly 0.150 knife-edge AND `strong` (≥0.70) unreachable → acceptance 5
fails for A. Fixture (ii) in the same 0.85–0.90 dead zone. FIX:
recalibrate both thresholds against the plug-in estimator at fixture
sample sizes, OR switch peakedness to per-symbol flatness_chi2; re-run all
nine rows WITH A SCRIPT. (Fixture D is genuinely solid — verified.)

## P1

**4. confusable_with never enumerated but load-bearing.** Strong-margin
and verdict rule (d) quantify over it; rule (d) is vacuously true when no
DiscriminatorSpec splits {top1,top2}. Fixture B passes ONLY if homophonic
∉ poly.confusable_with (else (d) fires — no poly↔homo discriminator —
margin 0.18<0.25 kills strong); fixture C likewise. FIX: pin confusable
sets per family; re-check B/C.

**5. (iii) trigger mislabeled; Part-6 "all-pending" narrative wrong.**
With the enumerated registry, mono in (iii) has 3 discriminators, only
order_layout not_computable → rule (c) ("every one pending") does NOT
fire; (iii) goes uncertain via (b)/(d). Rule (c) is near-dead code. FIX:
reword the trigger column; pin the (iii) test to observable outcomes
(uncertain + recommended_next[0]==disc_mono_transp), not mechanism (c).
Define a fallback when the actual top-2 has no covering discriminator.

**6. Baselined-atom emission rule unspecified.** Reliability rule implies
a baselined atom is emitted even when insignificant (p>0.05→low), so
every fixture's poly score silently gains 0.45·0.4=0.18 from a noise
periodic_ic_recovery atom (breaks C/B/(ii)). FIX: baselined atoms emit
ONLY when a stated structural trigger passes; the baseline then grades
reliability.

## P2

**7. Subtype scoring vacuous** — no catalog atom supports numeric_word_
position/char_position/skip_nth_word → all-zero sibling margins (E passes
only because it asserts presence). Map the P8 per-hypothesis flags to
subtype atoms.

**8. Minor definitional gaps:** Benford-inconsistency threshold for
independent_random_like cond-2 unset; digit-preference χ² must say
LAST-digit (first-digit χ² is significant for uniform-on-[1,3000]);
book_keylength_plausible bound unset; p-value formula needs
parenthesization "(#{draws≥observed}+1)/(n+1)"; encode_tokens note the
nonnegative assumption (to_bytes raises on negatives).

## Rework method (MANDATORY — this is why two paper passes failed)

Implement the Part-6 scoring + Part-5 span rule as a calibration script
(throwaway or kept as a spec-verification test), apply the P0/P1 fixes as
PRINCIPLED atoms/thresholds (not fixture-overfitting), and iterate the
weights/thresholds against cipher_id.py + real beale_1/beale_3 + generated
letter/homophonic/transposition/Vigenere streams until ALL nine fixtures
AND acceptance 2/5 pass. Write the DERIVED numbers into Part 5/6/9 +
acceptance + a re-derived Revision-notes table, and KEEP the calibration
script so the numbers are reproducible and future edits re-verify. Third
Fable review will re-execute.
