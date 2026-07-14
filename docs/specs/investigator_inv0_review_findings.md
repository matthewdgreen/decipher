# INV-0 Spec Review Findings (VERDICT: NEEDS REWORK)

Reviewer: Fable (verified). Integrates the project owner's (Matthew,
cryptographer) ten findings — all confirmed — plus six reviewer findings.
The spec's foundation is strong (ciphertext-only firewalled interfaces,
exception-isolated panels, mandatory counterevidence, verified Part-0
data ground truth, pin-test-first Part-7 wiring, honest provenance,
right-reason tests). But findings 1–5 mean **no conforming implementation
can exist yet** — required Part-9/acceptance outcomes are ambiguous or
provably unpassable. Rewrite the listed sections before a coder starts.

## P0 (blocking — a conforming implementation cannot exist)

**1. [owner] Family scoring is not executable.** Part 6 gives starting
points (SUSPICION_TO_FAMILY-mapped suspicion_scores; quagmire = parent ×
0.6; numeric_* from P8 atoms; others "from listed detector atoms or 0")
but never defines atom→score-delta conversion, weakens/counterevidence
reduction, reliability weighting, 0–1 normalization, or tie order — yet
confidence/verdict hinge on exact values (≥0.70, margin ≥0.25, top-two
margin <0.15). Demonstration: cipher_id caps scores["transposition"] at
0.40 while transposed English scores monoalphabetic ≈1.0 → the required
known-family columnar-transposition "confident" fixture is UNPASSABLE
from the suspicion base. Also `_compute_suspicion_scores` returns
{"unknown":1.0} for <10 tokens — a seventh key SUSPICION_TO_FAMILY does
not map. FIX: specify the exact formula (base + Σ per-atom weights for
supports, − for weakens, reliability multipliers, clamp/normalize),
deterministic tie-break (family id), "unknown"-key handling; verify on
paper that each Part-9 fixture's intended top family actually wins.

**2. [owner] Discriminator state inconsistent — worse: no discriminator
inventory exists at all.** DiscriminatorSpec defined; Part-6 verdict
fires `uncertain` when "top family's discriminators all pending" — but
INV-0 has no mechanism marking a discriminator run (coverage_status
hardcoded "untested"; ledger is INV-1), so under the natural all()
reading (vacuously true for empty tuples) EVERY report is uncertain and
the confident fixtures are unpassable. And the spec enumerates no
concrete discriminator (ids/splits/tool bindings) for any family, yet
fixtures require recommended_next[0] to be "the correct discriminator."
FIX: (a) enumerate the discriminator table in Part 1 (min: mono↔transp
via P5 gap, mono↔homophonic via homophone distribution, periodic↔
quagmire, numeric_book↔hoax via P8 flags); (b) define "run" as a
panel-evidence predicate (discriminator atoms present with ok status),
OR defer the all-pending verdict rule to INV-1 explicitly (amend rule
text).

**3. [owner] island_report cannot fail shuffled real-word salad.**
Verdict: coherent iff dict_rate ≥0.75 AND function_word_rate ≥0.25 AND
longest_coherent_span ≥5. Shuffling prose preserves dict_rate (~1.0) and
function_word_rate (~0.4); since every word stays a dictionary word, the
whole text is one run → span = full length → returns `coherent`, but
Part 9 requires `gibberish`. The only order-sensitive signal
(word_bigram_loglik) is explicitly barred from the verdict. This defeats
the module's anti-pareidolia purpose (bad basins ARE real-words-wrong-
order). FIX: make span construction order-sensitive (span extends only
while adjacent word pairs clear a bigram/transition plausibility floor;
closed-class-adjacency fallback where bigram resource absent); admit the
bigram term to the verdict when available OR specify the la/de fallback;
re-derive the shuffled-fixture expectation.

**4. [owner] Panel signature loses info P5 and numeric counterevidence
need.** Panels get (tokens, *, alphabet_size, language, …). P5 requires
letter n-gram scoring — impossible from dense integer tokens (nothing
says token 3 renders as D vs an S-token vs a Borg glyph; nothing
distinguishes letters/digits/glyphs; the design's alphabet-class item was
dropped from the INV-0 table). As written P5 can only return
not_computable → kills the P5 discriminator that near-miss fixture (iii)
requires. FIX: add explicit input — alphabet_class ∈ {letters, numeric,
symbols, mixed} + optional letter_rendering: str|None (normalized A–Z
when source alphabet is letters, produced by the CLI/tool layer) —
threaded through diagnose() and run_battery; gate P5/P9 on it.

**5. [reviewer] Fixture set internally inconsistent until 1–4 land.**
(a) Part 9 lists columnar-transposition-of-English as BOTH a known-family
confident fixture AND near-miss (iii)→uncertain, with no distinguishing
params. (b) The known-family word-position book cipher must be confident,
but child numeric_word_position + parent numeric_book_cipher plausibly
take the top two with margin <0.15 → forced uncertain (see #7). (c)
fixture (i) targets the token gate trivially; (ii)/(iv) depend on
undefined mechanics; no fixture isolates the all-discriminators-pending
trigger. FIX: parameterize each fixture (lengths, symbol counts, expected
top-two, which uncertain trigger fires); check each against the amended
arithmetic on paper before implementation.

## P1

**6. [owner] Shuffle-baseline stats — three sub-points + two same-class
gaps.** (a) Percentile is upper-tail only ("fraction strictly below
×100"); with the auto-relabel rule (null_percentile <95 → reliability
low), a significantly LOW statistic (percentile ≈0) is mislabeled
low-reliability — a compiled-rule bug. (b) n_shuffles=50 cannot
substantiate the "percentile ≥99.9" claims in Part 9/Acceptance (at n=50
only 100.0 is ≥99.9, bounding p at ~1/51). (c) sha256(token_bytes)
undefined for values >255 — Beale max 2906 → seed derivation crashes/
mangles on the flagship target. Same-class: the Part-4 modulo
Monte-Carlo is a parametric (uniform-draw) null, not a shuffle — seeding
unspecified; Part 6's view_hash "canonical rendering" unspecified and
becomes the INV-1 D5 dedup key. FIX: add tail ∈ {upper, lower,
two_sided} per statistic with a p-value field; scale reliability to tail;
require n_shuffles ≥1000 (or a stated exact bound) for any ≥99.9 claim;
define the canonical integer encoding once (e.g. 8-byte big-endian per
token) and reuse for cache keys + view_hash.

**7. [owner] Flat ranking manufactures false uncertainty.** ranked sorts
all families together: nulls_noise_layer (a composable modifier per its
own notes), numeric_book_cipher beside its three children, quagmire under
its parent. Top-two-margin <0.15 doesn't exclude parent/child or modifier
rows → a cleanly-diagnosed book cipher whose parent+child both score high
is forced uncertain, breaking fixture #5b. FIX: rank primaries only for
verdict/margin; report modifiers separately; score subtypes conditionally
within their parent (margin among siblings); the parent's score
represents the family in the primary ranking.

**8. [owner] Hoax flag evidence conjunction contradicts its fixture.**
hoax_or_random_generation cites "monotone-run excess" vs a shuffle
baseline — but a seeded iid uniform stream is distributionally identical
to its shuffles → no excess, so the Part-9 "seeded uniform-random → hoax
supported" test passes/fails by luck under the undefined combination
rule. Monotone-run excess is a structured-fabrication (Gillogly-artifact)
signature, not a uniform-randomness one. FIX: split into
independent_random_like (last-digit uniformity + weak/no Benford decay +
repeat rate at multinomial expectation) and structured_hoax_artifact
(monotone-run excess, digit preference); monotone runs supporting-only
for the latter; state the k-of-n combination rule per flag.

**9. [owner] DoI resource solution-calibrated, unlabeled; out-of-range
handling mandatory.** Constructed by iterating numbering "until the B2
self-check passes" — solution-calibrated (valid; B2 public) but unlabeled
and unbounded ("iterate" invites free numbering search). The gate
constrains only B2-referenced indices (≤1005); B1-referenced-but-B2-
unreferenced indices pass silently wrong. B1 max 2906 exceeds any DoI
word count (~1.3k) → the Gillogly B1 decode MUST define out-of-range
behavior (skip vs placeholder changes the run statistics). FIX: mark the
file solution-bearing (README + module-level bar on use in diagnose()/
candidate ranking, test-pinned); constrain iteration to an enumerated
documented-quirk list, each cited; specify source edition, normalization,
sha256, out-of-range rule (recommend: skip with positions recorded, count
reported).

## P2

**10. [owner] --max-period has no API path.** CLI lists --max-period but
neither diagnose(…) nor the panel/battery signature accepts it, despite
compute_cipher_fingerprint(…, max_period=26) taking it. FIX: add
max_period: int = 26 to diagnose()/run_battery, forwarded to fingerprint
+ P3/P4.

**11. [owner] CandidatePacket contract conflict, with concrete
divergence.** Design D9: CandidatePacket "gains an island_report field."
INV-0: "no schema change (rides inside validation)." Not equivalent:
null-mask packets carry validation=None (candidate_packet.py:150-151,
205) → get NO island data under INV-0, while D9 requires it on every
packet. FIX: pick the canonical contract now (amend design to
validation-nested with the null-mask gap noted, OR add the field);
record as a deviation either way.

**12. [reviewer] Duplicate segmentation on a hot path.**
validate_plaintext_finalist already runs segment_text; the new
island_report key re-segments the same letters. transform_evaluation runs
validation across finalist menus → ~2× segmentation there. FIX:
island_report accepts a precomputed segmentation (or Part-7 wiring passes
`segmented` through); keep the standalone signature intact.

**13. [reviewer] src/investigation/__init__.py ownership unspecified.**
No filename collision with M3 (reading.py, actions.py) or M4
(experiments.py), but the package __init__.py (M1's, M1-scoped docstring)
is a shared merge point three in-flight specs could each edit. FIX: one
line — "do not modify src/investigation/__init__.py; import families/
diagnosis by module path."

**14. [reviewer] Unsolved-reader field mapping unstated.** Manifest field
is related_records (list of dicts: record_id, relationship, area,
safe_context_layers); UnsolvedRecord declares related_record_ids:
list[str] without naming the source field/extraction. hold_for_review is
a rights_class value — the loader test should key off rights_class. FIX:
two sentences in Part 6.

## P3 (minor)

**15. [reviewer] P9 gate for numeric input unspecified.** language_guesser
is IC-only; a language_provisional atom over dense-factorized 298-symbol
Beale tokens is meaningless, yet P9 "always emits" it. Gate to
not_computable/low for non-letter alphabet classes (depends on #4).

**16. [reviewer] Campanelli panel not fully paper-free.** "Unique-token
growth curve," "gap structure," "explicit divergence calls" lack
operationalization (window/bin defs, what constitutes a divergence call).
Wase gets a deviation path; give Campanelli the same or concrete defs.
Otherwise the Part-4/8 P8 math is implementable as written — the
"without the papers" bar is met everywhere except these three.

## Sections needing rewrite before a coder starts

Part 1 (discriminator table), Part 2 (signature + alphabet class), Part 3
(tails/n/encoding), Part 4 (hoax flag split), Part 5 (order-sensitive
verdict), Part 6 (score formula, hierarchy-aware ranking,
discriminator-state rule, max_period), Part 9 (fixture parameterization
re-checked against the amended arithmetic). Findings 9–16 ride the same
revision. After rework, the revised spec gets a SECOND spec-review pass
(NEEDS REWORK depth warrants re-review before implementation).
