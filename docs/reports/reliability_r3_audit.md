# R3 — Verification and historical residual audit

Packet: reliability-r3-v1. No solver/provider calls or runtime changes.

**Status: local audit complete; required human source review is pending.**
All fractions below are provisional reference-alignment patterns, not reviewed historical causes.

## Four fixed historical anchors

| Page | Character edit units | E1 candidate | E3 candidate | Unclassified | Key replay matches | Human reviewed |
|---|---:|---:|---:|---:|---|---:|
| borg_0109v | 12 | 3 | 2 | 7 | True | 0 |
| borg_0045v | 48 | 4 | 1 | 43 | True | 0 |
| copiale_p017 | 194 | 5 | 1 | 188 | True | 0 |
| copiale_p068 | 268 | 0 | 1 | 267 | True | 0 |

Each row uses its recorded delivered candidate, not a post-hoc best branch. Full
character and word ledgers, notation units, confidence, sources, and checksums are
in the sibling JSON. Character edit units include insertions/deletions; word and
notation counts use separate denominators and must not be added to character fractions.
All low-confidence/E3 units and at least 10% of other units require human review.
Reviewed fractions are **unavailable**, not zero. Conflicts are not transcription scars.

## Saved verifier audit

- Original content recovered by exact attestation hash: 13/13 verdicts.
- Recorded accepts: 2/13; this is not an accuracy estimate.
- Strict-reference false rejections: 1/2 known-positive synthetic references.
- Strict-reference false acceptances: 0/0 known-negative references; rate is undefined with zero denominator.
- Uncertain/unavailable reconstruction labels: 11/13.
- Historical-policy false acceptance/rejection rates: **not estimable** (0 independently labeled verdicts).

Only the two predeclared synthetic control runs get reference-identity labels.
Recorded per-episode model usage is preserved; served identity and exact historical
prompt/policy version are missing. The observed contract is not a recovered policy
version. No verdict is reassigned to a changed rendering or treated as a current-provider rate.

## Calibration and next measurement

The runtime-only calibration file has 17 cases: thirteen constructed
controls and four unreviewed historical candidates. Construction labels and sources
are in a separate grading file. Intelligibility labels are author-provisional;
ambiguous cases explicitly retain null labels. No human review is fabricated.
The matched DOG/DOG pair has identical reader input but opposite reconstruction
labels. It demonstrates an information limit, not a model failure: text-only
verification cannot establish cipher correctness even when a reading is fluent.

The source-review queue contains 885 required units.
See `docs/reliability_r3_outcome_contract.md` for outcome axes, review requirements,
and the bounded prospective experiment. No prospective run is authorized or executed.
