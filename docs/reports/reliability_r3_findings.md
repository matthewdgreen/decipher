# R3 local audit — findings and handoff

2026-09-07. **Implementation and local measurement complete; human source review
pending.** No live verification or new solving session was run. No runtime
gate, terminal status, candidate, benchmark source, or saved artifact changed.

## What the evidence says

1. **The saved reader can reject an exact reconstruction for editorial reasons.**
   The unspaced `synth_en_250nb_s4` control (`d65f5f4876a7`, attestation
   `b401892c3b15`) exactly matches its saved reference letters, but the reader
   withheld solution acceptance while requesting articles/polished prose. It
   did recognize the passage as readable English. The spaced control
   (`29b8f89ad6ee`) was accepted. This is **one rejection among two exact
   controls**, not a stable 50% population rate and not proof that the current
   prompt still behaves this way.
2. **False acceptance remains unmeasured.** All 13 saved verdicts were recovered
   against their original UTF-8 content hashes. Only two have the predeclared
   synthetic reference-identity labels. The known-negative denominator is
   zero; the remaining 11 reconstruction labels are unavailable/unreviewed.
   Recorded verifier usage is `openai` / `gpt-5.5`; reliable served identity,
   original prompt, and historical policy version are absent. No historical
   verdict is interpreted using a retroactively reconstructed current policy.
3. **Text-only verification has an information limit.** The 17-case calibration
   packet includes identical `DOG` readings whose construction references are
   respectively `CAT` and `DOG`. Both are fluent and can arise from consistent
   partial substitution keys. A reader given only the candidate cannot know
   which is the intended reconstruction. Its acceptance is reading evidence,
   not an independent proof of cipher correctness.
4. **The historical causal mix is not yet established.** The four fixed outputs
   produce 522 diagnostic character edit units: 12 conditional/low-confidence
   consistent-mapping candidates, five occurrence-conflict candidates, and 505
   unresolved units. Do not treat the pooled counts as population frequencies;
   the pages have different lengths and damage levels. All four keys reproduce
   their candidate letter streams, which plainly does not make all four
   decipherments correct. Word edits and 60 reference-notation units are
   recorded separately, with no double-counted causal claim.

The three-axis [outcome proposal](../reliability_r3_outcome_contract.md) follows
from those distinctions. It does not propose weakening the existing gate on
the basis of this small sample. A bounded prospective comparison is specified
but needs explicit approval, model selection, and label review before any spend.

## Reproduction and verification

- [Generated report](reliability_r3_audit.md) and [full evidence JSON](reliability_r3_audit.json).
- Reader-only cases: `artifacts/reliability_r3/runtime/candidates.jsonl`.
- Separate construction/source labels: `artifacts/reliability_r3/grading/labels.json`.
- Script: `scripts/audit_verification_residuals.py`; source and script checksums
  are in the output. A second full generation produced identical files.
- **98 passed in 5.05 seconds** across the new audit tests and existing R0,
  candidate-selection, firewall, attestation, retention, and inspector tests.
  The full solver suite was not rerun: R3 changes no solver/runtime code.

## Remaining gate

The repair-rethink requirement calls for human review of all low-confidence
and E3 cases plus at least 10% of the rest. The queue contains 885 required
units across four pages (including overlapping character/word/notation views).
No human decisions have been supplied; reviewed fractions are unavailable.

Start with the [12-character Borg worksheet](reliability_r3_source_review.md).
Use a reviewer with suitable manuscript/Latin/German expertise; uncertainty
may remain after review. R3 stays open until this review is completed or the
user explicitly revises its scope. Do not silently advance to R4 or claim that
alignment patterns establish transcription errors.
