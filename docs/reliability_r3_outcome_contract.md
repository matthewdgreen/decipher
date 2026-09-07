# R3 — Proposed outcome contract and measurement gates

2026-09-07. **Offline proposal, not a runtime policy change.**
The governing sequence remains R0 → R1 → R2 → R3 → R4. This document does not
change `reader_accepts_as_solution`, declaration thresholds, terminal statuses,
solver routing, or provider authority.

## Three independent axes

| Axis | What evidence can establish | What it cannot establish alone |
|---|---|---|
| Cipher reconstruction | Recorded key/pipeline replay, candidate identity, repeatable correspondence to cipher tokens; separately, post-hoc agreement with a known reference | That a fluent replay is the intended decipherment; that the reference itself is a flawless source transcription |
| Intelligible reading | Recoverable language/meaning, uncertainty, support for a reading of the candidate **as written** | The correct cipher key, precise source fidelity, or justification for adding missing content |
| Editorial/source restoration | Individually supported expansions, spelling changes, lacuna restorations, or transcription corrections | Permission to silently replace cipher-derived output or treat an attractive conjecture as source-supported |

Proposed report shape (no new live schema is installed):

```json
{
  "candidate": {"content_hash": "...", "renderer_id": "..."},
  "cipher_reconstruction": {
    "mechanism_replay": "consistent | inconsistent | not_checked",
    "correctness_claim": "unverified | independently_supported",
    "evidence": []
  },
  "intelligible_reading": {
    "target_language_confidence": null,
    "semantic_recoverability": null,
    "assessment": "readable | partial | unreadable | unresolved",
    "reader_provenance": null
  },
  "editorial_restoration": {
    "canonical_text_unchanged": true,
    "proposals": []
  }
}
```

Each editorial proposal needs a source/token span, proposed text, rationale,
support class (`proposed_only`, `occurrence_conflict_supported`,
`independently_reading_supported`, or `source_supported`), and references to
the supporting evidence. An occurrence conflict establishes only that the
local replacement is not a globally consistent key edit. It does not establish
that the replacement is true. A human linguistic judgment is also not, by
itself, a manuscript-image check.

Known-reference accuracy and labels belong in a separate **grading-side**
record. They never appear in solving prompts, candidate rankings, repair
choices, or declaration gates. “Key replay consistent” is deliberately weaker
than “correct reconstruction”: the calibration packet includes identical
`DOG` readings with opposite construction labels. A text-only reader receives
identical information in both cases and cannot distinguish their cipher truth.

Historical spelling, abbreviations, missing spaces, and source lacunae are not
automatically modern-language defects. Conversely, fluency does not establish
source fidelity. Keep the current keyless/verified distinction intact while
this contract is evaluated.

## Historical residual methodology

The grading-side script `scripts/audit_verification_residuals.py` uses four
fixed, already-produced candidates:

- Borg 0109v: delivered branch of `8d5bce9769b1` from the R0 evidence set.
- Borg 0045v: the recorded `main` partial in `683a61a9b8cb`, Stage-1 **retry
  cohort**, replicate 1. This is the 83.6%-character endgame candidate required
  by the repair-rethink question, not the initial cohort's failed 46.4%
  preflight. The artifact's final summary names the branch explicitly.
- Copiale p017/p068: fixed July-13 null-mask baseline outputs `4220d4ce8509`
  and `f4d013acfd27`. These are substantially damaged diagnostic anchors, not
  high-accuracy endgames. Their residual mix must not be generalized to every
  Copiale candidate or to mature solves.

This is purposive diagnosis, not a randomly sampled solve-rate experiment.
No new search or post-hoc candidate promotion occurs. Full source paths and
checksums bind canonical/diplomatic transcriptions, reference plaintexts,
symbol maps, and page images. Source files are available locally but have not
received the required independent human review in this task.

For the provisional character ledger:

1. Use the existing scorer's alignment weights. Enumerate possible partners
   across **all** optimal alignments before assigning a mapping pattern;
   a tie-break is not unique historical evidence.
2. Classify repeated symbols whose uniquely aligned occurrences all require
   the same different letter as **E1 candidates**, conditional on reference
   fidelity and the one-letter-per-rendered-token model. Singleton evidence
   stays low-confidence.
3. Classify one uniquely aligned mismatch among otherwise matching occurrences
   as an **E3 occurrence-conflict candidate**, never a confirmed scar.
4. Leave gaps, alignment ambiguity, unsupported token correspondence, and mixed
   patterns unclassified. Do not redistribute their mass into convenient causes.
5. Classify boundary-only word discrepancies only when the letter streams are
   identical. With simultaneous letter damage, word residuals remain unresolved.
6. Inventory bracketed corrections/expansions and Copiale logogram markers
   separately. A marker's mnemonic is not spelled-out cipher text. The ordinary
   scorer remains unchanged; its metrics and normalization effects are reported
   beside, not replaced by, this diagnostic projection.

Character edit units, word edit units, and source-notation units have separate
denominators. The JSON includes fractions by class **and confidence**, exhaustive
residual ledgers, source-token indices where correspondence is supported, and
the separate current-scorer result. Reviewed fractions remain null until review.

## Human source-review gate

The [repair-rethink requirement](repair_mechanism_rethink.md#70-step-0--residual-composition-measurement-grading-side-runs-first)
requires at least 10% of labeled residuals, every low-confidence label, and every
proposed E3 label to receive human review against canonical transcription and
available source notes/images. R3 has **not** satisfied that gate yet. Do not
describe this provisional report as establishing the causal residual fractions.

The JSON report's `review_queue` identifies every required unit. Each has a
stable `review_id`, candidate hash, source checksums, and `review_binding`.
Review all mandatory cases plus the deterministic every-tenth sample of the
remaining units. Multiple units may be adjudicated together when they share
one demonstrated source cause, but record the complete list of review IDs.

For each decision record:

- reviewer name/role and date, explicitly distinguishing human from model;
- exact review IDs and `review_binding` values;
- source file/checksum and page/line/token evidence consulted;
- confirmed class or `unresolved`, confidence, and rationale;
- any proposed editorial reading in a separate field;
- whether source support actually exists (not merely an occurrence conflict).

Keep unresolved cases unresolved. Source-image review is not fabricated by
opening a file or by accepting a model's suggestion. Historical calibration
labels also stay unreviewed until separately adjudicated. A short starting
worksheet is in `docs/reports/reliability_r3_source_review.md`; it is a triage
batch, not a substitute for the complete required queue.

## Bounded prospective verification experiment — proposed, not authorized

Saved evidence is insufficient to estimate current false-acceptance rates.
Before any live experiment, adjudicate the packet's readability/acceptance
labels independently and freeze a policy target. Keep ambiguous historical
cases descriptive unless source review resolves them.

Proposed envelope:

- One explicitly chosen verifier/model, with configured and actually served
  identities reported separately. No model bake-off is implied.
- Seventeen fixed cases × two prompt conditions × two independent repetitions:
  **68 calls maximum**, one at a time. No outcome-dependent retries; failures
  consume a slot and remain in the denominator as process failures.
- Condition A: the current frozen independent-reader contract. Condition B:
  an offline diagnostic prompt separating readability from source/editorial
  perfection, returning the three axes above. Neither arm mutates a runtime gate.
- At most 2,000 input tokens and 1,000 output tokens per call. Preflight all
  cases; do not truncate a case silently. A case exceeding the cap requires an
  explicit protocol revision before any calls, not an adaptive exception.
- Proposed total spend ceiling **$10**, with per-call worst-case reservation
  using the approved model's then-current pricing. This is a ceiling, not a
  price estimate or authorization. Stop before a call whose reservation exceeds
  the remaining budget. A partially completed experiment is reported as partial.
- Freeze case order with seed `90307`, pair conditions per case, and reverse
  order for repetition two. Record actual prompt bytes/hash, policy/schema
  hash, code revision, candidate hash, token use, latency, failures, and cost.
- Reader inputs contain only opaque case ID, language, and candidate text.
  Ground truth, construction labels, keys, source answers, and prior verdicts
  remain in the separate grading store. The identical-text opposite-label pair
  is an information-limit control, not a claim that the reader should divine
  an unseen cipher key.

Report counts/denominators by case category, prompt condition, and repetition;
separate exact-reference mismatch from independently adjudicated reading-policy
error. Report uncertain labels and excluded cases explicitly. Two repetitions
per case measure limited repeatability, not population rates. This convenience
packet cannot establish broad calibration or superiority of any model.

No live experiment or reviewer call has been executed or authorized by R3.
R4 remains the next planned local measurement after R3's review disposition;
none of these findings predetermine sparse-null routing or a new repair engine.
