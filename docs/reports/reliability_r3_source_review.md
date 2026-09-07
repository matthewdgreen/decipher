# R3 — Human source-review starting batch

Status: **unreviewed**. This is an initial Borg 0109v worksheet, not a completed
human audit or a substitute for the full queue. No proposed edit below has been
applied to a candidate, source transcription, or benchmark reference.

The full [R3 audit](reliability_r3_audit.json) contains 885 required review
units: 515 character edits, 310 word edits, and 60 notation units. Those units
overlap in source passages; they are not 885 independent historical defects.
A reviewer may adjudicate groups sharing a demonstrated cause, recording all
affected IDs. The four source pages, not each row in isolation, are the review
scope. The [review protocol](../reliability_r3_outcome_contract.md#human-source-review-gate)
defines the mandatory coverage and evidence requirements.

## Start with Borg 0109v

This is the strongest retained historical partial in the packet: 96.5% under
the existing benchmark character scorer, with 12 diagnostic character edits.
That percentage is post-hoc reference agreement, not source authentication.

- [Page image](/Users/mgreen/Dropbox/src2/cipher_benchmark/benchmark/sources/borg/images/borg_0109v.jpg)
- [Diplomatic transcription](/Users/mgreen/Dropbox/src2/cipher_benchmark/benchmark/sources/borg/transcriptions/borg_0109v.diplomatic.txt)
- [Canonical transcription](/Users/mgreen/Dropbox/src2/cipher_benchmark/benchmark/sources/borg/transcriptions/borg_0109v.canonical.txt)
- [Reference plaintext](/Users/mgreen/Dropbox/src2/cipher_benchmark/benchmark/sources/borg/plaintext/borg_0109v.txt)
- [Symbol map](/Users/mgreen/Dropbox/src2/cipher_benchmark/benchmark/sources/borg/metadata/borg_symbol_map.json)

Candidate: `nullmask_main` from run `8d5bce9769b1`. Exact content hash and all
source checksums are recorded in the audit JSON. Review IDs below are prefixed
`borg_0109v:character:`. Source-token positions are **zero-based**, before the
candidate's null-mask removals. “Gap” is an alignment gap, not proof that a
source character was added accidentally.

| ID suffix | Source position / symbol | Candidate → reference | Provisional pattern | Reading context |
|---|---|---|---|---|
| 0 | 25 / S012 | U → gap | unresolved | CURPLICARE**UU**EL |
| 1 | 29 / S003 | A → gap | unresolved | UEL**A**PULLO |
| 2 | 70 / S006 | E → gap | unresolved | ETHI**E**PROCERTO |
| 3 | 80 / S013 | A → G | consistent mapping candidate | E**A**ER / E**G**ER |
| 4 | 121 / S013 | A → G | same mapping pattern | E**A**SIMILITER / E**G**SIMYLITER |
| 5 | 125 / S039 | I → Y | occurrence conflict candidate | SIM**I**LITER / SIM**Y**LITER |
| 6 | 156 / S019 | S → C | occurrence conflict candidate | APPLI**S**UI / APPLI**C**UI |
| 7 | 159 / S004 | L → gap | unresolved | APPLISUI**L**PULL |
| 8 | 164 / S047 | I → gap | unresolved | PULL**I**SAQUI |
| 9 | 166 / S014 | A → gap | unresolved | PULLIS**A**QUI |
| 10 | 256 / S013 | A → G | same mapping pattern | DEI**A**RTIA / DEI**G**RTIA |
| 11 | 288 / S014 | A → gap | unresolved | MORIEBATUR**A**ETIST |

Questions for the reviewer:

1. Does the repeated S013 pattern support a key correction, given the actual
   source glyphs and reference conventions? Confirm all three occurrences.
2. Are the S039 and S019 conflicts historical spelling/symbol-use variation,
   reference/transcription disagreement, or genuinely supported local source
   errors? Do not assume the benchmark spelling is necessarily right.
3. Do the extra aligned letters correspond to numeral/null conventions,
   abbreviations, segmentation, damaged source content, or an incorrect key?
   Leave each unresolved when the source evidence cannot decide.
4. Consult the 22 separate notation units and 25 word residuals for this page
   before treating a character discrepancy as an editorial/source restoration.

Suggested response format (one entry may list multiple IDs):

```json
{
  "reviewer": "human reviewer name and relevant expertise",
  "date": "YYYY-MM-DD",
  "review_ids": ["borg_0109v:character:3"],
  "review_bindings": ["copy matching review_binding from the audit JSON"],
  "source_evidence": ["file/checksum, page/line/token and what is visible"],
  "decision": "confirmed pattern or unresolved",
  "confidence": "high | medium | low",
  "rationale": "...",
  "editorial_proposal": null,
  "source_supported": false
}
```

Model-assisted triage may be useful, but must be labeled as model review and
does not satisfy the plan's explicit human-review requirement. No external
reviewer contact or paid verifier call has been made.
