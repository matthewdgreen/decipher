# Borg 0109v Case State

Last updated: 2026-04-26

## Current Read

`parity_borg_latin_borg_0109v` is no longer a total failure case for the
agentic path. The best current branch is readable as a medieval Latin
veterinary or pharmaceutical passage about treating sick chickens. The agent
consistently identifies the broad subject matter: a treatment applied briefly,
more chickens remaining alive, chickens dying, the cure being pleasant and
without pain, and a concluding applicability phrase.

This is a useful success, but not a clean solve. The decipherment still has
several local errors, suspicious compounds, and word-boundary artifacts. The
current high word score is edit-aware and insertion-friendly: it rewards exact
words after local insertions/deletions/gaps resynchronize, so it should be read
as "many anchor words are correct" rather than "the whole word stream is
perfectly aligned."

## Best Artifact

Best current checkpoint by edit-aware word score:

- Artifact: `artifacts/parity_borg_latin_borg_0109v/d41c93c2e6f9.json`
- Status: solved
- Branch: `main`
- Character score: `95.36%` (`329/345`, edit-aware character alignment)
- Word score: `77.92%` (`60/77`, edit-aware exact-word alignment)
- Agent confidence: `0.75`

Main decrypt:

```text
ETIAM | QUOD | IN | TALI | CUR | PLICARE | U | UEL | AM | PULLO | ET | BREUITER | UT | RLURES | MANERENT | UIUI | ET | HIE | PRO | CERTO | EXER | LIBEBITUR | SI | AUTEM | OMNES | PULLI | MORIEBANTUR | EX | SIMILITER | MORIEBATU | ET | QUIA | SEMEL | APPLISUI | L | PULLISA | QUI | OMNE | MORIEBANTUR | ET | QUIA | I | NOCTE | ERAT | NON | POTUI | PLURES | PULLOS | HABER | ETSI | HABUISSEMUS | L | BERASSEM | EUM | DEI | XR | TIA | SED | DEAECTU | PULL | RUM | MORIEBATURA | ET | IST | CURA | EST | MULTUM | SUAUI | ET | SINE | DOLOREM | ET | ARI | POTEST | QUIBUS | EUA
```

Agent reading summary from that artifact:

```text
This is a medieval Latin veterinary text about treating sick chickens. It discusses: applying a treatment briefly (BREUITER) so that more chickens might survive (MANERENT UIUI); even when all the chickens were dying (OMNES PULLI MORIEBANTUR), similarly others died too (EX SIMILITER); once having applied the treatment to the chickens who were all dying; that at night it was not possible to have more chickens (PLURES PULLOS HABER) even if they had had them (HABUISSEMUS); that the cure (CURA) is very pleasant (MULTUM SUAUI) and without pain (SINE DOLOREM); and that it is possible for those to whom it applies (ARI POTEST QUIBUS). A classic medieval Latin pharmaceutical/veterinary recipe passage about poultry disease treatment.
```

## Continuation Runs

Two continuation runs are useful negative/diagnostic evidence:

| Artifact | Parent | Char | Word | Notes |
|---|---:|---:|---:|---|
| `6a1b5bd9137b.json` | `d41c93c2e6f9` | `95.36%` | `75.32%` | Mostly reread/inspected; merged `PULL | RUM -> PULLRUM`, which made the text more phrase-like but hurt edit-aware word alignment. |
| `c6ed2fb86ec8.json` | `6a1b5bd9137b` | `95.36%` | `72.73%` | Tried additional local resegmentation such as `L | BERASSEM -> LBERASSEM`; final summary was lost to fallback in the original artifact, but the loop now recovers blocked declaration summaries. |

Neither continuation improved the headline character score. Both demonstrate
that plausible human-looking boundary merges can reduce benchmark word score
when the ground truth retains split forms.

## What Is Correct

The following regions are strong exact-word anchors under the current scorer:

- Opening: `ETIAM QUOD IN TALI CUR PLICARE ... UEL ... PULLO ET BREUITER UT`
- Survival/dying phrase: `MANERENT UIUI ET ... PRO CERTO ... SI AUTEM OMNES PULLI MORIEBANTUR`
- Middle: `MORIEBATU ET QUIA SEMEL`
- Later poultry phrase: `QUI OMNE MORIEBANTUR ET QUIA I NOCTE ERAT NON POTUI PLURES PULLOS HABER`
- Later: `HABUISSEMUS L BERASSEM EUM DEI ... TIA SED`
- Closing: `ET IST CURA EST MULTUM SUAUI ET SINE ... ET ... POTEST QUIBUS EUA`

The result is semantically coherent despite local corruption. This is why the
case is now better thought of as a "readable but messy historical-Latin
cleanup" problem rather than a first-pass solving problem.

## Remaining Problems

Known local errors or suspicious readings:

- `U` and `AM` near the start are decoded extras relative to the benchmark
  ground truth.
- `HIE` aligns around ground-truth `HI`; the final `E` is suspicious.
- `EXER | LIBEBITUR` is a readable-looking local repair, but the benchmark has
  `EGER | LIBE | BITUR`. This is a major example where boundary/readability
  and benchmark alignment disagree.
- `EX SIMILITER` is semantically plausible, but the benchmark has
  `EG SIMYLITER`. This is likely an orthography/transcription tension, not
  just a solver error.
- `APPLISUI | L | PULLISA` corresponds to benchmark `APPLIC | UI | PULLS,`.
  The agent sees the Latin shape, but the current key/boundaries do not cleanly
  match the benchmark segmentation.
- `ETSI` is a merge of benchmark `ET | SI`; it reads naturally but scores as an
  insertion/deletion pair.
- `XR | TIA` corresponds to benchmark `GR | TIA`; this may be an abbreviation
  or symbol-confusion problem.
- `DEAECTU` should likely be `DEFECTU`; this is a letter-level error.
- `PULL | RUM` is split in the best artifact and matches the benchmark split.
  Merging it to `PULLRUM` reads more like a word but lowers benchmark word
  score.
- `MORIEBATURA` corresponds to benchmark `MORIEBATUR,`; trailing `A` is an
  extra decoded character/letter.
- `DOLOREM` corresponds to benchmark `DOLORE.`; the final `M` is an extra.
- `ARI` corresponds to benchmark `F | RI`; the current reading is plausible
  only if treated as a boundary/letter repair, not as an exact benchmark match.

## Tooling Lessons From This Case

- Full-reading summaries are valuable: they make it obvious the branch is
  readable even when exact word alignment remains imperfect.
- Resume works and preserves state, but continuing from a readable branch can
  spend too much effort rereading. This prompted the read-only inspection
  sandbox and the "show all words for compact pages" panel change.
- Boundary tools need to be benchmark-aware. A boundary merge can improve
  human readability while lowering the benchmark score if the benchmark
  preserves historical or transcription-specific splits.
- The edit-aware word score is much more informative than the old positional
  word score, but it can still flatter a messy reading by resynchronizing after
  local insertions and deletions.
- `meta_declare_solution` summaries can be lost if the declaration is blocked
  and fallback later declares. The summary builder now recovers the latest
  attempted declaration summary for final display.

## Current Recommendation

Pause intensive work on `0109v` for now. It has served its purpose as a
high-signal loop-design case:

- first-pass solving is strong enough to produce a readable Latin branch;
- remaining work is mostly local historical-Latin cleanup, boundary policy,
  and benchmark-alignment nuance;
- additional iterations on the same branch are not currently producing clear
  gains.

Next useful tests:

- Borg `0045v`, to see whether the same reading/boundary behavior generalizes.
- The English Borg analog, to debug boundary and summary behavior in a language
  we can read directly.
- A small comparison packet of Borg pages after the read-only inspection
  sandbox, so we can measure whether the new loop spends fewer outer
  iterations on rereading.
