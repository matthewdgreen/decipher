# Non-LLM Candidate Ranker Plan

## Goal

Build a fast, transparent scorer that helps choose among solver finalist menus
without calling an LLM. The target decision is comparative:

> Given several candidates from the same cipher/search basin, which one is the
> better damaged plaintext?

This is deliberately different from judging whether text is fully solved.

## Ground-Truth Firewall

Training and calibration reports may use solved benchmark plaintext after
candidates have already been produced. Runtime scoring must consume only:

- candidate text or page-level runtime features,
- solver-native scores,
- language diagnostics,
- permitted context metadata,
- a frozen model trained offline.

Ground truth must not affect candidate generation, search routing, repair
adoption, branch selection, or agent tool outputs.

## Model Shape

Use the existing `LinearLanguageQualityModel` container first. It is simple,
inspectable, cheap, and already wired into null-mask ranking. Prefer pairwise
training over absolute regression:

- group candidates by menu/source page,
- label pairs by post-hoc character gap during offline calibration,
- learn features that rank better candidates above worse candidates.

The first implementation keeps the schema shared with
`LANGUAGE_QUALITY_FEATURES`. Text candidates populate text/lattice features;
multi-page repair candidates populate aggregate runtime/repair evidence
features and leave unavailable text fields neutral.

## Candidate Sources

Initial sources:

- null-mask finalist menus from automated artifacts,
- saved probe JSONL rows,
- multi-page global repair probe JSON files,
- later: transform finalist menus, transposition menus, polyalphabetic menus.

Current support:

- `scripts/train_language_quality_scorer.py --global-repair-json ...`
- `scripts/report_language_candidate_ranker.py --global-repair-json ...`
- Global repair candidates are preserved by source/edit identity during
  training-data dedupe, since many edits share the same preview prefix while
  changing page-level evidence.

## Evaluation Protocol

1. Candidate-only training first.
   Exclude clean ground-truth rows and corpus positives when testing finalist
   ranking. The task is "choose among damaged candidates," not "recognize clean
   plaintext."

2. Hold out by group.
   For Copiale, train on some pages/menus and hold out another page/menu. For
   global repairs, generate several menus from different source finalists or
   page packets so leave-one-group-out has real teeth.

3. Report top-N capture.
   Exact top-1 is useful but brittle when post-hoc labels differ by tiny
   amounts. Track top-3/top-5 capture, mean best-label rank, and feature deltas
   between top-predicted and best-labeled candidates.

4. Keep runtime use conservative.
   Even if the ranker improves calibration reports, use it as one vote beside
   scalar validation, ensemble score, and repair evidence until it repeatedly
   beats the current heuristic scorer on held-out menus.

## Immediate Experiments

1. Generate multiple global-repair menus:
   - different `--section` values,
   - different source labels,
   - smaller/larger pair envelopes,
   - p068/p084-focused and five-page packets.

2. Run leave-one-group-out ranker reports:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_language_candidate_ranker.py \
  --language de \
  --global-repair-json artifacts/copiale_multipage_experiment \
  --objective pairwise \
  --candidate-only \
  --feature-set all \
  --min-label-delta 0.001 \
  --output-dir artifacts/language_quality/global_repair_ranker_eval
```

First smoke status: using the two saved global-repair menus in
`artifacts/copiale_multipage_experiment`, the report path ran end-to-end
(`20` examples, `2` groups, `70` full-model pairs). This is only a wiring
check, not a calibration result: one saved menu predates the richer aggregate
feature schema, and two highly related groups are not enough to claim
generalization.

Fresh global-repair packet status: four newly generated menus in
`artifacts/language_quality/global_repair_ranker_inputs` produced `64`
examples across `4` groups. Pairwise leave-one-group-out ranking selected the
post-hoc best candidate at rank `1` for all four held-out groups. This is
promising because the original robust-score order had the post-hoc best at
ranks `14`, `4`, `8`, and `4`; however, it is still a calibration smoke, not
proof of generalization. The groups are related, two menus are near-duplicates,
and several label gaps are small.

Broader packet status: ten additional menus in
`artifacts/language_quality/global_repair_ranker_inputs_broad` produced `240`
examples across `10` groups. The learned pairwise ranker was useful but not
dominant: mean best-label rank `5.9`, top-1 `3/10`, top-3 `6/10`, top-5
`7/10`. The simple `language_quality` baseline remained competitive
(mean rank `7.2`, top-1 `5/10`, top-3 `6/10`, top-5 `7/10`). A combined
fourteen-group report improved the learned ranker to mean best-label rank
`5.21`, top-1 `5/14`, top-3 `8/14`, top-5 `9/14`, versus the
`language_quality` baseline at mean rank `7.64`, top-1 `6/14`, top-3 `8/14`,
top-5 `9/14`. Current read: the learned ranker adds signal, especially in
mean rank, but it is not yet strong enough to auto-adopt repairs. Keep it as
a report/reranking diagnostic until we have more independent menus and better
tie/near-tie handling.

Harsher holdout status: `scripts/report_language_candidate_ranker.py` now
supports `--holdout-group-by` so evaluations can hold out whole source
experiments instead of individual menus. This exposes sibling-menu leakage.
On the combined packet (`304` examples), ordinary menu holdout reports mean
rank `5.21`, but source-experiment holdout drops to mean rank `16.0`
(`1/7` top-1, `3/7` top-3). Regret metrics make this less dire but still not
production-ready: ordinary menu holdout has mean top-predicted label gap
`0.043%` and all `14/14` predictions within `0.5%` of the best post-hoc
candidate; source-experiment holdout has mean gap `0.346%`, with `6/7`
within `0.5%` and one genuinely bad held-out family
(`copiale_multipage_p017_p035_p052_p084_5cf9b8e553d9`). Current priority is
therefore not runtime adoption; it is generating more independent menus and
using clustered holdouts/regret as the main calibration lens.

Failure-family diagnostics: `scripts/report_language_ranker_failure_family.py`
now expands a bad held-out group into a compact markdown explanation, linking
the ranker summary back to the underlying global-repair candidate rows and
their page-level runtime/post-hoc evidence. The first report on
`copiale_multipage_p017_p035_p052_p084_5cf9b8e553d9` shows a real miss
(`1.82%` post-hoc label gap): the model preferred the single-null `S005`
family because it had stronger language-coherence, word-lattice, and
content-quality signals, while the post-hoc best used `S005,S072`. This points
to a likely next feature family: mask-family consistency and cross-candidate
evidence, not just per-candidate smoothness.

Mask-family feature experiment: global repair candidates now carry
ground-truth-free mask-family aggregate features computed from sibling
candidates with the same source experiment: support, average validation,
balanced score, dictionary score, binary score, and robust score. The ranker
report also supports `--training-group-by`, so pairwise training can compare
sibling menus inside one source experiment instead of learning only within an
individual menu where family features are constant. On source-experiment
holdout, family features with source-level training improved mean rank from
`16.0` to `12.71`, but did not fix the `5cf9...` family. Adding
nonnegative weights made `5cf9...` a near miss (`0.30%` gap instead of
`1.82%`) by moving the top pick into the correct `S005,S072` family, but hurt
overall top-k behavior. Current read: family evidence is real but still too
coarse; the next step is to separate "mask-family selection" from
"within-family edit ranking" rather than expecting one scalar model to do
both.

Two-stage mask-family experiment: `scripts/report_language_candidate_ranker.py`
now has `--two-stage-mask-family`, which trains a first-stage mask-family
model and then ranks edits within the selected family. On source-experiment
holdout, the first-stage family model was fairly good at family selection
(mean best-family rank `1.43`, best family top-1 `5/7`, top-3 `7/7`) and it
fixed the `5cf9...` wrong-family miss by selecting `S005,S072` with only a
`0.18%` top-label gap. But candidate-level rank worsened overall
(two-stage mean best-label rank `15.86` vs scalar `12.71`). Current read:
family selection is becoming useful, but edit ranking inside the selected
family needs its own objective and features; the two-stage architecture is
conceptually right but not yet an accuracy win.

Two-stage top-k follow-up: the evaluator now supports
`--two-stage-edit-group-by {training_group,mask_family}` and
`--two-stage-family-top-k N`, so the edit model can either train at the source
group level or within mask families, and can choose among a shortlist of the
top-k predicted mask families. On the same source-experiment holdout:

| Edit grouping | Family top-k | Scalar mean rank | Two-stage mean rank | Two-stage mean gap | Two-stage top-1/top-3/top-5 |
|---|---:|---:|---:|---:|---|
| `training_group` | 1 | 12.71 | 13.43 | 0.108% | 2/3/3 |
| `training_group` | 2 | 12.71 | 15.86 | 0.321% | 2/4/4 |
| `training_group` | 3 | 12.71 | 12.71 | 0.321% | 2/4/4 |
| `mask_family` | 1 | 12.71 | 16.43 | 0.108% | 2/3/3 |
| `mask_family` | 2 | 12.71 | 16.71 | 0.346% | 1/3/4 |
| `mask_family` | 3 | 12.71 | 16.00 | 0.346% | 1/3/4 |

This confirms the split signal: a strict family-first policy can reduce regret
when the family model avoids the wrong family, but it often chooses a mediocre
edit inside the chosen family. A top-3 family shortlist collapses back toward
the scalar model because the correct family is usually included and the edit
ranker dominates. Training edit ranking by mask family did not help with the
current features. Next work should add genuinely edit-level evidence
(per-page consensus stability, local-window repair deltas, and richer
cross-page disagreement features), rather than just regrouping the same
feature vector.

Edit-level feature slice: global repair candidates now expose a first batch of
ground-truth-free repair features:

- average and minimum runtime validation delta,
- fraction of pages with non-negative runtime validation delta,
- per-page agreement across validation, language-quality, dictionary, binary
  n-gram, and pseudo-word signals,
- validation-delta stability across pages,
- language/binary/dictionary/pseudo-word delta controls,
- edit-count control,
- repair-acceptance/positive-signal control.

These use only `repair_evidence.pages` and `repair_acceptance` runtime fields;
post-hoc character accuracy and calibration flags remain excluded from runtime
features. `feature-set=no_solver` excludes these features, since they are
solver/repair evidence rather than text-only language evidence.

First hard holdout result with edit features:

| Evaluation | Mean rank | Mean gap | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|---:|
| Previous scalar family-feature model | 12.71 | 0.321% | 2 | 4 | 4 |
| Edit-feature scalar model | 14.57 | 0.300% | 4 | 5 | 5 |
| Edit-feature two-stage top-3 | 14.57 | 0.300% | 4 | 5 | 5 |

This is better on capture count and regret, but worse on mean rank. The
important failure is still `5cf9...`: the runtime repair deltas and acceptance
signals strongly favor the single-null `S005` family, while the post-hoc best
candidate is `S005,S072`. That is exactly the kind of false positive this
ranker must learn to resist before runtime adoption. Current read: the new
features are useful diagnostic evidence, but they are not sufficient. The next
feature slice should focus on detecting "too-smooth local improvement" traps:
local edits that raise runtime score by creating plausible word islands while
damaging cross-page consistency.

Trap-detector feature slice: the ranker now also gets changed-window and
corroboration controls:

- correlated validation gain: runtime validation improvements must be backed
  by language-quality, binary n-gram, dictionary, pseudo-word, and page-level
  consensus signals;
- changed-window quality and quality delta;
- changed-window letter diversity, repetition, and changed-page rate.

These are still runtime-only features derived from `repair_evidence.pages`
and `changed_excerpt.before/after`; they do not use post-hoc calibration
flags. An unconstrained pairwise model overfit these features and regressed
slightly, but a nonnegative pairwise model gave the strongest hard-holdout
result so far:

| Evaluation | Mean rank | Mean gap | Top-1 | Top-3 | Top-5 | Within 0.5% |
|---|---:|---:|---:|---:|---:|---:|
| Previous scalar family-feature model | 12.71 | 0.321% | 2 | 4 | 4 | 6/7 |
| Edit-feature scalar model | 14.57 | 0.300% | 4 | 5 | 5 | 6/7 |
| Trap-feature scalar model | 14.86 | 0.386% | 2 | 5 | 5 | 6/7 |
| Trap-feature nonnegative scalar model | 6.14 | 0.186% | 2 | 3 | 4 | 7/7 |
| Trap-feature nonnegative two-stage top-3 | 6.14 | 0.186% | 2 | 3 | 4 | 7/7 |

The nonnegative constraint matters because most of these controls are
designed with a consistent orientation: higher should mean safer or more
supported. Letting the tiny calibration set assign arbitrary negative weights
made the model learn shortcuts. With nonnegative weights, the bad `5cf9...`
case moves into the correct `S005,S072` family, though the exact best edit is
still only rank `8`. This is not enough for automatic adoption, but it is now
useful as a conservative review/reranking signal: all held-out top picks are
within `0.5%` of the best known candidate, and the mean rank is materially
better than dictionary, validation, ensemble, and prior learned variants.

Cross-page robustness slice: global repair candidates now also expose
page-floor/range and edit-consistency controls:

- minimum and range of per-page repair signal consensus;
- range of per-page validation deltas;
- minimum and range of changed-window quality after repair;
- changed-window gain agreement;
- cross-page edit consistency, so edits supported by several independent page
  windows are preferred over one-off local artifacts.

This slice also hardened the tiny linear ranker numerically. Pairwise training
now falls back from singular `solve` to `lstsq`, nonnegative weights are
clipped, Pearson reporting ignores non-finite/overflowing values, and feature
normalization uses a small `0.01` scale floor. The scale floor matters because
near-constant runtime features can otherwise produce huge z-scores and
absurd-looking raw model scores on small calibration sets.

Latest hard-holdout result with cross-page features, nonnegative pairwise
training, and two-stage top-3 family selection:

| Evaluation | Mean rank | Mean gap | Top-1 | Top-3 | Top-5 | Within 0.5% | Best-family mean rank | Family top-1/top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trap-feature nonnegative scalar model | 6.14 | 0.186% | 2 | 3 | 4 | 7/7 | n/a | n/a |
| Cross-page + scale-floor two-stage top-3 | 6.71 | 0.186% | 2 | 3 | 4 | 7/7 | 1.14 | 6/7, 7/7 |

The cross-page features improved family selection slightly
(`6/7` exact family, `7/7` top-3 family), but did not improve exact edit
ranking. A too-large `0.05` feature-scale floor made the model overly blunt
(mean rank `7.14`, top-3 `0/7`), while `0.01` preserved the useful ranking
signal and kept raw scores bounded. Current read: cross-page features should
remain in reports and model packets, but they still are not evidence for
automatic repair adoption. The best runtime use is a conservative shortlist:
pick or display the top family/families, then review several edits inside
those families.

Review-shortlist slice: the ranker report now records exactly that safer
policy. `--two-stage-review-shortlist-k N` asks whether a small two-stage
review menu contains the post-hoc best candidate, and how close the best item
inside that menu is. This is an offline evaluation metric for agent/human
review, not a runtime adoption rule.

The report also emits a diverse review shortlist, which prevents one
high-scoring family from monopolizing the whole menu by taking at least the
best candidate from each top family before filling remaining slots. On the
same hard source-experiment holdout with cross-page features, nonnegative
pairwise training, and top-3 family selection:

| Review K | Flat exact-best | Flat mean gap | Diverse exact-best | Diverse mean gap | Within 0.5% |
|---:|---:|---:|---:|---:|---:|
| 5 | 4/7 | 0.066% | 4/7 | 0.048% | 7/7 |
| 8 | 5/7 | 0.053% | 5/7 | 0.020% | 7/7 |

This is a more honest success criterion than exact rank-1: the model is
strong enough to build a small review menu that almost always contains a
near-best edit, but still too uncertain to silently choose one repair.

Global-repair integration slice: `scripts/research/copiale/probe_copiale_multipage_global_repair.py`
now accepts `--language-quality-ranker <candidate_ranker.json>` and emits a
`language_quality_ranker` block in the JSON plus a
`Language-Ranker Review Shortlist` section in the markdown. The shortlist is
ground-truth-free: it is scored from runtime/global-repair features and a
saved offline model, then diversified by mask family. The report still prints
post-hoc character accuracy only as calibration. A smoke run on
`copiale_multipage_p017_p035_p052_p068_p084_c998db35f5d9` wrote:

```bash
artifacts/language_quality/global_repair_ranker_smoke/c998db35f5d9.global_repair.json
artifacts/language_quality/global_repair_ranker_smoke/c998db35f5d9.global_repair.md
```

During this integration, `global_repair_feature_dict()` was tightened so
runtime ranker features no longer use `calibration_suspicious_pages` or other
post-hoc flags. Those calibration flags remain useful in reports, but not in
model input.

Batch-review slice: `scripts/research/copiale/run_copiale_global_repair_batch.py` now runs the
global-repair probe over multiple labels from a multipage experiment section.
It discovers section labels, invokes the child probe with consistent ranker
settings, keeps child stdout quiet by default, and writes a manifest with the
ranker-review pick and robust-score pick for each label. Example:

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/research/copiale/run_copiale_global_repair_batch.py \
  artifacts/copiale_multipage_experiment/copiale_multipage_p017_p035_p052_p068_p084_c998db35f5d9.json \
  --section portfolio_local_repair \
  --label-count 5 \
  --language-quality-ranker artifacts/language_quality/global_repair_ranker_eval_source_holdout_crosspage_nonneg_two_stage_top3_diverse_review8/candidate_ranker.json \
  --ranker-review-top-k 5 \
  --ranker-family-top-k 3 \
  --artifact-dir artifacts/language_quality/global_repair_batch
```

Smoke runs wrote manifests under:

```bash
artifacts/language_quality/global_repair_batch_smoke_quiet/
artifacts/language_quality/global_repair_batch_smoke_run/
```

`scripts/research/copiale/report_copiale_global_repair_batch.py` summarizes those manifests by
loading the child global-repair JSON files, comparing the ranker-review pick
against the robust-score pick, listing recurring masks/edits, and showing
top candidates by post-hoc character accuracy, LQ ranker score, and robust
score. It also prints per-label rank matrices with post-hoc ground-truth
character score beside each runtime rank (`LQ`, robust, validation, and
balanced), so disagreements are visible instead of hidden behind "top"
language. Post-hoc character accuracy in this report is explicitly diagnostic
only.

A top-8 batch on the five-page packet
`copiale_multipage_p017_p035_p052_p068_p084_c998db35f5d9` found that the
ranker and robust picks are almost tied at the batch level: ranker better on
`3` labels, robust better on `4`, one tie, and mean ranker-minus-robust
post-hoc character accuracy of about `+0.02%`. The clearest current basin is
the `top9` / `S001,S072` family: several one-edit variants land around
`79.2-79.3%` average post-hoc character accuracy across the five pages. This
supports using the learned ranker as a shortlist builder and family finder,
not yet as an automatic final repair selector.

Next ranker work should shift from "add more scalar features" to "improve
edit choice within a selected family": stronger page-level consensus features,
explicit disagreement between pages, and perhaps a small listwise objective
that prefers a top-N safe shortlist over exact scalar order.

Word-hypothesis repair slice: `scripts/research/copiale/probe_copiale_word_hypothesis_repair.py`
is the first move away from scalar micro-edits. It scans damaged windows in a
multi-page finalist, proposes same-length German dictionary word hypotheses
for garbled substrings, turns each hypothesis into one or more global symbol
edits, then scores the edited shared key across all pages. This is still
runtime/ground-truth clean; post-hoc character scores are calibration only.
A first `top9` smoke produced multi-symbol hypotheses such as
`EEDERES -> ANDERES` (`S045->A`, `S113->N`) and `TEAKE -> TEILE`
(`S079->I`, `S080->L`). The old broad runtime scorer still overvalues some
implausible local repairs, so the script reports runtime ranking,
word-hypothesis ranking, and post-hoc calibration separately.

The next adjudication slice added occurrence-level collateral checks: for each
edited symbol occurrence, the script compares the strongest dictionary-like
word island covering that position before and after the edit. This explicitly
penalizes repairs that improve one visible word while damaging many other
word islands. On the `top9` smoke this demoted tempting but contextually
suspect runtime picks like `EUNDE -> HUNDE` and lifted repairs such as
`KUNGER -> JUNGER` and `TEAKE -> TEILE`, matching the strongest post-hoc
calibration basin in that small probe. Current scope is same-length
substitutions only; insertion/unmasking and deletion/null alignment are the
next repair-generation extension.

Portfolio batch slice: `scripts/research/copiale/run_copiale_word_hypothesis_batch.py` runs the
word-hypothesis probe across a finalist section and writes a manifest plus
comparison summary. The first 12-label `portfolio_local_repair` batch found
that adjudication improves over raw runtime/word-hypothesis ranking on average
and exactly chooses the best `top9` repair (`KUNGER -> JUNGER`, `S080:K->J`,
about `79.3%` post-hoc character accuracy). However, across the full portfolio
it often prefers very plausible local `LEHRLINGE/LEHRLINGEN` repairs while
post-hoc best repairs are frequently smaller `RECHTS`/`REDEN`/`DEREN`-style
edits. This suggests the next repair selector needs a notion of page/global
leverage: not only "is this target word plausible and collateral-safe?", but
"does this edit repair a structurally important repeated damaged cluster?"

Combination slice: the probe now supports compatible word-hypothesis sets with
`--max-hypothesis-set-size`, `--combination-candidate-limit`, and
`--max-combinations`. Singletons are always evaluated alongside combinations,
so a bad bundle cannot hide a useful individual repair. A `top9/top6` smoke
with pair search produced better post-hoc candidates than singleton-only
search, including `KUNGER -> JUNGER` plus `SETZEE -> SETZEN` around `79.4%`.
It also exposed the expected selector failure mode: additive target-word
scores can overvalue bundles of locally plausible repairs whose global
leverage is weak. The next selector slice should score marginal contribution
within a bundle: how much each added repair changes runtime quality, collateral
damage, and repeated-cluster evidence beyond the best singleton/subset.

Marginal-contribution diagnostics are now attached to each variant. For
multi-hypothesis bundles, the report compares the bundle against its best
evaluated subset and records deltas for adjudication, robust runtime score,
validation average, language quality, and post-hoc calibration. The first
experimental marginal selector confirms that this is useful evidence but not
yet a final selector: it makes additive target-word overfitting visible, but
can still prefer low-damage local repairs over the stronger post-hoc
`RECHTS`/`SETZEN`-style repairs. Treat the marginal score as a diagnostic
column for now. The next scoring step should add repeated-cluster/global
leverage features rather than only subset deltas.

No-target adjudication slice: the word-hypothesis probe now reports
`adjudication_no_target_score` and `target_leverage_score`. The former
intentionally excludes the repaired target word span and scores only collateral
effects, so it can distinguish "the edited word looks pretty" from "the edit
also helps or preserves the wider shared-key text." The latter is the gap
between full adjudication and no-target adjudication, useful for spotting
repairs whose apparent value is mostly target-word leverage. These are
diagnostic columns, not final selectors.

The probe also reports post-hoc calibration analogs:
`post_hoc_char_no_target_avg`, `post_hoc_char_target_avg`, and per-variant
baseline-excluding-target spans. These use benchmark plaintext only after the
variant has already been generated. They are intended to answer whether a
repair improved the wider text or merely the word we targeted. The current
implementation redacts the same projected character spans from decryption and
plaintext, which is appropriate for these aligned Copiale page projections but
should not be reused as a generic solver signal.

First scorer repair from the disagreement inspection:
- collateral word-island gains/damages are now reliability-weighted before
  entering adjudication. Exact short words and fuzzy long matches still count,
  but they no longer overpower broader evidence. This directly addresses the
  `S049:N->S` miss where broken `NUR`/near-`GEGENWART` islands were punished
  more strongly than the global GT calibration warranted.
- target-only repairs now receive an explicit runtime penalty when an edit has
  no collateral occurrences outside the hand-picked target words. This demotes
  attractive local traps such as `DIENSTAO -> DIENSTAG` plus
  `ARDEREN -> ANDEREN`, which improve target words but do not move the rest of
  the shared-key text.

Current result on the `top6/medium` diagnostic: the target-only trap is
successfully demoted, and `S049:N->S` is less severely punished by
`AdjNoTarget`, but the main selector still misses the best post-hoc repair.
It now prefers other locally plausible word repairs (`LEHRLINGE`,
`DIESER`/`DIENER`, `JUNGER`) that have collateral support but weak global
character improvement. The next step is therefore not simply more local
word-island tuning; it needs a stronger repeated-cluster/global-leverage
feature, or a second-stage shortlist reader/ranker, to distinguish locally
pretty repairs from repairs that actually improve the shared mapping.

Repeated-cluster/global-leverage slice: the word-hypothesis probe now builds
an edit-support table from the generated word hypotheses and computes
`global_leverage_score` for each variant. The score combines independent
symbol→letter proposal support, collateral occurrence/page breadth, weighted
word-island context, and a discount for target-only edits. This is still
ground-truth-free. The first `top6/medium` rerun shows useful movement but not
selection parity: target-only repairs stay demoted and `S049:N->S` gains a
positive global signal, but the highest global/adjudication rows are still
dominated by locally plausible `SETZEN`/`LEHRLINGE` families. The best
post-hoc row (`IUNGER -> JUNGER` plus `RECHTN -> RECHTS`) remains outside the
top exact scalar ranks. Treat `global_leverage_score` as another diagnostic
and portfolio-diversity signal for now; the likely next practical step is a
second-stage shortlist/reranker that reviews diverse high-scoring families
rather than relying on one scalar.

Compact repair-delta reporting now lives in
`scripts/research/copiale/report_copiale_word_repair_delta.py`. It compares each repair against
baseline or its best evaluated subset, showing the word corrections, symbol
edits, post-hoc character delta, adjudication delta, no-target adjudication
delta, marginal score, and robust-score delta. This gives a quick view of
whether a bundle is actually adding value beyond a singleton.

Controlled breadth-curve slice: `scripts/research/copiale/run_copiale_word_repair_breadth_curve.py`
runs fixed breadth settings over selected finalist labels while preserving the
full compact variant list. The first `top9/top6` small-vs-medium curve is at
`artifacts/language_quality/word_repair_breadth_curve/copiale_multipage_p017_p035_p052_p068_p084_c998db35f5d9/`.
Results:
- `top9`: medium breadth evaluated `244` variants but did not beat the small
  best candidate. Best post-hoc stayed around `79.4%`, with
  `KUNGER -> JUNGER` plus `SETZEE -> SETZEN`.
- `top6`: medium breadth evaluated `257` variants but also did not beat the
  small best candidate. Best post-hoc stayed around `79.1%`, centered on
  `RECHTN -> RECHTS` plus `IUNGER -> JUNGER`.
- None of the current runtime rankers captured the best post-hoc candidate in
  the top 10 for these labels. This is the important lesson: modest breadth is
  already finding the useful local candidates, but selector alignment remains
  weak. More breadth alone is unlikely to be the next best use of time unless
  paired with stronger global/repeated-cluster repair features.

3. Compare against simple policies:
   - robust score,
   - balanced score,
   - validation average,
   - existing language-quality blend.

4. If held-out rank improves, add an opt-in ranker column to repair reports.
   Do not auto-adopt repairs from the model until acceptance is calibrated.

## Open Questions

- Are the current aggregate repair features enough, or do we need richer
  window-level features?
- Should multi-page repair menus use a separate model from text finalist menus,
  or can one shared model work?
- Can synthetic corrupted candidates help, or do they overfit to unrealistic
  word-island artifacts?
- What is the minimum menu diversity needed before pairwise training becomes
  meaningful?
