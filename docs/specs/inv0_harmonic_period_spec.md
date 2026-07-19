# INV-0 harmonic period estimation — spec

Status: ready to implement
Author: Fable (main session, extra effort)
Motivating evidence: round-6 dogfood trace (investigation `bbd8eabb899b`),
`docs/evidence/mcp_dogfood_results.md` "Round 6", `docs/evidence/v3_vs_mcp_matrix.md`
finding F4.

## 1. Problem

The periodicity signal in INV-0 picks `best_period` as the single argmax of the
per-period mean-IC table. On a keyed-tableau (Quagmire III) cipher with a true
period of 8, the periodic-IC table shows a clean *harmonic ladder* — the true
period and all its multiples are elevated above the shuffle null:

```
period  8 -> 0.0728   (~70 tokens/column)   <- fundamental
period 16 -> 0.0718   (~35 tokens/column)
period 24 -> 0.0785   (~23 tokens/column)   <- naive argmax (sparsest, least reliable)
shuffle-null mean ~ 0.0446
```

Naive argmax reports `best_period = 24` — the sparsest multiple, whose column
size (~23) even makes the natural-language summary flag it "unreliable; treat as
noise." Kasiski's naive best is the *subharmonic* 2 (most spacings divide 2), so
neither headline names the true period. A competent reader infers 8 from the
ladder. The headline should too.

This also silently *under-credits the polyalphabetic family*: in
`cipher_id._compute_suspicion_scores` the periodic-IC recovery bonus (+0.35) is
gated behind `token_count // best_period >= 25`. With `best_period = 24`,
`566 // 24 = 23 < 25`, so the bonus is withheld. With the correct fundamental 8,
`566 // 8 = 70`, the bonus applies.

## 2. Fix — harmonic-family fundamental-period estimation

Replace the single-argmax selection of `best_period` with a harmonic-family
estimator that:

1. Scores each candidate period `p` by the reliability-weighted IC *elevation*
   (above a null estimate) aggregated across its harmonic family
   `p, 2p, 3p, ...` within `max_period`.
2. Requires `p` to itself carry signal (a *validity gate*), so a mere
   sub-harmonic (a divisor of the true period, whose own fold mixes key
   positions and therefore is NOT elevated) cannot win.
3. Prefers the smallest true fundamental — which, given the family score is a
   sum over a superset for the smallest divisor in a divisibility chain, is
   exactly the max-harmonic-score candidate (tie-break: smaller period).
4. Reconciles with Kasiski GCD support as a corroboration multiplier.
5. Falls back to naive argmax when there is no significant periodicity (so
   monoalphabetic and other non-periodic inputs keep today's behavior).

The **raw per-period `periodic_ic` table is unchanged** in all outputs — only
the *selected* `best_period` / `fundamental_period` and the summary prose change.

### 2.1 New function (public) — `analysis/cipher_id.py`

```python
def estimate_fundamental_period(
    periodic_ic: dict[int, float],
    token_count: int,
    *,
    max_period: int = 26,
    null_ic: float | None = None,
    kasiski_factors: dict[int, int] | None = None,
) -> tuple[int | None, dict[str, Any]]:
    ...
```

Module constants (in `cipher_id.py`):

```python
_HARMONIC_MIN_RELIABLE_COLS = 25   # tokens/column for full reliability weight (matches summary threshold)
_HARMONIC_FOLD_FRAC = 0.6          # p's own elevation must be >= this * family-max elevation to be a valid fundamental
_HARMONIC_SIGNIF_MARGIN = 0.010    # argmax elevation over null below this -> keep naive argmax (matches PERIODIC_RECOVERY_MIN)
_HARMONIC_KASISKI_BONUS = 0.25     # max multiplicative boost from full Kasiski corroboration
```

Algorithm:

```
periods = sorted(periodic_ic)
if empty -> return (None, {"reason": "no_periodic_table"})
argmax = period with max IC, tie-break smaller period
if null_ic is None -> null_ic = median(periodic_ic.values())
elev[k] = periodic_ic[k] - null_ic

if elev[argmax] <= _HARMONIC_SIGNIF_MARGIN:
    return (argmax, {folded: False, reason: "no_significant_periodicity", ...})

cols(k)  = token_count // k
rel(k)   = min(1.0, cols(k) / _HARMONIC_MIN_RELIABLE_COLS)     # tokens-per-column reliability weight

for each candidate p in periods:
    family      = [h*p for h in 1.. while h*p <= max_period, present in periodic_ic]
    family_max  = max(max(0, elev[m]) for m in family)
    score[p]    = sum(max(0, elev[m]) * rel(m) for m in family)
    if kasiski_factors: score[p] *= 1 + _HARMONIC_KASISKI_BONUS * (kasiski_factors.get(p,0) / max(kasiski_factors.values()))
    valid[p]    = elev[p] > 0 and (family_max <= 0 or elev[p] >= _HARMONIC_FOLD_FRAC * family_max)

candidates = [p for p in periods if valid[p] and score[p] > 0]
if none -> return (argmax, {folded: False, reason: "no_valid_harmonic_family", ...})

fundamental = max(candidates, key=lambda p: (score[p], -p))
detail = {
  naive_best_period: argmax,
  null_ic,
  folded: fundamental != argmax,
  reason: "harmonic_fold" | "argmax_is_fundamental",
  harmonic_family: [multiples of fundamental present in table],
  elevated_periods: [k with elev[k] > _HARMONIC_SIGNIF_MARGIN],
  kasiski_corroborates: bool(kasiski_factors and kasiski_factors.get(fundamental,0) > 0 and
                             kasiski_factors.get(fundamental,0) >= 0.5*max(kasiski_factors.values())),
}
return (fundamental, detail)
```

Why the validity gate is essential: for a true period 8, folding at a *divisor*
(2 or 4) mixes distinct key positions per column, so its own IC is NOT elevated
(round-6: elev[4]=0.011, elev[2]=0.001) even though its family (which contains
8, 16, 24) is. The gate `elev[p] >= 0.6 * family_max` rejects 2 and 4; 8 passes
(elev[8]=0.028 vs family_max 0.034). Among the survivors {8,16,24,...}, 8 has the
largest family sum, so it wins.

### 2.2 Fingerprint wiring — `analysis/cipher_id.py::compute_cipher_fingerprint`

Replace:

```python
best_period = max(periodic_ic_dict, key=lambda k: periodic_ic_dict[k])
best_period_ic = periodic_ic_dict[best_period]
```

with a call to `estimate_fundamental_period(periodic_ic_dict, n, max_period=max_period,
kasiski_factors=kasiski_gcds)` (compute the Kasiski analysis *before* this so its
`kasiski_gcds` is available). `best_period` becomes the fundamental;
`best_period_ic = periodic_ic_dict[best_period]`. Keep `periodic_ic_dict`
untouched. Thread the returned `detail` into the summary builder.

### 2.3 Natural-language summary — `_format_natural_language_summary`

Accept the harmonic `detail`. When `detail["folded"]` is true, add a sentence
naming the fundamental and the ladder, e.g.:

> "Period 8 is the fundamental: its multiples (16, 24) are also elevated (a
> harmonic ladder), so the sparser higher multiples are not the true period."

The existing col-size / recovery / Kasiski lines stay, now computed against the
fundamental (so the ~70-tokens/column reliable branch fires for round-6 instead
of the "treat as noise" branch). When Kasiski's naive best is a subharmonic of
the fundamental (fundamental % kasiski_best == 0), say so rather than reporting a
plain disagreement.

### 2.4 Kasiski corroboration in suspicion scoring — `_compute_suspicion_scores`

Line ~587: the corroboration `+0.20` currently fires only on exact equality
`kasiski_best == best_period`. Broaden it to a *harmonic relationship*: fire the
`+0.20` when `kasiski_best == best_period` OR one divides the other
(`best_period % kasiski_best == 0` or `kasiski_best % best_period == 0`). This
credits the round-6 case (fundamental 8, Kasiski subharmonic 2) correctly. The
`elif kasiski_best is not None: +0.05` fallback is unchanged.

AMENDMENT (post-review, 2026-07-19, spec-author sign-off): the harmonic branch
of the `+0.20` additionally requires a `_strong_periodic` gate — reliable
columns, recovery > 0.010 over the null, and best-period IC near the language
reference. Rationale: a small `kasiski_best` (e.g. 2) trivially divides most
periods, so the unconditional harmonic broadening would over-credit
non-periodic inputs; with the gate, weak-periodicity inputs degrade to the
pre-existing `+0.05` fallback (a strict superset of pre-change behavior). The
round-6 motivating case passes the gate and receives the full `+0.20`.
Exact-equality corroboration remains ungated, as before.

### 2.5 Periodicity panel — `analysis/panels.py::panel_periodicity`

Build the per-period IC table locally (same math as `_best_period_ic`, but keep
the per-period values), compute the fundamental with `estimate_fundamental_period`
(IC-only null=median is sufficient; optionally pass Kasiski factors from
`kasiski_report`), and add to `measurements`:

- `periodic_ic_table`: the raw `{period: mean_ic}` dict (unchanged values),
- `fundamental_period`: the folded estimate,
- `naive_best_period`: the argmax,
- `fundamental_detail`: the returned detail dict.

The significance test (`_best_period_ic` shuffle-null) and the
`periodic_ic_recovery` atom firing conditions are UNCHANGED; add
`fundamental_period` into the atom's `measurement`. `best_period_ic` /
`recovery` continue to use the max-IC scalar (that is the significance statistic;
folding only affects which integer period we report).

## 3. Family registry signpost — `investigation/families.py`

Add a pending-discriminator sequencing hint on the `quagmire_keyed` subtype so a
client that has strong `polyalphabetic_periodic` evidence is routed to the keyed
engine instead of abandoning the family after a plain-Vigenere failure.

- Add a `sequencing_hint: str = ""` field to `FamilySpec` (default empty; a plain
  additive field — every other family keeps the default, no other spec changes).
- Set it on `quagmire_keyed`:

  > "If a standard-tableau Vigenere-family search fails at the indicated period,
  > test a keyed-tableau/Quagmire search next before rejecting the
  > polyalphabetic family."

Evidence-honest: the panels cannot statically distinguish keyed vs standard
tableau. This is a *sequencing hint*, not a score change — no atom, weight,
detector, or discriminator changes; no verdict can flip. Word it generically
("keyed-tableau/Quagmire search") so it composes with the sibling
`quagmire3_shotgun` experiment work. **Do not touch
`src/investigation/experiments.py`.**

Surface it: `diagnosis.FamilyDiagnosis` gains a `sequencing_hint` field carried
from the registry into `to_dict()` (so `observe_diagnosis` / `decipher diagnose`
clients see it). It is display-only; `format_diagnosis` may append it under the
subtype line when non-empty. The registry import-time validator is unaffected
(new field is optional, defaulted).

## 4. Tests (new file `tests/test_harmonic_period.py`) + regressions

(a) **Round-6 fixture** — copy the 566-char ciphertext STRING literally into the
test file (ciphertext-only; never read plaintext/keys). Assert
`estimate_fundamental_period` and `compute_cipher_fingerprint(...).best_period`
both yield 8 (not 24), and that `panel_periodicity(...).measurements
["fundamental_period"] == 8`.

(b) **Synthetic Vigenere/Quagmire** — build 2-3 cases with known periods (5, 6,
8) via `analysis.polyalphabetic.encode_plaintext` (plain Vigenere) and
`encode_quagmire_plaintext` (keyed tableau) with fixed seeds/keywords; assert the
folded fundamental equals the true period for each. Include at least one case
whose naive argmax is a *higher multiple* of the true period, to prove folding.

(c) **Non-periodicity guard** — a monoalphabetic-substitution token stream must
NOT fold to a spurious small period: `estimate_fundamental_period` returns the
naive argmax with `folded == False` / `reason == "no_significant_periodicity"`.

(d) **Signpost** — assert `families.FAMILY_REGISTRY["quagmire_keyed"]
.sequencing_hint` is non-empty and mentions "keyed" or "Quagmire", and that a
`polyalphabetic_periodic`-dominant diagnosis surfaces it on the subtype dict.

(e) **Existing suites stay green**: at minimum `tests/test_cipher_id.py`,
`tests/test_panels.py`, `tests/test_diagnosis.py`, `tests/test_analysis.py`, and
the full suite `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`.

## 5. Firewall

Ciphertext-only. The round-6 fixture uses ONLY the ciphertext string. Sealed
plaintext / keys never enter any solving or diagnosis path; they may appear only
in a test's copied ciphertext input, never as an input to the estimator.
