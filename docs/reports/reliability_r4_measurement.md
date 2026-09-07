# R4 paired routing measurement

R3 source review remains open; historical scores are provisional.

Solver revision: `9eaf309fee3ece258e202ddd186e2ee290874b8c`. Local automated only; no paid calls.

Attempted 60/60 scheduled arms. Execution counts: `{'completed': 59, 'interrupted': 1}`.
Known worker wall: 877.6s; unknown duration for 1 attempts. Conservative charged budget: 1117.7s / 10,800s.
Complete, execution-context-matched primary synthetic pairs: 11/12. The tables retain incomplete and cross-context pairs descriptively, rather than dropping them.

## Primary synthetic comparisons

| Case | Family / role | Seed | Blind char | Supplied char | Delta | Blind → supplied execution |
|---|---|---:|---:|---:|---:|---|
| `r2364d0069826` | quagmire3 / held_out | 61001 | 39.8% | 100.0% | 60.2% | completed → completed |
| `r2714bba5ed4f` | vigenere / held_out | 61001 | 41.7% | 100.0% | 58.3% | completed → completed |
| `r28e40081dfcb` | simple_substitution / held_out | 61001 | 100.0% | 100.0% | 0.0% | completed → completed |
| `r30f56fe30434` | homophonic_substitution / held_out | 61001 | 49.4% | unknown | unknown | completed → interrupted |
| `r63d10cd9134b` | simple_substitution / development | 61001 | 100.0% | 100.0% | 0.0% | completed → completed |
| `r766311e4367b` | vigenere / development | 61001 | 32.1% | 100.0% | 67.9% | completed → completed |
| `r8c66ad5f5ed5` | homophonic_substitution / development | 61001 | 95.6% | 95.6% | 0.0% | completed → completed |
| `r8c9fc7f9326e` | substitution_transposition / development | 61001 | 100.0% | 100.0% | 0.0% | completed → completed |
| `r96f9e4d54195` | quagmire3 / development | 61001 | 31.5% | 100.0% | 68.5% | completed → completed |
| `rade10dabbb0c` | columnar_transposition / held_out | 61001 | 69.5% | 69.5% | 0.0% | completed → completed |
| `rd783dfec76e6` | substitution_transposition / held_out | 61001 | 100.0% | 100.0% | 0.0% | completed → completed |
| `rf6e129136170` | columnar_transposition / development | 61001 | 100.0% | 100.0% | 0.0% | completed → completed |

## Predeclared development repeats (not necessarily independent)

| Case | Family / role | Seed | Blind char | Supplied char | Delta | Blind → supplied execution |
|---|---|---:|---:|---:|---:|---|
| `r63d10cd9134b` | simple_substitution / development | 61002 | 100.0% | 100.0% | 0.0% | completed → completed |
| `r63d10cd9134b` | simple_substitution / development | 61003 | 100.0% | 100.0% | 0.0% | completed → completed |
| `r766311e4367b` | vigenere / development | 61002 | 32.1% | 100.0% | 67.9% | completed → completed |
| `r766311e4367b` | vigenere / development | 61003 | 32.1% | 100.0% | 67.9% | completed → completed |
| `r8c66ad5f5ed5` | homophonic_substitution / development | 61002 | 95.6% | 95.6% | 0.0% | completed → completed |
| `r8c66ad5f5ed5` | homophonic_substitution / development | 61003 | 95.6% | 95.6% | 0.0% | completed → completed |
| `r8c9fc7f9326e` | substitution_transposition / development | 61002 | 100.0% | 100.0% | 0.0% | completed → completed |
| `r8c9fc7f9326e` | substitution_transposition / development | 61003 | 100.0% | 100.0% | 0.0% | completed → completed |
| `r96f9e4d54195` | quagmire3 / development | 61002 | 31.5% | 100.0% | 68.5% | completed → completed |
| `r96f9e4d54195` | quagmire3 / development | 61003 | 31.5% | 100.0% | 68.5% | completed → completed |
| `rf6e129136170` | columnar_transposition / development | 61002 | 100.0% | 100.0% | 0.0% | completed → completed |
| `rf6e129136170` | columnar_transposition / development | 61003 | 100.0% | 100.0% | 0.0% | completed → completed |

## Historical diagnostics — provisional

| Case | Family / role | Seed | Blind char | Supplied char | Delta | Blind → supplied execution |
|---|---|---:|---:|---:|---:|---|
| `r28e01a013def` | homophonic_substitution / historical_diagnostic | 61001 | 76.4% | 76.4% | 0.0% | completed → completed |
| `r4b69511c80d8` | homophonic_substitution / historical_diagnostic | 61001 | 49.3% | 49.3% | 0.0% | completed → completed |
| `rb95fa4ff3b16` | simple_substitution / historical_diagnostic | 61001 | 91.3% | 91.3% | 0.0% | completed → completed |
| `rd4421d9528aa` | simple_substitution / historical_diagnostic | 61001 | 43.4% | 43.4% | 0.0% | completed → completed |

## Controls — no expected solve

| Case | Family / role | Seed | Blind char | Supplied char | Delta | Blind → supplied execution |
|---|---|---:|---:|---:|---:|---|
| `r344a95fba0cd` | None / random_control | 61001 | unknown | unknown | unknown | completed → completed |
| `r56a9357dfd05` | bifid / unsupported_control | 61001 | 32.6% | 32.6% | 0.0% | completed → completed |

## Route and saved-candidate evidence

| Case / seed | Blind route / solver | Supplied route / solver | Saved-minus-delivered char gap (blind / supplied) |
|---|---|---|---|
| `r2364d0069826` / 61001 | homophonic / zenith_native | periodic_polyalphabetic / quagmire3_shotgun_rust | 0.0% / 0.0% |
| `r2714bba5ed4f` / 61001 | homophonic / zenith_native | periodic_polyalphabetic / periodic_polyalphabetic_screen | 0.0% / 0.0% |
| `r28e01a013def` / 61001 | homophonic / zenith_native | homophonic / zenith_native | 0.0% / 0.0% |
| `r28e40081dfcb` / 61001 | substitution / native_substitution_continuous_anneal | substitution / native_substitution_continuous_anneal | 0.0% / 0.0% |
| `r30f56fe30434` / 61001 | homophonic / zenith_native | unknown / None | 0.0% / unknown |
| `r344a95fba0cd` / 61001 | homophonic / zenith_native | homophonic / zenith_native | unknown / unknown |
| `r4b69511c80d8` / 61001 | homophonic / zenith_native | homophonic / zenith_native | 0.0% / 0.0% |
| `r56a9357dfd05` / 61001 | transposition / transposition_permutation_search | transposition / transposition_permutation_search | 0.0% / 0.0% |
| `r63d10cd9134b` / 61001 | substitution / native_substitution_continuous_anneal | substitution / native_substitution_continuous_anneal | 0.0% / 0.0% |
| `r63d10cd9134b` / 61002 | substitution / native_substitution_continuous_anneal | substitution / native_substitution_continuous_anneal | 0.0% / 0.0% |
| `r63d10cd9134b` / 61003 | substitution / native_substitution_continuous_anneal | substitution / native_substitution_continuous_anneal | 0.0% / 0.0% |
| `r766311e4367b` / 61001 | transposition / transposition_permutation_search | periodic_polyalphabetic / periodic_polyalphabetic_screen | 0.0% / 0.0% |
| `r766311e4367b` / 61002 | transposition / transposition_permutation_search | periodic_polyalphabetic / periodic_polyalphabetic_screen | 0.0% / 0.0% |
| `r766311e4367b` / 61003 | transposition / transposition_permutation_search | periodic_polyalphabetic / periodic_polyalphabetic_screen | 0.0% / 0.0% |
| `r8c66ad5f5ed5` / 61001 | homophonic / zenith_native | homophonic / zenith_native | 0.0% / 0.0% |
| `r8c66ad5f5ed5` / 61002 | homophonic / zenith_native | homophonic / zenith_native | 0.0% / 0.0% |
| `r8c66ad5f5ed5` / 61003 | homophonic / zenith_native | homophonic / zenith_native | 0.0% / 0.0% |
| `r8c9fc7f9326e` / 61001 | composite_substitution_transposition / composite_substitution_transposition_peel | composite_substitution_transposition / composite_substitution_transposition_peel | 0.0% / 0.0% |
| `r8c9fc7f9326e` / 61002 | composite_substitution_transposition / composite_substitution_transposition_peel | composite_substitution_transposition / composite_substitution_transposition_peel | 0.0% / 0.0% |
| `r8c9fc7f9326e` / 61003 | composite_substitution_transposition / composite_substitution_transposition_peel | composite_substitution_transposition / composite_substitution_transposition_peel | 0.0% / 0.0% |
| `r96f9e4d54195` / 61001 | composite_substitution_transposition / composite_substitution_transposition_peel | periodic_polyalphabetic / quagmire3_shotgun_rust | 0.0% / 0.0% |
| `r96f9e4d54195` / 61002 | composite_substitution_transposition / composite_substitution_transposition_peel | periodic_polyalphabetic / quagmire3_shotgun_rust | 0.0% / 0.0% |
| `r96f9e4d54195` / 61003 | composite_substitution_transposition / composite_substitution_transposition_peel | periodic_polyalphabetic / quagmire3_shotgun_rust | 0.0% / 0.0% |
| `rade10dabbb0c` / 61001 | transposition / transposition_permutation_search | pure_transposition / transposition_permutation_search | 0.0% / 0.0% |
| `rb95fa4ff3b16` / 61001 | substitution / native_substitution_anneal | substitution / native_substitution_anneal | 0.0% / 0.0% |
| `rd4421d9528aa` / 61001 | homophonic / zenith_native | homophonic / zenith_native | 0.0% / 0.0% |
| `rd783dfec76e6` / 61001 | composite_substitution_transposition / composite_substitution_transposition_peel | composite_substitution_transposition / composite_substitution_transposition_peel | 0.0% / 0.0% |
| `rf6e129136170` / 61001 | transposition / transposition_permutation_search | pure_transposition / transposition_permutation_search | 0.0% / 0.0% |
| `rf6e129136170` / 61002 | transposition / transposition_permutation_search | pure_transposition / transposition_permutation_search | 0.0% / 0.0% |
| `rf6e129136170` / 61003 | transposition / transposition_permutation_search | pure_transposition / transposition_permutation_search | 0.0% / 0.0% |

## Limitations

- Primary synthetic cases, development repeats, historical diagnostics and controls are separate strata.
- Repeated request seeds are not automatically independent engine seeds.
- Missing/timeout output is unknown quality, never a zero-accuracy score.
- Saved-text maxima are post-hoc diagnostics, not exhaustive generated-menu quality or a selectable oracle.
- Historical scores use unreviewed frozen references; no source/repair conclusions until R3 review.
- No provider verification or live-agent advantage was measured.
- The sandbox-to-host cleanup change is recorded; any boundary-straddling pair is descriptive, not a clean causal/timing comparison.

The JSON companion binds each result file, request and delivered candidate by hash, and records timing, routes, seed evidence, delivery checks and available saved-text maxima.
