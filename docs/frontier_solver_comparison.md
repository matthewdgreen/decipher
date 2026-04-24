# Frontier Solver Comparison

This file tracks side-by-side results on the current automated frontier suite.

Current suite:
- [frontier/automated_solver_frontier.jsonl](/Users/mgreen/Dropbox/src2/decipher/frontier/automated_solver_frontier.jsonl)

Current comparison columns:
- `decipher-parity`: Decipher automated solver using `zenith_native` with the bundled parity model
- `zenith-external`: external Zenith through the harness
- `decipher-zenith-binary`: Decipher automated solver using `zenith_native` with the proprietary Zenith binary model

Status:
- `decipher-parity`: recorded
- `zenith-external`: recorded
- `decipher-zenith-binary`: recorded

## Commands

Decipher parity baseline:

```bash
DECIPHER_HOMOPHONIC_PARALLEL_SEEDS=8 \
DECIPHER_NGRAM_MODEL_EN=models/ngram5_en_parity.bin \
DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native \
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers decipher
```

External Zenith:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers external
```

Decipher with proprietary Zenith binary:

```bash
DECIPHER_HOMOPHONIC_PARALLEL_SEEDS=8 \
DECIPHER_NGRAM_MODEL_EN=other_tools/zenith-2026.2/zenith-model.array.bin \
DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native \
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers decipher
```

## Per-case Results

| Test ID | Class | decipher-parity char | decipher-parity time | zenith-external char | zenith-external time | decipher-zenith-binary char | decipher-zenith-binary time |
|---|---|---:|---:|---:|---:|---:|---:|
| `synth_en_40wb_s1` | `known_good` | 100.0% | 10.0s | 97.1% | 6.7s | 100.0% | 11.6s |
| `synth_en_80nb_s1` | `known_good` | 100.0% | 31.2s | 98.9% | 7.8s | 100.0% | 34.2s |
| `parity_fr_ss_synth_001` | `known_good` | 95.6% | 6.8s | 15.3% | 7.8s | 87.6% | 6.6s |
| `parity_it_ss_synth_001` | `known_good` | 50.3% | 8.0s | 36.3% | 7.8s | 45.4% | 7.4s |
| `synth_en_150honb_s1` | `known_good` | 100.0% | 29.0s | 99.6% | 9.9s | 100.0% | 21.7s |
| `synth_en_80honb_s2` | `shared_hard` | 98.7% | 47.9s | 97.8% | 8.4s | 98.7% | 45.0s |
| `synth_en_200honb_s3` | `shared_hard` | 99.9% | 39.9s | 99.1% | 16.4s | 99.7% | 33.9s |
| `synth_en_200honb_s6` | `shared_hard` | 99.9% | 43.4s | 99.4% | 14.7s | 99.9% | 31.2s |
| `parity_tool_zenith_goldbug` | `known_good` | 99.5% | 7.2s | 96.6% | 6.6s | 99.5% | 6.3s |
| `parity_tool_zenith_zodiac408` | `bad_result` | 97.5% | 67.2s | 99.0% | 8.7s | 99.3% | 38.7s |
| `parity_borg_latin_borg_0077v` | `bad_result` | 9.9% | 42.8s | 6.3% | 8.2s | 9.9% | 30.1s |
| `parity_borg_latin_borg_0109v` | `bad_result` | 11.9% | 6.9s | 9.3% | 7.4s | 13.9% | 5.3s |
| `parity_copiale_german_copiale_p052` | `bad_result` | 8.0% | 51.9s | failed | 0.0s | 8.0% | 32.7s |
| `synth_en_150honb_s3` | `known_good` | 100.0% | 28.6s | 99.5% | 13.6s | 100.0% | 26.5s |
| `synth_en_80honb_s1` | `slow_result` | 99.3% | 46.0s | 98.5% | 8.9s | 99.3% | 48.7s |
| `synth_en_80honb_s3` | `slow_result` | 100.0% | 48.2s | 99.8% | 11.6s | 99.8% | 51.8s |
| `synth_en_80honb_s5` | `slow_result` | 100.0% | 64.8s | 98.5% | 12.7s | 100.0% | 45.5s |
| `synth_en_80honb_s6` | `slow_result` | 100.0% | 68.2s | 99.3% | 13.0s | 100.0% | 45.7s |

## Class Summary

| Class | decipher-parity pass | decipher-parity avg char | decipher-parity avg time | zenith-external pass | zenith-external avg char | zenith-external avg time | decipher-zenith-binary pass | decipher-zenith-binary avg char | decipher-zenith-binary avg time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `known_good` | 5/7 | 92.2% | 17.3s | 4/7 | 77.6% | 8.6s | 4/7 | 90.4% | 16.3s |
| `shared_hard` | 3/3 | 99.5% | 43.7s | 2/3 | 98.8% | 13.2s | 3/3 | 99.4% | 36.7s |
| `slow_result` | 4/4 | 99.8% | 56.8s | 4/4 | 99.0% | 11.5s | 4/4 | 99.8% | 47.9s |
| `bad_result` | 1/4 | 31.8% | 42.2s | 1/4 | 28.7% | 6.1s | 1/4 | 32.8% | 26.7s |

## Current Read

- `shared_hard` is doing what we wanted: both Decipher and Zenith are strong,
  but neither is in a boring all-green regime.
- The proprietary Zenith binary helps Decipher materially on Zodiac and a bit
  on a few historical cases, but it does not close the multilingual
  simple-substitution gaps.
- Decipher is much stronger on the French and Italian simple-substitution
  parity cases in this suite.
- External Zenith is clearly stronger on Zodiac.
- Borg and Copiale remain unresolved for both, with Copiale still constrained
  by wrapper/alphabet limitations on the external Zenith side.
- Focused Borg follow-up on `parity_borg_latin_borg_0109v` found that forcing
  `zenith_native` with the bundled Latin model reaches a reproducible
  no-boundary Latin basin at about `13.9%` char accuracy when the full 8-seed
  budget is used, but it still does not clearly beat the default substitution
  path in practical usefulness.

## Deltas

### Decipher Zenith Binary minus Decipher Parity

Positive values mean `decipher-zenith-binary` is better than
`decipher-parity`.

| Test ID | Char delta | Time delta |
|---|---:|---:|
| `synth_en_40wb_s1` | +0.0 pts | +1.6s |
| `synth_en_80nb_s1` | +0.0 pts | +3.0s |
| `parity_fr_ss_synth_001` | -8.0 pts | -0.2s |
| `parity_it_ss_synth_001` | -4.9 pts | -0.6s |
| `synth_en_150honb_s1` | +0.0 pts | -7.3s |
| `synth_en_80honb_s2` | +0.0 pts | -2.9s |
| `synth_en_200honb_s3` | -0.2 pts | -6.0s |
| `synth_en_200honb_s6` | +0.0 pts | -12.2s |
| `parity_tool_zenith_goldbug` | +0.0 pts | -0.9s |
| `parity_tool_zenith_zodiac408` | +1.8 pts | -28.5s |
| `parity_borg_latin_borg_0077v` | +0.0 pts | -12.7s |
| `parity_borg_latin_borg_0109v` | +2.0 pts | -1.6s |
| `parity_copiale_german_copiale_p052` | +0.0 pts | -19.2s |
| `synth_en_150honb_s3` | +0.0 pts | -2.1s |
| `synth_en_80honb_s1` | +0.0 pts | +2.7s |
| `synth_en_80honb_s3` | -0.2 pts | +3.6s |
| `synth_en_80honb_s5` | +0.0 pts | -19.3s |
| `synth_en_80honb_s6` | +0.0 pts | -22.5s |

Headline read:
- Biggest gain: Zodiac (`+1.8 pts`, `-28.5s`)
- Noticeable historical gain: Borg `0109v` (`+2.0 pts`)
- Main regressions: French and Italian simple-substitution parity

### External Zenith minus Decipher Zenith Binary

Positive values mean external Zenith is better than Decipher running
`zenith_native` on the proprietary Zenith model.

| Test ID | Char delta | Time delta |
|---|---:|---:|
| `synth_en_40wb_s1` | -2.9 pts | -4.9s |
| `synth_en_80nb_s1` | -1.1 pts | -26.4s |
| `parity_fr_ss_synth_001` | -72.3 pts | +1.2s |
| `parity_it_ss_synth_001` | -9.1 pts | +0.4s |
| `synth_en_150honb_s1` | -0.4 pts | -11.8s |
| `synth_en_80honb_s2` | -0.9 pts | -36.6s |
| `synth_en_200honb_s3` | -0.6 pts | -17.5s |
| `synth_en_200honb_s6` | -0.5 pts | -16.5s |
| `parity_tool_zenith_goldbug` | -2.9 pts | +0.3s |
| `parity_tool_zenith_zodiac408` | -0.3 pts | -30.0s |
| `parity_borg_latin_borg_0077v` | -3.6 pts | -21.9s |
| `parity_borg_latin_borg_0109v` | -4.6 pts | +2.1s |
| `parity_copiale_german_copiale_p052` | failed wrapper | -32.7s |
| `synth_en_150honb_s3` | -0.5 pts | -12.9s |
| `synth_en_80honb_s1` | -0.8 pts | -39.8s |
| `synth_en_80honb_s3` | +0.0 pts | -40.2s |
| `synth_en_80honb_s5` | -1.5 pts | -32.8s |
| `synth_en_80honb_s6` | -0.7 pts | -32.7s |

Headline read:
- External Zenith is usually faster, sometimes much faster
- Decipher `zenith_native + Zenith binary` is generally more accurate on this suite
- Exception: Zodiac, where external Zenith is still slightly ahead on accuracy
