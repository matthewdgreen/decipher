# Agent Loop Milestone 2 Comparison

Status: April 2026 checkpoint for the boundary-projection inner-loop prototype.

## What Changed

- The v2 loop now uses a provider-neutral response adapter.
- Artifacts record structured `loop_events`.
- Late `boundary_projection` turns can retry inside the same outer iteration
  when a stale/gated tool is called.
- Late `boundary_projection` turns can also retry inside the same outer
  iteration when a full-reading proposal has the wrong normalized character
  count.
- The analyzer now labels `gated_tool_retry` and
  `same_length_projection_failed`.

## Artifact Summary

| Case | Artifact | Char | Word | Tools | Input tok | Output tok | Est. cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English analog, pre-workflow baseline | `artifacts/english_borg_analog_001/3e8e98fb06fb.json` | 99.2% | 2.9% | 2 | 26,305 | 799 | $0.09 |
| English analog, boundary workflow | `artifacts/english_borg_analog_001/f8c8ead3e9b2.json` | 100.0% | 100.0% | 6 | 99,320 | 1,893 | $0.33 |
| Borg, actuator-only gate tightening | `artifacts/parity_borg_latin_borg_0109v/bb419814f0c3.json` | 13.9% | 7.7% | 19 | 301,407 | 6,039 | $0.99 |
| Borg, count-retry prototype | `artifacts/parity_borg_latin_borg_0109v/0827d5850034.json` | 13.9% | 6.4% | 19 | 337,055 | 7,401 | $1.12 |
| Borg, first word-repair slice | `artifacts/parity_borg_latin_borg_0109v/495a27b339ba.json` | 13.9% | 9.1% | 19 | 336,400 | 6,665 | $1.11 |

## Readout

The English analog validates the boundary/full-reading abstraction. A branch
that was nearly character-correct but badly word-aligned moved from 99.2% char
/ 2.9% word to 100.0% / 100.0% once the agent could state a full reading and
apply boundary repair safely.

Borg `0109v` is different. The loop mechanics improved behavior but did not
move character accuracy. The count-retry runs show that the agent still
struggles to produce a complete 345-character reading proposal. The retry
events are useful observability: they prove the loop caught the failure inside
the boundary workflow, but the model still could not repair it before the
declaration turn.

The first Milestone 3 word-repair slice is included here as context because it
uses the Milestone 2 loop machinery. It improved Borg word score to 9.1% and,
more importantly, changed behavior: the agent used `act_apply_word_repair` for
`NREUITER -> BREUITER` and `SIMALITER -> SIMILITER`, then backed away from
`RLURES -> PLURES` after seeing broad collateral damage.

## Analyzer Labels

Milestone 2 added analyzer coverage for:

- `gated_tool_retry`: a disallowed tool was blocked and retried inside the
  same outer iteration.
- `same_length_projection_failed`: a reading/projection proposal had the
  wrong normalized character count.

Existing reading-discipline labels remain relevant:

- `unattempted_reading_fix`
- `score_overrode_reading`

`act_apply_word_repair` is now treated as a reading primitive by the analyzer.

## Closeout

Milestone 2 is mechanically complete enough to move on. The original plan also
called for a separate next-generation harness entry point; that is deliberately
deferred. The prototype is wired into `run_v2` so existing CLI and benchmark
paths exercise it directly while the workflow shape is still evolving.

The next capability milestone is the durable reading-repair agenda: collecting
candidate word repairs, tracking accepted/reverted/held repairs, and presenting
the declaration turn with unresolved hypotheses rather than just transcript
history.
