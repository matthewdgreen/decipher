# Codex/Astra CLI observation — 2026-09-07

Outcome: a successful blind Quagmire III recovery and useful interface evidence,
**not a demonstrated extension of the previous Codex solving frontier**.

## Fixed run and provenance

The user launched the [runtime-only handoff](../codex_astra_r1_handoff.md) in a
fresh solving task, selecting Astra. The task self-reported Codex/GPT-6; reliable
served-model metadata is unavailable. Report this as a user-selected Astra
observation, not a verified model-identity benchmark. The reported checkout was
`c4554c5` plus the R0/R1 working-tree changes.

Completed registry document:
`artifacts/codex_astra_r1/registry/0747b2f9a527/investigation.json`.
SHA-256: `2c2a5a3026223cec4a5f9e831ee2c033a418f59496622d8378831214130cb78d`.
This gitignored evidence is local; the report does not replace the full artifact.
No subsequent replay modified the closed investigation.

The input was development case `r96f9e4d54195`, 501 letters, with cipher family
withheld. Four local experiments completed in about 7.6 minutes from creation
to terminal closure:

1. Automated periodic search: unreadable ordinary-Vigenere candidate (2.398 s).
2. Quagmire keyword lengths 5–8, periods 4/8, 5,000 hillclimbs × 250 restarts:
   readable recovery (39.718 s).
3. Period-4 alternative, 5,000 × 500: damaged partial (36.097 s).
4. Period-8 confirmation, 5,000 × 500: same readable recovery (42.332 s).

`quagmire_readable`, `quagmire_alternative`, and `quagmire_period8_confirm`
share content hash
`e0cb0a20868dfae2bec1618fe1dd5152e6db06d66c62bf73eb3d40f37d01b6ed`.
After the result was fixed, grading found **501/501 letters correct**.
Ground truth did not enter search, comparison, or declaration.

The agent preserved unusual source wording rather than editorially repairing
it, tested a competing period hypothesis, and recorded a best partial with
`accepts_as_solution=false`. It closed **unsolved**, correctly: this was a
keyless investigation with zero independent-reader attestations. Exact
post-hoc recovery does not retroactively turn that into verified acceptance.

## What is—and is not—new

The [V3/MCP evidence matrix](../evidence/v3_vs_mcp_matrix.md) already records
100% Codex recovery on a blind 566-character round-6 Quagmire III case, using
the same Rust search family. The
[agentic frontier results](../evidence/agentic_frontier_suite_results.md)
record a prior Quagmire case requiring escalation to 64 restarts × 50,000
hillclimbs after failed shallower attempts. Previous Codex sessions already
demonstrated diagnosis, search escalation, alternative comparison, and use of
reading/verification tools; they also explored composite and homophonic cases.

The new session demonstrates that this client could navigate the structured
CLI, question an overconfident period-4 hint, retain alternatives, and report
interface defects. Those are encouraging observations, not established
capabilities absent from Sol. The historical reports often say only “Codex,”
without independently verified Sol identity; ciphertexts, budgets, code, and
interfaces also differ. This single successful instance cannot support
an Astra-versus-Sol improvement claim. A matched, blind, repeated comparison
would be needed; no such model run is scheduled by R2.

## Defects taken into R2

- `experiment-submit --wait` returned stale running status/slots after completion.
- The status window printed unmapped characters for metadata-decoded candidates
  while candidate/decode inspection already displayed the recovered reading.
- Duplicate finalists looked like separate readings; recovered mode keys were
  persisted but missing from normal candidate inspection.

R2 fixes these and adds hash-bound partial retention. The period diagnosis's
confidence, stale next-step guidance, and CLI abbreviation ergonomics remain
separate observations; they did not prevent this recovery and are not silently
bundled into R2 as new routing policy.
