# Fresh Codex solving session — CLI usability observation

Completed 2026-09-07; results are in the development-only observation report.
The original runtime-only brief below is retained for provenance. Do not rerun
this case as a blind held-out test or give its answer-bearing report to a solver.

Use a fresh Astra task in this **current local checkout**, not a fork of the
development task or a new worktree: the changes are uncommitted and the input
is intentionally gitignored.
The user selects Astra in Codex. This is a live-client observation, not a paid
Decipher API run, a model benchmark, or an independent code review.

## Task for the solving agent

Investigate the cipher in
`/Users/mgreen/Dropbox/src2/decipher/artifacts/codex_astra_r1/input.json`.
The input contains only ciphertext, its hash, format, language, and an opaque
case id. The cipher family and answer are intentionally not supplied.

Use the **structured investigation CLI** for all investigation operations.
Read `docs/mcp_onboarding.md`, especially CLI mode and investigation
methodology. You may inspect CLI help, the operation manifest, and relevant
implementation code to understand an operation. Do not modify solver code,
change gates, access external solvers, or read benchmark data, plaintext
libraries, generation scripts, tests, other artifacts, grading files, or
evidence/reports. Do not look up the ciphertext or try to discover its source.
Ground truth must not enter diagnosis, search, ranking, or your conclusions.

Use `.venv/bin/decipher investigation --registry-dir
/Users/mgreen/Dropbox/src2/decipher/artifacts/codex_astra_r1/registry ...`.
Extract only `ciphertext`, `format`, and `language` from the input to build the
canonical `start` object; do not pass the whole wrapper as operation arguments.
Record `git rev-parse HEAD`, dirty state, and the client model shown to you.
Do not claim served-model identity unless independently available.

Budget: at most 20 minutes of investigation and six solver experiments, one
running at a time. Set `DECIPHER_PARALLEL_WORKERS=4` and
`DECIPHER_QUAGMIRE_THREADS=4` for local commands. No external provider calls
are authorized: do not use `--allow-external`, API analysis, verifier
arbitration, or provider keys. One ordinary keyless `verify` request is allowed
to test the typed refusal. Do not make a fake attestation.

Diagnose, choose hypotheses from runtime evidence, search, collect/install
promising candidates, and inspect the reading. Preserve useful partials and
compare alternatives when appropriate. Demonstrate that a fresh CLI invocation
can read the installed candidate. Finish with an explicit, honest outcome:
without independent verification a promising decode is not a verified solution.
Use `declare-unsolved` when closing this keyless investigation, explaining the
verification limitation and whether any cryptanalytic uncertainty remains.

Return the investigation id and registry path, candidate branch names, route
and experiment budgets, decode/interpretation with uncertainties, whether
state survived fresh commands, and any confusing CLI behavior or missing
capability. Do not grade against an answer. The development task will do
post-hoc grading only after your result is fixed.

Do not edit repository source or planning documents. Leave investigation
artifacts in the dedicated registry for the development task to inspect.
