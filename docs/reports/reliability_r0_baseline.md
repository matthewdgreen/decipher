# R0 reliability baseline

Grading-side diagnostic. No solver or provider calls were made.

Audit code revision: `c4554c5323a64d0a0662220ad379f07dc427dc2a`; packet `reliability-r0-v1`.

## Saved artifacts

Historical delivery and current-code replay are separate measurements.

| Run | Replay | Historical delivered char | Best saved char, current render | Current selection gap | Roundtrip failures |
|---|---|---:|---:|---:|---:|
| `8d5bce9769b1` | completed | 96.50% | 96.50% | 0.00% | 0 |
| `ac129831aebc` | completed | 91.01% | 91.01% | 0.00% | 0 |
| `29b8f89ad6ee` | completed | 100.00% | 100.00% | 0.00% | 0 |
| `d65f5f4876a7` | completed | 100.00% | 100.00% | 0.00% | 0 |
| `9f547bfcf55a` | completed | unknown | unknown | unknown | 0 |
| `640e959623f4` | completed | 77.81% | 77.81% | 0.00% | 0 |
| `407e29ec7c70` | completed | unknown | unknown | unknown | 0 |
| `1d6d78083226` | completed | unknown | unknown | unknown | 0 |

## Pilot

18/18 runtime cases available. Generation used fixed keys/seeds without solver-based rerolls.
Six supported families have distinct-source development and held-out cases; four historical anchors and two negative controls complete the pilot.
Plaintext, keys, family labels, source references and role membership are in the gitignored grading store. Runtime JSON contains only opaque id, ciphertext, format, language and ciphertext hash.
Published corpus passages may be known to LLMs; freshness refers to keys/ciphertexts and the within-packet source split.

## Fixed R4 protocol

Two family-blind/family-supplied arms; identical language, models and limits. One primary run per case/arm, plus two additional predeclared seeds on each of six development cases: at most 60 runs.
180 seconds wall per run, 720 CPU seconds per process, four workers, one concurrent run, three hours total wall ceiling. Timeouts and engines ignoring supplied seeds are reported, not rerolled.
Near-exact character recovery: 99%; material paired character gap: 2 percentage points. Both are descriptive criteria, not declaration gates.

## Evidence limitations and next work

- No historical artifact is assumed to include every generated candidate. Best-saved is a lower bound on best-generated quality.
- Missing historical revision/model checksum remains unknown; the current model inventory cannot retroactively establish old provenance.
- Saved verifier verdicts apply only to their original content hashes. Corrected rendering requires a new verdict.
- R1 exercises persistence and interface parity; R2 tests comparison/retention, guided by the per-run findings below.
- R3 audits residual/verification labels; R4 alone measures new routing outcomes. Local replay makes no claim about Astra or live-agent improvement.

### `8d5bce9769b1`
- Historical snapshots flattened canonical source boundaries for null-mask branch(es): nullmask_main.
- Historical verifier rejection(s) were bound to the unsegmented content for: nullmask_main. Corrected renderings require fresh verification.
- Post-hoc best nullmask_main, wide_nullmask_result ranked 3 by the ground-truth-free scalar ordering.

### `ac129831aebc`
- No candidate-selection finding in this saved state.

### `29b8f89ad6ee`
- No candidate-selection finding in this saved state.

### `d65f5f4876a7`
- No candidate-selection finding in this saved state.

### `9f547bfcf55a`
- No candidate-selection finding in this saved state.

### `640e959623f4`
- No candidate-selection finding in this saved state.

### `407e29ec7c70`
- Historical snapshots flattened canonical source boundaries for null-mask branch(es): nullmask_route_rank.

### `1d6d78083226`
- No candidate-selection finding in this saved state.
