# Transposition + Homophonic Capability

Current status: Decipher has first-pass support for known ciphertext transform
replay before homophonic solving.

Implemented pieces:

- Zenith-compatible transform pipeline representation: `{"columns": ..., "rows": ..., "steps": [{"name": ..., "data": ...}]}`
- Token-order transformers for:
  - `Reverse`
  - `ShiftCharactersLeft`
  - `ShiftCharactersRight`
  - `LockCharacters`
  - `NDownMAcross`
  - `Transposition`
  - `UnwrapTransposition`
- Automated runner replay: a benchmark/frontier case may carry a known
  transform pipeline; the runner applies it before routing into the normal
  solver stack and records the pipeline in artifacts.
- Agent branch overlays: a branch can carry a transformed token order, so
  decode/search tools read the transformed stream instead of displaying a good
  key against the original scrambled order.
- Synthetic ladder packet:
  `frontier/transposition_homophonic_ladder.jsonl`
- Zodiac 340 known-replay fixture:
  `frontier/zodiac340_known_replay.jsonl`

Calibration notes from 2026-04-26:

- Synthetic ladder, dev/screen profile: all 5 rows pass; transform rows reach
  100.0% char accuracy.
- Synthetic ladder, full profile: all 5 rows pass; no-transform baseline
  reaches 99.8% and all known-transform rows reach 100.0%.
- Z340 known replay uses the ciphertext-transformer sequence from Zenith's
  `zenith-inference/src/main/resources/config/zenith.json`: two
  `NDownMAcross` passes, one right shift, one lock range, and final small
  reverse repairs.
- The Z340 transform replay itself is consistent with the published solution:
  applying the copied pipeline gives a 332/340 same-symbol consistency check
  against the known plaintext, so remaining errors are mostly solver/model
  quality rather than transform order.
- With Decipher's redistributable `models/ngram5_en_parity.bin`, Z340 known
  replay currently reaches 52.9% char accuracy in full budget.
- With the external Zenith binary model
  `other_tools/zenith-2026.2/zenith-model.array.bin`, the same known replay
  reaches 96.2% char accuracy in full budget.

Important limitation:

- Open-ended transform discovery is not solved yet. The current
  `search_transform_homophonic` tool is a bounded screen over simple candidate
  transforms, meant for experiments and agent guidance, not headline claims.

External solver audit from 2026-04-26:

- AZdecrypt was cloned locally at `other_tools/azdecrypt-src` from
  `https://github.com/doranchak/azdecrypt` (`cb416ea`). This is the most useful
  mature, source-level reference we found for the next search-harness design.
  Its README and FreeBASIC source show multiple relevant combined solvers:
  `Substitution + columnar rearrangement`, `Substitution + columnar
  transposition`, `Substitution + simple transposition`, `Substitution + row
  bound`, `Substitution + units`, `Columnar transposition (keyed)`, `Columnar
  rearrangement (keyed)`, `Grid rearrangement (keyed)`, `Simple
  transposition`, and `Periodic transposition`.
- AZdecrypt's design suggests that we are not entering greenfield territory at
  the algorithm-concept level. Mature solvers already treat transposition as a
  family of bounded search problems with explicit knobs: column counts, search
  depth, route/period families, matrix direction, and output of candidate
  matrices. For example, the shipped settings include
  `Substitution + columnar transposition & rearrangement` search depth and
  `Substitution + simple transposition` PC-cycle options.
- The most useful AZdecrypt idea for Decipher is a staged search harness:
  generate candidate transform matrices or pipelines, score them with cheap
  structural signals, run one or a few homophonic/substitution solves on the
  promising candidates, then spend full budget only on the survivors. AZdecrypt
  explicitly uses "periodic redundancy" for periodic-transposition search and
  records additional matrix output; this maps naturally to Decipher artifacts
  that should retain the top rejected transform candidates, not just the final
  winner.
- CryptoCrack was cloned locally at `other_tools/cryptocrack-repo` from
  `https://github.com/CryptoCrackProgram/CryptoCrack` (`ab93f37`). That repo is
  a ClickOnce/binary deployment with language data and manifests; a source-file
  scan found no `.cs`, `.vb`, `.cpp`, `.bas`, `.java`, or `.py` implementation
  files. CryptoCrack's public site and README are still useful for capability
  taxonomy: it is a broad classical-cipher GUI with many transposition solvers
  and multilingual language data, but it is not currently an inspectable source
  base for Decipher to port.
- zkdecrypto-lite and Zenith remain narrower references for this specific next
  step. Local zkdecrypto is mainly a homophonic hillclimber rather than a
  general transform-search framework. Zenith has the transform vocabulary and
  some experimental mutation search around `UnwrapTransposition`, but it does
  not appear to offer a mature open-ended Z340-style transform search harness.

Implications for Decipher's next search-harness plan:

- Build our own Decipher-native harness, but borrow AZdecrypt's shape:
  candidate-family enumeration, cheap structural ranking, staged budgets, and
  rich artifact output.
- Represent every candidate as a provenance-bearing transform pipeline, e.g.
  `{family, params, pipeline, inverse_mode, source}`. This keeps the harness
  compatible with the known-pipeline replay already implemented.
- Start with narrow families before open-ended search:
  - known replay and minor variants of the existing Zenith-style operations
  - whole/range reverse, shifts, lock ranges, `NDownMAcross`, and
    `UnwrapTransposition`
  - columnar rearrangement/transposition over bounded grid widths
  - simple route transforms such as spiral, L-route, split, row/column period,
    and inverted/non-inverted matrix forms
  - periodic transposition only after the matrix/statistics layer is solid
- Use staged budgets:
  - `screen`: transform-only signals plus maybe one cheap solver seed
  - `rank`: more seeds on top candidates
  - `full`: normal `zenith_native` budget on a small finalist set
- Record enough data for scientific comparison: original/transformed token
  order summary, candidate family and parameters, structural scores, solver
  score, readability/diagnostic score, selected candidate, and top-N rejected
  candidates.

Recommended next step:

- Improve the redistributable English n-gram model or add model-specific
  expectation rows for Z340. The transform layer is good enough for known
  replay; the large remaining gap is model/search quality on the solved Z340
  plaintext distribution.
