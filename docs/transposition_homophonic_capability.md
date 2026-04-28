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
  - `RouteRead` for simple grid routes:
    `columns_down`, `columns_up`, row/column boustrophedon, and clockwise /
    counterclockwise spiral reads
- Automated runner replay: a benchmark/frontier case may carry a known
  transform pipeline; the runner applies it before routing into the normal
  solver stack and records the pipeline in artifacts.
- Agent branch overlays: a branch can carry a transformed token order, so
  decode/search tools read the transformed stream instead of displaying a good
  key against the original scrambled order.
- Initial transform-search core: `analysis.transform_search` now provides
  cheap metadata-free suspicion diagnostics, plausible grid dimensions, a
  provenance-bearing `TransformCandidate`, and a structural candidate screen.
- Agent diagnostic tool: `observe_transform_suspicion` lets the agent inspect
  grid/order-scramble evidence before deciding whether to spend solver budget
  on `search_transform_homophonic`.
- Synthetic ladder packet:
  `frontier/transposition_homophonic_ladder.jsonl`
- Zodiac 340 known-replay fixture:
  `frontier/zodiac340_known_replay.jsonl`

Calibration notes from 2026-04-26:

- Synthetic ladder, dev/screen profile: the original 5-row known-transform
  ladder passes; transform rows reach 100.0% char accuracy.
- Synthetic ladder, full profile: the original 5-row known-transform ladder
  passes; no-transform baseline reaches 99.8% and all known-transform rows
  reach 100.0%.
- Hidden-transform route calibration, 2026-04-26:
  `synth_en_120thonb_hidden_route_s16` withholds its row-boustrophedon
  transform from the solver and runs with `--transform-search rank` plus
  screen homophonic budget. The harness selected
  `route_rows_boustrophedon` at columns=17/rows=27 and reached 100.0% char
  accuracy in 55.3s with the redistributable parity model.
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
- Z340 hidden-transform calibration, 2026-04-26:
  `zodiac340_transform_search_rank` supplies the original Z340 order with no
  known transform pipeline and runs with `--transform-search rank` plus screen
  homophonic budget. The suspicion router correctly recommends screening
  (`activation_score=11`) and prioritizes the 17x20 grid, but the bounded
  candidate family only selects a single partial `NDownMAcross` candidate
  (`down=1`, `across=2`, first 20 tokens). It reaches 40.3% char accuracy in
  51.7s with the redistributable parity model. This is a useful signal-bearing
  failure, not a recovered Z340 transform.
- Re-running the same hidden-transform fixture with the external Zenith model
  reaches 40.0% char accuracy and selects a different wrong candidate
  (`route_columns_down`, 16x21). This suggests the current failure is mainly
  transform-search breadth/ranking, not English n-gram model weakness.
- First generic deepening pass, 2026-04-26:
  the structural screen now records position-permutation metrics inspired by
  AZdecrypt's matrix-statistics view: nontriviality, adjacency preservation,
  repeated step rate, grid row/column step rates, periodic redundancy, inverse
  periodic redundancy, and a nontrivial periodic-structure score. Rank/full
  modes also add generic local mutations around promising candidates: small
  head/tail/row-boundary reversals, range shifts, and bounded row-band
  `NDownMAcross` variants. This expanded the Z340 hidden-search screen from
  46 scored base candidates to 158 scored candidates (112 local mutations).
  With the external Zenith model, the first version selected a mutated
  `ndownmacross_1_2+shift_head4` candidate and reached 38.5% char accuracy in
  80.3s. A follow-up selection-policy tweak added family-diverse solver probes,
  a small local-mutation penalty, and a much smaller plaintext-quality
  tie-breaker. That restored selection to the unmutated
  `route_columns_down` candidate at 40.0% char accuracy in 71.1s. This is not
  a solve, but it is an important diagnostic: the candidate generator is now
  exploring local repairs, while the ranker is no longer dominated by local
  mutation monocultures.
- First finalist-validation pass, 2026-04-26:
  the screen now also records an explicit `matrix_rank_score`, best normal and
  inverse periods, and position-order previews. Solver-backed rank artifacts
  now carry `validated_selection_score`, identity deltas, base-candidate
  deltas for local mutations, validation penalties, and validation reasons.
  The diverse ranker now tries to include the unmutated base candidate when it
  probes a local mutation, so repairs are judged against the transform they
  modify. On the Z340 hidden-transform fixture with the external Zenith model,
  this still selects `route_columns_down` at 40.0% char accuracy in 75.9s, but
  the artifact now clearly shows why local mutations lose: several fail to
  beat their base candidate or fall below the identity selection margin.
- First candidate-breadth pass, 2026-04-26:
  breadth now includes diagonal grid routes, offset-chain routes, non-identity
  columnar key probes (`reverse`, `even_odd`, `odd_even`, `outside_in`,
  `inside_out`) for both transposition and unwrap modes, and a generic
  `SplitGridRoute` transformer that can route or swap horizontal/vertical grid
  regions. This expanded the Z340 hidden-transform structural screen to 271
  deduped candidates (159 base + 112 local mutations). With the external
  Zenith model, the best solver-backed finalist shifted to
  `route_diagonal_down_right` on a 20x17 grid and reached 38.8% char accuracy
  in 73.4s. Split-grid candidates are generated and validated, but they did
  not survive into the top solver-probed finalists on this run. The result is
  still a useful negative calibration: breadth is active, but the current
  matrix/quality ranker is attracted to plausible-looking diagonal false
  positives.
- First two-stage rank pass, 2026-04-26:
  `rank` now expands the structural screen pool, then performs a
  family-diverse Stage A triage before spending solver probes in Stage B. The
  triage report records selected finalists and their family classes, making it
  obvious whether a run is over-focusing on one transform family. On the Z340
  hidden-transform fixture with the external Zenith model, Stage A selected a
  broad finalist set: identity, `NDownMAcross`, two column routes, one diagonal
  route, one offset chain, two columnar probes, one unwrap-columnar probe, and
  one local mutation. The selected Stage B winner was
  `unwrap_transposition_outside_in_key` on a 24x14 grid, but accuracy fell to
  37.4% in 63.1s. This is another useful negative calibration: family breadth
  is now working, but the solver/quality validation still promotes
  plausible-looking columnar false positives.
- First independent-confirmation pass, 2026-04-26:
  `rank` now has a Stage C after solver-backed ranking. It reruns the top
  finalists with independent seed offsets, always includes the identity
  control when available, records plaintext stability, and applies an explicit
  penalty to candidates outside the confirmation budget so unconfirmed
  candidates cannot silently bubble above confirmed finalists. On the Z340
  hidden-transform fixture with the external Zenith model, Stage C confirmed
  four candidates (`unwrap_transposition_outside_in_key`, `route_columns_down`,
  identity, and `route_diagonal_up_left`). All four were flagged as plaintext
  unstable across independent probes. The selected candidate remained
  `unwrap_transposition_outside_in_key`, reaching 37.4% char accuracy in
  83.3s (`artifacts/zodiac340_transform_search_rank_zenith_model_stagec3/...`).
  This is not progress toward a solve, but it is progress toward honest
  finalist accounting: the artifact now shows that the current winners are
  unstable transform false positives, not robust candidate transforms.
- First family-gated selection pass, 2026-04-26:
  Stage C now has explicit family-specific evidence gates. Finalists receive
  labels such as `robust_candidate`, `robust_baseline`,
  `unstable_false_positive`, `near_identity`, and `unconfirmed_candidate`.
  Diagonal, columnar, unwrap, and local-mutation families must clear stronger
  stability and identity-margin gates than simpler route/`NDownMAcross`
  families. The confirmation pass can also adaptively probe a couple of
  near-margin unconfirmed candidates.

  Calibration split:
  - On `synth_en_120thonb_hidden_route_s16`, the gated ranker still selects
    the true hidden `route_rows_boustrophedon` transform as a
    `robust_candidate` and reaches 100.0% char accuracy
    (`artifacts/transform_hidden_route_family_gates1/...`).
  - On `zodiac340_transform_search_rank` with the external Zenith model, the
    gated ranker no longer treats the columnar/unwrap winners as transform
    discoveries. It labels five transform finalists as
    `unstable_false_positive`, adaptively confirms two near-margin candidates,
    and selects the `000_identity` `robust_baseline` instead
    (`artifacts/zodiac340_transform_search_rank_zenith_model_family_gates1/...`).
    This reaches 39.1% char accuracy, but the important result is diagnostic:
    the current bounded search has not found a robust Z340 transform.
- First post-gating breadth pass, 2026-04-26:
  breadth now includes additional route/grille-style families and bounded
  AZdecrypt-inspired grid permutations:
  - `RouteRead` now supports checkerboard even/odd routes, zigzag diagonals,
    row/column interleaves, and progressive row/column shifts.
  - `GridPermute` supports bounded row/column order probes such as reverse,
    even/odd, odd/even, outside-in, and inside-out.
  - Medium candidate generation now includes small composite route templates
    such as route-then-head/tail reverse and route-then-head shift.
  - Medium candidate generation also includes bounded banded
    `NDownMAcross` families with optional lock/shift structure and local
    tail-style reversals. Earlier versions included an exact
    `z340_composite_zenith_template` calibration candidate; that direct
    injection has now been removed from generic candidate generation.
  - Structural screens record `family_counts`, `scored_family_counts`,
    `top_family_counts`, `rejected_reason_counts`, and an `anchor_candidates`
    lane that keeps simple canonical route families visible even when broad
    structural ranking is dominated by flashy false positives.

  Calibration split:
  - On `synth_en_120thonb_hidden_route_s16`, the broader ranker initially
    starved the true row-boustrophedon candidate out of Stage A. The
    `anchor_candidates` lane fixed this: the true route again survives triage,
    is labeled `robust_candidate`, and reaches 100.0% char accuracy
    (`artifacts/transform_hidden_route_breadth3/...`).
  - On `zodiac340_transform_search_rank` with the external Zenith model, the
    broadened screen now has 359 base candidates and 433 scored candidates
    after dedupe/mutations. Stage A includes `NDownMAcross`, route-column,
    diagonal, composite-route, grid-permute, offset-chain, and columnar
    families. The skeptical gates still select `000_identity` as
    `robust_baseline`; transform finalists are labeled unstable false
    positives. This reaches 39.1% char accuracy in 126.3s
    (`artifacts/zodiac340_transform_search_rank_zenith_model_breadth3/...`).
    In short: breadth is active, but the added families have not produced a
    robust Z340 transform yet.
- First near-miss/false-positive diagnostic pass, 2026-04-26:
  solver-backed rank artifacts now include a `diagnostics` block. It records a
  headline conclusion, label/family counts, selected-candidate status, and a
  compact `top_evidence` table for the top finalists. Each evidence row says
  whether the candidate passed independent-seed stability, identity-margin,
  plaintext-quality, and structural-delta checks, plus diagnostic reasons such
  as `failed_stability_gate`, `failed_identity_margin_gate`, or
  `weak_plaintext_quality_signal`.

  Calibration check:
  - On `synth_en_120thonb_hidden_route_s16`, the diagnostics conclude
    `robust_transform_candidate_found`; the true `route_rows_boustrophedon`
    finalist has evidence agreement 4/4 and no diagnostic reasons
    (`artifacts/transform_hidden_route_diagnostics1/...`). Runtime exceeded
    the old 120s watch threshold, so the row passed cryptanalytically but
    still needs expectation/runtime recalibration after breadth expansion.
- First fast-profile pass, 2026-04-26:
  automated/frontier transform search now supports
  `--transform-search-profile fast|broad`. `broad` preserves the current
  research-oriented rank behavior. `fast` keeps the medium structural screen
  and anchor lane, but trims local mutations, solver finalists, and adaptive
  confirmations. This is intended for regression/calibration runs, not
  headline breadth claims.

  Calibration split:
  - On `synth_en_120thonb_hidden_route_s16`, fast rank still selects the true
    `route_rows_boustrophedon` robust transform and reaches 100.0% char
    accuracy in 70.9s, back under the old 120s watch threshold
    (`artifacts/transform_hidden_route_fast_profile2/...`). The fast profile
    now correctly records zero adaptive confirmations.
  - On `zodiac340_transform_search_rank` with the external Zenith model, fast
    rank completes in 65.2s with `solver=transform_search_no_robust_transform`.
    It records the identity diagnostic candidate and the conclusion
    `no_robust_transform_unstable_false_positives`, rather than raising a
    capability-gap error or pretending to recover a transform
    (`artifacts/zodiac340_transform_search_rank_zenith_model_fast_profile3/...`).
- First bounded Z340-template pass, 2026-04-26, now superseded:
  generic breadth was not sufficient to make the known Z340 transform occur
  naturally in the finalist set. An earlier search revision added an explicit
  exact Z340 template candidate as a calibration control. That proved useful
  for validating the solver handoff, but it was too close to answer injection
  for ongoing search work and has been removed from generic candidate
  generation. The remaining path constructs the shape through the transform
  program grammar.

  Calibration split:
  - With fast rank and screen-budget homophonic probes, the hidden-transform
    Z340 fixture selected `z340_composite_zenith_template` as a
    `robust_candidate` and reaches 55.3% char accuracy in 66.9s
    (`artifacts/zodiac340_transform_search_rank_zenith_model_z340template_fast3/...`).
    The preview is visibly on the right text (`...YOU ARE...`), but the screen
    solver budget is undercooked.
  - Known-template replay with the external Zenith binary model and full
    homophonic budget reaches 96.2% char accuracy in 55.8s
    (`artifacts/zodiac340_known_replay_zenith_model_full_budget1/...`).
  - The automated runner performs a final full-budget homophonic
    refinement after cheap rank-mode transform selection when
    `--homophonic-budget full` is requested. With this path, hidden-transform
    Z340 rank selected the template and reached 96.2% char accuracy in 120.5s
    (`artifacts/zodiac340_transform_search_rank_zenith_model_z340template_fast_refine1/...`).
    This was bounded template recovery plus full solve, not open-ended
    discovery; it is retained here as a historical calibration result.
- First transform-program beam pass, 2026-04-26:
  broad transform search now has a `program_search` layer. It composes small
  legal operations from a grammar, structurally prunes each depth with a beam,
  and then hands surviving programs to the same solver-backed rank/confirmation
  path. The first grammar started as a Z340-relevant slice and has now been
  generalized into a small family of grid programs: route reads, split-grid
  route reads, variable split-row banded `NDownMAcross`, mid/late and upper
  lock bands, late/middle shifts, tail repair packs, and local reversals. This
  is the first step from "catalog contains a composite" toward "search
  constructs a composite."

  Calibration:
  - On `zodiac340_transform_search_rank` with the external Zenith model and
    `--transform-search-profile broad`, the program beam selected a
    constructed banded `NDownMAcross` program as the robust transform finalist
    after grammar expansion. Its operation labels were:
    `ndown_top_a2 -> late_shift_right -> mid_late_lock -> ndown_lower_a2 -> tail_repair_pack`.
  - With the final full-budget homophonic refinement, this constructed program
    reaches 96.2% char accuracy in 168.1s
    (`artifacts/zodiac340_transform_search_rank_zenith_model_program_broad3/...`).
  - The exact injected `z340_composite_zenith_template` calibration candidate
    has since been removed. The current generic path must construct the
    Z340-like shape through `program_search`.
  - A transient failure mode appeared when generic local mutations were allowed
    to stack on top of already-composite program-search candidates: the ranker
    selected a stable but wrong mutated program. Program candidates now carry
    their own repair operations and are not fed back into the generic local
    mutation stage.

- Hidden synthetic program-search packet, 2026-04-26:
  `frontier/transposition_homophonic_ladder.jsonl` now contains two hidden
  transform rows that withhold the transform pipeline from the solver. These
  are calibration rows for candidate-search behavior, not benchmark claims.

  - `synth_en_180thonb_hidden_program_s17` hides a non-Z340 banded program on a
    19-column grid. Broad search with screen-budget homophonic probes recovers
    a high-quality transformed solve: 95.8% char accuracy in 97.5s
    (`artifacts/transform_hidden_banded_program_broad_screen1/...`). The
    selected finalist was a direct banded candidate
    (`z340_composite_banded_ndown_across_2`), while a grammar-built
    `program_tail_repair_pack` candidate with the same visible reading was the
    next serious contender. This is a positive breadth signal, but not yet a
    pure program-selection win.
  - `synth_en_120thonb_hidden_route_repair_s18` hides a route read plus a local
    reverse. Broad search generated and solver-probed route-repair program
    candidates, but selected a diagonal-route false positive and reached only
    45.7% char accuracy in 139.9s
    (`artifacts/transform_hidden_route_repair_program_broad_screen1/...`).
    This is a useful negative calibration: route-family breadth exists, but
    finalist validation still needs better protection against structurally
    plausible wrong routes.

Important limitation:

- Open-ended transform discovery is not solved yet. The current
  `search_transform_homophonic` tool is a bounded solver-backed screen over
  explicit candidate families plus an initial transform-program grammar. It can
  now construct the known Z340-shaped calibration template from smaller
  operations, but this should still be described as grammar-bounded program
  search rather than general transposition discovery.
- `observe_transform_suspicion` is intentionally cheap and conservative. It
  can tell the agent that transform search is plausible, but it cannot prove a
  transform or solve a cipher by itself.

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

Current implementation slice:

- `src/analysis/transform_search.py` implements the first reusable harness
  layer:
  - `plausible_grid_dimensions(...)`
  - `inspect_transform_suspicion(...)`
  - `generate_transform_candidates(...)`
  - `screen_transform_candidates(...)`
  - `validate_transform_candidate(...)`
- `screen_transform_candidates(...)` can now opt into `program_search`, which
  composes bounded transform programs from grammar operations and beam-prunes
  each depth before solver-backed ranking.
- The candidate generator currently covers identity, whole reverse, whole
  shifts, row reversals, route reads, split-grid route reads, bounded
  `NDownMAcross`, simple columnar/unwrap probes, composite route repairs, and
  grammar-built transform programs for broad profiles. This is deliberately
  smaller than AZdecrypt's full family list.
- The candidate generator also has an explicit `wide` structural profile. It
  expands route offset chains, progressive row/column shifts, split-grid
  route combinations, row/column grid permutations, affine/cyclic columnar key
  probes, and local range reverse/shift repairs. Wide search is intentionally
  structural-only by default: it can produce a large candidate menu without
  launching homophonic annealing on every candidate.
- The agent prompt now tells the model to call `observe_transform_pipeline`
  and `observe_transform_suspicion` before escalating to
  `search_transform_candidates` when it is considering transposition-like
  explanations without explicit metadata. That structural-only tool exposes
  `breadth=fast|broad|wide`; only after reviewing the candidate menu should
  the agent promote a small finalist set with `search_transform_homophonic`.
  The agent-facing `search_transform_homophonic` tool has an
  `include_program_search` option for grid-like no-boundary cases.
- Automated and frontier runs now accept
  `--transform-search off|auto|screen|wide|rank|full|promote`. `auto` and `screen`
  record cheap suspicion/screen diagnostics in artifacts. `wide` runs a larger
  structural-only sweep and deliberately returns
  `transform_search_structural_only` on unsupported mixed transposition cases
  rather than pretending to solve them. `rank` and `full` run explicit bounded
  solver-backed candidate ranking and can select the best transformed
  candidate as the automated result. `promote` skips a fresh structural sweep
  and instead reuses a prior artifact's `transform_search.screen` block:

  ```bash
  --transform-search promote \
  --transform-promote-artifact artifacts/.../run.json \
  --transform-promote-top-n 10
  ```

  To force exact finalists, repeat
  `--transform-promote-candidate-id <candidate_id>`. Promotion artifacts record
  the source artifact path, requested candidate IDs or top-N, promoted
  candidate IDs, and the normal solver-backed rank/refine metadata.
  Initial validation against the 72k-candidate Z340 wide artifact showed why
  the final stage needs a bakeoff: exact promotion of
  `program_0853_d5_tail_repair_pack` evaluates identity plus that transform
  and reaches 96.2% char accuracy in 92.0s, while naive top-5 promotion
  initially selected a neighboring false-positive transform and reached only
  68.2%. The runner now full-refines the screen-selected transform plus
  close/selectable finalists before making the final pick. On the same top-5
  promotion packet, the full bakeoff switches from
  `program_0868_d5_tail_repair_pack` to `program_0853_d5_tail_repair_pack`,
  refines two candidates, and reaches 96.2% in 270.6s.
- The frontier/CLI surface also accepts
  `--transform-search-profile fast|broad|wide`. Use `fast` for regression runs
  and quick calibration, `broad` when evaluating candidate-family breadth, and
  `wide` when the goal is a large structural candidate sweep before solver
  promotion.
- `--transform-search-max-generated-candidates N` is the explicit safety knob
  for wide structural runs. Wide mode now uses a streaming structural scorer:
  it retains compact counters, family summaries, anchors, and top-N candidates
  instead of materializing every scored candidate and sorting the whole list.
  Candidate generation is now iterator-backed for this path as well, while
  the older list-returning API remains as a compatibility wrapper. On Z340,
  the current widened route+repair families now produce the requested 100k
  candidate pipelines under a 100k cap; the previous internal route-repair cap
  exhausted the generator at about 72k. On the Z340 hidden-transform fixture,
  the cap-aware wide structural run scored 100,000 generated candidates,
  deduped to 80,710 unique token orders, and completed in 155.6s. The hot path
  now avoids reapplying each transform to the ciphertext after deriving the
  position order and computes token-order metrics directly from that order.
  A follow-up breadth pass added a second local repair layer to route-based
  programs. The wide generator can now honor an explicit 600k cap on the Z340
  fixture, putting the structural search scale in the same rough order as the
  historical Z340 search. Large wide screens now switch to a NumPy-backed
  position-only metric pass (`fast_structural_metrics: true`) and keep
  high-cardinality generated-family names compact in summary counters. A
  600k structural-only Z340 run completed in 173.0s, generated 600,000
  candidates, deduped to 504,569 unique token orders, and selected
  `program_0853_d5_tail_repair_pack` as the top structural candidate. This is
  still only candidate generation and structural scoring; solver-backed
  validation must remain finalist-only.
- Frontier synthetic cases can set `hide_transform_pipeline_from_solver: true`
  to generate transformed ciphertext while withholding the pipeline from the
  automated runner. This creates honest candidate-search calibration rows
  without moving data into the external benchmark repository.
- Transform candidates are now validated as position permutations before
  structural scoring or solver-backed search. Invalid candidates, such as a
  non-permutation `NDownMAcross` probe on the 88-symbol Dorabella grid, are
  recorded as rejected/skipped candidates instead of crashing the tool.

Recommended next step:

- Use `--transform-search-profile fast` for routine hidden-route regression
  checks. Under `broad` and `wide`, continue generalizing the program grammar
  beyond the Z340-shaped first slice: add more route families, split-grid
  programs, periodic row/column operations, and non-Z340 local repair packs,
  using diagnostics to decide whether new families are robust candidates or
  just unstable false positives. The next scale milestone is to make the wide
  layer stream/report hundreds of thousands of structural candidates with
  better scoring and smarter finalist promotion, while continuing to avoid
  materializing every transformed token order in memory.
