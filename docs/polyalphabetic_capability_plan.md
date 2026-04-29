# Polyalphabetic Capability Plan

Status: planning document. Decipher currently has useful cipher-identification
signals for Vigenere-like periodic ciphers and an initial bounded
Vigenere-family implementation, but it does not yet have a complete
first-class polyalphabetic subsystem. Treat this plan as the starting point
for continued capability work.

Implementation note, April 2026: the first slice is now in place. Decipher has
an A-Z Vigenere-family screen in `src/analysis/polyalphabetic.py`, explicit
benchmark metadata routing for Vigenere/Beaufort/Gronsfeld-style cases, and
agent tools for cipher-ID observation, periodic-IC inspection, hypothesis
branch creation/rejection, and installing periodic candidates as
mode-tagged branches. The agent can also inspect phase-local streams with
`decode_show_phases` and manually set or adjust periodic keys with
`act_set_periodic_key`, `act_set_periodic_shift`, and
`act_adjust_periodic_shift`. This is still a bounded diagnostic/solver slice,
not a complete polyalphabetic subsystem.

## Goal

Add practical support for periodic polyalphabetic ciphers while keeping the
architecture honest about cipher type. A Vigenere-family solve should not be
forced through substitution-branch key mechanics, and an agent should not be
given a toolbox whose mutations are semantically wrong for the active cipher
hypothesis.

Initial target families:

- Vigenere and Caesar-shift periodic substitution.
- Beaufort and variant Beaufort.
- Gronsfeld-style numeric periodic keys.
- Autokey/running-key diagnostics as later extensions.
- Mixed or disguised forms such as substitution plus periodic shifts, only
  after plain periodic ciphers are stable.

Non-goals for the first implementation:

- Solving arbitrary rotor/machine ciphers.
- General nomenclator or codebook hybrids.
- Claiming open-ended cipher identification is solved.

## Milestone 0: Cipher Modes And Key Models

Introduce explicit cipher modes so tools, key state, search moves, and
artifacts match the hypothesized cipher family.

### Mode Registry

Add a small registry describing supported modes:

- `simple_substitution`
- `homophonic_substitution`
- `transposition_homophonic`
- `periodic_polyalphabetic`
- `unknown`

Each mode should declare:

- Key representation.
- Valid mutation tools.
- Valid search tools.
- Decode semantics.
- Score signals that are meaningful.
- Artifact fields required for reproducibility.

### Key Types

Current substitution keys are `dict[cipher_symbol_id, plaintext_symbol_id]`.
That is wrong for Vigenere-like ciphers, where the key is a periodic sequence
of shifts or alphabets.

Add typed key state, likely as dataclasses:

- `SubstitutionKey`: cipher symbol -> plaintext symbol.
- `HomophonicKey`: cipher symbol -> plaintext symbol, many-to-one allowed.
- `TransformPipelineKey`: token order overlay plus downstream key.
- `PeriodicShiftKey`: period, alphabet, shift sequence, direction, variant.
- `PeriodicAlphabetKey`: period plus one substitution alphabet per phase.
- `CompositeKey`: ordered pipeline of transform/key stages.

The workspace should be able to hold branch mode metadata and a mode-specific
key object. Legacy `branch.key` can remain for substitution branches, but new
mode-specific keys should not be crammed into the same dict.

### Mode-Specific Branches

Branches should record:

- `cipher_mode`
- `mode_confidence`
- `key_type`
- `key_state`
- `decode_pipeline`
- `hypothesis_notes`

Tools should reject operations whose semantics do not match the branch mode.
For example, `act_set_mapping` should not mutate a `PeriodicShiftKey`; a
polyalphabetic branch needs tools like `act_set_period_shift`.

## Milestone 1: Detection And Preflight Diagnostics

Build a detection layer that can run automatically in preflight and can also
be invoked by the agent.

We already have useful pieces in `src/analysis/cipher_id.py`:

- Index of coincidence.
- Entropy/frequency flatness.
- Periodic IC by candidate period.
- Kasiski-style repeated n-gram spacing support.
- Suspicion score for `polyalphabetic_vigenere`.

Next steps:

- Promote this into a stable preflight report used by benchmark/crack runs.
- Add artifacts fields for:
  - raw IC
  - expected-language IC
  - entropy
  - periodic IC table
  - best periodic IC periods
  - Kasiski period support
  - repeated sequence evidence
  - top cipher-family hypotheses with scores
- Add CLI output that explains the diagnosis in human terms:
  - "Raw IC is depressed, but every 5th stream recovers English-like IC."
  - "Kasiski spacings support periods 5 and 10."
  - "Homophonic and Vigenere remain ambiguous because symbol inventory is high."

### Agent Tools

Expose detection as low-cost tools:

- `observe_cipher_id`
  - Returns ranked cipher hypotheses.
  - Includes evidence and cautions.
- `observe_periodic_ic`
  - Shows period table and per-phase IC for candidate periods.
- `observe_kasiski`
  - Shows repeated sequences and spacing gcds.
  - Implemented as a detailed repeated n-gram/spacing/factor report.
- `observe_phase_frequency`
  - For a candidate period, shows frequency profiles for each phase.
  - Implemented for clean A-Z periodic candidates.
- `observe_periodic_shift_candidates`
  - For each candidate period, proposes likely Caesar shifts per phase.
  - Implemented with phase-local chi-squared candidates for Vigenere-family
    variants.

### Routing Policy

Preflight should not automatically force a polyalphabetic solve unless evidence
is strong. It should:

1. Record hypotheses.
2. Choose a conservative automated route when metadata is explicit.
3. Otherwise expose the hypotheses to the agent and suggest next tools.

Ambiguous cases should remain explicit:

- Low IC plus large symbol inventory may be homophonic, polyalphabetic, or
  noisy/transcriptional.
- Low IC plus strong period evidence favors polyalphabetic.
- Normal IC with bad language score may favor transposition or mixed systems.

## Milestone 2: Automated Polyalphabetic Solvers

Start with clean Vigenere-family solvers before mixed systems.

### Vigenere/Beaufort Period Search

Implement an automated route:

1. Candidate period generation:
   - Periodic IC peaks.
   - Kasiski spacing support.
   - Small exhaustive periods, e.g. 1-20, then configurable up to 40 or 60.
2. Per-period shift initialization:
   - Chi-squared against language monogram frequencies per phase.
   - Cross-correlation scoring over all shifts.
3. Joint scoring:
   - Decode with candidate key.
   - Score with n-gram model and dictionary/segmentation where applicable.
4. Local refinement:
   - Hill-climb over phase shifts.
   - Simulated annealing over shifts for longer keys.
   - Optional n-gram rescoring using continuous models.
5. Variant search:
   - Vigenere.
   - Beaufort.
   - Variant Beaufort.
   - Gronsfeld constrained shifts.

Output:

- Best key and period.
- Variant.
- Decoded text.
- Per-period candidate table.
- Competing near-tie keys.
- Stability metrics across seeds.

### Running-Key And Autokey Later

Running-key and autokey need different tools:

- Candidate primer/key source hypotheses.
- Known-plaintext/crib support.
- Language-model scoring for both plaintext and key stream.
- Alignment and dictionary tools for plausible key text.

Do not mix these into the first Vigenere solver except as diagnostics.

## Milestone 3: Agentic Polyalphabetic Tools

The agent should reason with period/key hypotheses, not individual cipher
symbol mappings.

Add tools:

- `search_periodic_polyalphabetic`
  - Runs automated period/variant search and installs candidate branches.
- `act_set_period`
  - Creates or changes a periodic branch's period.
- `act_set_period_shift`
  - Sets one phase shift. Implemented as `act_set_periodic_shift`.
- `act_adjust_period_shift`
  - Increment/decrement a phase shift and preview changed text. Implemented
    as `act_adjust_periodic_shift`.
- `act_set_periodic_key`
  - Sets the whole key, e.g. `LEMON` or shift list `[11,4,12,14,13]`.
- `decode_show_phases`
  - Shows plaintext grouped by phase with phase-local frequencies. Implemented.
- `workspace_compare_periodic_keys`
  - Compares two periodic branches by period, variant, key, decoded text, and
    internal scores.
- `act_rate_periodic_candidate`
  - Records the agent's contextual readability judgment, analogous to
    transform finalist rating.

Agent prompt changes:

- If polyalphabetic evidence is strong, do not use substitution repair tools.
- First decide period/variant/key family.
- Use phrase-level reading to adjust phase shifts only after the automated
  candidate is near-readable.
- Treat isolated words in a bad periodic decode as weak evidence, just as in
  no-boundary homophonic search.

## Milestone 4: Benchmarks And Synthetic Ladder

Create a polyalphabetic fixture packet before working on famous unsolved cases.
The first packet now lives at `frontier/polyalphabetic_ladder.jsonl` and is
backed by the same `decipher testgen` machinery as ad hoc synthetic cases.

### Synthetic Ladder

Generate local Decipher fixtures:

- Vigenere, known period, clean no-boundary text. Implemented in
  `frontier/polyalphabetic_ladder.jsonl`.
- Vigenere, hidden/random period, clean text. Supported on demand with
  `decipher testgen --cipher-system vigenere --poly-period N`, but not yet
  calibrated as a frontier row.
- Vigenere, no word boundaries. Implemented.
- Beaufort and variant Beaufort. Implemented in the ladder packet.
- Gronsfeld constrained keys. Implemented in the ladder packet.
- Short underdetermined cases.
- Noisy/transcription-error cases.
- Mixed substitution + Vigenere as an advanced row.

Each synthetic case should record:

- True mode.
- Period.
- Variant.
- Key.
- Whether metadata is exposed to the solver.
- Expected automated and agentic behavior.

### Real/Solved Calibration

Curate solved examples:

- Kryptos K1/K2 as recognizable public calibration cases, with context tiers
  that avoid leaking solution unless explicitly enabled.
- AZdecrypt Vigenere examples as solver-development fixtures.
- zkdecrypto `runningkey.test.txt` as a future hard/qualitative case.

### Evaluation

Track:

- Correct period.
- Correct variant.
- Key edit distance.
- Character accuracy.
- Word/readability score.
- Whether agent used mode-appropriate tools.
- Whether declaration happened from coherent text or word islands.

## Milestone 5: Integration With Automated/Agentic Runs

### Automated Runner

Extend automated routing:

- If metadata says `periodic_polyalphabetic`, run the polyalphabetic solver.
- If metadata is unknown but preflight strongly favors periodic
  polyalphabetic, optionally run a bounded screen.
- If evidence is mixed, record a capability/hypothesis report rather than
  silently using the wrong solver.

Artifacts should include:

- `cipher_id_report`
- `cipher_mode`
- `periodic_search`
- `selected_period`
- `selected_variant`
- `periodic_key`
- `candidate_periods`
- `agent_readability_judgments` where applicable

### Agent Loop

Initial context should include the cipher-id preflight summary. The agent
should be nudged to choose a mode before applying mutating tools:

1. Inspect candidate modes.
2. Create or choose a mode-specific branch.
3. Run mode-specific search.
4. Read/repair with mode-appropriate actuators.
5. Declare with mode, key, and confidence.

## Risks And Open Questions

- Short ciphers may be underdetermined even when the period is detectable.
- Homophonic and polyalphabetic ciphers can both flatten frequency
  distributions; detection must report ambiguity.
- Many famous polyalphabetic examples have public solutions, so context tiers
  must avoid accidental leakage.
- Mixed substitution+Vigenere may require composite key search that is much
  harder than plain Vigenere.
- Running-key systems are closer to language-model/crib search than ordinary
  periodic shifts.

## First Implementation Slice

Recommended first branch:

1. [x] Add `cipher_mode` metadata to workspace branches without changing existing
   substitution behavior.
2. [x] Add `observe_cipher_id` and `observe_periodic_ic` as agent tools backed by
   existing `analysis.cipher_id`.
3. [x] Add a simple automated Vigenere solver for known alphabet A-Z:
   period search -> per-phase chi-squared shifts -> n-gram refinement.
4. [x] Add a synthetic Vigenere-family ladder and on-demand testgen support.
5. [x] Import solved Kryptos K1/K2 calibration rows and support
   known-parameter keyed Vigenere replay.
6. [x] Add `search_periodic_polyalphabetic` to install candidate periodic branches.

Only after that should we attempt unknown-key Kryptos, running-key examples, or
general periodic substitution.

Current Kryptos-specific state:

- The benchmark repo now carries K1/K2 solved calibration records with
  solution-bearing `known_cipher_parameters`.
- Decipher can replay those parameters with a `PeriodicAlphabetKey` model:
  one shared keyed alphabet plus per-position shifts from the periodic key.
- This is deliberately labeled `keyed_vigenere_known_replay` in artifacts. It
  verifies mechanics, tableau convention, and unknown-symbol handling; it is
  not a claim that Decipher has recovered the Kryptos keys from ciphertext.
- Decipher can also recover the periodic key over supplied candidate keyed
  alphabets/tableau keywords via `keyed_vigenere_periodic_key_search`
  (`DECIPHER_KEYED_VIGENERE_MODE=search`). With the `KRYPTOS` keyed alphabet
  supplied, this recovers K2's `ABSCISSA` key and plaintext.
- First-pass keyword-tableau enumeration is available as
  `DECIPHER_KEYED_VIGENERE_MODE=tableau_search`. This mode tests the standard
  A-Z tableau first, then keyword-derived tableaux from
  `DECIPHER_KEYED_VIGENERE_TABLEAU_KEYWORDS`. With `KRYPTOS` in that explicit
  candidate list, K2 recovers as a genuine candidate-tableau selection plus
  periodic-key recovery.
- Experimental shared-tableau mutation search is available as
  `DECIPHER_KEYED_VIGENERE_MODE=alphabet_anneal`. It mutates the candidate
  alphabet with swaps, moves, and reversals, re-optimizes phase shifts after
  each mutation, and scores the whole plaintext. This is the first
  frequency/search scaffold beyond keyword lists, but broad blind recovery
  from plain A-Z is not reliable yet.
- K1 is currently too short for the plain n-gram/chi-square scorer to recover
  `PALIMPSEST` reliably without crib/context help; false English-ish basins
  can outrank the true text.
- K2's `?` source tokens are skipped and do not advance the periodic key.

Remaining keyed-Vigenere work:

- Larger and better-prioritized keyword candidate generation.
- Better mutation proposals for shared-alphabet search: phase-frequency
  constraints, pairwise offset constraints, and beam/anneal hybrids.
- Crib-aware scoring for short famous ciphers.
- Agent tools to try a tableau keyword, inspect the resulting periodic branch,
  and compare keyed-Vigenere hypotheses against ordinary Vigenere branches.
