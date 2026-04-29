# Unknown Cipher Agent Plan

Status: planning document. This is the general agent workflow for ciphers
whose type is unknown, ambiguous, or only partially described by benchmark
metadata. It coordinates the more specific capability plans, especially
`docs/polyalphabetic_capability_plan.md` and
`docs/transposition_homophonic_capability.md`.

Implementation note, April 2026: the first shared branch/tool conventions are
partially implemented. The agent can now call `observe_cipher_id`,
`observe_periodic_ic`, `workspace_create_hypothesis_branch`,
`workspace_reject_hypothesis`, `workspace_hypothesis_cards`, and
`search_periodic_polyalphabetic`. Periodic candidates are represented as
mode-tagged metadata branches rather than fake substitution keys.
Automated preflight artifacts now also carry `cipher_id_report`, and the
LLM-safe preflight summary includes a compact cipher-type fingerprint.
For periodic-polyalphabetic branches, the agent can inspect and repair the
mode-specific key state with `decode_show_phases`,
`act_set_periodic_key`, `act_set_periodic_shift`, and
`act_adjust_periodic_shift`.

## Goal

Give the agent a disciplined way to approach an unknown cipher:

1. Inspect cheap statistical evidence.
2. Form explicit cipher-type hypotheses.
3. Create mode-specific hypothesis branches.
4. Run the right automated and diagnostic tools for each hypothesis.
5. Compare hypotheses by both cryptanalytic scores and contextual readability.
6. Switch hypotheses when the evidence says the current path is wrong.

The current agent can already branch, run automated solvers, inspect context,
and compare outputs. The missing piece is a first-class "cipher mode" layer
that keeps the agent from treating every partial output as a damaged
substitution solve.

## Relationship To Other Plans

This plan is the router and orchestration layer.

- `polyalphabetic_capability_plan.md` defines the periodic-polyalphabetic
  child track: Vigenere, Beaufort, Gronsfeld, period detection, and
  mode-specific period/key tools.
- `transposition_homophonic_capability.md` defines the
  transposition-plus-homophonic child track: transform screens, finalist
  promotion, homophonic solving, and branch review.
- Future child plans should cover fractionation, nomenclators, polygraphic
  substitution, null/noise support, and geometric route/grille systems.

The mode registry, branch metadata, artifact shape, and tool-gating rules
should be shared across these plans rather than redefined separately.

## Current State

Useful pieces already exist:

- `src/analysis/cipher_id.py` computes cheap cipher fingerprints:
  - token and symbol counts
  - IC and delta from language reference IC
  - entropy/frequency flatness
  - periodic IC
  - Kasiski-style repeated n-gram spacing evidence
  - doubled-digraph rate
  - suspicion scores for several cipher families
- `tests/test_cipher_id.py` covers synthetic monoalphabetic, homophonic,
  Vigenere-like, and Playfair-like signals.
- `Workspace` branches already support metadata, tags, transform pipelines,
  custom token order, and custom word-boundary overlays.
- The agent already has branch cards, automated preflight, context tools,
  transform-finalist review, and declaration guards.

Important gaps:

- Cipher-ID output is not yet a standard preflight artifact for every run.
- The agent does not yet have dedicated `observe_cipher_id` or
  `observe_cipher_hypotheses` tools.
- Branch state is still fundamentally substitution-shaped:
  `dict[cipher_symbol_id, plaintext_symbol_id]`.
- The agent prompt is still mostly one-size-fits-all rather than selected by
  cipher-mode hypothesis.
- We do not yet record a structured history of rejected cipher-type
  hypotheses.

## Milestone 0: Shared Cipher Modes

Introduce explicit cipher modes as the unit of reasoning for unknown-cipher
work.

Initial mode registry:

- `unknown`
- `simple_substitution`
- `homophonic_substitution`
- `transposition`
- `transposition_homophonic`
- `periodic_polyalphabetic`
- `polygraphic_substitution`
- `fractionation_transposition`
- `nomenclator_codebook`
- `nulls_or_noisy_transcription`

Each mode should declare:

- Key or state representation.
- Decode semantics.
- Valid observation tools.
- Valid search tools.
- Valid mutation/repair tools.
- Meaningful scoring signals.
- Artifact fields needed for reproducibility.
- Human-readable "why this mode fits" and "why this mode is suspicious"
  summaries.

This registry must agree with the polyalphabetic plan's mode/key model. In
particular, periodic polyalphabetic ciphers need `PeriodicShiftKey` or
`PeriodicAlphabetKey`, not a fake substitution mapping.

## Milestone 1: Mode-Specific Branch State

Extend workspace branches so they can represent a cipher hypothesis, not just
a partial substitution key.

Branches should eventually record:

- `cipher_mode`
- `mode_confidence`
- `mode_evidence`
- `mode_status`: `active`, `promising`, `rejected`, `superseded`, `declared`
- `key_type`
- `key_state`
- `decode_pipeline`
- `source_hypothesis_branch`
- `hypothesis_notes`
- `rejection_reason`
- `agent_readability_score`
- `agent_readability_rationale`

Backward compatibility:

- Keep `branch.key` for existing substitution and homophonic branches.
- Use `branch.metadata["cipher_mode"]` as an initial bridge.
- Add typed `key_state` only when a non-substitution mode needs it.
- Do not force old branch snapshots or artifacts to migrate immediately.

This keeps current simple/homophonic work stable while opening the door for
periodic keys, transform pipelines, digraph keys, route hypotheses, and
fractionation states.

## Milestone 2: Preflight For Unknown Ciphers

Every unknown or under-described run should receive a cheap cipher-ID
preflight before any expensive solving tool runs.

Preflight should compute and record:

- Basic shape:
  - token count
  - unique symbol count
  - declared alphabet size
  - word-boundary structure
  - no-boundary vs grouped text
- IC evidence:
  - raw IC
  - language reference IC
  - IC delta
  - interpretation
- Frequency evidence:
  - normalized entropy
  - chi-squared or similar peakedness/flatness statistic
  - top symbol frequencies
- Periodic evidence:
  - periodic IC table
  - best periodic periods
  - Kasiski spacing support
  - phase-local frequency summaries for promising periods
- Digraph/polygraphic evidence:
  - doubled-digraph rate
  - even token count
  - pair structure hints
- Transform/order evidence:
  - whether language-like symbol statistics are present but adjacency looks
    poor
  - whether structural transform suspicion is high enough to justify a screen
- Context evidence:
  - known language, source family, date, medium, related records, and any
    explicit benchmark cipher-system metadata
- Ranked mode hypotheses:
  - score
  - confidence label
  - supporting evidence
  - counter-evidence
  - recommended next diagnostic tool

The report should be stored in artifacts as `cipher_id_report` and summarized
to the agent before the first turn.

## Milestone 3: Agent Observation Tools

Add low-cost tools that expose the preflight and allow deeper targeted
diagnosis.

Core tools:

- `observe_cipher_id`
  - Return the full ranked mode-hypothesis report.
  - Include evidence, cautions, and recommended next tools.
- `observe_cipher_shape`
  - Token count, symbol inventory, repeated symbols, boundary structure,
    line/grid metadata when present.
- `observe_periodic_ic`
  - Period table and per-phase IC.
- `observe_kasiski`
  - Repeated n-grams, spacing gcds, and candidate periods.
- `observe_phase_frequency`
  - Frequency profiles for a candidate period.
- `observe_transform_suspicion`
  - Current transform/order suspicion, including known grid dimensions when
    available.
- `observe_digraph_structure`
  - Doubled digraphs, pair constraints, Playfair/polygraphic hints.
- `observe_null_noise_suspicion`
  - Lightweight evidence for skipped symbols, nulls, OCR uncertainty, or
    inconsistent symbol normalization.

The agent should be able to call these repeatedly after creating transformed
or filtered branches. The same cipher can look monoalphabetic after null
removal, periodic after a transform, or homophonic after normalization.

## Milestone 4: Hypothesis Branch Workflow

Add explicit tools for managing cipher-type hypotheses.

Suggested tools:

- `workspace_create_hypothesis_branch`
  - Inputs: `source_branch`, `new_branch`, `cipher_mode`, `rationale`,
    optional `mode_confidence`.
  - Creates a branch tagged with the selected mode.
- `workspace_set_cipher_mode`
  - Updates mode metadata for a branch without changing the key.
  - Should require a rationale.
- `workspace_reject_hypothesis`
  - Marks a branch/mode as rejected and records why.
- `workspace_hypothesis_cards`
  - Shows all active/rejected mode branches, their evidence, scores, best
    preview text, and agent readability notes.
- `workspace_compare_hypotheses`
  - Compares branches across modes without pretending their scores are all
    commensurate.

The workflow should look like:

1. Start from `main`.
2. Read `cipher_id_report`.
3. Create one or more mode branches, e.g. `hyp_homophonic`,
   `hyp_period5_vigenere`, `hyp_transform_homophonic`.
4. Run mode-specific searches on each.
5. Rate/read results.
6. Reject weak branches with reasons.
7. Escalate promising branches.
8. Declare only when a branch is coherent enough, or explicitly report a
   qualitative unresolved hypothesis.

## Milestone 5: Mode-Specific Tool Gating

The agent should not receive a giant undifferentiated toolbox by default.

Tool visibility should have layers:

- Always available:
  - inspect context
  - observe cipher ID
  - create/switch/reject hypothesis branch
  - list branch/hypothesis cards
  - declare/report unresolved status
- Mode-specific foreground tools:
  - substitution: mapping, swaps, pattern/dictionary tools
  - homophonic: homophonic anneal, homophone diagnostics, no-boundary repair
  - transposition+homophonic: transform suspicion/search/finalist tools
  - periodic polyalphabetic: period/key search and phase-shift tools
  - Playfair/polygraphic: digraph grid/key tools
  - fractionation: coordinate/fractionation and route tools
- Escape tools:
  - switch mode
  - ask for broader hypothesis screen
  - mark current basin as bad

Implementation can start softly: keep tools technically callable, but make the
prompt and tool docs foreground only the active mode's tools. Later, executor
gating can reject semantically wrong tools with clear errors.

## Milestone 6: Automated Routing Policy

Automated mode should be conservative and auditable.

If benchmark metadata explicitly states a cipher system:

- Run the matching solver if supported.
- If unsupported, return a clear capability-gap artifact.

If metadata is missing or weak:

1. Compute `cipher_id_report`.
2. If one mode is strongly supported and has a cheap solver, run a bounded
   screen for that mode.
3. If multiple modes are plausible, run only cheap diagnostic screens unless
   the caller requested `--unknown-cipher-search`.
4. Record all skipped/unsupported plausible modes.
5. Never silently treat an ambiguous cipher as simple substitution just because
   that solver exists.

Suggested CLI controls:

- `--cipher-mode auto|unknown|simple_substitution|homophonic_substitution|...`
- `--unknown-cipher-search none|diagnose|screen|broad`
- `--max-mode-hypotheses N`
- `--mode-search-budget screen|full`

## Milestone 7: Agentic Unknown-Cipher Policy

The agent should be told to spend early effort on hypothesis selection unless
the preflight is already decisive.

Prompt policy:

- Name the current top hypotheses and why they are plausible.
- If no branch is coherent, do not repair word islands.
- If a branch has isolated words but no coherent clauses, treat it as a bad
  basin and try another mode/search.
- If the active mode's tools are exhausted, switch hypothesis rather than
  declaring a weak partial.
- When declaring an unresolved result, state:
  - best current hypothesis
  - hypotheses tested
  - hypotheses not yet tested
  - which additional search would be most valuable
  - whether further iterations are likely to help

This policy is consistent with recent Z340 lessons: the useful behavior was
not "repair the first wordy output"; it was recognizing a bad basin, escalating
to transform+homophonic search, and then rating coherent finalists.

## Milestone 8: Scoring And Comparison Across Modes

Scores from different modes are not automatically comparable.

For example:

- A homophonic anneal score measures language-model fit after symbol mapping.
- A transform structural score measures order plausibility before solving.
- A periodic IC score measures key-period evidence, not plaintext quality.
- A Playfair digraph score may be meaningful only after pair parsing.

Cross-mode comparison should use a structured card:

- mode
- branch
- tool path used
- candidate score(s)
- preview text
- agent readability score
- coherent phrase/clause evidence
- context fit
- known failure signs
- next useful action

The agent's contextual readability rating should be first-class, not buried in
free text. It is often the best signal once a candidate produces recognizable
language.

## Milestone 9: Artifacts And Audit Trail

Artifacts should make unknown-cipher reasoning inspectable.

Add fields such as:

- `cipher_id_report`
- `mode_hypotheses`
- `selected_cipher_mode`
- `hypothesis_branches`
- `rejected_hypotheses`
- `mode_switches`
- `diagnostic_tools_run`
- `unsupported_plausible_modes`
- `declaration_mode`
- `unresolved_status_summary`

For each hypothesis branch, record:

- parent branch
- mode
- rationale
- tool sequence
- best preview
- solver/search settings
- rating and rationale
- rejection or declaration notes

This lets us measure whether the agent is actually exploring or simply
declaring the first automated output.

## Milestone 10: Tests

Unit tests:

- `cipher_id_report` serialization.
- Mode registry entries and valid tool declarations.
- Branch metadata round-trips for mode hypotheses.
- Rejecting semantically wrong tool calls once gating is enabled.

Synthetic tests:

- Monoalphabetic case routes to substitution.
- Homophonic case routes to homophonic.
- Periodic Vigenere case routes to periodic polyalphabetic.
- Playfair-like case is flagged as digraphic/polygraphic.
- Ambiguous short text remains `unknown` with multiple hypotheses.

Agent fake-provider tests:

- Agent reads cipher-ID report and creates a matching hypothesis branch.
- Agent rejects a bad-basin branch and tries a second mode.
- Agent uses periodic tools for a Vigenere-like case rather than substitution
  repair.
- Agent uses transform+homophonic tools for a Z340-like case.
- Agent declares unresolved with a useful plan when no supported mode solves.

Live/opt-in tests:

- Neutral Z340 context probe.
- Dorabella-style qualitative unresolved run.
- Scorpion-style uncertain transcription run.
- Solved Vigenere/Kryptos calibration once curated.

## First Implementation Slice

Recommended first branch:

1. [x] Promote `analysis.cipher_id` into a standard `cipher_id_report` attached to
   automated artifacts. Agent initial-context injection remains planned, but
   the report is available on demand through `observe_cipher_id`.
2. [x] Add `observe_cipher_id` and `observe_cipher_shape`.
3. [x] Add `cipher_mode`, `mode_evidence`, and `mode_status` metadata conventions
   for workspace branches.
4. [x] Add `workspace_create_hypothesis_branch`,
   `workspace_reject_hypothesis`, and `workspace_hypothesis_cards`.
5. [x] Update the prompt so unknown-cipher runs explicitly start with hypothesis
   selection and bad-basin recognition.
6. [ ] Add broader fake-provider tests for:
   - selecting a mode
   - rejecting a mode
   - switching modes
   - not using substitution repair on a branch tagged
     `periodic_polyalphabetic`
7. [x] Add the first new child solver capability, the bounded
   Vigenere-family slice from `polyalphabetic_capability_plan.md`.

## Open Questions

- Should automated preflight run one bounded screen per plausible mode, or
  should it only diagnose unless the caller opts in?
- How aggressive should executor-level tool gating be before the mode registry
  is mature?
- Should mode-specific prompts be selected dynamically inside a run, or should
  the system prompt stay broad while tool descriptions become mode-specific?
- How should we compare "no solution but correct cipher-type diagnosis" against
  a wrong-mode partial plaintext?
- What is the minimum artifact schema change that preserves backward
  compatibility with existing runs?
