# Unknown Cipher Investigator Mode

Status: planning document. This is a higher-level companion to
`docs/unknown_cipher_agent_plan.md`.

The existing unknown-cipher plan mostly describes routing, cipher modes,
branch metadata, and mode-specific tool menus. Investigator Mode describes a
different agent posture for serious unsolved or ambiguous ciphers: the agent
should behave less like a solver choosing from a menu and more like a
cryptanalytic researcher building and testing theories.

## Motivation

Recent runs show a recurring failure pattern. When a cipher is not covered by
one of Decipher's mature solver families, the agent tends to:

1. Try a few familiar high-powered tools.
2. Interpret weak word islands as damaged-but-promising plaintext.
3. Spend iterations repairing a bad basin.
4. Declare unsolved, or declare too early, without leaving a useful research
   trail.

That is acceptable for ordinary benchmark automation, but it is not a
principled approach to unsolved ciphers. For an unknown cipher, a good outcome
may be a negative or partial research result: which families were tested,
which observations mattered, which hypotheses were weakened, and what the
next most informative experiment should be.

### Case Study: Feynman #2 Minimal Context

Artifact analysis for a GPT-5.4 minimal-context run on Feynman #2
(`2ff2c634eaa4`) showed a representative failure:

- The agent opened plausible monoalphabetic and Vigenere-like hypotheses.
- It spent most of the runtime on wrong-family searches, especially a broad
  Quagmire III search.
- It saw English-like but globally incoherent word islands and drifted into
  local repair rather than asking whether order/segmentation/word-level
  mechanics were wrong.
- It requested a missing tool and noted that transform-aware search would
  likely be valuable, but did not pivot early enough.
- It ultimately auto-declared a poor branch even though it had not recovered a
  readable plaintext.

This is exactly the kind of failure Investigator Mode should prevent. The
lesson is not merely "add a Feynman solver"; the broader lesson is that
English-like gibberish with no paraphrasable clause is evidence. It should
trigger a diagnostic fork: bad local substitution basin, wrong token order,
null/noise layer, word-boundary/segmentation issue, alternating-key mechanism,
or an unsupported custom construction.

## Goal

Investigator Mode should produce a structured research process:

1. Run a broad statistical fingerprint before solving.
2. Maintain an explicit evidence notebook.
3. Form several competing hypotheses.
4. Choose discriminating tests, not just favorite solvers.
5. Feed failed solves back into diagnosis.
6. Invite creative mechanism proposals when standard families do not explain
   the evidence.
7. End with a research note, even when there is no decipherment.
8. Treat research notes as future input, not just final output.

The agent should be rewarded for saying: "I do not have a plaintext, but here
is what the cipher appears not to be, here are the live hypotheses, and here
is the next experiment that would move the most uncertainty."

## Non-Goals

- Do not replace specialized solvers. Investigator Mode decides when and why
  to use them.
- Do not let the LLM see benchmark ground truth. All evidence must be
  ciphertext-derived or context-tier-derived.
- Do not make every run expensive. This mode is for unknown, disputed, or
  qualitative cases, not routine solved benchmark rows.
- Do not force a single top hypothesis when evidence is ambiguous.

## Core Concepts

### Evidence Notebook

Every investigator run should maintain a structured notebook in artifact state.

Notebook entries should include:

- observation name
- tool or source that produced it
- raw measurement
- plain-language interpretation
- cipher families supported
- cipher families weakened
- confidence and caveats
- follow-up tests suggested

Example:

```json
{
  "observation": "periodic_ic",
  "measurement": {"period_8_ic": 0.061, "period_9_ic": 0.039},
  "interpretation": "Period 8 has a strong phase-local monoalphabetic signal.",
  "supports": ["periodic_polyalphabetic", "quagmire_like"],
  "weakens": ["simple_substitution", "pure_transposition"],
  "next_tests": ["observe_phase_frequency(period=8)", "search_vigenere_family(period=8)"]
}
```

### Hypothesis Board

The agent should keep a board of competing explanations.

Each hypothesis card should include:

- cipher family or custom mechanism
- confidence
- supporting evidence
- counterevidence
- tests already run
- next discriminating test
- status: `live`, `favored`, `weakened`, `rejected`, `needs_new_tool`
- whether a solve attempt has been made
- what the solve attempt taught us

Hypotheses should be comparable. A homophonic hypothesis with a bad output
should not simply become "homophonic failed"; it should create evidence such
as "language-like word islands without coherent syntax, possible wrong order,
nulls, mixed system, or false basin."

### Experiment Queue

The agent should maintain a prioritized list of experiments. Experiments can be
statistical, structural, or solver-backed.

Each experiment should state:

- question it answers
- expected cost
- expected information gain
- hypotheses it distinguishes
- what result would strengthen or weaken each hypothesis

This is how the agent avoids banging away at known solvers when a cheap
diagnostic could disqualify the entire family.

### Bad-Basin And Budget Guards

Investigator Mode should recognize when a high-scoring candidate is not a real
reading basin.

Red flags:

- dictionary-ish words but no coherent clause
- repeated local repairs that damage other text
- high n-gram score with no human-paraphrasable sentence
- multiple solver families producing similar word islands
- zero or near-zero word continuity after segmentation

When these appear, the agent should stop local repair and ask what structural
assumption is wrong.

Initial guardrails:

- Do not run a broad Quagmire or periodic-polyalphabetic search unless cheap
  periodic evidence is strong enough, or a cheap periodic run produced a
  readable multi-word phrase.
- If a branch has strong letter/dictionary signal but no readable clause,
  prioritize transform/order, null/noise, segmentation, and mixed-mechanism
  diagnostics before more local substitution repair.
- Require every expensive family search to name the evidence that justifies
  its budget.
- After an expensive wrong-family search fails, mark the family as weakened
  and avoid repeating it unless a parameter change or new observation is
  recorded.

### Research Notes As Solver Memory

Research notes should be both:

- an output for humans, and
- an input to future solver runs.

An investigator run should be able to resume from prior notes without
repeating the same failed experiments. The note should capture what was tried,
what was learned, what was rejected, and which suggestions remain untested.

Useful note fields:

- current best hypothesis board
- rejected hypotheses with reasons
- experiments already run
- experiments that were proposed but not run
- tool gaps encountered
- promising partial observations
- human suggestions and how they were handled
- caveats about context, language, source reliability, or transcription
- next recommended actions

The agent should use this as a working memory. A later run should be able to
say: "The previous investigation already tested simple substitution,
homophonic, Quagmire, and broad transform screens; I will not repeat those
except to change a named parameter. The open suggestion is to test
word-alternating keys."

### Human Suggestions

Human users should be able to inject suggestions into the investigator
notebook. Suggestions should be treated as hypotheses or experiments, not as
ground truth.

Examples:

- "Try alternating alphabets by word."
- "This might be a book cipher using a religious text."
- "Look for every third character."
- "The line breaks may be meaningful."
- "Do not spend more time on homophonic search unless there is new evidence."

Each suggestion should record:

- source: human/user/context/document
- exact suggestion text
- affected hypotheses
- planned test
- status: `new`, `accepted_for_test`, `tested`, `rejected`, `needs_tool`
- result and rationale after testing

This gives the human a real collaborative role without letting user hints
silently override evidence.

### Creative Mechanism Proposals

After the standard battery, the agent should be explicitly invited to propose
mechanisms that are not yet first-class solver modes.

Examples:

- alternating keys by word or line
- reversing even words
- acrostic or first-letter extraction
- book-cipher or index-cipher behavior
- mixed plaintext languages
- deliberate nulls or spelling games
- shorthand/codebook hybrids
- geometric reading orders tied to layout
- homophonic substitution plus systematic transcription errors

Creative proposals must still be testable. The agent should describe what
statistical footprint the mechanism would leave and what small experiment
would check it.

## Diagnostic Battery

Investigator Mode should run or summarize a standard battery before expensive
solving.

### Shape

- token count
- unique symbol count
- alphabet class: letters, numbers, glyphs, mixed
- line lengths and grid dimensions
- word-boundary availability
- repeated words or repeated groups
- symbol inventory drift by line/page

### Frequency And Entropy

- raw IC and normalized IC
- entropy and top-symbol concentration
- chi-square against language-like distributions where appropriate
- flatness suggesting homophonic substitution
- peakedness suggesting simple substitution or plaintext-like symbols

### Repetition Structure

- repeated n-grams
- spacing/gcd support
- Kasiski-style signals
- doubled symbols and doubled digraphs
- repeated numeric groups for code/book candidates

### Periodicity

- periodic IC table
- phase-local symbol distributions
- phase-local language fit
- candidate periods and counterevidence

### Order And Layout

- adjacency/language mismatch
- line/column effects
- route/transposition suspicion
- grid dimensions and factorization
- whether frequency looks language-like but local n-grams do not

### Nulls, Noise, And Errors

- unusually frequent symbols that may be nulls
- rare symbols that may be errors or nomenclator/codeword markers
- length changes under null masks
- repeated bad basins after substitution/homophonic search

### Numeric And Book-Cipher Signals

- numeric range and gaps
- first/last digit distributions
- Benford and epsilon-Benford tests
- monotone or consecutive runs
- modulo structure
- required key-text length under candidate indexing schemes
- front-loading or skew that may indicate dictionary/book order

### Language And Script Uncertainty

- avoid assuming English when language is unknown
- test language-agnostic structure first
- record when a language model is provisional
- distinguish manuscript provenance from plaintext language

## Agent Workflow

### Phase 1: Briefing

The run begins with:

- target context tier
- cipher shape summary
- fingerprint summary
- initial hypothesis board
- initial experiment queue

The agent is told not to solve yet unless a trivial high-confidence route is
available.

### Phase 2: Evidence Gathering

The agent runs cheap diagnostics until the hypothesis board has at least:

- one favored hypothesis
- one plausible alternative
- explicit counterevidence for at least one common family
- a known next discriminating experiment

### Phase 3: Bounded Solve Attempts

The agent may run solver-backed searches, but each attempt must name the
hypothesis being tested and what result would count as success.

Bad basins should update the notebook. For example:

- readable fragments but no coherent syntax
- high score but nonsensical text
- strong local words after transform but no global continuity
- repeated failure across seeds

When a solve attempt yields English-like but unreadable text, the next
experiment should usually be a discriminating structural test, not another
round of local word repair.

### Phase 4: Creative Theory Pass

If standard hypotheses do not explain the observations, the agent runs a
creative theory pass:

1. List unusual observed features.
2. Propose 3-5 mechanisms that could create them.
3. For each mechanism, propose a cheap test.
4. Run the highest-value test if tooling exists.
5. If tooling does not exist, record a tool gap.

This is the phase where Feynman-like word-alternation or other puzzle-specific
mechanisms should have a chance to emerge.

### Phase 5: Research Note

The final output should not be only `solved` or `unsolved`. It should be a
reusable research note.

It should include:

- best current plaintext candidate, if any
- favored hypotheses
- rejected hypotheses and why
- important observations
- whether further iterations would help
- missing tools
- next recommended experiments
- confidence that no overclaim is being made

The note should be stored in both machine-readable artifact form and a
human-readable Markdown sidecar. Future runs should be able to load it with a
`--research-note` or `--resume-investigation` option.

### Phase 6: Resume Or Continue

When a user continues an investigation, the solver should:

1. Load prior notes and artifacts.
2. Summarize what has already been tried.
3. Identify stale, redundant, or superseded experiments.
4. Ask whether any new human suggestions should be added.
5. Continue from the highest-value untested experiment.

It should not repeat expensive searches unless:

- the previous run used a materially different parameter budget,
- new context changes the interpretation,
- a human explicitly asks for confirmation,
- or a regression test needs reproducibility.

## Interaction Modes

### Batch Mode

Batch mode is the current CLI style: run an investigation with a fixed
iteration/tool budget and produce artifacts plus a research note.

This is useful for overnight runs and reproducible benchmark packets.

### Presentation / Live Investigation Mode

Serious unsolved-cipher work may need a more interactive mode. The agent
should be able to present its current state, ask for human judgment, accept
suggestions, and continue without restarting.

A future presentation mode could show:

- current ciphertext and candidate decodes
- hypothesis board
- evidence notebook
- experiment queue
- completed expensive searches
- live tool activity and proof-of-liveness
- research note draft
- chat panel for user suggestions

The user should be able to:

- add a suggestion
- mark a hypothesis as interesting or unlikely
- ask for a specific diagnostic
- approve or deny an expensive search
- request a plain-language explanation of evidence
- pin a note for future runs

This mode should still preserve artifacts and should not expose benchmark
ground truth. It is a collaboration interface, not a shortcut around the
ground-truth firewall.

## First Implementation Steps

### Step 1: Investigator Report Schema

Add an artifact-level `investigator_report` object.

Initial fields:

- `cipher_shape`
- `diagnostic_battery`
- `evidence_notebook`
- `hypothesis_board`
- `experiment_queue`
- `solver_attempts`
- `creative_theory_passes`
- `human_suggestions`
- `research_note_path`
- `prior_note_sources`
- `final_research_note`

This can start as plain dictionaries in artifacts before becoming formal
dataclasses.

### Step 2: One Tool To Summarize Evidence

Add or extend a low-cost tool:

- `workspace_investigator_status`

It should return:

- current hypothesis board
- evidence gathered so far
- families not yet tested
- recommended next diagnostic
- warning if the agent is trying to repair a bad basin too early

This should be callable before declaration and during stalled runs.

### Step 3: Preflight Battery As A Single Packet

Consolidate existing observations into one initial packet:

- `observe_cipher_id`
- IC/entropy/frequency
- periodic/Kasiski
- transform suspicion
- numeric/book-cipher statistics when applicable
- null/noise hints

The packet should be compact enough to send to the agent, but detailed enough
to support later artifact analysis.

### Step 4: Experiment Queue Discipline

Teach the agent prompt that every expensive search should answer a named
question. Add a declaration guard for unknown-cipher runs:

- cannot declare solved/unsolved until it has reviewed the investigator
  status
- cannot spend many turns repairing one branch if the branch is tagged as a
  weak or incoherent basin
- should switch hypotheses or run a discriminating diagnostic when tool output
  contradicts the active mode
- before allowing broad Quagmire/periodic searches, require sufficient
  periodic evidence or a readable cheap-search candidate
- when an output is English-like but not readable, trigger a
  transform/order/null/segmentation/custom-mechanism diagnostic fork

### Step 4a: Endgame Declaration Hygiene

Unknown-cipher runs need a stronger final-budget policy.

At roughly 80-90% of the iteration budget, the loop should inject an endgame
check:

- if no readable clause exists, prepare `meta_declare_unsolved` with a research
  note rather than allowing stale branch fallback
- if a solution is being declared, require branch cards and an explicit branch
  selection
- compare against all existing branches, including automated preflight
  candidates, before selecting a final branch
- record why the declared branch is better than alternatives using
  runtime-visible evidence, not post-hoc ground truth

### Step 5: Synthetic And Famous-But-Controlled Tests

Build a test packet for investigator behavior, not just solver success.

Include:

- Feynman-like alternating-word synthetic ciphers
- no-boundary transposition-only and transposition+homophonic cases
- numeric/book-cipher decoys
- simple substitution with deliberate nulls/noise
- ambiguous Vigenere-vs-Quagmire examples
- famous examples such as Feynman, Beale, D'Agapeyeff, Voynich, and Dorabella
  only as qualitative or context-controlled checks

The scoring should measure whether the agent gathered and interpreted
evidence, not only whether it found plaintext.

### Step 6: Research Note Resume

Add a first minimal resume path:

- write a Markdown research note next to the artifact
- write a machine-readable note summary into the artifact
- accept `--research-note <path>` or extend `resume-artifact` to ingest the
  prior note
- make the opening prompt include "already tried / do not repeat" and "open
  suggestions"
- add tests proving repeated expensive tools are not called again when the note
  already records a failed attempt

### Step 7: Human Suggestion Intake

Add a narrow user-suggestion mechanism before building a full UI:

- CLI flag: `--suggestion "try alternating alphabets by word"`
- artifact field: `investigator_report.human_suggestions`
- tool or prompt section that converts suggestions into testable experiments
- final note section: "Human suggestions tested"

This gives us the collaborative loop before committing to a richer terminal or
browser interface.

### Step 8: Presentation Mode Prototype

Design a future interactive mode after the resume/suggestion machinery exists.

Possible directions:

- enhanced terminal UI with a right-side hypothesis/evidence panel and a chat
  prompt between iterations
- lightweight local browser dashboard reading the live artifact
- Codex/Claude-style tool transcript plus pinned research note

The key requirement is continuity: user chat should update the evidence
notebook and experiment queue, not disappear into a transient transcript.

## Open Design Questions

- Should Investigator Mode be a separate CLI flag, or automatically activate
  when `known_cipher_type` is absent?
- How much of the diagnostic battery should be computed automatically before
  iteration 1?
- Should creative theory passes use a larger model than routine tool steering?
- How do we keep evidence notebooks concise enough that token costs do not
  explode?
- Can local non-LLM scorers produce enough evidence for the agent to avoid
  wasting expensive turns?
- How should negative evidence be weighted against a solver's high internal
  score?
- What is the right storage format for human-editable research notes:
  Markdown with front matter, JSON plus Markdown render, or both?
- How should a resumed run decide that an old experiment is worth repeating
  with a larger budget?
- What UI is sufficient for meaningful human collaboration without turning
  Decipher into a full notebook application too early?

## Relationship To Evaluation

For unknown ciphers, "success" should include:

- accurate non-overclaiming
- correct identification of likely cipher families
- useful rejection of wrong families
- actionable next experiments
- clear record of missing tools
- preservation of candidate evidence for later human review

This differs from normal benchmark scoring. It needs separate smoke tests and
artifact analysis checks.
