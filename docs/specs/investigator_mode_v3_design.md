# Investigator Mode on Agent v3 — Design

Status: design document, opened 2026-07-14. Synthesized from the nine-agent
design workflow (three readers; four lenses — architecture, cryptanalytic,
research-process, human-collaboration; two critics; journal
`wf_f08a1ff7-8a6`). Architecture reference for the unknown-cipher
investigator posture on the v3 harness (`docs/specs/agent_v3_design.md`,
C1–C8/A1–A12/M1–M6, plus the binding M1 spec). Supersedes the step list in
`docs/unknown_cipher_investigator_mode.md` (whose terminology it keeps) and
absorbs the investigator-relevant parts of `docs/unknown_cipher_agent_plan.md`
and `docs/beale_statistical_fingerprinting.md`. Per-slice implementation
specs will be derived under `docs/specs/investigator_mode_inv*.md`. Replaces
no shipped behavior by itself.

## Why a mode, not more prompts

A solved-benchmark run is a *solve*: one lifecycle, one declaration. An
unsolved-cipher investigation is a *research program*: sessions over a
persistent state, terminated by a documented stop condition, whose primary
deliverable is a research note a future run provably uses. The benchmark now
has a 262-record unsolved area (Voynich folios, Zodiac 340 variants, K4,
Beale 1/3, D'Agapeyeff, Scorpion, …) with context-tier gating and no ground
truth; nothing in the current harness is shaped for it.

The documented failure pattern (Feynman #2 run `2ff2c634eaa4`; borg_0077v
wrong-basin declaration at dict_rate 0.35): try familiar high-powered tools,
read weak word islands as damaged-but-promising plaintext, spend the budget
repairing a bad basin, declare late and badly, leave no research trail. The
extracted lesson: **English-like gibberish with no paraphrasable clause is
evidence** — it should fork the investigation toward
order/null/segmentation/custom-mechanism diagnostics, not fund more local
repair. The deeper problem is unique to this mode: on a true unknown,
success can never be confirmed, so the mode cannot measure itself by solve
rate. Progress must be redefined as calibrated instruments, honest coverage
of the hypothesis space, and notes that transfer across sessions.

## Design principles

1. **Structure over coercion.** No phase machine, no per-turn nags:
   schemas, one admission policy at the experiment queue, compiled
   verdicts. The lead drives.
2. **Negative results are the product** — generated from a ledger, never
   hand-written prose.
3. **Commit before you look.** The firewall keeps answers out of evidence;
   pre-registration keeps outcomes out of judgment standards. Deviations
   are legal but logged — we prevent *silent* judgment, not judgment.
4. **The noise floor is a measured quantity.** Every signal carries a
   shuffle-null percentile; every readability claim is tested against the
   possibility that it is a scorer artifact.
5. **Measure the instruments.** Search power, reader reliability, rule-out
   error rates, and stop-reason accuracy are measured on testgen analogs
   where truth is knowable — otherwise the mode produces beautifully
   audited claims of unknown reliability.
6. **The human is a second client of state** — same serialized state, two
   renderers; human input enters one adjudicated channel as testable
   hypotheses, influential but never overriding.
7. **Judgment is scarce; measurement is cheap.** Cheap tiers do breadth,
   the lead adjudicates, overnight compute is token-free.
8. **One schema, one store** per fact type (evidence, hypothesis,
   coverage, plan, suggestion).

## Adjudicated decisions

Where lenses conflicted, critics' corrections take precedence. Binding:

- **D1 — No lifecycle state machine.** The phase enum and six new episode
  kinds (`context_brief`/`design`/`adjudicate`/`replicate`/`note`/
  `ingest_note`) are cut: they reintroduce gate-bounce and constitute a
  second M2. Salvaged as data: pre-registration, admission rules, the
  advisory stall counter, the stop-condition taxonomy.
- **D2 — One `theorize` kind, scheduled early.** Theorize/diverge merged
  (feasibility), but the creative pass runs once *early* plus on stall
  (completeness): the Feynman-#2 lesson is that the alternation-mechanism
  idea must exist before the budget is gone.
- **D3 — The human-collab `human_inbox` is the suggestion channel**; the
  architecture lens's `Suggestion` record is deleted, not merged.
- **D4 — Resume never parses prose.** `state.json` is the only resume
  input; note Markdown is generated-only; a lossy foreign-note importer
  may produce suggestions and coverage stubs, never state.
- **D5 — One coverage store, one dedup key.** Coverage lives in the
  `FamilyCoverageLedger`, keyed `(family, view_hash)` with `view_hash =
  sha256(tokens + transform_pipeline + null_mask + segmentation)`; cards
  *reference* ledger entries. Experiment fingerprint = `sha256(tool,
  canonical_args_minus_seeds, view_hash, cipher_hash)`; `budget_profile`
  stored separately so a bigger-budget rerun is a different admissible
  event.
- **D6 — Notifications gate on calibrated verdicts, not raw coherence**
  (raw scores are untrustworthy inside their band, per the verification
  design itself): notify on compiled verdict ≥ `promising`, replicated;
  everything below is digest.
- **D7 — Budget governance layered, minimal first.** One total budget +
  per-session cap + lockfile now; envelopes/approvals/unattended mode
  deferred until multi-day runs demonstrably overspend. Precedence: hard
  caps > admission policy > advisory context. Overnight batches are
  human-pre-approved at composition time.
- **D8 — The lineup protocol is a spike, not a commitment.** Committed
  noise filters: island report, decoy floor at promotion, brittleness
  probe. The k-way lineup lands post-M5 under D10 rules.
- **D9 — `CandidatePacket` is the mandatory result currency** for every
  search/experiment/verification output (gaining an `island_report`
  field); A6 attestation hashes bind to its canonical rendered text.
- **D10 — Contamination rules are explicit.** Decoys/harvested basins must
  be cross-cipher or synthetic — never from the same investigation; caches
  are namespaced per investigation; calibration-run data (ground truth
  present) is segregated from true-unknown state; suggestions on
  calibration runs are stamped `possible_gt_contamination`.

## Architecture

### I1. Family registry and ranked diagnosis (`src/investigation/families.py`, `diagnosis.py`)

`FamilySpec` registry — the enum everything keys on: id, parent
(`quagmire_keyed` under `polyalphabetic_periodic`), `solver_status`
(`solves_automated | agent_assists | diagnoses_only | planned |
unsupported`), `detectors`, `discriminators` (named cheap experiments
splitting confusable families), `confusable_with`, `min_tokens_diagnose`.
Families: the mode-registry set plus `nulls_noise_layer` (composable
modifier), `numeric_book_cipher` (children `word_position`,
`char_position`, `skip_nth_word`), `plaintext_or_hoax`, `unknown_custom`.
`solver_status` carries the external-tool inventory (dCode/Boxentriq/
AZdecrypt parity honesty): the report can say "plausible family, but
Decipher can only diagnose it," and AZdecrypt batch export becomes a
legitimate experiment for such families.

`DiagnosisReport` wraps `analysis/cipher_id.py` (kept as the raw layer):
`view_hash`, fingerprint, evidence atoms, `ranked` over ALL registry
families, report-level `verdict: "confident" | "uncertain"`,
`battery_coverage` (panel → ran | skipped(reason)), `recommended_next`.
Per-family: score (evidence weight, not probability), confidence, evidence
AND **counterevidence** (mandatory, always rendered), pending
discriminators, `solver_status`, ledger `coverage_status`.

Uncertainty is compiled arithmetic: `strong` needs score ≥ 0.70, ≥ 2
high-reliability atoms, margin ≥ 0.25 over the best confusable family; the
report is `uncertain` when top-two margin < 0.15, tokens < 60, the top
family's discriminators are all pending, or top-two are mutually
confusable with none run. An uncertain verdict MUST list ≥ 2 live families
and lead `recommended_next` with a discriminator between the top two.
Fixtures include ambiguous near-misses that fail if the report is
confidently wrong OR confidently right for the wrong reason.

### I2. Statistical panel battery (`src/analysis/`)

Nine panels, each a pure function returning
`PanelResult {measurements, atoms, reliability}` with token-count gates
(`reliability="not_computable"` instead of noisy numbers). Tool name =
`observe_<panel>`; the battery runs as preset cheap-tier `survey` episodes
(pre-M2 degenerate form: synchronous turn-0 calls).

| Panel | Content (delta over existing code) |
|---|---|
| P1 `shape` | alphabet class (letters/digits/glyphs/mixed), line/grid factorizations, boundary availability, **symbol-inventory drift** (chi² across halves/pages — multi-key/Voynich-section signal) |
| P2 `frequency` | exists (IC, entropy, flatness/peakedness) |
| P3 `repetition` | repeated n-grams w/ spacings/gcds + expected-repeat count under multinomial null |
| P4 `periodicity` | periodic IC **with shuffle-null percentile**, Kasiski corroboration, phase-local fit |
| P5 `order_layout` | transposition signature: monogram fit HIGH while bigram/quadgram fit LOW (quantified gap); row/col IC per grid factorization |
| P6 `nulls_noise` | null candidates, rare-symbol tail, IC trajectory under top-k null masks |
| P7 `polygraphic` | doubled-digraph rate, digraph-IC/monogram-IC ratio, 25/36-symbol coordinate signals |
| P8 `numeric_code` | **new** `analysis/numeric_code.py` — Beale-class battery below |
| P9 `language` | provisional language candidates; language-agnostic stats first |

**P8 numeric/book-cipher battery**
(`numeric_code_battery(values, *, related_profile=None)`): count/unique/
max/min/repeated-token rate, gap histogram; first- and last-digit
distributions (chi² vs uniform and vs Benford); **Benford and
epsilon-Benford deviation** (must reproduce Wase 2021's Beale numbers);
repeated numeric n-grams with null expectation; monotone/consecutive runs;
modulo structure (m ∈ 2..12, 26); **required key-text length** under
word-position and char-position hypotheses plus skip-Nth feasibility;
**front-loading index** (dictionary/book-order skew); plausibility flags
{word_first_letter_book_cipher, char_position_book_cipher, skip_nth_word,
shared_key_with_companion, hoax_or_random_generation}; optional distance
to an explicitly supplied public reference profile (Beale 2/DoI via
context tier, never from ground truth). Corpus search stays a separate,
ledger-gated experiment layer; "no match found" is recorded as
`searched_negative` with corpus snapshot + tokenization rules, never as
`ruled_out`.

### I3. Unified investigator state (extensions to `InvestigationState`)

One dataclass per fact type, one owner file; all serialize inside
`to_artifact_dict()` (additive per C8). Two M1-spec amendments: a
monotonic **`state_seq`** counter stamped on every record, and a per-turn
hook `on_turn_end(state)` writing `state.json` atomically and appending
`events.jsonl` — cheap now, expensive to retrofit.

**`EvidenceEntry`** (merges the lenses' EvidenceEntry/EvidenceAtom; the
baseline/reliability fields are load-bearing for anti-pareidolia):

```python
@dataclass
class EvidenceEntry:
    id: str; run_id: str; created_seq: int
    kind: str            # diagnostic|episode_finding|attestation|solve_lesson
                         # |tool_gap|suggestion_outcome|turn_summary
    panel: str | None
    observation: str     # "periodic_ic_peak"
    measurement: dict
    baseline: dict|None  # {"null_percentile": 99.2, "n_shuffles": 50}
    reliability: str     # high|low|not_computable
    interpretation: str  # one sentence, plain language
    supports: list[str]; weakens: list[str]   # hypothesis card ids
    confidence: str; caveats: str
    provenance: dict     # {tool, view_hash, episode_id|experiment_id|"preflight", turn}
```

Context (C2) renders digests only; measurements stay in state. Atoms with
`null_percentile < 95` are auto-labeled `reliability="low"` and cannot be
the sole basis for a `strong` family confidence.

**`HypothesisCard`** (the A10 single-writer board, extended): id,
mode-or-`custom_mechanism`, status (`live|favored|weakened|rejected|
superseded|needs_new_tool|declared`), confidence, evidence_for/against,
**priors** (distinct field — a prior can never move a card to `favored`
by itself), experiments_run (fingerprints), next_discriminating,
solve_attempted, solve_lessons, linked_branches, **scope** (record id or
page-group id — multipage groups are first-class: "all pages share one key
text" is one card), origin (`preflight|lead|theorize|human_suggestion|
prior_state`), rejection_reason. v2 `workspace_*_hypothesis` handlers
become adapters over the board (A10).

**`FamilyCoverageLedger`** — the negative-evidence ledger, single store:

```python
@dataclass
class RuleOutRecord:
    id: str; family: str; view_hash: str
    confidence: str          # weak|moderate|strong
    basis: list[str]         # evidence/experiment ids — must exist (compiled check)
    coverage_spec: dict      # searched subspace, machine-readable, e.g.
                             # {"tableau":"straight_a_z","periods":[2,26],"budget_profile":"screen"}
    measured_power: dict|None  # {"n_analogs": 20, "detect_rate": 0.85, ...} — see I8
    residual_risk: str       # what would reopen it: "keyed tableaux untested"
    budget_spent: dict; created_turn: int

class FamilyCoverageLedger:
    status: dict[tuple[str, str], str]  # (family, view_hash) -> untested|
        # diagnosed_plausible|diagnosed_unlikely|screened_negative
        # |searched_negative|ruled_out|active_hypothesis|solved
    ruleouts: list[RuleOutRecord]
    def coverage_debt(self, diagnosis) -> list[dict]: ...
```

"Vigenère ruled out" is almost never true; "Vigenère with straight A–Z
tableau, periods 2–26, ruled out at moderate confidence by exp_012, power
0.85 on 20 analogs" is. The ledger stores only the second kind. Rule-outs
below a power floor cap at `weak` regardless of the lead's judgment.
`coverage_debt` — families the diagnosis scores ≥ moderate whose coverage
doesn't reach the evidenced subregion — renders in every lead context and
must be empty or explicitly waived before `meta_declare_unsolved` (the K2
family-coverage guardrail generalized). `ruled_out` requires ≥ 2
independent basis entries, ≥ 1 an experiment. Views make coverage
transform-aware (a raw-view rule-out says nothing about a null-masked
view); variant *transcriptions* are views too — "the transcription is
wrong" is a testable hypothesis (glyph merge/split as transforms), not a
caveat field.

**`ExperimentPlan`** — the unified pre-registration record (collapses the
lenses' Proposal/Preregistration/ExpectedSignal into one object):

```python
@dataclass
class ExperimentPlan:
    id: str; question: str
    hypotheses: dict[str, str]    # card_id -> "strengthens"|"weakens"; must discriminate
    evidence_basis: list[str]     # ids justifying the spend; validated to exist;
                                  # machine-checked non-empty for cost_tier >= moderate
    expected_signal: str          # concrete & measurable, e.g. "periodic IC at p in 5..9
                                  #  > 1.5 sigma above 200-shuffle baseline"
    success_criteria: dict
    abandon_threshold: str        # pre-declared kill line, e.g. "best finalist dict_rate
                                  #  < 0.45 after 2M proposals -> family weakened,
                                  #  no resubmit without new evidence"
    failure_interpretation: dict  # {"ledger_action": "weaken"|"searched_negative", ...}
    cost_tier: str                # cheap_diagnostic|bounded_screen|expensive_search
    budget: dict; spec: dict|None # C5 Experiment spec; None => tool gap
    fingerprint: str              # D5 dedup key
    status: str; origin: str; registered_at: str; sha256: str  # immutable hash
```

Cheap diagnostics get a lightweight one-line form so the protocol doesn't
tax breadth. Adjudication compares results to `success_criteria`
mechanically and applies `failure_interpretation` to the ledger by
default; the lead may promote a "failed" result but must attach a
`goalpost_note` (a logged `protocol_deviation` — cheap to make, impossible
to hide). Abandon thresholds bind resubmission: same fingerprint at ≤ same
budget profile with prior outcome ∈ {failed, weak} is blocked unless
justification ∈ {`budget_increase`, `new_evidence:<id>`,
`human_request:<id>`, `regression_repro`} — the four sanctioned repeat
reasons, enforced in `AdmissionPolicy.check()` at queue submit (M4;
pattern-copies A2; no pre-M4 executor soft-gates — they recreate
gate-bounce). The Feynman guard is one admission rule: broad
periodic/Quagmire tiers require ≥ moderate periodic evidence or a readable
cheap-run candidate. Blocks return structured refusals naming the
admissible alternative.

**`human_inbox`** — suggestions as first-class testable hypotheses (D3):
id, seq/timestamps, source (`human|lead`), immutable `verbatim` text,
kind_hint, priority (`normal|pinned`), status (`new → seen →
converted|declined|needs_tool → tested → resolved`), adjudication
{converted_to, affects_hypotheses, note}, resolution,
`possible_gt_contamination`. Compiled rules: verbatim text never enters
`evidence_log` — only its *test result* does, tagged
`provenance.tested_suggestion`; declaration blocked while any item is
`new`/`seen` (declining with rationale satisfies the guard); pinning
forces re-adjudication but cannot force belief; the lead can file
`question` records addressed to the human. A suggestion naming a family
produces a card with `origin="human_suggestion"` and normal evidence
bookkeeping — influence without override. Intake: repeatable
`--suggestion` and `decipher investigate suggest` mid-run (merged from a
pending file at turn start; race-free since context is rebuilt every
turn). `assert_no_ground_truth_leak` runs on suggestion text for
benchmark-backed records.

**`InvestigationManifest`** (`investigations/<id>/manifest.json`):
investigation id; **target = record ref OR page-group ref** (reusing
`analysis/multipage.py`; context tier pinned — escalation is an explicit
logged manifest event, requested by the lead via inbox question,
authorized only by the human); state path; session list with per-run
cost; cumulative budget rollup; minimal budget policy (D7); note path.
Single-writer lockfile. Resume is v3's identity operation: `decipher
investigate <target> --investigation-id X` loads or creates manifest +
state; the ledger/board render "already tried / do not repeat"
structurally.

### I4. Episode kinds and model economics

Inherited kinds do investigator duty unchanged: `survey` (battery
elaboration, cheap tier), `search` (bounded solve attempts —
`EpisodeSpec.goal` gains a required `hypothesis_id`), `compare` (the
cross-mode scoring card; scores explicitly non-commensurate; readability
first-class via attestation refs), `verify` (I6). One new kind:

**`theorize`** (D2): inputs — anomaly digest (evidence the lead marks
unexplained), board including rejected families with reasons, 2–3
ciphertext windows, context-tier slice; toolset none-or-read-only; result
schema 3–5 × {mechanism, expected_statistical_footprint, cheap_check:
ExperimentPlan-draft | tool_gap}. Proposals without a testable footprint
are schema-invalid. Output lands as `custom_mechanism` cards plus queued
cheap checks; untestable ones become `needs_new_tool` + tool-gap evidence
(the note's "missing tools" section for free). Scheduled early and on
stall via an advisory standing suggestion — never a gate. Strong tier.

Tier economics are config on the C7 kind→session-factory registry (zero
new code; single-model Ollama parity preserved): **cheap** =
gpt-5.6-luna-class scouts (~1/5 lead cost, proven solve-capable at
$0.59/page) for survey fan-outs, first-pass verify, triage; **mid** =
sol/sonnet-class for compare/repair/declaration-grade verify — bounded
judgment against explicit written criteria; **lead** = gpt-5.5 for the
lead context, `theorize`, readings on promoted branches, adjudication of
surprising results. Scouts discover, the lead confirms — one lead-tier
confirmation per N cheap discoveries is the point of the tiering.
Overnight experiment batches are pure solver compute (zero tokens).
Surprising results (would flip the favored hypothesis or unlock > 10% of
remaining budget) replicate before touching the board: ≥ 1 independent
re-run varying seed/tier/ciphertext-window; 2-of-3 for readability claims;
cheap-tier discoveries replicate at a higher tier at least once.

### I5. Anti-pareidolia instruments (compiled, LLM-free core)

- **`analysis/null_baseline.py`** — `null_percentile(statistic_fn, tokens,
  n_shuffles=50)`: every signal atom ships with a shuffle-control
  percentile (frequency-preserving permutation, so survivors are *order*
  structure). Cached per (statistic, view_hash), namespaced per
  investigation (D10).
- **`analysis/coherence.py` `island_report(text, language)`** — the
  word-island illusion detector: dict_rate (exists),
  **function_word_rate** (closed-class fraction; real prose ≈ 0.35–0.55,
  bad basins < 0.15), word-bigram log-likelihood,
  **longest_coherent_span** (consecutive dictionary words containing ≥ 1
  function word), verdict `coherent | word_islands | gibberish`. Catches
  borg_0077v and the Feynman-#2 basins in microseconds. Needs ~150-word
  closed-class lists + word-bigram top lists (en/la/de). Consumed by every
  `CandidatePacket`; a `word_islands` branch gets `basin_flag="suspect"`,
  and repair episodes cannot target a suspect branch more than twice
  without an intervening structural experiment (the diagnostic fork,
  materialized as auto-enqueued plan drafts).
- **Decoy floor** (promotion-time only, D8): run the identical search spec
  at reduced budget on 2–3 permutation decoys of the same view; report
  `finalist_decoy_percentile` — the metric pre-registrations key on. A
  "great" anneal score that decoys also reach is noise by construction.
- **Convergent mirage** (in `branch_adjudicate`): when ≥ 2 branches under
  incompatible modes surface overlapping readable words, high island
  overlap flags `shared_mirage` on both — evidence about the scorer, not
  the cipher.

### I6. Verification and declaration

Builds on C6 verify episodes and `llm_reader.py`, whose firewall
(`assert_no_ground_truth_leak`, neutral ids, provenance stripping) is the
shared implementation for all verification surfaces. Graded instruments,
cheapest first: (1) compiled prechecks — `island_report` short-circuit
(`gibberish` ⇒ zero LLM verify spend) and **exact positional crib checks**
where public partial evidence exists (K4's EASTNORTHEAST/BERLINCLOCK,
zodiac153): crib inconsistency is an automatic hard block on declaration,
the one absolute check we own; (2) verify episodes (candidate text +
language only) — `memorization_risk` records add a familiarity check
("does this read as a known published text or claimed solution?"), the
firewall extended to model memory; unsolved-area declarations use two
verifies from different model families when available; (3) **stability
probe** — minimal key perturbations (2-symbol swaps / adjacent-column swap
/ ±1 phase), re-render, re-score: real plaintext degrades sharply and
locally, pareidolia degrades gracefully (`brittleness`); (4) the lineup
protocol (spike, D8) — the candidate among score-matched decoys, forced
ranking; indistinguishable-from-decoys caps what any raw coherence score
in that band may claim.

Compiled `lead_verdict(island, stability, crib, decoy_percentile,
attestation) → strong_lead | promising | indeterminate | likely_noise`;
`indeterminate` routes to a discriminating structural experiment, never
more repair. Anti-sycophancy binding: each glossed clause must quote a
verbatim span (checked mechanically) with a non-copy paraphrase; failures
downgrade to `unverified_reader_claim`, which cannot satisfy the M5 gate.
`meta_declare_solution` requires verdict ≥ `promising` on the exact
content hash (A6). On unsolved-area records the terminal output is
`candidate_decipherment` + attestation, never "solved" — promotion to
`solved_probable` is a human act outside the run; a surviving
`strong_lead` triggers the success protocol (clean-room re-derivation of
the key from the note by a fresh session before any claim leaves the
repo).

**Stop conditions** — `meta_declare_unsolved(stop_reason, note_ref)`,
checked at session boundaries and at 80% of budget (endgame hygiene, now
structural), five documented reasons:

1. `budget_exhausted` — at 90% of total, queue admission closes to new
   expensive experiments; the remainder is reserved for consolidation, so
   the note is never starved.
2. `diminishing_returns` — measured, not felt: `stall_counter` =
   consecutive adjudicated experiments with zero board delta; at K
   (default 4) the lead must run `theorize` (once per stall) or stop.
   Calibration (I8) validates the delta metric on analogs so status churn
   cannot game it; until validated, this stop needs human confirmation.
3. `frontier_documented` — the *good* stop: no admissible experiments
   remain; everything left is `needs_new_tool`, external resources, or
   over-budget under its own estimate. The note's frontier section
   quantifies each blocked lead (tool, corpus, $) as engineering backlog.
4. `likely_no_plaintext` — hoax/fabrication as a **positive, tested
   hypothesis**: `plaintext_or_hoax` is a registry family with its own
   discriminators (Gillogly-string-like artifacts, digit-preference and
   keyboard-adjacency signatures of human-fabricated randomness,
   statistics inconsistent with any enciphering process that could
   produce the observed IC). Hoax cannot be ruled out by searching; its
   evidence type is fit-to-fabrication-models, not coverage.
5. `underdetermined` — information-theoretic: unicity-distance /
   key-equivocation estimates per family; a 40-token 30-symbol cipher may
   admit many equally valid keys, and the correct verdict is
   "underdetermined at this length under this family" — distinct from
   "hard" and from "hoax." The note must also state which verdict tiers
   are even reachable per family: a nomenclator/codebook solution can be
   correct and still fail every readability instrument.

### I7. Research note (renderer, not store)

`render_research_note(state, manifest) → Markdown` — every section a pure
projection of state; JSON canonical in the artifact, Markdown
generated-only (D4). Sections: abstract with per-claim confidence; target
and provenance (transcription caveats verbatim from the manifest); methods
(code commit, tool versions, model ids per kind, seeds, budget, context
tier, whether human hints were provided); findings as claims — each with
evidence ids and a repro block {command, seeds, config_hash, code_commit,
expected}, nearly free because A5 experiments are pure functions;
**negative results first-class**, generated from the ledger (family,
confidence, coverage_spec, measured power, residual risk); board final
state; human suggestions tested (verbatim → test → outcome, including
declines); candidate decodes with attestations and explicit non-overclaim
statements; tool gaps; ranked next-most-informative experiments, each
carrying a ready ExperimentPlan draft a future session can submit without
redesign; reproducibility appendix. A compiled validator rejects any claim
whose evidence ids don't exist or whose repro block is incomplete —
structural honesty, not prompted honesty. The note is checkpointed each
session end, so interruption still yields the deliverable. Success is **a
note a future run provably uses**: acceptance includes a two-session
fake-provider test where session 2 loads session 1's state, does not
resubmit a covered search, and does submit the top ranked open lead.

### I8. Calibration layer (measuring the instruments)

Without this the mode cannot know whether it works (the progress-
measurement problem for never-confirmable targets).
`src/investigation/calibration.py` + testgen, the unexploited remedy:

- **Analog generation contract**: testgen produces ciphers matched to each
  target's measured fingerprint (length, alphabet size, IC, boundary
  availability, noise), spanning solvable / hard-but-solvable /
  unsolvable-by-current-tools / actually-random.
- **Metrics**: diagnosis top-k accuracy; correct stop_reason rate;
  **false-rule-out rate** (of families marked ruled_out/searched_negative
  at strong confidence on analogs, how often was it the true family?);
  confidence calibration curves; board-delta validity.
- **Measured search power**: run a plan's exact spec on analogs where the
  family IS true; `detect_rate` populates `RuleOutRecord.measured_power`.
  This attaches statistical power to negative evidence exactly where
  unknowns live — short/hard ciphers, where power is lowest and the ledger
  would otherwise systematically overclaim.
- **Reader calibration bridge**: Phase 4a's `run_reader_calibration.py`
  score-vs-accuracy tables set attestation per-band reliability priors,
  the lineup gap threshold, and the notify threshold (D6) empirically.
- **Cadence and segregation**: calibration re-runs when tooling changes
  (every rule-out inherits the tool version that produced it); calibration
  data is segregated from true-unknown state (D10); artifact records are
  tagged training-eligible, feeding the fine-tuned-scorer roadmap
  (`train_language_quality_scorer.py`) from investigation exhaust.

### I9. Human collaboration surface

CLI-first; all renderers over `state.json` + `events.jsonl` (no second
write path, no UI database): `watch` (board with per-family spend, queue
with approval flags, evidence tail since last review, best candidate
**rendered inseparably from its attestation verdict**, inbox, alerts);
`report --since last` (the seq-diff shift report: headline, board changes,
completed experiments with negative results called out, inbox
dispositions, action items; reading advances the review cursor);
`suggest / answer / approve / deny / mark / note`. `mark` writes advisory
`human_interest`; agent expected-information-gain and human interest
render as two separate columns, never silently blended. A watcher tails
`events.jsonl` (delivery pluggable, outside the loop); notify-class events
per D6 plus `approval_needed`, `run_error/stalled`, budget exhaustion —
default posture is quiet. The web dashboard is deferred post-M6, read-only
over the same files.

Never automated: claiming a solution; budget increases and over-threshold
approvals; repeating expensive failed searches absent a named material
change; context-tier escalation (lead requests, human authorizes);
external corpus acquisition beyond a human-whitelisted set (rights and
prompt-injection surface — context documents and `hold_for_review`
material are adversarial prompt input, handled with the same suspicion as
suggestions); treating suggestions as evidence; history destruction (board
and log are append-only — rejected hypotheses are the product, not
clutter); publishing.

## Milestones

Sequencing per the feasibility critic. A one-day schema-unification
mini-spec (I3's five dataclasses, one owner file each) precedes any code.

**INV-0 — Compiled analysis pack** (no v3 dependency; consumable by v2
immediately). Family registry; `DiagnosisReport` + uncertainty rules +
ambiguity fixtures; P8 numeric battery; `null_baseline.py`;
`island_report` + closed-class/bigram resources wired into v2 finalist
validation; `scripts/research/beale_report.py`. Acceptance:
`observe_diagnosis` on beale_1/beale_3 ranks `numeric_book_cipher` above
all substitution families; the report script emits the comparative Beale
1/2/3 statistics table with zero plaintext access and reproduces the
Wase/Campanelli broad conclusion; borg_0077v render yields `word_islands`;
ambiguous near-miss fixtures return `uncertain` with the correct
discriminator recommended.

**INV-1 — Investigator state + note + human channel** (needs M1). Unified
schemas; typed evidence; coverage *recording*; `human_inbox` +
`--suggestion` + A2-policy declaration guard; note renderer + claim
validator; `state_seq`/`events.jsonl`/`on_turn_end` (M1 spec amendment);
`watch`/`report`/`suggest` CLI; minimal manifest + lockfile + resume.
Acceptance: a full investigator run on Beale produces a defensible
research note (diagnosis atoms, hoax-vs-book evidence, frontier section);
the two-session resume test (I7); declaration blocked while an inbox item
is unadjudicated.

**INV-2 — Battery episodes + theorize** (needs M2). Battery as preset
`survey` specs; merged `theorize` kind scheduled early; card writes on the
A10 board; advisory stall counter in context. Acceptance: fake-provider
theorize test (anomalies in → testable mechanisms + cheap-check plan
drafts out; untestable → tool gaps); Beale rerun with episode fan-out.

**INV-3 — Plans + admission at the queue** (needs M4). `ExperimentPlan`
enforcement at submit (evidence-basis for expensive tiers, dedup
fingerprint + four repeat reasons, abandon-threshold binding);
total/per-session caps; first `measured_power` runs on one family.
Acceptance: fake-provider tests — unregistered experiment rejected;
abandon-hit blocks resubmission without a named change;
`english_like_no_clause` forces structural-fork drafts and blocks local
repair (the `2ff2c634eaa4` replay as a regression fixture); a D'Agapeyeff
investigation runs the queue-governed loop and yields honest negative
results with power annotations.

**INV-4 — Verification loop** (needs M5). Verify→evidence write-back with
basin taxonomy; island short-circuit wiring; brittleness probe; compiled
`lead_verdict` (minus lineup); familiarity flag; notify gating (D6);
reader-calibration priors on attestations. Acceptance: borg_0077v fixture
yields `likely_noise` by compiled verdict; a true-plaintext synthetic at
85% accuracy scores `promising`; declaration carries coverage summary +
attestation; crib mismatch hard-blocks a scripted K4-style case.

**INV-5 — Gated spikes** (post-M5/M6, only after INV-1..4 usage shows the
specific gap): lineup spike under D10 rules; per-family envelopes/
approvals/unattended governance; calibration-harness automation
(stop-reason and false-rule-out dashboards); presentation dashboard.

Cut list (do not build): the phase state machine; `design`/`adjudicate`/
`replicate`/`note`/`ingest_note` episode kinds; note parsers;
`investigator_report` and `workspace_investigator_status` as objects
(subsumed by state + renderers); a separate `diverge` kind; governance v1;
memorization machinery beyond the verify-prompt familiarity flag.

## Risks and mitigations

- **Coercion creep.** Every guard must live at one of two choke points
  (queue admission, declaration policy) or be advisory context. Review
  criterion for INV specs: no new per-turn gates or tool-call bounce loops.
- **Investigator tax on solvable ciphers.** Activation only by
  `--investigator`, or record `cipher_type` unknown/`*_candidate`, or
  status unsolved/disputed. M6's bake-off matrix gains a criterion:
  investigator-on-solvable ≥ v3-baseline accuracy at ≤ 1.2× cost on a
  small solvable subset.
- **Schema drift back to parallel stores.** The unification mini-spec is a
  hard prerequisite; reviews reject any INV slice introducing a second
  store for evidence, coverage, plans, or suggestions.
- **Gameable progress metrics.** Board-delta stall counting is validated
  on analogs (I8); until then `diminishing_returns` stops require human
  confirmation.
- **Contamination via verification assets.** D10 rules are firewall
  extensions and get firewall-grade tests (decoy provenance, cache
  namespacing, calibration segregation).
- **Unbounded ambition.** INV-5 items are gated on demonstrated need; the
  lineup and governance subsystems are the two known tarpits.

## Interactions with the existing program

- **v3 milestones**: INV-0 is parallel to M1 (no dependency); INV-1 rides
  M1 and amends its spec (`state_seq`, `on_turn_end`); INV-2 rides M2
  (theorize registers in the C7 kind factory); INV-3 rides M4
  (AdmissionPolicy pattern-copies A2); INV-4 rides M5, which already
  implements the core bad-basin oracle. Nothing blocks M6's bake-off.
- **Improvement program**: Phase 1 packets are the result currency (D9);
  Phase 4a reader calibration becomes load-bearing input to I8; the
  fine-tuned-scorer roadmap consumes investigation exhaust; Phase 2
  multipage machinery is reused for group-scoped investigations (Beale
  1+3, Scorpion composite, Voynich sections) — cross-page consistency
  under a shared-key hypothesis is a free verification instrument.
- **v2**: INV-0's island detector and diagnosis report wire into v2
  finalist validation immediately; no other INV work lands on v2.
- **Benchmark unsolved area**: context-layer booleans and
  `related_records` gating are enforced upstream of episodes; K4's
  privately confirmed plaintext stays quarantined (public cribs only);
  `hold_for_review` documents never render in prompts or views.
- **TOOLS.md**: new observe panels, `observe_diagnosis`,
  `coverage_record`/`coverage_report`, and the investigator CLI verbs are
  documented as they land, per the standing rule.

## First slice

**Pilot target: Beale 1/3** (D'Agapeyeff second, only after INV-3 —
before queue and coverage enforcement exist it would replay the Feynman-#2
failure with better bookkeeping). Beale is the right first target: its
deliverable is diagnosis + negative evidence + a frontier note —
achievable with zero search/episode machinery, group-scoped (two related
pages, one suspected key mechanism), and
`docs/beale_statistical_fingerprinting.md` already carries acceptance
criteria.

**Minimal buildable increment: INV-0, startable today, independent of
M1.** Family registry, DiagnosisReport with compiled uncertainty rules,
the P8 numeric battery, shuffle-null baselines, and the island report —
roughly 2–3 new modules plus small language resources, comparable in size
to a Phase-2.x. It pays off before any investigator loop exists: the Beale
1/2/3 comparative report with no plaintext access, the borg_0077v
wrong-basin fixture caught by a compiled detector, and v2 finalist
validation hardened. The anti-pareidolia core is loop-independent, and it
is the best value-per-effort item found anywhere in this design.

## Recorded design notes (2026-07-15, user-requested)

### Cross-run investigation memory (the "case file")

Proposal (Matthew, 2026-07-15): an INV run's results should be
recordable so that a LATER run on the SAME cipher can consume them —
optionally, not in every case, so it doesn't confound experiments.

Design sketch (for the INV-1+ slice that implements it):

- **What persists**: the DiagnosisReport (ranked families, atoms,
  verdict), discriminator results already run (with their statistics),
  verified NEGATIVE results ("periodic-IC null p=0.26 — polyalphabetic
  disfavored"), and any attested readings — i.e. evidence and rulings,
  not raw transcripts.
- **Keyed by ciphertext content hash** (same identity rule as M5
  attestations): "the same cipher" is decided by the token stream, not
  by filename or benchmark id — no benchmark metadata in the case file
  (firewall preserved).
- **Consumption is OPT-IN per run** (e.g. `--case-file <path>` or a
  registry keyed by hash with an explicit flag): default OFF so
  experiments are unconfounded. Every consumed case file is stamped
  into the artifact (provenance: which run produced it, when), so a
  run's context is always auditable.
- **Experiment protocol**: with-memory and without-memory arms are
  distinct conditions, never mixed in an aggregate. This gives the
  playbook ablation a third axis, cleanly separated:
  (a) +fingerprint = computed statistical EVIDENCE for this ciphertext;
  (b) +playbook = static METHODOLOGY (family checklists, discriminator
  recipes, composite discipline — no per-case data);
  (c) +case-file = MEMORY from prior investigations of this cipher.
  The 2026-07-14 sweep's universal composite failure is the standing
  motivation for (b); long-horizon ciphers (Beale-class, multi-session)
  are the motivation for (c).
