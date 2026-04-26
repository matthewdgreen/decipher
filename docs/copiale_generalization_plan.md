# Copiale And Generalization Plan

Status: new plan opened after completing the Agent Loop Redesign milestones and
Milestone 4 smoke coverage. Treat this as the next capability track, not as a
continuation of Borg-specific repair work.

## Goal

Make Decipher's agentic and automated paths generalize beyond readable Borg
Latin pages, starting with Copiale/German and then widening to other benchmark
cipher families. The goal is not just higher scores; each failure should point
to a missing model, cipher-system assumption, context hook, agent workflow, or
benchmark issue.

## Current Read

- Borg progress is real: multiple Latin pages now reach readable partial or
  strong outputs, and the English analog confirms the agent can exploit full
  readings when the right actuator exists.
- Copiale `p068` did not improve meaningfully over preflight. The agent found
  German-looking islands but not coherent sentence-level German.
- That failure is structurally different from Borg `0109v`: Copiale likely
  needs stronger German language modeling, better homophonic/nomenclator
  handling, and stricter declaration discipline around isolated words.
- The new Milestone 4 smoke suite gives us a regression base before beginning
  this broader work.

## Principles

1. Separate blind parity from context-aware runs.
   If the solver uses source family, language, or manuscript notes, artifacts
   and summaries must say so.

2. Avoid Borg-shaped overfitting.
   A workflow that helps word-boundary drift in Latin should not be assumed to
   solve German homophonic manuscripts.

3. Build diagnostics before broad fixes.
   For each failing Copiale page, we want to know whether the bottleneck is
   search, scoring, language model quality, symbol inventory, nulls/codewords,
   segmentation, or agent behavior.

4. Keep automated and agentic paths comparable.
   The agent may receive richer tools and optional context, but the no-LLM
   baseline must remain easy to run and artifact-compatible.

## Milestone 1: Copiale Evidence Packet

- [ ] Select a compact Copiale packet:
  `copiale_single_B_copiale_p017`, `p035`, `p052`, `p068`, and `p084` if
  available in the local benchmark checkout.
- [ ] Run automated-only baselines for the packet and store/compare artifacts.
- [ ] Run agentic baselines without extra manuscript context.
- [ ] Add an artifact comparison note summarizing:
  - first readable branch
  - final declared branch
  - German word/sentence coherence
  - edit-aware character and word scores
  - whether the agent declared on isolated islands
- [ ] Add the packet to a frontier/smoke file only after thresholds are stable.

## Milestone 2: German Model And Scoring Audit

- [ ] Verify the bundled German corpus/model provenance and normalization.
- [ ] Compare German dictionary scoring, quad scoring, and continuous n-gram
  scoring on known-good German plaintext samples.
- [ ] Build or select a stronger German continuous model if the current model
  is weak for 18th-century or Masonic-style prose.
- [ ] Ensure automated artifacts record the selected German model name, path,
  checksum, and provenance.
- [ ] Add a small model A/B packet so future model changes are visible.

## Milestone 3: Copiale Cipher-System Diagnostics

- [ ] Measure symbol inventory, homophone-family behavior, null candidates,
  repeated-symbol structure, and likely codeword/nomenclator spans per page.
- [ ] Add a Copiale-focused diagnostic report that does not require ground
  truth but can be compared against it when available.
- [ ] Decide whether the automated route should use:
  - plain homophonic substitution
  - homophonic plus null handling
  - nomenclator/codeword hypotheses
  - segmentation/transcription normalization first
- [ ] Surface diagnostic recommendations to the agent in concise branch cards.

## Milestone 4: Agent Workflows For German Manuscripts

- [ ] Add a "German coherence" workflow that distinguishes isolated dictionary
  hits from sentence-level German.
- [ ] Teach the agent to hold, not declare, branches that only contain islands
  such as articles, particles, or short common words.
- [ ] Add optional context-loading support for benchmark `context_records`,
  gated by a CLI flag so blind and context-aware runs remain separate.
- [ ] Add prompt notes for historical German spelling and Copiale-specific
  caution, but keep them concise and evidence-driven.
- [ ] Add fake-provider smoke tests for any new workflow gate before live runs.

## Milestone 5: Generalize Beyond Copiale

- [ ] Classify benchmark cases by cipher family and current solver capability:
  simple substitution, homophonic, nomenclator, transposition, polyalphabetic,
  OCR/transcription-heavy, and mixed/unknown.
- [ ] For each unsupported family, decide whether Decipher should:
  - implement a native baseline,
  - wrap an external baseline,
  - expose diagnosis-only tooling,
  - or mark the family out of scope for now.
- [ ] Extend frontier packets by family only when there is a meaningful
  baseline and a clear expected behavior.
- [ ] Keep live-agent smoke small; use fake-provider tests for loop mechanics
  and opt-in live runs for capability checks.

## Open Questions

- Is Copiale primarily a better-search problem, or are nulls/codewords the
  dominant obstacle?
- Does German continuous n-gram scoring help on Copiale text, or does
  historical spelling/transcription mismatch overwhelm it?
- What context is fair to provide in context-aware parity, and can external
  baselines consume comparable context?
- Which non-Borg benchmark family should become the second generalization
  target after Copiale?
