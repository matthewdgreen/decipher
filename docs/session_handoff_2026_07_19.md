# Session handoff — 2026-07-19

Self-contained context for a fresh session (the 2026-07-19 session tripped the
Fable safety gate on every review and was ended deliberately). Supersedes
`docs/session_handoff_2026_07_18.md`. Complements the persistent memory
(`~/.claude/.../memory/`), especially `parked-workplan` and `project-state`.

> **SAFETY-GATE NOTES (important):**
> 1. Every Fable *review* sub-agent this session gated down to `claude-opus-4-8`
>    on cryptanalytic content (8+ consecutive). Opus is capable; verdicts were
>    trusted where findings were independently reproduced. STILL run the
>    served-model grep per CLAUDE.md and report it. See memory `fable-review-gating`.
> 2. Decoded **Borg / historical-Latin** plaintext (esp. **borg_0171v**, a
>    turpentine recipe) misfires an Anthropic gate — use scalar/enum metadata
>    only for those pages. Synthetic original-prose plaintexts are fine.

## THE NEXT THREE STEPS (resume here)
1. **Composite hardening** — padding-trim (columnar padding blocks the declare
   gate on a perfect crack) + log the composite-branch-repair limitation.
   Detail: memory `composite-repair-padding-todo`. Acceptance: round-4 declares
   clean (`~/.config/decipher/dogfood_answers/round4_composite_answer.json`).
2. **Investigation CLI** — I-0 landed (`d541ea4`). Next I-1 (read verbs via the
   operation manifest) → I-2..I-6 → I-7 capstone (holistic Fable review +
   COMPLETE README rewrite; non-optional). Spec:
   `docs/specs/investigation_cli_spec.md`. Fold in
   `docs/specs/client_reading_tier_spec.md` (keyless graceful degradation, gated
   on I-0). USE MANUAL WORKTREES at the right base (isolation flag misfired to a
   stale base during I-0).
3. **Keyed-columnar exposure gap** — `analysis/columnar_search.py` solves keyed
   columnar 100% but only inside the composite peel; expose it as a standalone
   transposition route. Context: `docs/evidence/frontier_extension_2026_07.md`.

Other pending: PUSH (~20 unpushed commits); grade the agentic frontier suite
results when Matthew runs it; automated-frontier JSONL rows; the vestigial CSV
tier-1 cleanup. Full list: memory `parked-workplan`.

## What landed this session (newest first)
```
af73cd1 Frontier: agentic frontier suite (pasteable) + extension docs
a72574d Fix: agent search_homophonic_anneal path uses the bundled model too
054777c Fix: substitution/homophonic scorer uses bundled model not word-list (fresh-clone bug)
ae7bfc1 Bundle Zenith 2026.2 English model (Sol; GPLv3, provenance recorded)
2947bf8 Matrix: composite dogfood — LANTERN in-surface; round-4 session fail (later root-caused)
477e5fc Spec: client-reading verification tier
d541ea4 Investigation CLI I-0: service layer + operation manifest (byte-parity)
df9c2f0 Composite Slice C.2 — composite program COMPLETE
62570f4 Composite Slice B (content auto-route)
118f8ff Composite Slice C.1 (peel + shared columnar_search, closes F2)
034b600 Composite Slice A (diagnosis)
```
(plus the CLI/composite/polygraphic/client-reading specs and matrix updates).

## Headline results
- **Composite substitution+transposition program COMPLETE.** The gap that failed
  on every arm (round-4, Z340, INV "universally missed") is now diagnosed,
  auto-routed, solved (peel-and-solve), and in-surface (experiment type). F2
  (keyed-columnar) closed as a bonus. LANTERN generalization solved in-surface.
- **Fresh-clone model bug root-caused + fixed (both paths).** `_homophonic_model`
  only looked for the git-ignored proprietary Zenith CSV and fell to a WEAK
  word-list on any fresh clone. Fix: use the bundled `models/ngram5_*.bin` via a
  case-folding adapter (`homophonic.BinaryBackedNGramModel`). With `ae7bfc1`
  (Zenith model committed as the tracked registry default), a fresh clone solves
  round-4 100% — VERIFIED on Matthew's actual clone.
- **CLI I-0 landed** (byte-parity extraction + single operation manifest for both
  the MCP server and the future `decipher investigation` CLI).
- **Agentic frontier suite ready to run** (`docs/evidence/agentic_frontier_suite.md`).

## Specs written this session (under docs/specs/)
- `composite_substitution_transposition_spec.md` (implemented)
- `investigation_cli_spec.md` (I-0 done; I-1..I-7 pending)
- `client_reading_tier_spec.md` (pending, after I-0)
- `polygraphic_fractionation_solver_spec.md` (Codex draft; two Fable reviews
  folded; PF-0/PF-1 ready, later milestones scoped)

## Models / provenance
- The upstream **Zenith 2026.2 English model** is now committed at
  `models/ngram5_en_zenith.bin` (GPLv3, SHA/tag/corpora recorded in its
  `.metadata.json` + `docs/zenith_model_provenance.md`; "aggregate statistics"
  rationale for the Blog Corpus term; removal procedure documented). It is the
  registry DEFAULT English model; the Decipher-built `ngram5_en.bin` is the
  reproducible fallback. No non-commercial restriction added (GPLv3-consistent).
- Agent model for paid v3 runs: `gpt-5.5` (OpenAI). Server-side verify bills the
  configured provider (Anthropic keychain `service=decipher`).
