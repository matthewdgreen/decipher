# Spec: Improvement Program Phase 4a — LLM Reader Library + Runner Reranker

Parent plan: `docs/improvement_program_plan.md` (Phase 4, items 4.1–4.2 +
the 4.4 calibration bake-off; 4.3 is subsumed by V3-M5). Spec author:
Fable. Implementer: coding sub-agent. Depends on: Phase 1 packets
(landed), 2b menus (landed). Model guidance: default reader model is
`gpt-5.6-luna` (comparison-proven value tier: 94.2%/77.2% Borg at $0.59);
model id must be configurable.

## 4.1 — `src/analysis/llm_reader.py`

Promote the core of
`scripts/research/copiale/rank_candidate_texts_with_llm.py` into a
library:

- `LLMReaderConfig`: provider/model (default openai/gpt-5.6-luna),
  max_candidates (default 12), max_chars_per_candidate (default 700),
  max_calls, temperature-free deterministic settings where the provider
  allows, language.
- `rank_candidates(candidates: list[CandidatePacket|dict], config)
  -> ReaderResult`: builds a strict-firewall prompt containing ONLY
  candidate ids + text excerpts (+ language name); returns structured
  per-candidate scores {readability 0–10, coherent_clause bool, short
  rationale ≤140 chars} plus a ranking, votes-style. Parse failures
  retry once, then mark unreadable candidates as unscored (never crash
  the caller).
- **Input firewall is a hard rule**: no ground truth, no solver scores,
  no filenames/test ids, no mask/edit metadata in the prompt. A helper
  `reader_prompt_for(candidates, config)` is exposed so tests can
  leak-check the exact prompt text with `assert_no_ground_truth_leak`.
- Provider access via `agent.model_provider.make_model_provider`; usage
  and estimated cost accumulated into `ReaderResult.usage`.
- Text sourcing from packets: `text` or `preview`; for word-repair
  packets (text=None) use the preview plus `extras["page_previews"]`
  when present — never the full key/provenance.

## 4.2 — Runner opt-in reranker

- Env `DECIPHER_FINALIST_READER` (e.g. `llm:gpt-5.6-luna`; unset = off;
  strict parse, fail-closed like DECIPHER_WORD_REPAIR_ADOPT). When on,
  after a finalist menu is built (null-mask bakeoff selection, and the
  word-repair `would_adopt` computation), rerank the top-N packets with
  the reader and record a `finalist_reader` block in the relevant step:
  per-candidate reader scores, the reader-preferred candidate, model,
  tokens, cost. **Selection behavior change is opt-in within the
  opt-in**: by default the reader ANNOTATES only; a second env
  `DECIPHER_FINALIST_READER_SELECTS=1` lets the reader's top pick
  override the scalar selection (null-mask route) — mirroring the
  menu-only lesson: measure first.
- Lazy imports; API-key resolution via the existing `cli.get_api_key`
  pathway — but the runner must degrade gracefully (log a step note,
  skip the reader) when no key/network is available; never fail the run.
- Firewall tests: leak-check the constructed prompts in a stubbed-reader
  run; assert the reader block never contains ground truth.

## 4.4 — Calibration report (compute, report)

Extend `scripts/research/copiale/rank_candidate_texts_with_llm.py` (or a
sibling `run_reader_calibration.py`) to drive the promoted library over
the existing saved null-mask probe rows
(`artifacts/copiale_evidence_packet/null_probe_pair_masks_full_rows.jsonl`
if present — report if absent) and the 2b acceptance menus
(`artifacts/phase2b_acceptance/`), comparing: scalar validation_score_v2
rank, LinearLanguageQualityModel rank (if a trained model file exists),
and reader rank, against post-hoc char accuracy (post-hoc = calibration
only). Output a per-page table: which selector picks the best candidate,
top-3 capture. Budget cap: ≤ $3 of luna calls total; report actual.

## Tests

Fake-provider reader tests (structured parse, retry-once, unscored
fallback); prompt firewall leak-check; budget caps enforced
(max_candidates/chars truncation visible in prompt helper output);
env parsing strict; runner integration with stubbed reader (annotate
mode: selection unchanged, block recorded; selects mode: selection
overridden, both recorded); graceful no-key degradation.

## Out of scope

Agent-side verify/compare episodes (V3-M5); changing any default
selection behavior; training/fine-tuned rankers.

## Review follow-ups (deferred; from the Fable review — LAND WITH FIXES applied)

- F4: `llm:PROVIDER:MODEL` with a typo'd provider degrades silently to
  anthropic inference instead of raising (kept for ollama colon-tag
  compatibility; block records resolved provider/model — diagnosable).
- F5: JSON fallback parser uses first-{ to last-} slicing; prose after
  the JSON containing a brace defeats it — harden with
  `json.JSONDecoder().raw_decode` when convenient.
- F6 nits: finalist_reader block presence semantics differ between
  null-mask (always) and word-repair (only when enabled); invalid-env
  error block missing some standard keys; calibration budget check is
  pre-call (can overshoot by one page); max_calls=0 silently becomes 1.

Calibration verdict recorded: luna reader = modest mixed improvement
over scalar (top-1 oracle 2/5 vs 1/5 probe rows; mean 70.3% vs 69.9% on
2b menus; wins p035/p052, regresses p084) — supports annotate-by-default,
revisit SELECTS after the reader sees multipage-quality candidates.

## Deliverables

Files changed, suite counts, the 4.4 calibration table + reader cost,
deviations. No commits.
