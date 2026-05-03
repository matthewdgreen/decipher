# Agent Tool Reference

Complete reference for the 85 tools exposed to the v2 agentic loop
(`src/agent/tools_v2.py`). Tools are organized by namespace. Each entry gives
the tool name, what it does, its key parameters, and usage notes.

---

## workspace_* — Branch and hypothesis management

### `workspace_fork`
Copy the key of an existing branch into a new branch.

| Parameter | Type | Notes |
|-----------|------|-------|
| `new_name` | string | **required** |
| `from_branch` | string | defaults to `main` |

*Prefer `workspace_fork_best` when an automated preflight branch exists — this
tool will accidentally fork empty `main` if you are not careful about the source.*

---

### `workspace_fork_best`
Fork from the strongest currently-scored branch (by mapped count / score).

| Parameter | Type | Notes |
|-----------|------|-------|
| `new_name` | string | **required** |
| `prefer_branch` | string | optional override; the tool otherwise picks the best branch |

*The result always tells you which source branch was copied.*

---

### `workspace_list_branches`
List all branches with their mapped-symbol counts and tags. No parameters.

---

### `workspace_branch_cards`
Show compact state cards for one or all branches: scores, mapped count, readable
excerpt, repair agenda items, and risk warnings.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | omit to show all branches |

*Use before declaration when multiple branches or repair hypotheses exist.*

---

### `workspace_create_hypothesis_branch`
Create a branch tagged with an explicit cipher-mode hypothesis without assuming
a substitution key.

| Parameter | Type | Notes |
|-----------|------|-------|
| `new_name` | string | **required** |
| `cipher_mode` | string | **required** — e.g. `periodic_polyalphabetic`, `transposition_homophonic` |
| `rationale` | string | **required** |
| `from_branch` | string | defaults to `main` |
| `evidence_source` | string | `agent_inference` (default), `benchmark_context`, `ciphertext_statistics`, `solver_result`, `related_record`, or `other` |
| `mode_confidence` | string | `low` / `medium` / `high` |

*Use this early on unknown ciphers to separate hypotheses cleanly. If you read
exposed benchmark context as identifying the cipher family, set
`evidence_source=benchmark_context`; that is the explicit handshake that lets the
executor enforce context-aware tool discipline.*

---

### `workspace_reject_hypothesis`
Mark a hypothesis branch as rejected or superseded and record the reason.
Context-supported and statistically required family-level tools may block
premature rejection. For example, a keyed-Vigenere/Kryptos context prior, or a
low-IC periodic signal after ordinary Vigenere failure, requires
`search_quagmire3_keyword_alphabet` before rejecting the broader
polyalphabetic family.

Context-family priors are executor-enforced only after the agent records the
context reading on a hypothesis branch with `evidence_source=benchmark_context`.
The executor does not parse prose context itself. Once the agent declares a
keyed-tableau/polyalphabetic context assumption, off-family search tools such as
transform+homophonic or homophonic annealing are blocked unless the agent
explicitly passes `override_context_cipher_family=true` and a concrete
`context_override_rationale`. Artifacts then record that benchmark context was
overridden.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `reason` | string | **required** |
| `status` | string | `rejected` (default) or `superseded` |
| `acknowledge_pending_required_tools` | boolean | emergency override only; explain why pending required tools are impossible or irrelevant |

---

### `workspace_update_hypothesis`
Update cipher-mode hypothesis metadata for an existing branch without changing
its key, token order, or decoded text.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `cipher_mode` | string | optional replacement mode label |
| `mode_status` | string | `active`, `paused`, `rejected`, or `superseded` |
| `mode_confidence` | string | `low`, `medium`, or `high` |
| `evidence` | string | supporting evidence to record |
| `counter_evidence` | string | evidence against the hypothesis |
| `next_recommended_action` | string | what should be tried next |
| `evidence_source` | string | optional replacement source; use `benchmark_context` only after reading exposed context |

*Use after diagnostics or solver attempts so the artifact records why the
agent kept, paused, or abandoned a cipher-mode hypothesis.*

---

### `workspace_hypothesis_cards`
Show all branches filtered as cipher-type hypotheses: active/rejected mode,
evidence, decoded preview, and suggested next action. No parameters.

*Use before switching cipher modes or declaring on an unknown cipher.*

---

### `workspace_hypothesis_next_steps`
Return a mode-specific diagnostic/solver playbook for one hypothesis branch or
all active hypothesis branches. The result marks each recommended tool as
`pending` or `tried` based on this run's tool log and highlights the next
pending action. It also returns a soft `tool_menu` with always-available,
foreground, escape, and discouraged tools for the active mode.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | optional; omit to show all active hypotheses |

*Use this when the agent has a cipher-mode hypothesis but needs a compact
reminder of the right tool sequence, the right foreground tools, or when it may
be looping on the same diagnostic.*

*Declaration guard: branches tagged as cipher-mode hypotheses must call this
before `meta_declare_solution`, so the artifact records tried and pending
mode-specific work.*

---

### `workspace_delete`
Delete a branch. Cannot delete `main`.

| Parameter | Type | Notes |
|-----------|------|-------|
| `name` | string | **required** |

---

### `workspace_compare`
Side-by-side comparison of two branches: mapping agreements, disagreements, and
both transcriptions.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch_a` | string | **required** |
| `branch_b` | string | **required** |

---

### `workspace_merge`
Merge mappings from one branch into another.

| Parameter | Type | Notes |
|-----------|------|-------|
| `from_branch` | string | **required** |
| `into_branch` | string | **required** |
| `policy` | string | `non_conflicting` (default), `override` (source wins), `keep` (dest wins) |

---

## observe_* — Statistical observation

### `observe_frequency`
Frequency analysis on the raw ciphertext (not branch-specific).

| Parameter | Type | Notes |
|-----------|------|-------|
| `ngram` | string | `mono` (default), `bigram`, `trigram` |
| `top_n` | integer | for bigram/trigram, how many to return (default 30) |

---

### `observe_isomorph_clusters`
Group cipher words by their isomorph pattern. Repeated patterns heavily
constrain the plaintext.

| Parameter | Type | Notes |
|-----------|------|-------|
| `min_length` | integer | minimum word length to include (default 2) |
| `min_occurrences` | integer | only show patterns occurring this many times (default 1) |

---

### `observe_ic`
Index of Coincidence for the raw ciphertext. No parameters.

*English monoalphabetic ~0.067; random/polyalphabetic ~0.038.*

---

### `observe_cipher_id`
Cheap unknown-cipher fingerprint for a branch: IC, normalized entropy, periodic
IC, Kasiski GCD hints, doubled-digraph rate, symbol counts, and ranked
cipher-mode suspicion scores.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `max_period` | integer | maximum key period to probe (default 26) |

*Use before committing to a solver family. Re-run after applying a transform to
see whether the underlying cipher type has changed.*

---

### `observe_cipher_shape`
Compact structural view of a branch before choosing a cipher mode: token count,
symbol inventory, boundary structure, repeated n-grams, pairability, and
coordinate-looking symbol hints.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `top_n` | integer | top repeated n-grams to show (default 12) |

---

### `observe_periodic_ic`
Detailed Friedman periodic-IC and Kasiski view for Vigenère-family diagnosis.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `max_period` | integer | default 26 |
| `top_n` | integer | top period candidates to return (default 10) |

*Use when `observe_cipher_id` flags polyalphabetic or when IC is depressed but
alphabet size is near 26.*

---

### `observe_kasiski`
Detailed Kasiski repeated-sequence spacing report for periodic polyalphabetic
diagnosis. Shows repeated n-grams, their positions, spacings between
occurrences, and per-factor/period support counts.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `min_ngram` | integer | minimum n-gram length to search (default 3) |
| `max_ngram` | integer | maximum n-gram length to search (default 5) |
| `max_period` | integer | largest period to tabulate (default 40) |
| `top_n` | integer | top period candidates to return (default 12) |

*Use alongside `observe_periodic_ic` to corroborate the period estimate — a
strong Kasiski GCD spike plus a Friedman IC peak at the same period is strong
evidence for Vigenère-family ciphers.*

---

### `observe_phase_frequency`
Show per-phase symbol frequency profiles for a proposed periodic key length.
Each of the `period` phases (columns of the strided partition) is shown as an
independent frequency histogram so per-column shift candidates are visible.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `period` | integer | period to inspect; defaults to branch key period or fingerprint best period |
| `top_n` | integer | top symbols per phase to display (default 8) |

*Use after `observe_periodic_ic` or `observe_kasiski` has suggested a plausible
period, before calling `observe_periodic_shift_candidates` or setting shifts
manually.*

---

### `observe_periodic_shift_candidates`
For a proposed period and cipher variant, rank the most likely Caesar shift for
each key phase using monogram chi-squared against the reference language
frequency table.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `period` | integer | period to inspect; defaults to branch key period or fingerprint best period |
| `variant` | string | `vigenere` (default), `beaufort`, `variant_beaufort`, or `gronsfeld` |
| `top_n` | integer | top shift candidates per phase (default 5) |
| `sample` | integer | token sample size for chi-squared (default 80) |

*Use before calling `act_set_periodic_key` or `act_adjust_periodic_key` to see
which shifts are statistically favoured. The `variant` parameter must match the
cipher family being tested — different variants apply the shift in opposite
directions.*

---

### `observe_homophone_distribution`
Homophonic-cipher diagnostic. Estimates expected cipher-symbol counts per
plaintext letter from reference language frequencies, and compares with the
current branch if supplied.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | optional — omit for the raw distribution estimate |

*Use when the cipher alphabet is larger than the plaintext alphabet, IC is very
low, or many decoded letters are absent or overrepresented.*

---

### `observe_transform_pipeline`
Inspect ciphertext-transform state for a branch: current grid metadata, active
token-order overlay, and the applied Zenith-compatible transform pipeline (if any).

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |

---

### `observe_transform_suspicion`
Cheap diagnostic for deciding whether transform search is worth trying: plausible
grid dimensions, homophonic/order-scramble signals, and a conservative
recommendation.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `columns` | integer | optional suspected grid width |
| `baseline_status` | string | optional status string from a prior baseline solve |
| `baseline_score` | number | optional normalized quality score from a prior baseline |

*Use before spending solver budget on `search_pure_transposition` or
`search_transform_homophonic`.*

---

## search_* — Solver invocation

### `search_hill_climb`
Greedy per-symbol key refinement.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `rounds` | integer | default 30 |
| `restarts` | integer | default 5 |
| `score_fn` | string | `dictionary`, `quadgram`, `combined` (default `quadgram`) |

**Use only after `search_anneal` has produced a good starting point.** Hill
climbing from empty/random starts frequently stalls around 40% accuracy.

---

### `search_anneal`
Simulated annealing — the primary search tool for monoalphabetic substitution.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `steps` | integer | steps per restart (default 5000) |
| `restarts` | integer | independent restarts; best result kept (default 3) |
| `t_start` | number | initial temperature (default 1.0) |
| `t_end` | number | final temperature (default 0.005) |
| `score_fn` | string | `dictionary`, `quadgram`, `combined` (default `combined`) |
| `preserve_existing` | boolean | preserve partial anchors; complete inherited keys restart from scratch |
| `override_context_cipher_family` | boolean | deliberate context-family override only |
| `context_override_rationale` | string | required when overriding exposed benchmark cipher-family context |

*Typically achieves 85%+ on English/Latin in one call. After annealing, read the
decoded text. If a few errors remain, call `decode_diagnose_and_fix` before
declaring.*

---

### `search_automated_solver`
Run Decipher's current best automated solver stack (including `zenith_native` for
homophonic routing) and install the result onto the named branch.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `homophonic_budget` | string | `full` (default) or `screen` |
| `homophonic_refinement` | string | `none` (default), `two_stage`, `targeted_repair`, `family_repair` |
| `homophonic_solver` | string | `zenith_native` (default) or `legacy` |
| `override_context_cipher_family` | boolean | deliberate context-family override only |
| `context_override_rationale` | string | required when overriding exposed benchmark cipher-family context |

*Mirrors the no-LLM automated runner used for frontier/parity evaluation.*

---

### `search_homophonic_anneal`
Purpose-built solver for homophonic no-boundary ciphers. Uses continuous 5-gram
scoring plus a global letter-distribution objective.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `solver_profile` | string | `zenith_native` (default) or `legacy` |
| `epochs` | integer | independent annealing epochs (default 5) |
| `sampler_iterations` | integer | iterations per epoch (default 2000) |
| `t_start` / `t_end` | number | temperature schedule (defaults 0.012 / 0.006) |
| `order` | integer | n-gram order: 3, 4, or 5 (default 5) |
| `preserve_existing` | boolean | treat existing mappings as fixed anchors (default false) |
| `model_path` | string | optional path to a Zenith-style n-gram model |
| `distribution_weight` | number | weight for global letter-distribution penalty (default 4.0) |
| `diversity_weight` | number | weight for diversity penalty on short texts (default 1.5) |
| `top_n` | integer | return top N distinct epoch candidates (default 1) |
| `write_candidate_branches` | boolean | write non-best candidates to sibling branches |
| `override_context_cipher_family` | boolean | deliberate context-family override only |
| `context_override_rationale` | string | required when overriding exposed benchmark cipher-family context |

*Prefer this over `search_anneal` for hardest/no-boundary homophonic tests.*

---

### `search_transform_candidates`
Run a structural-only transform candidate search without spending homophonic
solver budget.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `columns` | integer | optional suspected grid width |
| `breadth` | string | `fast`, `broad` (default), or `wide` |
| `top_n` | integer | default 40 |
| `max_generated_candidates` | integer | safety cap (default 25000) |
| `include_program_search` | boolean | also run beam-search program composition (default false) |
| `program_max_depth` | integer | beam-search depth (default 5) |
| `program_beam_width` | integer | beam width (default 48) |
| `override_context_cipher_family` | boolean | deliberate context-family override only |
| `context_override_rationale` | string | required when overriding exposed benchmark cipher-family context |

*Use to decide whether a full `search_transform_homophonic` run is warranted.*

---

### `search_pure_transposition`
Run the Rust-backed pure-transposition screen and rank candidate reading orders
directly by plaintext quality. This is for transposition-only hypotheses such as
Kryptos K3-style matrix/route scrambling, not Zodiac/Z340-style
transposition+homophonic search.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `profile` | string | `small`, `medium`, or `wide` (default) |
| `top_n` | integer | ranked candidates to return (default 10) |
| `install_top_n` | integer | top candidates to install as readable branches (default 1) |
| `new_branch_prefix` | string | branch prefix (default `trans`) |
| `max_candidates` | integer | optional hard cap |
| `include_matrix_rotate` | boolean | include direct MatrixRotate candidates (default true) |
| `include_transmatrix` | boolean | include K3-style TransMatrix candidates (default true) |
| `transmatrix_min_width` | integer | default 2 |
| `transmatrix_max_width` | integer | optional maximum width |
| `threads` | integer | Rust worker count; 0 means auto-size |
| `override_context_cipher_family` | boolean | deliberate context-family override only |
| `context_override_rationale` | string | required when overriding exposed benchmark cipher-family context |

Installed branches carry a transform pipeline plus `decoded_text` metadata, so
use `workspace_branch_cards` or `decode_show` to read the candidate before
declaring.

The returned `search_session_id` lets the agent page through finalists, rate
readability, and install selected ranks later without rerunning the Rust screen.
Use `search_review_pure_transposition_finalists`,
`act_rate_transform_finalist`, and `act_install_pure_transposition_finalists`
when the top candidate is not obviously coherent.

Pure-transposition results include a `validation` block and
`validated_selection_score`. The validator oversamples the Rust n-gram finalist
pool, then reranks with stricter evidence: direct n-gram score, DP segmentation
quality, pseudo-word burden, and continuous dictionary hits of length four or
more. The validation block also includes `integrity`, a local-damage subscore
that distinguishes clean/canonical-looking text from readable candidates with
short pseudo-word scars or larger unexplained fragments. Treat this as
supporting evidence for coherence and cleanliness; the agent's contextual
readability judgment remains the primary signal when context is available.

The screen includes grid/route/columnar families, direct `MatrixRotate`, and
optional K3-style `TransMatrix` candidates. Identical repeated calls are cached
internally and return `cache.hit=true`; broader follow-up searches still need
to be run explicitly.

---

### `search_transform_homophonic`
Try a bounded set of transform candidates: apply each transform, run a short
homophonic search, rank by quality, and optionally install the best branch.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `columns` | integer | optional suspected grid width |
| `profile` | string | `small` (default), `medium`, or `wide` |
| `top_n` | integer | finalist count to return (default 3) |
| `max_generated_candidates` | integer | safety cap |
| `write_best_branch` | boolean | install best result as a branch (default true) |
| `write_candidate_branches` | boolean | also write top-N finalist branches (default false) |
| `candidate_branch_count` | integer | how many finalist branches to write (default 3) |
| `review_chars` | integer | max decoded preview chars per finalist (default 600) |
| `homophonic_budget` | string | `screen` (default) or `full` |
| `include_program_search` | boolean | also compose small pipelines from grammar (default false) |
| `override_context_cipher_family` | boolean | deliberate context-family override only |
| `context_override_rationale` | string | required when overriding exposed benchmark cipher-family context |

---

### `search_review_transform_finalists`
Page through finalists from a prior `search_transform_homophonic` session without
rerunning the search.

| Parameter | Type | Notes |
|-----------|------|-------|
| `search_session_id` | string | **required** — from a prior search |
| `start_rank` | integer | first rank to show (default 1) |
| `count` | integer | how many to show (default 5) |
| `review_chars` | integer | max preview chars per finalist (default 600) |
| `good_score_gap` | number | finalists within this gap are "good" (default 0.25) |

---

### `search_review_pure_transposition_finalists`
Page through finalists from a prior `search_pure_transposition` session without
rerunning the Rust screen. This is the pure order-only counterpart to
`search_review_transform_finalists`: it returns plaintext previews, direct
language scores, candidate family/parameters, validator evidence, and an
explicit agent-readability slot.

| Parameter | Type | Notes |
|-----------|------|-------|
| `search_session_id` | string | **required** — from a prior `search_pure_transposition` call |
| `start_rank` | integer | first rank to show (default 1) |
| `count` | integer | how many to show (default 5) |
| `review_chars` | integer | max preview chars per finalist (default 600) |
| `good_score_gap` | number | finalists within this score gap are "good" (default 0.25) |

Use this before installing branches when the best pure-transposition candidate
is not obviously coherent. If the previews are only word islands and the source
alphabet suggests a separate keying layer, switch to
`search_transform_homophonic` rather than doing local word repair.

---

### `search_periodic_polyalphabetic`
Vigenère-family search using Friedman + Kasiski period estimation and per-column
chi² frequency analysis. Installs the best candidates as mode-tagged hypothesis
branches.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `max_period` | integer | default 20 |
| `periods` | array of int | optional exact periods to test instead of searching |
| `variants` | array of string | `vigenere`, `beaufort`, `variant_beaufort`, `gronsfeld` |
| `top_n` | integer | candidates to return (default 5) |
| `install_top_n` | integer | candidates to install as branches (default 1; 0 = screen only) |
| `new_branch_prefix` | string | branch name prefix (default `poly`) |

*The periodic key is stored in branch metadata, not the substitution key.*
This tool covers ordinary A-Z periodic-shift families. Incoherent output from
this tool does not reject keyed-tableau/Kryptos/Quagmire-style ciphers; use
`search_quagmire3_keyword_alphabet` next when context or statistics still
support the polyalphabetic family.

---

### `search_quagmire3_keyword_alphabet`
Search Quagmire III keyword-shaped shared alphabets, derive cyclewords for
each candidate, and optionally install the best candidates as decoded
hypothesis branches.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | defaults to `main` |
| `keyword_lengths` | array of int | candidate alphabet-keyword lengths |
| `cycleword_lengths` | array of int | candidate periodic cycleword lengths |
| `initial_keywords` | array of string | optional context/crib starts; makes the run seeded, not blind |
| `engine` | string | `rust_shotgun` for the compiled parallel Blake-style restart/hillclimb loop; `python_screen` only for small reference/diagnostic probes |
| `steps` | integer | Python-screen keyword mutation steps per start (default 200) |
| `hillclimbs` | integer | Rust-shotgun proposals per restart (default 500) |
| `restarts` | integer | random starts per keyword length (default 8) |
| `threads` | integer | Rust-shotgun worker threads; `0` means all available cores |
| `seed` | integer | deterministic RNG seed (default 1) |
| `top_n` | integer | candidates to return (default 5) |
| `install_top_n` | integer | candidates to install as branches (default 1; 0 = screen only) |
| `estimate_only` | boolean | return budget/runtime estimates without running search; recommended before broad `rust_shotgun` runs |
| `screen_top_n` | integer | finalists to refine after screening (default 64) |
| `word_weight` | number | dictionary-word bonus in selection score (default 0.25) |
| `slip_probability` | number | Rust/Python search escape probability for accepting worse moves |
| `backtrack_probability` | number | Rust/Python search probability for returning to the local best state |
| `dictionary_keyword_limit` | integer | add first N dictionary keyword starts (default 0) |
| `new_branch_prefix` | string | branch name prefix (default `quag3`) |

*The installed branch stores a `QuagmireKey` in metadata, including
`alphabet_keyword`, `cycleword`, decoded text, and whether explicit initial
keywords seeded the search. Use this after ordinary Vigenère-family candidates
fail but periodic evidence remains. The Rust engine is intended for broad
blind/search-budget runs; the Python engine remains useful for small diagnostic
screens and parity comparisons. If `rust_shotgun` is unavailable, run
`PYTHONPATH=src .venv/bin/decipher doctor` for the exact local build command.
Do not treat `python_screen` as an equivalent runtime path for a large
`rust_shotgun` run; downsize parameters and label it as a reference/diagnostic
probe. For Rust runs, nominal proposals are
`len(keyword_lengths) * len(cycleword_lengths) * restarts * hillclimbs`; use
`estimate_only=true` to size the run before spending CPU time. Passing likely
title/source keywords through `initial_keywords` makes the result
context-seeded, not blind key recovery.*

---

## decode_* — Decryption display and diagnosis

### `decode_show`
Show the current branch transcription as paired cipher/decoded rows, word by word.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `start_word` | integer | first word to show (default 0) |
| `count` | integer | max words (capped at 50, default 25) |

---

### `decode_show_phases`
For a periodic polyalphabetic branch, show ciphertext grouped by key phase with
shifts, key letters/digits, decoded samples, and top phase-local symbols.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `period` | integer | optional override; defaults to branch metadata |
| `variant` | string | `vigenere`, `beaufort`, `variant_beaufort`, `gronsfeld` |
| `sample` | integer | samples per phase (default 24) |

*Use to inspect or manually adjust a Vigenère/Beaufort/Gronsfeld candidate.*

---

### `decode_unmapped_report`
Show which cipher symbols and plaintext letters are currently unmapped on a branch.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |

---

### `decode_ngram_heatmap`
Per-position n-gram log-probability across the transcription. Returns the K
worst-scoring n-grams with character indices.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `n` | integer | 2, 3, or 4 (default 4) |
| `worst_k` | integer | how many worst n-grams to return (default 20) |

*Use to locate regions of low language-model likelihood for targeted repair.*

---

### `decode_letter_stats`
Show the decoded letter-frequency distribution and compare it to the target
language reference. Highlights absent letters and badly overrepresented or
underrepresented ones.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |

*The fastest way to see which decoded letters are suspicious without running Python.*

---

### `decode_ambiguous_letter`
When a decoded letter appears to stand for multiple true plaintext letters in
different contexts, show which cipher symbols currently produce it and sample
contexts for each.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `decoded_letter` | string | **required** — e.g. `I` |
| `context` | integer | characters either side (default 6) |
| `max_contexts_per_symbol` | integer | context examples per cipher symbol (default 12) |

*Use before changing an overused decoded letter; identifies which specific cipher
symbol to target with `act_set_mapping`.*

---

### `decode_absent_letter_candidates`
When a plaintext letter is absent or underrepresented in a homophonic/no-boundary
decode, rank cipher symbols from overrepresented decoded letters as candidates for
the missing letter. Returns contexts and score deltas for each candidate.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `missing_letter` | string | **required** — e.g. `U` |
| `source_letters` | array of string | optional decoded letters to inspect as sources |
| `context` | integer | default 6 |
| `max_candidates` | integer | default 12 |
| `max_contexts_per_symbol` | integer | default 6 |

---

### `decode_diagnose`
Analyse a branch's decoded text and rank likely residual single-letter errors.
Runs DP word segmentation, finds pseudo-words, and searches the dictionary for
same-length words at edit distance 1.

Returns: `cipher_symbols_for_wrong`, `culprit_symbol`, `suggested_call`
(the exact `act_set_mapping` call to fix it), and `bulk_fix_call` (a single
`decode_diagnose_and_fix` call for all top candidates at once).

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `top_k` | integer | top corrections to return (default 5) |

*Call after `search_anneal` has converged but a few errors remain.*

---

### `decode_diagnose_and_fix`
Diagnose residual errors AND apply all high-confidence fixes in a single call.
Runs the same analysis as `decode_diagnose`, then tests each candidate with
`act_set_mapping` on the culprit symbol, reverting those that worsen the branch.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `top_k` | integer | max fixes to consider (default 5) |
| `min_evidence` | integer | minimum evidence count to auto-apply a fix (default 2) |
| `auto_revert_if_worse` | boolean | revert fixes that worsen the branch (default true) |

Returns: combined `score_delta`, `dict_rate_after`, `pseudo_words_remaining`,
and a recommendation (declare now vs. run again).

*Replaces the decode_diagnose → many-act_set_mapping loop with one tool call.*

---

### `decode_repair_no_boundary`
Conservative text-only repair pass for no-boundary or boundary-drifted output.
Segments the decoded plaintext, applies confident one-edit word repairs, and
tries local re-segmentation over suspicious windows. **Does not mutate the branch
key** — returns a repaired plaintext candidate for human review.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `max_rounds` | integer | repair rounds (default 3) |
| `max_window_words` | integer | adjacent word window for re-segmentation (default 3) |

---

### `decode_validate_reading_repair`
Read-only validation of a proposed best reading against the branch's current
decoded character stream. Reports whether the proposal is character-preserving,
shows first mismatch spans, and scores proposed words against the dictionary.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `proposed_text` | string | **required** — your best target-language reading |

*If character-preserving → apply with `act_resegment_by_reading`. If same
character count but different letters → apply boundary pattern with
`act_resegment_from_reading_repair`, then repair mismatches with `act_set_mapping`.*

---

### `decode_plan_word_repair`
Plan a reading-driven word repair (e.g. TREUITER → BREUITER) without mutating
the branch. Identifies responsible cipher symbol changes, returns a
`changed_words` preview, and an `act_apply_word_repair` suggested call.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `target_word` | string | **required** — intended reading |
| `cipher_word_index` | integer | optional numeric word index |
| `decoded_word` | string | optional current decoded word to locate |
| `occurrence` | integer | which matching occurrence (default 0) |

---

### `decode_plan_word_repair_menu`
Compare several possible same-length readings for the same decoded word without
mutating the branch. Shows each option's proposed cipher-symbol mappings,
intra-word conflicts, changed-word preview, dictionary-hit delta, collateral-change
count, and suggested `act_apply_word_repair` call.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `target_words` | array of string | **required** — candidate readings to compare |
| `cipher_word_index` | integer | optional numeric word index |
| `decoded_word` | string | optional current decoded word to locate |
| `occurrence` | integer | default 0 |

*Use before applying an uncertain word repair: choose from evidence rather than
applying the first plausible word.*

---

## score_* — Scoring

### `score_panel`
Full multi-signal scoring panel for a branch: `dictionary_rate`,
`quadgram_loglik_per_gram`, `bigram_loglik_per_gram`, `bigram_chi2`,
`pattern_consistency`, `constraint_satisfaction`, and mapped counts.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |

*`dictionary_rate` works for both word-boundary and no-boundary ciphers — for
no-boundary text it automatically segments before scoring.*

---

### `score_quadgram`
Mean log₁₀ quadgram probability of the branch's transcription.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |

---

### `score_dictionary`
Dictionary hit-rate of the branch's transcription, plus a sample of unrecognized
words. For no-boundary ciphers, automatically segments and returns a segmented
preview.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |

---

## corpus_* — Dictionary and corpus lookup

### `corpus_lookup_word`
Check whether a word is in the target-language wordlist, and if so its frequency rank.

| Parameter | Type | Notes |
|-----------|------|-------|
| `word` | string | **required** |

---

### `corpus_word_candidates`
Candidate plaintext words matching an encoded word's isomorph pattern. Optionally
filter to candidates compatible with the current branch's mappings.

| Parameter | Type | Notes |
|-----------|------|-------|
| `cipher_word_index` | integer | **required** |
| `consistent_with_branch` | string | optional branch to check consistency against |
| `limit` | integer | default 20 |

---

## act_* — Key mutations and structural edits

### `act_set_mapping`
Set a single cipher-symbol → plaintext-letter mapping on a branch.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `cipher_symbol` | string | **required** — the cipher symbol name, e.g. `S001`, `A` |
| `plain_letter` | string | **required** |
| `dry_run` | boolean | preview the mapping without mutating (default false) |
| `allow_mode_mismatch_repair` | boolean | override the non-substitution mode guard |

Returns `changed_words` (was → now) so you can judge by reading. Score delta is
advisory.

**This is the default primitive for reading-driven repair.** Surgical and
unidirectional — only words containing the target cipher symbol change.
Branches tagged as periodic, transform, fractionation, or polygraphic
hypotheses block this by default; override only when deliberately abandoning or
crossing that mode assumption.

---

### `act_set_periodic_key`
Set the full periodic key for a periodic polyalphabetic branch (e.g. Vigenère key
string LEMON, or a list of shifts). Updates mode-specific metadata and decoded
text, not `branch.key`.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `key` | string | key string, e.g. `LEMON` |
| `shifts` | array of int | alternative: pass shifts directly |
| `variant` | string | `vigenere` (default), `beaufort`, `variant_beaufort`, `gronsfeld` |

---

### `act_set_periodic_shift`
Set one phase shift in a periodic polyalphabetic branch.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `phase` | integer | **required** — zero-based phase index |
| `shift` | integer | **required** |
| `variant` | string | variant cipher type |

---

### `act_adjust_periodic_shift`
Increment or decrement one phase shift by a delta.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `phase` | integer | **required** |
| `delta` | integer | default +1 |
| `variant` | string | variant cipher type |

---

### `act_bulk_set`
Set multiple symbol → letter mappings in one call.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `mappings` | object | **required** — e.g. `{"S001": "E", "S002": "T"}` |
| `allow_mode_mismatch_repair` | boolean | override the non-substitution mode guard |

---

### `act_anchor_word`
Assert that an encoded word decodes to a specific plaintext word on a branch.
Directly assigns every cipher symbol to its corresponding plaintext letter.
Supports homophonic ciphers (multiple cipher symbols → same letter).

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `cipher_word_index` | integer | **required** |
| `plaintext` | string | **required** |
| `allow_mode_mismatch_repair` | boolean | override the non-substitution mode guard |

---

### `act_clear_mapping`
Remove the mapping for a single cipher symbol on a branch.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `cipher_symbol` | string | **required** |

---

### `act_swap_decoded`
Bidirectional population swap — exchanges all cipher symbols mapped to decoded
letter A with all cipher symbols mapped to decoded letter B across the entire
branch.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `letter_a` | string | **required** |
| `letter_b` | string | **required** |
| `auto_revert_if_worse` | boolean | default true |

**Use only for deliberate whole-population swaps (e.g. confirmed A↔E swap).**
For "this letter should be different in this word", use `act_set_mapping` on the
specific cipher symbol instead. When auto-reverted, returns
`unidirectional_alternatives` showing the targeted `act_set_mapping` calls that
would have achieved the intent without the bidirectional side-effects.

---

### `act_split_cipher_word`
Split one ciphertext word into two words on this branch, at a token offset.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `cipher_word_index` | integer | **required** |
| `split_at_token_offset` | integer | **required** — offset inside the word where the new word begins |

---

### `act_merge_cipher_words`
Merge two adjacent ciphertext words (by numeric word index) on this branch.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `left_word_index` | integer | **required** |

*Numeric word indices shift after every split/merge. When acting from decoded text,
prefer `act_merge_decoded_words` or re-run `decode_show` to confirm the current index.*

---

### `act_merge_decoded_words`
Merge the adjacent cipher words whose current decoded forms match `left_decoded`
and `right_decoded`. Safer than numeric-index merging after prior edits.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `left_decoded` | string | **required** — current decoded form of the left word |
| `right_decoded` | string | **required** — current decoded form of the right word |
| `occurrence` | integer | which matching adjacent pair to merge (default 0) |

---

### `act_apply_boundary_candidate`
Apply one of the boundary edits currently suggested for this branch. Recomputes
candidates on each call so indices remain stable across prior edits.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `candidate_index` | integer | index into the current `boundary_candidates` list (default 0) |

*Use when `decode_diagnose` or `decode_diagnose_and_fix` returns boundary
candidates.*

---

### `act_apply_word_repair`
Apply a same-length reading-driven word repair (e.g. TREUITER → BREUITER).
Locates the word, identifies differing cipher symbols, applies the mappings, and
returns `changed_words` for reading-based verification.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `target_word` | string | **required** — intended reading |
| `cipher_word_index` | integer | optional numeric word index |
| `decoded_word` | string | optional current decoded word to locate |
| `occurrence` | integer | default 0 |
| `dry_run` | boolean | preview without mutating (default false) |
| `allow_bad_basin_repair` | boolean | override the bad-basin guard (default false) |
| `allow_mode_mismatch_repair` | boolean | override the non-substitution mode guard |

---

### `act_resegment_by_reading`
Replace the branch's word boundaries with a complete proposed reading, while
preserving the exact decoded character stream (letter-for-letter). One-shot
boundary normalization.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `proposed_text` | string | **required** — complete reading with desired word boundaries |

*The letters must match exactly after removing spaces/punctuation. To also change
letters, first use `decode_validate_reading_repair`, then `act_set_mapping` for
the character repairs.*

---

### `act_resegment_from_reading_repair`
Apply only the word-boundary pattern implied by a proposed reading, while
preserving the branch's current decoded letters and key. Returns mismatch spans
for follow-up letter repairs.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `proposed_text` | string | **required** — proposed reading (letters may differ if character count matches) |

*Use when the branch is readable-but-damaged and your reading changes both spaces
and a few letters.*

---

### `act_resegment_window_by_reading`
Apply word-boundary changes to a local window of decoded words instead of
rewriting the entire stream.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `start_word_index` | integer | **required** — first decoded word in the window |
| `word_count` | integer | **required** — number of decoded words in the window |
| `proposed_text` | string | **required** — desired reading/boundaries for the window |

*Use for local repairs like LIBE\|BITUR → LIBEBITUR without touching the rest of
the plaintext.*

---

### `act_apply_transform_pipeline`
Apply a Zenith-compatible ciphertext transform pipeline to a branch's reading
order. Changes the token order so subsequent decode/search tools operate on the
transformed ciphertext; does not change symbol mappings.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `pipeline` | object | **required** — `{columns, rows, steps: [{name, data}]}` |
| `override_context_cipher_family` | boolean | deliberate context-family override only |
| `context_override_rationale` | string | required when overriding exposed benchmark cipher-family context |

---

### `act_install_transform_finalists`
Install selected finalists from a prior `search_transform_homophonic` session as
workspace branches, by 1-based rank, without rerunning the search.

| Parameter | Type | Notes |
|-----------|------|-------|
| `search_session_id` | string | **required** |
| `ranks` | array of int | **required** — 1-based finalist ranks to install |
| `branch_prefix` | string | optional prefix; defaults to `<source_branch>_transform_rank` |
| `override_context_cipher_family` | boolean | deliberate context-family override only |
| `context_override_rationale` | string | required when overriding exposed benchmark cipher-family context |

---

### `act_install_pure_transposition_finalists`
Install selected finalists from a prior `search_pure_transposition` session as
readable transform branches, by 1-based rank, without rerunning the Rust screen.

| Parameter | Type | Notes |
|-----------|------|-------|
| `search_session_id` | string | **required** |
| `ranks` | array of int | **required** — 1-based finalist ranks to install |
| `branch_prefix` | string | optional prefix; defaults to `<source_branch>_pure_rank` |
| `override_context_cipher_family` | boolean | deliberate context-family override only |
| `context_override_rationale` | string | required when overriding exposed benchmark cipher-family context |

Installed branches receive `pure_transposition_finalist` metadata, preserve the
transform pipeline, and carry `decoded_text` so branch cards and decode views can
read the candidate directly.

---

### `act_rate_transform_finalist`
Record the agent's contextual readability judgment for one transform-search
finalist (0 = garbage, 1 = word islands, 2 = islands with structure, 3 = partial
coherent clause, 4 = coherent plaintext). This is the primary ranking signal for
transform finalists. It works for both `search_transform_homophonic` and
`search_pure_transposition` sessions.

| Parameter | Type | Notes |
|-----------|------|-------|
| `search_session_id` | string | **required** |
| `rank` | integer | **required** — 1-based finalist rank |
| `readability_score` | number | **required** — 0–4 |
| `label` | string | **required** — `coherent_plaintext`, `partial_clause`, `word_islands_with_some_structure`, `word_islands_only`, `garbage` |
| `rationale` | string | **required** — quote or paraphrase what it appears to say |
| `coherent_clause` | string | optional paraphrasable clause if one exists |

---

## repair_agenda_* — Durable reading-repair bookkeeping

### `repair_agenda_list`
List durable reading-repair agenda items accumulated during this run.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | optional branch filter |
| `status` | string | optional status filter: `open`, `applied`, `held`, `rejected`, `blocked` |

*Use before declaring if you have been making or considering word-level repairs.*

---

### `repair_agenda_update`
Update the status or notes for a durable repair agenda item.

| Parameter | Type | Notes |
|-----------|------|-------|
| `item_id` | integer | **required** |
| `status` | string | **required** — `open`, `applied`, `held`, `rejected`, `blocked`, `reverted` |
| `notes` | string | optional notes |

---

## inspect_* / list_* — Benchmark context examination

These tools expose only manifest-declared context for the current benchmark test;
they do not read arbitrary filesystem paths.

### `inspect_benchmark_context`
Inspect the scoped benchmark context for this run: selected policy, injected
layers, target/context records, related records, and associated documents.

| Parameter | Type | Notes |
|-----------|------|-------|
| `include_layer_text` | boolean | include injected context layer text (default true) |

---

### `list_related_records`
List benchmark records explicitly allowed as related/context material for this
test. No parameters.

---

### `inspect_related_transcription`
Read the canonical transcription of an allowed related record.

| Parameter | Type | Notes |
|-----------|------|-------|
| `record_id` | string | **required** |
| `max_chars` | integer | default 6000 |

---

### `inspect_related_solution`
Read plaintext/solution text for an allowed related record — only when the
benchmark context policy explicitly permits solution-bearing related context
(controlled ablation runs, not blind parity).

| Parameter | Type | Notes |
|-----------|------|-------|
| `record_id` | string | **required** |
| `max_chars` | integer | default 6000 |

---

### `list_associated_documents`
List long-form documents associated with the current benchmark record (letters,
plaintext notes, envelopes, source commentary). No parameters.

---

### `inspect_associated_document`
Read an associated benchmark document by document ID.

| Parameter | Type | Notes |
|-----------|------|-------|
| `document_id` | string | **required** |
| `max_chars` | integer | default 6000 |

---

## run_python — Escape hatch

### `run_python`
Execute Python 3 stdlib code and return stdout/stderr. Timeout: 15 s.

| Parameter | Type | Notes |
|-----------|------|-------|
| `justification` | string | **required** — the question being answered, why first-class tools are insufficient, and what dedicated tool would avoid this call |
| `code` | string | **required** — Python 3 code; print results to stdout |

Every use is logged as evidence of a potential tool-design gap. If the same
computation is needed repeatedly, also call `meta_request_tool` to document it.

---

## meta_* — Run control

### `meta_request_tool`
Document a capability gap: describe a computation or lookup that no available
tool covers well. These requests appear prominently in benchmark reports.

| Parameter | Type | Notes |
|-----------|------|-------|
| `description` | string | **required** — what you need to compute |
| `rationale` | string | **required** — why existing tools don't cover it |
| `example_input` | string | optional |
| `example_output` | string | optional |

*Does not execute code — use `run_python` as a workaround first.*

---

### `meta_declare_solution`
Terminate the run and submit the named branch as a plausible final answer.

| Parameter | Type | Notes |
|-----------|------|-------|
| `branch` | string | **required** |
| `rationale` | string | **required** — reasoning and remaining uncertainty |
| `self_confidence` | number | **required** — 0.0–1.0 |
| `reading_summary` | string | **required** — plain-language summary of what the decipherment says |
| `further_iterations_helpful` | boolean | **required** — would more iterations likely improve the result? |
| `further_iterations_note` | string | **required** — what further iterations should try, or why they are not needed |
| `forced_partial` | boolean | intentionally submitting a partial hypothesis (default false) |

*Call this only for a genuinely plausible decipherment. If no branch is
coherent after the relevant tools have been tried, use `meta_declare_unsolved`
instead of labeling gibberish as solved. High-confidence periodic/Quagmire
branches that store a continuous decoded stream in metadata are blocked until
you install a word-boundary overlay with `act_resegment_by_reading`, unless you
are explicitly submitting a forced partial.*

---

### `meta_declare_unsolved`
Terminate the run as honestly unsolved/exhausted without submitting a bogus
solution.

| Parameter | Type | Notes |
|-----------|------|-------|
| `rationale` | string | **required** — why stopping is justified |
| `branches_considered` | array[string] | **required** — branches or hypotheses reviewed |
| `best_branch` | string | optional best branch for later inspection |
| `reading_summary` | string | **required** — what was learned, if anything |
| `further_iterations_helpful` | boolean | **required** |
| `further_iterations_note` | string | **required** |

*The executor may block this while active cipher-family hypotheses still have
required higher-level work pending and there is iteration budget left.*
