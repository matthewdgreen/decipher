# Prompt Reduction Strategy

Status: **Fully implemented** (2026-05-03). Changes #1, #2/#6 (compact style),
#3, #4, #5, #5b, and #5c are landed. Only #7 is deferred.

Originally identified when qwen2.5:32b on Ollama timed out on a Zodiac 408
run (prompt 23,314 tokens vs. Ollama's default 4,096-token context window).

---

## Why This Exists

Decipher's agent loop currently sends ~27,000 tokens of prompt on every
turn before the model produces a single output token. That number is
roughly stable across cipher cases because the bulk of it comes from
infrastructure (system prompt + 85 tool schemas), not from the cipher
text itself.

This matters for three reasons:

1. **Local models with Ollama.** Most Ollama model defaults cap context
   at 4,096–8,192 tokens. Anything beyond that is silently truncated,
   often catastrophically (the model receives the *tail* of the prompt
   and loses the system prompt and tool definitions at the head).
2. **Per-token cost on hosted models.** With Anthropic prompt caching,
   the first turn pays the full input cost; later turns pay the cache-read
   rate. Bigger prompts → more cache cost on every cipher.
3. **Latency.** Prefill on a 27 K-token prompt is the single biggest
   chunk of wall-clock time per turn, especially on local hardware.

---

## Token Budget Today

Measured from a real run (`artifacts/zodiac_408_global_parity/ac10e69b88a6.json`):

| Component                            |   Chars   | ≈ Tokens |   %    |
| ------------------------------------ | --------: | -------: | -----: |
| **Tool schemas (85 tools)**          |   66,948  |  ~16,700 | **62%** |
| **System prompt** (`get_system_prompt("en")`) | 31,034 | ~7,800 | 29% |
| User message (this run)              |    9,595  |   ~2,400 |     9% |
| **Total**                            | ~107,000  |  ~27,000 |        |

Within the system prompt:

| Section                                         | Chars  | Tokens |
| ----------------------------------------------- | -----: | -----: |
| Your toolkit (by namespace)                     |  8,485 | ~2,100 |
| Reading-driven repair — your highest-leverage move |  9,919 | ~2,500 |
| Your primary judgement instrument               |  6,191 | ~1,500 |
| Scoring notes                                   |  2,535 |   ~635 |
| How you're expected to work                     |  2,115 |   ~530 |
| Other (env, notation, opening, caveats)         |  1,800 |   ~450 |

Within the user message:

| Section                                  | Chars |
| ---------------------------------------- | ----: |
| Cipher text (Z-notation)                 | 2,079 |
| Automated native solver preflight        | 2,842 |
| Benchmark context (3 layers)             | 2,109 |
| Cipher-type fingerprint                  | 1,446 |
| Measured facts                           |   401 |
| Other section headers + framing          |   718 |

The single biggest individual tool definitions:

| Tool                              | Chars | Description chars |
| --------------------------------- | ----: | ----------------: |
| `search_homophonic_anneal`        | 3,009 |               611 |
| `search_quagmire3_keyword_alphabet` | 2,383 |             311 |
| `search_transform_homophonic`     | 2,180 |               312 |
| `search_anneal`                   | 2,031 |               802 |
| `search_pure_transposition`       | 1,916 |               366 |
| `meta_declare_solution`           | 1,763 |               396 |
| `act_set_mapping`                 | 1,407 |               714 |
| `act_swap_decoded`                | 1,369 |               921 |
| `decode_diagnose`                 | 1,298 |             1,034 |

---

## Opportunities, Ranked by Impact

### 1. Mode-driven tool filtering — ✅ IMPLEMENTED

**Measured savings (verified with `scripts/count_prompt_tokens.py`, 2026-05-03):**
- Zodiac 408 (homophonic 54-symbol, only transposition=0.00 excluded):
  3 tools hidden → **~820 tokens saved** (3%)
- Borg-style monoalphabetic 33-symbol (polyalphabetic_vigenere and transposition excluded):
  13 tools hidden → **~2,654 tokens saved** (10%)

The original "4,900 token" figure assumed all cipher families except
`monoalphabetic_substitution` would score < 0.15. In practice for Borg
(33-symbol Latin cipher), `homophonic_substitution` also scores ≥ 0.15
(the 33-symbol alphabet looks partially homophonic), keeping homophonic tools
active. The correct picture is ~2,654 tokens saved, not 4,900.

**Fingerprint fix applied (2026-05-03):** The Vigenère suspicion score
previously used a raw `ic_delta` that caused a false positive for
large-alphabet monoalphabetic ciphers (e.g. Borg's IC = 0.057 vs. Latin
reference 0.0737 triggered the "IC depressed" signal). Fixed by:
1. Normalising IC relative to `random_IC(alphabet_size)` so oversized alphabets
   are not penalised (Borg goes from vig=0.80 → vig=0.05).
2. Gating the periodic IC boost on a minimum column size (≥ 25 tokens/column);
   spurious peaks from short texts (e.g. period-19 with 14 tokens/column) no
   longer contribute.

**What was built** (`src/agent/loop_v2.py`):
- `_TOOL_MODE_TAGS` maps each mode-specific tool to its cipher family.
  Unmapped tools (workspace, observe, decode, score, corpus, meta, run_python)
  are always included.
- `_active_modes_from_suspicion()` returns modes with score ≥ 0.15.
- `_apply_mode_filter()` keeps untagged (always-on) tools plus tools
  whose tags intersect the active-mode set.
- Filter computed once before iteration 1 and reused — keeps the tool
  list stable across turns so Anthropic prompt caching is not broken.
- Short ciphers (< 30 tokens) bypass filtering (fingerprint unreliable).
- Hidden tool names surfaced as a one-line note appended to `## Your
  Workspace` in the initial context.

**Safety:** `meta_request_tool` is always present. If the fingerprint
is wrong, the agent can request any filtered tool by name.

**Token comparison CLI** (`scripts/count_prompt_tokens.py`):
```bash
# Zodiac 408 (homophonic)
PYTHONPATH=src python scripts/count_prompt_tokens.py \
  --test-id parity_tool_zenith_zodiac408 \
  --split ~/Dropbox/src2/cipher_benchmark/benchmark/splits/parity_zodiac.jsonl \
  --benchmark-root ~/Dropbox/src2/cipher_benchmark/benchmark

# Borg (monoalphabetic)
PYTHONPATH=src python scripts/count_prompt_tokens.py \
  --test-id borg_single_B_borg_0045v \
  --split ~/Dropbox/src2/cipher_benchmark/benchmark/splits/borg_tests.jsonl \
  --benchmark-root ~/Dropbox/src2/cipher_benchmark/benchmark
```
Add `--no-preflight` to skip the automated solver and get a faster estimate.

### 2. Drop "toolkit by namespace" prose from the system prompt — **~2 K savings**

The 8,485-char "Your toolkit (by namespace)" section in `prompts_v2.py`
enumerates tool namespaces in prose ("workspace_* lets you fork
branches and merge…"). The model already sees every tool name and
description in the structured tool list. This prose is pure
duplication.

**Proposal.** Replace with a ~500-char "tool families:" cheatsheet:

```
Tool families: workspace_* (branches), observe_* (read stats),
decode_* (read decoded text), score_* (quality), corpus_* (dictionary),
act_* (mutate keys), search_* (heavy solvers), meta_* (control flow).
Call meta_request_tool for any tool not currently in your tool list.
```

**Why it's safe.** Frontier models (Claude, GPT-5) navigate fine
without the prose. Local models could be given the cheatsheet plus
explicit one-line guidance per family.

**Effort.** Small.

**Risk.** Possible quality regression on local models that lean on the
namespace narration as a planning aid. Mitigate by gating behind
`--system-prompt-style compact` (see #6) and keeping the full version
the default for now.

### 3. Compress tool descriptions — ✅ IMPLEMENTED

**Measured savings (2026-05-03, verified with `count_prompt_tokens.py`):**
- Tool description chars: 25,594 → 12,627 (−12,967 chars)
- Tool schema total chars: 66,948 → 53,896 (−13,052 chars, **~3,263 tokens**)
- System prompt: +540 chars (+135 tokens) for two new sentences restoring
  workflow guidance removed from descriptions
- **Net savings: ~3,128 tokens**

**What was implemented:**
All 58 tool descriptions longer than 200 chars were trimmed to ≤ 220 chars.
Methodology ("when to call", "how it interacts with other tools") was moved
into the system prompt's `act_*` and `decode_*` namespace notes rather than
a new separate section — keeping the existing structure intact.

Two specific gaps restored to the system prompt:
1. `act_*` section: `score_delta` from `act_set_mapping` is advisory — a
   correct cipher-symbol fix can drop `dictionary_rate` while still being
   correct; use `changed_words` as the real signal.
2. `decode_*` section: after `search_anneal` converges with a few residual
   errors, call `decode_diagnose_and_fix` — it collapses the diagnose →
   many-`act_set_mapping` loop into one call.

**Regression test (2026-05-03):**
Ran `parity_borg_latin_borg_0109v` (33-symbol Latin monoalphabetic) after
all changes. Result: **95.36% char accuracy, status=solved** — identical to
the pre-compression baseline (95.36%, solved). The agent called
`decode_diagnose_and_fix` twice in the new run (vs 0 in the baseline), showing
the workflow guidance reached the model correctly.

**Side effect — fingerprint false-positive fix:**
The Vigenère suspicion score falsely spiked (0.05→0.80) for large-alphabet
monoalphabetic ciphers (e.g. 33-symbol Borg) due to raw IC delta. Fixed by
normalising IC against `random_IC(alphabet_size)` and gating periodic IC boosts
on ≥ 25 tokens/column. Borg now correctly scores `polyalphabetic_vigenere=0.05`,
enabling 13 tools to be hidden (vs 3 before the fix).

### 4. Deduplicate the preflight fingerprint — ✅ IMPLEMENTED

In every user message, the same fingerprint natural-language summary
appeared twice:

- Under `## Cipher-type fingerprint` (full block with IC, entropy,
  periodic IC, ranked suspicions, etc.).
- Verbatim again inside `## Automated native solver preflight`,
  prefixed `Summary:`.

**What was built** (`src/automated/runner.py`):
Removed the `natural_language_summary` line from
`format_automated_preflight_for_llm`. The ranked suspicions one-liner
and periodic IC note are retained in the preflight block as a compact
cross-reference. The full prose summary stays only in the fingerprint
section. Saves ~300–500 chars per run.

### 5. Collapse overlapping benchmark context layers — ✅ IMPLEMENTED

The `_format_prompt_context` function was emitting one annotated bullet
per layer, repeating the record ID on every line:

```
- record_id / minimal (Basic provenance): [text]
- record_id / standard (Standard benchmark context) (plaintext hint): [text]
- record_id / historical (Historical context) (plaintext hint): [text]
```

**What was built** (`src/benchmark/context.py`):
Layers from the same record are now grouped into a single block:

```
- record_id [Basic provenance | Standard benchmark context (plaintext hint) | Historical context (plaintext hint)]: [text1] [text2] [text3]
```

Saves ~100–200 chars per record per run (larger for multi-record tests).

### 5b. Compress parameter descriptions — ✅ IMPLEMENTED

**Measured savings (2026-05-03, verified with `count_prompt_tokens.py`):**
- Parameter description chars: 5,929 → 3,911 (−2,018 chars)
- Tool schema total chars: 53,896 → 51,898 (−1,998 chars, **~500 tokens**)
- Mode-filtered total: ~19,988 → ~19,595 tokens (**−393 tokens** in the Borg case)
- **Net savings (all-tools): ~500 tokens; mode-filtered: ~393 tokens**

**What was implemented:**
All 50 parameter `description` fields longer than 80 chars were trimmed to
≤ 89 chars. Philosophy: the parameter name, type, enum values, and default
carry most of the meaning — descriptions answer "what is this?" not "how
do I use it?". Nine identical `context_override_rationale` and nine identical
`override_context_cipher_family` parameter descriptions were replaced in one
pass each. Tests: 592 passed, 1 skipped (unchanged from pre-trim baseline).

### 5c. Compress scoring notes — ✅ IMPLEMENTED

**Measured savings (2026-05-03, verified with `count_prompt_tokens.py`):**
- System prompt chars: 32,680 → 30,809 (−1,871 chars, **~468 tokens**)
- Mode-filtered initial prompt: ~19,595 → **~19,128 tokens** (−467 tokens)

**What was implemented:**
The `## Scoring notes`, `## Notation system notes`, and `## Homophonic
no-boundary caution` subsections of the system prompt were compressed
from 3,740 chars to 1,825 chars (−1,915 chars). Cuts made:

1. `dictionary_rate` bullet: removed the verbose three-paragraph explanation
   (auto-segmenter detail, ceiling rationale, direction vs. threshold prose)
   that duplicated "primary judgement instrument" and "tool-output discipline"
   sections. Retained: auto-segmentation fact, ceiling below 1.0, "Declare on
   reading", homophonic-no-boundary weakness.
2. Removed the final "None of these scores has a threshold" paragraph (fully
   redundant with the "primary judgement instrument" section).
3. `## Notation system notes`: prose → two-bullet table (516 → ~200 chars).
4. `## Homophonic no-boundary caution`: removed the verbose restatement of
   bad-basin discipline; kept the actionable tool references
   (`observe_homophone_distribution`, `decode_absent_letter_candidates`).

Tests: 594 passed, 1 skipped. The one word-capitalisation mismatch found
by `test_system_prompt_carries_reading_first_discipline` was fixed inline.

### 6. Provider-conditional system-prompt size — ✅ IMPLEMENTED (with #2)

**Measured savings (2026-05-03, verified with direct calculation):**
- Toolkit section: 9,029 chars (full) → 1,983 chars (compact) = **−7,046 chars / ~1,761 tokens**
- Compact system prompt (Latin): 30,810 → 23,764 chars (~5,941 tokens)
- Mode-filtered total (Borg, compact): **~17,366 tokens** (down from 19,128 full)

**What was implemented:**

`src/agent/prompts_v2.py`:
- Extracted the `## Your toolkit (by namespace)` section (9,029 chars) into `_TOOLKIT_FULL`
- Added `_TOOLKIT_COMPACT` (1,983 chars): namespace cheatsheet + explicit search
  sequencing rules (automated_solver-first, preserve_existing gate, word-islands
  → transform pipeline, pure transposition, Quagmire sizing). Omits the verbose
  per-tool narrative and the multi-paragraph Quagmire/transform-finalist prose.
- `SYSTEM_PROMPT_TEMPLATE` uses `{toolkit_section}` placeholder.
- `get_system_prompt(language, style="full")` selects `_TOOLKIT_FULL` or
  `_TOOLKIT_COMPACT` based on `style`.

`src/agent/loop_v2.py`: `run_v2()` accepts `system_prompt_style: str = "full"`.

`src/benchmark/runner_v2.py`: `BenchmarkRunnerV2.__init__()` accepts
`system_prompt_style: str = "full"`.

`src/cli.py`:
- `--system-prompt-style {full,compact,auto}` on both `benchmark` and `crack`
  subcommands (default `auto`).
- `_resolve_system_prompt_style()` returns `"compact"` for `provider == "ollama"`,
  `"full"` otherwise (or the explicit value if not `auto`).

**Sentinel tests added** (`tests/test_agent_reliability.py`):
- `test_compact_system_prompt_style`: compact is ≥5,000 chars shorter than full,
  retains critical sequencing rules, lacks Quagmire-style narrative.
- `test_compact_is_default_for_ollama`: provider=ollama+auto→compact,
  provider=anthropic+auto→full, explicit overrides work.

Tests: 596 passed, 1 skipped.

**K2 regression test (2026-05-03):** Ran `kryptos_k2_keyed_vigenere` (372-token
no-boundary keyed-Vigenère) with `--max-iterations 25`. Result: **100% char
accuracy, status=solved** via `search_quagmire3_keyword_alphabet(engine='rust_shotgun')`
with keyword `ABSCISSA`, keyed-alphabet `KRYPTOSABCDEFGHIJLMNQUVWXZ`. The
full-toolkit Quagmire guidance (retained in `_TOOLKIT_FULL`) directed the agent
to the right solver. The compact toolkit also includes a brief Quagmire hint.

### 7. Move full preflight plaintext to a tool result — **~700 savings, but riskier**

Today the user message embeds the full preflight plaintext (the 408-char
decryption for Zodiac 408). The agent could fetch it on demand via
`decode_show automated_preflight`.

**Why it's risky.** The preflight plaintext is the single most useful
signal in the prompt — for many runs it's already 99% correct. Forcing
the agent to call a tool to see it adds a turn of latency and risks
the agent never looking at it. Skip unless something else makes the
preflight branch obviously discoverable.

---

## Combined Impact

| Change                                            | Savings (tokens)  | Status       | Risk    | Effort |
| ------------------------------------------------- | ----------------: | :----------: | :-----: | :----: |
| 1. Mode-driven tool filtering (mono/Borg case)    |           ~2,654 | ✅ done      | Low     | Small  |
| 1. Mode-driven tool filtering (homophonic case)   |             ~820 | ✅ done      | Low     | Small  |
| 2. Drop toolkit prose (compact mode)              |           ~1,761 | ✅ done      | Low     | Small  |
| 3. Compress all tool descriptions (≤220 chars)    |           ~3,128 | ✅ done      | None    | Medium |
| 4. Dedupe preflight fingerprint                   |             ~400 | ✅ done      | None    | Trivial |
| 5. Collapse benchmark context layers              |             ~150 | ✅ done      | None    | Trivial |
| 5b. Compress parameter descriptions (≤89 chars)  |             ~500 | ✅ done      | None    | Small  |
| 5c. Compress scoring/notation/caution sections   |             ~468 | ✅ done      | None    | Small  |
| 6. Compact style flag + Ollama auto-default        |            (incl. in #2) | ✅ done | Low | Small |
| 7. Move preflight plaintext to tool result        |             ~700 | deferred     | High    | Small  |
| **Implemented (1+2+3+4+5+5b+5c+6, Borg, full)**   |       **~7,872** | ✅           |         |        |
| **Initial prompt now (Borg, full, mode-filtered)**|     **~19,128**  | ✅           |         |        |
| **Initial prompt now (Borg, compact, filtered)**  |     **~17,366**  | ✅           |         |        |
| **Realistic remaining (2+6)**                     |       **~5–7 K** | proposal     |         |        |

Bringing the total prompt from ~27 K → ~14–17 K tokens means:

- Ollama runs can use `OLLAMA_NUM_CTX=20480` instead of 32768, roughly
  halving the KV-cache memory footprint. qwen2.5:32b that previously
  ran out of memory or timed out should fit comfortably.
- Anthropic per-turn cache-read costs drop ~50%.
- Prefill latency drops in proportion (the dominant per-turn cost on
  local hardware).

---

## Recommended Implementation Order

**Phase 1 — Land the safe wins (no behaviour change):**

1. (#4) Dedupe preflight fingerprint.
2. (#5) Collapse benchmark context layers, or default policy to `standard`.

These are obviously-correct and require no tests beyond a snapshot
diff.

**Phase 2 — Mode-driven tool filtering (#1):**

The largest single impact and the cleanest win. Ship behind a feature
flag (`--tool-filter=fingerprint|none`) so it can be regression-tested
side-by-side. Default to `none` until the hardest-case suite confirms
no regression.

**Phase 3 — System-prompt restructuring (#2 + #6):**

Combine into one change. Add `--system-prompt-style {full,compact}`,
default `compact` for Ollama, `full` elsewhere. Run hardest-case suite
on both. Once `compact` is proven equivalent on hosted models, flip
the default.

**Phase 4 — Tool description compression (#3):**

Highest-effort, highest-risk. Do last, with careful tool-call-sequence
diffs across the synthetic suite.

**Defer indefinitely:**

- (#7) Moving preflight plaintext into a tool result. Cost/benefit is
  unfavourable.

---

## Open Questions

- **Tool-tag taxonomy.** What's the right granularity for `mode_tags`?
  Cipher families (`monoalphabetic`, `homophonic`, `polyalphabetic`,
  `transposition`, `playfair`) seems right; sub-types
  (`vigenere` vs. `quagmire`) may or may not be worth distinguishing.

- **Fingerprint reliability.** If the fingerprint is wrong (e.g.
  reports `monoalphabetic_substitution=0.8` for an actually-homophonic
  cipher), tool filtering will withhold the right solver. The
  `meta_request_tool` escape hatch covers this, but only if the agent
  realises the leading mode is wrong. Worth measuring fingerprint
  precision/recall before relying on threshold-based filtering.

- **Provider-aware defaults.** Should compact/full be tied to provider
  (Ollama→compact) or to estimated token budget (any model with
  effective context ≤ 16 K → compact)? The latter is more principled
  but harder to predict at startup.

- **Caching.** Anthropic prompt caching gives most of its benefit when
  the system prompt and tool definitions are *stable* across turns.
  Mode-filtered tool sets break stability if filtering changes
  mid-run. Recommend computing the filter once at run start and
  freezing it.
