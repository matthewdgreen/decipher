# Decipher — CLAUDE.md

Project context for Claude Code sessions. Keep this file updated as the project evolves.

---

## What This Is

A CLI research tool for classical cipher cryptanalysis. Primary focus:
- **Monoalphabetic substitution ciphers** with arbitrary symbol alphabets
- **Historical manuscripts** (Borg cipher in Latin, Copiale cipher in German)
- **AI-assisted decipherment** using Claude tool-use API
- **Benchmark evaluation** against a dataset of solved historical ciphers

---

## Claude Code Orchestration Strategy

All implementation plans in this repo are executed with the following division
of labor. The main-session model (Fable) oversees strategy end to end.

- **Oversight/strategy** — Fable in the main session loop: plans, sequences,
  launches sub-agents, integrates results, and decides what lands.
- **Specification development** — Fable at extra effort with full context.
  Doing this in the main loop is fine (the main session is Fable). Specs are
  written documents under `docs/specs/`, detailed enough to implement without
  access to the originating conversation: exact files/lines, desired behavior,
  edge cases, and required tests.
- **Coding** — Opus or Sonnet sub-agents implementing from the written spec,
  chosen per task: Opus for careful multi-file or behavioral work, Sonnet for
  mechanical/small tasks. Coding agents do not invent scope; gaps in the spec
  go back to the spec author.
- **Code review** — Fable sub-agents reviewing the diff against the spec.
- **Phase-completion commit** — after a phase passes review (including the
  Fable-verification step below and any fixes it triggers), commit it. One
  commit per completed phase, message naming the plan and phase. If the
  working tree held unrelated pre-existing WIP when the phase started,
  checkpoint that WIP as its own commit first so the phase commit contains
  only phase work.
- **Fable-verification step** — whenever a Fable sub-agent finishes, inspect
  that session's local metadata (the sub-agent transcript JSONL under the
  session/tasks directory) and confirm the assistant turns were actually
  served by `claude-fable-5` and not gated down to Opus by the safety gate,
  e.g.:

  ```bash
  grep -ho '"model": *"[^"]*"' <transcript>.jsonl | sort | uniq -c
  ```

  Report the check result alongside the agent's findings. If a turn was
  served by a different model, flag it and decide whether to rerun.

---

## Key Files

```
src/
  cli.py                  — CLI entry point (benchmark, crack, testgen, resume-artifact,
                            diagnose, doctor subcommands)
  models/
    alphabet.py           — Alphabet class (symbol↔integer mapping, multisym support)
    cipher_text.py        — CipherText dataclass (raw text + alphabet + word structure)
    session.py            — Headless Session: cipher text, key dict, apply_key()
  analysis/
    frequency.py          — mono/bigram/trigram frequency, chi-squared
    ic.py                 — Index of Coincidence
    pattern.py            — Word isomorphs, pattern dictionary, match_pattern()
    dictionary.py         — load_word_set(), score_plaintext(), get_dictionary_path(lang)
    solver.py             — Algorithmic solver: hill_climb_swaps(), auto_solve()
    ngram.py              — N-gram language models with lazy caching
    signals.py            — Multi-signal scoring panel (6 metrics)
    segment.py            — Rank-aware no-boundary word segmentation
    zenith_solver.py      — Zenith-parity SA for homophonic ciphers: exact entropy score,
                            un-normalized acceptance, binary model loader (26^5 float32)
    model_registry.py     — Language-model variant registry: resolves models/ngram5_*.bin
                            per language+variant (env pin → variant → default); German
                            default = DTA Kernkorpus (historical_1600_1899)
    panels.py             — INV-0 nine statistical diagnostic panels → evidence atoms
    null_baseline.py      — INV-0 shuffle-null / parametric baselines for order-sensitive stats
    numeric_code.py       — INV-0 P8 numeric battery (Beale/book-cipher diagnosis)
    coherence.py          — INV-0 order-sensitive word-island coherence report
  automated/
    runner.py             — Automated-only/no-LLM runner; zenith_native profile dispatch
  agent/
    prompts_v2.py         — V2 brief-style system prompt (no rigid phases)
    tools_v2.py           — V2: 95 tools across 11 namespaces + WorkspaceToolExecutor
    loop_v2.py            — V2 agent loop with workspace integration
    loop_shared.py        — Loop helpers shared by v2 and v3 (branch snapshots, fallbacks)
    narrate.py            — NarrateAgentRenderer: default scrolling Claude-Code-style transcript
    model_provider.py     — Provider-neutral model interface: Anthropic, OpenAI, Gemini,
                            Ollama, OpenRouter adapters + live pricing fetch
  investigation/          — V3 agent loop (investigation lead) + investigator mode INV-0
    state.py              — InvestigationState: serializable source of truth for a v3 run (M1)
    loop_v3.py            — run_v3 lead loop: context rebuilt from state each turn (M1)
    episodes.py           — Fresh-context worker episodes the lead delegates to (M2)
    experiments.py        — Background automated-solver experiment queue (M4)
    actions.py            — Composite hypothesis actions for the v3 surfaces (M3)
    reading.py            — The Reading artifact (word-boundary overlay, M3)
    adapter.py            — Build a v3 InvestigationState from a stored v2 RunArtifact (M6)
    diagnosis.py          — INV-0 ranked cipher-family diagnosis (drives `decipher diagnose`)
    families.py           — INV-0 family + discriminator registry
  workspace/
    __init__.py           — Branch and Workspace classes for v2 agent
  preprocessing/
    s_token_converter.py  — S-token to letter normalization for API compatibility
  artifact/
    schema.py             — RunArtifact, BranchSnapshot, ToolCall dataclasses
  benchmark/
    loader.py             — BenchmarkLoader: reads JSONL manifest + splits + data files
    runner_v2.py          — V2 BenchmarkRunner: with artifacts and preprocessing
    scorer.py             — score_decryption(), format_report() (char/word accuracy)
    unsolved.py           — INV-0 firewalled reader for the benchmark `unsolved/` area
  services/
    claude_api.py         — ClaudeAPI: send_message(), estimate_cost(), retry/error helpers
  ocr/
    engine.py             — OCREngine: process_image(), process_text()
    vision.py             — VisionOCR: Claude Vision for symbol extraction
  ciphers/
    substitution.py       — SubstitutionCipher: encrypt/decrypt/random_key
    caesar.py             — CaesarCipher: brute_force()
  external/
    azdecrypt.py          — Stub for AZdecrypt integration (not implemented)
    cryptocrack.py        — Stub for CryptoCrack integration (not implemented)
resources/
  dictionaries/
    english_common.txt    — 5000 common English words (uppercase, freq-ordered)
    latin_common.txt      — 4440 Latin words (medical/pharmaceutical focus)
    german_common.txt     — 3057 German words (18th-century Masonic focus)
tests/
  test_models.py          — model and session tests
  test_analysis.py        — frequency, IC, pattern, dictionary tests
  test_ciphers.py         — cipher primitive tests
  test_benchmark.py       — loader, runner, scorer tests
  test_workspace.py       — branch workspace tests
  test_signals.py         — scoring panel tests
  test_segment.py         — no-boundary segmentation tests
  test_agent_reliability.py — loop fallback and reliability behavior tests
  test_zenith_solver.py   — binary model loading, entropy/score formula, SA recovery (23 tests)
```

**TOOLS.md** is the canonical human-readable reference for all agent tools.
When adding, removing, or significantly changing tools in
`src/agent/tools_v2.py`, update `TOOLS.md` to match: tool name, description,
parameter table, and usage notes. The tool count in the `tools_v2.py` line
above should also be kept current.

---

## Architecture Decisions

### Token model
All analysis works on `list[int]` token IDs, not strings. `Alphabet` is the bidirectional mapping. This supports both single-char (A-Z) and multi-char (S001, S002 OCR-style) symbol sets uniformly.

### Session and workspace state
`Session` is a lightweight headless container used by solver algorithms. V2 agent runs use `Workspace`, which holds the immutable cipher text plus named branch keys for hypothesis exploration. There are no Qt signal dependencies in the active CLI path.

### Key representation
`dict[int, int]` — cipher token ID → plaintext token ID. Partial keys are fine; unmapped tokens show as `?`. `apply_key()` uses the plaintext alphabet's `_multisym` flag to determine output spacing (not the cipher alphabet's flag — important fix).

### Multisym alphabets
Canonical benchmark transcriptions use space-separated S-tokens (S001 S002 ...) with ` | ` as word separator. `parse_canonical_transcription()` handles this. Newlines in source files are also word boundaries.

### Language support
`analysis/dictionary.py` has `get_dictionary_path(language)` for `en`, `la`, `de`.
`agent/prompts.py` has language-specific `FREQUENCY_ORDERS`, `LANGUAGE_NOTES`, and `get_system_prompt(language)`.
Benchmark auto-detects: borg→`la`, copiale→`de`.

### Benchmark dataset
Located at `~/Dropbox/src2/cipher_benchmark/benchmark/`.
- `manifest/records.jsonl` — ~900 page records (905 at last count)
- `splits/borg_tests.jsonl` — 45 tests (15 Track B: transcription→plaintext)
- `splits/copiale_tests.jsonl` — 45 tests (15 Track B)
- Track B (transcription2plaintext) = canonical S-token transcription → plaintext
- Borg: monoalphabetic, 33 symbols, Latin pharmaceutical text
- Copiale: homophonic, 86 symbols, German Masonic text
- **Synthetic single-substitution splits live in their own files** (e.g.
  `en_ss_synth_nb_tests.jsonl`); they are NOT in `all_tests.jsonl`, so a synthetic
  `--test-id` needs an explicit `--split` (the benchmark command now says so when a
  filter matches nothing).

### V3 agent loop (`--agent-loop v3`)
An investigation *lead* loop that replaces the flat v2 loop. `InvestigationState`
(`src/investigation/state.py`) is the serializable source of truth; the lead context
is rebuilt from it every turn, so loading a serialized state and continuing *is* the
resume path. Milestones landed M1–M6:
- **M1** — `InvestigationState` + lead loop + provider-native session seam.
- **M2** — *episodes*: fresh-context workers the lead delegates focused sub-tasks to
  (`episode_run`; kinds survey/search/reading/compare/repair/verify), isolated from the
  lead workspace until `episode_install_branch`.
- **M3/M4** — composite hypothesis actions + a background *experiment queue*
  (`experiment_submit` / `experiment_collect`) for long-running automated-solver compute.
- **M5/M5.3** — verification-gated declaration: `meta_declare_solution` is unblocked only
  by a fresh `verify`-episode attestation whose content hash matches the branch's current
  text AND whose `reader_accepts_as_solution` is true (Slice 6 reversed C6; weak
  attestations route repair/compare/broaden instead).
- **M6** — a v2/v3 bake-off matrix + a v2-artifact→v3-state adapter (`adapter.py`).

Selected with `--agentic --agent-loop v3` (default is `v2`). The default no-LLM
automated solver runs when `--agentic` is absent.

### Investigator mode (INV-0)
`decipher diagnose <file|->` is a local, **LLM-free** cipher-family diagnosis. It runs
the nine statistical panels (`src/analysis/panels.py`), scored against shuffle/parametric
nulls (`null_baseline.py`), a numeric Beale/book-cipher battery (`numeric_code.py`), and a
word-island coherence guard (`coherence.py`); families and discriminators live in
`src/investigation/families.py`, ranking in `src/investigation/diagnosis.py`. It emits a
`confident`/`uncertain` verdict with a recommended next discriminator (`--json` for the
full report). The same battery is exposed to the agent as the `observe_diagnosis` tool.
A strict input firewall keeps it ciphertext-only (no plaintext/keys/context).

### Agentic display (`--display`, default `narrate`)
Agentic runs default to the **narrate** renderer (`src/agent/narrate.py`): a scrolling,
pipe-safe, Claude-Code-style transcript (one line per lead tool call, indented `↳` lines
for episode internals, a cumulative cost/token ticker). `--display` also accepts `pretty`
(the Rich live dashboard), `raw`, and `jsonl` (machine stream); `auto` resolves to narrate.

### Language-model variant registry
`src/analysis/model_registry.py` removes the one-model-per-language rule: each
`models/ngram5_*.bin` may declare a `variant` slug + label in its sidecar metadata.
Resolution precedence is env pin (`DECIPHER_NGRAM_MODEL_<LANG>`) → explicit `--model-variant`
→ per-language default → bare `ngram5_<lang>.bin`. The **German default is now the DTA
Kernkorpus model** (`historical_1600_1899` = `ngram5_de_dta.bin`), which beats the old
Gutenberg `literary_19c` model on every measured German workload. The agent can switch
mid-run via `act_set_model_variant`.

---

## Major Achievements (April 2026)

### ✅ **V2 Agentic Framework Completed**
Successfully implemented state-of-the-art agent-driven cryptanalysis system:
- **Branching workspace** with fork/merge/compare operations (src/workspace/)
- **95 specialized tools** across 11 namespaces (src/agent/tools_v2.py)
- **Multi-signal scoring** with 6 different metrics (src/analysis/signals.py)
- **Agent-driven termination** via meta_declare_solution (no rigid phases)
- **Full observability** via comprehensive run artifacts (src/artifact/schema.py)
- **Synthetic hard benchmark solved exactly**: synth_en_250nb_s4 reached 100% in 7 iterations

### ✅ **API Compatibility Layer Implemented**
Robust preprocessing and framing for reliable API interaction:
- **Automatic S-token normalization** (src/preprocessing/s_token_converter.py)
- **Manuscript-analysis framing** for academic historical research tasks
- **Model selection**: Claude Sonnet 4.6 recommended for decipherment tasks
- **Transparent artifact tracking** of preprocessing applied

### ✅ **Advanced Cryptanalytic Capabilities**
V2 system demonstrates sophisticated reasoning:
- **Constraint propagation**: "AMAMUS → H=A, C=M, I=U, G=S"
- **Conflict detection**: "K=A but H=A from AMAMUS - conflict!"
- **Strategic progression**: Overview → patterns → word candidates → constraints
- **Latin domain expertise**: Identifies pharmaceutical vocabulary (CARERE, etc.)
- **Multi-hypothesis testing** across branching workspace

### ✅ **Reliability and Homophonic Guardrails Added**
Recent testgen work turned failure logs into tool-design improvements:
- **Final-iteration preflight**: the loop can declare a strong branch before an avoidable last API call
- **Best-branch fallback**: API overloads/errors preserve the best candidate instead of losing the run
- **Rank-aware segmentation**: no-boundary English is segmented using frequency-ranked dictionary costs
- **Homophonic diagnostics**: tools identify ambiguous letters, absent letters, and likely split homophones
- **`run_python` audit trail**: Python remains allowed, but every use records a justification and is highlighted in reports as a tool-design signal

### ✅ **Zenith-Parity Homophonic Solver — 99.3% on Zodiac 408**
`src/analysis/zenith_solver.py` is a faithful Python port of Zenith's SA algorithm.
Activated via `DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native`. Closes the gap from
83.6% to 99.3% in ~160 s. Two root-cause bugs fixed vs. old `zenith_exact` profile:
1. **Score**: `mean_log_prob / entropy^(1/2.75)` (Shannon entropy divisor), not `mean * IoC^(1/6)`.
2. **Acceptance**: `exp(delta / temp)` with no `ngram_count` normalization — the old
   normalization made the effective temperature ~202× too cold.

---

## Remaining Challenges

### 1. ⏳ **Hardest homophonic/no-boundary tests**
The hardest synthetic preset (`synth_en_200honb_s6`) is still the stress case:
homophonic *and* no word boundaries. Homophonic evidence is now exposed via tools,
but this combination remains the open frontier.

### 2. 🔄 **Non-English homophonic search**
`zenith_native` solves English boundary-separated homophonic ciphers (99.3% Zodiac 408).
The remaining gap is non-English homophonic ciphers (Copiale/German — no binary model yet).

### 3. 🎭 **Historical Copiale/Borg generalization**
Synthetic tests are useful for controlled iteration, but the historical benchmark still needs broader runs to separate synthetic overfitting from durable cryptanalytic progress.

---

## V2 Architecture (✅ Implemented)

Successfully replaced rigid v1 agent with sophisticated v2 framework:

### Core principle: Agent drives, tools assist
✅ **Implemented features:**
1. **Full visibility** — observe/decode/score tools for comprehensive analysis
2. **Rich tool set** — 95 tools across 11 namespaces (workspace, observe, search, decode, score, corpus, act, repair, inspect/list, run_python, meta)
3. **Agent freedom** — No phases, agent plans own strategy
4. **Hypothesis tracking** — Branching workspace preserves exploration history

### Tool Arsenal (95 tools implemented)
**TOOLS.md** is the canonical, always-current per-tool reference (name, params,
usage notes). Namespace counts (as of the 95-tool build):
✅ **workspace_\*** (12) — fork/fork_best, branch & hypothesis cards, compare, merge, …
✅ **observe_\*** (14) — frequency, isomorph clusters, IC, cipher_id, **diagnosis** (INV-0), periodic/Kasiski, homophone distribution, transform suspicion, …
✅ **search_\*** (14) — hill_climb, anneal, homophonic_anneal, transform/transposition/periodic/Quagmire searches + review/install companions
✅ **decode_\*** (13) — show, unmapped, heatmap, letter/ambiguous/absent-letter diagnostics, no-boundary + reading repair
✅ **score_\*** (3), **corpus_\*** (2)
✅ **act_\*** (24) — set/bulk/anchor/clear mappings, periodic keys, structural word split/merge, boundary + word-repair + transform installs, model-variant switch
✅ **repair_agenda_\*** (2), **inspect_\*/list_\*** (6 — benchmark context)
✅ **run_python** (1) — escape hatch with required justification
✅ **meta_\*** (4) — request_tool, declare_solution, declare_unsolved, attest_reading_comprehensibility

*(V3 lead-only tools — `episode_run`/`episode_install_branch`, `experiment_submit`/`experiment_collect` — are separate from these 95 and documented at the end of TOOLS.md.)*

### Termination criteria
✅ **Implemented:**
- Agent calls `meta_declare_solution` when confident
- Natural exhaustion at max_iterations
- No arbitrary score thresholds

### Advanced capabilities demonstrated
✅ **Constraint reasoning**: Detects mapping conflicts
✅ **Strategic thinking**: Plans multi-step analysis
✅ **Domain expertise**: Recognizes Latin pharmaceutical vocabulary
✅ **Hypothesis management**: Uses workspace branches effectively

---

## Running

```bash
# V2 Benchmark (recommended)
.venv/bin/decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark \
  --source borg --model claude-sonnet-4-6 --verbose

# V2 Single test with full analysis
.venv/bin/decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark \
  --test-id borg_single_B_borg_0045v --model claude-sonnet-4-6 --max-iterations 15

# V3 investigation loop (synthetic ids need their explicit --split)
.venv/bin/decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark \
  --split en_ss_synth_nb_tests.jsonl --test-id synth_en_200honb_s6 \
  --agentic --agent-loop v3 --model gpt-5.5 --max-iterations 25

# Local LLM-free cipher-family diagnosis (investigator INV-0)
.venv/bin/decipher diagnose path/to/ciphertext.txt
echo "S001 S002 S003 | S004 S005" | .venv/bin/decipher diagnose - --json

# V2 crack from text (automatic S-token preprocessing)
echo "S025 S012 S006 | S003 S007" | .venv/bin/decipher crack \
  --language la --model claude-sonnet-4-6 --canonical

# Hardest synthetic regression only
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --preset hardest --model claude-sonnet-4-6 --max-iterations 25 --verbose

# Zenith-parity native solver on Zodiac 408 (99.3% in ~160s)
DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native \
  PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py \
  --solvers decipher \
  --benchmark-split ~/Dropbox/src2/cipher_benchmark/benchmark/splits/parity_zodiac.jsonl \
  --benchmark-root ~/Dropbox/src2/cipher_benchmark/benchmark \
  --artifact-dir artifacts/zenith_native \
  --summary-jsonl artifacts/zenith_native/summary.jsonl \
  --summary-csv artifacts/zenith_native/summary.csv

# Legacy V1 commands
.venv/bin/decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark --source borg -v
.venv/bin/decipher crack -f input.txt --language la

# Run tests
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

---

## Development Setup

```bash
cd ~/Dropbox/src2/decipher
source .venv/bin/activate   # Python 3.11 venv
pip install -e .             # Install with entry points
```

Python 3.11 at `/opt/homebrew/bin/python3.11`. Venv at `.venv/`.

---

## Model Selection

**Confirmed agent model (2026-07-13, head-to-head verdict)**: `gpt-5.5`
(OpenAI). Beat `gpt-5.6-sol` at equal price on borg_0109v word accuracy
(mean 82.3% over {84.8, 79.7} vs Sol's 75.2% over {65.4, 85.9, 74.4 —
last with reasoning passback}), with bit-identical 95.9% char accuracy
across runs and fewer tokens on the hardest synthetic (275k vs 335k).
Evidence: `artifacts/agentic_model_comparison/`. Caveat: n=2–3 on one
page. The gpt-5.6 tiers are fully usable (Responses-API path + reasoning
passback landed): `gpt-5.6-luna` is the standout value tier ($0.59 for a
solved Borg page, 94.2%/77.2%) — first candidate for cheap v3 episode
workers. Agentic API spend bills the **OpenAI** account
(`.decipher_keys/openai_api_key`); `--model gpt-5.5` auto-routes to the
OpenAI provider; pricing table is current.

**V2 vs V3 (M6 bake-off).** A v2/v3 comparison exists
(`artifacts/m6_bakeoff/summary.jsonl`): v3 is much cheaper at comparable
char accuracy, but trails v2 on Borg *word* accuracy because more v3 runs
end in the best-branch fallback rather than an explicit declaration. No
single-number headline is claimed here; read the summary for specifics.

**Previous recommendation**: `claude-sonnet-4-6` — best Anthropic results on
historical manuscript analysis (Anthropic key lives in the macOS keychain,
`service=decipher`; that account currently has no credits).

### Supported Providers

Five providers are wired through `src/agent/model_provider.py`:

| Provider | Flag | Key location | Notes |
|---|---|---|---|
| `anthropic` | `--provider anthropic` | `ANTHROPIC_API_KEY` / keychain `anthropic_api_key` | Best quality; default |
| `openai` | `--provider openai` | `OPENAI_API_KEY` / keychain `openai_api_key` | GPT-5.x |
| `gemini` | `--provider gemini` | `GEMINI_API_KEY` / keychain `gemini_api_key` | Gemini 3.x |
| `ollama` | `--provider ollama` | None (local) | No API key; needs `ollama serve` |
| `openrouter` | `--provider openrouter` or `--provider or` | `OPENROUTER_API_KEY` / `.decipher_keys/openrouter_api_key` | 300+ models |

Provider is auto-detected in preference order: anthropic → openai → gemini → openrouter → ollama.
Any model ID containing `/` is inferred as OpenRouter (e.g. `--model meta-llama/llama-3.3-70b-instruct`).

### Anthropic Models
- **Claude Sonnet 4.6**: Strong performance on S-token sequences and Latin/German manuscript analysis. Recommended.
- **Claude Opus 4.7**: More conservative with historical encoded text; use Sonnet 4.6 for decipherment.

### OpenRouter Models — Tool-Calling Compatibility

OpenRouter proxies 300+ models through an OpenAI-compatible API. **Reliability of structured
tool calls varies widely by model.** Tested as of May 2026:

| Model | Tool calling | Notes |
|---|---|---|
| `tencent/hy3-preview:free` | ✅ Solves | Good tool-call discipline and reasoning; free tier; confirmed solve on synth_en_97q3nb_s50 |
| `meta-llama/llama-4-maverick` | ✅ Untested | Likely better than 3.3-70b; worth trying |
| `deepseek/deepseek-chat` (`deepseek-v3`) | ⚠️ Partial | Tool calls fire; gives up too early; see note |
| `qwen/qwen3-30b-a3b` | ✅ Untested | MoE; cheap at $0.09/$0.45 per M |
| `mistralai/mistral-small-3.2-24b-instruct` | ✅ Untested | $0.075/$0.20 per M |
| `meta-llama/llama-3.3-70b-instruct` | ⚠️ Poor | Tool calls fire but reasoning quality is very low; see note |
| `deepseek/deepseek-r1` | ❌ **Broken** | See note below |
| `deepseek/deepseek-r1-0528` | ❌ **Broken** | Same issue |

**DeepSeek-R1 tool-calling failure** (confirmed May 2026, artifact `cafa0b5e3363`):
R1 is a *reasoning* model fine-tuned for chain-of-thought, not agentic tool loops.
When given OpenAI-format tool definitions it outputs the tool call as a Markdown JSON code
block inside text rather than in the structured `tool_calls` response field. The agent loop
finds zero `tool_use` blocks, fires `no_tool_calls` on iteration 1, and exits immediately.
The model's *reasoning* is correct (it names the right tool and right arguments), but the
output format is wrong. There is also visible thinking-token bleed into the output text.
This is not fixable by prompt engineering — use DeepSeek-V3 instead.

**DeepSeek-V3 (`deepseek/deepseek-chat`) partial failure** (confirmed May 2026, artifact `bd7ca7931996`):
Tool calls fire correctly. Called `search_quagmire3_keyword_alphabet` at diagnostic budget
(4,000 proposals), decided the family was wrong after that minimal pass, then ignored the
explicit harness block requesting a moderate-budget search, and stopped calling tools in
the final iteration. The correct tool was identified; the failure is insufficient budget
escalation and non-compliance with harness feedback.

**Llama-3.3-70b-instruct failure** (confirmed May 2026, artifact `a7cba7261bac`):
Tool calls fire, but reasoning quality is very poor. Spent all 20 iterations fixating on
the keyless `automated_preflight` branch, repeating 7 identical failing `act_swap_decoded`
calls and 5 blocked `meta_declare_solution` attempts. Never called `search_quagmire3_keyword_alphabet`
or `workspace_branch_cards` despite both being explicitly required. One `search_anneal`
call crashed with `ZeroDivisionError` from a hallucinated `t_end=0` argument. Passed a
literal instruction string as the `proposed_text` argument to `act_resegment_by_reading`.
Strictly worse than DeepSeek-V3 in reasoning quality.

### Pricing
Cost estimation is live for OpenRouter: `estimate_provider_cost()` fetches
`https://openrouter.ai/api/v1/models` on first use (no auth required), caches to
`~/.config/decipher/openrouter_pricing.json` for 24 hours.
Run `decipher doctor --refresh-pricing` to force a refresh and see a diff.
Anthropic/OpenAI/Gemini pricing is hardcoded in `_PRICING` and updated with code releases.

### Configuration
Models configurable via `--model` CLI flag.

### Performance
Sonnet 4.6 on `synth_en_250nb_s4`: exact match in 7 iterations after reliability and segmentation fixes.
`synth_en_200honb_s6` is the active hardest homophonic/no-boundary stress test.
