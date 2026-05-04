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

## Key Files

```
src/
  cli.py                  — CLI entry point (benchmark, crack, testgen subcommands)
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
  automated/
    runner.py             — Automated-only/no-LLM runner; zenith_native profile dispatch
  agent/
    prompts_v2.py         — V2 brief-style system prompt (no rigid phases)
    tools_v2.py           — V2: 78 tools across 11 namespaces + WorkspaceToolExecutor
    loop_v2.py            — V2 agent loop with workspace integration
    model_provider.py     — Provider-neutral model interface: Anthropic, OpenAI, Gemini,
                            Ollama, OpenRouter adapters + live pricing fetch
    state.py              — AgentState, Checkpoint (checkpointing + rollback)
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
- `manifest/records.jsonl` — 638 page records
- `splits/borg_tests.jsonl` — 45 tests (15 Track B: transcription→plaintext)
- `splits/copiale_tests.jsonl` — 45 tests (15 Track B)
- Track B (transcription2plaintext) = canonical S-token transcription → plaintext
- Borg: monoalphabetic, 33 symbols, Latin pharmaceutical text
- Copiale: homophonic, 86 symbols, German Masonic text

---

## Major Achievements (April 2026)

### ✅ **V2 Agentic Framework Completed**
Successfully implemented state-of-the-art agent-driven cryptanalysis system:
- **Branching workspace** with fork/merge/compare operations (src/workspace/)
- **32 specialized tools** across 9 namespaces (src/agent/tools_v2.py)
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
The hardest synthetic preset (`synth_en_200honb_s6`) is the current stress case. The tool now exposes homophonic evidence explicitly, but the next run should confirm whether the agent uses those tools instead of ad hoc Python.

### 2. 🔄 **Homophonic search quality**
`zenith_native` solves English boundary-separated homophonic ciphers (99.3% Zodiac 408).
Remaining gaps: non-English homophonic ciphers (Copiale/German, no binary model yet),
no-boundary homophonic ciphers (`synth_en_200honb_s6`), and agent-tool exposure
(the `zenith_native` path is automated-runner-only for now).

### 3. 🎭 **Historical Copiale/Borg generalization**
Synthetic tests are useful for controlled iteration, but the historical benchmark still needs broader runs to separate synthetic overfitting from durable cryptanalytic progress.

### 4. 🔧 **Model selection**
Sonnet 4.6 performs best on historical manuscript analysis tasks. Opus 4.7 is more
conservative with encoded historical text. See Model Selection section for guidance.

---

## V2 Architecture (✅ Implemented)

Successfully replaced rigid v1 agent with sophisticated v2 framework:

### Core principle: Agent drives, tools assist
✅ **Implemented features:**
1. **Full visibility** — observe/decode/score tools for comprehensive analysis
2. **Rich tool set** — 32 tools across 9 namespaces (workspace, observe, decode, score, corpus, act, search, run_python, meta)
3. **Agent freedom** — No phases, agent plans own strategy
4. **Hypothesis tracking** — Branching workspace preserves exploration history

### Tool Arsenal (32 tools implemented)
✅ **workspace_* (5 tools)** — fork, list, delete, compare, merge
✅ **observe_* (4 tools)** — frequency, isomorph_clusters, ic, homophone_distribution
✅ **decode_* (8 tools)** — show, unmapped, heatmap, letter_stats, ambiguous_letter, absent_letter_candidates, diagnose, diagnose_and_fix
✅ **score_* (3 tools)** — panel, quadgram, dictionary
✅ **corpus_* (2 tools)** — lookup_word, word_candidates
✅ **act_* (5 tools)** — set_mapping, bulk_set, anchor_word, clear, swap_decoded
✅ **search_* (3 tools)** — hill_climb, anneal, homophonic_anneal
✅ **run_python (1 tool)** — allowed escape hatch with required justification
✅ **meta_* (2 tools)** — request_tool, declare_solution

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

**Recommended for V2**: `claude-sonnet-4-6` — Best results on historical manuscript analysis
**Default**: `claude-opus-4-7` — Configurable via `--model` flag

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
| `meta-llama/llama-3.3-70b-instruct` | ✅ Reliable | Good default; $0.10/$0.32 per M |
| `meta-llama/llama-4-maverick` | ✅ Reliable | $0.15/$0.60 per M |
| `deepseek/deepseek-v3-0324` / `deepseek-chat` | ✅ Reliable | V3 chat model; ~$0.20-0.28 in / ~$0.77-0.89 out per M |
| `qwen/qwen3-30b-a3b` | ✅ Reliable | MoE; cheap at $0.09/$0.45 per M |
| `mistralai/mistral-small-3.2-24b-instruct` | ✅ Reliable | $0.075/$0.20 per M |
| `deepseek/deepseek-r1` | ❌ **Broken** | See note below |
| `deepseek/deepseek-r1-0528` | ❌ **Broken** | Same issue |

**DeepSeek-R1 tool-calling failure** (confirmed May 2026, artifact `cafa0b5e3363`):
R1 is a *reasoning* model fine-tuned for chain-of-thought, not agentic tool loops.
When given OpenAI-format tool definitions it outputs the tool call as a Markdown JSON code
block inside text rather than in the structured `tool_calls` response field. The agent loop
finds zero `tool_use` blocks, fires `no_tool_calls` on iteration 1, and exits immediately.
The model's *reasoning* is correct (it names the right tool and right arguments), but the
output format is wrong. There is also visible thinking-token bleed into the output text.
This is not fixable by prompt engineering — use DeepSeek-V3 (`deepseek/deepseek-chat`) instead.

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
