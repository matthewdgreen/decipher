# Decipher — AGENTS.md

Project context for Codex sessions. Keep this file updated as the project evolves.

---

## What This Is

A CLI research tool for classical cipher cryptanalysis. Primary focus:
- **Monoalphabetic substitution ciphers** with arbitrary symbol alphabets
- **Historical manuscripts** (Borg cipher in Latin, Copiale cipher in German)
- **AI-assisted decipherment** using Codex tool-use API
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
  agent/
    prompts_v2.py         — V2 brief-style system prompt (no rigid phases)
    tools_v2.py           — V2: 32 tools across 9 namespaces + WorkspaceToolExecutor
    loop_v2.py            — V2 agent loop with workspace integration
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
  automated/
    runner.py             — Automated-only/no-LLM runner using native solving techniques
  services/
    claude_api.py         — ClaudeAPI: send_message(), estimate_cost(), retry/error helpers
  ocr/
    engine.py             — OCREngine: process_image(), process_text()
    vision.py             — VisionOCR: Codex Vision for symbol extraction
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
  test_automated_runner.py — no-LLM automated runner and CLI bypass tests
  test_agent_reliability.py — loop fallback and reliability behavior tests
```

---

## Architecture Decisions

### Token model
All analysis works on `list[int]` token IDs, not strings. `Alphabet` is the bidirectional mapping. This supports both single-char (A-Z) and multi-char (S001, S002 OCR-style) symbol sets uniformly.

### Session and workspace state
`Session` is a lightweight headless container used by solver algorithms. V2 agent runs use `Workspace`, which holds the immutable cipher text plus named branch keys for hypothesis exploration. There are no Qt signal dependencies in the active CLI path.

### Automated-only mode
`--automated-only` is a no-LLM path for `benchmark`, `crack`, and cached `testgen` runs. It lives in `src/automated/runner.py` so native parity work can evolve separately from `agent/loop_v2.py`. Artifacts are marked `run_mode: automated_only`, `automated_only: true`, with zero tokens and zero estimated cost. Testgen automated-only requires cached plaintext because generating fresh synthetic prose would require an LLM call.

### Automated preflight for LLM runs
LLM-enabled `benchmark`, `crack`, and testgen-suite runs now execute the native automated solver before iteration 1 by default. The result is stored in `artifact.automated_preflight`, exposed to the agent as an `automated_preflight` workspace branch, and summarized in the first context message without benchmark ground-truth accuracy. Use `--no-automated-preflight` to disable this when measuring the unaided agent loop.

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
- `manifest/records.jsonl` — currently 896 records: Borg, Copiale, DECODE/Gallica, multilingual synthetic substitution, and tool-bundled parity records
- `splits/borg_tests.jsonl` — 45 tests (15 Track B: transcription→plaintext)
- `splits/copiale_tests.jsonl` — 45 tests (15 Track B)
- `splits/*_ss_synth*_tests.jsonl` — multilingual synthetic simple-substitution Track B tests
- Track B (transcription2plaintext) = canonical S-token transcription → plaintext
- Borg: monoalphabetic, 33 symbols, Latin pharmaceutical text
- Copiale: homophonic, 86 symbols, German Masonic text
- Use `scripts/validate_benchmark.py` before relying on a benchmark checkout for parity work.
- Benchmark curation lives in `../cipher_benchmark`; Decipher solver/harness code lives here.
- Keep parity tests distinct from agentic-advantage tests. Parity asks whether the agent can match non-agentic solvers; advantage asks whether agentic context, OCR, diagnosis, or hypothesis management improves beyond them.

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
- **Model selection**: Codex Sonnet 4.6 recommended for decipherment tasks
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
- **Automated-only baseline**: `--automated-only` runs native solvers without LLM API calls and writes comparable zero-cost artifacts
- **Automated preflight**: LLM runs receive a no-LLM native candidate before iteration 1 and can adopt, repair, or reject it
- **Simple-substitution parity path**: English simple substitution now uses bijective continuous n-gram annealing, preserving one-to-one mappings while allowing unused plaintext letters to enter the key

---

## Remaining Challenges

### 1. ⏳ **Hardest homophonic/no-boundary tests**
The hardest synthetic preset (`synth_en_200honb_s6`) is the current stress case. The tool now exposes homophonic evidence explicitly, but the next run should confirm whether the agent uses those tools instead of ad hoc Python.

### 2. 🔄 **Homophonic search quality**
`search_homophonic_anneal` is now available and should be the first automated search tool for homophonic no-boundary ciphers. Continue seed sweeps against Zenith and zkdecrypto-lite, add top-N candidates, and classify failures as tool weakness vs agent wrong-tool choice.

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

# No-LLM automated baseline on the hardest synthetic cached instance
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --preset hardest --automated-only

# Chunkable no-LLM automated parity matrix
PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py \
  --presets tiny medium hard hardest \
  --artifact-dir artifacts/automated_parity_matrix/default

# Example small chunk: seed sweep, first shard only
PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py \
  --families simple-wb simple-nb homophonic-nb \
  --lengths 40 200 \
  --seeds 1-20 \
  --shard-count 4 --shard-index 0 \
  --artifact-dir artifacts/automated_parity_matrix/shard0

# No-LLM crack from canonical text
echo "S025 S012 S006 | S003 S007" | .venv/bin/decipher crack \
  --language la --canonical --automated-only

# Legacy V1 commands
.venv/bin/decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark --source borg -v
.venv/bin/decipher crack -f input.txt --language la

# Run tests (112 tests pass)
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q

# Validate benchmark checkout
PYTHONPATH=src .venv/bin/python scripts/validate_benchmark.py \
  ../cipher_benchmark/benchmark
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

### Model Notes
- **Claude Sonnet 4.6**: Strong performance on S-token sequences and Latin/German manuscript analysis
- **Claude Opus 4.7**: More conservative with historical encoded text; use Sonnet 4.6 for decipherment
- **Preprocessing**: S-token normalization (letter substitution) improves API compatibility across models

### Configuration
Models configurable via `--model` CLI flag.
API key stored in macOS Keychain under service `decipher`, account `anthropic_api_key`.
Also reads `ANTHROPIC_API_KEY` env var.

### Performance
Sonnet 4.6 on `synth_en_250nb_s4`: exact match in 7 iterations after reliability and segmentation fixes.
`synth_en_200honb_s6` is the active hardest homophonic/no-boundary stress test.
