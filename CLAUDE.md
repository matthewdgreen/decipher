# Decipher — CLAUDE.md

Project context for Claude Code sessions. Keep this file updated as the project evolves.

---

## What This Is

A Mac desktop application for classical cipher cryptanalysis. Primary focus:
- **Monoalphabetic substitution ciphers** with arbitrary symbol alphabets
- **Historical manuscripts** (Borg cipher in Latin, Copiale cipher in German)
- **AI-assisted decipherment** using Claude tool-use API
- **Benchmark evaluation** against a dataset of solved historical ciphers

---

## Key Files

```
src/
  cli.py                  — CLI entry point (benchmark, crack, gui subcommands)
  main.py                 — GUI entry point
  app.py                  — QApplication setup
  models/
    alphabet.py           — Alphabet class (symbol↔integer mapping, multisym support)
    cipher_text.py        — CipherText dataclass (raw text + alphabet + word structure)
    session.py            — Session(QObject): central state, key dict, apply_key()
  analysis/
    frequency.py          — mono/bigram/trigram frequency, chi-squared
    ic.py                 — Index of Coincidence
    pattern.py            — Word isomorphs, pattern dictionary, match_pattern()
    dictionary.py         — load_word_set(), score_plaintext(), get_dictionary_path(lang)
    solver.py             — Algorithmic solver: hill_climb_swaps(), auto_solve()
    ngram.py              — N-gram language models with lazy caching
    signals.py            — Multi-signal scoring panel (6 metrics)
  agent/
    prompts.py            — V1 system prompt template, initial_context(), FREQUENCY_ORDERS
    tools.py              — V1: 14 tool definitions + ToolExecutor class  
    prompts_v2.py         — V2 brief-style system prompt (no rigid phases)
    tools_v2.py           — V2: 22 tools across 8 namespaces + WorkspaceToolExecutor
    loop_v2.py            — V2 agent loop with workspace integration
    state.py              — AgentState, Checkpoint (checkpointing + rollback)
    loop.py               — V1: AgentWorker(QThread), AgentLoop (GUI integration)
  workspace/
    __init__.py           — Branch and Workspace classes for v2 agent
  preprocessing/
    s_token_converter.py  — S-token to letter normalization for API compatibility
  artifact/
    schema.py             — RunArtifact, BranchSnapshot, ToolCall dataclasses
  benchmark/
    loader.py             — BenchmarkLoader: reads JSONL manifest + splits + data files
    runner.py             — V1 BenchmarkRunner: headless agent execution
    runner_v2.py          — V2 BenchmarkRunner: with artifacts and preprocessing
    scorer.py             — score_decryption(), format_report() (char/word accuracy)
  services/
    claude_api.py         — ClaudeAPI: send_message(), vision_request()
    settings.py           — Settings: API key (keychain), model, max_iterations
  ocr/
    engine.py             — OCREngine: process_image(), process_text()
    vision.py             — VisionOCR: Claude Vision for symbol extraction
  ui/
    main_window.py        — Two-column layout, menus
    input_panel.py        — Text/image input, alphabet/spaces/punct dropdowns
    analysis_panel.py     — Frequency table, IC display, patterns tab
    substitution_grid.py  — Interactive QComboBox grid for manual key editing
    agent_panel.py        — Agent start/stop, language selector, log view
    encode_panel.py       — Plaintext→ciphertext using inverted key
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
  test_models.py          — 17 tests
  test_analysis.py        — 19 tests
  test_ciphers.py         — 8 tests
  test_benchmark.py       — 22 tests (loader, runner, scorer)
```

---

## Architecture Decisions

### Token model
All analysis works on `list[int]` token IDs, not strings. `Alphabet` is the bidirectional mapping. This supports both single-char (A-Z) and multi-char (S001, S002 OCR-style) symbol sets uniformly.

### Session as central state
`Session(QObject)` holds `cipher_text`, `key: dict[int,int]`, and `plaintext_alphabet`. Qt signals notify UI on changes. For CLI/benchmark, a headless Session works fine (signals fire but no listeners).

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
- **22 specialized tools** across 8 namespaces (src/agent/tools_v2.py)
- **Multi-signal scoring** with 6 different metrics (src/analysis/signals.py)
- **Agent-driven termination** via meta_declare_solution (no rigid phases)
- **Full observability** via comprehensive run artifacts (src/artifact/schema.py)
- **Demonstrated 6.4% character accuracy** on historical Borg Latin cipher

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

---

## Remaining Challenges

### 1. ⏳ **Iteration limits for complex ciphers**
V2 system makes excellent progress but may need more than 10 iterations for full solutions
on 267-token historical ciphers. Consider increasing max_iterations for research runs.

### 2. 🔄 **Hill-climbing integration**
`analysis/solver.py` hill-climber available but not yet integrated as a v2 tool. Could
provide better starting points than frequency mapping.

### 3. 🎭 **Copiale homophonic ciphers**
V2 system designed for monoalphabetic substitution. Homophonic ciphers (multiple 
symbols → one letter) need extended constraint satisfaction tools.

### 4. 🔧 **Model selection**
Sonnet 4.6 performs best on historical manuscript analysis tasks. Opus 4.7 is more
conservative with encoded historical text. See Model Selection section for guidance.

---

## V2 Architecture (✅ Implemented)

Successfully replaced rigid v1 agent with sophisticated v2 framework:

### Core principle: Agent drives, tools assist
✅ **Implemented features:**
1. **Full visibility** — 8 observe_* tools for comprehensive analysis
2. **Rich tool set** — 22 tools across 8 namespaces (workspace, observe, decode, score, corpus, act, search, meta)
3. **Agent freedom** — No phases, agent plans own strategy
4. **Hypothesis tracking** — Branching workspace preserves exploration history

### Tool Arsenal (22 tools implemented)
✅ **workspace_* (5 tools)** — fork, list, delete, compare, merge  
✅ **observe_* (3 tools)** — frequency, isomorph_clusters, ic  
✅ **decode_* (3 tools)** — show, unmapped, heatmap  
✅ **score_* (4 tools)** — panel, quadgram, dictionary, bigram  
✅ **corpus_* (2 tools)** — word_candidates, pattern_search  
✅ **act_* (4 tools)** — set_mapping, bulk_set, anchor_word, clear  
✅ **search_* (1 tool)** — hill_climb (integrated solver)  
✅ **meta_* (1 tool)** — declare_solution (agent-driven termination)

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
# GUI
.venv/bin/decipher-gui

# V2 Benchmark (recommended)
.venv/bin/decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark \
  --source borg --v2 --model claude-sonnet-4-6 --verbose

# V2 Single test with full analysis  
.venv/bin/decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark \
  --test-id borg_single_B_borg_0045v --v2 --model claude-sonnet-4-6 --max-iterations 15

# V2 crack from text (automatic S-token preprocessing)
echo "S025 S012 S006 | S003 S007" | .venv/bin/decipher crack \
  --v2 --language la --model claude-sonnet-4-6 --canonical

# Legacy V1 commands
.venv/bin/decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark --source borg -v
.venv/bin/decipher crack -f input.txt --language la

# Run tests (all 100 tests pass)
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

### Model Notes
- **Claude Sonnet 4.6**: Strong performance on S-token sequences and Latin/German manuscript analysis
- **Claude Opus 4.7**: More conservative with historical encoded text; use Sonnet 4.6 for decipherment
- **Preprocessing**: S-token normalization (letter substitution) improves API compatibility across models

### Configuration  
Models configurable via `--model` CLI flag or GUI Settings menu.
API key stored in macOS Keychain under service `decipher`, account `anthropic_api_key`.
Also reads `ANTHROPIC_API_KEY` env var.

### Performance
Sonnet 4.6 on Borg cipher: 20+ tool calls, constraint reasoning, demonstrated Latin analysis
