# Decipher

Decipher is a tool for performing automated cryptanalysis of classical ciphers,
with a focus on historical manuscripts. The goal is to achieve parity with, and
then improve on, the state-of-the-art in automated solver tools.

Currently supported cipher families:

- **Monoalphabetic substitution** — simple, homophonic (e.g. Zodiac 408/Copiale)
- **Transposition + homophonic** — known-pipeline replay and open-ended
  transform search (e.g. Zodiac 340 family)
- **Periodic polyalphabetic** — Vigenère, Beaufort, Variant Beaufort, Gronsfeld

Decipher's primary mode is its native automated solver stack: fast,
reproducible, and usable with only local computation. An experimental agentic
solver (requires an API key) layers an LLM on top for branching hypothesis
exploration and manual solving steps.

Decipher borrows solving algorithms (with attribution and license compliance)
from the Zenith solving tool. For licensing reasons we do not redistribute
Zenith's ngram models, and instead provide our own (as well as tooling to
generate additional models).

## Setup

Prerequisites:

- Python 3.11 or newer. See the [Python downloads](https://www.python.org/downloads/)
  and [venv documentation](https://docs.python.org/3/library/venv.html).
- Rust with Cargo. The recommended installer is [rustup](https://rustup.rs/).
- A local C/native build toolchain for Python extension builds:
  - macOS: install Xcode Command Line Tools with `xcode-select --install`.
  - Debian/Ubuntu: install `build-essential` and `python3.11-dev` or the
    matching `python3-dev` package for your Python.
  - Fedora/RHEL: install `gcc`, `gcc-c++`, `make`, and `python3-devel`.
- `pip` in the virtual environment. The setup script installs `maturin`
  automatically.

```bash
cd /path/to/decipher
scripts/setup_dev.sh
```

The script finds a Python 3.11+ interpreter, creates `.venv`, installs the
package in editable mode, and builds the Rust extension. If any prerequisite
is missing it prints platform-specific install instructions and exits.

To activate the venv in your current shell after setup:

```bash
source .venv/bin/activate
```

A bundled English 5-gram model ships at `models/ngram5_en.bin`, so a fresh
clone can run automated solves immediately. No API key is required for the
default automated workflows. See [Build Language Models](#build-language-models)
to add or replace models.

## Unit Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

For a fast smoke-check (completes in under two minutes):

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -m "not slow"
```

For the full map of test files, opt-in smoke suites, frontier/evaluation
packets, and longer synthetic runs, see
[`docs/test_inventory.md`](docs/test_inventory.md).

## Quick Start

The repository ships with two small benchmark fixtures so you can confirm a
fresh clone works end-to-end without any external data or API keys.

**Solve an English Borg analog (simple substitution with intentionally bad
word boundaries):**

```bash
decipher benchmark fixtures/benchmarks/english_borg_analog \
  --split english_borg_analog.jsonl
```

**Solve the Zodiac 340 known transform replay (transposition + homophonic):**

```bash
decipher benchmark fixtures/benchmarks/zodiac340_known_replay \
  --split zodiac340_known_replay.jsonl \
  --test-id zodiac340_known_replay \
  --transform-search rank
```

(Mixed transposition+homophonic ciphers require either an explicit pipeline
or a transform-search profile — see [Cipher Support](#transposition--homophonic).)

**Run the same English Borg analog with the agentic LLM solver** (requires
an API key — see [API key setup](#api-key-setup)):

```bash
decipher benchmark fixtures/benchmarks/english_borg_analog \
  --split english_borg_analog.jsonl \
  --agentic --model claude-sonnet-4-6
```

For more cipher families and tuning options, see
[Cipher Support](#cipher-support). For the complete agentic workflow, see
[Experimental Agentic Solving](#experimental-agentic-solving).

## Run the Historical Benchmark

`decipher benchmark` runs the automated solver against the curated historical
manuscript benchmark. The benchmark data lives in a separate repository:

- [cipher_benchmark](https://github.com/matthewdgreen/cipher_benchmark)

Clone it locally and substitute your checkout path below anywhere you see
`/path/to/cipher_benchmark/benchmark`.

```bash
# Borg Latin manuscript
decipher benchmark /path/to/cipher_benchmark/benchmark --source borg

# Copiale German manuscript
decipher benchmark /path/to/cipher_benchmark/benchmark --source copiale

# Single test by ID
decipher benchmark /path/to/cipher_benchmark/benchmark \
  --test-id borg_single_B_borg_0045v --verbose
```

To crack an arbitrary cipher from a file or stdin, use `decipher crack`:

```bash
# From a text file
decipher crack -f cipher.txt --language en

# From stdin
echo "T H E | Q U I C K | F O X" | decipher crack --language en

# Canonical S-token transcription
echo "S025 S012 S006 | S003 S007" | decipher crack --canonical --language la
```

## Experimental Agentic Solving

The agentic solver uses an LLM-driven loop for hypothesis exploration,
multi-step tool use, and cipher-type identification. It is an explicit opt-in
via `--agentic`.

```bash
decipher crack -f cipher.txt --language en --agentic

decipher benchmark /path/to/cipher_benchmark/benchmark \
  --source borg \
  --agentic \
  --model claude-sonnet-4-6 \
  --max-iterations 15
```

The automated solver runs as a free preflight pass before the agent starts, so
`--agentic` strictly adds capability on top of the automated path.

Agentic runs receive the automated preflight branch by default. The v2 tool
loop can invoke the local automated stack directly via `search_automated_solver`
and exposes all major solver paths — monoalphabetic anneal, homophonic anneal,
transform+homophonic, and periodic polyalphabetic — as dedicated tools. The
loop also includes a battery of observation tools, a reading-repair discipline
with a durable repair agenda, and benchmark context inspection tools that let
the agent examine related records and known solutions.

### Cipher-type identification

Before calling any solve tool, the agent receives a cipher-type fingerprint in
its initial context. This fingerprint is computed from cheap statistical
signals:

- **IC (Index of Coincidence)** — near language reference → monoalphabetic;
  depressed → polyalphabetic or homophonic
- **Normalized entropy / frequency flatness** — peaked → monoalphabetic;
  flat → homophonic or polyalphabetic
- **Periodic IC (Friedman)** — peak at period *k* recovering toward language
  IC → Vigenère key length *k*
- **Kasiski spacing GCDs** — repeated-trigram spacing factors → corroborate
  Vigenère period
- **Doubled-digraph rate** — near-zero or halved from random → Playfair
- **Alphabet size vs. unique symbols** — large gap → homophonic

The fingerprint produces a ranked suspicion list (`monoalphabetic_substitution`,
`homophonic_substitution`, `polyalphabetic_vigenere`, `transposition_homophonic`,
`playfair`, …) that the agent uses to prioritize which solver to try first.
The `observe_cipher_id` tool lets the agent re-run the fingerprint mid-run
after applying a transform.

### External context injection

You can inject free-form context (date, source, suspected technique) into the
agent's initial prompt before its first tool call:

```bash
# Inline text
decipher crack -f cipher.txt --agentic \
  --context "Found in an 18th-century French manuscript. Believed to be a Vigenère cipher."

# From a file
decipher crack -f cipher.txt --agentic \
  --context-file notes/cipher_notes.txt

# Both (inline text is prepended to the file contents)
decipher crack -f cipher.txt --agentic \
  --context "Key length may be 7." \
  --context-file notes/cipher_notes.txt
```

The same flags work on `decipher benchmark --agentic`.

### Generate and solve a synthetic test

`decipher testgen` generates a synthetic plaintext (cached after first
generation) and solves the resulting cipher.

```bash
# Cache a generated plaintext but skip solving
decipher testgen --preset tiny --language en --dry-run

# Solve a cached synthetic case with the agentic solver
decipher testgen --preset hardest --language en --agentic --model claude-sonnet-4-6
```

Presets:

| Preset | Words | Word boundaries | Cipher type |
|--------|-------|-----------------|-------------|
| `tiny` | ~40 | yes | simple substitution |
| `medium` | ~200 | yes | simple substitution |
| `hard` | ~250 | no | simple substitution |
| `hardest` | ~200 | no | homophonic substitution |

### Resume a prior agentic run

The `resume-artifact` command continues a saved agentic run, restoring the
workspace branches and running additional iterations:

```bash
decipher resume-artifact artifacts/my_cipher/run_abc123.json \
  --extra-iterations 10 \
  --model claude-sonnet-4-6
```

Useful options:

- `--branch NAME` — focus on a specific workspace branch from the prior run
- `--extra-iterations N` — additional iterations to run (default: 10)
- `--artifact-dir` — where to save the continuation artifact

### Inspect and diagnose artifacts

Agentic and automated runs write JSON artifacts under `artifacts/`. Use
`scripts/inspect_artifact.py` for a compact human-readable summary of a saved
run:

```bash
PYTHONPATH=src .venv/bin/python scripts/inspect_artifact.py \
  artifacts/my_cipher/run_abc123.json
```

Add `--analyze` to the inspector to ask an LLM to review any saved artifact and
explain success or failure modes. The analysis packet includes the tool
timeline, per-tool timing from `elapsed_ms`, branch scores, failed tool calls,
automated preflight, repair agenda, non-LLM analyzer findings, and decrypt
previews. Reported `char`/`word` scores are explicitly post-hoc comparisons to
known benchmark plaintext, not runtime-visible confidence.

```bash
PYTHONPATH=src .venv/bin/python scripts/inspect_artifact.py \
  artifacts/my_cipher/run_abc123.json \
  --analyze \
  --provider openai \
  --model gpt-5.4 \
  --analysis-mode deep
```

The script prints `Performing LLM analysis...` before the API call and reports
token usage plus estimated cost when the provider returns usage counters. The
default analysis output budget is 2,500 tokens; if a report ends mid-sentence,
re-run with a larger value such as `--analysis-max-tokens 5000`.

For agentic CLI runs, pass `--analyze` directly to `decipher benchmark`,
`decipher crack`, `decipher testgen`, or `decipher resume-artifact` to write a
sibling Markdown report next to each JSON artifact. For
`artifacts/foo/bar.json`, the automatic report is written to
`artifacts/foo/bar.analyzed.md`.

### LLM provider and model selection

Pass `--model` to choose a model; the provider is inferred from the name
prefix (`claude-` → Anthropic, `gpt-`/`o1`/`o3`/`o4` → OpenAI, `gemini-` →
Gemini, `provider/model` → OpenRouter). Pass `--provider` explicitly if
needed.

```bash
# Anthropic
decipher crack -f cipher.txt --agentic --model claude-sonnet-4-6

# OpenAI
decipher crack -f cipher.txt --agentic --model gpt-5.4

# Gemini
decipher crack -f cipher.txt --agentic --model gemini-3-flash-preview

# OpenRouter — provider inferred from the "/" in the model name
decipher crack -f cipher.txt --agentic --model meta-llama/llama-3.3-70b-instruct
decipher crack -f cipher.txt --agentic --model qwen/qwen3-30b-a3b

# Ollama (local — no API key required; Ollama must be running)
decipher crack -f cipher.txt --agentic --provider ollama --model qwen3:14b
```

Recommended for historical manuscript analysis: `claude-sonnet-4-6`. OpenRouter
models offer cost savings (5–40× cheaper per token) at some quality trade-off.

> **OpenRouter tool-calling note:** Not all OpenRouter models reliably emit
> structured tool calls or reason well enough for agentic cipher-cracking.
> Tested as of May 2026:
>
> - `tencent/hy3-preview:free` — good tool-call discipline and reasoning;
>   free tier; confirmed solve on quagmire3 no-boundary ciphers
> - `meta-llama/llama-4-maverick` — untested but likely best Llama option
> - `qwen/qwen3-30b-a3b`, `mistralai/mistral-small-3.2-24b-instruct` — untested
> - `deepseek/deepseek-chat` (V3) — tool calls fire but gives up after
>   diagnostic-budget searches and ignores harness feedback asking for more
> - `meta-llama/llama-3.3-70b-instruct` — tool calls fire but reasoning
>   quality is very poor; loops on failing calls, ignores required tools
> - `deepseek/deepseek-r1` / `deepseek-r1-0528` — **broken**: embeds tool
>   calls as Markdown text; use `deepseek/deepseek-chat` instead
>
> For production use, `claude-sonnet-4-6` via Anthropic remains the most
> reliable choice.

> **Ollama note:** The agentic solver relies heavily on structured tool
> calling. Use a model with documented tool-use support (e.g.
> `qwen3:14b`, `qwen3:8b`, `llama3.1:8b`). Run `decipher doctor` to see
> which Ollama models are currently installed.
>
> **Context window:** Decipher's initial prompt is typically 20–25 K tokens.
> Ollama's per-model default context window is often only 4 096 tokens,
> which silently truncates the prompt. Set `OLLAMA_NUM_CTX=32768` (or
> higher) to ensure the full context is received. The default when using
> `--provider ollama` is already 32 768, but if you load the model via a
> custom Modelfile with a lower `num_ctx`, set this env var to override.

### Terminal display mode

Agentic runs support four display modes via `--display`:

| Mode | Description |
|------|-------------|
| `auto` | `pretty` on an interactive terminal, `raw` when piped (default) |
| `pretty` | Rich terminal UI with live iteration and tool panels |
| `raw` | Plain-text streaming output |
| `jsonl` | Machine-readable JSONL event stream |

`--verbose` overrides to a verbose text stream.

### API key setup

Agentic mode supports five providers. You only need a key for the provider
you intend to use.

| Provider | `--provider` value | Environment variable(s) | Default model |
|----------|--------------------|------------------------|---------------|
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-5.4` |
| Google Gemini | `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-3-flash-preview` |
| OpenRouter | `openrouter` or `or` | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct` |
| Ollama (local) | `ollama` | *(none — no key required)* | `qwen3:14b` |

The provider is inferred automatically: `claude-` → Anthropic,
`gpt-`/`o1`/`o3`/`o4` → OpenAI, `gemini-` → Gemini, `provider/name` →
OpenRouter. Ollama model names have no standard prefix; pass
`--provider ollama` explicitly. If no `--provider` or `--model` is given,
the first provider with a configured key is used (anthropic → openai →
gemini → openrouter → ollama).

**Four ways to supply a key** (tried in this order):

1. Environment variable:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   export OPENAI_API_KEY=sk-...
   export GEMINI_API_KEY=...
   export OPENROUTER_API_KEY=sk-or-...
   ```

2. `.env` file in the repo root or working directory:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

3. Key file at `.decipher_keys/<provider>_api_key` (repo root or working
   directory):
   ```bash
   echo "sk-ant-..." > .decipher_keys/anthropic_api_key
   echo "sk-or-..."  > .decipher_keys/openrouter_api_key
   ```

4. macOS Keychain — service `decipher`, accounts `anthropic_api_key`,
   `openai_api_key`, `gemini_api_key`, `openrouter_api_key`.

Run `decipher doctor` to verify which providers are configured and which
models are known. Run `decipher doctor --refresh-pricing` to pull fresh
OpenRouter pricing (cached for 24 h at `~/.config/decipher/openrouter_pricing.json`).

`--no-automated-preflight` suppresses the default no-LLM preflight pass before
an agentic run (the preflight is generally cheap and useful).

### Agent tool namespaces

The v2 agent loop exposes tools across several namespaces. See
[TOOLS.md](TOOLS.md) for the complete reference with per-tool parameter
tables and usage notes.

| Namespace | Representative tools | Purpose |
|-----------|---------------------|---------|
| `workspace_*` | fork, fork_best, create_hypothesis_branch, reject_hypothesis, hypothesis_cards, list_branches, branch_cards, delete, compare, merge | Branch and hypothesis management |
| `observe_*` | frequency, ic, isomorph_clusters, cipher_id, cipher_shape, periodic_ic, kasiski, phase_frequency, periodic_shift_candidates, homophone_distribution, transform_pipeline, transform_suspicion | Statistical observation |
| `decode_*` | show, show_phases, unmapped_report, ngram_heatmap, letter_stats, ambiguous_letter, absent_letter_candidates, diagnose, diagnose_and_fix, repair_no_boundary, validate_reading_repair, plan_word_repair, plan_word_repair_menu | Decryption display and diagnosis |
| `score_*` | panel, quadgram, dictionary | Multi-signal scoring |
| `corpus_*` | lookup_word, word_candidates | Dictionary and corpus lookup |
| `act_*` | set_mapping, bulk_set, anchor_word, clear_mapping, swap_decoded, split/merge cipher words, apply_word_repair, resegment_by_reading, resegment_from/window, apply_transform_pipeline, install_transform_finalists, rate_transform_finalist, set/adjust_periodic_key | Key mutations and structural edits |
| `search_*` | hill_climb, anneal, homophonic_anneal, automated_solver, transform_candidates, transform_homophonic, review_transform_finalists, periodic_polyalphabetic | Solver invocation |
| `repair_agenda_*` | list, update | Durable reading-repair bookkeeping |
| `inspect_*` / `list_*` | inspect_benchmark_context, list_related_records, inspect_related_transcription, inspect_related_solution, list_associated_documents, inspect_associated_document | Benchmark context examination |
| `run_python` | (one tool) | Escape hatch with required justification |
| `meta_*` | request_tool, declare_solution | Run control |

## Cipher Support

This section covers cipher-family-specific options for advanced runs.

### Monoalphabetic substitution

Simple monoalphabetic substitution is the default route for ciphers whose
fingerprint matches the language reference IC and whose alphabet size is close
to the plaintext alphabet. No flags are required:

```bash
decipher crack -f cipher.txt --language en
```

### Homophonic substitution

Homophonic ciphers (multiple cipher symbols per plaintext letter, e.g. Zodiac
408, Copiale) are routed automatically when the cipher's alphabet is larger
than the plaintext alphabet and the frequency distribution is flat. Routing
uses the `zenith_native` solver path by default.

Tuning options for `decipher crack` and `decipher benchmark`:

```bash
decipher crack -f cipher.txt \
  --language en \
  --homophonic-budget full \
  --homophonic-refinement family_repair
```

| `--homophonic-budget` | Description |
|---|---|
| `full` | Default; full-budget annealing |
| `screen` | Faster; useful for diagnostic runs |

| `--homophonic-refinement` | Description |
|---|---|
| `none` | Default; no second-stage repair |
| `two_stage` | Two-stage annealing pass |
| `targeted_repair` | Targeted local repair after baseline solve |
| `family_repair` | Homophone-family repair sweep |
| `null_masks` | Opt-in null/codeword bakeoff (experimental) |

Tuning environment variables (rarely needed):

- `DECIPHER_HOMOPHONIC_SEARCH_PROFILE=dev|full` — shrink broad search for
  local iteration
- `DECIPHER_HOMOPHONIC_REPAIR_PROFILE=dev|full` — shrink repair breadth for
  local iteration
- `DECIPHER_HOMOPHONIC_POLISH=1` — opt into the experimental shared
  no-boundary segmentation/repair pass for post-`zenith_native` continuous
  output

### Periodic polyalphabetic (Vigenère family)

Vigenère, Beaufort, Variant Beaufort, Gronsfeld, and Quagmire I–IV ciphers
route automatically when the cipher's periodic IC peaks at a key length and
Kasiski GCDs corroborate. No flags are required for blind Vigenère solves:

```bash
decipher crack -f cipher.txt --language en
```

**Quagmire 3** is Vigenère with a keyword-scrambled cipher alphabet (the
tableau rows use a keyed rather than standard A–Z order). Kryptos K1 and K2
are both Quagmire 3 ciphers. The automated periodic path supports these as
known-parameter keyed-Vigenère calibration records via a `PeriodicAlphabetKey`
model. Artifacts label this as `keyed_vigenere_known_replay` so it is clear
that the run is verifying supplied tableau/key metadata rather than recovering
an unknown key from ciphertext.

For supplied-tableau key recovery and keyword-tableau enumeration, set:

- `DECIPHER_KEYED_VIGENERE_MODE=search` — search the periodic key over
  candidate keyed alphabets/tableau keywords; records
  `keyed_vigenere_periodic_key_search`.
- `DECIPHER_KEYED_VIGENERE_MODE=tableau_search` — test the standard A-Z
  tableau first, then keyword-derived tableaux from
  `DECIPHER_KEYED_VIGENERE_TABLEAU_KEYWORDS`.
- `DECIPHER_KEYED_VIGENERE_MODE=alphabet_anneal` — experimental shared-tableau
  mutation with phase re-optimization. Treat as a research diagnostic rather
  than a robust blind Kryptos solver.

### Transposition + homophonic

For ciphers that may combine a token-order transposition with homophonic
substitution (e.g. Zodiac 340), `--transform-search` activates the transform
candidate engine:

```bash
# Automated suspicion diagnostics only (cheap, no solver probes)
decipher crack -f cipher.txt --transform-search screen

# Structural triage → solver probes → independent confirmation
decipher crack -f cipher.txt --transform-search rank

# Unlimited solver budget — use for final runs
decipher crack -f cipher.txt --transform-search full

# Promote specific candidates from a prior screen artifact
decipher crack -f cipher.txt \
  --transform-search promote \
  --transform-promote-artifact artifacts/prior/run.json \
  --transform-promote-top-n 5
```

Transform search modes (`--transform-search`):

| Mode | Description |
|------|-------------|
| `off` | Disabled (default) |
| `auto` | Screen only when suspicion router signals are strong |
| `screen` | Record the structural candidate menu; no solver probes |
| `wide` | Larger structural-only sweep with extended candidate breadth |
| `rank` | 3-stage: structural triage → solver probes → independent confirmation |
| `full` | Like `rank` but with unlimited homophonic solver budget |
| `promote` | Probe specific candidates from a prior `screen`/`wide` artifact |

Candidate breadth profiles (`--transform-search-profile`):

| Profile | Description |
|---------|-------------|
| `fast` | Trims mutations and confirmations; recommended for regression runs |
| `broad` | Default; good balance of breadth and runtime |
| `wide` | Expanded structural sweep with more grid dimensions |

The Zenith-native transform rank path uses the Rust fast-kernel by default for
large solver-backed finalist checks:

```bash
decipher crack -f cipher.txt \
  --transform-search rank \
  --transform-search-profile wide \
  --homophonic-budget full
```

With Rust enabled, `rank` plus `--homophonic-budget full` may automatically
escalate unstable screen-budget finalist probes to full-budget ranking; the
artifact records this under `transform_search.rank_escalation`.

The transposition+homophonic frontier suite is at
`frontier/transposition_homophonic_ladder.jsonl`. The Zodiac 340 known-replay
fixture is at `frontier/zodiac340_known_replay.jsonl`.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py \
  --benchmark-split frontier/transposition_homophonic_ladder.jsonl \
  --transform-search rank
```

## Parallelism

Both the homophonic seed-parallel search and the Rust transform-rank engine
respect a single global worker count:

```bash
DECIPHER_PARALLEL_WORKERS=8 decipher crack -f cipher.txt --transform-search rank
```

Setting `DECIPHER_PARALLEL_WORKERS=1` is useful for deterministic single-threaded
profiling or CI runs where reproducible output is more important than speed.

## Build Language Models

Decipher can build Zenith-compatible binary n-gram models from public-domain
and licensed corpora (Project Gutenberg, OANC, MASC, BNC). The bundled English
model at `models/ngram5_en.bin` is enough for most use cases.

To build a fresh English model from Gutenberg + OANC + MASC:

```bash
PYTHONPATH=src .venv/bin/python -m tools.corpus run en \
  --source gutenberg --source oanc --source masc \
  --output models/ngram5_en.bin --max-books 100
```

To build a non-English Gutenberg-backed model (`de`, `fr`, `it`, `la`):

```bash
PYTHONPATH=src .venv/bin/python -m tools.corpus run de \
  --output models/ngram5_de.bin --max-books 100
```

Override the active model via environment variable:

```bash
DECIPHER_NGRAM_MODEL_EN=/path/to/other.bin decipher crack -f cipher.txt
```

For full source-by-source instructions including BNC licensed-import flow,
ANC TLS workarounds, model provenance, and large-corpus experiments, see
[`docs/language_models.md`](docs/language_models.md).

## Frontier and Parity Runs

```bash
PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py

PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/english_model_eval.jsonl \
  --solvers decipher
```

Both scripts also support `--solvers external` to run third-party solvers
(Zenith, zkdecrypto-lite) side-by-side. For installation steps, config files,
and comparison commands see
[`docs/external_solvers.md`](docs/external_solvers.md).

## Regression Suite

```bash
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --model claude-sonnet-4-6 \
  --max-iterations 20
```

Useful options:

- `--preset hardest`
- `--verbose`
- `--flush-cache`
- `--compare`

The fixed suite contains the same presets as `decipher testgen` (see
[Generate and solve a synthetic test](#generate-and-solve-a-synthetic-test)).
Tests that miss 100% character accuracy are copied to `errata/` with an
alignment report, verbose notes, and the full artifact.

## Errata Management

```bash
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py --list-errata

PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --rerun synth_en_250nb_s4

PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py --rerun-errata
```

## Rust Fast Kernels

Decipher requires the Rust/PyO3 module `decipher_fast` for normal CLI runs.
This avoids silently replacing broad compiled searches with slow Python
diagnostics. `scripts/setup_dev.sh` builds it automatically as part of first-time
setup.

To rebuild the module after changing Rust code:

```bash
scripts/build_rust_fast.sh
```

To verify the module is present and on the expected path:

```bash
PYTHONPATH=src .venv/bin/decipher doctor
```

If `decipher_fast` is missing, `benchmark`/`crack`/`testgen` runs abort
immediately with build instructions. The remaining Python fallback path is
reference and diagnostic scaffolding only; do not treat it as a runtime
fallback for large-scale searches.

## License

Decipher is licensed under the GNU General Public License, version 3. See
`LICENSE`.

## Attribution

The `zenith_native` homophonic solver in `src/analysis/zenith_solver.py` is
derived from the Zenith project by beldenge:

- [Zenith](https://github.com/beldenge/Zenith)

That solver path was adopted because it materially outperformed the earlier
native homophonic search. Decipher was therefore relicensed under GPLv3 so
this derived solver can be redistributed with explicit attribution and license
compatibility.

The original Zenith English binary model is not redistributed here. Decipher
includes tooling to build replacement language models from open and licensed
corpora.

Current provenance understanding:

- The `zenith_native` solver code path is redistributable under GPLv3 with
  attribution.
- The Zenith English binary model is still treated as **legally unresolved**
  for redistribution in Decipher.
- Earlier concern that **BNC** alone blocked redistribution turned out to be
  too pessimistic; BNC-derived products appear to be allowed.
- The main remaining uncertainty is the **Blog Authorship Corpus**, which
  Zenith documents as part of its training mix and which appears to be limited
  to **non-commercial research use**.

So for now, Decipher does **not** bundle the original Zenith model and instead
ships Decipher-built replacement models.
