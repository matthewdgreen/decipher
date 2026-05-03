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
python3.11 -m venv .venv
source .venv/bin/activate
scripts/setup_dev.sh
```

No API key is required for the default automated workflows.

This repository ships with a bundled English 5-gram model at
`models/ngram5_en.bin`, so a fresh clone can use the automated homophonic
solver immediately. You can override it with
`DECIPHER_NGRAM_MODEL_EN=/path/to/other.bin`. Automated homophonic runs
default to the `zenith_native` solver path. To exercise the older
pre-`zenith_native` code for comparison, pass `--legacy-homophonic`.

On multi-core machines, `zenith_native` auto-sizes parallel seed workers by
default. Override explicitly with `DECIPHER_HOMOPHONIC_PARALLEL_SEEDS=<N>`.

### Required Rust Fast Kernels

Decipher requires the Rust/PyO3 module `decipher_fast` for normal CLI runs.
This avoids silently replacing broad compiled searches with slow Python
diagnostics. The `scripts/setup_dev.sh` command above installs the Python
package and builds the Rust module.

If you only need to rebuild the Rust module after changing Rust code:

```bash
cd /path/to/decipher
scripts/build_rust_fast.sh
```

Minimal manual setup equivalent:

```bash
cd /path/to/decipher
source .venv/bin/activate
pip install -e .
.venv/bin/python -m pip install maturin
cd rust/decipher_fast
../../.venv/bin/python -m maturin develop --release
cd ../..
```

Check whether the module is available:

```bash
PYTHONPATH=src .venv/bin/decipher doctor
```

If `decipher_fast` is missing, benchmark/crack/testgen runs abort immediately
with build instructions. The remaining Python Quagmire path is reference and
diagnostic scaffolding only; do not treat it as a runtime fallback for
large-scale searches.

## Automated Solving

### Crack a cipher from a file or stdin

`decipher crack` runs the native automated solver by default.

```bash
# From a text file
decipher crack -f cipher.txt --language en

# From stdin
echo "T H E | Q U I C K | F O X" | decipher crack --language en

# Canonical S-token transcription
echo "S025 S012 S006 | S003 S007" | decipher crack --canonical --language la
```

Useful options:

```bash
decipher crack -f cipher.txt \
  --language en \
  --homophonic-budget full \
  --homophonic-refinement family_repair \
  --verbose
```

Homophonic tuning environment variables:

- `DECIPHER_HOMOPHONIC_PARALLEL_SEEDS=<N>` — override the auto-sized seed worker count
- `DECIPHER_HOMOPHONIC_SEARCH_PROFILE=dev|full` — shrink broad search for local iteration
- `DECIPHER_HOMOPHONIC_REPAIR_PROFILE=dev|full` — shrink repair breadth for local iteration
- `DECIPHER_HOMOPHONIC_POLISH=1` — opt into the experimental shared no-boundary
  segmentation/repair pass for post-`zenith_native` continuous output

### Transposition + homophonic search

For ciphers that may combine a token-order transposition with homophonic
substitution, `--transform-search` activates the transform candidate engine:

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
DECIPHER_TRANSFORM_RANK_THREADS=0 \
decipher crack -f cipher.txt \
  --transform-search rank \
  --transform-search-profile wide \
  --homophonic-budget full
```

`DECIPHER_TRANSFORM_RANK_THREADS=0` means "use all available cores." With
Rust enabled, `rank` plus `--homophonic-budget full` may automatically
escalate unstable screen-budget finalist probes to full-budget ranking; the
artifact records this under `transform_search.rank_escalation`.

For reference/regression comparisons against the older Python implementation,
set `DECIPHER_ZENITH_NATIVE_ENGINE=python` and
`DECIPHER_TRANSFORM_RANK_ENGINE=python` explicitly.

### Run the historical benchmark

`decipher benchmark` also uses the automated solver by default.

The benchmark data lives in a separate repository:

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

### Generate and solve a synthetic test

`decipher testgen` defaults to automated solving. Synthetic plaintext
generation may itself require an LLM call, so the default automated path works
only when that plaintext is already cached.

```bash
# Show the generated plaintext and cache it, but skip solving
decipher testgen --preset tiny --language en --dry-run

# Solve a cached synthetic case with the automated solver
decipher testgen --preset hardest --language en
```

Presets:

| Preset | Words | Word boundaries | Cipher type |
|--------|-------|-----------------|-------------|
| `tiny` | ~40 | yes | simple substitution |
| `medium` | ~200 | yes | simple substitution |
| `hard` | ~250 | no | simple substitution |
| `hardest` | ~200 | no | homophonic substitution |

### Automated Frontier / Parity Runs

```bash
PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py

PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/english_model_eval.jsonl \
  --solvers decipher
```

For the main automated frontier suite, external runs default to Zenith only,
which keeps routine comparisons fast:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers external
```

To include slower external wrappers such as `zkdecrypto-lite`, pass the full
external config explicitly:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers external \
  --external-config external_baselines/local_tools.json
```

The transposition+homophonic frontier suite (known-pipeline replay and
open-ended transform search) is at
`frontier/transposition_homophonic_ladder.jsonl`. The Zodiac 340 known-replay
fixture is at `frontier/zodiac340_known_replay.jsonl`.

```bash
# Run the transposition+homophonic ladder with transform search
PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py \
  --benchmark-split frontier/transposition_homophonic_ladder.jsonl \
  --transform-search rank
```

### Build Redistributable Language Models

Decipher can build Zenith-compatible binary n-gram models from public-domain
corpora and licensed local sources.

Source summary:

| Source | Languages | Tooling path | Notes |
|--------|-----------|--------------|-------|
| Project Gutenberg | `en`, `de`, `fr`, `it`, `la` | automatic download | Good bootstrap source, but literary-skewed |
| OANC | `en` | automatic download | Official ANC archive; tooling handles current TLS issue |
| MASC | `en` | automatic download | Official ANC archive; small but balanced |
| BNC | `en` | local licensed import | Raw corpus not redistributed; derive models only |

```bash
PYTHONPATH=src .venv/bin/python -m tools.corpus run en \
  --output models/ngram5_en.bin \
  --max-books 100

# Mix Gutenberg with OANC and MASC
PYTHONPATH=src .venv/bin/python -m tools.corpus run en \
  --source gutenberg \
  --source oanc \
  --source masc \
  --output models/ngram5_en.bin \
  --max-books 100

# Build non-English Gutenberg-backed models
PYTHONPATH=src .venv/bin/python -m tools.corpus run de --output models/ngram5_de.bin --max-books 100
PYTHONPATH=src .venv/bin/python -m tools.corpus run fr --output models/ngram5_fr.bin --max-books 100
PYTHONPATH=src .venv/bin/python -m tools.corpus run it --output models/ngram5_it.bin --max-books 100
PYTHONPATH=src .venv/bin/python -m tools.corpus run la --output models/ngram5_la.bin --max-books 100

# Larger Latin experiment
PYTHONPATH=src .venv/bin/python -m tools.corpus run la \
  --corpus-dir corpus_data/la_500 \
  --output models/ngram5_la_500.bin \
  --max-books 500

# Build from a licensed local BNC copy
PYTHONPATH=src .venv/bin/python -m tools.corpus run en \
  --source bnc \
  --bnc-source-dir /path/to/licensed/bnc \
  --output models/ngram5_en_bnc.bin
```

#### Source-specific instructions

##### Gutenberg

Fully automatic through `tools.corpus` for all currently supported languages.

One wrinkle for Latin: the current Project Gutenberg catalog only yields about
101 texts tagged `la` under the tool's `Type=text` filter, so the
`ngram5_la_500.bin` experiment is best read as "all currently available
Gutenberg Latin texts with a `max_books=500` cap", not literally 500 Latin
books.

##### OANC

Fully automatic through `tools.corpus`.

Official source pages:
- [Open ANC overview](https://anc.org/data/oanc/)
- [Open ANC download page](https://anc.org/data/oanc/download/)

The current tooling caches the downloaded archive under
`corpus_data/<lang>/_archives/` and relaxes TLS verification only for the
official `anc.org` hosts because the site currently serves an expired
certificate.

##### MASC

Fully automatic through `tools.corpus`.

Official source pages:
- [MASC overview](https://anc.org/data/masc/)
- [MASC data downloads](https://anc.org/data/masc/downloads/data-download/)

As with OANC, the tooling caches the archive locally and handles the current
ANC TLS issue automatically.

##### BNC

BNC is supported as a **licensed local import**, not a direct public downloader.
Decipher does not redistribute BNC corpus text; it only imports from your local
licensed copy and emits derived statistical models with explicit provenance.

Official source pages and mirrors:
- [OTA / Bodleian BNC XML Edition page](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554)
- [Direct OTA `2554.zip` bitstream](https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2554/2554.zip?isAllowed=y&sequence=3)
- [Oxford LLDS mirror](https://llds.ling-phil.ox.ac.uk/llds/xmlui/handle/20.500.14106/2554)
- [Oxford LLDS mirror (phonetics)](https://llds.phon.ox.ac.uk/llds/xmlui/handle/20.500.14106/2554)

Suggested resumable fetch command:

```bash
mkdir -p corpus_data/en/_archives && \
curl -L -C - --fail --output corpus_data/en/_archives/BNC-2554.zip \
  "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2554/2554.zip?isAllowed=y&sequence=3"
```

Then extract and point the corpus tool at that directory:

```bash
mkdir -p corpus_data/en/bnc_source && \
unzip -q corpus_data/en/_archives/BNC-2554.zip -d corpus_data/en/bnc_source

PYTHONPATH=src .venv/bin/python -m tools.corpus run en \
  --source bnc \
  --bnc-source-dir corpus_data/en/bnc_source \
  --output models/ngram5_en_bnc.bin
```

To force the automated solver to use that model:

```bash
DECIPHER_NGRAM_MODEL_EN=models/ngram5_en.bin \
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/english_model_eval.jsonl \
  --solvers decipher
```

By default, `zenith_native` first honors any explicit environment override such
as `DECIPHER_NGRAM_MODEL_EN=/path/to/model.bin`. If no override is set, it
looks for a repo-local bundled model such as `models/ngram5_en.bin`, and only
after that falls back to English-specific legacy Zenith locations.

Notes on source access:

- Gutenberg is fetched as plain text files.
- OANC and MASC are fetched from the official ANC site as archives.
- BNC is supported as a licensed local source via `--source bnc --bnc-source-dir ...`;
  Decipher records attribution/provenance and emits only derived models, not corpus text.
- Non-English models currently use Gutenberg-backed downloads through the same tooling.
- The ANC site currently serves an expired TLS certificate, so the corpus tooling
  relaxes certificate verification only for `anc.org` / `www.anc.org`.
- Model metadata automatically records the source list and provenance.

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

The fixed suite contains:

| Preset | Words | Word boundaries | Cipher type |
|--------|-------|-----------------|-------------|
| `tiny` | ~40 | yes | simple substitution |
| `medium` | ~200 | yes | simple substitution |
| `hard` | ~250 | no | simple substitution |
| `hardest` | ~200 | no | homophonic substitution |

Tests that miss 100% character accuracy are copied to `errata/` with an
alignment report, verbose notes, and the full artifact.

## Errata Management

```bash
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py --list-errata

PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --rerun synth_en_250nb_s4

PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py --rerun-errata
```

## Experimental Agentic Solving

The agentic solver uses an LLM-driven loop for hypothesis exploration,
multi-step tool use, and cipher-type identification. It is an explicit opt-in
via `--agentic`.

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
after applying a transform. For Vigenere-family suspicions, the agent can then
drill into `observe_kasiski`, `observe_phase_frequency`, and
`observe_periodic_shift_candidates` before running or repairing a periodic
solver branch.

The automated periodic path also supports known-parameter keyed Vigenere
calibration records, such as Kryptos K1/K2, using a `PeriodicAlphabetKey`
model. Artifacts label this as `keyed_vigenere_known_replay` so it is clear
that the run is verifying supplied tableau/key metadata rather than recovering
an unknown key from ciphertext. For supplied-tableau key recovery, set
`DECIPHER_KEYED_VIGENERE_MODE=search`; this searches the periodic key over
candidate keyed alphabets/tableau keywords and records
`keyed_vigenere_periodic_key_search`. For keyword-tableau enumeration, set
`DECIPHER_KEYED_VIGENERE_MODE=tableau_search`; this tests the standard A-Z
tableau first, then keyword-derived tableaux from
`DECIPHER_KEYED_VIGENERE_TABLEAU_KEYWORDS`. An experimental
`DECIPHER_KEYED_VIGENERE_MODE=alphabet_anneal` mode mutates the shared tableau
and re-optimizes phase shifts after each mutation; treat it as a research
diagnostic rather than a robust blind Kryptos solver.

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

The same flags work on `decipher benchmark --agentic`:

```bash
decipher benchmark /path/to/cipher_benchmark/benchmark \
  --agentic --source borg \
  --context "Borg cipher, monoalphabetic substitution, Latin pharmaceutical text."
```

### Crack with the agent

```bash
decipher crack -f cipher.txt --language en --agentic
```

### Benchmark with the agent

```bash
decipher benchmark /path/to/cipher_benchmark/benchmark \
  --source borg \
  --agentic \
  --model claude-sonnet-4-6 \
  --max-iterations 15
```

### Synthetic test generation and agentic solving

```bash
decipher testgen --preset medium --language en --agentic --model claude-sonnet-4-6
```

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

### LLM provider and model selection

Agentic mode supports multiple LLM providers:

```bash
# Anthropic (default)
decipher crack -f cipher.txt --agentic --model claude-sonnet-4-6

# OpenAI
decipher crack -f cipher.txt --agentic --provider openai --model gpt-4o

# Gemini
decipher crack -f cipher.txt --agentic --provider gemini --model gemini-2.0-flash
```

Recommended for historical manuscript analysis: `claude-sonnet-4-6`. The
provider is inferred automatically from a recognized model name, or pass
`--provider` explicitly.

### Terminal display mode

Agentic runs support four display modes via `--display`:

| Mode | Description |
|------|-------------|
| `auto` | `pretty` on an interactive terminal, `raw` when piped (default) |
| `pretty` | Rich terminal UI with live iteration and tool panels |
| `raw` | Plain-text streaming output |
| `jsonl` | Machine-readable JSONL event stream |

`--verbose` overrides to the legacy verbose text stream.

### API key setup

```bash
export ANTHROPIC_API_KEY=sk-...
```

Or store it in the macOS Keychain under service `decipher`, account
`anthropic_api_key`. OpenAI and Gemini keys follow the same pattern
(`OPENAI_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`).

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

## Unit Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```
