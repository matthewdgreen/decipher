# Decipher

Decipher is a tool for performing automated cryptanalysis for classical substitution 
ciphers, with a focus on historical manuscripts. The goal of this tool is to achieve
parity, and then improve on the state-of-the-art in automated cipher solver tools.
As a second and more experimental goal, decipher is intended to (optionally) integrate 
with an agentic LLM, enabling the LLM to perform manual solving steps that improve our
ability to cryptanalyze challenging ciphertexts.

Decipher's primary mode is its native automated solver stack: fast, reproducible, and
usable with only local computation. The experimental agentic solver requires an API key
and is documented separately below.

Decipher borrows solving algorithms (with attribution and license compliance) from the 
Zenith solving tool. For licensing reasons we do not redistribute Zenith's ngram models,
and instead provide our own (as well as tooling to generate additional models.)

## License

Decipher is licensed under the GNU General Public License, version 3. See
`LICENSE`.

## Attribution

The `zenith_native` homophonic solver in
`src/analysis/zenith_solver.py`
is derived from the Zenith project by beldenge:

- [Zenith](https://github.com/beldenge/Zenith)

That solver path was adopted because it materially outperformed the earlier
native homophonic search. Decipher was therefore relicensed under GPLv3 so
this derived solver can be redistributed with explicit attribution and license
compatibility.

The original Zenith English binary model is not redistributed here.
Decipher includes tooling to build replacement language models from open and
licensed corpora.

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

```bash
cd /path/to/decipher
source .venv/bin/activate
pip install -e .
```

No API key is required for the default automated workflows.

This repository currently ships with a bundled English
5-gram model at `models/ngram5_en.bin`,
so a fresh clone can use the automated homophonic solver immediately. You can
override it with `DECIPHER_NGRAM_MODEL_EN=/path/to/other.bin`.

## Automated Solving

### Crack a cipher from a file or stdin

`decipher crack` now runs the native automated solver by default.

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

`decipher testgen` defaults to automated solving too. Because synthetic
plaintext generation may itself require an LLM call, the default automated path
works only when that plaintext is already cached.

```bash
# Show the generated plaintext and cache it, but skip solving
decipher testgen --preset tiny --language en --dry-run

# Solve a cached synthetic case with the automated solver
decipher testgen --preset hardest --language en
```

Presets:

- `tiny`: ~40 words, word boundaries, simple substitution
- `medium`: ~200 words, word boundaries, simple substitution
- `hard`: ~250 words, no boundaries, simple substitution
- `hardest`: ~200 words, no boundaries, homophonic substitution

### Automated Frontier / Parity Runs

The non-agentic solver stack is also what powers the automated parity and
frontier evaluation workflows:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py

PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/english_model_eval.jsonl \
  --solvers decipher
```

### Build Redistributable Language Models

Decipher can build Zenith-compatible binary n-gram models from public-domain
corpora and licensed local sources.

Source summary:

| Source | Languages | Tooling path | Notes |
|---|---|---|---|
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

# Build from a licensed local BNC copy
PYTHONPATH=src .venv/bin/python -m tools.corpus run en \
  --source bnc \
  --bnc-source-dir /path/to/licensed/bnc \
  --output models/ngram5_en_bnc.bin
```

#### Source-specific instructions

##### Gutenberg

Fully automatic through `tools.corpus` for all currently supported languages.

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

Official source pages and mirrors we found:
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

Then extract it somewhere local and point the corpus tool at that directory:

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
as `DECIPHER_NGRAM_MODEL_EN=/path/to/model.bin`. If no override is set, it then
looks for a repo-local bundled model such as `models/ngram5_en.bin`, and only
after that falls back to English-specific legacy Zenith locations.

Notes on source access:

- Gutenberg is fetched as plain text files.
- OANC and MASC are fetched from the official ANC site as archives.
- BNC is supported as a licensed local source via `--source bnc --bnc-source-dir ...`;
  Decipher records attribution/provenance and emits only derived models, not corpus text.
- Non-English models currently use Gutenberg-backed downloads through the same tooling.
- The ANC site currently serves an expired TLS certificate, so the corpus
  tooling relaxes certificate verification only for `anc.org` / `www.anc.org`
  in order to automate those downloads.
- Model metadata automatically records the source list and provenance carried in
  the corpus manifest.

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

| Preset   | Words | Word boundaries | Cipher type          | Seed |
|----------|-------|-----------------|----------------------|------|
| tiny     | ~40   | yes             | simple substitution  | 1    |
| medium   | ~200  | yes             | simple substitution  | 3    |
| hard     | ~250  | no              | simple substitution  | 4    |
| hardest  | ~200  | no              | homophonic (58 syms) | 5    |

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

The agentic solver is still available for research workflows where we want
branching hypotheses, tool use, and LLM-guided reasoning. It is now an
explicit opt-in mode via `--agentic`.

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

If you want `decipher testgen` to generate uncached plaintext and solve it in
one go, use `--agentic`:

```bash
decipher testgen --preset medium --language en --agentic --model claude-sonnet-4-6
```

### API key setup

Agentic mode requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-...
```

Or store it in the macOS Keychain under service `decipher`, account
`anthropic_api_key`.

`--no-automated-preflight` is available if you want to suppress the default
native preflight pass before an agentic run, though the preflight is generally
useful and cheap.

## Unit Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```
