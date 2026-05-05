# Build Language Models

Decipher can build Zenith-compatible binary n-gram models from public-domain
corpora and licensed local sources.

## Source Summary

| Source | Languages | Tooling path | Notes |
|--------|-----------|--------------|-------|
| Project Gutenberg | `en`, `de`, `fr`, `it`, `la` | automatic download | Good bootstrap source, but literary-skewed |
| OANC | `en` | automatic download | Official ANC archive; tooling handles current TLS issue |
| MASC | `en` | automatic download | Official ANC archive; small but balanced |
| BNC | `en` | local licensed import | Raw corpus not redistributed; derive models only |

## Quick Reference

```bash
# Single-source English from Gutenberg
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

## Source-Specific Instructions

### Gutenberg

Fully automatic through `tools.corpus` for all currently supported languages.

One wrinkle for Latin: the current Project Gutenberg catalog only yields about
101 texts tagged `la` under the tool's `Type=text` filter, so the
`ngram5_la_500.bin` experiment is best read as "all currently available
Gutenberg Latin texts with a `max_books=500` cap", not literally 500 Latin
books.

### OANC

Fully automatic through `tools.corpus`.

Official source pages:
- [Open ANC overview](https://anc.org/data/oanc/)
- [Open ANC download page](https://anc.org/data/oanc/download/)

The current tooling caches the downloaded archive under
`corpus_data/<lang>/_archives/` and relaxes TLS verification only for the
official `anc.org` hosts because the site currently serves an expired
certificate.

### MASC

Fully automatic through `tools.corpus`.

Official source pages:
- [MASC overview](https://anc.org/data/masc/)
- [MASC data downloads](https://anc.org/data/masc/downloads/data-download/)

As with OANC, the tooling caches the archive locally and handles the current
ANC TLS issue automatically.

### BNC

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

## Selecting and Overriding Models

To force the automated solver to use a specific model:

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

## Source Access Notes

- Gutenberg is fetched as plain text files.
- OANC and MASC are fetched from the official ANC site as archives.
- BNC is supported as a licensed local source via `--source bnc --bnc-source-dir ...`;
  Decipher records attribution/provenance and emits only derived models, not corpus text.
- Non-English models currently use Gutenberg-backed downloads through the same tooling.
- The ANC site currently serves an expired TLS certificate, so the corpus tooling
  relaxes certificate verification only for `anc.org` / `www.anc.org`.
- Model metadata automatically records the source list and provenance.
