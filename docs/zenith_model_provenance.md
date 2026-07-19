# Zenith English Model Provenance

Decipher redistributes an unchanged copy of Zenith's pre-trained English
5-gram model at `models/ngram5_en_zenith.bin`. The model is the default English
continuous language model because it remains stronger than Decipher's smaller
homegrown models on the current English homophonic evaluation packets.

## Exact Artifact

| Field | Value |
|---|---|
| Upstream project | [beldenge/Zenith](https://github.com/beldenge/Zenith) |
| Upstream release | [2026.2](https://github.com/beldenge/Zenith/releases/tag/2026.2) |
| Upstream commit | `2609112011de18c67bf61d980ba998fdc68d198f` |
| Upstream filename | `zenith-model.array.bin` |
| Decipher filename | `models/ngram5_en_zenith.bin` |
| Size | 47,526,004 bytes |
| SHA-256 | `0fc92b96a6018347e936eff9417b3b54a8e144b016ea921c7769f1be0f24fe63` |
| Format | Zenith binary v1, order 5, 26^5 float32 array |
| Distinct observed 5-grams | 2,642,405 |
| Total 5-gram observations | 560,416,358 |

The file was copied without modification. Its machine-readable provenance is
recorded in `models/ngram5_en_zenith.bin.metadata.json`.

## Training Sources

Zenith's language-model documentation says the supplied model was built from:

- British National Corpus, XML Edition
- Leipzig Corpora Collection, English 2005
- Manually Annotated Sub-Corpus (MASC)
- Blog Authorship Corpus

The raw corpora are not included in Decipher. The upstream source repository
contains the model builder and names these four sources in
`zenith-language-model/README.md`.

## Redistribution Decision

Zenith tracks and releases the pre-trained model in its GPLv3 repository, and
Decipher is also GPLv3. On that basis, Decipher redistributes the unchanged
model with explicit attribution and provenance. This is a project decision,
not a claim that every underlying corpus license has been independently
adjudicated.

The known caveat is the Blog Authorship Corpus, whose source page describes a
non-commercial research purpose. Decipher is intended for non-commercial
research, but the repository's GPLv3 license itself does not impose a
non-commercial restriction. If a rights holder or upstream maintainer raises a
concern, maintainers should reassess promptly and may remove the bundled model
while retaining the Decipher-built alternatives.

## Alternatives and Selection

The bundled upstream model is selected by the registry variant
`zenith_upstream`. Decipher-built variants remain available, including:

- `gutenberg` at `models/ngram5_en.bin`
- `parity` at `models/ngram5_en_parity.bin`
- `mixed` at `models/ngram5_en_mixed.bin`

An explicit `DECIPHER_NGRAM_MODEL_EN=/path/to/model.bin` override still takes
precedence over the registry default. This keeps licensing-sensitive or fully
reproducible deployments able to select a Decipher-built model.

## Integrity Check

```bash
shasum -a 256 models/ngram5_en_zenith.bin
```

The expected digest is the SHA-256 value in the table above and in the model
sidecar.
