# Decipher

AI-assisted cryptanalysis of classical substitution ciphers, with a focus on
historical manuscripts. Uses Claude as the decipherment agent.

## Setup

```bash
cd ~/Dropbox/src2/decipher
source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=sk-...   # or store in macOS Keychain (service: decipher)
```

---

## Deciphering

### Crack a cipher from a file or stdin

```bash
# From a text file (space-separated letters, | as word separator)
decipher crack -f cipher.txt --language en

# From stdin
echo "T H E | Q U I C K | F O X" | decipher crack --language en

# Canonical S-token transcription (historical manuscript format)
echo "S025 S012 S006 | S003 S007" | decipher crack --canonical --language la

# Options
decipher crack -f cipher.txt \
  --language en          # en | la | de | fr | it (default: en)
  --model claude-sonnet-4-6 \
  --max-iterations 25 \
  --verbose              # show agent reasoning
```

### Run against the historical benchmark

```bash
# Borg Latin manuscript (15 transcription→plaintext tests)
decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark \
  --source borg --model claude-sonnet-4-6

# Single test by ID
decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark \
  --test-id borg_single_B_borg_0045v --model claude-sonnet-4-6 --verbose

# Copiale German cipher
decipher benchmark ~/Dropbox/src2/cipher_benchmark/benchmark \
  --source copiale --model claude-sonnet-4-6 --max-iterations 20
```

### Generate and crack a synthetic test (one-off)

```bash
# Dry-run: generate and cache plaintext, print it, skip the agent
decipher testgen --preset tiny --language en --dry-run

# Full run
decipher testgen --preset medium --language en --model claude-sonnet-4-6

# Presets:
#   tiny    (~40w,  word-boundary, simple substitution)
#   medium  (~200w, word-boundary, simple substitution)
#   hard    (~250w, no word-boundary, simple substitution)
#   hardest (~200w, no word-boundary, homophonic substitution — hardest to crack)
# --seed N  makes the cipher key reproducible across runs
```

---

## Regression test suite

Runs four synthetic tests (tiny / medium / hard / hardest) with fixed seeds
so the exact same ciphertext is produced on every run.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --model claude-sonnet-4-6 \
  --max-iterations 20

# Extra options
  --preset hardest   # run only one fixed preset: tiny | medium | hard | hardest
  --verbose          # show agent reasoning while running
  --flush-cache      # regenerate all plaintexts before running
  --compare          # also run a one-shot + code baseline on every test
```

The four fixed tests are:

| Preset   | Words | Word boundaries | Cipher type          | Seed |
|----------|-------|-----------------|----------------------|------|
| tiny     | ~40   | yes             | simple substitution  | 1    |
| medium   | ~200  | yes             | simple substitution  | 3    |
| hard     | ~250  | no              | simple substitution  | 4    |
| hardest  | ~200  | no              | homophonic (58 syms) | 5    |

The **hardest** test uses a homophonic substitution cipher: each plaintext
letter maps to one of 1–4 two-digit numeric tokens, with high-frequency
letters (E, T, A, …) receiving more homophones to defeat frequency analysis.

`run_python` remains an allowed agent tool because it can be useful for novel
analysis, but each call must include the agent's justification. Suite and errata
reports highlight those calls so they can be treated as evidence that a better
first-class tool may be needed.

Tests that score below 100% character accuracy are saved to `errata/` with a
verbose report, alignment diff, and the full agent reasoning log.

### Baseline comparison

`--compare` runs an additional one-shot baseline on every test immediately
after the agent. The baseline gives Claude a single `run_python` tool (stdlib
only, no domain knowledge) and asks it to derive the solution using code.
The final report shows three tables: agent results, baseline results, and a
delta comparison (positive = agent outperforms baseline).

---

## Errata management

```bash
# List all active errata (test ID, last score, run count, timestamp)
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py --list-errata

# Re-run a specific test by ID
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --rerun synth_en_250nb_s4

# Re-run multiple tests
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --rerun synth_en_250nb_s4 synth_en_200wb_s3

# Re-run all active errata at once
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py --rerun-errata
```

Re-runs always record results (pass or fail) so history accumulates under
`errata/{test_id}/{timestamp}/`. Once a test scores 100% on two consecutive
re-runs it is automatically archived to `errata/archive/` and removed from
the active list.

---

## Unit tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```
