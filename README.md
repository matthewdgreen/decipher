# Decipher

AI-assisted cryptanalysis of classical substitution ciphers, with a focus on
historical manuscripts. Uses Claude as the decipherment agent.

## Setup

```bash
cd ~/Dropbox/src2/decipher
source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=sk-...   # or store via the GUI (macOS Keychain)
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
decipher testgen --preset easy --language en --model claude-sonnet-4-6

# Presets: tiny (~40w, word-boundary), easy (~100w), medium (~200w), hard (~250w, no-boundary)
# --seed N  makes the cipher key reproducible across runs
```

---

## Regression test suite

Runs four synthetic tests (tiny / easy / medium / hard) with fixed seeds so
the exact same ciphertext is produced on every run.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py \
  --model claude-sonnet-4-6 \
  --max-iterations 20

# Extra options
  --verbose          # show agent reasoning while running
  --flush-cache      # regenerate all plaintexts before running
```

Tests that score below 100% character accuracy are saved to `errata/` with a
verbose report, alignment diff, and the full agent reasoning log.

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
