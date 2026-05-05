# External Solver Comparison

Decipher can run side-by-side comparisons against third-party solvers through a
thin wrapper harness. Results are recorded in the same artifact format as
Decipher's own automated runs so they can appear together in frontier reports.

Supported external solvers:

| Solver | Cipher class | Notes |
|--------|-------------|-------|
| Zenith | Homophonic substitution | Requires Java + the UI jar |
| zkdecrypto-lite | Homophonic substitution | Requires a local CMake build |

---

## Zenith

Zenith ships as a Spring Boot UI/server jar. The wrapper
(`external_baselines/wrappers/zenith_graphql.py`) starts the server on a
temporary localhost port, drives it via its GraphQL/WebSocket API, and writes
the best plaintext to a sidecar file.

### Prerequisites

- **Java 11+** — e.g. `brew install openjdk` on macOS. The wrapper defaults
  to `java` on `$PATH`; override with `--java /path/to/java`.
- **Zenith UI jar** — download the current release from the Zenith project page
  and place the jar at:

  ```
  other_tools/zenith-2026.2/zenith-ui-2026.2.jar
  ```

  The version directory and jar filename must match what is in
  `external_baselines/zenith_only.json` (update that file if you install a
  different version).

- **Zenith binary model** (optional) — the proprietary `zenith-model.array.bin`
  is read by Decipher's own `zenith_native` solver, not by the external Zenith
  wrapper. If you have it, place it alongside the jar:

  ```
  other_tools/zenith-2026.2/zenith-model.array.bin
  ```

  Then pass it via `DECIPHER_NGRAM_MODEL_EN` (see
  [Build Language Models](language_models.md)) to compare Decipher's
  `zenith_native` against the same n-gram model that Zenith itself uses.

### Quick smoke test

```bash
PYTHONPATH=src .venv/bin/python external_baselines/wrappers/zenith_graphql.py \
  --jar other_tools/zenith-2026.2/zenith-ui-2026.2.jar \
  --input /tmp/test_input.txt \
  --output /tmp/test_output.txt \
  --epochs 3 \
  --timeout-seconds 60
```

---

## zkdecrypto-lite

zkdecrypto-lite is an open-source homophonic solver written in C++. It must be
compiled locally and run from its source directory because it loads its language
model from a relative path (`language/eng`).

### Build

```bash
mkdir -p other_tools/zkdecrypto-build
cd other_tools/zkdecrypto-build
cmake ../zkdecrypto-src/zkdecrypto-lite
make -j"$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)"
```

This places the binary at `other_tools/zkdecrypto-build/zkdecrypto-lite`,
which is the path in `external_baselines/local_tools.json`.

### Quick smoke test

```bash
other_tools/zkdecrypto-build/zkdecrypto-lite /tmp/test_input.txt -t 10
```

---

## Config files

Two ready-made configs live in `external_baselines/`:

| File | Solvers included | Use case |
|------|-----------------|----------|
| `zenith_only.json` | Zenith | Default for `run_frontier_suite.py` external runs |
| `local_tools.json` | Zenith + zkdecrypto-lite | Full external comparison |

Both files point to the paths above. Edit them if your install locations differ.

---

## Running comparisons

### Frontier suite

```bash
# Zenith only (default external config)
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers external

# Zenith + zkdecrypto-lite
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers external \
  --external-config external_baselines/local_tools.json

# Decipher and Zenith side-by-side
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers decipher external
```

### Parity matrix

```bash
# --external-config is required when running external solvers here
PYTHONPATH=src .venv/bin/python scripts/run_automated_parity_matrix.py \
  --solvers external \
  --external-config external_baselines/zenith_only.json \
  --benchmark-split frontier/automated_solver_frontier.jsonl \
  --benchmark-root ~/Dropbox/src2/cipher_benchmark/benchmark \
  --artifact-dir artifacts/external_parity
```

### Decipher with the proprietary Zenith model

To compare Decipher's `zenith_native` solver using the same n-gram model that
Zenith itself ships:

```bash
DECIPHER_PARALLEL_WORKERS=8 \
DECIPHER_NGRAM_MODEL_EN=other_tools/zenith-2026.2/zenith-model.array.bin \
DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native \
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers decipher
```

Historical results from these comparison runs are tracked in
[`docs/frontier_solver_comparison.md`](frontier_solver_comparison.md).
