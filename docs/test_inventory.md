# Test Inventory

This file is the working map of Decipher's test and evaluation surface. Keep it
updated when adding new test files, opt-in smoke suites, or live evaluation
packets.

## Everyday Checks

Run the default local suite:

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

Current expected shape: all tests pass, with the historical Milestone 4 smoke
suite skipped unless explicitly enabled.

## Pytest Files

| File | What It Protects | Notes |
|---|---|---|
| `tests/test_models.py` | Core `Alphabet` and `CipherText` behavior. | Token encoding/decoding, multisymbol basics, word structure. |
| `tests/test_ciphers.py` | Classical cipher primitives. | Substitution and Caesar encryption/decryption sanity. |
| `tests/test_analysis.py` | Frequency, IC, and pattern-analysis helpers. | Low-level analysis primitives. |
| `tests/test_signals.py` | Signal panel and n-gram scoring. | Protects multi-signal scoring inputs used by agents/tools. |
| `tests/test_segment.py` | DP segmentation and no-boundary repair helpers. | Includes one-edit corrections and dictionary-assisted repair. |
| `tests/test_workspace.py` | Workspace and branch state. | Branch lifecycle, key edits, merges, boundary/transform overlays. |
| `tests/test_benchmark.py` | Benchmark loader and scoring. | Includes character/word scoring, insertion-friendly alignment, reports. |
| `tests/test_benchmark_validator.py` | Benchmark checkout validator. | Uses tiny fake benchmark fixtures. |
| `tests/test_automated_runner.py` | Automated/no-LLM solver runner and CLI bypass behavior. | Protects automated preflight, routing, artifacts, refinement options. |
| `tests/test_homophonic_anneal.py` | Homophonic and substitution annealing. | Includes continuous n-gram models, Zenith CSV/binary dispatch surfaces, postprocess metadata. |
| `tests/test_zenith_solver.py` | Zenith-parity native homophonic solver. | Binary model format, entropy score, acceptance rule, seeded recovery. |
| `tests/test_corpus_builder.py` | Corpus import/build/verify tooling. | Gutenberg/OANC/MASC/BNC normalization and model metadata. |
| `tests/test_frontier_suite.py` | Frontier suite parsing/evaluation. | Synthetic specs, benchmark-backed cases, reports, transposition ladder loading. |
| `tests/test_external_baselines.py` | External baseline harness. | Wrapper config, candidate extraction, failure handling. |
| `tests/test_automated_parity_matrix.py` | Automated parity matrix script helpers. | Ranges, artifact ingestion, matrix row generation. |
| `tests/test_parity_dashboard.py` | Artifact dashboard/report rendering. | Joins agent, external, split metadata, and parity caveats. |
| `tests/test_artifact_analyzer.py` | Artifact failure-label analyzer. | Flags wrong-tool use, premature declaration, score-overrode-reading, gated retries. |
| `tests/test_agent_reliability.py` | Agent-loop guardrails and provider adapter mechanics. | Fake-provider loop tests, tool gating, declaration discipline, OpenAI/Gemini adapter shape tests. |
| `tests/test_agent_display.py` | Raw/pretty/JSONL display formatting. | Live usage header, final summary compaction, readable tool summaries. |
| `tests/test_agent_resume.py` | Artifact resume/continuation. | Resume context, branch install, continuation declaration. |
| `tests/test_final_summary.py` | Human-readable final summaries. | Non-English summary, blocked declaration recovery, further-iteration notes. |
| `tests/test_cipher_transformers.py` | Transposition/homophonic transformer subsystem. | Transform semantics, known-pipeline replay, synthetic builder metadata, agent transform tool. |
| `tests/test_polyalphabetic.py` | Periodic polyalphabetic solver and tools. | Vigenere-family search, Kasiski/phase/shift diagnostics, keyed-Vigenere replay, unknown-symbol skipping, agent periodic branches, periodic key edits, automated routing, synthetic testgen generation. |
| `tests/test_milestone4_smoke.py` | Milestone 4 smoke packet. | Fast fake-provider agent tests run by default; historical automated baseline is opt-in. |

## Opt-In Pytest Suites

### Historical Milestone 4 Automated Smoke

This runs real benchmark-backed cases through the no-LLM automated runner. It
requires a local `cipher_benchmark` checkout and can take a couple of minutes.

```bash
DECIPHER_RUN_MILESTONE4_SMOKE=1 \
DECIPHER_BENCHMARK_ROOT=/Users/mgreen/Dropbox/src2/cipher_benchmark/benchmark \
PYTHONPATH=src .venv/bin/pytest tests/test_milestone4_smoke.py -q
```

What it checks:

- `frontier/agentic_milestone4_smoke.jsonl` has the expected historical cases.
- Automated-only artifacts record `run_mode=automated_only`.
- Total LLM tokens and estimated cost are exactly zero.
- Baseline char-accuracy and elapsed-time thresholds are met.

## Evaluation Packets And Harnesses

These are not ordinary unit tests. They are reproducible evaluation runs that
write artifacts and summaries.

### Agent Model Packet: Borg 0109v

Runs one Borg Latin case across multiple LLM providers/models:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_agent_model_packet.py
```

Inputs:

- `frontier/agent_model_eval_0109v.jsonl`

Outputs:

- `artifacts/agent_model_eval/summary.jsonl`
- `artifacts/agent_model_eval/summary.csv`
- `artifacts/agent_model_eval/<model-name>/.../*.json`

Use this for live provider comparisons. It consumes API credits. It currently
covers Claude, OpenAI, and Gemini model rows. The runner preserves previous
summary rows on partial reruns and has a per-row timeout.

### Frontier Automated Suite

Compact automated frontier/regression packet:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/automated_solver_frontier.jsonl \
  --solvers decipher
```

Use this for no-LLM automated solver regressions and comparisons against
external baselines.

### English Model Comparison Packets

Used to compare continuous English n-gram binaries with identical solver
settings:

```bash
DECIPHER_NGRAM_MODEL_EN=models/ngram5_en_parity.bin \
DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native \
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/english_model_comparison.jsonl \
  --solvers decipher
```

Also available:

- `frontier/english_model_eval.jsonl` for a narrower smoke packet.
- `frontier/english_model_comparison.jsonl` for a broader Zodiac-like packet.

### Transposition/Homophonic Ladder

Synthetic capability ladder for known transform replay:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/transposition_homophonic_ladder.jsonl \
  --solvers decipher
```

Use this before claiming progress on Z340-style transform+homophonic cases.

Known Z340 replay fixture:

```bash
DECIPHER_HOMOPHONIC_PARALLEL_SEEDS=8 \
DECIPHER_NGRAM_MODEL_EN=other_tools/zenith-2026.2/zenith-model.array.bin \
DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native \
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/zodiac340_known_replay.jsonl \
  --benchmark-root fixtures/benchmarks/zodiac340_known_replay \
  --solvers decipher \
  --homophonic-budget full
```

This is a capability fixture, not raw Z340 discovery: it applies the known
Zenith transform pipeline before homophonic solving.

### Periodic Polyalphabetic Ladder

Synthetic Vigenere-family packet:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/polyalphabetic_ladder.jsonl \
  --solvers decipher
```

Current rows cover Vigenere, Beaufort, Variant Beaufort, and Gronsfeld. A
default pytest regression in `tests/test_polyalphabetic.py` runs this packet
end-to-end from cached local plaintext with no LLM access.

Kryptos keyed-Vigenere calibration:

```bash
PYTHONPATH=src .venv/bin/decipher benchmark ../cipher_benchmark/benchmark \
  --split kryptos_tests.jsonl \
  --automated-only
```

This is known-parameter replay for K1/K2, not unknown-key recovery. The tests
also include a compact K1 vector and a K2 replay with skipped `?` symbols.

To force supplied-tableau periodic-key recovery rather than known-key replay:

```bash
DECIPHER_KEYED_VIGENERE_MODE=search \
PYTHONPATH=src .venv/bin/decipher benchmark ../cipher_benchmark/benchmark \
  --split kryptos_tests.jsonl \
  --test-id kryptos_k2_keyed_vigenere \
  --automated-only
```

This recovers K2's `ABSCISSA` key over the supplied `KRYPTOS` tableau. K1 is
currently a short-text stress case for future crib/context-aware scoring.

To run the first keyword-tableau enumeration layer, with standard Vigenere
tested before the `KRYPTOS` keyword tableau:

```bash
DECIPHER_KEYED_VIGENERE_MODE=tableau_search \
DECIPHER_KEYED_VIGENERE_TABLEAU_KEYWORDS=KRYPTOS \
PYTHONPATH=src .venv/bin/decipher benchmark ../cipher_benchmark/benchmark \
  --split kryptos_tests.jsonl \
  --test-id kryptos_k2_keyed_vigenere \
  --automated-only
```

Experimental shared-tableau mutation search:

```bash
DECIPHER_KEYED_VIGENERE_MODE=alphabet_anneal \
DECIPHER_KEYED_VIGENERE_TABLEAU_KEYWORDS=KRYPTOS \
DECIPHER_KEYED_VIGENERE_ANNEAL_STEPS=2000 \
DECIPHER_KEYED_VIGENERE_ANNEAL_RESTARTS=4 \
PYTHONPATH=src .venv/bin/decipher benchmark ../cipher_benchmark/benchmark \
  --split kryptos_tests.jsonl \
  --test-id kryptos_k2_keyed_vigenere \
  --automated-only
```

This mode is a research diagnostic for shared-alphabet mutation/refinement.
It is not yet a reliable blind Kryptos-tableau recovery claim.

### Real English Challenge Packet

Solved and qualitative real English challenge cases:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/real_english_challenge.jsonl \
  --solvers decipher
```

See `docs/real_english_challenge_packet.md` for intent and caveats.

## Scripts With Their Own Tests

Some scripts have focused pytest coverage:

- `scripts/validate_benchmark.py` -> `tests/test_benchmark_validator.py`
- `scripts/run_frontier_suite.py` -> `tests/test_frontier_suite.py`
- `scripts/run_automated_parity_matrix.py` -> `tests/test_automated_parity_matrix.py`
- `scripts/build_parity_dashboard.py` / dashboard helpers -> `tests/test_parity_dashboard.py`
- `scripts/run_agent_model_packet.py` -> compile-checked and exercised by dry
  runs; add explicit unit coverage if the packet runner grows more logic.

## Maintenance Rules

- New test files should be added to the table above in the same PR/commit.
- New opt-in tests must document the enabling environment variable and expected
  local data requirements.
- New live/provider tests must be opt-in and must write an artifact summary.
- Fake-provider tests are preferred for loop mechanics; live provider packets
  should measure capability, latency, and cost.
