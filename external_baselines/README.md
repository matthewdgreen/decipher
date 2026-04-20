# External Baselines

This is a sidecar harness for comparing Decipher against external automatic
cipher solvers without coupling those tools to the main agent runtime.

The harness prepares neutral input files for each synthetic test, runs a
configured command with a timeout, captures stdout/stderr/output files, extracts
a plaintext candidate, and scores it with Decipher's existing scorer.

## Prepared Files

Each solver run gets its own directory under `artifacts/external_baselines/`:

- `input/canonical.txt` — original Decipher transcription, including `|` word separators when present
- `input/tokens.txt` — whitespace-separated cipher symbols with word separators removed
- `input/compact.txt` — symbols concatenated without separators
- `input/letters.txt` — token stream remapped to single-character symbols
- `input/metadata.json` — test metadata plus the token-to-letter remapping for `letters.txt`
- `input/ground_truth.txt` — scoring reference, included for artifact completeness
- `output/solution.txt` — conventional file for tools or wrappers to write plaintext

## Command Templates

Commands are JSON arrays. The harness expands these placeholders:

- `{canonical_file}`
- `{tokens_file}`
- `{compact_file}`
- `{letters_file}`
- `{metadata_file}`
- `{ground_truth_file}`
- `{input_dir}`
- `{output_dir}`
- `{output_file}`
- `{work_dir}`
- `{test_id}`
- `{cipher_system}`

Prefer wrapper scripts when a solver has awkward CLI requirements. A wrapper can
translate the prepared files into the solver's native format and write the final
plaintext to `{output_file}`.

## Example

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_external_baselines.py \
  --preset hardest \
  --config external_baselines/examples.json
```

The included config uses wrapper scripts and placeholder paths. The local
machine-specific config should live in `external_baselines/local_tools.json`,
which is ignored by git.

## Local Tool Notes

`zkdecrypto-lite`:

- Build the CMake project under `other_tools/zkdecrypto-build/`.
- Run it from the extracted source directory so it can find `language/eng`.
- Use `external_baselines/wrappers/zkdecrypto_lite.py` to normalize its final
  `score,PLAINTEXT` line.

`Zenith`:

- Requires Java 25 for the 2026.2 release jar.
- The release is a local Spring Boot UI/API server, not a batch CLI.
- Use `external_baselines/wrappers/zenith_graphql.py`; it starts Zenith on a
  temporary localhost port, submits a solve through GraphQL, listens for the
  final WebSocket solution update, then shuts the server down.
