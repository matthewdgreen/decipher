# English Copiale Analog Fixture

This local benchmark fixture is a synthetic, readable-English analog of a
Copiale-like nomenclator. It is for intuition-building and solver diagnostics,
not historical evaluation.

The plaintext is invented archaic lodge prose. The cipher uses:

- homophonic symbols for A-Z letters
- null symbols
- whole-word logograms for: THE, AND, OF, THAT, MASTER, BROTHER, BRETHREN, LODGE, SIGN, HAND, ORDER

Generation parameters:

- seed: `1729`
- null rate: `0.055`
- logogram rate: `0.82`

The secret key is written to
`sources/english_copiale_analog/metadata/english_copiale_analog_001.key.json`
for post-hoc diagnostics only.

Automated run:

```bash
PYTHONPATH=src .venv/bin/decipher benchmark fixtures/benchmarks/english_copiale_analog \
  --split english_copiale_analog.jsonl \
  --test-id english_copiale_analog_001 \
  --automated-only \
  --homophonic-budget screen \
  --homophonic-refinement null_masks \
  --artifact-dir artifacts/english_copiale_analog_automated
```

Agentic run:

```bash
PYTHONPATH=src .venv/bin/decipher benchmark fixtures/benchmarks/english_copiale_analog \
  --split english_copiale_analog.jsonl \
  --test-id english_copiale_analog_001 \
  --agentic \
  --provider openai \
  --model gpt-5.4 \
  --benchmark-context standard \
  --max-iterations 30 \
  --homophonic-budget screen \
  --homophonic-refinement null_masks \
  --artifact-dir artifacts/english_copiale_analog_agentic \
  --analyze
```
