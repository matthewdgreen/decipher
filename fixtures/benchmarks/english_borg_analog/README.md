# English Borg Analog Fixture

This tiny benchmark root contains one hand-shaped English case for debugging
agent reading behavior on Borg-like outputs.

The plaintext is readable archaic English:

```text
THEREFORE THE OLD PHYSICKER DID APPLY A SALVE UNTO THE SORE HEN AND THE FOWL DID LIVE AFTERWARD HE WROTE THAT MANY SUCH CURES WERE SWEET AND WITHOUT PAIN
```

The cipher is a simple substitution, but the canonical transcription has
intentionally misleading word boundaries:

```text
THERE | FORE | THE | OLD | PHYSICK | ER | DID | AP | PLY | A | SALVE | UN | TO | ...
```

This mirrors the Borg `0109v` failure mode where the decoded text can be
locally readable, while word alignment and exact word scoring stay poor.

Run it with:

```bash
PYTHONPATH=src .venv/bin/decipher benchmark fixtures/benchmarks/english_borg_analog \
  --split english_borg_analog.jsonl \
  --test-id english_borg_analog_001 \
  --agentic \
  --model claude-sonnet-4-6 \
  --max-iterations 12 \
  --verbose
```
