# Real English Challenge Packet

Status: first packet assembled. The scored frontier file covers the real
English cases already imported into the benchmark. Additional unsolved or
mixed-mechanism ciphers are listed as qualitative candidates until they are
curated into benchmark records.

## Scored Packet

Runnable suite:

```bash
DECIPHER_HOMOPHONIC_PARALLEL_SEEDS=8 \
DECIPHER_NGRAM_MODEL_EN=models/ngram5_en_parity.bin \
DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native \
PYTHONPATH=src .venv/bin/python scripts/run_frontier_suite.py \
  --suite-file frontier/real_english_challenge.jsonl \
  --solvers decipher
```

Cases:

| Test ID | Cipher | Status | Mechanism | Role |
|---|---|---|---|---|
| `parity_tool_zenith_goldbug` | Gold-Bug | solved | simple substitution | easy no-boundary sanity check |
| `parity_tool_zenith_horacemann` | Horace Mann | solved | simple substitution | very short underdetermined check |
| `parity_tool_zenith_zodiac408` | Zodiac 408 | solved | homophonic substitution | real English homophonic parity anchor |
| `zodiac340_known_replay` | Zodiac 340 | solved | transposition + homophonic substitution | known-transform replay fixture in `frontier/zodiac340_known_replay.jsonl` |

These are the only non-synthetic English cases currently imported into the
main benchmark manifest.

Initial calibration run with `models/ngram5_en_parity.bin`,
`zenith_native`, and 8 parallel seeds:

| Test ID | Char Accuracy | Time | Result |
|---|---:|---:|---|
| `parity_tool_zenith_goldbug` | 99.5% | 11.8s | pass |
| `parity_tool_zenith_horacemann` | 100.0% | 5.0s | pass |
| `parity_tool_zenith_zodiac408` | 97.5% | 42.0s | pass |

## Current Solver Coverage

Decipher currently has strong support for:

- simple substitution
- no-boundary simple substitution
- homophonic substitution via `zenith_native`
- agentic reading, repair, and boundary workflows on top of automated preflight

Decipher now has first-pass known-transform replay for
transposition-plus-homophonic cases. This is enough to replay the Zenith Z340
transform pipeline and then run `zenith_native`, but it is not yet open-ended
transform discovery. Running the current automated path on raw Z340 still
mostly tests the wrong model of the cipher unless an explicit pipeline is
provided.

## Qualitative Candidates To Import Next

These local files are available but are not yet benchmark-scored records:

| Candidate | Local source | Status | Why it matters |
|---|---|---|---|
| Zodiac 340 raw/search | `other_tools/zenith-src/zenith-inference/src/main/resources/ciphers/zodiac340-original.json` and `zodiac340-transformed.json`; also zkdecrypto `340.zodiac.*.txt` files | solved | known replay works; raw transform discovery remains a future capability target |
| Zodiac Z13 | `other_tools/zkdecrypto-src/zkdecrypto-lite/cipher/13.zodiac.mynameis.txt` | unsolved/underdetermined | very short qualitative reasoning case; not useful for strict scoring |
| Zodiac Z32 | `other_tools/zkdecrypto-src/zkdecrypto-lite/cipher/32.zodiac.button.txt` | unsolved/underdetermined | very short map-code case; needs context-aware mode to be meaningful |
| Zodiac Z153 | `other_tools/zkdecrypto-src/zkdecrypto-lite/cipher/153.zodiac.unsolved.txt` | unsolved/uncertain | larger unsolved Zodiac-family qualitative case |
| Dorabella | `other_tools/zkdecrypto-src/zkdecrypto-lite/cipher/88.elgar.dorabella.txt` | unsolved/contested | short real English-ish/literary cipher; useful for agent hypothesis discipline |
| Kryptos K1-K3 | `other_tools/zenith-src/zenith-inference/src/main/resources/ciphers/kryptos1.json` etc. | solved | good future import for polyalphabetic/transposition support, not current substitution parity |
| Kryptos K4 | `other_tools/zenith-src/zenith-inference/src/main/resources/ciphers/kryptos4.json` | unsolved/contested | qualitative challenge; likely out of current solver scope |
| James Hampton | `other_tools/zenith-src/zenith-inference/src/main/resources/ciphers/jameshampton1.json`, `hamptonfull.json` | unsolved/unknown system | diagnostic-only; likely not a classical substitution benchmark |

## Suggested Next Import Work

1. Add qualitative unsolved records for Z13, Z32, Z153, Dorabella, K4, and
   Hampton under a separate unsolved/challenge split.
2. Teach the agent prompt and artifact schema to distinguish:
   - scored solved cases,
   - unsolved hypothesis cases,
   - unsupported mechanism cases,
   - and context-aware challenge cases.
3. Continue Z340 work by separating three modes in reports:
   known transform replay, bounded transform search, and raw/open-ended
   transform discovery.
