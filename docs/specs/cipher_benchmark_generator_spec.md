# Spec: On-Demand Cipher Benchmark Generator

Spec author: Fable (2026-07-15). Implements the "Benchmark generation on
demand" plan item + its 3-step pipeline shape + the family roadmap
(`docs/inv_family_roadmap.md`). Goal (user, 2026-07-15): implement the
benchmark test generator "in full specificity, for all relevant ciphers
(including some we have not yet tested against INV or the solver);
generate a plaintext corpus; implement the test generation code."

## HARD SCOPE BOUNDARY (Codex is concurrently editing the agentic system)

DO NOT EDIT: `src/agent/**`, `src/investigation/**`, `src/agent/model_provider.py`,
the v3 loop/episodes/context/actions/reading files, `tools_v2.py`, or their
tests. If a change seems to need one of those, STOP and flag it — do not
touch it. `src/benchmark/loader.py` and `scorer.py` are READ-ONLY references
(match their formats; do not modify). `src/testgen/plaintext.py`'s LLM path
stays as-is (we ADD a corpus path, not replace the module).

Everything this spec adds is NEW files under: `src/ciphers/`,
`src/testgen/` (new submodules), `resources/plaintext_library/`,
`scripts/`, `tests/`.

## Three slices (A and B are independent → parallel; C consumes both)

---

## Slice A — Cipher family primitives (`src/ciphers/`)

Each family is a `Cipher` subclass (src/ciphers/base.py ABC: encrypt/decrypt/
key_space_size over `list[int]` token ids + `Alphabet`) OR, where token-id
modeling is awkward (fractionation, encodings), a self-contained
encrypt/decrypt over uppercase strings with a documented interface. Each
MUST round-trip (decrypt(encrypt(pt)) == pt) — pinned by a test per family.

Implement these families (standard textbook/ACA definitions; cite the
definition in a docstring). Group into new modules:

- `ciphers/polyalphabetic.py`: Vigenère, Beaufort, Variant Beaufort,
  Gronsfeld, Porta, **Autokey** (key-autokey + text-autokey), **Running
  key** (key = another corpus passage). (builder.py already has vigenere/
  beaufort/variant/gronsfeld/quagmire3 logic inline — REUSE by extracting
  shared tableau helpers here; do not break builder.py's existing callers.)
- `ciphers/playfair.py`: **Playfair**, **Two-square**, **Four-square**
  (5x5 keyed squares, I/J merge; standard digraph rules).
- `ciphers/fractionation.py`: **Bifid**, **Trifid** (Polybius +
  fractionation), **ADFGX**, **ADFGVX** (Polybius→fractionate→columnar
  transposition with a keyword).
- `ciphers/transposition.py`: **Columnar** (complete + incomplete),
  **Railfence/Redefence**, **Route** (spiral/diagonal), **Amsco**,
  **Myszkowski**, **Cadenus**, **Nihilist transposition**. (Reuse builder's
  existing pure-transposition where equivalent.)
- `ciphers/numeric.py`: **Nihilist substitution** (Polybius + keyed
  addition → number pairs), **Straddling checkerboard**, **Hill (2x2)**
  (mod-26 matrix; auto-pad; invertible-key generation).
- `ciphers/encodings.py`: DETECTION-tier targets (not cryptanalysis, but
  INV must triage them): **Baconian**, **A1Z26**, **Morse**, **Base64**,
  **Base32**, **Hex**, **Binary**, **ROT47**, **Tap code**. encrypt only +
  a decode for the round-trip test.

Each family also provides a `random_key(rng, alphabet)` and a
`describe_key(key)` (human string for the ground-truth record). Determinism:
every key/choice derives from an injected `random.Random(seed)` — NO global
random, NO `Math.random`/`Date.now`-equivalent.

Tests (`tests/test_cipher_families.py`): per family — round-trip on a known
plaintext; a known-answer vector where a canonical one exists (e.g. Playfair
"HIDETHEGOLD…" standard example, Bifid textbook example); key determinism
(same seed → same key); key_space_size sanity.

---

## Slice B — Measured plaintext corpus (`src/testgen/corpus_library.py` +
`scripts/build_plaintext_library.py` + `resources/plaintext_library/`)

Build a SHIPPED, MEASURED plaintext library sampled from the local corpora
(`corpus_data/{en,de,fr,it,la}`, `corpora/dta` for historical German) — real
prose (Gutenberg/BNC/MASC/OANC/DTA), NOT the LLM. This is step 1 of the
pipeline (difficulty selection on MEASURED properties, per the plan).

**Builder** (`scripts/build_plaintext_library.py`, offline, no LLM):
- Sample passages of graded lengths (≈40, 120, 250, 500 words) from the
  corpora; clean to uppercase A-Z (+ language-appropriate; strip markup/
  Gutenberg headers/licenses — reuse any existing corpus cleaning; if none,
  implement conservative stripping and TEST it doesn't leak "PROJECT
  GUTENBERG").
- MEASURE each passage (reuse `analysis/frequency.py`, `analysis/ic.py`,
  `analysis/dictionary.py`): index_of_coincidence, unigram_chi2 vs language,
  dict_rate, mean_word_length, function_word_rate, bigram/quadgram
  log-likelihood (via `analysis/ngram.py` if cheap), unique-symbol ratio.
- TAG each: `language`, `era` (derive from provenance: dta→
  historical_1600_1899, la→classical, gutenberg→literary_19c, bnc/oanc/masc→
  modern), `provenance` (source corpus + file), `frequency_style`
  (`normal`|`unusual` by unigram_chi2 percentile within language),
  `topic` (best-effort from source/title; "general" fallback).
- WRITE `resources/plaintext_library/<lang>.jsonl` — one passage/line:
  `{id, language, era, provenance, source_file, text, length_words,
  length_chars, measured:{...}, frequency_style, topic, content_hash}`.
  Ground truth is the text itself (this is the plaintext source, so no
  firewall issue here — the firewall is at CASE assembly, Slice C). Cap the
  shipped library to a sane size (target ≤ ~40MB total; sample, don't ship
  6GB). Record the sampling manifest + per-language counts.

**Library API** (`src/testgen/corpus_library.py`): `load_library(lang)`,
`select(language=, era=, min_words=, max_words=, frequency_style=,
topic=, rng=)` → a passage record (deterministic given rng+filters), and a
`LibraryEmpty`-style error when filters match nothing (name the unmet
filter). NO network, NO LLM.

Tests (`tests/test_corpus_library.py`): a tiny FIXTURE library (checked into
tests/fixtures, a handful of hand-written passages with precomputed stats)
drives selection/filter/determinism/empty-filter tests — do NOT depend on
the multi-GB corpora in CI. One slow/opt-in test (marked, env-gated) may
exercise the real builder over a small corpus sample.

---

## Slice C — Generator orchestration (`src/testgen/family_registry.py` +
`scripts/generate_cipher_benchmark.py`) — consumes A + B

**Family registry** (`src/testgen/family_registry.py`): one entry per
generatable family binding {name, the Slice-A cipher, canonical output form
(letters vs S-tokens vs numeric — matching what BenchmarkLoader/scorer
expect; follow builder.py's `_format_canonical`/`_format_canonical_tokens`),
key-space sampler, applicable languages, difficulty knobs}. Cross-reference
`docs/inv_family_roadmap.md`; where INV's `families.py` already names a
family, use the SAME family id string (read-only cross-check — do not edit
families.py).

**Generator** (`scripts/generate_cipher_benchmark.py`), the 3-step pipeline:
1. plaintext selection via Slice-B `select()` (language/era/frequency/
   length/topic knobs);
2. key generation + encipherment via the Slice-A family;
3. batch: for each requested family × language × difficulty, produce N
   examples (deterministic seeds), AND emit graded CONTEXT TIERS per case:
   `none` → `language` → `era_provenance` → `rich` (era+provenance+topic+
   cipher-family family hint). Context tiers go in a per-case metadata field
   / context records — NEVER the plaintext or key.

**Output = loader-compatible benchmark tree** (match `BenchmarkLoader`
exactly, READ its format from loader.py; do not change loader):
- `<out>/manifest/records.jsonl` (BenchmarkRecord fields: id, source,
  cipher_type, plaintext_language, transcription_canonical_file,
  plaintext_file, has_key, + raw extras: era/provenance/measured/
  context_tier, key_description).
- `<out>/data/**` plaintext + canonical transcription files.
- `<out>/splits/*.jsonl` (BenchmarkTest: test_id, track, cipher_system,
  target_records, context_records, description) — one split per family +
  an `all_generated.jsonl`; context-tier variants as separate tracks
  (`transcription2plaintext` for none, context tracks for the richer tiers),
  mirroring the existing benchmark's track structure.
- A `ground_truth/` area (keys + plaintext) FIREWALLED from context —
  reuse `benchmark/unsolved.py` conventions if applicable; at minimum the
  context that a solver/model sees per tier must never contain the plaintext
  or the key. Pin with a firewall test (a `rich`-tier context record does
  NOT contain plaintext/key substrings).

**CLI**: `--families all|<list>`, `--languages`, `--per-family N`,
`--difficulty`, `--context-tiers`, `--out <dir>`, `--seed`, `--dry-run`
(print the plan/counts, no files). Deterministic; offline; no LLM.

Tests (`tests/test_benchmark_generator.py`): generate a small suite into a
tmp dir; assert BenchmarkLoader can load it and iterate every case;
round-trip a couple of generated cases (decrypt with the stored key ==
plaintext); the firewall test; context-tier presence/gradation;
determinism (same seed → byte-identical manifest); dry-run emits no files.

## Acceptance (local, no LLM)

1. Full suite green (baseline: current HEAD; +the new tests). Report count.
2. Generate a real multi-family suite (all Slice-A families × en, a few de/
   fr/it/la, per-family 3) into a tmp/artifacts dir; show BenchmarkLoader
   loads all cases and a table of {family, language, n, canonical form,
   sample IoC}. Round-trip-verify ≥1 case per family.
3. Confirm every generated `cipher_system` string is either an existing
   INV `families.py` id or documented as a NEW family (list them) — this is
   the "families not yet tested against INV/solver" deliverable.

## Deliverables

Files added, suite counts, the generated-suite table + loader-load proof,
round-trip confirmations per family, the firewall-test result, the list of
NEW families now generatable (vs INV/solver coverage), deviations. No
commits (orchestrator lands each slice after review).
