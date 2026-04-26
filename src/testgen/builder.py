from __future__ import annotations

import random
import string
import uuid

from benchmark.loader import BenchmarkTest, TestData
from analysis.transformers import TransformPipeline, make_inverse_input_for_pipeline
from testgen.cache import PlaintextCache
from testgen.plaintext import PlaintextGenerator
from testgen.spec import TestSpec


def build_test_case(
    spec: TestSpec,
    cache: PlaintextCache,
    api_key: str,
    seed: int | None = None,
    generator: PlaintextGenerator | None = None,
    generator_provider: str = "anthropic",
    generator_model: str | None = None,
) -> TestData:
    """Generate (or load from cache) a plaintext, encrypt it, return TestData.

    The cipher key is fresh each call unless seed is provided. The same
    plaintext can be re-encrypted many times with different keys.

    Args:
        spec:      Describes the desired plaintext (language, length, etc.)
        cache:     Disk-backed plaintext cache.
        api_key:   Provider API key (used only on cache miss).
        seed:      Optional integer seed for the cipher key RNG. Overrides spec.seed.
        generator: Inject a custom PlaintextGenerator (useful for tests/mocks).
    """
    # --- plaintext ---
    plaintext_str = cache.get(spec)
    if plaintext_str is None:
        gen = generator or PlaintextGenerator(
            api_key,
            provider=generator_provider,
            model=generator_model,
        )
        plaintext_str = gen.generate(spec)
        cache.put(spec, plaintext_str)

    words = plaintext_str.split()
    if not words:
        raise ValueError("Generated/cached plaintext is empty after normalisation")

    # --- cipher key and encryption ---
    effective_seed = seed if seed is not None else spec.seed
    rng = random.Random(effective_seed)

    transform_pipeline = TransformPipeline.from_raw(spec.transform_pipeline)

    if spec.homophonic:
        homo_key = _make_homophonic_key(rng)
        cipher_token_words = _apply_homophonic_key(words, homo_key, rng)
        if transform_pipeline is not None and not transform_pipeline.is_empty():
            flat_tokens = [token for word in cipher_token_words for token in word]
            scrambled = make_inverse_input_for_pipeline(flat_tokens, transform_pipeline)
            canonical = " ".join(scrambled)
            cipher_system = "transposition_homophonic"
        else:
            canonical = _format_canonical_tokens(cipher_token_words, spec.word_boundaries)
            cipher_system = "homophonic_substitution"
    else:
        key = _make_key(rng)
        cipher_words = _apply_key(words, key)
        if transform_pipeline is not None and not transform_pipeline.is_empty():
            flat_chars = [ch for word in cipher_words for ch in word]
            scrambled_chars = make_inverse_input_for_pipeline(flat_chars, transform_pipeline)
            canonical = " ".join(scrambled_chars)
            cipher_system = "transposition_substitution"
        else:
            canonical = _format_canonical(cipher_words, spec.word_boundaries)
            cipher_system = "simple_substitution"

    # --- plaintext for scorer ---
    # Word-boundary format: "THE QUICK FOX"  (scorer splits on spaces)
    # No-boundary format:   "THEQUICKFOX"    (scorer compares char-by-char)
    if spec.word_boundaries and (transform_pipeline is None or transform_pipeline.is_empty()):
        plaintext_for_score = " ".join(words)
    else:
        plaintext_for_score = "".join(words)

    # --- assemble TestData ---
    # Seeded runs get a stable, human-readable ID so the same test can be
    # tracked across multiple runs (regression testing).
    if effective_seed is not None:
        if spec.homophonic:
            cipher_flag = "thonb" if transform_pipeline is not None and not transform_pipeline.is_empty() else "honb"
        else:
            if transform_pipeline is not None and not transform_pipeline.is_empty():
                cipher_flag = "tnb"
            else:
                cipher_flag = "wb" if spec.word_boundaries else "nb"
        test_id = f"synth_{spec.language}_{spec.approx_length}{cipher_flag}_s{effective_seed}"
    else:
        test_id = f"synth_{spec.language}_{uuid.uuid4().hex[:8]}"
    boundary_label = (
        "transformed no-boundary"
        if transform_pipeline is not None and not transform_pipeline.is_empty()
        else ("word-boundary" if spec.word_boundaries else "no-boundary")
    )
    homo_label = ", homophonic" if spec.homophonic else ""
    transform_label = ", transformed" if transform_pipeline is not None and not transform_pipeline.is_empty() else ""
    test = BenchmarkTest(
        test_id=test_id,
        track="transcription2plaintext",
        cipher_system=cipher_system,
        target_records=[],
        context_records=[],
        description=(
            f"Synthetic {spec.language}, {boundary_label}{homo_label}{transform_label}, "
            f"~{len(words)} words, topic={spec.topic}"
        ),
    )
    return TestData(
        test=test,
        canonical_transcription=canonical,
        plaintext=plaintext_for_score,
        symbol_map=None,
        transform_pipeline=transform_pipeline.to_raw() if transform_pipeline is not None else None,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _make_key(rng: random.Random) -> dict[str, str]:
    """Random A-Z → A-Z substitution key."""
    shuffled = list(string.ascii_uppercase)
    rng.shuffle(shuffled)
    return dict(zip(string.ascii_uppercase, shuffled))


def _apply_key(words: list[str], key: dict[str, str]) -> list[str]:
    return ["".join(key[c] for c in word) for word in words]


def _format_canonical(cipher_words: list[str], word_boundaries: bool) -> str:
    if word_boundaries:
        return " | ".join(" ".join(w) for w in cipher_words)
    else:
        return " ".join(c for w in cipher_words for c in w)


# ---------------------------------------------------------------------------
# Homophonic cipher helpers
# ---------------------------------------------------------------------------

# Approximate English letter frequency groups — determines homophone count.
# HIGH letters (most common) get 4 symbols each, MED get 2, LOW get 1.
# Total symbols: 8×4 + 8×2 + 10×1 = 58
_HIGH_FREQ = set("ETAOINSH")   # 8 letters × 4 homophones = 32
_MED_FREQ  = set("RDLCUMWF")   # 8 letters × 2 homophones = 16
# remaining 10 letters × 1 homophone = 10  →  total 58


def _make_homophonic_key(rng: random.Random) -> dict[str, list[str]]:
    """Random homophonic key: each A-Z letter → list of 2-digit token strings.

    High-frequency letters receive more homophones (harder to crack by
    frequency analysis).  Tokens are zero-padded decimal: "01"–"58".
    """
    counts = {}
    for letter in string.ascii_uppercase:
        if letter in _HIGH_FREQ:
            counts[letter] = 4
        elif letter in _MED_FREQ:
            counts[letter] = 2
        else:
            counts[letter] = 1

    total = sum(counts.values())          # 58
    symbols = [f"{i:02d}" for i in range(1, total + 1)]
    rng.shuffle(symbols)

    key: dict[str, list[str]] = {}
    idx = 0
    for letter in string.ascii_uppercase:
        n = counts[letter]
        key[letter] = symbols[idx : idx + n]
        idx += n
    return key


def _apply_homophonic_key(
    words: list[str],
    key: dict[str, list[str]],
    rng: random.Random,
) -> list[list[str]]:
    """Encrypt word list using homophonic key; randomly select a homophone per letter."""
    return [
        [rng.choice(key[c]) for c in word]
        for word in words
    ]


def _format_canonical_tokens(
    cipher_token_words: list[list[str]],
    word_boundaries: bool,
) -> str:
    """Format a list-of-lists of multi-char tokens into a canonical transcription."""
    if word_boundaries:
        return " | ".join(" ".join(w) for w in cipher_token_words)
    else:
        return " ".join(t for w in cipher_token_words for t in w)
