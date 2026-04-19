from __future__ import annotations

import random
import string
import uuid

from benchmark.loader import BenchmarkTest, TestData
from testgen.cache import PlaintextCache
from testgen.plaintext import PlaintextGenerator
from testgen.spec import TestSpec


def build_test_case(
    spec: TestSpec,
    cache: PlaintextCache,
    api_key: str,
    seed: int | None = None,
    generator: PlaintextGenerator | None = None,
) -> TestData:
    """Generate (or load from cache) a plaintext, encrypt it, return TestData.

    The cipher key is fresh each call unless seed is provided. The same
    plaintext can be re-encrypted many times with different keys.

    Args:
        spec:      Describes the desired plaintext (language, length, etc.)
        cache:     Disk-backed plaintext cache.
        api_key:   Anthropic API key (used only on cache miss).
        seed:      Optional integer seed for the cipher key RNG. Overrides spec.seed.
        generator: Inject a custom PlaintextGenerator (useful for tests/mocks).
    """
    # --- plaintext ---
    plaintext_str = cache.get(spec)
    if plaintext_str is None:
        gen = generator or PlaintextGenerator(api_key)
        plaintext_str = gen.generate(spec)
        cache.put(spec, plaintext_str)

    words = plaintext_str.split()
    if not words:
        raise ValueError("Generated/cached plaintext is empty after normalisation")

    # --- cipher key ---
    effective_seed = seed if seed is not None else spec.seed
    key = _make_key(effective_seed)
    cipher_words = _apply_key(words, key)

    # --- canonical transcription ---
    canonical = _format_canonical(cipher_words, spec.word_boundaries)

    # --- plaintext for scorer ---
    # Word-boundary format: "THE QUICK FOX"  (scorer splits on spaces)
    # No-boundary format:   "THEQUICKFOX"    (scorer compares char-by-char)
    if spec.word_boundaries:
        plaintext_for_score = " ".join(words)
    else:
        plaintext_for_score = "".join(words)

    # --- assemble TestData ---
    # Seeded runs get a stable, human-readable ID so the same test can be
    # tracked across multiple runs (regression testing).
    if effective_seed is not None:
        wb_flag = "wb" if spec.word_boundaries else "nb"
        test_id = f"synth_{spec.language}_{spec.approx_length}{wb_flag}_s{effective_seed}"
    else:
        test_id = f"synth_{spec.language}_{uuid.uuid4().hex[:8]}"
    boundary_label = "word-boundary" if spec.word_boundaries else "no-boundary"
    test = BenchmarkTest(
        test_id=test_id,
        track="transcription2plaintext",
        cipher_system="simple_substitution",
        target_records=[],
        context_records=[],
        description=(
            f"Synthetic {spec.language}, {boundary_label}, "
            f"~{len(words)} words, topic={spec.topic}"
        ),
    )
    return TestData(
        test=test,
        canonical_transcription=canonical,
        plaintext=plaintext_for_score,
        symbol_map=None,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _make_key(seed: int | None) -> dict[str, str]:
    """Random A-Z → A-Z substitution key."""
    rng = random.Random(seed)
    shuffled = list(string.ascii_uppercase)
    rng.shuffle(shuffled)
    return dict(zip(string.ascii_uppercase, shuffled))


def _apply_key(words: list[str], key: dict[str, str]) -> list[str]:
    return ["".join(key[c] for c in word) for word in words]


def _format_canonical(cipher_words: list[str], word_boundaries: bool) -> str:
    if word_boundaries:
        # "A B C | D E F | G H I"
        return " | ".join(" ".join(w) for w in cipher_words)
    else:
        # "A B C D E F G H I"
        return " ".join(c for w in cipher_words for c in w)
