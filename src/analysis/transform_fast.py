"""Rust fast kernels for ciphertext transform screening."""
from __future__ import annotations

import os
from typing import Any

from analysis import ngram
from analysis.polyalphabetic_fast import require_fast


def score_transform_candidates_fast_batch(
    tokens: list[int],
    candidates: list[dict[str, Any]],
    *,
    threads: int = 0,
) -> list[dict[str, Any]]:
    """Score transform candidates with the Rust structural-metrics kernel.

    Python remains responsible for candidate generation, provenance, artifact
    shape, and deduplication. Rust owns the expensive position-permutation
    application and position-only structural metrics used by large streaming
    screens.
    """

    fast = require_fast()
    return list(
        fast.transform_structural_metrics_batch(
            [int(token) for token in tokens],
            candidates,
            int(max(0, threads)),
        )
    )


def search_k3_transmatrix_fast(
    symbol_values: list[int],
    *,
    language: str = "en",
    min_width: int = 2,
    max_width: int = 80,
    top_n: int = 10,
    threads: int = 0,
) -> dict[str, Any]:
    """Run the Rust K3-style TransMatrix width-pair search.

    This is a Rust-owned search path. Python supplies A-Z values and the
    language model, then preserves the returned candidates/provenance for
    orchestration and artifacts.
    """

    fast = require_fast()
    quad = ngram.NGRAM_CACHE.get(language, 4)
    return dict(
        fast.k3_transmatrix_search(
            [int(value) for value in symbol_values],
            quad,
            int(max(2, min_width)),
            int(max(2, max_width)),
            int(max(1, top_n)),
            int(max(0, threads)),
        )
    )


def score_pure_transposition_candidates_fast_batch(
    symbol_values: list[int],
    candidates: list[dict[str, Any]],
    *,
    language: str = "en",
    top_n: int = 25,
    threads: int = 0,
) -> dict[str, Any]:
    """Score pure-transposition candidates by transformed plaintext quality.

    Candidate generation/provenance stays in Python; Rust applies each
    transform pipeline and scores the resulting A-Z text directly.
    """

    fast = require_fast()
    quad = ngram.NGRAM_CACHE.get(language, 4)
    return dict(
        fast.pure_transposition_score_batch(
            [int(value) for value in symbol_values],
            candidates,
            quad,
            int(max(1, top_n)),
            int(max(0, threads)),
        )
    )


def transform_fast_threads_from_env() -> int:
    """Return requested transform-screen worker count.

    ``0`` means auto-size to all available Rayon worker threads.
    """

    raw = os.environ.get("DECIPHER_TRANSFORM_SCREEN_THREADS", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0
