"""Rust fast kernels for ciphertext transform screening."""
from __future__ import annotations

import os
from typing import Any

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
