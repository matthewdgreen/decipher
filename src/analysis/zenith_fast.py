"""Rust-backed Zenith-native homophonic solver wrapper.

Python remains the reference implementation in :mod:`analysis.zenith_solver`.
This module adapts the Rust fast kernel to the same ``HomophonicAnnealResult``
shape so automated and agentic orchestration can compare engines without
changing artifact logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis.homophonic import HomophonicAnnealResult, HomophonicCandidate
from analysis.polyalphabetic_fast import require_fast


def zenith_solve_fast(
    *,
    tokens: list[int],
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    model_path: str | Path,
    initial_key: dict[int, int] | None = None,
    fixed_cipher_ids: set[int] | None = None,
    epochs: int = 10,
    sampler_iterations: int = 5000,
    t_start: float = 0.012,
    t_end: float = 0.006,
    seed: int | None = None,
    top_n: int = 1,
) -> HomophonicAnnealResult:
    """Run one Zenith-native seed through the Rust fast kernel."""

    fast = require_fast()
    payload: dict[str, Any] = fast.zenith_solve_seed(
        str(Path(model_path).expanduser().resolve()),
        [int(token) for token in tokens],
        [int(pid) for pid in plaintext_ids],
        {int(k): str(v) for k, v in id_to_letter.items()},
        ({int(k): int(v) for k, v in initial_key.items()} if initial_key else {}),
        ([int(sid) for sid in sorted(fixed_cipher_ids)] if fixed_cipher_ids else []),
        int(max(1, epochs)),
        int(max(1, sampler_iterations)),
        float(t_start),
        float(t_end),
        int(seed or 0),
        int(max(1, top_n)),
    )
    candidates = [
        HomophonicCandidate(
            plaintext=str(item["plaintext"]),
            key={int(k): int(v) for k, v in dict(item["key"]).items()},
            score=float(item["score"]),
            normalized_score=float(item["normalized_score"]),
            epoch=int(item["epoch"]),
        )
        for item in payload.get("candidates", [])
    ]
    metadata = dict(payload.get("metadata") or {})
    # Keep the existing high-level solver family visible while making the
    # implementation engine explicit for artifacts and comparisons.
    metadata["solver"] = "zenith_native"
    metadata["engine"] = "rust"
    return HomophonicAnnealResult(
        plaintext=str(payload["plaintext"]),
        key={int(k): int(v) for k, v in dict(payload["key"]).items()},
        score=float(payload["score"]),
        normalized_score=float(payload["normalized_score"]),
        epochs=int(payload["epochs"]),
        sampler_iterations=int(payload["sampler_iterations"]),
        accepted_moves=int(payload["accepted_moves"]),
        improved_moves=int(payload["improved_moves"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        fixed_symbols=int(payload.get("fixed_symbols", 0)),
        metadata=metadata,
        candidates=candidates,
    )


def zenith_transform_candidates_batch_fast(
    *,
    tokens: list[int],
    candidates: list[dict[str, Any]],
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    model_path: str | Path,
    epochs: int,
    sampler_iterations: int,
    seeds: list[int],
    t_start: float = 0.012,
    t_end: float = 0.006,
    top_n: int = 1,
    threads: int = 0,
) -> dict[str, Any]:
    """Evaluate transform candidates with Rust pipeline + Zenith seed probes."""

    fast = require_fast()
    return dict(
        fast.zenith_transform_candidates_batch(
            str(Path(model_path).expanduser().resolve()),
            [int(token) for token in tokens],
            candidates,
            [int(pid) for pid in plaintext_ids],
            {int(k): str(v) for k, v in id_to_letter.items()},
            int(max(1, epochs)),
            int(max(1, sampler_iterations)),
            float(t_start),
            float(t_end),
            [int(seed) for seed in seeds],
            int(max(1, top_n)),
            int(max(0, threads)),
        )
    )
