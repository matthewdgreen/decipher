"""Rust fast kernels for periodic/keyed polyalphabetic search.

The CLI requires `decipher_fast` for normal runs so broad-search behavior is
not silently replaced by slower Python diagnostics. Some Python reference
implementations remain in `analysis.polyalphabetic` for tests and small
research probes, but they are not runtime fallbacks for Rust-scale searches.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from analysis import ngram
from analysis.polyalphabetic import (
    AZ_ALPHABET,
    cipher_values_from_text,
    generate_keyed_vigenere_constraint_alphabets,
    generate_keyed_vigenere_constraint_graph_alphabets,
    generate_keyed_vigenere_offset_graph_alphabets,
    normalize_keyed_alphabet,
)
from models.cipher_text import CipherText

try:
    import decipher_fast as _fast
except Exception as _fast_exc:  # noqa: BLE001
    _fast = None
    _FAST_IMPORT_ERROR = _fast_exc
else:
    _FAST_IMPORT_ERROR = None


FAST_AVAILABLE = _fast is not None
FAST_BUILD_COMMAND = (
    "cd rust/decipher_fast && ../../.venv/bin/python -m pip install maturin "
    "&& ../../.venv/bin/python -m maturin develop --release"
)


class OptionalFastKernelUnavailable(RuntimeError):
    """Raised when the required Rust fast kernel is unavailable."""


def fast_kernel_status() -> dict[str, Any]:
    """Return a small, user-facing status report for Rust kernels."""
    root = Path(__file__).resolve().parents[2]
    rust_dir = root / "rust" / "decipher_fast"
    return {
        "available": FAST_AVAILABLE,
        "module": "decipher_fast",
        "module_file": str(getattr(_fast, "__file__", "")) if _fast is not None else None,
        "import_error": f"{type(_FAST_IMPORT_ERROR).__name__}: {_FAST_IMPORT_ERROR}"
        if _FAST_IMPORT_ERROR is not None
        else None,
        "rust_source_dir": str(rust_dir),
        "build_command": FAST_BUILD_COMMAND,
        "features": [
            "normalized_ngram_score",
            "keyed_vigenere_alphabet_anneal",
            "quagmire3_shotgun_search",
            "transform_structural_metrics_batch",
            "zenith_solve_seed",
            "zenith_transform_candidates_batch",
        ],
        "note": (
            "Required runtime accelerator. Build this module before normal "
            "Decipher CLI runs; Python search paths are reference/diagnostic "
            "only and are not large-scale fallbacks."
        ),
    }


def fast_kernel_unavailable_message(*, feature: str = "the Rust fast kernel") -> str:
    status = fast_kernel_status()
    detail = f" Import error: {status['import_error']}." if status.get("import_error") else ""
    return (
        f"{feature} requires the Python module `decipher_fast`, but "
        f"it is not installed for this environment.{detail}\n\n"
        "Build the Rust fast kernels from the repo root with:\n"
        "  scripts/build_rust_fast.sh\n\n"
        "Manual equivalent:\n"
        f"  {FAST_BUILD_COMMAND}\n\n"
        "Then verify with:\n"
        "  PYTHONPATH=src .venv/bin/decipher doctor\n"
    )


def require_fast() -> Any:
    if _fast is None:
        raise OptionalFastKernelUnavailable(fast_kernel_unavailable_message())
    return _fast


def estimate_quagmire3_shotgun_budget(
    *,
    keyword_lengths: list[int] | None = None,
    cycleword_lengths: list[int] | None = None,
    hillclimbs: int = 500,
    restarts: int = 1000,
    threads: int = 0,
) -> dict[str, Any]:
    """Estimate Rust Quagmire III shotgun search size and rough wall time.

    This is intentionally conservative and user-facing. The actual rate depends
    on CPU, text length, compiler options, and thermal state; callers should
    treat the estimate as a sizing aid, not a benchmark claim.
    """
    keyword_lengths = keyword_lengths or [7]
    cycleword_lengths = cycleword_lengths or [8]
    restart_jobs = max(1, len(keyword_lengths)) * max(1, len(cycleword_lengths)) * max(1, int(restarts))
    nominal_proposals = restart_jobs * max(1, int(hillclimbs))
    available_threads = os.cpu_count() or 1
    effective_threads = available_threads if int(threads) <= 0 else max(1, int(threads))
    effective_threads = max(1, effective_threads)
    conservative_rate_per_thread = 12_000
    typical_rate_per_thread = 20_000
    conservative_seconds = nominal_proposals / max(1, conservative_rate_per_thread * effective_threads)
    typical_seconds = nominal_proposals / max(1, typical_rate_per_thread * effective_threads)
    return {
        "engine": "rust_shotgun",
        "keyword_lengths": keyword_lengths,
        "cycleword_lengths": cycleword_lengths,
        "restart_jobs": restart_jobs,
        "hillclimbs_per_restart": max(0, int(hillclimbs)),
        "nominal_proposals": nominal_proposals,
        "requested_threads": int(threads),
        "estimated_threads": effective_threads,
        "estimated_seconds_typical": round(typical_seconds, 1),
        "estimated_seconds_conservative": round(conservative_seconds, 1),
        "estimated_minutes_typical": round(typical_seconds / 60.0, 2),
        "estimated_minutes_conservative": round(conservative_seconds / 60.0, 2),
        "rate_assumption": (
            "Rough sizing only: assumes about 20k proposals/sec/thread typical "
            "and 12k proposals/sec/thread conservative for the current Rust "
            "kernel on K2-sized text."
        ),
        "sizing_guidance": [
            "diagnostic: <= 1,000,000 proposals",
            "moderate: about 5,000,000 to 50,000,000 proposals",
            "broad: >= 50,000,000 proposals; confirm user is willing to spend minutes of CPU time",
        ],
    }


def normalized_ngram_score_fast(text: str, log_probs: dict[str, float], n: int = 4) -> float:
    """Rust implementation of analysis.ngram.normalized_ngram_score."""
    fast = require_fast()
    return float(fast.normalized_ngram_score(text, log_probs, n))


def search_keyed_vigenere_alphabet_anneal_fast(
    cipher_text: CipherText,
    *,
    language: str = "en",
    max_period: int = 20,
    initial_alphabets: list[str] | None = None,
    include_standard_alphabet: bool = True,
    steps: int = 2000,
    restarts: int = 4,
    seed: int = 1,
    top_n: int = 5,
    guided: bool = True,
    guided_pool_size: int = 24,
    constraint_starts: int = 0,
    constraint_beam_size: int = 64,
    constraint_random_starts: int = 0,
    constraint_random_seed: int = 1,
    constraint_top_shifts: int = 3,
    constraint_top_letters: int = 3,
    constraint_target_window: int = 3,
    constraint_random_phases: int | None = None,
    offset_graph_starts: int = 0,
    offset_graph_seed: int = 1,
    offset_graph_samples: int = 4096,
    offset_graph_phase_count: int | None = None,
    offset_graph_top_cipher: int = 5,
    offset_graph_target_letters: int = 8,
    offset_graph_target_window: int = 4,
    constraint_graph_starts: int = 0,
    constraint_graph_seed: int = 1,
    constraint_graph_beam_size: int = 128,
    constraint_graph_phase_count: int | None = None,
    constraint_graph_top_cipher: int = 5,
    constraint_graph_target_letters: int = 8,
    constraint_graph_target_window: int = 4,
    constraint_graph_options_per_phase: int = 96,
    constraint_graph_materializations: int = 4,
) -> dict[str, Any]:
    """Run the Rust keyed-tableau annealer.

    This returns the same high-level shape as
    `search_keyed_vigenere_alphabet_anneal`, but currently accepts explicit
    alphabets only. Callers should derive keyword alphabets in Python first.
    """
    fast = require_fast()
    symbol_values, skipped = cipher_values_from_text(cipher_text)
    starts: list[str] = []
    if include_standard_alphabet:
        starts.append(AZ_ALPHABET)
    for alphabet in initial_alphabets or []:
        normalized = normalize_keyed_alphabet(keyed_alphabet=alphabet)
        if normalized not in starts:
            starts.append(normalized)
    constraint_result = None
    if constraint_starts > 0:
        constraint_result = generate_keyed_vigenere_constraint_alphabets(
            cipher_text,
            max_period=max_period,
            beam_size=constraint_beam_size,
            limit=constraint_starts,
            random_limit=constraint_random_starts,
            random_seed=constraint_random_seed,
            top_shifts=constraint_top_shifts,
            top_letters=constraint_top_letters,
            target_window=constraint_target_window,
            random_phases=constraint_random_phases,
        )
        for row in constraint_result.get("alphabets") or []:
            alphabet = row.get("keyed_alphabet")
            if isinstance(alphabet, str) and alphabet not in starts:
                starts.append(alphabet)
    offset_graph_result = None
    if offset_graph_starts > 0:
        offset_graph_result = generate_keyed_vigenere_offset_graph_alphabets(
            cipher_text,
            max_period=max_period,
            limit=offset_graph_starts,
            random_seed=offset_graph_seed,
            samples=offset_graph_samples,
            phase_count=offset_graph_phase_count,
            top_cipher_letters=offset_graph_top_cipher,
            target_letters=offset_graph_target_letters,
            target_window=offset_graph_target_window,
        )
        for row in offset_graph_result.get("alphabets") or []:
            alphabet = row.get("keyed_alphabet")
            if isinstance(alphabet, str) and alphabet not in starts:
                starts.append(alphabet)
    constraint_graph_result = None
    if constraint_graph_starts > 0:
        constraint_graph_result = generate_keyed_vigenere_constraint_graph_alphabets(
            cipher_text,
            max_period=max_period,
            limit=constraint_graph_starts,
            random_seed=constraint_graph_seed,
            beam_size=constraint_graph_beam_size,
            phase_count=constraint_graph_phase_count,
            top_cipher_letters=constraint_graph_top_cipher,
            target_letters=constraint_graph_target_letters,
            target_window=constraint_graph_target_window,
            options_per_phase=constraint_graph_options_per_phase,
            materializations_per_state=constraint_graph_materializations,
        )
        for row in constraint_graph_result.get("alphabets") or []:
            alphabet = row.get("keyed_alphabet")
            if isinstance(alphabet, str) and alphabet not in starts:
                starts.append(alphabet)

    quad = ngram.NGRAM_CACHE.get(language, 4)
    top_candidates = list(fast.keyed_vigenere_alphabet_anneal(
        symbol_values,
        quad,
        int(max_period),
        starts,
        int(max(0, steps)),
        int(max(1, restarts)),
        int(seed),
        int(max(1, top_n)),
        bool(guided),
        int(max(1, guided_pool_size)),
    ))
    best = top_candidates[0] if top_candidates else None
    period_values = list(range(1, min(int(max_period), max(1, len(symbol_values) // 4)) + 1))
    return {
        "status": "completed" if best else "no_candidates",
        "solver": "keyed_vigenere_alphabet_anneal_rust",
        "language": language,
        "token_count": len(symbol_values),
        "skipped_symbols": skipped[:20],
        "skipped_symbol_count": len(skipped),
        "periods_tested": period_values,
        "initial_alphabets_tested": [
            {
                "keyed_alphabet": alphabet,
                "alphabet_keyword": None,
                "candidate_type": (
                    "standard_alphabet" if alphabet == AZ_ALPHABET else "explicit_alphabet"
                ),
            }
            for alphabet in starts
        ],
        "steps_per_period": max(0, steps),
        "restarts_per_alphabet": max(1, restarts),
        "guided": bool(guided),
        "guided_pool_size": max(1, guided_pool_size),
        "constraint_starts": max(0, constraint_starts),
        "constraint_random_starts": max(0, constraint_random_starts),
        "constraint_generator": constraint_result,
        "offset_graph_starts": max(0, offset_graph_starts),
        "offset_graph_generator": offset_graph_result,
        "constraint_graph_starts": max(0, constraint_graph_starts),
        "constraint_graph_generator": constraint_graph_result,
        "top_candidates": top_candidates,
        "best_candidate": best,
        "scope_note": (
            "Rust optional fast kernel. Intended to match Python wordlist "
            "quadgram scoring semantics while accelerating keyed-tableau "
            "candidate search."
        ),
    }


def search_quagmire3_shotgun_fast(
    cipher_text: CipherText,
    *,
    language: str = "en",
    keyword_lengths: list[int] | None = None,
    cycleword_lengths: list[int] | None = None,
    hillclimbs: int = 500,
    restarts: int = 1000,
    seed: int = 1,
    top_n: int = 10,
    slip_probability: float = 0.001,
    backtrack_probability: float = 0.15,
    threads: int = 0,
    initial_keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Run the Rust Blake-style Quagmire III shotgun search.

    This is the compiled-kernel path intended for broad blind/keyed-tableau
    search. It mutates keyword-shaped shared alphabets, derives the best
    cycleword for each proposal, scores no-boundary A-Z quadgrams, and
    parallelizes independent restart jobs across local CPU cores.
    """
    fast = require_fast()
    symbol_values, skipped = cipher_values_from_text(cipher_text)
    quad = ngram.NGRAM_CACHE.get(language, 4)
    keyword_lengths = keyword_lengths or [7]
    cycleword_lengths = cycleword_lengths or [8]
    result = dict(fast.quagmire3_shotgun_search(
        symbol_values,
        quad,
        [int(v) for v in keyword_lengths],
        [int(v) for v in cycleword_lengths],
        int(max(0, hillclimbs)),
        int(max(1, restarts)),
        int(seed),
        int(max(1, top_n)),
        float(max(0.0, slip_probability)),
        float(max(0.0, backtrack_probability)),
        int(max(0, threads)),
        [str(v) for v in (initial_keywords or [])],
    ))
    candidates = list(result.get("top_candidates") or [])
    best = candidates[0] if candidates else None
    result.update({
        "language": language,
        "token_count": len(symbol_values),
        "skipped_symbols": skipped[:20],
        "skipped_symbol_count": len(skipped),
        "top_candidates": candidates,
        "best_candidate": best,
        "scope_note": (
            "Rust optional fast kernel for Blake-style Quagmire III search. "
            "Python orchestrates benchmark/context/artifact handling; Rust "
            "owns the parallel restart/hillclimb loop."
        ),
    })
    return result
