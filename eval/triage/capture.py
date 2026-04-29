"""Candidate capture: run transform search and dump the full ranked list.

For each PopulationEntry, this module calls screen_transform_candidates with
a large top_n (default 200) and persists the full scored list as a
CapturedCase JSON file.  This is the expensive-but-cacheable step.

Usage
-----
    from triage.capture import capture_case, load_or_capture

    captured = capture_case(entry, top_n=200, profile="small")
    captured.save(Path("eval/artifacts/candidates/my_case.json"))
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# src/ imports
from analysis.transform_search import screen_transform_candidates
from benchmark.loader import parse_canonical_transcription

from triage.types import CandidateRecord, CapturedCase, PopulationEntry

# Default capture parameters.
DEFAULT_TOP_N = 200
DEFAULT_PROFILE = "small"


def capture_case(
    entry: PopulationEntry,
    *,
    top_n: int = DEFAULT_TOP_N,
    profile: str = DEFAULT_PROFILE,
    include_mutations: bool = False,
    include_program_search: bool = False,
) -> CapturedCase:
    """Run transform search on one PopulationEntry and return a CapturedCase.

    No LLM calls, no downstream solver — structural scoring only.
    """
    cipher_text = parse_canonical_transcription(entry.canonical)
    tokens = cipher_text.tokens

    result = screen_transform_candidates(
        tokens,
        columns=entry.columns,
        profile=profile,
        top_n=top_n,
        include_mutations=include_mutations,
        include_program_search=include_program_search,
    )

    raw_candidates: list[dict[str, Any]] = result.get("top_candidates", [])
    total_scored: int = result.get("deduped_candidate_count", len(raw_candidates))

    candidates = []
    for rank, raw in enumerate(raw_candidates):
        c = CandidateRecord(
            candidate_id=f"{entry.case_id}::{rank}",
            original_rank=rank,
            family=raw.get("family", ""),
            params=raw.get("params", {}),
            pipeline=raw.get("pipeline"),
            score=raw.get("score", 0.0),
            delta_vs_identity=raw.get("delta_vs_identity", 0.0),
            metrics=raw.get("metrics", {}),
            token_order_hash=raw.get("token_order_hash", ""),
        )
        candidates.append(c)

    return CapturedCase(
        case_id=entry.case_id,
        population_entry=entry.to_dict(),
        capture_config={
            "top_n": top_n,
            "profile": profile,
            "include_mutations": include_mutations,
            "include_program_search": include_program_search,
            "token_count": len(tokens),
        },
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        total_scored=total_scored,
        candidates=candidates,
    )


def candidate_path(case_id: str, artifact_dir: Path) -> Path:
    """Canonical path for a case's candidate JSONL file."""
    return artifact_dir / "candidates" / f"{case_id}.json"


def load_or_capture(
    entry: PopulationEntry,
    artifact_dir: Path,
    *,
    top_n: int = DEFAULT_TOP_N,
    profile: str = DEFAULT_PROFILE,
    force_recapture: bool = False,
) -> CapturedCase:
    """Return a cached CapturedCase, or run capture if not yet cached."""
    path = candidate_path(entry.case_id, artifact_dir)
    if not force_recapture and path.exists():
        return CapturedCase.load(path)
    captured = capture_case(entry, top_n=top_n, profile=profile)
    captured.save(path)
    return captured
