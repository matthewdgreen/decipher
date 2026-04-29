"""Core dataclasses for the transform triage evaluation framework."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PopulationEntry:
    """One synthetic test case in the evaluation population."""

    case_id: str
    language: str
    transform_family: str          # "t_substitution" | "t_homophonic"
    columns: int | None            # grid width used to build the cipher
    rows: int | None               # grid height (may be None — derived at build time)
    approx_length: int             # target word count
    seed: int                      # RNG seed for the cipher key
    homophonic: bool
    canonical: str                 # space-separated cipher tokens (no | boundaries)
    plaintext: str                 # ground-truth plaintext (no spaces for nb)
    cipher_system: str             # "transposition_substitution" | "transposition_homophonic"
    ground_truth_pipeline: dict[str, Any] | None   # raw pipeline dict used to build cipher
    ground_truth_token_order_hash: str | None       # hash of gt pipeline on range(n)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "language": self.language,
            "transform_family": self.transform_family,
            "columns": self.columns,
            "rows": self.rows,
            "approx_length": self.approx_length,
            "seed": self.seed,
            "homophonic": self.homophonic,
            "canonical": self.canonical,
            "plaintext": self.plaintext,
            "cipher_system": self.cipher_system,
            "ground_truth_pipeline": self.ground_truth_pipeline,
            "ground_truth_token_order_hash": self.ground_truth_token_order_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PopulationEntry":
        return cls(
            case_id=d["case_id"],
            language=d["language"],
            transform_family=d["transform_family"],
            columns=d.get("columns"),
            rows=d.get("rows"),
            approx_length=d["approx_length"],
            seed=d["seed"],
            homophonic=d.get("homophonic", False),
            canonical=d["canonical"],
            plaintext=d["plaintext"],
            cipher_system=d["cipher_system"],
            ground_truth_pipeline=d.get("ground_truth_pipeline"),
            ground_truth_token_order_hash=d.get("ground_truth_token_order_hash"),
        )


@dataclass
class CandidateRecord:
    """One scored candidate from a transform search, possibly with ground-truth labels."""

    candidate_id: str
    original_rank: int             # rank in the raw screen_transform_candidates output
    family: str
    params: dict[str, Any]
    pipeline: dict[str, Any] | None
    score: float
    delta_vs_identity: float
    metrics: dict[str, float]
    token_order_hash: str

    # Labels — None means not yet computed
    transform_correct: bool | None = None
    readable_now: float | None = None        # char-accuracy of solver output vs plaintext
    rescuable: bool | None = None            # readable_now >= rescuable_threshold
    rescuable_char_accuracy: float | None = None
    decoded_text: str | None = None          # best decryption from run_automated

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "original_rank": self.original_rank,
            "family": self.family,
            "params": self.params,
            "pipeline": self.pipeline,
            "score": self.score,
            "delta_vs_identity": self.delta_vs_identity,
            "metrics": self.metrics,
            "token_order_hash": self.token_order_hash,
            "transform_correct": self.transform_correct,
            "readable_now": self.readable_now,
            "rescuable": self.rescuable,
            "rescuable_char_accuracy": self.rescuable_char_accuracy,
            "decoded_text": self.decoded_text,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CandidateRecord":
        return cls(
            candidate_id=d["candidate_id"],
            original_rank=d["original_rank"],
            family=d["family"],
            params=d.get("params", {}),
            pipeline=d.get("pipeline"),
            score=d["score"],
            delta_vs_identity=d["delta_vs_identity"],
            metrics=d.get("metrics", {}),
            token_order_hash=d["token_order_hash"],
            transform_correct=d.get("transform_correct"),
            readable_now=d.get("readable_now"),
            rescuable=d.get("rescuable"),
            rescuable_char_accuracy=d.get("rescuable_char_accuracy"),
            decoded_text=d.get("decoded_text"),
        )


@dataclass
class CapturedCase:
    """Full candidate list for one test case, with optional labels."""

    case_id: str
    population_entry: dict[str, Any]        # serialized PopulationEntry
    capture_config: dict[str, Any]          # parameters used for capture
    captured_at: str                        # ISO timestamp
    total_scored: int                       # total candidates scored by transform search
    candidates: list[CandidateRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "population_entry": self.population_entry,
            "capture_config": self.capture_config,
            "captured_at": self.captured_at,
            "total_scored": self.total_scored,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CapturedCase":
        return cls(
            case_id=d["case_id"],
            population_entry=d["population_entry"],
            capture_config=d.get("capture_config", {}),
            captured_at=d.get("captured_at", ""),
            total_scored=d.get("total_scored", len(d.get("candidates", []))),
            candidates=[CandidateRecord.from_dict(c) for c in d.get("candidates", [])],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "CapturedCase":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def entry(self) -> PopulationEntry:
        return PopulationEntry.from_dict(self.population_entry)
