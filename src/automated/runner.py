"""Automated-only solving without LLM API calls.

This module deliberately stays separate from ``agent.loop_v2``. It uses the
same native solver building blocks exposed to the agent tools, but runs them
deterministically from local code and writes a small dashboard-compatible
artifact marked ``run_mode: automated_only``.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import random
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from analysis import cipher_id as cipher_id_analysis
from analysis import dictionary, homophonic, ic, ngram, pattern, polyalphabetic
from analysis.segment import (
    repair_key_with_dictionary,
    repair_no_boundary_text,
    segment_text,
)
from analysis.solver import simulated_anneal
from analysis.transform_evaluation import (
    FinalistMenuEvaluationPlan,
    FinalistMenuValidationPolicy,
    evaluate_finalist_menu,
    validate_finalist_menu,
)
from analysis.transformers import TransformPipeline, apply_transform_pipeline
from analysis.transform_search import inspect_transform_suspicion, screen_transform_candidates
from benchmark.loader import TestData, parse_canonical_transcription, resolve_test_language
from benchmark.scorer import score_decryption
from models.alphabet import Alphabet
from models.cipher_text import CipherText
from models.session import Session


@dataclass
class AutomatedRunResult:
    test_id: str
    status: str
    final_decryption: str
    elapsed_seconds: float
    char_accuracy: float = 0.0
    word_accuracy: float = 0.0
    self_confidence: float | None = None
    iterations_used: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    artifact_path: str = ""
    error_message: str = ""
    solver: str = ""
    run_id: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    artifact: dict[str, Any] = field(default_factory=dict)


class AutomatedBenchmarkRunner:
    """Runs the automated-only pipeline on benchmark ``TestData``."""

    def __init__(
        self,
        artifact_dir: str | Path = "artifacts",
        language: str | None = None,
        verbose: bool = False,
        homophonic_budget: str = "full",
        homophonic_refinement: str = "none",
        homophonic_solver: str = "zenith_native",
        transform_search: str = "off",
        transform_search_profile: str = "broad",
        transform_search_max_generated_candidates: int | None = None,
        transform_promote_artifact: str | None = None,
        transform_promote_candidate_ids: list[str] | None = None,
        transform_promote_top_n: int | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.default_language = language
        self.verbose = verbose
        self.homophonic_budget = homophonic_budget
        self.homophonic_refinement = homophonic_refinement
        self.homophonic_solver = homophonic_solver
        self.transform_search = transform_search
        self.transform_search_profile = transform_search_profile
        self.transform_search_max_generated_candidates = transform_search_max_generated_candidates
        self.transform_promote_artifact = transform_promote_artifact
        self.transform_promote_candidate_ids = transform_promote_candidate_ids or []
        self.transform_promote_top_n = transform_promote_top_n

    def _resolve_language(self, test_data: TestData) -> str:
        return resolve_test_language(test_data, self.default_language)

    def run_test(self, test_data: TestData, language: str | None = None) -> AutomatedRunResult:
        lang = language or self._resolve_language(test_data)
        test_id = test_data.test.test_id
        start = time.time()
        try:
            cipher_text = parse_canonical_transcription(test_data.canonical_transcription)
        except Exception as exc:  # noqa: BLE001
            return AutomatedRunResult(
                test_id=test_id,
                status="error",
                final_decryption="",
                elapsed_seconds=time.time() - start,
                error_message=f"Failed to parse transcription: {exc}",
                solver="automated_only",
            )

        run_kwargs = {
            "cipher_text": cipher_text,
            "language": lang,
            "cipher_id": test_id,
            "ground_truth": test_data.plaintext,
            "cipher_system": test_data.test.cipher_system,
            "homophonic_budget": self.homophonic_budget,
            "homophonic_refinement": self.homophonic_refinement,
            "homophonic_solver": self.homophonic_solver,
        }
        if test_data.solver_hints:
            run_kwargs["solver_hints"] = test_data.solver_hints
        if self.transform_search != "off":
            run_kwargs["transform_search"] = self.transform_search
            run_kwargs["transform_search_profile"] = self.transform_search_profile
            run_kwargs["transform_search_max_generated_candidates"] = self.transform_search_max_generated_candidates
            run_kwargs["transform_promote_artifact"] = self.transform_promote_artifact
            run_kwargs["transform_promote_candidate_ids"] = self.transform_promote_candidate_ids
            run_kwargs["transform_promote_top_n"] = self.transform_promote_top_n
        if test_data.transform_pipeline:
            run_kwargs["transform_pipeline"] = test_data.transform_pipeline
        result = run_automated(**run_kwargs)
        artifact = dict(result.artifact)
        artifact["description"] = test_data.test.description
        artifact["cipher_system"] = test_data.test.cipher_system

        artifact_path = self.artifact_dir / "automated_only" / test_id / f"{result.run_id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

        result.artifact_path = str(artifact_path)
        result.elapsed_seconds = time.time() - start
        return result


@dataclass
class _AutomatedInternalResult:
    run_id: str
    cipher_id: str
    language: str
    status: str
    solver: str
    decryption: str
    key: dict[int, int]
    steps: list[dict[str, Any]]
    started_at: float
    finished_at: float
    cipher_alphabet_size: int
    cipher_token_count: int
    cipher_word_count: int
    ground_truth: str | None = None
    char_accuracy: float = 0.0
    word_accuracy: float = 0.0
    error_message: str = ""
    transform_pipeline: dict[str, Any] | None = None
    input_transform_pipeline: dict[str, Any] | None = None
    transform_selection: dict[str, Any] | None = None
    original_cipher_token_count: int | None = None
    transform_search: dict[str, Any] | None = None
    cipher_id_report: dict[str, Any] | None = None

    @property
    def elapsed_seconds(self) -> float:
        return self.finished_at - self.started_at

    def to_result(self) -> AutomatedRunResult:
        result = AutomatedRunResult(
            test_id=self.cipher_id,
            status=self.status,
            final_decryption=self.decryption,
            elapsed_seconds=self.elapsed_seconds,
            char_accuracy=self.char_accuracy,
            word_accuracy=self.word_accuracy,
            iterations_used=len(self.steps),
            total_tokens=0,
            estimated_cost_usd=0.0,
            error_message=self.error_message,
            solver=self.solver,
            run_id=self.run_id,
            steps=list(self.steps),
            artifact=self.to_artifact(),
        )
        return result

    def to_artifact(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_mode": "automated_only",
            "automated_only": True,
            "cipher_id": self.cipher_id,
            "test_id": self.cipher_id,
            "language": self.language,
            "status": self.status,
            "solver": self.solver,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "cipher_alphabet_size": self.cipher_alphabet_size,
            "cipher_token_count": self.cipher_token_count,
            "cipher_word_count": self.cipher_word_count,
            "transform_pipeline": self.transform_pipeline,
            "input_transform_pipeline": self.input_transform_pipeline,
            "transform_selection": self.transform_selection,
            "transform_search": self.transform_search,
            "cipher_id_report": self.cipher_id_report,
            "original_cipher_token_count": self.original_cipher_token_count,
            "decryption": self.decryption,
            "key": {str(k): v for k, v in self.key.items()},
            "steps": self.steps,
            "ground_truth": self.ground_truth,
            "char_accuracy": self.char_accuracy,
            "word_accuracy": self.word_accuracy,
            "error": self.error_message,
            "estimated_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }


def run_automated(
    cipher_text: CipherText,
    language: str = "en",
    cipher_id: str = "cli",
    ground_truth: str | None = None,
    cipher_system: str = "",
    solver_hints: dict[str, Any] | None = None,
    transform_pipeline: dict[str, Any] | None = None,
    homophonic_budget: str = "full",
    homophonic_refinement: str = "none",
    homophonic_solver: str = "zenith_native",
    transform_search: str = "off",
    transform_search_profile: str = "broad",
    transform_search_max_generated_candidates: int | None = None,
    transform_promote_artifact: str | None = None,
    transform_promote_candidate_ids: list[str] | None = None,
    transform_promote_top_n: int | None = None,
) -> AutomatedRunResult:
    """Run the best available local techniques without any LLM call."""
    started = time.time()
    run_id = uuid.uuid4().hex[:12]
    steps: list[dict[str, Any]] = []
    key: dict[int, int] = {}
    decryption = ""
    status = "error"
    solver = "automated_only"
    error = ""
    original_cipher_text = cipher_text
    parsed_transform = TransformPipeline.from_raw(transform_pipeline)
    transformed_step: dict[str, Any] | None = None
    transform_search_report: dict[str, Any] | None = None
    cipher_id_report: dict[str, Any] | None = None
    solver_hints = solver_hints or {}
    effective_transform_pipeline: dict[str, Any] | None = None
    transform_selection_report: dict[str, Any] | None = None
    if parsed_transform is not None and not parsed_transform.is_empty():
        transform_result = apply_transform_pipeline(cipher_text.tokens, parsed_transform)
        cipher_text = _cipher_text_from_tokens(
            transform_result.tokens,
            cipher_text.alphabet,
            source=f"{cipher_text.source}:transform",
        )
        effective_transform_pipeline = parsed_transform.to_raw()
        transform_selection_report = {
            "source": "input_transform_pipeline",
            "pipeline": effective_transform_pipeline,
            "original_token_count": len(original_cipher_text.tokens),
            "transformed_token_count": len(cipher_text.tokens),
        }
        transformed_step = {
            "name": "apply_cipher_transform",
            "pipeline": parsed_transform.to_raw(),
            "original_token_count": len(original_cipher_text.tokens),
            "transformed_token_count": len(cipher_text.tokens),
            "locked_positions": sum(1 for locked in transform_result.locked if locked),
        }

    if transform_search not in {"off", "auto", "screen", "wide", "rank", "full", "promote"}:
        raise ValueError("transform_search must be one of: off, auto, screen, wide, rank, full, promote")
    if transform_search == "promote" and not transform_promote_artifact:
        raise ValueError("--transform-search promote requires --transform-promote-artifact")
    transform_profile = _transform_search_profile_params(
        transform_search,
        transform_search_profile,
        max_generated_candidates=transform_search_max_generated_candidates,
    )
    if transform_search != "off" and transformed_step is None:
        suspicion = inspect_transform_suspicion(
            token_count=len(cipher_text.tokens),
            cipher_alphabet_size=cipher_text.alphabet.size,
            plaintext_alphabet_size=_plaintext_alphabet(language).size,
            word_group_count=len(cipher_text.words),
            cipher_system=cipher_system,
        )
        should_screen = transform_search in {"screen", "wide", "rank", "full"} or suspicion["recommendation"] in {
            "run_screen",
            "consider_screen",
        }
        if transform_search == "promote":
            screen = _promoted_transform_screen(
                transform_promote_artifact,
                candidate_ids=transform_promote_candidate_ids,
                top_n=transform_promote_top_n,
            )
        else:
            screen = (
                screen_transform_candidates(
                    cipher_text.tokens,
                    profile=transform_profile["screen_profile"],
                    top_n=transform_profile["top_n"],
                    max_generated_candidates=transform_profile["max_generated_candidates"],
                    streaming=transform_profile["streaming"],
                    include_mutations=transform_profile["include_mutations"],
                    mutation_seed_count=transform_profile["mutation_seed_count"],
                    include_program_search=transform_profile["include_program_search"],
                    program_max_depth=transform_profile["program_max_depth"],
                    program_beam_width=transform_profile["program_beam_width"],
                )
                if should_screen else None
            )
        rank = None
        rank_escalation = None
        if screen is not None and transform_search in {"rank", "full", "promote"}:
            rank_max_candidates = transform_profile["max_candidates"]
            if transform_search == "promote":
                promoted_candidate_count = len(screen.get("top_candidates") or []) + len(screen.get("anchor_candidates") or [])
                rank_max_candidates = max(rank_max_candidates, promoted_candidate_count + 1)
            rank = _rank_transform_candidates(
                cipher_text=cipher_text,
                language=language,
                screen=screen,
                budget="full" if transform_search == "full" else "screen",
                solver_profile=homophonic_solver,
                max_candidates=rank_max_candidates,
                confirm_count=transform_profile["confirm_count"],
                adaptive_confirmations=transform_profile["adaptive_confirmations"],
            )
            rank_escalation = None
            if _should_auto_escalate_transform_rank_to_full(
                transform_search=transform_search,
                homophonic_budget=homophonic_budget,
                homophonic_solver=homophonic_solver,
                rank=rank,
            ):
                rank_escalation = _transform_rank_escalation_summary(rank)
                rank = _rank_transform_candidates(
                    cipher_text=cipher_text,
                    language=language,
                    screen=screen,
                    budget="full",
                    solver_profile=homophonic_solver,
                    max_candidates=max(rank_max_candidates, 12),
                    confirm_count=transform_profile["confirm_count"],
                    adaptive_confirmations=transform_profile["adaptive_confirmations"],
                )
                rank_escalation.update({
                    "status": "escalated",
                    "escalated_budget": "full",
                    "escalated_selection": (rank or {}).get("selection")
                    if isinstance(rank, dict) else None,
                    "policy": (
                        "When Rust screen-budget transform ranking finds no "
                        "robust candidate but the user requested full "
                        "homophonic budget, Decipher automatically reruns the "
                        "same shortlist with full-budget Rust ranking instead "
                        "of returning only a diagnostic basin."
                    ),
                })
        transform_search_report = {
            "mode": transform_search,
            "profile": transform_search_profile,
            "profile_params": transform_profile,
            "suspicion": suspicion,
            "screen": screen,
            "rank": rank,
            "rank_escalation": rank_escalation,
            "status": "promoted" if transform_search == "promote" and screen else "screened" if screen else "not_screened",
            "note": (
                "Transform-search diagnostics and optional solver-backed "
                "candidate ranking. `screen` is diagnostic-only; `rank` and "
                "`full` may select a transformed candidate. `wide` is a "
                "larger structural-only search intended for later promotion. "
                "`promote` reuses candidates from an earlier structural artifact "
                "and spends solver probes only on that shortlist."
            ),
        }

    routing = _select_solver_path(
        cipher_text,
        language,
        cipher_system,
        has_transform_pipeline=transformed_step is not None,
    )
    fingerprint = cipher_id_analysis.compute_cipher_fingerprint(
        cipher_text.tokens,
        cipher_text.alphabet.size,
        language=language,
        word_group_count=len(cipher_text.words),
    )
    cipher_id_report = fingerprint.to_dict()

    try:
        if transformed_step is not None:
            steps.append(transformed_step)
        steps.append({
            "name": "route_automated_solver",
            "solver": routing["solver"],
            "route": routing["route"],
            "reason": routing["reason"],
            "cipher_system": cipher_system,
            "language": language,
            "cipher_id_report": cipher_id_report,
            "homophonic_budget": homophonic_budget,
            "homophonic_refinement": homophonic_refinement,
            "homophonic_solver": homophonic_solver,
            "transform_pipeline": parsed_transform.to_raw() if parsed_transform else None,
            "transform_search": transform_search_report,
        })
        if transform_search_report is not None:
            steps.append({
                "name": "screen_transform_candidates",
                **transform_search_report,
            })
        selected_transform_candidate = _selected_ranked_transform_candidate(transform_search_report)
        diagnostic_transform_candidate = _diagnostic_ranked_transform_candidate(transform_search_report)
        if selected_transform_candidate is not None:
            solver = "transform_search_homophonic"
            key = {
                int(k): int(v)
                for k, v in selected_transform_candidate.get("key", {}).items()
            }
            decryption = str(selected_transform_candidate.get("decryption") or "")
            effective_transform_pipeline = _effective_selected_transform_pipeline(selected_transform_candidate)
            transform_selection_report = _transform_selection_summary(
                selected_transform_candidate,
                source="transform_search_rank",
                promotion=(transform_search_report or {}).get("screen", {}).get("promotion")
                if isinstance((transform_search_report or {}).get("screen"), dict)
                else None,
                pipeline=effective_transform_pipeline,
            )
            steps.append({
                "name": "select_transform_candidate",
                "candidate_id": selected_transform_candidate.get("candidate_id"),
                "family": selected_transform_candidate.get("family"),
                "finalist_label": selected_transform_candidate.get("finalist_label"),
                "selects_transform": selected_transform_candidate.get("candidate_id") != "000_identity",
                "pipeline": selected_transform_candidate.get("pipeline"),
                "anneal_score": selected_transform_candidate.get("anneal_score"),
                "validated_selection_score": selected_transform_candidate.get("validated_selection_score"),
                "confirmed_selection_score": selected_transform_candidate.get("confirmed_selection_score"),
                "elapsed_seconds": selected_transform_candidate.get("elapsed_seconds"),
            })
            rank_report = (transform_search_report or {}).get("rank") or {}
            rank_budget = rank_report.get("budget") if isinstance(rank_report, dict) else None
            if homophonic_budget == "full" and rank_budget != "full":
                bakeoff = _refine_transform_finalist_bakeoff(
                    cipher_text=cipher_text,
                    language=language,
                    rank_report=rank_report,
                    selected_candidate=selected_transform_candidate,
                    budget=homophonic_budget,
                    refinement=homophonic_refinement,
                    solver_profile=homophonic_solver,
                    ground_truth=ground_truth,
                )
                winner = bakeoff.get("winner") or {}
                solver = "transform_search_homophonic_refined"
                key = {
                    int(k): int(v)
                    for k, v in (winner.get("key") or {}).items()
                }
                decryption = str(winner.get("decryption") or "")
                effective_transform_pipeline = _effective_selected_transform_pipeline(winner)
                transform_selection_report = _transform_selection_summary(
                    winner,
                    source="transform_search_full_refinement",
                    promotion=(transform_search_report or {}).get("screen", {}).get("promotion")
                    if isinstance((transform_search_report or {}).get("screen"), dict)
                    else None,
                    pipeline=effective_transform_pipeline,
                    screen_selected_candidate_id=selected_transform_candidate.get("candidate_id"),
                    selected_candidate_changed=bakeoff.get("selected_candidate_changed"),
                    refined_candidate_count=bakeoff.get("refined_candidate_count"),
                )
                steps.append({
                    "name": "refine_selected_transform_candidate_homophonic",
                    "candidate_id": winner.get("candidate_id"),
                    "family": winner.get("family"),
                    "screen_selected_candidate_id": selected_transform_candidate.get("candidate_id"),
                    "rank_budget": rank_budget,
                    "final_budget": homophonic_budget,
                    "pipeline": winner.get("pipeline"),
                    "locked_positions": winner.get("locked_positions"),
                    "homophonic_step": winner.get("homophonic_step"),
                    "solver": winner.get("solver"),
                    "bakeoff": bakeoff,
                })
        elif routing["route"] == "unsupported_mixed_transposition" and diagnostic_transform_candidate is not None:
            solver = "transform_search_no_robust_transform"
            key = {
                int(k): int(v)
                for k, v in diagnostic_transform_candidate.get("key", {}).items()
            }
            decryption = str(diagnostic_transform_candidate.get("decryption") or "")
            rank = (transform_search_report or {}).get("rank") or {}
            diagnostics = rank.get("diagnostics") if isinstance(rank, dict) else None
            steps.append({
                "name": "diagnostic_transform_search_no_robust_candidate",
                "candidate_id": diagnostic_transform_candidate.get("candidate_id"),
                "family": diagnostic_transform_candidate.get("family"),
                "finalist_label": diagnostic_transform_candidate.get("finalist_label"),
                "pipeline": diagnostic_transform_candidate.get("pipeline"),
                "anneal_score": diagnostic_transform_candidate.get("anneal_score"),
                "validated_selection_score": diagnostic_transform_candidate.get("validated_selection_score"),
                "confirmed_selection_score": diagnostic_transform_candidate.get("confirmed_selection_score"),
                "diagnostic_conclusion": (diagnostics or {}).get("conclusion") if isinstance(diagnostics, dict) else None,
                "note": (
                    "Transform search ran but no transform candidate passed "
                    "the confirmation/family evidence gates. This diagnostic "
                    "candidate is recorded to complete the run without "
                    "claiming transform recovery."
                ),
            })
        elif (
            routing["route"] == "unsupported_mixed_transposition"
            and transform_search_report is not None
            and transform_search_report.get("screen") is not None
            and transform_search_report.get("rank") is None
        ):
            solver = "transform_search_structural_only"
            decryption = ""
            steps.append({
                "name": "transform_search_structural_only",
                "mode": transform_search_report.get("mode"),
                "profile": transform_search_report.get("profile"),
                "candidate_count": (transform_search_report.get("screen") or {}).get("candidate_count"),
                "deduped_candidate_count": (transform_search_report.get("screen") or {}).get("deduped_candidate_count"),
                "top_candidate_count": len((transform_search_report.get("screen") or {}).get("top_candidates") or []),
                "note": (
                    "Structural transform search completed without running "
                    "homophonic solver probes. Promote a small finalist set "
                    "with rank/full before claiming a decipherment."
                ),
            })
        elif routing["route"] == "unsupported_mixed_transposition":
            raise ValueError(routing["reason"])
        elif routing["route"] == "homophonic":
            solver, key, decryption, step = _run_homophonic(
                cipher_text,
                language,
                budget=homophonic_budget,
                refinement=homophonic_refinement,
                solver_profile=homophonic_solver,
                ground_truth=ground_truth,
            )
            steps.append(step)
        elif routing["route"] == "pure_transposition":
            solver, key, decryption, step = _run_pure_transposition(
                cipher_text,
                language,
                cipher_system=cipher_system,
                solver_hints=solver_hints,
            )
            steps.append(step)
        elif routing["route"] == "periodic_polyalphabetic":
            solver, key, decryption, step = _run_periodic_polyalphabetic(
                cipher_text,
                language,
                cipher_system=cipher_system,
                solver_hints=solver_hints,
            )
            steps.append(step)
        else:
            solver, key, decryption, step = _run_substitution(cipher_text, language)
            steps.append(step)
        status = "completed" if decryption or solver == "transform_search_structural_only" else "error"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        status = "error"

    char_accuracy = 0.0
    word_accuracy = 0.0
    if ground_truth is not None and decryption:
        score = score_decryption(
            test_id=cipher_id,
            decrypted=decryption,
            ground_truth=ground_truth,
            agent_score=0.0,
            status=status,
        )
        char_accuracy = score.char_accuracy
        word_accuracy = score.word_accuracy

    internal = _AutomatedInternalResult(
        run_id=run_id,
        cipher_id=cipher_id,
        language=language,
        status=status,
        solver=solver,
        decryption=decryption,
        key=key,
        steps=steps,
        started_at=started,
        finished_at=time.time(),
        cipher_alphabet_size=cipher_text.alphabet.size,
        cipher_token_count=len(cipher_text.tokens),
        cipher_word_count=len(cipher_text.words),
        ground_truth=ground_truth,
        char_accuracy=char_accuracy,
        word_accuracy=word_accuracy,
        error_message=error,
        transform_pipeline=effective_transform_pipeline,
        input_transform_pipeline=parsed_transform.to_raw() if parsed_transform else None,
        transform_selection=transform_selection_report,
        original_cipher_token_count=len(original_cipher_text.tokens),
        transform_search=transform_search_report,
        cipher_id_report=cipher_id_report,
    )
    return internal.to_result()


def save_crack_artifact(
    result: AutomatedRunResult,
    cipher_text: CipherText,
    language: str,
    artifact_dir: str | Path,
) -> str:
    """Persist an automated-only crack result without benchmark ground truth."""
    path = Path(artifact_dir) / "automated_only" / result.test_id / f"{result.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(result.artifact)
    artifact.update({
        "cipher_id": result.test_id,
        "test_id": result.test_id,
        "language": language,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "cipher_alphabet_size": cipher_text.alphabet.size,
        "cipher_token_count": len(cipher_text.tokens),
        "cipher_word_count": len(cipher_text.words),
    })
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    result.artifact_path = str(path)
    return str(path)


def _selected_ranked_transform_candidate(
    transform_search_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not transform_search_report:
        return None
    rank = transform_search_report.get("rank")
    if not isinstance(rank, dict):
        return None
    selection = rank.get("selection")
    if isinstance(selection, dict):
        if not selection.get("selected"):
            return None
        selected_id = selection.get("selected_candidate_id")
        for candidate in rank.get("top_ranked_candidates") or []:
            if candidate.get("candidate_id") == selected_id:
                return candidate
        return None
    candidates = rank.get("top_ranked_candidates") or []
    if not candidates:
        return None
    best = candidates[0]
    if best.get("status") != "completed":
        return None
    if not best.get("decryption"):
        return None
    return best


def _rank_report_has_selected_candidate(rank: dict[str, Any] | None) -> bool:
    if not isinstance(rank, dict):
        return False
    selection = rank.get("selection")
    if isinstance(selection, dict):
        return bool(selection.get("selected") and selection.get("selected_candidate_id"))
    candidates = rank.get("top_ranked_candidates") or []
    return bool(candidates and candidates[0].get("status") == "completed")


def _should_auto_escalate_transform_rank_to_full(
    *,
    transform_search: str,
    homophonic_budget: str,
    homophonic_solver: str,
    rank: dict[str, Any] | None,
) -> bool:
    if transform_search != "rank":
        return False
    if homophonic_budget != "full" or homophonic_solver != "zenith_native":
        return False
    if not isinstance(rank, dict) or rank.get("budget") == "full":
        return False
    if _rank_report_has_selected_candidate(rank):
        return False
    if _zenith_native_engine() != "rust" or _transform_rank_engine() != "rust":
        return False
    return _env_bool("DECIPHER_TRANSFORM_AUTO_FULL_ESCALATION", True)


def _transform_rank_escalation_summary(rank: dict[str, Any] | None) -> dict[str, Any]:
    rank = rank or {}
    selection = rank.get("selection") if isinstance(rank.get("selection"), dict) else {}
    diagnostics = rank.get("diagnostics") if isinstance(rank.get("diagnostics"), dict) else {}
    top = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "family": candidate.get("family"),
            "finalist_label": candidate.get("finalist_label"),
            "confirmed_selection_score": candidate.get("confirmed_selection_score"),
            "anneal_score": candidate.get("anneal_score"),
        }
        for candidate in (rank.get("top_ranked_candidates") or [])[:8]
        if isinstance(candidate, dict)
    ]
    return {
        "trigger": "screen_rank_no_robust_candidate",
        "initial_budget": rank.get("budget"),
        "initial_selection": selection,
        "initial_diagnostic_conclusion": diagnostics.get("conclusion"),
        "initial_evaluated_candidates": rank.get("evaluated_candidates"),
        "initial_top_candidates": top,
    }


def _effective_selected_transform_pipeline(candidate: dict[str, Any]) -> dict[str, Any] | None:
    pipeline_raw = candidate.get("pipeline")
    pipeline = TransformPipeline.from_raw(pipeline_raw)
    if pipeline is None or pipeline.is_empty():
        return None
    return pipeline.to_raw()


def _transform_selection_summary(
    candidate: dict[str, Any],
    *,
    source: str,
    pipeline: dict[str, Any] | None,
    promotion: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    summary = {
        "source": source,
        "candidate_id": candidate.get("candidate_id"),
        "family": candidate.get("family"),
        "finalist_label": candidate.get("finalist_label"),
        "selects_transform": bool(pipeline),
        "pipeline": pipeline,
        "anneal_score": candidate.get("anneal_score"),
        "selection_score": candidate.get("selection_score"),
        "validated_selection_score": candidate.get("validated_selection_score"),
        "confirmed_selection_score": candidate.get("confirmed_selection_score"),
        "elapsed_seconds": candidate.get("elapsed_seconds"),
    }
    if promotion:
        summary["promotion"] = {
            "source_artifact": promotion.get("source_artifact"),
            "source_artifact_resolved": promotion.get("source_artifact_resolved"),
            "source_candidate_count": promotion.get("source_candidate_count"),
            "source_deduped_candidate_count": promotion.get("source_deduped_candidate_count"),
            "requested_candidate_ids": promotion.get("requested_candidate_ids"),
            "requested_top_n": promotion.get("requested_top_n"),
            "promoted_candidate_ids": promotion.get("promoted_candidate_ids"),
        }
    summary.update({key: value for key, value in extra.items() if value is not None})
    return summary


def _diagnostic_ranked_transform_candidate(
    transform_search_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not transform_search_report:
        return None
    rank = transform_search_report.get("rank")
    if not isinstance(rank, dict):
        return None
    candidates = rank.get("top_ranked_candidates") or []
    if not candidates:
        return None
    identity = next(
        (
            candidate for candidate in candidates
            if candidate.get("candidate_id") == "000_identity"
            and candidate.get("status") == "completed"
            and candidate.get("decryption")
        ),
        None,
    )
    if identity is not None:
        return identity
    return next(
        (
            candidate for candidate in candidates
            if candidate.get("status") == "completed"
            and candidate.get("decryption")
        ),
        None,
    )


def _refine_transform_finalist_bakeoff(
    *,
    cipher_text: CipherText,
    language: str,
    rank_report: dict[str, Any],
    selected_candidate: dict[str, Any],
    budget: str,
    refinement: str,
    solver_profile: str,
    ground_truth: str | None,
) -> dict[str, Any]:
    candidates = _full_refinement_finalists(rank_report, selected_candidate)
    refined: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        pipeline_raw = candidate.get("pipeline")
        started = time.time()
        try:
            pipeline = TransformPipeline.from_raw(pipeline_raw) or TransformPipeline()
            refined_cipher_text = cipher_text
            locked_positions = 0
            if not pipeline.is_empty():
                transform_result = apply_transform_pipeline(cipher_text.tokens, pipeline)
                refined_cipher_text = _cipher_text_from_tokens(
                    transform_result.tokens,
                    cipher_text.alphabet,
                    source=f"{cipher_text.source}:selected_transform_refine:{index}",
                )
                locked_positions = sum(1 for locked in transform_result.locked if locked)
            refined_solver, refined_key, refined_decryption, refined_step = _run_homophonic(
                refined_cipher_text,
                language,
                budget=budget,
                refinement=refinement,
                solver_profile=solver_profile,
                ground_truth=ground_truth,
            )
            anneal_score = _float_or_none(refined_step.get("anneal_score"))
            quality_score = _plaintext_quality_score(refined_decryption, language)
            structural_score = _float_or_none(candidate.get("structural_score"))
            if structural_score is None:
                structural_score = _float_or_none(candidate.get("score"))
            mutation_penalty = _transform_mutation_penalty(candidate)
            full_selection_score = _transform_selection_score(
                anneal_score=anneal_score,
                quality_score=quality_score,
                structural_score=structural_score,
                mutation_penalty=mutation_penalty,
            )
            refinement_selectable = _refinement_selectable_transform_candidate(candidate)
            refined.append({
                "candidate_id": candidate.get("candidate_id"),
                "family": candidate.get("family"),
                "finalist_label": candidate.get("finalist_label"),
                "selectable_transform_candidate": bool(candidate.get("selectable_transform_candidate")),
                "refinement_selectable": refinement_selectable,
                "pipeline": pipeline.to_raw(),
                "locked_positions": locked_positions,
                "solver": refined_solver,
                "homophonic_step": refined_step,
                "anneal_score": anneal_score,
                "plaintext_quality_score": round(quality_score, 6),
                "structural_score": structural_score,
                "full_selection_score": round(full_selection_score, 6),
                "screen_confirmed_selection_score": candidate.get("confirmed_selection_score"),
                "screen_validated_selection_score": candidate.get("validated_selection_score"),
                "elapsed_seconds": round(time.time() - started, 3),
                "decryption_preview": refined_decryption[:500],
                "decryption": refined_decryption,
                "key": {str(k): v for k, v in refined_key.items()},
            })
        except Exception as exc:  # noqa: BLE001
            skipped.append({
                "candidate_id": candidate.get("candidate_id"),
                "family": candidate.get("family"),
                "pipeline": pipeline_raw,
                "reason": f"{type(exc).__name__}: {exc}",
            })
    refined.sort(
        key=lambda item: (
            bool(item.get("refinement_selectable")),
            _float_or_none(item.get("full_selection_score")) or float("-inf"),
            _float_or_none(item.get("anneal_score")) or float("-inf"),
            _float_or_none(item.get("screen_confirmed_selection_score")) or float("-inf"),
        ),
        reverse=True,
    )
    winner = refined[0] if refined else {}
    return {
        "stage": "full_budget_transform_finalist_bakeoff",
        "winner": winner,
        "candidate_count": len(candidates),
        "refined_candidate_count": len(refined),
        "skipped_candidates": skipped,
        "refined_candidates": refined,
        "selected_candidate_changed": (
            bool(winner)
            and winner.get("candidate_id") != selected_candidate.get("candidate_id")
        ),
        "policy": (
            "When a screen-budget transform rank is followed by a full-budget "
            "run, refine the selected transform plus close/selectable "
            "finalists. The final pick preserves the ranker's robustness gates "
            "first, then compares full-budget selection scores, so unstable "
            "false positives can be reported but cannot replace a robust "
            "selected transform."
        ),
    }


def _refinement_selectable_transform_candidate(candidate: dict[str, Any]) -> bool:
    return (
        bool(candidate.get("selectable_transform_candidate"))
        or candidate.get("finalist_label") == "robust_candidate"
    )


def _full_refinement_finalists(
    rank_report: dict[str, Any],
    selected_candidate: dict[str, Any],
    *,
    limit: int = 3,
    score_margin: float = 0.06,
) -> list[dict[str, Any]]:
    ranked = [
        item for item in rank_report.get("top_ranked_candidates") or []
        if isinstance(item, dict)
        and item.get("status") == "completed"
        and item.get("pipeline")
        and item.get("candidate_id") != "000_identity"
    ]
    selected_id = str(selected_candidate.get("candidate_id"))
    selected_score = (
        _float_or_none(selected_candidate.get("confirmed_selection_score"))
        or _float_or_none(selected_candidate.get("validated_selection_score"))
        or _float_or_none(selected_candidate.get("selection_score"))
        or float("-inf")
    )
    finalists: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        if len(finalists) >= limit:
            return
        candidate_id = str(item.get("candidate_id"))
        if candidate_id in seen:
            return
        finalists.append(item)
        seen.add(candidate_id)

    for item in ranked:
        if str(item.get("candidate_id")) == selected_id:
            add(item)
            break
    if not finalists:
        add(selected_candidate)

    for item in ranked:
        if len(finalists) >= limit:
            break
        score = (
            _float_or_none(item.get("confirmed_selection_score"))
            or _float_or_none(item.get("validated_selection_score"))
            or _float_or_none(item.get("selection_score"))
            or float("-inf")
        )
        close = math.isfinite(selected_score) and score >= selected_score - score_margin
        selectable = _refinement_selectable_transform_candidate(item)
        if selectable or close:
            add(item)

    return finalists


def _promoted_transform_screen(
    artifact_path: str | None,
    *,
    candidate_ids: list[str] | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    if not artifact_path:
        raise ValueError("transform promotion requires a source artifact path")
    path = Path(artifact_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"transform promotion artifact not found: {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    transform_search = artifact.get("transform_search")
    if not isinstance(transform_search, dict):
        raise ValueError(f"artifact has no transform_search block: {path}")
    source_screen = transform_search.get("screen")
    if not isinstance(source_screen, dict):
        raise ValueError(f"artifact has no transform_search.screen block: {path}")

    identity = source_screen.get("identity_candidate")
    source_top = [
        candidate for candidate in source_screen.get("top_candidates") or []
        if isinstance(candidate, dict)
    ]
    source_anchor = [
        candidate for candidate in source_screen.get("anchor_candidates") or []
        if isinstance(candidate, dict)
    ]
    source_by_id: dict[str, dict[str, Any]] = {}
    for candidate in ([identity] if isinstance(identity, dict) else []) + source_top + source_anchor:
        source_by_id.setdefault(str(candidate.get("candidate_id")), candidate)

    requested_ids = [str(item) for item in (candidate_ids or []) if str(item).strip()]
    if requested_ids:
        missing = [candidate_id for candidate_id in requested_ids if candidate_id not in source_by_id]
        if missing:
            raise ValueError(
                "promoted transform candidate id(s) not found in source artifact: "
                + ", ".join(missing)
            )
        selected = [source_by_id[candidate_id] for candidate_id in requested_ids]
    else:
        limit = int(top_n) if top_n is not None else 10
        if limit < 1:
            raise ValueError("transform promotion top_n must be at least 1")
        selected = source_top[:limit]

    selected_by_id: dict[str, dict[str, Any]] = {}
    for candidate in selected:
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id != "000_identity":
            selected_by_id.setdefault(candidate_id, candidate)

    return {
        "profile": "promoted",
        "source_profile": source_screen.get("profile"),
        "candidate_count": len(selected_by_id),
        "deduped_candidate_count": len(selected_by_id),
        "top_candidates": list(selected_by_id.values()),
        "anchor_candidates": [],
        "identity_candidate": identity,
        "promotion": {
            "source_artifact": str(path),
            "source_artifact_resolved": str(path.resolve()),
            "source_mode": transform_search.get("mode"),
            "source_profile": transform_search.get("profile"),
            "source_status": transform_search.get("status"),
            "source_candidate_count": source_screen.get("candidate_count"),
            "source_deduped_candidate_count": source_screen.get("deduped_candidate_count"),
            "requested_candidate_ids": requested_ids,
            "requested_top_n": top_n,
            "promoted_candidate_ids": list(selected_by_id.keys()),
            "policy": (
                "Promotion reuses structural candidates from a prior wide/screen "
                "artifact, then runs bounded homophonic probes only on the "
                "selected shortlist plus identity as a control."
            ),
        },
    }


def _transform_search_profile_params(
    transform_search: str,
    profile: str,
    *,
    max_generated_candidates: int | None = None,
) -> dict[str, Any]:
    profile_key = (profile or "broad").strip().lower()
    if profile_key not in {"fast", "broad", "wide"}:
        raise ValueError("transform_search_profile must be one of: fast, broad, wide")
    is_rank = transform_search in {"rank", "full", "promote"}
    is_wide = transform_search == "wide" or profile_key == "wide"
    wide_limit = int(max_generated_candidates) if max_generated_candidates is not None else 25000
    broad_limit = int(max_generated_candidates) if max_generated_candidates is not None else 10000
    fast_limit = int(max_generated_candidates) if max_generated_candidates is not None else 5000
    if not is_rank:
        if is_wide:
            return {
                "profile": "wide",
                "screen_profile": "wide",
                "top_n": 500,
                "max_generated_candidates": wide_limit,
                "streaming": True,
                "include_mutations": False,
                "mutation_seed_count": 0,
                "include_program_search": True,
                "program_max_depth": 5,
                "program_beam_width": 48,
                "max_candidates": 0,
                "confirm_count": 0,
                "adaptive_confirmations": 0,
            }
        return {
            "profile": profile_key,
            "screen_profile": "small",
            "top_n": 8,
            "max_generated_candidates": fast_limit,
            "streaming": False,
            "include_mutations": False,
            "mutation_seed_count": 0,
            "include_program_search": False,
            "program_max_depth": 0,
            "program_beam_width": 0,
            "max_candidates": 0,
            "confirm_count": 0,
            "adaptive_confirmations": 0,
        }
    if profile_key == "fast":
        return {
            "profile": "fast",
            "screen_profile": "medium",
            "top_n": 60,
            "max_generated_candidates": fast_limit,
            "streaming": False,
            "include_mutations": False,
            "mutation_seed_count": 0,
            "include_program_search": False,
            "program_max_depth": 0,
            "program_beam_width": 0,
            "max_candidates": 8,
            "confirm_count": 3,
            "adaptive_confirmations": 0,
        }
    if profile_key == "wide":
        return {
            "profile": "wide",
            "screen_profile": "wide",
            "top_n": 500,
            "max_generated_candidates": wide_limit,
            "streaming": True,
            "include_mutations": False,
            "mutation_seed_count": 0,
            "include_program_search": True,
            "program_max_depth": 5,
            "program_beam_width": 48,
            "max_candidates": 24,
            "confirm_count": 4,
            "adaptive_confirmations": 2,
        }
    return {
        "profile": "broad",
        "screen_profile": "medium",
        "top_n": 120,
        "max_generated_candidates": broad_limit,
        "streaming": False,
        "include_mutations": True,
        "mutation_seed_count": 12 if transform_search == "full" else 8,
        "include_program_search": True,
        "program_max_depth": 5,
        "program_beam_width": 24,
        "max_candidates": 8 if transform_search == "full" else 10,
        "confirm_count": 3,
        "adaptive_confirmations": 2,
    }


def _rank_transform_candidates(
    *,
    cipher_text: CipherText,
    language: str,
    screen: dict[str, Any],
    budget: str,
    solver_profile: str,
    max_candidates: int,
    confirm_count: int = 3,
    adaptive_confirmations: int = 2,
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_pipeline: set[str] = set()
    raw_candidates, triage_report = _two_stage_transform_rank_candidates(screen, max_candidates=max_candidates)
    if (
        solver_profile == "zenith_native"
        and _zenith_native_engine() == "rust"
        and _transform_rank_engine() == "rust"
    ):
        ranked, skipped = _rank_transform_candidates_rust_batch(
            cipher_text=cipher_text,
            language=language,
            raw_candidates=raw_candidates,
            budget=budget,
        )
    else:
        for index, candidate in enumerate(raw_candidates):
            pipeline_raw = candidate.get("pipeline")
            pipeline_key = json.dumps(pipeline_raw, sort_keys=True)
            if pipeline_key in seen_pipeline:
                continue
            seen_pipeline.add(pipeline_key)
            try:
                pipeline = TransformPipeline.from_raw(pipeline_raw)
                if pipeline is None:
                    raise ValueError("missing transform pipeline")
                order = apply_transform_pipeline(list(range(len(cipher_text.tokens))), pipeline).tokens
                if sorted(order) != list(range(len(cipher_text.tokens))):
                    raise ValueError("transform candidate is not a position permutation")
                transform_result = apply_transform_pipeline(cipher_text.tokens, pipeline)
                transformed_cipher = _cipher_text_from_tokens(
                    transform_result.tokens,
                    cipher_text.alphabet,
                    source=f"{cipher_text.source}:transform_rank:{index}",
                )
                started = time.time()
                solver, key, decryption, step = _run_homophonic(
                    transformed_cipher,
                    language,
                    budget=budget,
                    refinement="none",
                    solver_profile=solver_profile,
                    ground_truth=None,
                )
                anneal_score = step.get("anneal_score")
                quality_score = _plaintext_quality_score(decryption, language)
                mutation_penalty = _transform_mutation_penalty(candidate)
                selection_score = _transform_selection_score(
                    anneal_score=anneal_score,
                    quality_score=quality_score,
                    structural_score=candidate.get("score"),
                    mutation_penalty=mutation_penalty,
                )
                ranked.append({
                    "candidate_id": candidate.get("candidate_id"),
                    "family": candidate.get("family"),
                    "provenance": candidate.get("provenance"),
                    "params": candidate.get("params"),
                    "pipeline": pipeline.to_raw(),
                    "status": "completed",
                    "solver": solver,
                    "anneal_score": anneal_score,
                    "plaintext_quality_score": round(quality_score, 6),
                    "local_mutation_penalty": mutation_penalty,
                    "selection_score": round(selection_score, 6),
                    "elapsed_seconds": round(time.time() - started, 3),
                    "decryption_preview": decryption[:500],
                    "decryption": decryption,
                    "key": {str(k): v for k, v in key.items()},
                    "structural_score": candidate.get("score"),
                    "structural_delta_vs_identity": candidate.get("delta_vs_identity"),
                    "matrix_rank_score": (candidate.get("metrics") or {}).get("matrix_rank_score"),
                    "best_period": (candidate.get("metrics") or {}).get("best_period"),
                    "inverse_best_period": (candidate.get("metrics") or {}).get("inverse_best_period"),
                })
            except Exception as exc:  # noqa: BLE001
                skipped.append({
                    "candidate_id": candidate.get("candidate_id"),
                    "family": candidate.get("family"),
                    "pipeline": pipeline_raw,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
    # Validation, confirmation, gating, and selection now flow through the
    # same finalist-menu skeleton used by pure-transposition screens. The
    # expensive probe engines remain path-specific.
    evaluation_report = evaluate_finalist_menu(
        ranked,
        plan=FinalistMenuEvaluationPlan(
            stage="transform_homophonic_finalist_menu_evaluation",
            pre_confirmation_score_field="validated_selection_score",
            pre_confirmation_secondary_fields=("selection_score", "anneal_score", "structural_score"),
            selection_policy=(
                "Two-stage rank: a broad structural screen is reduced to a "
                "family-diverse finalist set, solver probes produce candidate "
                "plaintexts, the shared finalist evaluator attaches plaintext "
                "validation evidence, then independent-seed confirmation and "
                "family gates decide whether a transform is selectable."
            ),
            note=(
                "Transform+homophonic finalist menu evaluated through the "
                "shared transform finalist skeleton; solver probes and "
                "confirmation batches remain Zenith/homophonic-specific."
            ),
        ),
        validate=lambda items: _validate_transform_finalists(items, language=language),
        confirm=lambda items: _confirm_transform_finalists(
            cipher_text=cipher_text,
            language=language,
            ranked=items,
            budget=budget,
            solver_profile=solver_profile,
            confirm_count=confirm_count,
            adaptive_confirmations=adaptive_confirmations,
        ),
        label=_label_transform_finalists,
        final_sort_key=_transform_final_sort_key,
        choose=_choose_transform_candidate,
        diagnose=_diagnose_transform_finalists,
    )
    ranked = list(evaluation_report.get("top_ranked_candidates") or ranked)
    return {
        "budget": budget,
        "max_candidates": max_candidates,
        "selection_policy": evaluation_report.get("selection_policy"),
        "triage": triage_report,
        "evaluated_candidates": evaluation_report.get("evaluated_candidates", len(ranked)),
        "skipped_candidates": skipped,
        "evaluation": {
            key: value
            for key, value in evaluation_report.items()
            if key != "top_ranked_candidates"
        },
        "validation": evaluation_report.get("validation"),
        "confirmation": evaluation_report.get("confirmation"),
        "finalists": evaluation_report.get("finalists"),
        "selection": evaluation_report.get("selection"),
        "diagnostics": evaluation_report.get("diagnostics"),
        "top_ranked_candidates": ranked,
        "note": (
            "Candidates are ranked by solver probes after the structural screen. "
            "This is bounded search, not exhaustive transform discovery."
        ),
    }


def _rank_transform_candidates_rust_batch(
    *,
    cipher_text: CipherText,
    language: str,
    raw_candidates: list[dict[str, Any]],
    budget: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate transform finalists with the Rust transform+Zenith batch kernel."""

    from analysis.zenith_fast import zenith_transform_candidates_batch_fast

    bin_path = _zenith_native_model_path(language)
    if bin_path is None:
        raise FileNotFoundError(
            f"zenith_native Rust transform ranking requires an ngram5 model for language={language!r}"
        )
    pt_alpha = _plaintext_alphabet(language)
    plaintext_ids = list(range(pt_alpha.size))
    id_to_letter = {i: pt_alpha.symbol_for(i).upper() for i in plaintext_ids}
    budget_params = _homophonic_budget_params(
        budget,
        len(cipher_text.tokens) < 600,
        search_profile=_homophonic_search_profile(),
    )
    seen_pipeline: set[str] = set()
    payload_candidates: list[dict[str, Any]] = []
    metadata_by_id: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        pipeline_raw = candidate.get("pipeline")
        pipeline_key = json.dumps(pipeline_raw, sort_keys=True)
        if pipeline_key in seen_pipeline:
            continue
        seen_pipeline.add(pipeline_key)
        candidate_id = str(candidate.get("candidate_id"))
        payload_candidates.append(candidate)
        metadata_by_id[candidate_id] = candidate

    threads = _transform_rank_threads()
    started = time.time()
    batch = zenith_transform_candidates_batch_fast(
        tokens=list(cipher_text.tokens),
        candidates=payload_candidates,
        plaintext_ids=plaintext_ids,
        id_to_letter=id_to_letter,
        model_path=bin_path,
        epochs=budget_params["epochs"],
        sampler_iterations=budget_params["sampler_iterations"],
        seeds=[int(seed) for seed in budget_params["seeds"]],
        top_n=3,
        threads=threads,
    )
    ranked: list[dict[str, Any]] = []
    for row in batch.get("results", []):
        candidate_id = str(row.get("candidate_id"))
        candidate = metadata_by_id.get(candidate_id, {})
        if row.get("status") != "completed":
            skipped.append({
                "candidate_id": candidate.get("candidate_id", candidate_id),
                "family": candidate.get("family") or row.get("family"),
                "pipeline": candidate.get("pipeline"),
                "reason": row.get("reason") or "rust_batch_candidate_failed",
            })
            continue
        decryption = str(row.get("decryption") or "")
        anneal_score = _float_or_none(row.get("normalized_score"))
        quality_score = _plaintext_quality_score(decryption, language)
        mutation_penalty = _transform_mutation_penalty(candidate)
        selection_score = _transform_selection_score(
            anneal_score=anneal_score,
            quality_score=quality_score,
            structural_score=candidate.get("score"),
            mutation_penalty=mutation_penalty,
        )
        ranked.append({
            "candidate_id": candidate.get("candidate_id", candidate_id),
            "family": candidate.get("family") or row.get("family"),
            "provenance": candidate.get("provenance"),
            "params": candidate.get("params"),
            "pipeline": candidate.get("pipeline"),
            "status": "completed",
            "solver": "zenith_native",
            "engine": "rust_batch",
            "anneal_score": anneal_score,
            "plaintext_quality_score": round(quality_score, 6),
            "local_mutation_penalty": mutation_penalty,
            "selection_score": round(selection_score, 6),
            "elapsed_seconds": round(float(row.get("elapsed_seconds") or 0.0), 3),
            "decryption_preview": decryption[:500],
            "decryption": decryption,
            "key": {str(k): int(v) for k, v in dict(row.get("key") or {}).items()},
            "structural_score": candidate.get("score"),
            "structural_delta_vs_identity": candidate.get("delta_vs_identity"),
            "matrix_rank_score": (candidate.get("metrics") or {}).get("matrix_rank_score"),
            "best_period": (candidate.get("metrics") or {}).get("best_period"),
            "inverse_best_period": (candidate.get("metrics") or {}).get("inverse_best_period"),
            "rust_batch": {
                "best_seed": row.get("best_seed"),
                "attempts": row.get("attempts") or [],
                "token_order_hash": row.get("token_order_hash"),
                "candidate_elapsed_seconds": row.get("elapsed_seconds"),
                "batch_elapsed_seconds": batch.get("elapsed_seconds"),
                "threads": batch.get("threads"),
                "seed_count": batch.get("seed_count"),
                "total_elapsed_seconds": round(time.time() - started, 3),
            },
        })
    return ranked, skipped


def _validate_transform_finalists(ranked: list[dict[str, Any]], *, language: str) -> dict[str, Any]:
    """Validate solver finalists against identity and mutation base candidates."""

    plaintext_report = validate_finalist_menu(
        ranked,
        policy=FinalistMenuValidationPolicy(
            language=language,
            plaintext_fields=("decryption", "decryption_preview"),
            base_score_field="selection_score",
            output_score_field="plaintext_validated_selection_score",
            adjustment_weight=0.03,
            score_precision=6,
        ),
    )
    by_id = {str(item.get("candidate_id")): item for item in ranked}
    identity = by_id.get("000_identity")
    identity_selection = _float_or_none(identity.get("selection_score")) if identity else None
    identity_anneal = _float_or_none(identity.get("anneal_score")) if identity else None
    mutation_penalized = 0
    identity_penalized = 0
    for item in ranked:
        selection = _float_or_none(item.get("selection_score")) or float("-inf")
        plaintext_adjusted_selection = (
            _float_or_none(item.get("plaintext_validated_selection_score"))
            if item.get("plaintext_validated_selection_score") is not None
            else selection
        )
        anneal = _float_or_none(item.get("anneal_score"))
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        base_id = params.get("base_candidate_id")
        base = by_id.get(str(base_id)) if base_id else None
        penalty = 0.0
        reasons: list[str] = []
        item["identity_selection_delta"] = (
            round(selection - identity_selection, 6)
            if identity_selection is not None else None
        )
        item["identity_anneal_delta"] = (
            round((anneal or float("-inf")) - identity_anneal, 6)
            if identity_anneal is not None and anneal is not None else None
        )
        if item.get("provenance") == "local_mutation":
            if base is None:
                penalty += 0.04
                reasons.append("mutation_base_not_evaluated")
            else:
                base_selection = _float_or_none(base.get("selection_score"))
                base_anneal = _float_or_none(base.get("anneal_score"))
                selection_delta = selection - base_selection if base_selection is not None else None
                anneal_delta = anneal - base_anneal if anneal is not None and base_anneal is not None else None
                item["base_candidate_id"] = base_id
                item["base_selection_delta"] = round(selection_delta, 6) if selection_delta is not None else None
                item["base_anneal_delta"] = round(anneal_delta, 6) if anneal_delta is not None else None
                if selection_delta is not None and selection_delta < 0.015:
                    penalty += 0.08
                    reasons.append("mutation_did_not_beat_base_selection")
                if anneal_delta is not None and anneal_delta < 0.0:
                    penalty += 0.04
                    reasons.append("mutation_worse_than_base_anneal")
            if penalty:
                mutation_penalized += 1
        if identity_selection is not None and item.get("candidate_id") != "000_identity":
            if selection < identity_selection - 0.02:
                penalty += 0.05
                reasons.append("below_identity_selection_margin")
                identity_penalized += 1
        item["validation_penalty"] = round(penalty, 6)
        item["validation_reasons"] = reasons
        item["validated_selection_score"] = round(plaintext_adjusted_selection - penalty, 6)
    return {
        "plaintext_validation": plaintext_report,
        "identity_candidate_id": "000_identity" if identity else None,
        "identity_selection_score": identity_selection,
        "identity_anneal_score": identity_anneal,
        "mutation_penalized_candidates": mutation_penalized,
        "identity_penalized_candidates": identity_penalized,
        "policy": (
            "Local mutations must beat their evaluated base candidate, and all "
            "non-identity finalists are compared against identity. Penalties "
            "affect finalist ordering but do not remove candidates from the artifact."
        ),
    }


def _confirm_transform_finalists(
    *,
    cipher_text: CipherText,
    language: str,
    ranked: list[dict[str, Any]],
    budget: str,
    solver_profile: str,
    confirm_count: int = 3,
    adaptive_confirmations: int = 2,
) -> dict[str, Any]:
    """Rerun top transform finalists with independent seeds.

    Stage B ranking can over-trust a single anneal basin. This confirmation
    pass gives the strongest finalists a fresh probe and ranks by stability.
    """

    finalists = [
        item for item in ranked
        if item.get("status") == "completed" and item.get("pipeline")
    ][:confirm_count]
    identity = next(
        (
            item for item in ranked
            if item.get("candidate_id") == "000_identity"
            and item.get("status") == "completed"
            and item.get("pipeline")
        ),
        None,
    )
    if identity is not None and all(item.get("candidate_id") != "000_identity" for item in finalists):
        finalists.append(identity)
    if (
        solver_profile == "zenith_native"
        and _zenith_native_engine() == "rust"
        and _transform_rank_engine() == "rust"
    ):
        return _confirm_transform_finalists_rust_batch(
            cipher_text=cipher_text,
            language=language,
            ranked=ranked,
            finalists=finalists,
            budget=budget,
            confirm_count=confirm_count,
            adaptive_confirmations=adaptive_confirmations,
        )
    confirmed = []
    skipped = []
    confirmed_ids: set[str] = set()

    def confirm_item(item: dict[str, Any], index: int, reason: str) -> None:
        seed_offset = 10_000 + index * 1_000
        started = time.time()
        try:
            pipeline = TransformPipeline.from_raw(item.get("pipeline"))
            if pipeline is None:
                raise ValueError("missing transform pipeline")
            transform_result = apply_transform_pipeline(cipher_text.tokens, pipeline)
            transformed_cipher = _cipher_text_from_tokens(
                transform_result.tokens,
                cipher_text.alphabet,
                source=f"{cipher_text.source}:transform_confirm:{index}",
            )
            solver, key, decryption, step = _run_homophonic(
                transformed_cipher,
                language,
                budget=budget,
                refinement="none",
                solver_profile=solver_profile,
                ground_truth=None,
                seed_offset=seed_offset,
            )
            anneal_score = step.get("anneal_score")
            quality_score = _plaintext_quality_score(decryption, language)
            mutation_penalty = _transform_mutation_penalty(item)
            confirmation_selection = _transform_selection_score(
                anneal_score=anneal_score,
                quality_score=quality_score,
                structural_score=item.get("structural_score"),
                mutation_penalty=mutation_penalty,
            )
            primary_score = (
                _float_or_none(item.get("validated_selection_score"))
                or _float_or_none(item.get("selection_score"))
                or float("-inf")
            )
            primary_text = str(item.get("decryption") or "")
            distance = _plaintext_distance_ratio(primary_text, decryption)
            stability_score = max(0.0, 1.0 - distance)
            confirmation_delta = (
                confirmation_selection - primary_score
                if math.isfinite(primary_score) else None
            )
            penalty = 0.08 * (1.0 - stability_score)
            reasons: list[str] = []
            if confirmation_delta is not None and confirmation_delta < -0.08:
                penalty += 0.08
                reasons.append("confirmation_selection_dropped")
            if stability_score < 0.55:
                penalty += 0.05
                reasons.append("confirmation_plaintext_unstable")
            confirmed_score = min(primary_score, confirmation_selection) if math.isfinite(primary_score) else confirmation_selection
            confirmed_score -= penalty
            confirmation = {
                "status": "completed",
                "solver": solver,
                "seed_offset": seed_offset,
                "confirmation_reason": reason,
                "budget": budget,
                "anneal_score": anneal_score,
                "plaintext_quality_score": round(quality_score, 6),
                "selection_score": round(confirmation_selection, 6),
                "selection_delta_vs_primary": (
                    round(confirmation_delta, 6)
                    if confirmation_delta is not None else None
                ),
                "plaintext_distance_ratio": round(distance, 6),
                "stability_score": round(stability_score, 6),
                "confirmation_penalty": round(penalty, 6),
                "confirmation_reasons": reasons,
                "elapsed_seconds": round(time.time() - started, 3),
                "decryption_preview": decryption[:500],
                "key": {str(k): v for k, v in key.items()},
            }
            item["confirmation"] = confirmation
            item["confirmed_selection_score"] = round(confirmed_score, 6)
            confirmed_ids.add(str(item.get("candidate_id")))
            confirmed.append({
                "candidate_id": item.get("candidate_id"),
                "family": item.get("family"),
                "seed_offset": seed_offset,
                "confirmation_reason": reason,
                "selection_score": confirmation["selection_score"],
                "selection_delta_vs_primary": confirmation["selection_delta_vs_primary"],
                "stability_score": confirmation["stability_score"],
                "confirmed_selection_score": item["confirmed_selection_score"],
                "reasons": reasons,
            })
        except Exception as exc:  # noqa: BLE001
            item["confirmation"] = {
                "status": "error",
                "seed_offset": seed_offset,
                "error": f"{type(exc).__name__}: {exc}",
            }
            item["confirmed_selection_score"] = (
                _float_or_none(item.get("validated_selection_score"))
                or _float_or_none(item.get("selection_score"))
                or float("-inf")
            ) - 0.12
            confirmed_ids.add(str(item.get("candidate_id")))
            skipped.append({
                "candidate_id": item.get("candidate_id"),
                "family": item.get("family"),
                "seed_offset": seed_offset,
                "confirmation_reason": reason,
                "reason": item["confirmation"]["error"],
            })

    for index, item in enumerate(finalists):
        confirm_item(item, index, "initial_finalist")
    best_confirmed = max(
        (
            _float_or_none(item.get("confirmed_selection_score")) or float("-inf")
            for item in ranked
            if str(item.get("candidate_id")) in confirmed_ids
        ),
        default=float("-inf"),
    )
    adaptive_margin = 0.04
    adaptive_count = 0
    max_adaptive_confirmations = max(0, adaptive_confirmations)
    if max_adaptive_confirmations > 0 and math.isfinite(best_confirmed):
        for item in ranked:
            if adaptive_count >= max_adaptive_confirmations:
                break
            candidate_id = str(item.get("candidate_id"))
            if candidate_id in confirmed_ids:
                continue
            base_score = (
                _float_or_none(item.get("validated_selection_score"))
                or _float_or_none(item.get("selection_score"))
                or float("-inf")
            )
            if base_score < best_confirmed - adaptive_margin:
                continue
            confirm_item(
                item,
                len(confirmed_ids),
                "adaptive_near_margin",
            )
            adaptive_count += 1
    unconfirmed_penalty = 0.12
    unconfirmed_count = 0
    for item in ranked:
        candidate_id = str(item.get("candidate_id"))
        if candidate_id in confirmed_ids:
            continue
        base_score = (
            _float_or_none(item.get("validated_selection_score"))
            or _float_or_none(item.get("selection_score"))
            or float("-inf")
        )
        item["confirmation"] = {
            "status": "not_run",
            "reason": "outside_confirmation_budget",
            "unconfirmed_penalty": unconfirmed_penalty,
        }
        item["confirmed_selection_score"] = round(base_score - unconfirmed_penalty, 6)
        unconfirmed_count += 1
    return {
        "stage": "independent_seed_confirmation",
        "confirmed_candidate_count": len(confirmed),
        "adaptive_confirmed_candidate_count": adaptive_count,
        "adaptive_margin": adaptive_margin,
        "unconfirmed_candidate_count": unconfirmed_count,
        "unconfirmed_penalty": unconfirmed_penalty,
        "skipped_candidates": skipped,
        "confirmed_candidates": confirmed,
        "policy": (
            "Stage C reruns the top transform finalists with independent seed "
            "offsets, always includes the identity control when available, "
            "and rewards candidates whose scores and plaintexts are stable "
            "across probes."
        ),
    }


def _confirm_transform_finalists_rust_batch(
    *,
    cipher_text: CipherText,
    language: str,
    ranked: list[dict[str, Any]],
    finalists: list[dict[str, Any]],
    budget: str,
    confirm_count: int = 3,
    adaptive_confirmations: int = 2,
) -> dict[str, Any]:
    """Confirm transform finalists using the Rust transform+Zenith batch kernel."""

    from analysis.zenith_fast import zenith_transform_candidates_batch_fast

    pt_alpha = _plaintext_alphabet(language)
    plaintext_ids = list(range(pt_alpha.size))
    id_to_letter = {i: pt_alpha.symbol_for(i).upper() for i in plaintext_ids}
    bin_path = _zenith_native_model_path(language)
    if bin_path is None:
        raise FileNotFoundError(
            f"Rust transform confirmation requires an ngram5 model for language={language!r}"
        )
    budget_params = _homophonic_budget_params(
        budget,
        len(cipher_text.tokens) < 600,
        search_profile=_homophonic_search_profile(),
    )
    confirmed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    confirmed_ids: set[str] = set()

    def run_confirmation_batch(
        items: list[dict[str, Any]],
        *,
        reason: str,
        start_index: int,
    ) -> None:
        payload_candidates: list[dict[str, Any]] = []
        metadata: dict[str, tuple[dict[str, Any], str, int, str]] = {}
        for offset, item in enumerate(items):
            seed_offset = 10_000 + (start_index + offset) * 1_000
            original_id = str(item.get("candidate_id"))
            batch_id = f"{original_id}__confirm_{seed_offset}"
            pipeline = item.get("pipeline")
            if not pipeline:
                item["confirmation"] = {
                    "status": "error",
                    "seed_offset": seed_offset,
                    "error": "missing transform pipeline",
                }
                item["confirmed_selection_score"] = (
                    _float_or_none(item.get("validated_selection_score"))
                    or _float_or_none(item.get("selection_score"))
                    or float("-inf")
                ) - 0.12
                confirmed_ids.add(original_id)
                skipped.append({
                    "candidate_id": item.get("candidate_id"),
                    "family": item.get("family"),
                    "seed_offset": seed_offset,
                    "confirmation_reason": reason,
                    "reason": "missing transform pipeline",
                })
                continue
            payload_candidates.append({
                "candidate_id": batch_id,
                "family": item.get("family"),
                "pipeline": pipeline,
                "grid": item.get("grid"),
                "seed_offset": seed_offset,
            })
            metadata[batch_id] = (item, original_id, seed_offset, reason)

        if not payload_candidates:
            return
        batch = zenith_transform_candidates_batch_fast(
            tokens=list(cipher_text.tokens),
            candidates=payload_candidates,
            plaintext_ids=plaintext_ids,
            id_to_letter=id_to_letter,
            model_path=bin_path,
            epochs=budget_params["epochs"],
            sampler_iterations=budget_params["sampler_iterations"],
            seeds=[int(seed) for seed in budget_params["seeds"]],
            top_n=1,
            threads=_transform_rank_threads(),
        )
        for row in batch.get("results", []):
            batch_id = str(row.get("candidate_id"))
            item, original_id, seed_offset, confirmation_reason = metadata[batch_id]
            if row.get("status") != "completed":
                item["confirmation"] = {
                    "status": "error",
                    "seed_offset": seed_offset,
                    "error": row.get("reason") or "rust_batch_candidate_failed",
                    "engine": "rust_batch",
                }
                item["confirmed_selection_score"] = (
                    _float_or_none(item.get("validated_selection_score"))
                    or _float_or_none(item.get("selection_score"))
                    or float("-inf")
                ) - 0.12
                confirmed_ids.add(original_id)
                skipped.append({
                    "candidate_id": item.get("candidate_id"),
                    "family": item.get("family"),
                    "seed_offset": seed_offset,
                    "confirmation_reason": confirmation_reason,
                    "reason": item["confirmation"]["error"],
                })
                continue
            decryption = str(row.get("decryption") or "")
            anneal_score = _float_or_none(row.get("normalized_score"))
            quality_score = _plaintext_quality_score(decryption, language)
            mutation_penalty = _transform_mutation_penalty(item)
            confirmation_selection = _transform_selection_score(
                anneal_score=anneal_score,
                quality_score=quality_score,
                structural_score=item.get("structural_score"),
                mutation_penalty=mutation_penalty,
            )
            primary_score = (
                _float_or_none(item.get("validated_selection_score"))
                or _float_or_none(item.get("selection_score"))
                or float("-inf")
            )
            primary_text = str(item.get("decryption") or "")
            distance = _plaintext_distance_ratio(primary_text, decryption)
            stability_score = max(0.0, 1.0 - distance)
            confirmation_delta = (
                confirmation_selection - primary_score
                if math.isfinite(primary_score) else None
            )
            penalty = 0.08 * (1.0 - stability_score)
            reasons: list[str] = []
            if confirmation_delta is not None and confirmation_delta < -0.08:
                penalty += 0.08
                reasons.append("confirmation_selection_dropped")
            if stability_score < 0.55:
                penalty += 0.05
                reasons.append("confirmation_plaintext_unstable")
            confirmed_score = (
                min(primary_score, confirmation_selection)
                if math.isfinite(primary_score) else confirmation_selection
            )
            confirmed_score -= penalty
            confirmation = {
                "status": "completed",
                "solver": "zenith_native",
                "engine": "rust_batch",
                "seed_offset": seed_offset,
                "best_seed": row.get("best_seed"),
                "confirmation_reason": confirmation_reason,
                "budget": budget,
                "anneal_score": anneal_score,
                "plaintext_quality_score": round(quality_score, 6),
                "selection_score": round(confirmation_selection, 6),
                "selection_delta_vs_primary": (
                    round(confirmation_delta, 6)
                    if confirmation_delta is not None else None
                ),
                "plaintext_distance_ratio": round(distance, 6),
                "stability_score": round(stability_score, 6),
                "confirmation_penalty": round(penalty, 6),
                "confirmation_reasons": reasons,
                "elapsed_seconds": round(float(row.get("elapsed_seconds") or 0.0), 3),
                "decryption_preview": decryption[:500],
                "key": {str(k): int(v) for k, v in dict(row.get("key") or {}).items()},
            }
            item["confirmation"] = confirmation
            item["confirmed_selection_score"] = round(confirmed_score, 6)
            confirmed_ids.add(original_id)
            confirmed.append({
                "candidate_id": item.get("candidate_id"),
                "family": item.get("family"),
                "seed_offset": seed_offset,
                "confirmation_reason": confirmation_reason,
                "selection_score": confirmation["selection_score"],
                "selection_delta_vs_primary": confirmation["selection_delta_vs_primary"],
                "stability_score": confirmation["stability_score"],
                "confirmed_selection_score": item["confirmed_selection_score"],
                "reasons": reasons,
            })

    run_confirmation_batch(finalists, reason="initial_finalist", start_index=0)
    best_confirmed = max(
        (
            _float_or_none(item.get("confirmed_selection_score")) or float("-inf")
            for item in ranked
            if str(item.get("candidate_id")) in confirmed_ids
        ),
        default=float("-inf"),
    )
    adaptive_margin = 0.04
    adaptive_items: list[dict[str, Any]] = []
    max_adaptive_confirmations = max(0, adaptive_confirmations)
    if max_adaptive_confirmations > 0 and math.isfinite(best_confirmed):
        for item in ranked:
            if len(adaptive_items) >= max_adaptive_confirmations:
                break
            candidate_id = str(item.get("candidate_id"))
            if candidate_id in confirmed_ids:
                continue
            base_score = (
                _float_or_none(item.get("validated_selection_score"))
                or _float_or_none(item.get("selection_score"))
                or float("-inf")
            )
            if base_score < best_confirmed - adaptive_margin:
                continue
            adaptive_items.append(item)
    run_confirmation_batch(
        adaptive_items,
        reason="adaptive_near_margin",
        start_index=len(confirmed_ids),
    )

    unconfirmed_penalty = 0.12
    unconfirmed_count = 0
    for item in ranked:
        candidate_id = str(item.get("candidate_id"))
        if candidate_id in confirmed_ids:
            continue
        base_score = (
            _float_or_none(item.get("validated_selection_score"))
            or _float_or_none(item.get("selection_score"))
            or float("-inf")
        )
        item["confirmation"] = {
            "status": "not_run",
            "reason": "outside_confirmation_budget",
            "unconfirmed_penalty": unconfirmed_penalty,
        }
        item["confirmed_selection_score"] = round(base_score - unconfirmed_penalty, 6)
        unconfirmed_count += 1
    return {
        "stage": "independent_seed_confirmation",
        "engine": "rust_batch",
        "confirmed_candidate_count": len(confirmed),
        "adaptive_confirmed_candidate_count": len(adaptive_items),
        "adaptive_margin": adaptive_margin,
        "unconfirmed_candidate_count": unconfirmed_count,
        "unconfirmed_penalty": unconfirmed_penalty,
        "skipped_candidates": skipped,
        "confirmed_candidates": confirmed,
        "policy": (
            "Stage C reruns the top transform finalists with independent seed "
            "offsets using the Rust transform+Zenith batch kernel, always "
            "includes the identity control when available, and rewards "
            "candidates whose scores and plaintexts are stable across probes."
        ),
    }


def _transform_final_sort_key(item: dict[str, Any]) -> tuple[bool, bool, float, float, float]:
    return (
        bool(item.get("selectable_transform_candidate")),
        item.get("status") == "completed",
        float(
            item.get("confirmed_selection_score")
            or item.get("validated_selection_score")
            or item.get("selection_score")
            or float("-inf")
        ),
        float(item.get("anneal_score") or float("-inf")),
        float(item.get("structural_score") or float("-inf")),
    )


def _label_transform_finalists(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    identity = next(
        (item for item in ranked if item.get("candidate_id") == "000_identity"),
        None,
    )
    identity_score = (
        _float_or_none(identity.get("confirmed_selection_score"))
        if identity else None
    )
    label_counts: Counter[str] = Counter()
    selectable_count = 0
    for item in ranked:
        gate = _transform_family_gate(item)
        confirmation = item.get("confirmation") if isinstance(item.get("confirmation"), dict) else {}
        status = confirmation.get("status")
        stability = _float_or_none(confirmation.get("stability_score"))
        score = _float_or_none(item.get("confirmed_selection_score"))
        identity_margin = (
            round(score - identity_score, 6)
            if score is not None
            and identity_score is not None
            and item.get("candidate_id") != "000_identity"
            else None
        )
        selectable = False
        if item.get("candidate_id") == "000_identity":
            if status == "completed" and stability is not None and stability >= gate["min_stability"]:
                label = "robust_baseline"
                selectable = True
            elif status == "completed":
                label = "unstable_baseline"
            else:
                label = "unconfirmed_baseline"
        elif status != "completed":
            label = "unconfirmed_candidate"
        elif stability is None or stability < gate["min_stability"]:
            label = "unstable_false_positive"
        elif identity_margin is not None and identity_margin < gate["required_identity_margin"]:
            label = "near_identity"
        else:
            label = "robust_candidate"
            selectable = True
        item["finalist_label"] = label
        item["selectable_transform_candidate"] = selectable
        item["family_gate"] = {
            **gate,
            "identity_margin": identity_margin,
            "confirmation_status": status,
            "stability_score": stability,
        }
        label_counts[label] += 1
        if selectable:
            selectable_count += 1
    return {
        "stage": "family_specific_evidence_gates",
        "identity_candidate_id": "000_identity" if identity else None,
        "identity_confirmed_selection_score": identity_score,
        "label_counts": dict(label_counts),
        "selectable_candidate_count": selectable_count,
        "policy": (
            "Finalists must survive independent-seed confirmation. Diagonal, "
            "columnar, unwrap, and local-mutation families require larger "
            "margins over identity than simple route/NDown families."
        ),
    }


def _choose_transform_candidate(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    for item in ranked:
        if item.get("selectable_transform_candidate"):
            is_identity = item.get("candidate_id") == "000_identity"
            return {
                "selected": True,
                "selected_candidate_id": item.get("candidate_id"),
                "family": item.get("family"),
                "finalist_label": item.get("finalist_label"),
                "selection_score": item.get("confirmed_selection_score"),
                "selects_transform": not is_identity,
                "reason": (
                    "identity_baseline_is_stronger_than_unstable_transforms"
                    if is_identity else "candidate_passed_confirmation_and_family_gate"
                ),
            }
    best = ranked[0] if ranked else {}
    return {
        "selected": False,
        "selected_candidate_id": None,
        "best_candidate_id": best.get("candidate_id"),
        "best_finalist_label": best.get("finalist_label"),
        "reason": "no_confirmed_candidate_passed_family_specific_gates",
    }


def _diagnose_transform_finalists(
    ranked: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Summarize near-miss vs. false-positive evidence for artifacts."""

    label_counts = Counter(str(item.get("finalist_label") or "unlabeled") for item in ranked)
    family_counts = Counter(_transform_family_class(item) for item in ranked)
    confirmed = [
        item for item in ranked
        if (item.get("confirmation") or {}).get("status") == "completed"
    ]
    robust_transforms = [
        item for item in confirmed
        if item.get("finalist_label") == "robust_candidate"
    ]
    unstable_false_positives = [
        item for item in confirmed
        if item.get("finalist_label") == "unstable_false_positive"
    ]
    near_identity = [
        item for item in confirmed
        if item.get("finalist_label") == "near_identity"
    ]
    unconfirmed = [
        item for item in ranked
        if (item.get("confirmation") or {}).get("status") == "not_run"
    ]
    if robust_transforms:
        conclusion = "robust_transform_candidate_found"
    elif selection.get("selected") and not selection.get("selects_transform"):
        conclusion = "identity_baseline_preferred_over_transform_candidates"
    elif unstable_false_positives:
        conclusion = "no_robust_transform_unstable_false_positives"
    elif near_identity:
        conclusion = "no_robust_transform_near_identity_only"
    else:
        conclusion = "no_robust_transform_found"

    top_evidence = [
        _transform_evidence_summary(item)
        for item in ranked[:10]
    ]
    return {
        "stage": "near_miss_false_positive_diagnostics",
        "conclusion": conclusion,
        "selected_candidate_id": selection.get("selected_candidate_id"),
        "selected_finalist_label": selection.get("finalist_label"),
        "selected_is_transform": selection.get("selects_transform"),
        "label_counts": dict(label_counts),
        "family_class_counts": dict(family_counts),
        "confirmed_candidate_count": len(confirmed),
        "robust_transform_count": len(robust_transforms),
        "unstable_false_positive_count": len(unstable_false_positives),
        "near_identity_count": len(near_identity),
        "unconfirmed_candidate_count": len(unconfirmed),
        "top_evidence": top_evidence,
        "policy": (
            "A transform finalist is treated as a near miss only when its "
            "plaintext quality, structural evidence, independent-seed "
            "stability, and margin over identity point in the same direction. "
            "High anneal score without stability is reported as a false "
            "positive, not progress."
        ),
    }


def _transform_evidence_summary(item: dict[str, Any]) -> dict[str, Any]:
    gate = item.get("family_gate") if isinstance(item.get("family_gate"), dict) else {}
    confirmation = item.get("confirmation") if isinstance(item.get("confirmation"), dict) else {}
    stability = _float_or_none(gate.get("stability_score"))
    min_stability = _float_or_none(gate.get("min_stability"))
    identity_margin = _float_or_none(gate.get("identity_margin"))
    required_margin = _float_or_none(gate.get("required_identity_margin"))
    quality = _float_or_none(item.get("plaintext_quality_score"))
    structural_delta = _float_or_none(item.get("structural_delta_vs_identity"))
    stability_pass = (
        stability is not None
        and min_stability is not None
        and stability >= min_stability
    )
    margin_pass = (
        item.get("candidate_id") == "000_identity"
        or (
            identity_margin is not None
            and required_margin is not None
            and identity_margin >= required_margin
        )
    )
    quality_signal = quality is not None and quality >= 0.25
    structural_signal = structural_delta is not None and structural_delta > 0.0
    reasons: list[str] = []
    if confirmation.get("status") != "completed":
        reasons.append(str(confirmation.get("reason") or confirmation.get("status") or "not_confirmed"))
    if not stability_pass and item.get("candidate_id") != "000_identity":
        reasons.append("failed_stability_gate")
    if not margin_pass:
        reasons.append("failed_identity_margin_gate")
    if not quality_signal:
        reasons.append("weak_plaintext_quality_signal")
    if not structural_signal and item.get("candidate_id") != "000_identity":
        reasons.append("weak_structural_delta")
    agreement_score = sum([
        bool(stability_pass),
        bool(margin_pass),
        bool(quality_signal),
        bool(structural_signal),
    ])
    if item.get("candidate_id") == "000_identity":
        agreement_score = sum([
            confirmation.get("status") == "completed",
            bool(stability_pass),
        ])
    return {
        "candidate_id": item.get("candidate_id"),
        "family": item.get("family"),
        "family_class": gate.get("family_class") or _transform_family_class(item),
        "finalist_label": item.get("finalist_label"),
        "confirmation_status": confirmation.get("status"),
        "confirmed_selection_score": item.get("confirmed_selection_score"),
        "anneal_score": item.get("anneal_score"),
        "plaintext_quality_score": item.get("plaintext_quality_score"),
        "structural_score": item.get("structural_score"),
        "structural_delta_vs_identity": item.get("structural_delta_vs_identity"),
        "stability_score": stability,
        "min_stability": min_stability,
        "stability_pass": stability_pass,
        "identity_margin": identity_margin,
        "required_identity_margin": required_margin,
        "identity_margin_pass": margin_pass,
        "quality_signal": quality_signal,
        "structural_signal": structural_signal,
        "evidence_agreement_score": agreement_score,
        "diagnostic_reasons": reasons,
    }


def _transform_family_gate(candidate: dict[str, Any]) -> dict[str, Any]:
    family_class = _transform_family_class(candidate)
    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    if candidate.get("candidate_id") == "000_identity" or family_class == "identity":
        required_identity_margin = 0.0
        min_stability = 0.40
    elif (
        family_class == "program_search"
        and params.get("template") == "banded_ndown_constructed"
    ):
        required_identity_margin = 0.08
        min_stability = 0.45
    elif family_class in {
        "diagonal_route",
        "columnar",
        "unwrap_columnar",
        "local_mutation",
        "grille_route",
        "interleave_route",
        "progressive_shift_route",
        "composite_route",
        "banded_ndown_lock_shift",
        "program_search",
        "grid_permute",
    }:
        required_identity_margin = 0.08
        min_stability = 0.65
    elif family_class in {"route_columns", "offset_chain", "whole"}:
        required_identity_margin = 0.05
        min_stability = 0.60
    elif family_class in {"ndownmacross", "route_rows", "split_grid", "row_reversals"}:
        required_identity_margin = 0.03
        min_stability = 0.55
    else:
        required_identity_margin = 0.05
        min_stability = 0.60
    return {
        "family_class": family_class,
        "required_identity_margin": required_identity_margin,
        "min_stability": min_stability,
    }


def _transform_mutation_penalty(candidate: dict[str, Any]) -> float:
    if candidate.get("provenance") == "local_mutation":
        return 0.08
    if candidate.get("provenance") == "program_search":
        params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
        if params.get("template") in {"banded_ndown_constructed", "route_repair_constructed"}:
            return 0.0
        return min(0.12, 0.02 * int(params.get("program_depth") or 1))
    return 0.0


def _transform_selection_score(
    *,
    anneal_score: Any,
    quality_score: float,
    structural_score: Any,
    mutation_penalty: float,
) -> float:
    return (
        float(anneal_score or float("-inf"))
        + quality_score * 0.05
        + float(structural_score or 0.0) * 0.03
        - mutation_penalty
    )


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _two_stage_transform_rank_candidates(
    screen: dict[str, Any],
    *,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a matrix/family-diverse finalist set for expensive solver probes."""

    pool: list[dict[str, Any]] = []
    identity = screen.get("identity_candidate")
    if identity:
        pool.append(identity)
    pool.extend(
        candidate
        for candidate in screen.get("top_candidates", [])
        if candidate.get("candidate_id") != "000_identity"
    )
    pool.extend(
        candidate
        for candidate in screen.get("anchor_candidates", [])
        if candidate.get("candidate_id") != "000_identity"
    )
    pool_by_id = {str(candidate.get("candidate_id")): candidate for candidate in pool}
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    selection_reasons: dict[str, str] = {}

    def maybe_add(candidate: dict[str, Any], reason: str) -> None:
        if len(selected) >= max_candidates:
            return
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id in seen_ids:
            return
        selected.append(candidate)
        seen_ids.add(candidate_id)
        selection_reasons[candidate_id] = reason

    if identity:
        maybe_add(identity, "identity_control")

    class_buckets: dict[str, list[dict[str, Any]]] = {}
    for candidate in pool:
        if candidate.get("candidate_id") == "000_identity":
            continue
        class_buckets.setdefault(_transform_family_class(candidate), []).append(candidate)

    for items in class_buckets.values():
        items.sort(key=_transform_triage_sort_key, reverse=True)

    priority = [
        "program_search",
        "route_rows",
        "route_columns",
        "banded_ndown_lock_shift",
        "ndownmacross",
        "row_reversals",
        "diagonal_route",
        "grille_route",
        "interleave_route",
        "progressive_shift_route",
        "composite_route",
        "grid_permute",
        "split_grid",
        "offset_chain",
        "columnar",
        "unwrap_columnar",
        "whole",
        "local_mutation",
        "other",
    ]
    class_counts: Counter[str] = Counter()
    for class_name in priority:
        bucket = class_buckets.get(class_name, [])
        if not bucket:
            continue
        if class_name == "program_search":
            reserved = 0
            if class_buckets.get("route_rows"):
                reserved += min(4, len(class_buckets["route_rows"]))
            if class_buckets.get("route_columns"):
                reserved += min(2, len(class_buckets["route_columns"]))
            limit = min(14, max(1, max_candidates - 1 - reserved))
        elif class_name == "route_rows":
            limit = 4
        elif class_name == "route_columns":
            limit = 2
        elif class_name in {"banded_ndown_lock_shift", "ndownmacross", "split_grid", "columnar", "unwrap_columnar", "composite_route"}:
            limit = 2
        else:
            limit = 1
        candidates = (
            _program_diverse_transform_candidates(bucket, limit=limit)
            if class_name == "program_search"
            else bucket[:limit]
        )
        for candidate in candidates:
            if len(selected) >= max_candidates:
                break
            maybe_add(candidate, f"family_diverse:{class_name}")
            class_counts[class_name] += 1
        if len(selected) >= max_candidates:
            break

    local_limit = max(1, max_candidates // 4)
    local_added = 0
    for candidate in class_buckets.get("local_mutation", []):
        if len(selected) >= max_candidates or local_added >= local_limit:
            break
        base_id = _transform_base_candidate_id(candidate)
        if base_id and base_id not in seen_ids and base_id in pool_by_id:
            maybe_add(pool_by_id[base_id], "base_for_local_mutation")
            if len(selected) >= max_candidates:
                break
        before = len(selected)
        maybe_add(candidate, "local_mutation_with_base")
        if len(selected) > before:
            local_added += 1

    if len(selected) < max_candidates:
        for candidate in sorted(pool, key=_transform_triage_sort_key, reverse=True):
            maybe_add(candidate, "triage_fill")
            if len(selected) >= max_candidates:
                break

    selected = selected[:max_candidates]
    report = {
        "stage": "structural_family_triage",
        "pool_candidate_count": len(pool),
        "screen_top_candidate_count": len(screen.get("top_candidates", []) or []),
        "selected_candidate_count": len(selected),
        "selected_candidates": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "family": candidate.get("family"),
                "family_class": _transform_family_class(candidate),
                "selection_reason": selection_reasons.get(str(candidate.get("candidate_id"))),
                "structural_score": candidate.get("score"),
                "matrix_rank_score": (candidate.get("metrics") or {}).get("matrix_rank_score"),
                "best_period": (candidate.get("metrics") or {}).get("best_period"),
            }
            for candidate in selected
        ],
        "class_counts": Counter(_transform_family_class(candidate) for candidate in selected),
        "selection_reasons": {
            str(candidate.get("candidate_id")): selection_reasons.get(str(candidate.get("candidate_id")))
            for candidate in selected
        },
        "policy": (
            "Stage A selects family-diverse structural finalists from a broad "
            "screen before Stage B spends homophonic solver probes."
        ),
    }
    report["class_counts"] = dict(report["class_counts"])
    return selected, report


def _program_diverse_transform_candidates(
    bucket: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep distinct program-search shapes alive through triage."""

    by_shape: dict[str, list[dict[str, Any]]] = {}
    for candidate in bucket:
        by_shape.setdefault(_program_shape_key(candidate), []).append(candidate)
    for items in by_shape.values():
        items.sort(key=_transform_triage_sort_key, reverse=True)

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def add(shape: str) -> bool:
        if len(selected) >= limit:
            return False
        for candidate in by_shape.get(shape, []):
            candidate_id = str(candidate.get("candidate_id"))
            if candidate_id in seen_ids:
                continue
            selected.append(candidate)
            seen_ids.add(candidate_id)
            return True
        return False

    def add_prefixed(prefix: str, max_items: int) -> None:
        added = 0
        keys = sorted(
            (key for key in by_shape if key.startswith(prefix)),
            key=lambda key: _transform_triage_sort_key(by_shape[key][0]),
            reverse=True,
        )
        for key in keys:
            if added >= max_items or len(selected) >= limit:
                break
            if add(key):
                added += 1

    def add_route_repair_by_grid(max_items: int) -> None:
        grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
        for candidate in bucket:
            if not _program_shape_key(candidate).startswith("route_repair_constructed:"):
                continue
            grid = candidate.get("grid") if isinstance(candidate.get("grid"), dict) else {}
            grouped.setdefault((grid.get("columns"), grid.get("rows")), []).append(candidate)
        groups = []
        for key, items in grouped.items():
            items.sort(key=_transform_triage_sort_key, reverse=True)
            groups.append((key, items[0]))
        groups.sort(key=lambda entry: _transform_triage_sort_key(entry[1]), reverse=True)
        added = 0
        for _key, candidate in groups:
            if added >= max_items or len(selected) >= limit:
                break
            candidate_id = str(candidate.get("candidate_id"))
            if candidate_id in seen_ids:
                continue
            selected.append(candidate)
            seen_ids.add(candidate_id)
            added += 1

    add_route_repair_by_grid(max_items=min(5, max(1, limit - 1)))
    add_prefixed("banded_ndown_constructed:", max_items=max(1, limit - len(selected)))
    add("program_other")
    for candidate in bucket:
        if len(selected) >= limit:
            break
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id in seen_ids:
            continue
        selected.append(candidate)
        seen_ids.add(candidate_id)
    return selected


def _program_shape_key(candidate: dict[str, Any]) -> str:
    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    template = str(params.get("template") or "")
    if template == "banded_ndown_constructed":
        return f"{template}:{_banded_program_variant_key(candidate)}"
    if template == "route_repair_constructed":
        grid = candidate.get("grid") if isinstance(candidate.get("grid"), dict) else {}
        labels = list(params.get("operation_labels") or [])
        route_label = next((str(label) for label in labels if str(label).startswith("route_")), "route")
        repair_label = next(
            (
                str(label)
                for label in labels
                if str(label).startswith("reverse_") or str(label).startswith("shift_")
            ),
            "repair",
        )
        return f"{template}:{route_label}:{repair_label}:{grid.get('columns')}:{grid.get('rows')}"
    return "program_other"


def _banded_program_variant_key(candidate: dict[str, Any]) -> str:
    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    labels = [str(label) for label in params.get("operation_labels") or []]
    top_across = "a?"
    split_value = _banded_program_split(candidate)
    split = f"s{split_value}" if split_value is not None else "s?"
    shift = "shift?"
    tail = "tail?"
    for label in labels:
        if label.startswith("ndown_top") and "_a" in label:
            top_across = label.rsplit("_", 1)[-1]
        if "shift_right" in label:
            shift = "right"
        elif "shift_left" in label:
            shift = "left"
        if label.startswith("tail_repair"):
            tail = label
    return f"{split}:{top_across}:{shift}:{tail}"


def _banded_program_split(candidate: dict[str, Any]) -> int | None:
    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    labels = [str(label) for label in params.get("operation_labels") or []]
    for label in labels:
        if label.startswith("ndown_top") and "_s" in label:
            parsed_split = _program_split_from_label(label)
            if parsed_split is not None:
                return parsed_split
    pipeline = candidate.get("pipeline") if isinstance(candidate.get("pipeline"), dict) else {}
    steps = pipeline.get("steps") if isinstance(pipeline.get("steps"), list) else []
    if steps:
        data = steps[0].get("data") if isinstance(steps[0], dict) else None
        if isinstance(data, dict) and data.get("rangeEnd") is not None:
            try:
                return int(data["rangeEnd"]) + 1
            except (TypeError, ValueError):
                return None
    return None


def _program_split_from_label(label: str) -> int | None:
    marker = "_s"
    if marker not in label:
        return None
    tail = label.split(marker, 1)[1]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else None


def _transform_family_class(candidate: dict[str, Any]) -> str:
    family = str(candidate.get("family") or "")
    if candidate.get("provenance") == "local_mutation":
        return "local_mutation"
    if family == "identity":
        return "identity"
    if family.startswith("ndownmacross"):
        return "ndownmacross"
    if family.startswith("banded_ndown_lock_shift"):
        return "banded_ndown_lock_shift"
    if family.startswith("program_"):
        return "program_search"
    if family.startswith("route_columns"):
        return "route_columns"
    if family.startswith("route_rows"):
        return "route_rows"
    if family.startswith("row_reversals"):
        return "row_reversals"
    if family.startswith("route_diagonal"):
        return "diagonal_route"
    if family.startswith("route_checkerboard"):
        return "grille_route"
    if family.startswith("route_row_column_interleave") or family.startswith("route_column_row_interleave"):
        return "interleave_route"
    if family.startswith("route_rows_progressive_shift") or family.startswith("route_columns_progressive_shift"):
        return "progressive_shift_route"
    if family.startswith("route_offset_chain"):
        return "offset_chain"
    if family.startswith("split_"):
        return "split_grid"
    if family.startswith("composite_"):
        return "composite_route"
    if family.startswith("grid_permute_"):
        return "grid_permute"
    if family.startswith("columnar_transposition"):
        return "columnar"
    if family.startswith("unwrap_transposition"):
        return "unwrap_columnar"
    if family.startswith("whole_"):
        return "whole"
    return "other"


def _transform_base_candidate_id(candidate: dict[str, Any]) -> str | None:
    params = candidate.get("params")
    if isinstance(params, dict) and params.get("base_candidate_id"):
        return str(params["base_candidate_id"])
    return None


def _transform_triage_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    structural = _float_or_none(candidate.get("score")) or 0.0
    matrix = _float_or_none(metrics.get("matrix_rank_score")) or 0.0
    periodic = max(
        _float_or_none(metrics.get("periodic_redundancy")) or 0.0,
        _float_or_none(metrics.get("inverse_periodic_redundancy")) or 0.0,
    )
    nontrivial = _float_or_none(metrics.get("position_nontriviality")) or 0.0
    template_bonus = 0.0
    if params.get("template") == "banded_ndown_constructed":
        template_bonus = 0.18
        labels = [str(label) for label in params.get("operation_labels") or []]
        if "tail_repair_pack" in labels:
            template_bonus += 0.02
        split = _banded_program_split(candidate)
        grid = candidate.get("grid") if isinstance(candidate.get("grid"), dict) else {}
        rows = grid.get("rows")
        if split is not None and isinstance(rows, int) and rows > 3:
            preferred = max(1, rows // 2 - 1)
            distance = abs(split - preferred)
            if distance == 0:
                template_bonus += 0.04
            elif distance <= 2:
                template_bonus += 0.025
            elif distance <= 4:
                template_bonus += 0.015
    elif params.get("template") == "route_repair_constructed":
        template_bonus = 0.10
    elif params.get("constructed_template_match") or params.get("calibration_template"):
        template_bonus = 0.12
    return (
        matrix * 0.45 + periodic * 0.25 + structural * 0.2 + nontrivial * 0.1 + template_bonus,
        matrix,
        structural,
        nontrivial,
    )


def _plaintext_quality_score(text: str, language: str) -> float:
    """Tiny no-boundary readability signal for transform finalist ranking."""

    cleaned = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if len(cleaned) < 20:
        return 0.0
    if language.lower().startswith("en"):
        fragments = (
            "THE", "AND", "ING", "ION", "ENT", "THAT", "WITH", "HER", "HIS",
            "FOR", "YOU", "NOT", "WAS", "HAVE", "THIS", "ARE",
        )
    else:
        fragments = ("THE", "AND", "ING", "ION", "ENT")
    fragment_hits = sum(cleaned.count(fragment) for fragment in fragments)
    fragment_rate = min(1.0, fragment_hits / max(1, len(cleaned) / 18))
    vowel_rate = sum(1 for ch in cleaned if ch in "AEIOU") / len(cleaned)
    vowel_score = max(0.0, 1.0 - abs(vowel_rate - 0.38) / 0.25)
    repeat_penalty = max(
        (len(match.group(0)) - 2) / len(cleaned)
        for match in re.finditer(r"([A-Z])\1{2,}", cleaned)
    ) if re.search(r"([A-Z])\1{2,}", cleaned) else 0.0
    return max(0.0, fragment_rate * 0.7 + vowel_score * 0.3 - repeat_penalty)


def format_automated_preflight_for_llm(
    result: AutomatedRunResult,
    max_plaintext_chars: int = 4000,
) -> str:
    """Format a no-LLM solver result as LLM-safe run context.

    This deliberately omits benchmark-only fields such as ground truth,
    character accuracy, and word accuracy. The LLM should treat the native
    result as a hypothesis to inspect, repair, or reject.
    """
    artifact = result.artifact or {}
    token_count = artifact.get("cipher_token_count", "?")
    alphabet_size = artifact.get("cipher_alphabet_size", "?")
    word_count = artifact.get("cipher_word_count", "?")
    steps = artifact.get("steps", []) or []

    lines = [
        "## Automated native solver preflight (no LLM access)",
        "",
        "A local automated solver ran before iteration 1. This result used no "
        "LLM calls, no LLM tokens, and no benchmark ground truth. Treat it as "
        "a hypothesis: if it reads coherently, inspect and repair residual "
        "errors; if it is incoherent, reject it and proceed independently.",
        "",
        f"- Status: {result.status}",
        f"- Solver: {result.solver or artifact.get('solver', 'unknown')}",
        "- Run mode: automated no-LLM",
        "- Cost: $0.00 (no LLM access)",
        f"- Cipher symbols: {alphabet_size}",
        f"- Cipher tokens: {token_count}",
        f"- Cipher word groups: {word_count}",
        f"- Automated branch available: `automated_preflight`",
    ]
    if result.error_message:
        lines.append(f"- Error: {result.error_message}")

    cipher_report = artifact.get("cipher_id_report")
    if isinstance(cipher_report, dict):
        scores = cipher_report.get("suspicion_scores") or {}
        ranked = sorted(
            ((str(mode), float(score)) for mode, score in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        lines += [
            "",
            "Cipher-type fingerprint:",
        ]
        summary = cipher_report.get("natural_language_summary")
        if isinstance(summary, str) and summary.strip():
            lines.append(f"- Summary: {summary}")
        if ranked:
            lines.append(
                "- Ranked mode suspicions: "
                + "; ".join(f"{mode}={score:.2f}" for mode, score in ranked[:5])
            )
        if cipher_report.get("best_period") is not None:
            lines.append(
                f"- Periodic IC best period: {cipher_report.get('best_period')} "
                f"(mean IC {cipher_report.get('best_period_ic')})"
            )
        lines.append(
            "- Agent tools: use `observe_cipher_id`, `observe_cipher_shape`, "
            "and mode-specific hypothesis branches before local repairs when "
            "the leading mode is uncertain."
        )

    primary_step = next(
        (step for step in steps if step.get("name") != "route_automated_solver"),
        steps[0] if steps else None,
    )

    if primary_step:
        lines += [
            "",
            "Native solver notes:",
            f"- Tool equivalent: {primary_step.get('name', 'unknown')}",
            f"- Model source: {primary_step.get('model_source', primary_step.get('solver', 'unknown'))}",
        ]
        if primary_step.get("model_note"):
            lines.append(f"- Model note: {primary_step['model_note']}")
        if "anneal_score" in primary_step:
            lines.append(f"- Anneal score: {primary_step['anneal_score']}")
        if "score" in primary_step:
            lines.append(f"- Score: {primary_step['score']}")
        if "candidates" in primary_step:
            lines.append(f"- Candidate count: {len(primary_step['candidates'])}")

    plaintext = result.final_decryption or ""
    if plaintext:
        truncated = len(plaintext) > max_plaintext_chars
        preview = plaintext[:max_plaintext_chars]
        lines += [
            "",
            "Best native candidate plaintext:",
            "```",
            preview + ("\n...[truncated]" if truncated else ""),
            "```",
        ]

    if primary_step and primary_step.get("candidates"):
        lines += ["", "Other native candidate previews:"]
        for candidate in primary_step["candidates"][1:4]:
            rank = candidate.get("rank", "?")
            score = candidate.get("anneal_score", candidate.get("score", "?"))
            preview = candidate.get("preview", "")
            lines.append(f"- Rank {rank}, score {score}: {preview[:240]}")

    return "\n".join(lines)


def _should_use_homophonic(
    cipher_text: CipherText,
    language: str,
    cipher_system: str = "",
) -> bool:
    return _select_solver_path(cipher_text, language, cipher_system)["route"] == "homophonic"


def _select_solver_path(
    cipher_text: CipherText,
    language: str,
    cipher_system: str = "",
    has_transform_pipeline: bool = False,
) -> dict[str, str]:
    pt_alpha = _plaintext_alphabet(language)
    cipher_name = cipher_system.lower()
    alphabet_size = cipher_text.alphabet.size
    word_groups = len(cipher_text.words)
    is_mixed_transposition = (
        any(token in cipher_name for token in ("transposition", "z340", "zodiac340"))
        and any(token in cipher_name for token in ("homophonic", "zodiac", "z340", "zodiac340"))
    )
    is_pure_transposition = (
        any(token in cipher_name for token in ("transmatrix", "kryptos3", "kryptos k3", "k3_transposition"))
        or (
            "transposition" in cipher_name
            and not any(token in cipher_name for token in ("homophonic", "zodiac", "z340", "zodiac340"))
        )
    )
    if any(token in cipher_name for token in ("vigenere", "vigenère", "beaufort", "gronsfeld", "polyalphabetic", "quagmire", "quag")):
        return {
            "route": "periodic_polyalphabetic",
            "solver": "periodic_polyalphabetic_screen",
            "reason": f"cipher_system={cipher_system or 'unknown'}",
        }

    if is_pure_transposition:
        return {
            "route": "pure_transposition",
            "solver": "k3_transmatrix_rust",
            "reason": f"cipher_system={cipher_system or 'unknown'}",
        }

    if is_mixed_transposition and not has_transform_pipeline:
        return {
            "route": "unsupported_mixed_transposition",
            "solver": "unsupported",
            "reason": (
                "mixed transposition+homophonic solving requires an explicit "
                "ciphertext transform pipeline or a bounded transform-search profile"
            ),
        }

    if any(token in cipher_name for token in ("homophonic", "zodiac", "copiale")):
        return {
            "route": "homophonic",
            "solver": "native_homophonic_anneal",
            "reason": f"cipher_system={cipher_system or 'unknown'}",
        }
    if alphabet_size > pt_alpha.size:
        return {
            "route": "homophonic",
            "solver": "native_homophonic_anneal",
            "reason": (
                f"cipher alphabet {alphabet_size} exceeds plaintext alphabet "
                f"{pt_alpha.size}"
            ),
        }
    if word_groups <= 1 and alphabet_size > 20:
        return {
            "route": "homophonic",
            "solver": "native_homophonic_anneal",
            "reason": (
                "single word-group and dense symbol inventory suggest "
                "no-boundary homophonic search"
            ),
        }
    return {
        "route": "substitution",
        "solver": "native_substitution_continuous_anneal" if language == "en" else "native_substitution_anneal",
        "reason": "default substitution path",
    }


def _run_pure_transposition(
    cipher_text: CipherText,
    language: str,
    cipher_system: str = "",
    solver_hints: dict[str, Any] | None = None,
) -> tuple[str, dict[int, int], str, dict[str, Any]]:
    """Run Rust-owned broad pure transposition screening."""

    from analysis.pure_transposition import (
        pure_transposition_profile_from_env,
        pure_transposition_threads_from_env,
        screen_pure_transposition,
    )

    solver_hints = solver_hints or {}
    known_params = solver_hints.get("known_cipher_parameters") if "known_cipher_parameters" in solver_hints else solver_hints
    profile = os.environ.get(
        "DECIPHER_PURE_TRANSPOSITION_PROFILE",
        pure_transposition_profile_from_env(),
    )
    max_candidates_raw = os.environ.get("DECIPHER_PURE_TRANSPOSITION_MAX_CANDIDATES", "").strip()
    max_candidates = int(max_candidates_raw) if max_candidates_raw else None
    transmatrix_min_width = int(
        os.environ.get(
            "DECIPHER_K3_TRANSMATRIX_MIN_WIDTH",
            str((known_params or {}).get("min_width", 2) if isinstance(known_params, dict) else 2),
        )
    )
    transmatrix_max_width_raw = os.environ.get("DECIPHER_K3_TRANSMATRIX_MAX_WIDTH", "").strip()
    transmatrix_max_width = int(transmatrix_max_width_raw) if transmatrix_max_width_raw else None
    include_matrix_rotate = _env_bool("DECIPHER_PURE_TRANSPOSITION_INCLUDE_MATRIX_ROTATE", default=True)
    include_transmatrix = _env_bool("DECIPHER_PURE_TRANSPOSITION_INCLUDE_TRANSMATRIX", default=True)
    include_route_composites = _env_bool("DECIPHER_PURE_TRANSPOSITION_INCLUDE_ROUTE_COMPOSITES", default=True)
    include_route_offsets = _env_bool("DECIPHER_PURE_TRANSPOSITION_INCLUDE_ROUTE_OFFSETS", default=True)
    include_mask_routes = _env_bool("DECIPHER_PURE_TRANSPOSITION_INCLUDE_MASK_ROUTES", default=True)
    top_n = int(os.environ.get("DECIPHER_PURE_TRANSPOSITION_TOP_N", os.environ.get("DECIPHER_K3_TRANSMATRIX_TOP_N", "25")))
    threads = pure_transposition_threads_from_env()
    result = screen_pure_transposition(
        cipher_text,
        language=language,
        profile=profile,
        top_n=top_n,
        max_candidates=max_candidates,
        include_matrix_rotate=include_matrix_rotate,
        include_transmatrix=include_transmatrix,
        include_route_composites=include_route_composites,
        include_route_offsets=include_route_offsets,
        include_mask_routes=include_mask_routes,
        transmatrix_min_width=transmatrix_min_width,
        transmatrix_max_width=transmatrix_max_width,
        threads=threads,
    )
    best = result.get("best_candidate")
    if not best:
        raise ValueError("pure transposition screen produced no candidate")
    step = {
        "name": "screen_pure_transposition",
        "solver": result.get("solver", "k3_transmatrix_rust"),
        "status": result.get("status"),
        "cipher_system": cipher_system,
        "profile": result.get("profile"),
        "candidate_count": result.get("candidate_count"),
        "valid_candidate_count": result.get("valid_candidate_count"),
        "threads": result.get("threads"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "cache": result.get("cache"),
        "candidate_plan": result.get("candidate_plan"),
        "validation_pool_size": result.get("validation_pool_size"),
        "family_counts": result.get("family_counts"),
        "top_family_counts": result.get("top_family_counts"),
        "transmatrix_min_width": result.get("transmatrix_min_width"),
        "transmatrix_max_width": result.get("transmatrix_max_width"),
        "selected": {
            "rank": best.get("rank"),
            "candidate_id": best.get("candidate_id"),
            "family": best.get("family"),
            "params": best.get("params"),
            "score": best.get("score"),
            "selection_score": best.get("selection_score"),
            "validated_selection_score": best.get("validated_selection_score"),
            "validation": best.get("validation"),
            "pipeline": best.get("pipeline"),
            "preview": best.get("preview"),
        },
        "top_candidates": result.get("top_candidates"),
        "note": (
            "Broad Rust-scored pure-transposition screen. It includes K3-style "
            "TransMatrix candidates plus grid/route/columnar families, and "
            "scores transformed text directly. It is separate from the "
            "transform+homophonic Z340 path."
        ),
    }
    return str(result.get("solver") or "pure_transposition_screen_rust"), {}, str(best.get("plaintext") or ""), step


def _run_periodic_polyalphabetic(
    cipher_text: CipherText,
    language: str,
    cipher_system: str = "",
    solver_hints: dict[str, Any] | None = None,
) -> tuple[str, dict[int, int], str, dict[str, Any]]:
    solver_hints = solver_hints or {}
    known_params = solver_hints.get("known_cipher_parameters") if "known_cipher_parameters" in solver_hints else solver_hints
    keyed_mode = os.environ.get("DECIPHER_KEYED_VIGENERE_MODE", "replay").strip().lower()
    quagmire_types = {
        "quagmire",
        "quagmirei",
        "quagmireii",
        "quagmireiii",
        "quagmireiv",
        "quagmire1",
        "quagmire2",
        "quagmire3",
        "quagmire4",
        "quagi",
        "quagii",
        "quagiii",
        "quagiv",
        "quag1",
        "quag2",
        "quag3",
        "quag4",
    }
    if (
        isinstance(known_params, dict)
        and str(known_params.get("type") or "").lower().replace("_", "").replace("-", "") in quagmire_types
        and keyed_mode in {"replay", "known_replay", "quagmire_replay"}
    ):
        qtype = (
            known_params.get("quagmire_type")
            or known_params.get("type")
            or known_params.get("variant")
            or "quag3"
        )
        cycleword = (
            known_params.get("cycleword")
            or known_params.get("periodic_key")
            or known_params.get("key")
            or ""
        )
        result = polyalphabetic.replay_quagmire(
            cipher_text,
            cycleword=str(cycleword),
            quagmire_type=qtype,
            plaintext_alphabet=known_params.get("plaintext_alphabet"),
            ciphertext_alphabet=known_params.get("ciphertext_alphabet"),
            plaintext_keyword=known_params.get("plaintext_keyword"),
            ciphertext_keyword=known_params.get("ciphertext_keyword"),
            alphabet_keyword=known_params.get("alphabet_keyword"),
            keyed_alphabet=known_params.get("keyed_alphabet"),
        )
        if result.get("status") != "completed":
            raise ValueError(result.get("reason", "Quagmire replay failed"))
        step = {
            "name": "replay_quagmire",
            "solver": result.get("solver"),
            "status": result.get("status"),
            "variant": result.get("variant"),
            "quagmire_type": result.get("quagmire_type"),
            "period": result.get("period"),
            "key_type": result.get("key_type"),
            "cycleword": result.get("cycleword"),
            "plaintext_alphabet": result.get("plaintext_alphabet"),
            "ciphertext_alphabet": result.get("ciphertext_alphabet"),
            "alphabet_keyword": result.get("alphabet_keyword"),
            "plaintext_keyword": result.get("plaintext_keyword"),
            "ciphertext_keyword": result.get("ciphertext_keyword"),
            "token_count": result.get("token_count"),
            "original_token_count": result.get("original_token_count"),
            "skipped_symbol_count": result.get("skipped_symbol_count"),
            "skipped_symbols": result.get("skipped_symbols"),
            "key_advances_over_skipped_symbols": result.get("key_advances_over_skipped_symbols"),
            "attribution": result.get("attribution"),
            "note": (
                "Known-parameter Quagmire replay from benchmark solver hints. "
                "This validates tableau semantics and artifact provenance; it "
                "is not unknown-key Quagmire search."
            ),
        }
        return str(result.get("solver") or "quagmire_known_replay"), {}, str(result.get("plaintext") or ""), step

    if keyed_mode in {"quagmire_search", "quag3_search", "quagmire3_search"}:
        quagmire_engine = os.environ.get("DECIPHER_QUAGMIRE_ENGINE", "rust_shotgun").strip().lower()
        keyword_lengths = _env_int_csv("DECIPHER_QUAGMIRE_KEYWORD_LENGTHS") or [7]
        cycleword_lengths = _env_int_csv("DECIPHER_QUAGMIRE_CYCLEWORD_LENGTHS") or list(
            range(1, int(os.environ.get("DECIPHER_POLYALPHABETIC_MAX_PERIOD", "12")) + 1)
        )
        initial_keywords = _env_csv("DECIPHER_QUAGMIRE_INITIAL_KEYWORDS")
        if quagmire_engine == "rust_shotgun":
            from analysis.polyalphabetic_fast import search_quagmire3_shotgun_fast

            result = search_quagmire3_shotgun_fast(
                cipher_text,
                language=language,
                keyword_lengths=keyword_lengths,
                cycleword_lengths=cycleword_lengths,
                hillclimbs=int(os.environ.get("DECIPHER_QUAGMIRE_HILLCLIMBS", "500")),
                restarts=int(os.environ.get("DECIPHER_QUAGMIRE_SEARCH_RESTARTS", "8")),
                seed=int(os.environ.get("DECIPHER_QUAGMIRE_SEARCH_SEED", "1")),
                top_n=10,
                slip_probability=float(os.environ.get("DECIPHER_QUAGMIRE_SLIP_PROB", "0.001")),
                backtrack_probability=float(os.environ.get("DECIPHER_QUAGMIRE_BACKTRACK_PROB", "0.15")),
                threads=int(os.environ.get("DECIPHER_QUAGMIRE_THREADS", "0")),
                initial_keywords=initial_keywords,
            )
        else:
            result = polyalphabetic.search_quagmire3_keyword_alphabet(
                cipher_text,
                language=language,
                keyword_lengths=keyword_lengths,
                cycleword_lengths=cycleword_lengths,
                initial_keywords=initial_keywords,
                steps=int(os.environ.get("DECIPHER_QUAGMIRE_SEARCH_STEPS", "500")),
                restarts=int(os.environ.get("DECIPHER_QUAGMIRE_SEARCH_RESTARTS", "8")),
                seed=int(os.environ.get("DECIPHER_QUAGMIRE_SEARCH_SEED", "1")),
                screen_top_n=int(os.environ.get("DECIPHER_QUAGMIRE_SCREEN_TOP_N", "128")),
                word_weight=float(os.environ.get("DECIPHER_QUAGMIRE_WORD_WEIGHT", "0.25")),
                slip_probability=float(os.environ.get("DECIPHER_QUAGMIRE_SLIP_PROB", "0.001")),
                backtrack_probability=float(os.environ.get("DECIPHER_QUAGMIRE_BACKTRACK_PROB", "0.15")),
                dictionary_keyword_limit=int(os.environ.get("DECIPHER_QUAGMIRE_DICTIONARY_STARTS", "0")),
                calibration_keyword=os.environ.get("DECIPHER_QUAGMIRE_CALIBRATION_KEYWORD"),
                top_n=10,
            )
        best = result.get("best_candidate") if isinstance(result, dict) else None
        if not best:
            raise ValueError(result.get("reason", "Quagmire III keyword search produced no candidate"))
        metadata = best.get("metadata") or {}
        step = {
            "name": "search_quagmire3_keyword_alphabet",
            "solver": result.get("solver") or "quagmire3_keyword_alphabet_search",
            "status": result.get("status"),
            "engine": quagmire_engine,
            "variant": best.get("variant"),
            "quagmire_type": metadata.get("quagmire_type", "quag3"),
            "period": best.get("period"),
            "key_type": metadata.get("key_type", "QuagmireKey"),
            "cycleword": metadata.get("cycleword") or best.get("key"),
            "shifts": best.get("shifts"),
            "score": best.get("score"),
            "alphabet_keyword": metadata.get("alphabet_keyword"),
            "plaintext_alphabet": metadata.get("plaintext_alphabet"),
            "ciphertext_alphabet": metadata.get("ciphertext_alphabet"),
            "keyword_lengths": result.get("keyword_lengths"),
            "cycleword_lengths": result.get("cycleword_lengths"),
            "initial_keywords": result.get("initial_keywords"),
            "dictionary_keyword_limit": result.get("dictionary_keyword_limit"),
            "dictionary_keywords_loaded": result.get("dictionary_keywords_loaded"),
            "calibration_keyword": result.get("calibration_keyword"),
            "exact_calibration_keyword_rank": result.get("exact_calibration_keyword_rank"),
            "best_calibration_keyword_distance": result.get("best_calibration_keyword_distance"),
            "steps_per_start": result.get("steps_per_start"),
            "hillclimbs_per_restart": result.get("hillclimbs_per_restart"),
            "restarts_per_length": result.get("restarts_per_length"),
            "restart_jobs": result.get("restart_jobs"),
            "nominal_proposals": result.get("nominal_proposals"),
            "threads": result.get("threads"),
            "keyword_states_screened": result.get("keyword_states_screened"),
            "screen_top_n": result.get("screen_top_n"),
            "refined_finalist_count": result.get("refined_finalist_count"),
            "word_weight": result.get("word_weight"),
            "screen_search": result.get("screen_search"),
            "slip_probability": result.get("slip_probability"),
            "backtrack_probability": result.get("backtrack_probability"),
            "accepted_screen_mutations": result.get("accepted_screen_mutations"),
            "slipped_screen_mutations": result.get("slipped_screen_mutations"),
            "backtrack_events": result.get("backtrack_events"),
            "seed": result.get("seed"),
            "top_candidates": result.get("top_candidates"),
            "attribution": result.get("attribution"),
            "note": (
                "Bounded Quagmire III keyword-alphabet search. It searches "
                "keyword-shaped alphabets and derives the cycleword for each "
                "candidate, inspired by Sam Blake's MIT-licensed "
                "polyalphabetic solver. This is a scaffold, not yet the full "
                "shotgun/backtracking implementation."
            ),
        }
        return str(result.get("solver") or "quagmire3_keyword_alphabet_search"), {}, str(best.get("plaintext") or ""), step

    if keyed_mode in {"alphabet_anneal", "tableau_anneal", "anneal"}:
        keywords = _env_csv("DECIPHER_KEYED_VIGENERE_TABLEAU_KEYWORDS")
        explicit_alphabets = _env_csv("DECIPHER_KEYED_VIGENERE_TABLEAUS")
        result = polyalphabetic.search_keyed_vigenere_alphabet_anneal(
            cipher_text,
            language=language,
            max_period=int(os.environ.get("DECIPHER_POLYALPHABETIC_MAX_PERIOD", "20")),
            initial_alphabets=explicit_alphabets,
            alphabet_keywords=keywords,
            include_standard_alphabet=True,
            steps=int(os.environ.get("DECIPHER_KEYED_VIGENERE_ANNEAL_STEPS", "2000")),
            restarts=int(os.environ.get("DECIPHER_KEYED_VIGENERE_ANNEAL_RESTARTS", "4")),
            seed=int(os.environ.get("DECIPHER_KEYED_VIGENERE_ANNEAL_SEED", "1")),
            guided=_env_bool("DECIPHER_KEYED_VIGENERE_ANNEAL_GUIDED", True),
            guided_pool_size=int(os.environ.get("DECIPHER_KEYED_VIGENERE_GUIDED_POOL", "24")),
            top_n=10,
        )
        best = result.get("best_candidate") if isinstance(result, dict) else None
        if not best:
            raise ValueError(result.get("reason", "keyed Vigenere alphabet anneal produced no candidate"))
        metadata = best.get("metadata") or {}
        step = {
            "name": "search_keyed_vigenere_alphabet_anneal",
            "solver": "keyed_vigenere_alphabet_anneal",
            "status": result.get("status"),
            "variant": best.get("variant"),
            "period": best.get("period"),
            "key_type": "PeriodicAlphabetKey",
            "key": best.get("key"),
            "shifts": best.get("shifts"),
            "score": best.get("score"),
            "keyed_alphabet": metadata.get("keyed_alphabet"),
            "alphabet_keyword": metadata.get("alphabet_keyword"),
            "initial_keyed_alphabet": metadata.get("initial_keyed_alphabet"),
            "initial_candidate_type": metadata.get("initial_candidate_type"),
            "periods_tested": result.get("periods_tested"),
            "initial_alphabets_tested": result.get("initial_alphabets_tested"),
            "steps_per_period": result.get("steps_per_period"),
            "restarts_per_alphabet": result.get("restarts_per_alphabet"),
            "guided": result.get("guided"),
            "guided_pool_size": result.get("guided_pool_size"),
            "top_candidates": result.get("top_candidates"),
            "note": (
                "Experimental shared-tableau mutation search. It re-optimizes "
                "periodic shifts after alphabet mutations; guided mode adds "
                "frequency/phase proposals. Current scope is near-basin "
                "refinement and research diagnostics, not robust blind "
                "Kryptos recovery."
            ),
        }
        return "keyed_vigenere_alphabet_anneal", {}, str(best.get("plaintext") or ""), step

    if keyed_mode in {"tableau_search", "keyword_search", "alphabet_search"}:
        keywords = _env_csv("DECIPHER_KEYED_VIGENERE_TABLEAU_KEYWORDS")
        explicit_alphabets = _env_csv("DECIPHER_KEYED_VIGENERE_TABLEAUS")
        result = polyalphabetic.search_keyed_vigenere(
            cipher_text,
            language=language,
            max_period=int(os.environ.get("DECIPHER_POLYALPHABETIC_MAX_PERIOD", "20")),
            keyed_alphabets=explicit_alphabets,
            alphabet_keywords=keywords,
            include_standard_alphabet=True,
            top_n=10,
            refine=True,
        )
        best = result.get("best_candidate") if isinstance(result, dict) else None
        if not best:
            raise ValueError(result.get("reason", "keyed Vigenere tableau search produced no candidate"))
        metadata = best.get("metadata") or {}
        step = {
            "name": "search_keyed_vigenere_tableaux",
            "solver": "keyed_vigenere_tableau_search",
            "status": result.get("status"),
            "variant": best.get("variant"),
            "period": best.get("period"),
            "key_type": "PeriodicAlphabetKey",
            "key": best.get("key"),
            "shifts": best.get("shifts"),
            "score": best.get("score"),
            "keyed_alphabet": metadata.get("keyed_alphabet"),
            "alphabet_keyword": metadata.get("alphabet_keyword"),
            "periods_tested": result.get("periods_tested"),
            "alphabet_candidates_tested": result.get("alphabet_candidates_tested"),
            "top_candidates": result.get("top_candidates"),
            "note": (
                "Searched standard Vigenere first, then keyword/explicit "
                "candidate keyed alphabets from environment. This recovers a "
                "tableau only within the provided candidate list."
            ),
        }
        return "keyed_vigenere_tableau_search", {}, str(best.get("plaintext") or ""), step

    if (
        isinstance(known_params, dict)
        and str(known_params.get("type") or "").lower() in {"keyed_vigenere", "kryptos_keyed_vigenere"}
        and keyed_mode in {"search", "solve"}
    ):
        result = polyalphabetic.search_keyed_vigenere(
            cipher_text,
            language=language,
            max_period=int(os.environ.get("DECIPHER_POLYALPHABETIC_MAX_PERIOD", "20")),
            keyed_alphabets=[str(known_params["keyed_alphabet"])] if known_params.get("keyed_alphabet") else None,
            alphabet_keywords=[str(known_params["alphabet_keyword"])] if known_params.get("alphabet_keyword") else None,
            top_n=10,
            refine=True,
        )
        best = result.get("best_candidate") if isinstance(result, dict) else None
        if not best:
            raise ValueError(result.get("reason", "keyed Vigenere search produced no candidate"))
        metadata = best.get("metadata") or {}
        step = {
            "name": "search_keyed_vigenere",
            "solver": "keyed_vigenere_periodic_key_search",
            "status": result.get("status"),
            "variant": best.get("variant"),
            "period": best.get("period"),
            "key_type": "PeriodicAlphabetKey",
            "key": best.get("key"),
            "shifts": best.get("shifts"),
            "score": best.get("score"),
            "keyed_alphabet": metadata.get("keyed_alphabet"),
            "alphabet_keyword": metadata.get("alphabet_keyword"),
            "periods_tested": result.get("periods_tested"),
            "alphabet_candidates_tested": result.get("alphabet_candidates_tested"),
            "top_candidates": result.get("top_candidates"),
            "note": (
                "Recovered the periodic key over supplied candidate keyed "
                "alphabet/tableau metadata. This is not arbitrary keyed-alphabet discovery."
            ),
        }
        return "keyed_vigenere_periodic_key_search", {}, str(best.get("plaintext") or ""), step

    if (
        isinstance(known_params, dict)
        and str(known_params.get("type") or "").lower() in {"keyed_vigenere", "kryptos_keyed_vigenere"}
        and known_params.get("periodic_key")
    ):
        result = polyalphabetic.replay_keyed_vigenere(
            cipher_text,
            key=str(known_params.get("periodic_key") or ""),
            keyed_alphabet=known_params.get("keyed_alphabet"),
            alphabet_keyword=known_params.get("alphabet_keyword"),
        )
        if result.get("status") != "completed":
            raise ValueError(result.get("reason", "keyed Vigenere replay failed"))
        step = {
            "name": "replay_keyed_vigenere",
            "solver": "keyed_vigenere_known_replay",
            "status": result.get("status"),
            "variant": result.get("variant"),
            "period": result.get("period"),
            "key_type": result.get("key_type"),
            "key": result.get("key"),
            "keyed_alphabet": result.get("keyed_alphabet"),
            "alphabet_keyword": result.get("alphabet_keyword"),
            "token_count": result.get("token_count"),
            "original_token_count": result.get("original_token_count"),
            "skipped_symbol_count": result.get("skipped_symbol_count"),
            "skipped_symbols": result.get("skipped_symbols"),
            "key_advances_over_skipped_symbols": result.get("key_advances_over_skipped_symbols"),
            "note": (
                "Known-parameter keyed Vigenere replay from benchmark solver "
                "hints. This is a calibration/replay path, not unknown-key search."
            ),
        }
        return "keyed_vigenere_known_replay", {}, str(result.get("plaintext") or ""), step

    variants = _periodic_variants_for_cipher_system(cipher_system)
    result = polyalphabetic.search_periodic_polyalphabetic(
        cipher_text,
        language=language,
        max_period=int(os.environ.get("DECIPHER_POLYALPHABETIC_MAX_PERIOD", "20")),
        variants=variants,
        top_n=10,
        refine=True,
    )
    best = result.get("best_candidate") if isinstance(result, dict) else None
    if not best:
        raise ValueError(result.get("reason", "periodic polyalphabetic search produced no candidate"))
    step = {
        "name": "search_periodic_polyalphabetic",
        "solver": "periodic_polyalphabetic_screen",
        "status": result.get("status"),
        "variant": best.get("variant"),
        "period": best.get("period"),
        "key_type": "PeriodicShiftKey",
        "key": best.get("key"),
        "shifts": best.get("shifts"),
        "score": best.get("score"),
        "periods_tested": result.get("periods_tested"),
        "variants_tested": result.get("variants_tested"),
        "top_candidates": result.get("top_candidates"),
        "note": (
            "Periodic polyalphabetic search returns mode-specific key state, "
            "not a substitution mapping. The artifact key is intentionally empty."
        ),
    }
    return "periodic_polyalphabetic_screen", {}, str(best.get("plaintext") or ""), step


def _periodic_variants_for_cipher_system(cipher_system: str) -> list[str] | None:
    name = (cipher_system or "").lower()
    if "gronsfeld" in name:
        return ["gronsfeld"]
    if "variant" in name and "beaufort" in name:
        return ["variant_beaufort"]
    if "beaufort" in name:
        return ["beaufort"]
    if "vigenere" in name or "vigenère" in name:
        return ["vigenere"]
    return None


def _env_csv(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _env_int_csv(name: str) -> list[int]:
    values: list[int] = []
    for part in _env_csv(name):
        values.append(int(part))
    return values


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _cipher_text_from_tokens(tokens: list[int], alphabet: Alphabet, source: str = "transform") -> CipherText:
    raw = alphabet.decode(tokens)
    return CipherText(raw=raw, alphabet=alphabet, source=source, separator=None)


def _run_homophonic(
    cipher_text: CipherText,
    language: str,
    budget: str = "full",
    refinement: str = "none",
    solver_profile: str = "zenith_native",
    ground_truth: str | None = None,
    seed_offset: int = 0,
) -> tuple[str, dict[int, int], str, dict[str, Any]]:
    pt_alpha = _plaintext_alphabet(language)
    plaintext_ids = list(range(pt_alpha.size))
    id_to_letter = {i: pt_alpha.symbol_for(i).upper() for i in plaintext_ids}
    letter_to_id = {letter: i for i, letter in id_to_letter.items()}
    word_list = _word_list(language)
    model, model_note = _homophonic_model(language, word_list)
    started = time.time()
    short_homophonic = len(cipher_text.tokens) < 600
    search_profile = _homophonic_search_profile()
    budget_params = _homophonic_budget_params(
        budget,
        short_homophonic,
        search_profile=search_profile,
    )
    if seed_offset:
        budget_params = dict(budget_params)
        budget_params["seeds"] = [
            int(seed) + seed_offset
            for seed in budget_params["seeds"]
        ]
        budget_params["seed_offset"] = seed_offset
    seeds = budget_params["seeds"]
    epochs = budget_params["epochs"]
    sampler_iterations = budget_params["sampler_iterations"]
    attempts = []
    result = None
    result_seed = None
    result_quality = None
    result_selection_score = float("-inf")
    selection_profile = _homophonic_selection_profile()
    move_profile = _homophonic_move_profile()
    aggregated_candidates: list[dict[str, Any]] = []
    score_profile = _homophonic_score_profile(solver_profile)

    # Dispatch to the Zenith-parity solver when score_profile == "zenith_native"
    if score_profile == "zenith_native":
        return _run_homophonic_zenith_native(
            cipher_text=cipher_text,
            language=language,
            budget=budget,
            ground_truth=ground_truth,
            pt_alpha=pt_alpha,
            plaintext_ids=plaintext_ids,
            id_to_letter=id_to_letter,
            letter_to_id=letter_to_id,
            short_homophonic=short_homophonic,
            budget_params=budget_params,
            started=started,
        )

    score_config = _homophonic_score_config(score_profile, short_homophonic)
    score_weights = _score_config_weights(score_config)
    use_early_stop = _homophonic_use_early_stop()
    repair_profile = _homophonic_repair_profile()
    for seed in seeds:
        best_completed_selection_score = result_selection_score
        early_stop_hook = None
        if use_early_stop and math.isfinite(best_completed_selection_score):
            early_stop_hook = _make_homophonic_early_stop_hook(
                language=language,
                word_list=word_list,
                best_completed_selection_score=best_completed_selection_score,
            )
        candidate = homophonic.homophonic_simulated_anneal(
            tokens=list(cipher_text.tokens),
            plaintext_ids=plaintext_ids,
            id_to_letter=id_to_letter,
            letter_to_id=letter_to_id,
            model=model,
            epochs=epochs,
            sampler_iterations=sampler_iterations,
            distribution_weight=score_weights["distribution_weight"],
            diversity_weight=score_weights["diversity_weight"],
            ioc_weight=score_weights["ioc_weight"],
            score_formula=score_config["score_formula"],
            window_step=score_config["window_step"],
            move_profile=move_profile,
            seed=seed,
            top_n=12,
            epoch_callback=early_stop_hook,
        )
        quality = _plaintext_quality(candidate.plaintext, candidate.key)
        diagnostics = _automated_candidate_diagnostics(
            candidate.plaintext,
            language=language,
            word_list=word_list,
        )
        selection_score = _score_homophonic_candidate_for_selection(
            candidate.normalized_score,
            quality,
            diagnostics,
            selection_profile="anneal_quality",
        )
        candidate_records = []
        for seed_candidate in candidate.candidates:
            candidate_quality = _plaintext_quality(seed_candidate.plaintext, seed_candidate.key)
            candidate_diagnostics = _automated_candidate_diagnostics(
                seed_candidate.plaintext,
                language=language,
                word_list=word_list,
            )
            candidate_selection_score = _score_homophonic_candidate_for_selection(
                seed_candidate.normalized_score,
                candidate_quality,
                candidate_diagnostics,
                selection_profile=selection_profile,
            )
            candidate_record = {
                "seed": seed,
                "epoch": seed_candidate.epoch,
                "anneal_score": round(seed_candidate.normalized_score, 4),
                "selection_score": round(candidate_selection_score, 4),
                "quality": candidate_quality,
                "diagnostics": candidate_diagnostics,
                "preview": seed_candidate.plaintext[:300],
                "key": seed_candidate.key,
                "plaintext": seed_candidate.plaintext,
            }
            if ground_truth is not None:
                candidate_record["debug_char_accuracy"] = round(
                    score_decryption(
                        test_id="candidate_debug",
                        decrypted=seed_candidate.plaintext,
                        ground_truth=ground_truth,
                        agent_score=0.0,
                        status="completed",
                    ).char_accuracy,
                    4,
                )
            candidate_records.append(candidate_record)
            aggregated_candidates.append(candidate_record)
        attempts.append({
            "seed": seed,
            "collapsed": quality["collapsed"],
            "quality_ok": quality["ok"],
            "selection_score": round(selection_score, 4),
            "anneal_score": round(candidate.normalized_score, 4),
            "quality": quality,
            "diagnostics": diagnostics,
            "epoch_traces": candidate.metadata.get("epoch_traces", []),
            "move_telemetry": candidate.metadata.get("move_telemetry", {}),
            "stopped_early": candidate.metadata.get("stopped_early", False),
            "stopped_after_epoch": candidate.metadata.get("stopped_after_epoch"),
            "stop_reason": candidate.metadata.get("stop_reason"),
            "preview": candidate.plaintext[:120],
            "candidates": candidate_records,
        })
        if result is None or selection_score > result_selection_score:
            result = candidate
            result_seed = seed
            result_quality = quality
            result_selection_score = selection_score
        if quality["ok"] and not short_homophonic:
            break
    if result is None:
        raise ValueError("homophonic anneal produced no result")

    selected_plaintext = result.plaintext
    selected_key = result.key
    selected_candidates = result.candidates
    selected_epochs = result.epochs
    selected_sampler_iterations = result.sampler_iterations
    selected_anneal_score = result.normalized_score
    selected_diagnostics = _automated_candidate_diagnostics(
        selected_plaintext,
        language=language,
        word_list=word_list,
    )
    selected_family_diagnostics = _homophonic_family_diagnostics(
        cipher_text,
        selected_key,
        id_to_letter,
        model,
        window_step=score_config["window_step"],
    )
    elite_limit = 2 if repair_profile == "dev" else 5
    elite_candidates = _select_diverse_homophonic_elites(
        aggregated_candidates,
        limit=elite_limit,
    )

    selection_summary: dict[str, Any] | None = None
    if selection_profile != "anneal_quality":
        reranked_pool = _rank_homophonic_candidate_pool(
            aggregated_candidates,
            selection_profile=selection_profile,
        )
        if reranked_pool:
            top_choice = reranked_pool[0]
            reranked_result = next(
                (
                    candidate
                    for attempt in attempts
                    for candidate in attempt["candidates"]
                    if candidate["plaintext"] == top_choice["plaintext"]
                ),
                None,
            )
            selection_summary = {
                "profile": selection_profile,
                "pool_size": len(reranked_pool),
                "selected_seed": top_choice["seed"],
                "selected_epoch": top_choice["epoch"],
                "selected_anneal_score": top_choice["anneal_score"],
                "selected_selection_score": top_choice["selection_score"],
                "selected_preview": top_choice["preview"][:160],
                "top_candidates": [
                    {
                        "rank": i + 1,
                        "seed": item["seed"],
                        "epoch": item["epoch"],
                        "anneal_score": item["anneal_score"],
                        "selection_score": item["selection_score"],
                        "preview": item["preview"][:160],
                        "diagnostics": item["diagnostics"],
                        **(
                            {"debug_char_accuracy": item["debug_char_accuracy"]}
                            if "debug_char_accuracy" in item
                            else {}
                        ),
                    }
                    for i, item in enumerate(reranked_pool[:10])
                ],
            }
            if reranked_result is not None:
                selected_plaintext = reranked_result["plaintext"]
                selected_key = reranked_result["key"]
                selected_candidates = [
                    homophonic.HomophonicCandidate(
                        plaintext=item["plaintext"],
                        key=item["key"],
                        score=item["anneal_score"],
                        normalized_score=item["anneal_score"],
                        epoch=item["epoch"],
                    )
                    for item in reranked_pool[:3]
                ]
                selected_anneal_score = top_choice["anneal_score"]
                result_seed = f"{top_choice['seed']}@{top_choice['epoch']}"
                result_quality = reranked_result["quality"]
                result_selection_score = top_choice["selection_score"]
                selected_diagnostics = reranked_result["diagnostics"]
                selected_family_diagnostics = _homophonic_family_diagnostics(
                    cipher_text,
                    selected_key,
                    id_to_letter,
                    model,
                    window_step=score_config["window_step"],
                )

    refinement_step: dict[str, Any] | None = None
    if refinement != "none":
        refine_params = _homophonic_refinement_params(
            refinement,
            budget,
            short_homophonic,
            repair_profile=repair_profile,
        )
        refine_config = _homophonic_score_config(refine_params["profile"], short_homophonic)
        refine_weights = _score_config_weights(refine_config)
        if refine_params["mode"] == "targeted_repair":
            refined, refinement_step = _run_targeted_homophonic_repair(
                cipher_text=cipher_text,
                language=language,
                word_list=word_list,
                selected_key=selected_key,
                selected_plaintext=selected_plaintext,
                model=model,
                plaintext_ids=plaintext_ids,
                id_to_letter=id_to_letter,
                letter_to_id=letter_to_id,
                refine_params=refine_params,
                refine_config=refine_config,
                refine_weights=refine_weights,
            )
        elif refine_params["mode"] == "family_repair":
            refined, refinement_step = _run_family_homophonic_repair(
                cipher_text=cipher_text,
                language=language,
                word_list=word_list,
                elite_candidates=elite_candidates,
                selected_key=selected_key,
                selected_plaintext=selected_plaintext,
                selected_quality=result_quality or _plaintext_quality(selected_plaintext, selected_key),
                selected_diagnostics=selected_diagnostics,
                model=model,
                plaintext_ids=plaintext_ids,
                id_to_letter=id_to_letter,
                letter_to_id=letter_to_id,
                refine_params=refine_params,
                refine_config=refine_config,
                refine_weights=refine_weights,
            )
        else:
            refined = homophonic.homophonic_simulated_anneal(
                tokens=list(cipher_text.tokens),
                plaintext_ids=plaintext_ids,
                id_to_letter=id_to_letter,
                letter_to_id=letter_to_id,
                model=model,
                initial_key=selected_key,
                epochs=refine_params["epochs"],
                sampler_iterations=refine_params["sampler_iterations"],
                t_start=refine_params["t_start"],
                t_end=refine_params["t_end"],
                distribution_weight=refine_weights["distribution_weight"],
                diversity_weight=refine_weights["diversity_weight"],
                ioc_weight=refine_weights["ioc_weight"],
                score_formula=refine_config["score_formula"],
                window_step=refine_config["window_step"],
                move_profile=move_profile,
                seed=refine_params["seed"],
                top_n=3,
            )
            refinement_step = {
                "mode": refinement,
                "profile": refine_params["profile"],
                "weights": refine_weights,
                "score_formula": refine_config["score_formula"],
                "window_step": refine_config["window_step"],
                "epochs": refine_params["epochs"],
                "sampler_iterations": refine_params["sampler_iterations"],
                "t_start": refine_params["t_start"],
                "t_end": refine_params["t_end"],
                "seed": refine_params["seed"],
            }
        if refined is not None:
            refined_quality = _plaintext_quality(refined.plaintext, refined.key)
            refined_diagnostics = _automated_candidate_diagnostics(
                refined.plaintext,
                language=language,
                word_list=word_list,
            )
            refined_selection_score = refined.normalized_score - refined_quality["penalty"]
            adoption_epsilon = 1e-4
            adopted = refined_selection_score > (result_selection_score + adoption_epsilon)
            refinement_step.update({
                "base_selection_score": round(result_selection_score, 4),
                "refined_selection_score": round(refined_selection_score, 4),
                "adopted": adopted,
                "adoption_epsilon": adoption_epsilon,
                "quality": refined_quality,
                "diagnostics": refined_diagnostics,
                "epoch_traces": refined.metadata.get("epoch_traces", []),
                "preview": refined.plaintext[:160],
            })
            if adopted:
                result = refined
                selected_plaintext = refined.plaintext
                selected_key = refined.key
                selected_candidates = refined.candidates
                selected_epochs = refined.epochs
                selected_sampler_iterations = refined.sampler_iterations
                selected_anneal_score = refined.normalized_score
                result_seed = f"{result_seed}->refine"
                result_quality = refined_quality
                result_selection_score = refined_selection_score
                selected_diagnostics = refined_diagnostics
                selected_family_diagnostics = _homophonic_family_diagnostics(
                    cipher_text,
                    selected_key,
                    id_to_letter,
                    model,
                    window_step=score_config["window_step"],
                )

    step = {
        "name": "search_homophonic_anneal",
        "solver": "native_homophonic_anneal",
        "model_source": model.source,
        "model_note": model_note,
        "homophonic_budget": budget,
        "budget_params": budget_params,
        "homophonic_refinement": refinement,
        "seed_offset": seed_offset,
        "selection_profile": selection_profile,
        "early_stop_enabled": use_early_stop,
        "search_profile": search_profile,
        "repair_profile": repair_profile,
        "move_profile": move_profile,
        "score_profile": score_profile,
        "score_weights": score_weights,
        "score_formula": score_config["score_formula"],
        "window_step": score_config["window_step"],
        "anneal_score": round(selected_anneal_score, 4),
        "selection_score": round(result_selection_score, 4),
        "quality": result_quality,
        "diagnostics": selected_diagnostics,
        "family_diagnostics": selected_family_diagnostics,
        "elapsed_seconds": round(time.time() - started, 3),
        "epochs": selected_epochs,
        "sampler_iterations": selected_sampler_iterations,
        "seed": result_seed,
        "seed_attempts": attempts,
        "move_telemetry": (
            next(
                (attempt.get("move_telemetry", {}) for attempt in attempts if attempt["seed"] == result_seed),
                {},
            )
            if isinstance(result_seed, int)
            else (
                next(
                    (
                        attempt.get("move_telemetry", {})
                        for attempt in attempts
                        if str(attempt["seed"]) == str(result_seed).split("@", 1)[0]
                    ),
                    {},
                )
            )
        ),
        "collapse_retries": sum(1 for attempt in attempts[:-1] if attempt["collapsed"]),
        "quality_retries": max(0, len(attempts) - 1),
        "selection": selection_summary,
        "refinement": refinement_step,
        "elite_candidates": [
            {
                "rank": i + 1,
                "seed": candidate.get("seed"),
                "epoch": candidate.get("epoch"),
                "anneal_score": candidate.get("anneal_score"),
                "selection_score": candidate.get("selection_score"),
                "preview": (candidate.get("plaintext") or "")[:200],
                "quality": candidate.get("quality"),
                "diagnostics": candidate.get("diagnostics"),
            }
            for i, candidate in enumerate(elite_candidates)
        ],
        "elite_seed_count": len({candidate.get("seed") for candidate in elite_candidates}),
        "candidates": [
            {
                "rank": i + 1,
                "epoch": candidate.epoch,
                "anneal_score": round(candidate.normalized_score, 4),
                "quality": _plaintext_quality(candidate.plaintext, candidate.key),
                "diagnostics": _automated_candidate_diagnostics(
                    candidate.plaintext,
                    language=language,
                    word_list=word_list,
                ),
                "preview": candidate.plaintext[:300],
            }
            for i, candidate in enumerate(selected_candidates)
        ],
    }
    return "native_homophonic_anneal", selected_key, selected_plaintext, step


def _is_collapsed_plaintext(plaintext: str) -> bool:
    return _plaintext_quality(plaintext, key=None)["collapsed"]


def _plaintext_quality(
    plaintext: str,
    key: dict[int, int] | None,
) -> dict[str, Any]:
    letters = [ch for ch in plaintext.upper() if "A" <= ch <= "Z"]
    if len(letters) < 50:
        return {
            "ok": True,
            "collapsed": False,
            "penalty": 0.0,
            "reasons": [],
            "letter_count": len(letters),
            "unique_letters": len(set(letters)),
            "top_letter_fraction": 0.0,
            "key_plaintext_letters": len(set(key.values())) if key else None,
        }
    counts: dict[str, int] = {}
    for letter in letters:
        counts[letter] = counts.get(letter, 0) + 1
    max_fraction = max(counts.values()) / len(letters)
    unique_letters = len(counts)
    key_plaintext_letters = len(set(key.values())) if key else None
    if len(letters) >= 350:
        min_unique = 14
        max_top_fraction = 0.22
    elif len(letters) >= 150:
        min_unique = 12
        max_top_fraction = 0.26
    else:
        min_unique = 10
        max_top_fraction = 0.32

    reasons: list[str] = []
    penalty = 0.0
    if max_fraction >= 0.35:
        reasons.append("single_letter_dominance")
        penalty += (max_fraction - 0.34) * 30.0
    if max_fraction > max_top_fraction:
        reasons.append("top_letter_too_frequent")
        penalty += (max_fraction - max_top_fraction) * 18.0
    if unique_letters < min_unique:
        reasons.append("low_plaintext_letter_diversity")
        penalty += (min_unique - unique_letters) * 0.45
    if key_plaintext_letters is not None and key_plaintext_letters < min_unique:
        reasons.append("key_maps_to_too_few_plaintext_letters")
        penalty += (min_unique - key_plaintext_letters) * 0.35

    # A lightweight monogram chi-square catches ETAOIN-ish soup that has enough
    # distinct letters to evade simple collapse checks.
    expected_total = sum(homophonic.ENGLISH_FREQUENCIES.values())
    chi = 0.0
    for letter, expected_pct in homophonic.ENGLISH_FREQUENCIES.items():
        expected = expected_pct / expected_total
        observed = counts.get(letter, 0) / len(letters)
        chi += ((observed - expected) ** 2) / max(expected, 1e-9)
    chi_per_letter = chi / 26
    if chi_per_letter > 0.05:
        reasons.append("poor_monogram_shape")
        penalty += min(2.0, (chi_per_letter - 0.05) * 12.0)

    collapsed = bool(reasons)
    return {
        "ok": not collapsed,
        "collapsed": collapsed,
        "penalty": round(penalty, 4),
        "reasons": reasons,
        "letter_count": len(letters),
        "unique_letters": unique_letters,
        "top_letter_fraction": round(max_fraction, 4),
        "key_plaintext_letters": key_plaintext_letters,
        "monogram_chi_per_letter": round(chi_per_letter, 4),
    }


def _run_substitution(
    cipher_text: CipherText,
    language: str,
) -> tuple[str, dict[int, int], str, dict[str, Any]]:
    if language == "en":
        return _run_substitution_continuous(cipher_text, language)

    session = Session()
    session.set_cipher_text(cipher_text)
    pt_alpha = Alphabet.standard_english()
    session.plaintext_alphabet = pt_alpha
    initial_key = _frequency_key(cipher_text, language, session.plaintext_alphabet)
    if not initial_key:
        raise ValueError("could not build initial key")

    words = _word_list(language)
    quadgrams = ngram.to_log_probs(ngram.build_ngram_counts(words, 4))
    best_score = float("-inf")
    best_key: dict[int, int] = {}
    best_decryption = ""
    restarts = 8
    started = time.time()

    rng = random.Random(0)
    for restart in range(restarts):
        if restart == 0:
            key = dict(initial_key)
        else:
            key = dict(initial_key)
            pt_ids = list(range(session.plaintext_alphabet.size))
            mutable = sorted(key)
            rng.shuffle(pt_ids)
            for idx, ct_id in enumerate(mutable):
                key[ct_id] = pt_ids[idx % len(pt_ids)]
        session.set_full_key(key)

        def score_fn() -> float:
            return ngram.normalized_ngram_score(session.apply_key(), quadgrams, n=4)

        score = simulated_anneal(
            session,
            score_fn,
            max_steps=5000,
            t_start=1.0,
            t_end=0.005,
            swap_fraction=0.55,
        )
        if score > best_score:
            best_score = score
            best_key = dict(session.key)
            best_decryption = session.apply_key()

    # --- Post-processing: key-consistent dictionary repair + anchor re-anneal ---
    id_to_letter = {i: pt_alpha.symbol_for(i).upper() for i in range(pt_alpha.size)}
    letter_to_id = {v: k for k, v in id_to_letter.items()}

    key_repair_info = _run_key_consistent_repair(
        cipher_text=cipher_text,
        key=best_key,
        language=language,
        word_list=words,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        score_fn=_quadgram_key_score_fn(
            list(cipher_text.tokens), id_to_letter, quadgrams
        ),
    )
    anchor_refine_info: dict[str, Any] | None = None
    if key_repair_info["applied"]:
        best_key = key_repair_info["key"]
        session.set_full_key(best_key)

        def score_fn_repaired() -> float:
            return ngram.normalized_ngram_score(session.apply_key(), quadgrams, n=4)

        repaired_base_score = score_fn_repaired()
        best_score_after_repair = repaired_base_score

        # Anchor-constrained re-anneal: freeze all symbols in words that now
        # decode to dictionary entries and re-search the remainder.
        anchor_refine_info = _maybe_anchor_refine_substitution(
            cipher_text=cipher_text,
            session=session,
            key=best_key,
            language=language,
            id_to_letter=id_to_letter,
            score_fn=score_fn_repaired,
        )
        if anchor_refine_info["applied"]:
            best_key = anchor_refine_info["key"]
            session.set_full_key(best_key)
            best_score_after_repair = anchor_refine_info["score"]

        best_decryption = session.apply_key()
        if best_score_after_repair > best_score:
            best_score = best_score_after_repair

    step = {
        "name": "search_anneal",
        "solver": "native_substitution_anneal",
        "score": round(best_score, 4),
        "restarts": restarts,
        "elapsed_seconds": round(time.time() - started, 3),
        "key_repair": key_repair_info,
    }
    if anchor_refine_info is not None:
        step["anchor_refine"] = anchor_refine_info
    return "native_substitution_anneal", best_key, best_decryption, step


def _run_substitution_continuous(
    cipher_text: CipherText,
    language: str,
) -> tuple[str, dict[int, int], str, dict[str, Any]]:
    pt_alpha = _plaintext_alphabet(language)
    plaintext_ids = list(range(pt_alpha.size))
    id_to_letter = {i: pt_alpha.symbol_for(i).upper() for i in plaintext_ids}
    letter_to_id = {v: k for k, v in id_to_letter.items()}
    word_list = _word_list(language)
    model, model_note = _homophonic_model(language, word_list)
    initial_key = _frequency_key(cipher_text, language, pt_alpha)
    result = homophonic.substitution_simulated_anneal(
        tokens=list(cipher_text.tokens),
        plaintext_ids=plaintext_ids,
        id_to_letter=id_to_letter,
        model=model,
        initial_key=initial_key,
        epochs=12,
        sampler_iterations=7000,
        distribution_weight=1.0,
        seed=0,
        top_n=3,
    )
    final_key = result.key

    # Key-consistent dictionary repair (word-boundary only).
    key_repair_info = _run_key_consistent_repair(
        cipher_text=cipher_text,
        key=final_key,
        language=language,
        word_list=word_list,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
    )
    if key_repair_info["applied"]:
        final_key = key_repair_info["key"]

    step = {
        "name": "search_substitution_continuous_anneal",
        "solver": "native_substitution_continuous_anneal",
        "model_source": model.source,
        "model_note": model_note,
        "anneal_score": round(result.normalized_score, 4),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "epochs": result.epochs,
        "sampler_iterations": result.sampler_iterations,
        "key_repair": key_repair_info,
        "candidates": [
            {
                "rank": i + 1,
                "anneal_score": round(candidate.normalized_score, 4),
                "preview": candidate.plaintext[:300],
            }
            for i, candidate in enumerate(result.candidates)
        ],
    }
    session = Session()
    session.set_cipher_text(cipher_text)
    session.plaintext_alphabet = pt_alpha
    session.set_full_key(final_key)
    decryption = session.apply_key()
    return "native_substitution_continuous_anneal", final_key, decryption, step


def _frequency_key(
    cipher_text: CipherText,
    language: str,
    pt_alpha: Alphabet,
) -> dict[int, int]:
    from agent.prompts import FREQUENCY_ORDERS
    from analysis import frequency

    order = FREQUENCY_ORDERS.get(language, FREQUENCY_ORDERS.get("en", "ETAOINSHRDLCUMWFGYPBVKJXQZ"))
    freq_data = frequency.sorted_frequency(cipher_text.tokens)
    key: dict[int, int] = {}
    used_pt: set[int] = set()
    for idx, (ct_id, _count) in enumerate(freq_data):
        if idx < len(order) and pt_alpha.has_symbol(order[idx]):
            pt_id = pt_alpha.id_for(order[idx])
            key[ct_id] = pt_id
            used_pt.add(pt_id)
    fallback = [i for i in range(pt_alpha.size) if i not in used_pt]
    for ct_id, _count in freq_data:
        if ct_id not in key and fallback:
            key[ct_id] = fallback.pop(0)
    return key


def _plaintext_alphabet(language: str) -> Alphabet:
    # Current benchmark languages all normalize into a Latin-script 26-letter
    # alphabet. Keep the helper explicit so future language-specific alphabets
    # or diacritic-aware variants have one place to plug in.
    return Alphabet.standard_english()


def _word_list(language: str) -> list[str]:
    path = dictionary.get_dictionary_path(language)
    return pattern.load_word_list(path) if path else []


def _homophonic_model(
    language: str,
    word_list: list[str],
) -> tuple[homophonic.ContinuousNGramModel, str]:
    candidate = _default_homophonic_model_path() if language == "en" else None
    if candidate and candidate.exists():
        return (
            homophonic.load_zenith_csv_model(candidate, order=5, max_ngrams=3_000_000),
            "Using local Zenith continuous n-gram model.",
        )
    return (
        homophonic.build_continuous_ngram_model(word_list, order=5),
        "Using language word-list fallback continuous model.",
    )


def _default_homophonic_model_path() -> Path | None:
    env_path = os.environ.get("DECIPHER_HOMOPHONIC_MODEL")
    if env_path:
        return Path(env_path).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in [
        repo_root / "other_tools" / "zenith-2026.2" / "zenith-model.csv",
        repo_root / "other_tools" / "zenith" / "zenith-model.csv",
    ]:
        if candidate.exists():
            return candidate
    return None


def _zenith_native_model_path(language: str = "en") -> Path | None:
    """Locate the binary model file for the ``zenith_native`` profile."""
    lang_key = (language or "en").strip().lower()
    env_path = os.environ.get(f"DECIPHER_NGRAM_MODEL_{lang_key.upper()}")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.exists() else None
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "models" / f"ngram5_{lang_key}.bin"
    if candidate.exists():
        return candidate
    if lang_key != "en":
        return None
    env_path = os.environ.get("DECIPHER_ZENITH_BINARY_MODEL")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.exists() else None
    for candidate in [
        repo_root / "other_tools" / "zenith-2026.2" / "zenith-model.array.bin",
        repo_root / "other_tools" / "zenith" / "zenith-model.array.bin",
    ]:
        if candidate.exists():
            return candidate
    return None


def _run_homophonic_zenith_native(
    cipher_text: CipherText,
    language: str,
    budget: str,
    ground_truth: str | None,
    pt_alpha: Any,
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    short_homophonic: bool,
    budget_params: dict[str, Any],
    started: float,
) -> tuple[str, dict[int, int], str, dict[str, Any]]:
    """Run the Zenith-parity homophonic solver (``zenith_native`` score profile).

    Uses the Zenith-derived solver implemented in ``analysis.zenith_solver``:
    Shannon-entropy counterweight plus un-normalized acceptance criterion.
    Falls back gracefully if the binary model file is not present.
    """
    from analysis.zenith_solver import load_zenith_binary_model, zenith_solve

    bin_path = _zenith_native_model_path(language)
    if bin_path is None:
        raise FileNotFoundError(
            "zenith_native profile requires a binary language model file. "
            f"For language={language!r}, set DECIPHER_NGRAM_MODEL_{language.upper()} "
            f"or place a model at models/ngram5_{language}.bin. "
            "English also supports the proprietary Zenith fallback via "
            "DECIPHER_ZENITH_BINARY_MODEL or "
            "other_tools/zenith-2026.2/zenith-model.array.bin."
        )

    seeds = budget_params["seeds"]
    epochs = budget_params["epochs"]
    sampler_iterations = budget_params["sampler_iterations"]
    parallel_seed_workers = _homophonic_parallel_seed_workers(len(seeds))
    engine = _zenith_native_engine()

    best_result = None
    best_score = float("-inf")
    attempts = []
    word_list = _word_list(language)

    seed_results: list[tuple[int, Any]] = []
    if parallel_seed_workers > 1 and len(seeds) > 1:
        max_workers = min(parallel_seed_workers, len(seeds))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _zenith_native_seed_worker,
                    tokens=list(cipher_text.tokens),
                    plaintext_ids=plaintext_ids,
                    id_to_letter=id_to_letter,
                    letter_to_id=letter_to_id,
                    model_path=str(bin_path),
                    epochs=epochs,
                    sampler_iterations=sampler_iterations,
                    seed=seed,
                    engine=engine,
                ): seed
                for seed in seeds
            }
            for future in concurrent.futures.as_completed(future_map):
                seed = future_map[future]
                seed_results.append((seed, future.result()))
        seed_results.sort(key=lambda item: seeds.index(item[0]))
    else:
        if engine == "rust":
            from analysis.zenith_fast import zenith_solve_fast

            for seed in seeds:
                seed_results.append((
                    seed,
                    zenith_solve_fast(
                        tokens=list(cipher_text.tokens),
                        plaintext_ids=plaintext_ids,
                        id_to_letter=id_to_letter,
                        model_path=bin_path,
                        epochs=epochs,
                        sampler_iterations=sampler_iterations,
                        seed=seed,
                        top_n=3,
                    ),
                ))
        else:
            model = load_zenith_binary_model(bin_path)
            for seed in seeds:
                seed_results.append((
                    seed,
                    zenith_solve(
                        tokens=list(cipher_text.tokens),
                        plaintext_ids=plaintext_ids,
                        id_to_letter=id_to_letter,
                        letter_to_id=letter_to_id,
                        model=model,
                        epochs=epochs,
                        sampler_iterations=sampler_iterations,
                        seed=seed,
                        top_n=3,
                    ),
                ))

    for seed, candidate in seed_results:
        quality = _plaintext_quality(candidate.plaintext, candidate.key)
        diagnostics = _automated_candidate_diagnostics(
            candidate.plaintext,
            language=language,
            word_list=word_list,
        )
        attempts.append({
            "seed": seed,
            "anneal_score": round(candidate.normalized_score, 4),
            "quality": quality,
            "diagnostics": diagnostics,
            "preview": candidate.plaintext[:120],
        })
        if candidate.normalized_score > best_score:
            best_score = candidate.normalized_score
            best_result = candidate
            if quality["ok"] and not short_homophonic:
                break

    if best_result is None:
        raise ValueError("zenith_native anneal produced no result")

    selected_plaintext = best_result.plaintext
    selected_key = best_result.key

    key_repair_info = _maybe_repair_zenith_native_key(
        cipher_text=cipher_text,
        bin_path=bin_path,
        key=selected_key,
        plaintext=selected_plaintext,
        language=language,
        word_list=word_list,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
    )
    if key_repair_info["applied"]:
        selected_key = key_repair_info["key"]
        selected_plaintext = key_repair_info["plaintext"]

    anchor_refine_info = _maybe_anchor_refine_zenith_native(
        cipher_text=cipher_text,
        bin_path=bin_path,
        key=selected_key,
        plaintext=selected_plaintext,
        anneal_score=best_score,
        language=language,
        word_list=word_list,
        plaintext_ids=plaintext_ids,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        budget_params=budget_params,
    )
    if anchor_refine_info["applied"]:
        selected_key = anchor_refine_info["key"]
        selected_plaintext = anchor_refine_info["plaintext"]
        best_score = anchor_refine_info["score"]

    polish_info = _maybe_polish_zenith_native_plaintext(
        selected_plaintext,
        language=language,
        word_list=word_list,
    )
    if polish_info["applied"]:
        selected_plaintext = polish_info["plaintext"]

    step: dict[str, Any] = {
        "name": "search_homophonic_anneal",
        "solver": "zenith_native",
        "model_source": str(bin_path),
        "model_note": "zenith_binary",
        "engine": engine,
        "homophonic_budget": budget,
        "budget_params": budget_params,
        "homophonic_refinement": "none",
        "selection_profile": "anneal_quality",
        "score_profile": "zenith_native",
        "score_formula": "zenith_entropy",
        "window_step": 2,
        "anneal_score": round(best_score, 4),
        "selection_score": round(best_score, 4),
        "quality": _plaintext_quality(selected_plaintext, selected_key),
        "diagnostics": _automated_candidate_diagnostics(
            selected_plaintext,
            language=language,
            word_list=word_list,
        ),
        "key_repair": key_repair_info,
        "anchor_refine": anchor_refine_info,
        "postprocess": polish_info,
        "elapsed_seconds": round(time.time() - started, 3),
        "epochs": best_result.epochs,
        "sampler_iterations": best_result.sampler_iterations,
        "parallel_seed_workers": parallel_seed_workers,
        "seed_attempts": attempts,
    }
    return "zenith_native", selected_key, selected_plaintext, step


_KEY_REPAIR_DISABLED_VALUES = {"0", "false", "no", "off"}


def _homophonic_key_repair_enabled() -> bool:
    """Key-consistent dictionary repair is on by default.

    Set ``DECIPHER_HOMOPHONIC_KEY_REPAIR=0`` to disable for bisection.
    """
    raw = os.environ.get("DECIPHER_HOMOPHONIC_KEY_REPAIR", "").strip().lower()
    if raw in _KEY_REPAIR_DISABLED_VALUES:
        return False
    return True


def _run_key_consistent_repair(
    *,
    cipher_text: CipherText,
    key: dict[int, int],
    language: str,
    word_list: list[str],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    score_fn: Callable[[dict[int, int]], float] | None = None,
    max_score_drop: float = 0.0,
    min_word_len: int = 5,
) -> dict[str, Any]:
    """Run the key-consistent dictionary repair and return a telemetry dict.

    Shared by both the zenith_native homophonic path and the substitution
    paths. Does **not** render a plaintext — callers decide how they want
    to present the repaired key (no-boundary flat string, ``|``-separated,
    etc.) because the two families use different conventions.

    ``score_fn`` is a language-model guard: when provided, a repair is
    rejected if its score drops by more than ``max_score_drop`` relative to
    the current key. This prevents greedy short-word "fixes" (e.g. PELLA →
    BELLA) that improve dictionary hit count locally but destroy the global
    n-gram structure — the failure mode observed on Borg 0109v when the
    repair ran dict-only. ``min_word_len`` defaults to 5 for the same
    reason: short Latin and English words are too dense to yield
    confident one-edit candidates.
    """
    info: dict[str, Any] = {
        "enabled": _homophonic_key_repair_enabled(),
        "applied": False,
        "word_boundary_count": len(cipher_text.words),
        "rounds": 0,
        "corrections": [],
        "before_dict_hits": None,
        "after_dict_hits": None,
        "min_word_len": min_word_len,
        "score_fn_guard": score_fn is not None,
        "key": dict(key),
    }

    if not info["enabled"]:
        info["reason"] = "disabled"
        return info

    # Single word group = the solver is running in true no-boundary mode; the
    # existing text-polish step already handles that case.
    if len(cipher_text.words) <= 1:
        info["reason"] = "no_word_boundaries"
        return info

    path = dictionary.get_dictionary_path(language)
    word_set = dictionary.load_word_set(path) if path else set()
    if not word_set:
        info["reason"] = "no_dictionary_available"
        return info

    freq_rank = {word.upper(): idx for idx, word in enumerate(word_list)} if word_list else None

    result = repair_key_with_dictionary(
        cipher_words=cipher_text.words,
        key=key,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        word_set=word_set,
        freq_rank=freq_rank,
        min_word_len=min_word_len,
        score_fn=score_fn,
        max_score_drop=max_score_drop,
    )

    info["rounds"] = result.rounds
    info["corrections"] = result.corrections
    info["before_dict_hits"] = result.before_hits
    info["after_dict_hits"] = result.after_hits
    info["before_total_words"] = result.before_words
    info["after_total_words"] = result.after_words
    info["before_preview"] = result.before_plaintext[:200]
    info["after_preview"] = result.after_plaintext[:200]

    if not result.applied:
        info["reason"] = result.reason or "no_improvement"
        return info

    info["applied"] = True
    info["key"] = result.repaired_key
    return info


def _zenith_model_score_fn(
    cipher_tokens: list[int],
    id_to_letter: dict[int, str],
    bin_path: Path,
) -> Callable[[dict[int, int]], float] | None:
    """Build a key → 5-gram log-prob sum score_fn using the zenith binary model.

    Returns ``None`` if the model cannot be loaded. The score is a plain sum
    of 5-gram log-probs across the decoded plaintext — sufficient as a
    monotonic guard for detecting whether a candidate repair makes the
    language-model fit worse. Not equivalent to zenith_solve's Shannon-
    entropy-normalized objective, but the repair only needs a sign check.
    """
    try:
        from analysis.zenith_solver import load_zenith_binary_model

        model = load_zenith_binary_model(bin_path)
    except Exception:  # noqa: BLE001 — no model means no guard; caller decides
        return None

    tokens = list(cipher_tokens)

    def score(candidate_key: dict[int, int]) -> float:
        letters: list[int] = []
        for tok in tokens:
            pt_id = candidate_key.get(tok)
            if pt_id is None:
                return float("-inf")
            letter = id_to_letter.get(pt_id, "A")
            code = ord(letter.lower()) - 97
            if not (0 <= code < 26):
                return float("-inf")
            letters.append(code)
        if len(letters) < 5:
            return 0.0
        total = 0.0
        for i in range(len(letters) - 4):
            total += model.lookup_lo(
                letters[i], letters[i + 1], letters[i + 2], letters[i + 3], letters[i + 4]
            )
        return total

    return score


def _quadgram_key_score_fn(
    cipher_tokens: list[int],
    id_to_letter: dict[int, str],
    quadgrams: dict[str, float],
) -> Callable[[dict[int, int]], float]:
    """Build a key → normalized-quadgram score_fn for the substitution path."""
    tokens = list(cipher_tokens)

    def score(candidate_key: dict[int, int]) -> float:
        letters: list[str] = []
        for tok in tokens:
            pt_id = candidate_key.get(tok)
            if pt_id is None:
                letters.append("?")
                continue
            letters.append(id_to_letter.get(pt_id, "?"))
        text = "".join(letters)
        return ngram.normalized_ngram_score(text, quadgrams, n=4)

    return score


def _maybe_repair_zenith_native_key(
    *,
    cipher_text: CipherText,
    bin_path: Path,
    key: dict[int, int],
    plaintext: str,
    language: str,
    word_list: list[str],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    min_word_len: int = 5,
) -> dict[str, Any]:
    """Apply key-consistent dictionary repair after the zenith SA converges.

    Only meaningful when the ciphertext preserves word boundaries. Returns
    a telemetry dict; when ``applied`` is True the caller should swap in the
    new ``key`` and ``plaintext``.

    The repair is guarded by a 5-gram log-prob score_fn: a candidate fix is
    rejected if the zenith model score drops relative to the current key.
    This closes the loop that caused greedy short-word false positives to
    regress Borg 0109v on the dict-only version.
    """
    score_fn = _zenith_model_score_fn(
        list(cipher_text.tokens), id_to_letter, bin_path
    )
    info = _run_key_consistent_repair(
        cipher_text=cipher_text,
        key=key,
        language=language,
        word_list=word_list,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        score_fn=score_fn,
        min_word_len=min_word_len,
    )
    info["plaintext"] = plaintext
    if not info["applied"]:
        return info
    repaired_plaintext = "".join(
        id_to_letter.get(info["key"].get(tok, -1), "?")
        for tok in cipher_text.tokens
    ).upper()
    info["plaintext"] = repaired_plaintext
    return info


_ANCHOR_REFINE_DISABLED_VALUES = {"0", "false", "no", "off"}


def _homophonic_anchor_refine_enabled() -> bool:
    """Anchor-constrained re-anneal is on by default.

    Set ``DECIPHER_HOMOPHONIC_ANCHOR_REFINE=0`` to disable for bisection.
    """
    raw = os.environ.get("DECIPHER_HOMOPHONIC_ANCHOR_REFINE", "").strip().lower()
    if raw in _ANCHOR_REFINE_DISABLED_VALUES:
        return False
    return True


def _collect_anchor_symbols(
    cipher_words: list[list[int]],
    key: dict[int, int],
    id_to_letter: dict[int, str],
    word_set: set[str],
    *,
    min_word_len: int = 3,
) -> tuple[set[int], list[str]]:
    """Return cipher symbols whose decoded word appears in ``word_set``.

    Longer anchor words give stronger evidence; we require length >= 3 to
    keep function words like "A"/"IN" from over-constraining the search
    basin on short filler matches.
    """
    anchors: set[int] = set()
    anchor_words: list[str] = []
    for tokens in cipher_words:
        if len(tokens) < min_word_len:
            continue
        letters = "".join(
            id_to_letter.get(key.get(t, -1), "?") for t in tokens
        ).upper()
        if "?" in letters:
            continue
        if letters in word_set:
            anchors.update(tokens)
            anchor_words.append(letters)
    return anchors, anchor_words


def _maybe_anchor_refine_zenith_native(
    *,
    cipher_text: CipherText,
    bin_path: Path,
    key: dict[int, int],
    plaintext: str,
    anneal_score: float,
    language: str,
    word_list: list[str],
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    budget_params: dict[str, Any],
) -> dict[str, Any]:
    """Short zenith_solve pass with anchor symbols frozen, restarted from ``key``.

    Only meaningful when the ciphertext preserves word boundaries, because
    anchor extraction relies on whole-word dictionary matches. The refinement
    is gated on a strict score improvement so a degenerate anneal cannot
    displace the primary result.
    """
    info: dict[str, Any] = {
        "enabled": _homophonic_anchor_refine_enabled(),
        "applied": False,
        "anchor_symbol_count": 0,
        "anchor_words": [],
        "base_score": round(anneal_score, 4),
        "refined_score": None,
        "key": dict(key),
        "plaintext": plaintext,
        "score": anneal_score,
    }

    if not info["enabled"]:
        info["reason"] = "disabled"
        return info
    if len(cipher_text.words) <= 1:
        info["reason"] = "no_word_boundaries"
        return info

    path = dictionary.get_dictionary_path(language)
    word_set = dictionary.load_word_set(path) if path else set()
    if not word_set:
        info["reason"] = "no_dictionary_available"
        return info

    anchors, anchor_words = _collect_anchor_symbols(
        cipher_text.words,
        key,
        id_to_letter,
        word_set,
    )
    info["anchor_symbol_count"] = len(anchors)
    info["anchor_words"] = sorted(set(anchor_words))[:20]

    # Need at least a few anchors to be worth the extra pass, and we must
    # leave some mutable symbols to actually explore.
    total_symbols = len(set(cipher_text.tokens))
    if len(anchors) < 3 or (total_symbols - len(anchors)) < 3:
        info["reason"] = "insufficient_anchors"
        return info

    try:
        from analysis.zenith_solver import load_zenith_binary_model, zenith_solve
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"import_failed:{exc}"
        return info

    try:
        model = load_zenith_binary_model(bin_path)
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"model_load_failed:{exc}"
        return info

    base_epochs = max(1, int(budget_params.get("epochs", 3)))
    base_iters = max(500, int(budget_params.get("sampler_iterations", 5000)))
    refine_epochs = max(1, min(3, base_epochs))
    refine_iters = max(500, base_iters // 2)

    try:
        refined = zenith_solve(
            tokens=list(cipher_text.tokens),
            plaintext_ids=plaintext_ids,
            id_to_letter=id_to_letter,
            letter_to_id=letter_to_id,
            model=model,
            initial_key=dict(key),
            fixed_cipher_ids=anchors,
            epochs=refine_epochs,
            sampler_iterations=refine_iters,
            seed=budget_params.get("seeds", [1])[0] + 10_000,
            top_n=1,
        )
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"refine_failed:{exc}"
        return info

    info["refined_score"] = round(refined.normalized_score, 4)
    info["refine_epochs"] = refine_epochs
    info["refine_iterations"] = refine_iters

    improvement_eps = 1e-4
    if refined.normalized_score <= anneal_score + improvement_eps:
        info["reason"] = "no_score_improvement"
        return info

    info["applied"] = True
    info["key"] = refined.key
    info["plaintext"] = refined.plaintext
    info["score"] = refined.normalized_score
    return info


def _maybe_anchor_refine_substitution(
    *,
    cipher_text: CipherText,
    session: Session,
    key: dict[int, int],
    language: str,
    id_to_letter: dict[int, str],
    score_fn: Callable[[], float],
    max_steps: int = 3000,
) -> dict[str, Any]:
    """Short hill-climb with dictionary anchors frozen, restarted from ``key``.

    Used by the ``_run_substitution`` non-English path after the key-consistent
    repair stage. Returns the same shape of telemetry dict as the zenith
    variant, including ``key``/``plaintext``/``score`` fields the caller
    can splice in when ``applied`` is True.
    """
    info: dict[str, Any] = {
        "enabled": _homophonic_anchor_refine_enabled(),
        "applied": False,
        "anchor_symbol_count": 0,
        "anchor_words": [],
        "base_score": None,
        "refined_score": None,
        "key": dict(key),
        "score": None,
    }

    if not info["enabled"]:
        info["reason"] = "disabled"
        return info
    if len(cipher_text.words) <= 1:
        info["reason"] = "no_word_boundaries"
        return info

    path = dictionary.get_dictionary_path(language)
    word_set = dictionary.load_word_set(path) if path else set()
    if not word_set:
        info["reason"] = "no_dictionary_available"
        return info

    anchors, anchor_words = _collect_anchor_symbols(
        cipher_text.words,
        key,
        id_to_letter,
        word_set,
    )
    info["anchor_symbol_count"] = len(anchors)
    info["anchor_words"] = sorted(set(anchor_words))[:20]

    total_symbols = len(set(cipher_text.tokens))
    if len(anchors) < 3 or (total_symbols - len(anchors)) < 3:
        info["reason"] = "insufficient_anchors"
        return info

    base_score = score_fn()
    info["base_score"] = round(base_score, 4)

    # Run a fresh, shorter SA pass with anchors frozen.
    session.set_full_key(dict(key))
    refined_score = simulated_anneal(
        session,
        score_fn,
        max_steps=max_steps,
        t_start=0.5,
        t_end=0.005,
        swap_fraction=0.55,
        fixed_cipher_ids=anchors,
    )
    info["refined_score"] = round(refined_score, 4)

    if refined_score <= base_score + 1e-4:
        # Revert; nothing better than where we started.
        session.set_full_key(dict(key))
        info["reason"] = "no_score_improvement"
        return info

    info["applied"] = True
    info["key"] = dict(session.key)
    info["score"] = refined_score
    return info


def _homophonic_polish_enabled() -> bool:
    return os.environ.get("DECIPHER_HOMOPHONIC_POLISH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _maybe_polish_zenith_native_plaintext(
    plaintext: str,
    *,
    language: str,
    word_list: list[str],
) -> dict[str, Any]:
    """Optionally segment and lightly repair no-boundary zenith-native output.

    This is intentionally conservative and currently opt-in via
    ``DECIPHER_HOMOPHONIC_POLISH`` so we can evaluate it without changing the
    frontier baseline. The repair loop operates on segmented words rather than
    mutating the underlying key, so the artifact records it as a postprocess
    step and keeps the original anneal telemetry intact.
    """
    info: dict[str, Any] = {
        "enabled": _homophonic_polish_enabled(),
        "applied": False,
        "mode": "segment_one_edit_local",
        "rounds": 0,
        "corrections": [],
        "plaintext": plaintext,
    }
    if not info["enabled"]:
        return info
    if any(ch.isspace() for ch in plaintext):
        info["reason"] = "plaintext_already_segmented"
        return info

    path = dictionary.get_dictionary_path(language)
    word_set = dictionary.load_word_set(path) if path else set()
    if not word_set:
        info["reason"] = "no_dictionary_available"
        return info

    freq_rank = {word.upper(): idx for idx, word in enumerate(word_list)}
    alpha_only = "".join(ch for ch in plaintext.upper() if "A" <= ch <= "Z")
    if not alpha_only:
        info["reason"] = "no_alpha_text"
        return info
    repair = repair_no_boundary_text(alpha_only, word_set, freq_rank=freq_rank)
    info["rounds"] = repair.rounds
    info["corrections"] = repair.corrections
    info["before"] = {
        "dict_rate": round(repair.before.dict_rate, 4),
        "segmentation_cost": round(repair.before.cost, 3),
        "pseudo_word_count": len(repair.before.pseudo_words),
        "segmented_preview": repair.before.segmented[:160],
    }
    info["after"] = {
        "dict_rate": round(repair.after.dict_rate, 4),
        "segmentation_cost": round(repair.after.cost, 3),
        "pseudo_word_count": len(repair.after.pseudo_words),
        "segmented_preview": repair.after.segmented[:160],
    }
    if not repair.applied:
        info["reason"] = repair.reason
        return info

    info["applied"] = True
    info["plaintext"] = repair.repaired_text
    info["key_consistent_with_output"] = False
    return info


def _homophonic_score_profile_for(default_profile: str) -> str:
    return (
        os.environ.get("DECIPHER_HOMOPHONIC_SCORE_PROFILE", default_profile)
        .strip()
        .lower()
        or default_profile
    )


def _homophonic_score_profile(solver_profile: str = "zenith_native") -> str:
    normalized = (solver_profile or "zenith_native").strip().lower()
    if normalized in {"zenith_native", "default"}:
        return _homophonic_score_profile_for("zenith_native")
    if normalized == "legacy":
        return _homophonic_score_profile_for("balanced")
    raise ValueError(
        "unsupported homophonic solver profile "
        f"'{solver_profile}' (expected one of: zenith_native, legacy)"
    )


def _homophonic_selection_profile() -> str:
    return (
        os.environ.get("DECIPHER_HOMOPHONIC_SELECTION_PROFILE", "anneal_quality")
        .strip()
        .lower()
        or "anneal_quality"
    )


def _homophonic_move_profile() -> str:
    return (
        os.environ.get("DECIPHER_HOMOPHONIC_MOVE_PROFILE", "single_symbol")
        .strip()
        .lower()
        or "single_symbol"
    )


def _homophonic_use_early_stop() -> bool:
    value = os.environ.get("DECIPHER_HOMOPHONIC_EARLY_STOP", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _homophonic_search_profile() -> str:
    return (
        os.environ.get("DECIPHER_HOMOPHONIC_SEARCH_PROFILE", "full")
        .strip()
        .lower()
        or "full"
    )


def _homophonic_repair_profile() -> str:
    return (
        os.environ.get("DECIPHER_HOMOPHONIC_REPAIR_PROFILE", "full")
        .strip()
        .lower()
        or "full"
    )


def _homophonic_parallel_seed_workers(seed_count: int | None = None) -> int:
    raw = os.environ.get("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS")
    if raw is None:
        cpu_count = os.cpu_count() or 1
        value = max(1, cpu_count - 1)
        if seed_count is not None:
            value = min(value, max(1, seed_count))
        return value
    raw = raw.strip() or "1"
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "DECIPHER_HOMOPHONIC_PARALLEL_SEEDS must be an integer >= 1"
        ) from exc
    value = max(1, value)
    if seed_count is not None:
        value = min(value, max(1, seed_count))
    return value


def _zenith_native_engine() -> str:
    raw = os.environ.get("DECIPHER_ZENITH_NATIVE_ENGINE", "rust").strip().lower()
    if raw in {"py", "python", "reference"}:
        return "python"
    if raw in {"rs", "rust", "fast"}:
        return "rust"
    raise ValueError(
        "DECIPHER_ZENITH_NATIVE_ENGINE must be one of: python, rust"
    )


def _transform_rank_engine() -> str:
    raw = os.environ.get("DECIPHER_TRANSFORM_RANK_ENGINE", "rust").strip().lower()
    if raw in {"py", "python", "reference"}:
        return "python"
    if raw in {"rs", "rust", "fast"}:
        return "rust"
    raise ValueError(
        "DECIPHER_TRANSFORM_RANK_ENGINE must be one of: python, rust"
    )


def _transform_rank_threads() -> int:
    raw = os.environ.get("DECIPHER_TRANSFORM_RANK_THREADS", "0").strip() or "0"
    try:
        return max(0, int(raw))
    except ValueError as exc:
        raise ValueError("DECIPHER_TRANSFORM_RANK_THREADS must be an integer >= 0") from exc


def _zenith_native_seed_worker(
    *,
    tokens: list[int],
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    model_path: str,
    epochs: int,
    sampler_iterations: int,
    seed: int,
    engine: str = "python",
):
    if engine == "rust":
        from analysis.zenith_fast import zenith_solve_fast

        return zenith_solve_fast(
            tokens=tokens,
            plaintext_ids=plaintext_ids,
            id_to_letter=id_to_letter,
            model_path=model_path,
            epochs=epochs,
            sampler_iterations=sampler_iterations,
            seed=seed,
            top_n=3,
        )

    from analysis.zenith_solver import load_zenith_binary_model, zenith_solve

    model = load_zenith_binary_model(model_path)
    return zenith_solve(
        tokens=tokens,
        plaintext_ids=plaintext_ids,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        model=model,
        epochs=epochs,
        sampler_iterations=sampler_iterations,
        seed=seed,
        top_n=3,
    )


def _score_config_weights(config: dict[str, Any]) -> dict[str, float]:
    return {
        "distribution_weight": config["distribution_weight"],
        "diversity_weight": config["diversity_weight"],
        "ioc_weight": config["ioc_weight"],
    }


def _homophonic_score_config(profile: str, short_homophonic: bool) -> dict[str, Any]:
    profiles = {
        "balanced": {
            "distribution_weight": 5.0,
            "diversity_weight": 3.0 if short_homophonic else 1.5,
            "ioc_weight": 0.0,
            "score_formula": "additive",
            "window_step": 1,
        },
        "ioc_ngram": {
            "distribution_weight": 0.0,
            "diversity_weight": 0.0,
            "ioc_weight": 1.0,
            "score_formula": "additive",
            "window_step": 1,
        },
        "ngram_only": {
            "distribution_weight": 0.0,
            "diversity_weight": 0.0,
            "ioc_weight": 0.0,
            "score_formula": "additive",
            "window_step": 1,
        },
        "ngram_distribution": {
            "distribution_weight": 5.0,
            "diversity_weight": 0.0,
            "ioc_weight": 0.0,
            "score_formula": "additive",
            "window_step": 1,
        },
        "zenith_like": {
            "distribution_weight": 0.0,
            "diversity_weight": 0.0,
            "ioc_weight": 1.0,
            "score_formula": "multiplicative_ioc",
            "window_step": 2,
        },
        "zenith_exact": {
            "distribution_weight": 0.0,
            "diversity_weight": 0.0,
            "ioc_weight": 1.0 / 6.0,
            "score_formula": "multiplicative_ioc",
            "window_step": 2,
        },
    }
    if profile not in profiles:
        allowed = ", ".join(sorted(profiles))
        raise ValueError(
            f"unsupported homophonic score profile '{profile}' "
            f"(expected one of: {allowed})"
        )
    return profiles[profile]


def _homophonic_score_weights(profile: str, short_homophonic: bool) -> dict[str, float]:
    return _score_config_weights(_homophonic_score_config(profile, short_homophonic))


def _make_homophonic_early_stop_hook(
    *,
    language: str,
    word_list: list[str],
    best_completed_selection_score: float,
) -> Any:
    history: list[float] = []

    def should_stop(epoch_info: dict[str, object]) -> bool:
        epoch = int(epoch_info.get("epoch", 0) or 0)
        plaintext = str(epoch_info.get("plaintext", "") or "")
        key = epoch_info.get("key")
        normalized_score = float(epoch_info.get("normalized_score", float("-inf")) or float("-inf"))
        if epoch < 3:
            history.append(normalized_score)
            return False
        quality = _plaintext_quality(plaintext, key if isinstance(key, dict) else None)
        diagnostics = _automated_candidate_diagnostics(
            plaintext,
            language=language,
            word_list=word_list,
        )
        selection_score = _score_homophonic_candidate_for_selection(
            normalized_score,
            quality,
            diagnostics,
            selection_profile="anneal_quality",
        )
        history.append(normalized_score)
        recent = history[-3:]
        progress = max(recent) - min(recent) if len(recent) >= 2 else 0.0
        dict_rate = float(diagnostics.get("dict_rate", 0.0) or 0.0)
        letter_count = max(1, int(diagnostics.get("letter_count", 0) or 0))
        segmentation_cost = float(diagnostics.get("segmentation_cost", 0.0) or 0.0)
        seg_cost_per_char = segmentation_cost / letter_count
        coherence_bad = (
            (not quality.get("ok"))
            or dict_rate < 0.5
            or seg_cost_per_char > 5.1
        )
        far_behind = selection_score < (best_completed_selection_score - 0.9)
        stalled = progress < 0.08
        return bool(coherence_bad and far_behind and stalled)

    return should_stop


def _score_homophonic_candidate_for_selection(
    anneal_score: float,
    quality: dict[str, Any],
    diagnostics: dict[str, Any],
    selection_profile: str,
) -> float:
    base = anneal_score - float(quality.get("penalty", 0.0))
    if selection_profile == "anneal_quality":
        return base
    if selection_profile == "pool_rerank_v1":
        letter_count = max(1, int(diagnostics.get("letter_count", 0) or 0))
        dict_rate = float(diagnostics.get("dict_rate", 0.0) or 0.0)
        segmentation_cost = float(diagnostics.get("segmentation_cost", 0.0) or 0.0)
        ioc_value = float(diagnostics.get("index_of_coincidence", 0.0) or 0.0)
        top_fraction = float(diagnostics.get("top_letter_fraction", 0.0) or 0.0)
        pseudo_word_count = float(diagnostics.get("pseudo_word_count", 0.0) or 0.0)
        segmentation_cost_per_char = segmentation_cost / letter_count
        ioc_penalty = abs(ioc_value - 0.0667)
        top_fraction_penalty = max(0.0, top_fraction - 0.14)
        return (
            base
            + 0.5 * dict_rate
            - 0.2 * segmentation_cost_per_char
            - 4.0 * ioc_penalty
            - 2.0 * top_fraction_penalty
            - 0.01 * pseudo_word_count
        )
    raise ValueError(
        f"unsupported homophonic selection profile '{selection_profile}' "
        "(expected one of: anneal_quality, pool_rerank_v1)"
    )


def _rank_homophonic_candidate_pool(
    candidates: list[dict[str, Any]],
    selection_profile: str,
) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate["plaintext"]
        existing = deduped.get(key)
        if existing is None or candidate["selection_score"] > existing["selection_score"]:
            deduped[key] = candidate
    return sorted(
        deduped.values(),
        key=lambda item: (item["selection_score"], item["anneal_score"]),
        reverse=True,
    )


def _candidate_family_signature(candidate: dict[str, Any]) -> tuple[int, ...]:
    key = candidate.get("key") or {}
    counts: dict[int, int] = {}
    for pt_id in key.values():
        counts[pt_id] = counts.get(pt_id, 0) + 1
    return tuple(sorted(counts.values(), reverse=True))


def _plaintext_distance_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    length = max(len(a), len(b))
    if length == 0:
        return 0.0
    mismatches = 0
    for i in range(length):
        ch_a = a[i] if i < len(a) else ""
        ch_b = b[i] if i < len(b) else ""
        if ch_a != ch_b:
            mismatches += 1
    return mismatches / length


def _select_diverse_homophonic_elites(
    candidates: list[dict[str, Any]],
    limit: int = 5,
    min_plaintext_distance: float = 0.08,
) -> list[dict[str, Any]]:
    ranked = _rank_homophonic_candidate_pool(candidates, selection_profile="anneal_quality")
    elites: list[dict[str, Any]] = []
    seen_signatures: set[tuple[int, ...]] = set()

    def can_add(candidate: dict[str, Any]) -> bool:
        signature = _candidate_family_signature(candidate)
        if signature in seen_signatures and any(
            _plaintext_distance_ratio(candidate["plaintext"], elite["plaintext"]) < min_plaintext_distance
            for elite in elites
        ):
            return False
        if any(
            _plaintext_distance_ratio(candidate["plaintext"], elite["plaintext"]) < min_plaintext_distance
            and signature == _candidate_family_signature(elite)
            for elite in elites
        ):
            return False
        return True

    seen_seeds: set[Any] = set()
    for candidate in ranked:
        seed = candidate.get("seed")
        if seed in seen_seeds:
            continue
        if not can_add(candidate):
            continue
        elites.append(candidate)
        seen_signatures.add(_candidate_family_signature(candidate))
        seen_seeds.add(seed)
        if len(elites) >= max(1, limit):
            break

    if len(elites) >= max(1, limit):
        return elites

    for candidate in ranked:
        if candidate in elites:
            continue
        if not can_add(candidate):
            continue
        elites.append(candidate)
        seen_signatures.add(_candidate_family_signature(candidate))
        if len(elites) >= max(1, limit):
            break
    return elites


def _suspicious_homophonic_symbols(
    cipher_text: CipherText,
    key: dict[int, int],
    id_to_letter: dict[int, str],
    model: homophonic.ContinuousNGramModel,
    limit: int,
    window_step: int = 1,
) -> list[int]:
    tokens = list(cipher_text.tokens)
    chars = [id_to_letter[key[token]] for token in tokens]
    window_scores = homophonic._initial_window_scores(chars, model, step=window_step)
    occurrences = homophonic._occurrence_map(tokens)
    scored: list[tuple[float, int, int, str]] = []
    for sid in sorted(set(tokens)):
        affected = homophonic._affected_windows(
            occurrences[sid],
            len(chars),
            model.order,
            step=window_step,
        )
        if not affected:
            continue
        avg_score = sum(window_scores[i] for i in affected) / len(affected)
        scored.append((avg_score, sid, len(occurrences[sid]), id_to_letter[key[sid]]))
    scored.sort()
    return [sid for _, sid, _, _ in scored[: max(1, limit)]]


def _homophonic_family_diagnostics(
    cipher_text: CipherText,
    key: dict[int, int],
    id_to_letter: dict[int, str],
    model: homophonic.ContinuousNGramModel,
    window_step: int = 1,
) -> dict[str, Any]:
    tokens = list(cipher_text.tokens)
    chars = [id_to_letter[key[token]] for token in tokens]
    window_scores = homophonic._initial_window_scores(chars, model, step=window_step)
    occurrences = homophonic._occurrence_map(tokens)
    total_letters = max(1, len(chars))
    char_counts: dict[str, int] = {}
    for ch in chars:
        char_counts[ch] = char_counts.get(ch, 0) + 1

    families: list[dict[str, Any]] = []
    expected_total = sum(homophonic.ENGLISH_FREQUENCIES.values()) or 1.0
    letter_to_symbols: dict[str, list[int]] = {}
    for sid in sorted(set(tokens)):
        letter = id_to_letter[key[sid]]
        letter_to_symbols.setdefault(letter, []).append(sid)

    for letter, symbol_ids in sorted(letter_to_symbols.items()):
        symbol_reports: list[dict[str, Any]] = []
        family_scores: list[float] = []
        occurrence_total = 0
        for sid in sorted(symbol_ids):
            affected = homophonic._affected_windows(
                occurrences[sid],
                len(chars),
                model.order,
                step=window_step,
            )
            avg_score = (
                sum(window_scores[i] for i in affected) / len(affected)
                if affected
                else float("-inf")
            )
            symbol_count = len(occurrences[sid])
            occurrence_total += symbol_count
            family_scores.append(avg_score)
            symbol_reports.append({
                "symbol_id": sid,
                "occurrence_count": symbol_count,
                "avg_window_score": round(avg_score, 4),
            })
        observed_fraction = occurrence_total / total_letters
        expected_fraction = homophonic.ENGLISH_FREQUENCIES.get(letter, 0.0) / expected_total
        avg_score = sum(family_scores) / max(1, len(family_scores))
        spread = (max(family_scores) - min(family_scores)) if len(family_scores) > 1 else 0.0
        overuse = max(0.0, observed_fraction - expected_fraction)
        underuse = max(0.0, expected_fraction - observed_fraction)
        suspicion_score = (-avg_score) + (2.5 * spread) + (8.0 * overuse) + (2.0 * underuse)
        families.append({
            "letter": letter,
            "symbol_ids": sorted(symbol_ids),
            "symbol_count": len(symbol_ids),
            "occurrence_count": occurrence_total,
            "observed_fraction": round(observed_fraction, 4),
            "expected_fraction": round(expected_fraction, 4),
            "avg_window_score": round(avg_score, 4),
            "score_spread": round(spread, 4),
            "overuse": round(overuse, 4),
            "underuse": round(underuse, 4),
            "suspicion_score": round(suspicion_score, 4),
            "symbols": symbol_reports,
        })

    families.sort(key=lambda family: family["suspicion_score"], reverse=True)
    return {
        "window_step": window_step,
        "family_count": len(families),
        "families": families,
    }


def _family_repair_gate(
    quality: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    dict_rate = float(diagnostics.get("dict_rate", 0.0) or 0.0)
    letter_count = max(1, int(diagnostics.get("letter_count", 0) or 0))
    segmentation_cost = float(diagnostics.get("segmentation_cost", 0.0) or 0.0)
    cost_per_char = segmentation_cost / letter_count
    ok = (
        bool(quality.get("ok"))
        and dict_rate >= 0.7
        and cost_per_char <= 4.8
    )
    reasons: list[str] = []
    if not quality.get("ok"):
        reasons.append("plaintext_quality_not_ok")
    if dict_rate < 0.7:
        reasons.append("dictionary_rate_too_low")
    if cost_per_char > 4.8:
        reasons.append("segmentation_cost_too_high")
    return {
        "ok": ok,
        "dict_rate": round(dict_rate, 4),
        "segmentation_cost_per_char": round(cost_per_char, 4),
        "reasons": reasons,
    }


def _symbol_letter_alternatives(
    cipher_text: CipherText,
    key: dict[int, int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    model: homophonic.ContinuousNGramModel,
    sid: int,
    candidate_letters: list[str],
    window_step: int = 1,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    tokens = list(cipher_text.tokens)
    occurrences = homophonic._occurrence_map(tokens)
    chars = [id_to_letter[key[token]] for token in tokens]
    window_scores = homophonic._initial_window_scores(chars, model, step=window_step)
    affected = homophonic._affected_windows(
        occurrences[sid],
        len(chars),
        model.order,
        step=window_step,
    )
    if not affected:
        return []
    current_letter = id_to_letter[key[sid]]
    old_sum = sum(window_scores[i] for i in affected)
    alternatives: list[dict[str, Any]] = []
    changed_positions = occurrences[sid]
    for letter in candidate_letters:
        if letter == current_letter or letter not in letter_to_id:
            continue
        original = [chars[pos] for pos in changed_positions]
        for pos in changed_positions:
            chars[pos] = letter
        new_sum = sum(
            homophonic._score_window(chars, start * window_step, model)
            if window_step > 1 else homophonic._score_window(chars, start, model)
            for start in affected
        )
        for pos, old in zip(changed_positions, original):
            chars[pos] = old
        delta = new_sum - old_sum
        alternatives.append({
            "symbol_id": sid,
            "from_letter": current_letter,
            "to_letter": letter,
            "local_delta": round(delta, 4),
        })
    alternatives.sort(key=lambda item: item["local_delta"], reverse=True)
    return alternatives[: max(1, top_k)]


def _expected_letter_fractions() -> dict[str, float]:
    total = sum(homophonic.ENGLISH_FREQUENCIES.values()) or 1.0
    return {
        letter: freq / total
        for letter, freq in homophonic.ENGLISH_FREQUENCIES.items()
    }


def _family_report_index(family_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        family["letter"]: family
        for family in (family_report.get("families") or [])
    }


def _branch_mutable_symbols(
    tokens: list[int],
    key: dict[int, int],
    id_to_letter: dict[int, str],
    letters: set[str],
    extra_symbols: set[int],
) -> list[int]:
    return sorted(
        sid
        for sid in sorted(set(tokens))
        if id_to_letter[key[sid]] in letters or sid in extra_symbols
    )


def _branch_score(
    *,
    local_delta: float,
    source_letter: str,
    target_letter: str,
    family_index: dict[str, dict[str, Any]],
) -> float:
    source = family_index.get(source_letter, {})
    target = family_index.get(target_letter, {})
    source_overuse = float(source.get("overuse", 0.0) or 0.0)
    source_spread = float(source.get("score_spread", 0.0) or 0.0)
    target_underuse = max(
        0.0,
        _expected_letter_fractions().get(target_letter, 0.0)
        - float(target.get("observed_fraction", 0.0) or 0.0),
    )
    target_overuse = float(target.get("overuse", 0.0) or 0.0)
    return (
        local_delta
        + 40.0 * source_overuse
        + 18.0 * source_spread
        + 65.0 * target_underuse
        - 25.0 * target_overuse
    )


def _family_competition_proposals(
    cipher_text: CipherText,
    key: dict[int, int],
    family_report: dict[str, Any],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    model: homophonic.ContinuousNGramModel,
    window_step: int = 1,
    family_limit: int = 3,
    top_k_per_symbol: int = 3,
    beam_limit: int = 4,
) -> list[dict[str, Any]]:
    suspect_families = (family_report.get("families") or [])[: max(1, family_limit)]
    if not suspect_families:
        return []
    tokens = list(cipher_text.tokens)
    active_letters = sorted({id_to_letter[key[sid]] for sid in sorted(set(tokens))})
    expected = _expected_letter_fractions()
    family_index = _family_report_index(family_report)
    underused_letters = sorted(
        expected,
        key=lambda letter: (
            expected[letter] - float(family_index.get(letter, {}).get("observed_fraction", 0.0) or 0.0),
            expected[letter],
        ),
        reverse=True,
    )
    proposals: list[dict[str, Any]] = []
    seen_keys: set[tuple[tuple[int, int], ...]] = set()
    for family in suspect_families:
        family_letter = family["letter"]
        candidate_letters = []
        for letter in active_letters + underused_letters:
            if letter == family_letter or letter in candidate_letters:
                continue
            candidate_letters.append(letter)
            if len(candidate_letters) >= max(6, top_k_per_symbol + 2):
                break
        family_symbol_alts: list[dict[str, Any]] = []
        for sid in family["symbol_ids"]:
            alternatives = _symbol_letter_alternatives(
                cipher_text,
                key,
                id_to_letter,
                letter_to_id,
                model,
                sid,
                candidate_letters,
                window_step=window_step,
                top_k=top_k_per_symbol,
            )
            for alt in alternatives:
                branch_updates = {sid: letter_to_id[alt["to_letter"]]}
                branch_key = tuple(sorted(branch_updates.items()))
                if branch_key in seen_keys:
                    continue
                seen_keys.add(branch_key)
                mutable_letters = {alt["from_letter"], alt["to_letter"]}
                mutable_symbols = _branch_mutable_symbols(
                    tokens,
                    key,
                    id_to_letter,
                    mutable_letters,
                    {sid},
                )
                score = _branch_score(
                    local_delta=float(alt["local_delta"]),
                    source_letter=alt["from_letter"],
                    target_letter=alt["to_letter"],
                    family_index=family_index,
                )
                proposals.append({
                    "kind": "single_symbol_reassign",
                    "score": round(score, 4),
                    "source_letter": alt["from_letter"],
                    "target_letter": alt["to_letter"],
                    "trigger_symbol": sid,
                    "mutable_symbols": mutable_symbols,
                    "branch_updates": branch_updates,
                    "description": f"{alt['from_letter']} -> {alt['to_letter']} via symbol {sid}",
                    "local_delta": alt["local_delta"],
                })
                family_symbol_alts.append(alt)
        if len(family["symbol_ids"]) >= 2 and family_symbol_alts:
            top_alts = sorted(family_symbol_alts, key=lambda alt: alt["local_delta"], reverse=True)[: max(4, top_k_per_symbol * 2)]
            for i, alt_a in enumerate(top_alts):
                for alt_b in top_alts[i + 1 :]:
                    if alt_a["symbol_id"] == alt_b["symbol_id"]:
                        continue
                    branch_updates = {
                        alt_a["symbol_id"]: letter_to_id[alt_a["to_letter"]],
                        alt_b["symbol_id"]: letter_to_id[alt_b["to_letter"]],
                    }
                    branch_key = tuple(sorted(branch_updates.items()))
                    if branch_key in seen_keys:
                        continue
                    seen_keys.add(branch_key)
                    mutable_letters = {
                        alt_a["from_letter"],
                        alt_a["to_letter"],
                        alt_b["to_letter"],
                    }
                    mutable_symbols = _branch_mutable_symbols(
                        tokens,
                        key,
                        id_to_letter,
                        mutable_letters,
                        set(branch_updates),
                    )
                    combo_score = (
                        _branch_score(
                            local_delta=float(alt_a["local_delta"]),
                            source_letter=alt_a["from_letter"],
                            target_letter=alt_a["to_letter"],
                            family_index=family_index,
                        )
                        + _branch_score(
                            local_delta=float(alt_b["local_delta"]),
                            source_letter=alt_b["from_letter"],
                            target_letter=alt_b["to_letter"],
                            family_index=family_index,
                        )
                        + 6.0
                    )
                    proposals.append({
                        "kind": "two_symbol_split",
                        "score": round(combo_score, 4),
                        "source_letter": family_letter,
                        "target_letter": f"{alt_a['to_letter']}+{alt_b['to_letter']}",
                        "trigger_symbol": alt_a["symbol_id"],
                        "mutable_symbols": mutable_symbols,
                        "branch_updates": branch_updates,
                        "description": (
                            f"split {family_letter} via {alt_a['symbol_id']}->{alt_a['to_letter']} "
                            f"and {alt_b['symbol_id']}->{alt_b['to_letter']}"
                        ),
                        "local_delta": round(float(alt_a["local_delta"]) + float(alt_b["local_delta"]), 4),
                    })
    proposals.sort(key=lambda item: item["score"], reverse=True)
    return proposals[: max(1, beam_limit)]


def _run_targeted_homophonic_repair(
    *,
    cipher_text: CipherText,
    language: str,
    word_list: list[str],
    selected_key: dict[int, int],
    selected_plaintext: str,
    model: homophonic.ContinuousNGramModel,
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    refine_params: dict[str, Any],
    refine_config: dict[str, Any],
    refine_weights: dict[str, float],
) -> tuple[homophonic.HomophonicAnnealResult, dict[str, Any]]:
    symbol_ids = sorted(set(cipher_text.tokens))
    plans = refine_params["repair_plans"]
    attempts: list[dict[str, Any]] = []
    best: homophonic.HomophonicAnnealResult | None = None
    best_selection_score = float("-inf")
    best_plan: dict[str, Any] | None = None

    for plan in plans:
        suspicious = _suspicious_homophonic_symbols(
            cipher_text,
            selected_key,
            id_to_letter,
            model,
            limit=plan["suspicious_limit"],
            window_step=refine_config["window_step"],
        )
        fixed_cipher_ids = set(symbol_ids) - set(suspicious)
        repaired = homophonic.homophonic_simulated_anneal(
            tokens=list(cipher_text.tokens),
            plaintext_ids=plaintext_ids,
            id_to_letter=id_to_letter,
            letter_to_id=letter_to_id,
            model=model,
            initial_key=selected_key,
            fixed_cipher_ids=fixed_cipher_ids,
            epochs=plan["epochs"],
            sampler_iterations=plan["sampler_iterations"],
            t_start=plan["t_start"],
            t_end=plan["t_end"],
            distribution_weight=refine_weights["distribution_weight"],
            diversity_weight=refine_weights["diversity_weight"],
            ioc_weight=refine_weights["ioc_weight"],
            score_formula=refine_config["score_formula"],
            window_step=refine_config["window_step"],
            move_profile=plan["move_profile"],
            seed=plan["seed"],
            top_n=3,
        )
        repaired_quality = _plaintext_quality(repaired.plaintext, repaired.key)
        repaired_diagnostics = _automated_candidate_diagnostics(
            repaired.plaintext,
            language=language,
            word_list=word_list,
        )
        repaired_selection_score = repaired.normalized_score - repaired_quality["penalty"]
        attempts.append({
            "plan": plan["name"],
            "suspicious_limit": plan["suspicious_limit"],
            "targeted_symbols": suspicious,
            "fixed_symbol_count": len(fixed_cipher_ids),
            "mutable_symbol_count": len(suspicious),
            "selection_score": round(repaired_selection_score, 4),
            "anneal_score": round(repaired.normalized_score, 4),
            "quality": repaired_quality,
            "diagnostics": repaired_diagnostics,
            "preview": repaired.plaintext[:160],
            "move_profile": plan["move_profile"],
            "epoch_traces": repaired.metadata.get("epoch_traces", []),
        })
        if best is None or repaired_selection_score > best_selection_score:
            best = repaired
            best_selection_score = repaired_selection_score
            best_plan = plan

    assert best is not None
    assert best_plan is not None
    refinement_step = {
        "mode": "targeted_repair",
        "profile": refine_params["profile"],
        "weights": refine_weights,
        "score_formula": refine_config["score_formula"],
        "window_step": refine_config["window_step"],
        "repair_plans": plans,
        "selected_plan": best_plan["name"],
        "selected_move_profile": best_plan["move_profile"],
        "selected_suspicious_limit": best_plan["suspicious_limit"],
        "attempts": attempts,
        "base_preview": selected_plaintext[:160],
    }
    return best, refinement_step


def _run_family_homophonic_repair(
    *,
    cipher_text: CipherText,
    language: str,
    word_list: list[str],
    elite_candidates: list[dict[str, Any]],
    selected_key: dict[int, int],
    selected_plaintext: str,
    selected_quality: dict[str, Any],
    selected_diagnostics: dict[str, Any],
    model: homophonic.ContinuousNGramModel,
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    letter_to_id: dict[str, int],
    refine_params: dict[str, Any],
    refine_config: dict[str, Any],
    refine_weights: dict[str, float],
) -> tuple[homophonic.HomophonicAnnealResult | None, dict[str, Any]]:
    gate = _family_repair_gate(selected_quality, selected_diagnostics)
    family_report = _homophonic_family_diagnostics(
        cipher_text,
        selected_key,
        id_to_letter,
        model,
        window_step=refine_config["window_step"],
    )
    symbol_ids = sorted(set(cipher_text.tokens))
    plans = refine_params["repair_plans"]
    attempts: list[dict[str, Any]] = []
    refinement_step: dict[str, Any] = {
        "mode": "family_repair",
        "profile": refine_params["profile"],
        "weights": refine_weights,
        "score_formula": refine_config["score_formula"],
        "window_step": refine_config["window_step"],
        "repair_plans": plans,
        "base_preview": selected_plaintext[:160],
        "gate": gate,
        "family_diagnostics": family_report,
        "attempts": attempts,
        "screening": [],
    }
    if not gate["ok"]:
        refinement_step.update({
            "skipped": True,
            "skip_reason": "candidate_not_readable_enough_for_family_repair",
        })
        return None, refinement_step

    best: homophonic.HomophonicAnnealResult | None = None
    best_selection_score = float("-inf")
    best_plan: dict[str, Any] | None = None
    elite_pool = elite_candidates or [{
        "seed": None,
        "epoch": None,
        "plaintext": selected_plaintext,
        "key": selected_key,
        "quality": selected_quality,
        "diagnostics": selected_diagnostics,
    }]
    refinement_step["elite_pool"] = [
        {
            "rank": idx + 1,
            "seed": elite.get("seed"),
            "epoch": elite.get("epoch"),
            "selection_score": elite.get("selection_score"),
            "anneal_score": elite.get("anneal_score"),
            "preview": (elite.get("plaintext") or "")[:160],
        }
        for idx, elite in enumerate(elite_pool)
    ]
    for elite_index, elite in enumerate(elite_pool, start=1):
        elite_key = elite.get("key") or selected_key
        elite_plaintext = elite.get("plaintext") or selected_plaintext
        elite_quality = elite.get("quality") or selected_quality
        elite_diagnostics = elite.get("diagnostics") or selected_diagnostics
        elite_gate = _family_repair_gate(elite_quality, elite_diagnostics)
        if not elite_gate["ok"]:
            attempts.append({
                "elite_rank": elite_index,
                "plan": None,
                "skipped": True,
                "skip_reason": "elite_not_readable_enough_for_family_repair",
                "gate": elite_gate,
                "preview": elite_plaintext[:160],
            })
            continue
        elite_family_report = _homophonic_family_diagnostics(
            cipher_text,
            elite_key,
            id_to_letter,
            model,
            window_step=refine_config["window_step"],
        )
        families = elite_family_report["families"]
        for plan in plans:
            suspect_families = families[: max(1, int(plan["family_limit"]))]
            competition_proposals = _family_competition_proposals(
                cipher_text,
                elite_key,
                elite_family_report,
                id_to_letter,
                letter_to_id,
                model,
                window_step=refine_config["window_step"],
                family_limit=plan["family_limit"],
                top_k_per_symbol=plan.get("top_k_per_symbol", 3),
                beam_limit=plan.get("beam_limit", 4),
            )
            if not competition_proposals:
                mutable_symbols = sorted({
                    sid
                    for family in suspect_families
                    for sid in family["symbol_ids"]
                })
                competition_proposals = [{
                    "kind": "fallback_family_open",
                    "score": 0.0,
                    "source_letter": None,
                    "target_letter": None,
                    "trigger_symbol": None,
                    "mutable_symbols": mutable_symbols,
                    "branch_updates": {},
                    "description": "fallback family-open repair",
                }]

            min_branch_score = float(plan.get("min_branch_score", float("-inf")))
            screened_proposals = [
                branch
                for branch in competition_proposals
                if float(branch.get("score", 0.0)) >= min_branch_score
            ]
            if not screened_proposals and competition_proposals:
                screened_proposals = [competition_proposals[0]]
            screen_limit = int(plan.get("screen_limit", 0) or 0)
            if screen_limit > 0:
                screened_proposals = screened_proposals[:screen_limit]
            refinement_step["screening"].append({
                "elite_rank": elite_index,
                "plan": plan["name"],
                "candidate_branch_count": len(competition_proposals),
                "screened_branch_count": len(screened_proposals),
                "min_branch_score": None if math.isinf(min_branch_score) and min_branch_score < 0 else round(min_branch_score, 4),
                "screen_limit": screen_limit or None,
            })

            for branch in screened_proposals:
                branch_key = dict(elite_key)
                branch_key.update(branch["branch_updates"])
                fixed_cipher_ids = set(symbol_ids) - set(branch["mutable_symbols"])
                repaired = homophonic.homophonic_simulated_anneal(
                    tokens=list(cipher_text.tokens),
                    plaintext_ids=plaintext_ids,
                    id_to_letter=id_to_letter,
                    letter_to_id=letter_to_id,
                    model=model,
                    initial_key=branch_key,
                    fixed_cipher_ids=fixed_cipher_ids,
                    epochs=plan["epochs"],
                    sampler_iterations=plan["sampler_iterations"],
                    t_start=plan["t_start"],
                    t_end=plan["t_end"],
                    distribution_weight=refine_weights["distribution_weight"],
                    diversity_weight=refine_weights["diversity_weight"],
                    ioc_weight=refine_weights["ioc_weight"],
                    score_formula=refine_config["score_formula"],
                    window_step=refine_config["window_step"],
                    move_profile=plan["move_profile"],
                    seed=plan["seed"],
                    top_n=3,
                )
                repaired_quality = _plaintext_quality(repaired.plaintext, repaired.key)
                repaired_diagnostics = _automated_candidate_diagnostics(
                    repaired.plaintext,
                    language=language,
                    word_list=word_list,
                )
                repaired_selection_score = repaired.normalized_score - repaired_quality["penalty"]
                attempts.append({
                    "elite_rank": elite_index,
                    "elite_seed": elite.get("seed"),
                    "elite_epoch": elite.get("epoch"),
                    "plan": plan["name"],
                    "family_limit": plan["family_limit"],
                    "targeted_letters": sorted({
                        id_to_letter[elite_key[sid]]
                        for sid in branch["mutable_symbols"]
                        if sid in elite_key
                    }),
                    "targeted_symbols": branch["mutable_symbols"],
                    "fixed_symbol_count": len(fixed_cipher_ids),
                    "mutable_symbol_count": len(branch["mutable_symbols"]),
                    "selection_score": round(repaired_selection_score, 4),
                    "anneal_score": round(repaired.normalized_score, 4),
                    "quality": repaired_quality,
                    "diagnostics": repaired_diagnostics,
                    "preview": repaired.plaintext[:160],
                    "move_profile": plan["move_profile"],
                    "epoch_traces": repaired.metadata.get("epoch_traces", []),
                    "family_snapshot": suspect_families,
                    "competition_branch": branch,
                    "elite_preview": elite_plaintext[:160],
                })
                if best is None or repaired_selection_score > best_selection_score:
                    best = repaired
                    best_selection_score = repaired_selection_score
                    best_plan = {**plan, "branch": branch, "elite_rank": elite_index, "elite_seed": elite.get("seed"), "elite_epoch": elite.get("epoch")}

    assert best is not None
    assert best_plan is not None
    refinement_step.update({
        "selected_plan": best_plan["name"],
        "selected_move_profile": best_plan["move_profile"],
        "selected_family_limit": best_plan["family_limit"],
        "selected_branch": best_plan.get("branch"),
        "selected_elite_rank": best_plan.get("elite_rank"),
        "selected_elite_seed": best_plan.get("elite_seed"),
        "selected_elite_epoch": best_plan.get("elite_epoch"),
    })
    return best, refinement_step


def _homophonic_budget_params(
    budget: str,
    short_homophonic: bool,
    search_profile: str = "full",
) -> dict[str, Any]:
    budget_key = (budget or "full").strip().lower()
    profile_key = (search_profile or "full").strip().lower()
    if profile_key not in {"full", "dev"}:
        raise ValueError(
            f"unsupported homophonic search profile '{search_profile}' "
            "(expected one of: full, dev)"
        )
    if budget_key == "full":
        if profile_key == "dev":
            return {
                "budget": "full",
                "search_profile": "dev",
                "seeds": [0, 1, 2, 3] if short_homophonic else [0, 1],
                "epochs": 5 if short_homophonic else 4,
                "sampler_iterations": 1800 if short_homophonic else 1500,
            }
        return {
            "budget": "full",
            "search_profile": "full",
            "seeds": list(range(8)) if short_homophonic else [0, 1, 2, 3],
            "epochs": 9 if short_homophonic else 7,
            "sampler_iterations": 4000 if short_homophonic else 3000,
        }
    if budget_key == "screen":
        if profile_key == "dev":
            return {
                "budget": "screen",
                "search_profile": "dev",
                "seeds": [0, 1] if short_homophonic else [0],
                "epochs": 3,
                "sampler_iterations": 900 if short_homophonic else 700,
            }
        return {
            "budget": "screen",
            "search_profile": "full",
            "seeds": [0, 1, 2, 3] if short_homophonic else [0, 1, 2],
            "epochs": 5 if short_homophonic else 4,
            "sampler_iterations": 1500 if short_homophonic else 1200,
        }
    raise ValueError(f"unsupported homophonic budget '{budget}' (expected one of: full, screen)")


def _homophonic_refinement_params(
    refinement: str,
    budget: str,
    short_homophonic: bool,
    repair_profile: str = "full",
) -> dict[str, Any]:
    refinement_key = (refinement or "none").strip().lower()
    profile_key = (repair_profile or "full").strip().lower()
    if profile_key not in {"full", "dev"}:
        raise ValueError(
            f"unsupported homophonic repair profile '{repair_profile}' "
            "(expected one of: full, dev)"
        )
    if refinement_key == "none":
        return {"mode": "none"}
    if refinement_key == "two_stage":
        if (budget or "full").strip().lower() == "screen":
            return {
                "mode": "two_stage",
                "profile": "ngram_distribution",
                "epochs": 1,
                "sampler_iterations": 900 if short_homophonic else 700,
                "t_start": 0.004,
                "t_end": 0.001,
                "seed": 0,
            }
        return {
            "mode": "two_stage",
            "profile": "ngram_distribution",
            "epochs": 1,
            "sampler_iterations": 1800 if short_homophonic else 1400,
            "t_start": 0.004,
            "t_end": 0.001,
                "seed": 0,
            }
    if refinement_key == "targeted_repair":
        if (budget or "full").strip().lower() == "screen":
            return {
                "mode": "targeted_repair",
                "profile": "balanced",
                "repair_profile": profile_key,
                "repair_plans": [
                    {
                        "name": "targeted8",
                        "suspicious_limit": 8,
                        "epochs": 2,
                        "sampler_iterations": 500,
                        "t_start": 0.006,
                        "t_end": 0.0015,
                        "seed": 0,
                        "move_profile": "mixed_v1_targeted",
                    }
                ],
            }
        if profile_key == "dev":
            return {
                "mode": "targeted_repair",
                "profile": "balanced",
                "repair_profile": profile_key,
                "repair_plans": [
                    {
                        "name": "targeted8",
                        "suspicious_limit": 8,
                        "epochs": 2,
                        "sampler_iterations": 500,
                        "t_start": 0.006,
                        "t_end": 0.0015,
                        "seed": 0,
                        "move_profile": "mixed_v1_targeted",
                    }
                ],
            }
        return {
            "mode": "targeted_repair",
            "profile": "balanced",
            "repair_profile": profile_key,
            "repair_plans": [
                {
                    "name": "targeted8",
                    "suspicious_limit": 8,
                    "epochs": 3,
                    "sampler_iterations": 900,
                    "t_start": 0.006,
                    "t_end": 0.0015,
                    "seed": 0,
                    "move_profile": "mixed_v1_targeted",
                },
                {
                    "name": "targeted12",
                    "suspicious_limit": 12,
                    "epochs": 3,
                    "sampler_iterations": 1100,
                    "t_start": 0.006,
                    "t_end": 0.0015,
                    "seed": 1,
                    "move_profile": "mixed_v1_targeted",
                },
            ],
        }
    if refinement_key == "family_repair":
        if (budget or "full").strip().lower() == "screen":
            return {
                "mode": "family_repair",
                "profile": "balanced",
                "repair_profile": profile_key,
                "repair_plans": [
                    {
                        "name": "family2",
                        "family_limit": 2,
                        "top_k_per_symbol": 3,
                        "beam_limit": 3,
                        "epochs": 2,
                        "sampler_iterations": 500,
                        "t_start": 0.006,
                        "t_end": 0.0015,
                        "seed": 0,
                        "move_profile": "mixed_v1_targeted",
                    }
                ],
            }
        if profile_key == "dev":
            return {
                "mode": "family_repair",
                "profile": "balanced",
                "repair_profile": profile_key,
                "repair_plans": [
                    {
                        "name": "family2",
                        "family_limit": 2,
                        "top_k_per_symbol": 2,
                        "beam_limit": 2,
                        "screen_limit": 1,
                        "min_branch_score": 0.0,
                        "epochs": 2,
                        "sampler_iterations": 500,
                        "t_start": 0.006,
                        "t_end": 0.0015,
                        "seed": 0,
                        "move_profile": "mixed_v1_targeted",
                    }
                ],
            }
        return {
            "mode": "family_repair",
            "profile": "balanced",
            "repair_profile": profile_key,
            "repair_plans": [
                {
                    "name": "family2",
                    "family_limit": 2,
                    "top_k_per_symbol": 3,
                    "beam_limit": 3,
                    "epochs": 3,
                    "sampler_iterations": 900,
                    "t_start": 0.006,
                    "t_end": 0.0015,
                    "seed": 0,
                    "move_profile": "mixed_v1_targeted",
                },
                {
                    "name": "family3",
                    "family_limit": 3,
                    "top_k_per_symbol": 3,
                    "beam_limit": 4,
                    "epochs": 3,
                    "sampler_iterations": 1100,
                    "t_start": 0.006,
                    "t_end": 0.0015,
                    "seed": 1,
                    "move_profile": "mixed_v1_targeted",
                },
            ],
        }
    raise ValueError(
        f"unsupported homophonic refinement '{refinement}' "
        "(expected one of: none, two_stage, targeted_repair, family_repair)"
    )


def _automated_candidate_diagnostics(
    plaintext: str,
    language: str,
    word_list: list[str],
) -> dict[str, Any]:
    upper = "".join(ch for ch in plaintext.upper() if "A" <= ch <= "Z")
    counts = Counter(upper)
    diagnostics: dict[str, Any] = {
        "letter_count": len(upper),
        "unique_letters": len(counts),
        "top_letter_fraction": round((max(counts.values()) / len(upper)), 4) if upper else 0.0,
        "index_of_coincidence": round(
            ic.index_of_coincidence([ord(ch) - ord("A") for ch in upper], 26),
            4,
        ) if len(upper) > 1 else 0.0,
    }
    path = dictionary.get_dictionary_path(language)
    word_set = dictionary.load_word_set(path) if path else set()
    if upper and word_set:
        freq_rank = {word.upper(): idx for idx, word in enumerate(word_list)}
        seg = segment_text(upper, word_set, freq_rank=freq_rank)
        diagnostics.update({
            "dict_rate": round(seg.dict_rate, 4),
            "segmentation_cost": round(seg.cost, 3),
            "segmented_preview": seg.segmented[:160],
            "pseudo_word_count": len(seg.pseudo_words),
        })
    return diagnostics
