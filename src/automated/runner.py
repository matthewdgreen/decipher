"""Automated-only solving without LLM API calls.

This module deliberately stays separate from ``agent.loop_v2``. It uses the
same native solver building blocks exposed to the agent tools, but runs them
deterministically from local code and writes a small dashboard-compatible
artifact marked ``run_mode: automated_only``.
"""
from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import json
import math
import os
import random
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from analysis import cipher_id as cipher_id_analysis
from analysis import dictionary, homophonic, ic, model_registry, ngram, pattern, polyalphabetic
from analysis.candidate_packet import (
    packet_from_null_mask_row,
    packet_from_pure_transposition_row,
    packet_from_transform_row,
)
from analysis.homophonic_nulls import (
    attach_null_mask_ensemble_scores,
    diagnose_cipher_for_null_candidates,
    generate_null_masks,
    null_mask_language_quality_rank_key,
    null_mask_rank_key,
    null_mask_validation_score_v2,
    select_null_candidate_symbols,
)
from analysis.language_scoring import LinearLanguageQualityModel, content_word_metrics
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
from analysis.transform_homophonic_batch import (
    run_zenith_transform_confirmation_batches,
    run_zenith_transform_rank_batch,
)
from analysis.transformers import TransformPipeline, apply_transform_pipeline
from analysis.transform_search import inspect_transform_suspicion, screen_transform_candidates
from automated.transform_homophonic_runtime import (
    build_transform_homophonic_batch_context,
    plaintext_quality_score,
    select_transform_confirmation_finalists,
    transform_homophonic_scoring_policy,
    transform_homophonic_probe_policy,
    transform_mutation_penalty,
    transform_selection_score,
)
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


# F8: the automated pipeline has no renderer (display is forced "off"). The CLI
# threads an optional ``on_step(name, status, elapsed)`` callback that PRINTS
# DIRECTLY (safe — no live display in automated mode). Rather than instrument the
# ~14 ``steps.append`` sites individually, wrap the steps list so every append
# fires the callback with the step's "name", optional "status", and the elapsed
# time since the previous step. No behavior change when ``on_step`` is None.
OnStep = Callable[[str, "str | None", float], None]


class _StepList(list):
    """A ``list`` that fires ``on_step`` each time a pipeline step is appended."""

    def __init__(self, on_step: "OnStep | None" = None, clock: "Callable[[], float] | None" = None) -> None:
        super().__init__()
        self._on_step = on_step
        self._clock = clock or time.monotonic
        self._last = self._clock()

    def append(self, step: Any) -> None:  # noqa: D401
        super().append(step)
        if self._on_step is None:
            return
        now = self._clock()
        elapsed = now - self._last
        self._last = now
        try:
            name = step.get("name", "step") if isinstance(step, dict) else "step"
            status = step.get("status") if isinstance(step, dict) else None
            self._on_step(str(name), status, elapsed)
        except Exception:  # noqa: BLE001 - progress printing must never crash a run
            pass


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
        model_variant: str | None = None,
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
        # ``None`` (default) / concrete slug / ``"auto"`` (source-mapped per test).
        self.model_variant = model_variant

    def _resolve_language(self, test_data: TestData) -> str:
        return resolve_test_language(test_data, self.default_language)

    def run_test(
        self,
        test_data: TestData,
        language: str | None = None,
        on_step: "OnStep | None" = None,
    ) -> AutomatedRunResult:
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

        # Resolve ``"auto"`` against the test's benchmark source (test-id prefix)
        # gated on the resolved run language; a concrete slug or ``None`` passes
        # through unchanged.
        source = test_id.split("_")[0]
        resolved_variant = resolve_model_variant(self.model_variant, source, lang)
        run_kwargs = {
            "cipher_text": cipher_text,
            "language": lang,
            "cipher_id": test_id,
            "ground_truth": test_data.plaintext,
            "cipher_system": test_data.test.cipher_system,
            "homophonic_budget": self.homophonic_budget,
            "homophonic_refinement": self.homophonic_refinement,
            "homophonic_solver": self.homophonic_solver,
            "model_variant": resolved_variant,
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
        if on_step is not None:
            run_kwargs["on_step"] = on_step
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


SOURCE_MODEL_VARIANTS: dict[str, tuple[str, str]] = {
    # Benchmark-context mapping consulted ONLY when ``model_variant == "auto"``
    # (explicit opt-in). Maps a benchmark source to (language, variant slug).
    "copiale": ("de", "historical_1600_1899"),
}


def resolve_model_variant(
    model_variant: str | None,
    source: str | None = None,
    language: str | None = None,
) -> str | None:
    """Turn a CLI/runner ``model_variant`` value into a concrete slug or None.

    ``None`` stays ``None`` (default resolution). ``"auto"`` consults
    :data:`SOURCE_MODEL_VARIANTS` for ``source``; the mapping applies only when
    the mapping's language matches the run ``language`` (when one is supplied) —
    e.g. copiale forced to ``--language la`` must NOT auto-select the German DTA
    model, so the mapping is skipped and default resolution applies. A source
    with no mapping also yields ``None``. Any other value is an explicit slug
    passed through unchanged.
    """
    if model_variant is None:
        return None
    if model_variant == "auto":
        mapping = SOURCE_MODEL_VARIANTS.get((source or "").strip().lower())
        if mapping is None:
            return None
        mapped_language, mapped_variant = mapping
        if language is not None and language.strip().lower() != mapped_language:
            # Clear skip: auto-mapping is language-gated (see docstring).
            return None
        return mapped_variant
    return model_variant


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
    model_variant: str | None = None,
    on_step: "OnStep | None" = None,
) -> AutomatedRunResult:
    """Run the automated pipeline, selecting an optional language-model variant.

    ``model_variant`` (default ``None`` == today's default resolution) is set as
    the calling thread's active selection for the duration of the run and
    restored afterward, so every internal ``_zenith_native_model_path`` choke
    point picks up the chosen variant without a signature change (the slot is a
    ``threading.local``, so concurrent runs on other threads are unaffected).
    ``"auto"`` is resolved by the caller (benchmark context); this entry point
    expects a concrete slug or ``None``.
    """
    prev_variant = set_active_model_variant(model_variant)
    try:
        return _run_automated_impl(
            cipher_text=cipher_text,
            language=language,
            cipher_id=cipher_id,
            ground_truth=ground_truth,
            cipher_system=cipher_system,
            solver_hints=solver_hints,
            transform_pipeline=transform_pipeline,
            homophonic_budget=homophonic_budget,
            homophonic_refinement=homophonic_refinement,
            homophonic_solver=homophonic_solver,
            transform_search=transform_search,
            transform_search_profile=transform_search_profile,
            transform_search_max_generated_candidates=transform_search_max_generated_candidates,
            transform_promote_artifact=transform_promote_artifact,
            transform_promote_candidate_ids=transform_promote_candidate_ids,
            transform_promote_top_n=transform_promote_top_n,
            model_variant=model_variant,
            on_step=on_step,
        )
    finally:
        set_active_model_variant(prev_variant)


def _run_automated_impl(
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
    model_variant: str | None = None,
    on_step: "OnStep | None" = None,
) -> AutomatedRunResult:
    """Run the best available local techniques without any LLM call."""
    started = time.time()
    run_id = uuid.uuid4().hex[:12]
    # F8: a step-append hook drives optional CLI progress narration. Plain list
    # when no callback (zero overhead / no behavior change).
    steps: list[dict[str, Any]] = _StepList(on_step)
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
        if model_variant is not None:
            # Provenance policy (spec-author sign-off, review round 2): EVERY
            # ``binary_ngram_model`` record — including on default runs — now
            # carries the two additive sidecar-sourced keys ``variant`` and
            # ``display_label`` (via ``_zenith_native_model_metadata``); that is
            # desirable, not a regression of the spec Part-4 "default None
            # byte-identical" pin, which covers resolution and the pre-existing
            # artifact fields, minus these two additive keys. The route-step
            # ``model_variant`` field below (the *requested* selection) remains
            # conditional: it is recorded only when a variant was explicitly
            # requested.
            steps[-1]["model_variant"] = model_variant
            resolved = _zenith_native_model_path(language)
            steps[-1]["binary_ngram_model"] = (
                _zenith_native_model_metadata(str(resolved)) if resolved else None
            )
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
            base_refinement = (
                "none"
                if _refinement_runs_null_masks(homophonic_refinement)
                or _is_word_repair_refinement(homophonic_refinement)
                else homophonic_refinement
            )
            solver, key, decryption, step = _run_homophonic(
                cipher_text,
                language,
                budget=homophonic_budget,
                refinement=base_refinement,
                solver_profile=homophonic_solver,
                ground_truth=ground_truth,
            )
            steps.append(step)
            # The mask travels with the basin word repair refines: plain
            # word_repair works on the unmasked solve, the composite works on
            # the null-mask bakeoff winner's masked basin.
            word_repair_mask: tuple[str, ...] = ()
            if _refinement_runs_null_masks(homophonic_refinement):
                bakeoff = _run_null_mask_bakeoff(
                    cipher_text=cipher_text,
                    language=language,
                    budget=homophonic_budget,
                    solver_profile=homophonic_solver,
                    base_solver=solver,
                    base_key=key,
                    base_decryption=decryption,
                    base_step=step,
                )
                steps.append(bakeoff)
                winner = bakeoff.get("selected") if isinstance(bakeoff, dict) else None
                if isinstance(winner, dict) and winner.get("status") == "completed":
                    solver = "null_mask_homophonic"
                    key = {
                        int(k): int(v)
                        for k, v in (winner.get("key") or {}).items()
                    }
                    decryption = str(winner.get("decryption") or "")
                    word_repair_mask = tuple(str(sym) for sym in (winner.get("mask") or []))
            if _is_word_repair_refinement(homophonic_refinement):
                word_repair_step, adopted = _run_word_repair_refinement(
                    cipher_text=cipher_text,
                    language=language,
                    refinement=homophonic_refinement,
                    base_solver=solver,
                    base_key=key,
                    base_decryption=decryption,
                    mask=word_repair_mask,
                )
                steps.append(word_repair_step)
                if adopted is not None:
                    solver, key, decryption = adopted
        elif routing["route"] == "pure_transposition":
            solver, key, decryption, step = _run_pure_transposition(
                cipher_text,
                language,
                cipher_system=cipher_system,
                solver_hints=solver_hints,
            )
            steps.append(step)
        elif routing["route"] == "transposition":
            solver, key, decryption, step = _run_transposition_solver(
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
            rescue = _maybe_rescue_substitution_run(
                cipher_text=cipher_text,
                language=language,
                cipher_id=cipher_id,
                initial_solver=solver,
                initial_key=key,
                initial_decryption=decryption,
                initial_step=step,
            )
            if rescue is not None:
                steps.append(rescue["step"])
                if rescue["selected_attempt_index"] is not None:
                    solver = rescue["solver"]
                    key = rescue["key"]
                    decryption = rescue["decryption"]
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
            quality_score = plaintext_quality_score(refined_decryption, language)
            structural_score = _float_or_none(candidate.get("structural_score"))
            if structural_score is None:
                structural_score = _float_or_none(candidate.get("score"))
            mutation_penalty = transform_mutation_penalty(candidate)
            full_selection_score = transform_selection_score(
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
                quality_score = plaintext_quality_score(decryption, language)
                mutation_penalty = transform_mutation_penalty(candidate)
                selection_score = transform_selection_score(
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
    # Additive artifact enrichment: attach a normalized candidate packet to each
    # ranked transform candidate. NOTE: the transform menu's ``finalists`` key is
    # a label-count summary, not a candidate-row list, so packets attach here to
    # ``top_ranked_candidates`` (the actual finalist rows). See implementation
    # report for this documented mismatch vs. the spec's wording.
    for transform_rank, transform_row in enumerate(ranked, start=1):
        transform_row["packet"] = packet_from_transform_row(
            transform_row, rank=transform_rank
        ).to_dict()
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

    batch_context = _transform_homophonic_batch_context(
        cipher_text=cipher_text,
        language=language,
        budget=budget,
        purpose="zenith_native Rust transform ranking",
    )
    probe_policy = transform_homophonic_probe_policy(
        budget=budget,
        adaptive_confirmations=0,
    )
    return run_zenith_transform_rank_batch(
        tokens=list(cipher_text.tokens),
        context=batch_context,
        top_n=probe_policy["rank_top_n"],
        raw_candidates=raw_candidates,
        scoring_policy=transform_homophonic_scoring_policy(language),
    )


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

    finalists = select_transform_confirmation_finalists(
        ranked,
        confirm_count=confirm_count,
    )
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
            quality_score = plaintext_quality_score(decryption, language)
            mutation_penalty = transform_mutation_penalty(item)
            confirmation_selection = transform_selection_score(
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

    batch_context = _transform_homophonic_batch_context(
        cipher_text=cipher_text,
        language=language,
        budget=budget,
        purpose="Rust transform confirmation",
    )
    probe_policy = transform_homophonic_probe_policy(
        budget=budget,
        adaptive_confirmations=adaptive_confirmations,
    )
    return run_zenith_transform_confirmation_batches(
        tokens=list(cipher_text.tokens),
        ranked=ranked,
        finalists=finalists,
        context=batch_context,
        scoring_policy=transform_homophonic_scoring_policy(language),
        confirmation_policy=probe_policy["confirmation_policy"],
        plaintext_distance_fn=_plaintext_distance_ratio,
    )


def _transform_homophonic_batch_context(
    *,
    cipher_text: CipherText,
    language: str,
    budget: str,
    purpose: str,
) -> Any:
    """Build shared model/budget/thread context for Rust transform+Zenith probes."""

    pt_alpha = _plaintext_alphabet(language)
    return build_transform_homophonic_batch_context(
        language=language,
        token_count=len(cipher_text.tokens),
        budget=budget,
        model_path=_zenith_native_model_path(language),
        plaintext_symbols=[pt_alpha.symbol_for(i).upper() for i in range(pt_alpha.size)],
        search_profile=_homophonic_search_profile(),
        budget_params_fn=_homophonic_budget_params,
        threads=_transform_rank_threads(),
        purpose=purpose,
    )


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


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_language_quality_model(path_raw: str, *, language: str) -> LinearLanguageQualityModel | None:
    if not path_raw:
        return None
    path = Path(path_raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    model = LinearLanguageQualityModel.load(path)
    if model.language != language:
        raise ValueError(
            f"language quality model language {model.language!r} does not match run language {language!r}"
        )
    return model


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
            # Natural-language summary is already in the ## Cipher-diagnostic preflight
            # section of the initial context — omit it here to avoid verbatim repetition.
        ]
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
    # --- keyed-column transposition family (columnar/railfence/redefence/
    # myszkowski/amsco/nihilist/cadenus) ---
    # The pure_transposition route above already claims cipher systems whose
    # name contains "transposition" (columnar/route/nihilist). Catch the ACA
    # families whose names do not, plus unlabeled ciphers whose monogram
    # distribution matches the language BY LETTER — a signature that a
    # transposition preserves but a substitution/homophonic destroys. This runs
    # only for A-Z-sized alphabets (larger alphabets were routed homophonic
    # above), so a plain-substitution or homophonic cipher never reaches a
    # transposition sweep.
    if not is_mixed_transposition:
        transposition_family = any(
            token in cipher_name
            for token in (
                "railfence", "rail_fence", "redefence", "redefense",
                "myszkowski", "amsco", "columnar", "nihilist", "cadenus",
            )
        )
        content_suspicious = False
        if not transposition_family and word_groups <= 1 and alphabet_size <= pt_alpha.size:
            try:
                from analysis.transposition_solver import transposition_suspicion

                content_suspicious = bool(
                    transposition_suspicion(cipher_text, language)["suspicious"]
                )
            except Exception:  # noqa: BLE001 — never let detection break routing
                content_suspicious = False
        if transposition_family or content_suspicious:
            return {
                "route": "transposition",
                "solver": "transposition_permutation_search",
                "reason": (
                    f"cipher_system={cipher_system or 'unknown'}"
                    if transposition_family
                    else "monogram distribution matches language by letter"
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
    include_turning_mask_routes = _env_bool("DECIPHER_PURE_TRANSPOSITION_INCLUDE_TURNING_MASK_ROUTES", default=True)
    include_block_routes = _env_bool("DECIPHER_PURE_TRANSPOSITION_INCLUDE_BLOCK_ROUTES", default=True)
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
        include_turning_mask_routes=include_turning_mask_routes,
        include_block_routes=include_block_routes,
        transmatrix_min_width=transmatrix_min_width,
        transmatrix_max_width=transmatrix_max_width,
        threads=threads,
    )
    best = result.get("best_candidate")
    if not best:
        raise ValueError("pure transposition screen produced no candidate")
    pure_top_candidates = list(result.get("top_candidates") or [])
    # Additive artifact enrichment: attach a normalized candidate packet per row.
    for pure_rank, pure_row in enumerate(pure_top_candidates, start=1):
        pure_row["packet"] = packet_from_pure_transposition_row(
            pure_row, rank=pure_rank
        ).to_dict()
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
        "top_candidates": pure_top_candidates,
        "note": (
            "Broad Rust-scored pure-transposition screen. It includes K3-style "
            "TransMatrix candidates plus grid/route/columnar families, and "
            "scores transformed text directly. It is separate from the "
            "transform+homophonic Z340 path."
        ),
    }
    # Additive: the Rust screen covers route/grid/matrix families but never
    # searches keyword-COLUMN orderings, so it leaves real columnar/nihilist at
    # the monoalphabetic floor. Run the permutation-search transposition solver
    # and keep whichever decryption reads more like the target language.
    rust_plaintext = str(best.get("plaintext") or "")
    solver_name = str(result.get("solver") or "pure_transposition_screen_rust")
    decryption = rust_plaintext
    # The permutation search adds keyed-COLUMN coverage (real columnar /
    # nihilist). The Rust screen already owns route/grid/matrix families
    # (route, K3 TransMatrix), so skip the extra work there.
    _perm_search_skip = ("route", "transmatrix", "kryptos", "k3")
    if not any(token in cipher_system.lower() for token in _perm_search_skip):
        try:
            from analysis.transposition_solver import full_score, solve_transposition

            perm = solve_transposition(
                cipher_text, language=language, family_hint=cipher_system,
            )
            if perm.get("status") == "completed":
                perm_plaintext = str(perm.get("plaintext") or "")
                rust_score = full_score(rust_plaintext, language) if rust_plaintext else float("-inf")
                perm_score = full_score(perm_plaintext, language) if perm_plaintext else float("-inf")
                step["permutation_search"] = {
                    "family": perm.get("family"),
                    "params": perm.get("params"),
                    "score": perm.get("score"),
                    "dict_rate": perm.get("dict_rate"),
                    "strategies_run": perm.get("strategies_run"),
                    "elapsed_seconds": perm.get("elapsed_seconds"),
                    "rust_full_score": round(rust_score, 6) if rust_score != float("-inf") else None,
                    "permutation_full_score": round(perm_score, 6) if perm_score != float("-inf") else None,
                    "adopted": perm_score > rust_score,
                    "preview": perm_plaintext[:120],
                }
                if perm_score > rust_score:
                    decryption = perm_plaintext
                    solver_name = "transposition_permutation_search"
        except Exception as exc:  # noqa: BLE001 — never let the additive solver break the run
            step["permutation_search"] = {"status": "error", "error": str(exc)[:200]}
    return solver_name, {}, decryption, step


def _run_transposition_solver(
    cipher_text: CipherText,
    language: str,
    cipher_system: str = "",
    solver_hints: dict[str, Any] | None = None,
) -> tuple[str, dict[int, int], str, dict[str, Any]]:
    """Run the permutation-search transposition solver (keyed-column families).

    Covers keyword columnar, railfence/redefence, myszkowski, amsco, and
    nihilist transposition, scored by the existing dictionary + n-gram language
    model and bounded by a time/candidate budget so a run never hangs.
    """

    from analysis.transposition_solver import solve_transposition

    result = solve_transposition(
        cipher_text, language=language, family_hint=cipher_system,
    )
    step = {
        "name": "solve_transposition",
        "solver": result.get("solver", "transposition_solver"),
        "status": result.get("status"),
        "cipher_system": cipher_system,
        "family": result.get("family"),
        "params": result.get("params"),
        "score": result.get("score"),
        "dict_rate": result.get("dict_rate"),
        "budget_seconds": result.get("budget_seconds"),
        "strategies_run": result.get("strategies_run"),
        "candidate_count": result.get("candidate_count"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "top_candidates": result.get("candidates"),
        "note": result.get("note"),
    }
    if result.get("status") != "completed":
        raise ValueError(
            str(result.get("reason") or "transposition solver produced no candidate")
        )
    return (
        "transposition_permutation_search",
        {},
        str(result.get("plaintext") or ""),
        step,
    )


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
    initial_key: dict[int, int] | None = None,
    fixed_cipher_ids: set[int] | None = None,
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
    bin_path = _zenith_native_model_path(language)

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
            initial_key=initial_key,
            fixed_cipher_ids=fixed_cipher_ids,
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
            initial_key=initial_key,
            fixed_cipher_ids=fixed_cipher_ids,
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
        if "binary_ngram_mean_log_prob" not in diagnostics:
            binary_score = _zenith_text_mean_log_prob(candidate.plaintext, bin_path)
            if binary_score is not None:
                diagnostics["binary_ngram_mean_log_prob"] = round(binary_score, 6)
                diagnostics["binary_ngram_model_source"] = str(bin_path)
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
        "initial_key_provided": initial_key is not None,
        "fixed_cipher_ids_count": len(fixed_cipher_ids or set()),
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


def _is_null_mask_refinement(refinement: str) -> bool:
    return (refinement or "none").strip().lower() in {
        "null_masks",
        "homophonic_nulls",
        "null_topn",
        # Backward-compatible aliases from the initial Copiale-specific probe.
        "copiale_nulls",
        "copiale_null_masks",
        "copiale_null_topn",
    }


def _is_word_repair_refinement(refinement: str) -> bool:
    """True for the Phase-2b word-repair refinements (plain + composite)."""
    return (refinement or "none").strip().lower() in {
        "word_repair",
        "null_masks+word_repair",
    }


def _refinement_runs_null_masks(refinement: str) -> bool:
    """True when the refinement includes the null-mask bakeoff stage.

    Covers the standalone null-mask values plus the ``null_masks+word_repair``
    composite, whose null-mask bakeoff runs before word repair.
    """
    key = (refinement or "none").strip().lower()
    return _is_null_mask_refinement(refinement) or key == "null_masks+word_repair"


# DECIPHER_WORD_REPAIR_* env vars map 1:1 onto the enumerated WordRepairConfig
# fields. Numeric values are parsed with int()/float(); a malformed value raises
# ValueError, matching the DECIPHER_NULL_MASK_* convention (int() there raises
# too). Unset vars keep the library (probe-CLI) defaults.
_WORD_REPAIR_ENV_INT_FIELDS: dict[str, str] = {
    "window_size": "DECIPHER_WORD_REPAIR_WINDOW_SIZE",
    "window_step": "DECIPHER_WORD_REPAIR_WINDOW_STEP",
    "min_word_len": "DECIPHER_WORD_REPAIR_MIN_WORD_LEN",
    "max_word_len": "DECIPHER_WORD_REPAIR_MAX_WORD_LEN",
    "max_edits": "DECIPHER_WORD_REPAIR_MAX_EDITS",
    "max_hypotheses": "DECIPHER_WORD_REPAIR_MAX_HYPOTHESES",
    "max_hypotheses_per_window": "DECIPHER_WORD_REPAIR_MAX_HYPOTHESES_PER_WINDOW",
}
_WORD_REPAIR_ENV_FLOAT_FIELDS: dict[str, str] = {
    "acceptance_margin": "DECIPHER_WORD_REPAIR_ACCEPTANCE_MARGIN",
    "min_page_drop": "DECIPHER_WORD_REPAIR_MIN_PAGE_DROP",
    "max_illusion_increase": "DECIPHER_WORD_REPAIR_MAX_ILLUSION_INCREASE",
}


def _word_repair_config_from_env() -> tuple[Any, dict[str, Any]]:
    """Return ``(WordRepairConfig, env_overrides)`` for the word-repair pipeline.

    Parses the ``DECIPHER_WORD_REPAIR_*`` surface once. Returns the effective
    config plus the subset of env vars that were actually set (for the artifact
    step). A malformed numeric value raises ``ValueError``.
    """
    from dataclasses import replace

    # Lazy import (see _run_word_repair_refinement for the circular-import
    # rationale): analysis.word_hypothesis_repair pulls in analysis.multipage,
    # which imports automated.runner at module top.
    from analysis.word_hypothesis_repair import WordRepairConfig

    overrides: dict[str, Any] = {}
    env_used: dict[str, Any] = {}
    for field_name, env_name in _WORD_REPAIR_ENV_INT_FIELDS.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw.strip() != "":
            value = int(raw)
            overrides[field_name] = value
            env_used[env_name] = value
    for field_name, env_name in _WORD_REPAIR_ENV_FLOAT_FIELDS.items():
        raw = os.environ.get(env_name)
        if raw is not None and raw.strip() != "":
            value = float(raw)
            overrides[field_name] = value
            env_used[env_name] = value
    config = replace(WordRepairConfig(), **overrides) if overrides else WordRepairConfig()
    return config, env_used


_WORD_REPAIR_ADOPT_TRUE_VALUES = {"1", "true", "yes", "on"}


def _word_repair_adopt_enabled() -> bool:
    """Strict, fail-closed parse of ``DECIPHER_WORD_REPAIR_ADOPT``.

    This flag guards the menu-only safety default (spec item 3 FINAL) — the
    property that forced two spec amendments — so unlike the repo-wide
    ``_env_bool`` convention (where anything outside the falsy set counts as
    true), adoption enables ONLY on an explicit affirmative value
    (case/whitespace tolerant). Unset, empty, or garbage values all mean
    disabled.
    """
    raw = os.environ.get("DECIPHER_WORD_REPAIR_ADOPT", "")
    return raw.strip().lower() in _WORD_REPAIR_ADOPT_TRUE_VALUES


# ---------------------------------------------------------------------------
# Opt-in LLM finalist reranker (improvement program Phase 4a, item 4.2).
#
# ``DECIPHER_FINALIST_READER`` turns on an LLM "reader" that scores the finalist
# menu (null-mask bakeoff, word-repair) purely by text quality and records a
# ``finalist_reader`` block on the step. It ANNOTATES by default; only when
# ``DECIPHER_FINALIST_READER_SELECTS=1`` may the reader's top pick override the
# scalar selection — and only on the null-mask route (mirroring the menu-only
# lesson: measure first). The reranker is entirely lazy-imported and degrades
# gracefully: a missing key/network or any reader error becomes a skipped/error
# block, never a failed run.
# ---------------------------------------------------------------------------

_FINALIST_READER_SELECTS_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class _FinalistReaderSpec:
    provider: str
    model: str


def _parse_finalist_reader_spec() -> _FinalistReaderSpec | None:
    """Strict, fail-closed parse of ``DECIPHER_FINALIST_READER``.

    Unset/empty means the reader is off (returns ``None``). A non-empty value
    must look like ``llm:MODEL`` (provider inferred from the model id) or
    ``llm:PROVIDER:MODEL`` (explicit provider); anything else raises
    ``ValueError`` so a typo fails loudly at the parse boundary rather than
    silently running a wrong config. The runner hook wraps this call so a bad
    value still cannot crash a run.
    """
    from agent.model_provider import canonical_provider, infer_provider_from_model

    raw = os.environ.get("DECIPHER_FINALIST_READER", "").strip()
    if not raw:
        return None
    prefix, sep, rest = raw.partition(":")
    if prefix.strip().lower() != "llm" or not sep:
        raise ValueError(
            "DECIPHER_FINALIST_READER must look like 'llm:MODEL' or "
            f"'llm:PROVIDER:MODEL', got {raw!r}"
        )
    rest = rest.strip()
    if not rest:
        raise ValueError("DECIPHER_FINALIST_READER is missing a model id")

    provider: str | None = None
    model = rest
    maybe_provider, colon, maybe_model = rest.partition(":")
    if colon and maybe_model.strip():
        try:
            provider = canonical_provider(maybe_provider.strip())
            model = maybe_model.strip()
        except ValueError:
            provider = None
            model = rest
    if provider is None:
        provider = infer_provider_from_model(model)
    return _FinalistReaderSpec(provider=provider, model=model)


def _finalist_reader_selects_enabled() -> bool:
    """Strict, fail-closed parse of ``DECIPHER_FINALIST_READER_SELECTS``.

    Selection override enables ONLY on an explicit affirmative value; unset,
    empty, or garbage all mean annotate-only (the safe default).
    """
    raw = os.environ.get("DECIPHER_FINALIST_READER_SELECTS", "")
    return raw.strip().lower() in _FINALIST_READER_SELECTS_TRUE_VALUES


def _finalist_reader_api_key(provider: str) -> str:
    """Resolve the reader's API key via the existing CLI pathway, silently.

    Uses ``cli._probe_api_key`` (the non-exiting sibling of
    ``cli.get_api_key``) so a missing key degrades to graceful skip instead of
    ``get_api_key``'s ``sys.exit(1)``.
    """
    from agent.model_provider import canonical_provider

    resolved = canonical_provider(provider)
    if resolved == "ollama":
        return ""
    try:
        from cli import _probe_api_key

        return _probe_api_key(resolved)
    except Exception:  # noqa: BLE001 — never fail the run over key resolution
        return ""


def _finalist_reader_rank(
    packets: list[Any],
    *,
    language: str,
    allow_select: bool,
) -> tuple[dict[str, Any] | None, int | None]:
    """Run the opt-in reader over ``packets`` and build a ``finalist_reader`` block.

    Returns ``(block, override_index)``. ``block`` is ``None`` iff the reader is
    disabled (env unset). ``override_index`` is the 0-based index into
    ``packets`` the reader prefers, but only when ``allow_select`` and
    ``DECIPHER_FINALIST_READER_SELECTS=1`` and the pick differs from the scalar
    winner (index 0); otherwise ``None``. Any error yields a skipped/error block
    and no override — the reader can never fail the run.
    """
    try:
        spec = _parse_finalist_reader_spec()
    except ValueError as exc:
        return (
            {
                "status": "error",
                "enabled": True,
                "reason": f"invalid_env: {exc}",
            },
            None,
        )
    if spec is None:
        return None, None

    selects_enabled = _finalist_reader_selects_enabled()
    mode = "select" if (allow_select and selects_enabled) else "annotate"
    base_block: dict[str, Any] = {
        "enabled": True,
        "provider": spec.provider,
        "model": spec.model,
        "mode": mode,
        "selects_enabled": selects_enabled,
        "allow_select": allow_select,
    }

    if not packets:
        return {**base_block, "status": "skipped", "reason": "no_candidates"}, None

    api_key = _finalist_reader_api_key(spec.provider)
    if not api_key and spec.provider != "ollama":
        return (
            {**base_block, "status": "skipped", "reason": "no_api_key"},
            None,
        )

    try:
        from analysis.llm_reader import (
            LLMReaderConfig,
            _candidate_original_id,
            rank_candidates,
        )

        config = LLMReaderConfig(
            provider=spec.provider,
            model=spec.model,
            language=language,
        )
        result = rank_candidates(packets, config, api_key=api_key)
    except Exception as exc:  # noqa: BLE001 — never fail the run over the reader
        return (
            {**base_block, "status": "skipped", "reason": f"reader_error: {exc}"},
            None,
        )

    result_dict = result.to_dict()
    block: dict[str, Any] = {
        **base_block,
        "status": "completed" if result.parse_ok else "unparsed",
        "top_n": len(result.scores),
        "scores": result_dict["scores"],
        "ranking": result_dict["ranking"],
        "reader_best_candidate_id": result.best_candidate_id,
        "reader_best_index": result.best_index,
        "usage": result_dict["usage"],
        "parse_ok": result.parse_ok,
        "unscored": result_dict["unscored"],
        "reader_error": result.error,
    }

    # Resolve the override into ``packets`` by matching the reader's mapped-back
    # candidate id, NOT ``result.best_index``. ``best_index`` indexes the PREPARED
    # list (after textless candidates are silently dropped), which is misaligned
    # with ``packets`` whenever a dropped candidate precedes the reader's pick;
    # indexing ``packets``/``top_finalists`` by it would install the wrong row.
    override_index: int | None = None
    if (
        allow_select
        and selects_enabled
        and result.parse_ok
        and result.best_candidate_id is not None
    ):
        for original_index, packet in enumerate(packets):
            if _candidate_original_id(packet, original_index) == result.best_candidate_id:
                if original_index != 0:
                    override_index = original_index
                break
    block["selection_overridden"] = override_index is not None
    if override_index is not None:
        block["overridden_from_index"] = 0
        block["overridden_to_index"] = override_index
    return block, override_index


def _parse_word_repair_edit_label(label: str) -> tuple[str, str] | None:
    """Parse a ``"<symbol>:<before>-><target>"`` variant edit label.

    Returns ``(symbol, target)`` or ``None`` if the label is malformed. Only
    single-letter A-Z targets are applied (word hypotheses are dictionary
    words, so ``propose_word_repairs`` never emits ``<null>`` targets).
    """
    symbol, sep, rest = str(label).partition(":")
    if not sep:
        return None
    _before, arrow, target = rest.partition("->")
    if not arrow:
        return None
    target = target.strip()
    if len(target) != 1 or not ("A" <= target <= "Z"):
        return None
    return symbol, target


@dataclass
class WordRepairMenu:
    """A page-group word-repair candidate menu.

    Shared return type for :func:`build_word_repair_menu` (single-page) and
    :func:`build_word_repair_menu_for_pages` (group-native) so the Phase-2b
    runner refinement, the Phase-2.4 multipage route, and the agent-facing
    ``search_word_repair_menu`` tool consume the same construction. ``packets``
    are :class:`analysis.candidate_packet.CandidatePacket` instances
    (``text=None`` per the F3 deferral); ``page`` (the group's first page --
    THE page for single-page callers) and ``alphabet`` are returned so the
    runner can project the adopted edit set without rebuilding the group.
    """

    packets: list[Any]
    baseline_validation: float
    page: Any
    alphabet: Any


def _single_page_group(cipher_text: CipherText) -> tuple[list[Any], Alphabet]:
    """Build the length-1 page group for single-page word-repair callers.

    Single source of truth for the Phase-2b page construction, shared by
    :func:`build_word_repair_menu` (agent tool path) and
    :func:`_run_word_repair_refinement` (runner refinement path).

    Lazy import (binding constraint 1): ``analysis.multipage`` imports
    ``automated.runner`` at module top and this module's public wrappers sit at
    EOF, so a top-of-file import would hit a partially-initialized module.
    """
    from analysis.multipage import PageBundle

    alphabet = cipher_text.alphabet
    symbols = [alphabet.symbol_for(token_id) for token_id in cipher_text.tokens]
    # plaintext is intentionally empty: the runtime path never reads it, and
    # keeping it blank guarantees no ground truth can enter the word-repair menu.
    page = PageBundle(
        test_id="page_0",
        canonical_transcription=" ".join(symbols),
        plaintext="",
        symbols=symbols,
        token_ids=list(cipher_text.tokens),
    )
    return [page], alphabet


def build_word_repair_menu(
    *,
    cipher_text: CipherText,
    base_key: dict[int, int],
    mask: tuple[str, ...],
    language: str,
    config: Any,
    dictionary_path: Any,
    model_path: Any,
    source_branch: str | None,
) -> WordRepairMenu:
    """Build the single-page word-repair menu (page group + baseline + packets).

    This is the exact Phase-2b construction: a length-1 group of one
    :class:`PageBundle` built from the ciphertext + the branch's effective
    key/mask, a baseline projection scored with ``score_page_runtime``, and the
    adjudicated candidate packets from ``propose_word_repairs``. The agent
    ``search_word_repair_menu`` tool calls this so the two paths cannot drift.

    Thin single-page wrapper over :func:`build_word_repair_menu_for_pages`
    (the group-native core the Phase-2.4 multipage route calls directly).
    """
    pages, alphabet = _single_page_group(cipher_text)
    return build_word_repair_menu_for_pages(
        pages=pages,
        alphabet=alphabet,
        base_key=base_key,
        mask=mask,
        language=language,
        config=config,
        dictionary_path=dictionary_path,
        model_path=model_path,
        source_branch=source_branch,
    )


def build_word_repair_menu_for_pages(
    *,
    pages: list[Any],
    alphabet: Any,
    base_key: dict[int, int],
    mask: tuple[str, ...],
    language: str,
    config: Any,
    dictionary_path: Any,
    model_path: Any,
    source_branch: str | None,
) -> WordRepairMenu:
    """Group-native word-repair menu construction (Phase-2.4 reconciliation).

    The shared core behind :func:`build_word_repair_menu`: a baseline
    projection scored with ``score_page_runtime``, and the adjudicated
    candidate packets from ``propose_word_repairs`` over the WHOLE page group
    (cross-page collateral is real when the group has more than one page).
    A length-1 ``pages`` list reproduces the single-page construction exactly.

    The baseline validation is scored on the group's first page's projection
    -- identical to the single-page construction (where it is THE page) and to
    the Phase-2.4 group behavior as shipped. The library's own
    ``repair_acceptance`` verdict (carried on each packet) already compares
    group-level metric deltas across all pages.

    Lazy imports (binding constraint 1): ``analysis.multipage`` imports
    ``automated.runner`` at module top and this module's public wrappers sit at
    EOF, so a top-of-file import would hit a partially-initialized module.
    """
    from analysis.multipage import project_pages, score_page_runtime
    from analysis.word_hypothesis_repair import propose_word_repairs

    baseline_row = project_pages(pages=pages, key=base_key, mask=mask)[0]
    baseline_runtime = score_page_runtime(
        baseline_row, key=base_key, mask=mask, language=language, model_path=model_path
    )
    baseline_validation = round(float(baseline_runtime.get("validation_score_v2") or 0.0), 6)
    packets = propose_word_repairs(
        pages=pages,
        shared_key=base_key,
        dictionary_path=dictionary_path,
        language=language,
        config=config,
        mask=mask,
        alphabet=alphabet,
        source_branch=source_branch,
        model_path=model_path,
    )
    return WordRepairMenu(
        packets=packets,
        baseline_validation=baseline_validation,
        page=pages[0],
        alphabet=alphabet,
    )


def apply_word_repair_edits(
    *,
    base_key: dict[int, int],
    edits: list[str],
    alphabet: Any,
    mask: tuple[str, ...],
) -> tuple[dict[int, int] | None, list[str], str]:
    """Whole-candidate word-repair edit application with the masked-symbol guard.

    Returns ``(new_key, applied_labels, reason)``. Mirrors the Phase-2b rule: if
    ANY edit label fails to parse/apply (unparseable, unknown symbol) or targets
    a symbol in the active mask, the ENTIRE candidate is rejected -- adopting a
    subset of a scored edit set would apply a variant the library never
    evaluated, and a masked symbol never appears in the scored projection. On
    rejection ``new_key`` is ``None``, ``applied_labels`` is empty, and
    ``reason`` is ``"no_applicable_edits"``. On success ``new_key`` is a fresh
    dict (``base_key`` is never mutated) and ``reason`` is ``""``.
    """
    masked_symbols = set(mask)
    if not edits:
        return None, [], "no_applicable_edits"
    new_key = dict(base_key)
    applied: list[str] = []
    for label in edits:
        parsed = _parse_word_repair_edit_label(label)
        if parsed is None:
            return None, [], "no_applicable_edits"
        symbol, target = parsed
        if not alphabet.has_symbol(symbol):
            return None, [], "no_applicable_edits"
        if symbol in masked_symbols:
            return None, [], "no_applicable_edits"
        new_key[alphabet.id_for(symbol)] = ord(target) - ord("A")
        applied.append(label)
    return new_key, applied, ""


def _run_word_repair_refinement(
    *,
    cipher_text: CipherText,
    language: str,
    refinement: str,
    base_solver: str,
    base_key: dict[int, int],
    base_decryption: str,
    mask: tuple[str, ...],
) -> tuple[dict[str, Any], tuple[str, dict[int, int], str] | None]:
    """Single-page adapter for :func:`_word_repair_refinement_on_pages`.

    Builds the length-1 page group (via :func:`_single_page_group`, shared with
    ``build_word_repair_menu``) from the run's cipher + solved key/mask and
    delegates to the group-native refinement. Kept as a thin, signature-stable
    wrapper (``base_decryption`` is accepted for call-site/test compatibility
    and is unused) so the single-page ``run_automated`` call site is unchanged
    while the Phase-2.4 multi-page route reuses the exact same gate on a real
    page group.
    """
    pages, alphabet = _single_page_group(cipher_text)
    return _word_repair_refinement_on_pages(
        pages=pages,
        alphabet=alphabet,
        language=language,
        refinement=refinement,
        base_solver=base_solver,
        base_key=base_key,
        mask=mask,
    )


def _word_repair_refinement_on_pages(
    *,
    pages: list[Any],
    alphabet: Alphabet,
    language: str,
    refinement: str,
    base_solver: str,
    base_key: dict[int, int],
    mask: tuple[str, ...],
) -> tuple[dict[str, Any], tuple[str, dict[int, int], str] | None]:
    """Run Phase-2a ``propose_word_repairs`` on a solved page group's basin.

    Proposes word-hypothesis repairs across the whole page group and evaluates
    the composed ground-truth-free gate (library repair-acceptance verdict AND
    strict ``validation_score_v2`` improvement). By default the gate is
    measurement-only: the ``search_word_repair`` step records the candidate
    menu, per-candidate gate decisions, and ``would_adopt`` (the candidate the
    gate selected, or None) without modifying the run. When
    ``DECIPHER_WORD_REPAIR_ADOPT=1`` the gate's selection is applied and
    ``(solver, key, decryption)`` is returned to swap in (the returned
    decryption is the first page's projection; group callers re-project every
    page from the adopted key).

    ``pages`` is a list of :class:`~analysis.multipage.PageBundle`; a length-1
    list reproduces the single-page behavior exactly. Cross-page collateral is
    real when the group has more than one page -- the library's cross-page
    adjudication was designed for exactly that case (Phase-2.4).
    """
    started = time.time()
    # Lazy import (binding constraint 1): analysis.multipage imports
    # automated.runner at module top, and this module's public wrappers sit at
    # EOF, so a top-of-file import would hit a partially-initialized module.
    # ``build_word_repair_menu_for_pages`` does its own lazy imports; the
    # adoption projection below still needs ``project_pages`` directly.
    from analysis.multipage import project_pages

    config, env_overrides = _word_repair_config_from_env()
    # Route model resolution through the runner's repo-root-anchored resolver so
    # library scoring never falls back to a CWD-relative models/ngram5_*.bin.
    resolved_model = zenith_native_model_path(language)
    dictionary_path = dictionary.get_dictionary_path(language)

    def _config_dict() -> dict[str, Any]:
        return {
            field_name: getattr(config, field_name)
            for field_name in _WORD_REPAIR_ENV_INT_FIELDS
        } | {
            field_name: getattr(config, field_name)
            for field_name in _WORD_REPAIR_ENV_FLOAT_FIELDS
        }

    def _skip_step(reason: str) -> dict[str, Any]:
        return {
            "name": "search_word_repair",
            "status": "skipped",
            "experimental": True,
            "mode": refinement,
            "base_solver": base_solver,
            "mask": list(mask),
            "language": language,
            "reason": reason,
            "effective_config": _config_dict(),
            "config_env_overrides": env_overrides,
            "binary_ngram_model": (
                _zenith_native_model_metadata(str(resolved_model)) if resolved_model else None
            ),
            "adopt_enabled": _word_repair_adopt_enabled(),
            "would_adopt": None,
            "adopted": None,
            "candidate_menu": [],
            "elapsed_seconds": round(time.time() - started, 3),
        }

    if not dictionary_path:
        return _skip_step("no_dictionary_for_language"), None
    if not base_key:
        return _skip_step("empty_base_key"), None

    # Menu construction is shared with the agent search_word_repair_menu tool
    # via build_word_repair_menu_for_pages (single source of truth for the
    # baseline projection + runtime score + adjudicated packets), called here
    # group-native so the Phase-2.4 multi-page route reuses it unchanged.
    menu = build_word_repair_menu_for_pages(
        pages=pages,
        alphabet=alphabet,
        base_key=base_key,
        mask=mask,
        language=language,
        config=config,
        dictionary_path=dictionary_path,
        model_path=resolved_model,
        source_branch=base_solver,
    )
    baseline_validation = menu.baseline_validation
    packets = menu.packets

    def _packet_validation(packet: Any) -> float:
        value = (packet.solver_scores or {}).get("page_validation_avg")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    def _packet_acceptance(packet: Any) -> dict[str, Any]:
        """The library's repair_acceptance verdict (carried in packet.validation)."""
        return packet.validation if isinstance(packet.validation, dict) else {}

    def _packet_adjudication_score(packet: Any) -> float | None:
        value = (packet.solver_scores or {}).get("adjudication_score")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # Composed gate (spec item 3, FINAL 2026-07-13): evaluate iff BOTH
    # (a) the library's repair-acceptance verdict accepts the candidate
    #     (annotate_acceptance/repair_acceptance with its own margins, computed
    #     inside propose_word_repairs and carried on packet.validation), AND
    # (b) validation_score_v2 strictly improves over the baseline projection.
    #
    # The gate is MEASUREMENT-ONLY by default: the rerun evidence showed no
    # available GT-free signal safely auto-adopts single-page repairs (the
    # verdict accepted all five packet-page candidates including one with
    # adjudication_score -3.00, and adjudication sign does not separate the
    # harmful adoptions). The step always records `would_adopt`; the key/
    # decryption are modified only when DECIPHER_WORD_REPAIR_ADOPT=1 (research
    # + multipage experiments). Menu consumers are Phase 2.5 agent review.
    adopt_enabled = _word_repair_adopt_enabled()
    adoption_epsilon = 1e-6
    improving = [
        packet for packet in packets
        if _packet_validation(packet) > baseline_validation + adoption_epsilon
    ]
    verdict_accepted = [
        packet for packet in packets
        if bool(_packet_acceptance(packet).get("accepted"))
    ]
    passing = [
        packet for packet in improving
        if bool(_packet_acceptance(packet).get("accepted"))
    ]
    best = max(passing, key=_packet_validation) if passing else None
    best_validation = _packet_validation(best) if best is not None else baseline_validation

    # Gate evaluation (always recorded, never mutates the run by itself).
    would_adopt: dict[str, Any] | None = None
    gate_reason = ""
    applied: list[str] = []
    new_key: dict[int, int] | None = None
    if not packets:
        gate_reason = "no_repairs_proposed"
    elif best is None:
        gate_reason = "no_candidate_passed_composed_gate"
    else:
        edit_labels = list((best.provenance or {}).get("edits") or [])
        # Whole-candidate applicability + masked-symbol guard (shared with the
        # agent install tool): if ANY edit label fails to parse/apply or targets
        # a masked symbol, the entire candidate is rejected — adopting a subset
        # of a scored edit set would apply a variant the library never
        # evaluated, and a masked symbol never appears in the scored projection.
        new_key, applied, _edit_reason = apply_word_repair_edits(
            base_key=base_key, edits=edit_labels, alphabet=alphabet, mask=mask
        )
        if new_key is None:
            gate_reason = "no_applicable_edits"
        else:
            gate_reason = "composed_gate_passed"
            would_adopt = {
                "edits": list(applied),
                "candidate_id": best.candidate_id,
                "preview": (best.preview or "")[:180],
                # Both composed-gate signals, explicit on the would-adopt entry.
                "acceptance": {
                    "accepted": bool(_packet_acceptance(best).get("accepted")),
                    "decision": _packet_acceptance(best).get("decision"),
                    "reasons": list(_packet_acceptance(best).get("reasons") or []),
                },
                "adjudication_score": _packet_adjudication_score(best),
                "page_validation_avg": round(best_validation, 6),
            }

    # Adoption (opt-in): apply the would-adopt candidate only when enabled.
    adopted_result: tuple[str, dict[int, int], str] | None = None
    adopted_edits: list[str] | None = None
    after_validation = baseline_validation
    if not adopt_enabled:
        adopted_reason = "menu_only_default"
    elif would_adopt is None:
        adopted_reason = gate_reason
    else:
        assert new_key is not None  # gate_reason == composed_gate_passed
        new_decryption = project_pages(pages=pages, key=new_key, mask=mask)[0]["decryption"]
        adopted_result = ("word_repair_homophonic", new_key, new_decryption)
        adopted_edits = list(applied)
        after_validation = best_validation
        adopted_reason = gate_reason

    def _gate_decision(packet: Any) -> dict[str, Any]:
        """Both gate signals, recorded for every candidate (adopted or rejected)."""
        acceptance = _packet_acceptance(packet)
        validation_value = _packet_validation(packet)
        improves = validation_value > baseline_validation + adoption_epsilon
        accepted = bool(acceptance.get("accepted"))
        return {
            "candidate_id": packet.candidate_id,
            "edits": list((packet.provenance or {}).get("edits") or []),
            "page_validation_avg": validation_value if math.isfinite(validation_value) else None,
            "validation_improves": improves,
            "acceptance_accepted": accepted,
            "acceptance_decision": acceptance.get("decision"),
            "adjudication_score": _packet_adjudication_score(packet),
            "passed_composed_gate": improves and accepted,
            "adopted": (
                adopted_result is not None
                and best is not None
                and packet.candidate_id == best.candidate_id
            ),
        }

    proposed = len(packets)
    adopted_count = 1 if adopted_result is not None else 0
    step = {
        "name": "search_word_repair",
        "status": "completed",
        "experimental": True,
        "policy": (
            "Opt-in word-hypothesis repair for homophonic ciphers. Same-length "
            "dictionary-word repairs are proposed on the solved single-page "
            "basin from ciphertext + the solver key only. The composed "
            "ground-truth-free gate (library repair-acceptance verdict AND "
            "strict validation_score_v2 improvement) is measurement-only by "
            "default: the step records would_adopt without modifying the key/"
            "decryption. Set DECIPHER_WORD_REPAIR_ADOPT=1 to apply the gate's "
            "selection (research/multipage experiments)."
        ),
        "mode": refinement,
        "base_solver": base_solver,
        "mask": list(mask),
        "language": language,
        "effective_config": _config_dict(),
        "config_env_overrides": env_overrides,
        "binary_ngram_model": (
            _zenith_native_model_metadata(str(resolved_model)) if resolved_model else None
        ),
        "adoption_epsilon": adoption_epsilon,
        "adopt_enabled": adopt_enabled,
        "validation_before": baseline_validation,
        "validation_after": round(float(after_validation), 6),
        "validation_delta": round(float(after_validation) - baseline_validation, 6),
        # What the composed gate selected (or None) — recorded in every mode so
        # artifacts keep measuring the gate even when it does not act.
        "would_adopt": would_adopt,
        "would_adopt_reason": gate_reason,
        "adopted": (
            {
                "solver": adopted_result[0],
                **would_adopt,
            }
            if adopted_result is not None and would_adopt is not None
            else None
        ),
        "adopted_reason": adopted_reason,
        "counts": {
            # propose_word_repairs returns the post-prescreen, adjudicated menu,
            # so proposed == prescreened == adjudicated; rejected = proposed
            # minus the single adopted candidate.
            "proposed": proposed,
            "prescreened": proposed,
            "adjudicated": proposed,
            "improving": len(improving),
            "verdict_accepted": len(verdict_accepted),
            "passed_composed_gate": len(passing),
            "adopted": adopted_count,
            "rejected": proposed - adopted_count,
        },
        # Both gate signals for every candidate, adopted or rejected (compact;
        # the full packets are in candidate_menu).
        "gate_decisions": [_gate_decision(packet) for packet in packets],
        "candidate_menu": [packet.to_dict() for packet in packets],
        "elapsed_seconds": round(time.time() - started, 3),
    }

    # Opt-in LLM finalist reader (Phase 4a item 4.2): the word-repair route is
    # ANNOTATE-ONLY (selection override is the null-mask route only), so the
    # reader records evidence beside the composed gate without changing
    # would_adopt. Disabled reader -> no key added.
    finalist_reader_block, _ = _finalist_reader_rank(
        list(packets),
        language=language,
        allow_select=False,
    )
    if finalist_reader_block is not None:
        step["finalist_reader"] = finalist_reader_block

    return step, adopted_result


def _run_null_mask_bakeoff(
    *,
    cipher_text: CipherText,
    language: str,
    budget: str,
    solver_profile: str,
    base_solver: str,
    base_key: dict[int, int],
    base_decryption: str,
    base_step: dict[str, Any],
) -> dict[str, Any]:
    """Experimental top-N null-mask finalist menu for homophonic ciphers.

    This is opt-in via ``homophonic_refinement=null_masks``. It does not use
    benchmark plaintext: candidates are generated from ciphertext plus the
    solver-produced baseline key, and ranked by language readability/collapse
    signals.
    """
    started = time.time()
    pt_alpha = _plaintext_alphabet(language)
    plaintext_ids = list(range(pt_alpha.size))
    id_to_letter = {i: pt_alpha.symbol_for(i).upper() for i in range(pt_alpha.size)}
    word_list = _word_list(language)
    profile = os.environ.get("DECIPHER_NULL_MASK_PROFILE", "wide").strip().lower()
    if profile in {"default", "standard"}:
        profile = "wide"
    if profile not in {"narrow", "wide"}:
        profile = "wide"

    def profile_default(name: str, narrow: str, wide: str) -> str:
        return os.environ.get(name, narrow if profile == "narrow" else wide)

    candidate_limit = int(profile_default("DECIPHER_NULL_MASK_CANDIDATE_LIMIT", "24", "48"))
    max_mask_size = int(profile_default("DECIPHER_NULL_MASK_MAX_SIZE", "2", "3"))
    max_masks = int(profile_default("DECIPHER_NULL_MASK_MAX_MASKS", "140", "1500"))
    beam_enabled = os.environ.get("DECIPHER_NULL_MASK_BEAM", "1").strip().lower() not in {"0", "false", "no"}
    beam_width = int(profile_default("DECIPHER_NULL_MASK_BEAM_WIDTH", "10", "36"))
    beam_max_size = int(profile_default("DECIPHER_NULL_MASK_BEAM_MAX_SIZE", "3", "3"))
    beam_max_masks = int(profile_default("DECIPHER_NULL_MASK_BEAM_MAX_MASKS", "60", "500"))
    neighborhood_enabled = os.environ.get("DECIPHER_NULL_MASK_NEIGHBORHOOD", "1").strip().lower() not in {"0", "false", "no"}
    neighborhood_top_n = int(profile_default("DECIPHER_NULL_MASK_NEIGHBORHOOD_TOP_N", "5", "24"))
    neighborhood_max_size = int(
        profile_default(
            "DECIPHER_NULL_MASK_NEIGHBORHOOD_MAX_SIZE",
            str(max(beam_max_size, max_mask_size + 1)),
            "3",
        )
    )
    neighborhood_max_masks = int(profile_default("DECIPHER_NULL_MASK_NEIGHBORHOOD_MAX_MASKS", "80", "500"))
    neighborhood_multi_view = _env_bool("DECIPHER_NULL_MASK_NEIGHBORHOOD_MULTI_VIEW", False)
    consensus_enabled = os.environ.get("DECIPHER_NULL_MASK_CONSENSUS_POLISH", "1").strip().lower() not in {"0", "false", "no"}
    consensus_top_n = int(os.environ.get("DECIPHER_NULL_MASK_CONSENSUS_TOP_N", "5"))
    consensus_min_agreement = float(os.environ.get("DECIPHER_NULL_MASK_CONSENSUS_MIN_AGREEMENT", "0.75"))
    consensus_min_fixed = int(os.environ.get("DECIPHER_NULL_MASK_CONSENSUS_MIN_FIXED", "8"))
    consensus_max_mutable = int(os.environ.get("DECIPHER_NULL_MASK_CONSENSUS_MAX_MUTABLE", "24"))
    consensus_budget = os.environ.get("DECIPHER_NULL_MASK_CONSENSUS_BUDGET", "screen").strip().lower()
    if consensus_budget not in {"screen", "full"}:
        consensus_budget = "screen"
    consensus_multi_view = _env_bool("DECIPHER_NULL_MASK_CONSENSUS_MULTI_VIEW", False)
    adaptive_enabled = _env_bool("DECIPHER_NULL_MASK_ADAPTIVE", False)
    adaptive_min_validation = float(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_MIN_VALIDATION", "0.75"))
    adaptive_near_tie_margin = float(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_NEAR_TIE_MARGIN", "0.05"))
    adaptive_candidate_limit = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_CANDIDATE_LIMIT", "32"))
    adaptive_max_mask_size = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_MAX_SIZE", "2"))
    adaptive_max_masks = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_MAX_MASKS", "320"))
    adaptive_beam_width = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_BEAM_WIDTH", "16"))
    adaptive_beam_max_size = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_BEAM_MAX_SIZE", "3"))
    adaptive_beam_max_masks = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_BEAM_MAX_MASKS", "120"))
    adaptive_bridge_anchor_count = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_BRIDGE_ANCHOR_COUNT", "10"))
    adaptive_bridge_top_rows = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_BRIDGE_TOP_ROWS", "8"))
    adaptive_bridge_max_masks = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_BRIDGE_MAX_MASKS", "160"))
    adaptive_bridge_restarts = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_BRIDGE_RESTARTS", "1"))
    adaptive_bridge_consensus_max_masks = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_BRIDGE_CONSENSUS_MAX_MASKS", "16"))
    adaptive_neighborhood_top_n = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_NEIGHBORHOOD_TOP_N", "10"))
    adaptive_neighborhood_max_size = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_NEIGHBORHOOD_MAX_SIZE", "3"))
    adaptive_neighborhood_max_masks = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_NEIGHBORHOOD_MAX_MASKS", "160"))
    adaptive_consensus_top_n = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_CONSENSUS_TOP_N", "8"))
    adaptive_consensus_budget = os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_CONSENSUS_BUDGET", "screen").strip().lower()
    if adaptive_consensus_budget not in {"screen", "full"}:
        adaptive_consensus_budget = "screen"
    adaptive_stability_top_n = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_STABILITY_TOP_N", "6"))
    adaptive_stability_restarts = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_STABILITY_RESTARTS", "2"))
    adaptive_stability_waves = int(os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_STABILITY_WAVES", "2"))
    adaptive_stability_budget = os.environ.get("DECIPHER_NULL_MASK_ADAPTIVE_STABILITY_BUDGET", "screen").strip().lower()
    if adaptive_stability_budget not in {"screen", "full"}:
        adaptive_stability_budget = "screen"
    language_quality_model_path_raw = os.environ.get("DECIPHER_NULL_MASK_LANGUAGE_QUALITY_MODEL", "").strip()
    language_quality_model = _load_language_quality_model(language_quality_model_path_raw, language=language)
    ranker = os.environ.get("DECIPHER_NULL_MASK_RANKER", "validation").strip().lower()
    if ranker not in {"validation", "ensemble", "language_quality"}:
        ranker = "validation"
    if ranker == "language_quality" and language_quality_model is None:
        ranker = "validation"
    top_n = int(profile_default("DECIPHER_NULL_MASK_TOP_N", "12", "100"))
    promote_top_n = int(profile_default("DECIPHER_NULL_MASK_PROMOTE_TOP_N", "12", "0"))
    promote_reruns = int(profile_default("DECIPHER_NULL_MASK_PROMOTE_RERUNS", "1", "0"))
    promote_budget = profile_default("DECIPHER_NULL_MASK_PROMOTE_BUDGET", "full", "full").strip().lower()
    if promote_budget not in {"screen", "full"}:
        promote_budget = "full"
    confirm_top_n = int(profile_default("DECIPHER_NULL_MASK_CONFIRM_TOP_N", "3", "0"))
    confirm_reruns = int(profile_default("DECIPHER_NULL_MASK_CONFIRM_RERUNS", "2", "0"))
    portfolio_top_n = int(profile_default("DECIPHER_NULL_MASK_PORTFOLIO_TOP_N", "0", "10"))
    portfolio_rankers = [
        item.strip().lower()
        for item in profile_default(
            "DECIPHER_NULL_MASK_PORTFOLIO_RANKERS",
            "validation,ensemble,language_quality",
            "validation,language_quality,ensemble",
        ).split(",")
        if item.strip()
    ]
    portfolio_rankers = [
        item for item in portfolio_rankers
        if item in {"validation", "ensemble", "language_quality"}
    ]
    null_budget = os.environ.get("DECIPHER_NULL_MASK_BUDGET", "screen").strip().lower()
    if null_budget not in {"screen", "full"}:
        null_budget = "screen"
    null_engine = os.environ.get("DECIPHER_NULL_MASK_ENGINE", "rust_batch").strip().lower()
    if null_engine not in {"rust_batch", "python_reference"}:
        null_engine = "rust_batch"
    null_threads = _null_mask_batch_threads()
    binary_model_path = _zenith_native_model_path(language)
    store_evaluated_text = _env_bool("DECIPHER_NULL_MASK_STORE_EVALUATED_TEXT", False)

    base_quality = base_step.get("quality") if isinstance(base_step.get("quality"), dict) else _plaintext_quality(base_decryption, base_key)
    diagnostics = diagnose_cipher_for_null_candidates(
        cipher_text,
        key=base_key,
        id_to_letter=id_to_letter,
        quality=base_quality,
    )
    candidate_symbols = select_null_candidate_symbols(diagnostics, limit=candidate_limit)
    masks = generate_null_masks(candidate_symbols, max_mask_size)
    if max_masks > 0:
        masks = [()] + masks[1:max_masks + 1]

    rows: list[dict[str, Any]] = []

    def finalist_row(
        *,
        mask: tuple[str, ...],
        filtered_length: int,
        solver: str,
        key: dict[int, int],
        decryption: str,
        step: dict[str, Any],
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        quality = step.get("quality") if isinstance(step.get("quality"), dict) else _plaintext_quality(decryption, key)
        diagnostics = (
            dict(step.get("diagnostics"))
            if isinstance(step.get("diagnostics"), dict)
            else _automated_candidate_diagnostics(
                decryption,
                language=language,
                word_list=word_list,
                binary_model_path=binary_model_path,
            )
        )
        if "binary_ngram_mean_log_prob" not in diagnostics:
            binary_score = _zenith_text_mean_log_prob(decryption, binary_model_path)
            if binary_score is not None:
                diagnostics["binary_ngram_mean_log_prob"] = round(binary_score, 6)
                diagnostics["binary_ngram_model_source"] = str(binary_model_path)
        row = {
            "mask": list(mask),
            "mask_size": len(mask),
            "filtered_length": filtered_length,
            "solver": solver,
            "status": "completed",
            "anneal_score": step.get("anneal_score"),
            "selection_score": step.get("selection_score", step.get("anneal_score")),
            "quality": quality,
            "diagnostics": diagnostics,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "preview": decryption[:300],
            "decryption": decryption,
            "key": {str(k): v for k, v in key.items()},
        }
        validation = null_mask_validation_score_v2(
            row,
            original_length=len(cipher_text.tokens),
            language=language,
        )
        row["validation_score_v2"] = validation["score"]
        row["validation_components_v2"] = validation["components"]
        return row

    def candidate_id(source: str, index: int) -> str:
        if source == "baseline" and index == 0:
            return "000_identity"
        return f"{source}:{index:06d}"

    def solve_mask(
        mask: tuple[str, ...],
        *,
        index: int,
        seed_offset: int,
        source: str,
        run_budget: str | None = None,
        initial_key: dict[int, int] | None = None,
        fixed_cipher_ids: set[int] | None = None,
        polish_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mask_set = set(mask)
        masked_token_ids = {
            cipher_text.alphabet.id_for(symbol)
            for symbol in mask_set
            if cipher_text.alphabet.has_symbol(symbol)
        }
        effective_fixed_cipher_ids = (
            set(fixed_cipher_ids or set()) - masked_token_ids
        )
        filtered_tokens = [
            token
            for token in cipher_text.tokens
            if cipher_text.alphabet.decode([token]) not in mask_set
        ]
        if len(filtered_tokens) < 50:
            return {
                "candidate_id": candidate_id(source, index),
                "evaluated_index": index,
                "mask": list(mask),
                "mask_size": len(mask),
                "filtered_length": len(filtered_tokens),
                "status": "skipped",
                "source": source,
                "seed_offset": seed_offset,
                "reason": "filtered_stream_too_short",
            }
        candidate_cipher = _cipher_text_from_tokens(
            filtered_tokens,
            cipher_text.alphabet,
            source=f"{cipher_text.source}:null_mask:{source}:{index}",
        )
        candidate_started = time.time()
        try:
            candidate_solver, candidate_key, candidate_decryption, candidate_step = _run_homophonic(
                candidate_cipher,
                language,
                budget=run_budget or null_budget,
                refinement="none",
                solver_profile=solver_profile,
                ground_truth=None,
                seed_offset=seed_offset,
                initial_key=initial_key,
                fixed_cipher_ids=effective_fixed_cipher_ids or None,
            )
            row = finalist_row(
                mask=mask,
                filtered_length=len(filtered_tokens),
                solver=candidate_solver,
                key=candidate_key,
                decryption=candidate_decryption,
                step=candidate_step,
                elapsed_seconds=time.time() - candidate_started,
            )
            row["source"] = source
            row["candidate_id"] = candidate_id(source, index)
            row["evaluated_index"] = index
            row["seed_offset"] = seed_offset
            row["budget"] = run_budget or null_budget
            if polish_metadata:
                row["consensus_polish"] = {
                    **polish_metadata,
                    "effective_fixed_symbol_count": len(effective_fixed_cipher_ids),
                    "masked_fixed_symbol_count": len(set(fixed_cipher_ids or set()) & masked_token_ids),
                }
            return row
        except Exception as exc:  # noqa: BLE001
            return {
                "candidate_id": candidate_id(source, index),
                "evaluated_index": index,
                "mask": list(mask),
                "mask_size": len(mask),
                "filtered_length": len(filtered_tokens),
                "status": "error",
                "source": source,
                "seed_offset": seed_offset,
                "reason": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.time() - candidate_started, 3),
            }

    def solve_mask_batch(
        jobs: list[dict[str, Any]],
        *,
        default_budget: str,
    ) -> list[dict[str, Any]]:
        if (
            null_engine != "rust_batch"
            or solver_profile != "zenith_native"
            or binary_model_path is None
        ):
            return [
                solve_mask(
                    tuple(job["mask"]),
                    index=int(job["index"]),
                    seed_offset=int(job["seed_offset"]),
                    source=str(job["source"]),
                    run_budget=job.get("run_budget") or default_budget,
                    initial_key=job.get("initial_key"),
                    fixed_cipher_ids=job.get("fixed_cipher_ids"),
                    polish_metadata=job.get("polish_metadata"),
                )
                for job in jobs
            ]
        ready_jobs: list[dict[str, Any]] = []
        rows_by_position: dict[int, dict[str, Any]] = {}
        for position, job in enumerate(jobs):
            mask = tuple(job["mask"])
            mask_set = set(mask)
            masked_token_ids = {
                cipher_text.alphabet.id_for(symbol)
                for symbol in mask_set
                if cipher_text.alphabet.has_symbol(symbol)
            }
            filtered_length = sum(
                1
                for token in cipher_text.tokens
                if token not in masked_token_ids
            )
            if filtered_length < 50:
                rows_by_position[position] = {
                    "candidate_id": candidate_id(str(job["source"]), int(job["index"])),
                    "evaluated_index": int(job["index"]),
                    "mask": list(mask),
                    "mask_size": len(mask),
                    "filtered_length": filtered_length,
                    "status": "skipped",
                    "source": job["source"],
                    "seed_offset": job["seed_offset"],
                    "reason": "filtered_stream_too_short",
                }
                continue
            ready_job = dict(job)
            ready_job["position"] = position
            ready_job["mask_token_ids"] = sorted(masked_token_ids)
            ready_job["filtered_length"] = filtered_length
            ready_job["short_homophonic"] = filtered_length < 600
            ready_job["run_budget"] = job.get("run_budget") or default_budget
            ready_jobs.append(ready_job)
        grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}
        for job in ready_jobs:
            grouped.setdefault(
                (str(job["run_budget"]), bool(job["short_homophonic"])),
                [],
            ).append(job)
        for (run_budget, short_homophonic), group in grouped.items():
            budget_params = _homophonic_budget_params(
                run_budget,
                short_homophonic,
                search_profile=_homophonic_search_profile(),
            )
            try:
                from analysis.zenith_fast import zenith_null_mask_candidates_batch_fast

                batch = zenith_null_mask_candidates_batch_fast(
                    tokens=list(cipher_text.tokens),
                    candidates=[
                        {
                            "candidate_id": str(job["index"]),
                            "source": str(job["source"]),
                            "seed_offset": int(job["seed_offset"]),
                            "mask_tokens": job["mask_token_ids"],
                            "initial_key": {
                                int(k): int(v)
                                for k, v in (job.get("initial_key") or {}).items()
                            },
                            "fixed_cipher_ids": [
                                int(sid)
                                for sid in sorted(job.get("fixed_cipher_ids") or set())
                            ],
                        }
                        for job in group
                    ],
                    plaintext_ids=plaintext_ids,
                    id_to_letter=id_to_letter,
                    model_path=binary_model_path,
                    epochs=int(budget_params["epochs"]),
                    sampler_iterations=int(budget_params["sampler_iterations"]),
                    seeds=[int(seed) for seed in budget_params["seeds"]],
                    top_n=1,
                    threads=null_threads,
                )
            except Exception as exc:  # noqa: BLE001
                for job in group:
                    rows_by_position[job["position"]] = {
                        "candidate_id": candidate_id(str(job["source"]), int(job["index"])),
                        "evaluated_index": int(job["index"]),
                        "mask": list(job["mask"]),
                        "mask_size": len(job["mask"]),
                        "filtered_length": job["filtered_length"],
                        "status": "error",
                        "source": job["source"],
                        "seed_offset": job["seed_offset"],
                        "reason": f"rust_batch_failed:{type(exc).__name__}: {exc}",
                        "elapsed_seconds": 0.0,
                    }
                continue
            for job, result in zip(group, batch.get("results") or []):
                mask = tuple(job["mask"])
                if result.get("status") != "completed":
                    rows_by_position[job["position"]] = {
                        "candidate_id": candidate_id(str(job["source"]), int(job["index"])),
                        "evaluated_index": int(job["index"]),
                        "mask": list(mask),
                        "mask_size": len(mask),
                        "filtered_length": result.get("filtered_length", job["filtered_length"]),
                        "status": result.get("status", "error"),
                        "source": job["source"],
                        "seed_offset": job["seed_offset"],
                        "reason": result.get("reason"),
                        "elapsed_seconds": round(float(result.get("elapsed_seconds") or 0.0), 3),
                    }
                    continue
                candidate_step = {
                    "anneal_score": result.get("normalized_score", result.get("score")),
                    "selection_score": result.get("normalized_score", result.get("score")),
                }
                row = finalist_row(
                    mask=mask,
                    filtered_length=int(result.get("filtered_length") or job["filtered_length"]),
                    solver="zenith_native",
                    key={int(k): int(v) for k, v in dict(result.get("key") or {}).items()},
                    decryption=str(result.get("decryption") or result.get("plaintext") or ""),
                    step=candidate_step,
                    elapsed_seconds=float(result.get("elapsed_seconds") or 0.0),
                )
                row["source"] = job["source"]
                row["candidate_id"] = candidate_id(str(job["source"]), int(job["index"]))
                row["evaluated_index"] = int(job["index"])
                row["seed_offset"] = job["seed_offset"]
                row["budget"] = run_budget
                row["engine"] = "rust_batch"
                row["best_seed"] = result.get("best_seed")
                row["batch_threads"] = batch.get("threads")
                row["accepted_moves"] = result.get("accepted_moves")
                row["improved_moves"] = result.get("improved_moves")
                row["attempts"] = result.get("attempts") or []
                if job.get("polish_metadata"):
                    masked_token_ids = set(job["mask_token_ids"])
                    fixed_cipher_ids = set(job.get("fixed_cipher_ids") or set())
                    row["consensus_polish"] = {
                        **job["polish_metadata"],
                        "effective_fixed_symbol_count": len(fixed_cipher_ids - masked_token_ids),
                        "masked_fixed_symbol_count": len(fixed_cipher_ids & masked_token_ids),
                    }
                rows_by_position[job["position"]] = row
        return [
            rows_by_position[position]
            for position in range(len(jobs))
            if position in rows_by_position
        ]

    baseline_row = finalist_row(
        mask=(),
        filtered_length=len(cipher_text.tokens),
        solver=base_solver,
        key=base_key,
        decryption=base_decryption,
        step=base_step,
        elapsed_seconds=0.0,
    )
    baseline_row["source"] = "baseline"
    baseline_row["candidate_id"] = candidate_id("baseline", 0)
    baseline_row["evaluated_index"] = 0
    baseline_row["seed_offset"] = 0
    baseline_row["budget"] = null_budget
    rows.append(baseline_row)

    initial_jobs = [
        {
            "mask": mask,
            "index": index,
            "seed_offset": index * 100,
            "source": "initial",
        }
        for index, mask in enumerate(masks[1:], start=1)
    ]
    rows.extend(solve_mask_batch(initial_jobs, default_budget=null_budget))

    def sort_completed_list(completed_rows: list[dict[str, Any]]) -> None:
        attach_null_mask_ensemble_scores(
            completed_rows,
            original_length=len(cipher_text.tokens),
            language=language,
            language_quality_model=language_quality_model,
        )
        if ranker == "language_quality":
            completed_rows.sort(key=null_mask_language_quality_rank_key, reverse=True)
        elif ranker == "ensemble":
            completed_rows.sort(key=null_mask_rank_key, reverse=True)
        else:
            completed_rows.sort(
                key=lambda item: (
                    _null_mask_rank_validation_score(item),
                    float(item.get("selection_score") or float("-inf")),
                ),
                reverse=True,
            )

    def ranked_completed_rows() -> list[dict[str, Any]]:
        completed_rows = [row for row in rows if row.get("status") == "completed"]
        sort_completed_list(completed_rows)
        return completed_rows

    def portfolio_rows(completed_rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        if portfolio_top_n <= 0:
            return completed_rows[:limit]
        return _null_mask_ranker_portfolio(
            completed_rows,
            limit=limit,
            rankers=portfolio_rankers,
            language_quality_enabled=language_quality_model is not None,
        )

    def usable_ranker_views() -> list[str]:
        ranker_views = [
            item for item in portfolio_rankers
            if item in {"validation", "ensemble"}
            or (item == "language_quality" and language_quality_model is not None)
        ]
        return ranker_views or ["validation"]

    completed = ranked_completed_rows()
    finalist_portfolio_report: dict[str, Any] = {
        "enabled": portfolio_top_n > 0,
        "top_n": portfolio_top_n,
        "rankers": portfolio_rankers,
        "selected_count": 0,
        "policy": (
            "When enabled, second-stage promotion draws from a round-robin "
            "portfolio of validation, language-quality, and ensemble finalist "
            "views instead of a single scalar ranking. This keeps rougher but "
            "structurally promising basins alive for deeper reruns."
        ),
    }
    beam_report: dict[str, Any] = {
        "enabled": beam_enabled and beam_max_size > max_mask_size and beam_width > 0 and beam_max_masks > 0,
        "width": beam_width,
        "max_size": beam_max_size,
        "max_masks": beam_max_masks,
        "generated_mask_count": 0,
        "completed_mask_count": 0,
        "policy": (
            "Stage A2 extends the best initial masks with additional candidate "
            "symbols, giving p068-like cases a bounded path to size-3 masks "
            "without enumerating every combination."
        ),
    }
    if completed and beam_report["enabled"]:
        seen_masks = {tuple(sorted(row.get("mask") or [])) for row in rows}
        beam_masks: list[tuple[str, ...]] = []
        extension_symbols = list(candidate_symbols)
        beam_source_rows = portfolio_rows(completed, limit=beam_width)
        for row in beam_source_rows:
            base = tuple(row.get("mask") or [])
            if not base or len(base) >= beam_max_size:
                continue
            for symbol in extension_symbols:
                if symbol in base:
                    continue
                expanded = tuple(sorted((*base, symbol)))
                if expanded in seen_masks:
                    continue
                seen_masks.add(expanded)
                beam_masks.append(expanded)
                if len(beam_masks) >= beam_max_masks:
                    break
            if len(beam_masks) >= beam_max_masks:
                break
        beam_jobs = [
            {
                "mask": mask,
                "index": 50_000 + index,
                "seed_offset": 50_000 + index * 100,
                "source": "beam",
            }
            for index, mask in enumerate(beam_masks, start=1)
        ]
        rows.extend(solve_mask_batch(beam_jobs, default_budget=null_budget))
        completed = ranked_completed_rows()
        beam_report.update({
            "generated_mask_count": len(beam_masks),
            "completed_mask_count": sum(
                1 for row in rows
                if row.get("source") == "beam" and row.get("status") == "completed"
            ),
            "source_masks": [
                row.get("mask") or [] for row in beam_source_rows[:12]
            ],
            "top_masks": [list(mask) for mask in beam_masks[:12]],
        })
    neighborhood_report: dict[str, Any] = {
        "enabled": neighborhood_enabled and neighborhood_top_n > 0 and neighborhood_max_masks > 0,
        "top_n": neighborhood_top_n,
        "max_size": neighborhood_max_size,
        "max_masks": neighborhood_max_masks,
        "multi_view_enabled": neighborhood_multi_view,
        "generated_mask_count": 0,
        "completed_mask_count": 0,
        "policy": (
            "Stage A3 explores add/remove/swap neighbors around strong finalist "
            "masks. In the wide profile, each ranker view gets its own "
            "neighborhood so stochastic basins are not lost when a different "
            "ranker happens to generate the same mask first."
        ),
    }
    if completed and neighborhood_report["enabled"]:
        neighborhood_jobs: list[dict[str, Any]] = []
        neighborhood_view_reports: list[dict[str, Any]] = []
        if neighborhood_multi_view:
            base_seen_masks = {tuple(sorted(row.get("mask") or [])) for row in rows}
            for view_index, ranker_view in enumerate(usable_ranker_views()):
                view_rows = sorted(
                    completed,
                    key=lambda row, ranker_view=ranker_view: _null_mask_ranker_sort_key(row, ranker_view),
                    reverse=True,
                )
                neighborhood_source_rows = _null_mask_unique_rows_by_mask(
                    view_rows,
                    limit=neighborhood_top_n,
                )
                view_seen_masks = set(base_seen_masks)
                neighborhood_masks = _null_mask_neighborhood_masks(
                    neighborhood_source_rows,
                    candidate_symbols=candidate_symbols,
                    seen_masks=view_seen_masks,
                    max_size=neighborhood_max_size,
                    max_masks=neighborhood_max_masks,
                )
                for index, mask in enumerate(neighborhood_masks, start=1):
                    neighborhood_jobs.append({
                        "mask": mask,
                        "index": 70_000 + view_index * 10_000 + index,
                        "seed_offset": 70_000 + index * 100,
                        "source": f"neighborhood_{ranker_view}",
                    })
                neighborhood_view_reports.append({
                    "ranker_view": ranker_view,
                    "source_masks": [
                        row.get("mask") or []
                        for row in neighborhood_source_rows[:12]
                    ],
                    "generated_mask_count": len(neighborhood_masks),
                    "top_masks": [list(mask) for mask in neighborhood_masks[:12]],
                })
        else:
            seen_masks = {tuple(sorted(row.get("mask") or [])) for row in rows}
            neighborhood_source_rows = portfolio_rows(completed, limit=neighborhood_top_n)
            neighborhood_masks = _null_mask_neighborhood_masks(
                neighborhood_source_rows,
                candidate_symbols=candidate_symbols,
                seen_masks=seen_masks,
                max_size=neighborhood_max_size,
                max_masks=neighborhood_max_masks,
            )
            neighborhood_jobs = [
                {
                    "mask": mask,
                    "index": 70_000 + index,
                    "seed_offset": 70_000 + index * 100,
                    "source": "neighborhood",
                }
                for index, mask in enumerate(neighborhood_masks, start=1)
            ]
            neighborhood_view_reports.append({
                "ranker_view": "portfolio",
                "source_masks": [
                    row.get("mask") or [] for row in neighborhood_source_rows[:12]
                ],
                "generated_mask_count": len(neighborhood_masks),
                "top_masks": [list(mask) for mask in neighborhood_masks[:12]],
            })
        rows.extend(solve_mask_batch(neighborhood_jobs, default_budget=null_budget))
        completed = ranked_completed_rows()
        neighborhood_report.update({
            "generated_mask_count": len(neighborhood_jobs),
            "completed_mask_count": sum(
                1 for row in rows
                if str(row.get("source") or "").startswith("neighborhood")
                and row.get("status") == "completed"
            ),
            "views": neighborhood_view_reports,
            "source_masks": neighborhood_view_reports[0]["source_masks"] if neighborhood_view_reports else [],
            "top_masks": neighborhood_view_reports[0]["top_masks"] if neighborhood_view_reports else [],
        })
    consensus_report: dict[str, Any] = {
        "enabled": consensus_enabled and consensus_top_n > 1,
        "top_n": consensus_top_n,
        "min_agreement": consensus_min_agreement,
        "min_fixed_symbols": consensus_min_fixed,
        "target_max_mutable_symbols": consensus_max_mutable,
        "budget": consensus_budget,
        "multi_view_enabled": consensus_multi_view,
        "generated_run_count": 0,
        "completed_run_count": 0,
        "policy": (
            "Stage A4 derives a consensus key from strong finalist views, freezes "
            "mappings they agree on, and reruns the same masks with only disputed "
            "symbols free to move. In the wide profile, validation, language-quality, "
            "and ensemble views are polished separately so one ranker does not erase "
            "a promising basin found by another. The consensus is computed from "
            "solver-produced finalist keys only; benchmark plaintext is not used."
        ),
    }
    if completed and consensus_report["enabled"]:
        consensus_rankers = usable_ranker_views()
        plan_records: list[tuple[str, dict[str, Any]]] = []
        if consensus_multi_view:
            for ranker_view in consensus_rankers:
                view_rows = sorted(
                    completed,
                    key=lambda row, ranker_view=ranker_view: _null_mask_ranker_sort_key(row, ranker_view),
                    reverse=True,
                )
                consensus_source_rows = _null_mask_unique_rows_by_mask(
                    view_rows,
                    limit=consensus_top_n,
                )
                plan_records.append((
                    ranker_view,
                    _null_mask_consensus_polish_plan(
                        consensus_source_rows,
                        cipher_text=cipher_text,
                        min_agreement=consensus_min_agreement,
                        min_fixed_symbols=consensus_min_fixed,
                        target_max_mutable_symbols=consensus_max_mutable,
                    ),
                ))
        else:
            consensus_source_rows = portfolio_rows(completed, limit=consensus_top_n)
            plan_records.append((
                "portfolio",
                _null_mask_consensus_polish_plan(
                    consensus_source_rows,
                    cipher_text=cipher_text,
                    min_agreement=consensus_min_agreement,
                    min_fixed_symbols=consensus_min_fixed,
                    target_max_mutable_symbols=consensus_max_mutable,
                ),
            ))

        consensus_rows: list[dict[str, Any]] = []
        view_reports: list[dict[str, Any]] = []
        for view_index, (ranker_view, plan) in enumerate(plan_records):
            view_report = {
                **plan["report"],
                "ranker_view": ranker_view,
                "source_masks": [
                    row.get("mask") or []
                    for row in plan["source_rows"][:consensus_top_n]
                ],
                "generated_run_count": 0,
                "completed_run_count": 0,
                "polished_masks": [],
            }
            if plan["enabled"]:
                consensus_jobs = []
                for rank, row in enumerate(plan["source_rows"], start=1):
                    mask = tuple(row.get("mask") or [])
                    row_key = {
                        int(k): int(v)
                        for k, v in (row.get("key") or {}).items()
                    }
                    initial_key = dict(row_key)
                    initial_key.update(plan["anchor_key"])
                    polish_metadata = {
                        **plan["row_metadata"],
                        "ranker_view": ranker_view,
                        "consensus_view_index": view_index,
                    }
                    consensus_jobs.append({
                        "mask": mask,
                        "index": 90_000 + view_index * 1_000 + rank,
                        "seed_offset": 90_000 + view_index * 10_000 + rank * 100,
                        "source": (
                            "consensus_polish"
                            if not consensus_multi_view
                            else f"consensus_polish_{ranker_view}"
                        ),
                        "run_budget": consensus_budget,
                        "initial_key": initial_key,
                        "fixed_cipher_ids": set(plan["fixed_cipher_ids"]),
                        "polish_metadata": polish_metadata,
                    })
                view_rows = solve_mask_batch(consensus_jobs, default_budget=consensus_budget)
                consensus_rows.extend(view_rows)
                rows.extend(view_rows)
                view_report.update({
                    "generated_run_count": len(view_rows),
                    "completed_run_count": sum(
                        1 for row in view_rows
                        if row.get("status") == "completed"
                    ),
                    "polished_masks": [row.get("mask") or [] for row in view_rows],
                })
            view_reports.append(view_report)
        if consensus_rows:
            completed = ranked_completed_rows()
        if consensus_multi_view:
            consensus_report.update({
                "views": view_reports,
                "generated_run_count": len(consensus_rows),
                "completed_run_count": sum(
                    1 for row in consensus_rows
                    if row.get("status") == "completed"
                ),
                "polished_masks": [row.get("mask") or [] for row in consensus_rows],
                "enabled_view_count": sum(1 for report in view_reports if report.get("enabled")),
            })
        elif view_reports:
            consensus_report.update(view_reports[0])
    adaptive_report: dict[str, Any] = {
        "enabled": adaptive_enabled,
        "triggered": False,
        "min_validation_score": adaptive_min_validation,
        "near_tie_margin": adaptive_near_tie_margin,
        "candidate_limit": adaptive_candidate_limit,
        "max_size": adaptive_max_mask_size,
        "max_masks": adaptive_max_masks,
        "beam_width": adaptive_beam_width,
        "beam_max_size": adaptive_beam_max_size,
        "beam_max_masks": adaptive_beam_max_masks,
        "bridge_anchor_count": adaptive_bridge_anchor_count,
        "bridge_top_rows": adaptive_bridge_top_rows,
        "bridge_max_masks": adaptive_bridge_max_masks,
        "bridge_restarts": adaptive_bridge_restarts,
        "bridge_consensus_max_masks": adaptive_bridge_consensus_max_masks,
        "neighborhood_top_n": adaptive_neighborhood_top_n,
        "neighborhood_max_size": adaptive_neighborhood_max_size,
        "neighborhood_max_masks": adaptive_neighborhood_max_masks,
        "consensus_top_n": adaptive_consensus_top_n,
        "consensus_budget": adaptive_consensus_budget,
        "stability_top_n": adaptive_stability_top_n,
        "stability_restarts": adaptive_stability_restarts,
        "stability_waves": adaptive_stability_waves,
        "stability_budget": adaptive_stability_budget,
        "decision": None,
        "generated_mask_count": 0,
        "completed_mask_count": 0,
        "policy": (
            "Optional baseline-then-escalate policy. The normal null-mask "
            "menu is ranked first; if ground-truth-free damage/confidence "
            "signals look weak, Decipher appends a wider mask/beam/"
            "neighborhood/consensus screen and ranks all candidates together."
        ),
    }
    if adaptive_enabled:
        decision = _null_mask_adaptive_decision(
            completed,
            consensus_report=consensus_report,
            min_validation_score=adaptive_min_validation,
            near_tie_margin=adaptive_near_tie_margin,
        )
        adaptive_report["decision"] = decision
        if decision.get("triggered"):
            adaptive_report["triggered"] = True
            seen_masks = {tuple(sorted(row.get("mask") or [])) for row in rows}
            adaptive_symbols = select_null_candidate_symbols(
                diagnostics,
                limit=adaptive_candidate_limit,
            )
            adaptive_masks = generate_null_masks(adaptive_symbols, adaptive_max_mask_size)
            if adaptive_max_masks > 0:
                adaptive_masks = [()] + adaptive_masks[1:adaptive_max_masks + 1]
            adaptive_initial_masks = [
                tuple(mask)
                for mask in adaptive_masks[1:]
                if tuple(sorted(mask)) not in seen_masks
            ]
            for mask in adaptive_initial_masks:
                seen_masks.add(tuple(sorted(mask)))
            adaptive_initial_jobs = [
                {
                    "mask": mask,
                    "index": 100_000 + index,
                    "seed_offset": 100_000 + index * 100,
                    "source": "adaptive_initial",
                }
                for index, mask in enumerate(adaptive_initial_masks, start=1)
            ]
            adaptive_rows = solve_mask_batch(adaptive_initial_jobs, default_budget=null_budget)
            rows.extend(adaptive_rows)
            completed = ranked_completed_rows()

            adaptive_bridge_masks = _null_mask_bridge_pair_masks(
                completed[:adaptive_bridge_top_rows],
                candidate_symbols=adaptive_symbols,
                anchor_symbols=candidate_symbols[:adaptive_bridge_anchor_count],
                seen_masks=seen_masks,
                max_masks=adaptive_bridge_max_masks,
            )
            adaptive_bridge_jobs = [
                {
                    "mask": mask,
                    "index": 105_000 + index,
                    "seed_offset": 105_000 + index * 100,
                    "source": "adaptive_bridge",
                }
                for index, mask in enumerate(adaptive_bridge_masks, start=1)
            ]
            for restart in range(1, max(0, adaptive_bridge_restarts) + 1):
                adaptive_bridge_jobs.extend(
                    {
                        "mask": mask,
                        "index": 105_000 + restart * 10_000 + index,
                        "seed_offset": 105_000 + restart * 100_000 + index * 100,
                        "source": "adaptive_bridge_restart",
                    }
                    for index, mask in enumerate(adaptive_bridge_masks, start=1)
                )
            adaptive_bridge_rows = solve_mask_batch(
                adaptive_bridge_jobs,
                default_budget=null_budget,
            )
            rows.extend(adaptive_bridge_rows)
            completed = ranked_completed_rows()

            adaptive_beam_masks: list[tuple[str, ...]] = []
            if adaptive_beam_width > 0 and adaptive_beam_max_masks > 0:
                for row in completed[:adaptive_beam_width]:
                    base = tuple(row.get("mask") or [])
                    if not base or len(base) >= adaptive_beam_max_size:
                        continue
                    for symbol in adaptive_symbols:
                        if symbol in base:
                            continue
                        expanded = tuple(sorted((*base, symbol)))
                        if expanded in seen_masks:
                            continue
                        seen_masks.add(expanded)
                        adaptive_beam_masks.append(expanded)
                        if len(adaptive_beam_masks) >= adaptive_beam_max_masks:
                            break
                    if len(adaptive_beam_masks) >= adaptive_beam_max_masks:
                        break
            adaptive_beam_jobs = [
                {
                    "mask": mask,
                    "index": 110_000 + index,
                    "seed_offset": 110_000 + index * 100,
                    "source": "adaptive_beam",
                }
                for index, mask in enumerate(adaptive_beam_masks, start=1)
            ]
            adaptive_beam_rows = solve_mask_batch(adaptive_beam_jobs, default_budget=null_budget)
            rows.extend(adaptive_beam_rows)
            completed = ranked_completed_rows()

            adaptive_neighborhood_masks: list[tuple[str, ...]] = []
            if adaptive_neighborhood_top_n > 0 and adaptive_neighborhood_max_masks > 0:
                adaptive_neighborhood_masks = _null_mask_neighborhood_masks(
                    completed[:adaptive_neighborhood_top_n],
                    candidate_symbols=adaptive_symbols,
                    seen_masks=seen_masks,
                    max_size=adaptive_neighborhood_max_size,
                    max_masks=adaptive_neighborhood_max_masks,
                )
            adaptive_neighborhood_jobs = [
                {
                    "mask": mask,
                    "index": 120_000 + index,
                    "seed_offset": 120_000 + index * 100,
                    "source": "adaptive_neighborhood",
                }
                for index, mask in enumerate(adaptive_neighborhood_masks, start=1)
            ]
            adaptive_neighborhood_rows = solve_mask_batch(
                adaptive_neighborhood_jobs,
                default_budget=null_budget,
            )
            rows.extend(adaptive_neighborhood_rows)
            completed = ranked_completed_rows()

            adaptive_consensus_rows: list[dict[str, Any]] = []
            adaptive_consensus_report: dict[str, Any] | None = None
            if adaptive_consensus_top_n > 1:
                adaptive_plan = _null_mask_consensus_polish_plan(
                    completed[:adaptive_consensus_top_n],
                    cipher_text=cipher_text,
                    min_agreement=consensus_min_agreement,
                    min_fixed_symbols=consensus_min_fixed,
                    target_max_mutable_symbols=consensus_max_mutable,
                )
                adaptive_consensus_report = adaptive_plan["report"]
                if adaptive_plan["enabled"]:
                    adaptive_consensus_jobs = []
                    consensus_seen_masks: set[tuple[str, ...]] = set()
                    for rank, row in enumerate(adaptive_plan["source_rows"], start=1):
                        mask = tuple(row.get("mask") or [])
                        consensus_seen_masks.add(tuple(sorted(mask)))
                        row_key = {
                            int(k): int(v)
                            for k, v in (row.get("key") or {}).items()
                        }
                        initial_key = dict(row_key)
                        initial_key.update(adaptive_plan["anchor_key"])
                        adaptive_consensus_jobs.append({
                            "mask": mask,
                            "index": 130_000 + rank,
                            "seed_offset": 130_000 + rank * 100,
                            "source": "adaptive_consensus_polish",
                            "run_budget": adaptive_consensus_budget,
                            "initial_key": initial_key,
                            "fixed_cipher_ids": set(adaptive_plan["fixed_cipher_ids"]),
                            "polish_metadata": adaptive_plan["row_metadata"],
                        })
                    bridge_consensus_masks: list[tuple[str, ...]] = []
                    for row in completed:
                        if not str(row.get("source") or "").startswith("adaptive_bridge"):
                            continue
                        mask = tuple(str(symbol) for symbol in (row.get("mask") or []))
                        if len(mask) != 2:
                            continue
                        canonical = tuple(sorted(mask))
                        if canonical in consensus_seen_masks:
                            continue
                        consensus_seen_masks.add(canonical)
                        bridge_consensus_masks.append(canonical)
                        if len(bridge_consensus_masks) >= adaptive_bridge_consensus_max_masks:
                            break
                    for mask in bridge_consensus_masks:
                        adaptive_consensus_jobs.append({
                            "mask": mask,
                            "index": 131_000 + len(adaptive_consensus_jobs),
                            "seed_offset": 131_000 + len(adaptive_consensus_jobs) * 100,
                            "source": "adaptive_bridge_consensus_polish",
                            "run_budget": adaptive_consensus_budget,
                            "initial_key": dict(adaptive_plan["anchor_key"]),
                            "fixed_cipher_ids": set(adaptive_plan["fixed_cipher_ids"]),
                            "polish_metadata": adaptive_plan["row_metadata"],
                        })
                    adaptive_consensus_rows = solve_mask_batch(
                        adaptive_consensus_jobs,
                        default_budget=adaptive_consensus_budget,
                    )
                    rows.extend(adaptive_consensus_rows)
                    completed = ranked_completed_rows()
            adaptive_stability_rows: list[dict[str, Any]] = []
            adaptive_stabilized: list[dict[str, Any]] = []
            if (
                adaptive_stability_top_n > 0
                and adaptive_stability_restarts > 0
                and adaptive_stability_waves > 0
            ):
                stabilized_masks: set[tuple[str, ...]] = set()
                stability_source_rows: list[dict[str, Any]] = []
                for wave in range(1, adaptive_stability_waves + 1):
                    wave_source_rows = _null_mask_unique_rows_by_mask(
                        [
                            row for row in completed
                            if tuple(sorted(str(symbol) for symbol in (row.get("mask") or [])))
                            not in stabilized_masks
                        ],
                        limit=adaptive_stability_top_n,
                    )
                    if not wave_source_rows:
                        break
                    stability_source_rows.extend(wave_source_rows)
                    stability_jobs = []
                    for rank, row in enumerate(wave_source_rows, start=1):
                        mask = tuple(str(symbol) for symbol in (row.get("mask") or []))
                        for restart in range(1, adaptive_stability_restarts + 1):
                            stability_jobs.append({
                                "mask": mask,
                                "index": 140_000 + wave * 100_000 + rank * 1_000 + restart,
                                "seed_offset": 140_000 + wave * 100_000 + rank * 10_000 + restart * 100,
                                "source": "adaptive_stability",
                                "run_budget": adaptive_stability_budget,
                            })
                    wave_stability_rows = solve_mask_batch(
                        stability_jobs,
                        default_budget=adaptive_stability_budget,
                    )
                    adaptive_stability_rows.extend(wave_stability_rows)
                    stability_rows_by_mask: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
                    for row in wave_stability_rows:
                        stability_rows_by_mask[
                            tuple(str(symbol) for symbol in (row.get("mask") or []))
                        ].append(row)
                    for row in wave_source_rows:
                        mask = tuple(str(symbol) for symbol in (row.get("mask") or []))
                        stabilized_masks.add(tuple(sorted(mask)))
                        _attach_null_mask_stability(row, stability_rows_by_mask.get(mask, []))
                        adaptive_stabilized.append({
                            "mask": row.get("mask") or [],
                            "rank_validation_score_v2": row.get("rank_validation_score_v2"),
                            "validation_score_v2": row.get("validation_score_v2"),
                            "stability": row.get("stability"),
                            "best_preview": row.get("preview"),
                        })
                    unstabilized_rows = [
                        row for row in completed
                        if tuple(sorted(str(symbol) for symbol in (row.get("mask") or [])))
                        not in stabilized_masks
                    ]
                    completed = stability_source_rows + unstabilized_rows
                    sort_completed_list(completed)
            generated_count = (
                len(adaptive_initial_jobs)
                + len(adaptive_bridge_jobs)
                + len(adaptive_beam_jobs)
                + len(adaptive_neighborhood_jobs)
                + (len(adaptive_consensus_rows) if adaptive_consensus_rows else 0)
                + (len(adaptive_stability_rows) if adaptive_stability_rows else 0)
            )
            adaptive_report.update({
                "candidate_symbols": adaptive_symbols,
                "generated_mask_count": generated_count,
                "completed_mask_count": sum(
                    1 for row in rows
                    if str(row.get("source") or "").startswith("adaptive_")
                    and row.get("status") == "completed"
                ),
                "initial_mask_count": len(adaptive_initial_masks),
                "bridge_mask_count": len(adaptive_bridge_masks),
                "bridge_run_count": len(adaptive_bridge_jobs),
                "beam_mask_count": len(adaptive_beam_masks),
                "neighborhood_mask_count": len(adaptive_neighborhood_masks),
                "consensus_run_count": len(adaptive_consensus_rows),
                "consensus": adaptive_consensus_report,
                "stability_run_count": len(adaptive_stability_rows),
                "stabilized_mask_count": len(adaptive_stabilized),
                "stabilized_finalists": adaptive_stabilized,
                "selected_after_adaptive": _compact_null_mask_row(completed[0]) if completed else None,
            })
    promotion_report: dict[str, Any] = {
        "enabled": promote_top_n > 0 and promote_reruns > 0,
        "top_n": promote_top_n,
        "reruns_per_mask": promote_reruns,
        "budget": promote_budget,
        "promoted_mask_count": 0,
        "policy": (
            "Stage B promotes the initial top null-mask finalists into a "
            "stronger second-stage solve before final ranking. This keeps "
            "multiple plausible basins alive long enough for a deeper "
            "ground-truth-free validation pass."
        ),
    }
    if completed and promotion_report["enabled"]:
        promoted: list[dict[str, Any]] = []
        promoted_rows: list[dict[str, Any]] = []
        promotion_source_rows = (
            _null_mask_ranker_portfolio(
                completed,
                limit=promote_top_n,
                rankers=portfolio_rankers,
                language_quality_enabled=language_quality_model is not None,
            )
            if portfolio_top_n > 0
            else completed[:promote_top_n]
        )
        if portfolio_top_n > 0:
            finalist_portfolio_report.update({
                "selected_count": len(promotion_source_rows),
                "selected_finalists": [
                    _compact_null_mask_row(row) for row in promotion_source_rows
                ],
            })
        for rank, row in enumerate(promotion_source_rows, start=1):
            mask = tuple(row.get("mask") or [])
            probe_rows = []
            for probe_index in range(1, promote_reruns + 1):
                seed_offset = 20_000 + rank * 1_000 + probe_index * 100
                probe = solve_mask(
                    mask,
                    index=rank * 100 + probe_index,
                    seed_offset=seed_offset,
                    source="promotion",
                    run_budget=promote_budget,
                )
                probe_rows.append(probe)
            _attach_null_mask_promotion(row, probe_rows)
            promoted_rows.append(row)
            promoted.append({
                "mask": row.get("mask") or [],
                "initial_rank": rank,
                "promoted_validation_score_v2": row.get("promoted_validation_score_v2"),
                "promotion": row.get("promotion"),
                "best_preview": row.get("preview"),
            })
        promotion_report.update({
            "promoted_mask_count": len(promoted),
            "promoted_finalists": promoted,
        })
        promoted_masks = {tuple(row.get("mask") or []) for row in promoted_rows}
        unpromoted_rows = [
            row for row in completed
            if tuple(row.get("mask") or []) not in promoted_masks
        ]
        completed = promoted_rows + unpromoted_rows
        sort_completed_list(completed)
    confirmation_report: dict[str, Any] = {
        "enabled": confirm_top_n > 0 and confirm_reruns > 0,
        "top_n": confirm_top_n,
        "reruns_per_mask": confirm_reruns,
        "confirmed_mask_count": 0,
        "policy": (
            "Stage B reruns the top null-mask finalists with independent seed "
            "offsets and records stability evidence. Selection follows the "
            "configured ground-truth-free ranker, which defaults to scalar "
            "validation; confirmation remains supporting metadata."
        ),
    }
    if completed and confirmation_report["enabled"]:
        confirmed = []
        confirmed_rows = []
        for rank, row in enumerate(completed[:confirm_top_n], start=1):
            mask = tuple(row.get("mask") or [])
            probe_rows = []
            for probe_index in range(1, confirm_reruns + 1):
                seed_offset = 10_000 + rank * 1_000 + probe_index * 100
                probe = solve_mask(
                    mask,
                    index=rank * 100 + probe_index,
                    seed_offset=seed_offset,
                    source="confirmation",
                )
                probe_rows.append(probe)
            completed_probes = [
                probe for probe in probe_rows
                if probe.get("status") == "completed"
            ]
            _attach_null_mask_confirmation(row, completed_probes)
            confirmed_rows.append(row)
            confirmed.append({
                "mask": row.get("mask") or [],
                "initial_rank": rank,
                "initial_validation_score_v2": row.get("validation_score_v2"),
                "confirmed_validation_score_v2": row.get("confirmed_validation_score_v2"),
                "confirmation": row.get("confirmation"),
                "best_preview": row.get("preview"),
            })
        confirmation_report.update({
            "confirmed_mask_count": len(confirmed),
            "confirmed_finalists": confirmed,
        })
        confirmed_masks = {tuple(row.get("mask") or []) for row in confirmed_rows}
        unconfirmed_rows = [
            row for row in completed
            if tuple(row.get("mask") or []) not in confirmed_masks
        ]
        completed = confirmed_rows + unconfirmed_rows
        sort_completed_list(completed)
    selected = completed[0] if completed else None
    selected_mask = tuple(selected.get("mask") or []) if selected else ()
    compact_rows = [
        _compact_null_mask_row(row, include_validation_text=store_evaluated_text)
        for row in rows
    ]
    top_finalists = completed[:top_n]
    # Additive artifact enrichment: attach a normalized candidate packet to each
    # finalist row so downstream consumers/artifacts have the interchange shape.
    for finalist_rank, finalist_row_dict in enumerate(top_finalists, start=1):
        finalist_row_dict["packet"] = packet_from_null_mask_row(
            finalist_row_dict, rank=finalist_rank
        ).to_dict()

    # Opt-in LLM finalist reader (Phase 4a item 4.2): annotate by default, and
    # override the scalar winner only under DECIPHER_FINALIST_READER_SELECTS.
    finalist_reader_block, reader_override_index = _finalist_reader_rank(
        [row.get("packet") for row in top_finalists],
        language=language,
        allow_select=True,
    )
    if reader_override_index is not None and 0 <= reader_override_index < len(top_finalists):
        chosen = top_finalists[reader_override_index]
        if isinstance(finalist_reader_block, dict):
            finalist_reader_block["scalar_selected_candidate_id"] = (
                selected.get("candidate_id") if isinstance(selected, dict) else None
            )
            finalist_reader_block["reader_selected_candidate_id"] = chosen.get("candidate_id")
            # The override index is resolved by candidate id (Phase 4a F2), so the
            # finalist actually installed must be the one whose candidate id the
            # reader returned as best. A mismatch means the prepared/packets
            # alignment regressed and the wrong row would be selected.
            assert (
                finalist_reader_block["reader_selected_candidate_id"]
                == finalist_reader_block["reader_best_candidate_id"]
            ), "finalist reader selected/best candidate id mismatch"
        selected = chosen
        selected_mask = tuple(chosen.get("mask") or [])

    return {
        "name": "search_null_masks",
        "status": "completed" if completed else "error",
        "experimental": True,
        "policy": (
            "Opt-in null/codeword bakeoff for homophonic ciphers. Candidate "
            "masks are generated from ciphertext diagnostics and the baseline "
            "solver key, then ranked by ground-truth-free language coherence "
            "signals. This is not the default automated route."
        ),
        "language": language,
        "budget": null_budget,
        "engine": null_engine,
        "threads": null_threads,
        "store_evaluated_text": store_evaluated_text,
        "solver_profile": solver_profile,
        "profile": profile,
        "binary_ngram_model": _zenith_native_model_metadata(str(binary_model_path)) if binary_model_path else None,
        "candidate_limit": candidate_limit,
        "candidate_symbols": candidate_symbols,
        "mask_count": len(rows),
        "evaluated_mask_count": len(rows),
        "completed_mask_count": len(completed),
        "max_mask_size": max_mask_size,
        "max_masks": max_masks,
        "beam": beam_report,
        "neighborhood": neighborhood_report,
        "top_n": top_n,
        "ranker": ranker,
        "language_quality_model": (
            {
                "path": str(Path(language_quality_model_path_raw).expanduser()),
                "language": language_quality_model.language,
                "version": language_quality_model.version,
                "feature_count": len(language_quality_model.feature_names),
                "training_summary": language_quality_model.training_summary,
            }
            if language_quality_model is not None
            else None
        ),
        "finalist_portfolio": finalist_portfolio_report,
        "promotion": promotion_report,
        "consensus_polish": consensus_report,
        "adaptive": adaptive_report,
        "confirmation": confirmation_report,
        "diagnostics": diagnostics,
        "selected_mask": list(selected_mask),
        "selected": selected,
        "finalist_reader": finalist_reader_block,
        "top_finalists": top_finalists,
        "evaluated_rows": compact_rows,
        "baseline_rank": next(
            (
                rank
                for rank, row in enumerate(completed, start=1)
                if not row.get("mask")
            ),
            None,
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def _compact_null_mask_row(
    row: dict[str, Any],
    *,
    include_validation_text: bool = False,
) -> dict[str, Any]:
    """Keep every null-mask row inspectable without storing every full decrypt."""
    compact = {
        "candidate_id": row.get("candidate_id"),
        "evaluated_index": row.get("evaluated_index"),
        "mask": row.get("mask") or [],
        "mask_size": row.get("mask_size"),
        "status": row.get("status"),
        "filtered_length": row.get("filtered_length"),
        "rank_validation_score_v2": row.get("rank_validation_score_v2"),
        "validation_score_v2": row.get("validation_score_v2"),
        "validation_components_v2": row.get("validation_components_v2"),
        "selection_score": row.get("selection_score"),
        "anneal_score": row.get("anneal_score"),
        "ensemble_score_v1": row.get("ensemble_score_v1"),
        "ensemble_vote_rate_v1": row.get("ensemble_vote_rate_v1"),
        "ensemble_features_v1": row.get("ensemble_features_v1"),
        "language_quality_raw_score": row.get("language_quality_raw_score"),
        "language_quality_score": row.get("language_quality_score"),
        "language_quality_rank_score": row.get("language_quality_rank_score"),
        "language_quality_model": row.get("language_quality_model"),
        "engine": row.get("engine"),
        "batch_threads": row.get("batch_threads"),
        "best_seed": row.get("best_seed"),
        "source": row.get("source"),
        "budget": row.get("budget"),
        "promoted_validation_score_v2": row.get("promoted_validation_score_v2"),
        "confirmed_validation_score_v2": row.get("confirmed_validation_score_v2"),
        "adaptive_stability_score_v2": row.get("adaptive_stability_score_v2"),
        "adaptive_stability_best_validation_score_v2": row.get("adaptive_stability_best_validation_score_v2"),
        "consensus_polish": row.get("consensus_polish"),
        "stability": row.get("stability"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "preview": str(row.get("preview") or "")[:180],
        "reason": row.get("reason"),
    }
    if include_validation_text:
        compact["validation_text"] = str(row.get("decryption") or row.get("validation_text") or "")
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    compact["diagnostics"] = {
        "dict_rate": diagnostics.get("dict_rate"),
        "segmentation_cost": diagnostics.get("segmentation_cost"),
        "segmented_word_count": diagnostics.get("segmented_word_count"),
        "pseudo_word_count": diagnostics.get("pseudo_word_count"),
        "pseudo_word_fraction": diagnostics.get("pseudo_word_fraction"),
        "short_word_fraction": diagnostics.get("short_word_fraction"),
        "long_pseudo_word_fraction": diagnostics.get("long_pseudo_word_fraction"),
        "dictionary_hit_count": diagnostics.get("dictionary_hit_count"),
        "dictionary_content_word_count": diagnostics.get("dictionary_content_word_count"),
        "dictionary_long_content_word_count": diagnostics.get("dictionary_long_content_word_count"),
        "dictionary_content_word_fraction": diagnostics.get("dictionary_content_word_fraction"),
        "dictionary_content_char_fraction": diagnostics.get("dictionary_content_char_fraction"),
        "dictionary_content_sample": diagnostics.get("dictionary_content_sample"),
        "dictionary_long_content_sample": diagnostics.get("dictionary_long_content_sample"),
        "binary_ngram_mean_log_prob": diagnostics.get("binary_ngram_mean_log_prob"),
        "unique_letters": diagnostics.get("unique_letters"),
        "top_letter_fraction": diagnostics.get("top_letter_fraction"),
    }
    compact["quality"] = {
        "collapsed": quality.get("collapsed"),
        "top_letter_fraction": quality.get("top_letter_fraction"),
        "unique_letters": quality.get("unique_letters"),
        "penalty": quality.get("penalty"),
    }
    return compact


def _null_mask_rank_validation_score(row: dict[str, Any]) -> float:
    """Return the ground-truth-free validation score used for null-mask ranking."""
    for field in (
        "rank_validation_score_v2",
        "confirmed_validation_score_v2",
        "promoted_validation_score_v2",
        "validation_score_v2",
    ):
        value = _float_or_none(row.get(field))
        if value is not None:
            return value
    return float("-inf")


def _null_mask_unique_rows_by_mask(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep the first row for each canonical mask in rank order."""
    if limit <= 0:
        return []
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        mask = tuple(sorted(str(symbol) for symbol in (row.get("mask") or [])))
        if mask in seen:
            continue
        seen.add(mask)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _null_mask_ranker_portfolio(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    rankers: list[str],
    language_quality_enabled: bool,
) -> list[dict[str, Any]]:
    """Return a round-robin finalist portfolio across ranker views.

    The portfolio is used to choose second-stage promotion targets.  It is
    intentionally mask-unique: promotion reruns the mask family, so including
    several same-mask rows usually spends budget without broadening basin
    coverage.
    """
    if limit <= 0:
        return []
    usable_rankers = [
        ranker for ranker in rankers
        if ranker in {"validation", "ensemble"}
        or (ranker == "language_quality" and language_quality_enabled)
    ]
    if not usable_rankers:
        usable_rankers = ["validation"]
    views = {
        ranker: sorted(
            rows,
            key=lambda row, ranker=ranker: _null_mask_ranker_sort_key(row, ranker),
            reverse=True,
        )
        for ranker in usable_rankers
    }
    positions = {ranker: 0 for ranker in usable_rankers}
    selected: list[dict[str, Any]] = []
    seen_masks: set[tuple[str, ...]] = set()
    while len(selected) < limit:
        advanced = False
        for ranker in usable_rankers:
            view = views[ranker]
            position = positions[ranker]
            while position < len(view):
                row = view[position]
                position += 1
                mask = tuple(sorted(str(symbol) for symbol in (row.get("mask") or [])))
                if mask in seen_masks:
                    continue
                seen_masks.add(mask)
                selected.append(row)
                advanced = True
                break
            positions[ranker] = position
            if len(selected) >= limit:
                break
        if not advanced:
            break
    return selected


def _null_mask_ranker_sort_key(row: dict[str, Any], ranker: str) -> tuple[float, float]:
    if ranker == "language_quality":
        lq_rank = _float_or_none(row.get("language_quality_rank_score"))
        lq_raw = _float_or_none(row.get("language_quality_raw_score"))
        return (
            lq_rank if lq_rank is not None else float("-inf"),
            lq_raw if lq_raw is not None else float("-inf"),
        )
    if ranker == "ensemble":
        ensemble = _float_or_none(row.get("ensemble_score_v1"))
        validation = _null_mask_rank_validation_score(row)
        return (
            ensemble if ensemble is not None else float("-inf"),
            validation,
        )
    validation = _null_mask_rank_validation_score(row)
    selection = _float_or_none(row.get("selection_score"))
    return (
        validation,
        selection if selection is not None else float("-inf"),
    )


def _null_mask_adaptive_decision(
    completed_rows: list[dict[str, Any]],
    *,
    consensus_report: dict[str, Any],
    min_validation_score: float,
    near_tie_margin: float,
) -> dict[str, Any]:
    """Decide whether a null-mask run should spend a broader search budget.

    This intentionally uses only solver-native and plaintext-shape signals.
    Benchmark plaintext/accuracy must never be part of this decision.
    """
    reasons: list[str] = []
    selected = completed_rows[0] if completed_rows else None
    metrics: dict[str, Any] = {
        "completed_count": len(completed_rows),
        "selected_mask": selected.get("mask") if selected else None,
    }
    if selected is None:
        return {
            "triggered": True,
            "reasons": ["no_completed_null_mask_candidate"],
            "metrics": metrics,
        }
    validation = _float_or_none(selected.get("validation_score_v2"))
    ensemble = _float_or_none(selected.get("ensemble_score_v1"))
    diagnostics = selected.get("diagnostics") if isinstance(selected.get("diagnostics"), dict) else {}
    quality = selected.get("quality") if isinstance(selected.get("quality"), dict) else {}
    components = (
        selected.get("validation_components_v2")
        if isinstance(selected.get("validation_components_v2"), dict)
        else {}
    )
    metrics.update({
        "validation_score_v2": validation,
        "ensemble_score_v1": ensemble,
        "source": selected.get("source"),
        "pseudo_word_fraction": diagnostics.get("pseudo_word_fraction"),
        "top_letter_fraction": quality.get("top_letter_fraction"),
        "collapsed": quality.get("collapsed"),
        "repetition_penalty": components.get("repetition_penalty"),
        "function_overuse_penalty": components.get("function_overuse_penalty"),
        "consensus_reason": consensus_report.get("reason"),
    })
    if validation is None or validation < min_validation_score:
        reasons.append("low_validation_score")
    try:
        pseudo_fraction = float(diagnostics.get("pseudo_word_fraction") or 0.0)
    except (TypeError, ValueError):
        pseudo_fraction = 0.0
    if pseudo_fraction >= 0.42 and (validation is None or validation < 1.0):
        reasons.append("high_pseudo_word_fraction")
    if quality.get("collapsed") and (validation is None or validation < 1.0):
        reasons.append("collapsed_or_damaged_plaintext_shape")
    if consensus_report.get("reason") == "wide_mutable_set" and (
        validation is None or validation < 1.0
    ):
        reasons.append("wide_consensus_mutable_set")
    if len(completed_rows) >= 2 and validation is not None:
        runner_up = completed_rows[1]
        runner_up_validation = _float_or_none(runner_up.get("validation_score_v2"))
        if (
            runner_up_validation is not None
            and validation - runner_up_validation <= near_tie_margin
            and validation < 1.0
        ):
            reasons.append("near_tie_finalist_menu")
            metrics["runner_up_validation_score_v2"] = runner_up_validation
            metrics["runner_up_mask"] = runner_up.get("mask")
    return {
        "triggered": bool(reasons),
        "reasons": reasons,
        "metrics": metrics,
    }


def _null_mask_bridge_pair_masks(
    rows: list[dict[str, Any]],
    *,
    candidate_symbols: list[str],
    anchor_symbols: list[str],
    seen_masks: set[tuple[str, ...]],
    max_masks: int,
) -> list[tuple[str, ...]]:
    """Generate long-range pair masks missed by the local-window generator."""
    if max_masks <= 0:
        return []
    anchors: list[str] = []
    for symbol in anchor_symbols:
        symbol = str(symbol)
        if symbol not in anchors:
            anchors.append(symbol)
    for row in rows:
        for symbol in row.get("mask") or []:
            symbol = str(symbol)
            if symbol not in anchors:
                anchors.append(symbol)
    out: list[tuple[str, ...]] = []
    for anchor in anchors:
        for symbol in candidate_symbols:
            symbol = str(symbol)
            if symbol == anchor:
                continue
            mask = tuple(sorted((anchor, symbol)))
            if mask in seen_masks:
                continue
            seen_masks.add(mask)
            out.append(mask)
            if len(out) >= max_masks:
                return out
    return out


def _null_mask_neighborhood_masks(
    rows: list[dict[str, Any]],
    *,
    candidate_symbols: list[str],
    seen_masks: set[tuple[str, ...]],
    max_size: int,
    max_masks: int,
) -> list[tuple[str, ...]]:
    """Generate add/remove/swap neighbors around strong null-mask finalists."""
    if max_masks <= 0:
        return []
    out: list[tuple[str, ...]] = []

    def add_mask(mask: tuple[str, ...]) -> bool:
        if len(out) >= max_masks:
            return False
        canonical = tuple(sorted(dict.fromkeys(mask)))
        if len(canonical) > max_size or canonical in seen_masks:
            return True
        seen_masks.add(canonical)
        out.append(canonical)
        return len(out) < max_masks

    for row in rows:
        base = tuple(sorted(str(symbol) for symbol in (row.get("mask") or [])))
        if not base:
            # The initial screen already includes every singleton candidate.
            continue
        for symbol in base:
            if not add_mask(tuple(item for item in base if item != symbol)):
                return out
        for symbol in candidate_symbols:
            symbol = str(symbol)
            if symbol in base:
                continue
            for existing in base:
                swapped = tuple(symbol if item == existing else item for item in base)
                if not add_mask(swapped):
                    return out
        for symbol in candidate_symbols:
            symbol = str(symbol)
            if symbol in base:
                continue
            if len(base) < max_size and not add_mask((*base, symbol)):
                return out
    return out


def _null_mask_consensus_polish_plan(
    rows: list[dict[str, Any]],
    *,
    cipher_text: CipherText,
    min_agreement: float,
    min_fixed_symbols: int,
    target_max_mutable_symbols: int,
) -> dict[str, Any]:
    """Build a ground-truth-free constrained rerun plan from finalist agreement."""
    source_rows = [
        row for row in rows
        if row.get("status") == "completed" and isinstance(row.get("key"), dict)
    ]
    symbol_ids = sorted(set(cipher_text.tokens))
    total_symbols = len(symbol_ids)
    report: dict[str, Any] = {
        "enabled": False,
        "source_row_count": len(source_rows),
        "fixed_symbol_count": 0,
        "mutable_symbol_count": total_symbols,
        "reason": None,
        "fixed_symbols": [],
        "mutable_symbols_sample": [],
    }
    if len(source_rows) < 2:
        report["reason"] = "insufficient_source_rows"
        return {
            "enabled": False,
            "source_rows": source_rows,
            "anchor_key": {},
            "fixed_cipher_ids": set(),
            "row_metadata": report,
            "report": report,
        }

    mask_counts: Counter[str] = Counter()
    assignment_counts: dict[int, Counter[int]] = {sid: Counter() for sid in symbol_ids}
    considered_counts: Counter[int] = Counter()
    for row in source_rows:
        row_mask = {str(symbol) for symbol in (row.get("mask") or [])}
        mask_counts.update(row_mask)
        row_key = {
            int(k): int(v)
            for k, v in (row.get("key") or {}).items()
        }
        for sid in symbol_ids:
            symbol = cipher_text.alphabet.symbol_for(sid)
            if symbol in row_mask or sid not in row_key:
                continue
            assignment_counts[sid][row_key[sid]] += 1
            considered_counts[sid] += 1

    source_count = len(source_rows)
    required_considered = max(2, math.ceil(source_count * 0.5))
    anchor_key: dict[int, int] = {}
    confidence_rows: list[dict[str, Any]] = []
    for sid in symbol_ids:
        symbol = cipher_text.alphabet.symbol_for(sid)
        considered = considered_counts[sid]
        if considered < required_considered:
            continue
        if mask_counts[symbol] / source_count >= 0.35:
            continue
        if not assignment_counts[sid]:
            continue
        plaintext_id, agreement_count = assignment_counts[sid].most_common(1)[0]
        agreement = agreement_count / max(1, considered)
        if agreement < min_agreement:
            continue
        anchor_key[sid] = plaintext_id
        confidence_rows.append({
            "symbol_id": sid,
            "symbol": symbol,
            "plaintext_id": plaintext_id,
            "agreement": round(agreement, 4),
            "agreement_count": agreement_count,
            "considered_count": considered,
            "masked_count": mask_counts[symbol],
        })

    fixed_ids = set(anchor_key)
    mutable_ids = set(symbol_ids) - fixed_ids
    confidence_rows.sort(
        key=lambda item: (
            float(item["agreement"]),
            int(item["agreement_count"]),
            int(item["considered_count"]),
        ),
        reverse=True,
    )
    report.update({
        "fixed_symbol_count": len(fixed_ids),
        "mutable_symbol_count": len(mutable_ids),
        "fixed_symbols": confidence_rows[:24],
        "mutable_symbols_sample": [
            cipher_text.alphabet.symbol_for(sid)
            for sid in sorted(mutable_ids)[:24]
        ],
        "masked_symbol_counts": dict(mask_counts),
        "required_considered_count": required_considered,
    })
    if len(fixed_ids) < min_fixed_symbols:
        report["reason"] = "insufficient_consensus_fixed_symbols"
        return {
            "enabled": False,
            "source_rows": source_rows,
            "anchor_key": anchor_key,
            "fixed_cipher_ids": fixed_ids,
            "row_metadata": report,
            "report": report,
        }
    if len(mutable_ids) < 1:
        report["reason"] = "no_mutable_symbols"
        return {
            "enabled": False,
            "source_rows": source_rows,
            "anchor_key": anchor_key,
            "fixed_cipher_ids": fixed_ids,
            "row_metadata": report,
            "report": report,
        }

    report["enabled"] = True
    report["reason"] = (
        "wide_mutable_set"
        if target_max_mutable_symbols > 0 and len(mutable_ids) > target_max_mutable_symbols
        else "consensus_ready"
    )
    return {
        "enabled": True,
        "source_rows": source_rows,
        "anchor_key": anchor_key,
        "fixed_cipher_ids": fixed_ids,
        "row_metadata": report,
        "report": report,
    }


def _attach_null_mask_stability(
    row: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> None:
    """Attach same-mask stability evidence and rank by robust validation.

    The candidate text/key for a mask still comes from its best observed basin,
    but the ranking score is conservative across independent reruns. This
    prevents a single lucky-looking damaged plaintext from dominating the menu.
    """
    candidates = [row] + probe_rows
    completed = [
        candidate for candidate in candidates
        if candidate.get("status") == "completed"
        and _float_or_none(candidate.get("validation_score_v2")) is not None
    ]
    if not completed:
        row["stability"] = {
            "status": "no_completed_stability_runs",
            "probe_count": len(probe_rows),
        }
        return
    scores = [
        float(candidate["validation_score_v2"])
        for candidate in completed
    ]
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    stddev = math.sqrt(variance)
    best = max(
        completed,
        key=lambda item: float(item.get("validation_score_v2") or float("-inf")),
    )
    best_score = float(best.get("validation_score_v2") or float("-inf"))
    worst_score = min(scores)
    robust_score = (0.70 * mean_score) + (0.30 * best_score) - min(0.20, 0.50 * stddev)

    for field in (
        "solver",
        "anneal_score",
        "selection_score",
        "validation_score_v2",
        "validation_components_v2",
        "quality",
        "diagnostics",
        "preview",
        "decryption",
        "key",
        "budget",
        "engine",
        "best_seed",
        "batch_threads",
    ):
        if field in best:
            row[field] = best[field]
    row["rank_validation_score_v2"] = round(robust_score, 6)
    row["adaptive_stability_score_v2"] = row["rank_validation_score_v2"]
    row["adaptive_stability_best_validation_score_v2"] = round(best_score, 6)
    row["adaptive_stability_mean_validation_score_v2"] = round(mean_score, 6)
    row["adaptive_stability_stddev_validation_score_v2"] = round(stddev, 6)
    row["stability"] = {
        "status": "completed",
        "probe_count": len(probe_rows),
        "completed_probe_count": len(completed) - 1,
        "mean_validation_score_v2": round(mean_score, 6),
        "best_validation_score_v2": round(best_score, 6),
        "worst_validation_score_v2": round(worst_score, 6),
        "stddev_validation_score_v2": round(stddev, 6),
        "rank_validation_score_v2": row["rank_validation_score_v2"],
        "best_source": best.get("source"),
        "best_budget": best.get("budget"),
        "runs": [_compact_null_mask_row(candidate) for candidate in completed],
    }


def _attach_null_mask_confirmation(
    row: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> None:
    """Attach independent-rerun confirmation stats to one null-mask finalist."""
    candidates = [row] + probe_rows
    completed = [
        candidate for candidate in candidates
        if candidate.get("status") == "completed"
        and candidate.get("validation_score_v2") is not None
    ]
    if not completed:
        row["confirmation"] = {
            "status": "no_completed_confirmation_runs",
            "probe_count": len(probe_rows),
        }
        return
    scores = [float(candidate["validation_score_v2"]) for candidate in completed]
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    stddev = math.sqrt(variance)
    best = max(completed, key=lambda item: float(item.get("validation_score_v2") or float("-inf")))
    worst_score = min(scores)
    stability_score = 1.0 / (1.0 + stddev)
    confirmed_score = mean_score + min(0.12, stability_score * 0.04)

    # Keep the mask-level decision robust, but use the best confirmed basin as
    # the representative decrypt/key if this mask wins.
    for field in (
        "solver",
        "anneal_score",
        "selection_score",
        "quality",
        "diagnostics",
        "preview",
        "decryption",
        "key",
    ):
        if field in best:
            row[field] = best[field]
    row["confirmed_validation_score_v2"] = round(confirmed_score, 6)
    row["confirmation_best_validation_score_v2"] = round(float(best["validation_score_v2"]), 6)
    row["confirmation"] = {
        "status": "completed",
        "probe_count": len(probe_rows),
        "completed_probe_count": len(completed) - 1,
        "mean_validation_score_v2": round(mean_score, 6),
        "best_validation_score_v2": round(float(best["validation_score_v2"]), 6),
        "worst_validation_score_v2": round(worst_score, 6),
        "stddev_validation_score_v2": round(stddev, 6),
        "stability_score": round(stability_score, 6),
        "confirmed_validation_score_v2": row["confirmed_validation_score_v2"],
        "runs": [_compact_null_mask_row(candidate) for candidate in completed],
    }


def _attach_null_mask_promotion(
    row: dict[str, Any],
    probe_rows: list[dict[str, Any]],
) -> None:
    """Promote one finalist with deeper same-mask solves and keep the best basin."""
    candidates = [row] + probe_rows
    completed = [
        candidate for candidate in candidates
        if candidate.get("status") == "completed"
        and candidate.get("validation_score_v2") is not None
    ]
    if not completed:
        row["promotion"] = {
            "status": "no_completed_promotion_runs",
            "probe_count": len(probe_rows),
        }
        return
    initial_score = row.get("validation_score_v2")
    best = max(completed, key=lambda item: float(item.get("validation_score_v2") or float("-inf")))
    best_score = float(best.get("validation_score_v2") or float("-inf"))
    try:
        initial_score_value = float(initial_score)
    except (TypeError, ValueError):
        initial_score_value = float("-inf")
    adoption_margin = float(os.environ.get("DECIPHER_NULL_MASK_PROMOTE_ADOPTION_MARGIN", "0.08"))
    adopted = best is row or best_score >= initial_score_value + adoption_margin
    representative = best if adopted else row
    for field in (
        "solver",
        "anneal_score",
        "selection_score",
        "validation_score_v2",
        "validation_components_v2",
        "quality",
        "diagnostics",
        "preview",
        "decryption",
        "key",
        "budget",
    ):
        if field in representative:
            row[field] = representative[field]
    row["promoted_validation_score_v2"] = row.get("validation_score_v2")
    row["promotion"] = {
        "status": "completed",
        "probe_count": len(probe_rows),
        "completed_probe_count": len(completed) - 1,
        "initial_validation_score_v2": initial_score,
        "best_validation_score_v2": row.get("validation_score_v2"),
        "best_probe_validation_score_v2": round(best_score, 6),
        "adoption_margin": adoption_margin,
        "adopted_promotion": adopted,
        "best_source": best.get("source"),
        "best_budget": best.get("budget"),
        "representative_source": representative.get("source"),
        "runs": [_compact_null_mask_row(candidate) for candidate in completed],
    }


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


def _maybe_rescue_substitution_run(
    *,
    cipher_text: CipherText,
    language: str,
    cipher_id: str,
    initial_solver: str,
    initial_key: dict[int, int],
    initial_decryption: str,
    initial_step: dict[str, Any],
) -> dict[str, Any] | None:
    """Rerun substitution anneal only when solver score suggests a bad basin."""

    if not initial_decryption:
        return None
    if not _env_bool("DECIPHER_SUBSTITUTION_RESCUE", True):
        return None
    threshold = _substitution_rescue_min_score()
    max_attempts = _substitution_rescue_attempts()
    if max_attempts <= 0:
        return None
    initial_internal_score = _float_or_none(initial_step.get("score"))
    if initial_internal_score is None or initial_internal_score >= threshold:
        return None

    best_solver = initial_solver
    best_key = dict(initial_key)
    best_decryption = initial_decryption
    best_internal_score = initial_internal_score
    selected_attempt_index: int | None = None
    attempts: list[dict[str, Any]] = []
    started = time.time()
    for attempt_index in range(1, max_attempts + 1):
        solver, key, decryption, step = _run_substitution(cipher_text, language)
        internal_score = _float_or_none(step.get("score"))
        if internal_score is None:
            continue
        attempt_summary = {
            "attempt_index": attempt_index,
            "solver": solver,
            "score": round(internal_score, 6),
            "elapsed_seconds": step.get("elapsed_seconds"),
            "selected": False,
        }
        if internal_score > best_internal_score:
            best_solver = solver
            best_key = dict(key)
            best_decryption = decryption
            best_internal_score = internal_score
            selected_attempt_index = attempt_index
            attempt_summary["selected"] = True
        attempts.append(attempt_summary)
        if best_internal_score >= threshold:
            break

    step = {
        "name": "substitution_rescue_restarts",
        "trigger": "initial_solver_score_below_threshold",
        "enabled": True,
        "threshold": threshold,
        "max_additional_attempts": max_attempts,
        "attempt_count": len(attempts),
        "initial_score": round(initial_internal_score, 6),
        "best_score": round(best_internal_score, 6),
        "selected_attempt_index": selected_attempt_index,
        "improved": selected_attempt_index is not None,
        "elapsed_seconds": round(time.time() - started, 3),
        "attempts": attempts,
        "policy": (
            "Substitution rescue keeps ordinary runs stochastic and cheap, but "
            "when the first anneal score lands below the configured threshold, "
            "it spends a few extra independent anneal attempts and keeps the "
            "best solver-scored result. Benchmark ground truth is deliberately "
            "not available to this rescue path."
        ),
    }
    return {
        "solver": best_solver,
        "key": best_key,
        "decryption": best_decryption,
        "selected_attempt_index": selected_attempt_index,
        "step": step,
}


def _substitution_rescue_min_score() -> float:
    raw = os.environ.get("DECIPHER_SUBSTITUTION_RESCUE_MIN_SCORE", "-5.35").strip() or "-5.35"
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("DECIPHER_SUBSTITUTION_RESCUE_MIN_SCORE must be a float") from exc
    return value


def _substitution_rescue_attempts() -> int:
    raw = os.environ.get("DECIPHER_SUBSTITUTION_RESCUE_ATTEMPTS", "3").strip() or "3"
    try:
        return max(0, int(raw))
    except ValueError as exc:
        raise ValueError("DECIPHER_SUBSTITUTION_RESCUE_ATTEMPTS must be an integer >= 0") from exc


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


# Per-THREAD active model-variant selection. Consulted by
# ``_zenith_native_model_path`` when no explicit ``variant`` is passed, so a run
# can select a non-default variant without threading the slug through every deep
# solver call site. ``run_automated`` sets the calling thread's slot from its
# ``model_variant`` param and restores the prior value in a ``finally`` (see the
# wrapper below); the agent's ``act_set_model_variant`` passes its selection
# explicitly at the direct call sites. A ``threading.local`` (not a plain
# global) so concurrent ``run_automated`` calls on different threads cannot
# clobber each other's selection; every internal resolution site runs on the
# thread that called ``run_automated`` (seed workers receive the already-
# resolved ``model_path``). ``None`` == default resolution (byte-identical to
# before). Fresh threads start at ``None``.
_ACTIVE_MODEL_VARIANT_SLOT = threading.local()


def _get_active_model_variant() -> str | None:
    """Return the calling thread's active model variant (default ``None``)."""
    return getattr(_ACTIVE_MODEL_VARIANT_SLOT, "value", None)


def set_active_model_variant(variant: str | None) -> str | None:
    """Set the calling thread's active model variant; return the previous value."""
    prev = _get_active_model_variant()
    _ACTIVE_MODEL_VARIANT_SLOT.value = variant
    return prev


def _zenith_native_model_path(
    language: str = "en", variant: str | None = None
) -> Path | None:
    """Locate the binary model file for the ``zenith_native`` profile.

    Delegates to :mod:`analysis.model_registry`. ``variant=None`` falls back to
    the calling thread's active selection (see ``set_active_model_variant``);
    passing ``variant`` explicitly overrides it. The models directory is
    anchored on this module's ``__file__`` (not the registry's) so tests that
    monkeypatch ``automated_runner.__file__`` keep pinning the resolver.
    """
    if variant is None:
        variant = _get_active_model_variant()
    models_dir = Path(__file__).resolve().parents[2] / "models"
    return model_registry.resolve_language_model(
        language, variant=variant, models_dir=models_dir
    )


@functools.lru_cache(maxsize=16)
def _zenith_native_model_metadata(path_text: str) -> dict[str, Any]:
    """Return artifact-safe provenance for a Zenith binary model path."""
    path = Path(path_text).expanduser()
    resolved = path.resolve() if path.exists() else path
    metadata_path = path.with_name(path.name + ".metadata.json")
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {"metadata_parse_error": str(metadata_path)}
    sha256 = metadata.get("sha256")
    if not sha256 and path.exists():
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        sha256 = h.hexdigest()
    out: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256,
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
    }
    for key in (
        "language",
        "variant",
        "display_label",
        "order",
        "format",
        "output_file",
        "unknown_log_prob",
        "build_timestamp_utc",
        "builder_version",
        "sources",
        "corpus_stats",
        "normalization",
        "redistribution_status",
    ):
        if key in metadata:
            out[key] = metadata[key]
    return out


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
    initial_key: dict[int, int] | None = None,
    fixed_cipher_ids: set[int] | None = None,
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
                    initial_key=initial_key,
                    fixed_cipher_ids=fixed_cipher_ids,
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
                        initial_key=initial_key,
                        fixed_cipher_ids=fixed_cipher_ids,
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
                        initial_key=initial_key,
                        fixed_cipher_ids=fixed_cipher_ids,
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
        "model_metadata": _zenith_native_model_metadata(str(bin_path)),
        "model_note": "zenith_binary",
        "engine": engine,
        "homophonic_budget": budget,
        "budget_params": budget_params,
        "homophonic_refinement": "none",
        "initial_key_provided": initial_key is not None,
        "fixed_cipher_ids_count": len(fixed_cipher_ids or set()),
        "fixed_cipher_ids_sample": sorted(fixed_cipher_ids or set())[:20],
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
            binary_model_path=bin_path,
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


def _zenith_text_mean_log_prob(text: str, bin_path: Path | None) -> float | None:
    """Mean 5-gram log probability for an already rendered plaintext."""
    if bin_path is None:
        return None
    try:
        from analysis.zenith_solver import load_zenith_binary_model

        model = load_zenith_binary_model(bin_path)
    except Exception:  # noqa: BLE001
        return None
    letters = [
        ord(ch.lower()) - 97
        for ch in text.upper()
        if "A" <= ch <= "Z"
    ]
    if len(letters) < 5:
        return None
    total = 0.0
    count = 0
    for i in range(len(letters) - 4):
        total += model.lookup_lo(
            letters[i], letters[i + 1], letters[i + 2], letters[i + 3], letters[i + 4]
        )
        count += 1
    return total / max(1, count)


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
    # Specific override > global override > auto-size from CPU count
    raw = (
        os.environ.get("DECIPHER_HOMOPHONIC_PARALLEL_SEEDS")
        or os.environ.get("DECIPHER_PARALLEL_WORKERS")
    )
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
            "DECIPHER_HOMOPHONIC_PARALLEL_SEEDS / DECIPHER_PARALLEL_WORKERS"
            " must be an integer >= 1"
        ) from exc
    value = max(1, value)
    if seed_count is not None:
        value = min(value, max(1, seed_count))
    return value


def _null_mask_batch_threads() -> int:
    raw = (
        os.environ.get("DECIPHER_NULL_MASK_THREADS")
        or os.environ.get("DECIPHER_TRANSFORM_RANK_THREADS")
        or os.environ.get("DECIPHER_PARALLEL_WORKERS")
        or "0"
    )
    try:
        value = int(str(raw).strip() or "0")
    except ValueError as exc:
        raise ValueError(
            "DECIPHER_NULL_MASK_THREADS / DECIPHER_TRANSFORM_RANK_THREADS / "
            "DECIPHER_PARALLEL_WORKERS must be an integer >= 0"
        ) from exc
    return max(0, value)


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
    # Specific override > global override > 0 (let Rayon/OS decide thread count)
    raw = (
        os.environ.get("DECIPHER_TRANSFORM_RANK_THREADS")
        or os.environ.get("DECIPHER_PARALLEL_WORKERS")
        or "0"
    )
    raw = raw.strip() or "0"
    try:
        return max(0, int(raw))
    except ValueError as exc:
        raise ValueError(
            "DECIPHER_TRANSFORM_RANK_THREADS / DECIPHER_PARALLEL_WORKERS"
            " must be an integer >= 0"
        ) from exc


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
    initial_key: dict[int, int] | None = None,
    fixed_cipher_ids: set[int] | None = None,
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
            initial_key=initial_key,
            fixed_cipher_ids=fixed_cipher_ids,
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
        initial_key=initial_key,
        fixed_cipher_ids=fixed_cipher_ids,
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
        "(expected one of: none, two_stage, targeted_repair, family_repair, null_masks)"
    )


def _automated_candidate_diagnostics(
    plaintext: str,
    language: str,
    word_list: list[str],
    binary_model_path: Path | None = None,
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
        word_count = len(seg.words)
        pseudo_word_count = len(seg.pseudo_words)
        short_word_count = sum(1 for word in seg.words if len(word) <= 2)
        long_pseudo_word_count = sum(1 for word in seg.pseudo_words if len(word) >= 8)
        diagnostics.update({
            "dict_rate": round(seg.dict_rate, 4),
            "segmentation_cost": round(seg.cost, 3),
            "segmented_preview": seg.segmented[:160],
            "segmented_word_count": word_count,
            "pseudo_word_count": pseudo_word_count,
            "pseudo_word_fraction": round(pseudo_word_count / word_count, 4) if word_count else 0.0,
            "short_word_fraction": round(short_word_count / word_count, 4) if word_count else 0.0,
            "long_pseudo_word_fraction": round(long_pseudo_word_count / word_count, 4) if word_count else 0.0,
        })
        diagnostics.update(content_word_metrics(seg.words, word_set, language))
    binary_score = _zenith_text_mean_log_prob(plaintext, binary_model_path)
    if binary_score is not None:
        diagnostics["binary_ngram_mean_log_prob"] = round(binary_score, 6)
        diagnostics["binary_ngram_model_source"] = str(binary_model_path)
    return diagnostics


# ---------------------------------------------------------------------------
# Public helper wrappers
#
# Promoted ``src/analysis`` modules (e.g. ``analysis.multipage``) must not
# import the underscore-private helpers above across module boundaries. These
# thin public aliases expose the same callables under stable public names so
# library code can depend on a public surface. Behavior is identical -- the
# private names remain as aliases for existing internal call sites.
# ---------------------------------------------------------------------------

run_homophonic_search = _run_homophonic
cipher_text_from_tokens = _cipher_text_from_tokens
automated_candidate_diagnostics = _automated_candidate_diagnostics
# NB: the public name ``plaintext_quality`` (not ``plaintext_quality_score``)
# because ``plaintext_quality_score`` is already an imported symbol in this
# module (a different, unrelated function). Deviation from the spec's suggested
# wrapper name, forced by the name collision.
plaintext_quality = _plaintext_quality
load_word_list = _word_list
# Repo-root-anchored binary-model resolver (expanduser + language
# normalization). Promoted libraries (``analysis.word_hypothesis_repair`` via
# the runner's word-repair refinement) route model resolution through this
# public alias so they never hit ``multipage``'s CWD-relative fallback.
zenith_native_model_path = _zenith_native_model_path
