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
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis import dictionary, homophonic, ic, ngram, pattern
from analysis.segment import segment_text
from analysis.solver import simulated_anneal
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
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.default_language = language
        self.verbose = verbose
        self.homophonic_budget = homophonic_budget
        self.homophonic_refinement = homophonic_refinement

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

        result = run_automated(
            cipher_text=cipher_text,
            language=lang,
            cipher_id=test_id,
            ground_truth=test_data.plaintext,
            cipher_system=test_data.test.cipher_system,
            homophonic_budget=self.homophonic_budget,
            homophonic_refinement=self.homophonic_refinement,
        )
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
    homophonic_budget: str = "full",
    homophonic_refinement: str = "none",
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
    routing = _select_solver_path(cipher_text, language, cipher_system)

    try:
        steps.append({
            "name": "route_automated_solver",
            "solver": routing["solver"],
            "route": routing["route"],
            "reason": routing["reason"],
            "cipher_system": cipher_system,
            "language": language,
            "homophonic_budget": homophonic_budget,
            "homophonic_refinement": homophonic_refinement,
        })
        if routing["route"] == "homophonic":
            solver, key, decryption, step = _run_homophonic(
                cipher_text,
                language,
                budget=homophonic_budget,
                refinement=homophonic_refinement,
                ground_truth=ground_truth,
            )
            steps.append(step)
        else:
            solver, key, decryption, step = _run_substitution(cipher_text, language)
            steps.append(step)
        status = "completed" if decryption else "error"
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
) -> dict[str, str]:
    pt_alpha = _plaintext_alphabet(language)
    cipher_name = cipher_system.lower()
    alphabet_size = cipher_text.alphabet.size
    word_groups = len(cipher_text.words)

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


def _run_homophonic(
    cipher_text: CipherText,
    language: str,
    budget: str = "full",
    refinement: str = "none",
    ground_truth: str | None = None,
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
    score_profile = _homophonic_score_profile()

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
    session.plaintext_alphabet = Alphabet.standard_english()
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

    step = {
        "name": "search_anneal",
        "solver": "native_substitution_anneal",
        "score": round(best_score, 4),
        "restarts": restarts,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return "native_substitution_anneal", best_key, best_decryption, step


def _run_substitution_continuous(
    cipher_text: CipherText,
    language: str,
) -> tuple[str, dict[int, int], str, dict[str, Any]]:
    pt_alpha = _plaintext_alphabet(language)
    plaintext_ids = list(range(pt_alpha.size))
    id_to_letter = {i: pt_alpha.symbol_for(i).upper() for i in plaintext_ids}
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
    step = {
        "name": "search_substitution_continuous_anneal",
        "solver": "native_substitution_continuous_anneal",
        "model_source": model.source,
        "model_note": model_note,
        "anneal_score": round(result.normalized_score, 4),
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "epochs": result.epochs,
        "sampler_iterations": result.sampler_iterations,
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
    session.set_full_key(result.key)
    decryption = session.apply_key()
    return "native_substitution_continuous_anneal", result.key, decryption, step


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
                ): seed
                for seed in seeds
            }
            for future in concurrent.futures.as_completed(future_map):
                seed = future_map[future]
                seed_results.append((seed, future.result()))
        seed_results.sort(key=lambda item: seeds.index(item[0]))
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

    step: dict[str, Any] = {
        "name": "search_homophonic_anneal",
        "solver": "zenith_native",
        "model_source": str(bin_path),
        "model_note": "zenith_binary",
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
        "elapsed_seconds": round(time.time() - started, 3),
        "epochs": best_result.epochs,
        "sampler_iterations": best_result.sampler_iterations,
        "parallel_seed_workers": parallel_seed_workers,
        "seed_attempts": attempts,
    }
    return "zenith_native", selected_key, selected_plaintext, step


def _homophonic_score_profile() -> str:
    return (
        os.environ.get("DECIPHER_HOMOPHONIC_SCORE_PROFILE", "balanced")
        .strip()
        .lower()
        or "balanced"
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
):
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
