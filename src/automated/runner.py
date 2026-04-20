"""Automated-only solving without LLM API calls.

This module deliberately stays separate from ``agent.loop_v2``. It uses the
same native solver building blocks exposed to the agent tools, but runs them
deterministically from local code and writes a small dashboard-compatible
artifact marked ``run_mode: automated_only``.
"""
from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis import dictionary, homophonic, ngram, pattern
from analysis.solver import simulated_anneal
from benchmark.loader import TestData, parse_canonical_transcription
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
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.default_language = language
        self.verbose = verbose

    def _resolve_language(self, test_data: TestData) -> str:
        if self.default_language:
            return self.default_language
        source = test_data.test.test_id.split("_")[0]
        return {"borg": "la", "copiale": "de", "it": "it", "fr": "fr"}.get(source, "en")

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

    try:
        if _should_use_homophonic(cipher_text):
            solver, key, decryption, step = _run_homophonic(cipher_text, language)
            steps.append(step)
        else:
            solver, key, decryption, step = _run_substitution(cipher_text, language)
            steps.append(step)
        status = "solved" if decryption else "error"
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


def _should_use_homophonic(cipher_text: CipherText) -> bool:
    return cipher_text.alphabet.size > 26 and len(cipher_text.words) <= 1


def _run_homophonic(
    cipher_text: CipherText,
    language: str,
) -> tuple[str, dict[int, int], str, dict[str, Any]]:
    pt_alpha = Alphabet.standard_english()
    plaintext_ids = list(range(pt_alpha.size))
    id_to_letter = {i: pt_alpha.symbol_for(i).upper() for i in plaintext_ids}
    letter_to_id = {letter: i for i, letter in id_to_letter.items()}
    word_list = _word_list(language)
    model, model_note = _homophonic_model(language, word_list)
    result = homophonic.homophonic_simulated_anneal(
        tokens=list(cipher_text.tokens),
        plaintext_ids=plaintext_ids,
        id_to_letter=id_to_letter,
        letter_to_id=letter_to_id,
        model=model,
        epochs=7,
        sampler_iterations=3000,
        distribution_weight=5.0,
        top_n=3,
    )
    step = {
        "name": "search_homophonic_anneal",
        "solver": "native_homophonic_anneal",
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
    return "native_homophonic_anneal", result.key, result.plaintext, step


def _run_substitution(
    cipher_text: CipherText,
    language: str,
) -> tuple[str, dict[int, int], str, dict[str, Any]]:
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
