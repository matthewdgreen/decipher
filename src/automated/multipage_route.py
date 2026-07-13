"""Multipage shared-key benchmark route (Improvement Program Phase 2.4).

A *page group* is a set of benchmark pages that share one symbol inventory and
(hypothetically) one substitution key. This route combines the group into one
ciphertext, runs the existing automated homophonic route ONCE on the combined
cipher (base solve + optional null-mask bakeoff), projects the shared key back
onto each page, optionally runs the word-hypothesis repair refinement on the
whole GROUP (the cross-page evidence case the library was built for), and emits
one artifact per page plus one group artifact.

The route deliberately reuses ``automated.runner`` internals rather than
forking any solver logic:

* the combined solve is a single :func:`automated.runner.run_automated` call on
  the combined :class:`~models.cipher_text.CipherText`;
* word repair reuses :func:`automated.runner._word_repair_refinement_on_pages`
  (the same composed gate + adoption the single-page route uses), passed the
  real multi-page group so cross-page collateral is real;
* projection/scoring reuse ``analysis.multipage``.

Standing lazy-import constraint (see ``tests/test_word_repair_runner.py`` /
``tests/test_multipage_route.py``): ``analysis.multipage`` imports
``automated.runner`` at its module top, so importing it (or
``analysis.word_hypothesis_repair``) at THIS module's top would risk a
partially-initialized runner. Those promoted libraries are therefore imported
inside function bodies only. ``automated.runner`` itself is fully loaded by the
time this module is imported, so it is imported normally at the top.

Ground-truth firewall
----------------------
The combined solve is run with ``ground_truth=None``; benchmark plaintext is
consumed only by the post-hoc per-page scorer (``attach_page_scores``), never by
the solve, the null-mask bakeoff, or the word-repair menu. Per-page plaintext is
retained only on the per-page artifact's ``ground_truth`` grading field.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from automated import runner as _runner

_TOKEN_RE = re.compile(r"S\d+")
_SYMBOL_INVENTORY_POLICY = "s_token_space"

# Group-definition JSON keys required by ``load_group_definition``.
_REQUIRED_GROUP_KEYS = ("name", "benchmark_split", "test_ids", "language")

# Path-safe slug for group names (F5): the name becomes an artifact directory
# component (<artifact_dir>/automated_multipage/<name>/...), so separators,
# dots, and any other path metacharacters are rejected outright.
_GROUP_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass
class MultipagePageResult:
    """Post-hoc summary for one projected page of a group run."""

    test_id: str
    status: str
    token_count: int
    filtered_length: int
    char_accuracy: float
    word_accuracy: float
    decryption: str
    artifact_path: str = ""


@dataclass
class MultipageRunResult:
    """Result of a single :func:`run_automated_multipage` call."""

    group_name: str
    run_id: str
    status: str
    solver: str
    language: str
    test_ids: list[str]
    homophonic_budget: str
    homophonic_refinement: str
    homophonic_solver: str
    combined_token_count: int
    combined_alphabet_size: int
    combined_cipher_sha256: str
    selected_mask: list[str]
    combined_solve_seconds: float
    elapsed_seconds: float
    aggregate_char_accuracy: float
    aggregate_word_accuracy: float
    word_repair_adopt_enabled: bool
    page_results: list[MultipagePageResult] = field(default_factory=list)
    word_repair: dict[str, Any] | None = None
    group_artifact_path: str = ""
    error_message: str = ""


def load_group_definition(path: str | Path) -> dict[str, Any]:
    """Load and validate a ``frontier/groups/<name>.json`` group definition.

    Required keys: ``name``, ``benchmark_split``, ``test_ids`` (non-empty list),
    ``language``. ``description`` is optional. Raises ``ValueError`` on any
    structural problem.
    """
    group_path = Path(path).expanduser()
    if not group_path.exists():
        raise ValueError(f"Group definition not found: {group_path}")
    try:
        data = json.loads(group_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Group definition is not valid JSON ({group_path}): {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Group definition must be a JSON object: {group_path}")
    missing = [key for key in _REQUIRED_GROUP_KEYS if key not in data]
    if missing:
        raise ValueError(
            f"Group definition {group_path} is missing required key(s): {', '.join(missing)}"
        )
    test_ids = data["test_ids"]
    if not isinstance(test_ids, list) or not test_ids or not all(
        isinstance(item, str) and item for item in test_ids
    ):
        raise ValueError(
            f"Group definition {group_path}: 'test_ids' must be a non-empty list of strings"
        )
    if len(set(test_ids)) != len(test_ids):
        raise ValueError(f"Group definition {group_path}: 'test_ids' contains duplicates")
    if not isinstance(data["name"], str) or not data["name"]:
        raise ValueError(f"Group definition {group_path}: 'name' must be a non-empty string")
    if not _GROUP_NAME_RE.fullmatch(data["name"]):
        raise ValueError(
            f"Group definition {group_path}: 'name' must be a path-safe slug "
            f"([A-Za-z0-9][A-Za-z0-9_-]*; no separators or dots), got {data['name']!r}"
        )
    if not isinstance(data["benchmark_split"], str) or not data["benchmark_split"]:
        raise ValueError(
            f"Group definition {group_path}: 'benchmark_split' must be a non-empty string"
        )
    if not isinstance(data["language"], str) or not data["language"]:
        raise ValueError(f"Group definition {group_path}: 'language' must be a non-empty string")
    return data


def _page_symbols(canonical_transcription: str) -> list[str]:
    """Split a canonical transcription into ciphertext symbols (word/page
    separators dropped)."""
    return [
        token
        for token in str(canonical_transcription).split()
        if token != "|"
    ]


def validate_shared_symbol_inventory(
    loader: Any, tests: dict[str, Any], test_ids: list[str]
) -> None:
    """Reject a page group whose pages do not share one symbol-inventory policy.

    The canonical benchmark inventory is the S-token space (``S\\d+``). Every
    page in the group must transcribe purely into S-tokens; a page carrying any
    non-S-token symbol (e.g. a raw-letter transcription) would silently pollute
    the shared alphabet and is rejected here, before combining. Raises
    ``ValueError`` naming the offending page and symbol.
    """
    for test_id in test_ids:
        data = loader.load_test_data(tests[test_id])
        symbols = _page_symbols(data.canonical_transcription)
        if not symbols:
            raise ValueError(
                f"Page {test_id!r} has an empty transcription; cannot combine into a group."
            )
        offending = [symbol for symbol in symbols if not _TOKEN_RE.fullmatch(symbol)]
        if offending:
            sample = ", ".join(sorted(set(offending))[:5])
            raise ValueError(
                f"Page {test_id!r} violates the shared symbol-inventory policy "
                f"({_SYMBOL_INVENTORY_POLICY}): non-S-token symbol(s) present "
                f"({sample}). All pages in a group must use the same S-token space."
            )


def _selected_null_mask(artifact: dict[str, Any]) -> tuple[str, ...]:
    """Read the null-mask bakeoff winner's mask from a combined-solve artifact.

    Returns ``()`` when the combined solve ran no null-mask bakeoff (base
    homophonic route or a non-null-mask refinement)."""
    for step in reversed(artifact.get("steps") or []):
        if not isinstance(step, dict) or step.get("name") != "search_null_masks":
            continue
        selected = step.get("selected")
        if isinstance(selected, dict):
            return tuple(str(symbol) for symbol in (selected.get("mask") or []))
    return ()


def _split_refinement(refinement: str) -> tuple[str, bool]:
    """Split a group refinement into ``(solve_refinement, run_word_repair)``.

    The combined solve runs the null-mask portion (if any) but never the
    single-page word-repair (which the group route runs separately, on the real
    page group). Reuses the runner's refinement predicates so the null-mask
    aliases stay in sync.
    """
    runs_word_repair = _runner._is_word_repair_refinement(refinement)
    runs_null_masks = _runner._refinement_runs_null_masks(refinement)
    if not runs_word_repair:
        # Plain refinements (none, null_masks, two_stage, ...) pass through: the
        # combined solve reproduces exactly what a single-page run would do.
        return refinement, False
    solve_refinement = "null_masks" if runs_null_masks else "none"
    return solve_refinement, True


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_automated_multipage(
    *,
    loader: Any,
    group: dict[str, Any],
    homophonic_budget: str = "full",
    homophonic_refinement: str = "none",
    homophonic_solver: str = "zenith_native",
    language: str | None = None,
    artifact_dir: str | Path = "artifacts",
    write_artifacts: bool = True,
) -> MultipageRunResult:
    """Run the multipage shared-key route for one page group.

    ``loader`` supplies ``load_tests(split)`` and ``load_test_data(test)`` (a
    :class:`~benchmark.loader.BenchmarkLoader`). ``group`` is a validated group
    definition dict (see :func:`load_group_definition`). ``language`` overrides
    the group's declared language when supplied.
    """
    # Lazy imports of the promoted libraries (see module docstring).
    from analysis.multipage import (
        attach_page_scores,
        build_combined_cipher,
        project_pages,
    )

    started = time.time()
    run_id = uuid.uuid4().hex[:12]
    group_name = str(group["name"])
    test_ids = list(group["test_ids"])
    split = str(group["benchmark_split"])
    lang = language or str(group.get("language") or "de")

    tests = {test.test_id: test for test in loader.load_tests(split)}
    missing = [test_id for test_id in test_ids if test_id not in tests]
    if missing:
        raise ValueError(
            f"Group {group_name!r}: test id(s) not found in split {split!r}: "
            f"{', '.join(missing)}"
        )

    validate_shared_symbol_inventory(loader, tests, test_ids)
    combined_cipher, pages = build_combined_cipher(
        loader, tests, test_ids, source=f"multipage:{group_name}"
    )
    combined_sha = hashlib.sha256(combined_cipher.raw.encode("utf-8")).hexdigest()

    solve_refinement, run_word_repair = _split_refinement(homophonic_refinement)
    group_cipher_id = f"multipage_{group_name}_{run_id}"

    solve_started = time.time()
    # Ground truth is intentionally withheld from the solve: per-page grading is
    # post-hoc only (attach_page_scores below).
    combined_result = _runner.run_automated(
        combined_cipher,
        language=lang,
        cipher_id=group_cipher_id,
        ground_truth=None,
        cipher_system="homophonic",
        homophonic_budget=homophonic_budget,
        homophonic_refinement=solve_refinement,
        homophonic_solver=homophonic_solver,
    )
    combined_solve_seconds = time.time() - solve_started
    combined_artifact = dict(combined_result.artifact)

    key: dict[int, int] = {
        int(k): int(v) for k, v in (combined_artifact.get("key") or {}).items()
    }
    mask = _selected_null_mask(combined_artifact)
    solver = combined_result.solver
    status = combined_result.status
    error_message = combined_result.error_message

    word_repair_step: dict[str, Any] | None = None
    if run_word_repair and key and status == "completed":
        # Firewall by construction (F1): the word-repair refinement must never
        # see benchmark plaintext, so strip it from the pages passed in. The
        # runtime path never reads it anyway, but blank fields make a leak
        # structurally impossible (matching the single-page wrapper, whose
        # PageBundle always carries plaintext=""). The original pages -- with
        # plaintext intact -- are kept solely for post-hoc attach_page_scores.
        stripped_pages = [replace(page, plaintext="") for page in pages]
        word_repair_step, adopted = _runner._word_repair_refinement_on_pages(
            pages=stripped_pages,
            alphabet=combined_cipher.alphabet,
            language=lang,
            refinement=homophonic_refinement,
            base_solver=solver,
            base_key=key,
            mask=mask,
        )
        if adopted is not None:
            solver = adopted[0]
            key = {int(k): int(v) for k, v in adopted[1].items()}
    elif run_word_repair:
        # Record why word repair was skipped (empty key or errored solve) so the
        # group artifact still documents the refinement decision.
        word_repair_step = {
            "name": "search_word_repair",
            "status": "skipped",
            "experimental": True,
            "mode": homophonic_refinement,
            "reason": "combined_solve_incomplete" if status != "completed" else "empty_base_key",
            "adopt_enabled": _runner._word_repair_adopt_enabled(),
            "would_adopt": None,
            "adopted": None,
            "candidate_menu": [],
        }

    page_rows = project_pages(pages=pages, key=key, mask=mask)
    # POST-HOC ONLY: grade each projected page against benchmark plaintext.
    attach_page_scores(page_rows)

    # Recover per-page token offsets in the combined stream (provenance).
    offsets: list[int] = []
    running = 0
    for page in pages:
        offsets.append(running)
        running += len(page.token_ids)

    finished = time.time()
    adopt_enabled = _runner._word_repair_adopt_enabled()

    page_results: list[MultipagePageResult] = []
    page_artifacts: list[dict[str, Any]] = []
    for index, (page, row) in enumerate(zip(pages, page_rows)):
        page_artifact = _build_page_artifact(
            row=row,
            page=page,
            group_name=group_name,
            group_run_id=run_id,
            page_index=index,
            page_count=len(pages),
            token_offset=offsets[index],
            combined_sha=combined_sha,
            key=key,
            mask=mask,
            language=lang,
            solver=solver,
            status=status,
            test_ids=test_ids,
        )
        page_artifacts.append(page_artifact)
        page_results.append(
            MultipagePageResult(
                test_id=page.test_id,
                status=status,
                token_count=int(row["token_count"]),
                filtered_length=int(row["filtered_length"]),
                char_accuracy=float(row.get("char_accuracy") or 0.0),
                word_accuracy=float(row.get("word_accuracy") or 0.0),
                decryption=str(row.get("decryption") or ""),
            )
        )

    aggregate_char = _mean([pr.char_accuracy for pr in page_results])
    aggregate_word = _mean([pr.word_accuracy for pr in page_results])

    group_artifact = _build_group_artifact(
        group_name=group_name,
        run_id=run_id,
        status=status,
        solver=solver,
        language=lang,
        test_ids=test_ids,
        homophonic_budget=homophonic_budget,
        homophonic_refinement=homophonic_refinement,
        solve_refinement=solve_refinement,
        homophonic_solver=homophonic_solver,
        combined_cipher=combined_cipher,
        combined_sha=combined_sha,
        combined_artifact=combined_artifact,
        combined_solve_seconds=combined_solve_seconds,
        final_key=key,
        mask=mask,
        word_repair_step=word_repair_step,
        adopt_enabled=adopt_enabled,
        page_rows=page_rows,
        offsets=offsets,
        aggregate_char=aggregate_char,
        aggregate_word=aggregate_word,
        started=started,
        finished=finished,
        error_message=error_message,
    )

    group_artifact_path = ""
    if write_artifacts:
        group_artifact_path = _write_artifacts(
            artifact_dir=Path(artifact_dir),
            group_name=group_name,
            run_id=run_id,
            group_artifact=group_artifact,
            page_artifacts=page_artifacts,
            page_results=page_results,
        )
        group_artifact["multipage_group_artifact"] = group_artifact_path

    return MultipageRunResult(
        group_name=group_name,
        run_id=run_id,
        status=status,
        solver=solver,
        language=lang,
        test_ids=test_ids,
        homophonic_budget=homophonic_budget,
        homophonic_refinement=homophonic_refinement,
        homophonic_solver=homophonic_solver,
        combined_token_count=len(combined_cipher.tokens),
        combined_alphabet_size=combined_cipher.alphabet.size,
        combined_cipher_sha256=combined_sha,
        selected_mask=list(mask),
        combined_solve_seconds=round(combined_solve_seconds, 3),
        elapsed_seconds=round(finished - started, 3),
        aggregate_char_accuracy=aggregate_char,
        aggregate_word_accuracy=aggregate_word,
        word_repair_adopt_enabled=adopt_enabled,
        page_results=page_results,
        word_repair=word_repair_step,
        group_artifact_path=group_artifact_path,
        error_message=error_message,
    )


def _multipage_group_metadata(
    *,
    group_name: str,
    group_run_id: str,
    page_index: int,
    page_count: int,
    token_offset: int,
    token_count: int,
    combined_sha: str,
    mask: tuple[str, ...],
    test_ids: list[str],
) -> dict[str, Any]:
    return {
        "name": group_name,
        "group_run_id": group_run_id,
        "combined_cipher_sha256": combined_sha,
        "page_index": page_index,
        "page_count": page_count,
        "token_offset": token_offset,
        "token_count": token_count,
        "selected_mask": list(mask),
        "test_ids": list(test_ids),
    }


def _build_page_artifact(
    *,
    row: dict[str, Any],
    page: Any,
    group_name: str,
    group_run_id: str,
    page_index: int,
    page_count: int,
    token_offset: int,
    combined_sha: str,
    key: dict[int, int],
    mask: tuple[str, ...],
    language: str,
    solver: str,
    status: str,
    test_ids: list[str],
) -> dict[str, Any]:
    """Per-page artifact: the single-page automated schema plus group metadata.

    The projected shared key/decryption are solver-facing; ``ground_truth`` and
    ``char_accuracy``/``word_accuracy`` are the post-hoc grading channel only.
    """
    decryption = str(row.get("decryption") or "")
    projection_step = {
        "name": "project_from_multipage_group",
        "group": group_name,
        "group_run_id": group_run_id,
        "page_index": page_index,
        "token_offset": token_offset,
        "token_count": int(row.get("token_count") or 0),
        "filtered_length": int(row.get("filtered_length") or 0),
        "selected_mask": list(mask),
        "policy": (
            "This page's plaintext is a projection of the shared group key onto "
            "the page's ciphertext (masked symbols dropped). The solve ran once "
            "on the combined group ciphertext; see the group artifact for solve "
            "and refinement steps."
        ),
    }
    return {
        "run_id": group_run_id,
        "run_mode": "automated_multipage",
        "automated_only": True,
        "cipher_id": page.test_id,
        "test_id": page.test_id,
        "language": language,
        "status": status,
        "solver": solver,
        "cipher_token_count": int(row.get("token_count") or 0),
        "filtered_token_count": int(row.get("filtered_length") or 0),
        "decryption": decryption,
        "key": {str(k): v for k, v in key.items()},
        "steps": [projection_step],
        "multipage_group": _multipage_group_metadata(
            group_name=group_name,
            group_run_id=group_run_id,
            page_index=page_index,
            page_count=page_count,
            token_offset=token_offset,
            token_count=int(row.get("token_count") or 0),
            combined_sha=combined_sha,
            mask=mask,
            test_ids=test_ids,
        ),
        # Post-hoc grading channel (ground truth retained here only).
        "ground_truth": page.plaintext,
        "char_accuracy": float(row.get("char_accuracy") or 0.0),
        "word_accuracy": float(row.get("word_accuracy") or 0.0),
        "estimated_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }


def _compact_page_projection(
    *, row: dict[str, Any], page_index: int, token_offset: int
) -> dict[str, Any]:
    """One page's projection + post-hoc score for the group artifact."""
    return {
        "test_id": str(row.get("test_id") or ""),
        "page_index": page_index,
        "token_offset": token_offset,
        "token_count": int(row.get("token_count") or 0),
        "filtered_length": int(row.get("filtered_length") or 0),
        "char_accuracy": float(row.get("char_accuracy") or 0.0),
        "word_accuracy": float(row.get("word_accuracy") or 0.0),
        "preview": str(row.get("decryption") or "")[:180],
    }


def _build_group_artifact(
    *,
    group_name: str,
    run_id: str,
    status: str,
    solver: str,
    language: str,
    test_ids: list[str],
    homophonic_budget: str,
    homophonic_refinement: str,
    solve_refinement: str,
    homophonic_solver: str,
    combined_cipher: Any,
    combined_sha: str,
    combined_artifact: dict[str, Any],
    combined_solve_seconds: float,
    final_key: dict[int, int],
    mask: tuple[str, ...],
    word_repair_step: dict[str, Any] | None,
    adopt_enabled: bool,
    page_rows: list[dict[str, Any]],
    offsets: list[int],
    aggregate_char: float,
    aggregate_word: float,
    started: float,
    finished: float,
    error_message: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_mode": "automated_multipage",
        "automated_only": True,
        "group_name": group_name,
        "test_ids": test_ids,
        "language": language,
        "status": status,
        "solver": solver,
        # The FINAL shared key (post word-repair adoption when it acted); the
        # combined solve's pre-refinement key stays under combined.key.
        "key": {str(k): v for k, v in final_key.items()},
        "homophonic_budget": homophonic_budget,
        "homophonic_refinement": homophonic_refinement,
        "solve_refinement": solve_refinement,
        "homophonic_solver": homophonic_solver,
        "word_repair_adopt_enabled": adopt_enabled,
        "started_at": started,
        "finished_at": finished,
        "elapsed_seconds": round(finished - started, 3),
        "combined": {
            "cipher_id": combined_artifact.get("cipher_id"),
            "run_id": combined_artifact.get("run_id"),
            "cipher_sha256": combined_sha,
            "token_count": len(combined_cipher.tokens),
            "alphabet_size": combined_cipher.alphabet.size,
            "word_count": len(combined_cipher.words),
            "selected_mask": list(mask),
            "solve_seconds": round(combined_solve_seconds, 3),
            "key": combined_artifact.get("key"),
        },
        "combined_solve_steps": combined_artifact.get("steps") or [],
        "word_repair": word_repair_step,
        "pages": [
            _compact_page_projection(
                row=row, page_index=index, token_offset=offsets[index]
            )
            for index, row in enumerate(page_rows)
        ],
        "aggregate": {
            "page_count": len(page_rows),
            "char_accuracy": aggregate_char,
            "word_accuracy": aggregate_word,
        },
        "environment": _environment_snapshot(),
        "error": error_message,
        "estimated_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }


def _environment_snapshot() -> dict[str, str]:
    prefixes = (
        "DECIPHER_NULL_MASK_",
        "DECIPHER_HOMOPHONIC_",
        "DECIPHER_WORD_REPAIR_",
        "DECIPHER_PARALLEL_",
        "DECIPHER_NGRAM_MODEL_",
    )
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if any(key.startswith(prefix) for prefix in prefixes)
    }


def _write_artifacts(
    *,
    artifact_dir: Path,
    group_name: str,
    run_id: str,
    group_artifact: dict[str, Any],
    page_artifacts: list[dict[str, Any]],
    page_results: list[MultipagePageResult],
) -> str:
    """Write the group artifact and per-page artifacts; return the group path.

    Layout::

        <artifact_dir>/automated_multipage/<group>/<run_id>.group.json
        <artifact_dir>/automated_multipage/<group>/<run_id>/<page_test_id>.json
    """
    base = artifact_dir / "automated_multipage" / group_name
    base.mkdir(parents=True, exist_ok=True)
    pages_dir = base / run_id
    pages_dir.mkdir(parents=True, exist_ok=True)

    for page_artifact, page_result in zip(page_artifacts, page_results):
        page_path = pages_dir / f"{page_result.test_id}.json"
        page_path.write_text(
            json.dumps(page_artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        page_result.artifact_path = str(page_path)

    group_path = base / f"{run_id}.group.json"
    group_path.write_text(
        json.dumps(group_artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return str(group_path)
