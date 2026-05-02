#!/usr/bin/env python3
"""Capture keyed-Vigenere/tableau candidates for Kryptos-style diagnostics.

This is intentionally separate from the transform-triage capture path.  The
question here is narrower: does the current keyed-tableau generator ever reach
the true-ish basin, or is the K2 problem still mostly proposal/search rather
than ranking?

Default mode is blind with respect to the tableau keyword: standard A-Z is used
as the start alphabet and the known Kryptos tableau is used only for offline
labels.  Use --include-known-keyword-control for a solution-bearing sanity
control; do not treat that as a blind solve.

Example
-------
    PYTHONPATH=src:eval python eval/scripts/capture_keyed_vigenere_candidates.py \\
        --benchmark-root ../cipher_benchmark/benchmark \\
        --test-id kryptos_k2_keyed_vigenere \\
        --seed-count 20 --steps 1000 --restarts 4 --max-period 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from analysis.polyalphabetic import (
    generate_keyed_vigenere_constraint_alphabets,
    generate_keyed_vigenere_constraint_graph_alphabets,
    generate_keyed_vigenere_offset_graph_alphabets,
    normalize_keyed_alphabet,
    search_keyed_vigenere_alphabet_anneal,
)
from benchmark.loader import BenchmarkLoader, parse_canonical_transcription
from benchmark.scorer import score_decryption


def _csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _csv_int_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def _load_test_data(benchmark_root: Path, split: str, test_id: str):
    loader = BenchmarkLoader(benchmark_root)
    tests = loader.load_tests(split)
    match = next((test for test in tests if test.test_id == test_id), None)
    if match is None:
        raise SystemExit(f"test_id not found in {split}: {test_id}")
    return loader.load_test_data(match)


def _tableau_distance(candidate: str | None, truth: str | None) -> dict[str, Any]:
    if not candidate or not truth:
        return {
            "tableau_exact": None,
            "tableau_position_matches": None,
            "tableau_hamming": None,
            "tableau_mean_abs_position_delta": None,
            "tableau_prefix_match": None,
        }
    truth_pos = {ch: i for i, ch in enumerate(truth)}
    cand_pos = {ch: i for i, ch in enumerate(candidate)}
    shared = sorted(set(truth_pos) & set(cand_pos))
    deltas = [abs(cand_pos[ch] - truth_pos[ch]) for ch in shared]
    prefix = 0
    for a, b in zip(candidate, truth, strict=False):
        if a != b:
            break
        prefix += 1
    return {
        "tableau_exact": candidate == truth,
        "tableau_position_matches": sum(1 for a, b in zip(candidate, truth, strict=False) if a == b),
        "tableau_hamming": sum(1 for a, b in zip(candidate, truth, strict=False) if a != b),
        "tableau_mean_abs_position_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "tableau_prefix_match": prefix,
    }


def _candidate_labels(candidate: dict[str, Any], truth_plaintext: str, truth_key: str | None, truth_tableau: str | None) -> dict[str, Any]:
    plaintext = str(candidate.get("plaintext") or "")
    score = score_decryption(
        "keyed_vigenere_candidate",
        plaintext,
        truth_plaintext,
        agent_score=0.0,
        status="completed",
    )
    metadata = candidate.get("metadata") or {}
    keyed_alphabet = metadata.get("keyed_alphabet")
    labels = {
        "char_accuracy": round(score.char_accuracy, 6),
        "word_accuracy": round(score.word_accuracy, 6),
        "key_exact": (candidate.get("key") == truth_key) if truth_key else None,
    }
    labels.update(_tableau_distance(keyed_alphabet, truth_tableau))
    return labels


def _candidate_identity(candidate: dict[str, Any]) -> tuple[Any, ...]:
    metadata = candidate.get("metadata") or {}
    return (
        metadata.get("keyed_alphabet"),
        candidate.get("period"),
        candidate.get("key"),
        tuple(candidate.get("shifts") or []),
    )


def _swap_perturb_alphabet(alphabet: str, *, swaps: int, rng: random.Random) -> str:
    letters = list(normalize_keyed_alphabet(keyed_alphabet=alphabet))
    swaps = max(0, min(int(swaps), len(letters) // 2))
    if swaps == 0:
        return "".join(letters)
    positions = list(range(len(letters)))
    rng.shuffle(positions)
    for i in range(swaps):
        a = positions[2 * i]
        b = positions[2 * i + 1]
        letters[a], letters[b] = letters[b], letters[a]
    return "".join(letters)


def _tableau_perturbation_alphabets(
    truth_tableau: str,
    *,
    swap_counts: list[int],
    samples_per_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return exact and randomly perturbed solution-tableau controls."""
    truth = normalize_keyed_alphabet(keyed_alphabet=truth_tableau)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for swaps in swap_counts:
        trials = 1 if swaps == 0 else max(1, samples_per_count)
        for sample_idx in range(trials):
            alphabet = _swap_perturb_alphabet(truth, swaps=swaps, rng=rng)
            if alphabet in seen:
                continue
            seen.add(alphabet)
            rows.append({
                "keyed_alphabet": alphabet,
                "control_type": "exact_tableau" if swaps == 0 else "tableau_swap_perturbation",
                "swap_count": swaps,
                "sample_index": sample_idx,
            })
    return rows


def _rank_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"candidate_count": 0}

    best_by_score = candidates[0]
    best_by_char = max(candidates, key=lambda item: item["labels"].get("char_accuracy") or 0.0)
    exact_tableau = [c for c in candidates if c["labels"].get("tableau_exact")]
    exact_key = [c for c in candidates if c["labels"].get("key_exact")]

    def _rank_of(candidate_id: str) -> int | None:
        for idx, candidate in enumerate(candidates, start=1):
            if candidate["candidate_id"] == candidate_id:
                return idx
        return None

    return {
        "candidate_count": len(candidates),
        "best_score": {
            "rank": 1,
            "candidate_id": best_by_score["candidate_id"],
            "score": best_by_score["score"],
            "key": best_by_score["key"],
            "period": best_by_score["period"],
            "char_accuracy": best_by_score["labels"].get("char_accuracy"),
            "tableau_hamming": best_by_score["labels"].get("tableau_hamming"),
            "preview": best_by_score["preview"],
        },
        "best_char_accuracy": {
            "rank": _rank_of(best_by_char["candidate_id"]),
            "candidate_id": best_by_char["candidate_id"],
            "score": best_by_char["score"],
            "key": best_by_char["key"],
            "period": best_by_char["period"],
            "char_accuracy": best_by_char["labels"].get("char_accuracy"),
            "tableau_hamming": best_by_char["labels"].get("tableau_hamming"),
            "preview": best_by_char["preview"],
        },
        "exact_tableau_count": len(exact_tableau),
        "first_exact_tableau_rank": _rank_of(exact_tableau[0]["candidate_id"]) if exact_tableau else None,
        "exact_key_count": len(exact_key),
        "first_exact_key_rank": _rank_of(exact_key[0]["candidate_id"]) if exact_key else None,
        "max_char_accuracy": best_by_char["labels"].get("char_accuracy"),
        "min_tableau_hamming": min(
            (c["labels"].get("tableau_hamming") for c in candidates if c["labels"].get("tableau_hamming") is not None),
            default=None,
        ),
    }


def _finalize_payload(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    split: str,
    test_id: str,
    leakage_mode: str,
    seeds: list[int],
    alphabet_keywords: list[str],
    initial_alphabets: list[str],
    truth_key: str | None,
    truth_tableau: str | None,
    truth_keyword: str | None,
    raw_candidates: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in raw_candidates:
        identity = _candidate_identity({
            "period": candidate["period"],
            "key": candidate["key"],
            "shifts": candidate["shifts"],
            "metadata": {
                "keyed_alphabet": candidate["parameters"].get("keyed_alphabet"),
            },
        })
        previous = deduped.get(identity)
        if previous is None or (candidate.get("score") or float("-inf")) > (previous.get("score") or float("-inf")):
            deduped[identity] = candidate

    candidates = sorted(deduped.values(), key=lambda item: item.get("score") or float("-inf"), reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["global_rank"] = rank

    elapsed = time.time() - started
    return {
        "schema": "keyed_vigenere_candidate_capture_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_id": test_id,
        "benchmark_root": str(benchmark_root),
        "split": split,
        "leakage_mode": leakage_mode,
        "config": {
            "seeds": seeds,
            "completed_seeds": sorted({int(c["seed"]) for c in raw_candidates}),
            "steps": args.steps,
            "restarts": args.restarts,
            "max_period": args.max_period,
            "top_per_seed": args.top_per_seed,
            "guided": args.guided,
            "guided_pool_size": args.guided_pool_size,
            "engine": getattr(args, "engine", "python"),
            "constraint_starts": args.constraint_starts,
            "constraint_beam_size": args.constraint_beam_size,
            "constraint_random_starts": args.constraint_random_starts,
            "constraint_random_seed": args.constraint_random_seed,
            "constraint_top_shifts": args.constraint_top_shifts,
            "constraint_top_letters": args.constraint_top_letters,
            "constraint_target_window": args.constraint_target_window,
            "constraint_random_phases": args.constraint_random_phases,
            "offset_graph_starts": args.offset_graph_starts,
            "offset_graph_seed": args.offset_graph_seed,
            "offset_graph_samples": args.offset_graph_samples,
            "offset_graph_phase_count": args.offset_graph_phase_count,
            "offset_graph_top_cipher": args.offset_graph_top_cipher,
            "offset_graph_target_letters": args.offset_graph_target_letters,
            "offset_graph_target_window": args.offset_graph_target_window,
            "constraint_graph_starts": args.constraint_graph_starts,
            "constraint_graph_seed": args.constraint_graph_seed,
            "constraint_graph_beam_size": args.constraint_graph_beam_size,
            "constraint_graph_phase_count": args.constraint_graph_phase_count,
            "constraint_graph_top_cipher": args.constraint_graph_top_cipher,
            "constraint_graph_target_letters": args.constraint_graph_target_letters,
            "constraint_graph_target_window": args.constraint_graph_target_window,
            "constraint_graph_options_per_phase": args.constraint_graph_options_per_phase,
            "constraint_graph_materializations": args.constraint_graph_materializations,
            "include_tableau_perturbation_control": args.include_tableau_perturbation_control,
            "tableau_perturbation_swaps": args.tableau_perturbation_swaps,
            "tableau_perturbation_samples": args.tableau_perturbation_samples,
            "tableau_perturbation_seed": args.tableau_perturbation_seed,
            "include_standard_alphabet": not args.no_standard_alphabet,
            "alphabet_keywords": alphabet_keywords,
            "initial_alphabet_count": len(initial_alphabets),
        },
        "truth_labels_available": bool(truth_key and truth_tableau),
        "truth": {
            "periodic_key": truth_key,
            "keyed_alphabet": truth_tableau,
            "alphabet_keyword": truth_keyword,
        } if args.include_truth_in_output else {
            "periodic_key": "<withheld>",
            "keyed_alphabet": "<withheld>",
            "alphabet_keyword": "<withheld>",
            "note": "Solution-bearing truth is used for offline labels but withheld from this artifact.",
        },
        "elapsed_seconds": round(elapsed, 3),
        "raw_candidate_count": len(raw_candidates),
        "deduped_candidate_count": len(candidates),
        "summary": _rank_summary(candidates),
        "candidates": candidates,
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture keyed-Vigenere/tableau candidates for K2-style diagnostics."
    )
    parser.add_argument("--benchmark-root", type=Path, default=Path("../cipher_benchmark/benchmark"))
    parser.add_argument("--split", default="kryptos_tests.jsonl")
    parser.add_argument("--test-id", default="kryptos_k2_keyed_vigenere")
    parser.add_argument("--artifact-dir", type=Path, default=Path("eval/artifacts/keyed_vigenere"))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--max-period", type=int, default=12)
    parser.add_argument("--top-per-seed", type=int, default=20)
    parser.add_argument("--guided", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guided-pool-size", type=int, default=24)
    parser.add_argument(
        "--engine",
        choices=["python", "rust", "auto"],
        default="python",
        help="Search engine to use. 'rust' requires optional decipher_fast extension.",
    )
    parser.add_argument("--alphabet-keywords", default="", help="Comma-separated candidate tableau keywords.")
    parser.add_argument("--initial-alphabets", default="", help="Comma-separated explicit keyed alphabets.")
    parser.add_argument(
        "--constraint-starts",
        type=int,
        default=0,
        metavar="N",
        help="Append N phase-frequency constraint-generated start alphabets.",
    )
    parser.add_argument("--constraint-beam-size", type=int, default=64)
    parser.add_argument(
        "--constraint-random-starts",
        type=int,
        default=0,
        metavar="N",
        help="Append N randomized phase-constraint start alphabets.",
    )
    parser.add_argument("--constraint-random-seed", type=int, default=1)
    parser.add_argument("--constraint-top-shifts", type=int, default=3)
    parser.add_argument("--constraint-top-letters", type=int, default=3)
    parser.add_argument("--constraint-target-window", type=int, default=3)
    parser.add_argument(
        "--constraint-random-phases",
        type=int,
        default=None,
        help="Apply at most N randomly selected phases per randomized start.",
    )
    parser.add_argument("--offset-graph-starts", type=int, default=0)
    parser.add_argument("--offset-graph-seed", type=int, default=1)
    parser.add_argument("--offset-graph-samples", type=int, default=4096)
    parser.add_argument("--offset-graph-phase-count", type=int, default=None)
    parser.add_argument("--offset-graph-top-cipher", type=int, default=5)
    parser.add_argument("--offset-graph-target-letters", type=int, default=8)
    parser.add_argument("--offset-graph-target-window", type=int, default=4)
    parser.add_argument("--constraint-graph-starts", type=int, default=0)
    parser.add_argument("--constraint-graph-seed", type=int, default=1)
    parser.add_argument("--constraint-graph-beam-size", type=int, default=128)
    parser.add_argument("--constraint-graph-phase-count", type=int, default=None)
    parser.add_argument("--constraint-graph-top-cipher", type=int, default=5)
    parser.add_argument("--constraint-graph-target-letters", type=int, default=8)
    parser.add_argument("--constraint-graph-target-window", type=int, default=4)
    parser.add_argument("--constraint-graph-options-per-phase", type=int, default=96)
    parser.add_argument("--constraint-graph-materializations", type=int, default=4)
    parser.add_argument(
        "--include-tableau-perturbation-control",
        action="store_true",
        help=(
            "Append the true tableau plus random swap perturbations as a "
            "solution-bearing scorer calibration control."
        ),
    )
    parser.add_argument(
        "--tableau-perturbation-swaps",
        default="0,1,2,4,8,12",
        help="Comma-separated swap counts to sample around the true tableau.",
    )
    parser.add_argument(
        "--tableau-perturbation-samples",
        type=int,
        default=8,
        help="Random perturbation samples per nonzero swap count.",
    )
    parser.add_argument("--tableau-perturbation-seed", type=int, default=1)
    parser.add_argument("--no-standard-alphabet", action="store_true")
    parser.add_argument(
        "--include-known-keyword-control",
        action="store_true",
        help="Append the solution-bearing benchmark tableau keyword as a sanity control.",
    )
    parser.add_argument(
        "--include-truth-in-output",
        action="store_true",
        help="Write the known key/tableau into the artifact. Labels are computed either way.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, keep completed seeds and run only missing seeds.",
    )
    parser.add_argument(
        "--checkpoint-every-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output after each seed (default: true).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    data = _load_test_data(args.benchmark_root, args.split, args.test_id)
    cipher_text = parse_canonical_transcription(data.canonical_transcription)
    truth_hints = data.solver_hints or {}
    truth_key = truth_hints.get("periodic_key")
    truth_tableau = truth_hints.get("keyed_alphabet")
    truth_keyword = truth_hints.get("alphabet_keyword")

    alphabet_keywords = _csv_values(args.alphabet_keywords)
    if args.include_known_keyword_control and truth_keyword and truth_keyword not in alphabet_keywords:
        alphabet_keywords.append(truth_keyword)
    initial_alphabets = _csv_values(args.initial_alphabets)
    rust_initial_alphabets = list(initial_alphabets)
    initial_alphabet_sources: dict[str, dict[str, Any]] = {}
    for alphabet in initial_alphabets:
        normalized = normalize_keyed_alphabet(keyed_alphabet=alphabet)
        initial_alphabet_sources.setdefault(normalized, {"source": "explicit_initial_alphabet"})
    for keyword in alphabet_keywords:
        normalized = normalize_keyed_alphabet(alphabet_keyword=keyword)
        if normalized not in rust_initial_alphabets:
            rust_initial_alphabets.append(normalized)
        initial_alphabet_sources.setdefault(
            normalized,
            {"source": "keyword_alphabet", "alphabet_keyword": keyword.upper()},
        )
    if args.include_tableau_perturbation_control:
        if not truth_tableau:
            raise SystemExit("--include-tableau-perturbation-control requires benchmark solver_hints.keyed_alphabet")
        perturb_rows = _tableau_perturbation_alphabets(
            truth_tableau,
            swap_counts=_csv_int_values(args.tableau_perturbation_swaps),
            samples_per_count=args.tableau_perturbation_samples,
            seed=args.tableau_perturbation_seed,
        )
        added = 0
        for row in perturb_rows:
            alphabet = row["keyed_alphabet"]
            if alphabet not in rust_initial_alphabets:
                rust_initial_alphabets.append(alphabet)
                initial_alphabets.append(alphabet)
                added += 1
            initial_alphabet_sources.setdefault(
                alphabet,
                {
                    "source": "tableau_perturbation_control",
                    "control_type": row["control_type"],
                    "swap_count": row["swap_count"],
                    "sample_index": row["sample_index"],
                },
            )
        if args.verbose:
            print(
                "Tableau perturbation controls: "
                f"swap_counts={args.tableau_perturbation_swaps} "
                f"samples={args.tableau_perturbation_samples} added={added}"
            )
    if args.constraint_starts > 0 or args.constraint_random_starts > 0:
        constraint_result = generate_keyed_vigenere_constraint_alphabets(
            cipher_text,
            max_period=args.max_period,
            beam_size=args.constraint_beam_size,
            limit=args.constraint_starts + args.constraint_random_starts,
            random_limit=args.constraint_random_starts,
            random_seed=args.constraint_random_seed,
            top_shifts=args.constraint_top_shifts,
            top_letters=args.constraint_top_letters,
            target_window=args.constraint_target_window,
            random_phases=args.constraint_random_phases,
        )
        added = 0
        for row in constraint_result.get("alphabets") or []:
            alphabet = row.get("keyed_alphabet")
            if isinstance(alphabet, str) and alphabet not in rust_initial_alphabets:
                rust_initial_alphabets.append(alphabet)
                initial_alphabets.append(alphabet)
                added += 1
            if isinstance(alphabet, str):
                initial_alphabet_sources.setdefault(
                    alphabet,
                    {"source": "phase_constraint_generator"},
                )
        if args.verbose:
            print(
                f"Constraint starts: deterministic={args.constraint_starts} "
                f"random={args.constraint_random_starts} added={added}"
            )
    if args.offset_graph_starts > 0:
        offset_graph_result = generate_keyed_vigenere_offset_graph_alphabets(
            cipher_text,
            max_period=args.max_period,
            limit=args.offset_graph_starts,
            random_seed=args.offset_graph_seed,
            samples=args.offset_graph_samples,
            phase_count=args.offset_graph_phase_count,
            top_cipher_letters=args.offset_graph_top_cipher,
            target_letters=args.offset_graph_target_letters,
            target_window=args.offset_graph_target_window,
        )
        added = 0
        for row in offset_graph_result.get("alphabets") or []:
            alphabet = row.get("keyed_alphabet")
            if isinstance(alphabet, str) and alphabet not in rust_initial_alphabets:
                rust_initial_alphabets.append(alphabet)
                initial_alphabets.append(alphabet)
                added += 1
            if isinstance(alphabet, str):
                initial_alphabet_sources.setdefault(
                    alphabet,
                    {"source": "offset_graph_generator"},
                )
        if args.verbose:
            print(
                f"Offset graph starts: requested={args.offset_graph_starts} "
                f"samples={args.offset_graph_samples} added={added}"
            )
    if args.constraint_graph_starts > 0:
        constraint_graph_result = generate_keyed_vigenere_constraint_graph_alphabets(
            cipher_text,
            max_period=args.max_period,
            limit=args.constraint_graph_starts,
            random_seed=args.constraint_graph_seed,
            beam_size=args.constraint_graph_beam_size,
            phase_count=args.constraint_graph_phase_count,
            top_cipher_letters=args.constraint_graph_top_cipher,
            target_letters=args.constraint_graph_target_letters,
            target_window=args.constraint_graph_target_window,
            options_per_phase=args.constraint_graph_options_per_phase,
            materializations_per_state=args.constraint_graph_materializations,
        )
        added = 0
        for row in constraint_graph_result.get("alphabets") or []:
            alphabet = row.get("keyed_alphabet")
            if isinstance(alphabet, str) and alphabet not in rust_initial_alphabets:
                rust_initial_alphabets.append(alphabet)
                initial_alphabets.append(alphabet)
                added += 1
            if isinstance(alphabet, str):
                initial_alphabet_sources.setdefault(
                    alphabet,
                    {"source": "constraint_graph_generator"},
                )
        if args.verbose:
            print(
                f"Constraint graph starts: requested={args.constraint_graph_starts} "
                f"beam={args.constraint_graph_beam_size} added={added}"
            )

    search_fn = search_keyed_vigenere_alphabet_anneal
    use_rust = False
    if args.engine in {"rust", "auto"}:
        try:
            from analysis.polyalphabetic_fast import (
                FAST_AVAILABLE,
                search_keyed_vigenere_alphabet_anneal_fast,
            )
        except Exception:  # noqa: BLE001
            FAST_AVAILABLE = False
        if FAST_AVAILABLE:
            search_fn = search_keyed_vigenere_alphabet_anneal_fast
            use_rust = True
            args.engine = "rust"
        elif args.engine == "rust":
            raise SystemExit(
                "decipher_fast is not installed. Build it with: "
                "cd rust/decipher_fast && ../../.venv/bin/python -m maturin develop --release"
            )
        else:
            args.engine = "python"

    seeds = args.seeds if args.seeds else list(range(args.seed_start, args.seed_start + args.seed_count))
    if args.include_tableau_perturbation_control:
        leakage_mode = "solution_tableau_perturbation_control"
    elif args.include_known_keyword_control:
        leakage_mode = "solution_keyword_control"
    else:
        leakage_mode = "blind"
    started = time.time()
    raw_candidates: list[dict[str, Any]] = []

    out_path = args.output
    if out_path is None:
        if args.include_tableau_perturbation_control:
            mode = "tableau_control"
        elif args.include_known_keyword_control:
            mode = "control"
        else:
            mode = "blind"
        constraint_total = args.constraint_starts + args.constraint_random_starts
        start_total = constraint_total + args.offset_graph_starts + args.constraint_graph_starts
        constraint = f"_cs{start_total}" if start_total else ""
        out_path = args.artifact_dir / (
            f"{args.test_id}_{mode}_{args.engine}{constraint}_"
            f"s{seeds[0]}_{seeds[-1]}_steps{args.steps}.json"
        )

    completed_seeds: set[int] = set()
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        raw_candidates = list(existing.get("candidates") or [])
        completed_seeds = {int(c["seed"]) for c in raw_candidates if c.get("seed") is not None}
        if args.verbose:
            print(f"Resuming {out_path}: {len(completed_seeds)} completed seeds, {len(raw_candidates)} candidates")

    for idx, seed in enumerate(seeds, start=1):
        if seed in completed_seeds:
            if args.verbose:
                print(f"[{idx}/{len(seeds)}] seed={seed} cached")
            continue
        if args.verbose:
            print(f"[{idx}/{len(seeds)}] seed={seed} engine={args.engine}")
        if use_rust:
            result = search_fn(
                cipher_text,
                language=data.plaintext_language or "en",
                max_period=args.max_period,
                initial_alphabets=rust_initial_alphabets,
                include_standard_alphabet=not args.no_standard_alphabet,
                steps=args.steps,
                restarts=args.restarts,
                seed=seed,
                guided=args.guided,
                guided_pool_size=args.guided_pool_size,
                constraint_starts=0,
                constraint_random_starts=0,
                offset_graph_starts=0,
                constraint_graph_starts=0,
                top_n=args.top_per_seed,
            )
        else:
            result = search_fn(
                cipher_text,
                language=data.plaintext_language or "en",
                max_period=args.max_period,
                initial_alphabets=initial_alphabets,
                alphabet_keywords=alphabet_keywords,
                include_standard_alphabet=not args.no_standard_alphabet,
                steps=args.steps,
                restarts=args.restarts,
                seed=seed,
                guided=args.guided,
                guided_pool_size=args.guided_pool_size,
                top_n=args.top_per_seed,
            )
        for seed_rank, candidate in enumerate(result.get("top_candidates") or [], start=1):
            metadata = candidate.get("metadata") or {}
            labels = _candidate_labels(candidate, data.plaintext, truth_key, truth_tableau)
            initial_alphabet = metadata.get("initial_keyed_alphabet")
            initial_source = initial_alphabet_sources.get(initial_alphabet, {})
            raw_candidates.append({
                "candidate_id": f"{args.test_id}::seed{seed}::rank{seed_rank}",
                "candidate_family": "keyed_vigenere",
                "generator": "keyed_vigenere_alphabet_anneal",
                "seed": seed,
                "seed_rank": seed_rank,
                "period": candidate.get("period"),
                "key": candidate.get("key"),
                "shifts": candidate.get("shifts"),
                "score": candidate.get("score"),
                "selection_score": candidate.get("selection_score"),
                "init_score": candidate.get("init_score"),
                "parameters": {
                    "keyed_alphabet": metadata.get("keyed_alphabet"),
                    "alphabet_keyword": metadata.get("alphabet_keyword"),
                    "initial_keyed_alphabet": metadata.get("initial_keyed_alphabet"),
                    "initial_candidate_type": metadata.get("initial_candidate_type"),
                    "initial_alphabet_source": initial_source.get("source"),
                    "initial_control_type": initial_source.get("control_type"),
                    "initial_control_swap_count": initial_source.get("swap_count"),
                    "initial_control_sample_index": initial_source.get("sample_index"),
                    "restart": metadata.get("restart"),
                    "mutation_search": metadata.get("mutation_search"),
                    "guided_proposals": metadata.get("guided_proposals"),
                    "random_proposals": metadata.get("random_proposals"),
                },
                "labels": labels,
                "plaintext": candidate.get("plaintext"),
                "preview": candidate.get("preview"),
            })

        if args.checkpoint_every_seed:
            payload = _finalize_payload(
                args=args,
                benchmark_root=args.benchmark_root,
                split=args.split,
                test_id=args.test_id,
                leakage_mode=leakage_mode,
                seeds=seeds,
                alphabet_keywords=alphabet_keywords,
                initial_alphabets=initial_alphabets,
                truth_key=truth_key,
                truth_tableau=truth_tableau,
                truth_keyword=truth_keyword,
                raw_candidates=raw_candidates,
                started=started,
            )
            _write_payload(out_path, payload)
            if args.verbose:
                summary = payload["summary"]
                completed = len(payload["config"]["completed_seeds"])
                print(
                    f"  checkpoint: seeds={completed}/{len(seeds)} "
                    f"deduped={payload['deduped_candidate_count']} "
                    f"best_char={summary.get('max_char_accuracy')}"
                )

    payload = _finalize_payload(
        args=args,
        benchmark_root=args.benchmark_root,
        split=args.split,
        test_id=args.test_id,
        leakage_mode=leakage_mode,
        seeds=seeds,
        alphabet_keywords=alphabet_keywords,
        initial_alphabets=initial_alphabets,
        truth_key=truth_key,
        truth_tableau=truth_tableau,
        truth_keyword=truth_keyword,
        raw_candidates=raw_candidates,
        started=started,
    )
    _write_payload(out_path, payload)
    candidates = payload["candidates"]

    print(f"Wrote {out_path}")
    print(f"Candidates: raw={len(raw_candidates)} deduped={len(candidates)} elapsed={payload['elapsed_seconds']:.1f}s")
    summary = payload["summary"]
    if candidates:
        print(
            "Best score: "
            f"char={summary['best_score']['char_accuracy']:.3f} "
            f"key={summary['best_score']['key']} "
            f"tableau_hamming={summary['best_score']['tableau_hamming']}"
        )
        print(
            "Best char:  "
            f"rank={summary['best_char_accuracy']['rank']} "
            f"char={summary['best_char_accuracy']['char_accuracy']:.3f} "
            f"key={summary['best_char_accuracy']['key']} "
            f"tableau_hamming={summary['best_char_accuracy']['tableau_hamming']}"
        )
        print(
            "Exact labels: "
            f"tableau_count={summary['exact_tableau_count']} "
            f"first_tableau_rank={summary['first_exact_tableau_rank']} "
            f"key_count={summary['exact_key_count']} "
            f"first_key_rank={summary['first_exact_key_rank']}"
        )


if __name__ == "__main__":
    main()
