#!/usr/bin/env python3
"""Capture Quagmire III keyword-prefix search diagnostics.

This is a solved-case research harness, not a production solver path.  It
answers a narrow question: does Decipher's keyword-shaped Quagmire search ever
enter the known prefix basin under a given budget?

Example:
    PYTHONPATH=src:eval .venv/bin/python eval/scripts/capture_quagmire_candidates.py \
      --benchmark-root ../cipher_benchmark/benchmark \
      --test-id kryptos_k2_keyed_vigenere \
      --keyword-lengths 7 \
      --cycleword-lengths 8 \
      --seed-count 20 \
      --steps 200 \
      --restarts 100 \
      --calibration-keyword KRYPTOS \
      --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.polyalphabetic import search_quagmire3_keyword_alphabet
from analysis.polyalphabetic_fast import FAST_AVAILABLE, search_quagmire3_shotgun_fast
from benchmark.loader import BenchmarkLoader, parse_canonical_transcription
from benchmark.scorer import score_decryption


def _csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _csv_int_values(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _load_test_data(benchmark_root: Path, split: str, test_id: str):
    loader = BenchmarkLoader(benchmark_root)
    tests = loader.load_tests(split)
    match = next((test for test in tests if test.test_id == test_id), None)
    if match is None:
        raise SystemExit(f"test_id not found in {split}: {test_id}")
    return loader.load_test_data(match)


def _candidate_record(
    *,
    test_id: str,
    seed: int,
    rank: int,
    candidate: dict[str, Any],
    plaintext: str,
    calibration_keyword: str | None = None,
) -> dict[str, Any]:
    metadata = candidate.get("metadata") or {}
    calibration_distance = metadata.get("calibration_keyword_distance")
    if calibration_distance is None and calibration_keyword:
        alphabet_keyword = str(metadata.get("alphabet_keyword") or "")
        prefix = alphabet_keyword[: len(calibration_keyword)].upper()
        if prefix:
            calibration_distance = sum(
                0 if a == b else 1 for a, b in zip(prefix, calibration_keyword)
            ) + abs(len(prefix) - len(calibration_keyword))
    score = score_decryption(
        f"{test_id}_quagmire_candidate",
        str(candidate.get("plaintext") or ""),
        plaintext,
        agent_score=0.0,
        status="completed",
    )
    return {
        "candidate_id": f"{test_id}::seed{seed}::rank{rank}",
        "seed": seed,
        "seed_rank": rank,
        "variant": candidate.get("variant"),
        "period": candidate.get("period"),
        "cycleword": metadata.get("cycleword") or candidate.get("key"),
        "alphabet_keyword": metadata.get("alphabet_keyword"),
        "score": candidate.get("score"),
        "selection_score": candidate.get("selection_score"),
        "word_score": metadata.get("word_score"),
        "calibration_keyword_distance": calibration_distance,
        "plaintext_preview": str(candidate.get("plaintext") or "")[:240],
        "char_accuracy": round(score.char_accuracy, 6),
        "word_accuracy": round(score.word_accuracy, 6),
        "metadata": metadata,
    }


def _summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"candidate_count": 0}
    ranked = sorted(
        candidates,
        key=lambda row: row.get("selection_score") if row.get("selection_score") is not None else float("-inf"),
        reverse=True,
    )
    for idx, row in enumerate(ranked, start=1):
        row["global_rank"] = idx
    best = ranked[0]
    best_char = max(ranked, key=lambda row: row.get("char_accuracy") or 0.0)
    distances = [
        row["calibration_keyword_distance"]
        for row in ranked
        if row.get("calibration_keyword_distance") is not None
    ]
    exact = next((row for row in ranked if row.get("calibration_keyword_distance") == 0), None)
    return {
        "candidate_count": len(ranked),
        "best_selection": {
            "rank": best["global_rank"],
            "seed": best["seed"],
            "alphabet_keyword": best["alphabet_keyword"],
            "cycleword": best["cycleword"],
            "selection_score": best["selection_score"],
            "word_score": best["word_score"],
            "calibration_keyword_distance": best.get("calibration_keyword_distance"),
            "char_accuracy": best["char_accuracy"],
            "preview": best["plaintext_preview"],
        },
        "best_char_accuracy": {
            "rank": best_char["global_rank"],
            "seed": best_char["seed"],
            "alphabet_keyword": best_char["alphabet_keyword"],
            "cycleword": best_char["cycleword"],
            "selection_score": best_char["selection_score"],
            "word_score": best_char["word_score"],
            "calibration_keyword_distance": best_char.get("calibration_keyword_distance"),
            "char_accuracy": best_char["char_accuracy"],
            "preview": best_char["plaintext_preview"],
        },
        "best_calibration_keyword_distance": min(distances) if distances else None,
        "exact_calibration_keyword_rank": exact["global_rank"] if exact else None,
    }


def _seed_label_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {
            "retained_candidate_count": 0,
            "best_calibration_keyword_distance": None,
            "exact_calibration_keyword_rank": None,
            "exact_calibration_keyword_count": 0,
            "best_char_accuracy": None,
        }
    ranked = sorted(
        candidates,
        key=lambda row: row.get("selection_score") if row.get("selection_score") is not None else float("-inf"),
        reverse=True,
    )
    distances = [
        row["calibration_keyword_distance"]
        for row in ranked
        if row.get("calibration_keyword_distance") is not None
    ]
    exact_rows = [
        (idx, row) for idx, row in enumerate(ranked, start=1)
        if row.get("calibration_keyword_distance") == 0
    ]
    best_char = max(ranked, key=lambda row: row.get("char_accuracy") or 0.0)
    return {
        "retained_candidate_count": len(ranked),
        "best_calibration_keyword_distance": min(distances) if distances else None,
        "exact_calibration_keyword_rank": exact_rows[0][0] if exact_rows else None,
        "exact_calibration_keyword_count": len(exact_rows),
        "best_char_accuracy": best_char.get("char_accuracy"),
        "best_char_keyword": best_char.get("alphabet_keyword"),
        "best_char_cycleword": best_char.get("cycleword"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Quagmire III keyword-prefix search diagnostics.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("../cipher_benchmark/benchmark"))
    parser.add_argument("--split", default="kryptos_tests.jsonl")
    parser.add_argument("--test-id", default="kryptos_k2_keyed_vigenere")
    parser.add_argument("--artifact-dir", type=Path, default=Path("eval/artifacts/quagmire"))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--keyword-lengths", default="7")
    parser.add_argument("--cycleword-lengths", default="8")
    parser.add_argument("--initial-keywords", default="")
    parser.add_argument("--dictionary-starts", type=int, default=0)
    parser.add_argument("--engine", choices=["python", "rust", "auto"], default="rust")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--hillclimbs", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--screen-top-n", type=int, default=128)
    parser.add_argument("--word-weight", type=float, default=0.25)
    parser.add_argument("--slip-probability", type=float, default=0.001)
    parser.add_argument("--backtrack-probability", type=float, default=0.15)
    parser.add_argument("--top-per-seed", type=int, default=10)
    parser.add_argument("--calibration-keyword", default="")
    parser.add_argument("--include-truth-in-output", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint-every-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    test_data = _load_test_data(args.benchmark_root, args.split, args.test_id)
    cipher_text = parse_canonical_transcription(test_data.canonical_transcription)
    seeds = args.seeds or list(range(args.seed_start, args.seed_start + args.seed_count))
    keyword_lengths = _csv_int_values(args.keyword_lengths)
    cycleword_lengths = _csv_int_values(args.cycleword_lengths)
    initial_keywords = _csv_values(args.initial_keywords)
    calibration_keyword = args.calibration_keyword.strip().upper()
    requested_engine = args.engine
    engine = args.engine
    if engine == "auto":
        engine = "rust"
    if engine == "rust" and not FAST_AVAILABLE:
        raise SystemExit(
            "Rust engine requested but decipher_fast is not installed. Build it with: "
            "scripts/build_rust_fast.sh"
        )
    hillclimbs = args.steps if args.hillclimbs is None else args.hillclimbs

    mode = "calibrated" if calibration_keyword else "blind"
    output = args.output
    if output is None:
        if engine == "rust":
            budget_label = f"rust_hill{hillclimbs}_restarts{args.restarts}"
        else:
            budget_label = f"python_steps{args.steps}_restarts{args.restarts}"
        output = args.artifact_dir / (
            f"{args.test_id}_quag3_{mode}_s{seeds[0]}_{seeds[-1]}_"
            f"{budget_label}.json"
        )

    started = time.time()
    raw_candidates: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []

    for idx, seed in enumerate(seeds, start=1):
        if args.verbose:
            print(f"[{idx}/{len(seeds)}] seed={seed}")
        seed_records: list[dict[str, Any]] = []
        if engine == "rust":
            result = search_quagmire3_shotgun_fast(
                cipher_text,
                language=test_data.plaintext_language or "en",
                keyword_lengths=keyword_lengths,
                cycleword_lengths=cycleword_lengths,
                hillclimbs=hillclimbs,
                restarts=args.restarts,
                seed=seed,
                top_n=args.top_per_seed,
                slip_probability=args.slip_probability,
                backtrack_probability=args.backtrack_probability,
                threads=args.threads,
                initial_keywords=initial_keywords,
            )
        else:
            result = search_quagmire3_keyword_alphabet(
                cipher_text,
                language=test_data.plaintext_language or "en",
                keyword_lengths=keyword_lengths,
                cycleword_lengths=cycleword_lengths,
                initial_keywords=initial_keywords,
                steps=args.steps,
                restarts=args.restarts,
                seed=seed,
                top_n=args.top_per_seed,
                screen_top_n=args.screen_top_n,
                word_weight=args.word_weight,
                slip_probability=args.slip_probability,
                backtrack_probability=args.backtrack_probability,
                dictionary_keyword_limit=args.dictionary_starts,
                calibration_keyword=calibration_keyword or None,
            )
        top = result.get("top_candidates") or []
        for rank, candidate in enumerate(top, start=1):
            record = _candidate_record(
                test_id=args.test_id,
                seed=seed,
                rank=rank,
                candidate=candidate,
                plaintext=test_data.plaintext,
                calibration_keyword=calibration_keyword or None,
            )
            raw_candidates.append(record)
            seed_records.append(record)
        label_summary = _seed_label_summary(seed_records)
        seed_summaries.append({
            "seed": seed,
            "engine": engine,
            "status": result.get("status"),
            "solver": result.get("solver"),
            "threads": result.get("threads"),
            "restart_jobs": result.get("restart_jobs"),
            "hillclimbs_per_restart": result.get("hillclimbs_per_restart"),
            "nominal_proposals": result.get("nominal_proposals"),
            "keyword_states_screened": result.get("keyword_states_screened"),
            "accepted_screen_mutations": result.get("accepted_screen_mutations"),
            "best_calibration_keyword_distance": (
                result.get("best_calibration_keyword_distance")
                if result.get("best_calibration_keyword_distance") is not None
                else label_summary["best_calibration_keyword_distance"]
            ),
            "exact_calibration_keyword_rank": (
                result.get("exact_calibration_keyword_rank")
                if result.get("exact_calibration_keyword_rank") is not None
                else label_summary["exact_calibration_keyword_rank"]
            ),
            "exact_calibration_keyword_count": label_summary["exact_calibration_keyword_count"],
            "retained_candidate_count": label_summary["retained_candidate_count"],
            "best_char_accuracy": label_summary["best_char_accuracy"],
            "best_char_keyword": label_summary["best_char_keyword"],
            "best_char_cycleword": label_summary["best_char_cycleword"],
        })
        if args.checkpoint_every_seed:
            payload = {
                "schema": "quagmire_candidate_capture_v1",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "test_id": args.test_id,
                "benchmark_root": str(args.benchmark_root),
                "split": args.split,
                "mode": mode,
                "config": {
                    "seeds": seeds,
                    "completed_seeds": [row["seed"] for row in seed_summaries],
                    "keyword_lengths": keyword_lengths,
                    "cycleword_lengths": cycleword_lengths,
                    "initial_keywords": initial_keywords,
                    "dictionary_starts": args.dictionary_starts,
                    "requested_engine": requested_engine,
                    "engine": engine,
                    "steps": args.steps,
                    "hillclimbs": hillclimbs,
                    "restarts": args.restarts,
                    "threads": args.threads,
                    "screen_top_n": args.screen_top_n,
                    "word_weight": args.word_weight,
                    "slip_probability": args.slip_probability,
                    "backtrack_probability": args.backtrack_probability,
                    "top_per_seed": args.top_per_seed,
                    "calibration_keyword": calibration_keyword or None,
                },
                "truth": {
                    "plaintext": test_data.plaintext if args.include_truth_in_output else "<withheld>",
                    "note": "Ground truth is used for offline labels; use --include-truth-in-output to persist it.",
                },
                "elapsed_seconds": round(time.time() - started, 3),
                "seed_summaries": seed_summaries,
                "summary": _summary(raw_candidates),
                "candidates": sorted(raw_candidates, key=lambda row: row.get("selection_score") or float("-inf"), reverse=True),
            }
            _write_json(output, payload)

    payload = {
        "schema": "quagmire_candidate_capture_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_id": args.test_id,
        "benchmark_root": str(args.benchmark_root),
        "split": args.split,
        "mode": mode,
        "config": {
            "seeds": seeds,
            "completed_seeds": [row["seed"] for row in seed_summaries],
            "keyword_lengths": keyword_lengths,
            "cycleword_lengths": cycleword_lengths,
            "initial_keywords": initial_keywords,
            "dictionary_starts": args.dictionary_starts,
            "requested_engine": requested_engine,
            "engine": engine,
            "steps": args.steps,
            "hillclimbs": hillclimbs,
            "restarts": args.restarts,
            "threads": args.threads,
            "screen_top_n": args.screen_top_n,
            "word_weight": args.word_weight,
            "slip_probability": args.slip_probability,
            "backtrack_probability": args.backtrack_probability,
            "top_per_seed": args.top_per_seed,
            "calibration_keyword": calibration_keyword or None,
        },
        "truth": {
            "plaintext": test_data.plaintext if args.include_truth_in_output else "<withheld>",
            "note": "Ground truth is used for offline labels; use --include-truth-in-output to persist it.",
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "seed_summaries": seed_summaries,
        "summary": _summary(raw_candidates),
        "candidates": sorted(raw_candidates, key=lambda row: row.get("selection_score") or float("-inf"), reverse=True),
    }
    _write_json(output, payload)
    print(f"Wrote {output}")
    print(
        f"Candidates: raw={len(raw_candidates)} elapsed={payload['elapsed_seconds']}s "
        f"best_distance={payload['summary'].get('best_calibration_keyword_distance')}"
    )
    best = payload["summary"].get("best_selection") or {}
    if best:
        print(
            f"Best: keyword={best.get('alphabet_keyword')} cycleword={best.get('cycleword')} "
            f"char={best.get('char_accuracy')} distance={best.get('calibration_keyword_distance')}"
        )


if __name__ == "__main__":
    main()
