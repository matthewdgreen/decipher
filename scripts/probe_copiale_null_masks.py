#!/usr/bin/env python3
"""Prototype null-mask search for Copiale evidence cases.

This is an experimental calibration tool, not a production solver route. It
generates candidate null/codeword masks without using plaintext, reruns the
homophonic solver on each filtered token stream, and then reports post-hoc
ground-truth scores so we can see whether null-aware search is worth promoting
into the automated runner.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from analysis.copiale_nulls import (
    format_validation_components,
    generate_null_masks,
    german_coherence_score,
    german_fragment_score,
    null_mask_validation_score,
    null_mask_validation_score_v2,
    repetitive_word_island_penalty,
    select_null_candidate_symbols,
)
from analysis.dictionary import get_dictionary_path, load_word_set, score_plaintext
from analysis.zenith_fast import zenith_solve_fast
from automated.runner import _automated_candidate_diagnostics, _plaintext_quality
from benchmark.loader import BenchmarkLoader, parse_canonical_transcription
from benchmark.scorer import score_decryption
from frontier.suite import load_frontier_suite
from scripts.report_copiale_evidence import diagnose_canonical_transcription


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Copiale null-mask candidates.")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument("--test-id", default="copiale_single_B_copiale_p052")
    parser.add_argument(
        "--suite-file",
        help="Optional frontier suite. When set, probe every test_id in the suite.",
    )
    parser.add_argument(
        "--summary-jsonl",
        help="Optional summary JSONL used to find per-test baseline artifacts.",
    )
    parser.add_argument(
        "--artifact",
        help="Optional baseline artifact used to seed candidate families.",
    )
    parser.add_argument("--model", default="models/ngram5_de.bin")
    parser.add_argument("--candidate-limit", type=int, default=14)
    parser.add_argument("--max-mask-size", type=int, default=1)
    parser.add_argument(
        "--max-masks",
        type=int,
        default=0,
        help="Optional cap on generated masks after the baseline. 0 means no cap.",
    )
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--sampler-iterations", type=int, default=1200)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--output-jsonl",
        help="Optional JSONL output with one probe summary per test.",
    )
    parser.add_argument(
        "--include-all-rows",
        action="store_true",
        help=(
            "Include every evaluated mask row in --output-jsonl. This makes "
            "later no-rerun finalist validation and scorer tuning possible."
        ),
    )
    args = parser.parse_args()

    loader = BenchmarkLoader(args.benchmark_root)
    tests = {test.test_id: test for test in loader.load_tests(args.split)}
    test_ids = _selected_test_ids(args)
    if len(test_ids) > 1 and args.artifact:
        raise SystemExit("--artifact can only be used with a single --test-id run")
    summary_rows = _read_summary(args.summary_jsonl)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    plaintext_ids = list(range(26))
    id_to_letter = {idx: chr(ord("A") + idx) for idx in plaintext_ids}
    word_list = load_word_set(get_dictionary_path("de"))
    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    for test_index, test_id in enumerate(test_ids, start=1):
        if test_id not in tests:
            raise SystemExit(f"test_id not found in {args.split}: {test_id}")
        test = tests[test_id]
        test_data = loader.load_test_data(test)
        artifact = _load_artifact(args.artifact) or _artifact_from_summary(summary_rows.get(test_id)) or {}
        if not artifact:
            print(f"[{test_id}] No baseline artifact supplied; using ciphertext-only null candidates.")
        if len(test_ids) > 1:
            print("=" * 96)
            print(f"[{test_index}/{len(test_ids)}] {test_id}")
        result = run_probe(
            test_id=test_id,
            canonical_transcription=test_data.canonical_transcription,
            plaintext=test_data.plaintext,
            artifact=artifact,
            model_path=args.model,
            candidate_limit=args.candidate_limit,
            max_mask_size=args.max_mask_size,
            max_masks=args.max_masks,
            seeds=seeds,
            plaintext_ids=plaintext_ids,
            id_to_letter=id_to_letter,
            word_list=word_list,
            epochs=args.epochs,
            sampler_iterations=args.sampler_iterations,
            top=args.top,
            include_all_rows=args.include_all_rows,
        )
        if output_path:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")


def run_probe(
    *,
    test_id: str,
    canonical_transcription: str,
    plaintext: str,
    artifact: dict[str, Any],
    model_path: str,
    candidate_limit: int,
    max_mask_size: int,
    max_masks: int,
    seeds: list[int],
    plaintext_ids: list[int],
    id_to_letter: dict[int, str],
    word_list: set[str],
    epochs: int,
    sampler_iterations: int,
    top: int,
    include_all_rows: bool = False,
) -> dict[str, Any]:
    cipher = parse_canonical_transcription(canonical_transcription)
    candidates = _candidate_symbols(
        canonical_transcription,
        artifact=artifact,
        limit=candidate_limit,
    )

    masks = generate_null_masks(candidates, max_mask_size)
    if max_masks > 0:
        masks = [()] + masks[1:max_masks + 1]

    print(f"Test: {test_id}")
    print(f"Candidate symbols ({len(candidates)}): {', '.join(candidates)}")
    print(f"Masks: {len(masks)}  seeds={seeds}  epochs={epochs}  iterations={sampler_iterations}")
    print()

    rows = []
    t0 = time.time()
    for idx, mask in enumerate(masks, start=1):
        mask_set = set(mask)
        filtered_tokens = [
            token
            for token in cipher.tokens
            if cipher.alphabet.decode([token]) not in mask_set
        ]
        best = None
        for seed in seeds:
            result = zenith_solve_fast(
                tokens=filtered_tokens,
                plaintext_ids=plaintext_ids,
                id_to_letter=id_to_letter,
                model_path=model_path,
                epochs=epochs,
                sampler_iterations=sampler_iterations,
                seed=seed,
                top_n=3,
            )
            if best is None or result.normalized_score > best.normalized_score:
                best = result
        assert best is not None
        quality = _plaintext_quality(best.plaintext, best.key)
        diagnostics = _automated_candidate_diagnostics(
            best.plaintext,
            language="de",
            word_list=word_list,
        )
        score = score_decryption(
            test_id,
            best.plaintext,
            plaintext,
            agent_score=score_plaintext(best.plaintext, word_list),
            status="completed",
        )
        selection_score = best.normalized_score - float(quality.get("penalty", 0.0))
        validation = null_mask_validation_score(
            {
                "mask": list(mask),
                "filtered_length": len(filtered_tokens),
                "selection_score": selection_score,
                "quality": quality,
                "diagnostics": diagnostics,
                "preview": best.plaintext[:120],
            },
            original_length=len(cipher.tokens),
        )
        validation_v2 = null_mask_validation_score_v2(
            {
                "mask": list(mask),
                "filtered_length": len(filtered_tokens),
                "selection_score": selection_score,
                "quality": quality,
                "diagnostics": diagnostics,
                "preview": best.plaintext[:120],
            },
            original_length=len(cipher.tokens),
        )
        row = {
            "mask": list(mask),
            "mask_size": len(mask),
            "filtered_length": len(filtered_tokens),
            "anneal_score": best.normalized_score,
            "selection_score": selection_score,
            "validation_score": validation["score"],
            "validation_components": validation["components"],
            "validation_score_v2": validation_v2["score"],
            "validation_components_v2": validation_v2["components"],
            "char_accuracy": score.char_accuracy,
            "word_accuracy": score.word_accuracy,
            "quality": quality,
            "diagnostics": diagnostics,
            "preview": best.plaintext[:120],
        }
        rows.append(row)
        print(
            f"[{idx:>3}/{len(masks)}] mask={','.join(mask) or '(none)'} "
            f"len={len(filtered_tokens)} char={score.char_accuracy:.1%} "
            f"word={score.word_accuracy:.1%} sel={selection_score:.3f} "
            f"val2={validation_v2['score']:.3f} "
            f"dict={diagnostics.get('dict_rate', 0.0):.3f} "
            f"top={quality.get('top_letter_fraction', 0.0):.3f}"
        )

    rows_by_selection = sorted(rows, key=lambda item: (-item["selection_score"], -item["char_accuracy"]))
    rows_by_validation = sorted(rows, key=lambda item: (-item["validation_score_v2"], -item["char_accuracy"]))
    rows_by_char = sorted(rows, key=lambda item: (-item["char_accuracy"], -item["selection_score"]))
    print()
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print("Top by solver selection score (no ground truth):")
    for row in rows_by_selection[: top]:
        print(
            f"  sel={row['selection_score']:.3f} char={row['char_accuracy']:.1%} "
            f"word={row['word_accuracy']:.1%} mask={','.join(row['mask']) or '(none)'} "
            f"len={row['filtered_length']} preview={row['preview']}"
        )
    print()
    print("Top by null-mask validation score (no ground truth):")
    for row in rows_by_validation[: top]:
        print(
            f"  val2={row['validation_score_v2']:.3f} sel={row['selection_score']:.3f} "
            f"char={row['char_accuracy']:.1%} word={row['word_accuracy']:.1%} "
            f"mask={','.join(row['mask']) or '(none)'} "
            f"components={format_validation_components(row['validation_components_v2'])} "
            f"preview={row['preview']}"
        )
    print()
    print("Top by post-hoc char accuracy:")
    for row in rows_by_char[: top]:
        print(
            f"  char={row['char_accuracy']:.1%} word={row['word_accuracy']:.1%} "
            f"mask={','.join(row['mask']) or '(none)'} "
            f"len={row['filtered_length']} sel={row['selection_score']:.3f} "
            f"preview={row['preview']}"
        )
    payload = {
        "test_id": test_id,
        "candidate_symbols": candidates,
        "mask_count": len(masks),
        "seeds": seeds,
        "epochs": epochs,
        "sampler_iterations": sampler_iterations,
        "elapsed_seconds": round(time.time() - t0, 3),
        "best_by_selection": rows_by_selection[0] if rows_by_selection else None,
        "best_by_validation": rows_by_validation[0] if rows_by_validation else None,
        "best_by_char_accuracy": rows_by_char[0] if rows_by_char else None,
        "top_by_selection": rows_by_selection[:top],
        "top_by_validation": rows_by_validation[:top],
        "top_by_char_accuracy": rows_by_char[:top],
    }
    if include_all_rows:
        payload["all_rows"] = rows
    return payload


def _selected_test_ids(args: argparse.Namespace) -> list[str]:
    if args.suite_file:
        return [case.test.test_id for case in load_frontier_suite(args.suite_file)]
    return [args.test_id]


def _read_summary(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    rows = {}
    if not p.exists():
        raise FileNotFoundError(p)
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        test_id = str(row.get("test_id") or "")
        if test_id:
            rows[test_id] = row
    return rows


def _artifact_from_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return _load_artifact(str(row.get("artifact_path") or ""))


def _load_artifact(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _candidate_symbols(
    canonical_text: str,
    artifact: dict[str, Any],
    limit: int,
) -> list[str]:
    diag = diagnose_canonical_transcription(canonical_text, artifact=artifact)
    return select_null_candidate_symbols(diag, limit=limit)


if __name__ == "__main__":
    main()
