#!/usr/bin/env python3
"""Run an experimental shared-key Copiale solve over multiple pages.

This is a diagnostic experiment, not a production benchmark route. It
concatenates several Copiale benchmark records into one ciphertext with one
shared S-token alphabet, runs the automated homophonic/null-mask solver, and
then projects the resulting shared key back onto each page for post-hoc
calibration.

Ground truth is only used after the solver has selected a result.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from analysis.homophonic_nulls import null_mask_validation_score_v2  # noqa: E402
from analysis.language_scoring import language_quality_feature_dict  # noqa: E402
from automated.runner import _cipher_text_from_tokens, _run_homophonic, run_automated  # noqa: E402
from automated.runner import _automated_candidate_diagnostics, _plaintext_quality, _word_list  # noqa: E402
from benchmark.loader import BenchmarkLoader  # noqa: E402
from benchmark.scorer import score_decryption  # noqa: E402
from models.alphabet import Alphabet  # noqa: E402
from models.cipher_text import CipherText  # noqa: E402


TOKEN_RE = re.compile(r"S\d+")
ROBUST_PAGE_RANK_POLICY = (
    "minimum page validation plus binary/evidence/window stability, "
    "minus variance and fragment-illusion penalty"
)


@dataclass(frozen=True)
class PageBundle:
    test_id: str
    canonical_transcription: str
    plaintext: str
    symbols: list[str]
    token_ids: list[int]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a shared-key multi-page Copiale experiment.")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument(
        "--test-ids",
        default="copiale_single_B_copiale_p017,copiale_single_B_copiale_p035,copiale_single_B_copiale_p052,copiale_single_B_copiale_p084",
        help="Comma-separated Copiale test ids to concatenate.",
    )
    parser.add_argument("--artifact-dir", default="artifacts/copiale_multipage_experiment")
    parser.add_argument("--homophonic-budget", choices=["screen", "full"], default="screen")
    parser.add_argument("--homophonic-refinement", default="null_masks")
    parser.add_argument("--homophonic-solver", default="zenith_native")
    parser.add_argument(
        "--structured-refine",
        action="store_true",
        help=(
            "After the combined solve, infer stable shared mappings from the "
            "combined finalist menu and rerun each page with those mappings pinned."
        ),
    )
    parser.add_argument("--consensus-top-n", type=int, default=8)
    parser.add_argument("--consensus-min-agreement", type=float, default=0.75)
    parser.add_argument("--structured-budget", choices=["screen", "full"], default="screen")
    parser.add_argument(
        "--elite-rerank-top-n",
        type=int,
        default=12,
        help="Top combined finalists to project back onto pages for page-aware reranking.",
    )
    parser.add_argument(
        "--elite-portfolio-size",
        type=int,
        default=12,
        help="Number of diverse page-aware finalists to preserve for later repair/validation.",
    )
    parser.add_argument(
        "--portfolio-refine",
        action="store_true",
        help="Run structured page refinement for each retained portfolio finalist.",
    )
    parser.add_argument(
        "--portfolio-refine-size",
        type=int,
        default=12,
        help="Maximum number of retained portfolio finalists to refine.",
    )
    parser.add_argument(
        "--portfolio-local-repair",
        action="store_true",
        help=(
            "Run a cheap localized repair probe for each retained portfolio finalist. "
            "This edits disputed symbols only inside damaged page windows."
        ),
    )
    parser.add_argument("--portfolio-local-repair-size", type=int, default=12)
    parser.add_argument("--local-repair-window-size", type=int, default=120)
    parser.add_argument("--local-repair-window-step", type=int, default=40)
    parser.add_argument("--local-repair-window-limit", type=int, default=6)
    parser.add_argument("--local-repair-max-symbols", type=int, default=7)
    parser.add_argument("--local-repair-max-alternatives", type=int, default=4)
    parser.add_argument("--local-repair-include-pairs", action="store_true")
    parser.add_argument(
        "--local-repair-min-validation-delta",
        type=float,
        default=0.03,
        help="Minimum validation-score gain required before a local edit can replace baseline text.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    test_ids = [item.strip() for item in args.test_ids.split(",") if item.strip()]
    if not test_ids:
        raise SystemExit("No --test-ids supplied.")
    loader = BenchmarkLoader(args.benchmark_root)
    tests = {test.test_id: test for test in loader.load_tests(args.split)}
    missing = [test_id for test_id in test_ids if test_id not in tests]
    if missing:
        raise SystemExit(f"Missing test ids in {args.split}: {', '.join(missing)}")

    combined_cipher, pages = build_combined_cipher(loader, tests, test_ids)
    combined_plaintext = "\n".join(page.plaintext for page in pages if page.plaintext)
    cipher_id = "copiale_multipage_" + "_".join(page.test_id.rsplit("_", 1)[-1] for page in pages)
    started = time.time()
    result = run_automated(
        combined_cipher,
        language="de",
        cipher_id=cipher_id,
        ground_truth=combined_plaintext,
        cipher_system="homophonic",
        homophonic_budget=args.homophonic_budget,
        homophonic_refinement=args.homophonic_refinement,
        homophonic_solver=args.homophonic_solver,
    )
    artifact = dict(result.artifact)
    selected_mask = selected_null_mask(artifact)
    page_rows = project_pages(
        pages=pages,
        key={int(k): int(v) for k, v in (artifact.get("key") or {}).items()},
        mask=selected_mask,
    )
    attach_page_scores(page_rows)
    structured_rows: list[dict[str, Any]] = []
    consensus = consensus_from_finalists(
        artifact=artifact,
        alphabet=combined_cipher.alphabet,
        top_n=args.consensus_top_n,
        min_agreement=args.consensus_min_agreement,
    )
    if args.structured_refine:
        structured_rows = refine_pages_with_shared_consensus(
            pages=pages,
            alphabet=combined_cipher.alphabet,
            selected_key={int(k): int(v) for k, v in (artifact.get("key") or {}).items()},
            selected_mask=selected_mask,
            consensus=consensus,
            budget=args.structured_budget,
            solver_profile=args.homophonic_solver,
        )
        attach_page_scores(structured_rows)
    elite_rerank = page_aware_elite_rerank(
        artifact=artifact,
        pages=pages,
        top_n=args.elite_rerank_top_n,
        portfolio_size=args.elite_portfolio_size,
    )
    portfolio_refinement = {}
    if args.portfolio_refine:
        portfolio_refinement = refine_elite_portfolio(
            artifact=artifact,
            pages=pages,
            alphabet=combined_cipher.alphabet,
            consensus=consensus,
            portfolio=elite_rerank.get("portfolio") or {},
            budget=args.structured_budget,
            solver_profile=args.homophonic_solver,
            limit=args.portfolio_refine_size,
        )
    portfolio_local_repair = {}
    if args.portfolio_local_repair:
        portfolio_local_repair = repair_elite_portfolio_locally(
            artifact=artifact,
            pages=pages,
            alphabet=combined_cipher.alphabet,
            consensus=consensus,
            portfolio=elite_rerank.get("portfolio") or {},
            limit=args.portfolio_local_repair_size,
            window_size=args.local_repair_window_size,
            window_step=args.local_repair_window_step,
            window_limit=args.local_repair_window_limit,
            max_symbols=args.local_repair_max_symbols,
            max_alternatives=args.local_repair_max_alternatives,
            include_pairs=args.local_repair_include_pairs,
            min_validation_delta=args.local_repair_min_validation_delta,
        )

    payload = {
        "experiment": "copiale_multipage_shared_key",
        "test_ids": test_ids,
        "cipher_id": cipher_id,
        "elapsed_seconds": round(time.time() - started, 3),
        "homophonic_budget": args.homophonic_budget,
        "homophonic_refinement": args.homophonic_refinement,
        "homophonic_solver": args.homophonic_solver,
        "environment": selected_environment_snapshot(),
        "combined": {
            "token_count": len(combined_cipher.tokens),
            "alphabet_size": combined_cipher.alphabet.size,
            "status": result.status,
            "solver": result.solver,
            "char_accuracy": result.char_accuracy,
            "word_accuracy": result.word_accuracy,
            "selected_mask": list(selected_mask),
            "artifact_run_id": result.run_id,
        },
        "consensus": consensus_summary(consensus),
        "pages": page_rows,
        "structured_pages": structured_rows,
        "elite_page_rerank": elite_rerank,
        "portfolio_refinement": portfolio_refinement,
        "portfolio_local_repair": portfolio_local_repair,
        "artifact": artifact,
    }
    markdown = render_markdown(payload)

    output_dir = resolve_path(Path(args.artifact_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = (
        resolve_path(Path(args.output))
        if args.output
        else output_dir / f"{cipher_id}_{result.run_id}.md"
    )
    json_output = (
        resolve_path(Path(args.json_output))
        if args.json_output
        else output.with_suffix(".json")
    )
    artifact_output = output.with_suffix(".artifact.json")
    output.write_text(markdown, encoding="utf-8")
    json_output.write_text(json.dumps(payload_without_large_artifact(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(markdown)
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")
    print(f"Wrote {artifact_output}")


def build_combined_cipher(
    loader: BenchmarkLoader,
    tests: dict[str, Any],
    test_ids: list[str],
) -> tuple[CipherText, list[PageBundle]]:
    page_symbols: list[list[str]] = []
    page_plaintexts: list[str] = []
    canonical_pages: list[str] = []
    seen: set[str] = set()
    alphabet_symbols: list[str] = []
    for test_id in test_ids:
        data = loader.load_test_data(tests[test_id])
        symbols = TOKEN_RE.findall(data.canonical_transcription)
        page_symbols.append(symbols)
        page_plaintexts.append(data.plaintext or "")
        canonical_pages.append(data.canonical_transcription)
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                alphabet_symbols.append(symbol)
    alphabet = Alphabet(alphabet_symbols)
    pages = [
        PageBundle(
            test_id=test_id,
            canonical_transcription=canonical_pages[index],
            plaintext=page_plaintexts[index],
            symbols=page_symbols[index],
            token_ids=[alphabet.id_for(symbol) for symbol in page_symbols[index]],
        )
        for index, test_id in enumerate(test_ids)
    ]
    raw = " | ".join(" ".join(page.symbols) for page in pages)
    return CipherText(raw=raw, alphabet=alphabet, source="copiale_multipage", separator=" | "), pages


def selected_null_mask(artifact: dict[str, Any]) -> tuple[str, ...]:
    for step in reversed(artifact.get("steps") or []):
        if not isinstance(step, dict) or step.get("name") != "search_null_masks":
            continue
        selected = step.get("selected")
        if isinstance(selected, dict):
            return tuple(str(symbol) for symbol in (selected.get("mask") or []))
    return ()


def consensus_from_finalists(
    *,
    artifact: dict[str, Any],
    alphabet: Alphabet,
    top_n: int,
    min_agreement: float,
) -> dict[str, dict[str, Any]]:
    rows = []
    for step in reversed(artifact.get("steps") or []):
        if not isinstance(step, dict) or step.get("name") != "search_null_masks":
            continue
        selected = step.get("selected")
        if isinstance(selected, dict):
            rows.append(selected)
        rows.extend(
            row for row in (step.get("top_finalists") or [])
            if isinstance(row, dict)
        )
        break
    if not rows:
        rows = [{"key": artifact.get("key") or {}, "mask": selected_null_mask(artifact)}]
    rows = rows[: max(1, top_n)]
    consensus: dict[str, dict[str, Any]] = {}
    for token_id in range(alphabet.size):
        symbol = alphabet.symbol_for(token_id)
        counts: dict[str, int] = {}
        for row in rows:
            mask = set(str(item) for item in (row.get("mask") or []))
            if symbol in mask:
                assignment = "<null>"
            else:
                key = row.get("key") if isinstance(row.get("key"), dict) else {}
                value = key.get(str(token_id), key.get(token_id))
                try:
                    value_int = int(value)
                except (TypeError, ValueError):
                    assignment = "?"
                else:
                    assignment = chr(ord("A") + value_int) if 0 <= value_int <= 25 else "?"
            counts[assignment] = counts.get(assignment, 0) + 1
        winner, winner_count = max(counts.items(), key=lambda item: (item[1], item[0]))
        agreement = winner_count / max(1, len(rows))
        consensus[symbol] = {
            "symbol": symbol,
            "token_id": token_id,
            "winner": winner,
            "agreement": round(agreement, 4),
            "stable": agreement >= min_agreement,
            "counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        }
    return consensus


def consensus_summary(consensus: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stable = [row for row in consensus.values() if row.get("stable")]
    stable_letters = [
        row for row in stable
        if isinstance(row.get("winner"), str)
        and len(str(row.get("winner"))) == 1
        and "A" <= str(row.get("winner")) <= "Z"
    ]
    stable_nulls = [row for row in stable if row.get("winner") == "<null>"]
    return {
        "symbol_count": len(consensus),
        "stable_symbol_count": len(stable),
        "stable_letter_count": len(stable_letters),
        "stable_null_count": len(stable_nulls),
        "disputed_symbol_count": len(consensus) - len(stable),
    }


def refine_pages_with_shared_consensus(
    *,
    pages: list[PageBundle],
    alphabet: Alphabet,
    selected_key: dict[int, int],
    selected_mask: tuple[str, ...],
    consensus: dict[str, dict[str, Any]],
    budget: str,
    solver_profile: str,
) -> list[dict[str, Any]]:
    stable_fixed_ids = {
        int(row["token_id"])
        for row in consensus.values()
        if row.get("stable")
        and isinstance(row.get("winner"), str)
        and len(str(row.get("winner"))) == 1
        and "A" <= str(row.get("winner")) <= "Z"
    }
    masked_symbols = set(selected_mask)
    rows = []
    for index, page in enumerate(pages, start=1):
        page_masked_ids = {
            alphabet.id_for(symbol)
            for symbol in masked_symbols
            if alphabet.has_symbol(symbol)
        }
        filtered_tokens = [
            token_id
            for symbol, token_id in zip(page.symbols, page.token_ids)
            if symbol not in masked_symbols
        ]
        present_ids = set(filtered_tokens)
        fixed_ids = (stable_fixed_ids & present_ids) - page_masked_ids
        page_cipher = _cipher_text_from_tokens(
            filtered_tokens,
            alphabet,
            source=f"copiale_structured_multipage:{page.test_id}",
        )
        started = time.time()
        try:
            solver, key, decryption, step = _run_homophonic(
                page_cipher,
                language="de",
                budget=budget,
                refinement="none",
                solver_profile=solver_profile,
                ground_truth=None,
                seed_offset=index * 10_000,
                initial_key=selected_key,
                fixed_cipher_ids=fixed_ids,
            )
            rows.append({
                "test_id": page.test_id,
                "status": "completed",
                "solver": solver,
                "token_count": len(page.token_ids),
                "filtered_length": len(filtered_tokens),
                "fixed_symbol_count": len(fixed_ids),
                "masked_symbol_count": len(page_masked_ids & present_ids),
                "elapsed_seconds": round(time.time() - started, 3),
                "decryption": decryption,
                "plaintext": page.plaintext,
                "key": {str(k): v for k, v in key.items()},
                "step": step,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "test_id": page.test_id,
                "status": "error",
                "solver": "structured_page_refine",
                "token_count": len(page.token_ids),
                "filtered_length": len(filtered_tokens),
                "fixed_symbol_count": len(fixed_ids),
                "masked_symbol_count": len(page_masked_ids & present_ids),
                "elapsed_seconds": round(time.time() - started, 3),
                "decryption": "",
                "plaintext": page.plaintext,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return rows


def project_pages(
    *,
    pages: list[PageBundle],
    key: dict[int, int],
    mask: tuple[str, ...],
) -> list[dict[str, Any]]:
    masked = set(mask)
    rows = []
    for page in pages:
        chars: list[str] = []
        for symbol, token_id in zip(page.symbols, page.token_ids):
            if symbol in masked:
                continue
            value = key.get(token_id)
            if value is None or value < 0 or value > 25:
                continue
            chars.append(chr(ord("A") + value))
        rows.append({
            "test_id": page.test_id,
            "token_count": len(page.token_ids),
            "filtered_length": len(chars),
            "decryption": "".join(chars),
            "plaintext": page.plaintext,
        })
    return rows


def attach_page_scores(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        score = score_decryption(
            row["test_id"],
            row["decryption"],
            row["plaintext"],
            agent_score=0.0,
            status=str(row.get("status") or "completed"),
        )
        row["char_accuracy"] = score.char_accuracy
        row["word_accuracy"] = score.word_accuracy
        row["preview"] = row["decryption"][:180]


def page_aware_elite_rerank(
    *,
    artifact: dict[str, Any],
    pages: list[PageBundle],
    top_n: int,
    portfolio_size: int,
) -> dict[str, Any]:
    finalists = finalist_rows(artifact, top_n=top_n)
    rows = []
    for finalist in finalists:
        key = {
            int(k): int(v)
            for k, v in (finalist.get("key") or {}).items()
        }
        mask = tuple(str(symbol) for symbol in (finalist.get("mask") or []))
        page_rows = project_pages(pages=pages, key=key, mask=mask)
        attach_page_scores(page_rows)
        runtime_scores = [score_page_runtime(row, key=key, mask=mask) for row in page_rows]
        metrics = page_runtime_metrics(runtime_scores)
        avg_char = mean(row["char_accuracy"] for row in page_rows)
        row_payload = {
            "label": finalist.get("_label", "finalist"),
            "finalist_order": int(finalist.get("_order") or 0),
            "source": finalist.get("source"),
            "mask": list(mask),
            "combined_validation_score_v2": finalist.get("validation_score_v2"),
            "combined_language_quality_rank_score": finalist.get("language_quality_rank_score"),
            "combined_selection_score": finalist.get("selection_score"),
            "post_hoc_char_avg": round(avg_char, 6),
            "post_hoc_page_chars": [
                {
                    "test_id": row["test_id"],
                    "char_accuracy": round(float(row["char_accuracy"]), 6),
                    "word_accuracy": round(float(row["word_accuracy"]), 6),
                }
                for row in page_rows
            ],
            "page_runtime_scores": runtime_scores,
            "preview": "; ".join(row["preview"][:60] for row in page_rows[:3]),
        }
        row_payload.update(metrics)
        row_payload["policy_scores"] = page_policy_scores(row_payload)
        rows.append(row_payload)
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["page_balanced_score"]),
            float(row["page_validation_avg"]),
            float(row.get("page_robust_score") or row["page_balanced_score"]),
        ),
        reverse=True,
    )
    selected_label = "selected"
    selected_rank = next(
        (idx for idx, row in enumerate(ranked, start=1) if row["label"] == selected_label),
        None,
    )
    policy_rows = policy_winners(rows)
    portfolio = build_elite_portfolio(rows, ranked=ranked, size=portfolio_size)
    return {
        "enabled": bool(finalists),
        "top_n": top_n,
        "portfolio_size": portfolio_size,
        "rank_policy": "page_validation_avg + 0.20*page_validation_min - 0.15*page_validation_std",
        "robust_policy": ROBUST_PAGE_RANK_POLICY,
        "selected_rank": selected_rank,
        "policy_winners": policy_rows,
        "portfolio": portfolio,
        "rows": ranked,
    }


def build_elite_portfolio(
    rows: list[dict[str, Any]],
    *,
    ranked: list[dict[str, Any]],
    size: int,
) -> dict[str, Any]:
    """Build a small diverse finalist portfolio without ground truth.

    The goal is recall, not scalar selection: preserve candidates chosen by
    different page-aware policies, the current selected finalist, and the main
    balanced ranking, then backfill by original finalist order.
    """
    if not rows:
        return {
            "size": 0,
            "rows": [],
            "captures_post_hoc_best": False,
            "post_hoc_best_label": None,
            "best_post_hoc_char_in_portfolio": None,
            "best_post_hoc_label_in_portfolio": None,
        }
    target = max(1, int(size))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any] | None, reason: str) -> None:
        if row is None or len(selected) >= target:
            return
        label = str(row.get("label") or "")
        if not label or label in seen:
            return
        copied = dict(row)
        copied["portfolio_reason"] = reason
        selected.append(copied)
        seen.add(label)

    by_label = {str(row.get("label")): row for row in rows}
    add(by_label.get("selected"), "current_selected")

    policy_names = sorted((rows[0].get("policy_scores") or {}).keys())
    for policy in policy_names:
        policy_ranked = sorted(
            rows,
            key=lambda row: (
                float((row.get("policy_scores") or {}).get(policy) or float("-inf")),
                float(row.get("page_validation_avg") or float("-inf")),
            ),
            reverse=True,
        )
        for row in policy_ranked[:2]:
            add(row, f"policy:{policy}")

    for row in ranked:
        add(row, "primary_rank")

    original_order = sorted(rows, key=lambda row: int(row.get("finalist_order") or 0))
    for row in original_order:
        add(row, "original_finalist_order")

    post_hoc_best = max(rows, key=lambda row: float(row.get("post_hoc_char_avg") or 0.0))
    portfolio_best = max(selected, key=lambda row: float(row.get("post_hoc_char_avg") or 0.0))
    return {
        "size": len(selected),
        "requested_size": target,
        "captures_post_hoc_best": any(row["label"] == post_hoc_best["label"] for row in selected),
        "post_hoc_best_label": post_hoc_best["label"],
        "post_hoc_best_char_avg": post_hoc_best["post_hoc_char_avg"],
        "best_post_hoc_label_in_portfolio": portfolio_best["label"],
        "best_post_hoc_char_in_portfolio": portfolio_best["post_hoc_char_avg"],
        "rows": [
            {
                "label": row["label"],
                "mask": row["mask"],
                "portfolio_reason": row.get("portfolio_reason"),
                "page_balanced_score": row["page_balanced_score"],
                "page_robust_score": row.get("page_robust_score"),
                "fragment_illusion_penalty": row.get("fragment_illusion_penalty"),
                "post_hoc_char_avg": row["post_hoc_char_avg"],
                "post_hoc_page_chars": row["post_hoc_page_chars"],
            }
            for row in selected
        ],
    }


def refine_elite_portfolio(
    *,
    artifact: dict[str, Any],
    pages: list[PageBundle],
    alphabet: Alphabet,
    consensus: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
    budget: str,
    solver_profile: str,
    limit: int,
) -> dict[str, Any]:
    finalist_by_label = {
        str(row.get("_label")): row
        for row in finalist_rows(artifact, top_n=10_000)
    }
    rows = []
    for portfolio_row in (portfolio.get("rows") or [])[: max(1, int(limit))]:
        label = str(portfolio_row.get("label") or "")
        finalist = finalist_by_label.get(label)
        if not finalist:
            continue
        selected_key = {
            int(k): int(v)
            for k, v in (finalist.get("key") or {}).items()
        }
        selected_mask = tuple(str(symbol) for symbol in (finalist.get("mask") or []))
        started = time.time()
        page_rows = refine_pages_with_shared_consensus(
            pages=pages,
            alphabet=alphabet,
            selected_key=selected_key,
            selected_mask=selected_mask,
            consensus=consensus,
            budget=budget,
            solver_profile=solver_profile,
        )
        attach_page_scores(page_rows)
        runtime_scores = [
            score_page_runtime(
                row,
                key={int(k): int(v) for k, v in (row.get("key") or {}).items()},
                mask=selected_mask,
            )
            for row in page_rows
            if row.get("status") == "completed"
        ]
        metrics = page_runtime_metrics(runtime_scores)
        avg_char = mean(row["char_accuracy"] for row in page_rows) if page_rows else 0.0
        row_payload = {
            "label": label,
            "source_portfolio_reason": portfolio_row.get("portfolio_reason"),
            "mask": list(selected_mask),
            "elapsed_seconds": round(time.time() - started, 3),
            "post_hoc_char_avg": round(avg_char, 6),
            "post_hoc_page_chars": [
                {
                    "test_id": row["test_id"],
                    "char_accuracy": round(float(row["char_accuracy"]), 6),
                    "word_accuracy": round(float(row["word_accuracy"]), 6),
                }
                for row in page_rows
            ],
            "page_runtime_scores": runtime_scores,
            "pages": compact_refined_pages(page_rows),
        }
        row_payload.update(metrics)
        rows.append(row_payload)
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("page_robust_score") or row["page_balanced_score"]),
            float(row["page_balanced_score"]),
            float(row["page_validation_avg"]),
        ),
        reverse=True,
    )
    post_hoc_ranked = sorted(
        rows,
        key=lambda row: float(row.get("post_hoc_char_avg") or 0.0),
        reverse=True,
    )
    selected_rank = next(
        (idx for idx, row in enumerate(ranked, start=1) if row["label"] == "selected"),
        None,
    )
    return {
        "enabled": True,
        "budget": budget,
        "solver_profile": solver_profile,
        "candidate_count": len(rows),
        "rank_policy": ROBUST_PAGE_RANK_POLICY,
        "selected_rank": selected_rank,
        "best_by_policy": ranked[0] if ranked else None,
        "best_by_post_hoc_char": post_hoc_ranked[0] if post_hoc_ranked else None,
        "rows": ranked,
    }


def repair_elite_portfolio_locally(
    *,
    artifact: dict[str, Any],
    pages: list[PageBundle],
    alphabet: Alphabet,
    consensus: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
    limit: int,
    window_size: int,
    window_step: int,
    window_limit: int,
    max_symbols: int,
    max_alternatives: int,
    include_pairs: bool,
    min_validation_delta: float,
) -> dict[str, Any]:
    finalist_by_label = {
        str(row.get("_label")): row
        for row in finalist_rows(artifact, top_n=10_000)
    }
    rows = []
    for portfolio_row in (portfolio.get("rows") or [])[: max(1, int(limit))]:
        label = str(portfolio_row.get("label") or "")
        finalist = finalist_by_label.get(label)
        if not finalist:
            continue
        selected_key = {
            int(k): int(v)
            for k, v in (finalist.get("key") or {}).items()
        }
        selected_mask = tuple(str(symbol) for symbol in (finalist.get("mask") or []))
        started = time.time()
        page_rows = [
            repair_page_locally(
                page=page,
                key=selected_key,
                mask=selected_mask,
                alphabet=alphabet,
                consensus=consensus,
                window_size=window_size,
                window_step=window_step,
                window_limit=window_limit,
                max_symbols=max_symbols,
                max_alternatives=max_alternatives,
                include_pairs=include_pairs,
                min_validation_delta=min_validation_delta,
            )
            for page in pages
        ]
        attach_page_scores(page_rows)
        runtime_scores = [
            score_page_runtime(row, key=selected_key, mask=selected_mask)
            for row in page_rows
        ]
        metrics = page_runtime_metrics(runtime_scores)
        avg_char = mean(row["char_accuracy"] for row in page_rows) if page_rows else 0.0
        changed_pages = sum(1 for row in page_rows if row.get("selected_edits") != ["baseline"])
        row_payload = {
            "label": label,
            "source_portfolio_reason": portfolio_row.get("portfolio_reason"),
            "mask": list(selected_mask),
            "elapsed_seconds": round(time.time() - started, 3),
            "changed_page_count": changed_pages,
            "post_hoc_char_avg": round(avg_char, 6),
            "post_hoc_page_chars": [
                {
                    "test_id": row["test_id"],
                    "char_accuracy": round(float(row["char_accuracy"]), 6),
                    "word_accuracy": round(float(row["word_accuracy"]), 6),
                    "selected_edits": row.get("selected_edits"),
                }
                for row in page_rows
            ],
            "page_runtime_scores": runtime_scores,
            "pages": compact_local_repair_pages(page_rows),
        }
        row_payload.update(metrics)
        rows.append(row_payload)
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("page_robust_score") or row["page_balanced_score"]),
            float(row["page_balanced_score"]),
            float(row["page_validation_avg"]),
        ),
        reverse=True,
    )
    post_hoc_ranked = sorted(
        rows,
        key=lambda row: float(row.get("post_hoc_char_avg") or 0.0),
        reverse=True,
    )
    return {
        "enabled": True,
        "candidate_count": len(rows),
        "window_size": window_size,
        "window_step": window_step,
        "window_limit": window_limit,
        "max_symbols": max_symbols,
        "max_alternatives": max_alternatives,
        "include_pairs": bool(include_pairs),
        "min_validation_delta": float(min_validation_delta),
        "rank_policy": ROBUST_PAGE_RANK_POLICY,
        "best_by_policy": ranked[0] if ranked else None,
        "best_by_post_hoc_char": post_hoc_ranked[0] if post_hoc_ranked else None,
        "rows": ranked,
    }


def repair_page_locally(
    *,
    page: PageBundle,
    key: dict[int, int],
    mask: tuple[str, ...],
    alphabet: Alphabet,
    consensus: dict[str, dict[str, Any]],
    window_size: int,
    window_step: int,
    window_limit: int,
    max_symbols: int,
    max_alternatives: int,
    include_pairs: bool,
    min_validation_delta: float,
) -> dict[str, Any]:
    text, sources = project_page_with_sources(page=page, key=key, mask=mask, alphabet=alphabet)
    windows = damaged_windows_for_text(
        text=text,
        sources=sources,
        consensus=consensus,
        window_size=window_size,
        step=window_step,
        limit=window_limit,
    )
    edit_sets = local_edit_sets(
        windows=windows,
        key=key,
        mask=mask,
        alphabet=alphabet,
        consensus=consensus,
        max_symbols=max_symbols,
        max_alternatives=max_alternatives,
        include_pairs=include_pairs,
    )
    baseline = score_local_variant(
        page=page,
        text=text,
        key=key,
        mask=mask,
        edits=("baseline",),
        window_label="baseline",
        changed_positions=0,
        deleted_positions=0,
        windows=windows,
    )
    variants = [baseline]
    for edit_set in edit_sets:
        for window in windows:
            repaired, changed, deleted = apply_window_edits(text, sources, edit_set, window)
            if changed == 0 and deleted == 0:
                continue
            variants.append(
                score_local_variant(
                    page=page,
                    text=repaired,
                    key=key,
                    mask=mask,
                    edits=edit_labels_for_local_set(edit_set, key=key, mask=mask, alphabet=alphabet),
                    window_label=f"{window['start']}-{window['end']}",
                    changed_positions=changed,
                    deleted_positions=deleted,
                    windows=windows,
                )
            )
    seen: set[tuple[tuple[str, ...], str]] = set()
    unique_variants = []
    for variant in variants:
        identity = (tuple(variant["selected_edits"]), variant["decryption"])
        if identity in seen:
            continue
        seen.add(identity)
        unique_variants.append(variant)
    unique_variants.sort(key=local_repair_rank_key, reverse=True)
    best = unique_variants[0]
    baseline_score = float(baseline.get("validation_score_v2") or 0.0)
    best_score = float(best.get("validation_score_v2") or 0.0)
    selected_source = (
        best
        if best.get("selected_edits") == ["baseline"]
        or best_score - baseline_score >= float(min_validation_delta)
        else baseline
    )
    selected = dict(selected_source)
    selected["best_variant_validation_delta"] = round(best_score - baseline_score, 6)
    selected["accepted_best_variant"] = selected_source is best
    selected["candidate_count"] = len(unique_variants)
    selected["window_count"] = len(windows)
    selected["top_local_variants"] = compact_local_variants(unique_variants[:8])
    return selected


def project_page_with_sources(
    *,
    page: PageBundle,
    key: dict[int, int],
    mask: tuple[str, ...],
    alphabet: Alphabet,
) -> tuple[str, list[str]]:
    masked = set(mask)
    chars: list[str] = []
    sources: list[str] = []
    for symbol, token_id in zip(page.symbols, page.token_ids):
        if symbol in masked:
            continue
        value = key.get(token_id)
        if value is None or value < 0 or value > 25:
            continue
        chars.append(chr(ord("A") + value))
        sources.append(alphabet.symbol_for(token_id))
    return "".join(chars), sources


def damaged_windows_for_text(
    *,
    text: str,
    sources: list[str],
    consensus: dict[str, dict[str, Any]],
    window_size: int,
    step: int,
    limit: int,
) -> list[dict[str, Any]]:
    if not text:
        return []
    size = max(20, int(window_size))
    stride = max(1, int(step))
    if len(text) <= size:
        starts = [0]
    else:
        starts = list(range(0, max(1, len(text) - size + 1), stride))
        if starts[-1] != len(text) - size:
            starts.append(len(text) - size)
    windows = []
    for start in starts:
        end = min(len(text), start + size)
        snippet = text[start:end]
        features = language_quality_feature_dict(snippet, language="de")
        symbol_counts = Counter(sources[start:end])
        disputed = []
        for symbol, count in symbol_counts.most_common():
            info = consensus.get(symbol) or {}
            if info.get("stable"):
                continue
            disputed.append({
                "symbol": symbol,
                "count": count,
                "winner": info.get("winner"),
                "agreement": info.get("agreement"),
                "assignments": info.get("counts") or {},
            })
        windows.append({
            "start": start,
            "end": end,
            "damage_score": round(window_damage_score(features), 6),
            "disputed_symbol_count": len(disputed),
            "disputed_symbols": disputed[:10],
            "text": snippet,
        })
    return sorted(
        windows,
        key=lambda row: (
            float(row["damage_score"]),
            int(row["disputed_symbol_count"]),
        ),
        reverse=True,
    )[: max(0, int(limit))]


def local_edit_sets(
    *,
    windows: list[dict[str, Any]],
    key: dict[int, int],
    mask: tuple[str, ...],
    alphabet: Alphabet,
    consensus: dict[str, dict[str, Any]],
    max_symbols: int,
    max_alternatives: int,
    include_pairs: bool,
) -> list[tuple[tuple[str, str], ...]]:
    edits: list[tuple[str, str]] = []
    seen_symbols: set[str] = set()
    for window in windows:
        for item in window.get("disputed_symbols") or []:
            symbol = str(item.get("symbol") or "")
            if not symbol or symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            current = current_assignment(symbol, key=key, mask=mask, alphabet=alphabet)
            assignments = consensus.get(symbol, {}).get("counts") or {}
            alternatives = [
                str(value) for value, _count in sorted(
                    assignments.items(),
                    key=lambda pair: (-int(pair[1]), str(pair[0]) == "<null>", str(pair[0])),
                )
                if str(value) not in {current, "?"}
            ][: max(1, int(max_alternatives))]
            for alternative in alternatives:
                if alternative == "<null>" or (len(alternative) == 1 and "A" <= alternative <= "Z"):
                    edits.append((symbol, alternative))
            if len(seen_symbols) >= max(1, int(max_symbols)):
                break
        if len(seen_symbols) >= max(1, int(max_symbols)):
            break
    edit_sets: list[tuple[tuple[str, str], ...]] = [(edit,) for edit in edits]
    if include_pairs:
        for left_index, left in enumerate(edits):
            for right in edits[left_index + 1:]:
                if left[0] != right[0]:
                    edit_sets.append((left, right))
    return edit_sets


def current_assignment(
    symbol: str,
    *,
    key: dict[int, int],
    mask: tuple[str, ...],
    alphabet: Alphabet,
) -> str:
    if symbol in set(mask):
        return "<null>"
    if not alphabet.has_symbol(symbol):
        return "?"
    value = key.get(alphabet.id_for(symbol))
    if value is None or value < 0 or value > 25:
        return "?"
    return chr(ord("A") + value)


def apply_window_edits(
    text: str,
    sources: list[str],
    edit_set: tuple[tuple[str, str], ...],
    window: dict[str, Any],
) -> tuple[str, int, int]:
    replacements = {symbol: value for symbol, value in edit_set}
    start = max(0, int(window.get("start") or 0))
    end = min(len(text), int(window.get("end") or start))
    chars: list[str] = []
    changed = 0
    deleted = 0
    for index, char in enumerate(text):
        replacement = replacements.get(sources[index]) if start <= index < end else None
        if replacement is None:
            chars.append(char)
        elif replacement == "<null>":
            deleted += 1
        elif len(replacement) == 1 and "A" <= replacement <= "Z":
            chars.append(replacement)
            if replacement != char:
                changed += 1
        else:
            chars.append(char)
    return "".join(chars), changed, deleted


def score_local_variant(
    *,
    page: PageBundle,
    text: str,
    key: dict[int, int],
    mask: tuple[str, ...],
    edits: tuple[str, ...],
    window_label: str,
    changed_positions: int,
    deleted_positions: int,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    row = {
        "test_id": page.test_id,
        "token_count": len(page.token_ids),
        "filtered_length": len(text),
        "decryption": text,
        "plaintext": page.plaintext,
    }
    runtime_score = score_page_runtime(row, key=key, mask=mask)
    repair_damage = mean(
        window_damage_for_text(text[max(0, int(window["start"])):min(len(text), int(window["end"]))])
        for window in windows
    ) if windows else 0.0
    return {
        **row,
        "status": "completed",
        "selected_edits": list(edits),
        "window": window_label,
        "changed_positions": changed_positions,
        "deleted_positions": deleted_positions,
        "repair_window_damage": round(repair_damage, 6),
        "validation_score_v2": round(runtime_score["validation_score_v2"], 6),
        "language_quality_mean": round(runtime_score["language_quality_mean"], 6),
        "dict_rate": runtime_score.get("dict_rate"),
        "runtime_score": runtime_score,
        "preview": text[:180],
    }


def edit_labels_for_local_set(
    edit_set: tuple[tuple[str, str], ...],
    *,
    key: dict[int, int],
    mask: tuple[str, ...],
    alphabet: Alphabet,
) -> tuple[str, ...]:
    return tuple(
        f"{symbol}:{current_assignment(symbol, key=key, mask=mask, alphabet=alphabet)}->{alternative}"
        for symbol, alternative in edit_set
    )


def local_repair_rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row.get("validation_score_v2") or 0.0),
        float(row.get("language_quality_mean") or 0.0),
        -float(row.get("repair_window_damage") or 0.0),
        -int(row.get("changed_positions") or 0) - int(row.get("deleted_positions") or 0),
    )


def window_damage_score(features: dict[str, float]) -> float:
    good = (
        0.28 * float(features.get("language_coherence") or 0.0)
        + 0.22 * float(features.get("language_shape") or 0.0)
        + 0.15 * float(features.get("language_evidence_dispersion") or 0.0)
        + 0.12 * float(features.get("function_content_balance") or 0.0)
        + 0.10 * float(features.get("repetition_control") or 0.0)
        + 0.08 * float(features.get("function_overuse_control") or 0.0)
        + 0.05 * float(features.get("short_fragment_control") or 0.0)
    )
    return max(0.0, min(1.0, 1.0 - good))


def window_damage_for_text(text: str) -> float:
    if not text:
        return 0.0
    return window_damage_score(language_quality_feature_dict(text, language="de"))


def compact_local_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "edits": row.get("selected_edits"),
            "window": row.get("window"),
            "changed_positions": row.get("changed_positions"),
            "deleted_positions": row.get("deleted_positions"),
            "validation_score_v2": row.get("validation_score_v2"),
            "language_quality_mean": row.get("language_quality_mean"),
            "repair_window_damage": row.get("repair_window_damage"),
            "preview": row.get("preview"),
        }
        for row in rows
    ]


def compact_local_repair_pages(page_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "test_id": row.get("test_id"),
            "selected_edits": row.get("selected_edits"),
            "window": row.get("window"),
            "candidate_count": row.get("candidate_count"),
            "window_count": row.get("window_count"),
            "best_variant_validation_delta": row.get("best_variant_validation_delta"),
            "accepted_best_variant": row.get("accepted_best_variant"),
            "changed_positions": row.get("changed_positions"),
            "deleted_positions": row.get("deleted_positions"),
            "validation_score_v2": row.get("validation_score_v2"),
            "language_quality_mean": row.get("language_quality_mean"),
            "repair_window_damage": row.get("repair_window_damage"),
            "char_accuracy": row.get("char_accuracy"),
            "word_accuracy": row.get("word_accuracy"),
            "preview": row.get("preview"),
            "top_local_variants": row.get("top_local_variants"),
        }
        for row in page_rows
    ]


def compact_refined_pages(page_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "test_id": row.get("test_id"),
            "status": row.get("status"),
            "filtered_length": row.get("filtered_length"),
            "fixed_symbol_count": row.get("fixed_symbol_count"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "char_accuracy": row.get("char_accuracy"),
            "word_accuracy": row.get("word_accuracy"),
            "preview": row.get("preview"),
        }
        for row in page_rows
    ]


def page_policy_scores(row: dict[str, Any]) -> dict[str, float]:
    avg = float(row["page_validation_avg"])
    min_score = float(row["page_validation_min"])
    std = float(row["page_validation_std"])
    lq = float(row["page_language_quality_avg"])
    content = float(row["page_content_char_avg"])
    pseudo = float(row["page_pseudo_word_avg"])
    binary = float(row["page_binary_component_avg"])
    shape = float(row["page_shape_component_avg"])
    robust = float(row.get("page_robust_score") or row.get("page_balanced_score") or 0.0)
    illusion = float(row.get("fragment_illusion_penalty") or 0.0)
    dispersion = float(row.get("page_evidence_dispersion_avg") or 0.0)
    stability = float(row.get("page_window_stability_avg") or 0.0)
    repetition = float(row.get("page_repetition_control_avg") or 0.0)
    combined = float(row.get("combined_validation_score_v2") or 0.0)
    return {
        "robust_anti_fragment": robust,
        "balanced_v1": avg + 0.20 * min_score - 0.15 * std,
        "avg_validation": avg,
        "conservative_min": avg + 0.35 * min_score - 0.30 * std,
        "lq_content_blend": avg + 0.35 * lq + 0.35 * content - 0.20 * pseudo,
        "shape_binary_blend": avg + 0.25 * shape + 0.20 * binary - 0.15 * std,
        "evidence_consistency": avg + 0.28 * binary + 0.20 * dispersion + 0.16 * stability + 0.16 * repetition - 0.65 * illusion,
        "combined_page_blend": avg + 0.10 * combined + 0.15 * lq - 0.10 * std,
    }


def policy_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    policy_names = sorted((rows[0].get("policy_scores") or {}).keys())
    winners = []
    for policy in policy_names:
        ranked = sorted(
            rows,
            key=lambda row: (
                float((row.get("policy_scores") or {}).get(policy) or float("-inf")),
                float(row.get("page_validation_avg") or float("-inf")),
            ),
            reverse=True,
        )
        winner = ranked[0]
        winners.append({
            "policy": policy,
            "winner": winner["label"],
            "mask": winner["mask"],
            "score": round(float((winner.get("policy_scores") or {}).get(policy) or 0.0), 6),
            "post_hoc_char_avg": winner["post_hoc_char_avg"],
            "rank_of_post_hoc_best": next(
                (
                    idx for idx, row in enumerate(ranked, start=1)
                    if row["post_hoc_char_avg"] == max(candidate["post_hoc_char_avg"] for candidate in rows)
                ),
                None,
            ),
        })
    return winners


def finalist_rows(artifact: dict[str, Any], *, top_n: int) -> list[dict[str, Any]]:
    rows = []
    for step in reversed(artifact.get("steps") or []):
        if not isinstance(step, dict) or step.get("name") != "search_null_masks":
            continue
        selected = step.get("selected")
        if isinstance(selected, dict):
            selected = dict(selected)
            selected["_label"] = "selected"
            selected["_order"] = 0
            rows.append(selected)
        for idx, row in enumerate(step.get("top_finalists") or [], start=1):
            if isinstance(row, dict):
                copied = dict(row)
                copied["_label"] = f"top{idx}"
                copied["_order"] = idx
                rows.append(copied)
        break
    if not rows and artifact.get("key"):
        rows.append({
            "_label": "selected",
            "_order": 0,
            "key": artifact.get("key") or {},
            "mask": selected_null_mask(artifact),
        })
    seen: set[tuple[tuple[str, ...], tuple[tuple[str, Any], ...]]] = set()
    unique = []
    for row in rows:
        identity = (
            tuple(str(symbol) for symbol in (row.get("mask") or [])),
            tuple(sorted((row.get("key") or {}).items())),
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(row)
    return unique[: max(1, top_n)]


def score_page_runtime(
    row: dict[str, Any],
    *,
    key: dict[int, int],
    mask: tuple[str, ...],
) -> dict[str, Any]:
    text = row["decryption"]
    word_list = _word_list("de")
    model_path = Path(os.environ.get("DECIPHER_NGRAM_MODEL_DE", "models/ngram5_de.bin"))
    binary_model_path = model_path if model_path.exists() else None
    quality = _plaintext_quality(text, key)
    diagnostics = _automated_candidate_diagnostics(
        text,
        language="de",
        word_list=word_list,
        binary_model_path=binary_model_path,
    )
    validation_row = {
        "mask": list(mask),
        "filtered_length": row["filtered_length"],
        "selection_score": 0.0,
        "quality": quality,
        "diagnostics": diagnostics,
        "decryption": text,
        "preview": text,
    }
    validation = null_mask_validation_score_v2(
        validation_row,
        original_length=int(row["token_count"]),
        language="de",
    )
    features = language_quality_feature_dict(
        text,
        diagnostics=diagnostics,
        language="de",
        original_length=int(row["token_count"]),
        filtered_length=int(row["filtered_length"]),
        mask_size=len(mask),
    )
    return {
        "test_id": row["test_id"],
        "validation_score_v2": float(validation["score"]),
        "validation_components_v2": dict(validation["components"]),
        "language_quality_mean": mean(float(value) for value in features.values()),
        "language_quality_features": {key: float(value) for key, value in features.items()},
        "dict_rate": float(diagnostics.get("dict_rate") or 0.0),
        "diagnostics": {
            key: diagnostics.get(key)
            for key in (
                "dict_rate",
                "dictionary_content_word_fraction",
                "dictionary_content_char_fraction",
                "dictionary_long_content_word_count",
                "pseudo_word_fraction",
                "long_pseudo_word_fraction",
                "binary_ngram_mean_log_prob",
                "segmentation_cost",
            )
            if key in diagnostics
        },
    }


def page_runtime_metrics(runtime_scores: list[dict[str, Any]]) -> dict[str, float]:
    if not runtime_scores:
        return {
            "page_validation_avg": 0.0,
            "page_validation_min": 0.0,
            "page_validation_std": 0.0,
            "page_balanced_score": 0.0,
            "page_robust_score": 0.0,
            "fragment_illusion_penalty": 0.0,
            "page_language_quality_avg": 0.0,
            "page_dict_avg": 0.0,
            "page_content_char_avg": 0.0,
            "page_pseudo_word_avg": 0.0,
            "page_binary_component_avg": 0.0,
            "page_shape_component_avg": 0.0,
            "page_evidence_dispersion_avg": 0.0,
            "page_window_stability_avg": 0.0,
            "page_repetition_control_avg": 0.0,
            "page_content_word_quality_avg": 0.0,
            "page_language_coherence_avg": 0.0,
        }
    avg_validation = mean(item["validation_score_v2"] for item in runtime_scores)
    min_validation = min(item["validation_score_v2"] for item in runtime_scores)
    std_validation = stddev(item["validation_score_v2"] for item in runtime_scores)
    page_balanced_score = avg_validation + 0.20 * min_validation - 0.15 * std_validation
    avg_lq = mean(item["language_quality_mean"] for item in runtime_scores)
    avg_dict = mean(item["dict_rate"] for item in runtime_scores)
    avg_content_chars = nested_mean(runtime_scores, "diagnostics", "dictionary_content_char_fraction")
    avg_pseudo = nested_mean(runtime_scores, "diagnostics", "pseudo_word_fraction")
    avg_binary_component = nested_mean(runtime_scores, "validation_components_v2", "binary_ngram_fit")
    avg_shape = nested_mean(runtime_scores, "validation_components_v2", "language_shape")
    avg_coherence = nested_mean(runtime_scores, "validation_components_v2", "language_coherence")
    avg_content_quality = nested_mean(runtime_scores, "validation_components_v2", "content_word_quality")
    avg_dispersion = nested_mean(runtime_scores, "language_quality_features", "language_evidence_dispersion")
    avg_stability = nested_mean(runtime_scores, "language_quality_features", "language_window_stability")
    avg_repetition = nested_mean(runtime_scores, "language_quality_features", "repetition_control")
    fragment_illusion = fragment_illusion_penalty(
        content_word_quality=avg_content_quality,
        language_coherence=avg_coherence,
        language_shape=avg_shape,
        binary_ngram_fit=avg_binary_component,
        evidence_dispersion=avg_dispersion,
        window_stability=avg_stability,
        repetition_control=avg_repetition,
    )
    page_robust_score = (
        min_validation
        + 0.35 * avg_binary_component
        + 0.25 * avg_dispersion
        + 0.20 * avg_stability
        + 0.15 * avg_repetition
        - 0.15 * std_validation
        - 0.75 * fragment_illusion
    )
    return {
        "page_validation_avg": round(avg_validation, 6),
        "page_validation_min": round(min_validation, 6),
        "page_validation_std": round(std_validation, 6),
        "page_balanced_score": round(page_balanced_score, 6),
        "page_robust_score": round(page_robust_score, 6),
        "fragment_illusion_penalty": round(fragment_illusion, 6),
        "page_language_quality_avg": round(avg_lq, 6),
        "page_dict_avg": round(avg_dict, 6),
        "page_content_char_avg": round(avg_content_chars, 6),
        "page_pseudo_word_avg": round(avg_pseudo, 6),
        "page_binary_component_avg": round(avg_binary_component, 6),
        "page_shape_component_avg": round(avg_shape, 6),
        "page_evidence_dispersion_avg": round(avg_dispersion, 6),
        "page_window_stability_avg": round(avg_stability, 6),
        "page_repetition_control_avg": round(avg_repetition, 6),
        "page_content_word_quality_avg": round(avg_content_quality, 6),
        "page_language_coherence_avg": round(avg_coherence, 6),
    }


def nested_mean(rows: list[dict[str, Any]], nested_key: str, feature: str) -> float:
    return mean(
        float((row.get(nested_key) or {}).get(feature) or 0.0)
        for row in rows
    )


def fragment_illusion_penalty(
    *,
    content_word_quality: float,
    language_coherence: float,
    language_shape: float,
    binary_ngram_fit: float,
    evidence_dispersion: float,
    window_stability: float,
    repetition_control: float,
) -> float:
    """Penalize fragment-rich basins unsupported by broader evidence.

    This is ground-truth-free. It targets Copiale-style false positives where
    German-looking word islands and smooth letter shape outrun the slower,
    more global signals: binary n-grams, evidence dispersion, window
    stability, and repetition control.
    """
    fragment_side = mean([content_word_quality, language_coherence, language_shape])
    support_side = mean([binary_ngram_fit, evidence_dispersion, window_stability, repetition_control])
    return max(0.0, min(1.0, fragment_side - support_side))


def mean(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / max(1, len(items))


def stddev(values: Any) -> float:
    items = [float(value) for value in values]
    if len(items) < 2:
        return 0.0
    avg = mean(items)
    return (sum((value - avg) ** 2 for value in items) / len(items)) ** 0.5


def render_markdown(payload: dict[str, Any]) -> str:
    combined = payload["combined"]
    lines = [
        "# Copiale Multi-Page Shared-Key Experiment",
        "",
        "Ground truth is used only after the combined solver run, for calibration.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| pages | {len(payload['pages'])} |",
        f"| tokens | {combined['token_count']} |",
        f"| symbols | {combined['alphabet_size']} |",
        f"| status | {combined['status']} |",
        f"| solver | {combined['solver']} |",
        f"| selected mask | {mask_label(combined['selected_mask'])} |",
        f"| combined char | {float(combined['char_accuracy']):.1%} |",
        f"| combined word | {float(combined['word_accuracy']):.1%} |",
        f"| stable shared letters | {payload['consensus']['stable_letter_count']} |",
        f"| stable shared nulls | {payload['consensus']['stable_null_count']} |",
        f"| disputed symbols | {payload['consensus']['disputed_symbol_count']} |",
        f"| elapsed seconds | {payload['elapsed_seconds']:.1f} |",
        "",
        "## Shared-Key Projection",
        "",
        "| Page | Tokens | Filtered | Char | Word | Preview |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["pages"]:
        lines.append(
            f"| {row['test_id']} | {row['token_count']} | {row['filtered_length']} | "
            f"{float(row['char_accuracy']):.1%} | {float(row['word_accuracy']):.1%} | "
            f"{escape_md(row['preview'])} |"
        )
    elite = payload.get("elite_page_rerank") or {}
    if elite.get("rows"):
        lines.extend([
            "",
            "## Page-Aware Elite Rerank",
            "",
            "Ranking is ground-truth-free. Character columns are post-hoc calibration only.",
            "",
            f"Policy: `{elite.get('rank_policy')}`",
            "",
            "### Policy Sweep",
            "",
            "| Policy | Winner | Mask | Score | Winner Post-Hoc Char | Post-Hoc Best Rank |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row in elite.get("policy_winners") or []:
            lines.append(
                f"| {row['policy']} | {row['winner']} | {mask_label(row['mask'])} | "
                f"{float(row['score']):.3f} | {float(row['post_hoc_char_avg']):.1%} | "
                f"{row.get('rank_of_post_hoc_best') or ''} |"
            )
        portfolio = elite.get("portfolio") or {}
        if portfolio.get("rows"):
            lines.extend([
                "",
                "### Diverse Portfolio",
                "",
                f"Requested size: `{portfolio.get('requested_size')}`; "
                f"captured post-hoc best: `{portfolio.get('captures_post_hoc_best')}`; "
                f"best in portfolio: `{portfolio.get('best_post_hoc_label_in_portfolio')}` "
                f"({float(portfolio.get('best_post_hoc_char_in_portfolio') or 0.0):.1%})",
                "",
                "| Rank | Finalist | Reason | Mask | Robust | Balanced | Post-Hoc Char Avg | Page Chars |",
                "|---:|---|---|---|---:|---:|---:|---|",
            ])
            for idx, row in enumerate(portfolio.get("rows") or [], start=1):
                page_chars = ", ".join(
                    f"{item['test_id'].rsplit('_', 1)[-1]}:{float(item['char_accuracy']):.1%}"
                    for item in row.get("post_hoc_page_chars") or []
                )
                lines.append(
                    f"| {idx} | {row['label']} | {row.get('portfolio_reason', '')} | "
                    f"{mask_label(row['mask'])} | {format_float(row.get('page_robust_score'))} | "
                    f"{float(row['page_balanced_score']):.3f} | "
                    f"{float(row['post_hoc_char_avg']):.1%} | {escape_md(page_chars)} |"
                )
        lines.extend([
            "",
            "### Balanced Ranking",
            "",
            "| Rank | Finalist | Mask | Robust | Balanced | Page Avg | Page Min | LQ Avg | Illusion | Post-Hoc Char Avg | Page Chars |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for idx, row in enumerate((elite.get("rows") or [])[:12], start=1):
            page_chars = ", ".join(
                f"{item['test_id'].rsplit('_', 1)[-1]}:{float(item['char_accuracy']):.1%}"
                for item in row.get("post_hoc_page_chars") or []
            )
            lines.append(
                f"| {idx} | {row['label']} | {mask_label(row['mask'])} | "
                f"{format_float(row.get('page_robust_score'))} | "
                f"{float(row['page_balanced_score']):.3f} | "
                f"{float(row['page_validation_avg']):.3f} | "
                f"{float(row['page_validation_min']):.3f} | "
                f"{float(row['page_language_quality_avg']):.3f} | "
                f"{format_float(row.get('fragment_illusion_penalty'))} | "
                f"{float(row['post_hoc_char_avg']):.1%} | {escape_md(page_chars)} |"
            )
    if payload.get("structured_pages"):
        lines.extend([
            "",
            "## Structured Page Refinement",
            "",
            "Stable shared mappings are pinned. Disputed/page-local symbols remain mutable.",
            "",
            "| Page | Tokens | Filtered | Fixed | Char | Word | Time | Preview |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        projection_by_id = {row["test_id"]: row for row in payload["pages"]}
        for row in payload["structured_pages"]:
            baseline = projection_by_id.get(row["test_id"], {})
            delta = float(row["char_accuracy"]) - float(baseline.get("char_accuracy") or 0.0)
            lines.append(
                f"| {row['test_id']} | {row['token_count']} | {row['filtered_length']} | "
                f"{row.get('fixed_symbol_count', 0)} | {float(row['char_accuracy']):.1%} "
                f"({delta:+.1%}) | {float(row['word_accuracy']):.1%} | "
                f"{float(row.get('elapsed_seconds') or 0.0):.1f} | {escape_md(row['preview'])} |"
            )
    portfolio_refinement = payload.get("portfolio_refinement") or {}
    if portfolio_refinement.get("rows"):
        best_policy = portfolio_refinement.get("best_by_policy") or {}
        best_posthoc = portfolio_refinement.get("best_by_post_hoc_char") or {}
        lines.extend([
            "",
            "## Portfolio Structured Refinement",
            "",
            "Each retained portfolio finalist is refined page-by-page with stable shared mappings pinned.",
            "Ranking is ground-truth-free; character columns are post-hoc calibration only.",
            "",
            f"Best by policy: `{best_policy.get('label')}` "
            f"({float(best_policy.get('post_hoc_char_avg') or 0.0):.1%} post-hoc avg); "
            f"best post-hoc: `{best_posthoc.get('label')}` "
            f"({float(best_posthoc.get('post_hoc_char_avg') or 0.0):.1%}).",
            "",
            "| Rank | Finalist | Source | Mask | Robust | Balanced | Page Avg | LQ Avg | Illusion | Post-Hoc Char Avg | Page Chars |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for idx, row in enumerate((portfolio_refinement.get("rows") or [])[:12], start=1):
            page_chars = ", ".join(
                f"{item['test_id'].rsplit('_', 1)[-1]}:{float(item['char_accuracy']):.1%}"
                for item in row.get("post_hoc_page_chars") or []
            )
            lines.append(
                f"| {idx} | {row['label']} | {row.get('source_portfolio_reason', '')} | "
                f"{mask_label(row['mask'])} | {format_float(row.get('page_robust_score'))} | "
                f"{float(row['page_balanced_score']):.3f} | "
                f"{float(row['page_validation_avg']):.3f} | "
                f"{float(row['page_language_quality_avg']):.3f} | "
                f"{format_float(row.get('fragment_illusion_penalty'))} | "
                f"{float(row['post_hoc_char_avg']):.1%} | {escape_md(page_chars)} |"
            )
    portfolio_local = payload.get("portfolio_local_repair") or {}
    if portfolio_local.get("rows"):
        best_policy = portfolio_local.get("best_by_policy") or {}
        best_posthoc = portfolio_local.get("best_by_post_hoc_char") or {}
        lines.extend([
            "",
            "## Portfolio Local Window Repair",
            "",
            "Each retained portfolio finalist is projected page-by-page, damaged windows are found without ground truth,",
            "and disputed-symbol edits are tried only inside those windows. Character columns are post-hoc calibration only.",
            "",
            f"Best by policy: `{best_policy.get('label')}` "
            f"({float(best_policy.get('post_hoc_char_avg') or 0.0):.1%} post-hoc avg); "
            f"best post-hoc: `{best_posthoc.get('label')}` "
            f"({float(best_posthoc.get('post_hoc_char_avg') or 0.0):.1%}).",
            "",
            "| Rank | Finalist | Source | Mask | Changed Pages | Robust | Balanced | Page Avg | LQ Avg | Illusion | Post-Hoc Char Avg | Page Chars |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for idx, row in enumerate((portfolio_local.get("rows") or [])[:12], start=1):
            page_chars = ", ".join(
                (
                    f"{item['test_id'].rsplit('_', 1)[-1]}:"
                    f"{float(item['char_accuracy']):.1%}"
                    f"[{';'.join(item.get('selected_edits') or [])}]"
                )
                for item in row.get("post_hoc_page_chars") or []
            )
            lines.append(
                f"| {idx} | {row['label']} | {row.get('source_portfolio_reason', '')} | "
                f"{mask_label(row['mask'])} | {int(row.get('changed_page_count') or 0)} | "
                f"{format_float(row.get('page_robust_score'))} | "
                f"{float(row['page_balanced_score']):.3f} | "
                f"{float(row['page_validation_avg']):.3f} | "
                f"{float(row['page_language_quality_avg']):.3f} | "
                f"{format_float(row.get('fragment_illusion_penalty'))} | "
                f"{float(row['post_hoc_char_avg']):.1%} | {escape_md(page_chars)} |"
            )
    lines.extend([
        "",
        "## Scoring-Inside-Search Options",
        "",
        "1. **Batch finalist re-ranking**: keep the fast annealer unchanged, but generate many more finalists and re-rank them with the language-quality scorer before promotion.",
        "2. **Periodic score checkpoints**: every N anneal epochs, score the current plaintext with cheaper language-quality features and keep an elite pool by a blended anneal/readability score.",
        "3. **Local repair objective**: when repairing damaged windows, score the changed window and nearby stable windows directly, instead of relying only on whole-text anneal score.",
        "4. **Two-temperature search**: use Zenith score for broad movement, then switch to a readability-biased objective only after a candidate clears a minimum language threshold.",
        "5. **Slow validator cascade**: run LQ/LLM validation only on a tiny top-N portfolio, never inside the hot loop.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def payload_without_large_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    compact["artifact"] = {
        "run_id": payload.get("artifact", {}).get("run_id"),
        "artifact_file": "see sibling .artifact.json",
    }
    compact["pages"] = [
        {key: value for key, value in row.items() if key != "plaintext"}
        for row in payload["pages"]
    ]
    return compact


def selected_environment_snapshot() -> dict[str, str]:
    prefixes = (
        "DECIPHER_NULL_MASK_",
        "DECIPHER_HOMOPHONIC_",
        "DECIPHER_PARALLEL_",
        "DECIPHER_NGRAM_MODEL_",
    )
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if any(key.startswith(prefix) for prefix in prefixes)
    }


def mask_label(mask: list[str] | tuple[str, ...]) -> str:
    return ",".join(mask) if mask else "(none)"


def escape_md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def resolve_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


if __name__ == "__main__":
    main()
