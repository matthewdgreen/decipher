#!/usr/bin/env python3
"""Run bounded annealing around Copiale repair-variant seeds.

This is the first "real repair" step after a broad null-mask search. It takes
the ground-truth-free repair agenda and local variant probe, freezes mappings
that are not implicated as disputed, applies each promising local edit as an
initial key, and reruns the homophonic solver on the selected null mask.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from automated.runner import (  # noqa: E402
    _plaintext_quality,
    _run_homophonic,
    _word_list,
    _automated_candidate_diagnostics,
)
from benchmark.loader import BenchmarkLoader, parse_canonical_transcription  # noqa: E402
from models.cipher_text import CipherText  # noqa: E402
from probe_copiale_repair_variants import (  # noqa: E402
    apply_assignment,
    load_selected_row,
)
from report_copiale_repair_agenda import parse_key  # noqa: E402
from analysis.homophonic_nulls import null_mask_validation_score_v2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run targeted Copiale repair anneals from repair-variant seeds."
    )
    parser.add_argument("repair_agenda_json")
    parser.add_argument("repair_variants_json")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument("--top-variants", type=int, default=5)
    parser.add_argument(
        "--require-edit-substring",
        default="",
        help="Only run variants whose edit labels contain this substring, e.g. '<null>'.",
    )
    parser.add_argument("--budget", choices=["screen", "full"], default="screen")
    parser.add_argument("--workers", type=int, default=1, help="DECIPHER_PARALLEL_WORKERS during each seed run.")
    parser.add_argument(
        "--pin-edits",
        action="store_true",
        help="Also freeze symbols changed by each variant so the anneal tests the proposed repair.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    agenda_path = resolve_path(Path(args.repair_agenda_json))
    variants_path = resolve_path(Path(args.repair_variants_json))
    agenda = json.loads(agenda_path.read_text(encoding="utf-8"))
    variants_payload = json.loads(variants_path.read_text(encoding="utf-8"))
    test_id = str(agenda.get("test_id") or variants_payload.get("test_id") or "")
    if not test_id:
        raise SystemExit("Missing test_id in repair inputs.")

    cipher = load_cipher(resolve_path(Path(args.benchmark_root)), args.split, test_id)
    selected_row, selected_artifact = load_selected_row(agenda)
    baseline_key = parse_key(selected_row.get("key"))
    baseline_mask = tuple(str(symbol) for symbol in (selected_row.get("mask") or []))
    mutable_symbols = disputed_symbols(agenda)
    stable_fixed_ids = fixed_symbol_ids(
        cipher,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        mutable_symbols=mutable_symbols,
    )
    candidate_variants = [
        row for row in (variants_payload.get("top_variants") or [])
        if isinstance(row, dict)
        and edit_substring_matches(row, args.require_edit_substring)
    ][: max(1, args.top_variants)]
    baseline_variant = {
        "edits": ["baseline"],
        "validation_score_v2": None,
        "preview": str(selected_row.get("decryption") or selected_row.get("preview") or "")[:160],
    }
    rows = [baseline_variant]
    rows.extend(
        row for row in candidate_variants
        if row.get("edits") != ["baseline"]
    )

    started = time.time()
    previous_workers = os.environ.get("DECIPHER_PARALLEL_WORKERS")
    os.environ["DECIPHER_PARALLEL_WORKERS"] = str(max(1, args.workers))
    try:
        results = [
            run_seed_variant(
                index=index,
                row=row,
                cipher=cipher,
                baseline_key=baseline_key,
                baseline_mask=baseline_mask,
                stable_fixed_ids=stable_fixed_ids,
                budget=args.budget,
                pin_edits=args.pin_edits,
            )
            for index, row in enumerate(rows, start=1)
        ]
    finally:
        if previous_workers is None:
            os.environ.pop("DECIPHER_PARALLEL_WORKERS", None)
        else:
            os.environ["DECIPHER_PARALLEL_WORKERS"] = previous_workers

    results.sort(
        key=lambda row: (
            float(row.get("validation_score_v2") or float("-inf")),
            float(row.get("anneal_score") or float("-inf")),
        ),
        reverse=True,
    )
    attach_result_deltas(results)
    payload = {
        "test_id": test_id,
        "repair_agenda": str(agenda_path),
        "repair_variants": str(variants_path),
        "selected_artifact": str(selected_artifact),
        "budget": args.budget,
        "workers": max(1, args.workers),
        "pin_edits": bool(args.pin_edits),
        "require_edit_substring": args.require_edit_substring,
        "baseline_mask": list(baseline_mask),
        "mutable_symbols": sorted(mutable_symbols),
        "fixed_symbol_count": len(stable_fixed_ids),
        "elapsed_seconds": round(time.time() - started, 3),
        "results": results,
    }
    markdown = render_markdown(payload)
    output = resolve_path(Path(args.output)) if args.output else variants_path.with_suffix(".targeted_repair.md")
    json_output = (
        resolve_path(Path(args.json_output))
        if args.json_output
        else output.with_suffix(".json")
    )
    output.write_text(markdown, encoding="utf-8")
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown)
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")


def run_seed_variant(
    *,
    index: int,
    row: dict[str, Any],
    cipher: CipherText,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    stable_fixed_ids: set[int],
    budget: str,
    pin_edits: bool,
) -> dict[str, Any]:
    mask = set(baseline_mask)
    key = dict(baseline_key)
    edited_ids: set[int] = set()
    edits = [str(edit) for edit in (row.get("edits") or ["baseline"])]
    for edit in edits:
        if edit == "baseline":
            continue
        symbol, _before, after = parse_edit(edit)
        if not cipher.alphabet.has_symbol(symbol):
            continue
        token_id = cipher.alphabet.id_for(symbol)
        apply_assignment(symbol, token_id, after, key, mask)
        edited_ids.add(token_id)

    filtered_tokens = [
        token for token in cipher.tokens
        if cipher.alphabet.symbol_for(token) not in mask
    ]
    filtered_cipher = CipherText(
        raw=cipher.alphabet.decode(filtered_tokens),
        alphabet=cipher.alphabet,
        source=f"{cipher.source}:targeted_repair:{index}",
        separator=None,
    )
    fixed_ids = set(stable_fixed_ids) - edited_ids
    if pin_edits:
        fixed_ids |= edited_ids
    fixed_ids = {
        token_id for token_id in fixed_ids
        if cipher.alphabet.symbol_for(token_id) not in mask
    }
    solver, repaired_key, plaintext, step = _run_homophonic(
        filtered_cipher,
        language="de",
        budget=budget,
        solver_profile="zenith_native",
        initial_key=key,
        fixed_cipher_ids=fixed_ids,
    )
    quality = _plaintext_quality(plaintext, repaired_key)
    diagnostics = _automated_candidate_diagnostics(
        plaintext,
        language="de",
        word_list=_word_list("de"),
    )
    validation = null_mask_validation_score_v2(
        {
            "mask": list(sorted(mask)),
            "filtered_length": len(filtered_tokens),
            "selection_score": step.get("selection_score") or step.get("anneal_score") or 0.0,
            "quality": quality,
            "diagnostics": diagnostics,
            "decryption": plaintext,
            "preview": plaintext,
        },
        original_length=len(cipher.tokens),
        language="de",
    )
    return {
        "rank_input": index,
        "edits": edits,
        "mask": sorted(mask),
        "solver": solver,
        "fixed_symbol_count": len(fixed_ids),
        "edited_symbol_count": len(edited_ids),
        "pin_edits": bool(pin_edits),
        "anneal_score": step.get("anneal_score"),
        "selection_score": step.get("selection_score"),
        "validation_score_v2": validation["score"],
        "validation_score_v2_no_selection": round(
            sum(
                float(value)
                for name, value in validation["components"].items()
                if name != "selection"
            ),
            6,
        ),
        "validation_components_v2": validation["components"],
        "quality": quality,
        "diagnostics": diagnostics,
        "preview": plaintext[:220],
        "decryption": plaintext,
        "key": repaired_key,
        "step": step,
    }


def parse_edit(edit: str) -> tuple[str, str, str]:
    symbol, rest = edit.split(":", 1)
    before, after = rest.split("->", 1)
    return symbol, before, after


def edit_substring_matches(row: dict[str, Any], substring: str) -> bool:
    if not substring:
        return True
    return any(substring in str(edit) for edit in (row.get("edits") or []))


def attach_result_deltas(results: list[dict[str, Any]]) -> None:
    baseline = next(
        (row for row in results if row.get("edits") == ["baseline"]),
        results[0] if results else None,
    )
    if baseline is None:
        return
    base_val = float(baseline.get("validation_score_v2") or 0.0)
    base_readability = float(baseline.get("validation_score_v2_no_selection") or 0.0)
    base_anneal = float(baseline.get("anneal_score") or 0.0)
    for row in results:
        row["delta_vs_baseline"] = {
            "validation_score_v2": round(float(row.get("validation_score_v2") or 0.0) - base_val, 6),
            "validation_score_v2_no_selection": round(
                float(row.get("validation_score_v2_no_selection") or 0.0) - base_readability,
                6,
            ),
            "anneal_score": round(float(row.get("anneal_score") or 0.0) - base_anneal, 6),
        }


def disputed_symbols(agenda: dict[str, Any]) -> set[str]:
    symbols = {
        str(item.get("symbol"))
        for item in (agenda.get("most_disputed_symbols") or [])
        if item.get("symbol")
    }
    for window in agenda.get("repair_windows") or []:
        for item in window.get("disputed_symbols") or []:
            if item.get("symbol"):
                symbols.add(str(item["symbol"]))
    return symbols


def fixed_symbol_ids(
    cipher: CipherText,
    *,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    mutable_symbols: set[str],
) -> set[int]:
    masked = set(baseline_mask)
    fixed = set()
    for token_id in baseline_key:
        symbol = cipher.alphabet.symbol_for(token_id)
        if symbol in masked or symbol in mutable_symbols:
            continue
        fixed.add(token_id)
    return fixed


def load_cipher(benchmark_root: Path, split: str, test_id: str) -> CipherText:
    loader = BenchmarkLoader(benchmark_root)
    tests = [test for test in loader.load_tests(split) if test.test_id == test_id]
    if not tests:
        raise SystemExit(f"Test not found in split {split}: {test_id}")
    data = loader.load_test_data(tests[0])
    return parse_canonical_transcription(data.canonical_transcription)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Copiale Targeted Repair: {payload['test_id']}",
        "",
        "Ground truth is not used. Each row starts from a local repair variant,",
        "freezes stable consensus mappings, and reruns a bounded homophonic anneal.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| budget | {payload['budget']} |",
        f"| variants run | {len(payload['results'])} |",
        f"| pin edits | {payload['pin_edits']} |",
        f"| edit filter | {payload['require_edit_substring'] or '(none)'} |",
        f"| fixed symbols | {payload['fixed_symbol_count']} |",
        f"| mutable symbols | {len(payload['mutable_symbols'])} |",
        f"| elapsed seconds | {payload['elapsed_seconds']:.1f} |",
        "",
        "## Results",
        "",
        "| Rank | Edits | Val2 | ΔVal2 | Readability | ΔRead | Anneal | Dict | Top Letter | Fixed | Preview |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(payload["results"], start=1):
        lines.append(
            f"| {rank} | {escape_md('; '.join(row['edits']))} | "
            f"{float(row['validation_score_v2']):.3f} | "
            f"{format_optional((row.get('delta_vs_baseline') or {}).get('validation_score_v2'))} | "
            f"{format_optional(row.get('validation_score_v2_no_selection'))} | "
            f"{format_optional((row.get('delta_vs_baseline') or {}).get('validation_score_v2_no_selection'))} | "
            f"{format_optional(row.get('anneal_score'))} | "
            f"{format_optional((row.get('diagnostics') or {}).get('dict_rate'))} | "
            f"{format_optional((row.get('quality') or {}).get('top_letter_fraction'))} | "
            f"{row.get('fixed_symbol_count', 0)} | {escape_md(row.get('preview', ''))} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def escape_md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def resolve_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


if __name__ == "__main__":
    main()
