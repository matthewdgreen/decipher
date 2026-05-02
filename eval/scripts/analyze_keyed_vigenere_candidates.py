#!/usr/bin/env python3
"""Summarize keyed-Vigenere/tableau candidate-capture artifacts.

This is the readout companion to capture_keyed_vigenere_candidates.py.  It
does not run search and does not call any LLM/API; it only ranks and summarizes
an existing JSON artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _log(message: str, *, enabled: bool, started: float) -> None:
    if not enabled:
        return
    elapsed = time.time() - started
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp} +{elapsed:7.1f}s] {message}", file=sys.stderr, flush=True)


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _candidate_line(candidate: dict[str, Any]) -> str:
    labels = candidate.get("labels") or {}
    params = candidate.get("parameters") or {}
    source = params.get("initial_alphabet_source") or params.get("initial_candidate_type")
    control = params.get("initial_control_type")
    if control:
        source = f"{control}:{params.get('initial_control_swap_count')}"
    return (
        f"rank={candidate.get('global_rank', 'n/a'):>4} "
        f"score={_fmt_float(candidate.get('score'), 5):>9} "
        f"char={_fmt_float(labels.get('char_accuracy')):>5} "
        f"hamming={str(labels.get('tableau_hamming')):>4} "
        f"key={str(candidate.get('key')):<14} "
        f"seed={str(candidate.get('seed')):<4} "
        f"start={str(source):<28} "
        f"preview={str(candidate.get('preview') or '')[:72]}"
    )


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _top_by(
    candidates: list[dict[str, Any]],
    key_name: str,
    *,
    reverse: bool,
    n: int,
    progress_every: int = 0,
    verbose: bool = False,
    started: float = 0.0,
) -> list[dict[str, Any]]:
    def _key(candidate: dict[str, Any]) -> tuple[bool, float]:
        labels = candidate.get("labels") or {}
        value = candidate.get(key_name)
        if value is None:
            value = labels.get(key_name)
        if value is None:
            return (False, 0.0)
        return (True, float(value))

    if progress_every > 0 and len(candidates) >= progress_every:
        _log(f"ranking {len(candidates)} candidates by {key_name}", enabled=verbose, started=started)
    ranked = sorted(candidates, key=_key, reverse=reverse)
    return ranked[:n]


def _diagnosis(
    payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    progress_every: int = 0,
    verbose: bool = False,
    started: float = 0.0,
) -> list[str]:
    lines: list[str] = []
    if not candidates:
        return ["No candidates were captured."]

    summary = payload.get("summary") or {}
    max_char = summary.get("max_char_accuracy")
    min_hamming = summary.get("min_tableau_hamming")
    exact_tableau_count = summary.get("exact_tableau_count") or 0
    exact_key_count = summary.get("exact_key_count") or 0

    if exact_tableau_count or exact_key_count:
        lines.append(
            "Generator reached an exact solution-bearing basin. If rank is poor, this is a ranking/scoring problem."
        )
    elif max_char is not None and max_char >= 0.85:
        lines.append(
            "Generator found a high-accuracy plaintext basin without exact tableau/key labels. Inspect top-by-char candidates; reranking may be enough."
        )
    elif min_hamming is not None and min_hamming <= 6:
        lines.append(
            "Generator got near the true tableau but did not decode cleanly. This points to refinement/reranking plus better shift coupling."
        )
    else:
        lines.append(
            "No near-true tableau or high-readability basin appears in the captured set. This points to proposal/search architecture, not just scoring."
        )

    scored = []
    chars = []
    for idx, candidate in enumerate(candidates, start=1):
        if progress_every > 0 and idx % progress_every == 0:
            _log(f"correlation scan {idx}/{len(candidates)}", enabled=verbose, started=started)
        score = candidate.get("score")
        char = (candidate.get("labels") or {}).get("char_accuracy")
        if score is not None and char is not None:
            scored.append(float(score))
            chars.append(float(char))
    corr = _pearson(scored, chars)
    if corr is not None:
        if corr < 0.2:
            lines.append(f"Score/char correlation is weak ({corr:.3f}); current fast score is probably not a sufficient ranker.")
        else:
            lines.append(f"Score/char correlation is {corr:.3f}; current fast score has some useful signal.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a keyed-Vigenere candidate-capture artifact.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Emit compact machine-readable summary.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print timestamped progress to stderr.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        metavar="N",
        help="Progress interval for candidate scans when --verbose is set (default: 10000).",
    )
    args = parser.parse_args()
    started = time.time()

    file_size = args.artifact.stat().st_size if args.artifact.exists() else 0
    _log(
        f"loading {args.artifact} ({file_size / (1024 * 1024):.1f} MiB)",
        enabled=args.verbose,
        started=started,
    )
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") or []
    summary = payload.get("summary") or {}
    _log(f"loaded {len(candidates)} candidates", enabled=args.verbose, started=started)

    if args.json:
        _log("building JSON summary", enabled=args.verbose, started=started)
        print(json.dumps({
            "artifact": str(args.artifact),
            "test_id": payload.get("test_id"),
            "leakage_mode": payload.get("leakage_mode"),
            "raw_candidate_count": payload.get("raw_candidate_count"),
            "deduped_candidate_count": payload.get("deduped_candidate_count"),
            "summary": summary,
            "diagnosis": _diagnosis(
                payload,
                candidates,
                progress_every=args.progress_every if args.verbose else 0,
                verbose=args.verbose,
                started=started,
            ),
        }, indent=2))
        _log("done", enabled=args.verbose, started=started)
        return

    print(f"Artifact: {args.artifact}")
    print(f"Test: {payload.get('test_id')}  leakage={payload.get('leakage_mode')}  schema={payload.get('schema')}")
    print(
        f"Candidates: raw={payload.get('raw_candidate_count')} "
        f"deduped={payload.get('deduped_candidate_count')} "
        f"elapsed={payload.get('elapsed_seconds')}s"
    )
    config = payload.get("config") or {}
    print(
        "Config: "
        f"seeds={len(config.get('seeds') or [])} "
        f"steps={config.get('steps')} "
        f"restarts={config.get('restarts')} "
        f"max_period={config.get('max_period')} "
        f"guided={config.get('guided')} "
        f"pool={config.get('guided_pool_size')}"
    )

    print("\nDiagnosis:")
    for line in _diagnosis(
        payload,
        candidates,
        progress_every=args.progress_every if args.verbose else 0,
        verbose=args.verbose,
        started=started,
    ):
        print(f"- {line}")

    print("\nTop by original score:")
    for candidate in candidates[: args.top]:
        print("  " + _candidate_line(candidate))

    print("\nTop by char accuracy:")
    for candidate in _top_by(
        candidates,
        "char_accuracy",
        reverse=True,
        n=args.top,
        progress_every=args.progress_every if args.verbose else 0,
        verbose=args.verbose,
        started=started,
    ):
        print("  " + _candidate_line(candidate))

    _log("checking for tableau hamming labels", enabled=args.verbose, started=started)
    if any((candidate.get("labels") or {}).get("tableau_hamming") is not None for candidate in candidates):
        print("\nClosest by tableau hamming:")
        for candidate in _top_by(
            candidates,
            "tableau_hamming",
            reverse=False,
            n=args.top,
            progress_every=args.progress_every if args.verbose else 0,
            verbose=args.verbose,
            started=started,
        ):
            print("  " + _candidate_line(candidate))
    _log("done", enabled=args.verbose, started=started)


if __name__ == "__main__":
    main()
