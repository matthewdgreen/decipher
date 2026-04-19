#!/usr/bin/env python3
"""Run a suite of four synthetic tests at varying difficulty and report results.

Tests that fall below 100% character accuracy are copied to the errata/
directory along with a verbose report and the full run artifact.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/run_testgen_suite.py [options]

Options:
    --model MODEL          Cracking model (default: claude-sonnet-4-6)
    --max-iterations N     Agent iterations per test (default: 20)
    --cache-dir DIR        Plaintext cache directory (default: testgen_cache)
    --artifact-dir DIR     Artifact directory (default: artifacts)
    --errata-dir DIR       Errata directory (default: errata)
    --flush-cache          Regenerate all plaintexts before running
    --verbose              Show agent reasoning while running
    --list-errata          List active errata and exit
    --rerun-errata         Re-run all active errata tests
    --rerun TEST_ID [...]  Re-run specific errata test(s) by ID

Re-run mode always records results (even 100%) so history accumulates.
If the last two recorded runs for a test are both 100%, the errata entry
is automatically archived to errata/archive/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Make src/ importable when run from the project root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.runner_v2 import BenchmarkRunnerV2
from benchmark.scorer import (
    format_alignment,
    format_char_diff,
    has_word_boundaries,
    score_decryption,
)
from services.claude_api import ClaudeAPI
from testgen.builder import build_test_case
from testgen.cache import PlaintextCache
from testgen.spec import DifficultyPreset, TestSpec, _PRESET_PARAMS


# ---------------------------------------------------------------------------
# Suite definition
# ---------------------------------------------------------------------------

# Fixed seeds make the cipher key deterministic, so the exact same ciphertext
# is produced on every run — enabling apples-to-apples regression comparisons.
SUITE: list[TestSpec] = [
    TestSpec.from_preset(DifficultyPreset.TINY,    language="en", seed=1),
    TestSpec.from_preset(DifficultyPreset.MEDIUM,  language="en", seed=3),
    TestSpec.from_preset(DifficultyPreset.HARD,    language="en", seed=4),
    TestSpec.from_preset(DifficultyPreset.HARDEST, language="en", seed=5),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SuiteResult:
    spec: TestSpec
    test_id: str
    status: str
    char_accuracy: float
    word_accuracy: float
    self_confidence: float | None
    iterations: int
    elapsed: float
    decryption: str
    ground_truth: str
    artifact_path: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Errata index helpers
# ---------------------------------------------------------------------------

def _list_active_errata(errata_dir: Path) -> list[str]:
    """Return sorted test IDs that have active (non-archived) errata."""
    if not errata_dir.exists():
        return []
    return sorted(
        d.name for d in errata_dir.iterdir()
        if d.is_dir() and d.name != "archive"
    )


def _load_spec_from_errata(test_id: str, errata_dir: Path) -> TestSpec:
    """Reconstruct TestSpec from the most recent errata run_info.json.

    Seed is parsed from the deterministic test_id format:
        synth_{lang}_{length}{wb|nb}_s{seed}
    """
    import re
    test_dir = errata_dir / test_id
    run_dirs = sorted(r for r in test_dir.iterdir() if r.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run records found in {test_dir}")
    info = json.loads((run_dirs[-1] / "run_info.json").read_text(encoding="utf-8"))
    s = info["spec"]

    # Prefer explicitly stored seed; fall back to parsing the test_id.
    seed = s.get("seed")
    if seed is None:
        m = re.search(r"_s(\d+)$", test_id)
        if m:
            seed = int(m.group(1))

    return TestSpec(
        language=s["language"],
        approx_length=s["approx_length"],
        word_boundaries=s["word_boundaries"],
        homophonic=s.get("homophonic", False),
        topic=s.get("topic", "general"),
        seed=seed,
    )


def _check_and_maybe_archive(test_id: str, errata_dir: Path) -> bool:
    """Archive errata/{test_id}/ if the last two recorded runs are both 100%.

    Returns True if the entry was archived.
    """
    test_dir = errata_dir / test_id
    if not test_dir.exists():
        return False
    run_dirs = sorted(r for r in test_dir.iterdir() if r.is_dir())
    if len(run_dirs) < 2:
        return False
    for run_dir in run_dirs[-2:]:
        info_file = run_dir / "run_info.json"
        if not info_file.exists():
            return False
        info = json.loads(info_file.read_text(encoding="utf-8"))
        if info.get("char_accuracy", 0.0) < 1.0:
            return False
    # Both last two runs are 100% — move to archive.
    archive_target = errata_dir / "archive" / test_id
    archive_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(test_dir), str(archive_target))
    return True


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_info() -> dict:
    """Return current commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip())
        description = subprocess.check_output(
            ["git", "describe", "--always", "--dirty"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return {"commit": commit, "dirty": dirty, "description": description}
    except Exception:  # noqa: BLE001
        return {"commit": "unknown", "dirty": False, "description": "unknown"}


# ---------------------------------------------------------------------------
# Errata
# ---------------------------------------------------------------------------

def _save_errata(
    sr: SuiteResult,
    git: dict,
    model: str,
    errata_dir: Path,
) -> Path:
    """Save errata for a non-perfect test. Returns the errata directory path."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    # Timestamped subdirectory so multiple runs of the same test accumulate history.
    ts_slug = timestamp.replace(":", "").replace("-", "")  # 20260419T120000Z
    out = errata_dir / sr.test_id / ts_slug
    out.mkdir(parents=True, exist_ok=True)

    # --- run_info.json ---
    run_info = {
        "test_id": sr.test_id,
        "timestamp": timestamp,
        "git_commit": git["commit"],
        "git_description": git["description"],
        "git_dirty": git["dirty"],
        "model": model,
        "spec": {
            "language": sr.spec.language,
            "approx_length": sr.spec.approx_length,
            "word_boundaries": sr.spec.word_boundaries,
            "homophonic": sr.spec.homophonic,
            "topic": sr.spec.topic,
            "seed": sr.spec.seed,
        },
        "status": sr.status,
        "char_accuracy": sr.char_accuracy,
        "word_accuracy": sr.word_accuracy if has_word_boundaries(sr.ground_truth) else None,
        "self_confidence": sr.self_confidence,
        "iterations_used": sr.iterations,
        "elapsed_seconds": round(sr.elapsed, 1),
        "error": sr.error or None,
    }
    (out / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- artifact.json (copy) ---
    if sr.artifact_path and Path(sr.artifact_path).exists():
        shutil.copy2(sr.artifact_path, out / "artifact.json")

    # --- report.txt ---
    report = _build_report_text(sr, run_info)
    (out / "report.txt").write_text(report, encoding="utf-8")

    return out


def _build_report_text(sr: SuiteResult, run_info: dict) -> str:
    lines: list[str] = []
    sep = "=" * 80

    # Header
    lines += [
        sep,
        f"ERRATA REPORT — {sr.test_id}",
        sep,
        f"Timestamp:   {run_info['timestamp']}",
        f"Git commit:  {run_info['git_description']}"
        + (" (UNCOMMITTED CHANGES)" if run_info["git_dirty"] else ""),
        f"Model:       {run_info['model']}",
        f"Language:    {sr.spec.language}",
        f"Preset:      ~{sr.spec.approx_length} words, "
        f"{'word-boundary' if sr.spec.word_boundaries else 'no-boundary'}",
        "",
        "SCORES",
        "-" * 40,
        f"Status:      {sr.status}",
        f"Char acc:    {sr.char_accuracy:.1%}",
    ]
    if has_word_boundaries(sr.ground_truth):
        lines.append(f"Word acc:    {sr.word_accuracy:.1%}")
    else:
        lines.append("Word acc:    N/A (no word boundaries)")
    conf = f"{sr.self_confidence:.2f}" if sr.self_confidence is not None else "n/a"
    lines += [
        f"Confidence:  {conf}",
        f"Iterations:  {sr.iterations}",
        f"Elapsed:     {sr.elapsed:.1f}s",
        "",
    ]

    # Ground truth vs decryption diff
    lines += [sep, "GROUND TRUTH vs DECRYPTION", sep]
    if sr.error:
        lines.append(f"Error: {sr.error}")
    elif not sr.decryption:
        lines.append("(no decryption produced)")
    elif has_word_boundaries(sr.ground_truth):
        lines.append(format_alignment(sr.decryption, sr.ground_truth, max_words=200))
    else:
        lines.append(format_char_diff(sr.decryption, sr.ground_truth, context=15))
    lines.append("")

    # Agent reasoning log (from artifact if available)
    artifact_path = Path(sr.artifact_path) if sr.artifact_path else None
    if artifact_path and artifact_path.exists():
        lines += [sep, "AGENT REASONING LOG", sep]
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            lines += _format_agent_log(artifact)
        except Exception as e:  # noqa: BLE001
            lines.append(f"(could not read artifact: {e})")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines) + "\n"


def _format_agent_log(artifact: dict) -> list[str]:
    """Extract per-iteration agent text + tool calls from a raw artifact dict."""
    lines: list[str] = []

    # Group tool calls by iteration
    tc_by_iter: dict[int, list[dict]] = {}
    for tc in artifact.get("tool_calls", []):
        it = tc.get("iteration", 0)
        tc_by_iter.setdefault(it, []).append(tc)

    # Walk messages to find agent text blocks, correlate with tool calls
    current_iter = 0
    for msg in artifact.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        current_iter += 1
        text_blocks = [
            c.get("text", "")
            for c in msg.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        tool_uses = [
            c for c in msg.get("content", [])
            if isinstance(c, dict) and c.get("type") == "tool_use"
        ]

        lines.append(f"--- Iteration {current_iter} ---")

        if text_blocks:
            for text in text_blocks:
                # Wrap long lines at 78 chars for readability
                for para in text.strip().split("\n"):
                    lines.append(f"  {para}")
            lines.append("")

        for tu in tool_uses:
            name = tu.get("name", "?")
            inp = tu.get("input", {})
            # Show key args concisely
            arg_parts = []
            for k, v in inp.items():
                sv = str(v)
                arg_parts.append(f"{k}={sv[:40]!r}" if len(sv) > 40 else f"{k}={sv!r}")
            args_str = ", ".join(arg_parts)

            # Find matching result from tool_calls log
            result_preview = ""
            for tc in tc_by_iter.get(current_iter, []):
                if tc.get("tool_name") == name:
                    r = tc.get("result", "")
                    result_preview = r[:120] + ("..." if len(r) > 120 else "")
                    break

            lines.append(f"  → {name}({args_str})")
            if result_preview:
                lines.append(f"       {result_preview}")
        lines.append("")

    # Branch scores summary
    branches = artifact.get("branches", [])
    if branches:
        lines.append("--- Branch scores ---")
        for b in branches:
            char = b.get("char_accuracy")
            char_str = f"{char:.1%}" if char is not None else "n/a"
            lines.append(
                f"  {b['name']:<20} mapped={b['mapped_count']:>2}  char={char_str}"
                + ("  ← declared" if artifact.get("solution") and
                   artifact["solution"].get("branch") == b["name"] else "")
            )

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_suite(args: argparse.Namespace) -> None:
    errata_dir = Path(args.errata_dir)

    # --list-errata: print active errata and exit.
    if args.list_errata:
        active = _list_active_errata(errata_dir)
        if not active:
            print("No active errata.")
        else:
            print(f"Active errata ({len(active)}):")
            for test_id in active:
                run_dirs = sorted((errata_dir / test_id).glob("*/run_info.json"))
                last = json.loads(run_dirs[-1].read_text(encoding="utf-8")) if run_dirs else {}
                char = last.get("char_accuracy")
                char_str = f"{char:.1%}" if char is not None else "?"
                ts = last.get("timestamp", "?")[:19]
                runs_count = len(list((errata_dir / test_id).iterdir()))
                print(f"  {test_id:<35} last={char_str}  runs={runs_count}  ts={ts}")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY") or _keychain_key()
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY or configure via the GUI.", file=sys.stderr)
        sys.exit(1)

    git = _git_info()
    print(f"Decipher commit: {git['description']}"
          + (" (dirty)" if git["dirty"] else ""))
    print(f"Model: {args.model}  max_iter={args.max_iterations}")
    print()

    cache = PlaintextCache(args.cache_dir)
    if args.flush_cache:
        n = cache.flush()
        print(f"Flushed {n} cache entries.\n")

    # Determine which specs to run and whether we're in rerun mode.
    rerun_mode = bool(args.rerun or args.rerun_errata)
    if args.rerun:
        suite = [_load_spec_from_errata(tid, errata_dir) for tid in args.rerun]
        print(f"Re-running {len(suite)} errata test(s): {', '.join(args.rerun)}\n")
    elif args.rerun_errata:
        test_ids = _list_active_errata(errata_dir)
        if not test_ids:
            print("No active errata to re-run.")
            return
        suite = [_load_spec_from_errata(tid, errata_dir) for tid in test_ids]
        print(f"Re-running all {len(suite)} active errata: {', '.join(test_ids)}\n")
    else:
        suite = list(SUITE)

    crack_api = ClaudeAPI(api_key=api_key, model=args.model)
    runner = BenchmarkRunnerV2(
        claude_api=crack_api,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        artifact_dir=args.artifact_dir,
    )

    results: list[SuiteResult] = []

    for i, spec in enumerate(suite, 1):
        preset_name = _preset_name(spec)
        boundary_label = "word-boundary" if spec.word_boundaries else "no-boundary"
        print(f"[{i}/{len(suite)}] {preset_name} — {spec.language}, {boundary_label}, "
              f"~{spec.approx_length} words")

        test_data = build_test_case(spec, cache, api_key)
        t0 = time.time()
        result = runner.run_test(test_data, language=spec.language)
        elapsed = time.time() - t0

        score = score_decryption(
            test_id=result.test_id,
            decrypted=result.final_decryption,
            ground_truth=test_data.plaintext,
            agent_score=0.0,
            status=result.status,
        )

        sr = SuiteResult(
            spec=spec,
            test_id=result.test_id,
            status=result.status,
            char_accuracy=score.char_accuracy,
            word_accuracy=score.word_accuracy,
            self_confidence=result.self_confidence,
            iterations=result.iterations_used,
            elapsed=elapsed,
            decryption=result.final_decryption,
            ground_truth=test_data.plaintext,
            artifact_path=result.artifact_path,
            error=result.error_message or "",
        )
        results.append(sr)

        conf_str = f"{sr.self_confidence:.2f}" if sr.self_confidence is not None else "n/a"
        word_str = f"{sr.word_accuracy:.1%}" if score.total_words > 1 else "N/A"
        print(f"  → {sr.status}  char={sr.char_accuracy:.1%}  word={word_str}  "
              f"conf={conf_str}  iter={sr.iterations}  {sr.elapsed:.0f}s")

        if rerun_mode:
            # Always record the run so history accumulates, then check for archive.
            errata_path = _save_errata(sr, git, args.model, errata_dir)
            print(f"  ↳ run recorded → {errata_path}")
            if _check_and_maybe_archive(sr.test_id, errata_dir):
                print(f"  ↳ solved 2× in a row — archived to errata/archive/{sr.test_id}/")
        elif sr.char_accuracy < 1.0:
            errata_path = _save_errata(sr, git, args.model, errata_dir)
            print(f"  ↳ errata saved → {errata_path}")
        print()

    _print_report(results)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(results: list[SuiteResult]) -> None:
    sep = "=" * 80
    print(sep)
    print("SUITE RESULTS")
    print(sep)
    print()

    print(f"  {'Preset':<10} {'Lang':>4} {'Bounds':<14} {'Status':<10} "
          f"{'Char%':>6} {'Word%':>6} {'Conf':>5} {'Iter':>4} {'Time':>6}")
    print("  " + "-" * 70)
    for sr in results:
        preset_name = _preset_name(sr.spec)
        boundary_label = "word-bound" if sr.spec.word_boundaries else "no-bound"
        conf_str = f"{sr.self_confidence:.2f}" if sr.self_confidence is not None else "n/a"
        word_str = f"{sr.word_accuracy:>5.1%}" if has_word_boundaries(sr.ground_truth) else "  N/A"
        print(f"  {preset_name:<10} {sr.spec.language:>4} {boundary_label:<14} {sr.status:<10} "
              f"{sr.char_accuracy:>5.1%} {word_str} {conf_str:>5} {sr.iterations:>4} {sr.elapsed:>5.0f}s")

    solved = sum(1 for r in results if r.status == "solved")
    avg_char = sum(r.char_accuracy for r in results) / len(results)
    print("  " + "-" * 70)
    print(f"  {'AVERAGE':<10} {'':>4} {'':14} {f'{solved}/{len(results)} solved':<10} "
          f"{avg_char:>5.1%}")
    print()

    for sr in results:
        preset_name = _preset_name(sr.spec)
        print(sep)
        print(f"  {preset_name}  [{sr.test_id}]")
        print(sep)

        if sr.error:
            print(f"  Error: {sr.error}")
            continue
        if not sr.decryption:
            print("  (no decryption produced)")
            continue
        if has_word_boundaries(sr.ground_truth):
            print(format_alignment(sr.decryption, sr.ground_truth, max_words=40))
        else:
            print(format_char_diff(sr.decryption, sr.ground_truth, context=12))
        print()

    print(sep)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preset_name(spec: TestSpec) -> str:
    for preset in DifficultyPreset:
        p = _PRESET_PARAMS[preset]
        if (p["approx_length"] == spec.approx_length
                and p["word_boundaries"] == spec.word_boundaries
                and p.get("homophonic", False) == spec.homophonic):
            return preset.value
    return f"custom-{spec.approx_length}"


def _keychain_key() -> str | None:
    try:
        import keyring
        return keyring.get_password("decipher", "anthropic_api_key")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run four synthetic cipher tests at varying difficulty and report results.",
    )
    parser.add_argument("--model", "-m", default="claude-sonnet-4-6")
    parser.add_argument("--max-iterations", "-i", type=int, default=20)
    parser.add_argument("--cache-dir", default="testgen_cache")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--errata-dir", default="errata")
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--list-errata", action="store_true",
                        help="List active errata and exit")
    parser.add_argument("--rerun-errata", action="store_true",
                        help="Re-run all active errata tests")
    parser.add_argument("--rerun", nargs="+", metavar="TEST_ID",
                        help="Re-run specific errata test(s) by ID")
    run_suite(parser.parse_args())


if __name__ == "__main__":
    main()
