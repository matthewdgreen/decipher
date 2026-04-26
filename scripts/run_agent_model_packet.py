#!/usr/bin/env python3
"""Run a small live-agent packet across multiple LLM providers/models."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = REPO_ROOT / "frontier" / "agent_model_eval_0109v.jsonl"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "agent_model_eval"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run agentic benchmark cases across a model packet.",
    )
    parser.add_argument("--packet-file", default=str(DEFAULT_PACKET))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--summary-jsonl")
    parser.add_argument("--summary-csv")
    parser.add_argument("--benchmark-path", help="Override benchmark path for every row.")
    parser.add_argument("--max-iterations", type=int, help="Override max iterations for every row.")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="Wall-clock timeout per model row.")
    parser.add_argument("--only", action="append", default=[], help="Run only packet names matching this value.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    packet_path = Path(args.packet_file)
    artifact_root = Path(args.artifact_dir)
    summary_jsonl = Path(args.summary_jsonl) if args.summary_jsonl else artifact_root / "summary.jsonl"
    summary_csv = Path(args.summary_csv) if args.summary_csv else artifact_root / "summary.csv"
    log_dir = artifact_root / "logs"
    artifact_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_packet(packet_path)
    if args.only:
        wanted = set(args.only)
        rows = [row for row in rows if row.get("name") in wanted]
    if not rows:
        print("No packet rows selected.", file=sys.stderr)
        return 1

    result_map: dict[str, dict[str, Any]] = _load_existing_summary(summary_jsonl)
    for index, row in enumerate(rows, start=1):
        name = str(row["name"])
        provider = str(row["provider"])
        model = str(row["model"])
        benchmark_path = args.benchmark_path or str(row["benchmark_path"])
        max_iterations = int(args.max_iterations or row.get("max_iterations", 15))
        run_artifact_dir = artifact_root / name
        command = [
            sys.executable,
            str(REPO_ROOT / "src" / "cli.py"),
            "benchmark",
            benchmark_path,
            "--split",
            str(row["split"]),
            "--test-id",
            str(row["test_id"]),
            "--agentic",
            "--provider",
            provider,
            "--model",
            model,
            "--max-iterations",
            str(max_iterations),
            "--display",
            str(row.get("display") or "raw"),
            "--artifact-dir",
            str(run_artifact_dir),
        ]

        print(f"\n[{index}/{len(rows)}] {name}: {provider}/{model}")
        print(" ".join(command))
        if args.dry_run:
            continue

        started = time.time()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout_seconds,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _decode_timeout_stream(exc.stdout)
            stderr = _decode_timeout_stream(exc.stderr)
            stderr = (stderr + "\n" if stderr else "") + (
                f"Timed out after {args.timeout_seconds}s"
            )
            returncode = 124
        elapsed = time.time() - started
        log_path = log_dir / f"{name}.log"
        log_path.write_text(
            stdout
            + ("\n--- STDERR ---\n" + stderr if stderr else ""),
            encoding="utf-8",
        )
        result = _summarize_run(
            row=row,
            returncode=returncode,
            elapsed_seconds=elapsed,
            log_path=log_path,
            stdout=stdout,
            stderr=stderr,
        )
        if timed_out:
            result["status"] = "timeout"
        result_map[name] = result
        ordered_results = _ordered_results(packet_path, result_map)
        _write_summaries(summary_jsonl, summary_csv, ordered_results)
        _print_result(result)

    if args.dry_run:
        return 0

    print(f"\nSummary JSONL: {summary_jsonl}")
    print(f"Summary CSV:   {summary_csv}")
    return 0 if all(row.get("returncode") == 0 for row in result_map.values()) else 2


def _load_packet(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = json.loads(stripped)
            for field in ["name", "provider", "model", "benchmark_path", "split", "test_id"]:
                if field not in row:
                    raise ValueError(f"{path}:{line_no} missing required field {field!r}")
            rows.append(row)
    return rows


def _load_existing_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            name = row.get("name")
            if isinstance(name, str):
                rows[name] = row
    return rows


def _ordered_results(packet_path: Path, result_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_names = [str(row["name"]) for row in _load_packet(packet_path)]
    rows = [result_map[name] for name in ordered_names if name in result_map]
    extras = sorted(name for name in result_map if name not in set(ordered_names))
    rows.extend(result_map[name] for name in extras)
    return rows


def _decode_timeout_stream(stream: Any) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)


def _summarize_run(
    *,
    row: dict[str, Any],
    returncode: int,
    elapsed_seconds: float,
    log_path: Path,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    artifact_path = _last_prefixed_value(stdout, "Artifact:")
    artifact = _load_artifact(artifact_path)
    status = artifact.get("status") or _status_from_stdout(stdout)
    solution = artifact.get("solution") or {}
    return {
        "name": row["name"],
        "provider": row["provider"],
        "model": row["model"],
        "test_id": row["test_id"],
        "returncode": returncode,
        "status": status,
        "char_accuracy": artifact.get("char_accuracy"),
        "word_accuracy": artifact.get("word_accuracy"),
        "self_confidence": solution.get("self_confidence"),
        "iterations": artifact.get("iterations") or _iterations_from_artifact(artifact),
        "total_input_tokens": artifact.get("total_input_tokens"),
        "total_output_tokens": artifact.get("total_output_tokens"),
        "total_cache_read_tokens": artifact.get("total_cache_read_tokens"),
        "estimated_cost_usd": artifact.get("estimated_cost_usd"),
        "elapsed_seconds": artifact.get("elapsed_seconds") or elapsed_seconds,
        "artifact_path": artifact_path,
        "log_path": str(log_path),
        "error": artifact.get("error_message") or _error_from_output(stdout, stderr),
    }


def _load_artifact(path: str) -> dict[str, Any]:
    if not path:
        return {}
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {}
    try:
        return json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _last_prefixed_value(text: str, prefix: str) -> str:
    value = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            value = stripped[len(prefix):].strip()
    return value


def _status_from_stdout(stdout: str) -> str:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Status:"):
            return stripped.split(",", 1)[0].split(":", 1)[1].strip()
    return "error"


def _iterations_from_artifact(artifact: dict[str, Any]) -> int | None:
    calls = artifact.get("tool_calls") or []
    iterations = [int(call.get("iteration") or 0) for call in calls if isinstance(call, dict)]
    events = artifact.get("loop_events") or []
    iterations.extend(
        int(event.get("outer_iteration") or 0)
        for event in events
        if isinstance(event, dict)
    )
    return max(iterations) if iterations else None


def _error_from_output(stdout: str, stderr: str) -> str:
    combined = "\n".join(part for part in [stdout, stderr] if part)
    for line in combined.splitlines():
        if "Error:" in line or "Traceback" in line or "API error" in line:
            return line.strip()
    return ""


def _write_summaries(jsonl_path: Path, csv_path: Path, rows: list[dict[str, Any]]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_result(row: dict[str, Any]) -> None:
    char = row.get("char_accuracy")
    word = row.get("word_accuracy")
    cost = row.get("estimated_cost_usd")
    char_s = f"{char:.1%}" if isinstance(char, (int, float)) else "n/a"
    word_s = f"{word:.1%}" if isinstance(word, (int, float)) else "n/a"
    cost_s = f"${cost:.4f}" if isinstance(cost, (int, float)) else "n/a"
    print(
        f"  -> rc={row['returncode']} status={row['status']} "
        f"char={char_s} word={word_s} cost={cost_s}"
    )
    if row.get("error"):
        print(f"     error: {row['error']}")
    if row.get("artifact_path"):
        print(f"     artifact: {row['artifact_path']}")
    print(f"     log: {row['log_path']}")


if __name__ == "__main__":
    raise SystemExit(main())
