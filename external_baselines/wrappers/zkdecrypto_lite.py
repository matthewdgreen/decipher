#!/usr/bin/env python3
"""Wrapper for zkdecrypto-lite.

zkdecrypto-lite must be run from its source directory because it loads
``language/eng`` with a relative path. It prints the final answer as:

    SCORE,PLAINTEXT

This wrapper normalizes that to ``SOLUTION: PLAINTEXT`` for the sidecar
external-baseline harness.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run zkdecrypto-lite and normalize output.")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    source_dir = Path(args.source_dir).resolve()
    input_file = Path(args.input).resolve()
    output_file = Path(args.output).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [str(binary), str(input_file), "-t", str(args.seconds)],
        cwd=source_dir,
        capture_output=True,
        text=True,
        timeout=args.seconds + 20,
    )
    combined = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    plaintext = _extract_final_plaintext(combined)
    if plaintext:
        normalized = f"SOLUTION: {plaintext}\n"
        output_file.write_text(normalized, encoding="utf-8")
        print(normalized, end="")
    else:
        output_file.write_text(combined, encoding="utf-8")
        print(combined, end="")
    return 0 if plaintext else (proc.returncode or 1)


def _extract_final_plaintext(text: str) -> str:
    candidates: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(-?\d+)\s*,\s*([A-Za-z]+)\s*$", line)
        if match:
            candidates.append((int(match.group(1)), match.group(2).upper()))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    raise SystemExit(main())
