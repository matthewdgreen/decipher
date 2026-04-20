#!/usr/bin/env python3
"""Build a parity dashboard from agent and external-baseline artifacts."""
from __future__ import annotations

import argparse

from artifact.dashboard import build_dashboard, render_json, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate parity artifacts into a dashboard table."
    )
    parser.add_argument(
        "--agent",
        action="append",
        default=[],
        help="Agent artifact file/dir/glob. Defaults to artifacts/**/*.json.",
    )
    parser.add_argument(
        "--external",
        action="append",
        default=[],
        help=(
            "External baseline artifact file/dir/glob. Defaults to "
            "artifacts/external_baselines/**/artifact.json."
        ),
    )
    parser.add_argument(
        "--automated",
        action="append",
        default=[],
        help=(
            "Automated-only/no-LLM artifact file/dir/glob. Defaults to "
            "artifacts/automated_only/**/*.json when present."
        ),
    )
    parser.add_argument(
        "--benchmark-root",
        default="../cipher_benchmark/benchmark",
        help="Benchmark root used to attach split metadata.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
    )
    args = parser.parse_args()

    agent_paths = args.agent or ["artifacts/**/*.json"]
    external_paths = args.external or ["artifacts/external_baselines/**/artifact.json"]
    automated_paths = args.automated or ["artifacts/automated_only/**/*.json"]
    rows = build_dashboard(
        agent_paths=agent_paths,
        external_paths=external_paths,
        automated_paths=automated_paths,
        benchmark_root=args.benchmark_root,
    )
    if args.format == "json":
        print(render_json(rows))
    else:
        print(render_markdown(rows))


if __name__ == "__main__":
    main()
