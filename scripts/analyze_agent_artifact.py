#!/usr/bin/env python3
"""Analyze v2 agent artifacts for parity/gap labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifact.analyzer import analyze_artifact, load_artifact, summarize_findings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one or more v2 run artifacts for parity gap labels."
    )
    parser.add_argument("artifacts", nargs="+", help="Artifact JSON file(s)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    summaries = []
    for artifact_path in args.artifacts:
        path = Path(artifact_path)
        artifact = load_artifact(path)
        findings = analyze_artifact(artifact)
        summary = summarize_findings(findings)
        summary["artifact"] = str(path)
        summary["cipher_id"] = artifact.get("cipher_id")
        summary["status"] = artifact.get("status")
        summary["char_accuracy"] = artifact.get("char_accuracy")
        summaries.append(summary)

    if args.json:
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return

    for summary in summaries:
        print(f"{summary['artifact']} — {summary.get('cipher_id')} ({summary.get('status')})")
        print(f"  findings: {summary['finding_count']}  labels={summary['labels']}")
        if summary.get("char_accuracy") is not None:
            print(f"  char_accuracy: {summary['char_accuracy']:.1%}")
        for finding in summary["findings"]:
            loc = ""
            if finding.get("iteration") is not None:
                loc = f" iter={finding['iteration']}"
            tool = f" tool={finding['tool_name']}" if finding.get("tool_name") else ""
            print(
                f"  - [{finding['severity']}] {finding['label']}{loc}{tool}: "
                f"{finding['message']}"
            )
        print()


if __name__ == "__main__":
    main()
