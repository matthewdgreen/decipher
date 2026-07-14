#!/usr/bin/env python3
"""Build the INV model-diagnosis synthetic suite + manifest (run policy step 0).

Constructs a curated multi-family synthetic cipher suite with FRESH plaintexts
(famous ciphers are memorization-poisoned and must not be used), writes a
manifest with canonical family enum labels, neutral case_ids, and per-case
plaintext hashes, and populates the plaintext cache along the way.

This IS run-policy step 0: on a cache miss it makes a tiny LLM call per row (one
short plaintext each) via the testgen plaintext generator; a warm cache makes
re-runs free and deterministic. Because ``PlaintextCache`` keys on
(language, approx_length, topic, frequency_style) and each row has a distinct
``topic``, every row gets a distinct plaintext (F5).

Ground truth (family, plaintext) is FIREWALLED: it lives only in the manifest
for post-hoc scoring, never in any model prompt.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_inv_diagnosis_suite.py \
        [--generator-provider openai] [--generator-model MODEL] \
        [--cache-dir testgen_cache] [--output frontier/inv_diagnosis_suite.jsonl] \
        [--offline]

    --offline   Refuse LLM calls; rebuild the manifest from a warm cache only
                (deterministic; also flags any silent cache regeneration).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.inv_diagnosis import (  # noqa: E402
    MANIFEST_REQUIRED_KEYS,
    SUITE_ROWS,
    build_case,
    manifest_row,
)
from testgen.cache import PlaintextCache  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "frontier" / "inv_diagnosis_suite.jsonl"
DEFAULT_CACHE_DIR = REPO_ROOT / "testgen_cache"


def _resolve_generator_key(provider: str) -> str:
    """Resolve an API key for the plaintext generator provider (cli helpers)."""
    from cli import get_api_key

    return get_api_key(provider)


def _load_existing_hashes(output: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not output.exists():
        return hashes
    for line in output.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("case_id"), str):
            hashes[row["case_id"]] = row.get("plaintext_sha256", "")
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-provider", default="openai")
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Rebuild manifest from a warm cache only; never call the LLM.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    cache = PlaintextCache(args.cache_dir)

    api_key = ""
    if not args.offline:
        try:
            api_key = _resolve_generator_key(args.generator_provider)
        except SystemExit:
            print(
                "No generator API key found; retry with --offline (warm cache) "
                "or configure a key for --generator-provider "
                f"{args.generator_provider}.",
                file=sys.stderr,
            )
            return 1

    prior_hashes = _load_existing_hashes(output)
    manifest_rows: list[dict] = []
    regenerated: list[str] = []

    for row in SUITE_ROWS:
        test_data = build_case(
            row,
            cache,
            api_key=api_key,
            generator_provider=args.generator_provider,
            generator_model=args.generator_model,
            offline=args.offline,
        )
        record = manifest_row(row, test_data)
        # Validate schema completeness before writing.
        missing = [k for k in MANIFEST_REQUIRED_KEYS if k not in record]
        if missing:
            print(f"{row.case_id}: manifest missing keys {missing}", file=sys.stderr)
            return 2
        prior = prior_hashes.get(row.case_id)
        if prior and prior != record["plaintext_sha256"]:
            regenerated.append(row.case_id)
        manifest_rows.append(record)
        print(
            f"  {row.case_id:18s} family={row.true_family:28s} "
            f"system={record['cipher_system']:26s} pt#={record['plaintext_sha256']}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in manifest_rows:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"\nWrote {len(manifest_rows)} cases to {output}")
    if regenerated:
        print(
            "WARNING (F5): plaintext hash changed vs prior manifest for "
            f"{regenerated} — cache was silently regenerated. Investigate before "
            "trusting comparisons across the change.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
