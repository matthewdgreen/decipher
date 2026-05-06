#!/usr/bin/env python3
"""Export benchmark transcriptions as AZdecrypt batch files.

AZdecrypt's batch parser supports numeric ciphertext rows. That is useful for
historical benchmark records such as Copiale, whose symbol inventory is larger
than the compact single-character alphabet used by Decipher's generic external
baseline harness.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from analysis.homophonic_nulls import (
    diagnose_cipher_for_null_candidates,
    generate_null_masks,
    select_null_candidate_symbols,
)
from benchmark.loader import BenchmarkLoader, BenchmarkTest, parse_canonical_transcription


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = (REPO_ROOT / "../cipher_benchmark/benchmark").resolve()
DEFAULT_AZDECRYPT_DIR = REPO_ROOT / "other_tools/azdecrypt-src/AZdecrypt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark tests to AZdecrypt numeric batch format."
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help=f"benchmark root (default: {DEFAULT_BENCHMARK_ROOT})",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="benchmark split JSONL, relative to benchmark/splits unless absolute",
    )
    parser.add_argument(
        "--track",
        default="transcription2plaintext",
        help="track filter (default: transcription2plaintext)",
    )
    parser.add_argument(
        "--test-id",
        action="append",
        default=[],
        help="test id to export; may be repeated. If omitted, exports all matching tests.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="batch file path to write (default: artifacts/azdecrypt_batch/<split>_<track>.txt)",
    )
    parser.add_argument(
        "--install-to-azdecrypt",
        action="store_true",
        help="also copy outputs into other_tools/azdecrypt-src/AZdecrypt/Ciphers/Batch/",
    )
    parser.add_argument(
        "--azdecrypt-dir",
        type=Path,
        default=DEFAULT_AZDECRYPT_DIR,
        help=f"AZdecrypt app directory (default: {DEFAULT_AZDECRYPT_DIR})",
    )
    parser.add_argument(
        "--numbers-per-line",
        type=int,
        default=40,
        help="numeric tokens per ciphertext line (default: 40)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500000,
        help="AZdecrypt iterations metadata per item (default: 500000)",
    )
    parser.add_argument(
        "--include-plaintext",
        action="store_true",
        help=(
            "include ground-truth plaintext for AZdecrypt accuracy calibration. "
            "Do not use for blind external baseline runs."
        ),
    )
    parser.add_argument(
        "--raw-cipher-only",
        action="store_true",
        help=(
            "write only the numeric ciphertext for a single selected test. "
            "Useful when loading a cipher into AZdecrypt's ordinary input window."
        ),
    )
    parser.add_argument(
        "--null-mask-variants",
        action="store_true",
        help=(
            "export ciphertext-derived null-mask variants instead of only the raw "
            "transcription. This does not use plaintext."
        ),
    )
    parser.add_argument(
        "--null-candidate-limit",
        type=int,
        default=14,
        help="candidate symbols to consider for --null-mask-variants (default: 14)",
    )
    parser.add_argument(
        "--max-mask-size",
        type=int,
        default=2,
        help="maximum null-mask size for --null-mask-variants (default: 2)",
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=80,
        help="cap generated non-empty null masks per test; 0 means no cap (default: 80)",
    )
    args = parser.parse_args()

    loader = BenchmarkLoader(args.benchmark_root)
    tests = loader.load_tests(args.split, track=args.track)
    if args.test_id:
        wanted = set(args.test_id)
        tests = [test for test in tests if test.test_id in wanted]
        missing = sorted(wanted - {test.test_id for test in tests})
        if missing:
            raise SystemExit(f"test id(s) not found after filters: {', '.join(missing)}")
    if not tests:
        raise SystemExit("no tests selected")
    if args.raw_cipher_only and len(tests) != 1:
        raise SystemExit("--raw-cipher-only requires exactly one selected test")

    output_file = args.output_file
    if output_file is None:
        split_stem = Path(args.split).stem
        track = _safe_name(args.track or "all")
        output_file = REPO_ROOT / "artifacts" / "azdecrypt_batch" / f"{split_stem}_{track}.txt"
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    batch_text, symbol_map = _build_batch(
        loader=loader,
        tests=tests,
        include_plaintext=args.include_plaintext,
        numbers_per_line=args.numbers_per_line,
        iterations=args.iterations,
        null_mask_variants=args.null_mask_variants,
        null_candidate_limit=args.null_candidate_limit,
        max_mask_size=args.max_mask_size,
        max_masks=args.max_masks,
    )
    if args.raw_cipher_only:
        only = next(iter(symbol_map.values()))
        variants = only.get("variants") or []
        if len(variants) != 1:
            raise SystemExit("--raw-cipher-only cannot be combined with --null-mask-variants")
        test_data = loader.load_test_data(tests[0])
        cipher_text = parse_canonical_transcription(test_data.canonical_transcription)
        batch_text = "\n".join(_wrap_numbers([token + 1 for token in cipher_text.tokens], args.numbers_per_line))
    output_file.write_text(batch_text, encoding="utf-8")
    map_file = output_file.with_suffix(output_file.suffix + ".symbol_map.json")
    map_file.write_text(json.dumps(symbol_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    installed: list[Path] = []
    if args.install_to_azdecrypt:
        batch_dir = args.azdecrypt_dir.expanduser().resolve() / "Ciphers" / "Batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        installed_batch = batch_dir / output_file.name
        installed_map = batch_dir / map_file.name
        installed_batch.write_text(batch_text, encoding="utf-8")
        installed_map.write_text(
            json.dumps(symbol_map, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        installed = [installed_batch, installed_map]

    print(f"Wrote {output_file}")
    print(f"Wrote {map_file}")
    if installed:
        for path in installed:
            print(f"Installed {path}")
    print(f"Tests: {len(tests)}")
    print("Open in AZdecrypt with File -> Batch ciphers (substitution).")


def _build_batch(
    *,
    loader: BenchmarkLoader,
    tests: list[BenchmarkTest],
    include_plaintext: bool,
    numbers_per_line: int,
    iterations: int,
    null_mask_variants: bool,
    null_candidate_limit: int,
    max_mask_size: int,
    max_masks: int,
) -> tuple[str, dict]:
    lines: list[str] = []
    symbol_map: dict[str, dict] = {}

    for test in tests:
        test_data = loader.load_test_data(test)
        cipher_text = parse_canonical_transcription(test_data.canonical_transcription)
        variants = _cipher_variants(
            test_id=test.test_id,
            cipher_text=cipher_text,
            null_mask_variants=null_mask_variants,
            null_candidate_limit=null_candidate_limit,
            max_mask_size=max_mask_size,
            max_masks=max_masks,
        )
        for variant in variants:
            item_name = _safe_name(variant["id"])

            lines.append(f"output_sub_directory={item_name}")
            lines.append(f"cipher_information={variant['id']}")
            lines.append(f"iterations={iterations}")
            for row in _wrap_numbers(variant["az_tokens"], numbers_per_line):
                lines.append(row)

            if include_plaintext and test_data.plaintext:
                lines.append("solution_plaintext=")
                plaintext = _az_plaintext(test_data.plaintext)
                for start in range(0, len(plaintext), 80):
                    lines.append(plaintext[start : start + 80])

            lines.append("")
        symbol_map[test.test_id] = {
            "records": [record.id for record in test_data.target_records],
            "track": test.track,
            "cipher_system": test.cipher_system,
            "plaintext_language": test_data.plaintext_language,
            "token_count": len(cipher_text.tokens),
            "word_count": len(cipher_text.words),
            "symbol_count": cipher_text.alphabet.size,
            "variant_count": len(variants),
            "variants": [
                {
                    "id": variant["id"],
                    "mask": variant["mask"],
                    "filtered_token_count": len(variant["az_tokens"]),
                }
                for variant in variants
            ],
            "azdecrypt_numeric_symbols": {
                str(index + 1): symbol
                for index, symbol in enumerate(cipher_text.alphabet.symbols)
            },
            "plaintext_included": include_plaintext,
        }

    return "\n".join(lines), symbol_map


def _cipher_variants(
    *,
    test_id: str,
    cipher_text,
    null_mask_variants: bool,
    null_candidate_limit: int,
    max_mask_size: int,
    max_masks: int,
) -> list[dict]:
    if not null_mask_variants:
        return [
            {
                "id": test_id,
                "mask": [],
                "az_tokens": [token + 1 for token in cipher_text.tokens],
            }
        ]

    diagnostics = diagnose_cipher_for_null_candidates(cipher_text)
    candidates = select_null_candidate_symbols(diagnostics, limit=null_candidate_limit)
    masks = generate_null_masks(candidates, max_mask_size)
    if max_masks > 0:
        masks = [()] + masks[1 : max_masks + 1]

    variants = []
    for index, mask in enumerate(masks):
        mask_set = set(mask)
        filtered = [
            token
            for token in cipher_text.tokens
            if cipher_text.alphabet.decode([token]) not in mask_set
        ]
        suffix = "raw" if not mask else "mask_" + "_".join(mask)
        variants.append({
            "id": f"{test_id}__{index:03d}_{suffix}",
            "mask": list(mask),
            "az_tokens": [token + 1 for token in filtered],
        })
    return variants


def _wrap_numbers(values: list[int], per_line: int) -> list[str]:
    if per_line < 1:
        raise SystemExit("--numbers-per-line must be >= 1")
    return [
        " ".join(str(value) for value in values[start : start + per_line])
        for start in range(0, len(values), per_line)
    ]


def _az_plaintext(text: str) -> str:
    return re.sub(r"[^A-Z]", "", text.upper())


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "azdecrypt_batch"


if __name__ == "__main__":
    main()
