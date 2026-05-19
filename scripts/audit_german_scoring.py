#!/usr/bin/env python3
"""Audit German scoring signals on known-good benchmark plaintext.

This is a calibration/reporting script. It uses benchmark plaintext only after
loading solved records, and does not participate in solver routing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.dictionary import get_dictionary_path, load_word_set, score_plaintext
from analysis.ngram import NGRAM_CACHE, normalized_ngram_score
from analysis.zenith_solver import load_zenith_binary_model
from benchmark.loader import BenchmarkLoader
from frontier.suite import load_frontier_suite


DEFAULT_SUITE = REPO_ROOT / "frontier" / "copiale_evidence_packet.jsonl"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT.parent / "cipher_benchmark" / "benchmark"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare German dictionary, wordlist quadgram, and binary 5-gram scores."
    )
    parser.add_argument("--suite-file", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        default=[REPO_ROOT / "models" / "ngram5_de.bin", REPO_ROOT / "models" / "ngram5_de_500.bin"],
        help="Zenith-format binary models to compare.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for machine-readable report output.",
    )
    args = parser.parse_args()

    report = build_report(
        suite_file=args.suite_file,
        benchmark_root=args.benchmark_root,
        model_paths=args.models,
    )
    print(render_markdown(report))
    if args.output_json:
        output = args.output_json if args.output_json.is_absolute() else REPO_ROOT / args.output_json
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_report(
    *,
    suite_file: Path,
    benchmark_root: Path,
    model_paths: list[Path],
) -> dict[str, Any]:
    loader = BenchmarkLoader(benchmark_root)
    cases = load_frontier_suite(suite_file)
    word_path = get_dictionary_path("de")
    word_set = load_word_set(word_path) if word_path else set()
    quad = NGRAM_CACHE.get("de", 4)
    models = [load_binary_model_info(path) for path in model_paths]
    rows = []
    for case in cases:
        test_data = loader.load_test_data(case.test)
        plaintext = test_data.plaintext.strip()
        if not plaintext:
            continue
        variants = {
            "plaintext": plaintext,
            "reversed": plaintext[::-1],
            "shuffled": deterministic_shuffle_text(plaintext, seed=case.test.test_id),
            "rotated": rotate_letters(plaintext, shift=7),
        }
        rows.append({
            "test_id": case.test.test_id,
            "plaintext_chars": len(normalize_de_az(plaintext)),
            "scores": {
                variant_name: score_text_variant(
                    text,
                    word_set=word_set,
                    quad=quad,
                    models=models,
                )
                for variant_name, text in variants.items()
            },
        })
    return {
        "suite_file": str(suite_file),
        "benchmark_root": str(benchmark_root),
        "dictionary_path": word_path,
        "dictionary_word_count": len(word_set),
        "models": [model["metadata"] for model in models],
        "rows": rows,
        "aggregate": aggregate_rows(rows, models),
    }


def load_binary_model_info(path: Path) -> dict[str, Any]:
    path = path.expanduser()
    metadata_path = path.with_name(path.name + ".metadata.json")
    metadata: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
    }
    if path.exists():
        metadata["size_bytes"] = path.stat().st_size
    if metadata_path.exists():
        try:
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update({
                key: source_metadata[key]
                for key in (
                    "language",
                    "order",
                    "format",
                    "output_file",
                    "sha256",
                    "build_timestamp_utc",
                    "sources",
                    "corpus_stats",
                    "normalization",
                    "redistribution_status",
                )
                if key in source_metadata
            })
        except json.JSONDecodeError as exc:
            metadata["metadata_parse_error"] = str(exc)
    if path.exists() and "sha256" not in metadata:
        metadata["sha256"] = sha256_file(path)
    return {
        "path": path,
        "metadata": metadata,
        "model": load_zenith_binary_model(path) if path.exists() else None,
    }


def score_text_variant(
    text: str,
    *,
    word_set: set[str],
    quad: dict[str, float],
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    clean = normalize_de_az(text)
    row: dict[str, Any] = {
        "dict_rate": round(score_plaintext(text, word_set), 6),
        "wordlist_quad": round(normalized_ngram_score(text, quad, n=4), 6),
        "az_length": len(clean),
    }
    for model_info in models:
        key = Path(model_info["metadata"]["path"]).stem
        model = model_info.get("model")
        row[f"{key}_mean_log_prob"] = (
            round(binary_mean_log_prob(clean, model), 6) if model is not None else None
        )
    return row


def binary_mean_log_prob(clean_az: str, model: Any) -> float:
    if len(clean_az) < model.order:
        return float("-inf")
    total = 0.0
    count = 0
    lower = clean_az.lower()
    for idx in range(len(lower) - model.order + 1):
        total += model.lookup(lower[idx: idx + model.order])
        count += 1
    return total / max(1, count)


def aggregate_rows(rows: list[dict[str, Any]], models: list[dict[str, Any]]) -> dict[str, Any]:
    score_keys = ["dict_rate", "wordlist_quad"]
    score_keys.extend(f"{Path(model['metadata']['path']).stem}_mean_log_prob" for model in models)
    aggregate: dict[str, Any] = {}
    for key in score_keys:
        margins = []
        wins = 0
        for row in rows:
            scores = row["scores"]
            plaintext_score = scores["plaintext"].get(key)
            controls = [
                scores[name].get(key)
                for name in ("reversed", "shuffled", "rotated")
                if scores[name].get(key) is not None
            ]
            if plaintext_score is None or not controls:
                continue
            best_control = max(controls)
            margin = plaintext_score - best_control
            margins.append(margin)
            if margin > 0:
                wins += 1
        aggregate[key] = {
            "plaintext_beats_all_controls": wins,
            "sample_count": len(margins),
            "mean_margin": round(sum(margins) / len(margins), 6) if margins else None,
            "min_margin": round(min(margins), 6) if margins else None,
        }
    return aggregate


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# German Scoring Audit",
        "",
        "This report uses known plaintext only for calibration. It must not feed solver routing.",
        "",
        "## Resources",
        "",
        f"- Dictionary: `{report['dictionary_path']}` ({report['dictionary_word_count']} words)",
    ]
    for model in report["models"]:
        sources = model.get("sources") or []
        source_text = ", ".join(
            f"{src.get('name', 'unknown')}:{src.get('text_files', '?')} files"
            for src in sources
        ) or "unknown"
        lines.append(
            f"- Model: `{model.get('path')}` sha256={model.get('sha256', 'unknown')} "
            f"source={source_text}"
        )
    lines.extend([
        "",
        "## Aggregate",
        "",
        "| Signal | Plaintext wins | Mean margin | Worst margin |",
        "|---|---:|---:|---:|",
    ])
    for signal, metrics in report["aggregate"].items():
        lines.append(
            f"| {signal} | {metrics['plaintext_beats_all_controls']}/{metrics['sample_count']} | "
            f"{_fmt(metrics['mean_margin'])} | {_fmt(metrics['min_margin'])} |"
        )
    lines.extend([
        "",
        "## Samples",
        "",
        "| Test | Chars | Dict | Quad | Binary models |",
        "|---|---:|---:|---:|---|",
    ])
    for row in report["rows"]:
        plain = row["scores"]["plaintext"]
        binary_bits = [
            f"{key.replace('_mean_log_prob', '')}={_fmt(value)}"
            for key, value in plain.items()
            if key.endswith("_mean_log_prob")
        ]
        lines.append(
            f"| {row['test_id']} | {row['plaintext_chars']} | "
            f"{plain['dict_rate']:.3f} | {plain['wordlist_quad']:.3f} | "
            f"{', '.join(binary_bits)} |"
        )
    return "\n".join(lines)


def deterministic_shuffle_text(text: str, *, seed: str) -> str:
    letters = list(text)
    rng = random.Random(seed)
    rng.shuffle(letters)
    return "".join(letters)


def rotate_letters(text: str, *, shift: int) -> str:
    out = []
    for ch in text.upper():
        if "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def normalize_de_az(text: str) -> str:
    text = text.upper()
    text = (
        text.replace("Ä", "AE")
        .replace("Ö", "OE")
        .replace("Ü", "UE")
        .replace("ẞ", "SS")
        .replace("ß", "SS")
    )
    return "".join(ch for ch in text if "A" <= ch <= "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
