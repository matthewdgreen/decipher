#!/usr/bin/env python3
"""Build a ground-truth-free repair agenda from Copiale null-mask finalists.

The report is meant to bridge broad null-mask search and local repair. It does
not score against benchmark plaintext. Instead it compares solver-produced
finalists, finds unstable key assignments across the top runtime candidates,
and highlights damaged-looking text windows where those disputed symbols occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from analysis.language_scoring import language_quality_feature_dict  # noqa: E402
from benchmark.loader import BenchmarkLoader, parse_canonical_transcription  # noqa: E402
from models.alphabet import Alphabet  # noqa: E402
from models.cipher_text import CipherText  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    rank: int
    artifact: Path
    mask: tuple[str, ...]
    source: str
    sort_score: float
    validation_score_v2: float | None
    language_quality_rank_score: float | None
    ensemble_score_v1: float | None
    selection_score: float | None
    decryption: str
    key: dict[int, int]
    filtered_length: int | None
    row: dict[str, Any]

    @property
    def label(self) -> str:
        mask = ",".join(self.mask) if self.mask else "(none)"
        source = self.source or "candidate"
        return f"#{self.rank} {mask} [{source}]"

    @property
    def preview(self) -> str:
        return self.decryption[:140]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report key-disagreement and local repair targets from null-mask finalists."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", action="append", default=[], help="Artifact JSON path.")
    source.add_argument(
        "--experiment-dir",
        default="",
        help="Breadth experiment directory containing ranker artifacts.",
    )
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument("--test-id", default="", help="Required with --experiment-dir if ambiguous.")
    parser.add_argument("--top-n", type=int, default=8, help="Top runtime finalists for consensus.")
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--window-step", type=int, default=40)
    parser.add_argument("--worst-windows", type=int, default=8)
    parser.add_argument("--min-agreement", type=float, default=0.75)
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    artifacts = (
        [resolve_path(Path(path)) for path in args.artifact]
        if args.artifact
        else discover_artifacts(resolve_path(Path(args.experiment_dir)), test_id=args.test_id)
    )
    if not artifacts:
        raise SystemExit("No artifacts found.")
    artifact_test_ids = artifact_test_id_map(artifacts)
    observed_test_ids = {test_id for test_id in artifact_test_ids.values() if test_id}
    if args.test_id:
        artifacts = [
            path for path in artifacts
            if artifact_test_ids.get(path, args.test_id) == args.test_id
        ]
        if not artifacts:
            raise SystemExit(f"No artifacts matched --test-id {args.test_id}")
    elif len(observed_test_ids) > 1:
        options = ", ".join(sorted(observed_test_ids))
        raise SystemExit(f"Multiple test ids found; pass --test-id. Options: {options}")
    test_id = args.test_id or next(iter(observed_test_ids), "")
    if not test_id:
        raise SystemExit("Could not determine test_id from artifact; pass --test-id.")
    cipher = load_cipher(resolve_path(Path(args.benchmark_root)), args.split, test_id)
    candidates = load_candidates(artifacts)
    if not candidates:
        raise SystemExit("No null-mask finalists found in artifacts.")

    payload = build_report(
        test_id=test_id,
        artifacts=artifacts,
        cipher=cipher,
        candidates=candidates,
        top_n=args.top_n,
        min_agreement=args.min_agreement,
        window_size=args.window_size,
        window_step=args.window_step,
        worst_windows=args.worst_windows,
    )
    markdown = render_markdown(payload)
    output = resolve_path(Path(args.output)) if args.output else artifacts[0].with_suffix(".repair_agenda.md")
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


def build_report(
    *,
    test_id: str,
    artifacts: list[Path],
    cipher: CipherText,
    candidates: list[Candidate],
    top_n: int,
    min_agreement: float,
    window_size: int,
    window_step: int,
    worst_windows: int,
) -> dict[str, Any]:
    candidates = sorted(candidates, key=lambda row: row.sort_score, reverse=True)
    consensus_pool = candidates[:top_n]
    consensus = consensus_assignments(consensus_pool, cipher, min_agreement=min_agreement)
    selected = candidates[0]
    windows = damaged_windows(
        selected,
        cipher,
        consensus=consensus,
        language="de",
        window_size=window_size,
        step=window_step,
        limit=worst_windows,
    )
    return {
        "test_id": test_id,
        "artifacts": [str(path) for path in artifacts],
        "candidate_count": len(candidates),
        "consensus_pool_size": len(consensus_pool),
        "min_agreement": float(min_agreement),
        "selected": candidate_summary(selected),
        "top_candidates": [candidate_summary(row) for row in candidates[:max(top_n, 12)]],
        "consensus_summary": summarize_consensus(consensus),
        "most_disputed_symbols": most_disputed_symbols(consensus, limit=20),
        "repair_windows": windows,
    }


def load_candidates(artifacts: list[Path]) -> list[Candidate]:
    rows: list[Candidate] = []
    seen: set[tuple[tuple[str, ...], str, str]] = set()
    for artifact in artifacts:
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        null_step = next(
            (
                step for step in payload.get("steps") or []
                if isinstance(step, dict) and step.get("name") == "search_null_masks"
            ),
            None,
        )
        if not isinstance(null_step, dict):
            continue
        candidate_rows: list[dict[str, Any]] = []
        selected = null_step.get("selected")
        if isinstance(selected, dict):
            candidate_rows.append(selected)
        candidate_rows.extend(
            row for row in (null_step.get("top_finalists") or [])
            if isinstance(row, dict)
        )
        for row in candidate_rows:
            text = str(row.get("decryption") or row.get("plaintext") or "")
            key = parse_key(row.get("key"))
            if not text or not key:
                continue
            mask = tuple(str(symbol) for symbol in (row.get("mask") or []))
            source = str(row.get("source") or "")
            identity = (mask, source, text)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(
                Candidate(
                    rank=0,
                    artifact=artifact,
                    mask=mask,
                    source=source,
                    sort_score=runtime_sort_score(row),
                    validation_score_v2=float_or_none(row.get("validation_score_v2")),
                    language_quality_rank_score=float_or_none(row.get("language_quality_rank_score")),
                    ensemble_score_v1=float_or_none(row.get("ensemble_score_v1")),
                    selection_score=float_or_none(row.get("selection_score")),
                    decryption=text,
                    key=key,
                    filtered_length=int(row.get("filtered_length") or 0) or None,
                    row=row,
                )
            )
    ranked = sorted(rows, key=lambda row: row.sort_score, reverse=True)
    return [
        Candidate(
            rank=idx,
            artifact=row.artifact,
            mask=row.mask,
            source=row.source,
            sort_score=row.sort_score,
            validation_score_v2=row.validation_score_v2,
            language_quality_rank_score=row.language_quality_rank_score,
            ensemble_score_v1=row.ensemble_score_v1,
            selection_score=row.selection_score,
            decryption=row.decryption,
            key=row.key,
            filtered_length=row.filtered_length,
            row=row.row,
        )
        for idx, row in enumerate(ranked, start=1)
    ]


def runtime_sort_score(row: dict[str, Any]) -> float:
    """Use the same broad-menu spirit as runtime: LQ when present, then validation."""
    lq = float_or_none(row.get("language_quality_rank_score"))
    validation = float_or_none(row.get("validation_score_v2"))
    ensemble = float_or_none(row.get("ensemble_score_v1"))
    selection = float_or_none(row.get("selection_score"))
    score = 0.0
    if lq is not None:
        score += lq * 1_000_000.0
    if validation is not None:
        score += validation * 1_000.0
    if ensemble is not None:
        score += ensemble * 10.0
    if selection is not None:
        score += selection
    return score


def consensus_assignments(
    candidates: list[Candidate],
    cipher: CipherText,
    *,
    min_agreement: float,
) -> dict[str, dict[str, Any]]:
    pt_alpha = Alphabet.standard_english()
    summary: dict[str, dict[str, Any]] = {}
    for token_id in sorted(set(cipher.tokens)):
        symbol = cipher.alphabet.symbol_for(token_id)
        counts: Counter[str] = Counter()
        for candidate in candidates:
            if symbol in candidate.mask:
                counts["<null>"] += 1
                continue
            pt_id = candidate.key.get(token_id)
            if pt_id is None:
                counts["?"] += 1
            elif 0 <= pt_id < pt_alpha.size:
                counts[pt_alpha.symbol_for(pt_id).upper()] += 1
            else:
                counts[str(pt_id)] += 1
        total = sum(counts.values())
        winner, winner_count = counts.most_common(1)[0] if counts else ("?", 0)
        agreement = winner_count / max(1, total)
        summary[symbol] = {
            "symbol": symbol,
            "total": total,
            "winner": winner,
            "agreement": round(agreement, 4),
            "stable": agreement >= min_agreement,
            "counts": dict(counts.most_common()),
        }
    return summary


def damaged_windows(
    candidate: Candidate,
    cipher: CipherText,
    *,
    consensus: dict[str, dict[str, Any]],
    language: str,
    window_size: int,
    step: int,
    limit: int,
) -> list[dict[str, Any]]:
    text, sources = reconstruct_candidate(candidate, cipher)
    if not text:
        text = candidate.decryption
        sources = [""] * len(text)
    if len(text) <= window_size:
        starts = [0]
    else:
        starts = list(range(0, max(1, len(text) - window_size + 1), max(1, step)))
        if starts[-1] != len(text) - window_size:
            starts.append(len(text) - window_size)
    rows: list[dict[str, Any]] = []
    for start in starts:
        end = min(len(text), start + window_size)
        window_text = text[start:end]
        features = language_quality_feature_dict(window_text, language=language)
        damage = window_damage_score(features)
        symbol_counts = Counter(symbol for symbol in sources[start:end] if symbol)
        disputed = []
        for symbol, count in symbol_counts.most_common():
            info = consensus.get(symbol)
            if not info or info.get("stable"):
                continue
            disputed.append({
                "symbol": symbol,
                "count": count,
                "winner": info.get("winner"),
                "agreement": info.get("agreement"),
                "assignments": info.get("counts"),
            })
            if len(disputed) >= 8:
                break
        rows.append({
            "start": start,
            "end": end,
            "damage_score": round(damage, 4),
            "language_coherence": round(float(features.get("language_coherence") or 0.0), 4),
            "language_shape": round(float(features.get("language_shape") or 0.0), 4),
            "repetition_control": round(float(features.get("repetition_control") or 0.0), 4),
            "function_overuse_control": round(float(features.get("function_overuse_control") or 0.0), 4),
            "disputed_symbol_count": len(disputed),
            "disputed_symbols": disputed,
            "text": window_text,
        })
    return sorted(
        rows,
        key=lambda row: (
            float(row["damage_score"]),
            int(row["disputed_symbol_count"]),
        ),
        reverse=True,
    )[:limit]


def window_damage_score(features: dict[str, float]) -> float:
    good = (
        0.28 * float(features.get("language_coherence") or 0.0)
        + 0.22 * float(features.get("language_shape") or 0.0)
        + 0.15 * float(features.get("language_evidence_dispersion") or 0.0)
        + 0.12 * float(features.get("function_content_balance") or 0.0)
        + 0.10 * float(features.get("repetition_control") or 0.0)
        + 0.08 * float(features.get("function_overuse_control") or 0.0)
        + 0.05 * float(features.get("short_fragment_control") or 0.0)
    )
    return max(0.0, min(1.0, 1.0 - good))


def reconstruct_candidate(candidate: Candidate, cipher: CipherText) -> tuple[str, list[str]]:
    pt_alpha = Alphabet.standard_english()
    chars: list[str] = []
    sources: list[str] = []
    masked = set(candidate.mask)
    for token_id in cipher.tokens:
        symbol = cipher.alphabet.symbol_for(token_id)
        if symbol in masked:
            continue
        pt_id = candidate.key.get(token_id)
        if pt_id is None or pt_id < 0 or pt_id >= pt_alpha.size:
            continue
        chars.append(pt_alpha.symbol_for(pt_id).upper())
        sources.append(symbol)
    reconstructed = "".join(chars)
    artifact_text = "".join(ch for ch in candidate.decryption.upper() if "A" <= ch <= "Z")
    if artifact_text and reconstructed[:80] != artifact_text[:80]:
        return artifact_text, [""] * len(artifact_text)
    return reconstructed, sources


def summarize_consensus(consensus: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stable = [item for item in consensus.values() if item.get("stable")]
    null_winners = [item for item in consensus.values() if item.get("winner") == "<null>"]
    return {
        "symbol_count": len(consensus),
        "stable_symbol_count": len(stable),
        "disputed_symbol_count": len(consensus) - len(stable),
        "null_winner_count": len(null_winners),
    }


def most_disputed_symbols(
    consensus: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = [
        item for item in consensus.values()
        if not bool(item.get("stable"))
    ]
    rows.sort(
        key=lambda item: (
            float(item.get("agreement") or 0.0),
            str(item.get("symbol") or ""),
        )
    )
    return rows[:limit]


def candidate_summary(candidate: Candidate) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "mask": list(candidate.mask),
        "source": candidate.source,
        "sort_score": round(candidate.sort_score, 4),
        "validation_score_v2": round_or_none(candidate.validation_score_v2),
        "language_quality_rank_score": round_or_none(candidate.language_quality_rank_score),
        "ensemble_score_v1": round_or_none(candidate.ensemble_score_v1),
        "selection_score": round_or_none(candidate.selection_score),
        "filtered_length": candidate.filtered_length,
        "preview": candidate.preview,
        "artifact": str(candidate.artifact),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    selected = payload["selected"]
    lines = [
        f"# Copiale Repair Agenda: {payload['test_id']}",
        "",
        "This report is ground-truth-free. It compares already-produced null-mask finalists,",
        "looks for key disagreement, and points at damaged windows for local repair.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| candidates | {payload['candidate_count']} |",
        f"| consensus pool | {payload['consensus_pool_size']} |",
        f"| stable symbols | {payload['consensus_summary']['stable_symbol_count']} |",
        f"| disputed symbols | {payload['consensus_summary']['disputed_symbol_count']} |",
        f"| selected mask | {mask_label(selected['mask'])} |",
        f"| selected source | {selected['source']} |",
        f"| selected LQ rank score | {format_optional(selected['language_quality_rank_score'])} |",
        f"| selected validation v2 | {format_optional(selected['validation_score_v2'])} |",
        "",
        "## Top Runtime Candidates",
        "",
        "| Rank | Mask | Source | LQ | Val2 | Ens | Selection | Preview |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["top_candidates"][:12]:
        lines.append(
            "| {rank} | {mask} | {source} | {lq} | {val} | {ens} | {sel} | {preview} |".format(
                rank=row["rank"],
                mask=mask_label(row["mask"]),
                source=escape_md(row["source"]),
                lq=format_optional(row["language_quality_rank_score"]),
                val=format_optional(row["validation_score_v2"]),
                ens=format_optional(row["ensemble_score_v1"]),
                sel=format_optional(row["selection_score"]),
                preview=escape_md(row["preview"]),
            )
        )
    lines.extend([
        "",
        "## Most Disputed Symbols",
        "",
        "| Symbol | Winner | Agreement | Assignments |",
        "|---|---|---:|---|",
    ])
    for row in payload["most_disputed_symbols"]:
        lines.append(
            f"| {escape_md(row['symbol'])} | {escape_md(str(row['winner']))} | "
            f"{float(row['agreement']):.2f} | {escape_md(format_counts(row['counts']))} |"
        )
    lines.extend([
        "",
        "## Repair Windows",
        "",
        "| Rank | Span | Damage | Coherence | Shape | Disputed | Text |",
        "|---:|---|---:|---:|---:|---:|---|",
    ])
    for idx, row in enumerate(payload["repair_windows"], start=1):
        lines.append(
            f"| {idx} | {row['start']}-{row['end']} | {row['damage_score']:.3f} | "
            f"{row['language_coherence']:.3f} | {row['language_shape']:.3f} | "
            f"{row['disputed_symbol_count']} | {escape_md(row['text'])} |"
        )
    lines.extend(["", "## Window Symbol Targets", ""])
    for idx, row in enumerate(payload["repair_windows"], start=1):
        if not row["disputed_symbols"]:
            continue
        lines.append(f"### Window {idx}: {row['start']}-{row['end']}")
        lines.append("")
        lines.append("| Symbol | Count In Window | Consensus Winner | Agreement | Assignments |")
        lines.append("|---|---:|---|---:|---|")
        for symbol in row["disputed_symbols"]:
            lines.append(
                f"| {escape_md(symbol['symbol'])} | {symbol['count']} | "
                f"{escape_md(str(symbol['winner']))} | {float(symbol['agreement']):.2f} | "
                f"{escape_md(format_counts(symbol['assignments']))} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def discover_artifacts(root: Path, *, test_id: str) -> list[Path]:
    paths = []
    for path in sorted(root.glob("*/automated_only/*/*.json")):
        if test_id and path.parent.name != test_id:
            continue
        paths.append(path)
    return paths


def artifact_test_id_map(paths: list[Path]) -> dict[Path, str]:
    result: dict[Path, str] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            result[path] = ""
            continue
        result[path] = str(payload.get("test_id") or path.parent.name)
    return result


def load_cipher(benchmark_root: Path, split: str, test_id: str) -> CipherText:
    loader = BenchmarkLoader(benchmark_root)
    tests = [test for test in loader.load_tests(split) if test.test_id == test_id]
    if not tests:
        raise SystemExit(f"Test not found in split {split}: {test_id}")
    data = loader.load_test_data(tests[0])
    return parse_canonical_transcription(data.canonical_transcription)


def parse_key(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[int, int] = {}
    for key, plaintext in value.items():
        try:
            parsed[int(key)] = int(plaintext)
        except (TypeError, ValueError):
            continue
    return parsed


def float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in counts.items())


def mask_label(mask: list[str] | tuple[str, ...]) -> str:
    return ",".join(mask) if mask else "(none)"


def escape_md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def resolve_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


if __name__ == "__main__":
    main()
