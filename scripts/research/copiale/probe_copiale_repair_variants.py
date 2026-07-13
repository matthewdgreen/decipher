#!/usr/bin/env python3
"""Try cheap local repair variants from a Copiale repair agenda.

This is a ground-truth-free probe. It starts from the selected null-mask
finalist, changes only disputed symbols named by the repair agenda, reconstructs
plaintext directly from the modified key/mask, and ranks variants with the
same runtime validation signals used by null-mask candidate menus. It does not
rerun the homophonic annealer; successful variants are evidence for the next
targeted repair search, not final solves.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from analysis.homophonic_nulls import null_mask_validation_score_v2  # noqa: E402
from analysis.language_scoring import language_quality_feature_dict  # noqa: E402
from automated.runner import _automated_candidate_diagnostics, _plaintext_quality, _word_list  # noqa: E402
from benchmark.loader import BenchmarkLoader, parse_canonical_transcription  # noqa: E402
from report_copiale_repair_agenda import (  # noqa: E402
    damaged_windows,
    parse_key,
    reconstruct_candidate,
    window_damage_score,
)


@dataclass(frozen=True)
class RepairVariant:
    edits: tuple[str, ...]
    mask: tuple[str, ...]
    key: dict[int, int]
    text: str
    validation_score_v2: float
    validation_components_v2: dict[str, float]
    language_quality_mean: float
    repair_window_damage: float
    diagnostics: dict[str, Any]
    quality: dict[str, Any]

    @property
    def preview(self) -> str:
        return self.text[:160]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe local repair variants from a null-mask repair agenda."
    )
    parser.add_argument("repair_agenda_json", help="JSON produced by report_copiale_repair_agenda.py")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument("--max-symbols", type=int, default=7)
    parser.add_argument("--max-alternatives", type=int, default=4)
    parser.add_argument("--include-pairs", action="store_true")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    agenda_path = resolve_path(Path(args.repair_agenda_json))
    agenda = json.loads(agenda_path.read_text(encoding="utf-8"))
    test_id = str(agenda.get("test_id") or "")
    if not test_id:
        raise SystemExit("Repair agenda has no test_id.")
    cipher = load_cipher(resolve_path(Path(args.benchmark_root)), args.split, test_id)
    selected_row, selected_artifact = load_selected_row(agenda)
    baseline_key = parse_key(selected_row.get("key"))
    baseline_mask = tuple(str(symbol) for symbol in (selected_row.get("mask") or []))
    binary_model_path = binary_model_from_row_or_artifact(selected_row, selected_artifact)

    edit_groups = build_edit_groups(
        agenda,
        cipher,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        max_symbols=args.max_symbols,
        max_alternatives=args.max_alternatives,
    )
    variants = evaluate_variants(
        cipher=cipher,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        edit_groups=edit_groups,
        agenda=agenda,
        binary_model_path=binary_model_path,
        include_pairs=args.include_pairs,
    )
    variants.sort(
        key=lambda item: (
            item.validation_score_v2,
            item.language_quality_mean,
            -item.repair_window_damage,
        ),
        reverse=True,
    )
    payload = {
        "test_id": test_id,
        "agenda": str(agenda_path),
        "selected_artifact": str(selected_artifact),
        "baseline": variant_to_dict(variants[0]) if variants else {},
        "variant_count": len(variants),
        "edit_groups": edit_groups,
        "top_variants": [variant_to_dict(row) for row in variants[: args.top_n]],
    }
    markdown = render_markdown(payload)
    output = resolve_path(Path(args.output)) if args.output else agenda_path.with_suffix(".repair_variants.md")
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


def build_edit_groups(
    agenda: dict[str, Any],
    cipher: Any,
    *,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    max_symbols: int,
    max_alternatives: int,
) -> list[dict[str, Any]]:
    symbol_pressure: dict[str, int] = {}
    for window in agenda.get("repair_windows") or []:
        for item in window.get("disputed_symbols") or []:
            symbol = str(item.get("symbol") or "")
            if symbol:
                symbol_pressure[symbol] = symbol_pressure.get(symbol, 0) + int(item.get("count") or 0)
    disputed = {
        str(item.get("symbol")): item
        for item in agenda.get("most_disputed_symbols") or []
        if item.get("symbol")
    }
    ranked_symbols = sorted(
        disputed,
        key=lambda symbol: (
            -symbol_pressure.get(symbol, 0),
            float(disputed[symbol].get("agreement") or 1.0),
            symbol,
        ),
    )[: max(0, max_symbols)]
    groups: list[dict[str, Any]] = []
    for symbol in ranked_symbols:
        if not cipher.alphabet.has_symbol(symbol):
            continue
        token_id = cipher.alphabet.id_for(symbol)
        current = current_assignment(symbol, token_id, baseline_key, baseline_mask)
        counts = disputed[symbol].get("counts") if isinstance(disputed[symbol].get("counts"), dict) else {}
        alternatives = [
            str(value)
            for value, _count in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
            if str(value) != current
        ][: max(0, max_alternatives)]
        if not alternatives:
            continue
        groups.append({
            "symbol": symbol,
            "token_id": token_id,
            "current": current,
            "pressure": symbol_pressure.get(symbol, 0),
            "alternatives": alternatives,
        })
    return groups


def evaluate_variants(
    *,
    cipher: Any,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    edit_groups: list[dict[str, Any]],
    agenda: dict[str, Any],
    binary_model_path: Path | None,
    include_pairs: bool,
) -> list[RepairVariant]:
    edits: list[tuple[dict[str, Any], str]] = []
    for group in edit_groups:
        for alternative in group["alternatives"]:
            edits.append((group, alternative))
    edit_sets: list[tuple[tuple[dict[str, Any], str], ...]] = [()]
    edit_sets.extend((edit,) for edit in edits)
    if include_pairs:
        edit_sets.extend(tuple(pair) for pair in itertools.combinations(edits, 2))
    variants: list[RepairVariant] = []
    seen: set[tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = set()
    for edit_set in edit_sets:
        mask = set(baseline_mask)
        key = dict(baseline_key)
        edit_labels = []
        for group, alternative in edit_set:
            symbol = group["symbol"]
            token_id = int(group["token_id"])
            before = current_assignment(symbol, token_id, key, tuple(sorted(mask)))
            apply_assignment(symbol, token_id, alternative, key, mask)
            edit_labels.append(f"{symbol}:{before}->{alternative}")
        identity = (tuple(sorted(mask)), tuple(sorted(key.items())))
        if identity in seen:
            continue
        seen.add(identity)
        variant = score_variant(
            cipher=cipher,
            key=key,
            mask=tuple(sorted(mask)),
            edits=tuple(edit_labels) or ("baseline",),
            agenda=agenda,
            binary_model_path=binary_model_path,
        )
        variants.append(variant)
    return variants


def score_variant(
    *,
    cipher: Any,
    key: dict[int, int],
    mask: tuple[str, ...],
    edits: tuple[str, ...],
    agenda: dict[str, Any],
    binary_model_path: Path | None,
) -> RepairVariant:
    candidate = _CandidateAdapter(mask=mask, key=key, decryption="")
    text, _sources = reconstruct_candidate(candidate, cipher)
    quality = _plaintext_quality(text, key)
    diagnostics = _automated_candidate_diagnostics(
        text,
        language="de",
        word_list=_word_list("de"),
        binary_model_path=binary_model_path,
    )
    row = {
        "mask": list(mask),
        "filtered_length": len(text),
        "selection_score": 0.0,
        "quality": quality,
        "diagnostics": diagnostics,
        "decryption": text,
        "preview": text,
    }
    validation = null_mask_validation_score_v2(
        row,
        original_length=len(cipher.tokens),
        language="de",
    )
    features = language_quality_feature_dict(
        text,
        diagnostics=diagnostics,
        language="de",
        original_length=len(cipher.tokens),
        filtered_length=len(text),
        mask_size=len(mask),
    )
    window_damage = mean_repair_window_damage(text, agenda)
    return RepairVariant(
        edits=edits,
        mask=mask,
        key=key,
        text=text,
        validation_score_v2=float(validation["score"]),
        validation_components_v2=dict(validation["components"]),
        language_quality_mean=sum(float(v) for v in features.values()) / max(1, len(features)),
        repair_window_damage=window_damage,
        diagnostics=diagnostics,
        quality=quality,
    )


def mean_repair_window_damage(text: str, agenda: dict[str, Any]) -> float:
    damages = []
    for window in agenda.get("repair_windows") or []:
        start = int(window.get("start") or 0)
        end = int(window.get("end") or start)
        if end <= start:
            continue
        snippet = text[start:min(end, len(text))]
        if not snippet:
            continue
        features = language_quality_feature_dict(snippet, language="de")
        damages.append(window_damage_score(features))
    return sum(damages) / max(1, len(damages))


def apply_assignment(
    symbol: str,
    token_id: int,
    assignment: str,
    key: dict[int, int],
    mask: set[str],
) -> None:
    if assignment == "<null>":
        mask.add(symbol)
        return
    mask.discard(symbol)
    if len(assignment) == 1 and "A" <= assignment <= "Z":
        key[token_id] = ord(assignment) - ord("A")


def current_assignment(
    symbol: str,
    token_id: int,
    key: dict[int, int],
    mask: tuple[str, ...],
) -> str:
    if symbol in set(mask):
        return "<null>"
    value = key.get(token_id)
    if value is None or value < 0 or value > 25:
        return "?"
    return chr(ord("A") + value)


def load_selected_row(agenda: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    selected = agenda.get("selected") if isinstance(agenda.get("selected"), dict) else {}
    artifact = resolve_path(Path(str(selected.get("artifact") or "")))
    if not artifact.exists():
        raise SystemExit(f"Selected artifact not found: {artifact}")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    null_step = next(
        (
            step for step in payload.get("steps") or []
            if isinstance(step, dict) and step.get("name") == "search_null_masks"
        ),
        None,
    )
    if not isinstance(null_step, dict):
        raise SystemExit(f"Selected artifact has no search_null_masks step: {artifact}")
    rows = []
    if isinstance(null_step.get("selected"), dict):
        rows.append(null_step["selected"])
    rows.extend(row for row in (null_step.get("top_finalists") or []) if isinstance(row, dict))
    selected_mask = tuple(str(symbol) for symbol in (selected.get("mask") or []))
    selected_source = str(selected.get("source") or "")
    selected_preview = str(selected.get("preview") or "")[:80]
    for row in rows:
        mask = tuple(str(symbol) for symbol in (row.get("mask") or []))
        source = str(row.get("source") or "")
        preview = str(row.get("decryption") or row.get("preview") or "")[:80]
        if mask == selected_mask and source == selected_source and preview == selected_preview:
            return row, artifact
    if rows:
        return rows[0], artifact
    raise SystemExit(f"No selected/top finalist rows found in {artifact}")


def binary_model_from_row_or_artifact(row: dict[str, Any], artifact: Path) -> Path | None:
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
    source = str(diagnostics.get("binary_ngram_model_source") or "")
    if not source:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        null_step = next(
            (
                step for step in payload.get("steps") or []
                if isinstance(step, dict) and step.get("name") == "search_null_masks"
            ),
            {},
        )
        model = null_step.get("binary_ngram_model") if isinstance(null_step, dict) else {}
        if isinstance(model, dict):
            source = str(model.get("path") or "")
    if not source:
        return None
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.exists() else None


def load_cipher(benchmark_root: Path, split: str, test_id: str) -> Any:
    loader = BenchmarkLoader(benchmark_root)
    tests = [test for test in loader.load_tests(split) if test.test_id == test_id]
    if not tests:
        raise SystemExit(f"Test not found in split {split}: {test_id}")
    data = loader.load_test_data(tests[0])
    return parse_canonical_transcription(data.canonical_transcription)


def variant_to_dict(row: RepairVariant) -> dict[str, Any]:
    return {
        "edits": list(row.edits),
        "mask": list(row.mask),
        "validation_score_v2": round(row.validation_score_v2, 6),
        "language_quality_mean": round(row.language_quality_mean, 6),
        "repair_window_damage": round(row.repair_window_damage, 6),
        "dict_rate": row.diagnostics.get("dict_rate"),
        "pseudo_word_fraction": row.diagnostics.get("pseudo_word_fraction"),
        "top_letter_fraction": row.quality.get("top_letter_fraction"),
        "preview": row.preview,
        "validation_components_v2": row.validation_components_v2,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Copiale Repair Variant Probe: {payload['test_id']}",
        "",
        "Ground truth is not used. Variants are direct local key/null edits from the repair agenda;",
        "they are ranked by runtime validation signals and are intended to seed targeted repair.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| variants | {payload['variant_count']} |",
        f"| selected artifact | `{payload['selected_artifact']}` |",
        "",
        "## Edit Groups",
        "",
        "| Symbol | Current | Pressure | Alternatives |",
        "|---|---|---:|---|",
    ]
    for group in payload["edit_groups"]:
        lines.append(
            f"| {group['symbol']} | {group['current']} | {group['pressure']} | "
            f"{', '.join(group['alternatives'])} |"
        )
    lines.extend([
        "",
        "## Top Variants",
        "",
        "| Rank | Edits | Val2 | LQ Mean | Window Damage | Dict | Top Letter | Preview |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for idx, row in enumerate(payload["top_variants"], start=1):
        lines.append(
            f"| {idx} | {escape_md('; '.join(row['edits']))} | "
            f"{row['validation_score_v2']:.3f} | {row['language_quality_mean']:.3f} | "
            f"{row['repair_window_damage']:.3f} | {format_optional(row.get('dict_rate'))} | "
            f"{format_optional(row.get('top_letter_fraction'))} | {escape_md(row['preview'])} |"
        )
    return "\n".join(lines).rstrip() + "\n"


class _CandidateAdapter:
    def __init__(self, *, mask: tuple[str, ...], key: dict[int, int], decryption: str) -> None:
        self.mask = mask
        self.key = key
        self.decryption = decryption


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def escape_md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def resolve_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


if __name__ == "__main__":
    main()
