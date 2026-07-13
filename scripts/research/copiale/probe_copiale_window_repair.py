#!/usr/bin/env python3
"""Probe localized repair edits inside damaged Copiale windows.

This is a ground-truth-free experiment. It starts from the selected
null-mask finalist in a repair agenda, applies disputed-symbol edits only
inside damaged text windows, and ranks the resulting texts with runtime
language/validation signals. Unlike ``probe_copiale_repair_variants.py``,
these edits do not change the global key.
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
from probe_copiale_repair_variants import (  # noqa: E402
    binary_model_from_row_or_artifact,
    build_edit_groups,
    current_assignment,
    load_selected_row,
)
from report_copiale_repair_agenda import (  # noqa: E402
    parse_key,
    reconstruct_candidate,
    window_damage_score,
)


@dataclass(frozen=True)
class LocalizedVariant:
    edits: tuple[str, ...]
    scope: str
    window_start: int | None
    window_end: int | None
    text: str
    changed_positions: int
    deleted_positions: int
    validation_score_v2: float
    validation_components_v2: dict[str, float]
    language_quality_mean: float
    repair_window_damage: float
    scoped_window_damage: float | None
    diagnostics: dict[str, Any]
    quality: dict[str, Any]

    @property
    def preview(self) -> str:
        if self.window_start is None:
            return self.text[:180]
        start = max(0, min(self.window_start, len(self.text)) - 20)
        end = min(len(self.text), max(self.window_end or self.window_start, self.window_start) + 80)
        return self.text[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe damaged-window-only repair edits from a Copiale repair agenda."
    )
    parser.add_argument("repair_agenda_json", help="JSON produced by report_copiale_repair_agenda.py")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument("--max-symbols", type=int, default=7)
    parser.add_argument("--max-alternatives", type=int, default=4)
    parser.add_argument("--include-pairs", action="store_true")
    parser.add_argument(
        "--scope",
        choices=["per_window", "all_windows", "both"],
        default="both",
        help="Apply edits to each damaged window separately, all damaged windows at once, or both.",
    )
    parser.add_argument("--window-limit", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--require-edit-substring", default="")
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

    baseline_candidate = _CandidateAdapter(mask=baseline_mask, key=baseline_key, decryption="")
    baseline_text, baseline_sources = reconstruct_candidate(baseline_candidate, cipher)
    if not baseline_sources or len(baseline_sources) != len(baseline_text):
        raise SystemExit(
            "Could not reconstruct symbol sources for the selected finalist; "
            "localized repair requires key-derived text."
        )

    edit_groups = build_edit_groups(
        agenda,
        cipher,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        max_symbols=args.max_symbols,
        max_alternatives=args.max_alternatives,
    )
    windows = normalized_windows(agenda, limit=args.window_limit, text_length=len(baseline_text))
    variants = evaluate_variants(
        baseline_text=baseline_text,
        baseline_sources=baseline_sources,
        baseline_key=baseline_key,
        baseline_mask=baseline_mask,
        edit_groups=edit_groups,
        agenda=agenda,
        original_length=len(cipher.tokens),
        binary_model_path=binary_model_path,
        include_pairs=args.include_pairs,
        scope=args.scope,
        windows=windows,
        required_substring=args.require_edit_substring,
    )
    variants.sort(
        key=lambda row: (
            row.validation_score_v2,
            row.language_quality_mean,
            -row.repair_window_damage,
            row.changed_positions,
        ),
        reverse=True,
    )
    baseline_variant = next(
        (row for row in variants if row.edits == ("baseline",)),
        variants[0] if variants else None,
    )
    payload = {
        "test_id": test_id,
        "agenda": str(agenda_path),
        "selected_artifact": str(selected_artifact),
        "scope": args.scope,
        "window_count": len(windows),
        "baseline": variant_to_dict(baseline_variant) if baseline_variant is not None else {},
        "variant_count": len(variants),
        "edit_groups": edit_groups,
        "repair_windows": [
            {"start": start, "end": end, "text": baseline_text[start:end]}
            for start, end in windows
        ],
        "top_variants": [variant_to_dict(row) for row in variants[: args.top_n]],
    }
    markdown = render_markdown(payload)
    output = (
        resolve_path(Path(args.output))
        if args.output
        else agenda_path.with_suffix(".window_repair.md")
    )
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


def evaluate_variants(
    *,
    baseline_text: str,
    baseline_sources: list[str],
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
    edit_groups: list[dict[str, Any]],
    agenda: dict[str, Any],
    original_length: int,
    binary_model_path: Path | None,
    include_pairs: bool,
    scope: str,
    windows: list[tuple[int, int]],
    required_substring: str,
) -> list[LocalizedVariant]:
    edits: list[tuple[dict[str, Any], str]] = []
    for group in edit_groups:
        for alternative in group["alternatives"]:
            edits.append((group, alternative))
    edit_sets: list[tuple[tuple[dict[str, Any], str], ...]] = [()]
    edit_sets.extend((edit,) for edit in edits)
    if include_pairs:
        for pair in itertools.combinations(edits, 2):
            if pair[0][0]["symbol"] == pair[1][0]["symbol"]:
                continue
            edit_sets.append(tuple(pair))

    scopes: list[tuple[str, tuple[int, int] | None]] = []
    if scope in {"per_window", "both"}:
        scopes.extend((f"window:{start}-{end}", (start, end)) for start, end in windows)
    if scope in {"all_windows", "both"}:
        scopes.append(("all_windows", None))

    variants: list[LocalizedVariant] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for edit_set in edit_sets:
        labels = edit_labels(edit_set, baseline_key=baseline_key, baseline_mask=baseline_mask)
        if required_substring and not any(required_substring in label for label in labels):
            continue
        if not edit_set:
            text = baseline_text
            variant = score_text_variant(
                text=text,
                edits=("baseline",),
                scope="baseline",
                window=None,
                changed_positions=0,
                deleted_positions=0,
                agenda=agenda,
                original_length=original_length,
                mask_size=len(baseline_mask),
                key=baseline_key,
                binary_model_path=binary_model_path,
            )
            variants.append(variant)
            continue
        for scope_label, window in scopes:
            active_windows = windows if window is None else [window]
            text, changed, deleted = apply_localized_edits(
                baseline_text,
                baseline_sources,
                edit_set,
                active_windows,
            )
            if changed == 0 and deleted == 0:
                continue
            identity = (scope_label, labels, text)
            if identity in seen:
                continue
            seen.add(identity)
            variants.append(
                score_text_variant(
                    text=text,
                    edits=labels,
                    scope=scope_label,
                    window=window,
                    changed_positions=changed,
                    deleted_positions=deleted,
                    agenda=agenda,
                    original_length=original_length,
                    mask_size=len(baseline_mask),
                    key=baseline_key,
                    binary_model_path=binary_model_path,
                )
            )
    return variants


def edit_labels(
    edit_set: tuple[tuple[dict[str, Any], str], ...],
    *,
    baseline_key: dict[int, int],
    baseline_mask: tuple[str, ...],
) -> tuple[str, ...]:
    labels = []
    for group, alternative in edit_set:
        symbol = str(group["symbol"])
        token_id = int(group["token_id"])
        before = current_assignment(symbol, token_id, baseline_key, baseline_mask)
        labels.append(f"{symbol}:{before}->{alternative}")
    return tuple(labels)


def apply_localized_edits(
    text: str,
    sources: list[str],
    edit_set: tuple[tuple[dict[str, Any], str], ...],
    windows: list[tuple[int, int]],
) -> tuple[str, int, int]:
    replacements = {
        str(group["symbol"]): str(alternative)
        for group, alternative in edit_set
    }
    active = [False] * len(text)
    for start, end in windows:
        for index in range(max(0, start), min(len(text), end)):
            active[index] = True
    chars: list[str] = []
    changed = 0
    deleted = 0
    for index, char in enumerate(text):
        replacement = replacements.get(sources[index]) if active[index] else None
        if replacement is None:
            chars.append(char)
            continue
        if replacement == "<null>":
            deleted += 1
            continue
        if len(replacement) == 1 and "A" <= replacement <= "Z":
            chars.append(replacement)
            if replacement != char:
                changed += 1
            continue
        chars.append(char)
    return "".join(chars), changed, deleted


def score_text_variant(
    *,
    text: str,
    edits: tuple[str, ...],
    scope: str,
    window: tuple[int, int] | None,
    changed_positions: int,
    deleted_positions: int,
    agenda: dict[str, Any],
    original_length: int,
    mask_size: int,
    key: dict[int, int],
    binary_model_path: Path | None,
) -> LocalizedVariant:
    quality = _plaintext_quality(text, key)
    diagnostics = _automated_candidate_diagnostics(
        text,
        language="de",
        word_list=_word_list("de"),
        binary_model_path=binary_model_path,
    )
    row = {
        "mask": [],
        "filtered_length": len(text),
        "selection_score": 0.0,
        "quality": quality,
        "diagnostics": diagnostics,
        "decryption": text,
        "preview": text,
    }
    validation = null_mask_validation_score_v2(
        row,
        original_length=original_length,
        language="de",
    )
    features = language_quality_feature_dict(
        text,
        diagnostics=diagnostics,
        language="de",
        original_length=original_length,
        filtered_length=len(text),
        mask_size=mask_size,
    )
    repair_damage = mean_repair_window_damage(text, agenda)
    scoped_damage = None
    if window is not None:
        start, end = window
        scoped_damage = window_damage_for_text(text, start, end)
    return LocalizedVariant(
        edits=edits,
        scope=scope,
        window_start=window[0] if window is not None else None,
        window_end=window[1] if window is not None else None,
        text=text,
        changed_positions=changed_positions,
        deleted_positions=deleted_positions,
        validation_score_v2=float(validation["score"]),
        validation_components_v2=dict(validation["components"]),
        language_quality_mean=sum(float(v) for v in features.values()) / max(1, len(features)),
        repair_window_damage=repair_damage,
        scoped_window_damage=scoped_damage,
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
        damage = window_damage_for_text(text, start, end)
        if damage is not None:
            damages.append(damage)
    return sum(damages) / max(1, len(damages))


def window_damage_for_text(text: str, start: int, end: int) -> float | None:
    snippet = text[max(0, start):min(end, len(text))]
    if not snippet:
        return None
    features = language_quality_feature_dict(snippet, language="de")
    return window_damage_score(features)


def normalized_windows(
    agenda: dict[str, Any],
    *,
    limit: int,
    text_length: int,
) -> list[tuple[int, int]]:
    rows = []
    for window in agenda.get("repair_windows") or []:
        start = max(0, int(window.get("start") or 0))
        end = min(text_length, int(window.get("end") or start))
        if end > start:
            rows.append((start, end))
    return rows[: max(0, limit)]


def load_cipher(benchmark_root: Path, split: str, test_id: str) -> Any:
    loader = BenchmarkLoader(benchmark_root)
    tests = [test for test in loader.load_tests(split) if test.test_id == test_id]
    if not tests:
        raise SystemExit(f"Test not found in split {split}: {test_id}")
    data = loader.load_test_data(tests[0])
    return parse_canonical_transcription(data.canonical_transcription)


def variant_to_dict(row: LocalizedVariant) -> dict[str, Any]:
    return {
        "edits": list(row.edits),
        "scope": row.scope,
        "window_start": row.window_start,
        "window_end": row.window_end,
        "changed_positions": row.changed_positions,
        "deleted_positions": row.deleted_positions,
        "validation_score_v2": round(row.validation_score_v2, 6),
        "language_quality_mean": round(row.language_quality_mean, 6),
        "repair_window_damage": round(row.repair_window_damage, 6),
        "scoped_window_damage": round(row.scoped_window_damage, 6) if row.scoped_window_damage is not None else None,
        "dict_rate": row.diagnostics.get("dict_rate"),
        "pseudo_word_fraction": row.diagnostics.get("pseudo_word_fraction"),
        "top_letter_fraction": row.quality.get("top_letter_fraction"),
        "preview": row.preview,
        "text": row.text,
        "validation_components_v2": row.validation_components_v2,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    baseline = payload.get("baseline") or {}
    baseline_val = float(baseline.get("validation_score_v2") or 0.0)
    baseline_damage = float(baseline.get("repair_window_damage") or 0.0)
    lines = [
        f"# Copiale Localized Window Repair Probe: {payload['test_id']}",
        "",
        "Ground truth is not used. Edits are applied only inside damaged windows;",
        "global key mappings are left unchanged.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| variants | {payload['variant_count']} |",
        f"| windows | {payload['window_count']} |",
        f"| scope | `{payload['scope']}` |",
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
        "## Top Localized Variants",
        "",
        "| Rank | Scope | Edits | Val2 | Delta Val2 | Damage | Delta Damage | Changed | Deleted | Dict | Preview |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for idx, row in enumerate(payload["top_variants"], start=1):
        val = float(row["validation_score_v2"])
        damage = float(row["repair_window_damage"])
        lines.append(
            f"| {idx} | {escape_md(row['scope'])} | {escape_md('; '.join(row['edits']))} | "
            f"{val:.3f} | {val - baseline_val:+.3f} | "
            f"{damage:.3f} | {damage - baseline_damage:+.3f} | "
            f"{int(row['changed_positions'])} | {int(row['deleted_positions'])} | "
            f"{format_optional(row.get('dict_rate'))} | {escape_md(row['preview'])} |"
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
