#!/usr/bin/env python3
"""Summarize finalist-validation evidence from Decipher artifacts.

The report is intended for transform/pure-transposition research runs. It walks
artifact JSON files, finds finalist menus with `validation` blocks, and prints a
compact table showing whether validation changed ranking and whether selected
candidates look coherent or merely word-island-like.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Artifact JSON file(s) or directories to scan recursively.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "jsonl"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Maximum finalist rows to print in markdown/csv output.",
    )
    parser.add_argument(
        "--include-unvalidated",
        action="store_true",
        help="Include finalist rows that do not yet have validation blocks.",
    )
    args = parser.parse_args()

    rows = []
    for artifact in _iter_artifact_paths([Path(p) for p in args.paths]):
        rows.extend(_extract_rows(artifact, include_unvalidated=args.include_unvalidated))

    rows.sort(key=_row_sort_key)
    output = _render(rows, fmt=args.format, limit=max(1, args.limit))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


def _iter_artifact_paths(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            candidates = sorted(path.rglob("*.json"))
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def _extract_rows(path: Path, *, include_unvalidated: bool) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [{
            "artifact": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }]

    common = {
        "artifact": str(path),
        "test_id": payload.get("test_id") or payload.get("cipher_id") or path.parent.name,
        "solver": payload.get("solver"),
        "status": payload.get("status"),
        "char_accuracy": payload.get("char_accuracy"),
        "word_accuracy": payload.get("word_accuracy"),
    }
    rows: list[dict[str, Any]] = []
    for context, container in _candidate_containers(payload):
        selected_candidate_id = _selected_candidate_id(container)
        for idx, candidate in enumerate(container.get("top_candidates") or [], start=1):
            row = _candidate_row(
                candidate,
                common=common,
                context=context,
                menu_index=idx,
                selected_candidate_id=selected_candidate_id,
            )
            if include_unvalidated or row.get("validation_label"):
                rows.append(row)
        selected = container.get("selected")
        if isinstance(selected, dict) and not _selected_already_in_top(selected, container):
            row = _candidate_row(
                selected,
                common=common,
                context=f"{context}.selected",
                menu_index=1,
                selected_candidate_id=selected.get("candidate_id"),
            )
            row["selected"] = True
            if include_unvalidated or row.get("validation_label"):
                rows.append(row)
    return rows


def _candidate_containers(node: Any, path: str = "$") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(node, dict):
        if isinstance(node.get("top_candidates"), list):
            yield path, node
        if isinstance(node.get("finalist_review"), list):
            yield f"{path}.finalist_review", {"top_candidates": node.get("finalist_review")}
        for key, value in node.items():
            if key in {"top_candidates", "finalist_review"}:
                continue
            yield from _candidate_containers(value, f"{path}.{key}")
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from _candidate_containers(value, f"{path}[{idx}]")


def _selected_candidate_id(container: dict[str, Any]) -> str | None:
    selected = container.get("selected")
    if isinstance(selected, dict):
        value = selected.get("candidate_id")
        return str(value) if value is not None else None
    best = container.get("best_candidate")
    if isinstance(best, dict):
        value = best.get("candidate_id")
        return str(value) if value is not None else None
    return None


def _selected_already_in_top(selected: dict[str, Any], container: dict[str, Any]) -> bool:
    selected_id = selected.get("candidate_id")
    if selected_id is None:
        return False
    for candidate in container.get("top_candidates") or []:
        if str(candidate.get("candidate_id")) == str(selected_id):
            return True
    return False


def _candidate_row(
    candidate: dict[str, Any],
    *,
    common: dict[str, Any],
    context: str,
    menu_index: int,
    selected_candidate_id: str | None,
) -> dict[str, Any]:
    validation = candidate.get("validation")
    if not isinstance(validation, dict):
        ranking = candidate.get("ranking_score")
        if isinstance(ranking, dict):
            supporting = ranking.get("supporting")
            if isinstance(supporting, dict) and isinstance(supporting.get("validation"), dict):
                validation = supporting["validation"]
    validation = validation if isinstance(validation, dict) else {}
    segmentation = validation.get("segmentation") if isinstance(validation.get("segmentation"), dict) else {}
    integrity = validation.get("integrity") if isinstance(validation.get("integrity"), dict) else {}
    candidate_id = candidate.get("candidate_id")
    if candidate_id is None and isinstance(candidate.get("candidate"), dict):
        candidate_id = candidate["candidate"].get("candidate_id")
    rank = _num(candidate.get("rank"))
    rust_rank = _num(candidate.get("rust_rank"))
    if rust_rank is None:
        rust_rank = _num(candidate.get("candidate_index"))
        if rust_rank is not None:
            rust_rank += 1
    selected = False
    if selected_candidate_id is not None and candidate_id is not None:
        selected = str(candidate_id) == selected_candidate_id
    if context.endswith(".selected"):
        selected = True
    return {
        **common,
        "context": context,
        "menu_index": menu_index,
        "rank": rank if rank is not None else menu_index,
        "rust_rank": rust_rank,
        "rank_delta": (
            rust_rank - rank
            if rust_rank is not None and rank is not None else None
        ),
        "selected": selected,
        "candidate_id": candidate_id,
        "family": candidate.get("family") or (candidate.get("candidate") or {}).get("family"),
        "score": _num(candidate.get("score")),
        "selection_score": _num(candidate.get("selection_score")),
        "validated_selection_score": _num(candidate.get("validated_selection_score")),
        "validation_label": validation.get("validation_label"),
        "validation_score": _num(validation.get("validation_score")),
        "recommendation": validation.get("recommendation"),
        "strict_word_hit_score": _num(validation.get("strict_word_hit_score")),
        "strict_word_hit_char_coverage": _num(validation.get("strict_word_hit_char_coverage")),
        "dict_rate": _num(segmentation.get("dict_rate")),
        "cost_per_char": _num(segmentation.get("cost_per_char")),
        "pseudo_word_fraction": _num(segmentation.get("pseudo_word_fraction")),
        "integrity_label": integrity.get("integrity_label"),
        "damage_score": _num(integrity.get("damage_score")),
        "pseudo_char_fraction": _num(integrity.get("pseudo_char_fraction")),
        "suspicious_short_pseudo_count": _num(integrity.get("suspicious_short_pseudo_count")),
        "preview": _preview(candidate, validation),
    }


def _preview(candidate: dict[str, Any], validation: dict[str, Any]) -> str:
    for key in ("decoded_preview", "preview", "plaintext"):
        value = candidate.get(key)
        if isinstance(value, str) and value:
            return _squash(value, 80)
    segmentation = validation.get("segmentation")
    if isinstance(segmentation, dict):
        value = segmentation.get("segmented_preview")
        if isinstance(value, str):
            return _squash(value, 80)
    return ""


def _render(rows: list[dict[str, Any]], *, fmt: str, limit: int) -> str:
    if fmt == "jsonl":
        return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if fmt == "csv":
        fields = _fields()
        from io import StringIO
        out = StringIO()
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows[:limit]:
            writer.writerow(row)
        return out.getvalue()
    return _render_markdown(rows, limit=limit)


def _render_markdown(rows: list[dict[str, Any]], *, limit: int) -> str:
    out: list[str] = []
    validated_count = sum(1 for row in rows if row.get("validation_label"))
    out.append("# Finalist Validation Report\n\n")
    out.append(f"Rows scanned: {len(rows)}  \n")
    out.append(f"Rows with validation evidence: {validated_count}\n\n")
    if rows:
        by_label = Counter(str(row.get("validation_label") or "unlabelled") for row in rows)
        by_integrity = Counter(str(row.get("integrity_label") or "none") for row in rows)
        by_recommendation = Counter(str(row.get("recommendation") or "none") for row in rows)
        selected = [row for row in rows if row.get("selected")]
        rank_changes = [
            row for row in rows
            if row.get("rank_delta") not in {None, 0}
        ]
        out.append("## Summary\n\n")
        out.append("| Metric | Value |\n|---|---:|\n")
        out.append(f"| selected rows | {len(selected)} |\n")
        out.append(f"| rows with validation rank change | {len(rank_changes)} |\n")
        out.append(f"| coherent/plausible rows | {sum(by_label[label] for label in ('coherent_candidate', 'plausible_candidate'))} |\n")
        out.append(f"| weak/gibberish rows | {sum(by_label[label] for label in ('weak_word_islands', 'gibberish_or_wrong_family', 'empty'))} |\n")
        out.append("\n")
        out.append("Labels: " + ", ".join(f"`{k}`={v}" for k, v in sorted(by_label.items())) + "\n\n")
        out.append("Integrity: " + ", ".join(f"`{k}`={v}" for k, v in sorted(by_integrity.items())) + "\n\n")
        out.append("Recommendations: " + ", ".join(f"`{k}`={v}" for k, v in sorted(by_recommendation.items())) + "\n\n")
    out.append("## Top Rows\n\n")
    if not rows:
        out.append("_No validation rows found. Re-run after finalist validation is enabled, or use --include-unvalidated to inspect candidate menus._\n")
        return "".join(out)
    out.append("| Test | Ctx | Sel | Rank | Rust | Δ | Family | Label | Integrity | VScore | Damage | Strict | Dict | Pseudo | Preview |\n")
    out.append("|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|\n")
    for row in rows[:limit]:
        out.append(
            "| "
            f"`{_md(row.get('test_id'))}` | "
            f"`{_short_context(row.get('context'))}` | "
            f"{'Y' if row.get('selected') else ''} | "
            f"{_fmt(row.get('rank'), 0)} | "
            f"{_fmt(row.get('rust_rank'), 0)} | "
            f"{_fmt(row.get('rank_delta'), 0)} | "
            f"`{_md(row.get('family'))}` | "
            f"`{_md(row.get('validation_label'))}` | "
            f"`{_md(row.get('integrity_label'))}` | "
            f"{_fmt(row.get('validation_score'), 2)} | "
            f"{_fmt(row.get('damage_score'), 2)} | "
            f"{_fmt(row.get('strict_word_hit_score'), 2)} | "
            f"{_fmt(row.get('dict_rate'), 2)} | "
            f"{_fmt(row.get('pseudo_word_fraction'), 2)} | "
            f"{_md(row.get('preview'))} |\n"
        )
    if len(rows) > limit:
        out.append(f"\n_Showing {limit} of {len(rows)} rows. Use `--limit` or `--format jsonl` for more._\n")
    return "".join(out)


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    selected_sort = 0 if row.get("selected") else 1
    label_order = {
        "coherent_candidate": 0,
        "plausible_candidate": 1,
        "weak_word_islands": 2,
        "gibberish_or_wrong_family": 3,
        "empty": 4,
    }
    return (
        str(row.get("test_id") or ""),
        selected_sort,
        label_order.get(str(row.get("validation_label") or ""), 9),
        -float(row.get("validation_score") or 0.0),
        int(row.get("rank") or 999999),
    )


def _fields() -> list[str]:
    return [
        "artifact",
        "test_id",
        "solver",
        "status",
        "char_accuracy",
        "word_accuracy",
        "context",
        "selected",
        "rank",
        "rust_rank",
        "rank_delta",
        "candidate_id",
        "family",
        "score",
        "selection_score",
        "validated_selection_score",
        "validation_label",
        "validation_score",
        "recommendation",
        "strict_word_hit_score",
        "strict_word_hit_char_coverage",
        "dict_rate",
        "cost_per_char",
        "pseudo_word_fraction",
        "integrity_label",
        "damage_score",
        "pseudo_char_fraction",
        "suspicious_short_pseudo_count",
        "preview",
    ]


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        if value is None:
            return ""
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if digits == 0:
        return str(int(f))
    return f"{f:.{digits}f}"


def _squash(text: str, max_len: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _short_context(value: Any) -> str:
    text = str(value or "")
    text = text.replace("$.steps", "steps")
    text = text.replace("$.transform_search", "transform_search")
    text = text.replace("$.automated_preflight", "preflight")
    return _squash(text, 42)


if __name__ == "__main__":
    raise SystemExit(main())
