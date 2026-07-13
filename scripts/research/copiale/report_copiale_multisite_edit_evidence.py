#!/usr/bin/env python3
"""Aggregate Copiale repair evidence by symbol edit across candidate menus.

This is an offline, ground-truth-free diagnostic report. It asks whether a
single symbol edit is supported by multiple damaged word contexts, rather than
being attractive because it repairs one isolated word island. Post-hoc character
accuracy is included only as calibration after the evidence has been collected.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CANDIDATE_LIST_KEYS = (
    "top_variants",
    "top_variants_by_adjudication",
    "top_variants_by_word_hypothesis",
    "top_variants_by_second_stage",
    "top_variants_by_second_stage_review",
    "top_variants_by_post_hoc",
    "top_accepted_variants",
    "top_combination_variants_by_adjudication",
    "top_combination_variants_by_marginal",
)


@dataclass
class EditEvidence:
    edit_key: str
    symbol: str
    target: str
    before_letters: set[str] = field(default_factory=set)
    candidate_count: int = 0
    source_files: set[str] = field(default_factory=set)
    labels: set[str] = field(default_factory=set)
    support_examples: set[str] = field(default_factory=set)
    support_hypothesis_count: int = 0
    support_page_count: int = 0
    support_window_count: int = 0
    competing_target_count: int = 0
    local_score_sum: float = 0.0
    best_multisite_score: float = float("-inf")
    best_post_hoc_char: float | None = None
    best_no_target_score: float | None = None
    best_adjudication_score: float | None = None
    best_global_leverage_score: float | None = None
    best_weighted_word_gain: float | None = None
    best_weighted_word_damage: float | None = None
    best_row: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        *,
        row: dict[str, Any],
        leverage: dict[str, Any],
        source_file: Path,
        label: str,
    ) -> None:
        self.candidate_count += 1
        self.source_files.add(str(source_file))
        if label:
            self.labels.add(label)
        support = leverage.get("support") if isinstance(leverage.get("support"), dict) else {}
        for example in support.get("examples") or []:
            self.support_examples.add(str(example))
        self.support_hypothesis_count = max(
            self.support_hypothesis_count,
            int(support.get("hypothesis_count") or 0),
        )
        self.support_page_count = max(self.support_page_count, int(support.get("page_count") or 0))
        self.support_window_count = max(self.support_window_count, int(support.get("window_count") or 0))
        self.competing_target_count = max(
            self.competing_target_count,
            int(support.get("competing_target_count") or 0),
        )
        self.local_score_sum = max(self.local_score_sum, float(support.get("local_score_sum") or 0.0))
        row_score = multisite_score(leverage)
        if row_score > self.best_multisite_score:
            adj = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
            self.best_multisite_score = row_score
            self.best_post_hoc_char = as_float(row.get("post_hoc_char_avg"))
            self.best_no_target_score = as_float(adj.get("adjudication_no_target_score"))
            self.best_adjudication_score = as_float(adj.get("adjudication_score"))
            self.best_global_leverage_score = as_float(leverage.get("global_leverage_score"))
            self.best_weighted_word_gain = as_float(leverage.get("weighted_word_gain"))
            self.best_weighted_word_damage = as_float(leverage.get("weighted_word_damage"))
            self.best_row = compact_row(row, leverage=leverage, source_file=source_file)

    @property
    def distinct_example_count(self) -> int:
        return len(self.support_examples)

    @property
    def distinct_observed_count(self) -> int:
        return len({example_observed_word(example) for example in self.support_examples})

    @property
    def multisite_flag(self) -> bool:
        return self.distinct_observed_count >= 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report multi-site support for Copiale candidate symbol edits."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Repair JSON files or directories.")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--second-stage-pool",
        type=int,
        default=12,
        help=(
            "Per-menu candidate pool, ranked by multi-site review score, to pass "
            "to the local-context second-stage reviewer."
        ),
    )
    parser.add_argument("--output", type=Path, help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, help="JSON output path.")
    args = parser.parse_args()

    paths = expand_json_paths(args.inputs)
    evidence = collect_evidence(paths)
    rows = sorted(
        [evidence_row(item) for item in evidence.values()],
        key=lambda row: (
            row["multisite"],
            row["multisite_score"],
            row["distinct_observed_words"],
            row["distinct_examples"],
            row["best_post_hoc_char"] or 0.0,
        ),
        reverse=True,
    )
    candidate_rows = build_candidate_rows(paths, evidence)
    candidate_shortlist = diverse_candidate_shortlist(candidate_rows, limit=max(args.top, 80))
    local_context_rows = local_context_second_stage_shortlist(
        candidate_rows,
        per_group_pool=max(1, args.second_stage_pool),
        limit=max(args.top, 80),
    )
    payload = {
        "experiment": "copiale_multisite_edit_evidence",
        "input_count": len(paths),
        "edit_count": len(rows),
        "rows": rows,
        "candidate_shortlist": candidate_shortlist,
        "local_context_second_stage": local_context_rows,
        "group_summaries": summarize_candidate_groups(candidate_rows),
        "local_context_group_summaries": summarize_local_context_groups(
            candidate_rows,
            per_group_pool=max(1, args.second_stage_pool),
        ),
        "local_context_mode_comparison": compare_local_context_modes(
            candidate_rows,
            per_group_pool=max(1, args.second_stage_pool),
        ),
        "second_stage_pool": max(1, args.second_stage_pool),
    }
    markdown = render_markdown(payload, top=args.top)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_output}")


def expand_json_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("*.word_hypothesis_repair.json")))
        elif item.is_file():
            paths.append(item)
    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def collect_evidence(paths: list[Path]) -> dict[str, EditEvidence]:
    evidence: dict[str, EditEvidence] = {}
    seen_rows: set[tuple[str, str, str, str]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = str(payload.get("label") or path.stem)
        for list_key in CANDIDATE_LIST_KEYS:
            rows = payload.get(list_key)
            if not isinstance(rows, list):
                continue
            for rank, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                row_identity = (
                    str(path),
                    list_key,
                    "|".join(str(edit) for edit in (row.get("edits") or [])),
                    str(rank),
                )
                if row_identity in seen_rows:
                    continue
                seen_rows.add(row_identity)
                edit_before = edit_before_map(row.get("edits") or [])
                adj = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
                for leverage in adj.get("symbol_leverage") or []:
                    if not isinstance(leverage, dict):
                        continue
                    symbol = str(leverage.get("symbol") or "")
                    target = str(leverage.get("target") or "")
                    if not symbol or not target:
                        continue
                    before = edit_before.get((symbol, target), "?")
                    edit_key = f"{symbol}:{before}->{target}"
                    item = evidence.get(edit_key)
                    if item is None:
                        item = EditEvidence(edit_key=edit_key, symbol=symbol, target=target)
                        evidence[edit_key] = item
                    if before:
                        item.before_letters.add(before)
                    item.update(row=row, leverage=leverage, source_file=path, label=label)
    return evidence


def build_candidate_rows(
    paths: list[Path],
    evidence: dict[str, EditEvidence],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = str(payload.get("label") or path.stem)
        for list_key in CANDIDATE_LIST_KEYS:
            rows = payload.get(list_key)
            if not isinstance(rows, list):
                continue
            for source_rank, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                edits = tuple(str(edit) for edit in (row.get("edits") or []))
                identity = (str(path), edits)
                if identity in seen:
                    continue
                seen.add(identity)
                score, components, edit_rows = candidate_multisite_review_score(row, evidence)
                candidates.append({
                    "source_file": str(path),
                    "label": label,
                    "source_list": list_key,
                    "source_rank": source_rank,
                    "edits": list(edits),
                    "review_score": round(score, 6),
                    "components": components,
                    "edit_evidence": edit_rows,
                    "post_hoc_char_avg": row.get("post_hoc_char_avg"),
                    "post_hoc_char_no_target_avg": row.get("post_hoc_char_no_target_avg"),
                    "adjudication_no_target_score": nested_float(row, "repair_adjudication", "adjudication_no_target_score"),
                    "adjudication_score": nested_float(row, "repair_adjudication", "adjudication_score"),
                    "local_context_review": local_context_review(row),
                    "preview": row.get("preview"),
                    "word_hypotheses": row.get("word_hypotheses") or [],
                })
    candidates.sort(
        key=lambda row: (
            row["review_score"],
            row["components"]["multisite_edit_count"],
            row["components"]["collateral_net"],
            row["components"]["adjudication_no_target"],
        ),
        reverse=True,
    )
    return candidates


def candidate_multisite_review_score(
    row: dict[str, Any],
    evidence: dict[str, EditEvidence],
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    edit_before = edit_before_map(row.get("edits") or [])
    adj = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    edit_rows: list[dict[str, Any]] = []
    multisite_edit_count = 0
    singleton_edit_count = 0
    collateral_net = 0.0
    collateral_damage = 0.0
    global_support = 0.0
    observed_support = 0
    competing_targets = 0
    for leverage in adj.get("symbol_leverage") or []:
        if not isinstance(leverage, dict):
            continue
        symbol = str(leverage.get("symbol") or "")
        target = str(leverage.get("target") or "")
        before = edit_before.get((symbol, target), "?")
        edit_key = f"{symbol}:{before}->{target}"
        item = evidence.get(edit_key)
        if item is None:
            continue
        gain = float(leverage.get("weighted_word_gain") or 0.0)
        damage = float(leverage.get("weighted_word_damage") or 0.0)
        local_collateral = gain - damage
        collateral_net += local_collateral
        collateral_damage += damage
        competing_targets += item.competing_target_count
        observed_support += item.distinct_observed_count
        if item.multisite_flag:
            multisite_edit_count += 1
            global_support += max(0.0, item.best_multisite_score)
        else:
            singleton_edit_count += 1
            global_support += min(1.5, max(0.0, item.best_multisite_score) * 0.2)
        edit_rows.append({
            "edit": edit_key,
            "multisite": item.multisite_flag,
            "distinct_observed_words": item.distinct_observed_count,
            "examples": sorted(item.support_examples)[:8],
            "global_multisite_score": round(item.best_multisite_score, 6),
            "collateral_gain_minus_damage": round(local_collateral, 6),
            "weighted_word_gain": round(gain, 6),
            "weighted_word_damage": round(damage, 6),
        })
    edited_symbol_count = len([edit for edit in (row.get("edits") or []) if str(edit).lower() != "baseline"])
    adjudication_no_target = float(adj.get("adjudication_no_target_score") or 0.0)
    adjudication_score = float(adj.get("adjudication_score") or 0.0)
    bundle_penalty = max(0, edited_symbol_count - 1) * 1.2
    collateral_damage_penalty = max(0.0, collateral_damage - max(0.8, 0.55 * max(1, multisite_edit_count)))
    target_only_penalty = max(0.0, float(adj.get("target_only_penalty") or 0.0))
    score = (
        2.8 * multisite_edit_count
        + 0.24 * global_support
        + 0.45 * observed_support
        + 1.05 * collateral_net
        + 0.75 * adjudication_no_target
        + 0.18 * adjudication_score
        - 0.45 * competing_targets
        - 1.35 * collateral_damage_penalty
        - 1.10 * target_only_penalty
        - bundle_penalty
    )
    components = {
        "multisite_edit_count": multisite_edit_count,
        "singleton_edit_count": singleton_edit_count,
        "edited_symbol_count": edited_symbol_count,
        "observed_support": observed_support,
        "global_support": round(global_support, 6),
        "collateral_net": round(collateral_net, 6),
        "collateral_damage": round(collateral_damage, 6),
        "collateral_damage_penalty": round(collateral_damage_penalty, 6),
        "adjudication_no_target": round(adjudication_no_target, 6),
        "adjudication_score": round(adjudication_score, 6),
        "competing_targets": competing_targets,
        "bundle_penalty": round(bundle_penalty, 6),
        "target_only_penalty": round(target_only_penalty, 6),
    }
    return score, components, edit_rows


def local_context_review(row: dict[str, Any]) -> dict[str, Any]:
    """Score target and collateral windows for a repair candidate.

    This score is intentionally local and ground-truth-free. It rewards target
    word repairs only when the surrounding occurrence impacts do not show the
    same edit damaging many non-target windows.
    """
    adj = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    impacts = adj.get("occurrence_impacts") if isinstance(adj.get("occurrence_impacts"), list) else []
    target_gain = 0.0
    collateral_gain = 0.0
    collateral_damage = 0.0
    target_sites = 0
    collateral_good = 0
    collateral_bad = 0
    changed_tests: set[str] = set()
    windows: list[dict[str, Any]] = []
    for item in impacts:
        if not isinstance(item, dict):
            continue
        delta = float(item.get("word_evidence_delta") or 0.0)
        quality_delta = float(item.get("quality_delta") or 0.0)
        score_delta = delta + 0.35 * quality_delta
        before = str(item.get("before") or "")
        after = str(item.get("after") or "")
        before_word = word_evidence_word(item.get("before_word_evidence"))
        after_word = word_evidence_word(item.get("after_word_evidence"))
        is_target = bool(item.get("in_hypothesis_target"))
        if before != after:
            changed_tests.add(str(item.get("test_id") or ""))
        if is_target:
            if score_delta > 0:
                target_sites += 1
            target_gain += max(0.0, score_delta)
        else:
            if score_delta > 0:
                collateral_good += 1
                collateral_gain += score_delta
            elif score_delta < 0:
                collateral_bad += 1
                collateral_damage += -score_delta
        if score_delta != 0 or is_target:
            windows.append({
                "test_id": item.get("test_id"),
                "target": is_target,
                "symbol": item.get("symbol"),
                "letter": item.get("target"),
                "delta": round(score_delta, 6),
                "before_word": before_word,
                "after_word": after_word,
                "before": before,
                "after": after,
            })
    target_only_penalty = float(adj.get("target_only_penalty") or 0.0)
    edited_symbol_count = int(adj.get("edited_symbol_count") or 0)
    bundle_penalty = max(0, edited_symbol_count - 1) * 0.45
    damage_pressure = collateral_damage / max(1, collateral_bad)
    score = (
        1.2 * target_gain
        + 0.55 * collateral_gain
        + 0.75 * target_sites
        + 0.25 * collateral_good
        - 1.35 * collateral_damage
        - 0.75 * collateral_bad
        - 1.1 * damage_pressure
        - 0.8 * target_only_penalty
        - bundle_penalty
    )
    windows.sort(
        key=lambda item: (
            item["target"],
            abs(float(item.get("delta") or 0.0)),
            str(item.get("test_id") or ""),
        ),
        reverse=True,
    )
    return {
        "score": round(score, 6),
        "target_gain": round(target_gain, 6),
        "target_sites": target_sites,
        "collateral_gain": round(collateral_gain, 6),
        "collateral_damage": round(collateral_damage, 6),
        "collateral_good": collateral_good,
        "collateral_bad": collateral_bad,
        "changed_test_count": len([test_id for test_id in changed_tests if test_id]),
        "target_only_penalty": round(target_only_penalty, 6),
        "bundle_penalty": round(bundle_penalty, 6),
        "windows": windows[:12],
    }


def word_evidence_word(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    observed = str(value.get("observed") or "")
    word = str(value.get("word") or "")
    if observed and word:
        return f"{observed}->{word}"
    return observed or word


def local_context_second_stage_score(row: dict[str, Any]) -> float:
    return local_context_mode_score(row, mode="strict_local")


def local_context_mode_score(row: dict[str, Any], *, mode: str) -> float:
    review = float(row.get("review_score") or 0.0)
    local = row.get("local_context_review") if isinstance(row.get("local_context_review"), dict) else {}
    local_score = float(local.get("score") or 0.0)
    components = row.get("components") if isinstance(row.get("components"), dict) else {}
    multisite_count = float(components.get("multisite_edit_count") or 0.0)
    collateral_net = float(components.get("collateral_net") or 0.0)
    target_gain = float(local.get("target_gain") or 0.0)
    collateral_gain = float(local.get("collateral_gain") or 0.0)
    collateral_damage = float(local.get("collateral_damage") or 0.0)
    target_sites = float(local.get("target_sites") or 0.0)
    collateral_bad = float(local.get("collateral_bad") or 0.0)
    if mode == "review_only":
        score = review
    elif mode == "strict_local":
        score = 0.55 * review + 1.45 * local_score + 0.6 * multisite_count + 0.25 * collateral_net
    elif mode == "soft_veto":
        score = (
            review
            + 0.6 * target_gain
            + 0.12 * collateral_gain
            - 0.20 * collateral_damage
            + 0.4 * target_sites
            - 0.15 * collateral_bad
        )
    elif mode == "target_first":
        score = (
            review
            + 1.1 * target_gain
            + 0.25 * target_sites
            + 0.08 * collateral_gain
            - 0.12 * collateral_damage
        )
    elif mode == "review_plus_target":
        score = review + 0.8 * target_gain + 0.35 * target_sites
    else:
        raise ValueError(f"unknown local-context mode: {mode}")
    return round(score, 6)


def local_context_second_stage_shortlist(
    rows: list[dict[str, Any]],
    *,
    per_group_pool: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group in candidate_groups(rows).values():
        pool = sorted(
            group,
            key=lambda row: (
                row["review_score"],
                row["components"]["multisite_edit_count"],
                row["components"]["collateral_net"],
            ),
            reverse=True,
        )[:per_group_pool]
        selected.extend(pool)
    for row in selected:
        row["local_context_second_stage_score"] = local_context_second_stage_score(row)
    ranked = diverse_candidate_shortlist_by_key(
        selected,
        key=lambda row: (
            row.get("local_context_second_stage_score") or 0.0,
            (row.get("local_context_review") or {}).get("score") or 0.0,
            row.get("review_score") or 0.0,
        ),
        limit=limit,
    )
    return ranked


def diverse_candidate_shortlist_by_key(
    rows: list[dict[str, Any]],
    *,
    key: Any,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=key, reverse=True)
    out: list[dict[str, Any]] = []
    seen_edit_families: set[str] = set()
    for row in ranked:
        family = edit_family(row.get("edits") or [])
        if family in seen_edit_families:
            continue
        out.append(row)
        seen_edit_families.add(family)
        if len(out) >= limit:
            return out
    for row in ranked:
        if row in out:
            continue
        out.append(row)
        if len(out) >= limit:
            break
    return sorted(out, key=key, reverse=True)


def candidate_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("source_file") or ""), []).append(row)
    return groups


def diverse_candidate_shortlist(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_edit_families: set[str] = set()
    for row in rows:
        family = edit_family(row.get("edits") or [])
        if family in seen_edit_families:
            continue
        out.append(row)
        seen_edit_families.add(family)
        if len(out) >= limit:
            return out
    for row in rows:
        if row in out:
            continue
        out.append(row)
        if len(out) >= limit:
            break
    return sorted(
        out,
        key=lambda row: (
            row["review_score"],
            row["components"]["multisite_edit_count"],
            row["components"]["collateral_net"],
            row["components"]["adjudication_no_target"],
        ),
        reverse=True,
    )


def summarize_candidate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = candidate_groups(rows)
    summaries = []
    for key, group in sorted(groups.items()):
        ranked = sorted(
            group,
            key=lambda row: (
                row["review_score"],
                row["components"]["multisite_edit_count"],
                row["components"]["collateral_net"],
                row["components"]["adjudication_no_target"],
            ),
            reverse=True,
        )
        best_posthoc = max(
            ranked,
            key=lambda row: float(row.get("post_hoc_char_avg") or 0.0),
            default={},
        )
        best_review = ranked[0] if ranked else {}
        best_rank = next(
            (
                index
                for index, row in enumerate(ranked, start=1)
                if row is best_posthoc
            ),
            None,
        )
        summaries.append({
            "source_file": key,
            "label": best_review.get("label") or best_posthoc.get("label"),
            "candidate_count": len(group),
            "best_review_score": best_review.get("review_score"),
            "best_review_post_hoc_char": best_review.get("post_hoc_char_avg"),
            "best_review_edits": best_review.get("edits") or [],
            "best_post_hoc_char": best_posthoc.get("post_hoc_char_avg"),
            "best_post_hoc_rank_by_review": best_rank,
            "best_post_hoc_edits": best_posthoc.get("edits") or [],
            "top3_capture": bool(best_rank is not None and best_rank <= 3),
            "top8_capture": bool(best_rank is not None and best_rank <= 8),
        })
    return summaries


def summarize_local_context_groups(
    rows: list[dict[str, Any]],
    *,
    per_group_pool: int,
) -> list[dict[str, Any]]:
    summaries = []
    for key, group in sorted(candidate_groups(rows).items()):
        review_pool = sorted(
            group,
            key=lambda row: (
                row["review_score"],
                row["components"]["multisite_edit_count"],
                row["components"]["collateral_net"],
            ),
            reverse=True,
        )[:per_group_pool]
        for row in review_pool:
            row["local_context_second_stage_score"] = local_context_second_stage_score(row)
        ranked = sorted(
            review_pool,
            key=lambda row: (
                row.get("local_context_second_stage_score") or 0.0,
                (row.get("local_context_review") or {}).get("score") or 0.0,
                row.get("review_score") or 0.0,
            ),
            reverse=True,
        )
        best_posthoc = max(
            review_pool,
            key=lambda row: float(row.get("post_hoc_char_avg") or 0.0),
            default={},
        )
        best_local = ranked[0] if ranked else {}
        best_rank = next(
            (
                index
                for index, row in enumerate(ranked, start=1)
                if row is best_posthoc
            ),
            None,
        )
        summaries.append({
            "source_file": key,
            "label": best_local.get("label") or best_posthoc.get("label"),
            "pool_size": len(review_pool),
            "best_local_score": best_local.get("local_context_second_stage_score"),
            "best_local_context_score": (best_local.get("local_context_review") or {}).get("score"),
            "best_local_post_hoc_char": best_local.get("post_hoc_char_avg"),
            "best_local_edits": best_local.get("edits") or [],
            "best_post_hoc_char": best_posthoc.get("post_hoc_char_avg"),
            "best_post_hoc_rank_by_local": best_rank,
            "best_post_hoc_edits": best_posthoc.get("edits") or [],
            "top3_capture": bool(best_rank is not None and best_rank <= 3),
            "top8_capture": bool(best_rank is not None and best_rank <= 8),
        })
    return summaries


def compare_local_context_modes(
    rows: list[dict[str, Any]],
    *,
    per_group_pool: int,
) -> list[dict[str, Any]]:
    modes = ["review_only", "strict_local", "soft_veto", "target_first", "review_plus_target"]
    comparisons: list[dict[str, Any]] = []
    for mode in modes:
        top3 = 0
        top8 = 0
        gaps: list[float] = []
        picks = []
        for key, group in sorted(candidate_groups(rows).items()):
            review_pool = sorted(
                group,
                key=lambda row: (
                    row["review_score"],
                    row["components"]["multisite_edit_count"],
                    row["components"]["collateral_net"],
                ),
                reverse=True,
            )[:per_group_pool]
            ranked = sorted(
                review_pool,
                key=lambda row: local_context_mode_score(row, mode=mode),
                reverse=True,
            )
            best_posthoc = max(
                review_pool,
                key=lambda row: float(row.get("post_hoc_char_avg") or 0.0),
                default={},
            )
            best_rank = next(
                (index for index, row in enumerate(ranked, start=1) if row is best_posthoc),
                None,
            )
            pick = ranked[0] if ranked else {}
            if best_rank is not None and best_rank <= 3:
                top3 += 1
            if best_rank is not None and best_rank <= 8:
                top8 += 1
            gaps.append(
                float(best_posthoc.get("post_hoc_char_avg") or 0.0)
                - float(pick.get("post_hoc_char_avg") or 0.0)
            )
            picks.append({
                "source_file": key,
                "pick_edits": pick.get("edits") or [],
                "pick_post_hoc_char": pick.get("post_hoc_char_avg"),
                "best_post_hoc_char": best_posthoc.get("post_hoc_char_avg"),
                "best_rank": best_rank,
            })
        comparisons.append({
            "mode": mode,
            "menus": len(picks),
            "top3_capture": top3,
            "top8_capture": top8,
            "mean_gap": round(sum(gaps) / len(gaps), 6) if gaps else None,
            "picks": picks,
        })
    return comparisons


def edit_family(edits: list[Any]) -> str:
    symbols = []
    for edit in edits:
        parsed = parse_edit(str(edit))
        if parsed:
            symbols.append(parsed[0])
    return "+".join(sorted(symbols)) or "baseline"


def edit_before_map(edits: list[Any]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for edit in edits:
        parsed = parse_edit(str(edit))
        if parsed is None:
            continue
        symbol, before, target = parsed
        out[(symbol, target)] = before
    return out


def parse_edit(edit: str) -> tuple[str, str, str] | None:
    if edit.strip().lower() == "baseline":
        return None
    if ":" not in edit or "->" not in edit:
        return None
    symbol, rest = edit.split(":", 1)
    before, target = rest.split("->", 1)
    return symbol.strip(), before.strip(), target.strip()


def multisite_score(leverage: dict[str, Any]) -> float:
    support = leverage.get("support") if isinstance(leverage.get("support"), dict) else {}
    examples = set(str(item) for item in (support.get("examples") or []))
    observed_count = len({example_observed_word(example) for example in examples})
    hypothesis_count = float(support.get("hypothesis_count") or 0.0)
    page_count = float(support.get("page_count") or 0.0)
    window_count = float(support.get("window_count") or 0.0)
    local_score = float(support.get("local_score_sum") or 0.0)
    competing = float(support.get("competing_target_count") or 0.0)
    weighted_gain = float(leverage.get("weighted_word_gain") or 0.0)
    weighted_damage = float(leverage.get("weighted_word_damage") or 0.0)
    window_gain = float(leverage.get("window_gain") or 0.0)
    window_damage = float(leverage.get("window_damage") or 0.0)
    global_leverage = float(leverage.get("global_leverage_score") or 0.0)
    multisite_bonus = 4.0 if observed_count >= 2 else 0.0
    return (
        multisite_bonus
        + 0.45 * local_score
        + 0.8 * hypothesis_count
        + 0.8 * page_count
        + 0.4 * window_count
        + 0.8 * (weighted_gain - weighted_damage)
        + 0.6 * (window_gain - window_damage)
        + 0.5 * global_leverage
        - 0.9 * competing
    )


def evidence_row(item: EditEvidence) -> dict[str, Any]:
    return {
        "edit": item.edit_key,
        "symbol": item.symbol,
        "target": item.target,
        "before_letters": sorted(item.before_letters),
        "multisite": item.multisite_flag,
        "multisite_score": round(item.best_multisite_score, 6),
        "candidate_count": item.candidate_count,
        "labels": sorted(item.labels),
        "distinct_examples": item.distinct_example_count,
        "distinct_observed_words": item.distinct_observed_count,
        "support_examples": sorted(item.support_examples),
        "support_hypothesis_count": item.support_hypothesis_count,
        "support_page_count": item.support_page_count,
        "support_window_count": item.support_window_count,
        "competing_target_count": item.competing_target_count,
        "local_score_sum": round(item.local_score_sum, 6),
        "best_post_hoc_char": item.best_post_hoc_char,
        "best_adjudication_score": item.best_adjudication_score,
        "best_no_target_score": item.best_no_target_score,
        "best_global_leverage_score": item.best_global_leverage_score,
        "best_weighted_word_gain": item.best_weighted_word_gain,
        "best_weighted_word_damage": item.best_weighted_word_damage,
        "best_row": item.best_row,
    }


def compact_row(row: dict[str, Any], *, leverage: dict[str, Any], source_file: Path) -> dict[str, Any]:
    return {
        "source_file": str(source_file),
        "edits": row.get("edits") or [],
        "word_hypotheses": row.get("word_hypotheses") or [],
        "preview": row.get("preview"),
        "post_hoc_char_avg": row.get("post_hoc_char_avg"),
        "symbol_leverage": leverage,
    }


def render_markdown(payload: dict[str, Any], *, top: int) -> str:
    rows = payload["rows"][:top]
    lines = [
        "# Copiale Multi-Site Edit Evidence",
        "",
        "This report is ground-truth-free for ranking. Post-hoc character accuracy is calibration only.",
        "",
        f"- Input JSON files: `{payload['input_count']}`",
        f"- Distinct edits: `{payload['edit_count']}`",
        "",
        "## Candidate Review Shortlist",
        "",
        "Candidates are ranked by independent multi-word edit support plus collateral health. Ground truth is shown only after ranking.",
        "",
        "### Per-Menu Calibration",
        "",
        "| Menu | Candidates | Review Pick | Review Char | Best Char | Best Char Rank | Top-3 | Top-8 |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for row in payload.get("group_summaries", []):
        lines.append(
            f"| `{Path(str(row.get('source_file') or '')).name}` | "
            f"{row.get('candidate_count', 0)} | "
            f"{escape_cell('<br>'.join(f'`{edit}`' for edit in (row.get('best_review_edits') or [])))} | "
            f"{format_pct(row.get('best_review_post_hoc_char'))} | "
            f"{format_pct(row.get('best_post_hoc_char'))} | "
            f"{row.get('best_post_hoc_rank_by_review') or ''} | "
            f"{'Y' if row.get('top3_capture') else 'n'} | "
            f"{'Y' if row.get('top8_capture') else 'n'} |"
        )
    lines.extend([
        "",
        "### Local-Context Second Stage",
        "",
        (
            "This stage reranks only the top multi-site review pool per menu "
            f"(`{payload.get('second_stage_pool')}` candidates each). It scores "
            "the repaired target windows against observed collateral windows, "
            "without using ground truth."
        ),
        "",
        "| Menu | Pool | Local Pick | Local Char | Best Char In Pool | Best Rank | Top-3 | Top-8 |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ])
    for row in payload.get("local_context_group_summaries", []):
        lines.append(
            f"| `{Path(str(row.get('source_file') or '')).name}` | "
            f"{row.get('pool_size', 0)} | "
            f"{escape_cell('<br>'.join(f'`{edit}`' for edit in (row.get('best_local_edits') or [])))} | "
            f"{format_pct(row.get('best_local_post_hoc_char'))} | "
            f"{format_pct(row.get('best_post_hoc_char'))} | "
            f"{row.get('best_post_hoc_rank_by_local') or ''} | "
            f"{'Y' if row.get('top3_capture') else 'n'} | "
            f"{'Y' if row.get('top8_capture') else 'n'} |"
        )
    lines.extend([
        "",
        "### Local-Context Global/Diverse Shortlist",
        "",
        "| Rank | Stage2 | Local | Review | Edits | Target Gain | Collateral Gain/Damage | Sites | Post-Hoc | Local Windows |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ])
    for index, row in enumerate(payload.get("local_context_second_stage", [])[:top], start=1):
        local = row.get("local_context_review") or {}
        windows = []
        for window in local.get("windows") or []:
            before_after = f"{window.get('before_word') or '?'}->{window.get('after_word') or '?'}"
            marker = "T" if window.get("target") else "C"
            windows.append(f"{marker}:{before_after} ({format_float(window.get('delta'))})")
        lines.append(
            f"| {index} | {format_float(row.get('local_context_second_stage_score'))} | "
            f"{format_float(local.get('score'))} | "
            f"{format_float(row.get('review_score'))} | "
            f"{escape_cell('<br>'.join(f'`{edit}`' for edit in (row.get('edits') or [])))} | "
            f"{format_float(local.get('target_gain'))} | "
            f"{format_float(local.get('collateral_gain'))}/{format_float(local.get('collateral_damage'))} | "
            f"{local.get('target_sites', 0)}T/{local.get('collateral_good', 0)}+/{local.get('collateral_bad', 0)}- | "
            f"{format_pct(row.get('post_hoc_char_avg'))} | "
            f"{escape_cell('<br>'.join(windows[:5]))} |"
        )
    lines.extend([
        "",
        "### Local-Context Calibration Modes",
        "",
        (
            "These rows compare alternative local-context blends over the same "
            "per-menu review pool. Post-hoc columns are calibration only; the "
            "runtime scores do not use ground truth."
        ),
        "",
        "| Mode | Menus | Top-3 Capture | Top-8 Capture | Mean Char Gap |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in payload.get("local_context_mode_comparison", []):
        lines.append(
            f"| `{row.get('mode')}` | "
            f"{row.get('menus', 0)} | "
            f"{row.get('top3_capture', 0)} | "
            f"{row.get('top8_capture', 0)} | "
            f"{format_pct(row.get('mean_gap'))} |"
        )
    lines.extend([
        "",
        "### Global/Diverse Shortlist",
        "",
        "| Rank | Score | Edits | Multi Edits | Observed | Collateral Net | AdjNoTarget | Post-Hoc | Examples | Preview |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for index, row in enumerate(payload.get("candidate_shortlist", [])[:top], start=1):
        examples = []
        for edit in row.get("edit_evidence") or []:
            examples.extend(str(item) for item in (edit.get("examples") or [])[:2])
        components = row.get("components") or {}
        lines.append(
            f"| {index} | {format_float(row.get('review_score'))} | "
            f"{escape_cell('<br>'.join(f'`{edit}`' for edit in (row.get('edits') or [])))} | "
            f"{components.get('multisite_edit_count', 0)} | "
            f"{components.get('observed_support', 0)} | "
            f"{format_float(components.get('collateral_net'))} | "
            f"{format_float(row.get('adjudication_no_target_score'))} | "
            f"{format_pct(row.get('post_hoc_char_avg'))} | "
            f"{escape_cell('<br>'.join(examples[:6]))} | "
            f"{escape_cell(str(row.get('preview') or '')[:120])} |"
        )
    lines.extend([
        "",
        "## Top Edits",
        "",
        "| Rank | Edit | Multi-site | Score | Observed | Examples | Hyp | Pages | Collateral Gain-Damage | Post-Hoc Best | AdjNoTarget | Preview |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for index, row in enumerate(rows, start=1):
        examples = "<br>".join(row["support_examples"][:5])
        gain = row.get("best_weighted_word_gain") or 0.0
        damage = row.get("best_weighted_word_damage") or 0.0
        lines.append(
            f"| {index} | `{row['edit']}` | {'Y' if row['multisite'] else 'n'} | "
            f"{format_float(row['multisite_score'])} | {row['distinct_observed_words']} | "
            f"{escape_cell(examples)} | "
            f"{row['support_hypothesis_count']} | {row['support_page_count']} | "
            f"{format_float(gain - damage)} | {format_pct(row.get('best_post_hoc_char'))} | "
            f"{format_float(row.get('best_no_target_score'))} | "
            f"{escape_cell(str((row.get('best_row') or {}).get('preview') or '')[:120])} |"
        )
    lines.extend(["", "## Strong Multi-Site Candidates", ""])
    strong = [
        row for row in payload["rows"]
        if row["multisite"] and row["distinct_observed_words"] >= 2 and row["multisite_score"] > 0
    ][:top]
    if not strong:
        lines.append("(none)")
    else:
        for row in strong:
            lines.append(f"### `{row['edit']}`")
            lines.append("")
            lines.append(f"- Score: `{format_float(row['multisite_score'])}`")
            lines.append(f"- Distinct observed damaged words: `{row['distinct_observed_words']}`")
            lines.append(f"- Support examples: {', '.join(f'`{x}`' for x in row['support_examples'][:8])}")
            lines.append(f"- Best post-hoc char: {format_pct(row.get('best_post_hoc_char'))}")
            lines.append(f"- AdjNoTarget: `{format_float(row.get('best_no_target_score'))}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nested_float(row: dict[str, Any], parent: str, child: str) -> float | None:
    value = row.get(parent)
    if not isinstance(value, dict):
        return None
    return as_float(value.get(child))


def format_pct(value: Any) -> str:
    number = as_float(value)
    return "" if number is None else f"{number * 100:.1f}%"


def format_float(value: Any) -> str:
    number = as_float(value)
    return "" if number is None else f"{number:.3f}"


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def example_observed_word(example: str) -> str:
    return str(example).split("->", 1)[0].strip()


if __name__ == "__main__":
    main()
