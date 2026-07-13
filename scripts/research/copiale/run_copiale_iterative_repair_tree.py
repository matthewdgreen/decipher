#!/usr/bin/env python3
"""Iterative local repair tree experiment for Copiale multi-page candidates.

This is an offline, local-only experiment. It repeatedly proposes word-level
repair edits from the current candidate text, ranks them with ground-truth-free
runtime/local signals, prunes candidates whose selected local score drops too
far, and then expands the surviving branches. Benchmark plaintext is used only
in the final report fields already produced by the shared candidate evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from benchmark.loader import BenchmarkLoader  # noqa: E402
from probe_copiale_multipage_global_repair import (  # noqa: E402
    annotate_acceptance,
    annotate_repair_evidence,
    apply_assignment,
    pages_to_alphabet,
    parse_key,
    resolve_path,
    selected_label_from_section,
    sibling_artifact_path,
    variant_summary,
)
from probe_copiale_word_hypothesis_repair import (  # noqa: E402
    attach_second_stage_portfolio,
    build_edit_support,
    build_page_windows,
    build_word_evidence_index,
    evaluate_hypotheses,
    generate_word_hypotheses,
    load_dictionary,
    next_token_id,
)
from run_copiale_multipage_experiment import (  # noqa: E402
    attach_page_scores,
    build_combined_cipher,
    consensus_from_finalists,
    finalist_rows,
    page_runtime_metrics,
    project_pages,
    score_page_runtime,
)


@dataclass
class TreeNode:
    node_id: str
    depth: int
    key: dict[int, int]
    mask: tuple[str, ...]
    cumulative_edits: list[str] = field(default_factory=list)
    parent_id: str | None = None
    source_variant: dict[str, Any] = field(default_factory=dict)
    runtime_score: float = 0.0
    prune_score: float = 0.0
    rank_score: float = 0.0
    post_hoc_char: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local iterative Copiale word-repair tree experiment."
    )
    parser.add_argument("experiment_json", help="JSON from run_copiale_multipage_experiment.py")
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument(
        "--section",
        choices=["portfolio_local_repair", "portfolio_refinement", "elite_page_rerank"],
        default="portfolio_local_repair",
    )
    parser.add_argument("--label", default="", help="Finalist label to use as the root, e.g. top6.")
    parser.add_argument("--dictionary", default="resources/dictionaries/german_common.txt")
    parser.add_argument("--depth", "-k", type=int, default=4)
    parser.add_argument("--branching", "-n", type=int, default=3)
    parser.add_argument("--beam-cap", type=int, default=18)
    parser.add_argument("--candidate-pool-per-node", type=int, default=24)
    parser.add_argument(
        "--sibling-similarity-threshold",
        type=float,
        default=0.0,
        help=(
            "If >0, avoid keeping too many near-duplicate children from the same "
            "parent. Similarity is Jaccard overlap over cumulative symbol->target edits."
        ),
    )
    parser.add_argument(
        "--sibling-similarity-cap",
        type=int,
        default=1,
        help="Maximum already-kept sibling branches allowed above the sibling similarity threshold.",
    )
    parser.add_argument(
        "--beam-similarity-threshold",
        type=float,
        default=0.0,
        help=(
            "If >0, apply the same near-duplicate cap while selecting the global "
            "beam for the next depth."
        ),
    )
    parser.add_argument(
        "--beam-similarity-cap",
        type=int,
        default=2,
        help="Maximum selected beam branches allowed above the beam similarity threshold.",
    )
    parser.add_argument("--parent-prune-delta", type=float, default=-0.75)
    parser.add_argument("--root-prune-delta", type=float, default=-1.25)
    parser.add_argument(
        "--prune-score",
        choices=[
            "runtime",
            "ranker",
            "adjudication_no_target",
            "adjudication",
            "second_stage",
            "validation",
            "language_quality",
        ],
        default="runtime",
        help="Ground-truth-free score used for parent/root pruning.",
    )
    parser.add_argument(
        "--allow-reedit-symbols",
        action="store_true",
        help="Allow a later tree step to change a symbol already edited on the path.",
    )
    parser.add_argument(
        "--ranker",
        choices=["adjudication_no_target", "adjudication", "second_stage", "target_first"],
        default="adjudication_no_target",
    )
    parser.add_argument("--consensus-top-n", type=int, default=12)
    parser.add_argument("--consensus-min-agreement", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--window-step", type=int, default=40)
    parser.add_argument("--windows-per-page", type=int, default=5)
    parser.add_argument("--min-word-len", type=int, default=5)
    parser.add_argument("--max-word-len", type=int, default=14)
    parser.add_argument("--max-edits", type=int, default=2)
    parser.add_argument("--max-hypotheses", type=int, default=120)
    parser.add_argument("--max-hypotheses-per-window", type=int, default=12)
    parser.add_argument("--max-hypothesis-set-size", type=int, default=1)
    parser.add_argument("--combination-candidate-limit", type=int, default=24)
    parser.add_argument("--max-combinations", type=int, default=200)
    parser.add_argument("--max-combined-edits", type=int, default=4)
    parser.add_argument("--allow-stable-edits", action="store_true")
    parser.add_argument("--acceptance-margin", type=float, default=0.03)
    parser.add_argument("--min-page-drop", type=float, default=0.02)
    parser.add_argument("--max-illusion-increase", type=float, default=0.02)
    parser.add_argument("--output-dir", default="artifacts/language_quality/iterative_repair_tree")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    log(args, "Loading Copiale multi-page experiment context...")
    context = load_context(args)
    log(args, "Scoring root candidate with local runtime metrics...")
    root_baseline = score_node_baseline(context, context["root_key"], context["root_mask"], args)
    root_runtime_score = runtime_prune_score(root_baseline)
    root_prune_score = prune_score(root_baseline, args.prune_score, args.ranker)
    root = TreeNode(
        node_id="root",
        depth=0,
        key=dict(context["root_key"]),
        mask=tuple(context["root_mask"]),
        runtime_score=root_runtime_score,
        prune_score=root_prune_score,
        rank_score=0.0,
        post_hoc_char=as_float(root_baseline.get("post_hoc_char_avg")),
        source_variant=root_baseline,
    )
    live_nodes = [root]
    all_nodes = [root]
    depth_reports: list[dict[str, Any]] = []
    seen_signatures = {node_signature(root)}
    log(
        args,
        f"Root label={context['label']} runtime={root_runtime_score:.3f} "
        f"prune={root_prune_score:.3f} posthoc={format_pct(root.post_hoc_char)}",
    )
    for depth in range(1, max(0, args.depth) + 1):
        depth_started = time.monotonic()
        log(args, f"\nDepth {depth}/{args.depth}: expanding {len(live_nodes)} node(s)")
        next_nodes: list[TreeNode] = []
        depth_pruned: list[dict[str, Any]] = []
        for node_index, node in enumerate(live_nodes, start=1):
            log(
                args,
                f"  [{depth}.{node_index}/{len(live_nodes)}] {node.node_id} "
                f"edits={len(node.cumulative_edits)} runtime={node.runtime_score:.3f}",
            )
            variants, baseline, generated = expand_node(context, node, args)
            ranked = rank_candidate_variants(variants, args.ranker)
            selected_for_node = 0
            considered = 0
            sibling_nodes: list[TreeNode] = []
            for variant in ranked[: max(args.candidate_pool_per_node, args.branching)]:
                considered += 1
                child = make_child_node(
                    context=context,
                    parent=node,
                    variant=variant,
                    depth=depth,
                    ordinal=len(next_nodes) + 1,
                    ranker=args.ranker,
                    prune_score_name=args.prune_score,
                )
                signature = node_signature(child)
                parent_delta = child.prune_score - node.prune_score
                root_delta = child.prune_score - root_prune_score
                prune_reason = ""
                if signature in seen_signatures:
                    prune_reason = "duplicate"
                elif (
                    not args.allow_reedit_symbols
                    and repeated_symbol_edit_path(child.cumulative_edits)
                ):
                    prune_reason = "reedit_symbol"
                elif too_many_similar_nodes(
                    child,
                    sibling_nodes,
                    threshold=args.sibling_similarity_threshold,
                    cap=args.sibling_similarity_cap,
                ):
                    prune_reason = "sibling_similarity"
                elif parent_delta < args.parent_prune_delta:
                    prune_reason = f"parent_delta={parent_delta:.3f}"
                elif root_delta < args.root_prune_delta:
                    prune_reason = f"root_delta={root_delta:.3f}"
                if prune_reason:
                    depth_pruned.append(pruned_summary(child, prune_reason, parent_delta, root_delta))
                    continue
                seen_signatures.add(signature)
                next_nodes.append(child)
                all_nodes.append(child)
                sibling_nodes.append(child)
                selected_for_node += 1
                log(
                    args,
                    f"    keep {child.node_id}: rank={child.rank_score:.3f} "
                    f"runtime={child.runtime_score:.3f} prune={child.prune_score:.3f} "
                    f"d_parent={parent_delta:+.3f} "
                    f"d_root={root_delta:+.3f} posthoc={format_pct(child.post_hoc_char)} "
                    f"edits={';'.join(variant.get('edits') or [])}",
                )
                if selected_for_node >= args.branching:
                    break
            log(
                args,
                f"    generated hypotheses={generated['hypothesis_count']} variants={generated['variant_count']} "
                f"considered={considered} kept={selected_for_node}",
            )
        next_nodes.sort(key=lambda item: (item.rank_score, item.runtime_score), reverse=True)
        if args.beam_cap > 0 and len(next_nodes) > args.beam_cap:
            selected, overflow = select_diverse_nodes(
                next_nodes,
                limit=args.beam_cap,
                threshold=args.beam_similarity_threshold,
                cap=args.beam_similarity_cap,
            )
            next_nodes = selected
            for child in overflow:
                depth_pruned.append(
                    pruned_summary(child, "beam_cap_or_similarity", 0.0, child.prune_score - root_prune_score)
                )
        depth_reports.append({
            "depth": depth,
            "expanded_nodes": len(live_nodes),
            "kept_nodes": len(next_nodes),
            "pruned_count": len(depth_pruned),
            "elapsed_seconds": round(time.monotonic() - depth_started, 3),
            "kept": [node_to_summary(row, include_variant=False) for row in next_nodes],
            "pruned": depth_pruned[:50],
        })
        log(
            args,
            f"Depth {depth} complete: kept={len(next_nodes)} pruned={len(depth_pruned)} "
            f"elapsed={time.monotonic() - depth_started:.1f}s",
        )
        live_nodes = next_nodes
        if not live_nodes:
            log(args, "No live nodes remain; stopping early.")
            break

    output_dir = resolve_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.experiment_json).stem}_{context['label']}_n{args.branching}_k{args.depth}"
    json_path = output_dir / f"{stem}.iterative_repair_tree.json"
    md_path = output_dir / f"{stem}.iterative_repair_tree.md"
    final_ranked = sorted(
        all_nodes,
        key=lambda item: (
            as_float(item.post_hoc_char) or 0.0,
            item.rank_score,
            item.runtime_score,
        ),
        reverse=True,
    )
    payload = {
        "experiment": "copiale_iterative_repair_tree",
        "source_experiment": str(context["experiment_path"]),
        "source_artifact": str(context["artifact_path"]),
        "label": context["label"],
        "settings": vars(args),
        "root_runtime_score": round(root_runtime_score, 6),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "node_count": len(all_nodes),
        "leaf_count": len(live_nodes),
        "depth_reports": depth_reports,
        "root": node_to_summary(root),
        "nodes_by_runtime_score": [
            node_to_summary(row)
            for row in sorted(all_nodes, key=lambda item: (item.runtime_score, item.rank_score), reverse=True)[:50]
        ],
        "nodes_by_rank_score": [
            node_to_summary(row)
            for row in sorted(all_nodes, key=lambda item: (item.rank_score, item.runtime_score), reverse=True)[:50]
        ],
        "nodes_by_post_hoc_char": [node_to_summary(row) for row in final_ranked[:50]],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    log(args, f"\nWrote {md_path}")
    log(args, f"Wrote {json_path}")


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    experiment_path = resolve_path(Path(args.experiment_json))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    artifact_path = sibling_artifact_path(experiment_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    test_ids = [str(item) for item in (experiment.get("test_ids") or [])]
    if not test_ids:
        raise SystemExit("Experiment JSON has no test_ids.")
    loader = BenchmarkLoader(resolve_path(Path(args.benchmark_root)))
    tests = {test.test_id: test for test in loader.load_tests(args.split)}
    _combined, pages = build_combined_cipher(loader, tests, test_ids)
    alphabet = pages_to_alphabet(pages)
    finalists = finalist_rows(artifact, top_n=10_000)
    label = args.label or selected_label_from_section(experiment, args.section)
    selected = next((row for row in finalists if row.get("_label") == label), None)
    if selected is None:
        raise SystemExit(f"No finalist label {label!r} found.")
    root_key = parse_key(selected.get("key"))
    root_mask = tuple(str(symbol) for symbol in (selected.get("mask") or []))
    consensus = consensus_from_finalists(
        artifact=artifact,
        alphabet=alphabet,
        top_n=args.consensus_top_n,
        min_agreement=args.consensus_min_agreement,
    )
    dictionary_path = resolve_path(Path(args.dictionary))
    dictionary = load_dictionary(dictionary_path, args.min_word_len, args.max_word_len)
    collateral_dictionary = load_dictionary(dictionary_path, 3, args.max_word_len)
    return {
        "experiment_path": experiment_path,
        "artifact_path": artifact_path,
        "experiment": experiment,
        "artifact": artifact,
        "test_ids": test_ids,
        "pages": pages,
        "alphabet": alphabet,
        "label": label,
        "root_key": root_key,
        "root_mask": root_mask,
        "consensus": consensus,
        "dictionary": dictionary,
        "collateral_index": build_word_evidence_index(collateral_dictionary),
    }


def expand_node(
    context: dict[str, Any],
    node: TreeNode,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    page_windows = build_page_windows(
        pages=context["pages"],
        alphabet=context["alphabet"],
        key=node.key,
        mask=node.mask,
        consensus=context["consensus"],
        window_size=args.window_size,
        window_step=args.window_step,
        windows_per_page=args.windows_per_page,
    )
    hypotheses = generate_word_hypotheses(
        page_windows=page_windows,
        dictionary=context["dictionary"],
        consensus=context["consensus"],
        alphabet=context["alphabet"],
        baseline_key=node.key,
        baseline_mask=node.mask,
        min_word_len=args.min_word_len,
        max_word_len=args.max_word_len,
        max_edits=args.max_edits,
        max_per_window=args.max_hypotheses_per_window,
        allow_stable_edits=args.allow_stable_edits,
    )
    hypotheses = hypotheses[: max(0, args.max_hypotheses)]
    variants = evaluate_hypotheses(
        pages=context["pages"],
        baseline_key=node.key,
        baseline_mask=node.mask,
        hypotheses=hypotheses,
        max_hypothesis_set_size=args.max_hypothesis_set_size,
        combination_candidate_limit=args.combination_candidate_limit,
        max_combinations=args.max_combinations,
        max_combined_edits=args.max_combined_edits,
        collateral_index=context["collateral_index"],
        edit_support=build_edit_support(hypotheses),
        progress=False,
    )
    baseline = next((row for row in variants if is_baseline_variant(row)), variants[0] if variants else {})
    if baseline:
        annotate_acceptance(
            variants,
            baseline=baseline,
            robust_margin=args.acceptance_margin,
            min_page_drop=args.min_page_drop,
            max_illusion_increase=args.max_illusion_increase,
            allow_pair_acceptance=True,
        )
        annotate_repair_evidence(variants, baseline=baseline)
        attach_second_stage_portfolio(variants, baseline=baseline)
    return variants, baseline, {"hypothesis_count": len(hypotheses), "variant_count": len(variants)}


def score_node_baseline(
    context: dict[str, Any],
    key: dict[int, int],
    mask: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    page_rows = project_pages(pages=context["pages"], key=key, mask=tuple(sorted(mask)))
    attach_page_scores(page_rows)
    runtime_scores = [
        score_page_runtime(row, key=key, mask=tuple(sorted(mask)))
        for row in page_rows
    ]
    metrics = page_runtime_metrics(runtime_scores)
    avg_char = sum(float(row.get("char_accuracy") or 0.0) for row in page_rows) / max(1, len(page_rows))
    return {
        "edits": ["baseline"],
        "mask": list(sorted(mask)),
        "post_hoc_char_avg": round(avg_char, 6),
        "post_hoc_page_chars": [
            {
                "test_id": row["test_id"],
                "char_accuracy": round(float(row.get("char_accuracy") or 0.0), 6),
                "word_accuracy": round(float(row.get("word_accuracy") or 0.0), 6),
            }
            for row in page_rows
        ],
        "page_previews": [
            {
                "test_id": row["test_id"],
                "preview": str(row.get("preview") or ""),
                "filtered_length": int(row.get("filtered_length") or 0),
            }
            for row in page_rows
        ],
        "preview": "; ".join(str(row.get("preview") or "")[:70] for row in page_rows[:3]),
        **metrics,
    }


def rank_candidate_variants(variants: list[dict[str, Any]], ranker: str) -> list[dict[str, Any]]:
    rows = [row for row in variants if not is_baseline_variant(row)]
    for row in rows:
        row["_iterative_rank_score"] = iterative_rank_score(row, ranker)
        row["_iterative_runtime_score"] = runtime_prune_score(row)
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("_iterative_rank_score") or 0.0),
            float(row.get("_iterative_runtime_score") or 0.0),
            -len(row.get("edits") or []),
        ),
        reverse=True,
    )


def iterative_rank_score(row: dict[str, Any], ranker: str) -> float:
    adj = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    stage = row.get("second_stage") if isinstance(row.get("second_stage"), dict) else {}
    if ranker == "adjudication_no_target":
        return float(adj.get("adjudication_no_target_score") or 0.0)
    if ranker == "adjudication":
        return float(adj.get("adjudication_score") or 0.0)
    if ranker == "second_stage":
        return float(stage.get("portfolio_score") or 0.0)
    if ranker == "target_first":
        return (
            float(adj.get("adjudication_score") or 0.0)
            + 0.65 * float(adj.get("target_leverage_score") or 0.0)
            + 0.15 * runtime_prune_score(row)
        )
    raise ValueError(ranker)


def runtime_prune_score(row: dict[str, Any]) -> float:
    # Scale to percentage-like units so a prune threshold around -1.0 means a
    # real runtime-quality regression, not numerical jitter.
    return 100.0 * float(row.get("page_robust_score") or 0.0)


def prune_score(row: dict[str, Any], score_name: str, ranker: str) -> float:
    adj = row.get("repair_adjudication") if isinstance(row.get("repair_adjudication"), dict) else {}
    stage = row.get("second_stage") if isinstance(row.get("second_stage"), dict) else {}
    if score_name == "runtime":
        return runtime_prune_score(row)
    if score_name == "ranker":
        return iterative_rank_score(row, ranker)
    if score_name == "adjudication_no_target":
        return float(adj.get("adjudication_no_target_score") or 0.0)
    if score_name == "adjudication":
        return float(adj.get("adjudication_score") or 0.0)
    if score_name == "second_stage":
        return float(stage.get("portfolio_score") or 0.0)
    if score_name == "validation":
        return 100.0 * float(row.get("page_validation_avg") or 0.0)
    if score_name == "language_quality":
        return 100.0 * float(row.get("page_language_quality_avg") or 0.0)
    raise ValueError(score_name)


def make_child_node(
    *,
    context: dict[str, Any],
    parent: TreeNode,
    variant: dict[str, Any],
    depth: int,
    ordinal: int,
    ranker: str,
    prune_score_name: str,
) -> TreeNode:
    key = dict(parent.key)
    mask = set(parent.mask)
    for edit in variant.get("edits") or []:
        parsed = parse_variant_edit(str(edit))
        if parsed is None:
            continue
        symbol, _before, target = parsed
        token_id = next_token_id(context["pages"], symbol)
        apply_assignment(symbol, token_id, target, key, mask)
    cumulative = parent.cumulative_edits + [
        str(edit) for edit in (variant.get("edits") or []) if str(edit).lower() != "baseline"
    ]
    return TreeNode(
        node_id=f"d{depth}_{ordinal:03d}",
        depth=depth,
        key=key,
        mask=tuple(sorted(mask)),
        cumulative_edits=cumulative,
        parent_id=parent.node_id,
        source_variant=variant_summary(variant),
        runtime_score=runtime_prune_score(variant),
        prune_score=prune_score(variant, prune_score_name, ranker),
        rank_score=iterative_rank_score(variant, ranker),
        post_hoc_char=as_float(variant.get("post_hoc_char_avg")),
    )


def parse_variant_edit(edit: str) -> tuple[str, str, str] | None:
    if edit.strip().lower() == "baseline":
        return None
    if ":" not in edit or "->" not in edit:
        return None
    symbol, rest = edit.split(":", 1)
    before, target = rest.split("->", 1)
    return symbol.strip(), before.strip(), target.strip()


def is_baseline_variant(row: dict[str, Any]) -> bool:
    edits = [str(edit).strip().lower() for edit in (row.get("edits") or [])]
    return not edits or edits == ["baseline"]


def repeated_symbol_edit_path(edits: list[str]) -> bool:
    seen: set[str] = set()
    for edit in edits:
        parsed = parse_variant_edit(str(edit))
        if parsed is None:
            continue
        symbol = parsed[0]
        if symbol in seen:
            return True
        seen.add(symbol)
    return False


def edit_assignment_set(node: TreeNode) -> set[tuple[str, str]]:
    assignments: set[tuple[str, str]] = set()
    for edit in node.cumulative_edits:
        parsed = parse_variant_edit(str(edit))
        if parsed is None:
            continue
        symbol, _before, target = parsed
        assignments.add((symbol, target))
    return assignments


def edit_similarity(left: TreeNode, right: TreeNode) -> float:
    left_assignments = edit_assignment_set(left)
    right_assignments = edit_assignment_set(right)
    if not left_assignments and not right_assignments:
        return 1.0
    union = left_assignments | right_assignments
    if not union:
        return 0.0
    return len(left_assignments & right_assignments) / len(union)


def too_many_similar_nodes(
    candidate: TreeNode,
    selected: list[TreeNode],
    *,
    threshold: float,
    cap: int,
) -> bool:
    if threshold <= 0.0 or cap < 0:
        return False
    similar_count = sum(1 for row in selected if edit_similarity(candidate, row) >= threshold)
    return similar_count >= cap


def select_diverse_nodes(
    ranked_nodes: list[TreeNode],
    *,
    limit: int,
    threshold: float,
    cap: int,
) -> tuple[list[TreeNode], list[TreeNode]]:
    if limit <= 0:
        return [], ranked_nodes
    selected: list[TreeNode] = []
    overflow: list[TreeNode] = []
    for node in ranked_nodes:
        if len(selected) >= limit:
            overflow.append(node)
            continue
        if too_many_similar_nodes(node, selected, threshold=threshold, cap=cap):
            overflow.append(node)
            continue
        selected.append(node)
    if len(selected) < limit:
        selected_ids = {id(node) for node in selected}
        for node in ranked_nodes:
            if id(node) in selected_ids:
                continue
            selected.append(node)
            selected_ids.add(id(node))
            if len(selected) >= limit:
                break
    selected_ids = {id(node) for node in selected}
    overflow = [node for node in ranked_nodes if id(node) not in selected_ids]
    return selected, overflow


def node_signature(node: TreeNode) -> tuple[tuple[int, int], tuple[str, ...]]:
    return tuple(sorted(node.key.items())), tuple(sorted(node.mask))


def node_to_summary(node: TreeNode, *, include_variant: bool = True) -> dict[str, Any]:
    row = {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "depth": node.depth,
        "cumulative_edits": node.cumulative_edits,
        "runtime_score": round(node.runtime_score, 6),
        "prune_score": round(node.prune_score, 6),
        "rank_score": round(node.rank_score, 6),
        "post_hoc_char": node.post_hoc_char,
        "preview": (node.source_variant or {}).get("preview"),
    }
    if include_variant:
        row["source_variant"] = node.source_variant
    return row


def pruned_summary(
    node: TreeNode,
    reason: str,
    parent_delta: float,
    root_delta: float,
) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "reason": reason,
        "edits": node.cumulative_edits,
        "rank_score": round(node.rank_score, 6),
        "runtime_score": round(node.runtime_score, 6),
        "prune_score": round(node.prune_score, 6),
        "parent_delta": round(parent_delta, 6),
        "root_delta": round(root_delta, 6),
        "post_hoc_char": node.post_hoc_char,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Copiale Iterative Repair Tree",
        "",
        "Ranking and pruning are ground-truth-free. Post-hoc character accuracy is calibration only.",
        "",
        f"- Source: `{payload['source_experiment']}`",
        f"- Label: `{payload['label']}`",
        f"- Elapsed seconds: `{payload['elapsed_seconds']}`",
        f"- Nodes produced: `{payload['node_count']}`",
        f"- Leaf nodes: `{payload['leaf_count']}`",
        f"- Root runtime score: `{payload['root_runtime_score']:.3f}`",
        "",
        "## Depth Summary",
        "",
        "| Depth | Expanded | Kept | Pruned | Elapsed |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("depth_reports", []):
        lines.append(
            f"| {row['depth']} | {row['expanded_nodes']} | {row['kept_nodes']} | "
            f"{row['pruned_count']} | {row['elapsed_seconds']:.1f}s |"
        )
    lines.extend([
        "",
        "## Best By Post-Hoc Character Accuracy",
        "",
        "| Rank | Node | Depth | Post-Hoc | Runtime | Rank Score | Edits | Preview |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ])
    for index, row in enumerate(payload.get("nodes_by_post_hoc_char", [])[:25], start=1):
        lines.append(node_table_row(index, row))
    lines.extend([
        "",
        "## Best By Runtime Score",
        "",
        "| Rank | Node | Depth | Post-Hoc | Runtime | Rank Score | Edits | Preview |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ])
    for index, row in enumerate(payload.get("nodes_by_runtime_score", [])[:25], start=1):
        lines.append(node_table_row(index, row))
    return "\n".join(lines).rstrip() + "\n"


def node_table_row(index: int, row: dict[str, Any]) -> str:
    return (
        f"| {index} | `{row.get('node_id')}` | {row.get('depth')} | "
        f"{format_pct(row.get('post_hoc_char'))} | "
        f"{format_float(row.get('runtime_score'))} | "
        f"{format_float(row.get('rank_score'))} | "
        f"{escape_cell('<br>'.join(f'`{edit}`' for edit in (row.get('cumulative_edits') or [])))} | "
        f"{escape_cell(str(row.get('preview') or '')[:140])} |"
    )


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_pct(value: Any) -> str:
    number = as_float(value)
    return "" if number is None else f"{number * 100:.1f}%"


def format_float(value: Any) -> str:
    number = as_float(value)
    return "" if number is None else f"{number:.3f}"


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def log(args: argparse.Namespace, message: str) -> None:
    if not args.quiet:
        print(message, flush=True)


if __name__ == "__main__":
    main()
