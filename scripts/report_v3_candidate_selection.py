#!/usr/bin/env python3
"""Explain v3 candidate generation and selection from saved artifacts.

This is a grading-side diagnostic. Ground truth, when present, is used only
after all candidates have already been generated and selected. Nothing in this
module is imported by solver or agent runtime code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.loop_shared import (  # noqa: E402
    _candidate_content_hash,
    _decoded_text_for_panel,
    _score_branch_for_panel,
)
from analysis.dictionary import get_dictionary_path, load_word_set  # noqa: E402
from benchmark.scorer import score_decryption  # noqa: E402
from investigation.state import InvestigationState  # noqa: E402


def _dictionary(language: str) -> tuple[set[str], dict[str, int]]:
    path = get_dictionary_path(language)
    if not path:
        return set(), {}
    word_set = load_word_set(path)
    words = [
        line.strip().upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return word_set, {word: index + 1 for index, word in enumerate(words)}


def _roles_for(name: str, roles: dict[str, Any]) -> list[str]:
    return [role for role, branch in roles.items() if branch == name]


def _null_mask(branch: Any) -> set[str]:
    block = branch.metadata.get("null_mask_finalist")
    if not isinstance(block, dict):
        block = branch.metadata.get("null_mask_selected")
    if not isinstance(block, dict):
        return set()
    return {str(symbol) for symbol in block.get("mask") or []}


def _unsegmented_null_mask_text(workspace: Any, name: str) -> str | None:
    branch = workspace.get_branch(name)
    mask = _null_mask(branch)
    if not branch.key or not mask:
        return None
    cipher_alpha = workspace.cipher_text.alphabet
    plaintext_alpha = workspace.plaintext_alphabet
    return "".join(
        plaintext_alpha.symbol_for(branch.key[token]) if token in branch.key else "?"
        for token in workspace.effective_tokens(name)
        if cipher_alpha.symbol_for(token) not in mask
    )


def _post_hoc_score(artifact: dict[str, Any], text: str) -> dict[str, Any] | None:
    truth = artifact.get("ground_truth")
    if not isinstance(truth, str) or not truth.strip():
        return None
    result = score_decryption(
        str(artifact.get("cipher_id") or artifact.get("test_id") or "artifact"),
        text,
        truth,
        0.0,
        "diagnostic",
    )
    return {
        "char_accuracy": result.char_accuracy,
        "word_accuracy": result.word_accuracy if result.total_words else None,
        "correct_chars": result.correct_chars,
        "total_chars": result.total_chars,
        "correct_words": result.correct_words,
        "total_words": result.total_words,
    }


def _snapshot_history(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: tuple[Any, ...] | None = None
    for event in artifact.get("loop_events") or []:
        if event.get("event") != "workspace_snapshot":
            continue
        payload = event.get("payload") or {}
        text = str(payload.get("decryption") or "")
        row = {
            "iteration": int(payload.get("iteration") or 0),
            "branch": str(payload.get("branch") or ""),
            "dict_rate": (payload.get("scores") or {}).get("dict_rate"),
            "quad": (payload.get("scores") or {}).get("quad"),
            "has_boundaries": any(char.isspace() for char in text.strip()),
            "preview": text[:180],
        }
        signature = (row["branch"], row["dict_rate"], row["quad"], row["has_boundaries"])
        if signature != previous:
            rows.append(row)
            previous = signature
    return rows


def analyze_artifact_data(
    artifact: dict[str, Any], artifact_path: Path | None = None
) -> dict[str, Any]:
    state_data = artifact.get("investigation_state")
    if not isinstance(state_data, dict):
        raise ValueError("artifact has no investigation_state")
    state = InvestigationState.from_artifact_dict(state_data)
    workspace = state.workspace
    word_set, freq_rank = _dictionary(state.language)
    roles = dict(artifact.get("branch_roles") or {})
    snapshots = _snapshot_history(artifact)
    historical_flat = {
        row["branch"]
        for row in snapshots
        if row["branch"] and not row["has_boundaries"]
    }

    att_by_hash: dict[str, list[dict[str, Any]]] = {}
    for attestation in artifact.get("attestations") or state.verify_attestations:
        att_by_hash.setdefault(str(attestation.get("content_hash") or ""), []).append(
            dict(attestation)
        )

    branches: list[dict[str, Any]] = []
    for name in workspace.branch_names():
        branch = workspace.get_branch(name)
        if not branch.key and not branch.metadata.get("decoded_text"):
            continue
        text = _decoded_text_for_panel(workspace, name)
        dict_rate, quad = _score_branch_for_panel(
            workspace, name, state.language, word_set, freq_rank
        )
        unsegmented = _unsegmented_null_mask_text(workspace, name)
        content_hash = _candidate_content_hash(text)
        unsegmented_hash = (
            _candidate_content_hash(unsegmented) if unsegmented is not None else None
        )
        effective_spans = workspace.effective_word_spans(name)
        boundary_mode = (
            "custom"
            if branch.word_spans is not None
            else "transformed_flat"
            if branch.token_order is not None
            else "source"
            if len(effective_spans) > 1
            else "none"
        )
        row = {
            "branch": name,
            "parent": branch.parent,
            "created_iteration": branch.created_iteration,
            "roles": _roles_for(name, roles),
            "tags": list(branch.tags),
            "renderer": str(branch.metadata.get("candidate_renderer") or "decoded_text_v1"),
            "boundary_mode": boundary_mode,
            "rendered_word_count": len(text.split()) if text.strip() else 0,
            "mapped_count": len(branch.key),
            "dict_rate": dict_rate,
            "quad": quad,
            "content_hash": content_hash,
            "attestations": att_by_hash.get(content_hash, []),
            "post_hoc": _post_hoc_score(artifact, text),
            "preview": text[:220],
            "historical_boundary_loss": name in historical_flat and boundary_mode == "source",
        }
        if unsegmented is not None and unsegmented != text:
            unseg_score = _post_hoc_score(artifact, unsegmented)
            row["unsegmented_variant"] = {
                "content_hash": unsegmented_hash,
                "attestations": att_by_hash.get(str(unsegmented_hash), []),
                "post_hoc": unseg_score,
                "preview": unsegmented[:220],
            }
        branches.append(row)

    def solver_key(row: dict[str, Any]) -> tuple[float, float, int]:
        return (
            float(row["dict_rate"]) if row["dict_rate"] is not None else float("-inf"),
            float(row["quad"]) if row["quad"] is not None else float("-inf"),
            int(row["mapped_count"]),
        )

    def grade_key(row: dict[str, Any]) -> tuple[float, float]:
        post_hoc = row["post_hoc"]
        word_accuracy = post_hoc.get("word_accuracy")
        return (
            float(post_hoc["char_accuracy"]),
            float(word_accuracy) if word_accuracy is not None else -1.0,
        )

    solver_order = sorted(branches, key=solver_key, reverse=True)
    previous_solver_key: tuple[float, float, int] | None = None
    solver_rank = 0
    for index, row in enumerate(solver_order, start=1):
        key = solver_key(row)
        if key != previous_solver_key:
            solver_rank = index
            previous_solver_key = key
        row["solver_rank"] = solver_rank

    graded = [row for row in branches if row.get("post_hoc")]
    graded.sort(key=grade_key, reverse=True)
    previous_grade_key: tuple[float, float] | None = None
    post_hoc_rank = 0
    for index, row in enumerate(graded, start=1):
        key = grade_key(row)
        if key != previous_grade_key:
            post_hoc_rank = index
            previous_grade_key = key
        row["post_hoc_rank"] = post_hoc_rank

    selected_name = str(
        roles.get("declared_or_selected_branch")
        or roles.get("workflow_branch")
        or ""
    )
    selected = next((row for row in branches if row["branch"] == selected_name), None)
    best = graded[0] if graded else None
    best_key = grade_key(best) if best else None
    best_branches = [
        row["branch"]
        for row in graded
        if grade_key(row) == best_key
    ]
    findings: list[str] = []
    boundary_loss = [row["branch"] for row in branches if row["historical_boundary_loss"]]
    if boundary_loss:
        findings.append(
            "Historical snapshots flattened canonical source boundaries for null-mask "
            f"branch(es): {', '.join(boundary_loss)}."
        )
    historical_rejections = []
    for row in branches:
        old = row.get("unsegmented_variant") or {}
        old_attestations = old.get("attestations") or []
        if row["historical_boundary_loss"] and any(
            not item.get("reader_accepts_as_solution") for item in old_attestations
        ):
            historical_rejections.append(row["branch"])
    if historical_rejections:
        findings.append(
            "Historical verifier rejection(s) were bound to the unsegmented content "
            f"for: {', '.join(historical_rejections)}. Corrected renderings require "
            "fresh verification."
        )
    selected_key = grade_key(selected) if selected and selected.get("post_hoc") else None
    if selected and best and selected_key != best_key:
        gap = best["post_hoc"]["char_accuracy"] - selected["post_hoc"]["char_accuracy"]
        findings.append(
            f"Selected {selected['branch']} while post-hoc best was {best['branch']} "
            f"(character gap {gap:.1%})."
        )
    best_solver_rank = min(
        (row.get("solver_rank", 1) for row in graded if row["branch"] in best_branches),
        default=1,
    )
    if best and best_solver_rank > 1:
        findings.append(
            f"Post-hoc best {', '.join(best_branches)} ranked {best_solver_rank} by "
            "the ground-truth-free scalar ordering."
        )

    return {
        "artifact": str(artifact_path) if artifact_path else None,
        "run_id": artifact.get("run_id"),
        "test_id": artifact.get("cipher_id") or artifact.get("test_id"),
        "status": artifact.get("status"),
        "language": state.language,
        "source_word_count": len(workspace.cipher_text.words),
        "branch_roles": roles,
        "selected_branch": selected_name or None,
        "post_hoc_best_branch": best["branch"] if best else None,
        "post_hoc_best_branches": best_branches,
        "findings": findings,
        "timeline": snapshots,
        "branches": solver_order,
    }


def analyze_artifact(path: Path) -> dict[str, Any]:
    return analyze_artifact_data(json.loads(path.read_text(encoding="utf-8")), path)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# V3 Candidate Selection Report",
        "",
        "Ground truth is used only for post-hoc grading after candidate generation and selection.",
        "",
    ]
    boundary_runs = [report for report in reports if any(
        row.get("historical_boundary_loss") for row in report["branches"]
    )]
    best_chars = [
        max(
            float(row["post_hoc"]["char_accuracy"])
            for row in report["branches"]
            if row.get("post_hoc")
        )
        for report in reports
        if any(row.get("post_hoc") for row in report["branches"])
    ]
    if boundary_runs or len(best_chars) > 1:
        lines.extend(["## Cross-Run Findings", ""])
        if boundary_runs:
            lines.append(
                f"- {len(boundary_runs)} run(s) made historical decisions on a "
                "boundary-flattened candidate. Their old verifier verdicts do not "
                "evaluate the corrected rendering."
            )
        if len(best_chars) > 1:
            spread = max(best_chars) - min(best_chars)
            lines.append(
                f"- Best generated post-hoc character accuracy ranges from "
                f"{_pct(min(best_chars))} to {_pct(max(best_chars))} "
                f"({spread:.1%} spread), so candidate-generation variance remains "
                "material after the rendering fix."
            )
        lines.append("")
    for report in reports:
        lines.extend([
            f"## {report['test_id']} / {report['run_id']}",
            "",
            f"- Status: `{report['status']}`",
            f"- Selected branch: `{report['selected_branch']}`",
            "- Post-hoc best branch(es): " + ", ".join(
                f"`{name}`" for name in report["post_hoc_best_branches"]
            ),
            f"- Canonical source words: {report['source_word_count']}",
            "",
        ])
        for finding in report["findings"]:
            lines.append(f"- **Finding:** {finding}")
        if report["findings"]:
            lines.append("")
        lines.extend([
            "| Scalar | GT | Branch | Roles | Renderer | Boundaries | Dict | Quad | GT char | GT word | Verify |",
            "|---:|---:|---|---|---|---|---:|---:|---:|---:|---|",
        ])
        for row in report["branches"]:
            post = row.get("post_hoc") or {}
            verdicts = row.get("attestations") or []
            verify = ", ".join(
                "accept" if item.get("reader_accepts_as_solution") else "reject"
                for item in verdicts
            ) or "-"
            lines.append(
                "| {sr} | {gr} | `{branch}` | {roles} | `{renderer}` | {boundaries} | "
                "{dr} | {quad} | {char} | {word} | {verify} |".format(
                    sr=row.get("solver_rank", "-"),
                    gr=row.get("post_hoc_rank", "-"),
                    branch=row["branch"],
                    roles=", ".join(row["roles"]) or "-",
                    renderer=row["renderer"],
                    boundaries=row["boundary_mode"],
                    dr=_num(row["dict_rate"]),
                    quad=_num(row["quad"]),
                    char=_pct(post.get("char_accuracy")),
                    word=_pct(post.get("word_accuracy")),
                    verify=verify,
                )
            )
        replay_rows = [row for row in report["branches"] if row.get("unsegmented_variant")]
        if replay_rows:
            lines.extend([
                "",
                "### Boundary Replay",
                "",
                "| Branch | Unsegmented GT word | Boundary-preserved GT word | GT char | Historical verify | Historical loss |",
                "|---|---:|---:|---:|---|---|",
            ])
            for row in replay_rows:
                current = row.get("post_hoc") or {}
                old = row["unsegmented_variant"].get("post_hoc") or {}
                old_attestations = row["unsegmented_variant"].get("attestations") or []
                old_verify = ", ".join(
                    "accept" if item.get("reader_accepts_as_solution") else "reject"
                    for item in old_attestations
                ) or "-"
                lines.append(
                    f"| `{row['branch']}` | {_pct(old.get('word_accuracy'))} | "
                    f"{_pct(current.get('word_accuracy'))} | "
                    f"{_pct(current.get('char_accuracy'))} | "
                    f"{old_verify} | "
                    f"{'yes' if row['historical_boundary_loss'] else 'no'} |"
                )
        lines.extend(["", "### Best-Branch Timeline", "", "| Turn | Branch | Dict | Quad | Boundaries |", "|---:|---|---:|---:|---|"])
        for row in report["timeline"]:
            lines.append(
                f"| {row['iteration']} | `{row['branch']}` | {_num(row['dict_rate'])} | "
                f"{_num(row['quad'])} | {'yes' if row['has_boundaries'] else 'no'} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Write <prefix>.json and <prefix>.md; otherwise print Markdown.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reports = [analyze_artifact(path) for path in args.artifacts]
    markdown = render_markdown(reports)
    if args.output_prefix:
        prefix = args.output_prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = prefix.with_suffix(".json")
        md_path = prefix.with_suffix(".md")
        json_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
