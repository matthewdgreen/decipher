"""Parity dashboard aggregation for agent and external baseline artifacts."""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact.analyzer import analyze_artifact, load_artifact, summarize_findings


@dataclass
class RunSummary:
    test_id: str
    source: str
    status: str
    char_accuracy: float | None
    word_accuracy: float | None = None
    solver: str = ""
    path: str = ""
    elapsed_seconds: float | None = None
    iterations: int | None = None
    cost_usd: float | None = None
    labels: dict[str, int] = field(default_factory=dict)


@dataclass
class DashboardRow:
    test_id: str
    family: str
    external_best: RunSummary | None = None
    automated_best: RunSummary | None = None
    native_best: RunSummary | None = None
    full_agent_best: RunSummary | None = None
    gap_labels: dict[str, int] = field(default_factory=dict)
    next_action: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "family": self.family,
            "external_best": _run_to_dict(self.external_best),
            "automated_best": _run_to_dict(self.automated_best),
            "native_best": _run_to_dict(self.native_best),
            "full_agent_best": _run_to_dict(self.full_agent_best),
            "gap_labels": self.gap_labels,
            "next_action": self.next_action,
            "metadata": self.metadata,
        }


def build_dashboard(
    agent_paths: list[str | Path],
    external_paths: list[str | Path] | None = None,
    automated_paths: list[str | Path] | None = None,
    benchmark_root: str | Path | None = None,
) -> list[DashboardRow]:
    """Build parity rows from artifact files/directories/globs."""
    metadata = load_split_metadata(benchmark_root) if benchmark_root else {}
    rows: dict[str, DashboardRow] = {}

    for summary in load_agent_summaries(agent_paths):
        row = rows.setdefault(
            summary.test_id,
            DashboardRow(
                test_id=summary.test_id,
                family=_family_for(summary.test_id, metadata.get(summary.test_id, {})),
                metadata=metadata.get(summary.test_id, {}),
            ),
        )
        row.full_agent_best = _pick_best(row.full_agent_best, summary)
        if _is_native_solver_run(summary):
            row.native_best = _pick_best(row.native_best, summary)

    for summary in load_external_summaries(external_paths or []):
        row = rows.setdefault(
            summary.test_id,
            DashboardRow(
                test_id=summary.test_id,
                family=_family_for(summary.test_id, metadata.get(summary.test_id, {})),
                metadata=metadata.get(summary.test_id, {}),
            ),
        )
        row.external_best = _pick_best(row.external_best, summary)

    for summary in load_automated_summaries(automated_paths or []):
        row = rows.setdefault(
            summary.test_id,
            DashboardRow(
                test_id=summary.test_id,
                family=_family_for(summary.test_id, metadata.get(summary.test_id, {})),
                metadata=metadata.get(summary.test_id, {}),
            ),
        )
        row.automated_best = _pick_best(row.automated_best, summary)

    for row in rows.values():
        row.gap_labels = dict(row.full_agent_best.labels) if row.full_agent_best else {}
        row.next_action = _next_action(row)

    return sorted(rows.values(), key=lambda r: (r.family, r.test_id))


def load_agent_summaries(paths: list[str | Path]) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for path in _expand_paths(paths):
        if _is_external_artifact(path):
            continue
        try:
            artifact = load_artifact(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not _looks_like_agent_artifact(artifact):
            continue
        findings = summarize_findings(analyze_artifact(artifact))
        summaries.append(RunSummary(
            test_id=str(artifact.get("cipher_id") or path.parent.name),
            source="full_agent",
            status=str(artifact.get("status") or ""),
            char_accuracy=_float_or_none(artifact.get("char_accuracy")),
            word_accuracy=_float_or_none(artifact.get("word_accuracy")),
            solver=_agent_solver_name(artifact),
            path=str(path),
            iterations=_max_iteration(artifact),
            cost_usd=_float_or_none(artifact.get("estimated_cost_usd")),
            labels=findings["labels"],
        ))
    return summaries


def load_external_summaries(paths: list[str | Path]) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for path in _expand_paths(paths):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _looks_like_external_artifact(artifact):
            continue
        summaries.append(RunSummary(
            test_id=str(artifact.get("test_id") or path.parent.parent.name),
            source="external",
            status=str(artifact.get("status") or ""),
            char_accuracy=_float_or_none(artifact.get("char_accuracy")),
            word_accuracy=_float_or_none(artifact.get("word_accuracy")),
            solver=str(artifact.get("solver") or ""),
            path=str(path),
            elapsed_seconds=_float_or_none(artifact.get("elapsed")),
        ))
    return summaries


def load_automated_summaries(paths: list[str | Path]) -> list[RunSummary]:
    """Load future no-LLM/automated-only artifacts.

    The planned automated mode may emit either a small result artifact with
    ``run_mode: automated_only`` or a RunArtifact-like object with the same
    marker. Supporting both shapes now keeps the dashboard stable when that
    CLI flag lands.
    """
    summaries: list[RunSummary] = []
    for path in _expand_paths(paths):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _looks_like_automated_artifact(artifact):
            continue
        summaries.append(RunSummary(
            test_id=str(artifact.get("test_id") or artifact.get("cipher_id") or path.parent.name),
            source="automated_only",
            status=str(artifact.get("status") or ""),
            char_accuracy=_float_or_none(artifact.get("char_accuracy")),
            word_accuracy=_float_or_none(artifact.get("word_accuracy")),
            solver=str(artifact.get("solver") or artifact.get("run_mode") or "automated_only"),
            path=str(path),
            elapsed_seconds=_float_or_none(
                artifact.get("elapsed") or artifact.get("elapsed_seconds")
            ),
            iterations=_int_or_none(artifact.get("iterations") or artifact.get("iterations_used")),
            cost_usd=_float_or_none(artifact.get("estimated_cost_usd")),
        ))
    return summaries


def load_split_metadata(benchmark_root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(benchmark_root)
    splits_dir = root / "splits"
    metadata: dict[str, dict[str, Any]] = {}
    for split_path in sorted(splits_dir.glob("*.jsonl")):
        with open(split_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                test_id = data.get("test_id")
                if not test_id:
                    continue
                merged = dict(metadata.get(test_id, {}))
                merged.update(data)
                merged.setdefault("split_file", split_path.name)
                metadata[str(test_id)] = merged
    return metadata


def render_markdown(rows: list[DashboardRow]) -> str:
    lines = [
        "| Test ID | Family | External Best | Automated Only | Native Best | Full Agent | Gap Labels | Next Action |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row.test_id,
                row.family,
                _format_run(row.external_best),
                _format_run(row.automated_best),
                _format_run(row.native_best),
                _format_run(row.full_agent_best),
                _format_labels(row.gap_labels),
                row.next_action,
            ])
            + " |"
        )
    return "\n".join(lines)


def render_json(rows: list[DashboardRow]) -> str:
    return json.dumps([row.to_dict() for row in rows], indent=2, ensure_ascii=False)


def _expand_paths(paths: list[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for item in paths:
        text = str(item)
        matches = [Path(p) for p in glob.glob(text, recursive=True)]
        if matches:
            for match in matches:
                if match.is_dir():
                    expanded.extend(sorted(match.rglob("*.json")))
                else:
                    expanded.append(match)
            continue
        path = Path(text)
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.json")))
        elif path.exists():
            expanded.append(path)
    return sorted(set(expanded))


def _looks_like_agent_artifact(artifact: dict[str, Any]) -> bool:
    return "cipher_id" in artifact and "tool_calls" in artifact and "branches" in artifact


def _looks_like_external_artifact(artifact: dict[str, Any]) -> bool:
    return "test_id" in artifact and "solver" in artifact and "command" in artifact


def _looks_like_automated_artifact(artifact: dict[str, Any]) -> bool:
    run_mode = artifact.get("run_mode") or artifact.get("mode") or artifact.get("source")
    if run_mode in {"automated_only", "no_llm", "non_llm"}:
        return True
    return bool(artifact.get("automated_only") is True)


def _is_external_artifact(path: Path) -> bool:
    return "external_baselines" in path.parts


def _agent_solver_name(artifact: dict[str, Any]) -> str:
    tool_names = [call.get("tool_name") for call in artifact.get("tool_calls", [])]
    if "search_homophonic_anneal" in tool_names:
        return "native_homophonic_anneal"
    if "search_anneal" in tool_names:
        return "native_anneal"
    if "search_hill_climb" in tool_names:
        return "native_hill_climb"
    return "agent_tools"


def _is_native_solver_run(summary: RunSummary) -> bool:
    return summary.solver.startswith("native_")


def _max_iteration(artifact: dict[str, Any]) -> int:
    calls = artifact.get("tool_calls", [])
    iterations = [
        int(call.get("iteration") or 0)
        for call in calls
        if isinstance(call, dict)
    ]
    return max(iterations) if iterations else 0


def _pick_best(current: RunSummary | None, candidate: RunSummary) -> RunSummary:
    if current is None:
        return candidate
    current_score = current.char_accuracy if current.char_accuracy is not None else -1.0
    candidate_score = candidate.char_accuracy if candidate.char_accuracy is not None else -1.0
    if candidate_score > current_score:
        return candidate
    if candidate_score == current_score:
        current_labels = sum(current.labels.values())
        candidate_labels = sum(candidate.labels.values())
        if candidate_labels < current_labels:
            return candidate
        if candidate_labels == current_labels:
            current_iters = current.iterations if current.iterations is not None else 10**9
            candidate_iters = candidate.iterations if candidate.iterations is not None else 10**9
            if candidate_iters < current_iters:
                return candidate
    return current


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _family_for(test_id: str, metadata: dict[str, Any]) -> str:
    if metadata.get("parity_family"):
        return str(metadata["parity_family"])
    if metadata.get("cipher_system"):
        return str(metadata["cipher_system"])
    if "honb" in test_id or "homophonic" in test_id or "zodiac" in test_id:
        return "homophonic_substitution"
    if "borg" in test_id:
        return "borg_latin"
    if "copiale" in test_id:
        return "copiale_german"
    if "ss_synth" in test_id or "synth" in test_id:
        return "simple_substitution"
    return "unknown"


def _next_action(row: DashboardRow) -> str:
    if row.gap_labels:
        labels = set(row.gap_labels)
        if "agent_wrong_tool" in labels:
            return "inspect prompt/tool routing"
        if "tool_underpowered" in labels or "tool_missing" in labels:
            return "port or strengthen native tool"
        if "premature_declaration" in labels:
            return "inspect repair/declaration guardrails"
        if "benchmark_data_issue" in labels:
            return "fix benchmark record"
        return "inspect artifact labels"

    agent = row.full_agent_best.char_accuracy if row.full_agent_best else None
    automated = row.automated_best.char_accuracy if row.automated_best else None
    external = row.external_best.char_accuracy if row.external_best else None
    if agent is None and automated is None and external is None:
        return "run external, automated, and agent baselines"
    if automated is None:
        return "run automated-only baseline"
    if agent is None:
        return "run full agent"
    if external is None:
        return "run external baseline"
    if automated is not None and automated + 0.01 < external:
        return "classify automated parity gap"
    if automated is not None and agent + 0.01 < automated:
        return "classify agent-vs-automated gap"
    if agent + 0.01 < external:
        return "classify parity gap"
    return "parity ok"


def _format_run(run: RunSummary | None) -> str:
    if run is None or run.char_accuracy is None:
        return "n/a"
    label = run.solver or run.status or run.source
    return f"{run.char_accuracy:.1%} ({label})"


def _format_labels(labels: dict[str, int]) -> str:
    if not labels:
        return ""
    return ", ".join(f"{k}:{v}" for k, v in sorted(labels.items()))


def _run_to_dict(run: RunSummary | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "test_id": run.test_id,
        "source": run.source,
        "status": run.status,
        "char_accuracy": run.char_accuracy,
        "word_accuracy": run.word_accuracy,
        "solver": run.solver,
        "path": run.path,
        "elapsed_seconds": run.elapsed_seconds,
        "iterations": run.iterations,
        "cost_usd": run.cost_usd,
        "labels": run.labels,
    }


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None
