from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.loader import BenchmarkLoader, BenchmarkTest, TestData
from testgen.builder import build_test_case
from testgen.cache import PlaintextCache
from testgen.spec import TestSpec


FRONTIER_CLASSES = frozenset({"known_good", "bad_result", "slow_result", "shared_hard"})


@dataclass
class FrontierCase:
    test: BenchmarkTest
    frontier_class: str
    frontier_tags: list[str] = field(default_factory=list)
    expected_solvers: list[str] = field(default_factory=list)
    expected_status_by_solver: dict[str, str] = field(default_factory=dict)
    min_char_accuracy_by_solver: dict[str, float] = field(default_factory=dict)
    max_elapsed_seconds_by_solver: dict[str, float] = field(default_factory=dict)
    max_gap_vs_solver: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    synthetic_spec: TestSpec | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def source_mode(self) -> str:
        return "synthetic" if self.synthetic_spec else "benchmark"


def load_frontier_suite(
    suite_path: str | Path,
    frontier_classes: set[str] | None = None,
    tags: set[str] | None = None,
    families: set[str] | None = None,
    test_ids: set[str] | None = None,
) -> list[FrontierCase]:
    cases: list[FrontierCase] = []
    with open(suite_path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            case = _parse_frontier_case(data, source=f"{suite_path}:{lineno}")
            if frontier_classes and case.frontier_class not in frontier_classes:
                continue
            if tags and not tags.intersection(case.frontier_tags):
                continue
            if families and case.test.cipher_system not in families:
                continue
            if test_ids and case.test.test_id not in test_ids:
                continue
            cases.append(case)
    return cases


def resolve_frontier_case(
    case: FrontierCase,
    benchmark_loader: BenchmarkLoader,
    cache: PlaintextCache,
    allow_generate: bool = False,
    generation_api_key: str = "",
) -> TestData:
    if case.synthetic_spec is None:
        return benchmark_loader.load_test_data(case.test)

    spec = case.synthetic_spec
    cached = cache.get(spec)
    if cached is None and not allow_generate:
        raise ValueError(
            f"synthetic cache miss for {spec}; rerun with --allow-generate "
            "or pre-populate the plaintext cache"
        )
    api_key = generation_api_key if cached is None else ""
    return build_test_case(spec, cache, api_key)


def evaluate_frontier_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_test_solver: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("test_id") or ""), canonical_solver_name(row.get("solver_key") or row.get("solver") or ""))
        current = best_by_test_solver.get(key)
        if current is None or _better_row(row, current):
            best_by_test_solver[key] = row

    evaluated: list[dict[str, Any]] = []
    for row in rows:
        failures: list[str] = []
        solver = canonical_solver_name(row.get("solver_key") or row.get("solver") or "")
        test_id = str(row.get("test_id") or "")
        status = str(row.get("status") or "")
        char_accuracy = float(row.get("char_accuracy") or 0.0)
        elapsed = float(row.get("elapsed_seconds") or 0.0)

        expected_status = row.get("expected_status_by_solver", {}).get(solver)
        if expected_status and status != expected_status:
            failures.append(f"status={status} expected={expected_status}")

        min_char = row.get("min_char_accuracy_by_solver", {}).get(solver)
        if min_char is not None and char_accuracy + 1e-9 < float(min_char):
            failures.append(
                f"char_accuracy={char_accuracy:.3f} below {float(min_char):.3f}"
            )

        max_elapsed = row.get("max_elapsed_seconds_by_solver", {}).get(solver)
        if max_elapsed is not None and elapsed - 1e-9 > float(max_elapsed):
            failures.append(
                f"elapsed_seconds={elapsed:.1f} above {float(max_elapsed):.1f}"
            )

        if solver == "decipher-automated":
            for ref_solver, max_gap in row.get("max_gap_vs_solver", {}).items():
                ref_row = best_by_test_solver.get((test_id, canonical_solver_name(ref_solver)))
                if ref_row is None:
                    continue
                ref_char = float(ref_row.get("char_accuracy") or 0.0)
                gap = ref_char - char_accuracy
                if gap - 1e-9 > float(max_gap):
                    failures.append(
                        f"gap_vs_{ref_solver}={gap:.3f} above {float(max_gap):.3f}"
                    )

        updated = dict(row)
        updated["meets_expectations"] = not failures
        updated["expectation_failures"] = failures
        evaluated.append(updated)
    return evaluated


def load_parity_rows(
    summary_paths: list[str | Path] | None = None,
    artifact_dirs: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summary_paths or []:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix == ".csv":
            with open(p, encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows.append(_normalize_row(row, source_file=str(p)))
        elif p.suffix == ".jsonl":
            with open(p, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(_normalize_row(json.loads(line), source_file=str(p)))

    for artifact_dir in artifact_dirs or []:
        root = Path(artifact_dir)
        if not root.exists():
            continue
        for path in root.glob("decipher/automated_only/*/*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(_normalize_row({
                "test_id": data.get("test_id"),
                "source": "",
                "family": data.get("cipher_system") or "",
                "cipher_system": data.get("cipher_system") or "",
                "language": "",
                "approx_length": "",
                "word_boundaries": data.get("cipher_word_count", 0) != 1,
                "homophonic": "homophonic" in str(data.get("cipher_system") or ""),
                "seed": "",
                "solver": "decipher-automated",
                "status": data.get("status"),
                "char_accuracy": data.get("char_accuracy") or 0.0,
                "word_accuracy": data.get("word_accuracy") or 0.0,
                "elapsed_seconds": data.get("elapsed_seconds") or 0.0,
                "candidates": 1,
                "artifact_path": str(path),
                "error": data.get("error_message", ""),
            }, source_file=str(root)))
        for path in root.glob("external/*/*/artifact.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(_normalize_row({
                "test_id": data.get("test_id"),
                "source": "",
                "family": data.get("cipher_system") or "",
                "cipher_system": data.get("cipher_system") or "",
                "language": "",
                "approx_length": "",
                "word_boundaries": False,
                "homophonic": "homophonic" in str(data.get("cipher_system") or ""),
                "seed": "",
                "solver": data.get("solver_name") or data.get("solver") or path.parts[-3],
                "status": data.get("status"),
                "char_accuracy": data.get("char_accuracy") or 0.0,
                "word_accuracy": data.get("word_accuracy") or 0.0,
                "elapsed_seconds": data.get("elapsed_seconds") or data.get("elapsed") or 0.0,
                "candidates": data.get("candidates_considered") or 0,
                "artifact_path": str(path),
                "error": data.get("error", ""),
            }, source_file=str(root)))
    return rows


def nominate_frontier_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("test_id") or ""),
            canonical_solver_name(row.get("solver_key") or row.get("solver") or ""),
        )
        current = best_rows.get(key)
        if current is None or _better_row(row, current):
            best_rows[key] = row

    by_test: dict[str, dict[str, dict[str, Any]]] = {}
    for (test_id, solver), row in best_rows.items():
        by_test.setdefault(test_id, {})[solver] = row

    nominations: list[dict[str, Any]] = []
    for test_id, solver_rows in sorted(by_test.items()):
        decipher = solver_rows.get("decipher-automated")
        if decipher is None:
            continue
        zenith = solver_rows.get("zenith")
        zk = solver_rows.get("zkdecrypto-lite")
        best_external = max(
            [row for row in [zenith, zk] if row is not None],
            key=lambda row: float(row.get("char_accuracy") or 0.0),
            default=None,
        )

        decipher_char = float(decipher.get("char_accuracy") or 0.0)
        decipher_elapsed = float(decipher.get("elapsed_seconds") or 0.0)
        family = str(
            decipher.get("family")
            or decipher.get("cipher_system")
            or ""
        )
        tags = set()
        reason = ""
        frontier_class = ""

        if (
            str(decipher.get("status") or "") in {"error", "failed"}
            or decipher_char < 0.9
            or (
                best_external is not None
                and float(best_external.get("char_accuracy") or 0.0) - decipher_char > 0.1
            )
            or "zodiac" in test_id.lower()
            or "borg" in test_id.lower()
            or "copiale" in test_id.lower()
        ):
            frontier_class = "bad_result"
            if "zodiac" in test_id.lower():
                tags.add("zodiac_like")
            if "borg" in test_id.lower() or "copiale" in test_id.lower():
                tags.add("historical")
            reason = _bad_result_reason(decipher, best_external)
        elif (
            decipher_char >= 0.97
            and best_external is not None
            and float(best_external.get("char_accuracy") or 0.0) >= 0.97
            and (
                decipher_char < 0.999
                or float(best_external.get("char_accuracy") or 0.0) < 0.999
            )
        ):
            frontier_class = "shared_hard"
            if "honb" in test_id.lower() or "homophonic" in family.lower():
                tags.update({"homophonic", "shared_hard"})
            reason = (
                f"Decipher and the best external solver both land in the hard-but-solvable band "
                f"({decipher_char:.1%} vs {float(best_external.get('char_accuracy') or 0.0):.1%})"
            )
        elif decipher_char >= 0.95 and decipher_elapsed >= 300:
            frontier_class = "slow_result"
            if "honb" in test_id.lower() or "homophonic" in family.lower():
                tags.add("short_homophonic")
            reason = (
                f"Decipher reached {decipher_char:.1%} but took "
                f"{decipher_elapsed:.1f}s"
            )
        elif decipher_char >= 0.99 and decipher_elapsed <= 120:
            frontier_class = "known_good"
            if "fr_ss" in test_id.lower() or "it_ss" in test_id.lower():
                tags.add("multilingual")
            if "honb" in test_id.lower() or "homophonic" in family.lower():
                tags.add("medium_homophonic")
            reason = (
                f"Decipher reached {decipher_char:.1%} in {decipher_elapsed:.1f}s"
            )
        else:
            continue

        nomination = {
            "test_id": test_id,
            "frontier_class": frontier_class,
            "frontier_tags": sorted(tags),
            "family": family or "unknown",
            "reason": reason,
            "decipher_char_accuracy": decipher_char,
            "decipher_elapsed_seconds": decipher_elapsed,
            "zenith_char_accuracy": float(zenith.get("char_accuracy") or 0.0) if zenith else None,
            "zkdecrypto_char_accuracy": float(zk.get("char_accuracy") or 0.0) if zk else None,
            "source_files": sorted({str(row.get("source_file") or "") for row in solver_rows.values()}),
        }
        nominations.append(nomination)
    return nominations


def _parse_frontier_case(data: dict[str, Any], source: str) -> FrontierCase:
    frontier_class = str(data.get("frontier_class") or "")
    if frontier_class not in FRONTIER_CLASSES:
        raise ValueError(f"{source}: invalid frontier_class {frontier_class!r}")

    target_records = list(data.get("target_records", []) or [])
    context_records = list(data.get("context_records", []) or [])
    synthetic_spec_data = data.get("synthetic_spec")
    if synthetic_spec_data and target_records:
        raise ValueError(f"{source}: case may not define both target_records and synthetic_spec")
    if not target_records and synthetic_spec_data is None:
        raise ValueError(f"{source}: case must define target_records or synthetic_spec")

    synthetic_spec = _parse_synthetic_spec(synthetic_spec_data, source)
    test = BenchmarkTest(
        test_id=str(data["test_id"]),
        track=str(data["track"]),
        cipher_system=str(data.get("cipher_system", "")),
        target_records=target_records,
        context_records=context_records,
        description=str(data.get("description", "")),
    )

    return FrontierCase(
        test=test,
        frontier_class=frontier_class,
        frontier_tags=_list_of_strings(data.get("frontier_tags", []), source, "frontier_tags"),
        expected_solvers=_list_of_strings(data.get("expected_solvers", []), source, "expected_solvers"),
        expected_status_by_solver=_string_map(data.get("expected_status_by_solver", {}), source, "expected_status_by_solver"),
        min_char_accuracy_by_solver=_float_map(data.get("min_char_accuracy_by_solver", {}), source, "min_char_accuracy_by_solver"),
        max_elapsed_seconds_by_solver=_float_map(data.get("max_elapsed_seconds_by_solver", {}), source, "max_elapsed_seconds_by_solver"),
        max_gap_vs_solver=_float_map(data.get("max_gap_vs_solver", {}), source, "max_gap_vs_solver"),
        notes=str(data.get("notes", "")),
        synthetic_spec=synthetic_spec,
        raw=dict(data),
    )


def _parse_synthetic_spec(data: Any, source: str) -> TestSpec | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"{source}: synthetic_spec must be an object")
    return TestSpec(
        language=str(data.get("language", "en")),
        approx_length=int(data.get("approx_length", 100)),
        word_boundaries=bool(data.get("word_boundaries", True)),
        homophonic=bool(data.get("homophonic", False)),
        seed=int(data["seed"]) if data.get("seed") is not None else None,
        topic=str(data.get("topic", "general")),
        frequency_style=str(data.get("frequency_style", "normal")),
    )


def _list_of_strings(value: Any, source: str, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{source}: {field_name} must be a list of strings")
    return list(value)


def _string_map(value: Any, source: str, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"{source}: {field_name} must be a map of strings")
    return dict(value)


def _float_map(value: Any, source: str, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{source}: {field_name} must be a map of numbers")
    out: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not isinstance(raw, (int, float)):
            raise ValueError(f"{source}: {field_name} must be a map of numbers")
        out[key] = float(raw)
    return out


def _normalize_row(row: dict[str, Any], source_file: str) -> dict[str, Any]:
    normalized = dict(row)
    normalized["test_id"] = str(normalized.get("test_id") or "")
    normalized["solver"] = str(normalized.get("solver") or "")
    normalized["solver_key"] = canonical_solver_name(normalized["solver"])
    normalized["status"] = str(normalized.get("status") or "")
    normalized["char_accuracy"] = float(normalized.get("char_accuracy") or 0.0)
    normalized["word_accuracy"] = float(normalized.get("word_accuracy") or 0.0)
    normalized["elapsed_seconds"] = float(normalized.get("elapsed_seconds") or 0.0)
    normalized["source_file"] = source_file
    return normalized


def _better_row(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    c_score = float(candidate.get("char_accuracy") or 0.0)
    cur_score = float(current.get("char_accuracy") or 0.0)
    if c_score != cur_score:
        return c_score > cur_score
    return float(candidate.get("elapsed_seconds") or 0.0) < float(current.get("elapsed_seconds") or 0.0)


def canonical_solver_name(name: Any) -> str:
    text = str(name or "")
    if text.startswith("zenith"):
        return "zenith"
    if text.startswith("zkdecrypto-lite"):
        return "zkdecrypto-lite"
    if text.startswith("decipher-automated"):
        return "decipher-automated"
    return text


def _bad_result_reason(
    decipher: dict[str, Any],
    best_external: dict[str, Any] | None,
) -> str:
    status = str(decipher.get("status") or "")
    char = float(decipher.get("char_accuracy") or 0.0)
    if status in {"error", "failed"}:
        return f"Decipher status={status}"
    if best_external is None:
        return f"Decipher accuracy only {char:.1%}"
    ext_char = float(best_external.get("char_accuracy") or 0.0)
    return (
        f"Decipher accuracy {char:.1%} trails {best_external.get('solver')} "
        f"at {ext_char:.1%}"
    )
