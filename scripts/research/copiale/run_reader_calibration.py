#!/usr/bin/env python3
"""Phase 4a item 4.4 — LLM reader calibration bake-off.

Drives the promoted :mod:`analysis.llm_reader` library over two saved finalist
sources and compares selectors against post-hoc char accuracy (post-hoc =
calibration only, never fed to the reader):

  * scalar ``validation_score_v2`` rank (``validation_score`` for the older
    probe rows -- reported as a deviation);
  * ``LinearLanguageQualityModel`` rank (only if a trained model file exists);
  * LLM reader rank (default ``gpt-5.6-luna``).

Sources:
  * null-mask probe rows
    (``artifacts/copiale_evidence_packet/null_probe_pair_masks_full_rows.jsonl``);
  * the 2b acceptance menus (``artifacts/phase2b_acceptance/decipher/...``),
    whose null-mask finalists carry full decryptions scored post-hoc against
    the artifact ground truth.

For each page the reader reranks the scalar top-N finalist menu (N =
``config.max_candidates``); the report records which selector picks the
oracle-best candidate (top-1) and whether the oracle best is captured in each
selector's top-3. A hard budget cap (default $3 of reader calls) aborts before
exceeding spend; actual spend is always reported.

The reader input firewall is enforced by the library: only neutral candidate
ids + text excerpts + the language name reach the model.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.llm_reader import LLMReaderConfig, rank_candidates  # noqa: E402
from benchmark.scorer import score_decryption  # noqa: E402

DEFAULT_PROBE_ROWS = (
    REPO_ROOT
    / "artifacts"
    / "copiale_evidence_packet"
    / "null_probe_pair_masks_full_rows.jsonl"
)
DEFAULT_ACCEPTANCE_DIR = (
    REPO_ROOT / "artifacts" / "phase2b_acceptance" / "decipher" / "automated_only"
)


@dataclass
class Candidate:
    candidate_id: str
    preview: str
    scalar_score: float | None
    char_accuracy: float
    lqm_score: float | None = None


@dataclass
class PageResult:
    source: str
    page_id: str
    pool_size: int
    scalar_field: str
    oracle_best_char: float
    oracle_best_id: str
    selectors: dict[str, dict[str, Any]] = field(default_factory=dict)
    reader_error: str | None = None
    reader_cost: float = 0.0


def _normalize_preview(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split())


def _dedupe_pool(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for cand in candidates:
        key = _normalize_preview(cand.preview).upper()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _rank_ids_by(candidates: list[Candidate], score_of, *, reverse: bool) -> list[str]:
    scored = [c for c in candidates if score_of(c) is not None]
    scored.sort(key=lambda c: score_of(c), reverse=reverse)
    return [c.candidate_id for c in scored]


def _capture_metrics(ranking: list[str], by_id: dict[str, Candidate], oracle_id: str) -> dict[str, Any]:
    top1 = ranking[0] if ranking else None
    top3 = ranking[:3]
    return {
        "top1_id": top1,
        "top1_char": by_id[top1].char_accuracy if top1 in by_id else None,
        "top1_is_oracle": top1 == oracle_id,
        "top3_captures_oracle": oracle_id in top3,
    }


def _load_lqm(language: str):
    """Return a trained LinearLanguageQualityModel, or None if no file exists."""
    import os

    from analysis.language_scoring import LinearLanguageQualityModel

    raw = os.environ.get("DECIPHER_NULL_MASK_LANGUAGE_QUALITY_MODEL", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        return None
    try:
        model = LinearLanguageQualityModel.load(path)
    except Exception:  # noqa: BLE001
        return None
    return model


def _probe_pages(path: Path, top_n: int) -> list[tuple[str, list[Candidate], str]]:
    pages: list[tuple[str, list[Candidate], str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        page_id = str(payload.get("test_id") or "unknown")
        rows = payload.get("all_rows") or []
        candidates: list[Candidate] = []
        for index, row in enumerate(rows):
            preview = _normalize_preview(row.get("preview") or "")
            char = row.get("char_accuracy")
            if not preview or char is None:
                continue
            candidates.append(
                Candidate(
                    candidate_id=f"{page_id}#{index}",
                    preview=preview,
                    scalar_score=row.get("validation_score"),
                    char_accuracy=float(char),
                )
            )
        # Scalar-produced finalist menu: top-N by scalar validation, deduped.
        candidates = _dedupe_pool(
            sorted(
                [c for c in candidates if c.scalar_score is not None],
                key=lambda c: c.scalar_score,
                reverse=True,
            )
        )[:top_n]
        if candidates:
            pages.append((page_id, candidates, "validation_score"))
    return pages


def _acceptance_pages(root: Path, top_n: int) -> list[tuple[str, list[Candidate], str]]:
    pages: list[tuple[str, list[Candidate], str]] = []
    for artifact_path in sorted(root.rglob("*.json")):
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        ground_truth = payload.get("ground_truth")
        page_id = str(payload.get("test_id") or artifact_path.parent.name)
        steps = payload.get("steps") or []
        null_step = next((s for s in steps if s.get("name") == "search_null_masks"), None)
        if null_step is None or not ground_truth:
            continue
        finalists = null_step.get("top_finalists") or []
        candidates: list[Candidate] = []
        for index, row in enumerate(finalists[:top_n]):
            decryption = row.get("decryption") or ""
            preview = _normalize_preview(row.get("preview") or decryption)
            if not decryption or not preview:
                continue
            score = score_decryption(
                test_id=page_id,
                decrypted=decryption,
                ground_truth=ground_truth,
                agent_score=0.0,
                status="completed",
            )
            candidates.append(
                Candidate(
                    candidate_id=str(row.get("candidate_id") or f"{page_id}#{index}"),
                    preview=preview,
                    scalar_score=row.get("validation_score_v2"),
                    char_accuracy=float(score.char_accuracy),
                )
            )
        candidates = _dedupe_pool(candidates)
        if candidates:
            pages.append((page_id, candidates, "validation_score_v2"))
    return pages


def calibrate_page(
    source: str,
    page_id: str,
    candidates: list[Candidate],
    scalar_field: str,
    *,
    config: LLMReaderConfig,
    api_key: str,
    lqm,
    call_reader: bool,
) -> PageResult:
    by_id = {c.candidate_id: c for c in candidates}
    oracle = max(candidates, key=lambda c: c.char_accuracy)

    result = PageResult(
        source=source,
        page_id=page_id,
        pool_size=len(candidates),
        scalar_field=scalar_field,
        oracle_best_char=oracle.char_accuracy,
        oracle_best_id=oracle.candidate_id,
    )

    scalar_ranking = _rank_ids_by(candidates, lambda c: c.scalar_score, reverse=True)
    result.selectors["scalar"] = _capture_metrics(scalar_ranking, by_id, oracle.candidate_id)

    if lqm is not None:
        for cand in candidates:
            try:
                cand.lqm_score = lqm.score_text(cand.preview)
            except Exception:  # noqa: BLE001
                cand.lqm_score = None
        lqm_ranking = _rank_ids_by(candidates, lambda c: c.lqm_score, reverse=True)
        if lqm_ranking:
            result.selectors["lqm"] = _capture_metrics(lqm_ranking, by_id, oracle.candidate_id)

    if call_reader:
        packets = [{"candidate_id": c.candidate_id, "preview": c.preview} for c in candidates]
        reader = rank_candidates(packets, config, api_key=api_key)
        result.reader_cost = float(reader.usage.get("estimated_cost_usd", 0.0) or 0.0)
        if reader.parse_ok and reader.ranking:
            result.selectors["reader"] = _capture_metrics(reader.ranking, by_id, oracle.candidate_id)
        else:
            result.reader_error = reader.error or "no_ranking"

    return result


def render_report(results: list[PageResult], *, config: LLMReaderConfig, total_cost: float, lqm_present: bool) -> str:
    lines: list[str] = [
        "# Phase 4a (4.4) — LLM Reader Calibration Bake-off",
        "",
        f"- Reader model: `{config.provider}/{config.model}`",
        f"- Pool per page: scalar top-{config.max_candidates} finalists (deduped by preview)",
        f"- Reader input firewall: candidate ids + text excerpts + language only",
        f"- LinearLanguageQualityModel: {'loaded' if lqm_present else 'absent (no trained model file) — column omitted'}",
        f"- Post-hoc char accuracy is calibration-only; never sent to the reader.",
        f"- **Total reader spend: ${total_cost:.4f}**",
        "",
    ]

    for source in ("null_probe_rows", "phase2b_acceptance_menus"):
        page_results = [r for r in results if r.source == source]
        if not page_results:
            continue
        lines += [f"## Source: `{source}`", ""]
        lines.append(
            "| Page | Pool | Oracle best char | Scalar top-1 char | Scalar picks oracle | "
            "Reader top-1 char | Reader picks oracle | Scalar top-3 capture | Reader top-3 capture | Reader |"
        )
        lines.append("|---|---:|---:|---:|:--:|---:|:--:|:--:|:--:|---|")
        for r in page_results:
            scalar = r.selectors.get("scalar", {})
            reader = r.selectors.get("reader")
            reader_note = "ok" if reader else (r.reader_error or "n/a")
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{r.page_id.split('_')[-1]}`",
                        str(r.pool_size),
                        _pct(r.oracle_best_char),
                        _pct(scalar.get("top1_char")),
                        "Y" if scalar.get("top1_is_oracle") else "n",
                        _pct(reader.get("top1_char")) if reader else "—",
                        ("Y" if reader.get("top1_is_oracle") else "n") if reader else "—",
                        "Y" if scalar.get("top3_captures_oracle") else "n",
                        ("Y" if reader.get("top3_captures_oracle") else "n") if reader else "—",
                        reader_note,
                    ]
                )
                + " |"
            )
        lines.append("")
        lines += _aggregate_block(page_results, lqm_present)
        lines.append("")

    return "\n".join(lines)


def _aggregate_block(page_results: list[PageResult], lqm_present: bool) -> list[str]:
    selectors = ["scalar"] + (["lqm"] if lqm_present else []) + ["reader"]
    lines = ["Aggregate (pages where the selector ran):", ""]
    lines.append("| Selector | Top-1 = oracle | Top-3 captures oracle | Mean top-1 char |")
    lines.append("|---|---:|---:|---:|")
    for name in selectors:
        rows = [r.selectors[name] for r in page_results if name in r.selectors]
        if not rows:
            continue
        n = len(rows)
        top1 = sum(1 for m in rows if m.get("top1_is_oracle"))
        top3 = sum(1 for m in rows if m.get("top3_captures_oracle"))
        chars = [m["top1_char"] for m in rows if m.get("top1_char") is not None]
        mean_char = sum(chars) / len(chars) if chars else 0.0
        lines.append(f"| {name} | {top1}/{n} | {top3}/{n} | {_pct(mean_char)} |")
    return lines


def _pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return "—"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4a 4.4 reader calibration bake-off.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--language", default="de")
    parser.add_argument("--top-n", type=int, default=12, help="Finalist pool size (reader max_candidates).")
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--probe-rows", type=Path, default=DEFAULT_PROBE_ROWS)
    parser.add_argument("--acceptance-dir", type=Path, default=DEFAULT_ACCEPTANCE_DIR)
    parser.add_argument("--budget-usd", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="Skip reader calls; scalar/LQM only.")
    parser.add_argument("--output", type=Path, help="Markdown report path.")
    parser.add_argument("--json-output", type=Path, help="JSON report path.")
    args = parser.parse_args()

    from agent.model_provider import canonical_provider

    provider = canonical_provider(args.provider)
    config = LLMReaderConfig(
        provider=provider,
        model=args.model,
        language=args.language,
        max_candidates=args.top_n,
        max_chars_per_candidate=args.max_chars,
    )

    api_key = ""
    if not args.dry_run:
        from cli import _probe_api_key

        api_key = _probe_api_key(provider)
        if not api_key and provider != "ollama":
            raise SystemExit(f"No API key configured for provider {provider!r}.")

    lqm = _load_lqm(args.language)
    lqm_present = lqm is not None

    pages: list[tuple[str, str, list[Candidate], str]] = []
    if args.probe_rows.exists():
        for page_id, cands, field_name in _probe_pages(args.probe_rows, args.top_n):
            pages.append(("null_probe_rows", page_id, cands, field_name))
    else:
        print(f"NOTE: probe rows file absent: {args.probe_rows}")
    if args.acceptance_dir.exists():
        for page_id, cands, field_name in _acceptance_pages(args.acceptance_dir, args.top_n):
            pages.append(("phase2b_acceptance_menus", page_id, cands, field_name))
    else:
        print(f"NOTE: acceptance dir absent: {args.acceptance_dir}")

    results: list[PageResult] = []
    total_cost = 0.0
    for source, page_id, cands, field_name in pages:
        call_reader = not args.dry_run
        if call_reader and total_cost >= args.budget_usd:
            print(f"Budget cap ${args.budget_usd:.2f} reached; remaining pages scalar-only.")
            call_reader = False
        result = calibrate_page(
            source, page_id, cands, field_name,
            config=config, api_key=api_key, lqm=lqm, call_reader=call_reader,
        )
        total_cost += result.reader_cost
        results.append(result)
        print(
            f"[{source}] {page_id}: pool={result.pool_size} "
            f"oracle={result.oracle_best_char:.3f} "
            f"reader_cost=${result.reader_cost:.4f} total=${total_cost:.4f}"
        )

    report = render_report(results, config=config, total_cost=total_cost, lqm_present=lqm_present)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print("\n" + report)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": provider,
            "model": args.model,
            "top_n": args.top_n,
            "total_reader_cost_usd": round(total_cost, 6),
            "lqm_present": lqm_present,
            "pages": [
                {
                    "source": r.source,
                    "page_id": r.page_id,
                    "pool_size": r.pool_size,
                    "scalar_field": r.scalar_field,
                    "oracle_best_char": r.oracle_best_char,
                    "oracle_best_id": r.oracle_best_id,
                    "selectors": r.selectors,
                    "reader_error": r.reader_error,
                    "reader_cost_usd": r.reader_cost,
                }
                for r in results
            ],
        }
        args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.json_output}")

    print(f"\nTOTAL READER SPEND: ${total_cost:.4f} (budget ${args.budget_usd:.2f})")


if __name__ == "__main__":
    main()
