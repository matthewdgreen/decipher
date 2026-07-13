#!/usr/bin/env python3
"""Ask an LLM to rank candidate plaintext excerpts by German text quality.

This is an offline diagnostic harness. It sends only candidate IDs and text
excerpts to the model; solver scores, repair edits, and ground-truth metrics
are kept out of the prompt and are used only in the saved calibration report.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))
sys.path.insert(0, str(REPO_ROOT))

from agent.model_provider import (  # noqa: E402
    ModelProviderError,
    ModelResponse,
    ModelUsage,
    TextBlock,
    canonical_provider,
    estimate_provider_cost,
    make_model_provider,
)


DEFAULT_SOURCE = "top_variants_by_second_stage_review"


@dataclass(frozen=True)
class CandidateText:
    candidate_id: str
    text: str
    source_rank: int
    source_key: str
    metadata: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Send top-N candidate plaintext excerpts to an LLM and ask it to "
            "rank them by German text quality."
        )
    )
    parser.add_argument("candidate_json", type=Path, help="Repair/candidate JSON file.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="List key to read candidates from.")
    parser.add_argument("--top", type=int, default=8, help="Number of candidates to send.")
    parser.add_argument(
        "--text-mode",
        choices=("preview", "full"),
        default="preview",
        help=(
            "preview reads --text-field from the candidate rows. full rebuilds "
            "full candidate page text from the source multipage experiment."
        ),
    )
    parser.add_argument("--text-field", default="preview", help="Candidate field containing text.")
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=0,
        help="Maximum characters per candidate text; 0 means no truncation.",
    )
    parser.add_argument(
        "--experiment-json",
        type=Path,
        help="Multipage experiment JSON for --text-mode full. Defaults to candidate source_experiment.",
    )
    parser.add_argument("--benchmark-root", default="../cipher_benchmark/benchmark")
    parser.add_argument("--split", default="copiale_tests.jsonl")
    parser.add_argument(
        "--dedupe-text",
        action="store_true",
        help="Drop later candidates whose visible text is identical.",
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--output", type=Path, help="Markdown report path.")
    parser.add_argument("--json-output", type=Path, help="JSON report path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write/print the prompt packet without calling the model.",
    )
    args = parser.parse_args()

    candidates = load_candidates(
        args.candidate_json,
        source_key=args.source,
        text_mode=args.text_mode,
        text_field=args.text_field,
        top=args.top,
        max_text_chars=args.max_text_chars,
        dedupe_text=args.dedupe_text,
        experiment_json=args.experiment_json,
        benchmark_root=Path(args.benchmark_root),
        split=args.split,
    )
    if not candidates:
        raise SystemExit(f"No candidate texts found in {args.candidate_json} key={args.source!r}")

    prompt = build_prompt(candidates)
    system = (
        "You rank damaged candidate plaintext excerpts. Return only the requested "
        "JSON object. Do not use tools. Do not discuss metadata you were not shown."
    )

    result: dict[str, Any] = {
        "input_file": str(args.candidate_json),
        "source": args.source,
        "text_mode": args.text_mode,
        "text_field": args.text_field,
        "top": args.top,
        "provider": canonical_provider(args.provider),
        "model": args.model,
        "candidates": [candidate_report_row(c) for c in candidates],
        "prompt": prompt,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        response_text = ""
        usage = ModelUsage()
        cost = 0.0
        parsed: Any = None
    else:
        print(f"Calling LLM ranker ({canonical_provider(args.provider)}/{args.model})...")
        response, usage = call_llm(
            provider=args.provider,
            model=args.model,
            system=system,
            prompt=prompt,
            max_tokens=args.max_tokens,
        )
        response_text = visible_response_text(response)
        parsed = parse_json_response(response_text)
        cost = estimate_provider_cost(
            canonical_provider(args.provider),
            args.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
        )

    result.update(
        {
            "response_text": response_text,
            "parsed_response": parsed,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_input_tokens,
                "estimated_cost_usd": cost,
            },
        }
    )

    md = render_markdown(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(md)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.json_output}")


def load_candidates(
    path: Path,
    *,
    source_key: str,
    text_mode: str,
    text_field: str,
    top: int,
    max_text_chars: int,
    dedupe_text: bool,
    experiment_json: Path | None,
    benchmark_root: Path,
    split: str,
) -> list[CandidateText]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(source_key)
    if not isinstance(rows, list):
        raise SystemExit(
            f"{path} does not contain list key {source_key!r}. "
            f"Available list keys: {', '.join(list_keys(payload))}"
        )
    full_text_context: FullTextContext | None = None
    if text_mode == "full":
        full_text_context = load_full_text_context(
            candidate_payload=payload,
            explicit_experiment_json=experiment_json,
            benchmark_root=benchmark_root,
            split=split,
        )
    candidates: list[CandidateText] = []
    seen_texts: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        if full_text_context is not None:
            text = full_text_for_candidate(row, full_text_context)
        else:
            text = str(row.get(text_field) or "").strip()
        text = normalize_candidate_text(text)
        if not text:
            continue
        if dedupe_text and text in seen_texts:
            continue
        seen_texts.add(text)
        candidate_id = f"C{len(candidates) + 1:02d}"
        candidates.append(
            CandidateText(
                candidate_id=candidate_id,
                text=text[:max_text_chars] if max_text_chars > 0 else text,
                source_rank=index,
                source_key=source_key,
                metadata=calibration_metadata(row),
            )
        )
        if len(candidates) >= top:
            break
    return candidates


@dataclass(frozen=True)
class FullTextContext:
    pages: list[Any]
    baseline_key: dict[int, int]
    baseline_mask: tuple[str, ...]


def load_full_text_context(
    *,
    candidate_payload: dict[str, Any],
    explicit_experiment_json: Path | None,
    benchmark_root: Path,
    split: str,
) -> FullTextContext:
    from benchmark.loader import BenchmarkLoader  # type: ignore
    from probe_copiale_multipage_global_repair import (  # type: ignore
        parse_key,
        resolve_path,
        selected_label_from_section,
        sibling_artifact_path,
    )
    from run_copiale_multipage_experiment import (  # type: ignore
        build_combined_cipher,
        finalist_rows,
    )

    experiment_path = (
        resolve_path(explicit_experiment_json)
        if explicit_experiment_json
        else resolve_path(Path(str(candidate_payload.get("source_experiment") or "")))
    )
    if not experiment_path.exists():
        raise SystemExit(f"Cannot find source experiment JSON for full text mode: {experiment_path}")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    artifact_path = sibling_artifact_path(experiment_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    test_ids = [str(item) for item in (experiment.get("test_ids") or [])]
    if not test_ids:
        test_ids = [str(item) for item in (candidate_payload.get("test_ids") or [])]
    if not test_ids:
        raise SystemExit("Full text mode needs test_ids in the source experiment or candidate JSON.")
    loader = BenchmarkLoader(resolve_path(benchmark_root))
    tests = {test.test_id: test for test in loader.load_tests(split)}
    _combined, pages = build_combined_cipher(loader, tests, test_ids)
    label = str(candidate_payload.get("label") or "") or selected_label_from_section(
        experiment,
        str(candidate_payload.get("section") or "portfolio_local_repair"),
    )
    selected = next((row for row in finalist_rows(artifact, top_n=10_000) if row.get("_label") == label), None)
    if selected is None:
        raise SystemExit(f"No finalist label {label!r} found in {artifact_path}.")
    baseline_key = parse_key(selected.get("key"))
    baseline_mask = tuple(str(symbol) for symbol in (selected.get("mask") or []))
    return FullTextContext(pages=pages, baseline_key=baseline_key, baseline_mask=baseline_mask)


def full_text_for_candidate(row: dict[str, Any], context: FullTextContext) -> str:
    from probe_copiale_multipage_global_repair import apply_assignment  # type: ignore
    from run_copiale_multipage_experiment import project_pages  # type: ignore

    key = dict(context.baseline_key)
    mask = set(context.baseline_mask)
    for edit in row.get("edits") or []:
        parsed = parse_candidate_edit(str(edit))
        if parsed is None:
            continue
        symbol, _before, target = parsed
        token_id = token_id_for_symbol(context.pages, symbol)
        apply_assignment(symbol, token_id, target, key, mask)
    page_rows = project_pages(pages=context.pages, key=key, mask=tuple(sorted(mask)))
    return "\n".join(str(page.get("decryption") or "") for page in page_rows)


def parse_candidate_edit(edit: str) -> tuple[str, str, str] | None:
    if edit.strip().lower() == "baseline":
        return None
    if ":" not in edit or "->" not in edit:
        return None
    symbol, rest = edit.split(":", 1)
    before, target = rest.split("->", 1)
    return symbol.strip(), before.strip(), target.strip()


def token_id_for_symbol(pages: list[Any], symbol: str) -> int:
    for page in pages:
        for page_symbol, token_id in zip(page.symbols, page.token_ids):
            if page_symbol == symbol:
                return int(token_id)
    raise SystemExit(f"Candidate edit references unknown symbol: {symbol}")


def normalize_candidate_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def list_keys(payload: dict[str, Any]) -> list[str]:
    return sorted(k for k, v in payload.items() if isinstance(v, list))


def calibration_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Return fields useful for post-hoc calibration, never for prompting."""
    metadata: dict[str, Any] = {}
    for key in (
        "post_hoc_char_avg",
        "post_hoc_char_no_target_avg",
        "adjudication_score",
        "adjudication_no_target_score",
        "page_language_quality_avg",
        "page_validation_avg",
        "page_robust_score",
        "word_hypothesis_score",
    ):
        if key in row:
            metadata[key] = row[key]
    if "edits" in row:
        metadata["edits"] = row["edits"]
    if "word_hypotheses" in row:
        metadata["word_hypotheses"] = row["word_hypotheses"]
    second_stage = row.get("second_stage")
    if isinstance(second_stage, dict):
        metadata["second_stage_portfolio_score"] = second_stage.get("portfolio_score")
        metadata["second_stage_flags"] = second_stage.get("flags")
    repair_adjudication = row.get("repair_adjudication")
    if isinstance(repair_adjudication, dict):
        metadata["repair_adjudication_score"] = repair_adjudication.get("adjudication_score")
        metadata["repair_adjudication_no_target_score"] = repair_adjudication.get(
            "adjudication_no_target_score"
        )
        metadata.setdefault(
            "adjudication_score",
            repair_adjudication.get("adjudication_score"),
        )
        metadata.setdefault(
            "adjudication_no_target_score",
            repair_adjudication.get("adjudication_no_target_score"),
        )
    return metadata


def build_prompt(candidates: list[CandidateText]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(f"{candidate.candidate_id}:\n{candidate.text}")
    candidate_block = "\n\n".join(blocks)
    return (
        "You will be given several candidate plaintext excerpts. They are all "
        "damaged outputs from a cipher-solving process.\n\n"
        "Task: rank the candidates by German text quality. Prefer candidates "
        "that read most like coherent German prose across whole phrases and "
        "sentences.\n\n"
        "Be especially careful about the word-island trap. A candidate should "
        "not score highly merely because it contains visible German words such "
        "as DASS, DIE, ARBEIT, HAND, BRUDER, JEDER, or LEHRLING. Those words "
        "only count as evidence if the surrounding words, endings, and phrase "
        "structure also make German sense. Penalize candidates where a good "
        "local word is embedded in nonsense, where neighboring words become "
        "less plausible, or where the text reads like disconnected German-ish "
        "fragments. Prefer the candidate with the best overall contextual "
        "readability, not the most obvious isolated German words.\n\n"
        "Important: Do not assume the text is archaic, dialectal, or from a "
        "specific historical source. Judge it simply as German text quality.\n\n"
        "Return exactly one JSON object with this shape:\n"
        "{\n"
        '  "ranking": [\n'
        '    {"candidate_id": "C01", "rank": 1, "score": 0-100, "reason": "short reason"}\n'
        "  ],\n"
        '  "best_candidate_id": "C01",\n'
        '  "notes": "brief overall note"\n'
        "}\n\n"
        "Candidate texts:\n\n"
        f"{candidate_block}"
    )


def call_llm(
    *,
    provider: str,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int,
) -> tuple[ModelResponse, ModelUsage]:
    from cli import _probe_api_key  # type: ignore

    resolved_provider = canonical_provider(provider)
    api_key = "" if resolved_provider == "ollama" else _probe_api_key(resolved_provider)
    if resolved_provider != "ollama" and not api_key:
        raise SystemExit(f"No API key configured for provider {resolved_provider!r}.")
    try:
        client = make_model_provider(provider=resolved_provider, api_key=api_key, model=model)
        response = client.send(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=max_tokens,
        )
    except ModelProviderError as exc:
        raise SystemExit(f"LLM ranker failed: {exc}") from exc
    return response, response.usage


def visible_response_text(response: ModelResponse) -> str:
    return "\n".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    ).strip()


def parse_json_response(text: str) -> Any:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def candidate_report_row(candidate: CandidateText) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source_key": candidate.source_key,
        "source_rank": candidate.source_rank,
        "text": candidate.text,
        "calibration": candidate.metadata,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# LLM Candidate Text Ranking",
        "",
        "Prompt sent only candidate IDs and text excerpts. Calibration columns below are post-hoc only.",
        "",
        f"- Input: `{result['input_file']}`",
        f"- Source: `{result['source']}`",
        f"- Provider/model: `{result['provider']}/{result['model']}`",
        f"- Dry run: `{result['dry_run']}`",
        "",
    ]
    usage = result.get("usage") or {}
    if not result.get("dry_run"):
        lines.extend(
            [
                "## Usage",
                "",
                f"- Input tokens: `{usage.get('input_tokens', 0)}`",
                f"- Output tokens: `{usage.get('output_tokens', 0)}`",
                f"- Cache-read tokens: `{usage.get('cache_read_tokens', 0)}`",
                f"- Estimated cost: `${float(usage.get('estimated_cost_usd', 0.0)):.4f}`",
                "",
            ]
        )
    parsed = result.get("parsed_response")
    if parsed is not None:
        lines.extend(["## Parsed Ranking", ""])
        ranking = parsed.get("ranking") if isinstance(parsed, dict) else None
        if isinstance(ranking, list):
            lookup = {row["candidate_id"]: row for row in result["candidates"]}
            lines.append("| Rank | Candidate | LLM Score | Post-Hoc Char | AdjNoTarget | Reason |")
            lines.append("|---:|---|---:|---:|---:|---|")
            for item in ranking:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("candidate_id") or "")
                candidate = lookup.get(cid, {})
                calib = candidate.get("calibration") or {}
                lines.append(
                    "| "
                    f"{item.get('rank', '')} | `{cid}` | {item.get('score', '')} | "
                    f"{format_pct(calib.get('post_hoc_char_avg'))} | "
                    f"{format_float(calib.get('adjudication_no_target_score'))} | "
                    f"{escape_cell(str(item.get('reason') or ''))} |"
                )
            lines.append("")
        notes = parsed.get("notes") if isinstance(parsed, dict) else None
        if notes:
            lines.extend(["Notes:", "", str(notes), ""])
    else:
        lines.extend(["## Raw Response", "", result.get("response_text") or "(no response)", ""])

    lines.extend(["## Candidate Texts", ""])
    for candidate in result["candidates"]:
        calib = candidate.get("calibration") or {}
        lines.extend(
            [
                f"### `{candidate['candidate_id']}`",
                "",
                (
                    f"Source rank `{candidate['source_rank']}`; "
                    f"post-hoc char {format_pct(calib.get('post_hoc_char_avg'))}; "
                    f"AdjNoTarget {format_float(calib.get('adjudication_no_target_score'))}."
                ),
                "",
                "```text",
                candidate["text"],
                "```",
                "",
            ]
        )
    if result.get("dry_run"):
        lines.extend(["## Prompt", "", "```text", result["prompt"], "```", ""])
    return "\n".join(lines)


def format_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return ""


def format_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return ""


def escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
