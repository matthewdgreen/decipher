#!/usr/bin/env python3
"""Probe likely logogram/codeword hypotheses from a damaged decipherment.

This harness is intentionally narrow. It preserves cipher symbols that the
current key renders as null/unmapped, asks for or locally tests whole-word
expansions, then rereads all recurrences with each hypothesis installed.

Generation and runtime ranking are ground-truth-free. If benchmark plaintext is
available, post-hoc character accuracy is reported only after hypotheses have
already been generated.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research" / "copiale"))

from agent.model_provider import canonical_provider, estimate_provider_cost  # noqa: E402
from analysis.language_scoring import language_quality_feature_dict  # noqa: E402
from benchmark.loader import BenchmarkLoader, parse_canonical_transcription, resolve_test_language  # noqa: E402
from benchmark.scorer import score_decryption  # noqa: E402
from rank_candidate_texts_with_llm import call_llm, parse_json_response, visible_response_text  # noqa: E402


DEFAULT_CANDIDATES = {
    "en": [
        "THE", "AND", "OF", "THAT", "TO", "IN", "WITH", "WAS", "FOR", "HIS",
        "MASTER", "BROTHER", "BRETHREN", "LODGE", "SIGN", "HAND", "ORDER",
        "WORD", "TOKEN", "LIGHT", "ROOM", "DOOR",
    ],
    "de": [
        "DER", "DIE", "DAS", "UND", "DEN", "DES", "DEM", "EIN", "EINE",
        "EINER", "NICHT", "SICH", "MAN", "MIT", "VON", "ZU", "AUF",
        "BRUDER", "BRUEDER", "MEISTER", "ORDEN", "ZEICHEN", "GRIFF",
        "HAND", "FUSS", "ARBEIT", "LOGE",
    ],
}


@dataclass(frozen=True)
class TokenView:
    symbol: str
    output_start: int
    output_end: int
    rendered: str
    assignment: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_json")
    parser.add_argument("--benchmark-root", default="fixtures/benchmarks/english_copiale_analog")
    parser.add_argument("--split", default="", help="Split JSONL. Defaults to first split found under benchmark root.")
    parser.add_argument("--test-id", default="")
    parser.add_argument("--mode", choices=["local", "llm"], default="local")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--candidate-words", default="")
    parser.add_argument("--max-symbols", type=int, default=16)
    parser.add_argument("--context-chars", type=int, default=44)
    parser.add_argument("--recurrence-limit", type=int, default=8)
    parser.add_argument(
        "--llm-candidate-review-top-words",
        type=int,
        default=10,
        help="In LLM mode, include recurrence rereads for this many candidate words per symbol.",
    )
    parser.add_argument(
        "--llm-candidate-review-recurrences",
        type=int,
        default=6,
        help="In LLM mode, include this many reread contexts per candidate expansion.",
    )
    parser.add_argument("--include-one-letter-symbols", action="store_true")
    parser.add_argument("--min-occurrences", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    started = time.monotonic()
    artifact_path = resolve_path(Path(args.artifact_json))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    benchmark_root = resolve_path(Path(args.benchmark_root))
    split = args.split or first_split(benchmark_root)
    test_id = args.test_id or str(artifact.get("test_id") or artifact.get("cipher_id") or "")
    if not test_id:
        raise SystemExit("Could not infer test id; pass --test-id.")

    loader = BenchmarkLoader(benchmark_root)
    tests = {test.test_id: test for test in loader.load_tests(split)}
    if test_id not in tests:
        available = ", ".join(sorted(tests)[:20])
        raise SystemExit(f"Test id {test_id!r} not found in {split}. Available: {available}")
    test_data = loader.load_test_data(tests[test_id])
    language = resolve_test_language(test_data, str(artifact.get("language") or "") or None)
    cipher_text = parse_canonical_transcription(test_data.canonical_transcription)
    key = parse_artifact_key(artifact)
    mask = selected_mask(artifact)
    views, baseline = render_token_views(cipher_text, key=key, mask=mask)
    groups = suspicious_symbol_groups(
        views,
        include_one_letter_symbols=args.include_one_letter_symbols,
        min_occurrences=args.min_occurrences,
        max_symbols=args.max_symbols,
    )
    contexts = [
        symbol_context_packet(
            symbol=symbol,
            views=symbol_views,
            baseline=baseline,
            context_chars=args.context_chars,
            recurrence_limit=args.recurrence_limit,
        )
        for symbol, symbol_views in groups
    ]
    candidate_words = explicit_candidate_words(args.candidate_words) or DEFAULT_CANDIDATES.get(language, DEFAULT_CANDIDATES["en"])
    llm_packet: dict[str, Any] = {}
    if args.mode == "llm":
        candidate_reviews = candidate_expansion_reviews(
            contexts=contexts,
            candidate_words=candidate_words,
            cipher_text=cipher_text,
            key=key,
            mask=mask,
            context_chars=args.context_chars,
            top_words=args.llm_candidate_review_top_words,
            recurrence_limit=args.llm_candidate_review_recurrences,
        )
        hypotheses, llm_packet = generate_llm_hypotheses(
            contexts=contexts,
            candidate_reviews=candidate_reviews,
            language=language,
            candidate_words=candidate_words,
            args=args,
        )
    else:
        hypotheses = generate_local_hypotheses(
            contexts=contexts,
            candidate_words=candidate_words,
        )
    evaluated = evaluate_hypotheses(
        hypotheses,
        cipher_text=cipher_text,
        key=key,
        mask=mask,
        baseline=baseline,
        ground_truth=test_data.plaintext,
        language=language,
        context_chars=args.context_chars,
        recurrence_limit=args.recurrence_limit,
    )
    payload = {
        "experiment": "logogram_hypothesis_probe",
        "artifact": str(artifact_path),
        "benchmark_root": str(benchmark_root),
        "split": split,
        "test_id": test_id,
        "language": language,
        "mode": args.mode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "baseline": {
            "length": len(baseline),
            "preview": baseline[:300],
            "language_quality": language_quality_feature_dict(baseline, language=language),
            "post_hoc_char": post_hoc_char(baseline, test_data.plaintext, test_id),
        },
        "settings": serializable_settings(args),
        "candidate_symbol_count": len(contexts),
        "candidate_symbols": contexts,
        "llm_packet": llm_packet,
        "hypotheses": hypotheses,
        "top_evaluated": evaluated[: args.top_n],
    }
    markdown = render_markdown(payload)
    output = (
        resolve_path(Path(args.output))
        if args.output
        else artifact_path.with_suffix(".logogram_hypotheses.md")
    )
    json_output = (
        resolve_path(Path(args.json_output))
        if args.json_output
        else output.with_suffix(".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {output}")
    print(f"Wrote {json_output}")


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def first_split(root: Path) -> str:
    split_dir = root / "splits"
    rows = sorted(split_dir.glob("*.jsonl"))
    if not rows:
        raise SystemExit(f"No split JSONL found under {split_dir}")
    return rows[0].name


def parse_artifact_key(artifact: dict[str, Any]) -> dict[int, int]:
    key = artifact.get("key")
    if not isinstance(key, dict):
        selected = selected_null_mask_row(artifact)
        key = selected.get("key") if isinstance(selected, dict) else {}
    parsed = {}
    for raw_key, raw_value in (key or {}).items():
        try:
            parsed[int(raw_key)] = int(raw_value)
        except (TypeError, ValueError):
            continue
    return parsed


def selected_mask(artifact: dict[str, Any]) -> tuple[str, ...]:
    selected = selected_null_mask_row(artifact)
    if isinstance(selected, dict) and isinstance(selected.get("mask"), list):
        return tuple(str(item) for item in selected.get("mask") or [])
    if isinstance(artifact.get("mask"), list):
        return tuple(str(item) for item in artifact.get("mask") or [])
    return ()


def selected_null_mask_row(artifact: dict[str, Any]) -> dict[str, Any]:
    for step in reversed(artifact.get("steps") or []):
        if isinstance(step, dict) and step.get("name") == "search_null_masks":
            selected = step.get("selected")
            if isinstance(selected, dict):
                return selected
    return {}


def render_token_views(cipher_text: Any, *, key: dict[int, int], mask: tuple[str, ...]) -> tuple[list[TokenView], str]:
    masked = set(mask)
    rendered_parts: list[str] = []
    views: list[TokenView] = []
    output_index = 0
    for symbol in cipher_text.alphabet.symbols:
        # Alphabet order is not token order; this loop is not used.
        _ = symbol
    tokens = [token for token in cipher_text.raw.replace("|", " ").split() if token]
    for symbol in tokens:
        token_id = cipher_text.alphabet.id_for(symbol)
        if symbol in masked:
            rendered = ""
            assignment = "<null>"
        else:
            value = key.get(token_id)
            if value is None or value < 0 or value > 25:
                rendered = ""
                assignment = "?"
            else:
                rendered = chr(ord("A") + value)
                assignment = rendered
        start = output_index
        if rendered:
            rendered_parts.append(rendered)
            output_index += len(rendered)
        views.append(TokenView(symbol=symbol, output_start=start, output_end=output_index, rendered=rendered, assignment=assignment))
    return views, "".join(rendered_parts)


def suspicious_symbol_groups(
    views: list[TokenView],
    *,
    include_one_letter_symbols: bool,
    min_occurrences: int,
    max_symbols: int,
) -> list[tuple[str, list[TokenView]]]:
    grouped: dict[str, list[TokenView]] = {}
    for view in views:
        if view.assignment in {"<null>", "?"} or (include_one_letter_symbols and len(view.rendered) == 1):
            grouped.setdefault(view.symbol, []).append(view)
    rows = [
        (symbol, symbol_views)
        for symbol, symbol_views in grouped.items()
        if len(symbol_views) >= min_occurrences
    ]
    rows.sort(
        key=lambda item: (
            0 if item[1][0].assignment in {"<null>", "?"} else 1,
            -len(item[1]),
            item[0],
        )
    )
    return rows[: max(0, max_symbols)]


def symbol_context_packet(
    *,
    symbol: str,
    views: list[TokenView],
    baseline: str,
    context_chars: int,
    recurrence_limit: int,
) -> dict[str, Any]:
    occurrences = []
    for view in views[: max(1, recurrence_limit)]:
        idx = view.output_start
        left = baseline[max(0, idx - context_chars):idx]
        right = baseline[idx:min(len(baseline), idx + context_chars)]
        marker = f"⟦{symbol}:{view.assignment}⟧"
        occurrences.append({
            "output_index": idx,
            "assignment": view.assignment,
            "context": left + marker + right,
        })
    return {
        "symbol": symbol,
        "assignment": views[0].assignment,
        "occurrence_count": len(views),
        "signal": "missing_or_unmapped_symbol" if views[0].assignment in {"<null>", "?"} else "one_letter_symbol",
        "occurrences": occurrences,
    }


def explicit_candidate_words(value: str) -> list[str]:
    if not value.strip():
        return []
    words = []
    for item in re.split(r"[,\s]+", value.upper()):
        cleaned = re.sub(r"[^A-Z]", "", item)
        if cleaned:
            words.append(cleaned)
    return sorted(set(words), key=words.index)


def generate_local_hypotheses(
    *,
    contexts: list[dict[str, Any]],
    candidate_words: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for context in contexts:
        if context.get("signal") != "missing_or_unmapped_symbol":
            continue
        occurrences = int(context.get("occurrence_count") or 0)
        recurrence_confidence = min(0.30, occurrences / 30.0 * 0.30)
        for rank, word in enumerate(candidate_words, start=1):
            candidate_prior = min(0.15, 0.15 / max(1.0, rank ** 0.35))
            rows.append({
                "symbol": context["symbol"],
                "kind": "logogram",
                "plaintext": word,
                "confidence": round(min(0.88, 0.42 + recurrence_confidence + candidate_prior), 3),
                "candidate_rank": rank,
                "status": "candidate",
                "reason": (
                    "Local candidate-word sweep over a missing/unmapped symbol; "
                    f"{occurrences} recurrence(s) make this a logogram-review candidate."
                ),
            })
    return rows


def candidate_expansion_reviews(
    *,
    contexts: list[dict[str, Any]],
    candidate_words: list[str],
    cipher_text: Any,
    key: dict[int, int],
    mask: tuple[str, ...],
    context_chars: int,
    top_words: int,
    recurrence_limit: int,
) -> list[dict[str, Any]]:
    """Build recurrence rereads for candidate logogram expansions.

    This turns the LLM task from open-ended completion into a focused
    recurrence-ranking task: "if S090 were THE/AND/OF, what do all occurrences
    look like?".
    """
    rows: list[dict[str, Any]] = []
    words = candidate_words[: max(1, top_words)]
    for context in contexts:
        if context.get("signal") != "missing_or_unmapped_symbol":
            continue
        symbol = str(context.get("symbol") or "")
        if not symbol:
            continue
        for rank, word in enumerate(words, start=1):
            expanded, expanded_views = render_with_expansion(
                cipher_text,
                key=key,
                mask=mask,
                expansions={symbol: word},
            )
            rows.append({
                "symbol": symbol,
                "candidate_plaintext": word,
                "candidate_rank": rank,
                "occurrence_count": int(context.get("occurrence_count") or 0),
                "rereads": reread_occurrences(
                    symbol=symbol,
                    views=expanded_views,
                    expanded=expanded,
                    context_chars=context_chars,
                    recurrence_limit=recurrence_limit,
                ),
            })
    rows.sort(
        key=lambda row: (
            -int(row.get("occurrence_count") or 0),
            int(row.get("candidate_rank") or 10_000),
            str(row.get("symbol") or ""),
        )
    )
    return rows


def generate_llm_hypotheses(
    *,
    contexts: list[dict[str, Any]],
    candidate_reviews: list[dict[str, Any]],
    language: str,
    candidate_words: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet: dict[str, Any] = {
        "provider": canonical_provider(args.provider),
        "model": args.model,
        "dry_run": bool(args.dry_run),
    }
    prompt = build_llm_prompt(
        contexts=contexts,
        candidate_reviews=candidate_reviews,
        language=language,
        candidate_words=candidate_words,
    )
    packet["prompt"] = prompt
    if args.dry_run:
        return [], packet
    print(f"Calling LLM logogram hypothesis generator ({canonical_provider(args.provider)}/{args.model})...", flush=True)
    response, usage = call_llm(
        provider=args.provider,
        model=args.model,
        system=(
            "You identify logogram/codeword hypotheses in damaged plaintext. "
            "Return only JSON and keep uncertainty explicit."
        ),
        prompt=prompt,
        max_tokens=args.max_tokens,
    )
    text = visible_response_text(response)
    parsed = parse_json_response(text)
    rows = []
    if isinstance(parsed, dict) and isinstance(parsed.get("hypotheses"), list):
        for item in parsed["hypotheses"]:
            row = normalize_llm_hypothesis(item)
            if row:
                rows.append(row)
    packet.update({
        "response_text": text,
        "parsed_response": parsed,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_input_tokens,
            "estimated_cost_usd": estimate_provider_cost(
                canonical_provider(args.provider),
                args.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_input_tokens,
            ),
        },
        "accepted_hypotheses": len(rows),
        "candidate_reviews": candidate_reviews,
    })
    return rows, packet


def build_llm_prompt(
    *,
    contexts: list[dict[str, Any]],
    candidate_reviews: list[dict[str, Any]],
    language: str,
    candidate_words: list[str],
) -> str:
    return (
        "You are reading damaged no-boundary plaintext from a homophonic/nomenclator cipher.\n"
        "A high-probability logogram signal is: an unmapped/null-rendered cipher symbol sits "
        "where the surrounding language appears to be missing an entire word. You must check "
        "all recurrences of the same symbol before assigning a plaintext word.\n\n"
        "For each symbol, either propose a logogram plaintext, mark it as a likely logogram "
        "with unknown plaintext, or reject it as probably just a null. Use only A-Z for "
        "plaintext expansions. Keep confidence calibrated.\n"
        "Important: nearby text may still contain other wrong letters or other missing "
        "logograms. Do not require a candidate expansion to make every recurrence perfect. "
        "Prefer the candidate whose rereads make the strongest global grammatical sense "
        "across multiple occurrences. Use missing_unknown only when no supplied candidate "
        "has a plausible recurrence majority.\n\n"
        f"Language: {language}\n"
        f"Useful candidate words: {', '.join(candidate_words)}\n\n"
        "Return exactly JSON:\n"
        "{\n"
        '  "hypotheses": [\n'
        '    {"symbol": "S090", "kind": "logogram|missing_unknown|null|reject", '
        '"plaintext": "THE", "confidence": 0.92, "status": "strong|weak|unknown|reject", '
        '"reason": "brief recurrence-aware reason"}\n'
        "  ]\n"
        "}\n\n"
        "Candidate symbols and raw recurrence contexts:\n"
        f"{json.dumps(contexts, ensure_ascii=False, indent=2)}\n\n"
        "Candidate expansion rereads to rank:\n"
        f"{json.dumps(candidate_reviews, ensure_ascii=False, indent=2)}"
    )


def normalize_llm_hypothesis(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    symbol = str(item.get("symbol") or "").strip()
    if not re.fullmatch(r"S\d+", symbol):
        return None
    kind = str(item.get("kind") or "logogram").strip().lower()
    plaintext = re.sub(r"[^A-Z]", "", str(item.get("plaintext") or "").upper())
    try:
        confidence = max(0.0, min(1.0, float(item.get("confidence"))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "symbol": symbol,
        "kind": kind,
        "plaintext": plaintext,
        "confidence": confidence,
        "status": str(item.get("status") or "").strip().lower() or "unknown",
        "reason": str(item.get("reason") or "").strip(),
    }


def evaluate_hypotheses(
    hypotheses: list[dict[str, Any]],
    *,
    cipher_text: Any,
    key: dict[int, int],
    mask: tuple[str, ...],
    baseline: str,
    ground_truth: str,
    language: str,
    context_chars: int,
    recurrence_limit: int,
) -> list[dict[str, Any]]:
    base_features = language_quality_feature_dict(baseline, language=language)
    base_lq = scalar_language_quality(base_features)
    rows = []
    for hypothesis in hypotheses:
        if hypothesis.get("kind") not in {"logogram", "missing_unknown"}:
            continue
        plaintext = str(hypothesis.get("plaintext") or "")
        if not plaintext:
            row = dict(hypothesis)
            row.update({
                "evaluated": False,
                "note": "No plaintext expansion supplied; recurrence evidence only.",
            })
            rows.append(row)
            continue
        expanded, expanded_views = render_with_expansion(cipher_text, key=key, mask=mask, expansions={hypothesis["symbol"]: plaintext})
        features = language_quality_feature_dict(expanded, language=language)
        lq = scalar_language_quality(features)
        row = dict(hypothesis)
        row.update({
            "evaluated": True,
            "occurrence_count": sum(1 for view in expanded_views if view.symbol == hypothesis["symbol"]),
            "expanded_length": len(expanded),
            "language_quality": round(lq, 6),
            "language_quality_delta": round(lq - base_lq, 6),
            "post_hoc_char": post_hoc_char(expanded, ground_truth, "logogram_probe"),
            "preview": expanded[:260],
            "recurrence_reread": reread_occurrences(
                symbol=hypothesis["symbol"],
                views=expanded_views,
                expanded=expanded,
                context_chars=context_chars,
                recurrence_limit=recurrence_limit,
            ),
        })
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row.get("confidence") or 0.0),
            int(row.get("occurrence_count") or 0),
            -int(row.get("candidate_rank") or 10_000),
            float(row.get("language_quality_delta") or -999.0),
            float(row.get("post_hoc_char") or 0.0),
        ),
        reverse=True,
    )
    return rows


def render_with_expansion(
    cipher_text: Any,
    *,
    key: dict[int, int],
    mask: tuple[str, ...],
    expansions: dict[str, str],
) -> tuple[str, list[TokenView]]:
    masked = set(mask)
    rendered_parts: list[str] = []
    views: list[TokenView] = []
    output_index = 0
    tokens = [token for token in cipher_text.raw.replace("|", " ").split() if token]
    for symbol in tokens:
        token_id = cipher_text.alphabet.id_for(symbol)
        if symbol in expansions:
            rendered = expansions[symbol]
            assignment = f"<logogram:{rendered}>"
        elif symbol in masked:
            rendered = ""
            assignment = "<null>"
        else:
            value = key.get(token_id)
            if value is None or value < 0 or value > 25:
                rendered = ""
                assignment = "?"
            else:
                rendered = chr(ord("A") + value)
                assignment = rendered
        start = output_index
        if rendered:
            rendered_parts.append(rendered)
            output_index += len(rendered)
        views.append(TokenView(symbol=symbol, output_start=start, output_end=output_index, rendered=rendered, assignment=assignment))
    return "".join(rendered_parts), views


def reread_occurrences(
    *,
    symbol: str,
    views: list[TokenView],
    expanded: str,
    context_chars: int,
    recurrence_limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for view in [row for row in views if row.symbol == symbol][: max(1, recurrence_limit)]:
        left = expanded[max(0, view.output_start - context_chars):view.output_start]
        mid = expanded[view.output_start:view.output_end]
        right = expanded[view.output_end:min(len(expanded), view.output_end + context_chars)]
        rows.append({
            "output_index": view.output_start,
            "context": left + "⟦" + mid + "⟧" + right,
        })
    return rows


def scalar_language_quality(features: dict[str, float]) -> float:
    keys = [
        "dictionary",
        "content_word_quality",
        "language_coherence",
        "language_shape",
        "binary_ngram_fit",
        "word_lattice_quality",
        "letter_diversity",
    ]
    positive = sum(float(features.get(key) or 0.0) for key in keys)
    penalties = (
        float(features.get("segmentation_cost") or 0.0)
        + float(features.get("top_letter_penalty") or 0.0)
        + float(features.get("repetition_penalty") or 0.0)
        + float(features.get("function_overuse_penalty") or 0.0)
    )
    return positive - penalties


def post_hoc_char(decryption: str, ground_truth: str, test_id: str) -> float | None:
    if not ground_truth:
        return None
    score = score_decryption(test_id, decryption, ground_truth, agent_score=0.0, status="completed")
    return round(float(score.char_accuracy), 6)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Logogram Hypothesis Probe",
        "",
        "Runtime generation/ranking is ground-truth-free. Post-hoc character accuracy is calibration only.",
        "",
        f"- Test: `{payload['test_id']}`",
        f"- Language: `{payload['language']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Candidate symbols: `{payload['candidate_symbol_count']}`",
        f"- Elapsed seconds: `{payload['elapsed_seconds']}`",
        "",
        "## Baseline",
        "",
        f"- Post-hoc char: {format_pct(payload.get('baseline', {}).get('post_hoc_char'))}",
        f"- Preview: `{escape_cell(str(payload.get('baseline', {}).get('preview') or '')[:260])}`",
        "",
    ]
    llm = payload.get("llm_packet") if isinstance(payload.get("llm_packet"), dict) else {}
    usage = llm.get("usage") if isinstance(llm.get("usage"), dict) else {}
    if usage:
        lines.extend([
            "## LLM Usage",
            "",
            f"- Input tokens: `{usage.get('input_tokens', 0)}`",
            f"- Output tokens: `{usage.get('output_tokens', 0)}`",
            f"- Estimated cost: `${float(usage.get('estimated_cost_usd') or 0.0):.4f}`",
            "",
        ])
    lines.extend([
        "## Candidate Missing/Collapsed Symbols",
        "",
        "| Symbol | Assignment | Signal | Count | Example Context |",
        "|---|---|---|---:|---|",
    ])
    for row in payload.get("candidate_symbols") or []:
        occurrence = (row.get("occurrences") or [{}])[0]
        lines.append(
            f"| `{row.get('symbol')}` | `{row.get('assignment')}` | `{row.get('signal')}` | "
            f"{row.get('occurrence_count')} | `{escape_cell(str(occurrence.get('context') or '')[:180])}` |"
        )
    lines.extend([
        "",
        "## Top Evaluated Hypotheses",
        "",
        "| Rank | Symbol | Kind | Plaintext | Conf | Occ | LQ Δ | Post-Hoc | Reason |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ])
    for index, row in enumerate(payload.get("top_evaluated") or [], start=1):
        lines.append(
            f"| {index} | `{row.get('symbol')}` | `{row.get('kind')}` | `{row.get('plaintext')}` | "
            f"{float(row.get('confidence') or 0.0):.2f} | {row.get('occurrence_count', '')} | "
            f"{format_float(row.get('language_quality_delta'))} | {format_pct(row.get('post_hoc_char'))} | "
            f"{escape_cell(str(row.get('reason') or ''))} |"
        )
    for row in (payload.get("top_evaluated") or [])[:8]:
        recurrences = row.get("recurrence_reread") if isinstance(row.get("recurrence_reread"), list) else []
        if not recurrences:
            continue
        lines.extend([
            "",
            f"### Recurrence Reread: `{row.get('symbol')}` → `{row.get('plaintext')}`",
            "",
        ])
        for occurrence in recurrences[:6]:
            lines.append(f"- `{escape_cell(str(occurrence.get('context') or ''))}`")
    lines.append("")
    return "\n".join(lines)


def format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return ""


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def serializable_settings(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


if __name__ == "__main__":
    main()
