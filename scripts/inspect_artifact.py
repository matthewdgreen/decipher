#!/usr/bin/env python3
"""inspect_artifact.py — compact human-readable summary of a v2 agent run artifact.

Usage:
    python scripts/inspect_artifact.py artifacts/foo/bar.json
    python scripts/inspect_artifact.py artifacts/foo/bar.json --analyze      # LLM narrative
    python scripts/inspect_artifact.py artifacts/foo/bar.json --analyze --provider openai --model gpt-5.4
    python scripts/inspect_artifact.py artifacts/*.json                       # batch
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from agent.model_provider import (  # noqa: E402
    ModelResponse,
    ModelProviderError,
    ModelUsage,
    TextBlock,
    default_model_for_provider,
    estimate_provider_cost,
    infer_provider_from_model,
    make_model_provider,
)
from artifact.analyzer import analyze_artifact, summarize_findings  # noqa: E402


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _text(block: Any) -> str:
    if isinstance(block, dict) and block.get("type") == "text":
        return block.get("text", "")
    return ""


def _result_text(block: Any) -> str:
    """Extract text from a tool_result content block."""
    c = block.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(_text(b) for b in c)
    return ""


def _parse_result(text: str) -> dict:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Build a compact per-iteration timeline
# ---------------------------------------------------------------------------

def build_timeline(artifact: dict) -> list[dict]:
    """Return one dict per *iteration* (= one assistant turn)."""
    messages = artifact.get("messages", [])

    # Walk messages in order; assign iteration counter to each assistant turn.
    # Tool results immediately follow the assistant turn they belong to.
    timeline: list[dict] = []
    pending_tool_use: list[dict] = []   # tool_use blocks in current assistant turn
    iter_num = 0

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        if role == "assistant":
            iter_num += 1
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input", {}),
                        "result": None,   # filled in when we see the tool_result
                    })
            entry = {
                "iter": iter_num,
                "reasoning": " ".join(text_parts).strip()[:500],
                "tools": tool_calls,
            }
            timeline.append(entry)
            pending_tool_use = tool_calls

        elif role == "user" and timeline:
            # Scan for tool_result blocks and match back to pending tool_use
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                result_id = block.get("tool_use_id")
                result_text = _result_text(block)
                result_obj = _parse_result(result_text)
                for tc in pending_tool_use:
                    if tc.get("id") == result_id:
                        tc["result"] = result_obj
                        tc["result_raw"] = result_text[:400]
                        break

    return timeline


# ---------------------------------------------------------------------------
# Extract key signals from timeline entries
# ---------------------------------------------------------------------------

def _score_from_result(result: dict | None) -> dict | None:
    if not result:
        return None
    # Various result shapes
    for key in ("after_scores", "scores", "signals"):
        v = result.get(key)
        if isinstance(v, dict):
            dr = v.get("dictionary_rate") or v.get("dict_rate")
            q = v.get("quadgram_loglik_per_gram") or v.get("quad")
            if dr is not None or q is not None:
                return {"dr": dr, "q": q}
    dr = result.get("dictionary_rate") or result.get("dict_rate")
    q = result.get("quadgram_loglik_per_gram") or result.get("quad")
    if dr is not None or q is not None:
        return {"dr": dr, "q": q}
    return None


def _gate_hit(result: dict | None) -> str | None:
    if result and result.get("reason") == "tool_gated":
        return result.get("attempted_tool", "?")
    return None


def _is_search_tool(name: str) -> bool:
    return name.startswith("search_")


def _result_dict_from_call(call: dict[str, Any]) -> dict[str, Any]:
    result = call.get("result")
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return _parse_result(result)
    return {}


# ---------------------------------------------------------------------------
# Format a single timeline entry as a compact line
# ---------------------------------------------------------------------------

def _fmt_score(s: dict | None) -> str:
    if not s:
        return ""
    parts = []
    if s.get("dr") is not None:
        parts.append(f"dr={s['dr']:.3f}")
    if s.get("q") is not None:
        parts.append(f"q={s['q']:.2f}")
    return " ".join(parts)


def format_timeline(timeline: list[dict]) -> str:
    lines: list[str] = []
    for entry in timeline:
        iter_n = entry["iter"]
        tools = entry.get("tools", [])
        tool_names = [t["name"] for t in tools]

        # Collect scores, gate hits, key outcomes
        scores: list[str] = []
        gates: list[str] = []
        notes: list[str] = []
        for tc in tools:
            r = tc.get("result")
            gate = _gate_hit(r)
            if gate:
                gates.append(gate)
                continue
            s = _score_from_result(r)
            sf = _fmt_score(s)
            if sf:
                scores.append(f"{tc['name']}→{sf}")
            # Special: quagmire budget class
            if tc["name"] == "search_quagmire3_keyword_alphabet" and isinstance(r, dict):
                bc = r.get("budget_class") or r.get("budget_sufficiency", "")
                cands = r.get("top_candidates") or []
                # Higher (less negative) score = better candidate
                best_score = max((c.get("score", float("-inf")) for c in cands), default=None)
                note = f"quag3 budget={bc}"
                if best_score is not None and best_score != float("-inf"):
                    note += f" best_score={best_score:.4f}"
                notes.append(note)
            # search_periodic_polyalphabetic
            if tc["name"] == "search_periodic_polyalphabetic" and isinstance(r, dict):
                cands = r.get("top_candidates") or []
                if cands:
                    best = cands[0]
                    notes.append(
                        f"periodic variant={best.get('variant')} "
                        f"period={best.get('period')} "
                        f"score={best.get('score', '?'):.4f}"
                    )

        # Build the line
        tool_str = ", ".join(tool_names) if tool_names else "(no tools)"
        parts = [f"iter {iter_n:>2}  {tool_str}"]
        if gates:
            parts.append(f"  ⛔ GATED: {', '.join(gates)}")
        if scores:
            parts.append(f"  [{', '.join(scores)}]")
        for note in notes:
            parts.append(f"  ★ {note}")
        # Reasoning snippet (first 100 chars)
        r_text = entry.get("reasoning", "")
        if r_text:
            snippet = r_text.replace("\n", " ")[:100]
            parts.append(f'\n       "{snippet}…"')
        lines.append("".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Header summary
# ---------------------------------------------------------------------------

def format_header(artifact: dict) -> str:
    lines: list[str] = []
    model = artifact.get("model") or "?"
    provider = artifact.get("provider") or "?"
    lang = artifact.get("language") or "?"
    cipher = artifact.get("cipher_system") or artifact.get("test_id") or "?"
    iters = artifact.get("iterations_used") or "?"
    char_acc = artifact.get("char_accuracy")
    word_acc = artifact.get("word_accuracy")
    declared = artifact.get("declared_solution") is not None
    best_branch = artifact.get("best_branch") or "?"

    lines.append("=" * 70)
    lines.append(f"  Model   : {model} ({provider})")
    lines.append(f"  Cipher  : {cipher}  language={lang}")
    lines.append(f"  Iters   : {iters}   declared={declared}")
    if char_acc is not None:
        lines.append(f"  Accuracy: char={char_acc:.1%}  word={word_acc:.1%}" if word_acc is not None else f"  Accuracy: char={char_acc:.1%}")
    lines.append(f"  Branch  : {best_branch}")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool usage summary
# ---------------------------------------------------------------------------

def format_tool_summary(timeline: list[dict]) -> str:
    counts: Counter = Counter()
    gate_counts: Counter = Counter()
    for entry in timeline:
        for tc in entry.get("tools", []):
            r = tc.get("result")
            gate = _gate_hit(r)
            if gate:
                gate_counts[tc["name"]] += 1
            else:
                counts[tc["name"]] += 1

    lines = ["Tool calls:"]
    for name, count in counts.most_common():
        lines.append(f"  {count:>3}× {name}")
    if gate_counts:
        lines.append("Gate hits (wasted iters):")
        for name, count in gate_counts.most_common():
            lines.append(f"  {count:>3}× {name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact machine-readable summary dict for LLM prompt
# ---------------------------------------------------------------------------

def build_llm_summary(artifact: dict, timeline: list[dict]) -> dict:
    """Build a compact JSON-serialisable dict to feed to an LLM for analysis."""
    tool_counts: Counter = Counter()
    gate_counts: Counter = Counter()
    search_results: list[dict] = []
    final_scores: dict = {}
    tool_calls = artifact.get("tool_calls", [])
    findings = summarize_findings(analyze_artifact(artifact))

    for entry in timeline:
        for tc in entry.get("tools", []):
            r = tc.get("result") or {}
            gate = _gate_hit(r)
            if gate:
                gate_counts[tc["name"]] += 1
                continue
            tool_counts[tc["name"]] += 1
            name = tc["name"]
            if _is_search_tool(name):
                s = _score_from_result(r)
                search_results.append({
                    "iter": entry["iter"],
                    "tool": name,
                    "score": s,
                    "budget_class": r.get("budget_class"),
                    "budget_sufficiency": r.get("budget_sufficiency"),
                    "top_score": (
                        # Scores are negative log-likelihoods; higher (less negative) = better
                        max(
                            (c.get("score", float("-inf")) for c in (r.get("top_candidates") or [])),
                            default=None,
                        )
                    ),
                    "preview": (
                        (r.get("top_candidates") or [{}])[0].get("preview", "")[:80]
                        if r.get("top_candidates") else
                        r.get("decoded_preview", "")[:80]
                    ),
                })
            # Track score panel results as final state
            if name == "score_panel" and r.get("signals"):
                s = r["signals"]
                final_scores = {
                    "iter": entry["iter"],
                    "branch": r.get("branch"),
                    "dict_rate": s.get("dictionary_rate"),
                    "quad": s.get("quadgram_loglik_per_gram"),
                }

    tool_call_counts = Counter(str(call.get("tool_name") or "") for call in tool_calls)
    failed_tool_calls = [
        {
            "iteration": call.get("iteration"),
            "tool": call.get("tool_name"),
            "status": _result_dict_from_call(call).get("status"),
            "error": _result_dict_from_call(call).get("error")
            or _result_dict_from_call(call).get("message")
            or str(call.get("result") or "")[:240],
            "arguments": _trim_obj(call.get("arguments") or {}, 700),
        }
        for call in tool_calls
        if _tool_call_failed(call)
    ][:18]
    branch_scores = _branch_score_summary(artifact)
    preflight = _automated_preflight_summary(artifact)
    cipher_hypotheses = _trim_obj(artifact.get("cipher_hypotheses") or [], 2200)
    repair_agenda = _trim_obj(artifact.get("repair_agenda") or [], 2200)
    final_decryption = (
        artifact.get("decryption")
        or artifact.get("final_decryption")
        or artifact.get("best_decryption")
        or ""
    )
    final_summary = artifact.get("final_summary") or artifact.get("solution_summary") or ""

    return {
        "model": artifact.get("model"),
        "provider": artifact.get("provider"),
        "test_id": artifact.get("test_id"),
        "status": artifact.get("status"),
        "cipher_system": artifact.get("cipher_system"),
        "language": artifact.get("language"),
        "char_accuracy": artifact.get("char_accuracy"),
        "word_accuracy": artifact.get("word_accuracy"),
        "score_meaning": "char_accuracy and word_accuracy are post-hoc comparisons to known benchmark plaintext when ground truth exists; they are not intrinsic solver confidence.",
        "iterations_used": artifact.get("iterations_used"),
        "declared": artifact.get("declared_solution") is not None,
        "solution": _trim_obj(artifact.get("solution") or artifact.get("declared_solution") or {}, 1600),
        "best_branch": artifact.get("best_branch"),
        "tool_counts": dict(tool_counts.most_common(20)),
        "artifact_tool_counts": dict(tool_call_counts.most_common(40)),
        "gate_hits": dict(gate_counts),
        "analyzer_findings": findings,
        "failed_tool_calls": failed_tool_calls,
        "branch_scores": branch_scores,
        "automated_preflight": preflight,
        "cipher_hypotheses": cipher_hypotheses,
        "repair_agenda": repair_agenda,
        "search_results": search_results,
        "final_scores": final_scores,
        "initial_fingerprint": _extract_fingerprint(artifact),
        "final_summary": str(final_summary)[:2400],
        "final_decryption_preview": str(final_decryption)[:2000],
        "final_decryption_tail": str(final_decryption)[-900:] if final_decryption else "",
        "reasoning_snippets": [
            e.get("reasoning", "")[:200]
            for e in timeline
            if e.get("reasoning")
        ][:6],
        "timeline": _trim_obj(timeline, 8000),
    }


def _tool_call_failed(call: dict[str, Any]) -> bool:
    result = _result_dict_from_call(call)
    status = str(result.get("status") or "").lower()
    if status in {"error", "failed", "rejected"}:
        return True
    if result.get("error") or result.get("exception"):
        return True
    raw = str(call.get("result") or "")
    return "traceback" in raw.lower() or "error:" in raw.lower()


def _branch_score_summary(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    branches = artifact.get("branches") or artifact.get("branch_scores") or []
    rows = []
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        rows.append({
            "name": branch.get("name") or branch.get("branch"),
            "char_accuracy": branch.get("char_accuracy"),
            "word_accuracy": branch.get("word_accuracy"),
            "mapped_count": branch.get("mapped_count"),
            "tags": branch.get("tags"),
            "metadata": _trim_obj(branch.get("metadata") or {}, 800),
            "preview": str(branch.get("decryption") or branch.get("preview") or "")[:700],
        })
    rows.sort(key=lambda item: float(item.get("char_accuracy") or 0.0), reverse=True)
    return rows[:16]


def _automated_preflight_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    preflight = artifact.get("automated_preflight")
    if not isinstance(preflight, dict):
        return {}
    return {
        "status": preflight.get("status"),
        "solver": preflight.get("solver"),
        "branch": preflight.get("branch"),
        "scores": _trim_obj(preflight.get("scores") or {}, 1000),
        "quality": _trim_obj(preflight.get("quality") or {}, 1000),
        "metadata": _trim_obj(preflight.get("metadata") or {}, 1200),
        "decryption_preview": str(preflight.get("decryption") or "")[:900],
    }


def _trim_obj(value: Any, limit: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= limit:
        return value
    return {"_truncated_json": text[:limit] + "..."}


def _extract_fingerprint(artifact: dict) -> str:
    """Pull the cipher fingerprint block from the first user message."""
    messages = artifact.get("messages", [])
    if not messages:
        return ""
    first = messages[0]
    content = first.get("content", "")
    if isinstance(content, str):
        idx = content.find("Cipher-type fingerprint")
        if idx != -1:
            return content[idx : idx + 600]
    return ""


# ---------------------------------------------------------------------------
# LLM narrative analysis
# ---------------------------------------------------------------------------

@dataclass
class LLMAnalysisResult:
    provider: str
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    estimated_cost_usd: float


def _call_llm(
    summary: dict,
    *,
    provider: str | None,
    model: str | None,
    max_tokens: int,
    analysis_mode: str,
) -> LLMAnalysisResult:
    resolved_provider = infer_provider_from_model(model, provider)
    resolved_model = model or default_model_for_provider(resolved_provider)
    try:
        from cli import _probe_api_key  # type: ignore

        api_key = "" if resolved_provider == "ollama" else _probe_api_key(resolved_provider)
        if not api_key and resolved_provider != "ollama":
            return LLMAnalysisResult(
                provider=resolved_provider,
                model=resolved_model,
                text=(
                    f"(LLM analysis failed: no API key configured for provider "
                    f"{resolved_provider!r})"
                ),
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                estimated_cost_usd=0.0,
            )
        client = make_model_provider(
            provider=resolved_provider,
            api_key=api_key,
            model=resolved_model,
        )
        response = client.send(
            messages=[{"role": "user", "content": _analysis_prompt(summary, analysis_mode)}],
            system=_analysis_system_prompt(),
            max_tokens=max_tokens,
        )
    except (ModelProviderError, Exception) as exc:  # noqa: BLE001
        return LLMAnalysisResult(
            provider=resolved_provider,
            model=resolved_model,
            text=f"(LLM analysis failed: {exc})",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            estimated_cost_usd=0.0,
        )

    text = "\n".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    ).strip()
    usage = response.usage
    cost = estimate_provider_cost(
        resolved_provider,
        resolved_model,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_input_tokens,
    )
    return LLMAnalysisResult(
        provider=resolved_provider,
        model=resolved_model,
        text=text or "(empty response)",
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens,
        estimated_cost_usd=cost,
    )


def _analysis_system_prompt() -> str:
    return (
        "You are a senior cryptanalysis engineering reviewer. Your job is to "
        "diagnose Decipher agent artifacts. Be concrete, skeptical, and useful. "
        "Never treat post-hoc ground-truth scores as evidence the agent had at "
        "runtime; use them only to grade the completed run."
    )


def _analysis_prompt(summary: dict[str, Any], analysis_mode: str) -> str:
    detail = (
        "Give a detailed report with sections: Verdict, What happened, Earliest "
        "bad decision, Tool-use diagnosis, Scoring/branch diagnosis, What the "
        "agent should have done, and Tooling improvements."
        if analysis_mode == "deep"
        else "Give a concise but specific report with sections: Verdict, Failure/success mode, Earliest decision point, and Best tooling improvement."
    )
    return (
        "Review this Decipher artifact summary. The JSON includes post-hoc "
        "benchmark scores, tool timeline, non-LLM analyzer findings, branch "
        "scores, automated preflight, failed tool calls, repair agenda, and "
        "decryption previews.\n\n"
        f"{detail}\n\n"
        "Requirements:\n"
        "- Distinguish runtime-visible evidence from post-hoc ground-truth grading.\n"
        "- Name specific tools and iterations when possible.\n"
        "- Call out wasted iterations, wrong cipher-family searches, premature declarations, bad basin repair, blocked tools, and missed high-value tools.\n"
        "- Suggest concrete changes to prompts, tools, scoring, gates, or search budgets.\n"
        "- If the run succeeded, still identify fragility and how to make success more reliable.\n\n"
        f"```json\n{json.dumps(summary, indent=2, ensure_ascii=False, default=str)}\n```"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def inspect_one(
    path: Path,
    analyze: bool,
    provider: str | None,
    llm_model: str | None,
    max_tokens: int,
    analysis_mode: str,
) -> None:
    artifact = load(path)
    timeline = build_timeline(artifact)

    print(format_header(artifact))
    print()
    print(format_tool_summary(timeline))
    print()
    print("─" * 70)
    print("Iteration timeline:")
    print(format_timeline(timeline))
    print()

    if analyze:
        summary = build_llm_summary(artifact, timeline)
        print("─" * 70)
        resolved_provider = infer_provider_from_model(llm_model, provider)
        resolved_model = llm_model or default_model_for_provider(resolved_provider)
        print(
            f"Performing LLM analysis ({resolved_provider}/{resolved_model})... "
            "This may take a moment.",
            flush=True,
        )
        result = _call_llm(
            summary,
            provider=provider,
            model=llm_model,
            max_tokens=max_tokens,
            analysis_mode=analysis_mode,
        )
        print(f"LLM analysis ({result.provider}/{result.model}):")
        print(result.text)
        print()
        print(
            "LLM usage: "
            f"input={result.input_tokens} output={result.output_tokens} "
            f"cache_read={result.cache_read_tokens} "
            f"estimated_cost=${result.estimated_cost_usd:.4f}"
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact human-readable summary of a v2 agent run artifact."
    )
    parser.add_argument("artifacts", nargs="+", help="Artifact JSON file(s)")
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Call an LLM to provide a compact failure-mode narrative.",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model for --analyze. Defaults to the provider default.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider for --analyze: anthropic, openai, gemini, openrouter, or ollama.",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=("standard", "deep"),
        default="standard",
        help="Depth of LLM artifact analysis prompt.",
    )
    parser.add_argument(
        "--analysis-max-tokens",
        type=int,
        default=900,
        help="Maximum output tokens for LLM analysis.",
    )
    args = parser.parse_args()

    for artifact_path in args.artifacts:
        path = Path(artifact_path)
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            continue
        print(f"\n{'━'*70}")
        print(f"  {path}")
        print(f"{'━'*70}")
        inspect_one(
            path,
            args.analyze,
            args.provider,
            args.model,
            args.analysis_max_tokens,
            args.analysis_mode,
        )


if __name__ == "__main__":
    main()
