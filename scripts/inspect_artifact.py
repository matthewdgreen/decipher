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
import contextlib
import json
import os
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


DEFAULT_ANALYSIS_MAX_TOKENS = 2_500


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


_EXPECTED_SLOW_TOOL_PREFIXES = (
    "search_",
)
_EXPECTED_SLOW_TOOL_NAMES = {
    "run_python",
}
_UNEXPECTED_SLOW_THRESHOLD_MS = 5_000
_SLOW_THRESHOLD_MS = 30_000


def _tool_expected_to_take_time(name: str) -> bool:
    return name in _EXPECTED_SLOW_TOOL_NAMES or name.startswith(_EXPECTED_SLOW_TOOL_PREFIXES)


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
    declared = (artifact.get("solution") or artifact.get("declared_solution")) is not None
    is_fallback = (
        artifact.get("status") == "fallback_declared"
        or bool(artifact.get("auto_declared"))
    )
    declared_str = f"{declared} (fallback)" if (declared and is_fallback) else str(declared)
    best_branch = artifact.get("best_branch") or "?"

    lines.append("=" * 70)
    lines.append(f"  Model   : {model} ({provider})")
    lines.append(f"  Cipher  : {cipher}  language={lang}")
    lines.append(f"  Iters   : {iters}   declared={declared_str}")
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


def analyze_tool_timing(artifact: dict[str, Any]) -> dict[str, Any]:
    """Summarize per-tool elapsed_ms data from the artifact."""
    calls = artifact.get("tool_calls") or []
    timed_calls = [
        call for call in calls
        if isinstance(call, dict) and int(call.get("elapsed_ms") or 0) > 0
    ]
    if not timed_calls:
        return {
            "has_timing": False,
            "message": "No nonzero per-tool elapsed_ms values found; this may be an older artifact.",
            "total_tool_ms": 0,
            "slow_tool_calls": [],
            "unexpected_slow_tool_calls": [],
            "top_tool_calls": [],
            "by_tool": [],
        }

    by_tool: dict[str, dict[str, Any]] = {}
    slow = []
    unexpected = []
    for call in timed_calls:
        name = str(call.get("tool_name") or "")
        elapsed_ms = int(call.get("elapsed_ms") or 0)
        bucket = by_tool.setdefault(name, {"tool": name, "count": 0, "total_ms": 0, "max_ms": 0})
        bucket["count"] += 1
        bucket["total_ms"] += elapsed_ms
        bucket["max_ms"] = max(bucket["max_ms"], elapsed_ms)
        row = {
            "iteration": call.get("iteration"),
            "tool": name,
            "elapsed_ms": elapsed_ms,
            "elapsed_seconds": round(elapsed_ms / 1000.0, 3),
            "expected_slow": _tool_expected_to_take_time(name),
            "arguments": _trim_obj(call.get("arguments") or {}, 700),
        }
        if elapsed_ms >= _SLOW_THRESHOLD_MS:
            slow.append(row)
        if (
            elapsed_ms >= _UNEXPECTED_SLOW_THRESHOLD_MS
            and not _tool_expected_to_take_time(name)
        ):
            unexpected.append(row)

    by_tool_rows = []
    for row in by_tool.values():
        by_tool_rows.append({
            **row,
            "total_seconds": round(row["total_ms"] / 1000.0, 3),
            "max_seconds": round(row["max_ms"] / 1000.0, 3),
            "mean_seconds": round((row["total_ms"] / row["count"]) / 1000.0, 3),
            "expected_slow": _tool_expected_to_take_time(row["tool"]),
        })
    by_tool_rows.sort(key=lambda item: (-item["total_ms"], item["tool"]))
    top_calls = sorted(
        [
            {
                "iteration": call.get("iteration"),
                "tool": call.get("tool_name"),
                "elapsed_ms": int(call.get("elapsed_ms") or 0),
                "elapsed_seconds": round(int(call.get("elapsed_ms") or 0) / 1000.0, 3),
                "expected_slow": _tool_expected_to_take_time(str(call.get("tool_name") or "")),
            }
            for call in timed_calls
        ],
        key=lambda item: -item["elapsed_ms"],
    )[:12]
    return {
        "has_timing": True,
        "total_tool_ms": sum(int(call.get("elapsed_ms") or 0) for call in timed_calls),
        "total_tool_seconds": round(
            sum(int(call.get("elapsed_ms") or 0) for call in timed_calls) / 1000.0,
            3,
        ),
        "timed_call_count": len(timed_calls),
        "slow_threshold_ms": _SLOW_THRESHOLD_MS,
        "unexpected_slow_threshold_ms": _UNEXPECTED_SLOW_THRESHOLD_MS,
        "slow_tool_calls": sorted(slow, key=lambda item: -item["elapsed_ms"])[:12],
        "unexpected_slow_tool_calls": sorted(unexpected, key=lambda item: -item["elapsed_ms"])[:12],
        "top_tool_calls": top_calls,
        "by_tool": by_tool_rows[:18],
    }


def format_timing_summary(timing: dict[str, Any]) -> str:
    lines = ["Timing:"]
    if not timing.get("has_timing"):
        lines.append(f"  {timing.get('message')}")
        return "\n".join(lines)
    lines.append(
        f"  total tool time: {float(timing.get('total_tool_seconds') or 0.0):.1f}s "
        f"across {timing.get('timed_call_count')} timed calls"
    )
    unexpected = timing.get("unexpected_slow_tool_calls") or []
    if unexpected:
        lines.append("  Unexpectedly slow small tools:")
        for row in unexpected[:8]:
            lines.append(
                f"    iter {row.get('iteration')}: {row.get('tool')} "
                f"{float(row.get('elapsed_seconds') or 0.0):.1f}s"
            )
    slow = timing.get("slow_tool_calls") or []
    if slow:
        lines.append("  Long-running tools:")
        for row in slow[:8]:
            expected = "expected" if row.get("expected_slow") else "unexpected"
            lines.append(
                f"    iter {row.get('iteration')}: {row.get('tool')} "
                f"{float(row.get('elapsed_seconds') or 0.0):.1f}s ({expected})"
            )
    lines.append("  Top tools by cumulative time:")
    for row in (timing.get("by_tool") or [])[:8]:
        marker = "expected-slow" if row.get("expected_slow") else "small"
        lines.append(
            f"    {row['tool']}: total={row['total_seconds']:.1f}s "
            f"max={row['max_seconds']:.1f}s count={row['count']} [{marker}]"
        )
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
    timing = analyze_tool_timing(artifact)

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
        "declared": (artifact.get("solution") or artifact.get("declared_solution")) is not None,
        "auto_declared": (
            artifact.get("status") == "fallback_declared"
            or bool(artifact.get("auto_declared"))
        ),
        "solution": _trim_obj(artifact.get("solution") or artifact.get("declared_solution") or {}, 1600),
        "best_branch": artifact.get("best_branch"),
        "tool_counts": dict(tool_counts.most_common(20)),
        "artifact_tool_counts": dict(tool_call_counts.most_common(40)),
        "gate_hits": dict(gate_counts),
        "analyzer_findings": findings,
        "failed_tool_calls": failed_tool_calls,
        "tool_timing": timing,
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
    stop_reason: str | None = None
    output_may_be_truncated: bool = False
    attempts: int = 1


def _call_llm(
    summary: dict,
    *,
    provider: str | None,
    model: str | None,
    max_tokens: int,
    analysis_mode: str,
    timeout_seconds: float | None = None,
    retry_empty_response: bool = True,
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
                stop_reason=None,
                output_may_be_truncated=False,
            )
        prompt = _analysis_prompt(summary, analysis_mode)
        system = _analysis_system_prompt()
        with _analysis_provider_timeout(resolved_provider, timeout_seconds):
            client = make_model_provider(
                provider=resolved_provider,
                api_key=api_key,
                model=resolved_model,
            )
            response = client.send(
                messages=[{"role": "user", "content": prompt}],
                system=system,
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
            stop_reason=None,
            output_may_be_truncated=False,
        )

    text = _visible_response_text(response)
    usage = response.usage
    stop_reason = _extract_stop_reason(response.raw)
    output_may_be_truncated = (
        bool(max_tokens and usage.output_tokens >= max_tokens)
        or str(stop_reason or "").lower() in {"length", "max_tokens", "max_output_tokens"}
    )
    attempts = 1

    if retry_empty_response and not text and (usage.output_tokens > 0 or output_may_be_truncated):
        retry_max_tokens = max(max_tokens, min(max_tokens * 2, 8_000))
        retry_prompt = (
            f"{prompt}\n\n"
            "The previous analyzer call returned no visible assistant text even "
            f"though the provider reported {usage.output_tokens} output tokens"
            + (f" and stop_reason={stop_reason!r}" if stop_reason else "")
            + ". For this retry, do not spend tokens on hidden reasoning. "
            "Return the final markdown analysis immediately, starting with "
            "`## Verdict`."
        )
        try:
            with _analysis_provider_timeout(resolved_provider, timeout_seconds):
                retry_response = client.send(
                    messages=[{"role": "user", "content": retry_prompt}],
                    system=(
                        f"{system}\n\n"
                        "Important for this call: produce visible markdown only. "
                        "Do not use hidden reasoning or a long preamble."
                    ),
                    max_tokens=retry_max_tokens,
                )
            retry_text = _visible_response_text(retry_response)
            retry_usage = retry_response.usage
            usage = ModelUsage(
                input_tokens=usage.input_tokens + retry_usage.input_tokens,
                output_tokens=usage.output_tokens + retry_usage.output_tokens,
                cache_read_input_tokens=(
                    usage.cache_read_input_tokens
                    + retry_usage.cache_read_input_tokens
                ),
            )
            stop_reason = _extract_stop_reason(retry_response.raw) or stop_reason
            output_may_be_truncated = (
                bool(retry_max_tokens and retry_usage.output_tokens >= retry_max_tokens)
                or str(stop_reason or "").lower()
                in {"length", "max_tokens", "max_output_tokens"}
            )
            attempts = 2
            if retry_text:
                text = (
                    "_Analyzer note: the first LLM call consumed completion "
                    "tokens but returned no visible text, so Decipher retried "
                    "with a visible-output-only instruction._\n\n"
                    f"{retry_text}"
                )
        except (ModelProviderError, Exception) as exc:  # noqa: BLE001
            attempts = 2
            text = (
                "(LLM analysis failed: first call returned no visible text, "
                f"and retry failed: {exc})"
            )

    if not text:
        text = _empty_llm_response_diagnostic(
            provider=resolved_provider,
            model=resolved_model,
            output_tokens=usage.output_tokens,
            stop_reason=stop_reason,
            attempts=attempts,
        )
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
        stop_reason=stop_reason,
        output_may_be_truncated=output_may_be_truncated,
        attempts=attempts,
    )


def _visible_response_text(response: ModelResponse) -> str:
    return "\n".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    ).strip()


@contextlib.contextmanager
def _analysis_provider_timeout(provider: str, timeout_seconds: float | None):
    """Temporarily apply analyzer-specific timeout env vars for adapters."""
    if not timeout_seconds or timeout_seconds <= 0:
        yield
        return
    env_name = None
    if provider == "openrouter":
        env_name = "OPENROUTER_TIMEOUT"
    elif provider == "ollama":
        env_name = "OLLAMA_TIMEOUT"
    if not env_name:
        yield
        return
    old_value = os.environ.get(env_name)
    os.environ[env_name] = str(timeout_seconds)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old_value


def _empty_llm_response_diagnostic(
    *,
    provider: str,
    model: str,
    output_tokens: int,
    stop_reason: str | None,
    attempts: int,
) -> str:
    reason = f" stop_reason={stop_reason!r}" if stop_reason else ""
    return (
        "(LLM analysis failed: provider returned no visible assistant text "
        f"after {attempts} attempt(s), while reporting output_tokens="
        f"{output_tokens}.{reason} This commonly happens with some gateway or "
        "reasoning-preview models when the completion budget is consumed by "
        "hidden/provider-specific reasoning before any visible answer is "
        "emitted. Try a larger --analysis-max-tokens value or a non-reasoning "
        "model for artifact analysis.)"
    )


def _extract_stop_reason(raw_response: Any) -> str | None:
    """Best-effort provider-neutral stop reason for LLM analyzer calls."""
    if raw_response is None:
        return None
    stop_reason = getattr(raw_response, "stop_reason", None)
    if stop_reason:
        return str(stop_reason)
    choices = getattr(raw_response, "choices", None) or []
    if choices:
        reason = getattr(choices[0], "finish_reason", None)
        if reason:
            return str(reason)
    candidates = getattr(raw_response, "candidates", None) or []
    if candidates:
        reason = getattr(candidates[0], "finish_reason", None)
        if reason:
            return str(reason)
    return None


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
        "benchmark scores, tool timeline, per-tool timing, non-LLM analyzer "
        "findings, branch scores, automated preflight, failed tool calls, "
        "repair agenda, and decryption previews.\n\n"
        f"{detail}\n\n"
        "Requirements:\n"
        "- Distinguish runtime-visible evidence from post-hoc ground-truth grading.\n"
        "- Name specific tools and iterations when possible.\n"
        "- Call out wasted iterations, unexpectedly slow small tools, wrong cipher-family searches, premature declarations, bad basin repair, blocked tools, and missed high-value tools.\n"
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
    timeout_seconds: float | None = None,
    retry_empty_response: bool = True,
) -> None:
    artifact = load(path)
    timeline = build_timeline(artifact)

    print(format_header(artifact))
    print()
    print(format_tool_summary(timeline))
    print()
    print(format_timing_summary(analyze_tool_timing(artifact)))
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
            timeout_seconds=timeout_seconds,
            retry_empty_response=retry_empty_response,
        )
        print(f"LLM analysis ({result.provider}/{result.model}):")
        print(result.text)
        print()
        print(
            "LLM usage: "
            f"input={result.input_tokens} output={result.output_tokens} "
            f"cache_read={result.cache_read_tokens} "
            f"estimated_cost=${result.estimated_cost_usd:.4f}"
            + (f" attempts={result.attempts}" if result.attempts > 1 else "")
            + (f" stop_reason={result.stop_reason}" if result.stop_reason else "")
        )
        if result.output_may_be_truncated:
            print(
                "Warning: LLM analysis may be truncated; output reached the "
                f"--analysis-max-tokens limit ({max_tokens}). Re-run with a "
                "larger value, for example --analysis-max-tokens 5000."
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
        default=DEFAULT_ANALYSIS_MAX_TOKENS,
        help=(
            "Maximum output tokens for LLM analysis. Increase this if the "
            "diagnosis ends mid-sentence."
        ),
    )
    parser.add_argument(
        "--analysis-timeout",
        type=float,
        default=None,
        help=(
            "Optional per-request timeout in seconds for adapters that expose "
            "timeouts, currently OpenRouter and Ollama."
        ),
    )
    parser.add_argument(
        "--analysis-no-empty-retry",
        action="store_true",
        help=(
            "Do not retry when a provider reports output tokens but returns no "
            "visible text. Useful for slow/free gateway models."
        ),
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
            args.analysis_timeout,
            not args.analysis_no_empty_retry,
        )


if __name__ == "__main__":
    main()
