#!/usr/bin/env python3
"""inspect_artifact.py — compact human-readable summary of a v2 agent run artifact.

Usage:
    python scripts/inspect_artifact.py artifacts/foo/bar.json
    python scripts/inspect_artifact.py artifacts/foo/bar.json --analyze      # LLM narrative
    python scripts/inspect_artifact.py artifacts/foo/bar.json --analyze --model claude-haiku-4-5
    python scripts/inspect_artifact.py artifacts/*.json                       # batch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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

    return {
        "model": artifact.get("model"),
        "cipher_system": artifact.get("cipher_system"),
        "language": artifact.get("language"),
        "char_accuracy": artifact.get("char_accuracy"),
        "word_accuracy": artifact.get("word_accuracy"),
        "iterations_used": artifact.get("iterations_used"),
        "declared": artifact.get("declared_solution") is not None,
        "best_branch": artifact.get("best_branch"),
        "tool_counts": dict(tool_counts.most_common(20)),
        "gate_hits": dict(gate_counts),
        "search_results": search_results,
        "final_scores": final_scores,
        "initial_fingerprint": _extract_fingerprint(artifact),
        "reasoning_snippets": [
            e.get("reasoning", "")[:200]
            for e in timeline
            if e.get("reasoning")
        ][:6],
    }


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

def _call_llm(summary: dict, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        return "(anthropic package not available — run `pip install anthropic`)"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try the project's own key-loading chain (env → key file → keychain)
        try:
            _here = Path(__file__).resolve().parent.parent / "src"
            sys.path.insert(0, str(_here))
            from cli import get_api_key  # type: ignore
            api_key = get_api_key("anthropic")
        except Exception:
            pass
    if not api_key:
        return "(No ANTHROPIC_API_KEY found — set env var or configure keychain)"

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "You are an expert cryptanalyst reviewing an automated cipher-solving agent run. "
        "Here is a compact summary of the run:\n\n"
        f"```json\n{json.dumps(summary, indent=2, ensure_ascii=False)}\n```\n\n"
        "In 150–200 words, identify:\n"
        "1. The primary failure mode (or confirm success)\n"
        "2. The earliest decision point where the run went wrong (if it failed)\n"
        "3. One concrete fix that would most improve future runs\n"
        "Be specific about tool names, iteration numbers, and score values. "
        "Do not repeat obvious facts; focus on diagnosis."
    )

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text if msg.content else "(empty response)"
    except Exception as exc:
        return f"(LLM call failed: {exc})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def inspect_one(path: Path, analyze: bool, llm_model: str) -> None:
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
        print(f"LLM analysis ({llm_model}):")
        print(_call_llm(summary, llm_model))
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
        default="claude-haiku-4-5",
        help="LLM model for --analyze (default: claude-haiku-4-5).",
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
        inspect_one(path, args.analyze, args.model)


if __name__ == "__main__":
    main()
