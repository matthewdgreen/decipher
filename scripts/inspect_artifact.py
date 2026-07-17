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
from types import SimpleNamespace
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
from agent.narrate import NarrateAgentRenderer  # noqa: E402
from artifact.analyzer import analyze_artifact, summarize_findings  # noqa: E402
from investigation.state import attestation_is_positive  # noqa: E402


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
# Shared analyzer helpers (M5.3 Slice 7)
# ---------------------------------------------------------------------------

def _inv_state(artifact: dict) -> dict:
    """The investigation_state sub-dict (v3), or {} for v2/older artifacts."""
    state = artifact.get("investigation_state")
    return state if isinstance(state, dict) else {}


def _short_hash(h: Any) -> str:
    """First 12 chars of a content hash; ``-`` when empty."""
    text = str(h or "")
    return text[:12] if text else "-"


def _attestation_verdict(attestation: dict) -> str:
    """positive/weak/negative — the same rule as format_attestations."""
    if attestation_is_positive(attestation):
        return "positive"
    if attestation.get("reader_accepts"):
        return "weak"
    return "negative"


def derive_run_facts(artifact: dict) -> dict:
    """Provider/model/iterations/branch/declaration/attestation/cost facts
    correct for the v2+v3 RunArtifact shape, with graceful fallbacks for the
    older ad-hoc shapes (automated-runner artifacts carry explicit
    provider/test_id/iterations_used keys — those take precedence)."""
    model = artifact.get("model") or "?"
    provider = artifact.get("provider")
    if not provider and model != "?":
        # Only infer from a KNOWN model; an unknown model reports "?" rather
        # than the inference helper's non-falsy default (nit).
        provider = infer_provider_from_model(model, None)
    provider = provider or "?"
    cipher = (
        artifact.get("cipher_id")
        or artifact.get("test_id")
        or artifact.get("cipher_system")
        or "?"
    )
    loop_version = artifact.get("loop_version") or "v2"
    status = artifact.get("status") or "?"

    iterations = artifact.get("iterations_used")
    if not isinstance(iterations, int):
        iterations = _iterations_used(artifact, artifact.get("loop_events") or [])
    if iterations is None:
        iterations = "?"

    declared = (artifact.get("solution") or artifact.get("declared_solution")) is not None
    fallback = (
        artifact.get("status") == "fallback_declared"
        or bool(artifact.get("auto_declared"))
    )
    final_branch = (
        (artifact.get("branch_roles") or {}).get("declared_or_selected_branch")
        or _declared_branch(artifact)
        or "?"
    )

    solution = artifact.get("solution")
    if isinstance(solution, dict) and isinstance(solution.get("attestation"), dict):
        attestation_status = f"{_attestation_verdict(solution['attestation'])} (declared)"
    elif artifact.get("attestations"):
        attestations = artifact["attestations"]
        n_positive = sum(1 for a in attestations if attestation_is_positive(a))
        attestation_status = f"{len(attestations)} recorded ({n_positive} positive)"
    else:
        attestation_status = "none"

    cost_usd = float(artifact.get("estimated_cost_usd") or 0.0)
    max_cost_usd = artifact.get("max_cost_usd")

    return {
        "model": model,
        "provider": provider,
        "cipher": cipher,
        "loop_version": loop_version,
        "status": status,
        "iterations": iterations,
        "declared": declared,
        "fallback": fallback,
        "final_branch": final_branch,
        "attestation_status": attestation_status,
        "cost_usd": cost_usd,
        "max_cost_usd": (
            float(max_cost_usd) if isinstance(max_cost_usd, (int, float)) else None
        ),
    }


# ---------------------------------------------------------------------------
# Header summary
# ---------------------------------------------------------------------------

def format_header(artifact: dict) -> str:
    facts = derive_run_facts(artifact)
    lang = artifact.get("language") or "?"
    char_acc = artifact.get("char_accuracy")
    word_acc = artifact.get("word_accuracy")
    declared_str = (
        f"{facts['declared']} (fallback)"
        if (facts["declared"] and facts["fallback"])
        else str(facts["declared"])
    )

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"  Model   : {facts['model']} ({facts['provider']})   loop={facts['loop_version']}")
    lines.append(f"  Cipher  : {facts['cipher']}  language={lang}")
    lines.append(f"  Iters   : {facts['iterations']}   status={facts['status']}   declared={declared_str}")
    if char_acc is not None:
        lines.append(
            f"  Accuracy: char={char_acc:.1%}  word={word_acc:.1%}"
            if word_acc is not None
            else f"  Accuracy: char={char_acc:.1%}"
        )
    lines.append(f"  Branch  : {facts['final_branch']}   attestation: {facts['attestation_status']}")
    cost_line = f"  Cost    : ${facts['cost_usd']:.4f}"
    if facts["max_cost_usd"] is not None:
        cost_line += f" / ${facts['max_cost_usd']:.2f} hard ceiling"
    lines.append(cost_line)
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool usage summary
# ---------------------------------------------------------------------------

def format_episodes(artifact: dict) -> str:
    """Minimal episodes table for v3 artifacts (M2). Empty string when none."""
    episodes = artifact.get("episodes") or []
    if not episodes:
        return ""
    lines = ["Episodes:"]
    lines.append(
        f"  {'kind':<10} {'status':<15} {'calls':>5}  {'snapshots':<24} summary"
    )
    for ep in episodes:
        kind = str(ep.get("kind") or "?")
        status = str(ep.get("status") or "?")
        reason = ep.get("failure_reason")
        if reason:
            status = f"{status}:{reason}"
        calls = ep.get("tool_call_count") or 0
        snaps = ", ".join(
            str(s.get("name")) for s in (ep.get("branch_snapshots") or [])
            if isinstance(s, dict)
        )
        summary = str(ep.get("summary") or "").replace("\n", " ")[:60]
        lines.append(f"  {kind:<10} {status:<15} {calls:>5}  {snaps:<24} {summary}")
    return "\n".join(lines)


def format_readings(artifact: dict) -> str:
    """Minimal readings table for v3 artifacts (M3). Empty string when none."""
    readings = artifact.get("readings") or []
    if not readings:
        return ""
    lines = ["Readings:"]
    lines.append(f"  {'id':<14} {'branch':<18} {'conf':>5} {'holes':>5}  preview")
    for r in readings:
        rid = str(r.get("reading_id") or "?")
        branch = str(r.get("branch") or "?")
        conf = r.get("overall_confidence")
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
        holes = len(r.get("holes") or [])
        preview = " ".join(
            str(f.get("text") or "") for f in (r.get("fragments") or [])
            if isinstance(f, dict)
        ).replace("\n", " ")[:50]
        lines.append(f"  {rid:<14} {branch:<18} {conf_s:>5} {holes:>5}  {preview}")
    return "\n".join(lines)


def format_attestations(artifact: dict) -> str:
    """Minimal verify-attestation table for v3 artifacts (M5). Empty when none.

    Marks the declared branch's attestation so a weak-but-declared solve is
    visible. Slice 6: adds a verdict column (positive/weak/negative) and the
    diplomatic verifier fields; legacy records classify via the frozen legacy
    rule and render n/a for absent fields."""
    attestations = artifact.get("attestations") or []
    if not attestations:
        return ""
    solution = artifact.get("solution") or {}
    declared_branch = str(solution.get("branch") or "") if isinstance(solution, dict) else ""
    declared_hash = ""
    if isinstance(solution, dict) and isinstance(solution.get("attestation"), dict):
        declared_hash = str(solution["attestation"].get("content_hash") or "")
    lines = ["Verify attestations:"]
    lines.append(
        f"  {'branch':<18} {'verdict':<9} {'lang':>5} {'recov':>5} "
        f"{'scope':<11} {'repair':<13} {'coher':>5} {'anoms':>5}  gloss"
    )
    for a in attestations:
        branch = str(a.get("branch") or "?")
        coher = a.get("coherence")
        coher_s = str(coher) if isinstance(coher, int) else "n/a"
        anoms = len(a.get("anomalies") or [])
        if attestation_is_positive(a):
            verdict = "positive"
        elif a.get("reader_accepts"):
            verdict = "weak"
        else:
            verdict = "negative"

        def _unit(key: str) -> str:
            v = a.get(key)
            numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
            return f"{float(v):.2f}" if numeric else "n/a"
        lang = _unit("target_language_confidence")
        recov = _unit("semantic_recoverability")
        scope = str(a.get("damage_scope") or "n/a")
        repair = str(a.get("repairability") or "n/a")
        gloss = str(a.get("gloss") or "").replace("\n", " ")[:36]
        is_declared = (
            (declared_hash and a.get("content_hash") == declared_hash)
            or (not declared_hash and branch == declared_branch)
        )
        marker = " *declared" if is_declared else ""
        lines.append(
            f"  {branch:<18} {verdict:<9} {lang:>5} {recov:>5} {scope:<11} "
            f"{repair:<13} {coher_s:>5} {anoms:>5}  {gloss}{marker}"
        )
    return "\n".join(lines)


def format_experiments(artifact: dict) -> str:
    """Minimal experiments table for v3 artifacts (M4). Empty string when none."""
    experiments = artifact.get("experiments") or []
    if not experiments:
        return ""
    lines = ["Experiments:"]
    lines.append(
        f"  {'id':<14} {'type':<18} {'status':<12} {'elapsed':>8}  summary"
    )
    for exp in experiments:
        eid = str(exp.get("experiment_id") or "?")
        etype = str(exp.get("type") or "?")
        status = str(exp.get("status") or "?")
        reason = exp.get("orphan_reason") or exp.get("error")
        if status in {"orphaned", "failed"} and reason:
            status = f"{status}:{str(reason)[:10]}"
        elapsed = exp.get("elapsed_seconds")
        el = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "n/a"
        summary = str(exp.get("summary") or "").replace("\n", " ")[:50]
        lines.append(f"  {eid:<14} {etype:<18} {status:<12} {el:>8}  {summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# M5.3 Slice 7 analyzer sections. Each returns "" when it has nothing to show.
# ---------------------------------------------------------------------------

def format_episode_budgets(artifact: dict) -> str:
    """Per-episode requested vs registered vs effective vs executed budget
    (master 515). Slice-7 ledger keys; pre-Slice-7 episodes render n/a/-."""
    episodes = artifact.get("episodes") or []
    if not episodes:
        return ""
    lines = ["Episode budgets (requested → effective / registered, executed):"]
    lines.append(
        f"  {'kind':<10} {'requested':>9} {'registered':>10} "
        f"{'effective':>9} {'executed':>8} {'skipped':>7} {'elapsed':>8}"
    )
    for ep in episodes:
        kind = str(ep.get("kind") or "?")
        requested = ep.get("requested_max_tool_calls")
        req_s = str(requested) if requested is not None else "-"
        registered = ep.get("registered_max_tool_calls")
        reg_s = str(registered) if registered is not None else "n/a"
        budget = ep.get("budget") if isinstance(ep.get("budget"), dict) else {}
        effective = budget.get("max_tool_calls")
        eff_s = str(effective) if effective is not None else "n/a"
        executed = int(ep.get("tool_call_count") or 0)
        skipped = int(ep.get("suppressed_over_budget_calls") or 0)
        elapsed = ep.get("elapsed_seconds")
        el_s = f"{float(elapsed):.1f}s" if isinstance(elapsed, (int, float)) else "n/a"
        lines.append(
            f"  {kind:<10} {req_s:>9} {reg_s:>10} {eff_s:>9} "
            f"{executed:>8} {skipped:>7} {el_s:>8}"
        )
    return "\n".join(lines)


def format_suppressed_calls(artifact: dict) -> str:
    """Episodes whose host skipped one or more over-budget tool_uses (master
    516). The ledger count is authoritative (Slice 1 writes it)."""
    episodes = [
        ep for ep in (artifact.get("episodes") or [])
        if int(ep.get("suppressed_over_budget_calls") or 0) > 0
    ]
    if not episodes:
        return ""
    lines = ["Suppressed over-budget episode calls:"]
    lines.append(
        f"  {'episode':<14} {'kind':<10} {'suppressed':>10} {'cap':>5} {'executed':>8}"
    )
    for ep in episodes:
        eid = str(ep.get("episode_id") or "?")
        kind = str(ep.get("kind") or "?")
        suppressed = int(ep.get("suppressed_over_budget_calls") or 0)
        budget = ep.get("budget") if isinstance(ep.get("budget"), dict) else {}
        cap = budget.get("max_tool_calls")
        cap_s = str(cap) if cap is not None else "n/a"
        executed = int(ep.get("tool_call_count") or 0)
        lines.append(f"  {eid:<14} {kind:<10} {suppressed:>10} {cap_s:>5} {executed:>8}")
    return "\n".join(lines)


def format_experiment_validation_failures(timeline: list[dict]) -> str:
    """Timeline experiment_submit/collect calls that failed typed validation
    (master 520). experiment dispatches return JSON directly (no ToolCall
    record), so this reads the timeline, not artifact["tool_calls"]."""
    rows: list[dict] = []
    for entry in timeline:
        for tc in entry.get("tools", []):
            name = str(tc.get("name") or "")
            if name not in {"experiment_submit", "experiment_collect"}:
                continue
            result = tc.get("result")
            if not isinstance(result, dict):
                continue
            error = str(result.get("error") or "")
            if not (
                result.get("config_errors")
                or error.startswith("invalid experiment config")
                or error.startswith("unknown experiment type")
            ):
                continue
            rows.append({
                "iteration": entry.get("iter"),
                "tool": name,
                "config_errors": result.get("config_errors") or [],
                "corrected_example": result.get("corrected_example") is not None,
            })
    if not rows:
        return ""
    lines = ["Experiment validation failures:"]
    for row in rows:
        errs = row["config_errors"][:2]
        err_text = "; ".join(str(e)[:90] for e in errs) if errs else "-"
        lines.append(
            f"  iter {row['iteration']}: {row['tool']}  {err_text}  "
            f"corrected_example: {'yes' if row['corrected_example'] else 'no'}"
        )
    return "\n".join(lines)


def format_repair_cycles(artifact: dict) -> str:
    """Repair transactions grouped by source content hash (master 517).
    ``repair_transactions`` lives inside investigation_state, not top-level."""
    transactions = _inv_state(artifact).get("repair_transactions") or []
    if not transactions:
        return ""
    groups: dict[str, list[dict]] = {}
    for tx in transactions:
        groups.setdefault(str(tx.get("source_content_hash") or ""), []).append(tx)
    lines = ["Repair cycles (grouped by source content hash):"]
    for source_hash, txs in groups.items():
        ordered = sorted(txs, key=lambda t: int(t.get("created_turn") or 0))
        if any("pair_digest" in t for t in txs):
            pairs = {str(t.get("pair_digest") or "") for t in txs}
            pair_note = f"{len(pairs)} pairs"
        else:
            readings = {str(t.get("reading_id") or "") for t in txs}
            pair_note = f"~{len(readings)} readings"
        statuses = []
        for t in ordered:
            st = str(t.get("status") or "?")
            statuses.append(f"failed({t.get('reason') or '?'})" if st == "failed" else st)
        att_keys = {
            str(t.get("attestation_key")) for t in txs
            if t.get("attestation_key") is not None
        }
        att_s = ", ".join(sorted(att_keys)) if att_keys else "n/a"
        lines.append(f"  {_short_hash(source_hash)}  {len(txs)} tx  {pair_note}")
        lines.append(f"    statuses: {', '.join(statuses)}")
        lines.append(f"    attestation_keys: {att_s}")
    return "\n".join(lines)


def format_saturation(artifact: dict) -> str:
    """Repair-saturation entries + derived exhaustion transition turn (master
    518). Slice-2 field; absent on pre-Slice-2 artifacts."""
    saturation = _inv_state(artifact).get("repair_saturation") or {}
    if not saturation:
        return ""
    transactions = _inv_state(artifact).get("repair_transactions") or []
    lines = ["Repair saturation transitions:"]
    for sat_key, entry in saturation.items():
        if not isinstance(entry, dict):
            continue
        cand = _short_hash(entry.get("candidate_content_hash"))
        att_key = str(entry.get("attestation_key") or "n/a")
        evidence_failures = int(entry.get("evidence_failures") or 0)
        process_total = sum(
            int(v or 0) for v in (entry.get("process_failures") or {}).values()
        )
        readings = int(entry.get("readings") or 0)
        exhausted = bool(entry.get("exhausted"))
        pending = entry.get("pending_experiment_id") or "-"
        created = entry.get("created_turn")
        updated = entry.get("updated_turn")
        lines.append(
            f"  {cand}  att={att_key}  evidence_failures={evidence_failures}  "
            f"process_failures={process_total}  readings={readings}  "
            f"exhausted={exhausted}  pending_experiment_id={pending}  "
            f"turns {created}->{updated}"
        )
        # Transition turn: the counted evidence failure that reached 2.
        count = 0
        transition_turn = None
        for tx in sorted(
            (t for t in transactions if t.get("saturation_key") == sat_key),
            key=lambda t: int(t.get("created_turn") or 0),
        ):
            if tx.get("counted_evidence_failure") is True:
                count += 1
                if count == 2:
                    transition_turn = tx.get("created_turn")
                    break
        if transition_turn is not None:
            lines.append(f"    exhausted at turn {transition_turn}")
    return "\n".join(lines)


def format_repair_transactions(artifact: dict) -> str:
    """Every installed and rejected repair transaction (master 519), in list
    order, with Slice-2 failure classification + Slice-4 acceptance summary."""
    transactions = _inv_state(artifact).get("repair_transactions") or []
    if not transactions:
        return ""
    lines = ["Repair transactions:"]
    for tx in transactions:
        tid = _short_hash(tx.get("transaction_id"))
        status = str(tx.get("status") or "?")
        parts = [f"{tid}  {status}"]
        if status == "failed":
            parts.append(f"reason={tx.get('reason') or '-'}")
        if "failure_class" in tx or "counted_evidence_failure" in tx:
            parts.append(
                f"class={tx.get('failure_class') or 'n/a'} "
                f"counted_evidence={tx.get('counted_evidence_failure')}"
            )
        else:
            parts.append("class=n/a")
        if status == "installed":
            parts.append(
                f"{tx.get('worker_winner') or '-'}->{tx.get('installed_branch') or '-'}"
            )
        retry_of = tx.get("retry_of")
        parts.append(f"retry_of={_short_hash(retry_of) if retry_of else '-'}")
        acceptance = tx.get("acceptance")
        if isinstance(acceptance, dict):
            checks = acceptance.get("checks") or []
            passed = sum(1 for c in checks if c.get("passed"))
            parts.append(f"checks {passed}/{len(checks)}")
            deltas = acceptance.get("score_deltas") or {}
            dr = deltas.get("dict_rate_delta")
            q = deltas.get("quad_delta")
            if isinstance(dr, (int, float)) and not isinstance(dr, bool):
                parts.append(f"dict_rate_delta={dr:+.4f}")
            if isinstance(q, (int, float)) and not isinstance(q, bool):
                parts.append(f"quad_delta={q:+.4f}")
        lines.append("  " + "  ".join(parts))
    return "\n".join(lines)


def format_branch_roles(artifact: dict) -> str:
    """The four distinguished branch roles (master 521, Part A). Empty for v2
    / pre-Slice-7 artifacts. The divergence marker is workflow vs best-scored
    ('workflow branch vs score-selected branch')."""
    roles = artifact.get("branch_roles")
    if not isinstance(roles, dict) or not roles:
        return ""
    best = roles.get("best_scored_branch")
    workflow = roles.get("workflow_branch")

    def _val(v: Any) -> str:
        return str(v) if v is not None else "-"

    marker = (
        "   [differs from best-scored]"
        if (workflow is not None and workflow != best)
        else ""
    )
    lines = ["Branch roles:"]
    lines.append(f"  best_scored_branch         : {_val(best)}")
    lines.append(f"  workflow_branch            : {_val(workflow)}{marker}")
    lines.append(f"  latest_installed_branch    : {_val(roles.get('latest_installed_branch'))}")
    lines.append(f"  declared_or_selected_branch: {_val(roles.get('declared_or_selected_branch'))}")
    return "\n".join(lines)


def format_repair_hypothesis_time(artifact: dict) -> str:
    """Cumulative repair-hypothesis compute time (master 522), read from the
    composite ToolCalls' real ``elapsed_ms`` (execute_composite records it) and
    the ``menu_source`` tally from hypothesis_test_word* results. Lead
    ``repair_transaction`` ToolCalls carry elapsed_ms=0 by design
    (_record_dispatch_result), so this neither double-counts nor misses the
    inner composite work."""
    names = {"hypothesis_test_words", "hypothesis_test_word", "hypothesis_apply_reading"}
    word_names = {"hypothesis_test_words", "hypothesis_test_word"}
    by_tool: dict[str, dict] = {}
    for tc in artifact.get("tool_calls") or []:
        name = str(tc.get("tool_name") or "")
        if name not in names:
            continue
        bucket = by_tool.setdefault(
            name, {"count": 0, "total_ms": 0, "max_ms": 0, "menus": Counter()}
        )
        elapsed_ms = int(tc.get("elapsed_ms") or 0)
        bucket["count"] += 1
        bucket["total_ms"] += elapsed_ms
        bucket["max_ms"] = max(bucket["max_ms"], elapsed_ms)
        if name in word_names:
            menu_source = _result_dict_from_call(tc).get("menu_source")
            if menu_source:
                bucket["menus"][str(menu_source)] += 1
    if not by_tool:
        return ""
    cumulative_s = sum(b["total_ms"] for b in by_tool.values()) / 1000.0
    lines = [f"Repair hypothesis time: {cumulative_s:.1f}s cumulative"]
    for name in sorted(by_tool, key=lambda n: -by_tool[n]["total_ms"]):
        b = by_tool[name]
        line = (
            f"  {name:<24} {b['count']} calls  "
            f"total={b['total_ms'] / 1000.0:.1f}s  max={b['max_ms'] / 1000.0:.1f}s"
        )
        if b["menus"]:
            menus = " ".join(f"{k}={v}" for k, v in sorted(b["menus"].items()))
            line += f"  menus: {menus}"
        lines.append(line)
    return "\n".join(lines)


# v3 composite actions (M3): rendered alongside the tool summary so their usage
# and residence (lead turn vs episode) is visible in the inspector.
_COMPOSITE_TOOL_NAMES = ("hypothesis_apply_reading", "hypothesis_test_word",
                         "branch_adjudicate")


def format_composite_calls(artifact: dict) -> str:
    """Compact rendering of every v3 composite-action call. Empty when none."""
    calls = [
        tc for tc in (artifact.get("tool_calls") or [])
        if str(tc.get("tool_name") or "") in _COMPOSITE_TOOL_NAMES
    ]
    if not calls:
        return ""
    lines = ["Composite actions:"]
    for tc in calls:
        name = str(tc.get("tool_name") or "?")
        where = f"episode {tc.get('episode_id')}" if tc.get("episode_id") else f"lead t{tc.get('iteration')}"
        try:
            result = json.loads(tc.get("result") or "")
        except (json.JSONDecodeError, TypeError):
            result = {}
        status = result.get("status") or result.get("verdict") or "?"
        extra = ""
        if name == "hypothesis_apply_reading":
            extra = f"edits={result.get('edits')} fork={result.get('fork')}"
        elif name == "hypothesis_test_word":
            extra = f"verdict={result.get('verdict')} menu_backed={result.get('menu_backed')}"
        elif name == "branch_adjudicate":
            extra = f"ranking={result.get('ranking')}"
        lines.append(f"  {name:<26} [{where}] {status}  {extra}")
    return "\n".join(lines)


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
# Automated-runner refinement steps (null-mask bakeoff + word repair)
#
# The v2 agent artifacts inspected above carry tool_calls/timeline; the
# automated-runner artifacts (run_automated / run_frontier_suite) instead carry
# a flat `steps` list. These compact renderers surface the two homophonic
# refinement steps so a word-repair run is inspectable without opening the raw
# JSON. search_word_repair mirrors the search_null_masks summary style.
# ---------------------------------------------------------------------------

def _fmt_delta(value: Any) -> str:
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_null_mask_step(step: dict[str, Any]) -> str:
    selected = step.get("selected") if isinstance(step.get("selected"), dict) else {}
    validation = selected.get("validation_score_v2")
    validation_text = f"{float(validation):.4f}" if isinstance(validation, (int, float)) else "n/a"
    return (
        f"  search_null_masks status={step.get('status')} "
        f"masks={step.get('mask_count')} completed={step.get('completed_mask_count')} "
        f"selected_mask={step.get('selected_mask')} validation={validation_text}"
    )


def _format_word_repair_step(step: dict[str, Any]) -> list[str]:
    status = step.get("status")
    header = f"  search_word_repair [{step.get('mode')}] status={status}"
    if status == "skipped":
        return [header + f" reason={step.get('reason')}"]
    counts = step.get("counts") or {}
    gate_counts = ""
    if "verdict_accepted" in counts or "passed_composed_gate" in counts:
        gate_counts = (
            f" verdict_accepted={counts.get('verdict_accepted')} "
            f"passed_gate={counts.get('passed_composed_gate')}"
        )
    lines = [
        header,
        (
            f"    counts: proposed={counts.get('proposed')} "
            f"prescreened={counts.get('prescreened')} "
            f"adjudicated={counts.get('adjudicated')} "
            f"improving={counts.get('improving')}"
            f"{gate_counts} "
            f"adopted={counts.get('adopted')} rejected={counts.get('rejected')}"
        ),
        (
            f"    validation: {step.get('validation_before')} -> "
            f"{step.get('validation_after')} (delta {_fmt_delta(step.get('validation_delta'))})"
        ),
    ]
    def _signals(entry: dict[str, Any]) -> str:
        signals = ""
        acceptance = entry.get("acceptance")
        if isinstance(acceptance, dict):
            signals += f" verdict={acceptance.get('decision')}"
        if entry.get("adjudication_score") is not None:
            signals += f" adjudication={_fmt_delta(entry.get('adjudication_score'))}"
        return signals

    adopted = step.get("adopted")
    if isinstance(adopted, dict):
        edits = ", ".join(str(edit) for edit in (adopted.get("edits") or []))
        lines.append(
            f"    adopted edits: {edits or '(none)'} [{adopted.get('solver')}]{_signals(adopted)}"
        )
    else:
        lines.append(f"    adopted: none ({step.get('adopted_reason')})")
        would = step.get("would_adopt")
        if isinstance(would, dict):
            edits = ", ".join(str(edit) for edit in (would.get("edits") or []))
            lines.append(f"    would adopt: {edits or '(none)'}{_signals(would)}")
    return lines


def format_automated_steps(artifact: dict[str, Any]) -> str:
    steps = artifact.get("steps")
    if not isinstance(steps, list):
        return ""
    lines: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        name = step.get("name")
        if name == "search_null_masks":
            lines.append(_format_null_mask_step(step))
        elif name == "search_word_repair":
            lines.extend(_format_word_repair_step(step))
    if not lines:
        return ""
    return "Automated refinement steps:\n" + "\n".join(lines)


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
    facts = derive_run_facts(artifact)  # M5.3 Slice 7: header/summary parity

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
        "provider": facts["provider"],
        "test_id": artifact.get("test_id"),
        "status": artifact.get("status"),
        "cipher_system": artifact.get("cipher_system"),
        "language": artifact.get("language"),
        "loop_version": facts["loop_version"],
        "char_accuracy": artifact.get("char_accuracy"),
        "word_accuracy": artifact.get("word_accuracy"),
        "score_meaning": "char_accuracy and word_accuracy are post-hoc comparisons to known benchmark plaintext when ground truth exists; they are not intrinsic solver confidence.",
        "iterations_used": facts["iterations"],
        "declared": facts["declared"],
        "auto_declared": facts["fallback"],
        "attestation_status": facts["attestation_status"],
        "solution": _trim_obj(artifact.get("solution") or artifact.get("declared_solution") or {}, 1600),
        "best_branch": facts["final_branch"],
        "final_branch": facts["final_branch"],
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
# --narrative: post-hoc transcript replay (CLI-2 spec Part 3)
#
# Convert ANY stored artifact into the same human-friendly transcript the live
# narrate renderer produces, by REPLAY: build a NarrateAgentRenderer and feed it
# the artifact's stored events. No second formatter, no LLM.
# ---------------------------------------------------------------------------

def _infer_source(cipher_id: str) -> str | None:
    """Best-effort benchmark source from the test id (for the narrate header)."""
    cid = (cipher_id or "").lower()
    for src in ("copiale", "borg"):
        if src in cid:
            return src
    if cid.startswith("synth") or "_synth" in cid or cid.startswith("en_ss"):
        return "synth"
    return None


def _declared_branch(artifact: dict) -> str:
    """Name of the branch whose decode is the run's final product."""
    sol = artifact.get("solution") or artifact.get("declared_solution")
    if isinstance(sol, dict) and sol.get("branch"):
        return str(sol["branch"])
    if artifact.get("best_branch"):
        return str(artifact["best_branch"])
    branches = artifact.get("branches") or []
    if branches and isinstance(branches[0], dict):
        return str(branches[0].get("name") or "")
    return ""


def _branch_decryption(artifact: dict, branch_name: str) -> str:
    for b in artifact.get("branches") or []:
        if isinstance(b, dict) and b.get("name") == branch_name:
            return str(b.get("decryption") or "")
    return ""


def _iterations_used(artifact: dict, events: list[dict]) -> int | None:
    if isinstance(artifact.get("iterations_used"), int):
        return artifact["iterations_used"]
    iters = [
        e.get("payload", {}).get("iteration")
        for e in events
        if isinstance(e, dict) and e.get("event") == "iteration_start"
    ]
    iters = [i for i in iters if isinstance(i, int)]
    if iters:
        return max(iters)
    tc_iters = [
        tc.get("iteration") for tc in (artifact.get("tool_calls") or [])
        if isinstance(tc, dict) and isinstance(tc.get("iteration"), int)
    ]
    return max(tc_iters) if tc_iters else None


def _result_summary_dict(result: Any) -> dict:
    """A dict result_summary for a tool_call synthesized from a stored ToolCall."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return _parse_result(result)  # {} when the payload is not a JSON object
    return {}


def _events_from_tool_calls(artifact: dict) -> list[dict]:
    """Synthesize a minimal (event, payload) stream from stored ``tool_calls``.

    v2 artifacts (and pre-947f8c6 runs) never captured ``loop_events`` but do
    carry the full tool_calls list; replaying those still yields the numbered
    tool lines + glosses. iteration_start events are inserted when the iteration
    counter advances so per-iteration grouping is preserved.
    """
    events: list[dict] = []
    last_iter: int | None = None
    for tc in artifact.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        it = tc.get("iteration")
        if isinstance(it, int) and it != last_iter:
            events.append({"event": "iteration_start", "payload": {"iteration": it}})
            last_iter = it
        name = str(tc.get("tool_name") or "tool")
        args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
        events.append({"event": "tool_start",
                       "payload": {"tool": name, "arguments": args}})
        events.append({"event": "tool_call",
                       "payload": {"tool": name,
                                   "result_summary": _result_summary_dict(tc.get("result"))}})
    return events


def _synth_finish_result(path: Path, artifact: dict, events: list[dict]) -> SimpleNamespace:
    """Build the finish() result object from stored artifact fields."""
    char = artifact.get("char_accuracy")
    word = artifact.get("word_accuracy")
    ground_truth = artifact.get("ground_truth")
    # has_ground_truth: False when the artifact carries no accuracy signal, or a
    # 0.0/absent score with no benchmark ground truth (avoids a misleading 0.0%).
    has_ground_truth = char is not None and not (
        float(char or 0.0) == 0.0 and not ground_truth and not word
    )
    branch = _declared_branch(artifact)
    started = float(artifact.get("started_at") or 0.0)
    finished = float(artifact.get("finished_at") or 0.0)
    elapsed = max(0.0, finished - started)
    return SimpleNamespace(
        status=str(artifact.get("status") or ""),
        char_accuracy=float(char) if char is not None else 0.0,
        word_accuracy=float(word) if word is not None else 0.0,
        has_ground_truth=has_ground_truth,
        iterations_used=_iterations_used(artifact, events),
        estimated_cost_usd=float(artifact.get("estimated_cost_usd") or 0.0),
        elapsed_seconds=elapsed,
        artifact_path=str(path),
        error_message=str(artifact.get("error_message") or ""),
        final_summary=str(artifact.get("final_summary") or ""),
        final_decryption=_branch_decryption(artifact, branch),
    )


def render_narrative(
    path: Path,
    artifact: dict,
    *,
    verbose: bool = False,
    stream: Any = None,
) -> None:
    """Replay a stored artifact through the live NarrateAgentRenderer."""
    stream = stream or sys.stdout
    renderer = NarrateAgentRenderer(stream, verbose=verbose)

    cipher_id = str(artifact.get("cipher_id") or artifact.get("test_id") or "?")
    loop_version = str(artifact.get("loop_version") or "v2")
    desc_bits = []
    for value, unit in (
        (artifact.get("cipher_alphabet_size"), "symbols"),
        (artifact.get("cipher_token_count"), "tokens"),
        (artifact.get("cipher_word_count"), "words"),
    ):
        if value:
            desc_bits.append(f"{value} {unit}")
    renderer.start_test(
        cipher_id,
        " · ".join(desc_bits),
        model=str(artifact.get("model") or "?"),
        max_iterations=int(artifact.get("max_iterations") or 0),
        language=artifact.get("language"),
        source=_infer_source(cipher_id),
        agent_loop=loop_version,
    )

    events = [e for e in (artifact.get("loop_events") or []) if isinstance(e, dict)]
    if events:
        for e in events:
            renderer.event(str(e.get("event") or ""), e.get("payload") or {})
    elif artifact.get("tool_calls"):
        # Graceful degradation: no captured event stream, but stored tool calls
        # let us reconstruct the numbered tool lines + glosses.
        events = _events_from_tool_calls(artifact)
        print(
            "  (reconstructed from stored tool calls; this artifact predates "
            "loop-event capture, so agent commentary and decode-progress lines "
            "are unavailable)",
            file=stream,
        )
        for e in events:
            renderer.event(str(e.get("event") or ""), e.get("payload") or {})
    else:
        print(
            "  (this artifact predates loop-event capture; showing the header "
            "and result only)",
            file=stream,
        )

    renderer.finish(_synth_finish_result(path, artifact, events))


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

    def _emit(text: str) -> None:
        if text:
            print(text)
            print()

    _emit(format_episodes(artifact))
    _emit(format_episode_budgets(artifact))
    _emit(format_suppressed_calls(artifact))
    _emit(format_readings(artifact))
    _emit(format_attestations(artifact))
    _emit(format_experiments(artifact))
    _emit(format_experiment_validation_failures(timeline))
    _emit(format_repair_cycles(artifact))
    _emit(format_saturation(artifact))
    _emit(format_repair_transactions(artifact))
    _emit(format_branch_roles(artifact))
    _emit(format_repair_hypothesis_time(artifact))
    composite_table = format_composite_calls(artifact)
    if composite_table:
        print(composite_table)
        print()
    print(format_tool_summary(timeline))
    print()
    print(format_timing_summary(analyze_tool_timing(artifact)))
    print()
    automated_steps = format_automated_steps(artifact)
    if automated_steps:
        print("─" * 70)
        print(automated_steps)
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
        "--narrative",
        action="store_true",
        help=(
            "Replay the artifact as a human-friendly narrate transcript (the "
            "same renderer the live agentic CLI uses). LLM-free; graceful on "
            "old artifacts. Use --verbose for full agent text and tool args."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="With --narrative: show full agent commentary, tool args, and decode text.",
    )
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
        if args.narrative:
            render_narrative(path, load(path), verbose=args.verbose)
            continue
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
