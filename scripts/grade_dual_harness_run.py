#!/usr/bin/env python3
"""Grade one arm of the v3-vs-MCP dual-harness comparison.

Produces a single JSON record per run (printed, optionally appended to a
results JSONL) covering BOTH measurement axes of the matrix
(docs/evidence/v3_vs_mcp_matrix.md):

  outcome quality  — char/word accuracy vs a sealed answer or recorded GT
  process quality  — terminal kind, attestation scalars, verifier spend,
                     turns/iterations, cost, escape-hatch annotation

SAFETY: prints scalars only. No plaintext, decoded text, gloss, or key
material is ever written to stdout (Borg/historical pages misfire a safety
gate; sealed synthetics must not leak into transcripts).

Arms:
  --investigation <id>   MCP arm: grade a registry investigation
                         (~/.config/decipher/investigations/<id>/), every
                         branch key applied and scored; terminal from meta.
  --artifact <path>      v3/v2 arm: grade a RunArtifact JSON. Uses recorded
                         benchmark accuracy when present, else grades branch
                         decryptions against the sealed answer.

Answers:
  --answer <path>        sealed answer JSON: either a single-case file
                         (round6 style) or a pack file with {"cases": {...}}
  --case <case_id>       case key when the answer file is a pack

Escape hatch (round-6 pattern — solution produced outside the surface):
  --claimed-text <path>  file holding the claimed plaintext to grade as an
                         additional "claimed" score.

Bookkeeping:
  --append <jsonl>       append the record to a results JSONL
  --arm-label <str>      e.g. v3-gpt5.5 / codex-mcp / claude-code-mcp
  --escape-hatch         mark that the solution left the tool surface
  --note <str>           free-form annotation
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def _letters(text: str) -> str:
    return "".join(ch for ch in (text or "").upper() if ch.isalpha())


def _truth_from_answer(answer: dict) -> tuple[str, str | None]:
    """Return (letter_stream, spaced_text_or_None)."""
    spaced = answer.get("plaintext_with_spaces")
    letters = answer.get("plaintext_letters") or _letters(spaced or "")
    if not letters:
        raise SystemExit("answer file has no plaintext_letters/plaintext_with_spaces")
    return letters, spaced


def _char_acc(candidate_letters: str, truth_letters: str) -> float | None:
    if not candidate_letters or len(candidate_letters) != len(truth_letters):
        return None
    return sum(a == b for a, b in zip(candidate_letters, truth_letters)) / len(truth_letters)


def _score_both(decoded_spaced: str, truth_spaced: str, case_id: str) -> tuple[float, float]:
    """Edit-aware char+word accuracy via the benchmark scorer (handles
    bracketed editorial plaintext, boundary drift, resync)."""
    from benchmark.scorer import score_decryption
    res = score_decryption(case_id, decoded_spaced, truth_spaced, 0.0, "scored")
    return res.char_accuracy, res.word_accuracy


def grade_investigation(inv_id: str, answer: dict | None, case_id: str) -> dict:
    root = Path.home() / ".config/decipher/investigations" / inv_id
    state = json.loads((root / "investigation.json").read_text())["state"]
    meta = json.loads((root / "meta.json").read_text())

    cipher = state["cipher"]
    tokens = cipher["tokens"]
    plain_syms = cipher["plaintext_symbols"]
    word_lengths = cipher.get("word_lengths") or []

    branches = []
    best = {"branch": None, "char_accuracy": None}
    for br in state.get("workspace", {}).get("branches", []):
        key = {int(k): v for k, v in (br.get("key") or {}).items()}
        row = {"branch": br.get("name"), "n_key": len(key)}
        # Transform/Quagmire installs carry the decode in metadata, not a
        # token-key dict — grade metadata decoded_text when no key exists.
        meta_decode = (br.get("metadata") or {}).get("decoded_text")
        if not key and meta_decode and answer is not None:
            truth_letters, truth_spaced = _truth_from_answer(answer)
            if truth_spaced:
                acc, wa = _score_both(meta_decode, truth_spaced, case_id)
            else:
                acc = _char_acc(_letters(meta_decode), truth_letters)
                wa = None
            row["char_accuracy"] = round(acc, 4) if acc is not None else None
            row["word_accuracy"] = round(wa, 4) if wa is not None else None
            row["source"] = "metadata_decoded_text"
            if acc is not None and (best["char_accuracy"] is None or acc > best["char_accuracy"]):
                best = {"branch": br.get("name"), "char_accuracy": round(acc, 4),
                        "word_accuracy": row.get("word_accuracy"),
                        "source": "metadata_decoded_text"}
            branches.append(row)
            continue
        if key and answer is not None:
            truth_letters, truth_spaced = _truth_from_answer(answer)
            order = br.get("token_order") or list(range(len(tokens)))
            stream = "".join(
                (plain_syms[key[tokens[i]]] if tokens[i] in key else "?")
                for i in order
            )
            acc = wa = None
            if truth_spaced:
                # edit-aware scoring (handles bracketed/editorial truth)
                if word_lengths:
                    words, i = [], 0
                    for wl in word_lengths:
                        words.append(stream[i:i + wl])
                        i += wl
                    decoded_spaced = " | ".join(" ".join(w) for w in words)
                else:
                    decoded_spaced = " ".join(stream)
                acc, wa = _score_both(decoded_spaced.replace("?", "x"),
                                      truth_spaced, case_id)
            else:
                acc = _char_acc(_letters(stream.replace("?", "x")), truth_letters)
                # '?' → 'x' keeps length; unmapped positions count as wrong
            row["char_accuracy"] = round(acc, 4) if acc is not None else None
            row["word_accuracy"] = round(wa, 4) if wa is not None else None
            if acc is not None and (best["char_accuracy"] is None or acc > best["char_accuracy"]):
                best = {"branch": br.get("name"), "char_accuracy": round(acc, 4),
                        "word_accuracy": row.get("word_accuracy")}
        branches.append(row)

    attests = [
        {k: a.get(k) for k in (
            "reader_accepts_as_solution", "target_language_confidence",
            "semantic_recoverability", "damage_scope", "coherence", "branch")}
        for a in state.get("verify_attestations", [])
    ]
    spend = state.get("budget_ledger") or []
    terminal = meta.get("terminal") or {}
    return {
        "arm_kind": "mcp_investigation",
        "id": inv_id,
        "client_name": meta.get("client_name"),
        "turns": state.get("turn"),
        "terminal_kind": terminal.get("kind"),
        "declared_turn": terminal.get("declared_turn"),
        "status": meta.get("status"),
        "n_branches": len(branches),
        "branches": branches,
        "best": best,
        "n_attestations": len(attests),
        "attestations": attests,
        "server_side_spend_usd": round(sum(s.get("cost_usd") or 0 for s in spend), 4),
        "n_experiments": len(state.get("experiment_queue") or []),
        "n_repair_transactions": len(state.get("repair_transactions") or []),
    }


def grade_artifact(path: Path, answer: dict | None, case_id: str) -> dict:
    art = json.loads(path.read_text())
    rec: dict = {
        "arm_kind": "run_artifact",
        "id": art.get("run_id"),
        "path": str(path),
        "loop_version": art.get("loop_version"),
        "model": art.get("model"),
        "status": art.get("status"),
        "estimated_cost_usd": art.get("estimated_cost_usd"),
        "iterations_cap": art.get("max_iterations"),
        "n_tool_calls": len(art.get("tool_calls") or []),
        "n_episodes": len(art.get("episodes") or []),
        "n_experiments": len(art.get("experiments") or []),
        "auto_declared": art.get("auto_declared"),
        "safety_gate_fired": art.get("safety_gate_fired"),
    }
    attests = [
        {k: (a.get(k) if isinstance(a, dict) else None) for k in (
            "reader_accepts_as_solution", "target_language_confidence",
            "semantic_recoverability", "damage_scope", "coherence", "branch")}
        for a in (art.get("attestations") or [])
    ]
    rec["n_attestations"] = len(attests)
    rec["attestations"] = attests

    if art.get("char_accuracy") is not None:
        rec["best"] = {
            "source": "recorded_benchmark_accuracy",
            "char_accuracy": art.get("char_accuracy"),
            "word_accuracy": art.get("word_accuracy"),
        }
        return rec

    if answer is None:
        rec["best"] = {"source": "none", "char_accuracy": None}
        return rec

    truth_letters, truth_spaced = _truth_from_answer(answer)
    best = {"source": "graded_branch_decryptions", "branch": None, "char_accuracy": None}
    for br in art.get("branches") or []:
        dec = br.get("decryption") or ""
        if not dec:
            continue
        if truth_spaced:
            acc, wa = _score_both(dec.replace("?", "x"), truth_spaced, case_id)
        else:
            acc = _char_acc(_letters(dec.replace("?", "x")), truth_letters)
            wa = None
        if acc is not None and (best["char_accuracy"] is None or acc > best["char_accuracy"]):
            best = {"source": "graded_branch_decryptions", "branch": br.get("name"),
                    "char_accuracy": round(acc, 4),
                    "word_accuracy": round(wa, 4) if wa is not None else None}
    rec["best"] = best
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    arm = ap.add_mutually_exclusive_group(required=True)
    arm.add_argument("--investigation")
    arm.add_argument("--artifact", type=Path)
    ap.add_argument("--answer", type=Path)
    ap.add_argument("--case", default=None)
    ap.add_argument("--claimed-text", type=Path)
    ap.add_argument("--append", type=Path)
    ap.add_argument("--arm-label", default=None)
    ap.add_argument("--escape-hatch", action="store_true")
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    answer = None
    case_id = args.case or "case"
    if args.answer:
        data = json.loads(args.answer.read_text())
        if "cases" in data:
            if not args.case or args.case not in data["cases"]:
                raise SystemExit(
                    f"--case required; available: {sorted(data['cases'])}")
            answer = data["cases"][args.case]
        else:
            answer = data
            case_id = args.case or data.get("round") and f"round{data['round']}" or "case"

    if args.investigation:
        rec = grade_investigation(args.investigation, answer, case_id)
    else:
        rec = grade_artifact(args.artifact, answer, case_id)

    if args.claimed_text and answer is not None:
        truth_letters, truth_spaced = _truth_from_answer(answer)
        claimed = args.claimed_text.read_text()
        acc = _char_acc(_letters(claimed), truth_letters)
        rec["claimed"] = {
            "char_accuracy": round(acc, 4) if acc is not None else None,
            "length_match": len(_letters(claimed)) == len(truth_letters),
        }

    rec["case_id"] = args.case or case_id
    rec["arm_label"] = args.arm_label
    rec["escape_hatch"] = bool(args.escape_hatch)
    rec["note"] = args.note
    rec["graded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(json.dumps(rec, indent=2))
    if args.append:
        with open(args.append, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
