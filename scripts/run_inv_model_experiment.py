#!/usr/bin/env python3
"""Thin cipher-family DIAGNOSIS runner for the INV model experiment (v1 task).

Per the spec (F1), this does NOT reuse ``run_v2``: ``run_v2`` injects the
automated ``compute_cipher_fingerprint`` suspicion ranking into the opening
prompt (which would hand the baseline's answer to the model) and mode-filters
the tool list. Instead this is a dedicated thin loop:

  * a ``WorkspaceToolExecutor`` restricted to an OBSERVE-ONLY tool subset,
  * its own short initial context (NO fingerprint injection, NO mode filter),
  * a new ``meta_declare_diagnosis`` tool (family enum + recommended_attack +
    confidence) handled by the runner itself as the parse target.

Served-model + safety-gate detection is captured for ALL THREE models at the
send site (F7); the artifact records the requested model, served model(s), and
``safety_gate_fired``. Gate-fired Anthropic runs are excluded from the
comparison by the scorer (reported as a contaminated bucket).

Only the task adapter is v1-specific; the executor, firewall, artifacts, models,
gate detection, metrics, and report are shared with the eventual INV-0 v2 task.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/run_inv_model_experiment.py \
        --models gpt-5.5 gpt-5.6-sol claude-fable-5 \
        --cases inv_diag_case_01 inv_diag_case_06 \
        --max-iterations 6 --artifact-root artifacts/inv_model_experiment
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent.model_provider import (  # noqa: E402
    estimate_provider_cost,
    served_model_from_response,
    served_model_matches,
)
from agent.tools_v2 import TOOL_DEFINITIONS, WorkspaceToolExecutor  # noqa: E402
from analysis import dictionary, pattern  # noqa: E402
from benchmark.loader import parse_canonical_transcription  # noqa: E402
from experiments.inv_diagnosis import (  # noqa: E402
    META_DECLARE_DIAGNOSIS_TOOL,
    OBSERVE_ONLY_TOOLS,
    SUITE_ROWS,
    assert_no_ground_truth_leak,
    build_case,
)
from testgen.cache import PlaintextCache  # noqa: E402
from workspace import Workspace  # noqa: E402
from agent.loop_v2 import _collect_assistant_blocks  # noqa: E402

DEFAULT_SUITE = REPO_ROOT / "frontier" / "inv_diagnosis_suite.jsonl"
DEFAULT_CACHE_DIR = REPO_ROOT / "testgen_cache"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "inv_model_experiment"

DIAGNOSIS_SYSTEM_PROMPT = (
    "You are a cryptanalyst performing CIPHER-FAMILY DIAGNOSIS on an unknown "
    "historical cipher. Your ONLY task is to identify the single most likely "
    "cipher family and recommend the next concrete attack. Do NOT attempt a "
    "full decryption and do NOT try to recover the key. You have a small set of "
    "observe-only analysis tools; use them to gather statistical evidence "
    "(shape, frequency, index of coincidence, periodicity). When you have "
    "enough evidence, call `meta_declare_diagnosis` EXACTLY ONCE with the "
    "family (chosen from the enum), a recommended_attack, and your confidence, "
    "then stop. If two mechanisms appear combined, use "
    "`transposition_substitution`; if no single standard family fits, use "
    "`unknown`. Do not over-commit to a standard family on weak evidence."
)

TRUNCATION_NUDGE = (
    "Your previous response was cut off before you finished (it hit the output "
    "token limit). Do not continue that reasoning. Call meta_declare_diagnosis "
    "NOW with your best current diagnosis, using the family enum, a "
    "recommended_attack, and your confidence."
)


def _observe_tool_schemas() -> list[dict[str, Any]]:
    """The observe-only tool schemas (from TOOL_DEFINITIONS) + the declare tool."""
    observe = [d for d in TOOL_DEFINITIONS if d["name"] in OBSERVE_ONLY_TOOLS]
    missing = OBSERVE_ONLY_TOOLS - {d["name"] for d in observe}
    if missing:
        raise RuntimeError(f"observe tools missing from TOOL_DEFINITIONS: {sorted(missing)}")
    return [*observe, META_DECLARE_DIAGNOSIS_TOOL]


def render_diagnosis_prompt(cipher_text: Any, ciphertext_display: str) -> str:
    """Build the short, fingerprint-free user prompt for a cipher.

    Neutral shape facts only (token/symbol counts, boundary presence). No
    suspicion ranking, no family flags — the firewall asserts this.
    """
    token_count = len(cipher_text.tokens)
    unique_symbols = len(set(cipher_text.tokens))
    has_boundaries = len(cipher_text.words) > 0
    boundary_note = (
        "word groups are marked (| separators)"
        if has_boundaries
        else "there are no word boundaries"
    )
    # Deliberately NO family menu here: the allowed families live in the
    # meta_declare_diagnosis tool enum (a case-invariant menu seen for every
    # case). Naming them in this case-specific task prompt would surface the true
    # family as one of the listed options and trip the firewall for no reason.
    return (
        f"Unknown cipher. It has {token_count} tokens over {unique_symbols} "
        f"distinct symbols, and {boundary_note}.\n\n"
        "Ciphertext:\n"
        f"{ciphertext_display}\n\n"
        "Gather evidence with the observe_* tools, then declare your diagnosis "
        "with meta_declare_diagnosis (choose the family from its enum)."
    )


def _build_executor(cipher_text: Any, language: str) -> WorkspaceToolExecutor:
    dict_path = dictionary.get_dictionary_path(language)
    word_set = dictionary.load_word_set(dict_path) if dict_path else set()
    word_list = pattern.load_word_list(dict_path) if dict_path else []
    pattern_dict = pattern.build_pattern_dictionary(word_list)
    workspace = Workspace(cipher_text=cipher_text)
    executor = WorkspaceToolExecutor(
        workspace=workspace,
        language=language,
        word_set=word_set,
        word_list=word_list,
        pattern_dict=pattern_dict,
    )
    # Firewall the executor to the observe-only subset. meta_declare_diagnosis is
    # handled by the runner and never routed here; any other tool is rejected.
    executor.set_allowed_tool_names(set(OBSERVE_ONLY_TOOLS))
    return executor


def _model_is_priced(provider: str, model: str) -> bool:
    """Whether our pricing table can estimate cost for (provider, model) (F6)."""
    return estimate_provider_cost(provider, model, 1_000_000, 0, 0) > 0.0


def run_diagnosis_task(
    *,
    test_data: Any,
    model_provider: Any,
    requested_model: str,
    provider_name: str,
    language: str = "en",
    max_iterations: int = 6,
    max_tokens: int = 8192,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the v1 diagnosis task for one (model, case). Returns an artifact dict.

    ``model_provider`` is any object with ``.send(...)`` returning a
    ``ModelResponse`` (a real provider, or the fake used by the runner test).
    """
    cipher_text = parse_canonical_transcription(test_data.canonical_transcription)
    prompt = render_diagnosis_prompt(cipher_text, test_data.canonical_transcription)

    # Firewall (F4): the case-specific rendered TASK prompt must not leak ground
    # truth. Scope is the user prompt — the fixed DIAGNOSIS_SYSTEM_PROMPT is
    # case-invariant boilerplate (its generic "use transposition_substitution /
    # unknown" guidance is identical for every case and identifies no case's
    # answer). The runner does not hold true_family (the scorer does), so the
    # family check is exercised by the dedicated firewall test which passes it.
    assert_no_ground_truth_leak(
        prompt,
        true_family="",  # not available to the runner; scorer holds it
        plaintext=test_data.plaintext,
        builder_description=test_data.test.description,
    )

    executor = _build_executor(cipher_text, language)
    tools = _observe_tool_schemas()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    diagnosis: dict[str, Any] | None = None
    declared_iteration: int | None = None
    tool_call_log: list[dict[str, Any]] = []
    served_models: list[str] = []
    stop_reasons: list[str] = []
    safety_gate_fired = False
    truncation_nudged = False
    status = "exhausted"
    error_message = ""
    total_in = total_out = total_cache = 0
    started = time.time()

    for iteration in range(1, max_iterations + 1):
        executor.set_iteration(iteration)
        try:
            response = model_provider.send(
                messages=messages,
                tools=tools,
                system=DIAGNOSIS_SYSTEM_PROMPT,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error_message = f"API error on iteration {iteration}: {exc}"
            if verbose:
                print(f"    {error_message}")
            break

        # Served-model / safety-gate capture (F7) — the send site.
        served = served_model_from_response(response)
        if served is not None:
            if served not in served_models:
                served_models.append(served)
            if not served_model_matches(requested_model, served):
                safety_gate_fired = True

        usage = response.usage
        total_in += usage.input_tokens
        total_out += usage.output_tokens
        total_cache += usage.cache_read_input_tokens

        # Capture the provider stop_reason (e.g. Anthropic 'refusal') when
        # exposed — it distinguishes a safety refusal from a model that simply
        # produced no tool call, which matters for the Fable arm (F9).
        stop_reason = getattr(response.raw, "stop_reason", None)
        if stop_reason and stop_reason not in stop_reasons:
            stop_reasons.append(str(stop_reason))

        assistant_blocks, tool_uses, text_parts = _collect_assistant_blocks(response)
        messages.append({"role": "assistant", "content": assistant_blocks})

        if not tool_uses:
            # F-1: a max_tokens truncation with no complete tool_use this turn is
            # NOT a refusal — the model was cut off mid-reasoning (biases against
            # verbose/thinking-heavy models). Give it one more turn with an
            # explicit nudge to declare now, converting the truncation into a
            # scored declaration instead of a no_tool_calls miss. One-shot: if it
            # truncates again we stop.
            if (
                stop_reason == "max_tokens"
                and not truncation_nudged
                and iteration < max_iterations
            ):
                truncation_nudged = True
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": TRUNCATION_NUDGE}],
                })
                if verbose:
                    print(f"    iter {iteration}: truncated (max_tokens); nudging to declare.")
                continue
            status = "refusal" if stop_reason == "refusal" else "no_tool_calls"
            if verbose:
                print(f"    iter {iteration}: no tool calls (stop_reason={stop_reason}); stopping.")
            break

        tool_results: list[dict[str, Any]] = []
        for call in tool_uses:
            name = str(call.get("name"))
            args = call.get("input") or {}
            tool_use_id = str(call.get("id") or "")
            tool_call_log.append({"iteration": iteration, "tool": name})
            if name == "meta_declare_diagnosis":
                diagnosis = {
                    "family": args.get("family"),
                    "recommended_attack": args.get("recommended_attack", ""),
                    "confidence": args.get("confidence"),
                    "rationale": args.get("rationale", ""),
                }
                declared_iteration = iteration
                status = "declared"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps({"ok": True, "recorded": diagnosis}),
                })
            else:
                result = executor.execute(name, args, tool_use_id)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})
        if verbose:
            called = ", ".join(str(c.get("name")) for c in tool_uses)
            print(f"    iter {iteration}: {called}")
        if diagnosis is not None:
            break

    elapsed = time.time() - started
    priced = _model_is_priced(provider_name, requested_model)
    cost = estimate_provider_cost(
        provider_name, requested_model, total_in, total_out, total_cache
    )

    return {
        "run_id": uuid.uuid4().hex[:12],
        "experiment": "inv_model_diagnosis",
        "task_version": "v1",
        "case_id": test_data.test.test_id,
        "cipher_system": test_data.test.cipher_system,
        "language": language,
        # ``model`` is the REQUESTED model (RunArtifact convention, F7).
        "model": requested_model,
        "provider": provider_name,
        "served_models": served_models,
        "safety_gate_fired": safety_gate_fired,
        "max_iterations": max_iterations,
        "max_tokens": max_tokens,
        "automated_preflight": None,  # preflight is OFF for the diagnosis task (F4)
        "status": status,
        "error_message": error_message,
        "stop_reasons": stop_reasons,
        "truncation_nudged": truncation_nudged,
        "diagnosis": diagnosis,
        "declared_iteration": declared_iteration,
        "tool_calls": tool_call_log,
        "tool_call_count": len(tool_call_log),
        "cipher_token_count": len(cipher_text.tokens),
        "cipher_unique_symbols": len(set(cipher_text.tokens)),
        "final_char_accuracy": None,  # diagnosis-only task never attempts a solve
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cache_read_tokens": total_cache,
        "estimated_cost_usd": cost,
        "cost_priced": priced,
        "elapsed_seconds": elapsed,
    }


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def _select_rows(case_ids: list[str]) -> list[Any]:
    by_id = {r.case_id: r for r in SUITE_ROWS}
    if not case_ids:
        return list(SUITE_ROWS)
    rows = []
    for cid in case_ids:
        if cid not in by_id:
            raise SystemExit(f"Unknown case_id {cid!r}. Known: {sorted(by_id)}")
        rows.append(by_id[cid])
    return rows


def _build_real_provider(model: str) -> tuple[Any, str]:
    from agent.model_provider import infer_provider_from_model, make_model_provider
    from cli import get_api_key

    provider = infer_provider_from_model(model, None)
    api_key = get_api_key(provider)
    return make_model_provider(provider=provider, api_key=api_key, model=model), provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[],
        help="case_ids to run (default: all suite rows).",
    )
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max output tokens per model turn (default 8192, matching loop_v2 "
        "for these providers). A too-small cap truncates verbose/thinking-heavy "
        "models before they declare (F-1).",
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--language", default="en")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--continue-on-model-error",
        action="store_true",
        help="If a model arm fails to initialize (e.g. no credits), skip it "
        "(graceful degradation, F9) instead of aborting.",
    )
    args = parser.parse_args()

    rows = _select_rows(args.cases)
    cache = PlaintextCache(args.cache_dir)
    artifact_root = Path(args.artifact_root)

    for model in args.models:
        try:
            provider_obj, provider_name = _build_real_provider(model)
        except SystemExit as exc:
            if args.continue_on_model_error:
                print(f"[skip] {model}: no provider/key ({exc}); continuing.", file=sys.stderr)
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            if args.continue_on_model_error:
                print(f"[skip] {model}: init failed ({exc}); continuing.", file=sys.stderr)
                continue
            raise

        model_dir = artifact_root / _sanitize(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            test_data = build_case(row, cache, offline=True)
            print(f"[{model}] {row.case_id} ...", flush=True)
            try:
                artifact = run_diagnosis_task(
                    test_data=test_data,
                    model_provider=provider_obj,
                    requested_model=model,
                    provider_name=provider_name,
                    language=args.language,
                    max_iterations=args.max_iterations,
                    max_tokens=args.max_tokens,
                    verbose=args.verbose,
                )
            except Exception as exc:  # noqa: BLE001
                if args.continue_on_model_error:
                    print(f"  [skip case] {row.case_id}: {exc}", file=sys.stderr)
                    continue
                raise
            out_path = model_dir / f"{row.case_id}.json"
            out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
            d = artifact.get("diagnosis") or {}
            cost = artifact.get("estimated_cost_usd")
            cost_s = f"${cost:.4f}" if artifact.get("cost_priced") else "n/a"
            print(
                f"  -> status={artifact['status']} family={d.get('family')} "
                f"conf={d.get('confidence')} served={artifact['served_models']} "
                f"gate={artifact['safety_gate_fired']} cost={cost_s} "
                f"tokens={artifact['total_input_tokens']}in/{artifact['total_output_tokens']}out"
            )
    print(f"\nArtifacts under {artifact_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
