"""v2 benchmark runner: uses the workspace loop and saves full artifacts."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.display import make_agent_renderer
from agent.final_summary import build_final_summary
from agent.loop_v2 import run_v2
from benchmark.loader import TestData, parse_canonical_transcription, resolve_test_language
from benchmark.scorer import (
    format_alignment,
    format_char_diff,
    has_word_boundaries,
    score_branch_decryptions,
    score_decryption,
)
from preprocessing import convert_s_tokens_to_letters, estimate_normalization_benefit


@dataclass
class RunResultV2:
    """Summary of one v2 run. Full details live in the RunArtifact JSON."""
    test_id: str
    status: str
    final_decryption: str
    self_confidence: float | None
    iterations_used: int
    elapsed_seconds: float
    char_accuracy: float = 0.0
    word_accuracy: float = 0.0
    artifact_path: str = ""
    error_message: str = ""
    tool_requests: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    final_branch: str = ""
    branch_scores: list[dict[str, Any]] = field(default_factory=list)
    alignment_report: str = ""
    final_summary: str = ""


class BenchmarkRunnerV2:
    """Runs v2 agent on benchmark tests and writes per-run artifacts."""

    def __init__(
        self,
        claude_api: Any,
        max_iterations: int = 50,
        verbose: bool = False,
        language: str | None = None,
        artifact_dir: str | Path = "artifacts",
        automated_preflight: bool = True,
        display_mode: str = "off",
        external_context: str | None = None,
    ) -> None:
        self.api = claude_api
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.default_language = language
        self.artifact_dir = Path(artifact_dir)
        self.automated_preflight = automated_preflight
        self.display_mode = display_mode
        self.external_context = external_context

    def _resolve_language(self, test_data: TestData) -> str:
        return resolve_test_language(test_data, self.default_language)

    def run_test(
        self,
        test_data: TestData,
        language: str | None = None,
        prior_context: str | None = None,
    ) -> RunResultV2:
        test_id = test_data.test.test_id
        lang = language or self._resolve_language(test_data)
        start = time.time()
        renderer = make_agent_renderer(self.display_mode)
        if renderer is not None:
            renderer.start_test(
                test_id,
                test_data.test.description,
                model=self.api.model,
                max_iterations=self.max_iterations,
            )

        try:
            # Normalize S-tokens to letters if beneficial for API compatibility
            raw_transcription = test_data.canonical_transcription
            policy_risk = estimate_normalization_benefit(raw_transcription)

            if policy_risk == "high":
                if self.verbose:
                    print(f"  [{test_id}] Applying S-token normalization for API compatibility")
                converted_transcription, token_mapping = convert_s_tokens_to_letters(raw_transcription)
                cipher_text = parse_canonical_transcription(converted_transcription)
                cipher_text._original_s_token_mapping = token_mapping  # Store for later
            else:
                cipher_text = parse_canonical_transcription(raw_transcription)
                cipher_text._original_s_token_mapping = None

        except Exception as e:
            return RunResultV2(
                test_id=test_id,
                status="error",
                final_decryption="",
                self_confidence=None,
                iterations_used=0,
                elapsed_seconds=time.time() - start,
                error_message=f"Failed to parse transcription: {e}",
            )

        if self.verbose:
            print(
                f"  [{test_id}] cipher parsed: "
                f"{cipher_text.alphabet.size} symbols, "
                f"{len(cipher_text.tokens)} tokens, "
                f"{len(cipher_text.words)} words, lang={lang}"
            )

        automated_preflight = None
        if self.automated_preflight:
            if renderer is not None:
                renderer.event("preflight_start", {})
            elif not self.verbose:
                print("  preflight(no-LLM)...", end="", flush=True)
            automated_preflight = _run_automated_preflight(
                cipher_text,
                lang,
                test_id,
                test_data.test.cipher_system,
            )
            if renderer is not None:
                renderer.event("preflight_result", {
                    "status": automated_preflight.get("status", "unknown"),
                    "solver": automated_preflight.get("solver", "unknown"),
                    "elapsed_seconds": automated_preflight.get("elapsed_seconds", 0.0),
                })
            elif self.verbose:
                solver = automated_preflight.get("solver", "unknown")
                status = automated_preflight.get("status", "unknown")
                elapsed = automated_preflight.get("elapsed_seconds", 0.0)
                print(
                    f"  [{test_id}] automated preflight: "
                    f"{status}, solver={solver}, {elapsed:.1f}s, "
                    "$0.00 (no LLM access)"
                )
            else:
                status = automated_preflight.get("status", "unknown")
                elapsed = automated_preflight.get("elapsed_seconds", 0.0)
                print(f" [{status}, {elapsed:.0f}s, $0.00 no LLM]", flush=True)

        def on_event(event: str, payload: dict) -> None:
            if renderer is not None:
                renderer.event(event, payload)
            elif not self.verbose:
                if event == "iteration_start":
                    print(f"  iter {payload['iteration']}...", end="", flush=True)
                elif event == "tool_call":
                    print(".", end="", flush=True)
                elif event in {"declared_solution", "run_complete",
                               "error", "max_iterations_reached"}:
                    print(f" [{event}]", flush=True)

        # Build the prior_context block: caller-supplied > external > benchmark
        benchmark_ctx = _format_benchmark_context(test_data)
        if self.external_context:
            auto_ctx: str | None = (
                f"{self.external_context}\n\n{benchmark_ctx}"
                if benchmark_ctx
                else self.external_context
            )
        else:
            auto_ctx = benchmark_ctx

        artifact = run_v2(
            cipher_text=cipher_text,
            claude_api=self.api,
            language=lang,
            max_iterations=self.max_iterations,
            cipher_id=test_id,
            prior_context=prior_context or auto_ctx,
            automated_preflight=automated_preflight,
            verbose=self.verbose,
            on_event=on_event,
        )

        # Post-hoc scoring against ground truth — all branches
        ground_truth = test_data.plaintext
        artifact.ground_truth = ground_truth

        branch_inputs = [
            (b.name, b.decryption, b.mapped_count) for b in artifact.branches
        ]
        branch_scores = score_branch_decryptions(test_id, branch_inputs, ground_truth)

        # Attach per-branch accuracy to snapshots
        branch_acc_map = {r["branch"]: r for r in branch_scores}
        for b in artifact.branches:
            if b.name in branch_acc_map:
                b.char_accuracy = branch_acc_map[b.name]["char_accuracy"]
                b.word_accuracy = branch_acc_map[b.name]["word_accuracy"]

        # Final branch = declared branch, or best branch by char accuracy
        final_branch = artifact.solution.branch if artifact.solution else None
        if not final_branch and branch_scores:
            final_branch = branch_scores[0]["branch"]
        final_decryption = next(
            (b.decryption for b in artifact.branches if b.name == final_branch),
            artifact.branches[0].decryption if artifact.branches else "",
        )
        final_score = branch_acc_map.get(final_branch or "", {})
        artifact.char_accuracy = final_score.get("char_accuracy", 0.0)
        artifact.word_accuracy = final_score.get("word_accuracy", 0.0)
        word_boundaries = has_word_boundaries(ground_truth) if ground_truth else False
        alignment_report = ""
        if ground_truth:
            if word_boundaries:
                alignment_report = format_alignment(
                    final_decryption,
                    ground_truth,
                    max_words=50,
                )
            else:
                alignment_report = format_char_diff(final_decryption, ground_truth)

        final_summary = build_final_summary(
            artifact,
            final_branch=final_branch or "",
            final_decryption=final_decryption,
        )
        artifact.final_summary = final_summary

        if self.verbose and ground_truth:
            print(f"\n  [{test_id}] Branch scores vs ground truth:")
            for r in branch_scores:
                declared = " <-- declared" if (artifact.solution and r["branch"] == artifact.solution.branch) else ""
                word_str = f"word={r['word_accuracy']:>5.1%}" if word_boundaries else "word=  N/A"
                print(
                    f"    {r['branch']:<20} mapped={r['mapped_count']:>2}  "
                    f"char={r['char_accuracy']:>5.1%}  {word_str}{declared}"
                )
            best = branch_scores[0] if branch_scores else None
            if best:
                best_decryption = next(
                    (b.decryption for b in artifact.branches if b.name == best["branch"]), ""
                )
                if word_boundaries:
                    print(f"\n  [{test_id}] Word alignment (best branch: {best['branch']}):")
                    print(format_alignment(best_decryption, ground_truth, max_words=50))
                else:
                    print(f"\n  [{test_id}] Character diff (best branch: {best['branch']}):")
                    print(format_char_diff(best_decryption, ground_truth))
                print()

        # Add preprocessing info to artifact if S-token conversion was used
        if hasattr(cipher_text, '_original_s_token_mapping') and cipher_text._original_s_token_mapping:
            artifact.preprocessing_applied = {
                "type": "s_token_conversion",
                "reason": "s_token_normalization",
                "mapping": cipher_text._original_s_token_mapping
            }
        else:
            artifact.preprocessing_applied = None

        # Persist artifact
        artifact_path = self.artifact_dir / test_id / f"{artifact.run_id}.json"
        try:
            artifact.save(artifact_path)
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                print(f"  warning: could not save artifact: {e}")

        iterations_used = (
            max(tc.iteration for tc in artifact.tool_calls)
            if artifact.tool_calls else 0
        )
        elapsed = time.time() - start
        result = RunResultV2(
            test_id=test_id,
            status=artifact.status,
            final_decryption=final_decryption,
            self_confidence=(
                artifact.solution.self_confidence if artifact.solution else None
            ),
            iterations_used=iterations_used,
            elapsed_seconds=elapsed,
            char_accuracy=artifact.char_accuracy or 0.0,
            word_accuracy=artifact.word_accuracy or 0.0,
            artifact_path=str(artifact_path),
            error_message=artifact.error_message,
            tool_requests=list(artifact.tool_requests),
            total_tokens=artifact.total_input_tokens + artifact.total_output_tokens,
            estimated_cost_usd=artifact.estimated_cost_usd,
            final_branch=final_branch or "",
            branch_scores=branch_scores,
            alignment_report=alignment_report,
            final_summary=final_summary,
        )
        if renderer is not None:
            renderer.finish(result)
        return result


def _format_benchmark_context(test_data: TestData) -> str | None:
    if not test_data.context_canonical_transcription:
        return None
    record_ids = ", ".join(test_data.test.context_records[:20])
    if len(test_data.test.context_records) > 20:
        record_ids += f", ... ({len(test_data.test.context_records)} total)"
    return (
        "Additional same-cipher benchmark context is available. "
        "Use it as auxiliary ciphertext signal, but score/declaration should "
        "focus on the target record(s).\n\n"
        f"Context records: {record_ids}\n\n"
        "Context canonical transcription:\n"
        f"{test_data.context_canonical_transcription}"
    )


def _run_automated_preflight(
    cipher_text,
    language: str,
    test_id: str,
    cipher_system: str,
) -> dict[str, Any]:
    from automated.runner import format_automated_preflight_for_llm, run_automated

    result = run_automated(
        cipher_text=cipher_text,
        language=language,
        cipher_id=test_id,
        ground_truth=None,
        cipher_system=cipher_system,
    )
    artifact = dict(result.artifact)
    artifact["summary"] = format_automated_preflight_for_llm(result)
    artifact["enabled"] = True
    return artifact
