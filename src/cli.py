#!/usr/bin/env python3
"""CLI entry point for Decipher — run benchmarks or crack ciphers headlessly."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _use_agentic_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "agentic", False))


def _resolve_agent_display(args: argparse.Namespace) -> str:
    requested = getattr(args, "display", "auto")
    if requested != "auto":
        return requested
    if getattr(args, "verbose", False):
        return "off"
    return "pretty" if sys.stdout.isatty() else "raw"


_PROVIDER_ENV_KEYS = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}

_PROVIDER_KEYRING_ACCOUNTS = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "gemini": "gemini_api_key",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_dotenv_key(provider: str) -> str:
    names = set(_PROVIDER_ENV_KEYS.get(provider, []))
    for path in [_repo_root() / ".env", Path.cwd() / ".env"]:
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() in names:
                    return value.strip().strip("'\"")
        except OSError:
            continue
    return ""


def _read_key_file(provider: str) -> str:
    root = _repo_root()
    candidates = [
        root / ".decipher_keys" / f"{provider}_api_key",
        Path.cwd() / ".decipher_keys" / f"{provider}_api_key",
    ]
    for env_name in _PROVIDER_ENV_KEYS.get(provider, []):
        candidates.extend([
            root / ".decipher_keys" / env_name,
            Path.cwd() / ".decipher_keys" / env_name,
        ])
    for path in candidates:
        try:
            if path.exists():
                value = path.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except OSError:
            continue
    return ""


def get_api_key(provider: str = "anthropic") -> str:
    from agent.model_provider import canonical_provider

    provider = canonical_provider(provider)
    for env_name in _PROVIDER_ENV_KEYS.get(provider, []):
        key = os.environ.get(env_name)
        if key:
            return key
    key = _read_dotenv_key(provider)
    if key:
        return key
    key = _read_key_file(provider)
    if key:
        return key
    try:
        import keyring

        key = keyring.get_password(
            "decipher",
            _PROVIDER_KEYRING_ACCOUNTS.get(provider, f"{provider}_api_key"),
        )
        if key:
            return key
    except Exception:
        pass
    env_hint = " or ".join(_PROVIDER_ENV_KEYS.get(provider, []))
    print(
        "Error: No API key found. "
        f"Set {env_hint}, put it in .env, put it in "
        f".decipher_keys/{provider}_api_key, or configure keychain account "
        f"`{_PROVIDER_KEYRING_ACCOUNTS.get(provider, f'{provider}_api_key')}`.",
        file=sys.stderr,
    )
    sys.exit(1)


def _resolve_provider_and_model(args: argparse.Namespace) -> tuple[str, str]:
    from agent.model_provider import (
        default_model_for_provider,
        infer_provider_from_model,
    )

    requested_provider = getattr(args, "provider", None)
    requested_model = getattr(args, "model", None)
    provider = infer_provider_from_model(requested_model, requested_provider)
    model = requested_model or default_model_for_provider(provider)
    return provider, model


def _make_agent_provider(args: argparse.Namespace):
    from agent.model_provider import make_model_provider

    provider, model = _resolve_provider_and_model(args)
    return make_model_provider(
        provider=provider,
        api_key=get_api_key(provider),
        model=model,
    )


def _read_external_context(args: argparse.Namespace) -> str | None:
    """Return the external context string, loading from file if --context-file given."""
    ctx = getattr(args, "context", None)
    ctx_file = getattr(args, "context_file", None)
    if ctx_file:
        try:
            file_text = Path(ctx_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"Error reading context file {ctx_file!r}: {exc}", file=sys.stderr)
            sys.exit(1)
        return f"{ctx}\n\n{file_text}" if ctx else file_text
    return ctx or None


def _require_rust_fast_kernel() -> None:
    from analysis.polyalphabetic_fast import FAST_AVAILABLE, fast_kernel_unavailable_message

    if not FAST_AVAILABLE:
        print(
            fast_kernel_unavailable_message(
                feature="Decipher runtime"
            ),
            file=sys.stderr,
        )
        sys.exit(2)


def cmd_doctor(args: argparse.Namespace) -> None:
    from analysis.polyalphabetic_fast import fast_kernel_status

    status = fast_kernel_status()
    if getattr(args, "json", False):
        import json

        print(json.dumps({"rust_fast_kernel": status}, indent=2))
        return

    print("Decipher environment check")
    print()
    print("Rust fast kernels:")
    if status["available"]:
        print("  status: available")
        print(f"  module: {status.get('module_file')}")
    else:
        print("  status: not installed")
        if status.get("import_error"):
            print(f"  import error: {status['import_error']}")
        print()
        print("  Build from the repo root with:")
        print("    scripts/build_rust_fast.sh")
        print()
        print("  Manual equivalent:")
        print(f"    {status['build_command']}")
    print()
    print("  Features:")
    for feature in status["features"]:
        print(f"    - {feature}")
    print()
    print(f"  Note: {status['note']}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from benchmark.loader import BenchmarkLoader

    agentic = _use_agentic_mode(args)
    display_mode = _resolve_agent_display(args) if agentic else "off"
    quiet_structured_display = agentic and display_mode in {"pretty", "jsonl"}

    loader = BenchmarkLoader(args.benchmark_path)
    split_file = args.split or (
        f"{args.source}_tests.jsonl" if args.source else "all_tests.jsonl"
    )
    tests = loader.load_tests(split_file, track=args.track, source=args.source)
    if args.test_id:
        tests = [t for t in tests if t.test_id == args.test_id]
    if args.limit:
        tests = tests[: args.limit]
    if not tests:
        print("No matching tests found.", file=sys.stderr)
        sys.exit(1)

    if not agentic:
        from automated.runner import AutomatedBenchmarkRunner

        runner = AutomatedBenchmarkRunner(
            verbose=args.verbose,
            language=args.language,
            artifact_dir=args.artifact_dir or "artifacts",
            homophonic_budget=args.homophonic_budget,
            homophonic_refinement=args.homophonic_refinement,
            homophonic_solver="legacy" if args.legacy_homophonic else "zenith_native",
            transform_search=args.transform_search,
            transform_search_profile=args.transform_search_profile,
            transform_search_max_generated_candidates=args.transform_search_max_generated_candidates,
            transform_promote_artifact=args.transform_promote_artifact,
            transform_promote_candidate_ids=args.transform_promote_candidate_id,
            transform_promote_top_n=args.transform_promote_top_n,
        )
        mode_label = "automated"
    else:
        from benchmark.runner_v2 import BenchmarkRunnerV2

        provider, model = _resolve_provider_and_model(args)
        api = _make_agent_provider(args)
        runner = BenchmarkRunnerV2(
            claude_api=api,
            max_iterations=args.max_iterations,
            verbose=args.verbose and display_mode == "off",
            language=args.language,
            artifact_dir=args.artifact_dir or "artifacts",
            automated_preflight=not args.no_automated_preflight,
            display_mode=display_mode,
            external_context=_read_external_context(args),
            benchmark_context_policy=args.benchmark_context,
        )
        mode_label = f"agentic ({provider}/{model})"

    if not quiet_structured_display:
        print(f"Running {len(tests)} test(s) — mode={mode_label}, max_iter={args.max_iterations}")
        print(f"Artifacts → {args.artifact_dir or 'artifacts'}/<test_id>/<run_id>.json\n")

    results = []
    for i, test in enumerate(tests):
        if not quiet_structured_display:
            print(f"[{i+1}/{len(tests)}] {test.test_id} — {test.description}")
        test_data = loader.load_test_data(test)
        result = runner.run_test(test_data)
        conf = f"{result.self_confidence:.2f}" if result.self_confidence is not None else "n/a"
        if not quiet_structured_display:
            print(
                f"  Status: {result.status}, "
                f"Char: {result.char_accuracy:.1%}, "
                f"Word: {result.word_accuracy:.1%}, "
                f"Conf: {conf}, "
                f"Iter: {result.iterations_used}, "
                f"Time: {result.elapsed_seconds:.1f}s"
            )
            print(f"  Artifact: {result.artifact_path}")
            if result.error_message:
                print(f"  Error: {result.error_message}")
            if args.verbose and result.final_decryption:
                print(f"  Decryption: {result.final_decryption[:200]}")
            print()
        results.append(result)

    if results and not quiet_structured_display:
        n = len(results)
        avg_char = sum(r.char_accuracy for r in results) / n
        avg_word = sum(r.word_accuracy for r in results) / n
        success_status = "completed" if not agentic else "solved"
        success_label = "completed runs" if not agentic else "declared solutions"
        successful = sum(1 for r in results if r.status == success_status)
        print(f"AVERAGE: {successful}/{n} {success_label}, char={avg_char:.1%}, word={avg_word:.1%}")


def cmd_crack(args: argparse.Namespace) -> None:
    from benchmark.loader import parse_canonical_transcription
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText

    if args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("Error: No input text provided.", file=sys.stderr)
        sys.exit(1)

    if args.canonical:
        ct = parse_canonical_transcription(text)
    else:
        ignore = {" ", "\t", "\n", "\r"}
        alphabet = Alphabet.from_text(text, ignore_chars=ignore)
        clean = " ".join(text.split())
        ct = CipherText(raw=clean, alphabet=alphabet, source="cli", separator=" ")

    print(f"Alphabet: {ct.alphabet.size} symbols, {len(ct.tokens)} tokens, {len(ct.words)} words")

    from pathlib import Path

    cipher_id = args.cipher_id or "cli"
    artifact_dir = args.artifact_dir or "artifacts"

    if not _use_agentic_mode(args):
        from automated.runner import run_automated, save_crack_artifact

        print("Running automated solver (no LLM API calls)...")
        homophonic_budget = getattr(args, "homophonic_budget", "full")
        homophonic_refinement = getattr(args, "homophonic_refinement", "none")
        run_kwargs = {
            "cipher_text": ct,
            "language": args.language,
            "cipher_id": cipher_id,
            "homophonic_budget": homophonic_budget,
            "homophonic_refinement": homophonic_refinement,
            "homophonic_solver": "legacy" if getattr(args, "legacy_homophonic", False) else "zenith_native",
        }
        if getattr(args, "transform_search", "off") != "off":
            run_kwargs["transform_search"] = args.transform_search
            run_kwargs["transform_search_profile"] = getattr(args, "transform_search_profile", "broad")
            run_kwargs["transform_search_max_generated_candidates"] = getattr(
                args,
                "transform_search_max_generated_candidates",
                None,
            )
            run_kwargs["transform_promote_artifact"] = getattr(args, "transform_promote_artifact", None)
            run_kwargs["transform_promote_candidate_ids"] = getattr(args, "transform_promote_candidate_id", [])
            run_kwargs["transform_promote_top_n"] = getattr(args, "transform_promote_top_n", None)
        artifact = run_automated(**run_kwargs)
        path = save_crack_artifact(artifact, ct, args.language, artifact_dir)
        print(f"\nArtifact saved: {path}")
        print(f"Status: {artifact.status}")
        print(f"Solver: {artifact.solver}")
        print(f"Time: {artifact.elapsed_seconds:.1f}s")
        if artifact.error_message:
            print(f"Error: {artifact.error_message}")
        print(f"\nFinal decryption:\n{artifact.final_decryption}")
        return

    from agent.loop_v2 import run_v2
    from agent.display import make_agent_renderer

    provider, model = _resolve_provider_and_model(args)
    api = _make_agent_provider(args)
    display_mode = _resolve_agent_display(args)
    renderer = make_agent_renderer(display_mode)
    if renderer is not None:
        renderer.start_test(
            cipher_id,
            "Interactive crack",
            model=model,
            max_iterations=args.max_iterations,
        )

    automated_preflight = None
    if not args.no_automated_preflight:
        from automated.runner import format_automated_preflight_for_llm, run_automated

        if renderer is not None:
            renderer.event("preflight_start", {})
        else:
            print("Running automated preflight (no LLM access)...")
        homophonic_budget = getattr(args, "homophonic_budget", "full")
        homophonic_refinement = getattr(args, "homophonic_refinement", "none")
        run_kwargs = {
            "cipher_text": ct,
            "language": args.language,
            "cipher_id": cipher_id,
            "homophonic_budget": homophonic_budget,
            "homophonic_refinement": homophonic_refinement,
            "homophonic_solver": "legacy" if getattr(args, "legacy_homophonic", False) else "zenith_native",
        }
        if getattr(args, "transform_search", "off") != "off":
            run_kwargs["transform_search"] = args.transform_search
            run_kwargs["transform_search_profile"] = getattr(args, "transform_search_profile", "broad")
            run_kwargs["transform_search_max_generated_candidates"] = getattr(
                args,
                "transform_search_max_generated_candidates",
                None,
            )
            run_kwargs["transform_promote_artifact"] = getattr(args, "transform_promote_artifact", None)
            run_kwargs["transform_promote_candidate_ids"] = getattr(args, "transform_promote_candidate_id", [])
            run_kwargs["transform_promote_top_n"] = getattr(args, "transform_promote_top_n", None)
        preflight = run_automated(**run_kwargs)
        automated_preflight = dict(preflight.artifact)
        automated_preflight["summary"] = format_automated_preflight_for_llm(preflight)
        automated_preflight["enabled"] = True
        if renderer is not None:
            renderer.event("preflight_result", {
                "status": preflight.status,
                "solver": preflight.solver,
                "elapsed_seconds": preflight.elapsed_seconds,
            })
        else:
            print(
                f"  preflight: {preflight.status}, solver={preflight.solver}, "
                "$0.00 (no LLM access)"
            )

    def on_event(event: str, payload: dict) -> None:
        if renderer is not None:
            renderer.event(event, payload)
        elif event == "iteration_start":
            print(f"  iter {payload['iteration']}...", end="", flush=True)
        elif event == "tool_call":
            print(".", end="", flush=True)
        elif event in {"declared_solution", "run_complete", "error", "max_iterations_reached"}:
            print(f" [{event}]")

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,
        language=args.language,
        max_iterations=args.max_iterations,
        cipher_id=cipher_id,
        prior_context=_read_external_context(args),
        automated_preflight=automated_preflight,
        verbose=args.verbose and display_mode == "off",
        on_event=on_event,
    )

    print(f"Status: {artifact.status}")
    if artifact.solution:
        print(f"Declared branch: {artifact.solution.branch}")
        print(f"Self-confidence: {artifact.solution.self_confidence:.2f}")
        print(f"Rationale: {artifact.solution.rationale}")
    iterations = max(tc.iteration for tc in artifact.tool_calls) if artifact.tool_calls else 0
    print(f"Iterations: {iterations}")
    print(f"Tool calls: {len(artifact.tool_calls)}")

    final_branch = artifact.solution.branch if artifact.solution else "main"
    final_dec = next(
        (b.decryption for b in artifact.branches if b.name == final_branch),
        artifact.branches[0].decryption if artifact.branches else "",
    )
    from agent.final_summary import build_final_summary
    final_summary = build_final_summary(
        artifact,
        final_branch=final_branch,
        final_decryption=final_dec,
    )
    artifact.final_summary = final_summary
    path = Path(artifact_dir) / cipher_id / f"{artifact.run_id}.json"
    try:
        artifact.save(path)
        print(f"\nArtifact saved: {path}")
    except Exception as e:  # noqa: BLE001
        print(f"\nWarning: failed to save artifact: {e}")

    if renderer is not None:
        from types import SimpleNamespace

        renderer.finish(SimpleNamespace(
            test_id=cipher_id,
            status=artifact.status,
            char_accuracy=0.0,
            word_accuracy=0.0,
            iterations_used=iterations,
            elapsed_seconds=artifact.finished_at - artifact.started_at,
            total_tokens=artifact.total_input_tokens + artifact.total_output_tokens,
            estimated_cost_usd=artifact.estimated_cost_usd,
            artifact_path=str(path),
            error_message=artifact.error_message,
            final_decryption=final_dec,
            final_branch=final_branch,
            branch_scores=[],
            alignment_report="",
            final_summary=final_summary,
        ))
    print(f"\nFinal decryption ({final_branch}):\n{final_dec}")


def cmd_resume_artifact(args: argparse.Namespace) -> None:
    """Continue an agentic decipherment from a saved artifact."""
    from pathlib import Path
    from types import SimpleNamespace

    from agent.display import make_agent_renderer
    from agent.final_summary import build_final_summary
    from agent.loop_v2 import run_v2
    from agent.resume import (
        cipher_text_from_artifact,
        load_artifact_dict,
        resume_context_from_artifact,
    )
    from benchmark.scorer import (
        format_alignment,
        format_char_diff,
        has_word_boundaries,
        score_branch_decryptions,
    )

    parent_path = Path(args.artifact).expanduser().resolve()
    prior_artifact = load_artifact_dict(parent_path)
    ct = cipher_text_from_artifact(prior_artifact)
    language = args.language or prior_artifact.get("language") or "en"
    cipher_id = args.cipher_id or str(prior_artifact.get("cipher_id") or parent_path.stem)
    extra_iterations = args.extra_iterations
    branch = args.branch or (prior_artifact.get("solution") or {}).get("branch")

    if not getattr(args, "model", None) and prior_artifact.get("model"):
        args.model = prior_artifact.get("model")
    provider, model = _resolve_provider_and_model(args)
    api = _make_agent_provider(args)
    display_mode = _resolve_agent_display(args)
    renderer = make_agent_renderer(display_mode)
    if renderer is not None:
        renderer.start_test(
            cipher_id,
            f"Resume artifact {parent_path.name}",
            model=model,
            max_iterations=extra_iterations,
        )
    else:
        print(
            f"Resuming {cipher_id} from {parent_path} "
            f"for {extra_iterations} additional iteration(s)."
        )

    prior_context = resume_context_from_artifact(
        prior_artifact,
        branch=branch,
        extra_iterations=extra_iterations,
    )

    def on_event(event: str, payload: dict) -> None:
        if renderer is not None:
            renderer.event(event, payload)
        elif event == "iteration_start":
            print(f"  iter {payload['iteration']}...", end="", flush=True)
        elif event == "tool_call":
            print(".", end="", flush=True)
        elif event in {"declared_solution", "run_complete", "error", "max_iterations_reached"}:
            print(f" [{event}]")

    artifact = run_v2(
        cipher_text=ct,
        claude_api=api,
        language=language,
        max_iterations=extra_iterations,
        cipher_id=cipher_id,
        prior_context=prior_context,
        automated_preflight=None,
        resume_from_artifact=prior_artifact,
        resume_branch=branch,
        parent_artifact_path=str(parent_path),
        verbose=args.verbose and display_mode == "off",
        on_event=on_event,
    )

    ground_truth = prior_artifact.get("ground_truth")
    artifact.ground_truth = ground_truth
    branch_inputs = [
        (b.name, b.decryption, b.mapped_count) for b in artifact.branches
    ]
    branch_scores = (
        score_branch_decryptions(cipher_id, branch_inputs, ground_truth)
        if isinstance(ground_truth, str) and ground_truth.strip()
        else []
    )
    branch_acc_map = {r["branch"]: r for r in branch_scores}
    for b in artifact.branches:
        if b.name in branch_acc_map:
            b.char_accuracy = branch_acc_map[b.name]["char_accuracy"]
            b.word_accuracy = branch_acc_map[b.name]["word_accuracy"]

    final_branch = artifact.solution.branch if artifact.solution else (branch or "main")
    final_decryption = next(
        (b.decryption for b in artifact.branches if b.name == final_branch),
        artifact.branches[0].decryption if artifact.branches else "",
    )
    final_score = branch_acc_map.get(final_branch or "", {})
    artifact.char_accuracy = final_score.get("char_accuracy", 0.0)
    artifact.word_accuracy = final_score.get("word_accuracy", 0.0)
    alignment_report = ""
    if isinstance(ground_truth, str) and ground_truth.strip():
        if has_word_boundaries(ground_truth):
            alignment_report = format_alignment(final_decryption, ground_truth, max_words=50)
        else:
            alignment_report = format_char_diff(final_decryption, ground_truth)

    final_summary = build_final_summary(
        artifact,
        final_branch=final_branch or "",
        final_decryption=final_decryption,
    )
    artifact.final_summary = final_summary

    artifact_dir = Path(args.artifact_dir or "artifacts")
    path = artifact_dir / cipher_id / f"{artifact.run_id}.json"
    try:
        artifact.save(path)
    except Exception as e:  # noqa: BLE001
        print(f"\nWarning: failed to save artifact: {e}")

    iterations = max((tc.iteration for tc in artifact.tool_calls), default=0)
    result = SimpleNamespace(
        test_id=cipher_id,
        status=artifact.status,
        char_accuracy=artifact.char_accuracy or 0.0,
        word_accuracy=artifact.word_accuracy or 0.0,
        iterations_used=iterations,
        elapsed_seconds=artifact.finished_at - artifact.started_at,
        total_tokens=artifact.total_input_tokens + artifact.total_output_tokens,
        estimated_cost_usd=artifact.estimated_cost_usd,
        artifact_path=str(path),
        error_message=artifact.error_message,
        final_decryption=final_decryption,
        final_branch=final_branch,
        branch_scores=branch_scores,
        alignment_report=alignment_report,
        final_summary=final_summary,
    )
    if renderer is not None:
        renderer.finish(result)
    else:
        conf = (
            f"{artifact.solution.self_confidence:.2f}"
            if artifact.solution else "n/a"
        )
        print(
            f"\nStatus: {artifact.status}, Char: {result.char_accuracy:.1%}, "
            f"Word: {result.word_accuracy:.1%}, Conf: {conf}, "
            f"Iter: {iterations}, Time: {result.elapsed_seconds:.1f}s"
        )
        print(f"Artifact: {path}")
        if artifact.error_message:
            print(f"Error: {artifact.error_message}")
        print(f"\nFinal summary:\n{final_summary}")
        print(f"\nFinal decryption ({final_branch}):\n{final_decryption}")


def cmd_testgen(args: argparse.Namespace) -> None:
    from benchmark.scorer import score_decryption
    from testgen.builder import build_test_case
    from testgen.cache import PlaintextCache
    from testgen.spec import DifficultyPreset, TestSpec

    cache = PlaintextCache(args.cache_dir)

    if args.list_cache:
        entries = cache.list_entries()
        if not entries:
            print("Cache is empty.")
        else:
            print(f"{'File':<30} {'Lang':>4} {'Words':>5}  {'Topic':<20}  Generated")
            print("-" * 80)
            for e in entries:
                if "error" in e:
                    print(f"  {e['file']}  (unreadable)")
                else:
                    print(
                        f"  {e['file']:<28} {e['language']:>4} {e['word_count']:>5}  "
                        f"{e['topic']:<20}  {e['generated_at']}"
                    )
        return

    if args.flush_all_cache:
        n = cache.flush()
        print(f"Flushed {n} cache entries.")

    spec = TestSpec.from_preset(DifficultyPreset(args.preset), language=args.language)
    if args.length:
        spec.approx_length = args.length
    if args.topic != "general":
        spec.topic = args.topic
    if args.no_boundaries:
        spec.word_boundaries = False
    if args.cipher_system:
        cipher_system = args.cipher_system.lower()
        if cipher_system == "simple_substitution":
            spec.homophonic = False
            spec.polyalphabetic_variant = None
        elif cipher_system == "homophonic_substitution":
            spec.homophonic = True
            spec.polyalphabetic_variant = None
            spec.word_boundaries = False
        else:
            spec.homophonic = False
            spec.polyalphabetic_variant = cipher_system
            spec.word_boundaries = False if not args.keep_boundaries else spec.word_boundaries
    if args.poly_key:
        spec.polyalphabetic_key = args.poly_key
    if args.poly_period:
        spec.polyalphabetic_period = args.poly_period
    if args.seed is not None:
        spec.seed = args.seed
    spec.__post_init__()

    if args.flush_cache:
        n = cache.flush(spec)
        print(f"Flushed {n} cache entry for this spec.")

    if not _use_agentic_mode(args):
        cached = cache.get(spec)
        if cached is None and not args.dry_run:
            print(
                "Error: default automated testgen mode cannot generate new plaintext, "
                "because that would require an LLM API call. Use --agentic to "
                "generate and solve in one command, use --dry-run to populate the "
                "cache, or choose a cached spec.",
                file=sys.stderr,
            )
            sys.exit(1)
        provider, _model = _resolve_provider_and_model(args)
        api_key = "" if cached is not None else get_api_key(provider)
    else:
        provider, _model = _resolve_provider_and_model(args)
        api_key = get_api_key(provider)

    test_data = build_test_case(
        spec,
        cache,
        api_key,
        seed=args.seed,
        generator_provider=provider,
    )

    pt_preview = test_data.plaintext[:120] + ("..." if len(test_data.plaintext) > 120 else "")
    ct_preview = test_data.canonical_transcription[:120] + "..."
    print(f"Test ID:   {test_data.test.test_id}")
    print(f"Plaintext: {pt_preview}")
    print(f"Cipher:    {ct_preview}")
    print(f"Desc:      {test_data.test.description}")

    if args.dry_run:
        print("\n[dry-run] Skipping agent.")
        return

    if not _use_agentic_mode(args):
        from automated.runner import AutomatedBenchmarkRunner

        runner = AutomatedBenchmarkRunner(
            verbose=args.verbose,
            language=args.language,
            artifact_dir=args.artifact_dir,
            homophonic_budget=args.homophonic_budget,
            homophonic_refinement=args.homophonic_refinement,
            homophonic_solver="legacy" if args.legacy_homophonic else "zenith_native",
        )
        print("\nRunning automated solver (no LLM API calls)...")
        result = runner.run_test(test_data, language=args.language)
    else:
        from benchmark.runner_v2 import BenchmarkRunnerV2

        display_mode = _resolve_agent_display(args)
        provider, crack_model = _resolve_provider_and_model(args)
        crack_api = _make_agent_provider(args)
        runner = BenchmarkRunnerV2(
            claude_api=crack_api,
            max_iterations=args.max_iterations,
            verbose=args.verbose and display_mode == "off",
            language=args.language,
            artifact_dir=args.artifact_dir,
            automated_preflight=not args.no_automated_preflight,
            display_mode=display_mode,
        )
        print(
            f"\nRunning agent (provider={provider}, model={crack_model}, "
            f"max_iter={args.max_iterations})..."
        )
        result = runner.run_test(test_data)

    score = score_decryption(
        test_id=result.test_id,
        decrypted=result.final_decryption,
        ground_truth=test_data.plaintext,
        agent_score=0.0,
        status=result.status,
    )
    conf = f"{result.self_confidence:.2f}" if result.self_confidence is not None else "n/a"
    word_str = f"{score.word_accuracy:.1%}" if score.total_words > 1 else "N/A"
    print(f"\nStatus:     {result.status}")
    print(f"Char:       {score.char_accuracy:.1%}   Word: {word_str}")
    print(f"Confidence: {conf}   Iterations: {result.iterations_used}   Time: {result.elapsed_seconds:.1f}s")
    print(f"Artifact:   {result.artifact_path}")
    if result.error_message:
        print(f"Error:      {result.error_message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="decipher",
        description="Decipher — Classical Cipher Cryptanalysis Tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    doctor = subparsers.add_parser(
        "doctor",
        help="Check optional local dependencies such as Rust fast kernels",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable dependency status.",
    )

    # benchmark
    bench = subparsers.add_parser("benchmark", help="Run benchmark tests against historical datasets")
    bench.add_argument("benchmark_path", help="Path to benchmark root directory")
    bench.add_argument("--split", "-s", help="Split file name (default: auto-detect from source)")
    bench.add_argument("--track", "-t", default="transcription2plaintext")
    bench.add_argument("--source", help="Source filter (e.g. 'borg', 'copiale')")
    bench.add_argument("--test-id", help="Run a single test by ID")
    bench.add_argument("--limit", "-n", type=int, help="Maximum number of tests to run")
    bench.add_argument("--max-iterations", "-i", type=int, default=25)
    bench.add_argument(
        "--provider",
        choices=["anthropic", "claude", "openai", "gemini", "google"],
        help="LLM provider for agentic runs. Default is inferred from --model, else anthropic.",
    )
    bench.add_argument(
        "--model",
        "-m",
        help="LLM model name. Defaults by provider.",
    )
    bench.add_argument("--language", "-l", choices=["en", "la", "de", "fr", "it", "unknown"])
    bench.add_argument(
        "--context",
        metavar="TEXT",
        help=(
            "Free-form external context injected into the agent's initial context "
            "(e.g. date, source, suspected technique). Prepended before any benchmark context."
        ),
    )
    bench.add_argument(
        "--context-file",
        metavar="PATH",
        help="Path to a text file containing external context (combined with --context if both given).",
    )
    bench.add_argument(
        "--benchmark-context",
        choices=[
            "none",
            "minimal",
            "standard",
            "historical",
            "related_metadata",
            "related_solutions",
            "max",
        ],
        default="max",
        help=(
            "Benchmark manifest context available to agentic runs. Default "
            "`max` injects concise record context and allows manifest-declared "
            "related records/documents through scoped tools; it does not dump "
            "long related plaintexts into the opening prompt."
        ),
    )
    bench.add_argument("--artifact-dir", help="Artifact output directory (default: ./artifacts)")
    bench.add_argument("--verbose", "-v", action="store_true")
    bench.add_argument(
        "--display",
        choices=["auto", "pretty", "raw", "jsonl"],
        default="auto",
        help=(
            "Agentic terminal display mode. auto uses pretty on an interactive "
            "terminal, raw when piped, and the legacy verbose stream with -v."
        ),
    )
    bench.add_argument(
        "--agentic",
        action="store_true",
        help="Use the experimental LLM agent instead of the default automated solver.",
    )
    bench.add_argument(
        "--automated-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    bench.add_argument("--no-automated-preflight", action="store_true",
                       help="Disable the default no-LLM automated preflight before agentic runs.")
    bench.add_argument(
        "--homophonic-budget",
        choices=["full", "screen"],
        default="full",
        help="Search budget for automated homophonic runs.",
    )
    bench.add_argument(
        "--homophonic-refinement",
        choices=["none", "two_stage", "targeted_repair", "family_repair"],
        default="none",
        help="Optional second-stage local refinement for automated homophonic runs.",
    )
    bench.add_argument(
        "--legacy-homophonic",
        action="store_true",
        help="Use the older pre-zenith_native homophonic solver path for comparison.",
    )
    bench.add_argument(
        "--transform-search",
        choices=["off", "auto", "screen", "wide", "rank", "full", "promote"],
        default="off",
        help=(
            "Run cheap transform-search diagnostics for automated runs. "
            "`auto` screens only when router signals are promising; `screen` "
            "records a structural candidate menu; `wide` runs a larger "
            "structural-only search; `rank`/`full` run bounded solver probes "
            "on top candidates; `promote` probes candidates from a prior "
            "wide/screen artifact."
        ),
    )
    bench.add_argument(
        "--transform-search-profile",
        choices=["fast", "broad", "wide"],
        default="broad",
        help=(
            "Candidate breadth profile for transform-search rank/full. "
            "`fast` is recommended for regression runs and trims mutations "
            "and confirmations; `wide` expands the structural-only candidate sweep."
        ),
    )
    bench.add_argument(
        "--transform-search-max-generated-candidates",
        type=int,
        help=(
            "Optional safety cap for transform-search structural candidate "
            "generation. Use with --transform-search wide for larger sweeps."
        ),
    )
    bench.add_argument(
        "--transform-promote-artifact",
        help="Source automated artifact containing transform_search.screen candidates to promote.",
    )
    bench.add_argument(
        "--transform-promote-candidate-id",
        action="append",
        default=[],
        help="Specific transform candidate id to promote from the source artifact. May be repeated.",
    )
    bench.add_argument(
        "--transform-promote-top-n",
        type=int,
        help="Promote the top N structural candidates from the source artifact.",
    )

    # crack
    crack = subparsers.add_parser("crack", help="Crack a cipher from file or stdin")
    crack.add_argument("--file", "-f", help="Input file (default: stdin)")
    crack.add_argument("--canonical", action="store_true",
                       help="Input is canonical S-token format (space-separated, | word breaks)")
    crack.add_argument("--max-iterations", "-i", type=int, default=25)
    crack.add_argument(
        "--provider",
        choices=["anthropic", "claude", "openai", "gemini", "google"],
        help="LLM provider for agentic runs. Default is inferred from --model, else anthropic.",
    )
    crack.add_argument("--model", "-m", help="LLM model name. Defaults by provider.")
    crack.add_argument("--language", "-l", choices=["en", "la", "de", "fr", "it", "unknown"],
                       default="en")
    crack.add_argument(
        "--context",
        metavar="TEXT",
        help=(
            "Free-form external context injected into the agent's initial context "
            "(e.g. date, source, suspected technique)."
        ),
    )
    crack.add_argument(
        "--context-file",
        metavar="PATH",
        help="Path to a text file containing external context (combined with --context if both given).",
    )
    crack.add_argument("--artifact-dir", help="Artifact output directory (default: ./artifacts)")
    crack.add_argument("--cipher-id", help="Identifier for this cipher (default: 'cli')")
    crack.add_argument("--verbose", "-v", action="store_true")
    crack.add_argument(
        "--display",
        choices=["auto", "pretty", "raw", "jsonl"],
        default="auto",
        help="Agentic terminal display mode.",
    )
    crack.add_argument(
        "--agentic",
        action="store_true",
        help="Use the experimental LLM agent instead of the default automated solver.",
    )
    crack.add_argument("--automated-only", action="store_true", help=argparse.SUPPRESS)
    crack.add_argument("--no-automated-preflight", action="store_true",
                       help="Disable the default no-LLM automated preflight before agentic runs.")
    crack.add_argument(
        "--homophonic-budget",
        choices=["full", "screen"],
        default="full",
        help="Search budget for automated homophonic runs.",
    )
    crack.add_argument(
        "--homophonic-refinement",
        choices=["none", "two_stage", "targeted_repair", "family_repair"],
        default="none",
        help="Optional second-stage local refinement for automated homophonic runs.",
    )
    crack.add_argument(
        "--legacy-homophonic",
        action="store_true",
        help="Use the older pre-zenith_native homophonic solver path for comparison.",
    )
    crack.add_argument(
        "--transform-search",
        choices=["off", "auto", "screen", "wide", "rank", "full", "promote"],
        default="off",
        help=(
            "Run cheap transform-search diagnostics in automated/preflight runs. "
            "`wide` runs a larger structural-only search; `rank`/`full` run "
            "bounded solver probes on top transform candidates; `promote` "
            "probes candidates from a prior wide/screen artifact."
        ),
    )
    crack.add_argument(
        "--transform-search-profile",
        choices=["fast", "broad", "wide"],
        default="broad",
        help=(
            "Candidate breadth profile for transform-search rank/full. "
            "`fast` is recommended for regression runs and trims mutations "
            "and confirmations; `wide` expands the structural-only candidate sweep."
        ),
    )
    crack.add_argument(
        "--transform-search-max-generated-candidates",
        type=int,
        help=(
            "Optional safety cap for transform-search structural candidate "
            "generation. Use with --transform-search wide for larger sweeps."
        ),
    )
    crack.add_argument(
        "--transform-promote-artifact",
        help="Source automated artifact containing transform_search.screen candidates to promote.",
    )
    crack.add_argument(
        "--transform-promote-candidate-id",
        action="append",
        default=[],
        help="Specific transform candidate id to promote from the source artifact. May be repeated.",
    )
    crack.add_argument(
        "--transform-promote-top-n",
        type=int,
        help="Promote the top N structural candidates from the source artifact.",
    )

    # resume-artifact
    resume = subparsers.add_parser(
        "resume-artifact",
        help="Continue an agentic decipherment from a saved artifact",
    )
    resume.add_argument("artifact", help="Path to a prior agentic artifact JSON")
    resume.add_argument(
        "--extra-iterations",
        "-i",
        type=int,
        default=10,
        help="Additional outer iterations to run from the restored state.",
    )
    resume.add_argument(
        "--branch",
        help="Branch to focus on from the prior artifact (default: declared branch).",
    )
    resume.add_argument(
        "--provider",
        choices=["anthropic", "claude", "openai", "gemini", "google"],
        help="LLM provider for the continuation. Default is inferred from --model.",
    )
    resume.add_argument("--model", "-m", help="LLM model name (default: prior artifact model)")
    resume.add_argument("--language", "-l", choices=["en", "la", "de", "fr", "it", "unknown"])
    resume.add_argument("--artifact-dir", help="Artifact output directory (default: ./artifacts)")
    resume.add_argument("--cipher-id", help="Override cipher id for the continuation artifact")
    resume.add_argument("--verbose", "-v", action="store_true")
    resume.add_argument(
        "--display",
        choices=["auto", "pretty", "raw", "jsonl"],
        default="auto",
        help="Agentic terminal display mode.",
    )

    # testgen
    tg = subparsers.add_parser("testgen", help="Generate a synthetic test case and run the agent")
    tg.add_argument("--language", "-l", choices=["en", "it", "de", "fr", "la"], default="en")
    tg.add_argument(
        "--preset", "-p",
        choices=["tiny", "medium", "hard", "hardest"],
        default="medium",
    )
    tg.add_argument("--length", type=int, help="Override approx word count from preset")
    tg.add_argument("--topic", default="general")
    tg.add_argument("--no-boundaries", action="store_true")
    tg.add_argument(
        "--cipher-system",
        choices=[
            "simple_substitution",
            "homophonic_substitution",
            "vigenere",
            "beaufort",
            "variant_beaufort",
            "gronsfeld",
        ],
        help=(
            "Synthetic cipher family to generate. Defaults to the selected preset; "
            "periodic polyalphabetic systems are generated no-boundary unless "
            "--keep-boundaries is supplied."
        ),
    )
    tg.add_argument(
        "--poly-key",
        help=(
            "Explicit periodic key for Vigenere-family synthetic cases. "
            "Use letters for Vigenere/Beaufort and digits for Gronsfeld."
        ),
    )
    tg.add_argument(
        "--poly-period",
        type=int,
        help="Random periodic key length when --poly-key is omitted.",
    )
    tg.add_argument(
        "--keep-boundaries",
        action="store_true",
        help="Keep word-boundary formatting for periodic synthetic cases.",
    )
    tg.add_argument("--seed", type=int)
    tg.add_argument("--flush-cache", action="store_true")
    tg.add_argument("--flush-all-cache", action="store_true")
    tg.add_argument("--list-cache", action="store_true")
    tg.add_argument("--dry-run", action="store_true")
    tg.add_argument("--max-iterations", "-i", type=int, default=25)
    tg.add_argument(
        "--provider",
        choices=["anthropic", "claude", "openai", "gemini", "google"],
        help="LLM provider for generation/agentic runs. Default is inferred from --model, else anthropic.",
    )
    tg.add_argument("--model", "-m")
    tg.add_argument("--artifact-dir", default="artifacts")
    tg.add_argument("--cache-dir", default="testgen_cache")
    tg.add_argument("--verbose", "-v", action="store_true")
    tg.add_argument(
        "--display",
        choices=["auto", "pretty", "raw", "jsonl"],
        default="auto",
        help="Agentic terminal display mode.",
    )
    tg.add_argument(
        "--agentic",
        action="store_true",
        help=(
            "Use the experimental LLM agent for solving and allow uncached synthetic "
            "plaintext generation."
        ),
    )
    tg.add_argument("--automated-only", action="store_true", help=argparse.SUPPRESS)
    tg.add_argument("--no-automated-preflight", action="store_true",
                    help="Disable the default no-LLM automated preflight before agentic runs.")
    tg.add_argument(
        "--homophonic-budget",
        choices=["full", "screen"],
        default="full",
        help="Search budget for automated homophonic runs.",
    )
    tg.add_argument(
        "--homophonic-refinement",
        choices=["none", "two_stage", "targeted_repair", "family_repair"],
        default="none",
        help="Optional second-stage local refinement for automated homophonic runs.",
    )
    tg.add_argument(
        "--legacy-homophonic",
        action="store_true",
        help="Use the older pre-zenith_native homophonic solver path for comparison.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)
    if args.command != "doctor":
        _require_rust_fast_kernel()

    dispatch = {
        "doctor": cmd_doctor,
        "benchmark": cmd_benchmark,
        "crack": cmd_crack,
        "resume-artifact": cmd_resume_artifact,
        "testgen": cmd_testgen,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
