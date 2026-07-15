#!/usr/bin/env python3
"""CLI entry point for Decipher — run benchmarks or crack ciphers headlessly."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path


def _use_agentic_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "agentic", False))


def _resolve_agent_display(args: argparse.Namespace) -> str:
    # F7: narrate is the default agentic display (a scrolling, pipe-safe,
    # Claude-Code-style transcript). ``auto`` also resolves to narrate. Unlike
    # the old behavior, ``-v`` no longer forces ``off`` — it means MORE detail
    # inside whichever renderer is active (decision 1).
    requested = getattr(args, "display", "narrate")
    if requested != "auto":
        return requested
    return "narrate"


_PROVIDER_ENV_KEYS = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    # Ollama is local — no API key required.
}

_PROVIDER_KEYRING_ACCOUNTS = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "gemini": "gemini_api_key",
    "openrouter": "openrouter_api_key",
}

# Providers that run locally and never need an API key.
_LOCAL_PROVIDERS = {"ollama"}

HOMOPHONIC_REFINEMENT_CHOICES = [
    "none",
    "two_stage",
    "targeted_repair",
    "family_repair",
    "null_masks",
    "homophonic_nulls",
    "copiale_nulls",
    "word_repair",
    "null_masks+word_repair",
]


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
    if provider in _LOCAL_PROVIDERS:
        return ""  # local providers need no key
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


def _probe_api_key(provider: str) -> str:
    """Return the API key for *provider* if one is configured, else ''.

    Silent version of get_api_key — never prints or exits.  Only meaningful
    for remote providers (anthropic, openai, gemini); local providers like
    ollama have no key and should be probed via _ollama_status() instead.
    """
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
    return ""


def _auto_detect_provider() -> str:
    """Return the first available provider in preference order.

    Preference: anthropic → openai → gemini → openrouter → ollama (only if
    server is reachable).  Falls back to "anthropic" so that the normal
    get_api_key() error message fires if nothing is configured.
    """
    for provider in ("anthropic", "openai", "gemini", "openrouter"):
        if _probe_api_key(provider):
            return provider
    # Ollama is local and needs no key, but only treat it as available if the
    # server is actually running — otherwise the user gets a confusing
    # connection-refused error rather than a helpful "no key" message.
    if _ollama_status()["running"]:
        return "ollama"
    return "anthropic"  # default so get_api_key() produces the right error


def _resolve_provider_and_model(args: argparse.Namespace) -> tuple[str, str]:
    from agent.model_provider import (
        default_model_for_provider,
        infer_provider_from_model,
    )

    requested_provider = getattr(args, "provider", None)
    requested_model = getattr(args, "model", None)
    if requested_provider or requested_model:
        # Explicit flags: honour them exactly as before.
        provider = infer_provider_from_model(requested_model, requested_provider)
    else:
        # Nothing specified: pick the first provider that has a key (or a
        # running Ollama instance) so the user doesn't need --provider when
        # only one provider is configured.
        provider = _auto_detect_provider()
    model = requested_model or default_model_for_provider(provider)
    return provider, model


def _resolve_system_prompt_style(args: argparse.Namespace) -> str:
    """Return 'full' or 'compact' based on --system-prompt-style and provider."""
    requested = getattr(args, "system_prompt_style", "auto")
    if requested != "auto":
        return requested
    provider, _ = _resolve_provider_and_model(args)
    return "compact" if provider == "ollama" else "full"


def _make_agent_provider(args: argparse.Namespace):
    from agent.model_provider import make_model_provider

    provider, model = _resolve_provider_and_model(args)
    return make_model_provider(
        provider=provider,
        api_key=get_api_key(provider),
        model=model,
    )


def _preflight_model_check(args: argparse.Namespace) -> None:
    """Validate provider+model before the renderer starts.

    Exits with a clear error message if the model is definitively not found.
    Prints a warning (but continues) for soft mismatches on non-OpenRouter
    providers.  Silent no-op when the check cannot be completed (e.g. no
    network).
    """
    from agent.model_provider import validate_model

    provider, model = _resolve_provider_and_model(args)
    ok, hint = validate_model(provider, model)
    if not ok:
        print(f"Error: {hint}", file=sys.stderr)
        sys.exit(2)
    if hint:  # non-empty hint on a True result is a warning
        print(hint, file=sys.stderr)


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


def _add_artifact_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "After writing each agentic artifact, run scripts/inspect_artifact.py "
            "--analyze and save a sibling .analyzed.md report."
        ),
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["standard", "deep"],
        default="standard",
        help="Depth of automatic artifact LLM analysis when --analyze is set.",
    )
    parser.add_argument(
        "--analysis-max-tokens",
        type=int,
        default=2500,
        help=(
            "Maximum output tokens for automatic artifact LLM analysis. "
            "Increase this if the diagnosis ends mid-sentence."
        ),
    )
    parser.add_argument(
        "--analysis-timeout",
        type=float,
        default=None,
        help=(
            "Optional per-request timeout in seconds for automatic artifact "
            "analysis adapters that expose timeouts, currently OpenRouter and "
            "Ollama."
        ),
    )
    parser.add_argument(
        "--analysis-no-empty-retry",
        action="store_true",
        help=(
            "Do not retry automatic artifact analysis when a provider reports "
            "output tokens but returns no visible text."
        ),
    )


def _analysis_output_path(artifact_path: str | Path) -> Path:
    return Path(artifact_path).with_suffix(".analyzed.md")


def _load_inspect_artifact_module():
    script = Path(__file__).resolve().parent.parent / "scripts" / "inspect_artifact.py"
    spec = importlib.util.spec_from_file_location("inspect_artifact_cli", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load artifact inspector: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _maybe_write_artifact_analysis(artifact_path: str | Path, args: argparse.Namespace) -> Path | None:
    if not getattr(args, "analyze", False):
        return None
    path = Path(artifact_path)
    if not path.exists():
        print(f"Warning: cannot analyze missing artifact: {path}", file=sys.stderr)
        return None
    output_path = _analysis_output_path(path)
    print(
        f"Performing LLM artifact analysis for {path} -> {output_path}...",
        file=sys.stderr,
        flush=True,
    )
    try:
        inspector = _load_inspect_artifact_module()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print(f"# Artifact Analysis: {path.name}")
            print()
            print(f"Source artifact: `{path}`")
            print()
            inspector.inspect_one(
                path,
                analyze=True,
                provider=getattr(args, "provider", None),
                llm_model=getattr(args, "model", None),
                max_tokens=getattr(args, "analysis_max_tokens", 2500),
                analysis_mode=getattr(args, "analysis_mode", "standard"),
                timeout_seconds=getattr(args, "analysis_timeout", None),
                retry_empty_response=not getattr(args, "analysis_no_empty_retry", False),
            )
        output_path.write_text(buffer.getvalue(), encoding="utf-8")
        print(f"Artifact analysis saved: {output_path}", file=sys.stderr)
        return output_path
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to analyze artifact {path}: {exc}", file=sys.stderr)
        return None


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


def _provider_key_status(provider: str) -> dict:
    """Return key presence and source for one provider (never reveals the key value)."""
    if provider in _LOCAL_PROVIDERS:
        return {"found": True, "source": "local — no key required"}
    for env_name in _PROVIDER_ENV_KEYS.get(provider, []):
        if os.environ.get(env_name):
            return {"found": True, "source": f"env:{env_name}"}
    if _read_dotenv_key(provider):
        return {"found": True, "source": ".env file"}
    if _read_key_file(provider):
        return {"found": True, "source": f".decipher_keys/{provider}_api_key"}
    try:
        import keyring
        account = _PROVIDER_KEYRING_ACCOUNTS.get(provider, f"{provider}_api_key")
        if keyring.get_password("decipher", account):
            return {"found": True, "source": f"keychain:{account}"}
    except Exception:
        pass
    return {"found": False, "source": None}


def _ollama_status() -> dict:
    """Check whether Ollama is reachable and return installed model names."""
    import os
    import urllib.error
    import urllib.request

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            import json
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        return {"running": True, "host": host, "models": models}
    except Exception:
        return {"running": False, "host": host, "models": []}


def cmd_doctor(args: argparse.Namespace) -> None:
    from analysis.polyalphabetic_fast import fast_kernel_status
    from agent.model_provider import (
        _PRICING,
        _default_openrouter_cache_path,
        _load_openrouter_disk_cache,
        default_model_for_provider,
        fetch_openrouter_pricing,
    )

    # --refresh-pricing: fetch OpenRouter live pricing and show a diff.
    if getattr(args, "refresh_pricing", False):
        cache_path = _default_openrouter_cache_path()
        old_pricing = _load_openrouter_disk_cache(cache_path) or {}
        print("Fetching OpenRouter model pricing…", flush=True)
        try:
            new_pricing = fetch_openrouter_pricing(cache_path=cache_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Fetched {len(new_pricing)} models → {cache_path}")
        # Show diff against previous cache for models we care about.
        all_ids = sorted(set(old_pricing) | set(new_pricing))
        changes = []
        for mid in all_ids:
            old = old_pricing.get(mid)
            new = new_pricing.get(mid)
            if old == new:
                continue
            if old is None:
                changes.append(f"  + {mid}: in=${new[0]:.3f}/M  out=${new[1]:.3f}/M")
            elif new is None:
                changes.append(f"  - {mid}: removed")
            else:
                changes.append(
                    f"  ~ {mid}: "
                    f"in ${old[0]:.3f}→${new[0]:.3f}/M  "
                    f"out ${old[1]:.3f}→${new[1]:.3f}/M"
                )
        if changes:
            print(f"\n{len(changes)} change(s) vs previous cache:")
            for line in changes:
                print(line)
        else:
            print("No changes vs previous cache.")
        return

    rust_status = fast_kernel_status()

    # Build provider info: key status + model list
    providers_info = {}
    for provider in ("anthropic", "openai", "gemini", "openrouter"):
        key_status = _provider_key_status(provider)
        models = sorted(_PRICING.get(provider, {}).keys())
        providers_info[provider] = {
            "default_model": default_model_for_provider(provider),
            "models": models,
            "key": key_status,
        }

    # Ollama: live check for running daemon + installed models
    ollama = _ollama_status()
    providers_info["ollama"] = {
        "default_model": default_model_for_provider("ollama"),
        "models": ollama["models"],
        "key": _provider_key_status("ollama"),
        "running": ollama["running"],
        "host": ollama["host"],
    }

    if getattr(args, "json", False):
        import json
        print(json.dumps({
            "rust_fast_kernel": rust_status,
            "providers": providers_info,
        }, indent=2))
        return

    print("Decipher environment check")
    print()

    # --- Rust kernels ---
    print("Rust fast kernels:")
    if rust_status["available"]:
        print("  status: available")
        print(f"  module: {rust_status.get('module_file')}")
    else:
        print("  status: not installed")
        if rust_status.get("import_error"):
            print(f"  import error: {rust_status['import_error']}")
        print()
        print("  Build from the repo root with:")
        print("    scripts/build_rust_fast.sh")
        print()
        print("  Manual equivalent:")
        print(f"    {rust_status['build_command']}")
    print()
    print("  Features:")
    for feature in rust_status["features"]:
        print(f"    - {feature}")
    print()
    print(f"  Note: {rust_status['note']}")
    print()

    # --- Cloud providers / API keys ---
    print("LLM providers:")
    for provider in ("anthropic", "openai", "gemini", "openrouter"):
        info = providers_info[provider]
        key = info["key"]
        key_line = f"key: {key['source']}" if key["found"] else "key: not found"
        print(f"  {provider}  ({key_line})")
        print(f"    default model : {info['default_model']}")
        print(f"    known models  : {', '.join(info['models']) or '(none)'}")
    print()
    print("  To add a key: export ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY /")
    print("  OPENROUTER_API_KEY, or write it to .decipher_keys/<provider>_api_key.")
    print()

    # --- Ollama ---
    ollama_info = providers_info["ollama"]
    host = ollama_info["host"]
    if ollama_info["running"]:
        installed = ollama_info["models"]
        print(f"Ollama  (running at {host})")
        print(f"  default model : {ollama_info['default_model']}")
        if installed:
            print(f"  installed     : {', '.join(installed)}")
        else:
            print("  installed     : (none — run: ollama pull qwen3:14b)")
    else:
        print(f"Ollama  (not running at {host})")
        print("  Start with: ollama serve")
        print("  Install models with: ollama pull qwen3:14b")
    print()
    print("  Usage: decipher crack -f cipher.txt --agentic --provider ollama --model qwen3:14b")


def _run_multipage_group_benchmark(args: argparse.Namespace, agentic: bool) -> None:
    """Run the automated multipage shared-key route for --multipage-group."""
    if agentic:
        print(
            "Error: --multipage-group is automated-only; agentic multipage is out "
            "of scope. Drop --agentic.",
            file=sys.stderr,
        )
        sys.exit(2)

    from benchmark.loader import BenchmarkLoader
    from automated.multipage_route import load_group_definition, run_automated_multipage

    try:
        group = load_group_definition(args.multipage_group)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    loader = BenchmarkLoader(args.benchmark_path)
    artifact_dir = args.artifact_dir or "artifacts"
    homophonic_solver = "legacy" if args.legacy_homophonic else "zenith_native"
    print(
        f"Running multipage group '{group['name']}' "
        f"({len(group['test_ids'])} pages) — automated, "
        f"budget={args.homophonic_budget}, refinement={args.homophonic_refinement}"
    )
    print(f"Artifacts → {artifact_dir}/automated_multipage/{group['name']}/\n")

    try:
        result = run_automated_multipage(
            loader=loader,
            group=group,
            homophonic_budget=args.homophonic_budget,
            homophonic_refinement=args.homophonic_refinement,
            homophonic_solver=homophonic_solver,
            language=args.language,
            artifact_dir=artifact_dir,
            model_variant=getattr(args, "model_variant", None),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Combined: status={result.status}, solver={result.solver}, "
        f"tokens={result.combined_token_count}, symbols={result.combined_alphabet_size}, "
        f"mask={result.selected_mask or '(none)'}"
    )
    print(
        f"Combined solve: {result.combined_solve_seconds:.1f}s "
        f"(total {result.elapsed_seconds:.1f}s)"
    )
    if result.word_repair is not None:
        adopted = result.word_repair.get("adopted")
        print(
            "Word repair (group): "
            f"adopt_enabled={result.word_repair_adopt_enabled}, "
            f"adopted={'yes' if adopted else 'no'}"
        )
    print("\nPer-page (post-hoc grading vs known plaintext):")
    for page in result.page_results:
        print(
            f"  {page.test_id}: char={page.char_accuracy:.1%}, "
            f"word={page.word_accuracy:.1%}, filtered={page.filtered_length}"
        )
    print(
        f"\nAGGREGATE: char={result.aggregate_char_accuracy:.1%}, "
        f"word={result.aggregate_word_accuracy:.1%}"
    )
    print(f"Group artifact: {result.group_artifact_path}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from benchmark.loader import BenchmarkLoader

    agentic = _use_agentic_mode(args)
    if getattr(args, "multipage_group", None):
        _run_multipage_group_benchmark(args, agentic)
        return
    if getattr(args, "analyze", False) and not agentic:
        print("Note: --analyze is only run for --agentic artifacts; ignoring.", file=sys.stderr)
    display_mode = _resolve_agent_display(args) if agentic else "off"

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
        if Path(split_file).is_absolute():
            searched = str(Path(split_file))
        else:
            searched = str(loader.root / "splits" / split_file)
        if args.split:
            origin = "specified via --split"
        elif args.source:
            origin = f"auto-detected from --source {args.source}"
        else:
            origin = "default (no --source or --split given)"
        applied = []
        if args.source:
            applied.append(f"--source {args.source}")
        if args.test_id:
            applied.append(f"--test-id {args.test_id}")
        if args.track:
            applied.append(f"--track {args.track}")
        if args.limit:
            applied.append(f"--limit {args.limit}")
        filters = ", ".join(applied) if applied else "(none)"
        print(
            f"No matching tests found in split '{split_file}' ({origin}).",
            file=sys.stderr,
        )
        print(f"  Searched: {searched}", file=sys.stderr)
        print(f"  Filters applied: {filters}", file=sys.stderr)
        if not args.split:
            print(
                "  Hint: synthetic test ids (e.g. 'synth_en_...') live in a "
                "dedicated split and need an explicit --split, e.g. "
                "'--split en_ss_synth_nb_tests.jsonl'.",
                file=sys.stderr,
            )
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
            model_variant=getattr(args, "model_variant", None),
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

        _preflight_model_check(args)
        provider, model = _resolve_provider_and_model(args)
        api = _make_agent_provider(args)
        if getattr(args, "model_variant", None):
            # Honest scope note: in agentic runs the flag reaches only the
            # automated (no-LLM) preflight solve inside BenchmarkRunnerV2; the
            # agent controls its own model variant via act_set_model_variant.
            if args.no_automated_preflight:
                print(
                    "Note: --model-variant is IGNORED for agentic runs with "
                    "--no-automated-preflight (it only affects the automated "
                    "preflight solve). The agent can still switch mid-run via "
                    "act_set_model_variant.",
                    file=sys.stderr,
                )
            else:
                print(
                    "Note: --model-variant applies to the automated (no-LLM) "
                    "preflight solve only in agentic runs; the agent can switch "
                    "its own model variant mid-run via act_set_model_variant.",
                    file=sys.stderr,
                )
        runner = BenchmarkRunnerV2(
            claude_api=api,
            max_iterations=args.max_iterations,
            verbose=args.verbose and display_mode == "off",
            renderer_verbose=args.verbose,
            language=args.language,
            artifact_dir=args.artifact_dir or "artifacts",
            automated_preflight=not args.no_automated_preflight,
            display_mode=display_mode,
            external_context=_read_external_context(args),
            benchmark_context_policy=args.benchmark_context,
            system_prompt_style=_resolve_system_prompt_style(args),
            homophonic_budget=args.homophonic_budget,
            homophonic_refinement=args.homophonic_refinement,
            homophonic_solver="legacy" if args.legacy_homophonic else "zenith_native",
            agent_loop=getattr(args, "agent_loop", "v2"),
            model_variant=getattr(args, "model_variant", None),
        )
        mode_label = f"agentic ({provider}/{model}, loop={getattr(args, 'agent_loop', 'v2')})"

    # F1: keep stdout machine-clean in jsonl mode by routing the human-readable
    # scrollback (progress lines, per-test result block, report table, artifact
    # paths) to stderr; every other mode prints to stdout. The pretty-mode
    # interleaving worry does not apply here — each renderer's lifecycle is fully
    # contained inside runner.run_test (F1), so no Live is active at these prints.
    human_out = sys.stderr if (agentic and display_mode == "jsonl") else sys.stdout

    # F8: the automated (no-LLM) path has no renderer; a direct-printing on_step
    # callback narrates each pipeline step boundary. Agentic runs render steps
    # through the renderer instead, so on_step stays None there.
    on_step = None
    if not agentic:
        def on_step(name: str, status: str | None, elapsed: float) -> None:
            tail = f" {status}" if status else " done"
            print(f"  · step: {name} …{tail} ({elapsed:.1f}s)", file=human_out, flush=True)

    print(
        f"Running {len(tests)} test(s) — mode={mode_label}, max_iter={args.max_iterations}",
        file=human_out,
    )
    print(
        f"Artifacts → {args.artifact_dir or 'artifacts'}/<test_id>/<run_id>.json\n",
        file=human_out,
    )

    results = []
    for i, test in enumerate(tests):
        print(f"[{i+1}/{len(tests)}] {test.test_id} — {test.description}", file=human_out)
        test_data = loader.load_test_data(test)
        if agentic:
            result = runner.run_test(test_data)
        else:
            result = runner.run_test(test_data, on_step=on_step)
        conf = f"{result.self_confidence:.2f}" if result.self_confidence is not None else "n/a"
        # Decision 5 / F1: the per-test result block is ALWAYS printed to
        # scrollback (in every display mode; jsonl routes it to stderr above) so
        # a human debugging the run always has status + accuracies + the artifact
        # path, even under a transient/garbled live panel or machine output.
        print(
            f"  Status: {result.status}, "
            f"Comparison to known ground-truth plaintext: "
            f"Char: {result.char_accuracy:.1%}, "
            f"Word: {result.word_accuracy:.1%}, "
            f"Conf: {conf}, "
            f"Iter: {result.iterations_used}, "
            f"Time: {result.elapsed_seconds:.1f}s",
            file=human_out,
        )
        print(f"  Artifact: {result.artifact_path}", file=human_out)
        if result.error_message:
            print(f"  Error: {result.error_message}", file=human_out)
        if args.verbose and result.final_decryption:
            print(f"  Decryption: {result.final_decryption[:200]}", file=human_out)
        print(file=human_out)
        if agentic:
            _maybe_write_artifact_analysis(result.artifact_path, args)
        results.append(result)

    if results:
        from benchmark.scorer import ReportRow, format_report

        rows = [
            ReportRow(
                test_id=r.test_id,
                status=r.status,
                char_accuracy=r.char_accuracy,
                word_accuracy=r.word_accuracy,
                duration=r.elapsed_seconds,
                cost=getattr(r, "estimated_cost_usd", 0.0) or 0.0,
            )
            for r in results
        ]
        # Decision 5: the per-test report table replaces the inline AVERAGE line.
        print(format_report(rows), file=human_out)

        fallback_count = sum(1 for r in results if r.status == "fallback_declared")
        if fallback_count:
            print(f"(fallback declarations: {fallback_count})", file=human_out)

        # Decision 4: a consolidated Artifacts list (one line per run) + the
        # run-dir root, always printed so the CLI is self-sufficient for humans.
        print("Artifacts:", file=human_out)
        for r in results:
            print(f"  {r.test_id}: {r.artifact_path}", file=human_out)
        print(f"Run dir root: {args.artifact_dir or 'artifacts'}", file=human_out)


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

    # F-2: resolve the agentic display once (the automated path has no renderer,
    # so "off"). Route the pre-run header line to stderr in jsonl mode so stdout
    # stays a pure machine-readable stream.
    display_mode = _resolve_agent_display(args) if _use_agentic_mode(args) else "off"
    crack_out = sys.stderr if display_mode == "jsonl" else sys.stdout
    print(
        f"Alphabet: {ct.alphabet.size} symbols, {len(ct.tokens)} tokens, "
        f"{len(ct.words)} words",
        file=crack_out,
    )
    if getattr(args, "analyze", False) and not _use_agentic_mode(args):
        print("Note: --analyze is only run for --agentic artifacts; ignoring.", file=sys.stderr)

    from pathlib import Path

    cipher_id = args.cipher_id or "cli"
    artifact_dir = args.artifact_dir or "artifacts"

    if not _use_agentic_mode(args):
        from automated.runner import run_automated, save_crack_artifact

        print("Running automated solver (no LLM API calls)...")
        homophonic_budget = getattr(args, "homophonic_budget", "full")
        homophonic_refinement = getattr(args, "homophonic_refinement", "none")

        # F8: narrate each pipeline step boundary directly (no renderer on the
        # automated path).
        def on_step(name: str, status: str | None, elapsed: float) -> None:
            tail = f" {status}" if status else " done"
            print(f"  · step: {name} …{tail} ({elapsed:.1f}s)", flush=True)

        # No benchmark source in `crack`; "auto" degrades to default (None).
        crack_variant = getattr(args, "model_variant", None)
        if crack_variant == "auto":
            crack_variant = None
        run_kwargs = {
            "cipher_text": ct,
            "language": args.language,
            "cipher_id": cipher_id,
            "homophonic_budget": homophonic_budget,
            "homophonic_refinement": homophonic_refinement,
            "homophonic_solver": "legacy" if getattr(args, "legacy_homophonic", False) else "zenith_native",
            "model_variant": crack_variant,
            "on_step": on_step,
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

    agent_loop = getattr(args, "agent_loop", "v2")
    _preflight_model_check(args)
    provider, model = _resolve_provider_and_model(args)
    api = _make_agent_provider(args)
    # display_mode + crack_out were resolved once at the top of cmd_crack (F-2).
    renderer = make_agent_renderer(display_mode, verbose=args.verbose)
    if renderer is not None:
        renderer.start_test(
            cipher_id,
            "Interactive crack",
            model=model,
            max_iterations=args.max_iterations,
            language=args.language,
            agent_loop=agent_loop,
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
        # No benchmark source in `crack`; "auto" degrades to default (None).
        crack_variant = getattr(args, "model_variant", None)
        if crack_variant == "auto":
            crack_variant = None
        run_kwargs = {
            "cipher_text": ct,
            "language": args.language,
            "cipher_id": cipher_id,
            "homophonic_budget": homophonic_budget,
            "homophonic_refinement": homophonic_refinement,
            "homophonic_solver": "legacy" if getattr(args, "legacy_homophonic", False) else "zenith_native",
            "model_variant": crack_variant,
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

    if agent_loop == "v3":
        from investigation.loop_v3 import run_v3

        artifact = run_v3(
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
    else:
        artifact = run_v2(
            cipher_text=ct,
            claude_api=api,
            language=args.language,
            max_iterations=args.max_iterations,
            cipher_id=cipher_id,
            prior_context=_read_external_context(args),
            automated_preflight=automated_preflight,
            verbose=args.verbose and display_mode == "off",
            system_prompt_style=_resolve_system_prompt_style(args),
            on_event=on_event,
        )

    # Decision 6: while a renderer is active (the pretty Rich Live in
    # particular), no plain prints — ALL terminal output goes through the
    # renderer. Compute + save first, run renderer.finish, THEN print the human
    # summary. In jsonl mode route that summary to stderr so stdout stays
    # machine-clean (F1).
    iterations = max(tc.iteration for tc in artifact.tool_calls) if artifact.tool_calls else 0
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
    save_error = None
    try:
        artifact.save(path)
    except Exception as e:  # noqa: BLE001
        save_error = str(e)

    if renderer is not None:
        from types import SimpleNamespace

        renderer.finish(SimpleNamespace(
            test_id=cipher_id,
            status=artifact.status,
            char_accuracy=0.0,
            word_accuracy=0.0,
            # `crack` has no ground truth; narrate suppresses the 0.0% line (F-4).
            has_ground_truth=False,
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

    # crack_out (stderr in jsonl) was resolved once at the top of cmd_crack.
    print(f"Status: {artifact.status}", file=crack_out)
    if artifact.solution:
        print(f"Declared branch: {artifact.solution.branch}", file=crack_out)
        print(f"Self-confidence: {artifact.solution.self_confidence:.2f}", file=crack_out)
        print(f"Rationale: {artifact.solution.rationale}", file=crack_out)
    print(f"Iterations: {iterations}", file=crack_out)
    print(f"Tool calls: {len(artifact.tool_calls)}", file=crack_out)
    if save_error is None:
        print(f"\nArtifact saved: {path}", file=crack_out)
    else:
        print(f"\nWarning: failed to save artifact: {save_error}", file=crack_out)
    print(f"\nFinal decryption ({final_branch}):\n{final_dec}", file=crack_out)
    _maybe_write_artifact_analysis(path, args)


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
    renderer = make_agent_renderer(display_mode, verbose=args.verbose)
    if renderer is not None:
        renderer.start_test(
            cipher_id,
            f"Resume artifact {parent_path.name}",
            model=model,
            max_iterations=extra_iterations,
            language=language,
            agent_loop=getattr(args, "agent_loop", "v2"),
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
            f"\nStatus: {artifact.status}, "
            f"Comparison to known ground-truth plaintext: "
            f"Char: {result.char_accuracy:.1%}, Word: {result.word_accuracy:.1%}, "
            f"Conf: {conf}, "
            f"Iter: {iterations}, Time: {result.elapsed_seconds:.1f}s"
        )
        print(f"Artifact: {path}")
        if artifact.error_message:
            print(f"Error: {artifact.error_message}")
        print(f"\nFinal summary:\n{final_summary}")
        print(f"\nFinal decryption ({final_branch}):\n{final_decryption}")
    _maybe_write_artifact_analysis(path, args)


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
    if getattr(args, "poly_tableau_keyword", None):
        spec.polyalphabetic_tableau_keyword = args.poly_tableau_keyword
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
    if getattr(args, "analyze", False) and not _use_agentic_mode(args):
        print("Note: --analyze is only run for --agentic artifacts; ignoring.", file=sys.stderr)
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
            renderer_verbose=args.verbose,
            language=args.language,
            artifact_dir=args.artifact_dir,
            automated_preflight=not args.no_automated_preflight,
            display_mode=display_mode,
            homophonic_budget=args.homophonic_budget,
            homophonic_refinement=args.homophonic_refinement,
            homophonic_solver="legacy" if args.legacy_homophonic else "zenith_native",
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
    print("Comparison to known ground-truth plaintext:")
    print(f"  Char:     {score.char_accuracy:.1%}   Word: {word_str}")
    print(f"Confidence: {conf}   Iterations: {result.iterations_used}   Time: {result.elapsed_seconds:.1f}s")
    print(f"Artifact:   {result.artifact_path}")
    if result.error_message:
        print(f"Error:      {result.error_message}")
    if _use_agentic_mode(args):
        _maybe_write_artifact_analysis(result.artifact_path, args)


def _diagnose_inputs_from_text(text: str):
    """Return (tokens, alphabet_size, alphabet_class, letter_rendering,
    numeric_values, word_group_count) for a raw ciphertext string."""
    from analysis.numeric_code import is_numeric_ciphertext, parse_numeric_ciphertext

    if is_numeric_ciphertext(text):
        values = parse_numeric_ciphertext(text)
        order = {v: i for i, v in enumerate(sorted(set(values)))}
        tokens = [order[v] for v in values]
        return tokens, max(len(set(tokens)), 26), "numeric", None, values, 0

    non_ws = [ch for ch in text if not ch.isspace()]
    is_pure_letters = bool(non_ws) and all("A" <= ch.upper() <= "Z" for ch in non_ws)
    if is_pure_letters:
        letters = "".join(ch.upper() for ch in text if ch.isalpha())
        tokens = [ord(c) - 65 for c in letters]
        word_group_count = len([w for w in text.split() if w]) if " " in text.strip() else 0
        return tokens, 26, "letters", letters, None, word_group_count

    # Symbol / S-token path — reuse the crack parsers.
    from benchmark.loader import parse_canonical_transcription
    from models.alphabet import Alphabet
    from models.cipher_text import CipherText

    if "S" in text.upper() and any(ch.isdigit() for ch in text):
        ct = parse_canonical_transcription(text)
    else:
        alphabet = Alphabet.from_text(text, ignore_chars={" ", "\t", "\n", "\r"})
        clean = " ".join(text.split())
        ct = CipherText(raw=clean, alphabet=alphabet, source="cli", separator=" ")
    return list(ct.tokens), ct.alphabet.size, "symbols", None, None, len(ct.words)


def cmd_diagnose(args: argparse.Namespace) -> None:
    import json as _json

    from analysis.numeric_code import (
        assert_not_solution_bearing,
        is_numeric_ciphertext,
        parse_numeric_ciphertext,
        profile_for_related,
    )
    from investigation.diagnosis import diagnose, format_diagnosis

    if getattr(args, "unsolved_id", None):
        from benchmark.unsolved import load_unsolved_record

        if not args.benchmark_root:
            print("Error: --benchmark-root is required with --unsolved-id.", file=sys.stderr)
            sys.exit(1)
        record = load_unsolved_record(args.benchmark_root, args.unsolved_id)
        if not record.canonical_text:
            print(f"Error: no canonical text available for {args.unsolved_id}.", file=sys.stderr)
            sys.exit(1)
        text = record.canonical_text
        language = args.language or record.metadata.get("plaintext_language") or "en"
    else:
        if args.input and args.input != "-":
            with open(args.input, encoding="utf-8") as f:
                text = f.read()
        else:
            text = sys.stdin.read()
        language = args.language or "en"

    if not text.strip():
        print("Error: no input text provided.", file=sys.stderr)
        sys.exit(1)

    # Firewall: never diagnose a solution-bearing key text (finding 9).
    try:
        assert_not_solution_bearing(text, source=getattr(args, "input", None) or "input")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    related_profile = None
    if getattr(args, "related_profile", None):
        with open(args.related_profile, encoding="utf-8") as f:
            rp_text = f.read()
        if is_numeric_ciphertext(rp_text):
            related_profile = profile_for_related(parse_numeric_ciphertext(rp_text))

    tokens, alphabet_size, alphabet_class, letter_rendering, numeric_values, wgc = (
        _diagnose_inputs_from_text(text)
    )
    report = diagnose(
        tokens,
        alphabet_size=alphabet_size,
        alphabet_class=alphabet_class,
        language=language,
        word_group_count=wgc,
        numeric_values=numeric_values,
        letter_rendering=letter_rendering,
        related_profile=related_profile,
        max_period=getattr(args, "max_period", 26),
    )
    if getattr(args, "json", False):
        print(_json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(format_diagnosis(report))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="decipher",
        description="Decipher — Classical Cipher Cryptanalysis Tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    doctor = subparsers.add_parser(
        "doctor",
        help="Check Rust kernels, LLM providers, and API key configuration",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable status as JSON.",
    )
    doctor.add_argument(
        "--refresh-pricing",
        action="store_true",
        dest="refresh_pricing",
        help=(
            "Fetch the latest model pricing from OpenRouter and update the local "
            "disk cache (~/.config/decipher/openrouter_pricing.json). "
            "Pricing for Anthropic/OpenAI/Gemini is hardcoded and updated with "
            "code releases."
        ),
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
        choices=["anthropic", "claude", "openai", "gemini", "google", "ollama", "openrouter", "or"],
        help="LLM provider for agentic runs. Default is inferred from --model, else anthropic.",
    )
    bench.add_argument(
        "--model",
        "-m",
        help="LLM model name. Defaults by provider.",
    )
    bench.add_argument(
        "--system-prompt-style",
        choices=["full", "compact", "auto"],
        default="auto",
        dest="system_prompt_style",
        help="System prompt size. 'compact' replaces the verbose toolkit section with a cheatsheet (~1,400 chars vs ~9,200). 'auto' selects compact for ollama, full otherwise.",
    )
    bench.add_argument("--language", "-l", choices=["en", "la", "de", "fr", "it", "unknown"])
    bench.add_argument(
        "--model-variant",
        default=None,
        help=(
            "Language-model variant slug for the automated solver (e.g. "
            "'historical_1600_1899'). Default None keeps today's model. Use "
            "'auto' to map by benchmark source (copiale -> German DTA)."
        ),
    )
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
        choices=["auto", "narrate", "pretty", "raw", "jsonl"],
        default="narrate",
        help=(
            "Agentic terminal display mode (default: narrate — a scrolling, "
            "pipe-safe transcript). auto also resolves to narrate; pretty is the "
            "Rich live dashboard; raw/jsonl are the legacy/machine streams. -v adds "
            "detail inside the active renderer."
        ),
    )
    _add_artifact_analysis_args(bench)
    bench.add_argument(
        "--agentic",
        action="store_true",
        help="Use the experimental LLM agent instead of the default automated solver.",
    )
    bench.add_argument(
        "--agent-loop",
        choices=["v2", "v3"],
        default="v2",
        dest="agent_loop",
        help=(
            "Agentic loop to use (default v2). v3 is the investigation lead loop: "
            "state rebuilt each turn, provider-native sessions, no declare gates."
        ),
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
        choices=HOMOPHONIC_REFINEMENT_CHOICES,
        default="none",
        help="Optional second-stage local refinement for automated homophonic runs.",
    )
    bench.add_argument(
        "--legacy-homophonic",
        action="store_true",
        help="Use the older pre-zenith_native homophonic solver path for comparison.",
    )
    bench.add_argument(
        "--multipage-group",
        metavar="PATH",
        help=(
            "Run the automated multipage shared-key route on a page-group "
            "definition JSON (e.g. frontier/groups/copiale_evidence.json). "
            "Combines the group's pages into one ciphertext, runs the homophonic "
            "route once (respecting --homophonic-budget/-refinement/-solver), "
            "projects the shared key back per page, and writes per-page plus "
            "group artifacts. Automated-only; incompatible with --agentic."
        ),
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
        choices=["anthropic", "claude", "openai", "gemini", "google", "ollama", "openrouter", "or"],
        help="LLM provider for agentic runs. Default is inferred from --model, else anthropic.",
    )
    crack.add_argument("--model", "-m", help="LLM model name. Defaults by provider.")
    crack.add_argument(
        "--system-prompt-style",
        choices=["full", "compact", "auto"],
        default="auto",
        dest="system_prompt_style",
        help="System prompt size. 'compact' replaces the verbose toolkit section with a cheatsheet (~1,400 chars vs ~9,200). 'auto' selects compact for ollama, full otherwise.",
    )
    crack.add_argument("--language", "-l", choices=["en", "la", "de", "fr", "it", "unknown"],
                       default="en")
    crack.add_argument(
        "--model-variant",
        default=None,
        help=(
            "Language-model variant slug for the automated solver/preflight "
            "(e.g. 'literary_19c'). Default None keeps today's model."
        ),
    )
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
        choices=["auto", "narrate", "pretty", "raw", "jsonl"],
        default="narrate",
        help="Agentic terminal display mode (default: narrate; auto also resolves to narrate).",
    )
    _add_artifact_analysis_args(crack)
    crack.add_argument(
        "--agentic",
        action="store_true",
        help="Use the experimental LLM agent instead of the default automated solver.",
    )
    crack.add_argument(
        "--agent-loop",
        choices=["v2", "v3"],
        default="v2",
        dest="agent_loop",
        help=(
            "Agentic loop to use (default v2). v3 is the investigation lead loop: "
            "state rebuilt each turn, provider-native sessions, no declare gates."
        ),
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
        choices=HOMOPHONIC_REFINEMENT_CHOICES,
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
        choices=["anthropic", "claude", "openai", "gemini", "google", "ollama", "openrouter", "or"],
        help="LLM provider for the continuation. Default is inferred from --model.",
    )
    resume.add_argument("--model", "-m", help="LLM model name (default: prior artifact model)")
    resume.add_argument("--language", "-l", choices=["en", "la", "de", "fr", "it", "unknown"])
    resume.add_argument("--artifact-dir", help="Artifact output directory (default: ./artifacts)")
    resume.add_argument("--cipher-id", help="Override cipher id for the continuation artifact")
    resume.add_argument("--verbose", "-v", action="store_true")
    resume.add_argument(
        "--display",
        choices=["auto", "narrate", "pretty", "raw", "jsonl"],
        default="narrate",
        help="Agentic terminal display mode (default: narrate; auto also resolves to narrate).",
    )
    _add_artifact_analysis_args(resume)

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
            "quagmire3",
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
            "For quagmire3 this is the cycleword. "
            "Use letters for Vigenere/Beaufort/Quagmire and digits for Gronsfeld."
        ),
    )
    tg.add_argument(
        "--poly-tableau-keyword",
        help=(
            "Keyword used to build the keyed alphabet for Quagmire III ciphers. "
            "Ignored for standard Vigenere-family variants."
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
        choices=["anthropic", "claude", "openai", "gemini", "google", "ollama", "openrouter", "or"],
        help="LLM provider for generation/agentic runs. Default is inferred from --model, else anthropic.",
    )
    tg.add_argument("--model", "-m")
    tg.add_argument("--artifact-dir", default="artifacts")
    tg.add_argument("--cache-dir", default="testgen_cache")
    tg.add_argument("--verbose", "-v", action="store_true")
    tg.add_argument(
        "--display",
        choices=["auto", "narrate", "pretty", "raw", "jsonl"],
        default="narrate",
        help="Agentic terminal display mode (default: narrate; auto also resolves to narrate).",
    )
    _add_artifact_analysis_args(tg)
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
        choices=HOMOPHONIC_REFINEMENT_CHOICES,
        default="none",
        help="Optional second-stage local refinement for automated homophonic runs.",
    )
    tg.add_argument(
        "--legacy-homophonic",
        action="store_true",
        help="Use the older pre-zenith_native homophonic solver path for comparison.",
    )

    # diagnose (INV-0): LLM-free local cipher-family diagnosis
    diag = subparsers.add_parser(
        "diagnose",
        help="Diagnose the likely cipher family of an unknown ciphertext (no LLM/API).",
    )
    diag.add_argument(
        "input",
        nargs="?",
        help="Input file, or '-' for stdin. Omit when using --unsolved-id.",
    )
    diag.add_argument("--unsolved-id", dest="unsolved_id",
                      help="Diagnose a record from the benchmark unsolved area by id.")
    diag.add_argument("--benchmark-root", dest="benchmark_root",
                      help="Benchmark root directory (required with --unsolved-id).")
    diag.add_argument("--language", "-l", default=None,
                      help="Assumed plaintext language (default: en / record language).")
    diag.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    diag.add_argument("--related-profile", dest="related_profile",
                      help="Numeric companion ciphertext file, profiled via the P8 battery.")
    diag.add_argument("--max-period", dest="max_period", type=int, default=26,
                      help="Maximum key period for periodic analysis (default 26).")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)
    if args.command not in ("doctor", "diagnose"):
        _require_rust_fast_kernel()

    dispatch = {
        "doctor": cmd_doctor,
        "benchmark": cmd_benchmark,
        "crack": cmd_crack,
        "resume-artifact": cmd_resume_artifact,
        "testgen": cmd_testgen,
        "diagnose": cmd_diagnose,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
